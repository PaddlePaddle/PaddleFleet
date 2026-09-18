# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HyperBody: unified single-PipelineLayer model (encoder + decoder in one).

Unlike the earlier *composite* design (a ``FleetLayer`` holding two independently
``provide()``-d sub-models glued by an ``image_embeds`` scatter), this is ONE
``PipelineLayer`` whose flat layer-desc list is:

    encoder frontend (NEW)  -> encoder trunk (REUSED HyperEncoder specs)
    -> bridge (NEW)         -> decoder embedding (REUSED GPTEmbedding, scatter)
    -> decoder trunk (REUSED) -> layer_norm (REUSED) -> lm_head (REUSED) -> loss

The layer-to-layer contract is a **dict** (``TransformerLayer.forward`` updates
``hidden_states`` in place and preserves every other key). Two facts drive the
whole design:

* the encoder trunk uses the *same* ``TransformerLayer`` LayerSpec as the
  decoder, so it consumes ``hidden_states`` / ``attention_mask`` /
  ``rotary_pos_emb`` and would ALSO consume ``input_ids`` /
  ``attn_mask_startend_row_indices`` if present -> decoder-bound fields are
  carried through the encoder region under ``_hb_dec_*`` prefixes and renamed
  back by the bridge;
* ``GPTEmbedding.forward`` rebuilds a fresh preproc dict from ``input_ids``
  (ignoring any upstream ``hidden_states``) -> the bridge DROPS the encoder
  hidden state; the decoder embedding regenerates ``[B,S,H_dec]`` from
  ``input_ids`` and scatters the encoder latents (``image_embeds``) at
  ``input_ids == image_token_id`` (CONTEXT_TOKEN) positions.

Scope (this phase): PP=1 single-stage random-init smoke (build/forward/backward).
Cross-stage seg_method / stage-pinning at the encoder->decoder boundary is
deferred (the bridge emits variable-length latents and the hidden shape changes
across the boundary -> cross-stage P2P is not viable yet).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import (
    LayerDesc,
    LayerSpec,
    PipelineLayer,
    build_spec_layer,
)

from paddlefleet.models.common.language_loss.language_loss import LanguageLoss
from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_spec
from paddlefleet.models.hyperbody import (
    get_hyperbody_decoder_layer_specs,
    get_hyperbody_encoder_layer_specs,
)
from paddlefleet.transformer.layer import FleetLayer

from ...nn.pp_model import GeneralModelForCausalLMPipe
from ..model_utils import PretrainedModel
from .configuration import HyperBodyConfig
from .providers import (
    HyperBodyDecoderModelProvider,
    HyperEncoderConfig,
    HyperEncoderProvider,
    _reject_unsupported_decoder_branches,
)

__all__ = [
    "HyperBodyModelDist",
    "HyperBodyModel",
    "HyperBodyForConditionalGeneration",
    "HyperBodyForCausalLMPipe",
    "HyperBodyModelPipe",
    "HyperBodyPretrainedModel",
]


# ======================================================================= #
# NEW wrapper layer 1: encoder frontend (pipeline-hostile logic isolated)  #
# ======================================================================= #
class HyperBodyEncoderFrontEnd(FleetLayer):
    """Encoder front end + query concat + prefix-LM mask + rope.

    Holds the pipeline-hostile pieces of ``HyperEncoderModel`` that cannot be a
    plain single-hidden-state layer: the per-sample multimodal context builder
    (with dummy gradient keepalive), the query-table concat, the prefix-LM dense
    mask construction and the rope table. Emits a dict whose ``hidden_states`` is
    the padded ``[B, C+Q(+pad), H_enc]`` sequence, plus the encoder
    ``attention_mask`` / ``rotary_pos_emb`` that the (shared) TransformerLayer
    trunk consumes, plus the geometry scalars and the decoder-bound fields under
    ``_hb_dec_*`` prefixes so the encoder trunk does not consume them.
    """

    def __init__(self, config, **kwargs) -> None:
        super().__init__(config=config)
        from paddlefleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )
        from paddlefleet.models.hyperbody import (
            AudioEncoderConv,
            ImageEncoderConv,
            MlpProjector,
        )
        from paddlefleet.models.hyperencoder.attn_backend import (
            use_packed_decoder,
        )

        self.config = config
        # dp (dense mask) vs triton+packed (single-call, PackedSeqParams layout).
        # use_packed_decoder validates packed=>triton (raises on packed+dp).
        self.packed = use_packed_decoder(config)
        short_q, long_q = config.hyperencoder_query_lengths

        self.embed_tokens = paddle.nn.Embedding(
            config.vocab_size, config.hidden_size
        )
        self.image_encoder = ImageEncoderConv(
            img_size=728,
            patch_size=14,
            in_chans=3,
            embed_dim=768,
            out_chans=256,
        )
        self.audio_encoder = AudioEncoderConv(num_mel_bins=128, embed_dim=768)
        self.projector = MlpProjector(
            {
                "projector_type": "linear",
                "input_dim": 768,
                "n_embed": config.hidden_size,
            }
        )
        self.query_short = paddle.nn.Embedding(short_q, config.hidden_size)
        self.query_long = paddle.nn.Embedding(long_q, config.hidden_size)
        self.rotary_pos_emb = RotaryEmbedding(
            head_dim=config.head_dim,
            rotary_percent=config.rotary_percent,
            rotary_base=config.rope_theta,
        )

        for m in (
            self.embed_tokens,
            self.image_encoder,
            self.audio_encoder,
            self.projector,
            self.query_short,
            self.query_long,
        ):
            m.to(dtype=config.params_dtype)
            for p in m.parameters():
                p.average_gradients_across_tp_domain = True

    # -- per-segment multimodal context builder (dummy keepalive) --
    def _embed_one_context(self, ids_i, img_i, aud_i):
        """One context segment ``[C_i]`` -> ``[C_i', H]`` with image/audio splice.

        Mirrors the reference ``build_context_embeds`` per-sample body: replace
        the IM_PATCH / AUDIO_PATCH placeholder run with the encoder-tower feature
        rows by "truncate and append"; when a modality is absent, push a dummy
        through the tower and add ``feat.sum()*0`` (a forward identity that keeps
        the tower's gradients alive).
        """
        from .configuration import AUDIO_PATCH_TOKEN, IM_PATCH_TOKEN

        emb_i = self.embed_tokens(ids_i)  # [C_i, H]

        pos = paddle.nonzero(ids_i == IM_PATCH_TOKEN)
        if pos.numel() > 0 and isinstance(img_i, paddle.Tensor):
            feat = self.projector(self.image_encoder(img_i.unsqueeze(0)))
            feat = (
                feat.transpose([0, 3, 1, 2])
                .flatten(2)
                .transpose([0, 2, 1])
                .squeeze(0)
            )
            emb_i = paddle.concat([emb_i[: int(pos[0])], feat], axis=0)
        else:
            dummy = paddle.zeros([1, 3, 128, 128], dtype=emb_i.dtype)
            feat = self.projector(self.image_encoder(dummy))
            emb_i = emb_i + feat.sum() * 0.0

        pos = paddle.nonzero(ids_i == AUDIO_PATCH_TOKEN)
        if pos.numel() > 0 and isinstance(aud_i, paddle.Tensor):
            feat = self.projector(self.audio_encoder(aud_i.unsqueeze(0)))
            feat = (
                feat.transpose([0, 3, 1, 2])
                .flatten(2)
                .transpose([0, 2, 1])
                .squeeze(0)
            )
            emb_i = paddle.concat([emb_i[: int(pos[0])], feat], axis=0)
        else:
            dummy = paddle.zeros([1, 128, 10], dtype=emb_i.dtype)
            feat = self.projector(self.audio_encoder(dummy))
            emb_i = emb_i + feat.sum() * 0.0

        return emb_i

    def _build_context_embeds(
        self, context_ids, image=None, audio=None, cu_seqlens_context=None
    ):
        """context_ids -> context embeds.

        Two shapes are supported:

        * **rectangular** (synthetic / equal-length): ``context_ids`` is
          ``[B, C]`` and ``cu_seqlens_context`` is None; returns a stacked
          ``[B, C, H]`` (every row the same length).
        * **packed / ragged** (real torch dump): ``context_ids`` is
          ``[1, C_total]`` and ``cu_seqlens_context`` is a boundary list
          ``[0, e0, e1, ...]`` splitting it into per-segment contexts of possibly
          different lengths; returns a **list** of ``[C_i', H]``. ``image`` /
          ``audio`` are per-segment lists (with ``None`` where absent).
        """
        if cu_seqlens_context is None:
            bs = context_ids.shape[0]
            imgs = image if image else [None] * bs
            auds = audio if audio else [None] * bs
            out = [
                self._embed_one_context(context_ids[i], imgs[i], auds[i])
                for i in range(bs)
            ]
            return paddle.stack(out, axis=0)

        # Packed / ragged: split [1, C_total] by cu_seqlens_context.
        bounds = [int(x) for x in cu_seqlens_context]
        n_seg = len(bounds) - 1
        imgs = image if image else [None] * n_seg
        auds = audio if audio else [None] * n_seg
        row = context_ids[0]
        return [
            self._embed_one_context(
                row[bounds[s] : bounds[s + 1]], imgs[s], auds[s]
            )
            for s in range(n_seg)
        ]

    def _add_unused_query_dependency(self, x, use_long_query):
        """Keep the *unused* query table(s) in the autograd graph (zero grad).

        ``use_long_query`` may be a single bool (homogeneous pack) or a
        per-segment ``list[bool]`` (mixed pack). A table counts as "used" only if
        at least one segment selects it; every table NOT used by any segment gets
        a ``weight.sum()*0`` dependency so its grad is zero (not None).
        """
        if isinstance(use_long_query, bool):
            used_long, used_short = use_long_query, not use_long_query
        else:
            flags = [bool(u) for u in use_long_query]
            used_long = any(flags)
            used_short = any(not f for f in flags)
        dep = None
        if not used_long:
            dep = self.query_long.weight.sum() * 0.0
        if not used_short:
            s = self.query_short.weight.sum() * 0.0
            dep = s if dep is None else dep + s
        if dep is None:
            # Both tables used (mixed pack): tie both with zero weight so neither
            # is orphaned regardless of trunk usage.
            dep = (
                self.query_long.weight.sum() + self.query_short.weight.sum()
            ) * 0.0
        return x + dep.astype(x.dtype)

    def _packed_rope(self, max_seqlen: int, seg_lens):
        """Assemble RoPE freqs for the packed sequence: ``[1, ΣL_i, 1, D]``.

        Each segment restarts its position from 0 (prefix-LM in-segment relative
        position), so this concatenates ``freqs[:, :L_i]`` per segment. At a
        single segment it reduces to ``freqs[:, :L_0]`` (identical to the dense
        path). Mirrors ``HyperEncoderModel._packed_rope`` (non-SP branch).
        """
        freqs = self.rotary_pos_emb(max_seq_len=max_seqlen)
        if len(seg_lens) == 1:
            return freqs[:, : seg_lens[0]]
        return paddle.concat([freqs[:, :L] for L in seg_lens], axis=1)

    def forward(self, dict_args):
        from paddlefleet.transformer.prefix_lm_mask import (
            build_dense_mask,
            prefix_lm_pad_len,
        )

        context_ids = dict_args["context_ids"]
        image = dict_args.get("image", None)
        audio = dict_args.get("audio", None)
        use_long_query = dict_args.get("use_long_query", False)
        cu_seqlens_context = dict_args.get("cu_seqlens_context", None)

        context_embeds = self._build_context_embeds(
            context_ids, image, audio, cu_seqlens_context
        )

        if self.packed:
            out = self._forward_packed(context_embeds, use_long_query)
        else:
            out = self._forward_dense(
                context_embeds,
                use_long_query,
                build_dense_mask,
                prefix_lm_pad_len,
            )

        # Decoder-bound fields ride through the encoder trunk under _hb_dec_*
        # prefixes (absorbed by TransformerLayer **kwargs, preserved by
        # rst={**dict_args,**rst}); the bridge renames them back.
        out["_hb_dec_input_ids"] = dict_args["input_ids"]
        out["_hb_dec_labels"] = dict_args.get("labels", None)
        out["_hb_dec_attn_mask_startend_row_indices"] = dict_args.get(
            "attn_mask_startend_row_indices", None
        )
        out["_hb_dec_position_ids"] = dict_args.get("position_ids", None)
        return out

    def _forward_dense(
        self,
        context_embeds,
        use_long_query,
        build_dense_mask,
        prefix_lm_pad_len,
    ):
        # Non-SP [B,S,H] dense-mask path (dp backend).
        bs, n_context, _ = context_embeds.shape
        qw = (self.query_long if use_long_query else self.query_short).weight
        queries = qw.unsqueeze(0).expand([bs, -1, -1])
        n_queries = queries.shape[1]

        x = paddle.concat([context_embeds, queries], axis=1)
        seq_len = x.shape[1]
        pad = prefix_lm_pad_len(
            seq_len,
            self.config.tensor_model_parallel_size,
            self.config.sequence_parallel,
            align=self.config.hyperencoder_seq_align,
        )
        if pad > 0:
            x = paddle.concat(
                [x, paddle.zeros([bs, pad, x.shape[-1]], dtype=x.dtype)], axis=1
            )
        mask = build_dense_mask([n_context], [n_queries], pad)
        rope = self.rotary_pos_emb(max_seq_len=x.shape[1])
        x = self._add_unused_query_dependency(x, use_long_query)

        return {
            "hidden_states": x,
            "attention_mask": mask,
            "rotary_pos_emb": rope,
            "packed_seq_params": None,
            "_hb_packed": None,
            "n_context": n_context,
            "n_queries": n_queries,
            "seq_len": seq_len,
            "use_long_query": use_long_query,
        }

    def _forward_packed(self, context_embeds, use_long_query):
        """Single packed call (triton core). Mirrors
        ``HyperEncoderModel.forward_decoder_packed`` (non-SP branch).

        The ``(context, query)`` segments are concatenated end-to-end into one
        ``[1, T, H]`` sequence; the segment layout is attached to
        ``PackedSeqParams.prefix_lm_layout`` for ``PrefixLMTritonCore`` and also
        carried under ``_hb_packed`` for the bridge's per-segment slice.
        ``attention_mask`` is None: the triton core rejects a dense mask and
        derives its semantics entirely from the layout.

        ``context_embeds`` may be a rectangular ``[B, C, H]`` (synthetic /
        equal-length) or a ragged **list** of ``[C_i, H]`` (real packed dump).
        ``use_long_query`` may be a single **bool** (homogeneous pack) or a
        per-segment **list[bool]** (mixed short/long query pack).
        """
        from paddlefleet.packed_seq_params import PackedSeqParams
        from paddlefleet.transformer.prefix_lm_mask import prefix_lm_pad_len
        from paddlefleet.transformer.prefix_lm_triton_core import (
            PREFIX_LM_LAYOUT_ATTR,
        )

        # ---- Normalize context to a per-segment list [C_i, H] ----
        if isinstance(context_embeds, (list, tuple)):
            seg_ctx = [
                c.squeeze(0) if c.ndim == 3 else c for c in context_embeds
            ]
            n_contexts = [int(c.shape[0]) for c in seg_ctx]
            hidden = int(seg_ctx[0].shape[1])
        else:
            bs0, n_context, hidden = context_embeds.shape
            seg_ctx = [context_embeds[i] for i in range(bs0)]
            n_contexts = [n_context] * bs0
        bs = len(seg_ctx)

        # ---- Per-segment query table selection ----
        if isinstance(use_long_query, bool):
            uql = [use_long_query] * bs
        else:
            uql = [bool(u) for u in use_long_query]
            if len(uql) != bs:
                raise ValueError(
                    f"use_long_query list length {len(uql)} does not match "
                    f"segment count {bs}"
                )
        qws = [(self.query_long if u else self.query_short).weight for u in uql]
        nq_list = [int(q.shape[0]) for q in qws]

        # ---- Concatenate (context, query) segments end-to-end ----
        segments, starts, running = [], [], 0
        for ci, nc, q, nq in zip(seg_ctx, n_contexts, qws, nq_list):
            segments.append(paddle.concat([ci, q], axis=0))
            starts.append(running)
            running += nc + nq
        x = paddle.concat(segments, axis=0).unsqueeze(0)  # [1, ΣL, H]
        seq_len = x.shape[1]

        pad = prefix_lm_pad_len(
            seq_len,
            self.config.tensor_model_parallel_size,
            self.config.sequence_parallel,
            align=self.config.hyperencoder_seq_align,
        )
        if pad > 0:
            x = paddle.concat(
                [x, paddle.zeros([1, pad, hidden], dtype=x.dtype)], axis=1
            )

        # The pad is folded into the last segment (not a segment of its own) so
        # the rope segmentation and the layout segmentation stay aligned.
        seg_lens = [nc + nq for nc, nq in zip(n_contexts, nq_list)]
        seg_lens[-1] += pad
        max_seqlen = max(seg_lens)

        layout = {
            "segment_starts": tuple(starts),
            "n_contexts": tuple(n_contexts),
            "n_queries": tuple(nq_list),
            "pad_len": pad,
        }
        psp = PackedSeqParams()
        setattr(psp, PREFIX_LM_LAYOUT_ATTR, layout)

        rope = self._packed_rope(max_seqlen, seg_lens)
        x = self._add_unused_query_dependency(x, use_long_query)

        return {
            "hidden_states": x,
            "attention_mask": None,
            "rotary_pos_emb": rope,
            "packed_seq_params": psp,
            "_hb_packed": layout,
            # Ragged pack: per-segment lists (bridge reads _hb_packed layout, not
            # these scalars; kept for symmetry with the dense return dict).
            "n_context": list(n_contexts),
            "n_queries": list(nq_list),
            "seq_len": seq_len,
            "use_long_query": use_long_query,
        }


# ======================================================================= #
# NEW wrapper layer 2: bridge (encoder trunk output -> decoder latents)    #
# ======================================================================= #
class HyperBodyEncoderBridge(FleetLayer):
    """Encoder trunk output -> per-sample out-projected LLM latents.

    Reapplies the encoder block's final ``HyperEncoderRMSNorm`` (lost when the
    ``HyperEncoderBlock`` is decomposed into per-layer specs), drops the pad,
    slices the trailing ``Q`` rows, runs the per-batch ``out_projector``
    (RowParallelLinear -> ``language_hidden_size``, adds bias by hand), and emits
    the *clean decoder* dict. It DROPS the encoder ``hidden_states`` so the
    downstream ``GPTEmbedding`` regenerates ``[B,S,H_dec]`` from ``input_ids``
    and scatters the encoder latents at CONTEXT_TOKEN positions.
    """

    def __init__(self, config, **kwargs) -> None:
        super().__init__(config=config)
        from paddlefleet.models.hyperencoder.norm import HyperEncoderRMSNorm
        from paddlefleet.tensor_parallel.layers import RowParallelLinear

        self.config = config
        self.final_norm = HyperEncoderRMSNorm(config)

        projector_config = copy.copy(config)
        projector_config.sequence_parallel = False
        self.out_projector = RowParallelLinear(
            config.hidden_size,
            config.language_hidden_size,
            config=projector_config,
            init_method=config.init_method,
            bias=True,
            input_is_parallel=False,
            skip_bias_add=False,
        )
        if getattr(self.out_projector, "bias", None) is not None:
            self.out_projector.bias.average_gradients_across_tp_domain = True

    def _project(self, latents):
        """Per-segment ``out_projector`` (RowParallelLinear, adds bias by hand)."""
        projected, bias = self.out_projector(latents)
        if bias is not None:
            projected = projected + bias
        return projected

    def forward(self, dict_args):
        layout = dict_args.get("_hb_packed", None)
        if layout is not None:
            image_embeds = self._forward_packed(dict_args, layout)
        else:
            image_embeds = self._forward_dense(dict_args)

        return {
            "input_ids": dict_args["_hb_dec_input_ids"],
            "position_ids": dict_args.get("_hb_dec_position_ids", None),
            "attention_mask": None,
            "attn_mask_startend_row_indices": dict_args[
                "_hb_dec_attn_mask_startend_row_indices"
            ],
            "decoder_input": None,
            "image_embeds": image_embeds,
            "video_embeds": None,
            "labels": dict_args["_hb_dec_labels"],
        }

    def _forward_dense(self, dict_args):
        h = dict_args["hidden_states"]
        n_context = dict_args["n_context"]
        seq_len = dict_args["seq_len"]

        h = self.final_norm(h)
        h = h[:, :seq_len]  # drop the prefix-LM pad
        latents = h[:, n_context:, :]  # trailing Q rows: [B, Q, H_enc]

        outs = [self._project(latents[i]) for i in range(latents.shape[0])]
        return paddle.concat(outs, axis=0)  # [B*Q, H_dec]

    def _forward_packed(self, dict_args, layout):
        # hidden_states is [1, ΣL_i(+pad), H_enc]; slice each segment's trailing
        # query rows via the packed layout, out-project, concat -> [ΣQ_i, H_dec].
        h = self.final_norm(dict_args["hidden_states"])[0]  # [T, H_enc]
        starts = layout["segment_starts"]
        n_contexts = layout["n_contexts"]
        n_queries = layout["n_queries"]

        outs = []
        for s, nc, nq in zip(starts, n_contexts, n_queries):
            latents = h[s + nc : s + nc + nq, :]  # [Q_i, H_enc]
            outs.append(self._project(latents))
        return paddle.concat(outs, axis=0)  # [ΣQ_i, H_dec]


# ======================================================================= #
# Unified PipelineLayer (mirrors GPTModel's get_layer_desc_list mechanics)  #
# ======================================================================= #
@dataclass
class HyperBodySublayersSpec:
    """Flat spec list for the unified model (encoder + bridge + decoder)."""

    encoder_frontend: LayerSpec = None
    encoder_layers: list[LayerSpec] = field(default_factory=list)
    bridge: LayerSpec = None
    decoder_embedding: LayerSpec = None
    decoder_layers: list[LayerSpec] = field(default_factory=list)
    layer_norm: LayerSpec = None
    lm_head: LayerSpec = None


class HyperBodyUnifiedModel(PipelineLayer):
    """One PipelineLayer: encoder frontend/trunk/bridge + decoder emb/trunk/head.

    Mirrors ``GPTModel``'s pipeline mechanics (``get_layer_desc_list`` +
    ``add_sequential_layer`` + ``get_sequential_layers``) but with the HyperBody
    layer order. Encoder-region layers use the ``encoder`` name prefix, decoder
    region uses ``model`` (matching ``GPTModel``), so weight keys do not clash.
    """

    def __init__(self, sublayers_spec, **kwargs) -> None:
        self.config = kwargs["config"]
        pp = self.config.pipeline_model_parallel_size
        # single(logical) <-> pipeline(numeric) key maps, built lazily by
        # ``_set_pipeline_name_mapping`` (mirrors GPTModel / PipelinePretrainedModel).
        self._single_to_pp_mapping = None
        self._pp_to_single_mapping = None

        self._sequential_layers = self.get_layer_desc_list(sublayers_spec)
        self.layers = self.get_sequential_layers()

        del kwargs["config"]
        if "tie_word_embeddings" in kwargs:
            del kwargs["tie_word_embeddings"]

        topology = (
            None if pp == 1 else fleet.get_hybrid_communicate_group().topology()
        )
        super().__init__(
            layers=self.layers,
            topology=topology,
            num_virtual_pipeline_stages=(
                self.config.virtual_pipeline_model_parallel_size
            ),
            **kwargs,
        )

    @staticmethod
    def add_sequential_layer(layers, layer_desc, name_prefix=""):
        layers.append({"layer": layer_desc, "name_prefix": name_prefix})

    def get_sequential_layers(self):
        return [x["layer"] for x in self._sequential_layers]

    def get_sequential_name_prefixes(self):
        return {
            str(i): x["name_prefix"]
            for i, x in enumerate(self._sequential_layers)
        }

    def get_layer_desc_list(self, spec: HyperBodySublayersSpec):
        layers = []
        # --- encoder region (name_prefix="encoder") ---
        self.add_sequential_layer(
            layers, LayerDesc(spec.encoder_frontend), "encoder"
        )
        for i, enc_layer in enumerate(spec.encoder_layers):
            self.add_sequential_layer(
                layers, LayerDesc(enc_layer), f"encoder.layers.{i}"
            )
        self.add_sequential_layer(
            layers, LayerDesc(spec.bridge), "encoder.bridge"
        )
        # --- decoder region (name_prefix="model") ---
        self.add_sequential_layer(
            layers, LayerDesc(spec.decoder_embedding), "model"
        )
        for i, dec_layer in enumerate(spec.decoder_layers):
            self.add_sequential_layer(
                layers, LayerDesc(dec_layer), f"model.layers.{i}"
            )
        self.add_sequential_layer(layers, LayerDesc(spec.layer_norm), "model")
        self.add_sequential_layer(
            layers, LayerDesc(spec.lm_head), "model.lm_head"
        )
        return layers

    # ------------------------------------------------------------------ #
    # Logical <-> pipeline key mapping (for weight save / flex_ckpt load) #
    # ------------------------------------------------------------------ #
    # ``PipelineLayer`` names its params ``{layer_idx}.rest`` (a bare integer
    # per sequential slot). ``_gen_aoa_config`` and ``save_pretrained`` speak
    # LOGICAL names (``encoder.image_encoder.*`` / ``model.layers.i.*`` / ...),
    # so ``state_dict`` / ``sharded_state_dict`` must remap numeric<->logical
    # exactly like ``GPTModel`` / ``PipelinePretrainedModel``. The unified model
    # has no SharedLayerDesc and (this phase) PP=VPP=1, so the mapping reduces to
    # prefixing each ``{idx}.rest`` with ``get_sequential_name_prefixes()[idx]``.
    def _set_pipeline_name_mapping(self):
        single_to_pp = {}
        pp_to_single = {}
        prefixes = self.get_sequential_name_prefixes()
        for k in list(super().state_dict().keys()):
            seg = k.split(".")
            idx = seg[0]
            if idx.isdigit():
                pref = prefixes[idx]
                single = ([] if pref == "" else [pref]) + seg[1:]
                single = ".".join(single)
            else:
                # non-numeric (e.g. directly-registered) keys pass through.
                single = k
            single_to_pp[single] = k
            pp_to_single[k] = single
        self._single_to_pp_mapping = single_to_pp
        self._pp_to_single_mapping = pp_to_single
        return single_to_pp

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        if self._pp_to_single_mapping is None:
            self._set_pipeline_name_mapping()
        for k in list(sd.keys()):
            v = sd.pop(k)
            sd[self._pp_to_single_mapping[k]] = v
        return sd

    def set_state_dict(self, state_dict, *args, **kwargs):
        if self._single_to_pp_mapping is None:
            self._set_pipeline_name_mapping()
        for k in list(state_dict.keys()):
            v = state_dict.pop(k)
            if k in self._single_to_pp_mapping:
                state_dict[self._single_to_pp_mapping[k]] = v
        return super().set_state_dict(state_dict, *args, **kwargs)

    def sharded_state_dict(self, *args, **kwargs):
        ssd = super().sharded_state_dict(*args, **kwargs)
        if self._pp_to_single_mapping is None:
            self._set_pipeline_name_mapping()
        for k in list(ssd.keys()):
            v = ssd.pop(k)
            v.key = self._pp_to_single_mapping[k]
            ssd[self._pp_to_single_mapping[k]] = v

        # Expert-parallel: local expert ids -> global ids (no-op when EP=1).
        import re

        def _bump_expert(s, inc):
            return re.sub(
                r"(?<=experts\.)\d+", lambda m: str(int(m.group(0)) + inc), s
            )

        renamed = {}
        for k, v in ssd.items():
            off = getattr(v, "global_expert_id_offset", None)
            if off is not None:
                nk = _bump_expert(k, off)
                v.key = nk
                delattr(v, "global_expert_id_offset")
                renamed[nk] = v
            else:
                renamed[k] = v
        return renamed


# ======================================================================= #
# View builders + top-level builder                                        #
# ======================================================================= #
def _build_decoder_view(config: HyperBodyConfig):
    """Materialize the decoder GPTConfig view (bare-named fields) + multimodal."""
    view = HyperBodyDecoderModelProvider.from_config(config)
    view.multimodal_embedding = True
    view.image_token_id = config.image_token_id
    view.video_token_id = config.video_token_id
    # Packed decoder RoPE gate: when the encoder runs packed, the decoder is fed
    # a real packed [1, ΣS] layout, so RoPE must restart per segment. The shared
    # GPTEmbedding rope call reads this flag (default off => no behavior change
    # for any other model). ``decoder_packed_rope`` decouples this knob from
    # ``hyperencoder_packed_decoder``; when unset (None) it falls back to the
    # encoder flag, so the current config behavior is unchanged.
    _pr = getattr(config, "decoder_packed_rope", None)
    view.packed_decoder_rope = bool(
        config.hyperencoder_packed_decoder if _pr is None else _pr
    )
    return view


def _build_encoder_view(config: HyperBodyConfig, decoder_hidden: int):
    """Strip the ``encoder_`` prefix into a transient HyperEncoderConfig, then
    materialize a HyperEncoderProvider view; wire the out-projector output width
    to the decoder hidden size and force eager attention.
    """
    # Recompute: the encoder trunk supports a single degree of freedom --
    # ``recompute_granularity in (None, "full")`` (HyperEncoderProvider.__post_init__
    # pins method="uniform"/num_layers=1). Mirror the decoder's *intent*: if the
    # run recomputes at all (flat granularity is truthy, e.g. "full"/"selective"),
    # the encoder also does full recompute -- its only mode -- otherwise None.
    # We deliberately forward only the mapped granularity, NOT method/num_layers:
    # forwarding a non-"uniform"/non-1 method or num_layers would trip the
    # provider's strict guard, and the provider derives them itself when on.
    # Recompute is numerically transparent, so mapping "selective" -> "full" here
    # only changes memory/speed, never results.
    _flat_recompute = getattr(config, "recompute_granularity", None)
    enc_recompute_granularity = "full" if _flat_recompute else None

    enc_cfg = HyperEncoderConfig(
        vocab_size=config.encoder_vocab_size,
        hidden_size=config.encoder_hidden_size,
        intermediate_size=config.encoder_intermediate_size,
        num_hidden_layers=config.encoder_num_hidden_layers,
        num_attention_heads=config.encoder_num_attention_heads,
        num_key_value_heads=config.encoder_num_key_value_heads,
        rms_norm_eps=config.encoder_rms_norm_eps,
        rope_theta=config.encoder_rope_theta,
        attention_dropout=config.encoder_attention_dropout,
        hidden_dropout_prob=config.encoder_hidden_dropout_prob,
        attention_bias=config.encoder_attention_bias,
        moe_intermediate_size=config.encoder_moe_intermediate_size,
        n_routed_experts=config.encoder_n_routed_experts,
        num_experts_per_tok=config.encoder_num_experts_per_tok,
        n_shared_experts=config.encoder_n_shared_experts,
        first_k_dense_replace=config.encoder_first_k_dense_replace,
        routed_scaling_factor=config.encoder_routed_scaling_factor,
        n_group=config.encoder_n_group,
        topk_group=config.encoder_topk_group,
        norm_topk_prob=config.encoder_norm_topk_prob,
        scoring_func=config.encoder_scoring_func,
        topk_method=config.encoder_topk_method,
        hyperencoder_query_lengths=config.hyperencoder_query_lengths,
        hyperencoder_seq_align=config.hyperencoder_seq_align,
        hyperencoder_attn_backend=config.hyperencoder_attn_backend,
        hyperencoder_packed_decoder=config.hyperencoder_packed_decoder,
        tensor_model_parallel_size=config.tensor_model_parallel_size,
        # ---- shared (non-``encoder_``-prefixed) runtime fields --------------- #
        # Geometry uses the ``encoder_`` prefix, but MoE/parallelism/fusion
        # runtime knobs are shared flat fields. They must mirror the decoder or
        # the encoder view silently falls back to HyperEncoderProvider dataclass
        # defaults, which diverge from the flat HyperBodyConfig defaults:
        #   * moe_expert_fusion:      flat True  vs provider False -> the AoA
        #     ``_gen_aoa_config`` encoder branch keys off the flat value and would
        #     emit grouped_gemm mappings the (non-fused) encoder never builds.
        #   * router_aux_loss_coef:   flat 0.001 vs base 1e-2 -> 10x aux-loss weight.
        #   * expert/context_model_parallel_size: encoder would pin EP/CP=1 even
        #     when the decoder runs EP/CP>1.
        # ``sequence_parallel`` is intentionally NOT forwarded: the provider
        # __post_init__ rejects an explicit value and derives it from tp_size.
        moe_expert_fusion=config.moe_expert_fusion,
        router_aux_loss_coef=config.router_aux_loss_coef,
        expert_model_parallel_size=config.expert_model_parallel_size,
        context_parallel_size=config.context_parallel_size,
        moe_shared_expert_overlap=config.moe_shared_expert_overlap,
        moe_router_load_balancing_type=config.moe_router_load_balancing_type,
        moe_token_dispatcher_type=config.moe_token_dispatcher_type,
        apply_rope_fusion=config.apply_rope_fusion,
        # Mirror the decoder's routed-scaling learnability: when the flat config
        # turns it on, the decoder's moe_router allocates a learnable
        # ``routed_scaling_factor_param``. Not forwarding it here would leave the
        # encoder on a fixed scale while the decoder learns one -> silent divergence.
        routed_scaling_factor_learnable=config.routed_scaling_factor_learnable,
        # Recompute intent, clamped to the encoder's only supported mode (see above).
        recompute_granularity=enc_recompute_granularity,
    )
    view = HyperEncoderProvider.from_config(enc_cfg)
    view.language_hidden_size = decoder_hidden
    view._attn_implementation = "eager"
    return view


def build_hyperbody_unified_model(
    config: HyperBodyConfig, *, num_stages=1, loss_fn=None
):
    """Assemble the unified single-PipelineLayer model.

    Mirrors ``build_hyperbody_decoder_model`` (narrowed ``gpt_builder``): build
    the decoder gpt spec, extract its embedding/trunk/norm/lm_head sub-specs,
    build the encoder frontend/trunk/bridge specs, then wrap everything in one
    ``HyperBodyUnifiedModel`` root LayerSpec and materialize via
    ``build_spec_layer``.
    """
    # #1: this phase only supports PP=1. Reject both an explicit num_stages!=1
    # and a config that requests pipeline_model_parallel_size>1 (the builder
    # hardcodes num_stages=1 downstream, so an unguarded PP>1 config would be
    # silently ignored rather than honored).
    pp_size = getattr(config, "pipeline_model_parallel_size", 1)
    if num_stages != 1 or (pp_size is not None and pp_size > 1):
        raise NotImplementedError(
            "HyperBody unified model only supports PP=1 in this phase."
        )

    decoder_view = _build_decoder_view(config)

    # #4: mirror the standalone decoder's rejection guards (MTP /
    # separate_mtp_headloss / EmptyLayer head|tail / ringmoe / meta-device) so
    # the unified builder fails loudly on unsupported branches too. Current
    # production configs trigger none of these => behavior unchanged.
    _reject_unsupported_decoder_branches(decoder_view)

    gpt_spec = get_gpt_spec(
        config=decoder_view,
        head_empty_layers_spec=[],
        transformer_layers_spec=get_hyperbody_decoder_layer_specs(decoder_view),
        tail_empty_layers_spec=[],
        mtp_layers_spec=None,
        vocab_size=decoder_view.vocab_size,
        tie_word_embeddings=decoder_view.tie_word_embeddings,
        max_sequence_length=decoder_view.max_sequence_length,
        position_embedding_type=decoder_view.position_embedding_type,
        rotary_percent=decoder_view.rotary_percent,
        rotary_base=decoder_view.rope_theta,
        swa_rotary_base=decoder_view.swa_rope_theta,
        rope_scaling=decoder_view.rope_scaling,
        parallel_output=decoder_view.parallel_output,
    )
    dec_sub = gpt_spec.sublayers_spec

    encoder_view = _build_encoder_view(config, decoder_view.hidden_size)
    enc_layer_specs = get_hyperbody_encoder_layer_specs(encoder_view)

    frontend_spec = LayerSpec(
        HyperBodyEncoderFrontEnd, extra_kwargs={"config": encoder_view}
    )
    bridge_spec = LayerSpec(
        HyperBodyEncoderBridge, extra_kwargs={"config": encoder_view}
    )

    sublayers_spec = HyperBodySublayersSpec(
        encoder_frontend=frontend_spec,
        encoder_layers=enc_layer_specs,
        bridge=bridge_spec,
        decoder_embedding=dec_sub.embedding,
        decoder_layers=dec_sub.transformer_layers,
        layer_norm=dec_sub.layer_norm,
        lm_head=dec_sub.lm_head,
    )

    root = LayerSpec(
        HyperBodyUnifiedModel,
        extra_kwargs={"config": decoder_view, "tie_word_embeddings": False},
        sublayers_spec=sublayers_spec,
    )
    return build_spec_layer(
        root,
        loss_fn=LanguageLoss(decoder_view) if loss_fn is None else loss_fn,
        num_stages=num_stages,
        seg_method="layer:TransformerLayer|EmptyLayer",
    )


# ======================================================================= #
# Registration / entry classes                                            #
# ======================================================================= #
class HyperBodyPretrainedModel(PretrainedModel):
    config_class = HyperBodyConfig
    base_model_prefix = "hyperbody"
    input_modalities = ["image", "audio", "text"]
    # The router gate is fp32 on the fleet side (PaddleFleet forces fp32 while
    # use_accuracy_compatible=False); both encoder- and decoder-region archives
    # store it as bf16, so every gate statement carries a src/dst dtype pair.
    _keep_in_fp32_modules = ["mlp.gate.weight"]

    @classmethod
    def _gen_aoa_config(cls, config):
        """Forward weight mapping (HF checkpoint name -> fleet structured name).

        The unified HyperBody archive interleaves TWO backbones plus the
        multimodal front end. Left-hand sides are the HF ``model.safetensors``
        keys; right-hand sides are this model's ``sharded_state_dict`` LOGICAL
        keys (numeric pipeline slots are remapped to logical prefixes by
        ``HyperBodyUnifiedModel.sharded_state_dict``):

        * decoder LLM (``model.*`` / ``lm_head.weight`` on the HF side) ->
          ``model.embedding.*`` / ``model.layers.i.*`` / ``model.norm.weight`` /
          ``model.lm_head.weight`` -- IDENTICAL to the standalone decoder aoa.
        * encoder front end + inner backbone (``model.encoder.*`` on the HF side)
          -> ``encoder.*`` / ``encoder.layers.L.*``; the inner backbone lives at
          ``model.encoder.decoder.model.model.*``.
        * bridge: the encoder backbone final norm
          (``model.encoder.decoder.model.model.norm``) -> ``encoder.bridge.final_norm``
          and the encoder->LLM output projection (top-level ``model.projector.*``)
          -> ``encoder.bridge.out_projector.*``.

        ``^T`` transposes HF ``nn.Linear`` ``(out,in)`` to Paddle ``(in,out)``;
        embedding / query / gate / norm keep the same axis order (no ``^T``).
        The dense-layer set is derived from ``moe_layer_freq[i]==0`` (decoder)
        and ``first_k_dense_replace`` leading layers (encoder). Routed-expert
        up/gate fusion follows each region's OWN standalone reference: the
        decoder concatenates with ``axis=1``, the encoder with ``fused_ffn``.
        """
        st = []

        # =============== DECODER region (HF model.* / lm_head) =============== #
        dec_experts = config.n_routed_experts
        dec_freq = config.moe_layer_freq
        dec_nh = config.num_attention_heads
        dec_kvh = config.num_key_value_heads

        st += [
            "model.embed_tokens.weight -> model.embedding.embed_tokens.weight",
            "model.norm.weight -> model.norm.weight",
        ]
        if config.tie_word_embeddings:
            st.append("model.embed_tokens.weight -> model.lm_head.weight")
        else:
            st.append("lm_head.weight -> model.lm_head.weight")

        for L in range(config.num_hidden_layers):
            hf = f"model.layers.{L}"
            pd = f"model.layers.{L}"
            st += [
                f"{hf}.input_layernorm.weight -> {pd}.input_layernorm.weight",
                f"{hf}.post_attention_layernorm.weight -> {pd}.post_attention_layernorm.weight",
                f"{hf}.self_attn.o_proj.weight^T -> {pd}.self_attn.o_proj.weight",
                f"{hf}.self_attn.q_proj.weight^T, {hf}.self_attn.k_proj.weight^T, "
                f"{hf}.self_attn.v_proj.weight^T -> {pd}.self_attn.qkv_proj.weight, "
                f"fused_qkv, num_heads={dec_nh}, num_key_value_groups={dec_kvh}",
            ]
            if not dec_freq[L]:
                st += [
                    f"{hf}.mlp.gate_proj.weight^T, {hf}.mlp.up_proj.weight^T "
                    f"-> {pd}.mlp.up_gate_proj.weight, fused_ffn",
                    f"{hf}.mlp.down_proj.weight^T -> {pd}.mlp.down_proj.weight",
                ]
                continue
            st += [
                f"{hf}.mlp.gate.weight -> {pd}.mlp.gate.weight, "
                f"src_dtype='bfloat16',dst_dtype='float32'",
                f"{hf}.mlp.shared_experts.gate_proj.weight^T, {hf}.mlp.shared_experts.up_proj.weight^T "
                f"-> {pd}.mlp.shared_experts.up_gate_proj.weight, fused_ffn",
                f"{hf}.mlp.shared_experts.down_proj.weight^T -> {pd}.mlp.shared_experts.down_proj.weight",
                f"{hf}.mlp.experts.$EXPERT_ID.gate_proj.weight^T, {hf}.mlp.experts.$EXPERT_ID.up_proj.weight^T "
                f"-> {pd}.mlp.experts.$EXPERT_ID.up_gate_proj.weight, axis=1",
                f"{hf}.mlp.experts.$EXPERT_ID.down_proj.weight^T -> {pd}.mlp.experts.$EXPERT_ID.down_proj.weight",
            ]
            if config.moe_expert_fusion:
                w1 = ",".join(
                    f"{pd}.mlp.experts.{e}.up_gate_proj.weight"
                    for e in range(dec_experts)
                )
                w2 = ",".join(
                    f"{pd}.mlp.experts.{e}.down_proj.weight"
                    for e in range(dec_experts)
                )
                st += [
                    f"{w1} -> {pd}.mlp.grouped_gemm_experts.weight1, axis=0",
                    f"{w2} -> {pd}.mlp.grouped_gemm_experts.weight2, axis=0",
                ]

        # =============== ENCODER region (HF model.encoder.*) =============== #
        # HF prefixes: front end at ``model.encoder.*``; inner backbone at
        # ``model.encoder.decoder.model.model.*``; output projection at the
        # top-level ``model.projector.*`` (bridge).
        enc = "model.encoder"
        enc_dec = "model.encoder.decoder.model.model"
        enc_layers = config.encoder_num_hidden_layers
        enc_experts = config.encoder_n_routed_experts
        enc_nh = config.encoder_num_attention_heads
        enc_kvh = config.encoder_num_key_value_heads
        enc_dense = set(range(int(config.encoder_first_k_dense_replace or 0)))

        # front-end towers / embedding / query tables (conv+pos_embed no ^T)
        for name in (
            "image_encoder.pos_embed",
            "image_encoder.patch_embed.proj.weight",
            "image_encoder.patch_embed.proj.bias",
            "audio_encoder.pos_embed",
            "audio_encoder.conv1.weight",
            "audio_encoder.conv1.bias",
            "audio_encoder.conv2.weight",
            "audio_encoder.conv2.bias",
        ):
            st.append(f"{enc}.{name} -> encoder.{name}")
        st.append(
            f"{enc}.projector.layers.weight^T -> encoder.projector.layers.weight"
        )
        st.append(
            f"{enc}.projector.layers.bias -> encoder.projector.layers.bias"
        )
        st.append(
            f"{enc_dec}.embed_tokens.weight -> encoder.embed_tokens.weight"
        )
        st.append(
            f"{enc}.decoder.query_short.weight -> encoder.query_short.weight"
        )
        st.append(
            f"{enc}.decoder.query_long.weight -> encoder.query_long.weight"
        )
        # bridge: encoder backbone final norm + encoder->LLM output projection
        st.append(f"{enc_dec}.norm.weight -> encoder.bridge.final_norm.weight")
        st.append(
            "model.projector.weight^T -> encoder.bridge.out_projector.weight"
        )
        st.append("model.projector.bias -> encoder.bridge.out_projector.bias")

        for L in range(enc_layers):
            hf = f"{enc_dec}.layers.{L}"
            pd = f"encoder.layers.{L}"
            st.append(
                f"{hf}.input_layernorm.weight -> {pd}.input_layernorm.weight"
            )
            st.append(
                f"{hf}.post_attention_layernorm.weight -> {pd}.post_attention_layernorm.weight"
            )
            st.append(
                f"{hf}.self_attn.q_proj.weight^T, {hf}.self_attn.k_proj.weight^T, "
                f"{hf}.self_attn.v_proj.weight^T -> {pd}.self_attn.qkv_proj.weight, "
                f"fused_qkv, num_heads={enc_nh}, num_key_value_groups={enc_kvh}"
            )
            st.append(
                f"{hf}.self_attn.o_proj.weight^T -> {pd}.self_attn.o_proj.weight"
            )
            if L in enc_dense:
                st.append(
                    f"{hf}.mlp.gate_proj.weight^T, {hf}.mlp.up_proj.weight^T "
                    f"-> {pd}.mlp.up_gate_proj.weight, fused_ffn"
                )
                st.append(
                    f"{hf}.mlp.down_proj.weight^T -> {pd}.mlp.down_proj.weight"
                )
                continue
            st.append(
                f"{hf}.mlp.gate.weight -> {pd}.mlp.gate.weight, "
                f"src_dtype='bfloat16',dst_dtype='float32'"
            )
            st.append(
                f"{hf}.mlp.shared_experts.gate_proj.weight^T, "
                f"{hf}.mlp.shared_experts.up_proj.weight^T "
                f"-> {pd}.mlp.shared_experts.up_gate_proj.weight, fused_ffn"
            )
            st.append(
                f"{hf}.mlp.shared_experts.down_proj.weight^T "
                f"-> {pd}.mlp.shared_experts.down_proj.weight"
            )
            st.append(
                f"{hf}.mlp.experts.$EXPERT_ID.gate_proj.weight^T, "
                f"{hf}.mlp.experts.$EXPERT_ID.up_proj.weight^T "
                f"-> {pd}.mlp.experts.$EXPERT_ID.up_gate_proj.weight, fused_ffn"
            )
            st.append(
                f"{hf}.mlp.experts.$EXPERT_ID.down_proj.weight^T "
                f"-> {pd}.mlp.experts.$EXPERT_ID.down_proj.weight"
            )
            if config.moe_expert_fusion:
                w1 = ",".join(
                    f"{pd}.mlp.experts.{e}.up_gate_proj.weight"
                    for e in range(enc_experts)
                )
                w2 = ",".join(
                    f"{pd}.mlp.experts.{e}.down_proj.weight"
                    for e in range(enc_experts)
                )
                st += [
                    f"{w1} -> {pd}.mlp.grouped_gemm_experts.weight1, axis=0",
                    f"{w2} -> {pd}.mlp.grouped_gemm_experts.weight2, axis=0",
                ]

        return {"aoa_statements": st}


class HyperBodyModelDist(HyperBodyPretrainedModel):
    """Factory returning the unified ``HyperBodyUnifiedModel`` (a PipelineLayer).

    Mirrors ``Qwen3VLModel.__new__``: build the pipe via the unified builder
    (PP=1 in this phase). Returns the pipe directly, not ``cls``.
    """

    config_class = HyperBodyConfig

    def __new__(cls, config: HyperBodyConfig, have_criterion: bool = True):
        # build_spec_layer always appends a loss layer for structure; the PP=1
        # entry applies pipe._loss_fn[0] manually, so have_criterion does not
        # change the assembled graph here.
        pipe = build_hyperbody_unified_model(config, num_stages=1)
        pipe._gen_aoa_config = cls._gen_aoa_config
        # No _gen_inv_aoa_config: flex_checkpoint auto-derives the save (reverse)
        # mapping from _gen_aoa_config (model_utils.py:3834). Leaving it unset is
        # required -- setting a stub would make hasattr() true and trip the save.
        pipe.config_to_save = config
        return pipe


# Alias kept for parity with the composite-era public name.
HyperBodyModel = HyperBodyModelDist


class HyperBodyForConditionalGeneration(HyperBodyPretrainedModel):
    """PP=1 smoke entry.

    Holds ``self.pipe`` (the unified ``HyperBodyUnifiedModel``). The pipe does
    NOT auto-apply its loss (stored in ``pipe._loss_fn[0]``), so this entry
    applies it manually when ``labels`` are given.
    """

    config_class = HyperBodyConfig
    # Fleet model: this wraps a PipelineLayer whose params carry logical fleet
    # names (not HF names). is_fleet=True makes from_pretrained SKIP the identity
    # dtype append (model_utils.py:3421) -- that block would otherwise emit bogus
    # `<pipe.-prefixed key> -> <same>, dtype=...` statements whose LHS has no
    # source in the HF safetensors. _gen_aoa_config already carries every needed
    # dtype spec (e.g. gate.weight bf16->fp32). Matches the *Pipe entries.
    is_fleet = True

    def __init__(self, config: HyperBodyConfig):
        super().__init__(config)
        self.pipe = build_hyperbody_unified_model(config, num_stages=1)

    # ---- weight name I/O: delegate straight to the pipe -------------------- #
    # The wrapper holds the graph under ``self.pipe``. The default nn.Layer
    # recursion would call ``self.pipe.sharded_state_dict(structured_name_prefix=
    # "pipe.")`` -> the pipe emits ``pipe.{idx}.rest`` keys, then its own
    # numeric<->logical remap looks up ``pipe.0.embed_tokens.weight`` in a table
    # built from UN-prefixed names and raises KeyError (R38). Delegating here (no
    # prefix) makes the wrapper expose the pipe's LOGICAL names verbatim, which is
    # exactly what ``_gen_aoa_config`` RHS + flex_checkpoint expect (and identical
    # to what the bare *Pipe entry produces). This unblocks both flex_checkpoint
    # LOAD (from_pretrained) and SAVE through the wrapper.
    def state_dict(self, *args, **kwargs):
        return self.pipe.state_dict(*args, **kwargs)

    def set_state_dict(self, state_dict, *args, **kwargs):
        return self.pipe.set_state_dict(state_dict, *args, **kwargs)

    def sharded_state_dict(self, *args, **kwargs):
        return self.pipe.sharded_state_dict(*args, **kwargs)

    def forward(
        self,
        input_ids: paddle.Tensor | None = None,
        context_ids: paddle.Tensor | None = None,
        image=None,
        audio=None,
        use_long_query: bool = False,
        labels: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        position_ids: paddle.Tensor | None = None,
        cu_seqlens: paddle.Tensor | None = None,
        cu_seqlens_context=None,
        **kwargs,
    ):
        if cu_seqlens is not None:
            # Packed decoder: derive both the block-diagonal causal flashmask
            # boundaries and the per-segment reset RoPE position_ids from the
            # decoder segment boundaries. cu_seqlens is a [num_seg + 1] int32
            # vector of cumulative lengths over the packed decoder token axis
            # (ΣS == input_ids.shape[-1]).
            import numpy as _np

            from paddlefleet.transformer.multi_token_prediction import (
                build_startend_row_indices_from_cu_seqlens,
            )

            total_len = int(input_ids.shape[-1])
            # [1, 1, ΣS, 1] int32: each token records its own segment `end`
            # boundary -> flashmask keeps attention inside the segment while
            # is_causal=True (decoder layer spec) makes it lower-triangular.
            attn_mask_startend_row_indices = (
                build_startend_row_indices_from_cu_seqlens(
                    cu_seqlens, batch_size=1, seq_len=total_len
                )
            )
            cu_np = (
                cu_seqlens.numpy()
                if isinstance(cu_seqlens, paddle.Tensor)
                else _np.asarray(cu_seqlens)
            )
            # Reset position_ids: 0..L_i-1 per segment, concatenated -> [1, ΣS].
            pos_np = _np.concatenate(
                [
                    _np.arange(
                        int(cu_np[j + 1]) - int(cu_np[j]), dtype=_np.int64
                    )
                    for j in range(len(cu_np) - 1)
                ]
            )
            position_ids = paddle.to_tensor(
                pos_np[None, :], place=input_ids.place
            )

        input_dict = {
            "input_ids": input_ids,
            "context_ids": context_ids,
            "image": image,
            "audio": audio,
            "use_long_query": use_long_query,
            "labels": labels,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
            "position_ids": position_ids,
            "cu_seqlens_context": cu_seqlens_context,
        }
        out = self.pipe(input_dict)
        logits = out["logits"] if isinstance(out, dict) else out
        if labels is None:
            return logits
        loss = self.pipe._loss_fn[0](logits, labels)
        if isinstance(loss, (list, tuple)):
            loss = loss[0]
        return loss


class HyperBodyModelPipe(HyperBodyPretrainedModel):
    config_class = HyperBodyConfig
    is_fleet = True

    def __new__(cls, config: HyperBodyConfig, have_criterion: bool = True):
        return HyperBodyModelDist.__new__(cls, config, have_criterion)


class HyperBodyForCausalLMPipe(
    HyperBodyPretrainedModel, GeneralModelForCausalLMPipe
):
    config_class = HyperBodyConfig
    is_fleet = True

    def __new__(cls, config: HyperBodyConfig, have_criterion: bool = True):
        return HyperBodyModelDist.__new__(cls, config, have_criterion)
