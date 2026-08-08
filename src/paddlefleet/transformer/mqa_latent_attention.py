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

"""Latent MQA core attention for hybrid MLA layers, with DSA.

``hybrid_mla_attention`` selects which core attention the
``csa_compress_ratios == -2`` (MLA) layers of a ``dsv4_hybrid`` model run:

* ``"mha"`` -- unchanged dense MLA (MHA); this module is not used.
* ``"mqa_dsa"`` -- :class:`MQALatentAttention` with a forced local window plus
  Lightning-indexer top-k, i.e. DeepSeek Sparse Attention on the KV latent.
  The indexer reuses the model-wide ``index_n_heads`` / ``index_head_dim`` /
  ``index_topk``, and ``dsa_indexer_use_sparse_loss`` selects the indexer-loss
  width exactly as it does for the CSA layers (see ``_forward_dsa``).
* ``"mqa_full_causal"`` -- :class:`MQALatentAttention` with the indexer dropped,
  attending to the full per-document causal set. That is mathematically
  identical to the dense MHA phase, so it isolates the absorption from the
  sparsity for equivalence experiments; it is ``O(s^2)`` in index memory and
  therefore not a production mode.

Note the ``mqa_*`` modes here are *latent* MQA (this module). A
``csa_compress_ratios`` entry of ``-1`` is *CSA full-causal MQA*, a different
layer kind handled by ``csa_attention.py``.

``MLASelfAttention`` performs the activation-level absorption (see its
``mqa_latent`` flag), so this module receives

    query [b, s, h, kv_lora_rank + qk_rope_head_dim]
    key   [b, s, 1, kv_lora_rank + qk_rope_head_dim]

and de-absorbs the value side with ``v_b_proj_weight``
(``[kv_lora_rank, h, v_head_dim]``). Every parameter stays byte-identical to
the MHA layout, so an MHA checkpoint loads into an MQA run unchanged.

``add_full_attention_sink_bias`` (or ``softmax_type``) adds one learnable
per-head sink logit as ``softmax_offset``, built by the same
``build_softmax_offset`` helper ``DotProductAttention`` uses, so the parameter
name matches the dense phase. It is fed to the block-sparse kernel as its
``attn_sink``, which then enables the finite-sink LSE correction and the
analytic sink gradient. The indexer KL target is unaffected: it is renormalised
over the selected set, where the sink mass cancels.

Multi-document equivalence: RoPE/YaRN scores depend only on ``pos_q - pos_k``,
the YaRN ``mscale`` is a constant and the Hadamard ``rotate_activation`` is
orthogonal, so no per-document position reset is needed. Equivalence to running
every document on its own therefore reduces to index correctness, which is what
``_derive_csa_doc_boundaries`` plus the index builders below guarantee.

Context parallel (``contiguous_allgather`` only, as the HCA layers of the same
model require at ``dsv4_hybrid_attention.py:607-611``) follows the same
"Miles pattern" as ``CSASelfAttention._forward_cp``: the query slice stays
sharded, only the low-dimensional KV latent is all-gathered to the global
length, and the output comes back sharded with no reduce-scatter. The index
tables are built over the *global* sequence and then row-sliced to this rank,
which is exactly right because ``_derive_csa_doc_boundaries`` and the builders
are ``seqlen``-agnostic and their column values are global token ids -- the same
space the all-gathered KV is indexed in. ``attn_mask_startend_row_indices``
already arrives at the global length under CP
(``dsv4_hybrid_attention.py:634``), while ``query`` / ``key`` / ``x`` / ``qr``
arrive local but RoPE'd with global positions
(``multi_latent_attention.py:1485-1545``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer

from paddlefleet.context_parallel_utils import ContextParallelGatherOp
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.transformer.cp_utils import all_gather_cp
from paddlefleet.transformer.csa_attention import (
    TileLangCSAIndexerLossAutoScaler,
    _build_mqa_causal_topk_idxs_from_doc_bounds,
    _build_window_topk_idxs_from_doc_bounds,
    _derive_csa_doc_boundaries,
    _validate_csa_docmask_shape,
)
from paddlefleet.transformer.dot_product_attention import build_softmax_offset
from paddlefleet.transformer.dsa_attention import DSAIndexerLossLoggingHelper
from paddlefleet.transformer.layer import FleetLayer

if TYPE_CHECKING:
    from paddlefleet.transformer.enums import AttnMaskType

# Working set of the KL-target gather, as a ``rows x slots`` budget: 256 rows x
# 512 slots x 576 dims is ~150MB of gathered bf16 keys, transient and freed every
# iteration. Measured at s=8192/h=64/topk=512/dk=576 on one B30Z: 15.9ms at 128
# rows, 13.3ms at 256, 12.4ms at 512, 12.4ms at 1024 -- past 256 the curve is
# flat, so larger chunks only buy peak memory. Budgeting the product keeps that
# working set when ``dsa_indexer_use_sparse_loss=False`` widens the table.
_TARGET_ROW_SLOTS = 256 * 512
# cuDNN indexer limits on the top-k table width: ``indexer_top_k/api.py:92``
# rejects ``top_k > 2048`` outright, and ``indexer_backward_sm100.__init__``
# asserts ``topk % block_I == 0`` with ``block_I = 128``.
_LOSS_TOPK_CAP = 2048
_TOPK_BLOCK = 128
_NEG_INF = -1e30
_EPS = 1e-10


@dataclass
class MQALatentAttentionSublayersSpec:
    """Sublayers spec for :class:`MQALatentAttention`.

    Args:
        indexer: ``DSAIndexer`` spec. ``hybrid_mla_attention="mqa_dsa"`` provides
            one; ``"mqa_full_causal"`` leaves it ``None`` and the layer attends
            to the full per-document causal set (dense MHA equivalent). Also
            ``None`` in the absorption equivalence unit tests.
    """

    indexer: LayerSpec | type = None


class MQALatentAttention(FleetLayer):
    """Sparse attention on the absorbed MLA KV latent (``core_attention``).

    Consumes the pre-absorbed ``query`` / ``key`` produced by
    ``MLASelfAttention`` and returns ``[b, s, h * v_head_dim]``, so the MLA
    output tail (gate, ``o_proj``) is unchanged.
    """

    def __init__(
        self,
        config,
        sublayers_spec: MQALatentAttentionSublayersSpec,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        softmax_scale: float | None = None,
        k_channels: int | None = None,
        v_channels: int | None = None,
        is_mtp_layer: bool = False,
        is_swa: bool = False,
        num_attention_heads: int | None = None,
        num_key_value_heads: int | None = None,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
    ):
        super().__init__(config=config)

        DSAIndexerLossLoggingHelper.register_total_num_layers(config)
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection

        # CP state, same derivation as csa_attention.py:2079-2090.
        cp_pg = pg_collection.cp if pg_collection is not None else None
        if cp_pg is not None and getattr(cp_pg, "nranks", 1) > 1:
            self.cp_group = cp_pg
            self.cp_size = cp_pg.nranks
            self.cp_rank = cp_pg.rank
            self.cp_enabled = True
            # Deliberately the *same* constraint the HCA layers of this model
            # assert (dsv4_hybrid_attention.py:607-611), not a weaker one: the
            # contiguous layout is what makes "build the index table over the
            # global sequence, then row-slice this rank's queries" correct, and
            # it is what makes the all-gathered KV land in natural global order.
            if (
                getattr(config, "cp_balance_mode", None)
                != "contiguous_allgather"
            ):
                raise NotImplementedError(
                    "latent MQA under context parallel requires "
                    "cp_balance_mode='contiguous_allgather' (the same mode the "
                    "hybrid model's HCA layers require), got "
                    f"{getattr(config, 'cp_balance_mode', None)!r}."
                )
        else:
            self.cp_group = None
            self.cp_size = 1
            self.cp_rank = 0
            self.cp_enabled = False

        # ``k_channels`` is the MHA q_head_dim (qk_nope + qk_rope), NOT the 576
        # latent width: absorption is exactly score-preserving, so the MHA
        # softmax scale must be kept.
        if softmax_scale is None:
            k_ch = k_channels if k_channels is not None else config.head_dim
            self.softmax_scale = float(k_ch**-0.5)
        else:
            self.softmax_scale = float(softmax_scale)

        self.window_size = int(config.csa_window_size)
        self.indexer = (
            build_spec_layer(
                sublayers_spec.indexer,
                config=config,
                layer_number=layer_number,
                pg_collection=pg_collection,
            )
            if sublayers_spec.indexer is not None
            else None
        )
        self.indexer_loss_coeff = float(
            getattr(config, "dsa_indexer_loss_coeff", 0.0) or 0.0
        )
        # Same switch, same meaning as the CSA layers of the hybrid model: it
        # selects the *width of the indexer loss* candidate table. On these
        # uncompressed layers it additionally selects the *attention* candidate
        # set, because the two are one decision here -- see below.
        self.indexer_use_sparse_loss = bool(
            getattr(config, "dsa_indexer_use_sparse_loss", False)
        )
        # ``dsa_indexer_use_sparse_loss=False`` is the phase-2 (warmup) mode:
        # the indexer is still being learned, so attention must not consume its
        # ranking yet -- it attends to the full per-document causal set, exactly
        # like ``hybrid_mla_attention="mqa_full_causal"``, while the indexer is
        # trained on the widest candidate table the kernel allows. Consuming a
        # freshly initialised indexer's top-k here would feed the (frozen)
        # backbone an essentially random sparse pattern for the whole phase.
        # ``True`` is the phase-3/4 mode: attention consumes window + top-k and
        # the KL is restricted to that same set.
        # ``transformer_config.__post_init__`` pins this to ``train_indexer_only``
        # so the two cannot disagree. Deliberately *not* cached as a second
        # attribute: ``_forward_dsa`` reads ``self.indexer_use_sparse_loss``
        # directly at both use sites, so a test (or anything else) flipping it on
        # a live module cannot desynchronise the loss width from the attention
        # set.
        # Learnable per-head attention-sink logit, from the model-wide
        # ``add_full_attention_sink_bias`` / ``softmax_type``. Built by the same
        # helper ``DotProductAttention`` uses, so the state_dict name
        # (``core_attention.softmax_offset``) and the switch are shared with the
        # dense MHA phase. ``None`` keeps the kernel on its sinkless ``-1e30``
        # path, bit-for-bit unchanged.
        self.softmax_offset = build_softmax_offset(
            self,
            config,
            num_attention_heads
            if num_attention_heads is not None
            else config.num_attention_heads,
            is_swa,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        attn_mask_startend_row_indices: Tensor | None = None,
        attn_mask_type: AttnMaskType | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params=None,
        use_rr_flash_attention: bool = False,
        past_key_values=None,
        layer_idx=None,
        use_cache: bool = False,
        x: Tensor | None = None,
        qr: Tensor | None = None,
        kv_compressed: Tensor | None = None,
        k_pos_emb: Tensor | None = None,
        q_absorbed: Tensor | None = None,
        v_b_proj_weight: Tensor | None = None,
        input_ids: Tensor | None = None,
    ) -> Tensor:
        """Absorbed-MQA forward.

        Args:
            query: ``[b, s, h, kv_lora_rank + qk_rope_head_dim]`` absorbed query.
            key:   ``[b, s, 1, kv_lora_rank + qk_rope_head_dim]`` shared latent.
            value: unused (``None``); the V side lives inside ``key``.
            attn_mask_startend_row_indices: ``[b, 1, s, 1]`` exclusive per-token
                document end rows, at the **global** sequence length under CP.
                ``None`` means a single document.
            x / qr: hidden states / q latent, inputs of the DSA indexer.
            v_b_proj_weight: ``[kv_lora_rank, h, v_head_dim]`` de-absorption
                weight (the V slice of ``kv_b_proj``).
            input_ids: ``[b, s]`` token ids, only used to build the indexer-loss
                row mask (``!= pad_token_id``). ``None`` falls back to the plain
                row mean, as CSA does at ``csa_attention.py:1306``.

        Returns:
            ``[b, s, h * v_head_dim]`` (this rank's query slice under CP)
        """
        if packed_seq_params is not None:
            raise NotImplementedError(
                "latent MQA does not support packed_seq_params; "
                "document masking is driven by "
                "attn_mask_startend_row_indices."
            )
        if v_b_proj_weight is None:
            raise ValueError(
                "MQALatentAttention requires v_b_proj_weight; it is only valid "
                "as the core_attention of an absorbed MLA layer."
            )

        b, s = int(query.shape[0]), int(query.shape[1])
        if b != 1:
            raise NotImplementedError(
                "latent MQA requires micro batch size 1 (documents "
                f"are packed along the sequence), got b={b}."
            )
        # Under CP this rank owns global query positions
        # [position_offset, position_offset + s). Everything index-related is
        # derived at s_global and row-sliced; the KV is all-gathered so that the
        # global column ids in those tables address it directly.
        s_global = s * self.cp_size
        position_offset = self.cp_rank * s
        kv = key.squeeze(2).contiguous()  # [b, s, kv_lora + qk_rope]
        kv = all_gather_cp(
            kv, dim=1, group=self.cp_group
        )  # -> [b, s_global, .]
        kv_lora_rank = int(v_b_proj_weight.shape[0])

        with paddle.no_grad():
            row_end = attn_mask_startend_row_indices
            if row_end is None:
                row_end = paddle.full(
                    [b, 1, s_global, 1], s_global, dtype="int32"
                )
            _validate_csa_docmask_shape(row_end, b, s_global)
            doc_start, doc_len, is_valid, doc_lens, _ = (
                _derive_csa_doc_boundaries(row_end, s_global)
            )

        if self.indexer is None:
            # ``hybrid_mla_attention="mqa_full_causal"`` (and the
            # absorption-equivalence unit tests): per-document full causal
            # attention, mathematically identical to dense MHA.
            token_indices = self._build_full_causal_indices(
                b, s_global, doc_start, is_valid, position_offset, s
            )
            core_out = self._sparse_attn(
                query, kv, token_indices, self.softmax_scale, kv_lora_rank
            )
            return self._deabsorb(core_out, v_b_proj_weight)

        return self._forward_dsa(
            query,
            kv,
            x,
            qr,
            v_b_proj_weight,
            doc_start,
            doc_len,
            is_valid,
            doc_lens,
            kv_lora_rank,
            input_ids,
            position_offset,
        )

    # ------------------------------------------------------------------
    # index construction / kernel plumbing
    # ------------------------------------------------------------------
    @staticmethod
    def _build_full_causal_indices(
        b, s_global, doc_start, is_valid, position_offset=0, s_local=None
    ) -> Tensor:
        """Per-document full-causal ``[b, s_local, s_global]`` int32 (``-1`` pad).

        Built over the global sequence and row-sliced to this CP rank, so the
        column values stay global token ids. ``O(s_global^2)`` while building,
        which is why this mode is an equivalence experiment, not production.
        """
        with paddle.no_grad():
            indices, _ = _build_mqa_causal_topk_idxs_from_doc_bounds(
                b, s_global, doc_start, is_valid
            )
            if s_local is not None and s_local != s_global:
                indices = indices[
                    :, position_offset : position_offset + s_local
                ]
            indices = indices.contiguous()
        indices.stop_gradient = True
        return indices

    def _indexer_valid_range(
        self,
        s_global,
        doc_start,
        doc_len,
        is_valid,
        position_offset=0,
        s_local=None,
    ):
        """Non-local candidate range per query, in **global token** space.

        The forced local window already covers the last ``window_size`` causal
        tokens, so the indexer must only rank what lies *before* it. Clamping
        the right edge to ``doc_start + causal_len - window_size`` removes every
        duplicate while leaving the full top-k budget for distant tokens.
        Because the clamped end never exceeds the kernel's own causal limit, no
        masked ``-inf`` column can enter the top-k.

        Built over the global sequence and row-sliced to this CP rank; the two
        columns stay global token ids, which is what the kernel's
        ``seq_offset``-aware causal bound expects.

        Returns:
            ``(valid_range [1, s_local, 2] int32, row_empty [1, s_local, 1])``.
        """
        positions = paddle.arange(s_global, dtype="int64")
        causal_avail = paddle.minimum(positions - doc_start + 1, doc_len)
        n_avail = paddle.clip(causal_avail - self.window_size, min=0)
        n_avail = paddle.where(is_valid, n_avail, paddle.zeros_like(n_avail))
        valid_range = paddle.stack(
            [doc_start, doc_start + n_avail], axis=-1
        ).cast("int32")
        if s_local is not None and s_local != s_global:
            valid_range = valid_range[
                position_offset : position_offset + s_local
            ]
            n_avail = n_avail[position_offset : position_offset + s_local]
        rows = int(valid_range.shape[0])
        return valid_range.unsqueeze(0), (n_avail == 0).reshape([1, rows, 1])

    def _sparse_attn(self, query, kv, token_indices, sm_scale, d_v):
        """Sparse MQA over the absorbed latent, via the shared cudnn backend.

        Same FlashMLA sparse forward + cuDNN DSA backward pair that the CSA/HCA
        layers use; the absorbed layout only differs in ``d_v`` (512 value dims
        out of a 576-wide query/key) and in the sink being optional --
        ``softmax_offset`` is ``None`` when ``add_full_attention_sink_bias`` is
        off, which the backend turns into a sinkless softmax. Query-head padding
        to the DSA-fixed ``h_q == 64`` is the backend's job.
        """
        from paddlefleet.fusions.mqa_sparse_attn import mqa_sparse_attn

        return mqa_sparse_attn(
            query,
            kv,
            token_indices,
            sm_scale,
            d_v,
            attn_sink=self.softmax_offset,
        )

    @staticmethod
    def _deabsorb(core_out, v_b_proj_weight) -> Tensor:
        """``[b, s, h * kv_lora_rank]`` -> ``[b, s, h * v_head_dim]``."""
        b, s, _ = core_out.shape
        kv_lora_rank, h, v_head_dim = v_b_proj_weight.shape
        out = core_out.reshape([b, s, h, kv_lora_rank])
        out = paddle.einsum("bshl,lhv->bshv", out, v_b_proj_weight)
        return out.reshape([b, s, h * v_head_dim])

    # ------------------------------------------------------------------
    # mqa_dsa
    # ------------------------------------------------------------------
    def _forward_dsa(
        self,
        query,
        kv,
        x,
        qr,
        v_b_proj_weight,
        doc_start,
        doc_len,
        is_valid,
        doc_lens,
        kv_lora_rank,
        input_ids=None,
        position_offset=0,
    ) -> Tensor:
        from paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            cudnn_indexer_topk_fwd,
        )

        b, s = int(query.shape[0]), int(query.shape[1])
        s_global = s * self.cp_size
        # The indexer loss is only attached on the grad-enabled forward. Under
        # full-layer recompute the first forward runs under no_grad and must
        # only materialise indices; both passes see identical inputs, so the
        # top-k kernel reselects the same columns. (Its tie-break is not
        # bit-stable in every shape -- one document spanning the whole sequence
        # drifts by ~2% of the slots between identical calls -- but any residual
        # drift only changes which columns the backward differentiates.)
        need_loss = (
            self.training
            and paddle.is_grad_enabled()
            and self.indexer_loss_coeff > 0
        )

        if not self.indexer_use_sparse_loss and not need_loss:
            # Phase-2 mode with nothing to learn this step (eval, or
            # ``dsa_indexer_loss_coeff == 0``): attention does not consume the
            # indexer, so skip its projections entirely instead of computing
            # them under ``no_grad`` and throwing the result away.
            with paddle.no_grad():
                token_indices = self._build_full_causal_indices(
                    b, s_global, doc_start, is_valid, position_offset, s
                )
            core_out = self._sparse_attn(
                query, kv, token_indices, self.softmax_scale, kv_lora_rank
            )
            return self._deabsorb(core_out, v_b_proj_weight)

        with paddle.no_grad():
            window_idxs = None
            if self.indexer_use_sparse_loss:
                window_idxs = _build_window_topk_idxs_from_doc_bounds(
                    b, s_global, self.window_size, doc_start, is_valid
                ).cast("int32")
                if self.cp_enabled:
                    window_idxs = window_idxs[
                        :, position_offset : position_offset + s
                    ]
            valid_range, row_empty = self._indexer_valid_range(
                s_global, doc_start, doc_len, is_valid, position_offset, s
            )

        x_det, qr_det = x.detach(), qr.detach()
        if need_loss:
            x_det.stop_gradient = False
            qr_det.stop_gradient = False
            q_idx, k_idx, w_idx = self.indexer.forward_before_topk(
                x_det, qr_det, position_offset, self.cp_group
            )
        else:
            with paddle.no_grad():
                q_idx, k_idx, w_idx = self.indexer.forward_before_topk(
                    x_det, qr_det, position_offset, self.cp_group
                )
        # ``DSAIndexer`` pre-bakes ``head_dim**-0.5`` into the weights, but both
        # cuDNN indexer kernels apply ``dim**-0.5`` themselves (the backward one
        # hardcodes it). Undo the pre-bake once so forward and backward agree.
        w_idx = w_idx * (float(self.indexer.head_dim) ** 0.5)

        # Both forward top-k paths return a table of exactly ``topk_effective``
        # columns (short rows are ``-1`` padded), and the backward kernel
        # ``indexer_backward_sm100.__init__`` asserts ``topk % block_I == 0``
        # with ``block_I=128``. So keep the configured budget instead of
        # clamping it to the sequence length, which would break that assert.
        attn_topk = int(self.indexer.index_topk)
        # ``dsa_indexer_use_sparse_loss`` means here exactly what it means for
        # the CSA layers of the same model
        # (``_resolve_csa_indexer_loss_topk_effective``, csa_attention.py:1091):
        # the indexer loss may score a *wider* table than attention consumes.
        #   True  -> KL over exactly the set attention uses (selected-set KL).
        #   False -> KL over the full candidate table, so a freshly initialised
        #            indexer is supervised on columns it did not pick and cannot
        #            reinforce its own initial ranking.
        # CSA's "full" is its *compressed* candidate range (``s / ratio``, e.g.
        # 64 columns at ratio 128). These layers are uncompressed, so the full
        # range is the causal span itself and the cuDNN indexer bounds it at
        # ``_LOSS_TOPK_CAP``: with s=8192 the loss covers the top 2048 scoring
        # columns, not all 8192. Going truly dense would mean materialising the
        # ``[s, h, s]`` attention distribution per layer -- the very thing this
        # path exists to avoid -- so the cap stands.
        loss_topk = attn_topk
        if need_loss and not self.indexer_use_sparse_loss:
            loss_topk = max(
                attn_topk,
                min(_LOSS_TOPK_CAP, s_global // _TOPK_BLOCK * _TOPK_BLOCK),
            )
        # The THD/varlen fast path builds ``cu_seqlens_k`` from ``doc_lens``,
        # i.e. it assumes a document-compacted K buffer. At ratio 1 the K buffer
        # is the raw token sequence, so the two only coincide when the documents
        # exactly tile the sequence; otherwise fall back to the dense path.
        # ``doc_lens`` stays global: ``_make_cu_seqlens`` is CP-aware and uses
        # ``seq_offset`` to pick out the documents this rank queries
        # (csa_indexer_fwd_cudnn.py:407-466).
        doc_lens_arg = (
            doc_lens.tolist() if int(doc_lens.sum()) == s_global else None
        )

        def select_topk(width, want_scores):
            out = cudnn_indexer_topk_fwd(
                q_idx.detach(),
                k_idx.detach(),
                w_idx.detach(),
                ratio=1,
                topk_effective=width,
                valid_range=valid_range,
                doc_lens=doc_lens_arg,
                seq_offset=position_offset,
                return_topk_scores=want_scores,
            )
            idx = paddle.where(row_empty, paddle.full_like(out[0], -1), out[0])
            return idx, (out[2] if want_scores else None)

        with paddle.no_grad():
            if not self.indexer_use_sparse_loss:
                # Phase 2: attention attends to the full per-document causal
                # set, bit-identical to ``hybrid_mla_attention="mqa_full_causal"``
                # -- the indexer's ranking is still being learned and must not
                # steer attention yet. Only the loss consumes the top-k table,
                # so there is exactly one ``select_topk`` call below.
                token_indices = self._build_full_causal_indices(
                    b, s_global, doc_start, is_valid, position_offset, s
                )
                topk_indices, attn_scores, reuse_for_loss = None, None, False
            else:
                # The attention width and the loss width are two separate
                # kernel calls, and deliberately so: the attention set must not
                # depend on how wide the loss happens to be.
                #
                # With today's ``_indexer_top_k_unfused`` (a deterministic
                # ``paddle.topk``) slicing one wide call would in fact give the
                # same columns -- measured over 6 shapes (pure causal s=512 and
                # s=1024, window-clamped s=512/1024, packed docs [256,256,512],
                # ratio=4): 0 ascending score steps in the emitted order and the
                # narrow table is 100% the wide table's prefix. But that is a
                # property of one helper, not of the interface: the pre-#1666
                # radix kernel emitted in ascending *position* order (same
                # shapes: ~61k ascending score steps, only 55.5% prefix match),
                # so a slice would have silently moved the attention set. Two
                # calls keep that decoupled by construction; the extra one runs
                # only on the grad-enabled forward of a full-loss step.
                #
                # ``return_topk_scores`` stays tied to ``need_loss`` alone,
                # never to the width decision, for the same reason: under the
                # old radix kernel the flag chose a different code path and
                # flipped 10-15% of the returned slots (it is a no-op for the
                # column set today, 0 slot mismatch on all 6 shapes above). The
                # scores of the narrow call are unused when the loss widens the
                # table.
                topk_indices, attn_scores = select_topk(attn_topk, need_loss)
                reuse_for_loss = loss_topk == attn_topk
                token_indices = paddle.concat(
                    [window_idxs, topk_indices], axis=-1
                ).contiguous()
        token_indices.stop_gradient = True

        core_out = self._sparse_attn(
            query, kv, token_indices, self.softmax_scale, kv_lora_rank
        )
        output = self._deabsorb(core_out, v_b_proj_weight)
        if not need_loss:
            return output

        with paddle.no_grad():
            if reuse_for_loss:
                loss_indices, loss_scores = topk_indices, attn_scores
            else:
                loss_indices, loss_scores = select_topk(loss_topk, True)
            valid = loss_indices >= 0
            scores = paddle.where(
                valid,
                loss_scores.cast("float32"),
                paddle.full(loss_indices.shape, _NEG_INF, dtype="float32"),
            )
            topk_probs = F.softmax(scores, axis=-1)
            topk_probs = paddle.where(
                valid, topk_probs, paddle.zeros_like(topk_probs)
            )
            target = self._attn_target(query.detach(), kv, loss_indices)
            kl = target * (
                paddle.log(target + _EPS) - paddle.log(topk_probs + _EPS)
            )
            # Masked reduction, same as csa_attention.py:1302-1306, over the same
            # row mask csa_attention.py:2411-2443 builds. It has to come from
            # ``input_ids``, not from the document metadata: a packed sequence's
            # trailing padding is folded into the last document's row range, so
            # ``attn_mask_startend_row_indices`` -- and therefore ``is_valid`` --
            # still marks those rows valid. ``input_ids != pad_token_id`` is the
            # backstop that catches them, keeping the padding out of both the
            # logged loss and the gradient denominator.
            loss_mask, valid_rows = self._indexer_loss_mask(input_ids, b, s)
            # Under CP each rank reduces over its own query rows only. With a
            # mask the denominator is already the *global* valid-row count, so
            # the per-rank losses sum to the global one; without a mask each rank
            # produced a local mean and needs ``/cp_size``. Same split as
            # csa_attention.py:2792-2810.
            loss_coeff = (
                self.indexer_loss_coeff
                if loss_mask is not None
                else self.indexer_loss_coeff / self.cp_size
            )
            kl_per_pos = kl.sum(axis=-1)
            if loss_mask is None:
                loss = kl_per_pos.mean() * loss_coeff
            else:
                loss = (kl_per_pos * loss_mask).sum() / valid_rows * loss_coeff

        DSAIndexerLossLoggingHelper.save_loss_to_tracker(
            loss=loss,
            layer_number=self.layer_number,
            num_layers=DSAIndexerLossLoggingHelper.get_total_num_layers(
                self.config
            ),
        )
        # Argument order follows ``TileLangCSAIndexerLossAutoScaler.forward``
        # (csa_attention.py): ``output, target, index_q, weights, index_k_comp,
        # topk_indices, topk_probs, loss_coeff, indexer_backend,
        # num_rows_override, loss_mask``. ``target`` sits second, right after
        # ``output`` -- the CSA call sites spell that as
        # ``apply(output, target, *state)``.
        return TileLangCSAIndexerLossAutoScaler.apply(
            output,
            target,
            q_idx,
            w_idx,
            k_idx,
            loss_indices,
            topk_probs,
            loss_coeff,
            "cudnn",
            # ``num_rows_override`` + ``loss_mask``: the backward zeroes the pad
            # rows and rescales the cuDNN kernel's built-in ``1/(B*Sq)`` into
            # ``1/valid_rows``, matching the forward reduction above. Both
            # ``None`` (no ``input_ids``) leaves the kernel's own mean in place.
            valid_rows,
            loss_mask,
        )

    def _indexer_loss_mask(self, input_ids, b, s):
        """``([b, s] float32 row mask, its row count)`` from ``input_ids``.

        ``(None, None)`` when no ``input_ids`` reached this layer (inference and
        the direct-construction unit tests), which keeps the plain row mean.

        Under CP the mask is this rank's row slice but the denominator is the
        **global** valid-row count, so summing the per-rank losses reproduces the
        single-rank reduction. ``input_ids`` arrives sharded unless
        ``experimental_dataflow``, exactly as at ``csa_attention.py:2419-2428``.
        """
        if input_ids is None:
            return None, None
        pad_token_id = getattr(self.config, "pad_token_id", 0)
        assert pad_token_id is not None, (
            "pad_token_id must be set in config when input_ids is provided"
        )
        if self.cp_enabled:
            if not getattr(self.config, "experimental_dataflow", False):
                input_ids = ContextParallelGatherOp.apply(
                    input_ids, axis=1, mode=self.config.cp_balance_mode
                )
            loss_mask_global = (
                input_ids.reshape([b, self.cp_size * s]) != pad_token_id
            ).astype(paddle.float32)
            valid_rows = max(float(loss_mask_global.sum()), 1.0)
            offset = self.cp_rank * s
            return loss_mask_global[:, offset : offset + s], valid_rows
        loss_mask = (input_ids.reshape([b, s]) != pad_token_id).astype(
            paddle.float32
        )
        return loss_mask, max(float(loss_mask.sum()), 1.0)

    def _attn_target(self, query, kv, topk_indices) -> Tensor:
        """KL target: head-summed attention probs restricted to the top-k set.

        The tilelang ``csa_attn_target_reducesum`` kernel requires a
        power-of-two head dim, which the 576-wide latent is not, and the dense
        ``_compute_attn_target_on_selected_set`` materialises ``[b, h, s, s]``
        (16x the FLOPs at ``s=8192, topk=512``). So gather the selected keys in
        query-row chunks instead: ``s*h*topk*dk`` MACs, no ``s*s`` tensor.

        The matmul runs in the input dtype (bf16) with fp32 accumulation, which
        is what the tilelang kernel does internally for the CSA layers; only the
        softmax and the L1 normalisation are fp32. The chunk height follows
        ``_TARGET_ROW_SLOTS / topk``, so the gather buffer stays the same size
        whether the table is the attention one or the wider loss one.

        Args:
            query: ``[1, s, h, dk]`` detached absorbed query (local rows).
            kv: ``[1, s_global, dk]`` latent keys (all-gathered under CP).
            topk_indices: ``[1, s, topk]`` int32 global column ids, ``-1`` for
                empty slots.

        Returns:
            ``[1, s, topk]`` float32 rows summing to 1 (0 for empty rows).
        """
        s, topk = int(query.shape[1]), int(topk_indices.shape[-1])
        dk = int(query.shape[-1])
        chunk = max(1, _TARGET_ROW_SLOTS // topk)
        q0, kv0, idx0 = query[0], kv[0], topk_indices[0]
        parts = []
        for start in range(0, s, chunk):
            end = min(start + chunk, s)
            idx_c = idx0[start:end].cast("int64")
            valid = idx_c >= 0
            safe = paddle.where(valid, idx_c, paddle.zeros_like(idx_c))
            k_sel = paddle.gather(kv0, safe.flatten(), axis=0).reshape(
                [end - start, topk, dk]
            )
            scores = (
                paddle.matmul(q0[start:end], k_sel, transpose_y=True).cast(
                    "float32"
                )
                * self.softmax_scale
            )
            scores = paddle.where(
                valid.unsqueeze(1), scores, paddle.full_like(scores, _NEG_INF)
            )
            probs = F.softmax(scores, axis=-1).sum(axis=1)  # head-sum [c, topk]
            parts.append(paddle.where(valid, probs, paddle.zeros_like(probs)))
        target = paddle.concat(parts, axis=0)
        target = target / target.sum(axis=-1, keepdim=True).clip(min=_EPS)
        return target.unsqueeze(0)
