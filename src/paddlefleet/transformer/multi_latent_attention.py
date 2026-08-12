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

import logging
import math
import os
from dataclasses import dataclass
from functools import partial
from typing import NoReturn

import paddle
from paddle import Tensor
from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer
from paddle.distributed.fleet.meta_parallel.zero_bubble_utils import (
    WeightGradStore,
)
from paddle.distributed.fleet.utils import recompute

from paddlefleet.context_parallel_utils import (
    ContextParallelAllGatherOp,
    ContextParallelScatterOp,
    preprocess_index,
    preprocess_index_dual_chunks,
)
from paddlefleet.models.common.embeddings import (
    apply_rotary_pos_emb,
)
from paddlefleet.models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding,
)
from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
    YarnRotaryEmbedding as YarnRotaryEmbedding,
    _yarn_get_mscale,
)
from paddlefleet.parallel_state import (
    get_context_parallel_world_size,
)
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.recompute_utils import (
    keep_indexer_grad_path,
    need_recompute_in_block,
    need_recompute_in_first_n,
)
from paddlefleet.tensor_parallel import RecomputeWithoutOutput
from paddlefleet.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from paddlefleet.transformer.attention import Attention
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import get_pg_rank, get_pg_size

logger = logging.getLogger(__name__)


def _is_incremental_decode(past_key_values, layer_idx, use_cache) -> bool:
    """Whether this call is an incremental decode step for ``layer_idx``.

    True once the layer has already written its prefill KV into the cache. Based
    on the cache state rather than the query length so that a one-token prompt
    is still treated as a prefill.
    """
    if not use_cache or past_key_values is None or layer_idx is None:
        return False
    has_layer_cache = getattr(past_key_values, "has_layer_cache", None)
    if has_layer_cache is None:
        return False
    return bool(has_layer_cache(layer_idx))


def build_hysparse_valid_range(
    attn_mask_startend_row_indices,
    seq_len,
    batch_size,
    window_size=None,
):
    """Build ``valid_range`` [B, S, 2] int32 for the HySparse TileLang ops.

    Each query token ``t`` gets a half-open valid key-column range ``[bos, eos)``:

    * ``eos = t + 1`` (causal upper bound).
    * ``bos`` = start of the document containing ``t`` (document mask). When
      ``window_size`` is given, ``bos`` is additionally clamped up to
      ``t - window_size + 1`` (causal sliding window).

    Document boundaries are recovered from the flashmask
    ``attn_mask_startend_row_indices`` of shape ``[B, *, S, *]`` whose first
    head / first channel holds, per token, the **exclusive end** of the document
    that token belongs to (same convention as
    ``utils.get_doc_lens`` / ``csa_attention._derive_csa_doc_boundaries``).
    When it is ``None`` a single document (``bos`` document part = 0) is assumed.

    The block grid used by the block-score / block-sparse operators is anchored
    at ``bos`` (document-relative blocks), so the *document* range (no window
    clamp) must be used for block scoring and the block-sparse branch, while the
    windowed range is used for the sliding-window main path.
    """
    positions = paddle.arange(seq_len, dtype="int64").unsqueeze(0)  # [1, S]
    if attn_mask_startend_row_indices is not None:
        # (C) Convention guard: we read the exclusive document-end from
        # channel [:, 0, :, 0]. This only holds for the flashmask layout
        # [B, num_masks, S, num_bounds] whose first mask / first bound carries
        # the per-token exclusive doc-end (== utils.get_doc_lens /
        # csa_attention._derive_csa_doc_boundaries). A silent upstream layout
        # change (extra mask channels, bidirectional bounds, transposed axes)
        # would make the [:, 0, :, 0] slice mean something else and corrupt
        # every downstream bos -> block bucket. Assert the structural shape
        # (host-side, free) so such a change fails loudly here instead of
        # silently mis-bucketing.
        # Use explicit raises (not assert): this is a production forward path
        # and asserts are stripped under `python -O`, which would let a changed
        # upstream layout silently mis-bucket every bos -> block instead of
        # failing loudly here.
        if attn_mask_startend_row_indices.ndim != 4:
            raise ValueError(
                "attn_mask_startend_row_indices must be 4-D "
                "[B, num_masks, S, num_bounds] so [:, 0, :, 0] is the "
                "per-token exclusive doc-end; got ndim="
                f"{attn_mask_startend_row_indices.ndim} "
                f"shape={attn_mask_startend_row_indices.shape}"
            )
        if attn_mask_startend_row_indices.shape[2] != seq_len:
            raise ValueError(
                "attn_mask_startend_row_indices axis-2 must be the "
                f"query length S={seq_len}; got shape="
                f"{attn_mask_startend_row_indices.shape} "
                "(layout changed? [:, 0, :, 0] would no longer be the doc-end)"
            )
        # A legal document mask carries a single exclusive-doc-end bound on the
        # last axis: the per-token exclusive doc-end read via [:, 0, :, 0].
        # shape[3] > 1 (e.g. bidirectional start+end bounds) would make bound 0
        # mean something other than the doc-end and mis-bucket every bos ->
        # block; reject it here. (Axis 1 may be > 1: a multi-head flashmask
        # whose heads share one doc layout is valid and read via head 0.)
        if attn_mask_startend_row_indices.shape[3] != 1:
            raise ValueError(
                "attn_mask_startend_row_indices must be a document mask with a "
                "single exclusive-doc-end bound (shape[3] == 1); got shape="
                f"{attn_mask_startend_row_indices.shape} "
                "([:, 0, :, 0] would no longer be the doc-end)"
            )
        # [B, *, S, *] -> [B_mask, S] exclusive document end per token.
        de = attn_mask_startend_row_indices[:, 0, :, 0].cast("int64")  # [Bm, S]
        # The flashmask row indices may carry a batch of 1 that broadcasts over
        # the data batch (all sequences share one document layout). Expand so
        # the produced valid_range matches the query/key/value batch instead of
        # the mask's batch.
        if de.shape[0] == 1 and batch_size > 1:
            de = de.expand([batch_size, seq_len])
        bsz = de.shape[0]
        pos_b = positions.expand([bsz, seq_len])  # [B, S]
        is_boundary = paddle.zeros([bsz, seq_len], dtype="bool")
        is_boundary[:, 0] = True
        # a new document starts at t when the previous token's doc-end equals t
        # and the doc-end value actually changes.
        is_boundary[:, 1:] = (pos_b[:, 1:] == de[:, :-1]) & (
            de[:, 1:] != de[:, :-1]
        )
        doc_start = paddle.cummax(
            is_boundary.cast("int64") * pos_b, axis=1
        ).values  # [B, S] most-recent document start <= t
    else:
        bsz = batch_size
        doc_start = paddle.zeros([bsz, seq_len], dtype="int64")

    pos_b = positions.expand([bsz, seq_len])
    bos = doc_start
    if window_size is not None and window_size > 0:
        bos = paddle.maximum(doc_start, pos_b - window_size + 1)
    eos = pos_b + 1
    valid_range = paddle.stack([bos, eos], axis=-1).cast("int32")  # [B, S, 2]
    return valid_range.contiguous()


def _ec_compatible_rope_apply(
    q_pe,
    k_pe,
    seq_len,
    rope_base=1000000.0,
    position_offset=0,
    position_ids=None,
    cp_balance_mode="dualchunk_allgather",
):
    """Apply RoPE using EC's complex multiplication method (no YaRN, no mscale).

    This exactly matches ErnieCore's compute_freqs_cis_mrope_and_apply_rotary_3d
    when position_ids are sequential [0, 1, 2, ..., seq_len-1] (text-only case
    where all 3 mRoPE axes have the same value).

    Args:
        q_pe: [B, S, H, D] query positional embedding portion
        k_pe: [B, S, 1, D] key positional embedding portion
        seq_len: sequence length
        rope_base: base frequency (default 1e6)
        position_offset: starting position index for autoregressive decode
        position_ids: optional [S] position ID in fastdeploy decode mode.
                     If None, defaults to [0, 1, ..., seq_len-1] (offset by position_offset).
    """
    head_dim = q_pe.shape[-1]
    # inv_freq same as EC: 1 / (base^(arange(0, dim, 2) / dim))
    freqs = 1.0 / (
        rope_base
        ** (paddle.arange(0, head_dim, 2, dtype="float32") / float(head_dim))
    )
    if get_context_parallel_world_size() > 1:
        # In EB dataflow and CP size > 1, shape of q is [b, s/cp, h, d],
        # we need to get full seq_len here
        seq_len = seq_len * get_context_parallel_world_size()

    # Compute positions: prefer 1D position_ids (fastdeploy decode), else use sequential with offset
    if position_ids is not None and position_ids.ndim == 1:
        positions = position_ids.astype(freqs.dtype)
    else:
        # position ids: [position_offset, position_offset+1, ..., position_offset+seq_len-1]
        positions = paddle.arange(
            position_offset, position_offset + seq_len, dtype="float32"
        )
    # freqs_table: [S, D/2]
    freqs_table = paddle.outer(positions, freqs)
    # Expand for batch: [1, S, D/2]
    freqs_expanded = freqs_table.unsqueeze(0)
    # Expand to match q_pe batch size: [B, S, D/2]
    freqs_expanded = freqs_expanded.expand(
        [q_pe.shape[0], seq_len, head_dim // 2]
    )
    # freqs_cis: complex [B, S, D/2] -> [B, S, 1, D/2]
    freqs_cis = paddle.polar(paddle.ones_like(freqs_expanded), freqs_expanded)
    freqs_cis = freqs_cis.unsqueeze(2)  # [B, S, 1, D/2]

    if get_context_parallel_world_size() > 1:
        # In EB dataflow and CP size > 1, freqs_cis is [b, s/cp, 1, d] in local
        # so, we need to scatter freqs_cis here
        freqs_cis = ContextParallelScatterOp.apply(
            freqs_cis, axis=1, mode=cp_balance_mode
        )

    # MD5 debug
    import hashlib as _hl

    _log_md5 = os.environ.get("LOG_LAYER_MD5", "0") == "1"
    if _log_md5:
        import paddle.distributed as _dist

        _r = _dist.get_rank() if _dist.is_initialized() else 0
        _fc_real = paddle.as_real(freqs_cis)
        _md5 = _hl.md5(_fc_real.cast("float32").numpy().tobytes()).hexdigest()
        print(
            f"[MD5 Probe PF] Rank={_r} freqs_cis MD5={_md5} shape={list(freqs_cis.shape)} q_dtype={q_pe.dtype}",
            flush=True,
        )

    # Apply to q_pe via complex multiplication (EC style: interleaved pairs)
    xq = paddle.reshape(
        q_pe.cast("float32"), [*q_pe.shape[:-1], -1, 2]
    )  # [B,S,H,D/2,2]
    xk = paddle.reshape(
        k_pe.cast("float32"), [*k_pe.shape[:-1], -1, 2]
    )  # [B,S,1,D/2,2]
    xq_ = paddle.as_complex(xq)  # [B,S,H,D/2]
    xk_ = paddle.as_complex(xk)  # [B,S,1,D/2]

    if _log_md5:
        _xq_data = paddle.as_real(xq_).cast("float32").numpy().tobytes()
        _xq_md5 = _hl.md5(_xq_data).hexdigest()
        print(
            f"[MD5 Probe PF] Rank={_r} xq_complex MD5={_xq_md5} shape={list(xq_.shape)}",
            flush=True,
        )

    xq_out = paddle.as_real(xq_ * freqs_cis)  # [B,S,H,D/2,2]
    xk_out = paddle.as_real(xk_ * freqs_cis)  # [B,S,1,D/2,2]

    xq_out = paddle.flatten(xq_out, start_axis=3)  # [B,S,H,D]
    xk_out = paddle.flatten(xk_out, start_axis=3)  # [B,S,1,D]

    return xq_out.cast(q_pe.dtype), xk_out.cast(k_pe.dtype)


@dataclass
class MLASelfAttentionSublayersSpec:
    """Sublayers for MLA self-attention layer."""

    q_a_layernorm: LayerSpec | type = None
    kv_a_layernorm: LayerSpec | type = None

    q_proj: LayerSpec | type = None
    q_a_proj: LayerSpec | type = None
    q_b_proj: LayerSpec | type = None
    kv_a_proj_with_mqa: LayerSpec | type = None
    kv_b_proj: LayerSpec | type = None
    core_attention: LayerSpec | type = None
    o_proj: LayerSpec | type = None
    gate_proj: LayerSpec | type = None


class FP8OverlapProj(paddle.autograd.PyLayer):
    """
    Replaces RowParallelLinear (no bias, mp==1) with explicit split backward.
    Defers dw computation via WeightGradStore to overlap with P2P communication.
    Bit-exact with F.linear(x, weight) for arbitrary batch dimensions.
    """

    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        # Bit-exact with RowParallelLinear mp==1, no bias:
        # F.linear(x, weight) = x @ weight, weight shape: [in, out]
        return paddle.nn.functional.linear(x, weight)

    @staticmethod
    def backward(ctx, out_grad):
        x, weight = ctx.saved_tensor()

        def _compute_weight_grad(x, out_grad, weight):
            with paddle.amp.auto_cast(False):
                # Flatten all leading batch dims to 2D before matmul,
                # so dw = x_2d.T @ out_grad_2d has shape [in, out] == weight.shape
                x_2d = x.reshape([-1, x.shape[-1]])  # [B*S, in]
                og_2d = out_grad.reshape([-1, out_grad.shape[-1]])  # [B*S, out]
                w_grad = paddle.matmul(
                    x_2d, og_2d, transpose_x=True
                )  # [in, out]
                # print("w_grad compute")

            if hasattr(weight, "main_grad"):
                if weight.main_grad is None:
                    weight.main_grad = paddle.zeros(
                        weight.shape, dtype=paddle.float32
                    )
                weight.main_grad.add_(w_grad)
            else:
                raise AssertionError("fp8 overlap need main_grad attribute")

            if hasattr(weight, "_apply_backward_hook"):
                weight._apply_backward_hook()

        # dx = out_grad @ weight.T, weight: [in, out] -> [out, in]
        dx = paddle.matmul(out_grad, weight, transpose_y=True)

        # dw computation (deferred via WeightGradStore)
        if not weight.stop_gradient:
            # print("enter overlap weight grad")
            WeightGradStore.enabled = True
            WeightGradStore.put(
                partial(
                    _compute_weight_grad, x.detach(), out_grad.detach(), weight
                )
            )
            WeightGradStore.enabled = False

        return dx, None


class MultiLatentAttention(Attention):
    """Multi-Latent Attention layer abstract class."""

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: MLASelfAttentionSublayersSpec,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        is_mtp_layer: bool = False,
    ) -> None:
        super().__init__(
            config=config,
            sublayers_spec=sublayers_spec,
            layer_number=layer_number,
            attention_type=attention_type,
            attn_mask_type=attn_mask_type,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )
        self.config: TransformerConfig
        is_dsv4_hybrid = (
            getattr(config, "experimental_attention_variant", None)
            == "dsv4_hybrid"
        )
        if is_dsv4_hybrid:
            self.q_lora_rank = config.hybrid_mla_q_lora_rank
            self.kv_lora_rank = config.hybrid_mla_kv_lora_rank
            self.qk_nope_head_dim = config.hybrid_mla_qk_nope_head_dim
            self.qk_rope_head_dim = config.hybrid_mla_qk_rope_head_dim
            self.v_head_dim = config.hybrid_mla_v_head_dim
            self.num_attention_heads = config.hybrid_mla_num_attention_heads
        else:
            self.q_lora_rank = config.q_lora_rank
            self.kv_lora_rank = config.kv_lora_rank
            self.qk_nope_head_dim = config.qk_nope_head_dim
            self.qk_rope_head_dim = config.qk_rope_head_dim
            self.v_head_dim = config.v_head_dim
            self.num_attention_heads = config.num_attention_heads
        # MLA has no GQA: K/V are always re-materialized from the shared latent
        # with ``num_attention_heads`` heads -- ``kv_b_proj`` is sized
        # ``num_attention_heads * (qk_nope_head_dim + v_head_dim)`` and the core
        # attention is built with ``num_key_value_heads=1`` below -- so the
        # config field never reaches a kernel here.
        #
        # For ``dsv4_hybrid`` the ``hybrid_mla_*`` dims are new and required to
        # be explicit (transformer_config.py:1450), so a mismatch there is a real
        # misconfiguration and is rejected. The plain ``num_key_value_heads``
        # field is shared with the model's dense/GQA layers and 25 in-tree MLA
        # configs legitimately disagree with ``num_attention_heads``
        # (DeepSeek-V4 ``dsv4_flash*`` uses 1 for "one latent"; the
        # eb5/ernielite ones inherit 4/8 from their dense siblings), so on that
        # path pin the layer-local count to what is actually materialized rather
        # than refusing to build.
        if is_dsv4_hybrid:
            kv_heads = config.hybrid_mla_num_key_value_heads
            if kv_heads != self.num_attention_heads:
                raise ValueError(
                    "MLA supports MHA only: hybrid_mla_num_key_value_heads "
                    f"({kv_heads}) must equal hybrid_mla_num_attention_heads "
                    f"({self.num_attention_heads})."
                )
        self.num_key_value_heads = self.num_attention_heads
        tp_size = get_pg_size(self.pg_collection.tp)
        assert self.num_attention_heads % tp_size == 0
        assert self.num_key_value_heads % tp_size == 0
        self.num_attention_heads_per_partition = (
            self.num_attention_heads // tp_size
        )
        self.num_key_value_heads_per_partition = (
            self.num_key_value_heads // tp_size
        )

        self.out_projection_size = self.v_head_dim * self.num_attention_heads

        if (
            self.is_swa
            and getattr(self.config, "swa_qk_nope_head_dim", None) is not None
        ):
            self.qk_nope_head_dim = self.config.swa_qk_nope_head_dim

        if (
            self.is_swa
            and getattr(self.config, "swa_qk_rope_head_dim", None) is not None
        ):
            self.qk_rope_head_dim = self.config.swa_qk_rope_head_dim

        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.head_dim = self.q_head_dim
        self.query_projection_size = self.q_head_dim * self.num_attention_heads
        self.key_projection_size = self.q_head_dim * self.num_key_value_heads
        self.value_projection_size = self.v_head_dim * self.num_key_value_heads
        self.out_projection_size = self.v_head_dim * self.num_attention_heads
        self.hidden_size_per_attention_head = self.q_head_dim
        self.value_hidden_size_per_attention_head = self.v_head_dim

        mscale = _yarn_get_mscale(
            self.config.rotary_scaling_factor, self.config.mscale_all_dim
        )
        self.softmax_scale = mscale * mscale / math.sqrt(self.q_head_dim)
        # mscale == 1.0 means softmax_scale equals default 1/sqrt(d), no need to pass explicitly
        self._softmax_scale_arg = None if mscale == 1.0 else self.softmax_scale

        # Hybrid MLA layers may run attention on the shared KV latent instead of
        # the per-head K/V produced by ``kv_b_proj`` ("runtime absorption").  This
        # keeps every parameter byte-identical to the MHA layout -- only the
        # activations change -- so an MHA checkpoint loads into an MQA run.
        self.mqa_latent = getattr(
            config, "experimental_attention_variant", None
        ) == "dsv4_hybrid" and getattr(
            config, "hybrid_mla_attention", "mha"
        ) in ("mqa_dsa", "mqa_full_causal")
        # ``mqa_split_kv_b_proj`` trades that property for speed:
        # ``kv_b_proj`` is replaced by standalone ``k_b_proj`` / ``v_b_proj``
        # absorption parameters, pre-laid-out so each side is one grouped GEMM
        # instead of a slice + ``einsum``. The two together hold exactly the
        # elements of ``kv_b_proj.weight``, which is therefore not built at all;
        # a checkpoint that predates them cannot be resumed in this mode until
        # the AOA statements that split it exist. Applies to both latent MQA
        # modes -- the split only concerns absorption, not the indexer.
        self.mqa_latent_split_kv_b = self.mqa_latent and getattr(
            config, "mqa_split_kv_b_proj", False
        )

        # ``apply_rope_fusion`` is a model-wide flag, but the fused MLA RoPE is
        # only applicable per (q, k) pair.  Latent MQA cannot use it because
        # ``fused_apply_mla_rope_for_kv`` consumes the per-head K/V that
        # absorption never materialises (``kv is None`` below).  The eager RoPE
        # is used for that layer instead of failing the whole run, so the
        # HCA/CSA layers -- which own an independent q/k pair and already route
        # through ``fused_apply_mla_rope_inplace``
        # (dsv4_hybrid_attention.py:1191, csa_attention.py:1063) -- still get
        # the fusion.  Mixing layouts is only unsafe *within* one q/k pair.
        #
        # This ``config.apply_rope_fusion and not self.mqa_latent`` test is
        # evaluated at each use site rather than cached here, so that toggling
        # ``config.apply_rope_fusion`` on an already-constructed model (e.g. the
        # KV-cache warning in generation/greedy_generator.py) still takes
        # effect.
        if self.mqa_latent:
            if self.config.apply_rope_fusion and not getattr(
                self.config, "mqa_latent_rope_fusion", False
            ):
                logger.warning(
                    "apply_rope_fusion has no effect on the RoPE of this "
                    "latent-MQA layer (layer_number=%s): the fused kernel it "
                    "selects needs the per-head K/V that absorption never "
                    "materialises, so this layer keeps the eager RoPE. Set "
                    "mqa_latent_rope_fusion=True to fuse it.",
                    getattr(self, "layer_number", -1),
                )
            if self.config.sequence_parallel:
                raise ValueError(
                    "latent MQA (hybrid_mla_attention='mqa_dsa' / "
                    "'mqa_full_causal') does not support sequence_parallel yet."
                )
            if get_pg_size(self.pg_collection.tp) != 1:
                raise ValueError(
                    "latent MQA (hybrid_mla_attention='mqa_dsa' / "
                    "'mqa_full_causal') does not support tensor parallel "
                    "(kv_b_proj is absorbed locally)."
                )

        if self.config.rope_type == "rope":
            self.rotary_pos_emb = RotaryEmbedding(
                self.qk_rope_head_dim,
                rotary_interleaved=self.config.rotary_interleaved,
                rotary_percent=1.0,
                rotary_base=self.rope_theta,
                cp_group=self.pg_collection.cp,
                use_accuracy_compatible=getattr(
                    self.config, "use_accuracy_compatible", False
                ),
            )
        elif self.config.rope_type == "yarn":
            self.rotary_pos_emb = YarnRotaryEmbedding(
                self.qk_rope_head_dim,
                rotary_interleaved=self.config.rotary_interleaved,
                rotary_base=self.rope_theta,
                scaling_factor=self.config.rotary_scaling_factor,
                original_max_position_embeddings=self.config.original_max_position_embeddings,
                beta_fast=self.config.beta_fast,
                beta_slow=self.config.beta_slow,
                mscale=self.config.mscale,
                mscale_all_dim=self.config.mscale_all_dim,
                # cp_group=self.pg_collection.cp,
                use_accuracy_compatible=getattr(
                    self.config, "use_accuracy_compatible", False
                ),
            )
        else:
            raise ValueError(
                f"Unsupported RoPE type: {self.config.rope_type}, supported types are "
                "'rope' and 'yarn'"
            )

        self.core_attention = build_spec_layer(
            sublayers_spec.core_attention,
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
            is_mtp_layer=self.is_mtp_layer,
            is_swa=self.is_swa,
            softmax_scale=self._softmax_scale_arg,
            k_channels=self.q_head_dim,
            v_channels=self.v_head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=1,
            cp_comm_type=cp_comm_type,
            pg_collection=self.pg_collection,
        )

        # Attention sink. Both sink-aware core attentions create their own
        # ``softmax_offset`` from ``add_full_attention_sink_bias`` /
        # ``softmax_type`` via ``build_softmax_offset``, so the parameter name is
        # ``self_attn.core_attention.softmax_offset`` in the dense and in the
        # non-absorbed-MQA phase alike and an MHA checkpoint stays loadable.
        #
        # The dense path feeds it to ``flashmask_attention_func(...,
        # learnable_sink=...)`` (dot_product_attention.py:653), which only exists
        # on the FA4 (cute) kernel: gated on ``FLAGS_flash_attn_version in
        # (3, 4)`` and asserting a bf16 sink (flash_mask/cute/interface.py).
        # With the default flag value of 2 the run dies at the *first forward*
        # with no hint about the cause, so fail at construction time. We
        # deliberately do NOT set the flag ourselves: it is process-global and
        # would switch every other (HCA / CSA) layer's kernel too.
        # Latent MQA needs neither check -- its block-sparse kernel
        # supports the sink natively and up-casts it internally. Neither do the
        # HySparse absorbed-MQA layers: ``gpt_layer_specs.py:318`` builds every
        # MLA layer as ``MQASelfAttention`` when ``enable_hy_sparse_attention``
        # is on, and for an SWA layer (``is_mqa``, :1919) that subclass runs the
        # TileLang / cuDNN MQA kernels directly with its own ``swa_attn_sink`` /
        # ``sparse_attn_sink``, dropping this redundant ``softmax_offset``
        # (:1942) as soon as the present ``__init__`` returns. Non-SWA HySparse
        # layers do fall back to the dense MLA forward, but they only get a
        # ``softmax_offset`` from ``add_full_attention_sink_bias``
        # (dot_product_attention.py:98), which the check below still covers.
        hysparse_absorbed_mqa = (
            self.config.enable_hy_sparse_attention and self.is_swa
        )
        if (
            not self.mqa_latent
            and not hysparse_absorbed_mqa
            and getattr(self.core_attention, "softmax_offset", None) is not None
        ):
            fa_version = int(
                paddle.get_flags(["FLAGS_flash_attn_version"])[
                    "FLAGS_flash_attn_version"
                ]
            )
            if fa_version not in (3, 4):
                raise RuntimeError(
                    "an MLA attention sink (add_full_attention_sink_bias / "
                    "softmax_type) needs the flashmask v4 (cute) kernel, but "
                    f"FLAGS_flash_attn_version={fa_version} (only 3 or 4 reach "
                    "that path). Either export FLAGS_flash_attn_version=4 "
                    "(NOTE: process-global -- it changes the HCA/CSA layers' "
                    "kernel as well), or run the hybrid MLA layers with "
                    "hybrid_mla_attention='mqa_dsa' / 'mqa_full_causal', whose "
                    "block-sparse kernel supports the sink natively."
                )
            if "bfloat16" not in str(self.config.params_dtype):
                raise RuntimeError(
                    "an MLA attention sink requires params_dtype='bfloat16', "
                    f"got {self.config.params_dtype!r}: the sink is created with "
                    "params_dtype and the flashmask v4 (cute) kernel asserts it "
                    "is bf16."
                )

        # Output.
        self.o_proj = build_spec_layer(
            sublayers_spec.o_proj,
            self.out_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.use_bias,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="proj",
            tp_group=self.pg_collection.tp,
        )

        # Gated attention
        self.gated_attention = getattr(self.config, "gated_attention", False)
        self.gated_attn_use_q_lora = getattr(
            self.config, "gated_attn_use_q_lora", False
        )
        if self.gated_attention and sublayers_spec.gate_proj is not None:
            # Gate input source: q_compressed (post q_a_layernorm, dim=q_lora_rank) when
            # gated_attn_use_q_lora is set, otherwise the full hidden_states.
            if self.gated_attn_use_q_lora:
                assert self.q_lora_rank is not None, (
                    "gated_attn_use_q_lora=True requires q_lora_rank is not None"
                )
                gate_in_dim = self.q_lora_rank
            else:
                gate_in_dim = self.config.hidden_size
            self.gate_proj = build_spec_layer(
                sublayers_spec.gate_proj,
                gate_in_dim,
                self.out_projection_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.use_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="mla_gate",
                tp_group=self.pg_collection.tp,
            )
            print(
                f"[GatedAttnCheck][init] layer={getattr(self, 'layer_number', -1)} "
                f"gated_attention={self.gated_attention} "
                f"gated_attn_use_q_lora={self.gated_attn_use_q_lora} "
                f"q_lora_rank={self.q_lora_rank} "
                f"hidden_size={self.config.hidden_size} "
                f"gate_in_dim={gate_in_dim} "
                f"gate_out_dim={self.out_projection_size}",
                flush=True,
            )
        else:
            self.gated_attention = False
            self.gate_proj = None

        self.recompute_gated_attn = (
            self.config.recompute_granularity == "selective"
            and self.config.recompute_modules is not None
            and "gated_attn" in self.config.recompute_modules
        )

        self.recompute_qkv_up_porj_and_rope = False
        if self.config.recompute_granularity == "selective":
            modules = self.config.recompute_modules
            if isinstance(modules, list) and "mla_qkv_recompute" in modules:
                self.recompute_qkv_up_porj_and_rope = (
                    True
                    if self.config.recompute_num_layers is None
                    else (
                        need_recompute_in_block(
                            self.layer_number,
                            self.config,
                            self.config.recompute_num_layers,
                        )
                        if self.config.recompute_method == "block"
                        else need_recompute_in_first_n(
                            self.layer_number,
                            self.config,
                            self.config.recompute_num_layers,
                        )
                    )
                )
            elif isinstance(modules, dict) and "mla_qkv_recompute" in modules:
                assert self.config.recompute_method in ["first_n", "block"]
                num_layers = modules["mla_qkv_recompute"]
                self.recompute_qkv_up_porj_and_rope = (
                    need_recompute_in_block(
                        self.layer_number, self.config, num_layers
                    )
                    if self.config.recompute_method == "block"
                    else need_recompute_in_first_n(
                        self.layer_number, self.config, num_layers
                    )
                )

        # VHA postmix: ungrouped low-rank cross-head mixing (I + U Vᵀ) on the head
        # axis, applied to the attention output (head space) before the output
        # projection. Reuses use_vha_attention / vha_postmix_rank. Ungrouped only:
        # MLA/MQA have no grouped o_proj to fold a block-diagonal mixer into.
        self.use_vha_postmix = getattr(config, "use_vha_attention", False)
        if self.use_vha_postmix:
            # Use an explicit ValueError (not assert): assertions are stripped
            # under `python -O`, which would silently let TP>1 mix only the
            # local heads of each rank and deviate from the declared full-head
            # postmix semantics.
            if get_pg_size(self.pg_collection.tp) != 1:
                raise ValueError(
                    "VHA postmix currently supports tensor parallel size 1 "
                    f"only, got tp={get_pg_size(self.pg_collection.tp)}."
                )
            nh = (
                self.num_attention_heads_per_partition
            )  # == num_attention_heads (TP=1)
            rank = getattr(config, "vha_postmix_rank", None)
            if rank is None:
                rank = nh // 4
            rank = max(1, min(rank, nh))
            self.vha_postmix_rank = rank
            self.vha_postmix_U = self.create_parameter(
                shape=[nh, rank],
                default_initializer=paddle.nn.initializer.Normal(
                    mean=0.0, std=0.01
                ),
            )
            self.vha_postmix_V = self.create_parameter(
                shape=[nh, rank],
                default_initializer=paddle.nn.initializer.Constant(
                    0.0
                ),  # identity at init
            )
        # Selective recompute for the VHA postmix. Only list configuration is
        # supported; honours recompute_num_layers + recompute_method
        # (first_n / block) like the other selective modules.
        modules = self.config.recompute_modules
        self.recompute_vha_postmix = False
        if (
            self.config.recompute_granularity == "selective"
            and modules is not None
        ):
            if isinstance(modules, dict) and "vha_postmix" in modules:
                raise ValueError(
                    "recompute_modules['vha_postmix'] only supports list "
                    "configuration"
                )
            if isinstance(modules, list) and "vha_postmix" in modules:
                n = self.config.recompute_num_layers
                if n is None:
                    self.recompute_vha_postmix = True
                elif self.config.recompute_method == "block":
                    self.recompute_vha_postmix = need_recompute_in_block(
                        self.layer_number, self.config, n
                    )
                elif self.config.recompute_method == "first_n":
                    self.recompute_vha_postmix = need_recompute_in_first_n(
                        self.layer_number, self.config, n
                    )
                else:
                    raise ValueError(
                        "recompute_method must be 'first_n' or 'block'"
                    )

    def _apply_vha_postmix(self, attn_out, U=None, V=None):
        # attn_out: [b, sq, nh_pp * v_head_dim] (head space, pre-gate / pre output proj).
        # Fused dense M = I + V @ U^T, then a single [nh,nh] GEMM on the head
        # axis: rank-independent and faster than the split low-rank form; differs
        # from the two-matmul path only by bf16 contraction order.
        if U is None:
            U = self.vha_postmix_U
        if V is None:
            V = self.vha_postmix_V
        b, sq = attn_out.shape[0], attn_out.shape[1]
        nh, d = self.num_attention_heads_per_partition, self.v_head_dim
        mixed = attn_out.reshape([b * sq, nh, d])
        M = paddle.matmul(V, U, transpose_y=True)  # [nh,r]@[r,nh]->[nh,nh]
        M = M + paddle.eye(nh, dtype=M.dtype)
        out = paddle.matmul(M, mixed)  # [nh,nh]@[B,nh,d]->[B,nh,d]
        return out.reshape([b, sq, nh * d])

    def _compute_absorbed_q(self, query):
        """
        Compute absorbed query for FD MLA decode kernel.

        The MLA decode kernel expects q in absorbed form:
            q_absorbed = [q_nope @ W_k_b, q_pe] per head
        where per-head dim = kv_lora_rank + qk_rope_head_dim (e.g. 576)

        Also returns wv_b for V de-absorption on the kernel output.

        Args:
            query: [b, s, heads, qk_nope_head_dim + qk_rope_head_dim]

        Returns:
            q_absorbed: [b, s, heads, kv_lora_rank + qk_rope_head_dim]
            wv_b: [heads, kv_lora_rank, v_head_dim]
        """
        qk_nope_head_dim = self.qk_nope_head_dim
        qk_rope_head_dim = self.qk_rope_head_dim
        kv_lora_rank = self.kv_lora_rank
        v_head_dim = self.v_head_dim
        num_heads = self.num_attention_heads_per_partition

        # Split query into nope and rope parts
        q_nope = query[
            ..., :qk_nope_head_dim
        ]  # [b, s, heads, qk_nope_head_dim]
        q_pe = query[..., qk_nope_head_dim:]  # [b, s, heads, qk_rope_head_dim]

        # Get kv_b_proj weight and reshape to per-head form
        # kv_b_proj.weight: [kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim)]
        kv_b_weight = self.kv_b_proj.weight
        w = kv_b_weight.reshape([kv_lora_rank, num_heads, -1]).transpose(
            perm=[1, 2, 0]
        )

        # w: [heads, qk_nope + v_head, kv_lora_rank]
        # wk_b: [heads, qk_nope_head_dim, kv_lora_rank]
        wk_b = w[:, :qk_nope_head_dim, :]
        # wv_b: [heads, kv_lora_rank, v_head_dim]
        wv_b = w[:, -v_head_dim:, :].transpose(perm=[0, 2, 1])

        # Absorption: q_nope @ wk_b => q_nope_absorbed
        # q_nope: [b, s, heads, qk_nope] -> [b*s, heads, qk_nope] -> [heads, b*s, qk_nope]
        orig_shape = q_nope.shape  # [b, s, heads, qk_nope]
        bs = orig_shape[0] * orig_shape[1]

        q_nope_3d = q_nope.reshape([bs, num_heads, qk_nope_head_dim]).transpose(
            [1, 0, 2]
        )
        q_pe_3d = q_pe.reshape([bs, num_heads, qk_rope_head_dim])
        # bmm: [heads, b*s, qk_nope] @ [heads, qk_nope, kv_lora_rank] -> [b*s, heads, kv_lora_rank]

        q_nope_absorbed = paddle.bmm(q_nope_3d, wk_b).transpose([1, 0, 2])
        # Concat: [b, s, heads, kv_lora_rank + qk_rope_head_dim]
        q_absorbed = paddle.concat([q_nope_absorbed, q_pe_3d], axis=-1)
        q_absorbed = q_absorbed.reshape(
            orig_shape[0], orig_shape[1], num_heads, -1
        )
        return q_absorbed, wv_b

    def forward(
        self,
        hidden_states,
        attention_mask,
        attn_mask_startend_row_indices: Tensor | None = None,
        key_value_states=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        in_recompute: bool = False,
        position_ids=None,
        shared_kv: list[Tensor] | None = None,
        **kwargs,
    ):
        """Forward pass for multi-latent attention"""
        from paddlefleet.transformer.transformer_layer import TransformerLayer

        _log = TransformerLayer._log_md5

        assert rotary_pos_emb is None, (
            "Rotary position embeddings should not be passed into MLA."
        )
        assert attention_bias is None, (
            "Attention bias should not be passed into MLA."
        )
        assert rotary_pos_cos is None and rotary_pos_sin is None, (
            "MLA does not support Flash Decoding"
        )

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention
        # Also get q_compressed for DSA indexer (if enabled)
        query, key, value, q_compressed, kv_compressed, k_pos_emb = (
            self.get_query_key_value_tensors(
                hidden_states,
                key_value_states,
                position_ids,
                packed_seq_params,
            )
        )

        layer_num = getattr(self, "layer_number", -1)
        _log(query, "attn_query", layer_num)
        _log(key, "attn_key", layer_num)
        if value is not None:
            _log(value, "attn_value", layer_num)

        attn_mask_type = self.attn_mask_type
        query = query.contiguous()
        key = key.contiguous()

        if value is not None:
            value = value.contiguous()

        # ==================================
        # core attention computation
        # ==================================

        # NOTE: For sequence parallel, the input is [seq, b, h],
        # transpose back to [b, seq, h] for attention computation
        # TODO: supports [seq, b, h] input in attention computation
        if self.config.sequence_parallel:
            query = query.transpose([1, 0, 2, 3]).contiguous()
            key = key.transpose([1, 0, 2, 3]).contiguous()
            value = value.transpose([1, 0, 2, 3]).contiguous()

        # Extract inference kwargs to pass through to core_attention
        past_key_values = kwargs.get("past_key_values")
        layer_idx = kwargs.get("layer_idx")
        use_cache = kwargs.get("use_cache", False)
        is_decode = _is_incremental_decode(
            past_key_values, layer_idx, use_cache
        )

        # The indexer-loss row mask needs ``input_ids``: the packed sequence's
        # trailing padding is invisible to ``attn_mask_startend_row_indices``.
        # Only the non-absorbed-MQA core attention accepts it (and only that one
        # owns an indexer), so keep the kwarg off every other core attention.
        core_attn_extra = {}
        if self.mqa_latent and kwargs.get("input_ids") is not None:
            core_attn_extra["input_ids"] = kwargs["input_ids"]

        if self.mqa_latent:
            # Query is already absorbed; the core attention only needs the V-side
            # de-absorption weight.
            q_absorbed = None
            if self.mqa_latent_split_kv_b:
                # Standalone parameter; fold back to the core attention's
                # grouped-matmul layout [n, v_head_dim, kv_lora_rank]. The
                # reshape is a zero-copy view of the 2-D parameter.
                wv_b = self.v_b_proj.reshape(
                    [
                        self.num_attention_heads_per_partition,
                        self.v_head_dim,
                        self.kv_lora_rank,
                    ]
                )
            else:
                # View of ``kv_b_proj.weight`` laid out for
                # ``einsum("bshl,lhv->bshv", out, w_v_b)``.
                wv_b = self.kv_b_proj.weight.reshape(
                    [
                        self.kv_lora_rank,
                        self.num_attention_heads_per_partition,
                        -1,
                    ]
                )[:, :, self.qk_nope_head_dim :]
        elif hasattr(self.core_attention.config, "forward_meta"):  # decode mode
            # Compute absorbed query and V de-absorption weight for FD MLA decode kernel
            # q_absorbed: [b, s, heads, kv_lora_rank + qk_rope_head_dim]
            # wv_b: [heads, kv_lora_rank, v_head_dim]
            q_absorbed, wv_b = self._compute_absorbed_q(query)
        else:
            q_absorbed, wv_b = None, None

        hy_sparse_full = (
            self.config.enable_hy_sparse_attention and shared_kv is not None
        )
        block_indices = None

        if hy_sparse_full and not is_decode:
            if get_context_parallel_world_size() > 1:
                cp_mode = getattr(
                    self.config, "cp_balance_mode", "dualchunk_allgather"
                )
                key = ContextParallelAllGatherOp.apply(key, 1, cp_mode)
                value = ContextParallelAllGatherOp.apply(value, 1, cp_mode)

            # HySparse full-attention layer. The full (dense) attention here is
            # computed by the MHA block-score TileLang op, which additionally
            # emits per-(query, key-block) max logits. We select the top-k key
            # blocks and share both the compressed KV latent and the selected
            # block indices with the downstream SWA layers' block-sparse branch.
            #
            # This branch is checked BEFORE recompute_core_attention: the FA4
            # block-score full path is a distinct computation that must produce
            # block_indices for the downstream SWA layer, and running the plain
            # recompute(core_attention) branch here would leave block_indices
            # undefined (UnboundLocalError at the shared_kv.append below) while
            # also failing to emit the top-k blocks. Activation recompute for
            # HySparse full layers is handled at the layer level
            # (HySparseTransformerLayer.full_recompute).
            #
            # Incremental decode (``is_decode``) skips this branch: the fused
            # block-score kernels score a whole query row against the keys of
            # the same forward, which a one-token step cannot provide. Decode
            # instead runs the generic core_attention path (which maintains the
            # KV cache) and re-derives the block scores from the cache in
            # ``_hy_sparse_decode_block_indices`` below.
            core_attn_out, block_indices = self._hy_sparse_full_attention(
                query,
                key,
                value,
                attn_mask_startend_row_indices,
            )
            if use_cache and past_key_values is not None:
                # This branch bypasses core_attention, so the layer's own KV
                # cache has to be seeded here for the later decode steps.
                past_key_values.update(key, value, layer_idx)
        elif self.recompute_core_attention and self.training:
            core_attn_out = recompute(
                self.core_attention,
                query,
                key,
                value,
                attention_mask.clone() if attention_mask is not None else None,
                attn_mask_startend_row_indices.clone()
                if attn_mask_startend_row_indices is not None
                else None,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                use_rr_flash_attention=self.use_rr_flash_attention,
                # DSA-specific parameters. ``recompute`` is a PyLayer, so with a
                # frozen backbone (``train_indexer_only``) every input here would
                # be detached, the segment output would inherit that, and the
                # indexer loss attached *inside* the segment would silently never
                # get a backward pass. This is the third segment that can hold an
                # indexer, next to the layer-level one (transformer_layer.py) and
                # ``DSv4HybridAttention``'s ``full_attn`` one.
                x=keep_indexer_grad_path(hidden_states, self.config),
                qr=q_compressed,
                # fastdeploy support
                kv_compressed=kv_compressed,
                k_pos_emb=k_pos_emb,
                q_absorbed=q_absorbed,
                v_b_proj_weight=wv_b,
                **core_attn_extra,
            )
        else:
            # Static batching attention kernel.
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_startend_row_indices,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                use_rr_flash_attention=self.use_rr_flash_attention
                and in_recompute,
                past_key_values=past_key_values,
                layer_idx=layer_idx,
                use_cache=use_cache,
                # DSA-specific parameters
                x=hidden_states,
                qr=q_compressed,
                # fastdeploy support
                kv_compressed=kv_compressed,
                k_pos_emb=k_pos_emb,
                q_absorbed=q_absorbed,
                v_b_proj_weight=wv_b,
                **core_attn_extra,
            )

        if self.recompute_qkv_up_porj_and_rope and self.training:
            assert getattr(self, "_qkv_recompute", None) is not None
            self._qkv_recompute.discard_output_and_register_recompute(
                core_attn_out
            )
            self._qkv_recompute = None

        _log(core_attn_out, "core_attn_out", layer_num)

        if hy_sparse_full:
            # Compressed KV latent shared with block-sparse attention in SWA
            # layers (single MQA head): [B, S, 1, kv_lora_rank + qk_rope_head_dim].
            shared_key = paddle.concat(
                [kv_compressed.unsqueeze(2), k_pos_emb], axis=-1
            )
            if use_cache and past_key_values is not None:
                # Inference: accumulate across decode steps and hand the SWA
                # layers the whole history instead of the current token only.
                shared_key = past_key_values.update_shared(
                    shared_key, layer_idx
                )
            if is_decode:
                # The decode attention kernel carries no block-scoring epilogue,
                # so score the single new query row against the cached keys
                # here. This keeps the SWA layers' sparse branch -- and hence
                # the model's output -- the same as a cache-less forward.
                block_indices = self._hy_sparse_decode_block_indices(
                    query, past_key_values, layer_idx
                )
            shared_kv.append(shared_key)
            # block_indices from the prefill block-score kernel above, or from
            # the decode-time scoring pass.
            shared_kv.append(block_indices)

        # =================
        # Output. [b, sq, h]
        # =================
        if self.config.sequence_parallel:
            core_attn_out = core_attn_out.transpose([1, 0, 2]).contiguous()

        # VHA postmix: low-rank cross-head mixing in head space, before the gate
        # (mix-then-gate). Skip the nested selective recompute when already
        # inside a layer-level recompute.
        if self.use_vha_postmix:
            if (
                self.recompute_vha_postmix
                and self.training
                and not in_recompute
            ):
                core_attn_out = recompute(
                    self._apply_vha_postmix, core_attn_out
                )
            else:
                core_attn_out = self._apply_vha_postmix(core_attn_out)

        # Apply gated attention
        if self.gated_attention:
            # Gate input source: q_compressed (post q_a_layernorm, dim=q_lora_rank) when
            # gated_attn_use_q_lora is set, otherwise hidden_states.
            gate_source = (
                q_compressed if self.gated_attn_use_q_lora else hidden_states
            )
            # [GatedAttnCheck][forward] debug print (kept commented for on-demand use):
            # if not getattr(self, "_gated_attn_fwd_logged", False):
            #     self._gated_attn_fwd_logged = True
            #     _src_is_qc = gate_source is q_compressed
            #     print(
            #         f"[GatedAttnCheck][forward] layer={getattr(self, 'layer_number', -1)} "
            #         f"gated_attn_use_q_lora={self.gated_attn_use_q_lora} "
            #         f"gate_source_is_q_compressed={_src_is_qc} "
            #         f"gate_source.shape={list(gate_source.shape)} "
            #         f"q_compressed.shape={list(q_compressed.shape)} "
            #         f"hidden_states.shape={list(hidden_states.shape)} "
            #         f"recompute_gated_attn={self.recompute_gated_attn}",
            #         flush=True,
            #     )
            if self.recompute_gated_attn:
                gate_recompute = RecomputeWithoutOutput()
                core_attn_out = gate_recompute.recompute(
                    self._gate,
                    gate_source,
                    core_attn_out,
                    preserve_rng_state=False,
                    share_grad_holder=True,
                )
            else:
                core_attn_out = self._gate(gate_source, core_attn_out)

        if getattr(self.config, "dw_p2p_overlap", False) and not getattr(
            self.config, "use_bias", False
        ):
            output = FP8OverlapProj.apply(core_attn_out, self.o_proj.weight)
            bias = None
        else:
            output, bias = self.o_proj(core_attn_out)

        if self.gated_attention and self.recompute_gated_attn:
            gate_recompute.discard_output_and_register_recompute(output)

        _log(output, "attn_o_proj_out", layer_num)

        return output, bias

    def _gate(self, gate_source, core_attn_out):
        gate, _ = self.gate_proj(gate_source)
        if self.config.sigmoid_gate_fusion:
            from paddlefleet.triton_ops import SigmoidGateFusionTriton

            core_attn_out = SigmoidGateFusionTriton.apply(core_attn_out, gate)
        else:
            core_attn_out = core_attn_out * paddle.nn.functional.sigmoid(gate)
        return core_attn_out

    def _hy_sparse_full_attention(
        self,
        query,
        key,
        value,
        attn_mask_startend_row_indices,
    ):
        """HySparse full-attention layer using the FA4-fused block-score op.

        Runs dense (decompressed) MHA attention through the FA4 sm100 kernel,
        whose softmax epilogue additionally emits per-(query, key-block) max
        logits at near-zero extra cost (``block_score_fa4_attn_fwd``). From those
        we recover block scores and select the top-k key blocks per query token.
        The selected block indices (shared across heads, document-relative) are
        returned so the downstream SWA layers' block-sparse branch can gather
        exactly the same blocks.

        Args:
            query: [B, S, H, Dk] decompressed query (H independent heads).
            key:   [B, S, H, Dk] decompressed key.
            value: [B, S, H, Dv] decompressed value.
            attn_mask_startend_row_indices: flashmask doc boundaries or ``None``.

        Returns:
            core_attn_out: [B, S, H*Dv] dense attention output.
            block_indices: [B, S, topk] int32 selected block ids (-1 padding).
        """
        from paddlefleet.tilelang_ops.hysparse import (
            block_score_fa4_attn_fwd,
            select_topk_blocks,
        )

        use_tl = getattr(self.config, "hy_sparse_full_attn_use_tilelang", False)
        if use_tl:
            from paddlefleet.tilelang_ops.hysparse.block_score_mha import (
                block_score_mha_attn_fwd,
            )

        b, kv_s, h, _dv = value.shape
        q_s = query.shape[1]
        block_B = self.config.hy_sparse_block_size
        topk = self.config.hy_sparse_topk
        sm_scale = self.softmax_scale

        # Build document ranges and flashmask rows in global sequence order,
        # then scatter the query axis back to the local dual-chunk layout.
        # The gathered K/V stays global while the fused kernels operate on
        # local queries.
        valid_range = build_hysparse_valid_range(
            attn_mask_startend_row_indices, kv_s, b
        )
        startend_row_indices = attn_mask_startend_row_indices
        cp_size = get_pg_size(self.pg_collection.cp)
        if cp_size > 1:
            cp_mode = getattr(
                self.config, "cp_balance_mode", "dualchunk_allgather"
            )
            valid_range = ContextParallelScatterOp.apply(
                valid_range, 1, cp_mode
            )
            if startend_row_indices is None:
                startend_row_indices = paddle.full(
                    [b, 1, kv_s, 1],
                    fill_value=kv_s,
                    dtype="int32",
                )
            if startend_row_indices is not None:
                cp_rank = get_pg_rank(self.pg_collection.cp)
                # Match DotProductAttention's CP FlashMask contract: first
                # expand global [LTS] to [LTS, UTE], then localize all query
                # row boundaries together for this rank's dual chunks.
                if startend_row_indices.shape[-1] == 1:
                    causal_end = paddle.arange(
                        kv_s, dtype=startend_row_indices.dtype
                    ).reshape([1, 1, kv_s, 1])
                    causal_end = paddle.expand_as(
                        causal_end, startend_row_indices
                    )
                    startend_row_indices = paddle.concat(
                        [startend_row_indices, causal_end], axis=-1
                    )
                else:
                    raise ValueError(
                        "HySparse CP expects one FlashMask boundaries, "
                        f"but got {startend_row_indices.shape[-1]}"
                    )
                if cp_mode == "dualchunk_allgather":
                    seq_blocksize = q_s // 2
                    startend_row_indices = preprocess_index_dual_chunks(
                        startend_row_indices,
                        chunk_id_first=cp_rank,
                        chunk_id_second=2 * cp_size - cp_rank - 1,
                        seq_blocksize=seq_blocksize,
                        max_seqlen_q=seq_blocksize,
                    )
                elif cp_mode == "contiguous_allgather":
                    startend_row_indices = preprocess_index(
                        startend_row_indices,
                        chunk_id=cp_rank,
                        seq_blocksize=q_s,
                        max_seqlen_q=q_s,
                    )
                else:
                    raise ValueError(
                        f"Unsupported HySparse CP balance mode: {cp_mode}"
                    )

        if use_tl:
            # Independent TileLang MHA scorer: masks purely via valid_range
            # (document + causal), no flashmask input needed.
            out, lse, block_logit = block_score_mha_attn_fwd(
                query,
                key,
                value,
                valid_range=valid_range,
                sm_scale=sm_scale,
                block_B=block_B,
                causal=cp_size == 1,
            )
        else:
            out, lse, block_logit = block_score_fa4_attn_fwd(
                query,
                key,
                value,
                valid_range=valid_range,
                sm_scale=sm_scale,
                block_B=block_B,
                causal=cp_size == 1,
                startend_row_indices=startend_row_indices,
            )

        block_indices = select_topk_blocks(
            block_logit,
            lse,
            valid_range,
            topk,
            block_B,
        )
        core_attn_out = out.reshape([b, q_s, h * _dv])
        return core_attn_out, block_indices

    def _hy_sparse_decode_block_indices(
        self, query, past_key_values, layer_idx
    ):
        """Select top-k key blocks for one incremental-decode query token.

        Prefill gets the block scores for free from the fused block-score kernel
        (:meth:`_hy_sparse_full_attention`); the decode kernel has no such
        epilogue, so the scoring row is recomputed from this layer's cached keys.
        Without it the downstream SWA layers would have to drop their
        block-sparse branch and quietly stop matching a cache-less forward.

        Args:
            query: [B, 1, H, Dk] this step's query (post-RoPE, decompressed).
            past_key_values: cache exposing ``get_layer_kv``.
            layer_idx: this layer's cache slot.

        Returns:
            [B, 1, topk] int32 document-relative block ids (-1 padding).
        """
        from paddlefleet.tilelang_ops.hysparse import (
            decode_block_logit,
            select_topk_blocks,
        )

        cached_key, _ = past_key_values.get_layer_kv(layer_idx)
        kv_len = cached_key.shape[1]
        block_B = self.config.hy_sparse_block_size
        # Generation feeds one document per batch row, so the block grid is
        # anchored at column 0 -- the same document layout the prefill
        # valid_range describes (``attn_mask_startend_row_indices`` filled with
        # the prompt length).
        valid_range = paddle.concat(
            [
                paddle.zeros([query.shape[0], 1, 1], dtype="int32"),
                paddle.full([query.shape[0], 1, 1], kv_len, dtype="int32"),
            ],
            axis=-1,
        )
        block_logit, lse = decode_block_logit(
            query,
            cached_key,
            valid_range,
            sm_scale=self.softmax_scale,
            block_B=block_B,
        )
        return select_topk_blocks(
            block_logit,
            lse,
            valid_range,
            self.config.hy_sparse_topk,
            block_B,
        )


class MLASelfAttention(MultiLatentAttention):
    """MLA Self-attention layer class

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: MLASelfAttentionSublayersSpec,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        is_mtp_layer: bool = False,
    ):
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        super().__init__(
            config=config,
            sublayers_spec=sublayers_spec,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )

        q_lora_rank = self.q_lora_rank
        kv_lora_rank = self.kv_lora_rank
        if q_lora_rank is None:
            # Not projecting query
            self.q_proj = build_spec_layer(
                sublayers_spec.q_proj,
                self.config.hidden_size,
                self.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="q_proj",
            )

        else:
            self.q_a_proj = build_spec_layer(
                sublayers_spec.q_a_proj,
                self.config.hidden_size,
                q_lora_rank,
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="q_a_proj",
                skip_weight_param_allocation=False,
                tp_group=pg_collection.tp,
            )

            self.q_b_proj = build_spec_layer(
                sublayers_spec.q_b_proj,
                q_lora_rank,
                self.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="q_b_proj",
                tp_group=pg_collection.tp,
            )

        self.kv_a_proj_with_mqa = build_spec_layer(
            sublayers_spec.kv_a_proj_with_mqa,
            self.config.hidden_size,
            kv_lora_rank + self.qk_rope_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="kv_a_proj_with_mqa",
            skip_weight_param_allocation=False,
            tp_group=pg_collection.tp,
        )

        # In split mode ``k_b_proj`` / ``v_b_proj`` below *replace* this
        # projection: together they hold exactly its elements, nothing in the
        # forward reads it, and it would never receive a gradient. Building it
        # anyway would double both the resident parameter bytes and the
        # checkpoint size for this projection, so it is not built at all.
        self.kv_b_proj = (
            None
            if self.mqa_latent_split_kv_b
            else build_spec_layer(
                sublayers_spec.kv_b_proj,
                kv_lora_rank,
                self.num_attention_heads
                * (self.qk_nope_head_dim + self.v_head_dim),
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="kv_b_proj",
                tp_group=pg_collection.tp,
            )
        )

        qk_norm_eps = getattr(self.config, "qk_norm_eps", None)
        if qk_norm_eps is None:
            qk_norm_eps = self.config.rms_norm_eps

        if self.mqa_latent_split_kv_b:
            # K absorption half of ``kv_b_proj``, which this mode replaces.
            # Logically [heads, kv_lora_rank, qk_nope_head_dim] --
            # ``fused_grouped_matmul``'s ``[G, R, D]`` contract, which lets the
            # kernel walk the head dim through strides -- but *stored* with the
            # leading two dims folded so the parameter itself is 2-D.
            #
            # The fold is what the checkpoint needs: AOA statements have only
            # split / concat / permute and no reshape primitive
            # (``flex_checkpoint/aoa/aoa_engine.py:519``), so a conversion from
            # an unsplit ``kv_b_proj.weight`` cannot produce a 3-D tensor.
            # Exposing a reshaped *view* from ``sharded_state_dict`` instead is
            # not safe: DCP reshapes the ``ShardedWeight.local_tensor`` in place
            # (``reshape_`` /``flatten_`` at
            # ``dcp/load_state_dict.py:557,741,998``), which on a view that
            # aliases the parameter corrupts the parameter's own shape.
            #
            # The 3-D form is recovered per forward with a zero-copy reshape.
            # TP is forced to 1 in this mode, so ``num_attention_heads_per_partition``
            # is the full head count and the parameter needs no TP attributes.
            params_dtype = self.config.params_dtype
            self.k_b_proj = self.create_parameter(
                shape=[
                    self.num_attention_heads_per_partition * kv_lora_rank,
                    self.qk_nope_head_dim,
                ],
                dtype=params_dtype,
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
            self.config.init_method(self.k_b_proj)

            # V de-absorption half, same contract and same fold: logically
            # [heads, v_head_dim, kv_lora_rank], stored as 2-D.
            self.v_b_proj = self.create_parameter(
                shape=[
                    self.num_attention_heads_per_partition * self.v_head_dim,
                    kv_lora_rank,
                ],
                dtype=params_dtype,
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
            self.config.init_method(self.v_b_proj)

        if q_lora_rank is not None:
            self.q_a_layernorm = build_spec_layer(
                sublayers_spec.q_a_layernorm,
                hidden_size=q_lora_rank,
                config=self.config,
                eps=qk_norm_eps,
            )

        self.kv_a_layernorm = build_spec_layer(
            sublayers_spec.kv_a_layernorm,
            hidden_size=kv_lora_rank,
            config=self.config,
            eps=qk_norm_eps,
        )

    def muon_slice_specs(self, muon_configs):
        """Muon orthogonal-slice specs for MLA projections (split_head only)."""
        from paddlefleet.transformer.muon_utils import ortho_per_head

        if (
            muon_configs.get("muon_qkv_update_mode", "split_head")
            != "split_head"
        ):
            return {}

        num_heads = self.num_attention_heads_per_partition
        qk_nope = self.qk_nope_head_dim
        qk_rope = self.qk_rope_head_dim
        kv_lora = self.kv_lora_rank

        specs = {}
        if hasattr(self, "q_b_proj"):
            specs["q_b_proj.weight"] = (
                ortho_per_head,
                {"heads": num_heads, "head_sizes": [qk_nope, qk_rope]},
            )
        specs["kv_a_proj_with_mqa.weight"] = (
            ortho_per_head,
            {"head_sizes": [kv_lora, qk_rope]},
        )
        if self.mqa_latent_split_kv_b:
            # ``kv_b_proj`` is split into the standalone ``k_b_proj`` /
            # ``v_b_proj`` absorption parameters, so the per-head blocks Muon
            # must orthogonalise moved with them. Both are 2-D with the head dim
            # folded into the leading axis, so a head is one equal block along
            # ``axis=-2`` (``-2`` rather than ``0`` so a 3-D input -- Muon
            # batching several same-shape parameters -- still splits the head
            # axis and not the batch axis):
            #   k_b_proj -> [kv_lora_rank, qk_nope_head_dim] per head, the same
            #               block as the K half of the unsplit weight
            #   v_b_proj -> [v_head_dim, kv_lora_rank] per head, i.e. the
            #               transpose of the V half
            # The V side is therefore orthogonalised in its transpose, which
            # puts the block back in the unsplit weight's orientation before
            # Muon sees it. Newton-Schulz alone would not care (it transposes
            # any block with rows > cols and back), but ``_scaling_fn`` does:
            # only ``muon_version=3``'s ``max(dout, din) ** 0.5`` is symmetric
            # in the two dims, while versions 1/2 scale with ``dout / din`` and
            # would apply the reciprocal ratio to a transposed block.
            specs["k_b_proj"] = (
                ortho_per_head,
                {"heads": num_heads, "axis": -2},
            )
            specs["v_b_proj"] = (
                ortho_per_head,
                {"heads": num_heads, "axis": -2, "transposed": True},
            )
            # ``kv_b_proj`` gets no gradient in this mode -- orthogonalising it
            # would only burn a per-head Newton-Schulz every step.
        else:
            specs["kv_b_proj.weight"] = (
                ortho_per_head,
                {"heads": num_heads, "head_sizes": [qk_nope, self.v_head_dim]},
            )
        if getattr(self, "gate_proj", None) is not None:
            specs["gate_proj.weight"] = (ortho_per_head, {"heads": num_heads})
        # MQA (subclass) runs a second gated branch for block-sparse attention.
        # sparse_gate_proj is built from gate_proj's in/out sizes, so it shares
        # the head-major column layout and needs the same per-head slicing.
        if getattr(self, "sparse_gate_proj", None) is not None:
            specs["sparse_gate_proj.weight"] = (
                ortho_per_head,
                {"heads": num_heads},
            )
        return specs

    def _is_cudagraph_active(self) -> bool:
        """Check if CUDA Graph capture or replay is currently active.

        Uses forward_meta.step_use_cudagraph flag set on core_attention.config
        by FastDeploy's model runner before CUDA graph capture/replay.
        """
        forward_meta = getattr(self.core_attention.config, "forward_meta", None)
        if forward_meta is None:
            return False
        return getattr(forward_meta, "step_use_cudagraph", False)

    def get_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # b = batch size, s = sequence length, h = hidden size, n = num attention heads
        # Attention heads [b, s, n*h]
        assert hidden_states.ndim == 3, (
            f"hidden_states should be 3D, [b, s, n*h], got {hidden_states.ndim}D"
        )

        # =========================================
        # Prepare RoPE and seqlen related params
        # =========================================
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            hidden_states, self.config, packed_seq_params
        )

        # rotary_pos_emb:[1, s, 1, 64]
        mscale = 1.0
        rotary_pos_cos = None
        rotary_pos_sin = None
        rotary_pos_emb = None
        packed_seq = (
            packed_seq_params is not None
            and packed_seq_params.qkv_format == "thd"
        )
        if not getattr(self.config, "mla_use_nope", False):
            if self.config.rope_type == "rope":
                rotary_pos_emb = self.rotary_pos_emb(
                    rotary_seq_len,
                    packed_seq=packed_seq,
                    position_ids=None if self.training else position_ids,
                )
            else:
                if bool(self.config.apply_rope_fusion) and not self.mqa_latent:
                    rotary_pos_cos, rotary_pos_sin = (
                        self.rotary_pos_emb.get_cached_cos_sin(
                            rotary_seq_len,
                            dtype=hidden_states.dtype,
                            packed_seq=packed_seq,
                        )
                    )
                    rotary_pos_emb = None
                    from paddlefleet.triton_ops.fused_mla_yarn_rope_apply import (
                        fused_apply_mla_rope_for_kv,
                        fused_apply_mla_rope_for_q,
                    )

                    assert (
                        fused_apply_mla_rope_for_q is not None
                        and fused_apply_mla_rope_for_kv is not None
                    ), "Fused MLA RoPE apply is not imported successfully"
                else:
                    rotary_pos_emb, mscale = self.rotary_pos_emb(
                        rotary_seq_len,
                        packed_seq=packed_seq,
                        position_ids=None if self.training else position_ids,
                    )
                    # mscale is already accounted for in self.softmax_scale; set to 1.0 to avoid double-applying
                    # mscale = 1.0

        cp_size = get_context_parallel_world_size()
        if cp_size > 1 and not getattr(self.config, "mla_use_nope", False):
            # Keep RoPE inputs local to the current CP rank before the fused
            # and non-fused apply paths consume them.
            if packed_seq_params is not None:
                raise ValueError(
                    "Context parallel RoPE scatter in MLA does not support "
                    "packed_seq_params yet."
                )
            if self.config.sequence_parallel:
                local_seq_len = (
                    hidden_states.shape[0]
                    * self.config.tensor_model_parallel_size
                )
            else:
                local_seq_len = hidden_states.shape[1]
            expected_rotary_seq_len = cp_size * local_seq_len
            if rotary_seq_len != expected_rotary_seq_len:
                raise ValueError(
                    "Context parallel requires rotary_seq_len to be the global "
                    f"sequence length, got rotary_seq_len={rotary_seq_len}, "
                    f"expected={expected_rotary_seq_len}, cp_size={cp_size}, "
                    f"local_seq_len={local_seq_len}, "
                    f"sequence_parallel={self.config.sequence_parallel}, "
                    f"tensor_model_parallel_size="
                    f"{self.config.tensor_model_parallel_size}."
                )
            if rotary_pos_cos is not None and rotary_pos_sin is not None:
                if (
                    rotary_pos_cos.shape[1] != rotary_seq_len
                    or rotary_pos_sin.shape[1] != rotary_seq_len
                ):
                    raise ValueError(
                        "Context parallel requires rotary_pos_cos/sin sequence "
                        f"length to match rotary_seq_len, got "
                        f"cos={rotary_pos_cos.shape}, "
                        f"sin={rotary_pos_sin.shape}, "
                        f"rotary_seq_len={rotary_seq_len}."
                    )
                rotary_pos_cos = ContextParallelScatterOp.apply(
                    rotary_pos_cos, axis=1, mode=self.config.cp_balance_mode
                ).contiguous()
                rotary_pos_sin = ContextParallelScatterOp.apply(
                    rotary_pos_sin, axis=1, mode=self.config.cp_balance_mode
                ).contiguous()
            elif rotary_pos_emb is not None:
                if rotary_pos_emb.shape[1] != rotary_seq_len:
                    raise ValueError(
                        "Context parallel requires rotary_pos_emb sequence "
                        f"length to match rotary_seq_len, got "
                        f"rotary_pos_emb={rotary_pos_emb.shape}, "
                        f"rotary_seq_len={rotary_seq_len}."
                    )
                rotary_pos_emb = ContextParallelScatterOp.apply(
                    rotary_pos_emb, axis=1, mode=self.config.cp_balance_mode
                )
            else:
                raise ValueError(
                    "Context parallel requires rotary_pos_emb or rotary_pos_cos/sin "
                    "to be prepared before applying MLA RoPE."
                )

        if (
            packed_seq_params is not None
            and packed_seq_params.qkv_format == "thd"
        ):
            if packed_seq_params.cu_seqlens_q_padded is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
            else:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
            if packed_seq_params.cu_seqlens_kv_padded is not None:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
            else:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        # =========================================
        # QKV down projection and layernorm
        # =========================================
        if self.q_lora_rank is not None:
            # if q_a_proj is ColumnParallelLinear:
            #     q_compressed: [b, s, q_lora_rank / TP]
            q_compressed, _ = self.q_a_proj(hidden_states)

            # When output is sharded (ColumnParallelLinear):
            # Gather output to restore output dim q_lora_rank;
            # Scatter sequence back to s / TP if sequence-parallel
            if q_compressed.size(-1) != self.q_lora_rank:
                q_compressed = gather_from_tensor_model_parallel_region(
                    q_compressed
                )
                if self.config.sequence_parallel:
                    q_compressed = scatter_to_sequence_parallel_region(
                        q_compressed
                    )
        else:
            q_compressed = hidden_states

        # if kv_a_proj_with_mqa is ColumnParallelLinear:
        #     kv_combined: [b, s, (kv_lora_rank + qk_rope_head_dim) / TP]
        kv_combined, _ = self.kv_a_proj_with_mqa(hidden_states)
        if kv_combined.size(-1) != self.kv_lora_rank + self.qk_rope_head_dim:
            # kv_combined: [b, s, (kv_lora_rank + qk_rope_head_dim)]
            kv_combined = gather_from_tensor_model_parallel_region(kv_combined)
            # kv_compressed:[b, s, kv_lora_rank], k_pos_emb: [b, s, qk_rope_head_dim]
            kv_compressed, k_pos_emb = paddle.split(
                kv_combined,
                [self.kv_lora_rank, self.qk_rope_head_dim],
                axis=-1,
            )
            if self.config.sequence_parallel:
                # kv_compressed:[b, s / TP, kv_lora_rank]
                kv_compressed = scatter_to_sequence_parallel_region(
                    kv_compressed
                )
        else:
            # kv_compressed:[b, s / TP, kv_lora_rank], k_pos_emb: [b, s / TP, qk_rope_head_dim]
            kv_compressed, k_pos_emb = paddle.split(
                kv_combined,
                [self.kv_lora_rank, self.qk_rope_head_dim],
                axis=-1,
            )
            if (
                get_pg_size(self.pg_collection.tp) > 1
                and self.config.sequence_parallel
            ):
                # k_pos_emb: [b, s, qk_rope_head_dim]
                k_pos_emb = gather_from_sequence_parallel_region(
                    k_pos_emb, group=self.pg_collection.tp
                )

        # if packed_seq_params is not None:
        #     # PaddleFleet batch-first: [b=1, t, h] -> squeeze dim0 (batch) -> [t, h]
        #     # (SP seq-first: [t, b=1, h] -> squeeze dim1 (batch) -> [t, h])
        #     batch_dim = 1 if self.config.sequence_parallel else 0
        #     q_compressed = q_compressed.squeeze(batch_dim)
        #     kv_compressed = kv_compressed.squeeze(batch_dim)
        #     k_pos_emb = k_pos_emb.squeeze(batch_dim)

        # =========================================
        # Apply norm
        # =========================================

        if self.q_lora_rank is not None:
            # q_compressed: [num_tokens, q_lora_rank]
            q_compressed = self.q_a_layernorm(q_compressed)

        kv_compressed = self.kv_a_layernorm(kv_compressed)

        # === MD5 probes for MLA intermediate values ===
        from paddlefleet.transformer.transformer_layer import TransformerLayer

        _log = TransformerLayer._log_md5
        _log(q_compressed, "mla_q_compressed_normed", self.layer_number)
        _log(kv_compressed, "mla_kv_compressed_normed", self.layer_number)
        _log(k_pos_emb, "mla_k_pos_emb_raw", self.layer_number)

        # =========================================
        # QKV up projection and RoPE apply
        # =========================================

        def qkv_up_proj_and_rope_apply(
            q_compressed,
            kv_compressed,
            k_pos_emb,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            position_ids=None,
        ):
            """
            Apply the up projection and RoPE to the query and key.
            When sequence packing enabled, the input tensors adopt a packed shape of [t, ...];
            otherwise, they maintain the unpacked shape [b, s, ...]. In subsequent code comments,
            we uniformly use [num_tokens, ...] to denote [b, s, ...] or [t, ...] for two cases.
            """
            if self.q_lora_rank is not None:
                # q_compressed: [num_tokens, q_lora_rank]
                # q: [num_tokens, n * (qk_nope_head_dim + qk_rope_head_dim)]
                q, _ = self.q_b_proj(q_compressed)
            else:
                # q_compressed: [num_tokens, hidden_size]
                # q: [num_tokens, n * (qk_nope_head_dim + qk_rope_head_dim)]
                q, _ = self.q_proj(q_compressed)

            # q: [num_tokens, n, q_head_dim]
            q = q.view(
                *q.size()[:-1],
                self.num_attention_heads_per_partition,
                self.q_head_dim,
            )

            # kv: [num_tokens, n * (qk_nope_head_dim + v_head_dim)]
            # Absorbed MQA never materialises the per-head K/V: ``kv_b_proj`` is
            # folded into q (K side) and into the attention output (V side).
            if self.mqa_latent:
                kv = None
            else:
                kv, _ = self.kv_b_proj(kv_compressed)

                # Debug: print kv shape
                # if self.layer_number == 0:
                #     print(f"[DEBUG MLA layer {self.layer_number}] kv shape after kv_b_proj: {kv.shape}", flush=True)

                # kv: [num_tokens, n, (qk_nope_head_dim + v_head_dim)]
                kv = kv.view(
                    *kv.size()[:-1],
                    self.num_attention_heads_per_partition,
                    self.qk_nope_head_dim + self.v_head_dim,
                )

            # if self.layer_number == 0:
            #     print(f"[DEBUG MLA layer {self.layer_number}] kv shape after view: {kv.shape}", flush=True)

            # [num_tokens, qk_rope_head_dim] -> [num_tokens, 1, qk_rope_head_dim]
            k_pos_emb = paddle.unsqueeze(k_pos_emb, -2)

            if getattr(self.config, "mla_use_nope", False):
                q_no_pe = q[..., : self.qk_nope_head_dim]
                q_pos_emb = q[..., self.qk_nope_head_dim :]
                k_no_pe, value = paddle.split(
                    kv,
                    [self.qk_nope_head_dim, self.v_head_dim],
                    axis=-1,
                )
                k_pe = k_pos_emb
                query = paddle.cat([q_no_pe, q_pos_emb], axis=-1)
                k_pos_emb = k_pos_emb.expand(
                    *k_pos_emb.shape[:-2],
                    self.num_attention_heads_per_partition,
                    k_pos_emb.shape[-1],
                )
                key = paddle.cat([k_no_pe, k_pos_emb], axis=-1)
            elif bool(self.config.apply_rope_fusion) and not self.mqa_latent:
                from paddlefleet.triton_ops.fused_mla_yarn_rope_apply import (
                    fused_apply_mla_rope_for_kv,
                    fused_apply_mla_rope_for_q,
                )

                assert not self.config.sequence_parallel, (
                    "sequence_parallel for apply_rope_fusion in mla is not supported yet."
                )
                assert cu_seqlens_q is None, (
                    "thd for apply_rope_fusion in mla is not supported yet."
                )
                cp_size = get_pg_size(self.pg_collection.cp)
                cp_rank = get_pg_rank(self.pg_collection.cp)
                q_len = q.size(1)
                if (
                    packed_seq_params is None
                    or self.config.context_parallel_size == 1
                ) and self.config.rope_type == "rope":
                    # During training, the sequence length is always
                    # the full rotary_pos_emb length, except for sequence packing + CP.
                    # We need the full rotary_pos_emb to cover the full sequence,
                    # so we do not shorten it here.
                    rotary_pos_emb = rotary_pos_emb[:, 0:q_len]
                if self.config.rope_type == "rope":
                    cos = paddle.cos(rotary_pos_emb).contiguous()
                    sin = paddle.sin(rotary_pos_emb).contiguous()
                else:
                    cos = rotary_pos_cos
                    sin = rotary_pos_sin
                if cos.shape[1] != q_len or sin.shape[1] != q_len:
                    raise ValueError(
                        "Fused MLA RoPE requires local cos/sin sequence "
                        f"length to match q_len, got cos={cos.shape}, "
                        f"sin={sin.shape}, q_len={q_len}."
                    )
                query = fused_apply_mla_rope_for_q(
                    q,
                    cos,
                    sin,
                    self.qk_nope_head_dim,
                    self.qk_rope_head_dim,
                    cu_seqlens_q,
                    cp_rank,
                    cp_size,
                )
                key, value = fused_apply_mla_rope_for_kv(
                    kv,
                    k_pos_emb,
                    cos,
                    sin,
                    self.qk_rope_head_dim,
                    self.qk_nope_head_dim,
                    self.v_head_dim,
                    cu_seqlens_kv,
                    cp_rank,
                    cp_size,
                )

                # dynamic_inference not supported for now
                if not self.training:
                    raise NotImplementedError(
                        "apply_rope_fusion does not support dynamic inference yet."
                    )

                k_pe = None
            else:
                # Determine seq length:
                #   packed 3D [t, n, d]      -> dim 0
                #   SP     4D [s, b, n, d]   -> dim 0
                #   normal 4D [b, s, n, d]   -> dim 1
                if q.ndim == 3 or self.config.sequence_parallel:
                    q_len = q.size(0)
                else:
                    q_len = q.size(1)

                # Determine RoPE start position from position_ids (for decode offset)
                # .item() triggers D2H sync which is forbidden inside CUDA graph capture:
                # it causes cudaErrorStreamCaptureUnsupported (900), invalidating the stream
                # BEFORE the try/except can save it.  Guard with _is_cudagraph_active() so
                # we never attempt the sync during capture at all.
                start_pos = 0
                if position_ids is not None and not self._is_cudagraph_active():
                    # Normal path: works when not inside CUDA Graph capture
                    if position_ids.numel() == q_len:
                        start_pos = int(position_ids.flatten()[0].item())

                if get_context_parallel_world_size() == 1 and (
                    packed_seq_params is None
                    or self.config.context_parallel_size == 1
                ):
                    if rotary_pos_emb.shape[1] >= start_pos + q_len:
                        rotary_pos_emb = rotary_pos_emb[
                            :, start_pos : start_pos + q_len
                        ]
                    else:
                        # During inference with KV cache, rotary_pos_emb was
                        # computed for the current input length only, but
                        # position_ids indicate we need embeddings at start_pos.
                        # Recompute with the correct offset.
                        if self.config.rope_type == "rope":
                            rotary_pos_emb = self.rotary_pos_emb(
                                q_len,
                                offset=start_pos,
                                packed_seq=packed_seq,
                                position_ids=None
                                if self.training
                                else position_ids,
                            )
                        else:
                            # mscale is constant for Yarn (depends only on
                            # model hyper-params), so we can safely drop the
                            # recomputed value and keep the outer-scope one.
                            rotary_pos_emb, _ = self.rotary_pos_emb(
                                q_len,
                                offset=start_pos,
                                packed_seq=packed_seq,
                                position_ids=None
                                if self.training
                                else position_ids,
                            )

                if packed_seq_params is not None:
                    raise ValueError(
                        "MLA qkv_up_proj_and_rope_apply does not support "
                        "packed_seq_params yet."
                    )
                expected_rotary_pos_emb_len = q_len
                if rotary_pos_emb.shape[1] != expected_rotary_pos_emb_len:
                    raise ValueError(
                        "MLA RoPE requires local rotary_pos_emb sequence "
                        f"length to match expected length, got "
                        f"rotary_pos_emb={rotary_pos_emb.shape}, "
                        f"expected={expected_rotary_pos_emb_len}, q_len={q_len}, "
                        f"sequence_parallel={self.config.sequence_parallel}, "
                        f"tensor_model_parallel_size="
                        f"{self.config.tensor_model_parallel_size}."
                    )

                # Replace paddle.split with zero-copy slice views.
                q_no_pe = q[..., : self.qk_nope_head_dim]
                q_pos_emb = q[..., self.qk_nope_head_dim :]

                # k_no_pe: [num_tokens, n, qk_nope_head_dim]
                # value: [num_tokens, n, v_head_dim]
                if self.mqa_latent:
                    k_no_pe, value = None, None
                else:
                    k_no_pe, value = paddle.split(
                        kv,
                        [self.qk_nope_head_dim, self.v_head_dim],
                        axis=-1,
                    )

                # When sequence_parallel is enabled and not packed,
                # q/k are seq-first [s, b, n, d] but rotary_pos_emb is
                # batch-first [1, s, 1, d]. Transpose to [s, 1, 1, d]
                # so broadcasting aligns correctly in _apply_rotary_pos_emb_bshd.
                if self.config.sequence_parallel and rotary_pos_emb.ndim == 4:
                    rotary_pos_emb = rotary_pos_emb.transpose([1, 0, 2, 3])

                k_rope_fused_with_cat = False
                if self.config.gpt_model_use_experimental_version:
                    # EC-compatible RoPE: complex rotation, no YaRN, no mscale
                    from paddlefleet.transformer.transformer_layer import (
                        TransformerLayer,
                    )

                    _log = TransformerLayer._log_md5
                    _log(q_pos_emb, "mla_q_pe_before_rope", self.layer_number)
                    _log(k_pos_emb, "mla_k_pe_before_rope", self.layer_number)
                    q_pos_emb, k_pos_emb = _ec_compatible_rope_apply(
                        q_pos_emb,
                        k_pos_emb,
                        q_len,
                        rope_base=self.rope_theta,  # Must match EC's config.rope_theta
                        position_offset=start_pos,
                        position_ids=position_ids,
                        cp_balance_mode=self.config.cp_balance_mode,
                    )
                    _log(q_pos_emb, "mla_q_pe_after_rope", self.layer_number)
                    _log(k_pos_emb, "mla_k_pe_after_rope", self.layer_number)
                elif (
                    getattr(self.config, "mqa_latent_rope_fusion", False)
                    # Only the absorbed-MQA layers, and this one does gate rather
                    # than assert: an unabsorbed MLA layer is a legitimate
                    # configuration that has its own fused path via
                    # ``apply_rope_fusion``, so it belongs on the eager branch
                    # here, not in an error. Absorbed layers are the ones that
                    # path cannot serve, because ``fused_apply_mla_rope_for_kv``
                    # needs the per-head K/V that absorption never materialises
                    # (:1895), which is why this kernel exists.
                    and self.mqa_latent
                ):
                    # Fused rotate_half path, bit-exact with the eager branch
                    # below (which it mirrors exactly: rotate_half, no
                    # de-interleaving, no inverse, mscale folded into cos/sin).
                    #
                    # Anything that would make that branch compute a different
                    # rotation must stop the run, not silently feed the kernel a
                    # layout it does not implement. These are production
                    # forward-time checks on user config / input, so they raise
                    # ValueError explicitly rather than assert: ``python -O``
                    # strips ``assert`` and would let the wrong layout reach
                    # ``fused_apply_rope_half`` and silently corrupt results.
                    # Matches the shape guards elsewhere in this file.
                    if self.config.multi_latent_attention:
                        raise ValueError(
                            "mqa_latent_rope_fusion does not implement the "
                            "multi_latent_attention de-interleave (0::2 / 1::2)"
                        )
                    if self.config.rotary_interleaved:
                        raise ValueError(
                            "mqa_latent_rope_fusion pairs the two halves of the "
                            "rope block, not alternating channels"
                        )
                    if getattr(self.config, "high_precision_rope", False):
                        raise ValueError(
                            "high_precision_rope for mqa_latent_rope_fusion is "
                            "not supported yet"
                        )
                    if self.config.sequence_parallel:
                        raise ValueError(
                            "sequence_parallel for mqa_latent_rope_fusion is not "
                            "supported yet: k_pos_emb needs the per-rank freqs "
                            "slice that apply_rotary_pos_emb does internally via "
                            "sp_group (rope_utils.py:221)"
                        )
                    if rotary_pos_emb is None:  # pragma: no cover
                        # Unreachable in practice: for a latent-MQA layer
                        # rotary_pos_emb is always a real tensor (the ``= None``
                        # at :1509 is gated on ``not mqa_latent``), and :1874
                        # already dereferences ``rotary_pos_emb.shape`` above.
                        # Kept as a defensive guard so a future caller that
                        # wires the cached cos/sin path in here fails loudly
                        # rather than feeding the kernel a missing angle tensor;
                        # excluded from coverage because it cannot fire.
                        raise ValueError(
                            "mqa_latent_rope_fusion needs the angle tensor, but "
                            "the caller prepared cached cos/sin instead"
                        )
                    if (  # pragma: no cover
                        cu_seqlens_q is not None or cu_seqlens_kv is not None
                    ):
                        # Unreachable in practice: cu_seqlens_q/kv are only set
                        # from packed_seq_params (:1576), but packed_seq_params
                        # is refused earlier at :1850. Kept as a defensive guard
                        # so a future caller wiring thd in here fails loudly
                        # instead of feeding an unimplemented layout to the
                        # kernel; excluded from coverage because it cannot fire.
                        raise ValueError(
                            "thd for mqa_latent_rope_fusion is not supported yet"
                        )

                    # Neither call works in place. This closure is replayed by
                    # ``recompute_qkv_up_porj_and_rope`` and ``k_pos_emb`` is one
                    # of its arguments, created outside, so an in-place rotation
                    # would be applied a second time on replay. ``q`` is created
                    # inside (:1710) and would survive that, but relying on where
                    # a line happens to sit is not a property worth depending on.
                    #
                    # ``q_no_pe`` feeds the absorption GEMM unrotated, so only
                    # the pe view is rotated: one pe-sized read and write.
                    from paddlefleet.triton_ops import fused_apply_rope_half

                    q_pos_emb = fused_apply_rope_half(
                        q_pos_emb,
                        rotary_pos_emb,
                        self.qk_rope_head_dim,
                        mscale,
                    )
                    # Defer k's rope to ``fused_rope_cat_key`` below, which
                    # rotates and concatenates in one pass.
                    k_rope_fused_with_cat = True
                else:
                    # q_pos_emb: [num_tokens, n, qk_rope_head_dim]
                    q_pos_emb = apply_rotary_pos_emb(
                        q_pos_emb,
                        rotary_pos_emb,
                        rotary_pos_cos,
                        rotary_pos_sin,
                        config=self.config,
                        cu_seqlens=cu_seqlens_q,
                        mscale=mscale,
                        cp_group=self.pg_collection.cp,
                        apply_rope_fusion=bool(self.config.apply_rope_fusion)
                        and not self.mqa_latent,
                    )
                    # k_pos_emb:[num_tokens, 1, qk_rope_head_dim]
                    k_pos_emb = apply_rotary_pos_emb(
                        k_pos_emb,
                        rotary_pos_emb,
                        rotary_pos_cos,
                        rotary_pos_sin,
                        config=self.config,
                        cu_seqlens=cu_seqlens_kv,
                        mscale=mscale,
                        cp_group=self.pg_collection.cp,
                        sp_group=self.pg_collection.tp
                        if self.config.sequence_parallel
                        else None,
                        apply_rope_fusion=bool(self.config.apply_rope_fusion)
                        and not self.mqa_latent,
                    )

                # query: [num_tokens, n, (qk_nope_head_dim + qk_rope_head_dim)]
                k_pe = k_pos_emb
                if self.mqa_latent:
                    # Runtime absorption.  ``kv_b_proj.weight`` is
                    # [kv_lora_rank, n * (qk_nope_head_dim + v_head_dim)]; its
                    # leading qk_nope_head_dim slice per head is W_k_b.  Because
                    #     q_nope . k_nope == (q_nope W_k_b) . kv_compressed
                    # the absorbed scores equal the MHA scores exactly, so the
                    # softmax scale must stay the MHA one (mscale^2/sqrt(256)),
                    # NOT 1/sqrt(576).
                    # ``getattr``: this closure also runs against the
                    # lightweight namespace the RoPE-fusion tests build in place
                    # of a real layer, which carries only the fields it touches.
                    if getattr(self, "mqa_latent_split_kv_b", False):
                        # ``k_b_proj`` holds W_k_b as the grouped-matmul weight
                        # [n, kv_lora_rank, qk_nope_head_dim], stored 2-D; the
                        # reshape is a zero-copy view. The Triton kernel walks
                        # q_no_pe's head dim through strides, so there is no
                        # slice, no einsum and no transpose here.
                        from paddlefleet.triton_ops import (
                            fused_grouped_matmul,
                        )

                        # [num_tokens, n, kv_lora_rank]
                        q_no_pe_absorbed = fused_grouped_matmul(
                            q_no_pe,
                            self.k_b_proj.reshape(
                                [
                                    self.num_attention_heads_per_partition,
                                    self.kv_lora_rank,
                                    self.qk_nope_head_dim,
                                ]
                            ),
                        )
                    else:
                        w_k_b = self.kv_b_proj.weight.reshape(
                            [
                                self.kv_lora_rank,
                                self.num_attention_heads_per_partition,
                                -1,
                            ]
                        )[:, :, : self.qk_nope_head_dim]
                        q_no_pe_absorbed = paddle.einsum(
                            "bshd,lhd->bshl", q_no_pe, w_k_b
                        )
                    # query: [num_tokens, n, kv_lora_rank + qk_rope_head_dim]
                    query = paddle.cat(
                        [q_no_pe_absorbed, q_pos_emb],
                        axis=-1,
                    )
                    # key: [num_tokens, 1, kv_lora_rank + qk_rope_head_dim].
                    # Its leading kv_lora_rank channels are the (absorbed) value.
                    if k_rope_fused_with_cat:
                        from paddlefleet.triton_ops import fused_rope_cat_key

                        # The kernel hands back both of the things the eager
                        # snippet produced: the concatenated key and the rotated
                        # pe on its own. Taking the latter as a slice of the
                        # former instead would be a non-contiguous read plus a
                        # copy, and a non-contiguous return value is rejected by
                        # ``RecomputeWithoutOutput(share_grad_holder=True)``
                        # below, which calls ``share_buffer_to`` on each of them.
                        key, k_pe = fused_rope_cat_key(
                            kv_compressed,
                            k_pos_emb,
                            rotary_pos_emb,
                            self.kv_lora_rank,
                            self.qk_rope_head_dim,
                            mscale,
                        )
                    else:
                        key = paddle.cat(
                            [kv_compressed.unsqueeze(-2), k_pos_emb], axis=-1
                        )
                    value = None
                else:
                    query = paddle.cat([q_no_pe, q_pos_emb], axis=-1)

                    # key: [num_tokens, n, (qk_nope_head_dim + qk_rope_head_dim)]
                    if k_pos_emb.ndim == 4:
                        k_pos_emb = k_pos_emb.expand(
                            -1, -1, self.num_attention_heads_per_partition, -1
                        )
                    else:
                        assert k_pos_emb.ndim == 3
                        k_pos_emb = k_pos_emb.expand(
                            -1, self.num_attention_heads_per_partition, -1
                        )
                    key = paddle.cat([k_no_pe, k_pos_emb], axis=-1)

            # if self.layer_number == 0:
            #     print(f"[DEBUG MLA layer {self.layer_number}] key final shape: {key.shape}, head_dim={key.shape[-1]}", flush=True)

            query = query.contiguous()
            key = key.contiguous()
            if value is not None:
                value = value.contiguous()

            return query, key, value, k_pe

        if self.recompute_qkv_up_porj_and_rope and self.training:
            self._qkv_recompute = RecomputeWithoutOutput()
            query, key, value, k_pos_emb = self._qkv_recompute.recompute(
                qkv_up_proj_and_rope_apply,
                q_compressed,
                kv_compressed,
                k_pos_emb,
                rotary_pos_emb,
                rotary_pos_cos,
                rotary_pos_sin,
                position_ids,
                preserve_rng_state=False,
                share_grad_holder=True,
            )
        else:
            query, key, value, k_pos_emb = qkv_up_proj_and_rope_apply(
                q_compressed,
                kv_compressed,
                k_pos_emb,
                rotary_pos_emb,
                rotary_pos_cos,
                rotary_pos_sin,
                position_ids,
            )

        return query, key, value, q_compressed, kv_compressed, k_pos_emb

    def backward_dw(self) -> NoReturn:
        """Execute weight gradient computation"""
        self._backward_kv_proj()
        self._backward_q_proj()
        self._backward_output_proj()
        # GATE backward?

    def _backward_kv_proj(self):
        """Computes weight gradients of KV projection layers"""
        # ``kv_b_proj`` does not exist in the split-absorption mode; the two
        # parameters that replace it are plain tensors with no delayed dw pass.
        if self.kv_b_proj is not None:
            self.kv_b_proj.backward_dw()
        self.kv_a_proj_with_mqa.backward_dw()

    def _backward_q_proj(self):
        """Computes weight gradients of Q projection layers"""
        if self.q_lora_rank is None:
            self.q_proj.backward_dw()
        else:
            self.q_a_proj.backward_dw()
            self.q_b_proj.backward_dw()

    def _backward_output_proj(self):
        """Computes weight gradients of output projection layer"""
        self.o_proj.backward_dw()


class MQASelfAttention(MLASelfAttention):
    """Multi-Query Attention."""

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: MLASelfAttentionSublayersSpec,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        is_mtp_layer: bool = False,
    ):
        super().__init__(
            config=config,
            sublayers_spec=sublayers_spec,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )

        assert not self.config.apply_rope_fusion, (
            "MQA does not support rope fusion."
        )

        # Use MQA only when HySparse is enabled and this is an SWA layer.
        # Otherwise, use its parent's forward method (MLA).
        self.is_mqa = config.enable_hy_sparse_attention and self.is_swa

        if self.is_mqa:
            # Adjust absorbed kv channels for core attention
            k_channels = self.config.kv_lora_rank + self.qk_rope_head_dim
            v_channels = self.config.kv_lora_rank

            self.core_attention.hidden_size_per_partition = (
                k_channels * self.num_attention_heads_per_partition
            )
            self.core_attention.k_channels = k_channels
            self.core_attention.v_channels = v_channels

            # The MQA path never calls ``self.core_attention(...)`` (it runs the
            # TileLang / cuDNN MQA kernels directly), so the ``softmax_offset``
            # that DotProductAttention registers for SWA layers under
            # ``add_swa_attention_sink_bias`` never participates in the forward
            # and keeps a zero gradient, tripping distributed unused-parameter
            # checks. The real sink logits live in ``swa_attn_sink`` /
            # ``sparse_attn_sink`` below, so drop this redundant parameter.
            # ``del`` goes through ``paddle.nn.Layer.__delattr__`` which removes
            # the entry from ``_parameters``; reset to ``None`` afterwards so any
            # generic ``core_attention.softmax_offset`` lookup still resolves.
            if getattr(self.core_attention, "softmax_offset", None) is not None:
                del self.core_attention.softmax_offset
                self.core_attention.softmax_offset = None

            # Gate for block sparse attention
            if self.gated_attention:
                self.sparse_gate_proj = build_spec_layer(
                    sublayers_spec.gate_proj,
                    self.gate_proj.input_size,
                    self.gate_proj.output_size,
                    config=self.config,
                    init_method=self.config.init_method,
                    gather_output=False,
                    bias=self.config.use_bias,
                    skip_bias_add=False,
                    is_expert=False,
                    tp_comm_buffer_name="mla_gate",
                    tp_group=self.pg_collection.tp,
                )

            # Learnable attention-sink bias for the SWA MQA path. Gated by
            # ``add_swa_attention_sink_bias`` (mirrors the DotProductAttention
            # SWA sink promotion). The two HySparse MQA branches are independent
            # softmaxes (main sliding-window path + block-sparse DSA path), so
            # each gets its own per-head sink logit; both are zero-initialised
            # (a zero sink logit == an off-by-one-style sink at logit 0). When
            # the switch is off, both stay ``None`` and the kernels run their
            # plain sinkless softmax exactly as before.
            self.add_swa_attention_sink_bias = getattr(
                self.config, "add_swa_attention_sink_bias", False
            )
            if self.add_swa_attention_sink_bias:
                num_heads = self.num_attention_heads_per_partition
                self.swa_attn_sink = self.create_parameter(
                    shape=[num_heads],
                    dtype="float32",
                    default_initializer=paddle.nn.initializer.Constant(0.0),
                )
                self.sparse_attn_sink = self.create_parameter(
                    shape=[num_heads],
                    dtype="float32",
                    default_initializer=paddle.nn.initializer.Constant(0.0),
                )
            else:
                self.swa_attn_sink = None
                self.sparse_attn_sink = None

            # VHA postmix for the block-sparse branch. The MQA path has two
            # independent gated branches (sliding-window main + block-sparse),
            # so each gets its own postmix params: the base vha_postmix_U/V
            # serve the main branch, this set serves the sparse branch.
            if self.use_vha_postmix:
                nh = self.num_attention_heads_per_partition  # TP==1 for MQA
                rank = self.vha_postmix_rank
                self.sparse_vha_postmix_U = self.create_parameter(
                    shape=[nh, rank],
                    default_initializer=paddle.nn.initializer.Normal(
                        mean=0.0, std=0.01
                    ),
                )
                self.sparse_vha_postmix_V = self.create_parameter(
                    shape=[nh, rank],
                    default_initializer=paddle.nn.initializer.Constant(0.0),
                )

    def forward(
        self,
        hidden_states,
        attention_mask,
        attn_mask_startend_row_indices: Tensor | None = None,
        key_value_states=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        in_recompute: bool = False,
        position_ids=None,
        shared_kv: list[Tensor] | None = None,
        **kwargs,
    ):
        """Forward pass for multi-latent attention"""
        from paddlefleet.transformer.transformer_layer import TransformerLayer

        _log = TransformerLayer._log_md5

        assert rotary_pos_emb is None, (
            "Rotary position embeddings should not be passed into MQA."
        )
        assert attention_bias is None, (
            "Attention bias should not be passed into MQA."
        )
        assert rotary_pos_cos is None and rotary_pos_sin is None, (
            "MQA does not support Flash Decoding"
        )
        if get_pg_size(self.pg_collection.tp) != 1:
            raise ValueError("MQA does not support tensor parallel.")

        if not self.is_mqa:
            return super().forward(
                hidden_states,
                attention_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                key_value_states=key_value_states,
                packed_seq_params=packed_seq_params,
                in_recompute=in_recompute,
                position_ids=position_ids,
                shared_kv=shared_kv,
                **kwargs,
            )

        # Inference (native KV cache path) state. ``is_decode`` is False during
        # training and during the prefill pass.
        past_key_values = kwargs.get("past_key_values")
        layer_idx = kwargs.get("layer_idx")
        use_cache = kwargs.get("use_cache", False)
        is_decode = _is_incremental_decode(
            past_key_values, layer_idx, use_cache
        )

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention
        # Also get q_compressed for DSA indexer (if enabled)
        query, key, value, q_compressed, kv_compressed, k_pos_emb = (
            self.get_query_key_value_tensors(
                hidden_states,
                key_value_states,
                position_ids,
                packed_seq_params,
            )
        )

        layer_num = getattr(self, "layer_number", -1)
        _log(query, "attn_query", layer_num)
        _log(key, "attn_key", layer_num)
        if value is not None:
            _log(value, "attn_value", layer_num)

        attn_mask_type = self.attn_mask_type
        query = query.contiguous()
        key = key.contiguous()

        if value is not None:
            value = value.contiguous()

        if get_context_parallel_world_size() > 1:
            cp_mode = getattr(
                self.config, "cp_balance_mode", "dualchunk_allgather"
            )
            key = ContextParallelAllGatherOp.apply(key, 1, cp_mode)
            value = ContextParallelAllGatherOp.apply(value, 1, cp_mode)

        # ==================================
        # core attention computation
        # ==================================
        from paddlefleet.tilelang_ops.hysparse import (
            sliding_window_mqa_attention,
        )

        b, s = query.shape[0], query.shape[1]
        block_B = self.config.hy_sparse_block_size
        sm_scale = self.softmax_scale
        window_size = self.config.sliding_window[0]

        # Absorbed-MLA MQA: one shared K/V head with
        # Dk=kv_lora_rank+qk_rope_head_dim and Dv=kv_lora_rank. Squeeze the head
        # axis to the [B, S_kv, D] layout the TileLang MQA kernels expect.
        shared_k = key.squeeze(2).contiguous()
        shared_v = value.squeeze(2).contiguous()

        if use_cache and past_key_values is not None:
            # Own (absorbed) KV cache. DynamicKVCache truncates SWA layers to
            # window_size after every update, so the returned history holds at
            # most window_size + 1 tokens during decode.
            shared_k, shared_v = past_key_values.update(
                shared_k, shared_v, layer_idx
            )
        kv_s = shared_k.shape[1]

        if is_decode:
            if s != 1:
                raise ValueError(
                    "HySparse MQA decode expects a single query token, got "
                    f"seq_len={s}. Chunked prefill is not supported."
                )
            # The cache holds the most recent ``kv_s`` tokens and the query is
            # the last of them, so the sliding window is the trailing
            # ``window_size`` columns in cache-local coordinates.
            bos = max(0, kv_s - window_size)
            window_valid_range = paddle.concat(
                [
                    paddle.full([b, 1, 1], bos, dtype="int32"),
                    paddle.full([b, 1, 1], kv_s, dtype="int32"),
                ],
                axis=-1,
            )
            # The block-sparse branch runs against the (untruncated) shared
            # latent, so its range is built from that length further below.
        else:
            # Windowed valid_range for the sliding-window main path;
            # document-anchored valid_range (no window clamp) for the
            # block-sparse branch so its blocks match the full layer's block
            # scoring / selected indices.
            window_valid_range = build_hysparse_valid_range(
                attn_mask_startend_row_indices,
                kv_s,
                b,
                window_size=window_size,
            )
            doc_valid_range = build_hysparse_valid_range(
                attn_mask_startend_row_indices, kv_s, b
            )
            if get_context_parallel_world_size() > 1:
                cp_mode = getattr(
                    self.config, "cp_balance_mode", "dualchunk_allgather"
                )
                window_valid_range = ContextParallelScatterOp.apply(
                    window_valid_range, 1, cp_mode
                )
                doc_valid_range = ContextParallelScatterOp.apply(
                    doc_valid_range, 1, cp_mode
                )

        # Sliding-window main path over the absorbed MQA dimensions.
        core_attn_out, _ = sliding_window_mqa_attention(
            query,
            shared_k,
            shared_v,
            window_valid_range,
            attn_sink=getattr(self, "swa_attn_sink", None),
            sm_scale=sm_scale,
            block_B=block_B,
        )
        core_attn_out = core_attn_out.reshape(
            [
                b,
                s,
                self.num_attention_heads_per_partition
                * self.config.kv_lora_rank,
            ]
        )

        _log(core_attn_out, "core_attn_out", layer_num)

        # =================
        # Absorb value. [b, sq, num_heads * v_head_dim]
        # =================

        kv_lora_rank = self.config.kv_lora_rank
        num_heads = self.num_attention_heads_per_partition

        v_absorb_weight = self.kv_b_proj.weight.reshape(
            [kv_lora_rank, num_heads, -1]
        )[:, :, self.qk_nope_head_dim :]

        def compute_absorbed_v(core_attn_out):
            core_attn_out = core_attn_out.view(
                *core_attn_out.shape[:-1], num_heads, kv_lora_rank
            )
            core_attn_out = paddle.einsum(
                "bshl,lhv->bshv", core_attn_out, v_absorb_weight
            )
            core_attn_out = core_attn_out.view(
                *core_attn_out.shape[:-2], num_heads * self.v_head_dim
            )
            return core_attn_out

        core_attn_out = compute_absorbed_v(core_attn_out)

        # VHA postmix (main branch): mix in head space before the gate.
        if self.use_vha_postmix:
            if (
                self.recompute_vha_postmix
                and self.training
                and not in_recompute
            ):
                core_attn_out = recompute(
                    self._apply_vha_postmix, core_attn_out
                )
            else:
                core_attn_out = self._apply_vha_postmix(core_attn_out)

        # =================
        # Sparse attention computation
        # =================
        shared_key, shared_block_indices = shared_kv
        if shared_block_indices is None:
            raise ValueError(
                "HySparse MQA layer "
                f"(layer_number={getattr(self, 'layer_number', -1)}) needs "
                "top-k block indices from its full-attention layer, but none "
                "were provided."
            )
        if is_decode:
            # The shared latent is never window-truncated, so its length is the
            # whole KV history; the decode query's document range covers all of
            # it (generation runs one document per batch row).
            if get_context_parallel_world_size() != 1:
                raise ValueError("doc_valid_range is not built for CP > 1")
            doc_valid_range = paddle.concat(
                [
                    paddle.zeros([b, 1, 1], dtype="int32"),
                    paddle.full([b, 1, 1], shared_key.shape[1], dtype="int32"),
                ],
                axis=-1,
            )
        if get_context_parallel_world_size() > 1:
            cp_mode = getattr(
                self.config, "cp_balance_mode", "dualchunk_allgather"
            )
            shared_key = ContextParallelAllGatherOp.apply(
                shared_key, 1, cp_mode
            )

        # Shared compressed KV latent from the full layer, with
        # Dk=kv_lora_rank+qk_rope_head_dim. Squeeze to [B, S_kv, D]; its leading
        # kv_lora_rank channels are the values.
        shared_key_sq = shared_key.squeeze(2).contiguous()

        # Block-sparse gather branch over the absorbed-MQA shared-head layout
        # (value == the leading kv_lora_rank slice of the shared latent).
        use_tl = getattr(
            self.config, "hy_sparse_block_sparse_use_tilelang", False
        )
        if use_tl:
            from paddlefleet.tilelang_ops.hysparse.block_sparse_mqa_tl import (
                block_sparse_mqa_attention_tl,
            )

            sparse_core_attn_out, _ = block_sparse_mqa_attention_tl(
                query,
                shared_key_sq,
                shared_block_indices,
                doc_valid_range,
                sm_scale=sm_scale,
                block_B=block_B,
                kv_lora_rank=self.config.kv_lora_rank,
                attn_sink=getattr(self, "sparse_attn_sink", None),
            )
        else:
            from paddlefleet.cudnn_ops import (
                block_sparse_mqa_attention_dsa,
                is_dsa_available,
            )

            if not is_dsa_available():
                raise RuntimeError(
                    "HySparse block-sparse attention requires the DSA backend "
                    "(FlashMLA sparse fwd + cuDNN DSA bwd), unavailable here: "
                    "it needs SM100 + FlashMLA + the cuDNN frontend."
                )
            sparse_core_attn_out, _ = block_sparse_mqa_attention_dsa(
                query,
                shared_key_sq,
                shared_block_indices,
                doc_valid_range,
                sm_scale=sm_scale,
                block_B=block_B,
                kv_lora_rank=self.config.kv_lora_rank,
                attn_sink=getattr(self, "sparse_attn_sink", None),
            )
        sparse_core_attn_out = sparse_core_attn_out.reshape(
            [
                b,
                s,
                self.num_attention_heads_per_partition
                * self.config.kv_lora_rank,
            ]
        )

        sparse_core_attn_out = compute_absorbed_v(sparse_core_attn_out)

        # VHA postmix (sparse branch): own param set, mix before the sparse gate.
        if self.use_vha_postmix:
            U, V = self.sparse_vha_postmix_U, self.sparse_vha_postmix_V
            if (
                self.recompute_vha_postmix
                and self.training
                and not in_recompute
            ):
                sparse_core_attn_out = recompute(
                    self._apply_vha_postmix, sparse_core_attn_out, U, V
                )
            else:
                sparse_core_attn_out = self._apply_vha_postmix(
                    sparse_core_attn_out, U, V
                )

        # =================
        # Output. [b, sq, h]
        # =================
        # Apply gated attention
        if self.gated_attention:
            # Gate input source: q_compressed (post q_a_layernorm, dim=q_lora_rank) when
            # gated_attn_use_q_lora is set, otherwise hidden_states.
            gate_source = (
                q_compressed if self.gated_attn_use_q_lora else hidden_states
            )
            core_attn_out = self._gate(gate_source, core_attn_out)

        # Add sparse attention output
        if self.gated_attention:
            gate, _ = self.sparse_gate_proj(gate_source)
            sparse_core_attn_out = (
                sparse_core_attn_out * paddle.nn.functional.sigmoid(gate)
            )
        core_attn_out += sparse_core_attn_out

        output, bias = self.o_proj(core_attn_out)

        _log(output, "attn_o_proj_out", layer_num)

        return output, bias

    def get_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        if not self.is_mqa:
            return super().get_query_key_value_tensors(
                hidden_states,
                key_value_states=key_value_states,
                position_ids=position_ids,
                packed_seq_params=packed_seq_params,
            )

        # b = batch size, s = sequence length, h = hidden size, n = num attention heads
        # Attention heads [b, s, n*h]
        assert hidden_states.ndim == 3, (
            f"hidden_states should be 3D, [b, s, n*h], got {hidden_states.ndim}D"
        )

        # =========================================
        # Prepare RoPE and seqlen related params
        # =========================================
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            hidden_states, self.config, packed_seq_params
        )

        # rotary_pos_emb: [1, s, 1, pe_dim]
        mscale = 1.0
        rotary_pos_cos = None
        rotary_pos_sin = None

        # Explicit raises (not assert): production forward path, asserts are
        # stripped under `python -O` and an unsupported rope_type /
        # packed_seq_params would then silently feed the RoPE/TileLang/DSA
        # kernels instead of failing here.
        if self.config.rope_type != "rope":
            raise ValueError(
                "MQA only supports rope_type 'rope', got "
                f"{self.config.rope_type}"
            )
        if packed_seq_params is not None:
            raise ValueError("MQA doesn't support packed_seq_params")

        rotary_pos_emb = self.rotary_pos_emb(
            rotary_seq_len,
            position_ids=None if self.training else position_ids,
        )
        if get_context_parallel_world_size() > 1:
            cp_mode = getattr(
                self.config, "cp_balance_mode", "dualchunk_allgather"
            )
            rotary_pos_emb = ContextParallelScatterOp.apply(
                rotary_pos_emb, 1, cp_mode
            )
            if rotary_pos_emb.shape[1] != hidden_states.shape[1]:
                raise ValueError(
                    "MQA context-parallel RoPE scatter produced an invalid "
                    f"sequence length: rotary={rotary_pos_emb.shape}, "
                    f"hidden_states={hidden_states.shape}."
                )

        # =========================================
        # QKV down projection and layernorm
        # =========================================
        if self.config.q_lora_rank is not None:
            q_compressed, _ = self.q_a_proj(hidden_states)
        else:
            q_compressed = hidden_states

        # kv_combined: [b, s, (kv_lora_rank + qk_rope_head_dim)]
        kv_combined, _ = self.kv_a_proj_with_mqa(hidden_states)

        # kv_compressed: [b, s, kv_lora_rank], k_pos_emb: [b, s, qk_rope_head_dim]
        kv_compressed, k_pos_emb = paddle.split(
            kv_combined,
            [self.config.kv_lora_rank, self.qk_rope_head_dim],
            axis=-1,
        )

        # =========================================
        # Apply norm
        # =========================================

        if self.config.q_lora_rank is not None:
            # q_compressed: [num_tokens, q_lora_rank]
            q_compressed = self.q_a_layernorm(q_compressed)

        kv_compressed = self.kv_a_layernorm(kv_compressed)

        # === MD5 probes for MLA intermediate values ===
        from paddlefleet.transformer.transformer_layer import TransformerLayer

        _log = TransformerLayer._log_md5
        _log(q_compressed, "mla_q_compressed_normed", self.layer_number)
        _log(kv_compressed, "mla_kv_compressed_normed", self.layer_number)
        _log(k_pos_emb, "mla_k_pos_emb_raw", self.layer_number)

        # =========================================
        # QKV up projection and RoPE apply
        # =========================================

        def qkv_up_proj_and_rope_apply(
            q_compressed,
            kv_compressed,
            k_pos_emb,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            position_ids=None,
        ):
            """
            Apply the up projection and RoPE to the query and key.
            When sequence packing enabled, the input tensors adopt a packed shape of [t, ...];
            otherwise, they maintain the unpacked shape [b, s, ...]. In subsequent code comments,
            we uniformly use [num_tokens, ...] to denote [b, s, ...] or [t, ...] for two cases.
            """
            if self.config.q_lora_rank is not None:
                # q_compressed: [num_tokens, q_lora_rank]
                # q: [num_tokens, n * (qk_nope_head_dim + qk_rope_head_dim)]
                q, _ = self.q_b_proj(q_compressed)
            else:
                # q_compressed: [num_tokens, hidden_size]
                # q: [num_tokens, n * (qk_nope_head_dim + qk_rope_head_dim)]
                q, _ = self.q_proj(q_compressed)

            # q: [num_tokens, n, q_head_dim]
            q = q.view(
                *q.size()[:-1],
                self.num_attention_heads_per_partition,
                self.q_head_dim,
            )

            kv_lora_rank = self.config.kv_lora_rank
            num_heads = self.num_attention_heads_per_partition

            q_no_pe = q[..., : self.qk_nope_head_dim]
            q_pos_emb = q[..., self.qk_nope_head_dim :]

            q_absorb_weight = self.kv_b_proj.weight.reshape(
                [kv_lora_rank, num_heads, -1]
            )[:, :, : self.qk_nope_head_dim]
            q_nope_absorbed = paddle.einsum(
                "bshd,lhd->bshl", q_no_pe, q_absorb_weight
            )

            q_pos_emb = apply_rotary_pos_emb(
                q_pos_emb,
                rotary_pos_emb,
                rotary_pos_cos,
                rotary_pos_sin,
                config=self.config,
                mscale=mscale,
            )
            k_pos_emb = apply_rotary_pos_emb(
                k_pos_emb.unsqueeze(-2),
                rotary_pos_emb,
                rotary_pos_cos,
                rotary_pos_sin,
                config=self.config,
                mscale=mscale,
            )

            kv_compressed = kv_compressed.unsqueeze(-2)

            query = paddle.concat([q_nope_absorbed, q_pos_emb], axis=-1)
            key = paddle.concat([kv_compressed, k_pos_emb], axis=-1)
            value = kv_compressed

            return query, key, value, k_pos_emb

        query, key, value, k_pos_emb = qkv_up_proj_and_rope_apply(
            q_compressed,
            kv_compressed,
            k_pos_emb,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            position_ids,
        )

        return query, key, value, q_compressed, kv_compressed, k_pos_emb
