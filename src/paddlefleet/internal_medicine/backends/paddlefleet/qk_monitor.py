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

"""
QK Stats Monitor for PaddleFleet.

Uses paddle hooks on core_attention to capture Q, K tensors and compute
attention statistics via Triton kernel on GPU.
"""

import logging

import paddle
from paddle import nn

from .base import PaddleProbe
from .layer_discovery import (
    MLA_RATIO,
    MQA_RATIO,
    get_decoder_layers,
    is_kda_layer,
    iter_monitor_layers,
)

logger = logging.getLogger(__name__)


class _MethodPatch:
    """Handle for a monkey-patched bound method, shaped like a paddle hook."""

    def __init__(self, module, name: str, original):
        self._module = module
        self._name = name
        self._original = original

    def remove(self) -> None:
        setattr(self._module, self._name, self._original)


def compute_sink_head_classification(
    sink_per_head: paddle.Tensor, threshold: float = 0.3
) -> dict:
    """Classify attention heads as sink vs non-sink.

    Mirrors the megatron-side ``sink_head_metrics.compute_sink_head_classification``
    using paddle ops. ``sink_per_head`` is a 1-D tensor of length num_heads
    holding the mean attention weight on token-0 per head (already averaged
    across batch).

    Returns a dict with three GPU 0-dim tensors:
        sink_head_ratio  — fraction of heads with sink weight > threshold
        sink_head_max    — max sink weight across heads
        sink_nonsink_gap — mean(sink) - mean(non-sink); 0 if no sinks; mean if all sinks
    """
    num_heads = int(sink_per_head.numel())
    if num_heads == 0:
        zero = sink_per_head.sum()
        return {
            "sink_head_ratio": zero,
            "sink_head_max": zero,
            "sink_nonsink_gap": zero,
        }

    is_sink = sink_per_head > threshold
    is_sink_f = is_sink.astype("float32")
    is_nonsink_f = paddle.logical_not(is_sink).astype("float32")
    sink_count = is_sink_f.sum()
    nonsink_count = is_nonsink_f.sum()
    sink_head_ratio = sink_count / float(num_heads)
    sink_head_max = sink_per_head.max()

    zero = sink_per_head.sum() * 0.0
    sink_sum = (sink_per_head * is_sink_f).sum()
    nonsink_sum = (sink_per_head * is_nonsink_f).sum()
    sink_mean = sink_sum / sink_count.clip(min=1.0)
    nonsink_mean = nonsink_sum / nonsink_count.clip(min=1.0)
    gap = paddle.where(
        sink_count == 0,
        zero,
        paddle.where(
            nonsink_count == 0, sink_per_head.mean(), sink_mean - nonsink_mean
        ),
    )

    return {
        "sink_head_ratio": sink_head_ratio,
        "sink_head_max": sink_head_max,
        "sink_nonsink_gap": gap,
    }


_triton_driver_patched = False


def _ensure_triton_driver():
    """Patch triton's NVIDIA CudaDriver to use paddle CUDA instead of torch.

    Triton 3.x's CudaDriver delegates to torch.cuda for device/stream management.
    In pfleet venvs, torch's CUDA may fail (driver version mismatch) while paddle's
    CUDA works fine. This patches the driver instance to use paddle equivalents.
    """
    global _triton_driver_patched
    if _triton_driver_patched:
        return
    try:
        from triton.backends.nvidia.driver import CudaDriver
    except ImportError:
        raise ImportError(
            "triton is required for GPU QK stats computation"
        ) from None

    CudaDriver.is_active = staticmethod(lambda: True)

    _orig_init = CudaDriver.__init__

    def _patched_init(self):
        _orig_init(self)
        self.get_current_device = lambda: int(
            paddle.framework._current_expected_place().get_device_id()
        )
        self.set_current_device = lambda dev: paddle.device.set_device(
            f"gpu:{dev}"
        )
        self.get_device_capability = (
            lambda dev=None: paddle.device.cuda.get_device_capability(dev)
        )
        self.get_current_stream = lambda dev: paddle.device.current_stream(
            f"gpu:{dev}"
        ).stream_base.cuda_stream

    CudaDriver.__init__ = _patched_init
    _triton_driver_patched = True


def _compute_qk_stats_triton(
    q: paddle.Tensor,
    k: paddle.Tensor,
    causal: bool = True,
    heads_per_group: int = 1,
    row_stride: int = 1,
    q_row_offset: int = 0,
    attn_sink: paddle.Tensor | None = None,
    softmax_scale: float | None = None,
) -> dict:
    """Triton kernel path for paddle tensors.

    q: [B, H, S_q, D] (query heads). k: [B, H_kv, S_k, D] (KV heads, NOT expanded).
    ``heads_per_group = H // H_kv``; the kernel maps each query head to its KV
    head internally so we never materialize a repeat_interleave of k.
    ``row_stride`` subsamples query rows (see kernel docstring).

    Under Context-Parallel (CP > 1), Q may stay local (``S_q = S/CP``) while
    K has been all_gather'd to full seq (``S_k = S``). ``q_row_offset`` is the
    starting global row index of the local Q shard, used only by the kernel's
    causal masking so that local row m compares against key col ``m + offset``
    rather than the ambiguous local ``m``.
    """
    _ensure_triton_driver()
    from ...core.triton_qk_kernel import qk_stats_partial_kernel

    batch, num_heads, seq_len_q, head_dim = q.shape
    seq_len_k = k.shape[2]
    scale = (
        1.0 / (head_dim**0.5) if softmax_scale is None else float(softmax_scale)
    )

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64 if head_dim <= 64 else 128

    # Stage-1 grid: parallelize over (B*H, num_M_blocks_over_Q) instead of (B*H,) only.
    m_block_span = BLOCK_M * row_stride
    num_m_blocks = (seq_len_q + m_block_span - 1) // m_block_span

    partial_shape = [batch, num_heads, num_m_blocks]
    partial_max = paddle.empty(partial_shape, dtype="float32")
    partial_sum_logit = paddle.empty(partial_shape, dtype="float32")
    partial_sum_row_mean = paddle.empty(partial_shape, dtype="float32")
    partial_count = paddle.empty(partial_shape, dtype="float32")
    partial_sum_entropy = paddle.empty(partial_shape, dtype="float32")
    partial_sum_sink = paddle.empty(partial_shape, dtype="float32")
    partial_valid_rows = paddle.empty(partial_shape, dtype="float32")

    grid = (batch * num_heads, num_m_blocks)

    # A None pointer is not a valid kernel argument, so pass a dummy buffer and
    # gate the fold on HAS_SINK (same pattern as the sparse path).
    sink_present = attn_sink is not None
    sink_buf = (
        attn_sink
        if sink_present
        else paddle.zeros([num_heads], dtype="float32")
    )

    qk_stats_partial_kernel[grid](
        q,
        k,
        sink_buf,
        partial_max,
        partial_sum_logit,
        partial_sum_row_mean,
        partial_count,
        partial_sum_entropy,
        partial_sum_sink,
        partial_valid_rows,
        batch,
        num_heads,
        seq_len_q,
        seq_len_k,
        head_dim,
        heads_per_group,
        num_m_blocks,
        q_row_offset,
        q.strides[0],
        q.strides[1],
        q.strides[2],
        q.strides[3],
        k.strides[0],
        k.strides[1],
        k.strides[2],
        k.strides[3],
        partial_max.strides[0],
        partial_max.strides[1],
        partial_max.strides[2],
        scale=scale,
        apply_causal_mask=causal,
        HAS_SINK=sink_present,
        ROW_STRIDE=row_stride,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # reduce: pure paddle GPU ops, deterministic reduction order, no D2H sync.
    max_logits = partial_max.max(axis=-1)  # [B, H]
    total_rows = partial_valid_rows.sum(axis=-1).clip(min=1.0)
    mean_logits = partial_sum_row_mean.sum(axis=-1) / total_rows
    entropy = partial_sum_entropy.sum(axis=-1) / total_rows
    sink = partial_sum_sink.sum(axis=-1) / total_rows

    return {
        "max_per_head": max_logits,
        "mean_per_head": mean_logits,
        "entropy_per_head": entropy,
        "sink_per_head": sink,
        "max_global": max_logits.max(),
        "mean_global": mean_logits.mean(),
        "entropy_global": entropy.mean(),
        "sink_global": sink.mean(),
    }


def compute_qk_stats_paddle(
    q: paddle.Tensor,
    k: paddle.Tensor,
    causal: bool = True,
    row_stride: int = 1,
    q_row_offset: int = 0,
    attn_sink: paddle.Tensor | None = None,
    softmax_scale: float | None = None,
) -> dict:
    """Compute QK stats via Triton kernel.

    Args:
        q: [B, S_q, H, D] — PaddleFleet core_attention input format
        k: [B, S_k, H_kv, D] — KV heads (may be fewer than query heads for GQA)
        row_stride: subsample every ``row_stride``-th query row (1 == exact)
        q_row_offset: global row-index start for this Q shard. Zero unless the
            caller is running with Context-Parallel and Q is kept local. Used
            only by the kernel's causal-mask logic.
        attn_sink: per-head sink logit ``[H]`` (float32), folded into the softmax
            as an extra key-less column. Pass the layer's
            ``core_attention.softmax_offset`` whenever it exists — without it the
            reported distribution is renormalised over the real keys only and does
            not match what the model computes.
        softmax_scale: runtime scale applied to ``QK^T`` by the attention layer.
            ``None`` preserves the standard ``1 / sqrt(head_dim)`` default.

    The [B, S, H, D] -> [B, H, S, D] reordering is expressed via strides
    (no contiguous copy); the triton kernel reads strided memory directly.
    GQA grouping is handled inside the kernel, so k is NOT repeat-expanded.
    """
    if not q.place.is_gpu_place():
        raise RuntimeError(
            "[PaddleQKMonitor] QK stats requires GPU (triton kernel)"
        )

    num_q_heads = q.shape[2]
    num_k_heads = k.shape[2]
    heads_per_group = num_q_heads // num_k_heads if num_k_heads > 0 else 1

    # [B, S, H, D] -> [B, H, S, D] without copying (stride permute only).
    q = q.transpose([0, 2, 1, 3])
    k = k.transpose([0, 2, 1, 3])

    return _compute_qk_stats_triton(
        q,
        k,
        causal=causal,
        heads_per_group=heads_per_group,
        row_stride=row_stride,
        q_row_offset=q_row_offset,
        attn_sink=attn_sink,
        softmax_scale=softmax_scale,
    )


def _segment_bounds(
    idx: paddle.Tensor, base: int
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Reduce a per-row key-index set to inclusive ``[lo, hi]`` bounds.

    ``idx`` is ``[B, S, K]`` with ``-1`` marking unused slots (the convention
    used by PaddleFleet's window / compressed index helpers). Rows whose set is
    empty get ``lo = base, hi = base - 1`` so the kernel skips them without
    dragging the block's scalar loop bounds outside the segment.
    """
    valid = idx >= 0
    far = paddle.full_like(idx, 1 << 30)
    lo = paddle.where(valid, idx, far).min(axis=-1)
    hi = paddle.where(valid, idx, paddle.full_like(idx, -1)).max(axis=-1)
    empty = hi < lo
    lo = paddle.where(empty, paddle.full_like(lo, base), lo)
    hi = paddle.where(empty, paddle.full_like(hi, base - 1), hi)
    return lo.astype("int32"), hi.astype("int32")


def sparse_bounds_from_topk(
    topk_idxs: paddle.Tensor, seq_len_orig: int
) -> dict:
    """Split the model's real key indices into window / compressed segments.

    ``topk_idxs`` is what ``CompressedSparseAttention`` feeds to its sparse
    attention kernel: ``[B, S, K]`` indices into ``kv_full = concat([kv,
    compressed_kv])``, with ``-1`` for unused slots. Indices below
    ``seq_len_orig`` address original KV (the sliding window), the rest address
    compressed KV. Splitting by value rather than by column width keeps this
    correct regardless of how the model orders the concatenation.

    Both segments are contiguous whenever the compressed keys are taken whole
    (``compress_ratio`` 128, or any ratio under ``csa_dense_mode``). A learned
    indexer selects a sparse subset instead, and the ranges then cover a
    superset of the real keys — layers with an indexer are warned about once at
    hook registration.

    ``sink_col`` is the row's earliest reachable key — the first compressed
    block when there is one (it summarises the start of the sequence/document),
    otherwise the oldest in-window position. For dense causal attention this
    degenerates to column 0, matching the legacy ``sink`` metric.
    """
    orig = paddle.where(
        topk_idxs < seq_len_orig, topk_idxs, paddle.full_like(topk_idxs, -1)
    )
    comp = paddle.where(
        topk_idxs >= seq_len_orig, topk_idxs, paddle.full_like(topk_idxs, -1)
    )

    win_lo, win_hi = _segment_bounds(orig, 0)
    cmp_lo, cmp_hi = _segment_bounds(comp, seq_len_orig)

    has_comp = cmp_hi >= cmp_lo
    sink_col = paddle.where(has_comp, cmp_lo, win_lo)

    return {
        "win_lo": win_lo,
        "win_hi": win_hi,
        "cmp_lo": cmp_lo,
        "cmp_hi": cmp_hi,
        "sink_col": sink_col.astype("int32"),
    }


def _compute_qk_stats_sparse_triton(
    q: paddle.Tensor,
    k: paddle.Tensor,
    bounds: dict,
    attn_sink: paddle.Tensor | None,
    scale: float,
    heads_per_group: int = 1,
    row_stride: int = 1,
) -> dict:
    """Triton path for two-segment sparse attention stats.

    ``q``: ``[B, H, S_q, D]``, ``k``: ``[B, H_kv, S_k, D]`` (``S_k`` covers
    ``kv_full``). Reuses the same partial-buffer reduction as the dense path so
    both produce identical metric semantics.
    """
    _ensure_triton_driver()
    from ...core.triton_qk_kernel import qk_stats_sparse_partial_kernel

    batch, num_heads, seq_len_q, head_dim = q.shape
    seq_len_k = k.shape[2]

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64 if head_dim <= 64 else 128

    m_block_span = BLOCK_M * row_stride
    num_m_blocks = (seq_len_q + m_block_span - 1) // m_block_span

    partial_shape = [batch, num_heads, num_m_blocks]
    partial_max = paddle.empty(partial_shape, dtype="float32")
    partial_sum_logit = paddle.empty(partial_shape, dtype="float32")
    partial_sum_row_mean = paddle.empty(partial_shape, dtype="float32")
    partial_count = paddle.empty(partial_shape, dtype="float32")
    partial_sum_entropy = paddle.empty(partial_shape, dtype="float32")
    partial_sum_sink = paddle.empty(partial_shape, dtype="float32")
    partial_valid_rows = paddle.empty(partial_shape, dtype="float32")

    win_lo = bounds["win_lo"]
    sink_present = attn_sink is not None
    sink_buf = (
        attn_sink
        if sink_present
        else paddle.zeros([num_heads], dtype="float32")
    )

    grid = (batch * num_heads, num_m_blocks)
    qk_stats_sparse_partial_kernel[grid](
        q,
        k,
        win_lo,
        bounds["win_hi"],
        bounds["cmp_lo"],
        bounds["cmp_hi"],
        bounds["sink_col"],
        sink_buf,
        partial_max,
        partial_sum_logit,
        partial_sum_row_mean,
        partial_count,
        partial_sum_entropy,
        partial_sum_sink,
        partial_valid_rows,
        batch,
        num_heads,
        seq_len_q,
        seq_len_k,
        head_dim,
        heads_per_group,
        num_m_blocks,
        q.strides[0],
        q.strides[1],
        q.strides[2],
        q.strides[3],
        k.strides[0],
        k.strides[1],
        k.strides[2],
        k.strides[3],
        win_lo.strides[0],
        win_lo.strides[1],
        partial_max.strides[0],
        partial_max.strides[1],
        partial_max.strides[2],
        scale=scale,
        HAS_SINK=sink_present,
        ROW_STRIDE=row_stride,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    max_logits = partial_max.max(axis=-1)
    total_rows = partial_valid_rows.sum(axis=-1).clip(min=1.0)
    mean_logits = partial_sum_row_mean.sum(axis=-1) / total_rows
    entropy = partial_sum_entropy.sum(axis=-1) / total_rows
    sink = partial_sum_sink.sum(axis=-1) / total_rows

    return {
        "max_per_head": max_logits,
        "mean_per_head": mean_logits,
        "entropy_per_head": entropy,
        "sink_per_head": sink,
        "max_global": max_logits.max(),
        "mean_global": mean_logits.mean(),
        "entropy_global": entropy.mean(),
        "sink_global": sink.mean(),
    }


def compute_qk_stats_sparse_paddle(
    query: paddle.Tensor,
    kv_full: paddle.Tensor,
    topk_idxs: paddle.Tensor,
    softmax_scale: float,
    attn_sink: paddle.Tensor | None = None,
    row_stride: int = 1,
) -> dict:
    """QK stats over the exact key set of a window + compressed-KV layer.

    Args:
        query: ``[B, S, H, D]`` as handed to the model's sparse attention.
        kv_full: ``[B, S + n_compressed, D]`` single-head (MQA) keys, i.e.
            original KV concatenated with compressed KV.
        topk_idxs: ``[B, S, K]`` real key indices into ``kv_full``.
        softmax_scale: the scale the layer itself uses (``v_head_dim ** -0.5``),
            not ``1 / sqrt(head_dim)`` of the monitored tensor.
        attn_sink: learned per-head sink logit ``[H]``, folded into the softmax
            denominator. ``None`` to omit.
    """
    if not query.place.is_gpu_place():
        raise RuntimeError(
            "[PaddleQKMonitor] QK stats requires GPU (triton kernel)"
        )

    seq_len_orig = query.shape[1]
    if kv_full.ndim == 3:
        kv_full = kv_full.unsqueeze(2)  # [B, S_k, 1, D]

    num_q_heads = query.shape[2]
    num_k_heads = kv_full.shape[2]
    heads_per_group = num_q_heads // num_k_heads if num_k_heads > 0 else 1

    bounds = sparse_bounds_from_topk(topk_idxs, seq_len_orig)

    q = query.transpose([0, 2, 1, 3])
    k = kv_full.transpose([0, 2, 1, 3])

    stats = _compute_qk_stats_sparse_triton(
        q,
        k,
        bounds,
        attn_sink,
        scale=softmax_scale,
        heads_per_group=heads_per_group,
        row_stride=row_stride,
    )
    return stats


class PaddleQKStatsMonitor(PaddleProbe):
    METRIC_PREFIX = "qk_stats"
    MAX_AGGREGATED = {
        "q_norm_max",
        "k_norm_max",
        "v_norm_max",
        "max",
        "entropy_max",
        "sink_head_max",
    }
    MIN_AGGREGATED = {"entropy_min"}

    def __init__(
        self,
        causal=True,
        log_per_layer=True,
        log_global=True,
        monitor_interval=1,
        verbose=False,
        sink_head_threshold: float = 0.3,
        row_stride: int = 1,
        exclude_families=None,
    ):
        super().__init__(
            exclude_families=exclude_families,
            log_per_layer=log_per_layer,
            log_global=log_global,
            monitor_interval=monitor_interval,
            verbose=verbose,
        )
        self.causal = causal
        self.tp_size = 1
        self.cp_size = 1
        self.cp_rank = 0
        self.cp_group = None
        self.cp_balance_mode = "dualchunk_allgather"
        self.pp_rank = 0
        self.sink_head_threshold = sink_head_threshold
        if int(row_stride) < 1:
            raise ValueError(
                f"[PaddleQKMonitor] row_stride must be >= 1, got {row_stride}"
            )
        self.row_stride = int(row_stride)
        # Warn at most once if a runtime seq_len ends up smaller than row_stride.
        self._warned_row_stride_gt_seqlen = False
        # Warn at most once if the CP-gather path fails at runtime (fallback:
        # skip this hook invocation entirely to preserve correctness).
        self._warned_cp_gather_failed = False

    def register_hooks(self, model: nn.Layer):
        try:
            from paddlefleet.process_groups_config import ProcessGroupCollection
            from paddlefleet.utils import get_pg_size

            pg = ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=["tp"]
            )
            self.tp_size = get_pg_size(pg.tp)
        except Exception:
            pass

        try:
            from paddlefleet.process_groups_config import ProcessGroupCollection
            from paddlefleet.utils import get_pg_size

            pg = ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=["cp"]
            )
            self.cp_group = pg.cp
            self.cp_size = get_pg_size(pg.cp)
            if self.cp_group is not None and self.cp_size > 1:
                # ``paddle.distributed.get_rank(group=...)`` returns the rank
                # within the given group, which is the seq-shard index we need.
                import paddle.distributed as dist

                self.cp_rank = dist.get_rank(group=self.cp_group)
        except Exception:
            pass

        try:
            from paddlefleet.parallel_state import (
                get_pipeline_model_parallel_rank,
            )

            self.pp_rank = get_pipeline_model_parallel_rank()
        except Exception:
            pass

        attention_layers = self._find_attention_layers(model)
        if not attention_layers:
            logger.warning("[PaddleQKMonitor] No attention layers found!")
            return

        if self.verbose:
            logger.info(
                f"[PaddleQKMonitor] Found {len(attention_layers)} attention layers. TP={self.tp_size} CP={self.cp_size}"
            )

        if self.cp_size > 1:
            config = getattr(attention_layers[0][1], "config", None)
            if config is None:
                config = getattr(
                    getattr(attention_layers[0][1], "core_attention", None),
                    "config",
                    None,
                )
            if config is None:
                config = getattr(attention_layers[0][2].layer, "config", None)
            self.cp_balance_mode = getattr(
                config, "cp_balance_mode", "dualchunk_allgather"
            )
            logger.warning(
                "[PaddleQKMonitor] CP=%d detected with mode=%s. Reconstructing "
                "full-sequence Q/K with the matching gather so causal, entropy "
                "and sink stats use global token order.",
                self.cp_size,
                self.cp_balance_mode,
            )

        for layer_idx, attn_module, item in attention_layers:
            if not self._is_sparse_layer(item) and hasattr(
                attn_module, "core_attention"
            ):
                for name in ("q", "k", "v"):
                    self.declare_layer_metric(
                        layer_idx, f"{name}_norm_mean", attn_type=item.attn_type
                    )
                    self.declare_layer_metric(
                        layer_idx, f"{name}_norm_max", attn_type=item.attn_type
                    )
            for m in (
                "max",
                "mean",
                "entropy_avg",
                "sink",
                "entropy_min",
                "entropy_max",
                "entropy_std",
                "sink_head_ratio",
                "sink_head_max",
                "sink_nonsink_gap",
            ):
                self.declare_layer_metric(
                    layer_idx, m, attn_type=item.attn_type
                )
            if (
                self._is_sparse_layer(item)
                or self._learnable_sink(attn_module) is not None
            ):
                self.declare_layer_metric(
                    layer_idx, "attn_sink_logit", attn_type=item.attn_type
                )

        self.allocate_buffers()

        for layer_idx, attn_module, item in attention_layers:
            is_sparse = self._is_sparse_layer(item)
            if is_sparse:
                patch = self._patch_sparse_attn(layer_idx, attn_module, item)
                if patch is not None:
                    self.hooks.append(patch)
                    continue
            if hasattr(attn_module, "core_attention"):
                hook = attn_module.core_attention.register_forward_pre_hook(
                    self._make_compute_hook(
                        layer_idx,
                        item.attn_type,
                        record_qkv_norms=not is_sparse,
                    )
                )
                self.hooks.append(hook)

        logger.info(f"[PaddleQKMonitor] Registered {len(self.hooks)} hooks.")

    def _is_sparse_layer(self, item) -> bool:
        """True for layers whose core attention is window + compressed KV.

        That is any ratio-described layer other than ``-2`` (MLA) and ``-1``
        (full-causal MQA). Those layers attend to a sliding window plus
        compressed KV under one softmax, which the dense causal kernel cannot
        describe.
        """
        ratio = item.compress_ratio
        if ratio is None or ratio in (MLA_RATIO, MQA_RATIO):
            return False
        core = getattr(item.layer, "self_attn", None) or getattr(
            item.layer, "self_attention", None
        )
        core = getattr(core, "core_attention", None)
        return hasattr(core, "compressed_sparse_attn")

    @staticmethod
    def _learnable_sink(attn_module):
        """Return a full-attention learnable sink logit tensor, or ``None``.

        Non-sparse layers get a per-head sink bias when
        ``add_full_attention_sink_bias`` / ``add_swa_attention_sink_bias``
        promotes ``softmax_type`` to ``"learnable"``: PaddleFleet keeps it on the
        core attention as ``softmax_offset`` and passes it to the kernel as
        ``learnable_sink``. Sparse layers instead receive their own ``attn_sink``
        as a ``compressed_sparse_attn`` argument, recorded in
        ``_patch_sparse_attn`` — so both layer kinds report ``attn_sink_logit``,
        from different sources.

        The ``"off-by-one"`` softmax also sets ``softmax_offset``, but to a frozen
        zero buffer; ``stop_gradient`` separates it from the learned parameter so
        a constant-0 series is never logged.
        """
        core = getattr(attn_module, "core_attention", None)
        offset = getattr(core, "softmax_offset", None)
        if offset is None or offset.stop_gradient:
            return None
        return offset

    def _patch_sparse_attn(self, layer_idx: int, attn_module, item):
        """Wrap ``CompressedSparseAttention.compressed_sparse_attn``.

        That call receives everything needed for an exact reproduction of the
        layer's attention distribution — the post-RoPE query, ``kv_full``, the
        real ``topk_idxs``, the learned ``attn_sink`` and the layer's own
        ``softmax_scale`` — so nothing has to be re-derived and the
        document-mask / indexer variants are covered by construction.
        """
        core = getattr(attn_module, "core_attention", None)
        original = getattr(core, "compressed_sparse_attn", None)
        if original is None:
            return None
        if self.cp_size > 1:
            logger.warning(
                "[PaddleQKMonitor] CP=%d with sparse (CSA/HCA) layers is not "
                "supported; skipping qk_stats for layer %d.",
                self.cp_size,
                layer_idx,
            )
            return None
        if item.has_indexer:
            # A learned indexer picks a sparse subset of the compressed blocks,
            # so the two [lo, hi] ranges cover a superset of the real keys.
            logger.warning(
                "[PaddleQKMonitor] layer %d runs a learned indexer; its qk_stats "
                "cover a superset of the selected compressed keys.",
                layer_idx,
            )

        monitor = self
        attn_type = item.attn_type

        def wrapped(
            query, kv_full, attn_sink, topk_idxs, softmax_scale, *args, **kwargs
        ):
            out = original(
                query,
                kv_full,
                attn_sink,
                topk_idxs,
                softmax_scale,
                *args,
                **kwargs,
            )
            if core.training and monitor._should_monitor():
                monitor._record_sparse_stats(
                    layer_idx,
                    attn_type,
                    query,
                    kv_full,
                    attn_sink,
                    topk_idxs,
                    softmax_scale,
                )
            return out

        core.compressed_sparse_attn = wrapped
        return _MethodPatch(core, "compressed_sparse_attn", original)

    def _record_sparse_stats(
        self,
        layer_idx,
        attn_type,
        query,
        kv_full,
        attn_sink,
        topk_idxs,
        softmax_scale,
    ) -> None:
        try:
            with paddle.no_grad():
                stats = compute_qk_stats_sparse_paddle(
                    query.detach(),
                    kv_full.detach(),
                    topk_idxs.detach(),
                    float(softmax_scale),
                    attn_sink=None
                    if attn_sink is None
                    else attn_sink.detach().astype("float32"),
                    row_stride=self.row_stride,
                )
                self._record_common_stats(layer_idx, attn_type, stats)
                if attn_sink is not None:
                    self.record_layer_metric(
                        layer_idx,
                        "attn_sink_logit",
                        attn_sink.detach().astype("float32").mean(),
                        attn_type=attn_type,
                    )
        except Exception as e:
            logger.error(
                f"[PaddleQKMonitor] Error sparse layer {layer_idx}: {e}"
            )

    def _record_common_stats(self, layer_idx, attn_type, stats) -> None:
        all_heads = stats["entropy_per_head"]
        self.record_layer_metric(
            layer_idx, "max", stats["max_global"], attn_type=attn_type
        )
        self.record_layer_metric(
            layer_idx, "mean", stats["mean_global"], attn_type=attn_type
        )
        self.record_layer_metric(
            layer_idx,
            "entropy_avg",
            stats["entropy_global"],
            attn_type=attn_type,
        )
        self.record_layer_metric(
            layer_idx, "sink", stats["sink_global"], attn_type=attn_type
        )
        self.record_layer_metric(
            layer_idx, "entropy_min", all_heads.min(), attn_type=attn_type
        )
        self.record_layer_metric(
            layer_idx, "entropy_max", all_heads.max(), attn_type=attn_type
        )
        self.record_layer_metric(
            layer_idx, "entropy_std", all_heads.std(), attn_type=attn_type
        )
        # sink_per_head: [B, H] — average across batch to get [H]
        sink_per_head = stats["sink_per_head"]
        sink_for_classify = (
            sink_per_head.mean(axis=0)
            if sink_per_head.ndim > 1
            else sink_per_head
        )
        sink_class = compute_sink_head_classification(
            sink_for_classify, threshold=self.sink_head_threshold
        )
        for name, val in sink_class.items():
            self.record_layer_metric(layer_idx, name, val, attn_type=attn_type)

    def _find_attention_layers(
        self, model: nn.Layer
    ) -> list[tuple[int, nn.Layer, object]]:
        def has_attention(layer):
            # KDA layers are excluded on purpose: they are linear attention, so
            # there are no QK logits, no softmax and no ``core_attention`` to hook.
            if is_kda_layer(layer):
                return False
            return hasattr(layer, "self_attn") or hasattr(
                layer, "self_attention"
            )

        layers = get_decoder_layers(model)
        if layers is None:
            transformer_layers = [
                sublayer
                for _name, sublayer in model.named_sublayers()
                if has_attention(sublayer)
            ]
            layers = transformer_layers if transformer_layers else None
        if layers is None:
            return []

        monitor_layers = iter_monitor_layers(
            layers, has_attention, pp_rank=self.pp_rank
        )
        self.mark_mtp_layers(item.idx for item in monitor_layers if item.is_mtp)
        attention_layers = []
        for item in monitor_layers:
            attn = getattr(item.layer, "self_attn", None) or getattr(
                item.layer, "self_attention", None
            )
            if attn is not None:
                attention_layers.append((item.idx, attn, item))
        return attention_layers

    def _cp_gather_seq(self, tensor: paddle.Tensor) -> paddle.Tensor | None:
        """Restore ``tensor`` to global sequence order across the CP group.

        Contiguous CP concatenates rank shards in rank order. Dual-chunk CP
        stores a front chunk plus a rear chunk on every rank, so a plain
        all-gather would leave both Q rows and K columns in the balanced
        layout. ``all_gather_balance`` undoes that layout; without it causal
        masks, entropy and sink would silently compare the wrong tokens.
        """
        if self.cp_group is None or self.cp_size <= 1:
            return tensor
        try:
            from paddlefleet.context_parallel_utils import (
                all_gather_balance,
                all_gather_contiguous,
            )

            gather = (
                all_gather_contiguous
                if self.cp_balance_mode.startswith("contiguous")
                else all_gather_balance
            )
            return gather(tensor, group=self.cp_group, axis=1)
        except Exception as e:
            if not self._warned_cp_gather_failed:
                logger.warning(
                    "[PaddleQKMonitor] CP all_gather failed (%s); skipping qk_stats "
                    "for this step to preserve correctness. Consider removing "
                    "qk_stats from internal_medicine_monitors under CP.",
                    e,
                )
                self._warned_cp_gather_failed = True
            return None

    def _make_compute_hook(
        self,
        layer_idx: int,
        attn_type: str | None = None,
        *,
        record_qkv_norms: bool = True,
    ):
        def hook_fn(layer, inputs):
            if not layer.training:
                return
            if not self._should_monitor():
                return
            try:
                query, key = inputs[0], inputs[1]
                value = inputs[2] if len(inputs) > 2 else None
                with paddle.no_grad():
                    if record_qkv_norms:
                        for name, tensor in (
                            ("q", query),
                            ("k", key),
                            ("v", value),
                        ):
                            if not isinstance(tensor, paddle.Tensor):
                                continue
                            vector_norm = paddle.linalg.norm(
                                tensor.detach().astype("float32"), axis=-1
                            )
                            self.record_layer_metric(
                                layer_idx,
                                f"{name}_norm_mean",
                                vector_norm.mean(),
                                attn_type=attn_type,
                            )
                            self.record_layer_metric(
                                layer_idx,
                                f"{name}_norm_max",
                                vector_norm.max(),
                                attn_type=attn_type,
                            )
                    if self.cp_size > 1:
                        # Restore both Q and K to global token order. Dual-chunk
                        # shards are two non-contiguous intervals, so a single
                        # q_row_offset cannot describe them; gathering Q as well
                        # lets the kernel run with offset 0 on the full sequence.
                        query = self._cp_gather_seq(query.detach())
                        key = self._cp_gather_seq(key.detach())
                        if query is None or key is None:
                            return
                    q_row_offset = 0

                    # query: [B, S_q, H, D]; seq_len is a static shape int (no D2H sync).
                    seq_len = query.shape[1]
                    effective_stride = self.row_stride
                    if effective_stride > seq_len:
                        # Stride coarser than the sequence would visit a single
                        # (or zero) row -> statistically useless. Clamp to a full
                        # pass and warn once instead of silently degrading.
                        if not self._warned_row_stride_gt_seqlen:
                            logger.warning(
                                "[PaddleQKMonitor] row_stride=%d exceeds seq_len=%d; "
                                "clamping to 1 (full pass) for this run.",
                                self.row_stride,
                                seq_len,
                            )
                            self._warned_row_stride_gt_seqlen = True
                        effective_stride = 1
                    # GQA grouping is handled inside the kernel; do NOT
                    # repeat_interleave the KV tensor on the hot path.
                    # ``layer`` is the core attention (that is where this hook is
                    # registered), so its sink bias is read straight off it. Fold
                    # it whenever it exists — including the frozen zero buffer of
                    # "off-by-one" softmax, which still adds exp(0) to the
                    # denominator and so changes entropy and sink.
                    sink_logit = getattr(layer, "softmax_offset", None)
                    stats = compute_qk_stats_paddle(
                        query,
                        key,
                        causal=self.causal,
                        row_stride=effective_stride,
                        q_row_offset=q_row_offset,
                        attn_sink=None
                        if sink_logit is None
                        else sink_logit.detach().astype("float32"),
                        softmax_scale=getattr(layer, "softmax_scale", None),
                    )

                self._record_common_stats(layer_idx, attn_type, stats)
                # A frozen offset ("off-by-one") has no learned magnitude to
                # track, so only the trainable one gets its own series.
                if sink_logit is not None and not sink_logit.stop_gradient:
                    self.record_layer_metric(
                        layer_idx,
                        "attn_sink_logit",
                        sink_logit.detach().astype("float32").mean(),
                        attn_type=attn_type,
                    )
            except Exception as e:
                logger.error(f"[PaddleQKMonitor] Error layer {layer_idx}: {e}")

        return hook_fn


def setup_qk_monitor(
    model,
    causal=True,
    verbose=False,
    log_per_layer=True,
    log_global=True,
    monitor_interval=1,
    sink_head_threshold: float = 0.3,
    row_stride: int = 1,
    monitor_dict=None,
    exclude_families=None,
):
    monitor = PaddleQKStatsMonitor(
        exclude_families=exclude_families,
        causal=causal,
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
        sink_head_threshold=sink_head_threshold,
        row_stride=row_stride,
    )
    monitor.register_hooks(model)
    logger.info(
        f"[PaddleQKMonitor] Setup complete. Monitoring {len(monitor.hooks)} layers."
    )
    if monitor_dict is not None:
        monitor_dict["qk_stats"] = monitor
    return model
