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

"""Host-side plan (a block-sparse schedule) for packed prefix-LM attention.

## What this is

The packed + Triton attention path does not compute the mask inside the
kernel. Instead it consumes a set of integer arrays precomputed on the host:

* `tok_seg_start` / `tok_seg_ctx`: per-token segment identity (pad is -1 / 0);
* `kv_lo` / `kv_hi`: the KV block range to scan per q-block for the forward
  pass and the dQ pass;
* `q_lo` / `q_hi`: the q-block range to scan per KV block for the dK/dV pass;
* `full_flags` / `full_flags_bwd`: whether each `(q-block, kv-block)` tile is
  FULL (all pairs allowed, so the kernel can take the mask-free fast path).

This layer is pure integer logic with no floating point, so a bug here cannot
be hidden by numerical noise: either a shape mismatch crashes outright, or an
incorrect scan range / FULL flag silently produces a wrong result. The plan is
validated by exact integer equality (`np.array_equal`) against a dense
reference implementation.

## Why this is a separate file that does not import paddle compute

The function bodies use only `int` / `list` / `range`; only the final step
packs the results into `paddle.Tensor`. Keeping the plan free of framework
compute means the result is deterministic and does not depend on operator
implementation details.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SegmentLayout:
    """Per-segment layout of one packed encoder sequence.

    Attributes:
        segment_starts: global start index of each segment in the packed sequence.
        n_contexts: context-prefix length of each segment (may be < 128, never 0).
        n_queries: query-suffix length of each segment (e.g. 256 / 8192).
        seq_total: total length including trailing pad (128-aligned).
        pad_start: index of the first pad token; `== seq_total` means no pad.
    """

    segment_starts: tuple[int, ...]
    n_contexts: tuple[int, ...]
    n_queries: tuple[int, ...]
    seq_total: int
    pad_start: int

    @property
    def segment_ends(self) -> tuple[int, ...]:
        """End index (exclusive) of each segment."""
        return tuple(
            s + c + q
            for s, c, q in zip(
                self.segment_starts, self.n_contexts, self.n_queries
            )
        )


def layout_from_hyperbody_dict(
    layout: dict[str, Any], seq_total: int
) -> SegmentLayout:
    """Build a `SegmentLayout` from a layout dict.

    The dict is attached to `packed_seq_params` by the packed forward pass;
    it is the copy that survives recompute, when thread-local state has already
    been popped.
    """
    segment_starts = tuple(int(x) for x in layout["segment_starts"])
    n_contexts = tuple(int(x) for x in layout["n_contexts"])
    n_queries = tuple(int(x) for x in layout["n_queries"])
    pad_len = int(layout.get("pad_len", 0))
    pad_start = seq_total - pad_len

    if not (len(segment_starts) == len(n_contexts) == len(n_queries)):
        raise ValueError(
            f"segment_starts/n_contexts/n_queries length mismatch: "
            f"{len(segment_starts)}/{len(n_contexts)}/{len(n_queries)}"
        )
    return SegmentLayout(
        segment_starts=segment_starts,
        n_contexts=n_contexts,
        n_queries=n_queries,
        seq_total=seq_total,
        pad_start=pad_start,
    )


def default_scale(head_dim: int) -> float:
    """`1/sqrt(head_dim)`."""
    import math

    return 1.0 / math.sqrt(float(head_dim))


# ---------------------------------------------------------------------------
# Per-token segment metadata + host-side block-sparse plan (pure integer)
# ---------------------------------------------------------------------------


def build_token_segment_arrays_list(
    layout: SegmentLayout,
) -> tuple[list[int], list[int]]:
    """Expand the layout into two per-token `int` lists.

    The mask is encoded as per-token segment identity (rather than one seg-id
    per q-block), so q-blocks that straddle a segment boundary need no special
    case: two tokens are in the same segment iff their `tok_seg_start` matches.
    Pad tokens are marked `-1` and never share a segment with any token (their
    self-loop is handled by an explicit pad-diagonal term inside the kernel).
    """
    n = layout.seq_total
    tok_seg_start = [-1] * n
    tok_seg_ctx = [0] * n
    for start, nctx, end in zip(
        layout.segment_starts, layout.n_contexts, layout.segment_ends
    ):
        for t in range(start, end):
            tok_seg_start[t] = int(start)
            tok_seg_ctx[t] = int(nctx)
    return tok_seg_start, tok_seg_ctx


def build_block_plan_list(
    layout: SegmentLayout, block_m: int, block_n: int
) -> tuple[list[int], list[int], int]:
    """Compute the `[kv_lo, kv_hi)` KV block range to scan per q-block.

    Pure host-side integer arithmetic (O(#segments), no kernel compilation).
    Fully masked KV blocks are skipped. The range does not need to be tight:
    over-scanning only costs performance, since the kernel predicate zeroes out
    any disallowed position anyway.

    The upper bound exploits the prefix-LM causal structure: query-suffix rows
    only attend to the context prefix plus earlier query rows, so a q-block's
    KV reach is at most `max(segment_ctx_end, q_block_end)`. This prunes the
    fully-masked tail that a naive `hi = segment_end` would scan for every
    early q-block, which is the dominant waste on long query segments.
    """
    n = layout.seq_total
    num_q_blocks = (n + block_m - 1) // block_m

    seg_start = layout.segment_starts
    seg_ctx = layout.n_contexts
    seg_ends = layout.segment_ends
    pad_start = layout.pad_start

    kv_lo = [0] * num_q_blocks
    kv_hi = [0] * num_q_blocks

    for qb in range(num_q_blocks):
        q0 = qb * block_m
        q1 = min(q0 + block_m, n)
        lo = n  # min over per-row seg_start
        hi = 0  # max reachable kv over rows
        has_pad = False
        for s, nctx, e in zip(seg_start, seg_ctx, seg_ends):
            # Does this segment intersect the q-row range [q0, q1)?
            if s < q1 and e > q0:
                ctx_end = s + nctx
                # Causal reach: context rows see ctx_end; query rows see their
                # own index (at most q1-1 within the block, clipped by e).
                # `max(ctx_end, min(q1, e))` is a tight upper bound that also
                # covers the full prefix for a pure-context q-block.
                seg_hi = max(ctx_end, min(q1, e))
                lo = min(lo, s)
                hi = max(hi, seg_hi)
        if (
            q1 > pad_start
        ):  # block contains pad rows -> need their diagonal columns
            has_pad = True
        if lo == n and not has_pad:
            # entirely outside all segments and no pad (should not happen) -> empty
            kv_lo[qb] = 0
            kv_hi[qb] = 0
            continue
        if has_pad:
            lo = min(lo, q0)
            hi = max(hi, q1)
        kv_lo[qb] = lo // block_n
        kv_hi[qb] = (hi + block_n - 1) // block_n

    return kv_lo, kv_hi, num_q_blocks


def build_block_plan_bwd_list(
    layout: SegmentLayout, block_m: int, block_n: int
) -> tuple[list[int], list[int], int]:
    """Reverse-lookup the `[q_lo, q_hi)` q-block range to scan per KV block for dK/dV.

    Symmetric to `build_block_plan_list`. The LO-side causal pruning mirrors the
    forward HI-side pruning: a KV column in the query suffix is only seen by rows
    with `q_idx >= k_idx`, so a pure query-suffix KV block is first reached from
    its own first column rather than from the segment start; a KV block hitting
    the context prefix is still reached by the whole segment (context is seen by
    every row in the segment).
    """
    n = layout.seq_total
    num_kv_blocks = (n + block_n - 1) // block_n
    seg_start = layout.segment_starts
    seg_ctx = layout.n_contexts
    seg_ends = layout.segment_ends
    pad_start = layout.pad_start

    q_lo = [0] * num_kv_blocks
    q_hi = [0] * num_kv_blocks

    for kb in range(num_kv_blocks):
        k0 = kb * block_n
        k1 = min(k0 + block_n, n)
        lo = n
        hi = 0
        has_pad = False
        for s, nctx, e in zip(seg_start, seg_ctx, seg_ends):
            if s < k1 and e > k0:
                ctx_end = s + nctx
                first_col = max(
                    k0, s
                )  # first KV column of this segment in the block
                if first_col < ctx_end:
                    # block hits this segment's context -> the whole segment reads it
                    seg_lo = s
                else:
                    # pure query-suffix columns -> causal: reached only from that column on
                    seg_lo = first_col
                lo = min(lo, seg_lo)
                hi = max(hi, e)
        if k1 > pad_start:
            has_pad = True
        if lo == n and not has_pad:
            q_lo[kb] = 0
            q_hi[kb] = 0
            continue
        if has_pad:
            lo = min(lo, k0)
            hi = max(hi, k1)
        q_lo[kb] = lo // block_m
        q_hi[kb] = (hi + block_m - 1) // block_m

    return q_lo, q_hi, num_kv_blocks


def _tile_is_full(
    seg_start: tuple[int, ...],
    seg_ctx: tuple[int, ...],
    seg_ends: tuple[int, ...],
    n: int,
    pad_start: int,
    q0: int,
    q1: int,
    k0: int,
    k1: int,
) -> bool:
    """Is every `(q, k)` pair in `[q0,q1) x [k0,k1)` allowed?

    Pure integer test (O(#segments), no dense mask materialized). A tile is FULL
    iff it is a complete in-range tile with no pad rows/columns, lies entirely
    within a single segment `(s, nctx, e)` (no cross-segment pairs), and the
    three-region predicate holds block-wise: either (A) the whole k-tile is
    context (`k1 <= ctx_end`), or (B) the whole q-tile is query suffix
    (`q0 >= ctx_end`) and the block is causally ordered (`q0 >= k1 - 1`). Tiles
    that cross segments, exceed `seq_len`, or touch pad are always PARTIAL (the
    conservative direction: only performance is lost, never correctness).

    Note: mislabeling the other way (marking a partial tile FULL) would produce
    a wrong result, so a tile is declared FULL only when it is provably safe.
    """
    if q1 > n or q1 > pad_start:  # out-of-range or pad-touching q-tile
        return False
    if k1 > n or k1 > pad_start:  # out-of-range or pad-touching kv-tile
        return False
    seg = None
    for s, nctx, e in zip(seg_start, seg_ctx, seg_ends):
        if s <= q0 and q1 <= e and s <= k0 and k1 <= e:
            seg = (s, nctx)
            break
    if seg is None:  # crosses segments -> some pairs not allowed
        return False
    s, nctx = seg
    ctx_end = s + nctx
    if k1 <= ctx_end:  # case (A): the whole k-tile is context
        return True
    if (
        q0 < ctx_end
    ):  # some q rows are context -> they do not see query-suffix k
        return False
    return q0 >= k1 - 1  # case (B): all query suffix and block-wise causal


def build_block_full_flags_list(
    layout: SegmentLayout,
    block_m: int,
    block_n: int,
    kv_lo: list[int],
    kv_hi: list[int],
) -> list[list[int]]:
    """FULL(1)/partial(0) flag for each `(q-block, kv-block)` in the forward scan range.

    Only tiles inside the forward scan range `[kv_lo, kv_hi)` are tested; tiles
    outside are always 0 (the forward / dQ kernel never visits them).
    """
    n = layout.seq_total
    num_q_blocks = (n + block_m - 1) // block_m
    num_kv_blocks = (n + block_n - 1) // block_n
    flags = [[0] * num_kv_blocks for _ in range(num_q_blocks)]

    seg_start = layout.segment_starts
    seg_ctx = layout.n_contexts
    seg_ends = layout.segment_ends
    pad_start = layout.pad_start

    for qb in range(num_q_blocks):
        q0 = qb * block_m
        q1 = q0 + block_m
        for blk in range(kv_lo[qb], kv_hi[qb]):
            k0 = blk * block_n
            k1 = k0 + block_n
            if _tile_is_full(
                seg_start, seg_ctx, seg_ends, n, pad_start, q0, q1, k0, k1
            ):
                flags[qb][blk] = 1
    return flags


def build_block_full_flags_bwd_list(
    layout: SegmentLayout,
    block_m: int,
    block_n: int,
    q_lo: list[int],
    q_hi: list[int],
) -> list[list[int]]:
    """FULL/partial flags for the dK/dV scan range.

    The FULL rule uses the same `_tile_is_full` and the same
    `[num_q_blocks, num_kv_blocks]` shape indexed `[qb, kb]`, but fills the
    reverse scan set: the dK/dV kernel loops KV-block outer / q-block inner, so
    the set of `(qb, kb)` tiles it visits differs from the forward set (the
    causal pruning is asymmetric) and must be computed separately. The dQ kernel
    shares the forward scan order and reuses the forward flags directly.
    """
    n = layout.seq_total
    num_q_blocks = (n + block_m - 1) // block_m
    num_kv_blocks = (n + block_n - 1) // block_n
    flags = [[0] * num_kv_blocks for _ in range(num_q_blocks)]

    seg_start = layout.segment_starts
    seg_ctx = layout.n_contexts
    seg_ends = layout.segment_ends
    pad_start = layout.pad_start

    for kb in range(num_kv_blocks):
        k0 = kb * block_n
        k1 = k0 + block_n
        for qb in range(q_lo[kb], q_hi[kb]):
            q0 = qb * block_m
            q1 = q0 + block_m
            if _tile_is_full(
                seg_start, seg_ctx, seg_ends, n, pad_start, q0, q1, k0, k1
            ):
                flags[qb][kb] = 1
    return flags


# ---------------------------------------------------------------------------
# Pack into paddle.Tensor -- dtypes must be int32 / int8
# ---------------------------------------------------------------------------


def build_exec_plan(layout: SegmentLayout, block_m: int, block_n: int) -> dict:
    """Build all integer arrays the kernel needs in one shot, as `paddle.Tensor`.

    dtypes: segment / range arrays are `int32`, flags are `int8`. The kernel
    dereferences by element type, so a wrong dtype silently reads wrong values
    instead of raising.
    """
    import paddle

    seg_start, seg_ctx = build_token_segment_arrays_list(layout)
    kv_lo, kv_hi, n_q_blocks = build_block_plan_list(layout, block_m, block_n)
    flags = build_block_full_flags_list(layout, block_m, block_n, kv_lo, kv_hi)
    q_lo, q_hi, n_kv_blocks = build_block_plan_bwd_list(
        layout, block_m, block_n
    )
    flags_bwd = build_block_full_flags_bwd_list(
        layout, block_m, block_n, q_lo, q_hi
    )

    def _i32(v):
        return paddle.to_tensor(v, dtype="int32")

    def _i8(v):
        return paddle.to_tensor(v, dtype="int8")

    return {
        "tok_seg_start": _i32(seg_start),
        "tok_seg_ctx": _i32(seg_ctx),
        "kv_lo": _i32(kv_lo),
        "kv_hi": _i32(kv_hi),
        "full_flags": _i8(flags),
        "q_lo": _i32(q_lo),
        "q_hi": _i32(q_hi),
        "full_flags_bwd": _i8(flags_bwd),
        "n_q_blocks": n_q_blocks,
        "n_kv_blocks": n_kv_blocks,
    }


# ---------------------------------------------------------------------------
# exec-plan LRU cache
# ---------------------------------------------------------------------------

# The plan is a pure function of `(layout, block_m, block_n)` plus one H2D copy.
# Rebuilding it on every call costs ~2ms of Python loops + transfer (for a 4k
# token pack) while the flash kernel itself is only ~0.05ms; with recompute
# enabled each layer pays it twice. So the device-side tensors are memoized by
# layout. The cache-size env var lets deployments tune the memory footprint.
_PLAN_CACHE_MAX = int(
    os.environ.get("HYPERBODY_TRITON_BLOCK_PLAN_CACHE_SIZE", "64")
)
_EXEC_PLAN_CACHE: OrderedDict[tuple, dict] = OrderedDict()


def _layout_cache_key(
    layout: SegmentLayout, block_m: int, block_n: int
) -> tuple:
    """Cache key for `(layout, block_m, block_n)`."""
    return (
        layout.segment_starts,
        layout.n_contexts,
        layout.n_queries,
        layout.seq_total,
        layout.pad_start,
        block_m,
        block_n,
    )


def get_exec_plan(layout: SegmentLayout, block_m: int, block_n: int) -> dict:
    """Return the cached exec plan (device-side tensors), LRU-capped at `_PLAN_CACHE_MAX`."""
    if _PLAN_CACHE_MAX <= 0:
        return build_exec_plan(layout, block_m, block_n)
    key = _layout_cache_key(layout, block_m, block_n)
    val = _EXEC_PLAN_CACHE.get(key)
    if val is not None:
        _EXEC_PLAN_CACHE.move_to_end(key)
        return val
    val = build_exec_plan(layout, block_m, block_n)
    _EXEC_PLAN_CACHE[key] = val
    if len(_EXEC_PLAN_CACHE) > _PLAN_CACHE_MAX:
        _EXEC_PLAN_CACHE.popitem(last=False)
    return val


# ---------------------------------------------------------------------------
# Triton kernel loading (lazy)
# ---------------------------------------------------------------------------

_KERNEL_NAMES = (
    "_prefix_lm_fwd_kernel",
    "_prefix_lm_delta_kernel",
    "_prefix_lm_dkdv_kernel",
    "_prefix_lm_dq_kernel",
)
_KERNELS: dict | None = None


def _kernels() -> dict:
    """Lazily load the 4 kernels and wrap them with paddle's Triton compat decorator.

    Loading is lazy for two reasons:
    * the plan functions above are pure integer code and should remain
      importable even when triton is not installed in the environment;
    * `paddle.enable_compat(scope={"triton"})` must be called BEFORE
      `import triton`, i.e. the torch-compat layer must be enabled before
      importing triton.
    """
    global _KERNELS
    if _KERNELS is None:
        import paddle

        from .utils import (
            enable_compat_on_triton_kernel,
            is_torch_compat_available,
        )

        if is_torch_compat_available():
            paddle.enable_compat(scope={"triton"})

        from . import prefix_lm_kernels as _pk

        _KERNELS = {
            n: enable_compat_on_triton_kernel(getattr(_pk, n))
            for n in _KERNEL_NAMES
        }
    return _KERNELS


def _st(x) -> tuple:
    """Element-wise stride (paddle's `.strides` matches torch's `.stride()`)."""
    return tuple(x.strides)


# ---------------------------------------------------------------------------
# Host-side wrappers for the forward / backward passes
# ---------------------------------------------------------------------------


# warps/stages are read from env vars: num_warps directly determines the
# softmax reduction tree, so it (and num_stages) are configurable to tune the
# launch configuration.
def _fwd_launch() -> tuple[int, int]:
    return (
        int(os.environ.get("HYPERBODY_TRITON_FWD_WARPS", "4")),
        int(os.environ.get("HYPERBODY_TRITON_FWD_STAGES", "2")),
    )


def _bwd_launch() -> tuple[int, int]:
    return (
        int(os.environ.get("HYPERBODY_TRITON_BWD_WARPS", "4")),
        int(os.environ.get("HYPERBODY_TRITON_BWD_STAGES", "2")),
    )


def triton_prefix_lm_forward(
    query,
    key,
    value,
    layout: SegmentLayout,
    *,
    scale: float,
    block_m: int = 64,
    block_n: int = 64,
    plan: dict | None = None,
):
    """Run the Triton prefix-LM flash forward.

    Args:
        query/key/value: `[B, H, T, D]`, ideally contiguous.
        layout: packed segment layout (`T == seq_total`).
        scale: softmax scale.
        plan: output of `get_exec_plan`; if omitted it is built on the fly
            (slower, same semantics).

    Returns:
        `(out, lse)`, where `out` is `[B,H,T,D]` and `lse` is `[B,H,T]` fp32.
    """
    import paddle

    b, h, t, d = query.shape
    if t != layout.seq_total:
        raise ValueError(f"seq len {t} != layout.seq_total {layout.seq_total}")
    p = plan if plan is not None else get_exec_plan(layout, block_m, block_n)

    out = paddle.empty_like(query)
    lse = paddle.empty([b, h, t], dtype="float32")
    warps, stages = _fwd_launch()
    _kernels()["_prefix_lm_fwd_kernel"][(p["n_q_blocks"], b * h)](
        query,
        key,
        value,
        out,
        lse,
        p["tok_seg_start"],
        p["tok_seg_ctx"],
        p["kv_lo"],
        p["kv_hi"],
        p["full_flags"],
        *_st(query),
        *_st(key),
        *_st(value),
        *_st(out),
        *_st(lse),
        *_st(p["full_flags"]),
        t,
        layout.pad_start,
        scale,
        H=h,
        HEAD_DIM=d,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=warps,
        num_stages=stages,
    )
    return out, lse


def triton_prefix_lm_backward(
    dout,
    query,
    key,
    value,
    out,
    lse,
    layout: SegmentLayout,
    *,
    scale: float,
    block_m: int = 64,
    block_n: int = 64,
    plan: dict | None = None,
):
    """Run the Triton prefix-LM flash backward.

    Three passes, no atomic accumulation: `delta` is independent per q-block;
    `dk`/`dv` are owned exclusively by the KV-block-outer program (using
    `q_lo`/`q_hi`/`full_flags_bwd`); `dq` is owned exclusively by the
    q-block-outer program (reusing the forward `kv_lo`/`kv_hi`/`full_flags`).
    The result is therefore deterministically reproducible.

    Returns:
        `(dq, dk, dv)`, same dtype as the inputs.
    """
    import paddle

    b, h, t, d = query.shape
    p = plan if plan is not None else get_exec_plan(layout, block_m, block_n)
    ks = _kernels()

    dout = dout.contiguous()
    delta = paddle.empty([b, h, t], dtype="float32")
    dq = paddle.empty_like(query)
    dk = paddle.empty_like(key)
    dv = paddle.empty_like(value)

    ks["_prefix_lm_delta_kernel"][(p["n_q_blocks"], b * h)](
        out,
        dout,
        delta,
        *_st(out),
        *_st(dout),
        *_st(delta),
        t,
        H=h,
        HEAD_DIM=d,
        BLOCK_M=block_m,
    )
    warps, stages = _bwd_launch()
    ks["_prefix_lm_dkdv_kernel"][(p["n_kv_blocks"], b * h)](
        query,
        key,
        value,
        dout,
        lse,
        delta,
        dk,
        dv,
        p["tok_seg_start"],
        p["tok_seg_ctx"],
        p["q_lo"],
        p["q_hi"],
        p["full_flags_bwd"],
        *_st(query),
        *_st(key),
        *_st(value),
        *_st(dout),
        *_st(lse),
        *_st(delta),
        *_st(dk),
        *_st(dv),
        *_st(p["full_flags_bwd"]),
        t,
        layout.pad_start,
        scale,
        H=h,
        HEAD_DIM=d,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=warps,
        num_stages=stages,
    )
    ks["_prefix_lm_dq_kernel"][(p["n_q_blocks"], b * h)](
        query,
        key,
        value,
        dout,
        lse,
        delta,
        dq,
        p["tok_seg_start"],
        p["tok_seg_ctx"],
        p["kv_lo"],
        p["kv_hi"],
        p["full_flags"],
        *_st(query),
        *_st(key),
        *_st(value),
        *_st(dout),
        *_st(lse),
        *_st(delta),
        *_st(dq),
        *_st(p["full_flags"]),
        t,
        layout.pad_start,
        scale,
        H=h,
        HEAD_DIM=d,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=warps,
        num_stages=stages,
    )
    return dq, dk, dv


# ---------------------------------------------------------------------------
# autograd wrapper + drop-in core-attention
# ---------------------------------------------------------------------------


def _attn_cls():
    """Lazily build the `PyLayer` subclass -- defining it requires importing
    paddle, while this module must stay usable in pure-integer mode."""
    global _ATTN_CLS
    if _ATTN_CLS is not None:
        return _ATTN_CLS

    import paddle

    class _PrefixLMTritonAttn(paddle.autograd.PyLayer):
        """Bind the Triton fwd/bwd into a single differentiable op.

        The backward reads the layout from a non-tensor field on ctx rather than
        any instance attribute: recompute reruns forward under `no_grad`, and
        instance attributes are not recompute-safe.
        """

        @staticmethod
        def forward(ctx, query, key, value, layout, scale, block_m, block_n):
            plan = get_exec_plan(layout, block_m, block_n)
            out, lse = triton_prefix_lm_forward(
                query,
                key,
                value,
                layout,
                scale=scale,
                block_m=block_m,
                block_n=block_n,
                plan=plan,
            )
            ctx.save_for_backward(query, key, value, out, lse)
            ctx.layout = layout
            ctx.scale = scale
            ctx.block_m = block_m
            ctx.block_n = block_n
            # PyLayer contract: stop_gradient inputs must return None
            ctx.needs_grad = (
                not query.stop_gradient,
                not key.stop_gradient,
                not value.stop_gradient,
            )
            return out

        @staticmethod
        def backward(ctx, dout):
            query, key, value, out, lse = ctx.saved_tensor()
            dq, dk, dv = triton_prefix_lm_backward(
                dout,
                query,
                key,
                value,
                out,
                lse,
                ctx.layout,
                scale=ctx.scale,
                block_m=ctx.block_m,
                block_n=ctx.block_n,
                plan=get_exec_plan(ctx.layout, ctx.block_m, ctx.block_n),
            )
            gq, gk, gv = ctx.needs_grad
            # One gradient slot per TENSOR input, in order; layout/scale/block_*
            # are non-tensors and do not occupy a slot.
            return (dq if gq else None, dk if gk else None, dv if gv else None)

    _ATTN_CLS = _PrefixLMTritonAttn
    return _ATTN_CLS


_ATTN_CLS = None


def triton_prefix_lm_attention(
    query,
    key,
    value,
    layout: SegmentLayout,
    *,
    scale: float,
    block_m: int = 64,
    block_n: int = 64,
):
    """Differentiable Triton prefix-LM attention; inputs are `[B, H, T, D]`."""
    return _attn_cls().apply(query, key, value, layout, scale, block_m, block_n)
