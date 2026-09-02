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

"""Unified CSA Sparse Attention entry with single-switch backend dispatch.

A single ``backend`` argument selects one of three implementations of the
final sparse MQA attention:
  - "unfused": pure-Paddle einsum forward + Paddle autograd backward
  - "tilelang": TileLang sparse MQA forward + TileLang backward
  - "cudnn": FlashMLA sparse forward + cuDNN DSA backward

Head counts below a kernel tile are supported by zero-padding, see
``_pad_query_heads``. Latent widths below 512 likewise, see
``_pad_latent_dim``.
"""

import functools
import math

import paddle
from paddle import Tensor

# Query-head counts the "cudnn" backend can run natively. FlashMLA's sparse
# prefill only instantiates these two tiles (``csrc/api/sparse_fwd.h`` raises
# "Unsupported h_q" for anything else), and the tile width is the M extent of
# the tcgen05 / wgmma instruction behind it (``B_H`` in
# ``sm100/prefill/sparse/fwd/head64/config.h``), so a layer with fewer heads
# cannot be given a narrower tile without a new kernel.
_DSA_HEAD_TILES = (64, 128)
# The only latent width the FlashMLA sparse prefill and the cuDNN DSA backward
# accept on this symmetric path; narrower layers are zero-padded up to it, see
# ``_pad_latent_dim``.
_DSA_LATENT_DIM = 512
# Sink logit that makes ``exp(sink - m) -> 0``, i.e. plain softmax. Used for the
# padded heads so they cannot perturb the real ones.
_NEG_SINK = -1e30


def _dsa_head_tile(num_heads: int) -> int:
    """Smallest cuDNN/FlashMLA query-head tile that fits ``num_heads``."""
    for tile in _DSA_HEAD_TILES:
        if num_heads <= tile:
            return tile
    raise ValueError(
        f"csa_sparse_attn 'cudnn' backend supports at most "
        f"{_DSA_HEAD_TILES[-1]} query heads per rank, got {num_heads}. "
        "Reduce num_attention_heads (per-rank after TP)."
    )


@functools.lru_cache(maxsize=16)
def _real_rows(
    num_tokens: int, num_heads: int, head_tile: int, keep: int, total: int
) -> Tensor:
    """Row ids of the real data in a ``[num_tokens * head_tile * total, g]`` view.

    Of every ``head_tile`` head rows the first ``num_heads`` are real, and of
    every ``total`` latent chunks the first ``keep`` are real; the rest is the
    zero padding added on the way into the kernel.

    Cached because it only depends on the shape. One entry is
    ``num_tokens * num_heads * keep`` int32 (0.75 MiB at 8k tokens x 24 heads
    x 1 chunk), and a process only ever sees one device, so a plain shape-keyed
    cache is enough.
    """
    head_rows = (
        paddle.arange(num_tokens, dtype="int32") * (head_tile * total)
    ).unsqueeze(1) + (
        paddle.arange(num_heads, dtype="int32") * total
    ).unsqueeze(0)
    return (
        head_rows.reshape([-1, 1])
        + paddle.arange(keep, dtype="int32").unsqueeze(0)
    ).flatten()


def _drop_padded_rows(
    x: Tensor, num_heads: int, head_tile: int, hn: int
) -> Tensor:
    """Undo ``_pad_query_heads`` / ``_pad_latent_dim`` on a kernel output.

    ``x`` is ``[..., head_tile, x.shape[-1]]``; the result is
    ``[..., num_heads, hn]``, i.e. the padded head rows *and* the padded latent
    columns are dropped. Returns ``x`` untouched when there is no padding.

    Both drops are done by a *single* row ``gather`` rather than by a strided
    slice per axis. The slice moves the same bytes but Paddle's strided-slice
    copy runs at roughly a quarter of the achievable bandwidth, and doing the
    latent axis first also materialises a full-head-tile intermediate. Measured
    on B30Z at the HCA shape (8192 tokens, 64->24 heads, 512->256 latent, so a
    512 MiB kernel output): 0.803 ms for ``x[..., :hn].contiguous()`` followed
    by a head-row gather (704 MiB moved), 0.045 ms for the fused gather
    (192 MiB moved) -- a 17x saving on what is otherwise ~half of the forward.

    Chunking is by ``gcd(hn, kernel_hn)`` so that keeping the first ``hn``
    columns is expressible as keeping whole rows, which holds for any width the
    kernel is padded from (512/256 -> one 256-wide row of two, 512/384 -> three
    128-wide rows of four).
    """
    kernel_hn = x.shape[-1]
    if num_heads == head_tile and hn == kernel_hn:
        return x
    g = math.gcd(hn, kernel_hn)
    total = kernel_hn // g
    rows = x.reshape([-1, g])
    num_tokens = rows.shape[0] // (head_tile * total)
    idx = _real_rows(num_tokens, num_heads, head_tile, hn // g, total)
    return paddle.gather(rows, idx, axis=0).reshape(
        [*x.shape[:-2], num_heads, hn]
    )


@functools.cache
def _dsa_bwd_runs_sub_tile_heads(num_heads: int) -> bool:
    """Whether the cuDNN DSA backward accepts this head count below its tile.

    Running the backward -- the expensive half -- on the real head count skips
    the padding entirely and keeps the saved ``q``/``out`` at the real width.
    That is only safe when ``num_heads`` is a multiple of the kernel's 64-head
    tile, so in practice it never fires below a tile and the caller pads.

    ``dsa_bwd_sm100.py`` requires ``num_head % 64 == 0`` and does not check it:

    * ``_get_workspace_size_LSE_OdO`` (``:175-183``) sizes ``workspace_LSE_OdO``
      as ``(b, h, round_up(q, 8), 8)`` and splits it into back-to-back
      ``sum_OdO`` / ``scaled_lse`` views of ``H * Q * 4`` bytes each -- **zero
      slack**, with ``scaled_lse`` ending at the allocation end.
    * Both views declare ``cute.assume(H, divby=64)`` (``:229``/``:233``).
      Violating a ``cute.assume`` is UB, not an error: a dynamic ``Integer``
      goes to ``ConstrainedIntType`` unchecked, so the kernel compiles happily.
    * ``:1176-1207`` tiles the **head** mode by 64 (``cute.flat_divide``, see
      the ``# (64, 1, M, B)`` comment) and issues an **unpredicated**
      32-thread x 2-value ``cp.async``, i.e. 64 fp32 / 256 B. With ``H = 24``
      only 96 B of each tile is in bounds, so the last query token reads 160 B
      and the second-to-last 64 B past the end of the allocation.

    It is an out-of-bounds **read**, so results stay correct (shape-preserving
    canaries on both workspaces come back untouched and ``dq`` is
    bit-identical); the failure mode is ``CUDA error(700)`` whenever those
    160 B happen to land on an unmapped page. That depends only on the
    allocator layout -- deterministic per layout, a lottery across layouts --
    which is why short runs can look clean while the access is always OOB.
    Measured on B30Z / SM100 at ``total_S_q=32768``: ``H`` 16 / 24 / 32 / 48
    fault, ``H`` 64 / 128 pass even with zero mapped headroom.
    Odd ``H`` is the same access failing earlier, at 8-byte alignment
    (``CUDA 716``), and is covered by the same condition.

    Revert to ``num_heads % 2 == 0`` once the kernel either predicates those
    two tile loads or over-allocates ``workspace_LSE_OdO`` (bumping its ``q``
    extent by 64 -- 12 KiB at ``H=24`` -- is enough; both fixes were verified
    to leave ``dq`` bit-identical).

    SM90 is excluded separately: its kernel packs heads differently
    (``qhead_per_kvhead`` tiling) and is not validated here.
    """
    major, _ = paddle.device.cuda.get_device_capability()
    return major >= 10 and num_heads % 64 == 0


def _pad_query_heads(query: Tensor, attn_sink: Tensor, head_tile: int):
    """Widen ``query``/``attn_sink`` from ``h`` to ``head_tile`` heads.

    The padded query rows are zeros and their sink logit is ``-1e30``, so they
    run a plain softmax over the same columns and cannot change the real heads'
    result; their output rows are dropped by the caller. Hence they receive a
    zero output gradient, which makes their contribution to ``dkv`` and
    ``d_sink`` exactly zero (both are ``dO``-weighted sums) and their ``dq``
    rows unused. So this is numerically exact, not an approximation, and the
    ``h == head_tile`` path stays bit-for-bit unchanged.

    Padding is used instead of a narrower kernel because the tile width is the
    MMA M extent (see ``_DSA_HEAD_TILES``): the cost is that the forward keeps
    doing ``head_tile`` rows of work for ``h`` real heads.
    """
    b, sq, h, hn = query.shape
    query = paddle.concat(
        [query, paddle.zeros([b, sq, head_tile - h, hn], dtype=query.dtype)],
        axis=2,
    )
    attn_sink = paddle.concat(
        [
            attn_sink.cast("float32"),
            paddle.full([head_tile - h], _NEG_SINK, dtype="float32"),
        ],
        axis=0,
    )
    return query, attn_sink


def _dsa_latent_dim(hn: int) -> int:
    """Latent width the "cudnn" backend must be driven at for a ``hn`` layer.

    Always ``_DSA_LATENT_DIM``: the FlashMLA sparse prefill accepts
    ``d_qk in {512, 576}`` and requires ``d_v == 512``
    (``csrc/api/sparse_fwd.h``), and this CSA path calls it symmetrically
    (``d_v`` defaults to the query's last dim), so 512 is the only width that
    passes. The cuDNN DSA backward is equally 512-shaped: its SM100 kernel
    unrolls ``dQ`` / ``dKV`` into exactly four 128-column sub-tiles.
    """
    if hn > _DSA_LATENT_DIM:
        raise ValueError(
            f"csa_sparse_attn 'cudnn' backend supports at most "
            f"{_DSA_LATENT_DIM} latent dims, got {hn}. "
            "Reduce v_head_dim."
        )
    return _DSA_LATENT_DIM


def _pad_latent_dim(x: Tensor, latent_dim: int) -> Tensor:
    """Zero-pad the last (latent) axis of ``x`` up to ``latent_dim``.

    Applied to ``q`` and ``kv`` together, so it is numerically exact rather
    than an approximation. ``kv`` is a single latent head shared as both key
    and value, so with both sides zero in the padded columns:

    * ``scores = q . k`` is unchanged bit-for-bit (the added terms are 0 * 0),
      hence so are the softmax, the ``lse`` and the sink.
    * ``out = sum_j p_j * kv_j`` has the true result in ``[:hn]`` and exactly
      zero in the padded columns, so the caller just slices them off.
    * backward likewise: ``dq = sum_j dp_j * kv_j`` and
      ``dkv = sum_j p_j * dout_j`` are zero over the padded columns (``dout``
      is padded with zeros too), and ``d_sink`` is untouched.

    The cost is that both GEMMs run at ``latent_dim`` for ``hn`` real columns.
    The ``hn == latent_dim`` path is unchanged, and no tensor is saved for
    backward in padded form -- backward re-pads instead, so activation memory
    stays at the real width.
    """
    pad = latent_dim - x.shape[-1]
    if pad == 0:
        return x
    zeros_shape = list(x.shape)
    zeros_shape[-1] = pad
    return paddle.concat([x, paddle.zeros(zeros_shape, dtype=x.dtype)], axis=-1)


def unfused_compressed_sparse_attn(
    query: Tensor,
    kv_full: Tensor,
    attn_sink: Tensor,
    topk_indices: Tensor,
    softmax_scale: float,
    topk_length: Tensor | None = None,
) -> Tensor:
    """Sparse attention with MQA and learnable attention sink.

    Args:
        query: [b, sq, np, hn] multi-head query
        kv_full: [b, n_kv, hn] single-head KV (original + compressed concatenated)
        attn_sink: [np] per-head learnable bias (attention sink)
        topk_indices: [b, sq, topk] indices into kv_full dim=1 (-1 = invalid)
        softmax_scale: attention scale factor
        topk_length: optional [b, sq] int32 valid prefix length; slots at
            ``>= topk_length[b, i]`` are ignored for query row ``i``.

    Returns:
        output: [b, sq, np * hn]
    """
    b, sq, np_heads, hn = query.shape
    topk = topk_indices.shape[-1]

    # Clamp negative indices to 0 for gathering, mask them later
    safe_indices = paddle.clip(topk_indices, min=0).cast(
        paddle.int64
    )  # [b, sq, topk]
    safe_indices_exp = safe_indices.unsqueeze(-1).expand(
        [-1, -1, -1, hn]
    )  # [b, sq, topk, hn]

    # Gather KV at selected positions: [b, n_kv, hn] -> [b, sq, topk, hn]
    kv_gathered = paddle.gather(
        kv_full.unsqueeze(1).expand([-1, sq, -1, -1]),
        dim=2,
        index=safe_indices_exp,
    )
    with paddle.amp.auto_cast(False):
        # Compute attention scores: [b, np, sq, topk]
        q = query.transpose([0, 2, 1, 3]).cast("float32")  # [b, np, sq, hn]
        kv_g = kv_gathered.cast("float32")
        scores = (
            paddle.einsum("bnsh,bskh->bnsk", q, kv_g) * softmax_scale
        )  # [b, np, sq, topk]
        # Mask invalid positions (topk_indices < 0) with -inf
        invalid = topk_indices < 0  # [b, sq, topk]
        if topk_length is not None:
            slots = paddle.arange(topk, dtype="int32").reshape([1, 1, topk])
            invalid = invalid | (
                slots >= topk_length.reshape([b, sq, 1]).cast("int32")
            )
        invalid_mask = invalid.unsqueeze(1)  # [b, 1, sq, topk]
        scores = scores.masked_fill(invalid_mask, float("-inf"))

        # Softmax with attention sink
        # sink: [np] -> [1, np, 1, 1]
        sink = attn_sink.reshape([1, np_heads, 1, 1])
        # Compute stable softmax: max over scores and sink
        scores_max = scores.max(axis=-1, keepdim=True)  # [b, np, sq, 1]
        scores_max = paddle.maximum(scores_max, sink)

        exp_scores = paddle.exp(scores - scores_max)  # [b, np, sq, topk]
        exp_sink = paddle.exp(sink - scores_max)  # [b, np, sq, 1]

        sum_exp = (
            exp_scores.sum(axis=-1, keepdim=True) + exp_sink
        )  # [b, np, sq, 1]
        attn_weights = exp_scores / sum_exp  # [b, np, sq, topk]

        # Weighted sum: [b, np, sq, topk] x [b, sq, topk, hn] -> [b, np, sq, hn]
        output = paddle.einsum("bnsk,bskh->bnsh", attn_weights, kv_g)
    output = output.cast(query.dtype)

    # Reshape: [b, np, sq, hn] -> [b, sq, np * hn]
    output = output.transpose([0, 2, 1, 3]).reshape([b, sq, np_heads * hn])
    return output


def _csa_compute_topk_length(topk_idxs_flat: Tensor) -> Tensor:
    """Per-query safe loop bound for the cuDNN backward kernel (``mTopkLength``).

    Returns ``[N]`` int32 = (index of last valid entry + 1) per row, i.e. a SAFE
    upper bound: all valid entries lie in ``[0, topk_length)``. Uses the trailing
    bound (NOT ``sum(valid)``) so the bound stays valid when ``-1`` entries are
    interleaved (multi-doc leading/interior holes). Clamped to >=1 so the kernel
    writes every dq row (dq is allocated uninitialized in the backend).

    WARNING: this bound is only safe for the FORWARD kernel, which predicates
    ``-1`` slots. Feeding it to ``csa_sparse_attn_bwd_cudnn`` together with a
    holey ``topk_idxs`` selects the backward's compact KV-load path, which is
    unguarded against interior ``-1`` (see ``_csa_compact_topk_idxs``) and
    gathers ``mKV[-1]``. Observed consequences: ``CUDA error(700)``, nan/inf, or
    a silent ~50% error in dq/dkv while the forward output stays correct. For the
    backward, either compact the indices with ``_csa_compact_topk_idxs`` and pass
    the exact count it returns, or pass ``topk_length=None``.

    Args:
        topk_idxs_flat: ``[N, W]`` int32 global indices, -1 == invalid.
    """
    with paddle.no_grad():
        _, w = topk_idxs_flat.shape
        valid = (topk_idxs_flat >= 0).astype("int32")  # [N, W]
        cols = paddle.arange(1, w + 1, dtype="int32").unsqueeze(0)  # [1, W]
        trailing_len = (valid * cols).max(-1)  # [N] last valid index + 1
        trailing_len = paddle.clip(trailing_len, min=1).astype("int32")
    return trailing_len


def _csa_compact_topk_idxs(topk_idxs_flat: Tensor) -> tuple[Tensor, Tensor]:
    """Densify a ``[N, W]`` global-index tensor for the cuDNN DSA backward.

    Moves valid (``>= 0``) entries into a contiguous prefix and pushes ``-1`` to
    the trailing region, returning ``(compact_idxs, topk_length)`` where
    ``topk_length`` is the exact per-row valid count.

    The kernel's compact (``topk_length`` given) KV-load path is unguarded
    against interior ``-1`` holes. Multi-doc CSA/HCA rows carry them (a query in a
    later document has *leading* ``-1`` for earlier docs' compressed slots, plus
    the window/compressed concat), and DSA's ``[top-k | window]`` carries them
    too. Compacting removes every interior hole, so the compact path is both safe
    (no ``mKV[-1]`` gather) and a genuine early-stop (``topk_length`` = true count,
    not the trailing bound).

    Compaction is *order-preserving*: valid entries keep their original
    left-to-right order (holes are just removed). Although the backward is
    set-based (dq/softmax are reductions over the key set, dkv scatters by global
    index), preserving order keeps the kept terms in the same accumulation
    sequence as the un-compacted layout, and holes contribute exact ``0.0``, so
    dq/out/lse stay bit-identical -- a value-sort would instead reorder them and
    drift in the last bits. Empty rows get ``topk_length == 0`` and hit the
    kernel's empty-row fast path. It must NOT clip to a minimum: clipping to 1
    would leave a ``-1`` at column 0 and reintroduce the ``mKV[-1]`` gather.
    """
    with paddle.no_grad():
        valid = topk_idxs_flat >= 0
        lengths = valid.astype("int32").sum(-1).astype("int32")
        w = int(topk_idxs_flat.shape[-1])
        cols = paddle.arange(w, dtype="int32")
        # Order-preserving densify: a valid entry sorts by its real column
        # (0..w-1); a hole gets col+w (w..2w-1) so it lands after every valid
        # while BOTH groups keep their original left-to-right order. So the kept
        # entries are walked in exactly the same sequence as the un-compacted
        # layout -- and holes contribute exact 0.0 -- so dq/out/lse accumulate
        # bit-identically (unlike a value-sort, which would reorder them).
        key = paddle.where(valid, cols, cols + w)
        order = paddle.argsort(key, axis=-1)
        compact = paddle.take_along_axis(topk_idxs_flat, order, axis=-1)
    return compact, lengths


@functools.cache
def _csa_bwd_honours_topk_length_holes() -> bool:
    """Whether it is safe to hand the cuDNN DSA backward a *compacted*
    ``topk_length`` (dense prefix, possibly ``0`` for all-invalid rows) on this
    arch.

    Compaction can yield ``topk_length == 0`` for a fully-``-1`` padding row, and
    only the SM100 kernel early-exits (zeros dQ) on ``topk <= 0``
    (``dsa_bwd_sm100.py``). The SM90 kernel has no such guard: for ``topK == 0``
    it computes ``n_block = n_block_max - 1 = -1`` and still runs the first-block
    load, reading top-k/KV from a negative row -> OOB/NaN
    (``dsa_bwd_sm90.py``). (Interior ``-1`` inside ``[0, topk_length)`` is a
    non-issue after our order-preserving compaction, which removes them on both
    archs; the empty-row exit is the remaining arch difference.)

    So a compacted ``topk_length`` is only safe on SM100. SM90 callers must keep
    ``topk_length=None`` (the guarded full-width path, which masks ``-1`` and
    zeroes empty rows). Return ``major >= 10``.
    """
    major, _ = paddle.device.cuda.get_device_capability()
    return major >= 10


class CSASparseAttention(paddle.autograd.PyLayer):
    _lse_indexer = None

    @staticmethod
    def forward(
        ctx,
        query,
        kv_full,
        attn_sink,
        topk_idxs,
        softmax_scale,
        backend,
        topk_length=None,
        indexer_topk=0,
        global_kv_idx_remap_fusion=False,
        topk_idxs_compacted=False,
    ):
        from paddlefleet.fusions.csa_sparse_attn_utils import prepare_inputs

        b, sq, np_heads, hn = query.shape
        ctx.query_shape = (b, sq, np_heads, hn)
        ctx.softmax_scale = float(softmax_scale)
        ctx.attn_sink_dtype = attn_sink.dtype
        ctx.backend = backend
        ctx.global_kv_idx_remap_fusion = global_kv_idx_remap_fusion
        # ``topk_length`` is a forward-only early-stop hint: correctness comes
        # from the ``-1`` padding in ``topk_idxs``, which backward already turns
        # into its own bound via ``_csa_compute_topk_length``. Only remember
        # whether it was passed, so backward can return the matching number of
        # placeholder gradients.
        ctx.has_topk_length = topk_length is not None
        # Paddle PyLayer requires None for stop_gradient inputs; record here.
        # In phase 2 (``train_indexer_only``) attn_sink is a frozen backbone
        # parameter while query/kv_full still carry activation gradients.
        ctx.query_needs_grad = not query.stop_gradient
        ctx.kv_full_needs_grad = not kv_full.stop_gradient
        ctx.attn_sink_needs_grad = not attn_sink.stop_gradient

        query, kv_full, attn_sink, topk_idxs = prepare_inputs(
            query,
            kv_full,
            attn_sink,
            topk_idxs,
        )

        # Compact the indices up front for the no-indexer-loss path (HCA,
        # indexer_topk == 0). Densifying the [window|compressed] layout lets the
        # forward AND backward run hole-free with a true-count early-stop, and the
        # backward never re-sorts (it just recounts). Order within the valid set
        # is irrelevant to attention/dq/dkv, and with no indexer loss there is no
        # lse_indexer column alignment to keep. When indexer_topk > 0 the holey
        # layout is preserved so the fused lse_indexer over the first
        # indexer_topk columns stays valid.
        # Compaction can produce topk_length == 0 for all-invalid rows, which only
        # SM100 early-exits safely (SM90 would gather from a negative KV row); so
        # only compact on SM100. SM90 keeps topk_length=None (guarded full-width).
        #
        # ``ctx.compacted_idxs`` is what lets the BACKWARD take the compact
        # KV-load path, which is unguarded against interior ``-1``. So it may
        # only be set when the indices are KNOWN hole-free: either this forward
        # compacted them, or the caller stated it already did
        # (``topk_idxs_compacted``). A caller that supplies ``topk_length``
        # without that promise keeps its own (possibly holey) layout, so the
        # backward falls back to the guarded full-width path instead of
        # gathering ``mKV[-1]`` -- correct, just without the early-stop.
        ctx.compacted_idxs = (
            backend == "cudnn"
            and indexer_topk == 0
            and _csa_bwd_honours_topk_length_holes()
            and (topk_length is None or topk_idxs_compacted)
        )
        if ctx.compacted_idxs and topk_length is None:
            # Fallback densify: the caller (HCA) normally pre-compacts once per
            # batch and passes topk_length, in which case topk_idxs is already
            # dense and we skip the sort. This handles paths that did not
            # pre-compact (e.g. CP without cached docmask metadata).
            topk_idxs, topk_length = _csa_compact_topk_idxs(topk_idxs)

        # Heads the kernels are driven with. The FlashMLA forward only has 64-
        # and 128-head tiles, so a smaller layer is zero-padded up to one; the
        # backward needs the same padding unless the head count already is a
        # multiple of the kernel's 64-head tile (see
        # ``_dsa_bwd_runs_sub_tile_heads``). ``ctx.bwd_heads`` records which
        # width the tensors saved below are in.
        head_tile = _dsa_head_tile(np_heads) if backend == "cudnn" else np_heads
        ctx.bwd_heads = (
            np_heads
            if head_tile == np_heads or _dsa_bwd_runs_sub_tile_heads(np_heads)
            else head_tile
        )
        # Latent width the kernels are driven with. Both the FlashMLA forward
        # and the cuDNN backward only exist at 512, so a narrower layer is
        # zero-padded up to it on the way in and dropped again on the way out
        # (fused with the head-row drop, see ``_drop_padded_rows``). Unlike the
        # head padding above this never changes what is saved for backward, so
        # it costs no activation memory.
        ctx.kernel_hn = _dsa_latent_dim(hn) if backend == "cudnn" else hn
        if backend == "cudnn":
            from paddlefleet.cudnn_ops.attn.csa_sparse_attn_fwd_cudnn import (
                flash_mla_sparse_attn,
            )

            q_in, sink_in = query, attn_sink
            if head_tile != np_heads:
                q_in, sink_in = _pad_query_heads(query, attn_sink, head_tile)

            output, lse, lse_indexer = flash_mla_sparse_attn(
                _pad_latent_dim(q_in, ctx.kernel_hn),
                _pad_latent_dim(kv_full, ctx.kernel_hn),
                sink_in,
                topk_idxs,
                sm_scale=ctx.softmax_scale,
                topk_length=topk_length,
                indexer_topk=indexer_topk,
                global_kv_idx_remap_fusion=global_kv_idx_remap_fusion,
            )
            if head_tile != np_heads:
                lse_real = lse[:, :, :np_heads].contiguous()
                if lse_indexer is not None:
                    lse_indexer = lse_indexer[:, :, :np_heads].contiguous()
            else:
                lse_real = lse
            if ctx.bwd_heads == np_heads:
                # The backward runs on the real heads, so nothing padded has to
                # stay alive: one gather takes the kernel output straight to the
                # real shape (see ``_drop_padded_rows``).
                output_real = _drop_padded_rows(
                    output, np_heads, head_tile, hn
                ).reshape([b, sq, np_heads, hn])
                save_q, save_sink = query, attn_sink
                save_out, save_lse = output_real, lse_real
            else:
                # Padded backward: the saved ``out`` must stay one head tile
                # wide, so only the padded latent columns come off it, and
                # the head rows are dropped from that narrower copy.
                output = _drop_padded_rows(output, head_tile, head_tile, hn)
                output_real = _drop_padded_rows(
                    output, np_heads, head_tile, hn
                ).reshape([b, sq, np_heads, hn])
                save_q, save_sink = q_in, sink_in
                save_out, save_lse = output, lse
            CSASparseAttention._lse_indexer = lse_indexer
        else:
            if topk_length is not None:
                raise NotImplementedError(
                    "topk_length is only supported by the 'cudnn' and "
                    f"'unfused' backends, got backend={backend!r}."
                )
            from paddlefleet.tilelang_ops.attn.sparse_mqa import sparse_attn

            output, lse = sparse_attn(
                query,
                kv_full,
                attn_sink,
                topk_idxs,
                sm_scale=ctx.softmax_scale,
            )
            output_real = output
            save_q, save_sink, save_out, save_lse = (
                query,
                attn_sink,
                output,
                lse,
            )
        ctx.save_for_backward(
            save_q, kv_full, save_sink, topk_idxs, save_out, save_lse
        )
        return output_real.reshape([b, sq, np_heads * hn])

    @staticmethod
    def backward(ctx, grad_output):
        query, kv_full, attn_sink, topk_idxs, output, lse = ctx.saved_tensor()
        b, sq, np_heads, hn = ctx.query_shape
        # Kernel-side head count: equal to ``np_heads`` except on the padded
        # path, where the saved tensors are one head tile wide.
        kh = ctx.bwd_heads

        if ctx.backend == "cudnn":
            from paddlefleet.cudnn_ops import csa_sparse_attn_bwd_cudnn
            from paddlefleet.fusions.csa_sparse_attn_utils import (
                local_to_global_flat,
            )

            _, s_kv, dkv_dim = kv_full.shape

            if kh != np_heads:
                # Padded path: widen the incoming gradient to the saved
                # tensors' head tile. The padded rows get a zero gradient, so
                # they add nothing to ``dkv`` / ``d_sink``.
                grad_output = paddle.concat(
                    [
                        grad_output.reshape([b, sq, np_heads, hn]),
                        paddle.zeros(
                            [b, sq, kh - np_heads, hn], dtype=grad_output.dtype
                        ),
                    ],
                    axis=2,
                )

            q_flat = query.reshape([b * sq, kh, hn])
            o_flat = output.reshape([b * sq, kh, hn])
            do_flat = grad_output.reshape([b * sq, kh, hn])
            kv_flat = kv_full.reshape([b * s_kv, dkv_dim])
            lse_flat = lse.reshape([b * sq, kh])
            topk_idxs_flat = local_to_global_flat(
                topk_idxs, s_kv, fused=ctx.global_kv_idx_remap_fusion
            )

            if ctx.kernel_hn != hn:
                # Same exact zero-padding the forward used (see
                # ``_pad_latent_dim``); ``out`` is zero over the padded columns
                # there, so re-padding the saved narrow copy reproduces it.
                q_flat = _pad_latent_dim(q_flat, ctx.kernel_hn)
                o_flat = _pad_latent_dim(o_flat, ctx.kernel_hn)
                do_flat = _pad_latent_dim(do_flat, ctx.kernel_hn)
                kv_flat = _pad_latent_dim(kv_flat, ctx.kernel_hn)

            # ``topk_idxs`` was already densified in the forward for the compacted
            # path (HCA, no indexer loss), so the saved indices have a dense valid
            # prefix with -1 only trailing. Recover the per-row count cheaply (a
            # sum, NO re-sort) and pass it -- the compact KV-load path is then
            # hole-free and early-stops. If the forward did not compact
            # (indexer_topk>0, holey layout kept for the fused lse_indexer), pass
            # None to take the guarded full-width path.
            if ctx.compacted_idxs:
                topk_length = (
                    (topk_idxs_flat >= 0)
                    .astype("int32")
                    .sum(-1)
                    .astype("int32")
                )
            else:
                topk_length = None

            dq_flat, dkv_flat, d_sink = csa_sparse_attn_bwd_cudnn(
                q_flat,
                kv_flat,
                o_flat,
                do_flat,
                lse_flat,
                attn_sink,
                topk_idxs_flat,
                softmax_scale=ctx.softmax_scale,
                topk_length=topk_length,
            )
            if ctx.kernel_hn != hn:
                # ``dkv`` is exactly zero over the padded columns. It is one
                # latent head (no head padding to undo) and ~8 MiB, so a slice
                # is fine here.
                dkv_flat = dkv_flat[..., :hn].contiguous()
            dkv = dkv_flat.reshape(kv_full.shape)
            # ``dq`` is exactly zero over the padded latent columns and its
            # padded head rows are unused, so one gather drops both.
            dq = _drop_padded_rows(dq_flat, np_heads, kh, hn).reshape(
                [b, sq, np_heads, hn]
            )
            if kh != np_heads:
                d_sink = d_sink[:np_heads]
            d_attn_sink = d_sink.reshape([np_heads]).cast(ctx.attn_sink_dtype)
        else:
            from paddlefleet.tilelang_ops.attn import sparse_mqa_bwd

            grad_output = grad_output.reshape([b, sq, np_heads, hn])
            dq, dkv, d_attn_sink = sparse_mqa_bwd.sparse_mqa_bwd_interface(
                query,
                kv_full,
                attn_sink,
                output,
                grad_output,
                topk_idxs,
                lse,
                ctx.softmax_scale,
            )
            dq = dq.reshape(query.shape)
            dkv = dkv.reshape(kv_full.shape)
            d_attn_sink = d_attn_sink.reshape(attn_sink.shape).cast(
                ctx.attn_sink_dtype
            )

        grads = (
            dq if ctx.query_needs_grad else None,
            dkv if ctx.kv_full_needs_grad else None,
            d_attn_sink if ctx.attn_sink_needs_grad else None,
            None,
        )
        if ctx.has_topk_length:
            # ``topk_length`` was passed as an extra (non-differentiable) tensor
            # input, so an extra placeholder gradient is required.
            grads = (*grads, None)
        return grads


def csa_sparse_attn(
    query,
    kv_full,
    attn_sink,
    topk_idxs,
    softmax_scale,
    backend="tilelang",
    topk_length=None,
    indexer_topk=0,
    global_kv_idx_remap_fusion=False,
    topk_idxs_compacted=False,
):
    """Unified CSA sparse attention entry point.

    Args:
        backend: one of {"unfused", "tilelang", "cudnn"}.
        topk_length: optional ``[b, sq]`` int32 valid prefix length per query
            row. Only the "cudnn" and "unfused" backends support it; it lets
            the kernel stop early instead of walking all ``topk`` slots, which
            is what makes the full-causal MQA layers affordable.
        topk_idxs_compacted: assert that ``topk_idxs`` has its valid entries in
            a contiguous prefix with no interior ``-1`` (what
            ``_csa_compact_topk_idxs`` produces). Only consulted together with
            ``topk_length`` on the "cudnn" backend, where it lets the backward
            keep the compact KV-load path; that path gathers ``mKV[-1]`` on an
            interior hole, so it defaults to False and the backward then takes
            the guarded full-width path instead. When ``topk_length`` is None
            this layer compacts on its own and the flag is irrelevant.
        global_kv_idx_remap_fusion: use the fused Triton local->global KV
            column index remap instead of the eager elementwise chain.
            Bit-identical either way; wired from the
            ``sparse_attn_global_kv_idx_remap_fusion`` config field. Only the
            "cudnn" backend builds that table, so it is ignored otherwise.

    ``query`` may carry any head count up to 128 on the "cudnn" backend; counts
    that are not one of the kernel's head tiles (e.g. 32 or 24) are handled
    inside ``CSASparseAttention`` by padding the forward, which is exact but
    keeps the forward cost of the tile. The same holds for the latent width:
    any ``v_head_dim`` up to 512 is accepted and zero-padded to 512, which is
    the only width both the FlashMLA forward and the cuDNN backward exist at.
    """
    if backend == "unfused":
        return unfused_compressed_sparse_attn(
            query,
            kv_full,
            attn_sink,
            topk_idxs,
            softmax_scale,
            topk_length=topk_length,
        )
    if backend not in ("tilelang", "cudnn"):
        raise ValueError(
            f"csa_sparse_attn_backend={backend!r} is invalid. "
            "Must be one of {'unfused', 'tilelang', 'cudnn'}."
        )
    output = CSASparseAttention.apply(
        query,
        kv_full,
        attn_sink,
        topk_idxs,
        softmax_scale,
        backend,
        topk_length,
        indexer_topk,
        global_kv_idx_remap_fusion,
        topk_idxs_compacted,
    )
    if CSASparseAttention._lse_indexer is not None:
        lse_indexer = CSASparseAttention._lse_indexer
        CSASparseAttention._lse_indexer = None
        return output, lse_indexer
    return output
