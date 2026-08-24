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
"""

import functools

import paddle
from paddle import Tensor


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
    bound (NOT ``sum(valid)``) so it stays correct when ``-1`` entries are
    interleaved (multi-doc leading/interior holes). Clamped to >=1 so the kernel
    writes every dq row (dq is allocated uninitialized in the backend).

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
        ctx.compacted_idxs = (
            backend == "cudnn"
            and indexer_topk == 0
            and _csa_bwd_honours_topk_length_holes()
        )
        if ctx.compacted_idxs and topk_length is None:
            # Fallback densify: the caller (HCA) normally pre-compacts once per
            # batch and passes topk_length, in which case topk_idxs is already
            # dense and we skip the sort. This handles paths that did not
            # pre-compact (e.g. CP without cached docmask metadata).
            topk_idxs, topk_length = _csa_compact_topk_idxs(topk_idxs)

        if backend == "cudnn":
            from paddlefleet.cudnn_ops.attn.csa_sparse_attn_fwd_cudnn import (
                flash_mla_sparse_attn,
            )

            output, lse, lse_indexer = flash_mla_sparse_attn(
                query,
                kv_full,
                attn_sink,
                topk_idxs,
                sm_scale=ctx.softmax_scale,
                topk_length=topk_length,
                indexer_topk=indexer_topk,
                global_kv_idx_remap_fusion=global_kv_idx_remap_fusion,
            )
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
        ctx.save_for_backward(query, kv_full, attn_sink, topk_idxs, output, lse)
        return output.reshape([b, sq, np_heads * hn])

    @staticmethod
    def backward(ctx, grad_output):
        query, kv_full, attn_sink, topk_idxs, output, lse = ctx.saved_tensor()
        b, sq, np_heads, hn = ctx.query_shape

        if ctx.backend == "cudnn":
            from paddlefleet.cudnn_ops import csa_sparse_attn_bwd_cudnn
            from paddlefleet.fusions.csa_sparse_attn_utils import (
                local_to_global_flat,
            )

            _, s_kv, dkv_dim = kv_full.shape

            q_flat = query.reshape([b * sq, np_heads, hn])
            o_flat = output.reshape([b * sq, np_heads, hn])
            do_flat = grad_output.reshape([b * sq, np_heads, hn])
            kv_flat = kv_full.reshape([b * s_kv, dkv_dim])
            lse_flat = lse.reshape([b * sq, np_heads])
            topk_idxs_flat = local_to_global_flat(
                topk_idxs, s_kv, fused=ctx.global_kv_idx_remap_fusion
            )

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
            dq = dq_flat.reshape(query.shape)
            dkv = dkv_flat.reshape(kv_full.shape)
            d_attn_sink = d_sink.reshape(attn_sink.shape).cast(
                ctx.attn_sink_dtype
            )
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
):
    """Unified CSA sparse attention entry point.

    Args:
        backend: one of {"unfused", "tilelang", "cudnn"}.
        topk_length: optional ``[b, sq]`` int32 valid prefix length per query
            row. Only the "cudnn" and "unfused" backends support it; it lets
            the kernel stop early instead of walking all ``topk`` slots, which
            is what makes the full-causal MQA layers affordable.
        global_kv_idx_remap_fusion: use the fused Triton local->global KV
            column index remap instead of the eager elementwise chain.
            Bit-identical either way; wired from the
            ``sparse_attn_global_kv_idx_remap_fusion`` config field. Only the
            "cudnn" backend builds that table, so it is ignored otherwise.
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
    )
    if CSASparseAttention._lse_indexer is not None:
        lse_indexer = CSASparseAttention._lse_indexer
        CSASparseAttention._lse_indexer = None
        return output, lse_indexer
    return output
