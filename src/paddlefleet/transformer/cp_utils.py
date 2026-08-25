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
Context Parallelism (CP) communication primitives and index utilities.

Contiguous CP layout: rank r holds global positions [r*sq_local, (r+1)*sq_local).
All-gather along seq dim produces natural global order; reduce-scatter inverts it.

This module has no internal dependencies on other paddlefleet.transformer modules,
so it can be safely imported by both csa_attention.py without circular imports.
"""

from __future__ import annotations

import paddle
import paddle.distributed as dist
from paddle import Tensor
from paddle.autograd.py_layer import PyLayer

# ===========================================================================
# Differentiable all-gather — delegates to ContextParallelAllGatherOp
# ===========================================================================
from paddlefleet.context_parallel_utils import ContextParallelAllGatherOp


def all_gather_cp(x: Tensor, dim: int, group) -> Tensor:
    """Differentiable all-gather for contiguous CP.

    Delegates to ContextParallelAllGatherOp (mode='contiguous_allgather'),
    which uses NCCL reduce_scatter in backward for axis=0 (more efficient).
    """
    if group is None or group.nranks <= 1:
        return x
    return ContextParallelAllGatherOp.apply(x, dim, "contiguous_allgather")


# ===========================================================================
# One-hop window exchange with the previous CP rank
# ===========================================================================


def wait(ops):
    """Post a batch of P2P ops and wait for them."""
    for task in dist.batch_isend_irecv(ops):
        task.wait()


class PrependPrevWindow(PyLayer):
    """Prepend the previous CP rank's last ``window`` rows; rank 0 gets zeros.

    Only ``window`` rows cross the wire, versus ``s_local`` for an all-gather.
    Backward is the mirror: the prefix gradient goes back to ``rank - 1`` and
    the one arriving from ``rank + 1`` is added to this rank's tail.
    """

    @staticmethod
    def forward(ctx, x, window, group):
        ctx.window, ctx.group = window, group
        r, peers = group.rank, group.ranks
        prefix = paddle.zeros([x.shape[0], window, x.shape[2]], dtype=x.dtype)
        ops = []
        if r > 0:
            ops.append(dist.P2POp(dist.irecv, prefix, peers[r - 1], group))
        if r < group.nranks - 1:
            tail = x[:, -window:, :].contiguous()
            ops.append(dist.P2POp(dist.isend, tail, peers[r + 1], group))
        wait(ops)
        return paddle.concat([prefix, x], axis=1)

    @staticmethod
    def backward(ctx, grad):
        window, group = ctx.window, ctx.group
        r, peers = group.rank, group.ranks
        grad_x = grad[:, window:, :].clone()  # clone: grad must stay untouched
        tail_grad = paddle.zeros_like(grad[:, :window, :])
        ops = []
        if r < group.nranks - 1:
            ops.append(dist.P2POp(dist.irecv, tail_grad, peers[r + 1], group))
        if r > 0:
            prefix_grad = grad[:, :window, :].contiguous()
            ops.append(dist.P2POp(dist.isend, prefix_grad, peers[r - 1], group))
        wait(ops)
        grad_x[:, -window:, :] += tail_grad
        return grad_x


def prepend_prev_window(x: Tensor, window: int, group) -> Tensor:
    """``[b, s, d]`` -> ``[b, window + s, d]``, prefixed by the previous rank.

    Row 0 of the result has global position ``rank * s - window``. A single hop
    only reaches ``rank - 1``, so ``window`` must not exceed ``s``.
    """
    if window <= 0:
        return x
    if window > x.shape[1]:
        raise ValueError(
            f"prepend_prev_window window ({window}) exceeds the local sequence "
            f"length ({x.shape[1]}): a single hop only reaches rank - 1, so the "
            "window may not span more than one CP shard"
        )
    if group is None or group.nranks <= 1:
        prefix = paddle.zeros([x.shape[0], window, x.shape[2]], dtype=x.dtype)
        return paddle.concat([prefix, x], axis=1)
    return PrependPrevWindow.apply(x, window, group)


# ===========================================================================
# CP-aware topk index generators
# ===========================================================================


def get_window_topk_idxs_cp(
    q_positions: Tensor,
    window_size: int,
    batch_size: int,
    sq_global: int,
) -> Tensor:
    """Sliding window indices using global q_positions.

    Args:
        q_positions: [sq_local] int64, global positions for this rank.
        window_size: sliding window size.
        batch_size: batch dimension.
        sq_global: global sequence length.

    Returns:
        [batch_size, sq_local, window_size] int32, -1 for invalid slots.
    """
    effective_window = min(window_size, sq_global)
    base = q_positions.unsqueeze(1)  # [sq_local, 1]
    offsets = paddle.arange(effective_window)  # [window_size]
    k_pos = (
        paddle.clip(base - effective_window + 1, min=0) + offsets
    )  # [sq_local, window_size]
    topk_idxs = paddle.where(k_pos > base, paddle.full_like(k_pos, -1), k_pos)
    return topk_idxs.unsqueeze(0).expand([batch_size, -1, -1]).cast("int32")


def get_compress_topk_idxs_cp(
    q_positions: Tensor,
    ratio: int,
    batch_size: int,
    offset: int,
    n_compressed_global: int,
) -> Tensor:
    """Static compressed topk indices using global q_positions (HCA path).

    Args:
        q_positions: [sq_local] global positions.
        ratio: compression ratio.
        batch_size: batch dimension.
        offset: kv_full offset for compressed positions (= sq_global).
        n_compressed_global: total compressed positions globally.

    Returns:
        [batch_size, sq_local, n_compressed_global] int32, -1 for invalid.
    """
    k_group_idx = paddle.arange(n_compressed_global)  # [n_comp]
    q_first_invalid = ((q_positions + 1) // ratio).unsqueeze(1)  # [sq_local, 1]
    invalid_mask = k_group_idx.unsqueeze(0) >= q_first_invalid
    matrix = paddle.where(
        invalid_mask,
        paddle.full([1], -1, dtype="int64"),
        k_group_idx.unsqueeze(0) + offset,
    )
    return matrix.unsqueeze(0).expand([batch_size, -1, -1]).cast("int32")


def map_compressed_topk_to_kv_full_cp(
    topk_indices_compressed: Tensor,
    q_positions: Tensor,
    ratio: int,
    offset: int,
) -> Tensor:
    """Map indexer topk indices to kv_full coordinates with CP-aware causal check.

    Args:
        topk_indices_compressed: [b, sq_local, topk_eff] compressed block ids.
        q_positions: [sq_local] global positions.
        ratio: compression ratio.
        offset: kv_full offset (= sq_global).

    Returns:
        [b, sq_local, topk_eff] int32 indices into kv_full, -1 for invalid.
    """
    n_valid = (
        ((q_positions + 1) // ratio)
        .unsqueeze(0)
        .unsqueeze(2)
        .cast(topk_indices_compressed.dtype)
    )  # [1, sq_local, 1], same dtype as input
    valid = (topk_indices_compressed >= 0) & (topk_indices_compressed < n_valid)
    return paddle.where(
        valid,
        topk_indices_compressed + offset,
        paddle.full_like(topk_indices_compressed, -1),
    )


def build_causal_mask_cp(
    q_positions: Tensor,
    n_compressed_global: int,
    ratio: int,
    batch_size: int,
) -> Tensor:
    """Build causal mask for CSA indexer with global positions.

    Args:
        q_positions: [sq_local] global positions.
        n_compressed_global: total compressed positions globally.
        ratio: compression ratio.
        batch_size: batch dimension.

    Returns:
        [batch_size, sq_local, n_compressed_global] float32, -inf for invalid.
    """
    compressed_ids = paddle.arange(n_compressed_global).unsqueeze(
        0
    )  # [1, n_comp]
    q_first_invalid = ((q_positions + 1) // ratio).unsqueeze(1)  # [sq_local, 1]
    mask = paddle.where(
        compressed_ids >= q_first_invalid,
        paddle.full([1], float("-inf"), dtype="float32"),
        paddle.zeros([1], dtype="float32"),
    )  # [sq_local, n_comp]
    return mask.unsqueeze(0).expand([batch_size, -1, -1])


# ===========================================================================
# Dual-chunk (zigzag) row swap — CP load balancing for the causal indexer
# ===========================================================================


def dualchunk_chunk_ids(cp_rank: int, cp_size: int) -> tuple[int, int]:
    """The two ids, out of ``2 * cp_size`` equal chunks, this rank computes.

    Contiguous CP hands rank ``r`` chunks ``(2r, 2r+1)``. This layout keeps
    ``2r`` and takes ``2*cp_size-1-2r`` instead, so the two ids sum to
    ``2*cp_size-1`` on **every** rank. A causal row's candidate count grows
    linearly with its global position, so a constant id sum means constant
    indexer work per rank: at cp16 the 31x spread between rank 0 and rank 15
    collapses to 1x.

    Keeping ``2r`` — rather than the more familiar ``(r, 2*cp_size-1-r)``
    pairing, which balances just as well — is what reduces the exchange to a
    single pairwise swap; see ``dualchunk_swap``.
    """
    lo = 2 * cp_rank
    return lo, 2 * cp_size - 1 - lo


def dualchunk_partner(cp_rank: int, cp_size: int) -> int:
    """Rank holding the chunk this one wants, or ``-1`` when nothing moves.

    Rank ``r`` wants chunk ``2*cp_size-1-2r``, which contiguous CP placed on
    rank ``cp_size-1-r``; that rank symmetrically wants ``2r+1`` from here, so
    the pairing is the involution ``r <-> cp_size-1-r``. Returns ``-1`` for a
    single-rank group and for the self-paired middle rank of an odd group.
    """
    if cp_size <= 1:
        return -1
    partner = cp_size - 1 - cp_rank
    return -1 if partner == cp_rank else partner


def dualchunk_swap(x: Tensor, group, axis: int = 1) -> Tensor:
    """Exchange the second half of ``x`` along ``axis`` with the partner rank.

    In: ``x`` is this rank's contiguous CP shard, i.e. chunks ``(2r, 2r+1)``.
    Out: chunks ``(2r, 2*cp_size-1-2r)`` — see ``dualchunk_chunk_ids``.

    Both ends of a pair give up their odd chunk and want the other's, so this
    is one symmetric pairwise exchange rather than a general all-to-all: each
    rank moves ``x.shape[axis] // 2`` rows in each direction and talks to
    exactly one peer.

    The map is an **involution**, so calling this again undoes it — which is
    why one function serves both directions.

    Not differentiable, deliberately. The MQA indexer forward runs wholly
    under ``paddle.no_grad()`` on detached inputs, and its gradient reaches the
    weights through ``TileLangCSAIndexerLossAutoScaler`` applied to the
    *unpermuted* tensors, so there is no gradient to route back through here.
    """
    if group is None or group.nranks <= 1:
        return x
    partner = dualchunk_partner(group.rank, group.nranks)
    if partner < 0:
        return x

    n = int(x.shape[axis])
    if n % 2 != 0:
        raise ValueError(
            "dualchunk_swap needs an even extent on the swapped axis, got "
            f"{n} on axis {axis} of shape {list(x.shape)}"
        )

    keep, give = paddle.split(x, 2, axis=axis)
    give = give.contiguous()
    take = paddle.empty(give.shape, dtype=give.dtype)

    # ``peer`` is a global rank: ``group.ranks`` maps the group-local index,
    # matching context_parallel_utils.py:1252-1261.
    peer = group.ranks[partner]
    send_op = dist.P2POp(dist.isend, give, peer, group)
    recv_op = dist.P2POp(dist.irecv, take, peer, group)
    # Order by rank so a pair never issues two sends before either recv. NCCL's
    # grouped p2p does not need this, but it costs one branch and removes a
    # hang mode if the ops ever degrade to blocking.
    ops = [send_op, recv_op] if group.rank < partner else [recv_op, send_op]
    for task in dist.batch_isend_irecv(ops):
        task.wait()

    return paddle.concat([keep, take], axis=axis).contiguous()
