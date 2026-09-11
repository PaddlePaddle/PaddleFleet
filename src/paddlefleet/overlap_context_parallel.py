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

"""FlashMask context parallel with the KV communication overlapped in-kernel.

Selected by the ``_overlap`` suffix of ``cp_balance_mode`` and dispatched from
``flashmask_attention_cp``. K/V are handed to the FA-4 kernel LOCAL together with
the CP group: the kernel all-gathers them on an internal stream and folds the
dK/dV reduce-scatter back to the local length, so this layer performs no explicit
communication and only has to express the mask in the gathered KV order.

Kept in a separate module so that the overlap path adds no import cost, and no
merge surface, to the non-overlap one.
"""

import inspect
import os

import paddle
from paddle.autograd.py_layer import PyLayer
from paddle.distributed import fleet

from paddlefleet.context_parallel_utils import (
    preprocess_index,
    preprocess_index_dual_chunks,
)

OVERLAP_SUPPORTED = False
try:
    if (
        paddle.device.cuda.device_count() > 0
        and paddle.device.cuda.get_device_capability(0)[0] == 10
    ):
        from paddlefleet_ops.flash_mask.cute.flashmask_utils import (
            FlashMaskInfoPaddle,
        )
        from paddlefleet_ops.flash_mask.cute.interface import (
            _flash_attn_bwd,
            _flash_attn_fwd,
        )

        # ``group`` is the entry point of the in-kernel overlap.
        OVERLAP_SUPPORTED = (
            "group" in inspect.signature(_flash_attn_fwd).parameters
        )
except (ImportError, AttributeError):
    OVERLAP_SUPPORTED = False


HIERARCHICAL_TRUE = ("1", "true", "on", "yes")
BLOCK_ORDER_CACHE = {}


def hierarchical_gpus_per_node(cp_size):
    """Node size of the hierarchical KV traversal, 0 if it is not in use.

    Mirrors the kernel's own decision, so the mask order cannot disagree with the
    buffer it produces: same parsing as ``OverlapFeatureFlags::parse_bool_env``,
    same single-node fallback as ``hier_is_effective``. The kernel takes its node
    size from the NCCL topology, which ``HIERARCHICAL_GPUS_PER_NODE`` must match.
    """
    if (
        os.environ.get("FLASHMASK_USE_HIERARCHICAL", "").lower()
        not in HIERARCHICAL_TRUE
    ):
        return 0
    gpus_per_node = int(os.environ.get("HIERARCHICAL_GPUS_PER_NODE", "8"))
    return gpus_per_node if cp_size > gpus_per_node else 0


def traversal_rank(logical_pos, rank, cp_size, gpus_per_node):
    """Owner of the chunk at ``logical_pos`` of the gather traversal.

    Circular for ``gpus_per_node == 0``, otherwise the kernel's
    ``hier::hier_target_rank``: our own congruence group (same intra-node slot,
    successive nodes) first, then the groups of the other slots of our node.
    """
    if gpus_per_node == 0:
        return (rank + logical_pos) % cp_size
    num_nodes = cp_size // gpus_per_node
    slot, node = rank % gpus_per_node, rank // gpus_per_node
    if logical_pos < num_nodes:
        return slot + ((node + logical_pos) % num_nodes) * gpus_per_node
    adjusted = logical_pos - num_nodes
    slot += adjusted // num_nodes + 1
    node += adjusted % num_nodes
    return slot % gpus_per_node + (node % num_nodes) * gpus_per_node


def block_order(cp_size, rank, backward, mode):
    """Natural block order of the gathered KV, as a cached index tensor.

    The buffer concatenates whole local chunks along the traversal, whose position
    0 is the local chunk: backward starts there, forward leaves it last by
    rotating the circular traversal by one and by reversing the hierarchical one
    (the kernel's ``hier_seqlen_id``). The balance mode only decides which of the
    ``2 * cp_size`` natural blocks a rank holds -- the mirrored pair
    ``(r, 2*cp_size-1-r)`` under DualChunkSwap, the adjacent pair
    ``(2*r, 2*r+1)`` under contiguous -- so one permutation expresses the balance
    layout and the traversal at once, for both traversal shapes.

    Everything here is per-process constant, hence cached: rebuilding the index
    would cost a blocking host-to-device copy on every call, and reading the
    environment once matches the kernel, which also samples it once.
    """
    key = (cp_size, rank, backward, mode)
    index = BLOCK_ORDER_CACHE.get(key)
    if index is None:
        gpus_per_node = hierarchical_gpus_per_node(cp_size)
        contiguous = mode == "contiguous_allgather_overlap"
        if backward:
            positions = range(cp_size)
        elif gpus_per_node:
            positions = range(cp_size - 1, -1, -1)
        else:
            positions = [*range(1, cp_size), 0]
        order = []
        for logical_pos in positions:
            owner = traversal_rank(logical_pos, rank, cp_size, gpus_per_node)
            if contiguous:
                order += [2 * owner, 2 * owner + 1]
            else:
                order += [owner, 2 * cp_size - 1 - owner]
        index = paddle.to_tensor(order, dtype="int64")
        BLOCK_ORDER_CACHE[key] = index
    return index


def localize_mask(startend_row_indices, seqlen_local, group, mode):
    """Fold the global FlashMask row bounds onto the local query rows.

    Only the values change; axis 2 keeps indexing key positions in natural
    sequence order, which ``gathered_kv_order`` then permutes.
    """
    if mode == "contiguous_allgather_overlap":
        # The local query rows are one contiguous run, so a single rebase does it.
        return preprocess_index(
            startend_row_indices,
            chunk_id=group.rank,
            seq_blocksize=seqlen_local,
            max_seqlen_q=seqlen_local,
        )
    if mode == "dualchunk_allgather_overlap":
        # Two non-adjacent halves, each rebased and the second one offset.
        seq_blocksize = seqlen_local // 2
        return preprocess_index_dual_chunks(
            startend_row_indices,
            chunk_id_first=group.rank,
            chunk_id_second=2 * group.world_size - group.rank - 1,
            seq_blocksize=seq_blocksize,
            max_seqlen_q=seq_blocksize,
        )
    raise ValueError(
        f"Unsupported overlapped FlashMask context parallel mode: {mode}"
    )


def gathered_kv_order(mask, group, backward, mode):
    """Permute the mask key axis into the order the in-kernel all-gather lands."""
    cp_size = group.world_size
    batch_size, _, seqlen, num_vecs = mask.shape
    n_blocks = 2 * cp_size
    return (
        mask.reshape([batch_size, -1, n_blocks, seqlen // n_blocks, num_vecs])
        .index_select(block_order(cp_size, group.rank, backward, mode), axis=2)
        .reshape([batch_size, -1, seqlen, num_vecs])
    )


class OverlappedFlashMaskContextParallel(PyLayer):
    """FlashMask CP attention over an in-kernel overlapped KV all-gather.

    Backward returns map positionally onto the forward TENSOR inputs:
    ``query(0)/key(1)/value(2)/startend_row_indices(3)/learnable_sink(4)``. The
    mask is stop_gradient, so slot 3 must be ``None`` whenever slot 4 is filled;
    when the sink needs no gradient the trailing slots are simply omitted.
    """

    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        startend_row_indices,
        learnable_sink,
        softmax_scale,
        mode,
    ):
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_context_parallel_group()
        mask = localize_mask(startend_row_indices, query.shape[1], group, mode)

        output, log_sum_exp = _flash_attn_fwd(
            query,
            key,
            value,
            startend_row_indices=gathered_kv_order(mask, group, False, mode),
            learnable_sink=learnable_sink,
            causal=False,
            return_lse=True,
            pack_gqa=False,
            softmax_scale=softmax_scale,
            group=group,
        )

        ctx.save_for_backward(query, key, value, output, log_sum_exp, mask)
        ctx.group = group
        ctx.softmax_scale = softmax_scale
        ctx.mode = mode
        ctx.learnable_sink = learnable_sink
        ctx.sink_requires_grad = (
            learnable_sink is not None and not learnable_sink.stop_gradient
        )
        return output

    @staticmethod
    def backward(ctx, output_grad):
        query, key, value, output, log_sum_exp, mask = ctx.saved_tensor()
        group = ctx.group

        flashmask_info = FlashMaskInfoPaddle(
            startend_row_indices=gathered_kv_order(mask, group, True, ctx.mode),
            is_causal=False,
        )
        # dK/dV are reduce-scattered inside the kernel, hence already local.
        # dsink is a parameter gradient over this rank's queries, so it stays a
        # partial that the optimizer's own reduction sums, as in the non-overlap
        # all-gather path.
        query_grad, key_grad, value_grad, sink_grad = _flash_attn_bwd(
            query,
            key,
            value,
            output,
            output_grad,
            log_sum_exp,
            flashmask_info,
            learnable_sink=ctx.learnable_sink,
            causal=False,
            softmax_scale=ctx.softmax_scale,
            deterministic=paddle.get_flags(["FLAGS_cudnn_deterministic"])[
                "FLAGS_cudnn_deterministic"
            ],
            group=group,
        )
        if ctx.sink_requires_grad:
            return query_grad, key_grad, value_grad, None, sink_grad
        return query_grad, key_grad, value_grad


def overlap_flashmask_attention_cp(
    query,
    key,
    value,
    startend_row_indices,
    fixed_seed_offset=None,
    dropout=0.0,
    causal=False,
    training=True,
    learnable_sink=None,
    softmax_scale=None,
    mode="dualchunk_allgather_overlap",
):
    """Overlapped counterpart of ``flashmask_attention_cp``, same signature.

    Everything the overlapped kernel does not implement is rejected here, so the
    layer itself can assume a plain non-causal FlashMask.
    """
    assert OVERLAP_SUPPORTED, (
        "Overlapped FlashMask context parallel requires an SM100 device and a "
        "paddlefleet_ops build whose flash_mask cute interface accepts `group`."
    )
    if dropout > 0.0:
        raise NotImplementedError(
            "Dropout is not supported in overlapped FlashMask context parallel."
        )
    if causal:
        raise NotImplementedError(
            "Overlapped FlashMask context parallel does not support causal."
        )
    if fixed_seed_offset is not None:
        raise NotImplementedError(
            "Fixed seed offset is not supported in overlapped FlashMask "
            "context parallel."
        )
    assert query.shape[1] % 2 == 0, (
        f"Query sequence length must be divisible by 2: the mask is permuted in "
        f"2 * cp_size blocks. Current query sequence length: {query.shape[1]}"
    )
    return OverlappedFlashMaskContextParallel.apply(
        query,
        key,
        value,
        startend_row_indices,
        learnable_sink,
        softmax_scale,
        mode,
    )
