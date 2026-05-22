# ruff: noqa
# TileLang fused forward kernel for DeepSeek V4 CSA compressed indexer.
#
# This kernel produces compressed top-k indices and top-k softmax probabilities
# directly from IndexQ/IndexKComp/Weights without materializing the full
# [B, S, S_comp] logits tensor. It follows the Megatron-DSA streaming top-k
# pattern, adapted to V4 CSA compressed-key semantics and BSHD/BSD layout.
#
# topk_effective semantics (controlled by caller, not by this kernel):
#   - Phase 2 (sparse warmup, dsa_indexer_use_sparse_loss=False):
#       topk_effective = n_compressed = floor(S / ratio).
#       The selected set covers the full causal compressed range, enabling
#       full-range KL loss equivalent to DSA dense warm-up.
#   - Phase 3 (sparse, dsa_indexer_use_sparse_loss=True):
#       topk_effective = min(index_topk, n_compressed), typically 512.
#       Standard selected-topk semantics for sparse training.
#   - Phase 1 (csa_dense_mode=True): this kernel is never called.
#
# Padding: topk_effective is internally padded to the next power-of-2 that is
# also divisible by block_K (for bitonic sort alignment). Padded slots in the
# output are filled with -1 (indices) and 0.0 (scores). The caller receives
# only the first topk_effective columns (padding is stripped).

import math

from paddlefleet.ops.tilelang_dsv4.compat import enable_tilelang_paddle_compat_before_import

enable_tilelang_paddle_compat_before_import()

import paddle
import tilelang
import torch
from tilelang import language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def tl_csa_indexer_topk_fwd_impl(
    heads: int,
    dim: int,
    topk: int,
    ratio: int,
    block_K: int = 32,
    dtype: str = "bfloat16",
    num_stages: int = 0,
    num_threads: int = 128,
):
    assert topk == tilelang.math.next_power_of_2(topk)
    assert topk % block_K == 0
    assert heads <= 64 and heads % 8 == 0
    assert num_stages == 0

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_comp = T.dynamic("seq_len_comp")

    INT32 = "int32"
    FP32 = "float"

    index_q_shape = [batch, seq_len, heads, dim]
    weights_shape = [batch, seq_len, heads]
    index_k_shape = [batch, seq_len_comp, dim]
    topk_indices_shape = [batch, seq_len, topk]
    topk_scores_shape = [batch, seq_len, topk]

    N = 2 * topk
    num_iters = int(round(math.log2(N)))
    sm_scale = dim**-0.5

    @T.macro
    def bitonic_sort(
        topk_index_shared: T.SharedBuffer([N], dtype=INT32),
        topk_value_shared: T.SharedBuffer([N], dtype=FP32),
    ):
        T.sync_threads()
        for i1 in T.serial(num_iters):
            for i2 in T.serial(i1 + 1):
                for i in T.Parallel(N):
                    ascending = (i & (1 << (i1 + 1))) != 0
                    j = i ^ (1 << (i1 - i2))
                    if i < j and (
                        (ascending and topk_value_shared[i] > topk_value_shared[j])
                        or (not ascending and topk_value_shared[i] < topk_value_shared[j])
                    ):
                        val = topk_value_shared[i]
                        topk_value_shared[i] = topk_value_shared[j]
                        topk_value_shared[j] = val
                        idx = topk_index_shared[i]
                        topk_index_shared[i] = topk_index_shared[j]
                        topk_index_shared[j] = idx
                T.sync_threads()

    @T.prim_func
    def tl_csa_indexer_topk_fwd_kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),
        IndexKComp: T.Tensor(index_k_shape, dtype),
        Weights: T.Tensor(weights_shape, FP32),
        TopkIndices: T.Tensor(topk_indices_shape, INT32),
        TopkScores: T.Tensor(topk_scores_shape, FP32),
    ):
        with T.Kernel(seq_len, batch, threads=num_threads) as (bx, by):
            i_t = bx
            i_b = by
            valid_end = T.min((i_t + 1) // ratio, seq_len_comp)

            topk_index_shared = T.alloc_shared([N], dtype=INT32)
            topk_value_shared = T.alloc_shared([N], dtype=FP32)
            T.fill(topk_index_shared, -1)
            T.fill(topk_value_shared, float("-inf"))
            T.sync_threads()

            index_q_shared = T.alloc_shared([heads, dim], dtype=dtype)
            T.copy(IndexQ[i_b, i_t, :, :], index_q_shared)
            T.sync_threads()

            weights_shared = T.alloc_shared([heads], dtype=FP32)
            T.copy(Weights[i_b, i_t, :], weights_shared)
            T.sync_threads()

            for i, j in T.Parallel(heads, dim):
                index_q_shared[i, j] = index_q_shared[i, j] * sm_scale
            T.sync_threads()

            num_blocks = T.ceildiv(valid_end, block_K)
            for bk_i in T.Pipelined(num_blocks, num_stages=num_stages):
                k_st = bk_i * block_K
                k_ed = T.min((bk_i + 1) * block_K, valid_end)

                index_k_shared = T.alloc_shared([block_K, dim], dtype=dtype)
                for i, j in T.Parallel(block_K, dim):
                    index_k_shared[i, j] = T.if_then_else(
                        k_st + i < k_ed, IndexKComp[i_b, k_st + i, j], 0
                    )
                T.sync_threads()

                logits = T.alloc_fragment((block_K, heads), FP32)
                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    logits,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=True,
                )
                T.sync_threads()

                for i, j in T.Parallel(block_K, heads):
                    logits[i, j] = T.max(logits[i, j], 0) * weights_shared[j]
                T.sync_threads()

                logits_sum = T.alloc_fragment(block_K, FP32)
                T.reduce_sum(logits, logits_sum, dim=1)
                T.sync_threads()

                offset = T.alloc_var(INT32)
                if k_st >= topk:
                    offset = topk + (k_st % topk)
                else:
                    offset = k_st
                T.sync_threads()

                for i in T.Parallel(block_K):
                    if k_st + i >= valid_end:
                        logits_sum[i] = float("-inf")
                    j = offset + i
                    topk_index_shared[j] = T.if_then_else(k_st + i < valid_end, k_st + i, -1)
                    topk_value_shared[j] = logits_sum[i]
                T.sync_threads()

                if k_ed > topk and k_ed % topk == 0:
                    bitonic_sort(topk_index_shared, topk_value_shared)

            bitonic_sort(topk_index_shared, topk_value_shared)

            logits_max_frag = T.alloc_fragment([1], dtype=FP32)
            logits_frag = T.alloc_fragment([topk], dtype=FP32)
            scores_shared = T.alloc_shared([topk], dtype=FP32)

            T.copy(topk_value_shared[:topk], logits_frag)
            T.sync_threads()
            T.reduce_max(logits_frag, logits_max_frag, dim=-1)
            T.sync_threads()

            for i in T.Parallel(topk):
                logits_frag[i] = T.if_then_else(
                    topk_index_shared[i] >= 0,
                    T.exp(logits_frag[i] - logits_max_frag[0]),
                    0,
                )
            T.sync_threads()

            lse_frag = T.alloc_fragment([1], dtype=FP32)
            T.reduce_sum(logits_frag, lse_frag)
            T.sync_threads()

            for i in T.Parallel(topk):
                scores_shared[i] = T.if_then_else(
                    topk_index_shared[i] >= 0,
                    logits_frag[i] / lse_frag[0],
                    0,
                )
            T.sync_threads()

            T.copy(topk_index_shared[:topk], TopkIndices[i_b, i_t, :])
            T.copy(scores_shared[:topk], TopkScores[i_b, i_t, :])

    return tl_csa_indexer_topk_fwd_kernel


def _next_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _pad_topk_output(indices: torch.Tensor, scores: torch.Tensor, topk: int):
    if indices.shape[-1] == topk:
        return indices, scores
    return indices[..., :topk].contiguous(), scores[..., :topk].contiguous()


def csa_indexer_topk_fwd_interface(
    index_q: torch.Tensor,
    index_k_comp: torch.Tensor,
    weights: torch.Tensor,
    ratio: int,
    topk_effective: int,
    block_K: int = 32,
    num_stages: int = 0,
    num_threads: int = 128,
):
    """Run V4 CSA fused compressed indexer forward.

    Args:
        index_q: [B, S, H_i, D_i] bf16/fp16, BSHD layout.
        index_k_comp: [B, S_comp, D_i] bf16/fp16, BSD layout.
        weights: [B, S, H_i] fp32 or castable to fp32.
        ratio: compression ratio. Valid compressed range for query t is
            [0, (t + 1) // ratio).
        topk_effective: requested output top-k. Phase 2 may set this to
            S_comp; Phase 3 usually sets this to dsa_indexer_topk.

    Returns:
        topk_indices: [B, S, topk_effective] int32, invalid slots are -1.
        topk_scores: [B, S, topk_effective] fp32 top-k softmax probabilities.
    """
    assert index_q.is_contiguous()
    assert index_k_comp.is_contiguous()
    assert weights.is_contiguous()
    assert index_q.ndim == 4
    assert index_k_comp.ndim == 3
    assert weights.ndim == 3

    batch, seq_len, heads, dim = index_q.shape
    batch_k, seq_len_comp, dim_k = index_k_comp.shape
    batch_w, seq_len_w, heads_w = weights.shape
    assert batch == batch_k == batch_w
    assert seq_len == seq_len_w
    assert heads == heads_w
    assert dim == dim_k
    assert topk_effective > 0

    padded_topk = _next_power_of_2(topk_effective)
    if padded_topk % block_K != 0:
        padded_topk = ((padded_topk + block_K - 1) // block_K) * block_K
        padded_topk = _next_power_of_2(padded_topk)

    kernel = tl_csa_indexer_topk_fwd_impl(
        heads=heads,
        dim=dim,
        topk=padded_topk,
        ratio=ratio,
        block_K=block_K,
        dtype="bfloat16",
        num_stages=num_stages,
        num_threads=num_threads,
    )

    if isinstance(index_q, paddle.Tensor):
        topk_indices = paddle.empty(
            [batch, seq_len, padded_topk], dtype="int32"
        )
        topk_scores = paddle.empty(
            [batch, seq_len, padded_topk], dtype="float32"
        )
    else:
        topk_indices = torch.empty(
            (batch, seq_len, padded_topk), device=index_q.device, dtype=torch.int32
        )
        topk_scores = torch.empty(
            (batch, seq_len, padded_topk), device=index_q.device, dtype=torch.float32
        )

    kernel(
        index_q,
        index_k_comp,
        weights.float().contiguous(),
        topk_indices,
        topk_scores,
    )

    return _pad_topk_output(topk_indices, topk_scores, topk_effective)
