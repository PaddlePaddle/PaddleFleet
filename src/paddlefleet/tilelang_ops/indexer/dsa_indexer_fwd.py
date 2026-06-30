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

"""DSA TileLang indexer forward kernels (topk reducesum)."""

import math

import paddle
import paddle.nn.functional as F
from paddle import Tensor

try:
    paddle.enable_compat(scope={"tilelang"})
    import tilelang
    import tilelang.language as T

    HAS_TILELANG = True
except ImportError:
    HAS_TILELANG = False

from .dsa_indexer import _cache_get_or_compile

if HAS_TILELANG:

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        }
    )
    def _tl_indexer_topk_reducesum(
        heads: int,
        dim: int,
        topk: int,
        sm_scale: float | None = None,
        block_K: int = 32,
        dtype: str = "bfloat16",
        num_stages: int = 0,
        num_threads: int = 128,
    ):
        assert topk == tilelang.math.next_power_of_2(topk)
        assert topk % block_K == 0
        assert heads <= 64 and heads % 8 == 0
        assert num_stages == 0

        max_warps = (block_K // 16) * max(heads // 8, 1)
        num_warps = num_threads // 32
        if num_warps > max_warps:
            num_warps = max_warps
            num_threads = num_warps * 32

        batch_plus_one = T.symbolic("batch_plus_one")
        seq_len = T.symbolic("seq_len")

        index_q_shape = [seq_len, heads, dim]
        weights_shape = [seq_len, heads]
        index_k_shape = [seq_len, dim]
        topk_indices_shape = [seq_len, topk]
        reducesum_shape = [seq_len, topk]
        offsets_shape = [batch_plus_one]
        token_indices_shape = [seq_len, 2]

        N = 2 * topk
        num_iters = int(round(math.log2(N)))
        if sm_scale is None:
            sm_scale = dim**-0.5

        @T.prim_func
        def tl_indexer_topk_reducesum_kernel(
            IndexQ: T.Tensor(index_q_shape, dtype),
            Weights: T.Tensor(weights_shape, dtype),
            IndexK: T.Tensor(index_k_shape, dtype),
            TopkIndices: T.Tensor(topk_indices_shape, "int32"),
            ReduceSum: T.Tensor(reducesum_shape, "float"),
            Offsets: T.Tensor(offsets_shape, "int32"),
            TokenIndices: T.Tensor(token_indices_shape, "int32"),
        ):
            with T.Kernel(seq_len, threads=num_threads) as (bx):
                i_b, i_t = TokenIndices[bx, 0], TokenIndices[bx, 1]
                bos, eos = Offsets[i_b], Offsets[i_b + 1]
                num_blocks = T.ceildiv(i_t + 1, block_K)

                topk_index_shared = T.alloc_shared([N], dtype="int32")
                topk_value_shared = T.alloc_shared([N], dtype="float")
                T.fill(topk_index_shared, -1)
                T.fill(topk_value_shared, float("-inf"))
                T.sync_threads()

                index_q_shared = T.alloc_shared([heads, dim], dtype=dtype)
                T.copy(IndexQ[bx, :, :], index_q_shared)
                T.sync_threads()

                weights_frag = T.alloc_shared([heads], dtype=dtype)
                T.copy(Weights[bx, :], weights_frag)
                T.sync_threads()

                for i, j in T.Parallel(heads, dim):
                    index_q_shared[i, j] = index_q_shared[i, j] * sm_scale
                T.sync_threads()

                for bk_i in T.Pipelined(num_blocks, num_stages=num_stages):
                    k_st = bk_i * block_K
                    k_ed = T.min((bk_i + 1) * block_K, eos - bos)

                    index_k_shared = T.alloc_shared([block_K, dim], dtype=dtype)
                    for i, j in T.Parallel(block_K, dim):
                        index_k_shared[i, j] = T.if_then_else(
                            k_st + i < k_ed, IndexK[bos + k_st + i, j], 0
                        )
                    T.sync_threads()

                    logits = T.alloc_fragment((block_K, heads), "float")
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
                        logits[i, j] = T.max(logits[i, j], 0) * weights_frag[j]
                    T.sync_threads()

                    logits_sum = T.alloc_fragment(block_K, "float")
                    T.reduce_sum(logits, logits_sum, dim=1)
                    T.sync_threads()

                    for i in T.Parallel(block_K):
                        if k_st + i > i_t:
                            logits_sum[i] = float("-inf")
                        topk_slot = (
                            (topk + (k_st % topk)) if k_st >= topk else k_st
                        ) + i
                        topk_index_shared[topk_slot] = k_st + i
                        topk_value_shared[topk_slot] = logits_sum[i]
                    T.sync_threads()

                    if k_ed > topk and k_ed % topk == 0:
                        # Bitonic sort
                        for i1 in T.serial(num_iters):
                            for i2 in T.serial(i1 + 1):
                                for i_ in T.Parallel(N):
                                    ascending = (i_ & (1 << (i1 + 1))) != 0
                                    j = i_ ^ (1 << (i1 - i2))
                                    if i_ < j and (
                                        (
                                            ascending
                                            and topk_value_shared[i_]
                                            > topk_value_shared[j]
                                        )
                                        or (
                                            not ascending
                                            and topk_value_shared[i_]
                                            < topk_value_shared[j]
                                        )
                                    ):
                                        val = topk_value_shared[i_]
                                        topk_value_shared[i_] = (
                                            topk_value_shared[j]
                                        )
                                        topk_value_shared[j] = val
                                        idx = topk_index_shared[i_]
                                        topk_index_shared[i_] = (
                                            topk_index_shared[j]
                                        )
                                        topk_index_shared[j] = idx
                                T.sync_threads()

                # Final bitonic sort
                for i1 in T.serial(num_iters):
                    for i2 in T.serial(i1 + 1):
                        for i_ in T.Parallel(N):
                            ascending = (i_ & (1 << (i1 + 1))) != 0
                            j = i_ ^ (1 << (i1 - i2))
                            if i_ < j and (
                                (
                                    ascending
                                    and topk_value_shared[i_]
                                    > topk_value_shared[j]
                                )
                                or (
                                    not ascending
                                    and topk_value_shared[i_]
                                    < topk_value_shared[j]
                                )
                            ):
                                val = topk_value_shared[i_]
                                topk_value_shared[i_] = topk_value_shared[j]
                                topk_value_shared[j] = val
                                idx = topk_index_shared[i_]
                                topk_index_shared[i_] = topk_index_shared[j]
                                topk_index_shared[j] = idx
                        T.sync_threads()

                logits_max_frag = T.alloc_fragment([1], dtype="float")
                logits_frag = T.alloc_fragment([topk], dtype="float")
                reducesum_shared = T.alloc_shared([topk], dtype="float")

                T.copy(topk_value_shared[:topk], logits_frag)
                T.sync_threads()
                T.reduce_max(logits_frag, logits_max_frag, dim=-1)
                T.sync_threads()
                for i in T.Parallel(topk):
                    logits_frag[i] = T.exp(logits_frag[i] - logits_max_frag[0])
                T.sync_threads()
                lse_frag = T.alloc_fragment([1], dtype="float")
                T.reduce_sum(logits_frag, lse_frag)
                T.sync_threads()
                for i in T.Parallel(topk):
                    reducesum_shared[i] = logits_frag[i] / lse_frag[0]
                T.sync_threads()

                for i in T.Parallel(topk):
                    if topk_index_shared[i] > i_t:
                        topk_index_shared[i] = -1
                T.sync_threads()

                T.copy(topk_index_shared[:topk], TopkIndices[bx, :])
                T.copy(reducesum_shared[:topk], ReduceSum[bx, :])

        return tl_indexer_topk_reducesum_kernel

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        }
    )
    def _tl_indexer_full_reducesum(
        heads: int,
        dim: int,
        topk: int,
        sm_scale: float | None = None,
        block_K: int = 32,
        dtype: str = "bfloat16",
        num_stages: int = 0,
        num_threads: int = 128,
    ):
        assert topk == tilelang.math.next_power_of_2(topk)
        assert topk % block_K == 0
        assert heads <= 64 and heads % 8 == 0
        assert num_stages == 0

        max_warps = (block_K // 16) * max(heads // 8, 1)
        num_warps = num_threads // 32
        if num_warps > max_warps:
            num_warps = max_warps
            num_threads = num_warps * 32

        batch_plus_one = T.symbolic("batch_plus_one")
        seq_len = T.symbolic("seq_len")

        index_q_shape = [seq_len, heads, dim]
        weights_shape = [seq_len, heads]
        index_k_shape = [seq_len, dim]
        topk_indices_shape = [seq_len, topk]
        reducesum_shape = [seq_len, topk]
        offsets_shape = [batch_plus_one]
        token_indices_shape = [seq_len, 2]

        if sm_scale is None:
            sm_scale = dim**-0.5

        @T.prim_func
        def tl_indexer_full_reducesum_kernel(
            IndexQ: T.Tensor(index_q_shape, dtype),
            Weights: T.Tensor(weights_shape, dtype),
            IndexK: T.Tensor(index_k_shape, dtype),
            TopkIndices: T.Tensor(topk_indices_shape, "int32"),
            ReduceSum: T.Tensor(reducesum_shape, "float"),
            Offsets: T.Tensor(offsets_shape, "int32"),
            TokenIndices: T.Tensor(token_indices_shape, "int32"),
        ):
            with T.Kernel(seq_len, threads=num_threads) as (bx):
                i_b, i_t = TokenIndices[bx, 0], TokenIndices[bx, 1]
                bos, eos = Offsets[i_b], Offsets[i_b + 1]
                seq_local_len = eos - bos
                num_blocks = T.ceildiv(topk, block_K)

                index_q_shared = T.alloc_shared([heads, dim], dtype=dtype)
                T.copy(IndexQ[bx, :, :], index_q_shared)
                T.sync_threads()

                weights_frag = T.alloc_shared([heads], dtype=dtype)
                T.copy(Weights[bx, :], weights_frag)
                T.sync_threads()

                for i, j in T.Parallel(heads, dim):
                    index_q_shared[i, j] = index_q_shared[i, j] * sm_scale
                T.sync_threads()

                for bk_i in T.Pipelined(num_blocks, num_stages=num_stages):
                    k_st = bk_i * block_K
                    k_ed = T.min((bk_i + 1) * block_K, seq_local_len)

                    index_k_shared = T.alloc_shared([block_K, dim], dtype=dtype)
                    for i, j in T.Parallel(block_K, dim):
                        index_k_shared[i, j] = T.if_then_else(
                            k_st + i < k_ed, IndexK[bos + k_st + i, j], 0
                        )
                    T.sync_threads()

                    logits = T.alloc_fragment((block_K, heads), "float")
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
                        logits[i, j] = T.max(logits[i, j], 0) * weights_frag[j]
                    T.sync_threads()

                    logits_sum = T.alloc_fragment(block_K, "float")
                    T.reduce_sum(logits, logits_sum, dim=1)
                    T.sync_threads()

                    for i in T.Parallel(block_K):
                        pos = k_st + i
                        valid = (
                            (pos <= i_t) & (pos < seq_local_len) & (pos < topk)
                        )
                        TopkIndices[bx, pos] = T.if_then_else(valid, pos, -1)
                        ReduceSum[bx, pos] = T.if_then_else(
                            valid, logits_sum[i], -T.infinity("float")
                        )
                    T.sync_threads()

        return tl_indexer_full_reducesum_kernel


def dsa_indexer_topk_reducesum_interface(
    q: Tensor,
    weights: Tensor,
    k: Tensor,
    topk: int,
    offsets: Tensor,
    token_indices: Tensor,
    use_full_loss: bool = False,
) -> tuple[Tensor, Tensor]:
    """Indexer topk + softmax. THD format: q [S,H,D], weights [S,H], k [S,D].

    Returns:
        topk_indices: [S, topk] int32
        topk_score: [S, topk] float32
    """
    if not HAS_TILELANG:
        raise RuntimeError(
            "tilelang is required for dsa_indexer_topk_reducesum_interface"
        )

    _, heads, dim = q.shape
    seq_len = q.shape[0]

    def _compile():
        if use_full_loss:
            return _tl_indexer_full_reducesum(
                heads=heads, dim=dim, topk=topk, dtype="bfloat16"
            )
        return _tl_indexer_topk_reducesum(
            heads=heads, dim=dim, topk=topk, dtype="bfloat16"
        )

    kernel_name = "indexer_full" if use_full_loss else "indexer_topk"
    key = (kernel_name, heads, dim, topk)
    kernel = _cache_get_or_compile(kernel_name, key, _compile)

    topk_indices = paddle.zeros([seq_len, topk], dtype="int32")
    topk_score = paddle.zeros([seq_len, topk], dtype="float32")
    q_bf16 = q.cast("bfloat16") if q.dtype != paddle.bfloat16 else q
    weights_bf16 = (
        weights.cast("bfloat16")
        if weights.dtype != paddle.bfloat16
        else weights
    )
    k_bf16 = k.cast("bfloat16") if k.dtype != paddle.bfloat16 else k
    kernel(
        q_bf16,
        weights_bf16,
        k_bf16,
        topk_indices,
        topk_score,
        offsets,
        token_indices,
    )
    if use_full_loss:
        # _tl_indexer_full_reducesum outputs raw weighted scores (not normalized),
        # unlike _tl_indexer_topk_reducesum which has in-kernel softmax.
        # We apply softmax here to produce a proper probability distribution.
        topk_score = F.softmax(topk_score, axis=-1, dtype="float32")
    return topk_indices, topk_score
