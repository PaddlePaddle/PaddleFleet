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

"""DSA TileLang indexer backward kernels."""

import paddle
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
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_WGMMA: True,
        }
    )
    def _tl_indexer_bwd(
        heads: int,
        dim: int,
        topk: int,
        sm_scale: float | None = None,
        block_I: int = 32,
        num_stages: int = 0,
        num_threads: int = 128,
    ):
        assert num_stages == 0
        assert topk == tilelang.math.next_power_of_2(topk)
        assert topk % block_I == 0
        assert heads <= 64 and heads % 8 == 0

        max_warps = (block_I // 16) * max(heads // 8, 1)
        num_warps = num_threads // 32
        if num_warps > max_warps:
            num_warps = max_warps
            num_threads = num_warps * 32

        batch_plus_one = T.symbolic("batch_plus_one")
        seq_len = T.symbolic("seq_len")

        index_q_shape = [seq_len, heads, dim]
        weights_shape = [seq_len, heads]
        index_k_shape = [seq_len, dim]
        shape_p = [seq_len, topk]
        topk_indices_shape = [seq_len, topk]
        offsets_shape = [batch_plus_one]
        token_indices_shape = [seq_len, 2]

        if sm_scale is None:
            sm_scale = dim**-0.5

        @T.prim_func
        def tl_indexer_bwd_kernel(
            IndexQ: T.Tensor(index_q_shape, "bfloat16"),
            Weights: T.Tensor(weights_shape, "bfloat16"),
            IndexK: T.Tensor(index_k_shape, "bfloat16"),
            dIndexQ: T.Tensor(index_q_shape, "bfloat16"),
            dWeights: T.Tensor(weights_shape, "bfloat16"),
            dIndexK: T.Tensor(index_k_shape, "float"),
            AttnScore: T.Tensor(shape_p, "float"),
            IndexScore: T.Tensor(shape_p, "float"),
            TopkIndices: T.Tensor(topk_indices_shape, "int32"),
            Offsets: T.Tensor(offsets_shape, "int32"),
            TokenIndices: T.Tensor(token_indices_shape, "int32"),
        ):
            with T.Kernel(seq_len, threads=num_threads) as (bx):
                i_b, i_t = TokenIndices[bx, 0], TokenIndices[bx, 1]
                bos = Offsets[i_b]
                num_blocks = T.ceildiv(topk, block_I)

                index_q_shared = T.alloc_shared([heads, dim], dtype="bfloat16")
                weights_shared = T.alloc_shared([heads], dtype="bfloat16")
                d_index_q_frag = T.alloc_fragment([heads, dim], dtype="float")
                d_weights_frag = T.alloc_fragment([heads], dtype="float")

                T.copy(IndexQ[bos + i_t, :, :], index_q_shared)
                T.copy(Weights[bos + i_t, :], weights_shared)
                T.fill(d_index_q_frag, 0)
                T.fill(d_weights_frag, 0)

                for i, j in T.Parallel(heads, dim):
                    index_q_shared[i, j] = index_q_shared[i, j] * sm_scale

                for bi_i in T.Pipelined(num_blocks, num_stages=num_stages):
                    i_st = bi_i * block_I
                    i_ed = (bi_i + 1) * block_I

                    indices_shared = T.alloc_shared([block_I], dtype="int32")
                    T.copy(TopkIndices[bos + i_t, i_st:i_ed], indices_shared)

                    index_k_shared = T.alloc_shared(
                        [block_I, dim], dtype="bfloat16"
                    )
                    for i, j in T.Parallel(block_I, dim):
                        pos = indices_shared[i]
                        index_k_shared[i, j] = T.if_then_else(
                            (pos > -1) & (pos <= i_t), IndexK[bos + pos, j], 0
                        )

                    attn_score_shared = T.alloc_shared([block_I], dtype="float")
                    index_score_shared = T.alloc_shared(
                        [block_I], dtype="float"
                    )
                    for i in T.Parallel(block_I):
                        attn_score_shared[i] = AttnScore[bos + i_t, i_st + i]
                        index_score_shared[i] = IndexScore[bos + i_t, i_st + i]

                    logits = T.alloc_fragment((block_I, heads), "float")
                    T.gemm(
                        index_k_shared,
                        index_q_shared,
                        logits,
                        transpose_A=False,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    for i, j in T.Parallel(block_I, heads):
                        logits[i, j] = T.max(logits[i, j], 0)

                    d_weights_i = T.alloc_fragment((block_I, heads), "float")
                    for i, j in T.Parallel(block_I, heads):
                        d_weights_i[i, j] = (
                            (index_score_shared[i] - attn_score_shared[i])
                            * logits[i, j]
                            / seq_len
                        )
                    T.reduce_sum(
                        d_weights_i, d_weights_frag, dim=0, clear=False
                    )

                    d_logits_qk = T.alloc_shared((block_I, heads), "float")
                    d_logits_qk_cast1 = T.alloc_fragment(
                        (block_I, heads), "bfloat16"
                    )
                    d_logits_qk_cast2 = T.alloc_fragment(
                        (block_I, heads), "bfloat16"
                    )

                    for i, j in T.Parallel(block_I, heads):
                        d_relu = 0.0
                        if logits[i, j] > 0:
                            d_relu = 1.0
                        d_logits_qk[i, j] = (
                            (index_score_shared[i] - attn_score_shared[i])
                            * d_relu
                            * weights_shared[j]
                            / seq_len
                        )

                    T.copy(d_logits_qk, d_logits_qk_cast1)
                    T.gemm(
                        d_logits_qk_cast1,
                        index_k_shared,
                        d_index_q_frag,
                        transpose_A=True,
                        transpose_B=False,
                        clear_accum=False,
                    )

                    T.copy(d_logits_qk, d_logits_qk_cast2)
                    d_index_k_frag = T.alloc_fragment(
                        [block_I, dim], dtype="float"
                    )
                    T.gemm(
                        d_logits_qk_cast2,
                        index_q_shared,
                        d_index_k_frag,
                        transpose_A=False,
                        transpose_B=False,
                        clear_accum=True,
                    )

                    for i, j in T.Parallel(block_I, dim):
                        pos = indices_shared[i]
                        if (pos > -1) & (pos <= i_t):
                            T.atomic_add(
                                dIndexK[bos + pos, j], d_index_k_frag[i, j]
                            )

                for i, j in T.Parallel(heads, dim):
                    d_index_q_frag[i, j] = d_index_q_frag[i, j] * sm_scale

                T.copy(d_index_q_frag, dIndexQ[bos + i_t, :, :])
                T.copy(d_weights_frag, dWeights[bos + i_t, :])

        return tl_indexer_bwd_kernel


def dsa_indexer_bwd_interface(
    q: Tensor,
    weights: Tensor,
    k: Tensor,
    attn_score: Tensor,
    index_score: Tensor,
    topk_indices: Tensor,
    offsets: Tensor,
    token_indices: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Indexer backward. THD format.

    Returns:
        dq: [S, heads, dim] bfloat16
        dweights: [S, heads] bfloat16
        dk: [S, dim] bfloat16
    """
    if not HAS_TILELANG:
        raise RuntimeError("tilelang is required for dsa_indexer_bwd_interface")

    _, heads, dim = q.shape
    topk = topk_indices.shape[-1]

    def _compile():
        return _tl_indexer_bwd(heads, dim, topk)

    key = ("indexer_bwd", heads, dim, topk)
    kernel = _cache_get_or_compile("indexer_bwd", key, _compile)

    dq = paddle.zeros_like(q)
    dweights = paddle.zeros_like(weights)
    dk = paddle.zeros([k.shape[0], k.shape[1]], dtype="float32")

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
        dq,
        dweights,
        dk,
        attn_score,
        index_score,
        topk_indices,
        offsets,
        token_indices,
    )
    return dq, dweights, dk.cast(q.dtype)
