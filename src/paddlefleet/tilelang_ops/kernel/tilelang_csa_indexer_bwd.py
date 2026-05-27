# ruff: noqa
# TileLang backward kernel for DeepSeek V4 CSA compressed indexer.
#
# This kernel consumes selected compressed block indices plus OGrad and computes
# gradients for IndexQ/Weights/IndexKComp without materializing full
# [B, S, S_comp] indexer logits. OGrad is the gradient w.r.t. selected indexer
# logits/scores supplied by the loss wrapper.
#
# topk_effective semantics (controlled by caller, not by this kernel):
#   - Phase 2 (sparse warmup, dsa_indexer_use_sparse_loss=False):
#       topk_effective = n_compressed. The backward covers the full compressed
#       candidate range, equivalent to full-range KL gradient.
#   - Phase 3 (sparse, dsa_indexer_use_sparse_loss=True):
#       topk_effective = min(index_topk, n_compressed), typically 512.
#       Backward only covers the selected-topk set.
#   - Phase 1 (csa_dense_mode=True): this kernel is never called.
#
# Padding: topk_effective is internally padded to the next power-of-2 that is
# also divisible by block_I. Padded slots (index == -1) are masked out and
# contribute zero gradient. The caller pads grad_scores with 0 for invalid
# slots before calling this kernel.

import math
import os

import paddle
import tilelang
from tilelang import language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def tl_csa_indexer_bwd_impl(
    heads: int,
    dim: int,
    topk: int,
    block_I: int = 32,
    dtype: str = "bfloat16",
    num_stages: int = 0,
    num_threads: int = 128,
    skip_grad_k_comp: bool = False,
):
    assert num_stages == 0
    assert topk == tilelang.math.next_power_of_2(topk)
    assert topk % block_I == 0
    assert heads <= 64 and heads % 8 == 0

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_comp = T.dynamic("seq_len_comp")

    FP32 = "float"
    INT32 = "int32"
    sm_scale = dim**-0.5

    index_q_shape = [batch, seq_len, heads, dim]
    weights_shape = [batch, seq_len, heads]
    index_k_shape = [batch, seq_len_comp, dim]
    topk_indices_shape = [batch, seq_len, topk]
    grad_scores_shape = [batch, seq_len, topk]

    @T.prim_func
    def tl_csa_indexer_bwd_kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),
        IndexKComp: T.Tensor(index_k_shape, dtype),
        Weights: T.Tensor(weights_shape, FP32),
        TopkIndices: T.Tensor(topk_indices_shape, INT32),
        OGrad: T.Tensor(grad_scores_shape, FP32),
        dIndexQ: T.Tensor(index_q_shape, dtype),
        dWeights: T.Tensor(weights_shape, FP32),
        dIndexKComp: T.Tensor(index_k_shape, FP32),
    ):
        with T.Kernel(seq_len, batch, threads=num_threads) as (bx, by):
            i_t = bx
            i_b = by

            index_q_shared = T.alloc_shared([heads, dim], dtype=dtype)
            index_q_scaled_shared = T.alloc_shared([heads, dim], dtype=dtype)
            weights_shared = T.alloc_shared([heads], dtype=FP32)
            indices_shared = T.alloc_shared([block_I], dtype=INT32)
            grad_shared = T.alloc_shared([block_I], dtype=FP32)
            index_k_shared = T.alloc_shared([block_I, dim], dtype=dtype)

            d_index_q_frag = T.alloc_fragment([heads, dim], dtype=FP32)
            d_weights_frag = T.alloc_fragment([heads], dtype=FP32)
            d_index_k_frag = T.alloc_fragment([block_I, dim], dtype=FP32)
            logits = T.alloc_fragment((block_I, heads), dtype=FP32)
            d_logits_qk = T.alloc_shared((block_I, heads), dtype=FP32)
            d_logits_qk_cast1 = T.alloc_fragment((block_I, heads), dtype=dtype)
            d_logits_qk_cast2 = T.alloc_fragment((block_I, heads), dtype=dtype)

            T.copy(IndexQ[i_b, i_t, :, :], index_q_shared)
            T.copy(Weights[i_b, i_t, :], weights_shared)
            T.sync_threads()

            for i, j in T.Parallel(heads, dim):
                index_q_scaled_shared[i, j] = index_q_shared[i, j] * sm_scale
            T.sync_threads()

            T.fill(d_index_q_frag, 0)
            T.fill(d_weights_frag, 0)
            num_blocks = T.ceildiv(topk, block_I)

            for bi_i in T.Pipelined(num_blocks, num_stages=num_stages):
                for i in T.Parallel(block_I):
                    indices_shared[i] = TopkIndices[i_b, i_t, bi_i * block_I + i]
                    grad_shared[i] = OGrad[i_b, i_t, bi_i * block_I + i]
                T.sync_threads()

                for i, j in T.Parallel(block_I, dim):
                    index_k_shared[i, j] = T.if_then_else(
                        (indices_shared[i] >= 0) & (indices_shared[i] < seq_len_comp),
                        IndexKComp[i_b, indices_shared[i], j],
                        0,
                    )
                T.sync_threads()

                T.gemm(
                    index_k_shared,
                    index_q_scaled_shared,
                    logits,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=True,
                )
                T.sync_threads()

                for i, j in T.Parallel(block_I, heads):
                    logits[i, j] = T.max(logits[i, j], 0)
                T.sync_threads()

                d_weights_i = T.alloc_fragment((block_I, heads), dtype=FP32)
                for i, j in T.Parallel(block_I, heads):
                    d_weights_i[i, j] = T.if_then_else(
                        (indices_shared[i] >= 0) & (indices_shared[i] < seq_len_comp),
                        grad_shared[i] * logits[i, j],
                        0,
                    )
                T.reduce_sum(d_weights_i, d_weights_frag, dim=0, clear=False)

                for i, j in T.Parallel(block_I, heads):
                    d_logits_qk[i, j] = T.if_then_else(
                        ((indices_shared[i] >= 0) & (indices_shared[i] < seq_len_comp)) & (logits[i, j] > 0),
                        grad_shared[i] * weights_shared[j],
                        0,
                    )
                T.sync_threads()

                T.copy(d_logits_qk, d_logits_qk_cast1)
                T.gemm(
                    d_logits_qk_cast1,
                    index_k_shared,
                    d_index_q_frag,
                    transpose_A=True,
                    transpose_B=False,
                    clear_accum=False,
                )

                if not skip_grad_k_comp:
                    T.copy(d_logits_qk, d_logits_qk_cast2)
                    T.gemm(
                        d_logits_qk_cast2,
                        index_q_scaled_shared,
                        d_index_k_frag,
                        transpose_A=False,
                        transpose_B=False,
                        clear_accum=True,
                    )

                    for i, j in T.Parallel(block_I, dim):
                        if (indices_shared[i] >= 0) & (indices_shared[i] < seq_len_comp):
                            T.atomic_add(dIndexKComp[i_b, indices_shared[i], j], d_index_k_frag[i, j])

            for i, j in T.Parallel(heads, dim):
                d_index_q_frag[i, j] = d_index_q_frag[i, j] * sm_scale

            T.copy(d_index_q_frag, dIndexQ[i_b, i_t, :, :])
            T.copy(d_weights_frag, dWeights[i_b, i_t, :])

    return tl_csa_indexer_bwd_kernel


def _deterministic_grad_k_comp_enabled():
    default = os.getenv("DSV4_TILELANG_DETERMINISTIC_BWD", "1")
    return os.getenv("DSV4_TILELANG_CSA_INDEXER_DETERMINISTIC_BWD", default).lower() in {"1", "true", "yes", "on"}


def _deterministic_grad_k_comp_reduction_mode():
    return os.getenv("DSV4_TILELANG_CSA_INDEXER_DETERMINISTIC_BWD_REDUCTION", "tilelang_twostage").lower()


@tilelang.jit(out_idx=[-1])
def tl_csa_grad_k_comp_contrib(
    heads: int,
    dim: int,
    topk: int,
    block_I: int = 32,
    dtype: str = "bfloat16",
    num_stages: int = 0,
    num_threads: int = 128,
):
    assert num_stages == 0
    assert topk == tilelang.math.next_power_of_2(topk)
    assert topk % block_I == 0
    assert heads <= 64 and heads % 8 == 0

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_comp = T.dynamic("seq_len_comp")

    FP32 = "float"
    INT32 = "int32"
    sm_scale = dim**-0.5

    index_q_shape = [batch, seq_len, heads, dim]
    weights_shape = [batch, seq_len, heads]
    index_k_shape = [batch, seq_len_comp, dim]
    topk_indices_shape = [batch, seq_len, topk]
    grad_scores_shape = [batch, seq_len, topk]
    contrib_shape = [batch, seq_len, topk, dim]

    @T.prim_func
    def contrib_kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),
        IndexKComp: T.Tensor(index_k_shape, dtype),
        Weights: T.Tensor(weights_shape, FP32),
        TopkIndices: T.Tensor(topk_indices_shape, INT32),
        OGrad: T.Tensor(grad_scores_shape, FP32),
        Contrib: T.Tensor(contrib_shape, FP32),
    ):
        with T.Kernel(seq_len, batch, T.ceildiv(topk, block_I), threads=num_threads) as (i_t, i_b, bi_i):
            index_q_shared = T.alloc_shared([heads, dim], dtype=dtype)
            index_q_scaled_shared = T.alloc_shared([heads, dim], dtype=dtype)
            weights_shared = T.alloc_shared([heads], dtype=FP32)
            indices_shared = T.alloc_shared([block_I], dtype=INT32)
            grad_shared = T.alloc_shared([block_I], dtype=FP32)
            index_k_shared = T.alloc_shared([block_I, dim], dtype=dtype)
            logits = T.alloc_fragment((block_I, heads), dtype=FP32)
            d_logits_qk = T.alloc_shared((block_I, heads), dtype=FP32)
            d_logits_qk_cast = T.alloc_fragment((block_I, heads), dtype=dtype)
            d_index_k_frag = T.alloc_fragment([block_I, dim], dtype=FP32)

            T.copy(IndexQ[i_b, i_t, :, :], index_q_shared)
            T.copy(Weights[i_b, i_t, :], weights_shared)
            T.sync_threads()

            for i, j in T.Parallel(heads, dim):
                index_q_scaled_shared[i, j] = index_q_shared[i, j] * sm_scale
            T.sync_threads()

            for i in T.Parallel(block_I):
                indices_shared[i] = TopkIndices[i_b, i_t, bi_i * block_I + i]
                grad_shared[i] = OGrad[i_b, i_t, bi_i * block_I + i]
            T.sync_threads()

            for i, j in T.Parallel(block_I, dim):
                index_k_shared[i, j] = T.if_then_else(
                    (indices_shared[i] >= 0) & (indices_shared[i] < seq_len_comp),
                    IndexKComp[i_b, indices_shared[i], j],
                    0,
                )
            T.sync_threads()

            T.gemm(
                index_k_shared,
                index_q_scaled_shared,
                logits,
                transpose_A=False,
                transpose_B=True,
                clear_accum=True,
            )
            T.sync_threads()

            for i, j in T.Parallel(block_I, heads):
                logits[i, j] = T.max(logits[i, j], 0)
            T.sync_threads()

            for i, j in T.Parallel(block_I, heads):
                d_logits_qk[i, j] = T.if_then_else(
                    ((indices_shared[i] >= 0) & (indices_shared[i] < seq_len_comp)) & (logits[i, j] > 0),
                    grad_shared[i] * weights_shared[j],
                    0,
                )
            T.sync_threads()

            T.copy(d_logits_qk, d_logits_qk_cast)
            T.gemm(
                d_logits_qk_cast,
                index_q_scaled_shared,
                d_index_k_frag,
                transpose_A=False,
                transpose_B=False,
                clear_accum=True,
            )

            for i, j in T.Parallel(block_I, dim):
                Contrib[i_b, i_t, bi_i * block_I + i, j] = T.if_then_else(
                    (indices_shared[i] >= 0) & (indices_shared[i] < seq_len_comp),
                    d_index_k_frag[i, j],
                    0,
                )

    return contrib_kernel


@tilelang.jit(out_idx=[-1])
def tl_csa_grad_k_comp_reduce(
    dim: int,
    topk: int,
    block_D: int = 32,
    dtype: str = "bfloat16",
    num_threads: int = 128,
):
    assert dim % block_D == 0

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_comp = T.dynamic("seq_len_comp")

    FP32 = "float"
    INT32 = "int32"
    index_k_shape = [batch, seq_len_comp, dim]
    topk_indices_shape = [batch, seq_len, topk]
    contrib_shape = [batch, seq_len, topk, dim]
    grad_k_shape = [batch, seq_len_comp, dim]

    @T.prim_func
    def reduce_kernel(
        IndexKComp: T.Tensor(index_k_shape, dtype),
        TopkIndices: T.Tensor(topk_indices_shape, INT32),
        Contrib: T.Tensor(contrib_shape, FP32),
        dIndexKComp: T.Tensor(grad_k_shape, FP32),
    ):
        with T.Kernel(seq_len_comp, batch, dim // block_D, threads=num_threads) as (key_i, batch_i, dim_blk):
            acc = T.alloc_fragment([block_D], FP32)
            T.clear(acc)
            for seq_i in T.serial(seq_len):
                for topk_i in T.serial(topk):
                    if TopkIndices[batch_i, seq_i, topk_i] == key_i:
                        for d_i in T.Parallel(block_D):
                            acc[d_i] += Contrib[batch_i, seq_i, topk_i, dim_blk * block_D + d_i]
            T.copy(acc, dIndexKComp[batch_i, key_i, dim_blk * block_D : (dim_blk + 1) * block_D])

    return reduce_kernel


def _tilelang_grad_k_comp_twostage(index_q, index_k_comp, weights, topk_indices, grad_scores):
    _, _, heads, dim = index_q.shape
    topk = topk_indices.shape[-1]
    if dim % 32 != 0:
        raise ValueError(f"DSV4_TILELANG_CSA_INDEXER_DETERMINISTIC_BWD_REDUCTION=tilelang_twostage requires dim divisible by 32, got {dim}")
    contrib_kernel = tl_csa_grad_k_comp_contrib(heads=heads, dim=dim, topk=topk, block_I=min(32, topk))
    reduce_kernel = tl_csa_grad_k_comp_reduce(dim=dim, topk=topk, block_D=min(32, dim))
    contrib = contrib_kernel(
        index_q,
        index_k_comp,
        weights.cast("float32").contiguous(),
        topk_indices,
        grad_scores.cast("float32").contiguous(),
    )
    return reduce_kernel(index_k_comp, topk_indices, contrib)


def _paddle_grad_k_comp_deterministic(index_q, index_k_comp, weights, topk_indices, grad_scores, sm_scale):
    """Deterministic grad_k_comp via Paddle tensor reductions.

    Recomputes the CSA indexer backward grad_k_comp contribution using
    Paddle operations in place of T.atomic_add.  The GEMM inputs are cast
    to bf16 before multiplication and accumulated in fp32, matching the
    TileLang kernel's numerical behavior.  Final reduction over kv slots
    is a fixed-order one-hot/matmul.  This serves as the pure-Paddle
    reference path.
    """
    B, S, _, D = index_q.shape
    _, S_comp, _ = index_k_comp.shape
    topk = topk_indices.shape[-1]
    dtype = index_q.dtype  # bf16

    mask = topk_indices != -1
    safe_idxs = paddle.where(mask, topk_indices, paddle.zeros_like(topk_indices)).cast("int64")

    batch_idx = paddle.arange(B, dtype="int64").reshape([B, 1, 1])
    batch_idx = paddle.expand(batch_idx, [B, S, topk])
    ik_gathered = index_k_comp[batch_idx, safe_idxs]

    iq_bf16 = (index_q.cast("float32") * sm_scale).cast(dtype)
    logits = paddle.einsum("bskd,bshd->bskh", ik_gathered.cast(dtype).cast("float32"), iq_bf16.cast("float32"))
    logits = paddle.nn.functional.relu(logits)

    w_expanded = weights.cast("float32").unsqueeze(2)
    grad_expanded = grad_scores.cast("float32").unsqueeze(-1)
    valid_mask = mask.unsqueeze(-1).cast("float32")
    active_mask = valid_mask * (logits > 0).cast("float32")
    d_logits = grad_expanded * w_expanded * active_mask

    d_index_k = paddle.einsum("bskh,bshd->bskd", d_logits.cast(dtype).cast("float32"), iq_bf16.cast("float32"))
    d_index_k = d_index_k * valid_mask

    contrib = d_index_k.reshape([B, S * topk, D])
    flat_indices = safe_idxs.reshape([B, S * topk])
    flat_valid = mask.reshape([B, S * topk])
    selectors = paddle.nn.functional.one_hot(flat_indices, num_classes=S_comp).cast("float32")
    selectors = selectors * flat_valid.unsqueeze(-1).cast("float32")
    return paddle.matmul(selectors.transpose([0, 2, 1]), contrib)


def _next_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _zeros_like(tensor, dtype=None):
    return paddle.zeros_like(tensor, dtype=dtype)


def _empty_like(tensor, dtype=None):
    return paddle.empty_like(tensor, dtype=dtype)


def _full(shape, fill_value, dtype):
    return paddle.full(shape, fill_value, dtype=dtype)


def _zeros(shape, dtype):
    return paddle.zeros(shape, dtype=dtype)


def _concat(tensors, axis):
    return paddle.concat(tensors, axis=axis)


def csa_indexer_bwd_interface(
    index_q,
    weights,
    index_k_comp,
    topk_indices,
    grad_scores,
    block_I: int = 32,
    num_stages: int = 0,
    num_threads: int = 128,
):
    """Run V4 CSA compressed indexer backward.

    Args:
        index_q: [B, S, H_i, D_i] bf16/fp16, BSHD layout.
        weights: [B, S, H_i] fp32 or castable to fp32.
        index_k_comp: [B, S_comp, D_i] bf16/fp16, BSD layout.
        topk_indices: [B, S, topk_effective] int32, invalid slots are -1.
        grad_scores: [B, S, topk_effective] fp32 OGrad for selected logits.

    Returns:
        grad_q: [B, S, H_i, D_i] same dtype as index_q.
        grad_weights: [B, S, H_i] fp32.
        grad_k_comp: [B, S_comp, D_i] fp32.
    """
    assert index_q.is_contiguous()
    assert weights.is_contiguous()
    assert index_k_comp.is_contiguous()
    assert topk_indices.is_contiguous()
    assert grad_scores.is_contiguous()
    assert index_q.ndim == 4
    assert weights.ndim == 3
    assert index_k_comp.ndim == 3
    assert topk_indices.ndim == 3
    assert grad_scores.ndim == 3

    batch, seq_len, heads, dim = index_q.shape
    batch_w, seq_len_w, heads_w = weights.shape
    batch_k, seq_len_comp, dim_k = index_k_comp.shape
    batch_i, seq_len_i, topk_effective = topk_indices.shape
    batch_g, seq_len_g, topk_g = grad_scores.shape

    assert batch == batch_w == batch_k == batch_i == batch_g
    assert seq_len == seq_len_w == seq_len_i == seq_len_g
    assert heads == heads_w
    assert dim == dim_k
    assert topk_effective == topk_g
    assert topk_effective > 0

    padded_topk = _next_power_of_2(topk_effective)
    if padded_topk % block_I != 0:
        padded_topk = ((padded_topk + block_I - 1) // block_I) * block_I
        padded_topk = _next_power_of_2(padded_topk)

    if padded_topk != topk_effective:
        pad = padded_topk - topk_effective
        topk_pad = _full(
            [batch, seq_len, pad],
            -1,
            topk_indices.dtype,
        )
        grad_pad = _zeros(
            [batch, seq_len, pad],
            grad_scores.dtype,
        )
        topk_indices = _concat([topk_indices, topk_pad], axis=-1).contiguous()
        grad_scores = _concat([grad_scores, grad_pad], axis=-1).contiguous()

    deterministic_grad_k_comp = _deterministic_grad_k_comp_enabled()

    kernel = tl_csa_indexer_bwd_impl(
        heads=heads,
        dim=dim,
        topk=padded_topk,
        block_I=block_I,
        dtype="bfloat16",
        num_stages=num_stages,
        num_threads=num_threads,
        skip_grad_k_comp=deterministic_grad_k_comp,
    )

    grad_q = _empty_like(index_q)
    grad_weights = _empty_like(weights, dtype="float32")
    grad_k_comp = _zeros_like(index_k_comp, dtype="float32")

    kernel(
        index_q,
        index_k_comp,
        weights.cast("float32").contiguous(),
        topk_indices,
        grad_scores.cast("float32").contiguous(),
        grad_q,
        grad_weights,
        grad_k_comp,
    )

    if deterministic_grad_k_comp:
        reduction_mode = _deterministic_grad_k_comp_reduction_mode()
        if reduction_mode == "tilelang_twostage":
            grad_k_comp = _tilelang_grad_k_comp_twostage(index_q, index_k_comp, weights, topk_indices, grad_scores)
        elif reduction_mode == "vectorized":
            grad_k_comp = _paddle_grad_k_comp_deterministic(
                index_q, index_k_comp, weights, topk_indices, grad_scores, dim ** -0.5,
            )
        else:
            raise ValueError(
                f"Unsupported DSV4_TILELANG_CSA_INDEXER_DETERMINISTIC_BWD_REDUCTION={reduction_mode!r}; "
                "expected 'tilelang_twostage' or 'vectorized'."
            )

    return grad_q, grad_weights, grad_k_comp
