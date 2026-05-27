# ruff: noqa
# Adapted from miles_plugins/models/deepseek_v4/ops/kernel/tilelang_sparse_mla_bwd.py.
# This module is imported only by explicit TileLang DSv4 callers and is not wired
# into PaddleFleet attention dispatch by default.
import os

import paddle
import tilelang
from tilelang import language as T

# Numerical constants for exp2/log2 domain flash-attention
_RECIPROCAL_LOG2 = 1.44269504  # 1 / ln(2)
_LOG2 = 0.6931471805599453     # ln(2)


@tilelang.jit(out_idx=[-1])
def preprocess(
    B,
    S,
    H,
    D,
    block_ND=32,
    num_stages=5,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    shape = [B, S, H, D]

    @T.prim_func
    def preprocess_kernel(
        O: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        Delta: T.Tensor([B, S, H], accum_dtype),
    ):
        with T.Kernel(H, T.ceildiv(S, block_ND), B) as (bx, by, bz):
            o = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            do = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            delta = T.alloc_fragment([block_ND], accum_dtype)
            acc = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            T.clear(acc)
            for k in T.Pipelined(T.ceildiv(D, block_ND), num_stages=num_stages):
                T.copy(O[bz, by * block_ND : (by + 1) * block_ND, bx, k * block_ND : (k + 1) * block_ND], o)
                T.copy(dO[bz, by * block_ND : (by + 1) * block_ND, bx, k * block_ND : (k + 1) * block_ND], do)
                for i, j in T.Parallel(block_ND, block_ND):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, by * block_ND : (by + 1) * block_ND, bx])

    return preprocess_kernel


@tilelang.jit(out_idx=[-1])
def postprocess(
    B,
    S_kv,
    D,
    block_N=64,
    threads=128,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    dkv_shape = [B, S_kv, D]

    @T.prim_func
    def postprocess_kernel(
        dKV: T.Tensor(dkv_shape, accum_dtype),
        dKV_out: T.Tensor(dkv_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(S_kv, block_N), B, threads=threads) as (bx, by):
            T.copy(
                dKV[by, bx * block_N : (bx + 1) * block_N, :],
                dKV_out[by, bx * block_N : (bx + 1) * block_N, :],
            )

    return postprocess_kernel


@tilelang.jit(out_idx=[-1])
def attn_sink_bwd_deterministic(
    B,
    S,
    H,
    block_H=64,
    threads=128,
    accum_dtype=T.float32,
):
    assert accum_dtype == T.float32
    assert H % block_H == 0, f"H ({H}) must be divisible by block_H ({block_H}) for deterministic dAttnSink"
    attn_sink_shape = [H]
    lse_shape = [B, S, H]
    delta_shape = [B, S, H]

    @T.prim_func
    def attn_sink_bwd_det_kernel(
        AttnSink: T.Tensor(attn_sink_shape, accum_dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
        Delta: T.Tensor(delta_shape, accum_dtype),
        dAttnSink: T.Tensor(attn_sink_shape, accum_dtype),
    ):
        with T.Kernel(H // block_H, threads=threads) as (bh):
            acc = T.alloc_fragment([block_H], accum_dtype)
            T.clear(acc)
            for b_i in T.serial(B):
                for s_i in T.serial(S):
                    for h_i in T.Parallel(block_H):
                        acc[h_i] += -Delta[b_i, s_i, bh * block_H + h_i] * T.exp2(
                            AttnSink[bh * block_H + h_i] * _RECIPROCAL_LOG2 - Lse[b_i, s_i, bh * block_H + h_i]
                        )
            T.copy(acc, dAttnSink[bh * block_H : (bh + 1) * block_H])

    return attn_sink_bwd_det_kernel


@tilelang.jit(
    out_idx=[-3],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def bwd(
    B,
    S,
    S_kv,
    H,
    D,
    topk,
    sm_scale=None,
    block_size=32,
    num_stages=0,
    threads=128,
    indices_dtype=T.int32,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
    skip_attn_sink=False,
    skip_dkv=False,
):
    assert topk % block_size == 0, f"topk ({topk}) must be divisible by block_size ({block_size})"
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32

    if sm_scale is None:
        sm_scale = D ** (-0.5)
    sm_scale_mul_reciprocal_log2 = sm_scale * _RECIPROCAL_LOG2

    q_shape = [B, S, H, D]
    kv_shape = [B, S_kv, D]
    o_shape = [B, S, H, D]
    indices_shape = [B, S, topk]
    delta_shape = [B, S, H]
    lse_shape = [B, S, H]
    attn_sink_shape = [H]

    padded_H = max(tilelang.math.next_power_of_2(H), 16)
    block_H = min(64, padded_H)
    assert padded_H % block_H == 0
    NH = padded_H // block_H
    BS = block_size
    NS = tilelang.cdiv(topk, block_size)

    split_store = 2

    @T.prim_func
    def sparse_mqa_bwd_kernel(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(kv_shape, dtype),
        dO: T.Tensor(o_shape, dtype),
        AttnSink: T.Tensor(attn_sink_shape, accum_dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
        Delta: T.Tensor(delta_shape, accum_dtype),
        dQ: T.Tensor(q_shape, dtype),
        dKV: T.Tensor(kv_shape, accum_dtype),
        dAttnSink: T.Tensor(attn_sink_shape, accum_dtype),
    ):
        with T.Kernel(S, B, NH, threads=threads) as (s_i, by, bz):
            Q_shared = T.alloc_shared([block_H, D], dtype)
            KV_shared = T.alloc_shared([BS, D], dtype)
            dO_shared = T.alloc_shared([block_H, D], dtype)
            mask = T.alloc_fragment([BS], "bool")

            P_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dP_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dQ_shared = T.alloc_shared([block_H, D], dtype)

            acc_p = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dp = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dq = T.alloc_fragment([block_H, D], accum_dtype)
            acc_dkv = T.alloc_fragment([BS, D], accum_dtype)
            acc_dkv_shared = T.alloc_shared([BS // split_store, D], accum_dtype)

            T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, :D], Q_shared)
            T.copy(dO[by, s_i, bz * block_H : (bz + 1) * block_H, :D], dO_shared)

            T.clear(acc_dq)

            for i_i in T.Pipelined(NS, num_stages=num_stages):
                for bi_i in T.Parallel(BS):
                    mask[bi_i] = Indices[by, s_i, i_i * BS + bi_i] != -1

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_p.dtype))

                for bi_i, d_i in T.Parallel(BS, D):
                    KV_shared[bi_i, d_i] = KV[by, Indices[by, s_i, i_i * BS + bi_i], d_i]

                T.gemm(Q_shared, KV_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.exp2(
                        acc_p[h_i, bi_i] * sm_scale_mul_reciprocal_log2 - Lse[by, s_i, bz * block_H + h_i]
                    )

                T.copy(acc_p, P_shared_cast)

                T.gemm(
                    dO_shared, KV_shared, acc_dp, transpose_B=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True
                )

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_dp[h_i, bi_i] = (
                        acc_p[h_i, bi_i] * (acc_dp[h_i, bi_i] - Delta[by, s_i, bz * block_H + h_i]) * sm_scale
                    )

                T.copy(acc_dp, dP_shared_cast)
                T.gemm(dP_shared_cast, KV_shared, acc_dq, policy=T.GemmWarpPolicy.FullCol)

                if not skip_dkv:
                    T.gemm(
                        dP_shared_cast,
                        Q_shared,
                        acc_dkv,
                        transpose_A=True,
                        policy=T.GemmWarpPolicy.FullCol,
                        clear_accum=True,
                    )
                    T.gemm(P_shared_cast, dO_shared, acc_dkv, transpose_A=True, policy=T.GemmWarpPolicy.FullCol)

                    for s in range(split_store):
                        for bi_i, d_i in T.Parallel(BS, D):
                            if bi_i < BS // split_store:
                                acc_dkv_shared[bi_i, d_i] = acc_dkv[bi_i + s * (BS // split_store), d_i]

                        for bi_i, d_i in T.Parallel(BS // split_store, D // 4):
                            T.atomic_addx4(
                                dKV[
                                    by,
                                    Indices[by, s_i, i_i * BS + bi_i + s * (BS // split_store)],
                                    d_i * 4,
                                ],
                                acc_dkv_shared[bi_i, d_i * 4],
                            )

            T.copy(acc_dq, dQ_shared)
            T.copy(dQ_shared, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, :D])

            if not skip_attn_sink:
                for h_i in T.Parallel(block_H):
                    T.atomic_add(
                        dAttnSink[bz * block_H + h_i],
                        -Delta[by, s_i, bz * block_H + h_i]
                        * T.exp2(AttnSink[bz * block_H + h_i] * _RECIPROCAL_LOG2 - Lse[by, s_i, bz * block_H + h_i]),
                    )

    return sparse_mqa_bwd_kernel


def _zeros_like_compat(tensor, dtype=None):
    return paddle.zeros_like(tensor, dtype=dtype)


def _deterministic_attn_sink_enabled():
    default = os.getenv("DSV4_TILELANG_DETERMINISTIC_BWD", "1")
    return os.getenv("DSV4_TILELANG_SPARSE_MLA_DETERMINISTIC_ATTN_SINK", default).lower() in {"1", "true", "yes", "on"}


def _deterministic_dkv_enabled():
    default = os.getenv("DSV4_TILELANG_DETERMINISTIC_BWD", "1")
    return os.getenv("DSV4_TILELANG_SPARSE_MLA_DETERMINISTIC_DKV", default).lower() in {"1", "true", "yes", "on"}


def _attn_sink_block_h(H):
    for block_H in (64, 32, 16, 8, 4, 2, 1):
        if H % block_H == 0:
            return block_H
    return 1


def _deterministic_dkv_reduction_mode():
    return os.getenv("DSV4_TILELANG_SPARSE_MLA_DKV_REDUCTION", "tilelang_bucketed").lower()


@tilelang.jit(out_idx=[-1])
def dkv_bwd_contrib(
    B,
    S,
    S_kv,
    H,
    D,
    topk,
    sm_scale=None,
    block_size=32,
    num_stages=0,
    threads=128,
    indices_dtype=T.int32,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    assert topk % block_size == 0
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    if sm_scale is None:
        sm_scale = D ** (-0.5)
    sm_scale_mul_reciprocal_log2 = sm_scale * _RECIPROCAL_LOG2

    q_shape = [B, S, H, D]
    kv_shape = [B, S_kv, D]
    do_shape = [B, S, H, D]
    indices_shape = [B, S, topk]
    lse_shape = [B, S, H]
    delta_shape = [B, S, H]
    padded_H = max(tilelang.math.next_power_of_2(H), 16)
    block_H = min(64, padded_H)
    assert padded_H % block_H == 0
    NH = padded_H // block_H
    BS = block_size
    NS = tilelang.cdiv(topk, block_size)
    contrib_shape = [B, NH, S, topk, D]

    @T.prim_func
    def dkv_contrib_kernel(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(kv_shape, dtype),
        dO: T.Tensor(do_shape, dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
        Delta: T.Tensor(delta_shape, accum_dtype),
        Contrib: T.Tensor(contrib_shape, accum_dtype),
    ):
        with T.Kernel(S, B, NH, threads=threads) as (s_i, by, bz):
            Q_shared = T.alloc_shared([block_H, D], dtype)
            KV_shared = T.alloc_shared([BS, D], dtype)
            dO_shared = T.alloc_shared([block_H, D], dtype)
            mask = T.alloc_fragment([BS], "bool")
            P_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dP_shared_cast = T.alloc_shared([block_H, BS], dtype)
            acc_p = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dp = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dkv = T.alloc_fragment([BS, D], accum_dtype)

            T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, :D], Q_shared)
            T.copy(dO[by, s_i, bz * block_H : (bz + 1) * block_H, :D], dO_shared)

            for i_i in T.Pipelined(NS, num_stages=num_stages):
                for bi_i in T.Parallel(BS):
                    mask[bi_i] = Indices[by, s_i, i_i * BS + bi_i] != -1
                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_p.dtype))
                for bi_i, d_i in T.Parallel(BS, D):
                    KV_shared[bi_i, d_i] = T.if_then_else(
                        mask[bi_i],
                        KV[by, Indices[by, s_i, i_i * BS + bi_i], d_i],
                        0,
                    )
                T.gemm(Q_shared, KV_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.exp2(
                        acc_p[h_i, bi_i] * sm_scale_mul_reciprocal_log2 - Lse[by, s_i, bz * block_H + h_i]
                    )
                T.copy(acc_p, P_shared_cast)
                T.gemm(
                    dO_shared, KV_shared, acc_dp, transpose_B=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True
                )
                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_dp[h_i, bi_i] = (
                        acc_p[h_i, bi_i] * (acc_dp[h_i, bi_i] - Delta[by, s_i, bz * block_H + h_i]) * sm_scale
                    )
                T.copy(acc_dp, dP_shared_cast)
                T.gemm(
                    dP_shared_cast,
                    Q_shared,
                    acc_dkv,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                )
                T.gemm(P_shared_cast, dO_shared, acc_dkv, transpose_A=True, policy=T.GemmWarpPolicy.FullCol)
                for bi_i, d_i in T.Parallel(BS, D):
                    Contrib[by, bz, s_i, i_i * BS + bi_i, d_i] = acc_dkv[bi_i, d_i]

    return dkv_contrib_kernel


# ---------------------------------------------------------------------------
# Bucketed deterministic dKV reduce
#
# The naive twostage reduce kernel scans S * topk index entries for every
# kv_i thread block which is O(S_kv * S * topk) work and dominates training
# shapes. We instead build a deterministic (stable argsort) inverted index on
# the Paddle side mapping each kv slot -> the list of (s, k) positions that
# referenced it, then a TileLang reduce kernel that walks only the entries in
# that bucket. Total work becomes O(S * topk) plus a small overhead per bucket
# instead of O(S_kv * S * topk), which translates into multiple orders of
# magnitude speed-up at training shape while preserving determinism.
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[-1])
def dkv_bwd_reduce_bucketed(
    B,
    S,
    S_kv,
    H,
    D,
    topk,
    max_bucket,
    block_D=32,
    threads=128,
    indices_dtype=T.int32,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    """Reduce kernel that consumes a pre-sorted bucket layout.

    SortPos[b, p] holds the flat (s * topk + k) position for the p-th hit of
    some bucket. BucketStarts[b, kv_i] / [b, kv_i+1] delimit which slice of
    SortPos belongs to bucket kv_i. The kernel pads each bucket to
    `max_bucket` entries with sentinel == -1 so the loop bound is static.
    """
    assert D % block_D == 0
    padded_H = max(tilelang.math.next_power_of_2(H), 16)
    block_H = min(64, padded_H)
    assert padded_H % block_H == 0
    NH = padded_H // block_H
    contrib_shape = [B, NH, S, topk, D]
    sortpos_shape = [B, S_kv, max_bucket]

    @T.prim_func
    def dkv_reduce_bucketed_kernel(
        Contrib: T.Tensor(contrib_shape, accum_dtype),
        SortPos: T.Tensor(sortpos_shape, indices_dtype),
        dKV: T.Tensor([B, S_kv, D], accum_dtype),
    ):
        with T.Kernel(S_kv, B, D // block_D, threads=threads) as (kv_i, b_i, d_blk):
            acc = T.alloc_fragment([block_D], accum_dtype)
            T.clear(acc)
            for nh_i in T.serial(NH):
                for p in T.serial(max_bucket):
                    pos = SortPos[b_i, kv_i, p]
                    if pos != -1:
                        s_i = pos // topk
                        k_i = pos % topk
                        for d_i in T.Parallel(block_D):
                            acc[d_i] += Contrib[b_i, nh_i, s_i, k_i, d_blk * block_D + d_i]
            T.copy(acc, dKV[b_i, kv_i, d_blk * block_D : (d_blk + 1) * block_D])

    return dkv_reduce_bucketed_kernel


def _bucketed_chunk_size():
    return max(1, int(os.getenv("DSV4_TILELANG_SPARSE_MLA_DKV_BUCKET_CHUNK_SIZE", "512")))


def _bucketed_cap(chunk_size, topk, S_kv):
    cap_env = os.getenv("DSV4_TILELANG_SPARSE_MLA_DKV_BUCKET_CAP", "")
    if cap_env:
        cap = int(cap_env)
    else:
        cap = chunk_size
    return max(32, ((cap + 31) // 32) * 32)


def _build_bucket_layout(topk_idxs, S_kv, bucket_cap):
    """Build padded SortPos[B, S_kv, bucket_cap] with -1 sentinel.

    The cap is a static value derived from chunk/static shape, not from the
    runtime maximum bucket size. This avoids TileLang recompilation storms.
    If an actual bucket exceeds the cap, the caller falls back to the pure
    Paddle deterministic path for that chunk rather than silently truncating.
    """
    B, S, topk = topk_idxs.shape
    flat_idx = topk_idxs.reshape([B, S * topk]).cast("int64")
    sentinel_bucket = S_kv
    flat_idx_safe = paddle.where(
        flat_idx == -1,
        paddle.full_like(flat_idx, sentinel_bucket),
        flat_idx,
    )
    sort_pos = paddle.argsort(flat_idx_safe, axis=-1, stable=True).cast("int32")
    sort_idx = paddle.take_along_axis(flat_idx_safe, sort_pos.cast("int64"), axis=-1)

    kv_range = paddle.arange(S_kv + 1, dtype=sort_idx.dtype).reshape([1, -1]).expand([B, S_kv + 1])
    bucket_bounds = paddle.searchsorted(sort_idx, kv_range, right=False)
    bucket_sizes = bucket_bounds[:, 1:S_kv + 1] - bucket_bounds[:, :S_kv]
    if int(bucket_sizes.max().item()) > bucket_cap:
        return None, True

    starts_per_p = paddle.take_along_axis(bucket_bounds[:, :S_kv + 1], sort_idx, axis=-1)
    slot_per_p = paddle.arange(S * topk, dtype="int64").reshape([1, -1]).expand([B, S * topk]) - starts_per_p

    table_size = S_kv * bucket_cap
    sort_table = paddle.full([B, S_kv, bucket_cap], -1, dtype="int32")
    valid_mask = sort_idx < S_kv
    dest_flat_real = sort_idx * bucket_cap + slot_per_p
    for b in range(B):
        vmask_b = valid_mask[b]
        dest_b = dest_flat_real[b][vmask_b].cast("int64")
        val_b = sort_pos[b][vmask_b]
        flat_b = sort_table[b].reshape([table_size])
        flat_b = paddle.scatter(flat_b, dest_b, val_b, overwrite=True)
        sort_table[b] = flat_b.reshape([S_kv, bucket_cap])
    return sort_table.contiguous(), False


def _tilelang_dkv_bucketed_chunk(q, kv, do, topk_idxs, lse, delta, sm_scale, chunk_size, bucket_cap):
    """Deterministic dKV for one fixed-size chunk."""
    B, S, H, D = q.shape
    _, S_kv, _ = kv.shape
    topk = topk_idxs.shape[-1]
    if D % 32 != 0:
        raise ValueError(
            f"DSV4_TILELANG_SPARSE_MLA_DKV_REDUCTION=tilelang_bucketed requires dim divisible by 32, got {D}"
        )
    if S != chunk_size:
        pad_s = chunk_size - S
        if pad_s < 0:
            raise ValueError(f"bucketed chunk got S={S} > chunk_size={chunk_size}")
        q = paddle.concat([q, paddle.zeros([B, pad_s, H, D], dtype=q.dtype)], axis=1).contiguous()
        do = paddle.concat([do, paddle.zeros([B, pad_s, H, D], dtype=do.dtype)], axis=1).contiguous()
        topk_idxs = paddle.concat(
            [topk_idxs, paddle.full([B, pad_s, topk], -1, dtype=topk_idxs.dtype)], axis=1
        ).contiguous()
        lse = paddle.concat([lse, paddle.zeros([B, pad_s, H], dtype=lse.dtype)], axis=1).contiguous()
        delta = paddle.concat([delta, paddle.zeros([B, pad_s, H], dtype=delta.dtype)], axis=1).contiguous()

    sort_table, overflow = _build_bucket_layout(topk_idxs, S_kv, bucket_cap)
    if overflow:
        return None
    contrib_kernel = dkv_bwd_contrib(B, chunk_size, S_kv, H, D, topk, sm_scale, block_size=32)
    contrib = contrib_kernel(q, kv, do, topk_idxs, lse, delta)
    reduce_kernel = dkv_bwd_reduce_bucketed(
        B, chunk_size, S_kv, H, D, topk, bucket_cap, block_D=min(32, D)
    )
    return reduce_kernel(contrib, sort_table).cast(kv.dtype)


_BUCKET_OVERFLOW_COUNTERS = {"hits": 0, "total": 0}


def _bucket_overflow_log_enabled():
    return os.getenv("DSV4_TILELANG_SPARSE_MLA_DKV_BUCKET_OVERFLOW_LOG", "0").lower() in {"1", "true", "yes", "on"}


def get_bucket_overflow_stats():
    """Return (overflow_chunks, total_chunks) for bucketed dKV path."""
    return _BUCKET_OVERFLOW_COUNTERS["hits"], _BUCKET_OVERFLOW_COUNTERS["total"]


def _tilelang_dkv_bucketed(q, kv, do, topk_idxs, lse, delta, sm_scale):
    B, S, H, D = q.shape
    _, S_kv, _ = kv.shape
    topk = topk_idxs.shape[-1]
    chunk_size = min(_bucketed_chunk_size(), S)
    bucket_cap = _bucketed_cap(chunk_size, topk, S_kv)
    dkv = paddle.zeros(kv.shape, dtype=kv.dtype)
    for start in range(0, S, chunk_size):
        end = min(start + chunk_size, S)
        _BUCKET_OVERFLOW_COUNTERS["total"] += 1
        dkv_chunk = _tilelang_dkv_bucketed_chunk(
            q[:, start:end].contiguous(),
            kv,
            do[:, start:end].contiguous(),
            topk_idxs[:, start:end].contiguous(),
            lse[:, start:end].contiguous(),
            delta[:, start:end].contiguous(),
            sm_scale,
            chunk_size,
            bucket_cap,
        )
        if dkv_chunk is None:
            _BUCKET_OVERFLOW_COUNTERS["hits"] += 1
            if _bucket_overflow_log_enabled():
                print(
                    f"[TileLangBwd] dkv bucketed overflow: chunk=[{start}:{end}] "
                    f"chunk_size={chunk_size} bucket_cap={bucket_cap} S_kv={S_kv} topk={topk} "
                    f"-> falling back to Paddle deterministic. "
                    f"counters={_BUCKET_OVERFLOW_COUNTERS}",
                    flush=True,
                )
            dkv_chunk = _paddle_dkv_deterministic(
                q[:, start:end].contiguous(),
                kv,
                do[:, start:end].contiguous(),
                topk_idxs[:, start:end].contiguous(),
                lse[:, start:end].contiguous(),
                delta[:, start:end].contiguous(),
                sm_scale,
            )
        dkv += dkv_chunk
    return dkv


@paddle.no_grad()
def _paddle_dkv_deterministic(q, kv, do, topk_idxs, lse, delta, sm_scale):
    """Deterministic dKV via Paddle tensor reductions (one-hot reduction).

    Recomputes the sparse MLA backward dKV contribution using deterministic
    Paddle operations: gather, float32 einsums for QK scores and dKV GEMMs,
    then a fixed-order one-hot reduction in place of T.atomic_addx4.  This
    serves as the pure-Paddle reference path and as a fallback when the
    bucketed TileLang reduce hits its static cap.
    """
    B, S, H, D = q.shape
    _, S_kv, _ = kv.shape
    if sm_scale is None:
        sm_scale = D ** (-0.5)
    mask = topk_idxs != -1
    safe_idxs = paddle.where(mask, topk_idxs, paddle.zeros_like(topk_idxs)).cast("int64")
    q_f32 = q.cast("float32")
    kv_f32 = kv.cast("float32")
    do_f32 = do.cast("float32")
    dkv = paddle.zeros([B, S_kv, D], dtype="float32")
    chunk_size = int(os.getenv("DSV4_TILELANG_SPARSE_MLA_DKV_CHUNK_SIZE", "64"))
    chunk_size = max(1, chunk_size)
    for start in range(0, S, chunk_size):
        end = min(start + chunk_size, S)
        idx_chunk = safe_idxs[:, start:end]
        mask_chunk = mask[:, start:end]
        batch_idx = paddle.arange(B, dtype="int64").reshape([B, 1, 1])
        batch_idx = paddle.expand(batch_idx, [B, end - start, safe_idxs.shape[-1]])
        kv_gathered = kv_f32[batch_idx, idx_chunk]
        q_chunk = q_f32[:, start:end]
        do_chunk = do_f32[:, start:end]
        scores = paddle.einsum("bshd,bskd->bshk", q_chunk, kv_gathered) * sm_scale * _RECIPROCAL_LOG2
        p = paddle.exp((scores - lse[:, start:end].unsqueeze(-1)) * _LOG2) * mask_chunk.unsqueeze(2).cast("float32")
        dp = paddle.einsum("bshd,bskd->bshk", do_chunk, kv_gathered)
        d_scores = p * (dp - delta[:, start:end].unsqueeze(-1)) * sm_scale
        contrib = paddle.einsum("bshk,bshd->bskd", d_scores, q_chunk)
        contrib += paddle.einsum("bshk,bshd->bskd", p, do_chunk)
        contrib = contrib * mask_chunk.unsqueeze(-1).cast("float32")
        selectors = paddle.nn.functional.one_hot(idx_chunk, num_classes=S_kv).cast("float32")
        dkv += paddle.einsum("bstk,bstd->bkd", selectors, contrib)
    return dkv.cast(kv.dtype)


def _select_dkv_fn():
    reduction_mode = _deterministic_dkv_reduction_mode()
    if reduction_mode == "tilelang_bucketed":
        return _tilelang_dkv_bucketed
    if reduction_mode == "one_hot":
        return _paddle_dkv_deterministic
    raise ValueError(
        f"Unsupported DSV4_TILELANG_SPARSE_MLA_DKV_REDUCTION={reduction_mode!r}; "
        "expected 'tilelang_bucketed' or 'one_hot'."
    )


def sparse_mqa_bwd_interface(q, kv, attn_sink, o, do, topk_idxs, lse, sm_scale=None):
    """Backward interface for DSv4 sparse MQA attention."""
    assert q.is_contiguous() and kv.is_contiguous()
    assert topk_idxs.is_contiguous() and lse.is_contiguous()
    B, S, H, D = q.shape
    _, S_kv, _ = kv.shape
    topk = topk_idxs.shape[-1]

    block_size = 32
    padded_topk = (topk + block_size - 1) // block_size * block_size
    if padded_topk != topk:
        pad = paddle.full([B, S, padded_topk - topk], -1, dtype=topk_idxs.dtype)
        topk_idxs = paddle.concat([topk_idxs, pad], axis=-1).contiguous()
        topk = padded_topk

    deterministic_attn_sink = _deterministic_attn_sink_enabled()
    deterministic_dkv = _deterministic_dkv_enabled()
    preprocess_kernel = preprocess(B, S, H, D)
    bwd_kernel = bwd(
        B,
        S,
        S_kv,
        H,
        D,
        topk,
        sm_scale,
        skip_attn_sink=deterministic_attn_sink,
        skip_dkv=deterministic_dkv,
    )
    postprocess_kernel = postprocess(B, S_kv, D)
    attn_sink_bwd_det_kernel = (
        attn_sink_bwd_deterministic(B, S, H, block_H=_attn_sink_block_h(H)) if deterministic_attn_sink else None
    )

    delta = preprocess_kernel(o, do)
    dkv = _zeros_like_compat(kv, dtype="float32")
    d_attn_sink = _zeros_like_compat(attn_sink)
    dq = bwd_kernel(q, kv, do, attn_sink, topk_idxs, lse, delta, dkv, d_attn_sink)
    if deterministic_attn_sink:
        d_attn_sink = attn_sink_bwd_det_kernel(attn_sink, lse, delta)
    if deterministic_dkv:
        dkv = _select_dkv_fn()(q, kv, do, topk_idxs, lse, delta, sm_scale)
    else:
        dkv = postprocess_kernel(dkv)

    return dq, dkv, d_attn_sink
