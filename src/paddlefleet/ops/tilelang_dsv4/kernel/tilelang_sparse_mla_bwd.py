# ruff: noqa
# Adapted from miles_plugins/models/deepseek_v4/ops/kernel/tilelang_sparse_mla_bwd.py.
# This module is imported only by explicit TileLang DSv4 callers and is not wired
# into PaddleFleet attention dispatch by default.
import os
import time

from paddlefleet.ops.tilelang_dsv4.compat import enable_tilelang_paddle_compat_before_import

enable_tilelang_paddle_compat_before_import()

import paddle
import tilelang
import torch
from tilelang import language as T


_PROFILE_COUNTERS = {}


def _profile_enabled():
    return os.getenv("DSV4_TILELANG_PROFILE", "0").lower() in {"1", "true", "yes", "on"}


def _profile_limit():
    try:
        return int(os.getenv("DSV4_TILELANG_PROFILE_STEPS", "20"))
    except ValueError:
        return 20


def _profile_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _profile_time_ms(fn):
    _profile_sync()
    start = time.perf_counter()
    result = fn()
    _profile_sync()
    return result, (time.perf_counter() - start) * 1000.0


def _profile_should_log(key):
    if not _profile_enabled():
        return False
    count = _PROFILE_COUNTERS.get(key, 0)
    if count >= _profile_limit():
        return False
    _PROFILE_COUNTERS[key] = count + 1
    return True


def _profile_log(phase, elapsed_ms=None, **kwargs):
    fields = [f"phase={phase}"]
    if elapsed_ms is not None:
        fields.append(f"elapsed_ms={elapsed_ms:.3f}")
    fields.extend(f"{key}={value}" for key, value in kwargs.items())
    print("[TileLangBwdProfile] " + " ".join(fields), flush=True)


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
):
    assert topk % block_size == 0, f"topk ({topk}) must be divisible by block_size ({block_size})"
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32

    if sm_scale is None:
        sm_scale = D ** (-0.5)
    sm_scale_mul_reciprocal_log2 = sm_scale * 1.44269504

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

            for h_i in T.Parallel(block_H):
                T.atomic_add(
                    dAttnSink[bz * block_H + h_i],
                    -Delta[by, s_i, bz * block_H + h_i]
                    * T.exp2(AttnSink[bz * block_H + h_i] * 1.44269504 - Lse[by, s_i, bz * block_H + h_i]),
                )

    return sparse_mqa_bwd_kernel


def _zeros_like_compat(tensor, dtype=None):
    if isinstance(tensor, paddle.Tensor):
        return paddle.zeros_like(tensor, dtype=dtype)
    return torch.zeros_like(tensor, dtype=dtype)


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
        pad = torch.full((B, S, padded_topk - topk), -1, device=topk_idxs.device, dtype=topk_idxs.dtype)
        topk_idxs = torch.cat([topk_idxs, pad], dim=-1).contiguous()
        topk = padded_topk

    preprocess_kernel = preprocess(B, S, H, D)
    bwd_kernel = bwd(B, S, S_kv, H, D, topk, sm_scale)
    postprocess_kernel = postprocess(B, S_kv, D)

    profile = _profile_should_log("sparse_mqa_bwd_interface")
    if profile:
        delta, elapsed_ms = _profile_time_ms(lambda: preprocess_kernel(o, do))
        _profile_log("bwd_preprocess", elapsed_ms, q_shape=tuple(q.shape), kv_shape=tuple(kv.shape), topk=topk)
        dkv, elapsed_ms = _profile_time_ms(lambda: _zeros_like_compat(kv, dtype="float32" if isinstance(kv, paddle.Tensor) else torch.float32))
        _profile_log("bwd_zero_dkv", elapsed_ms)
        d_attn_sink = _zeros_like_compat(attn_sink)
        dq, elapsed_ms = _profile_time_ms(
            lambda: bwd_kernel(q, kv, do, attn_sink, topk_idxs, lse, delta, dkv, d_attn_sink)
        )
        _profile_log("bwd_main_kernel", elapsed_ms)
        dkv, elapsed_ms = _profile_time_ms(lambda: postprocess_kernel(dkv))
        _profile_log("bwd_postprocess", elapsed_ms)
    else:
        delta = preprocess_kernel(o, do)
        dkv = _zeros_like_compat(kv, dtype="float32" if isinstance(kv, paddle.Tensor) else torch.float32)
        d_attn_sink = _zeros_like_compat(attn_sink)
        dq = bwd_kernel(q, kv, do, attn_sink, topk_idxs, lse, delta, dkv, d_attn_sink)
        dkv = postprocess_kernel(dkv)

    return dq, dkv, d_attn_sink
