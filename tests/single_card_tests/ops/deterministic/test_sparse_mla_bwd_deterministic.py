#!/usr/bin/env python3
"""Reproducibility test for sparse MLA backward with deterministic dKV and dAttnSink.

The deterministic codepath is enabled by default; these env vars can be set to "0"
to opt out and fall back to the high-throughput atomic_add path.

Usage:
    python -m pytest test_sparse_mla_bwd_deterministic.py -v

Standalone:
    python test_sparse_mla_bwd_deterministic.py --repeat 20
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import paddle
import pytest

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src"))
if REPO_SRC not in sys.path:
    sys.path.insert(0, REPO_SRC)

from paddlefleet.tilelang_ops import attention_core

_REQUIRED_ENV = {
    "DSV4_TILELANG_SPARSE_MLA_DETERMINISTIC_ATTN_SINK": "1",
    "DSV4_TILELANG_SPARSE_MLA_DETERMINISTIC_DKV": "1",
}


def _env_enabled():
    return all(os.getenv(name, "1").lower() in {"1", "true", "yes", "on"} for name in _REQUIRED_ENV)


pytestmark = pytest.mark.skipif(
    not _env_enabled(),
    reason="DSV4_TILELANG_SPARSE_MLA_DETERMINISTIC_ATTN_SINK / _DKV must not be explicitly disabled",
)


def _tensor_digest(tensor):
    arr = tensor.numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _validate_shape(heads, topk, dim):
    if heads < 64 or heads % 64 != 0 or topk % 64 != 0 or dim & (dim - 1) != 0:
        raise ValueError("Use supported TileLang sparse MLA shape, e.g. heads=64, topk=64, dim=32")


def _make_topk(batch, seq, seq_kv, topk, mode="normal", collision_keys=2):
    if mode == "collision":
        topk_tensor = paddle.arange(topk, dtype="int32") % max(1, collision_keys)
        return topk_tensor.reshape([1, 1, topk]).expand([batch, seq, topk]).contiguous()
    return paddle.randint(0, seq_kv, [batch, seq, topk], dtype="int32").contiguous()


def _clone_leaf(tensor):
    cloned = tensor.detach().clone()
    cloned.stop_gradient = False
    return cloned


def _make_inputs(batch=1, seq=16, seq_kv=8, heads=64, dim=32, topk=64, mode="normal", collision_keys=2, seed=20260524):
    _validate_shape(heads, topk, dim)
    paddle.seed(seed)
    np.random.seed(seed)
    q = paddle.randn([batch, seq, heads, dim], dtype="float32").cast("bfloat16").contiguous()
    kv = paddle.randn([batch, seq_kv, dim], dtype="float32").cast("bfloat16").contiguous()
    attn_sink = paddle.randn([heads], dtype="float32").contiguous()
    topk_idxs = _make_topk(batch, seq, seq_kv, topk, mode=mode, collision_keys=collision_keys)
    grad_out = paddle.randn([batch, seq, heads, dim], dtype="float32").cast("bfloat16").contiguous()
    sm_scale = dim ** -0.5
    return q, kv, attn_sink, topk_idxs, grad_out, sm_scale


def _run_once(q, kv, attn_sink, topk_idxs, grad_out, sm_scale):
    q_tl = _clone_leaf(q)
    kv_tl = _clone_leaf(kv)
    sink_tl = _clone_leaf(attn_sink)
    out = attention_core.tilelang_compressed_sparse_attn_paddle_compat_autograd(
        q_tl,
        kv_tl,
        sink_tl,
        topk_idxs,
        sm_scale,
    ).reshape(q_tl.shape)
    paddle.autograd.backward([out], [grad_out])
    if paddle.is_compiled_with_cuda():
        paddle.device.synchronize()
    return {
        "dq": q_tl.grad.detach(),
        "dkv": kv_tl.grad.detach(),
        "d_attn_sink": sink_tl.grad.detach(),
        "out": out.detach(),
    }


def run_repro(repeat=20, batch=1, seq=16, seq_kv=8, heads=64, dim=32, topk=64, mode="normal", collision_keys=2):
    q, kv, attn_sink, topk_idxs, grad_out, sm_scale = _make_inputs(
        batch=batch,
        seq=seq,
        seq_kv=seq_kv,
        heads=heads,
        dim=dim,
        topk=topk,
        mode=mode,
        collision_keys=collision_keys,
    )
    baseline = None
    for run_idx in range(repeat):
        current = _run_once(q, kv, attn_sink, topk_idxs, grad_out, sm_scale)
        digests = {name: _tensor_digest(tensor) for name, tensor in current.items()}
        if baseline is None:
            baseline = digests
            continue
        for name, digest in digests.items():
            assert digest == baseline[name], f"Run {run_idx}: {name} differs from baseline"


def test_repro_normal():
    run_repro(mode="normal")


def test_repro_collision():
    run_repro(mode="collision", collision_keys=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=16)
    parser.add_argument("--seq-kv", type=int, default=8)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--mode", choices=["normal", "collision", "both"], default="both")
    parser.add_argument("--collision-keys", type=int, default=2)
    args = parser.parse_args()

    if not _env_enabled():
        disabled = ", ".join(f"{name}={os.getenv(name)}" for name in _REQUIRED_ENV if os.getenv(name, "1").lower() not in {"1", "true", "yes", "on"})
        raise SystemExit(f"Deterministic codepath explicitly disabled via env: {disabled}")
    if paddle.is_compiled_with_cuda():
        paddle.device.set_device("gpu:0")

    modes = ["normal", "collision"] if args.mode == "both" else [args.mode]
    for mode in modes:
        print(f"=== Sparse MLA {mode} repeat={args.repeat} ===")
        run_repro(
            repeat=args.repeat,
            batch=args.batch,
            seq=args.seq,
            seq_kv=args.seq_kv,
            heads=args.heads,
            dim=args.dim,
            topk=args.topk,
            mode=mode,
            collision_keys=args.collision_keys,
        )
        print("PASSED")
