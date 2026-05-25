#!/usr/bin/env python3
"""Reproducibility test for CSA indexer backward with deterministic grad_k_comp.

Usage:
    DSV4_TILELANG_CSA_INDEXER_DETERMINISTIC_BWD=1 python -m pytest \
        test_csa_indexer_bwd_deterministic.py -v

Standalone:
    DSV4_TILELANG_CSA_INDEXER_DETERMINISTIC_BWD=1 python \
        test_csa_indexer_bwd_deterministic.py --repeat 20
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

from paddlefleet.ops.tilelang_dsv4 import csa_indexer_core as core

_ENV_VAR = "DSV4_TILELANG_CSA_INDEXER_DETERMINISTIC_BWD"
_THRESHOLD = 0.0


def _env_enabled():
    return os.getenv(_ENV_VAR, "0").lower() in {"1", "true", "yes", "on"}


pytestmark = pytest.mark.skipif(not _env_enabled(), reason=f"requires {_ENV_VAR}=1")


def _tensor_digest(tensor):
    arr = tensor.numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _make_inputs(batch=2, seq=256, seq_comp=128, heads=16, dim=128, seed=20260523):
    paddle.seed(seed)
    np.random.seed(seed)
    index_q = paddle.randn([batch, seq, heads, dim], dtype="float32").cast("bfloat16").contiguous()
    index_k_comp = paddle.randn([batch, seq_comp, dim], dtype="float32").cast("bfloat16").contiguous()
    weights = paddle.randn([batch, seq, heads], dtype="float32").contiguous()
    return index_q, index_k_comp, weights


def _make_collision_indices(batch, seq, topk, collision_keys=4):
    base = paddle.arange(topk, dtype="int32") % max(1, collision_keys)
    return base.reshape([1, 1, topk]).expand([batch, seq, topk]).contiguous()


def _run_bwd(index_q, weights, index_k_comp, topk_indices, grad_scores, block_I=32, num_stages=0, num_threads=128):
    return core.tilelang_csa_compressed_indexer_bwd_paddle(
        index_q,
        weights,
        index_k_comp,
        topk_indices,
        grad_scores,
        block_I=block_I,
        num_stages=num_stages,
        num_threads=num_threads,
    )


def run_repro(
    repeat=20,
    batch=2,
    seq=256,
    seq_comp=128,
    heads=16,
    dim=128,
    topk=64,
    mode="normal",
    collision_keys=4,
):
    index_q, index_k_comp, weights = _make_inputs(
        batch=batch,
        seq=seq,
        seq_comp=seq_comp,
        heads=heads,
        dim=dim,
    )
    if mode == "collision":
        topk_indices = _make_collision_indices(batch, seq, topk=topk, collision_keys=collision_keys)
    else:
        topk_indices, _ = core.tilelang_csa_compressed_indexer_topk_paddle(
            index_q,
            index_k_comp,
            weights,
            ratio=4,
            topk_effective=topk,
            block_K=32,
            num_stages=0,
            num_threads=128,
        )
    paddle.seed(30260523)
    grad_scores = paddle.randn(topk_indices.shape, dtype="float32")

    _, _, gk0 = _run_bwd(index_q, weights, index_k_comp, topk_indices, grad_scores)
    if paddle.is_compiled_with_cuda():
        paddle.device.synchronize()
    digest0 = _tensor_digest(gk0)

    max_seen_diff = 0.0
    for run_idx in range(1, repeat):
        _, _, gk = _run_bwd(index_q, weights, index_k_comp, topk_indices, grad_scores)
        if paddle.is_compiled_with_cuda():
            paddle.device.synchronize()
        if _tensor_digest(gk) != digest0:
            max_diff = float((gk0.cast("float64") - gk.cast("float64")).abs().max())
            max_seen_diff = max(max_seen_diff, max_diff)
            assert max_diff == _THRESHOLD, f"Run {run_idx}: max_diff={max_diff} exceeds bit-exact threshold"
    return max_seen_diff


def test_repro_normal():
    run_repro(mode="normal")


def test_repro_collision():
    run_repro(mode="collision")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--seq-comp", type=int, default=128)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--mode", choices=["normal", "collision", "both"], default="both")
    parser.add_argument("--collision-keys", type=int, default=4)
    args = parser.parse_args()

    if not _env_enabled():
        raise SystemExit(f"Missing deterministic env var: {_ENV_VAR}=1")
    if paddle.is_compiled_with_cuda():
        paddle.device.set_device("gpu:0")

    modes = ["normal", "collision"] if args.mode == "both" else [args.mode]
    for mode in modes:
        print(f"=== CSA indexer {mode} repeat={args.repeat} ===")
        max_diff = run_repro(
            repeat=args.repeat,
            batch=args.batch,
            seq=args.seq,
            seq_comp=args.seq_comp,
            heads=args.heads,
            dim=args.dim,
            topk=args.topk,
            mode=mode,
            collision_keys=args.collision_keys,
        )
        print(f"PASSED max_seen_diff={max_diff:.3e}")
