# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Performance / memory benchmark for the V4 CSA Indexer TileLang kernel
suite (Task 11 of the megatron-indexer-migration-plan).

Coverage matrix
---------------

* 11.1 Indexer forward time + memory: ``tilelang_csa_compressed_indexer_topk_paddle``
  vs the Paddle naive ``fused_qk_topk_naive`` (full ``[B,S,S_comp]`` index
  scores, then ``paddle.topk``).
* 11.2 Phase 2 full-candidate ``topk_effective = n_compressed`` for both
  forward AND backward (TileLang bwd wrapper vs ``FusedDSAIndexerLoss``
  with ``sparse_loss=False``).
* 11.3 Phase 3 selected-topk ``topk_effective = min(index_topk, n_compressed)``
  forward AND backward (TileLang bwd wrapper vs
  ``FusedDSAIndexerLoss(sparse_loss=True)``).
* 11.4 Sequence lengths ``S in {4K, 8K, 32K, 64K, 128K}`` via ``--seqs``,
  ``ratio=4``, ``index_topk=512``, plus configurable ``--batches``.
* 11.5 Per-stage breakdown: ``indexer_before_topk`` (synthetic input gen
  using the ``CSAIndexer.forward_before_topk`` shape contract — this is
  not the real indexer module but mirrors its output shapes so users can
  combine it with the indexer-module timing recorded elsewhere),
  ``indexer_qk_topk`` (the fwd kernel call), ``indexer_backward``
  (the bwd kernel / Paddle PyLayer backward), and ``sparse_attn_kernel``
  (``tilelang_compressed_sparse_attn_paddle_compat_autograd`` with the
  produced indices).

The 11.6 default-policy decision (whether Phase 2 enables TileLang full
candidate backward by default) is documented in
``.comate/specs/megatron-indexer-migration-plan/summary.md`` based on
the JSON output of this script — re-run with the production shapes
to refresh the numbers.

Import-order requirement
------------------------

``paddle.enable_compat(scope={'tilelang'})`` MUST run BEFORE any module
that transitively does ``import torch`` inside tilelang. The order
below mirrors the kernel-level test files
(``test_tilelang_dsv4_csa_indexer_*``).

CLI usage
---------

::

    python benchmark_tilelang_dsv4_csa_indexer.py \\
        --seqs 4096 8192 32768 \\
        --batches 1 \\
        --topk 512 --ratio 4 --warmup 3 --iters 10 \\
        --json out.json

Set ``--seqs auto`` (default) to run the default 4K..128K sweep.

Skips with a clear message when CUDA is unavailable.
"""

import argparse
import gc
import json
import os
import sys
import time

# Ensure the local PaddleFleet source is loaded instead of any stale install.
_LOCAL_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")
)
if _LOCAL_SRC not in sys.path:
    sys.path.insert(0, _LOCAL_SRC)
for _m in [
    m for m in list(sys.modules)
    if m == "paddlefleet" or m.startswith("paddlefleet.")
]:
    _mod = sys.modules.get(_m)
    _f = getattr(_mod, "__file__", "") or ""
    if _LOCAL_SRC not in _f:
        sys.modules.pop(_m, None)

# CRITICAL ORDER: paddle -> enable_compat({'tilelang'}) -> tilelang.
import paddle

paddle.enable_compat(scope={"tilelang"}, silent=True)


DEFAULT_SEQS = [4096, 8192, 32768, 65536, 131072]


# ---------------------------------------------------------------------------
# Timing / memory helpers
# ---------------------------------------------------------------------------


def _cuda_sync():
    paddle.device.cuda.synchronize()


def _reset_memory_stats():
    if hasattr(paddle.device.cuda, "reset_max_memory_allocated"):
        paddle.device.cuda.reset_max_memory_allocated()
    if hasattr(paddle.device.cuda, "reset_max_memory_reserved"):
        paddle.device.cuda.reset_max_memory_reserved()
    gc.collect()
    paddle.device.cuda.empty_cache()


def _peak_memory_mb():
    if hasattr(paddle.device.cuda, "max_memory_allocated"):
        return paddle.device.cuda.max_memory_allocated() / (1024 ** 2)
    return float("nan")


def _bench(fn, warmup: int, iters: int):
    """Run ``fn`` ``iters`` times and return ``(median_ms, peak_mem_mb)``.

    ``fn`` must take no arguments and may return any value. We do NOT keep
    its return value across iterations to avoid memory blow-up.

    On any exception (typically OOM), returns ``(float('nan'), float('nan'))``
    and prints a one-line diagnostic. The benchmark continues with the next
    stage so a single Paddle baseline OOM does not abort the full sweep."""
    try:
        # Warmup
        for _ in range(warmup):
            out = fn()
            del out
        _cuda_sync()
        _reset_memory_stats()

        times = []
        for _ in range(iters):
            _cuda_sync()
            t0 = time.perf_counter()
            out = fn()
            _cuda_sync()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
            del out
        times.sort()
        median = times[len(times) // 2]
        peak = _peak_memory_mb()
        return median, peak
    except (MemoryError, RuntimeError) as exc:
        msg = str(exc).splitlines()[0][:160]
        print(f"    [stage skip] {type(exc).__name__}: {msg}")
        gc.collect()
        paddle.device.cuda.empty_cache()
        return float("nan"), float("nan")


# ---------------------------------------------------------------------------
# Input generation (matches CSAIndexer.forward_before_topk shape contract)
# ---------------------------------------------------------------------------


def _make_inputs(b, sq, sk, h_i=64, d_i=128, dtype="bfloat16", seed=11):
    paddle.seed(seed)
    q = paddle.randn([b, sq, h_i, d_i]).astype(dtype)
    k = paddle.randn([b, sk, d_i]).astype(dtype)
    weights = paddle.randn([b, sq, h_i]).astype("float32")
    return q, k, weights


# ---------------------------------------------------------------------------
# Reference Paddle naive: ``fused_qk_topk_naive`` (Task 11.1 baseline)
# ---------------------------------------------------------------------------


def _paddle_naive_indexer_topk(q, k, weights, ratio, topk_effective):
    """Replicates the Paddle reference path used in
    ``test_tilelang_dsv4_csa_indexer_wrapper_fwd.py`` — full
    ``[B,S,S_comp]`` index scores + masked top-k. Used as the
    ``fused_qk_topk_naive`` baseline."""
    from paddlefleet.transformer.dsa_attention import fused_qk_topk_naive

    b, sq, _, _ = q.shape
    sk = k.shape[1]
    comp_ids = paddle.arange(sk, dtype="int64").reshape([1, 1, sk])
    valid_end = (
        paddle.arange(1, sq + 1, dtype="int64").reshape([1, sq, 1]) // ratio
    )
    valid_mask = (comp_ids < valid_end).expand([b, sq, sk])
    neg_inf = paddle.full([b, sq, sk], float("-inf"), dtype="float32")
    causal_mask = paddle.where(
        valid_mask, paddle.zeros_like(neg_inf), neg_inf
    )
    actual_topk = min(int(topk_effective), int(sk))
    return fused_qk_topk_naive(
        q, k, weights, index_topk=actual_topk, mask=causal_mask
    )


# ---------------------------------------------------------------------------
# Reference Paddle full PyLayer: ``FusedDSAIndexerLoss`` for backward parity
# ---------------------------------------------------------------------------


def _paddle_full_pylayer_loss_step(
    q, k, weights, query, key_for_loss, ratio, sparse_loss, index_topk
):
    """Run one fwd+bwd step of ``FusedDSAIndexerLoss`` matching the CSA
    integration site (``csa_attention.py`` ratio=4 path).

    Returns a callable that runs one step (used by ``_bench``)."""
    from paddlefleet.transformer.dsa_attention import FusedDSAIndexerLoss

    b, sq, h_i, d_i = q.shape
    sk = k.shape[1]
    softmax_scale = d_i ** -0.5

    # batch-first -> seq-first, mirroring csa_attention.py
    q_sf = q.transpose([1, 0, 2, 3]).detach()
    k_sf = k.transpose([1, 0, 2]).detach()
    w_sf = (weights * softmax_scale).transpose([1, 0, 2]).detach()
    query_sf = query.transpose([1, 0, 2, 3]).detach()
    key_for_loss_d = key_for_loss.detach()

    comp_ids = paddle.arange(sk, dtype="int64").reshape([1, 1, sk])
    valid_end = (
        paddle.arange(1, sq + 1, dtype="int64").reshape([1, sq, 1]) // ratio
    )
    valid_mask = (comp_ids < valid_end).expand([b, sq, sk])
    neg_inf = paddle.full([b, sq, sk], float("-inf"), dtype="float32")
    causal_mask = paddle.where(
        valid_mask, paddle.zeros_like(neg_inf), neg_inf
    )
    mask_for_loss = causal_mask.unsqueeze(1)
    topk_eff = min(int(index_topk), int(sk))

    def _step():
        q_in = q_sf.detach()
        q_in.stop_gradient = False
        w_in = w_sf.detach()
        w_in.stop_gradient = False
        k_in = k_sf.detach()
        k_in.stop_gradient = False
        loss = FusedDSAIndexerLoss.apply(
            q_in,
            w_in,
            k_in,
            query_sf,
            key_for_loss_d,
            softmax_scale,
            topk_eff,
            1.0,            # loss_coeff
            mask_for_loss,
            bool(sparse_loss),
            None,           # tp_group
        )
        loss.backward()
        return loss

    return _step


# ---------------------------------------------------------------------------
# TileLang fwd-only / bwd-only callables (Task 11.5 stage breakdown)
# ---------------------------------------------------------------------------


def _tilelang_fwd_callable(q, k, weights, ratio, topk_effective):
    from paddlefleet.ops.tilelang_dsv4 import (
        tilelang_csa_compressed_indexer_topk_paddle,
    )

    def _run():
        return tilelang_csa_compressed_indexer_topk_paddle(
            q, k, weights, ratio=int(ratio), topk_effective=int(topk_effective)
        )

    return _run


def _tilelang_bwd_callable(q, k, weights, topk_indices, grad_scores):
    from paddlefleet.ops.tilelang_dsv4 import (
        tilelang_csa_compressed_indexer_bwd_paddle,
    )

    def _run():
        return tilelang_csa_compressed_indexer_bwd_paddle(
            q, weights, k, topk_indices, grad_scores
        )

    return _run


def _tilelang_sparse_attn_callable(query, kv_full, attn_sink, topk_idxs, scale):
    from paddlefleet.ops.tilelang_dsv4 import (
        tilelang_compressed_sparse_attn_paddle_compat_autograd,
    )

    def _run():
        return tilelang_compressed_sparse_attn_paddle_compat_autograd(
            query, kv_full, attn_sink, topk_idxs, scale, topk_pad_to=64
        )

    return _run


# ---------------------------------------------------------------------------
# Per-shape benchmark
# ---------------------------------------------------------------------------


def _bench_shape(
    *,
    b: int,
    sq: int,
    ratio: int,
    index_topk: int,
    h_i: int,
    d_i: int,
    np_heads: int,
    v_head_dim: int,
    warmup: int,
    iters: int,
):
    """Returns a dict of per-stage (time_ms, peak_mb) measurements for one
    ``(b, sq, ratio, index_topk)`` shape combination."""
    sk = sq // ratio  # n_compressed
    topk_eff_phase2 = sk
    topk_eff_phase3 = min(index_topk, sk)

    print(
        f"\n[bench] B={b} S={sq} S_comp={sk} ratio={ratio} "
        f"topk_eff(P2)={topk_eff_phase2} topk_eff(P3)={topk_eff_phase3}",
        flush=True,
    )

    out = {
        "batch": b, "seq_len": sq, "n_compressed": sk, "ratio": ratio,
        "index_topk": index_topk,
        "topk_eff_phase2": topk_eff_phase2,
        "topk_eff_phase3": topk_eff_phase3,
    }

    # ---- Inputs ----------------------------------------------------------
    q, k, weights = _make_inputs(b, sq, sk, h_i=h_i, d_i=d_i)
    # Sparse attention inputs (single MQA head on the KV side, np_heads on Q).
    query_attn = paddle.randn([b, sq, np_heads, v_head_dim]).astype("bfloat16")
    kv_full = paddle.randn([b, sq + sk, v_head_dim]).astype("bfloat16")
    attn_sink = paddle.zeros([np_heads], dtype="float32")
    softmax_scale = v_head_dim ** -0.5

    # ---- 11.1 / 11.5 indexer_qk_topk: TileLang fwd, both phases ---------
    out["tl_fwd_phase2"] = _bench(
        _tilelang_fwd_callable(q, k, weights, ratio, topk_eff_phase2),
        warmup, iters,
    )
    out["tl_fwd_phase3"] = _bench(
        _tilelang_fwd_callable(q, k, weights, ratio, topk_eff_phase3),
        warmup, iters,
    )

    # ---- 11.1 paddle naive ``fused_qk_topk_naive`` (full [B,S,S_comp]) -
    out["paddle_naive_phase2"] = _bench(
        lambda: _paddle_naive_indexer_topk(q, k, weights, ratio, topk_eff_phase2),
        warmup, iters,
    )
    out["paddle_naive_phase3"] = _bench(
        lambda: _paddle_naive_indexer_topk(q, k, weights, ratio, topk_eff_phase3),
        warmup, iters,
    )

    # ---- 11.5 indexer_backward: TileLang bwd, both phases ---------------
    # Use the TileLang fwd output as topk_indices; grad_scores random.
    from paddlefleet.ops.tilelang_dsv4 import (
        tilelang_csa_compressed_indexer_topk_paddle,
    )

    tl_idx_p2, _ = tilelang_csa_compressed_indexer_topk_paddle(
        q, k, weights, ratio=ratio, topk_effective=topk_eff_phase2
    )
    tl_idx_p3, _ = tilelang_csa_compressed_indexer_topk_paddle(
        q, k, weights, ratio=ratio, topk_effective=topk_eff_phase3
    )
    grad_p2 = paddle.randn(list(tl_idx_p2.shape)).astype("float32")
    grad_p3 = paddle.randn(list(tl_idx_p3.shape)).astype("float32")
    out["tl_bwd_phase2"] = _bench(
        _tilelang_bwd_callable(q, k, weights, tl_idx_p2, grad_p2),
        warmup, iters,
    )
    out["tl_bwd_phase3"] = _bench(
        _tilelang_bwd_callable(q, k, weights, tl_idx_p3, grad_p3),
        warmup, iters,
    )

    # ---- 11.2 / 11.3 Paddle FusedDSAIndexerLoss fwd+bwd ----------------
    # Build the MLA "real" attention key for the loss target. CSA uses
    # compressed kv expanded across heads.
    # Per csa_attention: key_for_loss = compressed_kv.transpose([1,0,2]).unsqueeze(2).expand([-1,-1,np_heads,-1])
    compressed_kv = paddle.randn([b, sk, v_head_dim]).astype("bfloat16")
    key_for_loss = (
        compressed_kv.transpose([1, 0, 2])
        .unsqueeze(2)
        .expand([-1, -1, np_heads, -1])
    )
    out["paddle_pylayer_phase2"] = _bench(
        _paddle_full_pylayer_loss_step(
            q, k, weights, query_attn, key_for_loss,
            ratio=ratio, sparse_loss=False, index_topk=topk_eff_phase2,
        ),
        warmup, iters,
    )
    out["paddle_pylayer_phase3"] = _bench(
        _paddle_full_pylayer_loss_step(
            q, k, weights, query_attn, key_for_loss,
            ratio=ratio, sparse_loss=True, index_topk=index_topk,
        ),
        warmup, iters,
    )

    # ---- 11.5 sparse_attn_kernel: TileLang sparse attention -------------
    # Use the TileLang Phase 3 indices, mapped into kv_full space (same
    # convention as csa_attention.py: +offset / -1 invalid).
    offset = sq
    valid_end = (
        paddle.arange(1, sq + 1, dtype="int64").reshape([1, sq, 1]) // ratio
    )
    valid = (tl_idx_p3.cast("int64") < valid_end).expand_as(tl_idx_p3)
    compress_topk_idxs = paddle.where(
        valid,
        (tl_idx_p3.cast("int64") + offset).cast("int32"),
        paddle.full_like(tl_idx_p3, -1, dtype="int32"),
    )
    # Window indices: simple causal window of size 64 — enough to exercise
    # the kernel layout; production CSA picks via get_window_topk_idxs.
    window = 64
    window_idxs = paddle.arange(sq, dtype="int32").reshape([1, sq, 1])
    win_offset = paddle.arange(window, dtype="int32").reshape([1, 1, window])
    window_idxs = window_idxs - (window - 1) + win_offset
    window_idxs = paddle.where(
        window_idxs >= 0, window_idxs, paddle.full_like(window_idxs, -1)
    ).expand([b, sq, window])
    topk_idxs = paddle.concat(
        [window_idxs.cast("int32"), compress_topk_idxs.cast("int32")], axis=-1,
    )
    out["sparse_attn_kernel"] = _bench(
        _tilelang_sparse_attn_callable(
            query_attn, kv_full, attn_sink, topk_idxs, softmax_scale,
        ),
        warmup, iters,
    )

    # ---- Pretty-print ---------------------------------------------------
    def _fmt(stage):
        t, m = out[stage]
        if t != t:  # NaN check
            return "    n/a (OOM/skip)        "
        return f"{t:8.2f} ms / {m:8.1f} MB"
    print(f"  TileLang fwd  P2: {_fmt('tl_fwd_phase2')}")
    print(f"  TileLang fwd  P3: {_fmt('tl_fwd_phase3')}")
    print(f"  Paddle naive  P2: {_fmt('paddle_naive_phase2')}")
    print(f"  Paddle naive  P3: {_fmt('paddle_naive_phase3')}")
    print(f"  TileLang bwd  P2: {_fmt('tl_bwd_phase2')}")
    print(f"  TileLang bwd  P3: {_fmt('tl_bwd_phase3')}")
    print(f"  Paddle PyLyr  P2: {_fmt('paddle_pylayer_phase2')}")
    print(f"  Paddle PyLyr  P3: {_fmt('paddle_pylayer_phase3')}")
    print(f"  Sparse attn   P3: {_fmt('sparse_attn_kernel')}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_seqs(arg):
    if arg in ("auto", "default", None):
        return list(DEFAULT_SEQS)
    return [int(s) for s in arg]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="V4 CSA Indexer perf/memory benchmark (Task 11)"
    )
    parser.add_argument(
        "--seqs", nargs="+", default=None,
        help=f"Sequence lengths (default {DEFAULT_SEQS})",
    )
    parser.add_argument("--batches", nargs="+", type=int, default=[1])
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--topk", type=int, default=512,
                        help="dsa_indexer_topk (Phase 3)")
    parser.add_argument("--h_i", type=int, default=64,
                        help="Indexer heads (CSAIndexer.index_n_heads)")
    parser.add_argument("--d_i", type=int, default=128,
                        help="Indexer head dim")
    parser.add_argument("--np_heads", type=int, default=128,
                        help="Q heads for sparse attention")
    parser.add_argument("--v_head_dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--json", default=None,
                        help="If set, dump per-shape results as JSON.")
    parser.add_argument("--max_seq_skip_bwd", type=int, default=131072,
                        help=("Above this S, skip the Paddle full PyLayer "
                              "phase2 baseline (full [B,S,S_comp] scores) "
                              "to avoid OOM. The TileLang bwd is still "
                              "measured."))
    args = parser.parse_args(argv)

    if not paddle.device.is_compiled_with_cuda():
        print("[skip] CUDA build of Paddle is required.", file=sys.stderr)
        return 0
    if paddle.device.cuda.device_count() == 0:
        print("[skip] No CUDA device available.", file=sys.stderr)
        return 0

    seqs = _parse_seqs(args.seqs)
    print(
        f"Bench config: ratio={args.ratio} topk={args.topk} "
        f"h_i={args.h_i} d_i={args.d_i} np_heads={args.np_heads} "
        f"v_head_dim={args.v_head_dim} warmup={args.warmup} iters={args.iters}"
    )
    print(f"Sequence lengths: {seqs}")
    print(f"Batches: {args.batches}")

    results = []
    for b in args.batches:
        for sq in seqs:
            try:
                row = _bench_shape(
                    b=b, sq=sq, ratio=args.ratio, index_topk=args.topk,
                    h_i=args.h_i, d_i=args.d_i,
                    np_heads=args.np_heads, v_head_dim=args.v_head_dim,
                    warmup=args.warmup, iters=args.iters,
                )
                results.append(row)
            except Exception as exc:  # noqa: BLE001
                # OOM or kernel constraint at large S — record and continue.
                msg = f"{type(exc).__name__}: {exc}"
                print(f"  [skip B={b} S={sq}] {msg}", flush=True)
                results.append({
                    "batch": b, "seq_len": sq, "ratio": args.ratio,
                    "error": msg,
                })
            finally:
                gc.collect()
                paddle.device.cuda.empty_cache()

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
