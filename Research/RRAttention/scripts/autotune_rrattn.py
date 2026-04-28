#!/usr/bin/env python

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

from __future__ import annotations

import argparse
import csv
import gc
import statistics
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import paddle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rrattn.rrattention import (
    TUNABLE_FIELDS,
    RRAttnConfig,
    get_rrattn_config,
    rr_attn_estimate_triton_func,
)

DEFAULT_SEQ_LENS = "32768"
DEFAULT_MODES = "nomask"
DEFAULT_HEAD_DIMS = "64,128"
FIXED_BLOCK_SIZE = 128
TUNE_BLOCK_M = (64, 128)
TUNE_BLOCK_N = (16, 32, 64)
TUNE_NUM_WARPS = (4, 8)
TUNE_NUM_STAGES = (1, 2, 3)
TUNE_SEGMENT_SIZE = (64, 128, 256)


@dataclass
class CaseResult:
    head_dim: int
    config_id: int
    seq_len: int
    mode: str
    ms: float | None
    error: str | None = None
    skipped: bool = False


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_modes(value: str) -> list[str]:
    modes = [item.strip() for item in value.split(",") if item.strip()]
    bad_modes = [mode for mode in modes if mode not in {"nomask", "flashmask"}]
    if bad_modes:
        raise ValueError(f"Unsupported modes: {bad_modes}")
    return modes


def sync_cuda() -> None:
    try:
        paddle.device.synchronize()
    except Exception:
        try:
            paddle.device.cuda.synchronize()
        except Exception:
            pass


def empty_cache() -> None:
    gc.collect()
    try:
        paddle.device.cuda.empty_cache()
    except Exception:
        pass


def require_cuda(device: str) -> None:
    if not paddle.device.is_compiled_with_cuda():
        raise RuntimeError("This autotune script requires Paddle CUDA.")
    try:
        if paddle.device.cuda.device_count() <= 0:
            raise RuntimeError("No CUDA device is visible to Paddle.")
    except Exception as exc:
        raise RuntimeError("Unable to query Paddle CUDA devices.") from exc
    paddle.set_device(device)


def make_candidates(
    *,
    head_dim: int,
    stride: int,
    gpu_name: str | None,
) -> list[RRAttnConfig]:
    base = get_rrattn_config(head_dim, gpu_name=gpu_name)
    ratio = FIXED_BLOCK_SIZE // stride
    segment_sizes = tuple(
        segment_size
        for segment_size in TUNE_SEGMENT_SIZE
        if segment_size >= ratio and segment_size % ratio == 0
    )

    def cfg(**kwargs) -> RRAttnConfig:
        return replace(base, **kwargs)

    candidates = [cfg()]

    for block_m in TUNE_BLOCK_M:
        for block_n in TUNE_BLOCK_N:
            for num_warps in TUNE_NUM_WARPS:
                for num_stages in TUNE_NUM_STAGES:
                    for segment_size in segment_sizes:
                        candidates.append(
                            cfg(
                                block_m=block_m,
                                block_n=block_n,
                                num_warps=num_warps,
                                num_stages=num_stages,
                                segment_size=segment_size,
                            )
                        )

    unique = []
    seen = set()
    for item in candidates:
        key = tuple(getattr(item, field) for field in TUNABLE_FIELDS)
        if key not in seen:
            unique.append(item)
            seen.add(key)
    return unique


def make_startend(
    *,
    mode: str,
    batch_size: int,
    seq_len: int,
    flashmask_window: int,
) -> paddle.Tensor:
    if mode == "nomask":
        return paddle.full(
            [batch_size, 1, seq_len, 1], seq_len, dtype=paddle.int32
        )

    pos = paddle.arange(seq_len, dtype=paddle.int32).reshape([1, 1, seq_len, 1])
    q_end = paddle.full([1, 1, seq_len, 1], seq_len, dtype=paddle.int32)
    window_end = pos + min(seq_len, flashmask_window)
    return paddle.minimum(window_end, q_end).tile([batch_size, 1, 1, 1])


def make_inputs(
    args, seq_len: int, mode: str, head_dim: int
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    paddle.seed(args.seed + seq_len + head_dim)
    q = paddle.randn(
        [args.batch_size, seq_len, args.num_q_heads, head_dim],
        dtype=args.dtype,
    )
    k = paddle.randn(
        [args.batch_size, seq_len, args.num_kv_heads, head_dim],
        dtype=args.dtype,
    )
    startend = make_startend(
        mode=mode,
        batch_size=args.batch_size,
        seq_len=seq_len,
        flashmask_window=args.flashmask_window,
    )
    return q, k, startend


def run_estimate(
    q: paddle.Tensor,
    k: paddle.Tensor,
    startend: paddle.Tensor,
    *,
    args,
    config: RRAttnConfig,
) -> None:
    attn_sums, boundary_mask, selected_blocks = rr_attn_estimate_triton_func(
        q,
        k,
        startend,
        stride=args.stride,
        causal=True,
        threshold=args.threshold,
        chunk_size=args.chunk_size,
        config=config,
    )
    del attn_sums, boundary_mask, selected_blocks


def benchmark_case(
    q: paddle.Tensor,
    k: paddle.Tensor,
    startend: paddle.Tensor,
    *,
    args,
    config: RRAttnConfig,
    current_best_ms: float | None = None,
) -> tuple[float | None, str | None]:
    for _ in range(args.warmup):
        run_estimate(q, k, startend, args=args, config=config)
        sync_cuda()

    times = []
    check_iters = min(args.skip_check_iters, args.iters)
    enable_skip = (
        current_best_ms is not None
        and args.skip_slowdown > 0
        and check_iters > 0
        and check_iters < args.iters
    )
    for _ in range(args.iters):
        sync_cuda()
        start = time.perf_counter()
        run_estimate(q, k, startend, args=args, config=config)
        sync_cuda()
        times.append((time.perf_counter() - start) * 1000.0)

        if enable_skip and len(times) == check_iters:
            prelim_ms = statistics.mean(times)
            best_ms = float(current_best_ms)
            lower_bound_ms = sum(times) / args.iters
            if (
                prelim_ms > best_ms * args.skip_slowdown
                and lower_bound_ms > best_ms
            ):
                return (
                    None,
                    "early skip: "
                    f"{check_iters}-iter mean {prelim_ms:.4f} ms > "
                    f"{args.skip_slowdown:.1f}x current best {best_ms:.4f} ms; "
                    f"full-mean lower bound {lower_bound_ms:.4f} ms",
                )
    return statistics.mean(times), None


def format_config(config: RRAttnConfig) -> str:
    fields = ", ".join(
        f"{field}={getattr(config, field)!r}" for field in TUNABLE_FIELDS
    )
    return f"RRAttnConfig({fields})"


def summarize(
    results: list[CaseResult],
    configs: list[RRAttnConfig],
    *,
    head_dim: int,
    total_cases: int,
) -> list[tuple[float, int]]:
    rows = []
    for config_id, _ in enumerate(configs):
        vals = [
            item.ms
            for item in results
            if item.head_dim == head_dim
            and item.config_id == config_id
            and item.ms is not None
        ]
        if len(vals) != total_cases:
            score = float("inf")
        else:
            score = statistics.mean(vals)
        rows.append((score, config_id))
    return sorted(rows, key=lambda item: item[0])


def write_csv(
    path: Path,
    results: list[CaseResult],
    configs_by_head_dim: dict[int, list[RRAttnConfig]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "head_dim",
                "config_id",
                "seq_len",
                "mode",
                "ms",
                "error",
                "skipped",
                *TUNABLE_FIELDS,
            ]
        )
        for item in results:
            config = configs_by_head_dim[item.head_dim][item.config_id]
            writer.writerow(
                [
                    item.head_dim,
                    item.config_id,
                    item.seq_len,
                    item.mode,
                    "" if item.ms is None else f"{item.ms:.4f}",
                    item.error or "",
                    int(item.skipped),
                    *(getattr(config, field) for field in TUNABLE_FIELDS),
                ]
            )


def print_summary(
    *,
    results: list[CaseResult],
    configs_by_head_dim: dict[int, list[RRAttnConfig]],
    head_dims: list[int],
    seq_lens: list[int],
    modes: list[str],
    topk: int,
    early_skip_enabled: bool,
) -> None:
    total_cases = len(seq_lens) * len(modes)

    print("\nTuned metrics:")
    print(
        "  primary score: per-head_dim mean estimate latency in ms across selected seq_lens and modes"
    )
    print("  default seq_len is 32k; pass --seq-lens for broader sweeps")
    print(
        "  chunk_size is not tuned here; set it through --chunk-size or the rrattn wrapper"
    )
    if early_skip_enabled:
        print(
            "  early skip uses a full-mean lower bound, so skipped configs cannot beat the current best"
        )
    print(f"  candidate axes: block_m={TUNE_BLOCK_M}, block_n={TUNE_BLOCK_N}")
    print(
        f"  candidate axes: num_warps={TUNE_NUM_WARPS}, num_stages={TUNE_NUM_STAGES}"
    )
    print(f"  candidate axes: segment_size={TUNE_SEGMENT_SIZE}")

    best_by_head_dim = {}
    for head_dim in head_dims:
        configs = configs_by_head_dim[head_dim]
        ranked = summarize(
            results, configs, head_dim=head_dim, total_cases=total_cases
        )

        print(f"\nTop configs for head_dim={head_dim}:")
        printed = 0
        for score, config_id in ranked:
            if printed >= topk:
                break
            if score == float("inf"):
                continue
            print(
                f"  #{config_id}: score_ms={score:.4f} {format_config(configs[config_id])}"
            )
            printed += 1

        if printed == 0:
            print("  no complete config finished all cases")
            continue

        best_score, best_config_id = ranked[0]
        best_by_head_dim[head_dim] = (
            best_score,
            best_config_id,
            configs[best_config_id],
        )

    if best_by_head_dim:
        print("\nManual apply:")
        print(
            "  Edit rrattn/rrattention.py get_rrattn_config; for tuned head_dim values, use:"
        )
        for head_dim in head_dims:
            if head_dim not in best_by_head_dim:
                continue
            best_score, best_config_id, best_config = best_by_head_dim[head_dim]
            print(f"      if head_dim == {head_dim}:")
            print(
                f"          return {format_config(best_config)}  # config #{best_config_id}, score_ms={best_score:.4f}"
            )

    skipped = [item for item in results if item.skipped]
    if skipped:
        print(
            f"\nSkipped {len(skipped)} slow candidates with early-skip lower-bound checks"
        )

    failures = [item for item in results if item.error and not item.skipped]
    if failures:
        print("\nFailures:")
        for item in failures[:10]:
            print(
                f"  head_dim={item.head_dim} #{item.config_id} "
                f"seq_len={item.seq_len} mode={item.mode}: {item.error}"
            )
        if len(failures) > 10:
            print(f"  ... {len(failures) - 10} more")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autotune RRAttention estimate kernel configs."
    )
    parser.add_argument("--seq-lens", default=DEFAULT_SEQ_LENS)
    parser.add_argument("--modes", default=DEFAULT_MODES)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument(
        "--dtype", default="float16", choices=["float16", "bfloat16"]
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-q-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument(
        "--head-dim",
        type=int,
        default=128,
        help="Single head_dim used when --head-dims is empty.",
    )
    parser.add_argument(
        "--head-dims",
        default=DEFAULT_HEAD_DIMS,
        help=f"Comma-separated head_dim values. Default: {DEFAULT_HEAD_DIMS}.",
    )
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=16384)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--flashmask-window", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--skip-check-iters", type=int, default=100)
    parser.add_argument("--skip-slowdown", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N candidate configs.",
    )
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to save per-case results.",
    )
    parser.add_argument(
        "--gpu-name",
        default=None,
        help="Override GPU name used by get_rrattn_config.",
    )
    args = parser.parse_args()

    require_cuda(args.device)
    assert args.num_q_heads % args.num_kv_heads == 0, (
        "num_q_heads must be divisible by num_kv_heads"
    )
    assert 128 % args.stride == 0, "stride must divide fixed block size 128"
    assert args.chunk_size > 0, "chunk_size must be positive"
    assert args.skip_check_iters >= 0, "skip_check_iters must be non-negative"
    assert args.skip_slowdown >= 0, "skip_slowdown must be non-negative"

    seq_lens = parse_csv_ints(args.seq_lens)
    modes = parse_modes(args.modes)
    early_skip_enabled = (
        len(seq_lens) * len(modes) == 1
        and args.skip_check_iters > 0
        and args.skip_slowdown > 0
    )
    head_dims = (
        parse_csv_ints(args.head_dims) if args.head_dims else [args.head_dim]
    )
    configs_by_head_dim = {}
    for head_dim in head_dims:
        configs = make_candidates(
            head_dim=head_dim,
            stride=args.stride,
            gpu_name=args.gpu_name,
        )
        if args.limit is not None:
            configs = configs[: args.limit]
        configs_by_head_dim[head_dim] = configs

    print(
        f"Running head_dims={head_dims} over seq_lens={seq_lens}, modes={modes}"
    )
    print(f"Tunable fields: {', '.join(TUNABLE_FIELDS)}")

    results = []
    for head_dim in head_dims:
        configs = configs_by_head_dim[head_dim]
        print(f"\nHead dim {head_dim}: {len(configs)} configs")
        for seq_len in seq_lens:
            for mode in modes:
                print(
                    f"\nCase head_dim={head_dim} seq_len={seq_len} mode={mode}"
                )
                q, k, startend = make_inputs(args, seq_len, mode, head_dim)
                best_case_ms = None
                for config_id, config in enumerate(configs):
                    try:
                        ms, skip_reason = benchmark_case(
                            q,
                            k,
                            startend,
                            args=args,
                            config=config,
                            current_best_ms=best_case_ms
                            if early_skip_enabled
                            else None,
                        )
                        results.append(
                            CaseResult(
                                head_dim=head_dim,
                                config_id=config_id,
                                seq_len=seq_len,
                                mode=mode,
                                ms=ms,
                                error=skip_reason,
                                skipped=skip_reason is not None,
                            )
                        )
                        if skip_reason is not None:
                            print(f"  #{config_id}: skipped: {skip_reason}")
                        else:
                            best_case_ms = (
                                ms
                                if best_case_ms is None
                                else min(best_case_ms, ms)
                            )
                            print(f"  #{config_id}: {ms:.4f} ms")
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        results.append(
                            CaseResult(
                                head_dim=head_dim,
                                config_id=config_id,
                                seq_len=seq_len,
                                mode=mode,
                                ms=None,
                                error=error,
                            )
                        )
                        print(f"  #{config_id}: failed: {error}")
                        empty_cache()
                del q, k, startend
                sync_cuda()
                empty_cache()

    if args.csv is not None:
        write_csv(args.csv, results, configs_by_head_dim)
        print(f"\nWrote CSV: {args.csv}")

    print_summary(
        results=results,
        configs_by_head_dim=configs_by_head_dim,
        head_dims=head_dims,
        seq_lens=seq_lens,
        modes=modes,
        topk=args.topk,
        early_skip_enabled=early_skip_enabled,
    )


if __name__ == "__main__":
    main()
