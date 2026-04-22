#!/usr/bin/env python
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

from rrattn.config_rrattn import RRAttnConfig, TUNABLE_FIELDS, get_rrattn_config
from rrattn.kernels_rrattn import rr_attn_estimate_triton_func


DEFAULT_SEQ_LENS = "32768"
DEFAULT_MODES = "nomask"
DEFAULT_HEAD_DIMS = "64,128"


@dataclass
class CaseResult:
    head_dim: int
    config_id: int
    seq_len: int
    mode: str
    ms: float | None
    error: str | None = None


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
    gpu_name: str | None,
) -> list[RRAttnConfig]:
    base = get_rrattn_config(head_dim, gpu_name=gpu_name)

    def cfg(**kwargs) -> RRAttnConfig:
        return replace(base, **kwargs)

    candidates = [cfg()]

    for block_m in (64, 128):
        for num_warps in (4, 8):
            for num_stages in (1, 2, 3):
                candidates.append(
                    cfg(
                        block_m=block_m,
                        block_n=32,
                        num_warps=num_warps,
                        num_stages=num_stages,
                        segment_size=128,
                    )
                )

    for gqa_heads_per_cta in (1, 2, 4):
        candidates.append(cfg(gqa_heads_per_cta=gqa_heads_per_cta))

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
        return paddle.full([batch_size, 1, seq_len, 1], seq_len, dtype=paddle.int32)

    pos = paddle.arange(seq_len, dtype=paddle.int32).reshape([1, 1, seq_len, 1])
    q_end = paddle.full([1, 1, seq_len, 1], seq_len, dtype=paddle.int32)
    window_end = pos + min(seq_len, flashmask_window)
    return paddle.minimum(window_end, q_end).tile([batch_size, 1, 1, 1])


def make_inputs(args, seq_len: int, mode: str, head_dim: int) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
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
) -> float:
    for _ in range(args.warmup):
        run_estimate(q, k, startend, args=args, config=config)
        sync_cuda()

    times = []
    for _ in range(args.iters):
        sync_cuda()
        start = time.perf_counter()
        run_estimate(q, k, startend, args=args, config=config)
        sync_cuda()
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.mean(times)


def format_config(config: RRAttnConfig) -> str:
    fields = ", ".join(f"{field}={getattr(config, field)!r}" for field in TUNABLE_FIELDS)
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
            if item.head_dim == head_dim and item.config_id == config_id and item.ms is not None
        ]
        if len(vals) != total_cases:
            score = float("inf")
        else:
            score = statistics.mean(vals)
        rows.append((score, config_id))
    return sorted(rows, key=lambda item: item[0])


def write_csv(path: Path, results: list[CaseResult], configs_by_head_dim: dict[int, list[RRAttnConfig]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["head_dim", "config_id", "seq_len", "mode", "ms", "error", *TUNABLE_FIELDS])
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
) -> None:
    total_cases = len(seq_lens) * len(modes)

    print("\nTuned metrics:")
    print("  primary score: per-head_dim mean estimate latency in ms across selected seq_lens and modes")
    print("  default seq_len is 32k; pass --seq-lens for broader sweeps")
    print("  chunk_size is not tuned here; set it through --chunk-size or the rrattn wrapper")

    best_by_head_dim = {}
    for head_dim in head_dims:
        configs = configs_by_head_dim[head_dim]
        ranked = summarize(results, configs, head_dim=head_dim, total_cases=total_cases)

        print(f"\nTop configs for head_dim={head_dim}:")
        printed = 0
        for score, config_id in ranked:
            if printed >= topk:
                break
            if score == float("inf"):
                continue
            print(f"  #{config_id}: score_ms={score:.4f} {format_config(configs[config_id])}")
            printed += 1

        if printed == 0:
            print("  no complete config finished all cases")
            continue

        best_score, best_config_id = ranked[0]
        best_by_head_dim[head_dim] = (best_score, best_config_id, configs[best_config_id])

    if best_by_head_dim:
        print("\nManual apply:")
        print("  Edit rrattn/config_rrattn.py get_rrattn_config; for tuned head_dim values, use:")
        for head_dim in head_dims:
            if head_dim not in best_by_head_dim:
                continue
            best_score, best_config_id, best_config = best_by_head_dim[head_dim]
            print(f"      if head_dim == {head_dim}:")
            print(f"          return {format_config(best_config)}  # config #{best_config_id}, score_ms={best_score:.4f}")

    failures = [item for item in results if item.error]
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
    parser = argparse.ArgumentParser(description="Autotune RRAttention estimate kernel configs.")
    parser.add_argument("--seq-lens", default=DEFAULT_SEQ_LENS)
    parser.add_argument("--modes", default=DEFAULT_MODES)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-q-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128, help="Single head_dim used when --head-dims is empty.")
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
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N candidate configs.")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--csv", type=Path, default=None, help="Optional path to save per-case results.")
    parser.add_argument("--gpu-name", default=None, help="Override GPU name used by get_rrattn_config.")
    args = parser.parse_args()

    require_cuda(args.device)
    assert args.num_q_heads % args.num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads"
    assert 128 % args.stride == 0, "stride must divide fixed block size 128"
    assert args.chunk_size > 0, "chunk_size must be positive"

    seq_lens = parse_csv_ints(args.seq_lens)
    modes = parse_modes(args.modes)
    head_dims = parse_csv_ints(args.head_dims) if args.head_dims else [args.head_dim]
    configs_by_head_dim = {}
    for head_dim in head_dims:
        configs = make_candidates(
            head_dim=head_dim,
            gpu_name=args.gpu_name,
        )
        if args.limit is not None:
            configs = configs[: args.limit]
        configs_by_head_dim[head_dim] = configs

    print(f"Running head_dims={head_dims} over seq_lens={seq_lens}, modes={modes}")
    print(f"Tunable fields: {', '.join(TUNABLE_FIELDS)}")

    results = []
    for head_dim in head_dims:
        configs = configs_by_head_dim[head_dim]
        print(f"\nHead dim {head_dim}: {len(configs)} configs")
        for seq_len in seq_lens:
            for mode in modes:
                print(f"\nCase head_dim={head_dim} seq_len={seq_len} mode={mode}")
                q, k, startend = make_inputs(args, seq_len, mode, head_dim)
                for config_id, config in enumerate(configs):
                    try:
                        ms = benchmark_case(q, k, startend, args=args, config=config)
                        results.append(
                            CaseResult(
                                head_dim=head_dim,
                                config_id=config_id,
                                seq_len=seq_len,
                                mode=mode,
                                ms=ms,
                            )
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
    )


if __name__ == "__main__":
    main()
