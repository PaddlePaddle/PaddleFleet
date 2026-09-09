#!/usr/bin/env python3

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

"""
PF(Paddle) vs MG(Megatron) 每 step loss md5 对比

数据来源: 两侧训练日志里的两个精度对齐锚点（per-rank、per-microbatch 打印）
  锚点1  per_token_loss: rank=0 shape=[1, 50] md5=xxxx
         (CE 直出、mask/归一化前)
  锚点2  final_loss: rank=0 val=8.997344... md5=xxxx
         (mask + 除 valid_token 后的标量, 未跨 DP all-reduce)

用法:
  python compare_loss.py logs/paddle/20260731-172215 logs/torch/20260731-172403
  python compare_loss.py logs/paddle logs/torch  # 自动取各自最新的时间戳子目录
  # 也可直接传日志文件；-m 指定每 step 的 microbatch 数(默认 1)
"""

import argparse
import os
import re
import sys
from collections import defaultdict

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

_PER_TOKEN_RE = re.compile(
    r"per_token_loss:\s*rank=(?P<rank>\d+)\s+shape=\[(?P<shape>[^\]]*)\]\s+md5=(?P<md5>[0-9a-fA-F]+)"
)
_FINAL_RE = re.compile(
    r"final_loss:\s*rank=(?P<rank>\d+)\s+val=(?P<val>[-\d.eE+naN]+)\s+md5=(?P<md5>[0-9a-fA-F]+)"
)


def _is_log_file(name: str) -> bool:
    return name.startswith("workerlog.") or name.endswith((".log", ".out"))


def _resolve_run_dir(path: str) -> str:
    """
    支持两种输入:
      1. 原始模式: 直接指向某次运行的日志目录/文件，如 logs/paddle/20260731-172215
      2. 简化模式: 指向日志根目录，如 logs/paddle，自动选取其中最新的运行子目录
    """
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        sys.exit(f"路径不存在: {path}")
    if any(_is_log_file(n) for n in os.listdir(path)):
        return path
    # 只在"直接含日志文件"的子目录里挑最新的，跳过 checkpoint/vdl 等无关子目录
    # （这类目录常因训练过程中持续写文件而 mtime 更新，比日志目录还新）
    candidates = [
        os.path.join(path, name)
        for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
        and any(_is_log_file(n) for n in os.listdir(os.path.join(path, name)))
    ]
    if not candidates:
        sys.exit(
            f"目录下没有可解析的日志文件，也没有含日志文件的运行子目录: {path}"
        )
    latest = max(candidates, key=os.path.getmtime)
    print(f"[自动识别] {path} -> {latest}")
    return latest


def _iter_log_files(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        sys.exit(f"路径不存在: {path}")
    files = [
        os.path.join(path, name)
        for name in sorted(os.listdir(path))
        if os.path.isfile(os.path.join(path, name)) and _is_log_file(name)
    ]
    if not files:
        sys.exit(f"目录下没有可解析的日志文件: {path}")
    return files


def parse_dir(path: str) -> dict[int, dict[str, list[dict]]]:
    """
    返回 {rank: {'per_token': [...], 'final': [...]}}，列表按日志出现顺序，
    每个元素是一次 microbatch 的记录。
    """
    out: dict[int, dict[str, list[dict]]] = defaultdict(
        lambda: {"per_token": [], "final": []}
    )
    for f in _iter_log_files(path):
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = _ANSI_RE.sub("", line)
                if "per_token_loss:" not in line and "final_loss:" not in line:
                    continue
                # 同一行可能拼接了多条打印（多 rank stdout 交错），用 finditer
                for m in _PER_TOKEN_RE.finditer(line):
                    out[int(m.group("rank"))]["per_token"].append(
                        {
                            "shape": m.group("shape").strip(),
                            "md5": m.group("md5"),
                        }
                    )
                for m in _FINAL_RE.finditer(line):
                    out[int(m.group("rank"))]["final"].append(
                        {"val": m.group("val"), "md5": m.group("md5")}
                    )
    return out


def _cell(pf: dict, mg: dict, key: str) -> tuple[str, bool]:
    if pf is None and mg is None:
        return "both-missing", False
    if pf is None:
        return "仅MG", False
    if mg is None:
        return "仅PF", False
    return ("✅" if pf[key] == mg[key] else "❌"), pf[key] == mg[key]


def _get(recs: list[dict], i: int):
    return recs[i] if i < len(recs) else None


def _diff_str(pf: dict, mg: dict, key: str) -> str:
    try:
        return f"  diff={float(pf[key]) - float(mg[key]):+.6e}"
    except (TypeError, ValueError, KeyError):
        return ""


def compare(pf: dict, mg: dict, mb: int) -> int:
    ranks = sorted(set(pf) | set(mg))
    if not ranks:
        sys.exit(
            "两侧日志都没解析到 per_token_loss / final_loss，检查是否开启了 "
            "FLAGS_use_accuracy_compatible_kernel / _use_accuracy_compatible"
        )

    empty = {"per_token": [], "final": []}
    n_mb = max(
        max(len(pf.get(r, empty)[k]), len(mg.get(r, empty)[k]))
        for r in ranks
        for k in ("per_token", "final")
    )
    n_step = (n_mb + mb - 1) // mb

    print(f"ranks={ranks}  steps={n_step}  每 step microbatch={mb}")
    for r in ranks:
        print(
            f"  rank {r}: PF 记录 per_token={len(pf.get(r, empty)['per_token'])} "
            f"final={len(pf.get(r, empty)['final'])} | "
            f"MG 记录 per_token={len(mg.get(r, empty)['per_token'])} "
            f"final={len(mg.get(r, empty)['final'])}"
        )
    print()

    bad_steps = set()
    for step in range(1, n_step + 1):
        print("=" * 84)
        print(f"  STEP {step}")
        print("=" * 84)
        for rank in ranks:
            pf_r, mg_r = pf.get(rank, empty), mg.get(rank, empty)
            for mbi in range(mb):
                i = (step - 1) * mb + mbi
                if i >= n_mb:
                    break
                tag = f"rank{rank}" if mb == 1 else f"rank{rank} mb{mbi}"

                for kind, key, label in (
                    ("per_token", "shape", "per_token_loss"),
                    ("final", "val", "final_loss"),
                ):
                    p, m = _get(pf_r[kind], i), _get(mg_r[kind], i)
                    status, ok = _cell(p, m, "md5")
                    if not ok:
                        bad_steps.add(step)
                    print(f"  [{tag}] {label:<15s} {status}")
                    extra = (
                        _diff_str(p, m, key)
                        if (p and m and kind == "final" and not ok)
                        else ""
                    )
                    pf_key = f"{key}={p[key]}" if p else "-"
                    mg_key = f"{key}={m[key]}" if m else "-"
                    print(
                        f"      PF: md5={p['md5'] if p else '-':<34s} {pf_key}"
                    )
                    print(
                        f"      MG: md5={m['md5'] if m else '-':<34s} {mg_key}{extra}"
                    )
        print()

    print("=" * 84)
    if bad_steps:
        print(
            f"  结论: 存在不一致，首个分叉 step = {min(bad_steps)}；"
            f"不一致 step 列表 = {sorted(bad_steps)}"
        )
    else:
        print(
            "  结论: 所有 rank / 所有 step 的 per_token_loss + final_loss md5 完全一致 ✅"
        )
    print("=" * 84)
    return 1 if bad_steps else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="PF vs MG 每 step loss md5 对比")
    ap.add_argument(
        "pf_path",
        help="PF(Paddle) 日志目录或文件, 如 logs/paddle/20260731-172215；"
        "也可传 logs/paddle 自动取最新运行子目录",
    )
    ap.add_argument(
        "mg_path",
        help="MG(Torch) 日志目录或文件, 如 logs/torch/20260731-172403；"
        "也可传 logs/torch 自动取最新运行子目录",
    )
    ap.add_argument(
        "-m",
        "--microbatches",
        type=int,
        default=1,
        help="每个 global step 的 microbatch 数（默认 1）",
    )
    args = ap.parse_args()

    pf_path = _resolve_run_dir(args.pf_path)
    mg_path = _resolve_run_dir(args.mg_path)

    print(f"PF: {pf_path}")
    print(f"MG: {mg_path}\n")
    return compare(parse_dir(pf_path), parse_dir(mg_path), args.microbatches)


if __name__ == "__main__":
    sys.exit(main())
