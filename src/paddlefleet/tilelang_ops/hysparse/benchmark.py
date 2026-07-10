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

"""Latency benchmark for the HySparse (MQA) block-attention operators.

Times each operator (block-score fwd/bwd, block-sparse gather fwd/bwd, and the
full block-score -> TopK -> block-sparse pipeline) and reports wall time plus
peak allocated memory. Includes the target MLA scenario head_dim=576,
num_heads=64, seq_len up to 32k.

Run standalone::

    python -m paddlefleet.tilelang_ops.hysparse.benchmark            # full sweep
    python -m paddlefleet.tilelang_ops.hysparse.benchmark --mla-32k  # only 32k
    python -m paddlefleet.tilelang_ops.hysparse.benchmark \
        --seqlens 8192 --H 64 --D 576 --block-B 64 --topk 16
"""

import argparse
import time

import paddle

paddle.enable_compat(scope={"tilelang"}, silent=True)

from .block_score_attn import block_score_mqa_attn_fwd
from .block_score_attn_bwd import block_score_mqa_bwd_interface
from .block_sparse_attn_mqa import block_sparse_mqa_attn_fwd
from .block_sparse_attn_mqa_bwd import block_sparse_mqa_bwd_interface
from .pipeline import hysparse_forward_mqa
from .reference import make_causal_valid_range


def _sync():
    paddle.device.synchronize()


def _time_ms(fn, warmup=2, iters=5):
    """Median-free mean latency in ms over ``iters`` runs after ``warmup``."""
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.time()
    for _ in range(iters):
        fn()
    _sync()
    return (time.time() - t0) / iters * 1000.0


def _peak_gb():
    return paddle.device.cuda.max_memory_allocated() / 1e9


def _rand_inputs(B, S, H, D, seed=0):
    paddle.seed(seed)
    q = paddle.randn([B, S, H, D], dtype="bfloat16")
    k = paddle.randn([B, S, D], dtype="bfloat16")
    v = paddle.randn([B, S, D], dtype="bfloat16")
    return q, k, v


def bench_config(B, S, H, D, block_B, topk, do_bwd=True, iters=5):
    """Benchmark every HySparse MQA op for one shape; returns a result dict."""
    q, k, v = _rand_inputs(B, S, H, D)
    vr = make_causal_valid_range(S, batch=B)
    sm = D**-0.5
    res = {"B": B, "S": S, "H": H, "D": D, "block_B": block_B, "topk": topk}

    # block-score full-attention forward (emits per-block max logits)
    paddle.device.cuda.reset_max_memory_allocated()
    res["score_fwd"] = _time_ms(
        lambda: block_score_mqa_attn_fwd(
            q, k, v, vr, sm_scale=sm, block_B=block_B
        ),
        iters=iters,
    )
    o, lse, _ = block_score_mqa_attn_fwd(
        q, k, v, vr, sm_scale=sm, block_B=block_B
    )

    # end-to-end pipeline (scoring -> TopK -> gather sparse attention)
    paddle.device.cuda.reset_max_memory_allocated()
    res["pipeline"] = _time_ms(
        lambda: hysparse_forward_mqa(
            q, k, v, vr, topk, sm_scale=sm, block_B=block_B
        ),
        iters=iters,
    )
    _, _, idx, _, _ = hysparse_forward_mqa(
        q, k, v, vr, topk, sm_scale=sm, block_B=block_B
    )
    res["pipeline_peak_gb"] = _peak_gb()

    # block-sparse gather forward, given the pipeline's selected indices
    res["sparse_fwd"] = _time_ms(
        lambda: block_sparse_mqa_attn_fwd(
            q, k, v, idx, vr, sm_scale=sm, block_B=block_B
        ),
        iters=iters,
    )
    so, slse = block_sparse_mqa_attn_fwd(
        q, k, v, idx, vr, sm_scale=sm, block_B=block_B
    )

    if do_bwd:
        do = paddle.randn([B, S, H, D], dtype="bfloat16")
        res["score_bwd"] = _time_ms(
            lambda: block_score_mqa_bwd_interface(
                q, k, v, o, do, lse, vr, sm_scale=sm, block_B=block_B
            ),
            iters=iters,
        )
        res["sparse_bwd"] = _time_ms(
            lambda: block_sparse_mqa_bwd_interface(
                q, k, v, so, do, slse, idx, vr, sm_scale=sm, block_B=block_B
            ),
            iters=iters,
        )
    return res


_COLS = [
    ("S", "S", 7),
    ("H", "H", 4),
    ("D", "D", 5),
    ("topk", "topk", 5),
    ("score_fwd", "scoreFwd", 10),
    ("score_bwd", "scoreBwd", 10),
    ("sparse_fwd", "sparseFwd", 10),
    ("sparse_bwd", "sparseBwd", 10),
    ("pipeline", "pipeline", 10),
    ("pipeline_peak_gb", "peakGB", 8),
]


def _fmt(res):
    cells = []
    for key, _, w in _COLS:
        v = res.get(key)
        if v is None:
            s = "-"
        elif isinstance(v, float):
            s = f"{v:.1f}"
        else:
            s = str(v)
        cells.append(s.rjust(w))
    return " ".join(cells)


def _header():
    return " ".join(h.rjust(w) for _, h, w in _COLS)


def run_sweep(seqlens, H, D, block_B, topk, do_bwd, iters):
    print(
        f"HySparse MQA benchmark  (B=1, H={H}, D={D}, block_B={block_B}, "
        f"units=ms, bf16)"
    )
    print(_header())
    for S in seqlens:
        res = bench_config(
            1, S, H, D, block_B, topk, do_bwd=do_bwd, iters=iters
        )
        print(_fmt(res))


def main():
    p = argparse.ArgumentParser(description="HySparse MQA latency benchmark")
    p.add_argument(
        "--seqlens",
        type=int,
        nargs="+",
        default=[2048, 4096, 8192, 16384, 32768],
    )
    p.add_argument("--H", type=int, default=64)
    p.add_argument("--D", type=int, default=576)
    p.add_argument("--block-B", type=int, default=64)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument(
        "--no-bwd",
        action="store_true",
        help="skip backward kernels (forward/pipeline only)",
    )
    p.add_argument(
        "--mla-32k",
        action="store_true",
        help="only the MLA target: D=576,H=64,S=32768",
    )
    args = p.parse_args()

    if args.mla_32k:
        run_sweep(
            [32768],
            64,
            576,
            args.block_B,
            args.topk,
            not args.no_bwd,
            args.iters,
        )
        return
    run_sweep(
        args.seqlens,
        args.H,
        args.D,
        args.block_B,
        args.topk,
        not args.no_bwd,
        args.iters,
    )


if __name__ == "__main__":
    main()
