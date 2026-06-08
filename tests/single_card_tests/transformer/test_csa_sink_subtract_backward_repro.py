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

import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np
import paddle

try:
    import torch
    import torch.utils.dlpack
except ImportError:
    torch = None

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from paddlefleet.transformer.csa_attention import (
    _DSV4CSASinkSoftmaxTorchBackward,
)


class _SinkSubtract(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, scores, sink, scores_max):
        return scores - scores_max, sink - scores_max

    @staticmethod
    def backward(ctx, grad_scores_shifted, grad_sink_shifted):
        grad_scores = grad_scores_shifted
        grad_sink = grad_sink_shifted.sum(axis=(0, 2, 3), keepdim=True)
        grad_scores_max = (
            -grad_scores_shifted.sum(axis=-1, keepdim=True) - grad_sink_shifted
        )
        return grad_scores, grad_sink, grad_scores_max


def _md5_float32(array: np.ndarray) -> str:
    return hashlib.md5(array.astype("float32", copy=False).tobytes()).hexdigest()


def _make_inputs(seed=1234, shape=(2, 4, 128, 32)):
    rng = np.random.default_rng(seed)
    scores = (rng.standard_normal(shape).astype("float32") * 2.0).astype(
        "float32"
    )
    sink = (
        rng.standard_normal((1, shape[1], 1, 1)).astype("float32") * 0.2
    ).astype("float32")
    upstream = rng.standard_normal(shape).astype("float32")
    return scores, sink, upstream


def _diff_stats(lhs: np.ndarray, rhs: np.ndarray):
    abs_diff = np.abs(lhs - rhs)
    return int(np.count_nonzero(abs_diff)), float(abs_diff.max())


def _run_paddle_sink_softmax(scores_np, sink_np, upstream_np, mode="default"):
    scores = paddle.to_tensor(scores_np).cast("bfloat16").cast("float32")
    sink = paddle.to_tensor(sink_np).cast("bfloat16").cast("float32")
    scores.stop_gradient = False
    sink.stop_gradient = False

    if mode == "torch_backward":
        attn_weights = _DSV4CSASinkSoftmaxTorchBackward.apply(scores, sink)
    else:
        scores_max = paddle.maximum(scores.max(axis=-1, keepdim=True), sink)
        if mode == "subtract_custom":
            scores_shifted, sink_shifted = _SinkSubtract.apply(
                scores, sink, scores_max
            )
        elif mode == "default":
            scores_shifted = scores - scores_max
            sink_shifted = sink - scores_max
        else:
            raise ValueError(f"unknown mode: {mode}")

        exp_scores = paddle.exp(scores_shifted)
        exp_sink = paddle.exp(sink_shifted)
        denom = exp_scores.sum(axis=-1, keepdim=True) + exp_sink
        attn_weights = exp_scores / denom

    upstream = paddle.to_tensor(upstream_np).cast("bfloat16").cast("float32")
    loss = (attn_weights * upstream).sum()
    loss.backward()
    return (
        scores.grad.detach().cast("float32").cpu().numpy(),
        sink.grad.detach().cast("float32").cpu().numpy(),
    )


def _run_torch_sink_softmax(scores_np, sink_np, upstream_np):
    scores = (
        torch.tensor(scores_np, device="cuda", dtype=torch.bfloat16)
        .float()
        .requires_grad_(True)
    )
    sink = (
        torch.tensor(sink_np, device="cuda", dtype=torch.bfloat16)
        .float()
        .requires_grad_(True)
    )

    scores_max = torch.maximum(scores.max(dim=-1, keepdim=True).values, sink)
    exp_scores = torch.exp(scores - scores_max)
    exp_sink = torch.exp(sink - scores_max)
    denom = exp_scores.sum(dim=-1, keepdim=True) + exp_sink
    attn_weights = exp_scores / denom

    upstream = torch.tensor(upstream_np, device="cuda", dtype=torch.bfloat16).float()
    loss = (attn_weights * upstream).sum()
    loss.backward()
    return (
        scores.grad.detach().float().cpu().numpy(),
        sink.grad.detach().float().cpu().numpy(),
    )


@unittest.skipUnless(
    torch is not None and paddle.is_compiled_with_cuda() and torch.cuda.is_available(),
    "CSA sink subtract repro requires CUDA Paddle and CUDA Torch.",
)
class TestCSASinkSoftmaxBackwardRepro(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu:0")
        torch.cuda.set_device(0)

    def test_torch_backward_matches_torch_reference_across_seeds(self):
        shapes = [
            (2, 4, 128, 32),
            (1, 8, 64, 64),
            (3, 2, 257, 17),
        ]
        seeds = [0, 1, 2]
        num_cases = 0
        paddle_default_torch_diff_cases = 0
        subtract_custom_torch_diff_cases = 0
        torch_backward_torch_diff_cases = 0
        sink_torch_backward_diff_cases = 0
        sink_torch_diff_cases = 0

        for shape in shapes:
            for seed in seeds:
                scores_np, sink_np, upstream_np = _make_inputs(seed, shape)
                default_scores_grad, default_sink_grad = _run_paddle_sink_softmax(
                    scores_np, sink_np, upstream_np, mode="default"
                )
                subtract_scores_grad, _ = _run_paddle_sink_softmax(
                    scores_np, sink_np, upstream_np, mode="subtract_custom"
                )
                torch_bwd_scores_grad, torch_bwd_sink_grad = (
                    _run_paddle_sink_softmax(
                        scores_np, sink_np, upstream_np, mode="torch_backward"
                    )
                )
                torch_scores_grad, torch_sink_grad = _run_torch_sink_softmax(
                    scores_np, sink_np, upstream_np
                )

                default_torch_numdiff, default_torch_maxdiff = _diff_stats(
                    default_scores_grad, torch_scores_grad
                )
                subtract_torch_numdiff, subtract_torch_maxdiff = _diff_stats(
                    subtract_scores_grad, torch_scores_grad
                )
                torch_bwd_torch_numdiff, torch_bwd_torch_maxdiff = _diff_stats(
                    torch_bwd_scores_grad, torch_scores_grad
                )
                sink_torch_numdiff, _ = _diff_stats(
                    default_sink_grad, torch_sink_grad
                )
                sink_torch_bwd_numdiff, _ = _diff_stats(
                    torch_bwd_sink_grad, torch_sink_grad
                )

                print(
                    f"shape={shape} seed={seed} "
                    f"default_torch_numdiff={default_torch_numdiff} "
                    f"default_torch_maxdiff={default_torch_maxdiff:.12e} "
                    f"subtract_torch_numdiff={subtract_torch_numdiff} "
                    f"subtract_torch_maxdiff={subtract_torch_maxdiff:.12e} "
                    f"torch_bwd_torch_numdiff={torch_bwd_torch_numdiff} "
                    f"torch_bwd_torch_maxdiff={torch_bwd_torch_maxdiff:.12e} "
                    f"torch_bwd_md5={_md5_float32(torch_bwd_scores_grad)} "
                    f"torch_ref_md5={_md5_float32(torch_scores_grad)}"
                )

                num_cases += 1
                paddle_default_torch_diff_cases += int(default_torch_numdiff > 0)
                subtract_custom_torch_diff_cases += int(
                    subtract_torch_numdiff > 0
                )
                torch_backward_torch_diff_cases += int(
                    torch_bwd_torch_numdiff > 0
                )
                sink_torch_backward_diff_cases += int(
                    sink_torch_bwd_numdiff > 0
                )
                sink_torch_diff_cases += int(sink_torch_numdiff > 0)

        self.assertEqual(paddle_default_torch_diff_cases, num_cases)
        self.assertEqual(subtract_custom_torch_diff_cases, num_cases)
        self.assertEqual(torch_backward_torch_diff_cases, 0)
        self.assertEqual(sink_torch_backward_diff_cases, 0)
        self.assertEqual(sink_torch_diff_cases, 0)


if __name__ == "__main__":
    unittest.main()
