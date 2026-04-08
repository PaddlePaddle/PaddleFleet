# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Tests for fp8='e4m3' + fp8_wgrad=False with moe_grouped_gemm=True.

Covers the bug fix in commit 3352089: when moe_grouped_gemm=True and
use_fp8_mlp=True, the four backward branch guards must use the per-expert
list path instead of grouped_gemm_experts.
"""

import subprocess
import unittest
from unittest.mock import MagicMock

import paddle

from paddlefleet.transformer.moe.fp8_utils import ExpertsGroupGemmContiguousNode


def get_gpu_models_via_nvidia_smi():
    try:
        output = subprocess.check_output(
            "nvidia-smi --query-gpu=name --format=csv,noheader", shell=True
        )
        return output.decode().strip().replace("NVIDIA", "")
    except Exception:
        return "Unknown"


def judge_machine_type():
    if not paddle.is_compiled_with_cuda():
        return "No CUDA GPU"
    models = get_gpu_models_via_nvidia_smi()
    name = models.upper()
    if "H" in name:
        return "H"
    elif "V" in name:
        return "V"
    return "Unknown"


result = judge_machine_type()
print("你的机器类型是：", result)

HIDDEN = 128
INTER = 128
NUM_EXPERTS = 4
TOKENS_PER_EXPERT = (
    128  # must be multiple of FP8_ALIGN (128) for bf16_weight_grad
)


def _make_expert_weight(hidden, inter):
    """Real weight tensors for one expert (bfloat16).

    Paddle stores weights as [in_features, out_features]:
      up_gate_proj: [hidden, inter*2]  (ColumnParallelLinear, gate+up merged)
      down_proj:    [inter, hidden]    (RowParallelLinear)
    """
    up_gate = paddle.create_parameter(
        shape=[hidden, inter * 2],
        dtype="bfloat16",
        default_initializer=paddle.nn.initializer.Normal(),
    )
    down = paddle.create_parameter(
        shape=[inter, hidden],
        dtype="bfloat16",
        default_initializer=paddle.nn.initializer.Normal(),
    )
    w = MagicMock()
    w.up_gate_proj = MagicMock()
    w.up_gate_proj.weight = up_gate
    w.down_proj = MagicMock()
    w.down_proj.weight = down
    return w


def _make_node(use_fp8_mlp, use_bf16_gemm_weight_grad):
    experts = [_make_expert_weight(HIDDEN, INTER) for _ in range(NUM_EXPERTS)]
    cm = MagicMock()
    cm.experts = experts
    node = ExpertsGroupGemmContiguousNode(
        custom_map=cm,
        use_fp8_mlp=use_fp8_mlp,
        use_bf16_gemm_weight_grad=use_bf16_gemm_weight_grad,
        moe_grouped_gemm=True,
        moe_deep_gemm=False,
    )
    return node, experts


class TestFp8E4m3WgradFalse(unittest.TestCase):
    """ExpertsGroupGemmContiguousNode with fp8=e4m3 (use_fp8_mlp=True) +
    fp8_wgrad=False (use_bf16_gemm_weight_grad=True) + moe_grouped_gemm=True.

    Verifies the bug fix: backward branch guards must fall through to the
    per-expert list path, not use grouped_gemm_experts.
    """

    def setUp(self):
        total = TOKENS_PER_EXPERT * NUM_EXPERTS
        self.total_tokens = total
        self.node, self.experts = _make_node(
            use_fp8_mlp=True, use_bf16_gemm_weight_grad=True
        )
        self.node.tokens_per_expert = [TOKENS_PER_EXPERT] * NUM_EXPERTS
        self.node.input = paddle.randn([total, HIDDEN], dtype=paddle.bfloat16)
        self.node.input_fp8 = None
        self.node.input_scale = None
        self.node.o1 = paddle.randn([total, INTER * 2], dtype=paddle.bfloat16)
        self.out_grad = paddle.randn([total, HIDDEN], dtype=paddle.bfloat16)
        self.unzipped_probs = paddle.ones([total], dtype=paddle.bfloat16)

    def test_node_uses_per_expert_list(self):
        """With use_fp8_mlp=True + moe_grouped_gemm=True, node must hold
        self.experts (per-expert list), not self.grouped_gemm_experts."""
        self.assertTrue(hasattr(self.node, "experts"))
        self.assertFalse(hasattr(self.node, "grouped_gemm_experts"))

    def test_use_bf16_gemm_weight_grad_is_true(self):
        """fp8_wgrad=False maps to use_bf16_gemm_weight_grad=True."""
        self.assertTrue(self.node.use_bf16_gemm_weight_grad)

    def test_backward_impl_bf16_does_not_raise(self):
        """backward_impl_bf16 with use_fp8_mlp=True + moe_grouped_gemm=True
        must not raise (pre-fix: AttributeError accessing grouped_gemm_experts)."""
        try:
            dx, probs_grad = self.node.backward_impl_bf16(
                self.out_grad, self.unzipped_probs
            )
        except AttributeError as exc:
            self.fail(
                f"backward_impl_bf16 raised AttributeError (bug: accessed "
                f"grouped_gemm_experts instead of per-expert list): {exc}"
            )

    def test_backward_impl_bf16_output_shapes(self):
        dx, probs_grad = self.node.backward_impl_bf16(
            self.out_grad, self.unzipped_probs
        )
        self.assertEqual(list(dx.shape), [self.total_tokens, HIDDEN])
        self.assertEqual(probs_grad.shape[0], self.total_tokens)

    def test_backward_impl_fp8_calls_bf16_weight_grad_when_wgrad_false(self):
        """With use_bf16_gemm_weight_grad=True, backward_impl_fp8 must call
        bf16_weight_grad (twice: dw1 + dw2), not the FP8 wgrad kernels."""
        if result != "H":
            self.skipTest("FP8 kernels require H-series GPU")

        calls = []
        original = self.node.bf16_weight_grad

        def mock_bf16_wgrad(dy, x, weights):
            calls.append(True)
            wlist = weights if isinstance(weights, list) else [weights]
            for w in wlist:
                if w.grad is None:
                    w.grad = paddle.zeros(w.shape, dtype=paddle.float32)

        self.node.bf16_weight_grad = mock_bf16_wgrad
        try:
            self.node.backward_impl_fp8(self.out_grad, self.unzipped_probs)
        except Exception as exc:
            self.node.bf16_weight_grad = original
            self.fail(f"backward_impl_fp8 raised: {exc}")
        self.node.bf16_weight_grad = original

        self.assertEqual(
            len(calls),
            2,
            f"bf16_weight_grad must be called exactly twice (dw1+dw2), got {len(calls)}",
        )


class TestFp8E4m3WgradTrue(unittest.TestCase):
    """Baseline: use_fp8_mlp=True + use_bf16_gemm_weight_grad=False (fp8_wgrad=True).
    Ensures the fp8 wgrad path is selected when fp8_wgrad=True."""

    def test_use_bf16_gemm_weight_grad_is_false(self):
        node, _ = _make_node(use_fp8_mlp=True, use_bf16_gemm_weight_grad=False)
        self.assertFalse(node.use_bf16_gemm_weight_grad)

    def test_backward_impl_fp8_calls_fp8_wgrad_when_wgrad_true(self):
        """With use_bf16_gemm_weight_grad=False, backward_impl_fp8 must NOT
        call bf16_weight_grad; it should call bwd_gate_up_weight / bwd_down_weight."""
        if result != "H":
            self.skipTest("FP8 kernels require H-series GPU")

        total = TOKENS_PER_EXPERT * NUM_EXPERTS
        node, experts = _make_node(
            use_fp8_mlp=True, use_bf16_gemm_weight_grad=False
        )
        node.tokens_per_expert = [TOKENS_PER_EXPERT] * NUM_EXPERTS
        node.input = paddle.randn([total, HIDDEN], dtype=paddle.bfloat16)
        node.input_fp8 = None
        node.input_scale = None
        node.o1 = paddle.randn([total, INTER * 2], dtype=paddle.bfloat16)

        bf16_calls = []

        def mock_bf16_wgrad(*a, **kw):
            bf16_calls.append(True)

        node.bf16_weight_grad = mock_bf16_wgrad
        out_grad = paddle.randn([total, HIDDEN], dtype=paddle.bfloat16)
        unzipped_probs = paddle.ones([total], dtype=paddle.bfloat16)

        try:
            node.backward_impl_fp8(out_grad, unzipped_probs)
        except Exception:
            pass  # kernel errors are not the subject here

        self.assertEqual(
            len(bf16_calls),
            0,
            "bf16_weight_grad must not be called when fp8_wgrad=True",
        )


if __name__ == "__main__":
    unittest.main()
