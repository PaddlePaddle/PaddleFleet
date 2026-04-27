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

import unittest
from unittest.mock import patch

import paddle

from paddlefleet.transformer.moe.fp8_utils import (
    ExpertsGroupGemmContiguousNode,
    _ensure_zero_weight_grad,
    _get_fp8_weight_and_scale,
)


class _WeightWithMainGrad:
    def __init__(self):
        self.shape = [2, 3]
        self.main_grad = None
        self.stop_gradient = False
        self.hook_calls = 0

    def _apply_backward_hook(self):
        self.hook_calls += 1


class _WeightWithGrad:
    def __init__(self):
        self.shape = [2, 3]
        self.grad = None
        self.stop_gradient = False
        self.hook_calls = 0

    def _apply_backward_hook(self):
        self.hook_calls += 1


class TestEnsureZeroWeightGrad(unittest.TestCase):
    def test_initializes_and_clears_main_grad(self):
        weight = _WeightWithMainGrad()

        _ensure_zero_weight_grad(weight)

        self.assertEqual(weight.main_grad.shape, [2, 3])
        self.assertEqual(weight.main_grad.numpy().tolist(), [[0.0] * 3] * 2)
        self.assertEqual(weight.hook_calls, 1)

        weight.main_grad.set_value(paddle.ones([2, 3], dtype="float32"))
        _ensure_zero_weight_grad(weight)

        self.assertEqual(weight.main_grad.numpy().tolist(), [[0.0] * 3] * 2)
        self.assertEqual(weight.hook_calls, 2)

    def test_initializes_and_clears_grad_without_hook_when_stopped(self):
        weight = _WeightWithGrad()
        weight.stop_gradient = True

        _ensure_zero_weight_grad(weight)

        self.assertEqual(weight.grad.shape, [2, 3])
        self.assertEqual(weight.grad.numpy().tolist(), [[0.0] * 3] * 2)
        self.assertEqual(weight.hook_calls, 0)

        weight.stop_gradient = False
        weight.grad.set_value(paddle.ones([2, 3], dtype="float32"))
        _ensure_zero_weight_grad(weight)

        self.assertEqual(weight.grad.numpy().tolist(), [[0.0] * 3] * 2)
        self.assertEqual(weight.hook_calls, 1)


class TestExpertsGroupGemmContiguousNodeCounts(unittest.TestCase):
    def test_get_fp8_weight_and_scale_returns_cached_nontranspose(self):
        weight = _WeightWithGrad()
        weight.fp8_weight_stacked = paddle.ones([2, 3], dtype="float32")
        weight.fp8_scale_stacked = paddle.full([2, 1], 0.5, dtype="float32")

        fp8_weight, fp8_scale = _get_fp8_weight_and_scale(weight)

        self.assertIs(fp8_weight, weight.fp8_weight_stacked)
        self.assertIs(fp8_scale, weight.fp8_scale_stacked)

    def test_gen_m_indices_accepts_list_tensor_and_empty_counts(self):
        node = ExpertsGroupGemmContiguousNode.__new__(
            ExpertsGroupGemmContiguousNode
        )

        self.assertEqual(
            node.gen_m_indices([2, 0, 1]).numpy().tolist(),
            [0, 0, 2],
        )
        self.assertEqual(
            node.gen_m_indices(paddle.to_tensor([1, 0, 2], dtype="int64"))
            .numpy()
            .tolist(),
            [0, 2, 2],
        )
        self.assertEqual(
            node.gen_m_indices(paddle.to_tensor([], dtype="int64")).shape,
            [0],
        )

    def test_fwd_gate_up_builds_deep_gemm_indices_from_tensor_counts(self):
        node = ExpertsGroupGemmContiguousNode.__new__(
            ExpertsGroupGemmContiguousNode
        )
        node.moe_deep_gemm = True
        node.use_fp8_mlp = False
        expected = paddle.ones([3, 4], dtype="float32")
        bf16_calls = []

        def fwd_gate_up_bf16(*args):
            bf16_calls.append(args)
            return expected

        node.fwd_gate_up_bf16 = fwd_gate_up_bf16
        x = paddle.zeros([3, 4], dtype="float32")
        token_counts = paddle.to_tensor([1, 2], dtype="int64")

        out = node.fwd_gate_up(
            x,
            expert_w1=[],
            num_expert=2,
            tokens_per_expert=token_counts,
        )

        self.assertIs(out, expected)
        self.assertIs(node.tokens_per_expert, token_counts)
        self.assertEqual(
            node.tokens_per_expert_indices.numpy().tolist(), [0, 1, 1]
        )
        self.assertEqual(len(bf16_calls), 1)
        self.assertIs(bf16_calls[0][0], x)
        self.assertEqual(bf16_calls[0][1], [])

    def test_fwd_gate_up_builds_deep_gemm_indices_from_list_counts(self):
        node = ExpertsGroupGemmContiguousNode.__new__(
            ExpertsGroupGemmContiguousNode
        )
        node.moe_deep_gemm = True
        node.use_fp8_mlp = False
        expected = paddle.ones([3, 4], dtype="float32")
        node.fwd_gate_up_bf16 = lambda *_args: expected

        node.fwd_gate_up(
            paddle.zeros([3, 4], dtype="float32"),
            expert_w1=[],
            num_expert=2,
            tokens_per_expert=[2, 1],
        )

        self.assertEqual(
            node.tokens_per_expert_indices.numpy().tolist(), [0, 0, 1]
        )

    def test_zero_token_fp8_weight_grad_initializes_expert_grads(self):
        node = ExpertsGroupGemmContiguousNode.__new__(
            ExpertsGroupGemmContiguousNode
        )
        node.tokens_per_expert = paddle.to_tensor([0, 0], dtype="int64")
        node.fused_transpose_split_quant = lambda *_args: (
            [None, None],
            [None, None],
        )
        weights = [_WeightWithMainGrad(), _WeightWithGrad()]

        node.bwd_down_weight(
            paddle.empty([0, 3], dtype="float32"),
            paddle.empty([0, 3], dtype="float32"),
            weights,
        )

        self.assertEqual(weights[0].main_grad.shape, [2, 3])
        self.assertEqual(weights[1].grad.shape, [2, 3])
        self.assertEqual(weights[0].hook_calls, 1)
        self.assertEqual(weights[1].hook_calls, 1)

        node.bwd_gate_up_weight(
            paddle.empty([0, 3], dtype="float32"),
            paddle.empty([0, 3], dtype="float32"),
            weights,
            clear_input=True,
        )

        self.assertIsNone(node.input)
        self.assertIsNone(node.input_fp8)
        self.assertIsNone(node.input_scale)
        self.assertEqual(weights[0].hook_calls, 2)
        self.assertEqual(weights[1].hook_calls, 2)

    def test_bf16_weight_grad_runs_grouped_and_per_expert_contracts(self):
        node = ExpertsGroupGemmContiguousNode.__new__(
            ExpertsGroupGemmContiguousNode
        )
        node.dequant_input = False
        node.input = None
        node.moe_grouped_gemm = True
        node.use_fp8_mlp = False
        node.moe_deep_gemm = False
        node.tokens_per_expert = [1, 1]
        grouped_weight = _WeightWithGrad()
        grouped_weight.shape = [2, 3, 4]
        grouped_weight.grad = paddle.zeros(grouped_weight.shape)

        def batched_gemm(*_args, **_kwargs):
            return paddle.ones(grouped_weight.shape)

        with patch(
            "paddle.incubate.nn.functional.batched_gemm",
            batched_gemm,
        ):
            node.bf16_weight_grad(
                paddle.ones([2, 4], dtype="float32"),
                paddle.ones([2, 3], dtype="float32"),
                grouped_weight,
            )

        self.assertEqual(grouped_weight.hook_calls, 1)
        self.assertEqual(
            grouped_weight.grad.numpy().tolist(), [[[1.0] * 4] * 3] * 2
        )

        node.moe_grouped_gemm = False
        node.tokens_per_expert = [1, 0]
        weights = [_WeightWithGrad(), _WeightWithMainGrad()]
        grad_add_calls = []

        def fused_linear_param_grad_add(*args):
            grad_add_calls.append(args)

        with patch(
            "paddle._C_ops.fused_linear_param_grad_add",
            fused_linear_param_grad_add,
        ):
            node.bf16_weight_grad(
                paddle.ones([1, 4], dtype="float32"),
                paddle.ones([1, 3], dtype="float32"),
                weights,
            )

        self.assertEqual(len(grad_add_calls), 1)
        self.assertEqual(weights[0].hook_calls, 1)
        self.assertEqual(weights[1].hook_calls, 1)


if __name__ == "__main__":
    unittest.main()
