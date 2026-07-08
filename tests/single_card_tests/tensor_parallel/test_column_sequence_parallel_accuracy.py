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

import types
import unittest
from unittest.mock import patch

import paddle

from paddlefleet.tensor_parallel.layers import (
    ColumnParallelLinear,
    column_sequence_parallel_linear,
)


class TestColumnSequenceParallelAccuracyGate(unittest.TestCase):
    """Tests for the TP=1 BF16 accuracy gate in experimental column linear."""

    def _fake_tensor(self, dtype):
        return types.SimpleNamespace(dtype=dtype)

    def test_accuracy_gate_uses_f_linear_for_bf16_tp1(self):
        x = self._fake_tensor(paddle.bfloat16)
        weight = self._fake_tensor(paddle.bfloat16)
        bias = object()
        sentinel = object()

        with (
            patch(
                "paddlefleet.tensor_parallel.layers.F.linear",
                return_value=sentinel,
            ) as mock_linear,
            patch("paddle.incubate.nn.functional.fused_linear") as mock_fused,
        ):
            output = column_sequence_parallel_linear(
                x,
                weight,
                bias,
                mp_group=None,
                use_accuracy_compatible=True,
            )

        self.assertIs(output, sentinel)
        mock_linear.assert_called_once_with(x, weight, bias)
        mock_fused.assert_not_called()

    def test_default_path_keeps_fused_linear(self):
        x = self._fake_tensor(paddle.bfloat16)
        weight = self._fake_tensor(paddle.bfloat16)
        bias = object()
        sentinel = object()

        with (
            patch("paddlefleet.tensor_parallel.layers.F.linear") as mock_linear,
            patch(
                "paddle.incubate.nn.functional.fused_linear",
                return_value=sentinel,
            ) as mock_fused,
        ):
            output = column_sequence_parallel_linear(
                x,
                weight,
                bias,
                mp_group=None,
                use_accuracy_compatible=False,
            )

        self.assertIs(output, sentinel)
        mock_fused.assert_called_once_with(x, weight, bias)
        mock_linear.assert_not_called()

    def test_accuracy_gate_requires_bf16_weight(self):
        x = self._fake_tensor(paddle.bfloat16)
        weight = self._fake_tensor(paddle.float32)
        bias = object()
        sentinel = object()

        with (
            patch("paddlefleet.tensor_parallel.layers.F.linear") as mock_linear,
            patch(
                "paddle.incubate.nn.functional.fused_linear",
                return_value=sentinel,
            ) as mock_fused,
        ):
            output = column_sequence_parallel_linear(
                x,
                weight,
                bias,
                mp_group=None,
                use_accuracy_compatible=True,
            )

        self.assertIs(output, sentinel)
        mock_fused.assert_called_once_with(x, weight, bias)
        mock_linear.assert_not_called()

    def test_column_parallel_forward_passes_config_gate(self):
        layer = types.SimpleNamespace(
            weight=object(),
            bias=object(),
            skip_bias_add=False,
            tp_group=None,
            config=types.SimpleNamespace(
                gpt_model_use_experimental_version=True,
                use_accuracy_compatible=True,
            ),
        )
        input_ = object()
        sentinel = object()

        with patch(
            "paddlefleet.tensor_parallel.layers.column_sequence_parallel_linear",
            return_value=sentinel,
        ) as mock_column_linear:
            output, output_bias = ColumnParallelLinear.forward(layer, input_)

        self.assertIs(output, sentinel)
        self.assertIsNone(output_bias)
        mock_column_linear.assert_called_once_with(
            input_,
            layer.weight,
            layer.bias,
            mp_group=layer.tp_group,
            use_accuracy_compatible=True,
        )


if __name__ == "__main__":
    unittest.main()
