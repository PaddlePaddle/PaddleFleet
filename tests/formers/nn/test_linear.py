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

"""Behavior tests for paddlefleet.nn.linear.Linear factory.

Environment: 无卡 (CPU only). The ``default`` linear_type maps to Paddle's
native ``nn.Linear`` and is fully exercised on CPU with hand-derived matmul
references (forward + all three gradients). The TP variants (colwise / rowwise
/ sequence_*) map to fleet meta-parallel layers whose numerics require a real
tensor-parallel process group; their CPU-verifiable routing is checked via the
class mapping and get_linear_type, while their cross-rank numeric behavior is
explicitly skipped (see test_tp_variants_numerics_require_real_process_group).
Faking world_size + mocking collectives to assert numerics would be an
antipattern and is deliberately avoided.
"""

import unittest

import numpy as np
import paddle
import paddle.nn as nn

from paddlefleet.nn.linear import Linear
from paddlefleet.transformers import LlamaConfig
from paddlefleet.transformers.linear_utils import (
    ColumnParallelLinear,
    ColumnSequenceParallelLinear,
    RowParallelLinear,
    RowSequenceParallelLinear,
)


def _cpu_config(tensor_model_parallel_size=1, sequence_parallel=False):
    """Real config object exposing exactly the two fields the factory reads.

    get_linear_type only consumes ``tensor_model_parallel_size`` and
    ``sequence_parallel``; a real LlamaConfig carries genuine defaults for both
    (unlike a MagicMock, whose ``sequence_parallel`` would be truthy).
    """
    config = LlamaConfig()
    config.tensor_model_parallel_size = tensor_model_parallel_size
    config.sequence_parallel = sequence_parallel
    return config


class TestLinearDefaultForwardBackward(unittest.TestCase):
    """The default path builds a real nn.Linear; verify its numerics on CPU."""

    def setUp(self):
        paddle.set_device("cpu")
        # Fixed, distinct, non-symmetric values so a transposed weight, a wrong
        # matmul orientation, a dropped bias or a scaled gradient all change the
        # observed numbers. in_features=3, out_features=2.
        self.W = np.array(
            [[1.0, -2.0], [0.5, 3.0], [-1.0, 0.25]], dtype="float32"
        )
        self.b = np.array([0.5, -1.0], dtype="float32")
        self.x = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0]], dtype="float32")

    def _build_default_linear(self, has_bias=True):
        config = _cpu_config(tensor_model_parallel_size=1)
        linear = Linear.create(3, 2, config=config, has_bias=has_bias)
        # The default factory branch must produce Paddle's native Linear.
        self.assertIsInstance(linear, nn.Linear)
        # Paddle stores the weight as [in_features, out_features].
        self.assertEqual(linear.weight.shape, [3, 2])
        return linear

    def test_default_forward_matches_handcomputed_matmul(self):
        linear = self._build_default_linear(has_bias=True)
        linear.weight.set_value(paddle.to_tensor(self.W))
        linear.bias.set_value(paddle.to_tensor(self.b))

        out = linear(paddle.to_tensor(self.x))
        # Independent reference: y = x @ W + b computed with numpy, not the layer.
        expected = self.x @ self.W + self.b
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_default_backward_matches_handcomputed_gradients(self):
        linear = self._build_default_linear(has_bias=True)
        linear.weight.set_value(paddle.to_tensor(self.W))
        linear.bias.set_value(paddle.to_tensor(self.b))

        x = paddle.to_tensor(self.x)
        x.stop_gradient = False
        # Non-uniform upstream gradient exposes scale / orientation errors.
        g = np.array([[2.0, -1.0], [0.5, 4.0]], dtype="float32")

        out = linear(x)
        loss = (out * paddle.to_tensor(g)).sum()
        loss.backward()

        # d(loss)/d(out) == g, so:
        #   dx = g @ W^T,  dW = x^T @ g,  db = sum_over_batch(g)
        expected_dx = g @ self.W.T
        expected_dw = self.x.T @ g
        expected_db = g.sum(axis=0)

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(linear.weight.grad)
        self.assertIsNotNone(linear.bias.grad)
        np.testing.assert_allclose(
            x.grad.numpy(), expected_dx, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            linear.weight.grad.numpy(), expected_dw, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            linear.bias.grad.numpy(), expected_db, rtol=1e-6, atol=1e-6
        )

    def test_has_bias_false_drops_bias_term(self):
        # has_bias=False -> bias_attr=False -> no bias parameter, output is pure
        # matmul. Contrast against the bias path to prove the flag is consumed.
        linear = self._build_default_linear(has_bias=False)
        linear.weight.set_value(paddle.to_tensor(self.W))
        self.assertIsNone(linear.bias)

        out = linear(paddle.to_tensor(self.x))
        expected_no_bias = self.x @ self.W
        np.testing.assert_allclose(
            out.numpy(), expected_no_bias, rtol=1e-6, atol=1e-6
        )
        # The difference from the biased path is exactly the broadcast bias.
        biased = self._build_default_linear(has_bias=True)
        biased.weight.set_value(paddle.to_tensor(self.W))
        biased.bias.set_value(paddle.to_tensor(self.b))
        diff = biased(paddle.to_tensor(self.x)).numpy() - out.numpy()
        np.testing.assert_allclose(
            diff, np.broadcast_to(self.b, (2, 2)), rtol=1e-6, atol=1e-6
        )

    def test_weight_attr_is_passed_through_to_constructor(self):
        # A ParamAttr with a Constant initializer must reach the real
        # constructor; observe the initialized weight content, not just type.
        config = _cpu_config(tensor_model_parallel_size=1)
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(0.3)
        )
        linear = Linear.create(
            3, 2, weight_attr=weight_attr, has_bias=False, config=config
        )
        np.testing.assert_allclose(
            linear.weight.numpy(),
            np.full((3, 2), 0.3, dtype="float32"),
            rtol=1e-6,
            atol=1e-6,
        )


class TestGetLinearType(unittest.TestCase):
    """Type selection is pure Python logic; verify every branch and ordering."""

    def test_single_gpu_is_default_even_with_sequence_parallel(self):
        # tp<=1 short-circuits to "default" BEFORE the sequence_parallel check;
        # a wrong ordering would leak a "sequence_" prefix here.
        config = _cpu_config(
            tensor_model_parallel_size=1, sequence_parallel=True
        )
        self.assertEqual(
            Linear.get_linear_type(config, tp_plan="colwise"), "default"
        )
        self.assertEqual(
            Linear.get_linear_type(config, tp_plan="rowwise"), "default"
        )

    def test_multi_gpu_uses_tp_plan_verbatim(self):
        config = _cpu_config(tensor_model_parallel_size=2)
        self.assertEqual(
            Linear.get_linear_type(config, tp_plan="colwise"), "colwise"
        )
        self.assertEqual(
            Linear.get_linear_type(config, tp_plan="rowwise"), "rowwise"
        )

    def test_multi_gpu_sequence_parallel_prepends_prefix(self):
        config = _cpu_config(
            tensor_model_parallel_size=2, sequence_parallel=True
        )
        self.assertEqual(
            Linear.get_linear_type(config, tp_plan="colwise"),
            "sequence_colwise",
        )
        self.assertEqual(
            Linear.get_linear_type(config, tp_plan="rowwise"),
            "sequence_rowwise",
        )


class TestGetLinearKwargs(unittest.TestCase):
    """The factory translates a uniform has_bias into the per-class kwarg name.

    Exact-dict comparison catches both a wrong key name (e.g. native nn.Linear
    needs ``bias_attr`` while the TP layers need ``has_bias``) and a leaked
    field (colwise must not carry input_is_parallel, rowwise must not carry
    gather_output).
    """

    def test_default_uses_bias_attr_key(self):
        self.assertEqual(
            Linear.get_linear_kwargs("default", has_bias=True),
            {"bias_attr": True},
        )
        self.assertEqual(
            Linear.get_linear_kwargs("default", has_bias=False),
            {"bias_attr": False},
        )

    def test_colwise_carries_gather_output_only(self):
        self.assertEqual(
            Linear.get_linear_kwargs(
                "colwise", has_bias=True, gather_output=True
            ),
            {"has_bias": True, "gather_output": True},
        )

    def test_rowwise_carries_input_is_parallel_only(self):
        self.assertEqual(
            Linear.get_linear_kwargs(
                "rowwise", has_bias=False, input_is_parallel=False
            ),
            {"has_bias": False, "input_is_parallel": False},
        )

    def test_sequence_variants_mirror_their_base(self):
        self.assertEqual(
            Linear.get_linear_kwargs(
                "sequence_colwise", has_bias=True, gather_output=False
            ),
            {"has_bias": True, "gather_output": False},
        )
        self.assertEqual(
            Linear.get_linear_kwargs(
                "sequence_rowwise", has_bias=True, input_is_parallel=True
            ),
            {"has_bias": True, "input_is_parallel": True},
        )


class TestLinearCreateContracts(unittest.TestCase):
    def test_create_requires_linear_type_or_config(self):
        # Explicit exception contract, not a swallowed error.
        with self.assertRaises(ValueError):
            Linear.create(64, 128)

    def test_explicit_default_type_bypasses_config(self):
        paddle.set_device("cpu")
        linear = Linear.create(64, 128, linear_type="default", has_bias=False)
        self.assertIsInstance(linear, nn.Linear)
        self.assertEqual(linear.weight.shape, [64, 128])

    def test_mapping_routes_to_real_production_classes(self):
        # CPU-verifiable routing: the factory dispatches to the genuine
        # production classes. A mis-wired mapping (e.g. colwise -> rowwise)
        # would be caught here without instantiating a TP group.
        self.assertIs(Linear._global_mapping["default"], nn.Linear)
        self.assertIs(Linear._global_mapping["colwise"], ColumnParallelLinear)
        self.assertIs(Linear._global_mapping["rowwise"], RowParallelLinear)
        self.assertIs(
            Linear._global_mapping["sequence_colwise"],
            ColumnSequenceParallelLinear,
        )
        self.assertIs(
            Linear._global_mapping["sequence_rowwise"],
            RowSequenceParallelLinear,
        )

    @unittest.skip(
        "colwise/rowwise/sequence_* map to fleet meta-parallel layers whose "
        "forward/backward numerics (weight sharding, all-gather / reduce-scatter "
        "along the tp axis) are only meaningful under a real tensor-parallel "
        "process group. This belongs in tests/multi_card_tests; verifying it by "
        "faking world_size and mocking collectives would not prove cross-rank "
        "behavior. Routing to these classes is covered on CPU by "
        "TestGetLinearType and test_mapping_routes_to_real_production_classes."
    )
    def test_tp_variants_numerics_require_real_process_group(self):
        raise AssertionError("must run under a real multi-card TP group")


if __name__ == "__main__":
    unittest.main()
