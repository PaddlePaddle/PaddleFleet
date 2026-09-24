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

"""Unit tests for the accuracy-compatible helpers of ``transformer.mlp``.

The three helpers rewrite where a gradient comes from without changing the
forward value: the router scale reduces in fp32 over a padded row count,
the projection materializes the weight transpose for its input gradient,
and the swiglu is the unfused reference activation. Each is checked for
its forward value *and* for the gradients it routes, plus the branch
``MLP.forward`` picks between them.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import paddle
import paddle.nn.functional as F

from paddlefleet.transformer import mlp


def _tensor(array, stop_gradient=False):
    out = paddle.to_tensor(np.asarray(array, dtype="float32"))
    out.stop_gradient = stop_gradient
    return out


class AccuracyCompatibleSwigluTest(unittest.TestCase):
    """Unfused swiglu: silu on the gate half times the linear half."""

    def test_matches_reference_halves(self):
        hidden = _tensor([[1.0, -2.0, 0.5, 3.0], [0.0, 4.0, -1.5, 2.0]])

        out = mlp._accuracy_compatible_swiglu(hidden)

        gate, linear = hidden[:, :2], hidden[:, 2:]
        np.testing.assert_array_equal(
            out.numpy(), (F.silu(gate) * linear).numpy()
        )
        self.assertEqual(list(out.shape), [2, 2])

    def test_gradient_flows_through_both_halves(self):
        hidden = _tensor([[1.0, -2.0, 0.5, 3.0]])

        mlp._accuracy_compatible_swiglu(hidden).sum().backward()

        reference = _tensor([[1.0, -2.0, 0.5, 3.0]])
        gate, linear = reference[:, :2], reference[:, 2:]
        (F.silu(gate) * linear).sum().backward()
        np.testing.assert_allclose(
            hidden.grad.numpy(), reference.grad.numpy(), rtol=0, atol=0
        )


class AccuracyCompatibleRouterScaleTest(unittest.TestCase):
    """Forward is the plain product; the scale grad reduces in fp32."""

    activation_rows = [[1.0, 2.0, 3.0], [-4.0, 5.0, 0.5], [0.25, -1.0, 2.0]]
    scale_rows = [0.5, -2.0, 3.0]

    def _run(self, reduction_rows):
        activation = _tensor(self.activation_rows)
        scale = _tensor(self.scale_rows)

        out = mlp._accuracy_compatible_router_scale(
            activation, scale, reduction_rows
        )
        out.sum().backward()
        return activation, scale, out

    def test_forward_value_and_dtype(self):
        activation, scale, out = self._run(len(self.activation_rows))

        expected = np.asarray(
            self.activation_rows, dtype="float32"
        ) * np.asarray(self.scale_rows, dtype="float32").reshape(-1, 1)
        np.testing.assert_array_equal(out.numpy(), expected)
        self.assertEqual(out.dtype, activation.dtype)

    def test_activation_grad_comes_from_the_native_path(self):
        activation, _, _ = self._run(len(self.activation_rows))

        # d(sum(act * scale))/d(act) is the broadcast scale, and the scale
        # used there is detached, so this path must not mix in the PyLayer.
        expected = np.tile(
            np.asarray(self.scale_rows, dtype="float32").reshape(-1, 1),
            (1, len(self.activation_rows[0])),
        )
        np.testing.assert_array_equal(activation.grad.numpy(), expected)

    def test_scale_grad_is_padding_invariant(self):
        rows = len(self.activation_rows)
        expected = np.asarray(self.activation_rows, dtype="float32").sum(
            axis=-1
        )

        grads = []
        for reduction_rows in (rows, rows + 5):
            with self.subTest(reduction_rows=reduction_rows):
                _, scale, _ = self._run(reduction_rows)
                np.testing.assert_array_equal(scale.grad.numpy(), expected)
                grads.append(scale.grad.numpy())
        # Zero-padding the grouped reduction must not move a single bit.
        np.testing.assert_array_equal(grads[0], grads[1])

    def test_tensor_reduction_rows_match_int(self):
        # MoELayer passes ``numel()``, a 0-D Tensor, on one of its paths.
        rows = len(self.activation_rows) + 5
        _, int_scale, _ = self._run(rows)
        _, tensor_scale, _ = self._run(paddle.ones([rows]).numel())
        np.testing.assert_array_equal(
            tensor_scale.grad.numpy(), int_scale.grad.numpy()
        )


class AccuracyCompatibleProjectionTest(unittest.TestCase):
    """Linear whose input grad goes through a materialized transpose."""

    hidden_rows = [[1.0, 2.0, -1.0, 0.5], [0.0, -3.0, 2.0, 1.5]]
    weight_rows = [[1.0, -1.0], [2.0, 0.5], [-0.5, 3.0], [0.25, -2.0]]
    bias_row = [0.75, -0.25]

    def _projection(self, skip_bias_add):
        return SimpleNamespace(
            weight=_tensor(self.weight_rows),
            bias=_tensor(self.bias_row),
            skip_bias_add=skip_bias_add,
        )

    def test_skip_bias_add_defers_the_bias_to_the_caller(self):
        projection = self._projection(skip_bias_add=True)
        hidden = _tensor(self.hidden_rows)

        out, out_bias = mlp._accuracy_compatible_projection(projection, hidden)

        self.assertIs(out_bias, projection.bias)
        np.testing.assert_array_equal(
            out.numpy(), F.linear(hidden, projection.weight).numpy()
        )

    def test_fused_bias_is_added_and_receives_its_gradient(self):
        projection = self._projection(skip_bias_add=False)
        hidden = _tensor(self.hidden_rows)

        out, out_bias = mlp._accuracy_compatible_projection(projection, hidden)
        out.sum().backward()

        self.assertIsNone(out_bias)
        np.testing.assert_array_equal(
            out.numpy(),
            F.linear(hidden, projection.weight, projection.bias).numpy(),
        )
        np.testing.assert_array_equal(
            projection.bias.grad.numpy(),
            np.full(len(self.bias_row), len(self.hidden_rows), dtype="float32"),
        )

    def test_input_and_weight_grads_split_over_the_two_paths(self):
        projection = self._projection(skip_bias_add=True)
        hidden = _tensor(self.hidden_rows)

        out, _ = mlp._accuracy_compatible_projection(projection, hidden)
        out.sum().backward()

        hidden_np = np.asarray(self.hidden_rows, dtype="float32")
        weight_np = np.asarray(self.weight_rows, dtype="float32")
        grad_out = np.ones((hidden_np.shape[0], weight_np.shape[1]), "float32")
        # dgrad through the transposed weight, wgrad from the parameter path.
        np.testing.assert_array_equal(
            hidden.grad.numpy(), grad_out @ weight_np.T
        )
        np.testing.assert_array_equal(
            projection.weight.grad.numpy(), hidden_np.T @ grad_out
        )


class ForwardActivationBranchTest(unittest.TestCase):
    """``MLP.forward`` routes the unfused swiglu and the router scale."""

    tokens = 4
    intermediate = 3

    def _instance(self):
        return SimpleNamespace(
            config=SimpleNamespace(
                use_accuracy_compatible=True,
                tensor_model_parallel_size=1,
                bias_activation_fusion=False,
                gated_linear_unit=True,
                gpt_model_use_experimental_version=False,
            ),
            hidden_act=F.silu,
            up_gate_proj=SimpleNamespace(),
            down_proj=SimpleNamespace(),
            _dw_up_gate_point=None,
            _dw_down_point=None,
            inspect_name="coverage_mlp",
        )

    def _hidden_states(self):
        values = np.linspace(
            -2.0, 2.0, self.tokens * 2 * self.intermediate, dtype="float32"
        )
        return _tensor(values.reshape(self.tokens, 2 * self.intermediate))

    def _reference_swiglu(self, hidden):
        gate = hidden[:, : self.intermediate]
        linear = hidden[:, self.intermediate :]
        return F.silu(gate) * linear

    def _forward(self, hidden, per_token_scale, reduction_rows):
        """Run MLP.forward with identity projections around the activation."""
        real_router_scale = mlp._accuracy_compatible_router_scale
        with (
            patch.object(
                mlp,
                "deferrable_linear",
                side_effect=lambda config, point, layer, value: (value, None),
            ) as projections,
            patch.object(
                mlp, "get_tensor_model_parallel_world_size", return_value=1
            ),
            patch.object(
                mlp,
                "_accuracy_compatible_router_scale",
                side_effect=real_router_scale,
            ) as router_scale,
            patch.object(mlp, "nvtx_range_push"),
            patch.object(mlp, "nvtx_range_pop"),
            patch.object(mlp, "get_current_layer", return_value=0),
            patch.object(
                mlp,
                "inspect_tensor",
                side_effect=lambda name, layer, value: value,
            ),
        ):
            output, output_bias = mlp.MLP.forward(
                self._instance(),
                hidden,
                per_token_scale=per_token_scale,
                accuracy_compatible_router_reduction_rows=reduction_rows,
            )
        self.assertIsNone(output_bias)
        self.assertEqual(projections.call_count, 2)
        return output, router_scale

    def test_unscaled_branch_returns_plain_swiglu(self):
        hidden = self._hidden_states()

        output, router_scale = self._forward(hidden, None, None)

        router_scale.assert_not_called()
        np.testing.assert_array_equal(
            output.numpy(), self._reference_swiglu(hidden).numpy()
        )

    def test_scaled_branch_without_reduction_rows_multiplies_inline(self):
        hidden = self._hidden_states()
        scale = _tensor([0.5, -1.5, 2.0, 0.25])

        output, router_scale = self._forward(hidden, scale, None)

        router_scale.assert_not_called()
        expected = self._reference_swiglu(hidden) * scale.unsqueeze(-1)
        np.testing.assert_array_equal(output.numpy(), expected.numpy())
        self.assertEqual(output.dtype, hidden.dtype)

    def test_reduction_rows_select_the_grouped_router_scale(self):
        hidden = self._hidden_states()
        scale = _tensor([0.5, -1.5, 2.0, 0.25])
        reduction_rows = 2 * self.tokens

        output, router_scale = self._forward(hidden, scale, reduction_rows)

        router_scale.assert_called_once()
        self.assertEqual(router_scale.call_args.args[2], reduction_rows)
        # Same forward value as the inline multiply ...
        expected = self._reference_swiglu(hidden) * scale.unsqueeze(-1)
        np.testing.assert_array_equal(output.numpy(), expected.numpy())
        # ... but the scale gradient now comes from the fp32 row reduction.
        # Padded grouped reduction, so compare against the fp32 row sums up
        # to fp32 rounding (bit-for-bit padding invariance is checked in
        # AccuracyCompatibleRouterScaleTest).
        output.sum().backward()
        np.testing.assert_allclose(
            scale.grad.numpy(),
            self._reference_swiglu(hidden).numpy().sum(axis=-1),
            rtol=1e-6,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
