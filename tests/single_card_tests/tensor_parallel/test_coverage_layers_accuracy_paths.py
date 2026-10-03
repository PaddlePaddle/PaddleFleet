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

"""Unit tests for the accuracy-compatible paths of ``tensor_parallel.layers``.

Two pieces are covered:

* ``_EmbedFp32MainGrad.backward``, the embedding lookup whose weight
  gradient is deposited into an fp32 ``main_grad`` and deliberately never
  returned, so MixPrecision cannot merge a bf16 gradient on top.
* ``Linear._mark_replicated_grad_needs_tp_reduction``, the gate that
  decides when a replicated parameter's wgrad has to be reduced over the
  tensor-parallel group.

Both are exercised as unbound functions against stubs: single card, no
process group, and ``backward`` is a static method that only needs a ctx
with ``saved_tensor()`` and ``weight_ref``.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import paddle

from paddlefleet.tensor_parallel import layers

VOCAB = 4
HIDDEN = 3


def _weight():
    values = np.linspace(-1.0, 1.0, VOCAB * HIDDEN, dtype="float32")
    weight = paddle.to_tensor(values.reshape(VOCAB, HIDDEN))
    weight.stop_gradient = False
    return weight


def _ctx(weight, ids):
    return SimpleNamespace(
        saved_tensor=lambda: (paddle.to_tensor(ids, dtype="int64"),),
        weight_ref=weight,
    )


def _expected_wgrad(ids, grad_out):
    expected = np.zeros((VOCAB, HIDDEN), dtype="float32")
    for row, token in enumerate(ids):
        expected[token] += grad_out[row]
    return expected


class EmbedFp32MainGradBackwardTest(unittest.TestCase):
    """The wgrad is deposited in fp32 and never handed back to autograd."""

    ids = [2, 0, 2]

    def setUp(self):
        self.weight = _weight()
        self.grad_out = np.asarray(
            [[1.0, 2.0, -1.0], [0.5, 0.5, 0.5], [-3.0, 1.0, 0.25]],
            dtype="float32",
        )
        self.grad_output = paddle.to_tensor(self.grad_out)

    def _backward(self):
        return layers._EmbedFp32MainGrad.backward(
            _ctx(self.weight, self.ids), self.grad_output
        )

    def test_missing_buffer_is_created_from_the_row_gradient(self):
        grads = self._backward()

        # Nothing is returned for either input: weight.grad must stay empty
        # so MixPrecision cannot add a bf16 gradient on top.
        self.assertEqual(grads, (None, None))
        self.assertIsNone(self.weight.grad)
        self.assertEqual(self.weight.main_grad.dtype, paddle.float32)
        np.testing.assert_allclose(
            self.weight.main_grad.numpy(),
            _expected_wgrad(self.ids, self.grad_out),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_existing_buffer_is_accumulated_in_place(self):
        self.weight.main_grad = paddle.ones([VOCAB, HIDDEN], dtype="float32")
        buffer_before = self.weight.main_grad

        self._backward()

        self.assertIs(self.weight.main_grad, buffer_before)
        np.testing.assert_allclose(
            self.weight.main_grad.numpy(),
            1.0 + _expected_wgrad(self.ids, self.grad_out),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_repeated_ids_accumulate_instead_of_overwriting(self):
        self._backward()

        # Rows 0 and 2 of grad_output both belong to token 2.
        np.testing.assert_allclose(
            self.weight.main_grad.numpy()[2],
            self.grad_out[0] + self.grad_out[2],
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            self.weight.main_grad.numpy()[1],
            np.zeros(HIDDEN, dtype="float32"),
        )

    def test_grad_added_flag_is_raised_when_the_attribute_exists(self):
        self.weight.grad_added_to_main_grad = False

        self._backward()

        self.assertTrue(self.weight.grad_added_to_main_grad)

    def test_unused_weight_leaves_the_buffer_alone(self):
        self.weight.main_grad = None

        with patch.object(
            paddle.autograd, "grad", return_value=(None,)
        ) as autograd_grad:
            grads = self._backward()

        autograd_grad.assert_called_once()
        self.assertEqual(grads, (None, None))
        self.assertIsNone(self.weight.main_grad)

    def test_the_outer_grad_mode_is_restored(self):
        paddle.set_grad_enabled(False)
        self.addCleanup(paddle.set_grad_enabled, True)

        self._backward()

        # The body re-enables grad so the lookup is a real IndexingBackward;
        # the caller's mode has to come back afterwards.
        self.assertFalse(paddle.is_grad_enabled())
        self.assertIsNotNone(self.weight.main_grad)


class ReplicatedGradTpReductionGateTest(unittest.TestCase):
    """``Linear._mark_replicated_grad_needs_tp_reduction`` gate matrix."""

    def _linear(
        self,
        *,
        is_expert=False,
        use_accuracy_compatible=True,
        sequence_parallel=True,
        tensor_model_parallel_size=2,
    ):
        return SimpleNamespace(
            is_expert=is_expert,
            config=SimpleNamespace(
                use_accuracy_compatible=use_accuracy_compatible,
                sequence_parallel=sequence_parallel,
                tensor_model_parallel_size=tensor_model_parallel_size,
            ),
        )

    def _mark(self, linear):
        parameter = paddle.ones([2, 2], dtype="float32")
        with patch.object(
            layers, "mark_as_sequence_parallel_parameter"
        ) as marker:
            result = layers.Linear._mark_replicated_grad_needs_tp_reduction(
                linear, parameter
            )
        self.assertIsNone(result)
        return marker, parameter

    def test_sequence_parallel_replicated_parameter_is_marked(self):
        marker, parameter = self._mark(self._linear())

        marker.assert_called_once_with(parameter)

    def test_single_rank_is_not_marked(self):
        # Without TP there is no partial sum to reduce, so the default
        # Linear path stays untouched.
        marker, _ = self._mark(self._linear(tensor_model_parallel_size=1))

        marker.assert_not_called()

    def test_expert_parameters_are_left_to_the_expert_reduction(self):
        marker, _ = self._mark(self._linear(is_expert=True))

        marker.assert_not_called()

    def test_gate_is_off_without_accuracy_compatible(self):
        marker, _ = self._mark(self._linear(use_accuracy_compatible=False))

        marker.assert_not_called()

    def test_gate_is_off_without_sequence_parallel(self):
        marker, _ = self._mark(self._linear(sequence_parallel=False))

        marker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
