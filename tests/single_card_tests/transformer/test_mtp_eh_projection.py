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

"""Tests for the accuracy-compatible MTP ``eh_proj`` entry point."""

import unittest

import numpy as np
import paddle

from paddlefleet.transformer.multi_token_prediction import _mtp_eh_projection

HIDDEN = 4


class _Projection:
    """Stand-in for ColumnParallelLinear: the weight is stored as [in, out]."""

    def __init__(self, *, bias=True, skip_bias_add=None):
        self.weight = paddle.randn([2 * HIDDEN, HIDDEN])
        self.bias = paddle.randn([HIDDEN]) if bias else None
        if skip_bias_add is not None:
            self.skip_bias_add = skip_bias_add
        self.calls = 0

    def __call__(self, hidden_states):
        self.calls += 1
        return "module-output", None


class MtpEhProjectionTests(unittest.TestCase):
    def setUp(self):
        paddle.seed(0)
        self.hidden_states = paddle.randn([3, 2, 2 * HIDDEN])

    def _expected(self, projection, with_bias):
        out = paddle.matmul(self.hidden_states, projection.weight)
        if with_bias:
            out = out + projection.bias
        return out.numpy()

    def test_tp1_projects_with_the_stored_in_out_weight(self):
        projection = _Projection(skip_bias_add=False)
        out, output_bias = _mtp_eh_projection(
            projection, self.hidden_states, 1, use_accuracy_compatible=True
        )
        self.assertEqual(out.shape, [3, 2, HIDDEN])
        np.testing.assert_allclose(
            out.numpy(), self._expected(projection, True), rtol=1e-6
        )
        self.assertIsNone(output_bias)
        self.assertEqual(projection.calls, 0)

    def test_skip_bias_add_returns_the_bias_separately(self):
        projection = _Projection(skip_bias_add=True)
        out, output_bias = _mtp_eh_projection(
            projection, self.hidden_states, 1, use_accuracy_compatible=True
        )
        np.testing.assert_allclose(
            out.numpy(), self._expected(projection, False), rtol=1e-6
        )
        self.assertIs(output_bias, projection.bias)

    def test_modules_without_skip_bias_add_fold_the_bias(self):
        # paddle.incubate.nn.FusedLinear has no ``skip_bias_add`` attribute.
        projection = _Projection(skip_bias_add=None)
        out, output_bias = _mtp_eh_projection(
            projection, self.hidden_states, 1, use_accuracy_compatible=True
        )
        np.testing.assert_allclose(
            out.numpy(), self._expected(projection, True), rtol=1e-6
        )
        self.assertIsNone(output_bias)

    def test_other_configurations_call_the_module(self):
        for use_accuracy_compatible, tp in ((False, 1), (True, 2)):
            projection = _Projection(skip_bias_add=False)
            out = _mtp_eh_projection(
                projection,
                self.hidden_states,
                tp,
                use_accuracy_compatible=use_accuracy_compatible,
            )
            self.assertEqual(out, ("module-output", None))
            self.assertEqual(projection.calls, 1)


if __name__ == "__main__":
    unittest.main()
