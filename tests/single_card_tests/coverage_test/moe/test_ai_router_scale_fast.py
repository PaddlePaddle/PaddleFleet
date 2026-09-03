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

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../..",
        "src",
    ),
)

import numpy as np
import paddle

from paddlefleet.transformer.moe import moe_router
from paddlefleet.transformer.moe.moe_router import (
    apply_learnable_routed_scaling,
)

NUM_EXPERTS = 8
TOPK = 2


class TestApplyLearnableRoutedScaling(unittest.TestCase):
    """Unit tests for apply_learnable_routed_scaling / GatherExpertScale."""

    def setUp(self):
        np.random.seed(2026)
        paddle.seed(2026)
        self._saved_flag = moe_router._ROUTER_SCALE_FAST

    def tearDown(self):
        moe_router._ROUTER_SCALE_FAST = self._saved_flag

    def _make_inputs(self, num_tokens=6, pad_first_token=False):
        gate = np.random.uniform(0.1, 1.0, [num_tokens, TOPK]).astype("float32")
        idx = np.random.randint(0, NUM_EXPERTS, [num_tokens, TOPK]).astype(
            "int32"
        )
        param = np.random.uniform(0.5, 2.0, [NUM_EXPERTS]).astype("float32")
        if pad_first_token:
            idx[0, :] = -1
            gate[0, :] = 0.0
        return gate, idx, param

    def _run(self, gate, idx, param, fast):
        moe_router._ROUTER_SCALE_FAST = fast
        param_t = paddle.to_tensor(param, stop_gradient=False)
        out = apply_learnable_routed_scaling(
            paddle.to_tensor(gate), paddle.to_tensor(idx), param_t
        )
        return out, param_t

    def _reference(self, gate, idx, param):
        return gate * param[np.clip(idx, 0, None)]

    def _param_grad(self, gate, idx, param, grad_out, fast):
        out, param_t = self._run(gate, idx, param, fast)
        paddle.autograd.backward([out], [paddle.to_tensor(grad_out)])
        return param_t.grad.numpy()

    def test_forward_matches_reference(self):
        """Both paths implement top_gate * param[clip(top_idx, 0)]."""
        gate, idx, param = self._make_inputs()
        expected = self._reference(gate, idx, param)
        for fast in (False, True):
            out, _ = self._run(gate, idx, param, fast)
            self.assertEqual(out.shape, [gate.shape[0], TOPK])
            np.testing.assert_allclose(
                out.numpy(), expected, rtol=1e-6, atol=1e-6
            )

    def test_forward_bit_exact_between_paths(self):
        """The fast gather must not change the forward numerics at all."""
        gate, idx, param = self._make_inputs()
        slow, _ = self._run(gate, idx, param, False)
        fast, _ = self._run(gate, idx, param, True)
        np.testing.assert_array_equal(slow.numpy(), fast.numpy())

    def test_padded_indices_are_clipped(self):
        """top_idx == -1 must be gathered as expert 0 and stay zero-valued."""
        gate, idx, param = self._make_inputs(pad_first_token=True)
        for fast in (False, True):
            out, _ = self._run(gate, idx, param, fast)
            np.testing.assert_array_equal(
                out.numpy()[0], np.zeros([TOPK], dtype="float32")
            )
            np.testing.assert_allclose(
                out.numpy()[1:],
                self._reference(gate, idx, param)[1:],
                rtol=1e-6,
                atol=1e-6,
            )

    def test_param_grad_matches_reference(self):
        """Backward scatters grad_out * top_gate onto the selected experts."""
        gate, idx, param = self._make_inputs(num_tokens=32)
        grad_out = np.random.uniform(-1.0, 1.0, gate.shape).astype("float32")
        expected = np.zeros([NUM_EXPERTS], dtype="float64")
        np.add.at(
            expected,
            np.clip(idx, 0, None),
            (grad_out * gate).astype("float64"),
        )

        for fast in (False, True):
            grad = self._param_grad(gate, idx, param, grad_out, fast)
            np.testing.assert_allclose(grad, expected, rtol=1e-5, atol=1e-5)

    def test_param_grad_matches_between_paths(self):
        """The fast backward must reproduce the F.embedding path's grad.

        Compared head-to-head instead of only against numpy, so a divergence
        between the atomic scatter-add and the dense fp32 column sum shows up
        directly. Padded (-1) rows are covered too: their top_gate is 0, so
        neither path may leak grad into expert 0.
        """
        for pad_first_token in (False, True):
            with self.subTest(pad_first_token=pad_first_token):
                gate, idx, param = self._make_inputs(
                    num_tokens=64, pad_first_token=pad_first_token
                )
                grad_out = np.random.uniform(-1.0, 1.0, gate.shape).astype(
                    "float32"
                )
                slow_grad = self._param_grad(gate, idx, param, grad_out, False)
                fast_grad = self._param_grad(gate, idx, param, grad_out, True)
                grad_diff = fast_grad - slow_grad
                np.testing.assert_allclose(
                    fast_grad, slow_grad, rtol=1e-6, atol=1e-6
                )

    def test_fast_flag_reads_env_once(self):
        moe_router._ROUTER_SCALE_FAST = None
        with patch.dict(os.environ, {"FLEET_MOE_ROUTER_SCALE_FAST": "1"}):
            self.assertTrue(moe_router._router_scale_fast_enabled())
        # Cached: clearing the env afterwards must not flip the switch.
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(moe_router._router_scale_fast_enabled())

        moe_router._ROUTER_SCALE_FAST = None
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(moe_router._router_scale_fast_enabled())


if __name__ == "__main__":
    unittest.main()
