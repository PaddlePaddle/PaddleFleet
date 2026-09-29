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

"""Behavior tests for the learnable routed-scaling-factor gather.

Target production entry:
    paddlefleet.transformer.moe.moe_router
        apply_learnable_routed_scaling
        GatherExpertScale (PyLayer, the FLEET_MOE_ROUTER_SCALE_FAST path)
        _router_scale_fast_enabled (env-cached opt-in switch)

``apply_learnable_routed_scaling(top_gate, top_idx, param)`` multiplies each
selected expert's gate value by that expert's learnable scaling factor:

    out[i, j] = top_gate[i, j] * param[max(top_idx[i, j], 0)]

Padded slots carry top_idx == -1; the function clips indices to 0 (expert 0)
so the gather stays in range. There are two implementations that must agree:
the default ``F.embedding`` path and the opt-in ``GatherExpertScale`` PyLayer
selected by the cached ``FLEET_MOE_ROUTER_SCALE_FAST`` env flag. Forward is a
plain gather in both; the param gradient is a scatter-add of ``grad_out *
top_gate`` onto the selected experts.

All expected values here are derived by hand from that definition with small
fixed, position-distinguishable inputs -- no expected value is produced by
calling the function under test, and neither implementation is mocked. The
env-flag global ``_ROUTER_SCALE_FAST`` is a module-level cache; every test
saves and restores it via addCleanup so path selection cannot leak between
tests or into other files in the same process.

CPU-only: this file needs a working Paddle install (the math is device
independent, so we only claim to have verified the CPU path). When Paddle or
paddlefleet is not importable the whole case is skipped with an honest reason
rather than reported as passing.
"""

import os
import unittest
from unittest import mock

try:
    import numpy as np
    import paddle

    from paddlefleet.transformer.moe import moe_router
    from paddlefleet.transformer.moe.moe_router import (
        apply_learnable_routed_scaling,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet unavailable in this env
    np = None
    paddle = None
    moe_router = None
    apply_learnable_routed_scaling = None
    _IMPORT_ERROR = repr(exc)


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet not importable: {_IMPORT_ERROR}",
)
class TestApplyLearnableRoutedScaling(unittest.TestCase):
    """Forward / backward contract of apply_learnable_routed_scaling."""

    @classmethod
    def setUpClass(cls):
        # Device-independent gather/scatter math; pin to CPU so a visible
        # accelerator does not change what is exercised.
        paddle.set_device("cpu")

    def setUp(self):
        # _ROUTER_SCALE_FAST is a process-global cache for the env switch.
        # Save it and restore in cleanup so setting it here (to force each
        # implementation) never leaks into later tests or other files.
        saved = moe_router._ROUTER_SCALE_FAST
        self.addCleanup(
            lambda: setattr(moe_router, "_ROUTER_SCALE_FAST", saved)
        )

    def _apply(self, gate, idx, param, fast, param_grad=False, gate_grad=False):
        """Run the real entry with the chosen implementation selected.

        Setting the module cache to a concrete bool makes
        ``_router_scale_fast_enabled`` return it directly, so ``fast=False``
        exercises the F.embedding path and ``fast=True`` the GatherExpertScale
        PyLayer -- both are real production code, nothing is mocked.
        """
        moe_router._ROUTER_SCALE_FAST = fast
        gate_t = paddle.to_tensor(gate, dtype="float32")
        idx_t = paddle.to_tensor(idx, dtype="int32")
        param_t = paddle.to_tensor(param, dtype="float32")
        gate_t.stop_gradient = not gate_grad
        param_t.stop_gradient = not param_grad
        out = apply_learnable_routed_scaling(gate_t, idx_t, param_t)
        return out, gate_t, param_t

    def test_forward_hand_derived(self):
        """out[i,j] == top_gate[i,j] * param[top_idx[i,j]] on both paths."""
        param = [1.0, 2.0, 3.0, 4.0]  # 4 experts, distinct values
        idx = [[0, 2], [3, 1]]
        gate = [[0.5, 0.25], [2.0, 4.0]]
        # By hand: gate * param[idx]
        #   row0: [0.5*1, 0.25*3] = [0.5, 0.75]
        #   row1: [2.0*4, 4.0*2] = [8.0, 8.0]
        expected = np.array([[0.5, 0.75], [8.0, 8.0]], dtype=np.float32)
        for fast in (False, True):
            with self.subTest(fast=fast):
                out, _, _ = self._apply(gate, idx, param, fast)
                self.assertEqual(out.shape, [2, 2])
                np.testing.assert_allclose(
                    out.numpy(), expected, rtol=1e-6, atol=1e-6
                )

    def test_forward_paths_bit_exact(self):
        """The opt-in gather must not change the forward numerics at all."""
        param = [0.7, 1.3, -2.1, 5.5, 0.01]
        idx = [[0, 4], [2, 1], [3, 0]]
        gate = [[1.5, 0.2], [-0.3, 2.2], [0.9, 1.1]]
        slow, _, _ = self._apply(gate, idx, param, False)
        fast, _, _ = self._apply(gate, idx, param, True)
        np.testing.assert_array_equal(slow.numpy(), fast.numpy())

    def test_padding_index_clipped_to_expert_zero(self):
        """top_idx == -1 gathers expert 0 (clip min=0), not param[-1].

        param[0] and param[-1] are made distinct, and the padded slot is given
        a nonzero gate so the result distinguishes clip-to-0 from a raw
        negative gather (which would wrap to the last expert).
        """
        param = [1.0, 2.0, 3.0, 9.0]  # param[0]=1.0, param[-1]=9.0 differ
        idx = [[-1, 2]]
        gate = [[1.0, 0.5]]
        # clip(-1,0) -> expert 0: [1.0*param[0], 0.5*param[2]] = [1.0, 1.5]
        expected = np.array([[1.0, 1.5]], dtype=np.float32)
        for fast in (False, True):
            with self.subTest(fast=fast):
                out, _, _ = self._apply(gate, idx, param, fast)
                np.testing.assert_allclose(
                    out.numpy(), expected, rtol=1e-6, atol=1e-6
                )

    def test_param_gradient_scatter_add(self):
        """param.grad[e] = sum of grad_out*top_gate over slots routed to e."""
        param = [1.0, 2.0, 3.0, 4.0]
        idx = [[0, 2], [0, 3]]  # expert 0 selected twice
        gate = [[0.5, 1.0], [2.0, 3.0]]
        grad_out = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        # grad into gathered scales = grad_out * gate:
        #   [[5.0, 20.0], [60.0, 120.0]]
        # scatter-add onto experts by idx:
        #   e0: (0,0)+(1,0) = 5.0 + 60.0 = 65.0
        #   e1: none                     = 0.0
        #   e2: (0,1)                    = 20.0
        #   e3: (1,1)                    = 120.0
        expected = np.array([65.0, 0.0, 20.0, 120.0], dtype=np.float32)
        for fast in (False, True):
            with self.subTest(fast=fast):
                out, _, param_t = self._apply(
                    gate, idx, param, fast, param_grad=True
                )
                paddle.autograd.backward([out], [paddle.to_tensor(grad_out)])
                self.assertIsNotNone(param_t.grad)
                np.testing.assert_allclose(
                    param_t.grad.numpy(), expected, rtol=1e-6, atol=1e-6
                )

    def test_top_gate_gradient(self):
        """top_gate.grad[i,j] = grad_out[i,j] * param[clip(top_idx,0)]."""
        param = [1.0, 2.0, 3.0, 4.0]
        idx = [[0, 2], [3, 1]]
        gate = [[0.5, 0.25], [2.0, 4.0]]
        grad_out = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        # d out / d gate = param[idx]:
        #   row0: [10*param[0], 20*param[2]] = [10, 60]
        #   row1: [30*param[3], 40*param[1]] = [120, 80]
        expected = np.array([[10.0, 60.0], [120.0, 80.0]], dtype=np.float32)
        for fast in (False, True):
            with self.subTest(fast=fast):
                out, gate_t, _ = self._apply(
                    gate, idx, param, fast, gate_grad=True
                )
                paddle.autograd.backward([out], [paddle.to_tensor(grad_out)])
                self.assertIsNotNone(gate_t.grad)
                np.testing.assert_allclose(
                    gate_t.grad.numpy(), expected, rtol=1e-6, atol=1e-6
                )

    def test_paths_agree_on_param_gradient(self):
        """Both implementations produce the same param gradient (fp32)."""
        param = [0.7, 1.3, -2.1, 5.5, 0.01]
        idx = [[0, 4], [2, 1], [4, 0]]  # experts 0 and 4 repeat across rows
        gate = [[1.5, 0.2], [-0.3, 2.2], [0.9, 1.1]]
        grad_out = np.array(
            [[1.0, -2.0], [3.0, 0.5], [-1.5, 2.0]], dtype=np.float32
        )
        grads = {}
        for fast in (False, True):
            out, _, param_t = self._apply(
                gate, idx, param, fast, param_grad=True
            )
            paddle.autograd.backward([out], [paddle.to_tensor(grad_out)])
            grads[fast] = param_t.grad.numpy()
        np.testing.assert_allclose(
            grads[True], grads[False], rtol=1e-6, atol=1e-6
        )

    def test_fast_flag_reads_env_once_and_caches(self):
        """_router_scale_fast_enabled reads FLEET_MOE_ROUTER_SCALE_FAST once."""
        moe_router._ROUTER_SCALE_FAST = None
        with mock.patch.dict(os.environ, {"FLEET_MOE_ROUTER_SCALE_FAST": "1"}):
            self.assertTrue(moe_router._router_scale_fast_enabled())
        # Cached: clearing the env afterwards must not flip the switch.
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(moe_router._router_scale_fast_enabled())

    def test_fast_flag_default_off(self):
        """Absent/"0" env resolves to the F.embedding path (False)."""
        moe_router._ROUTER_SCALE_FAST = None
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(moe_router._router_scale_fast_enabled())

        moe_router._ROUTER_SCALE_FAST = None
        with mock.patch.dict(os.environ, {"FLEET_MOE_ROUTER_SCALE_FAST": "0"}):
            self.assertFalse(moe_router._router_scale_fast_enabled())


if __name__ == "__main__":
    unittest.main()
