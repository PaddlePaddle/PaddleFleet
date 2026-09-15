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

"""Behavior tests for paddlefleet.transformers.linear_utils.

linear_utils is a vendor-dispatch aliasing module (model layer). Its real
job is to bind the exported symbols to the correct underlying implementation
based on the runtime device, resolved once at import time via get_env_device.

Independent oracle: the true upstream Paddle classes are imported directly by
this test (paddle.nn.Linear, mpu.ColumnParallelLinear/RowParallelLinear,
sequence_parallel_utils.Column/RowSequenceParallelLinear). We assert the
module's exports are IDENTICAL to those upstream classes, so a mis-wired alias
(e.g. Column bound to Row, or a seq-parallel export left as the dummy Pass
fallback) is rejected. Column vs Row are distinct classes, so a swap changes
the identity, not just the type.

The exported Linear on the default/CPU path is a working affine layer; we
drive it with fixed weights/bias and compare against a hand-derived numpy
matmul (independent reference), so a transposed weight or dropped bias fails.

The device branch (npu / xpu) is real production logic that runs at import.
We exercise it by patching the SOURCE get_env_device and reloading the module,
then observing the resulting bindings, restoring the pristine module on
cleanup. This is no-card / CPU only; no distributed init is required.
"""

import importlib
import importlib.util
import unittest
from unittest import mock

import numpy as np
import paddle
import paddle.distributed.fleet.meta_parallel as mpu
from paddle import nn
from paddle.distributed.fleet.utils import sequence_parallel_utils

import paddlefleet.transformers.linear_utils as lu
import paddlefleet.transformers.mc2_parallel_linear as mc2
import paddlefleet.utils.tools as tools


class TestDefaultBindings(unittest.TestCase):
    """On the default (non-npu/non-xpu) path each export must be the exact
    upstream class, not merely 'some object'."""

    def test_linear_is_paddle_native(self):
        self.assertIs(lu.Linear, nn.Linear)

    def test_column_parallel_is_upstream(self):
        self.assertIs(lu.ColumnParallelLinear, mpu.ColumnParallelLinear)

    def test_row_parallel_is_upstream(self):
        self.assertIs(lu.RowParallelLinear, mpu.RowParallelLinear)

    def test_column_and_row_parallel_are_distinct(self):
        # A copy-paste swap (Column bound to Row) would collapse these.
        self.assertIsNot(lu.ColumnParallelLinear, lu.RowParallelLinear)

    def test_column_seq_parallel_is_real_not_dummy(self):
        self.assertIs(
            lu.ColumnSequenceParallelLinear,
            sequence_parallel_utils.ColumnSequenceParallelLinear,
        )
        # The except-branch fallback would be the local Pass dummy.
        self.assertNotEqual(
            lu.ColumnSequenceParallelLinear.__name__,
            "ColumnSequenceParallelLinearPass",
        )
        self.assertTrue(issubclass(lu.ColumnSequenceParallelLinear, nn.Layer))

    def test_row_seq_parallel_is_real_not_dummy(self):
        self.assertIs(
            lu.RowSequenceParallelLinear,
            sequence_parallel_utils.RowSequenceParallelLinear,
        )
        self.assertNotEqual(
            lu.RowSequenceParallelLinear.__name__,
            "RowSequenceParallelLinearPass",
        )
        self.assertTrue(issubclass(lu.RowSequenceParallelLinear, nn.Layer))

    def test_seq_parallel_column_and_row_distinct(self):
        self.assertIsNot(
            lu.ColumnSequenceParallelLinear, lu.RowSequenceParallelLinear
        )

    def test_all_exports_content(self):
        # __all__ is the public contract; assert its exact membership and that
        # every listed name resolves to the independently-expected object.
        self.assertEqual(
            set(lu.__all__),
            {
                "Linear",
                "ColumnParallelLinear",
                "RowParallelLinear",
                "ColumnSequenceParallelLinear",
                "RowSequenceParallelLinear",
            },
        )
        expected = {
            "Linear": nn.Linear,
            "ColumnParallelLinear": mpu.ColumnParallelLinear,
            "RowParallelLinear": mpu.RowParallelLinear,
            "ColumnSequenceParallelLinear": (
                sequence_parallel_utils.ColumnSequenceParallelLinear
            ),
            "RowSequenceParallelLinear": (
                sequence_parallel_utils.RowSequenceParallelLinear
            ),
        }
        for name, obj in expected.items():
            self.assertIs(getattr(lu, name), obj)


class TestExportedLinearComputes(unittest.TestCase):
    """The exported Linear must be a functioning affine layer, verified
    numerically against an independent numpy reference."""

    def test_forward_matches_hand_matmul(self):
        layer = lu.Linear(3, 2)
        # Fixed, all-distinct weight [in, out] and bias so a transpose or a
        # dropped bias changes the numbers.
        w = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype="float32")
        b = np.array([10.0, 20.0], dtype="float32")
        layer.weight.set_value(paddle.to_tensor(w))
        layer.bias.set_value(paddle.to_tensor(b))

        x = np.array([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]], dtype="float32")
        out = layer(paddle.to_tensor(x))

        ref = x @ w + b  # independent of the layer under test
        self.assertEqual(list(out.shape), [2, 2])
        np.testing.assert_allclose(out.numpy(), ref, atol=1e-5, rtol=1e-6)

    def test_bias_actually_added(self):
        # Distinguish "bias consumed" from "bias ignored": same weights, the
        # only change is a nonzero bias, so the delta must equal that bias.
        layer = lu.Linear(2, 2)
        w = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
        layer.weight.set_value(paddle.to_tensor(w))
        x = paddle.to_tensor(np.array([[3.0, 5.0]], dtype="float32"))

        layer.bias.set_value(paddle.zeros([2], dtype="float32"))
        no_bias = layer(x).numpy().copy()
        layer.bias.set_value(paddle.to_tensor(np.array([7.0, -4.0], "float32")))
        with_bias = layer(x).numpy()

        np.testing.assert_allclose(
            with_bias - no_bias, [[7.0, -4.0]], atol=1e-6
        )


class TestDeviceDispatch(unittest.TestCase):
    """Exercise the real import-time device branch by reloading the module
    with a patched SOURCE get_env_device, restoring pristine state after."""

    def setUp(self):
        # Restore the genuinely-imported (real device) module regardless of
        # what any test reloads, so we do not pollute other tests.
        self.addCleanup(lambda: importlib.reload(lu))

    def _reload_with_device(self, device):
        with mock.patch.object(tools, "get_env_device", return_value=device):
            importlib.reload(lu)

    def test_npu_without_mc2_kernels_keeps_seq_parallel_binding(self):
        # On a no-card host MC2 classes are None; the `is not None` guard must
        # therefore leave the seq-parallel exports as the upstream classes
        # rather than overriding them (or setting them to None).
        if mc2.MC2ColumnSeqParallelLinear is not None:
            self.skipTest("MC2 kernels present; guard scenario not exercised")
        self._reload_with_device("npu")
        self.assertIs(
            lu.ColumnSequenceParallelLinear,
            sequence_parallel_utils.ColumnSequenceParallelLinear,
        )
        self.assertIs(
            lu.RowSequenceParallelLinear,
            sequence_parallel_utils.RowSequenceParallelLinear,
        )
        self.assertIs(lu.Linear, nn.Linear)

    def test_xpu_without_paddle_xpu_falls_back_to_native(self):
        # The xpu branch tries to import paddle_xpu; when absent the
        # `except ImportError: pass` must keep the native implementations
        # (not crash, not leave them undefined).
        if importlib.util.find_spec("paddle_xpu") is not None:
            self.skipTest("paddle_xpu installed; ImportError fallback not hit")
        self._reload_with_device("xpu")
        self.assertIs(lu.Linear, nn.Linear)
        self.assertIs(lu.ColumnParallelLinear, mpu.ColumnParallelLinear)
        self.assertIs(lu.RowParallelLinear, mpu.RowParallelLinear)

    def test_reload_restores_default_bindings(self):
        # Confirms the cleanup contract: after reloading under a foreign
        # device and then restoring, the default bindings are back.
        self._reload_with_device("npu")
        importlib.reload(lu)  # explicit restore mirrors addCleanup
        self.assertIs(lu.Linear, nn.Linear)
        self.assertIs(lu.ColumnParallelLinear, mpu.ColumnParallelLinear)


if __name__ == "__main__":
    unittest.main()
