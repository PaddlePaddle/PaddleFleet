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

"""FusedTwoViewGate must be numerically identical to two independent
FusedGateDetachMatmul calls, forward and backward.

Split-feature routing scores each expert with two gate projections sharing one
input. Running them as two separate autograd nodes makes the shared input
accumulate three gradient terms (view0 + view1 + expert-dispatch); fp32 3-term
reassociation is then order-sensitive and drifts ~1 ULP when the gate's reshape
moves in/out of the PyLayer. FusedTwoViewGate fuses the two views into one node
so the input sees a single combined gate term -- the 2-term structure that is
bitwise-stable. These tests pin that the fusion changes only the accumulation
topology, never the per-view values.
"""

import unittest

import numpy as np
import paddle
from paddle.distributed.fleet.meta_parallel.zero_bubble_utils import (
    WeightGradStore,
)

from paddlefleet.transformer.moe.moe_router import (
    FusedGateDetachMatmul,
    FusedTwoViewGate,
)


def _md5(t):
    return t.astype("float32")._md5sum()


def _weight_with_main_grad(shape, value=None):
    """Parameter mimicking AMP-O2 main_grad + _apply_backward_hook.

    ``value`` fixes the weight data so two independently-created params compare
    apples to apples (initializer.Normal() would consume RNG differently each
    call and give different weights).
    """
    if value is None:
        init = paddle.nn.initializer.Normal()
    else:
        init = paddle.nn.initializer.Assign(value)
    w = paddle.create_parameter(
        shape=shape,
        dtype="float32",
        default_initializer=init,
    )
    w.main_grad = None
    w._hook_call_count = 0

    def _hook():
        w._hook_call_count += 1

    w._apply_backward_hook = _hook
    return w


class TestFusedTwoViewGate(unittest.TestCase):
    def setUp(self):
        paddle.seed(2026)
        np.random.seed(2026)
        # gate weights are [num_experts, hidden]; the op transposes internally.
        self.N, self.D, self.E = 32, 16, 8

    def _inputs(self, shape, dtype):
        x_np = np.random.randn(*shape).astype("float32")
        w0_np = np.random.randn(self.E, self.D).astype("float32")
        w1_np = np.random.randn(self.E, self.D).astype("float32")
        return x_np, w0_np, w1_np

    def _run_split(self, x_np, w0_np, w1_np, use_acc):
        """Two independent FusedGateDetachMatmul, grads accumulate on x."""
        x = paddle.to_tensor(x_np)
        x.stop_gradient = False
        w0 = paddle.to_tensor(w0_np)
        w0.stop_gradient = False
        w1 = paddle.to_tensor(w1_np)
        w1.stop_gradient = False
        l0 = FusedGateDetachMatmul.apply(x, w0, False, use_acc)
        l1 = FusedGateDetachMatmul.apply(x, w1, False, use_acc)
        (l0.sum() + l1.sum()).backward()
        return l0, l1, x.grad, w0.grad, w1.grad

    def _run_fused(self, x_np, w0_np, w1_np, use_acc):
        """FusedTwoViewGate: one node, combined x grad."""
        x = paddle.to_tensor(x_np)
        x.stop_gradient = False
        w0 = paddle.to_tensor(w0_np)
        w0.stop_gradient = False
        w1 = paddle.to_tensor(w1_np)
        w1.stop_gradient = False
        l0, l1 = FusedTwoViewGate.apply(x, w0, w1, False, use_acc)
        (l0.sum() + l1.sum()).backward()
        return l0, l1, x.grad, w0.grad, w1.grad

    def _assert_bitwise(self, x_np, w0_np, w1_np, use_acc):
        s = self._run_split(x_np, w0_np, w1_np, use_acc)
        f = self._run_fused(x_np, w0_np, w1_np, use_acc)
        names = ["logits_0", "logits_1", "x.grad", "w0.grad", "w1.grad"]
        for name, a, b in zip(names, s, f):
            self.assertEqual(
                _md5(a),
                _md5(b),
                f"{name} differs between split and fused (use_acc={use_acc})",
            )

    def test_else_branch_2d_fp32(self):
        # use_accuracy_compatible=False -> matmul_grad else-branch.
        x_np, w0_np, w1_np = self._inputs([self.N, self.D], "float32")
        self._assert_bitwise(x_np, w0_np, w1_np, False)

    def test_megatron_branch_2d(self):
        # target "megatron" -> use_accuracy_compatible path, two separate GEMMs.
        x_np, w0_np, w1_np = self._inputs([self.N, self.D], "float32")
        self._assert_bitwise(x_np, w0_np, w1_np, "megatron")

    def test_hf_branch_2d(self):
        # target "hf" -> hf_bitexact path, grad rounded to activation dtype.
        x_np, w0_np, w1_np = self._inputs([self.N, self.D], "float32")
        self._assert_bitwise(x_np, w0_np, w1_np, "hf")

    def test_rank3_input_reshapes_back(self):
        # 3D [s, b, d]: the fused node must flatten internally and return a 3D
        # x grad, matching two independent gates on the same 3D input.
        s, b = 4, 8
        x_np, w0_np, w1_np = self._inputs([s, b, self.D], "float32")
        self._assert_bitwise(x_np, w0_np, w1_np, False)
        # shape sanity: x grad keeps the 3D input shape
        _, _, xg, _, _ = self._run_fused(x_np, w0_np, w1_np, False)
        self.assertEqual(list(xg.shape), [s, b, self.D])

    def test_bf16_input(self):
        # bf16 activation, fp32 promote inside, grad cast back to bf16.
        x_np, w0_np, w1_np = self._inputs([self.N, self.D], "float32")
        xb = paddle.to_tensor(x_np, dtype="bfloat16")
        xb.stop_gradient = False
        w0 = paddle.to_tensor(w0_np)
        w0.stop_gradient = False
        w1 = paddle.to_tensor(w1_np)
        w1.stop_gradient = False

        xb2 = paddle.to_tensor(x_np, dtype="bfloat16")
        xb2.stop_gradient = False
        w0b = paddle.to_tensor(w0_np)
        w0b.stop_gradient = False
        w1b = paddle.to_tensor(w1_np)
        w1b.stop_gradient = False

        l0, l1 = FusedTwoViewGate.apply(xb, w0, w1, False, False)
        (l0.sum() + l1.sum()).backward()

        s0 = FusedGateDetachMatmul.apply(xb2, w0b, False, False)
        s1 = FusedGateDetachMatmul.apply(xb2, w1b, False, False)
        (s0.sum() + s1.sum()).backward()

        self.assertEqual(_md5(l0), _md5(s0))
        self.assertEqual(_md5(l1), _md5(s1))
        self.assertEqual(_md5(xb.grad), _md5(xb2.grad))
        self.assertEqual(xb.grad.dtype, xb2.grad.dtype)


class TestFusedTwoViewGateDeferDW(unittest.TestCase):
    """defer_dw=True: x.grad is immediate (combined 2D sum -> reshape), both
    views' weight grads are deferred to WeightGradStore and applied on pop().

    The equivalence to assert is fused(defer) == two independent gates(defer):
    both use the same deferred-wgrad op path, so this pins that fusing the two
    views does not corrupt either view's weight grad or the combined x.grad, and
    that a rank-3 input keeps its shape. (defer vs non-defer is NOT bitwise even
    in the original single-gate op -- defer uses matmul, non-defer matmul_grad.)
    """

    def setUp(self):
        paddle.seed(2026)
        np.random.seed(2026)
        WeightGradStore.clear()
        self.N, self.D, self.E = 32, 16, 8

    def tearDown(self):
        WeightGradStore.clear()

    def test_defer_dw_matches_split_and_keeps_rank3_shape(self):
        s, b = 4, 8
        x_np = np.random.randn(s, b, self.D).astype("float32")
        w0_np = np.random.randn(self.E, self.D).astype("float32")
        w1_np = np.random.randn(self.E, self.D).astype("float32")

        # ---- reference: two independent gates, defer_dw=True ----
        WeightGradStore.clear()
        xr = paddle.to_tensor(x_np)
        xr.stop_gradient = False
        w0r = _weight_with_main_grad([self.E, self.D], w0_np)
        w1r = _weight_with_main_grad([self.E, self.D], w1_np)
        s0 = FusedGateDetachMatmul.apply(xr, w0r, True, False)
        s1 = FusedGateDetachMatmul.apply(xr, w1r, True, False)
        (s0.sum() + s1.sum()).backward()
        ref_xg = _md5(xr.grad)
        WeightGradStore.flush()
        WeightGradStore.pop()
        ref_w0g = _md5(w0r.main_grad)
        ref_w1g = _md5(w1r.main_grad)

        # ---- FusedTwoViewGate, defer_dw=True ----
        WeightGradStore.clear()
        xd = paddle.to_tensor(x_np)
        xd.stop_gradient = False
        w0d = _weight_with_main_grad([self.E, self.D], w0_np)
        w1d = _weight_with_main_grad([self.E, self.D], w1_np)
        l0d, l1d = FusedTwoViewGate.apply(xd, w0d, w1d, True, False)
        (l0d.sum() + l1d.sum()).backward()

        # x.grad is immediate and keeps the rank-3 input shape
        self.assertIsNotNone(xd.grad)
        self.assertEqual(list(xd.grad.shape), [s, b, self.D])
        self.assertEqual(_md5(xd.grad), ref_xg)

        # weight grads deferred: PyLayer returns None, main_grad not yet set,
        # both views queued in the store.
        self.assertIsNone(w0d.grad)
        self.assertIsNone(w1d.grad)
        self.assertIsNone(w0d.main_grad)
        self.assertIsNone(w1d.main_grad)
        self.assertEqual(len(WeightGradStore.cache), 2)

        # run the deferred wgrad computations
        WeightGradStore.flush()
        WeightGradStore.pop()

        self.assertIsNotNone(w0d.main_grad)
        self.assertIsNotNone(w1d.main_grad)
        self.assertEqual(list(w0d.main_grad.shape), [self.E, self.D])
        self.assertEqual(list(w1d.main_grad.shape), [self.E, self.D])
        # each view's _apply_backward_hook fired exactly once
        self.assertEqual(w0d._hook_call_count, 1)
        self.assertEqual(w1d._hook_call_count, 1)

        # per-view deferred main_grad matches the independent-gate reference,
        # bitwise -- proves fusion did not swap/mix the two views' wgrads.
        self.assertEqual(_md5(w0d.main_grad), ref_w0g)
        self.assertEqual(_md5(w1d.main_grad), ref_w1g)


if __name__ == "__main__":
    unittest.main()
