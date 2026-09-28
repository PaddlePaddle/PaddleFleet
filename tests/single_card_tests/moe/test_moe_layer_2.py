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

"""Behavior tests for paddlefleet.transformer.moe.moe_layer.

Target: the differentiable fan-out ``PyLayer`` helpers that reproduce the
gradient-accumulation topology of an MoE block whose input ``hidden_states``
is consumed by several branches (router, dispatcher/experts, shared expert,
shared-expert gate). Getting these backward passes wrong silently drops or
mis-combines part of the gradient that flows back into ``hidden_states``:

  * ``ThreePathCloneAlignMG`` clones the input into router / dispatcher /
    shared branches; its backward must return the *sum* of all three upstream
    gradients (if one path were dropped, ``hidden_states`` would train on a
    truncated gradient).
  * ``HFMoeFanout`` clones the input four ways (expert-core / router / up_proj
    / gate_proj) and accumulates the upstream gradients, skipping any ``None``
    path.
  * ``HFMoeSlotFanout`` permutes the input by ``sorted_indices`` in the forward
    and, in the backward, regroups the permuted per-expert gradient into
    ``[num_tokens, topk, hidden]`` slots, sums the slots, and adds the
    shared-gate gradient. A wrong gather, a dropped slot, or a missing
    shared-gate term changes the recovered per-token gradient.

Each test drives a real backward pass through ``PyLayer.apply`` and compares
against expected values derived by hand in numpy (never by re-running the
production combine), using distinguishable per-row / per-branch values so a
dropped path, a wrong gather, or a swapped slot is rejected. These are pure
autograd/tensor-layout contracts that run on CPU float32; no GPU numerics and
no collectives are exercised. Paddle is imported under a guard so environments
without paddle / paddlefleet skip honestly instead of reporting a false pass.
"""

import unittest

import numpy as np

try:
    import paddle

    from paddlefleet.transformer.moe.moe_layer import (
        HFMoeFanout,
        HFMoeSlotFanout,
        ThreePathCloneAlignMG,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet_ops not installed
    paddle = None
    HFMoeFanout = None
    HFMoeSlotFanout = None
    ThreePathCloneAlignMG = None
    _IMPORT_ERROR = exc


@unittest.skipUnless(
    ThreePathCloneAlignMG is not None,
    f"paddle/paddlefleet not importable: {_IMPORT_ERROR}",
)
class TestThreePathCloneAlignMG(unittest.TestCase):
    """Router / dispatcher / shared three-way clone and its summed backward."""

    def test_forward_returns_three_identity_clones(self):
        """All three outputs equal the input (differentiable identity)."""
        data = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], dtype=np.float32)
        x = paddle.to_tensor(data)
        o_router, o_dispatcher, o_shared = ThreePathCloneAlignMG.apply(x)
        for out in (o_router, o_dispatcher, o_shared):
            np.testing.assert_array_equal(
                np.asarray(out.numpy(), dtype=np.float32), data
            )

    def test_backward_sums_all_three_branch_gradients(self):
        """dx must equal g_router + g_dispatcher + g_shared, element-wise.

        Distinct per-branch, per-element coefficients make the three upstream
        gradients different, so dropping any single branch (or double-counting
        one) would change ``x.grad`` and fail. The expected gradient is the
        independent numpy sum of the three coefficient tensors.
        """
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        c_router = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        c_dispatcher = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        c_shared = np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32)

        x = paddle.to_tensor(data)
        x.stop_gradient = False
        o_router, o_dispatcher, o_shared = ThreePathCloneAlignMG.apply(x)
        loss = (
            (o_router * paddle.to_tensor(c_router)).sum()
            + (o_dispatcher * paddle.to_tensor(c_dispatcher)).sum()
            + (o_shared * paddle.to_tensor(c_shared)).sum()
        )
        loss.backward()

        self.assertIsNotNone(x.grad)
        expected = c_router + c_dispatcher + c_shared
        np.testing.assert_allclose(
            np.asarray(x.grad.numpy(), dtype=np.float32),
            expected,
            rtol=0,
            atol=0,
        )


@unittest.skipUnless(
    HFMoeFanout is not None,
    f"paddle/paddlefleet not importable: {_IMPORT_ERROR}",
)
class TestHFMoeFanout(unittest.TestCase):
    """Four-way clone whose backward accumulates the used branch gradients."""

    def test_forward_returns_four_identity_clones(self):
        data = np.array([[2.0, -3.0], [0.5, 7.0]], dtype=np.float32)
        x = paddle.to_tensor(data)
        outs = HFMoeFanout.apply(x)
        self.assertEqual(len(outs), 4)
        for out in outs:
            np.testing.assert_array_equal(
                np.asarray(out.numpy(), dtype=np.float32), data
            )

    def test_backward_sums_all_four_branch_gradients(self):
        """dx == g_core + g_router + g_up + g_gate with distinct coefficients."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        c_core = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        c_router = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        c_up = np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32)
        c_gate = np.array(
            [[1000.0, 2000.0], [3000.0, 4000.0]], dtype=np.float32
        )

        x = paddle.to_tensor(data)
        x.stop_gradient = False
        o_core, o_router, o_up, o_gate = HFMoeFanout.apply(x)
        loss = (
            (o_core * paddle.to_tensor(c_core)).sum()
            + (o_router * paddle.to_tensor(c_router)).sum()
            + (o_up * paddle.to_tensor(c_up)).sum()
            + (o_gate * paddle.to_tensor(c_gate)).sum()
        )
        loss.backward()

        expected = c_core + c_router + c_up + c_gate
        np.testing.assert_allclose(
            np.asarray(x.grad.numpy(), dtype=np.float32),
            expected,
            rtol=0,
            atol=0,
        )

    def test_backward_skips_unused_gate_branch(self):
        """Leaving the 4th (gate) output out of the loss must not corrupt dx.

        The backward's ``if grad is not None`` guard is exercised: the unused
        output contributes no gradient, so dx == g_core + g_router + g_up. This
        holds whether the framework hands the unused slot ``None`` or a zero
        tensor, so it is a robust check of the skip branch either way.
        """
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        c_core = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
        c_router = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        c_up = np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32)

        x = paddle.to_tensor(data)
        x.stop_gradient = False
        o_core, o_router, o_up, _o_gate = HFMoeFanout.apply(x)
        loss = (
            (o_core * paddle.to_tensor(c_core)).sum()
            + (o_router * paddle.to_tensor(c_router)).sum()
            + (o_up * paddle.to_tensor(c_up)).sum()
        )
        loss.backward()

        expected = c_core + c_router + c_up
        np.testing.assert_allclose(
            np.asarray(x.grad.numpy(), dtype=np.float32),
            expected,
            rtol=0,
            atol=0,
        )


@unittest.skipUnless(
    HFMoeSlotFanout is not None,
    f"paddle/paddlefleet not importable: {_IMPORT_ERROR}",
)
class TestHFMoeSlotFanout(unittest.TestCase):
    """Permute forward and slot-regrouping backward for the HF-aligned path."""

    def test_forward_permutes_rows_and_clones_input(self):
        """permuted[i] == x[sorted_indices[i]]; second output is x itself."""
        data = np.array([[10.0, 11.0], [20.0, 21.0]], dtype=np.float32)
        sorted_idx = np.array([1, 0, 1, 0], dtype=np.int64)
        x = paddle.to_tensor(data)
        gather_index_flat = paddle.to_tensor([0, 1, 2, 3], dtype="int64")
        valid_rows = paddle.to_tensor([1, 1], dtype="int64")

        permuted, clone_out = HFMoeSlotFanout.apply(
            x,
            paddle.to_tensor(sorted_idx),
            gather_index_flat,
            valid_rows,
            2,  # num_tokens
            2,  # topk
            2,  # hidden
            False,  # has_padding
        )
        np.testing.assert_array_equal(
            np.asarray(permuted.numpy(), dtype=np.float32),
            data[sorted_idx],
        )
        np.testing.assert_array_equal(
            np.asarray(clone_out.numpy(), dtype=np.float32), data
        )

    def test_backward_regroups_slots_and_adds_shared_gate(self):
        """dx regroups permuted grad into topk slots, sums them, adds gate grad.

        Setup (num_tokens=2, topk=2, hidden=2, no padding):
          gather_index_flat = [0, 2, 1, 3] maps flattened (token, slot) rows to
          permuted rows, so
            gathered[t0] = [gp[0], gp[2]], gathered[t1] = [gp[1], gp[3]].
          The backward sums the two slots per token and adds grad_shared_gate.

        With grad_permuted gp = [[1,1],[2,2],[3,3],[4,4]] and grad_shared_gate
        gsg = [[100,100],[200,200]] the hand-derived result is:
            t0: gp[2] + gp[0] + gsg[0] = [3,3]+[1,1]+[100,100] = [104,104]
            t1: gp[3] + gp[1] + gsg[1] = [4,4]+[2,2]+[200,200] = [206,206]
        A wrong gather, a dropped slot, or a missing gate term changes this.
        """
        data = np.array([[10.0, 11.0], [20.0, 21.0]], dtype=np.float32)
        sorted_idx = np.array([0, 1, 0, 1], dtype=np.int64)
        gather_index_flat = paddle.to_tensor([0, 2, 1, 3], dtype="int64")
        valid_rows = paddle.to_tensor([1, 1], dtype="int64")

        coef_permuted = np.array(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], dtype=np.float32
        )
        coef_gate = np.array([[100.0, 100.0], [200.0, 200.0]], dtype=np.float32)

        x = paddle.to_tensor(data)
        x.stop_gradient = False
        permuted, clone_out = HFMoeSlotFanout.apply(
            x,
            paddle.to_tensor(sorted_idx),
            gather_index_flat,
            valid_rows,
            2,  # num_tokens
            2,  # topk
            2,  # hidden
            False,  # has_padding
        )
        loss = (permuted * paddle.to_tensor(coef_permuted)).sum() + (
            clone_out * paddle.to_tensor(coef_gate)
        ).sum()
        loss.backward()

        expected = np.array([[104.0, 104.0], [206.0, 206.0]], dtype=np.float32)
        self.assertIsNotNone(x.grad)
        np.testing.assert_allclose(
            np.asarray(x.grad.numpy(), dtype=np.float32),
            expected,
            rtol=0,
            atol=0,
        )

    def test_backward_topk_zero_returns_only_shared_gate_gradient(self):
        """topk == 0 short-circuit: dx equals the shared-gate gradient alone."""
        data = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        # No routed rows: sorted_indices empty -> permuted has zero rows.
        sorted_idx = paddle.to_tensor(
            np.zeros([0], dtype=np.int64), dtype="int64"
        )
        gather_index_flat = paddle.to_tensor(
            np.zeros([0], dtype=np.int64), dtype="int64"
        )
        valid_rows = paddle.to_tensor([1, 1], dtype="int64")
        coef_gate = np.array([[9.0, 9.0], [9.0, 9.0]], dtype=np.float32)

        x = paddle.to_tensor(data)
        x.stop_gradient = False
        permuted, clone_out = HFMoeSlotFanout.apply(
            x,
            sorted_idx,
            gather_index_flat,
            valid_rows,
            2,  # num_tokens
            0,  # topk
            2,  # hidden
            False,  # has_padding
        )
        loss = permuted.sum() + (clone_out * paddle.to_tensor(coef_gate)).sum()
        loss.backward()

        np.testing.assert_allclose(
            np.asarray(x.grad.numpy(), dtype=np.float32),
            coef_gate,
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
