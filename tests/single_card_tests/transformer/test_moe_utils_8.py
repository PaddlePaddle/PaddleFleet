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

"""Behavior tests for CPU-runnable helpers in ``transformer/moe/moe_utils.py``.

These cover the MoE token permute/unpermute combine math, the auxiliary-loss
gradient-injection ``PyLayer``, the ``FakeClone`` graph-only identity, the
``manual_backward`` deferred-backward machinery, and the small tensor helpers.
Every expected value is derived by hand (or with independent numpy) from the
routing/combine definition rather than from the production implementation.

Only the world-size==1 local path of ``_AllToAll`` is exercised here; that is
the interface's single-process passthrough, NOT a claim that the real
cross-rank all-to-all (split sizes, peer, direction) has been verified -- that
needs a real process group (multi-card).

CPU-only. Paddle is not guaranteed to be importable in this environment, so the
heavy imports are guarded and the whole module skips with an honest reason when
they are missing; it never fakes a pass.
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    import paddle

    from paddlefleet.transformer.moe.moe_utils import (
        AddAuxiliaryLoss,
        FakeClone,
        _AllToAll,
        detach_and_requires_grad_,
        is_tensor,
        manual_backward,
        permute,
        sort_chunks_by_idxs,
        unpermute,
    )

    _IMPORT_ERROR: ImportError | None = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    paddle = None
    _IMPORT_ERROR = exc


class _MoEUtilsCPUTestBase(unittest.TestCase):
    """Skip honestly when paddle/paddlefleet is absent; pin execution to CPU."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(
                f"paddle/paddlefleet import failed: {_IMPORT_ERROR!r}"
            )
        paddle.set_device("cpu")


class TestAddAuxiliaryLoss(_MoEUtilsCPUTestBase):
    """``AddAuxiliaryLoss`` is an identity on ``x`` whose sole purpose is to
    inject a unit gradient into the aux-loss scalar, but only when that scalar
    actually requires a gradient."""

    def test_forward_is_value_identity_on_x(self):
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        loss = paddle.to_tensor([5.0])
        out = AddAuxiliaryLoss.apply(x, loss)
        np.testing.assert_array_equal(out.numpy(), x.numpy())

    def test_backward_passes_x_grad_through_and_injects_unit_aux_grad(self):
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        x.stop_gradient = False
        loss = paddle.to_tensor([7.0])
        loss.stop_gradient = False  # -> required_aux_loss is True

        out = AddAuxiliaryLoss.apply(x, loss)
        upstream = paddle.to_tensor([[0.5, 1.5], [2.5, 3.5]])
        out.backward(upstream)

        # dOut/dx is the identity, so x.grad is exactly the upstream grad.
        np.testing.assert_array_equal(x.grad.numpy(), upstream.numpy())
        # The aux-loss grad is a fixed ones(1), independent of the upstream.
        np.testing.assert_array_equal(
            loss.grad.numpy(), np.array([1.0], dtype=np.float32)
        )

    def test_backward_omits_aux_grad_when_loss_is_detached(self):
        x = paddle.to_tensor([[2.0, 4.0]])
        x.stop_gradient = False
        loss = paddle.to_tensor([9.0])
        loss.stop_gradient = True  # -> required_aux_loss is False

        out = AddAuxiliaryLoss.apply(x, loss)
        out.backward(paddle.to_tensor([[3.0, 5.0]]))

        np.testing.assert_array_equal(
            x.grad.numpy(), np.array([[3.0, 5.0]], dtype=np.float32)
        )
        self.assertIsNone(loss.grad)


class TestFakeClone(_MoEUtilsCPUTestBase):
    """``FakeClone`` extracts the compute graph without copying data; it is a
    value identity forward and a straight passthrough backward."""

    def test_forward_preserves_values(self):
        x = paddle.to_tensor([[1.5, -2.0], [3.0, 4.25]])
        out = FakeClone.apply(x)
        np.testing.assert_array_equal(out.numpy(), x.numpy())

    def test_backward_is_gradient_identity(self):
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        x.stop_gradient = False
        out = FakeClone.apply(x)
        upstream = paddle.to_tensor([[10.0, 20.0], [30.0, 40.0]])
        out.backward(upstream)
        np.testing.assert_array_equal(x.grad.numpy(), upstream.numpy())


class TestAllToAllSingleProcess(_MoEUtilsCPUTestBase):
    """world-size==1 local passthrough only (see module docstring)."""

    def test_forward_returns_input_unchanged_when_world_size_1(self):
        x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        # group=None + no distributed init => get_world_size() == 1.
        out = _AllToAll.apply([2, 3], x, None, None, None)
        np.testing.assert_array_equal(out.numpy(), x.numpy())


class TestManualBackward(_MoEUtilsCPUTestBase):
    """``manual_backward`` returns the forward result plus, when not the first
    forward, a closure that replays the local backward on demand."""

    def test_first_fwd_returns_no_closure_and_real_forward_value(self):
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        x.stop_gradient = False
        bwd_f, out = manual_backward(lambda t: t * 2, True, x)
        self.assertIsNone(bwd_f)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 1)
        np.testing.assert_array_equal(out[0].numpy(), x.numpy() * 2)

    def test_deferred_backward_replays_hand_derivative(self):
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        x.stop_gradient = False
        bwd_f, out = manual_backward(lambda t: t * 3, False, x)
        self.assertIsNotNone(bwd_f)
        np.testing.assert_array_equal(out[0].numpy(), x.numpy() * 3)

        upstream = paddle.to_tensor([[1.0, 1.0], [1.0, 1.0]])
        grads = bwd_f(upstream)
        # d(3t)/dt == 3, so the replayed grad is 3 * upstream.
        self.assertEqual(len(grads), 1)
        np.testing.assert_array_equal(
            grads[0].numpy(), np.full((2, 2), 3.0, dtype=np.float32)
        )


class TestTensorHelpers(_MoEUtilsCPUTestBase):
    def test_is_tensor_discriminates_tensors_from_python_objects(self):
        self.assertTrue(is_tensor(paddle.to_tensor([1.0])))
        self.assertFalse(is_tensor(5))
        self.assertFalse(is_tensor([1, 2, 3]))
        self.assertFalse(is_tensor(None))

    def test_detach_preserves_values_and_per_arg_stop_gradient(self):
        trainable = paddle.to_tensor([1.0, 2.0])
        trainable.stop_gradient = False
        frozen = paddle.to_tensor([3.0, 4.0])
        frozen.stop_gradient = True

        out_trainable, out_frozen, passthrough = detach_and_requires_grad_(
            trainable, frozen, 42
        )

        np.testing.assert_array_equal(out_trainable.numpy(), [1.0, 2.0])
        np.testing.assert_array_equal(out_frozen.numpy(), [3.0, 4.0])
        self.assertFalse(out_trainable.stop_gradient)
        self.assertTrue(out_frozen.stop_gradient)
        self.assertEqual(passthrough, 42)  # non-tensors pass through untouched
        self.assertFalse(out_trainable is trainable)  # detached copy, not alias


class TestSortChunksByIdxs(_MoEUtilsCPUTestBase):
    def test_reorders_chunks_by_index_and_preserves_content(self):
        rows = paddle.arange(6 * 2, dtype="float32").reshape([6, 2])
        split_sizes = paddle.to_tensor([2, 1, 3])
        sorted_idxs = paddle.to_tensor([2, 0, 1])

        out, permuted_probs = sort_chunks_by_idxs(
            rows, split_sizes, sorted_idxs
        )

        # chunks: c0=rows[0:2], c1=rows[2:3], c2=rows[3:6]; ordered [c2, c0, c1].
        expected = rows.numpy()[[3, 4, 5, 0, 1, 2]]
        np.testing.assert_array_equal(out.numpy(), expected)
        self.assertIsNone(permuted_probs)


class TestPermute(_MoEUtilsCPUTestBase):
    def test_groups_tokens_by_expert_preserving_identity(self):
        # token0 -> experts {0, 1}; token1 -> {0}; token2 -> {1}.
        tokens = paddle.to_tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
        routing_map = paddle.to_tensor([[1, 1], [1, 0], [0, 1]])

        permuted, sorted_indices = permute(tokens, routing_map)

        # Expert-major grouping: expert0 gets tokens 0,1; expert1 gets 0,2.
        self.assertEqual(sorted_indices.tolist(), [0, 1, 0, 2])
        np.testing.assert_array_equal(
            permuted.numpy(), tokens.numpy()[[0, 1, 0, 2]]
        )


class TestUnpermute(_MoEUtilsCPUTestBase):
    def test_scatter_combine_sums_contributions_per_token(self):
        # Rows aligned with sorted_indices=[0,1,0,2] from the permute grouping.
        permuted = paddle.to_tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        )
        sorted_indices = paddle.to_tensor([0, 1, 0, 2])

        out = unpermute(permuted, sorted_indices, [3, 2])

        # token0 = row0 + row2, token1 = row1, token2 = row3.
        expected = np.array(
            [[6.0, 8.0], [3.0, 4.0], [7.0, 8.0]], dtype=np.float32
        )
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_applies_permuted_probs_before_combine(self):
        permuted = paddle.to_tensor(
            [[2.0, 2.0], [4.0, 4.0], [6.0, 6.0], [8.0, 8.0]]
        )
        sorted_indices = paddle.to_tensor([0, 1, 0, 2])
        routing_map = paddle.to_tensor([[1, 1], [1, 0], [0, 1]])
        # probs[token, expert]; only routed entries are gathered, expert-major:
        #   expert0 -> probs[0,0]=0.5, probs[1,0]=0.5
        #   expert1 -> probs[0,1]=0.25, probs[2,1]=2.0
        probs = paddle.to_tensor([[0.5, 0.25], [0.5, 0.0], [0.0, 2.0]])

        out = unpermute(
            permuted,
            sorted_indices,
            [3, 2],
            probs=probs,
            routing_map=routing_map,
        )

        # Scaled rows: [1,1], [2,2], [1.5,1.5], [16,16]; then combine per token.
        # token0 = [1,1]+[1.5,1.5]=[2.5,2.5]; token1=[2,2]; token2=[16,16].
        expected = np.array(
            [[2.5, 2.5], [2.0, 2.0], [16.0, 16.0]], dtype=np.float32
        )
        np.testing.assert_array_equal(out.numpy(), expected)


if __name__ == "__main__":
    unittest.main()
