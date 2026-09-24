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
"""Behavior tests for ``paddlefleet.transformer.moe.moe_utils``.

Every expectation here is hand-derived from the documented contract of the
function under test, independent of its implementation:

* ``AutoSBHistoryTracker`` -- a pure-Python warm-up memory-delta state machine;
  its iteration boundary detection and ``predicted_need_for_remaining`` output
  are traced by hand.
* ``permute`` / ``unpermute`` -- token grouping by expert is verified by the
  exact permuted row order and by a probability-weighted round trip that
  reconstructs the original tokens (weights per token sum to one).
* ``sort_chunks_by_idxs`` -- exact reordered chunk contents.
* ``AddAuxiliaryLoss`` / ``RandomSTE`` -- forward value plus the concrete
  backward gradient contract (identity pass-through / zero straight-through).
* ``is_tensor`` / ``detach_and_requires_grad_`` -- truth values and
  detach/pass-through semantics.
* ``all_gather_group`` / ``reduce_scatter_group`` -- only the ``nranks == 1``
  local fallback (a genuine world-size-1 path) and the input-validation guards
  that reject a call *before* any collective runs. These assert nothing about
  real cross-rank communication; that requires a real multi-card process group.

Paddle is required to import the module under test. When it is unavailable the
whole suite is skipped with an honest reason rather than reported as passing.
"""

import os
import sys
import types
import unittest

import numpy as np

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

try:
    import paddle

    from paddlefleet.transformer.moe import moe_utils

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - environment-dependent
    paddle = None
    moe_utils = None
    _IMPORT_ERROR = exc

_SKIP_REASON = f"paddle / paddlefleet.transformer.moe.moe_utils unavailable: {_IMPORT_ERROR}"


@unittest.skipUnless(moe_utils is not None, _SKIP_REASON)
class TestAutoSBHistoryTracker(unittest.TestCase):
    """Pure-Python warm-up memory-delta bookkeeping; all values hand-traced."""

    def test_full_iteration_delta_and_prediction(self):
        t = moe_utils.AutoSBHistoryTracker()
        self.assertTrue(t.in_warmup())

        # Warm-up forwards. max_delta tracks the largest drop between the free
        # memory reported at consecutive forward starts (clamped at >= 0).
        t.record_forward(1000)  # first: no previous sample, delta undefined
        self.assertEqual(t.max_delta, 0)
        t.record_forward(700)  # drop 300
        self.assertEqual(t.max_delta, 300)
        t.record_forward(500)  # drop 200 < 300, max stays 300
        self.assertEqual(t.max_delta, 300)
        self.assertEqual(t.step_idx, 3)
        self.assertEqual(t.forward_count, 3)

        # No completed iteration yet -> cannot predict (cold start).
        self.assertEqual(t.predicted_need_for_remaining(), 0)
        self.assertFalse(t.should_degrade(0))

        # Backward count must reach forward_count before the iteration closes.
        self.assertFalse(t.record_backward())  # 1 of 3
        self.assertFalse(t.record_backward())  # 2 of 3
        self.assertFalse(t.in_warmup())  # backward_count != 0 mid-iteration
        self.assertTrue(t.record_backward())  # 3 of 3 -> iteration boundary

        # Iteration rolled over: remembered 3 steps and a peak delta of 300,
        # and all live counters reset (back in warm-up).
        self.assertEqual(t.prev_total_steps, 3)
        self.assertEqual(t.prev_max_delta, 300)
        self.assertTrue(t.in_warmup())
        self.assertEqual(t.step_idx, 0)
        self.assertEqual(t.forward_count, 0)

        # predicted_need = int(prev_max_delta * remaining * 1.2) + 128 MiB,
        # remaining = prev_total_steps - step_idx + 1 = 3 - 0 + 1 = 4.
        margin = 128 * 1024 * 1024
        self.assertEqual(
            t.predicted_need_for_remaining(), int(300 * 4 * 1.2) + margin
        )
        self.assertEqual(t.predicted_need_for_remaining(), 1440 + margin)

        # should_degrade only fires in warm-up when free < predicted need.
        need = t.predicted_need_for_remaining()
        self.assertTrue(t.should_degrade(need - 1))
        self.assertFalse(t.should_degrade(need))
        self.assertFalse(t.should_degrade(need + 1))

        # A forward inside the new iteration advances step_idx, shrinking the
        # remaining estimate: remaining = 3 - 1 + 1 = 3.
        t.record_forward(900)
        self.assertEqual(t.step_idx, 1)
        self.assertEqual(
            t.predicted_need_for_remaining(), int(300 * 3 * 1.2) + margin
        )

    def test_stray_backward_without_forward_does_not_close_iteration(self):
        t = moe_utils.AutoSBHistoryTracker()
        # forward_count == 0, so the boundary condition is never met.
        self.assertFalse(t.record_backward())
        self.assertEqual(t.prev_total_steps, 0)
        self.assertEqual(t.prev_max_delta, 0)
        self.assertEqual(t.predicted_need_for_remaining(), 0)

    def test_get_auto_sb_history_returns_module_singleton(self):
        first = moe_utils.get_auto_sb_history()
        second = moe_utils.get_auto_sb_history()
        self.assertIs(first, second)
        self.assertIsInstance(first, moe_utils.AutoSBHistoryTracker)


@unittest.skipUnless(moe_utils is not None, _SKIP_REASON)
class TestIsTensorAndDetach(unittest.TestCase):
    """Type predicate and detach/pass-through semantics."""

    def test_is_tensor_truth_values(self):
        self.assertTrue(moe_utils.is_tensor(paddle.to_tensor([1.0, 2.0])))
        # Non-tensor inputs are all rejected.
        for value in (42, 3.14, [1, 2, 3], (1, 2), {"a": 1}, "tensor", None):
            self.assertFalse(
                moe_utils.is_tensor(value), msg=f"{value!r} is not a tensor"
            )

    def test_detach_preserves_value_and_stop_gradient_flag(self):
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        x.stop_gradient = False
        y = paddle.to_tensor([5.0, 6.0])
        y.stop_gradient = True

        dx, dy = moe_utils.detach_and_requires_grad_(x, y)

        # Detach yields a distinct tensor with identical content.
        self.assertIsNot(dx, x)
        np.testing.assert_array_equal(dx.numpy(), [[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(dy.numpy(), [5.0, 6.0])
        # The original stop_gradient flag is carried onto the detached copy.
        self.assertFalse(dx.stop_gradient)
        self.assertTrue(dy.stop_gradient)

    def test_non_tensor_args_pass_through_unchanged(self):
        sentinel = object()
        out = moe_utils.detach_and_requires_grad_(7, "keep", sentinel)
        self.assertEqual(out[0], 7)
        self.assertEqual(out[1], "keep")
        self.assertIs(out[2], sentinel)  # same object, not copied


@unittest.skipUnless(moe_utils is not None, _SKIP_REASON)
class TestPermuteUnpermute(unittest.TestCase):
    """Token grouping by expert and its probability-weighted inverse."""

    def _routing_map(self):
        # token 0 -> experts {0, 2}; token 1 -> {1}; token 2 -> {0}; token 3 -> {2}
        return paddle.to_tensor(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    def _tokens(self):
        return paddle.to_tensor(
            [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]]
        )

    def test_permute_groups_tokens_by_expert_in_order(self):
        permuted, sorted_indices = moe_utils.permute(
            self._tokens(), self._routing_map()
        )
        # masked_select scans experts (rows) then tokens (cols): expert 0 sees
        # tokens 0,2; expert 1 sees token 1; expert 2 sees tokens 0,3.
        self.assertEqual(sorted_indices.tolist(), [0, 2, 1, 0, 3])
        np.testing.assert_array_equal(
            permuted.numpy(),
            [
                [10.0, 11.0],  # token 0 @ expert 0
                [30.0, 31.0],  # token 2 @ expert 0
                [20.0, 21.0],  # token 1 @ expert 1
                [10.0, 11.0],  # token 0 @ expert 2
                [40.0, 41.0],  # token 3 @ expert 2
            ],
        )

    def test_unpermute_scatter_accumulates_and_zero_fills(self):
        # Directly exercise the scatter-add restore: a repeated destination
        # index accumulates, an index never written stays zero.
        permuted = paddle.to_tensor(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
        )
        sorted_indices = paddle.to_tensor([2, 0, 0, 3], dtype="int64")
        restored = moe_utils.unpermute(
            permuted,
            sorted_indices,
            restore_shape=[4, 2],
            probs=None,
            routing_map=None,
        )
        np.testing.assert_array_equal(
            restored.numpy(),
            [
                [5.0, 5.0],  # index 0 written twice: [2,2] + [3,3]
                [0.0, 0.0],  # index 1 never written
                [1.0, 1.0],  # index 2
                [4.0, 4.0],  # index 3
            ],
        )

    def test_probability_weighted_round_trip_reconstructs_tokens(self):
        tokens = self._tokens()
        routing_map = self._routing_map()
        # Per-token weights sum to 1, so combining the routed copies must
        # recover the original tokens exactly.
        probs = paddle.to_tensor(
            [
                [0.5, 0.0, 0.5],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        permuted, sorted_indices = moe_utils.permute(tokens, routing_map)
        restored = moe_utils.unpermute(
            permuted,
            sorted_indices,
            restore_shape=tokens.shape,
            probs=probs,
            routing_map=routing_map,
        )
        np.testing.assert_allclose(
            restored.numpy(), tokens.numpy(), rtol=1e-6, atol=1e-6
        )

    def test_permute_rejects_drop_and_pad(self):
        with self.assertRaises(AssertionError):
            moe_utils.permute(
                self._tokens(), self._routing_map(), drop_and_pad=True
            )


@unittest.skipUnless(moe_utils is not None, _SKIP_REASON)
class TestSortChunksByIdxs(unittest.TestCase):
    """Split along axis 0 and reassemble chunks in the requested order."""

    def test_reorders_chunks_by_content(self):
        x = paddle.to_tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        split_sizes = paddle.to_tensor([2, 1, 1], dtype="int64")
        sorted_idxs = paddle.to_tensor([2, 0, 1], dtype="int64")
        out, permuted_probs = moe_utils.sort_chunks_by_idxs(
            x, split_sizes, sorted_idxs
        )
        # chunks: c0=rows[0:2], c1=rows[2:3], c2=rows[3:4];
        # order [2,0,1] -> c2, c0, c1.
        np.testing.assert_array_equal(
            out.numpy(),
            [[3.0, 3.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
        )
        # probs handling is not implemented for this helper yet.
        self.assertIsNone(permuted_probs)


@unittest.skipUnless(moe_utils is not None, _SKIP_REASON)
class TestPyLayerGradients(unittest.TestCase):
    """Forward value plus the concrete backward gradient contract."""

    def test_add_auxiliary_loss_forward_and_gradients(self):
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        x.stop_gradient = False
        loss = paddle.to_tensor([0.5])
        loss.stop_gradient = False

        out = moe_utils.AddAuxiliaryLoss.apply(x, loss)
        # Forward returns a clone of x (same content, independent tensor).
        self.assertIsNot(out, x)
        np.testing.assert_array_equal(out.numpy(), x.numpy())

        upstream = paddle.to_tensor([[10.0, 20.0], [30.0, 40.0]])
        out.backward(upstream)
        # dx is the identity pass-through of the upstream gradient...
        np.testing.assert_array_equal(x.grad.numpy(), upstream.numpy())
        # ...and the aux loss receives a gradient of ones (weight 1.0).
        np.testing.assert_array_equal(loss.grad.numpy(), [1.0])

    def test_add_auxiliary_loss_rejects_non_scalar_loss(self):
        x = paddle.to_tensor([[1.0, 2.0]])
        bad_loss = paddle.to_tensor([0.1, 0.2])  # numel != 1
        with self.assertRaises(AssertionError):
            moe_utils.AddAuxiliaryLoss.apply(x, bad_loss)

    def test_random_ste_has_zero_gradient(self):
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        x.stop_gradient = False
        out = moe_utils.apply_random_logits(x)
        # Forward keeps shape/dtype (values are random, not asserted here).
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, x.dtype)

        out.backward(paddle.to_tensor([[7.0, 8.0], [9.0, 10.0]]))
        # Straight-through estimator: the gradient is exactly zero regardless
        # of the (non-zero) upstream gradient.
        np.testing.assert_array_equal(x.grad.numpy(), np.zeros_like(x.numpy()))


@unittest.skipUnless(moe_utils is not None, _SKIP_REASON)
class TestGroupCollectiveLocalContracts(unittest.TestCase):
    """world-size==1 fallback and pre-collective input-validation guards only.

    None of these assert real cross-rank communication. The single-rank branch
    returns a local clone before touching any collective, and the guard tests
    trigger the assertion that rejects a bad call *before* a collective runs.
    Genuine gather/reduce-scatter numerics require a real multi-card process
    group (see the distributed-training multi-card requirements).
    """

    def test_all_gather_group_single_rank_returns_clone(self):
        group = types.SimpleNamespace(nranks=1)
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        out = moe_utils.all_gather_group(x, group=group)
        self.assertIsNot(out, x)  # clone, not the same tensor
        np.testing.assert_array_equal(out.numpy(), x.numpy())

    def test_reduce_scatter_group_single_rank_returns_clone(self):
        group = types.SimpleNamespace(nranks=1)
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        out = moe_utils.reduce_scatter_group(x, group=group)
        self.assertIsNot(out, x)
        np.testing.assert_array_equal(out.numpy(), x.numpy())

    def test_all_gather_group_rejects_non_zero_axis(self):
        group = types.SimpleNamespace(nranks=2)
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        with self.assertRaises(AssertionError):
            moe_utils.all_gather_group(x, group=group, axis=1)

    def test_reduce_scatter_group_requires_divisible_leading_dim(self):
        group = types.SimpleNamespace(nranks=3)
        x = paddle.to_tensor(np.ones((5, 8), dtype="float32"))  # 5 % 3 != 0
        with self.assertRaises(AssertionError):
            moe_utils.reduce_scatter_group(x, group=group)


if __name__ == "__main__":
    unittest.main()
