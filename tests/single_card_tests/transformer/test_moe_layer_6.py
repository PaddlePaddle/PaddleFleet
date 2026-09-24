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
"""Behavior tests for the CPU-executable surface of the MoE layer utilities.

The MoE layer (``paddlefleet.transformer.moe.moe_layer``) delegates its
gradient-injection trick, token permute/combine math and warmup memory
accounting to ``paddlefleet.transformer.moe.moe_utils``. This module verifies
those real behaviors:

  * ``AddAuxiliaryLoss`` PyLayer: forward returns a value-clone of the
    activations while backward injects a *unit* gradient into the auxiliary
    loss leaf (and nothing when the loss is detached). This is the mechanism
    ``moe_layer`` uses at moe_layer.py:1766/2029 to fold aux/z losses into the
    graph, so the injected gradient must be exactly 1 regardless of the
    activation magnitude.
  * ``permute`` / ``unpermute``: tokens are grouped by their selected experts
    and the combine step scatter-adds each token's expert copies back, so a
    token routed to k experts is summed k times.
  * ``sort_chunks_by_idxs``: split-then-reorder of variable-size chunks.
  * ``AutoSBHistoryTracker``: warmup-only free-memory delta accounting and the
    degrade prediction (1.2x safety factor + 128MB margin).

Every numeric expectation is derived independently by hand / numpy, never by
calling the production helper under test. Inputs are distinguishable so that a
transpose, wrong-index or dropped-contribution bug cannot survive.

Heavy imports (paddle + the moe modules) are guarded; a missing runtime is
reported honestly as a skip and only ImportError/ModuleNotFoundError counts as
"dependency absent" so a real API break still surfaces.
"""

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

try:
    import numpy as np
    import paddle

    from paddlefleet.transformer.moe import moe_layer, moe_utils
    from paddlefleet.transformer.moe.moe_utils import (
        AddAuxiliaryLoss,
        AutoSBHistoryTracker,
        permute,
        sort_chunks_by_idxs,
        unpermute,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    "paddle/paddlefleet/numpy not importable in this environment: "
    f"{_IMPORT_ERROR}"
    if _IMPORT_ERROR is not None
    else ""
)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestAddAuxiliaryLoss(unittest.TestCase):
    """AddAuxiliaryLoss forward clone + unit-gradient injection contract."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_forward_returns_value_clone_not_alias(self):
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        loss = paddle.to_tensor([0.7])
        loss.stop_gradient = False
        out = AddAuxiliaryLoss.apply(x, loss)
        # Forward is ``x.clone()``: same values, distinct storage.
        self.assertIsNot(out, x)
        np.testing.assert_array_equal(out.numpy(), x.numpy())
        self.assertNotEqual(out.data_ptr(), x.data_ptr())

    def test_backward_injects_unit_loss_gradient(self):
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        x.stop_gradient = False
        loss = paddle.to_tensor([0.7])
        loss.stop_gradient = False
        upstream = paddle.to_tensor([[2.0, 3.0], [4.0, 5.0]])

        out = AddAuxiliaryLoss.apply(x, loss)
        out.backward(upstream)

        # Activation gradient passes straight through unchanged ...
        self.assertIsNotNone(x.grad)
        np.testing.assert_array_equal(x.grad.numpy(), upstream.numpy())
        # ... while the aux-loss leaf receives an injected unit gradient,
        # independent of the upstream activation gradient (moe_utils.py:565).
        self.assertIsNotNone(loss.grad)
        np.testing.assert_array_equal(loss.grad.numpy(), np.array([1.0]))

    def test_injected_gradient_is_unit_regardless_of_activation_scale(self):
        # The trick guarantees d(loss)=1 no matter how large the activations
        # or their upstream gradients are; a magnitude-dependent injection
        # would break MoE aux-loss weighting. Two very different scales, same
        # unit loss gradient.
        for scale in (1.0, 1e4):
            x = paddle.to_tensor([[1.0, -2.0], [3.0, -4.0]]) * scale
            x.stop_gradient = False
            loss = paddle.to_tensor([123.0 * scale])
            loss.stop_gradient = False
            upstream = paddle.to_tensor([[7.0, 8.0], [9.0, 10.0]]) * scale

            out = AddAuxiliaryLoss.apply(x, loss)
            out.backward(upstream)

            np.testing.assert_array_equal(x.grad.numpy(), upstream.numpy())
            np.testing.assert_array_equal(loss.grad.numpy(), np.array([1.0]))

    def test_backward_no_grad_when_loss_detached(self):
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        x.stop_gradient = False
        loss = paddle.to_tensor([0.5])
        loss.stop_gradient = True  # required_aux_loss -> False
        upstream = paddle.to_tensor([[2.0, 3.0], [4.0, 5.0]])

        out = AddAuxiliaryLoss.apply(x, loss)
        out.backward(upstream)

        np.testing.assert_array_equal(x.grad.numpy(), upstream.numpy())
        # Detached loss must not receive an injected gradient
        # (moe_utils.py:564: required_aux_loss guards the ones(1) branch).
        self.assertIsNone(loss.grad)

    def test_forward_rejects_non_scalar_loss(self):
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        bad_loss = paddle.to_tensor([0.1, 0.2])  # numel == 2
        # Contract: forward asserts numel(loss) == 1 (moe_utils.py:556).
        with self.assertRaises(AssertionError):
            AddAuxiliaryLoss.apply(x, bad_loss)

    def test_moe_layer_reexports_same_pylayer(self):
        # moe_layer consumes exactly this PyLayer (moe_layer.py:1766); the
        # re-export must be the identical object, not a shadow copy.
        self.assertIs(moe_layer.AddAuxiliaryLoss, moe_utils.AddAuxiliaryLoss)
        self.assertIs(moe_layer.AddAuxiliaryLoss, AddAuxiliaryLoss)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestPermuteUnpermute(unittest.TestCase):
    """Token permute (group-by-expert) and unpermute (scatter-add combine)."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_permute_groups_tokens_by_expert(self):
        # routing_map[t, e] == 1 iff token t selects expert e.
        #   token0 -> {e0}, token1 -> {e1}, token2 -> {e0, e1}
        # Grouping by expert (e0 tokens first, then e1), each in token order,
        # gives the flattened source-row order [0, 2, 1, 2] (derived by hand).
        tokens = paddle.to_tensor([[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]])
        routing_map = paddle.to_tensor([[1, 0], [0, 1], [1, 1]], dtype="int64")
        permuted, sorted_indices = permute(tokens, routing_map)

        self.assertEqual(sorted_indices.tolist(), [0, 2, 1, 2])
        expected = np.array(
            [[0.0, 1.0], [20.0, 21.0], [10.0, 11.0], [20.0, 21.0]]
        )
        np.testing.assert_array_equal(permuted.numpy(), expected)

    def test_unpermute_scatter_adds_contributions(self):
        # Independent of permute: feed explicit permuted rows and the
        # scatter index [0, 2, 1, 2]. Rows mapping to the same token are
        # summed (token 2 appears twice -> its two rows add).
        permuted = paddle.to_tensor(
            [[0.0, 1.0], [20.0, 21.0], [10.0, 11.0], [20.0, 21.0]]
        )
        sorted_indices = paddle.to_tensor([0, 2, 1, 2], dtype="int64")
        out = unpermute(permuted, sorted_indices, [3, 2])

        # token0 <- row0; token1 <- row2; token2 <- row1 + row3.
        expected = np.array([[0.0, 1.0], [10.0, 11.0], [40.0, 42.0]])
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_roundtrip_scales_token_by_expert_count(self):
        # A token routed to k experts is combined from k copies, so the
        # round trip multiplies each token row by its expert count.
        tokens = paddle.to_tensor([[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]])
        routing_map = paddle.to_tensor([[1, 0], [0, 1], [1, 1]], dtype="int64")
        counts = np.array([1, 1, 2]).reshape(3, 1)  # experts per token
        expected = tokens.numpy() * counts

        permuted, sorted_indices = permute(tokens, routing_map)
        out = unpermute(permuted, sorted_indices, [3, 2])
        np.testing.assert_array_equal(out.numpy(), expected)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestSortChunksByIdxs(unittest.TestCase):
    """Variable-size chunk split then reorder by index."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_reorders_variable_size_chunks(self):
        rows = paddle.to_tensor(
            [
                [0.0, 1.0],
                [2.0, 3.0],
                [4.0, 5.0],
                [6.0, 7.0],
                [8.0, 9.0],
                [10.0, 11.0],
            ]
        )
        split_sizes = paddle.to_tensor([2, 1, 3], dtype="int64")
        # chunk0 = rows[0:2], chunk1 = rows[2:3], chunk2 = rows[3:6].
        # order [2, 0, 1] -> chunk2 ++ chunk0 ++ chunk1.
        sorted_idxs = paddle.to_tensor([2, 0, 1], dtype="int64")
        out, probs = sort_chunks_by_idxs(rows, split_sizes, sorted_idxs)

        self.assertIsNone(probs)
        expected = np.array(
            [
                [6.0, 7.0],
                [8.0, 9.0],
                [10.0, 11.0],
                [0.0, 1.0],
                [2.0, 3.0],
                [4.0, 5.0],
            ]
        )
        np.testing.assert_array_equal(out.numpy(), expected)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestAutoSBHistoryTracker(unittest.TestCase):
    """Warmup free-memory delta accounting and degrade prediction.

    Pure-Python logic; all expectations hand-derived from the recurrence in
    moe_utils.py (AutoSBHistoryTracker).
    """

    def test_cold_start_cannot_predict_or_degrade(self):
        t = AutoSBHistoryTracker()
        self.assertTrue(t.in_warmup())
        # No prior iteration recorded -> no prediction, never degrade.
        self.assertEqual(t.predicted_need_for_remaining(), 0)
        self.assertFalse(t.should_degrade(0))
        self.assertFalse(t.should_degrade(10**12))

    def test_forward_delta_ignored_outside_warmup(self):
        t = AutoSBHistoryTracker()
        t.record_forward(100)  # warmup: step 1
        t.record_forward(60)  # warmup: step 2, delta 40
        self.assertEqual(t.step_idx, 2)
        # One backward: forward_count(2) != backward_count(1) -> not a new
        # iteration, but warmup is now over.
        self.assertFalse(t.record_backward())
        self.assertFalse(t.in_warmup())
        # A forward after warmup must not advance step_idx / max_delta, but
        # still counts toward forward_count.
        t.record_forward(10)
        self.assertEqual(t.step_idx, 2)
        self.assertEqual(t.forward_count, 3)
        self.assertEqual(t.max_delta, 40)

    def test_iteration_records_max_delta_and_predicts(self):
        t = AutoSBHistoryTracker()
        # Warmup: free memory drops 10G -> 7G -> 5G, so deltas 3G then 2G;
        # max_delta = 3G, 3 forward steps.
        t.record_forward(10_000_000_000)
        t.record_forward(7_000_000_000)
        t.record_forward(5_000_000_000)
        self.assertEqual(t.step_idx, 3)
        self.assertEqual(t.max_delta, 3_000_000_000)
        # Still cold for prediction until an iteration completes.
        self.assertEqual(t.predicted_need_for_remaining(), 0)

        # Three backwards complete the iteration (3 == 3 -> True on the last).
        self.assertFalse(t.record_backward())
        self.assertFalse(t.record_backward())
        self.assertTrue(t.record_backward())

        # New iteration: prev_total_steps=3, prev_max_delta=3G, counters reset.
        self.assertTrue(t.in_warmup())
        self.assertEqual(t.step_idx, 0)
        self.assertEqual(t.forward_count, 0)

        # predicted = int(prev_max_delta * remaining * 1.2) + 128MB, with
        # remaining = prev_total_steps - step_idx + 1 = 3 - 0 + 1 = 4.
        margin = 128 * 1024 * 1024
        expected0 = int(3_000_000_000 * 4 * 1.2) + margin
        self.assertEqual(t.predicted_need_for_remaining(), expected0)
        self.assertEqual(expected0, 14_534_217_728)

        # After one forward in the new iteration remaining shrinks to 3.
        t.record_forward(9_000_000_000)
        expected1 = int(3_000_000_000 * 3 * 1.2) + margin
        self.assertEqual(t.predicted_need_for_remaining(), expected1)
        self.assertEqual(expected1, 10_934_217_728)

    def test_should_degrade_compares_free_against_prediction(self):
        t = AutoSBHistoryTracker()
        for free in (10_000_000_000, 7_000_000_000, 5_000_000_000):
            t.record_forward(free)
        t.record_backward()
        t.record_backward()
        t.record_backward()

        need = t.predicted_need_for_remaining()  # 14_534_217_728 at step 0
        self.assertEqual(need, 14_534_217_728)
        # Strictly less than the predicted need -> degrade.
        self.assertTrue(t.should_degrade(need - 1))
        self.assertTrue(t.should_degrade(1_000_000_000))
        # Boundary is strict, and ample memory does not degrade.
        self.assertFalse(t.should_degrade(need))
        self.assertFalse(t.should_degrade(20_000_000_000))


if __name__ == "__main__":
    unittest.main()
