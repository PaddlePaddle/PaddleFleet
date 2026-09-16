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
"""Behavior tests for paddlefleet.transformer.moe.moe_utils.

Scope is the CPU-executable, single-rank surface of the MoE utilities:

  * ``permute`` / ``unpermute`` token routing math (default, non
    accuracy-compatible path): tokens are grouped by expert, and unpermute
    scatter-accumulates each token's per-expert contributions back, optionally
    weighting them by the router probs.
  * ``sort_chunks_by_idxs`` chunk split + reorder.
  * ``AddAuxiliaryLoss`` / ``FakeClone`` PyLayer forward value + backward
    gradient contracts.
  * ``is_tensor`` / ``detach_and_requires_grad_`` type + stop_gradient rules.
  * ``_all_gather_local_tokens`` world-size-1 local reshape (this only covers
    the single-rank local path; the real all-gather across EP ranks needs a
    genuine process group and is out of scope here).
  * The MoE balance logging orchestration: ``log_moe_losses`` /
    ``_log_summary`` / ``_log_tokens_per_expert`` / ``log_moe_balance`` -- what
    keys and values actually reach the training-logs sink.

Every numeric expectation is derived from an INDEPENDENT hand computation
(expert-major masked select order, explicit scatter-accumulate, per-(token,
expert) prob application), never by calling the production helper under test.
Distinguishable, non-uniform values are used so a routing-order, weight-order
or accumulation bug cannot survive.

Heavy imports (paddle + the moe module) are guarded so a missing runtime is
reported honestly as a skip; only ImportError/ModuleNotFoundError counts as
"dependency absent" so that real API breaks still surface.
"""

import os
import sys
import unittest
from unittest import mock

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

    from paddlefleet.transformer.moe import moe_utils
    from paddlefleet.transformer.moe.moe_utils import (
        AddAuxiliaryLoss,
        FakeClone,
        _all_gather_local_tokens,
        detach_and_requires_grad_,
        is_tensor,
        log_moe_balance,
        log_moe_losses,
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


# --- Shared, hand-built fixture for permute / unpermute -------------------
# 4 tokens, 3 experts. Routing (token -> experts):
#   token 0 -> {0, 2}   token 1 -> {1}
#   token 2 -> {0}      token 3 -> {1, 2}
# permute walks experts in order, and within an expert the tokens in
# ascending index (masked_select is C-order over [experts, tokens]):
#   expert 0: tokens 0, 2 ; expert 1: tokens 1, 3 ; expert 2: tokens 0, 3
# => sorted_indices == [0, 2, 1, 3, 0, 3]
_ROUTING_MAP = [
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [0, 1, 1],
]
_TOKENS = [
    [10.0, 11.0],
    [20.0, 21.0],
    [30.0, 31.0],
    [40.0, 41.0],
]
_EXPECTED_SORTED_INDICES = [0, 2, 1, 3, 0, 3]
_EXPECTED_PERMUTED = [
    [10.0, 11.0],
    [30.0, 31.0],
    [20.0, 21.0],
    [40.0, 41.0],
    [10.0, 11.0],
    [40.0, 41.0],
]


class _RecordingLogs:
    """Stand-in training-logs sink. Records every ``update`` payload.

    This mocks a genuine, non-under-test collaborator (the global training
    logs registry) so the moe_utils logging code can be exercised without a
    real trainer. ``is_moe_balance_logs_enabled`` lets us drive the balance
    gate in ``global_moe_balance_training_logs_enabled``.
    """

    def __init__(self, balance_enabled=True):
        self._balance_enabled = balance_enabled
        self.updates = []

    def is_moe_balance_logs_enabled(self):
        return self._balance_enabled

    def update(self, **kwargs):
        self.updates.append(dict(kwargs))


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestIsTensor(unittest.TestCase):
    def test_tensor_is_recognised(self):
        self.assertTrue(is_tensor(paddle.to_tensor([1.0, 2.0])))

    def test_non_tensors_rejected(self):
        self.assertFalse(is_tensor(42))
        self.assertFalse(is_tensor([1, 2, 3]))
        self.assertFalse(is_tensor("tensor"))
        self.assertFalse(is_tensor(None))


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestDetachAndRequiresGrad(unittest.TestCase):
    def test_preserves_stop_gradient_and_passes_non_tensors(self):
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        x.stop_gradient = False
        y = paddle.to_tensor([[5.0, 6.0], [7.0, 8.0]])
        y.stop_gradient = True

        ret = detach_and_requires_grad_(x, y, 42)

        self.assertEqual(len(ret), 3)
        # Detach returns a new tensor object, not the input.
        self.assertIsNot(ret[0], x)
        self.assertIsNot(ret[1], y)
        # Original stop_gradient flags are reproduced on the detached copies.
        self.assertFalse(ret[0].stop_gradient)
        self.assertTrue(ret[1].stop_gradient)
        # Values are preserved by detach.
        np.testing.assert_array_equal(ret[0].numpy(), x.numpy())
        np.testing.assert_array_equal(ret[1].numpy(), y.numpy())
        # Non-tensors pass straight through unchanged.
        self.assertEqual(ret[2], 42)
        self.assertFalse(is_tensor(ret[2]))


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestAddAuxiliaryLoss(unittest.TestCase):
    def test_forward_is_value_preserving_clone(self):
        x = paddle.to_tensor([[1.0, 2.0, 3.0]])
        loss = paddle.to_tensor(0.5)
        out = AddAuxiliaryLoss.apply(x, loss)
        # forward returns x.clone(): same values, independent object.
        np.testing.assert_array_equal(out.numpy(), x.numpy())
        self.assertIsNot(out, x)

    def test_backward_passes_x_grad_and_injects_unit_aux_grad(self):
        x = paddle.to_tensor([[1.0, 2.0, 3.0]])
        x.stop_gradient = False
        loss = paddle.to_tensor(0.5)
        loss.stop_gradient = False

        out = AddAuxiliaryLoss.apply(x, loss)
        # Non-uniform upstream grad so a passthrough bug cannot hide.
        upstream = paddle.to_tensor([[1.0, 10.0, 100.0]])
        out.backward(upstream)

        # grad_output flows to x unchanged ...
        self.assertIsNotNone(x.grad)
        np.testing.assert_array_equal(x.grad.numpy(), upstream.numpy())
        # ... and the aux loss receives a unit gradient (the whole point of
        # this PyLayer: it re-injects a grad of ones into the scalar aux loss
        # regardless of the upstream magnitude on x).
        self.assertIsNotNone(loss.grad)
        self.assertAlmostEqual(float(loss.grad.sum()), 1.0, places=6)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestFakeClone(unittest.TestCase):
    def test_contiguous_forward_and_backward_are_identity(self):
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        x.stop_gradient = False
        out = FakeClone.apply(x)
        np.testing.assert_array_equal(out.numpy(), x.numpy())

        upstream = paddle.to_tensor([[5.0, 6.0], [7.0, 8.0]])
        out.backward(upstream)
        self.assertIsNotNone(x.grad)
        np.testing.assert_array_equal(x.grad.numpy(), upstream.numpy())

    def test_non_contiguous_input_is_materialised_correctly(self):
        # transpose() yields a non-contiguous view -> FakeClone takes the
        # clone branch and must reproduce the transposed layout exactly.
        base = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        xt = base.transpose([1, 0])  # [3, 2], non-contiguous
        out = FakeClone.apply(xt)
        self.assertEqual(list(out.shape), [3, 2])
        np.testing.assert_array_equal(
            out.numpy(), np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestPermute(unittest.TestCase):
    def test_groups_tokens_by_expert_in_order(self):
        tokens = paddle.to_tensor(_TOKENS)
        routing_map = paddle.to_tensor(_ROUTING_MAP, dtype="float32")

        permuted, sorted_indices = permute(tokens, routing_map)

        self.assertEqual(
            sorted_indices.numpy().tolist(), _EXPECTED_SORTED_INDICES
        )
        np.testing.assert_array_equal(
            permuted.numpy(), np.array(_EXPECTED_PERMUTED)
        )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestUnpermute(unittest.TestCase):
    def test_scatter_accumulates_per_token_contributions(self):
        # Feed the known permute output back. Each token receives the sum of
        # its per-expert rows: tokens 0 and 3 appear twice, so they double.
        permuted = paddle.to_tensor(_EXPECTED_PERMUTED)
        sorted_indices = paddle.to_tensor(
            _EXPECTED_SORTED_INDICES, dtype="int64"
        )
        restored = unpermute(permuted, sorted_indices, restore_shape=[4, 2])
        expected = np.array(
            [
                [20.0, 22.0],  # token 0: [10,11] + [10,11]
                [20.0, 21.0],  # token 1: single expert
                [30.0, 31.0],  # token 2: single expert
                [80.0, 82.0],  # token 3: [40,41] + [40,41]
            ]
        )
        np.testing.assert_allclose(restored.numpy(), expected, atol=1e-5)

    def test_probs_applied_per_token_expert_pair_then_accumulated(self):
        permuted = paddle.to_tensor(_EXPECTED_PERMUTED)
        sorted_indices = paddle.to_tensor(
            _EXPECTED_SORTED_INDICES, dtype="int64"
        )
        routing_map = paddle.to_tensor(_ROUTING_MAP, dtype="float32")
        # probs[token, expert]; only routed positions are read (via routing
        # map), so masked-out entries are irrelevant.
        probs = paddle.to_tensor(
            [
                [0.5, 0.0, 0.9],  # token 0: e0=0.5, e2=0.9
                [0.0, 0.3, 0.0],  # token 1: e1=0.3
                [0.7, 0.0, 0.0],  # token 2: e0=0.7
                [0.0, 0.2, 0.4],  # token 3: e1=0.2, e2=0.4
            ],
            dtype="float32",
        )
        # Expert-major prob order matches sorted_indices:
        #   (t0,e0)=0.5 (t2,e0)=0.7 (t1,e1)=0.3 (t3,e1)=0.2 (t0,e2)=0.9 (t3,e2)=0.4
        # Weighted rows then scatter-accumulated back per token:
        restored = unpermute(
            permuted,
            sorted_indices,
            restore_shape=[4, 2],
            probs=probs,
            routing_map=routing_map,
        )
        expected = np.array(
            [
                [0.5 * 10 + 0.9 * 10, 0.5 * 11 + 0.9 * 11],  # token 0
                [0.3 * 20, 0.3 * 21],  # token 1
                [0.7 * 30, 0.7 * 31],  # token 2
                [0.2 * 40 + 0.4 * 40, 0.2 * 41 + 0.4 * 41],  # token 3
            ]
        )
        np.testing.assert_allclose(restored.numpy(), expected, atol=1e-5)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestSortChunksByIdxs(unittest.TestCase):
    def test_equal_chunks_reordered(self):
        # row i == [i, i] so a mis-split or mis-order is visible.
        data = paddle.to_tensor(
            [[float(i), float(i)] for i in range(6)], dtype="float32"
        )
        out, probs = sort_chunks_by_idxs(
            data,
            paddle.to_tensor([2, 2, 2]),
            paddle.to_tensor([2, 0, 1]),
        )
        # chunks c0=[0,1] c1=[2,3] c2=[4,5]; order [2,0,1] => c2,c0,c1
        expected = np.array(
            [[4, 4], [5, 5], [0, 0], [1, 1], [2, 2], [3, 3]], dtype="float32"
        )
        np.testing.assert_array_equal(out.numpy(), expected)
        self.assertIsNone(probs)

    def test_unequal_chunks_reordered(self):
        data = paddle.to_tensor(
            [[float(i), float(i)] for i in range(6)], dtype="float32"
        )
        out, probs = sort_chunks_by_idxs(
            data,
            paddle.to_tensor([1, 3, 2]),
            paddle.to_tensor([1, 2, 0]),
        )
        # c0=[0] c1=[1,2,3] c2=[4,5]; order [1,2,0] => c1,c2,c0
        expected = np.array(
            [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [0, 0]], dtype="float32"
        )
        np.testing.assert_array_equal(out.numpy(), expected)
        self.assertIsNone(probs)

    def test_probs_currently_unsupported_returns_none(self):
        # Production has a TODO and always returns None for permuted_probs,
        # even when probs are supplied. Lock that current contract.
        data = paddle.to_tensor([[1.0], [2.0]], dtype="float32")
        _, probs = sort_chunks_by_idxs(
            data,
            paddle.to_tensor([1, 1]),
            paddle.to_tensor([1, 0]),
            probs=paddle.to_tensor([0.3, 0.7], dtype="float32"),
        )
        self.assertIsNone(probs)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestAllGatherLocalTokensSingleRank(unittest.TestCase):
    def test_group_none_flattens_and_adds_rank_axis(self):
        # World-size-1 local path only: no real all-gather is exercised here.
        tokens = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        out = _all_gather_local_tokens(tokens, group=None)
        self.assertEqual(list(out.shape), [1, 4])
        np.testing.assert_array_equal(out.numpy(), [[1.0, 2.0, 3.0, 4.0]])


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestLogMoeLosses(unittest.TestCase):
    def test_logs_both_losses_with_and_without_layer_keys(self):
        logs = _RecordingLogs(balance_enabled=True)
        with mock.patch.object(
            moe_utils, "get_global_training_logs", return_value=logs
        ):
            log_moe_losses(
                layer_number=3,
                aux_loss=paddle.to_tensor(0.1),
                z_loss=paddle.to_tensor(0.01),
            )
        self.assertEqual(len(logs.updates), 1)
        payload = logs.updates[0]
        self.assertEqual(
            set(payload),
            {"aux_loss", "aux_loss_layer_3", "zloss", "zloss_layer_3"},
        )
        self.assertAlmostEqual(float(payload["aux_loss"]), 0.1, places=5)
        self.assertAlmostEqual(
            float(payload["aux_loss_layer_3"]), 0.1, places=5
        )
        self.assertAlmostEqual(float(payload["zloss"]), 0.01, places=5)
        self.assertAlmostEqual(float(payload["zloss_layer_3"]), 0.01, places=5)

    def test_layer_none_and_missing_zloss_drop_those_keys(self):
        logs = _RecordingLogs(balance_enabled=True)
        with mock.patch.object(
            moe_utils, "get_global_training_logs", return_value=logs
        ):
            log_moe_losses(layer_number=None, aux_loss=paddle.to_tensor(0.2))
        self.assertEqual(len(logs.updates), 1)
        self.assertEqual(set(logs.updates[0]), {"aux_loss"})
        self.assertAlmostEqual(float(logs.updates[0]["aux_loss"]), 0.2, 5)

    def test_disabled_balance_logs_skips_update(self):
        logs = _RecordingLogs(balance_enabled=False)
        with mock.patch.object(
            moe_utils, "get_global_training_logs", return_value=logs
        ):
            log_moe_losses(layer_number=1, aux_loss=paddle.to_tensor(0.1))
        self.assertEqual(logs.updates, [])


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestLogSummary(unittest.TestCase):
    def _run(self, key, layer_number, data, is_mtp_layer=False):
        logs = _RecordingLogs()
        with mock.patch.object(
            moe_utils, "get_global_training_logs", return_value=logs
        ):
            moe_utils._log_summary(
                key,
                layer_number,
                paddle.to_tensor(data, dtype="float32"),
                is_mtp_layer=is_mtp_layer,
            )
        return logs

    def test_empty_tensor_skips_update(self):
        logs = self._run("tokens", 4, [])
        self.assertEqual(logs.updates, [])

    def test_statistics_and_key_naming(self):
        # data = [1, 2, 6]: max=6 min=1 mean=3 median=2 (odd count).
        logs = self._run("tokens", 4, [1.0, 2.0, 6.0])
        self.assertEqual(len(logs.updates), 1)
        payload = logs.updates[0]
        prefix = "tokens_layer_4"
        self.assertEqual(
            set(payload),
            {
                f"{prefix}_max",
                f"{prefix}_min",
                f"{prefix}_var",
                f"{prefix}_median",
                f"{prefix}_mean",
                f"{prefix}_max_mean_ratio",
                f"{prefix}_min_mean_ratio",
            },
        )
        self.assertAlmostEqual(payload[f"{prefix}_max"], 6.0, places=5)
        self.assertAlmostEqual(payload[f"{prefix}_min"], 1.0, places=5)
        self.assertAlmostEqual(payload[f"{prefix}_mean"], 3.0, places=5)
        self.assertAlmostEqual(payload[f"{prefix}_median"], 2.0, places=5)
        self.assertAlmostEqual(
            payload[f"{prefix}_max_mean_ratio"], 6.0 / 3.0, places=5
        )
        self.assertAlmostEqual(
            payload[f"{prefix}_min_mean_ratio"], 1.0 / 3.0, places=5
        )
        # ddof convention of var is not pinned here; just require it real.
        self.assertTrue(np.isfinite(payload[f"{prefix}_var"]))
        self.assertGreater(payload[f"{prefix}_var"], 0.0)

    def test_mtp_layer_changes_prefix(self):
        logs = self._run("tokens", 4, [1.0, 2.0, 6.0], is_mtp_layer=True)
        payload = logs.updates[0]
        self.assertIn("tokens_mtp_layer_4_mean", payload)
        self.assertAlmostEqual(
            payload["tokens_mtp_layer_4_mean"], 3.0, places=5
        )

    def test_zero_mean_forces_unit_ratios(self):
        # mean == 0 -> both ratios are clamped to 1.0 by the guard.
        logs = self._run("x", 1, [-2.0, 0.0, 2.0])
        payload = logs.updates[0]
        self.assertAlmostEqual(payload["x_layer_1_mean"], 0.0, places=5)
        self.assertAlmostEqual(payload["x_layer_1_max"], 2.0, places=5)
        self.assertAlmostEqual(payload["x_layer_1_min"], -2.0, places=5)
        self.assertEqual(payload["x_layer_1_max_mean_ratio"], 1.0)
        self.assertEqual(payload["x_layer_1_min_mean_ratio"], 1.0)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestLogTokensPerExpert(unittest.TestCase):
    def _capture(self, layer_number, key, summary, count):
        # _log_summary is separately tested; here we only observe how
        # _log_tokens_per_expert orchestrates the two calls and what avg_data
        # it computes, so we spy on it and record the real payloads.
        calls = []

        def rec(k, ln, data, is_mtp_layer=False):
            calls.append(
                (k, ln, np.asarray(data.numpy(), dtype="float64"), is_mtp_layer)
            )

        with mock.patch.object(moe_utils, "_log_summary", side_effect=rec):
            moe_utils._log_tokens_per_expert(
                layer_number,
                key,
                paddle.to_tensor(summary, dtype="float32"),
                paddle.to_tensor(count, dtype="float32"),
            )
        return calls

    def test_nonzero_count_divides_summary_for_avg(self):
        calls = self._capture(5, "tpe", [2.0, 4.0, 8.0], [2.0])
        self.assertEqual(len(calls), 2)
        # first: the averaged series under "<key>_avg"
        self.assertEqual(calls[0][0], "tpe_avg")
        self.assertEqual(calls[0][1], 5)
        np.testing.assert_allclose(calls[0][2], [1.0, 2.0, 4.0])
        # second: the raw series under "<key>"
        self.assertEqual(calls[1][0], "tpe")
        self.assertEqual(calls[1][1], 5)
        np.testing.assert_allclose(calls[1][2], [2.0, 4.0, 8.0])

    def test_zero_count_replaced_by_one(self):
        # count == 0 -> replaced by ones, so avg equals the raw summary.
        calls = self._capture(1, "k", [3.0, 6.0], [0.0])
        self.assertEqual(calls[0][0], "k_avg")
        np.testing.assert_allclose(calls[0][2], [3.0, 6.0])
        np.testing.assert_allclose(calls[1][2], [3.0, 6.0])


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestLogMoeBalance(unittest.TestCase):
    def _capture(self, tokens_per_expert, num_experts_per_tok):
        tpe_calls = []
        card_calls = []

        def rec_tpe(layer_number, key, summary_data, count, is_mtp_layer=False):
            tpe_calls.append(
                (
                    layer_number,
                    key,
                    np.asarray(summary_data.numpy()),
                    np.asarray(count.numpy(), dtype="float64"),
                )
            )

        def rec_card(layer_number, local_tokens_by_rank, is_mtp_layer=False):
            card_calls.append(
                (layer_number, np.asarray(local_tokens_by_rank.numpy()))
            )

        with (
            mock.patch.object(
                moe_utils, "_log_tokens_per_expert", side_effect=rec_tpe
            ),
            mock.patch.object(
                moe_utils, "_log_local_tokens_per_card", side_effect=rec_card
            ),
        ):
            log_moe_balance(
                layer_number=7,
                moe_group=None,
                num_experts_per_tok=num_experts_per_tok,
                tokens_per_expert=tokens_per_expert,
            )
        return tpe_calls, card_calls

    def test_count_is_total_tokens_over_topk(self):
        tpe_calls, card_calls = self._capture([3, 5, 4], num_experts_per_tok=2)
        self.assertEqual(len(tpe_calls), 1)
        layer_number, key, summary, count = tpe_calls[0]
        self.assertEqual(layer_number, 7)
        self.assertEqual(key, "tokens_per_expert")
        np.testing.assert_array_equal(summary, [3, 5, 4])
        # (3 + 5 + 4) / topk(2) == 6.0
        np.testing.assert_allclose(count, [6.0])
        # the per-card helper receives the un-flattened [rank, expert] tensor
        self.assertEqual(len(card_calls), 1)
        self.assertEqual(card_calls[0][0], 7)
        np.testing.assert_array_equal(card_calls[0][1], [[3, 5, 4]])

    def test_topk_guard_defaults_to_one(self):
        # num_experts_per_tok None -> max(int(None or 1), 1) == 1, so count
        # is the raw total.
        tpe_calls, _ = self._capture([2, 2], num_experts_per_tok=None)
        _, _, summary, count = tpe_calls[0]
        np.testing.assert_array_equal(summary, [2, 2])
        np.testing.assert_allclose(count, [4.0])

    def test_none_tokens_per_expert_is_a_noop(self):
        tpe_calls, card_calls = self._capture(None, num_experts_per_tok=2)
        self.assertEqual(tpe_calls, [])
        self.assertEqual(card_calls, [])


if __name__ == "__main__":
    unittest.main()
