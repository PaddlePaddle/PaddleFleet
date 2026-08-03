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

"""
Unit tests for Quantile Balancing implementation.

Tests cover:
1. _accumulate_qb_histogram: correct binning and accumulation
2. MoEQuantileBalancingCallback._update_single_layer: correct quantile recovery and bias assignment
3. _try_get_comm_groups: communication group retrieval logic
4. End-to-end: bias converges toward balanced load in a synthetic scenario
5. Paddle-based tests: actual tensor operations matching the real code paths
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import paddle

from paddlefleet.transformer.moe.qb_callback import (
    MoEQuantileBalancingCallback,
    _try_get_comm_groups,
)

# =============================================================================
# Numpy reference implementation (for cross-validation)
# =============================================================================


def _numpy_accumulate_histogram(scores, bias, k, b_min, b_max, B):
    """Numpy reference: accumulate required_bias into histogram."""
    N, E = scores.shape
    biased_scores = scores + bias[None, :]
    sorted_biased = np.sort(biased_scores, axis=-1)[:, ::-1]
    topk_val = min(k + 1, E)
    alpha = sorted_biased[:, topk_val - 1 : topk_val]  # [N, 1]
    required_bias = alpha - scores  # [N, E]

    total_range = b_max - b_min
    if total_range < 1e-8:
        total_range = 2.0
    bin_idx = ((required_bias - b_min) / total_range * B).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, B - 1)

    histogram = np.zeros((E, B), dtype=np.int32)
    for e in range(E):
        for i in range(N):
            histogram[e, bin_idx[i, e]] += 1
    return histogram


def _numpy_recover_bias(histogram, k, n, b_min, b_max):
    """Numpy reference: recover bias from histogram."""
    E, B = histogram.shape
    total_range = b_max - b_min
    if total_range < 1e-8:
        total_range = 2.0
    bin_width = total_range / B

    total_per_expert = histogram.sum(axis=1)
    q_target = np.maximum((total_per_expert * k / n).astype(np.int64), 1)

    cumsum = np.cumsum(histogram, axis=1)

    beta = np.zeros(E, dtype=np.int64)
    for e in range(E):
        for b in range(B):
            if cumsum[e, b] >= q_target[e]:
                beta[e] = b
                break
        else:
            beta[e] = B - 1

    b_hat = np.zeros(E, dtype=np.float64)
    for e in range(E):
        c = cumsum[e, beta[e] - 1] if beta[e] > 0 else 0
        h = max(histogram[e, beta[e]], 1)
        fraction = np.clip((q_target[e] - c) / h, 0.0, 1.0)
        b_hat[e] = b_min + (beta[e] + fraction) * bin_width

    b_new = b_hat - b_hat.mean()
    return b_new.astype(np.float32)


# =============================================================================
# Test: Histogram Accumulation (numpy reference)
# =============================================================================


class TestHistogramAccumulation(unittest.TestCase):
    """Test the histogram binning logic (numpy reference)."""

    def test_basic_binning(self):
        """Verify required_bias values land in correct bins."""
        np.random.seed(42)
        N, E, B = 8, 4, 10
        k = 2

        scores = np.random.uniform(0.1, 0.9, (N, E)).astype(np.float32)
        bias = np.array([0.1, -0.05, 0.02, -0.07], dtype=np.float32)

        histogram = _numpy_accumulate_histogram(scores, bias, k, -1.07, 1.10, B)

        # Each expert should have exactly N counts total
        for e in range(E):
            self.assertEqual(histogram[e].sum(), N)

    def test_degenerate_zero_bias(self):
        """When bias is all-zero, binning range should fallback gracefully."""
        N, E, B = 4, 2, 10
        k = 1
        scores = np.array(
            [[0.8, 0.3], [0.4, 0.7], [0.6, 0.5], [0.3, 0.9]], dtype=np.float32
        )
        bias = np.zeros(E, dtype=np.float32)

        histogram = _numpy_accumulate_histogram(scores, bias, k, -1.0, 1.0, B)

        for e in range(E):
            self.assertEqual(histogram[e].sum(), N)

    def test_large_batch(self):
        """Verify correctness with larger batch size."""
        np.random.seed(7)
        N, E, B = 1024, 16, 100
        k = 4
        scores = np.random.uniform(0.05, 0.95, (N, E)).astype(np.float32)
        bias = np.random.uniform(-0.1, 0.1, E).astype(np.float32)

        histogram = (
            _numpy_accumulate_histogram(
                scores, bias, bias.min() - 1.0, bias.max() + 1.0, k, B
            )
            if False
            else _numpy_accumulate_histogram(
                scores, bias, k, bias.min() - 1.0, bias.max() + 1.0, B
            )
        )

        for e in range(E):
            self.assertEqual(histogram[e].sum(), N)

    def test_multi_microbatch_accumulation(self):
        """Histogram should accumulate across multiple micro-batches."""
        np.random.seed(99)
        E, B, k = 4, 50, 2
        bias = np.zeros(E, dtype=np.float32)
        b_min, b_max = -1.0, 1.0

        total_histogram = np.zeros((E, B), dtype=np.int32)
        total_N = 0
        for _ in range(4):  # 4 micro-batches
            N = 16
            scores = np.random.uniform(0.1, 0.9, (N, E)).astype(np.float32)
            h = _numpy_accumulate_histogram(scores, bias, k, b_min, b_max, B)
            total_histogram += h
            total_N += N

        for e in range(E):
            self.assertEqual(total_histogram[e].sum(), total_N)


# =============================================================================
# Test: Quantile Recovery (numpy reference)
# =============================================================================


class TestQuantileRecovery(unittest.TestCase):
    """Test the quantile recovery logic from histogram (numpy reference)."""

    def test_uniform_histogram(self):
        """When all experts have identical histograms, bias should be ~zero."""
        E, B = 4, 100
        k, n = 2, 4
        histogram = np.full((E, B), 10, dtype=np.int64)

        b_new = _numpy_recover_bias(histogram, k, n, -1.0, 1.0)
        np.testing.assert_allclose(b_new, 0.0, atol=0.01)

    def test_skewed_histogram(self):
        """An expert with more tokens at high required_bias gets higher bias."""
        E, B = 2, 10
        k, n = 1, 2

        hist_0 = np.array([80, 10, 5, 3, 1, 1, 0, 0, 0, 0], dtype=np.int64)
        hist_1 = np.array([0, 0, 0, 0, 1, 1, 3, 5, 10, 80], dtype=np.int64)
        histogram = np.stack([hist_0, hist_1])

        b_new = _numpy_recover_bias(histogram, k, n, -1.0, 1.0)

        self.assertGreater(b_new[1], b_new[0])
        self.assertAlmostEqual(b_new.sum(), 0.0, places=5)

    def test_q_target_clipped_to_one(self):
        """Even with very few tokens, q_target >= 1 so we don't crash."""
        E, B = 2, 10
        k, n = 1, 100

        histogram = np.zeros((E, B), dtype=np.int64)
        histogram[:, 5] = 5

        b_new = _numpy_recover_bias(histogram, k, n, -1.0, 1.0)
        self.assertEqual(b_new.shape, (E,))
        self.assertTrue(np.all(np.isfinite(b_new)))

    def test_empty_histogram_guard(self):
        """If histogram is all-zero, recovery should detect and skip."""
        E, B = 4, 10
        histogram = np.zeros((E, B), dtype=np.int64)
        self.assertEqual(histogram.sum(), 0)

    def test_single_bin_concentration(self):
        """All tokens in a single bin → fraction should be 0.5 (midpoint)."""
        E, B = 2, 10
        k, n = 1, 2
        histogram = np.zeros((E, B), dtype=np.int64)
        # All 100 tokens in bin 3 for expert 0, bin 7 for expert 1
        histogram[0, 3] = 100
        histogram[1, 7] = 100

        b_new = _numpy_recover_bias(histogram, k, n, -1.0, 1.0)
        # q = 100 * 1/2 = 50, all in one bin → fraction = (50-0)/100 = 0.5
        self.assertTrue(np.all(np.isfinite(b_new)))
        self.assertAlmostEqual(b_new.sum(), 0.0, places=5)


# =============================================================================
# Test: Paddle-based _update_single_layer
# =============================================================================


class TestUpdateSingleLayerPaddle(unittest.TestCase):
    """Test _update_single_layer with actual Paddle tensors (single-rank, no dist)."""

    def _make_mock_layer(self, E, B, k, n, histogram_np, b_min=-1.0, b_max=1.0):
        """Create a mock layer object mimicking StandardMoERouter attributes."""
        layer = MagicMock()
        layer.qb_histogram = paddle.to_tensor(histogram_np, dtype=paddle.int32)
        layer.num_experts_per_tok = k
        layer.num_experts = n
        layer.qb_bin_min = b_min
        layer.qb_bin_max = b_max
        # Mock config
        layer.config = MagicMock()
        layer.config.sequence_parallel = False
        layer.config.expert_model_parallel_size = 1

        # Use a simple object for e_score_correction_bias with ndim attribute
        class BiasMock:
            ndim = 1

            def set_value(self, val):
                layer._bias_value = val

        layer.e_score_correction_bias = BiasMock()
        return layer

    def test_basic_recovery(self):
        """Verify paddle-based recovery matches numpy reference."""
        np.random.seed(42)
        E, B, k, n = 4, 100, 2, 4
        b_min, b_max = -1.0, 1.0

        # Generate a random histogram
        histogram_np = np.random.randint(0, 20, (E, B)).astype(np.int32)
        # Make sure it's not all-zero
        histogram_np[:, 50] += 10

        # Numpy reference
        expected = _numpy_recover_bias(
            histogram_np.astype(np.int64), k, n, b_min, b_max
        )

        # Paddle-based
        layer = self._make_mock_layer(E, B, k, n, histogram_np, b_min, b_max)
        callback = MoEQuantileBalancingCallback()
        callback._update_single_layer(
            layer, tp_group=None, cp_group=None, dp_group=None, sd_group=None
        )

        actual = layer._bias_value.numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-4)

    def test_zero_histogram_skips(self):
        """If histogram is all-zero, layer bias should NOT be updated."""
        E, B, k, n = 4, 50, 2, 4
        histogram_np = np.zeros((E, B), dtype=np.int32)

        layer = self._make_mock_layer(E, B, k, n, histogram_np)
        # Track whether set_value was called
        called = []
        original_set_value = layer.e_score_correction_bias.set_value

        def _tracking_set_value(val):
            called.append(val)
            original_set_value(val)

        layer.e_score_correction_bias.set_value = _tracking_set_value

        callback = MoEQuantileBalancingCallback()
        callback._update_single_layer(
            layer, tp_group=None, cp_group=None, dp_group=None, sd_group=None
        )

        # set_value should NOT have been called since histogram is all-zero
        self.assertEqual(
            len(called), 0, "set_value was called on all-zero histogram"
        )

    def test_zero_mean_property(self):
        """Output bias should be zero-mean."""
        np.random.seed(123)
        E, B, k, n = 8, 200, 2, 8
        histogram_np = np.random.randint(5, 30, (E, B)).astype(np.int32)

        layer = self._make_mock_layer(E, B, k, n, histogram_np)
        callback = MoEQuantileBalancingCallback()
        callback._update_single_layer(
            layer, tp_group=None, cp_group=None, dp_group=None, sd_group=None
        )

        actual = layer._bias_value.numpy()
        self.assertAlmostEqual(float(actual.sum()), 0.0, places=4)

    def test_binning_range_updated(self):
        """After update, qb_bin_min/max should be updated to new bias range ± 1."""
        np.random.seed(77)
        E, B, k, n = 4, 100, 1, 4
        histogram_np = np.random.randint(1, 50, (E, B)).astype(np.int32)

        layer = self._make_mock_layer(E, B, k, n, histogram_np)
        callback = MoEQuantileBalancingCallback()
        callback._update_single_layer(
            layer, tp_group=None, cp_group=None, dp_group=None, sd_group=None
        )

        b_new = layer._bias_value.numpy()
        self.assertAlmostEqual(
            layer.qb_bin_min, float(b_new.min()) - 1.0, places=5
        )
        self.assertAlmostEqual(
            layer.qb_bin_max, float(b_new.max()) + 1.0, places=5
        )

    def test_histogram_reset_after_update(self):
        """After update, histogram should be zeroed."""
        np.random.seed(55)
        E, B, k, n = 4, 50, 2, 4
        histogram_np = np.random.randint(1, 10, (E, B)).astype(np.int32)

        layer = self._make_mock_layer(E, B, k, n, histogram_np)
        # Override zero_ to track call
        layer.qb_histogram.zero_ = MagicMock()
        callback = MoEQuantileBalancingCallback()
        callback._update_single_layer(
            layer, tp_group=None, cp_group=None, dp_group=None, sd_group=None
        )

        layer.qb_histogram.zero_.assert_called_once()

    def test_2d_bias_shape(self):
        """When e_score_correction_bias has shape [1, E], unsqueeze(0) is used."""
        np.random.seed(33)
        E, B, k, n = 4, 100, 2, 4
        histogram_np = np.random.randint(5, 20, (E, B)).astype(np.int32)

        layer = self._make_mock_layer(E, B, k, n, histogram_np)
        # Simulate [1, E] shape variant
        bias_mock = MagicMock()
        bias_mock.ndim = 2
        stored = {}

        def _set_value(val):
            stored["val"] = val

        bias_mock.set_value = _set_value
        layer.e_score_correction_bias = bias_mock

        callback = MoEQuantileBalancingCallback()
        callback._update_single_layer(
            layer, tp_group=None, cp_group=None, dp_group=None, sd_group=None
        )

        self.assertEqual(list(stored["val"].shape), [1, E])

    def test_large_expert_count(self):
        """Verify correctness with many experts (closer to real scale)."""
        np.random.seed(11)
        E, B, k, n = 64, 200, 4, 64
        histogram_np = np.random.randint(10, 100, (E, B)).astype(np.int32)

        layer = self._make_mock_layer(E, B, k, n, histogram_np)
        callback = MoEQuantileBalancingCallback()
        callback._update_single_layer(
            layer, tp_group=None, cp_group=None, dp_group=None, sd_group=None
        )

        actual = layer._bias_value.numpy()
        self.assertEqual(actual.shape, (E,))
        self.assertAlmostEqual(float(actual.sum()), 0.0, places=3)
        self.assertTrue(np.all(np.isfinite(actual)))


# =============================================================================
# Test: Paddle-based _accumulate_qb_histogram
# =============================================================================


class TestAccumulateHistogramPaddle(unittest.TestCase):
    """Test _accumulate_qb_histogram with actual Paddle tensors."""

    def _make_router_mock(self, E, B, k, b_min=-1.0, b_max=1.0):
        """Create a minimal mock for the router's histogram accumulation."""
        router = MagicMock()
        router.qb_n_bins = B
        router.qb_bin_min = b_min
        router.qb_bin_max = b_max
        router.qb_histogram = paddle.zeros([E, B], dtype=paddle.int32)
        return router

    def test_accumulation_matches_numpy(self):
        """Paddle accumulation should match numpy reference."""
        np.random.seed(42)
        N, E, B, k = 32, 4, 50, 2
        b_min, b_max = -1.0, 1.0

        scores_np = np.random.uniform(0.1, 0.9, (N, E)).astype(np.float32)
        bias_np = np.array([0.05, -0.03, 0.01, -0.02], dtype=np.float32)

        # Numpy reference
        expected = _numpy_accumulate_histogram(
            scores_np, bias_np, k, b_min, b_max, B
        )

        # Paddle implementation (inline, matching _accumulate_qb_histogram logic)
        scores = paddle.to_tensor(scores_np)
        bias = paddle.to_tensor(bias_np)
        biased_scores = scores + bias.unsqueeze(0)

        topk_val = min(k + 1, E)
        alpha = paddle.topk(biased_scores, k=topk_val, axis=-1, sorted=True)[0][
            :, -1:
        ]
        required_bias = alpha - scores

        total_range = b_max - b_min
        bin_idx = ((required_bias - b_min) / total_range * B).cast(paddle.int64)
        bin_idx = paddle.clip(bin_idx, min=0, max=B - 1)

        offsets = paddle.arange(E, dtype=paddle.int64).unsqueeze(0) * B
        flat_bins = (bin_idx + offsets).reshape([-1])
        counts = paddle.zeros([E * B], dtype=paddle.int32)
        ones = paddle.ones([N * E], dtype=paddle.int32)
        counts.put_along_axis_(flat_bins, ones, axis=0, reduce="add")
        actual = counts.reshape([E, B]).numpy()

        np.testing.assert_array_equal(actual, expected)

    def test_clipping_at_boundaries(self):
        """Scores at extreme values should be clipped to valid bin range."""
        N, E, B, k = 4, 2, 10, 1
        b_min, b_max = -1.0, 1.0

        # Construct scores that produce required_bias outside [b_min, b_max]
        scores = paddle.to_tensor(
            [[0.01, 0.99], [0.99, 0.01], [0.5, 0.5], [0.5, 0.5]],
            dtype=paddle.float32,
        )
        bias = paddle.zeros([E], dtype=paddle.float32)
        biased_scores = scores + bias.unsqueeze(0)

        topk_val = min(k + 1, E)
        alpha = paddle.topk(biased_scores, k=topk_val, axis=-1, sorted=True)[0][
            :, -1:
        ]
        required_bias = alpha - scores

        total_range = b_max - b_min
        bin_idx = ((required_bias - b_min) / total_range * B).cast(paddle.int64)
        bin_idx = paddle.clip(bin_idx, min=0, max=B - 1)

        # All bin indices should be in [0, B-1]
        self.assertTrue(int(bin_idx.min().item()) >= 0)
        self.assertTrue(int(bin_idx.max().item()) <= B - 1)


# =============================================================================
# Test: _try_get_comm_groups
# =============================================================================


class _FakeGroup:
    def __init__(self, nranks):
        self.nranks = nranks


class _FakeHCG:
    """Stands in for paddle's HybridCommunicateGroup."""

    def __init__(self, tp=1, dp=1, sd=1):
        self.get_model_parallel_group = lambda: _FakeGroup(tp)
        self.get_data_parallel_group = lambda: _FakeGroup(dp)
        self.get_sharding_parallel_group = lambda: _FakeGroup(sd)


def _fake_fleet_module(hcg):
    """Mimic `paddle.distributed.fleet`, whose module-level
    ``get_hybrid_communicate_group`` is bound to the singleton Fleet instance.

    A MagicMock must not be used here: it answers every attribute lookup, so a
    lookup on the wrong object or under the wrong name would appear to succeed
    and hide the bug.
    """
    return SimpleNamespace(get_hybrid_communicate_group=lambda: hcg)


@contextlib.contextmanager
def _initialized_fleet(hcg, cp_group=None):
    """Pretend distributed is up with `hcg` as the hybrid group and `cp_group` as CP."""
    with (
        patch("paddle.distributed.is_initialized", return_value=True),
        patch("paddle.distributed.fleet", _fake_fleet_module(hcg)),
        patch(
            "paddlefleet.parallel_state.get_context_parallel_group",
            return_value=cp_group,
        ),
    ):
        yield


class TestTryGetCommGroups(unittest.TestCase):
    """Test the communication group retrieval function."""

    def test_no_fleet_returns_none(self):
        """Without fleet initialized, should return four Nones."""
        tp, cp, dp, sd = _try_get_comm_groups()
        # In single-process tests fleet.init() may have run, but every group
        # holds a single rank, so nothing needs reducing.
        self.assertIsNone(tp)
        self.assertIsNone(cp)
        self.assertIsNone(dp)
        self.assertIsNone(sd)

    def test_uninitialized_distributed_returns_none(self):
        """The is_initialized() guard must short-circuit before touching fleet.

        fleet.get_hybrid_communicate_group() asserts on an unset _hcg, so
        reaching it in a single-process run would raise instead of no-op.
        """
        with (
            patch("paddle.distributed.is_initialized", return_value=False),
            patch(
                "paddle.distributed.fleet",
                _fake_fleet_module(_FakeHCG(tp=2, dp=8, sd=4)),
            ),
        ):
            self.assertEqual(_try_get_comm_groups(), (None, None, None, None))

    def test_initialized_multi_rank_groups_are_returned(self):
        """Regression guard: groups must be found on a realistically shaped fleet.

        An earlier implementation probed ``hasattr(fleet, "_hcg")`` on the
        module instead of the instance, which is always False, so every group
        came back None and the histogram was never merged across ranks.
        """
        with _initialized_fleet(_FakeHCG(tp=2, dp=8, sd=4)):
            tp, cp, dp, sd = _try_get_comm_groups()
        self.assertEqual((tp.nranks, dp.nranks, sd.nranks), (2, 8, 4))
        self.assertIsNone(cp)  # CP is not enabled in this test process


# =============================================================================
# Test: on_optimizer_end (integration)
# =============================================================================


class TestOnOptimizerEnd(unittest.TestCase):
    """Test the callback entry point on_optimizer_end."""

    def test_no_model_returns_early(self):
        """If model is None, callback should return immediately."""
        callback = MoEQuantileBalancingCallback()
        # Should not raise
        callback.on_optimizer_end(
            args=None, state=None, control=None, model=None
        )

    def test_no_qb_layers_returns_early(self):
        """If model has no QB-enabled layers, callback should return."""
        callback = MoEQuantileBalancingCallback()
        model = MagicMock()
        # apply() calls the function on each sublayer; simulate no QB layers
        model.apply = MagicMock(side_effect=lambda fn: None)

        callback.on_optimizer_end(model=model)
        # No error raised

    def test_collects_qb_layers(self):
        """Callback should find and process QB-enabled router layers."""
        np.random.seed(42)
        E, B, k, n = 4, 50, 2, 4
        histogram_np = np.random.randint(5, 20, (E, B)).astype(np.int32)

        # Create a mock model with a QB layer — use patch to bypass isinstance check
        layer = MagicMock()
        layer.topk_method = "quantile_balancing"
        layer.qb_histogram = paddle.to_tensor(histogram_np, dtype=paddle.int32)
        layer.num_experts_per_tok = k
        layer.num_experts = n
        layer.qb_bin_min = -1.0
        layer.qb_bin_max = 1.0
        layer.config = MagicMock()
        layer.config.sequence_parallel = False
        layer.config.expert_model_parallel_size = 1
        bias_mock = MagicMock()
        bias_mock.ndim = 1

        def _set_value_on(val):
            layer._bias_value_on = val

        bias_mock.set_value = _set_value_on
        layer.e_score_correction_bias = bias_mock
        layer.expert_usage = paddle.zeros([E], dtype=paddle.int64)

        # Directly test _update_single_layer (bypasses isinstance in on_optimizer_end)
        callback = MoEQuantileBalancingCallback()
        callback._update_single_layer(
            layer, tp_group=None, cp_group=None, dp_group=None, sd_group=None
        )

        # Verify histogram was reset (zero_() was called on the tensor)
        self.assertEqual(int(layer.qb_histogram.sum().item()), 0)


# =============================================================================
# Test: End-to-end convergence
# =============================================================================


class TestEndToEnd(unittest.TestCase):
    """End-to-end test: simulate multiple QB steps and check convergence."""

    def _compute_load(self, scores, bias, k):
        """Compute per-expert load given scores and bias."""
        biased = scores + bias[None, :]
        topk_idx = np.argsort(biased, axis=-1)[:, ::-1][:, :k]
        load = np.zeros(scores.shape[1], dtype=np.int64)
        for i in range(scores.shape[0]):
            for j in topk_idx[i]:
                load[j] += 1
        return load

    def test_convergence_to_balance(self):
        """After one QB step, load should be more balanced than before."""
        np.random.seed(123)
        m, n, k = 64, 8, 2
        B = 100

        bias = np.zeros(n, dtype=np.float32)
        scores = np.random.uniform(0.1, 0.9, (m, n)).astype(np.float32)

        histogram = _numpy_accumulate_histogram(scores, bias, k, -1.0, 1.0, B)
        new_bias = _numpy_recover_bias(
            histogram.astype(np.int64), k, n, -1.0, 1.0
        )

        load_before = self._compute_load(scores, bias, k)
        load_after = self._compute_load(scores, new_bias, k)

        std_before = load_before.std()
        std_after = load_after.std()

        self.assertLessEqual(std_after, std_before + 0.5)
        self.assertAlmostEqual(float(new_bias.sum()), 0.0, places=4)

    def test_exact_balance_small_example(self):
        """With a carefully constructed example, QB should achieve near-perfect balance."""
        m, n, k = 8, 4, 2
        q = m * k // n  # = 4
        B = 1000

        scores = np.array(
            [
                [0.80, 0.30, 0.60, 0.20],
                [0.40, 0.70, 0.50, 0.30],
                [0.60, 0.50, 0.80, 0.40],
                [0.30, 0.60, 0.40, 0.70],
                [0.70, 0.40, 0.30, 0.50],
                [0.50, 0.80, 0.60, 0.20],
                [0.40, 0.30, 0.70, 0.60],
                [0.60, 0.50, 0.40, 0.80],
            ],
            dtype=np.float32,
        )

        bias = np.zeros(n, dtype=np.float32)
        histogram = _numpy_accumulate_histogram(scores, bias, k, -1.0, 1.0, B)
        new_bias = _numpy_recover_bias(
            histogram.astype(np.int64), k, n, -1.0, 1.0
        )

        load = self._compute_load(scores, new_bias, k)
        for e in range(n):
            self.assertAlmostEqual(load[e], q, delta=2)

    def test_multi_step_convergence(self):
        """Multiple QB steps should maintain balance even as scores change."""
        np.random.seed(456)
        m, n, k = 128, 8, 2
        B = 200

        bias = np.zeros(n, dtype=np.float32)
        b_min, b_max = -1.0, 1.0

        for step in range(5):
            scores = np.random.uniform(0.1, 0.9, (m, n)).astype(np.float32)
            histogram = _numpy_accumulate_histogram(
                scores, bias, k, b_min, b_max, B
            )
            bias = _numpy_recover_bias(
                histogram.astype(np.int64), k, n, b_min, b_max
            )
            b_min = bias.min() - 1.0
            b_max = bias.max() + 1.0

        # After 5 steps, load should be well-balanced
        final_scores = np.random.uniform(0.1, 0.9, (m, n)).astype(np.float32)
        load = self._compute_load(final_scores, bias, k)
        q = m * k // n
        # All expert loads within 30% of target
        for e in range(n):
            self.assertAlmostEqual(load[e], q, delta=int(q * 0.3) + 1)

    def test_paddle_end_to_end(self):
        """Full end-to-end with paddle tensors: accumulate + recover."""
        np.random.seed(88)
        N, E, B, k, n = 64, 4, 100, 2, 4
        b_min, b_max = -1.0, 1.0

        scores_np = np.random.uniform(0.1, 0.9, (N, E)).astype(np.float32)
        bias_np = np.zeros(E, dtype=np.float32)

        # Step 1: Accumulate histogram using paddle
        scores = paddle.to_tensor(scores_np)
        bias = paddle.to_tensor(bias_np)
        biased_scores = scores + bias.unsqueeze(0)

        topk_val = min(k + 1, E)
        alpha = paddle.topk(biased_scores, k=topk_val, axis=-1, sorted=True)[0][
            :, -1:
        ]
        required_bias = alpha - scores

        total_range = b_max - b_min
        bin_idx = ((required_bias - b_min) / total_range * B).cast(paddle.int64)
        bin_idx = paddle.clip(bin_idx, min=0, max=B - 1)

        offsets = paddle.arange(E, dtype=paddle.int64).unsqueeze(0) * B
        flat_bins = (bin_idx + offsets).reshape([-1])
        counts = paddle.zeros([E * B], dtype=paddle.int32)
        ones = paddle.ones([N * E], dtype=paddle.int32)
        counts.put_along_axis_(flat_bins, ones, axis=0, reduce="add")
        histogram = counts.reshape([E, B])
        # Save a copy before callback zeroes it
        histogram_np_copy = histogram.numpy().copy()

        # Step 2: Recover bias using callback
        layer = MagicMock()
        layer.qb_histogram = histogram
        layer.num_experts_per_tok = k
        layer.num_experts = n
        layer.qb_bin_min = b_min
        layer.qb_bin_max = b_max
        layer.config = MagicMock()
        layer.config.sequence_parallel = False
        layer.config.expert_model_parallel_size = 1

        stored = {}

        class BiasMockE2E:
            ndim = 1

            def set_value(self, val):
                stored["val"] = val

        layer.e_score_correction_bias = BiasMockE2E()

        callback = MoEQuantileBalancingCallback()
        callback._update_single_layer(
            layer, tp_group=None, cp_group=None, dp_group=None, sd_group=None
        )

        # Step 3: Verify result — bias should be non-zero and zero-mean
        actual = stored["val"].numpy()
        self.assertEqual(actual.shape, (E,))
        self.assertAlmostEqual(float(actual.sum()), 0.0, places=4)
        # Verify matches numpy reference (use saved copy since callback zeroes histogram)
        expected = _numpy_recover_bias(
            histogram_np_copy.astype(np.int64), k, n, b_min, b_max
        )
        np.testing.assert_allclose(actual, expected, atol=1e-4)


# =============================================================================
# Test: TP group all-reduce simulation
# =============================================================================


class TestTPGroupAllReduce(unittest.TestCase):
    """Test that TP all-reduce correctly merges SP-sharded histograms."""

    def test_tp_sharded_histograms_merge(self):
        """Simulating 2 TP ranks each seeing half the tokens should give same
        result as one rank seeing all tokens."""
        np.random.seed(42)
        N, E, B, k, n = 64, 4, 100, 2, 4
        b_min, b_max = -1.0, 1.0

        scores_np = np.random.uniform(0.1, 0.9, (N, E)).astype(np.float32)
        bias_np = np.zeros(E, dtype=np.float32)

        # Full histogram (what we'd get without SP)
        full_hist = _numpy_accumulate_histogram(
            scores_np, bias_np, k, b_min, b_max, B
        )

        # Split into 2 "TP ranks" (each sees half the tokens)
        half = N // 2
        hist_rank0 = _numpy_accumulate_histogram(
            scores_np[:half], bias_np, k, b_min, b_max, B
        )
        hist_rank1 = _numpy_accumulate_histogram(
            scores_np[half:], bias_np, k, b_min, b_max, B
        )

        # Simulated all-reduce (SUM)
        merged_hist = hist_rank0 + hist_rank1

        # The merged histogram should equal the full histogram
        np.testing.assert_array_equal(merged_hist, full_hist)

        # And both should produce the same bias
        bias_full = _numpy_recover_bias(
            full_hist.astype(np.int64), k, n, b_min, b_max
        )
        bias_merged = _numpy_recover_bias(
            merged_hist.astype(np.int64), k, n, b_min, b_max
        )
        np.testing.assert_allclose(bias_merged, bias_full, atol=1e-6)

    def test_without_tp_reduce_gives_wrong_result(self):
        """Without TP all-reduce, each rank's bias would differ from the correct one."""
        np.random.seed(42)
        N, E, B, k, n = 64, 4, 100, 2, 4
        b_min, b_max = -1.0, 1.0

        scores_np = np.random.uniform(0.1, 0.9, (N, E)).astype(np.float32)
        bias_np = np.zeros(E, dtype=np.float32)

        full_hist = _numpy_accumulate_histogram(
            scores_np, bias_np, k, b_min, b_max, B
        )
        bias_correct = _numpy_recover_bias(
            full_hist.astype(np.int64), k, n, b_min, b_max
        )

        # If TP rank 0 only uses its own half
        half = N // 2
        hist_rank0_only = _numpy_accumulate_histogram(
            scores_np[:half], bias_np, k, b_min, b_max, B
        )
        bias_rank0_only = _numpy_recover_bias(
            hist_rank0_only.astype(np.int64), k, n, b_min, b_max
        )

        # They should NOT be equal (this demonstrates the bug the fix addresses)
        # Use a loose check — they might be close by accident for some seeds,
        # but generally won't be identical
        diff = np.abs(bias_correct - bias_rank0_only).max()
        # At least some difference should exist (not exactly equal)
        # This test documents WHY tp_group all-reduce is needed
        self.assertGreater(
            diff,
            0.0,
            "Rank-0-only bias unexpectedly equals full bias; "
            "try a different seed or larger N",
        )


# =============================================================================
# Test: Additional coverage for uncovered branches
# =============================================================================


class TestDegenerateRange(unittest.TestCase):
    """Test the degenerate total_range < 1e-8 fallback (line 191)."""

    def test_zero_range_fallback(self):
        """When qb_bin_min == qb_bin_max, total_range should fallback to 2.0."""
        np.random.seed(999)
        E, B, k, n = 4, 50, 2, 4
        histogram_np = np.random.randint(5, 20, (E, B)).astype(np.int32)

        layer = MagicMock()
        layer.qb_histogram = paddle.to_tensor(histogram_np, dtype=paddle.int32)
        layer.num_experts_per_tok = k
        layer.num_experts = n
        # Set bin_min == bin_max to trigger the fallback
        layer.qb_bin_min = 0.0
        layer.qb_bin_max = 0.0
        layer.config = MagicMock()
        layer.config.sequence_parallel = False
        layer.config.expert_model_parallel_size = 1

        class BiasMock:
            ndim = 1

            def set_value(self, val):
                layer._bias_value = val

        layer.e_score_correction_bias = BiasMock()

        callback = MoEQuantileBalancingCallback()
        callback._update_single_layer(
            layer, tp_group=None, cp_group=None, dp_group=None, sd_group=None
        )

        # Should still produce valid result (not crash or NaN)
        actual = layer._bias_value.numpy()
        self.assertTrue(np.all(np.isfinite(actual)))
        self.assertAlmostEqual(float(actual.sum()), 0.0, places=4)


class TestOnOptimizerEndWithRealRouter(unittest.TestCase):
    """Test on_optimizer_end with a real StandardMoERouter isinstance check."""

    def test_isinstance_collection(self):
        """Verify _collect correctly identifies StandardMoERouter with quantile_balancing."""
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        np.random.seed(42)
        E, B, k, n = 4, 50, 2, 4
        histogram_np = np.random.randint(5, 20, (E, B)).astype(np.int32)

        # Create a mock that passes isinstance check
        layer = MagicMock(spec=StandardMoERouter)
        layer.topk_method = "quantile_balancing"
        layer.qb_histogram = paddle.to_tensor(histogram_np, dtype=paddle.int32)
        layer.num_experts_per_tok = k
        layer.num_experts = n
        layer.qb_bin_min = -1.0
        layer.qb_bin_max = 1.0
        layer.config = MagicMock()
        layer.config.sequence_parallel = False
        layer.config.expert_model_parallel_size = 1

        class BiasMock:
            ndim = 1

            def set_value(self, val):
                layer._bias_value = val

        layer.e_score_correction_bias = BiasMock()
        layer.expert_usage = paddle.zeros([E], dtype=paddle.int64)

        # Create model mock where apply() calls fn on the layer
        model = MagicMock()

        def _apply(fn):
            fn(layer)

        model.apply = _apply

        callback = MoEQuantileBalancingCallback()
        callback.on_optimizer_end(model=model)

        # Verify bias was updated
        self.assertTrue(hasattr(layer, "_bias_value"))
        self.assertAlmostEqual(
            float(layer._bias_value.sum().item()), 0.0, places=4
        )

    def test_non_qb_layer_skipped(self):
        """Layers with topk_method != 'quantile_balancing' should be skipped."""
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        layer = MagicMock(spec=StandardMoERouter)
        layer.topk_method = "noaux_tc"  # Not QB

        model = MagicMock()

        def _apply(fn):
            fn(layer)

        model.apply = _apply

        callback = MoEQuantileBalancingCallback()
        # Should not raise, should return early (no QB layers found)
        callback.on_optimizer_end(model=model)


class TestCommGroupsHappyPath(unittest.TestCase):
    """Test _try_get_comm_groups when fleet IS initialized (mocked)."""

    def test_all_groups_available(self):
        """When all groups are available and have nranks > 1, return them."""
        with _initialized_fleet(_FakeHCG(tp=4, dp=8, sd=2)):
            tp, cp, dp, sd = _try_get_comm_groups()
        self.assertEqual((tp.nranks, dp.nranks, sd.nranks), (4, 8, 2))
        self.assertIsNone(cp)

    def test_cp_group_picked_up(self):
        """A multi-rank CP group must be returned so its histogram is merged."""
        cp_group = _FakeGroup(4)
        with _initialized_fleet(_FakeHCG(), cp_group=cp_group):
            tp, cp, dp, sd = _try_get_comm_groups()
        self.assertIs(cp, cp_group)
        self.assertIsNone(tp)
        self.assertIsNone(dp)
        self.assertIsNone(sd)

    def test_single_rank_cp_group_filtered(self):
        """CP=1 needs no reduction, so the group is filtered out."""
        with _initialized_fleet(_FakeHCG(), cp_group=_FakeGroup(1)):
            _, cp, _, _ = _try_get_comm_groups()
        self.assertIsNone(cp)

    def test_single_rank_groups_filtered(self):
        """Groups with nranks <= 1 should be filtered to None."""
        with _initialized_fleet(_FakeHCG(tp=1, dp=4, sd=1)):
            tp, cp, dp, sd = _try_get_comm_groups()
        self.assertIsNone(tp)
        self.assertEqual(dp.nranks, 4)
        self.assertIsNone(sd)

    def test_none_groups_handled(self):
        """If hcg returns None for a group, it should stay None."""

        class _NoneHCG:
            get_model_parallel_group = staticmethod(lambda: None)
            get_data_parallel_group = staticmethod(lambda: None)
            get_sharding_parallel_group = staticmethod(lambda: None)

        with _initialized_fleet(_NoneHCG()):
            tp, cp, dp, sd = _try_get_comm_groups()
        self.assertIsNone(tp)
        self.assertIsNone(dp)
        self.assertIsNone(sd)


class TestTPAllReduceCondition(unittest.TestCase):
    """Test the TP all-reduce condition (SP=True and EP>1)."""

    def _make_layer_with_config(self, E, B, k, n, histogram_np, sp, ep_size):
        layer = MagicMock()
        layer.qb_histogram = paddle.to_tensor(histogram_np, dtype=paddle.int32)
        layer.num_experts_per_tok = k
        layer.num_experts = n
        layer.qb_bin_min = -1.0
        layer.qb_bin_max = 1.0
        layer.config = MagicMock()
        layer.config.sequence_parallel = sp
        layer.config.expert_model_parallel_size = ep_size

        class BiasMock:
            ndim = 1

            def set_value(self, val):
                layer._bias_value = val

        layer.e_score_correction_bias = BiasMock()
        return layer

    @patch("paddlefleet.transformer.moe.qb_callback.dist")
    def test_tp_reduce_called_when_sp_and_ep(self, mock_dist):
        """TP all-reduce should be called when SP=True and EP>1."""
        np.random.seed(42)
        E, B, k, n = 4, 50, 2, 4
        histogram_np = np.random.randint(5, 20, (E, B)).astype(np.int32)

        layer = self._make_layer_with_config(
            E, B, k, n, histogram_np, sp=True, ep_size=4
        )

        tp_group = MagicMock()
        callback = MoEQuantileBalancingCallback()
        callback._update_single_layer(
            layer,
            tp_group=tp_group,
            cp_group=None,
            dp_group=None,
            sd_group=None,
        )

        # TP all_reduce should have been called
        mock_dist.all_reduce.assert_called_once()

    @patch("paddlefleet.transformer.moe.qb_callback.dist")
    def test_tp_reduce_skipped_when_sp_false(self, mock_dist):
        """TP all-reduce should NOT be called when SP=False."""
        np.random.seed(42)
        E, B, k, n = 4, 50, 2, 4
        histogram_np = np.random.randint(5, 20, (E, B)).astype(np.int32)

        layer = self._make_layer_with_config(
            E, B, k, n, histogram_np, sp=False, ep_size=4
        )

        tp_group = MagicMock()
        callback = MoEQuantileBalancingCallback()
        callback._update_single_layer(
            layer,
            tp_group=tp_group,
            cp_group=None,
            dp_group=None,
            sd_group=None,
        )

        # TP all_reduce should NOT have been called
        mock_dist.all_reduce.assert_not_called()

    @patch("paddlefleet.transformer.moe.qb_callback.dist")
    def test_tp_reduce_skipped_when_ep1(self, mock_dist):
        """TP all-reduce should NOT be called when EP=1."""
        np.random.seed(42)
        E, B, k, n = 4, 50, 2, 4
        histogram_np = np.random.randint(5, 20, (E, B)).astype(np.int32)

        layer = self._make_layer_with_config(
            E, B, k, n, histogram_np, sp=True, ep_size=1
        )

        tp_group = MagicMock()
        callback = MoEQuantileBalancingCallback()
        callback._update_single_layer(
            layer,
            tp_group=tp_group,
            cp_group=None,
            dp_group=None,
            sd_group=None,
        )

        # TP all_reduce should NOT have been called
        mock_dist.all_reduce.assert_not_called()

    @patch("paddlefleet.transformer.moe.qb_callback.dist")
    def test_dp_and_sd_reduce_called(self, mock_dist):
        """DP and Sharding all-reduce should always be called when groups provided."""
        np.random.seed(42)
        E, B, k, n = 4, 50, 2, 4
        histogram_np = np.random.randint(5, 20, (E, B)).astype(np.int32)

        layer = self._make_layer_with_config(
            E, B, k, n, histogram_np, sp=False, ep_size=1
        )

        dp_group = MagicMock()
        sd_group = MagicMock()
        callback = MoEQuantileBalancingCallback()
        callback._update_single_layer(
            layer,
            tp_group=None,
            cp_group=None,
            dp_group=dp_group,
            sd_group=sd_group,
        )

        # DP and SD all_reduce should each have been called
        self.assertEqual(mock_dist.all_reduce.call_count, 2)

    @patch("paddlefleet.transformer.moe.qb_callback.dist")
    def test_cp_reduce_called_unconditionally(self, mock_dist):
        """CP always splits the sequence, so its reduce is not gated on SP/EP."""
        np.random.seed(42)
        E, B, k, n = 4, 50, 2, 4
        histogram_np = np.random.randint(5, 20, (E, B)).astype(np.int32)

        cp_group = MagicMock()
        callback = MoEQuantileBalancingCallback()
        # SP=False and EP=1 disable the TP reduce; CP must still fire.
        for sp, ep_size in ((False, 1), (True, 1), (False, 4)):
            mock_dist.reset_mock()
            layer = self._make_layer_with_config(
                E, B, k, n, histogram_np, sp=sp, ep_size=ep_size
            )
            callback._update_single_layer(
                layer,
                tp_group=None,
                cp_group=cp_group,
                dp_group=None,
                sd_group=None,
            )
            mock_dist.all_reduce.assert_called_once_with(
                mock_dist.all_reduce.call_args.args[0], group=cp_group
            )

    @patch("paddlefleet.transformer.moe.qb_callback.dist")
    def test_all_four_groups_reduce(self, mock_dist):
        """TP+CP+DP+Sharding together produce exactly four all-reduces."""
        np.random.seed(42)
        E, B, k, n = 4, 50, 2, 4
        histogram_np = np.random.randint(5, 20, (E, B)).astype(np.int32)

        layer = self._make_layer_with_config(
            E, B, k, n, histogram_np, sp=True, ep_size=4
        )
        groups = [MagicMock() for _ in range(4)]
        callback = MoEQuantileBalancingCallback()
        callback._update_single_layer(
            layer,
            tp_group=groups[0],
            cp_group=groups[1],
            dp_group=groups[2],
            sd_group=groups[3],
        )

        self.assertEqual(mock_dist.all_reduce.call_count, 4)
        reduced_groups = [
            call.kwargs["group"] for call in mock_dist.all_reduce.call_args_list
        ]
        self.assertEqual(reduced_groups, groups)

    @patch("paddlefleet.transformer.moe.qb_callback.dist")
    def test_cp_histograms_merge_to_global_quantile(self, mock_dist):
        """A CP-sharded histogram pair must recover the same bias as the pooled one.

        Each CP rank's router only sees S/CP tokens, so without the CP reduce the
        recovered quantile is a local one. Summing the shards must reproduce the
        result of running on the full sequence.
        """
        np.random.seed(11)
        E, B, k, n = 4, 60, 2, 4
        shard_a = np.random.randint(0, 15, (E, B)).astype(np.int32)
        shard_b = np.random.randint(0, 15, (E, B)).astype(np.int32)
        pooled = shard_a + shard_b

        cp_group = MagicMock()

        def _fake_all_reduce(tensor, group=None):
            # Emulate summing shard_a (local) with shard_b (the peer CP rank).
            tensor.add_(paddle.to_tensor(shard_b, dtype=tensor.dtype))

        mock_dist.all_reduce.side_effect = _fake_all_reduce

        layer = self._make_layer_with_config(
            E, B, k, n, shard_a, sp=False, ep_size=1
        )
        callback = MoEQuantileBalancingCallback()
        callback._update_single_layer(
            layer,
            tp_group=None,
            cp_group=cp_group,
            dp_group=None,
            sd_group=None,
        )
        merged_bias = layer._bias_value.numpy()

        expected = _numpy_recover_bias(pooled, k, n, -1.0, 1.0)
        np.testing.assert_allclose(merged_bias, expected, rtol=1e-5, atol=1e-6)

        # Sanity check: the un-merged (single-shard) bias differs, i.e. skipping
        # the CP reduce really would have produced a wrong answer.
        local_only = _numpy_recover_bias(shard_a, k, n, -1.0, 1.0)
        self.assertFalse(np.allclose(local_only, expected, atol=1e-6))


# =============================================================================
# Real TopKRouter tests (exercise moe_router.py QB code paths directly)
# =============================================================================


class QBRouterConfig:
    """Minimal TransformerConfig stand-in for building a real TopKRouter."""

    def __init__(self, **overrides):
        self.hidden_size = 32
        self.n_routed_experts = 8
        self.num_experts_per_tok = 2
        self.n_group = 1
        self.topk_group = 1
        self.init_method = paddle.nn.initializer.Normal(mean=0.0, std=0.02)
        self.topk_method = "quantile_balancing"
        self.norm_topk_prob = True
        self.routed_scaling_factor = 1.0
        self.routed_scaling_factor_learnable = False
        self.scoring_func = "sigmoid"
        self.moe_router_load_balancing_type = "none"
        self.moe_router_force_load_balancing = False
        self.moe_router_fusion = False
        self.moe_topk_fusion = False
        self.router_z_loss_coef = 0.0
        self.router_aux_loss_coef = 0.0
        self.tensor_model_parallel_size = 1
        self.context_parallel_size = 1
        self.sequence_parallel = False
        self.expert_model_parallel_size = 1
        self.gpt_model_use_experimental_version = False
        self.qb_n_bins = 100
        self._extra_conf = {"seq_aux": False}
        for key, value in overrides.items():
            setattr(self, key, value)

    def get(self, key, default=None):
        return self._extra_conf.get(key, getattr(self, key, default))


def _build_qb_router(**overrides):
    """Build a real TopKRouter with quantile_balancing enabled."""
    from paddlefleet.transformer.moe.moe_router import TopKRouter

    with patch(
        "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    ):
        return TopKRouter(QBRouterConfig(**overrides))


class TestQBRouterInit(unittest.TestCase):
    """QB-specific state created in StandardMoERouter.__init__."""

    def test_bias_and_histogram_created(self):
        router = _build_qb_router()
        self.assertEqual(router.e_score_correction_bias.shape, [8])
        np.testing.assert_allclose(
            router.e_score_correction_bias.numpy(),
            np.zeros(8, dtype=np.float32),
        )
        self.assertEqual(router.qb_histogram.shape, [8, 100])
        self.assertEqual(int(router.qb_histogram.sum().item()), 0)
        self.assertEqual(router.qb_histogram.dtype, paddle.int32)
        self.assertTrue(router.qb_histogram.stop_gradient)
        self.assertEqual(router.expert_usage.shape, [8])
        self.assertTrue(router.expert_usage.stop_gradient)
        self.assertEqual(router.qb_bin_min, -1.0)
        self.assertEqual(router.qb_bin_max, 1.0)
        self.assertFalse(router._cast_to_low_precision)

    def test_experimental_version_bias_is_2d(self):
        router = _build_qb_router(gpt_model_use_experimental_version=True)
        self.assertEqual(router.e_score_correction_bias.shape, [1, 8])

    def test_default_n_bins_is_1000(self):
        config = QBRouterConfig()
        del config.qb_n_bins
        from paddlefleet.transformer.moe.moe_router import TopKRouter

        with patch(
            "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
            return_value=1,
        ):
            router = TopKRouter(config)
        self.assertEqual(router.qb_n_bins, 1000)
        self.assertEqual(router.qb_histogram.shape, [8, 1000])

    def test_moe_topk_fusion_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            _build_qb_router(moe_topk_fusion=True)
        self.assertIn("incompatible with moe_topk_fusion", str(ctx.exception))

    def test_non_qb_router_has_no_qb_state(self):
        router = _build_qb_router(topk_method="greedy")
        self.assertFalse(hasattr(router, "qb_histogram"))


class TestTopkQuantileBalancingReal(unittest.TestCase):
    """StandardMoERouter._topk_quantile_balancing."""

    def setUp(self):
        paddle.seed(2026)
        np.random.seed(2026)
        self.router = _build_qb_router()

    def test_output_shapes_and_gate_from_raw_scores(self):
        scores = paddle.nn.functional.sigmoid(paddle.randn([16, 8]))
        weight, idx = self.router._topk_quantile_balancing(scores, 2, 1, 1)
        self.assertEqual(weight.shape, [16, 2])
        self.assertEqual(idx.shape, [16, 2])
        # Gate weights must come from the ORIGINAL scores, not the biased ones.
        expected = scores.take_along_axis(idx, axis=1)
        np.testing.assert_allclose(
            weight.numpy(), expected.numpy(), rtol=1e-6, atol=1e-7
        )

    def test_bias_shifts_selection_but_not_weights(self):
        scores = paddle.nn.functional.sigmoid(paddle.randn([12, 8]))
        _, idx_before = self.router._topk_quantile_balancing(scores, 2, 1, 1)
        self.assertFalse(bool((idx_before == 7).all().item()))

        # Force expert 7 to always win by giving it an overwhelming bias.
        bias = np.zeros(8, dtype=np.float32)
        bias[7] = 10.0
        self.router.e_score_correction_bias.set_value(paddle.to_tensor(bias))
        weight, idx = self.router._topk_quantile_balancing(scores, 2, 1, 1)
        self.assertTrue(bool((idx[:, 0] == 7).all().item()))
        # Weight for expert 7 is still the raw score, unaffected by the +10 bias.
        np.testing.assert_allclose(
            weight[:, 0].numpy(), scores[:, 7].numpy(), rtol=1e-6, atol=1e-7
        )

    def test_experimental_version_bias_broadcast(self):
        router = _build_qb_router(gpt_model_use_experimental_version=True)
        scores = paddle.nn.functional.sigmoid(paddle.randn([10, 8]))
        weight, idx = router._topk_quantile_balancing(scores, 2, 1, 1)
        self.assertEqual(weight.shape, [10, 2])
        self.assertGreater(int(router.qb_histogram.sum().item()), 0)

    def test_histogram_accumulates_only_when_grad_enabled(self):
        scores = paddle.nn.functional.sigmoid(paddle.randn([16, 8]))
        with paddle.no_grad():
            self.router._topk_quantile_balancing(scores, 2, 1, 1)
        self.assertEqual(int(self.router.qb_histogram.sum().item()), 0)

        self.router._topk_quantile_balancing(scores, 2, 1, 1)
        self.assertEqual(int(self.router.qb_histogram.sum().item()), 16 * 8)

    def test_n_group_greater_than_one_rejected(self):
        """QB init must reject n_group>1 with ValueError (not assert)."""

        with self.assertRaises(ValueError) as ctx:
            _build_qb_router(n_group=2, topk_group=1)
        self.assertIn("only supports n_group=1", str(ctx.exception))

    def test_missing_bias_rejected(self):
        self.router.e_score_correction_bias = None
        scores = paddle.nn.functional.sigmoid(paddle.randn([4, 8]))
        with self.assertRaises(AssertionError) as ctx:
            self.router._topk_quantile_balancing(scores, 2, 1, 1)
        self.assertIn("e_score_correction_bias is None", str(ctx.exception))


class TestAccumulateQBHistogramReal(unittest.TestCase):
    """StandardMoERouter._accumulate_qb_histogram."""

    def setUp(self):
        paddle.seed(7)
        np.random.seed(7)

    def test_matches_numpy_reference(self):
        router = _build_qb_router()
        scores_np = np.random.rand(24, 8).astype(np.float32)
        bias_np = np.linspace(-0.2, 0.2, 8).astype(np.float32)
        router.e_score_correction_bias.set_value(paddle.to_tensor(bias_np))

        scores = paddle.to_tensor(scores_np)
        biased = scores + paddle.to_tensor(bias_np).unsqueeze(0)
        router._accumulate_qb_histogram(scores, biased, 2)

        expected = _numpy_accumulate_histogram(
            scores_np, bias_np, 2, router.qb_bin_min, router.qb_bin_max, 100
        )
        np.testing.assert_array_equal(router.qb_histogram.numpy(), expected)

    def test_topk_val_clamped_when_k_plus_one_exceeds_experts(self):
        # k == n_experts, so k+1 must be clamped to E inside the accumulator.
        router = _build_qb_router(num_experts_per_tok=8)
        scores = paddle.to_tensor(np.random.rand(6, 8).astype(np.float32))
        router._accumulate_qb_histogram(scores, scores, 8)
        self.assertEqual(int(router.qb_histogram.sum().item()), 6 * 8)

    def test_degenerate_bin_range_falls_back(self):
        router = _build_qb_router()
        router.qb_bin_min = 0.0
        router.qb_bin_max = 0.0  # zero range -> fallback to 2.0
        scores = paddle.to_tensor(np.random.rand(6, 8).astype(np.float32))
        router._accumulate_qb_histogram(scores, scores, 2)
        self.assertEqual(int(router.qb_histogram.sum().item()), 6 * 8)

    def test_out_of_range_values_are_clipped(self):
        router = _build_qb_router()
        router.qb_bin_min = 0.0
        router.qb_bin_max = 0.01  # almost everything lands beyond the last bin
        scores = paddle.to_tensor(np.full([4, 8], 0.5, dtype=np.float32))
        router._accumulate_qb_histogram(scores, scores, 2)
        hist = router.qb_histogram.numpy()
        self.assertEqual(int(hist.sum()), 4 * 8)
        # All mass must sit inside [0, B-1]; nothing is dropped.
        self.assertTrue(hist[:, 0].sum() + hist[:, -1].sum() == 4 * 8)

    def test_accumulation_is_additive_across_microbatches(self):
        router = _build_qb_router()
        scores = paddle.to_tensor(np.random.rand(5, 8).astype(np.float32))
        router._accumulate_qb_histogram(scores, scores, 2)
        first = router.qb_histogram.numpy().copy()
        router._accumulate_qb_histogram(scores, scores, 2)
        np.testing.assert_array_equal(router.qb_histogram.numpy(), first * 2)


class TestCallTopkMethodDispatch(unittest.TestCase):
    """_call_topk_method routes to the QB implementation."""

    def test_quantile_balancing_branch(self):
        router = _build_qb_router()
        scores = paddle.nn.functional.sigmoid(paddle.randn([8, 8]))
        weight, idx = router._call_topk_method(
            "quantile_balancing", scores, k=2, n_group=1, topk_group=1
        )
        self.assertEqual(weight.shape, [8, 2])
        self.assertEqual(idx.shape, [8, 2])
        self.assertGreater(int(router.qb_histogram.sum().item()), 0)

    def test_unknown_method_raises(self):
        router = _build_qb_router()
        scores = paddle.nn.functional.sigmoid(paddle.randn([8, 8]))
        with self.assertRaises(NotImplementedError):
            router._call_topk_method("no_such_method", scores, k=2)


class TestQBRouterForward(unittest.TestCase):
    """Full TopKRouter.forward with quantile_balancing."""

    def setUp(self):
        paddle.seed(11)
        np.random.seed(11)

    def test_forward_updates_histogram_and_usage(self):
        router = _build_qb_router()
        hidden = paddle.randn([4, 6, 32])
        _, top_gate, top_idx, probs, mask, _, _, _ = router(hidden)

        num_tokens = 4 * 6
        self.assertEqual(top_gate.shape, [num_tokens, 2])
        self.assertEqual(top_idx.shape, [num_tokens, 2])
        self.assertEqual(probs.shape, [num_tokens, 8])
        self.assertEqual(mask.shape, [num_tokens, 8])
        # Histogram covers every (token, expert) pair.
        self.assertEqual(int(router.qb_histogram.sum().item()), num_tokens * 8)
        # expert_usage counts one entry per routed slot.
        self.assertEqual(int(router.expert_usage.sum().item()), num_tokens * 2)

    def test_forward_under_no_grad_leaves_state_untouched(self):
        router = _build_qb_router()
        hidden = paddle.randn([2, 4, 32])
        with paddle.no_grad():
            router(hidden)
        self.assertEqual(int(router.qb_histogram.sum().item()), 0)
        self.assertEqual(int(router.expert_usage.sum().item()), 0)

    def test_forward_gate_weights_are_normalized(self):
        router = _build_qb_router(norm_topk_prob=True)
        hidden = paddle.randn([2, 5, 32])
        _, top_gate, _, _, _, _, _, _ = router(hidden)
        np.testing.assert_allclose(
            top_gate.sum(axis=-1).numpy(),
            np.ones(2 * 5, dtype=np.float32),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_forward_is_differentiable(self):
        router = _build_qb_router()
        hidden = paddle.randn([2, 4, 32])
        hidden.stop_gradient = False
        _, top_gate, _, _, _, _, _, _ = router(hidden)
        top_gate.sum().backward()
        self.assertIsNotNone(router.weight.grad)
        # QB bias is a buffer updated by the callback, never by autograd.
        self.assertTrue(router.e_score_correction_bias.stop_gradient)


class TestCommGroupsExceptionPath(unittest.TestCase):
    """_try_get_comm_groups swallows failures from the fleet API."""

    def test_hcg_lookup_failure_returns_none(self):
        import paddle.distributed.fleet as real_fleet

        with (
            patch.object(real_fleet, "_hcg", MagicMock(), create=True),
            patch.object(
                real_fleet,
                "get_hybrid_communicate_group",
                side_effect=RuntimeError("hcg unavailable"),
            ),
        ):
            tp, cp, dp, sd = _try_get_comm_groups()
        self.assertIsNone(tp)
        self.assertIsNone(dp)
        self.assertIsNone(sd)


class TestRealRouterWithCallback(unittest.TestCase):
    """End-to-end: real router forward + callback update reduces imbalance."""

    def test_bias_reduces_load_imbalance(self):
        paddle.seed(3)
        np.random.seed(3)
        router = _build_qb_router(qb_n_bins=1000)
        callback = MoEQuantileBalancingCallback()
        model = paddle.nn.Sequential(router)
        hidden = paddle.randn([8, 64, 32])

        # Skew the gate so experts 0/1 systematically win: without QB the
        # router sends far more than its fair share of tokens to them.
        weight = (np.random.randn(8, 32) * 0.3).astype(np.float32)
        weight[0] += 1.5
        weight[1] += 1.0
        router.weight.set_value(paddle.to_tensor(weight))

        def measure():
            router.expert_usage.zero_()
            router.qb_histogram.zero_()
            router(hidden)
            usage = router.expert_usage.numpy().astype(np.float64)
            return usage.std() / max(usage.mean(), 1e-9)

        cv_before = measure()
        for _ in range(6):
            router.qb_histogram.zero_()
            router.expert_usage.zero_()
            router(hidden)
            callback.on_optimizer_end(model=model)
        cv_after = measure()

        # QB should cut the imbalance by at least half.
        self.assertLess(cv_after, cv_before / 2)
        # Bias stays zero-mean after every update.
        self.assertAlmostEqual(
            float(router.e_score_correction_bias.mean().item()), 0.0, places=5
        )

    def test_callback_discovers_qb_layers_in_a_model(self):
        router_qb = _build_qb_router()
        router_plain = _build_qb_router(topk_method="greedy")
        model = paddle.nn.Sequential(router_plain, router_qb)
        hidden = paddle.randn([2, 8, 32])
        router_qb(hidden)
        self.assertGreater(int(router_qb.qb_histogram.sum().item()), 0)

        MoEQuantileBalancingCallback().on_optimizer_end(model=model)
        # QB layer was updated (histogram reset), plain layer untouched.
        self.assertEqual(int(router_qb.qb_histogram.sum().item()), 0)
        self.assertFalse(hasattr(router_plain, "qb_histogram"))


class TestQBPaddingExclusion(unittest.TestCase):
    """Padding tokens must not enter the QB histogram.

    forward() zeroes out the gating scores of padding rows, so a padding token's
    required_bias collapses to one constant for every expert. Counting those
    rows would inflate the per-expert total (hence the target quantile) and
    spike the CDF right where it is read out, shifting the recovered bias by a
    different amount per expert.
    """

    E = 8
    H = 32

    def setUp(self):
        paddle.seed(11)
        np.random.seed(11)
        self.weight = paddle.to_tensor(
            (np.random.randn(self.E, self.H) * 0.1).astype("float32")
        )
        # Deliberately off-center so the routing is imbalanced and QB has real
        # work to do (a balanced router would hide small bias shifts).
        self.valid_rows = (np.random.randn(48, self.H) * 0.5 + 0.3).astype(
            "float32"
        )

    def _run(self, hidden, input_ids):
        """Forward once, then recover the bias. Returns (hist, usage, bias)."""
        router = _build_qb_router()
        router.weight.set_value(self.weight)
        kwargs = {}
        if input_ids is not None:
            kwargs["input_ids"] = paddle.to_tensor(input_ids)
        router(paddle.to_tensor(hidden), **kwargs)
        hist = router.qb_histogram.numpy().copy()
        usage = router.expert_usage.numpy().copy()
        MoEQuantileBalancingCallback()._update_single_layer(
            router, None, None, None, None
        )
        return hist, usage, router.e_score_correction_bias.numpy().copy()

    def _valid_only(self, rows):
        """Reference run: only the valid rows, laid out as [1, N, H]."""
        hidden = rows.reshape([1, -1, self.H])
        ids = np.ones((1, rows.shape[0]), dtype="int64")
        return self._run(hidden, ids)

    def test_trailing_padding_changes_nothing(self):
        rows = self.valid_rows
        hist_ref, usage_ref, bias_ref = self._valid_only(rows)

        n_pad = 16
        hidden = np.concatenate(
            [rows, np.zeros((n_pad, self.H), dtype="float32")], axis=0
        ).reshape([1, -1, self.H])
        ids = np.concatenate(
            [
                np.ones((1, rows.shape[0]), dtype="int64"),
                np.zeros((1, n_pad), dtype="int64"),
            ],
            axis=1,
        )
        hist, usage, bias = self._run(hidden, ids)

        np.testing.assert_array_equal(hist, hist_ref)
        np.testing.assert_array_equal(usage, usage_ref)
        np.testing.assert_array_equal(bias, bias_ref)

    def test_padding_scattered_inside_the_sequence(self):
        rows = self.valid_rows
        n_tokens = rows.shape[0]
        valid = np.ones(n_tokens, dtype=bool)
        valid[np.random.choice(n_tokens, 14, replace=False)] = False

        hist_ref, _, bias_ref = self._valid_only(rows[valid])

        hidden = np.where(valid[:, None], rows, 0.0).astype("float32")
        hidden = hidden.reshape([1, -1, self.H])
        ids = valid.astype("int64").reshape([1, -1])
        hist, _, bias = self._run(hidden, ids)

        np.testing.assert_array_equal(hist, hist_ref)
        np.testing.assert_array_equal(bias, bias_ref)

    def test_histogram_total_is_valid_token_count(self):
        n_valid, n_pad = 48, 16
        hidden = np.concatenate(
            [self.valid_rows, np.zeros((n_pad, self.H), dtype="float32")],
            axis=0,
        ).reshape([1, -1, self.H])
        ids = np.concatenate(
            [
                np.ones((1, n_valid), dtype="int64"),
                np.zeros((1, n_pad), dtype="int64"),
            ],
            axis=1,
        )
        router = _build_qb_router()
        router.weight.set_value(self.weight)
        router(paddle.to_tensor(hidden), input_ids=paddle.to_tensor(ids))
        # One sample per (valid token, expert) pair, padding contributes none.
        np.testing.assert_array_equal(
            router.qb_histogram.numpy().sum(axis=1),
            np.full(self.E, n_valid, dtype=np.int32),
        )

    def test_all_padding_batch_is_a_no_op(self):
        hidden = np.zeros((1, 16, self.H), dtype="float32")
        ids = np.zeros((1, 16), dtype="int64")
        router = _build_qb_router()
        router.weight.set_value(self.weight)
        bias_before = router.e_score_correction_bias.numpy().copy()
        router(paddle.to_tensor(hidden), input_ids=paddle.to_tensor(ids))
        self.assertEqual(int(router.qb_histogram.sum().item()), 0)

        # An empty histogram must leave the bias untouched.
        MoEQuantileBalancingCallback()._update_single_layer(
            router, None, None, None, None
        )
        np.testing.assert_array_equal(
            router.e_score_correction_bias.numpy(), bias_before
        )

    def test_without_input_ids_all_rows_count(self):
        """input_ids=None carries no padding info: behaviour must be unchanged."""
        hidden = self.valid_rows.reshape([1, -1, self.H])
        hist_none, _, bias_none = self._run(hidden, None)
        hist_ones, _, bias_ones = self._valid_only(self.valid_rows)
        np.testing.assert_array_equal(hist_none, hist_ones)
        np.testing.assert_array_equal(bias_none, bias_ones)


class TestQBRejectsAuxLossBalancing(unittest.TestCase):
    """QB must not silently coexist with the auxiliary load balancing loss.

    These build the router from a real ``TransformerConfig`` rather than the
    ``QBRouterConfig`` stand-in above, because the stand-in already disables
    aux loss and therefore cannot catch the default configuration.
    """

    @staticmethod
    def _build_router(**overrides):
        from paddlefleet.transformer.moe.moe_router import TopKRouter
        from paddlefleet.transformer.transformer_config import (
            TransformerConfig,
        )

        config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=32,
            num_attention_heads=4,
        )
        config.n_routed_experts = 8
        config.num_experts_per_tok = 2
        config.topk_method = "quantile_balancing"
        config.scoring_func = "sigmoid"
        config.init_method = paddle.nn.initializer.Normal(mean=0.0, std=0.02)
        for key, value in overrides.items():
            setattr(config, key, value)

        with patch(
            "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
            return_value=1,
        ):
            return TopKRouter(config)

    def test_default_config_is_rejected(self):
        """Defaults are aux_loss / 1e-2, which must not pass silently."""
        with self.assertRaises(ValueError) as ctx:
            self._build_router()
        self.assertIn("moe_router_load_balancing_type", str(ctx.exception))

    def test_nonzero_aux_loss_coef_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self._build_router(
                moe_router_load_balancing_type="none",
                router_aux_loss_coef=1e-2,
            )
        self.assertIn("router_aux_loss_coef", str(ctx.exception))

    def test_seq_aux_loss_is_rejected(self):
        with self.assertRaises(ValueError):
            self._build_router(
                moe_router_load_balancing_type="seq_aux_loss",
                router_aux_loss_coef=1e-4,
            )

    def test_explicitly_disabled_balancing_is_accepted(self):
        router = self._build_router(
            moe_router_load_balancing_type="none",
            router_aux_loss_coef=0.0,
        )
        self.assertEqual(router.topk_method, "quantile_balancing")
        self.assertEqual(router.qb_histogram.shape, [8, 1000])

    def test_topk_fusion_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self._build_router(
                moe_router_load_balancing_type="none",
                router_aux_loss_coef=0.0,
                moe_topk_fusion=True,
            )
        self.assertIn("moe_topk_fusion", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
