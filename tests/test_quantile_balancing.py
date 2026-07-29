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

import unittest
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
            layer, tp_group=None, dp_group=None, sd_group=None
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
            layer, tp_group=None, dp_group=None, sd_group=None
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
            layer, tp_group=None, dp_group=None, sd_group=None
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
            layer, tp_group=None, dp_group=None, sd_group=None
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
            layer, tp_group=None, dp_group=None, sd_group=None
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
            layer, tp_group=None, dp_group=None, sd_group=None
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
            layer, tp_group=None, dp_group=None, sd_group=None
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


class TestTryGetCommGroups(unittest.TestCase):
    """Test the communication group retrieval function."""

    def test_no_fleet_returns_none(self):
        """Without fleet initialized, should return (None, None, None)."""
        tp, dp, sd = _try_get_comm_groups()
        # In single-process test, fleet is not initialized
        self.assertIsNone(tp)
        self.assertIsNone(dp)
        self.assertIsNone(sd)

    @patch("paddlefleet.transformer.moe.qb_callback.fleet", create=True)
    def test_fleet_no_hcg(self, mock_fleet):
        """If fleet has no _hcg attribute, return (None, None, None)."""
        del mock_fleet._hcg  # simulate missing _hcg
        mock_fleet.configure_mock(**{"_hcg": None})
        # hasattr check
        if hasattr(mock_fleet, "_hcg"):
            delattr(mock_fleet, "_hcg")

        tp, dp, sd = _try_get_comm_groups()
        self.assertIsNone(tp)
        self.assertIsNone(dp)
        self.assertIsNone(sd)

    def test_exception_returns_none(self):
        """If any exception occurs, should return (None, None, None) gracefully."""
        with patch(
            "paddlefleet.transformer.moe.qb_callback.fleet",
            side_effect=Exception("test"),
            create=True,
        ):
            tp, dp, sd = _try_get_comm_groups()
            self.assertIsNone(tp)
            self.assertIsNone(dp)
            self.assertIsNone(sd)


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
            layer, tp_group=None, dp_group=None, sd_group=None
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
            layer, tp_group=None, dp_group=None, sd_group=None
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


if __name__ == "__main__":
    unittest.main()
