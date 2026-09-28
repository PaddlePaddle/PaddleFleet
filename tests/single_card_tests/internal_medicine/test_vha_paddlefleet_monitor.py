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

import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

importlib.import_module("_backend_env").skip_unless_backend("paddlefleet")

try:
    paddle = importlib.import_module("paddle")
    nn = importlib.import_module("paddle.nn")
except (
    Exception
) as exc:  # pragma: no cover - depends on optional backend install
    raise unittest.SkipTest(f"paddle backend unavailable: {exc}") from exc

vha_metrics = importlib.import_module(
    "paddlefleet.internal_medicine.backends.paddlefleet.vha_metrics"
)
vha_monitor = importlib.import_module(
    "paddlefleet.internal_medicine.backends.paddlefleet.vha_monitor"
)
_training_logs_mod = importlib.import_module(
    "paddlefleet.internal_medicine.core.training_logs"
)
training_logs = _training_logs_mod.training_logs
TrainingLogs = _training_logs_mod.TrainingLogs

PaddleVHAHealthMonitor = vha_monitor.PaddleVHAHealthMonitor


def _factor(rows, rank_col_values):
    """Build an ``[nh, 1]`` factor from a list of scalars."""
    return paddle.to_tensor(
        [[v] for v in rank_col_values], dtype="float32"
    ).reshape([rows, 1])


class FakeVHAAttention(nn.Layer):
    """Stand-in for a VHA attention module: same postmix math, same attributes."""

    def __init__(
        self,
        num_heads,
        v_head_dim,
        u,
        v,
        sparse_u=None,
        sparse_v=None,
        premix=None,
    ):
        super().__init__()
        self.use_vha_postmix = True
        self.num_heads = num_heads
        self.v_head_dim = v_head_dim
        self.vha_postmix_U = u
        self.vha_postmix_V = v
        self.sparse_vha_postmix_U = sparse_u
        self.sparse_vha_postmix_V = sparse_v
        self.use_vha_premix = premix is not None
        self.vha_premix_weight = premix
        self.calls = 0

    def _apply_vha_postmix(self, attn_out, U=None, V=None):
        if U is None:
            U = self.vha_postmix_U
        if V is None:
            V = self.vha_postmix_V
        self.calls += 1
        b, sq = attn_out.shape[0], attn_out.shape[1]
        mixed = attn_out.reshape([b, sq, self.num_heads, self.v_head_dim])
        z = paddle.einsum("bthd,hr->btrd", mixed, U)
        delta = paddle.einsum("btrd,hr->bthd", z, V)
        return (mixed + delta).reshape(
            [b, sq, self.num_heads * self.v_head_dim]
        )


class FakeVHAAttentionHeadSpaceInput(FakeVHAAttention):
    """DSv4 hybrid on the fused inverse-RoPE path: 4D input, flat 3D output.

    ``dsv4_hybrid_attention._full_attn_forward`` reshapes ``core_attn_out`` to
    ``[b, sq, nh, v_head_dim]`` for the inverse RoPE and only the *unfused*
    branch reshapes it back, so with ``apply_rope_fusion=True`` the postmix call
    receives the unflattened layout while still returning the flat one.
    """

    def _apply_vha_postmix(self, attn_out, U=None, V=None):
        return super()._apply_vha_postmix(
            attn_out.reshape([attn_out.shape[0], attn_out.shape[1], -1]), U, V
        )


class FakeLayer(nn.Layer):
    def __init__(self, attn):
        super().__init__()
        self.self_attn = attn


def _model(layers):
    return SimpleNamespace(decoder=SimpleNamespace(layers=nn.LayerList(layers)))


class VHAMetricsTest(unittest.TestCase):
    def test_zero_v_is_identity_operator(self):
        # V is zero-initialised in PaddleFleet, so postmix must start as exactly
        # the identity: no spectral energy, no off-diagonal mixing.
        u = _factor(4, [1.0, 0.5, -0.25, 0.0])
        v = paddle.zeros([4, 1], dtype="float32")
        stats = vha_metrics.postmix_operator_stats(u, v)
        self.assertAlmostEqual(
            float(stats["postmix_uv_sigma_max"]), 0.0, places=6
        )
        self.assertAlmostEqual(
            float(stats["postmix_offdiag_ratio"]), 0.0, places=6
        )
        self.assertAlmostEqual(float(stats["postmix_v_fro"]), 0.0, places=6)

    def test_rank_one_offdiagonal_operator(self):
        # A = U Vᵀ has a single 1.0 at [0, 1]: sigma_max 1, eff_rank 1, and
        # ‖A_offdiag‖_F / ‖I + A‖_F = 1 / sqrt(4 + 1).
        u = _factor(4, [1.0, 0.0, 0.0, 0.0])
        v = _factor(4, [0.0, 1.0, 0.0, 0.0])
        stats = vha_metrics.postmix_operator_stats(u, v)
        self.assertAlmostEqual(
            float(stats["postmix_uv_sigma_max"]), 1.0, places=5
        )
        self.assertAlmostEqual(
            float(stats["postmix_uv_eff_rank"]), 1.0, places=5
        )
        self.assertAlmostEqual(
            float(stats["postmix_offdiag_ratio"]), 1.0 / 5.0**0.5, places=5
        )

    def test_diagonal_operator_has_no_offdiagonal_energy(self):
        # A per-head rescaling (diagonal A) must read as zero cross-head mixing.
        u = _factor(4, [1.0, 0.0, 0.0, 0.0])
        v = _factor(4, [1.0, 0.0, 0.0, 0.0])
        stats = vha_metrics.postmix_operator_stats(u, v)
        self.assertAlmostEqual(
            float(stats["postmix_offdiag_ratio"]), 0.0, places=6
        )
        self.assertAlmostEqual(
            float(stats["postmix_uv_sigma_max"]), 1.0, places=5
        )

    def test_grouped_factors_stay_block_diagonal(self):
        # Grouped postmix mixes heads only inside a group; a group-local diagonal
        # operator must not leak into cross-group off-diagonal energy.
        u = paddle.to_tensor(
            [[[1.0], [0.0]], [[1.0], [0.0]]], dtype="float32"
        )  # [2, 2, 1]
        stats = vha_metrics.postmix_operator_stats(u, u)
        self.assertAlmostEqual(
            float(stats["postmix_offdiag_ratio"]), 0.0, places=6
        )

    def test_delta_stats_identity_and_scaling(self):
        attn_out = paddle.to_tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype="float32")
        identity = vha_metrics.postmix_delta_stats(attn_out, attn_out)
        self.assertAlmostEqual(
            float(identity["postmix_delta_rel_mean"]), 0.0, places=6
        )
        self.assertAlmostEqual(
            float(identity["postmix_delta_rel_max"]), 0.0, places=6
        )
        self.assertAlmostEqual(
            float(identity["postmix_amax_gain_max"]), 1.0, places=6
        )

        doubled = vha_metrics.postmix_delta_stats(attn_out, attn_out * 2.0)
        # delta == mixed, so the relative correction is exactly 1.
        self.assertAlmostEqual(
            float(doubled["postmix_delta_rel_mean"]), 1.0, places=5
        )
        self.assertAlmostEqual(
            float(doubled["postmix_amax_gain_max"]), 2.0, places=5
        )

    def test_delta_stats_accepts_head_space_input(self):
        # DSv4 hybrid's fused inverse-RoPE path hands postmix a 4D
        # [b, sq, nh, v_head_dim] input while the return stays flat 3D. Both
        # sides must fold to the flat width, otherwise the subtraction sees
        # [sq, nh*d] against [sq*nh, d] and raises a broadcast error.
        flat = paddle.to_tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype="float32")
        head_space = flat.reshape([1, 1, 2, 2])
        stats = vha_metrics.postmix_delta_stats(head_space, flat * 2.0)
        reference = vha_metrics.postmix_delta_stats(flat, flat * 2.0)
        for key in reference:
            self.assertAlmostEqual(
                float(stats[key]), float(reference[key]), places=6, msg=key
            )

    def test_head_output_stats(self):
        # Two heads of dim 2: norms 1 and 2, pointing the same direction.
        out = paddle.to_tensor([[[1.0, 0.0, 2.0, 0.0]]], dtype="float32")
        stats = vha_metrics.head_output_stats(out, num_heads=2)
        self.assertAlmostEqual(float(stats["head_out_norm_max"]), 2.0, places=5)
        self.assertAlmostEqual(float(stats["head_out_norm_min"]), 1.0, places=5)
        self.assertAlmostEqual(
            float(stats["postmix_head_cos_mean"]), 1.0, places=5
        )

    def test_premix_near_identity_weight_has_zero_deviation(self):
        # PaddleFleet initialises premix as I + 0.1/√d · N(0,1) per KV group, so
        # the diagnostic is deviation from identity, not from orthogonality.
        w = paddle.eye(3, dtype="float32").unsqueeze(0)
        stats = vha_metrics.premix_stats(w)
        self.assertAlmostEqual(
            float(stats["premix_identity_dev"]), 0.0, places=5
        )
        self.assertAlmostEqual(float(stats["premix_sigma_max"]), 1.0, places=5)
        self.assertNotIn("premix_group_div_ratio", stats)  # single group

    def test_premix_shared_group_deviation_gives_ratio_zero(self):
        """Every KV group learned the same transform -> no group specialization."""
        deviation = paddle.to_tensor(
            [[0.0, 0.3, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype="float32"
        )
        w = (
            (paddle.eye(3, dtype="float32") + deviation)
            .unsqueeze(0)
            .expand([2, 3, 3])
        )
        stats = vha_metrics.premix_stats(w)
        self.assertAlmostEqual(
            float(stats["premix_group_div_ratio"]), 0.0, places=5
        )

    def test_premix_independent_group_deviations_give_ratio_two(self):
        """Disjoint per-group deviations -> the value expected under independence."""
        eye = paddle.eye(3, dtype="float32")
        d0 = paddle.zeros([3, 3], dtype="float32")
        d0[0, 1] = 1.0
        d1 = paddle.zeros([3, 3], dtype="float32")
        d1[1, 0] = 1.0
        w = paddle.stack([eye + d0, eye + d1])
        stats = vha_metrics.premix_stats(w)
        self.assertAlmostEqual(
            float(stats["premix_group_div_ratio"]), 2.0, places=5
        )

    def test_premix_non_square_weight_reports_orthogonality(self):
        """q_head_dim != head_dim keeps the scaled-orthogonal init, so track that."""
        w = paddle.eye(4, dtype="float32")[:, :3].unsqueeze(0)  # [1, 4, 3]
        stats = vha_metrics.premix_stats(w)
        self.assertIn("premix_orth_dev", stats)
        self.assertNotIn("premix_identity_dev", stats)
        self.assertAlmostEqual(float(stats["premix_orth_dev"]), 0.0, places=5)

    def test_premix_metric_names_match_produced_keys(self):
        for w in (
            paddle.eye(3, dtype="float32").unsqueeze(0),
            paddle.eye(3, dtype="float32").unsqueeze(0).expand([2, 3, 3]),
            paddle.eye(4, dtype="float32")[:, :3].unsqueeze(0),
        ):
            self.assertEqual(
                set(vha_metrics.premix_metric_names(w)),
                set(vha_metrics.premix_stats(w)),
            )


class VHAMonitorTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    @staticmethod
    def _attn_out(num_heads=4, v_head_dim=2):
        return paddle.ones([1, 3, num_heads * v_head_dim], dtype="float32")

    def test_identity_postmix_records_zero_delta(self):
        u = _factor(4, [1.0, 0.5, -0.25, 0.0])
        v = paddle.zeros([4, 1], dtype="float32")
        attn = FakeVHAAttention(4, 2, u, v)
        monitor = PaddleVHAHealthMonitor(log_per_layer=True, log_global=True)
        monitor.register_hooks(_model([FakeLayer(attn)]))

        attn._apply_vha_postmix(self._attn_out())
        monitor.step()

        latest = training_logs.get_latest(prefix="vha_health")
        self.assertAlmostEqual(
            latest["vha_health/layer_0/main_postmix_delta_rel_max"],
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            latest["vha_health/global_main_postmix_delta_rel_max"],
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            latest["vha_health/layer_0/main_postmix_amax_gain_max"],
            1.0,
            places=6,
        )
        monitor.remove_hooks()

    def test_active_postmix_records_nonzero_delta(self):
        u = _factor(4, [1.0, 0.0, 0.0, 0.0])
        v = _factor(4, [0.0, 1.0, 0.0, 0.0])
        attn = FakeVHAAttention(4, 2, u, v)
        monitor = PaddleVHAHealthMonitor()
        monitor.register_hooks(_model([FakeLayer(attn)]))

        attn._apply_vha_postmix(self._attn_out())
        monitor.step()

        latest = training_logs.get_latest(prefix="vha_health")
        self.assertGreater(
            latest["vha_health/layer_0/main_postmix_delta_rel_max"], 0.0
        )
        self.assertAlmostEqual(
            latest["vha_health/layer_0/main_postmix_uv_sigma_max"],
            1.0,
            places=5,
        )
        monitor.remove_hooks()

    def test_head_space_input_still_records(self):
        # Regression: on DSv4 hybrid + apply_rope_fusion the postmix input stays
        # 4D, which used to make every layer throw inside the wrapper and record
        # nothing at all.
        u = _factor(4, [1.0, 0.0, 0.0, 0.0])
        v = _factor(4, [0.0, 1.0, 0.0, 0.0])
        attn = FakeVHAAttentionHeadSpaceInput(4, 2, u, v)
        monitor = PaddleVHAHealthMonitor()
        monitor.register_hooks(_model([FakeLayer(attn)]))

        attn._apply_vha_postmix(self._attn_out().reshape([1, 3, 4, 2]))
        monitor.step()
        head_space = training_logs.get_latest(prefix="vha_health")
        monitor.remove_hooks()

        training_logs.reset()
        flat_attn = FakeVHAAttention(4, 2, u, v)
        flat_monitor = PaddleVHAHealthMonitor()
        flat_monitor.register_hooks(_model([FakeLayer(flat_attn)]))
        flat_attn._apply_vha_postmix(self._attn_out())
        flat_monitor.step()
        flat = training_logs.get_latest(prefix="vha_health")
        flat_monitor.remove_hooks()

        self.assertEqual(set(head_space), set(flat))
        self.assertGreater(
            head_space["vha_health/layer_0/main_postmix_delta_rel_max"], 0.0
        )
        for key in flat:
            self.assertAlmostEqual(
                head_space[key], flat[key], places=5, msg=key
            )

    def test_sparse_branch_keys_do_not_collide(self):
        u = _factor(4, [1.0, 0.0, 0.0, 0.0])
        v = paddle.zeros([4, 1], dtype="float32")
        sparse_u = _factor(4, [1.0, 0.0, 0.0, 0.0])
        sparse_v = _factor(4, [0.0, 1.0, 0.0, 0.0])
        attn = FakeVHAAttention(
            4, 2, u, v, sparse_u=sparse_u, sparse_v=sparse_v
        )
        monitor = PaddleVHAHealthMonitor()
        monitor.register_hooks(_model([FakeLayer(attn)]))

        attn._apply_vha_postmix(self._attn_out())
        attn._apply_vha_postmix(self._attn_out(), sparse_u, sparse_v)
        monitor.step()

        latest = training_logs.get_latest(prefix="vha_health")
        self.assertAlmostEqual(
            latest["vha_health/layer_0/main_postmix_uv_sigma_max"],
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            latest["vha_health/layer_0/sparse_postmix_uv_sigma_max"],
            1.0,
            places=5,
        )
        monitor.remove_hooks()

    def test_repeated_call_leaves_aggregates_unchanged(self):
        # Selective recompute replays _apply_vha_postmix with identical inputs;
        # the extra record must not move mean/max/min.
        u = _factor(4, [1.0, 0.0, 0.0, 0.0])
        v = _factor(4, [0.0, 1.0, 0.0, 0.0])
        attn = FakeVHAAttention(4, 2, u, v)
        monitor = PaddleVHAHealthMonitor()
        monitor.register_hooks(_model([FakeLayer(attn)]))

        attn._apply_vha_postmix(self._attn_out())
        attn._apply_vha_postmix(self._attn_out())
        monitor.step()
        replayed = training_logs.get_latest(prefix="vha_health")

        training_logs.reset()
        monitor.remove_hooks()
        attn2 = FakeVHAAttention(4, 2, u, v)
        monitor2 = PaddleVHAHealthMonitor()
        monitor2.register_hooks(_model([FakeLayer(attn2)]))
        attn2._apply_vha_postmix(self._attn_out())
        monitor2.step()
        once = training_logs.get_latest(prefix="vha_health")
        monitor2.remove_hooks()

        self.assertEqual(set(replayed), set(once))
        for key in once:
            self.assertAlmostEqual(replayed[key], once[key], places=5, msg=key)

    def test_remove_hooks_restores_original_method(self):
        u = _factor(4, [1.0, 0.0, 0.0, 0.0])
        v = paddle.zeros([4, 1], dtype="float32")
        attn = FakeVHAAttention(4, 2, u, v)
        monitor = PaddleVHAHealthMonitor()
        monitor.register_hooks(_model([FakeLayer(attn)]))
        monitor.remove_hooks()

        attn._apply_vha_postmix(self._attn_out())
        monitor.step()
        self.assertEqual(training_logs.get_latest(prefix="vha_health"), {})

    def test_no_vha_layers_is_a_noop(self):
        plain = FakeLayer(nn.Linear(4, 4))
        monitor = PaddleVHAHealthMonitor()
        monitor.register_hooks(_model([plain]))
        self.assertEqual(monitor._wrapped, [])

    def test_extremum_keys_are_classified_as_extrema(self):
        # Guards the naming contract: training_logs must reach the same verdict
        # as the monitor for every key, prefixes included.
        for key in (
            "vha_health/layer_0/hca_main_postmix_delta_rel_max",
            "vha_health/layer_0/hca_sparse_postmix_uv_sigma_max",
            "vha_health/global_hca_main_head_out_norm_max",
            "vha_health/layer_0/premix_sigma_max",
        ):
            self.assertTrue(TrainingLogs._is_max_metric(key), key)
        for key in (
            "vha_health/layer_0/hca_main_head_out_norm_min",
            "vha_health/global_hca_main_head_out_norm_min",
        ):
            self.assertTrue(TrainingLogs._is_min_metric(key), key)
        for key in (
            "vha_health/layer_0/hca_main_postmix_delta_rel_mean",
            "vha_health/layer_0/hca_main_postmix_offdiag_ratio",
            "vha_health/layer_0/hca_main_head_out_norm_std",
        ):
            self.assertFalse(TrainingLogs._is_max_metric(key), key)
            self.assertFalse(TrainingLogs._is_min_metric(key), key)


if __name__ == "__main__":
    unittest.main()
