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
import re
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

mhc_metrics = importlib.import_module(
    "paddlefleet.internal_medicine.backends.paddlefleet.mhc_metrics"
)
mhc_monitor = importlib.import_module(
    "paddlefleet.internal_medicine.backends.paddlefleet.mhc_monitor"
)
training_logs = importlib.import_module(
    "paddlefleet.internal_medicine.core.training_logs"
).training_logs

PaddleMHCHealthMonitor = mhc_monitor.PaddleMHCHealthMonitor


class FakeHC(nn.Layer):
    """Stand-in for HyperConnectionModule exposing compute_mappings."""

    def __init__(
        self, n, h_pre, h_post, h_res, h_res_logits=None, alpha=0.1, bias=None
    ):
        super().__init__()
        self.n = n
        self._h_pre = h_pre
        self._h_post = h_post
        self._h_res = h_res
        # Raw pre-Sinkhorn mixing logits. Zeros -> uniform rows -> every column
        # sum is exactly 1, i.e. the healthy case.
        if h_res_logits is None:
            h_res_logits = paddle.zeros([*h_pre.shape[:-1], n * n])
        self._h_res_logits = h_res_logits
        # Eq. (7) parameters, same layout as the real module: three scalar gating
        # factors plus one [n^2 + 2n] bias whose slices are b_pre / b_post / b_res.
        self.alpha_pre = paddle.full([1], alpha)
        self.alpha_post = paddle.full([1], alpha)
        self.alpha_res = paddle.full([1], alpha)
        self.bias = paddle.zeros([n * n + 2 * n]) if bias is None else bias

    def _compute_h(self, proj, r):
        return self._h_pre, self._h_post, self._h_res_logits

    def compute_mappings(self, x):
        # Mirrors the real module: _compute_h yields the raw logits, the Sinkhorn
        # projection then turns them into h_res. Going through self._compute_h is
        # what lets the monitor's instance-attribute shadowing see the logits.
        self._compute_h(None, None)
        return self._h_pre, self._h_post, self._h_res

    def fused_h_res_h_post_bda(
        self,
        h_res,
        original_residual,
        h_post,
        layer_output_with_bias,
        dropout_prob=0.0,
        training=True,
        fused=False,
    ):
        x, bias = layer_output_with_bias
        n = self.n
        leading = original_residual.shape[:-1]
        C = original_residual.shape[-1] // n
        mixed = paddle.bmm(
            h_res.reshape([-1, n, n]).transpose([0, 2, 1]),
            original_residual.reshape([-1, n, C]),
        ).reshape([*leading, n, C])
        xb = x if bias is None else x + bias
        return (h_post.unsqueeze(-1) * xb.unsqueeze(-2) + mixed).reshape(
            [*leading, n * C]
        )

    def forward(self, hidden_states):  # pragma: no cover - unused
        return hidden_states, self._h_res, self._h_post


class FakeLayer(nn.Layer):
    """Stand-in for HyperConnectionTransformerLayer with the two hc modules."""

    def __init__(self, attn, mlp):
        super().__init__()
        self.self_attention_hyper_connection = attn
        self.mlp_hyper_connection = mlp


def _mhc_model(layers):
    return SimpleNamespace(decoder=SimpleNamespace(layers=nn.LayerList(layers)))


class MHCMetricsTest(unittest.TestCase):
    def test_amax_gain_row_vs_col(self):
        # [[1,2],[3,4]] -> row sums [3,7] -> max abs 7 (fwd); col sums [4,6] -> 6 (bwd).
        m = paddle.to_tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype="float32")
        self.assertAlmostEqual(
            float(mhc_metrics.amax_gain(m, axis=-1)), 7.0, places=5
        )
        self.assertAlmostEqual(
            float(mhc_metrics.amax_gain(m, axis=-2)), 6.0, places=5
        )

    def test_doubly_stochastic_gain_is_one(self):
        n = 4
        ident = paddle.eye(n).reshape([1, 1, n, n])
        self.assertAlmostEqual(
            float(mhc_metrics.amax_gain(ident, axis=-1)), 1.0, places=5
        )
        self.assertAlmostEqual(
            float(mhc_metrics.amax_gain(ident, axis=-2)), 1.0, places=5
        )

    def test_gate_stats(self):
        h = paddle.to_tensor([[[0.0, 1.0, 2.0, 3.0]]], dtype="float32")
        mean, std = mhc_metrics.gate_stats(h)
        self.assertAlmostEqual(float(mean), 1.5, places=5)
        self.assertAlmostEqual(float(std), float(h.std()), places=5)

    def test_stream_concentration_bounds(self):
        # Uniform across streams -> 1 (lower bound); all mass on one stream -> n.
        uniform = paddle.full([3, 4], 0.7)
        one_hot = paddle.to_tensor([[1.0, 0.0, 0.0, 0.0]] * 3, dtype="float32")
        self.assertAlmostEqual(
            float(
                mhc_metrics.h_post_structure_stats(uniform)[
                    "h_post_stream_concentration"
                ]
            ),
            1.0,
            places=5,
        )
        self.assertAlmostEqual(
            float(
                mhc_metrics.h_post_structure_stats(one_hot)[
                    "h_post_stream_concentration"
                ]
            ),
            4.0,
            places=5,
        )

    def test_token_std_zero_when_gate_is_constant(self):
        const = paddle.full([8, 4], 1.0)
        varying = paddle.to_tensor([[0.2] * 4, [1.8] * 4], dtype="float32")
        self.assertAlmostEqual(
            float(
                mhc_metrics.h_post_structure_stats(const)["h_post_token_std"]
            ),
            0.0,
            places=6,
        )
        self.assertGreater(
            float(
                mhc_metrics.h_post_structure_stats(varying)["h_post_token_std"]
            ),
            0.5,
        )

    def test_branch_residual_share_against_explicit_terms(self):
        # h_res = identity -> residual term is exactly the incoming streams, so
        # the share reduces to b / (b + ‖x_l‖_F) with b = ‖h_post‖₂·‖x‖₂.
        n, c = 2, 3
        h_res = paddle.eye(n).reshape([1, n, n])
        streams = paddle.to_tensor(
            [[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]], dtype="float32"
        )
        h_post = paddle.to_tensor([[3.0, 4.0]], dtype="float32")  # ‖·‖₂ = 5
        x = paddle.to_tensor([[0.0, 0.0, 2.0]], dtype="float32")  # ‖·‖₂ = 2
        stats = mhc_metrics.branch_residual_share(
            h_res, streams.reshape([1, n * c]), h_post, x
        )
        branch = 5.0 * 2.0
        residual = float(paddle.sqrt(paddle.to_tensor(5.0)))  # ‖streams‖_F = √5
        expected = branch / (branch + residual)
        self.assertAlmostEqual(
            float(stats["branch_residual_share"]), expected, places=4
        )
        self.assertAlmostEqual(
            float(stats["branch_residual_share_max"]), expected, places=4
        )

    def test_branch_residual_share_zero_gate_kills_branch(self):
        n, c = 4, 5
        h_res = paddle.eye(n).reshape([1, n, n])
        streams = paddle.randn([1, n * c])
        x = paddle.randn([1, c])
        stats = mhc_metrics.branch_residual_share(
            h_res, streams, paddle.zeros([1, n]), x
        )
        self.assertAlmostEqual(
            float(stats["branch_residual_share"]), 0.0, places=6
        )

    def test_branch_residual_share_zero_residual_token_stays_bounded(self):
        # An all-zero token (padding, or an MTP layer's shifted slot) drives the
        # residual norm to 0. The raw ratio hit the epsilon floor and returned
        # ~1e6, hijacking the token mean; the bounded share must read 1.0 and
        # leave the mean dominated by the healthy token.
        n, c = 2, 3
        h_res = paddle.eye(n).reshape([1, n, n]).expand([2, n, n])
        streams = paddle.concat(
            [paddle.ones([1, n * c]), paddle.zeros([1, n * c])]
        )
        h_post = paddle.ones([2, n])
        x = paddle.ones([2, c])
        stats = mhc_metrics.branch_residual_share(h_res, streams, h_post, x)
        self.assertAlmostEqual(
            float(stats["branch_residual_share_max"]), 1.0, places=6
        )
        self.assertLessEqual(float(stats["branch_residual_share"]), 1.0)
        self.assertGreater(float(stats["branch_residual_share"]), 0.5)

    def test_branch_residual_share_all_zero_token_is_zero(self):
        # Both terms zero must not divide by zero; the eps guard makes it 0.
        n, c = 2, 3
        stats = mhc_metrics.branch_residual_share(
            paddle.zeros([1, n, n]),
            paddle.zeros([1, n * c]),
            paddle.zeros([1, n]),
            paddle.zeros([1, c]),
        )
        self.assertAlmostEqual(
            float(stats["branch_residual_share"]), 0.0, places=6
        )

    def test_logits_extrema_preserve_signed_raw_range(self):
        logits = paddle.to_tensor([[-200.0, -18.0, 0.0, 7.5]], dtype="float32")
        stats = mhc_metrics.h_res_logits_extrema(logits)
        self.assertAlmostEqual(
            float(stats["h_res_logits_min"]), -200.0, places=6
        )
        self.assertAlmostEqual(float(stats["h_res_logits_max"]), 7.5, places=6)

    def test_gate_logits_extrema_matches_explicit_formula(self):
        # pre  = r * proj[:n]     * alpha_pre  + b_pre  = 2*[1,-2]*0.5  + 0.25
        # post = r * proj[n:2n]   * alpha_post + b_post = 2*[3,-4]*-1.0 + 1.0
        n = 2
        proj = paddle.to_tensor(
            [[[1.0, -2.0, 3.0, -4.0, 0.0, 0.0, 0.0, 0.0]]], dtype="float32"
        )
        r = paddle.to_tensor([[[2.0]]], dtype="float32")
        bias = paddle.to_tensor(
            [0.25, 0.25, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype="float32"
        )
        stats = mhc_metrics.gate_logits_extrema(
            proj, r, paddle.full([1], 0.5), paddle.full([1], -1.0), bias, n
        )
        self.assertAlmostEqual(
            float(stats["h_pre_logits_min"]), -1.75, places=5
        )
        self.assertAlmostEqual(float(stats["h_pre_logits_max"]), 1.25, places=5)
        self.assertAlmostEqual(
            float(stats["h_post_logits_min"]), -5.0, places=5
        )
        self.assertAlmostEqual(float(stats["h_post_logits_max"]), 9.0, places=5)

    def test_gate_logits_survive_saturation_a_sigmoid_inverse_would_lose(self):
        # A logit of -200 saturates h_pre to exactly 0 in fp32, so recovering it
        # from the activated gate is impossible. Reading it pre-sigmoid must work.
        n = 1
        proj = paddle.to_tensor([[-200.0, 0.0, 0.0]], dtype="float32")
        r = paddle.ones([1, 1], dtype="float32")
        stats = mhc_metrics.gate_logits_extrema(
            proj, r, paddle.ones([1]), paddle.ones([1]), paddle.zeros([3]), n
        )
        self.assertAlmostEqual(
            float(stats["h_pre_logits_min"]), -200.0, places=4
        )
        self.assertAlmostEqual(
            float(paddle.nn.functional.sigmoid(proj[..., :n]).max()),
            0.0,
            places=6,
        )

    def test_mapping_param_stats_slices_the_shared_bias(self):
        # n = 2 -> bias is [b_pre(2), b_post(2), b_res(4)].
        n = 2
        bias = paddle.to_tensor(
            [1.0, -3.0, 0.5, 0.5, 2.0, -8.0, 0.0, 0.0], dtype="float32"
        )
        stats = mhc_metrics.mapping_param_stats(
            paddle.full([1], 0.1),
            paddle.full([1], 0.2),
            paddle.full([1], 0.3),
            bias,
            n,
        )
        self.assertAlmostEqual(float(stats["alpha_pre"]), 0.1, places=6)
        self.assertAlmostEqual(float(stats["alpha_post"]), 0.2, places=6)
        self.assertAlmostEqual(float(stats["alpha_res"]), 0.3, places=6)
        self.assertAlmostEqual(float(stats["bias_pre_mean"]), -1.0, places=6)
        self.assertAlmostEqual(float(stats["bias_pre_abs_max"]), 3.0, places=6)
        self.assertAlmostEqual(float(stats["bias_post_mean"]), 0.5, places=6)
        self.assertAlmostEqual(float(stats["bias_post_abs_max"]), 0.5, places=6)
        self.assertAlmostEqual(float(stats["bias_res_mean"]), -1.5, places=6)
        self.assertAlmostEqual(float(stats["bias_res_abs_max"]), 8.0, places=6)

    def test_logits_extrema_are_detached_fp32_scalars(self):
        logits = paddle.to_tensor([[-3.0, 2.0]], dtype="float16")
        logits.stop_gradient = False
        stats = mhc_metrics.h_res_logits_extrema(logits)
        for value in stats.values():
            self.assertEqual(value.shape, [])
            self.assertEqual(value.dtype, paddle.float32)
            self.assertTrue(value.stop_gradient)


class MHCMonitorTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()
        # Discovery uses isinstance against the real classes; point them at the fakes.
        self._orig_layer_cls = mhc_monitor.HyperConnectionTransformerLayer
        self._orig_mod_cls = mhc_monitor.HyperConnectionModule
        mhc_monitor.HyperConnectionTransformerLayer = FakeLayer
        mhc_monitor.HyperConnectionModule = FakeHC

    def tearDown(self):
        mhc_monitor.HyperConnectionTransformerLayer = self._orig_layer_cls
        mhc_monitor.HyperConnectionModule = self._orig_mod_cls
        training_logs.reset()

    def _identity_layer(self, n, s, b):
        ident = paddle.eye(n).reshape([1, 1, n, n]).expand([s, b, n, n])

        def make_hc():
            return FakeHC(
                n=n,
                h_pre=paddle.full([s, b, n], 0.5),
                h_post=paddle.ones([s, b, n]),
                h_res=paddle.assign(
                    ident
                ),  # own storage, not a view of the shared eye
            )

        return FakeLayer(attn=make_hc(), mlp=make_hc())

    def _drive(self, targets, x_dim):
        # Fire each wrapped compute_mappings; grad is enabled by default in dygraph.
        for _, _, mod in targets:
            mod.compute_mappings(paddle.randn([4, 2, x_dim]))

    def _prepare_and_attach(self, monitor, model):
        all_targets = monitor._prepare(model)
        monitor.allocate_buffers()
        monitor._attach(all_targets)
        # Return a single flat list of entries for convenience.
        return [
            entry for chunk_entries in all_targets for entry in chunk_entries
        ]

    def test_gate_and_amax_metrics_are_recorded(self):
        n, s, b = 4, 2, 3
        layer = self._identity_layer(n, s, b)
        model = _mhc_model([layer])

        monitor = PaddleMHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = self._prepare_and_attach(monitor, model)
        self.assertTrue(targets)

        self._drive(targets, x_dim=n * 8)
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        for comp in ("attn", "mlp"):
            for key in ("h_pre_mean", "h_pre_std", "h_post_mean", "h_post_std"):
                self.assertIn(f"mhc_health/layer_0/{comp}_{key}", latest)
            # h_pre == 0.5 everywhere, h_post == 1.0 everywhere.
            self.assertAlmostEqual(
                latest[f"mhc_health/layer_0/{comp}_h_pre_mean"], 0.5, places=5
            )
            self.assertAlmostEqual(
                latest[f"mhc_health/layer_0/{comp}_h_post_mean"], 1.0, places=5
            )
            # Both amax gains are emitted per layer and globally.
            for key in ("amax_gain_fwd", "amax_gain_bwd"):
                self.assertAlmostEqual(
                    latest[f"mhc_health/layer_0/{comp}_{key}"], 1.0, places=4
                )
                self.assertAlmostEqual(
                    latest[f"mhc_health/global_{comp}_{key}"], 1.0, places=4
                )

    def test_amax_gain_axes_are_not_swapped(self):
        # An asymmetric h_res pins the convention. Streams mix as
        # `out = h_res^T @ x`, so _fwd must read the COLUMN sums of h_res (1.5
        # here) and _bwd the ROW sums (1.0 here) — single layer and composite
        # agree. Identity matrices cannot tell the two apart, which is how the
        # fwd/bwd label mix-up stayed invisible.
        n, s, b = 2, 1, 1
        h_res = (
            paddle.to_tensor([[0.75, 0.25], [0.75, 0.25]])
            .reshape([1, 1, n, n])
            .expand([s, b, n, n])
        )
        hc = FakeHC(
            n=n,
            h_pre=paddle.full([s, b, n], 0.5),
            h_post=paddle.ones([s, b, n]),
            h_res=paddle.assign(h_res),
        )
        model = _mhc_model([FakeLayer(attn=hc, mlp=hc)])

        monitor = PaddleMHCHealthMonitor()
        targets = self._prepare_and_attach(monitor, model)
        self._drive(targets, x_dim=n * 8)
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        self.assertAlmostEqual(
            latest["mhc_health/global_attn_amax_gain_fwd"], 1.5, places=5
        )
        self.assertAlmostEqual(
            latest["mhc_health/global_attn_amax_gain_bwd"], 1.0, places=5
        )
        self.assertAlmostEqual(
            latest["mhc_health/global_attn_composite_amax_gain_fwd_max"],
            1.5,
            places=5,
        )
        self.assertAlmostEqual(
            latest["mhc_health/global_attn_composite_amax_gain_bwd_max"],
            1.0,
            places=5,
        )

    def test_per_element_mapping_series_are_recorded_row_major(self):
        # The n^2 + 2n per-element curves. An asymmetric h_res pins the layout:
        # cell{i} is row-major over h_res *as compute_mappings returns it*, so a
        # stray transpose (the orientation the composite product needs) would
        # swap cell1 and cell2.
        n, s, b = 2, 2, 3
        h_res = (
            paddle.to_tensor([[0.1, 0.2], [0.3, 0.4]])
            .reshape([1, 1, n, n])
            .expand([s, b, n, n])
        )
        hc = FakeHC(
            n=n,
            h_pre=paddle.to_tensor([0.25, 0.75])
            .reshape([1, 1, n])
            .expand([s, b, n]),
            h_post=paddle.to_tensor([1.0, 2.0])
            .reshape([1, 1, n])
            .expand([s, b, n]),
            h_res=paddle.assign(h_res),
        )
        model = _mhc_model([FakeLayer(attn=hc, mlp=hc)])

        monitor = PaddleMHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = self._prepare_and_attach(monitor, model)
        self._drive(targets, x_dim=n * 8)
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        for comp in ("attn", "mlp"):
            for i, expected in enumerate((0.1, 0.2, 0.3, 0.4)):
                key = f"mhc_health/layer_0/{comp}_h_res_cell{i}"
                self.assertIn(key, latest)
                self.assertAlmostEqual(latest[key], expected, places=5)
            for j, (pre, post) in enumerate(((0.25, 1.0), (0.75, 2.0))):
                self.assertAlmostEqual(
                    latest[f"mhc_health/layer_0/{comp}_h_pre_idx{j}"],
                    pre,
                    places=5,
                )
                self.assertAlmostEqual(
                    latest[f"mhc_health/layer_0/{comp}_h_post_idx{j}"],
                    post,
                    places=5,
                )

        # Vector metrics must not spawn `global_*` keys: the mean of one cell over
        # layers has no reading, and the flush-time derivation would emit it.
        self.assertNotIn("mhc_health/global_attn_h_res_cell0", latest)
        # The composite product still reads the transposed operator, unchanged by
        # sharing the token-mean reduce with the new series.
        self.assertIn(
            "mhc_health/layer_0/attn_composite_amax_gain_fwd_max", latest
        )

    def test_per_element_series_count_matches_n(self):
        n, s, b = 4, 2, 3
        model = _mhc_model([self._identity_layer(n, s, b)])
        monitor = PaddleMHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = self._prepare_and_attach(monitor, model)
        self._drive(targets, x_dim=n * 8)
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        # Exact `<tag><digits>` suffixes, not a substring test: `_h_pre_s` would
        # also catch the scalar `h_pre_std`, which is what the `idx` tag avoids.
        elem_re = re.compile(r"_(?:h_res_cell|h_pre_idx|h_post_idx)\d+$")
        for comp in ("attn", "mlp"):
            for tag, expected in (
                ("h_res_cell", n * n),
                ("h_pre_idx", n),
                ("h_post_idx", n),
            ):
                found = [
                    k for k in latest if re.search(rf"/{comp}_{tag}\d+$", k)
                ]
                self.assertEqual(len(found), expected, tag)
        # 2 components x (n^2 + 2n) = 48 new per-layer keys at n=4.
        self.assertEqual(len([k for k in latest if elem_re.search(k)]), 48)

    def test_logits_gradient_extrema_are_unscaled_without_changing_backward(
        self,
    ):
        n, s, b = 2, 1, 1
        source = paddle.zeros([s, b, n * n], dtype="float32")
        source.stop_gradient = False
        logits = source * 1.0
        hc = FakeHC(
            n=n,
            h_pre=paddle.full([s, b, n], 0.5),
            h_post=paddle.ones([s, b, n]),
            h_res=paddle.eye(n).reshape([s, b, n, n]),
            h_res_logits=logits,
        )
        model = _mhc_model(
            [
                FakeLayer(
                    attn=hc,
                    mlp=self._identity_layer(n, s, b).mlp_hyper_connection,
                )
            ]
        )
        monitor = PaddleMHCHealthMonitor(log_per_layer=True, log_global=True)
        self._prepare_and_attach(monitor, model)

        _, _, captured = hc._compute_h(None, None)
        weight = paddle.to_tensor([[[2.0, -7.0, 4.0, 1.5]]], dtype="float32")
        loss_scale = 16.0
        (captured * weight * loss_scale).sum().backward()

        # The hook observes but never modifies the autograd gradient.
        self.assertTrue(paddle.allclose(source.grad, weight * loss_scale))
        monitor.finalize_scaled_grad_metrics(
            SimpleNamespace(_scale=paddle.to_tensor(loss_scale))
        )
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        for key in (
            "mhc_health/layer_0/attn_h_res_logits_grad_min",
            "mhc_health/global_attn_h_res_logits_grad_min",
        ):
            self.assertAlmostEqual(latest[key], -7.0, places=5)
        for key in (
            "mhc_health/layer_0/attn_h_res_logits_grad_max",
            "mhc_health/global_attn_h_res_logits_grad_max",
        ):
            self.assertAlmostEqual(latest[key], 4.0, places=5)

    def test_logits_gradient_hook_skips_no_grad_recompute_forward(self):
        n, s, b = 2, 1, 1
        source = paddle.zeros([s, b, n * n], dtype="float32")
        source.stop_gradient = False
        hc = FakeHC(
            n=n,
            h_pre=paddle.full([s, b, n], 0.5),
            h_post=paddle.ones([s, b, n]),
            h_res=paddle.eye(n).reshape([s, b, n, n]),
            h_res_logits=source * 1.0,
        )
        model = _mhc_model(
            [
                FakeLayer(
                    attn=hc,
                    mlp=self._identity_layer(n, s, b).mlp_hyper_connection,
                )
            ]
        )
        monitor = PaddleMHCHealthMonitor()
        self._prepare_and_attach(monitor, model)

        with paddle.no_grad():
            hc._compute_h(None, None)
        min_key = "mhc_health/layer_0/attn_h_res_logits_grad_min"
        max_key = "mhc_health/layer_0/attn_h_res_logits_grad_max"
        self.assertEqual(monitor._gpu_cnt[min_key], 0)
        self.assertEqual(monitor._gpu_cnt[max_key], 0)

        _, _, replay_logits = hc._compute_h(None, None)
        replay_logits.sum().backward()
        self.assertEqual(monitor._gpu_cnt[min_key], 1)
        self.assertEqual(monitor._gpu_cnt[max_key], 1)
        monitor.step()
        latest = training_logs.get_latest(prefix="mhc_health")
        self.assertAlmostEqual(latest[min_key], 1.0, places=5)
        self.assertAlmostEqual(latest[max_key], 1.0, places=5)

    def test_logits_extrema_are_recorded_and_reduce_by_extremum_across_layers(
        self,
    ):
        # layer_0 has zero logits; layer_1 spans [-104, 7.5]. The global series
        # must reduce by MIN / MAX so an extreme layer is not averaged away.
        n, s, b = 4, 2, 3
        extreme = (
            paddle.to_tensor(
                [
                    *[[-25.7, 7.5, -100.0, -61.0], [-33.4, 0.0, -89.0, -48.0]],
                    *[[-34.0, -24.9, -104.0, 0.0], [-79.4, -72.6, -104.0, 0.0]],
                ],
                dtype="float32",
            )
            .reshape([1, 1, n * n])
            .expand([s, b, n * n])
        )

        def sick_hc():
            return FakeHC(
                n=n,
                h_pre=paddle.full([s, b, n], 0.5),
                h_post=paddle.ones([s, b, n]),
                h_res=paddle.eye(n).reshape([1, 1, n, n]).expand([s, b, n, n]),
                h_res_logits=paddle.assign(extreme),
            )

        model = _mhc_model(
            [
                self._identity_layer(n, s, b),
                FakeLayer(attn=sick_hc(), mlp=sick_hc()),
            ]
        )

        monitor = PaddleMHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = self._prepare_and_attach(monitor, model)
        self._drive(targets, x_dim=n * 8)
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        for comp in ("attn", "mlp"):
            self.assertAlmostEqual(
                latest[f"mhc_health/layer_0/{comp}_h_res_logits_min"],
                0.0,
                places=5,
            )
            self.assertAlmostEqual(
                latest[f"mhc_health/layer_1/{comp}_h_res_logits_min"],
                -104.0,
                places=5,
            )
            self.assertAlmostEqual(
                latest[f"mhc_health/global_{comp}_h_res_logits_min"],
                -104.0,
                places=5,
            )
            self.assertAlmostEqual(
                latest[f"mhc_health/layer_0/{comp}_h_res_logits_max"],
                0.0,
                places=5,
            )
            self.assertAlmostEqual(
                latest[f"mhc_health/layer_1/{comp}_h_res_logits_max"],
                7.5,
                places=5,
            )
            self.assertAlmostEqual(
                latest[f"mhc_health/global_{comp}_h_res_logits_max"],
                7.5,
                places=5,
            )

    def test_gate_logits_are_recorded_from_compute_h_arguments(self):
        # alpha = 0.1, bias = 0, r = 1 -> logits are proj/10 on the pre and post
        # slices: pre [1, -2], post [3, -4].
        n, s, b = 2, 1, 1
        layer = self._identity_layer(n, s, b)
        model = _mhc_model([layer])
        monitor = PaddleMHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = self._prepare_and_attach(monitor, model)

        proj = paddle.to_tensor(
            [[[10.0, -20.0, 30.0, -40.0, 0.0, 0.0, 0.0, 0.0]]], dtype="float32"
        )
        r = paddle.ones([s, b, 1], dtype="float32")
        for _, _, mod in targets:
            mod._compute_h(proj, r)
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        for comp in ("attn", "mlp"):
            self.assertAlmostEqual(
                latest[f"mhc_health/layer_0/{comp}_h_pre_logits_min"],
                -2.0,
                places=5,
            )
            self.assertAlmostEqual(
                latest[f"mhc_health/layer_0/{comp}_h_pre_logits_max"],
                1.0,
                places=5,
            )
            self.assertAlmostEqual(
                latest[f"mhc_health/layer_0/{comp}_h_post_logits_min"],
                -4.0,
                places=5,
            )
            self.assertAlmostEqual(
                latest[f"mhc_health/layer_0/{comp}_h_post_logits_max"],
                3.0,
                places=5,
            )
            self.assertAlmostEqual(
                latest[f"mhc_health/global_{comp}_h_pre_logits_min"],
                -2.0,
                places=5,
            )
            self.assertAlmostEqual(
                latest[f"mhc_health/global_{comp}_h_post_logits_max"],
                3.0,
                places=5,
            )

    def test_gate_logits_skipped_when_compute_h_gets_no_proj(self):
        # The stub call path (proj=r=None) must not raise and must leave the four
        # accumulators empty rather than writing a bogus value.
        n, s, b = 2, 1, 1
        model = _mhc_model([self._identity_layer(n, s, b)])
        monitor = PaddleMHCHealthMonitor()
        targets = self._prepare_and_attach(monitor, model)
        for _, _, mod in targets:
            mod._compute_h(None, None)
        self.assertEqual(
            monitor._gpu_cnt["mhc_health/layer_0/attn_h_pre_logits_min"], 0
        )
        monitor.step()
        self.assertNotIn(
            "mhc_health/layer_0/attn_h_pre_logits_min",
            training_logs.get_latest(prefix="mhc_health"),
        )

    def test_mapping_param_metrics_are_recorded_once_per_step(self):
        n, s, b = 2, 1, 1
        bias = paddle.to_tensor(
            [1.0, -3.0, 0.5, 0.5, 2.0, -8.0, 0.0, 0.0], dtype="float32"
        )

        def make_hc():
            return FakeHC(
                n=n,
                h_pre=paddle.full([s, b, n], 0.5),
                h_post=paddle.ones([s, b, n]),
                h_res=paddle.eye(n).reshape([1, 1, n, n]).expand([s, b, n, n]),
                alpha=0.25,
                bias=bias,
            )

        model = _mhc_model([FakeLayer(attn=make_hc(), mlp=make_hc())])
        monitor = PaddleMHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = self._prepare_and_attach(monitor, model)

        # Two microbatches: the parameters are step-level, so they must still be
        # recorded exactly once, and a second finalize must be a no-op.
        self._drive(targets, x_dim=n * 8)
        self._drive(targets, x_dim=n * 8)
        monitor.finalize_param_metrics()
        monitor.finalize_param_metrics()
        self.assertEqual(
            monitor._gpu_cnt["mhc_health/layer_0/attn_alpha_pre"], 1
        )
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        for comp in ("attn", "mlp"):
            for key in ("alpha_pre", "alpha_post", "alpha_res"):
                self.assertAlmostEqual(
                    latest[f"mhc_health/layer_0/{comp}_{key}"], 0.25, places=6
                )
            self.assertAlmostEqual(
                latest[f"mhc_health/layer_0/{comp}_bias_pre_mean"],
                -1.0,
                places=6,
            )
            self.assertAlmostEqual(
                latest[f"mhc_health/layer_0/{comp}_bias_pre_abs_max"],
                3.0,
                places=6,
            )
            self.assertAlmostEqual(
                latest[f"mhc_health/layer_0/{comp}_bias_post_mean"],
                0.5,
                places=6,
            )
            self.assertAlmostEqual(
                latest[f"mhc_health/layer_0/{comp}_bias_res_mean"],
                -1.5,
                places=6,
            )
            self.assertAlmostEqual(
                latest[f"mhc_health/layer_0/{comp}_bias_res_abs_max"],
                8.0,
                places=6,
            )
            self.assertAlmostEqual(
                latest[f"mhc_health/global_{comp}_alpha_res"], 0.25, places=6
            )

    def test_mapping_param_metrics_read_before_the_optimizer_update(self):
        # `on_optimizer_begin` fires before optimizer.step(), so the value logged
        # for this step must be the one its forward used, not the updated one.
        n, s, b = 2, 1, 1

        def make_hc():
            return FakeHC(
                n=n,
                h_pre=paddle.full([s, b, n], 0.5),
                h_post=paddle.ones([s, b, n]),
                h_res=paddle.eye(n).reshape([1, 1, n, n]).expand([s, b, n, n]),
                alpha=0.25,
            )

        model = _mhc_model([FakeLayer(attn=make_hc(), mlp=make_hc())])
        monitor = PaddleMHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = self._prepare_and_attach(monitor, model)

        self._drive(targets, x_dim=n * 8)
        monitor.finalize_scaled_grad_metrics()  # on_optimizer_begin
        for _, _, mod in targets:  # optimizer.step() moves the parameters
            mod.alpha_pre = paddle.full([1], 9.0)
        monitor.step()  # on_step_end, i.e. after the update

        latest = training_logs.get_latest(prefix="mhc_health")
        self.assertAlmostEqual(
            latest["mhc_health/layer_0/attn_alpha_pre"], 0.25, places=6
        )

    def test_mapping_param_metrics_skipped_on_unmonitored_step(self):
        # No forward ran, so the parameter series must not emit a lone sample the
        # activation series has no counterpart for.
        n, s, b = 2, 1, 1
        model = _mhc_model([self._identity_layer(n, s, b)])
        monitor = PaddleMHCHealthMonitor()
        self._prepare_and_attach(monitor, model)
        monitor.step()
        self.assertNotIn(
            "mhc_health/layer_0/attn_alpha_pre",
            training_logs.get_latest(prefix="mhc_health"),
        )

    def test_composite_is_built_in_layer_order_not_call_order(self):
        # Regression test for what retired the previous composite metric: it was
        # accumulated in call order, which under recompute is the reverse-layer
        # backward replay. Here the hooks are fired in reverse on purpose.
        n, s, b = 4, 1, 1
        mats = [
            paddle.to_tensor(
                [
                    [0.9, 0.1, 0.0, 0.0],
                    [0.1, 0.9, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype="float32",
            ),
            paddle.to_tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.5, 0.0, 0.0],
                    [0.0, 0.0, 0.5, 0.5],
                    [0.0, 0.0, 0.5, 0.5],
                ],
                dtype="float32",
            ),
            paddle.to_tensor(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
        ]

        def row_gain(m):
            return float(m.sum(axis=-1).abs().max())

        def col_gain(m):
            return float(m.sum(axis=-2).abs().max())

        components = ("attn", "mlp")
        branches = [
            (layer_idx, component, mat.transpose([1, 0]))
            for layer_idx, mat in enumerate(mats)
            for component in components
        ]

        expected_fwd = {}
        prefix = None
        for layer_idx, component, op in branches:
            prefix = op if prefix is None else paddle.matmul(op, prefix)
            expected_fwd[(layer_idx, component)] = row_gain(prefix)

        expected_bwd = {}
        suffix = None
        for layer_idx, component, op in reversed(branches):
            suffix = op if suffix is None else paddle.matmul(suffix, op)
            expected_bwd[(layer_idx, component)] = col_gain(suffix)
        # Guard against a vacuous test: reversing the physical branch order differs.
        wrong_prefix = None
        for _, _, op in reversed(branches):
            wrong_prefix = (
                op if wrong_prefix is None else paddle.matmul(op, wrong_prefix)
            )
        self.assertNotAlmostEqual(
            expected_fwd[(2, "mlp")], row_gain(wrong_prefix), places=4
        )

        def layer_for(mat):
            def make_hc():
                return FakeHC(
                    n=n,
                    h_pre=paddle.full([s, b, n], 0.5),
                    h_post=paddle.ones([s, b, n]),
                    h_res=paddle.assign(
                        mat.reshape([1, 1, n, n]).expand([s, b, n, n])
                    ),
                )

            return FakeLayer(attn=make_hc(), mlp=make_hc())

        model = _mhc_model([layer_for(m) for m in mats])
        monitor = PaddleMHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = self._prepare_and_attach(monitor, model)
        self._drive(
            list(reversed(targets)), x_dim=n * 8
        )  # deliberately out of order
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        for comp in components:
            for idx in range(len(mats)):
                key = (idx, comp)
                self.assertAlmostEqual(
                    latest[
                        f"mhc_health/layer_{idx}/{comp}_composite_amax_gain_fwd_max"
                    ],
                    expected_fwd[key],
                    places=4,
                )
                self.assertAlmostEqual(
                    latest[
                        f"mhc_health/layer_{idx}/{comp}_composite_amax_gain_bwd_max"
                    ],
                    expected_bwd[key],
                    places=4,
                )
            self.assertAlmostEqual(
                latest[f"mhc_health/global_{comp}_composite_amax_gain_fwd_max"],
                max(
                    value
                    for (idx, component), value in expected_fwd.items()
                    if component == comp
                ),
                places=4,
            )
            self.assertAlmostEqual(
                latest[f"mhc_health/global_{comp}_composite_amax_gain_bwd_max"],
                max(
                    value
                    for (idx, component), value in expected_bwd.items()
                    if component == comp
                ),
                places=4,
            )

    def test_composite_is_one_for_identity_chain(self):
        n, s, b = 4, 2, 3
        model = _mhc_model([self._identity_layer(n, s, b) for _ in range(3)])
        monitor = PaddleMHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = self._prepare_and_attach(monitor, model)
        self._drive(targets, x_dim=n * 8)
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        for comp in ("attn", "mlp"):
            for idx in range(3):
                self.assertAlmostEqual(
                    latest[
                        f"mhc_health/layer_{idx}/{comp}_composite_amax_gain_fwd_max"
                    ],
                    1.0,
                    places=5,
                )

    def test_composite_skips_mtp_layers(self):
        # MTP layers are off the main trunk; multiplying them into the chain has no
        # physical meaning. Marking layer_1 as MTP must leave the layer_0 value
        # untouched and emit nothing for layer_1.
        n, s, b = 4, 2, 3
        model = _mhc_model([self._identity_layer(n, s, b) for _ in range(2)])
        monitor = PaddleMHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = self._prepare_and_attach(monitor, model)
        monitor._mtp_layer_ids.add(1)
        self._drive(targets, x_dim=n * 8)
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        for comp in ("attn", "mlp"):
            self.assertAlmostEqual(
                latest[
                    f"mhc_health/layer_0/{comp}_composite_amax_gain_fwd_max"
                ],
                1.0,
                places=5,
            )
            self.assertNotIn(
                f"mhc_health/layer_1/{comp}_composite_amax_gain_fwd_max", latest
            )

    def test_composite_snapshot_is_cleared_each_step(self):
        n, s, b = 4, 2, 3
        model = _mhc_model([self._identity_layer(n, s, b)])
        monitor = PaddleMHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = self._prepare_and_attach(monitor, model)
        self._drive(targets, x_dim=n * 8)
        self.assertTrue(monitor._h_res_snapshot)
        monitor.step()
        # Stale matrices must not survive into the next step's product.
        self.assertFalse(monitor._h_res_snapshot)

    def test_composite_aggregates_each_microbatch_before_snapshot_overwrite(
        self,
    ):
        n, s, b = 2, 1, 1
        model = _mhc_model([self._identity_layer(n, s, b)])
        monitor = PaddleMHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = self._prepare_and_attach(monitor, model)

        amplified = paddle.diag(paddle.to_tensor([3.0, 1.0])).reshape(
            [1, 1, n, n]
        )
        for _, _, mod in targets:
            mod._h_res = amplified
        self._drive(targets, x_dim=n * 8)
        monitor.finalize_composite_microbatch()
        self.assertFalse(monitor._h_res_snapshot)

        identity = paddle.eye(n).reshape([1, 1, n, n])
        for _, _, mod in targets:
            mod._h_res = identity
        self._drive(targets, x_dim=n * 8)
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        self.assertAlmostEqual(
            latest["mhc_health/layer_0/attn_composite_amax_gain_fwd_max"],
            3.0,
            places=5,
        )
        self.assertAlmostEqual(
            latest["mhc_health/layer_0/mlp_composite_amax_gain_fwd_max"],
            9.0,
            places=5,
        )
        self.assertAlmostEqual(
            latest["mhc_health/layer_0/attn_composite_amax_gain_bwd_max"],
            9.0,
            places=5,
        )
        self.assertAlmostEqual(
            latest["mhc_health/layer_0/mlp_composite_amax_gain_bwd_max"],
            3.0,
            places=5,
        )

    def test_branch_residual_share_is_recorded_from_bda(self):
        n, s, b, c = 4, 2, 3, 6
        layer = self._identity_layer(n, s, b)
        model = _mhc_model([layer])

        monitor = PaddleMHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = self._prepare_and_attach(monitor, model)

        streams = paddle.randn([s, b, n * c])
        x = paddle.randn([s, b, c])
        for _, _, mod in targets:
            mod.fused_h_res_h_post_bda(
                h_res=mod._h_res,
                original_residual=streams,
                h_post=mod._h_post,
                layer_output_with_bias=(x, None),
                dropout_prob=0.0,
                training=True,
                fused=False,
            )
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        for comp in ("attn", "mlp"):
            for key in ("branch_residual_share", "branch_residual_share_max"):
                self.assertIn(f"mhc_health/layer_0/{comp}_{key}", latest)
            # h_post == 1 everywhere so the branch norm is √n·‖x‖ and h_res is the
            # identity, hence the share must sit strictly inside (0, 1].
            share = latest[f"mhc_health/layer_0/{comp}_branch_residual_share"]
            self.assertGreater(share, 0.0)
            self.assertLessEqual(share, 1.0)
            self.assertGreaterEqual(
                latest[f"mhc_health/layer_0/{comp}_branch_residual_share_max"],
                share,
            )

    def test_h_post_structure_keys_recorded(self):
        n, s, b = 4, 2, 3
        layer = self._identity_layer(n, s, b)
        model = _mhc_model([layer])
        monitor = PaddleMHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = self._prepare_and_attach(monitor, model)
        self._drive(targets, x_dim=n * 8)
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        for comp in ("attn", "mlp"):
            # h_post is constant 1.0 -> equal across streams and across tokens.
            self.assertAlmostEqual(
                latest[
                    f"mhc_health/layer_0/{comp}_h_post_stream_concentration"
                ],
                1.0,
                places=4,
            )
            self.assertAlmostEqual(
                latest[f"mhc_health/layer_0/{comp}_h_post_token_std"],
                0.0,
                places=5,
            )

    def test_remove_hooks_restores_both_wrapped_methods(self):
        n, s, b = 4, 2, 3
        layer = self._identity_layer(n, s, b)
        model = _mhc_model([layer])
        monitor = PaddleMHCHealthMonitor()
        hc = layer.self_attention_hyper_connection
        self._prepare_and_attach(monitor, model)
        self.assertIn("fused_h_res_h_post_bda", vars(hc))
        monitor.remove_hooks()
        self.assertNotIn("fused_h_res_h_post_bda", vars(hc))
        self.assertEqual(
            hc.fused_h_res_h_post_bda.__func__, FakeHC.fused_h_res_h_post_bda
        )

    def test_remove_hooks_restores_compute_mappings(self):
        n, s, b = 4, 2, 3
        layer = self._identity_layer(n, s, b)
        model = _mhc_model([layer])
        monitor = PaddleMHCHealthMonitor()
        original = layer.self_attention_hyper_connection.compute_mappings
        self._prepare_and_attach(monitor, model)
        self.assertIsNot(
            layer.self_attention_hyper_connection.compute_mappings, original
        )
        monitor.remove_hooks()
        # falls back to the (bound) class method after deleting the instance attr
        self.assertEqual(
            layer.self_attention_hyper_connection.compute_mappings.__func__,
            FakeHC.compute_mappings,
        )
        self.assertEqual(monitor._wrapped, [])

    def test_no_graph_retention(self):
        n, s, b = 4, 2, 3
        ident = paddle.eye(n).reshape([1, 1, n, n]).expand([s, b, n, n])
        # Outputs attached to a graph (require grad) — the wrapper must detach.
        leaf = paddle.zeros([s, b, n])
        leaf.stop_gradient = False
        h_pre = leaf + 0.5
        h_post = leaf + 1.0
        h_res = paddle.assign(ident) * (leaf.sum() + 1.0)  # grad-tracked
        hc = FakeHC(n=n, h_pre=h_pre, h_post=h_post, h_res=h_res)
        layer = FakeLayer(attn=hc, mlp=hc)
        model = _mhc_model([layer])

        monitor = PaddleMHCHealthMonitor()
        targets = self._prepare_and_attach(monitor, model)
        for _, _, mod in targets:
            mod.compute_mappings(paddle.randn([4, 2, n * 8]))

        # The 0-dim accumulators must not be graph-tracked: if the wrapper forgot
        # to detach, the graph stays pinned through them until backward.
        for key, acc in monitor._gpu_acc.items():
            self.assertTrue(acc.stop_gradient, msg=key)
        monitor.step()


class MHCVectorFamilyExclusionTest(unittest.TestCase):
    """`cell` / `stream` are per-unit families and must be droppable.

    Regression: `declare_layer_vector` judged the family on the undecorated base
    name, and `h_res` / `h_pre` / `h_post` classify as the *aggregate* families
    (`mix` / `gate`) that production keeps. Only the expanded element names carry
    `cell` / `stream`, so a non-debug run collected all 624 curves anyway.
    """

    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def _declare(self, exclude_families):
        monitor = mhc_monitor.PaddleMHCHealthMonitor(
            log_per_layer=True,
            log_global=False,
            exclude_families=exclude_families,
        )
        for name, tag, size in mhc_monitor._vector_metric_specs(4):
            monitor.declare_layer_vector(0, f"attn_{name}", size, elem_tag=tag)
        return monitor

    def test_cell_and_stream_vectors_are_dropped(self):
        monitor = self._declare("cell+stream")
        self.assertEqual(monitor._vector_keys, {})

    def test_dropping_cell_alone_keeps_the_stream_vectors(self):
        monitor = self._declare("cell")
        self.assertEqual(
            sorted(monitor._vector_keys),
            [
                "mhc_health/layer_0/attn_h_post",
                "mhc_health/layer_0/attn_h_pre",
            ],
        )

    def test_aggregate_families_do_not_drop_the_per_unit_vectors(self):
        """`mix` / `gate` must not take `h_res` / `h_pre` down with them."""
        monitor = self._declare("mix+gate")
        self.assertEqual(len(monitor._vector_keys), 3)

    def test_default_run_collects_everything(self):
        monitor = self._declare(None)
        self.assertEqual(len(monitor._vector_keys), 3)


class MHCMonitorNoOpTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_auto_skip_no_hc(self):
        # A plain model (no HyperConnectionTransformerLayer) -> wraps nothing.
        plain_layer = nn.Linear(4, 4)
        model = _mhc_model([plain_layer])
        monitor_dict = {}
        mhc_monitor.setup_mhc_monitor(model, monitor_dict=monitor_dict)
        self.assertEqual(monitor_dict, {})

    def test_no_op_when_unimportable(self):
        # Simulate mHC classes not importable -> setup is a total no-op, no raise.
        orig_layer = mhc_monitor.HyperConnectionTransformerLayer
        orig_mod = mhc_monitor.HyperConnectionModule
        mhc_monitor.HyperConnectionTransformerLayer = None
        mhc_monitor.HyperConnectionModule = None
        try:
            model = _mhc_model([nn.Linear(4, 4)])
            monitor_dict = {}
            returned = mhc_monitor.setup_mhc_monitor(
                model, monitor_dict=monitor_dict
            )
            self.assertIs(returned, model)
            self.assertEqual(monitor_dict, {})
        finally:
            mhc_monitor.HyperConnectionTransformerLayer = orig_layer
            mhc_monitor.HyperConnectionModule = orig_mod


if __name__ == "__main__":
    unittest.main()
