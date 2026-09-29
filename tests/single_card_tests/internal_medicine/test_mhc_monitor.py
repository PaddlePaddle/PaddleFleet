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

importlib.import_module("_backend_env").skip_unless_backend("megatron")

try:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
except (
    Exception
) as exc:  # pragma: no cover - depends on optional backend install
    raise unittest.SkipTest(f"torch backend unavailable: {exc}") from exc

mhc_metrics = importlib.import_module(
    "paddlefleet.internal_medicine.backends.megatron.mhc_metrics"
)
mhc_monitor = importlib.import_module(
    "paddlefleet.internal_medicine.backends.megatron.mhc_monitor"
)
training_logs = importlib.import_module(
    "paddlefleet.internal_medicine.core.training_logs"
).training_logs

MHCHealthMonitor = mhc_monitor.MHCHealthMonitor


class FakeHC(nn.Module):
    """Stand-in for HyperConnectionModule exposing compute_mappings + layer_number."""

    def __init__(self, n, layer_number, h_pre, h_post, h_res):
        super().__init__()
        self.n = n
        self.layer_number = layer_number
        self._h_pre = h_pre
        self._h_post = h_post
        self._h_res = h_res

    def compute_mappings(self, x):
        return self._h_pre, self._h_post, self._h_res

    def forward(
        self, hidden_states, mhc_recompute_manager=None
    ):  # pragma: no cover - unused
        return hidden_states, self._h_res, self._h_post


class FakeLayer(nn.Module):
    def __init__(self, layer_number, attn, mlp):
        super().__init__()
        self.layer_number = layer_number
        self.self_attention_hyper_connection = attn
        self.mlp_hyper_connection = mlp


def _mhc_model(layers):
    return SimpleNamespace(
        decoder=SimpleNamespace(layers=nn.ModuleList(layers))
    )


class MHCMetricsTest(unittest.TestCase):
    def test_amax_gain_row_vs_col(self):
        # [[1,2],[3,4]] -> row sums [3,7] -> max abs 7 (fwd); col sums [4,6] -> 6 (bwd).
        m = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        self.assertAlmostEqual(
            mhc_metrics.amax_gain(m, dim=-1).item(), 7.0, places=5
        )
        self.assertAlmostEqual(
            mhc_metrics.amax_gain(m, dim=-2).item(), 6.0, places=5
        )

    def test_doubly_stochastic_gain_is_one(self):
        n = 4
        ident = torch.eye(n).reshape(1, 1, n, n)
        self.assertAlmostEqual(
            mhc_metrics.amax_gain(ident, dim=-1).item(), 1.0, places=5
        )
        self.assertAlmostEqual(
            mhc_metrics.amax_gain(ident, dim=-2).item(), 1.0, places=5
        )

    def test_gate_stats(self):
        h = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]])
        mean, std = mhc_metrics.gate_stats(h)
        self.assertAlmostEqual(mean.item(), 1.5, places=5)
        self.assertAlmostEqual(std.item(), h.std().item(), places=6)


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

    def _identity_layer(self, n, s, b, layer_number=1):
        ident = torch.eye(n).reshape(1, 1, n, n).expand(s, b, n, n).contiguous()

        def make_hc():
            return FakeHC(
                n=n,
                layer_number=layer_number,
                h_pre=torch.full((s, b, n), 0.5),
                h_post=torch.ones(s, b, n),
                h_res=ident,
            )

        return FakeLayer(
            layer_number=layer_number, attn=make_hc(), mlp=make_hc()
        )

    def _drive(self, targets, x_dim):
        # Fire each wrapped compute_mappings (grad enabled -> _should_monitor passes).
        for _, _, mod, _, _ in targets:
            mod.compute_mappings(torch.randn(4, 2, x_dim))

    def test_identity_composite_stays_unit_gain(self):
        n, s, b = 4, 2, 3
        layer = self._identity_layer(n, s, b, layer_number=1)
        model = _mhc_model([layer])

        monitor = MHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = monitor._prepare_layers(model, chunk_id=0)
        self.assertTrue(targets)
        monitor.allocate_buffers(torch.device("cpu"))
        monitor._attach_hooks(targets)

        self._drive(targets, x_dim=n * 8)
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        for comp in ("attn", "mlp"):
            for key in (
                "h_pre_mean",
                "h_pre_std",
                "h_post_mean",
                "h_post_std",
                "amax_gain_fwd",
                "amax_gain_bwd",
                "composite_amax_gain_fwd",
                "composite_amax_gain_bwd",
            ):
                self.assertIn(f"mhc_health/layer_0/{comp}_{key}", latest)
            self.assertAlmostEqual(
                latest[f"mhc_health/layer_0/{comp}_amax_gain_fwd"],
                1.0,
                places=4,
            )
            self.assertAlmostEqual(
                latest[f"mhc_health/layer_0/{comp}_composite_amax_gain_fwd"],
                1.0,
                places=4,
            )
            self.assertAlmostEqual(
                latest[f"mhc_health/layer_0/{comp}_composite_amax_gain_bwd"],
                1.0,
                places=4,
            )
            # h_pre == 0.5 everywhere, h_post == 1.0 everywhere.
            self.assertAlmostEqual(
                latest[f"mhc_health/layer_0/{comp}_h_pre_mean"], 0.5, places=5
            )
            self.assertAlmostEqual(
                latest[f"mhc_health/layer_0/{comp}_h_post_mean"], 1.0, places=5
            )
        # global aggregate is derived too.
        self.assertIn("mhc_health/global_attn_amax_gain_fwd", latest)

    def test_remove_hooks_restores_compute_mappings(self):
        n, s, b = 4, 2, 3
        layer = self._identity_layer(n, s, b)
        model = _mhc_model([layer])
        monitor = MHCHealthMonitor()
        targets = monitor._prepare_layers(model, chunk_id=0)
        monitor.allocate_buffers(torch.device("cpu"))
        original = layer.self_attention_hyper_connection.compute_mappings
        monitor._attach_hooks(targets)
        self.assertIsNot(
            layer.self_attention_hyper_connection.compute_mappings, original
        )
        monitor.remove_hooks()
        # falls back to the (bound) class method
        self.assertEqual(
            layer.self_attention_hyper_connection.compute_mappings.__func__,
            FakeHC.compute_mappings,
        )
        self.assertEqual(monitor._wrapped, [])
        self.assertEqual(monitor._composite, {})

    def test_no_graph_retention(self):
        n, s, b = 4, 2, 3
        ident = torch.eye(n).reshape(1, 1, n, n).expand(s, b, n, n).contiguous()
        # Outputs attached to a graph (require grad) — the wrapper must detach.
        leaf = torch.zeros(s, b, n, requires_grad=True)
        h_pre = leaf + 0.5
        h_post = leaf + 1.0
        h_res = ident * (leaf.sum() + 1.0)  # requires_grad, has grad_fn
        hc = FakeHC(
            n=n, layer_number=1, h_pre=h_pre, h_post=h_post, h_res=h_res
        )
        layer = FakeLayer(layer_number=1, attn=hc, mlp=hc)
        model = _mhc_model([layer])

        monitor = MHCHealthMonitor()
        targets = monitor._prepare_layers(model, chunk_id=0)
        monitor.allocate_buffers(torch.device("cpu"))
        monitor._attach_hooks(targets)
        for _, _, mod, _, _ in targets:
            mod.compute_mappings(torch.randn(4, 2, n * 8))

        stored = monitor._composite[0]
        self.assertFalse(stored.requires_grad)
        self.assertIsNone(stored.grad_fn)
        monitor.step()
        self.assertEqual(monitor._composite, {})  # cleared between steps


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
