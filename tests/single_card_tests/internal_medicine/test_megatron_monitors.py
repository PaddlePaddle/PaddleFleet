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
import math
import sys
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

importlib.import_module("_backend_env").skip_unless_backend("megatron")

try:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    F = importlib.import_module("torch.nn.functional")
except (
    Exception
) as exc:  # pragma: no cover - depends on optional backend install
    raise unittest.SkipTest(f"torch backend unavailable: {exc}") from exc

MassiveActivationMonitor = importlib.import_module(
    "paddlefleet.internal_medicine.backends.megatron.massive_activation_monitor"
).MassiveActivationMonitor
MoESpecialistMonitor = importlib.import_module(
    "paddlefleet.internal_medicine.backends.megatron.moe_monitor"
).MoESpecialistMonitor
moe_monitor_module = importlib.import_module(
    "paddlefleet.internal_medicine.backends.megatron.moe_monitor"
)
PLEHealthMonitor = importlib.import_module(
    "paddlefleet.internal_medicine.backends.megatron.ple_monitor"
).PLEHealthMonitor
training_logs = importlib.import_module(
    "paddlefleet.internal_medicine.core.training_logs"
).training_logs
massive_activation_metrics = importlib.import_module(
    "paddlefleet.internal_medicine.backends.megatron.massive_activation_metrics"
)
compute_sink_head_classification = importlib.import_module(
    "paddlefleet.internal_medicine.backends.megatron.sink_head_metrics"
).compute_sink_head_classification
optim_update_module = importlib.import_module(
    "paddlefleet.internal_medicine.backends.megatron.optim_update_monitor"
)
OptimUpdateMonitor = optim_update_module.OptimUpdateMonitor
setup_optim_update_monitor = optim_update_module.setup_optim_update_monitor
megatron_backend = importlib.import_module(
    "paddlefleet.internal_medicine.backends.megatron"
)


class FakePLESublayer:
    act_fn = F.gelu


class WeakRefable:
    """Attribute bag that (unlike SimpleNamespace) supports weakref, so it can
    stand in for a router module in _load_balance_routers."""

    def __init__(self, **attrs):
        self.__dict__.update(attrs)


class FakeMoELayer(nn.Module):
    def __init__(self, experts):
        super().__init__()
        self.experts = experts
        self.shared_experts = None


class FakeLogitLensLayer(nn.Module):
    """Minimal Megatron-style layer: called all-keyword, returns (output, context)."""

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        return self.linear(hidden_states), None


class FakeDecoder(nn.Module):
    def __init__(self, num_layers, hidden_size):
        super().__init__()
        self.layers = nn.ModuleList(
            [FakeLogitLensLayer(hidden_size) for _ in range(num_layers)]
        )
        self.final_layernorm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        for layer in self.layers:
            x, _ = layer(hidden_states=x)
        return x


class FakeGPTModel(nn.Module):
    """Fake GPTModel exposing decoder.layers/final_layernorm + output_layer (head stage)."""

    def __init__(self, num_layers, hidden_size, vocab_size):
        super().__init__()
        self.decoder = FakeDecoder(num_layers, hidden_size)
        self.output_layer = nn.Linear(
            hidden_size, vocab_size, bias=False
        )  # weight [vocab, hidden]

    def forward(self, x, labels=None):
        # labels is accepted (and ignored) so the monitor's label-capture pre-hook can
        # read it from kwargs, mirroring Megatron GPTModel.forward(..., labels=...).
        return self.decoder(x)


class FakeHeadlessGPTModel(nn.Module):
    """Non-head PP stage: decoder present but no output_layer / tied weight accessor."""

    def __init__(self, num_layers, hidden_size):
        super().__init__()
        self.decoder = FakeDecoder(num_layers, hidden_size)

    def forward(self, x, labels=None):
        # labels is accepted (and ignored) so the monitor's label-capture pre-hook can
        # read it from kwargs, mirroring Megatron GPTModel.forward(..., labels=...).
        return self.decoder(x)


class MegatronMoEMonitorTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_router_metrics_flush_from_gpu_buffer(self):
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        for name in moe_monitor_module._ROUTER_METRICS:
            monitor.declare_layer_metric(0, name)
        monitor.allocate_buffers(torch.device("cpu"))

        router = SimpleNamespace(
            topk=2,
            _cached_scores_for_aux_loss=torch.tensor(
                [
                    [0.7, 0.2, 0.1],
                    [0.1, 0.6, 0.3],
                ],
                dtype=torch.float32,
            ),
        )
        monitor._compute_router_metrics(0, router, None, None)

        monitor.step()
        latest = training_logs.get_latest(prefix="moe_health")
        self.assertIn("moe_health/layer_0/router_entropy", latest)
        self.assertIn("moe_health/layer_0/score_sum_mean", latest)
        self.assertIn("moe_health/global_router_entropy", latest)
        self.assertIn("moe_health/global_score_sum_max", latest)

    def test_step_computes_expert_metrics_even_under_no_grad(self):
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        for name in moe_monitor_module._EXPERT_METRICS:
            monitor.declare_layer_metric(0, name)
        monitor.allocate_buffers(torch.device("cpu"))

        hidden_size = 4
        ffn_hidden = 8
        num_experts = 2
        experts = SimpleNamespace(
            num_local_experts=num_experts,
            config=SimpleNamespace(hidden_size=hidden_size),
            weight1=torch.nn.Parameter(
                torch.ones(num_experts * hidden_size, ffn_hidden)
            ),
            weight2=torch.nn.Parameter(
                torch.ones(num_experts * ffn_hidden, hidden_size)
            ),
        )
        moe_layer = FakeMoELayer(experts)
        monitor._monitored_moe_layers = [(0, weakref.ref(moe_layer))]

        with torch.no_grad():
            monitor.step()

        latest = training_logs.get_latest(prefix="moe_health")
        self.assertIn("moe_health/layer_0/expert_norm_mean", latest)
        self.assertIn("moe_health/global_expert_norm_mean", latest)

    def test_load_balance_metrics_from_reduced_tokens_per_expert(self):
        # Two bias-enabled layers, recorded in the order finalize_model_grads
        # stacks them. tokens_per_expert is the ALREADY-reduced global count.
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        monitor._expert_bias_enabled = True
        monitor._load_balance_layer_order = [0, 1]
        for layer_idx in (0, 1):
            for name in moe_monitor_module._LOAD_BALANCE_METRICS:
                monitor.declare_layer_metric(layer_idx, name)
        monitor.allocate_buffers(torch.device("cpu"))

        # layer 0: [5,3,2,2] -> max/min=5/2=2.5, max/median=5/2=2.5
        # layer 1: [10,0,4,6] -> max/min=10/1(clamp)=10, max/median=10/4=2.5
        reduced = torch.tensor([[5.0, 3.0, 2.0, 2.0], [10.0, 0.0, 4.0, 6.0]])
        monitor._record_load_balance_metrics(
            reduced, monitor._load_balance_layer_order
        )
        monitor.step()

        latest = training_logs.get_latest(prefix="moe_health")
        self.assertAlmostEqual(
            latest["moe_health/layer_0/load_max_min_ratio"], 2.5, places=5
        )
        self.assertAlmostEqual(
            latest["moe_health/layer_1/load_max_min_ratio"], 10.0, places=5
        )
        self.assertAlmostEqual(
            latest["moe_health/layer_0/load_max_median_ratio"], 2.5, places=5
        )
        self.assertAlmostEqual(
            latest["moe_health/layer_1/load_max_median_ratio"], 2.5, places=5
        )
        # CV = population std / mean. layer 0 mean=3, std=sqrt(((2)+(0)+(1)+(1))/4)=sqrt(1.5)
        self.assertAlmostEqual(
            latest["moe_health/layer_0/load_cv"], (1.5**0.5) / 3.0, places=5
        )
        # Normalized load entropy H(p)/log(E) and effective experts exp(H).
        self.assertAlmostEqual(
            latest["moe_health/layer_0/load_balance_entropy_norm"],
            0.943959,
            places=5,
        )
        self.assertAlmostEqual(
            latest["moe_health/layer_1/load_balance_entropy_norm"],
            0.742738,
            places=5,
        )
        self.assertAlmostEqual(
            latest["moe_health/layer_0/load_effective_experts"],
            3.701009,
            places=5,
        )
        self.assertAlmostEqual(
            latest["moe_health/layer_1/load_effective_experts"],
            2.800094,
            places=5,
        )
        self.assertIn("moe_health/global_load_max_min_ratio", latest)
        self.assertIn("moe_health/global_load_cv", latest)
        self.assertIn("moe_health/global_load_balance_entropy_norm", latest)
        self.assertIn("moe_health/global_load_effective_experts", latest)

    def test_load_balance_metrics_absent_without_any_source(self):
        # When no router exposes global_tokens_per_expert and none has expert-bias
        # enabled, the metric must not be declared (schema stays clean) — nothing
        # to record or patch.
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        self.assertFalse(monitor._expert_bias_enabled)
        self.assertFalse(monitor._global_lb_enabled)
        self.assertEqual(monitor._load_balance_layer_order, [])
        self.assertEqual(monitor._load_balance_routers, [])
        # Both global patches are idempotent no-ops when no source is present.
        monitor._patch_expert_bias_update()
        monitor._patch_reset_temporary_tensors()
        self.assertIsNone(monitor._orig_get_updated_expert_bias)
        self.assertIsNone(monitor._orig_reset_model_temporary_tensors)

    def test_prepare_layers_prefers_global_over_expert_bias(self):
        # A router exposing both global_tokens_per_expert and enable_expert_bias
        # must be routed to the global source, not the expert-bias fallback.
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        router = WeakRefable(
            global_tokens_per_expert=torch.zeros(4),
            ga_steps=torch.tensor(0.0),
            enable_expert_bias=True,
        )
        moe_layer = SimpleNamespace(router=router)
        monitor._find_moe_layers = lambda _model: [(0, moe_layer)]
        monitor._prepare_layers(object())
        self.assertTrue(monitor._global_lb_enabled)
        self.assertEqual([idx for idx, _ in monitor._load_balance_routers], [0])
        self.assertFalse(monitor._expert_bias_enabled)
        self.assertEqual(monitor._load_balance_layer_order, [])

    def test_load_balance_metrics_from_global_tokens_per_expert(self):
        # Preferred path: counts sourced from each router's own
        # global_tokens_per_expert buffer (already TPxCPxDP-reduced).
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        routers = [
            WeakRefable(
                global_tokens_per_expert=torch.tensor([5.0, 3.0, 2.0, 2.0]),
                ga_steps=torch.tensor(4.0),
            ),
            WeakRefable(
                global_tokens_per_expert=torch.tensor([10.0, 0.0, 4.0, 6.0]),
                ga_steps=torch.tensor(4.0),
            ),
        ]
        monitor._global_lb_enabled = True
        monitor._load_balance_routers = [
            (idx, weakref.ref(r)) for idx, r in enumerate(routers)
        ]
        for layer_idx in (0, 1):
            for name in moe_monitor_module._LOAD_BALANCE_METRICS:
                monitor.declare_layer_metric(layer_idx, name)
        monitor.allocate_buffers(torch.device("cpu"))

        monitor._record_global_load_balance_metrics()
        monitor.step()

        latest = training_logs.get_latest(prefix="moe_health")
        # Ratios are scale-invariant, so raw sums give the same values as the
        # expert-bias test rows: layer 0 -> 2.5, layer 1 -> 10 / 2.5.
        self.assertAlmostEqual(
            latest["moe_health/layer_0/load_max_min_ratio"], 2.5, places=5
        )
        self.assertAlmostEqual(
            latest["moe_health/layer_1/load_max_min_ratio"], 10.0, places=5
        )
        self.assertAlmostEqual(
            latest["moe_health/layer_0/load_max_median_ratio"], 2.5, places=5
        )
        self.assertAlmostEqual(
            latest["moe_health/layer_1/load_max_median_ratio"], 2.5, places=5
        )
        self.assertIn("moe_health/global_load_max_min_ratio", latest)

    def test_global_load_balance_skips_uninitialized_ga_steps(self):
        # A router whose global_aux_loss buffer has not accumulated yet
        # (ga_steps == 0) must be skipped, not divide-by-zero or emit garbage.
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        router = WeakRefable(
            global_tokens_per_expert=torch.zeros(4), ga_steps=torch.tensor(0.0)
        )
        monitor._global_lb_enabled = True
        monitor._load_balance_routers = [(0, weakref.ref(router))]
        for name in moe_monitor_module._LOAD_BALANCE_METRICS:
            monitor.declare_layer_metric(0, name)
        monitor.allocate_buffers(torch.device("cpu"))

        monitor._record_global_load_balance_metrics()
        monitor.step()

        latest = training_logs.get_latest(prefix="moe_health")
        self.assertNotIn("moe_health/layer_0/load_max_min_ratio", latest)

    def test_reset_temporary_tensors_patch_reads_before_zero(self):
        # The wrapper must rebind reset_model_temporary_tensors in
        # finalize_model_grads and read global_tokens_per_expert BEFORE the
        # original zeroes it.
        fmg = moe_monitor_module._finalize_model_grads
        original = fmg.reset_model_temporary_tensors

        router = WeakRefable(
            global_tokens_per_expert=torch.tensor(
                [6.0, 2.0, 3.0, 3.0]
            ),  # max/min=3, max/median=2
            ga_steps=torch.tensor(4.0),
        )

        # Stub original to zero the buffer, proving the wrapper read it first.
        def zeroing_reset(*_a, **_k):
            router.global_tokens_per_expert.zero_()
            router.ga_steps.zero_()

        fmg.reset_model_temporary_tensors = zeroing_reset

        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        monitor._global_lb_enabled = True
        monitor._load_balance_routers = [(0, weakref.ref(router))]
        for name in moe_monitor_module._LOAD_BALANCE_METRICS:
            monitor.declare_layer_metric(0, name)
        monitor.allocate_buffers(torch.device("cpu"))

        try:
            monitor._patch_reset_temporary_tensors()
            self.assertTrue(
                getattr(fmg.reset_model_temporary_tensors, "_im_patched", False)
            )

            fmg.reset_model_temporary_tensors(None, None)  # config, model
            # Buffer was zeroed by the original, proving order.
            self.assertEqual(float(router.global_tokens_per_expert.sum()), 0.0)
            monitor.step()

            latest = training_logs.get_latest(prefix="moe_health")
            self.assertAlmostEqual(
                latest["moe_health/layer_0/load_max_min_ratio"], 3.0, places=5
            )
            self.assertAlmostEqual(
                latest["moe_health/layer_0/load_max_median_ratio"],
                2.0,
                places=5,
            )
        finally:
            monitor.remove_hooks()
            fmg.reset_model_temporary_tensors = original

    def test_expert_bias_patch_rebinds_caller_and_fires(self):
        # The wrapper must rebind the name in finalize_model_grads (the caller),
        # observe the reduced tokens_per_expert, and unpatch cleanly.
        fmg = moe_monitor_module._finalize_model_grads
        original = fmg.get_updated_expert_bias
        # Stub the underlying update so the test needs no distributed group:
        # the real fn all-reduces tokens_per_expert, which requires dist init.
        # Our wrapper only cares that the tensor it receives is the reduced one.
        fmg.get_updated_expert_bias = lambda tpe, bias, rate, *a, **k: bias

        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        monitor._expert_bias_enabled = True
        monitor._load_balance_layer_order = [0]
        for name in moe_monitor_module._LOAD_BALANCE_METRICS:
            monitor.declare_layer_metric(0, name)
        monitor.allocate_buffers(torch.device("cpu"))

        try:
            monitor._patch_expert_bias_update()
            self.assertTrue(
                getattr(fmg.get_updated_expert_bias, "_im_patched", False)
            )

            # Caller hands the wrapper an ALREADY-reduced count row.
            tokens = torch.tensor(
                [[6.0, 2.0, 3.0, 3.0]]
            )  # max/min=6/2=3, max/median=6/3=2
            bias = torch.zeros_like(tokens)
            fmg.get_updated_expert_bias(tokens, bias, 0.0)
            monitor.step()

            latest = training_logs.get_latest(prefix="moe_health")
            self.assertAlmostEqual(
                latest["moe_health/layer_0/load_max_min_ratio"], 3.0, places=5
            )
            self.assertAlmostEqual(
                latest["moe_health/layer_0/load_max_median_ratio"],
                2.0,
                places=5,
            )
        finally:
            monitor.remove_hooks()
            fmg.get_updated_expert_bias = original


class MegatronPLEMonitorTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_global_hooks_are_disabled_when_log_global_is_false(self):
        monitor = PLEHealthMonitor(log_global=False)
        monitor._num_layers = 2
        monitor._hidden_size = 6
        monitor._hidden_size_ple = 3

        monitor._make_token_ple_hook()(None, None, torch.randn(2, 4, 6))
        monitor._make_proj_ple_hook()(None, None, torch.randn(2, 4, 6))

        self.assertIsNone(monitor._token_ple_buf)
        self.assertIsNone(monitor._proj_ple_buf)
        self.assertEqual(training_logs.get_latest(prefix="ple_health"), {})

    def test_layer_hook_records_residual_and_gate_metrics_as_one_observation(
        self,
    ):
        monitor = PLEHealthMonitor(
            log_per_layer=True, log_global=True, gate_sparsity_threshold=0.01
        )
        hidden_states = torch.ones(2, 3, 4)
        for name in ("residual_ratio", "gate_activation_mean", "gate_sparsity"):
            monitor.declare_layer_metric(5, name)
        monitor.allocate_buffers(hidden_states.device)

        monitor._gate_out_buf[5] = torch.ones(2, 3, 4)
        output = hidden_states * 1.5

        hook = monitor._make_ple_layer_hook(5, FakePLESublayer())
        hook(None, (hidden_states,), output)
        monitor.step()

        latest = training_logs.get_latest(prefix="ple_health")
        self.assertIn("ple_health/layer_5/residual_ratio", latest)
        self.assertIn("ple_health/layer_5/gate_activation_mean", latest)
        self.assertIn("ple_health/global_residual_ratio", latest)
        self.assertEqual(monitor._gate_out_buf, {})


class MegatronMassiveActivationMonitorTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_compute_and_log_records_pre_norm_metrics(self):
        monitor = MassiveActivationMonitor(
            log_per_layer=True,
            log_global=True,
            cosine_sample_pairs=4,
            absolute_thresholds=(2.0, 3.0),
        )
        hidden_states = torch.tensor(
            [
                [[1.0, -2.0, 0.5, 4.0]],
                [[3.0, 1.0, -0.5, 2.0]],
            ]
        )
        for name in monitor._layer_metric_names():
            monitor.declare_layer_metric(0, name)
        monitor.allocate_buffers(hidden_states.device)

        monitor._compute_residual_metrics(0, hidden_states)
        monitor.step()

        latest = training_logs.get_latest(prefix="massive_act")
        for key in (
            "channel_max",
            "channel_median",
            "channel_p95",
            "channel_p99",
            "channel_max_ratio",
            "massive_act_channel_count",
            "channel_count_gt_2",
            "channel_count_gt_3",
            "topk_channel_norm",
        ):
            self.assertIn(f"massive_act/layer_0/{key}", latest)
            self.assertIn(f"massive_act/global_{key}", latest)
        self.assertEqual(latest["massive_act/layer_0/channel_count_gt_2"], 2.0)
        self.assertEqual(latest["massive_act/layer_0/channel_count_gt_3"], 1.0)

    def test_spectral_norm_bounds_record_per_token_rms_ratio(self):
        monitor = MassiveActivationMonitor(
            log_per_layer=True,
            log_global=True,
            log_post_norm_metrics=False,
        )
        pre = torch.tensor(
            [
                [[1.0, -2.0, 0.5, 4.0]],
                [[3.0, 1.0, -0.5, 2.0]],
            ]
        )
        # post = 2 * pre => per-token RMS ratio is exactly 2.0 for every token,
        # so both the max and min bound collapse to 2.0.
        post = pre * 2.0
        for name in monitor._layer_metric_names():
            monitor.declare_layer_metric(0, name)
        monitor.allocate_buffers(pre.device)

        monitor._compute_spectral_norm(0, pre, post)
        monitor.step()

        latest = training_logs.get_latest(prefix="massive_act")
        for key in ("spectral_norm_max", "spectral_norm_min"):
            self.assertIn(f"massive_act/layer_0/{key}", latest)
            self.assertIn(f"massive_act/global_{key}", latest)
        self.assertAlmostEqual(
            latest["massive_act/layer_0/spectral_norm_max"], 2.0, places=5
        )
        self.assertAlmostEqual(
            latest["massive_act/layer_0/spectral_norm_min"], 2.0, places=5
        )
        # activation_rms is derived from the shared per-token pre-RMS, not a second
        # square of the input: it must equal sqrt(mean(pre**2)) over the whole input.
        self.assertIn("massive_act/layer_0/activation_rms", latest)
        expected_rms = (
            pre.reshape(-1, pre.shape[-1]).float().square().mean().sqrt().item()
        )
        self.assertAlmostEqual(
            latest["massive_act/layer_0/activation_rms"], expected_rms, places=5
        )
        # activation_rms_std is the std of the per-token pre-RMS (dispersion).
        self.assertIn("massive_act/layer_0/activation_rms_std", latest)
        expected_rms_std = (
            pre.reshape(-1, pre.shape[-1])
            .float()
            .square()
            .mean(dim=-1)
            .sqrt()
            .std()
            .item()
        )
        self.assertAlmostEqual(
            latest["massive_act/layer_0/activation_rms_std"],
            expected_rms_std,
            places=5,
        )

    def test_derived_activation_rms_matches_original_formula(self):
        # Regression guard: the merged/derived activation_rms (from the spectral
        # hook's per-token pre-RMS) must be numerically identical to the original
        # standalone compute_activation_scale_stats(h) over the same tensor.
        torch.manual_seed(0)
        hidden_states = (
            torch.randn(7, 3, 5) * 4.0 + 1.5
        )  # [S, B, H], non-trivial scale

        original = massive_activation_metrics.compute_activation_scale_stats(
            hidden_states
        )["activation_rms"]
        # activation_rms depends only on the pre tensor; post is irrelevant to it.
        derived = massive_activation_metrics.compute_spectral_norm_bounds(
            hidden_states,
            torch.zeros_like(hidden_states),
            include_activation_rms=True,
        )["activation_rms"]

        self.assertTrue(
            torch.allclose(original, derived, rtol=0, atol=1e-6),
            f"{original} != {derived}",
        )

    def test_grad_gain_bounds_equal_scaled_gradients(self):
        # grad_in = 2 * grad_out per token => the per-token ratio ‖dx‖/‖dy‖ is exactly
        # 2.0 for every token, so both the Lipschitz max and min bound collapse to 2.0.
        torch.manual_seed(0)
        grad_out = torch.randn(5, 3, 4)  # [S, B, H]
        grad_in = grad_out * 2.0

        bounds = massive_activation_metrics.compute_grad_gain_bounds(
            grad_in, grad_out
        )

        self.assertAlmostEqual(bounds["lipschitz_max"].item(), 2.0, places=5)
        self.assertAlmostEqual(bounds["lipschitz_min"].item(), 2.0, places=5)

    def test_grad_gain_hook_records_lipschitz_from_backward(self):
        # End-to-end: a scalar-mul layer y = 2*x has ∂L/∂x = 2·∂L/∂y for any loss, so
        # the backward-captured gradient-gain ratio is exactly 2.0 regardless of the
        # gradient direction. This also proves the tensor grad-hook path fires and
        # records (a module full_backward_hook would see an empty grad_input here,
        # since the layer is called all-keyword like Megatron does).
        monitor = MassiveActivationMonitor(
            log_per_layer=True,
            log_global=True,
            log_post_norm_metrics=False,
            log_activation_rms=False,
            log_lipschitz=True,
        )
        for name in monitor._layer_metric_names():
            monitor.declare_layer_metric(0, name)
        monitor.allocate_buffers(torch.device("cpu"))

        class ScalarMul(nn.Module):
            def forward(self, hidden_states):
                return (
                    hidden_states * 2.0,
                    None,
                )  # (output, context) like a Megatron layer

        layer = ScalarMul()
        hook = layer.register_forward_hook(
            monitor._make_grad_gain_hook(0), with_kwargs=True
        )

        x = torch.randn(3, 2, 4, requires_grad=True)  # [S, B, H]
        out, _ = layer(
            hidden_states=x
        )  # all-keyword call, as the Megatron block does
        out.sum().backward()
        hook.remove()

        monitor.step()

        latest = training_logs.get_latest(prefix="massive_act")
        for key in ("lipschitz_max", "lipschitz_min"):
            self.assertIn(f"massive_act/layer_0/{key}", latest)
            self.assertIn(f"massive_act/global_{key}", latest)
        self.assertAlmostEqual(
            latest["massive_act/layer_0/lipschitz_max"], 2.0, places=5
        )
        self.assertAlmostEqual(
            latest["massive_act/layer_0/lipschitz_min"], 2.0, places=5
        )

    def test_logit_lens_entropy_uniform_is_log_vocab(self):
        # Zero weight => every logit is 0 => uniform softmax => H == log(vocab).
        import math

        vocab, hidden = 8, 4
        h = torch.randn(3, 2, hidden)
        weight = torch.zeros(vocab, hidden)
        out = massive_activation_metrics.compute_logit_lens_entropy(
            h, weight, chunk_size=2
        )
        self.assertIn("logit_lens_entropy_mean", out)
        self.assertAlmostEqual(
            out["logit_lens_entropy_mean"].item(), math.log(vocab), places=5
        )

    def test_logit_lens_entropy_peaked_is_near_zero(self):
        # One vocab logit dominates => softmax ~ one-hot => H ~ 0.
        h = torch.tensor([[[1.0, 0.0]]])  # [s=1, b=1, hidden=2]
        weight = torch.tensor(
            [[100.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        )  # [vocab=3, hidden=2]
        out = massive_activation_metrics.compute_logit_lens_entropy(h, weight)
        self.assertLess(out["logit_lens_entropy_mean"].item(), 1e-3)
        self.assertGreaterEqual(out["logit_lens_entropy_mean"].item(), 0.0)

    def test_logit_lens_entropy_chunking_matches_single_pass(self):
        import math

        torch.manual_seed(0)
        vocab, hidden = 16, 5
        h = torch.randn(9, 2, hidden)
        weight = torch.randn(vocab, hidden)
        full = massive_activation_metrics.compute_logit_lens_entropy(
            h, weight, chunk_size=1000
        )
        chunked = massive_activation_metrics.compute_logit_lens_entropy(
            h, weight, chunk_size=4
        )
        for key in full:
            self.assertAlmostEqual(
                full[key].item(), chunked[key].item(), places=5
            )
        # Bounds: 0 <= mean H <= log(vocab).
        self.assertGreaterEqual(full["logit_lens_entropy_mean"].item(), -1e-5)
        self.assertLessEqual(
            full["logit_lens_entropy_mean"].item(), math.log(vocab) + 1e-4
        )

    def test_logit_lens_entropy_applies_final_norm(self):
        # A norm that zeros its input makes every logit 0 => uniform => log(vocab),
        # regardless of the (large) hidden values: proves final_norm is applied.
        import math

        vocab, hidden = 4, 3
        h = torch.randn(2, 2, hidden) * 10.0
        weight = torch.randn(vocab, hidden)
        zero_norm = nn.Linear(hidden, hidden, bias=False)
        with torch.no_grad():
            zero_norm.weight.zero_()
        out = massive_activation_metrics.compute_logit_lens_entropy(
            h, weight, final_norm=zero_norm
        )
        self.assertAlmostEqual(
            out["logit_lens_entropy_mean"].item(), math.log(vocab), places=5
        )

    def test_logit_lens_entropy_empty_input_is_zero(self):
        out = massive_activation_metrics.compute_logit_lens_entropy(
            torch.zeros(0, 4), torch.randn(6, 4)
        )
        for key in ("logit_lens_entropy_mean", "logit_lens_logsumexp_mean"):
            self.assertEqual(out[key].item(), 0.0)

    def test_logit_lens_logsumexp_uniform_equals_c_plus_log_vocab(self):
        # Constant logit c over vocab V => logsumexp = c + log(V). With a one-hot
        # weight column of value c and a one-hot hidden, every logit == c.
        import math

        vocab, hidden = 8, 4
        c = 3.0
        h = torch.zeros(3, 2, hidden)
        h[..., 0] = 1.0  # select column 0 of the weight
        weight = torch.zeros(vocab, hidden)
        weight[:, 0] = c  # every vocab logit == c
        out = massive_activation_metrics.compute_logit_lens_entropy(
            h, weight, chunk_size=2
        )
        expected = c + math.log(vocab)
        self.assertIn("logit_lens_logsumexp_mean", out)
        self.assertAlmostEqual(
            out["logit_lens_logsumexp_mean"].item(), expected, places=4
        )

    def test_logit_lens_entropy_end_to_end_records_keys(self):
        import math

        torch.manual_seed(0)
        num_layers, hidden, vocab = 2, 6, 10
        model = FakeGPTModel(num_layers, hidden, vocab)
        monitor = MassiveActivationMonitor(
            log_post_norm_metrics=False,
            log_activation_rms=False,
            log_lipschitz=False,
            log_logit_lens_entropy=True,
            logit_lens_chunk_size=3,
        )
        monitor.register_hooks(model)
        model(
            torch.randn(4, 2, hidden)
        )  # [S, B, H], grad enabled so _should_monitor fires
        monitor.step()

        latest = training_logs.get_latest(prefix="massive_act")
        for i in range(num_layers):
            for key in ("logit_lens_entropy_mean", "logit_lens_logsumexp_mean"):
                full_key = f"massive_act/layer_{i}/{key}"
                self.assertIn(full_key, latest)
                self.assertTrue(math.isfinite(latest[full_key]))
            mean_entropy = latest[
                f"massive_act/layer_{i}/logit_lens_entropy_mean"
            ]
            self.assertGreaterEqual(mean_entropy, -1e-4)
            self.assertLessEqual(mean_entropy, math.log(vocab) + 1e-3)
        self.assertIn("massive_act/global_logit_lens_entropy_mean", latest)
        self.assertIn("massive_act/global_logit_lens_logsumexp_mean", latest)
        monitor.remove_hooks()

    def test_logit_lens_entropy_absent_when_disabled(self):
        model = FakeGPTModel(2, 6, 10)
        monitor = MassiveActivationMonitor(
            log_post_norm_metrics=False,
            log_activation_rms=False,
            log_lipschitz=False,
            log_logit_lens_entropy=False,
        )
        monitor.register_hooks(model)
        model(torch.randn(4, 2, 6))
        monitor.step()

        latest = training_logs.get_latest(prefix="massive_act")
        self.assertFalse(any("logit_lens_entropy" in key for key in latest))
        monitor.remove_hooks()

    def test_logit_lens_entropy_no_head_stage_is_noop(self):
        # A PP stage without the LM head: _resolve_lm_head -> (None, None), no entropy
        # hook attaches, no entropy keys declared, and the forward runs without error.
        model = FakeHeadlessGPTModel(2, 6)
        monitor = MassiveActivationMonitor(
            log_post_norm_metrics=False,
            log_activation_rms=False,
            log_lipschitz=False,
            log_logit_lens_entropy=True,
        )
        self.assertEqual(monitor._resolve_lm_head(model), (None, None))
        monitor.register_hooks(model)
        model(torch.randn(4, 2, 6))
        monitor.step()

        latest = training_logs.get_latest(prefix="massive_act")
        self.assertFalse(any("logit_lens_entropy" in key for key in latest))
        monitor.remove_hooks()

    def test_logit_lens_entropy_layer_filter_restricts_declared_layers(self):
        # logit_lens_layers restricts the entropy metric to the listed global indices.
        model = FakeGPTModel(3, 6, 10)
        monitor = MassiveActivationMonitor(
            log_post_norm_metrics=False,
            log_activation_rms=False,
            log_lipschitz=False,
            log_logit_lens_entropy=True,
            logit_lens_layers=[1],
        )
        monitor.register_hooks(model)
        model(torch.randn(4, 2, 6))
        monitor.step()

        latest = training_logs.get_latest(prefix="massive_act")
        self.assertIn("massive_act/layer_1/logit_lens_entropy_mean", latest)
        self.assertNotIn("massive_act/layer_0/logit_lens_entropy_mean", latest)
        self.assertNotIn("massive_act/layer_2/logit_lens_entropy_mean", latest)
        monitor.remove_hooks()

    def test_logit_lens_cross_entropy_matches_reference(self):
        # CE via the logit lens must equal F.cross_entropy on the same logits.
        torch.manual_seed(0)
        tokens, hidden, vocab = 7, 5, 11
        h = torch.randn(tokens, hidden)
        weight = torch.randn(vocab, hidden)
        labels = torch.randint(0, vocab, (tokens,))
        out = massive_activation_metrics.compute_logit_lens_entropy(
            h, weight, labels=labels, want_entropy=False, chunk_size=3
        )
        self.assertIn("logit_lens_cross_entropy_mean", out)
        # want_entropy=False -> only the CE metric is produced.
        self.assertNotIn("logit_lens_entropy_mean", out)
        self.assertNotIn("logit_lens_logsumexp_mean", out)
        ref = F.cross_entropy((h @ weight.t()).float(), labels)
        self.assertAlmostEqual(
            out["logit_lens_cross_entropy_mean"].item(), ref.item(), places=4
        )

    def test_logit_lens_cross_entropy_peaked_correct_is_near_zero(self):
        # Logits sharply peaked at the correct label -> CE ~ 0.
        vocab = hidden = 8
        weight = torch.eye(vocab, hidden) * 50.0
        h = torch.eye(vocab, hidden)  # token i one-hot at coord i
        labels = torch.arange(vocab)
        out = massive_activation_metrics.compute_logit_lens_entropy(
            h, weight, labels=labels, want_entropy=False
        )
        self.assertLess(out["logit_lens_cross_entropy_mean"].item(), 1e-3)
        self.assertGreaterEqual(
            out["logit_lens_cross_entropy_mean"].item(), 0.0
        )

    def test_logit_lens_all_three_metrics_together(self):
        # entropy + logsumexp + cross-entropy share one projection when both flags on.
        torch.manual_seed(1)
        h = torch.randn(6, 4)
        weight = torch.randn(9, 4)
        labels = torch.randint(0, 9, (6,))
        out = massive_activation_metrics.compute_logit_lens_entropy(
            h, weight, labels=labels, want_entropy=True
        )
        for key in (
            "logit_lens_entropy_mean",
            "logit_lens_logsumexp_mean",
            "logit_lens_cross_entropy_mean",
        ):
            self.assertIn(key, out)
            self.assertTrue(math.isfinite(out[key].item()))

    def test_logit_lens_cross_entropy_label_count_mismatch_skips_ce(self):
        # A label/token count mismatch drops CE (no crash) but keeps entropy.
        h = torch.randn(6, 4)
        weight = torch.randn(9, 4)
        labels = torch.randint(0, 9, (5,))  # wrong count
        out = massive_activation_metrics.compute_logit_lens_entropy(
            h, weight, labels=labels, want_entropy=True
        )
        self.assertNotIn("logit_lens_cross_entropy_mean", out)
        self.assertIn("logit_lens_entropy_mean", out)

    def test_logit_lens_cross_entropy_end_to_end_matches_lm_loss(self):
        # End-to-end: the final layer's logit-lens CE equals the LM loss (final_norm +
        # head + F.cross_entropy) computed by hand, validating label alignment.
        torch.manual_seed(0)
        num_layers, hidden, vocab = 2, 6, 10
        seq, batch = 4, 2
        model = FakeGPTModel(num_layers, hidden, vocab)
        monitor = MassiveActivationMonitor(
            log_post_norm_metrics=False,
            log_activation_rms=False,
            log_lipschitz=False,
            log_logit_lens_entropy=False,
            log_logit_lens_cross_entropy=True,
            logit_lens_chunk_size=3,
        )
        monitor.register_hooks(model)
        x = torch.randn(seq, batch, hidden)  # [S, B, H]
        labels = torch.randint(0, vocab, (batch, seq))  # [B, S], like Megatron
        out_hidden = model(
            x, labels=labels
        )  # last layer output (== decoder return)
        monitor.step()

        latest = training_logs.get_latest(prefix="massive_act")
        for i in range(num_layers):
            key = f"massive_act/layer_{i}/logit_lens_cross_entropy_mean"
            self.assertIn(key, latest)
            self.assertTrue(math.isfinite(latest[key]))
        with torch.no_grad():
            final = model.decoder.final_layernorm(out_hidden)
            logits = final.reshape(-1, hidden) @ model.output_layer.weight.t()
            labels_aligned = labels.transpose(0, 1).reshape(
                -1
            )  # seq-major, matches hidden
            ref = F.cross_entropy(logits.float(), labels_aligned)
        self.assertAlmostEqual(
            latest[
                f"massive_act/layer_{num_layers - 1}/logit_lens_cross_entropy_mean"
            ],
            ref.item(),
            places=4,
        )
        self.assertIn(
            "massive_act/global_logit_lens_cross_entropy_mean", latest
        )
        monitor.remove_hooks()

    def test_logit_lens_cross_entropy_absent_when_disabled(self):
        model = FakeGPTModel(2, 6, 10)
        monitor = MassiveActivationMonitor(
            log_post_norm_metrics=False,
            log_activation_rms=False,
            log_lipschitz=False,
            log_logit_lens_cross_entropy=False,
        )
        monitor.register_hooks(model)
        model(torch.randn(4, 2, 6), labels=torch.randint(0, 10, (2, 4)))
        monitor.step()

        latest = training_logs.get_latest(prefix="massive_act")
        self.assertFalse(any("cross_entropy" in key for key in latest))
        monitor.remove_hooks()

    def test_hidden_spectral_entropy_rank_one_is_near_zero(self):
        # A rank-1 token set (all tokens on one direction) spans one effective
        # direction -> spectral entropy ~ 0.
        direction = torch.randn(8)
        coeffs = torch.randn(16, 1)
        h = coeffs * direction  # [16, 8], rank 1
        entropy = massive_activation_metrics.compute_hidden_spectral_entropy(h)
        self.assertLess(entropy.item(), 1e-4)
        self.assertGreaterEqual(entropy.item(), -1e-6)

    def test_hidden_spectral_entropy_orthonormal_rows_is_log_k(self):
        import math

        # k orthonormal rows -> k equal singular values -> uniform p_i -> H = log(k).
        k = 4
        h = torch.eye(k, 8)  # 4 orthonormal rows in R^8
        entropy = massive_activation_metrics.compute_hidden_spectral_entropy(h)
        self.assertAlmostEqual(entropy.item(), math.log(k), places=5)

    def test_hidden_spectral_entropy_empty_input_is_zero(self):
        entropy = massive_activation_metrics.compute_hidden_spectral_entropy(
            torch.zeros(0, 4)
        )
        self.assertEqual(entropy.item(), 0.0)

    def test_hidden_spectral_entropy_within_bounds(self):
        import math

        torch.manual_seed(0)
        n, d = 32, 8
        h = torch.randn(n, d)
        entropy = massive_activation_metrics.compute_hidden_spectral_entropy(h)
        self.assertGreaterEqual(entropy.item(), -1e-5)
        self.assertLessEqual(entropy.item(), math.log(min(n, d)) + 1e-4)

    def test_hidden_spectral_entropy_end_to_end_records_key(self):
        import math

        monitor = MassiveActivationMonitor(
            log_per_layer=True,
            log_global=True,
            log_post_norm_metrics=False,
            log_activation_rms=False,
            log_lipschitz=False,
            log_hidden_spectral_entropy=True,
        )
        self.assertIn("hidden_spectral_entropy", monitor._layer_metric_names())
        for name in monitor._layer_metric_names():
            monitor.declare_layer_metric(0, name)
        monitor.allocate_buffers(torch.device("cpu"))

        normalized = torch.randn(4, 2, 8)  # [S, B, H], post-RMSNorm hidden
        monitor._compute_spectral_entropy(0, normalized)
        monitor.step()

        latest = training_logs.get_latest(prefix="massive_act")
        self.assertIn("massive_act/layer_0/hidden_spectral_entropy", latest)
        self.assertIn("massive_act/global_hidden_spectral_entropy", latest)
        value = latest["massive_act/layer_0/hidden_spectral_entropy"]
        self.assertTrue(math.isfinite(value))
        self.assertGreaterEqual(value, -1e-5)
        self.assertLessEqual(value, math.log(8) + 1e-4)

    def test_hidden_spectral_entropy_absent_when_disabled(self):
        monitor = MassiveActivationMonitor(log_hidden_spectral_entropy=False)
        self.assertNotIn(
            "hidden_spectral_entropy", monitor._layer_metric_names()
        )


class SinkHeadClassificationTest(unittest.TestCase):
    """The gap computation is branchless to avoid a GPU->CPU sync on the hot
    path (Python comparisons on a tensor sink_count would .item()). These cases
    pin the branchless result against a readable branched reference.
    """

    THRESHOLD = 0.3

    def _reference_gap(self, sink_per_head):
        is_sink = sink_per_head > self.THRESHOLD
        num_heads = sink_per_head.numel()
        sink_count = int(is_sink.sum())
        if 0 < sink_count < num_heads:
            return (
                sink_per_head[is_sink].mean() - sink_per_head[~is_sink].mean()
            ).item()
        if sink_count == num_heads:
            return sink_per_head.mean().item()
        return 0.0

    def _assert_gap(self, sink_per_head):
        result = compute_sink_head_classification(
            sink_per_head, threshold=self.THRESHOLD
        )
        self.assertAlmostEqual(
            result["sink_nonsink_gap"].item(),
            self._reference_gap(sink_per_head),
            places=5,
        )

    def test_mixed_sink_and_nonsink(self):
        self._assert_gap(torch.tensor([0.5, 0.1, 0.8, 0.05]))

    def test_all_heads_are_sinks(self):
        self._assert_gap(torch.tensor([0.5, 0.6, 0.9]))

    def test_no_sinks(self):
        self._assert_gap(torch.tensor([0.1, 0.2, 0.05]))

    def test_single_head(self):
        self._assert_gap(torch.tensor([0.9]))
        self._assert_gap(torch.tensor([0.1]))

    def test_empty_input_is_zero(self):
        result = compute_sink_head_classification(
            torch.tensor([]), threshold=self.THRESHOLD
        )
        self.assertEqual(result["sink_nonsink_gap"].item(), 0.0)
        self.assertEqual(result["sink_head_ratio"].item(), 0.0)


class _FakeDistOpt:
    """Stand-in for DistributedOptimizer's main/model param pairing.

    Mirrors the real invariant the monitor depends on: ``shard_fp32_from_float16_groups``
    holds the fp32 master (already stepped to ``theta_new``) while
    ``shard_float16_groups`` still holds the bf16 ``theta_old`` until
    ``_copy_main_params_to_model_params`` runs.
    """

    is_stub_optimizer = False

    def __init__(self, theta_old: "torch.Tensor", dtype=None):
        dtype = dtype or torch.bfloat16
        # The fp32 master is the authoritative value and is NOT on the bf16 grid — it
        # persists across steps and accumulates sub-bf16 updates. The model param is its
        # rounded copy. Seeding the master FROM the bf16 param instead would put it
        # exactly on-grid, removing the rounding error the debias exists to cancel and
        # making these tests measure a state training never reaches.
        self.main_param = theta_old.detach().clone().float()
        self.model_param = self.main_param.to(dtype)
        self.shard_float16_groups = [[self.model_param]]
        self.shard_fp32_from_float16_groups = [[self.main_param]]
        self.copy_calls = 0

    def apply_update(self, delta: "torch.Tensor") -> None:
        """The inner ``optimizer.step()``: fp32 master moves, model param does not."""
        self.main_param.add_(delta)

    def _copy_main_params_to_model_params(self):
        self.copy_calls += 1
        self.model_param.copy_(self.main_param)


class MegatronOptimUpdateMonitorTest(unittest.TestCase):
    """optim/update_rms + param_rms: measured between optimizer.step() and copy-back."""

    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def _run_step(self, theta_old, delta, **kwargs):
        opt = _FakeDistOpt(theta_old)
        monitor = OptimUpdateMonitor(**kwargs)
        self.assertTrue(
            monitor.attach_optimizer(opt), "no copy method was wrapped"
        )
        opt.apply_update(delta)
        opt._copy_main_params_to_model_params()
        try:
            return opt, monitor.step()
        finally:
            monitor.remove_hooks()

    def test_reports_update_and_param_rms(self):
        torch.manual_seed(0)
        theta = torch.randn(50_000) * 0.02
        delta = torch.randn(50_000) * 3e-4
        opt, got = self._run_step(theta, delta)

        want_update = float(delta.pow(2).mean().sqrt())
        want_param = float(opt.main_param.pow(2).mean().sqrt())
        self.assertAlmostEqual(
            got["optim/update_rms"] / want_update, 1.0, places=2
        )
        self.assertAlmostEqual(
            got["optim/param_rms"] / want_param, 1.0, places=4
        )
        self.assertAlmostEqual(
            got["optim/update_param_ratio"],
            got["optim/update_rms"] / got["optim/param_rms"],
            places=9,
        )

    def test_metrics_land_in_training_logs(self):
        torch.manual_seed(1)
        self._run_step(torch.randn(4096) * 0.02, torch.randn(4096) * 1e-4)
        latest = training_logs.get_latest(prefix="optim")
        self.assertEqual(
            set(latest),
            {"optim/update_rms", "optim/param_rms", "optim/update_param_ratio"},
        )

    def test_debias_keeps_small_updates_accurate(self):
        """A bf16 theta_old rounds, which inflates the raw difference once the update is
        near the bf16 resolution — 11x at lr 3e-6. Debiasing must hold ~1.0x across the
        whole lr range, otherwise the metric is useless late in a cosine decay."""
        torch.manual_seed(2)
        for lr in (3e-4, 3e-5, 3e-6):
            theta = torch.randn(200_000) * 0.02
            delta = torch.randn(200_000) * lr
            want = float(delta.pow(2).mean().sqrt())
            with self.subTest(lr=lr, debias=True):
                _, got = self._run_step(theta, delta, debias_low_precision=True)
                self.assertAlmostEqual(
                    got["optim/update_rms"] / want, 1.0, places=1
                )
        # And confirm the raw form really is badly biased at the small end, so the
        # debias branch is not dead weight.
        theta = torch.randn(200_000) * 0.02
        delta = torch.randn(200_000) * 3e-6
        want = float(delta.pow(2).mean().sqrt())
        _, raw = self._run_step(theta, delta, debias_low_precision=False)
        self.assertGreater(raw["optim/update_rms"] / want, 3.0)

    def test_zero_update_reports_zero(self):
        """No update at all must not produce a floor from bf16 rounding."""
        torch.manual_seed(3)
        theta = torch.randn(20_000) * 0.02
        _, got = self._run_step(theta, torch.zeros(20_000))
        self.assertLess(got["optim/update_rms"], 1e-6)
        self.assertGreater(got["optim/param_rms"], 0.0)

    def test_copy_back_still_happens(self):
        """The wrapper must be transparent: the real copy has to run, exactly once."""
        torch.manual_seed(4)
        theta = torch.randn(8192) * 0.02
        delta = torch.randn(8192) * 1e-4
        opt, _ = self._run_step(theta, delta)
        self.assertEqual(opt.copy_calls, 1)
        self.assertTrue(
            torch.equal(
                opt.model_param, opt.main_param.to(opt.model_param.dtype)
            )
        )

    def test_patch_is_reverted_by_remove_hooks(self):
        opt = _FakeDistOpt(torch.randn(1024) * 0.02)
        original = opt._copy_main_params_to_model_params
        monitor = OptimUpdateMonitor()
        monitor.attach_optimizer(opt)
        self.assertIsNot(opt._copy_main_params_to_model_params, original)
        monitor.remove_hooks()
        # Reverted by dropping the instance attribute so class lookup resumes; bound
        # methods are recreated per access, so compare __func__.
        self.assertNotIn("_copy_main_params_to_model_params", vars(opt))
        self.assertIs(
            opt._copy_main_params_to_model_params.__func__, original.__func__
        )

    def test_patch_is_idempotent(self):
        opt = _FakeDistOpt(torch.randn(1024) * 0.02)
        monitor = OptimUpdateMonitor()
        monitor.attach_optimizer(opt)
        patched_once = opt._copy_main_params_to_model_params
        monitor.attach_optimizer(opt)
        try:
            self.assertIs(opt._copy_main_params_to_model_params, patched_once)
            self.assertEqual(len(monitor._patched), 1)
        finally:
            monitor.remove_hooks()

    def test_accumulates_across_chained_optimizers(self):
        """ChainedOptimizer (Muon + Adam) must contribute both sub-optimizers' shards to
        one pooled RMS, not just the first."""
        torch.manual_seed(5)
        theta_a, theta_b = (
            torch.randn(30_000) * 0.02,
            torch.randn(30_000) * 0.02,
        )
        delta_a, delta_b = (
            torch.randn(30_000) * 3e-4,
            torch.randn(30_000) * 3e-4,
        )
        opt_a, opt_b = _FakeDistOpt(theta_a), _FakeDistOpt(theta_b)
        chained = SimpleNamespace(chained_optimizers=[opt_a, opt_b])

        monitor = OptimUpdateMonitor()
        self.assertTrue(monitor.attach_optimizer(chained))
        try:
            opt_a.apply_update(delta_a)
            opt_b.apply_update(delta_b)
            opt_a._copy_main_params_to_model_params()
            opt_b._copy_main_params_to_model_params()
            got = monitor.step()
        finally:
            monitor.remove_hooks()

        want = float(torch.cat([delta_a, delta_b]).pow(2).mean().sqrt())
        self.assertAlmostEqual(got["optim/update_rms"] / want, 1.0, places=2)

    def test_step_is_empty_without_a_step(self):
        """step() with nothing pending must emit nothing rather than 0/0."""
        opt = _FakeDistOpt(torch.randn(512) * 0.02)
        monitor = OptimUpdateMonitor()
        monitor.attach_optimizer(opt)
        try:
            self.assertEqual(monitor.step(), {})
            self.assertEqual(training_logs.get_latest(prefix="optim"), {})
        finally:
            monitor.remove_hooks()

    def test_pending_is_cleared_between_steps(self):
        """Two consecutive steps must report each step's own update, not a running sum."""
        torch.manual_seed(6)
        opt = _FakeDistOpt(torch.randn(40_000) * 0.02)
        monitor = OptimUpdateMonitor()
        monitor.attach_optimizer(opt)
        try:
            big = torch.randn(40_000) * 1e-3
            opt.apply_update(big)
            opt._copy_main_params_to_model_params()
            first = monitor.step()["optim/update_rms"]

            small = torch.randn(40_000) * 1e-4
            opt.apply_update(small)
            opt._copy_main_params_to_model_params()
            second = monitor.step()["optim/update_rms"]
        finally:
            monitor.remove_hooks()

        self.assertAlmostEqual(
            first / float(big.pow(2).mean().sqrt()), 1.0, places=1
        )
        self.assertAlmostEqual(
            second / float(small.pow(2).mean().sqrt()), 1.0, places=1
        )
        self.assertLess(second, first / 5)

    def test_stub_optimizer_is_skipped(self):
        opt = _FakeDistOpt(torch.randn(512) * 0.02)
        opt.is_stub_optimizer = True
        monitor = OptimUpdateMonitor()
        monitor.attach_optimizer(opt)
        try:
            opt.apply_update(torch.randn(512) * 1e-4)
            opt._copy_main_params_to_model_params()
            self.assertEqual(monitor.step(), {})
        finally:
            monitor.remove_hooks()

    # ---------------- monitor_interval gating ----------------

    def test_reports_only_on_monitored_steps(self):
        """With monitor_interval=3 only every 3rd step reports. These are slow-moving
        scalars, so per-step reporting is just log noise plus two collectives."""
        torch.manual_seed(20)
        opt = _FakeDistOpt(torch.randn(8192) * 0.02)
        monitor = OptimUpdateMonitor(monitor_interval=3)
        monitor.attach_optimizer(opt)
        reported = []
        try:
            for _ in range(7):
                opt.apply_update(torch.randn(8192) * 3e-4)
                opt._copy_main_params_to_model_params()
                reported.append(bool(monitor.step()))
        finally:
            monitor.remove_hooks()
        # step_count runs 0,1,..; the gate fires when count % 3 == 0, i.e. steps 1, 4, 7.
        self.assertEqual(
            reported, [True, False, False, True, False, False, True]
        )

    def test_skipped_step_does_no_work_at_all(self):
        """The gate lives in the WRAPPER, not just at report time: on a skipped step
        nothing is accumulated, so there is no sum to reduce and no collective to issue."""
        torch.manual_seed(21)
        opt = _FakeDistOpt(torch.randn(4096) * 0.02)
        monitor = OptimUpdateMonitor(monitor_interval=5)
        monitor.attach_optimizer(opt)
        try:
            monitor.step_count = 2  # 2 % 5 != 0 -> unmonitored
            opt.apply_update(torch.randn(4096) * 3e-4)
            opt._copy_main_params_to_model_params()
            self.assertEqual(
                monitor._pending_dense, [], "wrapper must not accumulate"
            )
            self.assertEqual(monitor._pending_expert, [])
            self.assertEqual(monitor.step(), {})
            self.assertEqual(training_logs.get_latest(prefix="optim"), {})
        finally:
            monitor.remove_hooks()

    def test_copy_back_still_happens_on_skipped_steps(self):
        """Gating must never change the model: the real copy has to run either way."""
        torch.manual_seed(22)
        opt = _FakeDistOpt(torch.randn(2048) * 0.02)
        monitor = OptimUpdateMonitor(monitor_interval=100)
        monitor.attach_optimizer(opt)
        try:
            monitor.step_count = 7  # unmonitored
            opt.apply_update(torch.randn(2048) * 1e-4)
            opt._copy_main_params_to_model_params()
            self.assertEqual(opt.copy_calls, 1)
            self.assertTrue(
                torch.equal(
                    opt.model_param, opt.main_param.to(opt.model_param.dtype)
                )
            )
        finally:
            monitor.remove_hooks()

    def test_global_step_syncs_the_gate(self):
        """The trainer passes its own iteration via step(global_step=...); the gate must
        follow the trainer's counter so the interval lines up with the log interval."""
        opt = _FakeDistOpt(torch.randn(512) * 0.02)
        monitor = OptimUpdateMonitor(monitor_interval=10)
        monitor.attach_optimizer(opt)
        try:
            monitor.step(global_step=49)
            self.assertTrue(
                monitor._should_monitor(), "50 % 10 == 0 -> monitored"
            )
            monitor.step(global_step=50)
            self.assertFalse(
                monitor._should_monitor(), "51 % 10 != 0 -> skipped"
            )
        finally:
            monitor.remove_hooks()

    def test_setup_helper_threads_monitor_interval(self):
        opt = _FakeDistOpt(torch.randn(512) * 0.02)
        monitor_dict = {}
        setup_optim_update_monitor(
            None, optimizer=opt, monitor_dict=monitor_dict, monitor_interval=50
        )
        monitor = monitor_dict[optim_update_module.METRIC_PREFIX]
        try:
            self.assertEqual(monitor.monitor_interval, 50)
        finally:
            monitor.remove_hooks()

    def test_setup_helper_registers_under_the_metric_prefix(self):
        """The monitor_dict key must equal METRIC_PREFIX ("optim"), because callers print
        training_logs by iterating monitor_dict keys as metric prefixes — a mismatched key
        would silently drop these metrics from the log. The helper also returns ``model``,
        matching the other setup_* functions' contract."""
        opt = _FakeDistOpt(torch.randn(1024) * 0.02)
        monitor_dict = {}
        sentinel = object()
        returned = setup_optim_update_monitor(
            sentinel, optimizer=opt, monitor_dict=monitor_dict
        )
        monitor = monitor_dict[optim_update_module.METRIC_PREFIX]
        try:
            self.assertIs(
                returned,
                sentinel,
                "must return the model, like the other setups",
            )
            self.assertEqual(
                list(monitor_dict), [optim_update_module.METRIC_PREFIX]
            )
            self.assertTrue(monitor._patched)

            opt.apply_update(torch.randn(1024) * 3e-4)
            opt._copy_main_params_to_model_params()
            monitor.step()
            emitted = training_logs.get_latest(
                prefix=optim_update_module.METRIC_PREFIX
            )
            self.assertEqual(len(emitted), 3)
            for key in emitted:
                self.assertTrue(
                    key.startswith(f"{optim_update_module.METRIC_PREFIX}/")
                )
        finally:
            monitor.remove_hooks()

    def test_always_on_without_being_named_in_monitors(self):
        """The monitor must be installed by setup_monitors regardless of the ``monitors``
        spec — it is in _ALWAYS_ON_MONITORS, not _MONITOR_MAP — so no training script has
        to name it or thread the optimizer through a callback."""
        self.assertIn("optim", megatron_backend._ALWAYS_ON_MONITORS)
        self.assertNotIn("optim", megatron_backend._MONITOR_MAP)

        monitor_dict = {}
        model = FakeGPTModel(num_layers=1, hidden_size=8, vocab_size=16)
        megatron_backend.setup_monitors(
            model, monitors=["qk_stats"], monitor_dict=monitor_dict
        )
        monitor = monitor_dict.get("optim")
        try:
            self.assertIsInstance(monitor, OptimUpdateMonitor)
            self.assertTrue(
                monitor._patched, "class-level patch should be installed"
            )
        finally:
            if monitor is not None:
                monitor.remove_hooks()

    def test_class_patch_covers_optimizers_built_afterwards(self):
        """Patching the mcore optimizer CLASS is what removes the need for a trainer-side
        callback: an instance constructed after setup must still be measured."""
        monitor = OptimUpdateMonitor()
        if not monitor.attach_optimizer_classes():
            self.skipTest("megatron.core optimizer classes unavailable")
        try:
            from megatron.core.optimizer.distrib_optimizer import (
                DistributedOptimizer,
            )

            patched = DistributedOptimizer.__dict__[
                "_copy_main_params_to_model_params"
            ]
            self.assertTrue(getattr(patched, "_im_update_patched", False))
        finally:
            monitor.remove_hooks()
            from megatron.core.optimizer.distrib_optimizer import (
                DistributedOptimizer,
            )

            reverted = DistributedOptimizer.__dict__[
                "_copy_main_params_to_model_params"
            ]
            self.assertFalse(
                getattr(reverted, "_im_update_patched", False),
                "class must be restored",
            )

    def test_chunked_sum_matches_single_pass(self):
        """The shard is walked in fixed-size chunks to bound the temporary; the result
        must equal a single-pass computation across a chunk boundary."""
        torch.manual_seed(7)
        n = optim_update_module._CHUNK + 12345
        main = torch.randn(n)
        model = main.to(torch.bfloat16)
        ss_u, ss_p, ss_q, count = optim_update_module._pair_sums(main, model)
        self.assertEqual(count, n)
        # fp32 chunk accumulation vs a single fp32 reduction over ~1M elements: agreement
        # is fp32 round-off, not exact.
        self.assertAlmostEqual(
            float(ss_u) / float((main - model.float()).pow(2).sum()),
            1.0,
            places=3,
        )
        self.assertAlmostEqual(
            float(ss_p) / float(main.pow(2).sum()), 1.0, places=4
        )

    def test_tp_duplicate_params_are_counted_once(self):
        """A TP-REPLICATED param exists identically on every TP rank, so only rank 0 may
        count it; a TP-SHARDED param is distinct per rank and always counts. Getting this
        backwards inflates the RMS by the TP size."""
        replicated = torch.nn.Parameter(torch.randn(64))
        sharded = torch.nn.Parameter(torch.randn(64))
        sharded.tensor_model_parallel = True

        rank0 = SimpleNamespace(rank=lambda: 0)
        rank1 = SimpleNamespace(rank=lambda: 1)
        not_dup = optim_update_module._param_is_not_tp_duplicate
        self.assertTrue(
            not_dup(sharded, rank1), "TP-sharded is distinct on every rank"
        )
        self.assertTrue(not_dup(replicated, rank0))
        self.assertFalse(
            not_dup(replicated, rank1),
            "replicated must only count on TP rank 0",
        )

    def test_shared_params_are_excluded(self):
        """Tied embeddings are marked ``shared`` and appear on two PP stages; counting
        both would double-count them."""
        plain = torch.nn.Parameter(torch.randn(8))
        shared = torch.nn.Parameter(torch.randn(8))
        shared.shared = True
        self.assertTrue(optim_update_module._param_is_not_shared(plain))
        self.assertFalse(optim_update_module._param_is_not_shared(shared))

    def test_expert_and_dense_shards_are_bucketed_separately(self):
        """Expert params reduce over different groups than dense ones, so they must land
        in separate buckets (``allreduce=False`` is mcore's expert marker)."""
        torch.manual_seed(8)
        opt = _FakeDistOpt(torch.randn(4096) * 0.02)
        expert_model = (torch.randn(4096) * 0.02).to(torch.bfloat16)
        expert_main = expert_model.float()
        expert_model.allreduce = False
        opt.shard_fp32_from_float16_groups[0].append(expert_main)
        opt.shard_float16_groups[0].append(expert_model)

        monitor = OptimUpdateMonitor()
        monitor.attach_optimizer(opt)
        try:
            opt.apply_update(torch.randn(4096) * 3e-4)
            expert_main.add_(torch.randn(4096) * 3e-4)
            opt._copy_main_params_to_model_params()
            self.assertEqual(len(monitor._pending_dense), 1)
            self.assertEqual(len(monitor._pending_expert), 1)
            got = monitor.step()
            self.assertIn("optim/update_rms", got)
            self.assertEqual(monitor._pending_dense, [])
            self.assertEqual(monitor._pending_expert, [])
        finally:
            monitor.remove_hooks()


if __name__ == "__main__":
    unittest.main()
