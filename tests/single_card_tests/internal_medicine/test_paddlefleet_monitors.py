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
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

importlib.import_module("_backend_env").skip_unless_backend("paddlefleet")

try:
    paddle = importlib.import_module("paddle")
    nn = importlib.import_module("paddle.nn")
except (
    Exception
) as exc:  # pragma: no cover - depends on optional backend install
    raise unittest.SkipTest(f"paddle backend unavailable: {exc}") from exc

PaddleMassiveActivationMonitor = importlib.import_module(
    "paddlefleet.internal_medicine.backends.paddlefleet.massive_activation_monitor"
).PaddleMassiveActivationMonitor
PaddleMoEMonitor = importlib.import_module(
    "paddlefleet.internal_medicine.backends.paddlefleet.moe_monitor"
).PaddleMoEMonitor
moe_monitor_module = importlib.import_module(
    "paddlefleet.internal_medicine.backends.paddlefleet.moe_monitor"
)
attn_update_module = importlib.import_module(
    "paddlefleet.internal_medicine.backends.paddlefleet.attn_update_monitor"
)
PaddleAttnUpdateMonitor = attn_update_module.PaddleAttnUpdateMonitor
mlp_update_module = importlib.import_module(
    "paddlefleet.internal_medicine.backends.paddlefleet.mlp_update_monitor"
)
PaddleMLPUpdateMonitor = mlp_update_module.PaddleMLPUpdateMonitor
qk_monitor_module = importlib.import_module(
    "paddlefleet.internal_medicine.backends.paddlefleet.qk_monitor"
)
PaddleQKStatsMonitor = qk_monitor_module.PaddleQKStatsMonitor
paddlefleet_backend = importlib.import_module(
    "paddlefleet.internal_medicine.backends.paddlefleet"
)
layer_discovery = importlib.import_module(
    "paddlefleet.internal_medicine.backends.paddlefleet.layer_discovery"
)
training_logs = importlib.import_module(
    "paddlefleet.internal_medicine.core.training_logs"
).training_logs
TrainingLogs = importlib.import_module(
    "paddlefleet.internal_medicine.core.training_logs"
).TrainingLogs


class BrokenPaddleMoELayer:
    training = True

    @property
    def grouped_gemm_experts(self):
        raise RuntimeError("grouped expert read failed")


class DummyHook:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class DummyMonitor:
    def __init__(self):
        self.hooks = [DummyHook()]
        self.removed = False

    def remove_hooks(self):
        self.removed = True
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class PaddleFleetSetupTest(unittest.TestCase):
    def setUp(self):
        self.original_monitor_map = dict(paddlefleet_backend._MONITOR_MAP)

    def tearDown(self):
        paddlefleet_backend._MONITOR_MAP.clear()
        paddlefleet_backend._MONITOR_MAP.update(self.original_monitor_map)

    def test_setup_monitors_reuses_existing_monitor_for_same_config(self):
        created = []

        def setup_dummy(_model, monitor_dict=None, **_kwargs):
            monitor = DummyMonitor()
            created.append(monitor)
            monitor_dict["dummy"] = monitor

        paddlefleet_backend._MONITOR_MAP.clear()
        paddlefleet_backend._MONITOR_MAP["dummy"] = setup_dummy
        model = SimpleNamespace()
        first = {}
        second = {}

        paddlefleet_backend.setup_monitors(
            model, monitors=["dummy"], monitor_dict=first, monitor_interval=2
        )
        paddlefleet_backend.setup_monitors(
            model, monitors=["dummy"], monitor_dict=second, monitor_interval=2
        )

        self.assertEqual(len(created), 1)
        self.assertIs(second["dummy"], first["dummy"])
        self.assertFalse(first["dummy"].removed)

    def test_setup_monitors_replaces_existing_monitor_when_config_changes(self):
        created = []

        def setup_dummy(_model, monitor_dict=None, **_kwargs):
            monitor = DummyMonitor()
            created.append(monitor)
            monitor_dict["dummy"] = monitor

        paddlefleet_backend._MONITOR_MAP.clear()
        paddlefleet_backend._MONITOR_MAP["dummy"] = setup_dummy
        model = SimpleNamespace()
        first = {}
        second = {}

        paddlefleet_backend.setup_monitors(
            model, monitors=["dummy"], monitor_dict=first, monitor_interval=1
        )
        paddlefleet_backend.setup_monitors(
            model, monitors=["dummy"], monitor_dict=second, monitor_interval=2
        )

        self.assertEqual(len(created), 2)
        self.assertTrue(first["dummy"].removed)
        self.assertIs(second["dummy"], created[1])

    def test_setup_monitors_accepts_comma_separated_names(self):
        enabled = []

        def setup_dummy(
            _model, monitor_dict=None, monitor_name=None, **_kwargs
        ):
            enabled.append(monitor_name)
            monitor_dict[monitor_name] = DummyMonitor()

        paddlefleet_backend._MONITOR_MAP.clear()
        paddlefleet_backend._MONITOR_MAP.update(
            {
                "first": lambda model, **kwargs: setup_dummy(
                    model, monitor_name="first", **kwargs
                ),
                "second": lambda model, **kwargs: setup_dummy(
                    model, monitor_name="second", **kwargs
                ),
            }
        )

        paddlefleet_backend.setup_monitors(
            SimpleNamespace(),
            monitors=" first, second, first ",
            monitor_dict={},
        )

        self.assertEqual(enabled, ["first", "second"])

    def test_setup_monitors_propagates_and_cleans_setup_failure(self):
        partial = DummyMonitor()

        def setup_broken(_model, monitor_dict=None, **_kwargs):
            monitor_dict["broken"] = partial
            raise RuntimeError("setup failed")

        paddlefleet_backend._MONITOR_MAP.clear()
        paddlefleet_backend._MONITOR_MAP["broken"] = setup_broken
        model = SimpleNamespace()
        monitors = {}

        with self.assertRaisesRegex(RuntimeError, "setup failed"):
            paddlefleet_backend.setup_monitors(
                model, monitors=["broken"], monitor_dict=monitors
            )

        self.assertTrue(partial.removed)
        self.assertNotIn("broken", monitors)
        self.assertNotIn(
            "broken", getattr(model, paddlefleet_backend._MODEL_MONITOR_ATTR)
        )


class PaddleLayerDiscoveryTest(unittest.TestCase):
    def test_get_decoder_layers_flattens_virtual_pipeline_chunks(self):
        layer0 = SimpleNamespace(layer_idx=0)
        layer1 = SimpleNamespace(layer_idx=1)
        layer2 = SimpleNamespace(layer_idx=2)
        model = SimpleNamespace(
            _model_chunks=[
                SimpleNamespace(run_function=[layer0, layer1]),
                SimpleNamespace(run_function=[layer2]),
            ]
        )

        self.assertEqual(
            layer_discovery.get_decoder_layers(model), [layer0, layer1, layer2]
        )

    def test_get_decoder_layers_checks_wrapped_module_after_empty_layers_wrapper(
        self,
    ):
        layer = SimpleNamespace(layer_idx=0)
        model = SimpleNamespace(
            _layers=SimpleNamespace(),
            module=SimpleNamespace(run_function=[layer]),
        )

        self.assertEqual(layer_discovery.get_decoder_layers(model), [layer])

    def test_classify_attn_type_reads_is_swa_flag(self):
        """classify_attn_type returns swa/full based on self_attn.is_swa, None if absent."""
        swa_layer = SimpleNamespace(self_attn=SimpleNamespace(is_swa=True))
        full_layer = SimpleNamespace(self_attn=SimpleNamespace(is_swa=False))
        untagged_layer = SimpleNamespace(
            self_attn=SimpleNamespace()
        )  # no is_swa
        no_attn_layer = SimpleNamespace()  # no self_attn at all

        self.assertEqual(layer_discovery.classify_attn_type(swa_layer), "swa")
        self.assertEqual(layer_discovery.classify_attn_type(full_layer), "full")
        self.assertIsNone(layer_discovery.classify_attn_type(untagged_layer))
        self.assertIsNone(layer_discovery.classify_attn_type(no_attn_layer))

    def test_classify_attn_type_falls_back_to_self_attention(self):
        """Some models expose the module as ``self_attention`` instead of ``self_attn``."""
        swa_layer = SimpleNamespace(self_attention=SimpleNamespace(is_swa=True))
        self.assertEqual(layer_discovery.classify_attn_type(swa_layer), "swa")

    def test_iter_monitor_layers_populates_attn_type_for_swa_and_full(self):
        """MonitorLayer.attn_type is populated per layer for models that mix SWA and full."""
        # Simulate ernie-lite: layer 0 full, layer 1 SWA, layer 2 full
        full_layer_a = SimpleNamespace(self_attn=SimpleNamespace(is_swa=False))
        swa_layer = SimpleNamespace(self_attn=SimpleNamespace(is_swa=True))
        full_layer_b = SimpleNamespace(self_attn=SimpleNamespace(is_swa=False))

        result = layer_discovery.iter_monitor_layers(
            [full_layer_a, swa_layer, full_layer_b],
            lambda layer: hasattr(layer, "self_attn"),
        )

        self.assertEqual(
            [item.attn_type for item in result], ["full", "swa", "full"]
        )

    def test_iter_monitor_layers_attn_type_is_none_for_backward_compat_models(
        self,
    ):
        """Stacks with neither sliding_window nor csa_compress_ratios yield attn_type=None."""
        layer = SimpleNamespace(self_attn=SimpleNamespace())  # no is_swa
        result = layer_discovery.iter_monitor_layers(
            [layer], lambda x: hasattr(x, "self_attn")
        )
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0].attn_type)

    def test_iter_monitor_layers_marks_unwrapped_mtp_layer(self):
        config = SimpleNamespace(num_hidden_layers=1)
        main_layer = SimpleNamespace(
            self_attn=SimpleNamespace(is_swa=False), config=config
        )
        mtp_layer = SimpleNamespace(
            self_attn=SimpleNamespace(is_swa=False), config=config
        )
        mtp_wrapper = SimpleNamespace(
            transformer_layer=mtp_layer, config=config
        )

        result = layer_discovery.iter_monitor_layers(
            [main_layer, mtp_wrapper],
            lambda layer: hasattr(layer, "self_attn"),
        )

        self.assertEqual([item.idx for item in result], [0, 1])
        self.assertEqual([item.is_mtp for item in result], [False, True])

    def test_resolve_layer_idx_strips_head_empty_layer_offset(self):
        """PaddleFleet sets layer_number = logical_idx + num_empty_layers_add_in_head."""
        config = SimpleNamespace(num_empty_layers_add_in_head=2)
        layer = SimpleNamespace(layer_number=5, config=config)
        self.assertEqual(layer_discovery.resolve_layer_idx(layer, 0, 4), 3)

        # No head empty layers (the common case) keeps the raw number.
        plain = SimpleNamespace(layer_number=5, config=SimpleNamespace())
        self.assertEqual(layer_discovery.resolve_layer_idx(plain, 0, 4), 5)

        # Explicit logical attrs win and are never offset.
        self.assertEqual(
            layer_discovery.resolve_layer_idx(
                SimpleNamespace(layer_idx=9, config=config), 0, 4
            ),
            9,
        )

    def test_mtp_layer_idx_is_absolute_not_max_of_local_main_layers(self):
        """On a PP stage holding a partial stack, MTP must not be numbered max(local)+1."""
        config = SimpleNamespace(num_hidden_layers=43)
        # Last pipeline stage: only layers 36..37 of a 43-layer model, plus MTP.
        main_layers = [
            SimpleNamespace(
                self_attn=SimpleNamespace(is_swa=False),
                layer_number=n,
                config=config,
            )
            for n in (36, 37)
        ]
        mtp_wrapper = SimpleNamespace(
            transformer_layer=SimpleNamespace(
                self_attn=SimpleNamespace(is_swa=False), config=config
            ),
            layer_number=0,
            config=config,
        )

        result = layer_discovery.iter_monitor_layers(
            [*main_layers, mtp_wrapper],
            lambda layer: hasattr(layer, "self_attn"),
        )

        self.assertEqual([item.idx for item in result], [36, 37, 43])
        self.assertEqual([item.is_mtp for item in result], [False, False, True])

    def test_mtp_layer_idx_is_absolute_when_no_main_layer_matched(self):
        """The degenerate 'nothing matched' case must not produce id 0."""
        config = SimpleNamespace(num_hidden_layers=43)
        mtp_wrapper = SimpleNamespace(
            transformer_layer=SimpleNamespace(
                self_attn=SimpleNamespace(is_swa=False), config=config
            ),
            layer_number=0,
            config=config,
        )

        result = layer_discovery.iter_monitor_layers(
            [SimpleNamespace(), mtp_wrapper],
            lambda layer: hasattr(layer, "self_attn"),
        )

        self.assertEqual([item.idx for item in result], [43])


class PaddleMoEMonitorTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_gpu_buffer_gate_metrics_recorded(self):
        """Gate hook records metrics via GPU-buffer API without D2H sync."""
        monitor = PaddleMoEMonitor(log_per_layer=True, log_global=True)
        # Manually declare and allocate for layer 0
        for m in [
            "router_entropy",
            "score_sum_mean",
            "score_sum_min",
            "score_sum_max",
        ]:
            monitor.declare_layer_metric(0, m)
        monitor.allocate_buffers()

        # Simulate recording
        monitor.record_layer_metric(0, "router_entropy", paddle.to_tensor(2.0))
        monitor.step()

        latest = training_logs.get_latest(prefix="moe_health")
        self.assertAlmostEqual(
            latest["moe_health/layer_0/router_entropy"], 2.0, places=4
        )
        self.assertAlmostEqual(
            latest["moe_health/global_router_entropy"], 2.0, places=4
        )

    def test_distribution_metrics_distinguish_balanced_and_collapsed_load(self):
        balanced = moe_monitor_module._distribution_metrics(
            paddle.to_tensor([2.0, 2.0, 2.0, 2.0])
        )
        collapsed = moe_monitor_module._distribution_metrics(
            paddle.to_tensor([8.0, 0.0, 0.0, 0.0])
        )
        empty = moe_monitor_module._distribution_metrics(paddle.zeros([4]))

        self.assertAlmostEqual(float(balanced["cv"]), 0.0, places=6)
        self.assertAlmostEqual(float(balanced["entropy_norm"]), 1.0, places=6)
        self.assertAlmostEqual(float(balanced["max_frac"]), 0.25, places=6)
        self.assertGreater(float(collapsed["cv"]), float(balanced["cv"]))
        self.assertLess(
            float(collapsed["entropy_norm"]), float(balanced["entropy_norm"])
        )
        self.assertAlmostEqual(float(collapsed["max_frac"]), 1.0, places=6)
        self.assertTrue(
            all(math.isfinite(float(value)) for value in empty.values())
        )
        self.assertTrue(all(float(value) == 0.0 for value in empty.values()))

    def test_gate_hook_records_actual_assignment_and_positive_gate_mass(self):
        class Gate(nn.Layer):
            def __init__(self):
                super().__init__()
                self.num_experts_per_tok = 2
                self.n_group = 1
                self.topk_group = 1
                self.norm_topk_prob = False

            def gate_score_func(self, scores):
                return scores

            def forward(self, scores):
                gates = self.gate_score_func(scores)
                top_gate, top_idx = paddle.topk(
                    gates, self.num_experts_per_tok, axis=-1
                )
                mask = paddle.nn.functional.one_hot(
                    top_idx, num_classes=gates.shape[-1]
                ).sum(axis=1)
                combine_weights = gates * mask.astype("float32")
                signed = paddle.where(
                    paddle.arange(gates.shape[-1]).reshape([1, -1]) == 3,
                    -combine_weights,
                    combine_weights,
                )
                return None, top_gate, top_idx, signed, mask, None, None, None

        class MoE(nn.Layer):
            def __init__(self):
                super().__init__()
                self.gate = Gate()

        class TransformerLayer(nn.Layer):
            def __init__(self):
                super().__init__()
                self.layer_idx = 0
                self.mlp = MoE()

        class Model(nn.Layer):
            def __init__(self):
                super().__init__()
                self.layers = nn.LayerList([TransformerLayer()])

        router_input = paddle.to_tensor(
            [
                [0.60, 0.40, 0.10, 0.05],
                [0.10, 0.05, 0.60, 0.40],
                [0.60, 0.10, 0.40, 0.05],
                [0.10, 0.60, 0.05, 0.40],
            ],
            dtype="float32",
        )
        model = Model()
        monitor = PaddleMoEMonitor()
        monitor.register_hooks(model)
        model.layers[0].mlp.gate(router_input)
        monitor.step()

        metrics = training_logs.get_latest(prefix="moe_health/layer_0/")
        self.assertAlmostEqual(
            metrics["moe_health/layer_0/assignment_load_cv"], 0.0, places=6
        )
        self.assertAlmostEqual(
            metrics["moe_health/layer_0/assignment_load_entropy_norm"],
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            metrics["moe_health/layer_0/assignment_load_max_frac"],
            0.25,
            places=6,
        )
        self.assertAlmostEqual(
            metrics["moe_health/layer_0/gate_mass_max_frac"], 0.30, places=6
        )
        self.assertAlmostEqual(
            metrics["moe_health/layer_0/gate_mass_min_frac"], 0.20, places=6
        )
        self.assertGreater(metrics["moe_health/layer_0/router_margin_min"], 0.0)
        self.assertAlmostEqual(
            metrics["moe_health/layer_0/router_input_rms"],
            float(paddle.sqrt((router_input**2).mean())),
            places=6,
        )
        self.assertGreaterEqual(
            metrics["moe_health/layer_0/router_entropy_norm"], 0.0
        )
        self.assertLessEqual(
            metrics["moe_health/layer_0/router_entropy_norm"], 1.0
        )
        monitor.remove_hooks()

    def test_assignment_mask_ignores_invalid_expert_ids(self):
        probabilities = paddle.to_tensor(
            [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], dtype="float32"
        )
        invalid_indices = paddle.to_tensor([[-1, 3], [4, -2]], dtype="int64")
        outputs = (None, None, invalid_indices)

        assignment_mask = moe_monitor_module._assignment_mask(
            probabilities, outputs, k=2
        )

        self.assertEqual(float(assignment_mask.sum()), 0.0)

    def test_routing_margin_is_finite_for_invalid_assignment_rows(self):
        scores = paddle.to_tensor(
            [[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.5, 0.4, 0.2]],
            dtype="float32",
        )
        assignment_mask = paddle.to_tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            dtype="float32",
        )

        margin, valid = moe_monitor_module._routing_margin(
            scores, assignment_mask
        )
        metrics = moe_monitor_module._masked_margin_metrics(margin, valid)

        self.assertTrue(bool(paddle.isfinite(margin).all()))
        self.assertAlmostEqual(float(margin[0]), 0.0)
        self.assertAlmostEqual(float(margin[1]), 0.3)
        self.assertAlmostEqual(float(margin[2]), 0.0)
        self.assertEqual(valid.tolist(), [False, True, False])
        for value in metrics.values():
            if value is not metrics["router_margin_valid_ratio"]:
                self.assertAlmostEqual(float(value), 0.3)
        self.assertAlmostEqual(
            float(metrics["router_margin_valid_ratio"]), 1.0 / 3.0
        )

    def test_routing_margin_metrics_return_zero_for_all_invalid_rows(self):
        margin = paddle.zeros([3], dtype="float32")
        valid = paddle.zeros([3], dtype="bool")

        metrics = moe_monitor_module._masked_margin_metrics(margin, valid)

        self.assertTrue(
            all(bool(paddle.isfinite(value)) for value in metrics.values())
        )
        self.assertTrue(all(float(value) == 0.0 for value in metrics.values()))
        self.assertAlmostEqual(float(metrics["router_margin_valid_ratio"]), 0.0)

    def test_routing_margin_metrics_exclude_invalid_rows_from_all_reductions(
        self,
    ):
        margin = paddle.to_tensor([0.0, 0.1, 0.3, 0.0], dtype="float32")
        valid = paddle.to_tensor([False, True, True, False], dtype="bool")

        metrics = moe_monitor_module._masked_margin_metrics(margin, valid)

        self.assertAlmostEqual(float(metrics["router_margin_mean"]), 0.2)
        self.assertAlmostEqual(float(metrics["router_margin_min"]), 0.1)
        self.assertAlmostEqual(float(metrics["router_margin_p10"]), 0.12)
        self.assertAlmostEqual(float(metrics["router_margin_p01"]), 0.102)
        self.assertAlmostEqual(float(metrics["router_margin_valid_ratio"]), 0.5)

    def test_mtp_layer_marker_is_encoded_in_metric_key(self):
        monitor = PaddleMoEMonitor(log_per_layer=True, log_global=True)
        monitor.mark_mtp_layers([1])
        monitor.declare_layer_metric(0, "router_entropy")
        monitor.declare_layer_metric(1, "router_entropy")
        monitor.allocate_buffers()

        monitor.record_layer_metric(0, "router_entropy", paddle.to_tensor(2.0))
        monitor.record_layer_metric(1, "router_entropy", paddle.to_tensor(4.0))
        monitor.step()

        latest = training_logs.get_latest(prefix="moe_health")
        self.assertAlmostEqual(
            latest["moe_health/layer_0/router_entropy"], 2.0, places=4
        )
        self.assertAlmostEqual(
            latest["moe_health/layer_1_mtp/router_entropy"], 4.0, places=4
        )
        self.assertAlmostEqual(
            latest["moe_health/global_router_entropy"], 3.0, places=4
        )

    def test_per_expert_vector_expands_to_one_key_per_expert(self):
        monitor = PaddleMoEMonitor(log_per_layer=True, log_global=True)
        monitor.declare_layer_vector(0, "expert_token_share", 3)
        monitor.allocate_buffers()

        monitor.record_layer_vector(
            0, "expert_token_share", paddle.to_tensor([10.0, 20.0, 70.0])
        )
        monitor.record_layer_vector(
            0, "expert_token_share", paddle.to_tensor([30.0, 20.0, 50.0])
        )
        monitor.step()

        latest = training_logs.get_latest(prefix="moe_health")
        # Mean over the two records, one key per expert, and no global rollup.
        self.assertAlmostEqual(
            latest["moe_health/layer_0/expert_token_share_e0"], 20.0, places=4
        )
        self.assertAlmostEqual(
            latest["moe_health/layer_0/expert_token_share_e1"], 20.0, places=4
        )
        self.assertAlmostEqual(
            latest["moe_health/layer_0/expert_token_share_e2"], 60.0, places=4
        )
        self.assertNotIn("moe_health/global_expert_token_share", latest)

    def test_per_expert_vector_shares_the_single_scalar_flush(self):
        """Vector and scalar accumulators come back through one D2H together."""
        monitor = PaddleMoEMonitor(log_per_layer=True, log_global=False)
        monitor.declare_layer_metric(0, "router_entropy")
        monitor.declare_layer_vector(0, "expert_weight_share", 2)
        monitor.allocate_buffers()

        monitor.record_layer_metric(0, "router_entropy", paddle.to_tensor(2.0))
        monitor.record_layer_vector(
            0, "expert_weight_share", paddle.to_tensor([40.0, 60.0])
        )
        monitor.step()

        latest = training_logs.get_latest(prefix="moe_health")
        self.assertAlmostEqual(
            latest["moe_health/layer_0/router_entropy"], 2.0, places=4
        )
        self.assertAlmostEqual(
            latest["moe_health/layer_0/expert_weight_share_e0"], 40.0, places=4
        )
        self.assertAlmostEqual(
            latest["moe_health/layer_0/expert_weight_share_e1"], 60.0, places=4
        )

    def test_per_expert_vector_resets_between_steps(self):
        monitor = PaddleMoEMonitor(log_per_layer=True, log_global=False)
        monitor.declare_layer_vector(0, "expert_token_share", 2)
        monitor.allocate_buffers()

        monitor.record_layer_vector(
            0, "expert_token_share", paddle.to_tensor([25.0, 75.0])
        )
        monitor.step()
        monitor.record_layer_vector(
            0, "expert_token_share", paddle.to_tensor([60.0, 40.0])
        )
        monitor.step()

        latest = training_logs.get_latest(prefix="moe_health")
        self.assertAlmostEqual(
            latest["moe_health/layer_0/expert_token_share_e0"], 60.0, places=4
        )

    def test_per_expert_vector_skipped_when_log_per_layer_off(self):
        monitor = PaddleMoEMonitor(log_per_layer=False, log_global=True)
        monitor.declare_layer_vector(0, "expert_token_share", 2)
        monitor.allocate_buffers()

        monitor.record_layer_vector(
            0, "expert_token_share", paddle.to_tensor([25.0, 75.0])
        )
        monitor.step()

        self.assertEqual(training_logs.get_latest(prefix="moe_health"), {})

    def test_expert_shares_come_from_assignment_and_probs(self):
        """assignment -> token share, probs -> combine-weight share, both in percent."""
        monitor = PaddleMoEMonitor(log_per_layer=True, log_global=False)
        monitor.declare_layer_vector(0, "expert_token_share", 3)
        monitor.declare_layer_vector(0, "expert_weight_share", 3)
        monitor.allocate_buffers()

        # 2 tokens, 3 experts, top-1: token 0 -> e0, token 1 -> e2. `assignment`
        # is the same per-expert count the caller already summed for
        # assignment_load_*, so the two views stay exactly consistent.
        assignment = paddle.to_tensor([1.0, 0.0, 1.0])
        probs = paddle.to_tensor([[0.25, 0.0, 0.0], [0.0, 0.0, 0.75]])
        outputs = (None, None, None, probs, None, None, None, None)
        monitor._record_expert_shares(0, assignment, outputs)
        monitor.step()

        latest = training_logs.get_latest(prefix="moe_health")
        token = [
            latest[f"moe_health/layer_0/expert_token_share_e{i}"]
            for i in range(3)
        ]
        self.assertAlmostEqual(token[0], 50.0, places=4)
        self.assertAlmostEqual(token[1], 0.0, places=4)
        self.assertAlmostEqual(token[2], 50.0, places=4)
        self.assertAlmostEqual(sum(token), 100.0, places=4)
        self.assertAlmostEqual(
            latest["moe_health/layer_0/expert_weight_share_e0"], 25.0, places=4
        )
        self.assertAlmostEqual(
            latest["moe_health/layer_0/expert_weight_share_e2"], 75.0, places=4
        )

    def test_expert_token_share_survives_missing_probs(self):
        """A gate that returns no combine weights still yields the token share."""
        monitor = PaddleMoEMonitor(log_per_layer=True, log_global=False)
        monitor.declare_layer_vector(0, "expert_token_share", 2)
        monitor.declare_layer_vector(0, "expert_weight_share", 2)
        monitor.allocate_buffers()

        monitor._record_expert_shares(
            0, paddle.to_tensor([3.0, 1.0]), (None, None, None)
        )
        monitor.step()

        latest = training_logs.get_latest(prefix="moe_health")
        self.assertAlmostEqual(
            latest["moe_health/layer_0/expert_token_share_e0"], 75.0, places=4
        )
        self.assertAlmostEqual(
            latest["moe_health/layer_0/expert_token_share_e1"], 25.0, places=4
        )
        self.assertNotIn("moe_health/layer_0/expert_weight_share_e0", latest)

    def test_gpu_buffer_multi_layer_global_aggregation(self):
        """Global metrics are derived from layer accumulators at flush time."""
        monitor = PaddleMoEMonitor(log_per_layer=False, log_global=True)
        for layer_idx in (0, 1):
            for m in ["router_entropy", "score_sum_max"]:
                monitor.declare_layer_metric(layer_idx, m)
        monitor.allocate_buffers()

        monitor.record_layer_metric(0, "router_entropy", paddle.to_tensor(2.0))
        monitor.record_layer_metric(1, "router_entropy", paddle.to_tensor(4.0))
        monitor.record_layer_metric(0, "score_sum_max", paddle.to_tensor(0.8))
        monitor.record_layer_metric(1, "score_sum_max", paddle.to_tensor(0.9))
        monitor.step()

        latest = training_logs.get_latest(prefix="moe_health")
        self.assertAlmostEqual(
            latest["moe_health/global_router_entropy"], 3.0, places=4
        )
        self.assertAlmostEqual(
            latest["moe_health/global_score_sum_max"], 0.9, places=4
        )

    def test_singular_values_match_svd_reference(self):
        """The Gram spectrum equals the SVD spectrum (order is unspecified)."""
        paddle.seed(0)
        w = paddle.randn([32, 80])
        got = paddle.sort(
            moe_monitor_module._singular_values(w), descending=True
        )
        reference = paddle.linalg.svd(w, full_matrices=False)[1]
        self.assertEqual(list(got.shape), [32])
        self.assertTrue(bool(paddle.allclose(got, reference, atol=1e-3)))

    def test_singular_values_batched_matches_per_matrix(self):
        """A batched [E, m, n] input yields one spectrum per matrix."""
        paddle.seed(0)
        w = paddle.randn([3, 16, 8])
        batched = moe_monitor_module._singular_values(w)
        self.assertEqual(list(batched.shape), [3, 8])
        for e in range(3):
            single = moe_monitor_module._singular_values(w[e])
            self.assertTrue(
                bool(paddle.allclose(batched[e], single, atol=1e-4))
            )

    def test_singular_values_return_none_for_unsupported_input(self):
        self.assertIsNone(moe_monitor_module._singular_values(None))
        self.assertIsNone(
            moe_monitor_module._singular_values(paddle.randn([10]))
        )

    def test_spectrum_metrics_reduce_over_trailing_axis(self):
        """Batched sigma [E, k] reduces to one value per matrix."""
        sigma = paddle.to_tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]])
        srank = moe_monitor_module._stable_rank(sigma)
        entropy = moe_monitor_module._singular_value_entropy(sigma)
        self.assertEqual(list(srank.shape), [2])
        self.assertAlmostEqual(float(srank[0]), 4.0, places=4)
        self.assertAlmostEqual(float(srank[1]), 1.0, places=4)
        self.assertAlmostEqual(float(entropy[0]), math.log(4.0), places=4)
        self.assertAlmostEqual(float(entropy[1]), 0.0, places=4)

    def test_stable_rank_of_orthogonal_matrix_equals_full_rank(self):
        """Flat spectrum => srank == min(m, n), the stable-rank upper bound."""
        q, _ = paddle.linalg.qr(paddle.randn([32, 32]))
        sigma = moe_monitor_module._singular_values(q)
        self.assertAlmostEqual(
            float(moe_monitor_module._stable_rank(sigma)), 32.0, places=3
        )

    def test_stable_rank_of_rank_one_matrix_is_one(self):
        """One nonzero singular value => srank == 1, its lower bound."""
        w = paddle.randn([64, 1]) @ paddle.randn([1, 128])
        sigma = moe_monitor_module._singular_values(w)
        self.assertAlmostEqual(
            float(moe_monitor_module._stable_rank(sigma)), 1.0, places=3
        )

    def test_stable_rank_matches_frobenius_over_spectral_norm(self):
        """srank == ||W||_F^2 / ||W||_2^2 computed straight from the SVD."""
        paddle.seed(0)
        w = paddle.randn([32, 80])
        sigma = paddle.linalg.svd(w, full_matrices=False)[1]
        reference = float((sigma**2).sum() / sigma.max() ** 2)
        got = float(
            moe_monitor_module._stable_rank(
                moe_monitor_module._singular_values(w)
            )
        )
        self.assertAlmostEqual(got, reference, places=3)

    def test_singular_value_entropy_of_orthogonal_matrix_equals_log_full_rank(
        self,
    ):
        """Flat spectrum => H == log(min(m, n)), the entropy upper bound."""
        q, _ = paddle.linalg.qr(paddle.randn([32, 32]))
        sigma = moe_monitor_module._singular_values(q)
        self.assertAlmostEqual(
            float(moe_monitor_module._singular_value_entropy(sigma)),
            math.log(32.0),
            places=3,
        )

    def test_singular_value_entropy_of_rank_one_matrix_is_near_zero(self):
        """A single dominant singular value collapses H towards 0."""
        w = paddle.randn([64, 1]) @ paddle.randn([1, 128])
        sigma = moe_monitor_module._singular_values(w)
        self.assertLess(
            float(moe_monitor_module._singular_value_entropy(sigma)), 0.5
        )

    def test_singular_value_entropy_matches_svd_definition(self):
        """H == -sum(p log p) with p = sigma^2 / sum(sigma^2) from the SVD."""
        paddle.seed(0)
        w = paddle.randn([32, 80])
        sigma = paddle.linalg.svd(w, full_matrices=False)[1]
        p = sigma**2 / (sigma**2).sum()
        reference = float(-(p * p.log()).sum())
        got = float(
            moe_monitor_module._singular_value_entropy(
                moe_monitor_module._singular_values(w)
            )
        )
        self.assertAlmostEqual(got, reference, places=3)

    def test_singular_value_entropy_uses_alpha_two_weighting(self):
        """A skewed spectrum separates sigma^2 from sigma^1 weighting."""
        sigma = paddle.to_tensor([4.0, 2.0, 1.0])
        sq = [16.0, 4.0, 1.0]
        alpha2 = -sum(v / sum(sq) * math.log(v / sum(sq)) for v in sq)
        alpha1 = -sum(v / 7.0 * math.log(v / 7.0) for v in (4.0, 2.0, 1.0))
        got = float(moe_monitor_module._singular_value_entropy(sigma))
        self.assertAlmostEqual(got, alpha2, places=5)
        self.assertNotAlmostEqual(got, alpha1, places=2)

    def test_swiglu_gate_half_takes_first_half_of_output_dim(self):
        """glu() applies SiLU to the first chunk, so the gate is w[..., :out // 2]."""
        w = paddle.randn([2, 8, 16])
        gate = moe_monitor_module._swiglu_gate_half(w)
        self.assertEqual(list(gate.shape), [2, 8, 8])
        self.assertTrue(bool(paddle.allclose(gate, w[..., :8])))

    def test_expert_fc1_weight_resolves_supported_layouts(self):
        """grouped-gemm weight1, fused up_gate_proj and LayerList all resolve."""
        ggm_w = paddle.randn([2, 4, 8])
        ggm_layer = SimpleNamespace(
            grouped_gemm_experts=SimpleNamespace(weight1=ggm_w)
        )
        self.assertTrue(
            bool(
                paddle.allclose(
                    moe_monitor_module._expert_fc1_weight(ggm_layer), ggm_w
                )
            )
        )

        fused_w = paddle.randn([3, 4, 8])
        fused_layer = SimpleNamespace(
            grouped_gemm_experts=None,
            experts=SimpleNamespace(
                up_gate_proj=SimpleNamespace(weight=fused_w)
            ),
        )
        self.assertTrue(
            bool(
                paddle.allclose(
                    moe_monitor_module._expert_fc1_weight(fused_layer), fused_w
                )
            )
        )

        per_expert = [
            SimpleNamespace(
                up_gate_proj=SimpleNamespace(weight=paddle.randn([4, 8]))
            )
            for _ in range(2)
        ]
        list_layer = SimpleNamespace(
            grouped_gemm_experts=None, experts=per_expert
        )
        self.assertEqual(
            list(moe_monitor_module._expert_fc1_weight(list_layer).shape),
            [2, 4, 8],
        )

        self.assertIsNone(
            moe_monitor_module._expert_fc1_weight(SimpleNamespace())
        )

    def test_compute_gate_spectrum_metrics_records_expert_and_shared_gate(self):
        """Per-expert gate spectra reduce to mean/min/max; shared gate is scalar."""
        monitor = PaddleMoEMonitor(log_per_layer=True, log_global=True)
        for m in (
            "expert_gate_stable_rank_mean",
            "expert_gate_stable_rank_min",
            "expert_gate_stable_rank_max",
            "expert_gate_singular_entropy_mean",
            "expert_gate_singular_entropy_min",
            "expert_gate_singular_entropy_max",
            "shared_gate_stable_rank",
            "shared_gate_singular_entropy",
        ):
            monitor.declare_layer_metric(0, m)
        monitor.allocate_buffers()

        full_rank, _ = paddle.linalg.qr(
            paddle.randn([8, 8])
        )  # srank 8, H = log 8
        collapsed = paddle.randn([8, 1]) @ paddle.randn(
            [1, 8]
        )  # srank 1, H ~ 0
        gate = paddle.stack([full_rank, collapsed])  # [2, 8, 8]
        fc1 = paddle.concat(
            [gate, paddle.randn([2, 8, 8])], axis=-1
        )  # gate | up
        shared_fc1 = paddle.concat([full_rank, paddle.randn([8, 8])], axis=-1)
        moe_layer = SimpleNamespace(
            grouped_gemm_experts=None,
            experts=SimpleNamespace(up_gate_proj=SimpleNamespace(weight=fc1)),
            shared_experts=SimpleNamespace(
                up_gate_proj=SimpleNamespace(weight=shared_fc1)
            ),
        )

        monitor._compute_gate_spectrum_metrics(0, moe_layer)
        monitor.step()

        latest = training_logs.get_latest(prefix="moe_health")
        self.assertAlmostEqual(
            latest["moe_health/layer_0/expert_gate_stable_rank_max"],
            8.0,
            places=3,
        )
        self.assertAlmostEqual(
            latest["moe_health/layer_0/expert_gate_stable_rank_min"],
            1.0,
            places=3,
        )
        self.assertAlmostEqual(
            latest["moe_health/layer_0/expert_gate_stable_rank_mean"],
            4.5,
            places=3,
        )
        self.assertAlmostEqual(
            latest["moe_health/layer_0/expert_gate_singular_entropy_max"],
            math.log(8.0),
            places=3,
        )
        self.assertLess(
            latest["moe_health/layer_0/expert_gate_singular_entropy_min"], 0.5
        )
        self.assertAlmostEqual(
            latest["moe_health/layer_0/shared_gate_stable_rank"], 8.0, places=3
        )
        self.assertAlmostEqual(
            latest["moe_health/layer_0/shared_gate_singular_entropy"],
            math.log(8.0),
            places=3,
        )
        self.assertNotIn("moe_health/layer_0/gate_weight_stable_rank", latest)

    def test_compute_gate_spectrum_metrics_batches_shared_gate_of_different_shape(
        self,
    ):
        """Shared and routed gate matrices of different shape still share one eigensolve.

        This is the production layout: with ``moe_split_feature_routing`` the routed
        experts run on half the hidden features, so their gate is ``[512, 512]`` while
        the shared expert's is ``[1024, 512]``. Batching the gate matrices is
        impossible, but their Grams are both ``k x k``, so exactly one
        ``_gram_singular_values`` call must cover both -- alternating batch shapes
        makes cuSOLVER re-initialize its workspace and costs ~8x.
        """
        monitor = PaddleMoEMonitor(log_per_layer=True, log_global=True)
        for m in (
            "expert_gate_stable_rank_mean",
            "expert_gate_stable_rank_min",
            "expert_gate_stable_rank_max",
            "expert_gate_singular_entropy_mean",
            "expert_gate_singular_entropy_min",
            "expert_gate_singular_entropy_max",
            "shared_gate_stable_rank",
            "shared_gate_singular_entropy",
        ):
            monitor.declare_layer_metric(0, m)
        monitor.allocate_buffers()

        full_rank, _ = paddle.linalg.qr(
            paddle.randn([8, 8])
        )  # srank 8, H = log 8
        collapsed = paddle.randn([8, 1]) @ paddle.randn(
            [1, 8]
        )  # srank 1, H ~ 0
        fc1 = paddle.concat(
            [paddle.stack([full_rank, collapsed]), paddle.randn([2, 8, 8])],
            axis=-1,
        )
        # Shared expert sees twice as many input features: gate half is [16, 8], and
        # orthonormal columns again put every singular value at 1 -> srank 8, H = log 8.
        tall_gate, _ = paddle.linalg.qr(paddle.randn([16, 8]))
        shared_fc1 = paddle.concat([tall_gate, paddle.randn([16, 8])], axis=-1)
        self.assertNotEqual(tall_gate.shape, fc1.shape[1:])
        moe_layer = SimpleNamespace(
            grouped_gemm_experts=None,
            experts=SimpleNamespace(up_gate_proj=SimpleNamespace(weight=fc1)),
            shared_experts=SimpleNamespace(
                up_gate_proj=SimpleNamespace(weight=shared_fc1)
            ),
        )

        real_solve = moe_monitor_module._gram_singular_values
        calls = []

        def counting_solve(gram):
            calls.append(None if gram is None else tuple(gram.shape))
            return real_solve(gram)

        moe_monitor_module._gram_singular_values = counting_solve
        try:
            monitor._compute_gate_spectrum_metrics(0, moe_layer)
        finally:
            moe_monitor_module._gram_singular_values = real_solve
        monitor.step()

        self.assertEqual(
            calls,
            [(3, 8, 8)],
            "routed and shared Grams must batch into one solve",
        )
        latest = training_logs.get_latest(prefix="moe_health")
        self.assertAlmostEqual(
            latest["moe_health/layer_0/expert_gate_stable_rank_max"],
            8.0,
            places=3,
        )
        self.assertAlmostEqual(
            latest["moe_health/layer_0/expert_gate_stable_rank_min"],
            1.0,
            places=3,
        )
        self.assertAlmostEqual(
            latest["moe_health/layer_0/shared_gate_stable_rank"], 8.0, places=3
        )
        self.assertAlmostEqual(
            latest["moe_health/layer_0/shared_gate_singular_entropy"],
            math.log(8.0),
            places=3,
        )

    def test_compute_gate_spectrum_metrics_skips_layer_without_experts(self):
        """A layer with no expert weights must not raise and records nothing."""
        monitor = PaddleMoEMonitor(log_per_layer=True, log_global=True)
        monitor.declare_layer_metric(0, "expert_gate_stable_rank_mean")
        monitor.allocate_buffers()

        monitor._compute_gate_spectrum_metrics(0, SimpleNamespace())
        self.assertEqual(
            monitor._gpu_cnt["moe_health/layer_0/expert_gate_stable_rank_mean"],
            0,
        )

    def test_attn_type_tag_produces_typed_keys_and_split_global_aggregation(
        self,
    ):
        """When declare/record_layer_metric receives attn_type, keys are
        ``{prefix}/layer_N/{type}_{metric}`` and ``{prefix}/global_{type}_{metric}``.
        SWA and full aggregate independently -- window vs full never mix.

        MoE monitor is used here as a convenient concrete PaddleProbe subclass;
        in production attn_type only flows through attention-family monitors.
        """
        monitor = PaddleMoEMonitor(log_per_layer=True, log_global=True)
        # layer 0 is full, layer 1 is swa, layer 2 is full
        monitor.declare_layer_metric(0, "router_entropy", attn_type="full")
        monitor.declare_layer_metric(1, "router_entropy", attn_type="swa")
        monitor.declare_layer_metric(2, "router_entropy", attn_type="full")
        monitor.allocate_buffers()

        monitor.record_layer_metric(
            0, "router_entropy", paddle.to_tensor(2.0), attn_type="full"
        )
        monitor.record_layer_metric(
            1, "router_entropy", paddle.to_tensor(10.0), attn_type="swa"
        )
        monitor.record_layer_metric(
            2, "router_entropy", paddle.to_tensor(4.0), attn_type="full"
        )
        monitor.step()

        latest = training_logs.get_latest(prefix="moe_health")
        # Per-layer keys carry the attn_type prefix on the metric name.
        self.assertAlmostEqual(
            latest["moe_health/layer_0/full_router_entropy"], 2.0, places=4
        )
        self.assertAlmostEqual(
            latest["moe_health/layer_1/swa_router_entropy"], 10.0, places=4
        )
        self.assertAlmostEqual(
            latest["moe_health/layer_2/full_router_entropy"], 4.0, places=4
        )
        # Global aggregations split by attn_type: SWA global uses only layer 1;
        # full global averages layer 0 and layer 2. There is no untagged
        # ``global_router_entropy`` when all layers are typed.
        self.assertAlmostEqual(
            latest["moe_health/global_swa_router_entropy"], 10.0, places=4
        )
        self.assertAlmostEqual(
            latest["moe_health/global_full_router_entropy"], 3.0, places=4
        )
        self.assertNotIn("moe_health/global_router_entropy", latest)

    def test_fused_expert_sumsq_matches_per_expert_concat_norm(self):
        """Vectorized per-expert sum-of-squares sqrts to the old concat([w1_i, w2_i]).norm()."""
        moe_monitor_mod = importlib.import_module(
            "paddlefleet.internal_medicine.backends.paddlefleet.moe_monitor"
        )
        num_experts, h, i = 4, 8, 6
        w1 = paddle.randn([num_experts, h, i], dtype="float32")
        w2 = paddle.randn([num_experts, i, h], dtype="float32")

        got = paddle.sqrt(moe_monitor_mod._per_expert_stacked_sumsq(w1, w2))
        expected = paddle.stack(
            [
                paddle.concat([w1[e].flatten(), w2[e].flatten()]).norm()
                for e in range(num_experts)
            ]
        )
        self.assertEqual(list(got.shape), [num_experts])
        self.assertTrue(bool(paddle.allclose(got, expected, atol=1e-5)))

    def test_intermediate_shard_group_detects_allgather_layout(self):
        """An expert module holding only I/EP of each expert reports its EP group."""
        moe_monitor_mod = importlib.import_module(
            "paddlefleet.internal_medicine.backends.paddlefleet.moe_monitor"
        )
        ep_group = object()

        replicated = SimpleNamespace(
            intermediate_size_per_partition=2048,
            config=SimpleNamespace(moe_intermediate_size=2048),
            ep_group=ep_group,
        )
        self.assertIsNone(moe_monitor_mod._intermediate_shard_group(replicated))

        sharded = SimpleNamespace(
            intermediate_size_per_partition=256,
            config=SimpleNamespace(moe_intermediate_size=2048),
            ep_group=ep_group,
        )
        self.assertIs(
            moe_monitor_mod._intermediate_shard_group(sharded), ep_group
        )
        self.assertIsNone(moe_monitor_mod._intermediate_shard_group(None))

    def test_collect_expert_norms_fused_layout_records_metrics(self):
        """collect_expert_norms records expert norms + gate spectra for a fused-expert MoE layer."""
        monitor = PaddleMoEMonitor(
            log_per_layer=True, log_global=True, verbose=False
        )
        for m in [
            "expert_norm_mean",
            "expert_norm_std",
            "expert_norm_min",
            "expert_norm_max",
            "shared_expert_norm",
            "shared_routed_ratio",
            "expert_gate_stable_rank_mean",
            "expert_gate_stable_rank_min",
            "expert_gate_stable_rank_max",
            "expert_gate_singular_entropy_mean",
            "expert_gate_singular_entropy_min",
            "expert_gate_singular_entropy_max",
        ]:
            monitor.declare_layer_metric(0, m)
        monitor.allocate_buffers()

        num_experts, h, i = 3, 8, 6
        fused_experts = SimpleNamespace(
            up_gate_proj=SimpleNamespace(
                weight=paddle.randn([num_experts, h, i], dtype="float32")
            ),
            down_proj=SimpleNamespace(
                weight=paddle.randn([num_experts, i, h], dtype="float32")
            ),
        )
        shared = nn.Linear(h, i)
        moe_layer = SimpleNamespace(
            experts=fused_experts, shared_experts=shared
        )
        moe_layer.grouped_gemm_experts = None

        monitor._expert_norm_layers = [(0, moe_layer)]
        monitor.collect_expert_norms()
        monitor.step()

        latest = training_logs.get_latest(prefix="moe_health")
        self.assertIn("moe_health/layer_0/expert_norm_mean", latest)
        self.assertIn("moe_health/layer_0/shared_expert_norm", latest)
        self.assertGreater(latest["moe_health/layer_0/expert_norm_mean"], 0.0)
        # Same step-begin pass also collects the SwiGLU gate spectrum: the gate
        # half of a [3, 8, 6] fc1 is [3, 8, 3], so srank is in [1, 3].
        self.assertIn("moe_health/layer_0/expert_gate_stable_rank_mean", latest)
        self.assertGreaterEqual(
            latest["moe_health/layer_0/expert_gate_stable_rank_min"], 1.0
        )
        self.assertLessEqual(
            latest["moe_health/layer_0/expert_gate_stable_rank_max"], 3.0
        )

    def test_collect_expert_norms_respects_monitor_interval(self):
        """Expert-norm collection is gated by the global monitor_interval."""
        monitor = PaddleMoEMonitor(
            log_per_layer=True,
            log_global=True,
            monitor_interval=2,
            verbose=False,
        )
        for m in [
            "expert_norm_mean",
            "expert_norm_std",
            "expert_norm_min",
            "expert_norm_max",
        ]:
            monitor.declare_layer_metric(0, m)
        monitor.allocate_buffers()

        num_experts, h, i = 2, 4, 4
        fused_experts = SimpleNamespace(
            up_gate_proj=SimpleNamespace(
                weight=paddle.randn([num_experts, h, i], dtype="float32")
            ),
            down_proj=SimpleNamespace(
                weight=paddle.randn([num_experts, i, h], dtype="float32")
            ),
        )
        moe_layer = SimpleNamespace(experts=fused_experts, shared_experts=None)
        moe_layer.grouped_gemm_experts = None
        monitor._expert_norm_layers = [(0, moe_layer)]

        # step_count=1 -> 1 % 2 != 0 -> should NOT record
        monitor.step_count = 1
        monitor.collect_expert_norms()
        # nothing recorded -> count stays 0, flush emits nothing for this key
        self.assertEqual(
            monitor._gpu_cnt["moe_health/layer_0/expert_norm_mean"], 0
        )

    def test_expert_norm_collect_exception_does_not_crash(self):
        """collect_expert_norms swallows per-layer read errors without crashing the step."""
        monitor = PaddleMoEMonitor(
            log_per_layer=False, log_global=True, verbose=False
        )
        for m in [
            "expert_norm_mean",
            "expert_norm_std",
            "expert_norm_min",
            "expert_norm_max",
        ]:
            monitor.declare_layer_metric(2, m)
        monitor.allocate_buffers()

        # Expert norms are collected at step-begin from _expert_norm_layers,
        # not from a forward hook. A layer that raises on weight access must be
        # caught so the step still completes.
        monitor._expert_norm_layers = [(2, BrokenPaddleMoELayer())]
        monitor.collect_expert_norms()

        # Should not crash; step should still work
        monitor.step()

    def test_hash_routing_cache_supports_sqrtsoftplus(self):
        monitor = PaddleMoEMonitor(
            log_per_layer=False, log_global=True, verbose=False
        )
        logits = paddle.to_tensor([[0.0, 1.0]], dtype="float32")

        def original_hash_routing(logits, flat_ids):
            return "ok"

        gate = SimpleNamespace(
            gate_score_func=lambda logits: logits,
            _hash_routing=original_hash_routing,
            scoring_func="sqrtsoftplus",
        )
        monitor._patch_gate_cache(gate)

        self.assertEqual(
            gate._hash_routing(logits, paddle.to_tensor([0], dtype="int64")),
            "ok",
        )
        expected = paddle.sqrt(paddle.nn.functional.softplus(logits) + 1e-20)
        self.assertTrue(bool(paddle.allclose(gate._cached_gates, expected)))

        monitor.remove_hooks()
        self.assertIs(gate._hash_routing, original_hash_routing)

    def test_hash_routing_cache_respects_monitor_interval(self):
        monitor = PaddleMoEMonitor(
            log_per_layer=False,
            log_global=True,
            monitor_interval=2,
            verbose=False,
        )
        logits = paddle.to_tensor([[0.0, 1.0]], dtype="float32")
        gate = SimpleNamespace(
            gate_score_func=lambda logits: logits,
            _hash_routing=lambda logits, flat_ids: "ok",
            scoring_func="sigmoid",
        )
        monitor._patch_gate_cache(gate)
        monitor.step_count = 1

        self.assertEqual(
            gate._hash_routing(logits, paddle.to_tensor([0], dtype="int64")),
            "ok",
        )
        self.assertIsNone(gate._cached_gates)


class SplitFeatureRoutingCaptureTest(unittest.TestCase):
    """``moe_split_feature_routing`` routes on the SUM of two gate views.

    ``moe_router`` computes ``gates = f(logits_0) + f(logits_1)``, so the patched
    ``gate_score_func`` fires twice per forward. Overwriting the cache would leave
    the monitor with view 1 alone — half the routing signal — which silently skews
    every metric derived from it (router_entropy, score_sum_*, bias_affinity_jaccard).
    """

    class FakeGate:
        def __init__(self, split):
            self.moe_split_feature_routing = split
            self.num_experts_per_tok = 2

        def gate_score_func(self, logits):
            return paddle.nn.functional.sigmoid(logits)

    def _patched_gate(self, split):
        monitor = PaddleMoEMonitor()
        gate = self.FakeGate(split)
        monitor._patch_gate_cache(gate)
        return monitor, gate

    def test_split_routing_accumulates_both_views(self):
        _monitor, gate = self._patched_gate(split=True)
        logits_0 = paddle.to_tensor([[1.0, -1.0, 0.5, 0.0]], dtype="float32")
        logits_1 = paddle.to_tensor([[-2.0, 2.0, 0.0, 0.25]], dtype="float32")

        gate.gate_score_func(logits_0)
        gate.gate_score_func(logits_1)

        expected = paddle.nn.functional.sigmoid(
            logits_0
        ) + paddle.nn.functional.sigmoid(logits_1)
        self.assertTrue(
            paddle.allclose(gate._cached_gates, expected, atol=1e-6).item()
        )

    def test_split_routing_top_experts_come_from_the_sum(self):
        """View 1 alone ranks experts differently from the sum the router uses."""
        _monitor, gate = self._patched_gate(split=True)
        logits_0 = paddle.to_tensor([[4.0, 4.0, -4.0, -4.0]], dtype="float32")
        logits_1 = paddle.to_tensor([[-1.0, -2.0, 3.0, 2.0]], dtype="float32")

        gate.gate_score_func(logits_0)
        gate.gate_score_func(logits_1)

        summed_top = (
            paddle.topk(gate._cached_gates, 2, axis=-1)[1]
            .numpy()
            .flatten()
            .tolist()
        )
        view1_top = (
            paddle.topk(paddle.nn.functional.sigmoid(logits_1), 2, axis=-1)[1]
            .numpy()
            .flatten()
            .tolist()
        )
        self.assertEqual(sorted(summed_top), [0, 1])
        self.assertNotEqual(sorted(summed_top), sorted(view1_top))

    def test_single_view_router_keeps_last_write(self):
        """Without split routing the cache must not accumulate."""
        _monitor, gate = self._patched_gate(split=False)
        logits_0 = paddle.to_tensor([[1.0, -1.0, 0.5, 0.0]], dtype="float32")
        logits_1 = paddle.to_tensor([[-2.0, 2.0, 0.0, 0.25]], dtype="float32")

        gate.gate_score_func(logits_0)
        gate.gate_score_func(logits_1)

        expected = paddle.nn.functional.sigmoid(logits_1)
        self.assertTrue(
            paddle.allclose(gate._cached_gates, expected, atol=1e-6).item()
        )

    def test_hash_layers_opt_out_of_split_accumulation(self):
        """Hash layers bypass split routing and capture via _hash_routing instead."""
        gate = self.FakeGate(split=True)
        gate.is_hash_layer = True
        self.assertFalse(PaddleMoEMonitor._uses_split_routing(gate))

    def test_cache_cleared_between_forwards_prevents_leakage(self):
        """The gate post-hook clears the cache, so forwards never accumulate."""
        _monitor, gate = self._patched_gate(split=True)
        logits = paddle.to_tensor([[1.0, -1.0, 0.5, 0.0]], dtype="float32")

        gate.gate_score_func(logits)
        gate.gate_score_func(logits)
        two_views = gate._cached_gates.clone()

        gate._cached_gates = None  # what _make_gate_hook's finally block does
        gate.gate_score_func(logits)
        gate.gate_score_func(logits)

        self.assertTrue(
            paddle.allclose(gate._cached_gates, two_views, atol=1e-6).item()
        )


class BiasAffinityJaccardTest(unittest.TestCase):
    """The bias-free branch must mirror the router's group-limited topk exactly.

    PaddleFleet scores a group by the sum of its top-2 experts (tensor path
    ``moe_router.TopKRouter``, fused path ``moe_topk_fusion``: ``score = m1 + m2``).
    Using a group max instead selects different groups, so the metric would report
    routing divergence even with an all-zero correction bias.
    """

    @staticmethod
    def _router_topk(scores, k, n_group, topk_group):
        """Reference: the router's own selection (group score = top-2 sum)."""
        num_tokens, num_experts = scores.shape
        group_size = num_experts // n_group
        grouped = scores.reshape([num_tokens, n_group, group_size])
        group_scores = grouped.topk(2, axis=-1)[0].sum(axis=-1)
        _, top_groups = paddle.topk(group_scores, topk_group, axis=-1)
        mask = paddle.zeros([num_tokens, n_group], dtype="int32")
        mask = mask.put_along_axis(
            top_groups, paddle.to_tensor(1, dtype="int32"), axis=1
        )
        mask = (
            mask.unsqueeze(-1)
            .expand([-1, -1, group_size])
            .reshape([num_tokens, num_experts])
        )
        masked = paddle.where(
            mask > 0, scores, paddle.full_like(scores, float("-inf"))
        )
        return paddle.topk(masked, k, axis=-1)[1]

    def test_zero_bias_gives_identical_routing(self):
        """correction_bias is zero-initialised, so step 1 must read exactly 1.0."""
        paddle.seed(0)
        scores = paddle.rand([16, 8], dtype="float32")
        k, n_group, topk_group = 2, 2, 1

        top_idx = self._router_topk(scores, k, n_group, topk_group)
        jaccard = moe_monitor_module._compute_bias_affinity_jaccard(
            top_idx, scores, k, n_group, topk_group
        )
        self.assertAlmostEqual(float(jaccard), 1.0, places=6)

    def test_group_score_uses_top2_sum_not_max(self):
        """A case where max and top-2 sum disagree on which group wins.

        group0 has the single strongest expert, group1 the stronger pair:
        max picks group0, top-2 sum picks group1. The router picks group1, so a
        max-based reimplementation would score 0 here instead of 1.
        """
        scores = paddle.to_tensor(
            [[0.90, 0.05, 0.03, 0.02, 0.80, 0.70, 0.05, 0.05]],
            dtype="float32",
        )
        k, n_group, topk_group = 2, 2, 1

        top_idx = self._router_topk(scores, k, n_group, topk_group)
        # Sanity: the router really did take group1 (experts 4 and 5).
        self.assertEqual(sorted(top_idx.numpy().flatten().tolist()), [4, 5])

        jaccard = moe_monitor_module._compute_bias_affinity_jaccard(
            top_idx, scores, k, n_group, topk_group
        )
        self.assertAlmostEqual(float(jaccard), 1.0, places=6)

    def test_nonzero_bias_lowers_the_score(self):
        """With a bias that flips the winning group, the score must drop below 1."""
        scores = paddle.to_tensor(
            [[0.90, 0.80, 0.05, 0.05, 0.40, 0.30, 0.05, 0.05]],
            dtype="float32",
        )
        bias = paddle.to_tensor(
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], dtype="float32"
        )
        k, n_group, topk_group = 2, 2, 1

        top_idx = self._router_topk(scores + bias, k, n_group, topk_group)
        jaccard = moe_monitor_module._compute_bias_affinity_jaccard(
            top_idx, scores, k, n_group, topk_group
        )
        self.assertAlmostEqual(float(jaccard), 0.0, places=6)

    def test_ungrouped_router_path(self):
        paddle.seed(1)
        scores = paddle.rand([8, 6], dtype="float32")
        top_idx = paddle.topk(scores, 3, axis=-1)[1]
        jaccard = moe_monitor_module._compute_bias_affinity_jaccard(
            top_idx, scores, 3, n_group=1, topk_group=1
        )
        self.assertAlmostEqual(float(jaccard), 1.0, places=6)

    def test_single_expert_groups_do_not_crash(self):
        """group_size == 1 has no second expert to add; top-2 clamps to top-1."""
        scores = paddle.to_tensor([[0.9, 0.1, 0.5, 0.4]], dtype="float32")
        k, n_group, topk_group = 2, 4, 2
        num_tokens, num_experts = scores.shape
        grouped = scores.reshape([num_tokens, n_group, num_experts // n_group])
        _, top_groups = paddle.topk(grouped.squeeze(-1), topk_group, axis=-1)
        mask = paddle.zeros([num_tokens, n_group], dtype="int32")
        mask = mask.put_along_axis(
            top_groups, paddle.to_tensor(1, dtype="int32"), axis=1
        )
        masked = paddle.where(
            mask > 0, scores, paddle.full_like(scores, float("-inf"))
        )
        top_idx = paddle.topk(masked, k, axis=-1)[1]

        jaccard = moe_monitor_module._compute_bias_affinity_jaccard(
            top_idx, scores, k, n_group, topk_group
        )
        self.assertAlmostEqual(float(jaccard), 1.0, places=6)


class PaddleMassiveActivationMonitorTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_extract_hidden_states_supports_dict_and_positional_inputs(self):
        monitor = PaddleMassiveActivationMonitor()
        hidden_states = paddle.randn([2, 3, 4], dtype="float32")

        self.assertIs(
            monitor._extract_hidden_states(({"hidden_states": hidden_states},)),
            hidden_states,
        )
        self.assertIs(
            monitor._extract_hidden_states((hidden_states,)), hidden_states
        )
        self.assertIsNone(monitor._extract_hidden_states(()))
        self.assertIsNone(
            monitor._extract_hidden_states(({"other": hidden_states},))
        )

    def test_compute_and_log_records_pre_norm_metrics(self):
        monitor = PaddleMassiveActivationMonitor(
            log_per_layer=True,
            log_global=True,
            cosine_sample_pairs=4,
            absolute_thresholds=(2.0, 3.0),
        )
        hidden_states = paddle.to_tensor(
            [
                [[1.0, -2.0, 0.5, 4.0]],
                [[3.0, 1.0, -0.5, 2.0]],
            ],
            dtype="float32",
        )

        # Must declare + allocate before _compute_and_log
        metric_names = [
            "channel_max",
            "channel_median",
            "channel_p95",
            "channel_p99",
            "channel_max_ratio",
            "massive_act_channel_count",
            "topk_channel_norm",
            "activation_rms",
            "post_norm_sparsity",
            "post_norm_cosine",
            "channel_count_gt_2",
            "channel_count_gt_3",
        ]
        for m in metric_names:
            monitor.declare_layer_metric(0, m)
        monitor.allocate_buffers()

        monitor._compute_and_log(0, hidden_states, nn.Layer())
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
            "activation_rms",
        ):
            self.assertIn(f"massive_act/layer_0/{key}", latest)
            self.assertIn(f"massive_act/global_{key}", latest)
        self.assertEqual(latest["massive_act/layer_0/channel_count_gt_2"], 2.0)
        self.assertEqual(latest["massive_act/layer_0/channel_count_gt_3"], 1.0)

    def test_count_metrics_are_max_aggregated_across_layers(self):
        monitor = PaddleMassiveActivationMonitor(
            log_per_layer=False,
            log_global=True,
            absolute_thresholds=(2.0, 3.0),
        )

        # Declare and allocate for 2 layers
        for layer_idx in (0, 1):
            for m in [
                "massive_act_channel_count",
                "channel_count_gt_2",
                "channel_count_gt_3",
            ]:
                monitor.declare_layer_metric(layer_idx, m)
        monitor.allocate_buffers()

        monitor.record_layer_metric(
            0, "massive_act_channel_count", paddle.to_tensor(0.0)
        )
        monitor.record_layer_metric(
            0, "channel_count_gt_2", paddle.to_tensor(1.0)
        )
        monitor.record_layer_metric(
            0, "channel_count_gt_3", paddle.to_tensor(0.0)
        )
        monitor.record_layer_metric(
            1, "massive_act_channel_count", paddle.to_tensor(2.0)
        )
        monitor.record_layer_metric(
            1, "channel_count_gt_2", paddle.to_tensor(3.0)
        )
        monitor.record_layer_metric(
            1, "channel_count_gt_3", paddle.to_tensor(1.0)
        )
        monitor.step()

        latest = training_logs.get_latest(prefix="massive_act")
        self.assertEqual(
            latest["massive_act/global_massive_act_channel_count"], 2.0
        )
        self.assertEqual(latest["massive_act/global_channel_count_gt_2"], 3.0)
        self.assertEqual(latest["massive_act/global_channel_count_gt_3"], 1.0)

    def test_module_position_metrics_follow_actual_residual_path(self):
        class Branch(nn.Layer):
            def __init__(self, scale, bias):
                super().__init__()
                self.scale = scale
                self.bias = bias

            def forward(self, hidden_states):
                output = hidden_states * self.scale
                bias = paddle.full_like(hidden_states, self.bias)
                return output, bias

        class TransformerLayer(nn.Layer):
            def __init__(self):
                super().__init__()
                self.layer_idx = 0
                self.input_layernorm = nn.Identity()
                self.self_attn = Branch(scale=2.0, bias=1.0)
                self.post_attention_layernorm = nn.Identity()
                self.mlp = Branch(scale=-0.5, bias=-2.0)

            def forward(self, hidden_states):
                attn_out, attn_bias = self.self_attn(hidden_states)
                post_attn = hidden_states + attn_out + attn_bias
                mlp_input = self.post_attention_layernorm(post_attn)
                ffn_out, ffn_bias = self.mlp(mlp_input)
                return post_attn + ffn_out + ffn_bias

        class Model(nn.Layer):
            def __init__(self):
                super().__init__()
                self.layers = nn.LayerList([TransformerLayer()])

        hidden_states = paddle.to_tensor(
            [[[1.0, 2.0], [-1.0, 0.5]]], dtype="float32"
        )
        expected_attn_out = 2.0 * hidden_states
        expected_post_attn = hidden_states + expected_attn_out + 1.0
        expected_ffn_out = -0.5 * expected_post_attn
        expected_output = expected_post_attn + expected_ffn_out - 2.0
        expected_attn_ratio = paddle.sqrt(
            ((expected_post_attn - hidden_states) ** 2).mean()
        ) / paddle.sqrt((hidden_states**2).mean())
        expected_ffn_ratio = paddle.sqrt(
            ((expected_output - expected_post_attn) ** 2).mean()
        ) / paddle.sqrt((expected_post_attn**2).mean())

        training_logs.reset()
        model = Model()
        monitor = PaddleMassiveActivationMonitor(cosine_sample_pairs=2)
        monitor.register_hooks(model)
        actual_output = model.layers[0](hidden_states)
        monitor.step()

        self.assertTrue(paddle.allclose(actual_output, expected_output).item())
        metrics = training_logs.get_latest(prefix="massive_act/layer_0/")
        for position in (
            "layer_input",
            "attn_out",
            "post_attn_residual",
            "ffn_or_moe_out",
            "post_ffn_residual",
        ):
            for metric_name in ("rms", "abs_max", "abs_p99", "outlier_ratio"):
                self.assertIn(
                    f"massive_act/layer_0/{position}_{metric_name}", metrics
                )
        self.assertAlmostEqual(
            metrics["massive_act/layer_0/attn_out_rms"],
            float(paddle.sqrt((expected_attn_out**2).mean())),
            places=6,
        )
        self.assertAlmostEqual(
            metrics["massive_act/layer_0/post_attn_residual_abs_max"],
            float(expected_post_attn.abs().max()),
            places=6,
        )
        self.assertAlmostEqual(
            metrics["massive_act/layer_0/ffn_or_moe_out_rms"],
            float(paddle.sqrt((expected_ffn_out**2).mean())),
            places=6,
        )
        self.assertAlmostEqual(
            metrics["massive_act/layer_0/post_ffn_residual_abs_max"],
            float(expected_output.abs().max()),
            places=6,
        )
        self.assertAlmostEqual(
            metrics["massive_act/layer_0/attn_update_rms_ratio"],
            float(expected_attn_ratio),
            places=6,
        )
        self.assertAlmostEqual(
            metrics["massive_act/layer_0/ffn_update_rms_ratio"],
            float(expected_ffn_ratio),
            places=6,
        )
        monitor.remove_hooks()

    def test_new_position_hooks_fail_closed(self):
        hidden_states = paddle.randn([2, 3, 4], dtype="float32")

        def fail(*_args, **_kwargs):
            raise RuntimeError("injected monitor failure")

        for hook_kind in (
            "branch_output",
            "post_attn_residual",
            "layer_output",
        ):
            with self.subTest(hook_kind=hook_kind):
                monitor = PaddleMassiveActivationMonitor()
                monitor._record_position = fail
                layer = nn.Identity()
                if hook_kind == "branch_output":
                    handle = layer.register_forward_post_hook(
                        monitor._make_branch_output_hook(0, "attn_out", None)
                    )
                elif hook_kind == "post_attn_residual":
                    monitor._position_cache[0] = {
                        "layer_input": hidden_states.detach()
                    }
                    handle = layer.register_forward_pre_hook(
                        monitor._make_post_attn_residual_hook(0, None)
                    )
                else:
                    monitor._position_cache[0] = {
                        "layer_input": hidden_states.detach()
                    }
                    handle = layer.register_forward_post_hook(
                        monitor._make_layer_output_hook(0, None)
                    )

                actual = layer(hidden_states)

                self.assertTrue(paddle.allclose(actual, hidden_states).item())
                if hook_kind == "layer_output":
                    self.assertNotIn(0, monitor._position_cache)
                handle.remove()

    def test_update_ratio_hooks_fail_closed(self):
        hidden_states = paddle.randn([2, 3, 4], dtype="float32")

        def fail(*_args, **_kwargs):
            raise RuntimeError("injected update-ratio failure")

        monitor = PaddleMassiveActivationMonitor()
        monitor._record_position = lambda *_args, **_kwargs: None
        monitor._record_update_ratio = fail
        monitor._position_cache[0] = {"layer_input": hidden_states.detach()}
        layer = nn.Identity()
        handle = layer.register_forward_pre_hook(
            monitor._make_post_attn_residual_hook(0, None)
        )

        actual = layer(hidden_states)

        self.assertTrue(paddle.allclose(actual, hidden_states).item())
        self.assertIn("post_attn_residual", monitor._position_cache[0])
        handle.remove()

        monitor._position_cache[0] = {
            "layer_input": hidden_states.detach(),
            "post_attn_residual": hidden_states.detach(),
        }
        handle = layer.register_forward_post_hook(
            monitor._make_layer_output_hook(0, None)
        )

        actual = layer(hidden_states)

        self.assertTrue(paddle.allclose(actual, hidden_states).item())
        self.assertNotIn(0, monitor._position_cache)
        handle.remove()

    def test_output_extraction_failures_do_not_interrupt_forward(self):
        hidden_states = paddle.randn([2, 3, 4], dtype="float32")

        def fail(*_args, **_kwargs):
            raise RuntimeError("injected output extraction failure")

        for hook_kind in ("branch_output", "layer_output"):
            with self.subTest(hook_kind=hook_kind):
                monitor = PaddleMassiveActivationMonitor()
                monitor._extract_output_tensor = fail
                layer = nn.Identity()
                if hook_kind == "branch_output":
                    handle = layer.register_forward_post_hook(
                        monitor._make_branch_output_hook(0, "attn_out", None)
                    )
                else:
                    monitor._position_cache[0] = {
                        "layer_input": hidden_states.detach()
                    }
                    handle = layer.register_forward_post_hook(
                        monitor._make_layer_output_hook(0, None)
                    )

                actual = layer(hidden_states)

                self.assertTrue(paddle.allclose(actual, hidden_states).item())
                if hook_kind == "layer_output":
                    self.assertNotIn(0, monitor._position_cache)
                handle.remove()

    def test_module_position_metrics_support_mhc_residual_streams(self):
        class Branch(nn.Layer):
            def __init__(self, scale):
                super().__init__()
                self.scale = scale

            def forward(self, hidden_states):
                return hidden_states * self.scale, None

        class HyperConnection(nn.Layer):
            def __init__(self, num_streams):
                super().__init__()
                self.num_streams = num_streams

            def forward(self, hidden_states):
                stream_width = hidden_states.shape[-1] // self.num_streams
                streams = hidden_states.reshape(
                    [*hidden_states.shape[:-1], self.num_streams, stream_width]
                )
                return streams.mean(axis=-2), None, None

        class TransformerLayer(nn.Layer):
            def __init__(self):
                super().__init__()
                self.layer_idx = 0
                self.input_layernorm = nn.Identity()
                self.post_attention_layernorm = nn.Identity()
                self.self_attention_hyper_connection = HyperConnection(
                    num_streams=2
                )
                self.mlp_hyper_connection = HyperConnection(num_streams=2)
                self.self_attn = Branch(scale=0.5)
                self.mlp = Branch(scale=-0.25)

            def forward(self, hidden_states):
                attn_input, _, _ = self.self_attention_hyper_connection(
                    hidden_states
                )
                attn_out, _ = self.self_attn(attn_input)
                post_attn = hidden_states + paddle.concat(
                    [attn_out, 2.0 * attn_out], axis=-1
                )
                mlp_input, _, _ = self.mlp_hyper_connection(post_attn)
                ffn_out, _ = self.mlp(mlp_input)
                return post_attn + paddle.concat(
                    [ffn_out, 0.5 * ffn_out], axis=-1
                )

        class Model(nn.Layer):
            def __init__(self):
                super().__init__()
                self.layers = nn.LayerList([TransformerLayer()])

        hidden_states = paddle.to_tensor(
            [[[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 2.0, -2.0]]]
        )
        model = Model()
        monitor = PaddleMassiveActivationMonitor(cosine_sample_pairs=2)
        monitor.register_hooks(model)

        expected = model.layers[0](hidden_states)
        monitor.step()

        self.assertEqual(expected.shape, hidden_states.shape)
        metrics = training_logs.get_latest(prefix="massive_act/layer_0/")
        self.assertIn("massive_act/layer_0/attn_out_rms", metrics)
        self.assertIn("massive_act/layer_0/post_attn_residual_rms", metrics)
        self.assertIn("massive_act/layer_0/ffn_or_moe_out_rms", metrics)
        self.assertIn("massive_act/layer_0/post_ffn_residual_rms", metrics)
        self.assertIn("massive_act/layer_0/attn_update_rms_ratio", metrics)
        self.assertIn("massive_act/layer_0/ffn_update_rms_ratio", metrics)
        monitor.remove_hooks()


class PaddleQKMonitorTest(unittest.TestCase):
    def test_resolve_layer_idx_uses_shared_base_logic(self):
        monitor = PaddleQKStatsMonitor()

        self.assertEqual(
            monitor._resolve_layer_idx(SimpleNamespace(layer_idx=8), 0, 4), 8
        )
        self.assertEqual(
            monitor._resolve_layer_idx(SimpleNamespace(layer_number=2), 0, 4), 1
        )

        monitor.pp_rank = 1
        self.assertEqual(monitor._resolve_layer_idx(SimpleNamespace(), 2, 4), 6)

    def test_row_stride_must_be_positive(self):
        for bad in (0, -1, -32):
            with self.assertRaises(ValueError):
                PaddleQKStatsMonitor(row_stride=bad)

    def test_row_stride_default_is_exact_full_pass(self):
        self.assertEqual(PaddleQKStatsMonitor().row_stride, 1)

    def test_dense_hook_records_qkv_vector_norms(self):
        training_logs.reset()

        class CoreAttention(nn.Layer):
            def forward(self, query, key, value):
                return value

        class Attention(nn.Layer):
            def __init__(self):
                super().__init__()
                self.core_attention = CoreAttention()

        class TransformerLayer(nn.Layer):
            def __init__(self):
                super().__init__()
                self.layer_idx = 0
                self.self_attn = Attention()

        class Model(nn.Layer):
            def __init__(self):
                super().__init__()
                self.layers = nn.LayerList([TransformerLayer()])

        zero = paddle.zeros([1, 1], dtype="float32")
        fake_stats = {
            "max_global": zero.sum(),
            "mean_global": zero.sum(),
            "entropy_global": zero.sum(),
            "sink_global": zero.sum(),
            "entropy_per_head": zero,
            "sink_per_head": zero,
        }
        query = paddle.to_tensor(
            [[[[3.0, 4.0]], [[0.0, 5.0]]]], dtype="float32"
        )
        key = paddle.to_tensor(
            [[[[5.0, 12.0]], [[8.0, 15.0]]]], dtype="float32"
        )
        value = paddle.to_tensor(
            [[[[7.0, 24.0]], [[20.0, 21.0]]]], dtype="float32"
        )
        model = Model()
        monitor = PaddleQKStatsMonitor()

        with mock.patch.object(
            qk_monitor_module,
            "compute_qk_stats_paddle",
            return_value=fake_stats,
        ):
            monitor.register_hooks(model)
            model.layers[0].self_attn.core_attention(query, key, value)
            monitor.step()

        metrics = training_logs.get_latest(prefix="qk_stats/layer_0/")
        self.assertAlmostEqual(metrics["qk_stats/layer_0/q_norm_mean"], 5.0)
        self.assertAlmostEqual(metrics["qk_stats/layer_0/q_norm_max"], 5.0)
        self.assertAlmostEqual(metrics["qk_stats/layer_0/k_norm_mean"], 15.0)
        self.assertAlmostEqual(metrics["qk_stats/layer_0/k_norm_max"], 17.0)
        self.assertAlmostEqual(metrics["qk_stats/layer_0/v_norm_mean"], 27.0)
        self.assertAlmostEqual(metrics["qk_stats/layer_0/v_norm_max"], 29.0)
        monitor.remove_hooks()


class PaddleQKKernelComputeTest(unittest.TestCase):
    """GPU numerical tests for the shared triton qk_stats kernel via paddle."""

    @classmethod
    def setUpClass(cls):
        if (
            not paddle.device.is_compiled_with_cuda()
            or paddle.device.cuda.device_count() == 0
        ):
            raise unittest.SkipTest("qk_stats kernel requires a CUDA GPU")
        paddle.device.set_device("gpu:0")
        cls.qk = importlib.import_module(
            "paddlefleet.internal_medicine.backends.paddlefleet.qk_monitor"
        )

    def _gqa_inputs(self, B=1, S=128, Hq=8, Hkv=2, D=32, seed=0):
        paddle.seed(seed)
        q = paddle.randn([B, S, Hq, D], dtype="float32")
        k = paddle.randn([B, S, Hkv, D], dtype="float32")
        return q, k

    def test_kernel_grouping_matches_explicit_repeat_interleave(self):
        """GQA via in-kernel head mapping == materialized repeat_interleave."""
        q, k = self._gqa_inputs()
        heads_per_group = q.shape[2] // k.shape[2]

        grouped = self.qk.compute_qk_stats_paddle(
            q, k, causal=True, row_stride=1
        )

        k_expanded = k.repeat_interleave(heads_per_group, axis=2)
        expanded = self.qk.compute_qk_stats_paddle(
            q, k_expanded, causal=True, row_stride=1
        )

        for key in (
            "max_global",
            "mean_global",
            "entropy_global",
            "sink_global",
        ):
            self.assertTrue(
                paddle.allclose(
                    grouped[key], expanded[key], atol=1e-4, rtol=1e-4
                ).item(),
                f"{key} mismatch: grouped={grouped[key].item()} expanded={expanded[key].item()}",
            )

    def test_nondefault_softmax_scale_matches_dense_reference(self):
        q, k = self._gqa_inputs(B=1, S=64, Hq=4, Hkv=4, D=16, seed=17)
        scale = 0.137

        actual = self.qk.compute_qk_stats_paddle(
            q, k, causal=True, softmax_scale=scale
        )

        qh = q.transpose([0, 2, 1, 3])
        kh = k.transpose([0, 2, 1, 3])
        logits = paddle.matmul(qh, kh, transpose_y=True) * scale
        causal = paddle.tril(
            paddle.ones([q.shape[1], k.shape[1]], dtype="bool")
        )
        masked_logits = paddle.where(
            causal, logits, paddle.full_like(logits, -1e10)
        )
        probability = paddle.nn.functional.softmax(masked_logits, axis=-1)
        entropy = -(probability * paddle.log(probability.clip(min=1e-30))).sum(
            axis=-1
        )

        self.assertTrue(
            paddle.allclose(
                actual["max_global"], masked_logits.max(), atol=5e-3, rtol=1e-4
            ).item()
        )
        self.assertTrue(
            paddle.allclose(
                actual["entropy_global"], entropy.mean(), atol=1e-3, rtol=1e-4
            ).item()
        )
        self.assertTrue(
            paddle.allclose(
                actual["sink_global"],
                probability[..., 0].mean(),
                atol=1e-4,
                rtol=1e-4,
            ).item()
        )

    def test_monitor_hook_uses_core_attention_runtime_scale(self):
        class CoreAttention(nn.Layer):
            def __init__(self, softmax_scale):
                super().__init__()
                self.softmax_scale = softmax_scale

            def forward(self, query, key, value):
                return value

        class Attention(nn.Layer):
            def __init__(self, softmax_scale):
                super().__init__()
                self.core_attention = CoreAttention(softmax_scale)

        class TransformerLayer(nn.Layer):
            def __init__(self, softmax_scale):
                super().__init__()
                self.layer_idx = 0
                self.self_attn = Attention(softmax_scale)

        class Model(nn.Layer):
            def __init__(self, softmax_scale):
                super().__init__()
                self.layers = nn.LayerList([TransformerLayer(softmax_scale)])

        scale = 0.173
        q, k = self._gqa_inputs(B=1, S=32, Hq=2, Hkv=2, D=16, seed=23)
        value = paddle.randn(q.shape, dtype="float32")
        reference = self.qk.compute_qk_stats_paddle(
            q, k, causal=True, softmax_scale=scale
        )
        model = Model(scale)
        monitor = PaddleQKStatsMonitor()
        monitor.register_hooks(model)
        training_logs.reset()

        model.layers[0].self_attn.core_attention(q, k, value)
        monitor.step()

        metrics = training_logs.get_latest(prefix="qk_stats/layer_0/")
        self.assertAlmostEqual(
            metrics["qk_stats/layer_0/max"],
            float(reference["max_global"]),
            places=3,
        )
        self.assertAlmostEqual(
            metrics["qk_stats/layer_0/entropy_avg"],
            float(reference["entropy_global"]),
            places=3,
        )
        self.assertAlmostEqual(
            metrics["qk_stats/layer_0/sink"],
            float(reference["sink_global"]),
            places=4,
        )
        monitor.remove_hooks()

    def test_row_stride_is_near_unbiased_for_mean_class_metrics(self):
        """Subsampling query rows must keep the row-averaged metrics close to
        the full pass. entropy_global / mean_global / sink_global are all
        uniform averages over query rows, so a uniform stride is an unbiased,
        low-variance estimator. entropy has real magnitude -> tight relative
        check; mean and sink sit near zero for N(0,1) inputs -> absolute check.
        """
        q, k = self._gqa_inputs(S=512, seed=1)
        full = self.qk.compute_qk_stats_paddle(q, k, causal=True, row_stride=1)
        sub = self.qk.compute_qk_stats_paddle(q, k, causal=True, row_stride=8)

        rel = abs(
            sub["entropy_global"].item() - full["entropy_global"].item()
        ) / (abs(full["entropy_global"].item()) + 1e-6)
        self.assertLess(
            rel,
            0.1,
            f"entropy_global drifted: full={full['entropy_global'].item()} sub={sub['entropy_global'].item()}",
        )

        for key in ("mean_global", "sink_global"):
            self.assertLess(
                abs(sub[key].item() - full[key].item()),
                0.05,
                f"{key} drifted: full={full[key].item()} sub={sub[key].item()}",
            )

        # max is an extremum -> subsample is a lower bound (<= full).
        self.assertLessEqual(
            sub["max_global"].item(), full["max_global"].item() + 1e-4
        )

    def test_row_stride_one_is_exact_full_sequence(self):
        q, k = self._gqa_inputs(Hkv=8, seed=2)  # MHA, no grouping
        a = self.qk.compute_qk_stats_paddle(q, k, causal=True, row_stride=1)
        b = self.qk.compute_qk_stats_paddle(q, k, causal=True, row_stride=1)
        for key in (
            "max_global",
            "mean_global",
            "entropy_global",
            "sink_global",
        ):
            self.assertTrue(paddle.allclose(a[key], b[key], atol=1e-5).item())

    def test_contiguous_cp_k_gather_plus_offset_matches_full_pass(self):
        """Contiguous CP keeps local Q and gathers K.

        Rank r holds Q[r*chunk:(r+1)*chunk] and the full K. The kernel's
        ``q_row_offset`` plus a later per-head mean across ranks must match
        the CP=1 full-sequence reference.
        """
        q, k = self._gqa_inputs(B=1, S=128, Hq=8, Hkv=8, D=32, seed=3)
        full = self.qk.compute_qk_stats_paddle(q, k, causal=True, row_stride=1)
        cp_size = 2
        chunk = q.shape[1] // cp_size
        shards = []
        for rank in range(cp_size):
            q_local = q[:, rank * chunk : (rank + 1) * chunk, :, :]
            shards.append(
                self.qk.compute_qk_stats_paddle(
                    q_local,
                    k,
                    causal=True,
                    row_stride=1,
                    q_row_offset=rank * chunk,
                )
            )
        for key in ("entropy_per_head", "sink_per_head", "mean_per_head"):
            reassembled = sum(shard[key] for shard in shards) / float(cp_size)
            self.assertTrue(
                paddle.allclose(
                    full[key], reassembled, atol=1e-4, rtol=1e-4
                ).item(),
                f"{key} mismatch after contiguous CP reassembly",
            )
        reassembled_max = shards[0]["max_per_head"]
        for shard in shards[1:]:
            reassembled_max = paddle.maximum(
                reassembled_max, shard["max_per_head"]
            )
        self.assertTrue(
            paddle.allclose(
                full["max_per_head"], reassembled_max, atol=1e-4, rtol=1e-4
            ).item()
        )

    def test_dualchunk_cp_balance_reorder_matches_full_pass(self):
        """Dual-chunk CP must invert scatter_balance before QK stats.

        Each rank holds a front chunk plus a rear chunk. Concatenating those
        shards in rank order is the wrong layout; reassembling with the
        inverse of ``scatter_balance`` restores global token order and must
        match the CP=1 reference.
        """
        q, k = self._gqa_inputs(B=1, S=128, Hq=8, Hkv=8, D=32, seed=5)
        full = self.qk.compute_qk_stats_paddle(q, k, causal=True, row_stride=1)
        cp_size = 2
        interval = q.shape[1] // (cp_size * 2)
        seq_len = q.shape[1]

        def scatter_balance_local(tensor, rank):
            front = tensor[:, interval * rank : interval * (rank + 1), :, :]
            rear = tensor[
                :,
                seq_len - interval * (rank + 1) : seq_len - interval * rank,
                :,
                :,
            ]
            return paddle.concat([front, rear], axis=1)

        def gather_balance_local(shards):
            restored = []
            for rank in range(cp_size):
                restored.append(shards[rank][:, :interval, :, :])
            for rank in reversed(range(cp_size)):
                restored.append(shards[rank][:, interval:, :, :])
            return paddle.concat(restored, axis=1)

        q_shards = [scatter_balance_local(q, rank) for rank in range(cp_size)]
        k_shards = [scatter_balance_local(k, rank) for rank in range(cp_size)]
        q_wrong = paddle.concat(q_shards, axis=1)
        k_wrong = paddle.concat(k_shards, axis=1)
        wrong = self.qk.compute_qk_stats_paddle(
            q_wrong, k_wrong, causal=True, row_stride=1
        )
        restored = self.qk.compute_qk_stats_paddle(
            gather_balance_local(q_shards),
            gather_balance_local(k_shards),
            causal=True,
            row_stride=1,
        )
        for key in (
            "entropy_global",
            "sink_global",
            "mean_global",
            "max_global",
        ):
            self.assertTrue(
                paddle.allclose(
                    full[key], restored[key], atol=1e-4, rtol=1e-4
                ).item(),
                f"{key} mismatch after dual-chunk reorder",
            )
        # Entropy and max depend on which tokens sit in the causal window, so
        # the unreordered dual-chunk layout must not reproduce the reference.
        # sink/mean can stay near zero for N(0,1) inputs and are not used as
        # the negative control.
        for key in ("entropy_global", "max_global"):
            self.assertFalse(
                paddle.allclose(
                    full[key], wrong[key], atol=1e-3, rtol=1e-3
                ).item(),
                f"{key} unexpectedly matched the unreordered dual-chunk layout",
            )

    def test_q_row_offset_zero_matches_no_offset(self):
        """Passing q_row_offset=0 must be a no-op vs. the default path."""
        q, k = self._gqa_inputs(seed=4)
        default_pass = self.qk.compute_qk_stats_paddle(
            q, k, causal=True, row_stride=1
        )
        with_zero_offset = self.qk.compute_qk_stats_paddle(
            q, k, causal=True, row_stride=1, q_row_offset=0
        )
        for key in (
            "max_global",
            "mean_global",
            "entropy_global",
            "sink_global",
        ):
            self.assertTrue(
                paddle.allclose(
                    default_pass[key], with_zero_offset[key], atol=1e-6
                ).item(),
                f"{key} diverged with q_row_offset=0",
            )


def _build_hca_topk_idxs(seqlen, window_size, ratio):
    """Rebuild what CompressedSparseAttention feeds its sparse attention kernel.

    Mirrors ``get_window_topk_idxs`` + ``get_compress_topk_idxs`` (simple causal
    branch) from ``paddlefleet.transformer.csa_attention``: a left-inclusive
    sliding window over original KV, concatenated with every causally valid
    compressed block, offset by ``seqlen`` because both index ``kv_full``.
    """
    base = paddle.arange(seqlen).unsqueeze(1)
    offsets = paddle.arange(window_size)
    window = paddle.clip(base - window_size + 1, min=0) + offsets
    window = paddle.where(window > base, paddle.full_like(window, -1), window)

    n_compressed = seqlen // ratio
    k_indices = paddle.arange(n_compressed).unsqueeze(0).expand([seqlen, -1])
    causal_bound = paddle.arange(1, seqlen + 1).unsqueeze(1) // ratio
    compressed = paddle.where(
        k_indices >= causal_bound,
        paddle.full_like(k_indices, -1),
        k_indices + seqlen,
    )
    return (
        paddle.concat([window, compressed], axis=-1)
        .unsqueeze(0)
        .astype("int32")
    )


class DenseSinkFoldTest(unittest.TestCase):
    """The learned sink is a real softmax column and must be folded in.

    Full-attention layers hand ``core_attention.softmax_offset`` to their kernel as
    ``learnable_sink``, so the model's distribution spans ``real keys + 1``. Folding
    it into the stats kernel is what makes entropy/sink describe that distribution
    instead of one renormalised over the real keys only.
    """

    @classmethod
    def setUpClass(cls):
        if (
            not paddle.device.is_compiled_with_cuda()
            or paddle.device.cuda.device_count() == 0
        ):
            raise unittest.SkipTest("qk_stats kernel requires a CUDA GPU")
        paddle.device.set_device("gpu:0")
        cls.qk = importlib.import_module(
            "paddlefleet.internal_medicine.backends.paddlefleet.qk_monitor"
        )

    @staticmethod
    def _inputs(B=1, S=64, Hq=4, Hkv=2, D=32, seed=7):
        paddle.seed(seed)
        q = paddle.randn([B, S, Hq, D], dtype="float32")
        k = paddle.randn([B, S, Hkv, D], dtype="float32")
        return q, k

    @staticmethod
    def _reference(q, k, sink):
        """Materialized [S, S(+1)] softmax reference: entropy and col-0 probability."""
        B, S, Hq, D = q.shape
        heads_per_group = Hq // k.shape[2]
        qh = q.transpose([0, 2, 1, 3])
        kh = k.transpose([0, 2, 1, 3]).repeat_interleave(
            heads_per_group, axis=1
        )
        logits = paddle.matmul(qh, kh, transpose_y=True) / (
            D**0.5
        )  # [B, Hq, S, S]

        causal = paddle.tril(paddle.ones([S, S], dtype="bool"))
        logits = paddle.where(causal, logits, paddle.full_like(logits, -1e10))

        if sink is not None:
            sink_col = sink.reshape([1, Hq, 1, 1]).expand([B, Hq, S, 1])
            logits = paddle.concat([logits, sink_col], axis=-1)

        probs = paddle.nn.functional.softmax(logits, axis=-1)
        entropy = -(probs * paddle.log(paddle.clip(probs, min=1e-30))).sum(
            axis=-1
        )  # [B, Hq, S]
        return entropy.mean(axis=-1), probs[..., 0].mean(
            axis=-1
        )  # both [B, Hq]

    def test_folded_stats_match_materialized_reference(self):
        q, k = self._inputs()
        sink = paddle.to_tensor([0.5, -1.0, 2.0, 0.0], dtype="float32")

        stats = self.qk.compute_qk_stats_paddle(
            q, k, causal=True, row_stride=1, attn_sink=sink
        )
        ref_entropy, ref_sink = self._reference(q, k, sink)

        self.assertTrue(
            paddle.allclose(
                stats["entropy_per_head"], ref_entropy, atol=1e-3, rtol=1e-3
            ).item(),
            f"entropy: kernel={stats['entropy_per_head'].numpy()} ref={ref_entropy.numpy()}",
        )
        self.assertTrue(
            paddle.allclose(
                stats["sink_per_head"], ref_sink, atol=1e-3, rtol=1e-3
            ).item(),
            f"sink: kernel={stats['sink_per_head'].numpy()} ref={ref_sink.numpy()}",
        )

    def test_without_sink_matches_real_key_only_reference(self):
        """attn_sink=None keeps the pre-existing (real-key-only) semantics."""
        q, k = self._inputs()
        stats = self.qk.compute_qk_stats_paddle(q, k, causal=True, row_stride=1)
        ref_entropy, ref_sink = self._reference(q, k, None)

        self.assertTrue(
            paddle.allclose(
                stats["entropy_per_head"], ref_entropy, atol=1e-3, rtol=1e-3
            ).item()
        )
        self.assertTrue(
            paddle.allclose(
                stats["sink_per_head"], ref_sink, atol=1e-3, rtol=1e-3
            ).item()
        )

    def test_zero_sink_still_changes_the_distribution(self):
        """ "off-by-one" softmax: a frozen 0 logit still adds exp(0) to the denominator."""
        q, k = self._inputs()
        zeros = paddle.zeros([q.shape[2]], dtype="float32")

        without = self.qk.compute_qk_stats_paddle(
            q, k, causal=True, row_stride=1
        )
        with_zero = self.qk.compute_qk_stats_paddle(
            q, k, causal=True, row_stride=1, attn_sink=zeros
        )

        # Extra column takes probability mass, so col-0 probability must drop.
        self.assertLess(
            with_zero["sink_global"].item(), without["sink_global"].item()
        )
        # logit max/mean describe real keys only and must be untouched by the fold.
        self.assertAlmostEqual(
            with_zero["max_global"].item(),
            without["max_global"].item(),
            places=4,
        )
        self.assertAlmostEqual(
            with_zero["mean_global"].item(),
            without["mean_global"].item(),
            places=4,
        )

    def test_large_sink_dominates_and_collapses_entropy(self):
        q, k = self._inputs()
        huge = paddle.full([q.shape[2]], 50.0, dtype="float32")
        stats = self.qk.compute_qk_stats_paddle(
            q, k, causal=True, row_stride=1, attn_sink=huge
        )
        # Nearly all mass on the sink column -> entropy ~ 0, real-key sink ~ 0.
        self.assertLess(stats["entropy_global"].item(), 1e-3)
        self.assertLess(stats["sink_global"].item(), 1e-3)


class CompressRatioLayerClassificationTest(unittest.TestCase):
    """Per-layer attention-kind classification on ``csa_compress_ratios`` stacks."""

    @staticmethod
    def _ratio_layer(compress_ratio=None, window_size=None, indexer=None):
        # experimental_attention_variant is what makes csa_compress_ratios
        # authoritative for the per-layer kind.
        config = SimpleNamespace(experimental_attention_variant="dsv4_hybrid")
        if compress_ratio is None:
            core = SimpleNamespace()  # MLA branch: no CSA core
        else:
            core = SimpleNamespace(
                compress_ratio=compress_ratio,
                window_size=window_size,
                indexer=indexer,
                compressed_sparse_attn=lambda *a, **k: None,
            )
        return SimpleNamespace(
            self_attn=SimpleNamespace(config=config, core_attention=core)
        )

    def test_ratio_maps_to_layer_kind(self):
        cases = {-2: "mla", -1: "mqa", 0: "window", 4: "csa", 128: "hca"}
        for ratio, expected in cases.items():
            layer = self._ratio_layer(compress_ratio=ratio)
            self.assertEqual(
                layer_discovery.classify_attn_type(layer),
                expected,
                f"ratio={ratio}",
            )

    def test_mla_branch_without_csa_core_is_tagged_mla(self):
        """Ratio -2 builds MLASelfAttention, which exposes no compress_ratio."""
        layer = self._ratio_layer(compress_ratio=None)
        kind, ratio, window, has_indexer = layer_discovery.attn_meta(layer)
        self.assertEqual(kind, "mla")
        self.assertEqual(ratio, layer_discovery.MLA_RATIO)
        self.assertIsNone(window)
        self.assertFalse(has_indexer)

    def test_meta_carries_window_size_and_indexer_flag(self):
        layer = self._ratio_layer(
            compress_ratio=4, window_size=128, indexer=object()
        )
        kind, ratio, window, has_indexer = layer_discovery.attn_meta(layer)
        self.assertEqual((kind, ratio, window), ("csa", 4, 128))
        self.assertTrue(has_indexer)

    def test_hca_has_no_indexer(self):
        layer = self._ratio_layer(
            compress_ratio=128, window_size=128, indexer=None
        )
        self.assertFalse(layer_discovery.attn_meta(layer)[3])

    def test_stack_without_the_variant_field_keeps_is_swa_fallback(self):
        """Stacks with no ratio field must not gain ratio tags (metric-key compat)."""
        swa = SimpleNamespace(
            self_attn=SimpleNamespace(is_swa=True, config=SimpleNamespace())
        )
        self.assertEqual(layer_discovery.classify_attn_type(swa), "swa")

    def test_iter_monitor_layers_tags_mixed_hca_mla_stack(self):
        layers = [
            self._ratio_layer(compress_ratio=128, window_size=128),
            self._ratio_layer(compress_ratio=128, window_size=128),
            self._ratio_layer(compress_ratio=None),
        ]
        result = layer_discovery.iter_monitor_layers(layers, lambda layer: True)
        self.assertEqual(
            [item.attn_type for item in result], ["hca", "hca", "mla"]
        )
        self.assertEqual(
            [item.compress_ratio for item in result], [128, 128, -2]
        )


class LearnableSinkDetectionTest(unittest.TestCase):
    """Full-attention layers carry their sink bias as core_attention.softmax_offset.

    ``add_full_attention_sink_bias=true`` promotes softmax_type to "learnable", so
    MLA/MQA layers own a trainable per-head sink logit even though they never see
    the sparse ``attn_sink`` kernel argument. They must report attn_sink_logit too.
    """

    @staticmethod
    def _attn(offset):
        return SimpleNamespace(
            core_attention=SimpleNamespace(softmax_offset=offset)
        )

    def test_trainable_offset_is_detected(self):
        offset = paddle.zeros([4], dtype="float32")
        offset.stop_gradient = False
        self.assertIsNotNone(
            PaddleQKStatsMonitor._learnable_sink(self._attn(offset))
        )

    def test_frozen_offset_is_ignored(self):
        # "off-by-one" softmax uses a frozen zero buffer: a constant-0 series is
        # not worth a chart.
        offset = paddle.zeros([4], dtype="float32")
        offset.stop_gradient = True
        self.assertIsNone(
            PaddleQKStatsMonitor._learnable_sink(self._attn(offset))
        )

    def test_vanilla_softmax_has_no_sink(self):
        self.assertIsNone(
            PaddleQKStatsMonitor._learnable_sink(self._attn(None))
        )
        self.assertIsNone(
            PaddleQKStatsMonitor._learnable_sink(SimpleNamespace())
        )


class SparseBoundsTest(unittest.TestCase):
    """Window/compressed segment bounds derived from the model's real indices."""

    def setUp(self):
        self.qk = importlib.import_module(
            "paddlefleet.internal_medicine.backends.paddlefleet.qk_monitor"
        )

    def _hca_topk_idxs(self, seqlen, window_size, ratio):
        return _build_hca_topk_idxs(seqlen, window_size, ratio)

    def test_bounds_reproduce_the_index_set_exactly(self):
        seqlen, window_size, ratio = 32, 8, 4
        topk = self._hca_topk_idxs(seqlen, window_size, ratio)
        bounds = self.qk.sparse_bounds_from_topk(topk, seqlen)

        # Rebuild membership from the [lo, hi] ranges and compare to the real set.
        cols = paddle.arange(seqlen + seqlen // ratio).astype("int32")
        in_win = (cols.unsqueeze(0) >= bounds["win_lo"][0].unsqueeze(1)) & (
            cols.unsqueeze(0) <= bounds["win_hi"][0].unsqueeze(1)
        )
        in_cmp = (cols.unsqueeze(0) >= bounds["cmp_lo"][0].unsqueeze(1)) & (
            cols.unsqueeze(0) <= bounds["cmp_hi"][0].unsqueeze(1)
        )
        from_ranges = in_win | in_cmp

        expected = paddle.zeros_like(from_ranges)
        for row in range(seqlen):
            live = [int(v) for v in topk[0, row].tolist() if v >= 0]
            for col in live:
                expected[row, col] = True

        self.assertTrue(bool((from_ranges == expected).all()))

    def test_sink_col_points_at_first_compressed_block_when_present(self):
        seqlen, window_size, ratio = 32, 8, 4
        topk = self._hca_topk_idxs(seqlen, window_size, ratio)
        bounds = self.qk.sparse_bounds_from_topk(topk, seqlen)
        sink = bounds["sink_col"][0]
        # Row 0 sees no compressed block yet -> falls back to the window start.
        self.assertEqual(int(sink[0]), 0)
        # Later rows summarise sequence start via compressed block 0 (== seqlen).
        self.assertEqual(int(sink[seqlen - 1]), seqlen)

    def test_ranges_cover_a_superset_for_non_contiguous_topk(self):
        """A learned indexer selects a sparse compressed subset; ranges bracket it."""
        seqlen = 8
        window = paddle.arange(seqlen).unsqueeze(1).astype("int32")
        # Compressed picks blocks 0 and 3 only.
        compressed = paddle.to_tensor(
            [[seqlen, seqlen + 3]] * seqlen, dtype="int32"
        )
        topk = paddle.concat([window, compressed], axis=-1).unsqueeze(0)
        bounds = self.qk.sparse_bounds_from_topk(topk, seqlen)
        # The compressed range spans the selected blocks (and the gap between).
        self.assertEqual(int(bounds["cmp_lo"][0][0]), seqlen)
        self.assertEqual(int(bounds["cmp_hi"][0][0]), seqlen + 3)


class SparseQKKernelComputeTest(unittest.TestCase):
    """GPU numerical test: sparse kernel == explicit masked softmax reference."""

    @classmethod
    def setUpClass(cls):
        if (
            not paddle.device.is_compiled_with_cuda()
            or paddle.device.cuda.device_count() == 0
        ):
            raise unittest.SkipTest("qk_stats kernel requires a CUDA GPU")
        paddle.device.set_device("gpu:0")
        cls.qk = importlib.import_module(
            "paddlefleet.internal_medicine.backends.paddlefleet.qk_monitor"
        )

    @staticmethod
    def _reference_stats(query, kv_full, topk_idxs, scale, attn_sink):
        """Dense reference: one softmax over exactly the listed keys (+ sink)."""
        b, s, h, d = query.shape
        s_k = kv_full.shape[1]
        logits = paddle.matmul(
            query.transpose([0, 2, 1, 3]),
            kv_full.unsqueeze(1),
            transpose_y=True,
        )
        logits = logits * scale  # [B, H, S, S_k]

        allowed = paddle.zeros([b, s, s_k], dtype="bool")
        for bi in range(b):
            for row in range(s):
                for col in topk_idxs[bi, row].tolist():
                    if col >= 0:
                        allowed[bi, row, int(col)] = True
        mask = allowed.unsqueeze(1)

        neg = paddle.full_like(logits, -1e10)
        masked = paddle.where(mask, logits, neg)
        row_max = masked.max(axis=-1, keepdim=True)
        exp = paddle.where(
            mask, paddle.exp(masked - row_max), paddle.zeros_like(masked)
        )
        denom = exp.sum(axis=-1, keepdim=True)
        if attn_sink is not None:
            sink_logit = attn_sink.reshape([1, h, 1, 1])
            denom = denom + paddle.exp(sink_logit - row_max)
        probs = exp / denom
        entropy = -(probs * paddle.log(probs.clip(min=1e-30))).sum(axis=-1)
        if attn_sink is not None:
            p_sink = (
                paddle.exp(attn_sink.reshape([1, h, 1, 1]) - row_max) / denom
            )
            p_sink = p_sink.squeeze(-1)
            entropy = entropy - (p_sink * paddle.log(p_sink.clip(min=1e-30)))
        return {
            "entropy_per_head": entropy.mean(axis=-1),
            "max_per_head": masked.max(axis=-1).max(axis=-1),
        }

    def _hca_case(
        self, seqlen=32, window_size=8, ratio=4, heads=2, dim=32, seed=7
    ):
        paddle.seed(seed)
        n_compressed = seqlen // ratio
        query = paddle.randn([1, seqlen, heads, dim], dtype="float32")
        kv_full = paddle.randn([1, seqlen + n_compressed, dim], dtype="float32")
        topk = _build_hca_topk_idxs(seqlen, window_size, ratio)
        return query, kv_full, topk

    def test_sparse_stats_match_masked_reference(self):
        query, kv_full, topk = self._hca_case()
        scale = float(query.shape[-1]) ** -0.5
        stats = self.qk.compute_qk_stats_sparse_paddle(
            query, kv_full, topk, scale, attn_sink=None, row_stride=1
        )
        ref = self._reference_stats(query, kv_full, topk, scale, None)
        self.assertTrue(
            paddle.allclose(
                stats["entropy_per_head"],
                ref["entropy_per_head"],
                atol=1e-3,
                rtol=1e-3,
            ).item(),
            f"entropy mismatch: {stats['entropy_per_head']} vs {ref['entropy_per_head']}",
        )
        self.assertTrue(
            paddle.allclose(
                stats["max_per_head"], ref["max_per_head"], atol=1e-3, rtol=1e-3
            ).item()
        )

    def test_attn_sink_lowers_entropy_and_is_folded_into_denominator(self):
        query, kv_full, topk = self._hca_case(seed=11)
        scale = float(query.shape[-1]) ** -0.5
        heads = query.shape[2]
        sink = paddle.full([heads], 5.0, dtype="float32")  # dominant sink logit

        no_sink = self.qk.compute_qk_stats_sparse_paddle(
            query, kv_full, topk, scale, attn_sink=None, row_stride=1
        )
        with_sink = self.qk.compute_qk_stats_sparse_paddle(
            query, kv_full, topk, scale, attn_sink=sink, row_stride=1
        )
        ref = self._reference_stats(query, kv_full, topk, scale, sink)

        self.assertTrue(
            paddle.allclose(
                with_sink["entropy_per_head"],
                ref["entropy_per_head"],
                atol=1e-3,
                rtol=1e-3,
            ).item(),
            f"entropy with sink mismatch: {with_sink['entropy_per_head']} vs {ref['entropy_per_head']}",
        )
        # A dominant sink absorbs probability mass, so real-key entropy drops.
        self.assertLess(
            float(with_sink["entropy_global"]),
            float(no_sink["entropy_global"]),
        )
        # The sink is not a real key, so logit max/mean must be unchanged.
        self.assertTrue(
            paddle.allclose(
                with_sink["max_per_head"], no_sink["max_per_head"], atol=1e-6
            ).item()
        )


class PaddleAttnUpdateMonitorTest(unittest.TestCase):
    """delta2/delta3 = QK-product increment terms (arXiv:2606.28116 eq. 4)."""

    def setUp(self):
        training_logs.reset()
        paddle.seed(0)

    def tearDown(self):
        training_logs.reset()

    @staticmethod
    def _dense_reference(a, b, alpha=2.0):
        """Metrics straight off a full float64 SVD of the materialised product."""
        sigma = paddle.linalg.svd(
            (a @ b.T).astype("float64"), full_matrices=False
        )[1]
        squared = sigma * sigma
        total = squared.sum()
        weights = squared if alpha == 2.0 else squared ** (alpha / 2.0)
        p = (weights / weights.sum()).clip(min=1e-300)
        return {
            "norm": float(paddle.sqrt(total)),
            "stable_rank": float(total / squared.max()),
            "singular_spectrum": float(paddle.exp(-(p * p.log()).sum())),
        }

    def _assert_matches_dense(self, a, b, alpha=2.0, places=3):
        got = attn_update_module._spectrum_metrics(
            attn_update_module._squared_singular_values(a, b), alpha
        )
        reference = self._dense_reference(a, b, alpha)
        for name, expected in reference.items():
            self.assertAlmostEqual(
                float(got[name]) / expected,
                1.0,
                places=places,
                msg=f"{name}: {got[name]} vs {expected}",
            )

    def test_squared_singular_values_core_branch_matches_dense_svd(self):
        """2*head_dim < hidden takes the thin-QR core path; spectrum must be identical."""
        a = paddle.randn([64, 16])
        b = paddle.randn([64, 16])
        self.assertLess(a.shape[-1], a.shape[-2])
        self._assert_matches_dense(a, b, alpha=2.0)
        self._assert_matches_dense(a, b, alpha=1.0)

    def test_squared_singular_values_direct_branch_matches_dense_svd(self):
        """2*head_dim >= hidden makes the core no smaller than the product."""
        a = paddle.randn([32, 32])
        b = paddle.randn([32, 32])
        self._assert_matches_dense(a, b, alpha=2.0)
        self._assert_matches_dense(a, b, alpha=1.0)

    def test_spectrum_metrics_on_flat_spectrum_equals_rank(self):
        """k equal singular values: stable rank = S_alpha = k, norm = sqrt(sum of squares)."""
        metrics = attn_update_module._spectrum_metrics(paddle.full([8], 4.0))
        self.assertAlmostEqual(float(metrics["stable_rank"]), 8.0, places=4)
        self.assertAlmostEqual(
            float(metrics["singular_spectrum"]), 8.0, places=4
        )
        self.assertAlmostEqual(
            float(metrics["norm"]), math.sqrt(32.0), places=4
        )

    def test_spectrum_metrics_on_collapsed_spectrum_is_one(self):
        """All energy in one direction: both rank measures bottom out at 1."""
        metrics = attn_update_module._spectrum_metrics(
            paddle.to_tensor([9.0, 0.0, 0.0, 0.0])
        )
        self.assertAlmostEqual(float(metrics["stable_rank"]), 1.0, places=4)
        self.assertAlmostEqual(
            float(metrics["singular_spectrum"]), 1.0, places=3
        )
        self.assertAlmostEqual(float(metrics["norm"]), 3.0, places=4)

    def test_singular_spectrum_uses_alpha_two_weighting(self):
        """S_2 weights by sigma^2, so it differs from the sigma-weighted S_1."""
        squared = paddle.to_tensor([16.0, 4.0, 1.0])
        s2 = float(
            attn_update_module._spectrum_metrics(squared, 2.0)[
                "singular_spectrum"
            ]
        )
        s1 = float(
            attn_update_module._spectrum_metrics(squared, 1.0)[
                "singular_spectrum"
            ]
        )

        def entropy_rank(weights):
            total = sum(weights)
            return math.exp(
                -sum((w / total) * math.log(w / total) for w in weights)
            )

        self.assertAlmostEqual(s2, entropy_rank([16.0, 4.0, 1.0]), places=4)
        self.assertAlmostEqual(s1, entropy_rank([4.0, 2.0, 1.0]), places=4)
        self.assertLess(s2, s1)

    def test_batched_driver_matches_pair_by_pair_metrics(self):
        """Batching layers into one eigvalsh must not move any value."""
        pairs = [
            (paddle.randn([32, 12]), paddle.randn([32, 12])) for _ in range(5)
        ]
        batched = attn_update_module._spectrum_metrics_over_pairs(pairs)
        for index, (a, b) in enumerate(pairs):
            single = attn_update_module._spectrum_metrics(
                attn_update_module._squared_singular_values(a, b)
            )
            for key in attn_update_module._SPECTRUM_KEYS:
                self.assertAlmostEqual(
                    float(batched[key][index]) / float(single[key]),
                    1.0,
                    places=5,
                    msg=key,
                )

    def test_batched_driver_chunks_without_changing_results(self):
        """More pairs than fit in one chunk still line up with the flat order."""
        original = attn_update_module._MAX_BATCH_BYTES
        pairs = [
            (paddle.randn([16, 8]), paddle.randn([16, 8])) for _ in range(7)
        ]
        try:
            attn_update_module._MAX_BATCH_BYTES = (
                16 * 16 * 4 * 2
            )  # 2 pairs per chunk
            chunked = attn_update_module._spectrum_metrics_over_pairs(pairs)
        finally:
            attn_update_module._MAX_BATCH_BYTES = original
        flat = attn_update_module._spectrum_metrics_over_pairs(pairs)
        for key in attn_update_module._SPECTRUM_KEYS:
            self.assertEqual(chunked[key].shape, [7])
            self.assertTrue(
                paddle.allclose(chunked[key], flat[key], atol=1e-5).item(), key
            )

    @staticmethod
    def _attn(hidden=24, q_lora=12, head_dim=8, num_heads=2, with_norms=True):
        """Minimal DSv4-hybrid QK layout: q_down -> q_layernorm -> q_up, kv -> kv_layernorm."""
        attn = SimpleNamespace(
            linear_q_down_proj=SimpleNamespace(
                weight=paddle.randn([hidden, q_lora])
            ),
            linear_q_up_proj=SimpleNamespace(
                weight=paddle.randn([q_lora, head_dim * num_heads])
            ),
            linear_kv_proj=SimpleNamespace(
                weight=paddle.randn([hidden, head_dim])
            ),
        )
        if with_norms:
            attn.q_layernorm = SimpleNamespace(
                weight=paddle.rand([q_lora]) + 0.5
            )
            attn.kv_layernorm = SimpleNamespace(
                weight=paddle.rand([head_dim]) + 0.5
            )
        return attn

    @staticmethod
    def _perturb(attn, scale=0.05):
        """In-place parameter update, the way an optimizer step mutates weights."""
        for module in (
            attn.linear_q_down_proj,
            attn.linear_q_up_proj,
            attn.linear_kv_proj,
        ):
            paddle.assign(
                module.weight + scale * paddle.randn(module.weight.shape),
                module.weight,
            )

    def test_resolve_qk_factors_reads_head_layout(self):
        factors = attn_update_module.resolve_qk_factors(
            self._attn(head_dim=8, num_heads=2)
        )
        self.assertEqual(factors["head_dim"], 8)
        self.assertEqual(factors["num_heads"], 2)

    def test_resolve_qk_factors_returns_none_for_other_layouts(self):
        self.assertIsNone(
            attn_update_module.resolve_qk_factors(SimpleNamespace())
        )
        # q_up width not divisible by the kv head_dim -> not this circuit
        bad = self._attn(head_dim=8, num_heads=2)
        bad.linear_q_up_proj = SimpleNamespace(weight=paddle.randn([12, 13]))
        self.assertIsNone(attn_update_module.resolve_qk_factors(bad))

    def test_effective_wk_snapshot_does_not_alias_the_live_parameter(self):
        """Without a kv_layernorm the fp32 read is a no-op, so it must clone."""
        attn = self._attn(with_norms=False)
        factors = attn_update_module.resolve_qk_factors(attn)
        snapshot = attn_update_module.effective_wk(factors)
        before = snapshot.clone()
        self._perturb(attn, scale=1.0)
        self.assertTrue(paddle.allclose(snapshot, before).item())

    def test_effective_wq_folds_the_q_layernorm_scale(self):
        attn = self._attn(head_dim=8, num_heads=2)
        factors = attn_update_module.resolve_qk_factors(attn)
        expected = (
            attn.linear_q_down_proj.weight
            * attn.q_layernorm.weight.reshape([1, -1])
        ) @ attn.linear_q_up_proj.weight[:, 8:16]
        self.assertTrue(
            paddle.allclose(
                attn_update_module.effective_wq(factors, 1), expected, atol=1e-5
            ).item()
        )

    def _monitor_on(self, layers, **kwargs):
        monitor = PaddleAttnUpdateMonitor(monitor_interval=1, **kwargs)
        monitor.register_hooks(SimpleNamespace(layers=layers))
        return monitor

    def test_first_monitored_step_only_establishes_the_base_point(self):
        """delta2 needs two samples; the first step must emit nothing."""
        attn = self._attn()
        monitor = self._monitor_on([SimpleNamespace(self_attn=attn)])
        monitor.step()

        self.assertEqual(training_logs.get_latest(prefix="attn_update"), {})
        self._perturb(attn)
        monitor.step()
        self.assertIn(
            "attn_update/layer_0/delta2_norm",
            training_logs.get_latest(prefix="attn_update"),
        )

    def test_delta2_and_delta3_match_the_dense_increment_terms(self):
        """End-to-end: recorded metrics equal those of eq. (4)'s dense products."""
        attn = self._attn(hidden=24, q_lora=12, head_dim=8, num_heads=2)
        monitor = self._monitor_on(
            [SimpleNamespace(self_attn=attn)], num_heads_monitored=1
        )
        factors = attn_update_module.resolve_qk_factors(attn)
        wq_base = attn_update_module.effective_wq(factors, 0)
        wk_base = attn_update_module.effective_wk(factors)

        monitor.step()  # base point
        self._perturb(attn)
        monitor.step()  # increments against the base point

        wq_now = attn_update_module.effective_wq(factors, 0)
        wk_now = attn_update_module.effective_wk(factors)
        dq, dk = wq_now - wq_base, wk_now - wk_base
        dense = {
            "delta2": dq @ wk_base.T + wq_base @ dk.T,
            "delta3": dq @ dk.T,
        }

        latest = training_logs.get_latest(prefix="attn_update")
        for term, product in dense.items():
            reference = self._dense_reference(
                product, paddle.eye(product.shape[-1])
            )
            for key, expected in reference.items():
                got = latest[f"attn_update/layer_0/{term}_{key}"]
                self.assertAlmostEqual(
                    got / expected, 1.0, places=3, msg=f"{term}_{key}"
                )

    def test_delta3_is_second_order_and_far_smaller_than_delta2(self):
        """eq. (5): ||delta2||_F = O(||W|| ||dW||) vs ||delta3||_F = O(||dW||^2)."""
        attn = self._attn(hidden=24, q_lora=12, head_dim=8, num_heads=2)
        monitor = self._monitor_on([SimpleNamespace(self_attn=attn)])
        monitor.step()
        self._perturb(attn, scale=0.01)  # ||dW|| << ||W||
        monitor.step()

        latest = training_logs.get_latest(prefix="attn_update")
        self.assertLess(
            latest["attn_update/layer_0/delta3_norm"],
            latest["attn_update/layer_0/delta2_norm"],
        )

    def test_delta3_core_is_half_as_wide_as_delta2(self):
        """delta3's factors are [d, head_dim]; delta2 concatenates two of them."""
        attn = self._attn(head_dim=8, num_heads=2)
        monitor = self._monitor_on([SimpleNamespace(self_attn=attn)])
        monitor.step()
        self._perturb(attn)
        per_term = monitor._prepare_layer(0, monitor._layers[0][2])

        self.assertEqual(sorted(per_term), ["delta2", "delta3"])
        self.assertEqual(
            [list(m.shape) for m in per_term["delta2"][0]], [[24, 16], [24, 16]]
        )
        self.assertEqual(
            [list(m.shape) for m in per_term["delta3"][0]], [[24, 8], [24, 8]]
        )

    def test_delta2_is_per_layer_with_global_mean_over_layers(self):
        attn_a, attn_b = self._attn(), self._attn()
        monitor = self._monitor_on(
            [
                SimpleNamespace(self_attn=attn_a),
                SimpleNamespace(self_attn=attn_b),
            ]
        )
        monitor.step()
        self._perturb(attn_a)
        self._perturb(attn_b, scale=0.5)
        monitor.step()

        latest = training_logs.get_latest(prefix="attn_update")
        for name in attn_update_module.METRIC_NAMES:
            per_layer = [
                latest[f"attn_update/layer_{i}/{name}"] for i in (0, 1)
            ]
            self.assertAlmostEqual(
                latest[f"attn_update/global_{name}"],
                sum(per_layer) / 2.0,
                places=4,
                msg=name,
            )
        # A 10x larger update must show a larger increment norm.
        self.assertLess(
            latest["attn_update/layer_0/delta2_norm"],
            latest["attn_update/layer_1/delta2_norm"],
        )

    def test_per_layer_value_is_the_mean_over_monitored_heads(self):
        attn = self._attn(head_dim=8, num_heads=2)
        two_heads = self._monitor_on(
            [SimpleNamespace(self_attn=attn)], num_heads_monitored=2
        )
        one_head = self._monitor_on(
            [SimpleNamespace(self_attn=attn)], num_heads_monitored=1
        )
        for monitor in (two_heads, one_head):
            monitor.step()
        self._perturb(attn)

        one_head.step()
        head0 = training_logs.get_latest(prefix="attn_update")[
            "attn_update/layer_0/delta2_norm"
        ]
        training_logs.reset()
        two_heads.step()
        both = training_logs.get_latest(prefix="attn_update")[
            "attn_update/layer_0/delta2_norm"
        ]

        # Distinct heads give distinct increments, so the 2-head mean must move.
        self.assertNotAlmostEqual(head0, both, places=4)

    def test_mtp_layer_gets_its_own_suffixed_keys(self):
        main = SimpleNamespace(self_attn=self._attn())
        mtp_inner = SimpleNamespace(self_attn=self._attn())
        # An MTP id is num_hidden_layers + its local index, so the config has to be
        # reachable from the wrapper; without it the layer is skipped rather than
        # given an id derived from whatever main layers this rank happens to hold.
        wrapper = SimpleNamespace(
            transformer_layer=mtp_inner,
            config=SimpleNamespace(num_hidden_layers=1),
        )
        monitor = self._monitor_on([main, wrapper])
        monitor.step()
        self._perturb(main.self_attn)
        self._perturb(mtp_inner.self_attn)
        monitor.step()

        latest = training_logs.get_latest(prefix="attn_update")
        self.assertIn("attn_update/layer_0/delta2_norm", latest)
        self.assertIn("attn_update/layer_1_mtp/delta2_norm", latest)

    def test_unsupported_layers_are_skipped_without_declaring_metrics(self):
        monitor = PaddleAttnUpdateMonitor(monitor_interval=1)
        monitor.register_hooks(
            SimpleNamespace(
                layers=[SimpleNamespace(self_attn=SimpleNamespace())]
            )
        )
        monitor.step()

        self.assertEqual(monitor._layers, [])
        self.assertFalse(monitor._buffers_allocated)
        self.assertEqual(training_logs.get_latest(prefix="attn_update"), {})

    def test_monitor_interval_sets_the_sampling_distance(self):
        """Metrics only appear on steps that are multiples of monitor_interval."""
        attn = self._attn()
        monitor = PaddleAttnUpdateMonitor(monitor_interval=3)
        monitor.register_hooks(
            SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)])
        )

        monitor.step()  # step_count 0 -> base point
        for _ in range(2):
            self._perturb(attn)
            monitor.step()
            self.assertEqual(training_logs.get_latest(prefix="attn_update"), {})
        self._perturb(attn)
        monitor.step()  # step_count 3 -> first delta2, sampled 3 steps apart
        self.assertIn(
            "attn_update/layer_0/delta2_norm",
            training_logs.get_latest(prefix="attn_update"),
        )

    def test_registered_in_the_paddlefleet_monitor_map(self):
        self.assertIn("attn_update", paddlefleet_backend._MONITOR_MAP)
        registry = importlib.import_module(
            "paddlefleet.internal_medicine.core.registry"
        )
        self.assertIn("attn_update", registry.AVAILABLE_MONITORS["paddlefleet"])

    def test_setup_entry_point_registers_a_working_monitor(self):
        """The map holds ``setup_attn_update_monitor``, so that is what production calls.

        It has to hand the monitor back through ``monitor_dict`` -- nothing else
        keeps a reference, so ``step()`` would never be reached -- thread its
        keyword arguments into the monitor, and return the model untouched.
        """
        attn = self._attn()
        model = SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)])
        monitors = {}

        returned = attn_update_module.setup_attn_update_monitor(
            model,
            monitor_dict=monitors,
            monitor_interval=1,
            num_heads_monitored=2,
        )

        self.assertIs(returned, model)
        monitor = monitors["attn_update"]
        self.assertIsInstance(monitor, PaddleAttnUpdateMonitor)
        self.assertEqual(monitor.num_heads_monitored, 2)
        self.assertEqual([idx for idx, _t, _f in monitor._layers], [0])

        monitor.step()  # base point
        self._perturb(attn)
        monitor.step()
        self.assertIn(
            "attn_update/layer_0/delta2_norm",
            training_logs.get_latest(prefix="attn_update"),
        )

    # ── sampling interval ────────────────────────────────────────────────

    def test_sample_interval_defaults_to_monitor_interval(self):
        self.assertEqual(
            PaddleAttnUpdateMonitor(monitor_interval=7).monitor_interval, 7
        )

    def test_sample_interval_overrides_monitor_interval(self):
        """delta2 costs one eigensolve per sample, so it may sample more sparsely."""
        monitor = PaddleAttnUpdateMonitor(monitor_interval=1, sample_interval=4)
        self.assertEqual(monitor.monitor_interval, 4)

        attn = self._attn()
        monitor.register_hooks(
            SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)])
        )
        monitor.step()  # step_count 0 -> base point
        for _ in range(3):
            self._perturb(attn)
            monitor.step()
            self.assertEqual(training_logs.get_latest(prefix="attn_update"), {})
        self._perturb(attn)
        monitor.step()  # step_count 4 -> first delta2
        self.assertIn(
            "attn_update/layer_0/delta2_norm",
            training_logs.get_latest(prefix="attn_update"),
        )

    def test_sample_interval_rejects_non_positive_values(self):
        with self.assertRaises(ValueError):
            PaddleAttnUpdateMonitor(sample_interval=0)

    # ── attention layouts beyond DSv4-hybrid ─────────────────────────────

    @staticmethod
    def _mla_attn(
        hidden=16, q_lora=8, kv_lora=6, nope=4, rope=2, v_head_dim=5, heads=2
    ):
        """MLA: q_a -> q_a_layernorm -> q_b, kv_a -> kv_a_layernorm -> kv_b."""
        return SimpleNamespace(
            q_a_proj=SimpleNamespace(weight=paddle.randn([hidden, q_lora])),
            q_a_layernorm=SimpleNamespace(weight=paddle.rand([q_lora]) + 0.5),
            q_b_proj=SimpleNamespace(
                weight=paddle.randn([q_lora, heads * (nope + rope)])
            ),
            kv_a_proj_with_mqa=SimpleNamespace(
                weight=paddle.randn([hidden, kv_lora + rope])
            ),
            kv_a_layernorm=SimpleNamespace(weight=paddle.rand([kv_lora]) + 0.5),
            kv_b_proj=SimpleNamespace(
                weight=paddle.randn([kv_lora, heads * (nope + v_head_dim)])
            ),
            num_attention_heads_per_partition=heads,
            qk_nope_head_dim=nope,
            qk_rope_head_dim=rope,
        )

    def test_mla_layout_uses_only_the_nope_half_of_each_head(self):
        attn = self._mla_attn()
        factors = attn_update_module.resolve_qk_factors(attn)
        self.assertEqual(factors["kind"], "mla")
        self.assertEqual(
            factors["head_dim"], 4
        )  # qk_nope_head_dim, not q_head_dim
        self.assertEqual(factors["num_heads"], 2)

        # Head 1's content circuit composes through both latents.
        q_latent = attn.q_a_proj.weight * attn.q_a_layernorm.weight.reshape(
            [1, -1]
        )
        expected_q = (
            q_latent @ attn.q_b_proj.weight[:, 6:10]
        )  # head 1, first nope=4 of q_head_dim=6
        kv_latent = attn.kv_a_proj_with_mqa.weight[
            :, :6
        ] * attn.kv_a_layernorm.weight.reshape([1, -1])
        expected_k = (
            kv_latent @ attn.kv_b_proj.weight[:, 9:13]
        )  # head 1, first nope=4 of width=9
        self.assertTrue(
            paddle.allclose(
                attn_update_module.effective_wq(factors, 1),
                expected_q,
                atol=1e-5,
            ).item()
        )
        self.assertTrue(
            paddle.allclose(
                attn_update_module.effective_wk(factors, 1),
                expected_k,
                atol=1e-5,
            ).item()
        )

    def test_mla_without_q_lora_reads_q_proj_directly(self):
        attn = self._mla_attn()
        del attn.q_a_proj, attn.q_a_layernorm, attn.q_b_proj
        attn.q_proj = SimpleNamespace(weight=paddle.randn([16, 12]))
        factors = attn_update_module.resolve_qk_factors(attn)
        self.assertEqual(factors["kind"], "mla")
        self.assertIsNone(factors["q_a"])
        expected = attn.q_proj.weight[:, 6:10]
        self.assertTrue(
            paddle.allclose(
                attn_update_module.effective_wq(factors, 1), expected, atol=1e-5
            ).item()
        )

    @staticmethod
    def _fused_qkv_attn(hidden=16, head_dim=4, heads=4, kv_heads=2):
        """Standard attention: one qkv_proj grouped per KV head as Q|K|V."""
        group_dim = (heads // kv_heads) * head_dim + 2 * head_dim
        return SimpleNamespace(
            qkv_proj=SimpleNamespace(
                weight=paddle.randn([hidden, kv_heads * group_dim])
            ),
            num_attention_heads_per_partition=heads,
            num_query_groups_per_partition=kv_heads,
            hidden_size_per_attention_head=head_dim,
        )

    def test_fused_qkv_layout_slices_the_interleaved_group(self):
        attn = self._fused_qkv_attn()
        factors = attn_update_module.resolve_qk_factors(attn)
        self.assertEqual(factors["kind"], "fused_qkv")
        self.assertEqual(
            (factors["num_heads"], factors["num_kv_heads"]), (4, 2)
        )
        self.assertEqual(factors["group_dim"], 16)

        # Head 3 is the second query head of group 1: q at 16+4, k at 16+8.
        weight = attn.qkv_proj.weight
        self.assertTrue(
            paddle.allclose(
                attn_update_module.effective_wq(factors, 3), weight[:, 20:24]
            ).item()
        )
        self.assertTrue(
            paddle.allclose(
                attn_update_module.effective_wk(factors, 3), weight[:, 24:28]
            ).item()
        )
        # Heads 2 and 3 share group 1's single key matrix.
        self.assertTrue(
            paddle.allclose(
                attn_update_module.effective_wk(factors, 2),
                attn_update_module.effective_wk(factors, 3),
            ).item()
        )

    def test_fused_qkv_layout_rejects_a_width_that_contradicts_the_grouping(
        self,
    ):
        """The group arithmetic is checked against the real weight, not assumed."""
        attn = self._fused_qkv_attn()
        attn.qkv_proj = SimpleNamespace(weight=paddle.randn([16, 30]))
        self.assertIsNone(attn_update_module.resolve_qk_factors(attn))

    def test_split_qk_layout_maps_query_heads_onto_kv_groups(self):
        attn = SimpleNamespace(
            q_proj=SimpleNamespace(weight=paddle.randn([16, 16])),
            k_proj=SimpleNamespace(weight=paddle.randn([16, 8])),
            hidden_size_per_attention_head=4,
        )
        factors = attn_update_module.resolve_qk_factors(attn)
        self.assertEqual(factors["kind"], "split_qk")
        self.assertEqual(
            (factors["num_heads"], factors["num_kv_heads"]), (4, 2)
        )
        self.assertTrue(
            paddle.allclose(
                attn_update_module.effective_wq(factors, 2),
                attn.q_proj.weight[:, 8:12],
            ).item()
        )
        self.assertTrue(
            paddle.allclose(
                attn_update_module.effective_wk(factors, 2),
                attn.k_proj.weight[:, 4:8],
            ).item()
        )

    def test_split_qk_folds_a_per_head_qk_norm(self):
        attn = SimpleNamespace(
            q_proj=SimpleNamespace(weight=paddle.randn([16, 8])),
            k_proj=SimpleNamespace(weight=paddle.randn([16, 8])),
            q_norm=SimpleNamespace(weight=paddle.rand([4]) + 0.5),
            hidden_size_per_attention_head=4,
        )
        factors = attn_update_module.resolve_qk_factors(attn)
        expected = attn.q_proj.weight[:, 4:8] * attn.q_norm.weight.reshape(
            [1, -1]
        )
        self.assertTrue(
            paddle.allclose(
                attn_update_module.effective_wq(factors, 1), expected, atol=1e-5
            ).item()
        )

    def test_split_qk_slices_a_per_layer_qk_norm_by_head(self):
        """A norm covering all heads at once must be sliced, not broadcast.

        Folding head 1's columns with head 0's scale would silently corrupt the
        QK circuit, so the head offset is checked on both heads.
        """
        attn = SimpleNamespace(
            q_proj=SimpleNamespace(weight=paddle.randn([16, 8])),
            k_proj=SimpleNamespace(weight=paddle.randn([16, 8])),
            q_norm=SimpleNamespace(weight=paddle.rand([8]) + 0.5),
            hidden_size_per_attention_head=4,
        )
        factors = attn_update_module.resolve_qk_factors(attn)
        self.assertNotEqual(attn.q_norm.weight.shape[0], factors["head_dim"])

        for head in (0, 1):
            start = head * 4
            scale = attn.q_norm.weight[start : start + 4].reshape([1, -1])
            expected = attn.q_proj.weight[:, start : start + 4] * scale
            got = attn_update_module.effective_wq(factors, head)
            self.assertTrue(
                paddle.allclose(got, expected, atol=1e-5).item(), f"head {head}"
            )

    def test_qk_norm_of_an_unreadable_width_is_dropped_rather_than_guessed(
        self,
    ):
        """Neither per-head nor per-layer: fold nothing instead of the wrong slice."""
        attn = SimpleNamespace(
            q_proj=SimpleNamespace(weight=paddle.randn([16, 8])),
            k_proj=SimpleNamespace(weight=paddle.randn([16, 8])),
            q_norm=SimpleNamespace(weight=paddle.rand([5]) + 0.5),
            hidden_size_per_attention_head=4,
        )
        factors = attn_update_module.resolve_qk_factors(attn)
        got = attn_update_module.effective_wq(factors, 1)
        self.assertTrue(
            paddle.allclose(got, attn.q_proj.weight[:, 4:8], atol=1e-5).item()
        )

    def test_mixed_layouts_in_one_model_are_bucketed_by_shape(self):
        """Different circuit widths cannot share a batched eigensolve."""
        dsv4 = self._attn(hidden=16, q_lora=8, head_dim=8, num_heads=2)
        split = SimpleNamespace(
            q_proj=SimpleNamespace(weight=paddle.randn([16, 16])),
            k_proj=SimpleNamespace(weight=paddle.randn([16, 16])),
            hidden_size_per_attention_head=4,
        )
        monitor = self._monitor_on(
            [SimpleNamespace(self_attn=dsv4), SimpleNamespace(self_attn=split)]
        )
        self.assertEqual(
            [f["kind"] for _i, _t, f in monitor._layers],
            ["dsv4_hybrid", "split_qk"],
        )

        monitor.step()
        self._perturb(dsv4)
        paddle.assign(
            split.q_proj.weight + 0.05 * paddle.randn([16, 16]),
            split.q_proj.weight,
        )
        monitor.step()

        latest = training_logs.get_latest(prefix="attn_update")
        for name in attn_update_module.METRIC_NAMES:
            self.assertIn(f"attn_update/layer_0/{name}", latest)
            self.assertIn(f"attn_update/layer_1/{name}", latest)

    def test_bucketed_pairs_match_per_pair_results_in_caller_order(self):
        pairs = [
            (paddle.randn([12, 4]), paddle.randn([12, 4])),
            (paddle.randn([12, 6]), paddle.randn([12, 6])),
            (paddle.randn([12, 4]), paddle.randn([12, 4])),
        ]
        bucketed = attn_update_module._spectrum_metrics_over_pairs(pairs)
        for index, (a, b) in enumerate(pairs):
            single = attn_update_module._spectrum_metrics(
                attn_update_module._squared_singular_values(a, b)
            )
            for key in attn_update_module._SPECTRUM_KEYS:
                self.assertAlmostEqual(
                    float(bucketed[key][index]),
                    float(single[key]),
                    places=4,
                    msg=f"{key}[{index}]",
                )


if __name__ == "__main__":
    unittest.main()


class PaddleMLPUpdateMonitorTest(unittest.TestCase):
    """dW of the expert MLP, split per (layer, expert, gate/up/down)."""

    EXPERTS = 6
    LATENT = 32
    INTER = 16

    def setUp(self):
        training_logs.reset()
        paddle.seed(0)

    def tearDown(self):
        training_logs.reset()

    def _moe_layer(self):
        return SimpleNamespace(
            gate=SimpleNamespace(),
            experts=None,
            grouped_gemm_experts=SimpleNamespace(
                weight1=paddle.randn(
                    [self.EXPERTS, self.LATENT, 2 * self.INTER]
                ),
                weight2=paddle.randn([self.EXPERTS, self.INTER, self.LATENT]),
            ),
            shared_experts=SimpleNamespace(
                up_gate_proj=SimpleNamespace(
                    weight=paddle.randn([self.LATENT, 2 * self.INTER])
                ),
                down_proj=SimpleNamespace(
                    weight=paddle.randn([self.INTER, self.LATENT])
                ),
            ),
        )

    def _monitor(self, moe_layer, **kwargs):
        monitor = PaddleMLPUpdateMonitor(monitor_interval=1, **kwargs)
        monitor._layers = [(0, moe_layer)]
        for name in mlp_update_module.metric_names(
            monitor.log_spectrum, with_shared=True
        ):
            monitor.declare_layer_metric(0, name)
        monitor.allocate_buffers()
        return monitor

    @staticmethod
    def _bump(weight, scale=0.01):
        return weight + paddle.randn(weight.shape) * scale

    def _latest(self):
        return {
            key.split("/")[-1]: value
            for key, value in training_logs.get_latest(
                prefix="mlp_update"
            ).items()
        }

    def test_gate_and_up_split_the_fused_fc1_along_the_intermediate_axis(self):
        """gate | up must partition fc1's output width exactly, gate first."""
        fc1 = paddle.arange(
            self.EXPERTS * self.LATENT * 2 * self.INTER, dtype="float32"
        ).reshape([self.EXPERTS, self.LATENT, 2 * self.INTER])
        gate = moe_monitor_module._swiglu_gate_half(fc1)
        up = mlp_update_module._swiglu_up_half(fc1)
        self.assertEqual(
            list(gate.shape), [self.EXPERTS, self.LATENT, self.INTER]
        )
        self.assertTrue(bool(paddle.all(gate == fc1[..., : self.INTER])))
        self.assertTrue(bool(paddle.all(up == fc1[..., self.INTER :])))
        self.assertTrue(
            bool(paddle.all(paddle.concat([gate, up], axis=-1) == fc1))
        )

    def test_first_collect_only_establishes_the_base_point(self):
        """One reading cannot give an increment, so nothing may be logged."""
        monitor = self._monitor(self._moe_layer())
        monitor.collect_expert_norms()
        monitor.step()
        self.assertEqual(training_logs.get_latest(prefix="mlp_update"), {})

    def test_relative_update_matches_the_frobenius_ratio_per_expert(self):
        """r = ||dW||_F / ||W||_F, measured on each expert's own gate matrix."""
        moe_layer = self._moe_layer()
        monitor = self._monitor(moe_layer)
        monitor.collect_expert_norms()
        monitor.step()

        base = moe_layer.grouped_gemm_experts.weight1
        delta = paddle.randn(base.shape) * 0.02
        moe_layer.grouped_gemm_experts.weight1 = base + delta
        training_logs.reset()
        monitor.collect_expert_norms()
        monitor.step()

        now_gate = (base + delta)[..., : self.INTER].astype("float64")
        delta_gate = delta[..., : self.INTER].astype("float64")
        reference = paddle.sqrt(
            (delta_gate**2).sum(axis=[-2, -1])
        ) / paddle.sqrt((now_gate**2).sum(axis=[-2, -1]))
        latest = self._latest()
        self.assertAlmostEqual(
            latest["gate_rel_update_mean"], float(reference.mean()), places=6
        )
        self.assertAlmostEqual(
            latest["gate_rel_update_max"], float(reference.max()), places=6
        )
        self.assertAlmostEqual(
            latest["gate_rel_update_min"], float(reference.min()), places=6
        )

    def _stable_rank_against_dense(self, delta):
        """``(monitor value, dense float64 SVD value)`` for the gate half of ``delta``."""
        moe_layer = self._moe_layer()
        monitor = self._monitor(moe_layer)
        monitor.collect_expert_norms()
        monitor.step()
        moe_layer.grouped_gemm_experts.weight1 = (
            moe_layer.grouped_gemm_experts.weight1 + delta
        )
        training_logs.reset()
        monitor.collect_expert_norms()
        monitor.step()

        ranks = []
        for expert in range(self.EXPERTS):
            sigma = paddle.linalg.svd(
                delta[expert, :, : self.INTER].astype("float64"),
                full_matrices=False,
            )[1]
            ranks.append(float((sigma**2).sum() / sigma.max() ** 2))
        return self._latest()["gate_stable_rank_mean"], sum(ranks) / len(ranks)

    def test_stable_rank_is_exact_when_the_update_concentrates(self):
        """A dominant direction makes the power iteration converge immediately.

        This is the case the metric exists for, so it has to be tight here.
        """
        base = paddle.randn([self.EXPERTS, self.LATENT, 2 * self.INTER]) * 0.02
        left = paddle.randn([self.EXPERTS, self.LATENT, 1])
        right = paddle.randn([self.EXPERTS, 1, 2 * self.INTER])
        delta = base + 20.0 * paddle.matmul(left, right)
        got, reference = self._stable_rank_against_dense(delta)
        self.assertAlmostEqual(got / reference, 1.0, places=5)

    def test_stable_rank_bias_on_a_flat_update_is_small_and_one_sided(self):
        """A near-flat spectrum converges slowly, and only ever understates sigma_1.

        An understated sigma_1 inflates the stable rank, so the deviation must be
        upward and within a few percent -- the regime where the exact value does
        not change the reading anyway.
        """
        delta = paddle.randn([self.EXPERTS, self.LATENT, 2 * self.INTER]) * 0.02
        got, reference = self._stable_rank_against_dense(delta)
        self.assertGreaterEqual(got, reference - 1e-6)
        self.assertLess(got / reference, 1.05)

    def test_each_projection_is_measured_on_its_own_scale(self):
        """A large down update must not leak into the gate / up readings."""
        moe_layer = self._moe_layer()
        monitor = self._monitor(moe_layer)
        monitor.collect_expert_norms()
        monitor.step()

        experts = moe_layer.grouped_gemm_experts
        experts.weight1 = self._bump(experts.weight1, 0.01)
        experts.weight2 = self._bump(experts.weight2, 0.20)
        training_logs.reset()
        monitor.collect_expert_norms()
        monitor.step()

        latest = self._latest()
        self.assertGreater(
            latest["down_rel_update_mean"], 5 * latest["gate_rel_update_mean"]
        )
        self.assertAlmostEqual(
            latest["gate_rel_update_mean"],
            latest["up_rel_update_mean"],
            places=2,
        )

    def test_a_frozen_expert_reads_zero_update_and_unit_stable_rank(self):
        """dW = 0 is 0/0 for the stable rank; it must land on the lower bound 1."""
        moe_layer = self._moe_layer()
        monitor = self._monitor(moe_layer)
        monitor.collect_expert_norms()
        monitor.step()

        delta = (
            paddle.randn(moe_layer.grouped_gemm_experts.weight1.shape) * 0.02
        )
        delta[0] *= 0.0
        moe_layer.grouped_gemm_experts.weight1 = (
            moe_layer.grouped_gemm_experts.weight1 + delta
        )
        training_logs.reset()
        monitor.collect_expert_norms()
        monitor.step()

        latest = self._latest()
        self.assertAlmostEqual(latest["gate_rel_update_min"], 0.0, places=9)
        self.assertAlmostEqual(latest["gate_stable_rank_min"], 1.0, places=6)
        self.assertGreater(latest["gate_stable_rank_max"], 1.0)

    def test_expert_score_keeps_a_single_anomalous_projection_visible(self):
        """S[e] = max_m z(r_m[e]); an up-only outlier must not be averaged away."""
        moe_layer = self._moe_layer()
        monitor = self._monitor(moe_layer)
        monitor.collect_expert_norms()
        monitor.step()

        experts = moe_layer.grouped_gemm_experts
        delta = paddle.randn(experts.weight1.shape) * 0.01
        delta[2, :, self.INTER :] *= 25.0
        experts.weight1 = experts.weight1 + delta
        experts.weight2 = self._bump(experts.weight2, 0.01)
        training_logs.reset()
        monitor.collect_expert_norms()
        monitor.step()

        latest = self._latest()
        # A z-score over n samples cannot exceed (n - 1) / sqrt(n), so the metric
        # saturates: it is an outlier detector, not a magnitude.
        ceiling = (self.EXPERTS - 1) / math.sqrt(self.EXPERTS)
        self.assertGreater(latest["update_zmax_max"], 1.5)
        self.assertLessEqual(latest["update_zmax_max"], ceiling + 1e-4)
        self.assertGreater(latest["update_zmax_max"], latest["update_zmax_p90"])

    def test_spectrum_rides_a_coarser_clock_than_the_relative_updates(self):
        """log_spectrum with spectrum_interval=3: entropy on every third sample."""
        moe_layer = self._moe_layer()
        monitor = self._monitor(
            moe_layer, log_spectrum=True, spectrum_interval=3
        )
        seen = []
        for _ in range(7):
            experts = moe_layer.grouped_gemm_experts
            experts.weight1 = self._bump(experts.weight1)
            training_logs.reset()
            monitor.collect_expert_norms()
            monitor.step()
            keys = training_logs.get_latest(prefix="mlp_update")
            seen.append(
                (bool(keys), any("singular_entropy" in key for key in keys))
            )

        self.assertEqual(
            [measured for measured, _ in seen], [False] + [True] * 6
        )
        self.assertEqual(
            [spectrum for _, spectrum in seen],
            [False, True, False, False, True, False, False],
        )

    # ── discovery and wiring ─────────────────────────────────────────────

    def test_discovery_accepts_the_three_expert_block_layouts(self):
        """The expert block hangs off ``layer.mlp`` on this model family, off
        ``layer.moe`` elsewhere, and a bare ``MoELayer`` is its own module.
        """
        by_mlp = SimpleNamespace(mlp=self._moe_layer())
        by_moe = SimpleNamespace(moe=self._moe_layer())
        monitor = PaddleMLPUpdateMonitor(monitor_interval=1)

        found = mlp_update_module._find_moe_layers(
            SimpleNamespace(layers=[by_mlp, by_moe]), 0, monitor.mark_mtp_layers
        )

        self.assertEqual([idx for idx, _module in found], [0, 1])
        self.assertIs(found[0][1], by_mlp.mlp)
        self.assertIs(found[1][1], by_moe.moe)

    def test_discovery_falls_back_to_named_sublayers_for_a_bare_moe_layer(self):
        """A stack exposing no decoder-layer list is searched by class name."""
        latent, experts = self.LATENT, self.EXPERTS

        class MoELayer(nn.Layer):
            def __init__(self):
                super().__init__()
                self.gate = nn.Linear(latent, experts)

        class Block(nn.Layer):
            def __init__(self):
                super().__init__()
                self.block = MoELayer()

        model = Block()
        self.assertIsNone(layer_discovery.get_decoder_layers(model))

        monitor = PaddleMLPUpdateMonitor(monitor_interval=1)
        found = mlp_update_module._find_moe_layers(
            model, 0, monitor.mark_mtp_layers
        )

        self.assertEqual([idx for idx, _module in found], [0])
        self.assertIs(found[0][1], model.block)

        class Dense(nn.Layer):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(latent, experts)

        self.assertEqual(
            mlp_update_module._find_moe_layers(
                Dense(), 0, monitor.mark_mtp_layers
            ),
            [],
        )

    def test_down_weight_reads_every_expert_layout(self):
        """Mirrors ``_expert_fc1_weight``: grouped-gemm ``weight2``, a fused
        ``down_proj``, or a per-expert list that has to be stacked.
        """
        shape = [self.INTER, self.LATENT]
        fused = SimpleNamespace(
            grouped_gemm_experts=None,
            experts=SimpleNamespace(
                down_proj=SimpleNamespace(
                    weight=paddle.randn([self.EXPERTS, *shape])
                )
            ),
        )
        listed = SimpleNamespace(
            grouped_gemm_experts=None,
            experts=[
                SimpleNamespace(
                    down_proj=SimpleNamespace(weight=paddle.randn(shape))
                )
                for _ in range(3)
            ],
        )

        self.assertEqual(
            list(mlp_update_module._expert_fc2_weight(fused).shape),
            [self.EXPERTS, *shape],
        )
        self.assertEqual(
            list(mlp_update_module._expert_fc2_weight(listed).shape),
            [3, *shape],
        )
        for empty in (None, []):
            layer = SimpleNamespace(grouped_gemm_experts=None, experts=empty)
            self.assertIsNone(mlp_update_module._expert_fc2_weight(layer))

    def test_setup_entry_point_registers_a_working_monitor(self):
        """``_MONITOR_MAP`` holds ``setup_mlp_update_monitor``, so that is what
        production calls. It has to hand the monitor back through ``monitor_dict``
        -- nothing else keeps a reference, so ``collect_expert_norms`` would never
        be reached -- thread its keyword arguments through, and return the model
        untouched.
        """
        moe_layer = self._moe_layer()
        model = SimpleNamespace(layers=[SimpleNamespace(mlp=moe_layer)])
        monitors = {}

        returned = mlp_update_module.setup_mlp_update_monitor(
            model,
            monitor_dict=monitors,
            monitor_interval=1,
            log_spectrum=True,
            spectrum_interval=4,
        )

        self.assertIs(returned, model)
        monitor = monitors["mlp_update"]
        self.assertIsInstance(monitor, PaddleMLPUpdateMonitor)
        self.assertTrue(monitor.log_spectrum)
        self.assertEqual(monitor.spectrum_interval, 4)
        self.assertEqual([idx for idx, _module in monitor._layers], [0])

        monitor.collect_expert_norms()  # base point only
        monitor.step()
        experts = moe_layer.grouped_gemm_experts
        experts.weight1 = self._bump(experts.weight1)
        monitor.collect_expert_norms()
        monitor.step()

        self.assertIn(
            "mlp_update/layer_0/gate_rel_update_mean",
            training_logs.get_latest(prefix="mlp_update"),
        )

        # monitor_dict is optional; without one the model still comes back unchanged.
        self.assertIs(mlp_update_module.setup_mlp_update_monitor(model), model)

    def test_a_layer_without_a_shared_expert_declares_no_shared_keys(self):
        """Declaring a key the layer can never record would publish a series that
        stays permanently absent from the log.
        """
        moe_layer = self._moe_layer()
        moe_layer.shared_experts = None
        monitor = PaddleMLPUpdateMonitor(monitor_interval=1)
        monitor.register_hooks(
            SimpleNamespace(layers=[SimpleNamespace(mlp=moe_layer)])
        )

        declared = {key for key in monitor._gpu_cnt if "/layer_0/" in key}
        self.assertIn("mlp_update/layer_0/gate_rel_update_mean", declared)
        self.assertNotIn("mlp_update/layer_0/shared_gate_rel_update", declared)
        self.assertEqual(
            len(declared), 33
        )  # 36 minus one shared key per projection

    def test_a_model_without_moe_layers_allocates_nothing(self):
        """A dense stack must leave the monitor inert, not half-initialised."""
        monitor = PaddleMLPUpdateMonitor(monitor_interval=1)
        monitor.register_hooks(
            SimpleNamespace(
                layers=[SimpleNamespace(self_attn=SimpleNamespace())]
            )
        )

        self.assertEqual(monitor._layers, [])
        self.assertFalse(monitor._buffers_allocated)
        monitor.collect_expert_norms()
        monitor.step()
        self.assertEqual(training_logs.get_latest(prefix="mlp_update"), {})

    def test_remove_hooks_releases_the_base_snapshots(self):
        """The snapshots are the monitor's entire resident cost, so teardown has to
        drop them rather than merely stop measuring.
        """
        monitor = self._monitor(self._moe_layer())
        monitor.collect_expert_norms()
        self.assertEqual(list(monitor._snapshots), [0])

        monitor.remove_hooks()

        self.assertEqual(monitor._snapshots, {})
        self.assertEqual(monitor._layers, [])

    def test_sample_interval_overrides_monitor_interval(self):
        """The snapshot pair is the resident cost, so it may sample more sparsely
        than the shared monitor clock.
        """
        self.assertEqual(
            PaddleMLPUpdateMonitor(monitor_interval=7).monitor_interval, 7
        )
        self.assertEqual(
            PaddleMLPUpdateMonitor(
                monitor_interval=1, sample_interval=4
            ).monitor_interval,
            4,
        )

    def test_both_clocks_reject_non_positive_values(self):
        """``spectrum_interval`` is a modulus; 0 would raise mid-training instead."""
        with self.assertRaises(ValueError):
            PaddleMLPUpdateMonitor(sample_interval=0)
        with self.assertRaises(ValueError):
            PaddleMLPUpdateMonitor(spectrum_interval=0)

    def test_a_layer_that_raises_does_not_stop_the_later_layers(self):
        """One unreadable expert block must cost its own layer's metrics, no more."""
        good = self._moe_layer()
        monitor = PaddleMLPUpdateMonitor(monitor_interval=1, verbose=True)
        monitor._layers = [(0, BrokenPaddleMoELayer()), (1, good)]
        for layer_idx in (0, 1):
            for name in mlp_update_module.metric_names(False, with_shared=True):
                monitor.declare_layer_metric(layer_idx, name)
        monitor.allocate_buffers()

        monitor.collect_expert_norms()
        monitor.step()
        experts = good.grouped_gemm_experts
        experts.weight1 = self._bump(experts.weight1)
        training_logs.reset()
        monitor.collect_expert_norms()
        monitor.step()

        latest = training_logs.get_latest(prefix="mlp_update")
        self.assertIn("mlp_update/layer_1/gate_rel_update_mean", latest)
        self.assertNotIn("mlp_update/layer_0/gate_rel_update_mean", latest)

    def test_a_weight_that_changes_shape_rebases_instead_of_differencing(self):
        """After a resume or reshard the two readings describe different storage, so
        the pair is skipped and the fresh snapshot becomes the new base point.
        """
        moe_layer = self._moe_layer()
        monitor = self._monitor(moe_layer)
        monitor.collect_expert_norms()
        monitor.step()

        experts = moe_layer.grouped_gemm_experts
        experts.weight1 = paddle.randn(
            [self.EXPERTS + 2, self.LATENT, 2 * self.INTER]
        )
        experts.weight2 = paddle.randn(
            [self.EXPERTS + 2, self.INTER, self.LATENT]
        )
        training_logs.reset()
        monitor.collect_expert_norms()
        monitor.step()

        latest = training_logs.get_latest(prefix="mlp_update")
        self.assertNotIn("mlp_update/layer_0/gate_rel_update_mean", latest)
        self.assertNotIn("mlp_update/layer_0/update_zmax_max", latest)

        experts.weight1 = self._bump(experts.weight1)
        training_logs.reset()
        monitor.collect_expert_norms()
        monitor.step()
        self.assertIn(
            "mlp_update/layer_0/gate_rel_update_mean",
            training_logs.get_latest(prefix="mlp_update"),
        )

    def test_registered_in_the_paddlefleet_monitor_map(self):
        """The yaml switch resolves through _MONITOR_MAP."""
        self.assertIn("mlp_update", paddlefleet_backend._MONITOR_MAP)
        self.assertIs(
            paddlefleet_backend._MONITOR_MAP["mlp_update"],
            mlp_update_module.setup_mlp_update_monitor,
        )


class MetricFamilySelectionTest(unittest.TestCase):
    """Family selection has to bite at declare time, not at record or flush time.

    The schema must be complete before ``allocate_buffers``, and a key filtered
    any later would still cost its GPU accumulator and its slot in the cross-rank
    reduction — the whole point is to make those smaller.
    """

    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_excluded_families_never_reach_the_schema(self):
        monitor = PaddleMoEMonitor(
            log_per_layer=True, log_global=True, exclude_families="balance+norm"
        )
        monitor.declare_layer_metric(0, "router_entropy")
        monitor.declare_layer_metric(0, "gate_mass_max_min_ratio")
        monitor.declare_layer_metric(0, "expert_norm_mean")

        declared = monitor._mean_keys | monitor._max_keys | monitor._min_keys
        self.assertIn("moe_health/layer_0/router_entropy", declared)
        self.assertNotIn("moe_health/layer_0/gate_mass_max_min_ratio", declared)
        self.assertNotIn("moe_health/layer_0/expert_norm_mean", declared)
        # No global group either: a global is derived from its layer keys.
        self.assertEqual(
            list(monitor._layer_metric_groups),
            ["moe_health/global_router_entropy"],
        )

    def test_recording_a_filtered_key_is_a_silent_no_op(self):
        """Monitors keep calling record_* unconditionally; it must not raise."""
        monitor = PaddleMoEMonitor(
            log_per_layer=True, log_global=True, exclude_families="norm"
        )
        monitor.declare_layer_metric(0, "router_entropy")
        monitor.declare_layer_metric(0, "expert_norm_mean")
        monitor.allocate_buffers()

        monitor.record_layer_metric(0, "router_entropy", paddle.to_tensor(1.5))
        monitor.record_layer_metric(
            0, "expert_norm_mean", paddle.to_tensor(9.0)
        )
        monitor.step()

        latest = training_logs.get_latest(prefix="moe_health")
        self.assertAlmostEqual(
            latest["moe_health/layer_0/router_entropy"], 1.5, places=5
        )
        self.assertNotIn("moe_health/layer_0/expert_norm_mean", latest)

    def test_vector_families_are_filtered_too(self):
        """Per-expert vectors are the biggest family; they must be switchable."""
        off = PaddleMoEMonitor(
            log_per_layer=True, log_global=False, exclude_families="expert"
        )
        off.declare_layer_vector(0, "expert_token_share", 4)
        off.allocate_buffers()
        self.assertEqual(off._vector_keys, {})
        # record on an undeclared vector already no-ops via the missing buffer
        off.record_layer_vector(
            0, "expert_token_share", paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        )
        off.step()
        self.assertEqual(training_logs.get_latest(prefix="moe_health"), {})

        training_logs.reset()
        on = PaddleMoEMonitor(log_per_layer=True, log_global=False)
        on.declare_layer_vector(0, "expert_token_share", 4)
        self.assertEqual(
            on._vector_keys, {"moe_health/layer_0/expert_token_share": 4}
        )

    def test_explicit_declare_paths_are_filtered(self):
        """declare_mean/max/min are the other way monitors register keys."""
        monitor = PaddleMoEMonitor(
            log_per_layer=True, log_global=True, exclude_families="norm"
        )
        monitor.declare_mean("moe_health/global_router_entropy")
        monitor.declare_mean("moe_health/global_expert_norm_mean")
        self.assertIn("moe_health/global_router_entropy", monitor._mean_keys)
        self.assertNotIn(
            "moe_health/global_expert_norm_mean", monitor._mean_keys
        )
        self.assertIn(
            "moe_health/global_expert_norm_mean", monitor._disabled_keys
        )

    def test_the_default_declares_exactly_what_it_used_to(self):
        """Nothing excluded must be byte-identical to the old behaviour."""
        names = (
            "router_entropy",
            "gate_mass_max_min_ratio",
            "expert_norm_mean",
            "shared_frac",
        )
        for exclude in (None, "none"):
            monitor = PaddleMoEMonitor(
                log_per_layer=True, log_global=True, exclude_families=exclude
            )
            for name in names:
                monitor.declare_layer_metric(0, name)
            declared = (
                monitor._mean_keys | monitor._max_keys | monitor._min_keys
            )
            for name in names:
                self.assertIn(
                    f"moe_health/layer_0/{name}",
                    declared,
                    f"exclude={exclude!r}",
                )
            self.assertEqual(
                monitor._disabled_keys, set(), f"exclude={exclude!r}"
            )

    def test_an_unknown_family_name_raises_at_construction(self):
        metric_families = importlib.import_module(
            "paddlefleet.internal_medicine.core.metric_families"
        )
        with self.assertRaises(metric_families.UnknownFamilyError):
            PaddleMoEMonitor(
                log_per_layer=True, log_global=True, exclude_families="routerr"
            )

    def test_setup_monitors_routes_the_spec_string_to_the_right_monitor(self):
        """`"mhc_health:mix, moe_health:expert"` must reach each monitor separately."""
        seen = {}
        original = dict(paddlefleet_backend._MONITOR_MAP)

        def make(name):
            def setup(
                _model, monitor_dict=None, exclude_families=None, **_kwargs
            ):
                seen[name] = exclude_families
                monitor_dict[name] = DummyMonitor()

            return setup

        paddlefleet_backend._MONITOR_MAP.clear()
        paddlefleet_backend._MONITOR_MAP.update(
            {n: make(n) for n in ("mhc_health", "moe_health", "qk_stats")}
        )
        try:
            paddlefleet_backend.setup_monitors(
                SimpleNamespace(),
                monitors=["mhc_health", "moe_health", "qk_stats"],
                exclude_families="mhc_health:mix, moe_health:expert+act",
            )
        finally:
            paddlefleet_backend._MONITOR_MAP.clear()
            paddlefleet_backend._MONITOR_MAP.update(original)

        self.assertEqual(seen["mhc_health"], ["mix"])
        self.assertEqual(seen["moe_health"], ["expert", "act"])
        # Not named in the spec: stays None, i.e. collects everything.
        self.assertIsNone(seen["qk_stats"])

    def test_setup_monitors_without_the_option_changes_nothing(self):
        seen = {}
        original = dict(paddlefleet_backend._MONITOR_MAP)

        def setup(_model, monitor_dict=None, exclude_families=None, **_kwargs):
            seen["moe_health"] = exclude_families
            monitor_dict["moe_health"] = DummyMonitor()

        paddlefleet_backend._MONITOR_MAP.clear()
        paddlefleet_backend._MONITOR_MAP.update({"moe_health": setup})
        try:
            paddlefleet_backend.setup_monitors(
                SimpleNamespace(), monitors=["moe_health"]
            )
        finally:
            paddlefleet_backend._MONITOR_MAP.clear()
            paddlefleet_backend._MONITOR_MAP.update(original)
        self.assertIsNone(seen["moe_health"])

    def test_setup_moe_monitor_forwards_the_exclusion_to_the_monitor(self):
        """End to end through the real setup function, not a stub."""
        monitor_dict = {}
        moe_monitor_module.setup_moe_monitor(
            SimpleNamespace(layers=[]),
            monitor_dict=monitor_dict,
            exclude_families="expert+act",
        )
        self.assertEqual(
            monitor_dict["moe_health"].family_selection.excluded,
            frozenset({"expert", "act"}),
        )

    def test_setup_monitors_rejects_a_misspelled_monitor_name(self):
        """Otherwise the loop simply never matches it and nothing is switched off."""
        metric_families = importlib.import_module(
            "paddlefleet.internal_medicine.core.metric_families"
        )
        with self.assertRaises(metric_families.UnknownMonitorError):
            paddlefleet_backend.setup_monitors(
                SimpleNamespace(),
                monitors=["moe_health"],
                exclude_families="moe:expert",
            )


class _FakeDispatcher(nn.Layer):
    """``token_dispatcher``: the count lives on it as state, not in a return."""

    def __init__(self):
        super().__init__()
        self.tokens_per_expert = None

    def get_dispatched_routing(self):
        return (None, None, self.tokens_per_expert)


class _MinimalGate(nn.Layer):
    """Just enough for `_find_moe_layers` to recognise the module as MoE."""

    def __init__(self, num_experts=4):
        super().__init__()
        self.num_experts = num_experts


class _FakeMoELayerWithDispatch(nn.Layer):
    """MoE layer exposing only what the per-card token path reads."""

    def __init__(self, moe_group=None):
        super().__init__()
        self.gate = _MinimalGate()
        self.token_dispatcher = _FakeDispatcher()
        self.moe_group = moe_group
        self.dispatch_calls = 0

    def dispatch(self, tokens_per_expert):
        # Stands in for the real a2a: the dispatcher's state is only valid once
        # this has returned, which is why the monitor patches it.
        self.token_dispatcher.tokens_per_expert = tokens_per_expert
        self.dispatch_calls += 1
        return tokens_per_expert


def _dispatch_model(layers):
    wrapped = []
    for moe_layer in layers:
        layer = nn.Layer()
        layer.mlp = moe_layer
        layer.layer_idx = len(wrapped)
        wrapped.append(layer)
    return SimpleNamespace(layers=wrapped)


class PaddleMoECardTokensTest(unittest.TestCase):
    """Dispatched tokens per card — EP-group peak / mean / peak-mean ratio."""

    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    @staticmethod
    def _counts(*values):
        return paddle.to_tensor(list(values), dtype="int64")

    def test_the_three_card_metrics_are_emitted(self):
        moe_layer = _FakeMoELayerWithDispatch()
        monitor = PaddleMoEMonitor()
        monitor.register_hooks(_dispatch_model([moe_layer]))
        moe_layer.dispatch(self._counts(10, 20, 30))
        monitor.step()

        latest = training_logs.get_latest(
            prefix="moe_health/layer_0/card_tokens"
        )
        self.assertAlmostEqual(
            latest["moe_health/layer_0/card_tokens_max"], 60.0, places=5
        )
        self.assertAlmostEqual(
            latest["moe_health/layer_0/card_tokens_mean"], 60.0, places=5
        )
        # Single rank: this card *is* the peak and the mean.
        self.assertAlmostEqual(
            latest["moe_health/layer_0/card_tokens_peak_mean_ratio"],
            1.0,
            places=5,
        )
        monitor.remove_hooks()

    def test_microbatches_are_averaged_not_summed(self):
        # Otherwise a gradient-accumulation change would move the curve without
        # any change in balance.
        moe_layer = _FakeMoELayerWithDispatch()
        monitor = PaddleMoEMonitor()
        monitor.register_hooks(_dispatch_model([moe_layer]))
        moe_layer.dispatch(self._counts(10, 10))
        moe_layer.dispatch(self._counts(30, 30))
        monitor.step()

        latest = training_logs.get_latest(
            prefix="moe_health/layer_0/card_tokens"
        )
        self.assertAlmostEqual(
            latest["moe_health/layer_0/card_tokens_mean"], 40.0, places=5
        )
        monitor.remove_hooks()

    def test_the_dispatch_patch_keeps_the_original_return_value(self):
        moe_layer = _FakeMoELayerWithDispatch()
        monitor = PaddleMoEMonitor()
        monitor.register_hooks(_dispatch_model([moe_layer]))
        counts = self._counts(1, 2)
        self.assertIs(moe_layer.dispatch(counts), counts)
        self.assertEqual(moe_layer.dispatch_calls, 1)
        monitor.remove_hooks()

    def test_remove_hooks_restores_dispatch(self):
        moe_layer = _FakeMoELayerWithDispatch()
        monitor = PaddleMoEMonitor()
        monitor.register_hooks(_dispatch_model([moe_layer]))
        monitor.remove_hooks()
        self.assertFalse(hasattr(moe_layer, "_im_original_dispatch"))

        moe_layer.dispatch(self._counts(5, 5))
        monitor.step()
        self.assertEqual(
            training_logs.get_latest(prefix="moe_health/layer_0/card_tokens"),
            {},
        )

    def test_a_layer_without_a_dispatcher_declares_no_card_metrics(self):
        plain = nn.Layer()
        plain.gate = _MinimalGate()
        monitor = PaddleMoEMonitor()
        monitor.register_hooks(_dispatch_model([plain]))
        monitor.step()
        self.assertEqual(
            training_logs.get_latest(prefix="moe_health/layer_0/card_tokens"),
            {},
        )

    def test_monitor_interval_gates_the_accumulation(self):
        moe_layer = _FakeMoELayerWithDispatch()
        monitor = PaddleMoEMonitor(monitor_interval=0)
        monitor.register_hooks(_dispatch_model([moe_layer]))
        moe_layer.dispatch(self._counts(10, 20))
        monitor.step()
        self.assertEqual(
            training_logs.get_latest(prefix="moe_health/layer_0/card_tokens"),
            {},
        )
        monitor.remove_hooks()

    def test_card_metric_naming_contract(self):
        # training_logs picks the cross-rank reduction from the key name; it must
        # agree with MAX_AGGREGATED or the EP peak gets averaged across ranks.
        self.assertIn("card_tokens_max", PaddleMoEMonitor.MAX_AGGREGATED)
        self.assertTrue(
            TrainingLogs._is_max_metric("moe_health/layer_0/card_tokens_max")
        )
        for name in ("card_tokens_mean", "card_tokens_peak_mean_ratio"):
            key = f"moe_health/layer_0/{name}"
            self.assertFalse(TrainingLogs._is_max_metric(key), key)
            self.assertFalse(TrainingLogs._is_min_metric(key), key)

    def test_the_card_family_can_be_switched_off(self):
        moe_layer = _FakeMoELayerWithDispatch()
        monitor = PaddleMoEMonitor(exclude_families="card")
        monitor.register_hooks(_dispatch_model([moe_layer]))
        moe_layer.dispatch(self._counts(10, 20))
        monitor.step()
        self.assertEqual(
            training_logs.get_latest(prefix="moe_health/layer_0/card_tokens"),
            {},
        )
        monitor.remove_hooks()

    def test_the_ratio_is_the_mean_of_per_microbatch_ratios(self):
        """Reduction order, pinned without needing a process group.

        `max` is convex, so `mean_mb(max_card)` and `max_card(mean_mb)` differ
        whenever the hottest card changes between microbatches — and only the
        first matches PaddleFleet's `log_moe_balance`. Two microbatches with the
        peak on different cards: peaks 80 and 120, means 40 and 40.
        """
        peak_per_mb = paddle.to_tensor([[80.0, 120.0]], dtype="float32")
        mean_per_mb = paddle.to_tensor([[40.0, 40.0]], dtype="float32")
        peak, mean, ratio = PaddleMoEMonitor._card_token_stats(
            peak_per_mb, mean_per_mb
        )
        self.assertAlmostEqual(float(peak[0]), 100.0, places=5)
        self.assertAlmostEqual(float(mean[0]), 40.0, places=5)
        # mean(80/40, 120/40) = mean(2, 3) = 2.5
        self.assertAlmostEqual(float(ratio[0]), 2.5, places=5)

    def test_the_ratio_is_not_the_ratio_of_the_reported_means(self):
        """The two coincide only when the per-microbatch mean is constant.

        Documented in the README because someone will otherwise divide the two
        reported curves and expect the third.
        """
        peak_per_mb = paddle.to_tensor([[80.0, 120.0]], dtype="float32")
        mean_per_mb = paddle.to_tensor([[40.0, 80.0]], dtype="float32")
        peak, mean, ratio = PaddleMoEMonitor._card_token_stats(
            peak_per_mb, mean_per_mb
        )
        self.assertAlmostEqual(
            float(ratio[0]), 1.75, places=5
        )  # mean(2.0, 1.5)
        self.assertNotAlmostEqual(
            float(peak[0]) / float(mean[0]), float(ratio[0]), places=3
        )
