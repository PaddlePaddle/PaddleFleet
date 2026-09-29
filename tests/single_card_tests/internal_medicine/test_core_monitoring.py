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
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

Probe = importlib.import_module(
    "paddlefleet.internal_medicine.core.base_monitor"
).Probe
AVAILABLE_MONITORS = importlib.import_module(
    "paddlefleet.internal_medicine.core.registry"
).AVAILABLE_MONITORS
training_logs = importlib.import_module(
    "paddlefleet.internal_medicine.core.training_logs"
).training_logs


class DummyProbe(Probe):
    METRIC_PREFIX = "dummy"
    MAX_AGGREGATED = {"peak"}
    MIN_AGGREGATED = {"floor"}

    def register_hooks(self, model) -> None:
        return None


class CoreMonitoringTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_record_metrics_logs_per_layer_and_global_once_per_observation(
        self,
    ):
        probe = DummyProbe(log_per_layer=True, log_global=True)

        probe._record_metrics(0, {"mean": 2.0, "peak": 5.0, "floor": 3.0})
        probe._record_metrics(1, {"mean": 4.0, "peak": 2.0, "floor": 1.0})

        self.assertEqual(probe._global_count, 2)
        probe.step()

        latest = training_logs.get_latest(prefix="dummy")
        self.assertEqual(latest["dummy/layer_0/mean"], 2.0)
        self.assertEqual(latest["dummy/layer_1/mean"], 4.0)
        self.assertEqual(latest["dummy/global_mean"], 3.0)
        self.assertEqual(latest["dummy/global_peak"], 5.0)
        self.assertEqual(latest["dummy/global_floor"], 1.0)
        self.assertEqual(probe._global_count, 0)
        self.assertEqual(probe._global_accum, {})
        self.assertEqual(probe._global_metric_counts, {})

    def test_sparse_global_metrics_use_per_metric_counts(self):
        probe = DummyProbe(log_per_layer=False, log_global=True)

        probe._accumulate_global({"common": 2.0, "sparse": 4.0})
        probe._count_global_observation({"common", "sparse"})
        probe._accumulate_global({"common": 6.0})
        probe._count_global_observation({"common"})
        probe.step()

        latest = training_logs.get_latest(prefix="dummy")
        self.assertEqual(latest["dummy/global_common"], 4.0)
        self.assertEqual(latest["dummy/global_sparse"], 4.0)

    def test_gather_is_collective_even_when_this_rank_has_no_metrics(self):
        """A PP stage with no monitored layer must still enter the gather.

        ``_gather_fn`` wraps ``all_gather_object``; skipping it on an empty rank
        hangs every other rank in the collective (observed as a PP=4 deadlock).
        """
        calls = []

        def fake_gather(local_metrics):
            calls.append(dict(local_metrics))
            # Pretend another rank reported one layer.
            return [local_metrics, {"dummy/layer_0/mean": 4.0}]

        training_logs.set_gather_fn(fake_gather)
        try:
            self.assertEqual(training_logs.get_latest(), {})
            aggregated = training_logs.gather_and_aggregate()
        finally:
            training_logs.set_gather_fn(None)

        self.assertEqual(
            calls, [{}], "gather_fn must be called even with no local metrics"
        )
        self.assertEqual(aggregated, {"dummy/layer_0/mean": 4.0})

    def test_reduce_path_is_preferred_and_matches_gather_semantics(self):
        """The numeric reducer must agree with the object-gather it replaces."""
        training_logs.update(
            **{"dummy/a_mean": 2.0, "dummy/b_max": 5.0, "dummy/c_min": 1.0}
        )
        captured = {}

        def fake_reduce(schema):
            captured["schema"] = schema
            # Emulate a second rank: mean averages, max/min extend.
            return {"dummy/a_mean": 3.0, "dummy/b_max": 9.0, "dummy/c_min": 0.5}

        def fake_gather(_local):
            raise AssertionError(
                "gather_fn must not run when a reducer is installed"
            )

        training_logs.set_gather_fn(fake_gather)
        training_logs.set_reduce_fn(fake_reduce)
        try:
            aggregated = training_logs.gather_and_aggregate()
        finally:
            training_logs.set_reduce_fn(None)
            training_logs.set_gather_fn(None)

        self.assertEqual(aggregated["dummy/a_mean"], 3.0)
        self.assertEqual(aggregated["dummy/b_max"], 9.0)
        self.assertEqual(aggregated["dummy/c_min"], 0.5)

        schema = captured["schema"]
        self.assertEqual(schema.max_keys, ("dummy/b_max",))
        self.assertEqual(schema.min_keys, ("dummy/c_min",))
        self.assertEqual(schema.mean_keys, ("dummy/a_mean",))
        training_logs.reset()

    def test_reduce_falls_back_to_gather_when_reducer_declines(self):
        """A single-rank job (or an unsupported backend) keeps the old path."""
        training_logs.update(**{"dummy/a_mean": 2.0})
        training_logs.set_reduce_fn(lambda schema: None)
        training_logs.set_gather_fn(
            lambda local: [local, {"dummy/a_mean": 4.0}]
        )
        try:
            aggregated = training_logs.gather_and_aggregate()
        finally:
            training_logs.set_reduce_fn(None)
            training_logs.set_gather_fn(None)

        self.assertEqual(aggregated, {"dummy/a_mean": 3.0})
        training_logs.reset()

    def test_reduce_fingerprint_is_stable_across_processes(self):
        """PYTHONHASHSEED-salted hash() would re-align the layout every step."""
        schema = training_logs.build_reduce_schema(
            {"dummy/a_mean": 1.0, "dummy/b_max": 2.0}
        )
        # Recomputed from the same key layout, as another rank would.
        twin = training_logs.build_reduce_schema(
            {"dummy/b_max": 9.0, "dummy/a_mean": 9.0}
        )
        self.assertEqual(schema.fingerprint, twin.fingerprint)
        different = training_logs.build_reduce_schema({"dummy/a_mean": 1.0})
        self.assertNotEqual(schema.fingerprint, different.fingerprint)
        training_logs.reset()

    def test_log_flags_are_respected(self):
        probe = DummyProbe(log_per_layer=False, log_global=True)
        probe._record_metrics(0, {"mean": 2.0})
        probe.step()

        latest = training_logs.get_latest(prefix="dummy")
        self.assertNotIn("dummy/layer_0/mean", latest)
        self.assertEqual(latest["dummy/global_mean"], 2.0)

        training_logs.reset()
        probe = DummyProbe(log_per_layer=True, log_global=False)
        probe._record_metrics(0, {"mean": 7.0})
        probe.step()

        latest = training_logs.get_latest(prefix="dummy")
        self.assertEqual(latest["dummy/layer_0/mean"], 7.0)
        self.assertNotIn("dummy/global_mean", latest)
        self.assertEqual(probe._global_count, 0)

    def test_empty_metrics_do_not_count_or_log(self):
        probe = DummyProbe(log_per_layer=True, log_global=True)
        probe._record_metrics(0, {})
        probe.step()

        self.assertEqual(probe._global_count, 0)
        self.assertEqual(training_logs.get_latest(prefix="dummy"), {})

    def test_massive_activation_scale_keys_are_max_aggregated(self):
        for key in (
            "massive_act/layer_0/channel_max_ratio",
            "massive_act/layer_0/channel_median",
            "massive_act/layer_0/channel_p95",
            "massive_act/layer_0/channel_p99",
            "massive_act/layer_0/massive_act_channel_count",
            "massive_act/layer_0/channel_count_gt_100",
            "massive_act/layer_0/activation_rms",
            "massive_act/layer_0/spectral_norm_max",
            "massive_act/global_spectral_norm_max",
            "massive_act/layer_0/lipschitz_max",
            "massive_act/global_lipschitz_max",
            # attn_type-tagged keys (paddlefleet prepends mla_/hca_/csa_/...)
            "massive_act/layer_0/hca_massive_act_channel_count",
            "massive_act/layer_0/hca_channel_count_gt_10",
            "massive_act/global_hca_channel_count_gt_10",
        ):
            self.assertTrue(training_logs._is_max_metric(key), key)

    def test_spectral_norm_min_keys_are_min_aggregated(self):
        for key in (
            "massive_act/layer_0/spectral_norm_min",
            "massive_act/global_spectral_norm_min",
            "massive_act/layer_0/lipschitz_min",
            "massive_act/global_lipschitz_min",
        ):
            self.assertTrue(training_logs._is_min_metric(key), key)
            self.assertFalse(training_logs._is_max_metric(key), key)

    def test_resolve_layer_idx_prefers_explicit_attrs_then_layer_number_then_offset(
        self,
    ):
        probe = DummyProbe()

        self.assertEqual(
            probe._resolve_layer_idx(SimpleNamespace(layer_idx=9), 0, 4), 9
        )
        self.assertEqual(
            probe._resolve_layer_idx(SimpleNamespace(layer_number=3), 0, 4), 2
        )
        self.assertEqual(probe._resolve_layer_idx(SimpleNamespace(), 2, 4), 2)

        probe.pp_rank = 1
        self.assertEqual(probe._resolve_layer_idx(SimpleNamespace(), 2, 4), 6)
        self.assertEqual(
            probe._resolve_layer_idx(SimpleNamespace(), 2, 4, layer_offset=8),
            14,
        )

    def test_paddlefleet_registry_lists_massive_activation_monitor(self):
        self.assertIn("massive_act", AVAILABLE_MONITORS["paddlefleet"])

    def test_sampled_this_step_marks_exactly_the_steps_the_hooks_recorded(self):
        """The flag is what a trainer callback gates its cross-rank gather on.

        It has to agree with `_should_monitor()` as seen *inside* the hooks —
        i.e. the pre-increment `step_count` — or the callback either drops
        metrics or pays for an empty collective.
        """
        for interval in (1, 2, 3, 5, 200):
            probe = DummyProbe(monitor_interval=interval)
            recorded, flagged = [], []
            for global_step in range(1, 4 * interval + 2):
                if (
                    probe._should_monitor()
                ):  # forward: hooks see step_count pre-increment
                    recorded.append(global_step)
                probe.step()  # on_step_end
                if probe.sampled_this_step:  # on_log
                    flagged.append(global_step)
            self.assertEqual(recorded, flagged, f"interval={interval}")
            self.assertEqual(recorded[0], 1, f"interval={interval}")

    def test_sampled_this_step_survives_a_resume_at_an_unaligned_step(self):
        """`step_count` restarts at 0 on resume while `global_step` does not.

        Deriving the phase from `global_step % interval` drifts (and silently
        drops every sampled step); the flag cannot, because it only ever reads
        `step_count`.
        """
        interval = 200
        for resume_at in (0, 300, 5050, 999):
            probe = DummyProbe(monitor_interval=interval)
            recorded, flagged, by_global_step = [], [], []
            for global_step in range(
                resume_at + 1, resume_at + 1 + 3 * interval
            ):
                if probe._should_monitor():
                    recorded.append(global_step)
                probe.step()
                if probe.sampled_this_step:
                    flagged.append(global_step)
                if global_step % interval == 1:
                    by_global_step.append(global_step)
            self.assertEqual(recorded, flagged, f"resume_at={resume_at}")
            if resume_at % interval:
                self.assertNotEqual(
                    recorded, by_global_step, f"resume_at={resume_at}"
                )

    def test_sampled_this_step_is_false_when_monitoring_is_disabled(self):
        probe = DummyProbe(monitor_interval=0)
        probe.step()
        self.assertFalse(probe.sampled_this_step)

    def test_skip_next_step_suppresses_hooks_and_sampled_flag_once(self):
        probe = DummyProbe(monitor_interval=1)
        probe._flush_buffers = Mock()
        probe.skip_next_steps()

        self.assertFalse(probe._should_monitor())
        probe.step()
        self.assertFalse(probe.sampled_this_step)
        self.assertEqual(probe.step_count, 1)
        probe._flush_buffers.assert_not_called()

        self.assertTrue(probe._should_monitor())
        probe.step()
        self.assertTrue(probe.sampled_this_step)
        probe._flush_buffers.assert_called_once_with()

    def test_skip_next_step_preserves_process_local_interval_phase(self):
        probe = DummyProbe(monitor_interval=5)
        probe.skip_next_steps()
        sampled_steps = []

        for resumed_step in range(1, 8):
            if probe._should_monitor():
                sampled_steps.append(resumed_step)
            probe.step()

        self.assertEqual(sampled_steps, [6])

    def test_skip_next_steps_consumes_exactly_requested_count(self):
        probe = DummyProbe(monitor_interval=1)
        probe.skip_next_steps(2)
        sampled = []

        for step in range(1, 5):
            if probe._should_monitor():
                sampled.append(step)
            probe.step()

        self.assertEqual(sampled, [3, 4])

    def test_reducer_keeps_aligned_layout_for_heterogeneous_rank_keys(self):
        """PP/VPP ranks may own different keys; layout must still align once.

        The stale flag is local-fingerprint based. Comparing fingerprints
        across ranks would re-align every step because PP stages naturally
        report disjoint layer keys.
        """
        from paddlefleet.internal_medicine.backends.paddlefleet.gather import (
            _PaddleReducer,
        )

        class FakeTensor:
            def __init__(self, array):
                import numpy as np

                self._array = np.array(array, copy=True)

            def __setitem__(self, key, value):
                self._array[key] = value

            @property
            def shape(self):
                return self._array.shape

            def numpy(self):
                return self._array

        class FakePaddle:
            def to_tensor(self, data, dtype="int64"):
                import numpy as np

                return FakeTensor(np.array(data))

        class FakeDist:
            ReduceOp = SimpleNamespace(MAX="max")

            def __init__(self):
                self.sent = []
                self.payloads = []

            def all_reduce(self, tensor, op=None):
                # Record the bit the SUT wrote, then MAX-reduce those bits.
                # Sequential two-rank simulation: later ranks see the running
                # max, so call a stale rank first when mixed flags are needed.
                self.sent.append(int(tensor.numpy()[0]))
                tensor[:] = max(self.sent)

            def all_gather_object(self, gathered, payload):
                gathered.extend(self.payloads)

        schema_a = training_logs.build_reduce_schema(
            {"dummy/layer_0/mean": 2.0, "dummy/layer_0/peak": 5.0}
        )
        schema_b = training_logs.build_reduce_schema(
            {"dummy/layer_1/mean": 4.0, "dummy/layer_1/floor": 1.0}
        )
        self.assertNotEqual(schema_a.fingerprint, schema_b.fingerprint)

        fake_dist = FakeDist()
        fake_paddle = FakePaddle()
        fake_dist.payloads = [
            {
                "mean": schema_a.mean_keys,
                "max": schema_a.max_keys,
                "min": schema_a.min_keys,
            },
            {
                "mean": schema_b.mean_keys,
                "max": schema_b.max_keys,
                "min": schema_b.min_keys,
            },
        ]

        reducer_a = _PaddleReducer()
        reducer_b = _PaddleReducer()
        align_calls = {"n": 0}

        def wrap_align(reducer):
            original = reducer._align

            def counting_align(schema, dist):
                align_calls["n"] += 1
                original(schema, dist)

            reducer._align = counting_align
            return original

        original_a = wrap_align(reducer_a)
        original_b = wrap_align(reducer_b)

        # Empty cache → each rank must send stale=1. Hardcoding stale to 0
        # would make sent == [0, 0] and fail this assertion.
        self.assertFalse(
            reducer_a._agree_on_layout(schema_a, fake_paddle, fake_dist)
        )
        self.assertFalse(
            reducer_b._agree_on_layout(schema_b, fake_paddle, fake_dist)
        )
        self.assertEqual(fake_dist.sent, [1, 1])
        original_a(schema_a, fake_dist)
        original_b(schema_b, fake_dist)
        self.assertEqual(align_calls["n"], 0)
        self.assertEqual(set(reducer_a._layout), set(reducer_b._layout))
        self.assertEqual(
            set(reducer_a._layout),
            {
                "dummy/layer_0/mean",
                "dummy/layer_0/peak",
                "dummy/layer_1/mean",
                "dummy/layer_1/floor",
            },
        )

        # Local fingerprints still match each rank's cache, even though the
        # two ranks own disjoint keys. Hardcoding stale to 1 fails here.
        fake_dist.sent = []
        self.assertTrue(
            reducer_a._agree_on_layout(schema_a, fake_paddle, fake_dist)
        )
        self.assertTrue(
            reducer_b._agree_on_layout(schema_b, fake_paddle, fake_dist)
        )
        self.assertEqual(fake_dist.sent, [0, 0])
        self.assertEqual(align_calls["n"], 0)

        # One rank's schema changes: that rank sends 1, the other sends 0,
        # MAX is 1 so both must re-align. Call the stale rank first so the
        # sequential running-max mock matches a real collective.
        schema_b2 = training_logs.build_reduce_schema(
            {
                "dummy/layer_1/mean": 4.0,
                "dummy/layer_1/floor": 1.0,
                "dummy/layer_2/mean": 3.0,
            }
        )
        self.assertNotEqual(schema_b.fingerprint, schema_b2.fingerprint)
        fake_dist.sent = []
        self.assertFalse(
            reducer_b._agree_on_layout(schema_b2, fake_paddle, fake_dist)
        )
        self.assertFalse(
            reducer_a._agree_on_layout(schema_a, fake_paddle, fake_dist)
        )
        self.assertEqual(fake_dist.sent, [1, 0])


if __name__ == "__main__":
    unittest.main()
