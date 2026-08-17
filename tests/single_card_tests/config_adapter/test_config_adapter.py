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

"""Unit tests for paddlefleet.config_adapter (pure Python, no device)."""

import contextlib
import io
import json
import runpy
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from paddlefleet.config_adapter import (
    AdaptOptions,
    ConfigAdapter,
    inspect_config,
    main,
    parse_overrides,
    plan_precision_switches,
)
from paddlefleet.config_adapter.constraints import (
    align_layers,
    check_ep_shrink,
    check_hardware,
    check_pp_shrink,
    ep_candidates,
    min_shrink_cards,
    pp_candidates,
)
from paddlefleet.config_adapter.field_spec import (
    FIELD_SPECS,
    describe_missing,
    resolve_fields,
)
from paddlefleet.config_adapter.io_writers import JsonWriter, YamlWriter
from paddlefleet.config_adapter.layer_fields import (
    effective_mtp_layers,
    plan_layer_field_shrink,
)
from paddlefleet.config_adapter.model_config_resolver import (
    ModelConfigResolveError,
    build_adapted_dir,
    resolve_model_config,
    rewrite_model_name_or_path,
)
from paddlefleet.config_adapter.report import (
    ChangeLog,
    format_header,
    format_report,
)
from paddlefleet.config_adapter.strategies import (
    scale_accumulation,
    scale_batch,
)
from paddlefleet.config_adapter.topology import TopologyValidator
from paddlefleet.config_adapter.utils import (
    extract_parallel_params,
    multi_lcm,
    parse_value,
)

SOURCE_YAML = """\
model_name_or_path: ./model_dir
num_empty_layers_add_in_head: 0
num_empty_layers_add_in_tail: 0
separate_mtp_headloss: false
fa_version: 3
global_batch_size: 1536
per_device_train_batch_size: 1
gradient_accumulation_steps: 2
sharding_parallel_size: 96
data_parallel_size: 1
tensor_model_parallel_size: 1
expert_model_parallel_size: 64
pipeline_model_parallel_size: 8
virtual_pipeline_model_parallel_size: 2
context_parallel_size: 1
csa_indexer_backend: tilelang
csa_sparse_attn_backend: cudnn
max_steps: 1000000
"""

MODEL_CONFIG = {
    "num_hidden_layers": 64,
    "n_routed_experts": 256,
    "num_experts_per_tok": 8,
    "first_k_dense_replace": 1,
    "num_nextn_predict_layers": 1,
    "multimax_modules": ["lm_head"],
}


class ConfigAdapterTestBase(unittest.TestCase):
    """Builds a throw-away source YAML + model_config.json per test."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.yaml_path = self.root / "source.yaml"
        self.yaml_path.write_text(SOURCE_YAML, encoding="utf-8")
        self.model_dir = self.root / "model_dir"
        self.model_dir.mkdir()
        self.json_path = self.model_dir / "model_config.json"
        self.write_json(MODEL_CONFIG)
        self.output_dir = self.root / "out"

    def tearDown(self):
        self._tmp.cleanup()

    def write_yaml(self, text):
        """Overwrite the source YAML with ``text``."""
        self.yaml_path.write_text(text, encoding="utf-8")

    def write_json(self, data):
        """Overwrite the source model_config.json with ``data``."""
        self.json_path.write_text(
            json.dumps(data, indent=2) + "\n", encoding="utf-8"
        )

    def adapt(
        self,
        target_nodes,
        test_performance=False,
        test_accuracy=False,
        **kwargs,
    ):
        """Run one adaptation, returning ``(ok, message)``."""
        kwargs.setdefault("output_dir", self.output_dir)
        adapter = ConfigAdapter(
            options=AdaptOptions(test_performance, test_accuracy),
            target_nodes=target_nodes,
            **kwargs,
        )
        return adapter.adapt(self.yaml_path)

    def load_output_yaml(self, target_cards):
        """Load the generated YAML for a target card count."""
        path = self.output_dir / f"source_adapted_{target_cards}cards.yaml"
        return YamlWriter().load(path)

    def load_output_json(self, target_cards):
        """Load the generated model_config.json for a target card count."""
        path = (
            self.output_dir
            / "model_config_separated"
            / f"model_dir_adapted_{target_cards}cards"
            / "model_config.json"
        )
        return JsonWriter().load(path)


class TestOptions(unittest.TestCase):
    """The two switches are orthogonal and both optional."""

    def test_default_shrinks_and_scales_batch(self):
        options = AdaptOptions()
        self.assertFalse(options.freeze_parallel)
        self.assertFalse(options.inject_precision)
        self.assertEqual(options.batch_strategy, "scale_batch")
        self.assertEqual(options.label, "default")

    def test_accuracy_keeps_effective_batch(self):
        options = AdaptOptions(test_accuracy=True)
        self.assertFalse(options.freeze_parallel)
        self.assertTrue(options.inject_precision)
        self.assertEqual(options.batch_strategy, "scale_accumulation")

    def test_performance_freezes_acc_even_with_accuracy(self):
        options = AdaptOptions(test_performance=True, test_accuracy=True)
        self.assertTrue(options.freeze_parallel)
        self.assertTrue(options.inject_precision)
        self.assertEqual(options.batch_strategy, "scale_batch")
        self.assertEqual(options.label, "performance+accuracy")


class TestCandidates(unittest.TestCase):
    """A degree greater than 1 must never shrink to 1."""

    def test_ep_candidates_never_reach_one(self):
        self.assertNotIn(1, ep_candidates(64, tp=1))
        self.assertEqual(min(ep_candidates(64, tp=1)), 2)
        self.assertEqual(ep_candidates(2, tp=1), [])

    def test_pp_candidates_never_reach_one(self):
        self.assertEqual(pp_candidates(8), [4, 2])
        self.assertEqual(pp_candidates(2), [])

    def test_ep_candidates_respect_tp(self):
        # C3 requires ep_new % tp == 0.
        self.assertTrue(all(c % 4 == 0 for c in ep_candidates(64, tp=4)))


class TestPrecisionSwitches(unittest.TestCase):
    """Switch routing: pin whichever document declares the key."""

    def test_yaml_key_pinned_in_yaml(self):
        applied, skipped = plan_precision_switches(
            {"csa_sparse_attn_backend": "cudnn"}, {}
        )
        self.assertIn(
            ("yaml", "csa_sparse_attn_backend", "tilelang"),
            [(t, k, v) for t, k, v, _r in applied],
        )
        self.assertEqual(skipped, [])

    def test_key_declared_in_json_is_pinned_there_too(self):
        applied, _skipped = plan_precision_switches(
            {}, {"csa_sparse_attn_backend": "cudnn"}
        )
        targets = {(t, k) for t, k, _v, _r in applied}
        self.assertIn(("json", "csa_sparse_attn_backend"), targets)

    def test_json_switch_skipped_without_model_config(self):
        _applied, skipped = plan_precision_switches({}, None)
        self.assertTrue(any("multimax_modules" in note for note in skipped))


class TestOverrideParsing(unittest.TestCase):
    """``--set`` may name its target file, or let the tool decide."""

    def test_explicit_prefix_wins(self):
        yaml_map, json_map, auto_map = parse_overrides(
            ["yaml:max_steps=10", "json:n_routed_experts=32"]
        )
        self.assertEqual(yaml_map, {"max_steps": 10})
        self.assertEqual(json_map, {"n_routed_experts": 32})
        self.assertEqual(auto_map, {})

    def test_prefix_less_goes_to_auto(self):
        yaml_map, json_map, auto_map = parse_overrides(["max_steps=10"])
        self.assertEqual((yaml_map, json_map), ({}, {}))
        self.assertEqual(auto_map, {"max_steps": 10})

    def test_missing_value_is_an_error(self):
        with self.assertRaises(ValueError):
            parse_overrides(["max_steps"])


class TestValueParsing(unittest.TestCase):
    """``--set`` value inference and parallel-dim extraction."""

    def test_parse_value_types(self):
        self.assertIs(parse_value("true"), True)
        self.assertIs(parse_value("False"), False)
        self.assertIsNone(parse_value("null"))
        self.assertIsNone(parse_value("None"))
        self.assertEqual(parse_value("42"), 42)
        self.assertEqual(parse_value("1e-5"), 1e-05)
        self.assertEqual(parse_value("tilelang"), "tilelang")

    def test_extract_parallel_params_defaults(self):
        self.assertEqual(extract_parallel_params({}), (1, 1, 1, 1, 1))
        self.assertEqual(
            extract_parallel_params(
                {
                    "tensor_model_parallel_size": 2,
                    "pipeline_model_parallel_size": None,
                    "expert_model_parallel_size": 8,
                    "context_parallel_size": -1,
                    "sep_parallel_size": 4,
                }
            ),
            (2, 1, 8, 1, 4),
        )

    def test_multi_lcm(self):
        self.assertEqual(multi_lcm(4, 6, 8), 24)


class TestBatchStrategies(unittest.TestCase):
    """Batch scaling maths, including every refusal path."""

    def test_scale_batch_happy_path(self):
        config_map, reason, err = scale_batch(1536, 2, 768, 8)
        self.assertIsNone(err)
        self.assertEqual(config_map["global_batch_size"], 16)
        self.assertEqual(config_map["gradient_accumulation_steps"], 2)
        self.assertIn("等比缩放", reason)

    def test_scale_batch_without_gbs_keeps_acc(self):
        config_map, reason, err = scale_batch(None, 4, 768, 8)
        self.assertIsNone(err)
        self.assertEqual(config_map, {"gradient_accumulation_steps": 4})
        self.assertIn("global_batch_size", reason)

    def test_scale_batch_refuses_non_divisible(self):
        _map, _reason, err = scale_batch(1537, 2, 768, 8)
        self.assertIn("无法整除", err)

    def test_scale_batch_refuses_zero(self):
        # A source config declaring global_batch_size: 0 scales to 0.
        _map, _reason, err = scale_batch(0, 2, 768, 8)
        self.assertIn("<= 0", err)

    def test_scale_accumulation_happy_path(self):
        config_map, reason, err = scale_accumulation(1536, 2, 768, 8)
        self.assertIsNone(err)
        self.assertEqual(config_map["global_batch_size"], 1536)
        self.assertEqual(config_map["gradient_accumulation_steps"], 192)
        self.assertIn("等效 batch", reason)

    def test_scale_accumulation_without_gbs_keeps_acc(self):
        config_map, _reason, err = scale_accumulation(None, 4, 768, 8)
        self.assertIsNone(err)
        self.assertEqual(config_map, {"gradient_accumulation_steps": 4})

    def test_scale_accumulation_refuses_non_divisible(self):
        _map, _reason, err = scale_accumulation(1536, 3, 768, 7)
        self.assertIn("无法整除", err)


class TestTopologyConstraints(unittest.TestCase):
    """C1..C4 and the suggestion helper."""

    def test_valid_topology_reports_derived_groups(self):
        ok, _msg, details = TopologyValidator(8, 8).validate(1, 2, 4, 1, 1)
        self.assertTrue(ok)
        self.assertEqual(details["sharding"], 4)
        self.assertEqual(details["moe_sharding"], 1)

    def test_c1_violation(self):
        ok, msg, _d = TopologyValidator(8, 8).validate(3, 1, 1, 1, 1)
        self.assertFalse(ok)
        self.assertIn("C1", msg)

    def test_c2_violation(self):
        ok, msg, _d = TopologyValidator(8, 8).validate(1, 8, 64, 1, 1)
        self.assertFalse(ok)
        self.assertIn("C2", msg)

    def test_c3_violation(self):
        ok, msg, _d = TopologyValidator(8, 8).validate(4, 1, 2, 1, 1)
        self.assertFalse(ok)
        self.assertIn("C3", msg)

    def test_c4_violation(self):
        ok, msg, _d = TopologyValidator(8, 8).validate(1, 1, 1, 3, 1)
        self.assertFalse(ok)
        self.assertIn("C4", msg)

    def test_suggestions_are_multiples_of_the_minimum_unit(self):
        cards = TopologyValidator(8, 8).suggest_valid_cards(1, 2, 8, 1, 1)
        self.assertTrue(all(c % 16 == 0 for c in cards), cards)


class TestHardwareAndModelConstraints(unittest.TestCase):
    """E-family and M-family checks."""

    def test_e3_rejects_non_integer_dims(self):
        ok, why = check_hardware(8, 8, 1, 1, "2", 1, 1)
        self.assertFalse(ok)
        self.assertIn("E3", why)

    def test_e1_rejects_asymmetric_multi_node(self):
        ok, why = check_hardware(12, 8, 1, 1, 1, 1, 1)
        self.assertFalse(ok)
        self.assertIn("E1", why)

    def test_e2_rejects_tp_larger_than_the_node(self):
        ok, why = check_hardware(8, 8, 16, 1, 1, 1, 1)
        self.assertFalse(ok)
        self.assertIn("E2", why)

    def test_hardware_prereqs_pass(self):
        ok, why = check_hardware(8, 8, 1, 8, 64, 1, 1)
        self.assertTrue(ok, why)

    def test_min_shrink_cards_respects_the_floor(self):
        # PP and EP may only shrink to 2, so 2*2 = 4 cards is the floor here.
        self.assertEqual(min_shrink_cards(1, 8, 8, 1, 1, 8), 8)

    def test_m1_rejects_uneven_experts(self):
        ok, why, experts = check_ep_shrink(64, 56, 32, 8)
        self.assertFalse(ok)
        self.assertIn("M1", why)
        self.assertEqual(experts, 28)

    def test_m2_rejects_fewer_experts_than_topk(self):
        ok, why, _experts = check_ep_shrink(64, 2, 64, 8)
        self.assertFalse(ok)
        self.assertIn("M2", why)

    def test_ep_growth_is_vacuous(self):
        ok, _why, experts = check_ep_shrink(8, 8, 64, 8)
        self.assertTrue(ok)
        self.assertEqual(experts, 64)

    def test_pp_shrink_reports_layer_and_alignment(self):
        ok, why, meta = check_pp_shrink(8, 2, 64, 0, 0, 2)
        self.assertTrue(ok, why)
        self.assertEqual(meta["layers_new"], 16)
        self.assertEqual(meta["vpp_new"], 2)

    def test_pp_shrink_warns_on_few_layers(self):
        _ok, _why, meta = check_pp_shrink(8, 2, 8, 0, 0, 1)
        self.assertIn("推荐", meta["warning"])

    def test_pp_shrink_rejects_zero_layers(self):
        ok, why, _meta = check_pp_shrink(8, 2, 3, 0, 0, 1)
        self.assertFalse(ok)
        self.assertIn("至少需要 1 层", why)

    def test_m5_rejects_a_dense_prefix_that_does_not_fit(self):
        ok, why, _meta = check_pp_shrink(
            8, 2, 64, 0, 0, 1, first_k_dense_replace=32
        )
        self.assertFalse(ok)
        self.assertIn("M5", why)

    def test_align_layers_pads_the_tail(self):
        vpp_new, tail_new = align_layers(15, 0, 0, 2, 2)
        self.assertEqual(vpp_new, 2)
        self.assertEqual(tail_new, 1)

    def test_align_layers_refuses_pp_below_the_floor(self):
        self.assertEqual(align_layers(16, 0, 0, 1, 2), (None, None))


class TestFieldSpec(unittest.TestCase):
    """Alias resolution for model_config.json fields."""

    def test_resolves_aliases_and_remembers_the_key(self):
        resolved, missing = resolve_fields(
            {
                "num_hidden_layers": 8,
                "num_local_experts": 16,
                "moe_k": 4,
            }
        )
        self.assertEqual(missing, {})
        self.assertEqual(resolved["num_experts"].value, 16)
        self.assertEqual(
            resolved["num_experts"].writeback_key, "num_local_experts"
        )
        self.assertEqual(resolved["num_experts_per_tok"].value, 4)
        # Optional fields fall back to their declared default.
        self.assertEqual(resolved["first_k_dense_replace"].value, 0)
        self.assertEqual(resolved["first_k_dense_replace"].origin, "<default>")

    def test_missing_required_fields_are_reported(self):
        resolved, missing = resolve_fields({"num_hidden_layers": 8})
        self.assertIn("num_experts", missing)
        self.assertNotIn("num_experts", resolved)
        # Write-back falls back to the canonical spelling.
        message = describe_missing("num_experts", FIELD_SPECS["num_experts"])
        self.assertIn("n_routed_experts", message)
        self.assertIn("--set json:", message)


class TestModelConfigResolution(unittest.TestCase):
    """Locating and re-pointing model_config.json."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.model_dir = self.root / "model_dir"
        self.model_dir.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def test_empty_value_is_refused(self):
        with self.assertRaises(ModelConfigResolveError) as ctx:
            resolve_model_config(None, self.root)
        self.assertIn("model_name_or_path", str(ctx.exception))

    def test_missing_directory_lists_what_was_tried(self):
        with self.assertRaises(ModelConfigResolveError) as ctx:
            resolve_model_config("./nope", self.root)
        self.assertIn("已尝试", str(ctx.exception))

    def test_directory_without_json_is_refused(self):
        with self.assertRaises(ModelConfigResolveError) as ctx:
            resolve_model_config("./model_dir", self.root)
        self.assertIn("model_config.json", str(ctx.exception))

    def test_relative_and_absolute_values_both_resolve(self):
        (self.model_dir / "model_config.json").write_text("{}\n")
        for value in ("./model_dir", str(self.model_dir)):
            found_dir, json_path = resolve_model_config(value, self.root)
            self.assertEqual(found_dir, self.model_dir.resolve())
            self.assertTrue(json_path.is_file())

    def test_adapted_dir_layout(self):
        path = build_adapted_dir("/out", "model_dir", "8cards")
        self.assertEqual(
            path,
            Path("/out/model_config_separated/model_dir_adapted_8cards"),
        )

    def test_absolute_source_stays_absolute(self):
        value = rewrite_model_name_or_path(
            self.model_dir, self.root, source_was_absolute=True
        )
        self.assertTrue(Path(value).is_absolute())

    def test_relative_source_stays_relative_inside_the_yaml_dir(self):
        value = rewrite_model_name_or_path(
            self.model_dir, self.root, source_was_absolute=False
        )
        self.assertEqual(value, "model_dir")


class TestChangeReporting(unittest.TestCase):
    """Change bookkeeping and the rendered report."""

    def _info(self, **overrides):
        return {
            "input": "src.yaml",
            "output": "out.yaml",
            "profile": "accuracy",
            "profile_flag": "--test-accuracy",
            "batch_strategy": "scale_accumulation",
            "orig_cards_label": 768,
            "orig_nodes_label": 96,
            "orig_scale_label": "96 节点 / 768 卡",
            "target_cards": 8,
            "target_nodes": 1,
            "cards_per_node": 8,
            "dims_line": "TP 1->1",
            "sharding_line": "96 -> 4",
            "plan_note": "仅缩 EP",
            "model_config_output": None,
            "skipped_switches": [],
            "warnings": [],
            **overrides,
        }

    def test_no_op_writes_are_dropped(self):
        log = ChangeLog()
        log.record("yaml", [("modified", "a", 1, 1)], "no-op")
        log.record("yaml", [("modified", "b", 1, 2)], "real change")
        log.record("json", [("added", "c", None, 3)], "added")
        log.record_removed("yaml", "d", 9, "removed")
        self.assertEqual(len(log), 3)
        self.assertEqual([c.field for c in log.by_target("yaml")], ["b", "d"])

    def test_report_lists_every_kind_with_its_reason(self):
        log = ChangeLog()
        log.record("yaml", [("modified", "ep", 64, 4)], "EP 缩容")
        log.record("yaml", [("added", "new_key", None, 1)], "新增")
        log.record_removed("yaml", "fa_version", 3, "环境相关")
        log.record("json", [("modified", "experts", 256, 16)], "专家数")
        text = format_report(
            self._info(
                model_config_output="out.json",
                skipped_switches=["skipped one"],
                warnings=["watch out"],
            ),
            log,
        )
        self.assertIn("CHANGE field=ep old=64 new=4", text)
        self.assertIn("ADD field=new_key new=1", text)
        self.assertIn("DELETE field=fa_version old=3", text)
        self.assertIn("原因：EP 缩容", text)
        self.assertIn("model_config.json 改动", text)
        self.assertIn("跳过的精度开关：skipped one", text)
        self.assertIn("WARNING：watch out", text)
        self.assertIn("MODEL_CONFIG_OUTPUT=out.json", text)

    def test_report_says_so_when_nothing_changed(self):
        text = format_report(self._info(), ChangeLog())
        self.assertIn("YAML 改动：无", text)

    def test_header_is_a_comment_block(self):
        header = format_header(self._info())
        self.assertTrue(
            all(
                line.startswith("# [config_adapter]")
                for line in header.strip().splitlines()
            )
        )


class TestLayerFieldPlanning(unittest.TestCase):
    """Per-layer list rewrites, exercised without touching any file."""

    def _config(self, **overrides):
        return {
            "num_hidden_layers": 64,
            "num_nextn_predict_layers": 1,
            "csa_compress_ratios": [128, 128, 128, -2] * 16 + [-2],
            "window_attn_skip_freq": [0] * 65,
            "layer_types": ["full_attention"] * 64,
            **overrides,
        }

    def test_effective_mtp_follows_the_framework_rule(self):
        # mtp_num_layers wins whenever it is non-zero.
        self.assertEqual(
            effective_mtp_layers(
                {"mtp_num_layers": 2, "num_nextn_predict_layers": 0}
            ),
            2,
        )
        self.assertEqual(
            effective_mtp_layers({"num_nextn_predict_layers": 3}), 3
        )
        self.assertEqual(effective_mtp_layers({}), 0)
        self.assertEqual(effective_mtp_layers(None), 0)

    def test_truncates_layer_part_and_keeps_the_mtp_tail(self):
        changes, err = plan_layer_field_shrink(self._config(), 64, 16, 1)
        self.assertIsNone(err)
        by_key = {key: value for key, value, _reason in changes}
        self.assertEqual(len(by_key["csa_compress_ratios"]), 17)
        self.assertEqual(by_key["csa_compress_ratios"][-1], -2)
        self.assertEqual(len(by_key["window_attn_skip_freq"]), 17)
        self.assertEqual(len(by_key["layer_types"]), 16)

    def test_growing_or_equal_layer_counts_change_nothing(self):
        changes, err = plan_layer_field_shrink(self._config(), 64, 64, 1)
        self.assertIsNone(err)
        self.assertEqual(changes, [])

    def test_scalar_fields_are_ignored(self):
        config = self._config(window_attn_skip_freq=4)
        changes, err = plan_layer_field_shrink(config, 64, 16, 1)
        self.assertIsNone(err)
        self.assertNotIn(
            "window_attn_skip_freq", [key for key, _v, _r in changes]
        )

    def test_inconsistent_source_length_is_refused(self):
        config = self._config(csa_compress_ratios=[128] * 10)
        _changes, err = plan_layer_field_shrink(config, 64, 16, 1)
        self.assertIsNotNone(err)
        self.assertIn("不自洽", err)

    def test_losing_an_attention_family_is_refused(self):
        config = self._config(csa_compress_ratios=[128] * 60 + [-2] * 4 + [-2])
        _changes, err = plan_layer_field_shrink(config, 64, 16, 1)
        self.assertIsNotNone(err)
        self.assertIn("丢掉注意力类型", err)

    def test_missing_lists_need_no_rewrite(self):
        changes, err = plan_layer_field_shrink(
            {"num_hidden_layers": 64}, 64, 16, 0
        )
        self.assertIsNone(err)
        self.assertEqual(changes, [])


class TestDefaultAdaptation(ConfigAdapterTestBase):
    """No switch: just make the config fit, shrinking EP/PP if needed."""

    def test_shrinks_ep_and_pp_without_precision_switches(self):
        ok, message = self.adapt(target_nodes=1)
        self.assertTrue(ok, message)
        config = self.load_output_yaml(8)
        model_config = self.load_output_json(8)

        # EP and PP shrink together; neither collapses to 1.
        self.assertEqual(config["expert_model_parallel_size"], 4)
        self.assertEqual(config["pipeline_model_parallel_size"], 2)
        self.assertEqual(config["sharding_parallel_size"], 4)
        self.assertEqual(model_config["n_routed_experts"], 16)
        self.assertEqual(model_config["num_hidden_layers"], 16)
        # Default batch strategy shrinks GBS and leaves acc alone.
        self.assertEqual(config["global_batch_size"], 16)
        self.assertEqual(config["gradient_accumulation_steps"], 2)
        # No determinism switches unless --test-accuracy is given.
        self.assertEqual(config["csa_sparse_attn_backend"], "cudnn")
        self.assertEqual(model_config["multimax_modules"], ["lm_head"])
        # Environment-specific pin is always dropped.
        self.assertNotIn("fa_version", config)

    def test_existing_output_dir_needs_force(self):
        ok, message = self.adapt(target_nodes=1)
        self.assertTrue(ok, message)

        ok, message = self.adapt(target_nodes=1)
        self.assertFalse(ok)
        self.assertIn("--force", message)

        ok, message = self.adapt(target_nodes=1, force=True)
        self.assertTrue(ok, message)

    def test_unchanged_keys_are_not_reported(self):
        ok, message = self.adapt(
            target_nodes=1, auto_overrides={"max_steps": 1000000}
        )
        self.assertTrue(ok, message)
        # The value equals the source value, so nothing is logged for it.
        self.assertNotIn("max_steps", message)


class TestPerformanceSwitch(ConfigAdapterTestBase):
    """``--test-performance``: sharding + GBS only, everything else frozen."""

    def test_scales_gbs_and_sharding(self):
        ok, message = self.adapt(target_nodes=64, test_performance=True)
        self.assertTrue(ok, message)
        config = self.load_output_yaml(512)
        self.assertEqual(config["global_batch_size"], 1024)
        self.assertEqual(config["gradient_accumulation_steps"], 2)
        self.assertEqual(config["sharding_parallel_size"], 64)
        self.assertEqual(config["expert_model_parallel_size"], 64)
        self.assertEqual(config["pipeline_model_parallel_size"], 8)

    def test_rejects_incompatible_scale(self):
        ok, message = self.adapt(target_nodes=1, test_performance=True)
        self.assertFalse(ok)
        self.assertIn("C2", message)
        self.assertIn("--test-performance", message)

    def test_no_model_config_written(self):
        ok, message = self.adapt(target_nodes=64, test_performance=True)
        self.assertTrue(ok, message)
        self.assertFalse((self.output_dir / "model_config_separated").exists())
        self.assertEqual(
            JsonWriter().load(self.json_path)["n_routed_experts"], 256
        )


class TestAccuracySwitch(ConfigAdapterTestBase):
    """``--test-accuracy``: determinism switches + equivalent batch."""

    def test_keeps_effective_batch_and_pins_switches(self):
        ok, message = self.adapt(target_nodes=1, test_accuracy=True)
        self.assertTrue(ok, message)
        config = self.load_output_yaml(8)
        self.assertEqual(config["global_batch_size"], 1536)
        self.assertEqual(config["gradient_accumulation_steps"], 192)
        self.assertEqual(config["csa_sparse_attn_backend"], "tilelang")
        self.assertIsNone(self.load_output_json(8)["multimax_modules"])
        self.assertIn("精度对齐", message)

    def test_combined_with_performance_freezes_dims(self):
        ok, message = self.adapt(
            target_nodes=64, test_performance=True, test_accuracy=True
        )
        self.assertTrue(ok, message)
        config = self.load_output_yaml(512)
        # Frozen dims and acc, but the determinism switches still land.
        self.assertEqual(config["expert_model_parallel_size"], 64)
        self.assertEqual(config["gradient_accumulation_steps"], 2)
        self.assertEqual(config["global_batch_size"], 1024)
        self.assertEqual(config["csa_sparse_attn_backend"], "tilelang")
        self.assertIsNone(self.load_output_json(512)["multimax_modules"])

    def test_source_model_config_is_never_touched(self):
        ok, message = self.adapt(target_nodes=1, test_accuracy=True)
        self.assertTrue(ok, message)
        source = JsonWriter().load(self.json_path)
        self.assertEqual(source["num_hidden_layers"], 64)
        self.assertEqual(source["multimax_modules"], ["lm_head"])
        self.assertIn(
            "model_dir_adapted_8cards",
            self.load_output_yaml(8)["model_name_or_path"],
        )

    def test_ep_shrinks_to_two_not_one(self):
        self.write_yaml(
            SOURCE_YAML.replace(
                "pipeline_model_parallel_size: 8",
                "pipeline_model_parallel_size: 1",
            ).replace(
                "expert_model_parallel_size: 64",
                "expert_model_parallel_size: 8",
            )
        )
        ok, message = self.adapt(
            target_nodes=1, test_accuracy=True, cards_per_node=4
        )
        self.assertTrue(ok, message)
        self.assertEqual(
            self.load_output_yaml(4)["expert_model_parallel_size"], 4
        )

    def test_refuses_to_remove_the_last_ep_group(self):
        self.write_yaml(
            SOURCE_YAML.replace(
                "pipeline_model_parallel_size: 8",
                "pipeline_model_parallel_size: 1",
            ).replace(
                "expert_model_parallel_size: 64",
                "expert_model_parallel_size: 2",
            )
        )
        ok, message = self.adapt(
            target_nodes=1, test_accuracy=True, cards_per_node=1
        )
        self.assertFalse(ok)
        self.assertIn("最小值", message)


class TestAutoOverrideRouting(ConfigAdapterTestBase):
    """Prefix-less ``--set`` routes by which document declares the key."""

    def test_key_only_in_yaml(self):
        ok, message = self.adapt(
            target_nodes=1, auto_overrides={"max_steps": 10}
        )
        self.assertTrue(ok, message)
        self.assertEqual(self.load_output_yaml(8)["max_steps"], 10)
        self.assertNotIn("max_steps", self.load_output_json(8))

    def test_key_only_in_model_config_is_also_protected(self):
        ok, message = self.adapt(
            target_nodes=1, auto_overrides={"n_routed_experts": 128}
        )
        self.assertTrue(ok, message)
        # Routed to the JSON, and the EP shrink may not overwrite a --set.
        self.assertEqual(self.load_output_json(8)["n_routed_experts"], 128)
        self.assertNotIn("n_routed_experts", self.load_output_yaml(8))

    def test_key_in_both_documents_updates_both(self):
        self.write_yaml(SOURCE_YAML + "shared_flag: 1\n")
        self.write_json({**MODEL_CONFIG, "shared_flag": 1})
        ok, message = self.adapt(
            target_nodes=1, auto_overrides={"shared_flag": 2}
        )
        self.assertTrue(ok, message)
        self.assertEqual(self.load_output_yaml(8)["shared_flag"], 2)
        self.assertEqual(self.load_output_json(8)["shared_flag"], 2)

    def test_unknown_key_is_added_to_yaml(self):
        ok, message = self.adapt(
            target_nodes=1, auto_overrides={"brand_new_field": 7}
        )
        self.assertTrue(ok, message)
        self.assertEqual(self.load_output_yaml(8)["brand_new_field"], 7)
        self.assertNotIn("brand_new_field", self.load_output_json(8))

    def test_new_model_config_field_needs_explicit_prefix(self):
        ok, message = self.adapt(
            target_nodes=1, json_overrides={"brand_new_field": 7}
        )
        self.assertTrue(ok, message)
        self.assertEqual(self.load_output_json(8)["brand_new_field"], 7)
        self.assertNotIn("brand_new_field", self.load_output_yaml(8))


class TestErrorPaths(ConfigAdapterTestBase):
    """Failure branches of the adapter."""

    def test_empty_yaml_is_refused(self):
        self.write_yaml("")
        ok, message = self.adapt(target_nodes=1)
        self.assertFalse(ok)
        self.assertIn("配置文件为空", message)

    def test_json_override_without_a_model_config(self):
        self.write_yaml(SOURCE_YAML.replace("./model_dir", "./nope"))
        ok, message = self.adapt(
            target_nodes=1, json_overrides={"n_routed_experts": 32}
        )
        self.assertFalse(ok)
        self.assertIn("--set json:", message)

    def test_shrink_without_a_model_config(self):
        self.write_yaml(SOURCE_YAML.replace("./model_dir", "./nope"))
        ok, message = self.adapt(target_nodes=1)
        self.assertFalse(ok)
        self.assertIn("model_config.json", message)

    def test_unknown_source_scale_is_refused(self):
        # global_batch_size is present but nothing lets us infer the scale.
        self.write_yaml(
            SOURCE_YAML.replace("sharding_parallel_size: 96", "").replace(
                "gradient_accumulation_steps: 2", ""
            )
        )
        ok, message = self.adapt(target_nodes=1)
        self.assertFalse(ok)
        self.assertIn("无法推断源作业的卡数", message)

    def test_pp_only_shrink_for_a_dense_model(self):
        # EP=1 (no expert parallelism) leaves PP as the only axis to shrink,
        # and 4 cards cannot host PP=8 (C1).
        self.write_yaml(
            SOURCE_YAML.replace(
                "expert_model_parallel_size: 64",
                "expert_model_parallel_size: 1",
            )
        )
        ok, message = self.adapt(
            target_nodes=1, cards_per_node=4, test_accuracy=True
        )
        self.assertTrue(ok, message)
        config = self.load_output_yaml(4)
        self.assertEqual(config["pipeline_model_parallel_size"], 4)
        self.assertEqual(config["expert_model_parallel_size"], 1)
        self.assertEqual(self.load_output_json(4)["num_hidden_layers"], 32)
        self.assertIn("仅缩 PP", message)

    def test_missing_layer_count_blocks_pp_shrink(self):
        self.write_json(
            {k: v for k, v in MODEL_CONFIG.items() if k != "num_hidden_layers"}
        )
        self.write_yaml(
            SOURCE_YAML.replace(
                "expert_model_parallel_size: 64",
                "expert_model_parallel_size: 1",
            )
        )
        ok, message = self.adapt(target_nodes=1, cards_per_node=4)
        self.assertFalse(ok)
        self.assertIn("num_hidden_layers", message)


class TestCliErrorPaths(ConfigAdapterTestBase):
    """Argument validation, straight through main()."""

    def _run(self, argv):
        buffer, err = io.StringIO(), io.StringIO()
        with (
            contextlib.redirect_stdout(buffer),
            contextlib.redirect_stderr(err),
        ):
            code = main(argv)
        return code, buffer.getvalue() + err.getvalue()

    def test_missing_input_file(self):
        code, out = self._run(["--input", str(self.root / "nope.yaml")])
        self.assertEqual(code, 1)
        self.assertIn("输入文件不存在", out)

    def test_non_positive_cards_per_node(self):
        code, out = self._run(
            ["--input", str(self.yaml_path), "--cards-per-node", "0"]
        )
        self.assertEqual(code, 1)
        self.assertIn("cards-per-node", out)

    def test_non_positive_target_nodes(self):
        code, out = self._run(
            ["--input", str(self.yaml_path), "--target-nodes", "0"]
        )
        self.assertEqual(code, 1)
        self.assertIn("target-nodes", out)

    def test_malformed_set(self):
        code, out = self._run(
            [
                "--input",
                str(self.yaml_path),
                "--target-nodes",
                "1",
                "--set",
                "json:",
            ]
        )
        self.assertEqual(code, 1)
        self.assertIn("--set", out)

    def test_adaptation_failure_is_reported(self):
        code, out = self._run(
            [
                "--input",
                str(self.yaml_path),
                "--target-nodes",
                "1",
                "--test-performance",
                "--output-dir",
                str(self.output_dir),
            ]
        )
        self.assertEqual(code, 1)
        self.assertIn("适配失败", out)

    def test_python_m_entry_point(self):
        # `python -m paddlefleet.config_adapter` must reach the same main().
        argv = ["paddlefleet.config_adapter", "--input", str(self.yaml_path)]
        buffer = io.StringIO()
        with (
            mock.patch.object(sys, "argv", argv),
            contextlib.redirect_stdout(buffer),
            self.assertRaises(SystemExit) as ctx,
        ):
            runpy.run_module("paddlefleet.config_adapter", run_name="__main__")
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("VALID_NODES=", buffer.getvalue())


class TestLayerFields(ConfigAdapterTestBase):
    """Per-layer lists must follow num_hidden_layers, or the shrink is refused."""

    def _with_layer_fields(self, ratios):
        """Source JSON carrying per-layer lists consistent with 64 layers."""
        self.write_json(
            {
                **MODEL_CONFIG,
                "csa_compress_ratios": [*ratios, -2],  # 64 layers + 1 MTP
                "window_attn_skip_freq": [0] * 65,
                "layer_types": ["full_attention"] * 64,
            }
        )

    def test_lists_are_truncated_with_the_layer_count(self):
        self._with_layer_fields([128, 128, 128, -2] * 16)
        ok, message = self.adapt(target_nodes=1)
        self.assertTrue(ok, message)
        model_config = self.load_output_json(8)
        # 64 -> 16 layers, MTP entries kept at the tail.
        self.assertEqual(model_config["num_hidden_layers"], 16)
        self.assertEqual(len(model_config["csa_compress_ratios"]), 17)
        self.assertEqual(model_config["csa_compress_ratios"][-1], -2)
        self.assertEqual(len(model_config["window_attn_skip_freq"]), 17)
        self.assertEqual(len(model_config["layer_types"]), 16)
        # Both attention families of the source pattern survive.
        self.assertIn(128, model_config["csa_compress_ratios"][:16])
        self.assertIn(-2, model_config["csa_compress_ratios"][:16])
        self.assertIn("逐层配置", message)

    def test_refuses_when_an_attention_family_would_vanish(self):
        # Every -2 layer sits beyond the shrunk layer range.
        self._with_layer_fields([128] * 60 + [-2] * 4)
        ok, message = self.adapt(target_nodes=1)
        self.assertFalse(ok)
        self.assertIn("丢掉注意力类型", message)

    def test_refuses_when_the_source_lists_are_inconsistent(self):
        self.write_json({**MODEL_CONFIG, "csa_compress_ratios": [128] * 10})
        ok, message = self.adapt(target_nodes=1)
        self.assertFalse(ok)
        self.assertIn("不自洽", message)

    def test_scalar_forms_are_left_alone(self):
        self.write_json({**MODEL_CONFIG, "window_attn_skip_freq": 4})
        ok, message = self.adapt(target_nodes=1)
        self.assertTrue(ok, message)
        self.assertEqual(self.load_output_json(8)["window_attn_skip_freq"], 4)

    def test_mtp_num_layers_alias_wins_over_a_zero(self):
        # The framework resolves the MTP count as
        # `mtp_num_layers or num_nextn_predict_layers`, so a zero in the
        # second key must not hide a valid value in the first.
        self.write_json(
            {
                **MODEL_CONFIG,
                "num_nextn_predict_layers": 0,
                "mtp_num_layers": 2,
                "csa_compress_ratios": [128, 128, 128, -2] * 16 + [-2, -2],
                "layer_types": ["full_attention"] * 64,
            }
        )
        ok, message = self.adapt(target_nodes=1)
        self.assertTrue(ok, message)
        model_config = self.load_output_json(8)
        self.assertEqual(model_config["num_hidden_layers"], 16)
        # 16 layers + the 2 MTP entries.
        self.assertEqual(len(model_config["csa_compress_ratios"]), 18)
        self.assertEqual(len(model_config["layer_types"]), 16)


class TestFailureLeavesSourcesUntouched(ConfigAdapterTestBase):
    """Nothing is written until every check has passed."""

    def test_batch_failure_does_not_touch_either_source(self):
        # 1537 * 8 / 768 is not an integer, so batch scaling fails *after*
        # the model structure has been planned.
        self.write_yaml(
            SOURCE_YAML.replace(
                "global_batch_size: 1536", "global_batch_size: 1537"
            )
        )
        ok, message = self.adapt(target_nodes=1, in_place=True)
        self.assertFalse(ok)
        self.assertIn("无法整除", message)
        # Source model_config.json keeps its original structure ...
        source_json = JsonWriter().load(self.json_path)
        self.assertEqual(source_json["num_hidden_layers"], 64)
        self.assertEqual(source_json["n_routed_experts"], 256)
        # ... and the YAML still declares the original parallelism.
        config = YamlWriter().load(self.yaml_path)
        self.assertEqual(config["pipeline_model_parallel_size"], 8)
        self.assertEqual(config["expert_model_parallel_size"], 64)

    def test_no_adapted_dir_is_created_on_failure(self):
        self.write_yaml(
            SOURCE_YAML.replace(
                "global_batch_size: 1536", "global_batch_size: 1537"
            )
        )
        ok, _message = self.adapt(target_nodes=1)
        self.assertFalse(ok)
        self.assertFalse((self.output_dir / "model_config_separated").exists())


class TestPinnedModelPathConflict(ConfigAdapterTestBase):
    """A pinned model_name_or_path cannot coexist with a JSON rewrite."""

    def test_pinning_the_path_is_rejected(self):
        ok, message = self.adapt(
            target_nodes=1,
            yaml_overrides={"model_name_or_path": "./model_dir"},
        )
        self.assertFalse(ok)
        self.assertIn("model_name_or_path", message)
        self.assertIn("--in-place", message)

    def test_in_place_accepts_a_pinned_path(self):
        ok, message = self.adapt(
            target_nodes=1,
            in_place=True,
            yaml_overrides={"model_name_or_path": "./model_dir"},
        )
        self.assertTrue(ok, message)
        config = YamlWriter().load(self.yaml_path)
        self.assertEqual(config["model_name_or_path"], "./model_dir")
        self.assertEqual(
            JsonWriter().load(self.json_path)["num_hidden_layers"], 16
        )


class TestInPlace(ConfigAdapterTestBase):
    """In-place rewrites both sources and keeps model_name_or_path."""

    def test_rewrites_sources_without_a_banner(self):
        ok, message = self.adapt(
            target_nodes=1, test_accuracy=True, in_place=True
        )
        self.assertTrue(ok, message)
        config = YamlWriter().load(self.yaml_path)
        self.assertEqual(config["pipeline_model_parallel_size"], 2)
        self.assertEqual(config["model_name_or_path"], "./model_dir")
        self.assertNotIn(
            "[config_adapter]", self.yaml_path.read_text(encoding="utf-8")
        )
        self.assertEqual(
            JsonWriter().load(self.json_path)["num_hidden_layers"], 16
        )


class TestGeneratedPaths(ConfigAdapterTestBase):
    """model_name_or_path must stay resolvable."""

    def test_absolute_when_output_escapes_the_yaml_dir(self):
        outside = Path(tempfile.mkdtemp())
        try:
            ok, message = self.adapt(target_nodes=1, output_dir=outside)
            self.assertTrue(ok, message)
            path = YamlWriter().load(outside / "source_adapted_8cards.yaml")[
                "model_name_or_path"
            ]
            self.assertTrue(Path(path).is_absolute(), path)
            self.assertTrue(Path(path, "model_config.json").is_file())
        finally:
            shutil.rmtree(outside, ignore_errors=True)


class TestInspection(ConfigAdapterTestBase):
    """No target scale: report the source scale and legal node counts."""

    def test_reports_source_scale_and_valid_nodes(self):
        orig_cards, orig_nodes, valid_nodes = inspect_config(
            self.yaml_path, cards_per_node=8
        )
        self.assertEqual(orig_cards, 768)
        self.assertEqual(orig_nodes, 96)
        self.assertEqual(valid_nodes, [64])


class TestCli(ConfigAdapterTestBase):
    """End-to-end checks through the argv entry point."""

    def _run(self, argv):
        """Run the CLI, returning ``(exit_code, stdout)``."""
        buffer = io.StringIO()
        with (
            contextlib.redirect_stdout(buffer),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            code = main(argv)
        return code, buffer.getvalue()

    def test_target_without_any_switch_still_adapts(self):
        code, out = self._run(
            [
                "--input",
                str(self.yaml_path),
                "--target-nodes",
                "1",
                "--output-dir",
                str(self.output_dir),
            ]
        )
        self.assertEqual(code, 0, out)
        self.assertIn("未指定测试维度", out)

    def test_both_switches_accepted(self):
        code, out = self._run(
            [
                "--input",
                str(self.yaml_path),
                "--target-nodes",
                "64",
                "--test-performance",
                "--test-accuracy",
                "--output-dir",
                str(self.output_dir),
            ]
        )
        self.assertEqual(code, 0, out)
        self.assertIn("performance+accuracy", out)

    def test_inspection_output_is_machine_readable(self):
        code, out = self._run(["--input", str(self.yaml_path)])
        self.assertEqual(code, 0)
        self.assertIn("ORIGINAL_CARDS=768", out)
        self.assertIn("VALID_NODES=64", out)

    def test_in_place_writes_a_patch(self):
        code, out = self._run(
            [
                "--input",
                str(self.yaml_path),
                "--target-nodes",
                "1",
                "--test-accuracy",
                "--in-place",
            ]
        )
        self.assertEqual(code, 0, out)
        patch = self.yaml_path.parent / (self.yaml_path.name + ".patch")
        self.assertTrue(patch.is_file())
        text = patch.read_text(encoding="utf-8")
        self.assertIn("pipeline_model_parallel_size", text)
        self.assertIn("model_config.json", text)


if __name__ == "__main__":
    unittest.main()
