# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Behavior tests for paddlefleet.quantization.quantization_config.QuantizationConfig.

Expected values are hand-derived from the constructor signature, the
``quant_inference_mapping`` / ``fp8_format_mapping`` tables and the validation
branches in the production module. The function under test is never used to
build its own expected values.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

CONFIG_MODULE = "paddlefleet.quantization.quantization_config"


def _load_config_cls():
    from paddlefleet.quantization.quantization_config import QuantizationConfig

    return QuantizationConfig


class TestQuantizationConfigDefaults(unittest.TestCase):
    """Default values are the primary config contract."""

    def test_default_values(self):
        QuantizationConfig = _load_config_cls()
        config = QuantizationConfig()

        # Independently enumerated from the constructor signature defaults.
        expected = {
            "weight_quantize_algo": None,
            "quant_type": None,
            "shift": False,
            "smooth": False,
            "shift_smooth_all_linears": False,
            "quant_round_type": 0,
            "llm_int8_threshold": 6.0,
            "qlora_weight_double_quant": False,
            "qlora_weight_blocksize": 64,
            "qlora_weight_double_quant_block_size": 256,
            "weight_quant_method": "abs_max_channel_wise",
            # act_quant_method="abs_max" is mapped through quant_inference_mapping
            # ("abs_max" -> "abs_max"), so the stored value stays "abs_max".
            "act_quant_method": "abs_max",
            "activation_scheme": None,
            "fmt": None,
            "quant_method": None,
            "weight_block_size": None,
            "dtype": None,
            "ignore_modules": None,
            "group_size": -1,
            "apply_hadamard": False,
            "hadamard_block_size": 32,
            "quant_input_grad": False,
            "quant_weight_grad": False,
            "apply_online_actscale_step": 200,
            "actscale_moving_rate": 0.01,
            "fp8_format_type": "hybrid",
            "scale_epsilon": 1e-8,
            "dense_quant_type": "",
            "moe_quant_type": "",
            "quantization": "",
            "quantization_linear_list": [],
        }
        for name, value in expected.items():
            self.assertEqual(getattr(config, name), value, msg=name)

    def test_default_types(self):
        QuantizationConfig = _load_config_cls()
        config = QuantizationConfig()
        # bool defaults must be genuine bools, not truthy/falsy surrogates.
        self.assertIs(config.shift, False)
        self.assertIs(config.smooth, False)
        self.assertIs(config.apply_hadamard, False)
        self.assertIsInstance(config.group_size, int)
        self.assertIsInstance(config.qlora_weight_blocksize, int)
        self.assertIsInstance(config.llm_int8_threshold, float)
        self.assertIsInstance(config.actscale_moving_rate, float)


class TestQuantizationConfigValidation(unittest.TestCase):
    """Constructor validation of algorithms and quant types."""

    def test_unsupported_string_algo_raises(self):
        QuantizationConfig = _load_config_cls()
        with self.assertRaises(ValueError) as ctx:
            QuantizationConfig(weight_quantize_algo="totally_made_up")
        self.assertIn("not in supported list", str(ctx.exception))

    def test_unsupported_dict_algo_raises(self):
        QuantizationConfig = _load_config_cls()
        with self.assertRaises(ValueError) as ctx:
            QuantizationConfig(weight_quantize_algo={"totally_made_up": [".*"]})
        self.assertIn("not in supported list", str(ctx.exception))

    def test_supported_string_algos_accepted(self):
        QuantizationConfig = _load_config_cls()
        # Independently transcribed from the accepted-list branch (excludes
        # fp8linear, which is arch-gated and covered separately).
        for algo in [
            "weight_only_int8",
            "weight_only_int4",
            "llm.int8",
            "a8w8",
            "nf4",
            "fp4",
            "a8w8linear",
            "a8w4linear",
        ]:
            config = QuantizationConfig(weight_quantize_algo=algo)
            self.assertEqual(config.weight_quantize_algo, algo)

    def test_dict_algo_preserved(self):
        QuantizationConfig = _load_config_cls()
        algo = {"weight_only_int8": [".*mlp.*"]}
        config = QuantizationConfig(weight_quantize_algo=algo)
        self.assertEqual(config.weight_quantize_algo, algo)

    def test_unsupported_quant_type_raises(self):
        QuantizationConfig = _load_config_cls()
        with self.assertRaises(ValueError) as ctx:
            QuantizationConfig(quant_type="not_a_quant_type")
        self.assertIn("not in supported list", str(ctx.exception))

    def test_supported_quant_types_accepted(self):
        QuantizationConfig = _load_config_cls()
        for qt in [
            "weight_only_int8",
            "weight_only_int4",
            "a8w8",
            "a8w8c8",
            "a8w8_fp8",
            "a8w8c8_fp8",
        ]:
            config = QuantizationConfig(quant_type=qt)
            self.assertEqual(config.quant_type, qt)


class TestQuantizationConfigAliasesAndFlags(unittest.TestCase):
    """act_quant_method alias mapping and boolean flag handling."""

    def test_act_quant_method_avg_alias_maps_to_abs_max(self):
        QuantizationConfig = _load_config_cls()
        # quant_inference_mapping aliases "avg" -> "abs_max".
        config = QuantizationConfig(act_quant_method="avg")
        self.assertEqual(config.act_quant_method, "abs_max")

    def test_act_quant_method_channel_wise_passthrough(self):
        QuantizationConfig = _load_config_cls()
        config = QuantizationConfig(act_quant_method="abs_max_channel_wise")
        self.assertEqual(config.act_quant_method, "abs_max_channel_wise")

    def test_act_quant_method_unknown_raises_keyerror(self):
        QuantizationConfig = _load_config_cls()
        # Values outside quant_inference_mapping are rejected by the dict lookup.
        with self.assertRaises(KeyError):
            QuantizationConfig(act_quant_method="not_mapped")

    def test_explicit_shift_smooth_flags(self):
        QuantizationConfig = _load_config_cls()
        config = QuantizationConfig(
            shift=True, smooth=True, shift_smooth_all_linears=True
        )
        self.assertIs(config.shift, True)
        self.assertIs(config.smooth, True)
        self.assertIs(config.shift_smooth_all_linears, True)

    def test_extra_kwargs_are_ignored(self):
        QuantizationConfig = _load_config_cls()
        # Unknown constructor kwargs are swallowed by **kwargs, not stored.
        config = QuantizationConfig(unknown_param="value")
        self.assertFalse(hasattr(config, "unknown_param"))


class TestQuantizationConfigFp8Arch(unittest.TestCase):
    """fp8linear is gated to Hopper architectures (sm 89/90)."""

    def test_fp8linear_on_non_hopper_raises(self):
        QuantizationConfig = _load_config_cls()
        with patch(f"{CONFIG_MODULE}._get_arch_info", return_value=80):
            with self.assertRaises(RuntimeError) as ctx:
                QuantizationConfig(weight_quantize_algo="fp8linear")
            self.assertIn("Hopper", str(ctx.exception))

    def test_fp8linear_on_hopper_89_ok(self):
        QuantizationConfig = _load_config_cls()
        with patch(f"{CONFIG_MODULE}._get_arch_info", return_value=89):
            config = QuantizationConfig(weight_quantize_algo="fp8linear")
            self.assertEqual(config.weight_quantize_algo, "fp8linear")

    def test_fp8linear_on_hopper_90_ok(self):
        QuantizationConfig = _load_config_cls()
        with patch(f"{CONFIG_MODULE}._get_arch_info", return_value=90):
            config = QuantizationConfig(weight_quantize_algo="fp8linear")
            self.assertEqual(config.weight_quantize_algo, "fp8linear")

    def test_fp8linear_dict_on_non_hopper_raises(self):
        QuantizationConfig = _load_config_cls()
        with patch(f"{CONFIG_MODULE}._get_arch_info", return_value=80):
            with self.assertRaises(RuntimeError):
                QuantizationConfig(
                    weight_quantize_algo={"fp8linear": [".*mlp.*"]}
                )


class TestQuantizationConfigConsumers(unittest.TestCase):
    """Fields must be consumed by the predicate/property helpers."""

    def test_fp8_format_hybrid(self):
        QuantizationConfig = _load_config_cls()
        # hybrid uses e5m2 for grad_output; independent of the other two.
        fmt = QuantizationConfig(fp8_format_type="hybrid").fp8_format
        self.assertEqual(fmt["weight"], "float8_e4m3fn")
        self.assertEqual(fmt["activation"], "float8_e4m3fn")
        self.assertEqual(fmt["grad_output"], "float8_e5m2")

    def test_fp8_format_e4m3(self):
        QuantizationConfig = _load_config_cls()
        # e4m3 uses e4m3 for grad_output too, distinguishing it from hybrid.
        fmt = QuantizationConfig(fp8_format_type="e4m3").fp8_format
        self.assertEqual(fmt["weight"], "float8_e4m3fn")
        self.assertEqual(fmt["activation"], "float8_e4m3fn")
        self.assertEqual(fmt["grad_output"], "float8_e4m3fn")

    def test_is_weight_quantize_none_is_false(self):
        QuantizationConfig = _load_config_cls()
        self.assertFalse(QuantizationConfig().is_weight_quantize())

    def test_is_weight_quantize_dict_is_true(self):
        QuantizationConfig = _load_config_cls()
        config = QuantizationConfig(
            weight_quantize_algo={"weight_only_int8": [".*mlp.*"]}
        )
        self.assertTrue(config.is_weight_quantize())

    def test_is_weight_quantize_supported_strings_true(self):
        QuantizationConfig = _load_config_cls()
        for algo in [
            "weight_only_int8",
            "weight_only_int4",
            "llm.int8",
            "nf4",
            "fp4",
            "a8w8",
            "a8w8linear",
            "a8w4linear",
        ]:
            config = QuantizationConfig(weight_quantize_algo=algo)
            self.assertTrue(config.is_weight_quantize(), msg=algo)

    def test_merge_tensor_parallel_unsupported_algos_false(self):
        QuantizationConfig = _load_config_cls()
        # These four are explicitly listed as NOT mergeable.
        for algo in [
            "weight_only_int8",
            "weight_only_int4",
            "llm.int8",
            "a8w8",
        ]:
            config = QuantizationConfig(weight_quantize_algo=algo)
            self.assertFalse(
                config.is_support_merge_tensor_parallel(), msg=algo
            )

    def test_merge_tensor_parallel_other_algos_true(self):
        QuantizationConfig = _load_config_cls()
        # nf4/fp4/a8w8linear/a8w4linear fall into the else branch -> True.
        for algo in ["nf4", "fp4", "a8w8linear", "a8w4linear"]:
            config = QuantizationConfig(weight_quantize_algo=algo)
            self.assertTrue(config.is_support_merge_tensor_parallel(), msg=algo)

    def test_merge_tensor_parallel_none_is_true(self):
        QuantizationConfig = _load_config_cls()
        self.assertTrue(QuantizationConfig().is_support_merge_tensor_parallel())


class TestQuantizationConfigFromDict(unittest.TestCase):
    """from_dict construction and unused-kwargs handling."""

    def test_from_dict_builds_config(self):
        QuantizationConfig = _load_config_cls()
        config = QuantizationConfig.from_dict(
            {"weight_quantize_algo": "weight_only_int8", "group_size": 32}
        )
        self.assertEqual(config.weight_quantize_algo, "weight_only_int8")
        self.assertEqual(config.group_size, 32)

    def test_from_dict_extra_kwarg_overrides_attr(self):
        QuantizationConfig = _load_config_cls()
        # shift is a real attr, so it is set via setattr and consumed.
        config = QuantizationConfig.from_dict(
            {"weight_quantize_algo": "nf4"}, shift=True
        )
        self.assertEqual(config.weight_quantize_algo, "nf4")
        self.assertIs(config.shift, True)

    def test_from_dict_return_unused_keeps_unknown(self):
        QuantizationConfig = _load_config_cls()
        config, unused = QuantizationConfig.from_dict(
            {"weight_quantize_algo": "a8w8"},
            return_unused_kwargs=True,
            unknown_param="value",
        )
        self.assertEqual(config.weight_quantize_algo, "a8w8")
        # unknown_param is not a config attr, so it is returned as unused.
        self.assertEqual(unused, {"unknown_param": "value"})

    def test_from_dict_return_unused_consumes_known(self):
        QuantizationConfig = _load_config_cls()
        config, unused = QuantizationConfig.from_dict(
            {"weight_quantize_algo": "a8w8"},
            return_unused_kwargs=True,
            shift=True,
        )
        # shift is a real attr -> consumed, so unused ends up empty.
        self.assertIs(config.shift, True)
        self.assertEqual(unused, {})


class TestQuantizationConfigSerialization(unittest.TestCase):
    """to_dict / to_diff_dict / to_json_* and save-reload round trips."""

    def test_to_dict_is_deep_copy(self):
        QuantizationConfig = _load_config_cls()
        config = QuantizationConfig(weight_quantize_algo="nf4", group_size=32)
        d = config.to_dict()
        self.assertEqual(d["weight_quantize_algo"], "nf4")
        self.assertEqual(d["group_size"], 32)
        # Mutating the returned dict must not affect the live config.
        d["weight_quantize_algo"] = "mutated"
        self.assertEqual(config.weight_quantize_algo, "nf4")

    def test_to_diff_dict_only_changed(self):
        QuantizationConfig = _load_config_cls()
        config = QuantizationConfig(weight_quantize_algo="nf4")
        # Only weight_quantize_algo differs from defaults; everything else
        # equals the fresh-default config, so the diff is exactly this key.
        self.assertEqual(config.to_diff_dict(), {"weight_quantize_algo": "nf4"})

    def test_to_diff_dict_default_is_empty(self):
        QuantizationConfig = _load_config_cls()
        self.assertEqual(QuantizationConfig().to_diff_dict(), {})

    def test_to_json_string_diff(self):
        QuantizationConfig = _load_config_cls()
        config = QuantizationConfig(weight_quantize_algo="nf4", group_size=64)
        data = json.loads(config.to_json_string(use_diff=True))
        self.assertEqual(
            data, {"weight_quantize_algo": "nf4", "group_size": 64}
        )

    def test_to_json_string_full(self):
        QuantizationConfig = _load_config_cls()
        config = QuantizationConfig()
        data = json.loads(config.to_json_string(use_diff=False))
        # Full snapshot includes default fields that the diff would omit.
        self.assertIn("shift", data)
        self.assertIn("smooth", data)
        self.assertEqual(data["group_size"], -1)
        self.assertEqual(data["fp8_format_type"], "hybrid")

    def test_to_dict_from_dict_round_trip(self):
        QuantizationConfig = _load_config_cls()
        original = QuantizationConfig(
            weight_quantize_algo="a8w8",
            quant_type="a8w8",
            shift=True,
            group_size=32,
            hadamard_block_size=64,
            ignore_modules=[".*out_linear.*"],
        )
        restored = QuantizationConfig.from_dict(original.to_dict())
        for name in [
            "weight_quantize_algo",
            "quant_type",
            "shift",
            "group_size",
            "hadamard_block_size",
            "ignore_modules",
            "act_quant_method",
        ]:
            self.assertEqual(
                getattr(restored, name), getattr(original, name), msg=name
            )

    def test_to_json_file_save_reload(self):
        QuantizationConfig = _load_config_cls()
        original = QuantizationConfig(
            weight_quantize_algo="weight_only_int4",
            shift=True,
            group_size=128,
        )
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            original.to_json_file(path)
            with open(path, "r", encoding="utf-8") as f:
                on_disk = json.load(f)
            # Full dict is persisted (not the diff), then reconstructed.
            self.assertEqual(
                on_disk["weight_quantize_algo"], "weight_only_int4"
            )
            self.assertIs(on_disk["shift"], True)
            self.assertEqual(on_disk["group_size"], 128)
            reloaded = QuantizationConfig.from_dict(on_disk)
            self.assertEqual(reloaded.weight_quantize_algo, "weight_only_int4")
            self.assertIs(reloaded.shift, True)
            self.assertEqual(reloaded.group_size, 128)
            # A default field survives the save/reload cycle unchanged.
            self.assertEqual(reloaded.hadamard_block_size, 32)
        finally:
            os.unlink(path)


class TestQuantizationConfigMutableDefaultBug(unittest.TestCase):
    """Documents a real shared-mutable-default defect in the constructor.

    ``quantization_linear_list=[]`` is a mutable default argument evaluated once
    at function-definition time, so every QuantizationConfig() built without an
    explicit list shares the *same* list object. Mutating one instance's list
    leaks into all other default instances. The correct behavior is that each
    instance owns an independent list; this test asserts that correct behavior
    and is expected to fail until the production default is fixed
    (e.g. default None then assign a fresh list). Production is NOT modified.
    """

    @unittest.expectedFailure
    def test_default_lists_are_independent(self):
        QuantizationConfig = _load_config_cls()
        first = QuantizationConfig()
        second = QuantizationConfig()
        # Restore the shared module-level default even though this xfails.
        self.addCleanup(first.quantization_linear_list.clear)
        first.quantization_linear_list.append(".*mlp.*")
        # Correct: a distinct default instance must remain empty.
        self.assertEqual(second.quantization_linear_list, [])


if __name__ == "__main__":
    unittest.main()
