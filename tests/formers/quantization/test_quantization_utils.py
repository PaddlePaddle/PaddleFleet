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

"""Behavior tests for paddlefleet.quantization.quantization_utils.

Scope note (无卡 / CPU): this module is quantization *orchestration* -- it
selects the weight-quantize algorithm from the config, dispatches to the right
converter, and rewrites state-dict keys. The actual numeric quant/dequant math
(scale axis/shape, packing, rounding/saturation, dequant relation, FP8/FP4)
lives in GPU-only collaborators that are reached via ``.cuda()``:
``paddle.nn.quant.weight_quantize``, ``qat_utils.quantize``,
``quantization_linear.dequant_weight`` and ``qlora_weight_quantize``. Those
numeric protocols must be validated on a real GPU (single-card tests); they are
NOT exercised here. What is verified here is the CPU-testable control logic,
against independently hand-derived expected values (real config objects, not
MagicMock, and no call to the function under test to build the expected).
"""

import unittest
from unittest import mock

from paddlefleet.quantization.quantization_config import QuantizationConfig
from paddlefleet.quantization.quantization_utils import (
    convert_to_qlora_state_dict,
    convert_to_quantize_state_dict,
    convert_to_weight_quantize_state_dict,
    parse_weight_quantize_algo,
    update_loaded_state_dict_keys,
)

_MODULE = "paddlefleet.quantization.quantization_utils"


class ParseWeightQuantizeAlgoTest(unittest.TestCase):
    """parse_weight_quantize_algo: algo selection / ignore / regex fullmatch."""

    def test_string_algo_applies_regardless_of_name(self):
        cfg = QuantizationConfig(
            weight_quantize_algo="weight_only_int8", ignore_modules=None
        )
        # A plain string algo is returned for any (non-ignored) module name.
        self.assertEqual(
            parse_weight_quantize_algo(cfg, "model.layers.7.mlp.gate_proj"),
            "weight_only_int8",
        )

    def test_ignore_modules_uses_fullmatch_not_partial(self):
        cfg = QuantizationConfig(
            weight_quantize_algo="weight_only_int8",
            ignore_modules=["layer\\.0"],
        )
        # Exact fullmatch -> ignored -> None.
        self.assertIsNone(parse_weight_quantize_algo(cfg, "layer.0"))
        # "layer.01" is only a *prefix* match; re.fullmatch must reject it, so
        # the module is NOT ignored and the string algo still applies. A buggy
        # re.match/re.search-based guard would wrongly ignore it and return None.
        self.assertEqual(
            parse_weight_quantize_algo(cfg, "layer.01"), "weight_only_int8"
        )
        # Leading extra chars are likewise not a full match.
        self.assertEqual(
            parse_weight_quantize_algo(cfg, "xlayer.0"), "weight_only_int8"
        )

    def test_dict_algo_selects_matching_pattern(self):
        cfg = QuantizationConfig(
            weight_quantize_algo={
                "weight_only_int8": ["layer\\.0"],
                "weight_only_int4": ["layer\\.1"],
            },
            ignore_modules=None,
        )
        self.assertEqual(
            parse_weight_quantize_algo(cfg, "layer.0"), "weight_only_int8"
        )
        self.assertEqual(
            parse_weight_quantize_algo(cfg, "layer.1"), "weight_only_int4"
        )

    def test_dict_algo_no_match_returns_none(self):
        cfg = QuantizationConfig(
            weight_quantize_algo={"weight_only_int8": ["layer\\.0"]},
            ignore_modules=None,
        )
        self.assertIsNone(parse_weight_quantize_algo(cfg, "layer.99"))

    def test_dict_algo_last_matching_entry_wins(self):
        # Characterizes the real (order-dependent) behavior: the loop does NOT
        # break on the first match, so when a name matches patterns under more
        # than one algo, the last one in dict-insertion order is returned.
        cfg = QuantizationConfig(
            weight_quantize_algo={
                "weight_only_int8": [".*"],
                "weight_only_int4": [".*"],
            },
            ignore_modules=None,
        )
        self.assertEqual(
            parse_weight_quantize_algo(cfg, "layer.0"), "weight_only_int4"
        )

    def test_ignore_takes_precedence_over_dict_algo(self):
        cfg = QuantizationConfig(
            weight_quantize_algo={"weight_only_int8": ["layer\\.0"]},
            ignore_modules=["layer\\.0"],
        )
        # Even though the dict pattern would match, the ignore guard runs first.
        self.assertIsNone(parse_weight_quantize_algo(cfg, "layer.0"))


class UpdateLoadedStateDictKeysTest(unittest.TestCase):
    """update_loaded_state_dict_keys: weight-key -> quant-key rewriting."""

    def test_weight_only_int8_replaces_weight_with_quant_and_scale(self):
        cfg = QuantizationConfig(
            weight_quantize_algo="weight_only_int8",
            qlora_weight_double_quant=False,
        )
        keys = ["layer.0.weight"]
        result = update_loaded_state_dict_keys(keys, ["layer.0"], cfg)
        # Hand-derived: weight removed; quant_weight then weight_scale appended.
        # int8 is not an activation-quant algo, so NO activation_scale.
        self.assertEqual(
            result, ["layer.0.quant_weight", "layer.0.weight_scale"]
        )

    def test_activation_quant_algo_appends_activation_scale(self):
        cfg = QuantizationConfig(
            weight_quantize_algo="a8w8linear",
            qlora_weight_double_quant=False,
        )
        keys = ["layer.0.weight"]
        result = update_loaded_state_dict_keys(keys, ["layer.0"], cfg)
        # a8w8linear is an activation-quant algo -> activation_scale is added
        # in addition to quant_weight and weight_scale (exact order).
        self.assertEqual(
            result,
            [
                "layer.0.quant_weight",
                "layer.0.weight_scale",
                "layer.0.activation_scale",
            ],
        )

    def test_double_quant_appends_triple_scale_keys(self):
        cfg = QuantizationConfig(
            weight_quantize_algo="nf4",
            qlora_weight_double_quant=True,
        )
        keys = ["layer.0.weight"]
        result = update_loaded_state_dict_keys(keys, ["layer.0"], cfg)
        # Double-quant branch: quant_weight + qweight_scale + double_weight_scale
        # + weight_scale_offset, and (per the code) NO plain weight_scale.
        self.assertEqual(
            result,
            [
                "layer.0.quant_weight",
                "layer.0.qweight_scale",
                "layer.0.double_weight_scale",
                "layer.0.weight_scale_offset",
            ],
        )

    def test_already_quantized_keys_are_left_unchanged(self):
        cfg = QuantizationConfig(
            weight_quantize_algo="weight_only_int8",
            qlora_weight_double_quant=False,
        )
        keys = ["layer.0.quant_weight", "layer.0.weight_scale"]
        result = update_loaded_state_dict_keys(keys, ["layer.0"], cfg)
        # Guard: quant_weight AND weight_scale present -> skip, no mutation.
        self.assertEqual(
            result, ["layer.0.quant_weight", "layer.0.weight_scale"]
        )

    def test_missing_weight_leaves_list_unchanged(self):
        cfg = QuantizationConfig(
            weight_quantize_algo="weight_only_int8",
            qlora_weight_double_quant=False,
        )
        keys = ["layer.0.bias"]
        result = update_loaded_state_dict_keys(
            keys, ["layer.0"], cfg, ignore_warning=True
        )
        # Neither weight nor quant keys present -> warn/skip, no key injected.
        self.assertEqual(result, ["layer.0.bias"])


class ConvertWeightQuantizeStateDictGuardTest(unittest.TestCase):
    """convert_to_weight_quantize_state_dict: CPU-reachable early-return guards.

    The quantizing branch calls ``.cuda()`` and a GPU kernel, so only the
    early-return paths are exercised on CPU. Sentinel objects stand in for
    tensors because these paths never touch tensor values.
    """

    def test_returns_early_when_already_quantized_without_requantizing(self):
        cfg = QuantizationConfig(weight_quantize_algo="weight_only_int8")
        qw, ws, w = object(), object(), object()
        state_dict = {
            "layer.0.quant_weight": qw,
            "layer.0.weight_scale": ws,
            "layer.0.weight": w,  # present but must be left untouched
        }
        result = convert_to_weight_quantize_state_dict(
            state_dict, "layer.0", cfg, "float16", "weight_only_int8"
        )
        # Guard fired: the raw weight is NOT popped (no re-quantization / no
        # GPU call), and the existing quant tensors are preserved by identity.
        self.assertIn("layer.0.weight", result)
        self.assertIs(result["layer.0.quant_weight"], qw)
        self.assertIs(result["layer.0.weight_scale"], ws)
        self.assertIs(result["layer.0.weight"], w)

    def test_returns_unchanged_when_no_weight_present(self):
        cfg = QuantizationConfig(weight_quantize_algo="weight_only_int8")
        state_dict = {}
        result = convert_to_weight_quantize_state_dict(
            state_dict, "layer.0", cfg, "float16", "weight_only_int8"
        )
        self.assertIs(result, state_dict)
        self.assertEqual(result, {})


class ConvertToQuantizeStateDictDispatchTest(unittest.TestCase):
    """convert_to_quantize_state_dict: real parse-driven dispatch decisions."""

    def test_unsupported_algo_raises_not_implemented(self):
        # "a8w8" is accepted by QuantizationConfig's validation but is NOT one
        # of the algos handled by convert_to_quantize_state_dict (neither the
        # weight-quant list nor {fp4, nf4}). Driven through the REAL
        # parse_weight_quantize_algo (no mock), it must reach NotImplementedError
        # before any GPU work.
        cfg = QuantizationConfig(weight_quantize_algo="a8w8")
        state_dict = {"layer.0.weight": object()}
        with self.assertRaises(NotImplementedError):
            convert_to_quantize_state_dict(
                state_dict, ["layer.0"], cfg, "float16"
            )

    def test_ignored_module_is_skipped_without_touching_weight(self):
        # ignore_modules matches every name -> parse returns None -> the loop
        # `continue`s. The weight must remain (no pop, no .cuda(), no error).
        cfg = QuantizationConfig(
            weight_quantize_algo="weight_only_int8",
            ignore_modules=[".*"],
        )
        w = object()
        state_dict = {"layer.0.weight": w}
        result = convert_to_quantize_state_dict(
            state_dict, ["layer.0"], cfg, "float16"
        )
        self.assertIs(result["layer.0.weight"], w)
        self.assertEqual(list(result.keys()), ["layer.0.weight"])


class ConvertToQloraStateDictImportGuardTest(unittest.TestCase):
    """convert_to_qlora_state_dict: missing PaddleSlim dependency contract."""

    def test_raises_import_error_when_qlora_backend_absent(self):
        cfg = QuantizationConfig(
            weight_quantize_algo="nf4", qlora_weight_double_quant=False
        )
        state_dict = {"layer.0.weight": object()}
        # Patch only the (external) qlora collaborator to its unavailable state
        # and assert the module raises the specific ImportError contract. The
        # tested logic -- the `is None` guard -- is preserved, not mocked.
        with mock.patch(f"{_MODULE}.qlora_weight_quantize", None):
            with self.assertRaises(ImportError):
                convert_to_qlora_state_dict(
                    state_dict, "layer.0", cfg, "float16", "nf4"
                )


if __name__ == "__main__":
    unittest.main()
