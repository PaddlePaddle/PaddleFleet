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

"""Behavior tests for paddlefleet.cli.utils.mllm_utils.

The module builds a registry of multimodal model key groups (vision / aligner
/ llm) and uses it to (a) drop LoRA target modules and (b) freeze parameters
belonging to modules named by a ``freeze_config`` string.

Two matching rules differ and are exercised on purpose:
- get_multimodel_lora_target_modules uses ``prefix in name`` (substring),
  scanning prefixes longest-first so the more specific group wins.
- freeze_model_parameters uses a ``^(...)`` anchored regex, so only names that
  *start* with a registered prefix are affected.

All expected values are hand-derived from these two rules, not read back from
the production code. Importing paddlefleet pulls in paddle transitively, which
is absent in this environment, so the whole module is skipped honestly when the
import fails.
"""

import unittest
from types import SimpleNamespace

try:
    from paddlefleet.cli.utils import mllm_utils
    from paddlefleet.cli.utils.mllm_utils import (
        MLLMModelMapping,
        MultiModelKeys,
        freeze_model_parameters,
        get_multimodel_lora_target_modules,
        get_multimodel_target_modules,
        register_multimodel_keys,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # local env has no paddle; skip honestly
    mllm_utils = None
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    "paddlefleet.cli.utils.mllm_utils import failed "
    f"(paddle not installed in this env): {_IMPORT_ERROR}"
)


class _FakeParam:
    """Stand-in for a model parameter; only ``stop_gradient`` is consumed."""

    def __init__(self):
        # Sentinel: freeze_model_parameters must explicitly set a bool on every
        # visited parameter (True to freeze, False otherwise). A leftover None
        # means the parameter was never touched.
        self.stop_gradient = None


class _FakeModel:
    """Model exposing config.model_type and named_parameters(), no paddle."""

    def __init__(self, model_type, named):
        self.config = SimpleNamespace(model_type=model_type)
        self._named = named
        self.named_parameters_calls = 0

    def named_parameters(self):
        self.named_parameters_calls += 1
        return list(self._named)


@unittest.skipUnless(mllm_utils is not None, _SKIP_REASON)
class MultiModelKeysPostInitTest(unittest.TestCase):
    """__post_init__ normalizes llm/aligner/vision to lists."""

    def test_string_becomes_single_element_list(self):
        keys = MultiModelKeys(model_dtype="m", llm="model.language_model")
        self.assertEqual(keys.llm, ["model.language_model"])

    def test_none_becomes_empty_list(self):
        keys = MultiModelKeys(model_dtype="m", vision=None)
        self.assertEqual(keys.vision, [])

    def test_list_content_preserved_in_order(self):
        keys = MultiModelKeys(
            model_dtype="m", llm=["model.language_model", "lm_head"]
        )
        self.assertEqual(keys.llm, ["model.language_model", "lm_head"])

    def test_unspecified_group_defaults_to_empty_list(self):
        keys = MultiModelKeys(model_dtype="m")
        self.assertEqual(keys.llm, [])
        self.assertEqual(keys.aligner, [])
        self.assertEqual(keys.vision, [])

    def test_each_group_normalized_independently(self):
        keys = MultiModelKeys(
            model_dtype="m",
            llm="only.llm",
            aligner=["a.one", "a.two"],
            vision=None,
        )
        self.assertEqual(keys.llm, ["only.llm"])
        self.assertEqual(keys.aligner, ["a.one", "a.two"])
        self.assertEqual(keys.vision, [])
        # Inherited scalar field passes through unchanged.
        self.assertEqual(keys.model_dtype, "m")


@unittest.skipUnless(mllm_utils is not None, _SKIP_REASON)
class RegisterAndLookupTest(unittest.TestCase):
    """register_multimodel_keys / get_multimodel_target_modules contract."""

    def _cleanup_key(self, name):
        # Registry is a module global; drop test keys even on assertion failure.
        self.addCleanup(mllm_utils._MULTIMODEL_KEY_REGISTRY.pop, name, None)

    def test_falsy_model_type_returns_none(self):
        self.assertIsNone(get_multimodel_target_modules(None))
        self.assertIsNone(get_multimodel_target_modules(""))

    def test_unregistered_model_type_returns_none(self):
        self.assertIsNone(
            get_multimodel_target_modules("no_such_model_qzx_987")
        )

    def test_register_then_lookup_returns_same_object(self):
        name = "unit_test_mllm_key_alpha"
        self._cleanup_key(name)
        keys = MultiModelKeys(model_dtype=name, llm=["core"], vision=["eye"])
        register_multimodel_keys(keys)
        got = get_multimodel_target_modules(name)
        self.assertIs(got, keys)
        self.assertEqual(got.llm, ["core"])
        self.assertEqual(got.vision, ["eye"])

    def test_duplicate_without_exist_ok_raises_and_keeps_original(self):
        name = "unit_test_mllm_key_beta"
        self._cleanup_key(name)
        first = MultiModelKeys(model_dtype=name, llm=["first"])
        register_multimodel_keys(first)
        clash = MultiModelKeys(model_dtype=name, llm=["second"])
        with self.assertRaises(ValueError) as ctx:
            register_multimodel_keys(clash, exist_ok=False)
        self.assertIn("already been registered", str(ctx.exception))
        # Original registration must survive the rejected duplicate.
        self.assertIs(get_multimodel_target_modules(name), first)

    def test_exist_ok_overwrites_previous_registration(self):
        name = "unit_test_mllm_key_gamma"
        self._cleanup_key(name)
        first = MultiModelKeys(model_dtype=name, llm=["first"])
        second = MultiModelKeys(model_dtype=name, llm=["second"])
        register_multimodel_keys(first)
        register_multimodel_keys(second, exist_ok=True)
        got = get_multimodel_target_modules(name)
        self.assertIs(got, second)
        self.assertEqual(got.llm, ["second"])

    def test_builtin_qwen2_5_vl_registered_with_expected_groups(self):
        # Shipped registration; content, not just presence, is checked.
        keys = get_multimodel_target_modules(MLLMModelMapping.qwen2_5_vl)
        self.assertIsInstance(keys, MultiModelKeys)
        self.assertEqual(keys.vision, ["model.visual"])
        self.assertEqual(keys.aligner, ["model.visual.merger"])
        self.assertEqual(keys.llm, ["model.language_model", "lm_head"])


@unittest.skipUnless(mllm_utils is not None, _SKIP_REASON)
class LoraTargetModuleFilterTest(unittest.TestCase):
    """get_multimodel_lora_target_modules: substring, longest-first, order."""

    # qwen2_5_vl groups:
    #   vision  = model.visual
    #   aligner = model.visual.merger   (more specific than vision)
    #   llm     = model.language_model, lm_head
    # Prefixes are scanned longest-first, matched by substring (`prefix in tm`).
    TARGETS = [
        "model.visual.patch_embed.proj.weight",  # matches model.visual -> vision
        "model.visual.merger.mlp.weight",  # matches merger -> aligner (longer wins)
        "model.language_model.layers.0.self_attn.q_proj",  # -> llm
        "lm_head.weight",  # -> llm
        "encoder.model.visual.block",  # substring model.visual -> vision
    ]

    def _model(self, model_type):
        return SimpleNamespace(config=SimpleNamespace(model_type=model_type))

    def test_freeze_vision_removes_only_vision_targets(self):
        model = self._model("qwen2_5_vl")
        result = get_multimodel_lora_target_modules(
            model, list(self.TARGETS), "freeze_vision"
        )
        # vision entries (#0, #4) dropped; the merger stays because it resolves
        # to aligner via the longer prefix, not vision.
        self.assertEqual(
            result,
            [
                "model.visual.merger.mlp.weight",
                "model.language_model.layers.0.self_attn.q_proj",
                "lm_head.weight",
            ],
        )

    def test_freeze_llm_removes_only_llm_targets(self):
        model = self._model("qwen2_5_vl")
        result = get_multimodel_lora_target_modules(
            model, list(self.TARGETS), "freeze_llm"
        )
        self.assertEqual(
            result,
            [
                "model.visual.patch_embed.proj.weight",
                "model.visual.merger.mlp.weight",
                "encoder.model.visual.block",
            ],
        )

    def test_empty_freeze_config_keeps_all_in_order(self):
        model = self._model("qwen2_5_vl")
        result = get_multimodel_lora_target_modules(
            model, list(self.TARGETS), ""
        )
        self.assertEqual(result, self.TARGETS)

    def test_unregistered_model_returns_input_unchanged(self):
        model = self._model("totally_unknown_model")
        original = list(self.TARGETS)
        result = get_multimodel_lora_target_modules(
            model, original, "freeze_vision"
        )
        # Contract: same object returned, nothing filtered.
        self.assertIs(result, original)
        self.assertEqual(result, self.TARGETS)


@unittest.skipUnless(mllm_utils is not None, _SKIP_REASON)
class FreezeModelParametersTest(unittest.TestCase):
    """freeze_model_parameters: anchored prefix match sets stop_gradient."""

    def _run(self, model_type, names, freeze_config):
        params = [_FakeParam() for _ in names]
        model = _FakeModel(model_type, list(zip(names, params)))
        freeze_model_parameters(model, freeze_config)
        return model, params

    def test_freeze_vision_freezes_only_anchored_vision_names(self):
        names = [
            "model.visual.patch_embed.proj.weight",  # ^model.visual -> vision
            "model.visual.merger.mlp.0.weight",  # ^merger -> aligner (longer)
            "model.language_model.layers.0.self_attn.q_proj.weight",  # llm
            "lm_head.weight",  # llm
            "backbone.model.visual.block.weight",  # not anchored -> no match
        ]
        _, params = self._run("qwen2_5_vl", names, "freeze_vision")
        flags = [p.stop_gradient for p in params]
        # Only the parameter whose name *starts* with model.visual (and resolves
        # to the vision group) is frozen. The merger resolves to aligner; the
        # backbone.* name is not anchored so it is explicitly left trainable.
        self.assertEqual(flags, [True, False, False, False, False])

    def test_freeze_llm_freezes_language_model_and_lm_head(self):
        names = [
            "model.language_model.layers.0.mlp.gate_proj.weight",
            "lm_head.weight",
            "model.visual.patch_embed.proj.weight",
            "model.visual.merger.mlp.0.weight",
        ]
        _, params = self._run("qwen2_5_vl", names, "freeze_llm")
        flags = [p.stop_gradient for p in params]
        self.assertEqual(flags, [True, True, False, False])

    def test_non_frozen_params_are_actively_set_trainable(self):
        # Every visited parameter must end with an explicit bool, even those
        # not frozen (rules out an implementation that only touches matches).
        names = [
            "model.visual.patch_embed.weight",
            "model.language_model.layers.0.mlp.weight",
        ]
        _, params = self._run("qwen2_5_vl", names, "freeze_vision")
        self.assertEqual(params[0].stop_gradient, True)
        self.assertEqual(params[1].stop_gradient, False)
        self.assertNotIn(None, [p.stop_gradient for p in params])

    def test_model_without_config_returns_before_touching_params(self):
        model = SimpleNamespace()  # no .config attribute at all
        model.named_parameters_called = False

        def named_parameters():
            model.named_parameters_called = True
            return []

        model.named_parameters = named_parameters
        # Must short-circuit on the missing config and never enumerate params.
        freeze_model_parameters(model, "freeze_vision")
        self.assertFalse(model.named_parameters_called)

    def test_model_without_model_type_returns_early(self):
        model = SimpleNamespace(config=SimpleNamespace())  # no model_type
        param = _FakeParam()
        model.named_parameters = lambda: [("model.visual.x", param)]
        freeze_model_parameters(model, "freeze_vision")
        # Guard trips before iterating, so the parameter is left untouched.
        self.assertIsNone(param.stop_gradient)

    def test_unregistered_model_type_leaves_params_untouched(self):
        param = _FakeParam()
        model = _FakeModel("no_such_model_qzx_987", [("model.visual.x", param)])
        freeze_model_parameters(model, "freeze_vision")
        self.assertIsNone(param.stop_gradient)
        self.assertEqual(model.named_parameters_calls, 0)


if __name__ == "__main__":
    unittest.main()
