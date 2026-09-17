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
#
# Scope: pins the whole-model config-field protocol normalization -- the single
# place that resolves the model-declared AOA name attributes
# (``build_aoa_context``). An unmigrated model (without an explicit mapping)
# inherits the shared ERNIE mapping; an external model overrides it via that
# attribute.
import dataclasses
import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

from paddle.distributed.flex_checkpoint.aoa.generation import AOAContext

from paddlefleet.models.gpt.aoa_generator import (
    DEFAULT_CHECKPOINT_NAME_MAPPING,
    DEFAULT_CHECKPOINT_NAME_PREFIX,
    build_aoa_context,
)


class _Cfg:
    """Bare config stand-in; only the attributes a test sets are present."""


class _FakeModel:
    """Duck-types the attributes ``build_aoa_context`` reads.

    ``_pipeline_name_mapping`` is pre-populated (non-None) so the idempotent
    ``_set_pipeline_name_mapping`` side effect is skipped in the happy path.
    ``_model_name_prefix()`` stands in for the live model's single-name root,
    which is where the context takes it from; the value is held in a separately
    named attribute so it does not shadow the method.
    """

    def __init__(self, pp_to_single_mapping=None, model_name_prefix="model"):
        self._pipeline_name_mapping = {}
        self._pp_to_single_mapping = pp_to_single_mapping or {}
        self._model_name_prefix_value = model_name_prefix

    def _set_pipeline_name_mapping(self):
        self._pipeline_name_mapping = {}

    def _model_name_prefix(self):
        return self._model_name_prefix_value


class TestBuildAOAContextDefaults(unittest.TestCase):
    def test_unmigrated_config_uses_ernie_defaults(self):
        model = _FakeModel(pp_to_single_mapping={"s": "s"})
        ctx = build_aoa_context(model, _Cfg())
        self.assertIsInstance(ctx, AOAContext)
        self.assertEqual(
            ctx.checkpoint_name_prefix, DEFAULT_CHECKPOINT_NAME_PREFIX
        )
        self.assertEqual(
            dict(ctx.checkpoint_name_mapping), DEFAULT_CHECKPOINT_NAME_MAPPING
        )
        self.assertEqual(ctx.model_name_prefix, "model")
        self.assertEqual(dict(ctx.pp_to_single_mapping), {"s": "s"})

    def test_mapping_is_copied_not_aliased(self):
        ctx = build_aoa_context(_FakeModel(), _Cfg())
        self.assertIsNot(
            ctx.checkpoint_name_mapping, DEFAULT_CHECKPOINT_NAME_MAPPING
        )

    def test_explicit_empty_checkpoint_mapping_overrides_default(self):
        cfg = _Cfg()
        cfg.aoa_checkpoint_name_mapping = {}
        ctx = build_aoa_context(_FakeModel(), cfg)
        self.assertEqual(dict(ctx.checkpoint_name_mapping), {})

    def test_explicit_empty_checkpoint_prefix_is_preserved(self):
        # A checkpoint rooted at the top level (DeepSeek V4) declares "" and
        # must not silently fall back to the Ernie-series default.
        cfg = _Cfg()
        cfg.aoa_checkpoint_name_prefix = ""
        ctx = build_aoa_context(_FakeModel(), cfg)
        self.assertEqual(ctx.checkpoint_name_prefix, "")

    def test_explicit_none_falls_back_to_default(self):
        cfg = _Cfg()
        cfg.aoa_checkpoint_name_prefix = None
        ctx = build_aoa_context(_FakeModel(), cfg)
        self.assertEqual(
            ctx.checkpoint_name_prefix, DEFAULT_CHECKPOINT_NAME_PREFIX
        )


class TestBuildAOAContextOverrides(unittest.TestCase):
    def test_name_overrides_propagate(self):
        cfg = _Cfg()
        cfg.aoa_checkpoint_name_prefix = "hf"
        cfg.aoa_checkpoint_name_mapping = {"model.a.weight": "hf.b.weight"}
        ctx = build_aoa_context(_FakeModel(), cfg)
        self.assertEqual(ctx.checkpoint_name_prefix, "hf")
        self.assertEqual(
            dict(ctx.checkpoint_name_mapping), {"model.a.weight": "hf.b.weight"}
        )

    def test_model_name_prefix_comes_from_the_live_model(self):
        # The model single-name root comes from the live model, not the config:
        # it must stay the same value that names the pipeline layers. A model
        # rooted elsewhere cannot carry the ERNIE mapping, whose keys are
        # absolute names under ``model``, so it declares its own.
        cfg = _Cfg()
        cfg.aoa_checkpoint_name_mapping = {}
        ctx = build_aoa_context(_FakeModel(model_name_prefix="root"), cfg)
        self.assertEqual(ctx.model_name_prefix, "root")

    def test_context_is_frozen(self):
        ctx = build_aoa_context(_FakeModel(), _Cfg())
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ctx.checkpoint_name_prefix = "mut"

    def test_mapping_is_validated_against_the_live_model_root(self):
        # A key is matched against a whole single name, so one that does not
        # carry the model root could never match; the entry rejects it instead
        # of degrading to the identity fallback.
        cfg = _Cfg()
        cfg.aoa_checkpoint_name_mapping = {"other.a.weight": "hf.a.weight"}
        with self.assertRaises(ValueError):
            build_aoa_context(_FakeModel(), cfg)

    def test_none_pipeline_mapping_triggers_side_effect(self):
        model = _FakeModel()
        model._pipeline_name_mapping = None
        ctx = build_aoa_context(model, _Cfg())
        self.assertIsNotNone(model._pipeline_name_mapping)
        self.assertIsInstance(ctx, AOAContext)


if __name__ == "__main__":
    unittest.main()
