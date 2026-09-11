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
# place that resolves the model-declared AOA name / dtype / layout attributes
# (``build_aoa_context`` via ``_protocol_value``) and the MTP checkpoint-prefix
# attributes (``MTPCheckpointPrefixSpec`` via
# ``resolve_mtp_checkpoint_prefix_spec``). An unmigrated model (without an
# explicit mapping) inherits the shared ERNIE mapping; an external model
# overrides it via that attribute. Also carries a source-level lint pinning
# the direction-independence contract (no ``direction`` / ``reverse`` param, a
# ``_fwd_`` / ``_inv_`` helper split with no cross-direction calls and no
# ``aoa_config_reverse`` call).
import dataclasses
import os
import sys
import unittest
from types import SimpleNamespace

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
    DEFAULT_GATE_CHECKPOINT_LAYOUT,
    DEFAULT_MLP_GATE_UP_FUSED,
    DEFAULT_MODEL_NAME_PREFIX,
    DEFAULT_MOE_EXPERT_CHECKPOINT_LAYOUT,
    DEFAULT_QKV_CHECKPOINT_PREFUSED,
    FleetAOAContext,
    MTPCheckpointPrefixSpec,
    build_aoa_context,
    resolve_mtp_checkpoint_prefix_spec,
)
from paddlefleet.transformer.transformer_config import TransformerConfig


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

    def __init__(
        self,
        pp_to_single_mapping=None,
        model_name_prefix=DEFAULT_MODEL_NAME_PREFIX,
    ):
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
        self.assertIsInstance(ctx, FleetAOAContext)
        self.assertEqual(
            ctx.checkpoint_name_prefix, DEFAULT_CHECKPOINT_NAME_PREFIX
        )
        self.assertEqual(
            dict(ctx.checkpoint_name_mapping), DEFAULT_CHECKPOINT_NAME_MAPPING
        )
        self.assertEqual(dict(ctx.dtype_cast_rules), {})
        self.assertEqual(ctx.model_name_prefix, DEFAULT_MODEL_NAME_PREFIX)
        self.assertEqual(dict(ctx.pp_to_single_mapping), {"s": "s"})

    def test_layout_defaults(self):
        ctx = build_aoa_context(_FakeModel(), _Cfg())
        self.assertEqual(
            ctx.gate_checkpoint_layout, DEFAULT_GATE_CHECKPOINT_LAYOUT
        )
        self.assertEqual(
            ctx.qkv_checkpoint_prefused, DEFAULT_QKV_CHECKPOINT_PREFUSED
        )
        self.assertEqual(
            ctx.moe_expert_checkpoint_layout,
            DEFAULT_MOE_EXPERT_CHECKPOINT_LAYOUT,
        )
        self.assertEqual(ctx.mlp_gate_up_fused, DEFAULT_MLP_GATE_UP_FUSED)
        self.assertFalse(ctx.mapping_proj_checkpoint_transposed)

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
        cfg.aoa_dtype_cast_rules = None
        cfg.aoa_qkv_checkpoint_prefused = None
        ctx = build_aoa_context(_FakeModel(), cfg)
        self.assertEqual(
            ctx.checkpoint_name_prefix, DEFAULT_CHECKPOINT_NAME_PREFIX
        )
        self.assertEqual(dict(ctx.dtype_cast_rules), {})
        self.assertEqual(
            ctx.qkv_checkpoint_prefused, DEFAULT_QKV_CHECKPOINT_PREFUSED
        )


class TestBuildAOAContextOverrides(unittest.TestCase):
    def test_name_and_dtype_overrides_propagate(self):
        cfg = _Cfg()
        cfg.aoa_checkpoint_name_prefix = "hf"
        cfg.aoa_checkpoint_name_mapping = {"a.weight": "b.weight"}
        cfg.aoa_dtype_cast_rules = {
            "x": {"checkpoint_dtype": "bfloat16", "model_dtype": "float32"}
        }
        ctx = build_aoa_context(_FakeModel(), cfg)
        self.assertEqual(ctx.checkpoint_name_prefix, "hf")
        self.assertEqual(
            dict(ctx.checkpoint_name_mapping), {"a.weight": "b.weight"}
        )
        self.assertEqual(
            dict(ctx.dtype_cast_rules),
            {"x": {"checkpoint_dtype": "bfloat16", "model_dtype": "float32"}},
        )

    def test_layout_overrides_propagate(self):
        cfg = _Cfg()
        cfg.aoa_gate_checkpoint_layout = "interleaved"
        cfg.aoa_qkv_checkpoint_prefused = True
        cfg.aoa_moe_expert_checkpoint_layout = "packed"
        cfg.aoa_mapping_proj_checkpoint_transposed = True
        ctx = build_aoa_context(_FakeModel(), cfg)
        self.assertEqual(ctx.gate_checkpoint_layout, "interleaved")
        self.assertTrue(ctx.qkv_checkpoint_prefused)
        self.assertEqual(ctx.moe_expert_checkpoint_layout, "packed")
        self.assertTrue(ctx.mapping_proj_checkpoint_transposed)

    def test_falsy_mlp_gate_up_fused_is_preserved(self):
        # Qwen3-VL declares False; a truthiness-based default would swallow it.
        cfg = _Cfg()
        cfg.aoa_mlp_gate_up_fused = False
        ctx = build_aoa_context(_FakeModel(), cfg)
        self.assertFalse(ctx.mlp_gate_up_fused)

    def test_model_name_prefix_comes_from_the_live_model(self):
        # The model single-name root comes from the live model, not the config:
        # it must stay the same value that names the pipeline layers.
        ctx = build_aoa_context(_FakeModel(model_name_prefix="root"), _Cfg())
        self.assertEqual(ctx.model_name_prefix, "root")

    def test_context_is_frozen(self):
        ctx = build_aoa_context(_FakeModel(), _Cfg())
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ctx.checkpoint_name_prefix = "mut"

    def test_none_pipeline_mapping_triggers_side_effect(self):
        model = _FakeModel()
        model._pipeline_name_mapping = None
        ctx = build_aoa_context(model, _Cfg())
        self.assertIsNotNone(model._pipeline_name_mapping)
        self.assertIsInstance(ctx, AOAContext)


class TestMTPCheckpointPrefixSpec(unittest.TestCase):
    def test_defaults(self):
        spec = resolve_mtp_checkpoint_prefix_spec(_Cfg())
        self.assertEqual(spec.checkpoint_prefix, "layers.$LAYER_ID")
        self.assertEqual(spec.transformer_checkpoint_prefix, "layers.$LAYER_ID")
        self.assertFalse(spec.is_absolute)

    def test_transformer_inherits_own_prefix(self):
        cfg = _Cfg()
        cfg.aoa_mtp_checkpoint_prefix = "mtp.$LAYER_ID"
        spec = resolve_mtp_checkpoint_prefix_spec(cfg)
        self.assertEqual(spec.checkpoint_prefix, "mtp.$LAYER_ID")
        self.assertEqual(spec.transformer_checkpoint_prefix, "mtp.$LAYER_ID")

    def test_transformer_prefix_override_independent(self):
        cfg = _Cfg()
        cfg.aoa_mtp_checkpoint_prefix = "mtp.$LAYER_ID"
        cfg.aoa_mtp_transformer_checkpoint_prefix = "mtp.$LAYER_ID.tf"
        spec = resolve_mtp_checkpoint_prefix_spec(cfg)
        self.assertEqual(spec.checkpoint_prefix, "mtp.$LAYER_ID")
        self.assertEqual(spec.transformer_checkpoint_prefix, "mtp.$LAYER_ID.tf")

    def test_absolute_flag(self):
        cfg = _Cfg()
        cfg.aoa_mtp_checkpoint_prefix_absolute = True
        spec = resolve_mtp_checkpoint_prefix_spec(cfg)
        self.assertTrue(spec.is_absolute)

    def test_bare_spec_defaults(self):
        spec = MTPCheckpointPrefixSpec()
        self.assertEqual(spec.checkpoint_prefix, "layers.$LAYER_ID")
        self.assertEqual(spec.transformer_checkpoint_prefix, "layers.$LAYER_ID")
        self.assertFalse(spec.is_absolute)

    def test_explicit_empty_mtp_prefix_is_preserved(self):
        # A top-level MTP checkpoint declares "" and must not silently fall
        # back to the layers.$LAYER_ID default; the inner transformer prefix
        # then inherits that same "".
        cfg = _Cfg()
        cfg.aoa_mtp_checkpoint_prefix = ""
        spec = resolve_mtp_checkpoint_prefix_spec(cfg)
        self.assertEqual(spec.checkpoint_prefix, "")
        self.assertEqual(spec.transformer_checkpoint_prefix, "")


class TestAoaModularNotConfigInjectable(unittest.TestCase):
    """Pins that ``aoa_modular`` is a code-level marker, not a config field.

    The marker is a ``ClassVar`` a migrated provider flips in its class body;
    it must never be reachable from a yaml / ``config.json``. Two independent
    surfaces enforce this and this test pins both: the ``ClassVar`` keeps it
    out of the dataclass fields (so it is neither constructor-settable nor
    instance-shadowed), and ``_process_attribute`` rejects the key outright on
    the ``from_config`` load path.
    """

    def test_classvar_default_is_false(self):
        self.assertFalse(TransformerConfig.aoa_modular)

    def test_aoa_modular_is_not_a_dataclass_field(self):
        field_names = {f.name for f in dataclasses.fields(TransformerConfig)}
        self.assertNotIn("aoa_modular", field_names)

    def test_from_config_rejects_aoa_modular_key(self):
        # Rejected by the key itself, independent of the value carried.
        for value in (True, False):
            with self.assertRaises(ValueError):
                TransformerConfig.from_config(
                    SimpleNamespace(aoa_modular=value)
                )

    def test_reject_is_narrow(self):
        # An unrelated attribute still flows through the normal setattr path.
        inst = object.__new__(TransformerConfig)
        inst._process_attribute("some_unrelated_attr", 123)
        self.assertEqual(inst.some_unrelated_attr, 123)


if __name__ == "__main__":
    unittest.main()
