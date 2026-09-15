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

"""Behavior tests for the DeepSeek-V3 pretrain configuration.

Module under test:
``paddlefleet.cli.train.deepseek_v3_pretrain.configuration.DeepseekV2FastConfig``
(``model_type == "deepseek_v3"``), a ``PretrainedConfig`` subclass. This is the
"configuration and runtime infrastructure" layer: the value of each field is a
contract that downstream model construction consumes.

What these tests actually verify (not set-then-readback):

* The documented default values that callers rely on, hand-derived by reading
  the constructor signature. Defaults are a real contract; a silent change to
  one would flip model architecture.
* Derived / forced behaviour that the subclass computes rather than merely
  stores: ``num_key_value_heads`` back-fills from ``num_attention_heads`` only
  when ``None``; ``use_fp8`` is pinned to ``False`` even when the caller passes
  ``True``; ``tie_word_embeddings`` overrides the base class default of ``True``.
* Cross-boundary propagation: values handed to ``super().__init__`` (the special
  token ids) end up on the base-class attributes, and the base-class generation
  defaults are installed, proving the parent constructor genuinely ran.
* Serialization consumption: ``to_dict`` reads the stored fields and injects the
  class-level ``model_type``; ``from_dict`` reconstructs an equivalent config.
  These go through two independent real code paths, so a dropped field is
  observable.
* A real production defect (see ``expectedFailure`` below): the default
  ``topk_method`` value is misspelled and matches no branch in the MoE gate
  dispatcher.

Precision/boolean discipline: fp8 flags are checked with identity (``is False``
/ ``is True``) so a truthy string could never masquerade as the boolean default.

These tests run on CPU. Importing the production module pulls the
``paddlefleet`` package, which requires Paddle; when that dependency is absent
the import raises ``ImportError`` and every test skips (recorded, not silently
passed). No production code is modified by this file.
"""

import unittest

try:
    from paddlefleet.cli.train.deepseek_v3_pretrain.configuration import (
        DeepseekV2FastConfig,
    )
    from paddlefleet.transformers.configuration_utils import PretrainedConfig

    _IMPORT_ERROR = None
except ImportError as exc:  # Paddle backend not installed.
    DeepseekV2FastConfig = None
    PretrainedConfig = None
    _IMPORT_ERROR = exc


# Hand-derived from reading ``moe_gate.py`` (DeepseekV2MoEGate.topkgating and
# topkgating_nodrop dispatch on these literals; see lines ~487 and ~581, and
# the branch in modeling.py). Any config default outside this set never selects
# a top-k routing branch in the gate.
_GATE_RECOGNIZED_TOPK_METHODS = frozenset(
    {"greedy", "group_limited_greedy", "noaux_tc"}
)


class _ConfigTestBase(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(
                "paddlefleet import failed (dependency unavailable): "
                "{!r}".format(_IMPORT_ERROR)
            )


class TestClassLevelContract(_ConfigTestBase):
    """Class attributes select the architecture family and inference keys."""

    def test_model_type_identifies_deepseek_v3(self):
        # The registry/dispatch key is a hard string contract, not a free label.
        self.assertEqual(DeepseekV2FastConfig.model_type, "deepseek_v3")

    def test_keys_to_ignore_at_inference(self):
        self.assertEqual(
            DeepseekV2FastConfig.keys_to_ignore_at_inference,
            ["past_key_values"],
        )


class TestDefaults(_ConfigTestBase):
    """Documented constructor defaults are a consumed contract.

    Expectations are hand-derived from the constructor signature; a silent
    change to any of these would alter the instantiated model architecture.
    """

    def test_architecture_dimension_defaults(self):
        cfg = DeepseekV2FastConfig()
        self.assertEqual(cfg.vocab_size, 102400)
        self.assertEqual(cfg.hidden_size, 4096)
        self.assertEqual(cfg.intermediate_size, 11008)
        self.assertEqual(cfg.moe_intermediate_size, 1407)
        # Note: the class docstring claims 32 hidden layers, but the actual
        # constructor default is 30 -- behaviour, not the doc, is locked here.
        self.assertEqual(cfg.num_hidden_layers, 30)
        self.assertEqual(cfg.num_attention_heads, 32)
        self.assertEqual(cfg.num_key_value_heads, 32)

    def test_attention_head_dim_defaults(self):
        cfg = DeepseekV2FastConfig()
        self.assertEqual(cfg.kv_lora_rank, 512)
        self.assertEqual(cfg.q_lora_rank, 1536)
        self.assertEqual(cfg.qk_rope_head_dim, 64)
        self.assertEqual(cfg.qk_nope_head_dim, 128)
        self.assertEqual(cfg.v_head_dim, 128)

    def test_moe_and_sparse_defaults(self):
        cfg = DeepseekV2FastConfig()
        # Dense-by-default: expert counts are None until explicitly configured.
        self.assertIsNone(cfg.n_shared_experts)
        self.assertIsNone(cfg.n_routed_experts)
        self.assertIsNone(cfg.n_group)
        self.assertIsNone(cfg.topk_group)
        self.assertIsNone(cfg.num_experts_per_tok)
        self.assertEqual(cfg.ep_size, 1)
        self.assertEqual(cfg.moe_layer_freq, 1)
        self.assertEqual(cfg.first_k_dense_replace, 0)
        self.assertEqual(cfg.routed_scaling_factor, 1.0)
        self.assertEqual(cfg.scoring_func, "softmax")
        # norm_topk_prob defaults off; seq_aux defaults on.
        self.assertIs(cfg.norm_topk_prob, False)
        self.assertIs(cfg.seq_aux, True)

    def test_scalar_and_token_defaults(self):
        cfg = DeepseekV2FastConfig()
        self.assertEqual(cfg.hidden_act, "silu")
        self.assertEqual(cfg.max_position_embeddings, 2048)
        self.assertEqual(cfg.seq_length, 32768)
        self.assertEqual(cfg.rms_norm_eps, 1e-6)
        self.assertEqual(cfg.rope_theta, 10000.0)
        self.assertIsNone(cfg.rope_scaling)
        self.assertEqual(cfg.attention_dropout, 0.0)
        self.assertIs(cfg.attention_bias, False)
        self.assertIsNone(cfg.max_sequence_length)
        self.assertEqual(cfg.fa_version, 3)


class TestDerivedAndForcedFields(_ConfigTestBase):
    """Fields the subclass *computes* or *pins*, not merely stores."""

    def test_num_key_value_heads_backfills_only_when_none(self):
        # When None, the constructor derives it from num_attention_heads...
        derived = DeepseekV2FastConfig(
            num_attention_heads=7, num_key_value_heads=None
        )
        self.assertEqual(derived.num_key_value_heads, 7)
        # ...but an explicit value must survive untouched (no clobbering).
        explicit = DeepseekV2FastConfig(
            num_attention_heads=7, num_key_value_heads=3
        )
        self.assertEqual(explicit.num_key_value_heads, 3)

    def test_use_fp8_is_pinned_false_even_if_requested_true(self):
        # ``use_fp8`` is hard-set to False after super().__init__, so a caller
        # cannot turn it on through the constructor. Identity check rejects a
        # truthy non-bool sneaking through.
        cfg = DeepseekV2FastConfig(use_fp8=True)
        self.assertIs(cfg.use_fp8, False)
        self.assertIs(DeepseekV2FastConfig().use_fp8, False)

    def test_tie_word_embeddings_overrides_base_default(self):
        # The base PretrainedConfig defaults tie_word_embeddings to True; this
        # subclass passes False to super() by default, so the override must win.
        self.assertIs(PretrainedConfig().tie_word_embeddings, True)
        self.assertIs(DeepseekV2FastConfig().tie_word_embeddings, False)
        # And an explicit True still propagates through super().
        self.assertIs(
            DeepseekV2FastConfig(tie_word_embeddings=True).tie_word_embeddings,
            True,
        )

    def test_fp8_gemm_flags_are_boolean_defaults(self):
        cfg = DeepseekV2FastConfig()
        self.assertIs(cfg.dsv3_use_fp8_gemm, True)
        self.assertIs(cfg.dsv3_use_atten_recompute, True)
        self.assertIs(cfg.dsv3_use_fp8_dispatch, True)
        self.assertIs(cfg.use_ds_gemm, False)


class TestSuperConstructorPropagation(_ConfigTestBase):
    """Values handed to ``super().__init__`` reach base-class attributes."""

    def test_special_token_defaults_land_on_base_attributes(self):
        # bos/eos/pad are forwarded into super() and stored by the base class;
        # these subclass defaults differ from the base defaults (None), proving
        # the values crossed the inheritance boundary rather than defaulting.
        cfg = DeepseekV2FastConfig()
        self.assertEqual(cfg.bos_token_id, 100000)
        self.assertEqual(cfg.eos_token_id, 100001)
        self.assertIsNone(cfg.pad_token_id)

    def test_custom_special_tokens_propagate(self):
        cfg = DeepseekV2FastConfig(
            bos_token_id=5, eos_token_id=6, pad_token_id=9
        )
        self.assertEqual(cfg.bos_token_id, 5)
        self.assertEqual(cfg.eos_token_id, 6)
        self.assertEqual(cfg.pad_token_id, 9)

    def test_base_generation_defaults_are_installed(self):
        # These attributes exist only because super().__init__ ran and applied
        # the base generation defaults; the subclass never sets them.
        cfg = DeepseekV2FastConfig()
        self.assertEqual(cfg.max_length, 20)
        self.assertEqual(cfg.num_beams, 1)


class TestSerializationRoundTrip(_ConfigTestBase):
    """``to_dict`` consumes stored fields; ``from_dict`` reconstructs them."""

    def test_to_dict_emits_fields_and_class_model_type(self):
        cfg = DeepseekV2FastConfig(
            hidden_size=1234, n_routed_experts=17, first_k_dense_replace=3
        )
        d = cfg.to_dict()
        # model_type comes from the class attribute, not the instance __dict__.
        self.assertEqual(d["model_type"], "deepseek_v3")
        self.assertEqual(d["hidden_size"], 1234)
        self.assertEqual(d["n_routed_experts"], 17)
        self.assertEqual(d["first_k_dense_replace"], 3)
        # The serializer stamps its own version key.
        self.assertIn("paddlefleet_version", d)

    def test_from_dict_reconstructs_distinctive_fields(self):
        original = DeepseekV2FastConfig(
            vocab_size=32000,
            hidden_size=2048,
            num_hidden_layers=12,
            n_routed_experts=64,
            topk_method="group_limited_greedy",
            n_group=8,
            topk_group=4,
            num_experts_per_tok=6,
            norm_topk_prob=True,
            seq_aux=False,
        )
        rebuilt = DeepseekV2FastConfig.from_dict(original.to_dict())
        for field_name in (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "n_routed_experts",
            "topk_method",
            "n_group",
            "topk_group",
            "num_experts_per_tok",
        ):
            self.assertEqual(
                getattr(rebuilt, field_name),
                getattr(original, field_name),
                "field {!r} was lost across to_dict/from_dict".format(
                    field_name
                ),
            )
        # Booleans must round-trip as booleans, not stringified truthy values.
        self.assertIs(rebuilt.norm_topk_prob, True)
        self.assertIs(rebuilt.seq_aux, False)
        self.assertEqual(rebuilt.model_type, "deepseek_v3")


class TestKnownProductionDefects(_ConfigTestBase):
    """Defects confirmed by reading production; no source is modified."""

    @unittest.expectedFailure
    def test_default_topk_method_is_dispatchable_by_gate(self):
        # BUG: configuration.py defaults topk_method="gready" (misspelled),
        # but DeepseekV2MoEGate only dispatches on "greedy" /
        # "group_limited_greedy" / "noaux_tc" (moe_gate.py). The default value
        # therefore matches no routing branch. This expected-failure documents
        # the mismatch without touching production code.
        default_method = DeepseekV2FastConfig().topk_method
        self.assertIn(default_method, _GATE_RECOGNIZED_TOPK_METHODS)

    def test_default_topk_method_current_value_is_locked(self):
        # Companion to the expectedFailure above: pin the *actual* current
        # (buggy) default so an intentional fix is a visible, deliberate change.
        self.assertEqual(DeepseekV2FastConfig().topk_method, "gready")


if __name__ == "__main__":
    unittest.main()
