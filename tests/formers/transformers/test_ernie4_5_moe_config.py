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

"""Behavior tests for Ernie4_5_MoeConfig.

Module: 配置与运行基础设施 (config infrastructure), 无卡 / CPU-only.

These tests exercise the real production config constructor and its
serialization pipeline. Rather than setting an attribute and reading it
back (self-assignment antipattern), each test either:
  * derives an expected value independently and checks the constructor's
    computed/derived output, or
  * verifies a field is actually *consumed* by a downstream mechanism
    (RoPE parameter standardization, unsavable-key filtering during
    serialization, moe_group stripping, diff-dict defaulting).
"""

import json
import unittest

from paddlefleet.transformers.ernie4_5_moe.configuration import (
    Ernie4_5_MoeConfig,
)


class TestErnie45MoeConfigDefaults(unittest.TestCase):
    """Default values and precise bool/str typing of scalar fields."""

    def test_scalar_defaults(self):
        config = Ernie4_5_MoeConfig()
        self.assertEqual(config.model_type, "ernie4_5_moe")
        self.assertEqual(config.vocab_size, 103424)
        self.assertEqual(config.hidden_size, 2560)
        self.assertEqual(config.intermediate_size, 12288)
        self.assertEqual(config.max_position_embeddings, 32768)
        self.assertEqual(config.num_hidden_layers, 3)
        self.assertEqual(config.num_attention_heads, 2)
        self.assertEqual(config.hidden_act, "silu")
        self.assertEqual(config.rms_norm_eps, 1e-6)
        self.assertEqual(config.pad_token_id, 0)
        self.assertEqual(config.bos_token_id, 1)
        self.assertEqual(config.eos_token_id, 2)
        self.assertEqual(config.rope_theta, 10000)
        self.assertEqual(config.moe_num_experts, 16)
        self.assertEqual(config.moe_k, 2)
        self.assertEqual(config.moe_num_shared_experts, 2)
        self.assertEqual(config.moe_capacity, [64, 64, 64])
        self.assertEqual(config.moe_intermediate_size, 0)
        self.assertIsNone(config.num_key_value_heads)

    def test_boolean_and_string_defaults_are_exact(self):
        """Distinguish real booleans from truthy strings / ints.

        A regression that swapped a bool default for a truthy string
        (e.g. use_cache="") would pass a loose ``assertFalse`` but must
        fail ``assertIs(..., False)``.
        """
        config = Ernie4_5_MoeConfig()
        self.assertIs(config.use_cache, False)
        self.assertIs(config.use_rmsnorm, True)
        self.assertIs(config.use_bias, False)
        self.assertIs(config.moe_use_aux_free, True)
        self.assertIs(config.moe_group_experts, False)
        self.assertIs(config.moe_norm_gate_logits, True)
        self.assertIs(config.sinkhorn_2gate, True)
        self.assertIs(config.global_aux_loss, False)
        # scoring_func is a *string* selector, not a bool.
        self.assertEqual(config.scoring_func, "softmax")
        self.assertIsInstance(config.scoring_func, str)

    def test_explicit_overrides_are_respected(self):
        config = Ernie4_5_MoeConfig(
            vocab_size=32000,
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=6,
            num_attention_heads=8,
            use_cache=True,
            moe_use_aux_free=False,
            scoring_func="sigmoid",
        )
        self.assertEqual(config.vocab_size, 32000)
        self.assertEqual(config.hidden_size, 1024)
        self.assertEqual(config.intermediate_size, 4096)
        self.assertEqual(config.num_hidden_layers, 6)
        self.assertEqual(config.num_attention_heads, 8)
        # Overridden booleans must flip exactly, not stay at their default.
        self.assertIs(config.use_cache, True)
        self.assertIs(config.moe_use_aux_free, False)
        self.assertEqual(config.scoring_func, "sigmoid")


class TestErnie45MoeConfigDerived(unittest.TestCase):
    """Fields the constructor computes from other fields."""

    def test_head_dim_derived_from_hidden_and_heads(self):
        # Independent expectation: hidden_size // num_attention_heads.
        for hidden, heads in ((2560, 2), (1024, 8), (4096, 16)):
            config = Ernie4_5_MoeConfig(
                hidden_size=hidden, num_attention_heads=heads
            )
            self.assertEqual(config.head_dim, hidden // heads)

    def test_head_dim_explicit_takes_precedence(self):
        # When head_dim is given it must NOT be recomputed from hidden/heads.
        config = Ernie4_5_MoeConfig(
            hidden_size=2560, num_attention_heads=2, head_dim=64
        )
        self.assertEqual(config.head_dim, 64)
        self.assertNotEqual(config.head_dim, 2560 // 2)

    def test_moe_layer_end_index_resolves_sentinel(self):
        # -1 sentinel resolves to num_hidden_layers - 1, tracked per value.
        for num_layers in (3, 10, 24):
            config = Ernie4_5_MoeConfig(
                num_hidden_layers=num_layers, moe_layer_end_index=-1
            )
            self.assertEqual(config.moe_layer_end_index, num_layers - 1)

    def test_moe_layer_end_index_explicit_kept(self):
        # A non-sentinel value is stored verbatim, not clamped to layers-1.
        config = Ernie4_5_MoeConfig(num_hidden_layers=10, moe_layer_end_index=5)
        self.assertEqual(config.moe_layer_end_index, 5)


class TestErnie45MoeConfigRopePropagation(unittest.TestCase):
    """rope_theta must reach the derived rope_parameters consumed by RoPE."""

    def test_default_rope_theta_populates_rope_parameters(self):
        config = Ernie4_5_MoeConfig()
        # standardize_rope_params builds this derived dict from rope_theta;
        # a downstream RoPE init reads it, so it must actually be populated.
        self.assertEqual(config.rope_parameters["rope_type"], "default")
        self.assertEqual(config.rope_parameters["rope_theta"], 10000)

    def test_custom_rope_theta_flows_into_rope_parameters(self):
        config = Ernie4_5_MoeConfig(rope_theta=50000)
        self.assertEqual(config.rope_theta, 50000)
        # The custom theta must propagate, not just sit on the attribute.
        self.assertEqual(config.rope_parameters["rope_theta"], 50000)
        self.assertEqual(config.rope_parameters["rope_type"], "default")


class TestErnie45MoeConfigSerialization(unittest.TestCase):
    """to_dict / to_diff_dict / to_json_string behavior and field filtering."""

    def test_tie_word_embeddings_default_true_and_override(self):
        self.assertIs(Ernie4_5_MoeConfig().tie_word_embeddings, True)
        self.assertIs(
            Ernie4_5_MoeConfig(tie_word_embeddings=False).tie_word_embeddings,
            False,
        )

    def test_unsavable_key_dropped_only_when_saving_file(self):
        """register_unsavable_keys must actually gate serialization output.

        moe_use_aux_free is registered unsavable: it appears in the plain
        dict but is filtered out when saving_file=True. Checking both
        branches proves the key list is consumed, not merely stored.
        """
        config = Ernie4_5_MoeConfig()
        full = config.to_dict(saving_file=False)
        saved = config.to_dict(saving_file=True)
        self.assertIn("moe_use_aux_free", full)
        self.assertNotIn("moe_use_aux_free", saved)
        # A non-unsavable structural field survives both.
        self.assertIn("hidden_size", full)
        self.assertIn("hidden_size", saved)

    def test_moe_group_always_stripped_from_dict(self):
        # moe_group is unconditionally removed by to_dict (both branches).
        config = Ernie4_5_MoeConfig(moe_group="mp")
        self.assertNotIn("moe_group", config.to_dict(saving_file=False))
        self.assertNotIn("moe_group", config.to_dict(saving_file=True))

    def test_diff_dict_keeps_non_default_and_always_tie(self):
        config = Ernie4_5_MoeConfig(hidden_size=1024)
        diff = config.to_diff_dict()
        # Explicitly changed field is present with the new value.
        self.assertEqual(diff["hidden_size"], 1024)
        # tie_word_embeddings is white-listed: always emitted even at default.
        self.assertIn("tie_word_embeddings", diff)
        # A field left at its class default is pruned from the diff.
        self.assertNotIn("rms_norm_eps", diff)

    def test_to_json_string_diff_reports_model_type(self):
        config = Ernie4_5_MoeConfig()
        parsed = json.loads(config.to_json_string(use_diff=True))
        self.assertEqual(parsed["model_type"], "ernie4_5_moe")

    def test_to_json_string_full_contains_structural_fields(self):
        config = Ernie4_5_MoeConfig()
        parsed = json.loads(config.to_json_string(use_diff=False))
        self.assertEqual(parsed["vocab_size"], 103424)
        self.assertEqual(parsed["hidden_size"], 2560)
        # Internal serialization bookkeeping must not leak into JSON.
        self.assertNotIn("_unsavable_keys", parsed)


class TestErnie45MoeConfigKnownBug(unittest.TestCase):
    """Documents a confirmed set-and-ignore bug in the constructor.

    configuration.py accepts recompute_granularity / recompute_method /
    recompute_modules / recompute_num_layers (and the *_mtp_* variants) as
    constructor arguments, but the body unconditionally overwrites every
    ``self.recompute_*`` attribute with ``None`` (lines ~246-253, with a
    duplicated ``self.recompute_granularity = None``). The user-supplied
    values are therefore silently discarded and never consumed.

    This test asserts the CORRECT contract (the argument should be stored),
    so it FAILS against current code, flagging the regression. See report.
    """

    def test_recompute_granularity_should_be_consumed(self):
        config = Ernie4_5_MoeConfig(recompute_granularity="full")
        self.assertEqual(config.recompute_granularity, "full")

    def test_recompute_num_layers_should_be_consumed(self):
        config = Ernie4_5_MoeConfig(recompute_num_layers=2)
        self.assertEqual(config.recompute_num_layers, 2)


if __name__ == "__main__":
    unittest.main()
