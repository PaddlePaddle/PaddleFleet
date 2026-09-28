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

"""Behavior tests for MiniMaxM2Config.

Module: 配置与运行基础设施 (config infrastructure), 无卡 / CPU-only.

These tests exercise the real production config constructor
(``MiniMaxM2Config``) and one real consumer of the config fields
(``MiniMaxM2PreTrainedModel.get_layer_attn_split_info``). Rather than
setting an attribute and reading it straight back (the self-assignment
antipattern), each test either:

  * pins a default against an independently known literal from the model
    spec, checking exact bool/str typing so a truthy-string regression is
    rejected;
  * derives an expected value by hand (e.g. ``v_head_dim`` defaulting to
    ``head_dim``) and checks the constructor's computed output;
  * drives the RoPE standardization pipeline and checks the migrated /
    propagated ``rope_parameters`` dict a downstream RoPE init consumes; or
  * feeds the config into ``get_layer_attn_split_info`` and checks the
    hand-derived attention split layout, proving the flags are actually
    *consumed* (toggling ``use_gated_attn`` / ``use_vha_attention`` /
    ``v_head_dim`` changes the returned split, not merely a stored attr).

Note: importing the config already pulls in Paddle transitively, so these
run under the 无卡 (CPU-only) environment where Paddle CPU is installed;
they do not require an accelerator.
"""

import unittest

from paddlefleet.transformers.minimax_m2.configuration import MiniMaxM2Config


class TestMiniMaxM2ConfigDefaults(unittest.TestCase):
    """Default scalar values, pinned to the published MiniMax-M2 spec."""

    def test_scalar_defaults(self):
        config = MiniMaxM2Config()
        self.assertEqual(config.model_type, "minimax_m2")
        self.assertEqual(config.vocab_size, 200064)
        self.assertEqual(config.hidden_size, 3072)
        self.assertEqual(config.head_dim, 128)
        self.assertEqual(config.moe_intermediate_size, 1536)
        self.assertEqual(config.num_hidden_layers, 62)
        self.assertEqual(config.num_attention_heads, 48)
        self.assertEqual(config.num_key_value_heads, 8)
        self.assertEqual(config.hidden_act, "silu")
        self.assertEqual(config.max_position_embeddings, 196608)
        self.assertEqual(config.initializer_range, 0.02)
        self.assertEqual(config.rms_norm_eps, 1e-6)
        self.assertEqual(config.rope_theta, 5000000)
        self.assertEqual(config.rotary_dim, 64)
        self.assertEqual(config.attention_dropout, 0.0)

    def test_moe_scalar_defaults(self):
        config = MiniMaxM2Config()
        self.assertEqual(config.num_experts_per_tok, 8)
        self.assertEqual(config.n_shared_experts, 0)
        self.assertEqual(config.n_routed_experts, 256)
        self.assertEqual(config.n_group, 1)
        self.assertEqual(config.topk_group, 1)
        self.assertEqual(config.first_k_dense_replace, 0)
        self.assertEqual(config.moe_layer_freq, 1)
        self.assertEqual(config.num_mtp_modules, 3)
        self.assertEqual(config.mtp_transformer_layers, 1)

    def test_optional_defaults_are_none(self):
        # Distinguish "field absent / default None" from a stored 0 or "".
        config = MiniMaxM2Config()
        self.assertIsNone(config.rope_scaling)
        self.assertIsNone(config.sliding_window)
        self.assertIsNone(config.attn_type_list)
        self.assertIsNone(config.window_attn_skip_freq)
        self.assertIsNone(config.experimental_attention_variant)
        self.assertIsNone(config.moe_latent_size)

    def test_keys_to_ignore_at_inference(self):
        # Class attribute, not instance state; must be exactly this list.
        self.assertEqual(
            MiniMaxM2Config.keys_to_ignore_at_inference, ["past_key_values"]
        )


class TestMiniMaxM2ConfigBoolStringTyping(unittest.TestCase):
    """Exact bool vs string typing.

    A regression swapping a real bool default for a truthy string
    (e.g. ``use_mtp="yes"``) or a string selector for a bool would slip
    past a loose ``assertTrue`` / ``assertFalse`` but must fail these
    identity / instance checks.
    """

    def test_boolean_defaults_are_true(self):
        config = MiniMaxM2Config()
        self.assertIs(config.use_cache, True)
        self.assertIs(config.use_qk_norm, True)
        self.assertIs(config.use_mtp, True)
        self.assertIs(config.use_routing_bias, True)
        self.assertIs(config.norm_topk_prob, True)
        self.assertIs(config.seq_aux, True)
        self.assertIs(config.using_flex_token, True)

    def test_boolean_defaults_are_false(self):
        config = MiniMaxM2Config()
        self.assertIs(config.attention_bias, False)
        self.assertIs(config.disable_ffn_model_parallel, False)
        self.assertIs(config.fd_fallback, False)
        self.assertIs(config.use_gated_attn, False)
        self.assertIs(config.use_vha_attention, False)
        self.assertIs(config.add_full_attention_sink_bias, False)
        self.assertIs(config.add_swa_attention_sink_bias, False)
        self.assertIs(config.routed_scaling_factor_learnable, False)
        # use_fp8 is force-initialized to False inside __init__.
        self.assertIs(config.use_fp8, False)

    def test_string_selectors_are_strings(self):
        config = MiniMaxM2Config()
        for name, expected in (
            ("qk_norm_type", "per_layer"),
            ("scoring_func", "sigmoid"),
            ("topk_method", "noaux_tc"),
            ("hidden_act", "silu"),
            ("pp_seg_method", "layer:Glm4MoeDecoderLayer"),
        ):
            value = getattr(config, name)
            self.assertIsInstance(value, str)
            self.assertEqual(value, expected)


class TestMiniMaxM2ConfigOverrides(unittest.TestCase):
    """Explicit overrides must take effect, including explicit-off flags."""

    def test_scalar_overrides_respected(self):
        config = MiniMaxM2Config(
            vocab_size=100000,
            hidden_size=2048,
            num_hidden_layers=32,
            n_routed_experts=64,
            num_experts_per_tok=4,
        )
        self.assertEqual(config.vocab_size, 100000)
        self.assertEqual(config.hidden_size, 2048)
        self.assertEqual(config.num_hidden_layers, 32)
        self.assertEqual(config.n_routed_experts, 64)
        self.assertEqual(config.num_experts_per_tok, 4)

    def test_boolean_flags_flip_exactly(self):
        # Explicitly turning defaults off must land on real False, not stay
        # at the default True (and vice versa for a default-False flag).
        config = MiniMaxM2Config(
            use_cache=False,
            use_qk_norm=False,
            use_mtp=False,
            use_routing_bias=False,
            use_gated_attn=True,
        )
        self.assertIs(config.use_cache, False)
        self.assertIs(config.use_qk_norm, False)
        self.assertIs(config.use_mtp, False)
        self.assertIs(config.use_routing_bias, False)
        self.assertIs(config.use_gated_attn, True)

    def test_string_selector_override(self):
        config = MiniMaxM2Config(
            scoring_func="softmax", qk_norm_type="per_head"
        )
        self.assertEqual(config.scoring_func, "softmax")
        self.assertEqual(config.qk_norm_type, "per_head")


class TestMiniMaxM2ConfigDerived(unittest.TestCase):
    """Fields the constructor computes rather than stores verbatim."""

    def test_v_head_dim_defaults_to_head_dim(self):
        # Independent expectation: when v_head_dim is omitted it falls back
        # to head_dim, tracked across several head_dim values.
        for head_dim in (32, 64, 128):
            config = MiniMaxM2Config(head_dim=head_dim)
            self.assertEqual(config.v_head_dim, head_dim)

    def test_v_head_dim_explicit_takes_precedence(self):
        # A supplied v_head_dim must NOT be overwritten by head_dim.
        config = MiniMaxM2Config(head_dim=128, v_head_dim=64)
        self.assertEqual(config.v_head_dim, 64)
        self.assertNotEqual(config.v_head_dim, config.head_dim)


class TestMiniMaxM2ConfigRope(unittest.TestCase):
    """rope_scaling migration + rope_theta propagation into rope_parameters.

    ``rope_parameters`` is the derived dict a downstream RoPE init consumes,
    so it must actually be populated, not just mirror the raw input.
    """

    def test_default_rope_parameters_populated(self):
        config = MiniMaxM2Config()
        # standardize_rope_params builds a "default" entry from rope_theta.
        self.assertEqual(config.rope_parameters["rope_type"], "default")
        self.assertEqual(config.rope_parameters["rope_theta"], 5000000)

    def test_custom_rope_theta_flows_into_rope_parameters(self):
        config = MiniMaxM2Config(rope_theta=100000)
        self.assertEqual(config.rope_theta, 100000)
        # Must propagate into the derived dict, not merely sit on the attr.
        self.assertEqual(config.rope_parameters["rope_theta"], 100000)
        self.assertEqual(config.rope_parameters["rope_type"], "default")

    def test_rope_scaling_type_migrated_to_rope_type(self):
        # The BC path copies the legacy 'type' key into 'rope_type'.
        config = MiniMaxM2Config(rope_scaling={"type": "linear", "factor": 2.0})
        self.assertEqual(config.rope_scaling["rope_type"], "linear")
        # The original 'type' and 'factor' entries are preserved.
        self.assertEqual(config.rope_scaling["type"], "linear")
        self.assertEqual(config.rope_scaling["factor"], 2.0)
        # rope_theta is back-filled into the same dict from the config theta.
        self.assertEqual(config.rope_scaling["rope_theta"], 5000000)

    def test_rope_parameters_is_the_rope_scaling_object(self):
        # rope_parameters aliases the (migrated) rope_scaling dict, so both
        # views see the standardized rope_type. This is the contract the
        # RoPE init relies on when reading config.rope_parameters.
        config = MiniMaxM2Config(rope_scaling={"type": "linear", "factor": 2.0})
        self.assertIs(config.rope_parameters, config.rope_scaling)
        self.assertEqual(config.rope_parameters["rope_type"], "linear")


class TestMiniMaxM2ConfigConsumption(unittest.TestCase):
    """Config fields must be consumed, not merely stored.

    ``MiniMaxM2PreTrainedModel.get_layer_attn_split_info`` reads head_dim,
    v_head_dim, num_attention_heads, num_key_value_heads, use_gated_attn and
    use_vha_attention directly to compute the per-group QKV split layout.
    Each expected split below is derived by hand from the head geometry, so
    a regression that ignored one of these flags would produce a different
    split and fail here.

    num_empty_layers_add_in_head is supplied (via kwargs) because
    get_layer_attn_split_info subtracts it unconditionally; sliding_window
    is left None so is_layer_window_attention returns False and the standard
    (non-SWA) branch is exercised.
    """

    def _config(self, **overrides):
        base = {
            "hidden_size": 128,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 16,
            "num_hidden_layers": 4,
            "num_empty_layers_add_in_head": 0,
        }
        base.update(overrides)
        return MiniMaxM2Config(**base)

    def _split_info(self, config):
        from paddlefleet.transformers.minimax_m2.modeling import (
            MiniMaxM2PreTrainedModel,
        )

        return MiniMaxM2PreTrainedModel.get_layer_attn_split_info(
            config, layer_idx=0
        )

    def test_standard_gqa_split_layout(self):
        # Non-gated, non-VHA: per group = [Q, K, V].
        #   num_query_groups = num_key_value_heads = 4
        #   q_heads_per_group = 8 // 4 = 2
        #   q_dim = q_heads_per_group * head_dim = 2 * 16 = 32
        #   split = [q_dim, head_dim, v_head_dim] = [32, 16, 16]
        config = self._config()
        (
            num_heads,
            num_kv_heads,
            num_query_groups,
            split_dims,
            is_swa,
        ) = self._split_info(config)
        self.assertEqual(num_heads, 8)
        self.assertEqual(num_kv_heads, 4)
        self.assertEqual(num_query_groups, 4)
        self.assertEqual(split_dims, [32, 16, 16])
        self.assertFalse(is_swa)

    def test_gated_attn_inserts_gate_dim(self):
        # use_gated_attn=True: per group = [Q, Gate, K, V].
        #   gate_dim = heads_per_group * v_head_dim = 2 * 16 = 32
        #   split = [q_dim, gate_dim, head_dim, v_head_dim] = [32, 32, 16, 16]
        config = self._config(use_gated_attn=True)
        _, _, _, split_dims, _ = self._split_info(config)
        self.assertEqual(split_dims, [32, 32, 16, 16])
        # And the non-gated config must NOT carry the extra gate segment,
        # confirming the flag actually drives the branch.
        _, _, _, plain_dims, _ = self._split_info(self._config())
        self.assertEqual(len(plain_dims), 3)
        self.assertNotEqual(split_dims, plain_dims)

    def test_v_head_dim_is_consumed_distinctly_from_head_dim(self):
        # With v_head_dim != head_dim the V segment must reflect v_head_dim,
        # while Q/K keep head_dim -> proves the two fields are not conflated.
        #   head_dim=16, v_head_dim=8, q_dim = 2 * 16 = 32
        #   split = [q_dim, head_dim, v_head_dim] = [32, 16, 8]
        config = self._config(v_head_dim=8)
        _, _, _, split_dims, _ = self._split_info(config)
        self.assertEqual(split_dims, [32, 16, 8])

    def test_vha_attention_shrinks_query_group(self):
        # use_vha_attention=True divides q_heads_per_group by num_kv_heads:
        #   q_heads_per_group = (8 // 4) // 4 = 0  -> q_dim = 0
        #   split = [q_dim, head_dim, v_head_dim] = [0, 16, 16]
        # The distinct q_dim vs the standard case proves the flag is read.
        config = self._config(use_vha_attention=True)
        _, _, _, split_dims, _ = self._split_info(config)
        self.assertEqual(split_dims, [0, 16, 16])


if __name__ == "__main__":
    unittest.main()
