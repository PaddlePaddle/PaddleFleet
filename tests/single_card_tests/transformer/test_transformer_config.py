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

"""Behaviour tests for ``TransformerConfig.__post_init__`` derivations,
``register_attributes`` / ``_process_attribute`` transforms and validation.

These exercise the *derived* config behaviour (defaults computed from other
fields, validation raises, alias/rename mapping, callable resolution) rather
than reading back a value that was just assigned. The config object is pure
Python but is imported from a module that imports ``paddle`` at top level, so
the whole import is guarded and the suite is skipped (never faked green) when
paddle is unavailable.
"""

import functools
import unittest
from types import SimpleNamespace

_IMPORT_ERROR = None
try:
    import paddle.nn.functional as F

    from paddlefleet.transformer.activations import situ
    from paddlefleet.transformer.transformer_config import TransformerConfig
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    F = None
    situ = None
    TransformerConfig = None
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    ""
    if _HAS_DEPS
    else f"transformer_config import failed (needs paddle): {_IMPORT_ERROR!r}"
)


def _make_config(**overrides):
    defaults = {
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestDerivedDefaults(unittest.TestCase):
    """__post_init__ computes fields from other fields (not echoed inputs)."""

    def test_intermediate_size_derived_from_hidden_size(self):
        cfg = _make_config(hidden_size=64, intermediate_size=None)
        self.assertEqual(cfg.intermediate_size, 4 * 64)

    def test_explicit_intermediate_size_is_not_overwritten(self):
        # Control for the derivation above: an explicit value must survive.
        cfg = _make_config(hidden_size=64, intermediate_size=333)
        self.assertEqual(cfg.intermediate_size, 333)

    def test_head_dim_derived_as_floor_division(self):
        cfg = _make_config(
            hidden_size=256, num_attention_heads=8, head_dim=None
        )
        self.assertEqual(cfg.head_dim, 256 // 8)

    def test_head_dim_propagates_to_value_and_swa_heads(self):
        cfg = _make_config(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=8,
            head_dim=None,
        )
        # head_dim=32 then fans out to the value/sliding-window mirrors.
        self.assertEqual(cfg.head_dim, 32)
        self.assertEqual(cfg.v_head_dim, 32)
        self.assertEqual(cfg.swa_head_dim, 32)
        self.assertEqual(cfg.swa_v_head_dim, 32)
        self.assertEqual(cfg.swa_num_attention_heads, 8)
        self.assertEqual(cfg.swa_num_key_value_heads, 8)

    def test_num_key_value_heads_defaults_to_attention_heads(self):
        cfg = _make_config(num_attention_heads=6, num_key_value_heads=None)
        self.assertEqual(cfg.num_key_value_heads, 6)
        self.assertEqual(cfg.swa_num_key_value_heads, 6)

    def test_embedding_init_std_falls_back_to_init_std(self):
        cfg = _make_config(init_method_std=0.03, embedding_init_method_std=None)
        self.assertEqual(cfg.embedding_init_method_std, 0.03)

    def test_embedding_init_std_explicit_is_kept(self):
        # Control: an explicit embedding std must not be replaced by init std.
        cfg = _make_config(init_method_std=0.03, embedding_init_method_std=0.05)
        self.assertEqual(cfg.embedding_init_method_std, 0.05)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestQueryKeyScalingSideEffect(unittest.TestCase):
    """apply_query_key_layer_scaling forces fp32 softmax."""

    def test_scaling_forces_fp32_softmax(self):
        cfg = _make_config(
            apply_query_key_layer_scaling=True,
            attention_softmax_in_fp32=False,
        )
        self.assertTrue(cfg.attention_softmax_in_fp32)

    def test_no_scaling_leaves_softmax_flag_untouched(self):
        # Control so the test above is a real override, not a constant-True read.
        cfg = _make_config(
            apply_query_key_layer_scaling=False,
            attention_softmax_in_fp32=False,
        )
        self.assertFalse(cfg.attention_softmax_in_fp32)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestMoELayerFreqPattern(unittest.TestCase):
    """first_k_dense_replace expands moe_layer_freq into a per-layer pattern.

    Expected patterns are derived by hand from the rule: the first
    ``first_k_dense_replace`` layers are dense (0); each remaining layer i
    (0-based within the MoE span) is MoE (1) iff ``(i + 1) % freq == 0`` when a
    freq is given, else every remaining layer is MoE.
    """

    def test_default_freq_one_when_both_unset(self):
        cfg = _make_config()
        self.assertEqual(cfg.moe_layer_freq, 1)

    def test_pattern_every_second_moe_layer(self):
        cfg = _make_config(
            num_hidden_layers=6, first_k_dense_replace=2, moe_layer_freq=2
        )
        # dense x2 then (i+1)%2==0 over range(4) -> [0,1,0,1]
        self.assertEqual(cfg.moe_layer_freq, [0, 0, 0, 1, 0, 1])

    def test_pattern_every_third_moe_layer(self):
        cfg = _make_config(
            num_hidden_layers=8, first_k_dense_replace=2, moe_layer_freq=3
        )
        # dense x2 then (i+1)%3==0 over range(6) -> [0,0,1,0,0,1]
        self.assertEqual(cfg.moe_layer_freq, [0, 0, 0, 0, 1, 0, 0, 1])

    def test_pattern_all_moe_after_dense_prefix(self):
        cfg = _make_config(num_hidden_layers=5, first_k_dense_replace=1)
        # No freq given -> every non-dense layer is MoE.
        self.assertEqual(cfg.moe_layer_freq, [0, 1, 1, 1, 1])

    def test_first_k_dense_with_list_freq_is_rejected(self):
        with self.assertRaises(ValueError):
            _make_config(first_k_dense_replace=2, moe_layer_freq=[1, 0])


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestParallelAndLayerValidation(unittest.TestCase):
    """Cross-field validation raises with the right exception type."""

    def test_kv_heads_indivisible_by_tp_raises(self):
        with self.assertRaises(ValueError):
            _make_config(num_key_value_heads=3, tensor_model_parallel_size=2)

    def test_kv_heads_divisible_by_tp_ok(self):
        cfg = _make_config(num_key_value_heads=4, tensor_model_parallel_size=2)
        self.assertEqual(cfg.num_key_value_heads, 4)

    def test_zero_hidden_layers_raises_zero_division(self):
        # scaled_init_method_normal divides sigma by sqrt(multiplier*num_layers);
        # num_hidden_layers=0 surfaces as an opaque ZeroDivisionError today.
        with self.assertRaises(ZeroDivisionError):
            _make_config(num_hidden_layers=0)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestRecomputeGranularityValidation(unittest.TestCase):
    """recompute_granularity normalisation and full/selective guards."""

    def test_empty_string_becomes_none(self):
        cfg = _make_config(recompute_granularity="")
        self.assertIsNone(cfg.recompute_granularity)

    def test_unknown_granularity_rejected(self):
        with self.assertRaises(AssertionError):
            _make_config(recompute_granularity="invalid")

    def test_full_without_method_rejected(self):
        with self.assertRaises(AssertionError):
            _make_config(recompute_granularity="full")

    def test_full_with_block_method_ok(self):
        cfg = _make_config(
            recompute_granularity="full",
            recompute_method="block",
            recompute_num_layers=1,
        )
        self.assertEqual(cfg.recompute_granularity, "full")

    def test_selective_without_modules_rejected(self):
        with self.assertRaises(AssertionError):
            _make_config(recompute_granularity="selective")

    def test_selective_with_modules_ok(self):
        cfg = _make_config(
            recompute_granularity="selective",
            recompute_modules=["core_attn"],
        )
        self.assertEqual(cfg.recompute_granularity, "selective")


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestHiddenActResolution(unittest.TestCase):
    """_process_attribute maps hidden_act strings to the right callables."""

    def test_plain_name_resolves_to_functional(self):
        cfg = _make_config()
        cfg._process_attribute("hidden_act", "silu")
        self.assertIs(cfg.hidden_act, F.silu)

    def test_gelu_pytorch_tanh_uses_approximate_partial(self):
        cfg = _make_config()
        cfg._process_attribute("hidden_act", "gelu_pytorch_tanh")
        self.assertIsInstance(cfg.hidden_act, functools.partial)
        self.assertIs(cfg.hidden_act.func, F.gelu)
        self.assertEqual(cfg.hidden_act.keywords, {"approximate": True})

    def test_situ_name_resolves_to_situ(self):
        cfg = _make_config()
        cfg._process_attribute("hidden_act", "situ")
        self.assertIs(cfg.hidden_act, situ)

    def test_callable_passed_through(self):
        cfg = _make_config()

        def _identity(x):
            return x

        cfg._process_attribute("hidden_act", _identity)
        self.assertIs(cfg.hidden_act, _identity)

    def test_non_str_non_callable_rejected(self):
        cfg = _make_config()
        with self.assertRaises(TypeError):
            cfg._process_attribute("hidden_act", 123)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestFromConfigRegisterAttributes(unittest.TestCase):
    """from_config -> register_attributes applies transform/rename rules."""

    def test_dsa_alias_lands_on_canonical_field(self):
        src = SimpleNamespace(
            hidden_size=64, num_attention_heads=2, index_n_heads=7
        )
        cfg = TransformerConfig.from_config(src)
        # The alias key is rewritten to its canonical destination and consumed.
        self.assertEqual(cfg.dsa_index_n_heads, 7)
        self.assertEqual(cfg.hidden_size, 64)

    def test_renamed_key_is_rejected(self):
        src = SimpleNamespace(
            hidden_size=64, num_attention_heads=2, non_absorbed_mqa=True
        )
        with self.assertRaises(ValueError):
            TransformerConfig.from_config(src)

    def test_deprecated_quant_format_rejected(self):
        src = SimpleNamespace(
            hidden_size=64, num_attention_heads=2, sonicmoe_quant_format="x"
        )
        with self.assertRaises(ValueError):
            TransformerConfig.from_config(src)

    def test_rename_when_set_rejects_truthy_only(self):
        # mtp_num_layers is rejected only when it would change behaviour.
        truthy = SimpleNamespace(
            hidden_size=64, num_attention_heads=2, mtp_num_layers=2
        )
        with self.assertRaises(ValueError):
            TransformerConfig.from_config(truthy)

        falsy = SimpleNamespace(
            hidden_size=64, num_attention_heads=2, mtp_num_layers=0
        )
        cfg = TransformerConfig.from_config(falsy)
        self.assertEqual(cfg.mtp_num_layers, 0)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestGetAccessor(unittest.TestCase):
    """get() reads the post-init state, returning defaults for misses."""

    def test_get_returns_derived_value(self):
        cfg = _make_config(hidden_size=64, intermediate_size=None)
        # Reads the computed field, not a stored input.
        self.assertEqual(cfg.get("intermediate_size"), 256)

    def test_get_missing_returns_default(self):
        cfg = _make_config()
        self.assertIsNone(cfg.get("definitely_not_a_field"))
        self.assertEqual(cfg.get("definitely_not_a_field", 42), 42)


if __name__ == "__main__":
    unittest.main()
