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

"""Behavior tests for ``TransformerConfig.__post_init__`` derived logic.

These tests exercise the *real* construction entry (``TransformerConfig(...)``
runs ``__post_init__``) and check values the config layer DERIVES or VALIDATES,
never a plain ``set -> read back`` of a field that was passed in unchanged:

* the tri-state normalization of ``use_accuracy_compatible`` (True/"1"/"on"
  collapse to "megatron", falsy spellings collapse to False, unknown strings
  raise, non str/bool raise);
* ``dsa_indexer_loss_coeff`` collapsing None/0 to the float 0.0;
* the dimension defaults derived from ``hidden_size`` / ``num_attention_heads``
  (head_dim, v_head_dim, num_key_value_heads, intermediate_size) and that an
  explicit value suppresses the derivation;
* the ``num_key_value_heads`` vs ``tensor_model_parallel_size`` divisibility
  guard;
* ``apply_query_key_layer_scaling`` forcing ``attention_softmax_in_fp32``;
* the ``first_k_dense_replace`` + ``moe_layer_freq`` -> per-layer
  ``moe_layer_freq`` pattern list (a value actually consumed downstream, so a
  broken derivation would change the emitted pattern).

Every expected value below is derived by hand from the production source, not
copied from any fixture. The config module imports Paddle transitively, so the
import is guarded and the whole module is skipped (with the real error) when
Paddle is unavailable; nothing is faked to make it pass.
"""

import unittest

try:
    from paddlefleet.accuracy_target import (
        ACCURACY_TARGET_HF,
        ACCURACY_TARGET_MEGATRON,
    )
    from paddlefleet.transformer.transformer_config import TransformerConfig

    _IMPORT_ERROR: BaseException | None = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    TransformerConfig = None
    ACCURACY_TARGET_HF = None
    ACCURACY_TARGET_MEGATRON = None
    _IMPORT_ERROR = exc

_HAVE_CONFIG = _IMPORT_ERROR is None
_SKIP_REASON = (
    "paddlefleet.transformer.transformer_config could not be imported "
    f"(CPU-only, no Paddle): {_IMPORT_ERROR!r}"
)


@unittest.skipUnless(_HAVE_CONFIG, _SKIP_REASON)
class TestAccuracyTargetNormalization(unittest.TestCase):
    """``use_accuracy_compatible`` is normalized to a tri-state by post-init."""

    def test_default_is_false(self):
        cfg = TransformerConfig()
        # Falsy default must collapse to the bool False, not None/"" etc.
        self.assertIs(cfg.use_accuracy_compatible, False)

    def test_truthy_spellings_become_megatron(self):
        # True predates the "hf" target; "1"/"on"/"yes"/int 1 mean the same.
        for value in (
            True,
            1,
            "megatron",
            "MEGATRON",
            "true",
            "on",
            "yes",
            "1",
        ):
            with self.subTest(value=value):
                cfg = TransformerConfig(use_accuracy_compatible=value)
                self.assertEqual(
                    cfg.use_accuracy_compatible, ACCURACY_TARGET_MEGATRON
                )

    def test_hf_target_is_preserved(self):
        for value in ("hf", "HF", " hf "):
            with self.subTest(value=value):
                cfg = TransformerConfig(use_accuracy_compatible=value)
                self.assertEqual(
                    cfg.use_accuracy_compatible, ACCURACY_TARGET_HF
                )

    def test_falsy_spellings_become_false(self):
        for value in (False, 0, "", None, "false", "off", "no", "null", "0"):
            with self.subTest(value=value):
                cfg = TransformerConfig(use_accuracy_compatible=value)
                self.assertIs(cfg.use_accuracy_compatible, False)

    def test_megatron_and_hf_are_distinct_targets(self):
        # Guards against both branches collapsing to one value.
        self.assertNotEqual(ACCURACY_TARGET_MEGATRON, ACCURACY_TARGET_HF)

    def test_unknown_string_raises_value_error(self):
        with self.assertRaises(ValueError):
            TransformerConfig(use_accuracy_compatible="bf16-ish")

    def test_non_bool_non_str_raises_type_error(self):
        # A float is neither the True/1 special-case nor a string spelling.
        with self.assertRaises(TypeError):
            TransformerConfig(use_accuracy_compatible=1.5)


@unittest.skipUnless(_HAVE_CONFIG, _SKIP_REASON)
class TestDsaIndexerLossCoeffNormalization(unittest.TestCase):
    """None/0 collapse to float 0.0; a real coefficient passes through."""

    def test_default_is_float_zero(self):
        cfg = TransformerConfig()
        self.assertIsInstance(cfg.dsa_indexer_loss_coeff, float)
        self.assertEqual(cfg.dsa_indexer_loss_coeff, 0.0)

    def test_none_collapses_to_zero(self):
        cfg = TransformerConfig(dsa_indexer_loss_coeff=None)
        self.assertIsInstance(cfg.dsa_indexer_loss_coeff, float)
        self.assertEqual(cfg.dsa_indexer_loss_coeff, 0.0)

    def test_int_zero_collapses_to_float_zero(self):
        cfg = TransformerConfig(dsa_indexer_loss_coeff=0)
        self.assertIsInstance(cfg.dsa_indexer_loss_coeff, float)
        self.assertEqual(cfg.dsa_indexer_loss_coeff, 0.0)

    def test_nonzero_value_is_cast_to_float(self):
        cfg = TransformerConfig(dsa_indexer_loss_coeff=1)
        self.assertIsInstance(cfg.dsa_indexer_loss_coeff, float)
        self.assertEqual(cfg.dsa_indexer_loss_coeff, 1.0)

    def test_fractional_value_is_preserved(self):
        cfg = TransformerConfig(dsa_indexer_loss_coeff=0.25)
        self.assertEqual(cfg.dsa_indexer_loss_coeff, 0.25)


@unittest.skipUnless(_HAVE_CONFIG, _SKIP_REASON)
class TestDerivedDimensionDefaults(unittest.TestCase):
    """Dimensions left as None are derived from hidden_size / head count."""

    def test_head_dim_derived_from_hidden_size(self):
        # head_dim = hidden_size // num_attention_heads = 128 // 8 = 16.
        cfg = TransformerConfig(hidden_size=128, num_attention_heads=8)
        self.assertEqual(cfg.head_dim, 16)
        # v_head_dim defaults to head_dim when unset.
        self.assertEqual(cfg.v_head_dim, 16)

    def test_head_dim_floor_division(self):
        # 128 // 6 = 21 (floor), not 128/6; the derivation uses //.
        cfg = TransformerConfig(hidden_size=128, num_attention_heads=6)
        self.assertEqual(cfg.head_dim, 21)

    def test_explicit_head_dim_suppresses_derivation(self):
        # 128 // 8 would be 16; an explicit 40 must win, proving the guard
        # only derives when head_dim is None (not a blind overwrite).
        cfg = TransformerConfig(
            hidden_size=128, num_attention_heads=8, head_dim=40
        )
        self.assertEqual(cfg.head_dim, 40)
        # v_head_dim then follows the explicit head_dim, not hidden_size//heads.
        self.assertEqual(cfg.v_head_dim, 40)

    def test_explicit_v_head_dim_independent_of_head_dim(self):
        cfg = TransformerConfig(
            hidden_size=128, num_attention_heads=8, v_head_dim=9
        )
        self.assertEqual(cfg.head_dim, 16)
        self.assertEqual(cfg.v_head_dim, 9)

    def test_num_key_value_heads_defaults_to_attention_heads(self):
        cfg = TransformerConfig(hidden_size=128, num_attention_heads=8)
        self.assertEqual(cfg.num_key_value_heads, 8)

    def test_explicit_num_key_value_heads_kept(self):
        cfg = TransformerConfig(
            hidden_size=128, num_attention_heads=8, num_key_value_heads=2
        )
        self.assertEqual(cfg.num_key_value_heads, 2)

    def test_intermediate_size_derived_as_four_times_hidden(self):
        cfg = TransformerConfig(hidden_size=128, num_attention_heads=8)
        self.assertEqual(cfg.intermediate_size, 512)

    def test_explicit_intermediate_size_suppresses_derivation(self):
        cfg = TransformerConfig(
            hidden_size=128, num_attention_heads=8, intermediate_size=100
        )
        self.assertEqual(cfg.intermediate_size, 100)


@unittest.skipUnless(_HAVE_CONFIG, _SKIP_REASON)
class TestHeadCountValidation(unittest.TestCase):
    """num_key_value_heads must divide tensor_model_parallel_size."""

    def test_indivisible_kv_heads_raise(self):
        # 3 KV heads cannot be split across 2 TP ranks.
        with self.assertRaises(ValueError):
            TransformerConfig(
                hidden_size=128,
                num_attention_heads=4,
                num_key_value_heads=3,
                tensor_model_parallel_size=2,
            )

    def test_divisible_kv_heads_construct(self):
        cfg = TransformerConfig(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            tensor_model_parallel_size=2,
        )
        self.assertEqual(cfg.num_key_value_heads, 4)


@unittest.skipUnless(_HAVE_CONFIG, _SKIP_REASON)
class TestSoftmaxFp32Coupling(unittest.TestCase):
    """apply_query_key_layer_scaling forces attention_softmax_in_fp32=True."""

    def test_scaling_forces_fp32_softmax_even_when_disabled(self):
        # Explicitly request fp32 softmax OFF, then turn on QK-layer scaling:
        # post-init must override it back to True (a derived coupling, not a
        # value read back unchanged).
        cfg = TransformerConfig(
            hidden_size=128,
            num_attention_heads=8,
            apply_query_key_layer_scaling=True,
            attention_softmax_in_fp32=False,
        )
        self.assertIs(cfg.attention_softmax_in_fp32, True)

    def test_no_scaling_keeps_fp32_softmax_disabled(self):
        cfg = TransformerConfig(
            hidden_size=128,
            num_attention_heads=8,
            apply_query_key_layer_scaling=False,
            attention_softmax_in_fp32=False,
        )
        self.assertIs(cfg.attention_softmax_in_fp32, False)


@unittest.skipUnless(_HAVE_CONFIG, _SKIP_REASON)
class TestMoeLayerFreqPatternDerivation(unittest.TestCase):
    """first_k_dense_replace + moe_layer_freq -> per-layer 0/1 pattern list."""

    def test_scalar_freq_when_nothing_set(self):
        # Neither first_k_dense_replace nor moe_layer_freq set -> scalar 1,
        # left as an int (not expanded into a list).
        cfg = TransformerConfig(hidden_size=128, num_attention_heads=8)
        self.assertEqual(cfg.moe_layer_freq, 1)

    def test_dense_prefix_with_default_freq_is_all_moe_after(self):
        # first_k_dense_replace=2, num_hidden_layers=6, moe_layer_freq unset:
        # the 2 leading layers are dense (0), the remaining 4 are all MoE (1).
        cfg = TransformerConfig(
            hidden_size=128,
            num_attention_heads=8,
            num_hidden_layers=6,
            first_k_dense_replace=2,
        )
        self.assertEqual(cfg.moe_layer_freq, [0, 0, 1, 1, 1, 1])

    def test_dense_prefix_with_periodic_freq(self):
        # first_k_dense_replace=2, num_hidden_layers=6, moe_layer_freq=2:
        # leading 2 dense -> [0, 0]; then for i in range(4) mark MoE when
        # (i+1) % 2 == 0 -> [0, 1, 0, 1]; concatenated -> [0,0,0,1,0,1].
        cfg = TransformerConfig(
            hidden_size=128,
            num_attention_heads=8,
            num_hidden_layers=6,
            first_k_dense_replace=2,
            moe_layer_freq=2,
        )
        self.assertEqual(cfg.moe_layer_freq, [0, 0, 0, 1, 0, 1])

    def test_periodic_freq_changes_pattern(self):
        # A different period must produce a different consumed pattern, so the
        # value is genuinely used and not just stored. freq=3, 6 layers, 2
        # dense: tail range(4) marks MoE at (i+1)%3==0 -> i=2 only -> [0,0,1,0];
        # prefix [0,0] -> [0,0,0,0,1,0].
        cfg = TransformerConfig(
            hidden_size=128,
            num_attention_heads=8,
            num_hidden_layers=6,
            first_k_dense_replace=2,
            moe_layer_freq=3,
        )
        self.assertEqual(cfg.moe_layer_freq, [0, 0, 0, 0, 1, 0])


if __name__ == "__main__":
    unittest.main()
