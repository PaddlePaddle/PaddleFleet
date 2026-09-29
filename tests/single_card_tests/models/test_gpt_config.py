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

"""Behavioral tests for ``paddlefleet.models.gpt.gpt_config.GPTConfig``.

These tests target the *derived* and *validating* behavior that runs when a
``GPTConfig`` is constructed -- i.e. the ``TransformerConfig.__post_init__``
normalizations that the GPT config inherits and the field overrides GPTConfig
itself introduces -- rather than trivial "set field, read it back" round-trips
(which would pass even if ``__post_init__`` were deleted).

Expected values are hand-derived from the production sources:
``src/paddlefleet/accuracy_target.py`` (normalize_accuracy_target) and
``src/paddlefleet/transformer/transformer_config.py`` (__post_init__), never by
calling the production normalizer to produce its own expectation.

Importing GPTConfig pulls in paddle transitively; where paddle is not installed
the whole module is honestly skipped (it is not faked as passing).
"""

import unittest
from dataclasses import fields

try:
    from paddlefleet.models.gpt.gpt_config import GPTConfig
    from paddlefleet.transformer.transformer_config import TransformerConfig

    _IMPORT_ERROR = None
except (
    ImportError
) as exc:  # only a genuine missing-dependency, not other errors
    GPTConfig = None
    TransformerConfig = None
    _IMPORT_ERROR = exc

PADDLE_AVAILABLE = _IMPORT_ERROR is None
_SKIP_REASON = (
    "GPTConfig import requires the paddle runtime, which is not installed in "
    f"this environment: {_IMPORT_ERROR!r}. Construction-time normalization and "
    "validation were not executed here."
)


@unittest.skipUnless(PADDLE_AVAILABLE, _SKIP_REASON)
class TestGPTConfigAccuracyTargetNormalization(unittest.TestCase):
    """``use_accuracy_compatible`` is canonicalized in __post_init__ via
    ``normalize_accuracy_target``. This is a tri-state, not a bool: True (and 1)
    collapse to the explicit "megatron"; every falsy spelling to False; unknown
    targets raise. A plain "read the field back" test would miss all of that."""

    def test_default_accuracy_target_is_false(self):
        # Default field value False -> normalize_accuracy_target(False) -> False.
        config = GPTConfig()
        self.assertIs(config.use_accuracy_compatible, False)

    def test_true_normalized_to_megatron(self):
        # ``True`` predates the "hf" target, so it canonicalizes to "megatron".
        config = GPTConfig(use_accuracy_compatible=True)
        self.assertEqual(config.use_accuracy_compatible, "megatron")

    def test_int_one_normalized_to_megatron(self):
        # A YAML scalar 1 means the same as True (``value == 1`` branch).
        config = GPTConfig(use_accuracy_compatible=1)
        self.assertEqual(config.use_accuracy_compatible, "megatron")

    def test_hf_string_preserved(self):
        config = GPTConfig(use_accuracy_compatible="hf")
        self.assertEqual(config.use_accuracy_compatible, "hf")

    def test_megatron_string_preserved(self):
        config = GPTConfig(use_accuracy_compatible="megatron")
        self.assertEqual(config.use_accuracy_compatible, "megatron")

    def test_case_and_whitespace_insensitive(self):
        # strip().lower() maps "  HF " onto the canonical "hf".
        config = GPTConfig(use_accuracy_compatible="  HF ")
        self.assertEqual(config.use_accuracy_compatible, "hf")

    def test_true_word_string_normalized_to_megatron(self):
        # "true" is a _TRUE_WORDS spelling -> "megatron" (not left as the string).
        config = GPTConfig(use_accuracy_compatible="true")
        self.assertEqual(config.use_accuracy_compatible, "megatron")

    def test_false_word_string_normalized_to_false(self):
        # "none"/"off" are _FALSE_WORDS spellings -> the boolean False.
        for spelling in ("none", "off"):
            with self.subTest(spelling=spelling):
                config = GPTConfig(use_accuracy_compatible=spelling)
                self.assertIs(config.use_accuracy_compatible, False)

    def test_unknown_target_raises_valueerror(self):
        # An unknown target must raise rather than silently degrade to defaults.
        with self.assertRaises(ValueError) as ctx:
            GPTConfig(use_accuracy_compatible="bogus")
        self.assertIn("use_accuracy_compatible", str(ctx.exception))

    def test_non_bool_non_str_raises_typeerror(self):
        # 2 is neither falsy, nor True/1, nor a str -> TypeError branch.
        with self.assertRaises(TypeError):
            GPTConfig(use_accuracy_compatible=2)


@unittest.skipUnless(PADDLE_AVAILABLE, _SKIP_REASON)
class TestGPTConfigIndexerLossCoeffNormalization(unittest.TestCase):
    """__post_init__ runs ``float(self.dsa_indexer_loss_coeff or 0.0)`` so the
    stored value is always a float and never None; consumers key on ``> 0``."""

    def test_default_coeff_is_zero_float(self):
        config = GPTConfig()
        self.assertIsInstance(config.dsa_indexer_loss_coeff, float)
        self.assertEqual(config.dsa_indexer_loss_coeff, 0.0)

    def test_none_normalized_to_zero(self):
        # ``None or 0.0`` -> 0.0; None must not survive onto the config object.
        config = GPTConfig(dsa_indexer_loss_coeff=None)
        self.assertIsInstance(config.dsa_indexer_loss_coeff, float)
        self.assertEqual(config.dsa_indexer_loss_coeff, 0.0)

    def test_int_coerced_to_float(self):
        # ``2 or 0.0`` -> 2, then float(2) -> 2.0 (type change is observable).
        config = GPTConfig(dsa_indexer_loss_coeff=2)
        self.assertIsInstance(config.dsa_indexer_loss_coeff, float)
        self.assertEqual(config.dsa_indexer_loss_coeff, 2.0)

    def test_positive_value_preserved(self):
        config = GPTConfig(dsa_indexer_loss_coeff=0.5)
        self.assertEqual(config.dsa_indexer_loss_coeff, 0.5)

    def test_explicit_zero_stays_zero(self):
        # ``0 or 0.0`` -> 0.0 (falsy input still collapses through the ``or``).
        config = GPTConfig(dsa_indexer_loss_coeff=0)
        self.assertIsInstance(config.dsa_indexer_loss_coeff, float)
        self.assertEqual(config.dsa_indexer_loss_coeff, 0.0)


@unittest.skipUnless(PADDLE_AVAILABLE, _SKIP_REASON)
class TestGPTConfigRopeScalingOverride(unittest.TestCase):
    """GPTConfig redeclares ``rope_scaling`` as ``float = 1.0``, overriding the
    inherited ``TransformerConfig`` declaration ``dict = None``. This verifies
    the subclass field override actually changes the effective default."""

    def _field_default(self, cls, name):
        for f in fields(cls):
            if f.name == name:
                return f.default
        self.fail(f"{cls.__name__} has no field {name!r}")

    def test_gpt_overrides_parent_rope_scaling_default(self):
        # Independent expectation from the two source declarations.
        self.assertIsNone(
            self._field_default(TransformerConfig, "rope_scaling")
        )
        self.assertEqual(self._field_default(GPTConfig, "rope_scaling"), 1.0)
        # And the override reaches a constructed instance.
        self.assertEqual(GPTConfig().rope_scaling, 1.0)


@unittest.skipUnless(PADDLE_AVAILABLE, _SKIP_REASON)
class TestGPTConfigTieWordEmbeddingsGuard(unittest.TestCase):
    """__post_init__ guards ``enable_mtp_magic_send`` against a tied vocab table:
    with magic send on, ``tie_word_embeddings=True`` would leave the MTP stage
    with an all-zero, never-synced table, so it must raise. The guard is gated on
    ``enable_mtp_magic_send`` -- observing both the raise and the no-raise proves
    the GPT ``tie_word_embeddings`` field is actually consumed by the guard."""

    def test_magic_send_with_tied_embeddings_raises(self):
        # tie check is the first statement inside the enable_mtp_magic_send block.
        with self.assertRaises(ValueError) as ctx:
            GPTConfig(enable_mtp_magic_send=True, tie_word_embeddings=True)
        self.assertIn("tie_word_embeddings", str(ctx.exception))

    def test_tied_embeddings_without_magic_send_does_not_raise(self):
        # magic send defaults False, so the guard is skipped and construction
        # succeeds; the guard is conditional, not an unconditional ban on ties.
        self.assertFalse(GPTConfig().enable_mtp_magic_send)
        config = GPTConfig(tie_word_embeddings=True)
        self.assertTrue(config.tie_word_embeddings)


if __name__ == "__main__":
    unittest.main()
