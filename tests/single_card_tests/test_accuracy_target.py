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
"""Tests for ``paddlefleet.accuracy_target``.

``use_accuracy_compatible`` is a tri-state: ``False`` (throughput kernels),
``"megatron"`` (also spelled ``True``, its historical meaning) and ``"hf"``. The
invariants that the rest of the codebase depends on are:

* every non-default value is **truthy**, so the many
  ``if config.use_accuracy_compatible:`` sites keep meaning "in some alignment
  mode" without being touched;
* ``True`` keeps meaning Megatron, so existing configs and checkpoints do not
  change behaviour;
* only ``"hf"`` makes :func:`targets_hf` true, and it is the sole discriminator
  at the sites where the two references demand different arithmetic;
* an unknown target raises instead of degrading to the default kernels, which
  would turn a typo into a slow run aligned with nothing.
"""

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

from paddlefleet.accuracy_target import (
    ACCURACY_TARGET_HF,
    ACCURACY_TARGET_MEGATRON,
    ACCURACY_TARGETS,
    normalize_accuracy_target,
    targets_hf,
)


class TestNormalizeAccuracyTargetOff(unittest.TestCase):
    """Every spelling of "off" collapses to the single canonical ``False``."""

    def test_false_stays_false(self):
        """``False`` is returned as the bool, not as "" or 0."""
        result = normalize_accuracy_target(False)
        self.assertIs(result, False)

    def test_none_is_off(self):
        """A missing value (e.g. ``"use_accuracy_compatible": null``) is off."""
        self.assertIs(normalize_accuracy_target(None), False)

    def test_empty_string_is_off(self):
        """An empty YAML scalar is off rather than an unknown target."""
        self.assertIs(normalize_accuracy_target(""), False)

    def test_zero_is_off(self):
        """A YAML/argparse ``0`` is off."""
        self.assertIs(normalize_accuracy_target(0), False)

    def test_bool_words_are_off(self):
        """Config layers stringify booleans; those spellings must not raise."""
        for word in ["false", "False", "FALSE", "no", "off", "none", "null"]:
            with self.subTest(word=word):
                self.assertIs(normalize_accuracy_target(word), False)

    def test_off_values_are_falsy(self):
        """The result is usable directly in ``if config.use_...:``."""
        for raw in [False, None, "", 0, "false", "off"]:
            with self.subTest(raw=raw):
                self.assertFalse(bool(normalize_accuracy_target(raw)))


class TestNormalizeAccuracyTargetMegatron(unittest.TestCase):
    """``True`` predates the "hf" target and must keep meaning Megatron."""

    def test_true_becomes_megatron(self):
        """The stored value names its reference instead of staying a bare bool."""
        self.assertEqual(
            normalize_accuracy_target(True), ACCURACY_TARGET_MEGATRON
        )

    def test_one_becomes_megatron(self):
        """A YAML scalar ``1`` means the same thing as ``true``."""
        self.assertEqual(normalize_accuracy_target(1), ACCURACY_TARGET_MEGATRON)

    def test_bool_words_become_megatron(self):
        """Stringified booleans from the CLI/YAML layer resolve to Megatron."""
        for word in ["true", "True", "TRUE", "yes", "on"]:
            with self.subTest(word=word):
                self.assertEqual(
                    normalize_accuracy_target(word), ACCURACY_TARGET_MEGATRON
                )

    def test_explicit_megatron_is_idempotent(self):
        """Normalizing an already-canonical value is a no-op."""
        once = normalize_accuracy_target("megatron")
        self.assertEqual(once, ACCURACY_TARGET_MEGATRON)
        self.assertEqual(normalize_accuracy_target(once), once)


class TestNormalizeAccuracyTargetHF(unittest.TestCase):
    """The ``"hf"`` target, including the spellings a config file may produce."""

    def test_explicit_hf(self):
        self.assertEqual(normalize_accuracy_target("hf"), ACCURACY_TARGET_HF)

    def test_case_and_whitespace_tolerated(self):
        """``use_accuracy_compatible: HF`` and stray spaces still resolve."""
        for raw in ["HF", "Hf", " hf", "hf ", "  hf  "]:
            with self.subTest(raw=raw):
                self.assertEqual(
                    normalize_accuracy_target(raw), ACCURACY_TARGET_HF
                )

    def test_hf_is_idempotent(self):
        once = normalize_accuracy_target("HF")
        self.assertEqual(normalize_accuracy_target(once), ACCURACY_TARGET_HF)


class TestNormalizeAccuracyTargetRejects(unittest.TestCase):
    """A typo must raise, not silently select the default kernels."""

    def test_unknown_target_raises_value_error(self):
        for bad in ["megatron_lm", "huggingface", "torch", "megatron-lm", "mg"]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                normalize_accuracy_target(bad)

    def test_error_message_lists_the_valid_targets(self):
        """The message has to be actionable without reading the source."""
        with self.assertRaisesRegex(ValueError, "megatron.*hf|hf.*megatron"):
            normalize_accuracy_target("huggingface")

    def test_non_bool_non_str_raises_type_error(self):
        for bad in [2, 1.5, 3.0, object()]:
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                normalize_accuracy_target(bad)


class TestTargetsHF(unittest.TestCase):
    """``targets_hf`` is the only discriminator between the two references."""

    def test_true_only_for_hf(self):
        self.assertTrue(targets_hf(ACCURACY_TARGET_HF))

    def test_false_for_every_other_canonical_value(self):
        for value in [False, ACCURACY_TARGET_MEGATRON]:
            with self.subTest(value=value):
                self.assertFalse(targets_hf(value))

    def test_false_for_raw_true(self):
        """A caller that threaded a bare ``True`` gets Megatron, not HF.

        This is what keeps the historical two-argument calls (and the existing
        ``*_use_accuracy_compatible`` tests, which pass ``True``) on the Megatron
        arithmetic after the module-level env flag was removed.
        """
        self.assertFalse(targets_hf(True))

    def test_accepts_the_raw_field_value(self):
        """Call sites pass ``config.use_accuracy_compatible`` without normalizing."""
        self.assertTrue(targets_hf(normalize_accuracy_target("HF")))
        self.assertFalse(targets_hf(normalize_accuracy_target(True)))


class TestTriStateContract(unittest.TestCase):
    """Invariants the rest of the codebase relies on."""

    def test_both_targets_are_truthy(self):
        """So the ~73 untouched ``if config.use_accuracy_compatible:`` still work."""
        for target in ACCURACY_TARGETS:
            with self.subTest(target=target):
                self.assertTrue(bool(target))

    def test_exactly_two_targets(self):
        self.assertEqual(
            set(ACCURACY_TARGETS),
            {ACCURACY_TARGET_MEGATRON, ACCURACY_TARGET_HF},
        )

    def test_normalize_output_is_always_canonical(self):
        """Output is ``False`` or a member of ``ACCURACY_TARGETS``, nothing else."""
        for raw in [
            False,
            None,
            "",
            0,
            True,
            1,
            "true",
            "megatron",
            "hf",
            "HF",
        ]:
            with self.subTest(raw=raw):
                out = normalize_accuracy_target(raw)
                self.assertTrue(out is False or out in ACCURACY_TARGETS)


if __name__ == "__main__":
    unittest.main()
