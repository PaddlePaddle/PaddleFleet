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

"""Targeted unit tests for TransformerConfig.apply_multimax field validation."""

import unittest
import warnings


class TestMultimaxConfig(unittest.TestCase):
    """TransformerConfig.apply_multimax accepts None or a list of submodule names.

    Mirrors Megatron's ``recompute_modules`` style. Currently the only
    implemented entry is ``"lm_head"``; ``"attention"`` is reserved and
    triggers a not-implemented warning.

    YAML/JSON ergonomics:
    - unset key, ``null``, empty string, or empty list ``[]`` are all
      coerced to ``None`` (feature disabled).
    - a bare string (``apply_multimax: lm_head``) is auto-promoted to a
      single-element list for back-compat with older configs.
    """

    @classmethod
    def setUpClass(cls):
        from paddlefleet.transformer.transformer_config import TransformerConfig

        cls.TransformerConfig = TransformerConfig

    def _build(self, **overrides):
        defaults = {
            "num_hidden_layers": 4,
            "hidden_size": 64,
            "num_attention_heads": 4,
        }
        defaults.update(overrides)
        return self.TransformerConfig(**defaults)

    def test_default_is_none(self):
        cfg = self._build()
        self.assertIsNone(cfg.apply_multimax)

    def test_lm_head_list_accepted(self):
        cfg = self._build(apply_multimax=["lm_head"])
        self.assertEqual(cfg.apply_multimax, ["lm_head"])

    def test_bare_string_promoted_to_list(self):
        """Back-compat: ``apply_multimax: lm_head`` -> ``["lm_head"]``."""
        cfg = self._build(apply_multimax="lm_head")
        self.assertEqual(cfg.apply_multimax, ["lm_head"])

    def test_attention_accepted_with_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = self._build(apply_multimax=["attention"])
        self.assertEqual(cfg.apply_multimax, ["attention"])
        # 'attention' branch is unimplemented -> a banner warning must mention it.
        msgs = [str(x.message) for x in w]
        self.assertTrue(
            any("not implemented" in m and "attention" in m for m in msgs),
            f"expected unimplemented-attention warning, got: {msgs}",
        )

    def test_combined_lm_head_and_attention(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = self._build(apply_multimax=["lm_head", "attention"])
        self.assertEqual(cfg.apply_multimax, ["lm_head", "attention"])
        msgs = [str(x.message) for x in w]
        self.assertTrue(
            any("not implemented" in m and "attention" in m for m in msgs),
            f"expected unimplemented-attention warning, got: {msgs}",
        )

    def test_empty_string_coerced_to_none(self):
        """YAML `apply_multimax:` parses to '' -- must be canonicalized to None."""
        cfg = self._build(apply_multimax="")
        self.assertIsNone(cfg.apply_multimax)

    def test_empty_list_coerced_to_none(self):
        """YAML `apply_multimax: []` -- must be canonicalized to None."""
        cfg = self._build(apply_multimax=[])
        self.assertIsNone(cfg.apply_multimax)

    def test_invalid_entry_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._build(apply_multimax=["not_a_real_mode"])
        self.assertIn(
            "apply_multimax entries must each be one of", str(ctx.exception)
        )
        self.assertIn("not_a_real_mode", str(ctx.exception))

    def test_invalid_type_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._build(apply_multimax=123)
        self.assertIn(
            "apply_multimax must be None or a list[str]", str(ctx.exception)
        )

    def test_grep_friendly_banner_emitted(self):
        """[MULTIMAX-CONFIG] tag must be present so operators can grep train logs."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._build(apply_multimax=["lm_head"])
        msgs = [str(x.message) for x in w]
        self.assertTrue(
            any("[MULTIMAX-CONFIG]" in m for m in msgs),
            f"expected [MULTIMAX-CONFIG] banner, got: {msgs}",
        )

    def test_none_explicit_no_unimplemented_warning(self):
        """apply_multimax=None explicitly: no warnings fire (default disabled);
        the [MULTIMAX-CONFIG] banner is only emitted when the feature is on."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._build(apply_multimax=None)
        msgs = [str(x.message) for x in w]
        self.assertFalse(
            any("not implemented" in m for m in msgs),
            f"unexpected unimplemented warning for None: {msgs}",
        )
        self.assertFalse(
            any("[MULTIMAX-CONFIG]" in m for m in msgs),
            f"unexpected [MULTIMAX-CONFIG] banner for default None: {msgs}",
        )

    def test_invalid_entry_message_lists_choices(self):
        """ValueError message must enumerate the valid options so users can
        self-correct without reading the code."""
        with self.assertRaises(ValueError) as ctx:
            self._build(apply_multimax=["lm-head"])  # hyphen, not underscore
        msg = str(ctx.exception)
        for choice in ("lm_head", "attention"):
            self.assertIn(choice, msg, f"missing choice {choice!r} in: {msg}")
        self.assertIn("lm-head", msg, "offending value not echoed")

    def test_empty_string_does_not_warn_unimplemented(self):
        """Empty string is canonicalized to None; must not emit an
        'attention not implemented' warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = self._build(apply_multimax="")
        self.assertIsNone(cfg.apply_multimax)
        msgs = [str(x.message) for x in w]
        self.assertFalse(
            any("not implemented" in m for m in msgs),
            f"unexpected unimplemented warning for empty string: {msgs}",
        )


if __name__ == "__main__":
    unittest.main()
