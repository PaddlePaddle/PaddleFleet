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

"""Targeted unit tests for TransformerConfig.multimax field validation."""

import os
import sys
import unittest
import warnings

# Ensure the in-repo `src/paddlefleet` is loaded ahead of any installed
# (potentially stale) site-packages copy.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, _REPO_ROOT)


class TestMultimaxConfig(unittest.TestCase):
    """TransformerConfig.multimax accepts only None / 'lm_head' / 'attn' / 'all'.

    Empty string is silently coerced to None so YAML configs that leave
    the field as `multimax:` (parsed as empty string) behave like unset.
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
        self.assertIsNone(cfg.multimax)

    def test_lm_head_accepted(self):
        cfg = self._build(multimax="lm_head")
        self.assertEqual(cfg.multimax, "lm_head")

    def test_attn_accepted_with_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = self._build(multimax="attn")
        self.assertEqual(cfg.multimax, "attn")
        # 'attn' branch is unimplemented -> a banner warning must mention it.
        msgs = [str(x.message) for x in w]
        self.assertTrue(
            any("not implemented" in m and "attn" in m for m in msgs),
            f"expected unimplemented-attn warning, got: {msgs}",
        )

    def test_all_accepted_with_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = self._build(multimax="all")
        self.assertEqual(cfg.multimax, "all")
        msgs = [str(x.message) for x in w]
        self.assertTrue(
            any("not implemented" in m and "all" in m for m in msgs),
            f"expected unimplemented-all warning, got: {msgs}",
        )

    def test_empty_string_coerced_to_none(self):
        """YAML `multimax:` parses to '' -- must be canonicalized to None."""
        cfg = self._build(multimax="")
        self.assertIsNone(cfg.multimax)

    def test_invalid_value_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._build(multimax="not_a_real_mode")
        self.assertIn("multimax must be one of", str(ctx.exception))
        self.assertIn("not_a_real_mode", str(ctx.exception))

    def test_grep_friendly_banner_emitted(self):
        """[MULTIMAX-CONFIG] tag must be present so operators can grep train logs."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._build(multimax="lm_head")
        msgs = [str(x.message) for x in w]
        self.assertTrue(
            any("[MULTIMAX-CONFIG]" in m for m in msgs),
            f"expected [MULTIMAX-CONFIG] banner, got: {msgs}",
        )


class TestMtpWindowSizeField(unittest.TestCase):
    """The mtp_window_size / mtp_reuse_last_layer fields must be accepted."""

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

    def test_mtp_window_size_default_none(self):
        cfg = self._build()
        self.assertIsNone(cfg.mtp_window_size)

    def test_mtp_window_size_int_accepted(self):
        cfg = self._build(mtp_window_size=512)
        self.assertEqual(cfg.mtp_window_size, 512)

    def test_mtp_reuse_last_layer_default_false(self):
        cfg = self._build()
        self.assertFalse(cfg.mtp_reuse_last_layer)

    def test_mtp_reuse_last_layer_true_accepted(self):
        cfg = self._build(mtp_reuse_last_layer=True)
        self.assertTrue(cfg.mtp_reuse_last_layer)


if __name__ == "__main__":
    unittest.main()
