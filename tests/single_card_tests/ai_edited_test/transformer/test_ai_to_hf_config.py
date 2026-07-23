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
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import unittest

from paddlefleet.transformer.transformer_config import TransformerConfig


def _make(cls=TransformerConfig, **overrides):
    defaults = {
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
    }
    defaults.update(overrides)
    return cls(**defaults)


# --- helper source objects for the value-source resolution branch ----------
class _DictSource:
    """Source object exposing a ``to_dict`` method."""

    def __init__(self, data):
        self._data = dict(data)

    def to_dict(self):
        return dict(self._data)


class _PlainSource:
    """Source object with plain attributes (resolved via ``vars``)."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# --- config subclasses exercising the export hooks -------------------------
class _RuleConfig(TransformerConfig):
    """Rename-only + rename+value + same-name value-convert rules."""

    hf_export_rules = {
        "fleet_flag": "hf_flag",  # rename only (str spec)
        "fleet_scale": ("hf_scale", lambda v: v * 10),  # rename + value
    }


class _ClobberConfig(TransformerConfig):
    """A rename target that collides with an existing raw key name.

    ``src_field`` renames to ``shared``; the raw config also carries its own
    ``shared`` key. The first (rename) pass must win over the passthrough pass.
    """

    hf_export_rules = {
        "src_field": "shared",
    }


class _HookConfig(TransformerConfig):
    """Subclass overriding both structural hooks."""

    def _hf_export_postprocess(self, raw, out):
        out["added_by_hook"] = raw.get("hidden_size", 0) + 1

    def _hf_export_whitelist(self, raw):
        return {"added_by_hook", "keep"}


def silu(x):  # named callable for the hidden_act normalization branch
    return x


class TestSourceResolution(unittest.TestCase):
    """Value-source resolution branches of ``to_hf_config``."""

    def test_source_none_uses_fields(self):
        # No _hf_export_source: raw is built from dataclass fields + vars.
        cfg = _make()
        out = cfg.to_hf_config()
        self.assertIsInstance(out, dict)
        self.assertEqual(out["hidden_size"], 128)
        self.assertEqual(out["num_attention_heads"], 4)

    def test_source_with_to_dict(self):
        cfg = _make()
        cfg._hf_export_source = _DictSource({"a": 1, "b": 2})
        out = cfg.to_hf_config()
        self.assertEqual(out["a"], 1)
        self.assertEqual(out["b"], 2)
        # Provider-derived fields must NOT leak in when a source is set.
        self.assertNotIn("hidden_size", out)

    def test_source_is_dict(self):
        cfg = _make()
        cfg._hf_export_source = {"x": 9}
        out = cfg.to_hf_config()
        self.assertEqual(out, {"x": 9})

    def test_source_is_plain_object(self):
        cfg = _make()
        cfg._hf_export_source = _PlainSource(p=7, q="v")
        out = cfg.to_hf_config()
        self.assertEqual(out["p"], 7)
        self.assertEqual(out["q"], "v")

    def test_source_key_is_stripped(self):
        cfg = _make()
        cfg._hf_export_source = {"_hf_export_source": "junk", "ok": 1}
        out = cfg.to_hf_config()
        self.assertNotIn("_hf_export_source", out)
        self.assertEqual(out["ok"], 1)


class TestRules(unittest.TestCase):
    """``hf_export_rules`` rename / value-convert branches."""

    def test_rename_only(self):
        cfg = _make(cls=_RuleConfig)
        cfg._hf_export_source = {"fleet_flag": True}
        out = cfg.to_hf_config()
        self.assertEqual(out["hf_flag"], True)
        self.assertNotIn("fleet_flag", out)

    def test_rename_and_value_convert(self):
        cfg = _make(cls=_RuleConfig)
        cfg._hf_export_source = {"fleet_scale": 3}
        out = cfg.to_hf_config()
        self.assertEqual(out["hf_scale"], 30)
        self.assertNotIn("fleet_scale", out)

    def test_clobber_protection(self):
        # First (rename) pass must win over the passthrough pass when a rename
        # target collides with an existing raw key.
        cfg = _make(cls=_ClobberConfig)
        cfg._hf_export_source = {"src_field": "renamed", "shared": "original"}
        out = cfg.to_hf_config()
        self.assertEqual(out["shared"], "renamed")
        self.assertNotIn("src_field", out)


class TestNormalization(unittest.TestCase):
    """Generic value normalization for un-ruled fields."""

    def test_hidden_act_callable_to_name(self):
        cfg = _make()
        cfg._hf_export_source = {"hidden_act": silu}
        out = cfg.to_hf_config()
        self.assertEqual(out["hidden_act"], "silu")

    def test_hidden_act_string_untouched(self):
        cfg = _make()
        cfg._hf_export_source = {"hidden_act": "gelu"}
        out = cfg.to_hf_config()
        self.assertEqual(out["hidden_act"], "gelu")

    def test_params_dtype_prefix_stripped(self):
        cfg = _make()
        cfg._hf_export_source = {"params_dtype": "paddle.bfloat16"}
        out = cfg.to_hf_config()
        self.assertEqual(out["params_dtype"], "bfloat16")

    def test_params_dtype_none_untouched(self):
        cfg = _make()
        cfg._hf_export_source = {"params_dtype": None}
        out = cfg.to_hf_config()
        self.assertIsNone(out["params_dtype"])


class TestHooks(unittest.TestCase):
    """Structural postprocess + whitelist subclass hooks."""

    def test_postprocess_and_whitelist(self):
        cfg = _make(cls=_HookConfig)
        cfg._hf_export_source = {"hidden_size": 10, "keep": "yes", "drop": "no"}
        out = cfg.to_hf_config()
        self.assertEqual(out["added_by_hook"], 11)
        self.assertEqual(out["keep"], "yes")
        self.assertNotIn("drop", out)

    def test_whitelist_none_keeps_all(self):
        cfg = _make()
        cfg._hf_export_source = {"a": 1, "b": 2}
        out = cfg.to_hf_config()
        self.assertEqual(out["a"], 1)
        self.assertEqual(out["b"], 2)


if __name__ == "__main__":
    unittest.main()
