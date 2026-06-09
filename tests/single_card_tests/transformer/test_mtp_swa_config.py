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

"""Targeted unit tests for MultiTokenPredictionLayer._build_mtp_transformer_config.

Covers the branches added by the SWA-on-MTP feature:
- mtp_window_size=None  -> sliding_window=None (global attention)
- mtp_window_size=int   -> sliding_window=(int, 0) (causal SWA)

Plus the LayerSpec.extra_kwargs precedence guard and the is_mtp_layer marker.

The test calls the method as an unbound function with a minimal mock `self`,
so we don't need to construct a real MultiTokenPredictionLayer (which would
require a fully-initialized fleet/parallel context).
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)
# Ensure the in-repo `src/paddlefleet` is loaded ahead of any installed
# (potentially stale) site-packages copy, so we test the actual branch code.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))


def _make_mock_self(mtp_window_size, *, with_extra_kwargs=True):
    """Build a minimal `self` object exposing only what the method touches."""
    config = SimpleNamespace(
        mtp_window_size=mtp_window_size,
        sliding_window=(
            128,
            0,
        ),  # backbone uses SWA; MTP override should be independent
        window_attn_skip_freq=4,  # backbone has skip pattern; MTP must clear it
        is_mtp_layer=False,  # backbone marker; MTP must override to True
        # extra fields that copy.copy may carry; harmless
        hidden_size=128,
        num_attention_heads=4,
    )
    tl_spec = MagicMock()
    if with_extra_kwargs:
        tl_spec.extra_kwargs = {"config": "BACKBONE_CONFIG_PLACEHOLDER"}
    else:
        # Spec without extra_kwargs attribute -> precedence guard must skip silently.
        del tl_spec.extra_kwargs
    sublayers_spec = SimpleNamespace(transformer_layer=tl_spec)
    return SimpleNamespace(
        config=config,
        sublayers_spec=sublayers_spec,
        layer_number=7,
    )


class TestBuildMtpTransformerConfig(unittest.TestCase):
    """Unit tests for the _build_mtp_transformer_config helper."""

    @classmethod
    def setUpClass(cls):
        from paddlefleet.transformer.multi_token_prediction import (
            MultiTokenPredictionLayer,
        )

        # Wrap as staticmethod so descriptor binding doesn't inject `self`
        # (test instance) ahead of the mock object we pass in.
        cls._method = staticmethod(
            MultiTokenPredictionLayer._build_mtp_transformer_config
        )

    def test_window_size_none_yields_global_attention(self):
        """mtp_window_size=None -> MTP layer uses global attention (sliding_window=None)."""
        mock_self = _make_mock_self(mtp_window_size=None)
        cfg = self._method(mock_self)
        self.assertIsNone(cfg.sliding_window)
        # Other invariants still hold
        self.assertIsNone(cfg.window_attn_skip_freq)
        self.assertTrue(cfg.is_mtp_layer)

    def test_window_size_int_yields_causal_swa_tuple(self):
        """mtp_window_size=N -> sliding_window=(N, 0) (causal SWA, only past)."""
        mock_self = _make_mock_self(mtp_window_size=512)
        cfg = self._method(mock_self)
        self.assertEqual(cfg.sliding_window, (512, 0))
        self.assertIsNone(cfg.window_attn_skip_freq)
        self.assertTrue(cfg.is_mtp_layer)

    def test_window_attn_skip_freq_force_disabled(self):
        """Backbone's window_attn_skip_freq must be cleared so MTP never lands on a 'skip' slot."""
        mock_self = _make_mock_self(mtp_window_size=256)
        # Backbone has skip_freq=4, MTP override must be None
        self.assertEqual(mock_self.config.window_attn_skip_freq, 4)
        cfg = self._method(mock_self)
        self.assertIsNone(cfg.window_attn_skip_freq)

    def test_is_mtp_layer_marker_set(self):
        """The is_mtp_layer marker must always be True regardless of backbone value."""
        mock_self = _make_mock_self(mtp_window_size=None)
        self.assertFalse(mock_self.config.is_mtp_layer)
        cfg = self._method(mock_self)
        self.assertTrue(cfg.is_mtp_layer)

    def test_extra_kwargs_precedence_override(self):
        """LayerSpec.extra_kwargs['config'] must be replaced with the new mtp_config.

        This guards against the Paddle precedence bug: passing config=mtp_config
        to build_spec_layer alone is silently overridden by extra_kwargs['config'],
        so we must explicitly patch extra_kwargs.
        """
        mock_self = _make_mock_self(mtp_window_size=128)
        cfg = self._method(mock_self)
        ek = mock_self.sublayers_spec.transformer_layer.extra_kwargs
        # extra_kwargs['config'] must now be the freshly-built mtp_config (not the placeholder)
        self.assertIs(ek["config"], cfg)
        # And is_mtp_layer must be threaded through extra_kwargs too
        self.assertTrue(ek["is_mtp_layer"])

    def test_extra_kwargs_absent_does_not_raise(self):
        """If the spec has no extra_kwargs attribute, the override block is a silent no-op."""
        mock_self = _make_mock_self(mtp_window_size=64, with_extra_kwargs=False)
        # Should not raise.
        cfg = self._method(mock_self)
        self.assertEqual(cfg.sliding_window, (64, 0))

    def test_backbone_config_not_mutated(self):
        """copy.copy must produce a separate object so backbone config stays intact."""
        mock_self = _make_mock_self(mtp_window_size=256)
        original_sliding = mock_self.config.sliding_window
        original_skip = mock_self.config.window_attn_skip_freq
        cfg = self._method(mock_self)
        # mtp_config diverges
        self.assertNotEqual(cfg.sliding_window, original_sliding)
        # backbone untouched
        self.assertEqual(mock_self.config.sliding_window, original_sliding)
        self.assertEqual(mock_self.config.window_attn_skip_freq, original_skip)

    def test_window_size_zero_is_treated_as_int(self):
        """mtp_window_size=0 should map to sliding_window=(0, 0), not be coerced to None.

        This guards against an `if mtp_window_size:` mistake (0 is falsy in Python).
        """
        mock_self = _make_mock_self(mtp_window_size=0)
        cfg = self._method(mock_self)
        self.assertEqual(cfg.sliding_window, (0, 0))


if __name__ == "__main__":
    unittest.main()
