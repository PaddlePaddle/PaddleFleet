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
import unittest

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)


class TestTransformerConfigHashRoutingValidation(unittest.TestCase):
    """Cover hash routing validation in TransformerConfig.__post_init__
    (lines 990-1027)."""

    def _make_config_kwargs(self, **overrides):
        """Minimal kwargs to construct a valid TransformerConfig."""
        defaults = {
            "hidden_size": 64,
            "num_attention_heads": 2,
            "intermediate_size": 256,
            "num_hidden_layers": 8,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "moe_n_hash_layers": 1,
            "actual_vocab_size": 128,
            "scoring_func": "softmax",
        }
        defaults.update(overrides)
        return defaults

    def test_hash_missing_actual_vocab_size_raises(self):
        """Lines 990-991: actual_vocab_size is None with moe_n_hash_layers > 0."""
        from paddlefleet.transformer.transformer_config import TransformerConfig

        kwargs = self._make_config_kwargs()
        del kwargs["actual_vocab_size"]
        with self.assertRaises(ValueError):
            TransformerConfig(**kwargs)

    def test_hash_negative_actual_vocab_size_raises(self):
        """Lines 995-996: actual_vocab_size <= 0."""
        from paddlefleet.transformer.transformer_config import TransformerConfig

        with self.assertRaises(ValueError):
            TransformerConfig(**self._make_config_kwargs(actual_vocab_size=-1))

    def test_hash_too_many_hash_layers_raises(self):
        """Lines 1000-1001: moe_n_hash_layers > num_hidden_layers."""
        from paddlefleet.transformer.transformer_config import TransformerConfig

        with self.assertRaises(ValueError):
            TransformerConfig(
                **self._make_config_kwargs(
                    moe_n_hash_layers=100, num_hidden_layers=8
                )
            )

    def test_hash_invalid_scoring_func_raises(self):
        """Lines 1005-1006: scoring_func not in allowed set."""
        from paddlefleet.transformer.transformer_config import TransformerConfig

        with self.assertRaises(ValueError):
            TransformerConfig(**self._make_config_kwargs(scoring_func="relu"))

    def test_hash_no_topk_raises(self):
        """Lines 1011-1015: num_experts_per_tok is None or <= 0."""
        from paddlefleet.transformer.transformer_config import TransformerConfig

        with self.assertRaises(ValueError):
            TransformerConfig(**self._make_config_kwargs(num_experts_per_tok=0))

    def test_hash_too_few_routed_experts_raises(self):
        """Lines 1019-1023: n_routed_experts < num_experts_per_tok."""
        from paddlefleet.transformer.transformer_config import TransformerConfig

        with self.assertRaises(ValueError):
            TransformerConfig(
                **self._make_config_kwargs(
                    n_routed_experts=1, num_experts_per_tok=2
                )
            )

    def test_hash_valid_config_passes(self):
        """Valid hash routing config should not raise."""
        from paddlefleet.transformer.transformer_config import TransformerConfig

        config = TransformerConfig(**self._make_config_kwargs())
        self.assertEqual(config.moe_n_hash_layers, 1)
        self.assertEqual(config.actual_vocab_size, 128)

    def test_no_hash_layers_skips_validation(self):
        """moe_n_hash_layers=0 should skip hash validation entirely."""
        from paddlefleet.transformer.transformer_config import TransformerConfig

        # This would fail hash validation if it were checked, but
        # moe_n_hash_layers=0 means no hash routing, so no validation.
        config = TransformerConfig(
            hidden_size=64,
            num_attention_heads=2,
            intermediate_size=256,
            num_hidden_layers=8,
            moe_n_hash_layers=0,
        )
        self.assertEqual(config.moe_n_hash_layers, 0)


if __name__ == "__main__":
    unittest.main()
