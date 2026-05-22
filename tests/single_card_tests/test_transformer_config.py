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

import unittest

import paddle

from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer.transformer_config import TransformerConfig

strategy = paddle.distributed.fleet.DistributedStrategy()
initialize_fleet(strategy=strategy)


class TestMoeLayerFreqAndFirstKDenseReplace(unittest.TestCase):
    """Tests for the moe_layer_freq / first_k_dense_replace logic in TransformerConfig.__post_init__."""

    def test_both_none_defaults_moe_layer_freq_to_1(self):
        """When both first_k_dense_replace and moe_layer_freq are None, moe_layer_freq defaults to 1."""
        config = TransformerConfig(
            first_k_dense_replace=None,
            moe_layer_freq=None,
            num_hidden_layers=12,
        )
        self.assertEqual(config.moe_layer_freq, 1)

    def test_first_k_dense_replace_with_list_moe_layer_freq_raises(self):
        """When first_k_dense_replace is set and moe_layer_freq is a list (not int), should raise ValueError."""
        with self.assertRaises(ValueError):
            TransformerConfig(
                first_k_dense_replace=2,
                moe_layer_freq=[1, 0, 1, 0],
                num_hidden_layers=4,
            )

    def test_first_k_dense_replace_with_int_moe_layer_freq(self):
        """When first_k_dense_replace is set and moe_layer_freq is an int,
        it should generate a pattern based on the frequency."""
        config = TransformerConfig(
            first_k_dense_replace=2,
            moe_layer_freq=2,
            num_hidden_layers=8,
        )
        # first 2 layers are dense (0), remaining layers follow pattern: 1 if (i % 2 == 0) else 0
        # Pattern for range(8): i=0->1, i=1->0, i=2->1, i=3->0, i=4->1, i=5->0, i=6->1, i=7->0
        expected = [0, 0, 0, 1, 0, 1, 0, 1]
        self.assertEqual(config.moe_layer_freq, expected)

    def test_first_k_dense_replace_with_moe_layer_freq_none(self):
        """When first_k_dense_replace is set and moe_layer_freq is None,
        the remaining layers should all be MoE (all 1s)."""
        config = TransformerConfig(
            first_k_dense_replace=3,
            moe_layer_freq=None,
            num_hidden_layers=8,
        )
        # both-None check won't trigger since first_k_dense_replace is set,
        # moe_layer_freq stays None (falsy).
        # else branch: moe_layer_pattern = [1] * (8 - 3) = [1, 1, 1, 1, 1]
        # final = [0, 0, 0] + [1, 1, 1, 1, 1]
        expected = [0, 0, 0, 1, 1, 1, 1, 1]
        self.assertEqual(config.moe_layer_freq, expected)

    def test_first_k_dense_replace_with_moe_layer_freq_zero(self):
        """When first_k_dense_replace is set and moe_layer_freq is 0 (falsy int),
        the pattern should fall into the else branch producing all 1s for non-dense layers."""
        config = TransformerConfig(
            first_k_dense_replace=4,
            moe_layer_freq=0,
            num_hidden_layers=10,
        )
        # moe_layer_freq=0 is falsy, so moe_layer_pattern = [1] * (10 - 4) = [1, 1, 1, 1, 1, 1]
        expected = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        self.assertEqual(config.moe_layer_freq, expected)

    def test_first_k_dense_replace_with_moe_layer_freq_3(self):
        """When first_k_dense_replace is set with moe_layer_freq=3,
        every 3rd layer (index % 3 == 0) should be MoE."""
        config = TransformerConfig(
            first_k_dense_replace=1,
            moe_layer_freq=3,
            num_hidden_layers=7,
        )
        # first 1 layer is dense
        # Pattern for range(7): i=0->1, i=1->0, i=2->0, i=3->1, i=4->0, i=5->0, i=6->1
        expected = [0, 0, 0, 1, 0, 0, 1]
        self.assertEqual(config.moe_layer_freq, expected)

    def test_first_k_dense_replace_equals_num_hidden_layers(self):
        """Edge case: first_k_dense_replace equals num_hidden_layers,
        all layers should be dense (all 0s)."""
        config = TransformerConfig(
            first_k_dense_replace=6,
            moe_layer_freq=0,
            num_hidden_layers=6,
        )
        # moe_layer_freq=0 is falsy, so moe_layer_pattern = [1] * (6 - 6) = []
        expected = [0, 0, 0, 0, 0, 0]
        self.assertEqual(config.moe_layer_freq, expected)

    def test_only_moe_layer_freq_int_no_first_k_dense(self):
        """When only moe_layer_freq is set (as int) and first_k_dense_replace is None,
        moe_layer_freq should remain as the integer value."""
        config = TransformerConfig(
            first_k_dense_replace=None,
            moe_layer_freq=2,
            num_hidden_layers=8,
        )
        self.assertEqual(config.moe_layer_freq, 2)

    def test_only_first_k_dense_replace_no_moe_layer_freq(self):
        """When first_k_dense_replace is set and moe_layer_freq is not specified (defaults to None),
        should generate the correct pattern."""
        config = TransformerConfig(
            first_k_dense_replace=2,
            num_hidden_layers=6,
        )
        # moe_layer_freq=None => both-None check won't trigger because first_k_dense_replace is set
        # After the None-None check, moe_layer_freq is still None
        # first_k_dense_replace is truthy => enter the block
        # moe_layer_freq is None (falsy) => moe_layer_pattern = [1] * (6-2) = [1,1,1,1]
        expected = [0, 0, 1, 1, 1, 1]
        self.assertEqual(config.moe_layer_freq, expected)

    def test_first_k_dense_replace_1_moe_layer_freq_1(self):
        """first_k_dense_replace=1, moe_layer_freq=1: first layer dense, rest all MoE."""
        config = TransformerConfig(
            first_k_dense_replace=1,
            moe_layer_freq=1,
            num_hidden_layers=5,
        )
        # moe_layer_freq=1 is truthy int, pattern: 1 if (i % 1 == 0) else 0 => all 1s
        expected = [0, 1, 1, 1, 1]
        self.assertEqual(config.moe_layer_freq, expected)


class TestRoutedScalingFactorConfig(unittest.TestCase):
    """Tests for the routed_scaling_factor and routed_scaling_factor_learnable fields
    in TransformerConfig."""

    def test_routed_scaling_factor_default_is_1(self):
        """routed_scaling_factor defaults to 1.0 when not specified."""
        config = TransformerConfig(num_hidden_layers=4)
        self.assertAlmostEqual(config.routed_scaling_factor, 1.0)

    def test_routed_scaling_factor_learnable_default_is_false(self):
        """routed_scaling_factor_learnable defaults to False when not specified."""
        config = TransformerConfig(num_hidden_layers=4)
        self.assertFalse(config.routed_scaling_factor_learnable)

    def test_routed_scaling_factor_float(self):
        """routed_scaling_factor accepts a float value (e.g., 2.5 for DeepSeek-V3)."""
        config = TransformerConfig(
            num_hidden_layers=4, routed_scaling_factor=2.5
        )
        self.assertAlmostEqual(config.routed_scaling_factor, 2.5)

    def test_routed_scaling_factor_learnable_true(self):
        """routed_scaling_factor_learnable can be set to True."""
        config = TransformerConfig(
            num_hidden_layers=4,
            routed_scaling_factor=2.5,
            routed_scaling_factor_learnable=True,
        )
        self.assertAlmostEqual(config.routed_scaling_factor, 2.5)
        self.assertTrue(config.routed_scaling_factor_learnable)


class TestDsv4TileLangCSAIndexerConfig(unittest.TestCase):
    """Tests for the Task 4 TileLang CSA Indexer configuration switches.

    The validation block for these switches is gated on
    experimental_attention_variant == "dsv4_hybrid", so each negative/positive
    test sets up that hybrid context.
    """

    NL = 4

    def _hybrid_kwargs(self, **overrides):
        kw = dict(
            num_hidden_layers=self.NL,
            experimental_attention_variant="dsv4_hybrid",
            multi_latent_attention=True,
            csa_compress_ratios=[4] * self.NL,
            rope_type="yarn",
        )
        kw.update(overrides)
        return kw

    def test_defaults_are_false(self):
        """All three new switches default to False on a non-hybrid config."""
        config = TransformerConfig(num_hidden_layers=self.NL)
        self.assertFalse(config.dsv4_tilelang_enable_csa_indexer)
        self.assertFalse(config.dsv4_tilelang_csa_indexer_enable_backward)
        self.assertFalse(config.dsv4_tilelang_csa_indexer_debug_compare)

    def test_transform_rules_contain_new_keys(self):
        """transform_rules must include the three new HF config.json keys."""
        rules = TransformerConfig.transform_rules
        for k in (
            "dsv4_tilelang_enable_csa_indexer",
            "dsv4_tilelang_csa_indexer_enable_backward",
            "dsv4_tilelang_csa_indexer_debug_compare",
        ):
            self.assertIn(k, rules)
            self.assertEqual(rules[k], k)

    def test_enable_csa_indexer_requires_paddle_compat_backend(self):
        """Enabling the CSA Indexer kernel requires attention_paddle_compat backend."""
        with self.assertRaises(ValueError):
            TransformerConfig(
                **self._hybrid_kwargs(
                    dsv4_tilelang_enable_csa_indexer=True,
                )
            )

    def test_enable_csa_indexer_with_paddle_compat_ok(self):
        config = TransformerConfig(
            **self._hybrid_kwargs(
                dsv4_tilelang_backend="attention_paddle_compat",
                dsv4_tilelang_enable_csa_indexer=True,
            )
        )
        self.assertTrue(config.dsv4_tilelang_enable_csa_indexer)

    def test_backward_requires_enable_csa_indexer(self):
        with self.assertRaises(ValueError):
            TransformerConfig(
                **self._hybrid_kwargs(
                    dsv4_tilelang_backend="attention_paddle_compat",
                    dsv4_tilelang_enable_csa_indexer=False,
                    dsv4_tilelang_csa_indexer_enable_backward=True,
                )
            )

    def test_debug_compare_requires_enable_csa_indexer(self):
        with self.assertRaises(ValueError):
            TransformerConfig(
                **self._hybrid_kwargs(
                    dsv4_tilelang_backend="attention_paddle_compat",
                    dsv4_tilelang_enable_csa_indexer=False,
                    dsv4_tilelang_csa_indexer_debug_compare=True,
                )
            )

    def test_full_chain_valid(self):
        config = TransformerConfig(
            **self._hybrid_kwargs(
                dsv4_tilelang_backend="attention_paddle_compat",
                dsv4_tilelang_enable_csa_indexer=True,
                dsv4_tilelang_csa_indexer_enable_backward=True,
                dsv4_tilelang_csa_indexer_debug_compare=True,
            )
        )
        self.assertTrue(config.dsv4_tilelang_enable_csa_indexer)
        self.assertTrue(config.dsv4_tilelang_csa_indexer_enable_backward)
        self.assertTrue(config.dsv4_tilelang_csa_indexer_debug_compare)

    def test_switches_independent_of_phase_controls(self):
        """The new TileLang switches must not modify training-phase fields
        (csa_dense_mode / dsa_indexer_use_sparse_loss / dsa_indexer_loss_coeff)."""
        baseline = TransformerConfig(**self._hybrid_kwargs())
        config = TransformerConfig(
            **self._hybrid_kwargs(
                dsv4_tilelang_backend="attention_paddle_compat",
                dsv4_tilelang_enable_csa_indexer=True,
                dsv4_tilelang_csa_indexer_enable_backward=True,
                dsv4_tilelang_csa_indexer_debug_compare=True,
            )
        )
        self.assertEqual(config.csa_dense_mode, baseline.csa_dense_mode)
        self.assertEqual(
            config.dsa_indexer_use_sparse_loss,
            baseline.dsa_indexer_use_sparse_loss,
        )
        self.assertEqual(
            config.dsa_indexer_loss_coeff, baseline.dsa_indexer_loss_coeff
        )

    def test_from_config_propagates_switches_via_transform_rules(self):
        """from_config should map HF-style keys via transform_rules."""

        class _AttrDict:
            def __init__(self, **kw):
                self.__dict__.update(kw)

        cfg = _AttrDict(
            **self._hybrid_kwargs(
                dsv4_tilelang_backend="attention_paddle_compat",
                dsv4_tilelang_enable_csa_indexer=True,
                dsv4_tilelang_csa_indexer_enable_backward=True,
                dsv4_tilelang_csa_indexer_debug_compare=True,
            )
        )
        config = TransformerConfig.from_config(cfg)
        self.assertTrue(config.dsv4_tilelang_enable_csa_indexer)
        self.assertTrue(config.dsv4_tilelang_csa_indexer_enable_backward)
        self.assertTrue(config.dsv4_tilelang_csa_indexer_debug_compare)
        self.assertEqual(
            config.dsv4_tilelang_backend, "attention_paddle_compat"
        )


if __name__ == "__main__":
    unittest.main()