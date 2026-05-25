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


import math
import unittest
from unittest.mock import MagicMock, patch

import paddle


class TestYarnFindCorrectionDim(unittest.TestCase):
    """Tests for _yarn_find_correction_dim helper function."""

    def test_basic_calculation(self):
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_find_correction_dim,
        )

        result = _yarn_find_correction_dim(
            num_rotations=1.0,
            dim=64,
            rotary_base=10000,
            max_position_embeddings=2048,
        )
        self.assertIsInstance(result, float)
        self.assertTrue(result > 0)

    def test_different_num_rotations(self):
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_find_correction_dim,
        )

        r1 = _yarn_find_correction_dim(1.0, 64, 10000, 2048)
        r2 = _yarn_find_correction_dim(2.0, 64, 10000, 2048)
        # Higher rotations should give smaller correction dim
        self.assertLess(r2, r1)

    def test_custom_rotary_base(self):
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_find_correction_dim,
        )

        result = _yarn_find_correction_dim(
            num_rotations=1.0,
            dim=64,
            rotary_base=500000,
            max_position_embeddings=4096,
        )
        self.assertIsInstance(result, float)


class TestYarnFindCorrectionRange(unittest.TestCase):
    """Tests for _yarn_find_correction_range helper function."""

    def test_basic_range(self):
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_find_correction_range,
        )

        low, high = _yarn_find_correction_range(
            low_rot=32.0,
            high_rot=1.0,
            dim=64,
            rotary_base=10000,
            max_position_embeddings=4096,
        )
        self.assertIsInstance(low, int)
        self.assertIsInstance(high, int)
        self.assertGreaterEqual(low, 0)
        self.assertLessEqual(high, 63)
        self.assertLessEqual(low, high)

    def test_round_to_int_true(self):
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_find_correction_range,
        )

        low, high = _yarn_find_correction_range(
            low_rot=32.0,
            high_rot=1.0,
            dim=64,
            rotary_base=10000,
            max_position_embeddings=4096,
            round_to_int=True,
        )
        self.assertIsInstance(low, int)
        self.assertIsInstance(high, int)

    def test_round_to_int_false(self):
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_find_correction_range,
        )

        low, high = _yarn_find_correction_range(
            low_rot=32.0,
            high_rot=1.0,
            dim=64,
            rotary_base=10000,
            max_position_embeddings=4096,
            round_to_int=False,
        )
        # round_to_int=False returns float values, clamped
        self.assertIsInstance(low, float)
        self.assertIsInstance(high, float)
        # Clamped values should still be valid
        self.assertGreaterEqual(low, 0)
        self.assertLessEqual(high, 63)

    def test_clamped_values(self):
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_find_correction_range,
        )

        # Use extreme values that would exceed dim
        low, high = _yarn_find_correction_range(
            low_rot=0.1,
            high_rot=0.01,
            dim=32,
            rotary_base=10000,
            max_position_embeddings=2048,
        )
        self.assertGreaterEqual(low, 0)
        self.assertLessEqual(high, 31)


class TestYarnLinearRampMask(unittest.TestCase):
    """Tests for _yarn_linear_ramp_mask helper function."""

    def test_basic_ramp(self):
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_linear_ramp_mask,
        )

        mask = _yarn_linear_ramp_mask(min=0, max=10, dim=20)
        self.assertEqual(mask.shape, [20])
        # Linear ramp: mask[i] = clamp((i - min) / (max - min), 0, 1)
        # Values 0-9 ramp from 0.0 to 0.9, values 10-19 are clamped to 1.0
        self.assertAlmostEqual(mask[0].item(), 0.0, places=5)
        self.assertAlmostEqual(mask[10].item(), 1.0, places=5)
        self.assertAlmostEqual(mask[19].item(), 1.0, places=5)
        # Last 10 should all be 1.0
        self.assertTrue(
            paddle.allclose(mask[10:], paddle.ones([10]), atol=1e-6)
        )

    def test_equal_min_max(self):
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_linear_ramp_mask,
        )

        mask = _yarn_linear_ramp_mask(min=5, max=5, dim=10)
        self.assertEqual(mask.shape, [10])

    def test_offset_ramp(self):
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_linear_ramp_mask,
        )

        mask = _yarn_linear_ramp_mask(min=3, max=7, dim=10)
        self.assertEqual(mask.shape, [10])
        # Values should be clamped to [0, 1]
        self.assertTrue(paddle.all(mask >= 0))
        self.assertTrue(paddle.all(mask <= 1))


class TestYarnGetMscale(unittest.TestCase):
    """Tests for _yarn_get_mscale helper function."""

    def test_scale_le_1(self):
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_get_mscale,
        )

        result = _yarn_get_mscale(scale=1.0, mscale=1.0)
        self.assertEqual(result, 1.0)

    def test_scale_le_1_small(self):
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_get_mscale,
        )

        result = _yarn_get_mscale(scale=0.5, mscale=1.0)
        self.assertEqual(result, 1.0)

    def test_scale_gt_1(self):
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_get_mscale,
        )

        result = _yarn_get_mscale(scale=10.0, mscale=1.0)
        expected = 0.1 * 1.0 * math.log(10.0) + 1.0
        self.assertAlmostEqual(result, expected, places=5)

    def test_scale_gt_1_with_mscale(self):
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_get_mscale,
        )

        result = _yarn_get_mscale(scale=4.0, mscale=0.5)
        expected = 0.1 * 0.5 * math.log(4.0) + 1.0
        self.assertAlmostEqual(result, expected, places=5)


class TestYarnGetConcentrationFactor(unittest.TestCase):
    """Tests for _yarn_get_concentration_factor function."""

    def test_basic(self):
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_get_concentration_factor,
        )

        result = _yarn_get_concentration_factor(
            scaling_factor=1.0,
            mscale=1.0,
            mscale_all_dim=0.0,
        )
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_with_scaling(self):
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_get_concentration_factor,
        )

        result = _yarn_get_concentration_factor(
            scaling_factor=4.0,
            mscale=1.0,
            mscale_all_dim=0.0,
        )
        self.assertIsInstance(result, float)
        self.assertNotAlmostEqual(result, 1.0)

    def test_lru_cache_same_args(self):
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_get_concentration_factor,
        )

        r1 = _yarn_get_concentration_factor(2.0, 1.0, 0.0)
        r2 = _yarn_get_concentration_factor(2.0, 1.0, 0.0)
        self.assertEqual(r1, r2)


class TestYarnGetConcentrationFactorFromConfig(unittest.TestCase):
    """Tests for _yarn_get_concentration_factor_from_config function."""

    def test_with_full_config_attrs(self):
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_get_concentration_factor_from_config,
        )

        mock_config = MagicMock()
        mock_config.yarn_rotary_scaling_factor = 4.0
        mock_config.yarn_mscale = 1.0
        mock_config.yarn_mscale_all_dim = 0.0
        result = _yarn_get_concentration_factor_from_config(mock_config)
        self.assertIsInstance(result, float)

    def test_missing_attrs_returns_default(self):
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_get_concentration_factor_from_config,
        )

        mock_config = MagicMock()
        # Remove the attributes so hasattr returns False
        del mock_config.yarn_rotary_scaling_factor
        del mock_config.yarn_mscale
        del mock_config.yarn_mscale_all_dim
        result = _yarn_get_concentration_factor_from_config(mock_config)
        self.assertEqual(result, 1.0)

    def test_partial_attrs_returns_default(self):
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_get_concentration_factor_from_config,
        )

        mock_config = MagicMock()
        mock_config.yarn_rotary_scaling_factor = 4.0
        del mock_config.yarn_mscale
        del mock_config.yarn_mscale_all_dim
        result = _yarn_get_concentration_factor_from_config(mock_config)
        self.assertEqual(result, 1.0)


class TestYarnRotaryEmbeddingInit(unittest.TestCase):
    """Tests for YarnRotaryEmbedding initialization."""

    @patch(
        "paddlefleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_basic_init(self, mock_ps):
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            YarnRotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        yarn = YarnRotaryEmbedding(head_dim=64)
        self.assertEqual(yarn.dim, 64)
        self.assertIsNotNone(yarn.inv_freq_extra)
        self.assertIsNotNone(yarn.inv_freq_inter)

    @patch(
        "paddlefleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_init_with_custom_params(self, mock_ps):
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            YarnRotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        yarn = YarnRotaryEmbedding(
            head_dim=64,
            scaling_factor=2.0,
            original_max_position_embeddings=2048,
            beta_fast=16.0,
            beta_slow=0.5,
            mscale=0.5,
            mscale_all_dim=0.1,
        )
        self.assertEqual(yarn.scaling_factor, 2.0)
        self.assertEqual(yarn.original_max_position_embeddings, 2048)
        self.assertEqual(yarn.beta_fast, 16.0)
        self.assertEqual(yarn.beta_slow, 0.5)
        self.assertEqual(yarn.mscale, 0.5)
        self.assertEqual(yarn.mscale_all_dim, 0.1)

    @patch(
        "paddlefleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_init_with_interleaved(self, mock_ps):
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            YarnRotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        yarn = YarnRotaryEmbedding(
            head_dim=64,
            rotary_interleaved=True,
        )
        self.assertTrue(yarn.rotary_interleaved)

    @patch(
        "paddlefleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_init_with_interpolation(self, mock_ps):
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            YarnRotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        yarn = YarnRotaryEmbedding(
            head_dim=64,
            seq_len_interpolation_factor=2.0,
        )
        self.assertEqual(yarn.seq_len_interpolation_factor, 2.0)

    @patch(
        "paddlefleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_correction_range_round_to_int(self, mock_ps):
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            YarnRotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        yarn = YarnRotaryEmbedding(
            head_dim=64,
            correction_range_round_to_int=False,
        )
        self.assertFalse(yarn.correction_range_round_to_int)


class TestYarnRotaryEmbeddingForward(unittest.TestCase):
    """Tests for YarnRotaryEmbedding forward pass."""

    @patch(
        "paddlefleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_forward_basic(self, mock_ps):
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            YarnRotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        yarn = YarnRotaryEmbedding(head_dim=64)
        emb, mscale = yarn(max_seq_len=128)
        # Expected shape: [1, 128, 1, 64]
        self.assertEqual(emb.shape, [1, 128, 1, 64])
        self.assertIsInstance(mscale, float)

    @patch(
        "paddlefleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_forward_with_offset(self, mock_ps):
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            YarnRotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        yarn = YarnRotaryEmbedding(head_dim=64)
        emb, mscale = yarn(max_seq_len=64, offset=10)
        self.assertEqual(emb.shape, [1, 64, 1, 64])

    @patch(
        "paddlefleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_forward_with_scaling(self, mock_ps):
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            YarnRotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        yarn = YarnRotaryEmbedding(
            head_dim=64,
            scaling_factor=4.0,
        )
        emb, mscale = yarn(max_seq_len=128)
        self.assertEqual(emb.shape, [1, 128, 1, 64])

    @patch(
        "paddlefleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_forward_interleaved(self, mock_ps):
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            YarnRotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        yarn = YarnRotaryEmbedding(
            head_dim=64,
            rotary_interleaved=True,
        )
        emb, mscale = yarn(max_seq_len=32)
        self.assertEqual(emb.shape, [1, 32, 1, 64])

    @patch(
        "paddlefleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_forward_small_dim(self, mock_ps):
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            YarnRotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        yarn = YarnRotaryEmbedding(head_dim=32)
        emb, mscale = yarn(max_seq_len=16)
        self.assertEqual(emb.shape, [1, 16, 1, 32])


class TestYarnRotaryEmbeddingCache(unittest.TestCase):
    """Tests for YarnRotaryEmbedding caching mechanism."""

    @patch(
        "paddlefleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_set_cos_sin_cache(self, mock_ps):
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            YarnRotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        yarn = YarnRotaryEmbedding(head_dim=64)
        yarn._set_cos_sin_cache(seq_len=64, offset=0, dtype=paddle.float32)
        self.assertEqual(yarn.max_seq_len_cached, 64)
        self.assertEqual(yarn.offset_cached, 0)
        self.assertEqual(yarn.dtype_cached, paddle.float32)
        self.assertIsNotNone(yarn.cos_cached)
        self.assertIsNotNone(yarn.sin_cached)


class TestYarnRotaryEmbeddingIsSubclass(unittest.TestCase):
    """Tests to verify YarnRotaryEmbedding is a subclass of RotaryEmbedding."""

    @patch(
        "paddlefleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_inheritance(self, mock_ps):
        from paddlefleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            YarnRotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        yarn = YarnRotaryEmbedding(head_dim=64)
        self.assertIsInstance(yarn, RotaryEmbedding)

    @patch(
        "paddlefleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_has_parent_methods(self, mock_ps):
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            YarnRotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        yarn = YarnRotaryEmbedding(head_dim=64)
        # Should have methods from parent class
        self.assertTrue(hasattr(yarn, "get_cos_sin"))
        self.assertTrue(hasattr(yarn, "get_freqs_non_repeated"))


if __name__ == "__main__":
    unittest.main()
