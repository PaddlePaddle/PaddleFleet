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
"""Behavior tests for the YARN RoPE helpers and ``YarnRotaryEmbedding``.

Target module:
``paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding``

All expected values are HAND-DERIVED from the mathematical definition of each
function (never produced by calling the function under test):

  * ``_yarn_find_correction_dim`` -- closed form
    ``dim * ln(max_pos / (num_rot * 2*pi)) / (2 * ln(base))``.
  * ``_yarn_find_correction_range`` -- floor/ceil rounding and the
    ``max(low, 0)`` / ``min(high, dim-1)`` clamps.
  * ``_yarn_linear_ramp_mask`` -- ``clamp((i - min)/(max - min), 0, 1)`` and the
    ``min == max`` singularity guard.
  * ``_yarn_get_mscale`` / ``_yarn_get_concentration_factor`` -- the log-mscale
    formula and its numerator/denominator ratio.
  * ``_yarn_get_concentration_factor_from_config`` -- hasattr gating.
  * ``YarnRotaryEmbedding.forward`` -- content identity: position 0 yields a
    zero frequency row, the two halves of the (non-interleaved) embedding are
    duplicates, and a 2-D ``position_ids`` consumes only its first batch row.

``paddlefleet`` imports ``paddle`` at import time; when paddle is unavailable
the whole module is skipped with an honest reason (never a faked pass). Only
``ImportError`` is treated as a missing dependency -- API/compile errors are
allowed to surface.
"""

import math
import types
import unittest

try:
    import numpy as np
    import paddle

    from paddlefleet.models.common.embeddings import (
        yarn_rotary_pos_embedding as yarn_mod,
    )

    PADDLE_AVAILABLE = True
    IMPORT_ERROR = ""
except ImportError as exc:  # only genuine missing-dependency, not API errors
    PADDLE_AVAILABLE = False
    IMPORT_ERROR = repr(exc)

SKIP_REASON = (
    f"paddle / paddlefleet not importable in this environment: {IMPORT_ERROR}"
)


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestYarnFindCorrectionDim(unittest.TestCase):
    def test_matches_closed_form_and_concrete_value(self):
        # Independent closed form: dim*ln(mpe/(n*2pi)) / (2*ln(base)).
        n, dim, base, mpe = 1.0, 64, 10000.0, 2048
        expected = (dim * math.log(mpe / (n * 2 * math.pi))) / (
            2 * math.log(base)
        )
        got = yarn_mod._yarn_find_correction_dim(n, dim, base, mpe)
        self.assertAlmostEqual(got, expected, places=10)
        # Hand-computed concrete anchor (see standalone derivation).
        self.assertAlmostEqual(got, 20.105200671565424, places=9)

    def test_monotonic_decreasing_in_num_rotations(self):
        # More rotations -> smaller correction dim (log is decreasing in n).
        r1 = yarn_mod._yarn_find_correction_dim(1.0, 64, 10000.0, 2048)
        r2 = yarn_mod._yarn_find_correction_dim(2.0, 64, 10000.0, 2048)
        self.assertLess(r2, r1)
        # The gap equals dim*ln(2)/(2*ln(base)), independent of mpe.
        expected_gap = 64 * math.log(2.0) / (2 * math.log(10000.0))
        self.assertAlmostEqual(r1 - r2, expected_gap, places=10)


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestYarnFindCorrectionRange(unittest.TestCase):
    def test_rounded_bounds_concrete(self):
        # floor(10.47..)=10, ceil(22.51..)=23; neither hits a clamp here.
        low, high = yarn_mod._yarn_find_correction_range(
            32.0, 1.0, 64, 10000.0, 4096, round_to_int=True
        )
        self.assertEqual((low, high), (10, 23))
        self.assertIsInstance(low, int)
        self.assertIsInstance(high, int)

    def test_unrounded_returns_floats(self):
        low, high = yarn_mod._yarn_find_correction_range(
            32.0, 1.0, 64, 10000.0, 4096, round_to_int=False
        )
        self.assertNotIsInstance(low, int)
        self.assertNotIsInstance(high, int)
        self.assertAlmostEqual(low, 10.472240810318025, places=9)
        self.assertAlmostEqual(high, 22.513440636877274, places=9)

    def test_lower_bound_clamped_to_zero(self):
        # Huge low_rot -> negative correction dim -> max(low, 0) == 0.
        low, high = yarn_mod._yarn_find_correction_range(
            10000.0, 1.0, 64, 10000.0, 2048, round_to_int=True
        )
        self.assertEqual(low, 0)

    def test_upper_bound_clamped_to_dim_minus_one(self):
        # Tiny high_rot -> correction dim exceeds dim-1 -> min(high, 63) == 63.
        low, high = yarn_mod._yarn_find_correction_range(
            32.0, 1e-7, 64, 10000.0, 2048, round_to_int=True
        )
        self.assertEqual(high, 63)


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestYarnLinearRampMask(unittest.TestCase):
    def test_ramp_values_with_upper_clamp(self):
        # (arange(8) - 0)/(4 - 0) clamped to [0,1].
        mask = yarn_mod._yarn_linear_ramp_mask(0, 4, 8)
        self.assertEqual(list(mask.shape), [8])
        expected = np.array(
            [0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0], dtype=np.float32
        )
        np.testing.assert_allclose(mask.numpy(), expected, atol=1e-6)

    def test_singularity_guard_min_equals_max(self):
        # min == max: max += 0.001, so (arange(3))/0.001 -> [0, 1000, 2000],
        # clamped to [0, 1, 1]. The +0.001 avoids a divide-by-zero.
        mask = yarn_mod._yarn_linear_ramp_mask(0, 0, 3)
        expected = np.array([0.0, 1.0, 1.0], dtype=np.float32)
        np.testing.assert_allclose(mask.numpy(), expected, atol=1e-6)


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestYarnGetMscale(unittest.TestCase):
    def test_scale_le_one_is_identity(self):
        self.assertEqual(yarn_mod._yarn_get_mscale(1.0, 1.0), 1.0)
        self.assertEqual(yarn_mod._yarn_get_mscale(0.3, 5.0), 1.0)

    def test_scale_gt_one_uses_log_formula(self):
        # scale = e -> 0.1*mscale*ln(e) + 1 = 0.1*mscale + 1 (clean anchor).
        self.assertAlmostEqual(
            yarn_mod._yarn_get_mscale(math.e, 1.0), 1.1, places=9
        )
        self.assertAlmostEqual(
            yarn_mod._yarn_get_mscale(math.e, 2.0), 1.2, places=9
        )
        # General case cross-checked against the independent formula.
        self.assertAlmostEqual(
            yarn_mod._yarn_get_mscale(4.0, 0.5),
            0.1 * 0.5 * math.log(4.0) + 1.0,
            places=9,
        )


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestYarnGetConcentrationFactor(unittest.TestCase):
    def test_ratio_numerator_over_denominator(self):
        # scaling=e: num = mscale(e,1)=1.1, den = mscale(e,0)=1.0 -> 1.1.
        got = yarn_mod._yarn_get_concentration_factor(math.e, 1.0, 0.0)
        self.assertIsInstance(got, float)
        self.assertAlmostEqual(got, 1.1, places=9)

    def test_ratio_with_nonzero_all_dim(self):
        # num = mscale(e,1)=1.1, den = mscale(e,0.5)=1.05 -> 1.1/1.05.
        got = yarn_mod._yarn_get_concentration_factor(math.e, 1.0, 0.5)
        self.assertAlmostEqual(got, 1.1 / 1.05, places=9)

    def test_no_scaling_is_one(self):
        # scaling <= 1 -> both mscale terms are 1.0 -> ratio 1.0.
        self.assertAlmostEqual(
            yarn_mod._yarn_get_concentration_factor(1.0, 1.0, 0.0),
            1.0,
            places=9,
        )


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestYarnConcentrationFactorFromConfig(unittest.TestCase):
    def test_all_fields_present_computes_factor(self):
        cfg = types.SimpleNamespace(
            yarn_rotary_scaling_factor=math.e,
            yarn_mscale=1.0,
            yarn_mscale_all_dim=0.0,
        )
        self.assertAlmostEqual(
            yarn_mod._yarn_get_concentration_factor_from_config(cfg),
            1.1,
            places=9,
        )

    def test_missing_any_field_returns_one(self):
        # Only two of the three required attributes are present.
        cfg = types.SimpleNamespace(
            yarn_rotary_scaling_factor=math.e,
            yarn_mscale=1.0,
        )
        self.assertEqual(
            yarn_mod._yarn_get_concentration_factor_from_config(cfg), 1.0
        )


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestYarnRotaryEmbeddingForward(unittest.TestCase):
    """CPU-observable content behavior of ``YarnRotaryEmbedding.forward``.

    These assert identities that hold regardless of the exact ``inv_freq``
    values, so no reference re-derives the production frequency table:
      * position 0 -> zero frequency row (outer(0, inv_freq) == 0);
      * non-interleaved embedding duplicates freqs into its two halves;
      * a 2-D position_ids consumes only its first batch row;
      * the returned scalar equals the concentration factor.
    """

    def _make(self, **kwargs):
        from unittest import mock

        with mock.patch(
            "paddlefleet.models.common.embeddings."
            "rotary_pos_embedding.parallel_state"
        ) as mock_ps:
            mock_ps.get_context_parallel_group.return_value = None
            return yarn_mod.YarnRotaryEmbedding(head_dim=64, **kwargs)

    def test_position_zero_row_is_zero(self):
        yarn = self._make()
        emb, _ = yarn.forward(max_seq_len=8, offset=0, position_ids=None)
        self.assertEqual(list(emb.shape), [1, 8, 1, 64])
        # seq[0] == 0 -> freqs row 0 is all zeros -> emb row 0 all zeros.
        np.testing.assert_allclose(
            emb[0, 0, 0, :].numpy(), np.zeros(64, dtype=np.float32), atol=1e-6
        )
        # A later position must NOT be all zero (freqs actually vary).
        self.assertGreater(float(paddle.abs(emb[0, 7, 0, :]).max()), 0.0)

    def test_non_interleaved_halves_are_duplicated(self):
        yarn = self._make(rotary_interleaved=False)
        emb, _ = yarn.forward(max_seq_len=6, offset=0, position_ids=None)
        dim = 64
        first_half = emb[0, :, 0, : dim // 2].numpy()
        second_half = emb[0, :, 0, dim // 2 :].numpy()
        np.testing.assert_allclose(first_half, second_half, atol=1e-6)

    def test_offset_shifts_sequence(self):
        yarn = self._make()
        base, _ = yarn.forward(max_seq_len=8, offset=0, position_ids=None)
        shifted, _ = yarn.forward(max_seq_len=8, offset=3, position_ids=None)
        # seq = arange + offset, so shifted row i == base row (i+3).
        np.testing.assert_allclose(
            shifted[0, 0, 0, :].numpy(),
            base[0, 3, 0, :].numpy(),
            atol=1e-6,
        )

    def test_2d_position_ids_uses_first_row_only(self):
        yarn = self._make()
        row0 = paddle.arange(8, dtype=paddle.int64)
        row1 = row0 + 100  # different second row must be ignored
        pos_2d = paddle.stack([row0, row1], axis=0)  # [2, 8]
        emb_2d, _ = yarn.forward(max_seq_len=8, position_ids=pos_2d)
        emb_1d, _ = yarn.forward(max_seq_len=8, position_ids=row0)
        np.testing.assert_allclose(emb_2d.numpy(), emb_1d.numpy(), atol=1e-6)

    def test_returned_scalar_is_concentration_factor(self):
        scaling = math.e
        yarn = self._make(
            scaling_factor=scaling, mscale=1.0, mscale_all_dim=0.0
        )
        _, mscale = yarn.forward(max_seq_len=4, position_ids=None)
        expected = yarn_mod._yarn_get_concentration_factor(scaling, 1.0, 0.0)
        self.assertAlmostEqual(float(mscale), expected, places=9)
        self.assertAlmostEqual(float(mscale), 1.1, places=6)


if __name__ == "__main__":
    unittest.main()
