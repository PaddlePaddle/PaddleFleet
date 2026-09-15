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
"""Behavior tests for the YaRN rotary position embedding in
``paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding``.

Every expected value is derived independently from the documented YaRN math
using plain ``math``/``numpy`` -- the functions under test are never called to
produce their own reference:

  * ``_yarn_find_correction_dim`` is the closed form
    ``dim * log(max_pos / (rot * 2*pi)) / (2 * log(base))``; a chosen input
    makes the log argument exactly 1 so the result is exactly 0.
  * ``_yarn_find_correction_range`` floors/ceils the two correction dims and
    clamps to ``[0, dim-1]``; both the rounding and the clamping edges are
    checked with hand-derived bounds.
  * ``_yarn_linear_ramp_mask`` is ``clamp((arange(dim)-min)/(max-min), 0, 1)``
    with a ``min==max`` singularity guard; content is checked element by
    element.
  * ``_yarn_get_mscale`` is 1.0 for ``scale<=1`` and
    ``0.1*mscale*log(scale)+1`` otherwise; inputs are chosen so the log is a
    clean integer.
  * ``_yarn_get_concentration_factor`` is the ratio of two mscales.
  * ``_yarn_get_concentration_factor_from_config`` returns that ratio only when
    all three yarn fields are present, else 1.0.
  * ``YarnRotaryEmbedding`` with ``scaling_factor==1`` collapses to standard
    RoPE, so ``forward`` is compared against an independent outer-product table.

The whole ``paddlefleet`` package imports ``paddle`` at import time. When paddle
is not installed the module is skipped with an honest reason rather than faking
a pass; only a genuine missing-dependency ``ImportError`` triggers the skip.

``TestGetCachedCosSinBug`` documents a real production bug (see the test's
comment): ``get_cached_cos_sin`` slices the wrong axis, so requesting a shorter
sequence never truncates the cached table. It asserts the CORRECT behavior and
is marked ``expectedFailure``; production code is left untouched.
"""

import math
import unittest

try:
    import numpy as np
    import paddle

    from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
        YarnRotaryEmbedding,
        _yarn_find_correction_dim,
        _yarn_find_correction_range,
        _yarn_get_concentration_factor,
        _yarn_get_concentration_factor_from_config,
        _yarn_get_mscale,
        _yarn_linear_ramp_mask,
    )

    PADDLE_AVAILABLE = True
    IMPORT_ERROR = ""
except ImportError as exc:  # real missing dep only, not API/compile errors
    PADDLE_AVAILABLE = False
    IMPORT_ERROR = repr(exc)

SKIP_REASON = (
    f"paddle / paddlefleet not importable in this environment: {IMPORT_ERROR}"
)


def ref_correction_dim(
    num_rotations, dim, rotary_base, max_position_embeddings
):
    """Independent closed form of the inverse-dim YaRN formula."""
    return (
        dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
    ) / (2 * math.log(rotary_base))


def ref_mscale(scale, mscale):
    """Independent reference for the attention (m)scale."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestYarnFindCorrectionDim(unittest.TestCase):
    def test_matches_closed_form(self):
        cases = [
            (1.0, 64, 10000.0, 2048),
            (2.0, 128, 10000.0, 8192),
            (4.0, 96, 500000.0, 4096),
        ]
        for num_rot, dim, base, max_pos in cases:
            got = _yarn_find_correction_dim(num_rot, dim, base, max_pos)
            expected = ref_correction_dim(num_rot, dim, base, max_pos)
            self.assertAlmostEqual(got, expected, places=9)

    def test_zero_when_log_argument_is_one(self):
        # max_pos == num_rotations * 2*pi makes the log argument exactly 1,
        # so the whole expression must be exactly 0 for any dim/base.
        got = _yarn_find_correction_dim(
            num_rotations=1.0,
            dim=64,
            rotary_base=10000.0,
            max_position_embeddings=2 * math.pi,
        )
        self.assertAlmostEqual(got, 0.0, places=12)

    def test_scales_linearly_with_dim(self):
        # dim appears as a bare factor, so doubling dim doubles the result.
        small = _yarn_find_correction_dim(1.0, 64, 10000.0, 2048)
        large = _yarn_find_correction_dim(1.0, 128, 10000.0, 2048)
        self.assertAlmostEqual(large, 2.0 * small, places=9)

    def test_more_rotations_gives_smaller_dim(self):
        # num_rotations sits in the denominator inside the log, so a larger
        # value strictly decreases the correction dim.
        few = _yarn_find_correction_dim(1.0, 64, 10000.0, 2048)
        many = _yarn_find_correction_dim(8.0, 64, 10000.0, 2048)
        self.assertLess(many, few)


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestYarnFindCorrectionRange(unittest.TestCase):
    def test_rounding_floors_low_and_ceils_high(self):
        dim, base, max_pos = 64, 10000.0, 2048
        low_rot, high_rot = 32.0, 1.0  # YaRN beta_fast / beta_slow defaults
        low_raw = ref_correction_dim(low_rot, dim, base, max_pos)
        high_raw = ref_correction_dim(high_rot, dim, base, max_pos)
        # Guard: raw bounds are strictly inside [0, dim-1] here, so clamping
        # does not interfere with the rounding assertion.
        self.assertTrue(0 < low_raw < high_raw < dim - 1)

        low, high = _yarn_find_correction_range(
            low_rot, high_rot, dim, base, max_pos
        )
        self.assertEqual(low, math.floor(low_raw))
        self.assertEqual(high, math.ceil(high_raw))

    def test_round_to_int_false_keeps_raw_floats(self):
        dim, base, max_pos = 64, 10000.0, 2048
        low_rot, high_rot = 32.0, 1.0
        low, high = _yarn_find_correction_range(
            low_rot, high_rot, dim, base, max_pos, round_to_int=False
        )
        self.assertAlmostEqual(
            low, ref_correction_dim(low_rot, dim, base, max_pos), places=9
        )
        self.assertAlmostEqual(
            high, ref_correction_dim(high_rot, dim, base, max_pos), places=9
        )
        self.assertLessEqual(low, high)

    def test_clamps_to_zero_and_dim_minus_one(self):
        # low_rot huge -> negative raw dim -> clamped to 0.
        # high_rot tiny -> raw dim above dim-1 -> clamped to dim-1.
        dim = 64
        low, high = _yarn_find_correction_range(
            low_rot=1e12,
            high_rot=1e-12,
            dim=dim,
            rotary_base=10000.0,
            max_position_embeddings=2048,
        )
        self.assertEqual(low, 0)
        self.assertEqual(high, dim - 1)


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestYarnLinearRampMask(unittest.TestCase):
    def test_ramp_content_and_clamping(self):
        # linear = (arange(8) - 1) / (5 - 1); clamped to [0, 1].
        mask = _yarn_linear_ramp_mask(min=1.0, max=5.0, dim=8)
        expected = np.array(
            [0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0], dtype=np.float32
        )
        self.assertEqual(list(mask.shape), [8])
        np.testing.assert_allclose(mask.numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_min_equals_max_singularity_guard(self):
        # min == max: production nudges max by +0.001, so the ramp is a step at
        # the index equal to ``min`` (everything below is 0, at/above is ~1).
        mask = _yarn_linear_ramp_mask(min=2.0, max=2.0, dim=4)
        # index 0,1 -> negative -> 0 ; index 2 -> 0/0.001 -> 0 ; index 3 -> 1
        expected = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        np.testing.assert_allclose(mask.numpy(), expected, rtol=1e-6, atol=1e-6)


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestYarnGetMscale(unittest.TestCase):
    def test_scale_leq_one_is_identity(self):
        self.assertEqual(_yarn_get_mscale(scale=0.5), 1.0)
        self.assertEqual(_yarn_get_mscale(scale=1.0), 1.0)
        self.assertEqual(_yarn_get_mscale(scale=0.999, mscale=5.0), 1.0)

    def test_scale_gt_one_uses_log_formula(self):
        # scale == e makes log(scale) == 1, so the result is 0.1*mscale + 1.
        self.assertAlmostEqual(
            _yarn_get_mscale(scale=math.e, mscale=1.0), 1.1, places=9
        )
        self.assertAlmostEqual(
            _yarn_get_mscale(scale=math.e, mscale=2.0), 1.2, places=9
        )
        # scale == e**2 -> log == 2.
        self.assertAlmostEqual(
            _yarn_get_mscale(scale=math.e**2, mscale=0.5), 1.1, places=9
        )


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestYarnGetConcentrationFactor(unittest.TestCase):
    def test_scaling_leq_one_is_one(self):
        self.assertAlmostEqual(
            _yarn_get_concentration_factor(1.0, 1.0, 0.0), 1.0, places=12
        )
        self.assertAlmostEqual(
            _yarn_get_concentration_factor(0.5, 3.0, 2.0), 1.0, places=12
        )

    def test_ratio_of_two_mscales(self):
        # numerator uses mscale, denominator uses mscale_all_dim.
        # scaling == e: num = 0.1*1+1 = 1.1, den = 0.1*0+1 = 1.0 -> 1.1
        self.assertAlmostEqual(
            _yarn_get_concentration_factor(math.e, 1.0, 0.0), 1.1, places=9
        )
        # num = 1.2, den = 1.1 -> 1.2/1.1
        expected = ref_mscale(math.e, 2.0) / ref_mscale(math.e, 1.0)
        self.assertAlmostEqual(
            _yarn_get_concentration_factor(math.e, 2.0, 1.0),
            expected,
            places=9,
        )
        self.assertNotAlmostEqual(expected, 1.1, places=4)


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestYarnGetConcentrationFactorFromConfig(unittest.TestCase):
    def test_all_fields_present_returns_ratio(self):
        from types import SimpleNamespace

        cfg = SimpleNamespace(
            yarn_rotary_scaling_factor=math.e,
            yarn_mscale=1.0,
            yarn_mscale_all_dim=0.0,
        )
        self.assertAlmostEqual(
            _yarn_get_concentration_factor_from_config(cfg), 1.1, places=9
        )

    def test_missing_any_field_returns_one(self):
        from types import SimpleNamespace

        # Only two of the three required fields -> all(hasattr(...)) is False.
        cfg = SimpleNamespace(
            yarn_rotary_scaling_factor=math.e, yarn_mscale=1.0
        )
        self.assertEqual(_yarn_get_concentration_factor_from_config(cfg), 1.0)

        empty = SimpleNamespace()
        self.assertEqual(_yarn_get_concentration_factor_from_config(empty), 1.0)


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestYarnRotaryEmbeddingForward(unittest.TestCase):
    def _independent_emb(self, head_dim, seq_len, base=10000.0):
        # scaling_factor == 1 collapses YaRN to standard RoPE.
        inv_freq = np.array(
            [
                1.0 / (base ** ((2 * i) / head_dim))
                for i in range(head_dim // 2)
            ],
            dtype=np.float64,
        )
        pos = np.arange(seq_len, dtype=np.float64)
        freqs = np.outer(pos, inv_freq)  # [seq_len, head_dim//2]
        emb = np.concatenate([freqs, freqs], axis=-1)  # [seq_len, head_dim]
        return emb

    def test_forward_matches_standard_rope_when_scaling_one(self):
        head_dim, seq_len = 64, 12
        emb = YarnRotaryEmbedding(head_dim=head_dim, scaling_factor=1.0)
        emb_val, mscale = emb(max_seq_len=seq_len)
        # scaling == 1 -> concentration factor is exactly 1.0.
        self.assertAlmostEqual(float(mscale), 1.0, places=12)
        self.assertEqual(list(emb_val.shape), [1, seq_len, 1, head_dim])
        expected = self._independent_emb(head_dim, seq_len)
        got = emb_val.numpy()[0, :, 0, :]
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

    def test_forward_offset_shifts_positions(self):
        head_dim, seq_len, offset = 64, 8, 5
        emb = YarnRotaryEmbedding(head_dim=head_dim, scaling_factor=1.0)
        emb_val, _ = emb(max_seq_len=seq_len, offset=offset)
        # positions are arange(seq_len) + offset
        base = 10000.0
        inv_freq = np.array(
            [
                1.0 / (base ** ((2 * i) / head_dim))
                for i in range(head_dim // 2)
            ],
            dtype=np.float64,
        )
        pos = np.arange(seq_len, dtype=np.float64) + offset
        freqs = np.outer(pos, inv_freq)
        expected = np.concatenate([freqs, freqs], axis=-1)
        np.testing.assert_allclose(
            emb_val.numpy()[0, :, 0, :], expected, rtol=1e-5, atol=1e-5
        )

    def test_forward_mscale_reflects_scaling_factor(self):
        # scaling_factor == e -> mscale = 0.1*log(e)+1 = 1.1 (mscale_all_dim=0).
        emb = YarnRotaryEmbedding(
            head_dim=64,
            scaling_factor=math.e,
            mscale=1.0,
            mscale_all_dim=0.0,
        )
        _, mscale = emb(max_seq_len=8)
        self.assertAlmostEqual(float(mscale), 1.1, places=6)


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestGetCachedCosSinBug(unittest.TestCase):
    @unittest.expectedFailure
    def test_get_cached_cos_sin_truncates_to_requested_seq_len(self):
        # PRODUCTION BUG (yarn_rotary_pos_embedding.py:248):
        #   return (self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...])
        # ``cos_cached`` has shape [1, seq_len_cached, 1, dim] (the sequence axis
        # is axis 1, produced by ``emb[None, :, None, :]`` in ``forward``).
        # Slicing ``[:seq_len]`` therefore indexes axis 0, whose size is 1, and
        # never truncates the sequence. Requesting a shorter sequence returns
        # the full cached table instead of ``seq_len`` positions.
        # Correct behavior would slice the sequence axis: ``[:, :seq_len, ...]``.
        # This test asserts the CORRECT behavior and is expected to fail until
        # the production slice is fixed; production code is left unmodified.
        head_dim, cached_len, requested = 64, 32, 8
        emb = YarnRotaryEmbedding(
            head_dim=head_dim,
            scaling_factor=1.0,
            original_max_position_embeddings=cached_len,
        )
        cos, sin = emb.get_cached_cos_sin(seq_len=requested)
        # The sequence dimension (axis 1) should hold exactly ``requested``
        # positions. It currently stays at ``cached_len`` -> AssertionError.
        self.assertEqual(cos.shape[1], requested)
        self.assertEqual(sin.shape[1], requested)


if __name__ == "__main__":
    unittest.main()
