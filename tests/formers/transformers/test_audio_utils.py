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

"""Behavior tests for paddlefleet.transformers.audio_utils.

Data-layer (audio preprocessing) tests. Every numeric expectation is derived
independently from the DSP definition on tiny synthetic audio, never by calling
the function under test to produce the reference. CPU-only (audio DSP is pure
numpy); the module still does ``import paddle`` at import time, so running these
tests requires paddle installed even though none of the tested code touches it.
"""

import unittest
import warnings

import numpy as np

MODULE = "paddlefleet.transformers.audio_utils"


# --------------------------------------------------------------------------- #
# Independent references (canonical mel-scale definition, structurally distinct
# from the vectorised production code: explicit per-filter loops, integer
# bit_length for fft length, explicit cosine window formulas).
# --------------------------------------------------------------------------- #
def _ref_hz_to_mel_htk(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _ref_hz_to_mel_slaney(f):
    f = float(f)
    if f >= 1000.0:
        return 15.0 + np.log(f / 1000.0) * (27.0 / np.log(6.4))
    return 3.0 * f / 200.0


def _ref_mel_to_hz_slaney(m):
    m = float(m)
    if m >= 15.0:
        return 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0))
    return 200.0 * m / 3.0


def _ref_optimal_fft(n):
    # smallest power of two >= n, computed with exact integer arithmetic
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _ref_mel_filter_bank_htk(num_bins, num_mel, fmin, fmax, sr):
    fft_freqs = np.linspace(0, sr // 2, num_bins)
    mel_min = _ref_hz_to_mel_htk(fmin)
    mel_max = _ref_hz_to_mel_htk(fmax)
    mel_pts = np.linspace(mel_min, mel_max, num_mel + 2)
    f_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)  # htk mel->hz
    fb = np.zeros((num_bins, num_mel))
    for k in range(num_mel):
        left, center, right = f_pts[k], f_pts[k + 1], f_pts[k + 2]
        for i, f in enumerate(fft_freqs):
            rising = (f - left) / (center - left)
            falling = (right - f) / (right - center)
            fb[i, k] = max(0.0, min(rising, falling))
    return fb, f_pts


class TestHertzMel(unittest.TestCase):
    """hertz_to_mel / mel_to_hertz: exact scale anchors, both branches, inverse."""

    def test_htk_clean_anchor(self):
        # 1 + 6300/700 = 10, log10(10)=1 -> exactly 2595 mel (independent of impl)
        from paddlefleet.transformers.audio_utils import hertz_to_mel

        self.assertAlmostEqual(hertz_to_mel(0.0, "htk"), 0.0, places=6)
        self.assertAlmostEqual(hertz_to_mel(6300.0, "htk"), 2595.0, places=4)

    def test_htk_default_scale(self):
        from paddlefleet.transformers.audio_utils import hertz_to_mel

        self.assertAlmostEqual(
            hertz_to_mel(1234.5), hertz_to_mel(1234.5, "htk"), places=10
        )

    def test_slaney_linear_and_log_branches(self):
        # Below 1000 Hz slaney is linear (3f/200); at 6400 Hz the log term is
        # log(6.4)*27/log(6.4) = 27, so mel = 15 + 27 = 42 exactly.
        from paddlefleet.transformers.audio_utils import hertz_to_mel

        self.assertAlmostEqual(hertz_to_mel(200.0, "slaney"), 3.0, places=6)
        self.assertAlmostEqual(hertz_to_mel(400.0, "slaney"), 6.0, places=6)
        self.assertAlmostEqual(hertz_to_mel(1000.0, "slaney"), 15.0, places=6)
        self.assertAlmostEqual(hertz_to_mel(6400.0, "slaney"), 42.0, places=6)

    def test_slaney_array_crosses_boundary(self):
        # Array path must apply the linear branch below 1000 and the log branch
        # at/above it, element-wise (catches a scalar-only or reversed branch).
        from paddlefleet.transformers.audio_utils import hertz_to_mel

        freqs = np.array([200.0, 400.0, 1000.0, 6400.0])
        mels = hertz_to_mel(freqs, "slaney")
        self.assertEqual(mels.shape, (4,))
        np.testing.assert_allclose(mels, [3.0, 6.0, 15.0, 42.0], rtol=1e-6)

    def test_htk_array_matches_definition(self):
        from paddlefleet.transformers.audio_utils import hertz_to_mel

        freqs = np.array([0.0, 700.0, 6300.0])
        np.testing.assert_allclose(
            hertz_to_mel(freqs, "htk"), _ref_hz_to_mel_htk(freqs), rtol=1e-9
        )

    def test_mel_to_hertz_clean_anchors(self):
        from paddlefleet.transformers.audio_utils import mel_to_hertz

        self.assertAlmostEqual(mel_to_hertz(2595.0, "htk"), 6300.0, places=3)
        self.assertAlmostEqual(mel_to_hertz(3.0, "slaney"), 200.0, places=5)
        self.assertAlmostEqual(mel_to_hertz(15.0, "slaney"), 1000.0, places=5)
        self.assertAlmostEqual(mel_to_hertz(42.0, "slaney"), 6400.0, places=3)

    def test_mel_to_hertz_slaney_array_crosses_boundary(self):
        from paddlefleet.transformers.audio_utils import mel_to_hertz

        mels = np.array([3.0, 15.0, 42.0])
        freqs = mel_to_hertz(mels, "slaney")
        np.testing.assert_allclose(freqs, [200.0, 1000.0, 6400.0], rtol=1e-5)

    def test_roundtrip_both_scales(self):
        from paddlefleet.transformers.audio_utils import (
            hertz_to_mel,
            mel_to_hertz,
        )

        freqs = np.array([50.0, 200.0, 999.0, 1000.0, 2000.0, 6400.0])
        for scale in ("htk", "slaney"):
            recovered = mel_to_hertz(hertz_to_mel(freqs, scale), scale)
            np.testing.assert_allclose(recovered, freqs, rtol=1e-6)

    def test_invalid_scale_raises(self):
        from paddlefleet.transformers.audio_utils import (
            hertz_to_mel,
            mel_to_hertz,
        )

        with self.assertRaises(ValueError):
            hertz_to_mel(1000.0, "bad")
        with self.assertRaises(ValueError):
            mel_to_hertz(15.0, "bad")


class TestTriangularFilterBank(unittest.TestCase):
    """_create_triangular_filter_bank: full hand-derived triangular content."""

    def test_hand_derived_triangles(self):
        from paddlefleet.transformers.audio_utils import (
            _create_triangular_filter_bank,
        )

        # filter_freqs = [0,1,2,3] -> two filters centred at 1 and 2, unit width.
        # Filter 0 rises 0->1 over [0,1], falls 1->0 over [1,2].
        # Filter 1 rises 0->1 over [1,2], falls 1->0 over [2,3].
        fft_freqs = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        filter_freqs = np.array([0.0, 1.0, 2.0, 3.0])
        fb = _create_triangular_filter_bank(fft_freqs, filter_freqs)
        expected = np.array(
            [
                [0.0, 0.0],
                [0.5, 0.0],
                [1.0, 0.0],
                [0.5, 0.5],
                [0.0, 1.0],
                [0.0, 0.5],
                [0.0, 0.0],
            ]
        )
        self.assertEqual(fb.shape, (7, 2))
        np.testing.assert_allclose(fb, expected, atol=1e-12)

    def test_all_non_negative(self):
        from paddlefleet.transformers.audio_utils import (
            _create_triangular_filter_bank,
        )

        fft_freqs = np.linspace(0, 4000, 40)
        filter_freqs = np.linspace(0, 4000, 12)
        fb = _create_triangular_filter_bank(fft_freqs, filter_freqs)
        self.assertTrue(np.all(fb >= 0.0))


class TestMelFilterBank(unittest.TestCase):
    """mel_filter_bank: full matrix vs independent reimpl, slaney norm, warning."""

    def test_htk_matrix_matches_independent_reference(self):
        from paddlefleet.transformers.audio_utils import mel_filter_bank

        fb = mel_filter_bank(
            num_frequency_bins=20,
            num_mel_filters=6,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=16000,
            norm=None,
            mel_scale="htk",
        )
        ref, _ = _ref_mel_filter_bank_htk(20, 6, 0.0, 8000.0, 16000)
        self.assertEqual(fb.shape, (20, 6))
        np.testing.assert_allclose(fb, ref, rtol=1e-6, atol=1e-8)

    def test_slaney_norm_scales_each_band_by_width(self):
        # slaney norm divides each triangle by half its mel-band width:
        # enorm[k] = 2 / (f_pts[k+2] - f_pts[k]).  Verify the exact per-column
        # scaling rather than only "finite" (catches wrong axis / missing norm).
        from paddlefleet.transformers.audio_utils import mel_filter_bank

        kwargs = dict(
            num_frequency_bins=20,
            num_mel_filters=6,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=16000,
            mel_scale="htk",
        )
        fb_none = mel_filter_bank(norm=None, **kwargs)
        fb_slaney = mel_filter_bank(norm="slaney", **kwargs)
        _, f_pts = _ref_mel_filter_bank_htk(20, 6, 0.0, 8000.0, 16000)
        enorm = 2.0 / (f_pts[2:8] - f_pts[0:6])
        np.testing.assert_allclose(
            fb_slaney, fb_none * enorm[None, :], rtol=1e-6, atol=1e-10
        )

    def test_invalid_norm_raises(self):
        from paddlefleet.transformers.audio_utils import mel_filter_bank

        with self.assertRaises(ValueError):
            mel_filter_bank(
                num_frequency_bins=20,
                num_mel_filters=6,
                min_frequency=0.0,
                max_frequency=8000.0,
                sampling_rate=16000,
                norm="area",
            )

    def test_all_zero_column_warns(self):
        # Far more filters than bins -> at least one column is all zeros.
        from paddlefleet.transformers.audio_utils import mel_filter_bank

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fb = mel_filter_bank(
                num_frequency_bins=5,
                num_mel_filters=60,
                min_frequency=0.0,
                max_frequency=8000.0,
                sampling_rate=16000,
            )
        self.assertTrue((fb.max(axis=0) == 0.0).any())
        self.assertTrue(
            any(
                "mel filter has all zero values" in str(w.message)
                for w in caught
            )
        )


class TestOptimalFftLength(unittest.TestCase):
    """optimal_fft_length: smallest power of two >= window_length."""

    def test_exact_powers_and_rounding(self):
        from paddlefleet.transformers.audio_utils import optimal_fft_length

        for n in [1, 2, 3, 4, 5, 100, 256, 300, 400, 512, 513, 1000]:
            self.assertEqual(
                optimal_fft_length(n),
                _ref_optimal_fft(n),
                msg=f"window_length={n}",
            )
        # Spot-check the load-bearing DSP case: 400-sample window -> 512.
        self.assertEqual(optimal_fft_length(400), 512)


class TestWindowFunction(unittest.TestCase):
    """window_function: periodic vs symmetric shape, padding placement, errors."""

    def test_boxcar_is_all_ones(self):
        from paddlefleet.transformers.audio_utils import window_function

        win = window_function(8, name="boxcar", periodic=True)
        self.assertEqual(win.shape, (8,))
        np.testing.assert_allclose(win, np.ones(8), atol=1e-12)

    def test_hann_periodic_formula(self):
        # periodic Hann of length L == 0.5 - 0.5*cos(2*pi*n/L), n=0..L-1
        from paddlefleet.transformers.audio_utils import window_function

        L = 8
        win = window_function(L, name="hann", periodic=True)
        ref = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(L) / L)
        self.assertEqual(win.shape, (L,))
        np.testing.assert_allclose(win, ref, atol=1e-12)

    def test_hann_symmetric_formula_and_differs_from_periodic(self):
        from paddlefleet.transformers.audio_utils import window_function

        L = 8
        sym = window_function(L, name="hann", periodic=False)
        ref = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(L) / (L - 1))
        np.testing.assert_allclose(sym, ref, atol=1e-12)
        # symmetric window is endpoint-symmetric; periodic (DFT-even) is not.
        per = window_function(L, name="hann", periodic=True)
        self.assertTrue(np.allclose(sym, sym[::-1]))
        self.assertFalse(np.allclose(per, per[::-1]))

    def test_hamming_periodic_formula(self):
        from paddlefleet.transformers.audio_utils import window_function

        L = 8
        win = window_function(L, name="hamming", periodic=True)
        ref = 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(L) / L)
        np.testing.assert_allclose(win, ref, atol=1e-12)

    def test_name_aliases(self):
        from paddlefleet.transformers.audio_utils import window_function

        np.testing.assert_allclose(
            window_function(16, name="hann", periodic=True),
            window_function(16, name="hann_window", periodic=True),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            window_function(16, name="hamming", periodic=True),
            window_function(16, name="hamming_window", periodic=True),
            atol=1e-12,
        )

    def test_frame_length_centered_padding(self):
        # boxcar length 4 padded into a frame of 8, centered -> offset 2.
        from paddlefleet.transformers.audio_utils import window_function

        win = window_function(
            4, name="boxcar", periodic=True, frame_length=8, center=True
        )
        np.testing.assert_allclose(
            win, [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0], atol=1e-12
        )

    def test_frame_length_left_aligned_padding(self):
        from paddlefleet.transformers.audio_utils import window_function

        win = window_function(
            4, name="boxcar", periodic=True, frame_length=8, center=False
        )
        np.testing.assert_allclose(
            win, [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], atol=1e-12
        )

    def test_window_longer_than_frame_raises(self):
        from paddlefleet.transformers.audio_utils import window_function

        with self.assertRaises(ValueError):
            window_function(400, name="hann", frame_length=200)

    def test_invalid_name_raises(self):
        from paddlefleet.transformers.audio_utils import window_function

        with self.assertRaises(ValueError):
            window_function(400, name="blackman")


class TestPowerToDb(unittest.TestCase):
    """power_to_db: 10*log10 scaling, reference/min clamp, db_range clip."""

    def test_basic_values(self):
        from paddlefleet.transformers.audio_utils import power_to_db

        db = power_to_db(np.array([[1.0, 10.0, 100.0]]), reference=1.0)
        np.testing.assert_allclose(db, [[0.0, 10.0, 20.0]], atol=1e-6)

    def test_reference_shifts_by_10log10(self):
        from paddlefleet.transformers.audio_utils import power_to_db

        db = power_to_db(np.array([[10.0, 100.0]]), reference=10.0)
        np.testing.assert_allclose(db, [[0.0, 10.0]], atol=1e-6)

    def test_min_value_clips_before_log(self):
        # 1e-12 clipped up to min_value 1e-8 -> 10*log10(1e-8) = -80 dB
        from paddlefleet.transformers.audio_utils import power_to_db

        db = power_to_db(np.array([[1e-12]]), reference=1.0, min_value=1e-8)
        self.assertTrue(np.isfinite(db).all())
        np.testing.assert_allclose(db, [[-80.0]], atol=1e-6)

    def test_db_range_bounds_dynamic_range(self):
        # [1, 1e-10] -> [0, -100] dB; db_range=80 floors the min at -80.
        from paddlefleet.transformers.audio_utils import power_to_db

        db = power_to_db(np.array([1.0, 1e-10]), reference=1.0, db_range=80.0)
        np.testing.assert_allclose(db, [0.0, -80.0], atol=1e-6)
        self.assertAlmostEqual(float(db.max() - db.min()), 80.0, places=6)

    def test_invalid_arguments_raise(self):
        from paddlefleet.transformers.audio_utils import power_to_db

        one = np.array([[1.0]])
        for kwargs in (
            dict(reference=0.0),
            dict(reference=-1.0),
            dict(min_value=0.0),
            dict(min_value=-0.5),
            dict(db_range=0.0),
            dict(db_range=-10.0),
        ):
            with self.assertRaises(ValueError):
                power_to_db(one, **kwargs)


class TestAmplitudeToDb(unittest.TestCase):
    """amplitude_to_db: 20*log10 scaling and validation."""

    def test_basic_values(self):
        from paddlefleet.transformers.audio_utils import amplitude_to_db

        db = amplitude_to_db(np.array([[1.0, 10.0, 100.0]]), reference=1.0)
        np.testing.assert_allclose(db, [[0.0, 20.0, 40.0]], atol=1e-6)

    def test_reference_shifts_by_20log10(self):
        from paddlefleet.transformers.audio_utils import amplitude_to_db

        db = amplitude_to_db(np.array([[10.0, 100.0]]), reference=10.0)
        np.testing.assert_allclose(db, [[0.0, 20.0]], atol=1e-6)

    def test_db_range_bounds_dynamic_range(self):
        from paddlefleet.transformers.audio_utils import amplitude_to_db

        db = amplitude_to_db(
            np.array([1.0, 1e-5]), reference=1.0, db_range=80.0
        )
        # 20*log10 -> [0, -100]; clipped to 80 dB span.
        np.testing.assert_allclose(db, [0.0, -80.0], atol=1e-6)

    def test_invalid_arguments_raise(self):
        from paddlefleet.transformers.audio_utils import amplitude_to_db

        one = np.array([[1.0]])
        for kwargs in (
            dict(reference=0.0),
            dict(reference=-1.0),
            dict(min_value=0.0),
            dict(min_value=-0.1),
            dict(db_range=0.0),
            dict(db_range=-10.0),
        ):
            with self.assertRaises(ValueError):
                amplitude_to_db(one, **kwargs)


class TestSpectrogram(unittest.TestCase):
    """spectrogram: impulse/DC anchors, framing, preemphasis, mel, log, dB."""

    def test_impulse_gives_flat_magnitude_equal_to_window_tap(self):
        # A single unit impulse at position p, windowed, yields a buffer that is
        # window[p] at p and 0 elsewhere. |rfft(delta)| is flat across all bins,
        # so every bin equals |window[p]|.  (antipattern doc's impulse anchor)
        from paddlefleet.transformers.audio_utils import spectrogram

        window = np.arange(1.0, 9.0)  # length 8, distinct positive taps
        waveform = np.zeros(8, dtype=np.float64)
        waveform[3] = 1.0
        spec = spectrogram(
            waveform,
            window,
            frame_length=8,
            hop_length=8,
            center=False,
            power=1.0,
            onesided=True,
        )
        self.assertEqual(spec.shape, (5, 1))  # 8//2 + 1 one-sided bins
        np.testing.assert_allclose(spec, np.full((5, 1), 4.0), rtol=1e-6)

    def test_onesided_false_returns_full_bins(self):
        from paddlefleet.transformers.audio_utils import spectrogram

        window = np.arange(1.0, 9.0)
        waveform = np.zeros(8, dtype=np.float64)
        waveform[3] = 1.0
        spec = spectrogram(
            waveform,
            window,
            frame_length=8,
            hop_length=8,
            center=False,
            power=1.0,
            onesided=False,
        )
        self.assertEqual(spec.shape, (8, 1))
        np.testing.assert_allclose(spec, np.full((8, 1), 4.0), rtol=1e-6)

    def test_framing_hop_indexing_via_dc_and_nyquist(self):
        # boxcar window -> DC bin (k=0) = sum of frame samples;
        # Nyquist bin (k=N/2) = alternating sum. Verifies hop striding and the
        # number of frames independently.
        from paddlefleet.transformers.audio_utils import spectrogram

        window = np.ones(4)
        waveform = np.arange(8, dtype=np.float64)
        spec = spectrogram(
            waveform,
            window,
            frame_length=4,
            hop_length=2,
            center=False,
            power=1.0,
            onesided=True,
        )
        # num_frames = 1 + floor((8-4)/2) = 3; frames [0:4],[2:6],[4:8]
        self.assertEqual(spec.shape, (3, 3))
        np.testing.assert_allclose(spec[0], [6.0, 14.0, 22.0], rtol=1e-6)
        # |x0 - x1 + x2 - x3| = 2 for each frame
        np.testing.assert_allclose(spec[2], [2.0, 2.0, 2.0], rtol=1e-6)

    def test_preemphasis_high_pass_before_dft(self):
        # frame [1,2,3,4], preemph 0.5 -> [0.5, 1.5, 2.0, 2.5], DC sum = 6.5.
        from paddlefleet.transformers.audio_utils import spectrogram

        window = np.ones(4)
        waveform = np.array([1.0, 2.0, 3.0, 4.0])
        spec = spectrogram(
            waveform,
            window,
            frame_length=4,
            hop_length=4,
            center=False,
            power=1.0,
            preemphasis=0.5,
            onesided=True,
        )
        self.assertEqual(spec.shape, (3, 1))
        self.assertAlmostEqual(float(spec[0, 0]), 6.5, places=6)

    def test_mel_projection_orientation(self):
        # Flat spectrum of ones (impulse at tap 0 with unit window) projected by
        # mel_filters.T -> each mel row is the column-sum of the filter bank.
        from paddlefleet.transformers.audio_utils import spectrogram

        window = np.zeros(8)
        window[0] = 1.0
        waveform = np.zeros(8, dtype=np.float64)
        waveform[0] = 1.0
        mel_filters = np.arange(15.0).reshape(5, 3)  # (num_freq_bins, num_mel)
        spec = spectrogram(
            waveform,
            window,
            frame_length=8,
            hop_length=8,
            center=False,
            power=1.0,
            mel_filters=mel_filters,
            onesided=True,
        )
        self.assertEqual(spec.shape, (3, 1))
        np.testing.assert_allclose(
            spec[:, 0], mel_filters.sum(axis=0), rtol=1e-6
        )

    def test_log_and_log10(self):
        from paddlefleet.transformers.audio_utils import spectrogram

        window = np.zeros(8)
        window[2] = 100.0
        waveform = np.zeros(8, dtype=np.float64)
        waveform[2] = 1.0  # flat spectrum of 100.0
        spec10 = spectrogram(
            waveform,
            window,
            frame_length=8,
            hop_length=8,
            center=False,
            power=1.0,
            log_mel="log10",
        )
        np.testing.assert_allclose(spec10, np.full((5, 1), 2.0), atol=1e-5)

        window[2] = 1.0
        spec = spectrogram(
            waveform,
            window,
            frame_length=8,
            hop_length=8,
            center=False,
            power=1.0,
            log_mel="log",
        )  # log(1.0) == 0
        np.testing.assert_allclose(spec, np.zeros((5, 1)), atol=1e-6)

    def test_db_uses_amplitude_scale_for_power_one(self):
        from paddlefleet.transformers.audio_utils import spectrogram

        window = np.zeros(8)
        window[1] = 10.0
        waveform = np.zeros(8, dtype=np.float64)
        waveform[1] = 1.0  # flat spectrum of 10.0
        spec = spectrogram(
            waveform,
            window,
            frame_length=8,
            hop_length=8,
            center=False,
            power=1.0,
            log_mel="dB",
        )  # amplitude_to_db(10) = 20*log10(10) = 20
        np.testing.assert_allclose(spec, np.full((5, 1), 20.0), atol=1e-4)

    def test_center_padding_equals_manual_reflect_pad(self):
        # center=True must reflect-pad by frame_length//2 on each side; verify
        # against an independently reflect-padded waveform run with center=False.
        from paddlefleet.transformers.audio_utils import spectrogram

        rng = np.random.RandomState(0)
        waveform = rng.randn(64).astype(np.float64)
        window = np.hanning(17)[:-1].astype(np.float64)  # length 16
        common = dict(frame_length=16, hop_length=8, power=1.0, onesided=True)
        centered = spectrogram(waveform, window, center=True, **common)
        padded = np.pad(waveform, 8, mode="reflect")
        manual = spectrogram(padded, window, center=False, **common)
        np.testing.assert_allclose(centered, manual, rtol=1e-6, atol=1e-8)

    def test_power_none_returns_complex(self):
        from paddlefleet.transformers.audio_utils import spectrogram

        window = np.ones(4)
        waveform = np.arange(8, dtype=np.float64)
        spec = spectrogram(
            waveform,
            window,
            frame_length=4,
            hop_length=2,
            center=False,
            power=None,
        )
        self.assertTrue(np.iscomplexobj(spec))

    def test_dtype_is_applied_for_log(self):
        from paddlefleet.transformers.audio_utils import spectrogram

        window = np.ones(4)
        waveform = np.arange(8, dtype=np.float64)
        spec = spectrogram(
            waveform,
            window,
            frame_length=4,
            hop_length=2,
            center=False,
            power=1.0,
            log_mel="log",
            dtype=np.float64,
        )
        self.assertEqual(spec.dtype, np.float64)

    def test_validation_errors(self):
        from paddlefleet.transformers.audio_utils import spectrogram

        win4 = np.ones(4)
        wf = np.arange(8, dtype=np.float64)
        with self.assertRaises(ValueError):  # frame_length > fft_length
            spectrogram(wf, win4, frame_length=4, hop_length=2, fft_length=2)
        with self.assertRaises(ValueError):  # window length != frame_length
            spectrogram(wf, np.ones(3), frame_length=4, hop_length=2)
        with self.assertRaises(ValueError):  # hop_length == 0
            spectrogram(wf, win4, frame_length=4, hop_length=0)
        with self.assertRaises(ValueError):  # hop_length < 0
            spectrogram(wf, win4, frame_length=4, hop_length=-1)
        with self.assertRaises(ValueError):  # multi-dim waveform
            spectrogram(np.ones((2, 8)), win4, frame_length=4, hop_length=2)
        with self.assertRaises(ValueError):  # complex waveform
            spectrogram(
                wf.astype(np.complex64), win4, frame_length=4, hop_length=2
            )
        with self.assertRaises(ValueError):  # bad log_mel option
            spectrogram(
                wf,
                win4,
                frame_length=4,
                hop_length=2,
                power=1.0,
                log_mel="mystery",
            )
        with self.assertRaises(ValueError):  # dB with power 3.0
            spectrogram(
                wf,
                win4,
                frame_length=4,
                hop_length=2,
                power=3.0,
                log_mel="dB",
            )


class TestDeprecated(unittest.TestCase):
    """Deprecated shims: FutureWarning contract plus real framing/DFT behavior."""

    def test_get_mel_filter_banks_warns_and_delegates(self):
        from paddlefleet.transformers.audio_utils import (
            get_mel_filter_banks,
            mel_filter_bank,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = get_mel_filter_banks(
                nb_frequency_bins=30,
                nb_mel_filters=10,
                frequency_min=100.0,
                frequency_max=7000.0,
                sample_rate=16000,
                norm="slaney",
                mel_scale="htk",
            )
        self.assertTrue(
            any("deprecated" in str(w.message).lower() for w in caught)
        )
        # The parameter remapping (nb_->num_, frequency_*->*_frequency,
        # sample_rate->sampling_rate) must reach mel_filter_bank unchanged.
        expected = mel_filter_bank(
            num_frequency_bins=30,
            num_mel_filters=10,
            min_frequency=100.0,
            max_frequency=7000.0,
            sampling_rate=16000,
            norm="slaney",
            mel_scale="htk",
        )
        self.assertEqual(result.shape, (30, 10))
        np.testing.assert_array_equal(result, expected)

    def test_fram_wave_no_center_frames(self):
        from paddlefleet.transformers.audio_utils import fram_wave

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            waveform = np.arange(6, dtype=np.float64)
            frames = fram_wave(
                waveform, hop_length=2, fft_window_size=4, center=False
            )
        self.assertTrue(
            any("deprecated" in str(w.message).lower() for w in caught)
        )
        expected = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0, 5.0],
                [4.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        self.assertEqual(frames.shape, (4, 4))
        np.testing.assert_allclose(frames, expected, atol=1e-12)

    def test_stft_dc_and_nyquist_bins(self):
        from paddlefleet.transformers.audio_utils import stft

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            frames = np.array([[1.0, 2.0, 3.0, 4.0]])
            spec = stft(frames, np.ones(4))
        self.assertTrue(
            any("deprecated" in str(w.message).lower() for w in caught)
        )
        self.assertEqual(spec.shape, (3, 1))  # (fft_window_size>>1)+1 = 3
        self.assertAlmostEqual(spec[0, 0].real, 10.0, places=5)  # DC = sum
        self.assertAlmostEqual(spec[0, 0].imag, 0.0, places=5)
        self.assertAlmostEqual(spec[2, 0].real, -2.0, places=5)  # alt-sum

    def test_stft_larger_fft_window_size(self):
        from paddlefleet.transformers.audio_utils import stft

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            frames = np.ones((2, 4))
            spec = stft(frames, np.ones(4), fft_window_size=8)
        self.assertEqual(spec.shape, (5, 2))  # (8>>1)+1 = 5 bins
        np.testing.assert_allclose(spec[0].real, [4.0, 4.0], atol=1e-5)

    def test_stft_none_window(self):
        from paddlefleet.transformers.audio_utils import stft

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            spec = stft(np.array([[1.0, 2.0, 3.0, 4.0]]), None)
        self.assertEqual(spec.shape, (3, 1))
        self.assertAlmostEqual(spec[0, 0].real, 10.0, places=5)

    def test_stft_fft_smaller_than_frame_raises(self):
        from paddlefleet.transformers.audio_utils import stft

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with self.assertRaises(ValueError):
                stft(np.ones((2, 4)), np.ones(4), fft_window_size=2)


if __name__ == "__main__":
    unittest.main()
