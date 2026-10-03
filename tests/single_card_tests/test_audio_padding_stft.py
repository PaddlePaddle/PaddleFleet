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

import cmath
import unittest
import warnings

import numpy as np

from paddlefleet.transformers.audio_utils import fram_wave, stft


class TestPaddedAudioSpectrum(unittest.TestCase):
    def test_partial_and_empty_tail_frames_keep_their_fourier_values(self):
        samples = [[1, 2, 3, 4], [4, 5, 6, 0], [0, 0, 0, 0]]
        window = np.array([1.0, 0.5, 0.25, 0.125])
        for dtype in [np.float32, np.float64]:
            for fft_size in [4, 8]:
                with self.subTest(dtype=dtype, fft_size=fft_size):
                    waveform = np.arange(1, 7, dtype=dtype)
                    waveform.flags.writeable = False
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FutureWarning)
                        frames = fram_wave(
                            waveform,
                            hop_length=3,
                            fft_window_size=4,
                            center=False,
                        )
                        actual = stft(frames, window, fft_window_size=fft_size)
                    expected = np.array(
                        [
                            [
                                sum(
                                    sample
                                    * window[t]
                                    * cmath.exp(
                                        -2j * cmath.pi * k * t / fft_size
                                    )
                                    for t, sample in enumerate(frame)
                                )
                                for frame in samples
                            ]
                            for k in range(fft_size // 2 + 1)
                        ]
                    )
                    np.testing.assert_array_equal(frames, samples)
                    self.assertEqual(frames.dtype, dtype)
                    self.assertEqual(actual.dtype, np.complex64)
                    np.testing.assert_allclose(
                        actual, expected, rtol=1e-6, atol=1e-6
                    )
                    np.testing.assert_array_equal(waveform, [1, 2, 3, 4, 5, 6])


if __name__ == "__main__":
    unittest.main()
