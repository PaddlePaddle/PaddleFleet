#!/usr/bin/env python3
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

"""Behavior tests for the UE8M0 scale rounding used by the FP8 MoE path.

Target under test: paddlefleet.transformer.moe.fp8_utils.ceil_to_ue8m0.

ceil_to_ue8m0 reinterprets |x| as an IEEE-754 float32, bumps the biased
exponent by one whenever the mantissa is non-zero (i.e. x is not already an
exact power of two), zeros the mantissa, and returns the result. Behaviorally
this is: return the smallest power of two >= |x| (for normal magnitudes). This
is exactly the "round scale up to a power of two" contract that the UE8M0 scale
format requires, and it is fully observable on CPU.

Expected values below are hand-derived from the power-of-two definition, never
by calling ceil_to_ue8m0 to produce its own reference. Every expected value is
an exact power of two and therefore exactly representable in float32, so exact
equality is asserted.
"""

import unittest

import numpy as np

try:
    import paddle

    from paddlefleet.transformer.moe.fp8_utils import ceil_to_ue8m0

    _IMPORT_ERROR = None
except ImportError as exc:  # local env has no paddle / no built ops
    paddle = None
    ceil_to_ue8m0 = None
    _IMPORT_ERROR = exc


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle / paddlefleet.transformer.moe.fp8_utils not importable: "
    f"{_IMPORT_ERROR}",
)
class TestCeilToUe8m0(unittest.TestCase):
    """CPU-observable numeric behavior of ceil_to_ue8m0."""

    # (input, smallest power of two >= |input|) -- all hand-derived.
    #   3.0 = 1.5 * 2^1  -> mantissa != 0 -> 2^2  = 4.0
    #   5.0 = 1.25 * 2^2 -> mantissa != 0 -> 2^3  = 8.0
    #   6.0 = 1.5 * 2^2  -> mantissa != 0 -> 2^3  = 8.0
    #   7.0 = 1.75 * 2^2 -> mantissa != 0 -> 2^3  = 8.0
    #   0.75 = 1.5 * 2^-1-> mantissa != 0 -> 2^0  = 1.0
    #   0.1  = 1.6 * 2^-4-> mantissa != 0 -> 2^-3 = 0.125
    #   448.0 = 1.75 * 2^8 -> mantissa != 0 -> 2^9 = 512.0 (FP8 e4m3 amax)
    _CASES = [
        (1.0, 1.0),
        (2.0, 2.0),
        (3.0, 4.0),
        (4.0, 4.0),
        (5.0, 8.0),
        (6.0, 8.0),
        (7.0, 8.0),
        (8.0, 8.0),
        (0.5, 0.5),
        (0.75, 1.0),
        (0.25, 0.25),
        (0.1, 0.125),
        (448.0, 512.0),
    ]

    def test_rounds_up_to_power_of_two(self):
        """Each input maps to the smallest power of two >= its magnitude."""
        xs = np.array([c[0] for c in self._CASES], dtype=np.float32)
        expected = np.array([c[1] for c in self._CASES], dtype=np.float32)

        out = ceil_to_ue8m0(paddle.to_tensor(xs)).numpy()

        # Exact powers of two are exactly representable; require exact equality.
        np.testing.assert_array_equal(out, expected)

    def test_exact_powers_of_two_are_unchanged(self):
        """Values that are already powers of two must be returned unchanged."""
        powers = np.array([2.0**e for e in range(-6, 10)], dtype=np.float32)
        out = ceil_to_ue8m0(paddle.to_tensor(powers)).numpy()
        np.testing.assert_array_equal(out, powers)

    def test_takes_absolute_value(self):
        """Negative inputs round the same as their absolute value."""
        neg = np.array([-3.0, -0.75, -6.0, -448.0], dtype=np.float32)
        expected = np.array([4.0, 1.0, 8.0, 512.0], dtype=np.float32)
        out = ceil_to_ue8m0(paddle.to_tensor(neg)).numpy()
        np.testing.assert_array_equal(out, expected)

    def test_result_is_a_tight_power_of_two_upper_bound(self):
        """Independent invariant: out is a power of two, out >= |x|, out/2 < |x|.

        Uses non-power-of-two magnitudes so the strict lower bound out/2 < |x|
        holds; this rejects both "too small" (rounded down) and "too large"
        (over-rounded) implementations without reusing the function's own output
        as the reference.
        """
        xs = np.array(
            [0.3, 1.1, 2.6, 9.0, 17.0, 100.0, 300.0], dtype=np.float32
        )
        out = ceil_to_ue8m0(paddle.to_tensor(xs)).numpy().astype(np.float64)
        mags = np.abs(xs).astype(np.float64)

        # power of two: log2 is an integer
        log2 = np.log2(out)
        np.testing.assert_array_equal(log2, np.round(log2))
        # tight upper bound
        self.assertTrue(np.all(out >= mags), (out, mags))
        self.assertTrue(np.all(out / 2.0 < mags), (out / 2.0, mags))


if __name__ == "__main__":
    unittest.main()
