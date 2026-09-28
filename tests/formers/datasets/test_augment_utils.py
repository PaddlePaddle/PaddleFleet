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

import io
import random
import unittest

import numpy as np
from PIL import Image

from paddlefleet.datasets.template.augment_utils import (
    JpegCompression,
    RandomApply,
    RandomDiscreteRotation,
    RandomScale,
    RandomSingleSidePadding,
)


def make_image(height, width, seed=0):
    """Build a content-distinguishable RGB image and its backing array.

    Pixels are pseudo-random so any rotation, flip, crop or pixel move is
    observable, plus three asymmetric anchor corners make orientation
    unambiguous. Returns ``(PIL.Image, uint8 ndarray of shape (H, W, 3))``.
    """
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
    arr[0, 0] = [255, 0, 0]  # top-left red
    arr[0, width - 1] = [0, 255, 0]  # top-right green
    arr[height - 1, 0] = [0, 0, 255]  # bottom-left blue
    return Image.fromarray(arr), arr


def add_delta(delta):
    """Real, deterministic transform: add ``delta`` to every channel (clipped)."""

    def _t(img):
        arr = np.asarray(img, dtype=np.int16) + delta
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    return _t


def scale_values(factor):
    """Real, deterministic transform: multiply every channel (clipped)."""

    def _t(img):
        arr = np.asarray(img, dtype=np.int16) * factor
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    return _t


class TestRandomApply(unittest.TestCase):
    """RandomApply must run the wrapped transforms, in order, gated by ``p``."""

    def test_probability_one_applies_transform(self):
        img, arr = make_image(4, 5)
        out = RandomApply([add_delta(30)], p=1.0)(img)
        # p=1.0 => random.random() < 1.0 always true => transform runs once.
        expected = np.clip(arr.astype(np.int16) + 30, 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(np.asarray(out), expected)

    def test_probability_zero_skips_transform(self):
        img, arr = make_image(4, 5)
        out = RandomApply([add_delta(30)], p=0.0)(img)
        # p=0.0 => random.random() < 0.0 never true => the input is returned
        # untouched (not the transformed version).
        self.assertIs(out, img)
        np.testing.assert_array_equal(np.asarray(out), arr)

    def test_transforms_compose_in_order(self):
        img, arr = make_image(3, 3)
        out = RandomApply([add_delta(30), scale_values(2)], p=1.0)(img)
        # Independent expectation: scale(add(x)), clipping at each stage.
        after_add = np.clip(arr.astype(np.int16) + 30, 0, 255)
        expected = np.clip(after_add * 2, 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(np.asarray(out), expected)
        # Order is load-bearing: the reversed composition differs on this fixture
        # (e.g. the red anchor [255,0,0] -> [255,60,0] vs [255,30,30]).
        reversed_order = np.clip(
            np.clip(arr.astype(np.int16) * 2, 0, 255) + 30, 0, 255
        ).astype(np.uint8)
        self.assertFalse(np.array_equal(expected, reversed_order))


class TestRandomDiscreteRotation(unittest.TestCase):
    """Rotation must be an exact discrete turn of the input content."""

    def setUp(self):
        self.addCleanup(random.setstate, random.getstate())

    def test_zero_degree_returns_identical_content(self):
        img, arr = make_image(3, 5)
        out = RandomDiscreteRotation(degrees=[0])(img)
        self.assertEqual(out.size, img.size)
        np.testing.assert_array_equal(np.asarray(out), arr)

    def test_180_degree_matches_numpy_reference(self):
        img, arr = make_image(3, 5)
        out = RandomDiscreteRotation(degrees=[180])(img)
        # A 180-degree turn reverses both spatial axes; no interpolation needed,
        # so this is an exact independent reference.
        expected = arr[::-1, ::-1, :]
        self.assertEqual(out.size, img.size)
        np.testing.assert_array_equal(np.asarray(out), expected)

    def test_output_is_exact_discrete_rotation(self):
        img, arr = make_image(3, 5)
        random.seed(1234)
        out1 = np.asarray(
            RandomDiscreteRotation(degrees=[0, 90, 180, 270])(img)
        )
        random.seed(1234)
        out2 = np.asarray(
            RandomDiscreteRotation(degrees=[0, 90, 180, 270])(img)
        )
        # Same seed + fresh instance => identical result (deterministic given RNG).
        np.testing.assert_array_equal(out1, out2)
        # The result must equal one of the four exact k*90 rotations (no blur or
        # interpolation smear). The set {rot90^k} is direction-agnostic, so this
        # holds whether the turn is clockwise or counter-clockwise.
        candidates = [np.rot90(arr, k) for k in range(4)]
        self.assertTrue(
            any(
                c.shape == out1.shape and np.array_equal(c, out1)
                for c in candidates
            ),
            "rotation output is not an exact discrete rotation of the input",
        )

    def test_expand_swaps_dimensions_for_quarter_turn(self):
        img, _ = make_image(3, 5)  # size == (width=5, height=3)
        out = RandomDiscreteRotation(degrees=[90], expand=True)(img)
        # expand=True on a quarter turn swaps width and height.
        self.assertEqual(out.size, (img.size[1], img.size[0]))


class TestJpegCompression(unittest.TestCase):
    """JPEG round-trip must actually compress, honour quality, and yield RGB."""

    def _jpeg_roundtrip(self, img, quality):
        """Independent encode/decode through PIL's JPEG codec (ground truth)."""
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=quality)
        buf.seek(0)
        return np.asarray(Image.open(buf))

    def test_output_matches_independent_jpeg_roundtrip(self):
        img, _ = make_image(16, 16, seed=3)
        # quality_range=(q, q) => random.randint(q, q) == q deterministically.
        out = JpegCompression(quality_range=(75, 75))(img)
        self.assertEqual(out.mode, "RGB")
        self.assertEqual(out.size, img.size)
        np.testing.assert_array_equal(
            np.asarray(out), self._jpeg_roundtrip(img, 75)
        )

    def test_compression_changes_high_frequency_content(self):
        img, arr = make_image(16, 16, seed=5)  # noisy => JPEG is visibly lossy
        out = np.asarray(JpegCompression(quality_range=(10, 10))(img))
        # Real compression must alter a noisy image; an identity pass would not.
        self.assertFalse(np.array_equal(out, arr))

    def test_lower_quality_deviates_more_from_original(self):
        img, arr = make_image(16, 16, seed=5)
        orig = arr.astype(np.float64)
        low = np.asarray(JpegCompression(quality_range=(5, 5))(img)).astype(
            np.float64
        )
        high = np.asarray(JpegCompression(quality_range=(95, 95))(img)).astype(
            np.float64
        )
        # The quality parameter is consumed: lower quality => larger deviation.
        self.assertGreater(
            np.abs(low - orig).mean(), np.abs(high - orig).mean()
        )

    def test_converts_non_rgb_input_to_rgb(self):
        img, _ = make_image(12, 12, seed=7)
        rgba = img.convert("RGBA")
        out = JpegCompression(quality_range=(80, 80))(rgba)
        self.assertEqual(out.mode, "RGB")
        np.testing.assert_array_equal(
            np.asarray(out), self._jpeg_roundtrip(rgba, 80)
        )


class TestRandomScale(unittest.TestCase):
    """Scaling must resize by the factor with correct width/height ordering."""

    def test_unit_scale_nearest_preserves_content(self):
        img, arr = make_image(4, 6)
        # scale_range=(1.0, 1.0) => random.uniform == 1.0; nearest resize to the
        # same size is an exact identity.
        out = RandomScale(scale_range=(1.0, 1.0), interpolation="nearest")(img)
        self.assertEqual(out.size, img.size)
        np.testing.assert_array_equal(np.asarray(out), arr)

    def test_upscale_maps_width_and_height_correctly(self):
        img, _ = make_image(4, 6)  # non-square: width=6, height=4
        out = RandomScale(scale_range=(2.0, 2.0), interpolation="nearest")(img)
        # width -> int(6*2)=12, height -> int(4*2)=8. A width/height swap in the
        # (h, w) argument order would surface here because the image is non-square.
        self.assertEqual(out.size, (12, 8))

    def test_downscale_dimensions(self):
        img, _ = make_image(10, 8)  # non-square: width=8, height=10
        out = RandomScale(scale_range=(0.5, 0.5), interpolation="nearest")(img)
        self.assertEqual(out.size, (4, 5))  # (int(8*0.5), int(10*0.5))


class TestRandomSingleSidePadding(unittest.TestCase):
    """Padding must extend exactly one side with the fill colour."""

    def setUp(self):
        self.addCleanup(random.setstate, random.getstate())

    def test_rejects_non_sequence_padding_range(self):
        with self.assertRaises(AssertionError):
            RandomSingleSidePadding(padding_range=5, fill="white")

    def test_rejects_wrong_length_padding_range(self):
        with self.assertRaises(AssertionError):
            RandomSingleSidePadding(padding_range=(1, 2, 3), fill="white")

    def test_zero_padding_returns_unchanged(self):
        img, arr = make_image(5, 7)
        # padding_range=(0, 0) => pad_amount 0 => the original is returned as-is.
        out = RandomSingleSidePadding(padding_range=(0, 0), fill="white")(img)
        self.assertIs(out, img)
        self.assertEqual(out.size, img.size)
        np.testing.assert_array_equal(np.asarray(out), arr)

    def test_pads_exactly_one_side_with_fill(self):
        img, arr = make_image(5, 7)  # height=5, width=7, content is non-white
        pad, white = 4, 255
        random.seed(2024)
        out = np.asarray(
            RandomSingleSidePadding(padding_range=(pad, pad), fill="white")(img)
        )
        h, w = arr.shape[:2]
        # Independently construct the four possible single-side white paddings.
        left = np.full((h, w + pad, 3), white, dtype=np.uint8)
        left[:, pad:, :] = arr
        right = np.full((h, w + pad, 3), white, dtype=np.uint8)
        right[:, :w, :] = arr
        top = np.full((h + pad, w, 3), white, dtype=np.uint8)
        top[pad:, :, :] = arr
        bottom = np.full((h + pad, w, 3), white, dtype=np.uint8)
        bottom[:h, :, :] = arr
        candidates = [left, right, top, bottom]
        self.assertTrue(
            any(
                c.shape == out.shape and np.array_equal(c, out)
                for c in candidates
            ),
            "output is not a single-side white padding of width 4",
        )
        # Deterministic for a fixed seed with a fresh instance.
        random.seed(2024)
        out2 = np.asarray(
            RandomSingleSidePadding(padding_range=(pad, pad), fill="white")(img)
        )
        np.testing.assert_array_equal(out, out2)


if __name__ == "__main__":
    unittest.main()
