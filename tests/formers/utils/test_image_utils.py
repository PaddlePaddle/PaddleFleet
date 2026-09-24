# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

"""Behavior tests for paddlefleet.utils.image_utils.

These tests exercise the real image-processing operators end to end and
compare their output against independently hand-derived expected values
(pixel content, channel order, tensor layout, normalization arithmetic),
never against values produced by the function under test itself.
"""

import base64
import os
import tempfile
import unittest
from io import BytesIO

import numpy as np
from PIL import Image

from paddlefleet.utils.image_utils import (
    Bbox,
    DecodeImage,
    NormalizeImage,
    PadBatch,
    Permute,
    ResizeImage,
    check,
    img2base64,
    np2base64,
    pil2base64,
    two_dimension_sort_box,
    two_dimension_sort_layout,
)


def _png_bytes(arr):
    """Encode an HWC uint8 array as lossless PNG bytes (independent helper)."""
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


class TestDecodeImageContent(unittest.TestCase):
    def setUp(self):
        # Distinct per-pixel, per-channel values so channel swaps / axis
        # transposes are detectable. Lossless PNG keeps content exact.
        self.arr = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        self.png = _png_bytes(self.arr)
        self.b64 = base64.b64encode(self.png).decode("utf-8")

    def test_decode_from_base64_preserves_pixels(self):
        result = DecodeImage()({"im_base64": self.b64})
        out = result["image"]
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.dtype, np.uint8)
        # RGB channel order and every pixel value preserved.
        np.testing.assert_array_equal(out, self.arr)

    def test_decode_from_raw_bytes_preserves_pixels(self):
        result = DecodeImage()({"image": self.png})
        np.testing.assert_array_equal(result["image"], self.arr)

    def test_decode_sets_h_w_and_im_info(self):
        result = DecodeImage()({"im_base64": self.b64})
        self.assertEqual(result["h"], 2)
        self.assertEqual(result["w"], 3)
        self.assertEqual(result["h"], result["image"].shape[0])
        self.assertEqual(result["w"], result["image"].shape[1])
        info = result["im_info"]
        self.assertEqual(info.dtype, np.float32)
        np.testing.assert_array_equal(info, np.array([2, 3, 1.0], np.float32))

    def test_decode_overrides_mismatched_h_w(self):
        result = DecodeImage()({"im_base64": self.b64, "h": 999, "w": 999})
        # Real decoded size wins over caller-supplied mismatched values.
        self.assertEqual(result["h"], 2)
        self.assertEqual(result["w"], 3)


class TestResizeImageContent(unittest.TestCase):
    def test_nearest_2x_upscale_block_replication(self):
        # 2x2 image with four distinct colors; NEAREST 2x upscale must
        # replicate each source pixel into a 2x2 output block.
        img = np.array(
            [
                [[10, 20, 30], [40, 50, 60]],
                [[70, 80, 90], [100, 110, 120]],
            ],
            dtype=np.uint8,
        )
        out = ResizeImage(target_size=4, interp=0)({"image": img.copy()})[
            "image"
        ]
        self.assertEqual(out.shape, (4, 4, 3))

        expected = np.zeros((4, 4, 3), dtype=out.dtype)
        expected[0:2, 0:2] = img[0, 0]
        expected[0:2, 2:4] = img[0, 1]
        expected[2:4, 0:2] = img[1, 0]
        expected[2:4, 2:4] = img[1, 1]
        # Catches axis swaps, channel reorder and wrong interpolation family.
        np.testing.assert_array_equal(out, expected)

    def test_solid_color_preserved_and_square(self):
        # Non-square input becomes a square target_size x target_size image;
        # a solid color is preserved exactly regardless of interpolation.
        img = np.full((7, 4, 3), (13, 200, 77), dtype=np.uint8)
        out = ResizeImage(target_size=5)({"image": img.copy()})["image"]
        self.assertEqual(out.shape, (5, 5, 3))
        np.testing.assert_array_equal(
            out, np.full((5, 5, 3), (13, 200, 77), dtype=out.dtype)
        )

    def test_list_target_selects_size(self):
        img = np.full((6, 9, 3), (5, 15, 25), dtype=np.uint8)
        out = ResizeImage(target_size=[8])({"image": img.copy()})["image"]
        self.assertEqual(out.shape, (8, 8, 3))
        np.testing.assert_array_equal(
            out, np.full((8, 8, 3), (5, 15, 25), dtype=out.dtype)
        )

    def test_invalid_target_size_type_raises(self):
        with self.assertRaises(TypeError):
            ResizeImage(target_size=3.5)

    def test_non_numpy_image_raises(self):
        with self.assertRaises(TypeError):
            ResizeImage(target_size=8)({"image": [[1, 2, 3]]})

    def test_zero_min_side_raises(self):
        img = np.zeros((0, 4, 3), dtype=np.uint8)
        with self.assertRaises(ZeroDivisionError):
            ResizeImage(target_size=8)({"image": img})


class TestPermuteContent(unittest.TestCase):
    def setUp(self):
        # H=2, W=3, C=3 -> all axes distinct so a wrong transpose changes
        # shape and content.
        self.img = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)

    def test_hwc_to_chw_no_bgr(self):
        out = Permute(to_bgr=False)({"image": self.img.copy()})["image"]
        self.assertEqual(out.shape, (3, 2, 3))
        np.testing.assert_array_equal(out, np.transpose(self.img, (2, 0, 1)))

    def test_hwc_to_chw_with_bgr(self):
        out = Permute(to_bgr=True)({"image": self.img.copy()})["image"]
        expected = np.transpose(self.img, (2, 0, 1))[[2, 1, 0], :, :]
        self.assertEqual(out.shape, (3, 2, 3))
        np.testing.assert_array_equal(out, expected)

    def test_batch_input_each_sample_transposed(self):
        a = self.img.copy()
        b = (self.img + 100).copy()
        result = Permute(to_bgr=False)([{"image": a}, {"image": b}])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(
            result[0]["image"], np.transpose(self.img, (2, 0, 1))
        )
        np.testing.assert_array_equal(
            result[1]["image"], np.transpose(self.img + 100, (2, 0, 1))
        )

    def test_missing_image_key_raises(self):
        with self.assertRaises(AssertionError):
            Permute(to_bgr=False)({"data": np.zeros(5)})


class TestNormalizeImageContent(unittest.TestCase):
    def test_channel_first_scale_and_normalize(self):
        raw = np.array(
            [
                [[0.0, 255.0]],
                [[255.0, 0.0]],
                [[128.0, 64.0]],
            ],
            dtype=np.float32,
        )  # shape (C=3, H=1, W=2)
        mean = [0.0, 0.5, 1.0]
        std = [1.0, 2.0, 4.0]
        out = NormalizeImage(
            mean=mean, std=std, is_channel_first=True, is_scale=True
        )({"image": raw.copy()})["image"]
        expected = (
            raw / 255.0 - np.array(mean, np.float32).reshape(3, 1, 1)
        ) / np.array(std, np.float32).reshape(3, 1, 1)
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    def test_channel_last_scale_and_normalize(self):
        raw = np.array(
            [[[0.0, 255.0, 128.0], [255.0, 0.0, 64.0]]], dtype=np.float32
        )  # shape (H=1, W=2, C=3)
        mean = [0.1, 0.2, 0.3]
        std = [0.5, 0.25, 2.0]
        out = NormalizeImage(
            mean=mean, std=std, is_channel_first=False, is_scale=True
        )({"image": raw.copy()})["image"]
        expected = (
            raw / 255.0 - np.array(mean, np.float32).reshape(1, 1, 3)
        ) / np.array(std, np.float32).reshape(1, 1, 3)
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    def test_no_scale_subtracts_mean_directly(self):
        raw = np.array([[[10.0]], [[20.0]], [[30.0]]], dtype=np.float32)
        mean = [1.0, 2.0, 3.0]
        std = [2.0, 4.0, 5.0]
        out = NormalizeImage(
            mean=mean, std=std, is_channel_first=True, is_scale=False
        )({"image": raw.copy()})["image"]
        expected = (
            raw - np.array(mean, np.float32).reshape(3, 1, 1)
        ) / np.array(std, np.float32).reshape(3, 1, 1)
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    def test_zero_std_raises(self):
        with self.assertRaises(ValueError):
            NormalizeImage(mean=[0.5], std=[0])


class TestPadBatchContent(unittest.TestCase):
    def test_pads_to_stride_preserving_content(self):
        img1 = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
        img2 = (np.arange(2 * 2 * 5, dtype=np.float32) + 100).reshape(2, 2, 5)
        samples = [
            {
                "image": img1.copy(),
                "im_info": np.array([3, 3, 1.0], np.float32),
            },
            {
                "image": img2.copy(),
                "im_info": np.array([2, 5, 1.0], np.float32),
            },
        ]
        out = PadBatch(pad_to_stride=4, use_padded_im_info=True)(samples)

        # max H=3 -> 4, max W=5 -> 8, both divisible by stride 4.
        exp1 = np.zeros((2, 4, 8), dtype=np.float32)
        exp1[:, :3, :3] = img1
        exp2 = np.zeros((2, 4, 8), dtype=np.float32)
        exp2[:, :2, :5] = img2
        self.assertEqual(out[0]["image"].shape, (2, 4, 8))
        self.assertEqual(out[1]["image"].shape, (2, 4, 8))
        # Original content in the top-left corner, zeros elsewhere.
        np.testing.assert_array_equal(out[0]["image"], exp1)
        np.testing.assert_array_equal(out[1]["image"], exp2)
        # im_info updated to padded H, W.
        self.assertEqual(out[0]["im_info"][0], 4)
        self.assertEqual(out[0]["im_info"][1], 8)

    def test_zero_stride_returns_unchanged(self):
        img = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        samples = [{"image": img, "im_info": np.array([3, 4, 1.0], np.float32)}]
        out = PadBatch(pad_to_stride=0)(samples)
        self.assertIs(out[0]["image"], img)
        np.testing.assert_array_equal(out[0]["image"], img)


class TestCheck(unittest.TestCase):
    def test_english_and_digits_true(self):
        self.assertTrue(check("Hello"))
        self.assertTrue(check("123"))
        self.assertTrue(check("abc123"))
        self.assertTrue(check("a中"))  # any ascii letter/digit -> True

    def test_non_alphanumeric_false(self):
        self.assertFalse(check("中文"))
        self.assertFalse(check(""))
        self.assertFalse(check("!@#"))


class TestBase64Helpers(unittest.TestCase):
    def test_img2base64_encodes_whole_file(self):
        data = b"raw-image-bytes-\x00\x01\x02\xff\xfe payload"
        fd, path = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        try:
            with open(path, "wb") as f:
                f.write(data)
            result = img2base64(path)
            # Independent reference: base64 of the exact bytes written.
            self.assertEqual(result, base64.b64encode(data).decode("utf-8"))
        finally:
            os.unlink(path)

    def test_np2base64_roundtrip_size_and_color(self):
        arr = np.full((16, 24, 3), (30, 90, 150), dtype=np.uint8)
        b64 = np2base64(arr)
        img = Image.open(BytesIO(base64.b64decode(b64)))
        # HWC (16, 24) -> PIL size (width=24, height=16); default JPEG.
        self.assertEqual(img.size, (24, 16))
        self.assertEqual(img.format, "JPEG")
        decoded = np.asarray(img.convert("RGB"), dtype=np.float32)
        # Solid color survives JPEG within a small tolerance; distinct
        # per-channel means (spaced 60 apart) catch channel reordering.
        np.testing.assert_allclose(
            decoded.mean(axis=(0, 1)), [30, 90, 150], atol=15
        )

    def test_pil2base64_png_is_lossless(self):
        arr = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        b64 = pil2base64(Image.fromarray(arr), image_type="PNG")
        img = Image.open(BytesIO(base64.b64decode(b64)))
        self.assertEqual(img.format, "PNG")
        np.testing.assert_array_equal(np.asarray(img.convert("RGB")), arr)

    def test_pil2base64_default_is_jpeg(self):
        img_in = Image.new("RGB", (10, 20), color=(0, 0, 255))
        b64 = pil2base64(img_in)
        img = Image.open(BytesIO(base64.b64decode(b64)))
        self.assertEqual(img.format, "JPEG")
        self.assertEqual(img.size, (10, 20))

    def test_pil2base64_size_flag_returns_original_size(self):
        img_in = Image.new("RGB", (10, 20), color=(0, 0, 255))
        result, size = pil2base64(img_in, size=True)
        self.assertIsInstance(result, str)
        self.assertEqual(size, (10, 20))


class TestBbox(unittest.TestCase):
    def test_derived_geometry(self):
        b = Bbox(left=10, top=20, width=30, height=40)
        self.assertEqual(b.right, 40)
        self.assertEqual(b.bottom, 60)
        self.assertEqual(b.area(), 1200)
        self.assertEqual(b.center(), (25.0, 40.0))
        self.assertEqual(b.points(), ((10, 20), (40, 60)))
        self.assertEqual(b.list_int(), [10, 20, 30, 40])

    def test_negative_dims_rejected(self):
        with self.assertRaises(AssertionError):
            Bbox(left=0, top=0, width=-1, height=10)
        with self.assertRaises(AssertionError):
            Bbox(left=0, top=0, width=10, height=-1)

    def test_from_points_recovers_box(self):
        b = Bbox.from_points((10, 20), (40, 60))
        self.assertEqual((b.left, b.top, b.width, b.height), (10, 20, 30, 40))

    def test_union_and_intersection(self):
        a = Bbox(left=0, top=0, width=10, height=10)
        b = Bbox(left=5, top=5, width=10, height=10)
        u = Bbox.union(a, b)
        self.assertEqual((u.left, u.top, u.right, u.bottom), (0, 0, 15, 15))
        i = Bbox.intersection(a, b)
        self.assertEqual((i.left, i.top, i.right, i.bottom), (5, 5, 10, 10))
        self.assertEqual(i.area(), 25)

    def test_intersection_disjoint_is_empty(self):
        a = Bbox(left=0, top=0, width=10, height=10)
        b = Bbox(left=20, top=20, width=10, height=10)
        self.assertEqual(Bbox.intersection(a, b).area(), 0)

    def test_iou_values(self):
        a = Bbox(left=0, top=0, width=10, height=10)
        self.assertAlmostEqual(Bbox.iou(a, a), 1.0)
        # This iou divides intersection area by the ENCLOSING bbox area
        # (Bbox.union), not the set-union: intersection 25, enclosing 15x15.
        b = Bbox(left=5, top=5, width=10, height=10)
        self.assertAlmostEqual(Bbox.iou(a, b), 25.0 / 225.0)

    def test_hoverlap_and_hdistance(self):
        a = Bbox(left=0, top=0, width=20, height=10)
        b = Bbox(left=10, top=0, width=20, height=10)
        self.assertEqual(a.hoverlap(b), 10)
        far = Bbox(left=40, top=0, width=10, height=10)
        self.assertEqual(a.hdistance(far), 20)

    def test_translate_moves_origin_only(self):
        moved = Bbox(left=10, top=20, width=30, height=40).translate((5, 10))
        self.assertEqual(
            (moved.left, moved.top, moved.width, moved.height), (15, 30, 30, 40)
        )


class TestTwoDimensionSort(unittest.TestCase):
    def test_horizontal_ordering_returns_left_delta(self):
        b1 = Bbox(left=0, top=0, width=10, height=10)
        b2 = Bbox(left=20, top=0, width=10, height=10)
        # Vertically overlapping -> compares left first: 0 - 20 = -20.
        self.assertEqual(two_dimension_sort_box(b1, b2), -20)

    def test_vertical_ordering_returns_top_delta(self):
        b1 = Bbox(left=0, top=0, width=10, height=5)
        b2 = Bbox(left=0, top=20, width=10, height=5)
        # No vertical overlap -> compares top first: 0 - 20 = -20.
        self.assertEqual(two_dimension_sort_box(b1, b2), -20)

    def test_layout_delegates_to_bbox(self):
        l1 = {"bbox": Bbox(left=0, top=0, width=10, height=10)}
        l2 = {"bbox": Bbox(left=20, top=0, width=10, height=10)}
        self.assertEqual(two_dimension_sort_layout(l1, l2), -20)


class TestBboxCrossBoundaryBug(unittest.TestCase):
    @unittest.expectedFailure
    def test_is_cross_boundary_correct_semantics(self):
        # A box spanning x[5,25], y[5,25].
        box = Bbox(left=5, top=5, width=20, height=20)
        # It extends beyond a 10x10 region, so it DOES cross that boundary.
        self.assertTrue(box.is_cross_boundary(10, 10))
        # It fits entirely inside a 30x30 region, so it does NOT cross it.
        self.assertFalse(box.is_cross_boundary(30, 30))
        # BUG: is_cross_boundary returns `boundary.contain(self)`, i.e. it
        # reports True when the box is *inside* the boundary and False when it
        # crosses -- the inverse of its name. It also builds the boundary via
        # Bbox(top, left, width, height), swapping the top/left arguments.


if __name__ == "__main__":
    unittest.main()
