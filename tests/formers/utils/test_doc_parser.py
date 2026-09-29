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

"""Behavior tests for paddlefleet.utils.doc_parser.DocParser.

These exercise the CPU-only geometry / IO helpers with small, known inputs and
compare against independently hand-derived expected structure and content (no
paddleocr / GPU needed). Expected values are computed by hand in the comments,
never by calling the function under test.
"""

import base64
import os
import tempfile
import unittest
from io import BytesIO

import numpy as np
from PIL import Image

from paddlefleet.utils.doc_parser import DocParser


def _write_png(array):
    """Save a uint8 array as a lossless PNG temp file; return its path."""
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    Image.fromarray(array).save(path, format="PNG")
    return path


class TestNormalizeBox(unittest.TestCase):
    def test_scales_x_and_y_independently(self):
        # old=(w=100,h=400), new=(w=50,h=100): x-scale=0.5, y-scale=0.25.
        # box=[11,41,31,81] ->
        #   x1=int(11*50/100)=int(5.5)=5   y1=int(41*100/400)=int(10.25)=10
        #   x2=int(31*50/100)=int(15.5)=15 y2=int(81*100/400)=int(20.25)=20
        result = DocParser._normalize_box(
            [11, 41, 31, 81], (100, 400), (50, 100)
        )
        self.assertEqual(result, [5, 10, 15, 20])
        for v in result:
            self.assertIsInstance(v, int)

    def test_offset_applied_to_matching_axis(self):
        # offset_x applies to box[0]/box[2], offset_y to box[1]/box[3].
        # box=[10,40,30,80], offset_x=6, offset_y=8, same scales as above ->
        #   x1=int((10+6)*0.5)=8   y1=int((40+8)*0.25)=12
        #   x2=int((30+6)*0.5)=18  y2=int((80+8)*0.25)=22
        result = DocParser._normalize_box(
            [10, 40, 30, 80], (100, 400), (50, 100), offset_x=6, offset_y=8
        )
        self.assertEqual(result, [8, 12, 18, 22])


class TestExpandImageToA4Size(unittest.TestCase):
    # Fill original with 0..199 pattern so it is distinguishable from the
    # 255-valued padding and any transpose/misplacement is caught.
    def _pattern(self, h, w):
        return (np.arange(h * w * 3) % 200).astype("uint8").reshape(h, w, 3)

    def test_tall_centered_pads_left_and_right(self):
        # h=500,w=200: h/w=2.5>=1.42 -> exp_w=int(500/1.414-200)=153,
        # offset_x=int(153/2)=76. hstack([pad76, orig200, pad76]) width=352.
        arr = self._pattern(500, 200)
        out, ox, oy = DocParser.expand_image_to_a4_size(arr, center=True)
        self.assertEqual(out.shape, (500, 352, 3))
        self.assertEqual((ox, oy), (76, 0))
        np.testing.assert_array_equal(out[:, :76, :], 255)
        np.testing.assert_array_equal(out[:, 76:276, :], arr)
        np.testing.assert_array_equal(out[:, 276:, :], 255)

    def test_tall_not_centered_pads_right_only(self):
        # center=False: exp_w=153, hstack([orig, pad153]) width=353, offsets 0.
        arr = self._pattern(500, 200)
        out, ox, oy = DocParser.expand_image_to_a4_size(arr, center=False)
        self.assertEqual(out.shape, (500, 353, 3))
        self.assertEqual((ox, oy), (0, 0))
        np.testing.assert_array_equal(out[:, :200, :], arr)
        np.testing.assert_array_equal(out[:, 200:, :], 255)

    def test_wide_centered_pads_top_and_bottom(self):
        # h=200,w=300: h/w=0.667<=1.40 -> exp_h=int(300*1.414-200)=224,
        # offset_y=int(224/2)=112. vstack([pad112, orig200, pad112]) height=424.
        arr = self._pattern(200, 300)
        out, ox, oy = DocParser.expand_image_to_a4_size(arr, center=True)
        self.assertEqual(out.shape, (424, 300, 3))
        self.assertEqual((ox, oy), (0, 112))
        np.testing.assert_array_equal(out[:112, :, :], 255)
        np.testing.assert_array_equal(out[112:312, :, :], arr)
        np.testing.assert_array_equal(out[312:, :, :], 255)

    def test_wide_not_centered_pads_bottom_only(self):
        # center=False: exp_h=224, vstack([orig, pad224]) height=424, offsets 0.
        arr = self._pattern(200, 300)
        out, ox, oy = DocParser.expand_image_to_a4_size(arr, center=False)
        self.assertEqual(out.shape, (424, 300, 3))
        self.assertEqual((ox, oy), (0, 0))
        np.testing.assert_array_equal(out[:200, :, :], arr)
        np.testing.assert_array_equal(out[200:, :, :], 255)

    def test_near_a4_ratio_is_unchanged(self):
        # h=424,w=300: ratio=1.4133 in (1.40,1.42) -> no padding, offsets 0.
        arr = self._pattern(424, 300)
        out, ox, oy = DocParser.expand_image_to_a4_size(arr, center=True)
        self.assertEqual(out.shape, (424, 300, 3))
        self.assertEqual((ox, oy), (0, 0))
        np.testing.assert_array_equal(out, arr)


class TestGetBuffer(unittest.TestCase):
    def test_reads_exact_bytes_from_short_path(self):
        payload = b"hello world \x00\x01\x02\xff"
        fd, path = tempfile.mkstemp(suffix=".bin")
        os.write(fd, payload)
        os.close(fd)
        self.addCleanup(os.unlink, path)
        # short path that exists -> file bytes returned verbatim.
        buff = DocParser._get_buffer(path)
        self.assertEqual(buff, payload)

    def test_file_like_returns_readable_stream_with_same_bytes(self):
        payload = b"stream-bytes-\x10\x20\x30"
        fd, path = tempfile.mkstemp(suffix=".bin")
        os.write(fd, payload)
        os.close(fd)
        self.addCleanup(os.unlink, path)
        stream = DocParser._get_buffer(path, file_like=True)
        self.assertIsInstance(stream, BytesIO)
        self.assertEqual(stream.read(), payload)

    def test_long_base64_string_is_decoded(self):
        # data >= 1024 chars skips the path check and is base64-decoded.
        payload = bytes(range(256)) * 4  # 1024 known bytes
        b64 = base64.b64encode(payload).decode("utf-8")
        self.assertGreaterEqual(len(b64), 1024)  # forces the decode branch
        buff = DocParser._get_buffer(b64)
        self.assertEqual(buff, payload)

    def test_missing_short_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            DocParser._get_buffer("/no/such/kiro-doc-parser-file.jpg")


class TestReadImage(unittest.TestCase):
    def test_rgb_content_and_orientation_preserved(self):
        # H=3, W=5 distinct-valued RGB image; PNG is lossless.
        arr = np.arange(3 * 5 * 3, dtype="uint8").reshape(3, 5, 3)
        path = _write_png(arr)
        self.addCleanup(os.unlink, path)
        result = DocParser.read_image(path)
        self.assertEqual(result.shape, (3, 5, 3))
        self.assertEqual(result.dtype, np.uint8)
        np.testing.assert_array_equal(result, arr)

    def test_grayscale_is_converted_to_replicated_rgb(self):
        gray = np.arange(16, dtype="uint8").reshape(4, 4)  # mode "L"
        path = _write_png(gray)
        self.addCleanup(os.unlink, path)
        result = DocParser.read_image(path)
        # convert("RGB") replicates luminance into all three channels.
        expected = np.stack([gray, gray, gray], axis=-1)
        self.assertEqual(result.shape, (4, 4, 3))
        np.testing.assert_array_equal(result, expected)


class TestParseWithoutOcr(unittest.TestCase):
    def test_parse_image_metadata_no_expand(self):
        # W=5, H=3 image; do_ocr=False avoids the paddleocr dependency.
        arr = np.arange(3 * 5 * 3, dtype="uint8").reshape(3, 5, 3)
        path = _write_png(arr)
        self.addCleanup(os.unlink, path)
        doc = {"doc": path}
        result = DocParser().parse(doc, expand_to_a4_size=False, do_ocr=False)
        self.assertIs(result, doc)  # parse mutates and returns the same dict
        self.assertEqual(result["img_w"], 5)
        self.assertEqual(result["img_h"], 3)
        self.assertEqual(result["offset_x"], 0)
        self.assertEqual(result["offset_y"], 0)
        self.assertNotIn("layout", result)  # OCR skipped
        # doc["image"] is a JPEG re-encoding; decode it and confirm dimensions.
        decoded = base64.b64decode(result["image"])
        stored = Image.open(BytesIO(decoded))
        self.assertEqual(stored.size, (5, 3))  # PIL size is (width, height)

    def test_parse_with_a4_expand_propagates_offsets(self):
        # Tall H=500,W=200 -> expand adds offset_x=76, width 352 (see A4 tests).
        arr = (
            (np.arange(500 * 200 * 3) % 200)
            .astype("uint8")
            .reshape(500, 200, 3)
        )
        path = _write_png(arr)
        self.addCleanup(os.unlink, path)
        result = DocParser().parse(
            {"doc": path}, expand_to_a4_size=True, do_ocr=False
        )
        self.assertEqual(result["img_w"], 352)
        self.assertEqual(result["img_h"], 500)
        self.assertEqual(result["offset_x"], 76)
        self.assertEqual(result["offset_y"], 0)
        decoded = base64.b64decode(result["image"])
        stored = Image.open(BytesIO(decoded))
        self.assertEqual(stored.size, (352, 500))


if __name__ == "__main__":
    unittest.main()
