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

"""Behavior tests for paddlefleet.transformers.image_transforms.

These tests verify the actual pixel/content produced by each transform and the
channel ordering, not just output shapes. Expected results are derived by hand
on tiny arrays (or from numpy/trusted-library semantics that the production code
merely selects), so that a permutation of channels, a transposed axis, a wrong
crop offset, a swapped H/W, or an off-by-one padding border would be rejected.
"""

import unittest

import numpy as np
import paddle
import PIL.Image

from paddlefleet.transformers.image_transforms import (
    PaddingMode,
    center_crop,
    center_to_corners_format,
    convert_to_rgb,
    corners_to_center_format,
    get_resize_output_image_size,
    get_size_with_aspect_ratio,
    group_images_by_shape,
    id_to_rgb,
    normalize,
    pad,
    reorder_images,
    rescale,
    resize,
    rgb_to_id,
    to_channel_dimension_format,
    to_pil_image,
)
from paddlefleet.transformers.image_utils import ChannelDimension


def setUpModule():
    # CPU-only: this suite exercises numpy/PIL math and CPU paddle tensors.
    paddle.set_device("cpu")


class TestToChannelDimensionFormat(unittest.TestCase):
    """to_channel_dimension_format must actually transpose pixels, not relabel."""

    def test_last_to_first_reorders_pixels(self):
        # HWC image with fully distinguishable values (H=4, W=2, C=3 -> LAST inferred).
        image = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)
        result = to_channel_dimension_format(image, ChannelDimension.FIRST)
        self.assertEqual(result.shape, (3, 4, 2))
        # Every pixel must map result[c, h, w] == image[h, w, c].
        np.testing.assert_array_equal(result, image.transpose(2, 0, 1))
        for c in range(3):
            for h in range(4):
                for w in range(2):
                    self.assertEqual(result[c, h, w], image[h, w, c])

    def test_first_to_last_reorders_pixels(self):
        image = np.arange(3 * 4 * 2, dtype=np.float32).reshape(3, 4, 2)
        result = to_channel_dimension_format(image, ChannelDimension.LAST)
        self.assertEqual(result.shape, (4, 2, 3))
        np.testing.assert_array_equal(result, image.transpose(1, 2, 0))

    def test_noop_when_already_target(self):
        image = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)
        result = to_channel_dimension_format(image, ChannelDimension.LAST)
        # Already channels-last: returned unchanged (same content and object).
        self.assertIs(result, image)

    def test_string_channel_dim_matches_enum(self):
        image = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)
        result = to_channel_dimension_format(image, "channels_first")
        np.testing.assert_array_equal(result, image.transpose(2, 0, 1))

    def test_rejects_non_numpy(self):
        with self.assertRaises(ValueError):
            to_channel_dimension_format([[1, 2], [3, 4]], ChannelDimension.LAST)


class TestRescale(unittest.TestCase):
    """rescale multiplies by scale, casts dtype, and optionally reorders axes."""

    def test_multiplies_every_element(self):
        image = np.array([1.0, 2.0, 3.0, -4.0], dtype=np.float32)
        result = rescale(image, 2.0)
        np.testing.assert_array_almost_equal(result, [2.0, 4.0, 6.0, -8.0])

    def test_dtype_override(self):
        image = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = rescale(image, 0.5, dtype=np.float64)
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_array_almost_equal(result, [0.5, 1.0, 1.5])

    def test_scale_then_reformat_applies_in_order(self):
        # CHW input, scale by 10, then request channels-last output.
        image = np.arange(3 * 4 * 2, dtype=np.float32).reshape(3, 4, 2)
        result = rescale(image, 10.0, data_format=ChannelDimension.LAST)
        expected = (image * 10.0).transpose(1, 2, 0)
        self.assertEqual(result.shape, (4, 2, 3))
        np.testing.assert_array_almost_equal(result, expected)

    def test_rejects_non_numpy(self):
        with self.assertRaises(ValueError):
            rescale([[1, 2], [3, 4]], 2.0)


class TestNormalize(unittest.TestCase):
    """normalize computes (x - mean) / std per channel along the true axis."""

    def test_per_channel_last_format(self):
        # Distinct constant per channel; distinct mean/std -> distinct outputs,
        # so a channel swap or wrong broadcast axis is caught.
        image = np.zeros((2, 2, 3), dtype=np.float32)
        image[..., 0] = 10.0
        image[..., 1] = 20.0
        image[..., 2] = 30.0
        result = normalize(image, mean=[1.0, 2.0, 3.0], std=[2.0, 5.0, 10.0])
        self.assertEqual(result.shape, (2, 2, 3))
        np.testing.assert_array_almost_equal(result[..., 0], 4.5)  # (10-1)/2
        np.testing.assert_array_almost_equal(result[..., 1], 3.6)  # (20-2)/5
        np.testing.assert_array_almost_equal(result[..., 2], 2.7)  # (30-3)/10

    def test_per_channel_first_format(self):
        image = np.zeros((3, 2, 2), dtype=np.float32)
        image[0] = 10.0
        image[1] = 20.0
        image[2] = 30.0
        result = normalize(image, mean=[1.0, 2.0, 3.0], std=[2.0, 5.0, 10.0])
        self.assertEqual(result.shape, (3, 2, 2))
        np.testing.assert_array_almost_equal(result[0], 4.5)
        np.testing.assert_array_almost_equal(result[1], 3.6)
        np.testing.assert_array_almost_equal(result[2], 2.7)

    def test_scalar_mean_std_broadcast(self):
        image = np.ones((2, 2, 3), dtype=np.float32) * 0.5
        result = normalize(image, mean=0.5, std=0.5)
        np.testing.assert_array_almost_equal(result, np.zeros((2, 2, 3)))

    def test_wrong_mean_length_raises(self):
        image = np.ones((2, 2, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            normalize(image, mean=[0.5, 0.5], std=[0.5, 0.5, 0.5])

    def test_wrong_std_length_raises(self):
        image = np.ones((2, 2, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            normalize(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5])


class TestCenterCrop(unittest.TestCase):
    """center_crop extracts the centered region, padding with zeros if needed."""

    def test_crop_extracts_centered_region(self):
        # 4x4x3 with unique pixels; crop 2x2 -> rows 1:3, cols 1:3 (centered).
        image = np.arange(4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3)
        result = center_crop(image, (2, 2))
        self.assertEqual(result.shape, (2, 2, 3))
        np.testing.assert_array_equal(result, image[1:3, 1:3, :])

    def test_crop_to_output_format(self):
        image = np.arange(4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3)
        result = center_crop(image, (2, 2), data_format=ChannelDimension.FIRST)
        self.assertEqual(result.shape, (3, 2, 2))
        np.testing.assert_array_equal(
            result, image[1:3, 1:3, :].transpose(2, 0, 1)
        )

    def test_crop_larger_than_image_pads_with_zeros(self):
        # 2x2 image, crop 4x4 -> original centered at [1:3, 1:3], borders zero.
        image = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3) + 1.0
        result = center_crop(image, (4, 4))
        self.assertEqual(result.shape, (4, 4, 3))
        np.testing.assert_array_equal(result[1:3, 1:3, :], image)
        # All border rows/cols must be zero (the pad region).
        np.testing.assert_array_equal(result[0, :, :], 0.0)
        np.testing.assert_array_equal(result[3, :, :], 0.0)
        np.testing.assert_array_equal(result[:, 0, :], 0.0)
        np.testing.assert_array_equal(result[:, 3, :], 0.0)

    def test_crop_channels_first_input(self):
        image = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
        result = center_crop(image, (2, 2))
        self.assertEqual(result.shape, (3, 2, 2))
        np.testing.assert_array_equal(result, image[:, 1:3, 1:3])

    def test_invalid_size_raises(self):
        image = np.arange(4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3)
        with self.assertRaises(ValueError):
            center_crop(image, (2,))


class TestGetSizeWithAspectRatio(unittest.TestCase):
    """get_size_with_aspect_ratio returns exact (h, w), hand-derived."""

    def test_matched_short_edge_no_scaling(self):
        # width(100) <= height(200) and width == size -> unchanged.
        self.assertEqual(
            get_size_with_aspect_ratio((200, 100), 100), (200, 100)
        )
        self.assertEqual(
            get_size_with_aspect_ratio((100, 200), 100), (100, 200)
        )

    def test_scales_long_edge_to_preserve_ratio(self):
        # (h=400, w=200), size=100: short edge (w) -> 100, h -> 100*400/200 = 200.
        self.assertEqual(
            get_size_with_aspect_ratio((400, 200), 100), (200, 100)
        )

    def test_max_size_overrides_short_edge(self):
        # 2*256 = 512 > 400 -> raw = 400*400/800 = 200; size=200.
        # w<h: ow = 200, oh = int(200*800/400) = 400.
        self.assertEqual(
            get_size_with_aspect_ratio((800, 400), 256, max_size=400),
            (400, 200),
        )

    def test_max_size_not_triggered(self):
        # 2*256 = 512, not > 512 -> size stays 256; w<h: ow=256, oh=512.
        self.assertEqual(
            get_size_with_aspect_ratio((800, 400), 256, max_size=512),
            (512, 256),
        )


class TestGetResizeOutputImageSize(unittest.TestCase):
    """get_resize_output_image_size resolves the target (h, w), hand-derived."""

    def test_explicit_tuple_passthrough(self):
        image = np.random.rand(100, 200, 3).astype(np.float32)
        self.assertEqual(
            get_resize_output_image_size(image, (50, 100)), (50, 100)
        )

    def test_int_default_square(self):
        image = np.random.rand(100, 200, 3).astype(np.float32)
        self.assertEqual(
            get_resize_output_image_size(image, 224, default_to_square=True),
            (224, 224),
        )

    def test_single_element_list_is_square(self):
        image = np.random.rand(100, 200, 3).astype(np.float32)
        self.assertEqual(get_resize_output_image_size(image, [224]), (224, 224))

    def test_int_non_square_scales_long_edge(self):
        # image (h=100, w=200); short edge -> 100, long -> 100*200/100 = 200.
        image = np.random.rand(100, 200, 3).astype(np.float32)
        result = get_resize_output_image_size(
            image, 100, default_to_square=False
        )
        self.assertEqual(result, (100, 200))

    def test_max_size_shrinks_both_edges(self):
        # new_long 200 > max 150 -> new_short = int(150*100/200)=75, new_long=150.
        image = np.random.rand(100, 200, 3).astype(np.float32)
        result = get_resize_output_image_size(
            image, 100, default_to_square=False, max_size=150
        )
        self.assertEqual(result, (75, 150))

    def test_invalid_size_list_raises(self):
        image = np.random.rand(100, 200, 3).astype(np.float32)
        with self.assertRaises(ValueError):
            get_resize_output_image_size(image, [1, 2, 3])

    def test_max_size_too_small_raises(self):
        image = np.random.rand(100, 200, 3).astype(np.float32)
        with self.assertRaises(ValueError):
            get_resize_output_image_size(
                image, 100, default_to_square=False, max_size=50
            )


class TestResize(unittest.TestCase):
    """resize honors (height, width) order and preserves channel semantics."""

    def test_output_hw_order_and_channel_constants(self):
        # Constant-per-channel image: bilinear keeps each channel constant, so a
        # channel swap or an H/W swap in the target is detectable.
        image = np.zeros((8, 4, 3), dtype=np.uint8)
        image[..., 0] = 10
        image[..., 1] = 20
        image[..., 2] = 30
        result = resize(image, (2, 6))  # size == (height, width)
        self.assertEqual(result.shape, (2, 6, 3))  # not (6, 2, 3)
        np.testing.assert_array_equal(result[..., 0], 10)
        np.testing.assert_array_equal(result[..., 1], 20)
        np.testing.assert_array_equal(result[..., 2], 30)

    def test_output_channels_first(self):
        image = np.zeros((8, 4, 3), dtype=np.uint8)
        image[..., 0] = 10
        image[..., 1] = 20
        image[..., 2] = 30
        result = resize(image, (2, 6), data_format=ChannelDimension.FIRST)
        self.assertEqual(result.shape, (3, 2, 6))
        np.testing.assert_array_equal(result[0], 10)
        np.testing.assert_array_equal(result[1], 20)
        np.testing.assert_array_equal(result[2], 30)

    def test_return_pil(self):
        image = np.zeros((8, 4, 3), dtype=np.uint8)
        result = resize(image, (2, 6), return_numpy=False)
        self.assertIsInstance(result, PIL.Image.Image)
        self.assertEqual(result.size, (6, 2))  # PIL size is (width, height)

    def test_invalid_size_raises(self):
        image = np.zeros((8, 4, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            resize(image, (50,))


class TestToPilImage(unittest.TestCase):
    """to_pil_image preserves content and optionally rescales floats to uint8."""

    def test_uint8_numpy_preserves_pixels(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        image[0, 0] = [10, 20, 30]
        image[0, 1] = [40, 50, 60]
        image[1, 0] = [70, 80, 90]
        image[1, 1] = [100, 110, 120]
        result = to_pil_image(image, do_rescale=False)
        self.assertIsInstance(result, PIL.Image.Image)
        np.testing.assert_array_equal(np.array(result), image)

    def test_float_rescaled_to_0_255(self):
        image = np.array(
            [[[0.0, 0.5, 1.0]], [[1.0, 0.5, 0.0]]], dtype=np.float32
        )  # shape (2, 1, 3)
        result = to_pil_image(image)  # do_rescale defaults True for floats
        arr = np.array(result)
        # 0.0 -> 0, 0.5 -> 127 (0.5*255 = 127.5, truncated by uint8 cast), 1.0 -> 255
        np.testing.assert_array_equal(
            arr, np.array([[[0, 127, 255]], [[255, 127, 0]]], dtype=np.uint8)
        )

    def test_single_channel_squeezed_to_mode_l(self):
        image = np.arange(4, dtype=np.uint8).reshape(2, 2, 1)
        result = to_pil_image(image, do_rescale=False)
        self.assertEqual(len(result.split()), 1)
        np.testing.assert_array_equal(np.array(result), image[:, :, 0])

    def test_paddle_tensor_channels_first(self):
        # CHW int tensor: content must survive move to channels-last uint8.
        arr = np.arange(3 * 2 * 2, dtype=np.int64).reshape(3, 2, 2)
        tensor = paddle.to_tensor(arr)
        result = to_pil_image(tensor, do_rescale=False)
        self.assertIsInstance(result, PIL.Image.Image)
        np.testing.assert_array_equal(
            np.array(result), arr.transpose(1, 2, 0).astype(np.uint8)
        )

    def test_pil_passthrough(self):
        img = PIL.Image.new("RGB", (4, 4), color=(1, 2, 3))
        self.assertIs(to_pil_image(img), img)

    def test_invalid_input_raises(self):
        with self.assertRaises(ValueError):
            to_pil_image([[1, 2], [3, 4]])


class TestPad(unittest.TestCase):
    """pad selects the correct numpy mode and never pads the channel axis."""

    def test_constant_places_value_in_border(self):
        # (2, 2, 1) channels-last; pad 1 with zeros -> original centered.
        image = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)
        result = pad(image, padding=1, mode=PaddingMode.CONSTANT)
        self.assertEqual(result.shape, (4, 4, 1))
        np.testing.assert_array_equal(
            result[1:3, 1:3, 0], [[1.0, 2.0], [3.0, 4.0]]
        )
        np.testing.assert_array_equal(result[0, :, 0], 0.0)
        np.testing.assert_array_equal(result[3, :, 0], 0.0)
        np.testing.assert_array_equal(result[:, 0, 0], 0.0)
        np.testing.assert_array_equal(result[:, 3, 0], 0.0)

    def test_constant_values_fill_border(self):
        image = np.zeros((2, 2, 3), dtype=np.float32)
        result = pad(
            image, padding=1, mode=PaddingMode.CONSTANT, constant_values=7.0
        )
        self.assertEqual(result.shape, (4, 4, 3))  # channel count preserved
        self.assertEqual(result[0, 0, 0], 7.0)
        self.assertEqual(result[0, 0, 2], 7.0)

    def test_channel_axis_not_padded(self):
        image = np.ones((2, 2, 3), dtype=np.float32)
        result = pad(image, padding=1, mode=PaddingMode.CONSTANT)
        self.assertEqual(result.shape, (4, 4, 3))  # 3 channels, not 5

    def test_border_modes_on_single_row(self):
        # channels-first (C=1, H=1, W=4); pad width by 2, no height/channel pad.
        image = np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)
        expected = {
            PaddingMode.REFLECT: [3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0],
            PaddingMode.REPLICATE: [1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0],
            PaddingMode.SYMMETRIC: [2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 3.0],
        }
        for mode, exp in expected.items():
            result = pad(
                image,
                padding=((0, 0), (2, 2)),
                mode=mode,
                input_data_format=ChannelDimension.FIRST,
            )
            self.assertEqual(result.shape, (1, 1, 8), mode)
            np.testing.assert_array_equal(
                result[0, 0, :], exp, err_msg=str(mode)
            )

    def test_output_data_format_conversion(self):
        image = np.ones((2, 2, 3), dtype=np.float32)
        result = pad(
            image,
            padding=1,
            mode=PaddingMode.CONSTANT,
            data_format=ChannelDimension.FIRST,
        )
        self.assertEqual(result.shape, (3, 4, 4))


class TestConvertToRgb(unittest.TestCase):
    """convert_to_rgb only converts PIL images; other types pass through."""

    def test_pil_grayscale_to_rgb_replicates_channels(self):
        img = PIL.Image.new("L", (2, 2), color=137)
        result = convert_to_rgb(img)
        self.assertEqual(result.mode, "RGB")
        arr = np.array(result)
        self.assertEqual(arr.shape, (2, 2, 3))
        # Luminance replicated across all three channels.
        np.testing.assert_array_equal(arr[..., 0], 137)
        np.testing.assert_array_equal(arr[..., 1], 137)
        np.testing.assert_array_equal(arr[..., 2], 137)

    def test_numpy_passthrough_same_object(self):
        image = np.random.rand(4, 4, 3).astype(np.float32)
        self.assertIs(convert_to_rgb(image), image)

    def test_paddle_passthrough_same_object(self):
        tensor = paddle.randn([3, 4, 4])
        self.assertIs(convert_to_rgb(tensor), tensor)


class TestBoxFormatConversion(unittest.TestCase):
    """center<->corners conversions, hand-derived, for numpy and paddle."""

    def test_center_to_corners_numpy(self):
        # center (cx, cy, w, h) -> (cx-w/2, cy-h/2, cx+w/2, cy+h/2).
        boxes = np.array([[50.0, 50.0, 20.0, 40.0]])
        result = center_to_corners_format(boxes)
        np.testing.assert_array_almost_equal(result, [[40.0, 30.0, 60.0, 70.0]])

    def test_center_to_corners_paddle(self):
        boxes = paddle.to_tensor([[50.0, 50.0, 20.0, 40.0]])
        result = center_to_corners_format(boxes)
        np.testing.assert_array_almost_equal(
            result.numpy(), [[40.0, 30.0, 60.0, 70.0]]
        )

    def test_corners_to_center_numpy(self):
        boxes = np.array([[40.0, 30.0, 60.0, 70.0]])
        result = corners_to_center_format(boxes)
        np.testing.assert_array_almost_equal(result, [[50.0, 50.0, 20.0, 40.0]])

    def test_corners_to_center_paddle(self):
        boxes = paddle.to_tensor([[40.0, 30.0, 60.0, 70.0]])
        result = corners_to_center_format(boxes)
        np.testing.assert_array_almost_equal(
            result.numpy(), [[50.0, 50.0, 20.0, 40.0]]
        )

    def test_roundtrip_multiple_boxes(self):
        original = np.array(
            [[50.0, 50.0, 20.0, 40.0], [100.0, 90.0, 30.0, 50.0]]
        )
        recovered = corners_to_center_format(center_to_corners_format(original))
        np.testing.assert_array_almost_equal(original, recovered)

    def test_unsupported_type_raises(self):
        with self.assertRaises(ValueError):
            center_to_corners_format([[50, 50, 20, 40]])
        with self.assertRaises(ValueError):
            corners_to_center_format([[40, 30, 60, 70]])


class TestRgbIdConversion(unittest.TestCase):
    """rgb_to_id / id_to_rgb use id = r + 256*g + 65536*b; channel order matters."""

    def test_scalar_channel_weights(self):
        self.assertEqual(rgb_to_id([1, 2, 3]), 1 + 256 * 2 + 65536 * 3)

    def test_array_channel_order(self):
        # (255,0,0) -> 255 ; (0,255,0) -> 255*256 : distinguishes r vs g vs b.
        color = np.array(
            [[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8
        )
        result = rgb_to_id(color)
        self.assertEqual(result.shape, (1, 3))
        self.assertEqual(int(result[0, 0]), 255)
        self.assertEqual(int(result[0, 1]), 255 * 256)
        self.assertEqual(int(result[0, 2]), 255 * 65536)

    def test_id_to_rgb_scalar(self):
        self.assertEqual(id_to_rgb(1 + 256 * 2 + 65536 * 3), [1, 2, 3])

    def test_id_to_rgb_array(self):
        id_map = np.array([[1 + 256 * 2 + 65536 * 3]])
        result = id_to_rgb(id_map)
        self.assertEqual(result.shape, (1, 1, 3))
        self.assertEqual(result[0, 0].tolist(), [1, 2, 3])

    def test_roundtrip(self):
        color = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
        recovered = id_to_rgb(rgb_to_id(color))
        np.testing.assert_array_equal(color, recovered)


class TestGroupAndReorderImages(unittest.TestCase):
    """group_images_by_shape groups by trailing shape; reorder restores order."""

    def _images(self):
        a = paddle.to_tensor(np.arange(12, dtype="float32").reshape(3, 2, 2))
        b = paddle.to_tensor(
            (np.arange(12, dtype="float32") + 100).reshape(3, 2, 2)
        )
        c = paddle.to_tensor(
            (np.arange(48, dtype="float32") + 1000).reshape(3, 4, 4)
        )
        return a, b, c

    def test_grouping_by_shape_and_index_mapping(self):
        a, b, c = self._images()
        grouped, index = group_images_by_shape(
            [a, b, c], disable_grouping=False
        )
        # a and b share trailing shape (2, 2) and are stacked together.
        self.assertEqual(tuple(grouped[(2, 2)].shape), (2, 3, 2, 2))
        self.assertEqual(tuple(grouped[(4, 4)].shape), (1, 3, 4, 4))
        self.assertEqual(index[0], ((2, 2), 0))
        self.assertEqual(index[1], ((2, 2), 1))
        self.assertEqual(index[2], ((4, 4), 0))
        # The stacked group preserves each image's content.
        np.testing.assert_array_equal(grouped[(2, 2)][0].numpy(), a.numpy())
        np.testing.assert_array_equal(grouped[(2, 2)][1].numpy(), b.numpy())

    def test_reorder_restores_original_order_and_content(self):
        a, b, c = self._images()
        grouped, index = group_images_by_shape(
            [a, b, c], disable_grouping=False
        )
        reordered = reorder_images(grouped, index)
        self.assertEqual(len(reordered), 3)
        np.testing.assert_array_equal(reordered[0].numpy(), a.numpy())
        np.testing.assert_array_equal(reordered[1].numpy(), b.numpy())
        np.testing.assert_array_equal(reordered[2].numpy(), c.numpy())

    def test_disable_grouping_keeps_per_image_and_roundtrips(self):
        a, b, c = self._images()
        grouped, index = group_images_by_shape([a, b, c], disable_grouping=True)
        # Each image is kept separate with a leading batch dim of 1.
        self.assertEqual(tuple(grouped[0].shape), (1, 3, 2, 2))
        self.assertEqual(index[0], (0, 0))
        reordered = reorder_images(grouped, index)
        np.testing.assert_array_equal(reordered[0].numpy(), a.numpy())
        np.testing.assert_array_equal(reordered[1].numpy(), b.numpy())
        np.testing.assert_array_equal(reordered[2].numpy(), c.numpy())


if __name__ == "__main__":
    unittest.main()
