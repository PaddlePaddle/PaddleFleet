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

"""Behavior tests for the multimodal plugin data layer.

Focus: the text placeholder <-> media-feature correspondence.  A message
carries one placeholder per image / video; ``process_messages`` must expand
each placeholder into ``bos + token * seqlen + eos`` where ``seqlen`` is
derived from *that* media's grid metadata, consuming grids strictly in
appearance order across the whole message list.  Getting the ordering, the
per-media grid, or the merge arithmetic wrong changes the produced text, so
the assertions below compare full expanded strings against independently
computed expectations rather than shapes or counts.
"""

import unittest

import paddle
from PIL import Image

from paddlefleet.datasets.template.mm_plugin import (
    IMAGE_PLACEHOLDER,
    PLUGINS,
    VIDEO_PLACEHOLDER,
    BasePlugin,
    Qwen2VLPlugin,
    _check_video_is_nested_images,
    _make_batched_images,
    get_mm_plugin,
    register_mm_plugin,
)

# The module reads placeholder strings from the environment at import time;
# bind to the real constants so counts in _validate_messages line up.
IMAGE_PH = IMAGE_PLACEHOLDER
VIDEO_PH = VIDEO_PLACEHOLDER

# Expansion token distinct from the placeholder "<image>"/"<video>" so an
# expanded string is unambiguously distinguishable from an un-expanded one.
IMG = "<IMG>"
VID = "<VID>"
VISION_BOS = "<|vision_start|>"
VISION_EOS = "<|vision_end|>"


class _StubImageProcessor:
    """Minimal image processor: only ``merge_size`` is consumed by
    ``process_messages`` (``merge_length = merge_size ** 2``)."""

    def __init__(self, merge_size):
        self.merge_size = merge_size


class _StubProcessor:
    """Stand-in for a VLM processor.

    ``process_messages`` only touches ``image_processor`` (for ``merge_size``)
    and, via ``_validate_input``, the presence of image / video processors.
    The real regularize/encode path is not exercised here because grid
    metadata is supplied directly through ``mm_inputs``, exactly as the
    caller does after running the image processor.
    """

    def __init__(self, merge_size=2):
        self.image_processor = _StubImageProcessor(merge_size)
        # video_processor falls back to image_processor inside _validate_input.
        self.model_input_names = []


def _grid(t, h, w):
    """A single image/video grid entry as the real code sees it: an int
    tensor whose ``prod().item()`` gives patch count."""
    return paddle.to_tensor([t, h, w], dtype="int64")


def _expand(token, seqlen):
    return f"{VISION_BOS}{token * seqlen}{VISION_EOS}"


def _make_qwen2vl(expand_mm_tokens=True):
    return get_mm_plugin(
        "qwen2_vl",
        image_token=IMG,
        video_token=VID,
        audio_token=None,
        expand_mm_tokens=expand_mm_tokens,
    )


class TestProcessMessagesPlaceholderCorrespondence(unittest.TestCase):
    """Qwen2VLPlugin.process_messages: placeholder <-> grid correspondence."""

    def test_image_placeholders_expand_per_grid_in_order(self):
        plugin = _make_qwen2vl()
        processor = _StubProcessor(merge_size=2)  # merge_length = 4
        # Three images with deliberately different patch counts so each
        # expanded run has a distinguishable length.
        # prod // 4 -> seqlen: [1,2,4]=8->2, [1,4,4]=16->4, [1,2,2]=4->1
        mm_inputs = {
            "image_grid_thw": [_grid(1, 2, 4), _grid(1, 4, 4), _grid(1, 2, 2)]
        }
        images = [object(), object(), object()]
        messages = [
            {"role": "user", "content": f"A {IMAGE_PH} B {IMAGE_PH} C"},
            {"role": "user", "content": f"D {IMAGE_PH} E"},
        ]

        out = plugin.process_messages(
            messages, images, [], [], mm_inputs, processor
        )

        e0, e1, e2 = _expand(IMG, 2), _expand(IMG, 4), _expand(IMG, 1)
        self.assertEqual(out[0]["content"], f"A {e0} B {e1} C")
        self.assertEqual(out[1]["content"], f"D {e2} E")

    def test_grid_order_matters(self):
        # Swapping the two grids must change the produced text: this is what
        # pins "placeholder i consumes grid i" rather than "some grid".
        plugin = _make_qwen2vl()
        processor = _StubProcessor(merge_size=2)
        images = [object(), object()]
        messages = [{"role": "user", "content": f"{IMAGE_PH} {IMAGE_PH}"}]

        forward = plugin.process_messages(
            messages,
            images,
            [],
            [],
            {"image_grid_thw": [_grid(1, 2, 4), _grid(1, 2, 2)]},
            processor,
        )
        swapped = plugin.process_messages(
            messages,
            images,
            [],
            [],
            {"image_grid_thw": [_grid(1, 2, 2), _grid(1, 2, 4)]},
            processor,
        )
        self.assertEqual(
            forward[0]["content"], f"{_expand(IMG, 2)} {_expand(IMG, 1)}"
        )
        self.assertEqual(
            swapped[0]["content"], f"{_expand(IMG, 1)} {_expand(IMG, 2)}"
        )
        self.assertNotEqual(forward[0]["content"], swapped[0]["content"])

    def test_video_placeholders_expand_per_grid(self):
        plugin = _make_qwen2vl()
        processor = _StubProcessor(merge_size=2)  # merge_length = 4
        # [2,2,4]=16->4, [2,2,2]=8->2
        mm_inputs = {"video_grid_thw": [_grid(2, 2, 4), _grid(2, 2, 2)]}
        videos = [object(), object()]
        messages = [
            {"role": "user", "content": f"start {VIDEO_PH} mid {VIDEO_PH} end"}
        ]

        out = plugin.process_messages(
            messages, [], videos, [], mm_inputs, processor
        )

        v0, v1 = _expand(VID, 4), _expand(VID, 2)
        self.assertEqual(out[0]["content"], f"start {v0} mid {v1} end")

    def test_mixed_image_and_video_in_one_message(self):
        plugin = _make_qwen2vl()
        processor = _StubProcessor(merge_size=2)
        mm_inputs = {
            "image_grid_thw": [_grid(1, 2, 2)],  # 4 -> seqlen 1
            "video_grid_thw": [_grid(2, 2, 2)],  # 8 -> seqlen 2
        }
        messages = [
            {"role": "user", "content": f"img {IMAGE_PH} vid {VIDEO_PH} done"}
        ]

        out = plugin.process_messages(
            messages, [object()], [object()], [], mm_inputs, processor
        )

        self.assertEqual(
            out[0]["content"],
            f"img {_expand(IMG, 1)} vid {_expand(VID, 2)} done",
        )

    def test_expand_disabled_emits_single_token(self):
        # With expansion off, grid metadata is ignored and every placeholder
        # collapses to exactly one media token wrapped in vision markers.
        plugin = _make_qwen2vl(expand_mm_tokens=False)
        processor = _StubProcessor(merge_size=2)
        images = [object(), object()]
        messages = [{"role": "user", "content": f"x {IMAGE_PH} y {IMAGE_PH} z"}]

        out = plugin.process_messages(messages, images, [], [], {}, processor)

        one = _expand(IMG, 1)
        self.assertEqual(out[0]["content"], f"x {one} y {one} z")

    def test_input_messages_not_mutated(self):
        plugin = _make_qwen2vl()
        processor = _StubProcessor(merge_size=2)
        messages = [{"role": "user", "content": f"keep {IMAGE_PH} here"}]
        original = f"keep {IMAGE_PH} here"

        out = plugin.process_messages(
            messages,
            [object()],
            [],
            [],
            {"image_grid_thw": [_grid(1, 2, 2)]},
            processor,
        )

        self.assertEqual(messages[0]["content"], original)  # deepcopy'd
        self.assertNotEqual(out[0]["content"], original)

    def test_placeholder_count_mismatch_raises(self):
        # Two <image> placeholders but only one image is a contract violation.
        plugin = _make_qwen2vl()
        processor = _StubProcessor(merge_size=2)
        messages = [{"role": "user", "content": f"{IMAGE_PH} {IMAGE_PH}"}]
        with self.assertRaises(ValueError):
            plugin.process_messages(
                messages,
                [object()],
                [],
                [],
                {"image_grid_thw": [_grid(1, 2, 2)]},
                processor,
            )


class TestMakeBatchedImages(unittest.TestCase):
    """_make_batched_images: regroup a flat media list into per-sample lists,
    preserving element identity and order (batch-order metadata)."""

    def test_groups_by_lengths_preserving_order(self):
        images = ["a", "b", "c", "d", "e"]
        # Includes an empty middle group to pin the boundary walk.
        self.assertEqual(
            _make_batched_images(images, [2, 0, 3]),
            [["a", "b"], [], ["c", "d", "e"]],
        )

    def test_length_partition_changes_grouping(self):
        images = ["a", "b", "c", "d", "e"]
        self.assertEqual(
            _make_batched_images(images, [2, 3]), [["a", "b"], ["c", "d", "e"]]
        )
        self.assertEqual(
            _make_batched_images(images, [3, 2]), [["a", "b", "c"], ["d", "e"]]
        )


class TestCheckVideoIsNestedImages(unittest.TestCase):
    """_check_video_is_nested_images: True only for a list whose every element
    is a frame spec (str / dict / bytes-like / PIL image)."""

    def test_list_of_frame_specs_is_nested(self):
        self.assertTrue(
            _check_video_is_nested_images([{"type": "image"}, "frame.jpg"])
        )
        self.assertTrue(_check_video_is_nested_images(["a.jpg", "b.jpg"]))

    def test_pil_image_frames_are_nested(self):
        frames = [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))]
        self.assertTrue(_check_video_is_nested_images(frames))

    def test_list_with_invalid_element_is_not_nested(self):
        # A single non-frame element (int) breaks the all() guard.
        self.assertFalse(_check_video_is_nested_images(["a.jpg", 123]))

    def test_non_list_is_not_nested(self):
        self.assertFalse(_check_video_is_nested_images("video.mp4"))
        self.assertFalse(_check_video_is_nested_images(123))

    def test_empty_list_is_nested(self):
        # all() over an empty sequence is True by definition.
        self.assertTrue(_check_video_is_nested_images([]))


class TestPreprocessImage(unittest.TestCase):
    """BasePlugin._preprocess_image: pixel-budget resize and RGB conversion,
    checked against independently computed target dimensions."""

    def _plugin(self):
        return get_mm_plugin(
            "base", image_token="<image>", video_token=None, audio_token=None
        )

    def test_downscale_to_max_pixels(self):
        # 40*40 = 1600 > 400 -> factor sqrt(400/1600)=0.5 -> int(40*0.5)=20.
        image = Image.new("RGB", (40, 40))
        out = self._plugin()._preprocess_image(
            image, image_max_pixels=400, image_min_pixels=1
        )
        self.assertEqual((out.width, out.height), (20, 20))

    def test_upscale_to_min_pixels(self):
        # 4*4 = 16 < 64 -> factor sqrt(64/16)=2 -> 8x8.
        image = Image.new("RGB", (4, 4))
        out = self._plugin()._preprocess_image(
            image, image_max_pixels=10000, image_min_pixels=64
        )
        self.assertEqual((out.width, out.height), (8, 8))

    def test_converts_to_rgb(self):
        image = Image.new("RGBA", (20, 20))  # 400 px, within budget
        out = self._plugin()._preprocess_image(
            image, image_max_pixels=10000, image_min_pixels=1
        )
        self.assertEqual(out.mode, "RGB")
        self.assertEqual((out.width, out.height), (20, 20))  # not resized


class TestPluginRegistry(unittest.TestCase):
    """get_mm_plugin / register_mm_plugin: name -> class resolution and token
    passthrough, not mere non-None."""

    def test_get_resolves_class_and_forwards_tokens(self):
        plugin = get_mm_plugin(
            "qwen2_vl", image_token=IMG, video_token=VID, audio_token=None
        )
        self.assertIsInstance(plugin, Qwen2VLPlugin)
        self.assertEqual(plugin.image_token, IMG)
        self.assertEqual(plugin.video_token, VID)

    def test_get_base_returns_base_class(self):
        self.assertIs(type(get_mm_plugin("base")), BasePlugin)

    def test_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            get_mm_plugin("no_such_plugin_xyz")

    def test_register_then_resolve_roundtrip(self):
        class _ProbePlugin(BasePlugin):
            pass

        name = "unit_test_probe_plugin"
        self.addCleanup(lambda: PLUGINS.pop(name, None))  # restore registry
        register_mm_plugin(name, _ProbePlugin)

        plugin = get_mm_plugin(name, image_token="<z>")
        self.assertIs(type(plugin), _ProbePlugin)
        self.assertEqual(plugin.image_token, "<z>")

    def test_register_duplicate_name_raises(self):
        # "base" is pre-registered; re-registering must be rejected and must
        # not overwrite the existing entry.
        with self.assertRaises(ValueError):
            register_mm_plugin("base", BasePlugin)
        self.assertIs(PLUGINS["base"], BasePlugin)


if __name__ == "__main__":
    unittest.main()
