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

"""Behavior tests for the grounding plugin data layer.

Focus: the text placeholder <-> grounding-annotation correspondence.  A
message carries one ``<ref-object>`` per referred object and one ``<bbox>``
per bounding box; ``process_messages`` must expand each placeholder into the
Qwen grounding tokens using the object / box that occupies the *same*
appearance position, consuming the ``ref`` and ``bbox`` lists strictly in
order across the whole message list.  When ``norm_bbox == "norm1000"`` each
box is additionally rescaled to the 0-1000 range relative to *its own* image,
where x-coordinates (even indices) divide by width and y-coordinates (odd
indices) divide by height, and the owning image is selected per box via
``image_id``.  The assertions below compare full expanded strings and exact
integer coordinates against independently hand-computed expectations rather
than shapes, counts, or ``assertIn`` substrings, so swapping an object,
misaligning the running box index, or confusing the width/height axis is
rejected.
"""

import unittest

from paddlefleet.datasets.template.grounding_plugin import (
    PLUGINS,
    BaseGroundingPlugin,
    get_grounding_plugin,
    register_grounding_plugin,
)

REF_START = "<|object_ref_start|>"
REF_END = "<|object_ref_end|>"
BOX_START = "<|box_start|>"
BOX_END = "<|box_end|>"


class TestNormalizeBbox(unittest.TestCase):
    """normalize_bbox: raw-pixel truncation vs norm1000 rescaling."""

    def test_none_mode_truncates_toward_zero(self):
        # norm_bbox defaults to "none": floats are truncated with int(),
        # not rounded.  1.9 -> 1 and 4.99 -> 4 distinguish truncation from
        # rounding (which would otherwise give 2 and 5).
        plugin = BaseGroundingPlugin()
        result = plugin.normalize_bbox([1.9, 2.1, 3.5, 4.99])
        self.assertEqual(result, [1, 2, 3, 4])

    def test_none_mode_ignores_image_size(self):
        # With norm_bbox == "none" an image_size must not trigger rescaling.
        plugin = BaseGroundingPlugin()
        result = plugin.normalize_bbox(
            [12.0, 34.0, 56.0, 78.0], image_size=(100, 200)
        )
        self.assertEqual(result, [12, 34, 56, 78])

    def test_norm1000_scales_x_by_width_and_y_by_height(self):
        # width (3) != height (6) so the axis assignment is observable:
        # even indices divide by width, odd indices divide by height.
        #   x1 = round(1 / 3 * 1000) = 333
        #   y1 = round(1 / 6 * 1000) = 167
        #   x2 = round(2 / 3 * 1000) = 667
        #   y2 = round(2 / 6 * 1000) = 333
        plugin = BaseGroundingPlugin(norm_bbox="norm1000")
        result = plugin.normalize_bbox([1, 1, 2, 2], image_size=(3, 6))
        self.assertEqual(result, [333, 167, 667, 333])

    def test_norm1000_without_image_size_falls_back_to_truncation(self):
        # norm1000 needs an image_size; without one the raw-pixel branch
        # runs and truncates instead of rescaling.
        plugin = BaseGroundingPlugin(norm_bbox="norm1000")
        result = plugin.normalize_bbox([1.9, 2.1, 3.5, 4.99])
        self.assertEqual(result, [1, 2, 3, 4])

    def test_unsupported_mode_raises(self):
        plugin = BaseGroundingPlugin(norm_bbox="norm500")
        with self.assertRaises(ValueError):
            plugin.normalize_bbox([1, 2, 3, 4], image_size=(100, 100))

    def test_norm1000_rejects_nonpositive_dimensions(self):
        plugin = BaseGroundingPlugin(norm_bbox="norm1000")
        with self.assertRaises(ValueError):
            plugin.normalize_bbox([1, 2, 3, 4], image_size=(0, 100))
        with self.assertRaises(ValueError):
            plugin.normalize_bbox([1, 2, 3, 4], image_size=(100, -5))


class TestFormatting(unittest.TestCase):
    """format_ref_object / format_bbox token composition."""

    def test_format_ref_object_wraps_name(self):
        plugin = BaseGroundingPlugin()
        self.assertEqual(
            plugin.format_ref_object("red bicycle"),
            f"{REF_START}red bicycle{REF_END}",
        )

    def test_format_bbox_uses_raw_pixels_by_default(self):
        plugin = BaseGroundingPlugin()
        self.assertEqual(
            plugin.format_bbox([10.7, 20.2, 30.9, 40.1]),
            f"{BOX_START}(10,20),(30,40){BOX_END}",
        )

    def test_format_bbox_threads_image_size_through_norm1000(self):
        # format_bbox must forward image_size into normalize_bbox so the
        # emitted coordinates are the rescaled ones, not raw pixels.
        #   x1 = 50 / 200 * 1000 = 250    y1 = 50 / 400 * 1000 = 125
        #   x2 = 150 / 200 * 1000 = 750   y2 = 150 / 400 * 1000 = 375
        plugin = BaseGroundingPlugin(norm_bbox="norm1000")
        self.assertEqual(
            plugin.format_bbox([50, 50, 150, 150], image_size=(200, 400)),
            f"{BOX_START}(250,125),(750,375){BOX_END}",
        )


class TestProcessMessages(unittest.TestCase):
    """process_messages: placeholder <-> object/box correspondence."""

    def test_placeholders_consumed_in_appearance_order(self):
        # Three refs and three boxes span two messages; the running ref /
        # bbox indices must advance across the message boundary so each
        # placeholder pairs with the annotation at the same position.
        plugin = BaseGroundingPlugin()
        messages = [
            {"content": "A <ref-object> next to <ref-object> in <bbox>"},
            {"content": "then <ref-object> at <bbox> and <bbox>"},
        ]
        objects = {
            "ref": ["cat", "dog", "car"],
            "bbox": [[10, 20, 30, 40], [50, 60, 70, 80], [11, 22, 33, 44]],
        }
        result = plugin.process_messages(messages, objects)
        self.assertEqual(
            result[0]["content"],
            f"A {REF_START}cat{REF_END} next to {REF_START}dog{REF_END} "
            f"in {BOX_START}(10,20),(30,40){BOX_END}",
        )
        self.assertEqual(
            result[1]["content"],
            f"then {REF_START}car{REF_END} at "
            f"{BOX_START}(50,60),(70,80){BOX_END} and "
            f"{BOX_START}(11,22),(33,44){BOX_END}",
        )

    def test_message_without_placeholders_consumes_nothing(self):
        # A plain message stays byte-identical and must not advance the ref
        # index, so the next message still starts from ref 0.
        plugin = BaseGroundingPlugin()
        messages = [
            {"content": "plain text, no markers here"},
            {"content": "see <ref-object>"},
        ]
        objects = {"ref": ["fox"], "bbox": []}
        result = plugin.process_messages(messages, objects)
        self.assertEqual(result[0]["content"], "plain text, no markers here")
        self.assertEqual(result[1]["content"], f"see {REF_START}fox{REF_END}")

    def test_norm1000_selects_owning_image_per_box(self):
        # Two boxes in one message, each owned by a different image chosen
        # via image_id; width/height are looked up by that image index.
        #   box0 uses image 1 -> (w=1000, h=2000):
        #     50/1000*1000=50, 50/2000*1000=25,
        #     150/1000*1000=150, 150/2000*1000=75
        #   box1 uses image 0 -> (w=200, h=400):
        #     100/200*1000=500, 200/400*1000=500,
        #     300/200*1000=1500, 400/400*1000=1000
        plugin = BaseGroundingPlugin(norm_bbox="norm1000")
        messages = [{"content": "obj at <bbox> and <bbox>"}]
        objects = {
            "ref": [],
            "bbox": [[50, 50, 150, 150], [100, 200, 300, 400]],
            "width": [200, 1000],
            "height": [400, 2000],
            "image_id": [1, 0],
        }
        result = plugin.process_messages(messages, objects)
        self.assertEqual(
            result[0]["content"],
            f"obj at {BOX_START}(50,25),(150,75){BOX_END} and "
            f"{BOX_START}(500,500),(1500,1000){BOX_END}",
        )


class TestRegistry(unittest.TestCase):
    """register_grounding_plugin / get_grounding_plugin."""

    def setUp(self):
        # Snapshot the shared PLUGINS registry so a test-registered plugin
        # never leaks into other tests running in the same process.
        self._original = dict(PLUGINS)
        self.addCleanup(self._restore)

    def _restore(self):
        PLUGINS.clear()
        PLUGINS.update(self._original)

    def test_get_base_returns_default_instance(self):
        plugin = get_grounding_plugin(name="base")
        self.assertIsInstance(plugin, BaseGroundingPlugin)
        self.assertEqual(plugin.norm_bbox, "none")

    def test_get_forwards_kwargs_to_constructor(self):
        # kwargs must reach the dataclass constructor and take effect, not
        # merely yield an instance.  A norm1000 plugin actually rescales,
        # proving the forwarded field is consumed.
        plugin = get_grounding_plugin(name="base", norm_bbox="norm1000")
        self.assertEqual(plugin.norm_bbox, "norm1000")
        self.assertEqual(
            plugin.normalize_bbox([50, 50, 150, 150], image_size=(200, 400)),
            [250, 125, 750, 375],
        )

    def test_register_and_get_custom_plugin(self):
        class CustomGroundingPlugin(BaseGroundingPlugin):
            pass

        register_grounding_plugin("custom_grounding", CustomGroundingPlugin)
        plugin = get_grounding_plugin(name="custom_grounding")
        self.assertIsInstance(plugin, CustomGroundingPlugin)

    def test_register_duplicate_name_raises(self):
        with self.assertRaises(ValueError):
            register_grounding_plugin("base", BaseGroundingPlugin)

    def test_get_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            get_grounding_plugin(name="does_not_exist_grounding")


if __name__ == "__main__":
    unittest.main()
