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

"""Behavior tests for paddlefleet.utils.ie_utils.

Every assertion below exercises the REAL function and compares against an
expected value derived independently by hand (never by calling the function
under test to build its own oracle).

Documented production defect
----------------------------
``ie_utils`` runs ``from ..metrics import SpanEvaluator`` at module import
time, but ``paddlefleet.metrics`` does not exist anywhere in this repository.
That broken import makes the ENTIRE module unimportable in a clean install,
including the SpanEvaluator-independent functions (map_offset, pad_image_data,
unify_prompt_name, get_relation_type_dict, uie_loss_func).

We detect the real absence up-front (before any stubbing) so the defect can be
asserted directly (see TestIeUtilsImportContract), then install a minimal
stand-in for the unrelated missing module purely so the SpanEvaluator-free
functions can be imported and exercised for real. The stub is never consulted
by any behavior assertion.
"""

import importlib.util
import sys
import types
import unittest
from io import BytesIO

import numpy as np

# Detect whether the (currently missing) metrics dependency really resolves,
# BEFORE installing any stub, so the defect can be surfaced honestly.
try:
    METRICS_AVAILABLE = (
        importlib.util.find_spec("paddlefleet.metrics") is not None
    )
except ModuleNotFoundError:
    METRICS_AVAILABLE = False

# Isolate the unrelated, genuinely-absent module so the SpanEvaluator-free
# functions can be imported. This does not touch any logic under test here.
if not METRICS_AVAILABLE and "paddlefleet.metrics" not in sys.modules:
    _metrics_stub = types.ModuleType("paddlefleet.metrics")

    class _MissingSpanEvaluator:  # placeholder for an absent dependency
        pass

    _metrics_stub.SpanEvaluator = _MissingSpanEvaluator
    sys.modules["paddlefleet.metrics"] = _metrics_stub


class TestMapOffset(unittest.TestCase):
    """map_offset returns the index of the first span containing the offset."""

    def test_returns_first_containing_span_index(self):
        from paddlefleet.utils.ie_utils import map_offset

        mapping = [[0, 2], [2, 5], [5, 8]]
        # span[0] <= offset < span[1]; boundaries belong to the later span.
        self.assertEqual(map_offset(0, mapping), 0)
        self.assertEqual(map_offset(1, mapping), 0)
        self.assertEqual(map_offset(2, mapping), 1)
        self.assertEqual(map_offset(4, mapping), 1)
        self.assertEqual(map_offset(5, mapping), 2)
        self.assertEqual(map_offset(7, mapping), 2)

    def test_offset_past_the_end_and_on_upper_bound(self):
        from paddlefleet.utils.ie_utils import map_offset

        mapping = [[0, 2], [2, 5], [5, 8]]
        # 8 is the exclusive upper bound of the last span -> no match.
        self.assertEqual(map_offset(8, mapping), -1)
        self.assertEqual(map_offset(10, mapping), -1)

    def test_first_match_wins_on_overlap(self):
        from paddlefleet.utils.ie_utils import map_offset

        # 4 lies in both spans; iteration order returns the FIRST one (0).
        overlapping = [[0, 5], [3, 8]]
        self.assertEqual(map_offset(4, overlapping), 0)
        # 6 only lies in the second span.
        self.assertEqual(map_offset(6, overlapping), 1)

    def test_empty_mapping_returns_minus_one(self):
        from paddlefleet.utils.ie_utils import map_offset

        self.assertEqual(map_offset(0, []), -1)


class TestUnifyPromptName(unittest.TestCase):
    """unify_prompt_name dedups + sorts the trailing ``[...]`` option list."""

    def test_dedup_and_sort_preserving_prefix(self):
        from paddlefleet.utils.ie_utils import unify_prompt_name

        # options ["b","a","b","a"] -> set{"a","b"} -> sorted "a,b".
        self.assertEqual(
            unify_prompt_name("Classify[b,a,b,a]"), "Classify[a,b]"
        )

    def test_already_sorted_is_unchanged(self):
        from paddlefleet.utils.ie_utils import unify_prompt_name

        self.assertEqual(
            unify_prompt_name("Classify[a,b,c]"), "Classify[a,b,c]"
        )

    def test_single_option(self):
        from paddlefleet.utils.ie_utils import unify_prompt_name

        self.assertEqual(unify_prompt_name("Type[x]"), "Type[x]")

    def test_no_trailing_brackets_returns_input(self):
        from paddlefleet.utils.ie_utils import unify_prompt_name

        self.assertEqual(unify_prompt_name("Simple prompt"), "Simple prompt")
        self.assertEqual(unify_prompt_name("a,b,c"), "a,b,c")

    def test_brackets_not_at_end_are_ignored(self):
        from paddlefleet.utils.ie_utils import unify_prompt_name

        # The regex is anchored with ``$``; brackets mid-string do not match.
        self.assertEqual(unify_prompt_name("foo[a,b] bar"), "foo[a,b] bar")


class TestGetRelationTypeDict(unittest.TestCase):
    """get_relation_type_dict groups schema names by shared relation head."""

    def test_chinese_shared_suffix_groups_values(self):
        from paddlefleet.utils.ie_utils import get_relation_type_dict

        # "出版的书" and "作者的书" share the reversed prefix "书的"; the head
        # after dropping the leading "的" is "书", so both values group there.
        relation_data = [("出版的书", "v1"), ("作者的书", "v2")]
        result = get_relation_type_dict(relation_data, schema_lang="ch")
        self.assertEqual(result, {"书": ["v1", "v2"]})

    def test_chinese_no_shared_suffix_uses_last_segment(self):
        from paddlefleet.utils.ie_utils import get_relation_type_dict

        # No shared suffix -> each falls back to rsplit("的")[-1].
        relation_data = [("作者的书", "v1"), ("作者的文章", "v2")]
        result = get_relation_type_dict(relation_data, schema_lang="ch")
        self.assertEqual(result, {"书": ["v1"], "文章": ["v2"]})

    def test_english_falls_back_to_prefix_before_of(self):
        from paddlefleet.utils.ie_utils import get_relation_type_dict

        # compare() cannot match (common prefix ends "of ", not " of"), so the
        # fallback prefix before " of " ("author") is used for both entries.
        relation_data = [
            ("author of book", "v1"),
            ("author of article", "v2"),
        ]
        result = get_relation_type_dict(relation_data, schema_lang="en")
        self.assertEqual(result, {"author": ["v1", "v2"]})

    def test_empty_input_returns_empty_dict(self):
        from paddlefleet.utils.ie_utils import get_relation_type_dict

        self.assertEqual(get_relation_type_dict([]), {})


class TestUieLossFunc(unittest.TestCase):
    """uie_loss_func is the mean of start/end binary cross-entropy losses."""

    def test_matches_independent_bce_average(self):
        import paddle

        from paddlefleet.utils.ie_utils import uie_loss_func

        start_prob = np.array([[0.2, 0.7], [0.6, 0.3]], dtype=np.float32)
        end_prob = np.array([[0.4, 0.8], [0.5, 0.1]], dtype=np.float32)
        start_ids = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        end_ids = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        # Independent reference: mean BCE per branch, then averaged over the two
        # branches. Any missing /2 or wrong reduction changes this value.
        def bce_mean(prob, label):
            per_elem = -(
                label * np.log(prob) + (1.0 - label) * np.log(1.0 - prob)
            )
            return float(np.mean(per_elem))

        expected = (
            bce_mean(start_prob, start_ids) + bce_mean(end_prob, end_ids)
        ) / 2.0

        loss = uie_loss_func(
            (paddle.to_tensor(start_prob), paddle.to_tensor(end_prob)),
            (paddle.to_tensor(start_ids), paddle.to_tensor(end_ids)),
        )
        loss_value = float(loss)
        self.assertTrue(np.isfinite(loss_value))
        np.testing.assert_allclose(loss_value, expected, rtol=1e-5, atol=1e-6)


class TestPadImageData(unittest.TestCase):
    """pad_image_data decodes, resizes, normalizes and permutes to CHW."""

    def test_empty_input_returns_zero_chw_image(self):
        from paddlefleet.utils.ie_utils import pad_image_data

        for empty in (None, b""):
            result = pad_image_data(empty)
            self.assertEqual(result.shape, (3, 224, 224))
            np.testing.assert_array_equal(result, np.zeros((3, 224, 224)))

    def test_solid_color_is_normalized_per_channel_and_permuted(self):
        from PIL import Image

        from paddlefleet.utils.ie_utils import pad_image_data

        # A uniform image stays uniform through resize, so each output channel
        # plane must equal (pixel - mean) / std for that channel (is_scale is
        # off, layout is channel-last mean/std, then swapped to CHW, no BGR).
        r, g, b = 128, 64, 200
        img = Image.new("RGB", (100, 80), color=(r, g, b))
        buf = BytesIO()
        img.save(
            buf, format="PNG"
        )  # lossless -> decoded pixels exactly (r,g,b)

        result = pad_image_data(buf.getvalue())
        self.assertEqual(result.shape, (3, 224, 224))

        mean = [123.675, 116.280, 103.530]
        std = [58.395, 57.120, 57.375]
        expected = [
            (r - mean[0]) / std[0],
            (g - mean[1]) / std[1],
            (b - mean[2]) / std[2],
        ]
        for channel, value in enumerate(expected):
            np.testing.assert_allclose(
                result[channel],
                np.full((224, 224), value),
                rtol=1e-5,
                atol=1e-4,
            )


class TestIeUtilsImportContract(unittest.TestCase):
    """Document the broken ``from ..metrics import SpanEvaluator`` dependency."""

    @unittest.expectedFailure
    def test_metrics_dependency_resolves(self):
        # PRODUCTION DEFECT (ie_utils.py:22): imports paddlefleet.metrics, which
        # does not exist in this repository, making the whole module unimportable
        # in a clean install and leaving compute_metrics untestable. Correct
        # behavior is that the dependency resolves; this expected failure records
        # the bug and will flip to an unexpected success once metrics is added.
        self.assertTrue(
            METRICS_AVAILABLE,
            "paddlefleet.metrics is missing; ie_utils import is broken",
        )


if __name__ == "__main__":
    unittest.main()
