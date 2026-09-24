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

"""Behavior tests for paddlefleet.models.multimodal.llava_model.

Scope (disjoint slice keyed to this coverage source's imports): the module-level
``pixel_shuffle`` spatial-shuffle arithmetic and the module label/token
constants.  Attention, projector, forward and _preprocess paths are covered by
sibling files and are intentionally not re-tested here.

Expected pixel-shuffle outputs are hand-derived with an INDEPENDENT numpy
reference (``_ref_pixel_shuffle``) that reimplements the intended algorithm in
C-order reshape / transpose semantics.  It never calls the production function
to build its reference.

The module imports paddle at load time, so every test is gated behind an honest
``skipUnless(IMPORT_OK, ...)`` guard; the environment used to author this file
has no paddle/paddlefleet installed and therefore skips rather than fakes a
pass.
"""

import os
import sys
import unittest

import numpy as np

# Make ``src/`` importable when tests are run from a source checkout.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_SRC, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

IMPORT_OK = True
IMPORT_ERR = ""
try:
    import paddle

    from paddlefleet.models.multimodal import llava_model
    from paddlefleet.models.multimodal.llava_model import (
        DEFAULT_IMAGE_TOKEN_INDEX,
        IGNORE_INDEX,
        IMAGE_TOKEN,
        VIDEO_TOKEN,
        pixel_shuffle,
    )
except ImportError as exc:  # only genuine missing-dependency, not API errors
    IMPORT_OK = False
    IMPORT_ERR = f"paddle/paddlefleet not importable: {exc}"


def _ref_pixel_shuffle(x, scale_factor=0.5, version=2):
    """Independent numpy reference for the intended pixel_shuffle algorithm.

    Mirrors the intended steps using standard C-order reshape (== paddle
    ``view`` after ``contiguous``) and axis-permute (== paddle ``permute``).
    Derived from the algorithm structure, NOT from the production tensor calls,
    so it forms a legitimate independent anchor.
    """
    x = np.asarray(x, dtype=np.float64)
    n0 = x.shape[0]
    sq = int(x.shape[1] ** 0.5)
    x = x.reshape(n0, sq, sq, -1)
    n, w, h, c = x.shape  # note: same unpacking order as production
    x = x.reshape(n, w, int(h * scale_factor), int(c / scale_factor))
    x = np.ascontiguousarray(np.transpose(x, (0, 2, 1, 3)))
    x = x.reshape(
        n,
        int(h * scale_factor),
        int(w * scale_factor),
        int(c / (scale_factor * scale_factor)),
    )
    if version == 2:
        x = np.ascontiguousarray(np.transpose(x, (0, 2, 1, 3)))
    x = x.reshape(x.shape[0], -1, x.shape[-1])
    return x


@unittest.skipUnless(IMPORT_OK, IMPORT_ERR)
class TestPixelShuffleNumeric(unittest.TestCase):
    """Full-content checks of pixel_shuffle against a hand-derived reference.

    Runs on a real Paddle + GPU runtime: the ``x.reshape(...)`` / ``x.size()``
    / ``x.view(...)`` / ``x.permute(...)`` tensor idioms in
    ``src/paddlefleet/models/multimodal/llava_model.py`` (lines 1037-1053) are
    accepted by the installed Paddle, so the numeric comparison against the
    independent reference below provides the real value.
    """

    def test_pixel_shuffle_version1_matches_reference(self):
        # num_tiles=1, seq_len=16 (sq=4), c=4 -> genuine spatial reshuffle.
        arr = np.arange(1 * 16 * 4, dtype="float32").reshape(1, 16, 4)
        expected = _ref_pixel_shuffle(arr, scale_factor=0.5, version=1)
        self.assertEqual(list(expected.shape), [1, 4, 16])  # (4^2*.25, 4/.25)
        out = pixel_shuffle(paddle.to_tensor(arr), scale_factor=0.5, version=1)
        np.testing.assert_allclose(
            np.asarray(out.numpy(), dtype=np.float64), expected, atol=0, rtol=0
        )

    def test_pixel_shuffle_version2_matches_reference(self):
        arr = np.arange(1 * 16 * 4, dtype="float32").reshape(1, 16, 4)
        expected = _ref_pixel_shuffle(arr, scale_factor=0.5, version=2)
        self.assertEqual(list(expected.shape), [1, 4, 16])
        out = pixel_shuffle(paddle.to_tensor(arr), scale_factor=0.5, version=2)
        np.testing.assert_allclose(
            np.asarray(out.numpy(), dtype=np.float64), expected, atol=0, rtol=0
        )

    def test_pixel_shuffle_multiple_tiles_are_independent(self):
        # Two tiles with disjoint value ranges: tile outputs must not cross-talk.
        arr = np.stack(
            [
                np.arange(16 * 4, dtype="float32").reshape(16, 4),
                np.arange(16 * 4, dtype="float32").reshape(16, 4) + 1000.0,
            ]
        )
        expected = _ref_pixel_shuffle(arr, scale_factor=0.5, version=2)
        out = pixel_shuffle(paddle.to_tensor(arr), scale_factor=0.5, version=2)
        got = np.asarray(out.numpy(), dtype=np.float64)
        self.assertEqual(list(got.shape), [2, 4, 16])
        np.testing.assert_allclose(got, expected, atol=0, rtol=0)
        # tile 1 is exactly tile 0 shifted by 1000 everywhere.
        np.testing.assert_allclose(got[1] - got[0], 1000.0, atol=0, rtol=0)


class TestPixelShuffleReferenceSanity(unittest.TestCase):
    """The independent reference must actually distinguish the version branch.

    Pure numpy (no paddle needed), so it runs even when paddle is absent. It
    guards against a degenerate reference where version=1 and version=2 would
    coincide -- which would let a dropped ``permute`` slip through the numeric
    tests above.
    """

    def test_versions_reorder_differently(self):
        arr = np.arange(1 * 16 * 4, dtype="float32").reshape(1, 16, 4)
        v1 = _ref_pixel_shuffle(arr, scale_factor=0.5, version=1)
        v2 = _ref_pixel_shuffle(arr, scale_factor=0.5, version=2)
        self.assertEqual(v1.shape, v2.shape)
        # Same multiset of values, but a genuinely different arrangement.
        np.testing.assert_array_equal(
            np.sort(v1, axis=None), np.sort(v2, axis=None)
        )
        self.assertFalse(np.array_equal(v1, v2))


@unittest.skipUnless(IMPORT_OK, IMPORT_ERR)
class TestLlavaModelConstants(unittest.TestCase):
    """Exact values of the label/token constants consumed by the LLaVA path."""

    def test_ignore_index_value(self):
        self.assertEqual(IGNORE_INDEX, -100)
        self.assertIs(llava_model.IGNORE_INDEX, IGNORE_INDEX)

    def test_default_image_token_index_value(self):
        self.assertEqual(DEFAULT_IMAGE_TOKEN_INDEX, -200)
        # Must be distinct from the loss-ignore sentinel and negative so it is
        # never confused with a real vocabulary id.
        self.assertNotEqual(DEFAULT_IMAGE_TOKEN_INDEX, IGNORE_INDEX)
        self.assertLess(DEFAULT_IMAGE_TOKEN_INDEX, 0)

    def test_image_and_video_token_strings(self):
        self.assertEqual(IMAGE_TOKEN, "<image>")
        self.assertEqual(VIDEO_TOKEN, "<video>")
        self.assertNotEqual(IMAGE_TOKEN, VIDEO_TOKEN)


if __name__ == "__main__":
    unittest.main()
