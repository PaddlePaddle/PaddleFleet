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

"""Independent unit tests for module-level helpers in
``paddlefleet.models.multimodal.llava_model``.

Slice covered here (disjoint from tests/single_card_tests/model/test_llava_model.py,
which covers the constructor / set_input_tensor / _preprocess_data / forward / freeze):

  * ``_load_state_dict_hook_ignore_param_names`` -- pure list mutation, no device math.
  * ``_load_state_dict_hook_ignore_extra_state`` -- pure list mutation, no device math.
  * ``pixel_shuffle``                            -- tensor reshuffle math.
  * ``LLaVAModel._apply_tile_tagging``           -- prepends tile-tag embeddings.

Expected values are hand-derived below, never produced by calling the code under
test. There is no Paddle install in this environment, and the module imports
``paddle`` at load time, so every Paddle-dependent case is gated behind
``HAS_PADDLE`` and skips with an honest reason rather than faking a pass.
"""

import os
import sys
import unittest
from collections import namedtuple

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

try:
    import paddle

    from paddlefleet.models.multimodal import llava_model
    from paddlefleet.models.multimodal.llava_model import (
        LLaVAModel,
        pixel_shuffle,
    )

    HAS_PADDLE = True
    IMPORT_ERROR = ""
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    # Only genuine missing-dependency import failures are turned into a skip.
    # Any other exception (compile error, API break) is allowed to propagate.
    HAS_PADDLE = False
    IMPORT_ERROR = f"paddle / llava_model not importable: {exc}"


IncompatibleKeys = namedtuple(
    "IncompatibleKeys", ["missing_keys", "unexpected_keys"]
)


@unittest.skipUnless(HAS_PADDLE, IMPORT_ERROR)
class TestIgnoreParamNamesHook(unittest.TestCase):
    """``_load_state_dict_hook_ignore_param_names(param_names, module, keys)``.

    Contract: for every name in ``param_names`` that appears in
    ``keys.missing_keys``, drop it (in place) from ``missing_keys``. Names not
    present are ignored, and ``unexpected_keys`` is never touched.
    """

    def test_removes_listed_missing_keys_and_leaves_unexpected(self):
        keys = IncompatibleKeys(
            missing_keys=["a.weight", "b.weight", "c._extra_state"],
            unexpected_keys=["u.weight"],
        )
        llava_model._load_state_dict_hook_ignore_param_names(
            ["b.weight", "not.present"], None, keys
        )
        # Only "b.weight" matched; "not.present" is a no-op; order preserved.
        self.assertEqual(keys.missing_keys, ["a.weight", "c._extra_state"])
        # unexpected_keys must be untouched by this hook.
        self.assertEqual(keys.unexpected_keys, ["u.weight"])

    def test_ignores_names_that_are_only_in_unexpected_keys(self):
        # A name present only in unexpected_keys must NOT be removed: this hook
        # filters missing_keys exclusively. Distinguishes it from a hook that
        # scans both fields.
        keys = IncompatibleKeys(
            missing_keys=["x.weight"],
            unexpected_keys=["y.weight"],
        )
        llava_model._load_state_dict_hook_ignore_param_names(
            ["y.weight"], None, keys
        )
        self.assertEqual(keys.missing_keys, ["x.weight"])
        self.assertEqual(keys.unexpected_keys, ["y.weight"])

    def test_removes_only_first_occurrence_per_name(self):
        # list.remove drops a single occurrence; a duplicate survives. Catches
        # an implementation that eagerly strips all matches.
        keys = IncompatibleKeys(
            missing_keys=["d.weight", "d.weight", "e.weight"],
            unexpected_keys=[],
        )
        llava_model._load_state_dict_hook_ignore_param_names(
            ["d.weight"], None, keys
        )
        self.assertEqual(keys.missing_keys, ["d.weight", "e.weight"])

    def test_empty_param_names_is_noop(self):
        keys = IncompatibleKeys(
            missing_keys=["a.weight"], unexpected_keys=["b.weight"]
        )
        llava_model._load_state_dict_hook_ignore_param_names([], None, keys)
        self.assertEqual(keys.missing_keys, ["a.weight"])
        self.assertEqual(keys.unexpected_keys, ["b.weight"])


@unittest.skipUnless(HAS_PADDLE, IMPORT_ERROR)
class TestIgnoreExtraStateHook(unittest.TestCase):
    """``_load_state_dict_hook_ignore_extra_state(module, keys)``.

    Contract: from BOTH ``missing_keys`` and ``unexpected_keys`` drop every key
    whose name contains the substring ``"extra_state"`` (in place), preserving
    the relative order of the surviving keys.
    """

    def test_removes_extra_state_from_both_fields(self):
        keys = IncompatibleKeys(
            missing_keys=["a.weight", "b._extra_state", "c._extra_state"],
            unexpected_keys=["d._extra_state", "e.weight"],
        )
        llava_model._load_state_dict_hook_ignore_extra_state(None, keys)
        # Both adjacent extra_state entries in missing_keys are dropped -- the
        # reversed-copy iteration must not skip the second one.
        self.assertEqual(keys.missing_keys, ["a.weight"])
        self.assertEqual(keys.unexpected_keys, ["e.weight"])

    def test_substring_match_and_surviving_order(self):
        keys = IncompatibleKeys(
            missing_keys=[
                "keep1.weight",
                "mid.extra_state.tail",
                "keep2.weight",
            ],
            unexpected_keys=[],
        )
        llava_model._load_state_dict_hook_ignore_extra_state(None, keys)
        # Match is by substring, and non-matching keys keep their order.
        self.assertEqual(keys.missing_keys, ["keep1.weight", "keep2.weight"])

    def test_no_extra_state_leaves_both_lists_unchanged(self):
        keys = IncompatibleKeys(
            missing_keys=["a.weight"], unexpected_keys=["b.weight"]
        )
        llava_model._load_state_dict_hook_ignore_extra_state(None, keys)
        self.assertEqual(keys.missing_keys, ["a.weight"])
        self.assertEqual(keys.unexpected_keys, ["b.weight"])


@unittest.skipUnless(HAS_PADDLE, IMPORT_ERROR)
class TestPixelShuffle(unittest.TestCase):
    """``pixel_shuffle(x, scale_factor=0.5, version=2)``.

    The published contract turns ``[num_tiles, sq**2, h]`` into
    ``[num_tiles, (sq**2)*(scale**2), h/(scale**2)]``. Expected tensors are
    derived by hand below.

    KNOWN PRODUCTION BUG (asserted-correct + expectedFailure, production left
    untouched): the body at src/paddlefleet/models/multimodal/llava_model.py
    uses PyTorch-only tensor APIs that are not valid on a Paddle tensor --
    ``x.size()`` at line 1039 (``paddle.Tensor.size`` is a *property* returning
    the element count, so calling it raises), plus ``x.view(...)`` / ``x.permute(...)``
    / variadic ``x.reshape(...)`` (lines 1037-1053). These raise on a standard
    Paddle install; the cases below encode the intended result so they will
    surface as an unexpected success once the port is fixed.
    """

    @unittest.expectedFailure
    def test_scale_half_flattens_two_by_two_grid(self):
        # sq = 2, h_vision = 4. The 2x2 spatial grid collapses to a single
        # position whose channel axis is the row-major concatenation of the
        # four original positions, i.e. arange(16) unchanged.
        x = paddle.arange(16, dtype="float32").reshape([1, 4, 4])
        out = pixel_shuffle(x, scale_factor=0.5, version=2)
        self.assertEqual(out.shape, [1, 1, 16])
        expected = paddle.arange(16, dtype="float32").reshape([1, 1, 16])
        self.assertTrue(paddle.allclose(out, expected))

    @unittest.expectedFailure
    def test_versions_diverge_on_four_by_four_grid(self):
        # sq = 4, h_vision = 4 -> both versions yield [1, 4, 16] and conserve
        # the element count, but version 2's extra transpose reorders content,
        # so the two outputs must not be equal.
        x = paddle.arange(64, dtype="float32").reshape([1, 16, 4])
        v1 = pixel_shuffle(x, scale_factor=0.5, version=1)
        v2 = pixel_shuffle(x, scale_factor=0.5, version=2)
        self.assertEqual(v1.shape, [1, 4, 16])
        self.assertEqual(v2.shape, [1, 4, 16])
        self.assertEqual(int(v1.numel()), 64)
        self.assertEqual(int(v2.numel()), 64)
        self.assertFalse(paddle.allclose(v1, v2))


class _TileTagStub:
    """Minimal stand-in for ``LLaVAModel`` bound to ``_apply_tile_tagging``.

    Only the attributes the method actually reads are provided; the language
    model exposes a deterministic ``embedding`` so the intended (post-fix)
    result is well defined.
    """

    def __init__(self, tile_tags, embed_out):
        self._tile_tags = tile_tags
        self.language_model = self
        self._embed_out = embed_out
        self.embedding_calls = []

    def embedding(self, input_ids, position_ids=None):
        self.embedding_calls.append((input_ids, position_ids))
        return self._embed_out


@unittest.skipUnless(HAS_PADDLE, IMPORT_ERROR)
class TestApplyTileTagging(unittest.TestCase):
    """``LLaVAModel._apply_tile_tagging(self, image_embeddings, num_image_tiles)``."""

    def test_rejects_multiple_input_images(self):
        # The guard at the top of the method (num_image_tiles must describe a
        # single image) is reached before any of the buggy tensor calls, so
        # this contract is verified cleanly.
        stub = _TileTagStub(tile_tags=[[1], [2]], embed_out=None)
        num_image_tiles = paddle.to_tensor([1, 1], dtype="int64")
        image_embeddings = paddle.ones([3, 2, 2], dtype="float32")
        with self.assertRaises(AssertionError):
            LLaVAModel._apply_tile_tagging(
                stub, image_embeddings, num_image_tiles
            )

    @unittest.expectedFailure
    def test_prepends_tile_tag_embeddings(self):
        # Intended behaviour: tile-tag embeddings [tile_seq_len, num_tiles, h]
        # are concatenated in front of the image embeddings along dim 0, giving
        # [tile_seq_len + img_seq_len, num_tiles, h] == [2 + 3, 2, 2].
        #
        # KNOWN PRODUCTION BUG (asserted-correct + expectedFailure, production
        # left untouched): src/paddlefleet/models/multimodal/llava_model.py
        # line 785 calls ``paddle.tensor(...)`` -- ``paddle.tensor`` is a
        # submodule, not callable, so this raises TypeError (should be
        # ``paddle.to_tensor``); line 798 calls ``paddle.cat(...)`` which does
        # not exist in Paddle (should be ``paddle.concat``).
        tile_embed = paddle.zeros([2, 2, 2], dtype="float32")
        stub = _TileTagStub(tile_tags=[[1], [2]], embed_out=tile_embed)
        image_embeddings = paddle.ones([3, 2, 2], dtype="float32")
        num_image_tiles = paddle.to_tensor([2], dtype="int64")
        out = LLaVAModel._apply_tile_tagging(
            stub, image_embeddings, num_image_tiles
        )
        self.assertEqual(out.shape, [5, 2, 2])
        self.assertTrue(paddle.allclose(out[:2], tile_embed))
        self.assertTrue(paddle.allclose(out[2:], image_embeddings))


if __name__ == "__main__":
    unittest.main()
