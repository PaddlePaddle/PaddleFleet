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
"""Unit tests for paddlefleet.models.gpt.gpt_embedding.

Scope and honesty notes
------------------------
These tests target the CPU-executable, self-contained pieces of the embedding
module whose expected outputs can be hand-derived without a distributed
process group:

* ``GPTEmbeddingSpec``   - the dataclass field contract.
* ``make_contiguous``    - the pure recursive tensor-contiguity helper.
* ``GPTEmbedding.get_placeholder_mask`` (the ``input_ids is not None`` branch)
  - the multimodal placeholder-mask computation and the token/feature count
    validation. That branch reads only ``self.config.image_token_id`` and
    ``self.config.video_token_id``; it never touches the embedding sublayer,
    so it is driven here as a real, unbound method call against a minimal
    ``self`` carrying only the genuinely consumed config. The ``input_ids is
    None`` branch (which calls ``self.embedding``) and the full ``forward``
    path (RoPE tables, MTP split, context-parallel scatter, sequence-parallel)
    require a constructed model and a real process group; they are NOT covered
    here and are left to single/multi-card tests.

Paddle is a hard dependency of the module under test. When it (or the import
chain it pulls in) is unavailable the whole file skips with the concrete
import error instead of reporting a false pass.
"""

import types
import unittest

import numpy as np

try:
    import paddle

    from paddlefleet.models.gpt.gpt_embedding import (
        GPTEmbedding,
        GPTEmbeddingSpec,
        make_contiguous,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # only a genuinely missing dependency, not a bug
    paddle = None
    GPTEmbedding = None
    GPTEmbeddingSpec = None
    make_contiguous = None
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    "paddle / paddlefleet.models.gpt.gpt_embedding not importable: "
    f"{_IMPORT_ERROR!r}"
)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestGPTEmbeddingSpec(unittest.TestCase):
    """The dataclass has two positional fields, neither with a default."""

    def test_both_fields_retained_in_declared_order(self):
        # Distinguishable sentinels so a swapped field assignment is caught.
        lang = object()
        rope = object()
        spec = GPTEmbeddingSpec(lang, rope)
        self.assertIs(spec.language_embedding, lang)
        self.assertIs(spec.rope_embedding, rope)

    def test_rope_embedding_accepts_none(self):
        lang = object()
        spec = GPTEmbeddingSpec(language_embedding=lang, rope_embedding=None)
        self.assertIs(spec.language_embedding, lang)
        self.assertIsNone(spec.rope_embedding)

    def test_rope_embedding_is_required_despite_optional_type(self):
        # ``rope_embedding: LayerSpec | None`` carries no default value, so
        # omitting it is a TypeError (it is Optional-typed, not defaulted).
        with self.assertRaises(TypeError):
            GPTEmbeddingSpec(object())


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestMakeContiguous(unittest.TestCase):
    """``make_contiguous`` recurses into list/tuple and no-ops scalars."""

    def test_non_container_values_pass_through_unchanged(self):
        self.assertEqual(make_contiguous(7), 7)
        self.assertIsNone(make_contiguous(None))
        sentinel = "not-a-tensor"
        self.assertIs(make_contiguous(sentinel), sentinel)

    def test_already_contiguous_tensor_returned_by_identity(self):
        t = paddle.arange(6, dtype="float32").reshape([2, 3])
        self.assertTrue(t.is_contiguous())
        out = make_contiguous(t)
        # Contract: contiguous input is returned as-is, no copy.
        self.assertIs(out, t)

    def test_non_contiguous_tensor_is_copied_contiguous_preserving_values(self):
        base = paddle.arange(6, dtype="float32").reshape([2, 3])
        view = base.transpose([1, 0])  # [3, 2], non-contiguous
        self.assertFalse(view.is_contiguous())
        out = make_contiguous(view)
        self.assertTrue(out.is_contiguous())
        self.assertIsNot(out, view)
        # Values must match the transposed layout element-for-element.
        np.testing.assert_array_equal(out.numpy(), base.numpy().T)

    def test_list_and_tuple_type_preserved_with_contiguous_elements(self):
        base = paddle.arange(6, dtype="float32").reshape([2, 3])
        view = base.transpose([1, 0])  # non-contiguous
        contig = paddle.arange(4, dtype="float32").reshape([2, 2])

        as_list = make_contiguous([view, contig, 5])
        self.assertIsInstance(as_list, list)
        self.assertTrue(as_list[0].is_contiguous())
        np.testing.assert_array_equal(as_list[0].numpy(), base.numpy().T)
        self.assertIs(as_list[1], contig)  # already contiguous -> same object
        self.assertEqual(as_list[2], 5)

        as_tuple = make_contiguous((view, contig))
        self.assertIsInstance(as_tuple, tuple)
        self.assertTrue(as_tuple[0].is_contiguous())
        np.testing.assert_array_equal(as_tuple[0].numpy(), base.numpy().T)

    def test_nested_container_recurses(self):
        base = paddle.arange(6, dtype="float32").reshape([2, 3])
        view = base.transpose([1, 0])
        out = make_contiguous([[view], (view,)])
        self.assertIsInstance(out, list)
        self.assertIsInstance(out[0], list)
        self.assertIsInstance(out[1], tuple)
        self.assertTrue(out[0][0].is_contiguous())
        self.assertTrue(out[1][0].is_contiguous())
        np.testing.assert_array_equal(out[0][0].numpy(), base.numpy().T)


def _mask_self(image_token_id=1, video_token_id=2):
    """Minimal ``self`` for the ``input_ids is not None`` branch.

    That branch of ``get_placeholder_mask`` consumes only these two config
    fields and never touches ``self.embedding``; the real, unbound method is
    invoked against this object so the genuine mask/count logic runs.
    """
    return types.SimpleNamespace(
        config=types.SimpleNamespace(
            image_token_id=image_token_id,
            video_token_id=video_token_id,
        )
    )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestGetPlaceholderMask(unittest.TestCase):
    """``get_placeholder_mask`` with an explicit ``input_ids`` tensor."""

    def _call(self, dummy, input_ids, inputs_embeds, **kw):
        return GPTEmbedding.get_placeholder_mask(
            dummy, input_ids, inputs_embeds, **kw
        )

    def test_image_mask_exact_content_and_shape(self):
        dummy = _mask_self(image_token_id=1, video_token_id=2)
        input_ids = paddle.to_tensor([[1, 1, 0, 0]], dtype="int64")
        inputs_embeds = paddle.randn([1, 4, 8])
        image_features = paddle.randn([2, 8])  # 2 tokens * hidden 8 = 16

        image_mask, video_mask = self._call(
            dummy, input_ids, inputs_embeds, image_features=image_features
        )
        # Masks are broadcast to the full [B, S, H] embedding shape.
        self.assertEqual(list(image_mask.shape), [1, 4, 8])
        self.assertEqual(list(video_mask.shape), [1, 4, 8])

        expected_image = np.zeros([1, 4, 8], dtype=bool)
        expected_image[0, 0, :] = True
        expected_image[0, 1, :] = True
        np.testing.assert_array_equal(image_mask.numpy(), expected_image)
        # No video token id (2) present -> all False.
        np.testing.assert_array_equal(
            video_mask.numpy(), np.zeros([1, 4, 8], dtype=bool)
        )

    def test_image_and_video_masks_are_not_swapped(self):
        dummy = _mask_self(image_token_id=1, video_token_id=2)
        # image at s=0; video at s=1 and s=3. A swap or wrong token id fails.
        input_ids = paddle.to_tensor([[1, 2, 0, 2]], dtype="int64")
        inputs_embeds = paddle.randn([1, 4, 8])

        image_mask, video_mask = self._call(dummy, input_ids, inputs_embeds)

        expected_image = np.zeros([1, 4, 8], dtype=bool)
        expected_image[0, 0, :] = True
        expected_video = np.zeros([1, 4, 8], dtype=bool)
        expected_video[0, 1, :] = True
        expected_video[0, 3, :] = True
        np.testing.assert_array_equal(image_mask.numpy(), expected_image)
        np.testing.assert_array_equal(video_mask.numpy(), expected_video)

    def test_matching_image_feature_count_does_not_raise(self):
        dummy = _mask_self(image_token_id=1, video_token_id=2)
        input_ids = paddle.to_tensor([[1, 1, 0, 0]], dtype="int64")
        inputs_embeds = paddle.randn([1, 4, 8])
        image_features = paddle.randn([2, 8])
        # 2 image tokens * H(8) == 16 == features.numel() -> no error.
        image_mask, _ = self._call(
            dummy, input_ids, inputs_embeds, image_features=image_features
        )
        self.assertEqual(int(image_mask.numpy().sum()), 2 * 8)

    def test_mismatched_image_feature_count_raises_valueerror(self):
        dummy = _mask_self(image_token_id=1, video_token_id=2)
        input_ids = paddle.to_tensor([[1, 1, 0, 0]], dtype="int64")
        inputs_embeds = paddle.randn([1, 4, 8])
        # 2 image tokens need numel 16, but this has 8.
        image_features = paddle.randn([1, 8])
        with self.assertRaises(ValueError) as ctx:
            self._call(
                dummy, input_ids, inputs_embeds, image_features=image_features
            )
        self.assertIn("Image features", str(ctx.exception))

    def test_mismatched_video_feature_count_raises_valueerror(self):
        dummy = _mask_self(image_token_id=1, video_token_id=2)
        input_ids = paddle.to_tensor([[2, 2, 0, 0]], dtype="int64")
        inputs_embeds = paddle.randn([1, 4, 8])
        video_features = paddle.randn([1, 8])  # need 16, have 8
        with self.assertRaises(ValueError) as ctx:
            self._call(
                dummy, input_ids, inputs_embeds, video_features=video_features
            )
        self.assertIn("Videos features", str(ctx.exception))

    def test_count_guard_uses_numel_not_shape0(self):
        # Characterizes the real contract: the guard compares
        # n_tokens * hidden vs features.numel(), NOT features.shape[0].
        # A flat [16] tensor (numel 16 == 2*8) is accepted even though its
        # shape[0] (16) differs from the token count (2).
        dummy = _mask_self(image_token_id=1, video_token_id=2)
        input_ids = paddle.to_tensor([[1, 1, 0, 0]], dtype="int64")
        inputs_embeds = paddle.randn([1, 4, 8])
        image_features = paddle.randn([16])
        image_mask, _ = self._call(
            dummy, input_ids, inputs_embeds, image_features=image_features
        )
        self.assertEqual(int(image_mask.numpy().sum()), 16)


if __name__ == "__main__":
    unittest.main()
