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
"""Unit tests for ``paddlefleet.models.multimodal.llava_model``.

Slice chosen to stay DISJOINT from the sibling files (``test_llava_model``,
``test_llava_model_2/3/5``) which already saturate the two ``_load_state_dict``
hooks, ``set_input_tensor``, ``freeze``, ``shared_embedding_or_output_weight``
and the ``_apply_tile_tagging`` helper.  Keyed to this coverage source's
imports (``LLaVAModel`` and ``pixel_shuffle``) this file instead exercises the
``LLaVAModel.__init__`` control flow that none of the siblings touch:

* the sequence-/context-parallel guard that raises ``AssertionError`` early;
* the vision-model-type dispatch (unsupported -> ``ValueError``, the ``clip``
  collaborator call, the ``siglip`` drop-class-token assertion);
* the ``pixel_shuffle`` flag scaling ``vision_projection_input_size`` (x4);
* the ``hf://`` language-model branch, which contains a real production bug
  (``build_hf_model`` is called twice, the first result discarded, with
  inconsistent arity -- see ``TestLLaVAModelHFLanguageModel``);
* ``pixel_shuffle`` numeric content vs. an independent NumPy reference.

Honest environment note: paddle is NOT installed where this file was authored,
so every test is guarded by ``skipUnless``.  The guards report the real import
error rather than silently passing; the assertions below are written to be
meaningful when the file is run on a GPU host that has paddle.  The
constructor tests mock only the heavy *collaborators* (sub-models, projector,
logging, ``get_num_image_embeddings``) and drive the real ``LLaVAModel.__init__``
branch logic, observing raised exceptions / collaborator call arguments rather
than reading back injected attributes.
"""

import contextlib
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

_MOD = "paddlefleet.models.multimodal.llava_model"
_HF_MOD = "paddlefleet.models.huggingface.module"

try:
    import paddle

    from paddlefleet.models.multimodal import llava_model as llava_mod
    from paddlefleet.models.multimodal.llava_model import (
        LLaVAModel,
        pixel_shuffle,
    )

    _AVAILABLE = True
    _SKIP = ""
except ImportError as exc:  # pragma: no cover - depends on environment
    llava_mod = None
    LLaVAModel = None
    pixel_shuffle = None
    _AVAILABLE = False
    _SKIP = f"paddle/paddlefleet not importable: {exc!r}"


def _lang_config(**overrides):
    """A stand-in language ``TransformerConfig`` with the fields __init__ reads.

    Only the attributes the constructor actually consumes are pinned; the mock
    is a *collaborator* here, the branch decisions under test are real.
    """
    cfg = MagicMock(name="lang_config")
    cfg.hidden_size = 64
    cfg.params_dtype = "float32"
    cfg.sequence_parallel = False
    cfg.tp_comm_overlap = False
    cfg.context_parallel_size = 1
    cfg.tensor_model_parallel_size = 1
    cfg.pipeline_model_parallel_size = 1
    cfg.language_model_type = ""
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _vision_config(**overrides):
    cfg = MagicMock(name="vision_config")
    cfg.hidden_size = 64
    cfg.params_dtype = "float32"
    cfg.vision_model_type = "clip"
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


class _LLaVAConstructor:
    """Runs the real ``LLaVAModel.__init__`` with heavy collaborators mocked.

    Records the collaborator mocks so tests can assert on the dispatch, and
    never patches ``LLaVAModel.__init__`` itself.
    """

    def __init__(self, stack, num_img_embeddings=576):
        def _p(target, **kwargs):
            return stack.enter_context(patch(target, **kwargs))

        _p(f"{_MOD}.has_config_logger_enabled", return_value=False)
        _p(f"{_MOD}.log_single_rank")
        _p(
            f"{_MOD}.get_num_image_embeddings",
            return_value=num_img_embeddings,
        )
        self.gpt = _p(f"{_MOD}.GPTModel")
        self.clip = _p(f"{_MOD}.CLIPViTModel")
        self.radio = _p(f"{_MOD}.RADIOViTModel")
        self.projector = _p(f"{_MOD}.MultimodalProjector")
        self.build_hf = _p(f"{_HF_MOD}.build_hf_model")

    def build(self, lang_config=None, vision_config=None, **overrides):
        params = {
            "language_transformer_config": lang_config or _lang_config(),
            "language_transformer_layer_spec": MagicMock(),
            "language_vocab_size": 1000,
            "language_max_sequence_length": 512,
            "vision_transformer_config": vision_config or _vision_config(),
            "vision_transformer_layer_spec": MagicMock(),
            "drop_vision_class_token": False,
            "vision_projection_config": MagicMock(),
            "vision_projection_layer_spec": MagicMock(),
            "vision_projection_type": "mlp",
            "pg_collection": MagicMock(),
        }
        params.update(overrides)
        return LLaVAModel(**params)


def _ref_pixel_shuffle(x, scale_factor, version):
    """Independent NumPy re-implementation of the intended transform.

    Mirrors the documented reshape/permute sequence with ``numpy`` primitives;
    it does not call the production function, so a data-movement or version
    bug in production would diverge from this reference.
    """
    x = np.asarray(x, dtype=np.float64)
    n0, length = x.shape[0], x.shape[1]
    sq = int(round(length**0.5))
    x = x.reshape(n0, sq, sq, -1)
    n, w, h, c = x.shape
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
    return x.reshape(x.shape[0], -1, x.shape[-1])


@unittest.skipUnless(_AVAILABLE, _SKIP)
class TestLLaVAModelParallelGuards(unittest.TestCase):
    """The early guard in __init__ that forbids SP / CP for llava."""

    def test_sequence_parallel_raises_assertion(self):
        with contextlib.ExitStack() as stack:
            ctor = _LLaVAConstructor(stack)
            lang = _lang_config(sequence_parallel=True, context_parallel_size=1)
            with self.assertRaises(AssertionError) as caught:
                ctor.build(
                    lang_config=lang, add_encoder=False, add_decoder=False
                )
            self.assertIn("sequence_parallel", str(caught.exception))
            # Guard must fire before any sub-model is constructed.
            ctor.gpt.assert_not_called()
            ctor.clip.assert_not_called()

    def test_context_parallel_raises_assertion(self):
        with contextlib.ExitStack() as stack:
            ctor = _LLaVAConstructor(stack)
            lang = _lang_config(
                sequence_parallel=False, context_parallel_size=2
            )
            with self.assertRaises(AssertionError):
                ctor.build(
                    lang_config=lang, add_encoder=False, add_decoder=False
                )
            ctor.gpt.assert_not_called()

    def test_no_parallel_passes_and_stores_state(self):
        with contextlib.ExitStack() as stack:
            ctor = _LLaVAConstructor(stack, num_img_embeddings=576)
            model = ctor.build(add_encoder=False, add_decoder=False)
            # State computed by the real constructor (not injected).
            self.assertFalse(model.sequence_parallel_lm)
            self.assertEqual(model.context_parallel_lm, 1)
            self.assertIsNone(model.cp_group)
            self.assertFalse(model.add_encoder)
            self.assertFalse(model.add_decoder)
            self.assertIsNone(model.language_model)
            self.assertIsNone(model.vision_model)
            self.assertEqual(model.img_seq_len, 576)
            self.assertFalse(model._pixel_shuffle)
            self.assertEqual(
                model.image_token_index, llava_mod.DEFAULT_IMAGE_TOKEN_INDEX
            )
            # No encoder -> no vision projection / vision sub-model built.
            ctor.clip.assert_not_called()
            ctor.projector.assert_not_called()


@unittest.skipUnless(_AVAILABLE, _SKIP)
class TestLLaVAModelVisionTypeDispatch(unittest.TestCase):
    """The vision_model_type branch selection inside __init__."""

    def test_unsupported_type_raises_value_error(self):
        with contextlib.ExitStack() as stack:
            ctor = _LLaVAConstructor(stack)
            vision = _vision_config(vision_model_type="totally_unsupported")
            with self.assertRaises(ValueError) as caught:
                ctor.build(
                    vision_config=vision, add_encoder=True, add_decoder=False
                )
            self.assertIn("totally_unsupported", str(caught.exception))
            ctor.clip.assert_not_called()
            ctor.radio.assert_not_called()

    def test_clip_type_builds_clip_with_class_token(self):
        with contextlib.ExitStack() as stack:
            ctor = _LLaVAConstructor(stack)
            vision = _vision_config(vision_model_type="clip")
            ctor.build(
                vision_config=vision,
                add_encoder=True,
                add_decoder=False,
                drop_vision_class_token=False,
                img_h=336,
                img_w=336,
                patch_dim=14,
            )
            ctor.clip.assert_called_once()
            _, kwargs = ctor.clip.call_args
            # clip keeps the class token: len 1, add_class_token True.
            self.assertEqual(kwargs["class_token_len"], 1)
            self.assertTrue(kwargs["add_class_token"])
            self.assertEqual(kwargs["model_subtype"], "clip")
            self.assertEqual(kwargs["img_h"], 336)
            self.assertEqual(kwargs["img_w"], 336)
            self.assertEqual(kwargs["patch_dim"], 14)
            ctor.radio.assert_not_called()

    def test_siglip_with_drop_class_token_raises(self):
        with contextlib.ExitStack() as stack:
            ctor = _LLaVAConstructor(stack)
            vision = _vision_config(vision_model_type="siglip")
            with self.assertRaises(AssertionError) as caught:
                ctor.build(
                    vision_config=vision,
                    add_encoder=True,
                    add_decoder=False,
                    drop_vision_class_token=True,
                )
            self.assertIn("Siglip", str(caught.exception))
            # Assertion must fire before the sub-model is built.
            ctor.clip.assert_not_called()


@unittest.skipUnless(_AVAILABLE, _SKIP)
class TestLLaVAModelProjectionInputSize(unittest.TestCase):
    """pixel_shuffle flag scales vision_projection_input_size by 4."""

    def _projector_input_size(self, pixel_shuffle_flag):
        with contextlib.ExitStack() as stack:
            ctor = _LLaVAConstructor(stack)
            vision = _vision_config(vision_model_type="clip", hidden_size=64)
            ctor.build(
                vision_config=vision,
                add_encoder=True,
                add_decoder=False,
                pixel_shuffle=pixel_shuffle_flag,
            )
            ctor.projector.assert_called_once()
            args, _ = ctor.projector.call_args
            # 4th positional arg is vision_projection_input_size.
            return args[3]

    def test_pixel_shuffle_off_uses_plain_hidden_size(self):
        self.assertEqual(self._projector_input_size(False), 64)

    def test_pixel_shuffle_on_quadruples_hidden_size(self):
        # Single-switch delta: 64 -> 64 * 4.
        self.assertEqual(self._projector_input_size(True), 256)


@unittest.skipUnless(_AVAILABLE, _SKIP)
class TestLLaVAModelHFLanguageModel(unittest.TestCase):
    """The ``hf://`` language-model branch of __init__."""

    @unittest.expectedFailure
    def test_build_hf_model_called_exactly_once(self):
        """REAL BUG (llava_model.py:188-194): in the ``hf://`` branch
        ``build_hf_model`` is invoked twice -- first with
        ``(config, language_model_type)`` and the result is immediately
        overwritten by a second ``build_hf_model(config)`` call with a
        different arity.  The first call is dead work and the two arities are
        inconsistent.  The correct behaviour is a single construction call, so
        this test asserts ``call_count == 1`` and is marked expectedFailure to
        document the defect without editing production code.
        """
        with contextlib.ExitStack() as stack:
            ctor = _LLaVAConstructor(stack)
            lang = _lang_config(language_model_type="hf://dummy-llm")
            sentinel = object()
            ctor.build_hf.return_value = sentinel
            model = ctor.build(
                lang_config=lang, add_encoder=False, add_decoder=True
            )
            self.assertIs(model.language_model, sentinel)
            self.assertEqual(ctor.build_hf.call_count, 1)


@unittest.skipUnless(_AVAILABLE, _SKIP)
class TestPixelShuffleNumeric(unittest.TestCase):
    """pixel_shuffle content vs. an independent NumPy reference."""

    def _paddle_out(self, base, version):
        x = paddle.to_tensor(base, dtype="float32")
        out = pixel_shuffle(x, scale_factor=0.5, version=version)
        return np.asarray(out.numpy())

    def test_version2_matches_reference(self):
        base = np.arange(2 * 16 * 4, dtype=np.float32).reshape(2, 16, 4)
        got = self._paddle_out(base, version=2)
        ref = _ref_pixel_shuffle(base, 0.5, 2)
        self.assertEqual(list(got.shape), [2, 4, 16])
        np.testing.assert_allclose(got, ref, rtol=0, atol=0)

    def test_version1_matches_reference_and_differs_from_version2(self):
        base = np.arange(2 * 16 * 4, dtype=np.float32).reshape(2, 16, 4)
        got_v1 = self._paddle_out(base, version=1)
        ref_v1 = _ref_pixel_shuffle(base, 0.5, 1)
        np.testing.assert_allclose(got_v1, ref_v1, rtol=0, atol=0)
        # The ``version`` argument must actually change the layout for this
        # non-degenerate input (guards against it being ignored).
        self.assertFalse(
            np.array_equal(
                _ref_pixel_shuffle(base, 0.5, 1),
                _ref_pixel_shuffle(base, 0.5, 2),
            )
        )
        self.assertFalse(np.array_equal(got_v1, self._paddle_out(base, 2)))

    def test_shuffle_is_a_reordering_not_a_plain_reshape(self):
        base = np.arange(2 * 16 * 4, dtype=np.float32).reshape(2, 16, 4)
        got = self._paddle_out(base, version=2)
        # Value-preserving permutation: same multiset of elements.
        np.testing.assert_array_equal(
            np.sort(got.reshape(-1)),
            np.arange(2 * 16 * 4, dtype=np.float32),
        )
        # But NOT the trivial row-major reshape (real spatial shuffle happened).
        self.assertFalse(
            np.array_equal(got, base.reshape(2, 4, 16)),
        )


if __name__ == "__main__":
    unittest.main()
