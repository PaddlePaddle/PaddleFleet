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

"""CPU-only unit tests for paddlefleet.models.multimodal.llava_model.

Base file of the ``test_llava_model*`` family. It covers the core public
surface that carries logic on its own, keyed to the symbols the coverage
source imports:

* module constants (``IGNORE_INDEX``, ``DEFAULT_IMAGE_TOKEN_INDEX``,
  ``IMAGE_TOKEN``, ``VIDEO_TOKEN``);
* the two checkpoint-loading hooks
  ``_load_state_dict_hook_ignore_param_names`` /
  ``_load_state_dict_hook_ignore_extra_state`` (pure list surgery);
* ``LLaVAModel`` orchestration methods ``set_input_tensor`` (branch routing),
  ``freeze`` (per-module selectivity) and ``shared_embedding_or_output_weight``
  (delegation);
* ``pixel_shuffle`` numeric behaviour, compared to an independent numpy
  reference.

The module under test imports Paddle at load time. When Paddle / paddlefleet
is not importable the whole file is skipped with an honest reason rather than
reporting a pass. Sibling files (``test_llava_model_2..8``) own the heavier
paths (construction, ``_preprocess_data``, token-parallel packing, tile
tagging, ``forward``); this base file stays disjoint from them.
"""

import os
import sys
import unittest
from collections import namedtuple

try:
    import numpy as np
except ImportError:  # numpy genuinely absent
    np = None

_REPO_SRC = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
    "src",
)
if os.path.isdir(_REPO_SRC) and _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

_IMPORT_ERROR = None
try:
    import paddle

    from paddlefleet.models.multimodal.llava_model import (
        DEFAULT_IMAGE_TOKEN_INDEX,
        IGNORE_INDEX,
        IMAGE_TOKEN,
        VIDEO_TOKEN,
        LLaVAModel,
        _load_state_dict_hook_ignore_extra_state,
        _load_state_dict_hook_ignore_param_names,
        pixel_shuffle,
    )
except (ImportError, ModuleNotFoundError) as exc:  # honest capability probe
    # Only genuine missing-dependency import failures are treated as skip.
    # Compilation / API errors raise other exception types and surface as
    # real failures instead of being swallowed here.
    _IMPORT_ERROR = exc

_AVAILABLE = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle / paddlefleet not importable: {_IMPORT_ERROR!r}"
    if _IMPORT_ERROR is not None
    else ""
)
_NUMERIC_OK = _AVAILABLE and np is not None
_NUMERIC_SKIP = _SKIP_REASON or "numpy not importable"
IncompatibleKeys = namedtuple(
    "IncompatibleKeys", ["missing_keys", "unexpected_keys"]
)


class _Recorder:
    """Stub sub-module recording the tensor handed to set_input_tensor."""

    def __init__(self):
        self.calls = []

    def set_input_tensor(self, tensor):
        self.calls.append(tensor)


class _ParamStub:
    """Minimal parameter exposing the mutable stop_gradient flag."""

    def __init__(self):
        self.stop_gradient = False


class _ModuleStub:
    """Stub module returning a fixed parameter list from .parameters()."""

    def __init__(self, params):
        self._params = list(params)

    def parameters(self):
        return list(self._params)


class _LangModelStub:
    """Stub language model delegating shared_embedding_or_output_weight."""

    def __init__(self, weight):
        self._weight = weight

    def shared_embedding_or_output_weight(self):
        return self._weight


def _pixel_shuffle_reference(x, scale_factor, version):
    """Independent numpy re-derivation of the intended InternVL shuffle.

    Mirrors the documented op sequence using numpy reshape/transpose (which,
    on contiguous row-major arrays, matches torch view/permute+contiguous).
    Does not call the production function, so it can anchor the expected
    output on its own.
    """
    n0 = x.shape[0]
    side = int(x.shape[1] ** 0.5)
    x = x.reshape(n0, side, side, -1)
    n, w, h, c = x.shape
    x = x.reshape(n, w, int(h * scale_factor), int(c / scale_factor))
    x = np.transpose(x, (0, 2, 1, 3))
    x = x.reshape(
        n,
        int(h * scale_factor),
        int(w * scale_factor),
        int(c / (scale_factor * scale_factor)),
    )
    if version == 2:
        x = np.transpose(x, (0, 2, 1, 3))
    return x.reshape(x.shape[0], -1, x.shape[-1])


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestLLaVAConstants(unittest.TestCase):
    """Module constants form the label/image-token contract."""

    def test_ignore_index(self):
        self.assertEqual(IGNORE_INDEX, -100)

    def test_default_image_token_index(self):
        self.assertEqual(DEFAULT_IMAGE_TOKEN_INDEX, -200)

    def test_image_token_literal(self):
        self.assertEqual(IMAGE_TOKEN, "<image>")

    def test_video_token_literal(self):
        self.assertEqual(VIDEO_TOKEN, "<video>")


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestIgnoreParamNamesHook(unittest.TestCase):
    """_load_state_dict_hook_ignore_param_names drops only listed keys."""

    def test_removes_only_the_named_key(self):
        keys = IncompatibleKeys(
            missing_keys=["a.weight", "b.weight", "c.weight"],
            unexpected_keys=["u.weight"],
        )
        _load_state_dict_hook_ignore_param_names(["b.weight"], None, keys)
        # Exact remaining list and order: only the named key is gone.
        self.assertEqual(keys.missing_keys, ["a.weight", "c.weight"])
        # unexpected_keys is never touched by this hook.
        self.assertEqual(keys.unexpected_keys, ["u.weight"])

    def test_removes_each_of_several_named_keys(self):
        keys = IncompatibleKeys(
            missing_keys=["a.weight", "b.weight", "c.weight"],
            unexpected_keys=[],
        )
        _load_state_dict_hook_ignore_param_names(
            ["a.weight", "c.weight"], None, keys
        )
        self.assertEqual(keys.missing_keys, ["b.weight"])

    def test_no_match_leaves_list_intact(self):
        keys = IncompatibleKeys(
            missing_keys=["a.weight", "b.weight"],
            unexpected_keys=[],
        )
        _load_state_dict_hook_ignore_param_names(["z.weight"], None, keys)
        self.assertEqual(keys.missing_keys, ["a.weight", "b.weight"])


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestIgnoreExtraStateHook(unittest.TestCase):
    """_load_state_dict_hook_ignore_extra_state strips extra_state keys."""

    def test_removes_all_extra_state_from_both_lists(self):
        keys = IncompatibleKeys(
            missing_keys=["l0._extra_state", "l0.weight", "l1._extra_state"],
            unexpected_keys=["x._extra_state", "y.bias"],
        )
        _load_state_dict_hook_ignore_extra_state(None, keys)
        # Every key containing "extra_state" is removed from BOTH fields;
        # non-extra_state keys survive in original order.
        self.assertEqual(keys.missing_keys, ["l0.weight"])
        self.assertEqual(keys.unexpected_keys, ["y.bias"])

    def test_substring_match_removes_embedded_extra_state(self):
        keys = IncompatibleKeys(
            missing_keys=["block.extra_state_buf", "block.weight"],
            unexpected_keys=[],
        )
        _load_state_dict_hook_ignore_extra_state(None, keys)
        self.assertEqual(keys.missing_keys, ["block.weight"])

    def test_no_extra_state_leaves_lists_intact(self):
        keys = IncompatibleKeys(
            missing_keys=["a.weight"],
            unexpected_keys=["b.bias"],
        )
        _load_state_dict_hook_ignore_extra_state(None, keys)
        self.assertEqual(keys.missing_keys, ["a.weight"])
        self.assertEqual(keys.unexpected_keys, ["b.bias"])


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestSetInputTensorRouting(unittest.TestCase):
    """set_input_tensor routes the single input to the correct sub-module."""

    def _model(self):
        return LLaVAModel.__new__(LLaVAModel)

    def test_encoder_and_decoder_routes_to_vision(self):
        model = self._model()
        model.add_encoder = True
        model.add_decoder = True
        model.pre_process = True
        model.vision_model = _Recorder()
        model.language_model = _Recorder()
        tensor = object()
        model.set_input_tensor([tensor])
        self.assertEqual(len(model.vision_model.calls), 1)
        self.assertIs(model.vision_model.calls[0], tensor)
        self.assertEqual(model.language_model.calls, [])

    def test_encoder_only_routes_to_vision(self):
        model = self._model()
        model.add_encoder = True
        model.add_decoder = False
        model.pre_process = True
        model.vision_model = _Recorder()
        tensor = object()
        model.set_input_tensor([tensor])
        self.assertIs(model.vision_model.calls[0], tensor)

    def test_no_encoder_preprocess_sets_encoder_hidden_state(self):
        model = self._model()
        model.add_encoder = False
        model.add_decoder = True
        model.pre_process = True
        model.encoder_hidden_state = None
        # Bare (non-list) input must be wrapped, then stored verbatim.
        tensor = object()
        model.set_input_tensor(tensor)
        self.assertIs(model.encoder_hidden_state, tensor)

    def test_no_encoder_no_preprocess_routes_to_language(self):
        model = self._model()
        model.add_encoder = False
        model.add_decoder = True
        model.pre_process = False
        model.language_model = _Recorder()
        tensor = object()
        model.set_input_tensor([tensor])
        self.assertIs(model.language_model.calls[0], tensor)

    def test_multiple_inputs_rejected(self):
        model = self._model()
        model.add_encoder = True
        model.add_decoder = True
        model.pre_process = True
        model.vision_model = _Recorder()
        with self.assertRaises(AssertionError):
            model.set_input_tensor([object(), object()])


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestFreezeSelectivity(unittest.TestCase):
    """freeze() sets stop_gradient only on the selected modules."""

    def _model(self, lang, vision, proj=None):
        model = LLaVAModel.__new__(LLaVAModel)
        model.language_model = lang
        model.vision_model = vision
        if proj is not None:
            model.vision_projection = proj
        return model

    def test_freeze_language_only(self):
        p_l, p_v, p_p = _ParamStub(), _ParamStub(), _ParamStub()
        model = self._model(
            _ModuleStub([p_l]), _ModuleStub([p_v]), _ModuleStub([p_p])
        )
        model.freeze(True, False, False)
        self.assertTrue(p_l.stop_gradient)
        self.assertFalse(p_v.stop_gradient)
        self.assertFalse(p_p.stop_gradient)

    def test_freeze_vision_and_projection_only(self):
        p_l, p_v, p_p = _ParamStub(), _ParamStub(), _ParamStub()
        model = self._model(
            _ModuleStub([p_l]), _ModuleStub([p_v]), _ModuleStub([p_p])
        )
        model.freeze(False, True, True)
        self.assertFalse(p_l.stop_gradient)
        self.assertTrue(p_v.stop_gradient)
        self.assertTrue(p_p.stop_gradient)

    def test_freeze_skips_none_modules(self):
        p_v = _ParamStub()
        model = self._model(None, _ModuleStub([p_v]))
        # language_model is None even though requested frozen -> skipped safely.
        model.freeze(True, True, False)
        self.assertTrue(p_v.stop_gradient)

    def test_freeze_projection_absent_attribute_is_noop(self):
        p_l = _ParamStub()
        # vision_projection attribute intentionally not set.
        model = self._model(_ModuleStub([p_l]), None)
        model.freeze(False, False, True)
        self.assertFalse(p_l.stop_gradient)


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestSharedEmbeddingWeight(unittest.TestCase):
    """shared_embedding_or_output_weight gates on add_decoder."""

    def test_without_decoder_returns_none(self):
        model = LLaVAModel.__new__(LLaVAModel)
        model.add_decoder = False
        self.assertIsNone(model.shared_embedding_or_output_weight())

    def test_with_decoder_delegates_to_language_model(self):
        model = LLaVAModel.__new__(LLaVAModel)
        model.add_decoder = True
        sentinel = object()
        model.language_model = _LangModelStub(sentinel)
        self.assertIs(model.shared_embedding_or_output_weight(), sentinel)


@unittest.skipUnless(_NUMERIC_OK, _NUMERIC_SKIP)
class TestPixelShuffle(unittest.TestCase):
    """pixel_shuffle numeric content vs. an independent numpy reference.

    NOTE (real bug): production ``pixel_shuffle`` unpacks ``n, w, h, c =
    x.size()`` at src/paddlefleet/models/multimodal/llava_model.py:1039.
    In Paddle, ``Tensor.size`` is an int property (element count), not a
    callable, so ``x.size()`` raises ``TypeError`` for every input and the
    function never returns. These tests assert the intended output and are
    marked ``expectedFailure`` to document the bug without editing production.
    """

    def _input(self):
        # sq=4, scale=0.5 -> spatial 4x4 folds to 2x2, so the permutes
        # genuinely reorder elements (non-degenerate; version 1 != version 2).
        return np.arange(1 * 16 * 4, dtype="float32").reshape(1, 16, 4)

    @unittest.expectedFailure
    def test_version2_matches_independent_reference(self):
        x_np = self._input()
        ref = _pixel_shuffle_reference(x_np, 0.5, 2)
        out = pixel_shuffle(paddle.to_tensor(x_np), scale_factor=0.5, version=2)
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-6, atol=0.0)

    @unittest.expectedFailure
    def test_version1_matches_independent_reference(self):
        x_np = self._input()
        ref_v1 = _pixel_shuffle_reference(x_np, 0.5, 1)
        # Guard that the fixture actually distinguishes the two versions.
        self.assertFalse(
            np.array_equal(ref_v1, _pixel_shuffle_reference(x_np, 0.5, 2))
        )
        out = pixel_shuffle(paddle.to_tensor(x_np), scale_factor=0.5, version=1)
        np.testing.assert_allclose(out.numpy(), ref_v1, rtol=1e-6, atol=0.0)


if __name__ == "__main__":
    unittest.main()
