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

"""Model-layer tests for ``paddlefleet.models.multimodal.llava_model``.

Slice keyed to the imports ``llava_model`` (module), ``LLaVAModel`` and
``pixel_shuffle``.  The behaviours exercised here are the ones reachable from
those imports without constructing a full multimodal model:

* ``LLaVAModel.shared_embedding_or_output_weight`` -- the ``add_decoder`` guard
  and delegation to the language model.
* ``LLaVAModel.set_input_tensor`` -- list normalisation, the length-1 contract
  and the four mutually-exclusive routing branches.
* ``LLaVAModel.freeze`` -- which modules get ``stop_gradient`` set, that
  ``None`` modules and a missing ``vision_projection`` attribute are skipped,
  and that non-frozen modules are left untouched.
* ``_load_state_dict_hook_ignore_param_names`` /
  ``_load_state_dict_hook_ignore_extra_state`` -- exactly which keys are
  removed from ``missing_keys`` / ``unexpected_keys`` and which are kept.
* ``pixel_shuffle`` -- content (not just shape) against an independent NumPy
  reference, on a non-degenerate input where the ``version`` permute actually
  reorders elements, so version 1 and version 2 are distinguishable.

The methods are invoked as unbound functions against light-weight fake ``self``
objects so no full model construction (and no GPU) is required.  Only
``pixel_shuffle`` needs real tensor math.  Importing the module pulls in
``paddle`` transitively, so the whole suite skips with an honest reason when
Paddle is unavailable; nothing here fakes a pass.
"""

import os
import sys
import unittest
from collections import namedtuple

import numpy as np

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)
# Also make the ``src/`` layout importable when the package is not installed.
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
        ),
        "src",
    ),
)

try:
    import paddle

    from paddlefleet.models.multimodal import llava_model as llava_mod
    from paddlefleet.models.multimodal.llava_model import (
        LLaVAModel,
        pixel_shuffle,
    )

    _IMPORT_OK = True
    _IMPORT_ERR = ""
except ImportError as exc:  # pragma: no cover - depends on runtime deps
    _IMPORT_OK = False
    _IMPORT_ERR = repr(exc)
    paddle = None
    llava_mod = None
    LLaVAModel = None
    pixel_shuffle = None

_SKIP_REASON = (
    "paddlefleet.models.multimodal.llava_model not importable "
    "(missing paddle or package): " + _IMPORT_ERR
)


class _Recorder:
    """Stand-in for a sub-module exposing ``set_input_tensor``."""

    def __init__(self):
        self.received = []

    def set_input_tensor(self, value):
        self.received.append(value)


class _Param:
    def __init__(self, stop_gradient=False):
        self.stop_gradient = stop_gradient


class _ParamModule:
    """Stand-in module whose ``parameters()`` yields fake params."""

    def __init__(self, n=2):
        self._params = [_Param() for _ in range(n)]

    def parameters(self):
        return list(self._params)


class _FakeSelf:
    """Bare attribute holder used as ``self`` for unbound method calls."""


@unittest.skipUnless(_IMPORT_OK, _SKIP_REASON)
class TestSharedEmbeddingOrOutputWeight(unittest.TestCase):
    def test_returns_language_model_weight_when_decoder_present(self):
        sentinel = object()

        class _LM:
            def shared_embedding_or_output_weight(self):
                return sentinel

        model = _FakeSelf()
        model.add_decoder = True
        model.language_model = _LM()
        # Must return the *delegated* object identity, not merely non-None.
        self.assertIs(
            LLaVAModel.shared_embedding_or_output_weight(model), sentinel
        )

    def test_returns_none_without_decoder(self):
        model = _FakeSelf()
        model.add_decoder = False
        # language_model deliberately absent: the guard must short-circuit
        # before touching it.
        self.assertIsNone(LLaVAModel.shared_embedding_or_output_weight(model))


@unittest.skipUnless(_IMPORT_OK, _SKIP_REASON)
class TestSetInputTensor(unittest.TestCase):
    def _make(self, **flags):
        model = _FakeSelf()
        model.vision_model = _Recorder()
        model.language_model = _Recorder()
        model.encoder_hidden_state = None
        model.add_encoder = flags["add_encoder"]
        model.add_decoder = flags["add_decoder"]
        model.pre_process = flags["pre_process"]
        return model

    def test_encoder_and_decoder_routes_to_vision_and_wraps_scalar(self):
        marker = object()
        model = self._make(add_encoder=True, add_decoder=True, pre_process=True)
        # Bare (non-list) input must be normalised to a length-1 list.
        LLaVAModel.set_input_tensor(model, marker)
        self.assertEqual(model.vision_model.received, [marker])
        self.assertEqual(model.language_model.received, [])
        self.assertIsNone(model.encoder_hidden_state)

    def test_encoder_only_routes_to_vision_from_list(self):
        marker = object()
        model = self._make(
            add_encoder=True, add_decoder=False, pre_process=True
        )
        LLaVAModel.set_input_tensor(model, [marker])
        self.assertEqual(model.vision_model.received, [marker])
        self.assertEqual(model.language_model.received, [])

    def test_pre_process_branch_stores_encoder_hidden_state(self):
        marker = object()
        model = self._make(
            add_encoder=False, add_decoder=False, pre_process=True
        )
        LLaVAModel.set_input_tensor(model, marker)
        self.assertIs(model.encoder_hidden_state, marker)
        self.assertEqual(model.vision_model.received, [])
        self.assertEqual(model.language_model.received, [])

    def test_final_branch_routes_to_language_model(self):
        marker = object()
        model = self._make(
            add_encoder=False, add_decoder=True, pre_process=False
        )
        LLaVAModel.set_input_tensor(model, marker)
        self.assertEqual(model.language_model.received, [marker])
        self.assertEqual(model.vision_model.received, [])
        self.assertIsNone(model.encoder_hidden_state)

    def test_length_not_one_is_rejected(self):
        model = self._make(add_encoder=True, add_decoder=True, pre_process=True)
        with self.assertRaises(AssertionError):
            LLaVAModel.set_input_tensor(model, [object(), object()])


@unittest.skipUnless(_IMPORT_OK, _SKIP_REASON)
class TestFreeze(unittest.TestCase):
    def _make(self):
        model = _FakeSelf()
        model.language_model = _ParamModule()
        model.vision_model = _ParamModule()
        model.vision_projection = _ParamModule()
        return model

    def test_selective_freeze_sets_only_requested_modules(self):
        model = self._make()
        LLaVAModel.freeze(
            model,
            freeze_language_model=True,
            freeze_vision_model=False,
            freeze_vision_projection=True,
        )
        self.assertTrue(
            all(p.stop_gradient for p in model.language_model.parameters())
        )
        # Not requested -> must remain trainable.
        self.assertTrue(
            all(not p.stop_gradient for p in model.vision_model.parameters())
        )
        self.assertTrue(
            all(p.stop_gradient for p in model.vision_projection.parameters())
        )

    def test_freeze_all_sets_every_parameter(self):
        model = self._make()
        LLaVAModel.freeze(
            model,
            freeze_language_model=True,
            freeze_vision_model=True,
            freeze_vision_projection=True,
        )
        for mod in (
            model.language_model,
            model.vision_model,
            model.vision_projection,
        ):
            self.assertTrue(all(p.stop_gradient for p in mod.parameters()))

    def test_freeze_none_leaves_all_trainable(self):
        model = self._make()
        LLaVAModel.freeze(
            model,
            freeze_language_model=False,
            freeze_vision_model=False,
            freeze_vision_projection=False,
        )
        for mod in (
            model.language_model,
            model.vision_model,
            model.vision_projection,
        ):
            self.assertTrue(all(not p.stop_gradient for p in mod.parameters()))

    def test_none_module_is_skipped(self):
        model = self._make()
        model.vision_model = None
        # Requesting a freeze on a None module must be a no-op, not a crash,
        # and must not disturb the modules that do exist.
        LLaVAModel.freeze(
            model,
            freeze_language_model=True,
            freeze_vision_model=True,
            freeze_vision_projection=False,
        )
        self.assertTrue(
            all(p.stop_gradient for p in model.language_model.parameters())
        )
        self.assertTrue(
            all(
                not p.stop_gradient
                for p in model.vision_projection.parameters()
            )
        )

    def test_missing_vision_projection_attr_is_skipped(self):
        model = _FakeSelf()
        model.language_model = _ParamModule()
        model.vision_model = _ParamModule()
        # No ``vision_projection`` attribute at all: the ``hasattr`` guard must
        # prevent an AttributeError even when freezing is requested.
        LLaVAModel.freeze(
            model,
            freeze_language_model=False,
            freeze_vision_model=False,
            freeze_vision_projection=True,
        )
        self.assertFalse(hasattr(model, "vision_projection"))
        self.assertTrue(
            all(not p.stop_gradient for p in model.language_model.parameters())
        )


def _incompatible(missing, unexpected):
    incompatible_type = namedtuple(
        "IncompatibleKeys", ["missing_keys", "unexpected_keys"]
    )
    return incompatible_type(
        missing_keys=list(missing), unexpected_keys=list(unexpected)
    )


@unittest.skipUnless(_IMPORT_OK, _SKIP_REASON)
class TestLoadStateDictHooks(unittest.TestCase):
    def test_ignore_param_names_removes_only_listed_missing_keys(self):
        incompatible = _incompatible(
            missing=["a.weight", "b.weight", "c.weight"],
            unexpected=["b.weight", "x.bias"],
        )
        llava_mod._load_state_dict_hook_ignore_param_names(
            ["b.weight", "not.present"], None, incompatible
        )
        # Only ``b.weight`` is dropped from missing_keys; a listed name that is
        # absent is a silent no-op and unexpected_keys is untouched (even though
        # it also contains ``b.weight``).
        self.assertEqual(incompatible.missing_keys, ["a.weight", "c.weight"])
        self.assertEqual(incompatible.unexpected_keys, ["b.weight", "x.bias"])

    def test_ignore_param_names_empty_list_is_noop(self):
        incompatible = _incompatible(missing=["a.weight"], unexpected=["y"])
        llava_mod._load_state_dict_hook_ignore_param_names(
            [], None, incompatible
        )
        self.assertEqual(incompatible.missing_keys, ["a.weight"])
        self.assertEqual(incompatible.unexpected_keys, ["y"])

    def test_ignore_extra_state_strips_from_both_key_lists(self):
        incompatible = _incompatible(
            missing=["m.extra_state", "m.weight", "n.extra_state"],
            unexpected=["u.extra_state", "u.bias", "v.extra_state"],
        )
        llava_mod._load_state_dict_hook_ignore_extra_state(None, incompatible)
        # Every ``extra_state`` key is removed from *both* lists; real weights
        # are preserved and their order is kept.
        self.assertEqual(incompatible.missing_keys, ["m.weight"])
        self.assertEqual(incompatible.unexpected_keys, ["u.bias"])

    def test_ignore_extra_state_keeps_lists_without_extra_state(self):
        incompatible = _incompatible(
            missing=["m.weight"], unexpected=["u.bias", "v.bias"]
        )
        llava_mod._load_state_dict_hook_ignore_extra_state(None, incompatible)
        self.assertEqual(incompatible.missing_keys, ["m.weight"])
        self.assertEqual(incompatible.unexpected_keys, ["u.bias", "v.bias"])


def _reference_pixel_shuffle(arr, scale_factor=0.5, version=2):
    """Independent NumPy re-derivation of the InternVL pixel-shuffle spec.

    Reproduces the reshape/permute sequence with NumPy semantics rather than
    calling the function under test, so a wrong reorder in production is not
    mirrored here.
    """
    arr = np.asarray(arr, dtype=np.float64)
    num_tiles = arr.shape[0]
    side = int(arr.shape[1] ** 0.5)
    x = arr.reshape(num_tiles, side, side, -1)
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


@unittest.skipUnless(_IMPORT_OK, _SKIP_REASON)
class TestPixelShuffle(unittest.TestCase):
    def test_matches_reference_and_versions_differ(self):
        # side = 4 (16 = 4**2) and h_vision = 8 make every permuted axis larger
        # than 1, so the version-2 permute genuinely reorders elements.
        base = np.arange(128, dtype=np.float32).reshape(1, 16, 8)
        x = paddle.to_tensor(base)

        out_v1 = pixel_shuffle(x, scale_factor=0.5, version=1)
        out_v2 = pixel_shuffle(x, scale_factor=0.5, version=2)

        ref_v1 = _reference_pixel_shuffle(base, version=1)
        ref_v2 = _reference_pixel_shuffle(base, version=2)

        self.assertEqual(list(out_v1.shape), [1, 4, 32])
        self.assertEqual(list(out_v2.shape), [1, 4, 32])
        np.testing.assert_array_equal(out_v1.numpy(), ref_v1)
        np.testing.assert_array_equal(out_v2.numpy(), ref_v2)
        # The reference itself must distinguish the two versions on this input,
        # otherwise the content check above would be degenerate.
        self.assertFalse(np.array_equal(ref_v1, ref_v2))
        self.assertFalse(np.array_equal(out_v1.numpy(), out_v2.numpy()))


if __name__ == "__main__":
    unittest.main()
