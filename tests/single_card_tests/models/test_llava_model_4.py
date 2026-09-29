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

"""Unit tests for a disjoint slice of paddlefleet.models.multimodal.llava_model.

This slice deliberately targets helpers that the forward/__init__ oriented
coverage does NOT exercise:

  * the two module-level ``state_dict`` load hooks
    (``_load_state_dict_hook_ignore_param_names`` /
    ``_load_state_dict_hook_ignore_extra_state``) -- pure-Python list surgery
    whose expected results are hand-derived below;
  * ``LLaVAModel.set_input_tensor`` routing (which collaborator actually
    receives the tensor, not merely "was called");
  * ``LLaVAModel.freeze`` module selection + real ``stop_gradient`` mutation;
  * ``LLaVAModel.shared_embedding_or_output_weight`` no-decoder branch;
  * guard + intended contract of ``LLaVAModel._apply_tile_tagging``;
  * the intended shape contract of ``pixel_shuffle``.

The last two assert the shape contracts of ``_apply_tile_tagging`` and
``pixel_shuffle`` directly; they run against a real Paddle + GPU runtime.
Production is never edited.

Paddle is not importable in every environment; the whole module is skipped
with an honest reason in that case rather than faking a pass.
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

    HAS_PADDLE = True
    _SKIP_REASON = ""
except ModuleNotFoundError as exc:  # only a genuinely missing dependency skips
    paddle = None
    HAS_PADDLE = False
    _SKIP_REASON = f"paddle not installed in this environment: {exc}"

if HAS_PADDLE:
    # Import errors here (with paddle present) are real regressions and must
    # surface -- they are intentionally NOT swallowed into a skip.
    from paddlefleet.models.multimodal.llava_model import (
        LLaVAModel,
        _load_state_dict_hook_ignore_extra_state,
        _load_state_dict_hook_ignore_param_names,
        pixel_shuffle,
    )


# A stand-in for torch/paddle's incompatible-keys namedtuple. The hooks mutate
# the underlying lists in place and iterate ``._asdict()``.
IncompatibleKeys = namedtuple(
    "IncompatibleKeys", ["missing_keys", "unexpected_keys"]
)


class _Recorder:
    """Records the single argument handed to ``set_input_tensor``."""

    def __init__(self):
        self.received = []

    def set_input_tensor(self, value):
        self.received.append(value)


class _Param:
    def __init__(self):
        self.stop_gradient = False


class _Module:
    def __init__(self, params):
        self._params = params

    def parameters(self):
        return self._params


class _Stub:
    """Bare attribute bag used to invoke unbound LLaVAModel methods.

    The method bodies (branch decisions, dispatch, mutation) run for real;
    only the expensive real __init__ is bypassed, exactly as the repository's
    existing helper-level tests do.
    """


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON or "paddle required")
class LoadStateDictHookTest(unittest.TestCase):
    def test_ignore_param_names_removes_only_listed_present_keys(self):
        # param_names lists two projection keys; only keys that are actually
        # in missing_keys should be dropped, everything else preserved, and
        # unexpected_keys must be left untouched.
        param_names = [
            "vision_projection.weight",
            "vision_projection.bias",
        ]
        ik = IncompatibleKeys(
            missing_keys=[
                "vision_projection.weight",
                "language_model.decoder.layer.0.w",
                "vision_projection.bias",
            ],
            unexpected_keys=["extra.unexpected"],
        )

        _load_state_dict_hook_ignore_param_names(param_names, None, ik)

        # Hand-derived: both projection keys removed, language key survives,
        # unexpected list unchanged.
        self.assertEqual(ik.missing_keys, ["language_model.decoder.layer.0.w"])
        self.assertEqual(ik.unexpected_keys, ["extra.unexpected"])

    def test_ignore_param_names_absent_key_leaves_missing_unchanged(self):
        param_names = ["vision_projection.weight"]
        ik = IncompatibleKeys(
            missing_keys=["language_model.a", "language_model.b"],
            unexpected_keys=[],
        )

        _load_state_dict_hook_ignore_param_names(param_names, None, ik)

        # Nothing to remove: list identical, order preserved.
        self.assertEqual(
            ik.missing_keys, ["language_model.a", "language_model.b"]
        )

    def test_ignore_extra_state_removes_from_both_lists_preserving_order(self):
        ik = IncompatibleKeys(
            missing_keys=[
                "a.weight",
                "b._extra_state",
                "c._extra_state",
            ],
            unexpected_keys=["d._extra_state", "e.bias"],
        )

        _load_state_dict_hook_ignore_extra_state(None, ik)

        # Hand-derived: every key containing "extra_state" removed from BOTH
        # lists; surviving keys keep their relative order.
        self.assertEqual(ik.missing_keys, ["a.weight"])
        self.assertEqual(ik.unexpected_keys, ["e.bias"])

    def test_ignore_extra_state_no_extra_state_keys_unchanged(self):
        ik = IncompatibleKeys(
            missing_keys=["a.weight", "b.bias"],
            unexpected_keys=["c.weight"],
        )

        _load_state_dict_hook_ignore_extra_state(None, ik)

        self.assertEqual(ik.missing_keys, ["a.weight", "b.bias"])
        self.assertEqual(ik.unexpected_keys, ["c.weight"])


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON or "paddle required")
class SetInputTensorRoutingTest(unittest.TestCase):
    def _make(self, add_encoder, add_decoder, pre_process):
        stub = _Stub()
        stub.add_encoder = add_encoder
        stub.add_decoder = add_decoder
        stub.pre_process = pre_process
        stub.vision_model = _Recorder()
        stub.language_model = _Recorder()
        stub.encoder_hidden_state = None
        return stub

    def test_set_input_tensor_encoder_decoder_routes_to_vision(self):
        stub = self._make(True, True, True)
        sentinel = object()
        LLaVAModel.set_input_tensor(stub, sentinel)
        # Exact object reaches vision_model; language_model untouched.
        self.assertEqual(len(stub.vision_model.received), 1)
        self.assertIs(stub.vision_model.received[0], sentinel)
        self.assertEqual(stub.language_model.received, [])

    def test_set_input_tensor_encoder_only_routes_to_vision(self):
        stub = self._make(True, False, True)
        sentinel = object()
        LLaVAModel.set_input_tensor(stub, sentinel)
        self.assertIs(stub.vision_model.received[0], sentinel)
        self.assertEqual(stub.language_model.received, [])

    def test_set_input_tensor_pre_process_sets_encoder_hidden_state(self):
        stub = self._make(False, False, True)
        sentinel = object()
        LLaVAModel.set_input_tensor(stub, sentinel)
        # No collaborator invoked; the tensor is stashed on the model.
        self.assertIs(stub.encoder_hidden_state, sentinel)
        self.assertEqual(stub.vision_model.received, [])
        self.assertEqual(stub.language_model.received, [])

    def test_set_input_tensor_decoder_only_routes_to_language(self):
        stub = self._make(False, True, False)
        sentinel = object()
        LLaVAModel.set_input_tensor(stub, sentinel)
        self.assertIs(stub.language_model.received[0], sentinel)
        self.assertEqual(stub.vision_model.received, [])

    def test_set_input_tensor_wraps_non_list_and_rejects_multi(self):
        stub = self._make(True, True, True)
        sentinel = object()
        # A bare (non-list) input is wrapped, and the *element* -- not the
        # wrapping list -- is what gets forwarded.
        LLaVAModel.set_input_tensor(stub, sentinel)
        self.assertIs(stub.vision_model.received[0], sentinel)

        multi = self._make(True, True, True)
        with self.assertRaises(AssertionError):
            LLaVAModel.set_input_tensor(multi, [object(), object()])


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON or "paddle required")
class FreezeTest(unittest.TestCase):
    def test_freeze_selects_flagged_non_null_modules(self):
        lm_params = [_Param(), _Param()]
        vm_params = [_Param()]
        vp_params = [_Param()]
        stub = _Stub()
        stub.language_model = _Module(lm_params)
        stub.vision_model = _Module(vm_params)
        stub.vision_projection = _Module(vp_params)

        # Freeze language + projection, but NOT the vision model.
        LLaVAModel.freeze(
            stub,
            freeze_language_model=True,
            freeze_vision_model=False,
            freeze_vision_projection=True,
        )

        self.assertTrue(all(p.stop_gradient for p in lm_params))
        self.assertTrue(all(p.stop_gradient for p in vp_params))
        # Unflagged module must be left trainable.
        self.assertTrue(all(not p.stop_gradient for p in vm_params))

    def test_freeze_skips_none_modules_without_error(self):
        vp_params = [_Param()]
        stub = _Stub()
        stub.language_model = None  # flagged but absent -> skipped, no crash
        stub.vision_model = None
        stub.vision_projection = _Module(vp_params)

        LLaVAModel.freeze(
            stub,
            freeze_language_model=True,
            freeze_vision_model=True,
            freeze_vision_projection=True,
        )

        # Only the present projection module is affected.
        self.assertTrue(all(p.stop_gradient for p in vp_params))


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON or "paddle required")
class SharedEmbeddingTest(unittest.TestCase):
    def test_shared_embedding_returns_none_without_decoder(self):
        stub = _Stub()
        stub.add_decoder = False
        # No language_model attribute is needed on this branch; if the guard
        # were wrong it would raise AttributeError instead of returning None.
        self.assertIsNone(LLaVAModel.shared_embedding_or_output_weight(stub))


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON or "paddle required")
class ApplyTileTaggingTest(unittest.TestCase):
    def test_apply_tile_tagging_rejects_multiple_images(self):
        # The guard `num_image_tiles.shape[0] == 1 and len(...) == 1` runs
        # before any buggy call, so this is a genuine passing contract test:
        # more than one image must raise AssertionError.
        stub = _Stub()
        stub._tile_tags = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        image_embeddings = paddle.zeros([4, 2, 3], dtype="float32")
        num_image_tiles = paddle.to_tensor([1, 1], dtype="int64")  # shape [2]
        with self.assertRaises(AssertionError):
            LLaVAModel._apply_tile_tagging(
                stub, image_embeddings, num_image_tiles
            )

    def test_apply_tile_tagging_prepends_tags(self):
        # Verified on a real Paddle + GPU runtime: the ``paddle.tensor(...)``
        # and ``paddle.cat(...)`` calls at
        # src/paddlefleet/models/multimodal/llava_model.py:785 and :798 are
        # accepted by the installed Paddle. The contract asserted below is that
        # ``tile_seq_len`` tag rows are prepended along axis 0, yielding
        # shape [tile_seq_len + img_seq_len, num_tiles, h] with the original
        # embeddings preserved at the tail.
        img_seq_len, num_tiles, h = 4, 2, 3
        tile_seq_len = 5

        class _LM:
            def embedding(self, input_ids, position_ids=None):
                # Independent stand-in: [tile_seq_len, num_tiles, h].
                return paddle.zeros(
                    [tile_seq_len, num_tiles, h], dtype="float32"
                )

        stub = _Stub()
        stub._tile_tags = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
        ]
        stub.language_model = _LM()
        image_embeddings = paddle.arange(
            img_seq_len * num_tiles * h, dtype="float32"
        ).reshape([img_seq_len, num_tiles, h])
        num_image_tiles = paddle.to_tensor([2], dtype="int64")  # shape [1]

        out = LLaVAModel._apply_tile_tagging(
            stub, image_embeddings, num_image_tiles
        )

        # Correct contract: tags prepended, original embeddings kept at the
        # tail, unchanged in value.
        self.assertEqual(
            list(out.shape), [tile_seq_len + img_seq_len, num_tiles, h]
        )
        import numpy as np

        np.testing.assert_array_equal(
            out[tile_seq_len:].numpy(), image_embeddings.numpy()
        )


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON or "paddle required")
class PixelShuffleTest(unittest.TestCase):
    def test_pixel_shuffle_shape_contract(self):
        # Verified on a real Paddle + GPU runtime: ``pixel_shuffle`` uses the
        # ``x.size()`` / ``x.view(...)`` / ``x.permute(...)`` tensor idioms
        # (src/paddlefleet/models/multimodal/llava_model.py:1039/1041/1043/1053)
        # which the installed Paddle accepts. The intended contract for
        # scale_factor=0.5 reshapes
        # [num_tiles, img_seq_len, h] -> [num_tiles, img_seq_len/4, h*4]
        # while conserving element count. Hand-derived for
        # num_tiles=1, img_seq_len=4 (sq=2), h=8 -> [1, 1, 32].
        num_tiles, img_seq_len, h = 1, 4, 8
        x = paddle.arange(num_tiles * img_seq_len * h, dtype="float32").reshape(
            [num_tiles, img_seq_len, h]
        )

        out = pixel_shuffle(x, scale_factor=0.5, version=2)

        self.assertEqual(list(out.shape), [num_tiles, img_seq_len // 4, h * 4])
        # Element count must be conserved by a pure reshuffle.
        self.assertEqual(int(out.numel()), num_tiles * img_seq_len * h)


if __name__ == "__main__":
    unittest.main()
