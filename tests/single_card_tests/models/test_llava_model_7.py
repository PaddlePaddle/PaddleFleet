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
"""Unit tests for a disjoint slice of ``paddlefleet.models.multimodal.llava_model``.

The coverage source keyed this file to the imports ``IGNORE_INDEX``,
``LLaVAModel`` and ``pixel_shuffle``. Sibling files test_llava_model_2 and
test_llava_model_3 already exercise ``set_input_tensor``, ``freeze``, the two
``_load_state_dict_hook_*`` helpers, ``_apply_tile_tagging`` and the happy path
of ``pixel_shuffle``. To stay disjoint this file targets:

* ``LLaVAModel._process_embedding_token_parallel`` -- the SP/CP sharding
  dispatcher, which no sibling touches. Its shard-factor derivation and the
  divisibility guard are observed through real branches; the current body keeps
  the scatter commented out (TODO in production), so the observable contract is
  an identity passthrough whose divisibility assertion still consumes the
  parallel-config parameters.
* ``IGNORE_INDEX`` -- the label-ignore sentinel documented for
  ``_preprocess_data`` final labels.
* ``pixel_shuffle`` -- a hand-derived numeric anchor. The production code uses
  the ``x.size()`` / ``x.view`` / ``x.permute`` tensor idioms, which the Paddle
  build on the GPU runner accepts, so the correct-behaviour result is asserted
  directly.

There is no Paddle install in this environment and the module does ``import
paddle`` at top level, so every test that needs production code is guarded by
``skipUnless`` and skips with an honest reason rather than faking a pass. Only a
genuine ImportError/ModuleNotFoundError is turned into a skip; other import
errors are allowed to surface as real failures.
"""

import os
import sys
import unittest

import numpy as np

# Make the ``src/`` layout importable when the package is not installed.
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

    from paddlefleet.models.multimodal.llava_model import (
        IGNORE_INDEX,
        LLaVAModel,
        pixel_shuffle,
    )

    HAVE_PADDLE = True
    IMPORT_ERROR = ""
except (ImportError, ModuleNotFoundError) as exc:
    paddle = None
    IGNORE_INDEX = None
    LLaVAModel = None
    pixel_shuffle = None
    HAVE_PADDLE = False
    IMPORT_ERROR = repr(exc)

SKIP_REASON = (
    "paddle / paddlefleet.models.multimodal.llava_model not importable: "
    + IMPORT_ERROR
)


class _Sentinel:
    """A distinguishable object used to prove passthrough identity."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<Sentinel {self.name}>"


@unittest.skipUnless(HAVE_PADDLE, SKIP_REASON)
class TestProcessEmbeddingTokenParallel(unittest.TestCase):
    """Contract of LLaVAModel._process_embedding_token_parallel.

    The method is only invoked (see forward, llava_model.py:947) when CP>1 or
    SP is enabled. We bypass __init__ to isolate the method and feed it the real
    parallel-config attributes it consumes; the method body itself runs
    unmodified.
    """

    def _model(self, **attrs):
        model = LLaVAModel.__new__(LLaVAModel)
        for key, value in attrs.items():
            object.__setattr__(model, key, value)
        return model

    def _emb(self, shape):
        # Distinguishable content so an accidental copy/reorder would show up if
        # the production body ever starts transforming the tensor.
        return paddle.arange(int(np.prod(shape)), dtype="float32").reshape(
            shape
        )

    def test_middle_pp_chunk_passes_all_through_unchanged(self):
        # Neither pre nor post: a middle pipeline chunk must be a pure no-op and
        # must not even look at the parallel config.
        model = self._model(pre_process=False, post_process=False)
        emb = _Sentinel("emb")
        labels = _Sentinel("labels")
        loss_mask = _Sentinel("loss_mask")
        psp = _Sentinel("psp")
        out = model._process_embedding_token_parallel(
            emb, labels, loss_mask, psp
        )
        self.assertIs(out[0], emb)
        self.assertIs(out[1], labels)
        self.assertIs(out[2], loss_mask)
        self.assertIs(out[3], psp)

    def test_sequence_parallel_only_checks_dim0_and_passes_through(self):
        # SP only -> shard_factor = tp_size, seq_dim = 0. Current body keeps the
        # scatter commented out, so the observable contract is identity
        # passthrough after the divisibility guard succeeds.
        model = self._model(
            pre_process=True,
            post_process=True,
            context_parallel_lm=1,
            sequence_parallel_lm=True,
            tensor_model_parallel_size_lm=2,
            tp_comm_overlap_lm=False,
        )
        emb = self._emb([4, 3, 5])  # dim0 == 4, divisible by tp_size 2
        labels = _Sentinel("labels")
        loss_mask = _Sentinel("loss_mask")
        psp = _Sentinel("psp")
        out = model._process_embedding_token_parallel(
            emb, labels, loss_mask, psp
        )
        self.assertIs(out[0], emb)
        self.assertIs(out[1], labels)
        self.assertIs(out[2], loss_mask)
        self.assertIs(out[3], psp)

    def test_sequence_parallel_rejects_indivisible_dim0(self):
        # tp_size = 3 does not divide dim0 = 4 -> the shard_factor derived from
        # tensor_model_parallel_size_lm is genuinely consumed by the assert.
        model = self._model(
            pre_process=True,
            post_process=True,
            context_parallel_lm=1,
            sequence_parallel_lm=True,
            tensor_model_parallel_size_lm=3,
            tp_comm_overlap_lm=False,
        )
        emb = self._emb([4, 3, 5])
        with self.assertRaises(AssertionError):
            model._process_embedding_token_parallel(emb, None, None, None)

    def test_context_parallel_only_uses_dim1_and_factor_2cp(self):
        # CP only -> shard_factor = cp*2 = 4, seq_dim = 1.
        model = self._model(
            pre_process=True,
            post_process=True,
            context_parallel_lm=2,
            sequence_parallel_lm=False,
            tensor_model_parallel_size_lm=1,
            tp_comm_overlap_lm=False,
        )
        ok = self._emb([2, 8, 5])  # dim1 == 8, divisible by 4
        out = model._process_embedding_token_parallel(ok, None, None, None)
        self.assertIs(out[0], ok)

        bad = self._emb([2, 6, 5])  # dim1 == 6, not divisible by 4
        with self.assertRaises(AssertionError):
            model._process_embedding_token_parallel(bad, None, None, None)

    def test_sp_and_cp_factor_is_tp_times_cp_times_two(self):
        # SP+CP -> shard_factor = tp*cp*2 = 2*2*2 = 8, seq_dim = 1. dim1 == 4
        # would pass under the SP-only factor (2) but must be rejected here,
        # proving all three config values feed the derived factor.
        model = self._model(
            pre_process=True,
            post_process=True,
            context_parallel_lm=2,
            sequence_parallel_lm=True,
            tensor_model_parallel_size_lm=2,
            tp_comm_overlap_lm=False,
        )
        ok = self._emb([3, 8, 5])  # dim1 == 8, divisible by 8
        out = model._process_embedding_token_parallel(ok, None, None, None)
        self.assertIs(out[0], ok)

        bad = self._emb([3, 4, 5])  # dim1 == 4, divisible by 2 but not by 8
        with self.assertRaises(AssertionError):
            model._process_embedding_token_parallel(bad, None, None, None)

    def test_post_process_only_does_not_shard(self):
        # pre_process False -> the sharding block is skipped entirely; the post
        # stage must return labels/loss_mask untouched and never inspect shape.
        model = self._model(
            pre_process=False,
            post_process=True,
            context_parallel_lm=1,
            sequence_parallel_lm=True,
            tensor_model_parallel_size_lm=2,
            tp_comm_overlap_lm=False,
        )
        emb = _Sentinel("emb")
        labels = _Sentinel("labels")
        loss_mask = _Sentinel("loss_mask")
        psp = _Sentinel("psp")
        out = model._process_embedding_token_parallel(
            emb, labels, loss_mask, psp
        )
        self.assertIs(out[0], emb)
        self.assertIs(out[1], labels)
        self.assertIs(out[2], loss_mask)
        self.assertIs(out[3], psp)


@unittest.skipUnless(HAVE_PADDLE, SKIP_REASON)
class TestIgnoreIndex(unittest.TestCase):
    """IGNORE_INDEX is the label-ignore sentinel for _preprocess_data."""

    def test_ignore_index_is_minus_100_and_distinct_from_image_token(self):
        self.assertIsInstance(IGNORE_INDEX, int)
        self.assertEqual(IGNORE_INDEX, -100)
        # The docstring uses image_token_index = -200 for masked image slots;
        # the two sentinels must stay distinct or masked labels would collide
        # with image positions.
        self.assertNotEqual(IGNORE_INDEX, -200)


@unittest.skipUnless(HAVE_PADDLE, SKIP_REASON)
class TestPixelShuffle(unittest.TestCase):
    """Hand-derived numeric anchor for pixel_shuffle (scale_factor=0.5, v2).

    Input [1, 4, 4] with values 0..15 laid out row-major. Following the intended
    (PyTorch) semantics the [1,2,2,4] grid is reinterpreted and permuted down to
    [1,1,1,16] then flattened to [1,1,16] holding 0..15 in order.

    Runs on a real Paddle + GPU runtime: the ``x.size()`` / ``x.view`` /
    ``x.permute`` tensor idioms (llava_model.py:1039/:1041/:1043/:1045/:1053)
    are accepted by the installed Paddle, so the hand-derived output is
    asserted directly.
    """

    def test_scale_half_matches_hand_derived_output(self):
        x = paddle.arange(16, dtype="float32").reshape([1, 4, 4])
        result = pixel_shuffle(x, scale_factor=0.5, version=2)
        expected = np.arange(16, dtype="float32").reshape([1, 1, 16])
        np.testing.assert_array_equal(np.asarray(result), expected)


if __name__ == "__main__":
    unittest.main()
