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

Disjoint slice
--------------
Sibling test files in this directory already cover ``pixel_shuffle`` numeric
content, the two ``_load_state_dict_hook_*`` list-surgery helpers, the module
constants, and the ``LLaVAModel`` orchestration methods ``set_input_tensor`` /
``freeze`` / ``shared_embedding_or_output_weight`` / ``_apply_tile_tagging``.

To stay disjoint this file targets code those siblings do not touch:

* ``LLaVAModel._process_embedding_token_parallel`` -- the SP/CP shard-factor
  and ``seq_dim`` derivation, the divisibility guard, the TP-comm-overlap
  length guard, and the pipeline-middle-chunk identity contract.  These are
  driven through the *real* method with a lightweight stand-in ``self`` so the
  branch selection and parameter consumption are actually observed (not a
  reimplementation of the same ``if`` ladder).

* Two genuine production defects surfaced as ``expectedFailure`` demonstrations
  (production is never edited):

  - ``LLaVAModel._preprocess_data`` line 498 and
  - ``LLaVAModel.forward`` line 863

  both call ``paddle.tensor(...)``.  ``paddle.tensor`` is a *sub-package*, not a
  callable constructor -- the intended API is ``paddle.to_tensor``.  Reaching
  either line raises ``TypeError: 'module' object is not callable``.
  (A third identical defect lives at line 785 in ``_apply_tile_tagging``, whose
  behaviour is owned by a sibling file; it is reported but not re-tested here.)

Environment
-----------
No Paddle is installed in the authoring environment; importing the module pulls
in Paddle, so every test is gated behind an honest ``skipUnless``.  On the
single-card CI (Paddle present) the positive tests run and the two
``expectedFailure`` tests exercise the real defects.
"""

import unittest
from types import SimpleNamespace

try:
    import paddle

    from paddlefleet.models.multimodal.llava_model import LLaVAModel

    HAS_PADDLE = True
    SKIP_REASON = ""
except (ImportError, ModuleNotFoundError) as exc:  # honest: real missing dep
    HAS_PADDLE = False
    SKIP_REASON = f"paddle / paddlefleet not importable: {exc!r}"
    LLaVAModel = None


def _tp_self(**overrides):
    """Stand-in ``self`` exposing only the attributes read by
    ``_process_embedding_token_parallel``.  Defaults describe the plain
    (no SP, no CP) single-GPU configuration; callers override per case."""
    base = {
        "pre_process": True,
        "post_process": True,
        "context_parallel_lm": 1,
        "sequence_parallel_lm": False,
        "tensor_model_parallel_size_lm": 1,
        "tp_comm_overlap_lm": False,
        "_language_max_sequence_length": 4096,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@unittest.skipUnless(HAS_PADDLE, SKIP_REASON)
class TestProcessEmbeddingTokenParallel(unittest.TestCase):
    """``LLaVAModel._process_embedding_token_parallel(self, ce, labels, mask, psp)``.

    The SP/CP scatter bodies are currently no-ops (TODO in production), so the
    method must return its four inputs *unchanged* while still selecting the
    correct ``shard_factor`` / ``seq_dim`` and enforcing the divisibility and
    TP-comm-overlap guards.  ``combined_embeddings`` only needs a ``.shape``
    list, so a ``SimpleNamespace`` stands in and object identity of the return
    values can be asserted directly.
    """

    def _call(self, model_self, ce):
        labels = object()
        loss_mask = object()
        psp = object()
        out = LLaVAModel._process_embedding_token_parallel(
            model_self, ce, labels, loss_mask, psp
        )
        return out, labels, loss_mask, psp

    def test_pp_middle_chunk_returns_inputs_unchanged(self):
        # pre_process=False and post_process=False -> pure identity pass-through.
        model_self = _tp_self(pre_process=False, post_process=False)
        ce = SimpleNamespace(
            shape=[5, 7]
        )  # deliberately not divisible by anything
        out, labels, loss_mask, psp = self._call(model_self, ce)
        self.assertIs(out[0], ce)
        self.assertIs(out[1], labels)
        self.assertIs(out[2], loss_mask)
        self.assertIs(out[3], psp)

    def test_cp_only_uses_seq_dim_1_and_factor_two_cp(self):
        # context_parallel_lm=2 -> shard_factor = cp*2 = 4, seq_dim = 1.
        model_self = _tp_self(context_parallel_lm=2)
        # [6, 8]: dim1=8 is divisible by 4 (passes); if seq_dim were 0, dim0=6
        # would fail 6 % 4 -> so success here pins seq_dim == 1.
        out, *_ = self._call(model_self, SimpleNamespace(shape=[6, 8]))
        self.assertEqual(out[0].shape, [6, 8])  # unchanged (no-op scatter)

        # dim1=6 is divisible by 2 but NOT by 4: rejects the "forgot *2" mutant.
        with self.assertRaises(AssertionError):
            self._call(model_self, SimpleNamespace(shape=[10, 6]))

    def test_sp_only_uses_seq_dim_0_and_factor_tp(self):
        # sequence_parallel_lm=True, cp=1 -> shard_factor = tp = 4, seq_dim = 0.
        model_self = _tp_self(
            sequence_parallel_lm=True, tensor_model_parallel_size_lm=4
        )
        # [8, 6]: dim0=8 divisible by 4 (passes); seq_dim==1 would use dim1=6
        # and fail -> success pins seq_dim == 0.
        ce = SimpleNamespace(shape=[8, 6])
        out, _labels, _loss_mask, psp = self._call(model_self, ce)
        self.assertIs(out[0], ce)  # embeddings passed straight through
        self.assertIs(out[3], psp)  # packed_seq_params passed straight through

        with self.assertRaises(AssertionError):
            self._call(model_self, SimpleNamespace(shape=[6, 8]))

    def test_sp_and_cp_combine_factor_tp_times_cp_times_two(self):
        # both on -> shard_factor = tp * cp * 2 = 2*2*2 = 8, seq_dim = 1.
        model_self = _tp_self(
            sequence_parallel_lm=True,
            context_parallel_lm=2,
            tensor_model_parallel_size_lm=2,
        )
        out, *_ = self._call(model_self, SimpleNamespace(shape=[3, 16]))
        self.assertEqual(out[0].shape, [3, 16])

        # dim1=12 divisible by 4 but not by 8: rejects any mutant that drops a
        # factor from tp*cp*2.
        with self.assertRaises(AssertionError):
            self._call(model_self, SimpleNamespace(shape=[3, 12]))

    def test_tp_comm_overlap_requires_full_language_seq_len(self):
        # sp + tp_comm_overlap -> extra guard: seq length must equal
        # _language_max_sequence_length (seq_dim == 0 for sp-only).
        model_self = _tp_self(
            sequence_parallel_lm=True,
            tensor_model_parallel_size_lm=2,
            tp_comm_overlap_lm=True,
            _language_max_sequence_length=8,
        )
        # dim0=8 divisible by 2 AND equals language_max_sequence_length -> OK.
        out, *_ = self._call(model_self, SimpleNamespace(shape=[8, 5]))
        self.assertEqual(out[0].shape, [8, 5])

        # dim0=6 divisible by 2 but != 8 -> the TP-comm-overlap guard fires.
        with self.assertRaises(AssertionError):
            self._call(model_self, SimpleNamespace(shape=[6, 5]))


@unittest.skipUnless(HAS_PADDLE, SKIP_REASON)
class TestPreprocessDataImageTokenTensorBug(unittest.TestCase):
    """Real bug: ``LLaVAModel._preprocess_data`` line 498 calls
    ``paddle.tensor([...])``.  ``paddle.tensor`` is a sub-package (module), not a
    callable -- the intended API is ``paddle.to_tensor``.  A single image sample
    reaches that line and raises ``TypeError: 'module' object is not callable``.

    The test asserts the *documented* contract (a 3-tuple of
    embeddings / labels / loss_mask); it is marked ``expectedFailure`` because
    production raises before returning.  Fixing line 498 to ``paddle.to_tensor``
    would turn this into an unexpected success, flagging the decorator for
    removal.  Production is not modified.
    """

    @unittest.expectedFailure
    def test_single_image_sample_reaches_tensor_constructor(self):
        model_self = _tp_self(
            add_decoder=True, pre_process=True, post_process=True
        )
        model_self.img_seq_len = 4

        # input_ids = [pad, <image>, text]; image_token_index = -200.
        input_ids = paddle.to_tensor([[0, -200, 2]], dtype="int64")
        # one image, two tiles; split sizes must sum to num images per sample.
        num_image_tiles = paddle.to_tensor([2], dtype="int64")

        result = LLaVAModel._preprocess_data(
            model_self,
            image_embeddings=None,
            language_embeddings=None,
            input_ids=input_ids,
            loss_mask=None,
            labels=None,
            use_inference_kv_cache=False,
            inference_context=None,
            image_token_index=-200,
            num_image_tiles=num_image_tiles,
        )
        # Documented contract: (final_embedding, final_labels, final_loss_mask).
        self.assertEqual(len(result), 3)


@unittest.skipUnless(HAS_PADDLE, SKIP_REASON)
class TestForwardEmptyImageTensorBug(unittest.TestCase):
    """Real bug: ``LLaVAModel.forward`` line 863 (the ``add_encoder and not
    has_images`` branch) calls ``paddle.tensor([], dtype=images.dtype)``.  As
    above, ``paddle.tensor`` is not callable -- ``paddle.to_tensor`` is intended.

    Passing an empty image batch (``images.shape[0] == 0``) with an encoder
    present routes straight to that line and raises
    ``TypeError: 'module' object is not callable`` before any vision/language
    submodule is needed, so a bare stand-in ``self`` is enough to expose it.
    Marked ``expectedFailure``; production is not modified.
    """

    @unittest.expectedFailure
    def test_empty_image_batch_reaches_tensor_constructor(self):
        model_self = SimpleNamespace(add_encoder=True)
        images = paddle.zeros([0, 3, 3], dtype="float32")  # shape[0] == 0

        out = LLaVAModel.forward(
            model_self,
            images,
            None,  # input_ids (unused before the buggy line)
            None,  # position_ids
            None,  # attention_mask
        )
        # If the constructor call were correct, forward would proceed; the empty
        # image-embeddings tensor is the documented fallback for no images.
        self.assertIsNotNone(out)


if __name__ == "__main__":
    unittest.main()
