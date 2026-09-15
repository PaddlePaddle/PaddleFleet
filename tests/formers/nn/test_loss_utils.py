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
"""Genuine (no-card / CPU) behavior tests for the SFT loss utilities.

Covered production entries (all imported from real modules, no mocking of the
code under test):

* ``paddlefleet.nn.criterion.loss_utils.subbatch`` -- chunked application of a
  function along one axis, with concatenation order, uneven last chunk,
  ``same_arg_idx`` tensor reuse, width-mismatch guards and the recompute branch.
* ``paddlefleet.nn.criterion.loss_utils.calc_lm_head_logits`` -- with
  ``tensor_model_parallel_size == 1`` this reduces to a plain
  ``hidden @ weight.T (+ bias)`` matmul, verified against an independent numpy
  reference.
* ``paddlefleet.nn.criterion.sft_loss.sft_postprocess_loss`` -- valid-token
  masking and token-averaged reduction (the actual training objective).
* ``paddlefleet.nn.criterion.sft_loss.loss_impl`` -- float32 cast + delegation.

Every expected value is hand-derived on tiny, position-distinguishable inputs so
that dropped masking, swapped chunk order or a wrong reduction denominator is
rejected -- shape-only assertions are deliberately avoided.

Tensor-parallel / sequence-parallel numeric paths of ``calc_lm_head_logits`` and
``parallel_matmul`` require a real multi-rank process group and are NOT exercised
here (see the explicitly skipped placeholder); this file only verifies the
single-rank CPU path.
"""

import unittest
from types import SimpleNamespace

import numpy as np
import paddle

from paddlefleet.nn.criterion.loss_utils import calc_lm_head_logits, subbatch
from paddlefleet.nn.criterion.sft_loss import loss_impl, sft_postprocess_loss

paddle.set_device("cpu")

IGNORE_INDEX = -100


class TestSubbatch(unittest.TestCase):
    """subbatch splits an axis into bs-sized chunks, applies f, concatenates."""

    def test_no_split_when_axis_shorter_than_bs(self):
        # axis width (4) < bs (100): f is applied once to the whole input.
        x = paddle.arange(4 * 3, dtype="float32").reshape([4, 3])
        wrapped = subbatch(
            lambda t: t * 2.0 + 1.0, arg_idx=[0], axis=[0], bs=100, out_idx=0
        )
        out = wrapped(x)
        np.testing.assert_array_equal(out.numpy(), x.numpy() * 2.0 + 1.0)

    def test_split_preserves_order_and_handles_uneven_last_chunk(self):
        # length 10 with bs=3 -> chunks of 3,3,3,1; arange content exposes any
        # reordering or a dropped/duplicated chunk.
        x = paddle.arange(10, dtype="float32").reshape([10, 1])
        wrapped = subbatch(
            lambda t: t * 10.0, arg_idx=[0], axis=[0], bs=3, out_idx=0
        )
        out = wrapped(x)
        expected = (np.arange(10, dtype="float32") * 10.0).reshape([10, 1])
        np.testing.assert_array_equal(out.numpy(), expected)
        self.assertEqual(list(out.shape), [10, 1])

    def test_split_and_concat_along_named_out_axis(self):
        # Split along axis 1 (widths 4,4,2) and concat back along axis 1.
        x = paddle.arange(2 * 10, dtype="float32").reshape([2, 10])
        wrapped = subbatch(
            lambda t: t * 3.0, arg_idx=[0], axis=[1], bs=4, out_idx=1
        )
        out = wrapped(x)
        np.testing.assert_array_equal(out.numpy(), x.numpy() * 3.0)

    def test_same_arg_idx_reuses_first_arg_slice(self):
        # arg 1 must reuse arg 0's slice. f returns its second arg, so the output
        # equals arange (arg 0), NOT zeros (arg 1); broken reuse would give zeros.
        a = paddle.arange(6, dtype="float32").reshape([6, 1])
        b = paddle.zeros([6, 1], dtype="float32")
        wrapped = subbatch(
            lambda p, q: q,
            arg_idx=[0],
            axis=[0],
            bs=3,
            out_idx=0,
            same_arg_idx={1: 0},
        )
        out = wrapped(a, b)
        np.testing.assert_array_equal(out.numpy(), a.numpy())
        self.assertFalse(np.allclose(out.numpy(), b.numpy()))

    def test_mismatched_axis_width_raises(self):
        wrapped = subbatch(
            lambda p, q: p, arg_idx=[0, 1], axis=[0, 0], bs=3, out_idx=0
        )
        a = paddle.ones([6, 2], dtype="float32")
        b = paddle.ones([8, 2], dtype="float32")
        with self.assertRaises(AssertionError):
            wrapped(a, b)

    def test_same_arg_idx_forward_reference_raises(self):
        # same_arg_idx requires i > same_arg_idx[i]; {0: 1} violates it.
        wrapped = subbatch(
            lambda p, q: p,
            arg_idx=[0, 1],
            axis=[0, 0],
            bs=3,
            out_idx=0,
            same_arg_idx={0: 1},
        )
        a = paddle.ones([6, 1], dtype="float32")
        b = paddle.ones([6, 1], dtype="float32")
        with self.assertRaises(AssertionError):
            wrapped(a, b)

    def test_recompute_branch_matches_reference_and_flows_grad(self):
        # use_recompute=True routes each chunk through fleet recompute; forward
        # must equal the direct result and gradients must reach the input.
        x = paddle.arange(9, dtype="float32").reshape([9, 1])
        x.stop_gradient = False
        wrapped = subbatch(
            lambda t: t * 2.0,
            arg_idx=[0],
            axis=[0],
            bs=3,
            out_idx=0,
            use_recompute=True,
        )
        out = wrapped(x)
        np.testing.assert_array_equal(out.numpy(), x.numpy() * 2.0)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        np.testing.assert_array_equal(
            x.grad.numpy(), np.full([9, 1], 2.0, dtype="float32")
        )


def _loss_self(use_filtered_label_loss, return_tuple=True, training=True):
    """Minimal stand-in for the criterion layer bound as `self`.

    sft_postprocess_loss only reads these four attributes; using a real
    namespace (not a mock) keeps the production reduction/masking logic intact.
    """
    return SimpleNamespace(
        use_filtered_label_loss=use_filtered_label_loss,
        ignored_index=IGNORE_INDEX,
        return_tuple=return_tuple,
        training=training,
    )


class TestSftPostprocessLoss(unittest.TestCase):
    """sft_postprocess_loss: valid-token masking + token-averaged reduction."""

    def test_ignored_tokens_excluded_from_numerator_and_denominator(self):
        # Per-token losses 1,2,3,4; positions 1 and 3 are ignored (-100).
        # valid loss sum = 1 + 3 = 4 ; valid count = 2 ; loss = 4 / 2 = 2.0.
        # A missing mask would give (1+2+3+4)/4 = 2.5, which this rejects.
        per_token = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0]], dtype="float32")
        labels = paddle.to_tensor(
            [[5, IGNORE_INDEX, 7, IGNORE_INDEX]], dtype="int64"
        )
        loss, loss_sum = sft_postprocess_loss(
            _loss_self(use_filtered_label_loss=True), per_token, labels, None
        )
        self.assertAlmostEqual(float(loss.numpy()), 2.0, places=6)
        self.assertAlmostEqual(float(loss_sum.numpy()), 4.0, places=6)

    def test_explicit_loss_mask_governs_when_not_filtered(self):
        # use_filtered_label_loss=False and an explicit mask -> labels are NOT
        # consulted. Mask keeps positions 0 and 1: loss = (1+2)/2 = 1.5.
        per_token = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0]], dtype="float32")
        labels = paddle.to_tensor([[5, 6, 7, 8]], dtype="int64")  # all "valid"
        loss_mask = paddle.to_tensor([[1.0, 1.0, 0.0, 0.0]], dtype="float32")
        loss, loss_sum = sft_postprocess_loss(
            _loss_self(use_filtered_label_loss=False),
            per_token,
            labels,
            loss_mask,
        )
        self.assertAlmostEqual(float(loss.numpy()), 1.5, places=6)
        self.assertAlmostEqual(float(loss_sum.numpy()), 3.0, places=6)

    def test_filtered_label_loss_overrides_supplied_mask(self):
        # With use_filtered_label_loss=True the supplied (all-ones) mask is
        # discarded and the mask is recomputed from ignored labels.
        # labels-derived mask -> (3 + 6)/2 = 4.5 ; honouring the all-ones mask
        # would give (3 + 9 + 6)/3 = 6.0, so 4.5 proves the override happened.
        per_token = paddle.to_tensor([[3.0, 9.0, 6.0]], dtype="float32")
        labels = paddle.to_tensor([[9, IGNORE_INDEX, 11]], dtype="int64")
        supplied = paddle.to_tensor([[1.0, 1.0, 1.0]], dtype="float32")
        loss, loss_sum = sft_postprocess_loss(
            _loss_self(use_filtered_label_loss=True),
            per_token,
            labels,
            supplied,
        )
        self.assertAlmostEqual(float(loss.numpy()), 4.5, places=6)
        self.assertAlmostEqual(float(loss_sum.numpy()), 9.0, places=6)

    def test_pp_path_returns_scalar_by_training_flag(self):
        # return_tuple=False (pipeline path): training -> mean loss; eval -> sum.
        per_token = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0]], dtype="float32")
        labels = paddle.to_tensor(
            [[5, IGNORE_INDEX, 7, IGNORE_INDEX]], dtype="int64"
        )
        train_loss = sft_postprocess_loss(
            _loss_self(
                use_filtered_label_loss=True, return_tuple=False, training=True
            ),
            per_token,
            labels,
            None,
        )
        eval_loss = sft_postprocess_loss(
            _loss_self(
                use_filtered_label_loss=True, return_tuple=False, training=False
            ),
            per_token,
            labels,
            None,
        )
        self.assertNotIsInstance(train_loss, tuple)
        self.assertNotIsInstance(eval_loss, tuple)
        self.assertAlmostEqual(float(train_loss.numpy()), 2.0, places=6)  # 4/2
        self.assertAlmostEqual(float(eval_loss.numpy()), 4.0, places=6)  # sum


class TestLossImpl(unittest.TestCase):
    """loss_impl casts logits to float32 then delegates to self.loss_func."""

    def test_casts_to_float32_and_passes_through_result(self):
        captured = {}
        sentinel = paddle.to_tensor([7.0, 8.0], dtype="float32")

        def fake_loss_func(logits, labels):
            captured["dtype"] = logits.dtype
            captured["labels"] = labels
            return sentinel

        obj = SimpleNamespace(loss_func=fake_loss_func)
        logits = paddle.ones([2, 3], dtype="float16")
        labels = paddle.to_tensor([1, 2], dtype="int64")
        out = loss_impl(obj, logits, labels)
        self.assertEqual(captured["dtype"], paddle.float32)
        self.assertIs(captured["labels"], labels)
        np.testing.assert_array_equal(out.numpy(), sentinel.numpy())


class TestCalcLmHeadLogits(unittest.TestCase):
    """calc_lm_head_logits with tp_size==1 reduces to hidden @ weight.T (+bias)."""

    def _config(self, **kw):
        return SimpleNamespace(
            sequence_parallel=kw.get("sequence_parallel", False),
            tensor_parallel_output=kw.get("tensor_parallel_output", False),
            tensor_model_parallel_size=kw.get("tensor_model_parallel_size", 1),
            max_sequence_length=kw.get("max_sequence_length", 8),
        )

    def test_matches_independent_matmul_reference(self):
        hidden = (
            paddle.arange(2 * 3 * 4, dtype="float32").reshape([2, 3, 4]) * 0.01
        )
        weight = (
            paddle.arange(5 * 4, dtype="float32").reshape([5, 4]) * 0.02
        )  # [vocab, h]
        logits = calc_lm_head_logits(self._config(), hidden, weight, None)
        # parallel_matmul uses transpose_y=True: logits = hidden @ weight.T.
        expected = np.matmul(hidden.numpy(), weight.numpy().T)
        self.assertEqual(list(logits.shape), [2, 3, 5])
        np.testing.assert_allclose(
            logits.numpy(), expected, rtol=1e-5, atol=1e-6
        )

    def test_bias_is_added_after_matmul(self):
        hidden = (
            paddle.arange(1 * 2 * 4, dtype="float32").reshape([1, 2, 4]) * 0.1
        )
        weight = paddle.arange(3 * 4, dtype="float32").reshape([3, 4]) * 0.05
        bias = paddle.to_tensor([10.0, 20.0, 30.0], dtype="float32")
        with_bias = calc_lm_head_logits(self._config(), hidden, weight, bias)
        without_bias = calc_lm_head_logits(self._config(), hidden, weight, None)
        # The only difference must be the broadcast bias on the vocab axis.
        np.testing.assert_allclose(
            (with_bias - without_bias).numpy(),
            np.broadcast_to(bias.numpy(), with_bias.shape),
            rtol=1e-5,
            atol=1e-6,
        )


@unittest.skip(
    "Tensor-parallel / sequence-parallel numeric paths of calc_lm_head_logits "
    "and parallel_matmul require a real multi-rank process group; the "
    "single-rank CPU path is covered above. Not faked here to avoid presenting "
    "single-process results as multi-card evidence."
)
class TestCalcLmHeadLogitsTensorParallel(unittest.TestCase):
    def test_tensor_parallel_gather_and_split(self):
        raise AssertionError("placeholder for multi-card verification")


if __name__ == "__main__":
    unittest.main()
