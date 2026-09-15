# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Behavior tests for ``paddlefleet.tensor_parallel.cross_entropy``.

Module under test lives in the "分布式训练" module of the repository map. It
implements vocab-parallel cross entropy: the vocab dimension is split across
tensor-parallel ranks, so the loss needs per-partition masking of out-of-range
targets plus cross-rank ``all_reduce`` of ``logits_max``, ``predicted_logits``
and ``sum_exp_logits``.

Scope of THIS file (single card / world-size-1 local path only):

* The stateless numeric helpers in ``VocabParallelCrossEntropy`` are exercised
  with fixed, hand-derived inputs and independently computed expectations
  (masking, predicted-logit selection, loss arithmetic, softmax normalization,
  gradient assembly).
* The public ``vocab_parallel_cross_entropy`` / ``_VocabParallelCrossEntropy``
  forward+backward are driven with the tensor-parallel group forced to ``None``
  (the genuine no-TP collaborator). On that path ``vocab_start == 0`` and
  ``vocab_end == vocab_size``, so the vocab-parallel loss must equal ordinary
  cross entropy ``logsumexp(z) - z[target]`` and its gradient must equal
  ``softmax(z) - onehot(target)``. Both references are computed with plain numpy,
  which shares no code with the module under test. The max-subtraction inside the
  module cancels analytically, so the numpy reference is a valid independent
  anchor.

NOT covered here (requires a real multi-rank process group, see
``tests/multi_card_tests``): the ``all_reduce`` of ``logits_max`` /
``predicted_logits`` / ``sum_exp_logits`` across ranks and the per-rank
``vocab_start``/``vocab_end`` partitioning. Forcing the group to ``None`` only
validates the local, single-partition path; it does not prove cross-rank
reduction.

The module imports ``paddle`` at load time. This environment may have no paddle,
so imports are guarded and the test classes skip honestly rather than fake-pass.
"""

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)
# Also expose the ``src`` layout in case paddlefleet is not installed.
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

import numpy as np

try:
    from unittest import mock

    import paddle

    from paddlefleet.tensor_parallel import cross_entropy as ce_mod
    from paddlefleet.tensor_parallel.cross_entropy import (
        VocabParallelCrossEntropy,
        _VocabParallelCrossEntropy,
        vocab_parallel_cross_entropy,
    )

    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

SKIP_REASON = "paddle is not installed in this environment"


@unittest.skipUnless(HAS_PADDLE, SKIP_REASON)
class TestCalculateLogitsMax(unittest.TestCase):
    """VocabParallelCrossEntropy.calculate_logits_max returns float logits and
    the row-wise max along the vocab dimension."""

    def test_values_and_rowwise_max(self):
        # Integer inputs to also confirm the float() cast happens.
        logits = paddle.to_tensor([[1, 5, 3], [2, 0, -1]], dtype=paddle.int32)
        out_logits, logits_max = VocabParallelCrossEntropy.calculate_logits_max(
            logits
        )

        # Returned logits carry the same numeric content, now as float.
        self.assertIn(out_logits.dtype, (paddle.float32, paddle.float64))
        np.testing.assert_allclose(
            out_logits.numpy(),
            np.array([[1.0, 5.0, 3.0], [2.0, 0.0, -1.0]]),
        )
        # Max is taken over the last (vocab) axis, reducing that dim away.
        self.assertEqual(list(logits_max.shape), [2])
        np.testing.assert_allclose(logits_max.numpy(), np.array([5.0, 2.0]))


@unittest.skipUnless(HAS_PADDLE, SKIP_REASON)
class TestCalculatePredictedLogits(unittest.TestCase):
    """VocabParallelCrossEntropy.calculate_predicted_logits masks out-of-range
    targets, selects the shifted predicted logit, and returns the exp/sum."""

    def test_partition_masking_and_selection(self):
        # This partition owns global vocab ids [2, 4); its local width is 2.
        # Row-unique values make a wrong-column/wrong-row selection visible.
        logits = paddle.to_tensor(
            [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=paddle.float32
        )
        logits_max = paddle.to_tensor([20.0, 40.0, 60.0], dtype=paddle.float32)
        # target 2 -> local col 0 (in range), 3 -> local col 1 (in range),
        # 5 -> out of range (>= vocab_end).
        target = paddle.to_tensor([2, 3, 5], dtype=paddle.int64)

        (
            target_mask,
            masked_target_1d,
            predicted_logits,
            sum_exp_logits,
            exp_logits,
        ) = VocabParallelCrossEntropy.calculate_predicted_logits(
            logits, target, logits_max, vocab_start_index=2, vocab_end_index=4
        )

        self.assertEqual(
            [bool(x) for x in target_mask.numpy().tolist()],
            [False, False, True],
        )
        # target - vocab_start, with the out-of-range entry forced to 0.
        self.assertEqual(masked_target_1d.numpy().tolist(), [0, 1, 0])

        # After the in-place max subtraction each row becomes [-10, 0].
        # Predicted logit = shifted value at the (masked) target column;
        # the out-of-range row is explicitly zeroed.
        np.testing.assert_allclose(
            predicted_logits.numpy(), np.array([-10.0, 0.0, 0.0])
        )

        shifted = np.array([[-10.0, 0.0], [-10.0, 0.0], [-10.0, 0.0]])
        exp_ref = np.exp(shifted)
        np.testing.assert_allclose(exp_logits.numpy(), exp_ref, rtol=1e-6)
        np.testing.assert_allclose(
            sum_exp_logits.numpy(), exp_ref.sum(axis=-1), rtol=1e-6
        )


@unittest.skipUnless(HAS_PADDLE, SKIP_REASON)
class TestCalculateCrossEntropyLoss(unittest.TestCase):
    """VocabParallelCrossEntropy.calculate_cross_entropy_loss computes
    log(sum_exp) - predicted and normalizes exp_logits into a softmax."""

    def test_loss_and_inplace_softmax(self):
        exp_logits = paddle.to_tensor([[1.0, 3.0]], dtype=paddle.float32)
        predicted_logits = paddle.to_tensor([1.0], dtype=paddle.float32)
        sum_exp_logits = paddle.to_tensor([4.0], dtype=paddle.float32)

        softmax, loss = VocabParallelCrossEntropy.calculate_cross_entropy_loss(
            exp_logits, predicted_logits, sum_exp_logits
        )

        np.testing.assert_allclose(
            loss.numpy(), np.array([np.log(4.0) - 1.0]), rtol=1e-6
        )
        # exp_logits divided by sum -> [0.25, 0.75].
        np.testing.assert_allclose(
            softmax.numpy(), np.array([[0.25, 0.75]]), rtol=1e-6
        )
        # The normalization is in-place on the passed tensor.
        self.assertIs(softmax, exp_logits)


@unittest.skipUnless(HAS_PADDLE, SKIP_REASON)
class TestPrepareGradientCalculationOperands(unittest.TestCase):
    """prepare_gradient_calculation_operands exposes softmax as the base grad
    and builds softmax_update = 1 - mask."""

    def test_update_mask_and_identity(self):
        softmax = paddle.to_tensor(
            [[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]], dtype=paddle.float32
        )
        target_mask = paddle.to_tensor([False, True], dtype=paddle.bool)

        grad_2d, arange_1d, softmax_update, grad_input = (
            VocabParallelCrossEntropy.prepare_gradient_calculation_operands(
                softmax, target_mask
            )
        )

        # grad_input is the softmax tensor itself (gradient starts from it).
        self.assertIs(grad_input, softmax)
        # 1.0 where the target is in-range, 0.0 where it was masked out.
        np.testing.assert_allclose(softmax_update.numpy(), np.array([1.0, 0.0]))
        self.assertEqual(arange_1d.numpy().tolist(), [0, 1])
        np.testing.assert_allclose(grad_2d.numpy(), softmax.numpy())


@unittest.skipUnless(HAS_PADDLE, SKIP_REASON)
class TestCalculateGradients(unittest.TestCase):
    """calculate_gradients subtracts the target update in place and scales by
    the upstream gradient."""

    def test_subtract_target_and_scale(self):
        grad_input = paddle.to_tensor(
            [[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]], dtype=paddle.float32
        )
        # grad_2d shares storage with grad_input (same 2-D tensor here).
        grad_2d = grad_input
        arange_1d = paddle.to_tensor([0, 1], dtype=paddle.int64)
        masked_target_1d = paddle.to_tensor([2, 1], dtype=paddle.int64)
        softmax_update = paddle.to_tensor([1.0, 1.0], dtype=paddle.float32)
        grad_output = paddle.to_tensor([2.0, 3.0], dtype=paddle.float32)

        result = VocabParallelCrossEntropy.calculate_gradients(
            grad_2d,
            arange_1d,
            masked_target_1d,
            softmax_update,
            grad_input,
            grad_output,
        )

        # (softmax - onehot(target)) * grad_output, row-wise.
        expected = np.array([[0.4, 0.6, -1.0], [0.3, -1.2, 0.9]])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)


def _no_tp():
    """Patch the tensor-parallel group collaborator to None (the genuine
    single-partition path). Only the distributed group is replaced; the cross
    entropy math stays real."""
    return mock.patch.object(
        ce_mod, "get_tensor_model_parallel_group", return_value=None
    )


def _numpy_logsumexp(z):
    m = z.max(axis=-1, keepdims=True)
    return m[..., 0] + np.log(np.exp(z - m).sum(axis=-1))


@unittest.skipUnless(HAS_PADDLE, SKIP_REASON)
class TestVocabParallelCrossEntropyForward(unittest.TestCase):
    """The public loss on the no-TP path equals ordinary cross entropy."""

    def test_matches_plain_cross_entropy(self):
        z = np.array([[1.0, 2.0, 3.0], [0.5, -0.5, 2.0]], dtype=np.float32)
        target_ids = [2, 0]
        logits = paddle.to_tensor(z, dtype=paddle.float32)
        target = paddle.to_tensor(target_ids, dtype=paddle.int64)

        with _no_tp():
            loss = vocab_parallel_cross_entropy(logits, target)

        # Independent reference: logsumexp(z) - z[target]. The max-subtraction
        # inside the module cancels, so this must match exactly (up to fp).
        ref = _numpy_logsumexp(z) - z[np.arange(len(target_ids)), target_ids]
        np.testing.assert_allclose(loss.numpy(), ref, rtol=1e-5, atol=1e-6)

    def test_3d_input_preserves_leading_dims(self):
        rng = np.random.default_rng(0)
        z = rng.standard_normal((2, 3, 5)).astype(np.float32)
        tgt = rng.integers(0, 5, size=(2, 3)).astype(np.int64)
        logits = paddle.to_tensor(z)
        target = paddle.to_tensor(tgt)

        with _no_tp():
            loss = vocab_parallel_cross_entropy(logits, target)

        self.assertEqual(list(loss.shape), [2, 3])
        flat_z = z.reshape(-1, 5)
        flat_t = tgt.reshape(-1)
        ref = (
            _numpy_logsumexp(flat_z)
            - flat_z[np.arange(flat_t.shape[0]), flat_t]
        ).reshape(2, 3)
        np.testing.assert_allclose(loss.numpy(), ref, rtol=1e-5, atol=1e-6)

    def test_label_smoothing_matches_reference(self):
        z = np.array([[1.0, 2.0, 3.0], [0.5, -0.5, 2.0]], dtype=np.float32)
        target_ids = [2, 0]
        ls = 0.1
        logits = paddle.to_tensor(z, dtype=paddle.float32)
        target = paddle.to_tensor(target_ids, dtype=paddle.int64)

        with _no_tp():
            loss = vocab_parallel_cross_entropy(
                logits, target, label_smoothing=ls
            )

        # Independent reference following the module's smoothing definition:
        #   smoothing = ls * K / (K - 1)
        #   loss = (1 - smoothing) * base_ce - smoothing * mean(log_softmax)
        k = z.shape[-1]
        smoothing = ls * k / (k - 1)
        base = _numpy_logsumexp(z) - z[np.arange(len(target_ids)), target_ids]
        log_softmax = z - _numpy_logsumexp(z)[:, None]
        mean_log = log_softmax.mean(axis=-1)
        ref = (1.0 - smoothing) * base - smoothing * mean_log
        np.testing.assert_allclose(loss.numpy(), ref, rtol=1e-5, atol=1e-6)


@unittest.skipUnless(HAS_PADDLE, SKIP_REASON)
class TestVocabParallelCrossEntropyBackward(unittest.TestCase):
    """The gradient on the no-TP path equals softmax(z) - onehot(target),
    scaled by the upstream gradient."""

    def test_grad_equals_softmax_minus_onehot(self):
        z = np.array([[1.0, 2.0, 3.0], [0.5, -0.5, 2.0]], dtype=np.float32)
        target_ids = [2, 0]
        logits = paddle.to_tensor(z, dtype=paddle.float32)
        logits.stop_gradient = False
        target = paddle.to_tensor(target_ids, dtype=paddle.int64)

        with _no_tp():
            loss = vocab_parallel_cross_entropy(logits, target)
            # sum() gives grad_output = 1 for every row.
            loss.sum().backward()

        self.assertIsNotNone(logits.grad)
        softmax = np.exp(z - _numpy_logsumexp(z)[:, None])
        onehot = np.zeros_like(z)
        onehot[np.arange(len(target_ids)), target_ids] = 1.0
        ref_grad = softmax - onehot
        grad = logits.grad.numpy()
        np.testing.assert_allclose(grad, ref_grad, rtol=1e-4, atol=1e-6)

        # Guard against a scale/sign regression the reference itself can reject.
        self.assertGreater(np.abs(ref_grad).max(), 1e-2)
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(
                grad, 2.0 * ref_grad, rtol=1e-4, atol=1e-6
            )


if __name__ == "__main__":
    unittest.main()
