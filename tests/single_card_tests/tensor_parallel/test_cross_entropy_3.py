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

"""Behavior tests for vocab-parallel cross entropy, part 3: label smoothing.

Module under test: ``paddlefleet.tensor_parallel.cross_entropy``. In the
repository module map this is the "分布式训练" boundary (tensor parallel loss).

This file deliberately targets branches that a plain base / _2 file does NOT
exercise:

  * the ``label_smoothing > 0`` FORWARD branch of
    ``_VocabParallelCrossEntropy.forward`` (the smoothed-loss mixing formula),
  * the ``label_smoothing > 0`` BACKWARD branch of
    ``_VocabParallelCrossEntropy.backward`` (uniform mass subtracted from every
    class plus the reduced target correction),
  * the partition-offset + out-of-range MASKING path of
    ``VocabParallelCrossEntropy.calculate_predicted_logits`` with a non-zero
    ``vocab_start_index`` and a target vector mixing in-range / out-of-range
    ids.

Scope / environment: single-card, no tensor-parallel group. The tensor-model
parallel group/rank/world-size are genuine not-under-test collaborators
(distributed topology); they are patched so the code takes its real
world-size==1 local path (``tp_group is None``: no ``all_reduce``). This
validates ONLY the local single-rank numerics; the cross-rank ``all_reduce``
reductions are NOT exercised here and require a real process group.

All expected values are hand-derived with NumPy from the raw inputs; no
expected value is produced by calling the code under test.
"""

import unittest
import unittest.mock

try:
    import numpy as np
    import paddle

    from paddlefleet.tensor_parallel.cross_entropy import (
        VocabParallelCrossEntropy,
        _VocabParallelCrossEntropy,
    )

    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False


# Import path used only when Paddle is available; referenced by @patch strings.
_MOD = "paddlefleet.tensor_parallel.cross_entropy"


def _softmax_np(logits_row):
    """Numerically-stable softmax over a 1-D numpy row (independent ref)."""
    shifted = logits_row - logits_row.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


@unittest.skipUnless(HAS_PADDLE, "paddle is not installed in this environment")
class TestLabelSmoothingForward(unittest.TestCase):
    """Forward ``label_smoothing > 0`` branch of _VocabParallelCrossEntropy."""

    def _run_forward(self, logits, target, label_smoothing):
        # Patch the distributed topology collaborators so the real code takes
        # the tp_group is None (world-size 1) local path. These are not the
        # unit under test; the loss math stays real.
        with (
            unittest.mock.patch(
                f"{_MOD}.get_tensor_model_parallel_group", return_value=None
            ),
            unittest.mock.patch(
                f"{_MOD}.get_tensor_model_parallel_rank", return_value=0
            ),
            unittest.mock.patch(
                f"{_MOD}.get_tensor_model_parallel_world_size", return_value=1
            ),
        ):
            return _VocabParallelCrossEntropy.apply(
                logits, target, label_smoothing
            )

    def test_forward_label_smoothing_matches_hand_derived(self):
        """Smoothed loss = (1-s)*ce - s*mean(log softmax), s = ls*V/(V-1)."""
        logits_np = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        target_idx = 0
        label_smoothing = 0.2

        logits = paddle.to_tensor(logits_np, dtype=paddle.float32)
        target = paddle.to_tensor([target_idx], dtype=paddle.int64)

        loss = self._run_forward(logits, target, label_smoothing)

        # Independent reference.
        row = logits_np[0]
        vocab_size = row.shape[0]
        logsumexp = np.log(np.exp(row - row.max()).sum()) + row.max()
        plain_ce = logsumexp - row[target_idx]  # standard cross entropy
        smoothing = label_smoothing * vocab_size / (vocab_size - 1)
        mean_log_probs = np.log(_softmax_np(row)).mean()
        expected = (1.0 - smoothing) * plain_ce - smoothing * mean_log_probs

        np.testing.assert_allclose(
            loss.numpy().reshape(-1)[0], expected, rtol=1e-5, atol=1e-6
        )

    def test_smoothing_actually_changes_loss(self):
        """Guard: the smoothing branch is consumed, not a no-op.

        With ls=0.2 the returned loss must differ from the plain cross entropy
        (ls=0.0). If the ``label_smoothing > 0`` branch were skipped, the two
        would be equal.
        """
        logits_np = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        target = paddle.to_tensor([0], dtype=paddle.int64)

        loss_plain = self._run_forward(
            paddle.to_tensor(logits_np, dtype=paddle.float32), target, 0.0
        )
        loss_smoothed = self._run_forward(
            paddle.to_tensor(logits_np, dtype=paddle.float32), target, 0.2
        )

        row = logits_np[0]
        plain_ce = np.log(np.exp(row - row.max()).sum()) + row.max() - row[0]
        # Plain path equals standard CE ...
        np.testing.assert_allclose(
            loss_plain.numpy().reshape(-1)[0], plain_ce, rtol=1e-5, atol=1e-6
        )
        # ... and the smoothed path is measurably different.
        self.assertGreater(
            abs(
                float(loss_smoothed.numpy().reshape(-1)[0])
                - float(loss_plain.numpy().reshape(-1)[0])
            ),
            1e-3,
        )


@unittest.skipUnless(HAS_PADDLE, "paddle is not installed in this environment")
class TestLabelSmoothingBackward(unittest.TestCase):
    """Backward ``label_smoothing > 0`` branch of _VocabParallelCrossEntropy."""

    def test_backward_label_smoothing_matches_hand_derived_grad(self):
        """d/dlogits = softmax - s/V  (all) ; target also -= (1 - s).

        With grad_output = 1 (single-element loss summed), the gradient of the
        smoothed loss w.r.t. the input logits, for an in-range target, is:
            g[i]      = softmax[i] - s/V              for every class i
            g[target] = softmax[target] - (1 - s) - s/V
        where s = label_smoothing * V / (V - 1).
        """
        logits_np = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        target_idx = 0
        label_smoothing = 0.2

        logits = paddle.to_tensor(
            logits_np, dtype=paddle.float32, stop_gradient=False
        )
        target = paddle.to_tensor([target_idx], dtype=paddle.int64)

        with (
            unittest.mock.patch(
                f"{_MOD}.get_tensor_model_parallel_group", return_value=None
            ),
            unittest.mock.patch(
                f"{_MOD}.get_tensor_model_parallel_rank", return_value=0
            ),
            unittest.mock.patch(
                f"{_MOD}.get_tensor_model_parallel_world_size", return_value=1
            ),
        ):
            loss = _VocabParallelCrossEntropy.apply(
                logits, target, label_smoothing
            )
            loss.sum().backward()

        self.assertIsNotNone(logits.grad)

        # Independent reference gradient.
        row = logits_np[0]
        vocab_size = row.shape[0]
        softmax = _softmax_np(row)
        smoothing = label_smoothing * vocab_size / (vocab_size - 1)
        expected_grad = softmax - smoothing / vocab_size
        expected_grad[target_idx] -= 1.0 - smoothing

        np.testing.assert_allclose(
            logits.grad.numpy().reshape(-1),
            expected_grad,
            rtol=1e-5,
            atol=1e-6,
        )
        # Gradient of cross entropy (even smoothed) sums to zero over classes.
        self.assertAlmostEqual(float(logits.grad.numpy().sum()), 0.0, places=5)

    def test_smoothed_grad_differs_from_plain_grad(self):
        """Guard: the smoothing backward branch is taken, not the plain one.

        The plain (ls=0) gradient is softmax with 1 subtracted at the target.
        The smoothed gradient additionally removes uniform mass s/V from every
        class and reduces the target correction to (1 - s), so the two must
        differ.
        """
        logits_np = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        target_idx = 0
        row = logits_np[0]
        vocab_size = row.shape[0]
        softmax = _softmax_np(row)

        plain_grad = softmax.copy()
        plain_grad[target_idx] -= 1.0

        smoothing = 0.2 * vocab_size / (vocab_size - 1)
        smoothed_grad = softmax - smoothing / vocab_size
        smoothed_grad[target_idx] -= 1.0 - smoothing

        # Now drive the real code with ls=0.2 and confirm it matches the
        # smoothed reference (and therefore is NOT the plain gradient).
        logits = paddle.to_tensor(
            logits_np, dtype=paddle.float32, stop_gradient=False
        )
        target = paddle.to_tensor([target_idx], dtype=paddle.int64)
        with (
            unittest.mock.patch(
                f"{_MOD}.get_tensor_model_parallel_group", return_value=None
            ),
            unittest.mock.patch(
                f"{_MOD}.get_tensor_model_parallel_rank", return_value=0
            ),
            unittest.mock.patch(
                f"{_MOD}.get_tensor_model_parallel_world_size", return_value=1
            ),
        ):
            loss = _VocabParallelCrossEntropy.apply(logits, target, 0.2)
            loss.sum().backward()

        actual = logits.grad.numpy().reshape(-1)
        np.testing.assert_allclose(actual, smoothed_grad, rtol=1e-5, atol=1e-6)
        self.assertGreater(float(np.abs(actual - plain_grad).max()), 1e-3)


@unittest.skipUnless(HAS_PADDLE, "paddle is not installed in this environment")
class TestCalculatePredictedLogitsPartitionMasking(unittest.TestCase):
    """calculate_predicted_logits: non-zero partition offset + range masking."""

    def test_partition_offset_and_out_of_range_masking(self):
        """Target ids are shifted by vocab_start_index; out-of-range -> masked.

        Partition owns vocab ids [2, 5). masked_target = target - 2 for
        in-range ids; out-of-range ids are masked (target_mask True), their
        masked_target forced to 0, and their predicted logit forced to 0.0.
        logits_max is passed as zeros so the in-place shift is a no-op and the
        gathered predicted logit equals logits[row, masked_target] directly.
        """
        vocab_start_index = 2
        vocab_end_index = 5  # partition holds 3 columns
        logits_np = np.array(
            [
                [10.0, 11.0, 12.0],  # row 0
                [20.0, 21.0, 22.0],  # row 1
                [30.0, 31.0, 32.0],  # row 2
            ],
            dtype=np.float32,
        )
        # ids: 3 (in-range -> col 1), 0 (out-of-range), 4 (in-range -> col 2)
        target_np = np.array([3, 0, 4], dtype=np.int64)

        logits = paddle.to_tensor(logits_np, dtype=paddle.float32)
        logits_max = paddle.zeros([3], dtype=paddle.float32)
        target = paddle.to_tensor(target_np, dtype=paddle.int64)

        (
            target_mask,
            masked_target_1d,
            predicted_logits,
            sum_exp_logits,
            exp_logits,
        ) = VocabParallelCrossEntropy.calculate_predicted_logits(
            logits,
            target,
            logits_max,
            vocab_start_index,
            vocab_end_index,
        )

        # Independent expectations.
        expected_mask = [False, True, False]
        expected_masked_target = [1, 0, 2]  # (3-2), forced 0, (4-2)
        expected_predicted = [
            logits_np[0, 1],  # in-range gather
            0.0,  # masked -> forced to 0.0
            logits_np[2, 2],  # in-range gather
        ]
        expected_sum_exp = np.exp(logits_np).sum(axis=-1)

        self.assertEqual(
            [bool(v) for v in target_mask.numpy().reshape(-1)], expected_mask
        )
        self.assertEqual(
            [int(v) for v in masked_target_1d.numpy().reshape(-1)],
            expected_masked_target,
        )
        np.testing.assert_allclose(
            predicted_logits.numpy().reshape(-1),
            np.array(expected_predicted, dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            sum_exp_logits.numpy().reshape(-1),
            expected_sum_exp,
            rtol=1e-5,
            atol=1e-4,
        )
        # exp_logits are the per-element exponentials of the (unshifted) logits.
        np.testing.assert_allclose(
            exp_logits.numpy(),
            np.exp(logits_np),
            rtol=1e-5,
            atol=1e-3,
        )


if __name__ == "__main__":
    unittest.main()
