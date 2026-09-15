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

"""Behavior tests for vocab-parallel cross entropy static-method math.

Module under test:
``paddlefleet.tensor_parallel.cross_entropy.VocabParallelCrossEntropy``.
In the repository module map this is the "分布式训练" (tensor parallel)
boundary. This file (``_2`` of the cross_entropy set) targets the
per-partition numeric building blocks that the autograd Function composes:

    * ``calculate_predicted_logits`` -- in-place max subtraction, out-of-range
      target masking with a non-zero ``vocab_start_index`` offset, gather of
      the target logit, and the exp / sum-exp reduction.
    * ``calculate_cross_entropy_loss`` -- ``log(sum_exp) - predicted`` loss and
      the in-place softmax normalization of ``exp_logits``.
    * ``prepare_gradient_calculation_operands`` + ``calculate_gradients`` -- the
      backward-pass gradient assembly ``softmax_update = 1 - mask`` and the
      ``grad[arange, target] -= softmax_update`` / upstream-broadcast product.

Each expectation is hand-derived from small, position-distinguishable inputs
so that swapped masks, wrong offsets, a dropped subtraction, or a broken
upstream broadcast are all rejected. These are single-rank (world_size == 1)
CPU-executable paths; true cross-rank all-reduce is out of scope here.

paddle is not installed in this environment, so the whole suite is skipped via
``@unittest.skipUnless`` rather than faking a pass.
"""

import unittest

import numpy as np

try:
    import paddle

    from paddlefleet.tensor_parallel.cross_entropy import (
        VocabParallelCrossEntropy,
    )

    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False


@unittest.skipUnless(HAS_PADDLE, "paddle is not installed in this environment")
class TestCalculatePredictedLogits(unittest.TestCase):
    """VocabParallelCrossEntropy.calculate_predicted_logits numeric behavior."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_in_range_gather_and_sum_exp(self):
        """Full-vocab partition: max-subtract, gather target logit, sum-exp."""
        logits = paddle.to_tensor(
            [[1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 0.0, -1.0]],
            dtype="float32",
        )
        # logits_max mirrors calculate_logits_max output (per-row max).
        logits_max = paddle.to_tensor([4.0, 1.0], dtype="float32")
        target = paddle.to_tensor([[2], [1]], dtype="int64")

        (
            target_mask,
            masked_target_1d,
            predicted_logits,
            sum_exp_logits,
            exp_logits,
        ) = VocabParallelCrossEntropy.calculate_predicted_logits(
            logits, target, logits_max, 0, 4
        )

        # No target is out of [0, 4); nothing is masked.
        self.assertEqual(target_mask.astype("int64").tolist(), [[0], [0]])
        self.assertEqual(masked_target_1d.tolist(), [2, 1])

        # logits were shifted in place by subtracting the per-row max.
        shifted = np.array(
            [[-3.0, -2.0, -1.0, 0.0], [-1.0, 0.0, -1.0, -2.0]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            logits.numpy(), shifted, rtol=1e-6, atol=1e-6
        )

        # predicted = shifted[row, target]: shifted[0,2]=-1, shifted[1,1]=0.
        np.testing.assert_allclose(
            predicted_logits.numpy(),
            np.array([[-1.0], [0.0]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

        expected_exp = np.exp(shifted)
        np.testing.assert_allclose(
            exp_logits.numpy(), expected_exp, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            sum_exp_logits.numpy(),
            expected_exp.sum(axis=-1),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_offset_partition_masks_out_of_range_target(self):
        """Non-zero vocab_start_index: offset local ids and zero masked rows."""
        logits = paddle.to_tensor(
            [[10.0, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0]],
            dtype="float32",
        )
        logits_max = paddle.to_tensor([40.0, 4.0], dtype="float32")
        # Partition owns global vocab ids [4, 8). Row0 target 6 -> local 2
        # (kept); row1 target 0 is below the partition -> masked.
        target = paddle.to_tensor([[6], [0]], dtype="int64")

        (
            target_mask,
            masked_target_1d,
            predicted_logits,
            _sum_exp_logits,
            _exp_logits,
        ) = VocabParallelCrossEntropy.calculate_predicted_logits(
            logits, target, logits_max, 4, 8
        )

        self.assertEqual(target_mask.astype("int64").tolist(), [[0], [1]])
        # Row0: 6 - 4 = 2 (kept). Row1: masked local id is forced to 0.
        self.assertEqual(masked_target_1d.tolist(), [2, 0])

        # shifted row0 = [-30,-20,-10,0]; predicted[0] = shifted[0,2] = -10.
        # Row1 is masked, so its predicted logit is overwritten with 0.0.
        np.testing.assert_allclose(
            predicted_logits.numpy(),
            np.array([[-10.0], [0.0]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )


@unittest.skipUnless(HAS_PADDLE, "paddle is not installed in this environment")
class TestCalculateCrossEntropyLoss(unittest.TestCase):
    """VocabParallelCrossEntropy.calculate_cross_entropy_loss numeric behavior."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_loss_value_and_inplace_softmax(self):
        """loss = log(sum_exp) - predicted; exp_logits normalized in place."""
        exp_logits = paddle.to_tensor(
            [[1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 4.0, 2.0]],
            dtype="float32",
        )
        predicted_logits = paddle.to_tensor([1.5, 3.0], dtype="float32")
        sum_exp_logits = paddle.to_tensor([10.0, 10.0], dtype="float32")

        returned_softmax, loss = (
            VocabParallelCrossEntropy.calculate_cross_entropy_loss(
                exp_logits, predicted_logits, sum_exp_logits
            )
        )

        expected_loss = np.log(np.array([10.0, 10.0])) - np.array([1.5, 3.0])
        np.testing.assert_allclose(
            loss.numpy(), expected_loss, rtol=1e-6, atol=1e-6
        )

        expected_softmax = np.array(
            [[0.1, 0.2, 0.3, 0.4], [0.2, 0.2, 0.4, 0.2]],
            dtype=np.float32,
        )
        # Normalization is in place: the same object is returned and mutated.
        self.assertIs(returned_softmax, exp_logits)
        np.testing.assert_allclose(
            exp_logits.numpy(), expected_softmax, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            exp_logits.sum(axis=-1).numpy(),
            np.ones(2, dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )


@unittest.skipUnless(HAS_PADDLE, "paddle is not installed in this environment")
class TestGradientAssembly(unittest.TestCase):
    """prepare_gradient_calculation_operands + calculate_gradients behavior."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_backward_gradient_pipeline(self):
        """softmax_update = 1 - mask, target-slot decrement, upstream product."""
        softmax = paddle.to_tensor(
            [[0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.25, 0.25]],
            dtype="float32",
        )
        # Row1 target is out-of-partition (masked) -> its slot must NOT be
        # decremented, i.e. softmax_update = 0 for that row.
        target_mask = paddle.to_tensor([False, True], dtype="bool")

        grad_2d, arange_1d, softmax_update, grad_input = (
            VocabParallelCrossEntropy.prepare_gradient_calculation_operands(
                softmax, target_mask
            )
        )

        # grad_input aliases the input softmax storage.
        self.assertIs(grad_input, softmax)
        self.assertEqual(arange_1d.tolist(), [0, 1])
        np.testing.assert_allclose(
            softmax_update.numpy(),
            np.array([1.0, 0.0], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

        masked_target_1d = paddle.to_tensor([2, 0], dtype="int64")
        grad_output = paddle.to_tensor([2.0, 3.0], dtype="float32")

        result = VocabParallelCrossEntropy.calculate_gradients(
            grad_2d,
            arange_1d,
            masked_target_1d,
            softmax_update,
            grad_input,
            grad_output,
        )

        # Step 1: grad[0,2] -= 1.0 -> -0.7; grad[1,0] -= 0.0 (masked, kept).
        # Step 2: row0 *= 2, row1 *= 3.
        expected = np.array(
            [[0.2, 0.4, -1.4, 0.8], [0.75, 0.75, 0.75, 0.75]],
            dtype=np.float32,
        )
        self.assertIs(result, grad_input)
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=1e-6, atol=1e-5
        )
        # In-place: the original softmax tensor now holds the gradient.
        np.testing.assert_allclose(
            softmax.numpy(), expected, rtol=1e-6, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
