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

"""Behavior tests for paddlefleet.nn.attention.utils.repeat_kv.

These run on CPU only (no accelerator required); repeat_kv is a pure tensor
reshape/tile with no device-specific numerics.

Contract exercised (derived from the real implementation):
  * Input layout unpacked by the function is [batch, seqlen, num_kv_heads,
    head_dim]; the repeated axis is axis=2 (the third dim).  NOTE: the
    function docstring instead describes a [batch, num_kv_heads, seqlen,
    head_dim] layout and claims equivalence to repeat_interleave(axis=1);
    that docstring is inconsistent with the code, which operates on axis=2.
    The tests below assert the ACTUAL code behavior.
  * n_rep == 1 returns the input object itself (no copy).
  * n_rep > 1 interleaves: output head m maps to input kv head m // n_rep,
    i.e. equivalent to numpy.repeat(x, n_rep, axis=2).  This is the
    grouped/consecutive pattern, NOT a round-robin/block tile.
"""

import unittest

import numpy as np
import paddle

from paddlefleet.nn.attention.utils import repeat_kv


def _interleave_reference(x_np, n_rep):
    """Independent, hand-derived expected output.

    Builds the result by the interleave rule output[..., m, :] =
    input[..., m // n_rep, :] using explicit indexing. This does not call
    the function under test.
    """
    b, s, k, d = x_np.shape
    out = np.empty((b, s, k * n_rep, d), dtype=x_np.dtype)
    for m in range(k * n_rep):
        out[:, :, m, :] = x_np[:, :, m // n_rep, :]
    return out


class TestRepeatKV(unittest.TestCase):
    """Genuine behavior tests for repeat_kv (grouped-query KV expansion)."""

    def test_n_rep_one_returns_input_unchanged(self):
        # Contract: n_rep == 1 short-circuits and returns the same tensor.
        x = paddle.arange(2 * 3 * 4 * 5, dtype="float32").reshape([2, 3, 4, 5])
        result = repeat_kv(x, 1)
        # Identity: no copy is made on the pass-through path.
        self.assertIs(result, x)
        np.testing.assert_array_equal(result.numpy(), x.numpy())

    def test_interleave_grouping_is_consecutive_not_block(self):
        # Distinguishable per-head content so interleave vs block is visible.
        # x has 3 kv heads, each head a distinct row of 4 values.
        x_np = np.arange(1 * 1 * 3 * 4, dtype="float32").reshape([1, 1, 3, 4])
        result = repeat_kv(paddle.to_tensor(x_np), 2).numpy()

        expected = _interleave_reference(x_np, 2)  # heads -> kv [0,0,1,1,2,2]
        self.assertEqual(list(result.shape), [1, 1, 6, 4])
        np.testing.assert_array_equal(result, expected)

        # Per-head identity: output head m must equal input kv head m // 2.
        for m in range(6):
            np.testing.assert_array_equal(
                result[:, :, m, :], x_np[:, :, m // 2, :]
            )

        # It must NOT be the round-robin / block tile pattern
        # (heads -> kv [0,1,2,0,1,2]); that arrangement would silently
        # scramble which kv head each attention head reads.
        block = np.concatenate([x_np, x_np], axis=2)
        self.assertFalse(np.array_equal(result, block))

    def test_matches_numpy_repeat_reference_with_batch_and_seq(self):
        # Fully distinguishable content across batch/seq/kv/dim so any
        # cross-position mixing or axis swap is caught.
        x_np = np.arange(2 * 3 * 2 * 4, dtype="float32").reshape([2, 3, 2, 4])
        result = repeat_kv(paddle.to_tensor(x_np), 3).numpy()

        self.assertEqual(list(result.shape), [2, 3, 6, 4])
        np.testing.assert_array_equal(result, _interleave_reference(x_np, 3))
        # Cross-check against numpy's repeat_interleave semantics on axis=2.
        np.testing.assert_array_equal(result, np.repeat(x_np, 3, axis=2))

    def test_single_kv_head_broadcast_to_all(self):
        # One kv head expanded to n_rep attention heads: every output head
        # must equal the single source head.
        x_np = np.arange(2 * 8 * 1 * 4, dtype="float32").reshape([2, 8, 1, 4])
        result = repeat_kv(paddle.to_tensor(x_np), 4).numpy()

        self.assertEqual(list(result.shape), [2, 8, 4, 4])
        for m in range(4):
            np.testing.assert_array_equal(result[:, :, m, :], x_np[:, :, 0, :])

    def test_dtype_and_values_preserved_for_int(self):
        # Repetition must not alter dtype or values; int payload also checks
        # that no float cast sneaks in.
        x_np = np.arange(1 * 2 * 3 * 4, dtype="int64").reshape([1, 2, 3, 4])
        x = paddle.to_tensor(x_np)
        result = repeat_kv(x, 2)

        self.assertEqual(result.dtype, x.dtype)
        np.testing.assert_array_equal(
            result.numpy(), _interleave_reference(x_np, 2)
        )

    def test_output_is_new_tensor_when_repeating(self):
        # For n_rep > 1 a fresh tensor is produced (contrast with the
        # identity pass-through of n_rep == 1).
        x = paddle.arange(1 * 2 * 3 * 4, dtype="float32").reshape([1, 2, 3, 4])
        result = repeat_kv(x, 2)
        self.assertIsNot(result, x)
        self.assertEqual(list(result.shape), [1, 2, 6, 4])

    def test_backward_accumulates_gradient_per_source_head(self):
        # Each kv head is read by n_rep output heads, so its gradient is the
        # sum of the upstream gradients of those output heads. Independent
        # reference computed by explicit accumulation.
        x_np = np.arange(1 * 1 * 2 * 3, dtype="float32").reshape([1, 1, 2, 3])
        x = paddle.to_tensor(x_np)
        x.stop_gradient = False

        out = repeat_kv(x, 3)  # shape [1, 1, 6, 3]
        # Distinguishable, non-uniform upstream gradient.
        upstream_np = (
            np.arange(1 * 1 * 6 * 3, dtype="float32").reshape([1, 1, 6, 3])
            + 1.0
        )
        out.backward(paddle.to_tensor(upstream_np))

        expected_grad = np.zeros_like(x_np)
        for m in range(6):
            expected_grad[:, :, m // 3, :] += upstream_np[:, :, m, :]

        self.assertIsNotNone(x.grad)
        np.testing.assert_allclose(
            x.grad.numpy(), expected_grad, rtol=1e-6, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
