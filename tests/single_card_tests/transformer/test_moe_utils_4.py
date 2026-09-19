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

"""Behaviour tests for paddlefleet.transformer.moe.moe_utils.

These exercise the default (throughput) token permute/unpermute path plus the
``AddAuxiliaryLoss`` / ``RandomSTE`` autograd tricks. Expected values are
derived by hand with an independent NumPy reference (the expert-major grouping
order, the scatter-add combine, the injected unit gradient and the
straight-through zero gradient), never by calling the functions under test to
produce their own oracle.

Heavy imports are guarded: on a host without paddle/paddlefleet the whole
module skips with an honest reason instead of erroring at collection time.
"""

import unittest
from unittest.mock import patch

import numpy as np

try:
    import paddle

    from paddlefleet.transformer.moe.moe_utils import (
        AddAuxiliaryLoss,
        RandomSTE,
        apply_random_logits,
        permute,
        unpermute,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    paddle = None
    _IMPORT_ERROR = exc


requires_paddle = unittest.skipUnless(
    paddle is not None,
    f"paddle/paddlefleet import unavailable: {_IMPORT_ERROR}",
)


@requires_paddle
class TestPermute(unittest.TestCase):
    """``permute`` groups token rows by their designated expert."""

    def test_groups_tokens_by_expert_in_ascending_order(self):
        # routing_map[t, e] == 1 means token t is dispatched to expert e.
        # Expert-major flattening of the (expert, token) mask gives the
        # order in which rows are emitted:
        #   expert0: tokens 0, 2   expert1: tokens 1, 2   expert2: tokens 0, 3
        tokens_np = np.arange(4 * 3, dtype=np.float32).reshape([4, 3])
        routing_np = np.array(
            [[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.float32
        )
        expected_sorted = [0, 2, 1, 2, 0, 3]
        expected_permuted = tokens_np[expected_sorted]

        tokens = paddle.to_tensor(tokens_np)
        routing_map = paddle.to_tensor(routing_np)
        permuted, sorted_indices = permute(tokens, routing_map)

        self.assertEqual(sorted_indices.numpy().tolist(), expected_sorted)
        np.testing.assert_array_equal(permuted.numpy(), expected_permuted)

    def test_single_expert_preserves_original_order(self):
        # Every token routes to expert 0 only: the emitted order is the
        # identity and the permuted rows equal the inputs verbatim.
        tokens_np = np.arange(4 * 2, dtype=np.float32).reshape([4, 2])
        routing_np = np.array(
            [[1, 0], [1, 0], [1, 0], [1, 0]], dtype=np.float32
        )

        tokens = paddle.to_tensor(tokens_np)
        routing_map = paddle.to_tensor(routing_np)
        permuted, sorted_indices = permute(tokens, routing_map)

        self.assertEqual(sorted_indices.numpy().tolist(), [0, 1, 2, 3])
        np.testing.assert_array_equal(permuted.numpy(), tokens_np)

    def test_backward_accumulates_grad_per_selection(self):
        # A token selected by k experts appears k times in the permuted
        # output, so its gradient is the sum of the upstream rows that
        # gathered it (index_select backward = scatter-add).
        tokens_np = np.arange(4 * 2, dtype=np.float32).reshape([4, 2])
        routing_np = np.array(
            [[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.float32
        )
        # sorted order -> [0, 2, 1, 2, 0, 3]
        upstream_np = np.arange(6 * 2, dtype=np.float32).reshape([6, 2])
        expected_grad = np.stack(
            [
                upstream_np[0] + upstream_np[4],  # token 0: rows 0, 4
                upstream_np[2],  # token 1: row 2
                upstream_np[1] + upstream_np[3],  # token 2: rows 1, 3
                upstream_np[5],  # token 3: row 5
            ]
        )

        tokens = paddle.to_tensor(tokens_np)
        tokens.stop_gradient = False
        routing_map = paddle.to_tensor(routing_np)
        permuted, _ = permute(tokens, routing_map)
        permuted.backward(paddle.to_tensor(upstream_np))

        self.assertIsNotNone(tokens.grad)
        np.testing.assert_array_equal(tokens.grad.numpy(), expected_grad)

    def test_drop_and_pad_rejected(self):
        tokens = paddle.zeros([2, 2], dtype="float32")
        routing_map = paddle.to_tensor([[1, 0], [0, 1]], dtype="float32")
        with self.assertRaises(AssertionError):
            permute(tokens, routing_map, drop_and_pad=True)


@requires_paddle
class TestUnpermute(unittest.TestCase):
    """``unpermute`` scatter-adds permuted rows back to token positions."""

    def test_scatter_add_without_probs(self):
        # sorted_indices maps each permuted row to its destination token;
        # duplicate destinations accumulate (overwrite=False onto a zero
        # buffer).
        permuted_np = np.array(
            [[1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.float32
        )
        sorted_indices = [0, 2, 0, 1]
        # token0 <- rows 0 + 2, token1 <- row 3, token2 <- row 1
        expected = np.array([[4, 4], [4, 4], [2, 2]], dtype=np.float32)

        output = unpermute(
            paddle.to_tensor(permuted_np),
            paddle.to_tensor(sorted_indices, dtype="int64"),
            [3, 2],
        )
        np.testing.assert_array_equal(output.numpy(), expected)

    def test_probs_are_applied_before_scatter(self):
        # Full expert-major layout for routing:
        #   token0 -> experts 0,1 ; token1 -> expert0 ; token2 -> expert1
        # permute would emit rows in order (t0e0, t1e0, t0e1, t2e1), i.e.
        # sorted_indices == [0, 1, 0, 2]. probs.T masked-select follows the
        # same order, giving weights [0.5, 0.25, 2.0, 4.0].
        permuted_np = np.array(
            [[10, 20], [30, 40], [50, 60], [70, 80]], dtype=np.float32
        )
        sorted_indices = [0, 1, 0, 2]
        probs_np = np.array(
            [[0.5, 2.0], [0.25, 0.0], [0.0, 4.0]], dtype=np.float32
        )
        routing_np = np.array([[1, 1], [1, 0], [0, 1]], dtype=np.float32)
        weights = np.array([0.5, 0.25, 2.0, 4.0], dtype=np.float32)
        weighted = permuted_np * weights[:, None]
        expected = np.stack(
            [
                weighted[0] + weighted[2],  # token0 <- rows 0, 2
                weighted[1],  # token1 <- row 1
                weighted[3],  # token2 <- row 3
            ]
        )

        output = unpermute(
            paddle.to_tensor(permuted_np),
            paddle.to_tensor(sorted_indices, dtype="int64"),
            [3, 2],
            probs=paddle.to_tensor(probs_np),
            routing_map=paddle.to_tensor(routing_np),
        )
        np.testing.assert_allclose(output.numpy(), expected, rtol=1e-6)

    def test_probs_require_routing_map(self):
        permuted = paddle.zeros([2, 2], dtype="float32")
        sorted_indices = paddle.to_tensor([0, 1], dtype="int64")
        probs = paddle.to_tensor([[0.5], [0.5]], dtype="float32")
        with self.assertRaises(AssertionError):
            unpermute(permuted, sorted_indices, [2, 2], probs=probs)

    def test_drop_and_pad_rejected(self):
        permuted = paddle.zeros([2, 2], dtype="float32")
        sorted_indices = paddle.to_tensor([0, 1], dtype="int64")
        with self.assertRaises(AssertionError):
            unpermute(permuted, sorted_indices, [2, 2], drop_and_pad=True)


@requires_paddle
class TestPermuteUnpermuteRoundTrip(unittest.TestCase):
    """permute followed by unpermute (no probs) is the identity."""

    def test_single_route_round_trip_restores_tokens(self):
        # Each token routes to exactly one expert but experts interleave,
        # so the permuted order is a genuine reordering ([1, 3, 0, 2]) that
        # unpermute must undo exactly.
        tokens_np = np.arange(4 * 2, dtype=np.float32).reshape([4, 2]) + 1.0
        routing_np = np.array(
            [[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.float32
        )

        tokens = paddle.to_tensor(tokens_np)
        routing_map = paddle.to_tensor(routing_np)
        permuted, sorted_indices = permute(tokens, routing_map)

        self.assertEqual(sorted_indices.numpy().tolist(), [1, 3, 0, 2])
        restored = unpermute(permuted, sorted_indices, [4, 2])
        np.testing.assert_array_equal(restored.numpy(), tokens_np)


@requires_paddle
class TestAddAuxiliaryLoss(unittest.TestCase):
    """The aux-loss trick leaves the forward untouched but injects grad=1."""

    def test_forward_returns_value_of_x_independent_of_loss(self):
        x_np = np.arange(4 * 2, dtype=np.float32).reshape([4, 2])
        x = paddle.to_tensor(x_np)
        # A large loss value must not leak into the forward output.
        loss = paddle.to_tensor([123.0])
        out = AddAuxiliaryLoss.apply(x, loss)

        self.assertIsNot(out, x)  # x.clone(), not the same tensor
        np.testing.assert_array_equal(out.numpy(), x_np)

    def test_backward_injects_unit_grad_into_required_loss(self):
        x_np = np.arange(4 * 2, dtype=np.float32).reshape([4, 2])
        x = paddle.to_tensor(x_np)
        x.stop_gradient = False
        loss = paddle.to_tensor([0.5])
        loss.stop_gradient = False

        out = AddAuxiliaryLoss.apply(x, loss)
        out.sum().backward()

        # forward is a clone -> x receives the upstream grad (all ones).
        np.testing.assert_array_equal(x.grad.numpy(), np.ones_like(x_np))
        # the aux loss receives a hard-coded unit gradient.
        self.assertIsNotNone(loss.grad)
        np.testing.assert_array_equal(loss.grad.numpy(), np.array([1.0]))

    def test_backward_skips_grad_for_detached_loss(self):
        x_np = np.arange(4 * 2, dtype=np.float32).reshape([4, 2])
        x = paddle.to_tensor(x_np)
        x.stop_gradient = False
        loss = paddle.to_tensor([0.5])
        loss.stop_gradient = True

        out = AddAuxiliaryLoss.apply(x, loss)
        out.sum().backward()

        np.testing.assert_array_equal(x.grad.numpy(), np.ones_like(x_np))
        self.assertIsNone(loss.grad)

    def test_forward_rejects_non_scalar_loss(self):
        x = paddle.zeros([2, 2], dtype="float32")
        loss = paddle.to_tensor([0.5, 0.5])  # numel != 1
        with self.assertRaises(AssertionError):
            AddAuxiliaryLoss.apply(x, loss)


@requires_paddle
class TestRandomSTE(unittest.TestCase):
    """RandomSTE replaces the forward with noise and zeroes the gradient."""

    def test_forward_ignores_input_values(self):
        # Under the single-process branch the output is paddle.randn(shape)
        # cast to the input dtype: it depends on shape and the RNG, never on
        # the input content. Same seed + same shape -> identical output even
        # when the input values differ.
        shape = [4, 8]
        x1 = paddle.zeros(shape, dtype="float32")
        x2 = paddle.full(shape, 9.0, dtype="float32")

        with patch("paddle.distributed.get_world_size", return_value=1):
            paddle.seed(2024)
            out1 = RandomSTE.apply(x1)
            paddle.seed(2024)
            out2 = RandomSTE.apply(x2)

        self.assertEqual(out1.shape, shape)
        self.assertEqual(out1.dtype, x1.dtype)
        np.testing.assert_array_equal(out1.numpy(), out2.numpy())

    def test_backward_returns_zero_gradient(self):
        x = paddle.randn([4, 8], dtype="float32")
        x.stop_gradient = False

        with patch("paddle.distributed.get_world_size", return_value=1):
            out = RandomSTE.apply(x)
        out.sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)
        np.testing.assert_array_equal(
            x.grad.numpy(), np.zeros([4, 8], dtype=np.float32)
        )


@requires_paddle
class TestApplyRandomLogits(unittest.TestCase):
    """apply_random_logits is a thin wrapper over RandomSTE.apply."""

    def test_delegates_to_random_ste(self):
        logits = paddle.arange(4 * 8, dtype="float32").reshape([4, 8])

        with patch("paddle.distributed.get_world_size", return_value=1):
            paddle.seed(7)
            via_wrapper = apply_random_logits(logits)
            paddle.seed(7)
            via_layer = RandomSTE.apply(logits)

        self.assertEqual(via_wrapper.shape, logits.shape)
        self.assertEqual(via_wrapper.dtype, logits.dtype)
        np.testing.assert_array_equal(via_wrapper.numpy(), via_layer.numpy())


if __name__ == "__main__":
    unittest.main()
