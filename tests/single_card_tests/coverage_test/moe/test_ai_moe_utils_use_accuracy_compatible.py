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
"""Coverage for the ``use_accuracy_compatible`` permute/unpermute paths added
in commit 80a72f9 (MG-aligned gather/sum permute + unpermute). The aligned
paths must be numerically equivalent to the default paths in both forward and
backward while exercising the dedicated PyLayers."""

import unittest

import numpy as np
import paddle

from paddlefleet.transformer.moe.moe_utils import permute, unpermute


def _make_fixed_topk_routing_map(num_tokens, num_experts, topk, seed=0):
    """Build a [num_tokens, num_experts] 0/1 routing map with a fixed top-k."""
    paddle.seed(seed)
    scores = paddle.randn([num_tokens, num_experts])
    idx = paddle.topk(scores, k=topk, axis=-1).indices
    routing_map = paddle.zeros([num_tokens, num_experts])
    routing_map = routing_map.put_along_axis_(
        idx, paddle.to_tensor(1.0), axis=-1
    )
    return routing_map


def _make_padded_routing_map():
    """Routing map mixing all-zero (padding) rows with fixed top-k=2 rows.

    Row 0 is padding on purpose: the top-k probe must not read it.
    """
    return paddle.to_tensor(
        [
            [0.0, 0.0, 0.0, 0.0],  # padding
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],  # padding
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ],
        dtype="float32",
    )


class TestPermuteAligned(unittest.TestCase):
    def test_permute_forward_matches_default(self):
        rm = _make_fixed_topk_routing_map(6, 4, 2)
        tokens = paddle.randn([6, 8])
        p_a, si_a = permute(tokens, rm, use_accuracy_compatible=True)
        p_s, si_s = permute(tokens, rm, use_accuracy_compatible=False)
        np.testing.assert_allclose(p_a.numpy(), p_s.numpy(), atol=1e-6)
        np.testing.assert_array_equal(si_a.numpy(), si_s.numpy())

    def test_permute_backward_sums_topk_copies(self):
        topk = 2
        rm = _make_fixed_topk_routing_map(5, 3, topk)
        tokens = paddle.randn([5, 4])
        tokens.stop_gradient = False
        permuted, _ = permute(tokens, rm, use_accuracy_compatible=True)
        permuted.sum().backward()
        # Each token is copied to `topk` experts; d(sum)/d(token) == topk.
        self.assertEqual(tokens.grad.shape, [5, 4])
        np.testing.assert_allclose(
            tokens.grad.numpy(),
            np.full([5, 4], float(topk), dtype="float32"),
            atol=1e-6,
        )

    def test_permute_variable_topk_raises(self):
        # token 0 -> 2 experts, token 1 -> 1 expert => not a fixed top-k.
        rm = paddle.to_tensor(
            [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype="float32"
        )
        tokens = paddle.randn([2, 4])
        with self.assertRaises(ValueError):
            permute(tokens, rm, use_accuracy_compatible=True)

    def test_permute_all_padding_tokens(self):
        # No token routed anywhere => topk_val == 0, must not raise.
        rm = paddle.zeros([3, 4])
        tokens = paddle.randn([3, 8])
        permuted, sorted_indices = permute(
            tokens, rm, use_accuracy_compatible=True
        )
        self.assertEqual(permuted.shape[0], 0)

    def test_permute_mixed_padding_forward_matches_default(self):
        rm = _make_padded_routing_map()
        tokens = paddle.randn([5, 8])
        p_a, si_a = permute(tokens, rm, use_accuracy_compatible=True)
        p_s, si_s = permute(tokens, rm, use_accuracy_compatible=False)
        self.assertEqual(p_a.shape[0], 6)  # 3 valid rows * topk 2
        np.testing.assert_allclose(p_a.numpy(), p_s.numpy(), atol=1e-6)
        np.testing.assert_array_equal(si_a.numpy(), si_s.numpy())

    def test_permute_mixed_padding_backward_zeroes_padding_rows(self):
        rm = _make_padded_routing_map()
        tokens = paddle.randn([5, 4])
        tokens.stop_gradient = False
        permuted, _ = permute(tokens, rm, use_accuracy_compatible=True)
        permuted.sum().backward()
        expected = np.full([5, 4], 2.0, dtype="float32")
        expected[0] = 0.0
        expected[2] = 0.0
        np.testing.assert_allclose(tokens.grad.numpy(), expected, atol=1e-6)


class TestUnpermuteAligned(unittest.TestCase):
    def test_unpermute_forward_matches_default(self):
        rm = _make_fixed_topk_routing_map(6, 4, 2)
        tokens = paddle.randn([6, 8])
        p, si = permute(tokens, rm, use_accuracy_compatible=True)
        out_a = unpermute(
            p, si, tokens.shape, routing_map=rm, use_accuracy_compatible=True
        )
        out_s = unpermute(
            p, si, tokens.shape, routing_map=rm, use_accuracy_compatible=False
        )
        self.assertEqual(out_a.shape, [6, 8])
        np.testing.assert_allclose(out_a.numpy(), out_s.numpy(), atol=1e-5)

    def test_permute_unpermute_roundtrip(self):
        topk = 2
        rm = _make_fixed_topk_routing_map(6, 4, topk)
        tokens = paddle.randn([6, 8])
        p, si = permute(tokens, rm, use_accuracy_compatible=True)
        out = unpermute(
            p, si, tokens.shape, routing_map=rm, use_accuracy_compatible=True
        )
        # unpermute sums the topk copies back => topk * original token.
        np.testing.assert_allclose(
            out.numpy(), (topk * tokens).numpy(), atol=1e-5
        )

    def test_unpermute_backward_shape(self):
        rm = _make_fixed_topk_routing_map(5, 3, 2)
        tokens = paddle.randn([5, 4])
        tokens.stop_gradient = False
        p, si = permute(tokens, rm, use_accuracy_compatible=True)
        out = unpermute(
            p, si, tokens.shape, routing_map=rm, use_accuracy_compatible=True
        )
        out.sum().backward()
        self.assertEqual(tokens.grad.shape, [5, 4])

    def test_unpermute_with_probs_matches_default(self):
        rm = _make_fixed_topk_routing_map(6, 4, 2)
        tokens = paddle.randn([6, 8])
        probs = paddle.rand([6, 4])
        p, si = permute(tokens, rm, use_accuracy_compatible=True)
        out_a = unpermute(
            p,
            si,
            tokens.shape,
            probs=probs,
            routing_map=rm,
            use_accuracy_compatible=True,
        )
        out_s = unpermute(
            p,
            si,
            tokens.shape,
            probs=probs,
            routing_map=rm,
            use_accuracy_compatible=False,
        )
        self.assertEqual(out_a.shape, [6, 8])
        np.testing.assert_allclose(out_a.numpy(), out_s.numpy(), atol=1e-5)


class TestAlignedPaddedRoutingMap(unittest.TestCase):
    """Mixed valid / all-zero routing rows must work in forward and backward."""

    def test_unpermute_mixed_padding_forward_matches_default(self):
        rm = _make_padded_routing_map()
        tokens = paddle.randn([5, 8])
        p, si = permute(tokens, rm, use_accuracy_compatible=True)
        out_a = unpermute(
            p, si, tokens.shape, routing_map=rm, use_accuracy_compatible=True
        )
        out_s = unpermute(
            p, si, tokens.shape, routing_map=rm, use_accuracy_compatible=False
        )
        self.assertEqual(out_a.shape, [5, 8])
        np.testing.assert_allclose(out_a.numpy(), out_s.numpy(), atol=1e-5)
        # Padding rows produce zero output.
        np.testing.assert_allclose(
            out_a.numpy()[[0, 2]], np.zeros([2, 8], dtype="float32"), atol=0
        )

    def test_unpermute_mixed_padding_roundtrip(self):
        rm = _make_padded_routing_map()
        tokens = paddle.randn([5, 8])
        p, si = permute(tokens, rm, use_accuracy_compatible=True)
        out = unpermute(
            p, si, tokens.shape, routing_map=rm, use_accuracy_compatible=True
        )
        expected = (2 * tokens).numpy()
        expected[[0, 2]] = 0.0
        np.testing.assert_allclose(out.numpy(), expected, atol=1e-5)

    def test_unpermute_mixed_padding_backward(self):
        rm = _make_padded_routing_map()
        permuted = paddle.randn([6, 4])
        permuted.stop_gradient = False
        out = unpermute(
            permuted,
            None,
            [5, 4],
            routing_map=rm,
            use_accuracy_compatible=True,
        )
        out.sum().backward()
        # Every permuted row feeds exactly one valid output row.
        np.testing.assert_allclose(
            permuted.grad.numpy(),
            np.ones([6, 4], dtype="float32"),
            atol=1e-6,
        )

    def test_permute_unpermute_mixed_padding_backward(self):
        rm = _make_padded_routing_map()
        tokens = paddle.randn([5, 4])
        tokens.stop_gradient = False
        p, si = permute(tokens, rm, use_accuracy_compatible=True)
        out = unpermute(
            p, si, tokens.shape, routing_map=rm, use_accuracy_compatible=True
        )
        out.sum().backward()
        expected = np.full([5, 4], 2.0, dtype="float32")
        expected[0] = 0.0
        expected[2] = 0.0
        np.testing.assert_allclose(tokens.grad.numpy(), expected, atol=1e-6)

    def test_unpermute_all_padding_returns_zeros(self):
        rm = paddle.zeros([3, 4])
        permuted = paddle.zeros([0, 8])
        out = unpermute(
            permuted,
            None,
            [3, 8],
            routing_map=rm,
            use_accuracy_compatible=True,
        )
        np.testing.assert_allclose(
            out.numpy(), np.zeros([3, 8], dtype="float32"), atol=0
        )


if __name__ == "__main__":
    unittest.main()
