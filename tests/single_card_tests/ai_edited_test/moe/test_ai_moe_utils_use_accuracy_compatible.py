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

import os
import sys

_test_file = os.path.abspath(__file__)
_repo_root = _test_file
for _ in range(10):
    _repo_root = os.path.dirname(_repo_root)
    if os.path.isdir(os.path.join(_repo_root, "src", "paddlefleet")):
        break
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_repo_root, "src"))
for _mod in list(sys.modules.keys()):
    if _mod == "paddlefleet" or _mod.startswith("paddlefleet."):
        del sys.modules[_mod]

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


if __name__ == "__main__":
    unittest.main()
