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
"""Behavior tests for ``permute`` / ``unpermute`` in ``transformer/moe/moe_utils``.

These MoE dispatch primitives group tokens by their selected expert (``permute``)
and later scatter/sum the per-expert copies back to the original token slots
(``unpermute``). The ``use_accuracy_compatible`` (Megatron-aligned) path routes
through dedicated ``PyLayer``s that regroup a token's ``top_k`` copies through an
explicit gather index instead of a plain ``scatter_``.

Every expected value below is derived by hand from a small, fully written-out
routing map and distinguishable token contents -- never by calling the function
under test to produce its own reference, and never by comparing the two
production paths against each other. Concretely, for the fixed map

    token0 -> experts {0, 1}
    token1 -> experts {1, 2}
    token2 -> experts {0, 2}

expert-major grouping visits expert 0 (tokens 0, 2), then expert 1 (tokens 0, 1),
then expert 2 (tokens 1, 2), giving ``sorted_indices == [0, 2, 0, 1, 1, 2]``. The
gather / sum / gradient expectations all follow from that written-out ordering.

Paddle is required for the real numeric behavior. When it is unavailable the
whole module skips with an honest reason rather than reporting a fake pass.
"""

import os
import sys
import unittest

import numpy as np

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
for _candidate in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

try:
    import paddle

    from paddlefleet.transformer.moe.moe_utils import permute, unpermute

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle (or a paddle-backed import) is missing
    paddle = None
    permute = None
    unpermute = None
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    f"paddle is not importable in this environment: {_IMPORT_ERROR!r}"
    if _IMPORT_ERROR is not None
    else ""
)


def _tokens():
    """Three tokens with pairwise-distinct, position-distinguishable content."""
    return paddle.to_tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype="float32"
    )


def _routing_map_topk2():
    """token0->{0,1}, token1->{1,2}, token2->{0,2}; fixed top-k == 2."""
    return paddle.to_tensor(
        [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]], dtype="float32"
    )


def _routing_map_with_padding():
    """Row 0 is an all-zero padding row; rows 1,2 keep a fixed top-k == 2."""
    return paddle.to_tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], dtype="float32"
    )


@unittest.skipUnless(paddle is not None, _SKIP_REASON)
class TestPermute(unittest.TestCase):
    def test_default_forward_groups_tokens_by_expert(self):
        # Expert-major scan of the map yields sorted_indices [0,2,0,1,1,2];
        # permuted row i is tokens[sorted_indices[i]].
        permuted, sorted_indices = permute(
            _tokens(), _routing_map_topk2(), use_accuracy_compatible=False
        )
        np.testing.assert_array_equal(
            sorted_indices.numpy(), np.array([0, 2, 0, 1, 1, 2])
        )
        np.testing.assert_array_equal(
            permuted.numpy(),
            np.array(
                [
                    [1.0, 2.0],  # token0 (expert 0)
                    [5.0, 6.0],  # token2 (expert 0)
                    [1.0, 2.0],  # token0 (expert 1)
                    [3.0, 4.0],  # token1 (expert 1)
                    [3.0, 4.0],  # token1 (expert 2)
                    [5.0, 6.0],  # token2 (expert 2)
                ],
                dtype="float32",
            ),
        )

    def test_aligned_forward_matches_hand_derived(self):
        # The aligned PyLayer forward is a plain index_select, so its output must
        # equal the same hand-derived grouping (derived independently, not by
        # comparing against the default path).
        permuted, sorted_indices = permute(
            _tokens(), _routing_map_topk2(), use_accuracy_compatible=True
        )
        np.testing.assert_array_equal(
            sorted_indices.numpy(), np.array([0, 2, 0, 1, 1, 2])
        )
        np.testing.assert_array_equal(
            permuted.numpy(),
            np.array(
                [
                    [1.0, 2.0],
                    [5.0, 6.0],
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                ],
                dtype="float32",
            ),
        )

    def test_aligned_backward_sums_topk_copies(self):
        # Each token is copied to exactly top-k == 2 experts, so
        # d(sum(permuted))/d(token) == 2 for every element.
        tokens = _tokens()
        tokens.stop_gradient = False
        permuted, _ = permute(
            tokens, _routing_map_topk2(), use_accuracy_compatible=True
        )
        permuted.sum().backward()
        self.assertIsNotNone(tokens.grad)
        np.testing.assert_array_equal(
            tokens.grad.numpy(), np.full([3, 2], 2.0, dtype="float32")
        )

    def test_aligned_variable_topk_raises(self):
        # token0 -> 2 experts, token1 -> 1 expert: not a fixed top-k, so the
        # aligned gather-index builder must reject it.
        rm = paddle.to_tensor(
            [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype="float32"
        )
        tokens = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        with self.assertRaises(ValueError):
            permute(tokens, rm, use_accuracy_compatible=True)

    def test_aligned_all_padding_yields_empty_permutation(self):
        # No token is routed anywhere -> zero permuted rows, no exception.
        rm = paddle.zeros([3, 3], dtype="float32")
        tokens = paddle.to_tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype="float32"
        )
        permuted, sorted_indices = permute(
            tokens, rm, use_accuracy_compatible=True
        )
        self.assertEqual(list(permuted.shape), [0, 2])
        self.assertEqual(int(sorted_indices.numel()), 0)

    def test_padding_forward_excludes_padding_rows(self):
        # Padding row 0 contributes nothing; rows 1,2 each route to 2 experts.
        # Expert-major scan: e0->[t1], e1->[t1,t2], e2->[t2] => [1,1,2,2].
        permuted, sorted_indices = permute(
            _tokens(), _routing_map_with_padding(), use_accuracy_compatible=True
        )
        np.testing.assert_array_equal(
            sorted_indices.numpy(), np.array([1, 1, 2, 2])
        )
        np.testing.assert_array_equal(
            permuted.numpy(),
            np.array(
                [[3.0, 4.0], [3.0, 4.0], [5.0, 6.0], [5.0, 6.0]],
                dtype="float32",
            ),
        )


@unittest.skipUnless(paddle is not None, _SKIP_REASON)
class TestUnpermute(unittest.TestCase):
    def test_default_roundtrip_sums_topk_copies(self):
        # unpermute scatter-adds each token's top-k copies back, so the round
        # trip must recover top-k * original token = 2 * token.
        tokens = _tokens()
        rm = _routing_map_topk2()
        permuted, sorted_indices = permute(
            tokens, rm, use_accuracy_compatible=False
        )
        out = unpermute(
            permuted,
            sorted_indices,
            tokens.shape,
            routing_map=rm,
            use_accuracy_compatible=False,
        )
        np.testing.assert_array_equal(
            out.numpy(),
            np.array([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]], dtype="float32"),
        )

    def test_aligned_roundtrip_sums_topk_copies(self):
        # The aligned gather-sum path regroups each token's two permuted rows and
        # sums them; the round trip must also recover 2 * token.
        tokens = _tokens()
        rm = _routing_map_topk2()
        permuted, sorted_indices = permute(
            tokens, rm, use_accuracy_compatible=True
        )
        out = unpermute(
            permuted,
            sorted_indices,
            tokens.shape,
            routing_map=rm,
            use_accuracy_compatible=True,
        )
        np.testing.assert_array_equal(
            out.numpy(),
            np.array([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]], dtype="float32"),
        )

    def test_default_with_probs_weights_each_expert_copy(self):
        # probs are gathered in the same expert-major order and multiplied onto
        # each copy before the scatter-add, so output[j] == token_j * (sum of
        # that token's routed-expert weights). Non-routed entries are masked out.
        tokens = _tokens()
        rm = _routing_map_topk2()
        probs = paddle.to_tensor(
            [
                [0.5, 0.25, 0.0],  # token0: experts 0,1 -> 0.5 + 0.25 = 0.75
                [0.0, 0.1, 0.4],  # token1: experts 1,2 -> 0.1 + 0.4  = 0.5
                [0.2, 0.0, 0.3],  # token2: experts 0,2 -> 0.2 + 0.3  = 0.5
            ],
            dtype="float32",
        )
        permuted, sorted_indices = permute(
            tokens, rm, use_accuracy_compatible=False
        )
        out = unpermute(
            permuted,
            sorted_indices,
            tokens.shape,
            probs=probs,
            routing_map=rm,
            use_accuracy_compatible=False,
        )
        np.testing.assert_allclose(
            out.numpy(),
            np.array([[0.75, 1.5], [1.5, 2.0], [2.5, 3.0]], dtype="float32"),
            atol=1e-6,
        )

    def test_aligned_padding_rows_produce_zero_output(self):
        # Round trip through the aligned path with a padding row: valid rows sum
        # their two copies (2 * token) while the padding row stays exactly zero.
        tokens = _tokens()
        rm = _routing_map_with_padding()
        permuted, sorted_indices = permute(
            tokens, rm, use_accuracy_compatible=True
        )
        out = unpermute(
            permuted,
            sorted_indices,
            tokens.shape,
            routing_map=rm,
            use_accuracy_compatible=True,
        )
        np.testing.assert_array_equal(
            out.numpy(),
            np.array([[0.0, 0.0], [6.0, 8.0], [10.0, 12.0]], dtype="float32"),
        )

    def test_aligned_forward_and_backward_on_explicit_permuted_input(self):
        # Drive the aligned gather-sum PyLayer directly with written-out permuted
        # rows so both forward and gradient are checked against a hand trace.
        # gather_index_flat == [0,2,3,4,1,5]; grouped by token then summed:
        #   token0 = rows 0,2 ; token1 = rows 3,4 ; token2 = rows 1,5.
        rm = _routing_map_topk2()
        permuted = paddle.to_tensor(
            [
                [10.0, 11.0],
                [20.0, 21.0],
                [30.0, 31.0],
                [40.0, 41.0],
                [50.0, 51.0],
                [60.0, 61.0],
            ],
            dtype="float32",
        )
        permuted.stop_gradient = False
        out = unpermute(
            permuted,
            None,
            [3, 2],
            routing_map=rm,
            use_accuracy_compatible=True,
        )
        np.testing.assert_array_equal(
            out.numpy(),
            np.array(
                [
                    [10.0 + 30.0, 11.0 + 31.0],  # token0: rows 0,2
                    [40.0 + 50.0, 41.0 + 51.0],  # token1: rows 3,4
                    [20.0 + 60.0, 21.0 + 61.0],  # token2: rows 1,5
                ],
                dtype="float32",
            ),
        )
        # Each permuted row feeds exactly one output token's sum, so every row's
        # gradient of the summed output is 1.
        out.sum().backward()
        self.assertIsNotNone(permuted.grad)
        np.testing.assert_array_equal(
            permuted.grad.numpy(), np.ones([6, 2], dtype="float32")
        )


if __name__ == "__main__":
    unittest.main()
