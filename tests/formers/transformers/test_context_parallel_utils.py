# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

"""Behavior tests for context-parallel (CP) sequence-dim load-balance split.

Module under test: paddlefleet.transformers.context_parallel_utils

`split_inputs_sequence_dim_load_balance` implements the *dualchunk* (zigzag)
load-balancing position mapping used for causal-attention context parallelism:
the sequence is cut into ``2 * degree`` contiguous chunks and CP ``rank`` keeps
chunk ``rank`` together with its mirror chunk ``2*degree-1-rank``. Pairing an
early chunk with a late chunk equalizes the causal-attention work per rank and
differs from a plain *contiguous* split (where rank r would take a single chunk
r of ``degree`` chunks). This mapping is a pure index/reorder computation with
no cross-rank communication, so it is fully CPU-verifiable here. The expected
mapping below is derived independently by hand (numpy fancy-indexing), never by
calling the production split. Real cross-rank collective numerics are out of
scope for this no-card test (see the skipped auto-parallel case at the bottom).
"""

import unittest

import numpy as np
import paddle

from paddlefleet.transformers.context_parallel_utils import (
    auto_split_sequence_dim_load_balance,
    split_inputs_sequence_dim_load_balance,
)


def expected_load_balance_indices(seqlen, rank, degree):
    """Independent hand-derived mapping of original sequence positions that CP
    ``rank`` receives under ``2*degree`` dualchunk load balancing.

    Pure arithmetic on chunk indices; does not touch the production function.
    """
    assert seqlen % (2 * degree) == 0
    chunk = seqlen // (2 * degree)
    low = list(range(rank * chunk, (rank + 1) * chunk))
    mirror = 2 * degree - 1 - rank
    high = list(range(mirror * chunk, (mirror + 1) * chunk))
    return low + high


def positional_tensor(rows):
    """Build a [len(rows), width] tensor whose value equals its position id,
    so output columns directly reveal which positions a rank received."""
    return paddle.to_tensor(np.asarray(rows, dtype="float32"))


class TestSplitLoadBalancePositionMapping(unittest.TestCase):
    """CPU-verifiable dualchunk position/index mapping of the CP split."""

    def test_degree_one_returns_input_unchanged(self):
        # degree <= 1 short-circuits: same object, no reordering.
        tensor = paddle.arange(8, dtype="float32").reshape([1, 8])
        result = split_inputs_sequence_dim_load_balance(
            tensor, rank=0, degree=1
        )
        self.assertIs(result, tensor)
        np.testing.assert_array_equal(
            result.numpy(), np.arange(8, dtype="float32").reshape(1, 8)
        )

    def test_degree_zero_returns_input_unchanged(self):
        tensor = paddle.arange(8, dtype="float32").reshape([1, 8])
        result = split_inputs_sequence_dim_load_balance(
            tensor, rank=0, degree=0
        )
        self.assertIs(result, tensor)

    def test_rank0_degree2_takes_first_and_last_chunk(self):
        # 8 positions -> 4 chunks [0,1][2,3][4,5][6,7]; rank0 = chunk0+chunk3.
        tensor = paddle.arange(8, dtype="float32").reshape([1, 8])
        result = split_inputs_sequence_dim_load_balance(
            tensor, rank=0, degree=2
        )
        expected_idx = expected_load_balance_indices(8, rank=0, degree=2)
        self.assertEqual(expected_idx, [0, 1, 6, 7])  # pin the hand derivation
        np.testing.assert_array_equal(
            result.numpy(), tensor.numpy()[:, expected_idx]
        )

    def test_rank1_degree2_takes_two_middle_chunks(self):
        tensor = paddle.arange(8, dtype="float32").reshape([1, 8])
        result = split_inputs_sequence_dim_load_balance(
            tensor, rank=1, degree=2
        )
        expected_idx = expected_load_balance_indices(8, rank=1, degree=2)
        self.assertEqual(expected_idx, [2, 3, 4, 5])
        np.testing.assert_array_equal(
            result.numpy(), tensor.numpy()[:, expected_idx]
        )

    def test_all_ranks_cover_every_position_exactly_once(self):
        # Load balancing must be a partition: union == full sequence, no dup.
        for degree, seqlen in ((2, 8), (3, 12), (4, 16)):
            tensor = paddle.arange(seqlen, dtype="float32").reshape([1, seqlen])
            collected = []
            for rank in range(degree):
                out = split_inputs_sequence_dim_load_balance(
                    tensor, rank=rank, degree=degree
                )
                np.testing.assert_array_equal(
                    out.numpy(),
                    tensor.numpy()[
                        :, expected_load_balance_indices(seqlen, rank, degree)
                    ],
                )
                collected.extend(out.numpy().reshape(-1).tolist())
            self.assertEqual(sorted(collected), list(range(seqlen)))

    def test_each_rank_pairs_symmetric_chunks(self):
        # Distinguishes dualchunk from contiguous: the two chunk indices a rank
        # receives must sum to 2*degree-1 (early chunk paired with late chunk).
        degree, seqlen = 3, 12
        chunk = seqlen // (2 * degree)
        tensor = paddle.arange(seqlen, dtype="float32").reshape([1, seqlen])
        for rank in range(degree):
            out = split_inputs_sequence_dim_load_balance(
                tensor, rank=rank, degree=degree
            )
            positions = out.numpy().reshape(-1).astype(int).tolist()
            chunk_ids = sorted({p // chunk for p in positions})
            self.assertEqual(len(chunk_ids), 2)
            self.assertEqual(chunk_ids[0] + chunk_ids[1], 2 * degree - 1)

    def test_differs_from_contiguous_split(self):
        # A contiguous split would give rank0 the first seqlen/degree positions
        # [0,1,2,3]; the load-balanced mapping must NOT collapse to that.
        tensor = paddle.arange(8, dtype="float32").reshape([1, 8])
        result = split_inputs_sequence_dim_load_balance(
            tensor, rank=0, degree=2
        )
        contiguous_first = np.array([[0.0, 1.0, 2.0, 3.0]], dtype="float32")
        self.assertFalse(np.array_equal(result.numpy(), contiguous_first))
        np.testing.assert_array_equal(
            result.numpy(), np.array([[0.0, 1.0, 6.0, 7.0]], dtype="float32")
        )

    def test_batch_dim_preserved_split_along_sequence(self):
        # Two distinguishable rows; split is on axis=-1, rows stay independent.
        tensor = positional_tensor(
            [list(range(8)), [100 + i for i in range(8)]]
        )
        result = split_inputs_sequence_dim_load_balance(
            tensor, rank=0, degree=2
        )
        np.testing.assert_array_equal(
            result.numpy(),
            np.array([[0, 1, 6, 7], [100, 101, 106, 107]], dtype="float32"),
        )


class TestSplitLoadBalanceContainers(unittest.TestCase):
    """Container dispatch preserves keys/order and reorders each entry."""

    def test_dict_reorders_each_value_and_keeps_keys(self):
        a = paddle.arange(8, dtype="float32").reshape([1, 8])
        b = (paddle.arange(8, dtype="float32") + 100).reshape([1, 8])
        result = split_inputs_sequence_dim_load_balance(
            {"input_ids": a, "labels": b}, rank=0, degree=2
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result.keys()), ["input_ids", "labels"])
        np.testing.assert_array_equal(
            result["input_ids"].numpy(),
            np.array([[0, 1, 6, 7]], dtype="float32"),
        )
        np.testing.assert_array_equal(
            result["labels"].numpy(),
            np.array([[100, 101, 106, 107]], dtype="float32"),
        )

    def test_list_reorders_each_entry_and_keeps_order(self):
        first = paddle.arange(8, dtype="float32").reshape([1, 8])
        second = (paddle.arange(8, dtype="float32") + 100).reshape([1, 8])
        result = split_inputs_sequence_dim_load_balance(
            [first, second], rank=1, degree=2
        )
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        # rank1/degree2 -> positions [2,3,4,5]
        np.testing.assert_array_equal(
            result[0].numpy(), np.array([[2, 3, 4, 5]], dtype="float32")
        )
        np.testing.assert_array_equal(
            result[1].numpy(), np.array([[102, 103, 104, 105]], dtype="float32")
        )

    def test_none_value_in_dict_passes_through(self):
        result = split_inputs_sequence_dim_load_balance(
            {"attention_mask": None}, rank=0, degree=2
        )
        self.assertIsNone(result["attention_mask"])


class TestSplitLoadBalanceValidation(unittest.TestCase):
    """Explicit error contracts on malformed inputs / arguments."""

    def test_invalid_top_level_type_raises_value_error(self):
        with self.assertRaises(ValueError):
            split_inputs_sequence_dim_load_balance(
                "not_valid", rank=0, degree=2
            )

    def test_non_2d_tensor_raises_assertion(self):
        tensor = paddle.randn([2, 3, 4])
        with self.assertRaises(AssertionError):
            split_inputs_sequence_dim_load_balance(tensor, rank=0, degree=2)

    def test_non_tensor_entry_in_list_raises_assertion(self):
        with self.assertRaises(AssertionError):
            split_inputs_sequence_dim_load_balance(
                ["not_a_tensor"], rank=0, degree=2
            )

    def test_non_int_degree_raises_assertion(self):
        tensor = paddle.arange(8, dtype="float32").reshape([1, 8])
        with self.assertRaises(AssertionError):
            split_inputs_sequence_dim_load_balance(tensor, rank=0, degree=2.0)

    def test_partial_none_arguments_raise_assertion(self):
        # Only one of rank/degree given -> the fleet fast path is skipped and
        # the int assertion fires on the remaining None.
        tensor = paddle.arange(8, dtype="float32").reshape([1, 8])
        with self.assertRaises(AssertionError):
            split_inputs_sequence_dim_load_balance(tensor, rank=None, degree=2)


class TestAutoSplitLoadBalance(unittest.TestCase):
    """`auto_split_sequence_dim_load_balance` (auto-parallel dispatch)."""

    def test_invalid_type_raises_value_error(self):
        # Reaches the else-branch without invoking shard_seq_load_balance, so
        # this input-validation contract is genuinely CPU-verifiable.
        with self.assertRaises(ValueError):
            auto_split_sequence_dim_load_balance("not_valid")

    def test_shard_seq_load_balance_numerics_deferred(self):
        # The Tensor/dict/list branches delegate to paddle's
        # shard_seq_load_balance, whose sharding needs a real auto-parallel
        # mesh / process group (multi-card). Mocking it would fake the very
        # collective under test (antipattern: mocking the tested kernel and
        # claiming multi-card numerics from a single process). Its numeric
        # correctness belongs to a multi-card test with a real process group.
        self.skipTest(
            "shard_seq_load_balance requires a real auto-parallel process "
            "group; verified under multi-card, not on CPU (no-card)."
        )


if __name__ == "__main__":
    unittest.main()
