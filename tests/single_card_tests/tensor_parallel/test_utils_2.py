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

"""CPU behavior tests for ``paddlefleet.tensor_parallel.utils`` (set _2).

Repository module: "分布式训练" (tensor parallel). This file targets the
CPU-observable, single-rank logic of the TP tensor-split helpers, choosing
branches and observations that are NOT covered elsewhere:

  * ``split_tensor_along_last_dim`` -- the exact per-partition column content
    (not just shape), the list-vs-tuple return contract selected by
    ``contiguous_split_chunks``, and the ``divide`` non-divisibility guard.
  * ``split_tensor_into_1d_equal_chunks`` -- the flatten + ``start = rank *
    partition_size`` slice math on the ``new_buffer=False`` view path, asserted
    against the exact hand-derived slice this rank must receive. The default TP
    group resolver is a genuine not-under-test collaborator and is replaced by a
    plain topology holder; the numel / floor-div / index / view logic stays real.
  * ``VocabUtility.vocab_range_from_global_vocab_size`` -- the divide-then-
    delegate composition and its non-divisibility guard.
  * ``VocabUtility.vocab_range_from_per_partition_vocab_size`` -- the
    ``index_f = rank * per`` / ``index_l = index_f + per`` offset law.
  * ``gather_split_1d_tensor`` -- the 1-D input contract, asserting the raised
    message reports the *actual* ndim (not a constant).

The genuine collective path of ``gather_split_1d_tensor`` (``dist.stream.
all_gather`` across >1 rank) and the ``new_buffer=True`` allocate-and-copy path
are intentionally NOT exercised: the former requires a real multi-rank process
group and belongs in ``tests/multi_card_tests``; faking it here would prove
nothing about cross-rank ordering. Paddle is not installed in the no-card env,
hence the honest skip.
"""

import unittest
from unittest import mock

try:
    import numpy as np
    import paddle

    from paddlefleet.tensor_parallel import utils as tp_utils
    from paddlefleet.tensor_parallel.utils import (
        VocabUtility,
        gather_split_1d_tensor,
        split_tensor_along_last_dim,
        split_tensor_into_1d_equal_chunks,
    )

    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

_SKIP_REASON = "paddle is not installed in this environment"


class _Group:
    """Minimal real stand-in for a TP process-group value object.

    The tested split path only *reads* ``world_size`` and ``rank`` to derive a
    slice; it never invokes a collective on it. This is a plain data holder, not
    a mocked collective.
    """

    def __init__(self, world_size, rank=0):
        self.world_size = world_size
        self.rank = rank
        self.nranks = world_size


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestSplitTensorAlongLastDim(unittest.TestCase):
    """split_tensor_along_last_dim: content, return-type branch, guard."""

    def test_partitions_hold_exact_column_slices(self):
        # Distinguishable content so a wrong split axis / reordered chunks fail.
        tensor = paddle.arange(16, dtype="float32").reshape([2, 8])
        result = split_tensor_along_last_dim(tensor, 2)
        self.assertIsInstance(result, list)  # default path returns a list
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result[0].numpy(), tensor.numpy()[:, 0:4])
        np.testing.assert_array_equal(result[1].numpy(), tensor.numpy()[:, 4:8])
        # Concatenating the partitions must rebuild the original exactly.
        recon = np.concatenate([result[0].numpy(), result[1].numpy()], axis=1)
        np.testing.assert_array_equal(recon, tensor.numpy())

    def test_four_partitions_are_ordered_contiguous_column_blocks(self):
        tensor = paddle.arange(3 * 16, dtype="float32").reshape([3, 16])
        result = split_tensor_along_last_dim(tensor, 4)
        self.assertEqual(len(result), 4)
        for i, chunk in enumerate(result):
            np.testing.assert_array_equal(
                chunk.numpy(), tensor.numpy()[:, i * 4 : (i + 1) * 4]
            )

    def test_contiguous_flag_returns_tuple_with_same_values(self):
        tensor = paddle.arange(16, dtype="float32").reshape([2, 8])
        result = split_tensor_along_last_dim(
            tensor, 2, contiguous_split_chunks=True
        )
        # Distinct branch: the contiguous path returns a *tuple*, not a list,
        # while preserving the same per-partition content.
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result[0].numpy(), tensor.numpy()[:, 0:4])
        np.testing.assert_array_equal(result[1].numpy(), tensor.numpy()[:, 4:8])
        for chunk in result:
            self.assertTrue(chunk.is_contiguous())

    def test_non_divisible_last_dim_raises(self):
        # last dim 7 is not divisible by 2 -> divide()/ensure_divisibility asserts
        # before any split happens.
        tensor = paddle.arange(14, dtype="float32").reshape([2, 7])
        with self.assertRaises(AssertionError):
            split_tensor_along_last_dim(tensor, 2)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestSplitTensorInto1DEqualChunks(unittest.TestCase):
    """split_tensor_into_1d_equal_chunks: view path start = rank * partition."""

    def _split_for(self, tensor, world_size, rank):
        group = _Group(world_size=world_size, rank=rank)
        with mock.patch.object(
            tp_utils,
            "get_tensor_model_parallel_group_if_none",
            return_value=group,
        ):
            return split_tensor_into_1d_equal_chunks(tensor, new_buffer=False)

    def test_rank0_gets_first_flattened_half(self):
        tensor = paddle.arange(8, dtype="float32").reshape([2, 4])
        result = self._split_for(tensor, world_size=2, rank=0)
        # partition_size = 8 // 2 = 4; rank 0 -> flat[0:4]
        np.testing.assert_array_equal(
            result.numpy(), np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        )

    def test_rank1_gets_second_flattened_half(self):
        tensor = paddle.arange(8, dtype="float32").reshape([2, 4])
        result = self._split_for(tensor, world_size=2, rank=1)
        # start = 4 * 1 = 4 -> flat[4:8]; a rank/world swap would fail here.
        np.testing.assert_array_equal(
            result.numpy(), np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32)
        )

    def test_middle_rank_of_four_gets_its_block(self):
        tensor = paddle.arange(12, dtype="float32").reshape([3, 4])
        result = self._split_for(tensor, world_size=4, rank=2)
        # partition_size = 12 // 4 = 3; start = 3 * 2 = 6 -> flat[6:9]
        np.testing.assert_array_equal(
            result.numpy(), np.array([6.0, 7.0, 8.0], dtype=np.float32)
        )


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestVocabUtility(unittest.TestCase):
    """VocabUtility range math: delegation composition and offset law."""

    def test_global_range_divides_then_delegates(self):
        # global 100 / world 4 -> per 25; rank 1 -> [25, 50)
        self.assertEqual(
            VocabUtility.vocab_range_from_global_vocab_size(100, 1, 4), (25, 50)
        )

    def test_global_range_last_rank_reaches_end(self):
        # global 120 / world 3 -> per 40; rank 2 -> [80, 120)
        self.assertEqual(
            VocabUtility.vocab_range_from_global_vocab_size(120, 2, 3),
            (80, 120),
        )

    def test_global_range_non_divisible_raises(self):
        # 100 not divisible by 3 -> divide()/ensure_divisibility asserts.
        with self.assertRaises(AssertionError):
            VocabUtility.vocab_range_from_global_vocab_size(100, 0, 3)

    def test_per_partition_offset_scales_with_rank(self):
        # index_f = rank * per, index_l = index_f + per.
        self.assertEqual(
            VocabUtility.vocab_range_from_per_partition_vocab_size(30, 3, 5),
            (90, 120),
        )
        nxt = VocabUtility.vocab_range_from_per_partition_vocab_size(30, 4, 5)
        # Advancing one rank shifts both bounds by exactly one partition size.
        self.assertEqual(nxt, (120, 150))


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestGatherSplit1DTensorInputContract(unittest.TestCase):
    """gather_split_1d_tensor: the 1-D input guard reports the real ndim."""

    def test_rejects_2d_input_and_reports_ndim(self):
        tensor = paddle.zeros([2, 4], dtype="float32")
        with self.assertRaises(AssertionError) as ctx:
            gather_split_1d_tensor(tensor)
        # Message must reflect the actual rank, not a hardcoded constant.
        self.assertIn("but got 2", str(ctx.exception))

    def test_rejects_3d_input_and_reports_ndim(self):
        tensor = paddle.zeros([2, 3, 4], dtype="float32")
        with self.assertRaises(AssertionError) as ctx:
            gather_split_1d_tensor(tensor)
        self.assertIn("but got 3", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
