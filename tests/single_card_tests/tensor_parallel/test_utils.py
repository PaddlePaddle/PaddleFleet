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

"""Behavior tests for ``paddlefleet.tensor_parallel.utils``.

Module under test lives in the "分布式训练" module of the repository map. It
provides the tensor-parallel splitting/gathering helpers plus vocab-range
math used by vocab-parallel embeddings.

Scope of THIS file (single card / local, hand-derived numeric anchors):

* ``VocabUtility`` range math is pure integer arithmetic and is checked with
  independently computed expectations, including the fact that the
  per-partition variant derives its range from ``rank`` alone and ignores its
  ``world_size`` argument, and that the global variant funnels through
  ``divide`` (so a non-divisible vocab raises ``AssertionError``).
* ``split_tensor_along_last_dim`` is exercised with ``arange`` content so that
  the exact per-chunk *values* and the split *axis* are pinned, not just the
  shapes. The list-vs-tuple return contract of ``contiguous_split_chunks`` is
  also checked.
* ``split_tensor_into_1d_equal_chunks`` (``new_buffer=False`` view path) is
  driven with the group resolver ``get_tensor_model_parallel_group_if_none``
  replaced by a controlled group. That resolver is a genuine, not-under-test
  distributed collaborator (it returns ``None`` when paddle.distributed is not
  initialized, which this single-process CPU environment is); the slice index
  math ``start = partition_size * rank`` remains real and its exact output
  content is asserted per rank.
* ``gather_split_1d_tensor`` input guard: a non-1-D tensor must raise
  ``AssertionError`` before any collective runs.

NOT covered here (requires a real multi-rank process group, see
``tests/multi_card_tests``): the ``dist.stream.all_gather`` numeric path of
``gather_split_1d_tensor`` and the ``new_buffer=True`` allocate-and-copy path
of ``split_tensor_into_1d_equal_chunks``. Faking the collective on a single
process would only prove local orchestration, not cross-rank gather.

The module imports ``paddle`` at load time. This environment may have no
paddle, so imports are guarded and the test classes skip honestly rather than
fake-pass.
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
    from types import SimpleNamespace
    from unittest import mock

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

SKIP_REASON = "paddle is not installed in this environment"


class TestVocabUtility(unittest.TestCase):
    """Pure-integer vocab range math. No paddle tensor ops are involved, but
    the module still imports paddle at load time, so the class skips honestly
    when paddle is absent."""

    @unittest.skipUnless(HAS_PADDLE, SKIP_REASON)
    def test_per_partition_range_is_rank_times_size(self):
        # index_f = rank * per_partition_vocab_size ; index_l = index_f + size.
        self.assertEqual(
            VocabUtility.vocab_range_from_per_partition_vocab_size(100, 0, 2),
            (0, 100),
        )
        # rank 3, per-partition 32 -> [96, 128).
        self.assertEqual(
            VocabUtility.vocab_range_from_per_partition_vocab_size(32, 3, 8),
            (96, 128),
        )

    @unittest.skipUnless(HAS_PADDLE, SKIP_REASON)
    def test_per_partition_ignores_world_size_argument(self):
        # The per-partition variant computes only from rank + per-partition
        # size; the world_size argument does not enter the arithmetic.
        base = VocabUtility.vocab_range_from_per_partition_vocab_size(50, 1, 2)
        other = VocabUtility.vocab_range_from_per_partition_vocab_size(
            50, 1, 999
        )
        self.assertEqual(base, (50, 100))
        self.assertEqual(other, (50, 100))

    @unittest.skipUnless(HAS_PADDLE, SKIP_REASON)
    def test_global_range_divides_then_ranges(self):
        # per_partition = global // world_size, then rank-based range.
        self.assertEqual(
            VocabUtility.vocab_range_from_global_vocab_size(200, 0, 2),
            (0, 100),
        )
        self.assertEqual(
            VocabUtility.vocab_range_from_global_vocab_size(200, 1, 2),
            (100, 200),
        )
        # global 120, world 4 -> per_partition 30 ; rank 2 -> [60, 90).
        self.assertEqual(
            VocabUtility.vocab_range_from_global_vocab_size(120, 2, 4),
            (60, 90),
        )

    @unittest.skipUnless(HAS_PADDLE, SKIP_REASON)
    def test_global_range_non_divisible_raises(self):
        # 7 vocab across 2 ranks is not divisible: divide() asserts.
        with self.assertRaises(AssertionError):
            VocabUtility.vocab_range_from_global_vocab_size(7, 0, 2)


class TestSplitTensorAlongLastDim(unittest.TestCase):
    """split_tensor_along_last_dim splits along the last axis into equal
    chunks; content and axis are pinned with arange inputs."""

    @unittest.skipUnless(HAS_PADDLE, SKIP_REASON)
    def test_splits_last_axis_by_content(self):
        # [[0,1,2,3],[4,5,6,7]] split into 2 along axis=1.
        tensor = paddle.arange(8, dtype="float32").reshape([2, 4])
        result = split_tensor_along_last_dim(tensor, 2)
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(
            result[0].numpy(), np.array([[0.0, 1.0], [4.0, 5.0]])
        )
        np.testing.assert_array_equal(
            result[1].numpy(), np.array([[2.0, 3.0], [6.0, 7.0]])
        )

    @unittest.skipUnless(HAS_PADDLE, SKIP_REASON)
    def test_single_partition_returns_input_content(self):
        tensor = paddle.arange(6, dtype="float32").reshape([2, 3])
        result = split_tensor_along_last_dim(tensor, 1)
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0].numpy(), tensor.numpy())

    @unittest.skipUnless(HAS_PADDLE, SKIP_REASON)
    def test_contiguous_flag_returns_tuple_with_same_content(self):
        # contiguous_split_chunks=True changes the return container from a
        # list to a tuple; the values must still be the same partition.
        tensor = paddle.arange(8, dtype="float32").reshape([2, 4])
        as_list = split_tensor_along_last_dim(tensor, 2)
        as_tuple = split_tensor_along_last_dim(
            tensor, 2, contiguous_split_chunks=True
        )
        self.assertIsInstance(as_tuple, tuple)
        self.assertNotIsInstance(as_list, tuple)
        for got, ref in zip(as_tuple, as_list):
            np.testing.assert_array_equal(got.numpy(), ref.numpy())

    @unittest.skipUnless(HAS_PADDLE, SKIP_REASON)
    def test_non_divisible_last_dim_raises(self):
        # last_dim 7 split into 2 funnels through divide() -> AssertionError.
        tensor = paddle.arange(14, dtype="float32").reshape([2, 7])
        with self.assertRaises(AssertionError):
            split_tensor_along_last_dim(tensor, 2)


class TestSplitTensorInto1DEqualChunks(unittest.TestCase):
    """split_tensor_into_1d_equal_chunks (view path) returns this rank's
    contiguous slice ``[partition_size*rank : partition_size*(rank+1))`` of the
    flattened tensor. The group resolver is a genuine not-under-test
    collaborator and is replaced with a controlled group; the index math stays
    real."""

    @unittest.skipUnless(HAS_PADDLE, SKIP_REASON)
    def test_view_slice_rank0(self):
        group = SimpleNamespace(world_size=4, rank=0)
        tensor = paddle.arange(8, dtype="float32")
        with mock.patch.object(
            tp_utils,
            "get_tensor_model_parallel_group_if_none",
            return_value=group,
        ):
            out = split_tensor_into_1d_equal_chunks(tensor)
        # partition_size = 8 // 4 = 2 ; rank 0 -> indices [0, 2).
        np.testing.assert_array_equal(out.numpy(), np.array([0.0, 1.0]))

    @unittest.skipUnless(HAS_PADDLE, SKIP_REASON)
    def test_view_slice_rank2(self):
        group = SimpleNamespace(world_size=4, rank=2)
        tensor = paddle.arange(8, dtype="float32")
        with mock.patch.object(
            tp_utils,
            "get_tensor_model_parallel_group_if_none",
            return_value=group,
        ):
            out = split_tensor_into_1d_equal_chunks(tensor)
        # rank 2 -> indices [4, 6).
        np.testing.assert_array_equal(out.numpy(), np.array([4.0, 5.0]))


class TestGatherSplit1DTensorGuard(unittest.TestCase):
    """gather_split_1d_tensor requires a 1-D input and rejects others before
    reaching the collective."""

    @unittest.skipUnless(HAS_PADDLE, SKIP_REASON)
    def test_non_1d_input_raises_assertion(self):
        tensor = paddle.arange(6, dtype="float32").reshape([2, 3])
        with self.assertRaisesRegex(AssertionError, "should be 1"):
            gather_split_1d_tensor(tensor)


if __name__ == "__main__":
    unittest.main()
