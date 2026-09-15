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

"""CPU-observable behavior tests for ``tensor_parallel/layers``.

Module under test: ``paddlefleet.tensor_parallel.layers``. In the repository
module map this is the "分布式训练" boundary. This file deliberately targets the
*pure partitioning / branch logic* that is fully determined on CPU with a
world_size passed in by hand, not the GPU GEMM numerics:

* ``_initialize_affine_weight_cpu`` builds a full master weight, splits it into
  ``per_partition_size / stride`` wide chunks along ``partition_dim`` and hands
  rank ``r`` the chunks ``weight_list[r::world_size]`` concatenated back. The
  chunk selection is hand-derived from a known ``arange`` master so a wrong
  split width, wrong stride interleave or wrong rank slice is rejected by
  exact-content comparison, not by shape alone.
* ``param_is_not_tensor_parallel_duplicate`` returns true when the param is TP
  sharded *or* the caller is rank 0; the full truth table pins both OR clauses.
* the ``set_/set_defaults_/copy_`` attribute helpers record the documented
  default contract, refuse a double set, and copy all three attributes.

Real device GEMM / all-reduce numerics of the Linear layers are GPU-only and are
NOT driven here. ``paddle`` is not importable in this environment, so every case
is honestly skipped rather than faked; the asserts run unchanged where ``paddle``
is present.
"""

import os
import sys
import unittest

import numpy as np

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

try:
    import paddle

    HAS_PADDLE = True
except ImportError:
    paddle = None
    HAS_PADDLE = False

_SKIP_REASON = "paddle is not installed in this environment"


def _fill_arange(tensor):
    """init_method stand-in: fill ``tensor`` in place with 0,1,2,... row-major.

    Deterministic and injective per element, so the post-split content is a
    hand-computable selection of the master's columns.
    """
    numel = 1
    for size in tensor.shape:
        numel *= size
    tensor.set_value(
        paddle.arange(numel, dtype=tensor.dtype).reshape(tensor.shape)
    )


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestInitializeAffineWeightCpuPartition(unittest.TestCase):
    """Column partitioning / stride interleave of ``_initialize_affine_weight_cpu``.

    Fixture: master is a 4x8 ``arange`` matrix; column ``c`` of row ``r`` holds
    ``r * 8 + c``. Partitioning along ``partition_dim=1`` with
    ``per_partition_size=4`` yields chunks of width ``4 / stride``; rank ``r``
    keeps ``chunks[r::world_size]``. The expected column indices below are
    derived by hand from that contract, independent of the production split.
    """

    INPUT_SIZE = 4
    OUTPUT_SIZE = 8
    PER_PARTITION = 4

    def _run(self, stride, rank, world_size):
        from paddlefleet.tensor_parallel.layers import (
            _initialize_affine_weight_cpu,
        )

        weight = paddle.empty(
            [self.INPUT_SIZE, self.PER_PARTITION], dtype=paddle.float32
        )
        master = _initialize_affine_weight_cpu(
            weight,
            self.INPUT_SIZE,
            self.OUTPUT_SIZE,
            self.PER_PARTITION,
            1,  # partition_dim -> columns
            _fill_arange,
            stride=stride,
            return_master_weight=True,
            rank=rank,
            world_size=world_size,
            skip_set_tensor_parallel_attributes=True,
        )
        return weight, master

    def test_partition_selects_hand_derived_columns(self):
        master_ref = np.arange(
            self.INPUT_SIZE * self.OUTPUT_SIZE, dtype=np.float32
        ).reshape([self.INPUT_SIZE, self.OUTPUT_SIZE])

        # (stride, rank, world_size) -> exact columns rank must receive.
        # stride 1: contiguous halves. stride 2: interleaved width-2 chunks.
        cases = [
            (1, 0, 2, [0, 1, 2, 3]),
            (1, 1, 2, [4, 5, 6, 7]),
            (2, 0, 2, [0, 1, 4, 5]),
            (2, 1, 2, [2, 3, 6, 7]),
        ]
        for stride, rank, world_size, columns in cases:
            with self.subTest(stride=stride, rank=rank):
                weight, master = self._run(stride, rank, world_size)
                # The rebuilt master is the untouched arange fixture.
                np.testing.assert_array_equal(master.numpy(), master_ref)
                self.assertEqual(
                    list(weight.shape), [self.INPUT_SIZE, self.PER_PARTITION]
                )
                np.testing.assert_array_equal(
                    weight.numpy(), master_ref[:, columns]
                )

    def test_stride_changes_which_columns_land_on_rank(self):
        # Same rank, only stride differs: a stride-insensitive implementation
        # would hand back the same columns. The contract says otherwise.
        weight_s1, _ = self._run(stride=1, rank=0, world_size=2)
        weight_s2, _ = self._run(stride=2, rank=0, world_size=2)
        self.assertFalse(np.array_equal(weight_s1.numpy(), weight_s2.numpy()))

    def test_world_size_one_returns_full_matrix_and_tags_attributes(self):
        from paddlefleet.tensor_parallel.layers import (
            _initialize_affine_weight_cpu,
        )

        master_ref = np.arange(
            self.INPUT_SIZE * self.OUTPUT_SIZE, dtype=np.float32
        ).reshape([self.INPUT_SIZE, self.OUTPUT_SIZE])
        weight = paddle.empty(
            [self.INPUT_SIZE, self.OUTPUT_SIZE], dtype=paddle.float32
        )
        master = _initialize_affine_weight_cpu(
            weight,
            self.INPUT_SIZE,
            self.OUTPUT_SIZE,
            self.OUTPUT_SIZE,  # per_partition == full -> single chunk
            0,  # partition_dim -> rows
            _fill_arange,
            stride=1,
            return_master_weight=True,
            rank=0,
            world_size=1,
        )
        # world_size 1: rank 0 owns the whole matrix, unchanged.
        np.testing.assert_array_equal(weight.numpy(), master_ref)
        np.testing.assert_array_equal(master.numpy(), master_ref)
        # skip flag left False -> TP attributes recorded on the weight.
        self.assertTrue(weight.tensor_model_parallel)
        self.assertEqual(weight.partition_dim, 0)
        self.assertEqual(weight.partition_stride, 1)


class _NoTPAttr:
    """Param stand-in without any ``tensor_model_parallel`` attribute."""


class _WithTPAttr:
    def __init__(self, is_parallel):
        self.tensor_model_parallel = is_parallel


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestParamIsNotTensorParallelDuplicate(unittest.TestCase):
    """Full truth table of ``param_is_not_tensor_parallel_duplicate``.

    Contract: a param is kept (``True``) when it is genuinely TP sharded
    (``tensor_model_parallel`` truthy) OR when the caller sits on rank 0. Both
    OR clauses are exercised, including the discriminating ``False``-attr /
    rank-0 case where the second clause alone must rescue it.
    """

    def _call(self, param, rank):
        from unittest.mock import patch

        from paddlefleet.tensor_parallel import layers

        with patch.object(
            layers, "get_tensor_model_parallel_rank", return_value=rank
        ):
            return layers.param_is_not_tensor_parallel_duplicate(param)

    def test_sharded_param_kept_on_nonzero_rank(self):
        self.assertTrue(self._call(_WithTPAttr(True), rank=1))

    def test_flagged_false_on_nonzero_rank_is_duplicate(self):
        self.assertFalse(self._call(_WithTPAttr(False), rank=1))

    def test_flagged_false_on_rank_zero_is_kept(self):
        # Second clause (rank 0) rescues an explicitly non-sharded param.
        self.assertTrue(self._call(_WithTPAttr(False), rank=0))

    def test_missing_attr_on_nonzero_rank_is_duplicate(self):
        self.assertFalse(self._call(_NoTPAttr(), rank=1))

    def test_missing_attr_on_rank_zero_is_kept(self):
        self.assertTrue(self._call(_NoTPAttr(), rank=0))


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestTensorModelParallelAttributeHelpers(unittest.TestCase):
    """Default contract, double-set guard and copy fan-out of the TP attribute
    helpers. Values read back are the ones production assigns, not values the
    test planted (except in the copy source)."""

    def test_defaults_assign_documented_contract(self):
        from paddlefleet.tensor_parallel.layers import (
            set_defaults_if_not_set_tensor_model_parallel_attributes,
        )

        tensor = paddle.empty([2, 3], dtype=paddle.float32)
        set_defaults_if_not_set_tensor_model_parallel_attributes(tensor)
        # These constants come from production's default table.
        self.assertIs(tensor.tensor_model_parallel, False)
        self.assertEqual(tensor.partition_dim, -1)
        self.assertEqual(tensor.partition_stride, 1)

    def test_defaults_do_not_override_existing(self):
        from paddlefleet.tensor_parallel.layers import (
            set_defaults_if_not_set_tensor_model_parallel_attributes,
            set_tensor_model_parallel_attributes,
        )

        tensor = paddle.empty([2, 3], dtype=paddle.float32)
        set_tensor_model_parallel_attributes(tensor, True, 0, 3)
        set_defaults_if_not_set_tensor_model_parallel_attributes(tensor)
        # Pre-existing values survive; defaults must not clobber them.
        self.assertTrue(tensor.tensor_model_parallel)
        self.assertEqual(tensor.partition_dim, 0)
        self.assertEqual(tensor.partition_stride, 3)

    def test_set_rejects_double_set(self):
        from paddlefleet.tensor_parallel.layers import (
            set_tensor_model_parallel_attributes,
        )

        tensor = paddle.empty([2, 3], dtype=paddle.float32)
        set_tensor_model_parallel_attributes(tensor, True, 1, 2)
        with self.assertRaises(AssertionError):
            set_tensor_model_parallel_attributes(tensor, False, 0, 1)

    def test_copy_propagates_all_three_attributes(self):
        from paddlefleet.tensor_parallel.layers import (
            copy_tensor_model_parallel_attributes,
            set_tensor_model_parallel_attributes,
        )

        src = paddle.empty([2, 3], dtype=paddle.float32)
        dst = paddle.empty([2, 3], dtype=paddle.float32)
        set_tensor_model_parallel_attributes(src, True, 1, 2)
        copy_tensor_model_parallel_attributes(dst, src)
        # A copy that drops any one of the three fields is caught here.
        self.assertTrue(dst.tensor_model_parallel)
        self.assertEqual(dst.partition_dim, 1)
        self.assertEqual(dst.partition_stride, 2)


if __name__ == "__main__":
    unittest.main()
