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

import os
import sys
import unittest

import paddle
import paddle.distributed as dist

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

from paddlefleet.tensor_parallel.mappings import (
    _AllGatherFromTensorParallelRegion,
    _CopyToModelParallelRegion,
    _GatherFromModelParallelRegion,
    _GatherFromSequenceParallelRegion,
    _ReduceFromModelParallelRegion,
    _ReduceScatterToSequenceParallelRegion,
    _ReduceScatterToTensorParallelRegion,
    _ScatterToModelParallelRegion,
)
from paddlefleet.utils import get_tensor_model_parallel_group_if_none

TP_SIZE = 4
_initialized = False


class TestGatherFromModelParallelRegionBasic(unittest.TestCase):
    """Test _GatherFromModelParallelRegion gathers split tensor."""

    def test_gather_from_model_parallel_region_basic(self):
        """Gather should collect partial tensors into full tensor along last dim."""
        tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)

        # Create a partial tensor: each rank has a slice of the full tensor
        input_data = paddle.ones([8, 4]).cuda() * dist.get_rank()

        class Ctx:
            group = tp_group

        # Backward of Gather is Scatter
        output = _GatherFromModelParallelRegion.backward(Ctx(), input_data)
        rank = dist.get_rank()
        expected = input_data[:, rank % TP_SIZE].reshape([8, 1])
        self.assertTrue(paddle.equal_all(output, expected))

        # Forward gather: combine all rank slices
        input_data = paddle.ones([8]).cuda() * dist.get_rank()
        actual_output = _GatherFromModelParallelRegion.symbolic(
            None, input_data, tp_group
        )
        parts = [paddle.ones([8]).cuda() * r for r in range(TP_SIZE)]
        expected = paddle.concat(parts)
        self.assertTrue(paddle.equal_all(actual_output, expected))


class TestScatterToModelParallelRegionBasic(unittest.TestCase):
    """Test _ScatterToModelParallelRegion scatters full tensor."""

    def test_scatter_to_model_parallel_region_basic(self):
        """Scatter should split full tensor across ranks along last dim."""
        tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
        input_data = paddle.rand((8, 16)).cuda()

        # Forward scatter
        output_data = _ScatterToModelParallelRegion.symbolic(
            None, input_data, tp_group
        )
        rank = dist.get_rank()
        expected = input_data[:, rank * 4 : (rank + 1) * 4]
        self.assertTrue(paddle.equal_all(output_data, expected))


class TestCopyToModelParallelRegionIdentity(unittest.TestCase):
    """Test _CopyToModelParallelRegion passes through in forward."""

    def test_copy_to_model_parallel_region_identity(self):
        """Forward should be identity; backward should be all-reduce."""
        tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)

        input_data = paddle.ones([1]).cuda() * dist.get_rank()

        # Symbolic forward should be identity
        output = _CopyToModelParallelRegion.symbolic(None, input_data, tp_group)
        self.assertTrue(paddle.equal_all(input_data, output))

        # Backward should perform all-reduce
        class Ctx:
            group = tp_group

        output_backward = _CopyToModelParallelRegion.backward(Ctx(), input_data)
        expected = paddle.ones([1]).cuda() * sum(range(TP_SIZE))
        self.assertTrue(paddle.equal_all(output_backward, expected))


class TestReduceFromModelParallelRegionBasic(unittest.TestCase):
    """Test _ReduceFromModelParallelRegion reduces across ranks."""

    def test_reduce_from_model_parallel_region_basic(self):
        """Forward should all-reduce; backward should be identity."""
        tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
        input_data = paddle.ones([1]).cuda() * dist.get_rank()

        # Forward should all-reduce
        output = _ReduceFromModelParallelRegion.symbolic(
            None, input_data, tp_group
        )
        expected = paddle.ones([1]).cuda() * sum(range(TP_SIZE))
        self.assertTrue(paddle.equal_all(output, expected))

        # Backward should be identity
        class Ctx:
            group = tp_group

        output_backward = _ReduceFromModelParallelRegion.backward(
            Ctx(), input_data
        )
        self.assertTrue(paddle.equal_all(input_data, output_backward))


class TestGatherFromSequenceParallelRegion(unittest.TestCase):
    """Test _GatherFromSequenceParallelRegion gathers along sequence dim."""

    def test_gather_from_sequence_parallel_region(self):
        """Gather should collect partial sequence tensors into full tensor."""
        tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
        input_data = paddle.ones([4]).cuda() * dist.get_rank()

        output_data = _GatherFromSequenceParallelRegion.symbolic(
            None, input_data, tp_group
        )
        parts = [paddle.ones([4]).cuda() * r for r in range(TP_SIZE)]
        expected = paddle.concat(parts)
        self.assertTrue(paddle.equal_all(output_data, expected))

        # Backward: scatter along sequence dim
        full_input = paddle.concat(
            [paddle.ones([4]).cuda() * r for r in range(TP_SIZE)]
        )

        class Ctx:
            tensor_parallel_output_grad = True
            output_split_sizes = None
            group = tp_group
            use_global_buffer = False

        output_backward = _GatherFromSequenceParallelRegion.backward(
            Ctx(), full_input
        )
        rank = dist.get_rank()
        expected_backward = paddle.ones([1, 4]).cuda() * TP_SIZE * rank
        self.assertTrue(paddle.equal_all(output_backward, expected_backward))


class TestReduceScatterToSequenceParallelRegion(unittest.TestCase):
    """Test _ReduceScatterToSequenceParallelRegion reduce-scatters along seq dim."""

    def test_reduce_scatter_to_sequence_parallel_region(self):
        """Reduce-scatter should reduce then scatter along sequence dim."""
        tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)

        full_input = paddle.concat(
            [paddle.ones([4]).cuda() * r for r in range(TP_SIZE)]
        )

        output_data = _ReduceScatterToSequenceParallelRegion.symbolic(
            None, full_input, tp_group
        )
        rank = dist.get_rank()
        expected = paddle.ones([1, 4]).cuda() * TP_SIZE * rank
        self.assertTrue(paddle.equal_all(output_data, expected))

        # Backward: gather
        input_data = paddle.ones([4]).cuda() * dist.get_rank()

        class Ctx:
            input_split_sizes = None
            group = tp_group
            use_global_buffer = False

        output_backward = _ReduceScatterToSequenceParallelRegion.backward(
            Ctx(), input_data
        )
        parts = [paddle.ones([4]).cuda() * r for r in range(TP_SIZE)]
        expected_backward = paddle.concat(parts)
        self.assertTrue(paddle.equal_all(output_backward, expected_backward))


class TestAllGatherFromTensorParallelRegion(unittest.TestCase):
    """Test _AllGatherFromTensorParallelRegion gathers along last dim."""

    def test_all_gather_from_tensor_parallel_region(self):
        """Forward should gather along last dim; backward should reduce-scatter."""
        tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)

        # Forward: gather along last dim
        input_data = paddle.ones([4, 4]).cuda() * dist.get_rank()
        output = _AllGatherFromTensorParallelRegion.symbolic(
            None, input_data, tp_group
        )
        # Output should have TP_SIZE times the last dim
        self.assertEqual(output.shape[-1], 4 * TP_SIZE)

        # Backward: reduce-scatter along last dim
        class Ctx:
            group = tp_group

        grad_output = paddle.randn([4, 16]).cuda()
        grad_input = _AllGatherFromTensorParallelRegion.backward(
            Ctx(), grad_output
        )
        # Output should have last dim divided by TP_SIZE
        self.assertEqual(grad_input.shape[-1], 16 // TP_SIZE)


class TestReduceScatterToTensorParallelRegion(unittest.TestCase):
    """Test _ReduceScatterToTensorParallelRegion reduce-scatters along last dim."""

    def test_reduce_scatter_to_tensor_parallel_region(self):
        """Forward should reduce-scatter; backward should gather along last dim."""
        tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)

        # Forward: reduce-scatter along last dim
        input_data = paddle.randn([4, 16]).cuda()
        output = _ReduceScatterToTensorParallelRegion.symbolic(
            None, input_data, tp_group
        )
        # Output should have last dim divided by TP_SIZE
        self.assertEqual(output.shape[-1], 16 // TP_SIZE)

        # Backward: gather along last dim
        class Ctx:
            group = tp_group

        grad_output = paddle.randn([4, 4]).cuda()
        grad_input = _ReduceScatterToTensorParallelRegion.backward(
            Ctx(), grad_output
        )
        # Output should have TP_SIZE times the last dim
        self.assertEqual(grad_input.shape[-1], 4 * TP_SIZE)


if __name__ == "__main__":
    from tests.multi_card_tests.tensor_parallel.test_utilities import Utils

    Utils.initialize_model_parallel(
        tensor_parallel_size=TP_SIZE, pipeline_parallel_size=1
    )
    unittest.main()
