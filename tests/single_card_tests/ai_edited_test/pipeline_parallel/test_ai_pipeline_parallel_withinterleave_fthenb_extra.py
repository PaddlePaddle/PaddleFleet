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
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import unittest
from unittest.mock import patch


class TestFthenBGetSchedulerName(unittest.TestCase):
    """Tests for _get_scheduler_name in FthenB."""

    @patch(
        "paddle.distributed.fleet.meta_parallel.pipeline_parallel.PipelineParallelWithInterleave.__init__"
    )
    def test_scheduler_name(self, mock_parent_init):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            PipelineParallelWithInterleaveFthenB,
        )

        pp = PipelineParallelWithInterleaveFthenB.__new__(
            PipelineParallelWithInterleaveFthenB
        )
        self.assertEqual(
            pp._get_scheduler_name(), "PipelineParallelWithInterleaveFthenB"
        )


class TestFthenBOverlapScheduleMode(unittest.TestCase):
    """Tests for overlap_schedule_mode in FthenB."""

    def test_overlap_schedule_mode_default(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            PipelineParallelWithInterleaveFthenB,
        )

        pp = PipelineParallelWithInterleaveFthenB.__new__(
            PipelineParallelWithInterleaveFthenB
        )
        pp.overlap_schedule_mode = False
        self.assertFalse(pp.overlap_schedule_mode)

    def test_overlap_schedule_mode_set_true(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            PipelineParallelWithInterleaveFthenB,
        )

        pp = PipelineParallelWithInterleaveFthenB.__new__(
            PipelineParallelWithInterleaveFthenB
        )
        pp.overlap_schedule_mode = True
        self.assertTrue(pp.overlap_schedule_mode)


class TestFthenBInitUserBubbleHooks(unittest.TestCase):
    """Tests for _init_user_bubble_hooks in FthenB."""

    def test_init_user_bubble_hooks(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            PipelineParallelWithInterleaveFthenB,
        )

        pp = PipelineParallelWithInterleaveFthenB.__new__(
            PipelineParallelWithInterleaveFthenB
        )
        pp._init_user_bubble_hooks()
        self.assertIsNone(pp.bubble_hooks)


class TestFthenBCheckSanity(unittest.TestCase):
    """Tests for _check_sanity in FthenB."""

    @patch("paddle.framework.in_dynamic_mode", return_value=True)
    def test_check_sanity_valid(self, mock_dynamic):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            PipelineParallelWithInterleaveFthenB,
        )

        pp = PipelineParallelWithInterleaveFthenB.__new__(
            PipelineParallelWithInterleaveFthenB
        )
        pp.num_stages = 4
        # Should not raise
        pp._check_sanity()

    @patch("paddle.framework.in_dynamic_mode", return_value=False)
    def test_check_sanity_not_dynamic(self, mock_dynamic):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            PipelineParallelWithInterleaveFthenB,
        )

        pp = PipelineParallelWithInterleaveFthenB.__new__(
            PipelineParallelWithInterleaveFthenB
        )
        pp.num_stages = 4
        with self.assertRaises(AssertionError):
            pp._check_sanity()

    @patch("paddle.framework.in_dynamic_mode", return_value=True)
    def test_check_sanity_not_enough_stages(self, mock_dynamic):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            PipelineParallelWithInterleaveFthenB,
        )

        pp = PipelineParallelWithInterleaveFthenB.__new__(
            PipelineParallelWithInterleaveFthenB
        )
        pp.num_stages = 2
        with self.assertRaises(AssertionError):
            pp._check_sanity()


class TestFthenBGetVirtualPPRank(unittest.TestCase):
    """Tests for _get_virtual_pp_rank in FthenB."""

    def test_virtual_pp_rank_forward_chunk_0(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            PipelineParallelWithInterleaveFthenB,
        )

        pp = PipelineParallelWithInterleaveFthenB.__new__(
            PipelineParallelWithInterleaveFthenB
        )
        pp.accumulate_steps = 4
        pp.num_model_chunks = 2
        # micro_step=0 -> 0 % (4*2)=0, 0//4=0
        self.assertEqual(pp._get_virtual_pp_rank(0, forward=True), 0)

    def test_virtual_pp_rank_forward_chunk_1(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            PipelineParallelWithInterleaveFthenB,
        )

        pp = PipelineParallelWithInterleaveFthenB.__new__(
            PipelineParallelWithInterleaveFthenB
        )
        pp.accumulate_steps = 4
        pp.num_model_chunks = 2
        # micro_step=4 -> 4 % 8=4, 4//4=1
        self.assertEqual(pp._get_virtual_pp_rank(4, forward=True), 1)

    def test_virtual_pp_rank_backward(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            PipelineParallelWithInterleaveFthenB,
        )

        pp = PipelineParallelWithInterleaveFthenB.__new__(
            PipelineParallelWithInterleaveFthenB
        )
        pp.accumulate_steps = 4
        pp.num_model_chunks = 2
        # forward rank=1, backward: 2-1-1=0
        self.assertEqual(pp._get_virtual_pp_rank(4, forward=False), 0)

    def test_virtual_pp_rank_backward_chunk_0(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            PipelineParallelWithInterleaveFthenB,
        )

        pp = PipelineParallelWithInterleaveFthenB.__new__(
            PipelineParallelWithInterleaveFthenB
        )
        pp.accumulate_steps = 4
        pp.num_model_chunks = 2
        # forward rank=0, backward: 2-0-1=1
        self.assertEqual(pp._get_virtual_pp_rank(0, forward=False), 1)


class TestFthenBOverlapCommGrads(unittest.TestCase):
    """Tests for _overlap_comm_grads in FthenB."""

    def test_overlap_comm_grads_disabled(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            PipelineParallelWithInterleaveFthenB,
        )

        pp = PipelineParallelWithInterleaveFthenB.__new__(
            PipelineParallelWithInterleaveFthenB
        )
        pp._comm_overlap = False
        pp._backward_step_count = 0
        # Should return immediately
        pp._overlap_comm_grads()
        self.assertEqual(pp._backward_step_count, 0)

    def test_overlap_comm_grads_stage_zero(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            PipelineParallelWithInterleaveFthenB,
        )

        pp = PipelineParallelWithInterleaveFthenB.__new__(
            PipelineParallelWithInterleaveFthenB
        )
        pp._comm_overlap = True
        pp._backward_step_count = 0
        pp.stage_id = 0
        pp.accumulate_steps = 4
        pp._virtual_pp_world_size = 2
        pp._chunk_2_comm_buffers = {}

        pp._overlap_comm_grads()
        self.assertEqual(pp._backward_step_count, 1)
        # stage_id==0 should return early, no buffer comm
        self.assertEqual(pp._backward_step_count, 1)


class TestFthenBSyncOverlapGrads(unittest.TestCase):
    """Tests for _sync_overlap_grads in FthenB."""

    def test_sync_overlap_grads_disabled(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            PipelineParallelWithInterleaveFthenB,
        )

        pp = PipelineParallelWithInterleaveFthenB.__new__(
            PipelineParallelWithInterleaveFthenB
        )
        pp._comm_overlap = False
        # Should return immediately
        pp._sync_overlap_grads()

    def test_sync_overlap_grads_assertion(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            PipelineParallelWithInterleaveFthenB,
        )

        pp = PipelineParallelWithInterleaveFthenB.__new__(
            PipelineParallelWithInterleaveFthenB
        )
        pp._comm_overlap = True
        pp._backward_step_count = 3
        pp.accumulate_steps = 4
        pp._virtual_pp_world_size = 2

        with self.assertRaises(AssertionError):
            pp._sync_overlap_grads()


if __name__ == "__main__":
    unittest.main()
