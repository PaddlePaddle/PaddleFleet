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

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)


import unittest
from unittest.mock import MagicMock, patch


class TestPipelineUtils(unittest.TestCase):
    """Unit tests for pipeline_parallel/utils.py"""

    def test_is_pp_first_stage_true(self):
        from paddlefleet.pipeline_parallel.utils import is_pp_first_stage

        mock_group = MagicMock()
        with patch(
            "paddlefleet.pipeline_parallel.utils.get_pg_rank", return_value=0
        ):
            self.assertTrue(is_pp_first_stage(mock_group))

    def test_is_pp_first_stage_false(self):
        from paddlefleet.pipeline_parallel.utils import is_pp_first_stage

        mock_group = MagicMock()
        with patch(
            "paddlefleet.pipeline_parallel.utils.get_pg_rank", return_value=2
        ):
            self.assertFalse(is_pp_first_stage(mock_group))

    def test_is_pp_last_stage_true(self):
        from paddlefleet.pipeline_parallel.utils import is_pp_last_stage

        mock_group = MagicMock()
        with (
            patch(
                "paddlefleet.pipeline_parallel.utils.get_pg_rank",
                return_value=3,
            ),
            patch(
                "paddlefleet.pipeline_parallel.utils.get_pg_size",
                return_value=4,
            ),
        ):
            self.assertTrue(is_pp_last_stage(mock_group))

    def test_is_pp_last_stage_false(self):
        from paddlefleet.pipeline_parallel.utils import is_pp_last_stage

        mock_group = MagicMock()
        with (
            patch(
                "paddlefleet.pipeline_parallel.utils.get_pg_rank",
                return_value=1,
            ),
            patch(
                "paddlefleet.pipeline_parallel.utils.get_pg_size",
                return_value=4,
            ),
        ):
            self.assertFalse(is_pp_last_stage(mock_group))

    def test_is_vp_first_stage_none_size(self):
        from paddlefleet.pipeline_parallel.utils import is_vp_first_stage

        self.assertTrue(is_vp_first_stage(None, None))

    def test_is_vp_first_stage_size_one(self):
        from paddlefleet.pipeline_parallel.utils import is_vp_first_stage

        self.assertTrue(is_vp_first_stage(0, 1))

    def test_is_vp_first_stage_size_one_invalid_stage(self):
        from paddlefleet.pipeline_parallel.utils import is_vp_first_stage

        with self.assertRaises(AssertionError):
            is_vp_first_stage(1, 1)

    def test_is_vp_first_stage_rank_zero(self):
        from paddlefleet.pipeline_parallel.utils import is_vp_first_stage

        self.assertTrue(is_vp_first_stage(0, 4))

    def test_is_vp_first_stage_rank_nonzero(self):
        from paddlefleet.pipeline_parallel.utils import is_vp_first_stage

        self.assertFalse(is_vp_first_stage(2, 4))

    def test_is_vp_last_stage_none_size(self):
        from paddlefleet.pipeline_parallel.utils import is_vp_last_stage

        self.assertTrue(is_vp_last_stage(None, None))

    def test_is_vp_last_stage_size_one(self):
        from paddlefleet.pipeline_parallel.utils import is_vp_last_stage

        self.assertTrue(is_vp_last_stage(0, 1))

    def test_is_vp_last_stage_rank_last(self):
        from paddlefleet.pipeline_parallel.utils import is_vp_last_stage

        self.assertTrue(is_vp_last_stage(3, 4))

    def test_is_vp_last_stage_rank_not_last(self):
        from paddlefleet.pipeline_parallel.utils import is_vp_last_stage

        self.assertFalse(is_vp_last_stage(1, 4))

    def test_get_pp_first_rank(self):
        from paddlefleet.pipeline_parallel.utils import get_pp_first_rank

        mock_group = MagicMock()
        mock_group.ranks.return_value = [5, 10, 15]
        self.assertEqual(get_pp_first_rank(mock_group), 5)

    def test_get_pp_last_rank(self):
        from paddlefleet.pipeline_parallel.utils import get_pp_last_rank

        mock_group = MagicMock()
        mock_group.ranks.return_value = [5, 10, 15]
        self.assertEqual(get_pp_last_rank(mock_group), 15)

    def test_get_pp_next_rank_last_stage(self):
        from paddlefleet.pipeline_parallel.utils import get_pp_next_rank

        mock_group = MagicMock()
        with patch(
            "paddlefleet.pipeline_parallel.utils.is_pp_last_stage",
            return_value=True,
        ):
            self.assertIsNone(get_pp_next_rank(mock_group))

    def test_get_pp_next_rank_not_last(self):
        from paddlefleet.pipeline_parallel.utils import get_pp_next_rank

        mock_group = MagicMock()
        mock_group.ranks.return_value = [5, 10, 15]
        with (
            patch(
                "paddlefleet.pipeline_parallel.utils.is_pp_last_stage",
                return_value=False,
            ),
            patch(
                "paddlefleet.pipeline_parallel.utils.get_pg_rank",
                return_value=1,
            ),
        ):
            self.assertEqual(get_pp_next_rank(mock_group), 15)

    def test_get_pp_prev_rank_first_stage(self):
        from paddlefleet.pipeline_parallel.utils import get_pp_prev_rank

        mock_group = MagicMock()
        with patch(
            "paddlefleet.pipeline_parallel.utils.is_pp_first_stage",
            return_value=True,
        ):
            self.assertIsNone(get_pp_prev_rank(mock_group))

    def test_get_pp_prev_rank_not_first(self):
        from paddlefleet.pipeline_parallel.utils import get_pp_prev_rank

        mock_group = MagicMock()
        mock_group.ranks.return_value = [5, 10, 15]
        with (
            patch(
                "paddlefleet.pipeline_parallel.utils.is_pp_first_stage",
                return_value=False,
            ),
            patch(
                "paddlefleet.pipeline_parallel.utils.get_pg_rank",
                return_value=1,
            ),
        ):
            self.assertEqual(get_pp_prev_rank(mock_group), 5)

    def test_make_viewless(self):
        from paddlefleet.pipeline_parallel.utils import make_viewless

        mock_tensor = MagicMock()
        mock_tensor.requires_grad = True
        with patch(
            "paddlefleet.pipeline_parallel.utils.make_viewless_tensor",
            return_value=mock_tensor,
        ) as mock_fn:
            result = make_viewless(mock_tensor)
            mock_fn.assert_called_once()
            self.assertEqual(result, mock_tensor)

    def test_noop_schedule_node_forward(self):
        from paddlefleet.pipeline_parallel.utils import NoopScheduleNode

        node = NoopScheduleNode()
        inputs = "test_input"
        self.assertEqual(node.forward(inputs), "test_input")

    def test_noop_schedule_node_backward(self):
        from paddlefleet.pipeline_parallel.utils import NoopScheduleNode

        node = NoopScheduleNode()
        outgrads = "test_grad"
        self.assertEqual(node.backward(outgrads), "test_grad")

    def test_stream_acquire_context(self):
        from paddlefleet.pipeline_parallel.utils import stream_acquire_context

        mock_stream = MagicMock()
        mock_event = MagicMock()
        with stream_acquire_context(mock_stream, mock_event):
            pass
        mock_event.wait.assert_called_once_with(mock_stream)
        mock_event.record.assert_called_once_with(mock_stream)

    def test_schedule_node_init(self):
        from paddlefleet.pipeline_parallel.utils import ScheduleNode

        mock_stream = MagicMock()
        mock_event = MagicMock()
        fwd_func = MagicMock()
        node = ScheduleNode(
            forward_func=fwd_func,
            stream=mock_stream,
            event=mock_event,
        )
        self.assertEqual(node.name, "schedule_node")
        self.assertEqual(node.free_input, False)
        self.assertIsNone(node.inputs)
        self.assertIsNone(node.outputs)

    def test_schedule_node_init_with_name(self):
        from paddlefleet.pipeline_parallel.utils import ScheduleNode

        mock_stream = MagicMock()
        mock_event = MagicMock()
        fwd_func = MagicMock()
        node = ScheduleNode(
            forward_func=fwd_func,
            stream=mock_stream,
            event=mock_event,
            name="my_node",
        )
        self.assertEqual(node.name, "my_node")

    def test_schedule_node_reset_states(self):
        from paddlefleet.pipeline_parallel.utils import ScheduleNode

        mock_stream = MagicMock()
        mock_event = MagicMock()
        fwd_func = MagicMock()
        node = ScheduleNode(
            forward_func=fwd_func,
            stream=mock_stream,
            event=mock_event,
        )
        node.inputs = [MagicMock()]
        node.outputs = [MagicMock()]
        node._reset_states()
        self.assertIsNone(node.inputs)
        self.assertIsNone(node.outputs)

    def test_schedule_node_get_output(self):
        from paddlefleet.pipeline_parallel.utils import ScheduleNode

        mock_stream = MagicMock()
        mock_event = MagicMock()
        fwd_func = MagicMock()
        node = ScheduleNode(
            forward_func=fwd_func,
            stream=mock_stream,
            event=mock_event,
        )
        node.output = "test_output"
        self.assertEqual(node.get_output(), "test_output")

    def test_abstract_schedule_plan(self):
        from paddlefleet.pipeline_parallel.utils import AbstractSchedulePlan

        with self.assertRaises(TypeError):
            AbstractSchedulePlan()

    def test_set_streams_both_none(self):
        from paddlefleet.pipeline_parallel.utils import set_streams

        mock_comp_stream = MagicMock()
        mock_comm_stream = MagicMock()
        with (
            patch("paddle.cuda.current_stream", return_value=mock_comp_stream),
            patch("paddle.cuda.Stream", return_value=mock_comm_stream),
        ):
            set_streams()

    def test_set_streams_already_set(self):
        from paddlefleet.pipeline_parallel.utils import (
            set_streams,
        )

        mock_comp = MagicMock()
        mock_comm = MagicMock()
        with (
            patch(
                "paddlefleet.pipeline_parallel.utils._COMP_STREAM", mock_comp
            ),
            patch(
                "paddlefleet.pipeline_parallel.utils._COMM_STREAM", mock_comm
            ),
        ):
            set_streams()

    def test_get_comp_stream(self):
        from paddlefleet.pipeline_parallel.utils import get_comp_stream

        mock_stream = MagicMock()
        with patch(
            "paddlefleet.pipeline_parallel.utils._COMP_STREAM", mock_stream
        ):
            self.assertEqual(get_comp_stream(), mock_stream)

    def test_get_comm_stream(self):
        from paddlefleet.pipeline_parallel.utils import get_comm_stream

        mock_stream = MagicMock()
        with patch(
            "paddlefleet.pipeline_parallel.utils._COMM_STREAM", mock_stream
        ):
            self.assertEqual(get_comm_stream(), mock_stream)


if __name__ == "__main__":
    unittest.main()
