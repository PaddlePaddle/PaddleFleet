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
from unittest.mock import MagicMock, patch


class TestIsPpFirstStage(unittest.TestCase):
    """Tests for is_pp_first_stage."""

    @patch("paddlefleet.pipeline_parallel.utils.get_pg_rank", return_value=0)
    def test_first_stage(self, mock_rank):
        from paddlefleet.pipeline_parallel.utils import is_pp_first_stage

        mock_group = MagicMock()
        self.assertTrue(is_pp_first_stage(mock_group))

    @patch("paddlefleet.pipeline_parallel.utils.get_pg_rank", return_value=1)
    def test_not_first_stage(self, mock_rank):
        from paddlefleet.pipeline_parallel.utils import is_pp_first_stage

        mock_group = MagicMock()
        self.assertFalse(is_pp_first_stage(mock_group))


class TestIsPpLastStage(unittest.TestCase):
    """Tests for is_pp_last_stage."""

    @patch("paddlefleet.pipeline_parallel.utils.get_pg_rank", return_value=2)
    @patch("paddlefleet.pipeline_parallel.utils.get_pg_size", return_value=3)
    def test_last_stage(self, mock_size, mock_rank):
        from paddlefleet.pipeline_parallel.utils import is_pp_last_stage

        mock_group = MagicMock()
        self.assertTrue(is_pp_last_stage(mock_group))

    @patch("paddlefleet.pipeline_parallel.utils.get_pg_rank", return_value=1)
    @patch("paddlefleet.pipeline_parallel.utils.get_pg_size", return_value=3)
    def test_not_last_stage(self, mock_size, mock_rank):
        from paddlefleet.pipeline_parallel.utils import is_pp_last_stage

        mock_group = MagicMock()
        self.assertFalse(is_pp_last_stage(mock_group))


class TestIsVpFirstStage(unittest.TestCase):
    """Tests for is_vp_first_stage."""

    def test_vp_size_none(self):
        from paddlefleet.pipeline_parallel.utils import is_vp_first_stage

        self.assertTrue(is_vp_first_stage(0, None))

    def test_vp_size_one(self):
        from paddlefleet.pipeline_parallel.utils import is_vp_first_stage

        self.assertTrue(is_vp_first_stage(0, 1))

    def test_vp_stage_zero(self):
        from paddlefleet.pipeline_parallel.utils import is_vp_first_stage

        self.assertTrue(is_vp_first_stage(0, 4))

    def test_vp_stage_nonzero(self):
        from paddlefleet.pipeline_parallel.utils import is_vp_first_stage

        self.assertFalse(is_vp_first_stage(2, 4))

    def test_vp_size_none_with_none_stage(self):
        from paddlefleet.pipeline_parallel.utils import is_vp_first_stage

        self.assertTrue(is_vp_first_stage(None, None))

    def test_vp_size_none_invalid_stage(self):
        from paddlefleet.pipeline_parallel.utils import is_vp_first_stage

        with self.assertRaises(AssertionError):
            is_vp_first_stage(1, None)


class TestIsVpLastStage(unittest.TestCase):
    """Tests for is_vp_last_stage."""

    def test_vp_size_none(self):
        from paddlefleet.pipeline_parallel.utils import is_vp_last_stage

        self.assertTrue(is_vp_last_stage(0, None))

    def test_vp_last(self):
        from paddlefleet.pipeline_parallel.utils import is_vp_last_stage

        self.assertTrue(is_vp_last_stage(3, 4))

    def test_vp_not_last(self):
        from paddlefleet.pipeline_parallel.utils import is_vp_last_stage

        self.assertFalse(is_vp_last_stage(1, 4))

    def test_vp_size_none_invalid_stage(self):
        from paddlefleet.pipeline_parallel.utils import is_vp_last_stage

        with self.assertRaises(AssertionError):
            is_vp_last_stage(1, None)


class TestGetPpFirstRank(unittest.TestCase):
    """Tests for get_pp_first_rank."""

    @patch("paddlefleet.pipeline_parallel.utils.get_pg_rank")
    def test_first_rank(self, mock_rank):
        from paddlefleet.pipeline_parallel.utils import get_pp_first_rank

        mock_group = MagicMock()
        mock_group.ranks.return_value = [5, 6, 7]
        self.assertEqual(get_pp_first_rank(mock_group), 5)


class TestGetPpLastRank(unittest.TestCase):
    """Tests for get_pp_last_rank."""

    @patch("paddlefleet.pipeline_parallel.utils.get_pg_rank")
    def test_last_rank(self, mock_rank):
        from paddlefleet.pipeline_parallel.utils import get_pp_last_rank

        mock_group = MagicMock()
        mock_group.ranks.return_value = [5, 6, 7]
        self.assertEqual(get_pp_last_rank(mock_group), 7)


class TestGetPpNextRank(unittest.TestCase):
    """Tests for get_pp_next_rank."""

    @patch("paddlefleet.pipeline_parallel.utils.get_pg_size", return_value=4)
    @patch("paddlefleet.pipeline_parallel.utils.get_pg_rank", return_value=1)
    def test_next_rank(self, mock_rank, mock_size):
        from paddlefleet.pipeline_parallel.utils import get_pp_next_rank

        mock_group = MagicMock()
        mock_group.ranks.return_value = [10, 11, 12, 13]
        self.assertEqual(get_pp_next_rank(mock_group), 12)

    @patch(
        "paddlefleet.pipeline_parallel.utils.is_pp_last_stage",
        return_value=True,
    )
    def test_next_rank_last_stage(self, mock_last):
        from paddlefleet.pipeline_parallel.utils import get_pp_next_rank

        mock_group = MagicMock()
        self.assertIsNone(get_pp_next_rank(mock_group))


class TestGetPpPrevRank(unittest.TestCase):
    """Tests for get_pp_prev_rank."""

    @patch(
        "paddlefleet.pipeline_parallel.utils.is_pp_first_stage",
        return_value=False,
    )
    @patch("paddlefleet.pipeline_parallel.utils.get_pg_rank", return_value=2)
    def test_prev_rank(self, mock_rank, mock_first):
        from paddlefleet.pipeline_parallel.utils import get_pp_prev_rank

        mock_group = MagicMock()
        mock_group.ranks.return_value = [10, 11, 12, 13]
        self.assertEqual(get_pp_prev_rank(mock_group), 11)

    @patch(
        "paddlefleet.pipeline_parallel.utils.is_pp_first_stage",
        return_value=True,
    )
    def test_prev_rank_first_stage(self, mock_first):
        from paddlefleet.pipeline_parallel.utils import get_pp_prev_rank

        mock_group = MagicMock()
        self.assertIsNone(get_pp_prev_rank(mock_group))


class TestMakeViewless(unittest.TestCase):
    """Tests for make_viewless function."""

    @patch("paddlefleet.pipeline_parallel.utils.make_viewless_tensor")
    def test_make_viewless(self, mock_make):
        import paddle

        from paddlefleet.pipeline_parallel.utils import make_viewless

        mock_tensor = MagicMock(spec=paddle.Tensor)
        mock_tensor.requires_grad = True
        mock_result = MagicMock()
        mock_make.return_value = mock_result

        result = make_viewless(mock_tensor)
        mock_make.assert_called_once_with(
            inp=mock_tensor, requires_grad=True, keep_graph=True
        )
        self.assertEqual(result, mock_result)


class TestNoopScheduleNode(unittest.TestCase):
    """Tests for NoopScheduleNode."""

    def test_forward_passthrough(self):
        from paddlefleet.pipeline_parallel.utils import NoopScheduleNode

        node = NoopScheduleNode()
        self.assertEqual(node.forward("input"), "input")
        self.assertEqual(node.forward((1, 2)), (1, 2))
        self.assertEqual(node.forward(None), None)

    def test_backward_passthrough(self):
        from paddlefleet.pipeline_parallel.utils import NoopScheduleNode

        node = NoopScheduleNode()
        self.assertEqual(node.backward("grad"), "grad")
        self.assertEqual(node.backward((1, 2)), (1, 2))


class TestSetStreams(unittest.TestCase):
    """Tests for set_streams function."""

    @patch("paddle.cuda.Stream")
    @patch("paddle.cuda.current_stream")
    def test_set_streams_none_args(self, mock_current, mock_stream_cls):
        from paddlefleet.pipeline_parallel.utils import set_streams

        mock_current_stream = MagicMock()
        mock_current.return_value = mock_current_stream
        mock_new_stream = MagicMock()
        mock_stream_cls.return_value = mock_new_stream

        # Reset globals
        import paddlefleet.pipeline_parallel.utils as utils_module

        utils_module._COMP_STREAM = None
        utils_module._COMM_STREAM = None

        set_streams()
        self.assertEqual(utils_module._COMP_STREAM, mock_current_stream)
        self.assertEqual(utils_module._COMM_STREAM, mock_new_stream)

        # Reset globals
        utils_module._COMP_STREAM = None
        utils_module._COMM_STREAM = None

    def test_set_streams_already_set(self):
        import paddlefleet.pipeline_parallel.utils as utils_module
        from paddlefleet.pipeline_parallel.utils import set_streams

        utils_module._COMP_STREAM = MagicMock()
        utils_module._COMM_STREAM = MagicMock()

        set_streams()
        # Should return early, not change streams

        # Reset
        utils_module._COMP_STREAM = None
        utils_module._COMM_STREAM = None


class TestGetCompStreamGetCommStream(unittest.TestCase):
    """Tests for get_comp_stream and get_comm_stream."""

    def test_get_comp_stream_none(self):
        from paddlefleet.pipeline_parallel.utils import get_comp_stream

        result = get_comp_stream()
        self.assertIsNone(result)

    def test_get_comm_stream_none(self):
        from paddlefleet.pipeline_parallel.utils import get_comm_stream

        result = get_comm_stream()
        self.assertIsNone(result)


class TestAbstractSchedulePlan(unittest.TestCase):
    """Tests for AbstractSchedulePlan."""

    def test_cannot_instantiate(self):
        from paddlefleet.pipeline_parallel.utils import AbstractSchedulePlan

        with self.assertRaises(TypeError):
            AbstractSchedulePlan()

    def test_is_abstract(self):
        from paddlefleet.pipeline_parallel.utils import AbstractSchedulePlan

        self.assertTrue(hasattr(AbstractSchedulePlan, "run"))


if __name__ == "__main__":
    unittest.main()
