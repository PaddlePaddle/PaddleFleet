# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless distributed on the License is distributed on an "AS IS" BASIS,
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

import paddle

from paddlefleet.models.common.language_loss.language_loss import (
    LanguageLoss,
    MainLanguageLoss,
    MTPLanguageLoss,
    subbatch,
)


class TestSubbatchFunction(unittest.TestCase):
    """Tests for subbatch wrapper function."""

    def test_subbatch_returns_original_when_small_input(self):
        """subbatch should call original function when input size < bs."""

        def identity_fn(x):
            return x

        sb_fn = subbatch(
            identity_fn,
            arg_idx=[0],
            axis=[0],
            bs=100,
            out_idx=0,
        )
        x = paddle.randn([4, 8])
        result = sb_fn(x)
        self.assertTrue(paddle.equal(result, x))

    def test_subbatch_mismatched_arg_and_axis_lengths(self):
        """subbatch should raise when arg_idx and axis lengths don't match."""

        def dummy_fn(x):
            return x

        with self.assertRaises(AssertionError):
            subbatch(
                dummy_fn,
                arg_idx=[0],
                axis=[0, 1],  # Mismatched
                bs=4,
                out_idx=0,
            )

    def test_subbatch_wraps_function(self):
        """subbatch should preserve function name via functools.wraps."""

        def my_function(x):
            return x

        sb_fn = subbatch(my_function, arg_idx=[0], axis=[0], bs=100, out_idx=0)
        self.assertEqual(sb_fn.__name__, "my_function")


class TestLanguageLossInit(unittest.TestCase):
    """Tests for LanguageLoss initialization."""

    def test_default_ignored_index(self):
        """LanguageLoss should use -100 as ignored_index."""
        with patch.object(
            LanguageLoss, "__init__", lambda self, *a, **kw: None
        ):
            loss = LanguageLoss.__new__(LanguageLoss)
            loss.ignored_index = -100
            self.assertEqual(loss.ignored_index, -100)


class TestLanguageLossBuildScheduleNode(unittest.TestCase):
    """Tests for LanguageLoss.build_schedule_node."""

    def test_returns_schedule_node(self):
        """build_schedule_node should return a ScheduleNode."""
        from paddle.distributed.fleet.meta_parallel import ScheduleNode

        with patch.object(
            LanguageLoss, "__init__", lambda self, *a, **kw: None
        ):
            loss = LanguageLoss.__new__(LanguageLoss)
            loss.forward = MagicMock()
            node = loss.build_schedule_node()
            self.assertIsInstance(node, ScheduleNode)


class TestMainLanguageLossBuildScheduleNode(unittest.TestCase):
    """Tests for MainLanguageLoss.build_schedule_node."""

    def test_returns_schedule_node(self):
        """build_schedule_node should return a ScheduleNode with correct name."""
        from paddle.distributed.fleet.meta_parallel import ScheduleNode

        with patch.object(
            MainLanguageLoss, "__init__", lambda self, *a, **kw: None
        ):
            loss = MainLanguageLoss.__new__(MainLanguageLoss)
            loss.forward = MagicMock()
            node = loss.build_schedule_node()
            self.assertIsInstance(node, ScheduleNode)


class TestMTPLanguageLossBuildScheduleNode(unittest.TestCase):
    """Tests for MTPLanguageLoss.build_schedule_node."""

    def test_returns_schedule_node(self):
        """build_schedule_node should return a ScheduleNode."""
        from paddle.distributed.fleet.meta_parallel import ScheduleNode

        with patch.object(
            MTPLanguageLoss, "__init__", lambda self, *a, **kw: None
        ):
            loss = MTPLanguageLoss.__new__(MTPLanguageLoss)
            loss.forward = MagicMock()
            node = loss.build_schedule_node()
            self.assertIsInstance(node, ScheduleNode)


class TestLanguageLossMTPTracker(unittest.TestCase):
    """Tests for LanguageLoss MTP loss tracker."""

    def test_mtp_loss_tracker_is_class_attribute(self):
        """mtp_loss_tracker should be a class-level dict."""
        self.assertIsInstance(LanguageLoss.mtp_loss_tracker, dict)

    def test_main_language_loss_tracker_is_class_attribute(self):
        """MainLanguageLoss should have its own mtp_loss_tracker."""
        self.assertIsInstance(MainLanguageLoss.mtp_loss_tracker, dict)


if __name__ == "__main__":
    unittest.main()
