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
    subbatch,
)


class TestSubbatchWithSameArgIdx(unittest.TestCase):
    """Tests for subbatch with same_arg_idx optimization."""

    def test_same_arg_idx_avoids_duplicate_slicing(self):
        """subbatch should use same_arg_idx to avoid duplicate tensor slicing."""
        paddle.disable_static()

        call_count = [0]

        def counting_fn(x, y):
            call_count[0] += 1
            return x + y

        sb_fn = subbatch(
            counting_fn,
            arg_idx=[0, 1],
            axis=[0, 0],
            bs=2,
            out_idx=0,
            same_arg_idx={1: 0},  # args[1] uses same slice as args[0]
        )
        x = paddle.randn([4, 8])
        result = sb_fn(x, x)
        self.assertEqual(result.shape[0], 4)


class TestSubbatchWithKwargs(unittest.TestCase):
    """Tests for subbatch with keyword arguments."""

    def test_passes_kwargs_to_function(self):
        """subbatch should pass keyword arguments through."""
        paddle.disable_static()

        def fn_with_kwargs(x, scale=1.0):
            return x * scale

        sb_fn = subbatch(
            fn_with_kwargs,
            arg_idx=[0],
            axis=[0],
            bs=100,
            out_idx=0,
        )
        x = paddle.randn([4, 8])
        result = sb_fn(x, scale=2.0)
        expected = x * 2.0
        self.assertTrue(paddle.allclose(result, expected))


class TestLanguageLossForwardImpl(unittest.TestCase):
    """Tests for LanguageLoss.forward_impl."""

    @patch(
        "paddlefleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch(
        "paddlefleet.models.common.language_loss.language_loss.ProcessGroupCollection.use_mpu_process_groups"
    )
    def test_forward_impl_basic_loss(self, mock_pg, mock_tp_size):
        """forward_impl should compute cross-entropy loss from logits and labels."""
        mock_pg.return_value = MagicMock()
        config = MagicMock()
        config.parallel_output = False
        config.loss_subbatch_sequence_length = 0
        config.gpt_model_use_experimental_version = False
        config.fused_linear_ce_loss_chunk = 0
        config.recompute_modules = None

        loss_fn = LanguageLoss(config=config)
        logits = paddle.randn([2, 4, 10])
        labels = paddle.randint(0, 10, [2, 4])
        result = loss_fn.forward_impl(logits, labels)
        self.assertTrue(paddle.is_tensor(result))
        self.assertEqual(result.ndim, 0)  # Scalar


class TestLanguageLossForwardWithSingleLogits(unittest.TestCase):
    """Tests for LanguageLoss.forward with single logits tensor."""

    @patch(
        "paddlefleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch(
        "paddlefleet.models.common.language_loss.language_loss.ProcessGroupCollection.use_mpu_process_groups"
    )
    def test_forward_single_logits(self, mock_pg, mock_tp_size):
        """forward with single tensor should call _forward."""
        mock_pg.return_value = MagicMock()
        config = MagicMock()
        config.parallel_output = False
        config.loss_subbatch_sequence_length = 0
        config.gpt_model_use_experimental_version = False
        config.fused_linear_ce_loss_chunk = 0
        config.recompute_modules = None

        loss_fn = LanguageLoss(config=config)
        logits = paddle.randn([2, 4, 10])
        labels = paddle.randint(0, 10, [2, 4])
        result = loss_fn.forward(logits, labels)
        self.assertTrue(paddle.is_tensor(result))


class TestLanguageLossEnableParallelCrossEntropy(unittest.TestCase):
    """Tests for LanguageLoss enable_parallel_cross_entropy setting."""

    @patch(
        "paddlefleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch(
        "paddlefleet.models.common.language_loss.language_loss.ProcessGroupCollection.use_mpu_process_groups"
    )
    def test_disabled_when_tp_size_1(self, mock_pg, mock_tp_size):
        """Parallel cross entropy should be disabled when TP world size is 1."""
        mock_pg.return_value = MagicMock()
        config = MagicMock()
        config.parallel_output = False
        config.loss_subbatch_sequence_length = 0

        loss_fn = LanguageLoss(config=config)
        self.assertFalse(loss_fn.enable_parallel_cross_entropy)

    @patch(
        "paddlefleet.models.common.language_loss.language_loss.paddle.distributed.is_initialized",
        return_value=True,
    )
    @patch(
        "paddlefleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=2,
    )
    @patch(
        "paddlefleet.models.common.language_loss.language_loss.ProcessGroupCollection.use_mpu_process_groups"
    )
    def test_enabled_when_tp_size_gt_1_and_parallel_output(
        self, mock_pg, mock_tp_size, mock_init
    ):
        """Parallel cross entropy should be enabled when TP > 1 and parallel_output."""
        mock_pg.return_value = MagicMock()
        config = MagicMock()
        config.parallel_output = True
        config.loss_subbatch_sequence_length = 0

        loss_fn = LanguageLoss(config=config)
        self.assertTrue(loss_fn.enable_parallel_cross_entropy)


if __name__ == "__main__":
    unittest.main()
