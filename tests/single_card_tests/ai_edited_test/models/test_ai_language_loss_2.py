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


class TestLanguageLossInit(unittest.TestCase):
    """Test LanguageLoss initialization."""

    @patch(
        "paddlefleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch("paddle.distributed.is_initialized", return_value=False)
    def test_init_no_parallel(self, mock_dist, mock_tp):
        from paddlefleet.models.common.language_loss.language_loss import (
            LanguageLoss,
        )

        mock_config = MagicMock()
        mock_config.parallel_output = True
        mock_config.loss_subbatch_sequence_length = 0

        loss = LanguageLoss(config=mock_config)
        self.assertFalse(loss.enable_parallel_cross_entropy)
        self.assertFalse(loss.use_subbatch)

    @patch(
        "paddlefleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=4,
    )
    @patch("paddle.distributed.is_initialized", return_value=True)
    @patch("paddle.distributed.fleet.meta_parallel.ParallelCrossEntropy")
    def test_init_with_parallel(self, mock_pce, mock_dist, mock_tp):
        from paddlefleet.models.common.language_loss.language_loss import (
            LanguageLoss,
        )

        mock_config = MagicMock()
        mock_config.parallel_output = True
        mock_config.loss_subbatch_sequence_length = 0

        loss = LanguageLoss(config=mock_config)
        self.assertTrue(loss.enable_parallel_cross_entropy)
        mock_pce.assert_called_once()

    @patch(
        "paddlefleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=4,
    )
    @patch("paddle.distributed.is_initialized", return_value=True)
    @patch("paddle.distributed.fleet.meta_parallel.ParallelCrossEntropy")
    @patch(
        "paddlefleet.models.common.language_loss.language_loss.ieee_kernel_enabled",
        return_value=True,
    )
    def test_init_with_parallel_uac_uses_vocab_parallel_ce(
        self, mock_uac, mock_pce, mock_dist, mock_tp
    ):
        from paddlefleet.models.common.language_loss.language_loss import (
            LanguageLoss,
            _uac_vocab_parallel_ce,
        )

        mock_config = MagicMock()
        mock_config.parallel_output = True
        mock_config.loss_subbatch_sequence_length = 0

        loss = LanguageLoss(config=mock_config)
        self.assertTrue(loss.enable_parallel_cross_entropy)
        self.assertIs(loss.loss_func, _uac_vocab_parallel_ce)
        mock_pce.assert_not_called()

    def test_uac_vocab_parallel_ce_transposes_batch_first_logits(self):
        import paddle

        from paddlefleet.models.common.language_loss.language_loss import (
            _uac_vocab_parallel_ce,
        )

        logits = paddle.randn([2, 3, 4])
        labels = paddle.randint(0, 4, [2, 3])
        with patch(
            "paddlefleet.tensor_parallel.cross_entropy.vocab_parallel_cross_entropy",
            side_effect=lambda lg, lb: lg.sum(axis=-1),
        ) as mock_ce:
            out = _uac_vocab_parallel_ce(logits, labels)
        mock_ce.assert_called_once()
        lg, lb = mock_ce.call_args.args
        self.assertEqual(list(lg.shape), [3, 2, 4])
        self.assertEqual(list(lb.shape), [3, 2])
        self.assertEqual(list(out.shape), [2, 3])

    @patch(
        "paddlefleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch("paddle.distributed.is_initialized", return_value=False)
    def test_init_with_subbatch(self, mock_dist, mock_tp):
        from paddlefleet.models.common.language_loss.language_loss import (
            LanguageLoss,
        )

        mock_config = MagicMock()
        mock_config.parallel_output = True
        mock_config.loss_subbatch_sequence_length = 512

        loss = LanguageLoss(config=mock_config)
        self.assertTrue(loss.use_subbatch)


class TestLanguageLossForwardImpl(unittest.TestCase):
    """Test LanguageLoss.forward_impl method."""

    @patch(
        "paddlefleet.models.common.language_loss.language_loss.get_context_parallel_world_size",
        return_value=1,
    )
    @patch(
        "paddlefleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch("paddle.distributed.is_initialized", return_value=False)
    def test_forward_impl_basic(self, mock_dist, mock_tp, mock_cp):
        import paddle

        from paddlefleet.models.common.language_loss.language_loss import (
            LanguageLoss,
        )

        mock_config = MagicMock()
        mock_config.parallel_output = True
        mock_config.loss_subbatch_sequence_length = 0

        loss_fn = LanguageLoss(config=mock_config)
        logits = paddle.randn([2, 10, 100])
        labels = paddle.randint(0, 100, [2, 10])
        result = loss_fn.forward_impl(logits, labels)
        self.assertIsNotNone(result)

    @patch(
        "paddlefleet.models.common.language_loss.language_loss.get_context_parallel_world_size",
        return_value=1,
    )
    @patch(
        "paddlefleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch("paddle.distributed.is_initialized", return_value=False)
    def test_forward_impl_all_ignored(self, mock_dist, mock_tp, mock_cp):
        import paddle

        from paddlefleet.models.common.language_loss.language_loss import (
            LanguageLoss,
        )

        mock_config = MagicMock()
        mock_config.parallel_output = True
        mock_config.loss_subbatch_sequence_length = 0

        loss_fn = LanguageLoss(config=mock_config)
        logits = paddle.randn([2, 10, 100])
        labels = paddle.full([2, 10], -100, dtype="int64")
        result = loss_fn.forward_impl(logits, labels)
        self.assertIsNotNone(result)

    @patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "1"})
    @patch(
        "paddlefleet.models.common.language_loss.language_loss.get_context_parallel_world_size",
        return_value=1,
    )
    @patch(
        "paddlefleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch("paddle.distributed.is_initialized", return_value=False)
    def test_uac_nonexperimental_uses_fp32_sum_and_defer_token(
        self, mock_dist, mock_tp, mock_cp
    ):
        import inspect

        import paddle

        from paddlefleet.models.common.language_loss import language_loss as ll
        from paddlefleet.models.common.language_loss.language_loss import (
            DeferTokenNormalizationOp,
            LanguageLoss,
        )

        mock_config = MagicMock()
        mock_config.parallel_output = True
        mock_config.loss_subbatch_sequence_length = 0
        mock_config.gpt_model_use_experimental_version = False
        mock_config.use_accuracy_compatible = True
        mock_config.fused_linear_ce_loss_chunk = 0
        mock_config.experimental_dataflow = False

        src = inspect.getsource(LanguageLoss.forward_impl)
        self.assertIn("_normalize_loss_by_tokens", src)
        self.assertNotIn("cast(paddle.float64)", src)
        self.assertIn("DeferTokenNormalizationOp", inspect.getsource(ll))

        ll.clear_pending_gradient_divisor()
        loss_fn = LanguageLoss(config=mock_config)
        logits = paddle.randn([2, 4, 8])
        labels = paddle.randint(0, 8, [2, 4])
        out = loss_fn.forward_impl(logits, labels)
        self.assertIsNotNone(out)
        self.assertIsNotNone(ll.get_pending_gradient_divisor())
        ll.clear_pending_gradient_divisor()
        self.assertIs(
            DeferTokenNormalizationOp.__bases__[0], paddle.autograd.PyLayer
        )

    @patch.dict(os.environ, {"FLAGS_use_accuracy_compatible_kernel": "0"})
    @patch(
        "paddlefleet.models.common.language_loss.language_loss.get_context_parallel_world_size",
        return_value=1,
    )
    @patch(
        "paddlefleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch("paddle.distributed.is_initialized", return_value=False)
    def test_non_uac_token_normalize_divides_by_tensor_count(
        self, mock_dist, mock_tp, mock_cp
    ):
        import inspect

        import paddle

        from paddlefleet.models.common.language_loss.language_loss import (
            LanguageLoss,
        )

        src = inspect.getsource(LanguageLoss.forward_impl)
        self.assertIn("loss / lossmask.sum()", src)

        mock_config = MagicMock()
        mock_config.parallel_output = True
        mock_config.loss_subbatch_sequence_length = 0
        mock_config.gpt_model_use_experimental_version = False
        mock_config.use_accuracy_compatible = False
        mock_config.fused_linear_ce_loss_chunk = 0
        mock_config.experimental_dataflow = False

        loss_fn = LanguageLoss(config=mock_config)
        logits = paddle.randn([2, 4, 8])
        labels = paddle.randint(0, 8, [2, 4])
        out = loss_fn.forward_impl(logits, labels)
        per_token = loss_fn.loss_func(logits.cast("float32"), labels)
        lossmask = (
            (labels != loss_fn.ignored_index).reshape([-1]).cast(paddle.float32)
        )
        expected = (
            paddle.sum(per_token.cast(paddle.float32).reshape([-1]) * lossmask)
            / lossmask.sum()
        )
        self.assertTrue(bool((out == expected).numpy().all()))


class TestLanguageLossForwardWithMTP(unittest.TestCase):
    """Test LanguageLoss.forward with Multi-Token Prediction."""

    @patch("paddle.device.cuda.empty_cache")
    @patch(
        "paddlefleet.models.common.language_loss.language_loss.get_context_parallel_world_size",
        return_value=1,
    )
    @patch(
        "paddlefleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch("paddle.distributed.is_initialized", return_value=False)
    def test_forward_list_logits(self, mock_dist, mock_tp, mock_cp, mock_cache):
        import paddle

        from paddlefleet.models.common.language_loss.language_loss import (
            LanguageLoss,
        )

        mock_config = MagicMock()
        mock_config.parallel_output = True
        mock_config.loss_subbatch_sequence_length = 0
        mock_config.num_nextn_predict_layers = 2
        mock_config.mtp_load_weight_only = False
        mock_config.mtp_distillation_loss = False
        mock_config.train_mtp_only = False
        mock_config.add_mtp_loss = False
        mock_config.mtp_loss_scaling_factor = 1.0
        mock_config.recompute_modules = None
        mock_config.gpt_model_use_experimental_version = False
        # MagicMock attributes are truthy, so pin the real default: otherwise
        # the erndata packed-doc MTP branch is taken and the L+K label trim is
        # skipped.
        mock_config.use_erndata = False

        loss_fn = LanguageLoss(config=mock_config)
        logits = [
            paddle.randn([2, 10, 100]),
            paddle.randn([2, 10, 100]),
            paddle.randn([2, 10, 100]),
        ]
        labels = paddle.randint(0, 100, [2, 12])
        result = loss_fn.forward(logits, labels)
        self.assertIsNotNone(result)


class TestLanguageLossBuildScheduleNode(unittest.TestCase):
    """Test LanguageLoss.build_schedule_node."""

    @patch(
        "paddlefleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch("paddle.distributed.is_initialized", return_value=False)
    def test_returns_schedule_node(self, mock_dist, mock_tp):
        from paddlefleet.models.common.language_loss.language_loss import (
            LanguageLoss,
        )

        mock_config = MagicMock()
        mock_config.parallel_output = True
        mock_config.loss_subbatch_sequence_length = 0

        loss_fn = LanguageLoss(config=mock_config)
        node = loss_fn.build_schedule_node()
        self.assertIsNotNone(node)


class TestDistributedSoftmaxOp(unittest.TestCase):
    """Test DistributedSoftmaxOp static methods."""

    def test_forward_method_exists(self):
        from paddlefleet.models.common.language_loss.language_loss import (
            DistributedSoftmaxOp,
        )

        self.assertTrue(hasattr(DistributedSoftmaxOp, "forward"))
        self.assertTrue(callable(DistributedSoftmaxOp.forward))

    def test_backward_method_exists(self):
        from paddlefleet.models.common.language_loss.language_loss import (
            DistributedSoftmaxOp,
        )

        self.assertTrue(hasattr(DistributedSoftmaxOp, "backward"))
        self.assertTrue(callable(DistributedSoftmaxOp.backward))


class TestSubbatch(unittest.TestCase):
    """Test subbatch function."""

    def test_subbatch_small_input(self):
        import paddle

        from paddlefleet.models.common.language_loss.language_loss import (
            subbatch,
        )

        def simple_fn(x, y):
            return x + y

        sb_fn = subbatch(
            simple_fn, arg_idx=[0, 1], axis=[0, 0], bs=100, out_idx=0
        )
        x = paddle.randn([5, 10])
        y = paddle.randn([5, 10])
        result = sb_fn(x, y)
        # Input smaller than batch size, should call function directly
        self.assertIsNotNone(result)

    def test_subbatch_equal_batch_size(self):
        import paddle

        from paddlefleet.models.common.language_loss.language_loss import (
            subbatch,
        )

        def simple_fn(x, y):
            return x + y

        sb_fn = subbatch(
            simple_fn, arg_idx=[0, 1], axis=[0, 0], bs=5, out_idx=0
        )
        x = paddle.randn([5, 10])
        y = paddle.randn([5, 10])
        result = sb_fn(x, y)
        self.assertIsNotNone(result)

    def test_subbatch_assert_arg_axis_length(self):
        from paddlefleet.models.common.language_loss.language_loss import (
            subbatch,
        )

        def simple_fn(x, y):
            return x + y

        # Mismatched arg_idx and axis lengths should raise
        sb_fn = subbatch(simple_fn, arg_idx=[0], axis=[0, 1], bs=10, out_idx=0)
        import paddle

        x = paddle.randn([10, 10])
        y = paddle.randn([10, 10])
        with self.assertRaises(AssertionError):
            sb_fn(x, y)


class TestLanguageLossMTPTracker(unittest.TestCase):
    """Test LanguageLoss.mtp_loss_tracker class attribute."""

    def test_tracker_is_dict(self):
        from paddlefleet.models.common.language_loss.language_loss import (
            LanguageLoss,
        )

        self.assertIsInstance(LanguageLoss.mtp_loss_tracker, dict)

    def test_tracker_initially_empty(self):
        from paddlefleet.models.common.language_loss.language_loss import (
            LanguageLoss,
        )

        LanguageLoss.mtp_loss_tracker.clear()
        self.assertEqual(len(LanguageLoss.mtp_loss_tracker), 0)


class TestLanguageLossForwardSingleLogits(unittest.TestCase):
    """Test LanguageLoss.forward with single (non-list) logits."""

    @patch(
        "paddlefleet.models.common.language_loss.language_loss.get_context_parallel_world_size",
        return_value=1,
    )
    @patch(
        "paddlefleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch("paddle.distributed.is_initialized", return_value=False)
    def test_forward_single_logits(self, mock_dist, mock_tp, mock_cp):
        import paddle

        from paddlefleet.models.common.language_loss.language_loss import (
            LanguageLoss,
        )

        mock_config = MagicMock()
        mock_config.parallel_output = True
        mock_config.loss_subbatch_sequence_length = 0
        mock_config.recompute_modules = None

        loss_fn = LanguageLoss(config=mock_config)
        logits = paddle.randn([2, 10, 100])
        labels = paddle.randint(0, 100, [2, 10])
        result = loss_fn.forward(logits, labels)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
