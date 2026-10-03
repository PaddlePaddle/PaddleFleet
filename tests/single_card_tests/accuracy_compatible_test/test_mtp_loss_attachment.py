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

"""Check value preservation and scaled gradients through LanguageLoss.forward."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import paddle

from paddlefleet.models.common.language_loss import language_loss


class TestAccuracyCompatibleMTPLossAttachment(unittest.TestCase):
    def forward(
        self,
        main,
        auxiliary,
        *,
        compatible=True,
        add_mtp=True,
        scale=0.5,
        main_tokens=None,
    ):
        # Stub per-head CE, but execute the real label slicing, MTP averaging,
        # attachment, tracker update, and return path together.
        depth = len(auxiliary)
        instance = SimpleNamespace(
            config=SimpleNamespace(
                num_nextn_predict_layers=depth,
                mtp_load_weight_only=False,
                use_erndata=False,
                mtp_distillation_loss=False,
                train_mtp_only=False,
                gpt_model_use_experimental_version=False,
                add_mtp_loss=add_mtp,
                mtp_loss_scaling_factor=scale,
            ),
            use_accuracy_compatible=compatible,
        )
        active = [
            (i + 1, value)
            for i, value in enumerate(auxiliary)
            if value is not None
        ]
        values = iter([main, *(value for _, value in active)])

        def head_loss(*_):
            if instance._forward.call_count > 1:
                self.assertEqual(instance._deferred_main_tokens, main_tokens)
            return next(values)

        instance._forward = Mock(side_effect=head_loss)
        labels = paddle.arange(3 + depth).reshape([1, 3 + depth])
        logits = [
            object(),
            *(object() if value is not None else None for value in auxiliary),
        ]
        with (
            patch.dict(language_loss.LanguageLoss.mtp_loss_tracker, clear=True),
            patch.object(
                language_loss, "get_global_training_logs", return_value=None
            ),
            patch.object(
                language_loss,
                "get_pending_gradient_divisor",
                return_value=main_tokens,
            ),
        ):
            output = language_loss.LanguageLoss.forward(
                instance, logits, labels
            )
            self.assertEqual(
                len(language_loss.LanguageLoss.mtp_loss_tracker), len(active)
            )
        self.assertIsNone(instance._deferred_main_tokens)
        self.assertEqual(instance._forward.call_count, len(active) + 1)
        for index, call in zip(
            [0, *(i for i, _ in active)], instance._forward.call_args_list
        ):
            self.assertIs(call.args[0], logits[index])
            self.assertEqual(
                call.args[1].tolist(), [list(range(index, index + 3))]
            )
        return output

    def test_sampled_depths_keep_main_divisor_and_average_only_computed_losses(
        self,
    ):
        main, first, last = (self.scalar(value) for value in (1.0, 4.0, 8.0))
        output = self.forward(main, [first, last, None, None], main_tokens=17.0)
        self.assertEqual(output.numpy().tobytes(), main.numpy().tobytes())
        output.backward()
        self.assertEqual(main.grad.item(), 1.0)
        self.assertEqual(first.grad.item(), 0.25)
        self.assertEqual(last.grad.item(), 0.25)

    @staticmethod
    def scalar(value):
        tensor = paddle.full([], value, dtype="float32")
        tensor.stop_gradient = False
        return tensor

    def test_preserves_main_bits_and_scales_each_auxiliary_gradient(self):
        # Adding MAIN before cancelling a large auxiliary loses MAIN in FP32.
        for depth in (1, 2):
            with self.subTest(depth=depth):
                main = self.scalar(1.0)
                auxiliary = [self.scalar(67108864.0) for _ in range(depth)]
                output = self.forward(main, auxiliary)
                self.assertEqual(
                    output.numpy().tobytes(), main.numpy().tobytes()
                )
                output.backward(paddle.full([], 0.25, dtype="float32"))
                self.assertEqual(main.grad.item(), 0.25)
                for value in auxiliary:
                    self.assertEqual(value.grad.item(), 0.25 * 0.5 / depth)

    def test_disabled_auxiliary_does_not_acquire_gradient(self):
        main, auxiliary = self.scalar(1.0), self.scalar(67108864.0)
        output = self.forward(main, [auxiliary], add_mtp=False)
        self.assertIs(output, main)
        output.backward()
        self.assertEqual(main.grad.item(), 1.0)
        self.assertIsNone(auxiliary.grad)

    def test_ordinary_mode_still_reports_scaled_auxiliary_value(self):
        main, auxiliary = self.scalar(2.0), self.scalar(8.0)
        output = self.forward(main, [auxiliary], compatible=False)
        self.assertEqual(output.item(), 6.0)
        output.backward()
        self.assertEqual(main.grad.item(), 1.0)
        self.assertEqual(auxiliary.grad.item(), 0.5)


if __name__ == "__main__":
    paddle.set_device("gpu:0")
    unittest.main()
