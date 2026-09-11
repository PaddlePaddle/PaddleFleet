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

"""Numerical mode follows explicit configuration across calls in one process."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import paddle

from paddlefleet.fusions.fused_bias_dropout import get_bias_dropout_add
from paddlefleet.models.common.language_loss.language_loss import LanguageLoss
from paddlefleet.transformer.dsa_attention import _absorb_q_nope_k_up
from paddlefleet.transformer.multi_token_prediction import _mtp_eh_projection
from paddlefleet.transformer.transformer_config import TransformerConfig


class TestAccuracyCompatibleConfiguration(unittest.TestCase):
    def test_explicit_helper_mode_ignores_legacy_environment(self):
        query = paddle.arange(24, dtype="float32").reshape([2, 3, 4])
        weight = paddle.arange(40, dtype="float32").reshape([2, 4, 5])
        for enabled in (True, False, True):
            with (
                self.subTest(enabled=enabled),
                patch.dict(
                    os.environ,
                    {"MODEL_REPRO_IEEE_KERNEL": str(int(not enabled))},
                ),
                patch.object(paddle, "bmm", wraps=paddle.bmm) as bmm,
                patch.object(paddle, "einsum", wraps=paddle.einsum) as einsum,
            ):
                result = _absorb_q_nope_k_up(
                    query, weight, use_accuracy_compatible=enabled
                )
                self.assertEqual(bmm.call_count, int(enabled))
                self.assertEqual(einsum.call_count, int(not enabled))
                self.assertEqual(result.shape, [2, 3, 5])

    def test_loss_instances_keep_independent_configuration(self):
        group = SimpleNamespace(tp=None, cp=None, ep=None)
        losses = []
        for enabled in (False, True):
            config = TransformerConfig(
                num_hidden_layers=1,
                hidden_size=8,
                num_attention_heads=1,
                use_accuracy_compatible=enabled,
            )
            losses.append(LanguageLoss(config, pg_collection=group))
        for legacy in ("1", "0"):
            with patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": legacy}):
                self.assertFalse(losses[0].use_accuracy_compatible)
                self.assertTrue(losses[1].use_accuracy_compatible)

    def test_mtp_projection_mode_is_explicit_and_preserves_bias(self):
        class Projection:
            skip_bias_add = True
            weight = paddle.ones([4, 3])
            bias = paddle.arange(3, dtype="float32")

            def __call__(self, value):
                return paddle.zeros([2, 3]), self.bias

        projection = Projection()
        x = paddle.ones([2, 4])
        for enabled in (True, False, True):
            with patch.dict(
                os.environ, {"MODEL_REPRO_IEEE_KERNEL": str(int(not enabled))}
            ):
                output, bias = _mtp_eh_projection(
                    projection, x, 1, use_accuracy_compatible=enabled
                )
            self.assertIs(bias, projection.bias)
            self.assertEqual(
                output.tolist(), [[4.0 if enabled else 0.0] * 3] * 2
            )

    def test_dropout_closure_keeps_mode_through_backward(self):
        add = get_bias_dropout_add(
            training=True,
            fused=False,
            use_accuracy_compatible=True,
            tensor_parallel_size=1,
        )
        x = paddle.ones([2, 4])
        residual = paddle.full([2, 4], 2.0)
        x.stop_gradient = residual.stop_gradient = False
        with patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "0"}):
            output = add((x, None), residual, 0.0)
        with patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "1"}):
            output.sum().backward()
        self.assertEqual(output.tolist(), [[3.0] * 4] * 2)
        self.assertEqual(x.grad.tolist(), [[1.0] * 4] * 2)
        self.assertEqual(residual.grad.tolist(), [[1.0] * 4] * 2)


if __name__ == "__main__":
    paddle.set_device("gpu:0")
    unittest.main()
