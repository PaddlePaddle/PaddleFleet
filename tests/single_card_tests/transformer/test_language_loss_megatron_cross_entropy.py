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
"""Megatron-style cross-entropy composition used by the Kimi-K2 alignment path.

The composition must agree with paddle's fused ``CrossEntropyLoss`` on the loss
values and on the gradient to within fp32 rounding, honour ``ignore_index`` on
both the forward and the backward, and reproduce Megatron's hand-written
``(softmax - onehot) * g`` backward exactly.
"""

import unittest

import paddle

from paddlefleet.models.common.language_loss.language_loss import (
    LanguageLoss,
    _MegatronStyleCrossEntropy,
)
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import (
    set_kimik2_accuracy_compatible,
    use_kimik2_accuracy_compatible,
)

IGNORE_INDEX = -100


def _batch(seed=0, vocab=37):
    paddle.seed(seed)
    logits = paddle.randn([2, 5, vocab], dtype="float32")
    labels = paddle.randint(0, vocab, [2, 5], dtype="int64")
    # Two ignored positions, one per row, to cover the masked path.
    labels[0, 1] = IGNORE_INDEX
    labels[1, 4] = IGNORE_INDEX
    return logits, labels


class TestMegatronStyleCrossEntropy(unittest.TestCase):
    def setUp(self):
        self.logits, self.labels = _batch()
        self.mask = self.labels == IGNORE_INDEX
        self.loss_func = _MegatronStyleCrossEntropy(IGNORE_INDEX)

    def test_forward_matches_fused_kernel(self):
        reference = paddle.nn.CrossEntropyLoss(reduction="none")
        expected = reference(self.logits, self.labels).squeeze(-1)
        actual = self.loss_func(self.logits, self.labels)
        self.assertEqual(actual.shape, list(self.labels.shape))
        self.assertTrue(
            bool(paddle.allclose(actual, expected, atol=1e-5, rtol=1e-5)),
            "megatron composition disagrees with the fused kernel",
        )

    def test_ignored_positions_have_zero_loss(self):
        actual = self.loss_func(self.logits, self.labels)
        self.assertTrue(bool(paddle.all(actual[self.mask] == 0.0)))

    def test_backward_is_softmax_minus_onehot(self):
        logits = self.logits.detach()
        logits.stop_gradient = False
        loss = self.loss_func(logits, self.labels)
        upstream = paddle.randn(loss.shape, dtype="float32")
        paddle.autograd.backward([loss], [upstream])

        softmax = paddle.nn.functional.softmax(self.logits, axis=-1)
        safe = paddle.where(
            self.mask, paddle.zeros_like(self.labels), self.labels
        ).unsqueeze(-1)
        update = (
            paddle.logical_not(self.mask).astype(softmax.dtype).unsqueeze(-1)
        )
        onehot = paddle.put_along_axis(
            paddle.zeros_like(softmax), safe, update, axis=-1
        )
        g = paddle.where(
            self.mask, paddle.zeros_like(upstream), upstream
        ).unsqueeze(-1)
        expected = (softmax - onehot) * g

        self.assertTrue(
            bool(paddle.all(paddle.abs(logits.grad - expected) < 1e-5)),
            "backward does not reproduce (softmax - onehot) * g",
        )
        # Ignored rows must not receive any gradient.
        self.assertTrue(bool(paddle.all(logits.grad[self.mask] == 0.0)))


class TestKimiK2LossSelection(unittest.TestCase):
    """``LanguageLoss`` picks the composition only when it will actually run."""

    def setUp(self):
        previous = use_kimik2_accuracy_compatible()
        self.addCleanup(set_kimik2_accuracy_compatible, previous)

    def _config(self, **overrides):
        return TransformerConfig(
            num_hidden_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            parallel_output=False,
            loss_subbatch_sequence_length=0,
            **overrides,
        )

    def test_switch_selects_the_megatron_composition(self):
        loss = LanguageLoss(
            self._config(use_kimik2_accuracy=True), pg_collection=object()
        )
        self.assertIsInstance(loss.loss_func, _MegatronStyleCrossEntropy)

    def test_fused_linear_ce_is_rejected(self):
        """The fused path never calls ``loss_func``, so the switch would be
        silently ignored."""
        config = self._config(
            use_kimik2_accuracy=True, fused_linear_ce_loss_chunk=1024
        )
        with self.assertRaisesRegex(ValueError, "fused_linear_ce_loss_chunk"):
            LanguageLoss(config, pg_collection=object())


if __name__ == "__main__":
    unittest.main()
