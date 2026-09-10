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

"""Exercise production backward and loss normalization compatibility."""

from types import SimpleNamespace

import numpy as np
import paddle
import pytest

from paddlefleet.models.common.language_loss.language_loss import (
    LanguageLoss,
    clear_pending_gradient_divisor,
    get_pending_gradient_divisor,
)
from paddlefleet.tensor_parallel.layers import (
    general_gemm,
    linear_with_grad_accumulation_and_async_allreduce,
)
from paddlefleet.transformer.transformer_config import TransformerConfig


def test_biased_projection_rounds_gemm_before_bias():
    paddle.set_device("gpu:0")
    x = paddle.full([2, 3, 32], 1.0078125, dtype="bfloat16")
    weight = paddle.full([32, 16], 1.0078125, dtype="bfloat16")
    bias = paddle.full([16], -32.5, dtype="bfloat16")
    output, _ = general_gemm(x, weight, bias=bias, use_accuracy_compatible=True)
    # The exact dot product is 32.501953125. BF16 rounds it to 32.5
    # before cancellation with the bias; a fused bias retains the residual.
    assert paddle.count_nonzero(output).item() == 0


@pytest.mark.parametrize(
    "enabled,expert", [(True, True), (True, False), (False, True)]
)
def test_tp1_expert_preserves_fp32_main_gradient(enabled, expert):
    paddle.set_device("gpu:0")
    paddle.seed(4321)
    x = paddle.randn([19, 32]).cast("bfloat16")
    weight = paddle.randn([32, 16]).cast("bfloat16")
    grad_output = paddle.randn([19, 16]).cast("bfloat16")
    x.stop_gradient = weight.stop_gradient = False
    weight.is_expert_param = expert
    weight.main_grad = paddle.zeros(weight.shape, dtype="float32")
    output = linear_with_grad_accumulation_and_async_allreduce(
        x,
        weight,
        None,
        gradient_accumulation_fusion=False,
        allreduce_dgrad=False,
        sequence_parallel=False,
        tp_group=SimpleNamespace(nranks=1, rank=0, ranks=[0]),
        use_accuracy_compatible=enabled,
    )
    output.backward(grad_output)
    if enabled and expert:
        expected = paddle.matmul(
            grad_output.cast("float32"), x.cast("float32"), transpose_x=True
        ).T
        np.testing.assert_array_equal(
            weight.main_grad.numpy(), expected.numpy()
        )
        assert np.any(expected.numpy().view(np.uint32) & 0xFFFF)
        assert paddle.count_nonzero(weight.grad).item() == 0
    else:
        assert paddle.count_nonzero(weight.main_grad).item() == 0
        assert paddle.count_nonzero(weight.grad).item() > 0
    assert paddle.isfinite(x.grad).all().item()


@pytest.mark.parametrize("defer", [False, True])
def test_loss_normalization_preserves_gradient_timing(defer):
    paddle.set_device("gpu:0")
    clear_pending_gradient_divisor()
    config = TransformerConfig(
        num_hidden_layers=1,
        hidden_size=8,
        num_attention_heads=1,
        use_accuracy_compatible=True,
        defer_token_normalization=defer,
    )
    criterion = LanguageLoss(
        config, pg_collection=SimpleNamespace(tp=None, cp=None, ep=None)
    )
    logits = paddle.to_tensor(
        [
            [
                [0.2, 1.1, -0.4],
                [1.2, -0.3, 0.5],
                [-0.2, 0.4, 1.3],
                [1.0, 0.2, -0.1],
            ]
        ],
        stop_gradient=False,
    )
    labels = paddle.to_tensor([[1, 0, 2, -100]], dtype="int64")
    try:
        loss = criterion.forward_impl(logits, labels)
        reference_logits = logits.detach().clone()
        reference_logits.stop_gradient = False
        token_losses = paddle.nn.functional.cross_entropy(
            reference_logits,
            labels.unsqueeze(-1),
            reduction="none",
            ignore_index=-100,
        )
        reference = (
            token_losses.sum()
            if defer
            else token_losses.cast("float64").sum().cast("float32") / 3
        )
        reference.backward()
        loss.backward()
        np.testing.assert_array_equal(
            logits.grad.numpy(), reference_logits.grad.numpy()
        )
        np.testing.assert_allclose(
            loss.numpy(), token_losses.sum().numpy() / 3, rtol=1e-7
        )
        assert get_pending_gradient_divisor() == (3.0 if defer else None)
    finally:
        clear_pending_gradient_divisor()
