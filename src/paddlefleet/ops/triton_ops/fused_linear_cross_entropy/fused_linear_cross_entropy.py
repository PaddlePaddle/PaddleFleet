#!/usr/bin/env python3

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

"""
Fused Linear Cross Entropy：将线性层 + 交叉熵合并为一个显存高效的操作。

基于 Liger-Kernel 的分块思路：在每个 chunk 上只物化 [chunk, V] 的 logits，
Triton kernel 前向计算 loss 的同时原地写回梯度，避免保留完整
[BT, V] softmax/logits 中间张量。
"""

import paddle
import triton

from .cross_entropy import liger_cross_entropy_kernel
from .utils import element_mul_kernel

MAX_FUSED_SIZE = 65536 // 2


def fused_linear_cross_entropy_forward(
    _input,
    weight,
    target,
    bias=None,
    ignore_index=-100,
    reduction="none",
    num_chunks=1,
):
    """前向：分 chunk 计算 logits / loss / grad_input / grad_weight。

    参数:
        _input: [BT, H] 输入 hidden states (bf16/fp16)。
        weight: [V, H] 线性层权重 (与 F.linear 一致)。
        target: [BT] 整数目标标签。
        bias: [V] 可选偏置。
        ignore_index: 被忽略的标签值。
        reduction: "none" / "mean" / "sum"。
        num_chunks: 分多少个 chunk 做计算。
    """
    input_requires_grad = not _input.stop_gradient
    weight_requires_grad = not weight.stop_gradient
    orig_dtype = _input.dtype

    BT, H = _input.shape
    V = weight.shape[0]  # weight 是 [V, H]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    chunk_size = triton.cdiv(BT, num_chunks)

    grad_input = (
        paddle.zeros([BT, H], dtype=paddle.float32)
        if input_requires_grad
        else None
    )
    grad_weight = (
        paddle.zeros(weight.shape, dtype=paddle.float32)
        if (input_requires_grad and weight_requires_grad)
        else None
    )
    grad_bias = (
        paddle.zeros([bias.shape[0]], dtype=paddle.float32)
        if (input_requires_grad and bias is not None)
        else None
    )

    loss_1d = paddle.zeros([BT], dtype=paddle.float32)

    target_mask = target != ignore_index
    total_n_non_ignore = target_mask.sum().item()

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        _input_chunk = _input[start_idx:end_idx]

        logits_chunk = paddle.compat.nn.functional.linear(
            _input_chunk, weight, bias
        )

        target_chunk = target[start_idx:end_idx]
        n_rows = logits_chunk.shape[0]

        loss_1d_slice = loss_1d[start_idx:end_idx]

        logits_chunk = logits_chunk.contiguous()
        target_chunk = target_chunk.contiguous()

        liger_cross_entropy_kernel[(n_rows,)](
            X_ptr=logits_chunk,
            X_stride=logits_chunk.stride(-2),
            Y_ptr=target_chunk,
            Y_stride=target_chunk.stride(-1),
            loss_ptr=loss_1d_slice,
            loss_stride=loss_1d_slice.stride(-1),
            n_cols=V,
            n_non_ignore=total_n_non_ignore,
            ignore_index=ignore_index,
            reduction=reduction,
            HAS_GRADIENTS=input_requires_grad,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )

        loss_1d[start_idx:end_idx] = loss_1d_slice
        # kernel 已将梯度原地写回 logits_chunk（现在是 grad_logits，尚未归一化）
        grad_logits_chunk = logits_chunk

        if input_requires_grad:
            grad_input[start_idx:end_idx] = paddle.matmul(
                grad_logits_chunk, weight
            )

        if grad_weight is not None:
            with paddle.amp.auto_cast(False):
                paddle._C_ops.fused_linear_param_grad_add(
                    grad_logits_chunk,  # [C, V]
                    _input_chunk,  # [C, H]
                    grad_weight,  # [V, H]
                    None,
                    True,
                    False,
                )

        if grad_bias is not None:
            grad_bias.add_(grad_logits_chunk.sum(axis=0))

    if input_requires_grad:
        grad_input = grad_input.cast(orig_dtype)
    if grad_bias is not None:
        grad_bias = grad_bias.cast(bias.dtype)

    if reduction == "none":
        loss = loss_1d
    else:
        loss = paddle.sum(loss_1d)

    return loss, grad_input, grad_weight, grad_bias


def fused_linear_cross_entropy_backward(
    grad_output, grad_input, grad_weight, grad_bias
):
    """反向：当 grad_output != 1.0 时，对已保存的梯度做缩放。"""
    if grad_output.shape == [] and float(grad_output) == 1.0:
        return grad_input, grad_weight, grad_bias

    # none 模式：grad_output 是 [BT] 向量 (有效 token = 1/N，无效 = 0)。
    # forward 已将无效 token 的 grad_logits 清零，所有有效 token 缩放因子相同。
    if grad_output.ndim >= 1:
        grad_output = grad_output.max().reshape([])

    BT, H = grad_input.shape
    n_rows = BT
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))

    element_mul_kernel[(n_rows,)](
        grad_input,
        grad_input.stride(-2),
        grad_output,
        H,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
    )

    if grad_weight is not None:
        n_rows_w = grad_weight.shape[0]
        n_cols_w = grad_weight.shape[1]
        BLOCK_SIZE_W = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols_w))
        element_mul_kernel[(n_rows_w,)](
            grad_weight,
            grad_weight.stride(-2),
            grad_output,
            n_cols_w,
            BLOCK_SIZE=BLOCK_SIZE_W,
            num_warps=32,
        )

    if grad_bias is not None:
        V = grad_bias.shape[0]
        element_mul_kernel[(V,)](
            grad_bias,
            grad_bias.stride(-1),
            grad_output,
            1,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )

    return grad_input, grad_weight, grad_bias


class LigerFusedLinearCrossEntropyFunction(paddle.autograd.PyLayer):
    """fused linear + cross entropy 的自定义前向 / 反向。

    使用方式:
        loss = LigerFusedLinearCrossEntropyFunction.apply(
            _input,       # [BT, H]
            weight,       # [V, H]
            target,       # [BT]
            bias,         # [V] 或 None
            ignore_index,
            reduction,
            num_chunks,
        )
    """

    @staticmethod
    def forward(ctx, *args):
        has_bias = args[-4]
        ignore_index = args[-3]
        reduction = args[-2]
        num_chunks = args[-1]

        tensor_args = args[:-4]
        _input = tensor_args[0]
        weight = tensor_args[1]
        target = tensor_args[2]
        bias = tensor_args[3] if has_bias else None

        loss, grad_input, grad_weight, grad_bias = (
            fused_linear_cross_entropy_forward(
                _input=_input,
                weight=weight,
                target=target,
                bias=bias,
                ignore_index=ignore_index,
                reduction=reduction,
                num_chunks=num_chunks,
            )
        )

        ctx.save_for_backward(
            grad_input.detach() if grad_input is not None else None,
            grad_weight.detach() if grad_weight is not None else None,
            grad_bias.detach() if grad_bias is not None else None,
        )
        ctx.has_bias = has_bias
        ctx.weight_ref = weight
        ctx.weight_requires_grad = not weight.stop_gradient
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        (grad_input, grad_weight, grad_bias) = ctx.saved_tensor()
        grad_input, grad_weight, grad_bias = (
            fused_linear_cross_entropy_backward(
                grad_output, grad_input, grad_weight, grad_bias
            )
        )

        if ctx.weight_requires_grad and grad_weight is not None:
            weight = ctx.weight_ref
            if hasattr(weight, "main_grad"):
                if weight.main_grad is None:
                    weight.main_grad = paddle.zeros(
                        weight.shape, dtype=paddle.float32
                    )
                weight.main_grad.add_(grad_weight)
                if hasattr(weight, "_apply_backward_hook"):
                    weight._apply_backward_hook()
                grad_weight = None

        result = [grad_input, grad_weight, None]
        if ctx.has_bias:
            result.append(grad_bias)
        return tuple(result)

    @classmethod
    def apply(
        cls,
        _input,
        weight,
        target,
        bias=None,
        ignore_index=-100,
        reduction="none",
        num_chunks=1,
    ):
        tensor_args = [_input, weight, target]
        has_bias = bias is not None
        if has_bias:
            tensor_args.append(bias)
        scalar_args = [has_bias, ignore_index, reduction, num_chunks]
        return super().apply(*(tensor_args + scalar_args))


class LigerFusedLinearCrossEntropyLoss(paddle.nn.Layer):
    """nn.Layer 封装，便于在组网代码中直接替换 nn.CrossEntropyLoss。"""

    def __init__(
        self,
        ignore_index: int = -100,
        reduction: str = "none",
        num_chunks: int = 1,
    ):
        super().__init__()
        assert reduction in {"mean", "sum", "none"}, (
            f"reduction must be 'mean', 'sum', or 'none'. Got: {reduction}"
        )
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.num_chunks = num_chunks

    def forward(self, lin_weight, _input, target, bias=None):
        return LigerFusedLinearCrossEntropyFunction.apply(
            _input,
            lin_weight,
            target,
            bias,
            self.ignore_index,
            self.reduction,
            self.num_chunks,
        )
