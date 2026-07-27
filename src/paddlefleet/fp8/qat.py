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

import paddle

from .act_quant import act_quant

_FP4_INDEXER_OFFICIAL_MODE = 1
_FP4_EXPERT_WEIGHT_MODE = 2


class _FP4IndexerFakeQuant(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x: paddle.Tensor):
        from paddlefleet_ops import fused_mxfp4_fake_quant

        return fused_mxfp4_fake_quant(
            x.contiguous(), _FP4_INDEXER_OFFICIAL_MODE
        )

    @staticmethod
    def backward(ctx, grad: paddle.Tensor):
        return grad


def fp4_indexer_fake_quant(x: paddle.Tensor) -> paddle.Tensor:
    """Apply official DeepSeek-V4 MXFP4 fake quantization with identity STE."""
    return _FP4IndexerFakeQuant.apply(x)


class _FP4ExpertFakeQuant(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, weight: paddle.Tensor):
        from paddlefleet_ops import fused_mxfp4_fake_quant

        return fused_mxfp4_fake_quant(
            weight.contiguous(), _FP4_EXPERT_WEIGHT_MODE
        )

    @staticmethod
    def backward(ctx, grad: paddle.Tensor):
        return grad


def fp4_expert_fake_quant(weight: paddle.Tensor) -> paddle.Tensor:
    """Fake-quantize an expert weight along its input axis with identity STE."""
    if weight.ndim not in (2, 3):
        raise ValueError("expert weight must be rank 2 or rank 3")
    stacked_weight = weight.unsqueeze(0) if weight.ndim == 2 else weight
    quantized = _FP4ExpertFakeQuant.apply(stacked_weight)
    return quantized.squeeze(0) if weight.ndim == 2 else quantized


def apply_fp4_expert_fake_quant(weight, enabled):
    """Return original expert weights unless MXFP4 QAT is enabled."""
    if not enabled or weight is None:
        return weight
    if isinstance(weight, (list, tuple)):
        return type(weight)(
            apply_fp4_expert_fake_quant(item, True) for item in weight
        )
    if hasattr(weight, "fp8_weight_stacked") or hasattr(
        weight, "fp8_weight_stacked_transpose"
    ):
        raise RuntimeError(
            "MXFP4 expert QAT requires live BF16 master weights; "
            "offline FP8 expert weights are incompatible"
        )
    if weight.dtype not in (paddle.bfloat16, paddle.float16, paddle.float32):
        raise TypeError(
            f"MXFP4 expert QAT requires BF16/FP16/FP32 weights, got {weight.dtype}"
        )
    return fp4_expert_fake_quant(weight)


def fp8_simulate(x: paddle.Tensor, block_size: int):
    y, scale = act_quant(x.contiguous(), block_size, "ue8m0")
    shape = [*list(y.shape[:-1]), -1, block_size]
    y = y.reshape(shape).astype("float32") * scale.unsqueeze(-1)
    return y.flatten(-2, -1).astype(x.dtype)


class DeepSeekV4LinearQATFunc(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, kv, block_size=128):
        return fp8_simulate(kv, block_size)

    @staticmethod
    def backward(ctx, grad_kv):
        return grad_kv


fp8_simulate_qat = DeepSeekV4LinearQATFunc.apply
