# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import Tensor
from paddle.nn.functional import flashmask_attention

from .rr_attn_estimate_triton_op import rr_attn_estimate_triton_func


def rr_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    startend_row_indices: Tensor | None = None,
    *,
    causal: bool = False,
    training: bool = True,
    stride: int = 8,
    threshold: float = 1.0,
):
    if threshold == 1.0:
        return flashmask_attention(
            query,
            key,
            value,
            startend_row_indices,
            causal=causal,
            training=training,
        )

    with paddle.no_grad(), paddle.compat.use_torch_proxy_guard():
        _, boundary_mask, topp_mask = rr_attn_estimate_triton_func(
            query,
            key,
            startend_row_indices,
            stride=stride,
            causal=causal,
            threshold=threshold,
        )

    block_mask = paddle.logical_or(boundary_mask, topp_mask).to(paddle.int32)
    return flashmask_attention(
        query,
        key,
        value,
        startend_row_indices,
        causal=causal,
        training=training,
        block_mask=block_mask,
    )
