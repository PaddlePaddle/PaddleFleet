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
paddlefleet.ops.triton_ops.fused_linear_cross_entropy

将线性层与交叉熵 loss 融合的高效实现，通过分块计算和 Triton kernel
节省峰值显存。(从 ernie_core.ops 迁移至 paddlefleet.ops.triton_ops)

公开接口:
    LigerFusedLinearCrossEntropyFunction  —— PyLayer 函数式 API
    LigerFusedLinearCrossEntropyLoss      —— nn.Layer 封装
"""

from .fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyFunction,
    LigerFusedLinearCrossEntropyLoss,
)

__all__ = [
    "LigerFusedLinearCrossEntropyFunction",
    "LigerFusedLinearCrossEntropyLoss",
]
