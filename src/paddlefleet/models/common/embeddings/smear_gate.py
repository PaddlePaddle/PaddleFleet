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

"""
SmearGate: per-dimension gated blending of current and previous token embeddings.

output[t] = gate * emb[t] + (1 - gate) * emb[t-1]

Reference: OpenAI parameter-golf (SmearGate + OrthoInit + Muon WD)
https://github.com/openai/parameter-golf
"""

import paddle
import paddle.nn as nn
from paddle import Tensor


class SmearGate(nn.Layer):
    """Per-dimension gated blending of current and previous token embeddings.

    Args:
        hidden_size: model hidden dimension (D)
        init_value: initial gate logit value. sigmoid(3.0) ≈ 0.95 (near-identity)
    """

    def __init__(self, hidden_size: int, init_value: float = 3.0):
        super().__init__()
        self.gate_logit = self.create_parameter(
            shape=[hidden_size],
            default_initializer=nn.initializer.Constant(init_value),
        )

    def forward(self, embeddings: Tensor) -> Tensor:
        """
        Args:
            embeddings: [B, S, D]
        Returns:
            [B, S, D] smeared embeddings
        """
        gate = paddle.sigmoid(self.gate_logit)
        shifted = paddle.concat(
            [paddle.zeros_like(embeddings[:, :1, :]), embeddings[:, :-1, :]],
            axis=1,
        )
        return gate * embeddings + (1 - gate) * shifted
