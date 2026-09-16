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

"""Context-aware gated N-gram embedding with causal depthwise dilated convolution.

Uses DeepSeek V4-style sqrt(softplus) scoring with competitive normalization
to dynamically weight unigram and ngram sub-table contributions based on
local context captured via dilated causal convolutions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor

if TYPE_CHECKING:
    from paddlefleet.transformer.transformer_config import TransformerConfig


class NgramGatedEmbedding(nn.Layer):
    """Context-aware gated mixing of unigram + ngram sub-table projections.

    Each ngram level uses a causal depthwise conv with dilation = ngram_order,
    capturing context at the appropriate scale. Gate scores are computed via
    sqrt(softplus) and normalized to create competition.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n = config.ngram_emb_neighbor_num
        self.k = config.ngram_emb_split_num
        self.kernel_size = config.ngram_gate_conv_kernel_size
        self.route_scale = config.ngram_gate_route_scale

        self.num_experts = 1 + self.k * (self.n - 1)

        # Causal depthwise dilated convolutions: dilation = 1, 2, ..., n
        self.convs = nn.LayerList()
        for i in range(self.n):
            dilation = i + 1
            padding = (self.kernel_size - 1) * dilation
            self.convs.append(
                nn.Conv1D(
                    in_channels=self.hidden_size,
                    out_channels=self.hidden_size,
                    kernel_size=self.kernel_size,
                    groups=self.hidden_size,
                    padding=padding,
                    dilation=dilation,
                    bias_attr=False,
                    data_format="NCL",
                )
            )

        # Gate projection: dilation=1 → 1 score (unigram), others → K scores
        self.gate_projs = nn.LayerList()
        self.gate_projs.append(nn.Linear(self.hidden_size, 1, bias_attr=False))
        for _ in range(self.n - 1):
            self.gate_projs.append(
                nn.Linear(self.hidden_size, self.k, bias_attr=False)
            )

        self._init_gate_weights()

    def _init_gate_weights(self):
        for proj in self.gate_projs:
            nn.initializer.Normal(mean=0.0, std=0.01)(proj.weight)

    def forward(
        self, word_emb: Tensor, ngram_projections: list
    ) -> Tensor:
        """Compute gated mixture of unigram and ngram projections.

        Args:
            word_emb: [B, S, H] word embedding (unigram expert)
            ngram_projections: list of K*(N-1) tensors each [B, S, H]

        Returns:
            [B, S, H] gated output
        """
        B, S, H = word_emb.shape
        input_dtype = word_emb.dtype

        # [B, S, H] -> [B, H, S] for NCL conv
        x_ncl = word_emb.transpose([0, 2, 1])

        # Compute gate logits via dilated convs
        all_scores = []
        for conv, gate_proj in zip(self.convs, self.gate_projs):
            conv_out = conv(x_ncl)[..., :S]  # [B, H, S] causal trim
            conv_out = conv_out.transpose([0, 2, 1])  # [B, S, H]
            all_scores.append(gate_proj(conv_out))  # [B, S, 1] or [B, S, K]

        logits = paddle.concat(all_scores, axis=-1)  # [B, S, num_experts]

        # sqrt(softplus) scoring with fp32 for stability
        logits_fp32 = logits.cast("float32")
        scores = paddle.sqrt(F.softplus(logits_fp32))
        gate_weights = scores / scores.sum(axis=-1, keepdim=True)
        gate_weights = (gate_weights * self.route_scale).cast(input_dtype)

        # Weighted sum of experts
        experts = paddle.stack(
            [word_emb] + ngram_projections, axis=2
        )  # [B, S, num_experts, H]

        output = (
            gate_weights.unsqueeze(-1) * experts
        ).sum(axis=2)  # [B, S, H]

        return output
