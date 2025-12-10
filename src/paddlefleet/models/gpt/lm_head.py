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

from paddlefleet.tensor_parallel.layers import ColumnParallelLinear


class GPTLMHead(ColumnParallelLinear):
    def __init__(self, **kwargs):
        self.config = kwargs["config"]
        self.skip_weight_param_allocation = kwargs[
            "skip_weight_param_allocation"
        ]
        super().__init__(**kwargs)

    def forward(self, dict_args: dict):
        hidden_states = dict_args["hidden_states"]
        if self.skip_weight_param_allocation:
            logits, _ = super().forward(hidden_states, self.weight.T)
        else:
            logits, _ = super().forward(hidden_states)
        if self.config.sequence_parallel:
            logits = logits.transpose([1, 0, 2]).contiguous()
        return logits

    @property
    def embedding_weight(self):
        return self.weight
