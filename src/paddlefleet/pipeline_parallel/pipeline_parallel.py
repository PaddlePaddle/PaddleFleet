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
from paddle import nn

from .pp_layers import PipelineLayer


class NoPipelineParallel(nn.Layer):
    def __init__(self, layers, strategy):
        assert isinstance(layers, PipelineLayer)
        super().__init__()
        self._layers = layers
        self._strategy = strategy

    def forward_backward_pipeline(
        self,
        data,
        scaler=None,
        return_micro_batch_loss=False,
    ):
        input_list = data["input"]
        input_label = data["label"]
        for i, input_tensor in enumerate(input_list):
            output_tensor = self._layers.forward(input_tensor)
            loss = self._layers._loss_fn[0](output_tensor, input_label[i])
            paddle.autograd.backward(
                tensors=loss,
                # grad_tensors=grad_tensors,
            )
        return loss


class PipelineParallel(nn.Layer):
    def __init__(self, layers, hcg, strategy):
        assert isinstance(layers, PipelineLayer)
        super().__init__()
        self._layers = layers
        self._strategy = strategy
        self._hcg = hcg

    def forward(self, data, scaler=None, return_micro_batch_loss=False):
        pass
