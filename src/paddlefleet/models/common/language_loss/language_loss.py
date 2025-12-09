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

from paddlefleet.parallel_state import get_tensor_model_parallel_world_size
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.transformer.layer import FleetLayer
from paddlefleet.transformer.transformer_config import TransformerConfig


class LanguageLoss(FleetLayer):
    def __init__(
        self,
        config: TransformerConfig,
        pg_collection=None,
    ) -> None:
        super().__init__(config=config)
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection

        self.ignored_index = -100
        self.enable_parallel_cross_entropy = (
            paddle.distributed.is_initialized()
            and get_tensor_model_parallel_world_size() > 1
            and config.parallel_output
        )

        if self.enable_parallel_cross_entropy:
            self.loss_func = (
                paddle.distributed.fleet.meta_parallel.ParallelCrossEntropy()
            )
        else:
            self.loss_func = paddle.nn.CrossEntropyLoss(
                reduction="none",
            )

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        loss = self.loss_func(logits.cast("float32"), labels)

        lossmask = labels != self.ignored_index
        if (~lossmask).all():
            loss = paddle.mean(loss) * 0.0
        else:
            lossmask = lossmask.reshape([-1]).cast(paddle.float32)
            loss = paddle.sum(
                loss.cast(paddle.float32).reshape([-1]) * lossmask
            )
            loss = loss / lossmask.sum()

        return loss
