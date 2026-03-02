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
from paddle.distributed.fleet.utils import recompute

from paddlefleet.context_parallel_utils import ContextParallelGatherOp
from paddlefleet.parallel_state import (
    get_context_parallel_world_size,
    get_tensor_model_parallel_world_size,
)
from paddlefleet.pipeline_parallel import ScheduleNode
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.transformer.layer import FleetLayer
from paddlefleet.transformer.transformer_config import TransformerConfig


class LanguageLoss(FleetLayer):
    # Class-level tracker for MTP loss, read by trainer for logging.
    mtp_loss_tracker: dict[str, float] = {}

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection=None,
    ) -> None:
        super().__init__(config=config)
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection

        self.config = config
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

    def forward_impl(self, logits: Tensor, labels: Tensor) -> Tensor:
        loss = self.loss_func(logits.cast("float32"), labels)

        if get_context_parallel_world_size() > 1:
            loss = ContextParallelGatherOp.apply(loss, axis=1)
            labels = ContextParallelGatherOp.apply(labels, axis=1)

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

    def _forward(self, logits: Tensor, labels: Tensor):
        if (
            self.config.recompute_modules is not None
            and "loss_fn" in self.config.recompute_modules
        ):
            return recompute(self.forward_impl, logits, labels)
        return self.forward_impl(logits, labels)

    def forward(self, logits: Tensor | list, labels: Tensor) -> Tensor:
        if isinstance(logits, list):
            assert (
                self.config.num_nextn_predict_layers is not None
                and self.config.num_nextn_predict_layers > 0
            )
            assert len(logits) == self.config.num_nextn_predict_layers + 1
            labels_ori = labels
            lm_labels = labels[:, : -self.config.num_nextn_predict_layers]
            seq_length = lm_labels.shape[1]

            lm_loss = self._forward(logits[0], lm_labels)

            mtp_loss = []
            mtp_logits = logits[1:]
            for depth in range(self.config.num_nextn_predict_layers):
                logits_cur_depth = mtp_logits[depth]
                labels_cur_depth = labels_ori[
                    :, (depth + 1) : (depth + 1 + seq_length)
                ]
                loss_cur_depth = self._forward(
                    logits_cur_depth,
                    labels_cur_depth,
                )
                mtp_loss.append(loss_cur_depth)

            # Store detached MTP loss tensors into class-level tracker.
            # Use .detach() instead of .item() to avoid GPU synchronization on every
            # micro-batch. The trainer will call .item() only at logging steps.
            for i, loss_val in enumerate(mtp_loss):
                LanguageLoss.mtp_loss_tracker[f"mtp_{i + 1}_loss"] = (
                    loss_val.detach()
                )

            def add_loss(main_loss, loss):
                if self.config.add_mtp_loss:
                    return main_loss + loss - loss.detach()
                else:
                    return main_loss

            loss = add_loss(
                lm_loss,
                self.config.mtp_loss_scaling_factor
                * sum(mtp_loss)
                / len(mtp_loss),
            )

            return loss
        else:
            return self._forward(logits, labels)

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="LanguageLoss")
