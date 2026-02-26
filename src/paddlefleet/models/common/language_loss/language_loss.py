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
import paddle.distributed as dist
from paddle import Tensor
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.distributed.fleet.layers.mpu import mp_ops
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import AllGatherOp

from paddlefleet.context_parallel_utils import (
    ContextParallelGatherOp,
    ContextParallelScatterOp,
)
from paddlefleet.parallel_state import (
    get_context_parallel_world_size,
    get_tensor_model_parallel_world_size,
)
from paddlefleet.pipeline_parallel import ScheduleNode
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.transformer.layer import FleetLayer
from paddlefleet.transformer.transformer_config import TransformerConfig


class DistributedSoftmaxOp(PyLayer):
    @staticmethod
    def forward(ctx, x, axis=-1, mp_group=None):
        ctx.axis = axis
        ctx.mp_group = mp_group

        if mp_group is None:
            hcg = fleet.get_hybrid_communicate_group()
            ctx.mp_group = hcg.get_model_parallel_group()

        local_max = paddle.max(x, axis=axis, keepdim=True)

        all_max = AllGatherOp.apply(local_max, axis=0, group=ctx.mp_group)

        global_max = paddle.max(all_max, axis=0, keepdim=True)

        x_stable = x - global_max

        exp_x = paddle.exp(x_stable.cast("float32"))

        local_sum_exp = paddle.sum(exp_x, axis=axis, keepdim=True)

        sum_exp = mp_ops._mp_allreduce(
            local_sum_exp,
            group=mp_group,
            use_calc_stream=True,
            use_model_parallel=True,
        )

        softmax_output = exp_x / sum_exp

        ctx.save_for_backward(softmax_output, sum_exp)

        return softmax_output

    @staticmethod
    def backward(ctx, grad_output):
        softmax_output, global_sum_exp = ctx.saved_tensor()
        axis = ctx.axis
        mp_group = ctx.mp_group

        grad_softmax = grad_output * softmax_output

        local_sum_grad = paddle.sum(grad_softmax, axis=axis, keepdim=True)

        all_sum_grad = AllGatherOp.apply(local_sum_grad, axis=0, group=mp_group)
        global_sum_grad = paddle.sum(all_sum_grad, axis=0, keepdim=True)

        grad_input = softmax_output * (grad_output - global_sum_grad)

        return grad_input


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

            mtp_loss = []
            mtp_logits = logits[1:]

            # hardcoding
            # mtp_distillation = True
            mtp_distillation = False
            if not mtp_distillation:
                lm_loss = self._forward(logits[0], lm_labels)
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
            else:
                lm_loss = 0.0
                target_p_self_op_dist = DistributedSoftmaxOp.apply(
                    logits[0], axis=2
                )
                print("target_p_self_op_dist: ", target_p_self_op_dist)
                if get_context_parallel_world_size() > 1:
                    target_p_self_op_dist = ContextParallelGatherOp.apply(
                        target_p_self_op_dist, axis=1
                    )

                def padding(tensor, left=False, pad_len=1):
                    zeropadding = paddle.zeros_like(tensor[:, -pad_len:, :])
                    if left:
                        tensor = paddle.concat((zeropadding, tensor), axis=1)
                    else:
                        tensor = paddle.concat((tensor, zeropadding), axis=1)
                    return tensor

                if (
                    self.config.num_nextn_predict_layers > 0
                    and mtp_logits is not None
                ):
                    for depth in range(len(mtp_logits)):
                        prediction_scores_cur_depth = mtp_logits[depth]
                        labels_cur_depth = labels_ori[
                            :, (depth + 1) : (depth + 1 + seq_length)
                        ]
                        lossmask = (
                            labels_cur_depth != self.ignored_index
                        ).cast(paddle.float32)
                        print(
                            "mtp_prediction_scores_cur_depth: ",
                            prediction_scores_cur_depth,
                        )
                        print(
                            "mtp_prediction_scores_cur_depth: ",
                            prediction_scores_cur_depth.shape,
                        )
                        print("mtp_lossmask: ", lossmask.shape)
                        out_logp = paddle.log(
                            DistributedSoftmaxOp.apply(
                                prediction_scores_cur_depth, axis=2
                            )
                        )
                        print("out_logp: ", out_logp)

                        target_p = target_p_self_op_dist[
                            :, (depth + 1) :, :
                        ].clone()
                        target_p = padding(
                            target_p, left=False, pad_len=depth + 1
                        )
                        print("target_p: ", target_p)
                        if get_context_parallel_world_size() > 1:
                            target_p = ContextParallelScatterOp.apply(
                                target_p, axis=1
                            )
                        plogp = target_p * out_logp
                        print("plogp: ", plogp)

                        lossmask = lossmask[..., None]
                        xishu = lossmask.sum() + 1e-5
                        print("xishu: ", xishu)
                        if get_context_parallel_world_size() > 1:
                            lossmask = ContextParallelScatterOp.apply(
                                lossmask, axis=1
                            )

                        ploss = -paddle.sum(paddle.sum(lossmask * plogp, 2))
                        print("ploss: ", ploss)

                        if get_context_parallel_world_size() > 1:
                            dist.all_reduce(
                                ploss,
                                group=fleet.get_hybrid_communicate_group().get_context_parallel_group(),
                            )

                        ploss = ploss / xishu
                        mtp_loss.append(ploss)

                        # loss_matrix_cur_depth = loss_matrix_cur_depth.cast(paddle.float32) * lossmask_cur_depth

            print(f"[Ting LOG] mtp loss: {mtp_loss}")

            def add_loss(main_loss, loss):
                print(f"[Ting LOG] main_loss: {main_loss}, add_loss: {loss}")
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
