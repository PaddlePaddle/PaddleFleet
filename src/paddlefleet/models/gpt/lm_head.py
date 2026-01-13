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
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
    build_sharded_state_dict,
)
from paddle.nn.parameter import Parameter

from paddlefleet.pipeline_parallel import ScheduleNode
from paddlefleet.tensor_parallel.layers import (
    ColumnParallelLinear,
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
)


class GPTLMHead(ColumnParallelLinear):
    def __init__(self, **kwargs):
        self.config = kwargs["config"]
        self.skip_weight_param_allocation = kwargs[
            "skip_weight_param_allocation"
        ]

        kwargs["skip_weight_param_allocation"] = True
        super().__init__(**kwargs)

        stride = kwargs["stride"] if "stride" in kwargs.keys() else 1
        init_method = kwargs["init_method"]
        keep_master_weight_for_test = (
            kwargs["keep_master_weight_for_test"]
            if "keep_master_weight_for_test" in kwargs.keys()
            else False
        )

        if not self.skip_weight_param_allocation:
            if self.config.use_cpu_initialization:
                self.weight = Parameter(
                    paddle.empty(
                        [self.output_size_per_partition, self.input_size],
                        dtype=self.config.params_dtype,
                    )
                )
                if self.config.perform_initialization:
                    self.master_weight = _initialize_affine_weight_cpu(
                        self.weight,
                        self.output_size,
                        self.input_size,
                        self.output_size_per_partition,
                        0,
                        init_method,
                        stride=stride,
                        return_master_weight=keep_master_weight_for_test,
                        rank=self.rank,
                        world_size=self.world_size,
                    )
            else:
                self.weight = Parameter(
                    paddle.empty(
                        [self.output_size_per_partition, self.input_size],
                        dtype=self.config.params_dtype,
                    )
                )
                if self.config.perform_initialization:
                    _initialize_affine_weight_gpu(
                        self.weight,
                        init_method,
                        partition_dim=0,
                        stride=stride,
                        is_expert=self.is_expert,
                    )
            self.weight.is_distributed = True if self.world_size > 1 else False

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="GPTLMHead")

    def forward(self, dict_args: dict):
        hidden_states = dict_args["hidden_states"]
        if (
            self.config.recompute_modules is not None
            and "lm_head" in self.config.recompute_modules
        ):
            recompute_func = super().forward

            def recompute_handler(hidden_states, weight):
                logits, _ = recompute_func(hidden_states, weight)
                return logits

            logits = recompute_handler(hidden_states, self.weight.T)
        else:
            logits, _ = super().forward(hidden_states, self.weight.T)
        if self.config.sequence_parallel:
            logits = logits.transpose([1, 0, 2]).contiguous()
        return logits

    @property
    def embedding_weight(self):
        return self.weight

    def sharded_state_dict(
        self,
        structured_name_prefix: str = "",
    ):
        """Sharding along axis 0, bias sharded"""
        state_dict = self.state_dict(structured_name_prefix="")
        shard_rules = None if self.world_size == 1 else {"weight": 0, "bias": 0}
        return build_sharded_state_dict(
            state_dict, shard_rules, structured_name_prefix
        )
