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


from copy import deepcopy

import paddle
import paddle.nn.functional as F

from paddlefleet import utils
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.transformer.layer import FleetLayer
from paddlefleet.transformer.mlp import MLP, MLPSublayersSpec
from paddlefleet.transformer.transformer_config import TransformerConfig


class GroupedMLPExpert(FleetLayer):
    """An efficient implementation of the Experts layer using GroupedGEMM without TP/DP.

    Executes multiple experts in parallel using only expert parallelism.
    """

    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        experts: list,
        pg_collection: ProcessGroupCollection | None = None,
    ):
        super().__init__(config=config)
        self.config: TransformerConfig = config
        self.config.hidden_act = F.silu
        self.num_local_experts = num_local_experts
        assert not config.use_bias, (
            "Bias not supported in Grouped GEMM yet, please set 'use_bias' to False."
        )

        self.ep_group = pg_collection.ep if pg_collection else None
        self.expert_parallel = (
            utils.get_pg_size(self.ep_group) > 1 if self.ep_group else False
        )

        if self.config.gated_linear_unit:
            if self.config.hidden_act not in [F.silu, F.gelu]:
                raise ValueError(
                    "Activation function must be silu or gelu when using GroupedMLP."
                )

            def glu(x):
                x = paddle.chunk(x, 2, dim=-1)
                return self.config.hidden_act(x[0]) * x[1]

            self.activation_func = glu
        else:
            self.activation_func = self.config.hidden_act
        self.activation_recompute = (
            self.config.recompute_granularity == "selective"
            and "moe_act" in self.config.recompute_modules
        )
        if self.activation_recompute and self.config.fp8:
            raise ValueError(
                "moe_act recompute for fp8 cannot work with the legacy GroupedMLP."
            )

        # No tensor parallel - full sizes
        fc1_output_size = (
            self.config.moe_intermediate_size * self.num_local_experts
        )
        if config.gated_linear_unit:
            # Project to 4h. If using swiglu double the output width,
            # see https://arxiv.org/pdf/2002.05202.pdf
            fc1_output_size *= 2

        fc2_input_size = (
            self.config.moe_intermediate_size * self.num_local_experts
        )

        weight1_list = [x.up_gate_proj.weight for x in experts if x is not None]
        self.weight1 = paddle.stack(weight1_list, axis=0)
        weight2_list = [x.down_proj.weight for x in experts if x is not None]
        self.weight2 = paddle.stack(weight2_list, axis=0)

    def forward(
        self,
        permuted_local_hidden_states: paddle.Tensor,
        tokens_per_expert: paddle.Tensor,
    ):
        """Forward step of the GroupedMLP without TP/DP."""

        if permuted_local_hidden_states.numel() != 0:
            tokens_per_expert = tokens_per_expert.cpu().tolist()
            tokens_per_expert = [int(x) for x in tokens_per_expert]

            fc1_output = paddle.incubate.nn.functional.batched_gemm(
                permuted_local_hidden_states,
                self.weight1,
                tokens_per_expert,
            )
            if self.activation_recompute:
                raise NotImplementedError(
                    "Recompute in GroupedMLPExpert is not implemented"
                )
            else:
                intermediate_parallel = self.activation_func(fc1_output)
                fc2_output = paddle.incubate.nn.functional.batched_gemm(
                    intermediate_parallel, self.weight2, tokens_per_expert
                )
        else:
            # No token is allocated for local experts.
            assert paddle.count_nonzero(tokens_per_expert) == 0

            # Make sure params of experts still have gradients even given zero tokens.
            w1 = self.weight1.reshape(self.config.hidden_size, -1)
            w2 = self.weight2.reshape(-1, self.config.hidden_size)
            h = paddle.matmul(permuted_local_hidden_states, w1)
            if self.activation_recompute:
                raise NotImplementedError(
                    "Recompute in GroupedMLPExpert is not implemented"
                )
            else:
                h = self.activation_func(h)
                fc2_output = paddle.matmul(h, w2)

        return fc2_output, None

    def backward_dw(self):
        """Performs backward pass for weight gradients in Experts.
        Empty implementation for compatibility with SequentialMLP and TEGroupedMLP.
        """
        pass


class StandardMLPExpert(MLP):
    def __init__(
        self,
        config: TransformerConfig,
        moe_intermediate_size: int,
        is_expert: bool,
        mlp_spec: MLPSublayersSpec,
    ):
        if moe_intermediate_size == config.intermediate_size:
            super().__init__(
                config,
                mlp_spec,
                is_expert=is_expert,
                intermediate_size=moe_intermediate_size,
                # tp_group=pg_collection.expt_tp,
            )
        else:
            # Local SequentialMLP can still be used here by overriding the intermediate_size
            # with a deepcopied config.
            sequential_mlp_config = deepcopy(config)
            sequential_mlp_config.intermediate_size = moe_intermediate_size
            super().__init__(
                sequential_mlp_config,
                mlp_spec,
                is_expert=is_expert,
                intermediate_size=moe_intermediate_size,
                # tp_group=pg_collection.expt_tp,
            )
