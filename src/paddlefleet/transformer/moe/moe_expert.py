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
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
    build_sharded_state_dict,
    shard_weight,
)

from paddlefleet import utils
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.transformer.layer import FleetLayer
from paddlefleet.transformer.mlp import MLP, MLPSublayersSpec
from paddlefleet.transformer.transformer_config import TransformerConfig

from .moe_utils import k_grouped_bf16_gemm_tn_contiguous_aligned

try:
    from paddlefleet.ops import deep_gemm as paddlefleet_deep_gemm
except (ImportError, RuntimeError):
    pass


class BMMFunction(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, y, batch_sizes, trans_y=False):
        ctx.save_for_backward(x, y)
        ctx.batch_sizes = batch_sizes
        ctx.trans_y = trans_y
        return paddle.incubate.nn.functional.batched_gemm(
            x, y, batch_sizes, trans_rhs=trans_y
        )

    @staticmethod
    def backward(ctx, grad):
        x, y = ctx.saved_tensor()
        batch_sizes = ctx.batch_sizes
        trans_y = ctx.trans_y

        dx = None
        dx = paddle.incubate.nn.functional.batched_gemm(
            grad, y, batch_sizes, trans_rhs=not trans_y
        )

        dy = None
        lhs, rhs = (grad, x) if trans_y else (x, grad)
        dy = paddle.incubate.nn.functional.batched_gemm(
            lhs, rhs, batch_sizes, trans_lhs=True, trans_rhs=False
        )
        return dx, dy


class DeepGEMMBMMFunction(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, y, batch_sizes):
        ctx.save_for_backward(x, y)
        ctx.batch_sizes = batch_sizes
        out = paddle.zeros([x.shape[0], y.shape[2]], dtype="bfloat16")

        tokens_per_expert_indices = paddle.repeat_interleave(
            paddle.arange(batch_sizes.shape[0]), batch_sizes
        ).cast("int32")

        paddlefleet_deep_gemm.m_grouped_bf16_gemm_nn_contiguous(
            x, y, out, tokens_per_expert_indices
        )

        del tokens_per_expert_indices
        return out

    @staticmethod
    def backward(ctx, grad):
        x, y = ctx.saved_tensor()
        batch_sizes = ctx.batch_sizes

        tokens_per_expert_indices = paddle.repeat_interleave(
            paddle.arange(batch_sizes.shape[0]), batch_sizes
        ).cast("int32")

        dx = paddle.zeros_like(x)
        paddlefleet_deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            grad,
            y,
            dx,
            tokens_per_expert_indices,
        )
        dx = paddle.cast(dx, paddle.float)

        dy = paddle.zeros_like(y, dtype=paddle.float)
        k_grouped_bf16_gemm_tn_contiguous_aligned(
            a=x,
            b=grad,
            d=dy,
            ks=paddle.tolist(batch_sizes),
            ks_tensor=batch_sizes.cast("int32"),
            c=paddle.zeros_like(y, dtype=paddle.float),
        )

        del tokens_per_expert_indices
        return dx, dy


class GroupedMLPExpert(FleetLayer):
    """An efficient implementation of the Experts layer using GroupedGEMM without TP/DP.

    Executes multiple experts in parallel using only expert parallelism.
    """

    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        moe_deep_gemm,
        pg_collection: ProcessGroupCollection | None = None,
    ):
        super().__init__(config=config)
        self.config: TransformerConfig = config
        self.config.hidden_act = F.silu
        self.num_local_experts = num_local_experts
        self.moe_deep_gemm = moe_deep_gemm
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
        fc1_output_size = self.config.moe_intermediate_size
        if config.gated_linear_unit:
            # Project to 4h. If using swiglu double the output width,
            # see https://arxiv.org/pdf/2002.05202.pdf
            fc1_output_size *= 2

        fc2_input_size = self.config.moe_intermediate_size

        self.weight1 = paddle.create_parameter(
            shape=[
                self.num_local_experts,
                self.config.hidden_size,
                fc1_output_size,
            ],
            dtype="bfloat16",
            default_initializer=paddle.nn.initializer.Uniform(-0.001, 0.001),
            # default_initializer=paddle.nn.initializer.Normal(mean=0.0, std=0.01),
        )
        self.weight1.is_distributed = self.expert_parallel
        self.weight2 = paddle.create_parameter(
            shape=[
                self.num_local_experts,
                fc2_input_size,
                self.config.hidden_size,
            ],
            dtype="bfloat16",
            default_initializer=paddle.nn.initializer.Uniform(-0.001, 0.001),
            # default_initializer=paddle.nn.initializer.Normal(mean=0.0, std=0.01),
        )
        self.weight2.is_distributed = self.expert_parallel

    def forward(
        self,
        permuted_local_hidden_states: paddle.Tensor,
        tokens_per_expert: paddle.Tensor,
    ):
        """Forward step of the GroupedMLP without TP/DP."""

        if permuted_local_hidden_states.numel() != 0:
            tokens_per_expert = tokens_per_expert.cpu().tolist()
            tokens_per_expert = [int(x) for x in tokens_per_expert]

            if self.moe_deep_gemm:
                fc1_output = DeepGEMMBMMFunction.apply(
                    permuted_local_hidden_states,
                    self.weight1,
                    paddle.to_tensor(tokens_per_expert, dtype="int32"),
                )
            else:
                fc1_output = BMMFunction.apply(
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
                if self.moe_deep_gemm:
                    fc2_output = DeepGEMMBMMFunction.apply(
                        intermediate_parallel,
                        self.weight2,
                        paddle.to_tensor(tokens_per_expert, dtype="int32"),
                    )
                else:
                    fc2_output = BMMFunction.apply(
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

    def sharded_state_dict(
        self,
        structured_name_prefix: str = "",
    ):
        state_dict = self.state_dict(structured_name_prefix="")
        w1 = state_dict["weight1"].reshape(-1, self.weight1.shape[-1])
        w2 = state_dict["weight2"].reshape(-1, self.weight2.shape[-1])
        w1.name = self.weight1.name
        w2.name = self.weight2.name
        state_dict["weight1"] = w1
        state_dict["weight2"] = w2
        sharded_dict = {}
        full_key1 = f"{structured_name_prefix}weight1"
        full_key2 = f"{structured_name_prefix}weight2"
        if self.ep_group is None:
            sharded_dict = build_sharded_state_dict(
                state_dict, None, structured_name_prefix
            )
        else:
            sharded_dict[full_key1] = shard_weight(
                key=full_key1,
                weight=state_dict["weight1"],
                axis=0,
                group=self.ep_group,
            )
            sharded_dict[full_key2] = shard_weight(
                key=full_key2,
                weight=state_dict["weight2"],
                axis=0,
                group=self.ep_group,
            )
        return sharded_dict


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
