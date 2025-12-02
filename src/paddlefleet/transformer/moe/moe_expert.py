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

from paddlefleet.transformer.layer import FleetLayer
from paddlefleet.transformer.mlp import MLP, MLPSublayersSpec
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.transformer.transformer_config import TransformerConfig
from typing import Optional
import paddle
import paddle.nn.functional as F

class GroupedMLPExpert(FleetLayer):
    """An efficient implementation of the Experts layer using GroupedGEMM without TP/DP.

    Executes multiple experts in parallel using only expert parallelism.
    """

    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        experts: list,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)
        self.config: TransformerConfig = config
        self.config.hidden_act = F.silu
        self.num_local_experts = num_local_experts
        assert not config.add_bias_linear, (
            "bias not supported in Grouped GEMM yet, please set '--disable-bias-linear' instead."
        )

        self.ep_group = pg_collection.ep
        self.expert_parallel = self.ep_group.size() > 1 if self.ep_group else False

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

        # @jit_fuser
        # def activation_func_with_probs(x, probs):
        #     dtype = x.dtype
        #     res = self.activation_func(x) * probs
        #     return res.to(dtype)

        # self.activation_func_with_probs = activation_func_with_probs

        # No tensor parallel - full sizes
        fc1_output_size = (
            self.config.moe_ffn_hidden_size * self.num_local_experts
        )
        if config.gated_linear_unit:
            # Project to 4h. If using swiglu double the output width,
            # see https://arxiv.org/pdf/2002.05202.pdf
            fc1_output_size *= 2

        fc2_input_size = (
            self.config.moe_ffn_hidden_size * self.num_local_experts
        )

        # Initialize weight without partitioning
        # self.weight1 = paddle.create_parameter(
        #     shape=[self.config.hidden_size, fc1_output_size],
        #     dtype=config.params_dtype,
        #     default_initializer=paddle.nn.initializer.Uniform(),
        # )
        # self.weight2 = paddle.create_parameter(
        #     shape=[fc2_input_size, self.config.hidden_size],
        #     dtype=config.params_dtype,
        #     default_initializer=paddle.nn.initializer.Uniform(),
        # )
        print("GroupedMLPExpert experts.up_gate_proj.weight.shape ", experts[0].up_gate_proj.weight.shape)
        print("GroupedMLPExpert experts.down_proj.weight.shape ", experts[0].down_proj.weight.shape)
        weight1_list = [
            x.up_gate_proj.weight for x in experts if x is not None
        ]
        self.weight1 = paddle.stack(weight1_list, axis=0)
        weight2_list = [
            x.down_proj.weight for x in experts if x is not None
        ]
        self.weight2 = paddle.stack(weight2_list, axis=0)
        print("GroupedMLPExpert self.config.hidden_size ", self.config.hidden_size, ", fc1_output_size ", fc1_output_size, ", fc2_input_size ", fc2_input_size)
        print("GroupedMLPExpert Weight1 size ", self.weight1.shape)
        print("GroupedMLPExpert Weight2 size ", self.weight2.shape)

        # setattr(self.weight1, 'allreduce', not self.expert_parallel)
        # setattr(self.weight2, 'allreduce', not self.expert_parallel)

        # def remove_extra_states_check(self, incompatible_keys):
        #     """
        #     Remove _extra_state from unexpected keys.
        #     These keys are for dist ckpt compatibility with SequentialMLP.
        #     """
        #     keys = deepcopy(incompatible_keys.unexpected_keys)
        #     for key in keys:
        #         if '_extra_state' in key:
        #             print("WARNING: In GroupedMLPExpert, Removing extra state from {}".format(key))
        #             incompatible_keys.unexpected_keys.remove(key)

        # self.register_load_state_dict_post_hook(remove_extra_states_check)

    def forward(
        self,
        permuted_local_hidden_states: paddle.Tensor,
        tokens_per_expert: paddle.Tensor,
    ):
        """Forward step of the GroupedMLP without TP/DP."""

        if permuted_local_hidden_states.numel() != 0:
            tokens_per_expert = tokens_per_expert.cpu().tolist()
            tokens_per_expert = [int(x) for x in tokens_per_expert]

            fc1_output = paddle.incubate.nn.functional.legacy_batched_gemm(
                permuted_local_hidden_states,
                self.weight1,
                tokens_per_expert,
            )
            if self.activation_recompute:
                intermediate_parallel = self.activation_checkpoint.checkpoint(
                    self.activation_func, fc1_output
                )
                fc2_output = gg.ops.gmm(
                    intermediate_parallel, w2, tokens_per_expert, trans_b=False
                )
                self.activation_checkpoint.discard_output_and_register_recompute(
                    fc2_output
                )
            else:
                intermediate_parallel = self.activation_func(fc1_output)
                fc2_output = paddle.incubate.nn.functional.legacy_batched_gemm(
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
                h = self.activation_checkpoint.checkpoint(
                    self.activation_func, h
                )
                fc2_output = paddle.matmul(h, w2)
                self.activation_checkpoint.discard_output_and_register_recompute(
                    fc2_output
                )
            else:
                h = self.activation_func(h)
                fc2_output = paddle.matmul(h, w2)

        return fc2_output, None

    # @expert_dist_ckpt_decorator
    # def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
    #     """
    #     Maps local expert to global experts without DP/TP sharding.

    #     The sharded_state_dict for the weight parts are compatible with the SequentialMLP,
    #     as only expert parallelism is considered.

    #     When `singleton_local_shards` metadata flag is True, experts are broken down into
    #     separate tensors and stored under separate global keys. Additionally, similarly to MLP,
    #     layers with GLU activations are broken down into separate `w` and `v` tensors.
    #     """
    #     # singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)
    #     singleton_local_shards = True
    #     sharded_state_dict = {}
    #     ep_size = self.ep_group.size()
    #     ep_rank = self.ep_group.rank()
    #     num_global_experts = ep_size * self.num_local_experts
    #     local_expert_indices_offset = ep_rank * self.num_local_experts

    #     prepend_axis_num = len(sharded_offsets)
    #     replica_id = (0, 0, 0)  # Simplified replica_id without TP/DP

    #     def _break_into_individual_experts(
    #         experts_ten: paddle.Tensor, key: str, replica_id: ReplicaId
    #     ):
    #         """Breaks experts into individual tensors and stores them under separate global keys"""
    #         experts_state = []
    #         for local_expert_idx, expert_ten in enumerate(experts_ten):
    #             global_expert_idx = (
    #                 local_expert_indices_offset + local_expert_idx
    #             )
    #             expert_key = key.replace(
    #                 f"{prefix}experts.", f"{prefix}experts.{global_expert_idx}."
    #             )
    #             if singleton_local_shards:
    #                 # Store each expert as individual shard
    #                 experts_state.append(
    #                     ShardedTensor.from_rank_offsets(
    #                         expert_key,
    #                         expert_ten.contiguous(),
    #                         *sharded_offsets,
    #                         (
    #                             prepend_axis_num,
    #                             global_expert_idx,
    #                             num_global_experts,
    #                         ),
    #                         replica_id=replica_id,
    #                         prepend_axis_num=prepend_axis_num,
    #                     )
    #                 )
    #             else:
    #                 # Store as single tensor with expert dimension
    #                 if local_expert_idx == 0:
    #                     # Create a sharded tensor for all experts
    #                     experts_ten_stack = paddle.stack(experts_ten)
    #                     experts_state.append(
    #                         ShardedTensor.from_rank_offsets(
    #                             key,
    #                             experts_ten_stack.contiguous(),
    #                             *sharded_offsets,
    #                             (prepend_axis_num, ep_rank, ep_size),
    #                             replica_id=replica_id,
    #                             prepend_axis_num=prepend_axis_num,
    #                         )
    #                     )
    #                     break
    #         return experts_state

    #     @paddle.no_grad()
    #     def sh_ten_build_fn(
    #         key: str,
    #         t: paddle.Tensor,
    #         replica_id: ReplicaId,
    #         flattened_range: Optional[slice],
    #         with_glu: bool,
    #     ):
    #         """
    #         Simplified build function without TP parallelism.
    #         """
    #         if flattened_range is None:
    #             # Handle full weight tensors
    #             if singleton_local_shards:
    #                 if with_glu:
    #                     # Split GLU into w and v components
    #                     real_shape = (
    #                         self.num_local_experts,
    #                         self.config.hidden_size,
    #                         -1,
    #                     )
    #                     t = t.reshape(real_shape)
    #                     w_tensor, v_tensor = paddle.chunk(t, 2, dim=-1)

    #                     w_key = f"{key}_w"
    #                     v_key = f"{key}_v"
    #                     sub_states = {
    #                         "singleton_local_shards": LocalNonpersistentObject(
    #                             True
    #                         ),
    #                         "data": {
    #                             "w": _break_into_individual_experts(
    #                                 w_tensor, w_key, replica_id
    #                             ),
    #                             "v": _break_into_individual_experts(
    #                                 v_tensor, v_key, replica_id
    #                             ),
    #                         },
    #                     }
    #                 else:
    #                     # Regular case without GLU
    #                     real_shape = (
    #                         self.num_local_experts,
    #                         self.config.hidden_size,
    #                         -1,
    #                     )
    #                     t = t.reshape(real_shape)
    #                     sub_states = {
    #                         "singleton_local_shards": LocalNonpersistentObject(
    #                             True
    #                         ),
    #                         "data": _break_into_individual_experts(
    #                             t, key, replica_id
    #                         ),
    #                     }
    #             else:
    #                 # Non-singleton case - store all experts together
    #                 real_shape = (
    #                     self.num_local_experts,
    #                     self.config.hidden_size,
    #                     -1,
    #                 )
    #                 t = t.reshape(real_shape)
    #                 if with_glu:
    #                     w_tensor, v_tensor = paddle.chunk(t, 2, dim=-1)
    #                     # For simplicity, we'll store as separate tensors without TP sharding
    #                     sub_states = [
    #                         ShardedTensor.from_rank_offsets(
    #                             key,
    #                             w_tensor.contiguous(),
    #                             *sharded_offsets,
    #                             (prepend_axis_num, ep_rank, ep_size),
    #                             replica_id=replica_id,
    #                             prepend_axis_num=prepend_axis_num,
    #                         ),
    #                         ShardedTensor.from_rank_offsets(
    #                             key,
    #                             v_tensor.contiguous(),
    #                             *sharded_offsets,
    #                             (prepend_axis_num, ep_rank, ep_size),
    #                             replica_id=replica_id,
    #                             prepend_axis_num=prepend_axis_num,
    #                         ),
    #                     ]
    #                 else:
    #                     sub_states = ShardedTensor.from_rank_offsets(
    #                         key,
    #                         t.contiguous(),
    #                         *sharded_offsets,
    #                         (prepend_axis_num, ep_rank, ep_size),
    #                         replica_id=replica_id,
    #                         prepend_axis_num=prepend_axis_num,
    #                     )
    #         else:
    #             # Handle flattened optimizer states (simplified)
    #             if singleton_local_shards:
    #                 raise NotImplementedError(
    #                     "flattened_range not supported for GroupedMLP without TP"
    #                 )
    #             else:
    #                 # Simplified flattened state handling without TP
    #                 raise NotImplementedError(
    #                     "flattened_range handling for GroupedMLP without TP is complex, "
    #                     "consider using non-flattened states"
    #                 )
    #         return sub_states

    #     @paddle.no_grad()
    #     def sh_ten_merge_fn(sub_state_dict, with_glu: bool):
    #         """Simplified merge function without TP."""
    #         if isinstance(sub_state_dict, dict) and sub_state_dict.get(
    #             "singleton_local_shards"
    #         ):
    #             # Merge from singleton shards
    #             if with_glu:
    #                 w_tensors = sub_state_dict["data"]["w"]
    #                 v_tensors = sub_state_dict["data"]["v"]
    #                 experts_tensors = []
    #                 for w, v in zip(w_tensors, v_tensors):
    #                     expert_tensor = paddle.cat([w, v], dim=-1)
    #                     experts_tensors.append(expert_tensor)
    #                 merged = paddle.stack(experts_tensors)
    #             else:
    #                 merged = paddle.stack(sub_state_dict["data"])
    #         else:
    #             if isinstance(sub_state_dict, list):
    #                 # Handle list of tensors (GLU case)
    #                 if with_glu and len(sub_state_dict) == 2:
    #                     w_part, v_part = sub_state_dict
    #                     experts_tensors = []
    #                     for expert_idx in range(w_part.shape[0]):
    #                         w_expert = w_part[expert_idx]
    #                         v_expert = v_part[expert_idx]
    #                         expert_tensor = paddle.cat(
    #                             [w_expert, v_expert], dim=-1
    #                         )
    #                         experts_tensors.append(expert_tensor)
    #                     merged = paddle.stack(experts_tensors)
    #                 else:
    #                     merged = (
    #                         paddle.cat(sub_state_dict, dim=0)
    #                         if isinstance(sub_state_dict[0], paddle.Tensor)
    #                         else sub_state_dict
    #                     )
    #             else:
    #                 merged = sub_state_dict

    #         # Reshape to original weight format
    #         if isinstance(merged, paddle.Tensor) and merged.dim() == 3:
    #             # Reshape from (num_experts, hidden_size, ffn_size) to flat weight
    #             if with_glu:
    #                 weight_shape = (self.config.hidden_size, -1)
    #             else:
    #                 weight_shape = (-1, self.config.hidden_size)
    #             merged = merged.reshape(-1, self.config.hidden_size)

    #         return merged

    #     state_dict = self.state_dict(prefix="", keep_vars=True)
    #     for name, tensor in state_dict.items():
    #         if name == "weight1":
    #             with_glu = self.config.gated_linear_unit
    #             wkey = f"{prefix}experts.linear_fc1.weight"
    #         else:
    #             with_glu = False
    #             wkey = f"{prefix}experts.linear_fc2.weight"

    #         this_replica_id = list(copy.deepcopy(replica_id))
    #         flattened_range = None

    #         sharded_state_dict[f"{prefix}{name}"] = ShardedTensorFactory(
    #             wkey,
    #             tensor,
    #             partial(sh_ten_build_fn, with_glu=with_glu),
    #             partial(sh_ten_merge_fn, with_glu=with_glu),
    #             tuple(this_replica_id),
    #             flattened_range=flattened_range,
    #         )

    #     # Add fake _extra_state to be compatible with SequentialMLP
    #     for expert_local_idx in range(self.num_local_experts):
    #         expert_global_idx = local_expert_indices_offset + expert_local_idx
    #         if singleton_local_shards:
    #             expert_sharded_offsets = sharded_offsets
    #         else:
    #             expert_sharded_offsets = (
    #                 *sharded_offsets,
    #                 (
    #                     len(sharded_offsets),
    #                     expert_global_idx,
    #                     num_global_experts,
    #                 ),
    #             )
    #         for mod in ["linear_fc1", "linear_fc2"]:
    #             if singleton_local_shards:
    #                 expert_key = f"{prefix}experts.{expert_global_idx}.{mod}._extra_state"
    #             else:
    #                 expert_key = f"{prefix}experts.{mod}._extra_state"
    #             sharded_state_dict[
    #                 f"{prefix}expert{expert_global_idx}.{mod}._extra_state"
    #             ] = make_sharded_object_for_checkpoint(
    #                 None, expert_key, expert_sharded_offsets, replica_id
    #             )

    #     return sharded_state_dict

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
        fuse_up_gate: bool,
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
