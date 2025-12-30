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

# Referred to NVIDIA Megatron-LM https://github.com/NVIDIA/Megatron-LM.git
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.


from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import paddle.nn.functional as F

from ..model_parallel_config import ModelParallelConfig
from ..utils import init_method_normal, scaled_init_method_normal

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class TransformerConfig(ModelParallelConfig):
    """Configuration object for transformers."""

    ####################
    # model architecture
    ####################

    num_hidden_layers: int = 1
    """Number of transformer layers in a transformer block."""

    num_nextn_predict_layers: int = None
    """Number of Multi-Token Prediction (MTP) Layers."""

    mtp_loss_scaling_factor: float = None
    """Weighting factor of Multi-Token Prediction (MTP) loss."""

    num_empty_layers_add_in_head: int = 0
    """Number of EmptyLayer before the Decoder Layer.
    num_empty_layers_add_in_head=2 Example:
        EmptyLayer, EmptyLayer, Decoder, Dcoder, ...
    0 implies equal layer division across PP ranks."""

    num_empty_layers_add_in_tail: int = 0
    """Number of EmptyLayer after the Decoder Layer.
    num_empty_layers_add_in_tail=2 Example:
        ..., Decoder, Dcoder, EmptyLayer, EmptyLayer
    0 implies equal layer division across PP ranks."""

    # Note: need to implement PipelineParallelLayerLayout and import
    # pipeline_model_parallel_layout: str | list | PipelineParallelLayerLayout = None
    pipeline_model_parallel_layout: str | list = None
    """Custom definition of the pipeline parallel partitioning.
    Support type:
    - str: e.g., 'Et*3|(tt|)*29,m|L'. Stages are split by '|', replicated stages or layers
    can be described with multiplication. Commas can be used cosmetically.
    - list: e.g., [['embedding', 'decoder'], ['decoder', 'decoder', 'decoder', 'loss']].
    - PipelineParallelLayerLayout: a PipelineParallelLayerLayout object.
    If given either a string or a list, it will be transferred into a PipelineParallelLayerLayout
    in post init. Let i = a * pp_size + b, then layout[i] gives a list of the layers
    in the a-th vpp stage and the b-th pp stage, i.e., vpp(0)pp(0), vpp(0)pp(1), ...,
    vpp(i)pp(j), vpp(i)pp(j+1), ..., vpp(-1)pp(-2), vpp(-1)pp(-1).
    In the inner lists of layers, 'embedding' or 'E' denotes the embedding layer, 'loss' or 'L'
    denotes the loss function, and 'decoder' or 't' denotes the transformer decoder layer.
    Examples:
        [['embedding', 'decoder'], ['decoder', 'decoder', 'decoder', 'loss']]:
        pp = 2, vpp = None
        pp rank 0 holds: embedding, decoder
        pp rank 1 holds: decoder*3, loss
        'E|(tt|)*2,(t|)*4,mL':
        pp = 2, vpp = 4
        vpp rank 0 pp rank 0 holds: embedding
        vpp rank 0 pp rank 1~2 holds: decoder*2
        vpp rank 0 pp rank 3 holds: decoder
        vpp rank 1 pp rank 0~2 holds: decoder
        vpp rank 1 pp rank 3 holds: mtp, loss"""

    account_for_embedding_in_pipeline_split: bool = False
    """If set, the embedding layer will be treated as a standard transformer
    layer in the context of partition and placement for pipeline parallelism."""

    account_for_loss_in_pipeline_split: bool = False
    """If set, the loss layer will be treated as a standard transformer
    layer in the context of partition and placement for pipeline parallelism."""

    hidden_size: int = 0
    """Transformer hidden size."""

    num_attention_heads: int = 1
    """Number of transformer attention heads."""

    softmax_scale: float = None
    """Softmax scale for attention scaling."""

    softmax_type: Literal["vanilla", "off-by-one", "learnable"] = "vanilla"
    """Applies modified softmax from https://www.evanmiller.org/attention-is-off-by-one.html.
       Supports both TE FusedAttention and local unfused attention. Supports both a fixed offset and
       and learnable offset."""

    num_key_value_heads: int = None
    """Number of key-value heads for group query attention. If None, normal attention is used."""

    init_method: Callable | None = None
    """Method to initialize weights. Note that bias is always set to zero. Should be a function that
    takes a single Tensor and initializes it. If None, will be set to
    paddlefleet.utils.init_method_normal(init_method_std) which is paddle nn init normal with
    mean=0.0 and std=init_method_std."""

    head_dim: int = None
    """Projection weights dimension in multi-head attention. This is set to hidden_size //
    num_attention_heads if not provided."""

    hidden_dropout_prob: float = 0.0
    """Dropout probability for transformer hidden state."""

    attention_dropout: float = 0.0
    """Post attention dropout probability."""

    intermediate_size: int | None = None
    """Transformer Feed-Forward Network hidden size. This is set to 4*hidden_size
    if not provided."""

    gated_linear_unit: bool = False
    """Use a gated linear unit for the first linear layer in the MLP."""

    hidden_act: Callable = F.gelu
    """Activation function to use for the non-linearity in the MLP."""

    use_bias: bool = False
    """Include a bias term in all linear layers (QKV projections and Output projections, after core attention, and two in
    MLP layer)."""

    attention_bias: bool = False
    """Include a bias term in QKV projections."""

    output_layer_init_method: Callable | None = None
    """Method to initialize weights of the output layer of both attention and MLP blocks. If None,
    will be set to paddlefleet.utils.scaled_init_method_normal(init_method_std) which is paddle nn
    init normal with mean=0.0 and std=init_method_std / math.sqrt(2.0 * num_hidden_layers)."""

    rotary_interleaved: bool = False
    """True is rotate pairs of even and odd dimensions (RoFormer style), False is rotate pairs of
    first half and second half (LLaMa style). Default to False."""

    multi_latent_attention: bool = False
    """Whether to use multi-latent attention."""

    heterogeneous_block_specs: bool = False
    """Whether to use heterogeneous block specs (nemotron-nas architecture)."""

    sliding_window: tuple[int, int] = None
    """If not None, then will use sliding window attention. The size of the window is specified by
    the numbers inside the tuple; -1 is special value meaning "infinite window size"."""

    window_attn_skip_freq: int | list[int] = None
    """Frequency of full attention layers among sliding window attention layers. Accepts either:
    - An integer N: Represents a (N-1):1 ratio, one full attention layer after (N-1) SWA layers.
    - A list that defines a custom pattern, e.g.: [1,1,1,1,0,0,0,0], where 1 represents SWA. """

    calculate_per_token_loss: bool = False
    """Whether cross entropy loss is calculated over the actual number of non-padded tokens in the
    global batch, versus the default behavior of assuming all tokens are non-padded."""

    fp32_residual_connection: bool = False
    """If true, move residual connections to fp32."""

    rope_scaling: dict = None
    """Related parameters for rope_scaling, default is None."""

    rope_theta: float = 10000.0
    """The base period of the RoPE embeddings, default is 10000.0."""

    ####################
    # mixed-precision
    ####################
    apply_query_key_layer_scaling: bool = False
    """If true, scale Q * K^T by 1 / layer-number. This improve numeric stability when training with
    fp16."""

    attention_softmax_in_fp32: bool = True
    """If True, run attention masking and softmax in fp32. This should be True if
    apply_query_key_layer_scaling is True."""

    ####################
    # fusion
    ####################
    bias_activation_fusion: bool = False
    """If True, fuses bias addition and the activation function when possible."""

    masked_softmax_fusion: bool = False
    """If True, uses softmax fusion."""

    fuse_rms_norm: bool = False
    """Fused rms norm or not"""

    normalization: str = "RMSNorm"
    """Norm type"""

    use_qk_norm: bool = False
    """Whether to apply `normalization` type of normalization to the query and key embeddings."""

    rms_norm_eps: float = 1e-5
    """Epsilon value for norm."""

    bias_dropout_fusion: bool = False
    """If True, uses bias dropout fusion."""

    apply_rope_fusion: bool = False
    """If True, use fused RoPE kernel."""

    ####################
    # activation recomputation
    ####################
    recompute_granularity: str = None
    """Determines which type of activation recompute to use.  Fleet-core supports 'selective'
    activation checkpointing where the sublayers set in --recompute-modules is checkpointed.
    The default is "core_attn" which is the memory intensive part of attention.
    These memory intensive activations are also less compute intensive which makes activation
    checkpointing more efficient for LLMs (20B+).  See Reducing Activation Recomputation in Large
    Transformer Models (https://arxiv.org/abs/2205.05198) for more details.  'full' will checkpoint
    the entire transformer layer.  If None, no recompute is performed and all activations are saved.
    If set, must be 'selective' or 'full'. 'selective' always uses all layers.
    """

    recompute_method: str = None
    """Determines which transformer layers will be recomputed. uniform will uniformly divide the
    total number of transformer layers in a transformer block and recompute the input activation of
    each divided chunk at the specified granularity.  block will recompute the input activations for
    only a set number of transformer layers per pipeline stage.  The rest of the layers in the
    pipeline stage will not have any activations recomputed.  If None, and recompute is enabled, all
    layers will do recomputation. If set, must be 'uniform' or 'block'."""

    recompute_num_layers: int = None
    """When recompute_method is uniform, recompute_num_layers is the number of transformer layers in
    each uniformly divided recompute unit.  When recompute_method is block, recompute_num_layers is
    the number of transformer layers to recompute within each pipeline stage.  Must be None for
    'selective' activation checkpointing."""

    recompute_modules: list[str] | dict = None
    """The submodules to recompute.
    list: contains all submodule need recompute
    dict: keys contains all submodule need recompute, value means submodule in which layers need recompute
    """

    ####################
    # MoE related
    ####################
    n_routed_experts: int | None = None
    """Number of routed experts to use for MoE layer. When set, it replaces MLP with MoE layer. Set to None
    for no MoE."""

    n_shared_experts: int | None = None
    """Number of shared experts to use for MoE layer. When set, it replaces MLP with MoE layer. Set to None
    for no MoE."""

    num_experts_per_tok: int = 2
    """Number of experts to route to for each token."""

    scoring_func: str = "softmax"
    """Score function for MoE routing. Can be "softmax" or "sigmoid"."""

    moe_intermediate_size: int | None = None
    """MoE Feed-Forward Network hidden size"""

    topk_method: str = "greedy"
    """Options are greedy, group_limited_greedy, no_auxtc"""

    moe_token_dispatcher_type: str = "deepep"
    """The type of token dispatcher to use. The default is 'allgather'.
    Options are 'allgather','alltoall' and 'deepep'."""

    moe_router_load_balancing_type: str = "aux_loss"
    """"Options are aux_loss, seq_aux_loss, global_aux_loss, sinkhorn"""

    moe_layer_freq: int | list[int] | None = None
    """Frequency between MoE layers and Dense layers. Accepts either:
    - An integer N: Represents a 1:N ratio, meaning one expert layer for every N-1 dense layers.
    - A list that defines a custom pattern, e.g.: [1,1,1,0,1,1,1,0,1,1,1,0]"""

    first_k_dense_replace: int | None = None
    """the number of Dense layers.
    - An integer N: Represents the first N layers are dense layers, the remaining ones are moe layers."""

    moe_expert_capacity_factor: float | None = None
    """moe_expert_capacity_factor (float): The capacity factor for each expert, None means no token
    will be dropped. The default is None."""

    moe_pad_expert_input_to_capacity: bool = False
    """moe_pad_expert_input_to_capacity (bool): If True, pads the input for each expert to match
    the expert capacity length, effective only after the moe_expert_capacity_factor is set. The
    default setting is False."""

    moe_token_drop_policy: str = "probs"
    """The policy to drop tokens. Can be either "probs" or "position". If "probs", the tokens with
    the lowest probabilities will be dropped. If "position", tokens at the end of each batch will
    be dropped.
    """

    router_aux_loss_coef: float = 1e-2
    """Scaling coefficient for the aux loss. A starting value of 1e-2 is recommended."""

    norm_topk_prob: bool = True
    """Whether to normalize the topk probabilities."""

    n_group: int = 1
    """Number of groups for routed experts."""

    topk_group: int = 1
    """Number of selected groups per token for expert selection."""

    routed_scaling_factor: float = 1.0
    """Scaling factor for routing score in top-k selection, only works when moe_router_pre_softmax
    enabled. Defaults to None, which means no scaling."""

    moe_dequant_input: bool = False
    """Whether to dequantize input."""

    moe_expert_fusion: bool = True
    """Whether to fuse experts."""

    moe_subbatch_token_num_before_dispatch: int | None = None
    """Whether to enable subbatch before dispatch, the value means the number of tokens in one subbatch."""

    moe_subbatch_token_num_after_dispatch: int | None = None
    """Whether to enable subbatch after dispatch, the value means the number of tokens in one subbatch."""

    moe_grouped_gemm: bool = False
    """Whether to use grouped gemm."""

    router_z_loss_coef: float = None
    """Scaling coefficient for z-loss. Default is None."""

    moe_router_force_load_balancing: bool = False
    """Force load balancing with random logits for MoE router."""

    moe_router_fusion: bool = False
    """Whether to fuse MoE router."""

    moe_shared_expert_overlap: bool = False
    """Enable overlapping between shared expert computations and a2a combinet"""

    moe_deep_gemm: bool = True
    """Whether to use deep gemm. Only work when moe_grouped_gemm enabled."""

    ##################
    # Context Parallel
    ##################
    cp_comm_type: str | list[str] | None = None
    """Inter-gpu communication type for context parallelism. Not support now.
    str: all layers share same communication type.
    List[str]: each layer has its separate communication type.
    """

    ####################
    # fp8
    ####################
    fp8: str | None = None
    """If set, enables the use of FP8 precision through Transformer Engine. There are 2 predefined
    choices (1) 'e4m3' uniformly uses e4m3 for all FP8 tensors, (2) 'hybrid' uses e4m3 for all FP8
    activation and weight tensors and e5m2 for all FP8 output activation gradient tensors."""

    fp8_recipe: str = "blockwise"
    """If set, enables the use of FP8 precision. There are 2 predefined
    choices 1) 'mxfp8' for Blackwell architecture only, 2) 'blockwise' for blockwise scaling recipe"""

    fp8_wgrad: bool = True
    """Whether to use fp8 wgrad."""

    ####################
    # initialization
    ####################
    init_method: callable = None
    """Method to initialize weights. Note that bias is always set to zero. Should be a function that
    takes a single Tensor and initializes it. If None, will be set to
    paddlefleet.utils.init_method_normal(init_method_std) which is paddle nn init normal with
    mean=0.0 and std=init_method_std."""

    embedding_init_method: Callable | None = None
    """
    Method to initialize weights of the embedding layer. If None, will be set as described
    in init_method above.
    """

    embedding_init_method_std: float | None = None
    """
    Standard deviation of the zero mean normal for the default initialization method for the
    embedding layer. If None, will be set to init_method_std.
    """

    output_layer_init_method: callable = None
    """Method to initialize weights of the output layer of both attention and MLP blocks. If None,
    will be set to paddlefleet.utils.scaled_init_method_normal(init_method_std) which is paddle nn
    init normal with mean=0.0 and std=init_method_std / math.sqrt(2.0 * num_hidden_layers)."""

    init_method_std: float = 0.02
    """Standard deviation of the zero mean normal for the default initialization method, not used if
    init_method and output_layer_init_method are provided."""

    embedding_init_method: callable = None
    """
    Method to initialize weights of the embedding layer. If None, will be set as described
    in init_method above.
    """

    embedding_init_method_std: float = None
    """
    Standard deviation of the zero mean normal for the default initialization method for the
    embedding layer. If None, will be set to init_method_std.
    """

    init_model_with_meta_device: bool = False
    """
    If True, initializes the model with the meta device. This is helpful for
    training of very large models. This feature is only works when custom fsdp is turned on.
    """

    use_cpu_initialization: bool = False

    is_hybrid_model: bool = False
    """ Indicates whether this is a hybrid model. """

    ####################
    # miscellaneous
    ####################
    clone_scatter_output_in_embedding: bool = True
    """When set to True, clone the output of scatter_to_sequence_parallel_region in embedding layer
    to facilitate garbage collection of input."""

    @classmethod
    def from_config(cls, config_dict):
        instance = object.__new__(cls)
        instance.register_attributes(config_dict)
        instance.__post_init__()
        return instance

    def register_attributes(self, config):
        transform_rules = None
        if hasattr(self, "transform_rules"):
            transform_rules = self.transform_rules

        for key, value in config.__dict__.items():
            if transform_rules and key in transform_rules:
                self._process_attribute(transform_rules[key], value)
            else:
                self._process_attribute(key, value)

    def _process_attribute(self, key, value):
        if not isinstance(key, str) or not key.isidentifier():
            print(f"invalid key name: {key}")
            return

        if key == "hidden_act":
            if isinstance(value, str):
                func = getattr(F, value)
                setattr(self, key, func)
            elif callable(value):
                setattr(self, key, value)
            else:
                raise TypeError(
                    f"hidden_act must be str or callable, but get {type(value)}"
                )
        elif key == "dtype":
            self.params_dtype = value
        else:
            setattr(self, key, value)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def __post_init__(self):
        """Python dataclass method that is used to modify attributes after initialization.
        See https://docs.python.org/3/library/dataclasses.html#post-init-processing for more
        details.
        """
        super().__post_init__()
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size

        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.num_key_value_heads % self.tensor_model_parallel_size != 0:
            raise ValueError(
                f"num_key_value_heads ({self.num_key_value_heads}) must be a multiple of "
                f"tensor_model_parallel_size ({self.tensor_model_parallel_size})."
            )

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        # Set the embedding init method
        if self.embedding_init_method_std is None:
            # By default, use the same init std as you use for every other non-output layer.
            self.embedding_init_method_std = self.init_method_std

        if self.embedding_init_method is None:
            if self.init_method is None or (
                self.embedding_init_method_std != self.init_method_std
            ):
                # In this case, we set both the init method and the embedding init method to
                #  whatever std value requested (or defaulted) for the embedding_init_layer
                self.embedding_init_method = init_method_normal(
                    self.embedding_init_method_std
                )
            else:
                # Replicate the current behavior where if you are not changing the std of the
                #  embedding init differently and the init method is set, we fallback to the
                #  init method for this layer. Since we are here after an OR we know that
                #  init_method is not None
                self.embedding_init_method = self.init_method

        if self.init_method is None:
            self.init_method = init_method_normal(self.init_method_std)

        if self.first_k_dense_replace and self.moe_layer_freq:
            raise ValueError(
                "Cannot specify both first_k_dense_replace and moe_layer_freq."
            )
        if self.first_k_dense_replace is None and self.moe_layer_freq is None:
            self.moe_layer_freq = 1
        if self.first_k_dense_replace:
            self.moe_layer_freq = [0] * self.first_k_dense_replace + [1] * (
                self.num_hidden_layers - self.first_k_dense_replace
            )
        if self.recompute_granularity == "":
            self.recompute_granularity = None

        # recompute config check
        if self.recompute_granularity is not None:
            assert self.recompute_granularity in ["full", "selective"], (
                "recompute_granularity must be one of full and selective"
            )
            if self.recompute_granularity == "full":
                assert self.recompute_method in [
                    "block",
                    "first_n",
                    "uniform",
                ], (
                    "when recompute_granularity=full, recompute_method must be one of block, first_n and uniform"
                )
                assert self.recompute_num_layers is not None, (
                    "when recompute_granularity=full, recompute_num_layers mustn't be None"
                )
            elif self.recompute_granularity == "selective":
                assert self.recompute_method in ["block", "first_n", None], (
                    "when recompute_granularity=selective, recompute_method must be one of block and first_n"
                )
                assert self.recompute_modules is not None
            else:
                raise ValueError(
                    "recompute_granularity must be one of full and selective"
                )

        if self.output_layer_init_method is None:
            self.output_layer_init_method = scaled_init_method_normal(
                self.init_method_std,
                self.num_hidden_layers,
                multiplier=2.0 if not self.is_hybrid_model else 1.0,
            )

        # Set the embedding init method
        if self.embedding_init_method_std is None:
            # By default, use the same init std as you use for every other non-output layer.
            self.embedding_init_method_std = self.init_method_std

        if self.embedding_init_method is None:
            if self.init_method is None or (
                self.embedding_init_method_std != self.init_method_std
            ):
                # In this case, we set both the init method and the embedding init method to
                #  whatever std value requested (or defaulted) for the embedding_init_layer
                self.embedding_init_method = init_method_normal(
                    self.embedding_init_method_std
                )
            else:
                # Replicate the current behavior where if you are not changing the std of the
                #  embedding init differently and the init method is set, we fallback to the
                #  init method for this layer. Since we are here after an OR we know that
                #  init_method is not None
                self.embedding_init_method = self.init_method
