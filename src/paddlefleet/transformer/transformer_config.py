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

    num_layers: int = 0
    """Number of transformer layers in a transformer block."""

    mtp_num_layers: int = None
    """Number of Multi-Token Prediction (MTP) Layers."""

    mtp_loss_scaling_factor: float = None
    """Weighting factor of Multi-Token Prediction (MTP) loss."""

    num_layers_in_first_pipeline_stage: int = None
    """Number of transformer layers on first pipeline stage.
    None implies equal layer division across PP ranks."""

    num_layers_in_last_pipeline_stage: int = None
    """Number of transformer layers on last pipeline stage.
    None implies equal layer division across PP ranks."""

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

    num_attention_heads: int = 0
    """Number of transformer attention heads."""

    softmax_scale: float = None
    """Softmax scale for attention scaling."""

    softmax_type: Literal["vanilla", "off-by-one", "learnable"] = "vanilla"
    """Applies modified softmax from https://www.evanmiller.org/attention-is-off-by-one.html.
       Supports both TE FusedAttention and local unfused attention. Supports both a fixed offset and
       and learnable offset."""

    num_query_groups: int = None
    """Number of query groups for group query attention. If None, normal attention is used."""

    init_method: Callable | None = None
    """Method to initialize weights. Note that bias is always set to zero. Should be a function that
    takes a single Tensor and initializes it. If None, will be set to
    paddlefleet.utils.init_method_normal(init_method_std) which is paddle nn init normal with
    mean=0.0 and std=init_method_std."""

    kv_channels: int = None
    """Projection weights dimension in multi-head attention. This is set to hidden_size //
    num_attention_heads if not provided."""

    hidden_dropout: float = 0.1
    """Dropout probability for transformer hidden state."""

    attention_dropout: float = 0.1
    """Post attention dropout probability."""

    ffn_hidden_size: int | None = None
    """Transformer Feed-Forward Network hidden size. This is set to 4*hidden_size
    if not provided."""

    hidden_dropout: float = 0.1
    """Dropout probability for transformer hidden state."""

    gated_linear_unit: bool = False
    """Use a gated linear unit for the first linear layer in the MLP."""

    act_fn: Callable = F.gelu
    """Activation function to use for the non-linearity in the MLP."""

    use_bias: bool = True
    """Include a bias term in all linear layers (QKV projections, after core attention, and two in
    MLP layer)."""

    output_layer_init_method: Callable | None = None
    """Method to initialize weights of the output layer of both attention and MLP blocks. If None,
    will be set to paddlefleet.utils.scaled_init_method_normal(init_method_std) which is paddle nn
    init normal with mean=0.0 and std=init_method_std / math.sqrt(2.0 * num_layers)."""

    rotary_interleaved: bool = False
    """True is rotate pairs of even and odd dimensions (RoFormer style), False is rotate pairs of
    first half and second half (LLaMa style). Default to False."""

    multi_latent_attention: bool = False
    """Whether to use multi-latent attention."""

    heterogeneous_block_specs: bool = False
    """Whether to use heterogeneous block specs (nemotron-nas architecture)."""

    window_size: tuple[int, int] = None
    """If not None, then will use sliding window attention. The size of the window is specified by
    the numbers inside the tuple; -1 is special value meaning "infinite window size"."""

    window_attn_skip_freq: int | list[int] = None
    """Frequency of full attention layers among sliding window attention layers. Accepts either:
    - An integer N: Represents a (N-1):1 ratio, one full attention layer after (N-1) SWA layers.
    - A list that defines a custom pattern, e.g.: [1,1,1,1,0,0,0,0], where 1 represents SWA. """

    calculate_per_token_loss: bool = False
    """Whether cross entropy loss is calculated over the actual number of non-padded tokens in the
    global batch, versus the default behavior of assuming all tokens are non-padded."""

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

    layernorm_epsilon: float = 1e-5
    """Epsilon value for norm."""

    bias_dropout_fusion: bool = False
    """If True, uses bias dropout fusion."""

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

    ####################
    # MoE related
    ####################
    moe_shared_expert_intermediate_size: int | None = None
    """Shared expert total ffn hidden size.
    It should be equal to 'num_shared_experts * ffn_size_of_each_shared_expert' if
    there are multiple shared experts.
    None means no shared expert.
    By default, the shared experts execute before the router. However, when
    moe_shared_expert_overlap or overlap_moe_expert_parallel_comm is set,
    the shared experts execute after the router, before the routed experts.
    This makes the gradients from the router and the shared experts added in
    different orders to the hidden_states, causing minor numerical differences
    in the hidden_states gradient."""

    num_experts_per_tok: int = 2
    """Number of experts to route to for each token."""

    moe_num_experts: int | None = None
    """Number of experts to use for MoE layer. When set, it replaces MLP with MoE layer. Set to None
    for no MoE."""

    moe_intermediate_size: int | None = None
    """MoE Feed-Forward Network hidden size"""

    topk_method: str = "allgather"
    """The type of token dispatcher to use. The default is 'allgather'.
    Options are 'allgather','alltoall' and 'flex'."""

    moe_layer_freq: int | list[int] = 1
    """Frequency between MoE layers and Dense layers. Accepts either:
    - An integer N: Represents a 1:N ratio, meaning one expert layer for every N-1 dense layers.
    - A list that defines a custom pattern, e.g.: [1,1,1,0,1,1,1,0,1,1,1,0]"""

    ####################
    # initialization
    ####################
    init_method: callable = None
    """Method to initialize weights. Note that bias is always set to zero. Should be a function that
    takes a single Tensor and initializes it. If None, will be set to
    paddlefleet.utils.init_method_normal(init_method_std) which is paddle nn init normal with
    mean=0.0 and std=init_method_std."""

    output_layer_init_method: callable = None
    """Method to initialize weights of the output layer of both attention and MLP blocks. If None,
    will be set to paddlefleet.utils.scaled_init_method_normal(init_method_std) which is paddle nn
    init normal with mean=0.0 and std=init_method_std / math.sqrt(2.0 * num_layers)."""

    init_method_std: float = 0.02
    """Standard deviation of the zero mean normal for the default initialization method, not used if
    init_method and output_layer_init_method are provided."""

    init_model_with_meta_device: bool = False
    """
    If True, initializes the model with the meta device. This is helpful for
    training of very large models. This feature is only works when custom fsdp is turned on.
    """

    use_cpu_initialization: bool = False

    is_hybrid_model: bool = False
    """ Indicates whether this is a hybrid model. """

    def __post_init__(self):
        """Python dataclass method that is used to modify attributes after initialization.
        See https://docs.python.org/3/library/dataclasses.html#post-init-processing for more
        details.
        """
        super().__post_init__()
        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = 4 * self.hidden_size

        if self.kv_channels is None:
            self.kv_channels = self.hidden_size // self.num_attention_heads

        if self.num_query_groups is None:
            self.num_query_groups = self.num_attention_heads

        if self.num_query_groups % self.tensor_model_parallel_size != 0:
            raise ValueError(
                f"num_query_groups ({self.num_query_groups}) must be a multiple of "
                f"tensor_model_parallel_size ({self.tensor_model_parallel_size})."
            )

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        if self.init_method is None:
            self.init_method = init_method_normal(self.init_method_std)

        if self.output_layer_init_method is None:
            self.output_layer_init_method = scaled_init_method_normal(
                self.init_method_std,
                self.num_layers,
                multiplier=2.0 if not self.is_hybrid_model else 1.0,
            )
