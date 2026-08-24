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

import functools
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import paddle.nn.functional as F

from ..model_parallel_config import ModelParallelConfig
from ..utils import (
    get_magic_init_method,
    init_method_normal,
    scaled_init_method_normal,
    truncated_init_method_normal,
)
from .activations import situ

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig(ModelParallelConfig):
    """Configuration object for transformers."""

    ####################
    # model architecture
    ####################

    num_hidden_layers: int = 1
    """Number of transformer layers in a transformer block."""

    pad_token_id: int = 0
    """Token ID used for padding."""

    num_nextn_predict_layers: int = 0
    """Number of Multi-Token Prediction (MTP) Layers."""

    train_mtp_only: bool = False
    """Whether to train MTP only."""

    mtp_distillation_loss: bool = False
    """Whether to use distillation MTP loss."""

    mtp_num_layers: int = 0
    """MTP Layer number."""

    mtp_loss_scaling_factor: float = 0.1
    """Weighting factor of Multi-Token Prediction (MTP) loss."""

    add_mtp_loss: bool = True
    """Add mtp loss to final loss to enable mtp backward and weight update."""

    mtp_load_weight_only: bool = False
    """When True, use WeightOnlyMTPLayer (holds weights but skips MTP computation and embedding processing)."""

    use_dense_mtp: bool = False
    """When True, MTP layers use dense MLP instead of MoE in their internal transformer block."""

    mtp_shared_last_layer: bool = False
    """When True, MTP layers share the last backbone TransformerLayer parameters."""

    separate_mtp_headloss: bool = False
    """Separate MTP LMHead & Loss calculate for pipeline balance."""

    enable_mtp_magic_send: bool = False
    """When True, use magic send mechanism for MTP: broadcast input_ids to last PP stage
    and re-embed there, instead of pre-computing shifted embeddings at first stage
    and concatenating them through the pipeline."""

    separate_mtp_input: bool = False
    """When True, the shifted MTP embeddings computed by GPTEmbedding are handed to the
    MTP layer through a dedicated ``mtp_decoder_inputs`` entry in ``dict_args`` instead
    of being concatenated into ``hidden_states``. This removes the per-layer
    split/concat of the MTP chunks while leaving GPTEmbedding's shifted-embedding
    computation (including its CP/SP scatter) untouched, so the MTP layer must not
    re-scatter them. Intended for pipeline_model_parallel_size == 1, where there is no
    P2P send for the embeddings to piggyback on; ``enable_mtp_magic_send`` covers the
    PP > 1 case."""

    experimental_dataflow: bool = False
    """When True, use new experimental dataflow where mtp_startend_row_indices_all is passed as a
    separate input instead of being appended to attn_mask_startend_row_indices.
    The new dataflow requires: input_ids, labels, startend_row_indices (last dim=1, main seq only),
    mtp_startend_row_indices_all ([B, num_nextn, S, 1]), position_ids."""

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

    _attn_implementation: str = "default"
    """Attention implementation to use."""

    flashmask_use_varlen: bool = False
    """If True, convert flashmask to varlen in attention."""

    intermediate_size: int | None = None
    """Transformer Feed-Forward Network hidden size. This is set to 4*hidden_size
    if not provided."""

    gated_linear_unit: bool = False
    """Use a gated linear unit for the first linear layer in the MLP."""

    hidden_act: Callable = F.gelu
    """Activation function to use for the non-linearity in the MLP."""

    activation_situ_beta: float = 1.0
    """Scale of the tanh term in the SiTU gate activation."""

    activation_situ_linear_beta: float | None = None
    """Optional tanh scale applied to the linear branch of SiTU-GLU."""

    situ_glu_fusion: bool = True
    """Use fused Triton SiTU-GLU in FusionMoe BF16 routed experts."""

    use_bias: bool = False
    """Include a bias term in all linear layers (QKV projections and Output projections, after core attention, and two in
    MLP layer)."""

    moe_routed_expert_use_bias: bool | None = None
    """Override whether routed MoE expert MLP layers include bias terms. If None, use use_bias."""

    attention_bias: bool = False
    """Include a bias term in QKV projections."""

    output_layer_init_method: Callable | None = None
    """Method to initialize weights of the output layer of both attention and MLP blocks. If None,
    will be set to paddlefleet.utils.scaled_init_method_normal(init_method_std) which is paddle nn
    init normal with mean=0.0 and std=init_method_std / math.sqrt(2.0 * num_hidden_layers)."""

    rotary_interleaved: bool = False
    """True is rotate pairs of even and odd dimensions (RoFormer style), False is rotate pairs of
    first half and second half (LLaMa style). Default to False."""

    use_vha_attention: bool = False
    """If True, enables VHA premix/postmix extensions in standard self-attention."""

    vha_shared_kv: bool = False
    """If True, enables Shared KV to reduce KVCache"""

    vha_postmix_rank: int | None = None
    """Rank of the VHA postmix low-rank head mixing matrices."""

    vha_postmix_grouped: bool = False
    """Postmix head-mixing topology (DSv4 hybrid only). False (default): ungrouped
    full cross-head mixing over all num_attention_heads (the earlier VHA design).
    True: within-group block-diagonal mixing that only recombines heads inside each
    o_group (mixing stays within a group)."""

    fuse_inv_rope_into_vha_postmix: bool = False
    """Fuse the HCA inverse RoPE into the ungrouped VHA postmix GEMM (DSv4 hybrid).

    The unfused path materialises ``inv_rope(O)`` as a full-width tensor and feeds
    it to the postmix ``[nh,nh]`` GEMM, which costs one extra read+write of the
    whole attention output plus a second live copy of it. Because RoPE only
    touches the trailing ``qk_pos_emb_head_dim`` channels while the GEMM
    contracts the head axis, the same result can be assembled from a full-width
    GEMM on the *unrotated* output plus a narrow GEMM on the rotated pe channels,
    which never needs the wide intermediate.

    Bitwise identical to the unfused path -- forward, activation gradient and the
    postmix U/V gradients -- and asserted as such in
    ``tests/single_card_tests/test_inv_rope_vha_postmix_fusion.py``. Requires
    ``use_vha_attention`` and ``apply_rope_fusion``, and is skipped for
    ``vha_postmix_grouped``, ``high_precision_rope`` and when the postmix has its
    own selective recompute wrapper."""

    use_vha_premix: bool = False
    """If True (and use_vha_attention is also True), replaces the DSv4 hybrid Q up-projection
    (linear_q_up_proj) with a structured VHA premix: the compressed Q is reshaped into
    vha_premix_groups groups of dim d_q = q_lora_rank // vha_premix_groups, and each group is
    expanded into k = num_attention_heads // vha_premix_groups heads via a per-group weight.
    Requires q_lora_rank % vha_premix_groups == 0 and num_attention_heads % vha_premix_groups
    == 0. DSv4 hybrid attention only; postmix is unaffected (still keyed on use_vha_attention)."""

    vha_premix_groups: int | None = None
    """Number of groups g_q the compressed Q is split into for the VHA premix. Per-group latent
    dim is q_lora_rank // vha_premix_groups; per-group head expansion is num_attention_heads //
    vha_premix_groups. Only used when use_vha_premix is True."""

    vha_q_lora_rank: int | None = None
    """Rank of the VHA Q low-rank projection. When set, Q projects to this rank per head before premix expansion."""

    swa_vha_q_lora_rank: int | None = None
    """VHA Q low-rank projection rank for SWA layers. Defaults to swa_head_dim in __post_init__."""

    swa_vha_postmix_rank: int | None = None
    """VHA postmix rank for SWA layers. Defaults to swa_num_attention_heads // 4."""

    attention_value_scale: float | None = None
    """Scale factor applied to the value tensor before attention computation. If None, no scaling
    is applied. Used in architectures like MiMo that scale V for training stability."""

    add_full_attention_sink_bias: bool = False
    """Whether to add a learnable attention sink bias for full (non-SWA) attention layers.
    When True, softmax_type is promoted to 'learnable' for full attention layers."""

    add_swa_attention_sink_bias: bool = True
    """Whether to add a learnable attention sink bias for sliding window attention (SWA) layers.
    When True, softmax_type is promoted to 'learnable' for SWA layers."""

    swa_head_dim: int | None = None
    """Dimension of query/key heads for sliding window attention layers. Defaults to head_dim."""

    swa_v_head_dim: int | None = None
    """Dimension of value heads for sliding window attention layers. Defaults to v_head_dim."""

    swa_num_attention_heads: int | None = None
    """Number of attention heads for sliding window attention layers. Defaults to num_attention_heads."""

    swa_num_key_value_heads: int | None = None
    """Number of key/value heads (GQA groups) for sliding window attention layers. Defaults to num_key_value_heads."""

    swa_rope_theta: float | None = None
    """The base period of the RoPE embeddings for sliding window attention layers. Defaults to rope_theta."""

    swa_qk_nope_head_dim: int = None
    """Dimension of the nope part of QK heads for SWA layers. If None, falls back to qk_nope_head_dim."""

    swa_qk_rope_head_dim: int = None
    """Dimension of the rope part of QK heads for SWA layers. If None, falls back to qk_rope_head_dim."""

    head_wise_swa_ratio: float = 0.0
    """Ratio of KV heads that use sliding window attention within an SWA layer.
    0.0 means all heads use SWA; values between 0 and 1 create a mix where
    the first (1 - ratio) * num_heads are full attention and the rest are SWA."""

    multi_latent_attention: bool = False
    """Whether to use multi-latent attention."""

    heterogeneous_block_specs: bool = False
    """Whether to use heterogeneous block specs (nemotron-nas architecture)."""

    sliding_window: int | tuple[int, int] = None
    """If not None, then will use sliding window attention. The size of the window is specified by
    the numbers inside the tuple; -1 is special value meaning "infinite window size".
    Accepts a scalar int (HF-compatible causal one-sided semantics) or a (left, right) tuple
    (Fleet native two-sided semantics); `-1` means infinite window size."""

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

    apply_residual_connection_post_layernorm: bool = False
    """If True, uses the original BERT residue connection ordering."""

    activation_func_clamp_value: float = None
    """Clamp the output of the linear_fc1 in the activation function. Only used when activation_func
    is quick_gelu."""

    glu_linear_offset: float = 0.0
    """Offset term in the GLU activation function: activation_func(x[0]) * (x[1] + offset). Only
    used when gated_linear_unit is True"""

    multimodal_embedding: bool = False
    """Whether to use multimodal embedding."""

    multimax_modules: list[str] | None = None
    """Submodules to apply learnable SegLU-style modulation to before softmax.

    Mirrors the Megatron ``recompute_modules`` style: a list of submodule
    names. ``None`` (default) disables the feature globally. Currently
    supported list entries:

    - ``"lm_head"``: apply SegLU(x, ranges, ts) on the LM-head logits before
      the language-modeling softmax/cross-entropy. Adds two [4]-shape
      learnable parameters (multimax_ranges, multimax_ts) to the LM head.
      These are excluded from weight decay via the "multimax" substring
      filter in the trainer's no-decay rule.
    - ``"attention"``: apply on attention scores before softmax. Reserved;
      not implemented yet (emits a warning if listed).

    YAML/JSON behaviour:
    - unset key, ``multimax_modules: null``, or empty list ``multimax_modules: []``
      all map to Python ``None`` (feature disabled).
    - ``multimax_modules: [lm_head]`` enables the LM-head branch.
    """

    gated_attention: bool = False
    """If True, enables gated attention where a learnable sigmoid gate is applied to the
    attention output before the output projection. The gate is produced alongside the query
    from the fused QKV projection (doubling the query projection size). This allows the model
    to dynamically control the information flow from attention. See Qwen3.5 for reference."""

    gated_attn_use_q_lora: bool = False
    """If True, the gated attention gate uses the q_a_proj output (q_compressed, post
    q_a_layernorm, dim = q_lora_rank) as the gate input instead of hidden_states. This is a
    low-rank gate input for MLA networks that also reduces the gate projection parameter count.
    Requires q_lora_rank is not None. Only applies when gated_attention is True."""

    ####################
    # block attention residuals
    ####################
    block_attention_residuals: bool = False
    """Whether to use block attention residuals. When True,
    replaces standard fixed-weight residual connections with
    learned softmax attention over block-level representations."""

    attn_res_block_size: int = 1
    """Number of consecutive layers per block for
    block attention residuals. Controls how many layers
    accumulate standard residuals before applying the learned
    attention-weighted combination across blocks."""

    attn_res_fusion: bool = True
    """If True, use the FLA fused Triton kernel for Block Attention Residuals.
    Fuses RMSNorm + projection + softmax + weighted sum into a single
    kernel launch. Requires paddlefleet_ops with fla.ops.attnres.
    Falls back to PyLayer when unavailable or deterministic_mode=True."""

    ####################
    # mixed-precision
    ####################
    apply_query_key_layer_scaling: bool = False
    """If true, scale Q * K^T by 1 / layer-number. This improve numeric stability when training with
    fp16."""

    attention_softmax_in_fp32: bool = True
    """If True, run attention masking and softmax in fp32. This should be True if
    apply_query_key_layer_scaling is True."""

    high_precision_rope: bool = False
    swa_high_precision_norm: bool = False
    ####################
    # fusion
    ####################
    bias_activation_fusion: bool = False
    """If True, fuses bias addition and the activation function when possible."""

    masked_softmax_fusion: bool = False
    """If True, uses softmax fusion."""

    normalization: str = "RMSNorm"
    """Norm type"""

    use_qk_norm: bool = False
    """Whether to apply `normalization` type of normalization to the query and key embeddings."""

    qk_norm_eps: float | None = None
    """Epsilon for query/key normalization. If None, falls back to rms_norm_eps."""

    qk_norm_fusion: bool = False
    """If True, use Triton fused RMSNorm kernel for QK norm."""

    qk_norm_type: str = "per_head"
    """Type of qk normalization:
    - "per_head": normalize each attention head independently (default for most models)
    - "per_layer": normalize across all heads jointly (full-dimension, used by MiniMax)
    """

    rms_norm_eps: float = 1e-5
    """Epsilon value for norm."""

    layernorm_zero_centered_gamma: bool = False
    """If set to True, the LayerNorm is adjusted to center the gamma values around 0. This improves
    numerical stability."""

    bias_dropout_fusion: bool = False
    """If True, uses bias dropout fusion."""

    apply_rope_fusion: bool = False
    """If True, use fused RoPE kernel."""

    mqa_latent_rope_fusion: bool = False
    """If True, use the Triton rotate_half kernel for absorbed MQA's RoPE.

    Independent of ``apply_rope_fusion``, which selects
    ``fused_apply_mla_rope_for_q`` / ``_for_kv``. Those two de-interleave
    (``mla_output_remove_interleaving=False``) and ``_for_kv`` needs the
    per-head K/V that absorbed MQA never materialises, so they cannot serve the
    ``non_absorbed_mqa`` layers. This flag covers exactly those layers: it routes
    the *eager* branch's q RoPE through ``fused_apply_rope_half`` and its k RoPE
    plus the key concat through ``fused_rope_cat_key``, both bit-exact with that
    branch. Neither writes in place, so they stay correct when
    ``recompute_qkv_up_porj_and_rope`` replays the closure that owns
    ``k_pos_emb``.

    Silently falls back to eager on any layer that is not absorbed MQA
    (``mqa_latent``) -- an unabsorbed MLA layer has its own fused path via
    ``apply_rope_fusion``. Every other incompatibility is asserted rather than
    downgraded, matching how ``apply_rope_fusion`` handles its own
    (multi_latent_attention.py:1723): ``multi_latent_attention``,
    ``rotary_interleaved``, ``high_precision_rope``, ``sequence_parallel`` and
    THD ``cu_seqlens`` each select a rotation the kernel does not implement, and
    so are bf16 activations and fp32 freqs when mscale != 1, which the kernel
    asserts itself.
    """

    sigmoid_gate_fusion: bool = False
    """If True, use Triton fused sigmoid gate kernel."""

    dsv4_q_rms_norm_fusion: bool = False
    """If True, use the Triton weight-free fused kernel for query RMS norm in the
    DSV4 hybrid attention path. Only takes effect when swa_high_precision_norm=False."""

    dsa_sink_grad_fusion: bool = False
    """Whether to use the fused Triton attention-sink gradient epilogue.

    Scope: ``MQALatentAttention`` (``non_absorbed_mqa``) only. It is read in
    ``MQALatentAttention.__init__`` and forwarded from ``_sparse_attn``. HySparse's
    DSA gather (``paddlefleet.cudnn_ops.block_sparse_mqa_attention_dsa``) shares
    the same ``mqa_sparse_attn`` entry and can also carry a learnable sink, but it
    deliberately does not forward this flag: it stays on the eager epilogue no
    matter how this field is set.

    The sparse-attention backward computes ``d_sink`` analytically because the
    SM100 cuDNN DSA backward returns an all-zero ``d_sink``. The eager
    implementation materialises three ``[b, s, h, d_v]`` fp32 temporaries to
    produce ``[h]`` numbers; the kernel reads ``out``/``do`` once in their native
    dtype. Measured at b=1/s=8192/h=64/d_v=512: 2.664 ms -> 0.423 ms, transient
    3.0 GiB -> ~0.

    No effect on layers without a learnable sink (``add_full_attention_sink_bias``
    off), which have no sink gradient to compute.

    Not bitwise identical to the eager path: 1.9e-7 relative to the gradient
    vector's own scale (~1.6 fp32 ulp), entirely from the fp32 summation order of
    ``Delta = sum_dv(out * do)``. Both paths are deterministic run-to-run.
    """

    dsv4_yarn_rope_fusion: bool = False
    """If True, use the Triton fused kernel to build the YaRN RoPE frequency table
    in the DSV4 hybrid attention path. Only the DSV4 hybrid attention path reads this
    field; the standard MLA / DSA YaRN paths ignore it."""

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

    decoderlayer_act_offload_settings: dict = None
    """Settings for decoder layer activation offloading to CPU.

    A dict with two keys:
      - "type": str, the offload strategy type. Supported values:
          - "mod": offload layers where (layer_number % value[0] == value[1]).
                   "value" should be a list/tuple of two ints [divisor, remainder].
          - "layer_idxs": offload specific layers by index.
                   "value" should be a list of layer indices to offload.
      - "value": the strategy parameter, format depends on "type".

    Example:
        {"type": "mod", "value": [1, 0]}       # offload all layers (every layer % 1 == 0)
        {"type": "mod", "value": [2, 0]}       # offload even-numbered layers
        {"type": "layer_idxs", "value": [0, 5, 10]}  # offload layers 0, 5, 10
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
    """Score function for MoE routing. Options: "softmax", "sigmoid", "tanh",
    "relu", "gelu", "leaky_relu", "sftplus" (softplus, non-negative unbounded),
    "sqrtsoftplus" (sqrt(softplus), non-negative unbounded)."""

    moe_intermediate_size: int | None = None
    """MoE Feed-Forward Network hidden size"""

    topk_method: str = "greedy"
    """Options are greedy, group_limited_greedy, noaux_tc, quantile_balancing"""

    moe_token_dispatcher_type: str = "alltoall"
    """The type of token dispatcher to use. The default is 'alltoall'.
    Options are 'allgather', 'alltoall', 'deepep', and 'hybridep'."""

    moe_allgather_gate_overlap: bool = True
    """Whether to issue the AllGather before the gate so it overlaps with gate
    compute. Only honoured when ``moe_token_dispatcher_type='allgather'`` and
    ``expert_model_parallel_size > 1``; ignored otherwise."""

    moe_use_fusion_node: bool = True
    """Whether to use fusion node for MoE layer. Default is True"""

    moe_router_load_balancing_type: str = "aux_loss"
    """"Options are aux_loss, seq_aux_loss, global_aux_loss, sinkhorn, none.
    Use 'none' together with router_aux_loss_coef=0 when the router balances
    load on its own (required by topk_method='quantile_balancing')."""

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
    """Scalar multiplier applied to the selected top-k routing weights after expert selection.
    The final scaled weights are used in ``top_gate`` (``[S, K]``), which is passed to the
    dispatch/combine flow for expert output weighting.

    Default is ``1.0`` (no scaling effect). For example, set to ``2.5`` for DeepSeek-V3 to
    compensate for sigmoid scores not summing to 1 after top-k selection.

    When ``routed_scaling_factor_learnable=True``, this value is used as the initialization
    value for the per-expert learnable parameter."""

    routed_scaling_factor_learnable: bool = False
    """Whether to use a learnable per-expert scaling parameter instead of a fixed scalar.

    - ``False`` (default): apply ``routed_scaling_factor`` as a fixed scalar uniformly.
    - ``True``: create a trainable parameter of shape ``[num_experts]``, initialized to
      ``routed_scaling_factor``, and apply it via per-expert lookup after top-k selection."""

    moe_dequant_input: bool = False
    """Whether to dequantize input."""

    moe_expert_fusion: bool = False
    """Whether to fuse experts."""

    moe_subbatch_token_num_before_dispatch: int | None = None
    """Whether to enable subbatch before dispatch, the value means the number of tokens in one subbatch."""

    moe_subbatch_token_num_after_dispatch: int | None = None
    """Whether to enable subbatch after dispatch, the value means the number of tokens in one subbatch."""

    use_auto_subbatch: bool = False
    """When True, dynamically determine subbatch sizes based on VMM free block analysis
    instead of using a fixed moe_subbatch_token_num_after_dispatch value."""

    moe_subbatch_diag: bool = False
    """When True, print auto_subbatch diagnostic info (path, subbatch_rows, zip_unzip_fusion)
    after each forward/backward pass. Useful for debugging memory behavior."""

    auto_subbatch_mode: str | None = None
    """Auto-subbatch splitting strategy. This only selects the strategy when
    use_auto_subbatch=True; it does not enable auto-subbatch by itself.
    - None: use the default "post_permute" strategy.
    - "post_permute": run full moe_permute first, then subbatch in permuted space.
    - "pre_permute": split chunks in dispatched space first, then run
      permute→compute→unpermute independently for each chunk.
    """

    router_z_loss_coef: float = None
    """Scaling coefficient for z-loss. Default is None."""

    moe_router_force_load_balancing: bool = False
    """Force load balancing with random logits for MoE router."""

    qb_n_bins: int = 1000
    """Number of histogram bins for Quantile Balancing. Only used when
    topk_method='quantile_balancing'. Higher values give more precise
    quantile estimation at the cost of slightly more communication."""

    moe_split_feature_routing: bool = False
    """Enable multi-view (split-feature) MoE routing. When True, the router
    scores each expert with the sum of two independent views: the existing
    ``self.weight`` gate plus a new ``self.weight_1`` projection, i.e.
    ``score_func(logits_0) + score_func(logits_1)`` instead of a single gate
    projection. The expert FFN compute path is unchanged. Disabled by default;
    has no effect on hash-routing layers (moe_n_hash_layers), which keep using
    the original single gate."""

    moe_n_hash_layers: int = 0
    """Number of leading transformer layers that use hash-based MoE routing.
    Layers with layer_number < moe_n_hash_layers (0-indexed) use a pre-computed
    tid2eid lookup table for expert selection instead of learned top-k routing.
    Score weights are still computed from the gate logits. 0 disables hash routing."""

    actual_vocab_size: int | None = None
    """Padded actual vocabulary size. Required when moe_n_hash_layers > 0 for the
    tid2eid lookup buffer in hash-based MoE routing."""

    moe_router_fusion: bool = False
    """Whether to fuse MoE router."""

    moe_shared_expert_gate: bool = False
    """Enable gate for shared expert."""

    moe_shared_expert_overlap: bool = False
    """Enable overlapping between shared expert computations and a2a combinet"""

    moe_deep_gemm: bool = True
    """Whether to use DeepGEMM for the bf16 grouped-gemm MoE path. This option only takes effect when
    ``moe_expert_fusion=True`` and fp8 is disabled, it is ignored when fp8 is enabled."""

    moe_ep_barrier: bool = True
    """Whether to use barrier for expert parallelism."""

    moe_latent_size: int | None = None
    """The latent dimension size for latent MoE. Positive values enable latent MoE."""

    latent_moe_use_norm: bool = False
    """Apply RMSNorm to routed latent-MoE output before projecting it back to
    the model hidden size."""

    ##################
    # Context Parallel
    ##################
    cp_comm_type: str | list[str] | None = None
    """Inter-gpu communication type for context parallelism. Not support now.
    str: all layers share same communication type.
    List[str]: each layer has its separate communication type.
    """

    cp_balance_mode: str = "dualchunk_allgather"
    """Context parallel scatter/gather layout mode.
    "dualchunk_allgather": balanced front+rear chunk splitting (default).
    "contiguous_allgather": simple rank-order contiguous slicing.
    "contiguous_a2a".
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

    dw_p2p_overlap: bool = False
    """Whether to overlap p2p communication and matmul kernel in pp parallel on Blackwell."""

    use_ue8m0: bool = False
    """Whether to use UE8M0 packed scaling factors for FP8 on Blackwell GPUs."""

    use_fp8_qat: bool = False
    """Whether to enable FP8 Quantization-Aware Training (QAT)."""

    use_w4a8: bool = False
    """Whether to use w4a8 for mlp gemm."""

    use_w4a8_fused_quant: bool = False
    """Whether to use fused CUDA operators for W4A8 online quantization."""

    full_fp8_computation: bool = False
    """Master switch for FP8 on Linear / ColumnParallelLinear / RowParallelLinear
    and DSv4HybridSelfAttention / CSAIndexer. When ``False`` these layers stay
    in bf16 regardless of ``fp8`` / ``fp8_wgrad`` / ``use_ue8m0``. Pre-existing
    FP8 paths (e.g. ``moe_layer`` / ``moe_expert``) ignore this flag."""

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
    # Hyper-Connection (mHC) Configuration
    ####################
    enable_hyper_connections: bool = False
    """Enable mHC (Manifold-Constrained Hyper-Connections) residual connections."""

    num_residual_streams: int = 4
    """Number of residual streams (n in mHC paper)."""

    mhc_sinkhorn_iterations: int = 20
    """Number of Sinkhorn-Knopp iterations for doubly stochastic projection."""

    mhc_init_gating_factor: float = 0.01
    """Initial value of Gating Factor (alpha in paper)."""

    use_fused_mhc: bool = False
    """Use fused triton kernels for mHC operations (sinkhorn, h_aggregate, h_post_bda, proj_rms).
    Requires cuTile to be available."""

    high_precision_mhc: bool = True
    """Use high precision (float32) for mHC forward and backward computation."""

    ####################
    # miscellaneous
    ####################
    clone_scatter_output_in_embedding: bool = True
    """When set to True, clone the output of scatter_to_sequence_parallel_region in embedding layer
    to facilitate garbage collection of input."""

    ####################
    # SonicMoE
    ####################``
    using_sonic_moe: bool = False
    """When using_sonic_moe is enabled, the computation part of the moelayer will use the implementation provided by SonicMoE."""

    fp8_weight_quant_format: str = "32x32"
    """Quantization format for quantizing weights, options are 32x32 and 1x32. Currently only used in SonicMoE."""

    ####################
    # MLA
    ####################
    """Configuration object for paddlefleet Multi-Latent Attention (MLA) transformers.

    The initialization function has an argument for each parameter, including those in
    ModelParallelConfig. Included YaRN RoPE parameters that is fused in MLA.
    """

    q_lora_rank: int = 512
    """Rank of Query tensor's low rank representation."""

    kv_lora_rank: int = 512
    """Rank of Key and Value tensors' low rank representation."""

    qk_nope_head_dim: int = 64
    """Dimension of the head in the QK projection. q_head_dim = qk_nope_head_dim + qk_rope_head_dim. Original qk_head_dim"""

    qk_rope_head_dim: int = 64
    """Dimension of the position embedding in the QK projection. Original qk_pos_emb_head_dim."""

    mla_use_nope: bool = False
    """Whether to bypass rotary position embeddings in MLA and use the
    projected query and key channels directly."""

    hybrid_mla_q_lora_rank: int | None = None
    """Layer-local query low-rank width for MLA entries in a DSV4 hybrid model."""

    hybrid_mla_kv_lora_rank: int | None = None
    """Layer-local KV low-rank width for MLA entries in a DSV4 hybrid model."""

    hybrid_mla_qk_nope_head_dim: int | None = None
    """Layer-local non-positional QK width for hybrid MLA entries."""

    hybrid_mla_qk_rope_head_dim: int | None = None
    """Layer-local rotary QK width for hybrid MLA entries."""

    hybrid_mla_v_head_dim: int | None = None
    """Layer-local value-head width for hybrid MLA entries."""

    hybrid_mla_num_attention_heads: int | None = None
    """Layer-local query-head count for hybrid MLA entries."""

    hybrid_mla_num_key_value_heads: int | None = None
    """Layer-local KV-head count for hybrid MLA entries."""

    hybrid_mla_attention: str = "mha"
    """How the hybrid MLA layers (``csa_compress_ratios == -2``) run.

    Only ``-2`` layers are affected; window / CSA / HCA / CSA-full-causal-MQA
    layers ignore this field entirely.

    - ``"mha"`` (default): ``kv_b_proj`` materialises per-head K/V and dense flash
      attention runs on them. Leaving the field unset keeps this behaviour.
    - ``"mqa_dsa"``: latent MQA -- attention runs on the single shared
      ``kv_lora_rank + qk_rope_head_dim`` latent head and a DSA (Lightning)
      indexer selects the attended columns (forced local window + top-k).
    - ``"mqa_full_causal"``: latent MQA with no indexer -- attend to the full
      per-document causal set.

    Both ``mqa_*`` modes absorb the query against ``kv_b_proj.weight`` at
    *runtime* (activation level, not weight level), so the parameter layout stays
    byte-identical to ``"mha"`` and an MHA checkpoint loads unchanged. The only
    new weights are the DSA indexer's; ``"mqa_full_causal"`` adds none at all.

    ``"mqa_dsa"`` reuses the model-wide ``index_n_heads`` / ``index_head_dim`` /
    ``index_topk`` (``dsa_index_*`` internally), i.e. the same fields the CSA
    layers already read; a learnable per-head sink comes from the model-wide
    ``add_full_attention_sink_bias``. No hybrid-specific duplicates.

    ``"mqa_full_causal"`` isolates absorption from sparsity: absorption is
    activation-level and exact, so latent MQA over the *full* causal set is
    mathematically identical to ``"mha"`` and a warm-started run must track the
    MHA run to within kernel noise. Any drift is then attributable to the
    absorption or the softmax scale rather than to the top-k selection. Its index
    table is ``[b, s, s]`` int32, so memory grows with the square of the sequence
    length (268 MB per layer at ``s=8192``).

    Terminology: the ``mqa_*`` modes above are *latent MQA* (class
    ``MQALatentAttention``, ``-2`` layers). A ``csa_compress_ratios`` entry of
    ``-1`` is *CSA full-causal MQA* (class ``CompressedSparseAttention``) -- a
    different layer kind that this field does not touch.
    """

    mqa_split_kv_b_proj: bool = False
    """Split ``kv_b_proj`` into standalone ``k_b_proj`` / ``v_b_proj``
    absorption parameters instead of slicing it on every forward. Requires a
    latent MQA mode, i.e. ``hybrid_mla_attention`` in ``{"mqa_dsa",
    "mqa_full_causal"}`` -- the split only concerns absorption, so it is
    independent of whether an indexer runs.

    ``False``: both absorption weights are sliced out of ``kv_b_proj.weight`` on
    every forward and applied with an ``einsum``.

    ``True``: they are separate parameters. Each is logically the ``[G, R, D]``
    weight ``fused_grouped_matmul`` wants --
    ``k_b_proj``: ``[heads, kv_lora_rank, qk_nope_head_dim]`` (query absorption),
    ``v_b_proj``: ``[heads, v_head_dim, kv_lora_rank]`` (V de-absorption) -- but
    *stored* with the leading two dims folded, i.e. as 2-D
    ``[heads * kv_lora_rank, qk_nope_head_dim]`` and
    ``[heads * v_head_dim, kv_lora_rank]``; the 3-D form is recovered per forward
    with a zero-copy reshape. The fold exists because the AOA engine cannot
    change a tensor's rank, so the checkpoint side has to be 2-D. Each side is
    then one grouped Triton GEMM with no slice, no einsum and no transpose.
    ``kv_b_proj`` is not built at all in this mode -- the two parameters hold
    exactly its elements, so the resident parameter bytes and the checkpoint
    size are unchanged -- and the parameter set is no longer byte-compatible
    with a dense MHA phase: the checkpoint must already contain ``k_b_proj`` /
    ``v_b_proj``. Converting an older ``kv_b_proj.weight``-only checkpoint needs
    AOA statements that split it; those are not wired up yet, so such a
    checkpoint cannot be resumed with this flag on.

    Incompatible with ``enable_hy_sparse_attention``, whose ``MQASelfAttention``
    layer still absorbs against ``kv_b_proj.weight``.
    """

    hybrid_mla_cp_mode: str | None = None
    """Context parallel mode for the MLA layers only, overriding ``cp_balance_mode``.

    ``None`` (default) inherits ``cp_balance_mode``. Set to ``contiguous_a2a``
    to run Ulysses on the MLA layers of a DSV4 MLA+HCA hybrid while the HCA
    layers keep ``contiguous_allgather``; both modes share one global token
    layout, which is what makes mixing them safe.
    """

    mqa_indexer_cp_mode: str | None = None
    """Row layout the latent-MQA indexer's forward runs on, under context parallel.

    ``None`` (default) inherits ``cp_balance_mode``: the indexer scores this
    rank's own contiguous row slice, so under a causal mask its cost grows with
    the rank index. Measured at 256k/cp16: 2.2ms on cp0 vs 66.8ms on cp15 per
    layer per pass, i.e. the slowest rank does 1.94x the average and every other
    rank waits for it at the next collective.

    ``"dualchunk_p2p"`` splits the global sequence into ``2 * cp_size`` chunks
    and has rank ``r`` score chunks ``(2r, 2*cp_size-1-2r)`` instead of its own
    ``(2r, 2r+1)``. The ids sum to ``2*cp_size-1`` on every rank, and a causal
    row's candidate count grows linearly with its global position, so the work is
    equal everywhere. Only the indexer's rows move -- attention is already
    balanced at a fixed ``index_topk + window`` columns per row, so ``query`` and
    the layer output never travel. Rank ``r`` keeps the chunk it already owns and
    swaps the other with rank ``cp_size-1-r``, which reduces the exchange to a
    single point-to-point sendrecv rather than an all-to-all.

    The global token layout is untouched: this is a layer-local permutation,
    undone before the layer returns, so the HCA layers of the same model are
    unaffected and ``cp_balance_mode`` must stay contiguous.

    Only the sparse training phase honours this, i.e. ``hybrid_mla_attention=
    "mqa_dsa"`` with ``dsa_indexer_use_sparse_loss=True``. That is the only
    phase whose indexer runs a top-k over per-rank rows; the warmup phase scores
    the whole causal set through a different code path that does not permute
    rows, so the other combinations are rejected rather than accepted-and-inert.
    """

    v_head_dim: int | None = None
    """Dimension of the head in the V projection."""

    rope_type: str = "yarn"
    """Type of RoPE to use. Default to yarn, options are rope and yarn."""

    rotary_base: float = 10000
    """Rotary base for the rotary embeddings, used by rope and yarn."""

    rotary_percent: float = 1.0
    """Rotary percent for the rotary embeddings, used by rope."""

    rotary_scaling_factor: float = 40
    """Rotary scaling factor for the rotary embeddings, used by yarn."""

    original_max_position_embeddings: int = 4096
    """Original maximum position embeddings for the original model, used by yarn."""

    beta_fast: float = 32
    """Beta fast for YaRN RoPE, used by yarn."""

    beta_slow: float = 1
    """Beta slow for YaRN RoPE, used by yarn."""

    mscale: float = 1.0
    """Mscale for YaRN RoPE in Multi-Latent Attention, used by yarn."""

    mscale_all_dim: float = 0.0
    """Mscale all dimensions for YaRN RoPE in Multi-Latent Attention, used by yarn."""

    loss_subbatch_sequence_length: int = -1
    """Sequence length of subbatch for loss computation."""

    fused_linear_ce_loss_chunk: int = 0
    """Enable fused linear + cross-entropy loss when > 0.

    When set to a positive integer N, LM head skips materializing the full
    [B, S, V] logits tensor and instead passes (hidden_states, weight, bias)
    to LanguageLoss, which dispatches to LigerFusedLinearCrossEntropyFunction
    with num_chunks=N. Only compatible with tensor_model_parallel_size == 1
    (or parallel_output disabled)."""

    enable_hy_sparse_attention: bool = False
    """Enable the HySparse Attention variant.

    HySparse has the following features: (1) adding a Block Sparse Attention in SWA
    layers. (2) KV sharing between full attention and Block Sparse Attention. (3) using
    MQA instead of MLA.
    """

    hy_sparse_block_size: int = 64
    """HySparse key block size (``block_B``) used by the TileLang block-score /
    block-sparse attention operators. Key columns are grouped into contiguous
    blocks of this size (document-relative) for scoring and sparse selection.

    Default 64 follows the HySparse paper (arXiv:2602.03560, Table 1: "Sparse
    Attn Block Size = 64" for all 7B/80B configurations)."""

    hy_sparse_topk: int = 16
    """Number of key *blocks* selected per query token in the HySparse block-sparse
    branch (the ``topk`` fed to :func:`select_topk_blocks`). The full attention
    layer scores all blocks and the top-``hy_sparse_topk`` (shared across the
    query group by group-wise max) are attended by the SWA layers' block-sparse
    branch.

    Default 16 follows the HySparse paper (arXiv:2602.03560): the paper reports
    selection in *tokens* (k = 1024, "Sparse Attn TopK Tokens = 1024"), which maps
    to k / block_size = 1024 / 64 = 16 blocks. This field counts blocks, so 16 is
    the block-space equivalent of the paper's 1024-token budget."""

    hy_sparse_full_attn_use_tilelang: bool = False
    """Route the HySparse **full-attention block-score** branch through the
    independent TileLang operator (``block_score_mha_attn_fwd``) instead of the
    production FA4 fused block-score kernel (``block_score_fa4_attn_fwd``).

    Independent from :attr:`hy_sparse_block_sparse_use_tilelang`: the full-score
    and block-sparse-gather branches each pick their backend separately, so you
    can mix (e.g. TileLang scorer + production DSA gather) to isolate which
    branch an anomaly comes from.

    Set from the training YAML as a top-level key::

        enable_hy_sparse_attention: true
        hy_sparse_full_attn_use_tilelang: true      # default false -> FA4

    The TileLang op is numerically cross-checked against FA4 (bf16-level fwd+bwd
    agreement, exact block_logit and TopK-index bridge). Leave ``False`` for
    production runs (FA4 is faster)."""

    hy_sparse_block_sparse_use_tilelang: bool = False
    """Route the HySparse **block-sparse gather** branch through the independent
    TileLang operator (``block_sparse_mqa_attention_tl``) instead of the
    production cuDNN-DSA gather kernel (``block_sparse_mqa_attention_dsa``).

    Independent from :attr:`hy_sparse_full_attn_use_tilelang` (see there).

    Set from the training YAML as a top-level key::

        enable_hy_sparse_attention: true
        hy_sparse_block_sparse_use_tilelang: true   # default false -> DSA

    The TileLang op is numerically cross-checked against DSA (bf16-level fwd+bwd
    agreement) and needs no head padding / handles any ``kv_lora_rank`` natively.
    Leave ``False`` for production runs (DSA is faster)."""

    # cache_mla_latents: bool = False

    ####################
    # Linear attention (GatedDeltaNet / KimiDeltaAttention)
    ####################

    linear_conv_kernel_dim: int = 4
    """Kernel size of the short causal depthwise convolution applied to q/k/v.

    Corresponds to ``linear_attn_config["short_conv_kernel_size"]`` in Kimi's
    HuggingFace config."""

    linear_key_head_dim: int = 128
    """Per-head dimension of the linear-attention query/key vectors."""

    linear_value_head_dim: int = 128
    """Per-head dimension of the linear-attention value vectors. KDA requires it
    to equal :attr:`linear_key_head_dim` because its forget gate is per-channel
    over the key dimension."""

    linear_num_key_heads: int = 16
    """Number of linear-attention query/key heads. Must divide
    :attr:`linear_num_value_heads` (GVA) and the tensor parallel size."""

    linear_num_value_heads: int = 32
    """Number of linear-attention value heads. Must be divisible by the tensor
    parallel size."""

    linear_gate_lora_rank: int | None = None
    """KDA only. Bottleneck rank of the forget-gate (and, when
    :attr:`linear_use_full_rank_gate` is False, the output-gate) low-rank
    projection. None falls back to :attr:`linear_value_head_dim`, which is what
    Kimi uses."""

    linear_use_full_rank_gate: bool = True
    """KDA only. If True the output gate is a single full-rank projection folded
    into in_proj; if False it is a second low-rank pair (g_a_proj / g_b_proj).

    Corresponds to ``linear_attn_config["use_full_rank_gate"]``."""

    linear_gate_lower_bound: float | None = -5.0
    """KDA only. Lower bound of the log-space forget gate. When set, the gate is
    ``lower_bound * sigmoid(exp(A_log) * (a + dt_bias))``, naturally clamped to
    ``[lower_bound, 0)``. None switches to ``-exp(A_log) * softplus(a + dt_bias)``.

    Corresponds to ``linear_attn_config["gate_lower_bound"]``."""

    ####################
    # DSA (DeepSeek Sparse Attention)
    ####################

    dsa_index_n_heads: int | None = None
    """Number of DSA Indexer heads. None disables DSA; non-None activates
    DeepSeek V3.2 sparse attention path.

    Note: This field corresponds to the HuggingFace config.json field "index_n_heads".
    The mapping from HuggingFace field name to PaddleFleet internal field name is handled
    by TransformerConfig.transform_rules.
    """

    dsa_index_head_dim: int = 128
    """Per-head dimension for Indexer Q/K vectors.

    Note: This field corresponds to the HuggingFace config.json field "index_head_dim".
    The mapping from HuggingFace field name to PaddleFleet internal field name is handled
    by TransformerConfig.transform_rules.
    """

    dsa_index_topk: int = 2048
    """Number of token positions selected by Indexer per query token.

    Note: This field corresponds to the HuggingFace config.json field "index_topk".
    The mapping from HuggingFace field name to PaddleFleet internal field name is handled
    by TransformerConfig.transform_rules.
    """

    dsa_indexer_loss_coeff: float = 0.0
    """KL loss coefficient for DSA Indexer training. 0 disables the KL loss.

    Note: This field corresponds to the HuggingFace config.json field "indexer_loss_coeff".
    The mapping from HuggingFace field name to PaddleFleet internal field name is handled
    by TransformerConfig.transform_rules.
    """

    dsa_indexer_use_sparse_loss: bool = False
    """Whether to restrict DSA KL loss to top-k positions only.

    Note: This field corresponds to the HuggingFace config.json field "indexer_use_sparse_loss".
    The mapping from HuggingFace field name to PaddleFleet internal field name is handled
    by TransformerConfig.transform_rules.
    """

    dsa_indexer_rope_fusion: bool = False
    """Whether to use the fused Triton RoPE for the DSA indexer's q/k.

    Independent of the model-wide ``apply_rope_fusion``, which selects the MLA
    and HCA/CSA fused kernels. The indexer uses a third convention -- rope
    channels first, ``rotate_half`` instead of MLA's de-interleave -- so it has
    its own kernel (``fused_apply_rope_half``) and its own switch.

    Silently falls back to the eager path when
    ``dsa_indexer_rotary_interleaved`` or ``high_precision_rope`` is set (both
    select a different rotation the kernel does not implement). The remaining
    requirements are asserted by the kernel rather than silently skipped:
    bf16 activations, and fp32 freqs when mscale != 1 (i.e. YaRN).

    Bit-exact with the eager path, so it is safe to toggle mid-run.
    """

    dsa_indexer_rotary_interleaved: bool = False
    """
    Whether Indexer uses interleaved Rotary Position Embeddings.

    When False (default), Indexer uses non-interleaved RoPE with
    half-head frequencies [θ₁,θ₂,...,θ₁,θ₂,...].

    When True, Indexer uses interleaved RoPE with paired frequencies
    [θ₁,θ₁,θ₂,θ₂,...].

    This allows compatibility with MLA's YaRN RoPE which always generates
    interleaved frequencies.
    """

    ####################
    # CSA / DSv4 Hybrid Attention
    ####################

    experimental_attention_variant: str | None = None
    """Which experimental attention variant to use.
    Supported values: None (disabled), 'dsa', 'dsv4_hybrid'.
    When 'dsv4_hybrid', enables DeepSeekV4 Hybrid Attention with Compressed Sparse Attention.
    """

    csa_window_size: int = 128
    """Sliding window size for Compressed Sparse Attention (CSA).
    Each query attends to the last csa_window_size tokens via a sliding window.
    """

    csa_compress_ratios: list | None = None
    """Per-layer attention-kind assignment for the DSv4 hybrid attention stack.
    Length must equal num_hidden_layers (+ mtp_num_layers if present).
    Each entry encodes the layer kind via its integer ratio value:
      - -2: MLA layer — multi-head latent attention. How it actually runs is
        chosen by ``hybrid_mla_attention`` (``"mha"`` / ``"mqa_dsa"`` /
        ``"mqa_full_causal"``); this is the only ratio that field affects.
      - -1: CSA full-causal MQA layer — no window, no compressor, no indexer.
        A different layer kind from the latent MQA modes of ``-2``.
      - 0: window-only attention (no compression)
      - 2..127: CSA layer — overlapping compression (coff=2) with learned
        Lightning Indexer. The compression rate is a free parameter of CSA;
        any integer in [2, 127] is accepted (e.g. 4, 8, 16, ...), including
        non-power-of-2 values such as 3 or 6. The overlap pooling window
        becomes 2 * ratio tokens.
      - 128: HCA layer — non-overlapping compression, attend to all
        compressed positions
    Value 1 is rejected (ambiguous: no compression yet not window).
    """

    csa_compress_rotary_base: float = 40000.0
    """Rotary base for compressed KV positions in CSA.
    Used instead of the standard rotary_base when compress_ratio > 1 for a layer.
    """

    csa_dense_mode: bool = False
    """If True, skip CSAIndexer for CSA layers (1 < ratio < 128) and attend to all
    compressed positions.
    """

    indexer_init_from_scratch: bool | None = None
    """Whether the Indexer weights are initialized instead of loaded.

    Covers both indexer flavours of a dsv4-hybrid model: the ``CSAIndexer`` of
    the CSA layers (``1 < csa_compress_ratios[i] < 128``) and the ``DSAIndexer``
    of the latent MQA layers (``== -2`` with ``hybrid_mla_attention="mqa_dsa"``).

    This is about the *checkpoint*, not the training strategy, and is therefore
    separate from ``train_indexer_only``:

    * ``True`` -- resuming a phase-1 checkpoint: it has no Indexer tensors at all
      (phase 1 runs with ``csa_dense_mode=true`` /
      ``hybrid_mla_attention="mha"`` or ``"mqa_full_causal"``),
      so they must be initialized from scratch. Leaving it ``False`` makes the AOA
      engine abort with ``... indexer.linear_wq_b.weight should be assigned before!``.
    * ``False`` -- resuming a phase-2/3 checkpoint (e.g. a fault-tolerant restart):
      it does contain Indexer weights, and re-initializing them would
      **silently** throw away all Indexer training done so far.

    Because ``True`` is the destructive direction, it is rejected outright for
    the sparse phase (``dsa_indexer_use_sparse_loss=True``, i.e. phase 3/4) of
    *either* flavour: that phase always continues a warmup checkpoint, which
    does hold trained Indexer weights, and the add primitive would drop them.
    ``__post_init__`` raises rather than letting the run start.

    The flag is only read by the model's ``_gen_aoa_config``, which only runs on the
    HF-loading path. There it is **mandatory**: leaving it ``None`` raises, because
    which checkpoint a run starts from is a deliberate decision and must not be
    guessed. Non-HF paths (flex/DCP resume, fresh start) never read it at all.
    """

    train_indexer_only: bool = False
    """Phase 2 training strategy: train only the Indexer parameters.

    Requires that the model actually builds an Indexer -- either a ``CSAIndexer``
    (``csa_dense_mode=False`` plus a layer with ``1 < ratio < 128``) or a
    ``DSAIndexer`` (``hybrid_mla_attention="mqa_dsa"`` plus a ``-2`` layer) --
    and a positive ``dsa_indexer_loss_coeff`` so it receives a
    training signal. The backbone is frozen by the trainer before the optimizer is
    created; this flag additionally keeps the recompute segments differentiable so
    the attached indexer loss still gets a backward pass. It does not change any
    attention value, the KL candidate range, or the main attention sparsity.

    Only the base ``TransformerLayer`` keeps its recompute segment differentiable,
    so ``enable_hy_sparse_attention=True`` (which swaps in
    ``HySparseTransformerLayer``) is rejected rather than silently producing a
    gradient-free Indexer.
    """

    csa_indexer_backend: str = "tilelang"
    """CSA indexer backend. Single switch selecting one of three
    implementations of the compressed top-k indexer.

    One of {"unfused", "tilelang", "cudnn"}:
      * "unfused": Paddle/FusedDSAIndexerLoss reference path.
      * "tilelang" (default): TileLang top-k and selected-set loss path.
      * "cudnn": cuDNN indexer top-k/forward path.
    """

    csa_sparse_attn_backend: str = "tilelang"
    """CSA sparse attention backend. Single switch selecting one of three
    implementations of the final sparse MQA attention.

    One of {"unfused", "tilelang", "cudnn"}:
      * "unfused": pure-Paddle einsum forward + Paddle autograd backward
        (non-fused reference path).
      * "tilelang" (default): TileLang sparse MQA kernel forward + backward.
      * "cudnn": FlashMLA sparse forward kernel + cuDNN DSA backward
        kernel.
    """

    mqa_sparse_attn_backward_backend: str = "cudnn"
    """Backward kernel for the absorbed-MQA latent sparse attention (dkv).

    One of {"cudnn", "tilelang"}:
      * "cudnn" (default): cuDNN DSA backward. Fast, but ``dkv`` accumulates
        with atomics and is **not** run-to-run reproducible; the drift is
        bounded by ``test_block_sparse_dsa_gradcheck.py::TestDeterminism``.
      * "tilelang": deterministic backward via
        ``tilelang_ops.attn.mqa_latent_sparse_bwd``. ~14x slower on SM100 and
        bitwise stable for identical inputs, independent of
        ``FLAGS_cudnn_deterministic`` -- it always runs the atomic-free kernel
        rather than selecting one from that flag.
    The forward is always FlashMLA regardless of this switch; the tilelang
    forward kernel cannot accept ``d_qk=576`` (not a power of two).
    """

    csa_share_docmask_meta: bool = False
    """Share one ``CSADocMaskMetadata`` per (micro-batch, ratio, mask group).

    ``CSADocMaskMetadata`` is a pure function of the document mask, the compress
    ratio and the sequence length, so all DSv4-hybrid layers of a micro-batch
    that agree on those inputs can reuse a single instance instead of each
    rebuilding its own (see ``doc_mask_meta_registry.DocMaskMetaRegistry``). When
    ``False`` every layer builds its own metadata exactly as before.

    The trainer builds the whole step's metadata before
    ``forward_backward_pipeline`` and audits the per-layer forward counters at
    each step boundary (see ``PretrainingTrainer.training_pipeline_step``).
    """

    mqa_share_docmask_meta: bool = False
    """Same sharing for the latent-MQA (``csa_compress_ratios == -2``) layers.

    Separate from ``csa_share_docmask_meta`` because the two cover different
    layer kinds with different metadata classes (``MQADocMeta`` vs
    ``CSADocMaskMetadata``), so they can be enabled and measured independently.
    Requires the layers to actually run latent MQA -- ``__post_init__`` rejects
    the switch otherwise rather than let it be a silent no-op.
    """

    sparse_attn_global_kv_idx_remap_fusion: bool = False
    """Whether to fuse the per-batch-local -> flat-global KV column index remap
    (``idx + b * seqlen_kv``) consumed by the cuDNN / FlashMLA sparse-attention
    kernels (``csa_sparse_attn_utils._local_to_global_flat``).

    Not about MoE routing: these are KV *column* indices of the sparse-attention
    support (window + compressed slots), not expert top-k ids.

    The eager version spends seven elementwise kernels on the full
    ``[b * sq, topk]`` table (``full`` + ``greater_equal`` + ``arange`` +
    ``expand`` + ``scale`` + ``add`` + ``where``) to express a single pass; the
    Triton kernel does it in one. The result is bit-identical, so this only
    trades kernel count for a Triton dependency and can be flipped freely.

    Scope: every ``_local_to_global_flat`` call site -- the ``"cudnn"``
    sparse-attention forward and backward of both
    ``CompressedSparseAttention`` (HCA ``ratio=128`` and CSA/DSA
    ``1 < ratio < 128`` layers) and ``MQALatentAttention``. No effect on the
    ``"tilelang"`` / ``"unfused"`` backends, which never build the flat global
    index table, nor on ``block_sparse_mqa_attention_dsa``, which leaves it at
    the default. ``MQALatentAttention``'s forward is always FlashMLA, so it
    remaps regardless of ``mqa_sparse_attn_backward_backend``; only its
    backward follows that switch.
    """

    stage1_overlap: bool = False
    """
    overlap backward with sharding gradient reduce for non-pipeline parallelism
    """

    use_fast_hadamard: bool = False
    """Use Tridao's fast Hadamard transform for DSv4 rotate activation function."""

    o_groups: int = 8
    """Number of groups for grouped low-rank output projection (wo_a) in DSv4 Hybrid.
    Set to 0 to use a single linear output projection instead.
    """

    o_lora_rank: int = 1024
    """Low-rank dimension per group for the grouped output projection in DSv4 Hybrid."""

    qk_pos_emb_head_dim: int | None = None
    """Dimension of positional embedding portion in each QK head for DSv4 Hybrid.
    When set, the total head dim is split as: v_head_dim = qk_nope_dim + qk_pos_emb_head_dim.
    The positional embedding (RoPE) is applied only to the last qk_pos_emb_head_dim dims.
    """

    hca_rope_type: str | None = None
    """Per-attention-type RoPE variant for HCA layers (csa_compress_ratios == 128).
    Options: "rope" (plain RoPE) or "yarn" (YaRN). When None, keeps the historical
    default (compressed layers use YaRN). The RoPE width stays qk_pos_emb_head_dim.
    """

    csa_rope_type: str | None = None
    """Per-attention-type RoPE variant for CSA layers (2 <= csa_compress_ratios < 128).
    Options: "rope" (plain RoPE) or "yarn" (YaRN). When None, keeps the historical
    default (compressed layers use YaRN). The RoPE width stays qk_pos_emb_head_dim.
    """

    gpt_model_use_experimental_version: bool = False
    """Enable experimental version code paths for precision alignment."""

    use_accuracy_compatible: bool = False
    """Whether to enable accuracy-compatible kernels for cross-framework numerical
    alignment. Defaults to False."""

    moe_topk_fusion: bool = False
    """If True, use Triton fused MoE TopK kernel for expert selection."""

    routing_map_fusion: bool = False
    """If True, use Triton fused routing map kernel for MoE routing."""

    magic_init: bool = False
    """Use the magic initialization method."""

    use_truncated_normal_init: bool = False
    """Use truncated normal init N(0, sigma^2) clipped to
    [-truncated_normal_init_factor*sigma, truncated_normal_init_factor*sigma].
    Sigma prefers init_method_std, falling back to 0.5/sqrt(hidden_size)
    when init_method_std is None. Independent switch; takes precedence over
    magic_init when enabled."""

    truncated_normal_init_factor: float = 3.0
    """Truncation factor for use_truncated_normal_init: clip range is
    [-factor*sigma, factor*sigma]."""

    ####################
    # Ernie Trainer Configs
    ####################

    moe_logging: bool = False
    """Whether to enable MoE logging."""

    deepep_buffer_configs: dict | None = None
    """DeepEP buffer configuration."""

    # Field name mapping rules: HuggingFace config.json name -> TransformerConfig name
    transform_rules = {
        # DSA field mapping
        "index_n_heads": "dsa_index_n_heads",
        "index_head_dim": "dsa_index_head_dim",
        "index_topk": "dsa_index_topk",
        "indexer_loss_coeff": "dsa_indexer_loss_coeff",
        "indexer_use_sparse_loss": "dsa_indexer_use_sparse_loss",
        "indexer_rotary_interleaved": "dsa_indexer_rotary_interleaved",
        "indexer_rope_interleave": "dsa_indexer_rotary_interleaved",
        # CSA / DSv4 Hybrid field mapping
        "csa_window_size": "csa_window_size",
        "csa_compress_ratios": "csa_compress_ratios",
        "csa_compress_rotary_base": "csa_compress_rotary_base",
        "csa_dense_mode": "csa_dense_mode",
        "csa_indexer_backend": "csa_indexer_backend",
        "csa_sparse_attn_backend": "csa_sparse_attn_backend",
        "csa_share_docmask_meta": "csa_share_docmask_meta",
        "mqa_share_docmask_meta": "mqa_share_docmask_meta",
        "o_groups": "o_groups",
        "o_lora_rank": "o_lora_rank",
        "qk_pos_emb_head_dim": "qk_pos_emb_head_dim",
        "hca_rope_type": "hca_rope_type",
        "csa_rope_type": "csa_rope_type",
        "mqa_sparse_attn_backward_backend": "mqa_sparse_attn_backward_backend",
    }

    # Config keys that were renamed and deliberately left without a silent
    # alias: value -> the migration hint shown when a stale key is supplied.
    # ``_process_attribute``'s fallback is ``setattr``, so without this table a
    # stale key would be absorbed as a dead attribute and the feature it used to
    # switch on would silently stay off. Same intent as the
    # ``sonicmoe_quant_format`` guard below.
    renamed_config_keys = {
        "non_absorbed_mqa": (
            "Use hybrid_mla_attention instead: non_absorbed_mqa=True becomes "
            "hybrid_mla_attention='mqa_dsa', non_absorbed_mqa=False becomes "
            "hybrid_mla_attention='mha' (the default)."
        ),
        "non_absorbed_mqa_dense": (
            "Use hybrid_mla_attention instead: non_absorbed_mqa_dense=True "
            "becomes hybrid_mla_attention='mqa_full_causal', "
            "non_absorbed_mqa_dense=False becomes hybrid_mla_attention='mha' "
            "(the default)."
        ),
        "csa_train_indexer_only": "Use train_indexer_only instead.",
        "csa_indexer_init_from_scratch": "Use indexer_init_from_scratch instead.",
    }

    @classmethod
    def from_config(cls, config_dict):
        # note(zhangweilong): if cls(),will call __post_init__ directly,but __new__ will skip some attr init .please check provider attr
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
                if value == "gelu_pytorch_tanh":
                    func = functools.partial(F.gelu, approximate=True)
                elif value == "situ":
                    func = situ
                else:
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
        elif key == "sonicmoe_quant_format":
            raise ValueError(
                "sonicmoe_quant_format is deprecated. Use fp8_weight_quant_format instead."
            )
        elif key in self.renamed_config_keys:
            raise ValueError(
                f"{key} was renamed and is no longer supported. "
                f"{self.renamed_config_keys[key]} Update the config that still "
                f"sets {key}; it would otherwise be silently ignored."
            )
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
        if self.mtp_shared_last_layer:
            # When MTP reuses the last backbone TransformerLayer's parameters,
            # the MTP transformer block must have an identical structure to the
            # backbone-last layer (same MoE / dense shape). Force-disable
            # use_dense_mtp so the MTP layer matches whatever the backbone is.
            assert not self.use_dense_mtp, (
                "mtp_shared_last_layer cannot be True if use_dense_mtp= True"
            )

        if self.separate_mtp_input:
            # Raise instead of assert: with ``python -O`` assertions are stripped,
            # and an unsupported combination would then silently enter a path that
            # only holds for the layout below -- or crash much later inside
            # MultiTokenPredictionLayer with a missing ``mtp_decoder_inputs``.
            if self.num_nextn_predict_layers != 1:
                raise ValueError(
                    "separate_mtp_input only supports "
                    "num_nextn_predict_layers == 1, got "
                    f"num_nextn_predict_layers={self.num_nextn_predict_layers}. "
                    "The MTP input is consumed once and stripped from dict_args, "
                    "so deeper MTP layers would not receive it."
                )
            if self.pipeline_model_parallel_size != 1:
                raise ValueError(
                    "separate_mtp_input requires pipeline_model_parallel_size "
                    "== 1, got pipeline_model_parallel_size="
                    f"{self.pipeline_model_parallel_size}. Use "
                    "enable_mtp_magic_send for pipeline_model_parallel_size > 1."
                )
            if self.enable_mtp_magic_send:
                raise ValueError(
                    "separate_mtp_input and enable_mtp_magic_send are mutually "
                    "exclusive, got separate_mtp_input=True and "
                    "enable_mtp_magic_send=True. They are two transports for the "
                    "same tensor; pick the one matching the pipeline degree."
                )
            if self.mtp_load_weight_only:
                raise ValueError(
                    "separate_mtp_input is incompatible with "
                    "mtp_load_weight_only=True. GPTEmbedding does not build the "
                    "shifted MTP embeddings in that mode, so separate_mtp_input "
                    "would silently do nothing."
                )

        if self.enable_mtp_magic_send:
            assert not getattr(self, "tie_word_embeddings", False), (
                "enable_mtp_magic_send with tie_word_embeddings=True is not yet validated. "
                "Please disable tie_word_embeddings when using magic send MTP."
            )
            assert not self.mtp_shared_last_layer, (
                "enable_mtp_magic_send and mtp_shared_last_layer cannot both be True. "
                "Magic send uses per-layer mtp_embed with broadcast sync, which is "
                "incompatible with SharedLayerDesc-based last-layer reuse."
            )
            if self.num_nextn_predict_layers > 1:
                assert self.variable_seq_lengths, (
                    "enable_mtp_magic_send with num_nextn_predict_layers > 1 requires "
                    "variable_seq_lengths=True (dynamic-shape P2P)."
                )
            assert self.pipeline_model_parallel_size > 1, (
                "enable_mtp_magic_send requires pipeline_model_parallel_size > 1"
            )
            # Raise instead of assert: with ``python -O`` an assertion would be
            # stripped and the run would silently enter a branch that is known
            # to be wrong (double sequence-parallel scatter of the embedding and
            # an untruncated visual mask).
            if self.multimodal_embedding:
                raise ValueError(
                    "enable_mtp_magic_send with multimodal_embedding=True is "
                    "not supported: GPTEmbedding's magic-send branch truncates "
                    "(and, under sequence_parallel, already scatters) "
                    "decoder_input without producing mtp_emb_res, so the "
                    "multimodal branch would scatter a second time and leave "
                    "visual_pos_masks at full length."
                )
            if (
                self.virtual_pipeline_model_parallel_size is not None
                and self.virtual_pipeline_model_parallel_size > 1
            ):
                assert self.overlap_p2p_comm, (
                    "enable_mtp_magic_send with vpp requires overlap_p2p_comm=True"
                )
                assert self.variable_seq_lengths, (
                    "enable_mtp_magic_send with vpp requires variable_seq_lengths=True"
                )

        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size

        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.v_head_dim is None:
            self.v_head_dim = self.head_dim

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.swa_head_dim is None:
            self.swa_head_dim = self.head_dim
        if self.swa_v_head_dim is None:
            self.swa_v_head_dim = self.v_head_dim
        if self.swa_num_attention_heads is None:
            self.swa_num_attention_heads = self.num_attention_heads
        if self.swa_num_key_value_heads is None:
            self.swa_num_key_value_heads = self.num_key_value_heads
        if self.swa_rope_theta is None:
            self.swa_rope_theta = self.rope_theta

        if self.vha_q_lora_rank is None:
            self.vha_q_lora_rank = self.head_dim

        if self.swa_vha_q_lora_rank is None:
            self.swa_vha_q_lora_rank = self.swa_head_dim

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

        if self.use_truncated_normal_init:
            if self.truncated_normal_init_factor <= 0:
                raise ValueError(
                    "truncated_normal_init_factor must be positive when use_truncated_normal_init is True."
                )
            if self.init_method_std is None and self.hidden_size == 0:
                raise ValueError(
                    "hidden_size must be non-zero when init_method_std is None "
                    "and use_truncated_normal_init is True."
                )
            sigma = (
                self.init_method_std
                if self.init_method_std is not None
                else 0.5 / math.sqrt(self.hidden_size)
            )
            self.init_method = truncated_init_method_normal(
                sigma, truncate_factor=self.truncated_normal_init_factor
            )
            self.init_method_std = sigma
            logger.info(
                f"[init] use_truncated_normal_init=True: TruncNormal(0, sigma^2) clipped to "
                f"[-{self.truncated_normal_init_factor}*sigma, {self.truncated_normal_init_factor}*sigma], "
                f"sigma={sigma}"
            )
        elif self.magic_init:
            if self.hidden_size == 0:
                raise ValueError(
                    "hidden_size must be non-zero when magic_init is True."
                )
            sigma = math.sqrt(0.3333 / self.hidden_size)
            self.init_method = get_magic_init_method(sigma)
            self.init_method_std = sigma
        elif self.init_method is None:
            self.init_method = init_method_normal(self.init_method_std)

        if (
            self.first_k_dense_replace
            and self.moe_layer_freq is not None
            and not isinstance(self.moe_layer_freq, int)
        ):
            raise ValueError(
                "Cannot specify both first_k_dense_replace and moe_layer_freq."
            )
        if self.first_k_dense_replace is None and self.moe_layer_freq is None:
            self.moe_layer_freq = 1
        if self.first_k_dense_replace:
            if self.moe_layer_freq:
                moe_layer_pattern = [
                    1 if ((i + 1) % self.moe_layer_freq == 0) else 0
                    for i in range(
                        self.num_hidden_layers - self.first_k_dense_replace
                    )
                ]
            else:
                moe_layer_pattern = [1] * (
                    self.num_hidden_layers - self.first_k_dense_replace
                )
            self.moe_layer_freq = [
                0
            ] * self.first_k_dense_replace + moe_layer_pattern
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

        if self.use_truncated_normal_init or self.magic_init:
            self.output_layer_init_method = self.init_method
        elif self.output_layer_init_method is None:
            self.output_layer_init_method = scaled_init_method_normal(
                self.init_method_std,
                self.num_hidden_layers,
                multiplier=2.0 if not self.is_hybrid_model else 1.0,
            )

        # Set the embedding init method
        if self.embedding_init_method_std is None:
            # By default, use the same init std as you use for every other non-output layer.
            self.embedding_init_method_std = self.init_method_std

        if self.use_truncated_normal_init or self.magic_init:
            self.embedding_init_method = self.init_method
            self.embedding_init_method_std = self.init_method_std
        elif self.embedding_init_method is None:
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

        # Hyper-connection (mHC) validation
        if self.use_fused_mhc:
            if not self.enable_hyper_connections:
                raise ValueError(
                    "use_fused_mhc requires enable_hyper_connections=True."
                )

        # ``hybrid_mla_attention`` is validated unconditionally, i.e. outside the
        # ``dsv4_hybrid`` / ``-2 in csa_compress_ratios`` guards below. A mode that
        # no layer can honour is a configuration mistake and must fail at startup
        # rather than be silently ignored.
        hybrid_mla_modes = ("mha", "mqa_dsa", "mqa_full_causal")
        if self.hybrid_mla_attention not in hybrid_mla_modes:
            raise ValueError(
                f"hybrid_mla_attention={self.hybrid_mla_attention!r} is invalid. "
                "It must be one of: 'mha' (per-head K/V materialised, dense flash "
                "attention -- the default), 'mqa_dsa' (latent MQA + DSA indexer "
                "selecting window + top-k columns), or 'mqa_full_causal' (latent "
                "MQA with no indexer, full per-document causal set)."
            )
        if self.hybrid_mla_attention != "mha":
            ratios = self.csa_compress_ratios or []
            ratio_counts: dict = {}
            for r in ratios:
                key = int(r) if hasattr(r, "__index__") else r
                ratio_counts[key] = ratio_counts.get(key, 0) + 1
            if (
                self.experimental_attention_variant != "dsv4_hybrid"
                or -2 not in ratio_counts
            ):
                raise ValueError(
                    f"hybrid_mla_attention={self.hybrid_mla_attention!r} only "
                    "applies to MLA layers, i.e. csa_compress_ratios entries "
                    "equal to -2, but this config has none: "
                    "experimental_attention_variant="
                    f"{self.experimental_attention_variant!r}, ratio value "
                    f"counts={ratio_counts or 'csa_compress_ratios unset'}. "
                    "Either mark the target layers with -2, or set "
                    "hybrid_mla_attention='mha' (the default). Note that ratio "
                    "-1 is CSA full-causal MQA, a different layer kind that "
                    "hybrid_mla_attention does not control."
                )
        if self.hybrid_mla_attention == "mqa_full_causal":
            logger.warning(
                "hybrid_mla_attention='mqa_full_causal' attends over the whole "
                "per-document causal span on every MLA layer. It exists to "
                "isolate absorption from sparsity, not to save memory. FA4 "
                "dense flashmask is its only backend -- context parallelism "
                "included -- so it needs an SM100+ box (as the sparse phases "
                "do) with FLAGS_cudnn_deterministic off; anything else raises "
                "in the first forward. See "
                "MQALatentAttention._assert_dense_fa4."
            )
        if self.hybrid_mla_attention == "mqa_dsa":
            # On the ``-2`` layers ``dsa_indexer_use_sparse_loss`` decides both
            # the indexer-loss candidate set and the attention candidate set,
            # because they are one decision: while the indexer is still being
            # learned attention must not consume its ranking, so the warmup phase
            # runs full-causal attention and a KL over the whole causal set --
            # no top-k on either side. The two training phases are therefore
            # fixed pairs, and the two mixed combinations are configuration
            # mistakes rather than modes.
            if self.train_indexer_only and self.dsa_indexer_use_sparse_loss:
                raise ValueError(
                    "train_indexer_only=True with "
                    "dsa_indexer_use_sparse_loss=True is not a valid phase. "
                    "Training only the Indexer is the warmup phase, whose KL "
                    "spans the whole per-document causal set "
                    "(dsa_indexer_use_sparse_loss=False) so the freshly "
                    "initialised Indexer is supervised on columns it did not "
                    "pick; the sparse loss only scores the columns it already "
                    "selected and lets it reinforce its own initial ranking. "
                    "Set dsa_indexer_use_sparse_loss=False in the model config, "
                    "or drop train_indexer_only."
                )
            if (
                not self.train_indexer_only
                and not self.dsa_indexer_use_sparse_loss
            ):
                logger.warning(
                    "hybrid_mla_attention='mqa_dsa' with "
                    "dsa_indexer_use_sparse_loss=False runs the warmup shape "
                    "(full per-document causal attention plus a KL over the "
                    "whole causal set, no top-k anywhere) while every backbone "
                    "parameter still trains. The "
                    "production warmup phase pairs it with "
                    "train_indexer_only=True; the sparse training phase pairs "
                    "dsa_indexer_use_sparse_loss=True with a trainable backbone."
                )

        # Hyper-connection (mHC) validation
        if self.use_fused_mhc:
            if not self.enable_hyper_connections:
                raise ValueError(
                    "use_fused_mhc requires enable_hyper_connections=True."
                )

        # Shared document-mask metadata validation. Both switches only mean
        # something on a DSv4-hybrid model, and the MQA one additionally needs the
        # -2 layers to actually run latent MQA. A switch that no layer can honour
        # is a configuration mistake and must fail at startup rather than be
        # silently ignored -- same rule as hybrid_mla_attention above.
        #
        # ``experimental_dataflow`` is checked by the trainer instead, not here:
        # it is not a model_config.json field, so a partial construction (the
        # startup ``[plan]`` probe builds one from the JSON alone) would see the
        # ``False`` default and fail on a config that is actually fine.
        if self.csa_share_docmask_meta or self.mqa_share_docmask_meta:
            if self.experimental_attention_variant != "dsv4_hybrid":
                raise ValueError(
                    "csa_share_docmask_meta / mqa_share_docmask_meta share the "
                    "per-micro-batch document-mask metadata of the DSv4-hybrid "
                    "layers, but experimental_attention_variant="
                    f"{self.experimental_attention_variant!r}, so no layer would "
                    "ever read it. Set experimental_attention_variant="
                    "'dsv4_hybrid' or drop the switch."
                )
            if not self.enable_hyper_connections:
                raise ValueError(
                    "csa_share_docmask_meta / mqa_share_docmask_meta require "
                    "enable_hyper_connections=True. The per-micro-batch slot "
                    "index is handed to the attention modules by "
                    "HyperConnectionTransformerLayer's _docmask_meta_kwargs "
                    "override; the base TransformerLayer opts out, so with "
                    "enable_hyper_connections=False every layer would fall back "
                    "to building its own metadata and the switch would do "
                    "nothing at all."
                )
        if self.csa_share_docmask_meta:
            csa_kinds = [
                int(r)
                for r in (self.csa_compress_ratios or [])
                if int(r) in (-1, 0, 128) or 2 <= int(r) < 128
            ]
            if not csa_kinds:
                raise ValueError(
                    "csa_share_docmask_meta applies to the DSv4-hybrid layers, "
                    "i.e. csa_compress_ratios entries in {-1, 0, 128} or "
                    "2 <= r < 128, but this config has none: "
                    f"csa_compress_ratios={self.csa_compress_ratios!r}. Only -2 "
                    "layers are present, which are MLA -- use "
                    "mqa_share_docmask_meta for those."
                )
        if self.mqa_share_docmask_meta:
            has_latent_mqa = self.hybrid_mla_attention in (
                "mqa_dsa",
                "mqa_full_causal",
            ) and -2 in [int(r) for r in (self.csa_compress_ratios or [])]
            if not has_latent_mqa:
                raise ValueError(
                    "mqa_share_docmask_meta applies to the latent-MQA layers, "
                    "i.e. csa_compress_ratios entries equal to -2 under "
                    "hybrid_mla_attention='mqa_dsa' or 'mqa_full_causal', but "
                    f"this config has hybrid_mla_attention="
                    f"{self.hybrid_mla_attention!r} and "
                    f"{'no' if -2 not in [int(r) for r in (self.csa_compress_ratios or [])] else 'some'}"
                    " -2 layers. Under the default 'mha' those layers are dense "
                    "MLA and build no MQADocMeta, so the switch would be a silent "
                    "no-op. Use csa_share_docmask_meta for the HCA/CSA layers."
                )

        # DSv4 Hybrid Attention validation
        if self.experimental_attention_variant == "dsv4_hybrid":
            if self.csa_compress_ratios is None:
                raise ValueError(
                    "experimental_attention_variant='dsv4_hybrid' requires "
                    "csa_compress_ratios to be set."
                )
            mtp_num_layers = (
                self.mtp_num_layers or self.num_nextn_predict_layers
            )
            if (
                self.mtp_num_layers > 0
                and self.num_nextn_predict_layers > 0
                and self.mtp_num_layers != self.num_nextn_predict_layers
            ):
                raise ValueError(
                    "mtp_num_layers and num_nextn_predict_layers must be equal when "
                    f"both are positive, got {self.mtp_num_layers} and "
                    f"{self.num_nextn_predict_layers}"
                )
            if (
                len(self.csa_compress_ratios)
                != self.num_hidden_layers + mtp_num_layers
            ):
                raise ValueError(
                    f"csa_compress_ratios length ({len(self.csa_compress_ratios)}) "
                    f"must equal num_hidden_layers ({self.num_hidden_layers + mtp_num_layers})."
                )
            for i, r in enumerate(self.csa_compress_ratios):
                # Accept python int and numpy integer scalars (a ratios list
                # loaded from npy/np.load yields np.int64, which would otherwise
                # be rejected); reject bool / np.bool_ so True does not sneak
                # through as 1, and reject floats.
                is_integral = hasattr(r, "__index__") and type(
                    r
                ).__name__ not in ("bool", "bool_")
                if not (is_integral and (r in (-2, -1, 0) or 2 <= r <= 128)):
                    raise ValueError(
                        f"csa_compress_ratios[{i}]={r} is invalid. "
                        f"Each value must be -2 (MLA), -1 (full-causal MQA), "
                        f"0 (window), an integer in [2, 127] "
                        f"(CSA, overlap + Lightning Indexer), or 128 (HCA)."
                    )
            # There are two Indexer flavours, one per attention family, and both
            # are configured by the same two fields. ``csa_dense_mode`` drops the
            # CSAIndexer and only ``hybrid_mla_attention='mqa_dsa'`` builds the
            # DSAIndexer, exactly as ``gpt_layer_specs.py`` decides.
            has_csa_indexer = not self.csa_dense_mode and any(
                1 < int(ratio) < 128 for ratio in self.csa_compress_ratios
            )
            has_mqa_indexer = self.hybrid_mla_attention == "mqa_dsa" and any(
                int(ratio) == -2 for ratio in self.csa_compress_ratios
            )
            if (
                (has_csa_indexer or has_mqa_indexer)
                and self.dsa_indexer_use_sparse_loss
                and self.indexer_init_from_scratch
            ):
                # The sparse phase always continues a warmup checkpoint, which
                # does contain trained Indexer tensors. Asking for a
                # from-scratch Indexer there makes the caller's AOA config emit
                # the ``_ -> key`` add primitive, and that primitive *ignores* a
                # checkpoint tensor of the same name instead of preferring it,
                # so the whole warmup phase is thrown away with nothing louder
                # than an "Unexpected keys" warning. Starting the sparse phase
                # from a checkpoint that has no Indexer at all is not a
                # supported configuration either: attention consumes the
                # Indexer's ranking there (CSA through
                # ``_resolve_topk_effective``, latent MQA through the window +
                # top-k index table), so a random Indexer means attending to
                # random columns -- precisely what the warmup phase exists to
                # prevent. Fail here, before any weight is loaded.
                raise ValueError(
                    "indexer_init_from_scratch=True with "
                    "dsa_indexer_use_sparse_loss=True is not a valid phase. "
                    "This config builds an Indexer (CSAIndexer="
                    f"{has_csa_indexer}, DSAIndexer={has_mqa_indexer}) and the "
                    "sparse phase resumes the warmup checkpoint, which already "
                    "holds trained Indexer weights that the HF-loading add "
                    "primitive would silently discard. Set "
                    "indexer_init_from_scratch=False; only the warmup phase "
                    "(dsa_indexer_use_sparse_loss=False), which starts from a "
                    "checkpoint with no Indexer at all, may set it True."
                )
            if -2 in self.csa_compress_ratios:
                hybrid_mla_fields = (
                    "hybrid_mla_q_lora_rank",
                    "hybrid_mla_kv_lora_rank",
                    "hybrid_mla_qk_nope_head_dim",
                    "hybrid_mla_qk_rope_head_dim",
                    "hybrid_mla_v_head_dim",
                    "hybrid_mla_num_attention_heads",
                    "hybrid_mla_num_key_value_heads",
                )
                invalid = [
                    name
                    for name in hybrid_mla_fields
                    if not isinstance(getattr(self, name, None), int)
                    or isinstance(getattr(self, name, None), bool)
                    or getattr(self, name) <= 0
                ]
                if invalid:
                    raise ValueError(
                        "hybrid MLA dimensions must be explicit positive integers; "
                        f"invalid fields: {', '.join(invalid)}"
                    )
                if self.mqa_split_kv_b_proj and (
                    self.hybrid_mla_attention
                    not in ("mqa_dsa", "mqa_full_causal")
                ):
                    raise ValueError(
                        "mqa_split_kv_b_proj=True only means "
                        "anything for latent MQA, i.e. "
                        "hybrid_mla_attention='mqa_dsa' or 'mqa_full_causal'; "
                        "it splits those modes' kv_b_proj into standalone "
                        "k_b_proj / v_b_proj absorption parameters."
                    )
                if self.mqa_split_kv_b_proj and getattr(
                    self, "enable_hy_sparse_attention", False
                ):
                    # The split replaces ``kv_b_proj`` entirely, but HySparse
                    # swaps the layer class for ``MQASelfAttention``, whose
                    # forward and decode paths still read
                    # ``kv_b_proj.weight``. Allowing the combination would
                    # either resurrect the duplicate parameter or fail on a
                    # ``None`` attribute deep in the forward.
                    raise ValueError(
                        "mqa_split_kv_b_proj=True is incompatible with "
                        "enable_hy_sparse_attention: the HySparse MQA layer "
                        "still absorbs against kv_b_proj.weight, which the "
                        "split removes."
                    )
                if self.hybrid_mla_attention == "mqa_dsa":
                    # The -2 layers' indexer reuses the CSA indexer fields.
                    # ``index_n_heads`` / ``index_head_dim`` set the indexer's
                    # parameter shapes and are needed in both DSA phases; the
                    # cuDNN indexer forward requires D_i=128.
                    index_fields = [
                        "dsa_index_n_heads",
                        "dsa_index_head_dim",
                    ]
                    # ``index_topk`` is phase 3 only: the warmup phase
                    # (``dsa_indexer_use_sparse_loss=False``) runs no top-k at
                    # all -- its KL spans the whole per-document causal set -- so
                    # it must not be required to carry a top-k budget.
                    if self.dsa_indexer_use_sparse_loss:
                        index_fields.append("dsa_index_topk")
                    invalid_index = [
                        name
                        for name in index_fields
                        if not isinstance(getattr(self, name, None), int)
                        or isinstance(getattr(self, name, None), bool)
                        or getattr(self, name) <= 0
                    ]
                    if invalid_index:
                        raise ValueError(
                            "hybrid_mla_attention='mqa_dsa' runs a DSA indexer on "
                            "the hybrid MLA layers and needs "
                            f"{' / '.join(index_fields)} as positive integers; "
                            f"invalid fields: {', '.join(invalid_index)}"
                        )
                    if self.dsa_index_head_dim != 128:
                        raise ValueError(
                            "hybrid_mla_attention='mqa_dsa' uses the cuDNN "
                            "indexer, which requires index_head_dim=128, got "
                            f"{self.dsa_index_head_dim}."
                        )
                    # ``index_n_heads`` is deliberately *not* pinned to 64 here.
                    # The warmup phase's tilelang indexer does need exactly 64
                    # (measured: 8 dies inside the kernel with a bare
                    # "Check failed: (m_warp * n_warp == num_warps)"), but
                    # enforcing it at config time would make every small-geometry
                    # unit fixture unrepresentable. The check lives at the first
                    # use instead --
                    # ``MQALatentAttention._check_tilelang_full_candidate_support``
                    # -- which still raises before any kernel launch.
                    if self.dsa_indexer_use_sparse_loss and (
                        self.dsa_index_topk % 128 != 0
                        or self.dsa_index_topk > 2048
                    ):
                        # The cuDNN indexer backward asserts
                        # ``topk % block_I == 0`` with ``block_I=128`` and
                        # ``indexer_top_k/api.py:92`` rejects ``topk > 2048``
                        # outright. Fail here instead of minutes later at the
                        # first forward.
                        raise ValueError(
                            "index_topk must be a multiple of 128 and at most "
                            "2048 (cuDNN indexer backward block size / top-k "
                            f"limit), got {self.dsa_index_topk}."
                        )

            if (
                getattr(self, "csa_tilelang_enable_sparse_attn", None)
                is not None
            ):
                raise ValueError(
                    "csa_tilelang_enable_sparse_attn has been removed. Use "
                    "csa_sparse_attn_backend in {'unfused', 'tilelang', 'cudnn'} "
                    "instead (unfused=non-fused Paddle, tilelang=TileLang "
                    "fwd/bwd, cudnn=FlashMLA fwd + cuDNN bwd)."
                )
            if getattr(self, "csa_tilelang_enable_indexer", None) is not None:
                raise ValueError(
                    "csa_tilelang_enable_indexer has been removed. Use "
                    "csa_indexer_backend in {'unfused', 'tilelang', 'cudnn'} "
                    "instead (unfused=non-fused Paddle/FusedDSAIndexerLoss, "
                    "tilelang=TileLang indexer, cudnn=cuDNN indexer)."
                )
            if getattr(self, "csa_tilelang_backend", None) is not None:
                raise ValueError(
                    "csa_tilelang_backend has been removed. Use "
                    "csa_indexer_backend in {'unfused', 'tilelang', 'cudnn'} "
                    "and csa_sparse_attn_backend in {'unfused', 'tilelang', 'cudnn'} "
                    "instead."
                )
            valid_indexer_backends = {"unfused", "tilelang", "cudnn"}
            if self.csa_indexer_backend not in valid_indexer_backends:
                raise ValueError(
                    f"csa_indexer_backend={self.csa_indexer_backend!r} is invalid. "
                    "Must be one of {'unfused', 'tilelang', 'cudnn'}."
                )
            if self.csa_sparse_attn_backend not in {
                "unfused",
                "tilelang",
                "cudnn",
            }:
                raise ValueError(
                    f"csa_sparse_attn_backend={self.csa_sparse_attn_backend!r} is invalid. "
                    "Must be one of {'unfused', 'tilelang', 'cudnn'}."
                )
            if self.mqa_sparse_attn_backward_backend not in {
                "cudnn",
                "tilelang",
            }:
                raise ValueError(
                    f"mqa_sparse_attn_backward_backend="
                    f"{self.mqa_sparse_attn_backward_backend!r} is invalid. "
                    "Must be one of {'cudnn', 'tilelang'}."
                )

            # Per-attention-type RoPE variant validation (HCA / CSA).
            valid_rope_types = {"rope", "yarn"}
            for name in ("hca_rope_type", "csa_rope_type"):
                val = getattr(self, name, None)
                if val is not None and val not in valid_rope_types:
                    raise ValueError(
                        f"{name}={val!r} is invalid. Must be one of "
                        "{'rope', 'yarn'} or None (keep default)."
                    )
            if self.train_indexer_only:
                loss_coeff = getattr(self, "dsa_indexer_loss_coeff", None)
                if not loss_coeff or float(loss_coeff) <= 0:
                    raise ValueError(
                        "train_indexer_only=True requires a positive "
                        f"dsa_indexer_loss_coeff, got {loss_coeff!r}; otherwise the "
                        "Indexer receives no training signal."
                    )
                # Either Indexer flavour is enough to have something to train
                # (``has_csa_indexer`` / ``has_mqa_indexer`` computed above).
                if not (has_csa_indexer or has_mqa_indexer):
                    raise ValueError(
                        "train_indexer_only=True requires the model to build at "
                        "least one Indexer, and this config builds none, so there "
                        "would be no trainable parameter left. Either a CSA layer "
                        "(csa_dense_mode=False plus some 1 < csa_compress_ratios[i] "
                        "< 128) or a latent MQA layer with a DSA indexer "
                        "(hybrid_mla_attention='mqa_dsa' plus some "
                        "csa_compress_ratios[i] == -2). Got "
                        f"csa_dense_mode={self.csa_dense_mode}, "
                        f"hybrid_mla_attention={self.hybrid_mla_attention!r}, "
                        f"csa_compress_ratios={self.csa_compress_ratios}."
                    )
                if getattr(self, "enable_hy_sparse_attention", False):
                    # enable_hy_sparse_attention swaps the layer class for
                    # HySparseTransformerLayer, whose recompute call does not go
                    # through keep_indexer_grad_path(). With a frozen backbone the
                    # recompute segment output would stay stop_gradient=True and the
                    # attached indexer loss would *silently* never get a backward
                    # pass. HySparse also feeds a ``shared_kv`` kwarg that the CSA
                    # attention forward swallows via **kwargs without using it, so
                    # this combination is untested anyway. Fail loudly instead.
                    raise ValueError(
                        "train_indexer_only=True is incompatible with "
                        "enable_hy_sparse_attention=True: HySparseTransformerLayer "
                        "does not keep its recompute segment differentiable, so the "
                        "Indexer would silently receive no gradient."
                    )
        elif getattr(self, "train_indexer_only", False):
            raise ValueError(
                "train_indexer_only=True is only supported with "
                "experimental_attention_variant='dsv4_hybrid', got "
                f"{self.experimental_attention_variant!r}."
            )

        # swa_high_precision_norm is only supported for DSv4 models.
        if (
            self.swa_high_precision_norm
            and self.experimental_attention_variant != "dsv4_hybrid"
        ):
            raise ValueError(
                "swa_high_precision_norm=True is only supported when "
                "experimental_attention_variant='dsv4_hybrid'. "
                "High-precision norm mode is only adapted for DSv4 to align "
                "training and inference numerical behavior."
            )

        # Hash-based MoE routing consistency checks.
        if self.moe_n_hash_layers > 0:
            if (
                self.first_k_dense_replace is not None
                and self.first_k_dense_replace > 0
            ):
                raise ValueError(
                    f"first_k_dense_replace ({self.first_k_dense_replace}) and "
                    f"moe_n_hash_layers ({self.moe_n_hash_layers}) are mutually "
                    f"exclusive; the first_k_dense_replace dense layers would "
                    f"suppress hash routing. Set first_k_dense_replace=0 if you "
                    f"want hash routing."
                )
            if self.actual_vocab_size is None:
                raise ValueError(
                    "actual_vocab_size must be set when moe_n_hash_layers > 0; "
                    "it is required to allocate the tid2eid lookup buffer."
                )
            if self.actual_vocab_size <= 0:
                raise ValueError(
                    f"actual_vocab_size must be positive, got "
                    f"{self.actual_vocab_size}."
                )
            if self.moe_n_hash_layers > self.num_hidden_layers:
                raise ValueError(
                    f"moe_n_hash_layers ({self.moe_n_hash_layers}) cannot exceed "
                    f"num_hidden_layers ({self.num_hidden_layers})."
                )
            if self.scoring_func not in ("softmax", "sigmoid", "sqrtsoftplus"):
                raise ValueError(
                    f"Hash routing requires scoring_func in "
                    f"{{'softmax', 'sigmoid', 'sqrtsoftplus'}}, got "
                    f"{self.scoring_func!r}."
                )
            if (
                self.num_experts_per_tok is None
                or self.num_experts_per_tok <= 0
            ):
                raise ValueError(
                    "num_experts_per_tok (top-k) must be a positive integer "
                    "when moe_n_hash_layers > 0."
                )
            if (
                self.n_routed_experts is None
                or self.n_routed_experts < self.num_experts_per_tok
            ):
                raise ValueError(
                    f"n_routed_experts ({self.n_routed_experts}) must be >= "
                    f"num_experts_per_tok ({self.num_experts_per_tok}) "
                    f"when moe_n_hash_layers > 0."
                )

        if self.window_attn_skip_freq is not None:
            if (
                isinstance(self.window_attn_skip_freq, int)
                and self.window_attn_skip_freq <= 0
            ):
                raise ValueError(
                    f"window_attn_skip_freq must be a positive integer when "
                    f"specified as int, but got {self.window_attn_skip_freq}."
                )

        if (
            self.num_nextn_predict_layers > 0
            and self.window_attn_skip_freq is not None
        ):
            if not isinstance(self.window_attn_skip_freq, list):
                raise TypeError(
                    f"window_attn_skip_freq must be a list of length "
                    f"num_hidden_layers + num_nextn_predict_layers "
                    f"({self.num_hidden_layers} + {self.num_nextn_predict_layers} = "
                    f"{self.num_hidden_layers + self.num_nextn_predict_layers}) "
                    f"when num_nextn_predict_layers > 0, "
                    f"but got {type(self.window_attn_skip_freq).__name__} instead."
                )
            if (
                len(self.window_attn_skip_freq)
                != self.num_hidden_layers + self.num_nextn_predict_layers
            ):
                raise ValueError(
                    f"self.window_attn_skip_freq ({len(self.window_attn_skip_freq)}) "
                    f"must equal num_hidden_layers + num_nextn_predict_layers ({self.num_hidden_layers + self.num_nextn_predict_layers})."
                )
            # HySparse: MTP layers must be FULL attention layers, never SWA.
            # An SWA layer consumes shared_kv (compressed KV latent + block
            # indices) produced by an upstream full layer. The MTP boundary
            # (MultiTokenPredictionLayer._proj_and_transformer_layer) rebuilds a
            # fresh input_dict and does NOT forward shared_key/shared_block_indices
            # from the backbone, so an SWA MTP layer would receive shared_kv=[None,
            # None] and crash at shared_key.squeeze(2) in the block-sparse branch.
            # Fail fast here with a clear message instead.
            if self.enable_hy_sparse_attention:
                mtp_window_flags = self.window_attn_skip_freq[
                    self.num_hidden_layers :
                ]
                if any(flag != 0 for flag in mtp_window_flags):
                    raise ValueError(
                        "When enable_hy_sparse_attention is True, the MTP portion "
                        f"of window_attn_skip_freq (indices "
                        f"[{self.num_hidden_layers}:], i.e. {mtp_window_flags}) "
                        "must be all 0 (full attention layers). MTP layers cannot "
                        "be sliding-window (SWA) layers because they do not receive "
                        "the shared KV latent from the backbone across the MTP "
                        "boundary. Set the MTP entries to 0."
                    )

        # HySparse: the block-score (FA4) and block-sparse (DSA) backends only
        # support hy_sparse_block_size == 64. The FA4 block-score op requires
        # 128 % block_B == 0 and the SM100 DSA block-sparse gather requires
        # block_B == 64 (one block == one TopK tile chunk). Other values either
        # silently mis-bucket keys or fail deep in the CUDA kernels, so reject
        # them up front.
        if self.enable_hy_sparse_attention and self.hy_sparse_block_size != 64:
            raise ValueError(
                "hy_sparse_block_size must be 64 when enable_hy_sparse_attention "
                f"is True (got {self.hy_sparse_block_size}). The FA4 block-score "
                "op requires 128 % block_B == 0 and the SM100 DSA block-sparse "
                "gather requires block_B == 64 (TopK tile alignment)."
            )

        if (
            self.num_nextn_predict_layers == 0
            and self.window_attn_skip_freq is not None
        ):
            if (
                isinstance(self.window_attn_skip_freq, list)
                and len(self.window_attn_skip_freq) != self.num_hidden_layers
            ):
                raise ValueError(
                    f"self.window_attn_skip_freq ({len(self.window_attn_skip_freq)}) "
                    f"must equal num_hidden_layers ({self.num_hidden_layers})."
                )

        if not (0.0 <= self.head_wise_swa_ratio <= 1.0):
            raise ValueError(
                f"head_wise_swa_ratio must be between 0.0 and 1.0, "
                f"but got {self.head_wise_swa_ratio}."
            )

        # Multimax validation + grep-friendly confirmation banner.
        # Operators can verify the setting reached the model with:
        #   grep MULTIMAX <train.log>
        import warnings as _warnings

        _multimax = getattr(self, "multimax_modules", None)
        # YAML entry path returns OmegaConf containers (ListConfig), not
        # builtin list. Normalize to a plain Python list before any
        # isinstance(_multimax, list) check; otherwise the recommended
        # `multimax_modules: [lm_head]` form is rejected.
        try:
            from omegaconf import (
                ListConfig as _ListConfig,
                OmegaConf as _OmegaConf,
            )

            if isinstance(_multimax, _ListConfig):
                _multimax = _OmegaConf.to_container(_multimax, resolve=True)
                self.multimax_modules = _multimax
        except ImportError:
            pass
        # Allow yaml/json to leave the field unset, set to ``null``, pass an
        # empty string, or pass an empty list -- all map to the canonical
        # disabled sentinel ``None``.
        if _multimax in ("", []):
            _multimax = None
            self.multimax_modules = None
        # Back-compat: a plain string is treated as a single-element list
        # so older configs (multimax_modules: lm_head) keep working.
        if isinstance(_multimax, str):
            _multimax = [_multimax]
            self.multimax_modules = _multimax
        if _multimax is not None:
            if not isinstance(_multimax, list) or not all(
                isinstance(x, str) for x in _multimax
            ):
                raise ValueError(
                    f"multimax_modules must be None or a list[str], "
                    f"got {_multimax!r}."
                )
            _valid = {"lm_head", "attention"}
            _bad = [x for x in _multimax if x not in _valid]
            if _bad:
                raise ValueError(
                    f"multimax_modules entries must each be one of "
                    f"{sorted(_valid)}, got invalid entries {_bad!r} "
                    f"in {_multimax!r}."
                )
            if "attention" in _multimax:
                _warnings.warn(
                    f"[MULTIMAX-CONFIG] multimax_modules={_multimax}: "
                    "'attention' branch is not implemented yet; only the "
                    "lm_head modulation will take effect."
                )
            _warnings.warn(f"[MULTIMAX-CONFIG] multimax_modules={_multimax}")

        if self.cp_balance_mode not in {
            "dualchunk_allgather",
            "contiguous_allgather",
            "contiguous_a2a",
        }:
            raise ValueError(
                f"cp_balance_mode={self.cp_balance_mode!r} is invalid. "
                "Must be one of {'dualchunk_allgather', 'contiguous_allgather', 'contiguous_a2a'}."
            )

        # only support hybrid_mla_cp_mode == contiguous_a2a if not None
        if self.hybrid_mla_cp_mode is not None:
            if (
                self.hybrid_mla_cp_mode != "contiguous_a2a"
                or not self.cp_balance_mode.startswith("contiguous")
            ):
                raise ValueError(
                    f"hybrid_mla_cp_mode={self.hybrid_mla_cp_mode!r} with "
                    f"cp_balance_mode={self.cp_balance_mode!r} is invalid: "
                    "hybrid_mla_cp_mode must be None or 'contiguous_a2a' "
                    "(other paths are simply not currently supported yet), "
                    "and cp_balance_mode must be in the same token layout."
                )
            # ``csa_compress_ratios`` defaults to None off a dsv4_hybrid, so
            # ``or ()`` keeps the membership test total instead of relying on
            # the left operand short-circuiting it.
            if self.experimental_attention_variant != "dsv4_hybrid" or (
                -2 not in (self.csa_compress_ratios or ())
            ):
                raise ValueError(
                    "hybrid_mla_cp_mode can only be set in dsv4_hybrid with "
                    "-2 in csa_compress_ratios."
                )

        # only support mqa_indexer_cp_mode == dualchunk_p2p if not None
        if self.mqa_indexer_cp_mode is not None:
            if self.mqa_indexer_cp_mode != "dualchunk_p2p":
                raise ValueError(
                    f"mqa_indexer_cp_mode={self.mqa_indexer_cp_mode!r} is "
                    "invalid. Must be None or 'dualchunk_p2p'."
                )
            # The permutation is layer-local: the rows are swapped inside the
            # MQA layer and swapped back before it returns, which is only sound
            # while the *global* layout is the contiguous one the index tables
            # are built against ("build over the global sequence, then take this
            # rank's rows"). A dualchunk global layout would double-permute.
            #
            # Exact value rather than ``startswith("contiguous")``: under a real
            # CP group ``MQALatentAttention.__init__`` accepts only
            # ``contiguous_allgather``, so admitting ``contiguous_a2a`` here
            # would pass config validation and then raise NotImplementedError at
            # module construction -- a contract split between two layers.
            if self.cp_balance_mode != "contiguous_allgather":
                raise ValueError(
                    f"mqa_indexer_cp_mode={self.mqa_indexer_cp_mode!r} needs "
                    "cp_balance_mode='contiguous_allgather' (the only mode the "
                    "latent-MQA layer supports under context parallel), got "
                    f"{self.cp_balance_mode!r}."
                )
            # Same membership test as hybrid_mla_cp_mode above: no latent-MQA
            # layer means no indexer to rebalance.
            if self.experimental_attention_variant != "dsv4_hybrid" or (
                -2 not in (self.csa_compress_ratios or ())
            ):
                raise ValueError(
                    "mqa_indexer_cp_mode can only be set in dsv4_hybrid with "
                    "-2 in csa_compress_ratios."
                )
            # ``MQALatentAttention`` reads the switch in ``_forward_sparse``
            # only. The other two phases reach a different indexer path that
            # does not permute rows -- ``hybrid_mla_attention="mha"`` builds no
            # latent-MQA layer at all, and ``"mqa_dsa"`` with
            # ``dsa_indexer_use_sparse_loss=False`` is the warmup phase, whose
            # KL spans the whole per-document causal set with no top-k anywhere.
            # Accepting those would start successfully and silently deliver none
            # of the rebalance this switch advertises.
            if (
                self.hybrid_mla_attention != "mqa_dsa"
                or not self.dsa_indexer_use_sparse_loss
            ):
                raise ValueError(
                    f"mqa_indexer_cp_mode={self.mqa_indexer_cp_mode!r} only "
                    "takes effect in the sparse training phase, which is "
                    "hybrid_mla_attention='mqa_dsa' together with "
                    "dsa_indexer_use_sparse_loss=True. This config has "
                    f"hybrid_mla_attention={self.hybrid_mla_attention!r} and "
                    "dsa_indexer_use_sparse_loss="
                    f"{self.dsa_indexer_use_sparse_loss!r}, which runs an "
                    "indexer path that scores the full causal set and never "
                    "permutes rows, so the rebalance would be silently inert. "
                    "Drop mqa_indexer_cp_mode, or move to the sparse phase. "
                    "Note that train_indexer_only=True is the warmup phase by "
                    "definition and so cannot carry it either."
                )
            # The two-chunks-per-rank split needs an even per-rank row count.
            # ``TransformerConfig`` does not carry the sequence length, so that
            # is checked at the swap itself (``cp_utils.dualchunk_swap``).

        # separate_mtp_headloss validation.
        if self.separate_mtp_headloss:
            import warnings as _warnings

            # Resolve the effective number of MTP layers, following the same
            # logic used elsewhere in __post_init__ (see the csa branch above).
            mtp_num_layers = self.num_nextn_predict_layers
            mtp_enabled = mtp_num_layers > 0
            pp_enabled = self.pipeline_model_parallel_size > 1

            # 1. separate_mtp_headloss is only meaningful when both MTP and PP
            #    are enabled; otherwise warn and force-disable it.
            if not mtp_enabled or not pp_enabled:
                _warnings.warn(
                    "separate_mtp_headloss=True requires both MTP and pipeline "
                    "parallel to be enabled "
                    f"(mtp_num_layers={mtp_num_layers}, "
                    f"pipeline_model_parallel_size={self.pipeline_model_parallel_size}). "
                    "Forcing separate_mtp_headloss=False."
                )
                self.separate_mtp_headloss = False
            else:
                # Once MTP and PP are both enabled:
                # 2. Layer-count vs pp*vpp guard.
                #    The framework-enforced constraint is DIVISIBILITY: Paddle's
                #    interleave segmentation (SegmentLayers.do_segment) asserts
                #    the number of seg-weight-bearing layers is divisible by
                #    pp*vpp, so an indivisible layout fails at build time.
                #    Verified empirically on 8xH800 (pp=4, vpp=2): a divisible
                #    layout with quotient > 1 (multiple layers per stage) builds
                #    and constructs shared comm fine, so "exactly 1 layer per
                #    stage" is stricter than the framework needs. We keep the
                #    quotient==1 form as a conservative guard because the full
                #    fwd/bwd path for separate_mtp_headloss under pp>1 is not yet
                #    covered by any regression test; see
                #    tests/multi_card_tests/pipeline_parallel/test_separate_mtp_headloss_pp.py
                #    which asserts the build/segmentation/shared-layer-stage
                #    contract.
                pp_degree = self.pipeline_model_parallel_size
                vpp_degree = self.virtual_pipeline_model_parallel_size
                # When vpp is not enabled, vpp_degree may be None or a
                # sentinel like -1. Normalize it to 1 before computing
                # pp_degree*vpp_degree, otherwise the product would be wrong
                # (e.g. negative).
                if vpp_degree is None or vpp_degree <= 1:
                    print(
                        "[separate_mtp_headloss] vpp is not enabled "
                        f"(virtual_pipeline_model_parallel_size={vpp_degree}); "
                        "using vpp_degree=1 for the pp_degree*vpp_degree check."
                    )
                    vpp_degree = 1

                num_empty_layers = self.num_empty_layers_add_in_head + max(
                    0, self.num_empty_layers_add_in_tail - 1
                )
                total_layers = (
                    self.num_hidden_layers + mtp_num_layers + num_empty_layers
                )
                denom = pp_degree * vpp_degree
                if total_layers % denom != 0 or total_layers // denom != 1:
                    _warnings.warn(
                        "separate_mtp_headloss=True requires "
                        "(num_hidden_layers + num_mtp_layers + num_empty_layers) "
                        f"({self.num_hidden_layers} + {mtp_num_layers} + "
                        f"{num_empty_layers} = {total_layers}) to be divisible "
                        f"by pp_degree*vpp_degree ({pp_degree}*{vpp_degree} = "
                        f"{denom}) and the quotient to equal 1 (exactly one "
                        f"layer per pp*vpp stage), but got {total_layers} / "
                        f"{denom} = {total_layers / denom}. "
                        "Forcing separate_mtp_headloss=False."
                    )
                    self.separate_mtp_headloss = False

                # 3. separate_mtp_headloss reserves tail EmptyLayer slots for the
                #    separated MTP LMHead/Loss and main LMHead, so it needs at
                #    least 3 tail empty layers.
                if self.num_empty_layers_add_in_tail < 3:
                    _warnings.warn(
                        "separate_mtp_headloss=True requires "
                        "num_empty_layers_add_in_tail >= 3 "
                        f"(got {self.num_empty_layers_add_in_tail}). "
                        "Forcing separate_mtp_headloss=False."
                    )
                    self.separate_mtp_headloss = False
