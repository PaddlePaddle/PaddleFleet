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
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

from fleet.core.fusions.fused_bias_dropout import get_bias_dropout_add
from fleet.core.models.backends import BackendSpecProvider, LocalSpecProvider
from fleet.core.models.gpt.moe_layer_specs import get_moe_layer_spec_for_backend
from fleet.core.transformer.attention import (
    SelfAttention,
    SelfAttentionSublayers,
)
from fleet.core.transformer.enums import AttnMaskType, LayerType
from fleet.core.transformer.identity_op import IdentityOp
from fleet.core.transformer.mlp import MLP, MLPSublayers
from fleet.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlockSublayers,
    get_mtp_layer_offset,
    get_mtp_layer_spec_for_backend,
    get_mtp_num_layers_to_build,
)
from fleet.core.transformer.norm import L2Norm
from fleet.core.transformer.spec_utils import LayerSpec
from fleet.core.transformer.transformer_block import (
    TransformerBlockSublayers,
    get_num_layers_to_build,
)
from fleet.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSublayers,
    get_transformer_layer_offset,
)

if TYPE_CHECKING:
    from fleet.core.transformer.transformer_config import TransformerConfig


LNImpl = None


def get_gpt_layer_local_spec(
    num_experts: int | None = None,
    moe_grouped_gemm: bool | None = False,
    qk_layernorm: bool | None = False,
    multi_latent_attention: bool | None = False,
    normalization: str | None = None,
    qk_l2_norm: bool | None = False,
) -> LayerSpec:
    """Use this spec for an implementation using only layers in Fleet-Core.


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        fp8 (str, optional): Deprecated. For temporary Nemo compatibility.
        qk_l2_norm (bool, optional): To use l2 norm for queries/keys. Defaults to False.

    Returns:
        LayerSpec: Layer specification with Fleet-Core layers
    """

    backend = LocalSpecProvider()
    # Adjust for RMS norm.
    if normalization == "RMSNorm":
        layer_norm = backend.layer_norm(rms_norm=True, for_qk=False)
        qk_norm = backend.layer_norm(rms_norm=True, for_qk=True)
    else:
        layer_norm = backend.layer_norm(rms_norm=False, for_qk=False)
        qk_norm = backend.layer_norm(rms_norm=False, for_qk=True)

    mlp = get_mlp_layer_spec_for_backend(
        backend=backend,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
    )

    return LayerSpec(
        layer=TransformerLayer,
        sublayers=TransformerLayerSublayers(
            input_layernorm=layer_norm,
            self_attention=LayerSpec(
                layer=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                sublayers=SelfAttentionSublayers(
                    linear_qkv=backend.column_parallel_linear(),
                    core_attention=backend.core_attention(),
                    linear_proj=backend.row_parallel_linear(),
                    q_layernorm=(
                        L2Norm
                        if qk_l2_norm
                        else (qk_norm if qk_layernorm else IdentityOp)
                    ),
                    k_layernorm=(
                        L2Norm
                        if qk_l2_norm
                        else (qk_norm if qk_layernorm else IdentityOp)
                    ),
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=layer_norm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                "input_layernorm.": "self_attention.linear_qkv.layer_norm_",
                "pre_mlp_layernorm.": "mlp.linear_fc1.layer_norm_",
            },
        ),
    )


def get_mlp_layer_spec_for_backend(
    backend: BackendSpecProvider,
    num_experts: int | None = None,
    moe_grouped_gemm: bool | None = False,
) -> LayerSpec:
    """Helper function to get layer spec for MLP/MoE"""

    linear_fc2 = backend.row_parallel_linear()
    activation_func = None

    if num_experts is None:
        # Dense MLP w/ or w/o TE layers.
        layer = MLP
        if backend.fuse_layernorm_and_linear():
            linear_fc1 = backend.column_parallel_layer_norm_linear()
            assert linear_fc1 is not None
        else:
            linear_fc1 = backend.column_parallel_linear()
        return LayerSpec(
            layer=layer,
            sublayers=MLPSublayers(
                linear_fc1=linear_fc1,
                linear_fc2=linear_fc2,
                activation_func=activation_func,
            ),
        )
    else:
        # Mixture of experts with layers in fleet core.
        return get_moe_layer_spec_for_backend(
            backend=backend,
            num_experts=num_experts,
            moe_grouped_gemm=moe_grouped_gemm,
        )


def get_gpt_decoder_block_spec(
    config: TransformerConfig,
    normalization: str | None = None,
    qk_l2_norm: bool | None = False,
    vp_stage: int | None = None,
    pp_rank: int | None = None,
) -> TransformerBlockSublayers:
    """GPT block spec."""
    layer_norm_impl = LNImpl
    dense_layer_spec = get_gpt_layer_local_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        normalization=normalization,
        qk_l2_norm=qk_l2_norm,
    )
    moe_layer_spec = get_gpt_layer_local_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        normalization=normalization,
        qk_l2_norm=qk_l2_norm,
    )

    # Parse config.moe_layer_freq to determine the pattern of expert/dense layers.
    # 0 stands for dense layers, 1 stands for expert layers.
    # For integer N: Creates a pattern with one expert layer every N layers.
    # For string pattern: Evaluates the str directly (e.g. "[1,0,1]" for alternating expert/dense).
    if isinstance(config.moe_layer_freq, int):
        moe_layer_pattern = [
            1 if (i % config.moe_layer_freq == 0) else 0
            for i in range(config.num_layers)
        ]
    elif isinstance(config.moe_layer_freq, list):
        moe_layer_pattern = config.moe_layer_freq
        assert len(moe_layer_pattern) == config.num_layers, (
            f"Invalid length of moe_layer_pattern: {len(moe_layer_pattern)}, "
            f"expected {config.num_layers}, "
            f"current moe layer pattern: {config.moe_layer_freq}"
        )
    else:
        raise ValueError(
            f"Invalid moe_layer_freq: {type(config.moe_layer_freq)}, {config.moe_layer_freq}"
        )

    # Create the layer specs for the model.
    layer_specs = []
    for layer_number in range(config.num_layers):
        if moe_layer_pattern[layer_number] == 1:
            layer_specs.append(moe_layer_spec)
        elif moe_layer_pattern[layer_number] == 0:
            layer_specs.append(dense_layer_spec)
        else:
            raise ValueError(f"Invalid layer pattern: {moe_layer_pattern}")

    # Slice the layer specs to only include the layers that are built in this pipeline stage.
    # Note: MCore layer_number starts at 1
    num_layers_to_build = get_num_layers_to_build(
        config, vp_stage=vp_stage, pp_rank=pp_rank
    )

    if config.pipeline_model_parallel_layout is not None:
        local_layer_specs = [
            layer_specs[layer_id]
            for layer_id in config.pipeline_model_parallel_layout.get_layer_id_list(
                layer_type=LayerType.decoder, vp_stage=vp_stage, pp_rank=pp_rank
            )
        ]
    else:
        offset = get_transformer_layer_offset(
            config, vp_stage=vp_stage, pp_rank=pp_rank
        )
        local_layer_specs = layer_specs[offset : offset + num_layers_to_build]

    # Block spec.
    block_spec = TransformerBlockSublayers(
        layer_specs=local_layer_specs, layer_norm=layer_norm_impl
    )

    return block_spec


def get_gpt_mtp_block_spec(
    config: TransformerConfig,
    spec: TransformerBlockSublayers | LayerSpec,
    vp_stage: int | None = None,
    pp_rank: int | None = None,
) -> MultiTokenPredictionBlockSublayers:
    """GPT Multi-Token Prediction (MTP) block spec."""
    backend = LocalSpecProvider()
    return get_gpt_mtp_block_spec_for_backend(
        config=config,
        spec=spec,
        backend=backend,
        vp_stage=vp_stage,
        pp_rank=pp_rank,
    )


def get_gpt_mtp_block_spec_for_backend(
    config: TransformerConfig,
    spec: TransformerBlockSublayers | LayerSpec,
    backend: BackendSpecProvider,
    vp_stage: int | None = None,
    pp_rank: int | None = None,
) -> MultiTokenPredictionBlockSublayers:
    """GPT Multi-Token Prediction (MTP) block spec."""
    num_layers_to_build = get_mtp_num_layers_to_build(
        config, vp_stage=vp_stage, pp_rank=pp_rank
    )
    if num_layers_to_build == 0:
        return None

    if isinstance(spec, TransformerBlockSublayers):
        # get the spec for the last layer of decoder block
        transformer_layer_spec = spec.layer_specs[-1]
    elif isinstance(spec, LayerSpec) and spec.layer == TransformerLayer:
        transformer_layer_spec = spec
    else:
        raise ValueError(f"Invalid spec: {spec}")

    mtp_layer_spec = get_mtp_layer_spec_for_backend(
        transformer_layer_spec=transformer_layer_spec, backend=backend
    )
    mtp_num_layers = config.mtp_num_layers if config.mtp_num_layers else 0
    mtp_layer_specs = [mtp_layer_spec] * mtp_num_layers

    offset = get_mtp_layer_offset(config)
    # split the mtp layer specs to only include the layers that are built in this pipeline stage.
    mtp_layer_specs = mtp_layer_specs[offset : offset + num_layers_to_build]
    if len(mtp_layer_specs) > 0:
        assert len(mtp_layer_specs) == config.mtp_num_layers, (
            +"currently all of the mtp layers must stage in the same pipeline stage."
        )
        mtp_block_spec = MultiTokenPredictionBlockSublayers(
            layer_specs=mtp_layer_specs
        )
    else:
        mtp_block_spec = None

    return mtp_block_spec
