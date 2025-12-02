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

# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from paddlefleet.fusions.fused_bias_dropout import get_bias_dropout_add
from paddlefleet.fusions.fused_layer_norm import FusedLayerNorm
from paddlefleet.models.gpt.gpt_layer_specs import get_mlp_layer_spec
from paddlefleet.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from paddlefleet.transformer.attention import (
    SelfAttention,
    SelfAttentionSublayersSpec,
)
from paddlefleet.transformer.dot_product_attention import DotProductAttention
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.spec_utils import LayerSpec
from paddlefleet.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSublayersSpec,
)

LNImpl = FusedLayerNorm


def decoder_model_with_local_default_spec(
    num_experts: int | None = None,
    moe_grouped_gemm: bool = False,
    qk_layernorm: bool = False,
) -> LayerSpec:
    """LLava decoder local spec."""
    mlp = get_mlp_layer_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    return LayerSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSublayersSpec(
            input_layernorm=LNImpl,
            self_attention=LayerSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSublayersSpec(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=LNImpl,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )
