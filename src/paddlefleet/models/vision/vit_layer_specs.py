# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from paddlefleet.fusions.fused_bias_dropout import get_bias_dropout_add
from paddlefleet.fusions.fused_layer_norm import FusedLayerNorm
from paddlefleet.spec_utils import LayerSpec
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
from paddlefleet.transformer.mlp import MLP, MLPSublayersSpec
from paddlefleet.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSublayersSpec,
)

LNImpl = FusedLayerNorm


def get_vit_layer_with_local_spec() -> LayerSpec:
    """
    Returns ViT layer spec with Mcore local layers
    """
    mlp = _get_mlp_module_spec(use_te=False)
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


# Helper function to get module spec for MLP/MoE
def _get_mlp_module_spec(use_te: bool = True) -> LayerSpec:
    # Dense MLP w/ or w/o TE modules.
    return LayerSpec(
        module=MLP,
        submodules=MLPSublayersSpec(
            linear_fc1=ColumnParallelLinear,
            linear_fc2=RowParallelLinear,
        ),
    )
