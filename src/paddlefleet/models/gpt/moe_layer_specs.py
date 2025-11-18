# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

from paddlefleet.models.backends import BackendSpecProvider, LocalSpecProvider
from paddlefleet.transformer.mlp import MLPSublayersSpec
from paddlefleet.transformer.moe.moe_layer import MoELayer, MoESublayers
from paddlefleet.transformer.spec_utils import LayerSpec


def get_moe_layer_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
) -> LayerSpec:
    """Helper function to get layer spec for MoE"""
    backend = LocalSpecProvider()
    return get_moe_layer_spec_for_backend(
        backend=backend,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
    )


def get_moe_layer_spec_for_backend(
    backend: BackendSpecProvider,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
) -> LayerSpec:
    """Helper function to get layer spec for MoE"""
    assert num_experts is not None

    linear_fc1 = backend.column_parallel_linear()
    linear_fc2 = backend.row_parallel_linear()
    activation_func = backend.activation_func()

    mlp_spec = MLPSublayersSpec(
        linear_fc1=linear_fc1,
        linear_fc2=linear_fc2,
        activation_func=activation_func,
    )

    moe_layer_spec = LayerSpec(
        layer=MoELayer, params={"sublayers": MoESublayers(mlp_spec=mlp_spec)}
    )
    return moe_layer_spec
