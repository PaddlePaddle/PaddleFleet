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

import numpy as np
import paddle

from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddlefleet.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from paddlefleet.transformer.mlp import MLPSublayersSpec
from paddlefleet.transformer.moe.moe_layer import MoESublayers
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import get_tensor_model_parallel_group_if_none
from tests.multi_card_tests.tensor_parallel.test_utilities import Utils


def _build_moe_config():
    # TP=4 config. Weights are overwritten by hand in the numerical test, so
    # perform_initialization is disabled for determinism; params stay float32
    # (bf16=False) to keep the row-parallel reduction bit-exact.
    return TransformerConfig(
        hidden_size=16,
        num_attention_heads=4,
        n_routed_experts=4,
        num_experts_per_tok=2,
        tensor_model_parallel_size=4,
        expert_model_parallel_size=1,
        sequence_parallel=False,
        use_cpu_initialization=False,
        perform_initialization=False,
        bf16=False,
        params_dtype=paddle.float32,
        moe_intermediate_size=24,
        moe_deep_gemm=False,
        gated_linear_unit=True,
        n_shared_experts=0,
    )


def test_moe_spec_selects_parallel_expert_projections():
    # get_gpt_layer_local_spec routes the transformer-layer MLP through the MoE
    # path exactly when num_experts is truthy. Verify that branch decision by
    # its observable structure, not a weak "is not None" check: the MoE spec
    # carries a MoESublayers in extra_kwargs whose expert projections are the
    # tensor-parallel Column/Row linears, while the dense spec instead exposes a
    # plain MLPSublayersSpec and no such MoESublayers.
    config = _build_moe_config()

    moe_mlp = get_gpt_layer_local_spec(
        config, num_experts=4, moe_expert_fusion=False
    ).sublayers_spec.mlp
    sublayers = moe_mlp.extra_kwargs["sublayers"]
    assert isinstance(sublayers, MoESublayers)
    mlp_spec = sublayers.mlp_spec
    assert mlp_spec.up_gate_proj is ColumnParallelLinear
    assert mlp_spec.down_proj is RowParallelLinear

    dense_mlp = get_gpt_layer_local_spec(
        config, num_experts=None
    ).sublayers_spec.mlp
    assert isinstance(dense_mlp.sublayers_spec, MLPSublayersSpec)
    assert dense_mlp.sublayers_spec.up_gate_proj is ColumnParallelLinear
    assert dense_mlp.sublayers_spec.down_proj is RowParallelLinear
    # The discriminator between the two branches: only the MoE spec wraps the
    # projections in a MoESublayers extra kwarg.
    assert "sublayers" not in getattr(dense_mlp, "extra_kwargs", {})


def test_moe_expert_down_projection_all_reduces_across_tp():
    # The MoE spec selects RowParallelLinear for the expert down-projection.
    # Under TP=4 that projection scatters the contracted dimension across the
    # four tensor-parallel ranks and all-reduce-sums the partial products, so
    # the reduced result on every rank equals the full (unsharded) matmul.
    # Each rank owns a distinct slice of a hand-built weight and the expected
    # value is computed independently with NumPy; a wrong reduction, a dropped
    # rank, or a purely local (single-card) compute changes the exact output.
    config = _build_moe_config()

    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    world_size = tp_group.world_size
    rank = tp_group.rank
    assert world_size == 4, (
        f"this MoE down-projection reduction test needs TP=4, got {world_size}"
    )

    down_cls = (
        get_gpt_layer_local_spec(config, num_experts=4)
        .sublayers_spec.mlp.extra_kwargs["sublayers"]
        .mlp_spec.down_proj
    )
    assert down_cls is RowParallelLinear

    input_size, output_size = 8, 3  # input_size divisible by TP=4
    down = down_cls(
        input_size,
        output_size,
        config=config,
        init_method=config.init_method,
        bias=False,
        input_is_parallel=False,
        skip_bias_add=False,
    )

    # Full [input_size, output_size] weight with distinct entries. Rank r owns
    # contiguous rows [r*ip : (r+1)*ip]; the row-parallel forward scatters the
    # matching input columns to rank r, so summing over ranks reconstructs the
    # complete matmul.
    ip = input_size // world_size  # 2 rows per rank
    w_full = np.arange(
        1, input_size * output_size + 1, dtype=np.float32
    ).reshape(input_size, output_size)
    local_w = w_full[rank * ip : (rank + 1) * ip, :]
    down.weight.set_value(
        paddle.to_tensor(local_w, dtype="float32", place=down.weight.place)
    )

    x_np = np.arange(1, input_size + 1, dtype=np.float32).reshape(1, input_size)
    x = paddle.to_tensor(x_np, dtype="float32").cuda()

    with paddle.no_grad():
        output, output_bias = down(x)
    assert output_bias is None

    # Independent reference: the full row-parallel identity is a single dense
    # matmul over the whole weight, evaluated with NumPy (not the layer).
    expected_full = x_np @ w_full
    assert list(output.shape) == [1, output_size]
    np.testing.assert_allclose(
        output.numpy(), expected_full, rtol=1e-6, atol=1e-5
    )

    # Prove the cross-rank all-reduce actually ran: this rank's local-only
    # partial product (what a single-card computation would yield) is distinct
    # from the reduced output, so the summed result cannot be faked locally.
    local_partial = x_np[:, rank * ip : (rank + 1) * ip] @ local_w
    assert not np.allclose(output.numpy(), local_partial, rtol=1e-3, atol=1e-3)

    # The comparison is magnitude-sensitive: a scaled reference (a wrong
    # reduction coefficient) must be rejected, not absorbed by tolerance.
    assert np.linalg.norm(expected_full) > 0
    for bad in (0.5 * expected_full, 2.0 * expected_full):
        rejected = False
        try:
            np.testing.assert_allclose(bad, expected_full, rtol=1e-6, atol=1e-5)
        except AssertionError:
            rejected = True
        assert rejected


if __name__ == "__main__":
    Utils.initialize_model_parallel(4, 1)
    test_moe_spec_selects_parallel_expert_projections()
    test_moe_expert_down_projection_all_reduces_across_tp()
