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

import paddle
import pytest
from megatron.training.initialize import _set_random_seed
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.test_utilities import Utils

from paddlefleet.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from paddlefleet.transformer.moe.moe_layer import MoELayer
from paddlefleet.transformer.transformer_block import TransformerBlock
from paddlefleet.transformer.transformer_config import TransformerConfig


class TestMoELayerInit:
    def setup_method(self, method):
        pass

    @pytest.mark.parametrize("ep_communication_type", ["deepep", "alltoall"])
    @pytest.mark.parametrize("moe_num_experts", [1, 2])
    @pytest.mark.parametrize(
        "grouped_gemm", [False]
    )  # TODO: support grouped_gemm
    def test_te_moe_layer(
        self, moe_num_experts, ep_communication_type, grouped_gemm
    ):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        self.transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=12,
            num_attention_heads=4,
            moe_num_experts=moe_num_experts,
            use_cpu_initialization=True,
            ep_communication_type=ep_communication_type,
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=grouped_gemm,
            moe_ffn_hidden_size=128,
            add_bias_linear=False,
        )
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=moe_num_experts, moe_grouped_gemm=grouped_gemm
        )
        moe_layer = MoELayer(
            self.transformer_config,
            transformer_layer_spec.submodules.mlp.submodules,
        )
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("ep_communication_type", ["deepep", "alltoall"])
    @pytest.mark.parametrize("moe_num_experts", [1, 2])
    @pytest.mark.parametrize(
        "grouped_gemm", [False]
    )  # TODO: support grouped_gemm
    def test_legacy_moe_layer(
        self, moe_num_experts, ep_communication_type, grouped_gemm
    ):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        moe_num_experts = 4
        self.transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=12,
            num_attention_heads=4,
            moe_num_experts=moe_num_experts,
            use_cpu_initialization=True,
            ep_communication_type=ep_communication_type,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=grouped_gemm,
            add_bias_linear=False,
        )
        transformer_layer_spec = get_gpt_layer_local_spec(
            num_experts=moe_num_experts, moe_grouped_gemm=grouped_gemm
        )
        moe_layer = MoELayer(
            self.transformer_config,
            transformer_layer_spec.submodules.mlp.submodules,
        )
        Utils.destroy_model_parallel()

    @pytest.mark.skip(
        "Late init of parallel_state was broken after parallel states refactor MR2988."
    )
    @pytest.mark.parametrize("ep_communication_type", ["deepep", "alltoall"])
    @pytest.mark.parametrize(
        "grouped_gemm", [False]
    )  # TODO: support grouped_gemm
    @pytest.mark.parametrize(
        "tp_size,ep_size", [(1, 1)]
    )  # TODO: support tp_size, ep_size > 1
    def test_moe_with_late_initialize(
        self, ep_communication_type, grouped_gemm, tp_size, ep_size
    ):
        moe_num_experts = 4
        hidden_size = 12
        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=4,
            moe_num_experts=moe_num_experts,
            use_cpu_initialization=True,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            add_bias_linear=False,
            moe_grouped_gemm=grouped_gemm,
            ep_communication_type=ep_communication_type,
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            sequence_parallel=tp_size > 1,
            bf16=True,
            params_dtype=paddle.bfloat16,
        )
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=moe_num_experts, moe_grouped_gemm=grouped_gemm
        )

        # Fake initialization as NeMo does
        Utils.fake_initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
        )
        moe_layer = MoELayer(
            transformer_config, transformer_layer_spec.submodules.mlp.submodules
        ).cuda()

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        input_data = paddle.randn(16, 4, hidden_size, dtype=paddle.bfloat16)
        output = moe_layer(input_data)

        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


class TestInterleaveTransformerBlock:
    @pytest.mark.parametrize(
        "moe_layer_freq", [2, eval("[0,1,1,1]"), eval("[0]*2+[1]*2")]
    )
    def test_interleave_transformer_block(self, moe_layer_freq):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=4,
            moe_layer_freq=moe_layer_freq,
            moe_ffn_hidden_size=256,
            use_cpu_initialization=True,
            moe_num_experts=2,
            add_bias_linear=False,
        )
        self.parallel_transformer_block = TransformerBlock(
            self.transformer_config,
            get_gpt_decoder_block_spec(self.transformer_config, False),
        )

        # Check if the moe layer is interleaved correctly
        if isinstance(self.transformer_config.moe_layer_freq, int):
            moe_layer_pattern = [
                1 if (i % self.transformer_config.moe_layer_freq == 0) else 0
                for i in range(self.transformer_config.num_layers)
            ]
        else:
            moe_layer_pattern = self.transformer_config.moe_layer_freq

        for i, layer in enumerate(self.parallel_transformer_block.layers):
            is_moe_layer = isinstance(layer.mlp, MoELayer)
            assert is_moe_layer == moe_layer_pattern[i]

        # Test forward pass
        parallel_transformer_block = self.parallel_transformer_block
        config: TransformerConfig = parallel_transformer_block.config
        sequence_length = 32
        micro_batch_size = 2
        parallel_transformer_block.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = paddle.ones(
            (sequence_length, micro_batch_size, config.hidden_size)
        )
        hidden_states = hidden_states.cuda()

        attention_mask = paddle.ones(
            (1, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()
        hidden_states = parallel_transformer_block(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        assert hidden_states.shape[0] == sequence_length
        assert hidden_states.shape[1] == micro_batch_size
        assert hidden_states.shape[2] == config.hidden_size

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
