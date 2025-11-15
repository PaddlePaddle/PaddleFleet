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

import paddle

from paddlefleet.transformer.mlp import MLP, MLPSublayersSpec

# (TODO): need add tp case
# from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
# from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
# from tests.unit_tests.test_utilities import Utils
from paddlefleet.transformer.spec_utils import LayerSpec
from paddlefleet.transformer.transformer_config import TransformerConfig


class Linear(paddle.nn.Linear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: None,
        init_method: None,
        bias: bool = True,
        gather_output: bool = False,
        input_is_parallel: bool = False,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        skip_bias_add: bool = False,
        skip_weight_param_allocation: bool = False,
        embedding_activation_buffer=None,
        grad_output_buffer=None,
        is_expert: bool = False,
        disable_grad_reduce: bool = False,
        tp_group: None,
    ):
        super().__init__(
            in_features=input_size, out_features=output_size, bias_attr=bias
        )

    def forward(self, x):
        """Forward."""
        out = super().forward(x)
        return out, None


class TestParallelMLP:
    def setup_method(self, method):
        # Utils.initialize_model_parallel(1, 1)
        # model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            ffn_hidden_size=48,
            num_attention_heads=4,
        )
        # (TODO): need replace with gpt_model.mlp later,now temp use a simple mlp
        # mlp_spec =  get_gpt_layer_local_spec().submodules.mlp.submodules
        mlp_spec = LayerSpec(
            MLP,
            sublayers_spec=MLPSublayersSpec(
                linear_fc1=Linear, linear_fc2=Linear
            ),
        )
        self.mlp = MLP(transformer_config, mlp_spec.sublayers_spec)

    def teardown_method(self, method):
        # Utils.destroy_model_parallel()
        pass

    def test_constructor(self):
        assert isinstance(self.mlp, MLP)

        num_weights = sum([p.numel() for p in self.mlp.parameters()])
        assert num_weights == 1212

    """
    def test_cpu_forward(self, mlp):
        # [sequence length, micro batch size, hidden size]
        hidden_states = paddle.ones((32, 2, mlp.config.hidden_size))
        output, output_bias = mlp(hidden_states)
        assert output.shape[0] == 32
        assert output.shape[1] == 2
        assert output.shape[2] == mlp.config.hidden_size
        assert output_bias.shape[0] == mlp.config.hidden_size
        assert output.dtype == paddle.float32
    """

    def test_gpu_forward(self):
        mlp = self.mlp
        # [sequence length, batch size, hidden size]
        hidden_states = paddle.ones((32, 12, mlp.config.hidden_size))
        hidden_states = hidden_states.cuda()
        output, output_bias = mlp(hidden_states)
        print(output.shape)
        assert output.shape[0] == 32
        assert output.shape[1] == 12
        assert output.shape[2] == mlp.config.hidden_size
        assert output.dtype == paddle.float32
