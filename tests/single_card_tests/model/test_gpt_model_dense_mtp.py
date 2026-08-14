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


import functools
import random
import unittest
import warnings

import numpy as np
import paddle
import paddlefleet_ops
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import NoPipelineParallel

import paddlefleet.parallel_state as ps

paddlefleet_ops.is_sonic_moe_available = lambda: False

from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig
from paddlefleet.transformer.mlp import MLP
from paddlefleet.transformer.moe.moe_layer import MoELayer
from paddlefleet.transformer.multi_token_prediction import (
    MultiTokenPredictionLayer,
)
from paddlefleet.transformer.transformer_layer import TransformerLayer


class TestDenseMTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed = 46
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
            "sharding_degree": 1,
            "sep_degree": 1,
            "cp_degree": 1,
            "ep_degree": 1,
            "moe_sharding_degree": 1,
            "order": [
                "sharding",
                "moe_sharding",
                "pp",
                "sep",
                "cp",
                "dp",
                "ep",
                "mp",
            ],
        }
        fleet.init(is_collective=True, strategy=strategy)
        hcg = fleet.get_hybrid_communicate_group()
        ps.initialize_model_parallel(hcg)
        cls.strategy = strategy

        # Config with MoE + MTP + use_dense_mtp=True
        cls.config_dense_mtp = GPTConfig(
            num_hidden_layers=2,
            hidden_size=512,
            vocab_size=100,
            max_sequence_length=64,
            num_attention_heads=4,
            moe_expert_fusion=False,
            intermediate_size=1024,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            n_routed_experts=8,
            moe_intermediate_size=1024,
            moe_token_dispatcher_type="alltoall",
            n_shared_experts=1,
            use_bias=False,
            rotary_percent=1.0,
            rotary_base=10000,
            rope_scaling=1.0,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            tie_word_embeddings=True,
            use_qk_norm=True,
            num_nextn_predict_layers=1,
            use_dense_mtp=True,
        )
        cls.model_dense_mtp = gpt_builder(cls.config_dense_mtp, num_stages=1)

        # Config with MoE + MTP + use_dense_mtp=False (default)
        cls.config_moe_mtp = GPTConfig(
            num_hidden_layers=2,
            hidden_size=512,
            vocab_size=100,
            max_sequence_length=64,
            num_attention_heads=4,
            moe_expert_fusion=False,
            intermediate_size=1024,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            n_routed_experts=8,
            moe_intermediate_size=1024,
            moe_token_dispatcher_type="alltoall",
            n_shared_experts=1,
            use_bias=False,
            rotary_percent=1.0,
            rotary_base=10000,
            rope_scaling=1.0,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            tie_word_embeddings=True,
            use_qk_norm=True,
            num_nextn_predict_layers=1,
            use_dense_mtp=False,
        )
        cls.model_moe_mtp = gpt_builder(cls.config_moe_mtp, num_stages=1)

    def _find_mtp_layer(self, model):
        for layer in model.run_function:
            if isinstance(layer, MultiTokenPredictionLayer):
                return layer
        return None

    def _find_decoder_layers(self, model):
        layers = []
        for layer in model.run_function:
            if isinstance(layer, TransformerLayer):
                layers.append(layer)
        return layers

    def test_mtp_layer_uses_dense_mlp(self):
        """When use_dense_mtp=True, MTP layer's internal transformer should use dense MLP."""
        mtp_layer = self._find_mtp_layer(self.model_dense_mtp)
        assert mtp_layer is not None, (
            "Model should contain a MultiTokenPredictionLayer"
        )
        inner_mlp = mtp_layer.transformer_layer.mlp
        assert isinstance(inner_mlp, MLP), (
            f"MTP layer's MLP should be dense MLP, got {type(inner_mlp).__name__}"
        )
        assert not isinstance(inner_mlp, MoELayer), (
            "MTP layer's MLP should NOT be MoELayer when use_dense_mtp=True"
        )

    def test_decoder_layers_still_use_moe(self):
        """Decoder layers should still use MoE even when use_dense_mtp=True."""
        decoder_layers = self._find_decoder_layers(self.model_dense_mtp)
        assert len(decoder_layers) > 0, "Model should have decoder layers"
        for dl in decoder_layers:
            assert isinstance(dl.mlp, MoELayer), (
                f"Decoder layer's MLP should be MoELayer, got {type(dl.mlp).__name__}"
            )

    def test_forward_backward_with_dense_mtp(self):
        """Forward/backward pass should work correctly with use_dense_mtp=True."""
        seed = 46
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

        sequence_length = self.config_dense_mtp.max_sequence_length
        micro_batch_size = 1

        data = list(range(sequence_length))
        input_ids = paddle.to_tensor(data, dtype=paddle.int64).repeat(
            (micro_batch_size, 1)
        )
        position_ids = paddle.to_tensor(data, dtype=paddle.int64).repeat(
            (micro_batch_size, 1)
        )
        labels = paddle.to_tensor(
            list(range(1, sequence_length + 1)), dtype=paddle.int64
        ).repeat((micro_batch_size, 1))

        gpt_pipe_model = NoPipelineParallel(self.model_dense_mtp, self.strategy)
        data = (
            {
                "input_ids": [input_ids],
                "position_ids": [position_ids],
            },
            [labels],
        )
        loss = gpt_pipe_model.forward_backward_pipeline(data)

        assert loss is not None, "Loss should not be None"
        assert not paddle.isnan(loss).any(), "Loss should not contain NaN"
        assert not paddle.isinf(loss).any(), "Loss should not contain Inf"
        print(f"Loss with dense MTP: {loss.item()}")

    def test_default_mtp_uses_moe(self):
        """When use_dense_mtp=False (default), MTP layer should use MoE like the decoder."""
        mtp_layer = self._find_mtp_layer(self.model_moe_mtp)
        assert mtp_layer is not None, (
            "Model should contain a MultiTokenPredictionLayer"
        )
        inner_mlp = mtp_layer.transformer_layer.mlp
        assert isinstance(inner_mlp, MoELayer), (
            f"MTP layer's MLP should be MoELayer when use_dense_mtp=False, "
            f"got {type(inner_mlp).__name__}"
        )

    def test_mtp_reuse_last_layer_aliases_params(self):
        """When mtp_reuse_last_layer=True, MTP transformer params should reuse last backbone params."""
        config = GPTConfig(
            num_hidden_layers=2,
            hidden_size=512,
            vocab_size=100,
            max_sequence_length=64,
            num_attention_heads=4,
            intermediate_size=1024,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            use_bias=False,
            rotary_percent=1.0,
            rotary_base=10000,
            rope_scaling=1.0,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            tie_word_embeddings=True,
            use_qk_norm=True,
            num_nextn_predict_layers=1,
            use_dense_mtp=True,
            mtp_reuse_last_layer=True,
        )
        model = gpt_builder(config, num_stages=1)

        decoder_layers = self._find_decoder_layers(model)
        mtp_layer = self._find_mtp_layer(model)
        assert decoder_layers, "Model should have decoder layers"
        assert mtp_layer is not None, (
            "Model should contain a MultiTokenPredictionLayer"
        )

        backbone_params = dict(decoder_layers[-1].named_parameters())
        mtp_params = dict(mtp_layer.transformer_layer.named_parameters())
        not_shared = []
        for param_name, mtp_param in mtp_params.items():
            backbone_param = backbone_params.get(param_name)
            if backbone_param is not mtp_param:
                not_shared.append(param_name)

        assert len(mtp_params) > 0, "MTP transformer should have parameters"
        assert not not_shared, (
            f"MTP params should reuse last backbone params, not_shared={not_shared[:5]}"
        )

    def test_mtp_reuse_last_layer_overrides_use_dense_mtp(self):
        """mtp_reuse_last_layer=True should force use_dense_mtp=False so the MTP
        layer always mirrors the backbone-last layer (MoE in this config),
        regardless of the user-supplied use_dense_mtp value."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = GPTConfig(
                num_hidden_layers=2,
                hidden_size=512,
                vocab_size=100,
                max_sequence_length=64,
                num_attention_heads=4,
                moe_expert_fusion=False,
                intermediate_size=1024,
                normalization="RMSNorm",
                hidden_dropout_prob=0.0,
                attention_dropout=0.0,
                n_routed_experts=8,
                moe_intermediate_size=1024,
                moe_token_dispatcher_type="alltoall",
                n_shared_experts=1,
                use_bias=False,
                rotary_percent=1.0,
                rotary_base=10000,
                rope_scaling=1.0,
                init_method=functools.partial(
                    paddle.nn.init.xavier_uniform_, gain=1.0
                ),
                output_layer_init_method=functools.partial(
                    paddle.nn.init.xavier_uniform_, gain=1.0
                ),
                tie_word_embeddings=True,
                use_qk_norm=True,
                num_nextn_predict_layers=1,
                use_dense_mtp=True,
                mtp_reuse_last_layer=True,
            )

        assert config.use_dense_mtp is False, (
            "mtp_reuse_last_layer=True should force use_dense_mtp=False, "
            f"got use_dense_mtp={config.use_dense_mtp}"
        )
        assert any("MTP-REUSE-LAST-LAYER" in str(w.message) for w in caught), (
            "Override should emit a [MTP-REUSE-LAST-LAYER] warning"
        )

        model = gpt_builder(config, num_stages=1)
        mtp_layer = self._find_mtp_layer(model)
        assert mtp_layer is not None
        # Backbone is MoE, MTP must therefore also be MoE (override took effect).
        assert isinstance(mtp_layer.transformer_layer.mlp, MoELayer), (
            "MTP layer's MLP should be MoELayer after override, got "
            f"{type(mtp_layer.transformer_layer.mlp).__name__}"
        )

        # And the alias should still produce shared parameters.
        decoder_layers = self._find_decoder_layers(model)
        backbone_params = dict(decoder_layers[-1].named_parameters())
        mtp_params = dict(mtp_layer.transformer_layer.named_parameters())
        not_shared = [
            n for n, p in mtp_params.items() if backbone_params.get(n) is not p
        ]
        assert not not_shared, (
            f"MTP params should reuse last backbone params, not_shared={not_shared[:5]}"
        )


if __name__ == "__main__":
    unittest.main()
