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

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import NoPipelineParallel

import paddlefleet.parallel_state as ps
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


class TestMTPDepthSampling(unittest.TestCase):
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
        try:
            fleet.init(is_collective=True, strategy=strategy)
        except Exception:
            pass
        hcg = fleet.get_hybrid_communicate_group()
        try:
            ps.initialize_model_parallel(hcg)
        except Exception:
            pass
        cls.strategy = strategy

    def _cfg(self, mtp_depth_sampling, num_nextn=3):
        return GPTConfig(
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
            num_nextn_predict_layers=num_nextn,
            use_dense_mtp=False,
            mtp_depth_sampling=mtp_depth_sampling,
        )

    def _mtp0(self, model):
        for layer in model.run_function:
            if isinstance(layer, MultiTokenPredictionLayer):
                return layer
        return None

    def _run_step(self, model, config):
        seq = config.max_sequence_length
        data = list(range(seq))
        input_ids = paddle.to_tensor(data, dtype=paddle.int64).repeat((1, 1))
        position_ids = paddle.to_tensor(data, dtype=paddle.int64).repeat((1, 1))
        labels = paddle.to_tensor(
            list(range(1, seq + 1)), dtype=paddle.int64
        ).repeat((1, 1))
        pipe = NoPipelineParallel(model, self.strategy)
        loss = pipe.forward_backward_pipeline(
            (
                {"input_ids": [input_ids], "position_ids": [position_ids]},
                [labels],
            )
        )
        return loss

    def test_sampler_disabled_returns_full_depth(self):
        """mtp_depth_sampling=None -> _sample_mtp_depth short-circuits to D."""
        cfg = self._cfg(None)
        model = gpt_builder(cfg, num_stages=1)
        mtp0 = self._mtp0(model)
        assert mtp0._sample_mtp_depth() == cfg.num_nextn_predict_layers

    def test_sampler_fixed_k1(self):
        """P(K=1)=1 -> always sample K=1."""
        cfg = self._cfg([1.0, 0.0, 0.0])
        model = gpt_builder(cfg, num_stages=1)
        mtp0 = self._mtp0(model)
        ks = [mtp0._sample_mtp_depth() for _ in range(50)]
        assert set(ks) == {1}, f"expected all K==1, got {sorted(set(ks))}"

    def test_sampler_fixed_k3(self):
        """P(K=3)=1 -> always sample K=3 (= run all depths)."""
        cfg = self._cfg([0.0, 0.0, 1.0])
        model = gpt_builder(cfg, num_stages=1)
        mtp0 = self._mtp0(model)
        ks = [mtp0._sample_mtp_depth() for _ in range(50)]
        assert set(ks) == {3}, f"expected all K==3, got {sorted(set(ks))}"

    def test_sampler_distribution(self):
        """Mixed distribution -> K spans expected support, E[K] < D."""
        cfg = self._cfg([0.5, 0.5, 0.0])
        model = gpt_builder(cfg, num_stages=1)
        mtp0 = self._mtp0(model)
        ks = [mtp0._sample_mtp_depth() for _ in range(400)]
        assert set(ks) <= {1, 2}, f"K out of support: {sorted(set(ks))}"
        assert 1 in ks and 2 in ks, (
            f"both 1 and 2 should appear: {sorted(set(ks))}"
        )
        assert sum(ks) / len(ks) < 3, "E[K] must be < D=3"

    def test_forward_backward_sampling_k1(self):
        """Fixed K=1: forward/backward runs, loss finite, only depth-0 active."""
        cfg = self._cfg([1.0, 0.0, 0.0])
        model = gpt_builder(cfg, num_stages=1)
        mtp0 = self._mtp0(model)
        loss = self._run_step(model, cfg)
        assert loss is not None and not paddle.isnan(loss).any(), "loss NaN"
        assert not paddle.isinf(loss).any(), "loss Inf"
        assert getattr(mtp0, "_last_sampled_depth", None) == 1, (
            f"expected sampled K==1, got {getattr(mtp0, '_last_sampled_depth', None)}"
        )

    def test_forward_backward_sampling_full(self):
        """Fixed K=3 sampling equals running all depths; loss finite."""
        cfg = self._cfg([0.0, 0.0, 1.0])
        model = gpt_builder(cfg, num_stages=1)
        mtp0 = self._mtp0(model)
        loss = self._run_step(model, cfg)
        assert loss is not None and not paddle.isnan(loss).any(), "loss NaN"
        assert getattr(mtp0, "_last_sampled_depth", None) == 3

    def test_null_baseline_runs(self):
        """mtp_depth_sampling=None (default) trains normally (no skip path)."""
        cfg = self._cfg(None)
        model = gpt_builder(cfg, num_stages=1)
        mtp0 = self._mtp0(model)
        loss = self._run_step(model, cfg)
        assert loss is not None and not paddle.isnan(loss).any(), "loss NaN"
        assert not hasattr(mtp0, "_last_sampled_depth"), (
            "sampling state must not be set when feature is disabled"
        )


if __name__ == "__main__":
    unittest.main()
