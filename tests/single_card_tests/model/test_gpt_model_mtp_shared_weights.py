# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Single-card tests for mtp_shared_weights and mtp_depth_sampling."""

import functools
import inspect
import random
import unittest

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import (
    NoPipelineParallel,
    SharedLayerDesc,
)

import paddlefleet.parallel_state as ps
from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig
from paddlefleet.transformer.multi_token_prediction import (
    MultiTokenPredictionLayer,
)
from paddlefleet.transformer.transformer_layer import TransformerLayer

# mtp_shared_last_layer needs a paddle whose SharedLayerDesc understands
# shared_submodule_weight_only (the flag paddlefleet passes for the MTP body).
# Older paddle builds treat it as a layer kwarg and then choke on the
# named_parameters() generator returned by transformer_layer_weights.
PADDLE_SUPPORTS_SHARED_SUBMODULE = (
    "shared_submodule_weight_only"
    in inspect.signature(SharedLayerDesc.__init__).parameters
)


def _init_fleet():
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
        # Another test class in the same process may already have done this.
        pass
    hcg = fleet.get_hybrid_communicate_group()
    try:
        ps.initialize_model_parallel(hcg)
    except Exception:
        pass
    return strategy


def _mtp_layers(model):
    return [
        layer
        for layer in model.run_function
        if isinstance(layer, MultiTokenPredictionLayer)
    ]


def _decoder_layers(model):
    return [
        layer
        for layer in model.run_function
        if isinstance(layer, TransformerLayer)
    ]


def _run_step(model, config, strategy):
    seq = config.max_sequence_length
    data = list(range(seq))
    input_ids = paddle.to_tensor(data, dtype=paddle.int64).repeat((1, 1))
    position_ids = paddle.to_tensor(data, dtype=paddle.int64).repeat((1, 1))
    labels = paddle.to_tensor(
        list(range(1, seq + 1)), dtype=paddle.int64
    ).repeat((1, 1))
    pipe = NoPipelineParallel(model, strategy)
    return pipe.forward_backward_pipeline(
        (
            {"input_ids": [input_ids], "position_ids": [position_ids]},
            [labels],
        )
    )


class TestMTPSharedWeights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.strategy = _init_fleet()

    def _base_kwargs(self, num_nextn=2):
        return {
            "num_hidden_layers": 2,
            "hidden_size": 512,
            "vocab_size": 100,
            "max_sequence_length": 64,
            "num_attention_heads": 4,
            "moe_expert_fusion": False,
            "intermediate_size": 1024,
            "normalization": "RMSNorm",
            "hidden_dropout_prob": 0.0,
            "attention_dropout": 0.0,
            "n_routed_experts": 8,
            "moe_intermediate_size": 1024,
            "moe_token_dispatcher_type": "alltoall",
            "n_shared_experts": 1,
            "use_bias": False,
            "rotary_percent": 1.0,
            "rotary_base": 10000,
            "rope_scaling": 1.0,
            "init_method": functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            "output_layer_init_method": functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            "tie_word_embeddings": True,
            "use_qk_norm": True,
            "num_nextn_predict_layers": num_nextn,
        }

    def test_off_by_default(self):
        """Without mtp_shared_weights every MTP depth keeps its own parameters."""
        config = GPTConfig(**self._base_kwargs())
        model = gpt_builder(config, num_stages=1)
        mtp = _mtp_layers(model)
        assert len(mtp) == 2, f"expected 2 MTP layers, got {len(mtp)}"
        d0 = dict(mtp[0].named_parameters())
        shared = [n for n, p in mtp[1].named_parameters() if d0.get(n) is p]
        assert not shared, (
            f"depths must stay independent when the flag is off, shared={shared[:5]}"
        )

    @unittest.skipUnless(
        PADDLE_SUPPORTS_SHARED_SUBMODULE,
        "installed paddle's SharedLayerDesc lacks shared_submodule_weight_only",
    )
    def test_shares_everything(self):
        """mtp_shared_weights: depth 1 shares depth 0's FULL parameter set, i.e.
        the transformer_layer body plus enorm/hnorm/eh_proj/norm."""
        config = GPTConfig(
            **self._base_kwargs(),
            mtp_shared_weights=True,
        )
        model = gpt_builder(config, num_stages=1)
        mtp = _mtp_layers(model)
        assert len(mtp) == 2

        d0 = dict(mtp[0].named_parameters())
        assert d0, "MTP layer should expose parameters"
        not_shared = [
            name
            for name, p in mtp[1].named_parameters()
            if d0.get(name) is not p
        ]
        assert not not_shared, (
            f"depth-1 must share ALL depth-0 params, not_shared={not_shared[:8]}"
        )

        # The body is included, and so are the fusion modules.
        body = [n for n in d0 if n.startswith("transformer_layer.")]
        assert body, "expected transformer_layer.* params on the MTP layer"
        d1 = dict(mtp[1].named_parameters())
        for fusion in ("enorm.weight", "hnorm.weight", "eh_proj.weight"):
            assert fusion in d0, f"expected fusion param {fusion}"
            assert d0[fusion] is d1.get(fusion), (
                f"fusion {fusion} not shared across depths"
            )

    def test_mutually_exclusive_with_shared_last_layer(self):
        """One LayerDesc carries one SharedLayerDesc key, and the two flags want
        different pivots for the same MTP layers, so the combination is refused."""
        with self.assertRaisesRegex(
            ValueError,
            r"mtp_shared_weights and mtp_shared_last_layer cannot both be True",
        ):
            GPTConfig(
                **self._base_kwargs(),
                mtp_shared_last_layer=True,
                mtp_shared_weights=True,
            )

    def test_single_depth_rejected(self):
        """With one depth there is nothing to share; refuse instead of no-op."""
        with self.assertRaisesRegex(
            ValueError,
            r"mtp_shared_weights requires num_nextn_predict_layers >= 2",
        ):
            GPTConfig(
                **self._base_kwargs(num_nextn=1),
                mtp_shared_weights=True,
            )

    @unittest.skipUnless(
        PADDLE_SUPPORTS_SHARED_SUBMODULE,
        "installed paddle's SharedLayerDesc lacks shared_submodule_weight_only",
    )
    def test_emits_one_shared_key_for_all_depths(self):
        """Every MTP depth must land under the SAME key, and the backbone-last
        pivot of mtp_reuse_transformer must not be emitted: a pivot with no
        members would build a shared_comm group nobody joins."""
        config = GPTConfig(
            **self._base_kwargs(),
            mtp_shared_weights=True,
        )
        model = gpt_builder(config, num_stages=1)
        keys = [
            layer.layer_name
            for layer in model.layers
            if isinstance(layer, SharedLayerDesc)
        ]
        assert keys.count("mtp_shared_all") == 2, (
            f"expected one mtp_shared_all desc per MTP depth, got {keys}"
        )
        assert "mtp_reuse_transformer" not in keys, (
            f"mtp_reuse_transformer must not be emitted, got {keys}"
        )

    @unittest.skipUnless(
        PADDLE_SUPPORTS_SHARED_SUBMODULE,
        "installed paddle's SharedLayerDesc lacks shared_submodule_weight_only",
    )
    def test_forward_backward_with_shared_weights(self):
        """Sharing must not break the training step."""
        config = GPTConfig(
            **self._base_kwargs(),
            mtp_shared_weights=True,
        )
        model = gpt_builder(config, num_stages=1)
        loss = _run_step(model, config, self.strategy)
        assert loss is not None, "no loss returned"
        assert not paddle.isnan(loss).any(), "loss is NaN"
        assert not paddle.isinf(loss).any(), "loss is Inf"


class TestMTPDepthSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.strategy = _init_fleet()

    def _cfg(self, mtp_depth_sampling, num_nextn=3, train_mtp_only=False):
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
            train_mtp_only=train_mtp_only,
        )

    def _mtp0(self, model):
        layers = _mtp_layers(model)
        return layers[0] if layers else None

    def test_sampler_fixed_k1(self):
        """P(K=1)=1 -> always sample K=1."""
        cfg = self._cfg([1.0, 0.0, 0.0])
        mtp0 = self._mtp0(gpt_builder(cfg, num_stages=1))
        ks = [mtp0._sample_mtp_depth() for _ in range(50)]
        assert set(ks) == {1}, f"expected all K==1, got {sorted(set(ks))}"

    def test_sampler_fixed_kd(self):
        """P(K=D)=1 -> always sample K=D (runs every depth)."""
        cfg = self._cfg([0.0, 0.0, 1.0])
        mtp0 = self._mtp0(gpt_builder(cfg, num_stages=1))
        ks = [mtp0._sample_mtp_depth() for _ in range(50)]
        assert set(ks) == {3}, f"expected all K==3, got {sorted(set(ks))}"

    def test_sampler_distribution(self):
        """Mixed distribution -> K stays in support and E[K] < D."""
        cfg = self._cfg([0.5, 0.5, 0.0])
        mtp0 = self._mtp0(gpt_builder(cfg, num_stages=1))
        ks = [mtp0._sample_mtp_depth() for _ in range(400)]
        assert set(ks) <= {1, 2}, f"K out of support: {sorted(set(ks))}"
        assert 1 in ks and 2 in ks, f"both should appear: {sorted(set(ks))}"
        assert sum(ks) / len(ks) < 3, "E[K] must be < D=3"

    def test_forward_backward_k1(self):
        """K=1: step runs, loss finite, depth-0 records the sampled K."""
        cfg = self._cfg([1.0, 0.0, 0.0])
        model = gpt_builder(cfg, num_stages=1)
        mtp0 = self._mtp0(model)
        loss = _run_step(model, cfg, self.strategy)
        assert loss is not None and not paddle.isnan(loss).any(), "loss NaN"
        assert not paddle.isinf(loss).any(), "loss Inf"
        assert getattr(mtp0, "_last_sampled_depth", None) == 1, (
            f"expected K==1, got {getattr(mtp0, '_last_sampled_depth', None)}"
        )

    def test_forward_backward_full(self):
        """K=D behaves like running all depths; loss finite."""
        cfg = self._cfg([0.0, 0.0, 1.0])
        model = gpt_builder(cfg, num_stages=1)
        mtp0 = self._mtp0(model)
        loss = _run_step(model, cfg, self.strategy)
        assert loss is not None and not paddle.isnan(loss).any(), "loss NaN"
        assert getattr(mtp0, "_last_sampled_depth", None) == 3

    def test_forward_backward_train_mtp_only_k1(self):
        """train_mtp_only must honor the sampled prefix length."""
        cfg = self._cfg([1.0, 0.0, 0.0], train_mtp_only=True)
        model = gpt_builder(cfg, num_stages=1)
        mtp0 = self._mtp0(model)
        loss = _run_step(model, cfg, self.strategy)
        assert loss is not None and not paddle.isnan(loss).any(), "loss NaN"
        assert getattr(mtp0, "_last_sampled_depth", None) == 1

    def test_sampling_rejects_non_finite_probability(self):
        """NaN and infinity must fail during config validation."""
        for probs in ([float("nan"), 0.0, 1.0], [float("inf"), 0.0, 0.0]):
            with self.assertRaises(ValueError):
                self._cfg(probs)

    def test_null_baseline_runs(self):
        """mtp_depth_sampling=None (default) trains with no skip path at all."""
        cfg = self._cfg(None)
        model = gpt_builder(cfg, num_stages=1)
        mtp0 = self._mtp0(model)
        loss = _run_step(model, cfg, self.strategy)
        assert loss is not None and not paddle.isnan(loss).any(), "loss NaN"
        assert not hasattr(mtp0, "_last_sampled_depth"), (
            "sampling state must not be set when the feature is disabled"
        )
        assert not hasattr(cfg, "_mtp_sampled_depth"), (
            "no config-level sampling state should exist"
        )

    def test_lm_head_emits_none_for_skipped_depths(self):
        """The LM head must place None at every sampled-out depth and keep the
        list length at D+1, which is how the loss detects the skipped depths."""
        cfg = self._cfg([1.0, 0.0, 0.0])
        model = gpt_builder(cfg, num_stages=1)
        captured = {}

        for layer in model.run_function:
            if type(layer).__name__ == "GPTLMHead":
                original = layer.forward

                def spy(dict_args, _orig=original):
                    out = _orig(dict_args)
                    captured["logits"] = out
                    return out

                layer.forward = spy
                break

        _run_step(model, cfg, self.strategy)
        logits = captured.get("logits")
        assert logits is not None, "LM head was not exercised"
        assert len(logits) == cfg.num_nextn_predict_layers + 1, (
            f"expected D+1 entries, got {len(logits)}"
        )
        assert logits[1] is not None, "depth 0 must be computed at K=1"
        assert logits[2] is None and logits[3] is None, (
            "depths >= K must be None placeholders"
        )


if __name__ == "__main__":
    unittest.main()
