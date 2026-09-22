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
from unittest import mock

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


def _base_kwargs(num_nextn=2):
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


class TestMTPSharedWeights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.strategy = _init_fleet()

    def _base_kwargs(self, num_nextn=2):
        return _base_kwargs(num_nextn)

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
    def test_all_weights_excludes_mtp_embed(self):
        """mtp_embed must stay out of the shared key: GPTModel syncs it through
        _tie_mtp_embed_weights_intra_rank + _mtp_embed_global_group, so letting it
        into shared_comm too would allreduce its gradient twice."""
        config = GPTConfig(
            **self._base_kwargs(),
            mtp_shared_weights=True,
        )
        model = gpt_builder(config, num_stages=1)
        mtp = _mtp_layers(model)
        names = [n for n, _ in mtp[0].all_weights]
        assert names, "all_weights should not be empty"
        assert not [n for n in names if n.startswith("mtp_embed.")], (
            f"mtp_embed must be excluded from all_weights, got {names}"
        )
        # Sanity: the body and the fusion modules ARE covered.
        assert any(n.startswith("transformer_layer.") for n in names)
        assert "enorm.weight" in names

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


@unittest.skipUnless(
    PADDLE_SUPPORTS_SHARED_SUBMODULE,
    "installed paddle's SharedLayerDesc lacks shared_submodule_weight_only",
)
class TestMTPSharedWeightsGuards(unittest.TestCase):
    """The defensive branches of the sharing machinery.

    These paths only fire on a malformed build (diverged specs, MTP depths split
    across pipeline stages), so they are driven directly rather than through a
    real multi-card run.
    """

    @classmethod
    def setUpClass(cls):
        cls.strategy = _init_fleet()

    def _independent_model(self, num_nextn=2):
        """A model whose MTP depths are NOT shared, with the flag flipped on
        afterwards so _alias_shared_layer takes the widened branch when called
        by hand."""
        config = GPTConfig(**_base_kwargs(num_nextn))
        model = gpt_builder(config, num_stages=1)
        model.config.mtp_shared_weights = True
        mtp = _mtp_layers(model)
        assert len(mtp) == num_nextn
        return model, mtp

    def test_alias_raises_on_shape_mismatch(self):
        """A diverged spec must fail loudly, and via raise rather than assert so
        `python -O` cannot strip it."""
        model, mtp = self._independent_model()
        mtp[1].enorm.weight = paddle.create_parameter(
            shape=[mtp[0].enorm.weight.shape[0] + 1], dtype="float32"
        )
        with self.assertRaisesRegex(RuntimeError, r"shape_mismatch=1"):
            model._alias_shared_layer(mtp[1], mtp[0])

    def test_alias_raises_on_missing_param(self):
        """A parameter present on the destination depth but absent from the pivot
        is counted as missing and reported."""
        model, mtp = self._independent_model()
        mtp[1].add_parameter(
            "probe_only_on_dest",
            paddle.create_parameter(shape=[2], dtype="float32"),
        )
        with self.assertRaisesRegex(RuntimeError, r"missing=1"):
            model._alias_shared_layer(mtp[1], mtp[0])

    def test_all_weights_skips_mtp_embed_sublayer(self):
        """all_weights must drop mtp_embed even when it exists. Attached by hand
        here because a real mtp_embed needs enable_mtp_magic_send, which in turn
        requires pipeline_model_parallel_size > 1."""
        _, mtp = self._independent_model()
        layer = mtp[0]
        assert layer.mtp_embed is None, (
            "expected no mtp_embed without magic send"
        )
        layer.add_sublayer("mtp_embed", paddle.nn.Linear(4, 4))
        names = [n for n, _ in layer.all_weights]
        assert names, "all_weights should not be empty"
        assert not [n for n in names if n.startswith("mtp_embed.")], (
            f"mtp_embed must be filtered out, got {names}"
        )
        assert "mtp_embed.weight" in dict(layer.named_parameters()), (
            "the probe sublayer should be visible to named_parameters"
        )

    def _colocation_check(self, layout):
        """Run the co-location check with a faked pipe-group layout."""
        model, _ = self._independent_model()

        def _fake_all_gather_object(object_list, obj, group=None):
            object_list.extend(layout)

        with mock.patch.object(
            paddle.distributed,
            "all_gather_object",
            side_effect=_fake_all_gather_object,
        ):
            model._assert_mtp_depths_colocated_for_sampling()

    def test_colocation_accepts_depths_on_last_stage(self):
        """All depths on the last stage is the supported layout."""
        self._colocation_check([[], [0, 1]])

    def test_colocation_rejects_split_depths(self):
        """Split depths must raise: K rides in dict_args and does not cross a
        stage boundary, so the off-stage depths would silently run in full while
        the loss still normalises over K."""
        with self.assertRaisesRegex(
            RuntimeError, r"requires every MTP depth to live on the LAST"
        ):
            self._colocation_check([[0], [1]])

    def test_colocation_rejects_depths_off_last_stage(self):
        """Depths co-located but NOT on the last stage must also raise: the MTP LM
        head and the loss live on the last stage, so they would never see K and
        would project the skipped depths anyway."""
        with self.assertRaisesRegex(
            RuntimeError, r"requires every MTP depth to live on the LAST"
        ):
            self._colocation_check([[0, 1], []])

    def test_sampling_is_idempotent_under_recompute(self):
        """A recompute replay must reuse the forward's K, not draw a new one.

        Otherwise the backward would be computed for a different set of depths
        than the forward ran -- silently wrong gradients rather than a crash. Two
        things protect this: the caller's `"mtp_sampled_depth" not in dict_args`
        guard, and _sample_mtp_depth only advancing its counter outside a replay.
        The counter is the observable: one step over one micro-batch must advance
        it exactly once no matter how many times forward is entered.
        """
        config = GPTConfig(
            **_base_kwargs(num_nextn=3),
            use_dense_mtp=False,
            mtp_depth_sampling=[0.0, 1.0, 0.0],
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
        )
        model = gpt_builder(config, num_stages=1)
        depth0 = next(la for la in _mtp_layers(model) if la.layer_number == 0)

        loss = _run_step(model, config, self.strategy)
        assert loss is not None and not paddle.isnan(loss).any(), (
            "recompute + sampling step did not produce a usable loss"
        )
        assert depth0._mtp_sampling_counter == 1, (
            "one micro-batch must draw K exactly once; counter="
            f"{depth0._mtp_sampling_counter} means a recompute replay re-drew it"
        )
        assert depth0._last_sampled_depth == 2, (
            f"P(K=2)=1 was configured, got {depth0._last_sampled_depth}"
        )


if __name__ == "__main__":
    unittest.main()
