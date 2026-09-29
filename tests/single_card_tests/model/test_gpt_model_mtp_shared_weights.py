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
    resolve_mtp_sampled_depth,
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


def _count_mtp_body_calls(model):
    """Count transformer_layer forward entries per MTP depth.

    The sampled K is observed through its effect -- depth i runs its body iff
    i < K -- rather than through a stashed attribute on the layer, so the
    production path keeps no sampling state around just for the tests.
    Mirrors the body_calls helper in
    tests/multi_card_tests/pipeline_parallel/test_gpt_pp_mtp_depth_sampling.py.
    """
    body_calls = {}
    for layer in _mtp_layers(model):
        body_calls[layer.layer_number] = 0

        def _count(_mod, _inp, _depth=layer.layer_number):
            body_calls[_depth] += 1

        layer.transformer_layer.register_forward_pre_hook(_count)
    return body_calls


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

    @unittest.skipUnless(
        PADDLE_SUPPORTS_SHARED_SUBMODULE,
        "installed paddle's SharedLayerDesc lacks shared_submodule_weight_only",
    )
    def test_combines_with_shared_last_layer(self):
        """Combined mode shares the MTP body with the backbone last layer and
        shares each MTP depth's fusion parameters across depths."""
        config = GPTConfig(
            **self._base_kwargs(),
            mtp_shared_last_layer=True,
            mtp_shared_weights=True,
        )
        model = gpt_builder(config, num_stages=1)

        backbone_layers = [
            layer
            for layer in model.run_function
            if isinstance(layer, TransformerLayer)
        ]
        mtp_layers = _mtp_layers(model)
        assert len(backbone_layers) == 2
        assert len(mtp_layers) == 2

        backbone_params = dict(backbone_layers[-1].transformer_layer_weights)
        for mtp_layer in mtp_layers:
            mtp_body_params = dict(mtp_layer.transformer_layer_weights)
            for name, mtp_param in mtp_body_params.items():
                assert mtp_param is backbone_params[name], (
                    f"MTP body param {name} must share the backbone last layer"
                )

        d0 = dict(mtp_layers[0].all_weights)
        d1 = dict(mtp_layers[1].all_weights)
        for name, p0 in d0.items():
            if name.startswith("transformer_layer."):
                continue
            assert p0 is d1[name], (
                f"MTP fusion param {name} must share across depths"
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

    def test_peer_ranks_derive_matching_k(self):
        """Two independent MTP layer instances must agree on K without talking.

        The MTP block is pinned to one pipeline chunk, so within a rank K is drawn
        once and read back from dict_args. What is still derived independently is
        the draw on each DP / EP peer rank holding that block: they run the same
        schedule, so the same counter index must give the same K, or the MoE
        all-to-all of a computed depth and the per-depth loss reduction stop
        matching. Two layer objects with separate counters stand in for two peer
        ranks here. A drift silently trains a different depth set than the loss
        normalises over, so the whole sequence must match, not just the first draw.
        """
        config = self._cfg([0.2, 0.3, 0.5])
        depths = _mtp_layers(gpt_builder(config, num_stages=1))
        rank_a, rank_b = depths[0], depths[1]
        for layer in (rank_a, rank_b):
            layer.train()

        drawn_a, drawn_b = [], []
        for _ in range(8):
            # separate dicts == separate ranks: neither sees the other's K
            drawn_a.append(resolve_mtp_sampled_depth(rank_a, config, {}))
            drawn_b.append(resolve_mtp_sampled_depth(rank_b, config, {}))

        self.assertEqual(drawn_a, drawn_b)
        self.assertEqual(rank_a._mtp_sampling_counter, 8)
        self.assertEqual(rank_b._mtp_sampling_counter, 8)
        # a constant K would make the equality above vacuous
        self.assertGreater(len(set(drawn_a)), 1)

    def test_seed_offset_continues_the_draw_sequence(self):
        """mtp_depth_sampling_seed_offset=n must resume the stream after n draws.

        This is the answer to a restart zeroing the per-call counter: the offset is
        measured in draws, so setting it to the consumed micro-batch count makes
        the resumed job continue instead of replaying from the start.
        """
        config = self._cfg([0.2, 0.3, 0.5])
        layer = self._mtp0(gpt_builder(config, num_stages=1))
        layer.train()
        first_run = [
            resolve_mtp_sampled_depth(layer, config, {}) for _ in range(6)
        ]

        resumed_config = self._cfg([0.2, 0.3, 0.5])
        resumed_config.mtp_depth_sampling_seed_offset = 3
        resumed = self._mtp0(gpt_builder(resumed_config, num_stages=1))
        resumed.train()
        resumed_run = [
            resolve_mtp_sampled_depth(resumed, resumed_config, {})
            for _ in range(3)
        ]

        self.assertEqual(resumed_run, first_run[3:])

    def test_published_k_is_reused_not_redrawn(self):
        """Within a stage the first depth publishes K in dict_args; later consumers
        must read it back instead of drawing, which is also what makes a recompute
        replay idempotent."""
        config = self._cfg([0.2, 0.3, 0.5])
        depths = _mtp_layers(gpt_builder(config, num_stages=1))
        depths[0].train()
        depths[1].train()

        dict_args = {}
        first = resolve_mtp_sampled_depth(depths[0], config, dict_args)
        second = resolve_mtp_sampled_depth(depths[1], config, dict_args)

        self.assertEqual(first, second)
        self.assertEqual(depths[0]._mtp_sampling_counter, 1)
        # the reusing consumer must not touch its own counter
        self.assertEqual(getattr(depths[1], "_mtp_sampling_counter", 0), 0)

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
        """K=1: step runs, loss finite, only depth 0 runs its body."""
        cfg = self._cfg([1.0, 0.0, 0.0])
        model = gpt_builder(cfg, num_stages=1)
        body_calls = _count_mtp_body_calls(model)
        loss = _run_step(model, cfg, self.strategy)
        assert loss is not None and not paddle.isnan(loss).any(), "loss NaN"
        assert not paddle.isinf(loss).any(), "loss Inf"
        assert body_calls.get(0, 0) > 0, (
            f"depth 0 must run at K=1, body_calls={body_calls}"
        )
        assert all(n == 0 for d, n in body_calls.items() if d >= 1), (
            f"K=1 must skip every depth >= 1, body_calls={body_calls}"
        )

    def test_forward_backward_full(self):
        """K=D behaves like running all depths; loss finite."""
        cfg = self._cfg([0.0, 0.0, 1.0])
        model = gpt_builder(cfg, num_stages=1)
        body_calls = _count_mtp_body_calls(model)
        loss = _run_step(model, cfg, self.strategy)
        assert loss is not None and not paddle.isnan(loss).any(), "loss NaN"
        assert all(n > 0 for n in body_calls.values()), (
            f"K=D must run every depth, body_calls={body_calls}"
        )

    def test_forward_backward_train_mtp_only_k1(self):
        """train_mtp_only must honor the sampled prefix length."""
        cfg = self._cfg([1.0, 0.0, 0.0], train_mtp_only=True)
        model = gpt_builder(cfg, num_stages=1)
        body_calls = _count_mtp_body_calls(model)
        loss = _run_step(model, cfg, self.strategy)
        assert loss is not None and not paddle.isnan(loss).any(), "loss NaN"
        assert all(n == 0 for d, n in body_calls.items() if d >= 1), (
            f"K=1 must skip every depth >= 1, body_calls={body_calls}"
        )

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
        assert not hasattr(mtp0, "_mtp_sampling_counter"), (
            "sampling state must not be set when the feature is disabled"
        )
        assert not hasattr(cfg, "_mtp_sampled_depth"), (
            "no config-level sampling state should exist"
        )

    def _lm_head(self, model):
        return next(
            layer
            for layer in model.run_function
            if type(layer).__name__ == "GPTLMHead"
        )

    def test_lm_head_reads_k_back_instead_of_drawing(self):
        """The head must not own a draw counter any more.

        It used to re-derive K whenever its stage held no MTP depth, which is the
        construct that deadlocks under VPP / p2p overlap. Now the MTP block is
        pinned to one chunk and the head reads the published value, so a counter
        appearing on the head is a regression back to the independent draw.
        """
        cfg = self._cfg([0.34, 0.33, 0.33])
        model = gpt_builder(cfg, num_stages=1)
        head = self._lm_head(model)
        _run_step(model, cfg, self.strategy)
        assert not hasattr(head, "_mtp_sampling_counter"), (
            "the LM head must read K from dict_args, never draw it"
        )

    def test_lm_head_raises_when_k_was_never_published(self):
        """Sampling on but no K in dict_args must fail loudly, not fall back to D.

        Falling back would project the depths the MTP layers skipped and average
        the loss over D, i.e. train on garbage slices without any symptom.
        """
        cfg = self._cfg([1.0, 0.0, 0.0])
        head = self._lm_head(gpt_builder(cfg, num_stages=1))
        hidden = paddle.zeros([cfg.num_nextn_predict_layers + 1, 8])
        with (
            mock.patch.object(head, "_forward", side_effect=lambda t: t),
            self.assertRaisesRegex(
                RuntimeError, r"no MTP depth published mtp_sampled_depth"
            ),
        ):
            head.forward({"hidden_states": hidden})

    def test_local_chunks_keeps_the_mtp_block_together(self):
        """At one stage the whole MTP block must land in a single chunk.

        _assert_mtp_sampling_sites_colocated is built on this: a chunk is the unit
        dict_args flows through, so the depths and the head being in one chunk is
        exactly what lets K be published once and read back.
        """
        cfg = self._cfg([0.34, 0.33, 0.33])
        model = gpt_builder(cfg, num_stages=1)
        chunks = model._local_chunks()

        self.assertEqual(len(chunks), 1)
        depths = [
            la for la in chunks[0] if isinstance(la, MultiTokenPredictionLayer)
        ]
        self.assertEqual(len(depths), cfg.num_nextn_predict_layers)
        self.assertTrue(
            any(type(la).__name__ == "GPTLMHead" for la in chunks[0]),
            "the LM head must share the chunk that publishes K",
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

    def test_fusion_alias_raises_on_missing_param(self):
        """The combined mode aliases fusion params depth by depth; one the pivot
        depth lacks must fail loudly instead of staying unshared."""
        model, mtp = self._independent_model()
        mtp[1].add_parameter(
            "probe_only_on_dest",
            paddle.create_parameter(shape=[2], dtype="float32"),
        )
        with self.assertRaisesRegex(RuntimeError, r"is missing at depth 0"):
            model._alias_mtp_fusion_weights()

    def test_fusion_alias_raises_on_shape_mismatch(self):
        model, mtp = self._independent_model()
        mtp[1].enorm.weight = paddle.create_parameter(
            shape=[mtp[0].enorm.weight.shape[0] + 1], dtype="float32"
        )
        with self.assertRaisesRegex(RuntimeError, r"incompatible shapes"):
            model._alias_mtp_fusion_weights()

    def test_sampler_without_sampling_runs_every_depth(self):
        model, mtp = self._independent_model(num_nextn=3)
        self.assertIsNone(model.config.mtp_depth_sampling)
        self.assertEqual(mtp[0]._sample_mtp_depth(), 3)

    def test_sampler_counter_frozen_without_grad_in_training(self):
        """A no-grad forward in training mode is a recompute pre-pass: it must not
        advance the counter, or the replay would draw a different K."""
        config = GPTConfig(
            **_base_kwargs(num_nextn=3),
            use_dense_mtp=False,
            mtp_depth_sampling=[0.2, 0.3, 0.5],
        )
        depth0 = _mtp_layers(gpt_builder(config, num_stages=1))[0]
        depth0.train()
        with paddle.no_grad():
            k_first = depth0._sample_mtp_depth()
            self.assertEqual(depth0._sample_mtp_depth(), k_first)
        self.assertEqual(depth0._mtp_sampling_counter, 0)
        self.assertEqual(depth0._sample_mtp_depth(), k_first)
        self.assertEqual(depth0._mtp_sampling_counter, 1)

    def test_all_weights_skips_mtp_embed_sublayer(self):
        """all_weights must drop mtp_embed even when it exists. Attached by hand
        here because a real mtp_embed needs enable_mtp_magic_send, which in turn
        requires pipeline_model_parallel_size > 1."""
        _, mtp = self._independent_model()
        layer = mtp[0]
        # Probe with getattr: develop sets ``self.mtp_embed = None``
        # unconditionally, but release branches only create the attribute under
        # enable_mtp_magic_send, and this assertion should hold on both.
        assert getattr(layer, "mtp_embed", None) is None, (
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

    def _combined_sharing_colocation_check(self, layout):
        """Run the combined-sharing placement check with a faked PP layout."""
        model, _ = self._independent_model()

        def _fake_all_gather_object(object_list, obj, group=None):
            object_list.extend(layout)

        with mock.patch.object(
            paddle.distributed,
            "all_gather_object",
            side_effect=_fake_all_gather_object,
        ):
            model._assert_mtp_depths_colocated_for_combined_sharing()

    def test_combined_sharing_accepts_depths_on_one_stage(self):
        """Combined sharing needs the MTP depths co-located because its fusion
        params are rank-local aliases."""
        self._combined_sharing_colocation_check([[0, 1], []])
        self._combined_sharing_colocation_check([[], [0, 1]])

    def test_combined_sharing_rejects_split_depths(self):
        """Fusion params in the combined mode are rank-local aliases, so split
        MTP depths would leave them unshared across PP stages."""
        with self.assertRaisesRegex(
            RuntimeError,
            r"mtp_shared_weights \+ mtp_shared_last_layer requires all MTP",
        ):
            self._combined_sharing_colocation_check([[0], [1]])

    def test_shared_grad_allreduce_dedups_per_param_and_group(self):
        """A Parameter reached twice through the SAME comm group is reduced once.

        The alias machinery deliberately makes several MTP depths share one
        Parameter object, and paddle's base implementation walks every
        (key, weight_attr, param) triple with no dedup at all, so the same
        gradient would be all_reduced twice and come out doubled. GPTModel's
        override keys on (id(param), id(group)).

        The second half is the reason one layer must never carry two shared keys:
        two keys mean two different groups, so the dedup cannot catch them and the
        gradient really would be counted twice.
        """
        model, _ = self._independent_model()

        class _Holder(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.inner = paddle.nn.Linear(4, 4)

            @property
            def pair(self):
                return self.inner.named_parameters()

        holder = _Holder()
        holder.inner(paddle.ones([2, 4])).sum().backward()
        n_params = len(list(holder.pair))
        group_a, group_b = object(), object()

        def _reduce_calls(shared_comm):
            model.shared_comm = shared_comm
            calls = []
            with (
                mock.patch.object(
                    type(model),
                    "_get_mtp_embed_primary_weight",
                    return_value=None,
                ),
                mock.patch.object(
                    paddle.distributed,
                    "all_reduce",
                    side_effect=lambda tensor, group=None: calls.append(
                        id(group)
                    ),
                ),
            ):
                model.allreduce_shared_weight_gradients()
            return calls

        same_group = _reduce_calls(
            {
                "k1": {
                    "layer": holder,
                    "weight_attr": ["pair"],
                    "group": group_a,
                },
                "k2": {
                    "layer": holder,
                    "weight_attr": ["pair"],
                    "group": group_a,
                },
            }
        )
        self.assertEqual(len(same_group), n_params)

        two_groups = _reduce_calls(
            {
                "k1": {
                    "layer": holder,
                    "weight_attr": ["pair"],
                    "group": group_a,
                },
                "k2": {
                    "layer": holder,
                    "weight_attr": ["pair"],
                    "group": group_b,
                },
            }
        )
        self.assertEqual(len(two_groups), 2 * n_params)

    def _sampling_colocation_check(self, layout):
        """Run the sampling placement check with a faked PP / chunk layout.

        Each gathered element is one rank's list of (mtp_depths, mtp_lm_heads),
        one entry per chunk that rank runs.
        """
        model, _ = self._independent_model(num_nextn=3)

        def _fake_all_gather_object(object_list, obj, group=None):
            object_list.extend(layout)

        with mock.patch.object(
            paddle.distributed,
            "all_gather_object",
            side_effect=_fake_all_gather_object,
        ):
            model._assert_mtp_sampling_sites_colocated()

    def test_sampling_accepts_whole_mtp_block_in_one_chunk(self):
        """K rides in dict_args, so every consumer must be in the drawing chunk."""
        self._sampling_colocation_check([[([0, 1, 2], 1)], [([], 0)]])
        self._sampling_colocation_check([[([], 0)], [([0, 1, 2], 1)]])

    def test_sampling_rejects_split_depths(self):
        """Split depths would make two chunks derive K independently -- the case
        that deadlocks once their entry counts drift under VPP / p2p overlap."""
        with self.assertRaisesRegex(
            RuntimeError, r"mtp_depth_sampling requires every MTP depth"
        ):
            self._sampling_colocation_check([[([0], 1)], [([1, 2], 0)]])

    def test_sampling_rejects_missing_depth(self):
        """The holder chunk must carry the whole 0..D-1 range, not a subset."""
        with self.assertRaisesRegex(
            RuntimeError, r"mtp_depth_sampling requires every MTP depth"
        ):
            self._sampling_colocation_check([[([0, 2], 1)], [([], 0)]])

    def test_sampling_rejects_lm_head_in_another_chunk(self):
        """The head reads K back rather than deriving it, so it must be able to."""
        with self.assertRaisesRegex(
            RuntimeError, r"mtp_depth_sampling requires every MTP depth"
        ):
            self._sampling_colocation_check([[([0, 1, 2], 0)], [([], 1)]])

    def test_sampling_ignores_an_undetectable_lm_head(self):
        """No K-consuming head visible anywhere -> do not fail the build.

        GPTMainLMHead / GPTMTPLMHead override forward and never read K, so a model
        wired that way legitimately reports zero heads. GPTLMHead.forward still
        raises at the first step if K really is missing, so skipping the head half
        of the check here cannot hide a silent divergence.
        """
        self._sampling_colocation_check([[([0, 1, 2], 0)], [([], 0)]])

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
        body_calls = _count_mtp_body_calls(model)

        loss = _run_step(model, config, self.strategy)
        assert loss is not None and not paddle.isnan(loss).any(), (
            "recompute + sampling step did not produce a usable loss"
        )
        assert depth0._mtp_sampling_counter == 1, (
            "one micro-batch must draw K exactly once; counter="
            f"{depth0._mtp_sampling_counter} means a recompute replay re-drew it"
        )
        assert body_calls.get(0, 0) > 0 and body_calls.get(1, 0) > 0, (
            f"P(K=2)=1 must run depths 0 and 1, body_calls={body_calls}"
        )
        assert body_calls.get(2, 0) == 0, (
            f"P(K=2)=1 must skip depth 2, body_calls={body_calls}"
        )


if __name__ == "__main__":
    unittest.main()
