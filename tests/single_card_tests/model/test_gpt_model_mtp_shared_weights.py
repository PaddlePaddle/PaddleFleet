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
import os
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
    draw_mtp_sampled_depth,
    resolve_mtp_sampled_depth,
)
from paddlefleet.transformer.transformer_layer import TransformerLayer


def _at_train_step(step):
    """Pretend the trainer is on optimizer step ``step``.

    K is a pure function of (config.seed, train step), and the step reaches the
    sampler through the TRAINER_GLOBAL_STEP environment variable the trainer
    exports before every forward. Tests that want a different draw move the step,
    which is the only knob there is now.
    """
    return mock.patch.dict(os.environ, {"TRAINER_GLOBAL_STEP": str(step)})


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
        """Independent sites must agree on K without talking to each other.

        Nothing synchronises the draw: each DP / EP peer rank, each pipeline
        stage and each virtual chunk derives K on its own from (seed, step). If
        two of them disagreed, the MoE all-to-all of a computed depth and the
        per-depth loss reduction would stop matching -- a drift silently trains a
        different depth set than the loss normalises over. Separate dicts stand in
        for separate sites here, and the whole step sequence must match, not just
        one draw.
        """
        config = self._cfg([0.2, 0.3, 0.5])
        peer_config = self._cfg([0.2, 0.3, 0.5])

        drawn_a, drawn_b = [], []
        for step in range(8):
            with _at_train_step(step):
                # separate dicts == separate sites: neither sees the other's K
                drawn_a.append(resolve_mtp_sampled_depth(config, {}))
                drawn_b.append(resolve_mtp_sampled_depth(peer_config, {}))

        self.assertEqual(drawn_a, drawn_b)
        # a constant K would make the equality above vacuous
        self.assertGreater(len(set(drawn_a)), 1)

    def test_resume_continues_the_draw_sequence(self):
        """A restart must carry on the stream, not replay it.

        There is no sampling state to save or restore: the trainer already
        restores global_step before the first forward, and K is a function of it,
        so a job that died after 3 steps and resumes at step 3 draws exactly what
        the uninterrupted run would have drawn. Modelled here by re-deriving the
        tail of a sequence from a fresh config object.
        """
        config = self._cfg([0.2, 0.3, 0.5])
        first_run = []
        for step in range(6):
            with _at_train_step(step):
                first_run.append(resolve_mtp_sampled_depth(config, {}))

        resumed_config = self._cfg([0.2, 0.3, 0.5])
        resumed_run = []
        for step in range(3, 6):
            with _at_train_step(step):
                resumed_run.append(
                    resolve_mtp_sampled_depth(resumed_config, {})
                )

        self.assertEqual(resumed_run, first_run[3:])

    def test_published_k_is_reused_not_redrawn(self):
        """dict_args is a cache: a consumer that finds K there must not re-draw.

        Re-deriving would give the same answer within a step, so the observable
        has to be a step that moves underneath the cached value -- after which
        reading the dict must still return the originally published K. That is
        what keeps every depth and the LM head of one forward on one K even if
        the trainer bumps the step mid-flight.
        """
        config = self._cfg([1.0, 0.0, 0.0])
        dict_args = {}
        with _at_train_step(0):
            first = resolve_mtp_sampled_depth(config, dict_args)
        self.assertEqual(dict_args["mtp_sampled_depth"], first)

        dict_args["mtp_sampled_depth"] = 99
        with _at_train_step(7):
            self.assertEqual(resolve_mtp_sampled_depth(config, dict_args), 99)

    def test_sampler_is_stateless_within_a_step(self):
        """Repeated draws inside one optimizer step must all give the same K.

        This is what makes a recompute replay and every micro-batch of the step
        agree for free, and it is the property that replaced the old per-call
        counter -- there is no state left to get out of sync.
        """
        cfg = self._cfg([0.2, 0.3, 0.5])
        mtp0 = self._mtp0(gpt_builder(cfg, num_stages=1))
        mtp0.train()
        with _at_train_step(11):
            ks = {mtp0._sample_mtp_depth() for _ in range(5)}
            with paddle.no_grad():
                ks.add(mtp0._sample_mtp_depth())
        self.assertEqual(len(ks), 1, f"K must be fixed within a step, got {ks}")

    def test_sampler_without_a_train_step_stays_constant(self):
        """Neither step variable set -> step 0 for everyone, so K never moves.

        Correct but degenerate, and the sampler warns rather than failing: a
        training loop that does not export the step still trains, it just trains
        one fixed depth prefix. Pinning it here documents that the fallback is
        deliberate and, more importantly, that it is not random per call.
        """
        cfg = self._cfg([0.2, 0.3, 0.5])
        with mock.patch.dict(os.environ, {}, clear=False):
            for name in ("TRAINER_GLOBAL_STEP", "PDC_INIT_STEP"):
                os.environ.pop(name, None)
            ks = {draw_mtp_sampled_depth(cfg) for _ in range(20)}
        with _at_train_step(0):
            self.assertEqual(ks, {draw_mtp_sampled_depth(cfg)})
        self.assertEqual(len(ks), 1)

    def test_sampler_step_sources_match_has_recovered(self):
        """The step is read from the same pair, in the same order, as
        recompute_utils.has_recovered(): TRAINER_GLOBAL_STEP first, then
        PDC_INIT_STEP.

        Reusing that reader's contract is the point -- these variables already
        exist in the repo and the pretraining trainers already write the first
        one, so sampling adds no new launcher requirement. The order matters:
        PDC_INIT_STEP is fixed for a job, so preferring it would pin K.
        """
        cfg = self._cfg([0.2, 0.3, 0.5])
        # Steps chosen so the two sources give DIFFERENT K, otherwise the
        # precedence assertion below would hold whichever one was read.
        with _at_train_step(3):
            k_trainer = draw_mtp_sampled_depth(cfg)
        with mock.patch.dict(os.environ, {"PDC_INIT_STEP": "2"}):
            os.environ.pop("TRAINER_GLOBAL_STEP", None)
            k_pdc = draw_mtp_sampled_depth(cfg)
        self.assertNotEqual(
            k_trainer,
            k_pdc,
            "pick steps whose K differ or this test is vacuous",
        )

        # both set -> TRAINER_GLOBAL_STEP wins
        with mock.patch.dict(
            os.environ, {"TRAINER_GLOBAL_STEP": "3", "PDC_INIT_STEP": "2"}
        ):
            self.assertEqual(draw_mtp_sampled_depth(cfg), k_trainer)

        # a malformed value must not be trusted; fall through to the next name
        with mock.patch.dict(
            os.environ,
            {"TRAINER_GLOBAL_STEP": "not-an-int", "PDC_INIT_STEP": "2"},
        ):
            self.assertEqual(draw_mtp_sampled_depth(cfg), k_pdc)

    def test_sampler_fixed_k1(self):
        """P(K=1)=1 -> always sample K=1."""
        cfg = self._cfg([1.0, 0.0, 0.0])
        mtp0 = self._mtp0(gpt_builder(cfg, num_stages=1))
        ks = []
        for step in range(50):
            with _at_train_step(step):
                ks.append(mtp0._sample_mtp_depth())
        assert set(ks) == {1}, f"expected all K==1, got {sorted(set(ks))}"

    def test_sampler_fixed_kd(self):
        """P(K=D)=1 -> always sample K=D (runs every depth)."""
        cfg = self._cfg([0.0, 0.0, 1.0])
        mtp0 = self._mtp0(gpt_builder(cfg, num_stages=1))
        ks = []
        for step in range(50):
            with _at_train_step(step):
                ks.append(mtp0._sample_mtp_depth())
        assert set(ks) == {3}, f"expected all K==3, got {sorted(set(ks))}"

    def test_sampler_distribution(self):
        """Mixed distribution -> K stays in support and E[K] < D."""
        cfg = self._cfg([0.5, 0.5, 0.0])
        mtp0 = self._mtp0(gpt_builder(cfg, num_stages=1))
        ks = []
        for step in range(400):
            with _at_train_step(step):
                ks.append(mtp0._sample_mtp_depth())
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
        loss = _run_step(model, cfg, self.strategy)
        assert loss is not None and not paddle.isnan(loss).any(), "loss NaN"
        assert not hasattr(cfg, "_mtp_sampled_depth"), (
            "no config-level sampling state should exist"
        )

    def _lm_head(self, model):
        return next(
            layer
            for layer in model.run_function
            if type(layer).__name__ == "GPTLMHead"
        )

    def test_lm_head_derives_the_same_k_as_the_depths(self):
        """The head must land on the depths' K even with nothing published.

        This is the property that removed the co-location requirement: the head
        may sit on a different pipeline chunk from the MTP depths, where dict_args
        never reaches it, so it re-derives K from (seed, step) instead. Driven
        with an empty dict to model exactly that split, and compared against what
        the depths would compute for the same step. Falling back to D here would
        project the slices the depths skipped and average the loss over D -- a
        silently wrong run, not a crash.
        """
        cfg = self._cfg([0.0, 1.0, 0.0])
        head = self._lm_head(gpt_builder(cfg, num_stages=1))
        hidden = paddle.zeros([cfg.num_nextn_predict_layers + 1, 8])
        with (
            mock.patch.object(head, "_forward", side_effect=lambda t: t),
            _at_train_step(5),
        ):
            logits = head.forward({"hidden_states": hidden})
            expected_k = draw_mtp_sampled_depth(cfg)

        self.assertEqual(expected_k, 2)
        self.assertEqual(len(logits), cfg.num_nextn_predict_layers + 1)
        computed = [i for i in range(1, len(logits)) if logits[i] is not None]
        self.assertEqual(computed, list(range(1, expected_k + 1)))

    def test_lm_head_keeps_no_sampling_state(self):
        """No counters anywhere: the head derives or reads, never accumulates.

        A counter reappearing on the head is a regression to the old per-call
        stream, whose index had to stay aligned with the MTP depths' -- the
        construct that could drift under VPP / p2p overlap and deadlock.
        """
        cfg = self._cfg([0.34, 0.33, 0.33])
        model = gpt_builder(cfg, num_stages=1)
        head = self._lm_head(model)
        _run_step(model, cfg, self.strategy)
        assert not hasattr(head, "_mtp_sampling_counter"), (
            "the LM head must derive K from (seed, step), never accumulate state"
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

    def test_sampling_is_idempotent_under_recompute(self):
        """A recompute replay must reuse the forward's K, not draw a new one.

        Otherwise the backward would be computed for a different set of depths
        than the forward ran -- silently wrong gradients rather than a crash. With
        K a pure function of (seed, train step) this holds by construction: the
        replay happens inside the same step, so it re-derives the same value even
        when it gets a fresh dict_args. The depth counts are the observable -- the
        step must skip exactly the depths P(K=2)=1 excludes, however many times
        forward is entered.
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
        body_calls = _count_mtp_body_calls(model)

        loss = _run_step(model, config, self.strategy)
        assert loss is not None and not paddle.isnan(loss).any(), (
            "recompute + sampling step did not produce a usable loss"
        )
        assert body_calls.get(0, 0) > 0 and body_calls.get(1, 0) > 0, (
            f"P(K=2)=1 must run depths 0 and 1, body_calls={body_calls}"
        )
        assert body_calls.get(2, 0) == 0, (
            f"P(K=2)=1 must skip depth 2, body_calls={body_calls}"
        )


if __name__ == "__main__":
    unittest.main()
