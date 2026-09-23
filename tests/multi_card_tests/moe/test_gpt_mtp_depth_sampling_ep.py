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

"""mtp_depth_sampling under expert parallelism.

Every MoE layer of an MTP depth issues an EP all-to-all, so all EP ranks must run
exactly the same depths on every micro-batch, although each rank feeds different
tokens. On a GPT model with EP=2 this checks that:

  * the sampled K sequence is identical on every rank and actually varies;
  * with K fixed, the loss is the LM loss plus the scaled average of the first K
    per-depth losses of the unsampled model (same weights, same data);
  * optimizer steps through the hybrid-parallel optimizer keep the replicated
    (non-expert) parameters identical across ranks, and leave the parameters of
    the sampled-out depths untouched;
  * the same holds with mtp_shared_weights + mtp_shared_last_layer, where the
    shared parameters keep their ties and are trained by every sampled K.

Runs at any world size (EP = world size); CI runs it on 2 GPUs.
"""

import functools
import hashlib
import unittest

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.common.language_loss.language_loss import (
    LanguageLoss,
)
from paddlefleet.models.gpt import GPTConfig
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer.multi_token_prediction import (
    MultiTokenPredictionLayer,
)

VOCAB = 128
SEQ = 64
NUM_MTP = 3
ACC_STEPS = 4
SEED = 46
MTP_SCALE = 0.3

EP_SIZE = None
STRATEGY = None


def setUpModule():
    global EP_SIZE, STRATEGY
    EP_SIZE = dist.get_world_size()
    STRATEGY = fleet.DistributedStrategy()
    STRATEGY.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": EP_SIZE,
        "sep_degree": 1,
        "cp_degree": 1,
        "ep_degree": EP_SIZE,
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
    STRATEGY.pipeline_configs = {
        "accumulate_steps": ACC_STEPS,
        "micro_batch_size": 1,
    }
    initialize_fleet(STRATEGY)
    paddle.seed(SEED)
    model_parallel_cuda_manual_seed(SEED)


def _make_config(mtp_depth_sampling, **extra):
    return GPTConfig(
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        num_hidden_layers=2,
        hidden_size=256,
        num_attention_heads=4,
        intermediate_size=512,
        normalization="RMSNorm",
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=True,
        parallel_output=True,
        tie_word_embeddings=True,
        position_embedding_type="rope",
        rotary_percent=1.0,
        rotary_base=10000,
        rope_scaling=1.0,
        init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        output_layer_init_method=functools.partial(
            paddle.nn.init.xavier_uniform_, gain=1.0
        ),
        use_qk_norm=True,
        pipeline_model_parallel_size=1,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=EP_SIZE,
        moe_token_dispatcher_type="alltoall",
        moe_expert_fusion=False,
        n_shared_experts=1,
        n_routed_experts=8,
        moe_intermediate_size=512,
        gated_linear_unit=True,
        num_nextn_predict_layers=NUM_MTP,
        mtp_loss_scaling_factor=MTP_SCALE,
        mtp_depth_sampling=mtp_depth_sampling,
        **extra,
    )


def _build(mtp_depth_sampling, state_dict=None, **extra):
    paddle.seed(SEED)
    model = gpt_builder(_make_config(mtp_depth_sampling, **extra), num_stages=1)
    if state_dict is not None:
        model.set_state_dict(state_dict)
    return model


def _batch(same_micro_batches):
    """ACC_STEPS micro-batches; different tokens on every EP rank."""
    rng = np.random.default_rng(SEED + 1000 * dist.get_rank())
    n = 1 if same_micro_batches else ACC_STEPS
    data = [
        paddle.to_tensor(rng.integers(0, VOCAB, (1, SEQ + NUM_MTP + 1)))
        for _ in range(n)
    ]
    if same_micro_batches:
        data = data * ACC_STEPS
    position_ids = paddle.arange(SEQ + NUM_MTP).reshape([1, -1])
    return (
        {
            "input_ids": [d[:, :-1] for d in data],
            "position_ids": [position_ids] * ACC_STEPS,
        },
        [d[:, 1:] for d in data],
    )


def _mtp_layers(model):
    return [
        la
        for la in model.run_function
        if isinstance(la, MultiTokenPredictionLayer)
    ]


def _record_sampled_depth(model):
    """Record the K drawn for every micro-batch and the depths that ran."""
    ks, ran = [], []
    layers = _mtp_layers(model)
    assert len(layers) == NUM_MTP

    def _start_micro_batch(_layer, _inputs):
        ran.append(set())

    def _k_hook(_layer, inputs, _out):
        k = int(inputs[0]["mtp_sampled_depth"])
        ks.append(k)
        # Checked before depth 1 runs: a mismatch would otherwise deadlock in
        # that depth's EP all-to-all instead of failing. Every rank sees the
        # same gathered list, so all of them raise together.
        gathered = _all_gather(k)
        if len(set(gathered)) != 1:
            raise AssertionError(f"K differs across EP ranks: {gathered}")

    layers[0].register_forward_pre_hook(_start_micro_batch)
    layers[0].register_forward_post_hook(_k_hook)
    for layer in layers:

        def _body_hook(_mod, _inp, _depth=layer.layer_number):
            ran[-1].add(_depth)

        layer.transformer_layer.register_forward_pre_hook(_body_hook)
    return ks, ran


def _all_gather(obj):
    out = []
    dist.all_gather_object(out, obj)
    return out


def _assert_on_all_ranks(testcase, errors):
    """Fail on every rank if any rank failed, so no rank hangs in a collective."""
    all_errors = [e for errs in _all_gather(errors) for e in errs]
    testcase.assertFalse(all_errors, "\n".join(all_errors))


def _expert_param_names(model):
    return {n for n, p in model.named_parameters() if ".experts." in n}


class TestMTPDepthSamplingEP(unittest.TestCase):
    def test_sampled_depth_is_rank_consistent(self):
        model = _build([0.4, 0.3, 0.3])
        ks, ran = _record_sampled_depth(model)
        pipe = fleet.distributed_model(model)
        losses = []
        for _ in range(3):
            losses.append(
                float(pipe.forward_backward_pipeline(_batch(False), None))
            )

        errors = []
        if not all(np.isfinite(losses)):
            errors.append(f"rank {dist.get_rank()}: non-finite loss {losses}")
        if len(ks) != 3 * ACC_STEPS:
            errors.append(f"rank {dist.get_rank()}: {len(ks)} K draws")
        for k, depths in zip(ks, ran):
            if depths != set(range(k)):
                errors.append(
                    f"rank {dist.get_rank()}: K={k} but depths {depths} ran"
                )
        gathered = _all_gather(ks)
        if any(g != gathered[0] for g in gathered):
            errors.append(f"K differs across EP ranks: {gathered}")
        if len(set(ks)) < 2:
            errors.append(f"K never varied: {ks}")
        _assert_on_all_ranks(self, errors)
        print(f"[MTP-SAMPLING-EP] ep={EP_SIZE} K={ks}", flush=True)

    def test_fixed_k_loss_is_prefix_average(self):
        ref = _build(None)
        state = {k: v.clone() for k, v in ref.state_dict().items()}
        loss_full = float(
            fleet.distributed_model(ref).forward_backward_pipeline(
                _batch(True), None
            )
        )
        per_depth = [
            float(LanguageLoss.mtp_loss_tracker[f"mtp_{i + 1}_loss"])
            for i in range(NUM_MTP)
        ]
        lm_loss = loss_full - MTP_SCALE * sum(per_depth) / NUM_MTP

        errors = []
        for k in (1, 2):
            dist_k = [1.0 if i == k - 1 else 0.0 for i in range(NUM_MTP)]
            model = _build(dist_k, state)
            loss_k = float(
                fleet.distributed_model(model).forward_backward_pipeline(
                    _batch(True), None
                )
            )
            want = lm_loss + MTP_SCALE * sum(per_depth[:k]) / k
            if not np.isclose(loss_k, want, rtol=1e-5, atol=1e-6):
                errors.append(
                    f"rank {dist.get_rank()} K={k}: loss {loss_k} != "
                    f"lm + prefix average {want} (per_depth={per_depth})"
                )
            stale = [
                key
                for key in LanguageLoss.mtp_loss_tracker
                if int(key.split("_")[1]) > k
            ]
            if stale:
                errors.append(f"K={k}: skipped depths still tracked {stale}")
        _assert_on_all_ranks(self, errors)

    def test_optimizer_steps_under_ep(self):
        model = _build([1.0, 0.0, 0.0])
        expert_names = _expert_param_names(model)
        self.assertTrue(expert_names, "model has no expert parameters")
        before = {n: p.numpy().copy() for n, p in model.named_parameters()}
        pipe = fleet.distributed_model(model)
        opt = fleet.distributed_optimizer(
            paddle.optimizer.SGD(
                learning_rate=0.1, parameters=model.parameters()
            )
        )
        losses = [float(pipe.train_batch(_batch(False), opt)) for _ in range(3)]

        errors = []
        if not all(np.isfinite(losses)):
            errors.append(f"rank {dist.get_rank()}: non-finite loss {losses}")
        after = {n: p.numpy() for n, p in model.named_parameters()}
        changed = {n for n in after if not np.array_equal(after[n], before[n])}
        body_names = {
            depth: {
                model_name
                for model_name, p in model.named_parameters()
                if any(p is q for _, q in la.named_parameters())
            }
            for depth, la in enumerate(_mtp_layers(model))
        }
        if not body_names[0] & changed:
            errors.append("depth 0 (always run) did not train")
        for depth in range(1, NUM_MTP):
            moved = sorted(body_names[depth] & changed)
            if moved:
                errors.append(f"sampled-out depth {depth} changed: {moved[:3]}")

        replicated = sorted(n for n in after if n not in expert_names)
        gathered = _all_gather(
            {n: hashlib.md5(after[n].tobytes()).hexdigest() for n in replicated}
        )
        for n in replicated:
            if len({g[n] for g in gathered}) != 1:
                errors.append(f"replicated param {n} diverged across ranks")
        if not (changed - expert_names):
            errors.append("no replicated parameter was updated")
        _assert_on_all_ranks(self, errors)
        print(
            f"[MTP-SAMPLING-EP] ep={EP_SIZE} train losses={losses}", flush=True
        )

    def test_combined_sharing_with_sampling(self):
        model = _build(
            [0.6, 0.3, 0.1],
            mtp_shared_weights=True,
            mtp_shared_last_layer=True,
        )
        ks, _ = _record_sampled_depth(model)
        layers = _mtp_layers(model)
        ties = [[p for _, p in la.named_parameters()] for la in layers]
        expert_names = _expert_param_names(model)
        before = {n: p.numpy().copy() for n, p in model.named_parameters()}
        pipe = fleet.distributed_model(model)
        opt = fleet.distributed_optimizer(
            paddle.optimizer.SGD(
                learning_rate=0.1, parameters=model.parameters()
            )
        )
        losses = [float(pipe.train_batch(_batch(False), opt)) for _ in range(3)]

        errors = []
        if not all(np.isfinite(losses)):
            errors.append(f"rank {dist.get_rank()}: non-finite loss {losses}")
        if len(set(ks)) < 2:
            errors.append(f"K never varied: {ks}")
        for depth in range(1, NUM_MTP):
            for p0, pd in zip(ties[0], ties[depth]):
                if p0 is not pd:
                    errors.append(f"depth {depth} lost its tie: {pd.name}")
                    break
        after = {n: p.numpy() for n, p in model.named_parameters()}
        shared_changed = [
            n
            for n, p in model.named_parameters()
            if any(p is q for q in ties[0])
            and not np.array_equal(after[n], before[n])
        ]
        if not shared_changed:
            errors.append("shared MTP parameters were not trained")
        replicated = sorted(n for n in after if n not in expert_names)
        gathered = _all_gather(
            {n: hashlib.md5(after[n].tobytes()).hexdigest() for n in replicated}
        )
        for n in replicated:
            if len({g[n] for g in gathered}) != 1:
                errors.append(f"replicated param {n} diverged across ranks")
        _assert_on_all_ranks(self, errors)
        print(
            f"[MTP-SAMPLING-EP] ep={EP_SIZE} combined sharing K={ks} "
            f"losses={losses}",
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
