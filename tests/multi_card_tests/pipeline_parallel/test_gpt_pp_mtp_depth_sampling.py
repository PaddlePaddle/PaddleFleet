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

"""Multi-card (pipeline-parallel) regression test for mtp_depth_sampling.

The single-card TestMTPDepthSampling cannot catch the pp>1 failure mode: the
MTP layer only runs on the LAST pipeline stage, so any collective it issues
(the earlier broadcast(src=0) K-sync) is joined by no other stage and
deadlocks. The collective-free sampler must let a pp>1 forward_backward
complete. This test builds a small MoE+MTP model under pp=2 and asserts the
step finishes with a finite loss for:
  * sampling disabled (None)            -> baseline
  * fixed K=1                           -> exercises the skip path (depths>=1)
  * mixed distribution (varying K)      -> K changes per micro-batch, must stay
                                           rank-consistent (else MoE all-to-all
                                           on the last stage would deadlock)
  * mtp_shared_weights + sampling       -> the combination the config layer allows

Each case also counts transformer_layer invocations per MTP depth, so a
regression that stopped skipping is caught: the old assertions only checked that
the loss was finite, which stays true when every depth runs.
"""

import functools
import os
import unittest

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet import distributed_model

from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig
from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer.multi_token_prediction import (
    MultiTokenPredictionLayer,
)

PP_DEGREE = 2
MTP_DEGREE = 3
REPO_FLAG = os.getenv("repo_flag")
SKIP_TESTS = REPO_FLAG != "paddlefleet"


def _run_pp(mtp_depth_sampling, seed=46, **extra_config):
    config = GPTConfig(
        moe_expert_fusion=False,
        vocab_size=128,
        max_sequence_length=64,
        num_hidden_layers=4,
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
        pipeline_model_parallel_size=PP_DEGREE,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        n_shared_experts=1,
        n_routed_experts=8,
        moe_intermediate_size=512,
        gated_linear_unit=True,
        num_nextn_predict_layers=MTP_DEGREE,
        mtp_depth_sampling=mtp_depth_sampling,
        **extra_config,
    )

    micro = 1
    num_acc = 2
    np.random.seed(seed)
    paddle.seed(seed)

    model = gpt_builder(
        config,
        num_stages=PP_DEGREE,
        seg_method="layer:TransformerLayer|EmptyLayer",
    )
    pipe = distributed_model(model)

    # Count transformer_layer invocations per MTP depth so the tests can observe
    # that depths >= K were actually skipped, not merely that the step survived.
    # A forward pre-hook is used rather than replacing layer.transformer_layer:
    # paddle's Layer.__setattr__ raises TypeError when a name already bound to a
    # sublayer is assigned a non-Layer value.
    mtp_layers = [
        layer
        for layer in model.run_function
        if isinstance(layer, MultiTokenPredictionLayer)
    ]
    body_calls = {}
    for layer in mtp_layers:
        body_calls[layer.layer_number] = 0

        def _count(_mod, _inp, _depth=layer.layer_number):
            body_calls[_depth] += 1

        layer.transformer_layer.register_forward_pre_hook(_count)

    data = paddle.randint(low=0, high=128, shape=(micro, 64 + MTP_DEGREE + 1))
    input_ids = data[:, :-1]
    labels = data[:, 1:]
    position_ids = paddle.arange(input_ids.shape[1]).reshape([1, -1])
    inputs = (
        {
            "input_ids": [input_ids] * num_acc,
            "position_ids": [position_ids] * num_acc,
        },
        [labels] * num_acc,
    )
    loss = pipe.forward_backward_pipeline(inputs, None)
    return loss, mtp_layers, body_calls


@unittest.skipIf(SKIP_TESTS, "requires repo_flag=paddlefleet multi-card env")
class TestMTPDepthSamplingPP(unittest.TestCase):
    """pp=2 forward_backward must COMPLETE (no collective deadlock)."""

    @classmethod
    def setUpClass(cls):
        # fleet / parallel state can only be initialized once per process, so
        # do it here (pp=2, mp=1, ep=1 is identical for every test below); each
        # test only rebuilds the model with different mtp_depth_sampling config.
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": PP_DEGREE,
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
            "pp_configs": {
                "overlap_p2p_comm": True,
                "enable_dynamic_shape": True,
            },
        }
        strategy.pipeline_configs = {
            "accumulate_steps": 2,
            "micro_batch_size": 1,
        }
        initialize_fleet(strategy)

    def _assert_finite(self, loss):
        assert loss is not None, "loss is None"
        assert not paddle.isnan(loss).any(), "loss has NaN"
        assert not paddle.isinf(loss).any(), "loss has Inf"

    def test_pp_sampling_disabled(self):
        loss, mtp_layers, body_calls = _run_pp(None)
        self._assert_finite(loss)
        # Baseline: with sampling off every depth this rank holds must run.
        if mtp_layers:
            assert all(n > 0 for n in body_calls.values()), (
                f"sampling off must run every depth, body_calls={body_calls}"
            )

    def test_pp_sampling_fixed_k1(self):
        # Depths >= 1 skipped on the last stage every step; the collective-free
        # sampler must not hang (the old broadcast(src=0) deadlocked here).
        loss, mtp_layers, body_calls = _run_pp([1.0, 0.0, 0.0])
        self._assert_finite(loss)
        if not mtp_layers:
            return
        # The step surviving is not enough: a regression that stopped skipping
        # would also produce a finite loss. Assert the skip actually happened.
        assert body_calls.get(0, 0) > 0, (
            f"depth 0 must always run, body_calls={body_calls}"
        )
        skipped = {d: n for d, n in body_calls.items() if d >= 1}
        assert all(n == 0 for n in skipped.values()), (
            f"K=1 must skip every depth >= 1, body_calls={body_calls}"
        )
        depth0 = next((la for la in mtp_layers if la.layer_number == 0), None)
        assert depth0 is not None, (
            "co-location is enforced for sampling, so a rank holding MTP layers "
            f"must hold depth 0; got {sorted(body_calls)}"
        )
        assert getattr(depth0, "_last_sampled_depth", None) == 1, (
            f"expected K=1, got {getattr(depth0, '_last_sampled_depth', None)}"
        )

    def test_pp_sampling_mixed(self):
        # K varies per micro-batch; every rank running the MTP layer must draw
        # the same K deterministically or the MoE all-to-all deadlocks.
        loss, mtp_layers, body_calls = _run_pp([0.34, 0.33, 0.33])
        self._assert_finite(loss)
        if mtp_layers:
            assert body_calls.get(0, 0) > 0, (
                f"depth 0 must always run, body_calls={body_calls}"
            )

    def test_pp_sampling_with_shared_weights(self):
        """The combination the config layer now allows: cross-depth full sharing
        plus sampling. All MTP depths stay on one stage, which
        _assert_mtp_depths_colocated_for_sampling requires."""
        loss, mtp_layers, body_calls = _run_pp(
            [1.0, 0.0, 0.0], mtp_shared_weights=True
        )
        self._assert_finite(loss)
        if not mtp_layers:
            return
        skipped = {d: n for d, n in body_calls.items() if d >= 1}
        assert all(n == 0 for n in skipped.values()), (
            f"K=1 must skip every depth >= 1, body_calls={body_calls}"
        )
        # Sharing must still hold: depth 1 sees depth 0's parameters.
        if len(mtp_layers) >= 2:
            d0 = dict(mtp_layers[0].all_weights)
            not_shared = [
                n for n, p in mtp_layers[1].all_weights if d0.get(n) is not p
            ]
            assert not not_shared, (
                f"depths must share all_weights, not_shared={not_shared[:5]}"
            )


if __name__ == "__main__":
    unittest.main()
