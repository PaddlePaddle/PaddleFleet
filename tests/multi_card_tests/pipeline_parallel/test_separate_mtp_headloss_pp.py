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

# Regression test for the separate_mtp_headloss PP/VPP layout contract.
#
# It builds the real GPT pipeline model under PP=4/VPP=2 with
# separate_mtp_headloss enabled through the NORMAL configuration path (no manual
# post-construction override), i.e. GPTConfig.__post_init__ must keep the flag
# on by itself. The geometry therefore satisfies the guard contract:
#   num_hidden(5) + num_mtp(1) + head_empty(0) + effective_tail_empty(2) = 8
#   = pp*vpp, i.e. exactly one seg-weight-bearing layer per virtual stage.
# The seg_method mirrors what the production entry uses when the flag is on
# (PaddleFormers gpt_provider.py): MultiTokenPredictionLayer also carries a
# segmentation weight, which is why the guard counts the MTP layers.
#
# Scope: this exercises config validation + model construction + Paddle's
# interleave segmentation + _construct_shared_comm (called inside
# PipelineLayer.__init__). It intentionally does NOT run forward/backward.

import functools
import unittest

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import SharedLayerDesc

from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig
from paddlefleet.training.initialize import initialize_fleet

PP_DEGREE = 4
VPP_DEGREE = 2
MTP_DEGREE = 1
# separate_mtp_headloss adds MultiTokenPredictionLayer to the segmentation
# weights, matching the production entry point.
SEG_METHOD = "layer:TransformerLayer|EmptyLayer|MultiTokenPredictionLayer"


def _build_config(num_empty_layers_add_in_head=0):
    config = GPTConfig(
        moe_expert_fusion=False,
        vocab_size=1024,
        max_sequence_length=128,
        num_hidden_layers=5,
        hidden_size=512,
        num_attention_heads=4,
        intermediate_size=1024,
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
        # seg-weight layers = num_hidden(5) + mtp(1) + head(0) + tail_eff(2) = 8
        # = pp*vpp, so the __post_init__ guard keeps separate_mtp_headloss on.
        num_empty_layers_add_in_head=num_empty_layers_add_in_head,
        num_empty_layers_add_in_tail=3,
        pipeline_model_parallel_size=PP_DEGREE,
        virtual_pipeline_model_parallel_size=VPP_DEGREE,
        tensor_model_parallel_size=2,
        expert_model_parallel_size=2,
        sequence_parallel=True,
        n_shared_experts=1,
        n_routed_experts=8,
        moe_intermediate_size=1024,
        bf16=True,
        gated_linear_unit=True,
        bias_activation_fusion=True,
        num_nextn_predict_layers=MTP_DEGREE,
        separate_mtp_headloss=True,
    )
    return config


class TestSeparateMtpHeadlossPP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # initialize_fleet must run exactly once per process, so it cannot live
        # in setUp once this class holds more than one test.
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 2,
            "pp_degree": PP_DEGREE,
            "sharding_degree": 1,
            "sep_degree": 1,
            "cp_degree": 1,
            "ep_degree": 2,
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
            "accumulate_steps": 8,
            "micro_batch_size": 1,
        }
        initialize_fleet(strategy)

    def test_build_segmentation_and_shared_layer_stages(self):
        config = _build_config()
        # No manual override: GPTConfig.__post_init__ itself must accept this
        # layout, otherwise the guard contract rejects every normal config.
        self.assertTrue(config.separate_mtp_headloss)

        gpt_model = gpt_builder(
            config,
            num_stages=PP_DEGREE,
            seg_method=SEG_METHOD,
        )

        # _construct_shared_comm runs inside PipelineLayer.__init__; a successful
        # build means segmentation + shared-comm construction did not raise.
        self.assertTrue(hasattr(gpt_model, "shared_comm"))

        # segment_parts partitions all layer descs into pp*vpp virtual stages.
        seg = gpt_model.segment_parts
        self.assertEqual(len(seg) - 1, PP_DEGREE * VPP_DEGREE)

        # quotient == 1: pp*vpp virtual stages for 8 seg-weight layers, so no
        # virtual stage may hold more than one of them.
        span = [seg[i + 1] - seg[i] for i in range(len(seg) - 1)]
        self.assertEqual(sum(span), len(gpt_model._layers_desc))
        self.assertTrue(all(s >= 1 for s in span))

        # Shared "embed" descs: front embedding + separated MTP-LMHead +
        # separated main-LMHead. There must be >= 3 of them.
        embed_idxs = [
            idx
            for idx, layer in enumerate(gpt_model._layers_desc)
            if isinstance(layer, SharedLayerDesc)
            and layer.layer_name == "embed"
        ]
        self.assertGreaterEqual(len(embed_idxs), 3)

        # Front embedding lands on the first pp stage and the separated
        # main-LMHead on the last one. The separated MTP-LMHead sits between the
        # MTP layers and the remaining tail EmptyLayers, so its stage is decided
        # by that layout: it must be downstream of the embedding and no later
        # than the main head, and the two shared heads must reach a pp stage
        # different from stage 0 so the embedding weight is actually communicated.
        stages = {
            idx: gpt_model.get_stage_from_index(idx) for idx in embed_idxs
        }
        self.assertEqual(stages[embed_idxs[0]], 0)
        self.assertEqual(stages[embed_idxs[-1]], PP_DEGREE - 1)
        self.assertGreater(stages[embed_idxs[-2]], 0)
        self.assertLessEqual(stages[embed_idxs[-2]], stages[embed_idxs[-1]])

    def test_guard_disables_layout_with_non_unit_quotient(self):
        # Negative counterpart: proves the positive assertion above is not
        # vacuous, i.e. __post_init__ really evaluates the guard. head=1 makes
        # the seg-weight layer count 5 + 1 + (1 + 2) = 9, which is neither
        # divisible by pp*vpp=8 nor a quotient of 1, so the flag must be
        # force-disabled during construction.
        with self.assertWarns(UserWarning):
            config = _build_config(num_empty_layers_add_in_head=1)
        self.assertFalse(config.separate_mtp_headloss)


if __name__ == "__main__":
    unittest.main()
