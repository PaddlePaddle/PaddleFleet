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
# separate_mtp_headloss enabled and a geometry whose seg-weight-bearing layer
# count (num_hidden + head_empty + effective_tail_empty = 11 + 3 + 2 = 16) is
# divisible by pp*vpp=8 with quotient 2 (i.e. >1 layer per stage). It then
# asserts on the pipeline segmentation and the shared-layer (embed / separated
# MTP-LMHead / separated main-LMHead) stage placement.
#
# Scope: this exercises model construction + Paddle's interleave segmentation +
# _construct_shared_comm (called inside PipelineLayer.__init__). It intentionally
# does NOT run forward/backward: the full training dataflow for
# separate_mtp_headloss under pp>1 is not covered by any existing test and is out
# of scope here. The purpose is to prove that "exactly 1 layer per pp*vpp stage"
# is NOT required for the pipeline/shared-comm machinery — a divisible layout with
# quotient > 1 builds and constructs shared comm successfully.

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
MTP_DEGREE = 3


def _build_config():
    config = GPTConfig(
        moe_expert_fusion=False,
        vocab_size=1024,
        max_sequence_length=128,
        num_hidden_layers=11,
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
        # head=3 => seg-weight layers = num_hidden(11) + head(3) + tail_eff(2) = 16,
        # divisible by pp*vpp=8 with quotient 2 (> 1 layer per stage).
        num_empty_layers_add_in_head=3,
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
    )
    # The __post_init__ pre-check currently force-disables separate_mtp_headloss
    # unless quotient == 1. We deliberately re-enable it here to exercise the
    # real build/segmentation/shared-comm path at quotient > 1.
    config.separate_mtp_headloss = True
    return config


class TestSeparateMtpHeadlossPP(unittest.TestCase):
    def setUp(self):
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
        self.assertTrue(config.separate_mtp_headloss)

        gpt_model = gpt_builder(
            config,
            num_stages=PP_DEGREE,
            seg_method="layer:TransformerLayer|EmptyLayer",
        )

        # _construct_shared_comm runs inside PipelineLayer.__init__; a successful
        # build means segmentation + shared-comm construction did not raise.
        self.assertTrue(hasattr(gpt_model, "shared_comm"))

        # segment_parts partitions all layer descs into pp*vpp virtual stages.
        seg = gpt_model.segment_parts
        self.assertEqual(len(seg) - 1, PP_DEGREE * VPP_DEGREE)

        # quotient > 1: at least one virtual stage holds more than one layer.
        span = [seg[i + 1] - seg[i] for i in range(len(seg) - 1)]
        self.assertTrue(max(span) > 1)

        # Shared "embed" descs: front embedding + separated MTP-LMHead +
        # separated main-LMHead. There must be >= 3 of them.
        embed_idxs = [
            idx
            for idx, layer in enumerate(gpt_model._layers_desc)
            if isinstance(layer, SharedLayerDesc)
            and layer.layer_name == "embed"
        ]
        self.assertGreaterEqual(len(embed_idxs), 3)

        # Front embedding lands on the first pp stage; the two separated heads
        # land on the last pp stage. Multiple shared layers legitimately share a
        # stage -> "exactly 1 layer per stage" is not required.
        stages = {
            idx: gpt_model.get_stage_from_index(idx) for idx in embed_idxs
        }
        self.assertEqual(stages[embed_idxs[0]], 0)
        self.assertEqual(stages[embed_idxs[-1]], PP_DEGREE - 1)
        self.assertEqual(stages[embed_idxs[-2]], PP_DEGREE - 1)


if __name__ == "__main__":
    unittest.main()
