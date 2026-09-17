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

"""PP2/CP2 invariant test for Erndata MTP magic-send embeddings.

Run with four GPUs::

    python -m paddle.distributed.launch --gpus=0,1,2,3 \
        tests/multi_card_tests/pipeline_parallel/\
        test_gpt_pp_mtp_magic_cp_weight_sync.py

Every CP coordinate owns two physical copies of the logical token embedding:
the stage-0 ``GPTEmbedding`` weight and the MTP-stage ``mtp_embed`` weight.
They are initialized by broadcast and receive a dedicated PP all-reduce for
their gradients.  Because Erndata shards embeddings with a plain local slice,
both copies must also receive the trainer's CP-world-size gradient scaling.
This test checks equality at initialization, after gradient synchronization and
CP scaling, and after every optimizer step.  The old bug, where only the MTP
copy had ``context_parallel_disable_scale_grad=True``, fails on the first step.
"""

import functools
import os
import unittest

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

import paddlefleet
from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig

PP_SIZE = 2
CP_SIZE = 2
WORLD_SIZE = PP_SIZE * CP_SIZE
VOCAB_SIZE = 4096
HIDDEN_SIZE = 512
NUM_STEPS = 10


def setUpModule():
    if "FLAGS_selected_gpus" not in os.environ:
        raise unittest.SkipTest("requires paddle.distributed.launch")

    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": PP_SIZE,
        # Paddle's expert-aware HCG represents CP as an overlapping axis;
        # matching sharding/EP degrees selects that topology (see the existing
        # test_gpt_mtp_megatron_cp.py coverage).
        "sharding_degree": CP_SIZE,
        "sep_degree": 1,
        "cp_degree": CP_SIZE,
        "ep_degree": CP_SIZE,
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
    if dist.get_world_size() != WORLD_SIZE:
        raise unittest.SkipTest(
            f"requires {WORLD_SIZE} ranks (PP={PP_SIZE}, CP={CP_SIZE}), "
            f"got {dist.get_world_size()}"
        )


def _make_config():
    return GPTConfig(
        vocab_size=VOCAB_SIZE,
        max_sequence_length=16,
        num_hidden_layers=2,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=2048,
        normalization="RMSNorm",
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=True,
        parallel_output=True,
        tie_word_embeddings=False,
        position_embedding_type="rope",
        rotary_percent=1.0,
        rotary_base=10000,
        rope_scaling=1.0,
        gated_linear_unit=True,
        num_nextn_predict_layers=1,
        mtp_loss_scaling_factor=0.3,
        use_erndata=True,
        enable_mtp_magic_send=True,
        variable_seq_lengths=True,
        context_parallel_size=CP_SIZE,
        cp_balance_mode="dualchunk_allgather",
        experimental_dataflow=False,
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=PP_SIZE,
        bf16=False,
        init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        output_layer_init_method=functools.partial(
            paddle.nn.init.xavier_uniform_, gain=1.0
        ),
    )


def _grad_for(param):
    if hasattr(param, "main_grad") and param.main_grad is not None:
        return param.main_grad
    return param.grad


class TestErndataMagicSendCPWeightSync(unittest.TestCase):
    def _assert_pp_copies_equal(self, weight, phase):
        pipe_group = fleet.get_hybrid_communicate_group().get_pipe_parallel_group()
        copies = []
        dist.all_gather(copies, weight.detach(), group=pipe_group)
        self.assertEqual(len(copies), PP_SIZE)
        for other in copies[1:]:
            np.testing.assert_array_equal(
                copies[0].astype("float32").numpy(),
                other.astype("float32").numpy(),
                err_msg=f"stage-0 and MTP embedding diverged {phase}",
            )

    def test_embedding_weights_stay_equal_across_steps(self):
        paddle.seed(2026)
        model = gpt_builder(
            _make_config(),
            num_stages=PP_SIZE,
            seg_method="layer:TransformerLayer|EmptyLayer",
        )
        weight = model._get_mtp_embed_primary_weight()
        self.assertIsNotNone(weight)

        # Erndata uses extract_local_cp_chunks (plain slicing), so both physical
        # PP copies need the trainer's CP multiplier; neither may opt out.
        self.assertFalse(
            getattr(weight, "context_parallel_disable_scale_grad", False)
        )
        self._assert_pp_copies_equal(weight, "after initialization")

        optimizer = paddle.optimizer.SGD(
            learning_rate=0.01, parameters=model.parameters()
        )
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()

        for step in range(NUM_STEPS):
            # Give the two PP copies different local contributions.  The model's
            # dedicated MTP PP all-reduce must first make their gradients equal.
            local_value = float((step + 1) * (pp_rank + 1))
            weight.grad = paddle.full_like(weight, local_value)
            model.allreduce_shared_weight_gradients()

            grad = _grad_for(weight)
            self.assertIsNotNone(grad)
            grad_copies = []
            pipe_group = hcg.get_pipe_parallel_group()
            dist.all_gather(grad_copies, grad.detach(), group=pipe_group)
            np.testing.assert_array_equal(
                grad_copies[0].astype("float32").numpy(),
                grad_copies[1].astype("float32").numpy(),
                err_msg=f"embedding gradients differ after PP all-reduce at step {step}",
            )

            # Mirror Trainer.hybrid_parallel_scale_param_grad.  This is the exact
            # operation whose asymmetric marker caused the reviewed CP bug.
            if not getattr(weight, "context_parallel_disable_scale_grad", False):
                with paddle.no_grad():
                    grad.scale_(CP_SIZE)

            scaled_copies = []
            dist.all_gather(scaled_copies, grad.detach(), group=pipe_group)
            np.testing.assert_array_equal(
                scaled_copies[0].astype("float32").numpy(),
                scaled_copies[1].astype("float32").numpy(),
                err_msg=f"embedding gradients differ after CP scaling at step {step}",
            )

            optimizer.step()
            optimizer.clear_grad()
            self._assert_pp_copies_equal(weight, f"after optimizer step {step}")


if __name__ == "__main__":
    unittest.main()
