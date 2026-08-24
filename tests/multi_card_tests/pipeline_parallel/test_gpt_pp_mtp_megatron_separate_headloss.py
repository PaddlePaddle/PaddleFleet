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

"""PP>1 + separate_mtp_headloss=True + use_erndata=True end-to-end.

Companion to ``test_gpt_pp_mtp_megatron.py`` (fused GPTLMHead + LanguageLoss).
This variant exercises the SEPARATE head/loss path:
  * ``GPTMTPLMHead`` runs on the last stage and stashes cu_seqlens_q;
  * ``MTPLanguageLoss.forward`` rolls per-depth labels per packed doc via
    ``_megatron_label_for_depth`` (pad_value=ignored_index);
  * ``MainLanguageLoss.forward`` keeps the main label full-length L.

Layout (guarded by GPTConfig.__post_init__ for separate_mtp_headloss):
  total = num_hidden(1) + mtp(1) + num_empty(head0 + max(0,tail3-1)=2) = 4
        = pp(4) * vpp(1)  -> exactly one seg-weight layer per virtual stage.
separate_mtp_headloss needs tail>=3 (reserves tail slots for the separated
MTP-LMHead / main-LMHead) and pp>2 is only required when vpp>1, so this uses
vpp=1 with pp=4 (4 GPUs, no interleave). seg_method includes
MultiTokenPredictionLayer so the MTP layer carries a segmentation weight,
matching the production entry point.

Megatron data contract: length-L tensors + a single int32 cu_seqlens_q. A
finite loss + clean exit proves the last-stage stash reached both the MTP and
main losses (otherwise ``_megatron_label_for_depth`` raises RuntimeError).
"""

import functools
import random
import unittest

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet import distributed_model

import paddlefleet
from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig
from paddlefleet.training.initialize import initialize_fleet

PP_DEGREE = 4
VPP_DEGREE = 1
MTP_DEGREE = 1
SEG_METHOD = "layer:TransformerLayer|EmptyLayer|MultiTokenPredictionLayer"


def _set_random_seed(seed_: int):
    if seed_ is not None and seed_ > 0:
        seed = seed_ + (
            100 * paddlefleet.parallel_state.get_pipeline_model_parallel_rank()
        )
        random.seed(seed)
        np.random.seed(seed)
        paddle.manual_seed(seed)
        if (
            paddle.distributed.is_initialized()
            and paddle.cuda.device_count() > 0
        ):
            paddlefleet.tensor_parallel.model_parallel_cuda_manual_seed(seed)
    else:
        raise ValueError(f"Seed ({seed_}) should be a positive integer.")


def _build_config(vocab_size, seq_len):
    return GPTConfig(
        moe_expert_fusion=False,
        vocab_size=vocab_size,
        max_sequence_length=seq_len,
        num_hidden_layers=1,
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
        num_empty_layers_add_in_head=0,
        num_empty_layers_add_in_tail=3,
        pipeline_model_parallel_size=PP_DEGREE,
        virtual_pipeline_model_parallel_size=VPP_DEGREE,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        sequence_parallel=False,
        bf16=True,
        gated_linear_unit=True,
        bias_activation_fusion=True,
        num_nextn_predict_layers=MTP_DEGREE,
        mtp_loss_scaling_factor=0.3,
        separate_mtp_headloss=True,
        use_erndata=True,
        overlap_p2p_comm=True,
    )


def run_pp(seed, batch_size, seq_len, vocab_size, cu_seqlens_list):
    config = _build_config(vocab_size, seq_len)

    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": config.tensor_model_parallel_size,
        "pp_degree": config.pipeline_model_parallel_size,
        "sharding_degree": 1,
        "sep_degree": 1,
        "cp_degree": 1,
        "ep_degree": config.tensor_model_parallel_size,
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
            "forward_backward_overlap_scheduler": False,
            "overlap_p2p_comm": True,
            "enable_dynamic_shape": True,
        },
    }
    micro_batch_size = 1
    num_acc = batch_size // micro_batch_size
    strategy.pipeline_configs = {
        "accumulate_steps": num_acc,
        "micro_batch_size": micro_batch_size,
    }
    initialize_fleet(strategy)

    _set_random_seed(seed)

    gpt_model = gpt_builder(
        config,
        num_stages=config.pipeline_model_parallel_size,
        seg_method=SEG_METHOD,
    )
    gpt_model = paddle.amp.decorate(
        models=gpt_model, optimizers=None, level="O2", dtype="bfloat16"
    )
    gpt_pipe_model = distributed_model(gpt_model)

    # Megatron contract: length-L tensors (NO L+K padding).
    input_ids = paddle.randint(
        low=0, high=vocab_size, shape=(micro_batch_size, seq_len)
    )
    labels = paddle.randint(
        low=0, high=vocab_size, shape=(micro_batch_size, seq_len)
    )
    position_ids = (
        paddle.arange(seq_len, dtype=paddle.int64)
        .reshape([1, seq_len])
        .tile([micro_batch_size, 1])
    )
    cu_seqlens_q = paddle.to_tensor(cu_seqlens_list, dtype="int32")

    # separate_mtp_headloss runs MTPLanguageLoss as an in-pipeline layer that
    # reads labels from the pipeline dict, so labels must ride inputs[0] (via
    # GPTEmbedding) in addition to being the final-loss label in inputs[1]
    # (consumed by MainLanguageLoss).
    inputs = (
        {
            "input_ids": [input_ids] * num_acc,
            "position_ids": [position_ids] * num_acc,
            "cu_seqlens_q": [cu_seqlens_q] * num_acc,
            "labels": [labels] * num_acc,
        },
        [labels] * num_acc,
    )

    loss = gpt_pipe_model.forward_backward_pipeline(inputs, None)
    return loss, gpt_pipe_model


class TestPPMTPMegatronSeparateHeadloss(unittest.TestCase):
    def setUp(self):
        self.seed = 46
        self.batch_size = 4
        self.seq_len = 32
        self.vocab_size = 1024

    def test_pp_mtp_megatron_separate_headloss(self):
        if (
            not paddle.device.current_device_is_cpu
            and paddle.device.get_device_capability()[0] < 9
        ):
            return

        # Multi-document packing: 3 docs inside the length-32 sequence.
        multi_doc = [0, 12, 20, self.seq_len]
        loss, _ = run_pp(
            self.seed,
            self.batch_size,
            self.seq_len,
            self.vocab_size,
            multi_doc,
        )

        val = float(loss)
        print(f"[PP-MTP-MEGATRON-SEP] separate_headloss loss={val}", flush=True)
        assert np.isfinite(val), f"loss must be finite, got {val}"

        # The separated MainLanguageLoss populates its own MTP loss tracker only
        # after the per-depth megatron roll succeeds -- which requires the
        # cu_seqlens_q that GPTMTPLMHead._stash_cu_seqlens_q delivered to the
        # loss stage. If present on this rank, every recorded MTP loss is finite.
        from paddlefleet.models.common.language_loss.language_loss import (
            MainLanguageLoss,
        )

        tracker = dict(MainLanguageLoss.mtp_loss_tracker)
        if tracker:
            print(
                f"[PP-MTP-MEGATRON-SEP] mtp_loss_tracker={tracker}", flush=True
            )
            for k, v in tracker.items():
                assert np.isfinite(float(v)), f"MTP loss {k}={v} must be finite"


if __name__ == "__main__":
    unittest.main()
