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

"""PP>1 + use_erndata=True end-to-end loss smoke test.

Pins the fix in ``GPTLMHead.forward`` / ``GPTMainLMHead.forward``:
``_stash_cu_seqlens_q`` writes the pipeline dict's ``cu_seqlens_q`` onto
``LanguageLoss._cu_seqlens_q_stash`` on the LAST pipeline stage (which never
runs ``GPTEmbedding.forward``), so ``LanguageLoss`` can roll MTP labels per
packed-document boundary under PP>1 without any external dataloader broadcast.

Megatron data contract (differs from the ernie5 template!):
  * tensors are length-L (NO L+K padding);
  * a single int32 ``cu_seqlens_q`` = ``[0, d1, ..., L]`` carries the packed-doc
    boundaries (batch-global semantics);
  * ``GPTEmbedding`` (stage 0) rolls embeddings per-doc via ``cu_seqlens_q``,
    the MTP layer derives per-depth attn masks from it, and ``LanguageLoss``
    (last stage) rolls labels per-doc using the stashed ``cu_seqlens_q``.

If the last-stage stash were missing, ``LanguageLoss.forward`` would raise
``RuntimeError("... requires cu_seqlens_q to be stashed ...")`` instead of
silently rolling labels across document boundaries — so a finite loss + a
clean exit is the positive signal here.
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

PP_DEGREE = 2
MTP_DEGREE = 1


def _set_random_seed(seed_: int):
    """Set random seed for reproducibility (different seed per PP stage)."""
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
        num_empty_layers_add_in_head=1,
        num_empty_layers_add_in_tail=1,
        pipeline_model_parallel_size=PP_DEGREE,
        virtual_pipeline_model_parallel_size=1,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        sequence_parallel=False,
        bf16=True,
        gated_linear_unit=True,
        bias_activation_fusion=True,
        # MTP + megatron data style: the whole point of this test.
        num_nextn_predict_layers=MTP_DEGREE,
        mtp_loss_scaling_factor=0.3,
        use_erndata=True,
        overlap_p2p_comm=False,
        batch_p2p_comm=True,
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
            "overlap_p2p_comm": False,
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
        seg_method="layer:TransformerLayer|EmptyLayer",
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
    # cu_seqlens_q: int32 1D packed-doc boundaries, batch-global. Must end at L.
    cu_seqlens_q = paddle.to_tensor(cu_seqlens_list, dtype="int32")

    inputs = (
        {
            "input_ids": [input_ids] * num_acc,
            "position_ids": [position_ids] * num_acc,
            "cu_seqlens_q": [cu_seqlens_q] * num_acc,
        },
        [labels] * num_acc,
    )

    loss = gpt_pipe_model.forward_backward_pipeline(inputs, None)
    return loss, gpt_pipe_model, cu_seqlens_q


class TestPPMTPMegatron(unittest.TestCase):
    def setUp(self):
        self.seed = 46
        self.batch_size = 4
        self.seq_len = 32
        self.vocab_size = 1024

    def test_pp_mtp_megatron(self):
        # Only meaningful on GPU (needs NCCL P2P + flash attention).
        if (
            not paddle.device.current_device_is_cpu
            and paddle.device.get_device_capability()[0] < 9
        ):
            return

        # Multi-document packing: 3 docs inside the length-32 sequence.
        multi_doc = [0, 12, 20, self.seq_len]
        loss, _, _ = run_pp(
            self.seed,
            self.batch_size,
            self.seq_len,
            self.vocab_size,
            multi_doc,
        )

        val = float(loss)
        print(f"[PP-MTP-MEGATRON] multi_doc loss={val}", flush=True)
        assert np.isfinite(val), f"loss must be finite, got {val}"

        # Correctness signal (rank-agnostic): the MTP loss tracker is only
        # populated on the loss-computing (last PP) rank, and only *after* the
        # megatron per-depth label roll succeeds. That roll requires the
        # cu_seqlens_q that ``GPTLMHead._stash_cu_seqlens_q`` delivered to the
        # loss stage — without it, ``LanguageLoss.forward`` raises RuntimeError
        # (and we would never reach here with a finite loss). If the tracker is
        # populated on this rank, assert every recorded MTP loss is finite.
        from paddlefleet.models.common.language_loss.language_loss import (
            LanguageLoss,
        )

        tracker = dict(LanguageLoss.mtp_loss_tracker)
        if tracker:
            print(f"[PP-MTP-MEGATRON] mtp_loss_tracker={tracker}", flush=True)
            for k, v in tracker.items():
                assert np.isfinite(float(v)), f"MTP loss {k}={v} must be finite"


if __name__ == "__main__":
    unittest.main()
