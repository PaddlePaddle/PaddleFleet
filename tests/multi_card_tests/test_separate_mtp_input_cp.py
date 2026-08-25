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

"""Context-parallel bitwise equivalence test for ``separate_mtp_input``.

``separate_mtp_input`` only changes *how* the shifted MTP embeddings reach
``MultiTokenPredictionLayer`` (dedicated ``mtp_decoder_inputs`` key instead of
being concatenated into ``hidden_states`` and split again in every backbone
layer).  The tensors themselves -- including their CP scatter -- are still
produced by ``GPTEmbedding``, so loss and every parameter gradient must match
the concat/split baseline bit for bit.

The comparison is done inside one process group: two models are built, given
identical weights, and fed identical inputs.

Run with:
    python -m paddle.distributed.launch --gpus=0,1 \
        tests/multi_card_tests/test_separate_mtp_input_cp.py
"""

import functools
import os
import sys
import unittest

# Prefer the local source tree over an installed paddlefleet, mirroring
# PYTHONPATH in script/train_gpu.sh.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_repo_root, "src"))

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import (
    NoPipelineParallel,
)

from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig
from paddlefleet.tensor_parallel.random import (
    model_parallel_cuda_manual_seed,
)
from paddlefleet.training.initialize import initialize_fleet

VOCAB = 512
SEQ = 64  # main decoder length; MTP consumes num_nextn extra positions
NUM_MTP = 1
BATCH = 1
SEED = 20260819

CP_SIZE = None
STRATEGY = None


def setUpModule():
    global CP_SIZE, STRATEGY
    CP_SIZE = dist.get_world_size()
    STRATEGY = fleet.DistributedStrategy()
    STRATEGY.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        # CP is layered on top of the sharding dimension.  ep_degree > 1 selects
        # paddle's expert-aware HybridCommunicateGroup, which is what makes the
        # overlapping sharding/cp/ep dims legal (same shape as the other CP
        # tests in this directory).
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
    initialize_fleet(STRATEGY)
    paddle.seed(SEED)
    model_parallel_cuda_manual_seed(SEED)


def _make_config(separate, cp_mode, mhc):
    extra = (
        {"enable_hyper_connections": True, "num_residual_streams": 4}
        if mhc
        else {}
    )
    return GPTConfig(
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        num_hidden_layers=2,
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=512,
        # MLA, matching the shape constraints used by the CP dataflow test
        multi_latent_attention=True,
        q_lora_rank=128,
        kv_lora_rank=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=48,
        rope_theta=10000,
        gated_linear_unit=True,
        hidden_act=F.silu,
        # MTP
        num_nextn_predict_layers=NUM_MTP,
        mtp_loss_scaling_factor=0.3,
        use_dense_mtp=True,
        separate_mtp_input=separate,
        # CP
        context_parallel_size=CP_SIZE,
        cp_balance_mode=cp_mode,
        experimental_dataflow=True,
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        # general
        normalization="RMSNorm",
        rms_norm_eps=1e-6,
        apply_rope_fusion=False,
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=False,
        parallel_output=True,
        tie_word_embeddings=True,
        position_embedding_type="rope",
        rotary_percent=1.0,
        rotary_base=10000,
        rope_scaling=1.0,
        bf16=True,
        autocast_dtype=paddle.bfloat16,
        params_dtype=paddle.bfloat16,
        gpt_model_use_experimental_version=False,
        init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        output_layer_init_method=functools.partial(
            paddle.nn.init.xavier_uniform_, gain=1.0
        ),
        **extra,
    )


def _make_inputs():
    """Identical inputs for both runs (built once, cloned per run)."""
    paddle.seed(SEED)
    total = SEQ + NUM_MTP
    data = paddle.randint(low=1, high=VOCAB, shape=(BATCH, total + 1)).cuda()
    dist.broadcast(data, src=0)
    input_ids = data[:, :-1].contiguous()
    labels = data[:, 1:].contiguous()

    # causal boundaries, last dim = 2 (experimental dataflow)
    attn_mask_startend_row_indices = paddle.zeros(
        [BATCH, 1, SEQ, 2], dtype=paddle.int32
    ).cuda()
    for i in range(SEQ):
        attn_mask_startend_row_indices[:, :, i, 0] = SEQ
        attn_mask_startend_row_indices[:, :, i, 1] = i

    mtp_hidden_inputs_mask_all = paddle.ones(
        [BATCH, NUM_MTP, SEQ], dtype=paddle.int32
    ).cuda()
    mtp_hidden_inputs_mask_all[:, :, -2:] = 0
    mtp_startend_row_indices_all = paddle.full(
        [BATCH, NUM_MTP, SEQ, 1], fill_value=SEQ, dtype=paddle.int32
    ).cuda()

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
        "mtp_startend_row_indices_all": mtp_startend_row_indices_all,
        "mtp_hidden_inputs_mask_all": mtp_hidden_inputs_mask_all,
    }


def _forward_backward(model, raw_inputs):
    """Run one fwd+bwd step and return (loss, {param_name: grad})."""
    pipe_model = NoPipelineParallel(model, STRATEGY)
    labels = raw_inputs["labels"].clone()
    inputs = (
        {
            "input_ids": [raw_inputs["input_ids"].clone()],
            "labels": [labels],
            "attn_mask_startend_row_indices": [
                raw_inputs["attn_mask_startend_row_indices"].clone()
            ],
            "mtp_startend_row_indices_all": [
                raw_inputs["mtp_startend_row_indices_all"].clone()
            ],
            "mtp_hidden_inputs_mask_all": [
                raw_inputs["mtp_hidden_inputs_mask_all"].clone()
            ],
        },
        labels,
    )
    loss = pipe_model.forward_backward_pipeline(inputs)
    grads = {
        name: param.grad.clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    return loss, grads


class TestSeparateMTPInputCP(unittest.TestCase):
    """separate_mtp_input must be bit-identical to the baseline under CP > 1."""

    def _assert_bitwise(self, a, b, what):
        self.assertEqual(list(a.shape), list(b.shape), f"{what}: shape differs")
        self.assertEqual(a.dtype, b.dtype, f"{what}: dtype differs")
        # float32 widening is lossless for bf16/fp16, so equality on the widened
        # values is equivalent to a bit-for-bit comparison (and equal_all has no
        # bfloat16 kernel).
        left = np.asarray(a.astype("float32").numpy())
        right = np.asarray(b.astype("float32").numpy())
        if not np.array_equal(left, right):
            diff = np.max(np.abs(left - right))
            self.fail(f"{what}: not bitwise equal (max abs diff {diff})")

    def _compare(self, cp_mode, mhc):
        raw_inputs = _make_inputs()

        paddle.seed(SEED)
        base_model = gpt_builder(
            _make_config(False, cp_mode, mhc), num_stages=1
        )
        sep_model = gpt_builder(_make_config(True, cp_mode, mhc), num_stages=1)
        # identical weights: the flag changes dataflow only, never parameters
        sep_model.set_state_dict(base_model.state_dict())

        base_loss, base_grads = _forward_backward(base_model, raw_inputs)
        sep_loss, sep_grads = _forward_backward(sep_model, raw_inputs)

        self.assertTrue(
            paddle.isfinite(base_loss).item(),
            f"baseline loss is not finite: {base_loss.item()}",
        )
        self._assert_bitwise(sep_loss, base_loss, "loss")

        self.assertEqual(
            sorted(base_grads), sorted(sep_grads), "gradient key sets differ"
        )
        self.assertGreater(len(base_grads), 0, "no gradients were produced")
        for name in base_grads:
            self._assert_bitwise(
                sep_grads[name], base_grads[name], f"grad[{name}]"
            )

        if dist.get_rank() == 0:
            print(
                f"[cp={CP_SIZE} mode={cp_mode} mhc={mhc}] loss="
                f"{np.array(base_loss.astype('float32'))} "
                f"{len(base_grads)} grads bitwise equal"
            )

    def test_contiguous_allgather(self):
        self._compare("contiguous_allgather", mhc=False)

    def test_dualchunk_allgather(self):
        self._compare("dualchunk_allgather", mhc=False)

    def test_contiguous_allgather_with_mhc(self):
        self._compare("contiguous_allgather", mhc=True)


if __name__ == "__main__":
    paddle.set_default_dtype("bfloat16")
    unittest.main()
