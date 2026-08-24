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

"""Tensor/sequence-parallel bitwise equivalence test for ``separate_mtp_input``.

Under ``sequence_parallel=True`` the shifted MTP embeddings are laid out as
``[S/tp, B, H]`` by ``GPTEmbedding``.  ``separate_mtp_input`` forwards exactly
those tensors to ``MultiTokenPredictionLayer`` through ``mtp_decoder_inputs``
(no second scatter), so loss and every parameter gradient must match the
concat/split baseline bit for bit.

Two models are built in one process group, given identical weights, and fed
identical inputs.

Run with:
    python -m paddle.distributed.launch --gpus=0,1 \
        tests/multi_card_tests/tensor_parallel/test_separate_mtp_input_tp_sp.py
"""

import functools
import os
import sys
import unittest

# Prefer the local source tree over an installed paddlefleet.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(_repo_root, "src"))

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import (
    NoPipelineParallel,
)

import paddlefleet.parallel_state as ps
from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig
from paddlefleet.tensor_parallel.random import (
    model_parallel_cuda_manual_seed,
)

VOCAB = 512
SEQ = 64  # main decoder length; must stay divisible by TP for SP scatter
NUM_MTP = 1
BATCH = 1
SEED = 20260819

TP_SIZE = None
STRATEGY = None


def setUpModule():
    global TP_SIZE, STRATEGY
    TP_SIZE = dist.get_world_size()
    STRATEGY = fleet.DistributedStrategy()
    STRATEGY.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": TP_SIZE,
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
    fleet.init(is_collective=True, strategy=STRATEGY)
    hcg = fleet.get_hybrid_communicate_group()
    ps.initialize_model_parallel(hcg)
    model_parallel_cuda_manual_seed(SEED)


def _make_config(separate, mhc):
    extra = (
        {"enable_hyper_connections": True, "num_residual_streams": 4}
        if mhc
        else {}
    )
    return GPTConfig(
        vocab_size=VOCAB,
        max_sequence_length=SEQ + NUM_MTP,
        num_hidden_layers=2,
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=512,
        # MTP
        num_nextn_predict_layers=NUM_MTP,
        mtp_loss_scaling_factor=0.3,
        use_dense_mtp=True,
        separate_mtp_input=separate,
        # TP / SP
        tensor_model_parallel_size=TP_SIZE,
        sequence_parallel=True,
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        # general
        normalization="RMSNorm",
        rms_norm_eps=1e-6,
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        use_bias=False,
        position_embedding_type="rope",
        rotary_percent=1.0,
        rotary_base=10000,
        rope_scaling=1.0,
        tie_word_embeddings=True,
        gpt_model_use_experimental_version=False,
        init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        output_layer_init_method=functools.partial(
            paddle.nn.init.xavier_uniform_, gain=1.0
        ),
        **extra,
    )


def _make_inputs():
    """Identical inputs for both runs, broadcast so every TP rank agrees."""
    total = SEQ + NUM_MTP
    paddle.seed(SEED)
    data = paddle.randint(low=1, high=VOCAB, shape=(BATCH, total + 1)).cuda()
    dist.broadcast(data, src=0)
    position_ids = (
        paddle.arange(total, dtype=paddle.int64).reshape([1, -1]).cuda()
    )
    mask_all = paddle.ones([BATCH, NUM_MTP, SEQ], dtype=paddle.float32).cuda()
    mask_all[:, :, -2:] = 0.0
    startend_all = paddle.zeros(
        [BATCH, NUM_MTP, SEQ, 1], dtype=paddle.int32
    ).cuda()
    return {
        "input_ids": data[:, :-1].contiguous(),
        "labels": data[:, 1:].contiguous(),
        "position_ids": position_ids.expand([BATCH, total]).contiguous(),
        "mtp_hidden_inputs_mask_all": mask_all,
        "mtp_startend_row_indices_all": startend_all,
    }


def _forward_backward(model, raw_inputs):
    """Run one fwd+bwd step and return (loss, {param_name: grad})."""
    pipe_model = NoPipelineParallel(model, STRATEGY)
    labels = raw_inputs["labels"].clone()
    inputs = (
        {
            "input_ids": [raw_inputs["input_ids"].clone()],
            "position_ids": [raw_inputs["position_ids"].clone()],
            "mtp_startend_row_indices_all": raw_inputs[
                "mtp_startend_row_indices_all"
            ].clone(),
            "mtp_hidden_inputs_mask_all": raw_inputs[
                "mtp_hidden_inputs_mask_all"
            ].clone(),
        },
        [labels],
    )
    loss = pipe_model.forward_backward_pipeline(inputs)
    grads = {
        name: param.grad.clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    return loss, grads


class TestSeparateMTPInputTPSP(unittest.TestCase):
    """separate_mtp_input must be bit-identical to the baseline under TP/SP."""

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

    def _compare(self, mhc):
        raw_inputs = _make_inputs()

        paddle.seed(SEED)
        base_model = gpt_builder(_make_config(False, mhc), num_stages=1)
        sep_model = gpt_builder(_make_config(True, mhc), num_stages=1)
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
                f"[tp={TP_SIZE} sp=True mhc={mhc}] loss="
                f"{np.array(base_loss.astype('float32'))} "
                f"{len(base_grads)} grads bitwise equal"
            )

    def test_sequence_parallel(self):
        self._compare(mhc=False)

    def test_sequence_parallel_with_mhc(self):
        self._compare(mhc=True)


if __name__ == "__main__":
    unittest.main()
