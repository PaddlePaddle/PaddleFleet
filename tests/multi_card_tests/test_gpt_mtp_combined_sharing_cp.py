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

"""mtp_shared_weights + mtp_shared_last_layer under context parallelism.

Weight sharing is a statement about parameters, so it must not interact with how
CP splits the sequence. This checks it as an equivalence against an unshared
model on the same CP group:

  * copy the combined-sharing model's values into a model built without any MTP
    sharing, so every would-be tied copy starts equal;
  * run the same forward/backward on both;
  * the losses must match, and every shared Parameter's gradient must equal the
    sum of the gradients of the copies it stands for (backbone last layer + each
    MTP body for the transformer weights, each MTP depth for the fusion weights).

Runs at any world size; CI runs it with CP=2.
"""

import functools
import os
import sys
import unittest

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_repo_root, "src"))

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import NoPipelineParallel

from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer.multi_token_prediction import (
    MultiTokenPredictionLayer,
)

VOCAB = 1024
SEQ = 32  # divisible by 2 * cp_size, as the zigzag split requires
NUM_MTP = 2
NUM_HIDDEN_LAYERS = 2
BATCH = 1
SEED = 46
CU_SEQLENS = [0, 12, 20, SEQ]

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


def _make_config(shared):
    return GPTConfig(
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=4,
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
        apply_rope_fusion=False,
        gated_linear_unit=True,
        num_nextn_predict_layers=NUM_MTP,
        mtp_loss_scaling_factor=0.3,
        mtp_shared_weights=shared,
        mtp_shared_last_layer=shared,
        use_erndata=True,
        context_parallel_size=CP_SIZE,
        cp_balance_mode="dualchunk_allgather",
        experimental_dataflow=False,
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        bf16=True,
        gpt_model_use_experimental_version=False,
        init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        output_layer_init_method=functools.partial(
            paddle.nn.init.xavier_uniform_, gain=1.0
        ),
    )


def _make_inputs():
    paddle.seed(SEED)
    data = paddle.randint(low=1, high=VOCAB, shape=(BATCH, SEQ + 1)).cuda()
    if CP_SIZE > 1:
        dist.broadcast(data, src=0)
    end = np.zeros(SEQ, dtype=np.int32)
    for j in range(len(CU_SEQLENS) - 1):
        s, e = CU_SEQLENS[j], CU_SEQLENS[j + 1]
        end[s:e] = e
    return {
        "input_ids": data[:, :-1].contiguous(),
        "labels": data[:, 1:].contiguous(),
        "position_ids": paddle.arange(SEQ, dtype=paddle.int64)
        .reshape([1, SEQ])
        .tile([BATCH, 1])
        .cuda(),
        "cu_seqlens_q": paddle.to_tensor(CU_SEQLENS, dtype="int32").cuda(),
        "attn_mask_startend_row_indices": paddle.to_tensor(
            end[None, None, :, None]
        )
        .tile([BATCH, 1, 1, 1])
        .cuda(),
    }


def _forward_backward(model, raw):
    pipe_model = NoPipelineParallel(model, STRATEGY)
    labels = raw["labels"].clone()
    micro = {
        key: [raw[key].clone()]
        for key in (
            "input_ids",
            "position_ids",
            "cu_seqlens_q",
            "attn_mask_startend_row_indices",
        )
    }
    micro["labels"] = [labels]
    return pipe_model.forward_backward_pipeline((micro, labels))


def _layer_pairs(shared_model, ref_model):
    shared_layers = list(shared_model.run_function)
    ref_layers = list(ref_model.run_function)
    assert len(shared_layers) == len(ref_layers)
    for s, r in zip(shared_layers, ref_layers):
        assert type(s) is type(r), (type(s), type(r))
    return list(zip(shared_layers, ref_layers))


def _tie_groups(shared_model, ref_model):
    """Map each shared-model Parameter to the ref Parameters it stands for."""
    groups = {}
    for s_layer, r_layer in _layer_pairs(shared_model, ref_model):
        r_named = dict(r_layer.named_parameters())
        for name, s_param in s_layer.named_parameters():
            entry = groups.setdefault(id(s_param), (s_param, {}))
            r_param = r_named[name]
            entry[1][id(r_param)] = r_param
    return list(groups.values())


class TestMTPCombinedSharingCP(unittest.TestCase):
    def test_matches_unshared_model(self):
        if (
            not paddle.device.current_device_is_cpu
            and paddle.device.get_device_capability()[0] < 9
        ):
            self.skipTest("requires SM90+ for the CP flashmask kernels")

        paddle.seed(SEED)
        shared = gpt_builder(_make_config(shared=True), num_stages=1)
        paddle.seed(SEED)
        ref = gpt_builder(_make_config(shared=False), num_stages=1)

        groups = _tie_groups(shared, ref)
        with paddle.no_grad():
            for s_param, r_params in groups:
                for r_param in r_params.values():
                    r_param.set_value(s_param)

        mtp_layers = [
            la
            for la in shared.run_function
            if isinstance(la, MultiTokenPredictionLayer)
        ]
        self.assertEqual(len(mtp_layers), NUM_MTP)
        body_ids = {id(p) for _, p in mtp_layers[0].transformer_layer_weights}
        fusion_ids = {
            id(p)
            for n, p in mtp_layers[0].all_weights
            if not n.startswith("transformer_layer.")
        }
        sizes = {id(s): len(r) for s, r in groups}
        self.assertTrue(body_ids and fusion_ids)
        for pid in body_ids:
            self.assertEqual(
                sizes[pid],
                NUM_MTP + 1,
                "body param must stand for the backbone last layer and every "
                "MTP depth",
            )
        for pid in fusion_ids:
            self.assertEqual(
                sizes[pid],
                NUM_MTP,
                "fusion param must stand for every MTP depth",
            )

        shared = paddle.amp.decorate(
            models=shared, optimizers=None, level="O2", dtype="bfloat16"
        )
        ref = paddle.amp.decorate(
            models=ref, optimizers=None, level="O2", dtype="bfloat16"
        )
        # decorate must cast in place; a replaced Parameter would break the ties.
        for s_param, _ in groups:
            self.assertIn(id(s_param), {id(p) for p in shared.parameters()})

        raw = _make_inputs()
        loss_shared = float(_forward_backward(shared, raw).astype("float32"))
        loss_ref = float(_forward_backward(ref, raw).astype("float32"))
        print(
            f"[MTP-COMBINED-CP] cp={CP_SIZE} shared={loss_shared} ref={loss_ref}",
            flush=True,
        )
        self.assertTrue(np.isfinite(loss_shared))
        np.testing.assert_allclose(loss_shared, loss_ref, rtol=1e-6, atol=0)

        checked = 0
        for s_param, r_params in groups:
            if s_param.grad is None:
                continue
            got = s_param.grad.astype("float32").numpy()
            want = sum(
                r.grad.astype("float32").numpy()
                for r in r_params.values()
                if r.grad is not None
            )
            denom = max(float(np.linalg.norm(want)), 1e-6)
            rel = float(np.linalg.norm(got - want)) / denom
            self.assertLess(
                rel,
                2e-2,
                f"grad of {s_param.name} (stands for {len(r_params)} copies) "
                f"differs from the sum of the unshared grads: rel={rel}",
            )
            checked += 1
        self.assertGreater(checked, 0, "no gradients were compared")


if __name__ == "__main__":
    unittest.main()
