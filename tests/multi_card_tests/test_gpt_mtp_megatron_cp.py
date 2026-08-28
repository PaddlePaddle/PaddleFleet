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

"""CP-aware ``use_erndata=True`` end-to-end test (CP=1 vs CP=2).

The megatron MTP branch in ``GPTEmbedding`` slices the sequence itself with
``extract_local_zigzag_chunks`` instead of going through
``ContextParallelScatterOp`` (which is gated on ``experimental_dataflow``, a
flag this style forbids). The RoPE tables are built by
``RotaryEmbedding.get_rotary_seq_len``, which scales the rank-local length back
up by ``cp_group.world_size`` and therefore always yields the FULL length L --
so they must be zigzag-sliced with the same layout, otherwise every rank
applies the positions ``0..L/cp-1`` to chunks that actually live at
``[interval*r, interval*(r+1)) u [L-interval*(r+1), L-interval*r)``.

The primary check is a LAYOUT one (``test_rope_matches_zigzag_layout``): the
``rotary_pos_emb`` the real ``GPTEmbedding`` hands to the decoder must equal the
zigzag slice of the full-length table this rank's ``RotaryEmbedding`` produces.

``test_cp_invariant_loss`` is a coarse companion: same weights (CPU init, so
rank-independent) and same data must give the same loss at any ``cp_degree``.
``REF_LOSS`` is the single-card value; run

    python -m paddle.distributed.launch --gpus=0 \
        tests/multi_card_tests/test_gpt_mtp_megatron_cp.py

to regenerate it and

    python -m paddle.distributed.launch --gpus=0,1 \
        tests/multi_card_tests/test_gpt_mtp_megatron_cp.py

for the CP=2 comparison. Note the loss alone is a weak signal on a randomly
initialised 2-layer model: measured against a deliberately broken
contiguous-prefix RoPE it only moves 9.246039 -> 9.247475 (1.6e-4 relative),
inside the bf16 tolerance below. Hence the layout assertion.
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
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import NoPipelineParallel

from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.training.initialize import initialize_fleet

VOCAB = 1024
SEQ = 32  # divisible by 2 * cp_size, as the zigzag split requires
NUM_MTP = 1
BATCH = 1
SEED = 46
# Packed multi-document layout: 3 docs inside the length-32 sequence.
CU_SEQLENS = [0, 12, 20, SEQ]

# Single-card (cp_degree=1) reference loss, see module docstring.
REF_LOSS = 9.246039390563965

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
        # Same overlapping sharding/cp/ep shape as the other CP tests in this
        # directory: ep_degree > 1 selects paddle's expert-aware
        # HybridCommunicateGroup, which is what makes the overlap legal.
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


def _make_config():
    return GPTConfig(
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        num_hidden_layers=2,
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=512,
        normalization="RMSNorm",
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        # CPU init keeps the weights bit-identical across world sizes, which is
        # what makes the CP=1 reference loss comparable.
        use_cpu_initialization=True,
        parallel_output=True,
        tie_word_embeddings=True,
        position_embedding_type="rope",
        rotary_percent=1.0,
        rotary_base=10000,
        rope_scaling=1.0,
        apply_rope_fusion=False,
        gated_linear_unit=True,
        # MTP, megatron data style
        num_nextn_predict_layers=NUM_MTP,
        mtp_loss_scaling_factor=0.3,
        use_erndata=True,
        # CP
        context_parallel_size=CP_SIZE,
        cp_balance_mode="dualchunk_allgather",
        experimental_dataflow=False,
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        # bf16 compute with fp32 master params: `use_cpu_initialization` builds
        # its master weight in fp32 (`_initialize_affine_weight_cpu`) and copies
        # it into the parameter as-is, so a bf16 `params_dtype` would fail the
        # dtype check. `paddle.amp.decorate(level="O2")` below casts for compute
        # instead, matching tests/multi_card_tests/pipeline_parallel/
        # test_gpt_pp_mtp_megatron.py.
        bf16=True,
        gpt_model_use_experimental_version=False,
        init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        output_layer_init_method=functools.partial(
            paddle.nn.init.xavier_uniform_, gain=1.0
        ),
    )


def _make_inputs(with_mask=True):
    """Megatron contract: length-L tensors (no L+K padding) + cu_seqlens_q.

    ``with_mask=False`` reproduces the minimal erndata contract, where only
    ``cu_seqlens_q`` carries the doc boundaries and no materialized flashmask is
    supplied (erndata emits ``attn_mask_startend_row_indices`` only when
    ``pack_by_cu_seqlen=True``). ``GPTEmbedding`` must then derive the main mask
    itself, otherwise the CP branch of ``DotProductAttention`` synthesizes an
    all-visible mask and runs flashmask with ``causal=False``, silently dropping
    causality and doc boundaries from the backbone.
    """
    paddle.seed(SEED)
    data = paddle.randint(low=1, high=VOCAB, shape=(BATCH, SEQ + 1)).cuda()
    # Same data on every CP rank. Skipped at world size 1, where fleet leaves
    # the global communication group uninitialized.
    if CP_SIZE > 1:
        dist.broadcast(data, src=0)
    input_ids = data[:, :-1].contiguous()
    labels = data[:, 1:].contiguous()

    position_ids = (
        paddle.arange(SEQ, dtype=paddle.int64)
        .reshape([1, SEQ])
        .tile([BATCH, 1])
        .cuda()
    )
    cu_seqlens_q = paddle.to_tensor(CU_SEQLENS, dtype="int32").cuda()

    out = {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": position_ids,
        "cu_seqlens_q": cu_seqlens_q,
    }
    if with_mask:
        # Full-length per-doc flashmask boundaries. CP attention allgathers KV
        # and remaps these global row values itself
        # (preprocess_index_dual_chunks), so they are intentionally NOT sliced.
        # 1 column because gpt_model_use_experimental_version=False.
        end = np.zeros(SEQ, dtype=np.int32)
        for j in range(len(CU_SEQLENS) - 1):
            s, e = CU_SEQLENS[j], CU_SEQLENS[j + 1]
            end[s:e] = e
        out["attn_mask_startend_row_indices"] = (
            paddle.to_tensor(end[None, None, :, None])
            .tile([BATCH, 1, 1, 1])
            .cuda()
        )
    return out


def _forward_backward(model, raw):
    pipe_model = NoPipelineParallel(model, STRATEGY)
    labels = raw["labels"].clone()
    micro = {
        "input_ids": [raw["input_ids"].clone()],
        "position_ids": [raw["position_ids"].clone()],
        "cu_seqlens_q": [raw["cu_seqlens_q"].clone()],
        "labels": [labels],
    }
    if "attn_mask_startend_row_indices" in raw:
        micro["attn_mask_startend_row_indices"] = [
            raw["attn_mask_startend_row_indices"].clone()
        ]
    return pipe_model.forward_backward_pipeline((micro, labels))


def _find_embedding(model):
    """The GPTEmbedding instance inside a built (single-stage) GPT model."""
    from paddlefleet.models.gpt.gpt_embedding import GPTEmbedding

    for layer in model.sublayers(include_self=True):
        if isinstance(layer, GPTEmbedding):
            return layer
    raise AssertionError("no GPTEmbedding found in the built model")


class TestMTPMegatronCPRope(unittest.TestCase):
    def test_rope_matches_zigzag_layout(self):
        """The decoder's RoPE table must be this rank's zigzag chunk.

        Runs the real ``GPTEmbedding.forward`` (real ``RotaryEmbedding``, real
        CP group) and compares its ``rotary_pos_emb`` against the zigzag slice
        of the full-length table. A contiguous prefix -- what the code produced
        before the fix -- fails this at any CP degree > 1.
        """
        from paddlefleet.parallel_state import get_context_parallel_rank
        from paddlefleet.transformer.multi_token_prediction import (
            extract_local_zigzag_chunks,
        )

        paddle.seed(SEED)
        model = gpt_builder(_make_config(), num_stages=1)
        emb = _find_embedding(model)
        raw = _make_inputs()

        out = emb.forward(
            {
                "input_ids": raw["input_ids"],
                "position_ids": raw["position_ids"],
                "cu_seqlens_q": raw["cu_seqlens_q"],
            }
        )
        rope_local = out["rotary_pos_emb"]

        # Rank-local sequence length: the megatron branch keeps the full L and
        # zigzag-slices it, so both hidden states and RoPE must be L / cp.
        self.assertEqual(out["hidden_states"].shape[1], SEQ // CP_SIZE)
        self.assertEqual(rope_local.shape[1], SEQ // CP_SIZE)

        # Full-length table straight from this rank's RotaryEmbedding, sliced
        # with the layout the embeddings used.
        full = emb.rotary_pos_emb(SEQ)
        cp_rank = get_context_parallel_rank() if CP_SIZE > 1 else 0
        expected = extract_local_zigzag_chunks(full, cp_rank, CP_SIZE, axis=1)
        np.testing.assert_array_equal(
            rope_local.astype("float32").numpy(),
            expected.astype("float32").numpy(),
        )

        if CP_SIZE > 1:
            # And it must NOT be the contiguous prefix (the pre-fix layout).
            contiguous = full[:, : SEQ // CP_SIZE]
            self.assertFalse(
                bool(
                    paddle.all(
                        rope_local.astype("float32")
                        == contiguous.astype("float32")
                    )
                ),
                "rank>0 must not receive the contiguous RoPE prefix",
            )


class TestMTPMegatronCP(unittest.TestCase):
    def test_cp_invariant_loss(self):
        # Needs flash attention (SM90+) for the CP flashmask path.
        if (
            not paddle.device.current_device_is_cpu
            and paddle.device.get_device_capability()[0] < 9
        ):
            self.skipTest("requires SM90+ for the CP flashmask kernels")

        paddle.seed(SEED)
        model = gpt_builder(_make_config(), num_stages=1)
        model = paddle.amp.decorate(
            models=model, optimizers=None, level="O2", dtype="bfloat16"
        )
        loss = _forward_backward(model, _make_inputs())

        val = float(loss.astype("float32"))
        print(f"[MTP-MEGATRON-CP] cp={CP_SIZE} loss={val}", flush=True)
        self.assertTrue(np.isfinite(val), f"loss must be finite, got {val}")

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertGreater(len(grads), 0, "no gradients were produced")
        for g in grads:
            self.assertTrue(
                bool(paddle.isfinite(g.astype("float32")).all()),
                "gradients must be finite",
            )

        if REF_LOSS is not None:
            # bf16 tolerance: CP changes the attention reduction order
            # (allgathered KV) but not the math. A wrong RoPE layout is off by
            # far more than this.
            np.testing.assert_allclose(val, REF_LOSS, rtol=5e-3, atol=0)

        from paddlefleet.models.common.language_loss.language_loss import (
            LanguageLoss,
        )

        tracker = dict(LanguageLoss.mtp_loss_tracker)
        self.assertTrue(tracker, "MTP loss tracker was never populated")
        for k, v in tracker.items():
            self.assertTrue(
                np.isfinite(float(v)), f"MTP loss {k}={v} must be finite"
            )


class TestMTPMegatronMainMaskFromCuSeqlens(unittest.TestCase):
    """The main flashmask must be derived when the contract omits it.

    erndata only guarantees length-L tensors + cu_seqlens_q. When
    ``attn_mask_startend_row_indices`` is absent, ``GPTEmbedding`` must build it
    from ``cu_seqlens_q``; otherwise the CP branch of ``DotProductAttention``
    fills in an all-visible mask and calls flashmask with ``causal=False``,
    which drops both causality and document boundaries from the backbone.
    """

    def test_embedding_derives_main_mask(self):
        paddle.seed(SEED)
        model = gpt_builder(_make_config(), num_stages=1)
        emb = _find_embedding(model)
        raw = _make_inputs(with_mask=False)
        self.assertNotIn("attn_mask_startend_row_indices", raw)

        out = emb.forward(
            {
                "input_ids": raw["input_ids"],
                "position_ids": raw["position_ids"],
                "cu_seqlens_q": raw["cu_seqlens_q"],
            }
        )

        mask = out.get("attn_mask_startend_row_indices")
        self.assertIsNotNone(
            mask,
            "GPTEmbedding must derive the main mask from cu_seqlens_q",
        )
        # Full length L (not the rank-local L/cp): CP attention allgathers KV
        # and remaps these global row values itself.
        self.assertEqual(list(mask.shape), [BATCH, 1, SEQ, 1])
        self.assertEqual(mask.dtype, paddle.int32)

        # Values must be the per-doc end rows implied by CU_SEQLENS.
        expected = np.zeros(SEQ, dtype=np.int32)
        for j in range(len(CU_SEQLENS) - 1):
            s, e = CU_SEQLENS[j], CU_SEQLENS[j + 1]
            expected[s:e] = e
        np.testing.assert_array_equal(
            mask.numpy()[0, 0, :, 0],
            expected,
        )

    def test_derived_mask_matches_explicit_mask(self):
        """Deriving must be equivalent to passing the mask in explicitly."""
        paddle.seed(SEED)
        model = gpt_builder(_make_config(), num_stages=1)
        emb = _find_embedding(model)

        raw_with = _make_inputs(with_mask=True)
        raw_without = _make_inputs(with_mask=False)

        out_with = emb.forward(
            {
                "input_ids": raw_with["input_ids"],
                "position_ids": raw_with["position_ids"],
                "cu_seqlens_q": raw_with["cu_seqlens_q"],
                "attn_mask_startend_row_indices": raw_with[
                    "attn_mask_startend_row_indices"
                ],
            }
        )
        out_without = emb.forward(
            {
                "input_ids": raw_without["input_ids"],
                "position_ids": raw_without["position_ids"],
                "cu_seqlens_q": raw_without["cu_seqlens_q"],
            }
        )
        np.testing.assert_array_equal(
            out_with["attn_mask_startend_row_indices"].numpy(),
            out_without["attn_mask_startend_row_indices"].numpy(),
        )

    def test_cp_loss_without_explicit_mask(self):
        """End-to-end on the minimal contract: same loss as with the mask."""
        if (
            not paddle.device.current_device_is_cpu
            and paddle.device.get_device_capability()[0] < 9
        ):
            self.skipTest("requires SM90+ for the CP flashmask kernels")

        paddle.seed(SEED)
        model = gpt_builder(_make_config(), num_stages=1)
        model = paddle.amp.decorate(
            models=model, optimizers=None, level="O2", dtype="bfloat16"
        )
        loss = _forward_backward(model, _make_inputs(with_mask=False))
        val = float(loss.astype("float32"))
        print(
            f"[MTP-MEGATRON-CP] cp={CP_SIZE} loss(no explicit mask)={val}",
            flush=True,
        )
        self.assertTrue(np.isfinite(val), f"loss must be finite, got {val}")
        if REF_LOSS is not None:
            np.testing.assert_allclose(val, REF_LOSS, rtol=5e-3, atol=0)


if __name__ == "__main__":
    unittest.main()
