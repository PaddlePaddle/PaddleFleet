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

"""Real multi-card (PP=4) behavior tests for ``paddlefleet.gpt_builders``.

Launched with ``paddle.distributed.launch`` over 4 GPUs (PP_DEGREE=4). The
tests build real GPT pipeline models with ``gpt_builder`` on a live 4-stage
pipeline group and exercise real pipeline send/recv collectives. Expected
values are derived by hand / from independent references -- never by calling
the function under test and never taken from any coverage_test file.

Independent references used here:

* Zero-weight cross entropy: with every parameter set to 0 the pre-softmax
  logits are identically 0, so softmax is exactly uniform and the token-mean
  cross entropy equals ``ln(vocab_size)`` for every token, regardless of the
  transformer wiring. This is an EXACT (not banded) anchor that a wrong loss
  reduction (summed instead of token-averaged, mis-scaled, or over the wrong
  denominator) cannot satisfy, and it flows through all four pipeline stages.
* Layer partition: ``gpt_builder`` must split the decoder over the stages so
  that the union of every stage's owned layer numbers is exactly
  ``0..num_hidden_layers-1`` with no gaps or duplicates -- a genuine cross-rank
  contract checked with a real ``all_gather_object`` over the pp group.
* FLOPs-per-token: computed by hand from the model dimensions, including the
  output (lm-head) projection term.

Known production bugs (do NOT edit production to satisfy these tests):

* ``models/gpt/gpt_model.py`` ``GPTModel.use_fp8`` -- the
  ``_num_virtual_pipeline_stages > 1`` branch iterates chunks and
  ``return True`` on a match but is missing the trailing ``return False`` that
  the non-VPP branch has, so a VPP model with fp8 disabled returns ``None``.
  ``test_use_fp8_vpp_returns_false`` asserts the CORRECT ``is False`` and is
  marked ``@unittest.expectedFailure``.
* ``models/gpt/utils.py`` ``GPTModelEstimator.estimate_flops_per_token`` --
  ``output_logits_flops`` multiplies by ``0`` when ``num_nextn_predict_layers``
  is ``None`` (non-MTP), dropping the lm-head FLOPs entirely.
  ``test_estimate_flops_per_token_includes_lm_head`` asserts the CORRECT total
  (with the lm-head term) and is marked ``@unittest.expectedFailure``.
"""

import functools
import math
import unittest

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet import distributed_model

from paddlefleet import parallel_state
from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig
from paddlefleet.models.gpt.utils import GPTModelEstimator
from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer.transformer_layer import TransformerLayer

PP_DEGREE = 4
NUM_LAYERS = 8
HIDDEN = 64
HEADS = 4
FFN = 128
VOCAB = 128
SEQ_LEN = 16
SEED = 2024
SEG_METHOD = "layer:TransformerLayer|EmptyLayer"


def _build_config(**overrides):
    """Build a small dense GPT config that drives the real pipeline builder.

    Dropout is disabled and CPU initialization is used so behaviour is
    deterministic. MoE and TP are left off, exercising the dense per-layer
    branch of ``gpt_builder``.
    """
    xavier = functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0)
    defaults = dict(  # noqa: C408
        vocab_size=VOCAB,
        max_sequence_length=SEQ_LEN,
        num_hidden_layers=NUM_LAYERS,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        intermediate_size=FFN,
        normalization="RMSNorm",
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=True,
        parallel_output=True,
        tie_word_embeddings=True,
        position_embedding_type="rope",
        rotary_percent=1.0,
        init_method=xavier,
        output_layer_init_method=xavier,
        pipeline_model_parallel_size=PP_DEGREE,
    )
    defaults.update(overrides)
    return GPTConfig(**defaults)


def _make_lm_batch(micro_batch_size, num_acc):
    """Deterministic next-token-prediction microbatches placed on GPU."""
    data = paddle.randint(
        low=0, high=VOCAB, shape=(micro_batch_size, SEQ_LEN + 1)
    ).cuda()
    input_ids = data[:, :-1]
    labels = data[:, 1:]
    position_ids = (
        paddle.arange(0, SEQ_LEN, dtype="int64")
        .unsqueeze(0)
        .expand([micro_batch_size, -1])
        .cuda()
    )
    inputs = (
        {
            "input_ids": [input_ids] * num_acc,
            "position_ids": [position_ids] * num_acc,
        },
        [labels] * num_acc,
    )
    return inputs


@unittest.skipUnless(
    paddle.is_compiled_with_cuda(), "gpt_builder PP tests require a CUDA build"
)
class TestGPTBuilderPipelineParallel(unittest.TestCase):
    """PP=4 behaviour tests for models produced by ``gpt_builder``."""

    @classmethod
    def setUpClass(cls):
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
        }
        strategy.pipeline_configs = {
            "accumulate_steps": 4,
            "micro_batch_size": 1,
        }
        initialize_fleet(strategy)
        paddle.seed(SEED)

    # PLACEHOLDER_TESTS

    def test_zero_weight_pipeline_loss_is_uniform_cross_entropy(self):
        """Zeroing every parameter makes the pp loss exactly ``ln(VOCAB)``.

        With all weights (and biases, and RMSNorm gains) set to 0 the hidden
        state stays 0 through every stage and the logits are identically 0, so
        softmax is exactly uniform and each token's cross entropy is
        ``ln(VOCAB)``. The token-mean reduction over the whole pipeline must
        therefore equal ``ln(VOCAB)`` EXACTLY (up to fp32 rounding). This is an
        independent analytic anchor -- not a call to the builder to produce its
        own expected -- and it runs through the real 4-stage pipeline
        forward/backward with live send/recv collectives. A summed (rather than
        token-averaged) reduction, a wrong denominator, or a broken pipeline
        would leave this exact value.
        """
        config = _build_config()
        model = gpt_builder(config, num_stages=PP_DEGREE, seg_method=SEG_METHOD)
        pp_model = distributed_model(model)

        with paddle.no_grad():
            for param in pp_model.parameters():
                param.set_value(paddle.zeros_like(param))

        inputs = _make_lm_batch(micro_batch_size=1, num_acc=4)
        loss = pp_model.forward_backward_pipeline(inputs, None)
        self.assertIsNotNone(loss)

        loss_arr = np.asarray(loss.astype("float32").numpy()).reshape(-1)
        self.assertEqual(loss_arr.size, 1, "pp loss must be a single scalar")
        loss_val = float(loss_arr[0])
        self.assertTrue(math.isfinite(loss_val), f"non-finite loss {loss_val}")

        expected = math.log(VOCAB)
        np.testing.assert_allclose(loss_val, expected, rtol=1e-3, atol=1e-3)

    def test_gpt_builder_layer_partition_is_complete_and_disjoint(self):
        """The builder splits the decoder into a clean cross-rank partition.

        Every ``TransformerLayer`` must live on exactly one pipeline stage.
        Gathering each stage's owned global layer numbers over the real pp
        process group must reconstruct ``0..NUM_LAYERS-1`` with no gaps and no
        duplicates, and every stage must own at least one layer. A dropped,
        duplicated, or mis-numbered layer changes the gathered union and fails
        the assertions. This is a genuine multi-rank contract: the check only
        passes if all four stages report consistent, non-overlapping slices.
        """
        config = _build_config()
        model = gpt_builder(config, num_stages=PP_DEGREE, seg_method=SEG_METHOD)

        pp_world = parallel_state.get_pipeline_model_parallel_world_size()
        self.assertEqual(pp_world, PP_DEGREE)

        local_numbers = sorted(
            int(layer.layer_number)
            for layer in model.run_function
            if isinstance(layer, TransformerLayer)
        )
        self.assertGreater(
            len(local_numbers), 0, "this stage owns no TransformerLayer"
        )

        gathered = []
        dist.all_gather_object(gathered, local_numbers)
        self.assertEqual(len(gathered), PP_DEGREE)

        flat = [num for per_rank in gathered for num in per_rank]
        self.assertEqual(
            sorted(flat),
            list(range(NUM_LAYERS)),
            f"gathered partition {gathered} is not a clean cover of "
            f"0..{NUM_LAYERS - 1}",
        )
        self.assertEqual(
            len(flat), len(set(flat)), f"layers duplicated: {gathered}"
        )

    @unittest.expectedFailure
    def test_use_fp8_vpp_returns_false(self):
        """CORRECT behaviour: a VPP model with fp8 disabled reports ``False``.

        ``config.fp8`` is ``None`` so every ``TransformerLayer.use_fp8()`` is
        ``False`` and ``GPTModel.use_fp8`` should return the Python bool
        ``False`` (callers branch on it). The virtual-pipeline branch in
        production is missing its trailing ``return False``, so it currently
        returns ``None``; asserting the correct ``is False`` fails today and is
        marked ``expectedFailure``. Adding the missing ``return False`` makes
        this an unexpected success, flagging the fix. Production is NOT edited.
        """
        config = _build_config(virtual_pipeline_model_parallel_size=2)
        self.assertIsNone(config.fp8)
        model = gpt_builder(config, num_stages=PP_DEGREE, seg_method=SEG_METHOD)
        self.assertGreater(
            getattr(model, "_num_virtual_pipeline_stages", 1),
            1,
            "expected the virtual-pipeline branch to be exercised",
        )
        self.assertIs(model.use_fp8(), False)

    @unittest.expectedFailure
    def test_estimate_flops_per_token_includes_lm_head(self):
        """CORRECT behaviour: non-MTP FLOPs/token must include the lm-head.

        Independent hand derivation for a dense, MHA, non-MTP model with the
        dimensions below (fwd+bwd factor 3, matmul factor 2, causal halving of
        the score/context term):

            mlp   = 3*2*H * (2 * (L*I))                = 12*H*L*I
            proj  = H*D * (2*A + 2*KV)
            attn  = (S*A*D*2) // 2                     (causal)
            attn_flops = 3*2*L * (proj + attn)
            lm_head    = 3*2*H*V                       (one output projection)

        The production ``output_logits_flops`` multiplies by 0 for non-MTP,
        dropping ``lm_head`` entirely, so ``estimate_flops_per_token`` is short
        by exactly ``lm_head``. Asserting the correct total (with lm-head)
        fails today -> ``expectedFailure``. Production is NOT edited.
        """
        H, I, L, A, D, KV, S, V = 16, 32, 2, 4, 4, 4, 8, 64
        estimator = GPTModelEstimator(
            seq_length=S,
            vocab_size=V,
            untie_embeddings_and_output_weights=False,
            num_hidden_layers=L,
            hidden_size=H,
            intermediate_size=I,
            gated_linear_unit=False,
            num_attention_heads=A,
            head_dim=D,
            num_kv_heads=KV,
            causal_mask=True,
            multi_latent_attention=False,
            moe_layer_freq=[0] * L,
            moe_num_experts=0,
            moe_intermediate_size=0,
            moe_shared_expert_intermediate_size=None,
            moe_topk=0,
            num_nextn_predict_layers=None,
        )

        mlp = 12 * H * L * I
        proj = H * D * (2 * A + 2 * KV)
        attn = (S * A * D * 2) // 2
        attn_flops = 3 * 2 * L * (proj + attn)
        lm_head = 3 * 2 * H * V
        expected = mlp + attn_flops + lm_head

        self.assertEqual(estimator.estimate_flops_per_token(), expected)


if __name__ == "__main__":
    unittest.main()
