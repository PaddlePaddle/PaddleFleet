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

"""Real multi-card (PP=4) behavior tests for ``paddlefleet.models.gpt.GPTModel``.

Run under ``paddle.distributed.launch`` with 4 GPUs (PP_DEGREE=4). Two real GPT
pipeline models are built on the live 4-rank pipeline topology:

* ``model_no_vpp`` -- plain pipeline (virtual_pipeline_model_parallel_size=1),
  so ``GPTModel.use_fp8`` takes the non-VPP ``else`` branch.
* ``model_vpp``    -- interleaved pipeline (virtual_pipeline_model_parallel_size=2),
  so ``GPTModel.use_fp8`` takes the VPP branch that walks ``_model_chunks``.

Independent references (never taken from the code under test):

* ``use_fp8`` must return the Python bool ``False`` for a model whose config has
  ``fp8 is None`` on every rank, because every ``TransformerLayer.use_fp8`` is
  ``self.config.fp8 is not None`` == ``False``. The correct answer is the exact
  object ``False`` -- not ``None`` -- so callers can branch on it.
* The initial cross-entropy of an untrained language model over ``V`` classes is
  ~ ``ln(V)`` (near-uniform softmax at initialization). This anchors the
  pipeline forward/backward loss without calling the model to produce its own
  expected value.

Known production bug (do NOT edit production to satisfy this test):
``gpt_model.py`` ``GPTModel.use_fp8`` -- the ``_num_virtual_pipeline_stages > 1``
branch iterates the chunks and ``return True`` on a match but has no trailing
``return False``, so a VPP model without fp8 returns ``None`` instead of
``False``. ``test_use_fp8_vpp_returns_false_bug`` asserts the CORRECT behavior
(``is False``) and is marked ``@unittest.expectedFailure``; when the missing
``return False`` is added, it flips to an unexpected success and fails, flagging
that the guard here can be removed.
"""

import functools
import math
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
SEED = 46
MICRO_BATCH_SIZE = 1
BATCH_SIZE = 4
SEQ_LEN = 128
VOCAB_SIZE = 1024


def _set_random_seed(seed_):
    """Give each pipeline stage a distinct but reproducible seed."""
    if seed_ is None or seed_ <= 0:
        raise ValueError(f"Seed ({seed_}) should be a positive integer.")
    stage = paddlefleet.parallel_state.get_pipeline_model_parallel_rank()
    seed = seed_ + 100 * stage
    np.random.seed(seed)
    paddle.manual_seed(seed)
    if paddle.distributed.is_initialized() and paddle.cuda.device_count() > 0:
        paddlefleet.tensor_parallel.model_parallel_cuda_manual_seed(seed)


def _make_config(virtual_pipeline_model_parallel_size):
    """Build a GPTConfig with fp8 left disabled (config.fp8 is None)."""
    return GPTConfig(
        vocab_size=VOCAB_SIZE,
        max_sequence_length=SEQ_LEN,
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
        use_qk_norm=True,
        num_empty_layers_add_in_head=2,
        num_empty_layers_add_in_tail=3,
        init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        output_layer_init_method=functools.partial(
            paddle.nn.init.xavier_uniform_, gain=1.0
        ),
        pipeline_model_parallel_size=PP_DEGREE,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
    )


def _build_model(config):
    return gpt_builder(
        config,
        num_stages=config.pipeline_model_parallel_size,
        seg_method="layer:TransformerLayer|EmptyLayer",
    )


@unittest.skipUnless(
    paddle.is_compiled_with_cuda(), "GPTModel PP tests require a CUDA build"
)
class TestGPTModelPipelineParallel(unittest.TestCase):
    """PP=4 behavior tests for GPTModel.use_fp8 and pipeline forward/backward."""

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
        num_acc = BATCH_SIZE // MICRO_BATCH_SIZE
        strategy.pipeline_configs = {
            "accumulate_steps": num_acc,
            "micro_batch_size": MICRO_BATCH_SIZE,
        }
        initialize_fleet(strategy)
        _set_random_seed(SEED)

        # Raw GPTModel (PipelineLayer) instances; use_fp8 is called on these.
        cls.model_no_vpp = _build_model(
            _make_config(virtual_pipeline_model_parallel_size=1)
        )
        cls.model_vpp = _build_model(
            _make_config(virtual_pipeline_model_parallel_size=2)
        )
        # Only the plain-PP model is wrapped for the real forward/backward run.
        cls.pipe_no_vpp = distributed_model(cls.model_no_vpp)

    def test_use_fp8_no_vpp_returns_false(self):
        """Non-VPP branch: real per-rank layers, fp8 disabled -> exactly False.

        Each rank's ``run_function`` holds a distinct slice of the pipeline, so
        this exercises the real per-rank iteration. The contract is the object
        ``False`` (``assertIs``): a stray ``None`` or ``True`` is rejected.
        """
        self.assertTrue(self.model_no_vpp._num_virtual_pipeline_stages == 1)
        self.assertIs(self.model_no_vpp.use_fp8(), False)

    @unittest.expectedFailure
    def test_use_fp8_vpp_returns_false_bug(self):
        """VPP branch: fp8 disabled must yield ``False`` (correct behavior).

        Production ``GPTModel.use_fp8`` omits ``return False`` after the VPP
        chunk loop, so it returns ``None`` here. Asserting the correct ``False``
        currently fails -- marked expectedFailure. Fixing production (adding the
        missing ``return False``) turns this into an unexpected success, which
        unittest reports as a failure, signaling this guard can be dropped.
        """
        self.assertGreater(self.model_vpp._num_virtual_pipeline_stages, 1)
        self.assertIs(self.model_vpp.use_fp8(), False)

    def test_pp_forward_backward_initial_loss_and_grads(self):
        """Real PP forward+backward: loss ~ ln(V) and gradients actually flow.

        ``ln(VOCAB_SIZE)`` is the independent reference for an untrained LM's
        cross-entropy (near-uniform softmax at init). The reduced loss must be a
        single finite scalar inside a band around it -- this rejects ``None``,
        ``NaN``, a collapsed ~0 loss and a wrong (summed) reduction. Backward
        must leave finite, not-all-zero gradients on this rank's parameters.
        """
        num_acc = BATCH_SIZE // MICRO_BATCH_SIZE
        data = paddle.randint(
            low=0, high=VOCAB_SIZE, shape=(MICRO_BATCH_SIZE, SEQ_LEN + 1)
        ).cuda()
        input_ids = data[:, :-1]
        labels = data[:, 1:]
        position_ids = (
            paddle.arange(0, SEQ_LEN, dtype="int64")
            .unsqueeze(0)
            .expand([MICRO_BATCH_SIZE, -1])
            .cuda()
        )
        inputs = (
            {
                "input_ids": [input_ids] * num_acc,
                "position_ids": [position_ids] * num_acc,
            },
            [labels] * num_acc,
        )

        loss = self.pipe_no_vpp.forward_backward_pipeline(inputs, None)
        self.assertIsNotNone(loss)

        loss_arr = np.asarray(loss.astype("float32").numpy()).reshape(-1)
        self.assertEqual(
            loss_arr.size, 1, "reduced pipeline loss must be a single scalar"
        )
        loss_val = float(loss_arr[0])
        self.assertTrue(math.isfinite(loss_val), "loss is not finite")

        ref = math.log(VOCAB_SIZE)  # independent: untrained-LM CE ~ ln(V)
        self.assertGreater(loss_val, 0.5 * ref)
        self.assertLess(loss_val, 1.6 * ref)

        grad_names = []
        any_nonzero = False
        for name, param in self.pipe_no_vpp.named_parameters():
            if param.grad is None:
                continue
            grad = np.asarray(param.grad.astype("float32").numpy())
            self.assertTrue(
                np.isfinite(grad).all(), f"non-finite gradient in {name}"
            )
            grad_names.append(name)
            if np.abs(grad).max() > 0.0:
                any_nonzero = True
        self.assertGreater(
            len(grad_names), 0, "no parameter on this rank received a gradient"
        )
        self.assertTrue(any_nonzero, "every gradient on this rank is zero")


if __name__ == "__main__":
    unittest.main()
