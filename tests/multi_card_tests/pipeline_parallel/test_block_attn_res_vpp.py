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

# Regression monitor for the BlockAttnRes pipeline communication optimization.
#
# The optimization only changes how residual-block gradients are communicated
# in the backward pass; the forward and the loss must stay identical, and after
# a full backward the per-parameter gradients must match the non-optimized run
# within a small tolerance. We therefore run the same model / data / init twice
# -- once with BLOCK_ATTEN_RES_COMM_OPT off and once on -- and compare loss and
# gradients.
#
# The optimization lives in PaddlePaddle (PipelineParallelWithInterleave). Until
# that PR is merged, the installed paddle has no such feature, so this test
# skips itself; it activates automatically once paddle ships the feature.
import functools
import os
import random
import unittest

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet import distributed_model
from paddle.distributed.fleet.meta_parallel import (
    PipelineParallelWithInterleave,
)

import paddlefleet
from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig
from paddlefleet.training.initialize import initialize_fleet

PP_DEGREE = 4

# The comm optimization adds these methods to PipelineParallelWithInterleave.
# If the installed paddle does not have them, the feature is not merged yet.
FEATURE_AVAILABLE = hasattr(
    PipelineParallelWithInterleave, "_merge_block_cache"
)

# Tolerances. Forward is unaffected -> loss should match tightly. Gradients go
# through the optimized communication path -> allow a small tolerance.
LOSS_RTOL, LOSS_ATOL = 1e-5, 1e-6
GRAD_RTOL, GRAD_ATOL = 1e-3, 1e-5


def _set_random_seed(seed_: int):
    """Set random seed for reproducibility (per pipeline stage)."""
    seed = seed_ + (
        100 * paddlefleet.parallel_state.get_pipeline_model_parallel_rank()
    )
    random.seed(seed)
    np.random.seed(seed)
    paddle.manual_seed(seed)
    if paddle.distributed.is_initialized() and paddle.cuda.device_count() > 0:
        paddlefleet.tensor_parallel.model_parallel_cuda_manual_seed(seed)


def _make_config(vocab_size, seq_len):
    return GPTConfig(
        vocab_size=vocab_size,
        max_sequence_length=seq_len,
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
        num_empty_layers_add_in_head=2,
        num_empty_layers_add_in_tail=3,
        pipeline_model_parallel_size=PP_DEGREE,
        virtual_pipeline_model_parallel_size=2,
        block_attention_residuals=True,
        attn_res_block_size=2,
    )


def _build_and_run(config, seed, inputs, comm_opt):
    """Build a fresh model (identical init for a given seed) and run one
    forward+backward with the comm optimization on/off. Returns (loss, grads)."""
    os.environ["BLOCK_ATTEN_RES_COMM_OPT"] = "1" if comm_opt else "0"

    _set_random_seed(seed)
    gpt_model = gpt_builder(
        config,
        num_stages=config.pipeline_model_parallel_size,
        seg_method="layer:TransformerLayer|EmptyLayer",
    )
    pipe_model = distributed_model(gpt_model)

    # Confirm the switch really took effect for this build.
    assert pipe_model._block_atten_res_opt is comm_opt, (
        f"_block_atten_res_opt={pipe_model._block_atten_res_opt}, "
        f"expected {comm_opt}"
    )

    loss = pipe_model.forward_backward_pipeline(inputs, None)

    grads = {
        name: param.grad.detach().astype("float32").numpy().copy()
        for name, param in gpt_model.named_parameters()
        if param.grad is not None
    }
    return loss, grads


def _global_grad_norm(grads):
    total = 0.0
    for g in grads.values():
        total += float(np.sum(np.square(g)))
    return float(np.sqrt(total))


class TestBlockAttnResVPPEquivalence(unittest.TestCase):
    def setUp(self):
        self.seed = 46
        self.batch_size = 12
        self.seq_len = 128
        self.vocab_size = 1024
        self._env_backup = os.environ.get("BLOCK_ATTEN_RES_COMM_OPT")

    def tearDown(self):
        # _build_and_run flips this switch; do not leak it to other tests.
        if self._env_backup is None:
            os.environ.pop("BLOCK_ATTEN_RES_COMM_OPT", None)
        else:
            os.environ["BLOCK_ATTEN_RES_COMM_OPT"] = self._env_backup

    @unittest.skipUnless(
        FEATURE_AVAILABLE,
        "BlockAttnRes comm optimization is not present in the installed "
        "paddle yet.",
    )
    def test_block_attn_res_opt_matches_baseline(self):
        config = _make_config(self.vocab_size, self.seq_len)

        micro_batch_size = 1
        num_acc = self.batch_size // micro_batch_size

        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": config.pipeline_model_parallel_size,
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
            # BlockAttnRes comm opt does not support the overlap scheduler.
            # BlockAttnRes sends "hidden + blocks", and the number of blocks
            # grows along the pipeline, so the p2p shape meta must not be
            # cached across sends -> dynamic shape is required (VPP itself
            # requires p2p_cache_shape, so this is the only way).
            "pp_configs": {
                "forward_backward_overlap_scheduler": False,
                "overlap_p2p_comm": False,
                "enable_dynamic_shape": True,
            },
        }
        strategy.pipeline_configs = {
            "accumulate_steps": num_acc,
            "micro_batch_size": micro_batch_size,
        }
        initialize_fleet(strategy)

        # Build one fixed batch and reuse it for both runs.
        _set_random_seed(self.seed)
        data = paddle.randint(
            low=0,
            high=self.vocab_size,
            shape=(micro_batch_size, self.seq_len + 1),
        )
        input_ids = data[:, :-1]
        labels = data[:, 1:]
        position_ids = (
            paddle.arange(self.seq_len, dtype=paddle.int64)
            .unsqueeze(0)
            .expand([micro_batch_size, -1])
        )
        inputs = (
            {
                "input_ids": [input_ids] * num_acc,
                "position_ids": [position_ids] * num_acc,
            },
            [labels] * num_acc,
        )

        # Baseline (optimization off) then optimized (on), identical init/data.
        loss_off, grads_off = _build_and_run(
            config, self.seed, inputs, comm_opt=False
        )
        loss_on, grads_on = _build_and_run(
            config, self.seed, inputs, comm_opt=True
        )

        # 1) Forward is unaffected: loss must match tightly (last stage only).
        if loss_off is not None and loss_on is not None:
            off = loss_off.astype("float32").numpy()
            on = loss_on.astype("float32").numpy()
            assert np.all(np.isfinite(off)) and np.all(np.isfinite(on)), (
                f"non-finite loss: off={off}, on={on}"
            )
            np.testing.assert_allclose(
                on,
                off,
                rtol=LOSS_RTOL,
                atol=LOSS_ATOL,
                err_msg="loss differs between comm-opt on/off",
            )

        # 2) Same set of parameters must receive gradients in both runs.
        assert set(grads_off.keys()) == set(grads_on.keys()), (
            "gradient parameter sets differ between runs: "
            f"only_off={set(grads_off) - set(grads_on)}, "
            f"only_on={set(grads_on) - set(grads_off)}"
        )
        assert len(grads_on) > 0, "no gradients captured"

        # 3) Per-parameter gradients must match within tolerance (this covers
        #    the last VPP chunk parameters too).
        for name in sorted(grads_on.keys()):
            g_on, g_off = grads_on[name], grads_off[name]
            assert np.all(np.isfinite(g_on)), f"non-finite grad (on): {name}"
            np.testing.assert_allclose(
                g_on,
                g_off,
                rtol=GRAD_RTOL,
                atol=GRAD_ATOL,
                err_msg=f"gradient mismatch for param {name}",
            )

        # 4) Global grad-norm relative diff must be tiny.
        norm_off = _global_grad_norm(grads_off)
        norm_on = _global_grad_norm(grads_on)
        rel = abs(norm_on - norm_off) / (norm_off + 1e-12)
        assert rel < 1e-3, (
            f"grad_norm relative diff too large: off={norm_off}, "
            f"on={norm_on}, rel={rel}"
        )


if __name__ == "__main__":
    unittest.main()
