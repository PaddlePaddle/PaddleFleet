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


import functools
import random
import unittest

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import NoPipelineParallel

import paddlefleet.parallel_state as ps
from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig
from paddlefleet.transformer.block_attn_res import BlockAttnRes


def _init_fleet(seed=46):
    """Seed everything and bring up a 1x1 hybrid-parallel environment."""
    random.seed(seed)
    np.random.seed(seed)
    paddle.manual_seed(seed)
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
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
    fleet.init(is_collective=True, strategy=strategy)
    # initialize_model_parallel() refuses to run twice, so drop the buffer a
    # previous test case in this process may have left behind.
    ps.destroy_global_memory_buffer()
    ps.initialize_model_parallel(fleet.get_hybrid_communicate_group())
    return strategy


class TestBlockAttnRes(unittest.TestCase):
    def setUp(self):
        self.strategy = _init_fleet()

    def test_block_attn_res(self):
        config = GPTConfig(
            num_hidden_layers=4,
            hidden_size=512,
            rotary_base=10000,
            vocab_size=100,
            rotary_percent=1.0,
            rope_scaling=1.0,
            position_embedding_type="rope",
            num_attention_heads=4,
            intermediate_size=1024,
            max_sequence_length=64,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            tie_word_embeddings=True,
            use_qk_norm=True,
            block_attention_residuals=True,
            attn_res_block_size=2,
        )
        gpt_model = gpt_builder(config, num_stages=1)

        sequence_length = config.max_sequence_length
        micro_batch_size = 1

        data = list(range(sequence_length))
        input_ids = paddle.to_tensor(data, dtype=paddle.int64).repeat(
            (micro_batch_size, 1)
        )
        position_ids = paddle.to_tensor(data, dtype=paddle.int64).repeat(
            (micro_batch_size, 1)
        )
        attention_mask = paddle.ones(
            (micro_batch_size, 1, sequence_length, sequence_length),
            dtype=bool,
        )
        labels = paddle.to_tensor(
            list(range(1, sequence_length + 1)), dtype=paddle.int64
        ).repeat((micro_batch_size, 1))

        gpt_pipe_model = NoPipelineParallel(gpt_model, self.strategy)
        data = (
            {
                "input_ids": [input_ids],
                "position_ids": [position_ids],
                "attention_mask": [attention_mask],
            },
            [labels],
        )

        loss = gpt_pipe_model.forward_backward_pipeline(data)

        # Verify loss is finite
        assert paddle.isfinite(loss).item(), (
            f"loss is not finite: {loss.item()}"
        )
        print("block_attn_res loss", loss.item())

        # Verify gradients exist and are finite
        for name, param in gpt_model.named_parameters():
            assert param.grad is not None, f"param {name} has no gradient"
            grad_norm = param.grad.detach().norm().item()
            assert np.isfinite(grad_norm), (
                f"param {name} has non-finite gradient: {grad_norm}"
            )

        # Verify block_attn_res parameters have gradients
        has_block_attn_res_param = False
        for name, param in gpt_model.named_parameters():
            if "block_attn_res" in name:
                has_block_attn_res_param = True
                assert param.grad is not None, (
                    f"block_attn_res param {name} has no gradient"
                )
        assert has_block_attn_res_param, (
            "No block_attn_res parameters found in model"
        )


class TestBlockAttnResFullRecompute(unittest.TestCase):
    """full_recompute must not change block attnres results.

    ``blocks`` is mutable state threaded across transformer layers: under full
    recompute it is passed into the recomputed forward as an immutable tuple
    while the authoritative append happens outside recompute. This test
    verifies that loss and gradients match between eager and recompute paths.

    The model crosses two block boundaries (4 layers, ``attn_res_block_size=4``
    => a block closes every 2 layers), so ``blocks`` is non-empty for most
    layers on both the forward pass and the backward re-run.
    """

    NUM_LAYERS = 4
    ATTN_RES_BLOCK_SIZE = 4

    def setUp(self):
        self.strategy = _init_fleet()

    def _make_config(self, full_recompute):
        recompute_kwargs = (
            {
                "recompute_granularity": "full",
                "recompute_method": "uniform",
                "recompute_num_layers": 1,
            }
            if full_recompute
            else {}
        )
        return GPTConfig(
            num_hidden_layers=self.NUM_LAYERS,
            hidden_size=128,
            rotary_base=10000,
            vocab_size=100,
            rotary_percent=1.0,
            rope_scaling=1.0,
            position_embedding_type="rope",
            num_attention_heads=4,
            intermediate_size=256,
            max_sequence_length=32,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            # Untied so the embedding gradient is a pure input-side gradient
            # instead of a sum of input and lm_head contributions.
            tie_word_embeddings=False,
            use_qk_norm=True,
            block_attention_residuals=True,
            attn_res_block_size=self.ATTN_RES_BLOCK_SIZE,
            **recompute_kwargs,
        )

    def _make_data(self, config):
        sequence_length = config.max_sequence_length
        tokens = list(range(sequence_length))
        return (
            {
                "input_ids": [
                    paddle.to_tensor(tokens, dtype=paddle.int64).repeat((1, 1))
                ],
                "position_ids": [
                    paddle.to_tensor(tokens, dtype=paddle.int64).repeat((1, 1))
                ],
                "attention_mask": [
                    paddle.ones(
                        (1, 1, sequence_length, sequence_length), dtype=bool
                    )
                ],
            },
            [
                paddle.to_tensor(
                    list(range(1, sequence_length + 1)), dtype=paddle.int64
                ).repeat((1, 1))
            ],
        )

    def _run(self, full_recompute, weights=None):
        """Forward+backward one step.

        Returns (loss, grads, blocks_seen, state_dict). ``blocks_seen`` is the
        sequence of ``len(blocks)`` handed to every BlockAttnRes call in
        execution order, i.e. what the block attnres math actually consumed.
        Under full recompute the backward re-run appends more entries to it.
        """
        config = self._make_config(full_recompute)
        model = gpt_builder(config, num_stages=1)
        if weights is not None:
            model.set_state_dict(weights)

        blocks_seen = []
        original_block_attn_res_forward = BlockAttnRes.forward

        def spying_forward(layer_self, partial_block, blocks, *args, **kwargs):
            blocks_seen.append(0 if blocks is None else len(blocks))
            return original_block_attn_res_forward(
                layer_self, partial_block, blocks, *args, **kwargs
            )

        BlockAttnRes.forward = spying_forward
        try:
            loss = NoPipelineParallel(
                model, self.strategy
            ).forward_backward_pipeline(self._make_data(config))
        finally:
            BlockAttnRes.forward = original_block_attn_res_forward

        grads = {
            name: param.grad.detach().clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }
        return loss.item(), grads, blocks_seen, model.state_dict()

    def test_full_recompute_matches_eager(self):
        eager_loss, eager_grads, eager_blocks, weights = self._run(False)
        recompute_loss, recompute_grads, recompute_blocks, _ = self._run(
            True, weights=weights
        )

        # --- Block bookkeeping ---------------------------------------------
        # The run must actually accumulate blocks, otherwise everything below
        # would still pass with block propagation broken in both paths.
        self.assertGreater(
            max(eager_blocks),
            0,
            f"no block boundary was crossed: {eager_blocks}",
        )
        # The forward pass must consume blocks exactly as it does without
        # recompute.
        self.assertEqual(
            recompute_blocks[: len(eager_blocks)],
            eager_blocks,
            "full recompute changed the blocks seen during forward",
        )

        # --- Loss ----------------------------------------------------------
        self.assertAlmostEqual(eager_loss, recompute_loss, places=5)

        # --- Gradients -----------------------------------------------------
        self.assertEqual(sorted(eager_grads), sorted(recompute_grads))
        for name, eager_grad in eager_grads.items():
            max_diff = (eager_grad - recompute_grads[name]).abs().max().item()
            self.assertLess(
                max_diff,
                1e-5,
                f"gradient mismatch for {name}: max abs diff {max_diff}",
            )

        # The input-side gradient and the block attnres parameters are the two
        # things a lost or duplicated block corrupts first, so pin them down
        # explicitly rather than relying on the sweep above.
        input_grads = [n for n in eager_grads if "embed_tokens" in n]
        self.assertTrue(
            input_grads, f"no input gradient found: {list(eager_grads)}"
        )
        block_attn_res_grads = [n for n in eager_grads if "block_attn_res" in n]
        self.assertTrue(
            block_attn_res_grads,
            f"no block_attn_res gradient found: {list(eager_grads)}",
        )
        for name in input_grads + block_attn_res_grads:
            np.testing.assert_allclose(
                eager_grads[name].numpy(),
                recompute_grads[name].numpy(),
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"gradient mismatch for {name}",
            )


if __name__ == "__main__":
    unittest.main()
