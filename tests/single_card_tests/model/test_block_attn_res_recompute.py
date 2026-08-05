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

"""
Unit tests for commit 9d30001: attn res support recompute.

Covers:
- BlockAttnResFunc (PyLayer forward/backward)
- TransformerLayer._should_skip_block_attn_res
- TransformerLayer._is_block_boundary
- TransformerLayer._forward_impl_block_attn_res_split_recompute
- TransformerLayer.forward dispatch (block_attn_res + full_recompute)
- BlockAttnRes.forward training vs eval path
"""

import functools
import random
import unittest
from unittest.mock import MagicMock

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import NoPipelineParallel

import paddlefleet.parallel_state as ps
from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig
from paddlefleet.transformer.block_attn_res import (
    BlockAttnRes,
    BlockAttnResFunc,
)


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
    ps.destroy_global_memory_buffer()
    ps.initialize_model_parallel(fleet.get_hybrid_communicate_group())
    return strategy


# ---------------------------------------------------------------------------
# Test 1: BlockAttnResFunc PyLayer forward/backward correctness
# ---------------------------------------------------------------------------
class TestBlockAttnResFunc(unittest.TestCase):
    """Test BlockAttnResFunc PyLayer directly."""

    def setUp(self):
        paddle.manual_seed(42)
        self.B, self.S, self.H = 2, 8, 16
        self.norm_eps = 1e-6

    def _make_inputs(self, num_blocks=2, requires_grad=True):
        partial_block = paddle.randn([self.B, self.S, self.H])
        partial_block.stop_gradient = not requires_grad
        proj_weight = paddle.randn([self.H])
        proj_weight.stop_gradient = not requires_grad
        norm_weight = paddle.ones([self.H])
        norm_weight.stop_gradient = not requires_grad
        blocks = []
        for _ in range(num_blocks):
            b = paddle.randn([self.B, self.S, self.H])
            b.stop_gradient = not requires_grad
            blocks.append(b)
        return partial_block, proj_weight, norm_weight, blocks

    def _reference_forward(
        self, partial_block, proj_weight, norm_weight, blocks
    ):
        """Reference implementation without PyLayer."""
        all_repr = [*blocks, partial_block]
        logits_list = []
        for repr_i in all_repr:
            variance = (
                repr_i.astype("float32").pow(2).mean(axis=-1, keepdim=True)
            )
            rms = paddle.sqrt(variance + self.norm_eps)
            k_i = repr_i / rms * norm_weight
            logit_i = (k_i * proj_weight).sum(axis=-1)
            logits_list.append(logit_i)
        logits = paddle.stack(logits_list, axis=0)
        weights = paddle.nn.functional.softmax(logits, axis=0)
        h = weights[0].unsqueeze(-1) * all_repr[0]
        for i in range(1, len(all_repr)):
            h = h + weights[i].unsqueeze(-1) * all_repr[i]
        return h

    def test_forward_matches_reference(self):
        """BlockAttnResFunc.forward output matches naive implementation."""
        partial_block, proj_weight, norm_weight, blocks = self._make_inputs(
            num_blocks=3, requires_grad=False
        )
        # PyLayer forward
        result = BlockAttnResFunc.apply(
            partial_block, proj_weight, norm_weight, self.norm_eps, *blocks
        )
        # Reference
        ref = self._reference_forward(
            partial_block, proj_weight, norm_weight, blocks
        )
        np.testing.assert_allclose(
            result.numpy(), ref.numpy(), rtol=1e-5, atol=1e-6
        )

    def test_forward_single_block(self):
        """Works with only one block (minimum case)."""
        partial_block, proj_weight, norm_weight, _ = self._make_inputs(
            num_blocks=0, requires_grad=False
        )
        # No blocks — only partial_block
        result = BlockAttnResFunc.apply(
            partial_block, proj_weight, norm_weight, self.norm_eps
        )
        # With zero blocks, all_repr = [partial_block], softmax over 1 element = 1.0
        # So output should equal partial_block
        np.testing.assert_allclose(
            result.numpy(), partial_block.numpy(), rtol=1e-5, atol=1e-6
        )

    def test_backward_gradients(self):
        """Backward produces finite gradients for all inputs."""
        partial_block, proj_weight, norm_weight, blocks = self._make_inputs(
            num_blocks=2, requires_grad=True
        )
        result = BlockAttnResFunc.apply(
            partial_block, proj_weight, norm_weight, self.norm_eps, *blocks
        )
        loss = result.sum()
        loss.backward()

        self.assertIsNotNone(partial_block.grad)
        self.assertTrue(paddle.isfinite(partial_block.grad).all().item())
        self.assertIsNotNone(proj_weight.grad)
        self.assertTrue(paddle.isfinite(proj_weight.grad).all().item())
        self.assertIsNotNone(norm_weight.grad)
        self.assertTrue(paddle.isfinite(norm_weight.grad).all().item())
        for i, b in enumerate(blocks):
            self.assertIsNotNone(b.grad, f"block {i} has no gradient")
            self.assertTrue(paddle.isfinite(b.grad).all().item())

    def test_backward_grad_numerical_check(self):
        """Compare PyLayer gradients vs autograd reference gradients."""
        partial_block, proj_weight, norm_weight, blocks = self._make_inputs(
            num_blocks=2, requires_grad=True
        )
        # Run PyLayer path
        result_pylayer = BlockAttnResFunc.apply(
            partial_block, proj_weight, norm_weight, self.norm_eps, *blocks
        )
        loss_pylayer = result_pylayer.sum()
        loss_pylayer.backward()
        grad_partial_pylayer = partial_block.grad.clone()
        grad_proj_pylayer = proj_weight.grad.clone()
        grad_norm_pylayer = norm_weight.grad.clone()
        grad_blocks_pylayer = [b.grad.clone() for b in blocks]

        # Clear grads
        partial_block.clear_gradient()
        proj_weight.clear_gradient()
        norm_weight.clear_gradient()
        for b in blocks:
            b.clear_gradient()

        # Run reference path (standard autograd)
        ref_result = self._reference_forward(
            partial_block, proj_weight, norm_weight, blocks
        )
        loss_ref = ref_result.sum()
        loss_ref.backward()

        np.testing.assert_allclose(
            grad_partial_pylayer.numpy(),
            partial_block.grad.numpy(),
            rtol=1e-4,
            atol=1e-5,
            err_msg="partial_block grad mismatch",
        )
        np.testing.assert_allclose(
            grad_proj_pylayer.numpy(),
            proj_weight.grad.numpy(),
            rtol=1e-4,
            atol=1e-5,
            err_msg="proj_weight grad mismatch",
        )
        np.testing.assert_allclose(
            grad_norm_pylayer.numpy(),
            norm_weight.grad.numpy(),
            rtol=1e-4,
            atol=1e-5,
            err_msg="norm_weight grad mismatch",
        )
        for i, (g_py, b) in enumerate(zip(grad_blocks_pylayer, blocks)):
            np.testing.assert_allclose(
                g_py.numpy(),
                b.grad.numpy(),
                rtol=1e-4,
                atol=1e-5,
                err_msg=f"block[{i}] grad mismatch",
            )

    def test_backward_no_grad_inputs(self):
        """Inputs with stop_gradient=True get None gradients."""
        partial_block, proj_weight, norm_weight, blocks = self._make_inputs(
            num_blocks=2, requires_grad=True
        )
        # Mark proj_weight as not needing grad
        proj_weight.stop_gradient = True
        blocks[0].stop_gradient = True

        result = BlockAttnResFunc.apply(
            partial_block, proj_weight, norm_weight, self.norm_eps, *blocks
        )
        loss = result.sum()
        loss.backward()

        self.assertIsNone(proj_weight.grad)
        self.assertIsNone(blocks[0].grad)
        # Others should still have grads
        self.assertIsNotNone(partial_block.grad)
        self.assertIsNotNone(norm_weight.grad)
        self.assertIsNotNone(blocks[1].grad)


# ---------------------------------------------------------------------------
# Test 2: BlockAttnRes module training vs eval path
# ---------------------------------------------------------------------------
class TestBlockAttnResModule(unittest.TestCase):
    """Test BlockAttnRes module forward in training and eval modes."""

    def setUp(self):
        self.strategy = _init_fleet()
        self.B, self.S, self.H = 2, 8, 64

    def _make_config(self):
        return GPTConfig(
            num_hidden_layers=2,
            hidden_size=self.H,
            rotary_base=10000,
            vocab_size=100,
            rotary_percent=1.0,
            rope_scaling=1.0,
            position_embedding_type="rope",
            num_attention_heads=4,
            intermediate_size=128,
            max_sequence_length=self.S,
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

    def test_training_uses_pylayer(self):
        """In training mode, BlockAttnRes uses BlockAttnResFunc."""
        config = self._make_config()
        model = gpt_builder(config, num_stages=1)
        model.train()

        # Find a BlockAttnRes submodule
        bar = None
        for m in model.sublayers():
            if isinstance(m, BlockAttnRes):
                bar = m
                break
        self.assertIsNotNone(bar, "No BlockAttnRes found in model")

        partial_block = paddle.randn([self.B, self.S, self.H])
        partial_block.stop_gradient = False
        blocks = [paddle.randn([self.B, self.S, self.H])]
        blocks[0].stop_gradient = False

        bar.train()
        result = bar(partial_block, blocks)
        # Should be able to backward
        loss = result.sum()
        loss.backward()
        self.assertIsNotNone(partial_block.grad)

    def test_eval_no_pylayer(self):
        """In eval mode, BlockAttnRes uses standard forward (no PyLayer)."""
        config = self._make_config()
        model = gpt_builder(config, num_stages=1)
        model.eval()

        bar = None
        for m in model.sublayers():
            if isinstance(m, BlockAttnRes):
                bar = m
                break
        self.assertIsNotNone(bar)

        partial_block = paddle.randn([self.B, self.S, self.H])
        blocks = [paddle.randn([self.B, self.S, self.H])]

        bar.eval()
        with paddle.no_grad():
            result = bar(partial_block, blocks)
        self.assertTrue(paddle.isfinite(result).all().item())

    def test_training_eval_consistent(self):
        """Training and eval paths produce same output."""
        config = self._make_config()
        model = gpt_builder(config, num_stages=1)

        bar = None
        for m in model.sublayers():
            if isinstance(m, BlockAttnRes):
                bar = m
                break
        self.assertIsNotNone(bar)

        partial_block = paddle.randn([self.B, self.S, self.H])
        blocks = [paddle.randn([self.B, self.S, self.H])]

        bar.train()
        out_train = bar(partial_block, blocks)

        bar.eval()
        with paddle.no_grad():
            out_eval = bar(partial_block, blocks)

        np.testing.assert_allclose(
            out_train.detach().numpy(),
            out_eval.numpy(),
            rtol=1e-5,
            atol=1e-6,
            err_msg="Training and eval outputs differ",
        )


# ---------------------------------------------------------------------------
# Test 3: _should_skip_block_attn_res and _is_block_boundary
# ---------------------------------------------------------------------------
class TestTransformerLayerHelpers(unittest.TestCase):
    """Test helper methods added in commit 9d30001."""

    def setUp(self):
        self.strategy = _init_fleet()

    def _make_config(self, block_size=4, num_layers=4):
        return GPTConfig(
            num_hidden_layers=num_layers,
            hidden_size=64,
            rotary_base=10000,
            vocab_size=100,
            rotary_percent=1.0,
            rope_scaling=1.0,
            position_embedding_type="rope",
            num_attention_heads=4,
            intermediate_size=128,
            max_sequence_length=16,
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
            attn_res_block_size=block_size,
        )

    def test_should_skip_for_normal_layer(self):
        """Normal (non-MTP) layers should NOT skip block attn res."""
        config = self._make_config()
        model = gpt_builder(config, num_stages=1)
        from paddlefleet.transformer.transformer_layer import TransformerLayer

        for m in model.sublayers():
            if isinstance(m, TransformerLayer):
                self.assertFalse(m._should_skip_block_attn_res())
                break

    def test_is_block_boundary(self):
        """_is_block_boundary correct for block_size=4 (span=4, K3 semantics)."""
        config = self._make_config(block_size=4, num_layers=6)
        model = gpt_builder(config, num_stages=1)
        from paddlefleet.transformer.transformer_layer import TransformerLayer

        boundaries = []
        for m in model.sublayers():
            if isinstance(m, TransformerLayer):
                if m._is_block_boundary():
                    boundaries.append(m.layer_number)
        # block_span == attn_res_block_size == 4 (matches K3's
        # `layer_idx % attn_res_block_size == 0`), so layers 0 and 4 are
        # boundaries.
        self.assertTrue(len(boundaries) > 0)
        for b in boundaries:
            self.assertEqual(b % 4, 0)

    def test_is_block_boundary_block_size_2(self):
        """block_size=2 => span=2, every other layer is a boundary."""
        config = self._make_config(block_size=2, num_layers=4)
        model = gpt_builder(config, num_stages=1)
        from paddlefleet.transformer.transformer_layer import TransformerLayer

        checked = 0
        for m in model.sublayers():
            if isinstance(m, TransformerLayer):
                if hasattr(m, "attn_res_block_size") and m.attn_res_block_size:
                    self.assertEqual(
                        m._is_block_boundary(), m.layer_number % 2 == 0
                    )
                    checked += 1
        self.assertTrue(checked > 0)


# ---------------------------------------------------------------------------
# Test 4: _forward_impl_block_attn_res_split_recompute
# ---------------------------------------------------------------------------
class TestSplitRecompute(unittest.TestCase):
    """Test block_attn_res + full_recompute split path."""

    def setUp(self):
        self.strategy = _init_fleet()

    def _make_config(self, full_recompute=True):
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
            num_hidden_layers=4,
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
            tie_word_embeddings=False,
            use_qk_norm=True,
            block_attention_residuals=True,
            attn_res_block_size=4,
            **recompute_kwargs,
        )

    def _make_data(self, config):
        seq_len = config.max_sequence_length
        tokens = list(range(seq_len))
        return (
            {
                "input_ids": [
                    paddle.to_tensor(tokens, dtype=paddle.int64).reshape(
                        [1, -1]
                    )
                ],
                "position_ids": [
                    paddle.to_tensor(tokens, dtype=paddle.int64).reshape(
                        [1, -1]
                    )
                ],
                "attention_mask": [
                    paddle.ones((1, 1, seq_len, seq_len), dtype=bool)
                ],
            },
            [
                paddle.to_tensor(
                    list(range(1, seq_len + 1)), dtype=paddle.int64
                ).reshape([1, -1])
            ],
        )

    def _run_model(self, full_recompute, weights=None):
        config = self._make_config(full_recompute)
        model = gpt_builder(config, num_stages=1)
        if weights is not None:
            model.set_state_dict(weights)
        loss = NoPipelineParallel(
            model, self.strategy
        ).forward_backward_pipeline(self._make_data(config))
        grads = {
            name: param.grad.detach().clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }
        return loss.item(), grads, model.state_dict()

    def test_split_recompute_loss_matches(self):
        """Loss with full_recompute matches eager (no recompute)."""
        eager_loss, eager_grads, weights = self._run_model(False)
        recompute_loss, recompute_grads, _ = self._run_model(True, weights)

        self.assertAlmostEqual(
            eager_loss,
            recompute_loss,
            places=4,
            msg=f"Loss mismatch: eager={eager_loss}, recompute={recompute_loss}",
        )

    def test_split_recompute_grads_match(self):
        """Gradients with full_recompute match eager."""
        _, eager_grads, weights = self._run_model(False)
        _, recompute_grads, _ = self._run_model(True, weights)

        self.assertEqual(
            sorted(eager_grads.keys()), sorted(recompute_grads.keys())
        )
        for name in eager_grads:
            max_diff = (
                (eager_grads[name] - recompute_grads[name]).abs().max().item()
            )
            self.assertLess(
                max_diff,
                1e-4,
                f"Grad mismatch for {name}: max diff = {max_diff}",
            )

    def test_split_recompute_block_attn_res_grads(self):
        """block_attn_res parameters get correct gradients under recompute."""
        _, eager_grads, weights = self._run_model(False)
        _, recompute_grads, _ = self._run_model(True, weights)

        bar_params = [n for n in eager_grads if "block_attn_res" in n]
        self.assertTrue(bar_params, "No block_attn_res params found")
        for name in bar_params:
            np.testing.assert_allclose(
                eager_grads[name].numpy(),
                recompute_grads[name].numpy(),
                rtol=1e-4,
                atol=1e-5,
                err_msg=f"block_attn_res grad mismatch: {name}",
            )


# ---------------------------------------------------------------------------
# Test 5: MTP layer skips block_attn_res (IdentityOp)
# ---------------------------------------------------------------------------
class TestMTPSkipsBlockAttnRes(unittest.TestCase):
    """MTP layers should use IdentityOp for block_attn_res."""

    def setUp(self):
        self.strategy = _init_fleet()

    def test_mtp_layer_has_identity_op(self):
        """When block_attention_residuals is on, MTP layers get IdentityOp."""
        config = GPTConfig(
            num_hidden_layers=4,
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
            tie_word_embeddings=True,
            use_qk_norm=True,
            block_attention_residuals=True,
            attn_res_block_size=4,
            num_nextn_predict_layers=1,
        )
        model = gpt_builder(config, num_stages=1)
        from paddlefleet.transformer.identity_op import IdentityOp
        from paddlefleet.transformer.transformer_layer import TransformerLayer

        for m in model.sublayers():
            if isinstance(m, TransformerLayer) and getattr(
                m, "is_mtp_layer", False
            ):
                self.assertTrue(m._should_skip_block_attn_res())
                self.assertIsInstance(
                    m.block_attn_res_before_attention, IdentityOp
                )
                self.assertIsInstance(m.block_attn_res_before_mlp, IdentityOp)


# ---------------------------------------------------------------------------
# Test 6: Forward dispatch - block_attn_res without recompute
# ---------------------------------------------------------------------------
class TestForwardDispatchNoRecompute(unittest.TestCase):
    """block_attn_res + no recompute uses _forward_impl path."""

    def setUp(self):
        self.strategy = _init_fleet()

    def test_forward_backward_no_recompute(self):
        """Model with block_attn_res but no recompute trains correctly."""
        config = GPTConfig(
            num_hidden_layers=4,
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
            tie_word_embeddings=True,
            use_qk_norm=True,
            block_attention_residuals=True,
            attn_res_block_size=4,
        )
        model = gpt_builder(config, num_stages=1)
        seq_len = config.max_sequence_length
        tokens = list(range(seq_len))
        data = (
            {
                "input_ids": [
                    paddle.to_tensor(tokens, dtype=paddle.int64).reshape(
                        [1, -1]
                    )
                ],
                "position_ids": [
                    paddle.to_tensor(tokens, dtype=paddle.int64).reshape(
                        [1, -1]
                    )
                ],
                "attention_mask": [
                    paddle.ones((1, 1, seq_len, seq_len), dtype=bool)
                ],
            },
            [
                paddle.to_tensor(
                    list(range(1, seq_len + 1)), dtype=paddle.int64
                ).reshape([1, -1])
            ],
        )
        loss = NoPipelineParallel(
            model, self.strategy
        ).forward_backward_pipeline(data)

        self.assertTrue(paddle.isfinite(loss).item())
        # All params should have gradients
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} has no gradient")
            self.assertTrue(
                paddle.isfinite(param.grad).all().item(),
                f"{name} has non-finite gradient",
            )


# ---------------------------------------------------------------------------
# Test 7: blocks list is correctly mutated across layers
# ---------------------------------------------------------------------------
class TestBlocksPropagation(unittest.TestCase):
    """Verify blocks list grows across layers as expected."""

    def setUp(self):
        self.strategy = _init_fleet()

    def test_blocks_accumulate(self):
        """blocks list should grow at block boundaries."""
        config = GPTConfig(
            num_hidden_layers=6,
            hidden_size=64,
            rotary_base=10000,
            vocab_size=100,
            rotary_percent=1.0,
            rope_scaling=1.0,
            position_embedding_type="rope",
            num_attention_heads=4,
            intermediate_size=128,
            max_sequence_length=16,
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
            attn_res_block_size=4,  # span=2, boundaries at layer 0,2,4
        )
        model = gpt_builder(config, num_stages=1)

        blocks_seen = []
        original_forward = BlockAttnRes.forward

        def spy_forward(self_bar, partial_block, blocks):
            blocks_seen.append(len(blocks) if blocks else 0)
            return original_forward(self_bar, partial_block, blocks)

        BlockAttnRes.forward = spy_forward
        try:
            seq_len = config.max_sequence_length
            tokens = list(range(seq_len))
            data = (
                {
                    "input_ids": [
                        paddle.to_tensor(tokens, dtype=paddle.int64).reshape(
                            [1, -1]
                        )
                    ],
                    "position_ids": [
                        paddle.to_tensor(tokens, dtype=paddle.int64).reshape(
                            [1, -1]
                        )
                    ],
                    "attention_mask": [
                        paddle.ones((1, 1, seq_len, seq_len), dtype=bool)
                    ],
                },
                [
                    paddle.to_tensor(
                        list(range(1, seq_len + 1)), dtype=paddle.int64
                    ).reshape([1, -1])
                ],
            )
            loss = NoPipelineParallel(
                model, self.strategy
            ).forward_backward_pipeline(data)
        finally:
            BlockAttnRes.forward = original_forward

        # blocks should have grown: not all zeros
        self.assertTrue(
            max(blocks_seen) > 0,
            f"blocks never grew: {blocks_seen}",
        )


# ---------------------------------------------------------------------------
# Test 8: ValueError for block_attn_res + full_recompute + act_offload
# ---------------------------------------------------------------------------
class TestOffloadIncompatibility(unittest.TestCase):
    """block_attn_res + full_recompute + activation offload must raise."""

    def setUp(self):
        self.strategy = _init_fleet()

    def test_raises_value_error(self):
        """Should raise ValueError when all three are enabled."""
        with self.assertRaises(ValueError) as ctx:
            config = GPTConfig(
                num_hidden_layers=4,
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
                tie_word_embeddings=True,
                use_qk_norm=True,
                block_attention_residuals=True,
                attn_res_block_size=4,
                recompute_granularity="full",
                recompute_method="uniform",
                recompute_num_layers=1,
                decoderlayer_act_offload_settings={
                    "type": "full",
                    "value": "all",
                },
            )
            gpt_builder(config, num_stages=1)
        self.assertIn("decoderlayer_act_offload", str(ctx.exception))

    def test_no_error_without_offload(self):
        """No error when offload is disabled."""
        config = GPTConfig(
            num_hidden_layers=4,
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
            tie_word_embeddings=True,
            use_qk_norm=True,
            block_attention_residuals=True,
            attn_res_block_size=4,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
        )
        # Should not raise
        model = gpt_builder(config, num_stages=1)
        self.assertIsNotNone(model)


# ---------------------------------------------------------------------------
# Test 9: _use_pylayer gating
# ---------------------------------------------------------------------------
class TestUsePylayerGating(unittest.TestCase):
    """BlockAttnRes._use_pylayer is True only for RMSNorm."""

    def setUp(self):
        self.strategy = _init_fleet()

    def test_rmsnorm_uses_pylayer(self):
        """With RMSNorm, _use_pylayer should be True."""
        config = GPTConfig(
            num_hidden_layers=2,
            hidden_size=64,
            rotary_base=10000,
            vocab_size=100,
            rotary_percent=1.0,
            rope_scaling=1.0,
            position_embedding_type="rope",
            num_attention_heads=4,
            intermediate_size=128,
            max_sequence_length=16,
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
        model = gpt_builder(config, num_stages=1)

        found = False
        for m in model.sublayers():
            if isinstance(m, BlockAttnRes):
                self.assertTrue(m._use_pylayer)
                found = True
                break
        self.assertTrue(found, "No BlockAttnRes found in model")

    def test_layernorm_does_not_use_pylayer(self):
        """With LayerNorm (non-RMSNorm), _use_pylayer should be False and
        forward/backward still work correctly via standard autograd path."""
        config = GPTConfig(
            num_hidden_layers=2,
            hidden_size=64,
            rotary_base=10000,
            vocab_size=100,
            rotary_percent=1.0,
            rope_scaling=1.0,
            position_embedding_type="rope",
            num_attention_heads=4,
            intermediate_size=128,
            max_sequence_length=16,
            normalization="LayerNorm",
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
        model = gpt_builder(config, num_stages=1)

        bar = None
        for m in model.sublayers():
            if isinstance(m, BlockAttnRes):
                bar = m
                break
        self.assertIsNotNone(bar, "No BlockAttnRes found in model")
        # LayerNorm is NOT RMSNorm, so _use_pylayer must be False
        self.assertFalse(
            bar._use_pylayer,
            f"Expected _use_pylayer=False for LayerNorm, got True. "
            f"norm type: {type(bar.norm)}",
        )

        # Verify forward/backward still work via standard autograd path
        B, S, H = 2, 16, 64
        partial_block = paddle.randn([B, S, H])
        partial_block.stop_gradient = False
        blocks = [paddle.randn([B, S, H])]
        blocks[0].stop_gradient = False

        bar.train()
        result = bar(partial_block, blocks)
        self.assertEqual(result.shape, [B, S, H])
        self.assertTrue(paddle.isfinite(result).all().item())

        # Backward should produce valid gradients
        loss = result.sum()
        loss.backward()
        self.assertIsNotNone(partial_block.grad)
        self.assertTrue(paddle.isfinite(partial_block.grad).all().item())
        self.assertIsNotNone(blocks[0].grad)
        self.assertTrue(paddle.isfinite(blocks[0].grad).all().item())


# ---------------------------------------------------------------------------
# Test: OutputBlockAttnResPipe layout uniqueness, forward/backward, state_dict
# ---------------------------------------------------------------------------
class TestOutputBlockAttnResPipe(unittest.TestCase):
    """Verify OutputBlockAttnResPipe is inserted exactly once in all layouts."""

    def setUp(self):
        self.strategy = _init_fleet()

    def _make_config(self, experimental=False, num_mtp=0):
        return GPTConfig(
            num_hidden_layers=4,
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
            tie_word_embeddings=True,
            use_qk_norm=True,
            block_attention_residuals=True,
            attn_res_block_size=2,
            gpt_model_use_experimental_version=experimental,
            num_nextn_predict_layers=num_mtp,
        )

    def _count_output_attn_res_layers(self, model):
        """Count OutputBlockAttnResPipe instances in model sublayers."""
        from paddlefleet.transformer.block_attn_res import (
            OutputBlockAttnResPipe,
        )

        return sum(
            1
            for m in model.sublayers()
            if isinstance(m, OutputBlockAttnResPipe)
        )

    def test_standard_layout_has_exactly_one(self):
        """Standard layout (no experimental, no MTP) has one OutputBlockAttnResPipe."""
        config = self._make_config(experimental=False, num_mtp=0)
        model = gpt_builder(config, num_stages=1)
        self.assertEqual(self._count_output_attn_res_layers(model), 1)

    def test_experimental_mtp_layout_has_exactly_one(self):
        """Experimental MTP layout has one OutputBlockAttnResPipe (not duplicated)."""
        config = self._make_config(experimental=True, num_mtp=1)
        model = gpt_builder(config, num_stages=1)
        self.assertEqual(self._count_output_attn_res_layers(model), 1)

    def test_experimental_mtp_forward_backward(self):
        """Experimental MTP: OutputBlockAttnResPipe handles concat/split correctly.

        Directly exercise the pipe with MTP-shaped inputs (avoids the full
        model's data pipeline dependencies).
        """
        from paddlefleet.transformer.block_attn_res import (
            BlockAttnResSublayersSpec,
            OutputBlockAttnResPipe,
        )
        from paddlefleet.transformer.paddle_norm import RMSNorm

        config = self._make_config(experimental=True, num_mtp=1)
        pipe = OutputBlockAttnResPipe(
            config, BlockAttnResSublayersSpec(norm=RMSNorm)
        )
        pipe.train()

        B, S, H = 2, 8, config.hidden_size
        num_mtp = config.num_nextn_predict_layers  # 1
        # hidden_states = concat of [main, mtp_0, ...] along axis=0
        hidden = paddle.randn([(num_mtp + 1) * S, B, H])
        hidden.stop_gradient = False
        blocks = [paddle.randn([S, B, H]) for _ in range(2)]
        for b in blocks:
            b.stop_gradient = False

        out = pipe({"hidden_states": hidden, "blocks": blocks})
        self.assertIn("hidden_states", out)
        # Output shape preserved
        self.assertEqual(out["hidden_states"].shape, [(num_mtp + 1) * S, B, H])
        # blocks consumed
        self.assertNotIn("blocks", out)

        # Backward path
        loss = out["hidden_states"].sum()
        loss.backward()
        self.assertIsNotNone(pipe.block_attn_res.proj_weight.grad)
        self.assertTrue(
            paddle.isfinite(pipe.block_attn_res.proj_weight.grad).all().item()
        )
        self.assertIsNotNone(blocks[0].grad)
        self.assertIsNotNone(hidden.grad)

    def test_state_dict_round_trip(self):
        """state_dict has exactly one set of output_attn_res keys; round-trip preserves values."""
        config = self._make_config(experimental=True, num_mtp=1)
        model = gpt_builder(config, num_stages=1)

        sd = model.state_dict()
        attn_res_keys = [k for k in sd.keys() if "output_attn_res" in k]
        # Should have proj_weight and norm.weight (exactly 2 keys)
        self.assertEqual(len(attn_res_keys), 2)

        # No duplicate keys (dict guarantees uniqueness, but verify naming)
        proj_keys = [k for k in attn_res_keys if "proj_weight" in k]
        norm_keys = [k for k in attn_res_keys if "norm" in k]
        self.assertEqual(len(proj_keys), 1)
        self.assertEqual(len(norm_keys), 1)

        # Mutate and reload. Save expected values first because
        # set_state_dict may consume/clear the input dict.
        expected = {}
        for k in attn_res_keys:
            v = paddle.ones_like(sd[k]) * 3.14
            sd[k] = v
            expected[k] = v.numpy().copy()
        model.set_state_dict(sd)

        sd2 = model.state_dict()
        for k in attn_res_keys:
            np.testing.assert_allclose(sd2[k].numpy(), expected[k], rtol=1e-6)

    def test_forward_passthrough_when_disabled(self):
        """block_attention_residuals=False: pipe returns dict_args untouched."""
        from paddlefleet.transformer.block_attn_res import (
            BlockAttnResSublayersSpec,
            OutputBlockAttnResPipe,
        )
        from paddlefleet.transformer.paddle_norm import RMSNorm

        config = self._make_config(experimental=False, num_mtp=0)
        config.block_attention_residuals = False
        pipe = OutputBlockAttnResPipe(
            config, BlockAttnResSublayersSpec(norm=RMSNorm)
        )
        hidden = paddle.randn([8, 2, config.hidden_size])
        dict_args = {"hidden_states": hidden, "blocks": ["b"]}
        out = pipe(dict_args)
        # Passthrough: same object, blocks retained, no aggregation.
        self.assertIs(out, dict_args)
        self.assertIs(out["hidden_states"], hidden)
        self.assertIn("blocks", out)

    def test_build_schedule_node(self):
        """build_schedule_node wraps forward with the expected name."""
        import sys
        import types
        from unittest.mock import patch

        from paddlefleet.transformer.block_attn_res import (
            BlockAttnResSublayersSpec,
            OutputBlockAttnResPipe,
        )
        from paddlefleet.transformer.paddle_norm import RMSNorm

        config = self._make_config(experimental=False, num_mtp=0)
        pipe = OutputBlockAttnResPipe(
            config, BlockAttnResSublayersSpec(norm=RMSNorm)
        )

        # ScheduleNode is imported lazily inside build_schedule_node from the
        # paddle pipeline module; stub it so the test does not depend on the
        # installed paddle version exposing that symbol.
        pp_mod = types.ModuleType(
            "paddle.distributed.fleet.meta_parallel.pipeline_parallel"
        )

        class _FakeSN:
            def __init__(self, fn, name):
                self.fn = fn
                self.name = name

        pp_mod.ScheduleNode = _FakeSN
        with patch.dict(
            sys.modules,
            {
                "paddle.distributed.fleet.meta_parallel.pipeline_parallel": (
                    pp_mod
                )
            },
        ):
            node = pipe.build_schedule_node()
        self.assertEqual(node.name, "OutputBlockAttnResPipe")
        self.assertEqual(node.fn.__func__, OutputBlockAttnResPipe.forward)


class TestGPTMainLMHeadInit(unittest.TestCase):
    """GPTMainLMHead.__init__ must drop the block_attn_res kwarg.

    Output BlockAttnRes moved to OutputBlockAttnResPipe, so the head must not
    forward that kwarg to ColumnParallelLinear (covers lm_head.py pop line).
    """

    def test_init_pops_block_attn_res_kwarg(self):
        from unittest.mock import patch

        from paddlefleet.models.gpt.lm_head import GPTLMHead, GPTMainLMHead

        sentinel = object()
        captured = {}

        def fake_gptlmhead_init(self, **kwargs):
            captured.update(kwargs)

        with patch.object(GPTLMHead, "__init__", fake_gptlmhead_init):
            GPTMainLMHead(block_attn_res=sentinel, config=MagicMock())
        self.assertNotIn("block_attn_res", captured)


if __name__ == "__main__":
    unittest.main()
