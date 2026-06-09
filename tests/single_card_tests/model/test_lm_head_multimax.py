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

"""Tests for GPTLMHead with multimax feature."""

import functools
import random
import unittest

import numpy as np
import paddle
from paddle.distributed import fleet

import paddlefleet.parallel_state as ps
from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig


class TestMultimaxLMHead(unittest.TestCase):
    """Tests for GPTLMHead multimax initialization and forward paths."""

    @classmethod
    def setUpClass(cls):
        seed = 48
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)
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
        hcg = fleet.get_hybrid_communicate_group()
        ps.initialize_model_parallel(hcg)
        cls.strategy = strategy

        # Config with multimax=lm_head
        cls.config_multimax = GPTConfig(
            num_hidden_layers=2,
            hidden_size=256,
            vocab_size=64,
            max_sequence_length=32,
            num_attention_heads=4,
            intermediate_size=512,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            use_bias=False,
            rotary_percent=1.0,
            rotary_base=10000,
            rope_scaling=1.0,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            tie_word_embeddings=True,
            multimax="lm_head",
        )
        cls.model_multimax = gpt_builder(cls.config_multimax, num_stages=1)

        # Config without multimax
        cls.config_no_multimax = GPTConfig(
            num_hidden_layers=2,
            hidden_size=256,
            vocab_size=64,
            max_sequence_length=32,
            num_attention_heads=4,
            intermediate_size=512,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            use_bias=False,
            rotary_percent=1.0,
            rotary_base=10000,
            rope_scaling=1.0,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            tie_word_embeddings=True,
            multimax=None,
        )
        cls.model_no_multimax = gpt_builder(cls.config_no_multimax, num_stages=1)

        # Config with multimax + fused path
        cls.config_fused = GPTConfig(
            num_hidden_layers=2,
            hidden_size=256,
            vocab_size=64,
            max_sequence_length=32,
            num_attention_heads=4,
            intermediate_size=512,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            use_bias=False,
            rotary_percent=1.0,
            rotary_base=10000,
            rope_scaling=1.0,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            tie_word_embeddings=True,
            multimax="lm_head",
            fused_linear_ce_loss_chunk=1,
        )
        cls.model_fused = gpt_builder(cls.config_fused, num_stages=1)

    def _find_lm_head(self, model):
        from paddlefleet.models.gpt.lm_head import GPTLMHead
        for layer in model.run_function:
            if isinstance(layer, GPTLMHead):
                return layer
        return None

    def test_multimax_lmhead_creates_params(self):
        """When multimax=lm_head, GPTLMHead should create multimax_ranges/ts params."""
        lm_head = self._find_lm_head(self.model_multimax)
        self.assertIsNotNone(lm_head)
        self.assertTrue(hasattr(lm_head, "use_multimax_lmhead"))
        self.assertTrue(lm_head.use_multimax_lmhead)
        self.assertTrue(hasattr(lm_head, "multimax_ranges"))
        self.assertTrue(hasattr(lm_head, "multimax_ts"))
        self.assertEqual(lm_head.multimax_ranges.shape, [4])
        self.assertEqual(lm_head.multimax_ts.shape, [4])

    def test_no_multimax_lmhead_no_params(self):
        """When multimax=None, GPTLMHead should NOT create multimax params."""
        lm_head = self._find_lm_head(self.model_no_multimax)
        self.assertIsNotNone(lm_head)
        self.assertFalse(getattr(lm_head, "use_multimax_lmhead", False))
        self.assertFalse(hasattr(lm_head, "multimax_ranges"))
        self.assertFalse(hasattr(lm_head, "multimax_ts"))

    def test_multimax_params_init_zero(self):
        """multimax params should init to zero (SeLU is identity at step 0)."""
        lm_head = self._find_lm_head(self.model_multimax)
        zeros = paddle.zeros_like(lm_head.multimax_ranges)
        self.assertTrue(paddle.allclose(lm_head.multimax_ranges, zeros).item())
        self.assertTrue(paddle.allclose(lm_head.multimax_ts, zeros).item())

    def test_fused_path_returns_5tuple(self):
        """With fused_linear_ce_loss_chunk>0 and multimax, forward returns 5-tuple."""
        lm_head = self._find_lm_head(self.model_fused)
        self.assertIsNotNone(lm_head)

        # Create dummy input
        batch_size, seq_len, hidden_size = 2, 8, 256
        hidden_states = paddle.randn([seq_len, batch_size, hidden_size])

        # Call forward with dict args (expected signature)
        output = lm_head.forward({"hidden_states": hidden_states})

        # Should return 5-tuple: (hidden, weight, bias, multimax_ranges, multimax_ts)
        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 5)


if __name__ == "__main__":
    unittest.main()
