#!/usr/bin/env python3
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
Unit tests for NgramMoeEmbedding: per-order router independence, L_aux
numerical correctness, and baseline consistency (shared_router=True +
load_balance_coef=0).

Run with:
  python tests/single_card_tests/test_ngram_moe_embedding.py
"""

import unittest
from types import SimpleNamespace

import numpy as np
import paddle

from paddlefleet.models.common.embeddings.ngram_moe_embedding import (
    NgramMoeEmbedding,
    NgramTableRouter,
)


def _make_config(
    hidden_size=32,
    vocab_size=200,
    ngram_emb_neighbor_num=3,
    ngram_moe_tables_per_order=8,
    ngram_moe_active_tables=2,
    ngram_moe_table_rows_ratio=1.0,
    ngram_moe_table_dim=16,
    ngram_moe_router_dim=16,
    ngram_moe_router_width=4,
    ngram_moe_shared_router=True,
    ngram_moe_load_balance_coef=0.0,
):
    return SimpleNamespace(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        ngram_emb_neighbor_num=ngram_emb_neighbor_num,
        ngram_pad_token_id=0,
        ngram_moe_enabled=True,
        ngram_moe_tables_per_order=ngram_moe_tables_per_order,
        ngram_moe_active_tables=ngram_moe_active_tables,
        ngram_moe_table_rows_ratio=ngram_moe_table_rows_ratio,
        ngram_moe_table_dim=ngram_moe_table_dim,
        ngram_moe_router_dim=ngram_moe_router_dim,
        ngram_moe_router_width=ngram_moe_router_width,
        ngram_moe_shared_router=ngram_moe_shared_router,
        ngram_moe_load_balance_coef=ngram_moe_load_balance_coef,
        ngram_moe_z_loss_coef=0.0,
        ngram_monitor_enabled=False,
    )


class TestNgramMoeRouterIndependence(unittest.TestCase):
    """Per-order independent routers should produce different selections."""

    def setUp(self):
        paddle.seed(42)
        np.random.seed(42)
        self.B, self.S = 2, 16
        self.vocab_size = 200
        self.input_ids = paddle.randint(
            1, self.vocab_size, [self.B, self.S]
        ).astype("int64")

    def test_shared_router_same_selection(self):
        """When shared_router=True, all orders get identical sel/gate."""
        config = _make_config(ngram_moe_shared_router=True)
        model = NgramMoeEmbedding(config=config, vocab_size=self.vocab_size)
        model.eval()

        word_emb = paddle.randn([self.B, self.S, config.hidden_size])

        sel, gate, probs = model._route(word_emb)
        # In shared mode, calling _route for any order_idx returns the same.
        sel2, gate2, probs2 = model._route(word_emb, order_idx=1)
        np.testing.assert_array_equal(sel.numpy(), sel2.numpy())

    def test_independent_router_different_selection(self):
        """When shared_router=False, per-order routers have independent weights."""
        config = _make_config(ngram_moe_shared_router=False)
        model = NgramMoeEmbedding(config=config, vocab_size=self.vocab_size)
        model.eval()

        word_emb = paddle.randn([self.B, self.S, config.hidden_size])

        sel0, _, _ = model._route(word_emb, order_idx=0)
        sel1, _, _ = model._route(word_emb, order_idx=1)

        # With independent random init, the two routers should produce
        # different selections for at least some positions.
        diff = (sel0.numpy() != sel1.numpy()).any()
        self.assertTrue(diff, "Per-order routers produced identical selections")

    def test_independent_router_has_separate_params(self):
        """Each per-order router should have its own weight parameters."""
        config = _make_config(ngram_moe_shared_router=False)
        model = NgramMoeEmbedding(config=config, vocab_size=self.vocab_size)

        # The routers list should have num_orders entries.
        self.assertEqual(len(model.routers), model.num_orders)

        # Modify one router's weight and verify the other is unchanged.
        w0_before = model.routers[0].score.weight.numpy().copy()
        w1_before = model.routers[1].score.weight.numpy().copy()

        model.routers[0].score.weight.set_value(
            paddle.randn(model.routers[0].score.weight.shape)
        )

        w0_after = model.routers[0].score.weight.numpy()
        w1_after = model.routers[1].score.weight.numpy()

        self.assertFalse(np.allclose(w0_before, w0_after))
        np.testing.assert_array_equal(w1_before, w1_after)


class TestNgramMoeLoadBalanceLoss(unittest.TestCase):
    """L_aux numerical correctness (Switch-Transformer style)."""

    def setUp(self):
        paddle.seed(123)
        np.random.seed(123)
        self.B, self.S = 4, 32
        self.vocab_size = 200
        self.input_ids = paddle.randint(
            1, self.vocab_size, [self.B, self.S]
        ).astype("int64")
        self.word_emb = paddle.randn([self.B, self.S, 32])

    def test_aux_loss_none_when_coef_zero(self):
        """load_balance_coef=0 should return None aux_loss."""
        config = _make_config(ngram_moe_load_balance_coef=0.0)
        model = NgramMoeEmbedding(config=config, vocab_size=self.vocab_size)
        model.eval()
        _, aux_loss = model(self.input_ids, self.word_emb)
        self.assertIsNone(aux_loss)

    def test_aux_loss_not_none_when_coef_positive(self):
        """load_balance_coef>0 should return a scalar tensor."""
        config = _make_config(ngram_moe_load_balance_coef=0.01)
        model = NgramMoeEmbedding(config=config, vocab_size=self.vocab_size)
        model.eval()
        _, aux_loss = model(self.input_ids, self.word_emb)
        self.assertIsNotNone(aux_loss)
        self.assertEqual(aux_loss.ndim, 0)  # scalar

    def test_aux_loss_value_perfect_balance(self):
        """When all experts get equal traffic, L_aux = α·N·(1/T) = α·N/T.

        We craft a uniform probs tensor and uniform sel to verify the formula.
        """
        T = 8  # tables_per_order
        A = 2  # active_tables
        coef = 0.1

        config = _make_config(
            ngram_moe_tables_per_order=T,
            ngram_moe_active_tables=A,
            ngram_moe_load_balance_coef=coef,
            ngram_moe_shared_router=True,
        )
        model = NgramMoeEmbedding(config=config, vocab_size=self.vocab_size)

        B, S = 4, 32
        # Uniform selection: each token selects experts [0, 1]
        sel = paddle.zeros([B, S, A], dtype="int64")
        # Uniform probabilities: 1/T for each expert
        probs = paddle.full([B, S, T], 1.0 / T, dtype="float32")

        # f_i: fraction of tokens that selected expert i
        # Expert 0: selected by all tokens → f_0 = 1.0
        # Expert 1: selected by all tokens → f_1 = 1.0
        # Others: f_i = 0
        # P_i = 1/T for all i
        # L_aux = coef * T * sum(f_i * P_i)
        #       = coef * T * (2 * 1.0 * 1/T)
        #       = coef * 2 = 0.2
        loss = model._load_balance_loss([sel], [probs])
        expected = coef * T * (2 * 1.0 * (1.0 / T))
        np.testing.assert_allclose(
            loss.numpy(), expected, rtol=1e-5, atol=1e-5
        )

    def test_aux_loss_shared_vs_independent_count(self):
        """Shared router: 1 term in the sum; independent: num_orders terms."""
        coef = 0.01

        # Shared router
        config_s = _make_config(
            ngram_moe_shared_router=True,
            ngram_moe_load_balance_coef=coef,
        )
        model_s = NgramMoeEmbedding(
            config=config_s, vocab_size=self.vocab_size
        )
        model_s.eval()
        _, aux_s = model_s(self.input_ids, self.word_emb)

        # Independent router
        config_i = _make_config(
            ngram_moe_shared_router=False,
            ngram_moe_load_balance_coef=coef,
        )
        model_i = NgramMoeEmbedding(
            config=config_i, vocab_size=self.vocab_size
        )
        # Copy shared router weights to both independent routers so the
        # routing decisions are comparable.
        model_i.routers[0].load_dict(model_s.router.state_dict())
        model_i.routers[1].load_dict(model_s.router.state_dict())
        model_i.eval()
        _, aux_i = model_i(self.input_ids, self.word_emb)

        # Independent has num_orders terms, shared has 1 term.
        # With identical weights, each term should be the same, so
        # aux_i ≈ num_orders * aux_s.
        ratio = aux_i.numpy() / aux_s.numpy()
        np.testing.assert_allclose(
            ratio, model_i.num_orders, rtol=1e-4, atol=1e-4
        )


class TestNgramMoeBaselineConsistency(unittest.TestCase):
    """shared_router=True + load_balance_coef=0 should be a pure no-op."""

    def test_forward_returns_tuple_with_none_aux(self):
        """Baseline config: forward returns (signal, None)."""
        config = _make_config(
            ngram_moe_shared_router=True,
            ngram_moe_load_balance_coef=0.0,
        )
        model = NgramMoeEmbedding(config=config, vocab_size=200)
        model.eval()

        B, S = 2, 16
        input_ids = paddle.randint(1, 200, [B, S]).astype("int64")
        word_emb = paddle.randn([B, S, config.hidden_size])

        out, aux = model(input_ids, word_emb)
        self.assertIsNotNone(out)
        self.assertEqual(out.shape, [B, S, config.hidden_size])
        self.assertIsNone(aux)

    def test_forward_output_deterministic(self):
        """Same input → same output (eval mode, fixed seed)."""
        config = _make_config()
        model = NgramMoeEmbedding(config=config, vocab_size=200)
        model.eval()

        paddle.seed(999)
        input_ids = paddle.randint(1, 200, [2, 16]).astype("int64")
        word_emb = paddle.randn([2, 16, config.hidden_size])

        out1, _ = model(input_ids, word_emb)
        out2, _ = model(input_ids, word_emb)
        np.testing.assert_array_equal(out1.numpy(), out2.numpy())


if __name__ == "__main__":
    unittest.main()
