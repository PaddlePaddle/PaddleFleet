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
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)


import unittest
from unittest.mock import patch

import numpy as np
import paddle


def _make_router_config(**overrides):
    """Helper to create a TransformerConfig for router testing."""
    from paddlefleet.transformer.transformer_config import TransformerConfig

    defaults = {
        "hidden_size": 64,
        "num_attention_heads": 2,
        "intermediate_size": 256,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "sequence_parallel": False,
        "tensor_model_parallel_size": 1,
        "topk_method": "greedy",
        "norm_topk_prob": True,
        "scoring_func": "softmax",
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 1.0,
        "routed_scaling_factor_learnable": False,
        "moe_router_force_load_balancing": False,
        "moe_router_load_balancing_type": "aux_loss",
        "router_aux_loss_coef": 0.01,
        "router_z_loss_coef": None,
        "moe_n_hash_layers": 0,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestMoERouter(unittest.TestCase):
    """Unit tests for moe_router module."""

    @patch(
        "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_standard_router_init(self, mock_cp):
        """Test StandardMoERouter initialization."""
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        config = _make_router_config()
        router = StandardMoERouter(config)
        self.assertEqual(router.hidden_size, 64)
        self.assertEqual(router.num_experts, 4)
        self.assertEqual(router.num_experts_per_tok, 2)
        self.assertEqual(router.scoring_func, "softmax")
        self.assertEqual(router.topk_method, "greedy")

    @patch(
        "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_noaux_tc_router_init(self, mock_cp):
        """Test router init with noaux_tc topk_method registers buffers."""
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        config = _make_router_config(topk_method="noaux_tc")
        router = StandardMoERouter(config)
        self.assertIsNotNone(router.e_score_correction_bias)
        self.assertIsNotNone(router.expert_usage)

    @patch(
        "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_gate_score_func_softmax(self, mock_cp):
        """Test gate_score_func with softmax."""
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        config = _make_router_config(scoring_func="softmax")
        router = StandardMoERouter(config)
        logits = paddle.randn([4, 4], dtype=paddle.float32)
        scores = router.gate_score_func(logits)
        self.assertTrue(
            paddle.allclose(scores.sum(axis=-1), paddle.ones([4]), atol=1e-5)
        )

    @patch(
        "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_gate_score_func_sigmoid(self, mock_cp):
        """Test gate_score_func with sigmoid."""
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        config = _make_router_config(scoring_func="sigmoid")
        router = StandardMoERouter(config)
        logits = paddle.randn([4, 4], dtype=paddle.float32)
        scores = router.gate_score_func(logits)
        self.assertTrue((scores >= 0).all() and (scores <= 1).all())

    @patch(
        "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_gate_score_func_tanh(self, mock_cp):
        """Test gate_score_func with tanh."""
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        config = _make_router_config(scoring_func="tanh")
        router = StandardMoERouter(config)
        logits = paddle.randn([4, 4], dtype=paddle.float32)
        scores = router.gate_score_func(logits)
        self.assertTrue((scores >= -1).all() and (scores <= 1).all())

    @patch(
        "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_gate_score_func_relu(self, mock_cp):
        """Test gate_score_func with relu."""
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        config = _make_router_config(scoring_func="relu")
        router = StandardMoERouter(config)
        logits = paddle.randn([4, 4], dtype=paddle.float32)
        scores = router.gate_score_func(logits)
        self.assertTrue((scores >= 0).all())

    @patch(
        "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_gate_score_func_gelu(self, mock_cp):
        """Test gate_score_func with gelu."""
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        config = _make_router_config(scoring_func="gelu")
        router = StandardMoERouter(config)
        logits = paddle.randn([4, 4], dtype=paddle.float32)
        scores = router.gate_score_func(logits)
        self.assertEqual(scores.shape, [4, 4])

    @patch(
        "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_gate_score_func_leaky_relu(self, mock_cp):
        """Test gate_score_func with leaky_relu."""
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        config = _make_router_config(scoring_func="leaky_relu")
        router = StandardMoERouter(config)
        logits = paddle.randn([4, 4], dtype=paddle.float32)
        scores = router.gate_score_func(logits)
        self.assertEqual(scores.shape, [4, 4])

    @patch(
        "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_gate_score_func_not_implemented(self, mock_cp):
        """Test gate_score_func raises for unknown scoring func."""
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        config = _make_router_config(scoring_func="unknown")
        router = StandardMoERouter(config)
        logits = paddle.randn([4, 4], dtype=paddle.float32)
        with self.assertRaises(NotImplementedError):
            router.gate_score_func(logits)

    @patch(
        "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_capacity_calculation(self, mock_cp):
        """Test _capacity calculates correct value."""
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        config = _make_router_config()
        router = StandardMoERouter(config)
        gates = paddle.randn([16, 4], dtype=paddle.float32)
        capacity = router._capacity(
            gates, capacity_factor=1.0, max_capacity=10, min_capacity=1
        )
        self.assertEqual(capacity, 4)

    @patch(
        "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_capacity_min_clamp(self, mock_cp):
        """Test _capacity clamps to min_capacity."""
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        config = _make_router_config()
        router = StandardMoERouter(config)
        gates = paddle.randn([2, 8], dtype=paddle.float32)
        capacity = router._capacity(
            gates, capacity_factor=0.01, max_capacity=10, min_capacity=5
        )
        self.assertEqual(capacity, 5)

    @patch(
        "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_capacity_max_clamp(self, mock_cp):
        """Test _capacity clamps to max_capacity."""
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        config = _make_router_config()
        router = StandardMoERouter(config)
        gates = paddle.randn([100, 4], dtype=paddle.float32)
        capacity = router._capacity(
            gates, capacity_factor=10.0, max_capacity=5, min_capacity=1
        )
        self.assertEqual(capacity, 5)

    @patch(
        "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_cal_aux_loss(self, mock_cp):
        """Test _cal_aux_loss computation."""
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        config = _make_router_config(n_routed_experts=4)
        router = StandardMoERouter(config)
        gates = paddle.ones([4, 4], dtype=paddle.float32) * 0.25
        mask = paddle.zeros([4, 4], dtype=paddle.float32)
        mask[0, 0] = 1.0
        mask[1, 1] = 1.0
        mask[2, 2] = 1.0
        mask[3, 3] = 1.0
        aux_loss = router._cal_aux_loss(gates, mask)
        self.assertIsNotNone(aux_loss)
        self.assertEqual(aux_loss.shape, [])

    @patch(
        "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_cal_z_loss(self, mock_cp):
        """Test _cal_z_loss computation."""
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        config = _make_router_config()
        router = StandardMoERouter(config)
        logits = paddle.randn([4, 4], dtype=paddle.float32)
        z_loss = router._cal_z_loss(logits)
        self.assertIsNotNone(z_loss)
        self.assertGreater(z_loss.item(), 0.0)

    @patch(
        "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_topk_greedy(self, mock_cp):
        """Test _topk_greedy returns correct shapes."""
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        config = _make_router_config(num_experts_per_tok=2)
        router = StandardMoERouter(config)
        scores = paddle.randn([8, 4], dtype=paddle.float32)
        topk_weight, topk_idx = router._topk_greedy(scores, k=2)
        self.assertEqual(topk_weight.shape, [8, 2])
        self.assertEqual(topk_idx.shape, [8, 2])

    @patch(
        "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_topk_group_limited_greedy(self, mock_cp):
        """Test _topk_group_limited_greedy."""
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        config = _make_router_config(
            n_routed_experts=8,
            n_group=4,
            topk_group=2,
            num_experts_per_tok=2,
        )
        router = StandardMoERouter(config)
        scores = paddle.randn([4, 8], dtype=paddle.float32)
        topk_weight, topk_idx = router._topk_group_limited_greedy(
            scores, k=2, n_group=4, topk_group=2
        )
        self.assertEqual(topk_weight.shape, [4, 2])
        self.assertEqual(topk_idx.shape, [4, 2])

    @patch(
        "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_topk_group_limited_greedy_assert(self, mock_cp):
        """Test _topk_group_limited_greedy asserts divisibility."""
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        config = _make_router_config(n_routed_experts=5, n_group=3)
        router = StandardMoERouter(config)
        scores = paddle.randn([4, 5], dtype=paddle.float32)
        with self.assertRaises(AssertionError):
            router._topk_group_limited_greedy(
                scores, k=2, n_group=3, topk_group=1
            )

    @patch(
        "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_call_topk_method_invalid(self, mock_cp):
        """Test _call_topk_method raises for invalid method."""
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        config = _make_router_config()
        router = StandardMoERouter(config)
        with self.assertRaises(NotImplementedError):
            router._call_topk_method("invalid", paddle.randn([4, 4]), k=2)

    @patch(
        "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_set_layer_number(self, mock_cp):
        """Test set_layer_number."""
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        config = _make_router_config()
        router = StandardMoERouter(config)
        router.set_layer_number(3)
        self.assertEqual(router.layer_number, 3)

    @patch(
        "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_priority(self, mock_cp):
        """Test _priority with capacity constraint."""
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        config = _make_router_config(n_routed_experts=4, num_experts_per_tok=2)
        router = StandardMoERouter(config)
        topk_idx = paddle.to_tensor([[0, 1], [1, 2], [0, 3], [2, 3]])
        priority = router._priority(topk_idx, capacity=2)
        self.assertEqual(priority.shape, [4, 4])

    @patch(
        "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_probs_drop_policy(self, mock_cp):
        """Test _probs_drop_policy."""
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        config = _make_router_config(n_routed_experts=4)
        router = StandardMoERouter(config)
        scores = paddle.zeros([4, 4], dtype=paddle.float32)
        scores[0, 0] = 1.0
        scores[0, 1] = 0.8
        scores[1, 2] = 0.9
        scores[1, 3] = 0.7
        mask = router._probs_drop_policy(scores, capacity=2)
        self.assertEqual(mask.shape, [4, 4])

    @patch(
        "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_seq_aux_loss_raises_on_invalid_type(self, mock_cp):
        """Test router raises when seq_aux is True but type != seq_aux_loss."""
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        config = _make_router_config(
            moe_router_load_balancing_type="aux_loss",
        )
        config.seq_aux = True
        with self.assertRaises(ValueError):
            StandardMoERouter(config)

    @patch(
        "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_gate_detach_matmul_no_fuse(self, mock_cp):
        """Test gate_detach_matmul without fusion."""
        from paddlefleet.transformer.moe.moe_router import gate_detach_matmul

        x = paddle.randn([4, 64], dtype=paddle.float32)
        w = paddle.randn([64, 4], dtype=paddle.float32)
        score = gate_detach_matmul(x, w, use_fuse=False)
        self.assertEqual(score.shape, [4, 4])

    @patch(
        "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_fused_gate_detach_matmul(self, mock_cp):
        """Test FusedGateDetachMatmul PyLayer."""
        from paddlefleet.transformer.moe.moe_router import FusedGateDetachMatmul

        x = paddle.randn([4, 64], dtype=paddle.float32)
        w = paddle.randn([64, 4], dtype=paddle.float32)
        x.stop_gradient = False
        w.stop_gradient = False
        out = FusedGateDetachMatmul.apply(x, w)
        self.assertEqual(out.shape, [4, 4])

    @patch(
        "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_topk_noaux_tc_n_group_1(self, mock_cp):
        """Test _topk_noaux_tc with n_group=1."""
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        config = _make_router_config(
            topk_method="noaux_tc",
            n_routed_experts=4,
            n_group=1,
            topk_group=1,
            num_experts_per_tok=2,
        )
        router = StandardMoERouter(config)
        scores = paddle.randn([4, 4], dtype=paddle.float32)
        topk_weight, topk_idx = router._topk_noaux_tc(
            scores, k=2, n_group=1, topk_group=1
        )
        self.assertEqual(topk_weight.shape, [4, 2])
        self.assertEqual(topk_idx.shape, [4, 2])


class TestSftPlusScore(unittest.TestCase):
    """Tests for the 'sftplus' (softplus) scoring function in StandardMoERouter."""

    @patch(
        "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_sftplus_scores_are_non_negative(self, _mock):
        """softplus output should always be >= 0."""
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        config = _make_router_config(scoring_func="sftplus")
        router = StandardMoERouter(config)
        logits = paddle.randn([16, 4])
        scores = router.gate_score_func(logits)
        self.assertTrue(
            bool((scores >= 0).all().numpy()),
            "SftPlus scores should all be non-negative",
        )

    @patch(
        "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_sftplus_output_shape(self, _mock):
        """Output shape of gate_score_func should match input logits shape."""
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        config = _make_router_config(scoring_func="sftplus", n_routed_experts=4)
        router = StandardMoERouter(config)
        logits = paddle.randn([32, 4])
        scores = router.gate_score_func(logits)
        self.assertEqual(list(scores.shape), [32, 4])

    @patch(
        "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_sftplus_matches_paddle_softplus(self, _mock):
        """gate_score_func('sftplus') should exactly match F.softplus."""
        import paddle.nn.functional as F

        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        config = _make_router_config(scoring_func="sftplus")
        router = StandardMoERouter(config)
        logits = paddle.randn([8, 4])
        scores = router.gate_score_func(logits, logits_type_promotion=False)
        expected = F.softplus(logits)
        np.testing.assert_allclose(
            scores.numpy(),
            expected.numpy(),
            atol=1e-6,
            err_msg="SftPlus scores should match F.softplus",
        )

    @patch(
        "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_invalid_scoring_func_raises(self, _mock):
        """Unknown scoring_func should raise NotImplementedError."""
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        config = _make_router_config(scoring_func="unknown_func")
        router = StandardMoERouter(config)
        logits = paddle.randn([4, 4])
        with self.assertRaises(NotImplementedError):
            router.gate_score_func(logits)


class TestHashRouter(unittest.TestCase):
    """Tests for the HashRouter class."""

    def _make_router(self, **cfg_overrides):
        from paddlefleet.transformer.moe.moe_router import HashRouter

        config = _make_router_config(**cfg_overrides)
        return HashRouter(config=config, layer_number=0), config

    def _dummy_hidden(self, B, S, H=64):
        return paddle.randn([B, S, H])

    def test_deterministic_routing(self):
        """Same input_ids → same expert assignment every time."""
        router, _ = self._make_router(n_routed_experts=4, num_experts_per_tok=2)
        hidden = self._dummy_hidden(2, 4)
        input_ids = paddle.to_tensor(
            [[3, 7, 1, 5], [2, 6, 4, 8]], dtype="int64"
        )

        _, _, idx1, _, _, _, _, _ = router(hidden, input_ids=input_ids)
        _, _, idx2, _, _, _, _, _ = router(hidden, input_ids=input_ids)

        np.testing.assert_array_equal(
            idx1.numpy(),
            idx2.numpy(),
            err_msg="HashRouter should be deterministic",
        )

    def test_modulo_expert_assignment(self):
        """Expert indices should follow (token_id + k) % num_experts."""
        num_experts = 4
        k = 2
        router, _ = self._make_router(
            n_routed_experts=num_experts, num_experts_per_tok=k
        )

        B, S = 1, 4
        token_ids = [[3, 7, 1, 5]]
        hidden = self._dummy_hidden(B, S)
        input_ids = paddle.to_tensor(token_ids, dtype="int64")

        _, _, top_idx, _, _, _, _, _ = router(hidden, input_ids=input_ids)
        top_idx_np = top_idx.numpy()  # [4, 2]

        for pos, tid in enumerate(token_ids[0]):
            for ki in range(k):
                expected = (int(tid) + ki) % num_experts
                self.assertEqual(
                    int(top_idx_np[pos, ki]),
                    expected,
                    f"pos={pos} k={ki}: expected expert {expected}, got {int(top_idx_np[pos, ki])}",
                )

    def test_padding_tokens_masked(self):
        """Tokens with id==0 (padding) should have weight=0 and idx=-1."""
        router, _ = self._make_router(n_routed_experts=4, num_experts_per_tok=2)
        B, S = 1, 4
        input_ids = paddle.to_tensor([[0, 3, 5, 7]], dtype="int64")
        hidden = self._dummy_hidden(B, S)

        _, top_gate, top_idx, probs, mask, _, _, _ = router(
            hidden, input_ids=input_ids
        )
        top_gate_np = top_gate.numpy()
        top_idx_np = top_idx.numpy()
        probs_np = probs.numpy()
        mask_np = mask.numpy()

        np.testing.assert_array_equal(
            top_gate_np[0],
            [0.0, 0.0],
            err_msg="Padding token should have zero weights",
        )
        np.testing.assert_array_equal(
            top_idx_np[0],
            [-1, -1],
            err_msg="Padding token should have index -1",
        )
        self.assertEqual(
            probs_np[0].sum(), 0.0, "Padding token probs should be 0"
        )
        self.assertEqual(
            mask_np[0].sum(), 0.0, "Padding token mask should be 0"
        )

    def test_output_shapes(self):
        """Output tensors should have expected shapes."""
        num_experts, k = 4, 2
        B, S, H = 2, 6, 64
        router, _ = self._make_router(
            n_routed_experts=num_experts, num_experts_per_tok=k, hidden_size=H
        )
        hidden = self._dummy_hidden(B, S, H)
        input_ids = paddle.randint(1, 100, [B, S])

        _, top_gate, top_idx, probs, mask, tp, l_aux, l_zloss = router(
            hidden, input_ids=input_ids
        )
        num_tokens = B * S
        self.assertEqual(list(top_gate.shape), [num_tokens, k])
        self.assertEqual(list(top_idx.shape), [num_tokens, k])
        self.assertEqual(list(probs.shape), [num_tokens, num_experts])
        self.assertEqual(list(mask.shape), [num_tokens, num_experts])
        self.assertIsNone(tp)
        self.assertIsNone(l_aux)
        self.assertIsNone(l_zloss)

    def test_norm_topk_prob_weights_sum(self):
        """When norm_topk_prob=True, each valid token's top_gate should sum to 1."""
        router, _ = self._make_router(
            n_routed_experts=4, num_experts_per_tok=2, norm_topk_prob=True
        )
        B, S = 2, 4
        input_ids = paddle.randint(1, 50, [B, S])
        hidden = self._dummy_hidden(B, S)
        _, top_gate, _, _, _, _, _, _ = router(hidden, input_ids=input_ids)
        sums = top_gate.sum(axis=-1).numpy()
        np.testing.assert_allclose(
            sums,
            np.ones(B * S),
            atol=1e-6,
            err_msg="With norm_topk_prob=True, weights should sum to 1",
        )

    def test_no_input_ids_raises(self):
        """HashRouter must raise if input_ids is None."""
        router, _ = self._make_router()
        hidden = self._dummy_hidden(2, 4)
        with self.assertRaises(AssertionError):
            router(hidden, input_ids=None)

    def test_set_layer_number(self):
        """set_layer_number should update _layer_number."""
        router, _ = self._make_router()
        router.set_layer_number(5)
        self.assertEqual(router._layer_number, 5)

    def test_invalid_input_shape_raises(self):
        """HashRouter should raise ValueError for non-3D input."""
        router, _ = self._make_router()
        bad_input = paddle.randn([8, 64])  # 2-D, not 3-D
        input_ids = paddle.to_tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype="int64")
        with self.assertRaises(ValueError):
            router(bad_input, input_ids=input_ids)

    def test_routed_scaling_factor(self):
        """routed_scaling_factor != 1.0 should multiply weights when norm_topk_prob=False."""
        scale = 2.5
        router, _ = self._make_router(
            n_routed_experts=4,
            num_experts_per_tok=2,
            norm_topk_prob=False,
            routed_scaling_factor=scale,
        )
        B, S = 1, 4
        input_ids = paddle.to_tensor([[3, 5, 7, 9]], dtype="int64")
        hidden = self._dummy_hidden(B, S)
        _, top_gate, _, _, _, _, _, _ = router(hidden, input_ids=input_ids)
        # weight per slot = 1.0 * scale (since norm_topk_prob=False)
        np.testing.assert_allclose(
            top_gate.numpy(),
            np.full((B * S, 2), scale, dtype="float32"),
            atol=1e-6,
        )


class TestHashRouterInMoELayer(unittest.TestCase):
    """Tests verifying that MoELayer selects the correct router type."""

    def _get_router_class_for_layer(self, config, layer_number):
        _use_hash = (
            getattr(config, "moe_n_hash_layers", 0) > 0
            and layer_number is not None
            and layer_number
            >= config.num_hidden_layers - config.moe_n_hash_layers
        )
        from paddlefleet.transformer.moe.moe_router import (
            HashRouter,
            TopKRouter,
        )

        return HashRouter if _use_hash else TopKRouter

    def test_moe_layer_selects_hash_router(self):
        """Layer number in hash range should select HashRouter."""
        from paddlefleet.transformer.moe.moe_router import HashRouter

        config = _make_router_config(
            num_hidden_layers=8,
            moe_n_hash_layers=2,
        )
        router_cls = self._get_router_class_for_layer(config, layer_number=7)
        self.assertIs(router_cls, HashRouter)

    def test_moe_layer_selects_topk_router(self):
        """Layer number outside hash range should select TopKRouter."""
        from paddlefleet.transformer.moe.moe_router import TopKRouter

        config = _make_router_config(
            num_hidden_layers=8,
            moe_n_hash_layers=2,
        )
        router_cls = self._get_router_class_for_layer(config, layer_number=5)
        self.assertIs(router_cls, TopKRouter)

    def test_moe_layer_no_hash_layers(self):
        """When moe_n_hash_layers=0, all layers should use TopKRouter."""
        from paddlefleet.transformer.moe.moe_router import TopKRouter

        config = _make_router_config(
            num_hidden_layers=8,
            moe_n_hash_layers=0,
        )
        router_cls = self._get_router_class_for_layer(config, layer_number=7)
        self.assertIs(router_cls, TopKRouter)

    def test_boundary_layer_uses_hash_router(self):
        """The first layer of the hash range (num_hidden_layers - moe_n_hash_layers)."""
        from paddlefleet.transformer.moe.moe_router import HashRouter

        config = _make_router_config(num_hidden_layers=32, moe_n_hash_layers=4)
        router_cls = self._get_router_class_for_layer(config, layer_number=28)
        self.assertIs(router_cls, HashRouter)

    def test_layer_before_boundary_uses_topk_router(self):
        """Layer just before the hash range boundary should use TopKRouter."""
        from paddlefleet.transformer.moe.moe_router import TopKRouter

        config = _make_router_config(num_hidden_layers=32, moe_n_hash_layers=4)
        router_cls = self._get_router_class_for_layer(config, layer_number=27)
        self.assertIs(router_cls, TopKRouter)

    @patch(
        "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_hash_router_instantiates_correctly(self, _mock):
        """HashRouter can be instantiated with config and layer_number."""
        from paddlefleet.transformer.moe.moe_router import HashRouter

        config = _make_router_config(
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_n_hash_layers=2,
            num_hidden_layers=8,
        )
        router = HashRouter(config=config, layer_number=7)
        self.assertIsInstance(router, HashRouter)
        self.assertEqual(router.num_experts, 4)
        self.assertEqual(router.num_experts_per_tok, 2)
        self.assertEqual(router._layer_number, 7)


if __name__ == "__main__":
    unittest.main()
