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
        "moe_router_force_load_balancing": False,
        "moe_router_load_balancing_type": "aux_loss",
        "moe_deep_gemm": False,
        "router_aux_loss_coef": 0.01,
        "router_z_loss_coef": None,
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
        # FusedGateDetachMatmul.forward does w = w.T internally, then F.linear(x, w.T).
        # So w must be [E, D] (n_experts, hidden) to produce output [B, E].
        w = paddle.randn([4, 64], dtype=paddle.float32)
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


if __name__ == "__main__":
    unittest.main()
