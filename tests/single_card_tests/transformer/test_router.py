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

import unittest
from unittest.mock import patch

import numpy as np
import paddle
import paddle.nn.functional as F

# Adjust the import path according to your actual project structure
from paddlefleet.transformer.moe.moe_router import (
    FusedGateDetachMatmul,
    TopKRouter,
)


# ================= Mock Dependency Environment =================
class MockTransformerConfig:
    """
    Mock configuration object to simulate
    paddlefleet.transformer.transformer_config.TransformerConfig
    """

    def __init__(self):
        # Basic Model Parameters
        self.hidden_size = 64
        self.n_routed_experts = 8
        self.num_experts_per_tok = 2
        self.n_group = 1
        self.topk_group = 1

        # Router Specific Parameters
        self.topk_method = "noaux_tc"
        self.norm_topk_prob = True
        self.routed_scaling_factor = 1.0
        self.scoring_func = "softmax"
        self.moe_router_load_balancing_type = "aux_loss"
        self.moe_router_force_load_balancing = False
        self.moe_router_fusion = True

        # Loss Coefficients
        self.router_z_loss_coef = 0.01
        self.router_aux_loss_coef = 0.01

        # Parallelism Parameters
        self.tensor_model_parallel_size = 1
        self.context_parallel_size = 1
        self.sequence_parallel = False

        # Internal storage to simulate the .get() method behavior
        self._extra_conf = {"seq_aux": False}

    def get(self, key, default=None):
        """
        Simulate the dictionary-like get behavior of the config object.
        It prioritizes _extra_conf, then falls back to object attributes.
        """
        return self._extra_conf.get(key, getattr(self, key, default))


# ================= Test Class Definition =================


class TestRouterComponents(unittest.TestCase):
    def setUp(self):
        paddle.seed(2025)
        np.random.seed(2025)

    def test_fused_op_gradient(self):
        """
        Test the correctness of forward and backward gradients for the
        custom FusedGateDetachMatmul operator.
        """
        B, D_in, D_out = 4, 16, 8
        x = paddle.randn([B, D_in], dtype="float32")
        w = paddle.randn([D_in, D_out], dtype="float32")

        x.stop_gradient = False
        w.stop_gradient = False

        # 1. Custom Operator Path
        y_custom = FusedGateDetachMatmul.apply(x, w)
        loss_custom = y_custom.sum()
        loss_custom.backward()
        x_grad_custom = x.grad.clone()
        w_grad_custom = w.grad.clone()

        x.clear_grad()
        w.clear_grad()

        # 2. Paddle Native Operator Path (Baseline)
        y_ref = F.linear(x, w)
        loss_ref = y_ref.sum()
        loss_ref.backward()

        # Verify numerical consistency
        np.testing.assert_allclose(
            y_custom.numpy(),
            y_ref.numpy(),
            rtol=1e-5,
            err_msg="Forward output mismatch",
        )
        np.testing.assert_allclose(
            x_grad_custom.numpy(),
            x.grad.numpy(),
            rtol=1e-5,
            err_msg="Input gradient mismatch",
        )
        np.testing.assert_allclose(
            w_grad_custom.numpy(),
            w.grad.numpy(),
            rtol=1e-5,
            err_msg="Weight gradient mismatch",
        )


class TestTopKRouter(unittest.TestCase):
    def setUp(self):
        self.config = MockTransformerConfig()

        # Mock the parallel state function to prevent errors during single-card testing.
        # Note: Adjust the patch path if get_context_parallel_world_size is imported differently.
        patcher = patch(
            "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
            return_value=1,
        )
        self.mock_cp = patcher.start()
        self.addCleanup(patcher.stop)

    def test_initialization_modes(self):
        """
        Test that initialization behavior differs based on `topk_method`.
        """
        # Case 1: noaux_tc (Common mode for DeepEP)
        self.config.topk_method = "noaux_tc"
        router_tc = TopKRouter(self.config)
        # Verify that score correction bias is registered
        self.assertTrue(hasattr(router_tc, "e_score_correction_bias"))
        # Verify that expert_usage is initialized
        self.assertTrue(hasattr(router_tc, "expert_usage"))

        # Case 2: greedy (Standard mode)
        self.config.topk_method = "greedy"
        router_greedy = TopKRouter(self.config)
        # Verify that these attributes do NOT exist in greedy mode
        self.assertFalse(hasattr(router_greedy, "e_score_correction_bias"))
        self.assertFalse(hasattr(router_greedy, "expert_usage"))

    def test_call_topk_method_directly(self):
        """
        Directly test `_call_topk_method` to ensure it returns a tuple (gate, idx).
        This isolates the routing logic from the forward pass preprocessing
        and prevents unpacking errors.
        """
        router = TopKRouter(self.config)
        batch_size = 2
        seq_len = 5
        # Simulate Gates [B*S, E]
        gates = paddle.rand(
            [batch_size * seq_len, self.config.n_routed_experts]
        )

        # Test 1: noaux_tc
        res = router._call_topk_method(
            "noaux_tc", gates, k=2, n_group=1, topk_group=1
        )
        self.assertIsNotNone(
            res, "_call_topk_method returned None for 'noaux_tc'"
        )
        self.assertIsInstance(res, tuple, "Should return a tuple")
        self.assertEqual(len(res), 2, "Should return (top_gate, top_idx)")

        # Test 2: greedy
        res_greedy = router._call_topk_method("greedy", gates, k=2)
        self.assertIsNotNone(
            res_greedy, "_call_topk_method returned None for 'greedy'"
        )
        self.assertEqual(len(res_greedy), 2)

    def test_forward_shape_and_logic(self):
        """
        Test the input/output shapes of the forward pass and verify
        TopKRouter-specific return values (e.g., None for capacity).
        """
        self.config.topk_method = "noaux_tc"
        router = TopKRouter(self.config)

        batch_size = 2
        seq_len = 10
        hidden_size = self.config.hidden_size

        # Input must be 3D [B, S, H]
        hidden_states = paddle.randn([batch_size, seq_len, hidden_size])

        # Execute Forward
        outputs = router(hidden_states)

        # Ensure output is not None
        self.assertIsNotNone(outputs, "Forward returned None")

        (
            capacity,  # Should be None
            top_gate,
            top_idx,
            gates_masked,
            mask,
            token_priority,  # Should be None
            l_aux,
            l_zloss,
        ) = outputs

        # 1. Verify DeepEP/TopKRouter specific None return values
        self.assertIsNone(capacity, "Capacity should be None for TopKRouter")
        self.assertIsNone(
            token_priority, "Token priority should be None for TopKRouter"
        )

        # 2. Verify Shapes
        expected_tokens = batch_size * seq_len
        k = self.config.num_experts_per_tok
        n_experts = self.config.n_routed_experts

        self.assertEqual(top_gate.shape, [expected_tokens, k])
        self.assertEqual(top_idx.shape, [expected_tokens, k])
        self.assertEqual(mask.shape, [expected_tokens, n_experts])

        # 3. Verify Loss Calculation
        if self.config.router_aux_loss_coef > 0:
            self.assertIsNotNone(l_aux)
            self.assertEqual(l_aux.shape, [])  # Expecting a scalar

        if self.config.router_z_loss_coef > 0:
            self.assertIsNotNone(l_zloss)
            self.assertEqual(l_zloss.shape, [])  # Expecting a scalar

    def test_input_dimension_assertion(self):
        """
        Ensure the router raises ValueError for incorrect input dimensions
        (e.g., 2D tensors instead of 3D).
        """
        router = TopKRouter(self.config)
        # Input 2D [B*S, H] -> Should raise ValueError as TopKRouter strictly checks len(shape)==2
        hidden_states = paddle.randn([20, self.config.hidden_size])
        with self.assertRaises(ValueError):
            router(hidden_states)

    def test_expert_usage_update(self):
        """
        Verify that `expert_usage` is correctly updated when running in 'noaux_tc' mode.
        """
        self.config.topk_method = "noaux_tc"
        router = TopKRouter(self.config)

        # Initial state should be all zeros
        initial_usage = router.expert_usage.numpy().copy()
        self.assertEqual(initial_usage.sum(), 0)

        hidden_states = paddle.randn([2, 5, self.config.hidden_size])
        router(hidden_states)

        new_usage = router.expert_usage.numpy()

        # Usage sum should equal Total Tokens * K
        expected_hits = 2 * 5 * self.config.num_experts_per_tok
        self.assertEqual(new_usage.sum(), expected_hits)
        self.assertGreater(new_usage.sum(), initial_usage.sum())

    def test_greedy_no_usage_update(self):
        """
        Verify that `expert_usage` logic is ignored (attribute does not exist)
        when running in 'greedy' mode.
        """
        self.config.topk_method = "greedy"
        router = TopKRouter(self.config)

        hidden_states = paddle.randn([2, 5, self.config.hidden_size])

        # Run forward to ensure no errors occur due to accessing missing attributes
        outputs = router(hidden_states)
        self.assertIsNotNone(outputs)

        # Double check that the attribute still does not exist
        self.assertFalse(hasattr(router, "expert_usage"))


if __name__ == "__main__":
    unittest.main()
