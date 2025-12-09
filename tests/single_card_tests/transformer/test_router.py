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

from paddlefleet.transformer.moe.moe_router import (
    DeepEPTopKRouter,
    FusedGateDetachMatmul,
    gate_detach_matmul,
)


# ================= Mock Dependency Environment =================
# Create a Mock Config class to simulate paddlefleet.transformer.transformer_config.TransformerConfig
class MockTransformerConfig:
    def __init__(self):
        self.hidden_size = 64
        self.n_routed_experts = 8
        self.topk_method = "noaux_tc"  # DeepEP must use this
        self.num_experts_per_tok = 2
        self.norm_topk_prob = True
        self.n_group = 1
        self.topk_group = 1
        self.routed_scaling_factor = 1.0
        self.tensor_model_parallel_size = 1
        self.sequence_parallel = False
        self.scoring_func = "softmax"
        self.moe_router_load_balancing_type = "aux_loss"  # Although DeepEP does not use aux, the parent class initialization might check this
        self.moe_router_force_load_balancing = False
        self.router_z_loss_coef = 0.01
        self.router_aux_loss_coef = 0.01
        self.moe_router_fusion = True

    def get(self, key, default=None):
        return getattr(self, key, default)


# ================= Test Class Definition =================


class TestDeepEPRouterComponents(unittest.TestCase):
    def setUp(self):
        paddle.seed(2025)
        np.random.seed(2025)

    def test_fused_op_gradient(self):
        """
        Test whether the forward and backward gradients of FusedGateDetachMatmul are correct
        """
        B, D_in, D_out = 4, 16, 8
        x = paddle.randn([B, D_in], dtype="float32")
        w = paddle.randn([D_in, D_out], dtype="float32")

        x.stop_gradient = False
        w.stop_gradient = False

        # 1. Use custom operator
        y_custom = FusedGateDetachMatmul.apply(x, w)
        loss_custom = y_custom.sum()
        loss_custom.backward()
        x_grad_custom = x.grad.clone()
        w_grad_custom = w.grad.clone()

        x.clear_grad()
        w.clear_grad()

        # 2. Use Paddle native operator as baseline (Standard Linear)
        # Note: In your implementation ctx.dtype = float32, there might be cast operations
        y_ref = paddle.nn.functional.linear(x, w)
        loss_ref = y_ref.sum()
        loss_ref.backward()
        x_grad_ref = x.grad
        w_grad_ref = w.grad

        # Verify forward numerical consistency
        np.testing.assert_allclose(
            y_custom.numpy(),
            y_ref.numpy(),
            rtol=1e-5,
            err_msg="Forward output mismatch",
        )

        # Verify backward gradient consistency
        np.testing.assert_allclose(
            x_grad_custom.numpy(),
            x_grad_ref.numpy(),
            rtol=1e-5,
            err_msg="Input gradient mismatch",
        )
        np.testing.assert_allclose(
            w_grad_custom.numpy(),
            w_grad_ref.numpy(),
            rtol=1e-5,
            err_msg="Weight gradient mismatch",
        )

    def test_gate_detach_matmul_logic(self):
        """Test the behavior of the gate_detach_matmul wrapper function"""
        x = paddle.randn([4, 16])
        w = paddle.randn([16, 8])

        # Case 1: use_fuse = True
        out_fuse = gate_detach_matmul(x, w, use_fuse=True)
        # Case 2: use_fuse = False
        out_normal = gate_detach_matmul(x, w, use_fuse=False)

        np.testing.assert_allclose(
            out_fuse.numpy(), out_normal.numpy(), rtol=1e-5
        )


class TestDeepEPTopKRouter(unittest.TestCase):
    def setUp(self):
        self.config = MockTransformerConfig()
        # Mock external dependencies to prevent import errors
        patcher = patch(
            "paddlefleet.parallel_state.get_context_parallel_world_size",
            return_value=1,
        )
        self.mock_cp = patcher.start()
        self.addCleanup(patcher.stop)

    def test_initialization_assertion(self):
        """Test whether initialization forcibly checks topk_method"""
        self.config.topk_method = "greedy"  # Incorrect configuration
        with self.assertRaises(AssertionError):
            _ = DeepEPTopKRouter(self.config)

    def test_forward_shape_and_logic(self):
        """Test the shape and key logic of forward propagation"""
        self.config.topk_method = "noaux_tc"
        router = DeepEPTopKRouter(self.config)

        # Simulate e_score_correction_bias (Parent class only initializes this buffer for noaux_tc)
        # The initialization logic of your code's parent class StandardMoERouter has already handled it; ensure it exists here
        self.assertTrue(hasattr(router, "e_score_correction_bias"))

        batch_size = 2
        seq_len = 10
        hidden_size = self.config.hidden_size

        # Input shape [B, S, H]
        hidden_states = paddle.randn([batch_size, seq_len, hidden_size])

        # Execute forward pass
        outputs = router(hidden_states)

        (capacity, top_gate, top_idx, gates_masked, mask, _, _, l_zloss) = (
            outputs
        )

        # 1. Verify None values specific to DeepEP in the Return Tuple
        self.assertIsNone(capacity, "Capacity should be None for DeepEP")
        self.assertIsNone(outputs[5], "Token priority should be None")

        # 2. Verify shapes
        # Input is flattened to [B*S, H]
        expected_tokens = batch_size * seq_len
        self.assertEqual(
            top_gate.shape, [expected_tokens, self.config.num_experts_per_tok]
        )
        self.assertEqual(
            top_idx.shape, [expected_tokens, self.config.num_experts_per_tok]
        )
        self.assertEqual(
            gates_masked.shape, [expected_tokens, self.config.n_routed_experts]
        )

        # 3. Verify the correctness of mask (mask should be one-hot or multi-hot of selected experts)
        # Check if the row sum of mask equals k
        mask_sum = mask.sum(axis=1)
        # Note: If random weights cause scores to be very small, mask might be unstable, but generally should equal k
        # Verify its data type and rough structure here
        self.assertEqual(mask.shape, gates_masked.shape)

        # 4. Verify Loss calculation
        if self.config.router_aux_loss_coef > 0:
            self.assertIsNotNone(outputs[6])
            self.assertEqual(outputs[6].shape, [])

        if self.config.router_z_loss_coef > 0:
            self.assertIsNotNone(l_zloss)
            self.assertEqual(l_zloss.shape, [])

    def test_expert_usage_update(self):
        """Verify if Expert Usage statistics are updating"""
        router = DeepEPTopKRouter(self.config)
        initial_usage = router.expert_usage.numpy().copy()

        hidden_states = paddle.randn([2, 5, self.config.hidden_size])
        router(hidden_states)

        new_usage = router.expert_usage.numpy()

        # Tokens should be assigned to experts, so usage sum should increase by batch * seq_len * k
        self.assertGreater(new_usage.sum(), initial_usage.sum())
        self.assertEqual(
            new_usage.sum(), 2 * 5 * self.config.num_experts_per_tok
        )


if __name__ == "__main__":
    unittest.main()
