# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddlefleet.transformer.moe.moe_layer import MoELayer
from paddlefleet.transformer.moe.moe_router import StandardMoERouter
from paddlefleet.transformer.transformer_config import TransformerConfig


class TestTop2Router:
    def setup_method(self, method):
        moe_num_experts = 4
        self.transformer_config = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            moe_num_experts=moe_num_experts,
            use_cpu_initialization=True,
            moe_router_load_balancing_type="aux_loss",
            num_experts_per_tok=2,
            moe_intermediate_size=15,
            # moe_aux_loss_coeff=0,
            # bf16=True,
            # params_dtype=torch.bfloat16,
            # add_bias_linear=False,
        )
        transformer_layer_spec = get_gpt_layer_local_spec(
            num_experts=moe_num_experts, moe_grouped_gemm=False
        )
        self.sequential_mlp = MoELayer(
            self.transformer_config,
            transformer_layer_spec.sublayers_spec.mlp.extra_kwargs["sublayers"],
        )
        self.router = self.sequential_mlp.router
        print("done initializing")

    def teardown_method(self, method):
        pass

    @pytest.mark.internal
    def test_constructor(self):
        assert isinstance(self.router, StandardMoERouter)

        num_weights = sum([p.numel() for p in self.router.parameters()])
        assert num_weights == 12 * 4, num_weights

    @pytest.mark.internal
    @pytest.mark.parametrize("score_function", ["sigmoid", "softmax"])
    def test_router_forward(self, score_function):
        with torch.no_grad():
            self.router = self.router.cuda()
            self.router.config.moe_router_score_function = score_function
            # [num tokens, hidden size]
            hidden_states = torch.randn((32, 2, self.router.config.hidden_size))
            hidden_states = hidden_states.cuda().bfloat16()
            scores, indices = self.router(hidden_states)

    @pytest.mark.internal
    def test_aux_loss(self):
        self.sequential_mlp = self.sequential_mlp.cuda()

        # Without aux loss
        hidden_states = torch.randn((32, 2, self.router.config.hidden_size))
        hidden_states = hidden_states.cuda().bfloat16()
        out = self.sequential_mlp(hidden_states)[0]
        out.sum().mul_(0).backward()
        assert self.sequential_mlp.router.weight.grad.abs().sum() == 0

        # With aux loss
        self.transformer_config.moe_aux_loss_coeff = 1
        out = self.sequential_mlp(hidden_states)[0]
        out.sum().mul_(0).backward()
        assert self.sequential_mlp.router.weight.grad.abs().sum() > 0

        # With Z loss
        # TODO: Not implemented yet
        # self.transformer_config.moe_aux_loss_coeff = 0
        # self.transformer_config.moe_z_loss_coeff = 1
        # self.sequential_mlp.router.weight.grad.fill_(0)
        # out = self.sequential_mlp(hidden_states)[0]
        # out.sum().mul_(0).backward()
        # assert self.sequential_mlp.router.weight.grad.abs().sum() > 0

    @pytest.mark.internal
    def test_router_dtype(self):
        self.router = self.router.cuda()
        self.sequential_mlp = self.sequential_mlp.cuda()
        hidden_states = torch.randn(
            (32, 2, self.router.config.hidden_size), dtype=torch.bfloat16
        )
        hidden_states = hidden_states.cuda()

        # Test with default setting (bf16)
        self.router.config.moe_router_dtype = None
        with torch.no_grad():
            scores, routing_map = self.router(hidden_states)
            out = self.sequential_mlp(hidden_states)
            assert scores.dtype == torch.bfloat16, (
                "Router output should be bf16 by default"
            )
            assert out[0].dtype == torch.bfloat16

        # Test with fp32 enabled
        self.router.config.moe_router_dtype = "fp32"
        with torch.no_grad():
            scores, routing_map = self.router(hidden_states)
            out = self.sequential_mlp(hidden_states)
            assert scores.dtype == torch.float32, (
                "Router output should be fp32 when enabled"
            )
            assert out[0].dtype == torch.bfloat16
            self.sequential_mlp.config.moe_token_dispatcher_type = "alltoall"
            out = self.sequential_mlp(hidden_states)
            assert out[0].dtype == torch.bfloat16
            self.sequential_mlp.config.moe_token_dispatcher_type = "allgather"

        # Test with fp64 enabled
        self.router.config.moe_router_dtype = "fp64"
        with torch.no_grad():
            scores, routing_map = self.router(hidden_states)
            out = self.sequential_mlp(hidden_states)
            assert scores.dtype == torch.float64, (
                "Router output should be fp64 when enabled"
            )
            assert out[0].dtype == torch.bfloat16

    # TODO: Not implemented yet
    # @pytest.mark.internal
    # def test_force_load_balancing(self):
    #     hidden_states = torch.randn(
    #         (32, 2, self.router.config.hidden_size), device="cuda", dtype=torch.bfloat16
    #     )
    #     hidden_states.requires_grad = True

    #     # First forward pass with normal routing
    #     normal_scores, normal_routing_map = self.router(hidden_states)

    #     # Second forward pass with force load balancing
    #     self.router.config.moe_router_force_load_balancing = True
    #     force_scores, force_routing_map = self.router(hidden_states)

    #     assert normal_scores.shape == force_scores.shape
    #     assert normal_routing_map.shape == force_routing_map.shape
    #     assert torch.equal(normal_scores, force_scores) == False

    #     # Backward pass for force load balancing
    #     self.router.zero_grad()
    #     force_scores.sum().backward()
    #     assert hidden_states.grad is not None
    #     assert self.router.weight.grad.norm() > 0

    #     self.router.config.moe_router_force_load_balancing = False

    # TODO: capacity_factor,pad_to_capacity not implemented yet
    # @pytest.mark.internal
    # @pytest.mark.parametrize("capacity_factor", [None, 1.0, 2.0])
    # @pytest.mark.parametrize("drop_policy", ["probs", "position"])
    # @pytest.mark.parametrize("pad_to_capacity", [True, False])
    # def test_token_dropping(self, capacity_factor, drop_policy, pad_to_capacity):
    #     if capacity_factor is None and pad_to_capacity:
    #         pytest.skip("Capacity factor is None, so no token dropping should be applied")

    #     num_tokens = 32
    #     self.router = self.router.cuda()
    #     self.router.config.moe_expert_capacity_factor = capacity_factor
    #     self.router.config.moe_token_drop_policy = drop_policy
    #     self.router.config.moe_pad_expert_input_to_capacity = pad_to_capacity

    #     hidden_states = torch.randn(
    #         (num_tokens, self.router.config.hidden_size), dtype=torch.bfloat16, device="cuda"
    #     )
    #     hidden_states.requires_grad = True
    #     probs, routing_map = self.router(hidden_states)

    #     if capacity_factor is not None:
    #         if pad_to_capacity:
    #             assert (
    #                 routing_map.sum().item()
    #                 == num_tokens * self.router.config.num_experts_per_tok * capacity_factor
    #             )
    #         else:
    #             assert (
    #                 routing_map.sum().item()
    #                 <= num_tokens * self.router.config.num_experts_per_tok * capacity_factor
    #             )
    #     else:
    #         assert routing_map.sum().item() == num_tokens * self.router.config.num_experts_per_tok

    #     # restore the config
    #     self.router.config.moe_expert_capacity_factor = None
    #     self.router.config.moe_token_drop_policy = "probs"
    #     self.router.config.moe_pad_expert_input_to_capacity = False
