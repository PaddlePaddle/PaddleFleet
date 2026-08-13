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

import os
import unittest

os.environ["MOONEP_MEM_HANDLE_TYPE"] = "fd"

import paddle
import paddle.distributed as dist
from paddlefleet_ops import is_moonep_available


class TestMoonEP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not is_moonep_available():
            raise unittest.SkipTest("MoonEP is not available")
        if not dist.is_initialized():
            dist.init_parallel_env()
        cls.rank = dist.get_rank()
        cls.world_size = dist.get_world_size()
        if cls.world_size != 2:
            dist.destroy_process_group()
            raise unittest.SkipTest("MoonEP smoke test requires two ranks")

        from paddlefleet_ops.moonep import Buffer, MoonEPCommPlan

        cls.Buffer = Buffer
        cls.MoonEPCommPlan = MoonEPCommPlan

    @classmethod
    def tearDownClass(cls):
        from paddlefleet.transformer.moe import finalize_moonep

        finalize_moonep()
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_dispatch_combine_round_trip(self):
        S, H, K = 128, 128, 1
        E = self.world_size * 2
        buffer = self.Buffer(
            S=S,
            H=H,
            K=K,
            E=E,
            num_ep_ranks=self.world_size,
            num_sms=8,
        )
        try:
            hidden = paddle.arange(S * H, dtype="float32").reshape([S, H])
            hidden = (hidden / 1024 + self.rank).astype(paddle.bfloat16)
            route_weights = (
                paddle.arange(S, dtype="float32").reshape([S, K]) / S
            )

            remote_rank = (self.rank + 1) % self.world_size
            topk_experts = (
                paddle.arange(S, dtype="int32") % 2 + remote_rank * 2
            ).reshape([S, K])
            tokens_per_expert = paddle.bincount(
                topk_experts.reshape([-1]), minlength=E
            ).astype("int32")

            dispatched, dispatched_weights, cu_seqlens, plan = buffer.dispatch(
                hidden,
                route_weights,
                topk_experts,
                tokens_per_expert,
            )
            self.assertIsInstance(plan, self.MoonEPCommPlan)
            self.assertEqual(list(plan.dst.shape), [S * K])
            self.assertEqual(list(cu_seqlens.shape), [E + E // self.world_size])

            combined, combined_weights, _ = buffer.combine(
                plan=plan,
                hidden_nvsh=dispatched,
                route_weights_nvs=dispatched_weights,
            )
            paddle.device.synchronize()

            self.assertTrue(
                bool(
                    paddle.equal_all(
                        combined.astype("float32"),
                        hidden.astype("float32"),
                    )
                )
            )
            self.assertTrue(
                bool(paddle.equal_all(combined_weights, route_weights))
            )
        finally:
            buffer.destroy()

    def test_moe_layers_share_buffer_and_reuse_expert_activation(self):
        import paddle.nn.functional as F

        from paddlefleet.process_groups_config import ProcessGroupCollection
        from paddlefleet.tensor_parallel.random import (
            model_parallel_cuda_manual_seed,
        )
        from paddlefleet.transformer.moe.moe_layer import (
            MoELayer,
            MoESublayers,
        )
        from paddlefleet.transformer.transformer_config import (
            TransformerConfig,
        )

        group = dist.new_group(list(range(self.world_size)))
        pg_collection = ProcessGroupCollection(ep=group, expt_dp=None)
        model_parallel_cuda_manual_seed(
            123, tp_rank=0, ep_rank=self.rank, etp_rank=0
        )
        E, B, S, H, I, K = 4, 2, 128, 512, 1024, 2
        config = TransformerConfig(
            hidden_size=H,
            num_attention_heads=4,
            n_routed_experts=E,
            n_shared_experts=0,
            num_experts_per_tok=K,
            moe_intermediate_size=I,
            gated_linear_unit=True,
            hidden_act=F.gelu,
            use_bias=False,
            bf16=True,
            params_dtype=paddle.bfloat16,
            perform_initialization=False,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=self.world_size,
            sequence_parallel=False,
            moe_token_dispatcher_type="moonep",
            moe_expert_fusion=True,
            moe_use_fusion_node=True,
            moe_deep_gemm=False,
            router_aux_loss_coef=0.0,
        )
        layer = MoELayer(config, MoESublayers(), pg_collection)
        second_layer = MoELayer(config, MoESublayers(), pg_collection)
        self.assertFalse(layer.moe_use_fusion_node)

        local_ids = paddle.arange(B, dtype="float32") + self.rank * B
        local_w1 = (
            local_ids.reshape([B, 1, 1]) * 0.01
            + paddle.arange(H, dtype="float32").reshape([1, H, 1]) * 1e-4
            + paddle.arange(2 * I, dtype="float32").reshape([1, 1, 2 * I])
            * 1e-5
        ).astype("bfloat16")
        local_w2 = (
            local_ids.reshape([B, 1, 1]) * 0.02
            + paddle.arange(I, dtype="float32").reshape([1, I, 1]) * 1e-4
            + paddle.arange(H, dtype="float32").reshape([1, 1, H]) * 1e-5
        ).astype("bfloat16")
        for current_layer in (layer, second_layer):
            current_layer.grouped_gemm_experts.weight1.copy_(local_w1)
            current_layer.grouped_gemm_experts.weight2.copy_(local_w2)

        gathered_w1 = [
            paddle.empty_like(local_w1) for _ in range(self.world_size)
        ]
        gathered_w2 = [
            paddle.empty_like(local_w2) for _ in range(self.world_size)
        ]
        dist.all_gather(gathered_w1, local_w1, group=group)
        dist.all_gather(gathered_w2, local_w2, group=group)
        reference_w1 = paddle.concat(gathered_w1, axis=0).detach()
        reference_w2 = paddle.concat(gathered_w2, axis=0).detach()
        reference_w1.stop_gradient = False
        reference_w2.stop_gradient = False

        hidden_base = (
            paddle.arange(S * H, dtype="float32").reshape([S, H]) * 1e-4
            + self.rank * 0.01
        ).astype("bfloat16")
        hidden = hidden_base.detach()
        hidden.stop_gradient = False
        second_hidden = hidden_base.detach()
        second_hidden.stop_gradient = False
        reference_hidden = hidden_base.detach()
        reference_hidden.stop_gradient = False

        local_start = self.rank * B
        remote_start = (1 - self.rank) * B
        expert_offset = paddle.arange(S, dtype="int32") % B
        topk_indices = paddle.stack(
            [
                expert_offset + local_start,
                expert_offset + remote_start,
            ],
            axis=1,
        )
        valid_token_mask = (
            paddle.arange(S, dtype="int32").reshape([S, 1]) < S - 16
        )
        valid_route_mask = valid_token_mask.expand([S, K])
        topk_indices = paddle.where(
            valid_route_mask,
            topk_indices,
            paddle.full_like(topk_indices, -1),
        )
        weights_base = paddle.stack(
            [
                paddle.full([S], 0.4, dtype="float32"),
                paddle.full([S], 0.6, dtype="float32"),
            ],
            axis=1,
        )
        topk_weights = weights_base.detach()
        topk_weights.stop_gradient = False
        second_topk_weights = weights_base.detach()
        second_topk_weights.stop_gradient = False
        reference_topk_weights = weights_base.detach()
        reference_topk_weights.stop_gradient = False
        safe_topk_indices = paddle.where(
            valid_route_mask,
            topk_indices,
            paddle.zeros_like(topk_indices),
        )
        one_hot = F.one_hot(
            safe_topk_indices.astype("int64"), num_classes=E
        ).astype("float32")
        route_mask = valid_route_mask.astype("float32")
        routing_map = one_hot.sum(axis=1).astype("bool") & valid_token_mask
        probs = (
            one_hot * (topk_weights.detach() * route_mask).unsqueeze(-1)
        ).sum(axis=1)

        output = layer.custom_forward(
            hidden,
            probs,
            routing_map,
            topk_weights=topk_weights,
            topk_indices=topk_indices,
        )
        second_output = second_layer.custom_forward(
            second_hidden,
            probs,
            routing_map,
            topk_weights=second_topk_weights,
            topk_indices=topk_indices,
        )
        self.assertIs(
            layer.token_dispatcher._comm_manager._buffer,
            second_layer.token_dispatcher._comm_manager._buffer,
        )

        reference_route = (
            one_hot * (reference_topk_weights * route_mask).unsqueeze(-1)
        ).sum(axis=1)
        reference_output = paddle.zeros_like(reference_hidden)
        for expert_id in range(E):
            fc1 = paddle.matmul(reference_hidden, reference_w1[expert_id])
            gate, up = paddle.chunk(fc1, 2, axis=-1)
            expert_output = paddle.matmul(
                F.gelu(gate) * up, reference_w2[expert_id]
            )
            reference_output += expert_output * reference_route[
                :, expert_id
            ].astype(expert_output.dtype).unsqueeze(-1)

        grad = (
            paddle.arange(S * H, dtype="float32").reshape([S, H]) * 1e-5 + 0.5
        )
        (
            (output.astype("float32") * grad).sum()
            + (second_output.astype("float32") * grad).sum()
        ).backward()
        (reference_output.astype("float32") * grad).sum().backward()
        dist.all_reduce(reference_w1.grad, group=group)
        dist.all_reduce(reference_w2.grad, group=group)
        local_slice = slice(self.rank * B, (self.rank + 1) * B)

        for actual, expected, name, atol in (
            (output, reference_output, "output", 3e-2),
            (second_output, reference_output, "second_output", 3e-2),
            (hidden.grad, reference_hidden.grad, "hidden_grad", 4e-2),
            (
                second_hidden.grad,
                reference_hidden.grad,
                "second_hidden_grad",
                4e-2,
            ),
            (
                topk_weights.grad,
                reference_topk_weights.grad,
                "router_grad",
                4e-2,
            ),
            (
                second_topk_weights.grad,
                reference_topk_weights.grad,
                "second_router_grad",
                4e-2,
            ),
            (
                layer.grouped_gemm_experts.weight1.grad,
                reference_w1.grad[local_slice],
                "weight1_grad",
                5e-2,
            ),
            (
                second_layer.grouped_gemm_experts.weight1.grad,
                reference_w1.grad[local_slice],
                "second_weight1_grad",
                5e-2,
            ),
            (
                layer.grouped_gemm_experts.weight2.grad,
                reference_w2.grad[local_slice],
                "weight2_grad",
                5e-2,
            ),
            (
                second_layer.grouped_gemm_experts.weight2.grad,
                reference_w2.grad[local_slice],
                "second_weight2_grad",
                5e-2,
            ),
        ):
            self.assertTrue(
                bool(
                    paddle.allclose(
                        actual.astype("float32"),
                        expected.astype("float32"),
                        rtol=3e-2,
                        atol=atol,
                    )
                ),
                f"{name} mismatch",
            )

        padding_slice = slice(S - 16, S)
        for actual, name in (
            (output[padding_slice], "padding_output"),
            (second_output[padding_slice], "second_padding_output"),
            (hidden.grad[padding_slice], "padding_hidden_grad"),
            (
                second_hidden.grad[padding_slice],
                "second_padding_hidden_grad",
            ),
            (topk_weights.grad[padding_slice], "padding_router_grad"),
            (
                second_topk_weights.grad[padding_slice],
                "second_padding_router_grad",
            ),
        ):
            actual = actual.astype("float32")
            self.assertTrue(
                bool(
                    paddle.equal_all(
                        actual,
                        paddle.zeros_like(actual),
                    )
                ),
                f"{name} must be zero",
            )


if __name__ == "__main__":
    unittest.main()
