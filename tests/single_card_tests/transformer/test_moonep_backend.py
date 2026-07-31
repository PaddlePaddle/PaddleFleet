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

import contextlib
import types
import unittest
from unittest import mock

import paddle

from paddlefleet.transformer.moe import moe_layer, token_dispatcher
from paddlefleet.transformer.moe.moe_layer import MoELayer
from paddlefleet.transformer.moe.token_dispatcher import (
    MoEFlexTokenDispatcher,
    _MoonEPManager,
    is_hybrid_ep_backend_selected,
)


class TestMoonEPDispatcher(unittest.TestCase):
    def test_moonep_is_a_valid_non_hybridep_backend(self):
        self.assertFalse(is_hybrid_ep_backend_selected("moonep"))

    def test_moonep_selects_its_own_manager(self):
        group = types.SimpleNamespace(world_size=2, nranks=2)
        with mock.patch.object(
            token_dispatcher, "is_moonep_available", return_value=True
        ):
            dispatcher = MoEFlexTokenDispatcher(
                num_local_experts=2,
                num_experts_per_tok=2,
                n_routed_experts=4,
                ep_group=group,
                dispatcher_type="moonep",
            )
        self.assertIsInstance(dispatcher._comm_manager, _MoonEPManager)

    def test_setup_metadata_uses_logical_expert_counts(self):
        manager = object.__new__(_MoonEPManager)
        manager.router_topk = 2
        manager.num_experts = 4
        routing_map = paddle.to_tensor(
            [[1, 1, 0, 0], [0, 1, 0, 1]], dtype="bool"
        )
        probs = paddle.to_tensor(
            [[0.6, 0.4, 0.0, 0.0], [0.0, 0.7, 0.0, 0.3]],
            dtype="float32",
        )
        topk_weights = paddle.to_tensor(
            [[0.6, 0.4], [0.7, 0.3]], dtype="float32"
        )
        topk_indices = paddle.to_tensor([[0, 1], [1, 3]], dtype="int64")

        manager.setup_metadata(routing_map, probs, topk_weights, topk_indices)

        self.assertEqual(manager.token_indices.dtype, paddle.int32)
        self.assertEqual(manager.token_probs.dtype, paddle.float32)
        self.assertEqual(manager.tokens_per_expert.tolist(), [1, 2, 0, 1])

    def test_setup_metadata_sanitizes_padding_routes(self):
        manager = object.__new__(_MoonEPManager)
        manager.router_topk = 2
        manager.num_experts = 4
        routing_map = paddle.to_tensor(
            [[1, 1, 0, 0], [0, 0, 0, 0]], dtype="bool"
        )
        probs = paddle.to_tensor(
            [[0.6, 0.4, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            dtype="float32",
        )
        topk_weights = paddle.to_tensor(
            [[0.6, 0.4], [0.7, 0.3]], dtype="float32"
        )
        topk_weights.stop_gradient = False
        topk_indices = paddle.to_tensor([[0, 1], [-1, -1]], dtype="int64")

        manager.setup_metadata(routing_map, probs, topk_weights, topk_indices)

        self.assertEqual(manager.token_indices.tolist(), [[0, 1], [0, 0]])
        self.assertTrue(
            bool(
                paddle.allclose(
                    manager.token_probs,
                    paddle.to_tensor([[0.6, 0.4], [0.0, 0.0]], dtype="float32"),
                )
            )
        )
        self.assertEqual(manager.tokens_per_expert.tolist(), [3, 1, 0, 0])
        self.assertEqual(int(manager.tokens_per_expert.sum().item()), 4)

        manager.token_probs.sum().backward()
        self.assertEqual(topk_weights.grad.tolist(), [[1.0, 1.0], [0.0, 0.0]])

    def test_dispatch_rejects_non_bf16_hidden_states(self):
        manager = object.__new__(_MoonEPManager)
        manager._bridge = object()
        manager._buffer = object()
        manager._buffer_signature = (2, 4, "paddle.bfloat16", 1, 2, 1)
        manager.router_topk = 1
        manager.num_experts = 2
        manager.num_local_experts = 1
        with self.assertRaisesRegex(ValueError, "BF16 hidden states"):
            manager._ensure_buffer(paddle.ones([2, 4], dtype="float32"))

    def test_dispatch_overlap_is_explicitly_unsupported(self):
        manager = object.__new__(_MoonEPManager)
        with self.assertRaisesRegex(
            NotImplementedError, "does not support dispatch overlap"
        ):
            manager.dispatch_overlap(
                paddle.ones([2, 4], dtype="bfloat16"),
                paddle.zeros([2, 1], dtype="int32"),
                paddle.ones([2, 1], dtype="float32"),
            )

    def test_buffer_rejects_inconsistent_rank_signatures(self):
        manager = object.__new__(_MoonEPManager)
        manager.group = object()
        manager.router_topk = 1
        manager.num_experts = 2
        manager.num_local_experts = 1
        manager._bridge = mock.Mock()
        manager._buffer = None

        def gather_signatures(output, signature, group):
            del group
            output[:] = [signature, (3, *signature[1:])]

        with (
            mock.patch.object(
                paddle.distributed, "get_world_size", return_value=2
            ),
            mock.patch.object(
                paddle.distributed,
                "all_gather_object",
                side_effect=gather_signatures,
            ),
            self.assertRaisesRegex(ValueError, "identical dispatch signature"),
        ):
            manager._ensure_buffer(paddle.ones([2, 4], dtype="bfloat16"))

    def test_moonep_config_rejects_unsupported_modes(self):
        config = types.SimpleNamespace(
            bf16=True,
            gated_linear_unit=True,
            moe_expert_capacity_factor=None,
            moe_latent_size=None,
            hidden_size=128,
        )
        layer = types.SimpleNamespace(
            config=config,
            moe_expert_fusion=True,
            fp8=False,
            fp8_dispatch=False,
            use_w4a8=False,
            use_ue8m0=False,
            using_sonic_moe=False,
            moe_deep_gemm=False,
            num_experts_per_tok=2,
            use_latent_moe=False,
            moe_intermediate_size=128,
            moe_use_fusion_node=False,
            moe_shared_expert_overlap=False,
        )
        for target, value, error in (
            ("bf16", False, "requires bf16=True"),
            ("use_ue8m0", True, "UE8M0 or SonicMoE"),
            ("using_sonic_moe", True, "UE8M0 or SonicMoE"),
        ):
            with self.subTest(target=target):
                owner = config if target == "bf16" else layer
                original = getattr(owner, target)
                setattr(owner, target, value)
                with self.assertRaisesRegex(ValueError, error):
                    MoELayer._validate_moonep_config(layer)
                setattr(owner, target, original)


class TestMoonEPBalanceLog(unittest.TestCase):
    @staticmethod
    def _layer(dispatcher_type):
        layer = types.SimpleNamespace()
        layer.use_latent_moe = False
        layer.moe_token_dispatcher_type = dispatcher_type
        layer.layer_number = 1
        layer.is_mtp_layer = False
        layer.moe_group = object()
        layer.num_experts_per_tok = 2
        layer.token_dispatcher = mock.Mock()
        layer.token_dispatcher.get_dispatched_routing.return_value = (
            None,
            None,
            paddle.to_tensor([1, 1], dtype="int64"),
        )
        layer.dispatch = mock.Mock(return_value=(paddle.ones([2, 4]), None))
        layer.routed_experts_compute = mock.Mock(
            side_effect=lambda hidden: hidden
        )
        layer.combine = mock.Mock(side_effect=lambda hidden: hidden)
        return layer

    def _run_custom_forward(self, dispatcher_type):
        layer = self._layer(dispatcher_type)
        with (
            mock.patch.object(
                moe_layer.framework,
                "_dygraph_tracer",
                return_value=types.SimpleNamespace(_has_grad=True),
            ),
            mock.patch.object(
                moe_layer,
                "global_moe_balance_training_logs_enabled",
                return_value=True,
            ),
            mock.patch.object(moe_layer, "log_moe_balance") as log_balance,
            mock.patch.object(
                moe_layer,
                "profile",
                side_effect=lambda _name: contextlib.nullcontext(),
            ),
        ):
            MoELayer.custom_forward(
                layer,
                paddle.ones([2, 4]),
                paddle.ones([2, 2]),
                paddle.ones([2, 2], dtype="bool"),
            )
        return log_balance

    def test_moonep_skips_balance_log(self):
        self._run_custom_forward("moonep").assert_not_called()

    def test_deepep_keeps_balance_log(self):
        self._run_custom_forward("deepep").assert_called_once()


if __name__ == "__main__":
    unittest.main()
