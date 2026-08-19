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

from paddlefleet.transformer.moe import (
    moe_expert,
    moe_layer,
    moonep,
    token_dispatcher,
)
from paddlefleet.transformer.moe.moe_expert import GroupedMLPExpert
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

    def test_setup_metadata_falls_back_to_probability_topk(self):
        manager = object.__new__(_MoonEPManager)
        manager.router_topk = 2
        manager.num_experts = 4
        routing_map = paddle.to_tensor([[1, 0, 1, 0]], dtype="bool")
        probs = paddle.to_tensor([[0.6, 0.0, 0.4, 0.0]], dtype="float32")

        manager.setup_metadata(routing_map, probs)

        self.assertEqual(manager.token_indices.tolist(), [[0, 2]])
        self.assertEqual(manager.tokens_per_expert.tolist(), [1, 0, 1, 0])

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

    def test_manager_requires_bound_experts_and_active_plan(self):
        manager = object.__new__(_MoonEPManager)
        manager._bridge = None
        with self.assertRaisesRegex(RuntimeError, "must be bound"):
            manager._ensure_buffer(paddle.ones([2, 4], dtype="bfloat16"))

        manager.handle = None
        with self.assertRaisesRegex(RuntimeError, "active plan"):
            manager.runtime_expert_weights(object())
        with self.assertRaisesRegex(NotImplementedError, "E\\+B group counts"):
            manager.get_dispatched_metadata()

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

    def test_runtime_layout_trims_and_restores_padding(self):
        manager = object.__new__(_MoonEPManager)
        manager.tokens_per_expert = paddle.to_tensor([1, 2], dtype="int64")
        manager.dispatched_probs = paddle.to_tensor(
            [1.0, 0.5, 0.25, 0.0, 0.0], dtype="float32"
        )
        manager._num_dispatched_tokens = None
        dispatched = paddle.arange(10, dtype="float32").reshape([5, 2])

        valid = manager.get_permuted_hidden_states_by_experts(dispatched)
        restored = manager.get_restored_hidden_states_by_experts(
            paddle.ones_like(valid)
        )

        self.assertEqual(valid.shape, [3, 2])
        self.assertTrue(
            bool(
                paddle.equal_all(
                    restored,
                    paddle.to_tensor(
                        [
                            [1.0, 1.0],
                            [0.5, 0.5],
                            [0.25, 0.25],
                            [0.0, 0.0],
                            [0.0, 0.0],
                        ]
                    ),
                )
            )
        )

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
        for owner, target, value, error in (
            (config, "bf16", False, "requires bf16=True"),
            (layer, "moe_expert_fusion", False, "moe_expert_fusion=True"),
            (layer, "fp8", True, "BF16 expert compute only"),
            (layer, "use_ue8m0", True, "UE8M0 or SonicMoE"),
            (layer, "using_sonic_moe", True, "UE8M0 or SonicMoE"),
            (layer, "moe_deep_gemm", True, "moe_deep_gemm"),
            (
                config,
                "moe_expert_capacity_factor",
                1.0,
                "token dropping or capacity padding",
            ),
            (layer, "num_experts_per_tok", 33, "num_experts_per_tok <= 32"),
            (
                layer,
                "moe_intermediate_size",
                64,
                "dimensions to be multiples of 128",
            ),
        ):
            with self.subTest(target=target):
                original = getattr(owner, target)
                setattr(owner, target, value)
                with self.assertRaisesRegex(ValueError, error):
                    MoELayer._validate_moonep_config(layer)
                setattr(owner, target, original)

        config.gated_linear_unit = False
        layer.moe_use_fusion_node = True
        layer.moe_shared_expert_overlap = True
        MoELayer._validate_moonep_config(layer)
        self.assertFalse(layer.moe_use_fusion_node)
        self.assertFalse(layer.moe_shared_expert_overlap)

    def test_grouped_expert_uses_runtime_weights_and_activation(self):
        activation = mock.Mock(side_effect=lambda hidden: hidden + 1)
        expert = types.SimpleNamespace(
            moe_deep_gemm=False,
            activation_recompute=False,
            activation_func=activation,
            weight1=paddle.zeros([1]),
            weight2=paddle.zeros([1]),
        )
        hidden = paddle.ones([2, 2])
        tokens_per_expert = paddle.to_tensor([2], dtype="int64")
        runtime_weight1 = paddle.ones([1, 2, 4])
        runtime_weight2 = paddle.ones([1, 4, 2])
        fc1_output = paddle.full([2, 4], 2.0)
        expected = paddle.full([2, 2], 3.0)

        with mock.patch.object(
            moe_expert.BMMFunction,
            "apply",
            side_effect=(fc1_output, expected),
        ) as grouped_gemm:
            output, _ = GroupedMLPExpert.forward(
                expert,
                hidden,
                tokens_per_expert,
                expert_weights=(runtime_weight1, runtime_weight2),
            )

        self.assertIs(grouped_gemm.call_args_list[0].args[1], runtime_weight1)
        self.assertIs(grouped_gemm.call_args_list[1].args[1], runtime_weight2)
        activation.assert_called_once()
        self.assertIs(activation.call_args.args[0], fc1_output)
        self.assertTrue(bool(paddle.equal_all(output, expected)))

    def test_expert_forward_delegates_runtime_weights_to_grouped_expert(self):
        hidden = paddle.ones([2, 2])
        tokens_per_expert = paddle.to_tensor([2], dtype="int64")
        runtime_weights = (mock.sentinel.weight1, mock.sentinel.weight2)
        expected = paddle.full([2, 2], 3.0)
        grouped_expert = mock.Mock(return_value=(expected, None))
        layer = types.SimpleNamespace(
            _use_grouped_mlp_expert=True,
            grouped_gemm_experts=grouped_expert,
        )

        output = MoELayer.expert_forward(
            layer,
            hidden,
            tokens_per_expert,
            expert_weights=runtime_weights,
        )

        self.assertIs(output, expected)
        grouped_expert.assert_called_once_with(
            hidden,
            tokens_per_expert,
            expert_weights=runtime_weights,
        )

        layer._use_grouped_mlp_expert = False
        with self.assertRaisesRegex(ValueError, "grouped expert storage"):
            MoELayer.expert_forward(
                layer,
                hidden,
                tokens_per_expert,
                expert_weights=runtime_weights,
            )

    def test_runtime_weights_delegate_gradient_reduction(self):
        class _Bridge:
            def __init__(self):
                self.full_weights = (
                    paddle.zeros([2, 2], dtype="float32"),
                    paddle.zeros([2, 2], dtype="float32"),
                )
                self.plan = None

            def prepare(self, weight1, weight2, plan):
                self.plan = plan
                for runtime, local in zip(
                    self.full_weights, (weight1, weight2)
                ):
                    runtime.copy_(local)

            def prefetch(self, _plan, projection_index=None):
                del projection_index

            def reduce_grads(self, plan, grad_weight1, grad_weight2):
                self.plan = plan
                return grad_weight1 * 5, grad_weight2 * 7

        grouped_experts = types.SimpleNamespace(
            weight1=paddle.ones([2, 2], dtype="float32"),
            weight2=paddle.ones([2, 2], dtype="float32"),
        )
        grouped_experts.weight1.stop_gradient = False
        grouped_experts.weight2.stop_gradient = False
        bridge = _Bridge()
        plan = object()

        runtime_weight1, runtime_weight2 = moonep.moonep_runtime_weights(
            grouped_experts, bridge, plan
        )
        (runtime_weight1 * 2).sum().add((runtime_weight2 * 3).sum()).backward()

        self.assertIs(bridge.plan, plan)
        self.assertTrue(
            bool(
                paddle.equal_all(
                    grouped_experts.weight1.grad,
                    paddle.full([2, 2], 10.0),
                )
            )
        )
        self.assertTrue(
            bool(
                paddle.equal_all(
                    grouped_experts.weight2.grad,
                    paddle.full([2, 2], 21.0),
                )
            )
        )

    def test_runtime_weights_are_restored_per_bmm_backward(self):
        class _Bridge:
            def __init__(self):
                self.full_weights = (
                    paddle.zeros([1, 2, 2], dtype="float32"),
                    paddle.zeros([1, 2, 2], dtype="float32"),
                )

            @paddle.no_grad()
            def prepare(self, weight1, weight2, plan):
                for runtime, local in zip(
                    self.full_weights, (weight1, weight2)
                ):
                    runtime.copy_(local)
                self.prefetch(plan)

            @paddle.no_grad()
            def prefetch(self, plan, projection_index=None):
                indices = (
                    range(len(self.full_weights))
                    if projection_index is None
                    else (projection_index,)
                )
                for index in indices:
                    self.full_weights[index].copy_(plan.weights[index])

            def reduce_grads(self, _plan, grad_weight1, grad_weight2):
                return grad_weight1, grad_weight2

        grouped_experts = types.SimpleNamespace(
            weight1=paddle.ones([1, 2, 2], dtype="float32"),
            weight2=paddle.ones([1, 2, 2], dtype="float32"),
        )
        for weight in (grouped_experts.weight1, grouped_experts.weight2):
            weight.stop_gradient = False
        expert = types.SimpleNamespace(
            moe_deep_gemm=False,
            activation_recompute=False,
            activation_func=lambda hidden: hidden,
        )
        plans = [
            types.SimpleNamespace(
                weights=(
                    paddle.to_tensor([[[value, 0.0], [0.0, value + 1]]]),
                    paddle.eye(2).reshape([1, 2, 2]),
                )
            )
            for value in (2.0, 5.0)
        ]
        bridge = _Bridge()
        tokens_per_expert = paddle.to_tensor([1], dtype="int64")
        inputs = [
            paddle.to_tensor([[1.0, 1.0]], stop_gradient=False) for _ in plans
        ]

        outputs = []
        for hidden, plan in zip(inputs, plans):
            runtime_weights = moonep.moonep_runtime_weights(
                grouped_experts, bridge, plan
            )
            output, _ = GroupedMLPExpert.forward(
                expert,
                hidden,
                tokens_per_expert,
                expert_weights=runtime_weights,
            )
            outputs.append(output)

        (outputs[0].sum() + outputs[1].sum()).backward()

        self.assertEqual(inputs[0].grad.tolist(), [[2.0, 3.0]])
        self.assertEqual(inputs[1].grad.tolist(), [[5.0, 6.0]])


class TestMoonEPBufferCache(unittest.TestCase):
    def setUp(self):
        moonep._buffer_cache.clear()
        moonep._bridges.clear()
        self.addCleanup(moonep._buffer_cache.clear)
        self.addCleanup(moonep._bridges.clear)

    def test_cache_reuses_only_matching_signatures(self):
        group = object()
        buffers = [mock.Mock(), mock.Mock()]
        common = {
            "H": 128,
            "K": 2,
            "E": 4,
            "B": 2,
            "num_ep_ranks": 2,
            "group": group,
        }

        with (
            mock.patch.object(moonep, "_MOONEP_AVAILABLE", True),
            mock.patch.object(
                moonep, "MoonEPBuffer", side_effect=buffers
            ) as buffer_cls,
        ):
            first = moonep.get_moonep_buffer(S=2, num_sms="8", **common)
            first_again = moonep.get_moonep_buffer(S=2, num_sms=8, **common)
            second = moonep.get_moonep_buffer(S=3, **common)

        self.assertIs(first, buffers[0])
        self.assertIs(first_again, buffers[0])
        self.assertIs(second, buffers[1])
        self.assertEqual(buffer_cls.call_count, 2)

    def test_finalize_destroys_each_cached_buffer_once(self):
        class _Resource:
            def __init__(self):
                self.destroy_calls = 0

            def destroy(self):
                self.destroy_calls += 1

        buffers = [_Resource(), _Resource()]
        bridge = _Resource()
        moonep._buffer_cache.update(
            {("first",): buffers[0], ("second",): buffers[1]}
        )
        moonep._bridges.add(bridge)

        moonep.finalize_moonep()
        moonep.finalize_moonep()

        self.assertEqual([buffer.destroy_calls for buffer in buffers], [1, 1])
        self.assertEqual(bridge.destroy_calls, 1)
        self.assertEqual(moonep._buffer_cache, {})


class TestMoonEPWeightBridge(unittest.TestCase):
    def test_prefetch_can_restore_one_projection(self):
        bridge = object.__new__(moonep.MoonEPWeightBridge)
        bridge.buffer = mock.Mock()
        bridge.buffer._require_ctx.return_value = {"num_sms": 8}
        bridge.rank = 0
        bridge.num_experts = 1
        bridge.projections = [
            types.SimpleNamespace(
                full_weight=paddle.zeros([2, 1, 1], dtype="bfloat16")
            )
            for _ in range(2)
        ]
        plan = types.SimpleNamespace(
            experts_to_copy=paddle.zeros([1, 1], dtype="int32")
        )

        with mock.patch.object(
            moonep, "launch_prefetch", create=True
        ) as launch:
            bridge.prefetch(plan)
            self.assertEqual(launch.call_count, 2)
            launch.reset_mock()
            bridge.prefetch(plan, projection_index=1)

        launch.assert_called_once()
        self.assertEqual(
            launch.call_args.args[1].data_ptr(),
            bridge.projections[1].full_weight[bridge.num_experts :].data_ptr(),
        )

    def test_unavailable_runtime_and_unaligned_storage_are_rejected(self):
        with (
            mock.patch.object(moonep, "_MOONEP_AVAILABLE", False),
            self.assertRaisesRegex(ImportError, "MoonEP is unavailable"),
        ):
            moonep._require_moonep()

        group = object()
        with (
            mock.patch.object(moonep, "_MOONEP_AVAILABLE", True),
            mock.patch.object(paddle.distributed, "get_rank", return_value=0),
            mock.patch.object(
                paddle.distributed, "get_world_size", return_value=2
            ),
            mock.patch.object(
                moonep, "get_vmm_granularity", return_value=256, create=True
            ),
            self.assertRaisesRegex(ValueError, "must be VMM aligned"),
        ):
            moonep._allocate_mapping(
                [1], paddle.float32, group, with_reduce_view=False
            )

    def test_fabric_mapping_includes_owner_and_reduce_views(self):
        group = object()
        expert_handle = paddle.to_tensor([11, 12], dtype="int64")
        slot_handle = paddle.to_tensor([21, 22], dtype="int64")
        gathered_experts = paddle.to_tensor([[11, 12], [13, 14]], dtype="int64")
        gathered_slots = paddle.to_tensor([[21, 22], [23, 24]], dtype="int64")

        with (
            mock.patch.object(moonep, "_MOONEP_AVAILABLE", True),
            mock.patch.object(paddle.distributed, "get_rank", return_value=1),
            mock.patch.object(
                paddle.distributed, "get_world_size", return_value=2
            ),
            mock.patch.object(
                moonep, "get_vmm_granularity", return_value=128, create=True
            ),
            mock.patch.object(
                moonep, "_use_fabric_for_group", return_value=True, create=True
            ),
            mock.patch.object(
                moonep,
                "nvl_dist_alloc",
                side_effect=(
                    (
                        mock.sentinel.expert_keepalive,
                        expert_handle,
                        mock.sentinel.expert_owned,
                    ),
                    (
                        mock.sentinel.slot_keepalive,
                        slot_handle,
                        mock.sentinel.slot_owned,
                    ),
                ),
                create=True,
            ),
            mock.patch.object(
                moonep,
                "_all_gather_shareables",
                side_effect=(gathered_experts, gathered_slots),
                create=True,
            ) as gather,
            mock.patch.object(
                moonep,
                "nvl_dist_map",
                side_effect=(mock.sentinel.full, mock.sentinel.reduce_view),
                create=True,
            ),
            mock.patch.object(
                moonep, "nvl_release_mem_handle", create=True
            ) as release_handle,
        ):
            mapped, mapped_reduce, keepalives = moonep._allocate_mapping(
                [128], paddle.bfloat16, group, with_reduce_view=True
            )

        self.assertIs(mapped, mock.sentinel.full)
        self.assertIs(mapped_reduce, mock.sentinel.reduce_view)
        self.assertEqual(
            keepalives,
            (mock.sentinel.expert_keepalive, mock.sentinel.slot_keepalive),
        )
        self.assertEqual(gather.call_count, 2)
        release_handle.assert_has_calls(
            [
                mock.call(mock.sentinel.expert_owned),
                mock.call(mock.sentinel.slot_owned),
            ]
        )

    def test_bridge_validates_layout_and_requires_a_buffer(self):
        common = {
            "group": object(),
            "num_local_experts": 1,
            "weight1_shape": [1, 128, 128],
            "weight2_shape": [1, 128, 128],
        }
        with (
            mock.patch.object(paddle.distributed, "get_rank", return_value=0),
            mock.patch.object(
                paddle.distributed, "get_world_size", return_value=2
            ),
        ):
            with self.assertRaisesRegex(ValueError, "even expert distribution"):
                moonep.MoonEPWeightBridge(num_experts=3, **common)
            with self.assertRaisesRegex(
                ValueError, "grouped 3-D expert weights"
            ):
                moonep.MoonEPWeightBridge(
                    num_experts=2,
                    **{**common, "weight1_shape": [128, 128]},
                )
            with (
                mock.patch.object(
                    moonep,
                    "_allocate_mapping",
                    side_effect=RuntimeError("stop after validation"),
                ),
                self.assertRaisesRegex(RuntimeError, "stop after validation"),
            ):
                moonep.MoonEPWeightBridge(num_experts=2, **common)

        bridge = object.__new__(moonep.MoonEPWeightBridge)
        bridge.buffer = None
        with self.assertRaisesRegex(RuntimeError, "no communication buffer"):
            bridge.prepare(None, None, None)

        bridge.buffer = mock.Mock()
        bridge.rank = 0
        bridge.num_local_experts = 1
        bridge.projections = []
        plan = object()
        with (
            mock.patch.object(
                moonep, "launch_inter_rank_sync", create=True
            ) as sync,
            mock.patch.object(bridge, "prefetch") as prefetch,
        ):
            bridge.prepare(None, None, plan)
        sync.assert_called_once_with(bridge.buffer._require_ctx.return_value)
        prefetch.assert_called_once_with(plan)

    def test_destroy_is_idempotent(self):
        bridge = object.__new__(moonep.MoonEPWeightBridge)
        bridge.projections = [object()]
        bridge.buffer = object()
        bridge._destroyed = False

        bridge.destroy()
        bridge.destroy()

        self.assertEqual(bridge.projections, [])
        self.assertIsNone(bridge.buffer)
        self.assertTrue(bridge._destroyed)


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
