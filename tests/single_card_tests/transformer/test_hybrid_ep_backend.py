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

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import paddle

from paddlefleet.transformer.moe import token_dispatcher
from paddlefleet.transformer.moe.fp8_utils import FP8_ALIGN
from paddlefleet.transformer.moe.fused_a2a import (
    HybridEPCombine,
    HybridEPDispatch,
    _replay_hybrid_ep_dispatch_backward,
)
from paddlefleet.transformer.moe.fusion_layer_utils import (
    HybridEPMoePyLayer,
    _hybrid_ep_prepare_expert_counts,
    _pad_front_rows,
    _restore_hybrid_ep_prob_grad_shape,
)
from paddlefleet.transformer.moe.moe_layer import MoELayer, MoESublayers
from paddlefleet.transformer.moe.token_dispatcher import (
    MoEFlexTokenDispatcher,
    _HybridEPManager,
    is_hybrid_ep_backend_selected,
)
from paddlefleet.transformer.transformer_config import TransformerConfig


def _make_moe_config(**overrides):
    defaults = {
        "hidden_size": 64,
        "num_attention_heads": 2,
        "intermediate_size": 256,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 128,
        "gated_linear_unit": True,
        "sequence_parallel": False,
        "tensor_model_parallel_size": 1,
        "moe_token_dispatcher_type": "hybridep",
        "moe_use_fusion_node": True,
        "moe_grouped_gemm": False,
        "moe_ep_barrier": True,
        "fp8": None,
        "fp8_wgrad": True,
        "using_sonic_moe": False,
        "router_aux_loss_coef": 0.01,
        "router_z_loss_coef": None,
        "topk_method": "greedy",
        "norm_topk_prob": True,
        "scoring_func": "softmax",
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 1.0,
        "moe_router_force_load_balancing": False,
        "moe_router_load_balancing_type": "aux_loss",
        "n_shared_experts": 1,
        "moe_shared_expert_overlap": True,
        "recompute_granularity": None,
        "recompute_modules": [],
        "use_bias": False,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _make_pg_collection(moe_world_size=2):
    return SimpleNamespace(
        ep=SimpleNamespace(world_size=moe_world_size),
        expt_dp=object(),
        tp=SimpleNamespace(size=lambda: 1),
        cp=SimpleNamespace(rank=lambda: 0, size=lambda: 1),
    )


class _PyLayerContext:
    def set_grad_in_dtype_consistent(self, enabled):
        self.grad_in_dtype_consistent = enabled

    def save_for_backward(self, tensors):
        self._saved_tensors = tensors

    def saved_tensor(self):
        return (self._saved_tensors,)


def _make_hybrid_ep_handle(
    num_dispatched_tokens=2,
    local_expert_routing_map=None,
    tokens_per_rank=8,
    token_data_type="BF16",
    num_experts_per_rank=2,
):
    if local_expert_routing_map is None:
        local_expert_routing_map = paddle.ones(
            [num_dispatched_tokens, num_experts_per_rank],
            dtype="bool",
        )
    return (
        None,
        None,
        None,
        paddle.to_tensor(num_dispatched_tokens, dtype="int64"),
        local_expert_routing_map,
        None,
        tokens_per_rank,
        SimpleNamespace(
            token_data_type=token_data_type,
            num_of_experts_per_rank=num_experts_per_rank,
        ),
        None,
    )


class _RecordingHybridEPBuffer:
    def __init__(
        self,
        dispatch_results=(),
        combine_results=(),
        replay_config=None,
    ):
        self.dispatch_results = list(dispatch_results)
        self.combine_results = list(combine_results)
        self.replay_config = replay_config
        self.dispatch_calls = []
        self.combine_calls = []
        self.update_template_config_calls = []

    def dispatch_with_permute(self, **kwargs):
        self.dispatch_calls.append(kwargs)
        return self.dispatch_results.pop(0)

    def combine_with_unpermute(self, **kwargs):
        self.combine_calls.append(kwargs)
        return self.combine_results.pop(0)

    def update_template_config(self, **kwargs):
        self.update_template_config_calls.append(kwargs)
        return self.replay_config


def _new_hybrid_manager(**overrides):
    manager = _HybridEPManager.__new__(_HybridEPManager)
    manager.group = SimpleNamespace(nranks=2)
    manager.router_topk = 2
    manager.num_experts = 4
    manager.num_local_experts = 2
    manager.routing_map = None
    manager.routing_probs = None
    manager.token_indices = None
    manager.token_probs = None
    manager.dispatched_indices = None
    manager.dispatched_probs = None
    manager.tokens_per_expert = None
    manager.padded_tokens_per_expert = None
    manager.handle = None
    manager._buffer = None
    manager._buffer_hidden_dim = None
    manager._buffer_max_num_of_tokens_per_rank = 0
    for key, value in overrides.items():
        setattr(manager, key, value)
    return manager


class TestHybridEPBackendSelection(unittest.TestCase):
    def test_backend_defaults_to_deepep_without_backend_config(self):
        with patch.object(token_dispatcher, "HAVE_HYBRID_EP", True):
            self.assertFalse(is_hybrid_ep_backend_selected())

    def test_non_hybrid_dispatchers_do_not_select_hybrid_backend(self):
        with patch.object(token_dispatcher, "HAVE_HYBRID_EP", True):
            for dispatcher_type in ("allgather", "alltoall", "deepep"):
                with self.subTest(dispatcher_type=dispatcher_type):
                    self.assertFalse(
                        is_hybrid_ep_backend_selected(dispatcher_type)
                    )

    def test_backend_selects_hybrid_explicitly(self):
        with patch.object(token_dispatcher, "HAVE_HYBRID_EP", True):
            self.assertTrue(is_hybrid_ep_backend_selected("hybridep"))

    def test_backend_rejects_invalid_and_unavailable_hybrid(self):
        for dispatcher_type in ("unknown", "hybrid", "hybrid_ep", "deep_ep"):
            with (
                self.subTest(dispatcher_type=dispatcher_type),
                self.assertRaisesRegex(ValueError, "moe_token_dispatcher_type"),
            ):
                is_hybrid_ep_backend_selected(dispatcher_type)
        with (
            patch.object(token_dispatcher, "HAVE_HYBRID_EP", False),
            self.assertRaisesRegex(ImportError, "HybridEP runtime"),
        ):
            is_hybrid_ep_backend_selected("hybridep")

    def test_flex_dispatcher_uses_hybrid_backend(self):
        group = SimpleNamespace(world_size=2)

        with patch.object(token_dispatcher, "HAVE_HYBRID_EP", True):
            dispatcher = MoEFlexTokenDispatcher(
                num_local_experts=2,
                num_experts_per_tok=2,
                n_routed_experts=4,
                ep_group=group,
                dispatcher_type="hybridep",
            )

        self.assertIsInstance(dispatcher._comm_manager, _HybridEPManager)
        self.assertIs(dispatcher._comm_manager.group, group)
        self.assertEqual(dispatcher._comm_manager.num_local_experts, 2)

    def test_hybrid_ep_dispatch_keeps_counts_on_device(self):
        routing_map = paddle.to_tensor(
            [[True, False], [False, True]], dtype="bool"
        )
        manager = _new_hybrid_manager(
            group=SimpleNamespace(nranks=1),
            router_topk=1,
            num_experts=2,
            num_local_experts=2,
            routing_map=routing_map,
            routing_probs=paddle.to_tensor(
                [[1.0, 0.0], [0.0, 1.0]], dtype="float32"
            ),
        )
        padded_counts = paddle.to_tensor([1, 1], dtype="int64")
        buffer = _RecordingHybridEPBuffer(
            dispatch_results=[
                (
                    paddle.zeros([2, 4], dtype="float32"),
                    paddle.ones([2], dtype="float32"),
                    None,
                    padded_counts,
                    _make_hybrid_ep_handle(
                        num_dispatched_tokens=2,
                        local_expert_routing_map=routing_map,
                    ),
                )
            ]
        )
        manager._get_buffer = lambda hidden_states: buffer

        manager._dispatch_with_permute_impl(
            paddle.zeros([2, 4], dtype="float32"),
            paddle.to_tensor([[0], [1]], dtype="int64"),
            paddle.ones([2, 1], dtype="float32"),
            use_fp8=False,
        )

        self.assertTrue(buffer.dispatch_calls[-1]["non_blocking"])
        self.assertIs(manager.padded_tokens_per_expert, padded_counts)
        self.assertIsInstance(manager.tokens_per_expert, paddle.Tensor)

    def test_hybrid_ep_dispatch_quantizes_when_fp8_enabled(self):
        manager = _new_hybrid_manager(
            group=SimpleNamespace(nranks=1),
            router_topk=1,
            num_experts=2,
            num_local_experts=2,
            routing_map=paddle.to_tensor(
                [[True, False], [False, True]], dtype="bool"
            ),
            routing_probs=paddle.to_tensor(
                [[1.0, 0.0], [0.0, 1.0]], dtype="float32"
            ),
        )
        padded_counts = paddle.to_tensor([1, 1], dtype="int64")
        buffer = _RecordingHybridEPBuffer(
            dispatch_results=[
                (
                    paddle.zeros([2, 4], dtype="float32"),
                    paddle.ones([2], dtype="float32"),
                    paddle.ones([2, 1], dtype="float32"),
                    padded_counts,
                    _make_hybrid_ep_handle(
                        num_dispatched_tokens=2,
                        local_expert_routing_map=manager.routing_map,
                    ),
                )
            ]
        )
        quantized_hidden = paddle.full([2, 4], 2.0, dtype="float32")
        scaling_factor = paddle.ones([2, 1], dtype="float32")
        manager._get_buffer = lambda hidden_states: buffer

        with (
            patch(
                "paddle.incubate.nn.functional.fp8_quant_blockwise",
                return_value=(quantized_hidden, scaling_factor),
            ) as mock_quant,
        ):
            manager._dispatch_with_permute_impl(
                paddle.ones([2, 4], dtype="float32"),
                paddle.to_tensor([[0], [1]], dtype="int64"),
                paddle.ones([2, 1], dtype="float32"),
                use_fp8=True,
            )

        mock_quant.assert_called_once()
        dispatch_kwargs = buffer.dispatch_calls[-1]
        self.assertIs(dispatch_kwargs["hidden"], quantized_hidden)
        self.assertTrue(dispatch_kwargs["use_fp8"])
        self.assertEqual(dispatch_kwargs["pad_multiple"], FP8_ALIGN)
        self.assertEqual(dispatch_kwargs["scaling_factor"].shape, [1, 2])


class TestHybridEPManagerContract(unittest.TestCase):
    def test_buffer_is_reused_until_shape_or_capacity_changes(self):
        constructed = []

        class FakeHybridEPBuffer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                constructed.append(self)

        manager = _new_hybrid_manager(num_local_experts=3)
        fake_deep_ep = SimpleNamespace(HybridEPBuffer=FakeHybridEPBuffer)

        with patch.object(token_dispatcher, "deep_ep", fake_deep_ep):
            first = manager._get_buffer(paddle.zeros([4, 16]))
            second = manager._get_buffer(paddle.zeros([2, 16]))
            third = manager._get_buffer(
                paddle.zeros([2, 16]), max_num_of_tokens_per_rank=8
            )
            fourth = manager._get_buffer(paddle.zeros([2, 32]))

        self.assertIs(first, second)
        self.assertIsNot(first, third)
        self.assertIsNot(third, fourth)
        self.assertEqual(len(constructed), 3)
        self.assertEqual(constructed[0].kwargs["hidden_dim"], 16)
        self.assertEqual(constructed[0].kwargs["num_local_experts"], 3)
        self.assertTrue(constructed[0].kwargs["load_cached_kernels"])

    def test_topk_indices_are_converted_to_dense_metadata(self):
        manager = _new_hybrid_manager(num_experts=4)
        token_indices = paddle.to_tensor(
            [[1, -1], [0, 2], [3, 1]], dtype="int64"
        )
        token_weights = paddle.to_tensor(
            [[0.5, 0.0], [0.25, 0.75], [0.6, 0.4]],
            dtype="float16",
        )

        routing_map, probs = manager._indices_to_dense_metadata(
            token_indices, token_weights
        )

        self.assertEqual(
            routing_map.numpy().tolist(),
            [
                [False, True, False, False],
                [True, False, True, False],
                [False, True, False, True],
            ],
        )
        self.assertEqual(probs.dtype, paddle.float32)
        self.assertTrue(
            paddle.allclose(
                probs,
                paddle.to_tensor(
                    [
                        [0.0, 0.5, 0.0, 0.0],
                        [0.25, 0.0, 0.75, 0.0],
                        [0.0, 0.4, 0.0, 0.6],
                    ],
                    dtype="float32",
                ),
                atol=1e-3,
            ).item()
        )

    def test_dispatch_metadata_prefers_cached_dense_metadata(self):
        routing_map = paddle.to_tensor([[True, False]], dtype="bool")
        routing_probs = paddle.to_tensor([[1.0, 0.0]], dtype="float32")
        manager = _new_hybrid_manager(
            routing_map=routing_map,
            routing_probs=routing_probs,
        )

        cached_map, cached_probs = manager._get_dispatch_metadata(None, None)
        self.assertIs(cached_map, routing_map)
        self.assertIs(cached_probs, routing_probs)

        manager.routing_map = None
        with self.assertRaisesRegex(AssertionError, "routing metadata"):
            manager._get_dispatch_metadata(None, None)

    def test_setup_metadata_uses_router_topk_when_available(self):
        manager = _new_hybrid_manager(router_topk=2, num_experts=4)
        routing_map = paddle.to_tensor(
            [[True, False, True, False], [False, True, True, False]],
            dtype="bool",
        )
        probs = paddle.to_tensor(
            [[0.7, 0.0, 0.3, 0.0], [0.0, 0.6, 0.4, 0.0]],
            dtype="float16",
        )
        topk_weights = paddle.to_tensor(
            [[0.7, 0.3], [0.6, 0.4]], dtype="float32"
        )
        topk_indices = paddle.to_tensor([[0, 2], [1, 2]], dtype="int64")

        manager.setup_metadata(
            routing_map,
            probs,
            topk_weights=topk_weights,
            topk_indices=topk_indices,
        )

        self.assertEqual(manager.routing_probs.dtype, paddle.float32)
        self.assertTrue(
            paddle.allclose(
                manager.token_probs,
                paddle.to_tensor([[0.7, 0.3], [0.6, 0.4]], dtype="float32"),
                atol=1e-6,
            ).item()
        )
        self.assertEqual(
            manager.token_indices.numpy().tolist(), [[0, 2], [1, 2]]
        )
        self.assertTrue(manager.token_indices.stop_gradient)

    def test_setup_metadata_falls_back_to_dense_topk(self):
        manager = _new_hybrid_manager(router_topk=2, num_experts=4)
        routing_map = paddle.to_tensor(
            [[True, False, True, False]], dtype="bool"
        )
        probs = paddle.to_tensor([[0.2, 0.1, 0.7, 0.0]], dtype="float32")

        manager.setup_metadata(routing_map, probs)

        self.assertEqual(manager.token_indices.numpy().tolist(), [[2, 0]])
        self.assertTrue(
            paddle.allclose(
                manager.token_probs,
                paddle.to_tensor([[0.7, 0.2]], dtype="float32"),
                atol=1e-6,
            ).item()
        )

    def test_runtime_metadata_accessors_follow_hybrid_ep_contract(self):
        manager = _new_hybrid_manager(router_topk=2, num_local_experts=3)
        self.assertEqual(manager._get_num_permuted_tokens_upper_bound(5), 401)

        local_expert_routing_map = paddle.to_tensor(
            [
                [True, False, False],
                [False, True, True],
                [False, True, False],
            ],
            dtype="bool",
        )
        tokens_per_expert = manager._extract_tokens_per_expert(
            2, local_expert_routing_map
        )
        self.assertEqual(tokens_per_expert.numpy().tolist(), [1, 1, 1])

        hidden_states = paddle.ones([2, 3], dtype="float32")
        self.assertIs(
            manager.get_permuted_hidden_states_by_experts(hidden_states),
            hidden_states,
        )
        self.assertIs(
            manager.get_restored_hidden_states_by_experts(hidden_states),
            hidden_states,
        )

        manager.dispatched_probs = paddle.to_tensor([0.25, 0.5])
        restored = manager.get_restored_hidden_states_by_experts(hidden_states)
        self.assertEqual(
            restored.numpy().tolist(),
            [[0.25, 0.25, 0.25], [0.5, 0.5, 0.5]],
        )

        with self.assertRaisesRegex(NotImplementedError, "does not expose"):
            manager.get_dispatched_metadata()
        manager.dispatched_indices = paddle.to_tensor([[0], [1]], dtype="int64")
        dispatched_indices, dispatched_probs = manager.get_dispatched_metadata()
        self.assertIs(dispatched_indices, manager.dispatched_indices)
        self.assertIs(dispatched_probs, manager.dispatched_probs)
        manager.tokens_per_expert = tokens_per_expert
        self.assertIs(
            manager.get_number_of_tokens_per_expert(), tokens_per_expert
        )

    def test_dispatch_overlap_stores_topk_metadata_and_scale_handle(self):
        manager = _new_hybrid_manager(num_local_experts=2)
        hidden_states = paddle.zeros([2, 4])
        token_indices = paddle.to_tensor([[0, 1], [1, 0]], dtype="int64")
        token_weights = paddle.ones([2, 2], dtype="float32")
        dispatched = paddle.ones([4, 4], dtype="float32")
        dispatched_probs = paddle.ones([4], dtype="float32")
        scale = paddle.ones([4, 1], dtype="float32")
        buffer = _RecordingHybridEPBuffer(
            dispatch_results=[
                (
                    dispatched,
                    dispatched_probs,
                    scale,
                    paddle.to_tensor([2, 2], dtype="int64"),
                    _make_hybrid_ep_handle(num_dispatched_tokens=4),
                )
            ]
        )
        manager._get_buffer = lambda hidden_states: buffer

        result, fp8_handle = manager.dispatch_overlap(
            hidden_states,
            token_indices,
            token_weights,
            fp8_dispatch=False,
            async_finish=True,
        )

        self.assertTrue(paddle.equal_all(result, dispatched).item())
        self.assertEqual(fp8_handle, {"scale": scale})
        self.assertIs(manager.token_indices, token_indices)
        self.assertIs(manager.token_probs, token_weights)
        self.assertIs(manager.dispatched_probs, dispatched_probs)
        self.assertIsNone(manager.dispatched_indices)
        self.assertFalse(buffer.dispatch_calls[-1]["use_fp8"])

    def test_dispatch_reuses_setup_metadata(self):
        manager = _new_hybrid_manager(
            token_indices=paddle.to_tensor([[0]], dtype="int64"),
            token_probs=paddle.ones([1, 1], dtype="float32"),
            num_local_experts=2,
        )
        hidden_states = paddle.zeros([1, 4], dtype="float32")
        buffer = _RecordingHybridEPBuffer(
            dispatch_results=[
                (
                    hidden_states,
                    paddle.ones([1], dtype="float32"),
                    None,
                    paddle.to_tensor([1, 0], dtype="int64"),
                    _make_hybrid_ep_handle(num_dispatched_tokens=1),
                )
            ]
        )
        manager._get_buffer = lambda hidden_states: buffer

        result, scale_handle = manager.dispatch(
            hidden_states, async_finish=True
        )

        self.assertTrue(paddle.equal_all(result, hidden_states).item())
        self.assertIsNone(scale_handle)
        self.assertFalse(buffer.dispatch_calls[-1]["use_fp8"])

    def test_dispatch_preprocess_overlap_records_topk_metadata(self):
        dispatcher = MoEFlexTokenDispatcher.__new__(MoEFlexTokenDispatcher)
        dispatcher._comm_manager = SimpleNamespace()
        hidden_states = paddle.zeros([2, 3, 4], dtype="float32")
        token_probs = paddle.ones([6, 2], dtype="float32")
        token_indices = paddle.zeros([6, 2], dtype="int64")

        flattened = dispatcher.dispatch_preprocess_overlap(
            hidden_states,
            token_probs,
            token_indices,
        )

        self.assertEqual(flattened.shape, [6, 4])
        self.assertEqual(dispatcher.hidden_shape, [2, 3, 4])
        self.assertIsNone(dispatcher._comm_manager.routing_map)
        self.assertIsNone(dispatcher._comm_manager.routing_probs)
        self.assertIs(dispatcher._comm_manager.token_probs, token_probs)
        self.assertIs(dispatcher._comm_manager.token_indices, token_indices)


class TestHybridEPFusedA2ABridge(unittest.TestCase):
    def test_dispatch_bridge_records_runtime_state_and_maps_prob_grads(self):
        ctx = _PyLayerContext()
        grad_dense_probs = paddle.to_tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype="float32"
        )
        grad_x = paddle.ones([2, 4], dtype="float32")
        buffer = _RecordingHybridEPBuffer(
            combine_results=[(grad_x, grad_dense_probs)]
        )
        handle = ("handle",)

        class DispatchingManager:
            def _dispatch_with_permute_impl(
                self, x, token_indices, token_probs, use_fp8
            ):
                self.dispatch_args = (
                    x,
                    token_indices,
                    token_probs,
                    use_fp8,
                )
                self._buffer = buffer
                self.handle = handle
                return (
                    x + 1,
                    token_probs.reshape([-1]),
                    paddle.ones([2, 1], dtype="float32"),
                )

        manager = DispatchingManager()
        x = paddle.zeros([2, 4], dtype="float32")
        token_indices = paddle.to_tensor([[2, 0], [1, 1]], dtype="int64")
        token_probs = paddle.ones([2, 2], dtype="float32")

        recv_x, recv_probs, scale = HybridEPDispatch.forward(
            ctx, x, token_indices, token_probs, manager, fp8_dispatch=True
        )

        self.assertEqual(recv_x.numpy().tolist(), [[1.0] * 4, [1.0] * 4])
        self.assertEqual(recv_probs.shape, [4])
        self.assertEqual(scale.shape, [2, 1])
        self.assertIs(ctx.buffer, buffer)
        self.assertIs(ctx.handle, handle)
        self.assertIs(ctx.token_indices, token_indices)
        self.assertEqual(ctx.hidden_dtype, paddle.float32)
        self.assertTrue(ctx.use_fp8_dispatch)
        self.assertFalse(ctx.grad_in_dtype_consistent)
        self.assertTrue(manager.dispatch_args[-1])

        grad_hidden, grad_indices, grad_probs = HybridEPDispatch.backward(
            ctx,
            paddle.ones([2, 4], dtype="float64"),
            paddle.ones([2, 2], dtype="float16"),
        )

        self.assertIs(grad_hidden, grad_x)
        self.assertIsNone(grad_indices)
        self.assertTrue(
            paddle.allclose(
                grad_probs,
                paddle.to_tensor([[0.3, 0.1], [0.5, 0.5]], dtype="float32"),
                atol=1e-6,
            ).item()
        )
        self.assertEqual(buffer.combine_calls[-1]["pad_multiple"], FP8_ALIGN)
        self.assertEqual(
            buffer.combine_calls[-1]["hidden"].dtype, paddle.float32
        )
        self.assertEqual(
            buffer.combine_calls[-1]["probs"].dtype, paddle.float32
        )

    def test_dispatch_bridge_allows_missing_prob_grad(self):
        ctx = _PyLayerContext()
        ctx.buffer = _RecordingHybridEPBuffer(
            combine_results=[(paddle.ones([1, 2], dtype="float32"), None)]
        )
        ctx.handle = ("handle",)
        ctx.token_indices = paddle.to_tensor([[0]], dtype="int64")
        ctx.hidden_dtype = paddle.float32
        ctx.use_fp8_dispatch = False

        _, _, grad_probs = HybridEPDispatch.backward(
            ctx,
            paddle.ones([1, 2], dtype="float32"),
            None,
        )

        self.assertIsNone(grad_probs)
        self.assertIsNone(ctx.buffer.combine_calls[-1]["probs"])
        self.assertIsNone(ctx.buffer.combine_calls[-1]["pad_multiple"])

    def test_replay_dispatch_backward_rebuilds_fp8_handle(self):
        original_config = SimpleNamespace(
            token_data_type="UINT8",
            num_of_experts_per_rank=2,
        )
        replay_config = SimpleNamespace(token_data_type="UINT16")
        handle = (
            "sparse_to_dense",
            "rdma_to_attn",
            "attn_to_rdma",
            "num_tokens",
            "local_map",
            "rank_prefix",
            16,
            original_config,
            "tail",
        )
        buffer = _RecordingHybridEPBuffer(
            dispatch_results=[
                (
                    paddle.arange(12, dtype="float32").reshape([3, 4]),
                    None,
                    None,
                    None,
                    None,
                )
            ],
            replay_config=replay_config,
        )

        grad_x = _replay_hybrid_ep_dispatch_backward(
            buffer,
            handle,
            paddle.ones([2, 4], dtype="float32"),
            num_permuted_tokens=2,
            use_fp8_dispatch=True,
        )

        self.assertEqual(grad_x.shape, [2, 4])
        self.assertEqual(
            buffer.update_template_config_calls,
            [
                {
                    "hidden_dim": 4,
                    "num_of_tokens_per_rank": 16,
                    "num_local_experts": 2,
                    "use_fp8": False,
                }
            ],
        )
        replay_handle = buffer.dispatch_calls[-1]["handle"]
        self.assertIs(replay_handle[7], replay_config)
        self.assertEqual(buffer.dispatch_calls[-1]["pad_multiple"], FP8_ALIGN)
        self.assertFalse(buffer.dispatch_calls[-1]["non_blocking"])

    def test_combine_bridge_replays_dispatch_in_backward(self):
        ctx = _PyLayerContext()
        combined = paddle.ones([2, 4], dtype="float32")
        replay_config = SimpleNamespace(token_data_type="UINT16")
        buffer = _RecordingHybridEPBuffer(
            combine_results=[(combined, None)],
            dispatch_results=[
                (
                    paddle.full([2, 4], 2.0, dtype="float32"),
                    None,
                    None,
                    None,
                    None,
                )
            ],
            replay_config=replay_config,
        )
        handle = _make_hybrid_ep_handle(
            tokens_per_rank=8,
            token_data_type="UINT8",
            num_experts_per_rank=2,
        )
        manager = SimpleNamespace(handle=handle, _buffer=buffer)

        result = HybridEPCombine.forward(
            ctx,
            paddle.zeros([2, 4], dtype="float32"),
            manager,
        )

        self.assertIs(result, combined)
        self.assertFalse(result.stop_gradient)
        self.assertIs(ctx.buffer, buffer)
        self.assertIs(ctx.handle, handle)
        self.assertTrue(ctx.use_fp8_dispatch)
        self.assertEqual(ctx.num_permuted_tokens, 2)

        grad_x = HybridEPCombine.backward(
            ctx, paddle.ones([2, 4], dtype="float32")
        )

        self.assertEqual(grad_x.numpy().tolist(), [[2.0] * 4, [2.0] * 4])
        self.assertEqual(buffer.dispatch_calls[-1]["pad_multiple"], FP8_ALIGN)


class TestHybridEPCombineContract(unittest.TestCase):
    def test_manager_combine_rejects_overlap(self):
        manager = _HybridEPManager.__new__(_HybridEPManager)
        with self.assertRaisesRegex(NotImplementedError, "combine overlap"):
            manager.combine(paddle.zeros([1, 4]), {"fn": object()})

    def test_manager_combine_clears_runtime_state(self):
        manager = _HybridEPManager.__new__(_HybridEPManager)
        manager.dispatched_probs = object()
        manager.handle = _make_hybrid_ep_handle(token_data_type="BF16")
        combined = paddle.ones([1, 4], dtype="float32")
        manager._buffer = _RecordingHybridEPBuffer(
            combine_results=[(combined, None)]
        )
        hidden = paddle.zeros([1, 4])

        result = manager.combine(hidden)

        self.assertTrue(paddle.equal_all(result, combined).item())
        self.assertIsNone(manager.dispatched_probs)
        self.assertIsNone(manager.handle)
        self.assertIs(manager._buffer.combine_calls[-1]["hidden"], hidden)


class TestHybridEPMoeFusionContract(unittest.TestCase):
    def test_prepare_expert_counts_matches_expert_compute_contract(self):
        manager = SimpleNamespace(
            padded_tokens_per_expert=paddle.to_tensor([2, 0, 1], dtype="int32")
        )
        custom_map = SimpleNamespace(
            token_dispatcher=SimpleNamespace(_comm_manager=manager)
        )

        counts, num_tokens = _hybrid_ep_prepare_expert_counts(
            custom_map,
            use_fp8_mlp=False,
            moe_grouped_gemm=False,
        )

        self.assertEqual(counts, [2, 0, 1])
        self.assertEqual(num_tokens, 3)

        manager.padded_tokens_per_expert = paddle.to_tensor(
            [4, 2], dtype="int32"
        )
        counts, num_tokens = _hybrid_ep_prepare_expert_counts(
            custom_map,
            use_fp8_mlp=True,
            moe_grouped_gemm=True,
        )

        self.assertIsInstance(counts, paddle.Tensor)
        self.assertEqual(counts.dtype, paddle.int64)
        self.assertEqual(counts.numpy().tolist(), [4, 2])
        self.assertEqual(int(num_tokens.item()), 6)

    def test_prepare_expert_counts_requires_hybrid_ep_counts(self):
        custom_map = SimpleNamespace(
            token_dispatcher=SimpleNamespace(
                _comm_manager=SimpleNamespace(padded_tokens_per_expert=None)
            )
        )

        with self.assertRaisesRegex(AssertionError, "padded_tokens_per_expert"):
            _hybrid_ep_prepare_expert_counts(
                custom_map,
                use_fp8_mlp=False,
                moe_grouped_gemm=False,
            )

    def test_pad_and_prob_grad_restore_shape(self):
        tensor = paddle.ones([2, 3], dtype="float32")
        self.assertIs(_pad_front_rows(tensor, (2, 3)), tensor)

        padded = _pad_front_rows(tensor, (4, 3))
        self.assertEqual(padded.shape, [4, 3])
        self.assertEqual(
            padded.numpy().tolist(),
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        )

        restored = _restore_hybrid_ep_prob_grad_shape(
            paddle.to_tensor([[0.25], [0.5]], dtype="float32"),
            (4,),
        )
        self.assertEqual(restored.shape, [4])
        self.assertEqual(restored.numpy().tolist(), [0.25, 0.5, 0.0, 0.0])

        with self.assertRaisesRegex(AssertionError, "expected to stay 1D"):
            _restore_hybrid_ep_prob_grad_shape(
                paddle.ones([2], dtype="float32"),
                (2, 1),
            )
        with self.assertRaisesRegex(AssertionError, "normalize back to 1D"):
            _restore_hybrid_ep_prob_grad_shape(
                paddle.ones([2, 2], dtype="float32"),
                (2,),
            )

    def test_hybrid_ep_moe_pylayer_slices_padded_rows_and_restores_grads(self):
        class FakeExpertsNode:
            instance = None

            def __init__(self, custom_map, **kwargs):
                self.custom_map = custom_map
                self.kwargs = kwargs
                self.clear_count = 0
                FakeExpertsNode.instance = self

            def forward(
                self, hidden_states, probs, tokens_per_expert, scale=None
            ):
                self.forward_hidden_shape = hidden_states.shape
                self.forward_probs_shape = probs.shape
                self.forward_tokens_per_expert = tokens_per_expert
                self.forward_scale = scale
                return hidden_states + probs.unsqueeze(-1)

            def cached_tensors(self):
                return [paddle.to_tensor([1], dtype="int64")]

            def clear_cached_tensors(self):
                self.clear_count += 1

            def set_cached_tensors(self, tensors):
                self.restored_tensors = tensors

            def backward(self, output_grad, dispatched_probs):
                self.backward_output_shape = output_grad.shape
                self.backward_probs_shape = dispatched_probs.shape
                return (
                    paddle.full([3, 4], 2.0, dtype="float32"),
                    paddle.ones([3, 1], dtype="float32"),
                )

            def reset_state(self):
                self.reset_called = True

        manager = SimpleNamespace(
            padded_tokens_per_expert=paddle.to_tensor([2, 1], dtype="int64")
        )
        custom_map = SimpleNamespace(
            token_dispatcher=SimpleNamespace(_comm_manager=manager)
        )
        ctx = _PyLayerContext()

        with patch(
            "paddlefleet.transformer.moe.fusion_layer_utils.ExpertsGroupGemmContiguousNode",
            FakeExpertsNode,
        ):
            out = HybridEPMoePyLayer.forward(
                ctx,
                paddle.ones([4, 4], dtype="float32"),
                paddle.arange(4, dtype="float32"),
                custom_map,
                use_fp8_mlp=False,
                moe_deep_gemm=False,
                moe_grouped_gemm=False,
                recompute_moe_gate_up=True,
                use_bf16_gemm_weight_grad=True,
                is_first_fwd=True,
            )
            hidden_grad, probs_grad = HybridEPMoePyLayer.backward(
                ctx, paddle.ones_like(out)
            )

        node = FakeExpertsNode.instance
        self.assertEqual(out.shape, [3, 4])
        self.assertEqual(node.forward_hidden_shape, [3, 4])
        self.assertEqual(node.forward_probs_shape, [3])
        self.assertEqual(node.forward_tokens_per_expert, [2, 1])
        self.assertIsNone(node.forward_scale)
        self.assertTrue(node.kwargs["recompute_moe_gate_up"])
        self.assertTrue(node.kwargs["use_bf16_gemm_weight_grad"])
        self.assertEqual(node.clear_count, 2)
        self.assertEqual(node.backward_output_shape, [3, 4])
        self.assertEqual(node.backward_probs_shape, [3])
        self.assertTrue(node.reset_called)
        self.assertEqual(hidden_grad.shape, [4, 4])
        self.assertEqual(probs_grad.shape, [4])
        self.assertEqual(hidden_grad[-1].numpy().tolist(), [0.0] * 4)
        self.assertEqual(probs_grad.numpy().tolist(), [1.0, 1.0, 1.0, 0.0])


class TestHybridEPMoELayerContract(unittest.TestCase):
    def test_dispatch_preprocess_keeps_manager_topk_outputs(self):
        layer = MoELayer.__new__(MoELayer)
        layer.use_latent_moe = False
        dispatcher = MoEFlexTokenDispatcher.__new__(MoEFlexTokenDispatcher)
        token_probs = paddle.ones([6, 2], dtype="float32")
        token_indices = paddle.zeros([6, 2], dtype="int64")
        flattened = paddle.zeros([6, 4], dtype="float32")
        preprocess_call = {}

        def dispatch_preprocess_overlap(*args):
            preprocess_call["args"] = args
            return flattened

        dispatcher.dispatch_preprocess_overlap = dispatch_preprocess_overlap
        dispatcher._comm_manager = SimpleNamespace(
            token_probs=token_probs,
            token_indices=token_indices,
        )
        layer.token_dispatcher = dispatcher
        hidden_states = paddle.zeros([2, 3, 4], dtype="float32")

        result = layer.dispatch_preprocess(
            (hidden_states, token_probs, token_indices)
        )

        self.assertIs(result[0], flattened)
        self.assertIs(result[1], token_indices)
        self.assertIs(result[2], token_probs)
        preprocess_args = preprocess_call["args"]
        self.assertIs(preprocess_args[0], hidden_states)
        self.assertIs(preprocess_args[1], token_probs)
        self.assertIs(preprocess_args[2], token_indices)

    def test_compute_dispatch_omits_indices_for_hybrid_ep_fusion(self):
        layer = MoELayer.__new__(MoELayer)
        layer.moe_use_fusion_node = True
        layer.fp8_dispatch = True
        layer.use_hybrid_ep_backend = True
        dispatched_hidden = paddle.ones([4, 8], dtype="float32")
        dispatched_probs = paddle.ones([4], dtype="float32")
        fp8_handle = {"scale": paddle.ones([4, 1], dtype="float32")}
        tokens_per_expert = paddle.to_tensor([2, 2], dtype="int64")
        dispatcher = MoEFlexTokenDispatcher.__new__(MoEFlexTokenDispatcher)

        def token_dispatch_overlap(*args, **kwargs):
            dispatcher.dispatch_args = args
            dispatcher.dispatch_kwargs = kwargs
            return dispatched_hidden, fp8_handle

        dispatcher.token_dispatch_overlap = token_dispatch_overlap
        dispatcher._comm_manager = SimpleNamespace(
            dispatched_probs=dispatched_probs,
            tokens_per_expert=tokens_per_expert,
            dispatched_indices=paddle.zeros([4, 1], dtype="int64"),
        )
        layer.token_dispatcher = dispatcher
        hidden_states = paddle.zeros([2, 8], dtype="float32")
        token_indices = paddle.zeros([2, 2], dtype="int64")
        token_weights = paddle.ones([2, 2], dtype="float32")

        (
            guarded_hidden,
            dispatched_indices,
            returned_probs,
            returned_fp8_handle,
            returned_tokens_per_expert,
            guard_status,
        ) = layer.compute_dispatch(
            (hidden_states, token_indices, token_weights),
            async_finish=True,
        )

        self.assertEqual(guarded_hidden.shape, [0])
        self.assertIsNone(dispatched_indices)
        self.assertIs(returned_probs, dispatched_probs)
        self.assertIs(returned_fp8_handle, fp8_handle)
        self.assertIs(returned_tokens_per_expert, tokens_per_expert)
        self.assertIs(guard_status["x"], dispatched_hidden)
        self.assertTrue(guard_status["x"].stop_gradient)
        dispatch_args = dispatcher.dispatch_args
        dispatch_kwargs = dispatcher.dispatch_kwargs
        self.assertIs(dispatch_args[0], hidden_states)
        self.assertIs(dispatch_args[1], token_indices)
        self.assertIs(dispatch_args[2], token_weights)
        self.assertTrue(dispatch_args[3])
        self.assertTrue(dispatch_kwargs["async_finish"])

    def test_compute_experts_uses_hybrid_ep_fusion_output(self):
        layer = MoELayer.__new__(MoELayer)
        layer.moe_use_fusion_node = True
        layer.use_hybrid_ep_backend = True
        fused_out = paddle.ones([4, 8], dtype="float32")
        fusion_call = {}

        def run_hybrid_ep_fusion(*args, **kwargs):
            fusion_call["args"] = args
            fusion_call["kwargs"] = kwargs
            return fused_out

        layer._run_hybrid_ep_fusion = run_hybrid_ep_fusion
        tokens_per_expert = paddle.to_tensor([2, 2], dtype="int64")
        comm_manager = SimpleNamespace(tokens_per_expert=None)
        layer.token_dispatcher = SimpleNamespace(_comm_manager=comm_manager)
        original_hidden = paddle.ones([4, 8], dtype="float32")
        guard_status = {"x": original_hidden}
        dispatched_probs = paddle.ones([4], dtype="float32")
        fp8_handle = {"scale": paddle.ones([4, 1], dtype="float32")}

        result = layer.compute_experts(
            (
                paddle.empty([0], dtype="float32"),
                None,
                dispatched_probs,
                fp8_handle,
                tokens_per_expert,
                guard_status,
            ),
            is_first_fwd=True,
        )

        self.assertIs(result, fused_out)
        self.assertIs(comm_manager.tokens_per_expert, tokens_per_expert)
        self.assertIs(fusion_call["args"][0], original_hidden)
        self.assertIs(fusion_call["args"][1], dispatched_probs)
        self.assertEqual(
            fusion_call["kwargs"],
            {"fp8_dispatched_handle": fp8_handle, "is_first_fwd": True},
        )

    def test_hybrid_ep_backend_disables_shared_expert_overlap(self):
        config = _make_moe_config()
        pg_collection = _make_pg_collection()

        with (
            patch(
                "paddlefleet.transformer.moe.moe_layer.utils.get_pg_size",
                return_value=2,
            ),
            patch(
                "paddlefleet.transformer.moe.moe_layer.utils.get_pg_rank",
                return_value=0,
            ),
            patch(
                "paddlefleet.transformer.moe.moe_layer.paddlefleet.ops.is_sonic_moe_available",
                return_value=False,
            ),
            patch(
                "paddlefleet.transformer.moe.moe_layer.paddle.version.cuda",
                return_value="12.2",
            ),
            patch(
                "paddlefleet.transformer.moe.moe_layer.paddle.is_compiled_with_cuda",
                return_value=False,
            ),
            patch.object(token_dispatcher, "HAVE_HYBRID_EP", True),
            patch(
                "paddlefleet.transformer.moe.moe_layer.TopKRouter",
                return_value=paddle.nn.Layer(),
            ),
            patch(
                "paddlefleet.transformer.moe.moe_layer.StandardMLPExpert",
                return_value=paddle.nn.Layer(),
            ),
            patch(
                "paddlefleet.transformer.moe.moe_layer.StandardMLPSharedExpert",
                return_value=paddle.nn.Layer(),
            ),
            patch(
                "paddlefleet.transformer.moe.moe_layer.MoEFlexTokenDispatcher",
                return_value=MagicMock(),
            ) as mock_dispatcher,
        ):
            layer = MoELayer(
                config,
                sublayers=MoESublayers(),
                pg_collection=pg_collection,
            )

        self.assertFalse(layer.moe_shared_expert_overlap)
        self.assertEqual(
            mock_dispatcher.call_args.kwargs["dispatcher_type"], "hybridep"
        )


if __name__ == "__main__":
    unittest.main()
