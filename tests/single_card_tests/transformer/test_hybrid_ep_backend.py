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
from unittest.mock import MagicMock, patch

import paddle

from paddlefleet.transformer.moe import token_dispatcher
from paddlefleet.transformer.moe.fused_a2a import (
    HybridEPCombine,
    hybrid_ep_combine,
)
from paddlefleet.transformer.moe.moe_layer import MoELayer, MoESublayers
from paddlefleet.transformer.moe.token_dispatcher import (
    MoEFlexTokenDispatcher,
    _HybridEPManager,
    get_selected_deep_ep_backend_name,
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
        "moe_token_dispatcher_type": "deepep",
        "moe_flex_dispatcher_backend": "hybridep",
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
    pg = MagicMock()
    pg.ep = MagicMock()
    pg.ep.world_size = moe_world_size
    pg.expt_dp = MagicMock()
    pg.tp = MagicMock()
    pg.tp.size.return_value = 1
    pg.cp = MagicMock()
    pg.cp.rank.return_value = 0
    pg.cp.size.return_value = 1
    return pg


class TestHybridEPBackendSelection(unittest.TestCase):
    def test_backend_defaults_to_deepep_and_ignores_env(self):
        with (
            patch.dict(
                os.environ,
                {
                    "PF_DEEP_EP_BACKEND": "hybrid",
                    "PADDLEFLEET_DEEP_EP_BACKEND": "hybrid",
                },
            ),
            patch.object(token_dispatcher, "HAVE_HYBRID_EP", True),
        ):
            self.assertEqual(get_selected_deep_ep_backend_name(), "deepep")

    def test_backend_selects_hybrid_explicitly(self):
        with patch.object(token_dispatcher, "HAVE_HYBRID_EP", True):
            self.assertEqual(
                get_selected_deep_ep_backend_name("hybridep"), "hybrid"
            )
            self.assertTrue(is_hybrid_ep_backend_selected("hybrid_ep"))

    def test_backend_deep_ep_v2_is_reserved(self):
        with self.assertRaisesRegex(NotImplementedError, "deep_ep_v2"):
            get_selected_deep_ep_backend_name("deep_ep_v2")

    def test_flex_dispatcher_uses_hybrid_backend(self):
        mock_group = MagicMock()
        mock_group.world_size = 2
        mock_manager_cls = MagicMock()

        with (
            patch.object(token_dispatcher, "HAVE_HYBRID_EP", True),
            patch.object(
                token_dispatcher, "_HybridEPManager", mock_manager_cls
            ),
        ):
            dispatcher = MoEFlexTokenDispatcher(
                num_local_experts=2,
                num_experts_per_tok=2,
                n_routed_experts=4,
                ep_group=mock_group,
                backend_name="hybridep",
                needs_host_counts=True,
            )

        self.assertIs(dispatcher._comm_manager, mock_manager_cls.return_value)
        self.assertEqual(
            mock_manager_cls.call_args.kwargs["needs_host_counts"], True
        )


class TestHybridEPCombineContract(unittest.TestCase):
    def test_manager_combine_rejects_overlap(self):
        manager = _HybridEPManager.__new__(_HybridEPManager)
        with self.assertRaisesRegex(NotImplementedError, "combine overlap"):
            manager.combine(paddle.zeros([1, 4]), {"fn": MagicMock()})

    def test_manager_combine_clears_runtime_state(self):
        manager = _HybridEPManager.__new__(_HybridEPManager)
        manager.dispatched_probs = object()
        manager.handle = object()
        hidden = paddle.zeros([1, 4])
        combined = object()
        with patch.object(
            token_dispatcher, "hybrid_ep_combine", return_value=combined
        ) as mock_combine:
            result = manager.combine(hidden)

        self.assertIs(result, combined)
        self.assertIsNone(manager.dispatched_probs)
        self.assertIsNone(manager.handle)
        mock_combine.assert_called_once_with(hidden, manager)

    def test_hybrid_ep_combine_uses_sync_pylayer(self):
        x = paddle.randn([4, 64], dtype=paddle.float32)
        manager = MagicMock()
        combined = object()
        with patch.object(
            HybridEPCombine, "apply", return_value=combined
        ) as mock_apply:
            result = hybrid_ep_combine(x, manager)

        self.assertIs(result, combined)
        mock_apply.assert_called_once_with(x, manager)

    def test_hybrid_ep_combine_has_no_overlap_argument(self):
        with self.assertRaises(TypeError):
            hybrid_ep_combine(
                paddle.randn([4, 64], dtype=paddle.float32),
                MagicMock(),
                {"fn": MagicMock()},
            )


class TestHybridEPMoELayerContract(unittest.TestCase):
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
            patch(
                "paddlefleet.transformer.moe.moe_layer.is_hybrid_ep_backend_selected",
                return_value=True,
            ) as mock_is_hybrid,
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
        mock_is_hybrid.assert_called_once_with("hybridep")
        self.assertEqual(
            mock_dispatcher.call_args.kwargs["backend_name"], "hybridep"
        )


if __name__ == "__main__":
    unittest.main()
