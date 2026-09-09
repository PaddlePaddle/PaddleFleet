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
import sys
import unittest
from unittest.mock import patch

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import numpy as np
import paddle

from paddlefleet.transformer.moe.moe_layer import MoELayer


class MinimalMoE:
    _use_grouped_mlp_expert = False


class Expert:
    def __init__(self, offset):
        self.offset = offset
        self.inputs = []

    def __call__(self, x):
        self.inputs.append(x)
        return x + self.offset, None


class Gate:
    def __init__(self):
        self.calls = []
        self.layer_number = None

    def __call__(self, hidden_states, input_ids=None, origin_input_ids=None):
        self.calls.append((hidden_states, input_ids, origin_input_ids))
        return "gate-output"

    def set_layer_number(self, layer_number, is_mtp_layer=False):
        self.layer_number = layer_number


class Combiner:
    def __init__(self):
        self.calls = []

    def combine(
        self,
        hidden_states,
        handle,
        async_finish=False,
        use_rr_deepep_combine=False,
    ):
        self.calls.append(
            (hidden_states, handle, async_finish, use_rr_deepep_combine)
        )
        return hidden_states + 3


class Dispatcher:
    def __init__(self):
        self._comm_manager = Combiner()


class SharedExpert:
    def __call__(self, residuals):
        return residuals + 5, None


class TestMoELayerLightweightMethods(unittest.TestCase):
    def test_expert_forward_concatenates_non_empty_expert_outputs(self):
        model = MinimalMoE()
        model.moe_rank = 0
        model.num_experts_per_device = 3
        model.experts = [Expert(1), Expert(10), Expert(100)]
        dispatched_input = paddle.arange(8, dtype="float32").reshape([4, 2])

        output = MoELayer.expert_forward(
            model, dispatched_input, paddle.to_tensor([1, 0, 3], dtype="int64")
        )

        self.assertEqual(
            output.numpy().tolist(),
            [[1.0, 2.0], [102.0, 103.0], [104.0, 105.0], [106.0, 107.0]],
        )
        self.assertEqual(model.experts[0].inputs[0].shape, [1, 2])
        self.assertEqual(model.experts[2].inputs[0].shape, [3, 2])
        self.assertEqual(model.experts[1].inputs, [])

    def test_expert_forward_returns_input_when_all_sections_are_empty(self):
        model = MinimalMoE()
        model.moe_rank = 0
        model.num_experts_per_device = 1
        model.experts = [Expert(1)]
        dispatched_input = paddle.empty([0, 2], dtype="float32")

        output = MoELayer.expert_forward(model, dispatched_input, [0])

        self.assertIs(output, dispatched_input)
        self.assertEqual(model.experts[0].inputs, [])

    def test_expert_forward_requires_router_probs_with_accuracy_kernel(self):
        model = MinimalMoE()
        model.moe_rank = 0
        model.num_experts_per_device = 1
        model.use_accuracy_compatible = True
        model.token_dispatcher = type("Dispatcher", (), {})()
        model.experts = [Expert(1)]
        dispatched_input = paddle.ones([1, 2], dtype="float32")

        with (
            patch(
                "paddlefleet.transformer.moe.moe_layer.use_accuracy_compatible_kernel",
                return_value=True,
            ),
            self.assertRaisesRegex(
                RuntimeError, "requires dispatched router probabilities"
            ),
        ):
            MoELayer.expert_forward(model, dispatched_input, [1])

    def test_expert_forward_tiny_m_padding_preserves_backward(self):
        class TwiceExpert:
            def __init__(self):
                self.inputs = []

            def __call__(self, x):
                self.inputs.append(x)
                return x * 2, None

        model = MinimalMoE()
        model.moe_rank = 0
        model.num_experts_per_device = 2
        model.use_accuracy_compatible = True
        model.token_dispatcher = type("Dispatcher", (), {})()
        model.token_dispatcher.global_input_probs = None
        model.experts = [TwiceExpert(), TwiceExpert()]

        dispatched_input = paddle.arange(8, dtype="float32").reshape([2, 4])
        dispatched_input.stop_gradient = False
        output = MoELayer.expert_forward(
            model,
            dispatched_input,
            paddle.to_tensor([1, 1], dtype="int64"),
        )

        self.assertEqual([x.shape[0] for x in model.experts[0].inputs], [32])
        self.assertEqual([x.shape[0] for x in model.experts[1].inputs], [32])
        np.testing.assert_allclose(output.numpy(), dispatched_input.numpy() * 2)
        output.sum().backward()
        np.testing.assert_allclose(dispatched_input.grad.numpy(), 2.0)

    def test_expert_forward_tiny_m_padding_preserves_router_scale(self):
        class ScaledExpert:
            def __init__(self):
                self.scales = []

            def __call__(self, x, per_token_scale):
                self.scales.append(per_token_scale)
                return x * per_token_scale.unsqueeze(-1), None

        model = MinimalMoE()
        model.moe_rank = 0
        model.num_experts_per_device = 1
        model.use_accuracy_compatible = True
        model.token_dispatcher = type("Dispatcher", (), {})()
        model.token_dispatcher.global_input_probs = paddle.to_tensor(
            [0.25, 0.75], dtype="float32"
        )
        model.experts = [ScaledExpert()]
        dispatched_input = paddle.ones([2, 3], dtype="float32")

        with patch(
            "paddlefleet.transformer.moe.moe_layer.use_accuracy_compatible_kernel",
            return_value=True,
        ):
            output = MoELayer.expert_forward(model, dispatched_input, [2])

        self.assertEqual(model.experts[0].scales[0].shape, [32])
        np.testing.assert_allclose(
            model.experts[0].scales[0][:2].numpy(), [0.25, 0.75]
        )
        np.testing.assert_allclose(model.experts[0].scales[0][2:].numpy(), 0.0)
        np.testing.assert_allclose(
            output.numpy(), [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]]
        )

    def test_accuracy_fusion_forward_populates_overlap_output(self):
        model = MinimalMoE()
        model.use_accuracy_compatible = True
        hidden_states = paddle.ones([2, 3], dtype="float32")
        shared_calls = []

        def shared_expert(x):
            shared_calls.append(x)
            return x + 2.0, None

        model.custom_forward = lambda *args, **kwargs: hidden_states + 1.0
        overlap_handle = {
            "fn": shared_expert,
            "fn_args": (hidden_states,),
        }

        output = MoELayer.fusion_moe_forward(
            model,
            hidden_states,
            probs=None,
            routing_map=None,
            combine_overlap_handle=overlap_handle,
        )

        self.assertEqual(len(shared_calls), 1)
        self.assertEqual(
            output.numpy().tolist(), (hidden_states + 1.0).numpy().tolist()
        )
        self.assertEqual(
            overlap_handle["fn_out"][0].numpy().tolist(),
            (hidden_states + 2.0).numpy().tolist(),
        )

    def test_compute_gate_and_hybrid_fusion_predicate(self):
        model = MinimalMoE()
        model.expert_model_parallel_size = 2
        model.sequence_parallel = True
        model.gate = Gate()
        model.moe_use_fusion_node = True
        model.use_hybrid_ep_backend = False
        hidden_states = paddle.ones([2, 2], dtype="float32")
        input_ids = paddle.ones([2], dtype="int64")

        self.assertEqual(
            MoELayer.compute_gate(model, hidden_states, input_ids=input_ids),
            "gate-output",
        )
        self.assertIs(model.gate.calls[0][0], hidden_states)
        self.assertIs(model.gate.calls[0][1], input_ids)
        self.assertFalse(MoELayer._use_hybrid_ep_fusion(model))

        model.use_hybrid_ep_backend = True
        self.assertTrue(MoELayer._use_hybrid_ep_fusion(model))

    def test_compute_combine_uses_fusion_or_regular_path(self):
        model = MinimalMoE()
        hidden_states = paddle.ones([2], dtype="float32")
        model.moe_use_fusion_node = True
        model.token_dispatcher = Dispatcher()

        output = MoELayer.compute_combine(
            model, hidden_states, async_finish=True
        )

        self.assertEqual(output.numpy().tolist(), [4.0, 4.0])
        self.assertEqual(model.token_dispatcher._comm_manager.calls[0][2], True)

        model.moe_use_fusion_node = False
        model.combine = lambda value, *args, **kwargs: value + 7
        output = MoELayer.compute_combine(model, hidden_states)
        self.assertEqual(output.numpy().tolist(), [8.0, 8.0])

    def test_aux_loss_compute_reshapes_and_adds_shared_expert(self):
        model = MinimalMoE()
        model.use_latent_moe = False
        model.training = False
        model.router_aux_loss_coef = 0.0
        model.shared_experts = SharedExpert()
        model.expert_model_parallel_size = 2
        model.sequence_parallel = True
        hidden_states = paddle.ones([4, 2], dtype="float32")
        residuals = paddle.zeros([2, 2, 2], dtype="float32")

        output = MoELayer.aux_loss_compute(
            model, (hidden_states, paddle.to_tensor([1.0]), None, residuals)
        )

        self.assertEqual(output.shape, [2, 2, 2])
        self.assertEqual(output.numpy().tolist()[0][0], [6.0, 6.0])

    def test_use_fp8_and_set_layer_number(self):
        model = MinimalMoE()
        model.moe_use_fusion_node = False
        model.fp8 = True
        self.assertFalse(MoELayer.use_fp8(model))

        model.moe_use_fusion_node = True
        self.assertTrue(MoELayer.use_fp8(model))

        model.gate = Gate()
        # MinimalMoE is not a MoELayer subclass, so stub out the hooks
        # set_layer_number calls but that are not under test here: expert param
        # coloring and the layer-scoped recompute re-resolve.
        model._color_expert_params = lambda: None
        model._update_layer_aware_recompute = lambda: None
        MoELayer.set_layer_number(model, 11)
        self.assertEqual(model.layer_number, 11)
        self.assertEqual(model.gate.layer_number, 11)

        model.gate = object()
        with self.assertRaises(AssertionError):
            MoELayer.set_layer_number(model, 12)


if __name__ == "__main__":
    unittest.main()
