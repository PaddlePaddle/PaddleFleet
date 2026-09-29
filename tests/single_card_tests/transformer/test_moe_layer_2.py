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

try:
    import numpy as np
    import paddle

    from paddlefleet.transformer.moe.moe_layer import MoELayer

    HAS_PADDLE = True
    IMPORT_ERROR = ""
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    HAS_PADDLE = False
    IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

MOE_LAYER_MODULE = "paddlefleet.transformer.moe.moe_layer"


class _StubMoE:
    """Minimal stand-in that carries only the attributes the unbound
    ``MoELayer`` methods under test actually read.

    The real ``MoELayer.<method>`` code runs against this stub; only the
    genuine, not-under-test collaborators (experts / gate / dispatcher /
    combiner) are lightweight doubles with distinguishable behaviour.
    """

    _use_grouped_mlp_expert = False


class _AddExpert:
    """Expert that returns ``x + offset`` and records the rows it saw."""

    def __init__(self, offset):
        self.offset = offset
        self.inputs = []

    def __call__(self, x):
        self.inputs.append(x)
        return x + self.offset, None


class _RecordingGate:
    """Router double: records every call and its layer-number updates."""

    def __init__(self, ret="gate-output"):
        self.ret = ret
        self.calls = []
        self.layer_number = None
        self.is_mtp_layer = None

    def __call__(self, hidden_states, input_ids=None, origin_input_ids=None):
        self.calls.append((hidden_states, input_ids, origin_input_ids))
        return self.ret

    def set_layer_number(self, layer_number, is_mtp_layer=False):
        self.layer_number = layer_number
        self.is_mtp_layer = is_mtp_layer


class _Combiner:
    """Deepep/hybridep comm-manager double: returns ``x + 3`` and records
    the positional handle / keyword flags it was invoked with."""

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
            {
                "hidden_states": hidden_states,
                "handle": handle,
                "async_finish": async_finish,
                "use_rr_deepep_combine": use_rr_deepep_combine,
            }
        )
        return hidden_states + 3


class _Dispatcher:
    def __init__(self):
        self._comm_manager = _Combiner()


class _SharedExpert:
    """Shared-expert double returning ``residuals + 5`` and recording input."""

    def __init__(self):
        self.inputs = []

    def __call__(self, residuals):
        self.inputs.append(residuals)
        return residuals + 5, None


@unittest.skipUnless(
    HAS_PADDLE,
    f"paddle / paddlefleet.transformer.moe.moe_layer unavailable: {IMPORT_ERROR}",
)
class TestMoELayerExpertForward(unittest.TestCase):
    def test_routes_each_section_to_its_expert_and_skips_empty(self):
        # tokens_per_expert = [1, 0, 3] -> expert 0 gets row 0, expert 1 gets
        # nothing (empty section skipped), expert 2 gets rows 1..3.
        model = _StubMoE()
        model.moe_rank = 0
        model.num_experts_per_device = 3
        experts = [_AddExpert(1.0), _AddExpert(10.0), _AddExpert(100.0)]
        model.experts = experts
        base = np.arange(8, dtype="float32").reshape([4, 2])
        dispatched = paddle.to_tensor(base)

        with patch(
            f"{MOE_LAYER_MODULE}.use_accuracy_compatible_kernel",
            return_value=False,
        ):
            out = MoELayer.expert_forward(
                model, dispatched, paddle.to_tensor([1, 0, 3], dtype="int64")
            )

        expected = np.concatenate([base[0:1] + 1.0, base[1:4] + 100.0], axis=0)
        np.testing.assert_array_equal(out.numpy(), expected)
        # token identity: expert 0 saw exactly row 0, expert 2 saw rows 1..3.
        self.assertEqual(len(experts[0].inputs), 1)
        np.testing.assert_array_equal(experts[0].inputs[0].numpy(), base[0:1])
        self.assertEqual(experts[1].inputs, [])
        self.assertEqual(len(experts[2].inputs), 1)
        np.testing.assert_array_equal(experts[2].inputs[0].numpy(), base[1:4])

    def test_moe_rank_offsets_local_expert_indices(self):
        # moe_rank=1, num_experts_per_device=2 -> local sections 0,1 map to
        # global experts 2,3. Swapping the offset would route to 0,1 and fail.
        model = _StubMoE()
        model.moe_rank = 1
        model.num_experts_per_device = 2
        experts = [
            _AddExpert(1.0),
            _AddExpert(10.0),
            _AddExpert(100.0),
            _AddExpert(1000.0),
        ]
        model.experts = experts
        base = np.arange(9, dtype="float32").reshape([3, 3])
        dispatched = paddle.to_tensor(base)

        with patch(
            f"{MOE_LAYER_MODULE}.use_accuracy_compatible_kernel",
            return_value=False,
        ):
            out = MoELayer.expert_forward(
                model, dispatched, paddle.to_tensor([2, 1], dtype="int64")
            )

        expected = np.concatenate(
            [base[0:2] + 100.0, base[2:3] + 1000.0], axis=0
        )
        np.testing.assert_array_equal(out.numpy(), expected)
        self.assertEqual(experts[0].inputs, [])
        self.assertEqual(experts[1].inputs, [])
        np.testing.assert_array_equal(experts[2].inputs[0].numpy(), base[0:2])
        np.testing.assert_array_equal(experts[3].inputs[0].numpy(), base[2:3])

    def test_returns_same_input_when_all_sections_empty(self):
        model = _StubMoE()
        model.moe_rank = 0
        model.num_experts_per_device = 1
        expert = _AddExpert(1.0)
        model.experts = [expert]
        dispatched = paddle.empty([0, 2], dtype="float32")

        with patch(
            f"{MOE_LAYER_MODULE}.use_accuracy_compatible_kernel",
            return_value=False,
        ):
            out = MoELayer.expert_forward(model, dispatched, [0])

        self.assertIs(out, dispatched)
        self.assertEqual(expert.inputs, [])

    def test_accuracy_kernel_requires_dispatched_probs(self):
        model = _StubMoE()
        model.moe_rank = 0
        model.num_experts_per_device = 1
        model.use_accuracy_compatible = True
        model.token_dispatcher = type("_D", (), {})()  # no global_input_probs
        model.experts = [_AddExpert(1.0)]
        dispatched = paddle.ones([1, 2], dtype="float32")

        with (
            patch(
                f"{MOE_LAYER_MODULE}.use_accuracy_compatible_kernel",
                return_value=True,
            ),
            self.assertRaisesRegex(RuntimeError, "requires dispatched"),
        ):
            MoELayer.expert_forward(model, dispatched, [1])

    def test_tiny_m_padding_preserves_value_and_backward(self):
        class _TwiceExpert:
            def __init__(self):
                self.inputs = []

            def __call__(self, x):
                self.inputs.append(x)
                return x * 2, None

        model = _StubMoE()
        model.moe_rank = 0
        model.num_experts_per_device = 2
        model.use_accuracy_compatible = True
        model.token_dispatcher = type("_D", (), {})()
        model.token_dispatcher.global_input_probs = None
        experts = [_TwiceExpert(), _TwiceExpert()]
        model.experts = experts

        base = np.arange(8, dtype="float32").reshape([2, 4])
        dispatched = paddle.to_tensor(base)
        dispatched.stop_gradient = False

        with patch(
            f"{MOE_LAYER_MODULE}.use_accuracy_compatible_kernel",
            return_value=False,
        ):
            out = MoELayer.expert_forward(
                model, dispatched, paddle.to_tensor([1, 1], dtype="int64")
            )

        # 1-row sections (0 < m < 17) are padded up to 32 rows before the
        # expert, then sliced back to the original row count.
        self.assertEqual(experts[0].inputs[0].shape[0], 32)
        self.assertEqual(experts[1].inputs[0].shape[0], 32)
        np.testing.assert_allclose(out.numpy(), base * 2)

        out.sum().backward()
        np.testing.assert_allclose(
            dispatched.grad.numpy(), np.full_like(base, 2.0)
        )

    def test_tiny_m_padding_pads_router_scale_with_zeros(self):
        class _ScaledExpert:
            def __init__(self):
                self.scales = []

            def __call__(self, x, per_token_scale):
                self.scales.append(per_token_scale)
                return x * per_token_scale.unsqueeze(-1), None

        model = _StubMoE()
        model.moe_rank = 0
        model.num_experts_per_device = 1
        model.use_accuracy_compatible = True
        model.token_dispatcher = type("_D", (), {})()
        model.token_dispatcher.global_input_probs = paddle.to_tensor(
            [0.25, 0.75], dtype="float32"
        )
        expert = _ScaledExpert()
        model.experts = [expert]
        dispatched = paddle.ones([2, 3], dtype="float32")

        with patch(
            f"{MOE_LAYER_MODULE}.use_accuracy_compatible_kernel",
            return_value=True,
        ):
            out = MoELayer.expert_forward(model, dispatched, [2])

        scale = expert.scales[0]
        self.assertEqual(scale.shape[0], 32)
        np.testing.assert_allclose(scale[:2].numpy(), [0.25, 0.75])
        np.testing.assert_allclose(scale[2:].numpy(), np.zeros(30))
        np.testing.assert_allclose(
            out.numpy(),
            np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]]),
        )


@unittest.skipUnless(
    HAS_PADDLE,
    f"paddle / paddlefleet.transformer.moe.moe_layer unavailable: {IMPORT_ERROR}",
)
class TestMoELayerFusionForward(unittest.TestCase):
    def test_accuracy_path_runs_overlap_fn_and_delegates_to_custom_forward(
        self,
    ):
        model = _StubMoE()
        model.use_accuracy_compatible = True
        hidden = paddle.ones([2, 3], dtype="float32")
        probs = paddle.to_tensor([0.1, 0.9], dtype="float32")
        routing_map = paddle.to_tensor([[1, 0], [0, 1]], dtype="int64")

        captured = {}

        def custom_forward(hs, p, rm, topk_weights=None, topk_indices=None):
            captured["hs"] = hs
            captured["probs"] = p
            captured["routing_map"] = rm
            return hs + 1.0

        model.custom_forward = custom_forward

        shared_inputs = []

        def shared_expert(x):
            shared_inputs.append(x)
            return x + 2.0, None

        handle = {"fn": shared_expert, "fn_args": (hidden,)}
        out = MoELayer.fusion_moe_forward(
            model,
            hidden,
            probs=probs,
            routing_map=routing_map,
            combine_overlap_handle=handle,
        )

        # custom_forward received exactly the forwarded tensors.
        self.assertIs(captured["hs"], hidden)
        self.assertIs(captured["probs"], probs)
        self.assertIs(captured["routing_map"], routing_map)
        np.testing.assert_array_equal(out.numpy(), (hidden + 1.0).numpy())
        # overlap fn ran once and its output was stashed as a tuple.
        self.assertEqual(len(shared_inputs), 1)
        self.assertIs(shared_inputs[0], hidden)
        self.assertEqual(len(handle["fn_out"]), 2)
        np.testing.assert_array_equal(
            handle["fn_out"][0].numpy(), (hidden + 2.0).numpy()
        )
        self.assertIsNone(handle["fn_out"][1])

    def test_accuracy_path_does_not_rerun_overlap_when_fn_out_present(self):
        model = _StubMoE()
        model.use_accuracy_compatible = True
        hidden = paddle.ones([2, 3], dtype="float32")
        model.custom_forward = lambda hs, p, rm, **kw: hs + 1.0

        shared_inputs = []

        def shared_expert(x):
            shared_inputs.append(x)
            return x + 2.0, None

        handle = {"fn": shared_expert, "fn_args": (hidden,), "fn_out": ("pre",)}
        out = MoELayer.fusion_moe_forward(
            model,
            hidden,
            probs=None,
            routing_map=None,
            combine_overlap_handle=handle,
        )

        self.assertEqual(shared_inputs, [])  # guard prevents re-running fn
        self.assertEqual(handle["fn_out"], ("pre",))  # left untouched
        np.testing.assert_array_equal(out.numpy(), (hidden + 1.0).numpy())

    def test_accuracy_path_without_overlap_handle(self):
        model = _StubMoE()
        model.use_accuracy_compatible = True
        hidden = paddle.ones([2, 3], dtype="float32")
        model.custom_forward = lambda hs, p, rm, **kw: hs + 7.0

        out = MoELayer.fusion_moe_forward(
            model,
            hidden,
            probs=None,
            routing_map=None,
            combine_overlap_handle=None,
        )
        np.testing.assert_array_equal(out.numpy(), (hidden + 7.0).numpy())


@unittest.skipUnless(
    HAS_PADDLE,
    f"paddle / paddlefleet.transformer.moe.moe_layer unavailable: {IMPORT_ERROR}",
)
class TestMoELayerComputeGate(unittest.TestCase):
    def test_no_gather_when_expert_parallel_gt_one(self):
        model = _StubMoE()
        model.expert_model_parallel_size = 2
        model.sequence_parallel = True
        model.gate = _RecordingGate()
        hidden = paddle.ones([2, 2], dtype="float32")
        input_ids = paddle.ones([2], dtype="int64")

        with patch(f"{MOE_LAYER_MODULE}.GatherOp") as gather_op:
            ret = MoELayer.compute_gate(model, hidden, input_ids=input_ids)

        gather_op.apply.assert_not_called()
        self.assertEqual(ret, "gate-output")
        self.assertEqual(len(model.gate.calls), 1)
        self.assertIs(model.gate.calls[0][0], hidden)  # ungathered tensor
        self.assertIs(model.gate.calls[0][1], input_ids)

    def test_gathers_before_gate_when_sp_and_no_expert_parallel(self):
        model = _StubMoE()
        model.expert_model_parallel_size = 1
        model.sequence_parallel = True
        model.gate = _RecordingGate()
        hidden = paddle.ones([2, 2], dtype="float32")
        gathered = paddle.arange(4, dtype="float32").reshape([2, 2])
        input_ids = paddle.ones([2], dtype="int64")

        with patch(f"{MOE_LAYER_MODULE}.GatherOp") as gather_op:
            gather_op.apply.return_value = gathered
            MoELayer.compute_gate(model, hidden, input_ids=input_ids)

        gather_op.apply.assert_called_once_with(hidden)
        # the gate consumes the *gathered* tensor, not the raw hidden states.
        self.assertIs(model.gate.calls[0][0], gathered)
        self.assertIs(model.gate.calls[0][1], input_ids)

    def test_hybrid_ep_fusion_requires_both_flags(self):
        model = _StubMoE()
        for fusion, hybrid, expected in [
            (True, True, True),
            (True, False, False),
            (False, True, False),
            (False, False, False),
        ]:
            model.moe_use_fusion_node = fusion
            model.use_hybrid_ep_backend = hybrid
            self.assertEqual(
                MoELayer._use_hybrid_ep_fusion(model),
                expected,
                msg=f"fusion={fusion} hybrid={hybrid}",
            )


@unittest.skipUnless(
    HAS_PADDLE,
    f"paddle / paddlefleet.transformer.moe.moe_layer unavailable: {IMPORT_ERROR}",
)
class TestMoELayerCombineAndAux(unittest.TestCase):
    def test_compute_combine_fusion_path_forwards_flags(self):
        model = _StubMoE()
        model.moe_use_fusion_node = True
        model.token_dispatcher = _Dispatcher()
        hidden = paddle.ones([2], dtype="float32")

        out = MoELayer.compute_combine(model, hidden, async_finish=True)

        np.testing.assert_array_equal(out.numpy(), [4.0, 4.0])
        call = model.token_dispatcher._comm_manager.calls[0]
        self.assertIsNone(call["handle"])
        self.assertTrue(call["async_finish"])
        self.assertIs(call["hidden_states"], hidden)

    def test_compute_combine_regular_path_delegates_to_combine(self):
        model = _StubMoE()
        model.moe_use_fusion_node = False
        seen = []

        def combine(value, *args, **kwargs):
            seen.append(value)
            return value + 7

        model.combine = combine
        hidden = paddle.ones([2], dtype="float32")

        out = MoELayer.compute_combine(model, hidden)

        np.testing.assert_array_equal(out.numpy(), [8.0, 8.0])
        self.assertIs(seen[0], hidden)

    def test_aux_loss_compute_reshapes_and_adds_shared_expert(self):
        model = _StubMoE()
        model.use_latent_moe = False
        model.training = False
        model.router_aux_loss_coef = 0.0
        shared = _SharedExpert()
        model.shared_experts = shared
        model.expert_model_parallel_size = 2  # > 1 -> no ScatterOp
        model.sequence_parallel = True

        hs_np = np.arange(8, dtype="float32").reshape([4, 2])
        res_np = (np.arange(8, dtype="float32") * 0.1).reshape([2, 2, 2])
        hidden = paddle.to_tensor(hs_np)
        residuals = paddle.to_tensor(res_np)

        out = MoELayer.aux_loss_compute(
            model, (hidden, paddle.to_tensor([1.0]), None, residuals)
        )

        # reshape must preserve element order; shared adds residuals + 5.
        expected = hs_np.reshape([2, 2, 2]) + (res_np + 5.0)
        self.assertEqual(out.shape, [2, 2, 2])
        np.testing.assert_allclose(out.numpy(), expected)
        self.assertIs(shared.inputs[0], residuals)


@unittest.skipUnless(
    HAS_PADDLE,
    f"paddle / paddlefleet.transformer.moe.moe_layer unavailable: {IMPORT_ERROR}",
)
class TestMoELayerFp8AndLayerNumber(unittest.TestCase):
    def test_use_fp8_requires_fusion_node_and_fp8(self):
        model = _StubMoE()
        for fusion, fp8, expected in [
            (True, True, True),
            (True, False, False),
            (False, True, False),
            (False, False, False),
        ]:
            model.moe_use_fusion_node = fusion
            model.fp8 = fp8
            self.assertEqual(
                MoELayer.use_fp8(model),
                expected,
                msg=f"fusion={fusion} fp8={fp8}",
            )

    def test_set_layer_number_propagates_to_gate_and_runs_hooks(self):
        model = _StubMoE()
        hook_calls = []
        model._color_expert_params = lambda: hook_calls.append("color")
        model._update_layer_aware_recompute = lambda: hook_calls.append(
            "recompute"
        )
        model.gate = _RecordingGate()

        MoELayer.set_layer_number(model, 11, is_mtp_layer=True)

        self.assertEqual(model.layer_number, 11)
        self.assertTrue(model.is_mtp_layer)
        self.assertEqual(model.gate.layer_number, 11)
        self.assertTrue(model.gate.is_mtp_layer)
        self.assertEqual(sorted(hook_calls), ["color", "recompute"])

    def test_set_layer_number_defaults_is_mtp_layer_false(self):
        model = _StubMoE()
        model._color_expert_params = lambda: None
        model._update_layer_aware_recompute = lambda: None
        model.gate = _RecordingGate()

        MoELayer.set_layer_number(model, 4)

        self.assertFalse(model.is_mtp_layer)
        self.assertEqual(model.gate.layer_number, 4)
        self.assertFalse(model.gate.is_mtp_layer)

    def test_set_layer_number_requires_gate_with_set_layer_number(self):
        model = _StubMoE()
        model._color_expert_params = lambda: None
        model._update_layer_aware_recompute = lambda: None
        model.gate = object()  # lacks set_layer_number

        with self.assertRaises(AssertionError):
            MoELayer.set_layer_number(model, 12)


if __name__ == "__main__":
    unittest.main()
