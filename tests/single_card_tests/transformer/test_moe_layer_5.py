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
from types import SimpleNamespace

import numpy as np

# Heavy stack (paddle + paddlefleet) is optional in a CPU-only checkout.  Guard
# ONLY the two dependency-missing errors so a genuine import regression (compile
# error, renamed symbol, broken kernel binding) still fails loudly instead of
# being silently skipped.
try:
    import paddle

    from paddlefleet.transformer.moe import moe_layer
    from paddlefleet.transformer.moe.moe_layer import MoELayer

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    paddle = None
    moe_layer = None
    MoELayer = None
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    ""
    if _IMPORT_ERROR is None
    else f"paddle/paddlefleet not importable in this environment: {_IMPORT_ERROR!r}"
)


def _make_offset_marker(offset):
    """Build a stand-in for a genuine sequence-parallel collaborator
    (``GatherOp`` / ``ScatterOp``).

    The replacement records every input and returns ``input + offset`` with a
    distinguishable, per-collaborator offset so a test can prove *which* tensor
    the production code forwarded downstream (not merely that a call happened).
    Real distributed gather/scatter is a multi-rank op and is deliberately NOT
    exercised in this single CPU process; these tests only verify the
    single-card branch selection and the value plumbing around it.
    """

    class _Marker:
        calls = []

        @staticmethod
        def apply(x):
            _Marker.calls.append(x)
            return x + offset

    return _Marker


class _MoEMethodTestBase(unittest.TestCase):
    """Drive real ``MoELayer`` methods against a lightweight attribute bag.

    The methods under test are unbound and read collaborators/flags off
    ``self``; calling them with a ``SimpleNamespace`` executes the genuine
    method body (branch logic, arg forwarding, reshape/cast arithmetic) while
    letting us inject distinguishable stand-ins for the non-under-test
    collaborators (distributed ops, fused GPU kernels).  We never assert on
    attributes we merely stuffed in -- only on what the production code does
    with them.
    """

    def _patch_module_attr(self, name, value):
        original = getattr(moe_layer, name)
        setattr(moe_layer, name, value)
        self.addCleanup(setattr, moe_layer, name, original)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestComputeGateGatherBranch(_MoEMethodTestBase):
    def _run(self, ep_size, sequence_parallel):
        gather = _make_offset_marker(100.0)
        self._patch_module_attr("GatherOp", gather)

        captured = {}

        def gate(hidden_states, input_ids=None, origin_input_ids=None):
            captured["hidden"] = hidden_states
            captured["input_ids"] = input_ids
            captured["origin_input_ids"] = origin_input_ids
            return "gate-result"

        fake_self = SimpleNamespace(
            expert_model_parallel_size=ep_size,
            sequence_parallel=sequence_parallel,
            gate=gate,
        )
        hidden = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype="float32")
        input_ids = paddle.to_tensor([[7, 8, 9]], dtype="int64")
        origin_ids = paddle.to_tensor([[7, 8, 9, 0]], dtype="int64")

        result = MoELayer.compute_gate(
            fake_self, hidden, input_ids=input_ids, origin_input_ids=origin_ids
        )
        self.assertEqual(result, "gate-result")
        # input_ids / origin_input_ids must be forwarded verbatim regardless of
        # the gather branch.
        self.assertIs(captured["input_ids"], input_ids)
        self.assertIs(captured["origin_input_ids"], origin_ids)
        return gather, captured, hidden

    def test_gathers_only_for_single_card_sequence_parallel(self):
        # ep<=1 AND sequence_parallel -> the gather runs and the gate consumes
        # the gathered tensor (original + 100), not the raw hidden state.
        gather, captured, hidden = self._run(ep_size=1, sequence_parallel=True)
        self.assertEqual(len(gather.calls), 1)
        np.testing.assert_array_equal(gather.calls[0].numpy(), hidden.numpy())
        np.testing.assert_array_equal(
            captured["hidden"].numpy(), (hidden + 100.0).numpy()
        )

    def test_no_gather_when_not_sequence_parallel(self):
        # sequence_parallel False -> gather skipped, gate sees the raw hidden.
        gather, captured, hidden = self._run(ep_size=1, sequence_parallel=False)
        self.assertEqual(gather.calls, [])
        self.assertIs(captured["hidden"], hidden)

    def test_no_gather_when_expert_parallel(self):
        # ep>1 disables the single-card gather even with sequence_parallel on.
        gather, captured, hidden = self._run(ep_size=2, sequence_parallel=True)
        self.assertEqual(gather.calls, [])
        self.assertIs(captured["hidden"], hidden)


class _RecordingAddAux:
    """Spy for ``AddAuxiliaryLoss`` (a genuine autograd trick collaborator).

    The real op is value-identity in forward (see ``moe_utils.py`` line 559:
    ``return x.clone()``); the aux loss only enters the backward graph.  The
    spy preserves that value contract (returns ``x``) while recording the
    ``(x, loss)`` pairs so a test can assert the scaling and ordering of the
    aux/z loss injection.
    """

    calls = []

    @staticmethod
    def apply(x, loss):
        _RecordingAddAux.calls.append((x, loss))
        return x


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestAuxLossCompute(_MoEMethodTestBase):
    def _base_self(self, **overrides):
        cfg = {
            "use_latent_moe": False,
            "training": True,
            "router_aux_loss_coef": 0.5,
            "shared_experts": None,
            "expert_model_parallel_size": 1,
            "sequence_parallel": False,
        }
        cfg.update(overrides)
        return SimpleNamespace(**cfg)

    def test_forward_value_reshapes_adds_shared_and_scatters(self):
        # Forward output = reshape(hidden, residual.shape) + shared_experts(res)
        # then ScatterOp (single-card sequence-parallel branch).  AddAuxiliaryLoss
        # is kept real: it is value-identity, so it must not perturb the result.
        scatter = _make_offset_marker(1000.0)
        self._patch_module_attr("ScatterOp", scatter)

        fake_self = self._base_self(
            shared_experts=lambda res: (res * 2.0,),
            sequence_parallel=True,
        )
        hidden = paddle.arange(8, dtype="float32").reshape([4, 2])
        residuals = paddle.arange(8, dtype="float32").reshape([2, 2, 2])
        aux_loss = paddle.to_tensor(2.0, dtype="float32")
        z_loss = paddle.to_tensor(3.0, dtype="float32")

        out = MoELayer.aux_loss_compute(
            fake_self, (hidden, aux_loss, z_loss, residuals)
        )

        # Independent numpy reference: reshape + 2*residuals + scatter offset.
        expected = (
            hidden.numpy().reshape(2, 2, 2) + residuals.numpy() * 2.0 + 1000.0
        )
        self.assertEqual(list(out.shape), [2, 2, 2])
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(len(scatter.calls), 1)
        # ScatterOp is fed the shared-augmented, reshaped tensor (pre-offset).
        np.testing.assert_allclose(
            scatter.calls[0].numpy(),
            hidden.numpy().reshape(2, 2, 2) + residuals.numpy() * 2.0,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_no_scatter_when_not_sequence_parallel(self):
        scatter = _make_offset_marker(1000.0)
        self._patch_module_attr("ScatterOp", scatter)

        fake_self = self._base_self(sequence_parallel=False)
        hidden = paddle.arange(8, dtype="float32").reshape([4, 2])
        residuals = paddle.zeros([2, 2, 2], dtype="float32")

        out = MoELayer.aux_loss_compute(
            fake_self,
            (
                hidden,
                paddle.to_tensor(1.0, dtype="float32"),
                paddle.to_tensor(1.0, dtype="float32"),
                residuals,
            ),
        )
        self.assertEqual(scatter.calls, [])
        np.testing.assert_allclose(
            out.numpy(), hidden.numpy().reshape(2, 2, 2), rtol=1e-6, atol=1e-6
        )

    def test_training_scales_aux_and_appends_zloss(self):
        # Observe the exact operands handed to AddAuxiliaryLoss: aux first,
        # scaled by router_aux_loss_coef, then z_loss on that output.
        _RecordingAddAux.calls = []
        self._patch_module_attr("AddAuxiliaryLoss", _RecordingAddAux)

        fake_self = self._base_self(router_aux_loss_coef=0.5)
        hidden = paddle.arange(8, dtype="float32").reshape([4, 2])
        residuals = paddle.zeros([2, 2, 2], dtype="float32")
        aux_loss = paddle.to_tensor(2.0, dtype="float32")
        z_loss = paddle.to_tensor(3.0, dtype="float32")

        MoELayer.aux_loss_compute(
            fake_self, (hidden, aux_loss, z_loss, residuals)
        )

        self.assertEqual(len(_RecordingAddAux.calls), 2)
        aux_x, aux_val = _RecordingAddAux.calls[0]
        self.assertIs(aux_x, hidden)
        self.assertAlmostEqual(float(aux_val), 1.0, places=6)  # 2.0 * 0.5
        z_x, z_val = _RecordingAddAux.calls[1]
        self.assertIs(z_x, hidden)  # spy returns x unchanged
        self.assertIs(z_val, z_loss)

    def test_inference_skips_aux_and_zloss(self):
        # training=False -> AddAuxiliaryLoss must not be invoked at all.
        _RecordingAddAux.calls = []
        self._patch_module_attr("AddAuxiliaryLoss", _RecordingAddAux)

        fake_self = self._base_self(training=False)
        hidden = paddle.arange(8, dtype="float32").reshape([4, 2])
        residuals = paddle.zeros([2, 2, 2], dtype="float32")

        out = MoELayer.aux_loss_compute(
            fake_self,
            (
                hidden,
                paddle.to_tensor(9.0, dtype="float32"),
                paddle.to_tensor(9.0, dtype="float32"),
                residuals,
            ),
        )
        self.assertEqual(_RecordingAddAux.calls, [])
        np.testing.assert_allclose(
            out.numpy(), hidden.numpy().reshape(2, 2, 2), rtol=1e-6, atol=1e-6
        )

    def test_zero_coef_skips_aux_but_keeps_zloss(self):
        _RecordingAddAux.calls = []
        self._patch_module_attr("AddAuxiliaryLoss", _RecordingAddAux)

        fake_self = self._base_self(router_aux_loss_coef=0.0)
        hidden = paddle.arange(8, dtype="float32").reshape([4, 2])
        residuals = paddle.zeros([2, 2, 2], dtype="float32")
        z_loss = paddle.to_tensor(3.0, dtype="float32")

        MoELayer.aux_loss_compute(
            fake_self,
            (hidden, paddle.to_tensor(2.0, dtype="float32"), z_loss, residuals),
        )
        # coef falsy -> aux branch skipped; only z_loss is injected.
        self.assertEqual(len(_RecordingAddAux.calls), 1)
        self.assertIs(_RecordingAddAux.calls[0][1], z_loss)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestExpertsAndCombineDispatch(_MoEMethodTestBase):
    def test_combine_fusion_branch_uses_comm_manager(self):
        # Fusion node -> combine is delegated to the dispatcher comm manager
        # with (hidden, None, async_finish=...); self.combine must be untouched.
        combine_calls = []

        def cm_combine(hidden_states, handle, async_finish=False):
            combine_calls.append((hidden_states, handle, async_finish))
            return hidden_states + 3.0

        self_combine_calls = []

        fake_self = SimpleNamespace(
            moe_use_fusion_node=True,
            token_dispatcher=SimpleNamespace(
                _comm_manager=SimpleNamespace(combine=cm_combine)
            ),
            combine=lambda *a, **k: self_combine_calls.append((a, k)),
        )
        hidden = paddle.ones([2, 3], dtype="float32")
        out = MoELayer.compute_combine(fake_self, hidden, async_finish=True)

        self.assertEqual(len(combine_calls), 1)
        passed_hidden, passed_handle, passed_async = combine_calls[0]
        self.assertIs(passed_hidden, hidden)
        self.assertIsNone(passed_handle)
        self.assertTrue(passed_async)
        self.assertEqual(self_combine_calls, [])
        np.testing.assert_array_equal(out.numpy(), (hidden + 3.0).numpy())

    def test_combine_dense_branch_uses_self_combine(self):
        cm_calls = []
        self_calls = []

        fake_self = SimpleNamespace(
            moe_use_fusion_node=False,
            token_dispatcher=SimpleNamespace(
                _comm_manager=SimpleNamespace(
                    combine=lambda *a, **k: cm_calls.append((a, k))
                )
            ),
            combine=lambda h: (self_calls.append(h), h + 7.0)[1],
        )
        hidden = paddle.ones([2, 3], dtype="float32")
        out = MoELayer.compute_combine(fake_self, hidden)

        self.assertEqual(cm_calls, [])
        self.assertEqual(len(self_calls), 1)
        self.assertIs(self_calls[0], hidden)
        np.testing.assert_array_equal(out.numpy(), (hidden + 7.0).numpy())

    def test_experts_dense_branch_routes_through_routed_experts_compute(self):
        # Non-fusion path unpacks (hidden, topk_weights) and runs
        # routed_experts_compute on the hidden state only.
        routed_calls = []

        fake_self = SimpleNamespace(
            moe_use_fusion_node=False,
            routed_experts_compute=lambda h: (
                routed_calls.append(h),
                h + 5.0,
            )[1],
        )
        hidden = paddle.ones([2, 3], dtype="float32")
        out = MoELayer.compute_experts(fake_self, (hidden, None))

        self.assertEqual(len(routed_calls), 1)
        self.assertIs(routed_calls[0], hidden)
        np.testing.assert_array_equal(out.numpy(), (hidden + 5.0).numpy())


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestSingleCardGroupedGemmSonic(_MoEMethodTestBase):
    def _run(self, fp8_value):
        gge_calls = []

        def grouped_gemm_experts(*args, **kwargs):
            gge_calls.append((args, kwargs))
            # Return float64 so the mandatory cast back to the input dtype is
            # observable; value is a distinguishable function of the input.
            return args[0].astype("float64") + 40.0

        fake_self = SimpleNamespace(
            using_sonic_moe=True,
            fp8=fp8_value,
            grouped_gemm_experts=grouped_gemm_experts,
            recompute_moe_gate_up=True,
        )
        hidden = paddle.ones([2, 3], dtype="float32")
        routing = paddle.to_tensor([[1, 1], [1, 1]], dtype="bool")
        probs = paddle.to_tensor([[0.7, 0.3], [0.4, 0.6]], dtype="float32")
        topk_indices = paddle.to_tensor([[0, 1], [1, 0]], dtype="int64")
        topk_weights = paddle.to_tensor(
            [[0.7, 0.3], [0.6, 0.4]], dtype="float32"
        )

        out = MoELayer._forward_single_card_grouped_gemm_moe(
            fake_self,
            hidden,
            routing,
            probs,
            topk_indices=topk_indices,
            topk_weights=topk_weights,
        )
        return out, gge_calls, hidden, topk_indices, topk_weights

    def test_sonic_branch_forwards_args_and_casts_back(self):
        out, gge_calls, hidden, topk_indices, topk_weights = self._run(
            fp8_value=None
        )
        self.assertEqual(len(gge_calls), 1)
        args, kwargs = gge_calls[0]
        self.assertIs(args[0], hidden)
        self.assertIs(args[1], topk_indices)
        self.assertIs(args[2], topk_weights)
        self.assertFalse(args[3])  # use_fp8 == (self.fp8 is not None)
        self.assertTrue(kwargs["recompute_moe_gate_up"])
        # Output must be cast back to the input (float32) dtype.
        self.assertEqual(out.dtype, hidden.dtype)
        np.testing.assert_allclose(
            out.numpy(), (hidden.numpy() + 40.0), rtol=1e-6, atol=1e-6
        )

    def test_sonic_branch_use_fp8_true_when_fp8_configured(self):
        out, gge_calls, hidden, _, _ = self._run(fp8_value=object())
        args, _ = gge_calls[0]
        self.assertTrue(args[3])  # fp8 is not None -> use_fp8 True
        self.assertEqual(out.dtype, hidden.dtype)


class _SonicExpertStub(moe_layer.SonicMoEExpert if moe_layer else object):
    """Subclass so ``isinstance(_, SonicMoEExpert)`` holds without running the
    heavy ``nn.Layer`` constructor; records the ``quant_weight`` dispatch on a
    class-level list to avoid ``nn.Layer.__setattr__`` (which needs a real
    ``__init__``)."""

    quant_calls = []

    def __init__(self):  # deliberately skip super().__init__ (heavy nn.Layer)
        pass

    def quant_weight(self):
        type(self).quant_calls.append(1)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestFp8QuantWeightGuards(_MoEMethodTestBase):
    def test_returns_without_quantizing_when_not_fusion_or_not_fp8(self):
        # Guard: not (moe_use_fusion_node and fp8) -> no-op early return.  The
        # experts object must be left completely untouched.
        experts = SimpleNamespace()
        for fusion, fp8 in ((False, True), (True, False), (False, False)):
            fake_self = SimpleNamespace(
                moe_use_fusion_node=fusion,
                fp8=fp8,
                grouped_gemm_experts=experts,
            )
            self.assertIsNone(MoELayer.fp8_quant_weight(fake_self))
        self.assertFalse(hasattr(experts, "fp8_weight_stacked"))

    def test_sonic_expert_dispatches_to_quant_weight(self):
        _SonicExpertStub.quant_calls = []
        experts = _SonicExpertStub()
        fake_self = SimpleNamespace(
            moe_use_fusion_node=True,
            fp8=True,
            grouped_gemm_experts=experts,
        )
        self.assertIsNone(MoELayer.fp8_quant_weight(fake_self))
        self.assertEqual(_SonicExpertStub.quant_calls, [1])

    def test_individual_mode_grouped_experts_not_implemented(self):
        # Non-Sonic grouped experts + batch_mode=False is an explicit
        # NotImplementedError (moe_layer.py line ~2262), not a silent no-op.
        fake_self = SimpleNamespace(
            moe_use_fusion_node=True,
            fp8=True,
            use_ue8m0=False,
            grouped_gemm_experts=SimpleNamespace(),
        )
        with self.assertRaises(NotImplementedError):
            MoELayer.fp8_quant_weight(fake_self, batch_mode=False)


if __name__ == "__main__":
    unittest.main()
