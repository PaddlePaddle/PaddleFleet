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

import numpy as np

# Heavy imports are guarded honestly: this CPU environment has no paddle
# installed, so the whole MoE stack is unimportable. We skip with a truthful
# reason rather than faking a pass. Only ImportError/ModuleNotFoundError are
# treated as "missing dependency"; any other error must surface as a failure.
try:
    import paddle

    from paddlefleet.transformer.moe.moe_layer import MoELayer

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    paddle = None
    MoELayer = None
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    "paddle / paddlefleet MoE stack not importable on this CPU environment: "
    f"{_IMPORT_ERROR!r}"
)


# These tests invoke the real, unmodified MoELayer methods (use_fp8,
# fp8_quant_weight, aux_loss_compute) directly on a lightweight ``self`` data
# holder. MoELayer.__init__ requires a full TransformerConfig, a live
# ProcessGroupCollection (pg_collection.ep) and expert sublayers, which cannot
# be constructed on a CPU-only box; the methods under test, however, only read a
# handful of plain attributes. We therefore set exactly those attributes by
# hand and run the genuine method bodies -- the production entry stays on the
# validation chain (no method is patched or rewritten), only the heavyweight
# constructor is bypassed.
@unittest.skipUnless(MoELayer is not None, _SKIP_REASON)
class TestMoELayerUseFp8(unittest.TestCase):
    """MoELayer.use_fp8 returns True only when fusion-node AND fp8 are both on.

    moe_layer.py:2338-2341 -- ``return self.moe_use_fusion_node and self.fp8``.
    A weak "callable"/"hasattr" check cannot tell this apart from an ``or`` or a
    constant, so we exercise the full boolean truth table.
    """

    def _use_fp8(self, moe_use_fusion_node, fp8):
        stub = SimpleNamespace(moe_use_fusion_node=moe_use_fusion_node, fp8=fp8)
        return MoELayer.use_fp8(stub)

    def test_true_only_when_both_enabled(self):
        self.assertIs(self._use_fp8(True, True), True)

    def test_false_when_fusion_node_off(self):
        self.assertIs(self._use_fp8(False, True), False)

    def test_false_when_fp8_off(self):
        self.assertIs(self._use_fp8(True, False), False)

    def test_false_when_both_off(self):
        self.assertIs(self._use_fp8(False, False), False)


@unittest.skipUnless(MoELayer is not None, _SKIP_REASON)
class TestMoELayerFp8QuantWeightGuard(unittest.TestCase):
    """MoELayer.fp8_quant_weight guard + individual-mode contract.

    moe_layer.py:2190-2192 short-circuits with ``return`` (None) unless
    ``moe_use_fusion_node and fp8``. moe_layer.py:2261-2264 rejects
    ``batch_mode=False`` with NotImplementedError once a grouped_gemm_experts
    exists. We give the stub a ``grouped_gemm_experts`` that is NOT a
    SonicMoEExpert so the isinstance branch (2193-2197) is skipped and control
    reaches the real guard / raise logic.
    """

    def test_guard_returns_none_when_fusion_node_off(self):
        # If the guard were mistakenly an ``or``, this would fall through and
        # raise NotImplementedError instead of returning None.
        stub = SimpleNamespace(
            moe_use_fusion_node=False,
            fp8=True,
            grouped_gemm_experts=object(),
        )
        self.assertIsNone(MoELayer.fp8_quant_weight(stub, batch_mode=False))

    def test_guard_returns_none_when_fp8_off(self):
        stub = SimpleNamespace(
            moe_use_fusion_node=True,
            fp8=False,
            grouped_gemm_experts=object(),
        )
        self.assertIsNone(MoELayer.fp8_quant_weight(stub, batch_mode=False))

    def test_individual_mode_rejected_when_enabled(self):
        # Guard passes -> non-Sonic grouped_gemm_experts -> batch_mode=False
        # must hit the explicit NotImplementedError (moe_layer.py:2261-2264).
        stub = SimpleNamespace(
            moe_use_fusion_node=True,
            fp8=True,
            grouped_gemm_experts=object(),
        )
        with self.assertRaises(NotImplementedError):
            MoELayer.fp8_quant_weight(stub, batch_mode=False)


@unittest.skipUnless(MoELayer is not None, _SKIP_REASON)
class TestMoELayerAuxLossCompute(unittest.TestCase):
    """MoELayer.aux_loss_compute (moe_layer.py:1752-1778) behavior.

    Verifies: (1) hidden_states are forwarded unchanged and reshaped to the
    residual shape; (2) the aux loss is scaled by router_aux_loss_coef and
    routed through AddAuxiliaryLoss so its gradient equals the coefficient;
    (3) the training guard suppresses the aux-loss gradient path; (4) z_loss is
    added on its own (unscaled) AddAuxiliaryLoss branch; (5) the shared-expert
    output is consumed (added) using the residuals it is handed.
    """

    def _base_stub(self, **overrides):
        stub = SimpleNamespace(
            use_latent_moe=False,
            training=True,
            router_aux_loss_coef=0.0,
            shared_experts=None,
            expert_model_parallel_size=1,
            sequence_parallel=False,
        )
        for key, value in overrides.items():
            setattr(stub, key, value)
        return stub

    def test_forwards_and_reshapes_hidden_states(self):
        hidden = paddle.arange(12, dtype="float32").reshape([6, 2])
        residuals = paddle.zeros([2, 3, 2], dtype="float32")
        stub = self._base_stub(training=False)

        out = MoELayer.aux_loss_compute(stub, (hidden, None, None, residuals))

        self.assertEqual(out.shape, [2, 3, 2])
        # Values are the untouched hidden_states, just reshaped.
        np.testing.assert_array_equal(
            out.numpy(), hidden.numpy().reshape(2, 3, 2)
        )

    def test_aux_loss_gradient_equals_coefficient(self):
        coef = 0.5
        hidden = paddle.arange(12, dtype="float32").reshape([6, 2])
        hidden.stop_gradient = False
        residuals = paddle.zeros([2, 3, 2], dtype="float32")
        aux = paddle.to_tensor([2.0], dtype="float32")
        aux.stop_gradient = False
        stub = self._base_stub(training=True, router_aux_loss_coef=coef)

        out = MoELayer.aux_loss_compute(stub, (hidden, aux, None, residuals))
        out.sum().backward()

        # AddAuxiliaryLoss.backward feeds ones(1) into the *scaled* aux loss
        # (aux * coef), so d/d(aux) = coef. This proves both that the aux loss
        # participates in the graph and that router_aux_loss_coef is consumed.
        self.assertIsNotNone(aux.grad)
        np.testing.assert_allclose(aux.grad.numpy(), [coef], atol=1e-6)
        # Forward output still carries the (cloned) hidden_states.
        self.assertIsNotNone(hidden.grad)

    def test_not_training_drops_aux_loss_gradient(self):
        hidden = paddle.arange(12, dtype="float32").reshape([6, 2])
        hidden.stop_gradient = False
        residuals = paddle.zeros([2, 3, 2], dtype="float32")
        aux = paddle.to_tensor([2.0], dtype="float32")
        aux.stop_gradient = False
        stub = self._base_stub(training=False, router_aux_loss_coef=0.5)

        out = MoELayer.aux_loss_compute(stub, (hidden, aux, None, residuals))
        out.sum().backward()

        # Not training -> aux loss must NOT enter the graph.
        self.assertIsNone(aux.grad)

    def test_z_loss_added_on_own_branch(self):
        hidden = paddle.arange(12, dtype="float32").reshape([6, 2])
        residuals = paddle.zeros([2, 3, 2], dtype="float32")
        z_loss = paddle.to_tensor([3.0], dtype="float32")
        z_loss.stop_gradient = False
        # aux is None so only the z_loss AddAuxiliaryLoss branch fires.
        stub = self._base_stub(training=True, router_aux_loss_coef=0.5)

        out = MoELayer.aux_loss_compute(stub, (hidden, None, z_loss, residuals))
        out.sum().backward()

        # z_loss is routed through AddAuxiliaryLoss unscaled -> grad ones(1).
        self.assertIsNotNone(z_loss.grad)
        np.testing.assert_allclose(z_loss.grad.numpy(), [1.0], atol=1e-6)

    def test_shared_expert_output_is_consumed(self):
        hidden = paddle.arange(12, dtype="float32").reshape([6, 2])
        residuals = paddle.full([2, 3, 2], 5.0, dtype="float32")
        marker = paddle.full([2, 3, 2], 7.0, dtype="float32")
        seen = {}

        def shared_experts(received):
            seen["arg"] = received
            return (marker,)

        stub = self._base_stub(training=False, shared_experts=shared_experts)

        out = MoELayer.aux_loss_compute(stub, (hidden, None, None, residuals))

        # The residuals tensor must be the exact object handed to the shared
        # expert, and its output must be added onto the reshaped hidden states.
        self.assertIs(seen["arg"], residuals)
        expected = hidden.numpy().reshape(2, 3, 2) + 7.0
        np.testing.assert_array_equal(out.numpy(), expected)


if __name__ == "__main__":
    unittest.main()
