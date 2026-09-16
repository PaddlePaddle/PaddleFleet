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

"""``FLAGS_use_dsv4_accuracy`` must gate the TP linear + AOA-config call sites.

Companion to ``test_dsv4_accuracy_flag_gating.py`` (which pins the MoE / SwiGLU
/ scheduler / collate sites). This file covers the remaining single-card
reachable sites where the flag swaps a numeric kernel or a weight-conversion
rule:

* ``tensor_parallel.layers.LinearWithFrozenWeight.backward`` - the dgrad GEMM
  goes through the DSV4 ``te_matmul`` replay only when the flag is on;
  otherwise it stays on the historical ``grad_output @ weight.t()``.
* ``tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication``
  - the flag routes the dgrad through ``te_matmul`` and adds the
  ``linear_seqfirst_wgrad`` replay for the weight gradient.
* ``transformers.aoa_config_base.MoEAOAConfigGenerator._get_moe_expert_statements``
  - with the flag off the router ``mlp.gate.weight`` conversion is pinned to
  ``dtype='float32'``; with it on the native dtype is kept.

Every test pins both sides of the flag with an observable difference: which
kernel/branch runs (``assert_called`` / ``assert_not_called``) and/or the
resulting numeric gradient. The DSV4 replay helpers (``te_matmul``,
``linear_seqfirst_wgrad``) call into torch / transformer_engine, so they are
replaced with paddle stubs that reproduce the reference arithmetic; this keeps
the assertion on the *code path* the flag controls while keeping the numerics
valid and single-card.

The two ``fused_bias_swiglu`` sites named in the task (``clamped_swiglu_back``
at line 245 and ``clamped_weighted_swiglu_back`` at line 348) are already
covered by ``test_dsv4_accuracy_flag_gating.py`` and are intentionally not
duplicated here.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import paddle

from paddlefleet import accuracy_compatible_patch, utils
from paddlefleet.tensor_parallel import layers
from paddlefleet.transformers import aoa_config_base


def _dsv4_flag(module, enabled):
    return patch.object(
        module, "use_dsv4_accuracy_compatible", return_value=enabled
    )


def _te_matmul_ref(grad_output, weight):
    """Reference dgrad matching ``te_matmul``'s ``grad_output @ weight.t()``."""
    return grad_output.matmul(weight.t())


def _seqfirst_wgrad_ref(input, grad_output, weight):
    """Reference wgrad matching ``linear_seqfirst_wgrad``'s ``input.t() @ g``."""
    return input.t().matmul(grad_output)


class TestLinearWithFrozenWeightDgrad(unittest.TestCase):
    """``LinearWithFrozenWeight.backward`` (layers.py:504).

    The frozen-weight linear only produces an input gradient. The flag decides
    whether that dgrad is computed by the DSV4 ``te_matmul`` replay or by the
    historical ``grad_output.matmul(weight.t())``.
    """

    def _run(self, enabled):
        paddle.seed(7)
        x = paddle.randn([4, 8], dtype="float32")
        x.stop_gradient = False
        w = paddle.randn([8, 6], dtype="float32")
        w.stop_gradient = True  # frozen weight
        with (
            _dsv4_flag(layers, enabled),
            patch.object(
                accuracy_compatible_patch,
                "te_matmul",
                side_effect=_te_matmul_ref,
            ) as te,
        ):
            out = layers.LinearWithFrozenWeight.apply(x, w, None, False, None)
            out.sum().backward()
        # grad_output for ``out.sum()`` is all ones.
        reference = paddle.ones_like(out).matmul(w.t())
        return x.grad, te, reference

    def test_flag_on_routes_dgrad_through_te_matmul(self):
        grad, te, reference = self._run(True)
        te.assert_called_once()
        self.assertTrue(bool(paddle.allclose(grad, reference)))

    def test_flag_off_uses_plain_matmul(self):
        grad, te, reference = self._run(False)
        te.assert_not_called()
        self.assertTrue(bool(paddle.allclose(grad, reference)))


class TestLinearGradAccumBackward(unittest.TestCase):
    """``LinearWithGradAccumulationAndAsyncCommunication.backward``.

    Exercises the two flag-gated sites in the same backward on a single card
    (``tp_group=None``, no sequence-parallel / all-reduce collectives):

    * layers.py:1036 - the bf16 dgrad goes through ``te_matmul`` when the flag
      is on, otherwise through ``general_gemm(grad_output, weight.t())``.
    * layers.py:1104 - the weight gradient is produced by the
      ``linear_seqfirst_wgrad`` replay when the flag is on, otherwise by
      ``general_gemm(total_input.t(), grad_output)``.
    """

    def _run(self, enabled):
        paddle.seed(11)
        x = paddle.randn([4, 8], dtype="float32")
        x.stop_gradient = False
        w = paddle.randn([8, 6], dtype="float32")
        w.stop_gradient = False
        with (
            _dsv4_flag(layers, enabled),
            patch.object(
                accuracy_compatible_patch,
                "te_matmul",
                side_effect=_te_matmul_ref,
            ) as te,
            patch.object(
                accuracy_compatible_patch,
                "linear_seqfirst_wgrad",
                side_effect=_seqfirst_wgrad_ref,
            ) as seqfirst,
        ):
            out = layers.LinearWithGradAccumulationAndAsyncCommunication.apply(
                x,
                w,
                None,  # bias
                False,  # gradient_accumulation_fusion
                False,  # allreduce_dgrad
                False,  # sequence_parallel
                None,  # grad_output_buffer
                None,  # wgrad_deferral_limit
                None,  # tp_group
            )
            out.sum().backward()
        ones = paddle.ones_like(out)
        dgrad_ref = ones.matmul(w.t())
        wgrad_ref = x.t().matmul(ones)
        return x.grad, w.grad, te, seqfirst, dgrad_ref, wgrad_ref

    def test_flag_on_uses_te_matmul_and_seqfirst_wgrad(self):
        xgrad, wgrad, te, seqfirst, dgrad_ref, wgrad_ref = self._run(True)
        te.assert_called_once()  # layers.py:1036 dgrad replay
        seqfirst.assert_called_once()  # layers.py:1104 wgrad replay
        self.assertTrue(bool(paddle.allclose(xgrad, dgrad_ref)))
        self.assertTrue(bool(paddle.allclose(wgrad, wgrad_ref)))

    def test_flag_off_skips_both_dsv4_replays(self):
        xgrad, wgrad, te, seqfirst, dgrad_ref, wgrad_ref = self._run(False)
        te.assert_not_called()  # layers.py:1036 stays on general_gemm
        seqfirst.assert_not_called()  # layers.py:1104 stays on general_gemm
        self.assertTrue(bool(paddle.allclose(xgrad, dgrad_ref)))
        self.assertTrue(bool(paddle.allclose(wgrad, wgrad_ref)))


class TestMoeGateWeightDtypeGating(unittest.TestCase):
    """``_get_moe_expert_statements`` (aoa_config_base.py:469).

    The router ``mlp.gate.weight`` conversion statement is pinned to
    ``dtype='float32'`` with the flag off (the historical fp32 gate), and keeps
    the native dtype with the flag on. The flag is imported locally inside the
    method, so it is patched on ``paddlefleet.utils`` per the task rules.
    """

    def _gate_statement(self, enabled):
        params = aoa_config_base.MoEAOAConfigParams(has_shared_experts=False)
        with _dsv4_flag(utils, enabled):
            statements = aoa_config_base.MoEAOAConfigGenerator._get_moe_expert_statements(
                params, "m.layers.0", "model.layers.0"
            )
        gate = [s for s in statements if ".mlp.gate.weight ->" in s]
        self.assertEqual(len(gate), 1)
        return gate[0]

    def test_flag_off_pins_gate_weight_to_float32(self):
        self.assertIn("dtype='float32'", self._gate_statement(False))

    def test_flag_on_keeps_native_gate_weight_dtype(self):
        self.assertNotIn("dtype='float32'", self._gate_statement(True))

    def test_the_flag_actually_changes_the_statement(self):
        self.assertNotEqual(
            self._gate_statement(False), self._gate_statement(True)
        )


if __name__ == "__main__":
    unittest.main()
