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

"""Behaviour tests for paddlefleet.nn.moe.moe_alltoall_layer (MoE AllToAll).

Module: 分布式训练 (MoE AllToAll dispatcher) crossed with 模型层 (MoE routing /
expert math). Environment: 无卡 (CPU only).

What is verified here, with independently derived expectations, keeping the real
production code on the observed call chain:

* ``combining(hard_gate=True)`` — the hard-gate combine is a real ``F.embedding``
  gather by ``scatter_index``; we check it returns each row selected by index
  (expert ownership expressed at the combine side), not just a shape.
* ``MOEAlltoAllLayer.forward_experts`` — with world_size==1 the reshape /
  transpose / per-slot dispatch routes chunk *i* to expert *i*. Experts are set
  to distinct scalar-identity maps so a wrong-expert routing or a scrambled
  reshape produces wrong values, not merely a wrong shape.
* ``fused_gate_logits_process`` — the routing-probability computation, including
  the ``group_experts`` normalisation that divides each group by its own max.
* ``_calc_router_loss`` — the control branch selected when
  ``router_aux_loss_coef == 0`` (the "must use gate prob to avoid zero pointer"
  term), verifying it stays tied to ``gate_prob`` and yields a zero tensor.
* ``__init__`` world_size==1 (dummy-moe) local path — num_local_experts mapping,
  gate-parameter tagging and the single-rank expert flag fallback.

Explicitly NOT verified (see report / anti-pattern rules):

* Cross-rank AllToAll dispatch and combine numerics. ``AlltoAll.forward`` is a
  pure identity when ``dist.get_world_size(group) <= 1`` (see all_to_all.py), so
  a single process cannot prove peer selection, split sizes or expert re-homing.
  Faking world_size>1 and mocking the collective would only assert "was called"
  and is rejected by the distributed-test rules; that path needs a real process
  group (multi-card) and is left to tests/multi_card_tests.
* The soft-gate combine (``GateCombine`` / ``moe_combine``) and
  ``moe_gate_dispatch`` numerics: these are ``paddle.incubate`` fused custom ops
  that are not exercised on CPU, so their numeric path is not asserted here.

dist.get_world_size / dist.get_rank are pinned to the honest single-rank values
(1 / 0) only during construction to select the world_size==1 local path; this is
NOT a fake multi-rank + mocked-collective setup.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddlefleet.nn.moe.moe_alltoall_layer import (
    MOEAlltoAllLayer,
    combining,
)

_MODPATH = "paddlefleet.nn.moe.moe_alltoall_layer"


def _np_softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


class _NoHCGFleet:
    """A fleet stand-in without ``_hcg`` so is_mp_moe resolves to False."""


def _make_gate(in_dim, num_experts):
    """Build a real gate (nn.Linear) with the attributes the layer reads.

    A concrete Linear is used (not a MagicMock) so that gate parameters really
    exist and get tagged by the constructor, and so ``gate.act`` is the real
    softmax that fused_gate_logits_process consumes.
    """
    gate = nn.Linear(in_dim, num_experts)
    gate.config = SimpleNamespace(
        moe_use_hard_gate=False,
        norm_gate_logits=False,
        router_aux_loss_coef=0.0,
        moe_use_aux_free=True,
        moe_orthogonal_loss_lambda=0.0,
        router_z_loss_coef=0.0,
    )
    gate.act = F.softmax
    gate.experts_type_ids = paddle.zeros([num_experts], dtype="int64")
    return gate


def _make_layer(gate, experts, **kwargs):
    """Construct the layer on the world_size==1 local path.

    Patches only wrap construction, since world_size / rank / num_local_experts
    are cached as attributes and no method under test issues a collective.
    """
    with (
        patch(f"{_MODPATH}.dist.get_world_size", return_value=1),
        patch(f"{_MODPATH}.dist.get_rank", return_value=0),
        patch(f"{_MODPATH}.fleet.fleet", _NoHCGFleet()),
    ):
        return MOEAlltoAllLayer(
            gate=gate, experts=experts, layer_idx=0, **kwargs
        )


class TestCombiningHardGate(unittest.TestCase):
    """combining(hard_gate=True) is an index gather over the expert rows."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_hard_gate_gathers_rows_by_scatter_index(self):
        # x rows are uniquely identifiable so a wrong index is visible.
        s, dim, k = 3, 4, 2
        x_np = (np.arange(s * dim, dtype="float32") + 1.0).reshape([s, dim])
        # Non-trivial, non-identity index map: each (i,j) selects a known row.
        idx_np = np.array([[2, 0, 1], [1, 2, 0]], dtype="int64")  # [k, s]

        x = paddle.to_tensor(x_np)
        scatter_index = paddle.to_tensor(idx_np)
        combine_weights = paddle.ones(
            [s, k], dtype="float32"
        )  # unused on this path

        out = combining(x, combine_weights, scatter_index, hard_gate=True)

        # F.embedding(idx[k,s], x[s,dim]) -> [k,s,dim]; squeeze(-2) is a no-op
        # because that axis (s=3) is not 1.
        expected = x_np[idx_np]  # [k, s, dim] independent gather
        self.assertEqual(list(out.shape), [k, s, dim])
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_hard_gate_wrong_index_would_change_output(self):
        # Guard: the assertion above is sensitive to the index mapping.
        s, dim, k = 3, 4, 2
        x_np = (np.arange(s * dim, dtype="float32") + 1.0).reshape([s, dim])
        idx_np = np.array([[0, 1, 2], [2, 1, 0]], dtype="int64")
        out = combining(
            paddle.to_tensor(x_np),
            paddle.ones([s, k], dtype="float32"),
            paddle.to_tensor(idx_np),
            hard_gate=True,
        )
        wrong = x_np[np.array([[1, 1, 1], [1, 1, 1]], dtype="int64")]
        self.assertFalse(np.array_equal(out.numpy(), wrong))


class TestForwardExpertsOwnership(unittest.TestCase):
    """forward_experts routes each expert slot to its own expert (ws==1)."""

    def setUp(self):
        paddle.set_device("cpu")

    def _scalar_expert(self, dim, scale):
        """Linear configured as x -> scale * x (weight = scale*I, bias = 0)."""
        expert = nn.Linear(dim, dim)
        expert.weight.set_value(
            paddle.to_tensor(scale * np.eye(dim, dtype="float32"))
        )
        expert.bias.set_value(paddle.zeros([dim], dtype="float32"))
        return expert

    def test_each_slot_uses_its_own_expert(self):
        dim = 4
        gate = _make_gate(in_dim=dim, num_experts=2)
        # Two experts with distinct scalar-identity behaviour: sending the
        # wrong chunk to the wrong expert changes the numbers, not just shapes.
        experts = nn.LayerList(
            [self._scalar_expert(dim, 2.0), self._scalar_expert(dim, 5.0)]
        )
        for p in experts.parameters():
            p.expert = False
            p.no_sync = False

        layer = _make_layer(gate, experts)
        self.assertEqual(layer.num_local_experts, 2)

        # dispatched_input [num_experts=2, capacity=3, dim]; distinct content.
        cap = 3
        disp_np = (np.arange(2 * cap * dim, dtype="float32") + 1.0).reshape(
            [2, cap, dim]
        )
        dispatched = paddle.to_tensor(disp_np)

        out = layer.forward_experts(dispatched)

        # For world_size==1 the reshape([1,2,-1,dim]) + transpose([1,0,2,3]) +
        # unbind(0) hands chunk i == dispatched[i] to expert i, then
        # stack(axis=1) -> [1, 2, cap, dim].
        self.assertEqual(list(out.shape), [1, 2, cap, dim])
        np.testing.assert_allclose(
            out.numpy()[0, 0], 2.0 * disp_np[0], rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            out.numpy()[0, 1], 5.0 * disp_np[1], rtol=1e-6, atol=1e-6
        )
        # Cross-check: slot 0 must NOT equal expert-1's transform of chunk 0.
        self.assertFalse(np.allclose(out.numpy()[0, 0], 5.0 * disp_np[0]))


class TestFusedGateLogitsProcess(unittest.TestCase):
    """Routing-probability computation and group normalisation."""

    def setUp(self):
        paddle.set_device("cpu")

    def _layer(self, num_experts, k, group_experts):
        gate = _make_gate(in_dim=num_experts, num_experts=num_experts)
        experts = nn.LayerList([nn.Linear(4, 4) for _ in range(num_experts)])
        for p in experts.parameters():
            p.expert = False
            p.no_sync = False
        return _make_layer(gate, experts, k=k, group_experts=group_experts)

    def test_plain_softmax_prob(self):
        layer = self._layer(num_experts=4, k=2, group_experts=False)
        logits_np = np.array(
            [[2.0, 0.0, -1.0, 1.0], [0.5, 0.5, 3.0, -2.0]], dtype="float32"
        )
        prob, max_prob = layer.fused_gate_logits_process(
            paddle.to_tensor(logits_np)
        )
        self.assertIsNone(max_prob)
        expected = _np_softmax(logits_np, axis=-1)
        self.assertEqual(list(prob.shape), [2, 4])
        np.testing.assert_allclose(prob.numpy(), expected, rtol=1e-5, atol=1e-6)
        # Softmax rows sum to 1 in the non-group path.
        np.testing.assert_allclose(
            prob.numpy().sum(axis=-1), np.ones(2), rtol=1e-5, atol=1e-6
        )

    def test_group_experts_divides_each_group_by_its_max(self):
        # num_experts = k * per_group so the reshape [s, k, -1] is exact.
        num_experts, k = 8, 2
        layer = self._layer(num_experts=num_experts, k=k, group_experts=True)
        s = 2
        logits_np = np.array(
            [
                [2.0, 0.0, -1.0, 1.0, 0.5, 0.5, 3.0, -2.0],
                [1.0, -1.0, 0.0, 2.0, -0.5, 0.5, 1.5, 0.2],
            ],
            dtype="float32",
        )
        prob, max_prob = layer.fused_gate_logits_process(
            paddle.to_tensor(logits_np)
        )

        # Independent reference: softmax within each group of size 4, then
        # divide each group by its own max; max_prob keeps that per-group max.
        grouped = logits_np.reshape([s, k, -1])
        sm = _np_softmax(grouped, axis=-1)
        gmax = np.max(sm, axis=-1, keepdims=True)  # [s, k, 1]
        ref_prob = (sm / gmax).reshape([s, num_experts])

        self.assertIsNotNone(max_prob)
        self.assertEqual(list(max_prob.shape), [s, k, 1])
        np.testing.assert_allclose(max_prob.numpy(), gmax, rtol=1e-5, atol=1e-6)
        self.assertEqual(list(prob.shape), [s, num_experts])
        np.testing.assert_allclose(prob.numpy(), ref_prob, rtol=1e-5, atol=1e-6)
        # The per-group max entry must normalise to exactly 1.0; this catches a
        # regression that forgot the division (plain softmax has no 1.0 entry).
        reshaped = prob.numpy().reshape([s, k, -1])
        np.testing.assert_allclose(
            reshaped.max(axis=-1), np.ones([s, k]), rtol=1e-5, atol=1e-6
        )


class TestCalcRouterLossZeroCoef(unittest.TestCase):
    """The router_aux_loss_coef == 0 control branch."""

    def setUp(self):
        paddle.set_device("cpu")

    def _layer(self):
        num_experts = 4
        gate = _make_gate(in_dim=num_experts, num_experts=num_experts)
        experts = nn.LayerList([nn.Linear(4, 4), nn.Linear(4, 4)])
        for p in experts.parameters():
            p.expert = False
            p.no_sync = False
        return _make_layer(gate, experts, group_experts=False)

    def test_zero_coef_returns_gate_prob_tied_zero_tensor(self):
        layer = self._layer()
        # All loss coefficients are 0 -> the branch returns
        # self.zero * gate_prob[0, 0], a real tensor (not python float 0.0).
        gate_prob = paddle.abs(
            paddle.to_tensor(
                np.array(
                    [[0.3, 0.7, 0.1, 0.9], [0.2, 0.2, 0.5, 0.1]],
                    dtype="float32",
                )
            )
        )
        dispatch_mask = paddle.ones([4], dtype="float32")
        gate_logits = paddle.to_tensor(np.zeros([2, 4], dtype="float32"))

        result = layer._calc_router_loss(
            dispatch_mask,
            gate_logits,
            gate_prob,
            4,
            False,
            0,
        )
        # Must be a tensor (proves the gate_prob term was added, guarding the
        # "avoid zero pointer" contract) and numerically zero.
        self.assertIsInstance(result, paddle.Tensor)
        self.assertTrue(np.isfinite(result.numpy()).all())
        np.testing.assert_allclose(
            result.numpy(), np.zeros_like(result.numpy()), atol=0.0
        )


class TestInitWorldSizeOneLocalPath(unittest.TestCase):
    """Construction on the world_size==1 (dummy-moe) local path."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_local_mapping_and_param_tagging(self):
        num_experts = 4
        gate = _make_gate(in_dim=8, num_experts=num_experts)
        experts = nn.LayerList([nn.Linear(4, 4) for _ in range(num_experts)])
        for p in experts.parameters():
            p.expert = True  # pre-set opposite to prove the constructor writes.
            p.no_sync = True

        layer = _make_layer(gate, experts, k=2)

        # world_size==1 -> num_local_experts == len(experts).
        self.assertEqual(layer.world_size, 1)
        self.assertEqual(layer.rank, 0)
        self.assertEqual(layer.num_local_experts, num_experts)
        self.assertEqual(layer.k, 2)
        self.assertFalse(layer.multimodal_experts)
        self.assertFalse(layer.use_correction_bias)
        self.assertIs(layer.config, gate.config)

        # dummy-moe (world_size==1): expert params are treated as regular
        # (non-expert, sync-on) params.
        for p in layer.experts.parameters():
            self.assertFalse(p.expert)
            self.assertFalse(p.no_sync)
        # Gate params are tagged is_gate.
        for p in layer.gate.parameters():
            self.assertTrue(p.is_gate)

    def test_multimodal_expert_partition(self):
        num_experts = 4
        gate = _make_gate(in_dim=8, num_experts=num_experts)
        experts = nn.LayerList([nn.Linear(4, 4) for _ in range(num_experts)])
        for p in experts.parameters():
            p.expert = False
            p.no_sync = False

        layer = _make_layer(gate, experts, moe_num_experts=[2, 2])
        self.assertTrue(layer.multimodal_experts)
        # world_size==1 -> each modality keeps its full local expert count.
        self.assertEqual(layer.num_local_multimodal_experts, [2, 2])
        self.assertEqual(layer.multimodal_expert_index, [0, 2, 4])


if __name__ == "__main__":
    unittest.main()
