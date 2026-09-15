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

"""Behaviour tests for the abstract MoE layer module (paddlefleet.nn.moe).

Scope: model layer / MoE routing + gating, no-card (CPU). See unit-test-rules
"模型与训练目标 / MoE" and "分布式训练 / Dispatcher".

``paddlefleet.nn.moe.abstract.MOELayerBase`` itself is an empty ``nn.Layer``
marker with no behaviour of its own; testing its ``state_dict`` / ``train`` /
``named_parameters`` only exercises ``paddle.nn.Layer`` internals, not MoE code.
Its ONE genuine contract is architectural: it is the shared base every
concrete MoE layer is dispatched to, so ``create_moe_block`` can annotate and
downstream code can ``isinstance``-check against it. That invariant is checked
here; everything else below tests the *real* CPU-verifiable routing/gating
control math that the abstract module coordinates (``topk_gate`` + the
``moe_block`` dispatch guard).

Every expected value is derived independently with NumPy / by hand from small,
sign-varied, position-distinguishable inputs, so the tests can reject swapped
mask polarity, wrong capacity branch, a dropped fp32 cast, a swapped
expert-weight interleave, a sign-flipped transport cost, or a broken
orthogonal-loss normalization.

Intentionally NOT exercised (recorded as an explicit skip below): the real
token dispatch / combine numerics in ``MOEAllGatherLayerV2`` /
``MOEAlltoAllLayer`` and the ``global_aux_loss`` / correction-bias all_reduce
paths. Those depend on multiple ranks each holding different tokens and then
exchanging them (AllToAll / AllGather / ReduceScatter); faking ``world_size``
and mocking the collective would only assert local orchestration, never the
cross-rank numerics (unit-test-antipatterns type 13). They need a real process
group on real cards.
"""

import unittest

import numpy as np
import paddle
import paddle.nn as nn

from paddlefleet.nn.moe.abstract import MOELayerBase
from paddlefleet.nn.moe.moe_block import create_moe_block
from paddlefleet.nn.moe.topk_gate import (
    TopKGate,
    cast_if_needed,
    compute_optimal_transport,
    gate_detach_matmul,
    masked_fill,
)


def setUpModule():
    # No-card: force CPU so nothing silently claims device numerics.
    paddle.set_device("cpu")


class _GateConfig:
    """Minimal but real config object for TopKGate (attribute + .get access)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def get(self, key, default=None):
        return getattr(self, key, default)


def make_gate(
    hidden_size=4,
    num_experts=4,
    capacity=(2.0, 3.0, 5.0),
    scoring_func="softmax",
    fuse=False,
):
    """Construct a REAL single-modality TopKGate through its real __init__.

    global_aux_loss=False and moe_orthogonal_loss_lambda=0 keep construction
    free of any Fleet / process-group dependency, so it runs on CPU.
    """
    cfg = _GateConfig(
        fuse_gate_detach_matmul=fuse,
        hidden_size=hidden_size,
        moe_num_experts=num_experts,
        moe_capacity=list(capacity),
        global_aux_loss=False,
        sinkhorn_2gate=False,
        sinkhorn_temp=1.0,
        moe_use_aux_free=False,
        scoring_func=scoring_func,
        moe_norm_gate_logits=True,
        router_aux_loss_coef=0.0,
        router_z_loss_coef=0.0,
        moe_orthogonal_loss_lambda=0.0,
        moe_k=2,
        moe_group_experts=False,
        moe_group_orthogonal_loss=False,
    )
    return TopKGate(cfg, layer_idx=0, group=None)


def make_mm_gate(hidden_size=2, num_experts=(2, 2), world_size=2):
    """Construct a REAL multimodel TopKGate (two expert groups, soft gate)."""
    cfg = _GateConfig(
        fuse_gate_detach_matmul=False,
        hidden_size=hidden_size,
        moe_num_experts=list(num_experts),
        moe_capacity=[2.0, 3.0, 5.0],
        global_aux_loss=False,
        sinkhorn_2gate=False,
        sinkhorn_temp=1.0,
        moe_use_aux_free=False,
        scoring_func="softmax",
        moe_norm_gate_logits=True,
        router_aux_loss_coef=0.0,
        router_z_loss_coef=0.0,
        moe_orthogonal_loss_lambda=0.0,
        moe_k=2,
        moe_group_experts=False,
        moe_group_orthogonal_loss=False,
        multimodel_experts=True,
        moe_world_size=world_size,
        moe_use_hard_gate=False,
        moe_use_token_type_bias=False,
    )
    return TopKGate(cfg, layer_idx=0, group=None)


def np_logsumexp(x, axis):
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=axis, keepdims=True)
    return m.squeeze(axis) + np.log(np.exp(x - m).sum(axis=axis))


# PLACEHOLDER_TESTS


class TestGateTensorHelpers(unittest.TestCase):
    """Pure routing/gating helpers from topk_gate (CPU numerics)."""

    def test_masked_fill_fills_true_positions(self):
        # Distinguishable content + mixed mask -> catches swapped where() polarity.
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32"
        )
        mask = paddle.to_tensor(
            [[True, False, True], [False, True, False]], dtype="bool"
        )
        out = masked_fill(x, mask, -9.0)
        expected = np.where(
            np.array([[True, False, True], [False, True, False]]),
            -9.0,
            x.numpy(),
        )
        np.testing.assert_array_equal(out.numpy(), expected)
        # untouched positions keep their ORIGINAL distinct values
        self.assertEqual(out.numpy()[0, 1], 2.0)
        self.assertEqual(out.numpy()[1, 0], 4.0)

    def test_cast_if_needed_identity_and_cast(self):
        x = paddle.to_tensor([1.0, 2.0], dtype="float32")
        # same dtype -> SAME object returned (no needless copy), per contract
        self.assertIs(cast_if_needed(x, paddle.float32), x)
        # differing dtype -> new tensor, target dtype, values preserved
        y = cast_if_needed(x, paddle.float64)
        self.assertIsNot(y, x)
        self.assertEqual(y.dtype, paddle.float64)
        np.testing.assert_allclose(y.numpy(), [1.0, 2.0])

    def test_gate_detach_matmul_fused_matches_reference(self):
        # gate logits = input @ gate_weight computed in fp32; both the fused
        # PyLayer path and the plain path must equal the independent reference.
        x = paddle.to_tensor(
            [[1.0, -2.0], [3.0, 4.0], [-1.0, 0.5]], dtype="float32"
        )
        w = paddle.to_tensor(
            [[2.0, 0.0, -1.0], [1.0, 3.0, 0.5]], dtype="float32"
        )
        ref = x.numpy().astype(np.float64) @ w.numpy().astype(np.float64)
        out_plain = gate_detach_matmul(x, w, use_fuse=False)
        out_fused = gate_detach_matmul(x, w, use_fuse=True)
        self.assertEqual(out_plain.dtype, paddle.float32)
        self.assertEqual(out_fused.dtype, paddle.float32)
        np.testing.assert_allclose(out_plain.numpy(), ref, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(out_fused.numpy(), ref, rtol=1e-6, atol=1e-6)

    def test_compute_optimal_transport_marginals_and_cost_monotonic(self):
        # Sinkhorn balanced assignment: mass flows to LOW cost, and the last
        # column-scaling step enforces the column marginals. A sign flip
        # (softmax(+M)) would invert the monotonicity below.
        M = paddle.to_tensor([[0.0, 4.0], [4.0, 0.0]], dtype="float32")
        r = paddle.to_tensor([1.0, 1.0], dtype="float32")
        c = paddle.to_tensor([1.0, 1.0], dtype="float32")
        P, _ = compute_optimal_transport(M, r, c, lam=1.0, max_iters=100)
        p = P.numpy()
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue((p >= 0).all())
        # column marginals satisfied (guaranteed by final column scaling)
        np.testing.assert_allclose(p.sum(axis=0), [1.0, 1.0], atol=1e-2)
        # cheap diagonal cells get strictly more mass than expensive off-diagonal
        self.assertGreater(p[0, 0], p[0, 1])
        self.assertGreater(p[1, 1], p[1, 0])


# PLACEHOLDER_TESTS2


class TestTopKGateControl(unittest.TestCase):
    """Real TopKGate construction + capacity / forward / gate-weight control."""

    def test_get_capacity_train_eval_branches(self):
        # capacity = int(cap * num_tokens // num_experts); the cap picked
        # depends on train vs eval and (in eval) num_tokens vs num_experts.
        # Distinct caps (2/3/5) make a swapped branch observable.
        gate = make_gate(num_experts=4, capacity=(2.0, 3.0, 5.0))
        gate.train()
        # training -> cap[0]=2.0 : int(2.0*8//4)=4
        self.assertEqual(gate.get_capacity(8), 4)
        gate.eval()
        # eval, num_tokens(8) >= num_experts(4) -> cap[1]=3.0 : int(3.0*8//4)=6
        self.assertEqual(gate.get_capacity(8), 6)
        # eval, num_tokens(2) < num_experts(4) -> cap[2]=5.0 : int(5.0*2//4)=2
        self.assertEqual(gate.get_capacity(2), 2)
        # explicit cap_factor overrides the branch entirely: int(1.0*8//4)=2
        self.assertEqual(gate.get_capacity(8, cap_factor=1.0), 2)

    def test_get_capacity_asserts_positive(self):
        # cap_factor small enough to floor to zero must trip the >0 guard.
        gate = make_gate(num_experts=4)
        with self.assertRaises(AssertionError):
            gate.get_capacity(8, cap_factor=0.1)  # int(0.1*8//4)=int(0.0)=0

    def test_forward_logits_capacity_and_router_loss(self):
        gate = make_gate(hidden_size=4, num_experts=4, capacity=(2.0, 3.0, 5.0))
        gate.train()
        x = paddle.to_tensor(
            [
                [1.0, 0.0, -1.0, 2.0],
                [0.5, -0.5, 1.5, -2.0],
                [2.0, 1.0, 0.0, -1.0],
                [-1.0, 3.0, 0.5, 0.5],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [-2.0, -1.0, 0.0, 1.0],
                [3.0, -3.0, 2.0, -2.0],
            ],
            dtype="float32",
        )
        logits, capacity, router_loss = gate.forward(x)
        # logits = input @ gate.weight in fp32 (reference uses the real weight,
        # but the matmul is recomputed independently with numpy).
        ref = x.numpy().astype(np.float64) @ gate.weight.numpy().astype(
            np.float64
        )
        np.testing.assert_allclose(logits.numpy(), ref, rtol=1e-5, atol=1e-5)
        # training capacity: int(2.0*8//4)=4
        self.assertEqual(capacity, 4)
        # router_loss is a fresh differentiable zero of shape [1]
        self.assertEqual(list(router_loss.shape), [1])
        np.testing.assert_array_equal(router_loss.numpy(), [0.0])
        self.assertFalse(router_loss.stop_gradient)

    def test_get_gate_weight_non_multimodel_identity(self):
        # single modality: gate weight is returned unchanged for either flag.
        gate = make_gate(hidden_size=3, num_experts=5)
        self.assertIs(gate.get_gate_weight(transform_weight=True), gate.weight)
        self.assertIs(gate.get_gate_weight(transform_weight=False), gate.weight)

    def test_get_gate_weight_multimodel_concat(self):
        # transform_weight=False -> plain concat of the two modality gates.
        gate = make_mm_gate(hidden_size=2, num_experts=(2, 2), world_size=2)
        gate.weight.set_value(
            paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        )
        gate.weight_1.set_value(
            paddle.to_tensor([[10.0, 20.0], [30.0, 40.0]], dtype="float32")
        )
        out = gate.get_gate_weight(transform_weight=False)
        expected = np.array([[1.0, 2.0, 10.0, 20.0], [3.0, 4.0, 30.0, 40.0]])
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_get_gate_weight_multimodel_interleave(self):
        # transform_weight=True groups, per world-slot, that slot's experts from
        # BOTH modality groups: columns become [w0c0, w1c0, w0c1, w1c1].
        # A wrong interleave (e.g. plain concat) would fail here even though the
        # shape [2,4] is identical -> this is what a shape-only test misses.
        gate = make_mm_gate(hidden_size=2, num_experts=(2, 2), world_size=2)
        gate.weight.set_value(
            paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        )
        gate.weight_1.set_value(
            paddle.to_tensor([[10.0, 20.0], [30.0, 40.0]], dtype="float32")
        )
        out = gate.get_gate_weight(transform_weight=True)
        expected = np.array([[1.0, 10.0, 2.0, 20.0], [3.0, 30.0, 4.0, 40.0]])
        self.assertEqual(list(out.shape), [2, 4])
        np.testing.assert_array_equal(out.numpy(), expected)


# PLACEHOLDER_TESTS3


class TestRouterRegularizers(unittest.TestCase):
    """z-loss and orthogonal-loss router regularizers (CPU numerics)."""

    def test_z_loss_matches_reference(self):
        gate = make_gate(hidden_size=4, num_experts=3)
        logits = paddle.to_tensor(
            [[1.0, 2.0, -1.0], [0.0, 0.5, 3.0], [-2.0, 1.0, 1.0]],
            dtype="float32",
        )
        out = gate._cal_z_loss(logits)
        # z-loss = mean over rows of logsumexp(row)^2
        lse = np_logsumexp(logits.numpy(), axis=1)
        expected = float((lse**2).mean())
        np.testing.assert_allclose(
            out.numpy().reshape([]), expected, rtol=1e-5, atol=1e-6
        )

    def test_z_loss_with_mask_matches_reference(self):
        gate = make_gate(hidden_size=4, num_experts=3)
        logits = paddle.to_tensor(
            [[1.0, 2.0, -1.0], [0.0, 0.5, 3.0], [-2.0, 1.0, 1.0]],
            dtype="float32",
        )
        mask = paddle.to_tensor([1.0, 0.0, 1.0], dtype="float32")
        out = gate._cal_z_loss(logits, loss_mask=mask)
        lse = np_logsumexp(logits.numpy(), axis=1)
        m = mask.numpy()
        expected = float((lse**2 * m).sum() / max(m.sum(), 1e-6))
        np.testing.assert_allclose(
            out.numpy().reshape([]), expected, rtol=1e-5, atol=1e-6
        )

    def test_orthogonal_loss_zero_for_orthonormal_and_positive_for_duplicate(
        self,
    ):
        # rows (per expert) are L2-normalized, then ||W W^T - I||^2 / size.
        gate = make_gate(hidden_size=2, num_experts=2)
        # experts orthonormal in model_dim space -> Gram = I -> loss 0
        gate.weight.set_value(
            paddle.to_tensor([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
        )
        loss_orth = gate._cal_orthogonal_loss_opt_each_weight(
            gate.weight, use_group=False
        )
        np.testing.assert_allclose(
            loss_orth.numpy().reshape([]), 0.0, atol=1e-6
        )
        # two identical experts (both (1,0)) -> Gram=[[1,1],[1,1]],
        # (Gram - I) = [[0,1],[1,0]] -> sum(sq)=2, size=4 -> 0.5
        gate.weight.set_value(
            paddle.to_tensor([[1.0, 1.0], [0.0, 0.0]], dtype="float32")
        )
        loss_dup = gate._cal_orthogonal_loss_opt_each_weight(
            gate.weight, use_group=False
        )
        np.testing.assert_allclose(
            loss_dup.numpy().reshape([]), 0.5, rtol=1e-5, atol=1e-6
        )


class TestMoEBlockDispatchAndBase(unittest.TestCase):
    """create_moe_block dispatch guard + the abstract base architectural contract."""

    def test_create_moe_block_rejects_invalid_mode(self):
        # The dispatch control raises before building any (device-bound) layer.
        with self.assertRaises(ValueError):
            create_moe_block(
                gate=None,
                experts=[],
                layer_idx=0,
                moe_mode="not-a-real-mode",
            )

    def test_abstract_base_is_shared_marker_for_concrete_layers(self):
        # The only genuine contract of the empty abstract base: it is an
        # nn.Layer and both concrete MoE layers (the two create_moe_block
        # dispatch targets) subclass it, so the annotated return type and any
        # downstream isinstance(x, MOELayerBase) checks hold.
        from paddlefleet.nn.moe.moe_allgather_layer import MOEAllGatherLayerV2
        from paddlefleet.nn.moe.moe_alltoall_layer import MOEAlltoAllLayer

        self.assertTrue(issubclass(MOELayerBase, nn.Layer))
        self.assertTrue(issubclass(MOEAllGatherLayerV2, MOELayerBase))
        self.assertTrue(issubclass(MOEAlltoAllLayer, MOELayerBase))

    @unittest.skip(
        "Real token dispatch/combine numerics (AllToAll/AllGather/ReduceScatter "
        "in MOEAllGatherLayerV2 / MOEAlltoAllLayer forward, and the "
        "global_aux_loss / correction-bias all_reduce paths) require multiple "
        "ranks each holding different tokens and exchanging them. Faking "
        "world_size + mocking the collective would only prove local "
        "orchestration, not cross-rank numerics (antipattern type 13). Needs a "
        "real process group on real cards (tests/multi_card_tests/moe/)."
    )
    def test_multi_card_dispatch_combine(self):
        pass


if __name__ == "__main__":
    unittest.main()
