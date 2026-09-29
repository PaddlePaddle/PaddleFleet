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

"""Behaviour tests for ``paddlefleet.nn.moe.moe_block``.

Module: 模型层 (MoE factory + correction-bias statics) crossed with 分布式训练
(the layer the factory builds). Environment: 无卡 (CPU, world_size == 1).

``moe_block`` exposes two public surfaces:

* ``create_moe_block`` -- a factory dispatching to ``MOEAllGatherLayerV2`` /
  ``MOEAlltoAllLayer`` (or raising for a bad mode). It forwards a *positional*
  argument list, and the two target classes have different signatures, so the
  real risk is a mis-mapped / dropped argument. These tests construct the real
  layers (no mock of the constructed class) and read back the observable
  attributes to prove every argument reached the right slot.
* ``MoEStatics`` -- holds the aux-free routing ``e_score_correction_bias`` and
  the ``expert_usage`` counter. The correction bias is a *selection* bias, not a
  routing weight, so at init it must be neutral (zeros); the tests below pin its
  group/expert layout, dtype, gradient / distributed flags and the equal-group
  contract with distinct dimensions so a shape swap cannot pass.

To connect the factory output to the routing maths the domain rules call out
(gating value, normalisation, routed scaling, and "correction bias is NOT the
final weight"), the created block itself is driven:

* ``fused_gate_logits_process`` on an alltoall block -- softmax/sigmoid gating
  value, and the ``group_experts`` path that divides each group by its own max
  (routed scaling: the resulting weights need NOT sum to 1).
* ``fused_gate_logits_process_fused`` on an allgather block wired with a real,
  non-zero ``MoEStatics`` bias -- the bias moves the top-k *selection* but the
  emitted combine weight is the *un-biased* probability at the chosen experts.

Explicitly NOT verified here (see report), because they need custom fused ops or
a real process group that CPU/world_size==1 cannot exercise:

* ``moe_gate_dispatch`` / ``moe_gate_dispatch_partial_nosoftmaxtopk`` dispatch
  and the ``norm_gate_logits`` combine-weight normalisation that sits behind
  them (paddle.incubate fused ops, GPU).
* The soft-gate combine (``GateCombine`` / ``moe_combine``), GPU only.
* Cross-rank AllToAll / AllGather expert dispatch and re-combine numerics; a
  single process cannot prove peer selection / split sizes / expert re-homing.
  Faking world_size>1 + mocking the collective would only assert "was called"
  and is rejected by the distributed-test rules (multi_card_tests owns that).

``dist.get_world_size`` / ``dist.get_rank`` are pinned to the honest single-rank
values (1 / 0) only during construction to select the world_size==1 local path;
this is NOT a fake multi-rank + mocked-collective setup.
"""

import unittest
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np

try:
    import paddle
    import paddle.nn.functional as F
    from paddle import nn

    _PADDLE_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - environment without paddle
    paddle = None
    nn = None
    F = None
    _PADDLE_IMPORT_ERROR = exc


_MODPATH_A = "paddlefleet.nn.moe.moe_alltoall_layer"


def _np_softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def _np_sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class _Cfg:
    """Minimal config object exposing both attribute access and ``.get``.

    The production config supports both styles; ``MoEStatics`` uses ``.get`` and
    ``.moe_num_experts`` while the layers read plain attributes.
    """

    def __init__(self, **kw):
        self._d = dict(kw)
        for key, val in kw.items():
            setattr(self, key, val)

    def get(self, key, default=None):
        return self._d.get(key, default)


class _NoHCGFleet:
    """A fleet stand-in without ``_hcg`` so ``is_mp_moe`` resolves to False."""


def _make_gate(in_dim, num_experts, scoring_func="softmax"):
    """A real ``nn.Linear`` gate carrying the attributes the layer reads.

    A concrete Linear (not a mock) is used so gate parameters really exist and
    get tagged by the constructor, and so ``gate.act`` is the real softmax /
    sigmoid that the fused gate-logit processing consumes.
    """
    gate = nn.Linear(in_dim, num_experts)
    gate.config = _Cfg(
        moe_use_hard_gate=False,
        norm_gate_logits=True,
        router_aux_loss_coef=0.0,
        moe_use_aux_free=True,
        moe_orthogonal_loss_lambda=0.0,
        router_z_loss_coef=0.0,
        moe_num_experts=num_experts,
        moe_world_size=1,
        moe_k=2,
        multimodel_experts=False,
    )
    gate.norm_gate_logits = True
    gate.act = F.sigmoid if scoring_func == "sigmoid" else F.softmax
    gate.experts_type_ids = None
    return gate


def _experts(num_experts, dim=4):
    experts = nn.LayerList([nn.Linear(dim, dim) for _ in range(num_experts)])
    for p in experts.parameters():
        p.expert = False
        p.no_sync = False
    return experts


@contextmanager
def _single_rank_construction():
    """Pin construction onto the honest world_size==1 local path.

    Only wraps construction: world_size / rank / num_local_experts are cached as
    attributes and none of the methods under test issues a collective.
    """
    with (
        patch(f"{_MODPATH_A}.dist.get_world_size", return_value=1),
        patch(f"{_MODPATH_A}.dist.get_rank", return_value=0),
        patch(f"{_MODPATH_A}.fleet.fleet", _NoHCGFleet()),
    ):
        yield


class _MoeBlockBase(unittest.TestCase):
    """Common setup: import the real module, pin CPU, build real layers."""

    def setUp(self):
        if _PADDLE_IMPORT_ERROR is not None:
            self.skipTest(f"paddle unavailable: {_PADDLE_IMPORT_ERROR!r}")
        paddle.set_device("cpu")
        try:
            from paddlefleet.nn.moe.moe_allgather_layer import (
                MOEAllGatherLayerV2,
            )
            from paddlefleet.nn.moe.moe_alltoall_layer import (
                MOEAlltoAllLayer,
            )
            from paddlefleet.nn.moe.moe_block import (
                MoEStatics,
                create_moe_block,
            )
        except ImportError as exc:
            # moe_block pulls in moe_allgather_layer, which imports several
            # paddle.incubate MoE ops + a LoRA quant layer at module load.
            # A precise ImportError (missing op/symbol) is a real capability
            # gap on this build and is recorded as a skip, not swallowed.
            self.skipTest(f"moe module import failed: {exc!r}")
        self.create_moe_block = create_moe_block
        self.MoEStatics = MoEStatics
        self.MOEAllGatherLayerV2 = MOEAllGatherLayerV2
        self.MOEAlltoAllLayer = MOEAlltoAllLayer

    def _create(
        self,
        *,
        num_experts=4,
        dim=4,
        in_dim=4,
        scoring_func="softmax",
        **kwargs,
    ):
        gate = _make_gate(in_dim, num_experts, scoring_func)
        experts = _experts(num_experts, dim=dim)
        with _single_rank_construction():
            return self.create_moe_block(
                gate=gate, experts=experts, layer_idx=0, **kwargs
            )


class TestCreateMoeBlockDispatch(_MoeBlockBase):
    """create_moe_block forwards its *positional* args to the right slots.

    The two target classes have different signatures, so the real risk is a
    mis-mapped or dropped argument. Rather than mock the constructed class and
    assert "was called", these build the real layer and read observable
    attributes back to prove each value reached the intended slot.
    """

    def test_allgather_positional_forwarding(self):
        block = self._create(
            num_experts=6,
            recompute=True,
            k=3,
            enable_reverse_token_drop=True,
            all_to_all_dropout=0.25,
            group_experts=True,
            use_expert_out_alltoall=False,
            use_padding=False,
            dense_token_type=7,
            moe_mode="allgather",
        )
        self.assertIsInstance(block, self.MOEAllGatherLayerV2)
        # Base-class slots.
        self.assertEqual(block.k, 3)
        self.assertTrue(block.recompute)
        self.assertEqual(block.all_to_all_dropout, 0.25)
        self.assertTrue(block.group_experts)
        self.assertEqual(block.num_local_experts, 6)
        self.assertFalse(block.use_correction_bias)
        # Allgather-only slots (dropped by the alltoall branch).
        self.assertTrue(block.enable_reverse_token_drop)
        self.assertFalse(block.use_expert_out_alltoall)
        self.assertFalse(block.use_padding)
        self.assertEqual(block.dense_token_type, 7)

    def test_alltoall_forwarding_and_dropped_params(self):
        block = self._create(
            num_experts=4,
            recompute=True,
            k=1,
            enable_reverse_token_drop=True,  # not forwarded to alltoall
            all_to_all_dropout=0.2,
            group_experts=False,
            use_padding=False,  # not forwarded to alltoall
            dense_token_type=9,  # not forwarded to alltoall
            moe_mode="alltoall",
        )
        self.assertIsInstance(block, self.MOEAlltoAllLayer)
        self.assertNotIsInstance(block, self.MOEAllGatherLayerV2)
        self.assertEqual(block.k, 1)
        self.assertTrue(block.recompute)
        self.assertEqual(block.all_to_all_dropout, 0.2)
        self.assertFalse(block.group_experts)
        self.assertEqual(block.num_local_experts, 4)
        # The alltoall constructor signature has no allgather-only params;
        # forwarding them would raise, and they must not appear as attributes.
        self.assertFalse(hasattr(block, "enable_reverse_token_drop"))
        self.assertFalse(hasattr(block, "use_padding"))
        self.assertFalse(hasattr(block, "dense_token_type"))
        self.assertFalse(hasattr(block, "use_expert_out_alltoall"))

    def test_moe_statics_enables_correction_bias(self):
        cfg = _Cfg(moe_num_experts=4, multimodel_experts=False)
        statics = self.MoEStatics(cfg, layer_idx=0)
        block = self._create(
            num_experts=4, moe_statics=statics, moe_mode="alltoall"
        )
        # moe_statics reached the constructor slot -> correction bias on,
        # and the same object is retained.
        self.assertTrue(block.use_correction_bias)
        self.assertIs(block.moe_statics, statics)

    def test_invalid_mode_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            self._create(moe_mode="nonsense")
        self.assertIn("Invalid moe_mode", str(ctx.exception))


class TestMoEStatics(_MoeBlockBase):
    """MoEStatics holds the aux-free *selection* bias and the usage counter.

    The correction bias is a selection bias, not a routing weight, so at init
    it must be neutral (all zeros). Layout, dtype, gradient/distributed flags
    and the equal-group contract are pinned with distinct group vs expert
    dimensions so a shape swap cannot pass.
    """

    def test_bias_neutral_zero_init_and_shape(self):
        cfg = _Cfg(moe_num_experts=4, multimodel_experts=False)
        statics = self.MoEStatics(cfg, layer_idx=0)
        bias = statics.e_score_correction_bias
        self.assertEqual(list(bias.shape), [1, 4])
        # Neutral at init: a non-zero init would silently bias top-k selection.
        np.testing.assert_array_equal(
            bias.numpy(), np.zeros([1, 4], dtype="float32")
        )

    def test_distinct_group_and_expert_dims(self):
        # 3 groups x 8 experts: a [groups, experts] <-> [experts, groups]
        # swap would give [8, 3] and fail this exact-shape check.
        cfg = _Cfg(moe_num_experts=[8, 8, 8], multimodel_experts=True)
        statics = self.MoEStatics(cfg, layer_idx=2)
        self.assertEqual(list(statics.e_score_correction_bias.shape), [3, 8])
        self.assertEqual(list(statics.expert_usage.shape), [3, 8])

    def test_dtypes(self):
        cfg = _Cfg(moe_num_experts=4, multimodel_experts=False)
        statics = self.MoEStatics(cfg, layer_idx=0)
        self.assertEqual(statics.e_score_correction_bias.dtype, paddle.float32)
        self.assertEqual(statics.expert_usage.dtype, paddle.int64)

    def test_flags_stop_gradient_and_distributed(self):
        cfg = _Cfg(moe_num_experts=4, multimodel_experts=False)
        statics = self.MoEStatics(cfg, layer_idx=0)
        self.assertTrue(statics.e_score_correction_bias.stop_gradient)
        self.assertTrue(statics.expert_usage.stop_gradient)
        self.assertTrue(statics.e_score_correction_bias.is_distributed)

    def test_expert_usage_zeros(self):
        cfg = _Cfg(moe_num_experts=4, multimodel_experts=False)
        statics = self.MoEStatics(cfg, layer_idx=0)
        np.testing.assert_array_equal(
            statics.expert_usage.numpy(), np.zeros([1, 4], dtype="int64")
        )

    def test_multimodel_unequal_group_size_asserts(self):
        cfg = _Cfg(moe_num_experts=[4, 8, 4], multimodel_experts=True)
        with self.assertRaises(AssertionError):
            self.MoEStatics(cfg, layer_idx=0)


class TestCreatedBlockRoutes(_MoeBlockBase):
    """Drive the factory-created block's routing math (gating value / scaling).

    These run the real ``fused_gate_logits_process`` reached *through*
    ``create_moe_block`` (all CPU, no fused op), with independent numpy
    references, to prove the factory wired gate.act / k / group_experts into a
    working routing path.
    """

    def test_softmax_gating_value_sums_to_one(self):
        block = self._create(
            num_experts=4,
            k=2,
            group_experts=False,
            scoring_func="softmax",
            moe_mode="alltoall",
        )
        logits = np.array(
            [[2.0, 0.0, -1.0, 1.0], [0.5, 0.5, 3.0, -2.0]], dtype="float32"
        )
        prob, max_prob = block.fused_gate_logits_process(
            paddle.to_tensor(logits)
        )
        self.assertIsNone(max_prob)
        np.testing.assert_allclose(
            prob.numpy(), _np_softmax(logits, axis=-1), rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            prob.numpy().sum(axis=-1), np.ones(2), rtol=1e-5, atol=1e-6
        )

    def test_sigmoid_gating_value_not_normalised(self):
        block = self._create(
            num_experts=4,
            k=2,
            group_experts=False,
            scoring_func="sigmoid",
            moe_mode="alltoall",
        )
        logits = np.array(
            [[2.0, 0.0, -1.0, 1.0], [0.5, 0.5, 3.0, -2.0]], dtype="float32"
        )
        prob, max_prob = block.fused_gate_logits_process(
            paddle.to_tensor(logits)
        )
        self.assertIsNone(max_prob)
        # Independent reference: elementwise sigmoid, which does NOT sum to 1.
        np.testing.assert_allclose(
            prob.numpy(), _np_sigmoid(logits), rtol=1e-5, atol=1e-6
        )
        row_sums = prob.numpy().sum(axis=-1)
        self.assertFalse(np.allclose(row_sums, np.ones(2), atol=1e-3))

    def test_group_experts_routed_scaling_need_not_sum_to_one(self):
        # num_experts = k * per_group so reshape [s, k, -1] is exact.
        num_experts, k, s = 8, 2, 2
        block = self._create(
            num_experts=num_experts,
            k=k,
            group_experts=True,
            scoring_func="softmax",
            moe_mode="alltoall",
        )
        logits = np.array(
            [
                [2.0, 0.0, -1.0, 1.0, 0.5, 0.5, 3.0, -2.0],
                [1.0, -1.0, 0.0, 2.0, -0.5, 0.5, 1.5, 0.2],
            ],
            dtype="float32",
        )
        prob, max_prob = block.fused_gate_logits_process(
            paddle.to_tensor(logits)
        )
        # Independent reference: softmax within each size-4 group, then divide
        # each group by its own max (the routed-scaling step).
        grouped = logits.reshape([s, k, -1])
        sm = _np_softmax(grouped, axis=-1)
        gmax = np.max(sm, axis=-1, keepdims=True)
        ref_prob = (sm / gmax).reshape([s, num_experts])

        self.assertIsNotNone(max_prob)
        self.assertEqual(list(max_prob.shape), [s, k, 1])
        np.testing.assert_allclose(max_prob.numpy(), gmax, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(prob.numpy(), ref_prob, rtol=1e-5, atol=1e-6)
        # The per-group max entry normalises to exactly 1.0 (a regression that
        # forgot the /max would leave plain softmax with no 1.0 entry).
        reshaped = prob.numpy().reshape([s, k, -1])
        np.testing.assert_allclose(
            reshaped.max(axis=-1), np.ones([s, k]), rtol=1e-5, atol=1e-6
        )
        # Routed scaling: the emitted weights need NOT sum to 1 per row.
        self.assertFalse(
            np.allclose(prob.numpy().sum(axis=-1), np.ones(s), atol=1e-2)
        )


class TestCorrectionBiasNotFinalWeight(_MoeBlockBase):
    """The aux-free correction bias moves *selection*, not the emitted weight.

    Wire an allgather block with a real, non-zero ``MoEStatics`` bias and run
    the real ``fused_gate_logits_process_fused``. The bias must shift which
    experts win the top-k, but the combine weight handed back must be the
    *un-biased* probability at the chosen experts -- the exact contract the
    domain rules call out ("correction bias is NOT the final weight").
    """

    def test_bias_shifts_selection_but_weight_is_unbiased_prob(self):
        num_experts, k = 4, 2
        cfg = _Cfg(moe_num_experts=num_experts, multimodel_experts=False)
        statics = self.MoEStatics(cfg, layer_idx=0)
        block = self._create(
            num_experts=num_experts,
            k=k,
            group_experts=False,
            moe_statics=statics,
            moe_mode="allgather",
        )
        self.assertTrue(block.use_correction_bias)

        # Non-zero bias only on expert 2, large enough to force it into top-k.
        statics.e_score_correction_bias.set_value(
            paddle.to_tensor([[0.0, 0.0, 5.0, 0.0]], dtype="float32")
        )
        logits = np.array([[3.0, 2.0, 0.0, -1.0]], dtype="float32")
        sm = _np_softmax(logits, axis=-1)[0]  # unbiased reference

        try:
            lm_we, prob_ret, _ = block.fused_gate_logits_process_fused(
                paddle.to_tensor(logits)
            )
        except (RuntimeError, NotImplementedError, OSError) as exc:
            # expand_modality_expert_id is a paddle.incubate fused op that may
            # not be built for CPU on this environment; the selection/weight
            # numerics then cannot run here (recorded, not swallowed).
            self.skipTest(f"fused expert-id op unavailable on CPU: {exc!r}")

        # The returned full probability is the plain (un-biased) softmax.
        np.testing.assert_allclose(
            prob_ret.numpy()[0], sm, rtol=1e-5, atol=1e-6
        )
        lm = lm_we.numpy()[0]
        weights = lm[:k]
        ids = lm[k:].astype("int64")

        # Bias moved selection: unbiased top-2 would be {0, 1}; expert 2 wins.
        self.assertIn(2, list(ids))
        # Each emitted weight is the UN-biased prob at the chosen expert...
        for i in range(k):
            np.testing.assert_allclose(
                weights[i], sm[ids[i]], rtol=1e-5, atol=1e-6
            )
        # ...and specifically NOT the biased score for expert 2 (~5.03).
        j = list(ids).index(2)
        self.assertLess(weights[j], 1.0)
        self.assertFalse(np.isclose(weights[j], sm[2] + 5.0, atol=1e-2))


class TestOutOfScopeNumerics(_MoeBlockBase):
    """Explicitly-recorded skips for paths a single CPU process cannot prove.

    These are documented as visible skips (not silently omitted) so the report
    of what is / is not covered stays honest; the numerics belong to GPU fused
    ops or a real multi-rank process group (owned by multi_card_tests).
    """

    def test_moe_gate_dispatch_and_norm_combine_weights(self):
        self.skipTest(
            "moe_gate_dispatch + norm_gate_logits combine-weight "
            "normalisation are paddle.incubate fused ops (GPU); not "
            "exercisable on CPU/world_size==1."
        )

    def test_soft_gate_combine(self):
        self.skipTest(
            "GateCombine / moe_combine soft combine is a GPU incubate op; "
            "not exercisable on CPU."
        )

    def test_cross_rank_dispatch_and_recombine(self):
        self.skipTest(
            "Cross-rank AllToAll / AllGather expert dispatch+recombine needs "
            "a real process group; a single process cannot prove peer "
            "selection / split sizes / expert re-homing, and faking "
            "world_size>1 + mocking the collective is rejected by the "
            "distributed-test rules (owned by tests/multi_card_tests)."
        )


if __name__ == "__main__":
    unittest.main()
