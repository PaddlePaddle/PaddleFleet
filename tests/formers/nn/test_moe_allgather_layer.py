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

"""No-card (CPU) behavior tests for the SonicMoE AllGather MoE layer.

Scope and evidence boundary (分布式训练 module, MoE AllGather):
  - ``ReshardCombineWeight`` is a PyLayer that converts expert-partitioned
    combine weights to sequence-partitioned form. On its forward it records
    which positions were zeroed out (positions NOT owned by the local rank's
    experts) and on backward it must NOT propagate gradient back into those
    non-local positions. That local mask/masked_fill contract is pure paddle
    arithmetic and is verified here on CPU with REAL autograd, using a genuine
    single-rank process group (``Group(0, None, [0])``) so that the underlying
    ``reduce_scatter_group`` / ``all_gather_group`` collectives take their real
    ``nranks == 1`` local path (a true clone), NOT a mocked collective.
  - The cross-rank numerics of AllGather + ReduceScatter-sum (each rank holds a
    different expert intermediate shard and the shards are summed across ranks)
    depend on multiple ranks each holding distinct data. Those are deliberately
    NOT asserted here; a single process cannot exercise real inter-rank
    reduce-scatter/all-gather. They require a multi-card run with a real process
    group (see the 分布式训练 multi-card requirements). This file therefore
    validates only the local single-rank control/mask logic and the layer
    configuration/derivation logic, and explicitly claims nothing about
    inter-rank collective numerics.
"""

import unittest
from unittest import mock

import numpy as np
import paddle
import paddle.nn as nn
from paddle.distributed.communication.group import Group

from paddlefleet.nn.moe.abstract import MOELayerBase
from paddlefleet.nn.moe.moe_allgather_layer import (
    MOEAllGatherLayerV2,
    ReshardCombineWeight,
)
from paddlefleet.nn.moe.moe_alltoall_layer import MOEAlltoAllLayer


def _single_rank_group():
    """A real, nranks==1 group.

    ``reduce_scatter_group`` and ``all_gather_group`` both short-circuit to a
    plain ``input.clone()`` when ``group.nranks == 1``. This is the genuine
    local code path (not a mock), so we can exercise the real forward/backward
    of ``ReshardCombineWeight`` on CPU without a distributed launcher.
    """
    return Group(0, None, [0])


class _Ctx:
    """Minimal stand-in for the PyLayer context holder.

    PyLayer only uses ``ctx`` as an attribute bag between forward and backward.
    A plain object (not a MagicMock) is used so that reads of ``ctx.mask`` /
    ``ctx.group`` return exactly what forward stored, with no auto-magic.
    """


class ReshardCombineWeightMaskTest(unittest.TestCase):
    """Local (single-rank) forward/backward contract of ReshardCombineWeight."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_forward_is_identity_and_records_ownership_mask(self):
        # Distinguishable values; the zero rows/cells mark positions that are
        # NOT owned by this rank's experts (they were zeroed upstream).
        data = [[1.0, 2.0], [0.0, 0.0], [3.0, 4.0], [5.0, 0.0]]
        x = paddle.to_tensor(data, dtype="float32")
        group = _single_rank_group()
        ctx = _Ctx()

        out = ReshardCombineWeight.forward(ctx, x, group=group)

        # nranks==1 reduce_scatter is a local clone -> values preserved exactly.
        np.testing.assert_array_equal(out.numpy(), np.asarray(data, np.float32))
        # The stored mask must mark exactly the zero positions (non-local
        # experts), position by position -- not merely "some" or a count.
        expected_mask = np.asarray(data, np.float32) == 0.0
        np.testing.assert_array_equal(ctx.mask.numpy(), expected_mask)
        self.assertIs(ctx.group, group)

    def test_backward_zeroes_gradient_at_nonlocal_positions(self):
        # Drive REAL autograd through the PyLayer on the single-rank path.
        input_np = np.array([[1.0, 0.0, 3.0], [0.0, 5.0, 6.0]], dtype="float32")
        x = paddle.to_tensor(input_np)
        x.stop_gradient = False
        group = _single_rank_group()

        out = ReshardCombineWeight.apply(x, group=group)
        # Non-uniform, all-nonzero upstream grad so masking is observable and
        # a wrong fill value / inverted mask cannot hide.
        upstream_np = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32"
        )
        out.backward(paddle.to_tensor(upstream_np))

        # Independent expectation: gradient survives only where the forward
        # input was non-zero (owned by this rank); elsewhere it must be 0.
        expected = np.where(input_np == 0.0, 0.0, upstream_np)
        self.assertIsNotNone(x.grad)
        np.testing.assert_array_equal(x.grad.numpy(), expected)

        # Guard against a vacuous test: at least one position is zeroed and at
        # least one retains the full upstream gradient.
        self.assertTrue((expected == 0.0).any())
        self.assertTrue((expected == upstream_np).any())

    def test_backward_all_local_preserves_full_gradient(self):
        # No zeros -> every position is owned locally -> mask all False ->
        # gradient passes through unchanged.
        input_np = np.array([[7.0, 8.0], [9.0, 10.0]], dtype="float32")
        x = paddle.to_tensor(input_np)
        x.stop_gradient = False
        out = ReshardCombineWeight.apply(x, group=_single_rank_group())
        upstream_np = np.array([[2.0, 3.0], [4.0, 5.0]], dtype="float32")
        out.backward(paddle.to_tensor(upstream_np))
        np.testing.assert_array_equal(x.grad.numpy(), upstream_np)

    def test_backward_all_nonlocal_zeroes_entire_gradient(self):
        # All zeros -> no position owned locally -> mask all True -> gradient
        # fully suppressed.
        x = paddle.zeros([3, 4], dtype="float32")
        x.stop_gradient = False
        out = ReshardCombineWeight.apply(x, group=_single_rank_group())
        upstream = paddle.arange(12, dtype="float32").reshape([3, 4]) + 1.0
        out.backward(upstream)
        np.testing.assert_array_equal(
            x.grad.numpy(), np.zeros([3, 4], dtype="float32")
        )

    def test_backward_direct_consumes_stored_mask_and_group(self):
        # Direct static-call form: prove backward reads ctx.mask / ctx.group
        # and applies masked_fill at exactly the recorded positions.
        ctx = _Ctx()
        ctx.group = _single_rank_group()
        ctx.mask = paddle.to_tensor(
            [[False, True], [True, False], [False, False]], dtype="bool"
        )
        grad_np = np.array(
            [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype="float32"
        )
        result = ReshardCombineWeight.backward(ctx, paddle.to_tensor(grad_np))
        expected = np.where(ctx.mask.numpy(), 0.0, grad_np)
        np.testing.assert_array_equal(result.numpy(), expected)


class MOEAllGatherLayerV2ConfigTest(unittest.TestCase):
    """Configuration / derivation logic of MOEAllGatherLayerV2.

    The distributed infra collaborators (world size / rank / fleet hcg) are
    replaced with a single-process (world_size == 1) stub. This exercises only
    the constructor's local derivation logic; it makes no claim about multi-rank
    expert sharding, which requires a real process group.
    """

    def setUp(self):
        paddle.set_device("cpu")

    def _make_gate(self):
        gate = mock.MagicMock()
        gate.config = mock.MagicMock()
        gate.config.router_aux_loss_coef = 0.0
        gate.config.moe_use_aux_free = True
        gate.config.moe_use_hard_gate = False
        gate.config.norm_gate_logits = False
        gate.config.moe_orthogonal_loss_lambda = 0.0
        gate.config.router_z_loss_coef = 0.0
        gate.config.moe_world_size = 1
        gate.config.moe_rank = 0
        gate.config.sequence_parallel = False
        gate.num_experts_tensor = paddle.to_tensor(2, dtype="int64")
        gate.parameters.return_value = []
        return gate

    def _build(self, num_experts, moe_num_experts=None, **overrides):
        experts = nn.LayerList([nn.Linear(4, 4) for _ in range(num_experts)])

        class _NoHCGFleet:
            pass

        with (
            mock.patch(
                "paddlefleet.nn.moe.moe_alltoall_layer.dist.get_world_size",
                return_value=1,
            ),
            mock.patch(
                "paddlefleet.nn.moe.moe_alltoall_layer.dist.get_rank",
                return_value=0,
            ),
            mock.patch(
                "paddlefleet.nn.moe.moe_alltoall_layer.fleet.fleet",
                _NoHCGFleet(),
            ),
        ):
            return MOEAllGatherLayerV2(
                gate=self._make_gate(),
                experts=experts,
                layer_idx=0,
                moe_num_experts=moe_num_experts,
                **overrides,
            )

    def test_multimodal_flag_tracks_expert_partition_shape(self):
        # A multi-entry expert partition (e.g. [lm=4, mm=2]) is multimodal;
        # a single-entry list or None is not. Only this one input flips the
        # flag, so a broken predicate is caught by the contrast.
        self.assertTrue(
            self._build(6, moe_num_experts=[4, 2]).multimodal_experts
        )
        self.assertFalse(self._build(4, moe_num_experts=[4]).multimodal_experts)
        self.assertFalse(
            self._build(2, moe_num_experts=None).multimodal_experts
        )

    def test_num_local_experts_derivation_single_rank(self):
        layer = self._build(6, moe_num_experts=[6])
        # world_size == 1 -> all experts are local.
        self.assertEqual(layer.num_local_experts, 6)

    def test_control_flags_default_and_override(self):
        default = self._build(2)
        self.assertFalse(default.enable_reverse_token_drop)
        self.assertTrue(default.use_padding)
        self.assertTrue(default.use_expert_out_alltoall)
        self.assertEqual(default.dense_token_type, 3)

        custom = self._build(
            2,
            enable_reverse_token_drop=True,
            use_padding=False,
            use_expert_out_alltoall=False,
            dense_token_type=5,
        )
        self.assertTrue(custom.enable_reverse_token_drop)
        self.assertFalse(custom.use_padding)
        self.assertFalse(custom.use_expert_out_alltoall)
        self.assertEqual(custom.dense_token_type, 5)

    def test_inheritance_selects_allgather_dispatch_path(self):
        # The AllGather layer must remain a subclass of the AlltoAll layer and
        # the abstract MoE base; this is the contract that routes forward()
        # through the AllGather implementation.
        self.assertTrue(issubclass(MOEAllGatherLayerV2, MOEAlltoAllLayer))
        self.assertTrue(issubclass(MOEAllGatherLayerV2, MOELayerBase))


if __name__ == "__main__":
    unittest.main()
