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

"""Behaviour tests for paddlefleet.nn.moe_deepep.modular_moe_layer.ModularMoELayer.

Module: model layer (MoE), environment: no-card (CPU). We keep the real
ModularMoELayer construction, the real StandardMoEGate routing math and the
real ``_forward_traditional_moe`` orchestration in the verification chain, and
compare against independently hand-derived references (numpy softmax/top-k, and
per-token weighted expert sums computed by calling the already-constructed
expert layers directly).

Isolation / scope notes:
* fleet / paddle.distributed are patched only to force the *single-rank*
  (expert_model_parallel_size == 1, no expert-parallel group) construction
  path. This is the world-size==1 fallback the layer explicitly supports; no
  collective is executed and no multi-card numeric claim is made.
* tensor_model_parallel_size is fixed to 1 so ``Linear.create`` resolves to the
  CPU-capable ``paddle.nn.Linear`` for both the gate projection and the expert
  MLPs, keeping the forward math CPU-executable.
* The EP paths (AllToAllMoECommunication / DeepEPMoECommunication) require a
  real multi-rank process group (and, for DeepEP, GPU kernels); their
  dispatch/combine numerics are NOT verified here and are covered by explicit
  skips below rather than by faking world_size + mocking collectives.
"""

import unittest
from unittest.mock import patch

import numpy as np
import paddle

from paddlefleet.nn.mlp import MLP
from paddlefleet.nn.moe_deepep.modular_moe_layer import ModularMoELayer
from paddlefleet.transformers.configuration_utils import PretrainedConfig

_MODULE = "paddlefleet.nn.moe_deepep.modular_moe_layer"


def _softmax(logits):
    """Independent (numpy) softmax over the last axis."""
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _make_layer(
    hidden_size=8,
    moe_intermediate_size=16,
    num_experts=4,
    num_shared_experts=0,
    num_experts_per_tok=2,
    norm_topk_prob=True,
    topk_method="greedy",
    model_type="qwen2_moe",
    transpose_gate_weight=False,
    moe_token_dispatcher_type="alltoall",
    n_group=1,
    topk_group=1,
    router_aux_loss_coef=0.0,
    moe_expert_capacity_factor=0.0,
    expert_class=MLP,
):
    """Construct a real ModularMoELayer on the single-rank CPU path.

    fleet/dist are forced into the "no expert parallel" branch. The expert
    class is the real production ``MLP`` (StandardMLPExpert subclasses it);
    with tensor_model_parallel_size == 1 every Linear is a plain CPU
    ``nn.Linear`` so the whole layer is CPU-executable.
    """
    pretrained_config = PretrainedConfig(
        hidden_size=hidden_size,
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        max_seq_len=128,
        hidden_act="silu",
        moe_token_dispatcher_type=moe_token_dispatcher_type,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=1.0,
        router_aux_loss_coef=router_aux_loss_coef,
        moe_subbatch_token_num_before_dispatch=-1,
        moe_expert_capacity_factor=moe_expert_capacity_factor,
        moe_token_drop_policy="probs",
    )
    moe_config = {
        "gate_activation": "softmax",
        "eval_capacity_factor": 1.0,
        "group": None,
        "global_aux_loss": False,
        "use_rts": True,
        "top2_2nd_expert_sampling": True,
        "seq_aux": True,
        "train_topk_method": topk_method,
        "inference_topk_method": topk_method,
        "use_flexible_loss": False,
        "expert_dropout": 0.0,
        "loss_configs": None,
        "loss_combiner_name": "weighted_sum",
    }

    with (
        patch(f"{_MODULE}.fleet") as mock_fleet,
        patch(f"{_MODULE}.dist") as mock_dist,
    ):
        mock_fleet.get_hybrid_communicate_group.side_effect = Exception(
            "no fleet"
        )
        mock_dist.get_world_size.return_value = 1
        layer = ModularMoELayer(
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            num_experts_per_tok=num_experts_per_tok,
            norm_topk_prob=norm_topk_prob,
            expert_activation="silu",
            moe_config=moe_config,
            model_type=model_type,
            expert_class=expert_class,
            transpose_gate_weight=transpose_gate_weight,
            pretrained_config=pretrained_config,
        )
    return layer


class TestConstructionAndWiring(unittest.TestCase):
    """Single-rank construction, config propagation and expert wiring."""

    def setUp(self):
        paddle.set_device("cpu")
        paddle.seed(0)

    def test_single_rank_expert_parallel_state(self):
        """_init_expert_parallel else-branch: no EP group, rank 0, all experts local."""
        layer = _make_layer(num_experts=6)
        self.assertEqual(layer.expert_model_parallel_size, 1)
        self.assertEqual(layer.moe_rank, 0)
        self.assertIsNone(layer.moe_group)
        # On a single device every expert lives locally.
        self.assertEqual(layer.num_experts_per_device, 6)
        self.assertIsNone(layer.token_dispatcher)

    def test_all_experts_populated_and_wired_single_rank(self):
        """Experts LayerList is full (no None holes) and each MLP is wired to
        moe_intermediate_size with fused up/gate projection."""
        hidden, inter, n = 8, 16, 4
        layer = _make_layer(
            hidden_size=hidden, moe_intermediate_size=inter, num_experts=n
        )
        self.assertEqual(len(layer.experts), n)
        for i in range(n):
            expert = layer.experts[i]
            # i // num_experts_per_device == moe_rank(0) for all i -> populated.
            self.assertIsNotNone(expert)
            self.assertIsInstance(expert, MLP)
            self.assertEqual(expert.intermediate_size, inter)
            self.assertTrue(expert.fuse_up_gate)
            # Fused up/gate: [hidden, inter*2]; down: [inter, hidden].
            self.assertEqual(
                list(expert.up_gate_proj.weight.shape), [hidden, inter * 2]
            )
            self.assertEqual(
                list(expert.down_proj.weight.shape), [inter, hidden]
            )

    def test_gate_constructed_with_routing_params(self):
        """Gate is a StandardMoEGate carrying the routing config; weight is
        [hidden, num_experts] when not transposed."""
        from paddlefleet.nn.moe_deepep.moe_gate import StandardMoEGate

        layer = _make_layer(
            hidden_size=8,
            num_experts=4,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            topk_method="greedy",
        )
        self.assertIsInstance(layer.gate, StandardMoEGate)
        self.assertEqual(layer.gate.num_experts, 4)
        self.assertEqual(layer.gate.num_experts_per_tok, 2)
        self.assertEqual(layer.gate.topk_method, "greedy")
        self.assertTrue(layer.gate.norm_topk_prob)
        self.assertEqual(list(layer.gate.weight.shape), [8, 4])

    def test_gate_weight_transposed_shape(self):
        """transpose_gate_weight flips the stored gate weight orientation."""
        layer = _make_layer(
            hidden_size=8, num_experts=4, transpose_gate_weight=True
        )
        self.assertTrue(layer.gate.transpose_gate_weight)
        self.assertEqual(list(layer.gate.weight.shape), [4, 8])

    def test_shared_experts_created_only_when_requested(self):
        """num_shared_experts controls presence and hidden width of the shared MLP."""
        inter = 16
        none_layer = _make_layer(
            moe_intermediate_size=inter, num_shared_experts=0
        )
        self.assertIsNone(none_layer.shared_experts)

        shared_layer = _make_layer(
            moe_intermediate_size=inter, num_shared_experts=3
        )
        self.assertIsNotNone(shared_layer.shared_experts)
        self.assertIsInstance(shared_layer.shared_experts, MLP)
        # Shared expert width scales with the shared-expert count.
        self.assertEqual(
            shared_layer.shared_experts.intermediate_size, inter * 3
        )

    def test_communication_type_selection(self):
        """dispatcher_type selects the communication class; invalid -> ValueError."""
        from paddlefleet.nn.moe_deepep.moe_communication import (
            AllToAllMoECommunication,
            DeepEPMoECommunication,
        )

        a2a = _make_layer(moe_token_dispatcher_type="alltoall")
        self.assertIsInstance(a2a.communication, AllToAllMoECommunication)

        deepep = _make_layer(moe_token_dispatcher_type="deepep")
        self.assertIsInstance(deepep.communication, DeepEPMoECommunication)

        with self.assertRaises(ValueError):
            _make_layer(moe_token_dispatcher_type="invalid")

    def test_drop_tokens_flag_follows_capacity_factor(self):
        """drop_tokens is enabled iff a non-zero expert capacity factor is set."""
        no_drop = _make_layer(moe_expert_capacity_factor=0.0)
        self.assertFalse(no_drop.drop_tokens)
        self.assertFalse(no_drop.gate.drop_tokens)

        drop = _make_layer(moe_expert_capacity_factor=1.0)
        self.assertTrue(drop.drop_tokens)
        self.assertTrue(drop.gate.drop_tokens)

    def test_get_expert_info_reports_real_state(self):
        """get_expert_info reflects the actual single-rank layer state (exact values)."""
        layer = _make_layer(num_experts=4)
        info = layer.get_expert_info()
        self.assertEqual(info["num_experts"], 4)
        self.assertEqual(info["num_experts_per_device"], 4)
        self.assertEqual(info["expert_model_parallel_size"], 1)
        self.assertEqual(info["moe_rank"], 0)
        self.assertFalse(info["is_parallel_enabled"])
        self.assertFalse(info["use_flexible_loss"])

    def test_loss_control_methods_warn_and_noop_without_flexible_loss(self):
        """With use_flexible_loss=False the loss-control entrypoints must warn
        and return without mutating the gate (control-flag guard)."""
        layer = _make_layer()
        self.assertFalse(layer.use_flexible_loss)
        for call in (
            lambda: layer.remove_loss_function("aux"),
            lambda: layer.update_loss_weights({"aux": 0.1}),
            lambda: layer.set_loss_combiner("sum"),
        ):
            with self.assertLogs(_MODULE, level="WARNING"):
                call()


class TestGateRouting(unittest.TestCase):
    """Real StandardMoEGate top-k routing math against an independent numpy oracle."""

    def setUp(self):
        paddle.set_device("cpu")
        paddle.seed(0)

    def test_greedy_topk_selection_and_normalized_weights(self):
        """logits = x @ W ; scores = softmax(logits) ; select top-k ; renormalise.

        The gate weight is pinned to a fixed matrix chosen so the per-token
        scores are strictly separated (no ties). We independently recompute the
        selection and the sum-to-one renormalised weights in numpy and compare
        against the gate outputs, and assert that a swapped selection would
        genuinely differ.
        """
        hidden, num_experts, k = 3, 4, 2
        layer = _make_layer(
            hidden_size=hidden,
            num_experts=num_experts,
            num_experts_per_tok=k,
            norm_topk_prob=True,
            topk_method="greedy",
        )

        W = np.array(
            [
                [2.0, -1.0, 0.5, 0.2],
                [0.0, 1.5, -0.5, 1.0],
                [1.0, 0.0, 2.0, -0.5],
            ],
            dtype=np.float32,
        )
        layer.gate.weight.set_value(
            paddle.to_tensor(W, dtype=layer.gate.weight.dtype)
        )
        x = np.array([[1.0, 0.0, -1.0], [0.5, 1.0, 0.5]], dtype=np.float32)

        # Independent reference.
        logits = x @ W
        scores = _softmax(logits)
        order = np.argsort(-scores, axis=-1)
        idx_ref = order[:, :k]
        w_sel = np.take_along_axis(scores, idx_ref, axis=-1)
        w_ref = w_sel / w_sel.sum(axis=-1, keepdims=True)  # routed_scaling=1.0
        mask_ref = np.zeros((x.shape[0], num_experts), dtype=np.float32)
        np.put_along_axis(mask_ref, idx_ref, 1.0, axis=-1)

        (_, top_gate, top_idx, _, mask, _, _, _) = layer.gate(
            paddle.to_tensor(x)
        )

        self.assertEqual(top_idx.numpy().tolist(), idx_ref.tolist())
        np.testing.assert_allclose(
            top_gate.numpy(), w_ref, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_array_equal(mask.numpy(), mask_ref)
        # Renormalised weights sum to one per token.
        np.testing.assert_allclose(
            top_gate.numpy().sum(axis=-1), np.ones(x.shape[0]), atol=1e-6
        )
        # Guard: reversing the selected pair would not match the reference.
        self.assertGreater(
            np.abs(idx_ref - idx_ref[:, ::-1]).max(),
            0,
            "fixture must select two distinct experts per token",
        )

    def test_norm_topk_prob_off_keeps_raw_softmax_weights(self):
        """With norm_topk_prob=False the selected weights are the raw softmax
        probabilities (times routed_scaling_factor=1.0), not renormalised."""
        hidden, num_experts, k = 3, 4, 2
        layer = _make_layer(
            hidden_size=hidden,
            num_experts=num_experts,
            num_experts_per_tok=k,
            norm_topk_prob=False,
            topk_method="greedy",
        )
        W = np.array(
            [
                [2.0, -1.0, 0.5, 0.2],
                [0.0, 1.5, -0.5, 1.0],
                [1.0, 0.0, 2.0, -0.5],
            ],
            dtype=np.float32,
        )
        layer.gate.weight.set_value(
            paddle.to_tensor(W, dtype=layer.gate.weight.dtype)
        )
        x = np.array([[1.0, 0.0, -1.0], [0.5, 1.0, 0.5]], dtype=np.float32)

        scores = _softmax(x @ W)
        idx_ref = np.argsort(-scores, axis=-1)[:, :k]
        w_ref = np.take_along_axis(scores, idx_ref, axis=-1)

        (_, top_gate, top_idx, _, _, _, _, _) = layer.gate(paddle.to_tensor(x))
        self.assertEqual(top_idx.numpy().tolist(), idx_ref.tolist())
        np.testing.assert_allclose(
            top_gate.numpy(), w_ref, rtol=1e-5, atol=1e-6
        )
        # Raw softmax subsets do not sum to one in general (distinguishes the
        # normalised branch from the un-normalised one).
        self.assertTrue(
            (np.abs(w_ref.sum(axis=-1) - 1.0) > 1e-3).any(),
            "fixture must exercise the un-normalised branch",
        )


class TestForwardOrchestration(unittest.TestCase):
    """Real _forward_traditional_moe / forward orchestration on CPU.

    References are built by calling the already-constructed expert (and shared
    expert) layers directly as fixed functions, so the gather / per-slot
    weighting / scatter-add orchestration is verified independently of its own
    implementation.
    """

    def setUp(self):
        paddle.set_device("cpu")
        paddle.seed(0)

    def _routed_reference(self, layer, tokens_2d, sel, weights):
        """out[t] = sum_s weights[t,s] * experts[sel[t,s]](tokens[t])."""
        t_np = tokens_2d.numpy()
        d = t_np.shape[1]
        out = np.zeros_like(t_np)
        for t in range(t_np.shape[0]):
            row = paddle.to_tensor(t_np[t : t + 1])
            acc = np.zeros(d, dtype=t_np.dtype)
            for s in range(sel.shape[1]):
                e = int(sel[t, s])
                y = layer.experts[e](row).numpy().reshape(-1)
                acc += float(weights[t, s]) * y
            out[t] = acc
        return out

    def test_forward_traditional_moe_routes_weights_and_scatters(self):
        """Direct _forward_traditional_moe: controlled routing incl. an expert
        that receives no token (skip branch); reference via direct expert calls."""
        hidden, inter, n, k = 8, 16, 4, 2
        layer = _make_layer(
            hidden_size=hidden,
            moe_intermediate_size=inter,
            num_experts=n,
            num_experts_per_tok=k,
        )
        x = paddle.to_tensor(
            np.linspace(-1.0, 1.0, 4 * hidden, dtype=np.float32).reshape(
                [4, hidden]
            )
        )
        # expert 3 receives no token -> tokens_per_expert[3] == 0 -> skipped.
        sel = np.array([[0, 1], [1, 2], [0, 2], [0, 2]], dtype=np.int64)
        weights = np.array(
            [[0.7, 0.3], [0.4, 0.6], [0.5, 0.5], [0.2, 0.8]], dtype=np.float32
        )

        out = layer._forward_traditional_moe(
            x,
            paddle.to_tensor(sel),
            paddle.to_tensor(weights),
        ).numpy()
        ref = self._routed_reference(layer, x, sel, weights)
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)

        # Guard: mis-weighting the slots produces a materially different result,
        # proving the per-slot topk weights are actually consumed.
        bad_ref = self._routed_reference(layer, x, sel, weights[:, ::-1])
        self.assertGreater(np.abs(ref - bad_ref).max(), 1e-3)

    def test_forward_batch1_integrates_gate_route_and_shared(self):
        """End-to-end forward (batch=1, single rank): gate selection drives the
        routed sum, the shared expert output is added, shape is preserved."""
        hidden, inter, n, k = 8, 16, 4, 2
        layer = _make_layer(
            hidden_size=hidden,
            moe_intermediate_size=inter,
            num_experts=n,
            num_experts_per_tok=k,
            num_shared_experts=1,
        )
        x = paddle.to_tensor(
            np.linspace(-0.5, 0.5, 1 * 4 * hidden, dtype=np.float32).reshape(
                [1, 4, hidden]
            )
        )
        # Gate is a collaborator here (its math is verified in TestGateRouting).
        (_, top_gate, top_idx, _, _, _, _, _) = layer.gate(x)
        sel = top_idx.numpy()
        weights = top_gate.numpy()

        reshaped = x.reshape([-1, hidden])
        routed_ref = self._routed_reference(layer, reshaped, sel, weights)
        shared_ref = layer.shared_experts(x).numpy()
        expected = routed_ref.reshape([1, 4, hidden]) + shared_ref

        out = layer(x).numpy()
        self.assertEqual(list(out.shape), [1, 4, hidden])
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)

        # Guard: dropping the shared contribution would change the output,
        # proving the shared expert branch is genuinely summed in.
        self.assertGreater(
            np.abs(out - routed_ref.reshape([1, 4, hidden])).max(), 1e-4
        )

    @unittest.expectedFailure
    def test_forward_shared_experts_batch_gt1_shape_bug(self):
        """BUG (reported): for batch_size > 1 the single-rank forward adds the
        2D routed output [B*S, D] to the 3D shared-expert output [B, S, D]
        *before* reshaping to the original shape, which broadcasts-errors
        (B*S vs S). This test asserts the CORRECT combined result; it is marked
        expectedFailure until the source reshapes the routed output before the
        shared-expert addition. (batch_size == 1 happens to broadcast cleanly,
        hiding the defect.)"""
        hidden, inter, n, k = 8, 16, 4, 2
        layer = _make_layer(
            hidden_size=hidden,
            moe_intermediate_size=inter,
            num_experts=n,
            num_experts_per_tok=k,
            num_shared_experts=1,
        )
        x = paddle.to_tensor(
            np.linspace(-0.5, 0.5, 2 * 4 * hidden, dtype=np.float32).reshape(
                [2, 4, hidden]
            )
        )
        (_, top_gate, top_idx, _, _, _, _, _) = layer.gate(x)
        reshaped = x.reshape([-1, hidden])
        routed_ref = self._routed_reference(
            layer, reshaped, top_idx.numpy(), top_gate.numpy()
        )
        shared_ref = layer.shared_experts(x).numpy()
        expected = routed_ref.reshape([2, 4, hidden]) + shared_ref

        out = layer(x).numpy()  # currently raises a broadcast ValueError
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


class TestUnsupportedEnvironments(unittest.TestCase):
    """Behaviours that require a real multi-rank process group (and GPU for
    DeepEP); recorded as skips rather than faked on a single CPU process."""

    @unittest.skip(
        "AllToAllMoECommunication dispatch/combine numerics require a real "
        "multi-rank expert-parallel process group (_AllToAll collective). "
        "Faking world_size + mocking the collective on one CPU process would "
        "only assert orchestration, not the cross-rank expert routing/return; "
        "belongs in tests/multi_card_tests. Single-rank construction and the "
        "traditional (non-EP) forward are covered above."
    )
    def test_alltoall_ep_dispatch_numeric(self):
        pass

    @unittest.skip(
        "DeepEPMoECommunication.token_permutation/token_unpermutation require "
        "the DeepEP kernels and a real multi-rank group on GPU; not "
        "CPU-verifiable here. Covered by dedicated multi-card DeepEP tests."
    )
    def test_deepep_ep_dispatch_numeric(self):
        pass


if __name__ == "__main__":
    unittest.main()
