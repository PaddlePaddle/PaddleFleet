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

"""Behaviour tests for paddlefleet.nn.moe_deepep.moe_gate.StandardMoEGate.

Scope: model layer / MoE gating, no-card (CPU). The real gate math stays in the
verification chain; every expected value is derived independently with NumPy
from tiny, sign-varied logits so the test can reject swapped experts, missing
normalization, missing scaling, or the correction-bias-as-weight bug.

Central contract checked here (see unit-test-rules "模型与训练目标 / MoE"):
  * top-k *selection* picks the right experts and returns the ORIGINAL gate
    values, in descending order;
  * norm_topk_prob divides the selected gate values by their sum;
  * routed_scaling_factor multiplies afterwards -- with a factor != 1 the row
    sum is NOT 1, so we assert against the scaled reference, not against 1;
  * for noaux_tc the e_score_correction_bias steers *which* experts are chosen
    but is NOT the returned weight (the weight is the bias-free score).

Cross-rank paths (global_aux_loss all_gather, sequence_parallel AllGatherOp in
_cal_seq_aux_loss with tensor_model_parallel_size > 1) are intentionally NOT
exercised here: proving them needs a real process group, and faking world_size
plus mocking the collective would only assert local orchestration, not the
cross-rank numerics. That is recorded as an explicit skip below. The tp==1
local branch of _cal_seq_aux_loss IS a genuine CPU computation and is verified.
"""

import unittest

import numpy as np
import paddle
import paddle.nn as nn

from paddlefleet.nn.moe_deepep.moe_gate import StandardMoEGate


def np_softmax(row):
    row = np.asarray(row, dtype=np.float64)
    e = np.exp(row - row.max())
    return e / e.sum()


def np_sigmoid(row):
    return 1.0 / (1.0 + np.exp(-np.asarray(row, dtype=np.float64)))


def make_gate(
    num_experts=8,
    expert_hidden_size=16,
    topk_method="greedy",
    num_experts_per_tok=2,
    norm_topk_prob=True,
    drop_tokens=False,
    moe_expert_capacity_factor=0.0,
    moe_token_drop_policy="probs",
    transpose_gate_weight=False,
    scoring_func="softmax",
    seq_length=32,
    n_group=1,
    topk_group=1,
    routed_scaling_factor=1.0,
    seq_aux=True,
):
    """Construct a real StandardMoEGate (no mocked __init__)."""
    moe_config = {
        "gate_activation": scoring_func,
        "eval_capacity_factor": 1.0,
        "group": None,
        "global_aux_loss": False,
        "use_rts": True,
        "top2_2nd_expert_sampling": True,
        "seq_aux": seq_aux,
    }
    return StandardMoEGate(
        num_experts=num_experts,
        expert_hidden_size=expert_hidden_size,
        drop_tokens=drop_tokens,
        topk_method=topk_method,
        num_experts_per_tok=num_experts_per_tok,
        norm_topk_prob=norm_topk_prob,
        moe_config=moe_config,
        seq_length=seq_length,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=routed_scaling_factor,
        moe_subbatch_token_num_before_dispatch=-1,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        moe_expert_capacity_factor=moe_expert_capacity_factor,
        moe_token_drop_policy=moe_token_drop_policy,
        transpose_gate_weight=transpose_gate_weight,
    )


class TestGateScoring(unittest.TestCase):
    """gate_score_func and z-loss: real activation math vs NumPy reference."""

    def test_softmax_scoring_matches_numpy(self):
        gate = make_gate(scoring_func="softmax")
        logits_np = np.array(
            [[2.0, 1.0, 0.0, -1.0], [-3.0, 0.5, 4.0, 1.0]], dtype=np.float32
        )
        scores = gate.gate_score_func(paddle.to_tensor(logits_np))
        ref = np.stack([np_softmax(logits_np[0]), np_softmax(logits_np[1])])
        np.testing.assert_allclose(scores.numpy(), ref, rtol=1e-6, atol=1e-6)

    def test_sigmoid_scoring_matches_numpy(self):
        gate = make_gate(scoring_func="sigmoid")
        logits_np = np.array([[0.0, 2.0, -2.0, 1.0]], dtype=np.float32)
        scores = gate.gate_score_func(paddle.to_tensor(logits_np))
        ref = np_sigmoid(logits_np)
        np.testing.assert_allclose(scores.numpy(), ref, rtol=1e-6, atol=1e-6)
        # sigmoid is element-wise: rows must NOT be renormalized to sum 1 here.
        self.assertFalse(np.allclose(scores.numpy().sum(axis=-1), 1.0))

    def test_unknown_scoring_falls_back_to_softmax(self):
        gate = make_gate(scoring_func="not_a_real_func")
        logits_np = np.array([[1.0, 3.0, 0.0, -2.0]], dtype=np.float32)
        scores = gate.gate_score_func(paddle.to_tensor(logits_np))
        # Fallback must reproduce softmax exactly, not merely "sum to 1".
        np.testing.assert_allclose(
            scores.numpy(), np_softmax(logits_np[0])[None], rtol=1e-6, atol=1e-6
        )

    def test_scoring_output_is_float32_for_low_precision_input(self):
        gate = make_gate(scoring_func="softmax")
        logits = paddle.to_tensor([[1.0, 2.0, 0.0, -1.0]], dtype="float16")
        scores = gate.gate_score_func(logits)
        self.assertEqual(scores.dtype, paddle.float32)
        ref = np_softmax(np.array([1.0, 2.0, 0.0, -1.0]))
        # float16 logits -> float32 softmax; compare with a fp16-rounding tol.
        np.testing.assert_allclose(scores.numpy()[0], ref, rtol=2e-3, atol=2e-3)

    def test_z_loss_matches_logsumexp_square_mean(self):
        gate = make_gate()
        logits_np = np.array(
            [[2.0, 1.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32
        )
        z = gate._cal_z_loss(paddle.to_tensor(logits_np))
        lse = np.log(np.exp(logits_np).sum(axis=1))
        ref = np.mean(lse**2)
        self.assertEqual(list(z.shape), [])
        np.testing.assert_allclose(z.numpy(), ref, rtol=1e-5, atol=1e-5)


class TestTopkSelection(unittest.TestCase):
    """Top-k selection: which experts, which values, ordering."""

    def test_greedy_selects_top_experts_with_values(self):
        gate = make_gate(num_experts_per_tok=2)
        scores = paddle.to_tensor(
            [[0.1, 0.4, 0.2, 0.3], [0.5, 0.05, 0.25, 0.2]], dtype="float32"
        )
        w, idx = gate._topk_greedy(scores, k=2)
        # Descending selection: row0 -> experts 1(0.4),3(0.3); row1 -> 0,2.
        self.assertEqual(idx.numpy().tolist(), [[1, 3], [0, 2]])
        np.testing.assert_allclose(
            w.numpy(), [[0.4, 0.3], [0.5, 0.25]], rtol=1e-6, atol=1e-6
        )
        # Values are returned in descending order (col0 >= col1).
        self.assertTrue(bool((w[:, 0] >= w[:, 1]).all()))

    def test_group_limited_greedy_excludes_unselected_group(self):
        # 2 groups of 2 experts; only 1 group kept. A high-scoring expert in
        # the dropped group must be excluded even though it beats kept ones.
        gate = make_gate(
            num_experts=4,
            topk_method="group_limited_greedy",
            num_experts_per_tok=2,
            n_group=2,
            topk_group=1,
        )
        scores = paddle.to_tensor([[0.1, 0.6, 0.2, 0.3]], dtype="float32")
        w, idx = gate._topk_group_limited_greedy(
            scores, k=2, n_group=2, topk_group=1
        )
        # group0=[0.1,0.6] max 0.6, group1=[0.2,0.3] max 0.3 -> keep group0.
        # Within group0, top2 = expert1(0.6), expert0(0.1). Expert3(0.3) is
        # excluded despite outscoring expert0, proving group masking works.
        self.assertEqual(idx.numpy().tolist(), [[1, 0]])
        np.testing.assert_allclose(
            w.numpy(), [[0.6, 0.1]], rtol=1e-6, atol=1e-6
        )

    def test_group_limited_greedy_asserts_divisible(self):
        gate = make_gate(num_experts=8, n_group=3)
        scores = paddle.randn([4, 8])
        with self.assertRaises(AssertionError):
            gate._topk_group_limited_greedy(
                scores, k=2, n_group=3, topk_group=1
            )

    def test_noaux_tc_bias_steers_selection_but_weight_is_bias_free(self):
        # THE key MoE-gating contract: correction bias adjusts affinity for
        # *routing only*; the returned gate value is the ORIGINAL score.
        gate = make_gate(
            num_experts=4,
            topk_method="noaux_tc",
            num_experts_per_tok=2,
            n_group=1,
            topk_group=1,
        )
        scores_np = np.array([[0.10, 0.20, 0.70, 0.50]], dtype=np.float32)
        bias_np = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        gate.e_score_correction_bias.set_value(paddle.to_tensor(bias_np))
        scores = paddle.to_tensor(scores_np)

        w, idx = gate._topk_noaux_tc(scores, k=2, n_group=1, topk_group=1)

        # scores+bias = [1.10, 0.20, 0.70, 0.50] -> top2 = experts 0, 2.
        self.assertEqual(idx.numpy().tolist(), [[0, 2]])
        # Without the bias the top2 would be experts 2 and 3: the bias truly
        # changed routing.
        plain = np.argsort(-scores_np[0])[:2].tolist()
        self.assertEqual(sorted(plain), [2, 3])
        # Returned weights are the ORIGINAL (bias-free) scores at [0, 2],
        # i.e. [0.10, 0.70] -- NOT the bias-adjusted [1.10, 0.70].
        np.testing.assert_allclose(
            w.numpy(), [[0.10, 0.70]], rtol=1e-6, atol=1e-6
        )
        self.assertFalse(np.allclose(w.numpy(), [[1.10, 0.70]]))


class TestCapacityPriorityOneHot(unittest.TestCase):
    """Capacity math, cumulative-priority dropping, and one-hot encoding."""

    def test_capacity_value_and_factor(self):
        gate = make_gate()
        gates = paddle.zeros([16, 4])
        # (num_tokens // num_experts) * factor
        self.assertEqual(gate._capacity(gates, capacity_factor=1.0), 4)
        self.assertEqual(gate._capacity(gates, capacity_factor=2.0), 8)

    def test_capacity_asserts_positive(self):
        gate = make_gate()
        gates = paddle.zeros([3, 16])  # 3 // 16 == 0 -> capacity 0
        with self.assertRaises(AssertionError):
            gate._capacity(gates, capacity_factor=1.0)

    def test_priority_drops_by_cumulative_capacity(self):
        gate = make_gate(num_experts=4)
        topk_idx = paddle.to_tensor([[0, 1], [0, 2], [1, 3]])
        # capacity=1: each expert accepts only its first (row-major) claimant.
        # token1's second claim on expert0 is dropped (expert0 full);
        # token2's claim on expert1 is dropped (expert1 full).
        prio = gate._priority(topk_idx, capacity=1)
        expected = np.array(
            [[1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        np.testing.assert_array_equal(prio.numpy(), expected)

    def test_priority_high_capacity_keeps_all(self):
        gate = make_gate(num_experts=4)
        topk_idx = paddle.to_tensor([[0, 1], [0, 2], [1, 3]])
        prio = gate._priority(topk_idx, capacity=8)
        expected = np.array(
            [[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.float32
        )
        np.testing.assert_array_equal(prio.numpy(), expected)

    def test_one_hot_to_float_content_and_dtype(self):
        gate = make_gate()
        out = gate._one_hot_to_float(paddle.to_tensor([0, 2, 1]), num_classes=4)
        np.testing.assert_array_equal(
            out.numpy(),
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]],
        )
        self.assertEqual(out.dtype, paddle.float32)

    def test_one_hot_to_int64_content_and_dtype(self):
        gate = make_gate()
        out = gate._one_hot_to_int64(paddle.to_tensor([3, 0]), num_classes=4)
        np.testing.assert_array_equal(out.numpy(), [[0, 0, 0, 1], [1, 0, 0, 0]])
        self.assertEqual(out.dtype, paddle.int64)

    def test_one_hot_casts_float_input(self):
        gate = make_gate()
        out = gate._one_hot_to_float(
            paddle.to_tensor([0.0, 1.0, 2.0]), num_classes=4
        )
        np.testing.assert_array_equal(
            out.numpy(),
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        )


class TestAuxAndOrthogonalLoss(unittest.TestCase):
    """Local (single-rank) auxiliary-loss and orthogonal-loss numerics."""

    def test_cal_aux_loss_local(self):
        gate = make_gate(num_experts=4)
        gates = paddle.to_tensor(
            [[0.6, 0.2, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1]], dtype="float32"
        )
        mask = paddle.to_tensor(
            [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype="float32"
        )
        aux = gate._cal_aux_loss(gates, mask)
        me = gates.numpy().mean(axis=0)
        ce = mask.numpy().mean(axis=0)
        ref = float((me * ce).sum() * 4)
        self.assertEqual(list(aux.shape), [])
        np.testing.assert_allclose(aux.numpy(), ref, rtol=1e-6, atol=1e-6)

    def test_cal_seq_aux_loss_tp1_local(self):
        # tensor_model_parallel_size == 1 path is a genuine local computation.
        gate = make_gate(num_experts=4)
        probs = paddle.to_tensor(
            [[0.6, 0.2, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1]], dtype="float32"
        )
        routing_map = paddle.to_tensor(
            [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype="float32"
        )
        loss = gate._cal_seq_aux_loss(
            probs, top_k=2, routing_map=routing_map, max_seq_len=2
        )
        # denom = max_seq_len*top_k/E = 2*2/4 = 1
        cost = routing_map.numpy().sum(axis=0) / 1.0  # [1,1,1,0]
        contrib = cost * (probs.numpy().sum(axis=0) / 2.0)
        ref = float(contrib.sum())  # single batch -> mean == the row
        np.testing.assert_allclose(loss.numpy(), ref, rtol=1e-6, atol=1e-6)

    def test_orthogonal_loss_numeric(self):
        gate = make_gate(num_experts=2, expert_hidden_size=4)
        # Columns: e0=[1,0,0,0], e1=[1,1,0,0]. After F.normalize(axis=0):
        # e0 stays, e1 -> [1,1,0,0]/sqrt(2). Gram off-diagonal = 1/sqrt(2).
        gate.weight.set_value(
            paddle.to_tensor(
                [[1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
                dtype="float32",
            )
        )
        loss = gate._cal_orthogonal_loss()
        # (Gram - I) has two off-diagonal entries of 1/sqrt(2); squared 0.5
        # each; mean over the 2x2 matrix = (0+0.5+0.5+0)/4 = 0.25.
        self.assertEqual(list(loss.shape), [])
        np.testing.assert_allclose(loss.numpy(), 0.25, rtol=1e-5, atol=1e-5)


def build_identity_gate(**kwargs):
    """Gate whose linear projection is the identity, so ``logits == input``.

    This lets us drive the *real* forward/topkgating entry with hand-chosen
    logits and hand-derive every routing output end to end.
    """
    kwargs.setdefault("num_experts", 4)
    kwargs.setdefault("expert_hidden_size", 4)
    gate = make_gate(**kwargs)
    gate.weight.set_value(paddle.eye(4, dtype="float32"))
    return gate


class TestForwardRouting(unittest.TestCase):
    """End-to-end forward: selection, normalization, scaling, masking."""

    LOGITS = np.array([[2.0, 1.0, 0.0, -1.0]], dtype=np.float32)

    def _ref(self, norm, scale):
        probs = np_softmax(self.LOGITS[0])
        idx = np.argsort(-probs)[:2]  # [0, 1]
        vals = probs[idx]
        denom = vals.sum() if norm else 1.0
        top_gate = (vals / denom) * scale
        masked = np.zeros(4)
        masked[idx] = probs[idx]
        masked = (masked / (probs[idx].sum() if norm else 1.0)) * scale
        return idx, top_gate, masked

    def test_forward_norm_true_scale_one(self):
        gate = build_identity_gate(
            norm_topk_prob=True, routed_scaling_factor=1.0
        )
        x = paddle.to_tensor(self.LOGITS)
        cap, top_gate, top_idx, gates_masked, mask, _, _, _ = gate(x)
        idx, ref_gate, ref_masked = self._ref(norm=True, scale=1.0)

        self.assertEqual(top_idx.numpy().tolist(), [idx.tolist()])
        np.testing.assert_allclose(
            top_gate.numpy(), [ref_gate], rtol=1e-5, atol=1e-6
        )
        # Selected gate values normalized to sum 1 (factor == 1).
        np.testing.assert_allclose(top_gate.numpy().sum(), 1.0, atol=1e-6)
        np.testing.assert_array_equal(mask.numpy(), [[1, 1, 0, 0]])
        np.testing.assert_allclose(
            gates_masked.numpy(), [ref_masked], rtol=1e-5, atol=1e-6
        )
        # drop_tokens=False -> capacity is the max per-expert assignment count.
        self.assertEqual(cap, 1)

    def test_forward_norm_false_keeps_raw_gate_values(self):
        gate = build_identity_gate(
            norm_topk_prob=False, routed_scaling_factor=1.0
        )
        x = paddle.to_tensor(self.LOGITS)
        _, top_gate, top_idx, gates_masked, _, _, _, _ = gate(x)
        idx, ref_gate, ref_masked = self._ref(norm=False, scale=1.0)

        self.assertEqual(top_idx.numpy().tolist(), [idx.tolist()])
        # Without normalization the raw softmax probabilities survive, so the
        # row does NOT sum to 1.
        np.testing.assert_allclose(
            top_gate.numpy(), [ref_gate], rtol=1e-5, atol=1e-6
        )
        self.assertFalse(np.allclose(top_gate.numpy().sum(), 1.0))
        np.testing.assert_allclose(
            gates_masked.numpy(), [ref_masked], rtol=1e-5, atol=1e-6
        )

    def test_forward_routed_scaling_factor_applied(self):
        gate = build_identity_gate(
            norm_topk_prob=True, routed_scaling_factor=2.0
        )
        x = paddle.to_tensor(self.LOGITS)
        _, top_gate, _, gates_masked, _, _, _, _ = gate(x)
        idx, ref_gate, ref_masked = self._ref(norm=True, scale=2.0)

        np.testing.assert_allclose(
            top_gate.numpy(), [ref_gate], rtol=1e-5, atol=1e-6
        )
        # With routed_scaling_factor=2 the normalized weights are scaled, so
        # the row sums to 2, not 1.
        np.testing.assert_allclose(top_gate.numpy().sum(), 2.0, atol=1e-5)
        np.testing.assert_allclose(
            gates_masked.numpy(), [ref_masked], rtol=1e-5, atol=1e-6
        )

    def test_forward_mask_marks_exactly_selected_experts(self):
        gate = build_identity_gate()
        x = paddle.to_tensor(
            [[2.0, 1.0, 0.0, -1.0], [-1.0, 0.0, 1.0, 2.0]], dtype="float32"
        )
        _, _, top_idx, gates_masked, mask, _, _, _ = gate(x)
        # token0 -> experts {0,1}; token1 -> experts {3,2}.
        self.assertEqual(top_idx.numpy().tolist(), [[0, 1], [3, 2]])
        np.testing.assert_array_equal(
            mask.numpy(), [[1, 1, 0, 0], [0, 0, 1, 1]]
        )
        # mask must be 1 exactly where gates_masked is non-zero.
        np.testing.assert_array_equal(
            (gates_masked.numpy() != 0).astype("float32"), mask.numpy()
        )

    def test_forward_invalid_topk_method_raises(self):
        gate = build_identity_gate(topk_method="does_not_exist")
        with self.assertRaises(NotImplementedError):
            gate(paddle.to_tensor(self.LOGITS))

    def test_forward_noaux_tc_accumulates_real_expert_usage(self):
        # noaux_tc updates expert_usage by the actual per-expert selection
        # counts; check the exact counts and that they accumulate.
        gate = build_identity_gate(
            topk_method="noaux_tc",
            num_experts_per_tok=2,
            n_group=2,
            topk_group=2,  # keep all groups -> plain top-2 (bias defaults to 0)
        )
        x = paddle.to_tensor(
            [[2.0, 1.0, 0.0, -1.0], [1.0, 2.0, 0.0, -1.0]], dtype="float32"
        )
        # token0 -> {0,1}, token1 -> {1,0}; counts per expert = [2,2,0,0].
        np.testing.assert_array_equal(gate.expert_usage.numpy(), [0, 0, 0, 0])
        gate(x)
        np.testing.assert_array_equal(gate.expert_usage.numpy(), [2, 2, 0, 0])
        gate(x)  # accumulates across calls
        np.testing.assert_array_equal(gate.expert_usage.numpy(), [4, 4, 0, 0])


class TestCrossRankSkipped(unittest.TestCase):
    def test_cross_rank_aux_paths_need_real_process_group(self):
        self.skipTest(
            "global_aux_loss (dist.all_gather over me/ce) and the "
            "sequence_parallel branch of _cal_seq_aux_loss "
            "(tensor_model_parallel_size>1 AllGatherOp) require a real "
            "multi-rank process group. Faking world_size and mocking the "
            "collective would only assert local orchestration, not the "
            "cross-rank reduced numerics (see unit-test-antipatterns #13). "
            "The tp==1 local branch is verified in "
            "TestAuxAndOrthogonalLoss.test_cal_seq_aux_loss_tp1_local."
        )


if __name__ == "__main__":
    unittest.main()
