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
"""Behavior tests for the base MoE gate (``paddlefleet.transformers.moe_gate``).

These run无卡 (CPU only): the gate routing/scoring math is device independent,
so we force the CPU place and drive the real gate APIs. Every expected value is
derived independently (by hand / NumPy), never by calling the routine under
test to build its own reference.
"""

from __future__ import annotations

import unittest

import numpy as np
import paddle

from paddlefleet.transformers.moe_gate import MoEGateMixin, PretrainedMoEGate

paddle.set_device("cpu")


class _GateConfig:
    """Minimal config object consumed by ``PretrainedMoEGate.__init__``."""

    def __init__(self, scoring_func=None, seq_aux=False):
        self.scoring_func = scoring_func
        self.seq_aux = seq_aux
        # Only referenced by the seq-aux path, which these tests do not enter.
        self.seq_length = 128
        self.moe_subbatch_token_num_before_dispatch = 0
        self.tensor_model_parallel_size = 1
        self.sequence_parallel = False


def _make_gate(num_experts=4, scoring_func=None, seq_aux=False, **kwargs):
    """Construct a real gate through its real __init__ (no mocking)."""
    config = _GateConfig(scoring_func=scoring_func, seq_aux=seq_aux)
    return PretrainedMoEGate(
        config=config,
        num_experts=num_experts,
        expert_hidden_size=16,
        **kwargs,
    )


class TestGateScoreFunc(unittest.TestCase):
    """gate_score_func must apply the named activation, not merely stay in range."""

    def _score(self, scoring_func, logits):
        gate = _make_gate(num_experts=len(logits[0]))
        # gate_score_func reads getattr(self, "scoring_func"); set it explicitly.
        gate.scoring_func = scoring_func
        return gate.gate_score_func(paddle.to_tensor(logits, dtype="float32"))

    def test_softmax_exact(self):
        out = self._score("softmax", [[1.0, 2.0, 3.0]])
        # Independent softmax of [1,2,3].
        expected = [[0.09003057, 0.24472847, 0.66524096]]
        np.testing.assert_allclose(out.numpy(), expected, atol=1e-6)

    def test_sigmoid_exact(self):
        out = self._score("sigmoid", [[0.0, 2.0, -2.0]])
        expected = [[0.5, 0.88079708, 0.11920292]]
        np.testing.assert_allclose(out.numpy(), expected, atol=1e-6)

    def test_tanh_exact(self):
        out = self._score("tanh", [[0.0, 1.0, -1.0]])
        expected = [[0.0, 0.76159416, -0.76159416]]
        np.testing.assert_allclose(out.numpy(), expected, atol=1e-6)

    def test_relu_exact(self):
        out = self._score("relu", [[-1.0, 0.5, 2.0]])
        expected = [[0.0, 0.5, 2.0]]
        np.testing.assert_allclose(out.numpy(), expected, atol=1e-6)

    def test_unknown_falls_back_to_softmax(self):
        # An unsupported name must produce softmax, not e.g. identity/zeros.
        logits = [[1.0, 2.0, 3.0]]
        out = self._score("does_not_exist", logits)
        expected = [[0.09003057, 0.24472847, 0.66524096]]
        np.testing.assert_allclose(out.numpy(), expected, atol=1e-6)
        # Distinguish softmax from a passthrough of the raw logits.
        self.assertFalse(np.allclose(out.numpy(), logits))


class TestTopkGreedy(unittest.TestCase):
    """_topk_greedy must pick the k largest experts, sorted descending, with values."""

    def test_selection_values_and_order(self):
        gate = _make_gate(num_experts=5)
        scores = paddle.to_tensor(
            [[0.1, 0.5, 0.3, 0.9, 0.2], [0.4, 0.8, 0.2, 0.1, 0.7]],
            dtype="float32",
        )
        topk_weight, topk_idx = gate._topk_greedy(scores, k=2)
        # Row0: 0.9@3 then 0.5@1. Row1: 0.8@1 then 0.7@4.
        np.testing.assert_array_equal(topk_idx.numpy(), [[3, 1], [1, 4]])
        np.testing.assert_allclose(
            topk_weight.numpy(), [[0.9, 0.5], [0.8, 0.7]], atol=1e-6
        )


class TestGroupLimitedGreedy(unittest.TestCase):
    """_topk_group_limited_greedy restricts routing to the selected expert groups."""

    def test_excludes_experts_outside_selected_group(self):
        # 6 experts, 3 groups of 2, keep only the single best group.
        gate = _make_gate(num_experts=6, n_group=3, topk_group=1)
        # Groups: g0=[0.9,0.3] max0.9, g1=[0.85,0.8] max0.85, g2=[0.2,0.1] max0.2.
        scores = paddle.to_tensor(
            [[0.9, 0.3, 0.85, 0.8, 0.2, 0.1]], dtype="float32"
        )
        topk_weight, topk_idx = gate._topk_group_limited_greedy(
            scores, k=2, n_group=3, topk_group=1
        )
        # Only g0 is kept, so expert2 (0.85, globally 2nd) is excluded and
        # expert1 (0.3) is chosen instead -> differs from a plain global top-2.
        np.testing.assert_array_equal(topk_idx.numpy(), [[0, 1]])
        np.testing.assert_allclose(topk_weight.numpy(), [[0.9, 0.3]], atol=1e-6)


class TestNoauxTcCorrectionBias(unittest.TestCase):
    """noaux_tc: the correction bias steers selection but is NOT the final weight."""

    def test_bias_changes_selection_not_returned_weight(self):
        gate = _make_gate(num_experts=4, n_group=2, topk_group=1)
        gate.eval()  # eval path returns original affinity scores as the weight
        # Raw affinity scores; without bias the top-2 would be experts 3 and 2.
        scores = paddle.to_tensor([[0.1, 0.2, 0.5, 0.6]], dtype="float32")
        # A large bias on expert 0 forces group g0=[0,1] to win selection.
        gate.e_score_correction_bias = paddle.to_tensor(
            [10.0, 0.0, 0.0, 0.0], dtype="float32"
        )
        topk_weight, topk_idx = gate._topk_noaux_tc(
            scores, k=2, n_group=2, topk_group=1
        )
        # Selection moved to experts 0 and 1 because of the bias.
        np.testing.assert_array_equal(topk_idx.numpy(), [[0, 1]])
        # The returned weights are the ORIGINAL scores (0.1, 0.2), NOT the
        # biased values (10.1, 0.2): correction bias is not the final weight.
        np.testing.assert_allclose(topk_weight.numpy(), [[0.1, 0.2]], atol=1e-6)


class TestTopkGatingWeights(unittest.TestCase):
    """topkgating (greedy, no-drop) returns raw / normalized / scaled routing weights."""

    # Fixed, tie-free gates so selection and weights are deterministic.
    GATES = [
        [0.1, 0.6, 0.2, 0.1],
        [0.5, 0.1, 0.3, 0.1],
        [0.4, 0.3, 0.2, 0.1],
    ]
    # Hand-derived top-2 (descending) per token.
    EXP_IDX = [[1, 2], [0, 2], [0, 1]]
    EXP_RAW = [[0.6, 0.2], [0.5, 0.3], [0.4, 0.3]]

    def _run(self, **kwargs):
        gate = _make_gate(
            num_experts=4, top_k=2, topk_method="greedy", **kwargs
        )
        gates = paddle.to_tensor(self.GATES, dtype="float32")
        return gate.topkgating(gates)

    def test_raw_weights_not_normalized(self):
        # norm off, scaling 1.0 -> returned weights are the raw gate values.
        capacity, combine, dispatch, priority, l_aux, l_zloss = self._run(
            norm_topk_prob=False, routed_scaling_factor=1.0
        )
        np.testing.assert_array_equal(dispatch.numpy(), self.EXP_IDX)
        np.testing.assert_allclose(combine.numpy(), self.EXP_RAW, atol=1e-6)
        # Raw top-2 weights need NOT sum to 1.
        sums = combine.numpy().sum(axis=-1)
        np.testing.assert_allclose(sums, [0.8, 0.8, 0.7], atol=1e-6)
        # No token dropped (capacity == per-expert load == 2): all kept.
        self.assertEqual(int(capacity), 2)
        np.testing.assert_array_equal(priority.numpy(), np.ones((3, 2)))

    def test_norm_topk_prob_sums_to_one(self):
        _, combine, _, _, _, _ = self._run(
            norm_topk_prob=True, routed_scaling_factor=1.0
        )
        expected = [
            [0.6 / 0.8, 0.2 / 0.8],
            [0.5 / 0.8, 0.3 / 0.8],
            [0.4 / 0.7, 0.3 / 0.7],
        ]
        np.testing.assert_allclose(combine.numpy(), expected, atol=1e-6)
        np.testing.assert_allclose(
            combine.numpy().sum(axis=-1), [1.0, 1.0, 1.0], atol=1e-6
        )

    def test_routed_scaling_breaks_sum_to_one(self):
        # With routed scaling ON (and norm off) weights are raw * factor and
        # deliberately do NOT sum to 1.
        _, combine, _, _, _, _ = self._run(
            norm_topk_prob=False, routed_scaling_factor=2.0
        )
        expected = 2.0 * np.array(self.EXP_RAW)
        np.testing.assert_allclose(combine.numpy(), expected, atol=1e-6)
        np.testing.assert_allclose(
            combine.numpy().sum(axis=-1), [1.6, 1.6, 1.4], atol=1e-6
        )

    def test_scaling_applies_after_normalization(self):
        # norm on + scaling 2.0 -> normalized weights (sum 1) then scaled (sum 2).
        _, combine, _, _, _, _ = self._run(
            norm_topk_prob=True, routed_scaling_factor=2.0
        )
        np.testing.assert_allclose(
            combine.numpy().sum(axis=-1), [2.0, 2.0, 2.0], atol=1e-6
        )


class TestPriorityCapacity(unittest.TestCase):
    """_priority drops tokens beyond capacity in assignment order (token identity)."""

    def test_overflow_token_is_dropped(self):
        gate = _make_gate(num_experts=4)
        # Tokens 0,1,2 -> expert 0; token 3 -> expert 1. capacity = 2.
        topk_idx = paddle.to_tensor([[0], [0], [0], [1]], dtype="int64")
        priority = gate._priority(topk_idx, capacity=2)
        # Expert 0 keeps its first two tokens (0,1); token 2 overflows -> dropped.
        expected = [
            [1.0, 0.0, 0.0, 0.0],  # token0 kept on expert0
            [1.0, 0.0, 0.0, 0.0],  # token1 kept on expert0
            [0.0, 0.0, 0.0, 0.0],  # token2 dropped (capacity exceeded)
            [0.0, 1.0, 0.0, 0.0],  # token3 kept on expert1
        ]
        np.testing.assert_array_equal(priority.numpy(), expected)


class TestCapacity(unittest.TestCase):
    """_capacity = (num_tokens // num_experts) * factor, and must stay positive."""

    def test_capacity_value(self):
        gate = _make_gate(num_experts=8)
        gates = paddle.zeros([32, 8], dtype="float32")
        self.assertEqual(gate._capacity(gates, capacity_factor=1.0), 4)

    def test_capacity_zero_raises(self):
        gate = _make_gate(num_experts=8)
        gates = paddle.zeros([4, 8], dtype="float32")  # 4 // 8 == 0
        with self.assertRaises(AssertionError):
            gate._capacity(gates, capacity_factor=1.0)


class TestZLoss(unittest.TestCase):
    """_cal_z_loss = mean over rows of logsumexp(row) ** 2."""

    def test_z_loss_exact(self):
        gate = _make_gate(num_experts=4)
        logits = paddle.to_tensor(
            [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype="float32"
        )
        # row0 logsumexp = ln(4); row1 = 1 + ln(4).
        lse0 = np.log(4.0)
        lse1 = 1.0 + np.log(4.0)
        expected = (lse0**2 + lse1**2) / 2.0
        out = gate._cal_z_loss(logits)
        self.assertEqual(list(out.shape), [])
        np.testing.assert_allclose(float(out), expected, atol=1e-6)


class TestTop1GatingNoDrop(unittest.TestCase):
    """top1gating's no-drop branch derives capacity from the per-expert load.

    With ``drop_tokens=False`` (the default, since ``moe_expert_capacity_factor``
    defaults to 0.0) ``top1gating`` sets

        new_capacity = max(exp_counts)
        capacity     = int(min(new_capacity, <#tokens>))

    Independent oracle for the fixed logits below: argmax routes token0 to
    expert0 and token1 to expert1, so the one-hot assignment gives
    ``exp_counts == [1, 1, 0, 0]``, ``max(exp_counts) == 1`` and, since that is
    the smaller operand against the 2-token count, ``capacity == 1``. This is
    hand-derived, not read back from the routine under test.
    """

    def test_top1gating_no_drop_capacity(self):
        gate = _make_gate(num_experts=4, use_rts=False)
        logits = paddle.to_tensor(
            [[3.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]], dtype="float32"
        )
        capacity, combine, dispatch, exp_counts, l_aux, l_zloss = (
            gate.top1gating(logits)
        )
        # capacity is a plain python int equal to the max per-expert load (1).
        self.assertIsInstance(capacity, int)
        self.assertEqual(capacity, 1)
        # Each of the two tokens lands on its own expert; the rest are empty.
        np.testing.assert_array_equal(exp_counts.numpy(), [1.0, 1.0, 0.0, 0.0])


class TestMixinIsBase(unittest.TestCase):
    """Sanity: PretrainedMoEGate really inherits the mixin routing helpers."""

    def test_inheritance(self):
        gate = _make_gate()
        self.assertIsInstance(gate, MoEGateMixin)


if __name__ == "__main__":
    unittest.main()
