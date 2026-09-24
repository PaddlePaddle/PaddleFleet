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

"""Behavior tests for paddlefleet.transformers.moe_gate_auto.

These are CPU / 无卡 tests: every input is small, fixed, and all-distinct so a
routing swap, wrong normalization, dropped scaling factor or transposed layout
changes the NUMBERS (not just the shape). Expected values are derived by hand or
from an INDEPENDENT numpy reference (numpy softmax / logsumexp / erf); we never
call the function under test to build its own expectation.

Contract points exercised (model-layer MoE routing):
  * which experts are selected (top-k / group-limited);
  * the raw gating values that survive at the selected positions;
  * norm_topk_prob normalization (selected weights renormalized to sum 1);
  * routed_scaling_factor scaling (weights need NOT sum to 1 when it is on);
  * expert-count / capacity bookkeeping and the auxiliary / z losses.

Where the production code is genuinely broken, the test asserts the CORRECT
behavior and is marked @unittest.expectedFailure with the reason inline; the
production source is NOT modified.
"""

import math
import unittest
from types import SimpleNamespace

import numpy as np
import paddle

from paddlefleet.transformers.moe_gate_auto import (
    MoEGateMixin,
    PretrainedMoEGate,
)

# 无卡: force CPU so nothing tries to touch an accelerator.
paddle.set_device("cpu")


def _np_softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _make_gate(num_experts=4, expert_hidden_size=8, **kwargs):
    """Construct a REAL PretrainedMoEGate (runs the real __init__)."""
    cfg = SimpleNamespace(seq_aux=kwargs.pop("seq_aux", False))
    return PretrainedMoEGate(
        config=cfg,
        num_experts=num_experts,
        expert_hidden_size=expert_hidden_size,
        **kwargs,
    )


def _make_mixin(scoring_func):
    gate = type("_Gate", (MoEGateMixin,), {})()
    gate.scoring_func = scoring_func
    return gate


class TestGateScoreFunc(unittest.TestCase):
    """gate_score_func must apply the named activation; unknown -> softmax."""

    LOGITS = [[-1.0, 0.0, 1.0, 2.0]]

    def _run(self, scoring_func):
        gate = _make_mixin(scoring_func)
        out = gate.gate_score_func(
            paddle.to_tensor(self.LOGITS, dtype="float32")
        )
        return out.numpy()

    def test_softmax(self):
        x = np.array(self.LOGITS, dtype=np.float64)
        np.testing.assert_allclose(
            self._run("softmax"), _np_softmax(x), atol=1e-6
        )

    def test_sigmoid(self):
        x = np.array(self.LOGITS, dtype=np.float64)
        np.testing.assert_allclose(
            self._run("sigmoid"), 1.0 / (1.0 + np.exp(-x)), atol=1e-6
        )

    def test_tanh(self):
        x = np.array(self.LOGITS, dtype=np.float64)
        np.testing.assert_allclose(self._run("tanh"), np.tanh(x), atol=1e-6)

    def test_relu(self):
        x = np.array(self.LOGITS, dtype=np.float64)
        np.testing.assert_allclose(
            self._run("relu"), np.maximum(x, 0.0), atol=1e-6
        )

    def test_gelu(self):
        x = np.array(self.LOGITS, dtype=np.float64)
        erf = np.vectorize(math.erf)
        ref = 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))  # exact (erf) gelu
        np.testing.assert_allclose(self._run("gelu"), ref, atol=1e-6)

    def test_leaky_relu(self):
        x = np.array(self.LOGITS, dtype=np.float64)
        ref = np.where(x > 0, x, 0.01 * x)  # paddle default negative_slope=0.01
        np.testing.assert_allclose(self._run("leaky_relu"), ref, atol=1e-6)

    def test_unknown_defaults_to_softmax(self):
        x = np.array(self.LOGITS, dtype=np.float64)
        np.testing.assert_allclose(
            self._run("banana"), _np_softmax(x), atol=1e-6
        )


class TestOneHotHelpers(unittest.TestCase):
    """Full one-hot content and dtype, not just shape."""

    def test_one_hot_to_float(self):
        gate = _make_gate()
        out = gate._one_hot_to_float(
            paddle.to_tensor([0, 1, 2], dtype="int64"), 4
        )
        expected = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32
        )
        np.testing.assert_array_equal(out.numpy(), expected)
        self.assertEqual(out.dtype, paddle.float32)

    def test_one_hot_to_int64(self):
        gate = _make_gate()
        out = gate._one_hot_to_int64(paddle.to_tensor([2, 0], dtype="int64"), 3)
        expected = np.array([[0, 0, 1], [1, 0, 0]], dtype=np.int64)
        np.testing.assert_array_equal(out.numpy(), expected)
        self.assertEqual(out.dtype, paddle.int64)


class TestCapacity(unittest.TestCase):
    """_capacity == (num_tokens // num_experts) * factor, with guards."""

    def test_capacity_values(self):
        gate = _make_gate(num_experts=8)
        gates = paddle.ones([32, 8], dtype="float32")
        self.assertEqual(gate._capacity(gates, 1.0), 4)  # 32//8 * 1.0
        self.assertEqual(gate._capacity(gates, 2.0), 8)  # 32//8 * 2.0

    def test_capacity_must_be_positive(self):
        gate = _make_gate(num_experts=8)
        gates = paddle.ones([32, 8], dtype="float32")
        with self.assertRaises(AssertionError):
            gate._capacity(gates, 0.1)  # int(4 * 0.1) == 0 -> guarded

    def test_capacity_requires_2d(self):
        gate = _make_gate(num_experts=8)
        with self.assertRaises(AssertionError):
            gate._capacity(paddle.ones([2, 16, 8], dtype="float32"), 1.0)


class TestZLoss(unittest.TestCase):
    """_cal_z_loss == mean over rows of logsumexp(row)**2."""

    def test_z_loss_matches_numpy(self):
        gate = _make_gate()
        logits = np.array(
            [[0.1, 0.4, 0.3, 0.2], [0.5, 0.1, 0.1, 0.3]], dtype=np.float64
        )
        out = gate._cal_z_loss(paddle.to_tensor(logits, dtype="float32"))
        lse = np.log(np.exp(logits).sum(axis=1))  # independent reference
        ref = np.mean(lse**2)
        self.assertEqual(list(out.shape), [])
        np.testing.assert_allclose(float(out), ref, atol=1e-5)


class TestAuxLoss(unittest.TestCase):
    """_cal_aux_loss == sum(mean(gates)*mean(mask)) * num_experts."""

    def test_aux_loss_hand_value(self):
        gate = _make_gate(num_experts=4, global_aux_loss=False)
        gates = paddle.to_tensor(
            [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], dtype="float32"
        )
        # token0 -> expert0, token1 -> expert3
        mask = paddle.to_tensor([[1, 0, 0, 0], [0, 0, 0, 1]], dtype="float32")
        # me = [0.25,0.25,0.25,0.25]; ce = [0.5,0,0,0.5]
        # sum(me*ce) = 0.25; * num_experts(4) = 1.0
        out = gate._cal_aux_loss(gates, mask)
        self.assertEqual(list(out.shape), [])
        np.testing.assert_allclose(float(out), 1.0, atol=1e-6)


class TestOrthogonalLoss(unittest.TestCase):
    """_cal_orthogonal_loss == mean((normalize(W,axis=0)^T W - I)**2)."""

    def test_orthogonal_loss_hand_value(self):
        gate = _make_gate(num_experts=2)
        w = np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]], dtype="float32")
        gate.weight = paddle.create_parameter(
            shape=[3, 2],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Assign(w),
        )
        # normalize columns: c0=[1,0,0], c1=[1,1,0]/sqrt2
        # M = Wn^T Wn = [[1, 1/sqrt2],[1/sqrt2, 1]]; (M-I)^2 mean = 0.25
        out = gate._cal_orthogonal_loss()
        self.assertEqual(list(out.shape), [])
        np.testing.assert_allclose(float(out), 0.25, atol=1e-6)


class TestTopkGreedy(unittest.TestCase):
    """_topk_greedy returns the sorted top-k values and their indices."""

    def test_topk_greedy_hand_values(self):
        gate = _make_gate(num_experts=4)
        scores = paddle.to_tensor(
            [[0.1, 0.4, 0.3, 0.2], [0.5, 0.1, 0.1, 0.3]], dtype="float32"
        )
        weight, idx = gate._topk_greedy(scores, k=2)
        np.testing.assert_allclose(
            weight.numpy(), [[0.4, 0.3], [0.5, 0.3]], atol=1e-6
        )
        np.testing.assert_array_equal(idx.numpy(), [[1, 2], [0, 3]])


class TestTopkGroupLimitedGreedy(unittest.TestCase):
    """Group limiting must confine the top-k to the selected group(s)."""

    def test_single_group_single_expert(self):
        gate = _make_gate(num_experts=4, n_group=2, topk_group=1)
        # groups: g0=[0.1,0.9], g1=[0.5,0.2]; group scores [0.9,0.5] -> keep g0
        scores = paddle.to_tensor([[0.1, 0.9, 0.5, 0.2]], dtype="float32")
        weight, idx = gate._topk_group_limited_greedy(
            scores, k=1, n_group=2, topk_group=1
        )
        np.testing.assert_allclose(weight.numpy(), [[0.9]], atol=1e-6)
        np.testing.assert_array_equal(idx.numpy(), [[1]])

    def test_group_limit_excludes_high_expert_in_dropped_group(self):
        gate = _make_gate(num_experts=4, n_group=2, topk_group=1)
        # g0=[0.1,0.3] (max 0.3), g1=[0.9,0.8] (max 0.9) -> keep g1 only.
        # Even though we take top-2 experts, BOTH must come from g1 (idx 2,3);
        # g0's experts are masked to 0 and excluded.
        scores = paddle.to_tensor([[0.1, 0.3, 0.9, 0.8]], dtype="float32")
        weight, idx = gate._topk_group_limited_greedy(
            scores, k=2, n_group=2, topk_group=1
        )
        # topk with sorted=False: compare as sets to avoid ordering assumptions.
        self.assertEqual(sorted(idx.numpy().reshape(-1).tolist()), [2, 3])
        np.testing.assert_allclose(
            np.sort(weight.numpy().reshape(-1)), [0.8, 0.9], atol=1e-6
        )


class TestTopkGating(unittest.TestCase):
    """Core top-k routing: selection, surviving raw gate values, normalization.

    Fixed gates (already a probability distribution per token, as the caller
    supplies), B=1, S=2, E=4, top_k=2, greedy, no token dropping. Everything is
    hand-derived below.

        token0 gates = [0.1, 0.4, 0.3, 0.2] -> top2 experts {1,2}
        token1 gates = [0.5, 0.1, 0.1, 0.3] -> top2 experts {0,3}
        exp_counts = [1,1,1,1]; capacity = max(exp_counts) = 1.
    """

    GATES = [[[0.1, 0.4, 0.3, 0.2], [0.5, 0.1, 0.1, 0.3]]]

    def _gate(self, **kw):
        return _make_gate(
            num_experts=4, top_k=2, topk_method="greedy", seq_aux=False, **kw
        )

    def test_no_norm_keeps_raw_selected_gates(self):
        gate = self._gate(norm_topk_prob=False)
        cap, cw, dm, exp_counts, l_aux, l_zloss = gate.topkgating(
            paddle.to_tensor(self.GATES, dtype="float32")
        )
        self.assertEqual(cap, 1)
        self.assertEqual(list(cw.shape), [2, 4, 1])
        expected = np.array(
            [[[0.0], [0.4], [0.3], [0.0]], [[0.5], [0.0], [0.0], [0.3]]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(cw.numpy(), expected, atol=1e-6)
        # dispatch mask picks exactly the selected experts.
        np.testing.assert_array_equal(
            dm.numpy().astype(bool),
            np.array([[[0], [1], [1], [0]], [[1], [0], [0], [1]]], dtype=bool),
        )
        np.testing.assert_array_equal(exp_counts.numpy(), [1, 1, 1, 1])
        np.testing.assert_allclose(float(l_aux), 2.0, atol=1e-6)
        # z-loss on the gates, independent numpy reference.
        g = np.array(self.GATES, dtype=np.float64).reshape(2, 4)
        ref_z = np.mean(np.log(np.exp(g).sum(axis=1)) ** 2)
        np.testing.assert_allclose(float(l_zloss), ref_z, atol=1e-5)

    def test_norm_topk_prob_renormalizes_to_sum_one(self):
        gate = self._gate(norm_topk_prob=True)
        cap, cw, dm, exp_counts, l_aux, l_zloss = gate.topkgating(
            paddle.to_tensor(self.GATES, dtype="float32")
        )
        # selected gates renormalized: row0 /0.7, row1 /0.8
        expected = np.array(
            [
                [[0.0], [0.4 / 0.7], [0.3 / 0.7], [0.0]],
                [[0.5 / 0.8], [0.0], [0.0], [0.3 / 0.8]],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(cw.numpy(), expected, atol=1e-6)
        # With normalization ON the per-token weights sum to 1.
        np.testing.assert_allclose(
            cw.numpy().sum(axis=(1, 2)), [1.0, 1.0], atol=1e-6
        )


class TestTop2Gating(unittest.TestCase):
    """top2gating selects top-2 experts and renormalizes their softmax gates.

    Softmax is applied INSIDE top2gating (scoring_func defaults to softmax), so
    numpy softmax is an independent oracle for the selected experts and the
    normalized combine weights. top2_2nd_expert_sampling=False removes the
    Gumbel noise, making the result deterministic.
    """

    LOGITS = [[1.0, 2.0, 3.0, 0.0], [0.0, 1.0, 0.0, 2.0]]

    def test_selection_and_normalized_weights(self):
        gate = _make_gate(num_experts=4, top2_2nd_expert_sampling=False)
        cap, cw, dm, exp_counts, l_aux, l_zloss = gate.top2gating(
            paddle.to_tensor(self.LOGITS, dtype="float32")
        )
        probs = _np_softmax(self.LOGITS, axis=1)
        # independent per-expert combine weight [S,E]: only the 2 selected
        # experts non-zero, each = its softmax prob / (p_top1 + p_top2).
        expected = np.zeros((2, 4), dtype=np.float64)
        for s in range(2):
            order = np.argsort(-probs[s])
            e1, e2 = order[0], order[1]
            denom = probs[s, e1] + probs[s, e2]
            expected[s, e1] = probs[s, e1] / denom
            expected[s, e2] = probs[s, e2] / denom
        # sum over the capacity axis to recover per-expert weight.
        got = cw.numpy().sum(axis=2)
        np.testing.assert_allclose(got, expected, atol=1e-5)
        # each token's two weights renormalize to 1.
        np.testing.assert_allclose(got.sum(axis=1), [1.0, 1.0], atol=1e-6)
        # top-1 expert of each token carries strictly more weight than top-2.
        self.assertGreater(expected[0].max(), 0.5)
        self.assertGreater(expected[1].max(), 0.5)


class TestTopkGatingPart1(unittest.TestCase):
    """topkgating_part1 produces the same mask / counts / losses as the mono path."""

    GATES = TestTopkGating.GATES

    def test_part1_mask_and_losses(self):
        gate = _make_gate(
            num_experts=4, top_k=2, topk_method="greedy", seq_aux=False
        )
        exp_counts, l_aux, l_zloss = gate.topkgating_part1(
            paddle.to_tensor(self.GATES, dtype="float32"), None
        )
        np.testing.assert_array_equal(exp_counts.numpy(), [1, 1, 1, 1])
        np.testing.assert_allclose(float(l_aux), 2.0, atol=1e-6)
        # mask marks exactly the selected experts.
        np.testing.assert_array_equal(
            gate.mask.numpy().astype(bool),
            np.array([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=bool),
        )


class TestProductionBugs(unittest.TestCase):
    """Tests asserting the CORRECT behavior for code paths that are broken.

    Each is marked @expectedFailure with the concrete defect. Production code is
    intentionally NOT modified.
    """

    @unittest.expectedFailure
    def test_top1gating_no_drop_should_return_weighted_experts(self):
        """BUG (top1gating, no-drop branch): the capacity line reads
            capacity = int(min(new_capacity, paddle.tensor(mask1.size(0))))
        `paddle.tensor` is a module (not callable) and Tensor.size is an int
        property (not callable), so the no-drop path raises before returning.
        Additionally `gates = gates / gates * mask1_float` collapses the gate
        value to 1.0 (or NaN on zeros) instead of keeping the softmax weight.
        Correct behavior: return combine_weights [S, E, capacity] carrying the
        selected expert's gate probability.
        """
        gate = _make_gate(
            num_experts=4, use_rts=False
        )  # default -> drop_tokens=False
        logits = paddle.to_tensor(
            [[0.1, 0.4, 0.3, 0.2], [0.5, 0.1, 0.1, 0.3]], dtype="float32"
        )
        cap, cw, dm, exp_counts, l_aux, l_zloss = gate.top1gating(logits)
        gates = _np_softmax(
            [[0.1, 0.4, 0.3, 0.2], [0.5, 0.1, 0.1, 0.3]], axis=1
        )
        # token0 argmax expert1, token1 argmax expert0; weight == softmax prob.
        self.assertEqual(list(cw.shape[:2]), [2, 4])
        per_expert = cw.numpy().sum(axis=2)
        np.testing.assert_allclose(per_expert[0, 1], gates[0, 1], atol=1e-5)
        np.testing.assert_allclose(per_expert[1, 0], gates[1, 0], atol=1e-5)

    @unittest.expectedFailure
    def test_routed_scaling_factor_should_scale_weights(self):
        """BUG (topkgating): `top_gate = top_gate * routed_scaling_factor` is
        computed but `top_gate` is never consumed on the no-drop path (the
        combine weights are built from `gates * mask`), so routed_scaling_factor
        never reaches the output. Correct behavior: doubling the factor doubles
        the (un-normalized) combine weights.
        """
        gates = paddle.to_tensor(
            [[[0.1, 0.4, 0.3, 0.2], [0.5, 0.1, 0.1, 0.3]]], dtype="float32"
        )
        g1 = _make_gate(
            num_experts=4,
            top_k=2,
            topk_method="greedy",
            norm_topk_prob=False,
            routed_scaling_factor=1.0,
        )
        g3 = _make_gate(
            num_experts=4,
            top_k=2,
            topk_method="greedy",
            norm_topk_prob=False,
            routed_scaling_factor=3.0,
        )
        cw1 = g1.topkgating(gates)[1].numpy()
        cw3 = g3.topkgating(gates)[1].numpy()
        np.testing.assert_allclose(cw3, 3.0 * cw1, atol=1e-6)

    @unittest.expectedFailure
    def test_part1_part2_should_match_monolithic_topkgating(self):
        """BUG (topkgating_part1/part2, no-drop): part1 sets self.capacity=None,
        then part2 calls _one_hot_to_float(locations*mask, self.capacity) i.e.
        F.one_hot(..., num_classes=None), which fails. The split path should
        reproduce the monolithic topkgating combine weights.
        """
        gates = paddle.to_tensor(
            [[[0.1, 0.4, 0.3, 0.2], [0.5, 0.1, 0.1, 0.3]]], dtype="float32"
        )
        mono = _make_gate(
            num_experts=4, top_k=2, topk_method="greedy", norm_topk_prob=False
        )
        expected_cw = mono.topkgating(gates)[1].numpy()

        split = _make_gate(
            num_experts=4, top_k=2, topk_method="greedy", norm_topk_prob=False
        )
        split.topkgating_part1(gates, None)
        combine_weights, _ = split.topkgating_part2(gates.reshape([-1, 4]))
        np.testing.assert_allclose(
            combine_weights.numpy(), expected_cw, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
