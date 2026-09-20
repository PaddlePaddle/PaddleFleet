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

"""CPU-only behavior tests for the device-independent pure logic in
``paddlefleet.transformer.moe.moe_router``.

Covered (all runnable on CPU, no collectives, no ``world_size`` faking):
  * ``StandardMoERouter._capacity`` - per-expert capacity integer arithmetic
    and its min/max clamps and the ``capacity > 0`` assertion.
  * ``StandardMoERouter._cal_aux_loss`` - the load-balancing auxiliary loss
    ``sum(mean(gates) * mean(mask)) * num_experts``.
  * ``StandardMoERouter._priority`` - capacity-limited cumulative priority that
    actually DROPS the later token that overflows an expert.
  * ``StandardMoERouter._topk_group_limited_greedy`` - group pre-selection then
    top-k, verifying the chosen expert ids and their gate weights.
  * ``HFBitexactSoftmax`` - the fp32 softmax PyLayer forward AND its custom
    backward against an independent numpy Jacobian-vector reference.
  * ``FusedGateDetachMatmul`` - the (non-hf, non-accuracy) fused gate matmul
    forward ``x @ w.T`` AND backward (both ``x`` and ``w`` gradients).
  * ``StandardMoERouter._probs_drop_policy`` - a genuine production defect is
    documented via ``@unittest.expectedFailure`` (see that test).

The pure math methods are invoked as real *unbound* methods with a minimal
stand-in ``self`` that only carries the data attributes they read
(``num_experts``). The production method body runs unchanged; no formula is
re-implemented in the test. Constructing a full ``StandardMoERouter`` needs a
complete ``TransformerConfig`` plus a live parallel context, which is out of
scope for CPU pure-logic verification.

Every expected value is derived by hand / independent numpy, never by calling
the code under test. Distributed routing paths (sequence/context/expert
parallel gathers inside ``_cal_seq_aux_loss``/``_cal_z_loss``) are intentionally
NOT covered: faking ``world_size`` + mocking collectives would only prove local
orchestration, not cross-rank semantics.
"""

import unittest

import numpy as np

try:
    import paddle

    from paddlefleet.transformer.moe.moe_router import (
        FusedGateDetachMatmul,
        HFBitexactSoftmax,
        StandardMoERouter,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest dependency probe
    paddle = None
    FusedGateDetachMatmul = None
    HFBitexactSoftmax = None
    StandardMoERouter = None
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle / paddlefleet not importable on this CPU host: {_IMPORT_ERROR!r}"
)


class _RouterStub:
    """Minimal ``self`` carrying only the data attributes the pure methods read.

    It holds no behavior: every method under test is the real, unmodified
    production method invoked as ``StandardMoERouter.<method>(stub, ...)``.
    """

    def __init__(self, num_experts):
        self.num_experts = num_experts


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestCapacity(unittest.TestCase):
    """``_capacity`` = clamp(int((num_tokens // num_experts) * factor))."""

    def setUp(self):
        paddle.set_device("cpu")

    def _capacity(self, num_tokens, num_experts, factor, max_cap, min_cap):
        gates = paddle.zeros([num_tokens, num_experts], dtype="float32")
        stub = _RouterStub(num_experts)
        return StandardMoERouter._capacity(
            stub, gates, factor, max_cap, min_cap
        )

    def test_plain_value(self):
        # 8 // 2 = 4; int(4 * 1.5) = 6; inside [1, 100] -> 6.
        self.assertEqual(self._capacity(8, 2, 1.5, 100, 1), 6)

    def test_truncates_toward_zero(self):
        # 10 // 3 = 3; int(3 * 1.4) = int(4.2) = 4 (floor, not round).
        self.assertEqual(self._capacity(10, 3, 1.4, 100, 1), 4)

    def test_min_clamp_raises_floor(self):
        # 4 // 4 = 1; int(1 * 0.5) = 0; below min_capacity=2 -> clamped to 2.
        self.assertEqual(self._capacity(4, 4, 0.5, 100, 2), 2)

    def test_max_clamp_lowers_ceiling(self):
        # 100 // 2 = 50; int(50 * 2.0) = 100; above max_capacity=8 -> 8.
        self.assertEqual(self._capacity(100, 2, 2.0, 8, 1), 8)

    def test_zero_capacity_trips_assertion(self):
        # 4 // 4 = 1; int(1 * 0.0) = 0; min=0 keeps it 0 -> assert capacity > 0.
        with self.assertRaises(AssertionError):
            self._capacity(4, 4, 0.0, 100, 0)

    def test_non_2d_gates_rejected(self):
        stub = _RouterStub(2)
        with self.assertRaises(AssertionError):
            StandardMoERouter._capacity(
                stub, paddle.zeros([4], dtype="float32"), 1.0, 100, 1
            )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestCalAuxLoss(unittest.TestCase):
    """``_cal_aux_loss`` = sum(mean_tok(gates) * mean_tok(mask)) * num_experts."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_matches_independent_reference(self):
        num_experts = 3
        gates = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.1, 0.6, 0.3],
                [0.4, 0.4, 0.2],
                [0.0, 0.5, 0.5],
            ],
            dtype=np.float32,
        )
        mask = np.array(
            [[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 1]], dtype=np.float32
        )
        # Independent hand reference (no call into the code under test).
        me = gates.mean(axis=0)
        ce = mask.mean(axis=0)
        expected = float((me * ce).sum() * num_experts)

        stub = _RouterStub(num_experts)
        out = StandardMoERouter._cal_aux_loss(
            stub,
            paddle.to_tensor(gates),
            paddle.to_tensor(mask),
        )
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_scales_with_num_experts_factor(self):
        # Same per-column means but the trailing ``* num_experts`` factor must
        # be consumed: doubling num_experts (with matching width) doubles loss.
        gates = paddle.to_tensor([[0.5, 0.5], [0.25, 0.75]], dtype="float32")
        mask = paddle.to_tensor([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
        me = np.array([0.375, 0.625], dtype=np.float32)
        ce = np.array([0.5, 0.5], dtype=np.float32)
        expected = float((me * ce).sum() * 2)
        out = StandardMoERouter._cal_aux_loss(_RouterStub(2), gates, mask)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestPriority(unittest.TestCase):
    """``_priority`` keeps a token->expert slot only while the running per-expert
    count is within ``capacity``; overflowing later tokens are dropped."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_capacity_drops_overflowing_token(self):
        # 3 tokens, top_k=1, num_experts=3, capacity=1.
        #   token0 -> e0, token1 -> e0 (e0 already full -> DROP), token2 -> e1.
        # Row-major cumsum over the flattened [token*k] slots enforces the
        # first-come rule, so the expected assignment is:
        #   token0: e0 kept, token1: nothing, token2: e1 kept.
        topk_idx = paddle.to_tensor([[0], [0], [1]], dtype="int64")
        out = StandardMoERouter._priority(_RouterStub(3), topk_idx, capacity=1)
        expected = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_capacity_two_keeps_both(self):
        # capacity=2: the two tokens routed to e0 both fit; the third to e1.
        topk_idx = paddle.to_tensor([[0], [0], [1]], dtype="int64")
        out = StandardMoERouter._priority(_RouterStub(3), topk_idx, capacity=2)
        expected = np.array(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_topk_two_collapses_to_per_token_expert_set(self):
        # top_k=2 with plenty of capacity: each token's row marks exactly the
        # experts it selected (order within k does not matter after the sum).
        topk_idx = paddle.to_tensor([[0, 2], [1, 2]], dtype="int64")
        out = StandardMoERouter._priority(_RouterStub(3), topk_idx, capacity=8)
        expected = np.array(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32
        )
        np.testing.assert_array_equal(out.numpy(), expected)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestTopkGroupLimitedGreedy(unittest.TestCase):
    """``_topk_group_limited_greedy``: pick ``topk_group`` groups by their max
    score, zero out the rest, then top-k experts over the masked scores."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_group_preselection_then_topk(self):
        # 4 experts, n_group=2 -> group0={e0,e1}, group1={e2,e3}; topk_group=1.
        #   token0: group maxes = [0.2, 0.9] -> keep group1 -> top1 expert = e2.
        #   token1: group maxes = [0.8, 0.2] -> keep group0 -> top1 expert = e0.
        scores = paddle.to_tensor(
            [[0.1, 0.2, 0.9, 0.3], [0.8, 0.1, 0.2, 0.05]], dtype="float32"
        )
        weight, idx = StandardMoERouter._topk_group_limited_greedy(
            _RouterStub(4), scores, k=1, n_group=2, topk_group=1
        )
        self.assertEqual(idx.numpy().tolist(), [[2], [0]])
        np.testing.assert_allclose(
            weight.numpy(),
            np.array([[0.9], [0.8]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_masked_group_expert_cannot_be_selected(self):
        # A large score living in the NON-selected group must be suppressed:
        #   token0: group maxes = [0.95, 0.9] -> keep group0 -> e0 (0.95),
        #   NOT e2 (0.9) even though it is the 2nd highest overall.
        scores = paddle.to_tensor([[0.95, 0.05, 0.9, 0.1]], dtype="float32")
        weight, idx = StandardMoERouter._topk_group_limited_greedy(
            _RouterStub(4), scores, k=1, n_group=2, topk_group=1
        )
        self.assertEqual(idx.numpy().tolist(), [[0]])
        np.testing.assert_allclose(
            weight.numpy(), np.array([[0.95]], dtype=np.float32), atol=1e-6
        )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestHFBitexactSoftmax(unittest.TestCase):
    """``HFBitexactSoftmax`` fp32 softmax forward and its custom backward."""

    def setUp(self):
        paddle.set_device("cpu")

    @staticmethod
    def _np_softmax(x):
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def test_forward_matches_reference_softmax(self):
        x_np = np.array(
            [[1.0, 2.0, -1.0, 0.5], [0.0, 0.0, 3.0, -2.0]], dtype=np.float32
        )
        out = HFBitexactSoftmax.apply(paddle.to_tensor(x_np))
        np.testing.assert_allclose(
            out.numpy(), self._np_softmax(x_np), rtol=1e-6, atol=1e-6
        )
        # Softmax rows sum to 1 - guards against an un-normalized regression.
        np.testing.assert_allclose(
            out.numpy().sum(axis=-1), np.ones(2, np.float32), atol=1e-6
        )

    def test_backward_matches_independent_jvp(self):
        # Distinguishable, non-uniform upstream grad so a wrong epilogue (e.g.
        # forgetting the ``- sum(g*p) * p`` term) cannot cancel out.
        x_np = np.array(
            [[1.0, 2.0, -1.0, 0.5], [0.0, 0.0, 3.0, -2.0]], dtype=np.float32
        )
        g_np = np.array(
            [[0.3, -0.7, 0.2, 1.1], [-0.5, 0.4, 0.9, -0.2]], dtype=np.float32
        )
        x = paddle.to_tensor(x_np)
        x.stop_gradient = False
        out = HFBitexactSoftmax.apply(x)
        out.backward(paddle.to_tensor(g_np))

        # Independent softmax Jacobian-vector product: dx = p*(g - sum(g*p)).
        p = self._np_softmax(x_np)
        inner = (g_np * p).sum(axis=-1, keepdims=True)
        dx_ref = g_np * p - inner * p

        self.assertIsNotNone(x.grad)
        np.testing.assert_allclose(x.grad.numpy(), dx_ref, rtol=1e-5, atol=1e-6)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestFusedGateDetachMatmul(unittest.TestCase):
    """``FusedGateDetachMatmul`` default (non-hf, non-accuracy) path computes
    ``x @ w.T`` and returns matching gradients for both ``x`` and ``w``."""

    def setUp(self):
        paddle.set_device("cpu")

    def _inputs(self):
        # n=2, H=3, E=4; distinguishable, non-uniform so a transpose/order
        # mistake shows up in the numbers.
        x_np = np.array([[1.0, 2.0, -1.0], [0.5, -0.5, 2.0]], dtype=np.float32)
        w_np = np.array(
            [
                [1.0, 0.0, -1.0],
                [2.0, 1.0, 0.5],
                [-1.0, 0.5, 1.5],
                [0.25, -2.0, 1.0],
            ],
            dtype=np.float32,
        )
        return x_np, w_np

    def test_forward_is_x_matmul_w_transpose(self):
        x_np, w_np = self._inputs()
        out = FusedGateDetachMatmul.apply(
            paddle.to_tensor(x_np),
            paddle.to_tensor(w_np),
            False,
            False,
        )
        np.testing.assert_allclose(
            out.numpy(), x_np @ w_np.T, rtol=1e-5, atol=1e-5
        )

    def test_backward_x_and_w_grads(self):
        x_np, w_np = self._inputs()
        g_np = np.array(
            [[0.5, -1.0, 2.0, 0.25], [1.5, 0.0, -0.5, 1.0]], dtype=np.float32
        )
        x = paddle.to_tensor(x_np)
        w = paddle.to_tensor(w_np)
        x.stop_gradient = False
        w.stop_gradient = False
        out = FusedGateDetachMatmul.apply(x, w, False, False)
        out.backward(paddle.to_tensor(g_np))

        # y = x @ w.T  ->  dx = g @ w ,  dw = g.T @ x  (independent numpy).
        dx_ref = g_np @ w_np
        dw_ref = g_np.T @ x_np
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)
        np.testing.assert_allclose(x.grad.numpy(), dx_ref, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(w.grad.numpy(), dw_ref, rtol=1e-5, atol=1e-5)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestProbsDropPolicyKnownBug(unittest.TestCase):
    """``_probs_drop_policy`` is currently broken and cannot run on any paddle
    build: it calls ``paddle.topk(..., dim=0)`` (paddle uses ``axis``, not
    ``dim``) and ``paddle.zeros(num_tokens, num_experts, dtype=paddle.bool)``
    (paddle.zeros takes a single ``shape`` arg, so ``num_experts`` collides with
    the keyword ``dtype``). Either raises ``TypeError`` before any masking runs.

    This test asserts the *intended* per-expert top-``capacity`` behavior and is
    marked ``expectedFailure`` so the defect is recorded without touching
    production code. When the two API calls are fixed, remove the decorator.
    """

    def setUp(self):
        paddle.set_device("cpu")

    @unittest.expectedFailure
    def test_keeps_top_capacity_tokens_per_expert(self):
        # Already-gated scores (zeros for non-selected). capacity=1 -> each
        # expert keeps only its single highest-scoring token.
        #   e0 column [0.9, 0.1, 0.0] -> token0 ; e1 column [0.0, 0.7, 0.5] -> token1.
        scores = paddle.to_tensor(
            [[0.9, 0.0], [0.1, 0.7], [0.0, 0.5]], dtype="float32"
        )
        out = StandardMoERouter._probs_drop_policy(
            _RouterStub(2), scores, capacity=1
        )
        expected = np.array(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32
        )
        np.testing.assert_array_equal(out.numpy(), expected)


if __name__ == "__main__":
    unittest.main()
