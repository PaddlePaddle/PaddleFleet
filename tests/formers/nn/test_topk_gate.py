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

"""Behaviour tests for ``paddlefleet.nn.moe.topk_gate``.

Scope: model layer / MoE gating, no-card (CPU only). Every expected value is
derived independently with NumPy / hand arithmetic from tiny, sign- and
value-distinguishable inputs, so the tests reject swapped experts, wrong
matmul orientation, a dropped fp32 cast, missing normalization, a wrong
capacity formula, the concat-vs-interleave weight-merge bug, and the
correction-bias-as-load-count distinction.

What this module does NOT try to prove (recorded as explicit skips):
  * ``global_aux_loss`` / the ``dist.all_gather`` cross-rank averaging inside
    ``_cal_aux_loss`` and the ``dist.stream.all_reduce`` over the correction
    dispatch mask: proving these needs a real process group. Faking
    ``world_size`` + mocking the collective would only assert local
    orchestration, never the cross-rank numerics (see antipattern 13), so the
    ``group`` here is ``None`` and only single-rank (``tokens_mask is None``)
    paths are exercised.
  * The fused ``cal_aux_loss`` incubate op (taken when
    ``gate_prob.shape[0] >= gate_prob.shape[1]``) is a compiled kernel; the
    aux-loss tests deliberately use ``rows < cols`` so the pure-paddle fallback
    branch runs and can be checked against an independent me*ce reference.

The real production entry (``TopKGate`` / ``gate_detach_matmul`` /
``FusedGateDetachMatmul`` / ``masked_fill`` / ``compute_optimal_transport``)
stays in the verification chain -- nothing under test is re-implemented to
generate its own expected.
"""

import unittest

import numpy as np
import paddle
import paddle.nn.functional as F

# No-card: force CPU and keep any visible accelerator out of the math.
paddle.set_device("cpu")

from paddlefleet.nn.moe.topk_gate import (  # noqa: E402
    FusedGateDetachMatmul,
    TopKGate,
    cast_if_needed,
    compute_optimal_transport,
    gate_detach_matmul,
    masked_fill,
)


class _GateConfig:
    """Minimal, independently-defined config exposing exactly the attributes
    ``TopKGate`` reads. Attribute access + a dict-style ``get`` mirror the real
    config protocol; values are chosen per-test, not copied from any fixture.
    """

    _DEFAULTS = dict(
        hidden_size=8,
        moe_num_experts=4,
        moe_capacity=[1.0, 1.0, 1.0],
        moe_k=2,
        fuse_gate_detach_matmul=False,
        scoring_func="softmax",
        global_aux_loss=False,
        sinkhorn_2gate=False,
        sinkhorn_temp=1.0,
        moe_use_aux_free=False,
        router_aux_loss_coef=0.01,
        router_z_loss_coef=0.0,
        moe_orthogonal_loss_lambda=0.0,
        moe_norm_gate_logits=False,
        moe_group_experts=False,
        moe_use_token_type_bias=False,
        moe_world_size=1,
        multimodel_experts=False,
        moe_use_hard_gate=False,
        moe_group_orthogonal_loss=False,
    )

    def __init__(self, **overrides):
        for key, value in {**self._DEFAULTS, **overrides}.items():
            setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)


def _build_gate(config, gate_weight=None):
    return TopKGate(config, layer_idx=0, group=None, gate_weight=gate_weight)


class TestMaskedFill(unittest.TestCase):
    """masked_fill must overwrite exactly the True positions and leave every
    other value untouched (content, not just the filled ones)."""

    def test_fills_only_masked_positions(self):
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32"
        )
        mask = paddle.to_tensor(
            [[True, False, False], [False, False, True]], dtype="bool"
        )
        out = masked_fill(x, mask, -9.0)
        expected = np.array(
            [[-9.0, 2.0, 3.0], [4.0, 5.0, -9.0]], dtype=np.float32
        )
        np.testing.assert_array_equal(out.numpy(), expected)
        # original tensor is not mutated in place
        np.testing.assert_array_equal(
            x.numpy(), np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], np.float32)
        )


class TestCastIfNeeded(unittest.TestCase):
    """Same dtype returns the identical object; a different dtype casts while
    preserving the numeric values."""

    def test_same_dtype_is_identity(self):
        x = paddle.to_tensor([1.0, 2.0], dtype="float32")
        out = cast_if_needed(x, paddle.float32)
        self.assertIs(out, x)

    def test_different_dtype_casts_and_preserves_values(self):
        x = paddle.to_tensor([1.0, 2.0, 3.0], dtype="float64")
        out = cast_if_needed(x, paddle.float32)
        self.assertEqual(out.dtype, paddle.float32)
        np.testing.assert_allclose(out.numpy(), [1.0, 2.0, 3.0], rtol=0, atol=0)


class TestGateDetachMatmul(unittest.TestCase):
    """The routing logits are ``input @ weight`` computed in fp32. Fused and
    non-fused paths must agree with the independent NumPy matmul, and a
    low-precision input must be up-cast to fp32 before the matmul."""

    def _fixed_inputs(self):
        x = np.array(
            [[1.0, -2.0, 0.5, 3.0], [-1.0, 0.0, 2.0, -0.5]], dtype=np.float32
        )
        w = np.array(
            [
                [0.5, -1.0, 2.0],
                [1.0, 0.0, -0.5],
                [-2.0, 1.5, 0.0],
                [0.25, -0.5, 1.0],
            ],
            dtype=np.float32,
        )
        return x, w

    def test_unfused_matches_numpy(self):
        x, w = self._fixed_inputs()
        out = gate_detach_matmul(
            paddle.to_tensor(x), paddle.to_tensor(w), use_fuse=False
        )
        np.testing.assert_allclose(out.numpy(), x @ w, rtol=1e-6, atol=1e-6)

    def test_fused_matches_unfused_and_numpy(self):
        x, w = self._fixed_inputs()
        out = gate_detach_matmul(
            paddle.to_tensor(x), paddle.to_tensor(w), use_fuse=True
        )
        np.testing.assert_allclose(out.numpy(), x @ w, rtol=1e-5, atol=1e-5)
        self.assertEqual(out.dtype, paddle.float32)

    def test_low_precision_input_upcast_to_fp32(self):
        x, w = self._fixed_inputs()
        out = gate_detach_matmul(
            paddle.to_tensor(x, dtype="float16"),
            paddle.to_tensor(w),
            use_fuse=False,
        )
        self.assertEqual(out.dtype, paddle.float32)
        np.testing.assert_allclose(out.numpy(), x @ w, rtol=1e-2, atol=1e-2)


class TestFusedGateDetachMatmul(unittest.TestCase):
    """Forward equals fp32 F.linear; backward returns the true matmul
    gradients; and the LoRA fix means a stop_gradient weight yields ``None``
    weight-grad while the input still receives its gradient."""

    def _fixed(self):
        x = np.array([[1.0, -2.0, 0.5], [3.0, 0.0, -1.0]], dtype=np.float32)
        w = np.array([[2.0, -1.0], [0.5, 1.0], [-1.5, 0.25]], dtype=np.float32)
        g = np.array(
            [[1.0, -2.0], [0.5, 3.0]], dtype=np.float32
        )  # upstream grad
        return x, w, g

    def test_forward_matches_reference(self):
        x, w, _ = self._fixed()
        out = FusedGateDetachMatmul.apply(
            paddle.to_tensor(x), paddle.to_tensor(w)
        )
        np.testing.assert_allclose(out.numpy(), x @ w, rtol=1e-6, atol=1e-6)

    def test_backward_grads_when_weight_trainable(self):
        x, w, g = self._fixed()
        xt = paddle.to_tensor(x)
        wt = paddle.to_tensor(w)
        xt.stop_gradient = False
        wt.stop_gradient = False
        out = FusedGateDetachMatmul.apply(xt, wt)
        out.backward(paddle.to_tensor(g))
        self.assertIsNotNone(xt.grad)
        self.assertIsNotNone(wt.grad)
        # dX = g @ W^T ; dW = X^T @ g  -- scale-sensitive, not just non-None.
        np.testing.assert_allclose(
            xt.grad.numpy(), g @ w.T, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            wt.grad.numpy(), x.T @ g, rtol=1e-5, atol=1e-5
        )

    def test_stop_gradient_weight_gets_no_grad_but_input_does(self):
        x, w, g = self._fixed()
        xt = paddle.to_tensor(x)
        wt = paddle.to_tensor(w)
        xt.stop_gradient = False
        wt.stop_gradient = True  # frozen weight (LoRA base)
        out = FusedGateDetachMatmul.apply(xt, wt)
        out.backward(paddle.to_tensor(g))
        self.assertIsNone(wt.grad)
        self.assertIsNotNone(xt.grad)
        np.testing.assert_allclose(
            xt.grad.numpy(), g @ w.T, rtol=1e-5, atol=1e-5
        )


class TestGetCapacity(unittest.TestCase):
    """capacity = int(cap * num_tokens // num_experts), with cap chosen by the
    train / eval-small / eval-large branch or an explicit override, and a
    strict > 0 assertion. Each branch uses a distinct cap so a wrong branch
    selection is visible in the numeric result."""

    def _gate(self, **cfg):
        return _build_gate(_GateConfig(moe_num_experts=4, **cfg))

    def test_training_uses_cap0(self):
        gate = self._gate(moe_capacity=[1.0, 1.5, 2.0])
        gate.training = True
        # 1.0 * 16 // 4 = 4
        self.assertEqual(gate.get_capacity(16), 4)

    def test_eval_small_tokens_uses_cap2(self):
        gate = self._gate(moe_capacity=[1.0, 1.5, 2.0])
        gate.training = False
        # num_tokens(3) < num_experts(4) -> cap[2]=2.0 ; 2.0 * 3 // 4 = 1
        self.assertEqual(gate.get_capacity(3), 1)

    def test_eval_large_tokens_uses_cap1(self):
        gate = self._gate(moe_capacity=[1.0, 1.5, 2.0])
        gate.training = False
        # num_tokens(16) >= num_experts(4) -> cap[1]=1.5 ; 1.5 * 16 // 4 = 6
        self.assertEqual(gate.get_capacity(16), 6)

    def test_explicit_cap_factor_overrides_branch(self):
        gate = self._gate(moe_capacity=[1.0, 1.5, 2.0])
        gate.training = True  # would pick cap[0] but factor wins
        # 2.0 * 16 // 4 = 8
        self.assertEqual(gate.get_capacity(16, cap_factor=2.0), 8)

    def test_zero_capacity_raises(self):
        gate = self._gate(moe_capacity=[0.001, 0.001, 0.001])
        gate.training = True
        # 0.001 * 8 // 4 == 0 -> assertion fails
        with self.assertRaises(AssertionError):
            gate.get_capacity(8)


class TestForward(unittest.TestCase):
    """forward returns (logits, capacity, router_loss). logits is the real
    fp32 ``input @ weight`` routing projection; capacity follows get_capacity;
    router_loss is a fresh zeros[1] that stays differentiable."""

    def test_forward_logits_capacity_and_router_loss(self):
        H, E, S = 3, 4, 8
        weight = np.array(
            [
                [0.5, -1.0, 2.0, 0.25],
                [1.0, 0.0, -0.5, 1.5],
                [-2.0, 1.5, 0.0, -1.0],
            ],
            dtype=np.float32,
        )
        x = np.array(
            [
                [1.0, -2.0, 0.5],
                [-1.0, 0.0, 2.0],
                [0.5, 1.0, -1.0],
                [2.0, -0.5, 0.0],
                [0.0, 3.0, -2.0],
                [-1.5, 0.5, 1.0],
                [1.0, 1.0, 1.0],
                [-2.0, -2.0, 2.0],
            ],
            dtype=np.float32,
        )
        config = _GateConfig(
            hidden_size=H, moe_num_experts=E, moe_capacity=[1.0, 1.0, 1.0]
        )
        gate = _build_gate(config, gate_weight=paddle.to_tensor(weight))
        gate.training = True

        logits, capacity, router_loss = gate(paddle.to_tensor(x))

        np.testing.assert_allclose(
            logits.numpy(), x @ weight, rtol=1e-6, atol=1e-6
        )
        self.assertEqual(list(logits.shape), [S, E])
        self.assertEqual(capacity, 1 * S // E)  # == 2
        self.assertIsInstance(capacity, int)
        self.assertEqual(list(router_loss.shape), [1])
        np.testing.assert_array_equal(router_loss.numpy(), [0.0])
        self.assertFalse(router_loss.stop_gradient)


class TestGetGateWeightMerge(unittest.TestCase):
    """For multimodel experts, get_gate_weight merges the per-modality gate
    weights. transform_weight=False is a plain concat; transform_weight=True
    interleaves by local-expert / world-rank layout. The two must differ, and
    each must match its own hand-derived expected -- this rejects the
    concat-vs-interleave swap that a shape-only check would miss."""

    def _multimodel_gate(self):
        config = _GateConfig(
            hidden_size=2,
            moe_num_experts=[4, 4],
            multimodel_experts=True,
            moe_world_size=2,
            moe_group_experts=False,
        )
        gate = _build_gate(config)
        w0 = np.array([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], np.float32)
        w1 = np.array(
            [[100.0, 101.0, 102.0, 103.0], [104.0, 105.0, 106.0, 107.0]],
            np.float32,
        )
        gate.weight.set_value(paddle.to_tensor(w0))
        gate.weight_1.set_value(paddle.to_tensor(w1))
        return gate, w0, w1

    def test_no_transform_is_concat(self):
        gate, w0, w1 = self._multimodel_gate()
        out = gate.get_gate_weight(transform_weight=False)
        np.testing.assert_array_equal(
            out.numpy(), np.concatenate([w0, w1], axis=-1)
        )

    def test_transform_interleaves_by_rank(self):
        gate, w0, w1 = self._multimodel_gate()
        out = gate.get_gate_weight(transform_weight=True)
        # world_size=2: rank0 holds experts 0,1 of each modality, rank1 holds
        # experts 2,3. Flattened row = [w0[0:2], w1[0:2], w0[2:4], w1[2:4]].
        expected = np.stack(
            [
                np.concatenate([w0[h, 0:2], w1[h, 0:2], w0[h, 2:4], w1[h, 2:4]])
                for h in range(2)
            ]
        )
        np.testing.assert_array_equal(out.numpy(), expected)
        # interleave must NOT equal the plain concat
        self.assertFalse(
            np.array_equal(out.numpy(), np.concatenate([w0, w1], axis=-1))
        )


class TestZLoss(unittest.TestCase):
    """z-loss = mean over rows of (logsumexp_over_experts(logits))^2, or the
    mask-weighted average when a loss_mask is given. Verified against an
    independent NumPy logsumexp."""

    def _logits(self):
        return np.array(
            [
                [1.0, -2.0, 0.5, 3.0],
                [-1.0, 0.0, 2.0, -0.5],
                [0.25, 0.25, 0.25, 0.25],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _lse(a):
        m = a.max(axis=1, keepdims=True)
        return m[:, 0] + np.log(np.exp(a - m).sum(axis=1))

    def test_z_loss_no_mask(self):
        logits = self._logits()
        gate = _build_gate(_GateConfig())
        out = gate._cal_z_loss(paddle.to_tensor(logits))
        expected = np.mean(self._lse(logits) ** 2)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_z_loss_with_mask(self):
        logits = self._logits()
        mask = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        gate = _build_gate(_GateConfig())
        out = gate._cal_z_loss(
            paddle.to_tensor(logits), loss_mask=paddle.to_tensor(mask)
        )
        lse2 = self._lse(logits) ** 2
        expected = (lse2 * mask).sum() / max(mask.sum(), 1e-6)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)
        # the masked-out row (row 1) must not affect the result
        self.assertFalse(np.allclose(out.numpy(), np.mean(lse2)))


class TestOrthogonalLoss(unittest.TestCase):
    """orthogonal loss row-normalizes the transposed gate weight, forms
    G = W_hat @ W_hat^T, and returns ||G - I||_2^2 / G.size. Orthonormal
    experts give ~0; a non-orthogonal weight matches the independent value."""

    def test_orthonormal_weight_is_near_zero(self):
        # weight is [H, E]; transpose -> [E, H]. Identity rows stay identity
        # after normalization, so G = I and loss = 0.
        weight = np.eye(4, dtype=np.float32)
        gate = _build_gate(
            _GateConfig(hidden_size=4, moe_num_experts=4),
            gate_weight=paddle.to_tensor(weight),
        )
        out = gate._cal_orthogonal_loss()
        self.assertEqual(list(out.shape), [1])
        np.testing.assert_allclose(out.numpy(), [0.0], atol=1e-6)

    def test_non_orthogonal_weight_matches_reference(self):
        weight = np.array(
            [
                [1.0, 0.5, -1.0, 2.0],
                [0.0, 1.0, 1.0, -0.5],
                [2.0, -1.0, 0.5, 1.0],
            ],
            dtype=np.float32,
        )  # [H=3, E=4]
        gate = _build_gate(
            _GateConfig(hidden_size=3, moe_num_experts=4),
            gate_weight=paddle.to_tensor(weight),
        )
        out = gate._cal_orthogonal_loss()

        w = weight.T  # [E, H]
        wnorm = np.linalg.norm(w, axis=1, keepdims=True)
        w_hat = w / np.maximum(wnorm, 1e-12)
        g = w_hat @ w_hat.T
        d = g - np.eye(4, dtype=np.float32)
        expected = (d**2).sum() / d.size
        np.testing.assert_allclose(
            out.numpy(), [expected], rtol=1e-5, atol=1e-6
        )


class TestAuxLoss(unittest.TestCase):
    """Auxiliary (load-balancing) loss along the pure-paddle fallback branch,
    forced by rows < cols so the compiled cal_aux_loss op is not used.

    Fallback contract (no tokens_mask): seqlen = numel/num_experts = rows;
    me = sum(gate_prob, axis=0)/seqlen ; ce = dispatch_mask/seqlen ;
    l_aux = sum(me*ce)*num_experts, divided by moe_k when use_group.

    Correction bias (moe_use_aux_free) is a *selection* signal: it makes the
    load count come from the gate_prob's own top-k, so a garbage dispatch_mask
    is ignored -- proving the bias drives which experts are counted, not the
    weight applied.
    """

    # 3 tokens x 8 experts -> rows(3) < cols(8) -> fallback branch.
    _GATE_PROB = np.array(
        [
            [0.30, 0.05, 0.20, 0.10, 0.04, 0.11, 0.15, 0.05],
            [0.05, 0.25, 0.06, 0.31, 0.10, 0.09, 0.08, 0.06],
            [0.12, 0.11, 0.09, 0.08, 0.07, 0.13, 0.10, 0.30],
        ],
        dtype=np.float32,
    )

    def _fallback_ref(self, gate_prob, dispatch_mask, num_experts, moe_k=None):
        rows = gate_prob.shape[0]
        seqlen = rows * gate_prob.shape[1] / num_experts
        me = gate_prob.sum(axis=0) / seqlen
        ce = dispatch_mask.astype(np.float32) / seqlen
        l_aux = float((me * ce).sum() * num_experts)
        if moe_k is not None:
            l_aux = l_aux / moe_k
        return l_aux

    def test_fallback_uses_provided_dispatch_mask(self):
        gate = _build_gate(
            _GateConfig(moe_num_experts=8, moe_k=2, moe_use_aux_free=False)
        )
        dispatch_mask = np.array([2, 0, 1, 3, 0, 0, 1, 1], dtype=np.int64)
        out = gate._cal_aux_loss(
            paddle.to_tensor(self._GATE_PROB),
            paddle.to_tensor(dispatch_mask),
        )
        expected = self._fallback_ref(self._GATE_PROB, dispatch_mask, 8)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_use_group_divides_by_moe_k(self):
        gate = _build_gate(
            _GateConfig(moe_num_experts=8, moe_k=2, moe_use_aux_free=False)
        )
        dispatch_mask = np.array([2, 0, 1, 3, 0, 0, 1, 1], dtype=np.int64)
        base = gate._cal_aux_loss(
            paddle.to_tensor(self._GATE_PROB),
            paddle.to_tensor(dispatch_mask),
            use_group=False,
        )
        grouped = gate._cal_aux_loss(
            paddle.to_tensor(self._GATE_PROB),
            paddle.to_tensor(dispatch_mask),
            use_group=True,
        )
        np.testing.assert_allclose(
            grouped.numpy(), base.numpy() / 2, rtol=1e-5, atol=1e-6
        )

    def test_correction_bias_recomputes_counts_from_topk(self):
        """With moe_use_aux_free=True the passed dispatch_mask is discarded and
        the load counts are rebuilt from the top-k of gate_prob. Depends on the
        int_bincount op being available on CPU.
        """
        gate = _build_gate(
            _GateConfig(moe_num_experts=8, moe_k=2, moe_use_aux_free=True)
        )
        # Independent top-2 counts over 8 experts from _GATE_PROB.
        counts = np.zeros(8, dtype=np.int64)
        for row in self._GATE_PROB:
            for idx in np.argsort(row)[-2:]:
                counts[idx] += 1
        expected = self._fallback_ref(self._GATE_PROB, counts, 8)

        garbage_mask = np.full([8], 999, dtype=np.int64)  # must be ignored
        out = gate._cal_aux_loss(
            paddle.to_tensor(self._GATE_PROB),
            paddle.to_tensor(garbage_mask),
        )
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)
        # sanity: had the garbage mask been used, the value would differ.
        garbage_val = self._fallback_ref(self._GATE_PROB, garbage_mask, 8)
        self.assertFalse(np.allclose(expected, garbage_val))


class TestComputeOptimalTransport(unittest.TestCase):
    """Sinkhorn optimal transport (used by sinkhorn_2gate routing). Its defining
    contract is checked -- non-negative, finite, and marginals matched: after
    the final column scaling the column sums equal the target c, and the row
    sums approach r. The property is verified independently, not by re-running
    the same iteration."""

    def test_marginals_are_matched(self):
        M = paddle.to_tensor(
            [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]], dtype="float32"
        )
        r = paddle.ones([3], dtype="float32")
        c = paddle.ones([3], dtype="float32")
        P, _ = compute_optimal_transport(M, r, c, lam=1.0, max_iters=50)
        p = P.numpy()
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue((p >= 0).all())
        # last operation in the loop is a column rescale -> column sums == c.
        np.testing.assert_allclose(p.sum(axis=0), [1.0, 1.0, 1.0], atol=1e-4)
        # row sums approach r (looser, Sinkhorn alternation).
        np.testing.assert_allclose(p.sum(axis=1), [1.0, 1.0, 1.0], atol=1e-2)


if __name__ == "__main__":
    unittest.main()
