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

"""Behavior tests for ernie4_5_moe_vl ... fusion_ops.npu_fusion_ops.

These exercise the REAL entries ``npu_combining`` and
``npu_cal_aux_loss_func`` with fixed, distinguishable tiny inputs and
compare against INDEPENDENT hand-derived expectations (numpy re-derivation
plus literal anchors). No production function is ever used to build its own
expected value.

Environment: 无卡 (CPU only). Despite the ``npu_`` prefix, both functions are
pure paddle ops:
  * ``npu_combining`` = ``F.embedding`` gather + weighted sum.
  * ``npu_cal_aux_loss_func`` (with ``global_aux_loss=False``) = sum / clip /
    broadcast-multiply.
They produce the same numeric result on CPU as on any accelerator, so the
math is validated here on CPU. The ONLY device/multi-card-bound path is
``global_aux_loss=True``, which calls ``paddle.distributed.all_gather`` over a
real process group; that path is NOT exercised here and is covered by an
explicit skip with reason (see TestNpuCalAuxLossGlobalReduce).
"""

import unittest

import numpy as np
import paddle


# --------------------------------------------------------------------------
# Independent references (hand re-derivation; never call production code).
# --------------------------------------------------------------------------
def _combining_ref(x, combine_weights, scatter_index, hard_gate=False):
    """Independent numpy reference for npu_combining.

    x[seq, dim]; combine_weights[s, k]; scatter_index[s, k].
    gathered[i, j] = x[scatter_index[i, j]]  -> [s, k, dim]
    hard_gate: squeeze the k axis (k must be 1); weights are IGNORED.
    else: y[i] = sum_j combine_weights[i, j] * gathered[i, j].
    """
    x = np.asarray(x, dtype=np.float64)
    idx = np.asarray(scatter_index)
    gathered = x[idx]  # [s, k, dim]
    if hard_gate:
        return gathered[:, 0, :]  # squeeze(-2) with k == 1
    w = np.asarray(combine_weights, dtype=np.float64)
    return (w[:, :, None] * gathered).sum(axis=1)


def _aux_loss_ref(
    gate_prob,
    dispatch_mask,
    tokens_mask,
    dispatch_tokens_mask,
    num_experts,
    use_group,
    moe_k,
):
    """Independent numpy reference for npu_cal_aux_loss_func value."""
    gate_prob = np.asarray(gate_prob, dtype=np.float64)
    dispatch_mask = np.asarray(dispatch_mask, dtype=np.float64)

    scale = None
    if dispatch_tokens_mask is not None:
        dtm = np.asarray(dispatch_tokens_mask, dtype=np.float64)
        seqlen_float = dtm.sum()
        if tokens_mask is not None and gate_prob.shape[0] != dtm.shape[0]:
            tm = np.asarray(tokens_mask, dtype=np.float64)
            scale = seqlen_float / max(float(tm.sum()), 1e-6)
    elif tokens_mask is not None:
        seqlen_float = np.asarray(tokens_mask, dtype=np.float64).sum()
    else:
        seqlen_float = gate_prob.size / num_experts
    seqlen_float = max(float(seqlen_float), 1e-6)

    if dispatch_mask.ndim == 2:
        dispatch_mask = dispatch_mask.sum(axis=0)
    ce = dispatch_mask / seqlen_float
    me = gate_prob.sum(axis=0) / seqlen_float
    l_aux = float((me * ce).sum() * num_experts)
    if use_group:
        l_aux = l_aux / moe_k
    if scale is not None:
        # value == l_aux + (scale - 1) * l_aux.detach() == scale * l_aux
        l_aux = l_aux * scale
    return l_aux


class TestNpuCombining(unittest.TestCase):
    """Numeric behavior of the real npu_combining entry (CPU / 无卡)."""

    @classmethod
    def setUpClass(cls):
        paddle.set_device("cpu")

    def _call(self, x, cw, idx, hard_gate=False):
        from paddlefleet.transformers.ernie4_5_moe_vl.model.fusion_ops.npu_fusion_ops import (  # noqa: E501
            npu_combining,
        )

        return npu_combining(
            paddle.to_tensor(x, dtype="float32"),
            paddle.to_tensor(cw, dtype="float32"),
            paddle.to_tensor(idx, dtype="int64"),
            hard_gate=hard_gate,
        )

    def test_weighted_combine_matches_reference(self):
        # x rows are distinguishable; scatter_index picks specific rows so a
        # wrong gather axis / weight broadcast would change the content.
        x = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        cw = [[0.5, 0.5], [1.0, 0.0], [0.2, 0.8]]
        idx = [[0, 1], [1, 2], [2, 0]]
        out = self._call(x, cw, idx)

        self.assertEqual(list(out.shape), [3, 2])
        # Hand-derived:
        #   row0 = .5*[1,2] + .5*[3,4] = [2, 3]
        #   row1 = 1*[3,4] + 0*[5,6]   = [3, 4]
        #   row2 = .2*[5,6] + .8*[1,2] = [1.8, 2.8]
        expected = _combining_ref(x, cw, idx)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            out.numpy(),
            [[2.0, 3.0], [3.0, 4.0], [1.8, 2.8]],
            rtol=1e-6,
            atol=1e-6,
        )

    def test_scatter_index_selects_correct_rows(self):
        # combine_weights = 1 for a single selected column so the output is
        # exactly the gathered row; catches wrong row selection / axis.
        x = [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]]
        cw = [[1.0], [1.0], [1.0]]
        idx = [[2], [0], [1]]
        out = self._call(x, cw, idx)
        np.testing.assert_allclose(
            out.numpy(),
            [[70.0, 80.0, 90.0], [10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
            rtol=1e-6,
            atol=1e-6,
        )

    def test_hard_gate_gathers_rows_and_ignores_weights(self):
        # hard_gate path squeezes the k=1 axis AND does not apply weights.
        # Pass deliberately wrong weights (zeros) to prove they are ignored.
        x = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        cw_zero = [[0.0], [0.0], [0.0]]
        idx = [[2], [0], [1]]
        out = self._call(x, cw_zero, idx, hard_gate=True)

        self.assertEqual(list(out.shape), [3, 2])
        expected = _combining_ref(x, cw_zero, idx, hard_gate=True)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)
        # Gathered rows x[[2,0,1]] regardless of the zero weights.
        np.testing.assert_allclose(
            out.numpy(),
            [[5.0, 6.0], [1.0, 2.0], [3.0, 4.0]],
            rtol=1e-6,
            atol=1e-6,
        )
        # Negative control: the soft path with zero weights would be all zeros,
        # proving hard_gate really bypasses weight application.
        self.assertFalse(np.allclose(out.numpy(), 0.0))


class TestNpuCalAuxLossFunc(unittest.TestCase):
    """Numeric behavior of the real npu_cal_aux_loss_func (CPU / 无卡)."""

    @classmethod
    def setUpClass(cls):
        paddle.set_device("cpu")

    # Shared distinguishable fixture (non-uniform to expose axis errors).
    GATE = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.1, 0.3, 0.1],
    ]  # col sums [.6,.3,.6,.5]
    DMASK = [[1.0, 0.0, 2.0, 0.0], [0.0, 3.0, 0.0, 1.0]]  # sum0 [1,3,2,1]
    NE = 4

    def _call(self, **kw):
        from paddlefleet.transformers.ernie4_5_moe_vl.model.fusion_ops.npu_fusion_ops import (  # noqa: E501
            npu_cal_aux_loss_func,
        )

        return npu_cal_aux_loss_func(**kw)

    def _t(self, data, dtype="float32"):
        return paddle.to_tensor(data, dtype=dtype)

    def test_no_masks_numel_normalization(self):
        # tokens_mask and dispatch_tokens_mask None -> seqlen = numel/E = 8/4 = 2.
        # ce = [1,3,2,1]/2; me = [.6,.3,.6,.5]/2;
        # l_aux = sum(me*ce)*4 = (3.2/4)*4 = 3.2
        l_aux, a, b = self._call(
            gate_prob=self._t(self.GATE),
            dispatch_mask=self._t(self.DMASK),
            tokens_mask=None,
            dispatch_tokens_mask=None,
            num_experts=self.NE,
            use_group=False,
            moe_k=2,
        )
        self.assertEqual(list(l_aux.shape), [])
        self.assertIsNone(a)
        self.assertIsNone(b)
        expected = _aux_loss_ref(
            self.GATE, self.DMASK, None, None, self.NE, False, 2
        )
        np.testing.assert_allclose(
            float(l_aux.numpy()), expected, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            float(l_aux.numpy()), 3.2, rtol=1e-6, atol=1e-6
        )

    def test_tokens_mask_changes_normalization(self):
        # tokens_mask only changes seqlen normalization; it does NOT mask
        # gate_prob. tokens_mask=[1,0] -> seqlen = 1 (not 2).
        # l_aux = sum(([.6,.3,.6,.5])*([1,3,2,1]))*4 = 3.2*4 = 12.8
        l_aux, _, _ = self._call(
            gate_prob=self._t(self.GATE),
            dispatch_mask=self._t(self.DMASK),
            tokens_mask=self._t([1.0, 0.0]),
            dispatch_tokens_mask=None,
            num_experts=self.NE,
            use_group=False,
            moe_k=2,
        )
        expected = _aux_loss_ref(
            self.GATE, self.DMASK, [1.0, 0.0], None, self.NE, False, 2
        )
        np.testing.assert_allclose(
            float(l_aux.numpy()), expected, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            float(l_aux.numpy()), 12.8, rtol=1e-6, atol=1e-6
        )
        # Negative control: differs from the numel-normalized value (3.2),
        # proving tokens_mask.sum() drives seqlen.
        self.assertFalse(np.isclose(float(l_aux.numpy()), 3.2))

    def test_use_group_divides_by_moe_k(self):
        # Same as numel case (3.2) but use_group divides by moe_k=2 -> 1.6.
        l_aux, _, _ = self._call(
            gate_prob=self._t(self.GATE),
            dispatch_mask=self._t(self.DMASK),
            tokens_mask=None,
            dispatch_tokens_mask=None,
            num_experts=self.NE,
            use_group=True,
            moe_k=2,
        )
        expected = _aux_loss_ref(
            self.GATE, self.DMASK, None, None, self.NE, True, 2
        )
        np.testing.assert_allclose(
            float(l_aux.numpy()), expected, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            float(l_aux.numpy()), 1.6, rtol=1e-6, atol=1e-6
        )
        self.assertFalse(np.isclose(float(l_aux.numpy()), 3.2))

    def test_dtype_mismatch_tokens_mask_is_cast(self):
        # int32 tokens_mask must be cast to gate_prob dtype and yield the same
        # value as an equivalent float tokens_mask ([1,0] -> 12.8).
        l_aux_int, _, _ = self._call(
            gate_prob=self._t(self.GATE),
            dispatch_mask=self._t(self.DMASK),
            tokens_mask=self._t([1, 0], dtype="int32"),
            dispatch_tokens_mask=None,
            num_experts=self.NE,
            use_group=False,
            moe_k=2,
        )
        np.testing.assert_allclose(
            float(l_aux_int.numpy()), 12.8, rtol=1e-6, atol=1e-6
        )

    def test_dispatch_mask_1d_is_used_directly(self):
        # When dispatch_mask is already 1D [E], no sum(0) reduction happens.
        # Use [1,3,2,1] directly (matches sum0 of the 2D fixture) -> 3.2.
        dmask_1d = [1.0, 3.0, 2.0, 1.0]
        l_aux, _, _ = self._call(
            gate_prob=self._t(self.GATE),
            dispatch_mask=self._t(dmask_1d),
            tokens_mask=None,
            dispatch_tokens_mask=None,
            num_experts=self.NE,
            use_group=False,
            moe_k=2,
        )
        expected = _aux_loss_ref(
            self.GATE, dmask_1d, None, None, self.NE, False, 2
        )
        np.testing.assert_allclose(
            float(l_aux.numpy()), expected, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            float(l_aux.numpy()), 3.2, rtol=1e-6, atol=1e-6
        )

    def test_scale_factor_applied_to_value(self):
        # dispatch_tokens_mask present with shape[0] != gate_prob.shape[0] AND
        # tokens_mask present -> scale = seqlen(=12) / tokens_mask.sum(=2) = 6.
        # seqlen_float becomes 12 (from dispatch_tokens_mask.sum()):
        #   base l_aux = sum(([.6,.3,.6,.5]/12)*([1,3,2,1]/12))*4 = 12.8/144
        #   final = base * 6 = 76.8/144 = 0.533333...
        dtm = [[1.0, 1.0, 1.0, 1.0]] * 3  # shape [3,4] -> sum = 12
        l_aux, _, _ = self._call(
            gate_prob=self._t(self.GATE),
            dispatch_mask=self._t(self.DMASK),
            tokens_mask=self._t([1.0, 1.0]),
            dispatch_tokens_mask=self._t(dtm),
            num_experts=self.NE,
            use_group=False,
            moe_k=2,
        )
        expected = _aux_loss_ref(
            self.GATE, self.DMASK, [1.0, 1.0], dtm, self.NE, False, 2
        )
        np.testing.assert_allclose(
            float(l_aux.numpy()), expected, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            float(l_aux.numpy()), 76.8 / 144.0, rtol=1e-6, atol=1e-6
        )
        # Negative control: without scale (base value) it would be 12.8/144.
        self.assertFalse(
            np.isclose(float(l_aux.numpy()), 12.8 / 144.0, rtol=1e-4)
        )

    def test_scale_uses_detach_so_gradient_is_not_scaled(self):
        # The scale correction is `l_aux + (scale-1)*l_aux.detach()`: it scales
        # the VALUE (by 6) but the gradient w.r.t. gate_prob keeps coefficient
        # 1 (second term detached). d l_aux / d gate_prob[s,e] = E*ce[e]/seqlen
        # with seqlen = 12, ce = [1,3,2,1]/12 -> grad = 4*[1,3,2,1]/144.
        gate = self._t(self.GATE)
        gate.stop_gradient = False
        dtm = [[1.0, 1.0, 1.0, 1.0]] * 3
        l_aux, _, _ = self._call(
            gate_prob=gate,
            dispatch_mask=self._t(self.DMASK),
            tokens_mask=self._t([1.0, 1.0]),
            dispatch_tokens_mask=self._t(dtm),
            num_experts=self.NE,
            use_group=False,
            moe_k=2,
        )
        l_aux.backward()
        grad = gate.grad.numpy()

        row = np.array([4.0, 12.0, 8.0, 4.0]) / 144.0  # coefficient-1 grad
        expected_grad = np.stack([row, row])
        np.testing.assert_allclose(grad, expected_grad, rtol=1e-5, atol=1e-7)
        # Guard: a naive `scale * l_aux` (no detach) would give a 6x gradient.
        self.assertFalse(np.allclose(grad, 6.0 * expected_grad, rtol=1e-3))


class TestNpuCalAuxLossGlobalReduce(unittest.TestCase):
    """global_aux_loss path needs a real process group; not a 无卡 test."""

    @unittest.skip(
        "npu_cal_aux_loss_func(global_aux_loss=True) calls "
        "paddle.distributed.all_gather over a real EP/DP process group to "
        "average me/ce across ranks. That cross-rank reduction cannot be "
        "validated on CPU / single process (无卡); it requires a multi-card "
        "run with distinguishable per-rank inputs verifying the all_gather "
        "collect + mean(0). Local mock collectives would only prove "
        "orchestration, not the reduction numerics."
    )
    def test_global_aux_loss_all_gather_reduction(self):
        pass


if __name__ == "__main__":
    unittest.main()
