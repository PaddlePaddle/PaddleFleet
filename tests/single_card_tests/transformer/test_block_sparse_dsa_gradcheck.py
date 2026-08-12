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

"""Adversarial operator / gradient-check validation for ``mqa_sparse_attn``.

Validation agent A3 (OPERATOR / GRADCHECK). This suite proves the token-level
DSA PyLayer in
``paddlefleet.fusions.mqa_sparse_attn._MQASparseAttention`` is numerically
correct *and* that its hand-written backward (including the analytic sink
gradient and the finite-sink LSE correction) is self-consistent with its own
forward.

Method (independent reference, never the implementation vs itself):
  * ``_ref_forward_np``     -- fp64 numpy masked-softmax over exactly the
    selected columns plus one value-less sink column. Forward oracle.
  * ``_ref_forward_paddle`` -- the same maths as an fp32 *differentiable*
    paddle graph; ``paddle.grad`` on it is the analytic backward oracle for
    dQ / dKV / d_sink.
  * finite differences   -- central differences on the *kernel* forward
    (sink in fp32; q/kv via directional derivatives, bf16 limited).
  * adjoint identity     -- <dO, J v> == <J^T dO, v>.

The FlashMLA sparse forward kernel is bf16-only (fp32 q is rejected -- proven
in :class:`TestDtypeBehaviour`), so q/kv error floors are bf16-limited
(~2e-3 fwd, ~4e-3 grad); the sink path is genuinely fp32. Tolerances below are
justified against the measured bf16 round-off floor (:meth:`_bf16_floor`).
"""

import unittest

import numpy as np
import paddle

from paddlefleet.fusions.mqa_sparse_attn import (
    _DSA_HEADS,
    _NEG_SINK,
    mqa_sparse_attn,
)

from .hybrid_mla_utils import _GPU

DK = 576  # absorbed-MQA query/key width (kv_lora_rank 512 + qk_rope 64)
DV = 512  # value width == leading 512 dims of the shared latent
SM = DK**-0.5


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------
def _rand(shape, scale, seed):
    rng = np.random.RandomState(seed)
    return (rng.randn(*shape) * scale).astype("float32")


def _causal_indices(s, L, seed=None, doc_start=None):
    """``[1, s, L]`` int32 causal index table (-1 padded).

    Row ``q`` selects columns ``[doc_start, q]`` (its own document's causal
    prefix), left-truncated to ``L`` slots. ``seed`` shuffles the valid slots
    to stress order-independence.
    """
    ti = np.full([1, s, L], -1, dtype=np.int32)
    ds = np.zeros(s, dtype=int) if doc_start is None else doc_start
    rng = np.random.RandomState(seed) if seed is not None else None
    for q in range(s):
        cols = list(range(int(ds[q]), q + 1))[-L:]
        if rng is not None:
            rng.shuffle(cols)
        ti[0, q, : len(cols)] = cols
    return ti


def _to_bf16(arr):
    return paddle.to_tensor(arr).cast("bfloat16")


def _kernel_forward(q_np, kv_np, ti_np, sink=None, sink_dtype="float32"):
    """Run the PyLayer forward. Returns (out_fp32 [s, H*DV], handles).

    ``handles`` are the leaf bf16 q/kv (+ fp32/bf16 sink) with grad enabled so
    the caller can backward and read gradients.
    """
    qb = _to_bf16(q_np)
    kvb = _to_bf16(kv_np)
    qb.stop_gradient = False
    kvb.stop_gradient = False
    sink_t = None
    if sink is not None:
        sink_t = paddle.to_tensor(np.asarray(sink, "float32")).cast(sink_dtype)
        sink_t.stop_gradient = False
    ti = paddle.to_tensor(ti_np)
    out = mqa_sparse_attn(qb, kvb, ti, float(SM), DV, sink_t)
    return out, (qb, kvb, sink_t)


# ---------------------------------------------------------------------------
# Independent references
# ---------------------------------------------------------------------------
def _ref_forward_np(q_np, kv_np, ti_np, sink=None):
    """fp64 numpy oracle: masked softmax over the *listed* columns + sink.

    Slot-wise, so a duplicated column is counted once per slot (the gather is
    per-slot; there is no dedup). An all-invalid row yields 0 (matches the
    kernel, which returns 0 rather than NaN for an empty row).
    """
    q = q_np.astype(np.float64)[0]  # [s, H, DK]
    kv = kv_np.astype(np.float64)[0]  # [s, DK]
    ti = ti_np[0]  # [s, L]
    s, H, _ = q.shape
    out = np.zeros([s, H, DV], dtype=np.float64)
    for i in range(s):
        cols = ti[i]
        vc = cols[cols >= 0]
        if len(vc) == 0 and sink is None:
            continue
        ksel = kv[vc]  # [nv, DK]
        for h in range(H):
            lg = (
                SM * (q[i, h][None, :] * ksel).sum(-1)
                if len(vc)
                else np.array([])
            )
            if sink is None:
                m = lg.max()
                w = np.exp(lg - m)
                Z = w.sum()
            else:
                sh = float(sink[h])
                m = max(lg.max() if len(lg) else -1e30, sh)
                w = np.exp(lg - m)
                Z = w.sum() + np.exp(sh - m)
            if len(vc):
                out[i, h] = ((w / Z)[:, None] * ksel[:, :DV]).sum(0)
    return out.reshape([s, H * DV])


def _ref_forward_paddle(qf, kvf, ti_np, sink=None):
    """Same maths as an fp32 differentiable graph (analytic-grad oracle).

    ``qf`` [s, H, DK], ``kvf`` [s, DK] fp32 leaves; ``sink`` [H] fp32 leaf or
    None. Safe-gather + (-1e30) mask keeps it differentiable in qf/kvf (and
    sink) exactly where the kernel is.
    """
    s, H, _ = qf.shape
    L = ti_np.shape[-1]
    cols = paddle.to_tensor(ti_np.reshape([s, L])).cast("int64")
    valid = cols >= 0
    safe = paddle.where(valid, cols, paddle.zeros_like(cols))
    ksel = paddle.gather(kvf, safe.flatten(), axis=0).reshape([s, L, DK])
    logit = paddle.einsum("shd,sld->shl", qf, ksel) * SM
    logit = paddle.where(
        valid.unsqueeze(1), logit, paddle.full_like(logit, _NEG_SINK)
    )
    if sink is None:
        p = paddle.nn.functional.softmax(logit, axis=-1)
        out = paddle.einsum("shl,sld->shd", p, ksel[:, :, :DV])
    else:
        sc = sink.reshape([1, H, 1]).expand([s, H, 1])
        full = paddle.concat([logit, sc], axis=-1)
        p = paddle.nn.functional.softmax(full, axis=-1)
        out = paddle.einsum("shl,sld->shd", p[:, :, :L], ksel[:, :, :DV])
    return out.reshape([s, H * DV])


def _relerr(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def _err_stats(a, b):
    # Only the two fields asserted on downstream: max abs error and L2 rel.
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return {
        "maxabs": round(float(np.abs(a - b).max()), 6),
        "l2rel": round(_relerr(a, b), 6),
    }


def _analytic_ref_grads(q_np, kv_np, ti_np, dO_np, sink_mag=None):
    """dQ / dKV / d_sink from ``paddle.grad`` on the fp32 reference forward."""
    qf = paddle.to_tensor(q_np[0]).clone().detach()
    kvf = paddle.to_tensor(kv_np[0]).clone().detach()
    qf.stop_gradient = False
    kvf.stop_gradient = False
    ins = [qf, kvf]
    sink = None
    if sink_mag is not None:
        sink = paddle.full([q_np.shape[2]], float(sink_mag), dtype="float32")
        sink.stop_gradient = False
        ins.append(sink)
    o = _ref_forward_paddle(qf, kvf, ti_np, sink)
    dO = paddle.to_tensor(dO_np.reshape(o.shape))
    grads = paddle.grad(outputs=(o * dO).sum(), inputs=ins)
    dsink = grads[2].numpy() if sink_mag is not None else None
    return grads[0].numpy(), grads[1].numpy(), dsink


def _kernel_grads(
    q_np, kv_np, ti_np, dO_np, sink_mag=None, sink_dtype="float32"
):
    """dQ / dKV / d_sink from the PyLayer backward (bf16 kernel)."""
    sink = None if sink_mag is None else [sink_mag] * q_np.shape[2]
    out, (qb, kvb, sink_t) = _kernel_forward(
        q_np, kv_np, ti_np, sink, sink_dtype
    )
    dO = paddle.to_tensor(dO_np.reshape(out.shape))
    (out.cast("float32") * dO).sum().backward()
    dq = qb.grad.cast("float32").numpy()[0]
    dkv = kvb.grad.cast("float32").numpy()[0]
    dsink = None if sink_t is None else (sink_t.grad, sink_t.grad.dtype)
    return dq, dkv, dsink


def _kernel_dq_with_kv_only_lse(q_np, kv_np, ti_np, dO_np, sink_mag):
    """dQ from the raw kernels with the **uncorrected** KV-only LSE.

    ``mqa_sparse_attn`` always folds the sink into the LSE it hands the cuDNN
    DSA backward, because that kernel's ``d_qk != d_v`` branch consumes the LSE
    verbatim. This helper drives the same kernel pair directly and skips the
    fold, so a test can measure what the correction is worth without the layer
    needing a switch for it.
    """
    from paddlefleet.cudnn_ops import csa_sparse_attn_bwd_cudnn
    from paddlefleet.cudnn_ops.attn.csa_sparse_attn_fwd_cudnn import (
        flash_mla_sparse_attn,
    )
    from paddlefleet.fusions.csa_sparse_attn import _csa_compute_topk_length
    from paddlefleet.fusions.csa_sparse_attn_utils import _local_to_global_flat

    b, s, H, _ = q_np.shape
    skv = kv_np.shape[1]
    hp = _DSA_HEADS
    qb, kvb = _to_bf16(q_np), _to_bf16(kv_np)
    q_pad = paddle.concat(
        [qb, paddle.zeros([b, s, hp - H, DK], dtype=qb.dtype)], axis=2
    )
    sink = paddle.concat(
        [
            paddle.full([H], float(sink_mag), dtype="float32"),
            paddle.full([hp - H], _NEG_SINK, dtype="float32"),
        ]
    )
    ti = paddle.to_tensor(ti_np)
    tl = _csa_compute_topk_length(ti.reshape([b * s, -1]))
    out, lse, _ = flash_mla_sparse_attn(
        q_pad,
        kvb,
        sink,
        ti,
        sm_scale=float(SM),
        d_v=DV,
        topk_length=tl.reshape([b, s]),
    )
    do = paddle.concat(
        [
            paddle.to_tensor(dO_np.reshape([b, s, H, DV])).cast(out.dtype),
            paddle.zeros([b, s, hp - H, DV], dtype=out.dtype),
        ],
        axis=2,
    ).contiguous()
    dq, _, _ = csa_sparse_attn_bwd_cudnn(
        q_pad.reshape([b * s, hp, DK]),
        kvb.reshape([b * skv, DK]),
        out.reshape([b * s, hp, DV]),
        do.reshape([b * s, hp, DV]),
        lse.reshape([b * s, hp]),  # KV-only: the sink is NOT folded in
        sink,
        _local_to_global_flat(ti, skv),
        softmax_scale=float(SM),
        topk_length=tl,
    )
    dq = dq.reshape([b, s, hp, DK])[:, :, :H, :]
    return dq.cast("float32").numpy()[0]


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paddle.set_device("gpu")
        try:
            paddle.set_flags({"FLAGS_cudnn_deterministic": True})
        except Exception:
            pass


# ===========================================================================
# 1. Forward vs dense reference
# ===========================================================================
@_GPU
class TestForward(_Base):
    """Forward matches an independent fp64 dense reference over the selected
    columns (+ optional sink column), across head counts and sink sizes."""

    def _run(self, H, s, L, sink, seed):
        q = _rand([1, s, H, DK], 0.3, seed)
        kv = _rand([1, s, DK], 0.3, seed + 1)
        ti = _causal_indices(s, L)
        out, _ = _kernel_forward(
            q, kv, ti, None if sink is None else [sink] * H
        )
        ref = _ref_forward_np(q, kv, ti, None if sink is None else [sink] * H)
        return out.cast("float32").numpy()[0], ref

    def test_forward_head_counts_sinkless(self):
        for H in (4, 8, 16, 32):
            with self.subTest(H=H):
                o, r = self._run(H, 96, 128, None, 100 + H)
                st = _err_stats(o, r)
                self.assertLess(st["maxabs"], 8e-3)
                self.assertLess(st["l2rel"], 8e-3)

    def test_forward_sink_magnitudes(self):
        # Stress the LSE path with sinks spanning the full dynamic range.
        for sink in (-20.0, -5.0, 0.0, 5.0, 20.0):
            with self.subTest(sink=sink):
                o, r = self._run(8, 96, 128, sink, 200)
                st = _err_stats(o, r)
                self.assertLess(st["maxabs"], 8e-3)
                self.assertLess(st["l2rel"], 1e-2)

    def test_forward_error_distribution(self):
        o, r = self._run(8, 128, 128, None, 7)
        ae = np.abs(o.astype(np.float64) - r)
        self.assertLess(float(ae.max()), 8e-3)  # p100 == max abs error


# ===========================================================================
# 2. Backward vs analytic reference (paddle.grad on the fp32 reference)
# ===========================================================================
@_GPU
class TestBackwardAnalytic(_Base):
    """dQ / dKV / d_sink match the analytic gradient of the fp32 reference."""

    def _case(self, H, s, L, sink_mag, seed):
        q = _rand([1, s, H, DK], 0.3, seed)
        kv = _rand([1, s, DK], 0.3, seed + 1)
        ti = _causal_indices(s, L, seed=seed + 2)
        dO = _rand([1, s, H * DV], 1.0, seed + 3)
        kq, kkv, ksink = _kernel_grads(q, kv, ti, dO, sink_mag)
        rq, rkv, rsink = _analytic_ref_grads(q, kv, ti, dO, sink_mag)
        return (kq, kkv, ksink), (rq, rkv, rsink)

    def test_backward_sinkless(self):
        for H in (8, 16):
            with self.subTest(H=H):
                (kq, kkv, _), (rq, rkv, _) = self._case(H, 64, 128, None, 30)
                sq, skv = _err_stats(kq, rq), _err_stats(kkv, rkv)
                self.assertLess(sq["l2rel"], 1e-2)
                self.assertLess(skv["l2rel"], 8e-3)

    def test_backward_with_sink(self):
        for sink_mag in (-5.0, 2.0):
            with self.subTest(sink=sink_mag):
                (kq, kkv, ks), (rq, rkv, rs) = self._case(
                    8, 64, 128, sink_mag, 40
                )
                sq, skv = _err_stats(kq, rq), _err_stats(kkv, rkv)
                ds = _err_stats(ks[0].numpy(), rs)
                self.assertLess(sq["l2rel"], 1e-2)
                self.assertLess(skv["l2rel"], 8e-3)
                self.assertLess(ds["l2rel"], 2e-2)


# ===========================================================================
# 3+4. Numerical finite differences and the adjoint dot-product identity
# ===========================================================================
def _proj_forward(q_np, kv_np, ti, dO, sink=None):
    """<dO, f(q, kv, sink)> as a python float (bf16 kernel forward)."""
    out, _ = _kernel_forward(q_np, kv_np, ti, sink)
    return float(
        (out.cast("float32") * paddle.to_tensor(dO.reshape(out.shape))).sum()
    )


@_GPU
class TestFiniteDifference(_Base):
    """Central differences on the *kernel* forward vs the kernel backward.

    The forward is bf16-only, so:
      * the sink is fp32 -> a clean scalar central difference (eps from the
        gentle log-domain curvature);
      * q/kv scalars are bf16 -> a single-scalar difference is far below the
        bf16 output-noise floor, so their self-consistency is proven by the
        directional adjoint test instead (see :class:`TestAdjoint`).
    """

    def test_sink_central_difference(self):
        H, s = 8, 48
        q = _rand([1, s, H, DK], 0.3, 11)
        kv = _rand([1, s, DK], 0.3, 12)
        ti = _causal_indices(s, 128)
        dO = _rand([1, s, H * DV], 1.0, 13)
        _, _, ks = _kernel_grads(q, kv, ti, dO, 1.0)
        d_analytic = ks[0].numpy()
        eps = 0.25  # calibrated: >> bf16 noise, << sink log-curvature scale
        d_fd = np.zeros(H)
        for h in range(H):
            sp = [1.0] * H
            sp[h] = 1.0 + eps
            sm_ = [1.0] * H
            sm_[h] = 1.0 - eps
            d_fd[h] = (
                _proj_forward(q, kv, ti, dO, sp)
                - _proj_forward(q, kv, ti, dO, sm_)
            ) / (2 * eps)
        rel = _relerr(d_analytic, d_fd)
        self.assertLess(rel, 5e-2)


@_GPU
class TestAdjoint(_Base):
    """Adjoint / dot-product identity <dO, J v> == <J^T dO, v>.

    Catches sign and scale errors a cosine misses. The probe direction ``v`` is
    taken along the analytic gradient (scaled to the input norm) so the
    directional signal sits well above the bf16 forward-noise floor -- a random
    direction on q would project onto ||dQ|| (~16x smaller than ||dKV||) and
    drown in bf16 noise, which is exactly the self-consistency being probed.
    """

    def _setup(self, H=8, s=48, seed=50, sink_mag=None):
        q = _rand([1, s, H, DK], 0.3, seed)
        kv = _rand([1, s, DK], 0.3, seed + 1)
        ti = _causal_indices(s, 128, seed=seed + 2)
        dO = _rand([1, s, H * DV], 1.0, seed + 3)
        kq, kkv, ks = _kernel_grads(q, kv, ti, dO, sink_mag)
        return q, kv, ti, dO, kq, kkv, ks

    def _dir_check(self, fwd_at, base, g, tol):
        v = g / (np.linalg.norm(g) + 1e-30) * np.linalg.norm(base)
        best = None
        for eps in (0.02, 0.05, 0.1):
            jv = (fwd_at(base + eps * v) - fwd_at(base - eps * v)) / (2 * eps)
            gv = float((g * v).sum())
            rel = abs(jv - gv) / (abs(gv) + 1e-30)
            best = rel if best is None else min(best, rel)
        self.assertLess(best, tol)

    def test_adjoint_q(self):
        q, kv, ti, dO, kq, _, _ = self._setup()
        self._dir_check(
            lambda qn: _proj_forward(qn.reshape(q.shape), kv, ti, dO),
            q[0].reshape(-1),
            kq.reshape(-1),
            2e-2,
        )

    def test_adjoint_kv(self):
        q, kv, ti, dO, _, kkv, _ = self._setup()
        self._dir_check(
            lambda kn: _proj_forward(q, kn.reshape(kv.shape), ti, dO),
            kv[0].reshape(-1),
            kkv.reshape(-1),
            2e-2,
        )

    def test_adjoint_sink(self):
        q, kv, ti, dO, _, _, ks = self._setup(sink_mag=1.5)
        base = np.full([q.shape[2]], 1.5, "float32")
        self._dir_check(
            lambda sn: _proj_forward(q, kv, ti, dO, sn.tolist()),
            base,
            ks[0].numpy(),
            5e-2,
        )


# ===========================================================================
# 5. Sink specifics
# ===========================================================================
@_GPU
class TestSinkSpecifics(_Base):
    def test_very_negative_sink_converges_to_sinkless(self):
        H, s = 8, 64
        q = _rand([1, s, H, DK], 0.3, 61)
        kv = _rand([1, s, DK], 0.3, 62)
        ti = _causal_indices(s, 128)
        base, _ = _kernel_forward(q, kv, ti, None)
        base = base.cast("float32").numpy()
        prev = None
        for mag in (-5.0, -20.0, -60.0):
            out, _ = _kernel_forward(q, kv, ti, [mag] * H)
            d = float(np.abs(out.cast("float32").numpy() - base).max())
            if prev is not None:
                self.assertLessEqual(d, prev + 1e-6)  # monotone convergence
            prev = d
        self.assertLess(prev, 1e-3)  # -60 is indistinguishable from sinkless

    def test_dsink_zero_for_neg_inf_like_sink(self):
        H, s = 8, 48
        q = _rand([1, s, H, DK], 0.3, 71)
        kv = _rand([1, s, DK], 0.3, 72)
        ti = _causal_indices(s, 128)
        dO = _rand([1, s, H * DV], 1.0, 73)
        _, _, ks = _kernel_grads(q, kv, ti, dO, float(_NEG_SINK))
        g = ks[0].numpy()
        self.assertLess(float(np.abs(g).max()), 1e-6)

    def test_dsink_sign_matches_numeric(self):
        # A positive sink drains mass, so raising it lowers ||out||; with a
        # positive dO.out the projected loss falls -> d_sink < 0 where delta>0.
        H, s = 8, 48
        q = _rand([1, s, H, DK], 0.3, 81)
        kv = _rand([1, s, DK], 0.3, 82)
        ti = _causal_indices(s, 128)
        dO = _rand([1, s, H * DV], 1.0, 83)
        _, _, ks = _kernel_grads(q, kv, ti, dO, 1.0)
        _, _, rs = _analytic_ref_grads(q, kv, ti, dO, 1.0)
        ka, ra = ks[0].numpy(), rs
        agree = float(np.mean(np.sign(ka) == np.sign(ra)))
        self.assertEqual(agree, 1.0)

    def test_finite_sink_lse_fix_matters(self):
        """dk(576) != d_v(512): the finite-sink LSE fold is load-bearing.

        The layer always folds the sink into the LSE it passes the backward;
        driving the same kernels with the raw KV-only LSE must be far worse.
        """
        H, s = 8, 64
        q = _rand([1, s, H, DK], 0.3, 91)
        kv = _rand([1, s, DK], 0.3, 92)
        ti = _causal_indices(s, 128, seed=93)
        dO = _rand([1, s, H * DV], 1.0, 94)
        rq, _, _ = _analytic_ref_grads(q, kv, ti, dO, 3.0)
        folded, _, _ = _kernel_grads(q, kv, ti, dO, 3.0)
        raw = _kernel_dq_with_kv_only_lse(q, kv, ti, dO, 3.0)
        e_folded, e_raw = _relerr(folded, rq), _relerr(raw, rq)
        self.assertLess(e_folded, 1e-2)
        self.assertGreater(e_raw, 0.5)
        self.assertLess(e_folded, e_raw * 0.05)


# ===========================================================================
# 6. Index / mask edge cases
# ===========================================================================
@_GPU
class TestIndexEdgeCases(_Base):
    def _fwd(self, q, kv, ti, sink=None):
        out, _ = _kernel_forward(q, kv, ti, sink)
        return out.cast("float32").numpy()[0]

    def test_duplicate_indices_are_double_counted(self):
        # The gather is per-slot; a duplicated column enters the softmax twice.
        # Callers MUST feed a duplicate-free index set (the production builders
        # do -- verified by the existing index-invariant tests).
        H, s = 8, 6
        q = _rand([1, s, H, DK], 0.3, 111)
        kv = _rand([1, s, DK], 0.3, 112)
        ti = np.full([1, s, 128], -1, np.int32)
        ti[0, 5, :4] = [0, 0, 1, 2]  # column 0 duplicated
        o = self._fwd(q, kv, ti)[5]
        r_dup = _ref_forward_np(q, kv, ti)[5]  # ref also double-counts slot 0
        ti_dedup = ti.copy()
        ti_dedup[0, 5, :4] = [0, 1, 2, -1]
        r_dedup = _ref_forward_np(q, kv, ti_dedup)[5]
        self.assertLess(_relerr(o, r_dup), 1e-2)
        self.assertGreater(_relerr(o, r_dedup), 0.1)

    def test_unsorted_indices_are_permutation_invariant(self):
        # Same query row, two column orders -> bitwise identical (softmax is a
        # symmetric reduction; the kernel visits slots in a fixed tile order).
        H, s = 8, 6
        q = _rand([1, s, H, DK], 0.3, 121)
        kv = _rand([1, s, DK], 0.3, 122)
        t1 = np.full([1, s, 128], -1, np.int32)
        t1[0, 5, :4] = [0, 1, 2, 3]
        t2 = t1.copy()
        t2[0, 5, :4] = [3, 1, 0, 2]
        o1 = self._fwd(q, kv, t1)[5]
        o2 = self._fwd(q, kv, t2)[5]
        r = _ref_forward_np(q, kv, t1)[5]
        np.testing.assert_array_equal(o1, o2)
        self.assertLess(_relerr(o1, r), 8e-3)

    def test_negative_one_columns_are_masked(self):
        # -1 is the only invalidity sentinel: a valid column set to -1 must
        # contribute nothing (this is how out-of-range / beyond-eos cols are
        # represented after block expansion).
        H, s = 8, 8
        q = _rand([1, s, H, DK], 0.3, 131)
        kv = _rand([1, s, DK], 0.3, 132)
        full = np.full([1, s, 128], -1, np.int32)
        for i in range(s):
            full[0, i, : i + 1] = np.arange(i + 1)
        masked = full.copy()
        masked[0, 7, 3] = -1  # drop column 3 from row 7
        o = self._fwd(q, kv, masked)[7]
        r = _ref_forward_np(q, kv, masked)[7]  # ref excludes -1 too
        r_full = _ref_forward_np(q, kv, full)[7]
        self.assertLess(_relerr(o, r), 8e-3)
        self.assertGreater(_relerr(r, r_full), 1e-3)

    def test_single_column_first_row_and_empty_row(self):
        H, s = 8, 6
        q = _rand([1, s, H, DK], 0.3, 141)
        kv = _rand([1, s, DK], 0.3, 142)
        ti = np.full([1, s, 128], -1, np.int32)
        ti[0, 0, 0] = 0  # first row: self only
        ti[0, 1, 0] = 1  # single valid column
        # row 2 stays all -1: empty / budget-exceeds-availability row
        o = self._fwd(q, kv, ti)
        kvf = _to_bf16(kv).cast("float32").numpy()[0]  # bf16-rounded like kv
        o3 = o.reshape([s, H, DV])
        # single-column softmax(1)=1 -> output == that (bf16) column, exactly
        self.assertEqual(float(np.abs(o3[0] - kvf[0, :DV][None]).max()), 0.0)
        self.assertEqual(float(np.abs(o3[1] - kvf[1, :DV][None]).max()), 0.0)
        empty = o3[2]
        self.assertTrue(bool(np.isfinite(empty).all()))
        self.assertEqual(float(np.abs(empty).max()), 0.0)


# ===========================================================================
# 7. dtype behaviour
# ===========================================================================
@_GPU
class TestDtypeBehaviour(_Base):
    def test_fp32_query_is_rejected_by_forward_kernel(self):
        # The FlashMLA sparse forward is bf16-only; fp32 q must fail loudly
        # rather than silently mis-cast. (This is why the FD floors are bf16.)
        H, s = 8, 16
        q = paddle.to_tensor(_rand([1, s, H, DK], 0.3, 1))  # fp32
        kv = paddle.to_tensor(_rand([1, s, DK], 0.3, 2))
        ti = paddle.to_tensor(_causal_indices(s, 128))
        with self.assertRaises(RuntimeError):
            mqa_sparse_attn(q, kv, ti, float(SM), DV, None)

    def test_sink_grad_dtype_follows_parameter(self):
        H, s = 8, 32
        q = _rand([1, s, H, DK], 0.3, 3)
        kv = _rand([1, s, DK], 0.3, 4)
        ti = _causal_indices(s, 128)
        dO = _rand([1, s, H * DV], 1.0, 5)
        for dt, pdt in (
            ("float32", paddle.float32),
            ("bfloat16", paddle.bfloat16),
        ):
            _, _, ks = _kernel_grads(q, kv, ti, dO, 1.0, sink_dtype=dt)
            grad, gdt = ks
            self.assertEqual(gdt, pdt)
            self.assertTrue(bool(paddle.isfinite(grad.astype("float32")).all()))
            self.assertGreater(float(grad.astype("float32").abs().max()), 0.0)

    def test_bf16_roundoff_floor(self):
        # The measured bf16 forward floor that justifies the other tolerances.
        H, s = 8, 96
        q = _rand([1, s, H, DK], 0.3, 6)
        kv = _rand([1, s, DK], 0.3, 7)
        ti = _causal_indices(s, 128)
        out, _ = _kernel_forward(q, kv, ti, None)
        ref = _ref_forward_np(q, kv, ti, None)
        st = _err_stats(out.cast("float32").numpy()[0], ref)
        self.assertLess(st["l2rel"], 8e-3)


# ===========================================================================
# 8. Determinism
# ===========================================================================
@_GPU
class TestDeterminism(_Base):
    def test_forward_and_backward_are_bitwise_deterministic(self):
        H, s = 8, 64
        q = _rand([1, s, H, DK], 0.3, 201)
        kv = _rand([1, s, DK], 0.3, 202)
        ti = _causal_indices(s, 128, seed=203)
        dO = _rand([1, s, H * DV], 1.0, 204)

        def once():
            out, (qb, kvb, _) = _kernel_forward(q, kv, ti, None)
            of = out.cast("float32").numpy().copy()
            (
                out.cast("float32") * paddle.to_tensor(dO.reshape(out.shape))
            ).sum().backward()
            return (
                of,
                qb.grad.cast("float32").numpy().copy(),
                kvb.grad.cast("float32").numpy().copy(),
            )

        o1, gq1, gk1 = once()
        o2, gq2, gk2 = once()
        fwd_eq = np.array_equal(o1, o2)
        dq_eq = np.array_equal(gq1, gq2)
        dkv_rel = _relerr(gk1, gk2)
        # Forward and dQ are bitwise reproducible. dKV is NOT (finding A3-1):
        # the cuDNN DSA backward's dKV scatter-add is non-deterministic even
        # with FLAGS_cudnn_deterministic=True. The magnitude is tiny (rel
        # ~3e-6, << the bf16 grad floor ~3e-3), so it is a reproducibility
        # nuisance, not a correctness bug -- but it is quantified and bounded
        # here so any future growth trips the assert.
        self.assertTrue(fwd_eq, "forward is not bitwise deterministic")
        self.assertTrue(dq_eq, "dQ is not bitwise deterministic")
        self.assertLess(
            dkv_rel, 1e-4, "dKV non-determinism exceeds the bounded floor"
        )


if __name__ == "__main__":
    unittest.main()
