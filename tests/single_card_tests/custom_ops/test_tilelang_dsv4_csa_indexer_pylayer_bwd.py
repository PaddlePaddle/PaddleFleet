# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Backward correctness tests for the V4 CSA Indexer TileLang loss path
(``TileLangCSAIndexerLoss`` PyLayer) against the Paddle reference
``FusedDSAIndexerLoss``.

Both PyLayers consume the same ``(q, k, weights)`` produced by
``CSAIndexer.forward_before_topk`` and the same detached MLA
``(query, key)``, then emit a scalar KL loss whose backward yields
``dQ / dWeights / dKComp`` against the indexer inputs.

Scale alignment
---------------
The TileLang kernel applies a ``dim**-0.5`` scale on the indexer Q
before the GEMM, while the Paddle reference (``_compute_index_scores_fused``)
does not apply that scale. Because ``ReLU(alpha * x) == alpha * ReLU(x)``
for positive ``alpha``, scaling Q by ``alpha = dim**-0.5`` is
equivalent to scaling the per-head ``weights`` by the same ``alpha``.
We therefore feed ``weights * dim**-0.5`` to the Paddle reference and
``weights`` to TileLang to make the scalar losses identical. By the
chain rule, gradients then satisfy:

    dW_tilelang == dW_paddle * (dim**-0.5)
    dQ_tilelang == dQ_paddle
    dK_tilelang == dK_paddle

These tests require CUDA + the TileLang stack and skip otherwise.

Import-order requirement
------------------------
``paddle.enable_compat(scope={'tilelang'})`` must run BEFORE any module
that transitively does ``import tilelang`` (which in turn does
``import torch``). Otherwise tilelang's torch reference becomes the real
``torch`` module and the kernel allocates real Torch output buffers,
which the strict ``tilelang_csa_compressed_indexer_*_paddle`` wrappers
will reject.
"""

import os
import sys
import unittest

# Ensure the local PaddleFleet source is loaded instead of any stale install.
_LOCAL_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")
)
if _LOCAL_SRC not in sys.path:
    sys.path.insert(0, _LOCAL_SRC)
for _m in [
    m for m in list(sys.modules)
    if m == "paddlefleet" or m.startswith("paddlefleet.")
]:
    _mod = sys.modules.get(_m)
    _f = getattr(_mod, "__file__", "") or ""
    if _LOCAL_SRC not in _f:
        sys.modules.pop(_m, None)

# CRITICAL ORDER: paddle -> enable_compat({'tilelang'}) -> tilelang.
# Doing this here (at module top, before any tilelang/torch import that
# the kernels rely on) makes Paddle's torch proxy take over so the
# TileLang kernel allocates Paddle output buffers via ``torch.empty``.
import paddle
paddle.enable_compat(scope={"tilelang"}, silent=True)


def _build_csa_causal_mask(b, sq, sk, ratio):
    """Return ``[b, 1, sq, sk]`` -inf/0 mask matching the kernel's
    causal validity ``comp_id < (t + 1) // ratio``."""
    comp_ids = paddle.arange(sk, dtype="int64").reshape([1, 1, 1, sk])
    valid_end = (
        paddle.arange(1, sq + 1, dtype="int64").reshape([1, 1, sq, 1]) // ratio
    )
    valid = (comp_ids < valid_end).expand([b, 1, sq, sk])
    neg_inf = paddle.full([b, 1, sq, sk], float("-inf"), dtype="float32")
    return paddle.where(valid, paddle.zeros_like(neg_inf), neg_inf)


def _make_inputs(b, sq, sk, h_i, d_i, np_, hn, seed=2027):
    """Generate ``(q, k, weights, query_mla, key_comp_mla)`` matching the
    ``CSAIndexer.forward_before_topk`` and MLA contracts."""
    paddle.seed(seed)
    q = paddle.randn([b, sq, h_i, d_i]).astype("bfloat16")
    k = paddle.randn([b, sk, d_i]).astype("bfloat16")
    weights = paddle.randn([b, sq, h_i]).astype("float32")
    query_mla = paddle.randn([b, sq, np_, hn]).astype("bfloat16").detach()
    key_comp_mla = paddle.randn([b, sk, np_, hn]).astype("bfloat16").detach()
    return q, k, weights, query_mla, key_comp_mla


def _run_tilelang(q, k, weights, query_mla, key_comp_mla, ratio,
                  topk_eff, softmax_scale, loss_coeff):
    """Run TileLangCSAIndexerLoss forward+backward, return loss + grads."""
    from paddlefleet.transformer.csa_attention import TileLangCSAIndexerLoss

    qd = q.detach().clone()
    kd = k.detach().clone()
    wd = weights.detach().clone()
    qd.stop_gradient = False
    kd.stop_gradient = False
    wd.stop_gradient = False

    loss = TileLangCSAIndexerLoss.apply(
        qd, wd, kd,
        query_mla, key_comp_mla,
        int(ratio), int(topk_eff),
        float(softmax_scale), float(loss_coeff),
        None,
    )
    loss.backward()
    return loss, qd.grad, wd.grad, kd.grad


def _run_paddle(q, k, weights, query_mla, key_comp_mla, ratio,
                topk, softmax_scale, loss_coeff, sparse_loss):
    """Run FusedDSAIndexerLoss forward+backward (seq-first) with weights
    pre-scaled by ``dim**-0.5`` to align with the TileLang internal scale.

    Returns ``(loss, dQ_bf, dW_bf_scaled_back, dK_bf)`` in batch-first
    layout with dW already multiplied by ``dim**-0.5`` so it compares
    element-wise to the TileLang dW.
    """
    from paddlefleet.transformer.dsa_attention import FusedDSAIndexerLoss

    d_i = q.shape[-1]
    alpha = d_i ** -0.5

    q_sf = q.detach().transpose([1, 0, 2, 3]).clone()
    k_sf = k.detach().transpose([1, 0, 2]).clone()
    w_sf = (weights.detach() * alpha).transpose([1, 0, 2]).clone()

    q_sf.stop_gradient = False
    k_sf.stop_gradient = False
    w_sf.stop_gradient = False

    query_sf = query_mla.transpose([1, 0, 2, 3]).detach()
    key_sf = key_comp_mla.transpose([1, 0, 2, 3]).detach()

    sk = k.shape[1]
    mask = _build_csa_causal_mask(q.shape[0], q.shape[1], sk, ratio)

    loss = FusedDSAIndexerLoss.apply(
        q_sf, w_sf, k_sf,
        query_sf, key_sf,
        float(softmax_scale),
        int(topk),
        float(loss_coeff),
        mask,
        bool(sparse_loss),
        None,
    )
    loss.backward()

    dq_bf = q_sf.grad.transpose([1, 0, 2, 3])
    dk_bf = k_sf.grad.transpose([1, 0, 2])
    dw_bf = w_sf.grad.transpose([1, 0, 2]) * alpha
    return loss, dq_bf, dw_bf, dk_bf


def _assert_close(actual, expected, rtol, atol, msg):
    """All-Paddle ``assert_allclose``."""
    a = actual.cast("float32") if actual.dtype != paddle.float32 else actual
    e = expected.cast("float32") if expected.dtype != paddle.float32 else expected
    if not paddle.allclose(a, e, rtol=rtol, atol=atol).item():
        diff = (a - e).abs()
        raise AssertionError(
            f"{msg}\n  max abs diff: {diff.max().item():.4e}\n"
            f"  max rel diff: {(diff / e.abs().clip(min=1e-12)).max().item():.4e}"
        )


@unittest.skipUnless(
    paddle.device.is_compiled_with_cuda()
    and paddle.device.cuda.device_count() > 0,
    "TileLang CSA Indexer kernel requires CUDA",
)
class TestTileLangDSV4CSAIndexerLossGrad(unittest.TestCase):
    """Covers Tasks 9.1 / 9.2 / 9.3 / 9.4 / 9.5 / 9.6."""

    B, SQ, SK, H_I, D_I, NP, HN, RATIO = 1, 16, 4, 64, 128, 1, 128, 4
    LOSS_COEFF = 1.0

    def _common_inputs(self, seed=2027):
        return _make_inputs(
            self.B, self.SQ, self.SK,
            self.H_I, self.D_I,
            self.NP, self.HN,
            seed=seed,
        )

    def _assert_grads_close(self, tl_grads, pd_grads, label):
        tl_dq, tl_dw, tl_dk = tl_grads
        pd_dq, pd_dw, pd_dk = pd_grads
        _assert_close(tl_dq, pd_dq, rtol=8e-2, atol=3e-2, msg=f"{label}: dQ mismatch")
        _assert_close(tl_dw, pd_dw, rtol=8e-2, atol=3e-2, msg=f"{label}: dWeights mismatch")
        _assert_close(tl_dk, pd_dk, rtol=8e-2, atol=3e-2, msg=f"{label}: dKComp mismatch")

    # -- 9.1 / 9.3 ------------------------------------------------------

    def test_phase3_selected_topk_grad_matches_fused_dsa_indexer_loss(self):
        """Phase 3: ``topk_effective=dsa_indexer_topk`` vs sparse_loss=True."""
        q, k, w, qm, km = self._common_inputs()
        topk_eff = 2
        softmax_scale = self.HN ** -0.5

        tl_loss, tl_dq, tl_dw, tl_dk = _run_tilelang(
            q, k, w, qm, km, self.RATIO, topk_eff,
            softmax_scale, self.LOSS_COEFF,
        )
        pd_loss, pd_dq, pd_dw, pd_dk = _run_paddle(
            q, k, w, qm, km, self.RATIO, topk_eff,
            softmax_scale, self.LOSS_COEFF,
            sparse_loss=True,
        )

        _assert_close(tl_loss, pd_loss, rtol=5e-2, atol=2e-3,
                      msg="Phase3 selected-topk loss mismatch")
        self._assert_grads_close(
            (tl_dq, tl_dw, tl_dk),
            (pd_dq, pd_dw, pd_dk),
            "Phase3 selected-topk",
        )

    # -- 9.2 ------------------------------------------------------------

    def test_phase2_full_range_grad_matches_fused_dsa_indexer_loss(self):
        """Phase 2: ``topk_effective=n_compressed`` vs sparse_loss=False."""
        q, k, w, qm, km = self._common_inputs(seed=2028)
        topk_eff = self.SK
        softmax_scale = self.HN ** -0.5

        tl_loss, tl_dq, tl_dw, tl_dk = _run_tilelang(
            q, k, w, qm, km, self.RATIO, topk_eff,
            softmax_scale, self.LOSS_COEFF,
        )
        pd_loss, pd_dq, pd_dw, pd_dk = _run_paddle(
            q, k, w, qm, km, self.RATIO, topk_eff,
            softmax_scale, self.LOSS_COEFF,
            sparse_loss=False,
        )

        _assert_close(tl_loss, pd_loss, rtol=5e-2, atol=2e-3,
                      msg="Phase2 full-range loss mismatch")
        self._assert_grads_close(
            (tl_dq, tl_dw, tl_dk),
            (pd_dq, pd_dw, pd_dk),
            "Phase2 full-range",
        )

    # -- 9.4 ------------------------------------------------------------

    def test_invalid_padding_rows_have_zero_grad(self):
        """Query positions with no causally-valid compressed block must
        propagate zero gradient through the loss path.

        For ``ratio=4`` and ``sq=16`` rows ``t=0,1,2`` have ``valid_end=0``:
        every selected slot is ``-1``, the selected softmax is masked to
        zero, KL contribution is 0, so dQ/dW on those rows must be zero.
        """
        q, k, w, qm, km = self._common_inputs(seed=2029)
        topk_eff = 2
        softmax_scale = self.HN ** -0.5

        _loss, tl_dq, tl_dw, _tl_dk = _run_tilelang(
            q, k, w, qm, km, self.RATIO, topk_eff,
            softmax_scale, self.LOSS_COEFF,
        )

        dq_invalid = tl_dq[:, :3, :, :].cast("float32")
        dw_invalid = tl_dw[:, :3, :].cast("float32")
        _assert_close(
            dq_invalid, paddle.zeros_like(dq_invalid),
            rtol=0.0, atol=1e-5,
            msg="dQ on rows with valid_end=0 must be zero",
        )
        _assert_close(
            dw_invalid, paddle.zeros_like(dw_invalid),
            rtol=0.0, atol=1e-5,
            msg="dWeights on rows with valid_end=0 must be zero",
        )

    # -- 9.5 ------------------------------------------------------------

    def test_padded_topk_does_not_change_grad(self):
        """When ``topk_effective > n_compressed`` the kernel pads tail
        slots with ``-1`` indices and zero probabilities; padded slots
        must contribute nothing to softmax/KL/gradients."""
        q, k, w, qm, km = self._common_inputs(seed=2030)
        softmax_scale = self.HN ** -0.5

        _l_full, dq_full, dw_full, dk_full = _run_tilelang(
            q, k, w, qm, km, self.RATIO, self.SK,
            softmax_scale, self.LOSS_COEFF,
        )
        _l_pad, dq_pad, dw_pad, dk_pad = _run_tilelang(
            q, k, w, qm, km, self.RATIO, self.SK + 2,
            softmax_scale, self.LOSS_COEFF,
        )

        _assert_close(dq_full, dq_pad, rtol=1e-3, atol=1e-4,
                      msg="dQ must be invariant to topk padding past n_compressed")
        _assert_close(dw_full, dw_pad, rtol=1e-3, atol=1e-4,
                      msg="dWeights must be invariant to topk padding")
        _assert_close(dk_full, dk_pad, rtol=1e-3, atol=1e-4,
                      msg="dKComp must be invariant to topk padding")

    # -- 9.6 ------------------------------------------------------------

    def test_training_step_no_nan_and_loss_backprops(self):
        """End-to-end training-step sanity: loss is finite, gradients are
        finite, and the indexer loss flows non-zero gradient to all three
        inputs (``q, weights, k``)."""
        q, k, w, qm, km = self._common_inputs(seed=2031)
        topk_eff = 2
        softmax_scale = self.HN ** -0.5

        loss, dq, dw, dk = _run_tilelang(
            q, k, w, qm, km, self.RATIO, topk_eff,
            softmax_scale, self.LOSS_COEFF,
        )

        loss_fp = loss.cast("float32")
        self.assertTrue(paddle.isfinite(loss_fp).all().item(),
                        "indexer loss is not finite")
        self.assertGreater(loss_fp.item(), 0.0,
                           "indexer KL loss must be strictly positive")

        for name, g in (("dQ", dq), ("dWeights", dw), ("dKComp", dk)):
            gf = g.cast("float32")
            self.assertTrue(
                paddle.isfinite(gf).all().item(),
                f"{name} contains NaN/Inf",
            )
            self.assertGreater(
                gf.abs().max().item(), 0.0,
                f"{name} is identically zero — loss did not backprop",
            )


if __name__ == "__main__":
    unittest.main()
