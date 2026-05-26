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

import os
import sys
import unittest

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

import paddle
import paddle.nn.functional as F

paddle.enable_compat(scope={"tilelang"}, silent=True)


# =========================================================================
# Helpers
# =========================================================================

def _cuda_or_skip(testcase):
    if not paddle.device.is_compiled_with_cuda():
        testcase.skipTest("CUDA build of Paddle is required")
    if paddle.device.cuda.device_count() == 0:
        testcase.skipTest("No CUDA device available")


def _make_indexer_inputs(b, sq, sk, h_i, d_i, dtype="bfloat16", seed=2026):
    paddle.seed(seed)
    q = paddle.randn([b, sq, h_i, d_i]).astype(dtype)
    k = paddle.randn([b, sk, d_i]).astype(dtype)
    w = paddle.randn([b, sq, h_i]).astype("float32")
    return q, k, w


def _make_loss_inputs(b, sq, sk, h_i, d_i, np_, hn, seed=2027):
    paddle.seed(seed)
    q = paddle.randn([b, sq, h_i, d_i]).astype("bfloat16")
    k = paddle.randn([b, sk, d_i]).astype("bfloat16")
    weights = paddle.randn([b, sq, h_i]).astype("float32")
    query_mla = paddle.randn([b, sq, np_, hn]).astype("bfloat16").detach()
    key_comp_mla = paddle.randn([b, sk, np_, hn]).astype("bfloat16").detach()
    return q, k, weights, query_mla, key_comp_mla


def _build_csa_causal_mask(b, sq, sk, ratio):
    comp_ids = paddle.arange(sk, dtype="int64").reshape([1, 1, 1, sk])
    valid_end = (
        paddle.arange(1, sq + 1, dtype="int64").reshape([1, 1, sq, 1]) // ratio
    )
    valid = (comp_ids < valid_end).expand([b, 1, sq, sk])
    neg_inf = paddle.full([b, 1, sq, sk], float("-inf"), dtype="float32")
    return paddle.where(valid, paddle.zeros_like(neg_inf), neg_inf)


def _all_equal(tensor, value):
    return bool((tensor == value).all().item())


def _sorted_compare_indices(out_indices, ref_indices):
    out_sorted = paddle.sort(out_indices, axis=-1)
    ref_sorted = paddle.sort(ref_indices, axis=-1)
    return bool((out_sorted == ref_sorted).all().item())


def _assert_close(actual, expected, rtol, atol, msg):
    a = actual.cast("float32") if actual.dtype != paddle.float32 else actual
    e = expected.cast("float32") if expected.dtype != paddle.float32 else expected
    if not paddle.allclose(a, e, rtol=rtol, atol=atol).item():
        diff = (a - e).abs()
        raise AssertionError(
            f"{msg}\n  max abs diff: {diff.max().item():.4e}\n"
            f"  max rel diff: {(diff / e.abs().clip(min=1e-12)).max().item():.4e}"
        )


# =========================================================================
# Reference implementations
# =========================================================================

def _ref_csa_indexer_topk(index_q, index_k_comp, weights, ratio, topk_effective):
    scores = paddle.einsum("bshd,btd->bsht", index_q.cast("float32"), index_k_comp.cast("float32"))
    scores = F.relu(scores)
    scores = (scores * weights.cast("float32").unsqueeze(-1)).sum(axis=2)
    scores = scores * (index_q.shape[-1] ** -0.5)
    batch, seq_len, seq_len_comp = scores.shape
    comp_ids = paddle.arange(seq_len_comp, dtype="int64").reshape([1, 1, seq_len_comp])
    positions = paddle.arange(1, seq_len + 1, dtype="int64").reshape([1, seq_len, 1])
    valid_end = positions // ratio
    valid_mask = comp_ids < valid_end
    scores = paddle.where(valid_mask, scores, paddle.full_like(scores, float("-inf")))
    actual_topk = min(topk_effective, seq_len_comp)
    topk_scores_raw, topk_indices = paddle.topk(scores, k=actual_topk, axis=-1)
    valid_topk = paddle.take_along_axis(
        paddle.expand(valid_mask, [batch, seq_len, seq_len_comp]).cast("int32"),
        topk_indices, axis=-1,
    ).cast("bool")
    topk_indices = paddle.where(valid_topk, topk_indices, paddle.full_like(topk_indices, -1))
    topk_scores_raw = paddle.where(valid_topk, topk_scores_raw, paddle.full_like(topk_scores_raw, float("-inf")))
    topk_probs = F.softmax(topk_scores_raw, axis=-1)
    topk_probs = paddle.where(valid_topk, topk_probs, paddle.zeros_like(topk_probs))
    if topk_effective > actual_topk:
        pad = topk_effective - actual_topk
        topk_indices = paddle.concat(
            [topk_indices, paddle.full([batch, seq_len, pad], -1, dtype=topk_indices.dtype)], axis=-1,
        )
        topk_probs = paddle.concat(
            [topk_probs, paddle.zeros([batch, seq_len, pad], dtype=topk_probs.dtype)], axis=-1,
        )
    return topk_indices.cast("int32"), topk_probs.cast("float32")


def _paddle_ref_csa_indexer_topk(q, k, weights, ratio, topk_effective):
    from paddlefleet.transformer.dsa_attention import fused_qk_topk_naive

    b, sq, h_i, d_i = q.shape
    sk = k.shape[1]
    sm_scale = d_i ** -0.5
    comp_ids = paddle.arange(sk, dtype="int64").reshape([1, 1, sk])
    valid_end = (paddle.arange(1, sq + 1, dtype="int64").reshape([1, sq, 1]) // ratio)
    valid_mask = (comp_ids < valid_end).expand([b, sq, sk])
    neg_inf = paddle.full([b, sq, sk], float("-inf"), dtype="float32")
    causal_mask = paddle.where(valid_mask, paddle.zeros_like(neg_inf), neg_inf)
    actual_topk = min(int(topk_effective), int(sk))
    index_scores, ref_topk_indices = fused_qk_topk_naive(q, k, weights, index_topk=actual_topk, mask=causal_mask)
    index_scores_scaled = index_scores * sm_scale
    masked_scaled = index_scores_scaled + causal_mask
    topk_scores_raw, topk_indices = paddle.topk(masked_scaled, k=actual_topk, axis=-1)
    topk_indices = paddle.clip(topk_indices, min=0, max=sk - 1)
    valid_topk = paddle.take_along_axis(valid_mask.cast("int32"), topk_indices, axis=-1).cast("bool")
    topk_indices = paddle.where(valid_topk, topk_indices.cast("int32"), paddle.full_like(topk_indices, -1, dtype="int32"))
    topk_scores_raw = paddle.where(valid_topk, topk_scores_raw, paddle.full_like(topk_scores_raw, float("-inf")))
    row_has_valid = valid_topk.any(axis=-1, keepdim=True)
    safe_scores = paddle.where(row_has_valid, topk_scores_raw, paddle.zeros_like(topk_scores_raw))
    topk_probs = F.softmax(safe_scores.cast("float32"), axis=-1)
    topk_probs = paddle.where(row_has_valid, topk_probs, paddle.zeros_like(topk_probs))
    topk_probs = paddle.where(valid_topk, topk_probs, paddle.zeros_like(topk_probs))
    if int(topk_effective) > actual_topk:
        pad = int(topk_effective) - actual_topk
        topk_indices = paddle.concat([topk_indices, paddle.full([b, sq, pad], -1, dtype="int32")], axis=-1)
        topk_probs = paddle.concat([topk_probs, paddle.zeros([b, sq, pad], dtype="float32")], axis=-1)
    return topk_indices, topk_probs


# =========================================================================
# Kernel tests
# =========================================================================

class TestTileLangCSAIndexerKernel(unittest.TestCase):
    """Correctness of raw TileLang kernel interfaces."""

    def setUp(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA is required")
        paddle.set_device("gpu")

    def _run_kernel_fwd_case(self, topk_effective):
        from paddlefleet.tilelang_ops.kernel.tilelang_csa_indexer_fwd import csa_indexer_topk_fwd_interface

        batch, seq_len, seq_len_comp, heads, dim, ratio = 1, 16, 4, 64, 128, 4
        q, k, w = _make_indexer_inputs(batch, seq_len, seq_len_comp, heads, dim, seed=2026)
        out_idx, out_scores = csa_indexer_topk_fwd_interface(
            q, k, w, ratio=ratio, topk_effective=topk_effective, block_K=32, num_threads=128,
        )
        ref_idx, ref_scores = _ref_csa_indexer_topk(q, k, w, ratio, topk_effective)
        self.assertEqual(tuple(out_idx.shape), (batch, seq_len, topk_effective))
        self.assertTrue(paddle.all(out_idx.cpu() == ref_idx.cpu()).item())
        valid = ref_idx >= 0
        paddle.testing.assert_close(out_scores.cpu()[valid.cpu()], ref_scores.cpu()[valid.cpu()], rtol=6e-2, atol=2e-2)
        self.assertTrue(paddle.all(out_scores.cpu()[~valid.cpu()] == ref_scores.cpu()[~valid.cpu()]).item())
        self.assertTrue(paddle.all(out_idx[:, :3, :] == -1).item())

    def test_kernel_fwd_selected_topk(self):
        self._run_kernel_fwd_case(topk_effective=2)

    def test_kernel_fwd_full_candidate(self):
        self._run_kernel_fwd_case(topk_effective=4)

    def test_kernel_fwd_output_padding(self):
        self._run_kernel_fwd_case(topk_effective=6)

    def _ref_csa_indexer_bwd(self, index_q, weights, index_k_comp, topk_indices, grad_scores):
        q = index_q.detach().clone(); q.stop_gradient = False
        w = weights.detach().clone(); w.stop_gradient = False
        k = index_k_comp.detach().clone(); k.stop_gradient = False
        scores = paddle.einsum("bshd,btd->bsht", q.cast("float32"), k.cast("float32"))
        scores = F.relu(scores * (q.shape[-1] ** -0.5))
        scores = (scores * w.cast("float32").unsqueeze(-1)).sum(axis=2)
        valid = topk_indices >= 0
        safe_indices = paddle.clip(topk_indices, min=0).cast("int64")
        selected = paddle.take_along_axis(scores, safe_indices, axis=-1)
        selected = paddle.where(valid, selected, paddle.zeros_like(selected))
        (selected * grad_scores.cast("float32")).sum().backward()
        return q.grad, w.grad, k.grad

    def _run_kernel_bwd_case(self, topk_effective):
        from paddlefleet.tilelang_ops.kernel.tilelang_csa_indexer_fwd import csa_indexer_topk_fwd_interface
        from paddlefleet.tilelang_ops.kernel.tilelang_csa_indexer_bwd import csa_indexer_bwd_interface

        batch, seq_len, seq_len_comp, heads, dim, ratio = 1, 16, 4, 64, 128, 4
        q, k, w = _make_indexer_inputs(batch, seq_len, seq_len_comp, heads, dim, seed=2027)
        topk_indices, _ = csa_indexer_topk_fwd_interface(
            q, k, w, ratio=ratio, topk_effective=topk_effective, block_K=32, num_threads=128,
        )
        grad_scores = paddle.randn([batch, seq_len, topk_effective], dtype="float32")
        grad_scores = paddle.where(topk_indices >= 0, grad_scores, paddle.zeros_like(grad_scores)).contiguous()
        out_dq, out_dw, out_dk = csa_indexer_bwd_interface(
            q, w, k, topk_indices.contiguous(), grad_scores, block_I=32, num_threads=128,
        )
        ref_dq, ref_dw, ref_dk = self._ref_csa_indexer_bwd(q, w, k, topk_indices, grad_scores)
        self.assertEqual(tuple(out_dq.shape), tuple(q.shape))
        paddle.testing.assert_close(out_dq.cast("float32").cpu(), ref_dq.cast("float32").cpu(), rtol=6e-2, atol=2e-2)
        paddle.testing.assert_close(out_dw.cpu(), ref_dw.cpu(), rtol=6e-2, atol=3e-2)
        paddle.testing.assert_close(out_dk.cpu(), ref_dk.cast("float32").cpu(), rtol=6e-2, atol=3e-2)

    def test_kernel_bwd_selected_topk(self):
        self._run_kernel_bwd_case(topk_effective=2)

    def test_kernel_bwd_full_candidate(self):
        self._run_kernel_bwd_case(topk_effective=4)

    def test_kernel_bwd_output_padding(self):
        self._run_kernel_bwd_case(topk_effective=6)


# =========================================================================
# Wrapper tests
# =========================================================================

@unittest.skipUnless(
    paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0,
    "TileLang CSA Indexer kernel requires CUDA",
)
class TestTileLangCSAIndexerWrapperForward(unittest.TestCase):
    def setUp(self):
        from paddlefleet.tilelang_ops import tilelang_csa_compressed_indexer_topk_paddle
        self._kernel = tilelang_csa_compressed_indexer_topk_paddle

    def test_phase3_selected_topk_matches_reference(self):
        b, sq, sk, h_i, d_i, ratio = 1, 16, 4, 64, 128, 4
        topk_effective = 2
        q, k, w = _make_indexer_inputs(b, sq, sk, h_i, d_i)
        out_idx, out_prob = self._kernel(q, k, w, ratio=ratio, topk_effective=topk_effective)
        ref_idx, ref_prob = _paddle_ref_csa_indexer_topk(q, k, w, ratio, topk_effective)
        self.assertEqual(list(out_idx.shape), [b, sq, topk_effective])
        self.assertTrue(_sorted_compare_indices(out_idx, ref_idx))
        valid = ref_idx >= 0
        self.assertTrue(paddle.allclose(
            paddle.masked_select(out_prob, valid), paddle.masked_select(ref_prob, valid),
            rtol=8e-2, atol=3e-2,
        ).item())

    def test_phase2_full_candidate_matches_reference(self):
        b, sq, sk, h_i, d_i, ratio = 1, 16, 4, 64, 128, 4
        topk_effective = sk
        q, k, w = _make_indexer_inputs(b, sq, sk, h_i, d_i)
        out_idx, out_prob = self._kernel(q, k, w, ratio=ratio, topk_effective=topk_effective)
        ref_idx, ref_prob = _paddle_ref_csa_indexer_topk(q, k, w, ratio, topk_effective)
        self.assertTrue(_sorted_compare_indices(out_idx, ref_idx))
        valid = ref_idx >= 0
        self.assertTrue(paddle.allclose(
            paddle.masked_select(out_prob, valid), paddle.masked_select(ref_prob, valid),
            rtol=8e-2, atol=3e-2,
        ).item())

    def test_padded_n_compressed_matches_reference(self):
        b, sq, sk, h_i, d_i, ratio = 1, 16, 4, 64, 128, 4
        topk_effective = 6
        q, k, w = _make_indexer_inputs(b, sq, sk, h_i, d_i)
        out_idx, out_prob = self._kernel(q, k, w, ratio=ratio, topk_effective=topk_effective)
        ref_idx, ref_prob = _paddle_ref_csa_indexer_topk(q, k, w, ratio, topk_effective)
        self.assertEqual(list(out_idx.shape), [b, sq, topk_effective])
        self.assertTrue(_all_equal(out_idx[:, :, sk:], -1))
        self.assertTrue(_all_equal(out_prob[:, :, sk:], 0))
        self.assertTrue(_sorted_compare_indices(out_idx, ref_idx))

    def test_causal_t0_t1_t2_have_no_compressed_block(self):
        b, sq, sk, h_i, d_i, ratio = 1, 16, 4, 64, 128, 4
        q, k, w = _make_indexer_inputs(b, sq, sk, h_i, d_i)
        out_idx, out_prob = self._kernel(q, k, w, ratio=ratio, topk_effective=4)
        self.assertTrue(_all_equal(out_idx[:, :3, :], -1))
        self.assertTrue(_all_equal(out_prob[:, :3, :], 0))

    def test_causal_t3_only_block_zero_visible(self):
        b, sq, sk, h_i, d_i, ratio = 1, 16, 4, 64, 128, 4
        q, k, w = _make_indexer_inputs(b, sq, sk, h_i, d_i)
        out_idx, _ = self._kernel(q, k, w, ratio=ratio, topk_effective=4)
        row = out_idx[0, 3].numpy().tolist()
        self.assertIn(0, row)
        self.assertEqual(sum(int(x == -1) for x in row), 3)

    def test_causal_t7_blocks_zero_and_one_visible(self):
        b, sq, sk, h_i, d_i, ratio = 1, 16, 4, 64, 128, 4
        q, k, w = _make_indexer_inputs(b, sq, sk, h_i, d_i)
        out_idx, _ = self._kernel(q, k, w, ratio=ratio, topk_effective=4)
        row = sorted(out_idx[0, 7].numpy().tolist())
        valid = [x for x in row if x != -1]
        self.assertEqual(sorted(valid), [0, 1])

    def test_short_sequence_with_valid_end_less_than_topk(self):
        b, sq, sk, h_i, d_i, ratio = 1, 8, 4, 64, 128, 4
        topk_effective = 4
        q, k, w = _make_indexer_inputs(b, sq, sk, h_i, d_i)
        out_idx, _ = self._kernel(q, k, w, ratio=ratio, topk_effective=topk_effective)
        ref_idx, _ = _paddle_ref_csa_indexer_topk(q, k, w, ratio, topk_effective)
        for t in range(sq):
            valid_end = (t + 1) // ratio
            row = out_idx[0, t].numpy().tolist()
            n_valid = sum(int(x >= 0) for x in row)
            self.assertEqual(n_valid, min(valid_end, topk_effective))
        self.assertTrue(_sorted_compare_indices(out_idx, ref_idx))


# =========================================================================
# PyLayer backward tests
# =========================================================================

@unittest.skipUnless(
    paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0,
    "TileLang CSA Indexer kernel requires CUDA",
)
class TestTileLangCSAIndexerLossGrad(unittest.TestCase):
    B, SQ, SK, H_I, D_I, NP, HN, RATIO = 1, 16, 4, 64, 128, 1, 128, 4
    LOSS_COEFF = 1.0

    def _common_inputs(self, seed=2027):
        return _make_loss_inputs(self.B, self.SQ, self.SK, self.H_I, self.D_I, self.NP, self.HN, seed=seed)

    def _assert_grads_close(self, tl_grads, pd_grads, label):
        tl_dq, tl_dw, tl_dk = tl_grads
        pd_dq, pd_dw, pd_dk = pd_grads
        _assert_close(tl_dq, pd_dq, rtol=8e-2, atol=3e-2, msg=f"{label}: dQ mismatch")
        _assert_close(tl_dw, pd_dw, rtol=8e-2, atol=3e-2, msg=f"{label}: dWeights mismatch")
        _assert_close(tl_dk, pd_dk, rtol=8e-2, atol=3e-2, msg=f"{label}: dKComp mismatch")

    def _run_tilelang(self, q, k, weights, query_mla, key_comp_mla, ratio, topk_eff, softmax_scale, loss_coeff):
        from paddlefleet.transformer.csa_attention import TileLangCSAIndexerLoss
        qd = q.detach().clone(); qd.stop_gradient = False
        kd = k.detach().clone(); kd.stop_gradient = False
        wd = weights.detach().clone(); wd.stop_gradient = False
        loss = TileLangCSAIndexerLoss.apply(
            qd, wd, kd, query_mla, key_comp_mla,
            int(ratio), int(topk_eff), float(softmax_scale), float(loss_coeff), None,
        )
        loss.backward()
        return loss, qd.grad, wd.grad, kd.grad

    def _run_paddle(self, q, k, weights, query_mla, key_comp_mla, ratio, topk, softmax_scale, loss_coeff, sparse_loss):
        from paddlefleet.transformer.dsa_attention import FusedDSAIndexerLoss
        d_i = q.shape[-1]; alpha = d_i ** -0.5
        q_sf = q.detach().transpose([1, 0, 2, 3]).clone(); q_sf.stop_gradient = False
        k_sf = k.detach().transpose([1, 0, 2]).clone(); k_sf.stop_gradient = False
        w_sf = (weights.detach() * alpha).transpose([1, 0, 2]).clone(); w_sf.stop_gradient = False
        query_sf = query_mla.transpose([1, 0, 2, 3]).detach()
        key_sf = key_comp_mla.transpose([1, 0, 2, 3]).detach()
        mask = _build_csa_causal_mask(q.shape[0], q.shape[1], k.shape[1], ratio)
        loss = FusedDSAIndexerLoss.apply(
            q_sf, w_sf, k_sf, query_sf, key_sf,
            float(softmax_scale), int(topk), float(loss_coeff), mask, bool(sparse_loss), None,
        )
        loss.backward()
        dq_bf = q_sf.grad.transpose([1, 0, 2, 3])
        dk_bf = k_sf.grad.transpose([1, 0, 2])
        dw_bf = w_sf.grad.transpose([1, 0, 2]) * alpha
        return loss, dq_bf, dw_bf, dk_bf

    def test_phase3_selected_topk_grad(self):
        q, k, w, qm, km = self._common_inputs()
        topk_eff = 2; softmax_scale = self.HN ** -0.5
        tl_loss, tl_dq, tl_dw, tl_dk = self._run_tilelang(q, k, w, qm, km, self.RATIO, topk_eff, softmax_scale, self.LOSS_COEFF)
        pd_loss, pd_dq, pd_dw, pd_dk = self._run_paddle(q, k, w, qm, km, self.RATIO, topk_eff, softmax_scale, self.LOSS_COEFF, sparse_loss=True)
        _assert_close(tl_loss, pd_loss, rtol=5e-2, atol=2e-3, msg="Phase3 loss mismatch")
        self._assert_grads_close((tl_dq, tl_dw, tl_dk), (pd_dq, pd_dw, pd_dk), "Phase3")

    def test_phase2_full_range_grad(self):
        q, k, w, qm, km = self._common_inputs(seed=2028)
        topk_eff = self.SK; softmax_scale = self.HN ** -0.5
        tl_loss, tl_dq, tl_dw, tl_dk = self._run_tilelang(q, k, w, qm, km, self.RATIO, topk_eff, softmax_scale, self.LOSS_COEFF)
        pd_loss, pd_dq, pd_dw, pd_dk = self._run_paddle(q, k, w, qm, km, self.RATIO, topk_eff, softmax_scale, self.LOSS_COEFF, sparse_loss=False)
        _assert_close(tl_loss, pd_loss, rtol=5e-2, atol=2e-3, msg="Phase2 loss mismatch")
        self._assert_grads_close((tl_dq, tl_dw, tl_dk), (pd_dq, pd_dw, pd_dk), "Phase2")

    def test_invalid_padding_rows_have_zero_grad(self):
        q, k, w, qm, km = self._common_inputs(seed=2029)
        topk_eff = 2; softmax_scale = self.HN ** -0.5
        _loss, tl_dq, tl_dw, _tl_dk = self._run_tilelang(q, k, w, qm, km, self.RATIO, topk_eff, softmax_scale, self.LOSS_COEFF)
        for g, name in ((tl_dq, "dQ"), (tl_dw, "dWeights")):
            gf = g[:, :3].cast("float32")
            _assert_close(gf, paddle.zeros_like(gf), rtol=0.0, atol=1e-5, msg=f"{name} on valid_end=0 rows")

    def test_padded_topk_does_not_change_grad(self):
        q, k, w, qm, km = self._common_inputs(seed=2030)
        softmax_scale = self.HN ** -0.5
        _l_full, dq_full, dw_full, dk_full = self._run_tilelang(q, k, w, qm, km, self.RATIO, self.SK, softmax_scale, self.LOSS_COEFF)
        _l_pad, dq_pad, dw_pad, dk_pad = self._run_tilelang(q, k, w, qm, km, self.RATIO, self.SK + 2, softmax_scale, self.LOSS_COEFF)
        _assert_close(dq_full, dq_pad, rtol=1e-3, atol=1e-4, msg="dQ invariant to topk padding")
        _assert_close(dw_full, dw_pad, rtol=1e-3, atol=1e-4, msg="dWeights invariant to topk padding")
        _assert_close(dk_full, dk_pad, rtol=1e-3, atol=1e-4, msg="dKComp invariant to topk padding")

    def test_training_step_no_nan_and_loss_backprops(self):
        q, k, w, qm, km = self._common_inputs(seed=2031)
        topk_eff = 2; softmax_scale = self.HN ** -0.5
        loss, dq, dw, dk = self._run_tilelang(q, k, w, qm, km, self.RATIO, topk_eff, softmax_scale, self.LOSS_COEFF)
        loss_fp = loss.cast("float32")
        self.assertTrue(paddle.isfinite(loss_fp).all().item(), "loss is not finite")
        self.assertGreater(loss_fp.item(), 0.0, "KL loss must be positive")
        for name, g in (("dQ", dq), ("dWeights", dw), ("dKComp", dk)):
            gf = g.cast("float32")
            self.assertTrue(paddle.isfinite(gf).all().item(), f"{name} contains NaN/Inf")
            self.assertGreater(gf.abs().max().item(), 0.0, f"{name} is identically zero")


if __name__ == "__main__":
    unittest.main()