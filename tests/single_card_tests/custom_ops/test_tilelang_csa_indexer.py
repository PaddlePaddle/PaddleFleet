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

import unittest

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
    key_comp_mla = paddle.randn([b, sk, hn]).astype("bfloat16").detach()
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
    e = (
        expected.cast("float32")
        if expected.dtype != paddle.float32
        else expected
    )
    if not paddle.allclose(a, e, rtol=rtol, atol=atol).item():
        diff = (a - e).abs()
        raise AssertionError(
            f"{msg}\n  max abs diff: {diff.max().item():.4e}\n"
            f"  max rel diff: {(diff / e.abs().clip(min=1e-12)).max().item():.4e}"
        )


# =========================================================================
# Reference implementations
# =========================================================================


def _ref_csa_indexer_topk(
    index_q, index_k_comp, weights, ratio, topk_effective
):
    scores = paddle.einsum(
        "bshd,btd->bsht", index_q.cast("float32"), index_k_comp.cast("float32")
    )
    scores = F.relu(scores)
    scores = (scores * weights.cast("float32").unsqueeze(-1)).sum(axis=2)
    scores = scores * (index_q.shape[-1] ** -0.5)
    batch, seq_len, seq_len_comp = scores.shape
    comp_ids = paddle.arange(seq_len_comp, dtype="int64").reshape(
        [1, 1, seq_len_comp]
    )
    positions = paddle.arange(1, seq_len + 1, dtype="int64").reshape(
        [1, seq_len, 1]
    )
    valid_end = positions // ratio
    valid_mask = comp_ids < valid_end
    scores = paddle.where(
        valid_mask, scores, paddle.full_like(scores, float("-inf"))
    )
    actual_topk = min(topk_effective, seq_len_comp)
    topk_scores_raw, topk_indices = paddle.topk(scores, k=actual_topk, axis=-1)
    valid_topk = paddle.take_along_axis(
        paddle.expand(valid_mask, [batch, seq_len, seq_len_comp]).cast("int32"),
        topk_indices,
        axis=-1,
    ).cast("bool")
    topk_indices = paddle.where(
        valid_topk, topk_indices, paddle.full_like(topk_indices, -1)
    )
    topk_scores_raw = paddle.where(
        valid_topk,
        topk_scores_raw,
        paddle.full_like(topk_scores_raw, float("-inf")),
    )
    topk_probs = F.softmax(topk_scores_raw, axis=-1)
    topk_probs = paddle.where(
        valid_topk, topk_probs, paddle.zeros_like(topk_probs)
    )
    if topk_effective > actual_topk:
        pad = topk_effective - actual_topk
        topk_indices = paddle.concat(
            [
                topk_indices,
                paddle.full(
                    [batch, seq_len, pad], -1, dtype=topk_indices.dtype
                ),
            ],
            axis=-1,
        )
        topk_probs = paddle.concat(
            [
                topk_probs,
                paddle.zeros([batch, seq_len, pad], dtype=topk_probs.dtype),
            ],
            axis=-1,
        )
    return topk_indices.cast("int32"), topk_probs.cast("float32")


def _paddle_ref_csa_indexer_topk(q, k, weights, ratio, topk_effective):
    from paddlefleet.transformer.dsa_attention import fused_qk_topk_naive

    b, sq, h_i, d_i = q.shape
    sk = k.shape[1]
    sm_scale = d_i**-0.5
    comp_ids = paddle.arange(sk, dtype="int64").reshape([1, 1, sk])
    valid_end = (
        paddle.arange(1, sq + 1, dtype="int64").reshape([1, sq, 1]) // ratio
    )
    valid_mask = (comp_ids < valid_end).expand([b, sq, sk])
    neg_inf = paddle.full([b, sq, sk], float("-inf"), dtype="float32")
    causal_mask = paddle.where(valid_mask, paddle.zeros_like(neg_inf), neg_inf)
    actual_topk = min(int(topk_effective), int(sk))
    index_scores, ref_topk_indices = fused_qk_topk_naive(
        q, k, weights, index_topk=actual_topk, mask=causal_mask
    )
    index_scores_scaled = index_scores * sm_scale
    masked_scaled = index_scores_scaled + causal_mask
    topk_scores_raw, topk_indices = paddle.topk(
        masked_scaled, k=actual_topk, axis=-1
    )
    topk_indices = paddle.clip(topk_indices, min=0, max=sk - 1)
    valid_topk = paddle.take_along_axis(
        valid_mask.cast("int32"), topk_indices, axis=-1
    ).cast("bool")
    topk_indices = paddle.where(
        valid_topk,
        topk_indices.cast("int32"),
        paddle.full_like(topk_indices, -1, dtype="int32"),
    )
    topk_scores_raw = paddle.where(
        valid_topk,
        topk_scores_raw,
        paddle.full_like(topk_scores_raw, float("-inf")),
    )
    row_has_valid = valid_topk.any(axis=-1, keepdim=True)
    safe_scores = paddle.where(
        row_has_valid, topk_scores_raw, paddle.zeros_like(topk_scores_raw)
    )
    topk_probs = F.softmax(safe_scores.cast("float32"), axis=-1)
    topk_probs = paddle.where(
        row_has_valid, topk_probs, paddle.zeros_like(topk_probs)
    )
    topk_probs = paddle.where(
        valid_topk, topk_probs, paddle.zeros_like(topk_probs)
    )
    if int(topk_effective) > actual_topk:
        pad = int(topk_effective) - actual_topk
        topk_indices = paddle.concat(
            [topk_indices, paddle.full([b, sq, pad], -1, dtype="int32")],
            axis=-1,
        )
        topk_probs = paddle.concat(
            [topk_probs, paddle.zeros([b, sq, pad], dtype="float32")], axis=-1
        )
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
        from paddlefleet.tilelang_ops.indexer.csa_indexer_fwd import (
            csa_indexer_topk_fwd_interface,
        )

        batch, seq_len, seq_len_comp, heads, dim, ratio = 1, 16, 4, 64, 128, 4
        q, k, w = _make_indexer_inputs(
            batch, seq_len, seq_len_comp, heads, dim, seed=2026
        )
        out_idx, out_scores = csa_indexer_topk_fwd_interface(
            q,
            k,
            w,
            ratio=ratio,
            topk_effective=topk_effective,
            block_K=32,
            num_threads=128,
        )
        ref_idx, ref_scores = _ref_csa_indexer_topk(
            q, k, w, ratio, topk_effective
        )
        self.assertEqual(tuple(out_idx.shape), (batch, seq_len, topk_effective))
        self.assertTrue(paddle.all(out_idx.cpu() == ref_idx.cpu()).item())
        valid = ref_idx >= 0
        paddle.testing.assert_close(
            out_scores.cpu()[valid.cpu()],
            ref_scores.cpu()[valid.cpu()],
            rtol=6e-2,
            atol=2e-2,
        )
        self.assertTrue(
            paddle.all(
                out_scores.cpu()[~valid.cpu()] == ref_scores.cpu()[~valid.cpu()]
            ).item()
        )
        self.assertTrue(paddle.all(out_idx[:, :3, :] == -1).item())

    def test_kernel_fwd_selected_topk(self):
        self._run_kernel_fwd_case(topk_effective=2)

    def test_kernel_fwd_full_candidate(self):
        self._run_kernel_fwd_case(topk_effective=4)

    def test_kernel_fwd_output_padding(self):
        self._run_kernel_fwd_case(topk_effective=6)

    def _ref_csa_indexer_bwd(
        self, index_q, weights, index_k_comp, topk_indices, grad_scores
    ):
        q = index_q.detach().clone()
        q.stop_gradient = False
        w = weights.detach().clone()
        w.stop_gradient = False
        k = index_k_comp.detach().clone()
        k.stop_gradient = False
        scores = paddle.einsum(
            "bshd,btd->bsht", q.cast("float32"), k.cast("float32")
        )
        scores = F.relu(scores * (q.shape[-1] ** -0.5))
        scores = (scores * w.cast("float32").unsqueeze(-1)).sum(axis=2)
        valid = topk_indices >= 0
        safe_indices = paddle.clip(topk_indices, min=0).cast("int64")
        selected = paddle.take_along_axis(scores, safe_indices, axis=-1)
        selected = paddle.where(valid, selected, paddle.zeros_like(selected))
        (selected * grad_scores.cast("float32")).sum().backward()
        return q.grad, w.grad, k.grad

    def _run_kernel_bwd_case(self, topk_effective):
        from paddlefleet.tilelang_ops.indexer.csa_indexer_bwd import (
            csa_indexer_bwd_interface,
        )
        from paddlefleet.tilelang_ops.indexer.csa_indexer_fwd import (
            csa_indexer_topk_fwd_interface,
        )

        batch, seq_len, seq_len_comp, heads, dim, ratio = 1, 16, 4, 64, 128, 4
        q, k, w = _make_indexer_inputs(
            batch, seq_len, seq_len_comp, heads, dim, seed=2027
        )
        topk_indices, _ = csa_indexer_topk_fwd_interface(
            q,
            k,
            w,
            ratio=ratio,
            topk_effective=topk_effective,
            block_K=32,
            num_threads=128,
        )
        grad_scores = paddle.randn(
            [batch, seq_len, topk_effective], dtype="float32"
        )
        grad_scores = paddle.where(
            topk_indices >= 0, grad_scores, paddle.zeros_like(grad_scores)
        ).contiguous()
        out_dq, out_dw, out_dk = csa_indexer_bwd_interface(
            q,
            w,
            k,
            topk_indices.contiguous(),
            grad_scores,
            block_I=32,
            num_threads=128,
        )
        ref_dq, ref_dw, ref_dk = self._ref_csa_indexer_bwd(
            q, w, k, topk_indices, grad_scores
        )
        self.assertEqual(tuple(out_dq.shape), tuple(q.shape))
        paddle.testing.assert_close(
            out_dq.cast("float32").cpu(),
            ref_dq.cast("float32").cpu(),
            rtol=6e-2,
            atol=2e-2,
        )
        paddle.testing.assert_close(
            out_dw.cpu(), ref_dw.cpu(), rtol=6e-2, atol=3e-2
        )
        paddle.testing.assert_close(
            out_dk.cpu(), ref_dk.cast("float32").cpu(), rtol=6e-2, atol=3e-2
        )

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
    paddle.device.is_compiled_with_cuda()
    and paddle.device.cuda.device_count() > 0,
    "TileLang CSA Indexer kernel requires CUDA",
)
class TestTileLangCSAIndexerWrapperForward(unittest.TestCase):
    def setUp(self):
        from paddlefleet.tilelang_ops import csa_indexer_topk_fwd

        self._kernel = csa_indexer_topk_fwd

    def test_phase3_selected_topk_matches_reference(self):
        b, sq, sk, h_i, d_i, ratio = 1, 16, 4, 64, 128, 4
        topk_effective = 2
        q, k, w = _make_indexer_inputs(b, sq, sk, h_i, d_i)
        out_idx, out_prob = self._kernel(
            q, k, w, ratio=ratio, topk_effective=topk_effective
        )
        ref_idx, ref_prob = _paddle_ref_csa_indexer_topk(
            q, k, w, ratio, topk_effective
        )
        self.assertEqual(list(out_idx.shape), [b, sq, topk_effective])
        self.assertTrue(_sorted_compare_indices(out_idx, ref_idx))
        valid = ref_idx >= 0
        self.assertTrue(
            paddle.allclose(
                paddle.masked_select(out_prob, valid),
                paddle.masked_select(ref_prob, valid),
                rtol=8e-2,
                atol=3e-2,
            ).item()
        )

    def test_phase2_full_candidate_matches_reference(self):
        b, sq, sk, h_i, d_i, ratio = 1, 16, 4, 64, 128, 4
        topk_effective = sk
        q, k, w = _make_indexer_inputs(b, sq, sk, h_i, d_i)
        out_idx, out_prob = self._kernel(
            q, k, w, ratio=ratio, topk_effective=topk_effective
        )
        ref_idx, ref_prob = _paddle_ref_csa_indexer_topk(
            q, k, w, ratio, topk_effective
        )
        self.assertTrue(_sorted_compare_indices(out_idx, ref_idx))
        valid = ref_idx >= 0
        self.assertTrue(
            paddle.allclose(
                paddle.masked_select(out_prob, valid),
                paddle.masked_select(ref_prob, valid),
                rtol=8e-2,
                atol=3e-2,
            ).item()
        )

    def test_padded_n_compressed_matches_reference(self):
        b, sq, sk, h_i, d_i, ratio = 1, 16, 4, 64, 128, 4
        topk_effective = 6
        q, k, w = _make_indexer_inputs(b, sq, sk, h_i, d_i)
        out_idx, out_prob = self._kernel(
            q, k, w, ratio=ratio, topk_effective=topk_effective
        )
        ref_idx, ref_prob = _paddle_ref_csa_indexer_topk(
            q, k, w, ratio, topk_effective
        )
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
        out_idx, _ = self._kernel(
            q, k, w, ratio=ratio, topk_effective=topk_effective
        )
        ref_idx, _ = _paddle_ref_csa_indexer_topk(
            q, k, w, ratio, topk_effective
        )
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
    paddle.device.is_compiled_with_cuda()
    and paddle.device.cuda.device_count() > 0,
    "TileLang CSA Indexer kernel requires CUDA",
)
class TestTileLangCSAIndexerLossGrad(unittest.TestCase):
    B, SQ, SK, H_I, D_I, NP, HN, RATIO = 1, 16, 4, 64, 128, 1, 128, 4
    LOSS_COEFF = 1.0

    def _common_inputs(self, seed=2027):
        return _make_loss_inputs(
            self.B,
            self.SQ,
            self.SK,
            self.H_I,
            self.D_I,
            self.NP,
            self.HN,
            seed=seed,
        )

    def _assert_grads_close(self, tl_grads, pd_grads, label):
        tl_dq, tl_dw, tl_dk = tl_grads
        pd_dq, pd_dw, pd_dk = pd_grads
        _assert_close(
            tl_dq, pd_dq, rtol=8e-2, atol=3e-2, msg=f"{label}: dQ mismatch"
        )
        _assert_close(
            tl_dw,
            pd_dw,
            rtol=8e-2,
            atol=3e-2,
            msg=f"{label}: dWeights mismatch",
        )
        _assert_close(
            tl_dk, pd_dk, rtol=8e-2, atol=3e-2, msg=f"{label}: dKComp mismatch"
        )

    def _run_tilelang(
        self,
        q,
        k,
        weights,
        query_mla,
        key_comp_mla,
        ratio,
        topk_eff,
        softmax_scale,
        loss_coeff,
    ):
        from paddlefleet.transformer.csa_attention import TileLangCSAIndexerLoss

        qd = q.detach().clone()
        qd.stop_gradient = False
        kd = k.detach().clone()
        kd.stop_gradient = False
        wd = weights.detach().clone()
        wd.stop_gradient = False
        loss, _topk_indices = TileLangCSAIndexerLoss.apply(
            qd,
            wd,
            kd,
            query_mla,
            key_comp_mla,
            int(ratio),
            int(topk_eff),
            float(softmax_scale),
            float(loss_coeff),
            None,
        )
        loss.backward()
        return loss, qd.grad, wd.grad, kd.grad

    def _run_paddle(
        self,
        q,
        k,
        weights,
        query_mla,
        key_comp_mla,
        ratio,
        topk,
        softmax_scale,
        loss_coeff,
        sparse_loss,
    ):
        from paddlefleet.transformer.dsa_attention import FusedDSAIndexerLoss

        d_i = q.shape[-1]
        alpha = d_i**-0.5
        q_sf = q.detach().transpose([1, 0, 2, 3]).clone()
        q_sf.stop_gradient = False
        k_sf = k.detach().transpose([1, 0, 2]).clone()
        k_sf.stop_gradient = False
        w_sf = (weights.detach() * alpha).transpose([1, 0, 2]).clone()
        w_sf.stop_gradient = False
        query_sf = query_mla.transpose([1, 0, 2, 3]).detach()
        key_expanded = key_comp_mla.unsqueeze(2).expand(
            [q.shape[0], k.shape[1], query_mla.shape[2], query_mla.shape[3]]
        )
        key_sf = key_expanded.transpose([1, 0, 2, 3]).detach()
        mask = _build_csa_causal_mask(q.shape[0], q.shape[1], k.shape[1], ratio)
        loss = FusedDSAIndexerLoss.apply(
            q_sf,
            w_sf,
            k_sf,
            query_sf,
            key_sf,
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

    def _assert_attn_target_matches_paddle(
        self, q, k, w, qm, km, topk_eff, softmax_scale, label
    ):
        from paddlefleet.tilelang_ops import (
            csa_attn_target_reducesum,
            csa_indexer_topk_fwd,
        )
        from paddlefleet.transformer.csa_attention import (
            _compute_attn_target_on_selected_set,
        )

        topk_indices, _ = csa_indexer_topk_fwd(
            q, k, w, ratio=self.RATIO, topk_effective=topk_eff
        )
        tl_target = csa_attn_target_reducesum(
            qm, km, topk_indices, softmax_scale
        )
        pd_target = _compute_attn_target_on_selected_set(
            qm, km, topk_indices, softmax_scale, None
        )
        _assert_close(
            tl_target,
            pd_target,
            rtol=6e-2,
            atol=2e-2,
            msg=f"{label}: attention target mismatch",
        )

    def test_attn_target_reducesum_matches_paddle(self):
        q, k, w, qm, km = self._common_inputs(seed=2032)
        topk_eff = 2
        softmax_scale = self.HN**-0.5
        self._assert_attn_target_matches_paddle(
            q, k, w, qm, km, topk_eff, softmax_scale, "single-head"
        )

    def test_attn_target_reducesum_multi_head_matches_paddle(self):
        q, k, w, qm, km = _make_loss_inputs(
            self.B,
            self.SQ,
            self.SK,
            self.H_I,
            self.D_I,
            128,
            self.HN,
            seed=2033,
        )
        topk_eff = 2
        softmax_scale = self.HN**-0.5
        self._assert_attn_target_matches_paddle(
            q, k, w, qm, km, topk_eff, softmax_scale, "multi-head"
        )

    def test_phase3_selected_topk_grad(self):
        q, k, w, qm, km = self._common_inputs()
        topk_eff = 2
        softmax_scale = self.HN**-0.5
        tl_loss, tl_dq, tl_dw, tl_dk = self._run_tilelang(
            q,
            k,
            w,
            qm,
            km,
            self.RATIO,
            topk_eff,
            softmax_scale,
            self.LOSS_COEFF,
        )
        pd_loss, pd_dq, pd_dw, pd_dk = self._run_paddle(
            q,
            k,
            w,
            qm,
            km,
            self.RATIO,
            topk_eff,
            softmax_scale,
            self.LOSS_COEFF,
            sparse_loss=True,
        )
        _assert_close(
            tl_loss, pd_loss, rtol=5e-2, atol=2e-3, msg="Phase3 loss mismatch"
        )
        self._assert_grads_close(
            (tl_dq, tl_dw, tl_dk), (pd_dq, pd_dw, pd_dk), "Phase3"
        )

    def test_phase2_full_range_grad(self):
        q, k, w, qm, km = self._common_inputs(seed=2028)
        topk_eff = self.SK
        softmax_scale = self.HN**-0.5
        tl_loss, tl_dq, tl_dw, tl_dk = self._run_tilelang(
            q,
            k,
            w,
            qm,
            km,
            self.RATIO,
            topk_eff,
            softmax_scale,
            self.LOSS_COEFF,
        )
        pd_loss, pd_dq, pd_dw, pd_dk = self._run_paddle(
            q,
            k,
            w,
            qm,
            km,
            self.RATIO,
            topk_eff,
            softmax_scale,
            self.LOSS_COEFF,
            sparse_loss=False,
        )
        _assert_close(
            tl_loss, pd_loss, rtol=5e-2, atol=2e-3, msg="Phase2 loss mismatch"
        )
        self._assert_grads_close(
            (tl_dq, tl_dw, tl_dk), (pd_dq, pd_dw, pd_dk), "Phase2"
        )

    def test_invalid_padding_rows_have_zero_grad(self):
        q, k, w, qm, km = self._common_inputs(seed=2029)
        topk_eff = 2
        softmax_scale = self.HN**-0.5
        _loss, tl_dq, tl_dw, _tl_dk = self._run_tilelang(
            q,
            k,
            w,
            qm,
            km,
            self.RATIO,
            topk_eff,
            softmax_scale,
            self.LOSS_COEFF,
        )
        for g, name in ((tl_dq, "dQ"), (tl_dw, "dWeights")):
            gf = g[:, :3].cast("float32")
            _assert_close(
                gf,
                paddle.zeros_like(gf),
                rtol=0.0,
                atol=1e-5,
                msg=f"{name} on valid_end=0 rows",
            )

    def test_padded_topk_does_not_change_grad(self):
        q, k, w, qm, km = self._common_inputs(seed=2030)
        softmax_scale = self.HN**-0.5
        _l_full, dq_full, dw_full, dk_full = self._run_tilelang(
            q, k, w, qm, km, self.RATIO, self.SK, softmax_scale, self.LOSS_COEFF
        )
        _l_pad, dq_pad, dw_pad, dk_pad = self._run_tilelang(
            q,
            k,
            w,
            qm,
            km,
            self.RATIO,
            self.SK + 2,
            softmax_scale,
            self.LOSS_COEFF,
        )
        _assert_close(
            dq_full,
            dq_pad,
            rtol=1e-3,
            atol=1e-4,
            msg="dQ invariant to topk padding",
        )
        _assert_close(
            dw_full,
            dw_pad,
            rtol=1e-3,
            atol=1e-4,
            msg="dWeights invariant to topk padding",
        )
        _assert_close(
            dk_full,
            dk_pad,
            rtol=1e-3,
            atol=1e-4,
            msg="dKComp invariant to topk padding",
        )

    def test_training_step_no_nan_and_loss_backprops(self):
        q, k, w, qm, km = self._common_inputs(seed=2031)
        topk_eff = 2
        softmax_scale = self.HN**-0.5
        loss, dq, dw, dk = self._run_tilelang(
            q,
            k,
            w,
            qm,
            km,
            self.RATIO,
            topk_eff,
            softmax_scale,
            self.LOSS_COEFF,
        )
        loss_fp = loss.cast("float32")
        self.assertTrue(
            paddle.isfinite(loss_fp).all().item(), "loss is not finite"
        )
        self.assertGreater(loss_fp.item(), 0.0, "KL loss must be positive")
        for name, g in (("dQ", dq), ("dWeights", dw), ("dKComp", dk)):
            gf = g.cast("float32")
            self.assertTrue(
                paddle.isfinite(gf).all().item(), f"{name} contains NaN/Inf"
            )
            self.assertGreater(
                gf.abs().max().item(), 0.0, f"{name} is identically zero"
            )


@unittest.skipUnless(
    paddle.device.is_compiled_with_cuda()
    and paddle.device.cuda.device_count() > 0,
    "TileLang CSA attn_target kernel requires CUDA",
)
class TestCSAAttnTargetReducesum(unittest.TestCase):
    """Isolated correctness tests for csa_attn_target_reducesum kernel."""

    def _run_and_compare(self, b, sq, sk, np_, hn, topk_eff, ratio, seed=2040):
        from paddlefleet.tilelang_ops import (
            csa_attn_target_reducesum,
            csa_indexer_topk_fwd,
        )
        from paddlefleet.transformer.csa_attention import (
            _compute_attn_target_on_selected_set,
        )

        paddle.seed(seed)
        h_i, d_i = 64, 128
        q = paddle.randn([b, sq, h_i, d_i]).astype("bfloat16")
        k = paddle.randn([b, sk, d_i]).astype("bfloat16")
        w = paddle.randn([b, sq, h_i]).astype("float32")
        query_mla = paddle.randn([b, sq, np_, hn]).astype("bfloat16").detach()
        # MLA invariant: compressed key is shared across all heads → [B, S_comp, D]
        key_comp_mla = (
            paddle.randn([b, sk, hn]).astype("bfloat16").contiguous().detach()
        )
        softmax_scale = hn**-0.5

        topk_indices, _ = csa_indexer_topk_fwd(
            q, k, w, ratio=ratio, topk_effective=topk_eff
        )
        tl_target = csa_attn_target_reducesum(
            query_mla, key_comp_mla, topk_indices, softmax_scale
        )
        pd_target = _compute_attn_target_on_selected_set(
            query_mla, key_comp_mla, topk_indices, softmax_scale, None
        )
        _assert_close(
            tl_target,
            pd_target,
            rtol=6e-2,
            atol=2e-2,
            msg=f"target mismatch [b={b},sq={sq},sk={sk},np={np_},topk={topk_eff}]",
        )
        # L1 normalization: valid rows should sum to ~1
        valid = topk_indices >= 0
        row_valid = valid.any(axis=-1)
        if row_valid.any().item():
            row_sums = tl_target[row_valid].sum(axis=-1)
            _assert_close(
                row_sums,
                paddle.ones_like(row_sums),
                rtol=1e-4,
                atol=1e-4,
                msg="L1 normalization violated",
            )
        # Invalid rows should be all-zero
        if (~row_valid).any().item():
            zeros = tl_target[~row_valid]
            self.assertTrue(
                (zeros.cast("float32").abs() < 1e-6).all().item(),
                "invalid rows should be all-zero",
            )

    def test_multi_head_replicate(self):
        """heads=128 (>64) triggers REPLICATE_H=2 path in kernel.

        The kernel splits 128 MLA heads into two groups of 64, computes
        partial softmax per group, then sums and L1-normalizes. Keys are
        head-shared (MLA invariant) so both groups read the same K vector.
        A bug would show as broken partial-sum aggregation.
        """
        self._run_and_compare(
            b=1, sq=16, sk=4, np_=128, hn=128, topk_eff=2, ratio=4, seed=2041
        )

    def test_multi_block_topk_with_padding(self):
        """topk_eff=48 with block_I=32 → padded to 64 → 2 tile iterations.

        This tests: the online softmax correctly accumulates across
        multiple K-blocks rather than just one; (2) the interface layer
        correctly pads topk_indices with -1 to align to block_I and trims
        back to topk_eff=48 in the output; (3) the pad slots with index=-1
        are masked to -inf and contribute zero probability.
        """
        self._run_and_compare(
            b=2, sq=32, sk=16, np_=64, hn=128, topk_eff=48, ratio=4, seed=2042
        )

    def test_online_softmax_numerical_stability(self):
        """Validate online softmax under adversarial numeric conditions.

        Exercises three properties specific to 2-pass online softmax:
        1. Cross-block max shift: block 1 has logits ~10x larger than block 0,
           forcing the rescale exp(old_max - new_max) to a very small value.
           A bug in the online update would produce incorrect probabilities.
        2. All-invalid row: a query position with all topk_indices = -1 must
           produce all-zero output (tests NaN-safe path when row_max = -inf).
        3. Single valid entry: only 1 valid index per row; output must be 1.0
           for that entry (tests exact normalization in degenerate case).
        """
        from paddlefleet.tilelang_ops import csa_attn_target_reducesum
        from paddlefleet.transformer.csa_attention import (
            _compute_attn_target_on_selected_set,
        )

        _cuda_or_skip(self)
        paddle.set_device("gpu")

        b, np_, hn = 1, 64, 128
        sk = 8  # compressed sequence length
        topk = 64  # 2 blocks of block_I=32
        softmax_scale = hn**-0.5

        # Construct keys: first 4 keys are "small", last 4 are "large" (10x scale).
        # This ensures block 1 (indices 4-7) dominates, forcing cross-block rescale.
        paddle.seed(7777)
        key_small = paddle.randn([b, 4, hn]).astype("bfloat16") * 0.1
        key_large = paddle.randn([b, 4, hn]).astype("bfloat16") * 1.0
        key_comp_mla = (
            paddle.concat([key_small, key_large], axis=1).contiguous().detach()
        )

        # --- Case 1: cross-block max shift ---
        # Place small-key indices (0-3) in block 0 (positions 0-3) and large-key
        # indices (4-7) in block 1 (positions 32-35), so that the online softmax
        # must rescale row_sum when it encounters the larger block_max in block 1.
        sq = 4
        query_mla = paddle.randn([b, sq, np_, hn]).astype("bfloat16").detach()
        topk_indices_case1 = paddle.full([b, sq, topk], -1, dtype="int32")
        for t in range(sq):
            topk_indices_case1[0, t, 0:4] = paddle.to_tensor(
                [0, 1, 2, 3], dtype="int32"
            )
            topk_indices_case1[0, t, 32:36] = paddle.to_tensor(
                [4, 5, 6, 7], dtype="int32"
            )
        topk_indices_case1 = topk_indices_case1.contiguous()
        tl_out = csa_attn_target_reducesum(
            query_mla, key_comp_mla, topk_indices_case1, softmax_scale
        )
        pd_out = _compute_attn_target_on_selected_set(
            query_mla, key_comp_mla, topk_indices_case1, softmax_scale, None
        )
        _assert_close(
            tl_out,
            pd_out,
            rtol=5e-3,
            atol=2e-3,
            msg="cross-block max shift mismatch",
        )
        # L1 check
        row_sums = tl_out.sum(axis=-1)
        _assert_close(
            row_sums,
            paddle.ones_like(row_sums),
            rtol=1e-4,
            atol=1e-4,
            msg="L1 norm violated (case 1)",
        )

        # --- Case 2: all-invalid row ---
        sq = 2
        query_mla2 = paddle.randn([b, sq, np_, hn]).astype("bfloat16").detach()
        # Row 0: all invalid; Row 1: has valid entries
        topk_indices_case2 = paddle.full([b, sq, topk], -1, dtype="int32")
        topk_indices_case2[0, 1, 0] = 5
        topk_indices_case2[0, 1, 1] = 6
        topk_indices_case2 = topk_indices_case2.contiguous()
        tl_out2 = csa_attn_target_reducesum(
            query_mla2, key_comp_mla, topk_indices_case2, softmax_scale
        )
        # Row 0 must be all-zero
        self.assertTrue(
            (tl_out2[0, 0].cast("float32").abs() < 1e-6).all().item(),
            "all-invalid row should be zero",
        )
        # Row 1 must sum to 1
        row1_sum = tl_out2[0, 1].sum().item()
        self.assertAlmostEqual(
            row1_sum, 1.0, places=4, msg=f"single-row L1 violated: {row1_sum}"
        )

        # --- Case 3: single valid entry → must be exactly 1.0 ---
        sq = 2
        query_mla3 = paddle.randn([b, sq, np_, hn]).astype("bfloat16").detach()
        topk_indices_case3 = paddle.full([b, sq, topk], -1, dtype="int32")
        topk_indices_case3[0, 0, 0] = 3  # only 1 valid per row
        topk_indices_case3[0, 1, 0] = 7
        topk_indices_case3 = topk_indices_case3.contiguous()
        tl_out3 = csa_attn_target_reducesum(
            query_mla3, key_comp_mla, topk_indices_case3, softmax_scale
        )
        # The single valid slot must have probability = 1.0
        self.assertAlmostEqual(
            tl_out3[0, 0, 0].item(),
            1.0,
            places=4,
            msg="single-valid-entry prob should be 1.0",
        )
        self.assertAlmostEqual(
            tl_out3[0, 1, 0].item(),
            1.0,
            places=4,
            msg="single-valid-entry prob should be 1.0",
        )
        # All other slots must be 0
        self.assertTrue(
            (tl_out3[0, 0, 1:].cast("float32").abs() < 1e-6).all().item(),
            "non-valid slots should be zero",
        )


if __name__ == "__main__":
    unittest.main()
