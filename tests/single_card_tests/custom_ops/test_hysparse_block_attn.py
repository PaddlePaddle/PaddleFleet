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

"""Accuracy tests for the HySparse (MQA) block-attention TileLang operators.

Validates, against the naive Paddle reference (散算子):
* Block-score attention (Algorithm 1, MQA): full attention with a single shared
  K/V head + block-max scores, fwd & bwd.
* Block-sparse attention (MQA gather): per-query block-sparse attn over a single
  shared K/V head, fwd & bwd.
* The end-to-end pipeline (block-score -> TopK -> block-sparse).

Both causal and causal+document masking are covered via ``valid_range``.
"""

import unittest

import numpy as np
import paddle

paddle.enable_compat(scope={"tilelang"}, silent=True)

from paddlefleet.tilelang_ops.hysparse.block_score_attn import (
    block_score_mqa_attn_fwd,
    block_scores_from_logit,
)
from paddlefleet.tilelang_ops.hysparse.block_score_attn_bwd import (
    block_score_mqa_bwd_interface,
)
from paddlefleet.tilelang_ops.hysparse.block_sparse_attn_mqa import (
    block_sparse_mqa_attn_fwd,
)
from paddlefleet.tilelang_ops.hysparse.block_sparse_attn_mqa_bwd import (
    block_sparse_mqa_bwd_interface,
)
from paddlefleet.tilelang_ops.hysparse.pipeline import (
    hysparse_forward_mqa,
)
from paddlefleet.tilelang_ops.hysparse.reference import (
    make_causal_valid_range,
    ref_block_score_attn_mqa,
    ref_block_sparse_attn_mqa,
)


def _cuda_or_skip(testcase):
    if not paddle.device.is_compiled_with_cuda():
        testcase.skipTest("CUDA build of Paddle is required")
    if paddle.device.cuda.device_count() == 0:
        testcase.skipTest("No CUDA device available")


def _maxerr(a, b):
    a = a.astype("float32").numpy()
    b = b.astype("float32").numpy()
    return float(np.abs(a - b).max())


def _cos(a, b):
    a = a.astype("float32").numpy().ravel()
    b = b.astype("float32").numpy().ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float((a * b).sum() / denom)


def _rand_qkv_mqa(b, s, h, d, seed=0):
    """Query with H heads; K/V a single shared head [B, S, D]."""
    paddle.seed(seed)
    q = paddle.randn([b, s, h, d], dtype="bfloat16")
    k = paddle.randn([b, s, d], dtype="bfloat16")
    v = paddle.randn([b, s, d], dtype="bfloat16")
    return q, k, v


def _grad_ref_qkv(q, k, v):
    """float32 leaf copies of q/k/v for autograd reference gradients."""
    qf = q.astype("float32")
    qf.stop_gradient = False
    kf = k.astype("float32")
    kf.stop_gradient = False
    vf = v.astype("float32")
    vf.stop_gradient = False
    return qf, kf, vf


# tolerances: bf16 matmul + fp32 accumulation
OUT_ATOL = 5e-2
LSE_ATOL = 1e-3
GRAD_ATOL = 8e-2


class TestBlockScoreMQA(unittest.TestCase):
    """Block-score attention (MQA): shared-K/V full attn + block-max scores."""

    def _run(self, doc_lengths=None, block_B=64, H=8, D=64):
        _cuda_or_skip(self)
        B, S = 2, 256
        q, k, v = _rand_qkv_mqa(B, S, H, D, seed=1)
        vr = make_causal_valid_range(S, batch=B, doc_lengths=doc_lengths)
        sm_scale = D**-0.5

        qf, kf, vf = _grad_ref_qkv(q, k, v)
        ref_out, ref_lse, ref_sblk = ref_block_score_attn_mqa(
            qf, kf, vf, vr, sm_scale=sm_scale, block_B=block_B
        )

        out, lse, block_logit = block_score_mqa_attn_fwd(
            q, k, v, vr, sm_scale=sm_scale, block_B=block_B
        )
        sblk = block_scores_from_logit(block_logit, lse, sm_scale)

        self.assertLess(_maxerr(out, ref_out), OUT_ATOL)
        self.assertLess(_maxerr(lse, ref_lse), LSE_ATOL)
        self.assertLess(_maxerr(sblk, ref_sblk), 1e-3)

        # backward
        do = paddle.randn([B, S, H, D], dtype="bfloat16")
        (ref_out * do.astype("float32")).sum().backward()
        dq, dk, dv = block_score_mqa_bwd_interface(
            q, k, v, out, do, lse, vr, sm_scale=sm_scale, block_B=block_B
        )
        self.assertLess(_maxerr(dq, qf.grad), GRAD_ATOL)
        self.assertLess(_maxerr(dk, kf.grad), GRAD_ATOL)
        self.assertLess(_maxerr(dv, vf.grad), GRAD_ATOL)

    def test_causal(self):
        self._run(doc_lengths=None)

    def test_causal_document(self):
        self._run(doc_lengths=[96, 160])

    def test_block_b_32(self):
        # smaller key block size (must still divide the auto-picked block_N).
        self._run(doc_lengths=[96, 160], block_B=32)

    def test_block_b_128(self):
        # larger key block size exercises a different tiling / mask layout.
        self._run(doc_lengths=None, block_B=128)

    def test_single_query_head(self):
        # H=1: heads-on-M padding edge (PH = max(pow2(1), 16) = 16).
        self._run(doc_lengths=None, H=1)

    def test_backward_block_n_invariance(self):
        """The block-score backward sub-tiles the key dim by ``block_N``; any
        valid ``block_N`` dividing ``block_B`` must give the SAME gradients
        (only the fp-accumulation / per-sub-tile bf16 recast order changes).
        Validates the sub-tiling introduced for the D=576 MB=64 backward.
        """
        _cuda_or_skip(self)
        B, S, H, D, block_B = 2, 256, 8, 64, 64
        q, k, v = _rand_qkv_mqa(B, S, H, D, seed=13)
        vr = make_causal_valid_range(S, batch=B, doc_lengths=[96, 160])
        sm = D**-0.5

        qf, kf, vf = _grad_ref_qkv(q, k, v)
        ref_out, _, _ = ref_block_score_attn_mqa(
            qf, kf, vf, vr, sm_scale=sm, block_B=block_B
        )
        out, lse, _ = block_score_mqa_attn_fwd(
            q, k, v, vr, sm_scale=sm, block_B=block_B
        )
        do = paddle.randn([B, S, H, D], dtype="bfloat16")
        (ref_out * do.astype("float32")).sum().backward()

        # ratio=1 (block_N == block_B) vs ratio=2 (block_N = block_B // 2)
        dq1, dk1, dv1 = block_score_mqa_bwd_interface(
            q,
            k,
            v,
            out,
            do,
            lse,
            vr,
            sm_scale=sm,
            block_B=block_B,
            block_M=64,
            block_N=64,
        )
        dq2, dk2, dv2 = block_score_mqa_bwd_interface(
            q,
            k,
            v,
            out,
            do,
            lse,
            vr,
            sm_scale=sm,
            block_B=block_B,
            block_M=64,
            block_N=32,
        )
        # the two tilings must agree tightly (differences only from fp order)
        self.assertLess(_maxerr(dq1, dq2), 1e-2)
        self.assertLess(_maxerr(dk1, dk2), 1e-2)
        self.assertLess(_maxerr(dv1, dv2), 1e-2)
        # and both must match the fp32 reference
        for dq, dk, dv in ((dq1, dk1, dv1), (dq2, dk2, dv2)):
            self.assertLess(_maxerr(dq, qf.grad), GRAD_ATOL)
            self.assertLess(_maxerr(dk, kf.grad), GRAD_ATOL)
            self.assertLess(_maxerr(dv, vf.grad), GRAD_ATOL)

    def test_backward_packed_deep_docs(self):
        """Backward with several packed documents whose later tiles start deep
        into the sequence (bos >> 0). Exercises the document-tight key-block
        window (leading-block skip) in the block-score backward: skipped blocks
        must contribute nothing so dq/dk/dv still match the fp32 reference.
        """
        _cuda_or_skip(self)
        B, H, D, block_B = 1, 8, 64, 64
        # deep docs: three documents, each many blocks long and not aligned to
        # block_B, so per-tile min(bos) lands well past column 0.
        doc_lengths = [130, 200, 180]
        S = sum(doc_lengths)
        q, k, v = _rand_qkv_mqa(B, S, H, D, seed=17)
        vr = make_causal_valid_range(S, batch=B, doc_lengths=doc_lengths)
        sm_scale = D**-0.5

        qf, kf, vf = _grad_ref_qkv(q, k, v)
        ref_out, _, _ = ref_block_score_attn_mqa(
            qf, kf, vf, vr, sm_scale=sm_scale, block_B=block_B
        )
        out, lse, _ = block_score_mqa_attn_fwd(
            q, k, v, vr, sm_scale=sm_scale, block_B=block_B
        )
        do = paddle.randn([B, S, H, D], dtype="bfloat16")
        (ref_out * do.astype("float32")).sum().backward()
        dq, dk, dv = block_score_mqa_bwd_interface(
            q, k, v, out, do, lse, vr, sm_scale=sm_scale, block_B=block_B
        )
        self.assertLess(_maxerr(dq, qf.grad), GRAD_ATOL)
        self.assertLess(_maxerr(dk, kf.grad), GRAD_ATOL)
        self.assertLess(_maxerr(dv, vf.grad), GRAD_ATOL)

    def test_empty_rows(self):
        """A query row with no valid key (bos == eos) must give 0 output,
        -inf lse, and 0 gradient -- matching the guarded reference.
        """
        _cuda_or_skip(self)
        B, S, H, D, block_B = 1, 128, 8, 64, 64
        q, k, v = _rand_qkv_mqa(B, S, H, D, seed=31)
        vr = make_causal_valid_range(S, batch=B).clone()
        empty = 40
        vr[:, empty, 0] = empty  # bos == eos -> empty half-open range
        vr[:, empty, 1] = empty
        vr = vr.contiguous()
        sm = D**-0.5

        qf, kf, vf = _grad_ref_qkv(q, k, v)
        ref_out, ref_lse, _ = ref_block_score_attn_mqa(
            qf, kf, vf, vr, sm_scale=sm, block_B=block_B
        )
        out, lse, _ = block_score_mqa_attn_fwd(
            q, k, v, vr, sm_scale=sm, block_B=block_B
        )
        # empty row: zero output and -inf lse
        self.assertLess(float(out[:, empty].abs().max()), 1e-6)
        self.assertTrue(bool((lse[:, empty] == float("-inf")).all().item()))
        # non-empty rows still match the reference
        self.assertLess(_maxerr(out, ref_out), OUT_ATOL)

        do = paddle.randn([B, S, H, D], dtype="bfloat16")
        (ref_out * do.astype("float32")).sum().backward()
        dq, dk, dv = block_score_mqa_bwd_interface(
            q, k, v, out, do, lse, vr, sm_scale=sm, block_B=block_B
        )
        self.assertLess(float(dq[:, empty].abs().max()), 1e-6)
        self.assertLess(_maxerr(dq, qf.grad), GRAD_ATOL)
        self.assertLess(_maxerr(dk, kf.grad), GRAD_ATOL)
        self.assertLess(_maxerr(dv, vf.grad), GRAD_ATOL)


class TestBlockSparseMQA(unittest.TestCase):
    """Block-sparse attention (MQA gather): gathers only selected blocks."""

    def _make_indices(self, vr, block_B, nsel, seed=0):
        """Random **document-relative** block ids per query token.

        Relative block ``j`` of a token spans key columns
        ``[bos + j*block_B, bos + (j+1)*block_B)``; valid blocks are
        ``0 .. ceil((eos-bos)/block_B) - 1``.
        """
        rng = np.random.default_rng(seed)
        vr_np = vr.numpy()
        B, S, _ = vr_np.shape
        idx = np.full([B, S, nsel], -1, dtype=np.int32)
        for b in range(B):
            for t in range(S):
                bos = int(vr_np[b, t, 0])
                eos = int(vr_np[b, t, 1])
                n = (eos - bos + block_B - 1) // block_B
                cand = list(range(n))
                chosen = (
                    cand
                    if len(cand) <= nsel
                    else list(rng.choice(cand, size=nsel, replace=False))
                )
                for jj, blk in enumerate(chosen):
                    idx[b, t, jj] = blk
        return paddle.to_tensor(idx, dtype="int32")

    def _run(self, S=256, doc_lengths=None, block_B=64):
        _cuda_or_skip(self)
        B, H, D = 2, 8, 64
        nsel = 3
        q, k, v = _rand_qkv_mqa(B, S, H, D, seed=5)
        vr = make_causal_valid_range(S, batch=B, doc_lengths=doc_lengths)
        idx = self._make_indices(vr, block_B, nsel)
        sm_scale = D**-0.5

        qf, kf, vf = _grad_ref_qkv(q, k, v)
        ref_out, ref_lse = ref_block_sparse_attn_mqa(
            qf, kf, vf, idx, vr, sm_scale=sm_scale, block_B=block_B
        )
        out, lse = block_sparse_mqa_attn_fwd(
            q, k, v, idx, vr, sm_scale=sm_scale, block_B=block_B
        )
        self.assertLess(_maxerr(out, ref_out), OUT_ATOL)
        finite = np.isfinite(ref_lse.numpy())
        lse_err = np.abs(lse.numpy()[finite] - ref_lse.numpy()[finite]).max()
        self.assertLess(float(lse_err), LSE_ATOL)

        do = paddle.randn([B, S, H, D], dtype="bfloat16")
        (ref_out * do.astype("float32")).sum().backward()
        dq, dk, dv = block_sparse_mqa_bwd_interface(
            q, k, v, out, do, lse, idx, vr, sm_scale=sm_scale, block_B=block_B
        )
        self.assertLess(_maxerr(dq, qf.grad), GRAD_ATOL)
        self.assertLess(_maxerr(dk, kf.grad), GRAD_ATOL)
        self.assertLess(_maxerr(dv, vf.grad), GRAD_ATOL)

    def test_causal(self):
        self._run(doc_lengths=None)

    def test_causal_document(self):
        self._run(doc_lengths=[96, 160])

    def test_unaligned_seqlen(self):
        # S_kv not a multiple of block_B exercises the K/V padding path.
        self._run(S=200, doc_lengths=None)

    def test_block_b_32(self):
        self._run(doc_lengths=[96, 160], block_B=32)

    def test_block_b_128(self):
        self._run(S=384, doc_lengths=None, block_B=128)

    def test_all_blocks_equals_full(self):
        """Selecting every block must reproduce full-range attention (block-score)."""
        _cuda_or_skip(self)
        B, S, H, D = 2, 256, 8, 64
        block_B = 64
        num_blocks = (S + block_B - 1) // block_B
        q, k, v = _rand_qkv_mqa(B, S, H, D, seed=3)
        vr = make_causal_valid_range(S, batch=B)
        sm_scale = D**-0.5

        idx_all = (
            paddle.arange(num_blocks, dtype="int32")
            .reshape([1, 1, num_blocks])
            .expand([B, S, num_blocks])
            .contiguous()
        )
        out_a, _ = block_sparse_mqa_attn_fwd(
            q, k, v, idx_all, vr, sm_scale=sm_scale, block_B=block_B
        )
        out_b, _, _ = block_score_mqa_attn_fwd(
            q, k, v, vr, sm_scale=sm_scale, block_B=block_B
        )
        self.assertLess(_maxerr(out_a, out_b), 1e-3)


class TestPipelineMQA(unittest.TestCase):
    """End-to-end MQA block-score -> TopK -> block-sparse pipeline."""

    def test_pipeline_matches_reference(self):
        _cuda_or_skip(self)
        B, S, H, D = 2, 256, 8, 64
        block_B = 64
        topk = 3
        q, k, v = _rand_qkv_mqa(B, S, H, D, seed=6)
        vr = make_causal_valid_range(S, batch=B)
        sm_scale = D**-0.5

        sparse_out, sparse_lse, indices, full_out, full_lse = (
            hysparse_forward_mqa(
                q, k, v, vr, topk, sm_scale=sm_scale, block_B=block_B
            )
        )

        # full-attention branch must match the MQA block-score reference
        qf, kf, vf = _grad_ref_qkv(q, k, v)
        ref_full_out, _, _ = ref_block_score_attn_mqa(
            qf, kf, vf, vr, sm_scale=sm_scale, block_B=block_B
        )
        self.assertLess(_maxerr(full_out, ref_full_out), OUT_ATOL)

        # sparse branch must match the MQA reference given the SAME indices
        ref_sparse_out, _ = ref_block_sparse_attn_mqa(
            qf, kf, vf, indices, vr, sm_scale=sm_scale, block_B=block_B
        )
        self.assertLess(_maxerr(sparse_out, ref_sparse_out), OUT_ATOL)

        # indices are shared across heads, in range, causal
        idx_np = indices.numpy()
        qblk = (np.arange(S) // block_B).reshape([1, S, 1])
        picked = idx_np >= 0
        self.assertTrue(
            bool(
                (
                    idx_np[picked]
                    <= np.broadcast_to(qblk, idx_np.shape)[picked]
                ).all()
            )
        )

    def test_topk_exceeds_num_blocks(self):
        # topk larger than the number of key blocks must still return a stable
        # [B, S, topk] index tensor, padding the surplus slots with -1.
        _cuda_or_skip(self)
        B, S, H, D = 1, 96, 4, 64
        block_B = 64  # num_blocks = ceil(96/64) = 2
        topk = 5
        q, k, v = _rand_qkv_mqa(B, S, H, D, seed=11)
        vr = make_causal_valid_range(S, batch=B)
        _, _, indices, _, _ = hysparse_forward_mqa(
            q, k, v, vr, topk, block_B=block_B
        )
        self.assertEqual(list(indices.shape), [B, S, topk])
        # at most num_blocks real ids per row; the rest are -1 padding
        idx_np = indices.numpy()
        self.assertTrue(bool((idx_np < 2).all()))
        self.assertTrue(bool((idx_np[:, -1, :] >= -1).all()))
        self.assertTrue(bool((idx_np == -1).any()))


class TestDocEquivalenceMQA(unittest.TestCase):
    """Documents packed with a document mask must give the SAME sparse result
    as running each document standalone -- even when document lengths are not
    multiples of block_B. This is the whole point of the document-relative
    block coordinates: block selection is anchored at each document's start, so
    a packed batch and per-document batches select identical (relative) blocks.
    """

    def _run(self, doc_lengths, topk, block_B=64, H=8, D=64, seed=11):
        _cuda_or_skip(self)
        S = sum(doc_lengths)
        q, k, v = _rand_qkv_mqa(1, S, H, D, seed=seed)
        vr = make_causal_valid_range(S, batch=1, doc_lengths=doc_lengths)
        sm_scale = D**-0.5

        packed_out, _, _, _, _ = hysparse_forward_mqa(
            q, k, v, vr, topk, sm_scale=sm_scale, block_B=block_B
        )

        start = 0
        for L in doc_lengths:
            end = start + L
            q_d = q[:, start:end].contiguous()
            k_d = k[:, start:end].contiguous()
            v_d = v[:, start:end].contiguous()
            vr_d = make_causal_valid_range(L, batch=1)
            d_out, _, _, _, _ = hysparse_forward_mqa(
                q_d, k_d, v_d, vr_d, topk, sm_scale=sm_scale, block_B=block_B
            )
            err = _maxerr(packed_out[:, start:end], d_out)
            self.assertLess(err, 2e-3, f"doc [{start},{end}) err={err}")
            start = end

    def test_two_docs_select_all(self):
        # topk >= #blocks/doc -> every valid block selected in both layouts
        self._run(doc_lengths=[96, 160], topk=3)

    def test_two_docs_true_sparsity(self):
        # topk < #blocks in the long doc -> genuine subset selection must agree
        self._run(doc_lengths=[100, 200], topk=2)

    def test_three_docs(self):
        self._run(doc_lengths=[64, 96, 130], topk=2)


class TestHeadDim576MQA(unittest.TestCase):
    """MLA-shaped head_dim=576 (non-power-of-2), num_heads=64.

    Validates the ops run and stay accurate at the large MLA head dim. Output
    is checked against the fp32 reference with the usual absolute tolerance;
    gradients are checked by cosine similarity because absolute errors grow
    with the (large) gradient magnitudes at D=576.
    """

    def test_score_fwd_bwd(self):
        _cuda_or_skip(self)
        B, S, H, D = 1, 192, 64, 576
        block_B = 64
        q, k, v = _rand_qkv_mqa(B, S, H, D, seed=7)
        vr = make_causal_valid_range(S, batch=B)
        sm_scale = D**-0.5

        qf, kf, vf = _grad_ref_qkv(q, k, v)
        ref_out, ref_lse, ref_sblk = ref_block_score_attn_mqa(
            qf, kf, vf, vr, sm_scale=sm_scale, block_B=block_B
        )
        out, lse, block_logit = block_score_mqa_attn_fwd(
            q, k, v, vr, sm_scale=sm_scale, block_B=block_B
        )
        sblk = block_scores_from_logit(block_logit, lse, sm_scale)
        self.assertLess(_maxerr(out, ref_out), OUT_ATOL)
        self.assertLess(_maxerr(lse, ref_lse), LSE_ATOL)
        self.assertLess(_maxerr(sblk, ref_sblk), 1e-3)

        do = paddle.randn([B, S, H, D], dtype="bfloat16")
        (ref_out * do.astype("float32")).sum().backward()
        dq, dk, dv = block_score_mqa_bwd_interface(
            q, k, v, out, do, lse, vr, sm_scale=sm_scale, block_B=block_B
        )
        self.assertGreater(_cos(dq, qf.grad), 0.999)
        self.assertGreater(_cos(dk, kf.grad), 0.999)
        self.assertGreater(_cos(dv, vf.grad), 0.999)

    def test_sparse_fwd_bwd_and_pipeline(self):
        _cuda_or_skip(self)
        B, S, H, D = 1, 192, 64, 576
        block_B, topk = 64, 2
        q, k, v = _rand_qkv_mqa(B, S, H, D, seed=8)
        vr = make_causal_valid_range(S, batch=B)
        sm_scale = D**-0.5

        sparse_out, sparse_lse, idx, full_out, _ = hysparse_forward_mqa(
            q, k, v, vr, topk, sm_scale=sm_scale, block_B=block_B
        )
        qf, kf, vf = _grad_ref_qkv(q, k, v)
        ref_full, _, _ = ref_block_score_attn_mqa(
            qf, kf, vf, vr, sm_scale=sm_scale, block_B=block_B
        )
        self.assertLess(_maxerr(full_out, ref_full), OUT_ATOL)
        ref_sparse, _ = ref_block_sparse_attn_mqa(
            qf, kf, vf, idx, vr, sm_scale=sm_scale, block_B=block_B
        )
        self.assertLess(_maxerr(sparse_out, ref_sparse), OUT_ATOL)

        # gather backward with head-tiled M (block_H) for the large head dim
        do = paddle.randn([B, S, H, D], dtype="bfloat16")
        (ref_sparse * do.astype("float32")).sum().backward()
        dq, dk, dv = block_sparse_mqa_bwd_interface(
            q,
            k,
            v,
            sparse_out,
            do,
            sparse_lse,
            idx,
            vr,
            sm_scale=sm_scale,
            block_B=block_B,
        )
        self.assertGreater(_cos(dq, qf.grad), 0.999)
        self.assertGreater(_cos(dk, kf.grad), 0.999)
        self.assertGreater(_cos(dv, vf.grad), 0.999)


class TestBenchmarkSmoke(unittest.TestCase):
    """The benchmark harness runs and yields finite latencies for every op."""

    def test_bench_config_small(self):
        _cuda_or_skip(self)
        from paddlefleet.tilelang_ops.hysparse.benchmark import bench_config

        res = bench_config(
            B=1,
            S=512,
            H=16,
            D=64,
            block_B=64,
            topk=4,
            do_bwd=True,
            iters=1,
        )
        for key in (
            "score_fwd",
            "score_bwd",
            "sparse_fwd",
            "sparse_bwd",
            "pipeline",
            "pipeline_peak_gb",
        ):
            self.assertIn(key, res)
            self.assertTrue(np.isfinite(res[key]))
            self.assertGreater(res[key], 0.0)


class TestBatchEquivalenceMQA(unittest.TestCase):
    """Batch size > 1: batch elements are fully independent, and each row may
    carry its own document layout. A packed B=2 batch must reproduce (a) each
    row run standalone as B=1, and (b) each document run standalone -- even
    when the two rows pack completely different document lengths.
    """

    def _vr_multi(self, S, layouts):
        """Stack per-row causal+document valid_ranges into [B, S, 2]."""
        rows = [
            make_causal_valid_range(S, batch=1, doc_lengths=dl)
            for dl in layouts
        ]
        return paddle.concat(rows, axis=0).contiguous()

    def test_row_isolation(self):
        # B=2 packed == two independent B=1 runs (different per-row layouts).
        _cuda_or_skip(self)
        S, H, D, block_B, topk = 256, 8, 64, 64, 2
        layouts = [[100, 156], [96, 64, 96]]
        q, k, v = _rand_qkv_mqa(2, S, H, D, seed=21)
        vr = self._vr_multi(S, layouts)
        sm = D**-0.5

        packed, _, _, _, _ = hysparse_forward_mqa(
            q, k, v, vr, topk, sm_scale=sm, block_B=block_B
        )
        for b, dl in enumerate(layouts):
            vr_b = make_causal_valid_range(S, batch=1, doc_lengths=dl)
            solo, _, _, _, _ = hysparse_forward_mqa(
                q[b : b + 1],
                k[b : b + 1],
                v[b : b + 1],
                vr_b,
                topk,
                sm_scale=sm,
                block_B=block_B,
            )
            self.assertLess(_maxerr(packed[b : b + 1], solo), 2e-3)

    def test_per_row_doc_equivalence(self):
        # Each (row, doc) slice equals that document run on its own, with the
        # two rows using different, non-block-aligned document layouts.
        _cuda_or_skip(self)
        S, H, D, block_B, topk = 256, 8, 64, 64, 2
        layouts = [[100, 156], [96, 64, 96]]
        q, k, v = _rand_qkv_mqa(2, S, H, D, seed=22)
        vr = self._vr_multi(S, layouts)
        sm = D**-0.5

        packed, _, _, _, _ = hysparse_forward_mqa(
            q, k, v, vr, topk, sm_scale=sm, block_B=block_B
        )
        for b, docs in enumerate(layouts):
            start = 0
            for L in docs:
                end = start + L
                d_out, _, _, _, _ = hysparse_forward_mqa(
                    q[b : b + 1, start:end].contiguous(),
                    k[b : b + 1, start:end].contiguous(),
                    v[b : b + 1, start:end].contiguous(),
                    make_causal_valid_range(L, batch=1),
                    topk,
                    sm_scale=sm,
                    block_B=block_B,
                )
                err = _maxerr(packed[b : b + 1, start:end], d_out)
                self.assertLess(err, 2e-3, f"row{b} [{start},{end}) err={err}")
                start = end

    def test_backward_batched(self):
        # dq/dk/dv for a B=2 batch match the fp32 reference (block-score).
        _cuda_or_skip(self)
        B, S, H, D, block_B = 2, 256, 8, 64, 64
        q, k, v = _rand_qkv_mqa(B, S, H, D, seed=23)
        vr = make_causal_valid_range(S, batch=B, doc_lengths=[96, 160])
        sm = D**-0.5

        qf, kf, vf = _grad_ref_qkv(q, k, v)
        ref_out, _, _ = ref_block_score_attn_mqa(
            qf, kf, vf, vr, sm_scale=sm, block_B=block_B
        )
        out, lse, _ = block_score_mqa_attn_fwd(
            q, k, v, vr, sm_scale=sm, block_B=block_B
        )
        do = paddle.randn([B, S, H, D], dtype="bfloat16")
        (ref_out * do.astype("float32")).sum().backward()
        dq, dk, dv = block_score_mqa_bwd_interface(
            q, k, v, out, do, lse, vr, sm_scale=sm, block_B=block_B
        )
        self.assertLess(_maxerr(dq, qf.grad), GRAD_ATOL)
        self.assertLess(_maxerr(dk, kf.grad), GRAD_ATOL)
        self.assertLess(_maxerr(dv, vf.grad), GRAD_ATOL)


if __name__ == "__main__":
    unittest.main()
