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

"""Accuracy tests for the MHA (per-head K/V) block-score TileLang operator.

Validates the decompressed-MLA block-score attention (each query head has its
own K/V head) against the naive Paddle reference (散算子), forward and backward.
Kept separate from the MQA suite (test_hysparse_block_attn.py) so the existing
MQA cases are untouched. The coverage mirrors ``TestBlockScoreMQA`` (block_B
variants, single head, block_N sub-tiling invariance, packed deep documents,
empty rows, batched backward, asymmetric Dk!=Dv, MLA-shaped D=256/H=64).

Unlike the MQA op, MHA uses **absolute** block coordinates, so packed-document
== per-document standalone equivalence does NOT hold for the emitted block
scores; masking correctness is still validated through ``valid_range``.
"""

import unittest

import numpy as np
import paddle

paddle.enable_compat(scope={"tilelang"}, silent=True)

from paddlefleet.tilelang_ops.hysparse.block_score_attn import (
    block_scores_from_logit,
)
from paddlefleet.tilelang_ops.hysparse.block_score_attn_mha import (
    block_score_mha_attn_fwd,
)
from paddlefleet.tilelang_ops.hysparse.block_score_attn_mha_bwd import (
    block_score_mha_bwd_interface,
)
from paddlefleet.tilelang_ops.hysparse.reference import (
    make_causal_valid_range,
    ref_block_score_attn_mha,
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


def _rand_qkv_mha(b, s, h, dk, dv, seed=0):
    """Per-head K/V: Q [B,S,H,Dk]; K [B,S_kv,H,Dk]; V [B,S_kv,H,Dv]."""
    paddle.seed(seed)
    q = paddle.randn([b, s, h, dk], dtype="bfloat16")
    k = paddle.randn([b, s, h, dk], dtype="bfloat16")
    v = paddle.randn([b, s, h, dv], dtype="bfloat16")
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


class TestBlockScoreMHA(unittest.TestCase):
    """Per-head K/V full attention + block-max scores, fwd & bwd.

    Mirrors ``TestBlockScoreMQA`` (test_hysparse_block_attn.py) case-for-case.
    """

    def _run(self, doc_lengths=None, block_B=64, H=8, D=64, Dv=None):
        _cuda_or_skip(self)
        B, S = 2, 256
        Dv = D if Dv is None else Dv
        q, k, v = _rand_qkv_mha(B, S, H, D, Dv, seed=1)
        vr = make_causal_valid_range(S, batch=B, doc_lengths=doc_lengths)
        sm_scale = D**-0.5

        qf, kf, vf = _grad_ref_qkv(q, k, v)
        ref_out, ref_lse, ref_sblk = ref_block_score_attn_mha(
            qf, kf, vf, vr, sm_scale=sm_scale, block_B=block_B
        )

        out, lse, block_logit = block_score_mha_attn_fwd(
            q, k, v, vr, sm_scale=sm_scale, block_B=block_B
        )
        sblk = block_scores_from_logit(block_logit, lse, sm_scale)

        self.assertEqual(list(out.shape), [B, S, H, Dv])
        self.assertLess(_maxerr(out, ref_out), OUT_ATOL)
        self.assertLess(_maxerr(lse, ref_lse), LSE_ATOL)
        self.assertLess(_maxerr(sblk, ref_sblk), 1e-3)

        do = paddle.randn([B, S, H, Dv], dtype="bfloat16")
        (ref_out * do.astype("float32")).sum().backward()
        dq, dk, dv = block_score_mha_bwd_interface(
            q, k, v, out, do, lse, vr, sm_scale=sm_scale, block_B=block_B
        )
        self.assertEqual(list(dk.shape), [B, S, H, D])
        self.assertEqual(list(dv.shape), [B, S, H, Dv])
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
        # H=1: one program per (query-tile, head, batch) with a single head.
        self._run(doc_lengths=None, H=1)

    def test_backward_block_n_invariance(self):
        """The MHA block-score backward sub-tiles the key dim by ``block_N``;
        any valid ``block_N`` dividing ``block_B`` must give the SAME gradients
        (only the fp-accumulation / per-sub-tile recast order changes).
        """
        _cuda_or_skip(self)
        B, S, H, D, block_B = 2, 256, 8, 64, 64
        q, k, v = _rand_qkv_mha(B, S, H, D, D, seed=13)
        vr = make_causal_valid_range(S, batch=B, doc_lengths=[96, 160])
        sm = D**-0.5

        qf, kf, vf = _grad_ref_qkv(q, k, v)
        ref_out, _, _ = ref_block_score_attn_mha(
            qf, kf, vf, vr, sm_scale=sm, block_B=block_B
        )
        out, lse, _ = block_score_mha_attn_fwd(
            q, k, v, vr, sm_scale=sm, block_B=block_B
        )
        do = paddle.randn([B, S, H, D], dtype="bfloat16")
        (ref_out * do.astype("float32")).sum().backward()

        dq1, dk1, dv1 = block_score_mha_bwd_interface(
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
        dq2, dk2, dv2 = block_score_mha_bwd_interface(
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
        self.assertLess(_maxerr(dq1, dq2), 1e-2)
        self.assertLess(_maxerr(dk1, dk2), 1e-2)
        self.assertLess(_maxerr(dv1, dv2), 1e-2)
        for dq, dk, dv in ((dq1, dk1, dv1), (dq2, dk2, dv2)):
            self.assertLess(_maxerr(dq, qf.grad), GRAD_ATOL)
            self.assertLess(_maxerr(dk, kf.grad), GRAD_ATOL)
            self.assertLess(_maxerr(dv, vf.grad), GRAD_ATOL)

    def test_backward_packed_deep_docs(self):
        """Backward with several packed documents whose later tiles start deep
        into the sequence (bos >> 0). Exercises the document-tight key-block
        window (leading-block skip) in the MHA block-score backward.
        """
        _cuda_or_skip(self)
        B, H, D, block_B = 1, 8, 64, 64
        doc_lengths = [130, 200, 180]
        S = sum(doc_lengths)
        q, k, v = _rand_qkv_mha(B, S, H, D, D, seed=17)
        vr = make_causal_valid_range(S, batch=B, doc_lengths=doc_lengths)
        sm_scale = D**-0.5

        qf, kf, vf = _grad_ref_qkv(q, k, v)
        ref_out, _, _ = ref_block_score_attn_mha(
            qf, kf, vf, vr, sm_scale=sm_scale, block_B=block_B
        )
        out, lse, _ = block_score_mha_attn_fwd(
            q, k, v, vr, sm_scale=sm_scale, block_B=block_B
        )
        do = paddle.randn([B, S, H, D], dtype="bfloat16")
        (ref_out * do.astype("float32")).sum().backward()
        dq, dk, dv = block_score_mha_bwd_interface(
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
        q, k, v = _rand_qkv_mha(B, S, H, D, D, seed=31)
        vr = make_causal_valid_range(S, batch=B).clone()
        empty = 40
        vr[:, empty, 0] = empty  # bos == eos -> empty half-open range
        vr[:, empty, 1] = empty
        vr = vr.contiguous()
        sm = D**-0.5

        qf, kf, vf = _grad_ref_qkv(q, k, v)
        ref_out, ref_lse, _ = ref_block_score_attn_mha(
            qf, kf, vf, vr, sm_scale=sm, block_B=block_B
        )
        out, lse, _ = block_score_mha_attn_fwd(
            q, k, v, vr, sm_scale=sm, block_B=block_B
        )
        self.assertLess(float(out[:, empty].abs().max()), 1e-6)
        self.assertTrue(bool((lse[:, empty] == float("-inf")).all().item()))
        self.assertLess(_maxerr(out, ref_out), OUT_ATOL)

        do = paddle.randn([B, S, H, D], dtype="bfloat16")
        (ref_out * do.astype("float32")).sum().backward()
        dq, dk, dv = block_score_mha_bwd_interface(
            q, k, v, out, do, lse, vr, sm_scale=sm, block_B=block_B
        )
        self.assertLess(float(dq[:, empty].abs().max()), 1e-6)
        self.assertLess(_maxerr(dq, qf.grad), GRAD_ATOL)
        self.assertLess(_maxerr(dk, kf.grad), GRAD_ATOL)
        self.assertLess(_maxerr(dv, vf.grad), GRAD_ATOL)

    def test_batched_backward(self):
        # dq/dk/dv for a B=2 batch (mixed doc layout) match the fp32 reference.
        self._run(doc_lengths=[96, 160])

    def test_asymmetric_dk_dv(self):
        # Dk != Dv (query/key dim 128, value/output dim 64): exercises the
        # kernel's separate D_v path in fwd + bwd.
        self._run(doc_lengths=None, D=128, Dv=64)

    def test_fwd_bwd_dv256_h64(self):
        # target decompressed-MLA shape: D=256, H=64. Gradients checked by
        # cosine (magnitudes grow at large H).
        _cuda_or_skip(self)
        B, S, H, D = 1, 192, 64, 256
        block_B = 64
        q, k, v = _rand_qkv_mha(B, S, H, D, D, seed=7)
        vr = make_causal_valid_range(S, batch=B)
        sm_scale = D**-0.5

        qf, kf, vf = _grad_ref_qkv(q, k, v)
        ref_out, ref_lse, ref_sblk = ref_block_score_attn_mha(
            qf, kf, vf, vr, sm_scale=sm_scale, block_B=block_B
        )
        out, lse, block_logit = block_score_mha_attn_fwd(
            q, k, v, vr, sm_scale=sm_scale, block_B=block_B
        )
        sblk = block_scores_from_logit(block_logit, lse, sm_scale)
        self.assertLess(_maxerr(out, ref_out), OUT_ATOL)
        self.assertLess(_maxerr(lse, ref_lse), LSE_ATOL)
        self.assertLess(_maxerr(sblk, ref_sblk), 1e-3)

        do = paddle.randn([B, S, H, D], dtype="bfloat16")
        (ref_out * do.astype("float32")).sum().backward()
        dq, dk, dv = block_score_mha_bwd_interface(
            q, k, v, out, do, lse, vr, sm_scale=sm_scale, block_B=block_B
        )
        self.assertGreater(_cos(dq, qf.grad), 0.999)
        self.assertGreater(_cos(dk, kf.grad), 0.999)
        self.assertGreater(_cos(dv, vf.grad), 0.999)


if __name__ == "__main__":
    unittest.main()
