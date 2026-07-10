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

"""Accuracy tests for the TileLang causal sliding-window attention (SWA).

Validates both the MQA (single shared K/V head, absorbed-MLA shape Dk=576/
Dv=512 that FA4 cannot run) and MHA (per-head K/V, Dk=Dv<=256 that FA4 can run)
SWA wrappers against the naive Paddle windowed reference, forward and backward.
The windowed mask is expressed through ``make_sliding_window_valid_range`` and
computed by the block-score kernels' ``eos - bos`` early-exit.
"""

import unittest

import numpy as np
import paddle

paddle.enable_compat(scope={"tilelang"}, silent=True)

from paddlefleet.tilelang_ops.hysparse.reference import (
    make_sliding_window_valid_range,
    ref_block_score_attn_mqa,
)
from paddlefleet.tilelang_ops.hysparse.swa_attn import (
    sliding_window_mqa_attention,
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


OUT_ATOL = 5e-2
LSE_ATOL = 1e-3
GRAD_ATOL = 8e-2


class TestSlidingWindowMQA(unittest.TestCase):
    """SWA with a single shared K/V head (absorbed-MLA MQA shape)."""

    def _run(
        self,
        S=256,
        window_size=64,
        H=8,
        D=64,
        Dv=None,
        doc_lengths=None,
        block_B=64,
    ):
        _cuda_or_skip(self)
        B = 2
        Dv = D if Dv is None else Dv
        paddle.seed(5)
        q = paddle.randn([B, S, H, D], dtype="bfloat16")
        k = paddle.randn([B, S, D], dtype="bfloat16")
        v = paddle.randn([B, S, Dv], dtype="bfloat16")
        vr = make_sliding_window_valid_range(
            S, window_size, batch=B, doc_lengths=doc_lengths
        )
        sm = D**-0.5

        qf = q.astype("float32")
        qf.stop_gradient = False
        kf = k.astype("float32")
        kf.stop_gradient = False
        vf = v.astype("float32")
        vf.stop_gradient = False
        ref_out, ref_lse, _ = ref_block_score_attn_mqa(
            qf, kf, vf, vr, sm_scale=sm, block_B=block_B
        )

        q.stop_gradient = False
        k.stop_gradient = False
        v.stop_gradient = False
        out, lse = sliding_window_mqa_attention(
            q, k, v, vr, sm_scale=sm, block_B=block_B
        )
        self.assertEqual(list(out.shape), [B, S, H, Dv])
        self.assertLess(_maxerr(out, ref_out), OUT_ATOL)
        finite = np.isfinite(ref_lse.numpy())
        lse_err = np.abs(lse.numpy()[finite] - ref_lse.numpy()[finite]).max()
        self.assertLess(float(lse_err), LSE_ATOL)

        do = paddle.randn([B, S, H, Dv], dtype="bfloat16")
        (ref_out * do.astype("float32")).sum().backward()
        (out.astype("float32") * do.astype("float32")).sum().backward()
        self.assertLess(_maxerr(q.grad, qf.grad), GRAD_ATOL)
        self.assertLess(_maxerr(k.grad, kf.grad), GRAD_ATOL)
        self.assertLess(_maxerr(v.grad, vf.grad), GRAD_ATOL)

    def test_causal_window(self):
        self._run()

    def test_window_document(self):
        self._run(doc_lengths=[96, 160])

    def test_small_window(self):
        self._run(window_size=32)

    def test_window_ge_seqlen(self):
        # window >= S degenerates to full causal attention.
        self._run(S=128, window_size=256)

    def test_window_size_one(self):
        # W=1: every query attends to exactly itself (bos == eos-1).
        self._run(window_size=1)

    def test_unaligned_seqlen(self):
        # S not a multiple of block_B exercises the K/V padding path.
        self._run(S=200, window_size=48)

    def test_block_b_32(self):
        self._run(window_size=48, block_B=32)

    def test_block_b_128(self):
        self._run(S=384, window_size=100, block_B=128)

    def test_absorbed_mla_shape(self):
        # the FA4 gap: Dk=576 > 256, Dv=512. Grad checked by cosine.
        _cuda_or_skip(self)
        B, S, H, D, Dv = 1, 192, 4, 576, 512
        window_size = 64
        paddle.seed(7)
        q = paddle.randn([B, S, H, D], dtype="bfloat16")
        k = paddle.randn([B, S, D], dtype="bfloat16")
        v = paddle.randn([B, S, Dv], dtype="bfloat16")
        vr = make_sliding_window_valid_range(S, window_size, batch=B)
        sm = D**-0.5

        qf = q.astype("float32")
        qf.stop_gradient = False
        kf = k.astype("float32")
        kf.stop_gradient = False
        vf = v.astype("float32")
        vf.stop_gradient = False
        ref_out, ref_lse, _ = ref_block_score_attn_mqa(
            qf, kf, vf, vr, sm_scale=sm, block_B=64
        )

        q.stop_gradient = False
        k.stop_gradient = False
        v.stop_gradient = False
        out, lse = sliding_window_mqa_attention(q, k, v, vr, sm_scale=sm)
        self.assertLess(_maxerr(out, ref_out), OUT_ATOL)
        finite = np.isfinite(ref_lse.numpy())
        lse_err = np.abs(lse.numpy()[finite] - ref_lse.numpy()[finite]).max()
        self.assertLess(float(lse_err), LSE_ATOL)

        do = paddle.randn([B, S, H, Dv], dtype="bfloat16")
        (ref_out * do.astype("float32")).sum().backward()
        (out.astype("float32") * do.astype("float32")).sum().backward()
        self.assertGreater(_cos(q.grad, qf.grad), 0.999)
        self.assertGreater(_cos(k.grad, kf.grad), 0.999)
        self.assertGreater(_cos(v.grad, vf.grad), 0.999)


if __name__ == "__main__":
    unittest.main()
