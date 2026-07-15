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

"""Backward / autograd coverage for the FA4-fused full block-score attention
(:mod:`paddlefleet.tilelang_ops.hysparse.block_score_fa4`).

The consistency test (``test_hysparse_fa4_topk_consistency``) only drives the
forward path. Here we exercise the ``_BlockScoreFA4Attn`` PyLayer *backward*
(FA4 sm100 bwd kernel) and the ``sm_scale=None`` default, verifying the
attention-output gradient against a plain dense-attention reference (the
``block_logit`` / ``lse`` outputs are non-differentiable and carry no grad).

FA4 block-score fusion runs only on SM 10.x (Blackwell); the test skips
otherwise.
"""

import math
import unittest

import paddle

paddle.enable_compat(scope={"tilelang"}, silent=True)

_NEG_INF = float("-inf")


def _sm100_or_skip(tc):
    if not paddle.device.is_compiled_with_cuda():
        tc.skipTest("CUDA build of Paddle required")
    if paddle.device.cuda.device_count() == 0:
        tc.skipTest("no CUDA device available")
    if paddle.device.cuda.get_device_capability()[0] != 10:
        tc.skipTest("FA4 block-score fusion requires SM 10.x (Blackwell)")


def _ref_causal_attn(q, k, v, sm_scale):
    """Differentiable dense causal MHA reference. q/k/v [B,S,H,D] fp32."""
    b, s, h, d = q.shape
    sk = k.shape[1]
    qf = q.transpose([0, 2, 1, 3])
    kf = k.transpose([0, 2, 1, 3])
    vf = v.transpose([0, 2, 1, 3])
    logits = paddle.matmul(qf, kf, transpose_y=True) * sm_scale  # [B,H,S,Sk]
    row = paddle.arange(s).reshape([s, 1])
    col = paddle.arange(sk).reshape([1, sk])
    masked = (col > row + (sk - s)).reshape([1, 1, s, sk])
    logits = paddle.where(masked, paddle.full_like(logits, _NEG_INF), logits)
    p = paddle.nn.functional.softmax(logits, axis=-1)
    out = paddle.matmul(p, vf)  # [B,H,S,D]
    return out.transpose([0, 2, 1, 3])  # [B,S,H,D]


def _cos(a, b):
    import numpy as np

    af = a.astype("float32").numpy().reshape(-1)
    bf = b.astype("float32").numpy().reshape(-1)
    denom = (np.linalg.norm(af) * np.linalg.norm(bf)) + 1e-12
    return float(np.dot(af, bf) / denom)


class TestBlockScoreFA4Backward(unittest.TestCase):
    def test_backward_matches_dense_reference(self):
        _sm100_or_skip(self)
        from paddlefleet.tilelang_ops.hysparse import block_score_fa4_attn_fwd

        b, s, h, d = 1, 256, 8, 64
        paddle.seed(2026)
        q = paddle.randn([b, s, h, d], dtype="bfloat16")
        k = paddle.randn([b, s, h, d], dtype="bfloat16")
        v = paddle.randn([b, s, h, d], dtype="bfloat16")
        sm_scale = 1.0 / math.sqrt(d)

        qf = q.detach()
        kf = k.detach()
        vf = v.detach()
        qf.stop_gradient = False
        kf.stop_gradient = False
        vf.stop_gradient = False
        out, lse, block_logit = block_score_fa4_attn_fwd(
            qf, kf, vf, sm_scale=sm_scale, block_B=64, causal=True
        )
        g = paddle.randn(out.shape, dtype="float32").astype("bfloat16")
        out.backward(g)

        qr = q.astype("float32").detach()
        kr = k.astype("float32").detach()
        vr = v.astype("float32").detach()
        qr.stop_gradient = False
        kr.stop_gradient = False
        vr.stop_gradient = False
        ref = _ref_causal_attn(qr, kr, vr, sm_scale)
        ref.backward(g.astype("float32"))

        self.assertGreater(_cos(qf.grad, qr.grad), 0.99)
        self.assertGreater(_cos(kf.grad, kr.grad), 0.99)
        self.assertGreater(_cos(vf.grad, vr.grad), 0.99)

    def test_default_sm_scale(self):
        _sm100_or_skip(self)
        from paddlefleet.tilelang_ops.hysparse import block_score_fa4_attn_fwd

        b, s, h, d = 1, 128, 4, 64
        paddle.seed(7)
        q = paddle.randn([b, s, h, d], dtype="bfloat16")
        k = paddle.randn([b, s, h, d], dtype="bfloat16")
        v = paddle.randn([b, s, h, d], dtype="bfloat16")
        # sm_scale=None -> defaults to d ** -0.5.
        out_default, _, _ = block_score_fa4_attn_fwd(
            q, k, v, block_B=64, causal=True
        )
        out_explicit, _, _ = block_score_fa4_attn_fwd(
            q, k, v, sm_scale=d**-0.5, block_B=64, causal=True
        )
        self.assertEqual(list(out_default.shape), [b, s, h, d])
        self.assertGreater(_cos(out_default, out_explicit), 0.999)


if __name__ == "__main__":
    unittest.main()
