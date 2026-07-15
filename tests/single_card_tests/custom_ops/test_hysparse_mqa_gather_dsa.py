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

"""Correctness tests for the HySparse block-sparse *DSA* backend
(:func:`paddlefleet.cudnn_ops.block_sparse_mqa_attention_dsa`), which routes the
block selection through DeepSeek-v4's FlashMLA sparse forward + cuDNN DSA
backward.

We validate the absorbed-MQA layout (Dk=576 query/key, Dv=512 value == leading
slice of the shared latent) against a differentiable dense masked-softmax
reference over the exact selected-block column set (forward output and dq/dkv
grads). The comparison requires cosine > 0.99. The suite skips gracefully when
FlashMLA / cuDNN-frontend / a Blackwell (SM100) GPU is unavailable.
"""

import math
import unittest

import paddle

paddle.enable_compat(scope={"tilelang"}, silent=True)


def _dsa_unavailable_reason():
    if not paddle.device.is_compiled_with_cuda():
        return "CUDA build of Paddle required"
    if paddle.device.cuda.device_count() == 0:
        return "no CUDA device available"
    try:
        from paddlefleet.cudnn_ops import is_dsa_available

        if not is_dsa_available():
            return "FlashMLA sparse fwd + cuDNN DSA bwd not available"
    except (ImportError, RuntimeError):
        return "hysparse DSA import failed"
    cc = paddle.device.cuda.get_device_capability()
    if cc[0] < 10:
        return f"DSA sparse fwd requires SM100+, got {cc}"
    return None


_SKIP_REASON = None


def _skip_if_no_dsa(tc):
    global _SKIP_REASON
    if _SKIP_REASON is None:
        _SKIP_REASON = _dsa_unavailable_reason() or ""
    if _SKIP_REASON:
        tc.skipTest(_SKIP_REASON)


def _allow_mask(indices, valid_range, s_kv, block_B):
    """Bool [B, S, S_kv]: col allowed iff in a selected block ∩ [bos, eos)."""
    import numpy as np

    idx = indices.numpy()
    vr = valid_range.numpy()
    b, s, _ = idx.shape
    allow = np.zeros([b, s, s_kv], dtype=bool)
    for bi in range(b):
        for i in range(s):
            bos, eos = int(vr[bi, i, 0]), int(vr[bi, i, 1])
            for blk in idx[bi, i]:
                if blk < 0:
                    continue
                c0 = bos + int(blk) * block_B
                for col in range(c0, min(c0 + block_B, eos)):
                    if 0 <= col < s_kv:
                        allow[bi, i, col] = True
    return paddle.to_tensor(allow)


def _ref_masked_attn(q, k, v, allow, sm_scale, attn_sink=None):
    """Differentiable dense masked MQA attention reference (fp32).

    ``attn_sink`` [H] fp32 (optional) adds a virtual sink column to the softmax
    denominator (attention sink / off-by-one).
    """
    neg_inf = float("-inf")
    logits = paddle.einsum("bshd,bkd->bshk", q, k) * sm_scale
    neg = paddle.full_like(logits, neg_inf)
    mask = allow.unsqueeze(2)  # [B,S,1,Skv]
    logits = paddle.where(mask, logits, neg)
    row_has = allow.any(axis=-1)
    m = logits.max(axis=-1, keepdim=True)
    if attn_sink is not None:
        m = paddle.maximum(
            m, attn_sink.reshape([1, 1, -1, 1]).astype("float32")
        )
    m = paddle.where(paddle.isfinite(m), m, paddle.zeros_like(m))
    p = paddle.exp(logits - m)
    denom = p.sum(axis=-1, keepdim=True)
    if attn_sink is not None:
        denom = denom + paddle.exp(
            attn_sink.reshape([1, 1, -1, 1]).astype("float32") - m
        )
    denom = paddle.where(denom > 0, denom, paddle.ones_like(denom))
    p = p / denom
    out = paddle.einsum("bshk,bkc->bshc", p, v)
    out = out * row_has.astype("float32").unsqueeze(-1).unsqueeze(-1)
    return out


def _causal_valid_range(b, s):
    eos = (
        paddle.arange(1, s + 1, dtype="int32")
        .reshape([1, s, 1])
        .expand([b, s, 1])
    )
    bos = paddle.zeros([b, s, 1], dtype="int32")
    return paddle.concat([bos, eos], axis=-1).contiguous()


def _cos(a, b):
    import numpy as np

    af = a.astype("float32").numpy().reshape(-1)
    bf = b.astype("float32").numpy().reshape(-1)
    denom = (np.linalg.norm(af) * np.linalg.norm(bf)) + 1e-12
    return float(np.dot(af, bf) / denom)


def _select_blocks(b, s, block_B, topk):
    """Per-token relative block ids: block 0 plus the running block (pos//BB),
    padded with -1 to width ``topk``."""
    pos = paddle.arange(s)
    b0 = paddle.zeros([b, s, 1], dtype="int32")
    b1 = (pos // block_B).cast("int32").reshape([1, s, 1]).expand([b, s, 1])
    idx = paddle.concat([b0, b1], axis=-1)  # [b, s, 2]
    if topk > 2:
        pad = paddle.full([b, s, topk - 2], -1, dtype="int32")
        idx = paddle.concat([idx, pad], axis=-1)
    return idx.contiguous()


class TestBlockSparseDSA(unittest.TestCase):
    BLOCK_B = 64
    Dk = 576
    Dv = 512

    def _make(self, b, s, h, topk, seed=7):
        paddle.seed(seed)
        q = paddle.randn([b, s, h, self.Dk]).cast("bfloat16")
        kf = paddle.randn([b, s, self.Dk]).cast("bfloat16")
        vr = _causal_valid_range(b, s)
        idx = _select_blocks(b, s, self.BLOCK_B, topk)
        sm_scale = 1.0 / math.sqrt(self.Dk)
        return q, kf, vr, idx, sm_scale

    def _run_dsa(self, q, kf, idx, vr, sm_scale):
        from paddlefleet.cudnn_ops import block_sparse_mqa_attention_dsa

        qd = q.detach().clone()
        qd.stop_gradient = False
        kd = kf.detach().clone()
        kd.stop_gradient = False
        out, _ = block_sparse_mqa_attention_dsa(
            qd,
            kd,
            idx,
            vr,
            sm_scale=sm_scale,
            block_B=self.BLOCK_B,
            kv_lora_rank=self.Dv,
        )
        out.sum().backward()
        return out, qd.grad, kd.grad

    def _run_ref(self, q, kf, idx, vr, sm_scale):
        b, s, h = q.shape[0], q.shape[1], q.shape[2]
        s_kv = kf.shape[1]
        qr = q.detach().cast("float32")
        qr.stop_gradient = False
        kr = kf.detach().cast("float32")
        kr.stop_gradient = False
        vr_ = kr[:, :, : self.Dv]
        allow = _allow_mask(idx, vr, s_kv, self.BLOCK_B)
        out = _ref_masked_attn(qr, kr, vr_, allow, sm_scale)
        out = out.reshape([b, s, h * self.Dv])
        out.sum().backward()
        return out, qr.grad, kr.grad

    def test_dsa_vs_dense_reference_h4(self):
        _skip_if_no_dsa(self)
        b, s, h, topk = 1, 192, 4, 2
        q, kf, vr, idx, sm = self._make(b, s, h, topk)
        out_dsa, dq_dsa, dkv_dsa = self._run_dsa(q, kf, idx, vr, sm)
        out_ref, dq_ref, dkv_ref = self._run_ref(q, kf, idx, vr, sm)
        self.assertEqual(list(out_dsa.shape), [b, s, h * self.Dv])
        self.assertGreater(_cos(out_dsa, out_ref), 0.99)
        self.assertGreater(_cos(dq_dsa, dq_ref), 0.99)
        self.assertGreater(_cos(dkv_dsa, dkv_ref), 0.99)

    def test_dsa_vs_dense_reference_h64(self):
        _skip_if_no_dsa(self)
        b, s, h, topk = 1, 256, 64, 4
        q, kf, vr, idx, sm = self._make(b, s, h, topk, seed=13)
        out_dsa, dq_dsa, dkv_dsa = self._run_dsa(q, kf, idx, vr, sm)
        out_ref, dq_ref, dkv_ref = self._run_ref(q, kf, idx, vr, sm)
        self.assertGreater(_cos(out_dsa, out_ref), 0.99)
        self.assertGreater(_cos(dq_dsa, dq_ref), 0.99)
        self.assertGreater(_cos(dkv_dsa, dkv_ref), 0.99)

    def test_learnable_sink_matches_reference(self):
        # A finite learnable per-head sink: DSA forward + dq/dkv/d_sink grads
        # must match the dense reference that folds the sink into the softmax.
        _skip_if_no_dsa(self)
        from paddlefleet.cudnn_ops import block_sparse_mqa_attention_dsa

        b, s, h, topk = 1, 192, 4, 2
        q, kf, vr, idx, sm = self._make(b, s, h, topk, seed=21)
        paddle.seed(29)
        # Moderate sink magnitude: a large random sink shrinks the effective
        # softmax mass and amplifies bf16 rounding in the dq GEMM, which would
        # make the comparison test bf16 noise rather than the sink math.
        sink0 = paddle.randn([h], dtype="float32") * 0.5

        qd = q.detach().clone()
        kd = kf.detach().clone()
        sink_d = sink0.detach().clone()
        for t in (qd, kd, sink_d):
            t.stop_gradient = False
        out_dsa, _ = block_sparse_mqa_attention_dsa(
            qd,
            kd,
            idx,
            vr,
            sm_scale=sm,
            block_B=self.BLOCK_B,
            kv_lora_rank=self.Dv,
            attn_sink=sink_d,
        )
        out_dsa.sum().backward()

        s_kv = kf.shape[1]
        qr = q.detach().cast("float32")
        kr = kf.detach().cast("float32")
        sink_r = sink0.detach().clone()
        for t in (qr, kr, sink_r):
            t.stop_gradient = False
        vr_ = kr[:, :, : self.Dv]
        allow = _allow_mask(idx, vr, s_kv, self.BLOCK_B)
        out_ref = _ref_masked_attn(qr, kr, vr_, allow, sm, attn_sink=sink_r)
        out_ref = out_ref.reshape([b, s, h * self.Dv])
        out_ref.sum().backward()

        self.assertGreater(_cos(out_dsa, out_ref), 0.99)
        self.assertIsNotNone(sink_d.grad)
        self.assertGreater(_cos(qd.grad, qr.grad), 0.99)
        self.assertGreater(_cos(sink_d.grad, sink_r.grad), 0.99)

    def test_neg_sink_matches_sinkless(self):
        # Sinkless (attn_sink=None) must match a very-negative explicit sink:
        # exp(sink - m) -> 0, so the softmax denominator is unchanged.
        _skip_if_no_dsa(self)
        from paddlefleet.cudnn_ops import block_sparse_mqa_attention_dsa

        b, s, h, topk = 1, 192, 4, 2
        q, kf, vr, idx, sm = self._make(b, s, h, topk, seed=33)
        neg_sink = paddle.full([h], -1e30, dtype="float32")
        out_none, _ = block_sparse_mqa_attention_dsa(
            q,
            kf,
            idx,
            vr,
            sm_scale=sm,
            block_B=self.BLOCK_B,
            kv_lora_rank=self.Dv,
        )
        out_neg, _ = block_sparse_mqa_attention_dsa(
            q,
            kf,
            idx,
            vr,
            sm_scale=sm,
            block_B=self.BLOCK_B,
            kv_lora_rank=self.Dv,
            attn_sink=neg_sink,
        )
        self.assertGreater(_cos(out_none, out_neg), 0.999999)


if __name__ == "__main__":
    unittest.main()
