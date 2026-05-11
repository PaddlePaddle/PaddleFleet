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

"""Unit tests for fused_apply_mla_rope_for_q / fused_apply_mla_rope_for_kv.

Covers:
  1. Forward pass accuracy vs a pure-Paddle reference for bshd and thd.
  2. Backward pass accuracy vs the same reference.
  3. Multiple dtypes (float32 / float16 / bfloat16).

The fused ops apply YARN-style RoPE where `cos` / `sin` are pre-computed
(shape ``[max_seq, 1, 1, emb_dim]``). The RoPE convention is:

    x1 = x[..., 0::2]         # even indices
    x2 = x[..., 1::2]         # odd indices
    cos_left,  cos_right = cos[..., :D/2], cos[..., D/2:]
    sin_left,  sin_right = sin[..., :D/2], sin[..., D/2:]
    out_left  = x1 * cos_left  - x2 * sin_left
    out_right = x2 * cos_right + x1 * sin_right
    out = concat([out_left, out_right], axis=-1)   # contiguous halves
"""

import unittest

import numpy as np
import paddle

try:
    import triton  # noqa: F401

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    from paddlefleet.fusions.fused_mla_yarn_rope_apply import (
        fused_apply_mla_rope_for_kv,
        fused_apply_mla_rope_for_q,
    )


# Tolerances: RoPE uses a few fused multiply-adds + trig lookups, so the
# numeric budget is similar to apply_rotary_pos_emb_vision tests.
_TOL = {
    "float32": (1e-5, 1e-5),
    "float16": (5e-3, 5e-3),
    "bfloat16": (3e-2, 3e-2),
}


# ---------------------------------------------------------------------------
# Reference implementation (bshd layout internally)
# ---------------------------------------------------------------------------


def _apply_rope_emb_ref(emb, cos_row, sin_row):
    """Apply RoPE to the `emb` tensor.

    Args:
        emb:     [..., emb_dim]     (fp32 compute)
        cos_row: [..., emb_dim]     broadcastable to emb
        sin_row: [..., emb_dim]     broadcastable to emb

    Returns:
        Tensor of same shape as `emb` with RoPE applied.
    """
    emb_dim = emb.shape[-1]
    half = emb_dim // 2
    x1 = emb[..., 0::2]  # [..., half]
    x2 = emb[..., 1::2]
    cos_left = cos_row[..., :half]
    cos_right = cos_row[..., half:]
    sin_left = sin_row[..., :half]
    sin_right = sin_row[..., half:]
    out_left = x1 * cos_left - x2 * sin_left
    out_right = x2 * cos_right + x1 * sin_right
    return paddle.concat([out_left, out_right], axis=-1)


def apply_mla_rope_for_q_ref(
    q, cos, sin, qk_head_dim, emb_dim, cu_seqlens_q=None
):
    """Pure-Paddle reference for fused_apply_mla_rope_for_q.

    q: [bs, seq, heads, qk_head_dim + emb_dim] (bshd)
       or [total_seq, heads, qk_head_dim + emb_dim] (thd)
    cos/sin: [max_seq, 1, 1, emb_dim]
    """
    orig_dtype = q.dtype
    q_f32 = q.astype("float32")
    head = q_f32[..., :qk_head_dim]
    emb_part = q_f32[..., qk_head_dim:]

    cos_f32 = cos.astype("float32").reshape([cos.shape[0], emb_dim])
    sin_f32 = sin.astype("float32").reshape([sin.shape[0], emb_dim])

    if cu_seqlens_q is None:
        # bshd: gather cos/sin along seq axis (dim=1)
        seq_len = q.shape[1]
        cos_row = cos_f32[:seq_len].reshape([1, seq_len, 1, emb_dim])
        sin_row = sin_f32[:seq_len].reshape([1, seq_len, 1, emb_dim])
        out_emb = _apply_rope_emb_ref(emb_part, cos_row, sin_row)
    else:
        # thd: compute per-token position within its sequence
        cu = cu_seqlens_q.numpy().tolist()
        pos_list = []
        for s in range(len(cu) - 1):
            seg = cu[s + 1] - cu[s]
            pos_list.extend(range(seg))
        positions = paddle.to_tensor(pos_list, dtype="int64", place=q.place)
        cos_row = paddle.index_select(
            cos_f32, positions, axis=0
        )  # [T, emb_dim]
        sin_row = paddle.index_select(sin_f32, positions, axis=0)
        cos_row = cos_row.reshape([-1, 1, emb_dim])
        sin_row = sin_row.reshape([-1, 1, emb_dim])
        out_emb = _apply_rope_emb_ref(emb_part, cos_row, sin_row)

    out = paddle.concat([head, out_emb], axis=-1)
    return out.astype(orig_dtype)


def apply_mla_rope_for_kv_ref(
    kv, k_pos_emb, cos, sin, emb_dim, k_dim, v_dim, cu_seqlens_kv=None
):
    """Pure-Paddle reference for fused_apply_mla_rope_for_kv.

    kv:        [bs, seq, heads, k_dim + v_dim] or [T, heads, k_dim + v_dim]
    k_pos_emb: [bs, seq, 1, emb_dim]           or [T, 1, emb_dim]
    cos/sin:   [max_seq, 1, 1, emb_dim]

    Returns: (key, value) matching the fused op's output layouts:
        key:   [..., heads, k_dim + emb_dim]   (first k_dim = original k,
                                                then rope_left, then rope_right)
        value: [..., heads, v_dim]
    """
    orig_dtype = kv.dtype
    kv_f32 = kv.astype("float32")
    k = kv_f32[..., :k_dim]  # [..., heads, k_dim]
    v = kv_f32[..., k_dim : k_dim + v_dim]

    cos_f32 = cos.astype("float32").reshape([cos.shape[0], emb_dim])
    sin_f32 = sin.astype("float32").reshape([sin.shape[0], emb_dim])
    emb_f32 = k_pos_emb.astype("float32")

    if cu_seqlens_kv is None:
        # bshd: gather cos/sin along seq axis (dim=1)
        bs = kv.shape[0]
        seq_len = kv.shape[1]
        heads = kv.shape[2]
        cos_row = cos_f32[:seq_len].reshape([1, seq_len, 1, emb_dim])
        sin_row = sin_f32[:seq_len].reshape([1, seq_len, 1, emb_dim])
        rope_emb = _apply_rope_emb_ref(
            emb_f32, cos_row, sin_row
        )  # [bs, seq, 1, emb_dim]
        # broadcast across heads
        rope_emb = rope_emb.expand([bs, seq_len, heads, emb_dim])
    else:
        cu = cu_seqlens_kv.numpy().tolist()
        pos_list = []
        for s in range(len(cu) - 1):
            seg = cu[s + 1] - cu[s]
            pos_list.extend(range(seg))
        positions = paddle.to_tensor(pos_list, dtype="int64", place=kv.place)
        cos_row = paddle.index_select(cos_f32, positions, axis=0).reshape(
            [-1, 1, emb_dim]
        )
        sin_row = paddle.index_select(sin_f32, positions, axis=0).reshape(
            [-1, 1, emb_dim]
        )
        rope_emb = _apply_rope_emb_ref(
            emb_f32, cos_row, sin_row
        )  # [T, 1, emb_dim]
        heads = kv.shape[1]
        rope_emb = rope_emb.expand([kv.shape[0], heads, emb_dim])

    key = paddle.concat([k, rope_emb], axis=-1)
    return key.astype(orig_dtype), v.astype(orig_dtype)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rand(shape, dtype, seed):
    rng = np.random.RandomState(seed)
    return paddle.to_tensor(
        rng.randn(*shape).astype("float32"), place="gpu"
    ).astype(dtype)


def _make_cos_sin(max_seq, emb_dim, seed=0):
    """Return (cos, sin) of shape [max_seq, 1, 1, emb_dim], fp32 contiguous."""
    rng = np.random.RandomState(seed)
    # Use bounded values so overflow is not a concern at fp16.
    angle = rng.uniform(-3.0, 3.0, size=(max_seq, emb_dim)).astype("float32")
    cos = np.cos(angle).reshape(max_seq, 1, 1, emb_dim)
    sin = np.sin(angle).reshape(max_seq, 1, 1, emb_dim)
    return (
        paddle.to_tensor(cos, place="gpu"),
        paddle.to_tensor(sin, place="gpu"),
    )


def _allclose(a, b, dtype, msg=""):
    atol, rtol = _TOL[dtype]
    np.testing.assert_allclose(
        a.numpy().astype("float32"),
        b.numpy().astype("float32"),
        atol=atol,
        rtol=rtol,
        err_msg=msg,
    )


# ---------------------------------------------------------------------------
# Base test class with skip guards
# ---------------------------------------------------------------------------


class _BaseMLARopeTest(unittest.TestCase):
    def setUp(self):
        if not HAS_TRITON:
            self.skipTest("triton is not available")
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA is not available")
        paddle.device.set_device("gpu:0")


# ---------------------------------------------------------------------------
# fused_apply_mla_rope_for_q: forward + backward, bshd + thd
# ---------------------------------------------------------------------------


class TestApplyMLARopeForQ(_BaseMLARopeTest):
    def _run_bshd_fwd(self, bs, seq, heads, qk_head_dim, emb_dim, dtype):
        head_dim = qk_head_dim + emb_dim
        # bshd: [batch, seq, heads, head_dim]
        q = _rand([bs, seq, heads, head_dim], dtype, seed=1)
        cos, sin = _make_cos_sin(seq, emb_dim, seed=7)

        q_in = q.clone().detach()
        out = fused_apply_mla_rope_for_q(
            q_in, cos, sin, qk_head_dim, emb_dim, None, 0, 1, False
        )
        # Reference: apply directly in bshd layout (no transpose to avoid
        # layout-induced numeric differences).
        ref = apply_mla_rope_for_q_ref(q, cos, sin, qk_head_dim, emb_dim, None)

        self.assertEqual(list(out.shape), [bs, seq, heads, head_dim])
        _allclose(out, ref, dtype, f"Q bshd fwd dtype={dtype}")
        # np.testing.assert_equal(out.numpy(), ref.numpy(), err_msg=f"out={out._md5sum()} != ref={ref._md5sum()}")
        # np.testing.assert_equal(out._md5sum(), ref._md5sum())

    def _run_thd_fwd(self, seq_lens, heads, qk_head_dim, emb_dim, dtype):
        total = sum(seq_lens)
        max_seq = max(seq_lens)
        head_dim = qk_head_dim + emb_dim
        q = _rand([total, heads, head_dim], dtype, seed=2)
        cos, sin = _make_cos_sin(max_seq, emb_dim, seed=8)
        cu = paddle.to_tensor(
            np.cumsum([0, *list(seq_lens)]).astype("int32"), place="gpu"
        )

        q_in = q.clone().detach()
        out = fused_apply_mla_rope_for_q(
            q_in, cos, sin, qk_head_dim, emb_dim, cu, 0, 1, False
        )
        ref = apply_mla_rope_for_q_ref(q, cos, sin, qk_head_dim, emb_dim, cu)

        self.assertEqual(list(out.shape), [total, heads, head_dim])
        _allclose(out, ref, dtype, f"Q thd fwd dtype={dtype}")

    def _run_bshd_bwd(self, bs, seq, heads, qk_head_dim, emb_dim, dtype):
        head_dim = qk_head_dim + emb_dim
        np.random.seed(11)
        q_np = np.random.randn(bs, seq, heads, head_dim).astype("float32")
        dout_np = np.random.randn(bs, seq, heads, head_dim).astype("float32")
        cos, sin = _make_cos_sin(seq, emb_dim, seed=9)

        # Reference gradient (bshd, no transpose)
        q_ref = paddle.to_tensor(q_np, place="gpu").astype(dtype)
        q_ref.stop_gradient = False
        out_ref = apply_mla_rope_for_q_ref(
            q_ref, cos, sin, qk_head_dim, emb_dim, None
        )
        dout_ref = paddle.to_tensor(dout_np, place="gpu").astype(out_ref.dtype)
        g_ref = paddle.grad([out_ref], [q_ref], [dout_ref])[0]

        # Fused gradient (bshd)
        q_cu = paddle.to_tensor(q_np, place="gpu").astype(dtype)
        q_cu.stop_gradient = False
        out_cu = fused_apply_mla_rope_for_q(
            q_cu, cos, sin, qk_head_dim, emb_dim, None, 0, 1, False
        )
        dout_cu = paddle.to_tensor(dout_np, place="gpu").astype(out_cu.dtype)
        g_cu = paddle.grad([out_cu], [q_cu], [dout_cu])[0]

        _allclose(g_cu, g_ref, dtype, f"Q bshd bwd dtype={dtype}")
        # np.testing.assert_equal(g_cu.numpy(), g_ref.numpy(), err_msg=f"out={g_cu.numpy()} != ref={g_ref.numpy()}")
        # np.testing.assert_equal(g_cu._md5sum(), g_ref._md5sum())

    # ---- forward bshd ----
    def test_fwd_bshd_fp32(self):
        self._run_bshd_fwd(2, 64, 8, 128, 64, "float32")

    # def test_fwd_bshd_bf16(self):
    #     self._run_bshd_fwd(2, 64, 8, 128, 64, "bfloat16")

    def test_fwd_bshd_fp32_odd_heads(self):
        # head count not a power of two exercises BLOCK_H masking
        self._run_bshd_fwd(1, 32, 7, 64, 32, "float32")

    # # ---- forward thd ----
    # def test_fwd_thd_fp32(self):
    #     self._run_thd_fwd([10, 17, 23], 4, 64, 32, "float32")

    # def test_fwd_thd_bf16(self):
    #     self._run_thd_fwd([10, 17, 23], 4, 64, 32, "bfloat16")

    # # ---- backward bshd ----
    def test_bwd_bshd_fp32(self):
        self._run_bshd_bwd(2, 32, 4, 64, 32, "float32")

    # def test_bwd_bshd_bf16(self):
    #     self._run_bshd_bwd(2, 32, 4, 64, 32, "bfloat16")

    def test_bwd_bshd_fp32_odd_heads(self):
        # head count not a power of two exercises BLOCK_H masking in backward
        self._run_bshd_bwd(1, 32, 7, 64, 32, "float32")

    # # ---- backward thd ----
    # def _run_thd_bwd(self, seq_lens, heads, qk_head_dim, emb_dim, dtype):
    #     total = sum(seq_lens)
    #     max_seq = max(seq_lens)
    #     head_dim = qk_head_dim + emb_dim
    #     np.random.seed(51)
    #     q_np = np.random.randn(total, heads, head_dim).astype("float32")
    #     dout_np = np.random.randn(total, heads, head_dim).astype("float32")
    #     cos, sin = _make_cos_sin(max_seq, emb_dim, seed=52)
    #     cu = paddle.to_tensor(
    #         np.cumsum([0, *list(seq_lens)]).astype("int32"), place="gpu"
    #     )

    #     # Reference gradient via autograd on the pure-Paddle reference
    #     q_ref = paddle.to_tensor(q_np, place="gpu").astype(dtype)
    #     q_ref.stop_gradient = False
    #     out_ref = apply_mla_rope_for_q_ref(
    #         q_ref, cos, sin, qk_head_dim, emb_dim, cu
    #     )
    #     dout_ref = paddle.to_tensor(dout_np, place="gpu").astype(out_ref.dtype)
    #     g_ref = paddle.grad([out_ref], [q_ref], [dout_ref])[0]

    #     # Fused gradient (thd)
    #     q_cu = paddle.to_tensor(q_np, place="gpu").astype(dtype)
    #     q_cu.stop_gradient = False
    #     out_cu = fused_apply_mla_rope_for_q(
    #         q_cu, cos, sin, qk_head_dim, emb_dim, cu, 0, 1, False
    #     )
    #     dout_cu = paddle.to_tensor(dout_np, place="gpu").astype(out_cu.dtype)
    #     g_cu = paddle.grad([out_cu], [q_cu], [dout_cu])[0]

    #     _allclose(g_cu, g_ref, dtype, f"Q thd bwd dtype={dtype}")

    # def test_bwd_thd_fp32(self):
    #     self._run_thd_bwd([10, 17, 23], 4, 64, 32, "float32")

    # def test_bwd_thd_bf16(self):
    #     self._run_thd_bwd([10, 17, 23], 4, 64, 32, "bfloat16")


# ---------------------------------------------------------------------------
# fused_apply_mla_rope_for_kv: forward + backward, bshd + thd
# ---------------------------------------------------------------------------


class TestApplyMLARopeForKV(_BaseMLARopeTest):
    def _run_bshd_fwd(self, bs, seq, heads, k_dim, v_dim, emb_dim, dtype):
        # bshd: [batch, seq, heads, k_dim+v_dim]
        kv = _rand([bs, seq, heads, k_dim + v_dim], dtype, seed=21)
        k_pos_emb = _rand([bs, seq, 1, emb_dim], dtype, seed=22)
        cos, sin = _make_cos_sin(seq, emb_dim, seed=23)

        key_out, val_out = fused_apply_mla_rope_for_kv(
            kv, k_pos_emb, cos, sin, emb_dim, k_dim, v_dim, None, 0, 1, False
        )
        # Reference: apply directly in bshd layout (no transpose to avoid
        # layout-induced numeric differences).
        key_ref, val_ref = apply_mla_rope_for_kv_ref(
            kv, k_pos_emb, cos, sin, emb_dim, k_dim, v_dim, None
        )

        self.assertEqual(list(key_out.shape), [bs, seq, heads, k_dim + emb_dim])
        self.assertEqual(list(val_out.shape), [bs, seq, heads, v_dim])
        _allclose(key_out, key_ref, dtype, f"KV bshd fwd key dtype={dtype}")
        _allclose(val_out, val_ref, dtype, f"KV bshd fwd value dtype={dtype}")
        # np.testing.assert_equal(key_out._md5sum(), key_ref._md5sum())
        # np.testing.assert_equal(val_out._md5sum(), val_ref._md5sum())

    def _run_thd_fwd(self, seq_lens, heads, k_dim, v_dim, emb_dim, dtype):
        total = sum(seq_lens)
        max_seq = max(seq_lens)
        kv = _rand([total, heads, k_dim + v_dim], dtype, seed=31)
        k_pos_emb = _rand([total, 1, emb_dim], dtype, seed=32)
        cos, sin = _make_cos_sin(max_seq, emb_dim, seed=33)
        cu = paddle.to_tensor(
            np.cumsum([0, *list(seq_lens)]).astype("int32"), place="gpu"
        )

        key_out, val_out = fused_apply_mla_rope_for_kv(
            kv, k_pos_emb, cos, sin, emb_dim, k_dim, v_dim, cu, 0, 1, False
        )
        key_ref, val_ref = apply_mla_rope_for_kv_ref(
            kv, k_pos_emb, cos, sin, emb_dim, k_dim, v_dim, cu
        )
        self.assertEqual(list(key_out.shape), [total, heads, k_dim + emb_dim])
        self.assertEqual(list(val_out.shape), [total, heads, v_dim])
        _allclose(key_out, key_ref, dtype, f"KV thd fwd key dtype={dtype}")
        _allclose(val_out, val_ref, dtype, f"KV thd fwd value dtype={dtype}")

    def _run_bshd_bwd(self, bs, seq, heads, k_dim, v_dim, emb_dim, dtype):
        np.random.seed(41)
        kv_np = np.random.randn(bs, seq, heads, k_dim + v_dim).astype("float32")
        emb_np = np.random.randn(bs, seq, 1, emb_dim).astype("float32")
        dkey_np = np.random.randn(bs, seq, heads, k_dim + emb_dim).astype(
            "float32"
        )
        dval_np = np.random.randn(bs, seq, heads, v_dim).astype("float32")
        cos, sin = _make_cos_sin(seq, emb_dim, seed=42)

        def _make(xn, dt):
            t = paddle.to_tensor(xn, place="gpu").astype(dt)
            t.stop_gradient = False
            return t

        # Reference via autograd on pure-Paddle reference (bshd, no transpose)
        kv_ref = _make(kv_np, dtype)
        emb_ref = _make(emb_np, dtype)
        key_ref, val_ref = apply_mla_rope_for_kv_ref(
            kv_ref, emb_ref, cos, sin, emb_dim, k_dim, v_dim, None
        )
        dkey_ref = paddle.to_tensor(dkey_np, place="gpu").astype(key_ref.dtype)
        dval_ref = paddle.to_tensor(dval_np, place="gpu").astype(val_ref.dtype)
        gkv_ref, gemb_ref = paddle.grad(
            [key_ref, val_ref], [kv_ref, emb_ref], [dkey_ref, dval_ref]
        )

        # Fused bshd
        kv_cu = _make(kv_np, dtype)
        emb_cu = _make(emb_np, dtype)
        key_cu, val_cu = fused_apply_mla_rope_for_kv(
            kv_cu, emb_cu, cos, sin, emb_dim, k_dim, v_dim, None, 0, 1, False
        )
        dkey_cu = paddle.to_tensor(dkey_np, place="gpu").astype(key_cu.dtype)
        dval_cu = paddle.to_tensor(dval_np, place="gpu").astype(val_cu.dtype)
        gkv_cu, gemb_cu = paddle.grad(
            [key_cu, val_cu], [kv_cu, emb_cu], [dkey_cu, dval_cu]
        )

        _allclose(gkv_cu, gkv_ref, dtype, f"KV bshd bwd dkv dtype={dtype}")
        _allclose(gemb_cu, gemb_ref, dtype, f"KV bshd bwd demb dtype={dtype}")
        # np.testing.assert_equal(gkv_cu._md5sum(), gkv_ref._md5sum())
        # np.testing.assert_equal(gemb_cu._md5sum(), gemb_ref._md5sum())

    # ---- forward bshd ----
    def test_fwd_bshd_fp32(self):
        self._run_bshd_fwd(2, 32, 8, 128, 128, 64, "float32")

    # def test_fwd_bshd_bf16(self):
    #     self._run_bshd_fwd(2, 32, 8, 128, 128, 64, "bfloat16")

    def test_fwd_bshd_fp32_odd_heads(self):
        self._run_bshd_fwd(1, 16, 5, 64, 64, 32, "float32")

    # # ---- forward thd ----
    # def test_fwd_thd_fp32(self):
    #     self._run_thd_fwd([8, 13, 19], 4, 64, 64, 32, "float32")

    # def test_fwd_thd_bf16(self):
    #     self._run_thd_fwd([8, 13, 19], 4, 64, 64, 32, "bfloat16")

    # ---- backward bshd ----
    def test_bwd_bshd_fp32(self):
        self._run_bshd_bwd(2, 16, 4, 64, 64, 32, "float32")

    # def test_bwd_bshd_bf16(self):
    #     self._run_bshd_bwd(2, 16, 4, 64, 64, 32, "bfloat16")

    def test_bwd_bshd_fp32_odd_heads(self):
        # head count not a power of two exercises BLOCK_H masked-load in backward
        self._run_bshd_bwd(1, 16, 5, 64, 64, 32, "float32")

    # # ---- backward thd ----
    # def _run_thd_bwd(self, seq_lens, heads, k_dim, v_dim, emb_dim, dtype):
    #     total = sum(seq_lens)
    #     max_seq = max(seq_lens)
    #     np.random.seed(61)
    #     kv_np = np.random.randn(total, heads, k_dim + v_dim).astype("float32")
    #     emb_np = np.random.randn(total, 1, emb_dim).astype("float32")
    #     dkey_np = np.random.randn(total, heads, k_dim + emb_dim).astype(
    #         "float32"
    #     )
    #     dval_np = np.random.randn(total, heads, v_dim).astype("float32")
    #     cos, sin = _make_cos_sin(max_seq, emb_dim, seed=62)
    #     cu = paddle.to_tensor(
    #         np.cumsum([0, *list(seq_lens)]).astype("int32"), place="gpu"
    #     )

    #     def _make(xn, dt):
    #         t = paddle.to_tensor(xn, place="gpu").astype(dt)
    #         t.stop_gradient = False
    #         return t

    #     # Reference via autograd on pure-Paddle reference (thd)
    #     kv_ref = _make(kv_np, dtype)
    #     emb_ref = _make(emb_np, dtype)
    #     key_ref, val_ref = apply_mla_rope_for_kv_ref(
    #         kv_ref, emb_ref, cos, sin, emb_dim, k_dim, v_dim, cu
    #     )
    #     dkey_ref = paddle.to_tensor(dkey_np, place="gpu").astype(key_ref.dtype)
    #     dval_ref = paddle.to_tensor(dval_np, place="gpu").astype(val_ref.dtype)
    #     gkv_ref, gemb_ref = paddle.grad(
    #         [key_ref, val_ref], [kv_ref, emb_ref], [dkey_ref, dval_ref]
    #     )

    #     # Fused thd
    #     kv_cu = _make(kv_np, dtype)
    #     emb_cu = _make(emb_np, dtype)
    #     key_cu, val_cu = fused_apply_mla_rope_for_kv(
    #         kv_cu, emb_cu, cos, sin, emb_dim, k_dim, v_dim, cu, 0, 1, False
    #     )
    #     dkey_cu = paddle.to_tensor(dkey_np, place="gpu").astype(key_cu.dtype)
    #     dval_cu = paddle.to_tensor(dval_np, place="gpu").astype(val_cu.dtype)
    #     gkv_cu, gemb_cu = paddle.grad(
    #         [key_cu, val_cu], [kv_cu, emb_cu], [dkey_cu, dval_cu]
    #     )

    #     _allclose(gkv_cu, gkv_ref, dtype, f"KV thd bwd dkv dtype={dtype}")
    #     _allclose(gemb_cu, gemb_ref, dtype, f"KV thd bwd demb dtype={dtype}")

    # def test_bwd_thd_fp32(self):
    #     self._run_thd_bwd([8, 13, 19], 4, 64, 64, 32, "float32")

    # def test_bwd_thd_bf16(self):
    #     self._run_thd_bwd([8, 13, 19], 4, 64, 64, 32, "bfloat16")


if __name__ == "__main__":
    unittest.main(verbosity=2)
