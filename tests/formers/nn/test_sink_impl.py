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

"""Behavior tests for paddlefleet.nn.attention.sink_impl.

The sink ("attention sink") mechanism adds an extra, always-attended logit
`sink` (one per query head) to the softmax denominator. If Z = sum_j exp(s_j)
is the ordinary attention denominator and lse = log(Z) is the log-sum-exp
returned by the underlying flash kernel, then the sink-adjusted output is

    out = raw_output * Z / (exp(sink) + Z)
        = raw_output * 1 / (exp(sink - lse) + 1)

where raw_output = softmax(QK^T) @ V is the ordinary (non-sink) attention
output. `multiplier = 1/(exp(sink - lse) + 1)` is exactly the tensor the
production forward computes and applies per (batch, position, head).

The core flash/flashmask kernels (`_C_ops.flash_attn`,
`paddle.nn.functional.flashmask_attention`) are GPU-only, so their numerics
are NOT verified here (无卡). Instead we:
  * exercise the REAL shape / GQA / sink-size validation guards (they run
    before any kernel call, so they are CPU-verifiable);
  * mock ONLY the GPU-only flash/flashmask dispatch (a genuinely
    not-under-test collaborator) with distinguishable (raw_output, lse) and
    verify the REAL sink composition, lse<->position/head correspondence,
    flash/flashmask routing, GQA key/value expansion and argument forwarding
    against independent hand-derived NumPy references.
No fabricated GPU kernel numerics are asserted.
"""

import unittest
from unittest import mock

import numpy as np
import paddle

from paddlefleet.nn.attention.sink_impl import sink_attention_forward

_FWD = "paddlefleet.nn.attention.sink_impl._flash_attention_forward_dispatch"
_FMFWD = (
    "paddlefleet.nn.attention.sink_impl._flashmask_attention_forward_dispatch"
)


def _t(arr):
    return paddle.to_tensor(np.asarray(arr, dtype=np.float32))


def _distinguishable_raw(b, s, h, d):
    """Positive, position/head-distinguishable [B, S, H, D] tensor content."""
    return np.arange(b * s * h * d, dtype=np.float32).reshape(b, s, h, d) + 1.0


def _sink_multiplier(lse_bhs, sink_h):
    """Independent multiplier reference 1/(exp(sink-lse)+1); lse is [B, H, S]."""
    lse_bsh = np.transpose(np.asarray(lse_bhs, np.float64), (0, 2, 1))  # B,S,H
    sink = np.asarray(sink_h, np.float64).reshape(1, 1, -1)
    return 1.0 / (np.exp(sink - lse_bsh) + 1.0)


def _sink_reference(raw_bshd, lse_bhs, sink_h):
    """Expected sink output = raw_output * multiplier, broadcast over head_dim."""
    m = _sink_multiplier(lse_bhs, sink_h)[..., None]  # B,S,H,1
    return np.asarray(raw_bshd, np.float64) * m


def _repeat_kv_np(x_bshd, n_rep):
    """Independent block-repeat of KV heads matching utils.repeat_kv semantics."""
    x = np.asarray(x_bshd)
    b, s, h, d = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, :, None, :]
    x = np.tile(x, (1, 1, 1, n_rep, 1))
    return x.reshape(b, s, h * n_rep, d)


class TestSinkAttentionValidation(unittest.TestCase):
    """Real forward guards reject malformed inputs before any GPU kernel runs;
    these are CPU-verifiable contract checks with distinguishable bad inputs."""

    def setUp(self):
        paddle.set_device("cpu")

    def _valid(self):
        q = _t(_distinguishable_raw(2, 4, 2, 8))
        k = _t(_distinguishable_raw(2, 4, 2, 8))
        v = _t(_distinguishable_raw(2, 4, 2, 8))
        sink = _t([0.1, -0.2])
        return q, k, v, sink

    def test_rejects_non_4d_query(self):
        _, k, v, sink = self._valid()
        q = _t(np.ones((2, 8)))
        with self.assertRaises(AssertionError):
            sink_attention_forward(q, k, v, sink)

    def test_rejects_non_1d_sink(self):
        q, k, v, _ = self._valid()
        sink = _t(np.ones((2, 2)))
        with self.assertRaises(AssertionError):
            sink_attention_forward(q, k, v, sink)

    def test_rejects_batch_mismatch(self):
        q, _, v, sink = self._valid()
        k = _t(_distinguishable_raw(3, 4, 2, 8))
        with self.assertRaises(AssertionError):
            sink_attention_forward(q, k, v, sink)

    def test_rejects_head_dim_mismatch(self):
        q, _, v, sink = self._valid()
        k = _t(_distinguishable_raw(2, 4, 2, 16))
        with self.assertRaises(AssertionError):
            sink_attention_forward(q, k, v, sink)

    def test_rejects_kv_head_mismatch(self):
        q, k, _, sink = self._valid()
        v = _t(
            _distinguishable_raw(2, 4, 3, 8)
        )  # value kv-heads != key kv-heads
        with self.assertRaises(AssertionError):
            sink_attention_forward(q, k, v, sink)

    def test_rejects_non_divisible_gqa(self):
        q = _t(_distinguishable_raw(2, 4, 3, 8))  # 3 q-heads
        k = _t(_distinguishable_raw(2, 4, 2, 8))  # 2 kv-heads, 3 % 2 != 0
        v = _t(_distinguishable_raw(2, 4, 2, 8))
        sink = _t([0.1, 0.2, 0.3])
        with self.assertRaises(AssertionError):
            sink_attention_forward(q, k, v, sink)

    def test_rejects_sink_size_not_matching_q_heads(self):
        q, k, v, _ = self._valid()
        sink = _t([0.1, 0.2, 0.3, 0.4])  # 4 sink entries but only 2 q-heads
        with self.assertRaises(AssertionError):
            sink_attention_forward(q, k, v, sink)

    def test_rejects_seq_mismatch_on_flash_path(self):
        q, _, _, sink = self._valid()
        k = _t(_distinguishable_raw(2, 8, 2, 8))  # seq_k != seq_q, no startend
        v = _t(_distinguishable_raw(2, 8, 2, 8))
        with self.assertRaises(AssertionError):
            sink_attention_forward(q, k, v, sink)

    def test_rejects_dense_mask_with_startend_row_indices(self):
        q, k, v, sink = self._valid()
        mask = _t(np.zeros((2, 2, 4, 4)))
        sei = paddle.randint(0, 4, [2, 3], dtype="int32")
        with self.assertRaises(AssertionError):
            sink_attention_forward(
                q, k, v, sink, attention_mask=mask, startend_row_indices=sei
            )


class TestSinkComposition(unittest.TestCase):
    """Real sink composition layered on a mocked (GPU-only) flash kernel.

    Only the flash dispatch is mocked; the multiplier math, LSE transpose and
    per-head sink mapping run for real and are checked against an independent
    NumPy reference. Kernel numerics are explicitly out of scope (无卡).
    """

    def setUp(self):
        paddle.set_device("cpu")

    def _run_flash(self, raw_np, lse_np, sink_np):
        """Drive sink_attention_forward on the flash (startend=None) path with
        the flash dispatch returning the given raw_output / lse."""
        b, s, h, d = raw_np.shape
        q = _t(_distinguishable_raw(b, s, h, d))
        k = _t(_distinguishable_raw(b, s, h, d))
        v = _t(_distinguishable_raw(b, s, h, d))
        sink = _t(sink_np)
        with mock.patch(_FWD, return_value=(_t(raw_np), _t(lse_np))):
            out = sink_attention_forward(q, k, v, sink)
        return out.numpy().astype(np.float64)

    def test_matches_independent_sink_reference(self):
        b, s, h, d = 1, 3, 2, 4
        raw = _distinguishable_raw(b, s, h, d)
        # Distinct, stable lse for every (b, h, s); last dim == s (no truncation).
        lse = (
            np.arange(b * h * s, dtype=np.float32).reshape(b, h, s) * 0.1 - 0.3
        )
        sink = np.array([0.2, -0.5], dtype=np.float32)
        out = self._run_flash(raw, lse, sink)
        ref = _sink_reference(raw, lse, sink)
        np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-5)

    def test_zero_sink_zero_lse_halves_output(self):
        # multiplier = 1/(exp(0-0)+1) = 0.5 exactly.
        b, s, h, d = 1, 2, 1, 3
        raw = _distinguishable_raw(b, s, h, d)
        lse = np.zeros((b, h, s), dtype=np.float32)
        sink = np.zeros((h,), dtype=np.float32)
        out = self._run_flash(raw, lse, sink)
        np.testing.assert_allclose(out, raw * 0.5, atol=1e-6, rtol=0)

    def test_large_negative_sink_recovers_raw_output(self):
        # sink -> -inf => multiplier -> 1 => output == raw_output.
        b, s, h, d = 1, 2, 2, 3
        raw = _distinguishable_raw(b, s, h, d)
        lse = np.zeros((b, h, s), dtype=np.float32)
        sink = np.full((h,), -60.0, dtype=np.float32)
        out = self._run_flash(raw, lse, sink)
        np.testing.assert_allclose(out, raw, atol=1e-5, rtol=1e-5)

    def test_large_positive_sink_suppresses_output(self):
        # sink -> +inf => multiplier -> 0 => output == 0, a real change from raw.
        b, s, h, d = 1, 2, 2, 3
        raw = _distinguishable_raw(b, s, h, d)
        lse = np.zeros((b, h, s), dtype=np.float32)
        sink = np.full((h,), 60.0, dtype=np.float32)
        out = self._run_flash(raw, lse, sink)
        np.testing.assert_allclose(out, np.zeros_like(raw), atol=1e-5, rtol=0)
        self.assertGreater(float(np.abs(raw).max()), 1.0)

    def test_per_head_sink_mapping(self):
        # head 0 sink -> -inf (keep raw); head 1 sink -> +inf (suppress).
        # Catches sink<->head misalignment in the reshape/expand.
        b, s, h, d = 1, 2, 2, 3
        raw = _distinguishable_raw(b, s, h, d)
        lse = np.zeros((b, h, s), dtype=np.float32)
        sink = np.array([-60.0, 60.0], dtype=np.float32)
        out = self._run_flash(raw, lse, sink)
        np.testing.assert_allclose(out[:, :, 0, :], raw[:, :, 0, :], atol=1e-5)
        np.testing.assert_allclose(
            out[:, :, 1, :], np.zeros((b, s, d)), atol=1e-5
        )
        # The two heads' raw contents differ, so a head-swap would be visible.
        self.assertGreater(
            float(np.abs(raw[:, :, 0, :] - raw[:, :, 1, :]).max()), 1e-3
        )

    def test_lse_position_correspondence(self):
        # Every (h, s) gets a distinct lse; an lse transpose bug (seq<->head)
        # would misalign values or shapes and fail the exact comparison.
        b, s, h, d = 1, 3, 2, 2
        raw = np.ones((b, s, h, d), dtype=np.float32)
        lse = np.arange(b * h * s, dtype=np.float32).reshape(b, h, s) + 1.0
        sink = np.array([0.3, -0.4], dtype=np.float32)
        out = self._run_flash(raw, lse, sink)
        ref = _sink_reference(raw, lse, sink)
        np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-5)


class TestSinkDispatchRouting(unittest.TestCase):
    """Routing between flash / flashmask, argument forwarding and GQA expansion
    of the real orchestration, with the GPU-only kernels mocked."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_routes_to_flash_without_startend_row_indices(self):
        b, s, h, d = 1, 2, 2, 3
        q = _t(_distinguishable_raw(b, s, h, d))
        k = _t(_distinguishable_raw(b, s, h, d))
        v = _t(_distinguishable_raw(b, s, h, d))
        sink = _t([0.1, 0.2])
        raw = _t(_distinguishable_raw(b, s, h, d))
        lse = _t(np.zeros((b, h, s), dtype=np.float32))
        with (
            mock.patch(_FWD, return_value=(raw, lse)) as flash,
            mock.patch(_FMFWD, return_value=(raw, lse)) as flashmask,
        ):
            sink_attention_forward(q, k, v, sink)
        flash.assert_called_once()
        flashmask.assert_not_called()

    def test_routes_to_flashmask_with_startend_row_indices(self):
        b, s, h, d = 1, 2, 2, 3
        q = _t(_distinguishable_raw(b, s, h, d))
        k = _t(_distinguishable_raw(b, s, h, d))
        v = _t(_distinguishable_raw(b, s, h, d))
        sink = _t([0.1, 0.2])
        raw = _t(_distinguishable_raw(b, s, h, d))
        lse = _t(np.zeros((b, h, s), dtype=np.float32))
        sei = paddle.randint(0, s, [b, 3], dtype="int32")
        with (
            mock.patch(_FWD, return_value=(raw, lse)) as flash,
            mock.patch(_FMFWD, return_value=(raw, lse)) as flashmask,
        ):
            sink_attention_forward(q, k, v, sink, startend_row_indices=sei)
        flashmask.assert_called_once()
        flash.assert_not_called()

    def test_forwards_scale_causal_dropout_and_qkv(self):
        b, s, h, d = 1, 2, 2, 4
        q = _t(_distinguishable_raw(b, s, h, d))
        k = _t(_distinguishable_raw(b, s, h, d))
        v = _t(_distinguishable_raw(b, s, h, d))
        sink = _t([0.0, 0.0])
        raw_np = _distinguishable_raw(b, s, h, d)
        lse_np = np.zeros((b, h, s), dtype=np.float32)
        captured = {}

        def spy(query, key, value, dropout=0.0, causal=False, **kwargs):
            captured["args"] = (query, key, value, dropout, causal)
            captured["kwargs"] = kwargs
            return _t(raw_np), _t(lse_np)

        with mock.patch(_FWD, side_effect=spy):
            out = sink_attention_forward(
                q, k, v, sink, dropout_p=0.0, softmax_scale=0.5, causal=True
            )
        query, key, value, dropout, causal = captured["args"]
        self.assertIs(query, q)  # q forwarded unchanged
        self.assertIs(key, k)  # n_rep == 1 -> repeat_kv returns its input
        self.assertIs(value, v)
        self.assertEqual(dropout, 0.0)
        self.assertTrue(causal)
        self.assertEqual(captured["kwargs"]["softmax_scale"], 0.5)
        # The mocked raw output is actually consumed (sink=0, lse=0 -> * 0.5).
        np.testing.assert_allclose(
            out.numpy().astype(np.float64), raw_np * 0.5, atol=1e-6
        )

    def test_gqa_repeats_kv_before_kernel(self):
        # 4 query heads, 2 kv heads -> kernel must receive block-repeated kv.
        b, s, hq, hkv, d = 1, 2, 4, 2, 3
        q = _t(_distinguishable_raw(b, s, hq, d))
        k = _t(_distinguishable_raw(b, s, hkv, d))
        v = _t(_distinguishable_raw(b, s, hkv, d))
        sink = _t([0.0, 0.0, 0.0, 0.0])
        raw_np = _distinguishable_raw(b, s, hq, d)
        lse_np = np.zeros((b, hq, s), dtype=np.float32)
        captured = {}

        def spy(query, key, value, *a, **kw):
            captured["key"] = key.numpy().copy()
            captured["value"] = value.numpy().copy()
            return _t(raw_np), _t(lse_np)

        with mock.patch(_FWD, side_effect=spy):
            sink_attention_forward(q, k, v, sink)
        n_rep = hq // hkv
        np.testing.assert_array_equal(
            captured["key"], _repeat_kv_np(k.numpy(), n_rep)
        )
        np.testing.assert_array_equal(
            captured["value"], _repeat_kv_np(v.numpy(), n_rep)
        )
        self.assertEqual(captured["key"].shape[2], hq)


class TestSinkLSETruncation(unittest.TestCase):
    """Padded-LSE truncation control logic (the seqlen_q_rounded path)."""

    def setUp(self):
        paddle.set_device("cpu")

    def _run(self, raw_np, lse_np, sink_np):
        b, s, h, d = raw_np.shape
        q = _t(_distinguishable_raw(b, s, h, d))
        k = _t(_distinguishable_raw(b, s, h, d))
        v = _t(_distinguishable_raw(b, s, h, d))
        sink = _t(sink_np)
        with mock.patch(_FWD, return_value=(_t(raw_np), _t(lse_np))):
            out = sink_attention_forward(q, k, v, sink)
        return out.numpy().astype(np.float64)

    def test_single_head_padded_lse_uses_leading_values(self):
        # Single head, padded lse [1, 1, 8]; only the first seq_len=4 are real.
        b, s, h, d = 1, 4, 1, 3
        raw = _distinguishable_raw(b, s, h, d)
        lse_pad = np.arange(8, dtype=np.float32).reshape(1, 1, 8)
        sink = np.array([0.25], dtype=np.float32)
        out = self._run(raw, lse_pad, sink)
        lse_real = lse_pad[:, :, :s]  # correct per-head truncation (H == 1)
        ref = _sink_reference(raw, lse_real, sink)
        np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-5)

    @unittest.expectedFailure
    def test_multihead_padded_lse_should_preserve_per_head_values(self):
        """BUG EXPOSURE: with >1 head and a padded LSE, each head must keep its
        own leading seq_len LSE values, i.e. lse[:, :, :seq_len]. The current
        production truncation
            lse.flatten()[: B*H*seq_len].reshape(B, H, seq_len)
        instead pulls head 0's padding tail into head 1 (for H >= 2), so this
        test asserts the CORRECT head-preserving contract and is expected to
        FAIL until the truncation is fixed. See report. Marked expectedFailure
        so the real production defect surfaces without editing production code.
        """
        b, s, h, d = 1, 4, 2, 3
        raw = _distinguishable_raw(b, s, h, d)
        # Padded seq dim (8 > seq_len 4); heads carry distinct lse values.
        lse_pad = np.arange(2 * 8, dtype=np.float32).reshape(1, 2, 8)
        sink = np.array([0.1, -0.2], dtype=np.float32)
        out = self._run(raw, lse_pad, sink)
        lse_correct = lse_pad[:, :, :s]  # head-preserving truncation
        ref = _sink_reference(raw, lse_correct, sink)
        np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
