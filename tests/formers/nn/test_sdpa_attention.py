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

"""Behavior tests for paddlefleet.nn.attention.sdpa_attention.sdpa_attention_forward.

These tests drive the real production entry point (no mocking of the SDPA
kernel) on CPU and compare against an INDEPENDENT NumPy reference that
hand-computes softmax(Q @ K^T / sqrt(d)) @ V. They verify masked visibility,
causal restriction, GQA broadcast and head/output layout correspondence with
distinguishable Q/K/V, rather than shape-only contracts.

Notes on the exercised production contract:
- Input layout is [batch, heads, seq, dim]; the function transposes internally
  to [batch, seq, heads, dim] before calling paddle SDPA.
- On the non-sink path the effective softmax scale is the paddle default
  1/sqrt(head_dim); the `scaling` argument is NOT forwarded there, so the
  reference uses 1/sqrt(head_dim).
- The function returns (output, None); output is reshaped to
  [batch, seq, heads * dim].
"""

import unittest

import numpy as np
import paddle
import paddle.nn as nn

from paddlefleet.nn.attention.sdpa_attention import sdpa_attention_forward


class _AttnModule(nn.Layer):
    """Minimal carrier for the `is_causal` attribute read by the production code."""

    def __init__(self, is_causal=True):
        super().__init__()
        self.is_causal = is_causal


def _reference_sdpa(
    query, key, value, scale, is_causal=False, additive_mask=None
):
    """Independent NumPy attention reference.

    query/key/value are laid out as [batch, heads, seq, dim] (the production
    input layout). Grouped-query attention maps query head ``h`` to key/value
    head ``h // (hq // hk)``. Returns [batch, seq_q, hq * dim] to match the
    production output layout.
    """
    q = np.asarray(query, dtype=np.float64)
    k = np.asarray(key, dtype=np.float64)
    v = np.asarray(value, dtype=np.float64)
    b, hq, lq, d = q.shape
    _, hk, lk, _ = k.shape
    assert hq % hk == 0, "query heads must be a multiple of kv heads"
    n_rep = hq // hk
    out = np.zeros((b, hq, lq, d), dtype=np.float64)
    for bi in range(b):
        for h in range(hq):
            kh = h // n_rep
            scores = (q[bi, h] @ k[bi, kh].T) * scale  # [lq, lk]
            if additive_mask is not None:
                mm = np.asarray(additive_mask, dtype=np.float64)[bi]
                mh = mm[h] if mm.shape[0] == hq else mm[0]
                scores = scores + mh
            if is_causal:
                idx = np.arange(lq)[:, None]
                jdx = np.arange(lk)[None, :]
                # Bottom-right aligned causal mask: query i sees key j <= i+(lk-lq).
                blocked = jdx > (idx + (lk - lq))
                scores = np.where(blocked, -np.inf, scores)
            scores = scores - np.max(scores, axis=-1, keepdims=True)
            e = np.exp(scores)
            probs = e / np.sum(e, axis=-1, keepdims=True)
            out[bi, h] = probs @ v[bi, kh]
    return out.transpose(0, 2, 1, 3).reshape(b, lq, hq * d)


def _qkv(b, hq, hkv, lq, lk, d, seed):
    """Distinguishable, reproducible Q/K/V in [batch, heads, seq, dim] layout."""
    rng = np.random.RandomState(seed)
    q = rng.rand(b, hq, lq, d).astype(np.float32) - 0.5
    k = rng.rand(b, hkv, lk, d).astype(np.float32) - 0.5
    v = rng.rand(b, hkv, lk, d).astype(np.float32) - 0.5
    return q, k, v


class TestSDPAAttentionForward(unittest.TestCase):
    """Real forward-behavior tests for sdpa_attention_forward on CPU."""

    @classmethod
    def setUpClass(cls):
        # SDPA is exercisable on CPU; probe the raw paddle op once so that an
        # unsupported backend is reported as a skip (not swallowed per-test).
        paddle.set_device("cpu")
        cls._skip_reason = None
        probe = paddle.to_tensor(np.zeros((1, 2, 1, 4), dtype=np.float32))
        try:
            paddle.nn.functional.scaled_dot_product_attention(
                probe,
                probe,
                probe,
                None,
                0.0,
                is_causal=False,
                training=False,
                enable_gqa=True,
            )
        except (RuntimeError, NotImplementedError, OSError) as exc:
            cls._skip_reason = (
                "paddle scaled_dot_product_attention backend unavailable on "
                f"CPU: {exc}"
            )

    def setUp(self):
        if self._skip_reason:
            self.skipTest(self._skip_reason)
        paddle.set_device("cpu")

    def _forward(self, module, q, k, v, **kwargs):
        out, weights = sdpa_attention_forward(
            module,
            paddle.to_tensor(q),
            paddle.to_tensor(k),
            paddle.to_tensor(v),
            **kwargs,
        )
        return out, weights

    def test_full_attention_matches_manual_softmax(self):
        """Non-causal, no-mask output equals hand-computed softmax(QK^T/sqrt(d))V,
        returns None weights, and reshapes to [batch, seq, heads*dim]."""
        b, h, lq, lk, d = 1, 1, 3, 3, 4
        q, k, v = _qkv(b, h, h, lq, lk, d, seed=1)
        scale = d**-0.5
        module = _AttnModule(is_causal=True)
        module.eval()
        out, weights = self._forward(module, q, k, v, is_causal=False)

        self.assertIsNone(weights)
        self.assertEqual(list(out.shape), [b, lq, h * d])
        ref = _reference_sdpa(q, k, v, scale, is_causal=False)
        np.testing.assert_allclose(
            out.numpy().astype(np.float64), ref, atol=1e-4, rtol=1e-4
        )

    def test_causal_masking_restricts_visibility(self):
        """is_causal inferred True (seq>1, no mask, module.is_causal). Query 0 may
        only see key 0, so its output row equals value[0]; all rows match the
        independent causal reference."""
        b, h, seq, d = 1, 1, 4, 4
        q, k, v = _qkv(b, h, h, seq, seq, d, seed=2)
        scale = d**-0.5
        module = _AttnModule(is_causal=True)
        module.eval()
        out, _ = self._forward(module, q, k, v)

        ref = _reference_sdpa(q, k, v, scale, is_causal=True)
        np.testing.assert_allclose(
            out.numpy().astype(np.float64), ref, atol=1e-4, rtol=1e-4
        )
        # Position 0 attends to key 0 only -> output row 0 must equal value[0].
        np.testing.assert_allclose(
            out.numpy()[0, 0].astype(np.float64),
            v[0, 0, 0].astype(np.float64),
            atol=1e-4,
            rtol=1e-4,
        )
        # A causal bug that let position 0 also see later keys would move row 0
        # away from value[0]; confirm value[1] is distinct so this is a real check.
        self.assertGreater(float(np.abs(v[0, 0, 0] - v[0, 0, 1]).max()), 1e-2)

    def test_attention_mask_blocks_key_position(self):
        """An additive mask of -1e9 on one key column removes that key from every
        query's context: output matches the reference computed with the mask, and
        is invariant to the masked key's value."""
        b, h, lq, lk, d = 1, 1, 2, 3, 4
        q, k, v = _qkv(b, h, h, lq, lk, d, seed=3)
        scale = d**-0.5
        mask = np.zeros((b, 1, lq, lk), dtype=np.float32)
        mask[:, :, :, 1] = -1e9  # block key index 1 for all queries
        module = _AttnModule(is_causal=True)
        module.eval()

        out, _ = self._forward(
            module, q, k, v, attention_mask=paddle.to_tensor(mask)
        )
        ref = _reference_sdpa(
            q, k, v, scale, is_causal=False, additive_mask=mask
        )
        np.testing.assert_allclose(
            out.numpy().astype(np.float64), ref, atol=1e-4, rtol=1e-4
        )

        # Replacing the blocked key's value must not change the output.
        v2 = v.copy()
        v2[0, 0, 1] = v2[0, 0, 1] + 123.0
        out2, _ = self._forward(
            module, q, k, v2, attention_mask=paddle.to_tensor(mask)
        )
        np.testing.assert_allclose(
            out.numpy().astype(np.float64),
            out2.numpy().astype(np.float64),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_multihead_output_layout_correspondence(self):
        """With distinguishable per-head Q/K/V, each head's [dim] block in the
        flattened [seq, heads*dim] output equals that head's own attention,
        catching head-swap / reshape-order bugs."""
        b, h, seq, d = 1, 2, 3, 3
        q, k, v = _qkv(b, h, h, seq, seq, d, seed=4)
        scale = d**-0.5
        module = _AttnModule(is_causal=True)
        module.eval()
        out, _ = self._forward(module, q, k, v, is_causal=False)
        out_np = out.numpy().astype(np.float64)
        self.assertEqual(list(out.shape), [b, seq, h * d])

        for hi in range(h):
            head_ref = _reference_sdpa(
                q[:, hi : hi + 1],
                k[:, hi : hi + 1],
                v[:, hi : hi + 1],
                scale,
                is_causal=False,
            )  # [b, seq, d]
            np.testing.assert_allclose(
                out_np[:, :, hi * d : (hi + 1) * d],
                head_ref,
                atol=1e-4,
                rtol=1e-4,
            )

    def test_gqa_broadcasts_shared_kv(self):
        """enable_gqa: 2 query heads sharing 1 KV head each attend to the same
        keys/values but with their own queries. Output matches the reference that
        broadcasts the single KV head, and the two head blocks differ."""
        b, hq, hkv, seq, d = 1, 2, 1, 3, 4
        q, k, v = _qkv(b, hq, hkv, seq, seq, d, seed=5)
        scale = d**-0.5
        module = _AttnModule(is_causal=True)
        module.eval()
        out, _ = self._forward(module, q, k, v, is_causal=False)
        out_np = out.numpy().astype(np.float64)
        self.assertEqual(list(out.shape), [b, seq, hq * d])

        ref = _reference_sdpa(q, k, v, scale, is_causal=False)
        np.testing.assert_allclose(out_np, ref, atol=1e-4, rtol=1e-4)
        # Distinct queries -> the two head blocks must not be identical.
        self.assertGreater(
            float(np.abs(out_np[:, :, :d] - out_np[:, :, d:]).max()), 1e-3
        )

    def test_single_token_query_not_causal(self):
        """A single query position (seq_q == 1) infers is_causal=False and attends
        over all keys; output equals the full non-causal softmax reference."""
        b, h, lq, lk, d = 1, 1, 1, 3, 4
        q, k, v = _qkv(b, h, h, lq, lk, d, seed=6)
        scale = d**-0.5
        module = _AttnModule(is_causal=True)
        module.eval()
        out, _ = self._forward(module, q, k, v)

        self.assertEqual(list(out.shape), [b, lq, h * d])
        ref = _reference_sdpa(q, k, v, scale, is_causal=False)
        np.testing.assert_allclose(
            out.numpy().astype(np.float64), ref, atol=1e-4, rtol=1e-4
        )


if __name__ == "__main__":
    unittest.main()
