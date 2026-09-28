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

"""Behavior tests for the softmax-offset (attention-sink) attention in
``paddlefleet.transformer.dot_product_attention``.

The production function ``scaled_dot_product_attention_with_softmax_offset``
implements an *off-by-one* / learnable-sink softmax: a virtual sink token
contributes ``exp(sink - row_max)`` to the denominator only, so the real
attention weights sum to < 1 (part of the mass is absorbed by the sink). It is
NOT equivalent to an additive mask, which softmax cancels out.

Every expected value here is recomputed independently in NumPy (float64) from
Q/K/V, the per-head sink value and the scale. The references never call the
code under test, so a wrong scale, a missing sink term, a wrong GQA head
mapping or a wrong causal alignment would be rejected.

No paddle/GPU is required by this environment; imports are guarded and the
tests skip honestly when paddle is unavailable. The exercised math is pure
CPU float32 (matmul/exp), so on a real single-card (or CPU-with-paddle) host
these tests execute rather than skip.
"""

import unittest

import numpy as np

try:
    import paddle

    from paddlefleet.transformer.dot_product_attention import (
        build_softmax_offset,
        scaled_dot_product_attention_with_softmax_offset,
    )

    PADDLE_IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest: no paddle here
    paddle = None
    build_softmax_offset = None
    scaled_dot_product_attention_with_softmax_offset = None
    PADDLE_IMPORT_ERROR = exc

# A sink so negative that exp(sink - row_max) underflows to 0, i.e. the
# off-by-one softmax reduces to a plain softmax. Used to neutralize the sink.
NEG_SINK = -1.0e18

_SKIP = PADDLE_IMPORT_ERROR is not None
_REASON = f"paddle unavailable: {PADDLE_IMPORT_ERROR}"


def _ref_sink_attention(
    q,
    k,
    v,
    offset,
    scale,
    is_causal=False,
    mask=None,
    mask_is_bool=False,
    head_map="interleave",
):
    """Independent NumPy reference for sink-softmax attention (float64).

    q:[B,Q,Hq,dq] k:[B,K,Hkv,dk] v:[B,K,Hkv,dv] offset:[Hq]. Returns [B,Q,Hq,dv].
    ``head_map`` selects how a q-head picks its kv-head. Production reshapes Q
    into ``groups`` per kv-head, i.e. repeat-interleave semantics
    (kv = h // groups); "tile" (kv = h % Hkv) is the wrong mapping used only to
    prove the test discriminates.
    """
    q = np.asarray(q, np.float64)
    k = np.asarray(k, np.float64)
    v = np.asarray(v, np.float64)
    offset = np.asarray(offset, np.float64)
    B, Q, Hq, _dq = q.shape
    _, Klen, Hkv, dv = v.shape
    groups = Hq // Hkv
    out = np.zeros((B, Q, Hq, dv), np.float64)
    for b in range(B):
        for h in range(Hq):
            kv = (h // groups) if head_map == "interleave" else (h % Hkv)
            scores = (q[b, :, h, :] @ k[b, :, kv, :].T) * scale  # [Q, K]
            if is_causal and Q > 1:
                diag = Klen - Q  # bottom-right causal alignment
                jj = np.arange(Klen)[None, :]
                ii = np.arange(Q)[:, None]
                scores = np.where(jj > ii + diag, -np.inf, scores)
            if mask is not None:
                m = np.asarray(mask)
                mh = m[0, 0] if m.shape[1] == 1 else m[0, h]  # [Q, K]
                if mask_is_bool:
                    scores = np.where(mh.astype(bool), -np.inf, scores)
                else:
                    scores = scores + mh.astype(np.float64)
            s = float(offset[h])
            row_max = np.maximum(scores.max(axis=-1, keepdims=True), s)
            exp_s = np.exp(scores - row_max)
            row_sum = exp_s.sum(axis=-1, keepdims=True) + np.exp(s - row_max)
            weights = exp_s / row_sum
            out[b, :, h, :] = weights @ v[b, :, kv, :]
    return out


def _ref_plain_softmax(q, k, v, scale):
    """Plain softmax attention (NO sink term) — deliberately a different
    formula so the negative-sink reduction test is not self-referential."""
    q = np.asarray(q, np.float64)
    k = np.asarray(k, np.float64)
    v = np.asarray(v, np.float64)
    B, Q, Hq, _ = q.shape
    _, Klen, Hkv, dv = v.shape
    groups = Hq // Hkv
    out = np.zeros((B, Q, Hq, dv), np.float64)
    for b in range(B):
        for h in range(Hq):
            kv = h // groups
            scores = (q[b, :, h, :] @ k[b, :, kv, :].T) * scale
            e = np.exp(scores - scores.max(axis=-1, keepdims=True))
            w = e / e.sum(axis=-1, keepdims=True)
            out[b, :, h, :] = w @ v[b, :, kv, :]
    return out


def _run(
    q,
    k,
    v,
    offset,
    scale=None,
    is_causal=False,
    mask=None,
    mask_bool=False,
    dropout_p=0.0,
    training=False,
    seed=None,
):
    """Call the production function on fixed inputs, return float64 output."""
    if seed is not None:
        paddle.seed(seed)
    q_head_dim = int(np.asarray(q).shape[-1])
    kwargs = {
        "softmax_offset": paddle.to_tensor(np.asarray(offset), dtype="float32"),
        "q_head_dim": q_head_dim,
        "is_causal": is_causal,
        "dropout_p": dropout_p,
        "training": training,
    }
    if scale is not None:
        kwargs["scale"] = scale
    if mask is not None:
        kwargs["attn_mask_kv"] = paddle.to_tensor(
            np.asarray(mask), dtype="bool" if mask_bool else "float32"
        )
    out = scaled_dot_product_attention_with_softmax_offset(
        paddle.to_tensor(np.asarray(q), dtype="float32"),
        paddle.to_tensor(np.asarray(k), dtype="float32"),
        paddle.to_tensor(np.asarray(v), dtype="float32"),
        **kwargs,
    )
    return np.asarray(out.numpy(), np.float64)


def _default_scale(q):
    return float(np.asarray(q).shape[-1]) ** -0.5


@unittest.skipIf(_SKIP, _REASON)
class _Base(unittest.TestCase):
    def _close(self, actual, ref, rtol=2e-5, atol=1e-5):
        actual = np.asarray(actual, np.float64)
        ref = np.asarray(ref, np.float64)
        self.assertTrue(np.isfinite(actual).all(), "output not finite")
        self.assertTrue(np.isfinite(ref).all(), "reference not finite")
        np.testing.assert_allclose(actual, ref, rtol=rtol, atol=atol)

    def _not_close(self, actual, other, rtol=1e-3, atol=1e-3):
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(
                np.asarray(actual, np.float64),
                np.asarray(other, np.float64),
                rtol=rtol,
                atol=atol,
            )


class TestSinkSoftmaxNumerics(_Base):
    # Fixed, distinguishable MHA inputs: B=1, Q=2, K=3, H=1, d=2.
    Q = [[[[1.0, 2.0]], [[-1.0, 0.5]]]]
    K = [[[[0.5, 1.0]], [[1.0, -1.0]], [[2.0, 0.3]]]]
    V = [[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]]]

    def test_offbyone_has_sink_term_not_plain_softmax(self):
        scale = _default_scale(self.Q)
        out = _run(self.Q, self.K, self.V, offset=[0.0])
        ref_sink = _ref_sink_attention(self.Q, self.K, self.V, [0.0], scale)
        ref_plain = _ref_plain_softmax(self.Q, self.K, self.V, scale)
        # Off-by-one output must match the independent sink reference ...
        self._close(out, ref_sink)
        # ... and must NOT equal plain softmax (proves the sink term is present
        # and real weights sum to < 1; deleting the +exp(sink-row_max) fails).
        self._not_close(out, ref_plain)

    def test_negative_sink_reduces_to_plain_softmax(self):
        scale = _default_scale(self.Q)
        out = _run(self.Q, self.K, self.V, offset=[NEG_SINK])
        ref_plain = _ref_plain_softmax(self.Q, self.K, self.V, scale)
        self._close(out, ref_plain)

    def test_large_sink_absorbs_almost_all_weight(self):
        scale = _default_scale(self.Q)
        out = _run(self.Q, self.K, self.V, offset=[50.0])
        ref_sink = _ref_sink_attention(self.Q, self.K, self.V, [50.0], scale)
        ref_plain = _ref_plain_softmax(self.Q, self.K, self.V, scale)
        self._close(out, ref_sink)
        # Sink >> scores => output collapses toward zero, while a plain softmax
        # over the same inputs would produce O(1) output. This isolates the
        # smallness to the sink rather than to trivial inputs.
        self.assertLess(np.abs(out).max(), 1e-6)
        self.assertGreater(np.abs(ref_plain).max(), 0.1)


class TestSinkGQA(_Base):
    def test_gqa_uses_repeat_interleave_head_mapping(self):
        rng = np.random.RandomState(0)
        b, q_len, kv_len, d = 1, 3, 5, 4
        num_q, num_kv = 4, 2  # groups = 2
        # Distinguishable per-kv-head content so the q->kv mapping is testable.
        q = rng.randn(b, q_len, num_q, d)
        k = rng.randn(b, kv_len, num_kv, d)
        v = rng.randn(b, kv_len, num_kv, d)
        offset = np.zeros(num_q)  # off-by-one sink, per q-head
        scale = _default_scale(q)

        out = _run(q, k, v, offset=offset)
        ref_interleave = _ref_sink_attention(
            q, k, v, offset, scale, head_map="interleave"
        )
        ref_tile = _ref_sink_attention(q, k, v, offset, scale, head_map="tile")
        self.assertEqual(list(out.shape), [b, q_len, num_q, d])
        self._close(out, ref_interleave)
        # A tile mapping (kv = h % Hkv) would pair q-heads with the wrong
        # kv-heads; production must NOT match it.
        self._not_close(out, ref_tile)


class TestSinkCausal(_Base):
    def test_causal_uses_bottom_right_diagonal_alignment(self):
        rng = np.random.RandomState(1)
        b, q_len, kv_len, d = 1, 2, 4, 4
        q = rng.randn(b, q_len, 1, d)
        k = rng.randn(b, kv_len, 1, d)
        v = rng.randn(b, kv_len, 1, d)
        offset = np.zeros(1)
        scale = _default_scale(q)

        out = _run(q, k, v, offset=offset, is_causal=True)
        ref_correct = _ref_sink_attention(
            q, k, v, offset, scale, is_causal=True
        )  # diag = kv_len - q_len = 2

        # Wrong (top-left, diagonal=0) causal reference for discrimination.
        def _ref_diag0():
            out0 = np.zeros((b, q_len, 1, d))
            sc = (q[0, :, 0, :] @ k[0, :, 0, :].T) * scale
            jj = np.arange(kv_len)[None, :]
            ii = np.arange(q_len)[:, None]
            sc = np.where(jj > ii, -np.inf, sc)
            rmax = np.maximum(sc.max(-1, keepdims=True), 0.0)
            e = np.exp(sc - rmax)
            den = e.sum(-1, keepdims=True) + np.exp(0.0 - rmax)
            out0[0, :, 0, :] = (e / den) @ v[0, :, 0, :]
            return out0

        self._close(out, ref_correct)
        self._not_close(out, _ref_diag0())


class TestSinkCausalSingleQuery(_Base):
    def test_causal_skipped_when_query_len_is_one(self):
        # query.shape[1] == 1 must NOT enter the causal branch: the single
        # (last) query attends to all cached keys. A bug that applied a plain
        # tril here would mask keys 1..K-1 and diverge.
        rng = np.random.RandomState(2)
        d = 4
        q = rng.randn(1, 1, 1, d)
        k = rng.randn(1, 3, 1, d)
        v = rng.randn(1, 3, 1, d)
        offset = np.zeros(1)
        scale = _default_scale(q)

        out = _run(q, k, v, offset=offset, is_causal=True)
        ref_full = _ref_sink_attention(q, k, v, offset, scale, is_causal=False)
        self._close(out, ref_full)


class TestSinkMask(_Base):
    Q = [[[[1.0, 2.0]], [[-1.0, 0.5]]]]  # [1,2,1,2]
    K = [[[[0.5, 1.0]], [[1.0, -1.0]], [[2.0, 0.3]]]]  # [1,3,1,2]
    V = [[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]]]

    def test_bool_mask_true_positions_are_masked(self):
        scale = _default_scale(self.Q)
        # True == masked (per production). Each query row keeps >=1 key.
        mask = [[[[False, True, False], [True, False, True]]]]
        out = _run(
            self.Q, self.K, self.V, offset=[0.0], mask=mask, mask_bool=True
        )
        ref = _ref_sink_attention(
            self.Q, self.K, self.V, [0.0], scale, mask=mask, mask_is_bool=True
        )
        self._close(out, ref)
        # Inverting the mask must change the result: proves True (not False) is
        # the masked value, and that the mask is actually consumed.
        inv = [[[[True, False, True], [False, True, False]]]]
        ref_inv = _ref_sink_attention(
            self.Q, self.K, self.V, [0.0], scale, mask=inv, mask_is_bool=True
        )
        self._not_close(out, ref_inv)

    def test_additive_float_mask_is_added_to_scores(self):
        scale = _default_scale(self.Q)
        # Finite additive bias (not -inf): checks the addition, not just
        # zeroing. A no-op or wrong-sign add would fail.
        mask = [[[[0.0, -3.0, 0.0], [-3.0, 0.0, -3.0]]]]
        out = _run(self.Q, self.K, self.V, offset=[0.0], mask=mask)
        ref = _ref_sink_attention(
            self.Q, self.K, self.V, [0.0], scale, mask=mask
        )
        ref_nomask = _ref_sink_attention(self.Q, self.K, self.V, [0.0], scale)
        self._close(out, ref)
        self._not_close(out, ref_nomask)


class TestSinkScaleArg(_Base):
    Q = [[[[1.0, 2.0]], [[-1.0, 0.5]]]]
    K = [[[[0.5, 1.0]], [[1.0, -1.0]], [[2.0, 0.3]]]]
    V = [[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]]]

    def test_explicit_scale_overrides_default(self):
        custom = 0.123  # deliberately != default d**-0.5 == 0.70710678
        out_custom = _run(self.Q, self.K, self.V, offset=[0.0], scale=custom)
        out_default = _run(self.Q, self.K, self.V, offset=[0.0])
        ref_custom = _ref_sink_attention(self.Q, self.K, self.V, [0.0], custom)
        self._close(out_custom, ref_custom)
        # Custom scale must actually be consumed (differs from the default).
        self._not_close(out_custom, out_default)


class TestSinkDropoutTrainingFlag(_Base):
    Q = [[[[1.0, 2.0]], [[-1.0, 0.5]]]]
    K = [[[[0.5, 1.0]], [[1.0, -1.0]], [[2.0, 0.3]]]]
    V = [[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]]]

    def test_dropout_is_noop_in_eval_and_active_in_training(self):
        # training=False => dropout must be a no-op regardless of dropout_p,
        # so it equals the dropout_p=0 result. This also guards the historical
        # regression where the code referenced an undefined `self.training`
        # (that would raise NameError before returning any output).
        out_eval = _run(
            self.Q, self.K, self.V, offset=[0.0], dropout_p=0.5, training=False
        )
        out_nodrop = _run(self.Q, self.K, self.V, offset=[0.0], dropout_p=0.0)
        self._close(out_eval, out_nodrop)

        # training=True with p=0.5 => dropout is applied, so the output changes
        # (with a fixed seed roughly half the weights are zeroed and the rest
        # rescaled). Proves the `training` argument is honored.
        out_train = _run(
            self.Q,
            self.K,
            self.V,
            offset=[0.0],
            dropout_p=0.5,
            training=True,
            seed=123,
        )
        self.assertTrue(np.isfinite(out_train).all())
        self.assertEqual(
            list(out_train.shape), list(np.asarray(out_eval).shape)
        )
        self._not_close(out_train, out_eval)


@unittest.skipIf(_SKIP, _REASON)
class TestBuildSoftmaxOffsetSelection(unittest.TestCase):
    """``build_softmax_offset`` decides whether the sink branch is taken at all
    (returns None => plain SDPA; a tensor => sink path). This wires config
    ``softmax_type`` and the sink-bias flags to the parameter forward reads."""

    def _cfg(self, **overrides):
        from types import SimpleNamespace

        base = {
            "softmax_type": "vanilla",
            "add_full_attention_sink_bias": False,
            "add_swa_attention_sink_bias": False,
            "params_dtype": "float32",
            "perform_initialization": False,
            "init_method": None,
        }
        base.update(overrides)
        return SimpleNamespace(**base)

    def _layer(self):
        return paddle.nn.Layer()

    def test_vanilla_returns_none(self):
        self.assertIsNone(
            build_softmax_offset(
                self._layer(), self._cfg(), num_heads=4, is_swa=False
            )
        )

    def test_off_by_one_returns_zeros_of_num_heads(self):
        off = build_softmax_offset(
            self._layer(),
            self._cfg(softmax_type="off-by-one"),
            num_heads=4,
            is_swa=False,
        )
        self.assertIsNotNone(off)
        self.assertEqual(list(off.shape), [4])
        np.testing.assert_array_equal(off.numpy(), np.zeros(4, np.float32))

    def test_full_sink_bias_forces_learnable_trainable_param(self):
        off = build_softmax_offset(
            self._layer(),
            self._cfg(add_full_attention_sink_bias=True),  # overrides vanilla
            num_heads=3,
            is_swa=False,
        )
        self.assertIsNotNone(off)
        self.assertEqual(list(off.shape), [3])
        self.assertFalse(off.stop_gradient)  # learnable parameter

    def test_full_sink_bias_is_gated_by_not_is_swa(self):
        # add_full_attention_sink_bias must NOT trigger for a sliding-window
        # layer; with softmax_type vanilla and no swa bias it stays None.
        self.assertIsNone(
            build_softmax_offset(
                self._layer(),
                self._cfg(add_full_attention_sink_bias=True),
                num_heads=3,
                is_swa=True,
            )
        )

    def test_swa_sink_bias_forces_learnable_for_swa_layer(self):
        off = build_softmax_offset(
            self._layer(),
            self._cfg(add_swa_attention_sink_bias=True),
            num_heads=2,
            is_swa=True,
        )
        self.assertIsNotNone(off)
        self.assertFalse(off.stop_gradient)

    def test_unknown_softmax_type_raises_value_error(self):
        with self.assertRaises(ValueError):
            build_softmax_offset(
                self._layer(),
                self._cfg(softmax_type="bogus"),
                num_heads=2,
                is_swa=False,
            )


if __name__ == "__main__":
    unittest.main()
