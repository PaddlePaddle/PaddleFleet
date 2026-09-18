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

"""No-card behavior tests for two pure pieces of ``dot_product_attention``:

* ``scaled_dot_product_attention_with_softmax_offset`` -- the manual
  sink-softmax attention used when a per-head ``softmax_offset`` is present.
  It is CPU-computable (plain paddle matmul/exp), so this file drives the real
  entry point and compares against an *independent* fp64 numpy reference plus a
  literal hand-derived anchor. The reference expands K/V by ``np.repeat`` for
  GQA -- a different strategy than the production reshape-into-groups path --
  so a wrong group<->kv-head mapping would be caught rather than mirrored.

* ``build_softmax_offset`` -- the switch that turns ``softmax_type`` /
  ``add_*_attention_sink_bias`` / ``is_swa`` into ``None`` (vanilla), a fixed
  zeros tensor (off-by-one) or a trainable per-head parameter (learnable), with
  ``is_swa`` gating which sink-bias flag promotes to learnable.

Both entry points only need paddle on CPU. When paddle (or paddlefleet) is not
importable here the whole module skips with the honest reason; only
ImportError/ModuleNotFoundError is treated as "dependency missing".
"""

import math
import unittest

try:
    import numpy as np
    import paddle

    from paddlefleet.transformer.dot_product_attention import (
        build_softmax_offset,
        scaled_dot_product_attention_with_softmax_offset,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # no paddle / paddlefleet
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    f"paddle/paddlefleet not importable in this environment: {_IMPORT_ERROR}"
)


def _ref_sink_attention(
    query, key, value, scale, sink, is_causal=False, attn_mask_kv=None
):
    """Independent fp64 numpy reference for the sink-softmax attention.

    Inputs are numpy arrays laid out like the production entry point:
    ``query`` [B, Q, Hq, dq], ``key``/``value`` [B, K, Hkv, d]. ``sink`` is a
    per-q-head [Hq] logit. Mirrors the documented algorithm

        scores  = Q @ K^T * scale
        row_max = max(scores.max(-1, keepdim), sink)
        exp_s   = exp(scores - row_max)
        row_sum = exp_s.sum(-1, keepdim) + exp(sink - row_max)   # virtual token
        weights = exp_s / row_sum                                # sum(weights) < 1
        out     = weights @ V

    GQA is handled by repeating K/V so that q-head ``h`` reads kv-head
    ``h // groups`` -- deliberately a different mechanism than production's
    reshape, to avoid sharing an implementation bug with the code under test.
    """
    q = query.astype(np.float64).transpose(0, 2, 1, 3)  # B, Hq, Q, dq
    k = key.astype(np.float64).transpose(0, 2, 1, 3)  # B, Hkv, K, dk
    v = value.astype(np.float64).transpose(0, 2, 1, 3)  # B, Hkv, K, dv

    b, hq, q_len, _ = q.shape
    hkv = k.shape[1]
    groups = hq // hkv
    if groups > 1:
        k = np.repeat(k, groups, axis=1)  # kv-head for q-head h is h // groups
        v = np.repeat(v, groups, axis=1)

    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale  # B, Hq, Q, K
    k_len = scores.shape[-1]

    if is_causal and q_len > 1:
        allowed = np.tril(np.ones((q_len, k_len)), k=k_len - q_len)
        scores = np.where(allowed[None, None] == 0, -np.inf, scores)

    if attn_mask_kv is not None:
        m = np.asarray(attn_mask_kv)
        if m.dtype == np.bool_:
            scores = np.where(m, -np.inf, scores)
        else:
            scores = scores + m.astype(np.float64)

    sink_r = np.asarray(sink, dtype=np.float64).reshape(1, hq, 1, 1)
    row_max = np.maximum(scores.max(axis=-1, keepdims=True), sink_r)
    exp_s = np.exp(scores - row_max)
    row_sum = exp_s.sum(axis=-1, keepdims=True) + np.exp(sink_r - row_max)
    weights = exp_s / row_sum
    out = np.matmul(weights, v)  # B, Hq, Q, dv
    return out.transpose(0, 2, 1, 3)  # B, Q, Hq, dv


def _plain_softmax_attention(query, key, value, scale):
    """Independent fp64 reference for ordinary (sink-free) attention."""
    q = query.astype(np.float64).transpose(0, 2, 1, 3)
    k = key.astype(np.float64).transpose(0, 2, 1, 3)
    v = value.astype(np.float64).transpose(0, 2, 1, 3)
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
    scores = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis=-1, keepdims=True)
    out = np.matmul(w, v)
    return out.transpose(0, 2, 1, 3)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestSinkSoftmaxAttentionNumeric(unittest.TestCase):
    """scaled_dot_product_attention_with_softmax_offset numeric behavior."""

    def setUp(self):
        # Sink-softmax is device-independent math; run it on CPU here.
        paddle.set_device("cpu")

    def _t(self, arr):
        return paddle.to_tensor(np.asarray(arr, dtype="float32"))

    def test_mha_matches_hand_derived_anchor(self):
        # B=1, Q=1, K=2, single head, dim=2, scale=1.0, sink logit = 0.
        #   q = [1, 0]
        #   k0 = [1, 0], k1 = [0, 1]  -> scores = [1.0, 0.0]
        #   v0 = [1, 2], v1 = [3, 5]
        # row_max = max(1, sink=0) = 1
        # exp_s   = [e^0, e^-1] = [1, 0.3678794412]
        # row_sum = (1 + 0.3678794412) + e^(0-1) = 1.7357588823
        # w0 = 1 / 1.7357588823, w1 = e^-1 / 1.7357588823
        # out = [w0*1 + w1*3, w0*2 + w1*5]
        query = self._t([[[[1.0, 0.0]]]])  # [1,1,1,2]
        key = self._t([[[[1.0, 0.0]], [[0.0, 1.0]]]])  # [1,2,1,2]
        value = self._t([[[[1.0, 2.0]], [[3.0, 5.0]]]])  # [1,2,1,2]
        sink = self._t([0.0])  # one head

        out = scaled_dot_product_attention_with_softmax_offset(
            query, key, value, softmax_offset=sink, scale=1.0
        )

        # Hand math with the stdlib, independent of numpy and of production.
        e = math.exp(-1.0)
        row_sum = (1.0 + e) + e
        w0, w1 = 1.0 / row_sum, e / row_sum
        expected = [w0 * 1.0 + w1 * 3.0, w0 * 2.0 + w1 * 5.0]
        got = out.numpy().reshape(-1).tolist()
        self.assertEqual(list(out.shape), [1, 1, 1, 2])
        self.assertAlmostEqual(got[0], expected[0], places=6)
        self.assertAlmostEqual(got[1], expected[1], places=6)
        # Coarse literal cross-check so the anchor is pinned to numbers, not
        # only to a formula: expected ~= [1.211991, 2.212024].
        self.assertAlmostEqual(got[0], 1.212, places=3)
        self.assertAlmostEqual(got[1], 2.212, places=3)

    def test_mha_against_independent_reference(self):
        rng = np.random.RandomState(0)
        q = rng.randn(2, 5, 4, 8).astype("float32")
        k = rng.randn(2, 5, 4, 8).astype("float32")
        v = rng.randn(2, 5, 4, 8).astype("float32")
        sink = rng.randn(4).astype("float32")  # distinct per-head sink logits
        scale = 0.35

        out = scaled_dot_product_attention_with_softmax_offset(
            self._t(q),
            self._t(k),
            self._t(v),
            softmax_offset=self._t(sink),
            scale=scale,
        )
        ref = _ref_sink_attention(q, k, v, scale, sink)
        np.testing.assert_allclose(
            out.numpy().astype("float64"), ref, rtol=1e-5, atol=1e-5
        )

    def test_gqa_group_to_kv_head_mapping(self):
        # Hq=6, Hkv=2 -> groups=3. Distinct per-head content so a wrong
        # group<->kv mapping (or expanding V instead of grouping) diverges.
        rng = np.random.RandomState(1)
        q = rng.randn(2, 4, 6, 8).astype("float32")
        k = rng.randn(2, 4, 2, 8).astype("float32")
        v = rng.randn(2, 4, 2, 8).astype("float32")
        sink = rng.randn(6).astype("float32")
        scale = 8**-0.5

        out = scaled_dot_product_attention_with_softmax_offset(
            self._t(q),
            self._t(k),
            self._t(v),
            softmax_offset=self._t(sink),
            q_head_dim=8,
            scale=scale,
        )
        ref = _ref_sink_attention(q, k, v, scale, sink)
        np.testing.assert_allclose(
            out.numpy().astype("float64"), ref, rtol=1e-5, atol=1e-5
        )

    def test_causal_mask_blocks_future_positions(self):
        rng = np.random.RandomState(2)
        q = rng.randn(1, 4, 2, 8).astype("float32")
        k = rng.randn(1, 4, 2, 8).astype("float32")
        v = rng.randn(1, 4, 2, 8).astype("float32")
        sink = np.zeros(2, dtype="float32")
        scale = 8**-0.5

        out = scaled_dot_product_attention_with_softmax_offset(
            self._t(q),
            self._t(k),
            self._t(v),
            is_causal=True,
            softmax_offset=self._t(sink),
            scale=scale,
        )
        ref = _ref_sink_attention(q, k, v, scale, sink, is_causal=True)
        np.testing.assert_allclose(
            out.numpy().astype("float64"), ref, rtol=1e-5, atol=1e-5
        )
        # A causal query at position 0 must not depend on future keys: perturbing
        # only key/value positions >0 leaves the row-0 output unchanged.
        k2 = k.copy()
        v2 = v.copy()
        k2[:, 1:] += 3.0
        v2[:, 1:] += 7.0
        out2 = scaled_dot_product_attention_with_softmax_offset(
            self._t(q),
            self._t(k2),
            self._t(v2),
            is_causal=True,
            softmax_offset=self._t(sink),
            scale=scale,
        )
        np.testing.assert_allclose(
            out.numpy()[:, 0], out2.numpy()[:, 0], rtol=1e-5, atol=1e-5
        )
        # ...while a later position DOES change (guards against the mask
        # accidentally blocking everything).
        self.assertGreater(
            float(np.abs(out.numpy()[:, -1] - out2.numpy()[:, -1]).max()),
            1e-3,
        )

    def test_bool_attn_mask_kv_forces_minus_inf(self):
        rng = np.random.RandomState(3)
        q = rng.randn(1, 2, 2, 8).astype("float32")
        k = rng.randn(1, 3, 2, 8).astype("float32")
        v = rng.randn(1, 3, 2, 8).astype("float32")
        sink = np.zeros(2, dtype="float32")
        scale = 8**-0.5
        # [B, Hq, Q, K] bool mask, True == masked out. Block the last key.
        mask = np.zeros((1, 2, 2, 3), dtype=bool)
        mask[..., 2] = True

        out = scaled_dot_product_attention_with_softmax_offset(
            self._t(q),
            self._t(k),
            self._t(v),
            attn_mask_kv=paddle.to_tensor(mask),
            softmax_offset=self._t(sink),
            scale=scale,
        )
        ref = _ref_sink_attention(q, k, v, scale, sink, attn_mask_kv=mask)
        np.testing.assert_allclose(
            out.numpy().astype("float64"), ref, rtol=1e-5, atol=1e-5
        )

    def test_negligible_sink_reduces_to_plain_softmax(self):
        # A very negative sink logit contributes exp(sink-row_max) ~= 0 to the
        # denominator, so the output must collapse onto ordinary softmax.
        rng = np.random.RandomState(4)
        q = rng.randn(1, 3, 2, 8).astype("float32")
        k = rng.randn(1, 3, 2, 8).astype("float32")
        v = rng.randn(1, 3, 2, 8).astype("float32")
        scale = 8**-0.5
        sink = np.full(2, -1e9, dtype="float32")

        out = scaled_dot_product_attention_with_softmax_offset(
            self._t(q),
            self._t(k),
            self._t(v),
            softmax_offset=self._t(sink),
            scale=scale,
        )
        plain = _plain_softmax_attention(q, k, v, scale)
        np.testing.assert_allclose(
            out.numpy().astype("float64"), plain, rtol=1e-5, atol=1e-5
        )

    def test_dominant_sink_starves_real_tokens(self):
        # A large positive sink logit wins the softmax denominator, driving all
        # real-token weights toward zero -> output ~= 0. This is exactly what an
        # additive mask could NOT do, so it proves the sink is consumed as a
        # virtual token rather than a plain bias.
        rng = np.random.RandomState(5)
        q = rng.randn(1, 3, 2, 8).astype("float32")
        k = rng.randn(1, 3, 2, 8).astype("float32")
        v = rng.randn(1, 3, 2, 8).astype("float32")
        scale = 8**-0.5
        sink = np.full(2, 60.0, dtype="float32")

        out = scaled_dot_product_attention_with_softmax_offset(
            self._t(q),
            self._t(k),
            self._t(v),
            softmax_offset=self._t(sink),
            scale=scale,
        )
        self.assertLess(float(np.abs(out.numpy()).max()), 1e-10)
        # Sanity: the same inputs with a negligible sink produce a clearly
        # non-zero output, so the starvation above is caused by the sink.
        out_ref = scaled_dot_product_attention_with_softmax_offset(
            self._t(q),
            self._t(k),
            self._t(v),
            softmax_offset=self._t(np.full(2, -1e9, dtype="float32")),
            scale=scale,
        )
        self.assertGreater(float(np.abs(out_ref.numpy()).max()), 1e-2)


if _IMPORT_ERROR is None:
    from types import SimpleNamespace

    class _ProbeLayer(paddle.nn.Layer):
        """Real Layer so ``build_softmax_offset`` calls a genuine
        ``create_parameter`` (learnable path) rather than a stub."""

    def _cfg(
        softmax_type="vanilla",
        add_full=False,
        add_swa=False,
        perform_init=False,
        init_method=None,
        params_dtype="float32",
    ):
        return SimpleNamespace(
            softmax_type=softmax_type,
            add_full_attention_sink_bias=add_full,
            add_swa_attention_sink_bias=add_swa,
            perform_initialization=perform_init,
            init_method=init_method,
            params_dtype=params_dtype,
        )


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestBuildSoftmaxOffset(unittest.TestCase):
    """build_softmax_offset: softmax_type + sink-bias promotion + is_swa gate."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_vanilla_returns_none(self):
        self.assertIsNone(
            build_softmax_offset(
                _ProbeLayer(), _cfg("vanilla"), 4, is_swa=False
            )
        )

    def test_off_by_one_returns_fixed_zeros(self):
        out = build_softmax_offset(
            _ProbeLayer(), _cfg("off-by-one"), 4, is_swa=False
        )
        self.assertIsNotNone(out)
        self.assertEqual(list(out.shape), [4])
        # Fixed sink: all zeros, and not a trainable parameter.
        np.testing.assert_array_equal(out.numpy(), np.zeros(4, dtype="float32"))
        self.assertNotIsInstance(out, paddle.base.framework.EagerParamBase)

    def test_learnable_creates_trainable_parameter(self):
        layer = _ProbeLayer()
        out = build_softmax_offset(
            layer, _cfg("learnable", perform_init=False), 5, is_swa=False
        )
        self.assertIsNotNone(out)
        self.assertEqual(list(out.shape), [5])
        # A create_parameter() result is a trainable EagerParamBase.
        self.assertIsInstance(out, paddle.base.framework.EagerParamBase)
        self.assertFalse(out.stop_gradient)

    def test_learnable_runs_init_method_on_created_param(self):
        # perform_initialization=True must feed the freshly created parameter
        # through init_method; assert the parameter actually carries the value
        # the init_method wrote (parameter effect observed, not just "called").
        def init_method(p):
            with paddle.no_grad():
                p.set_value(paddle.full(p.shape, 0.25, dtype=p.dtype))

        out = build_softmax_offset(
            _ProbeLayer(),
            _cfg("learnable", perform_init=True, init_method=init_method),
            3,
            is_swa=False,
        )
        np.testing.assert_allclose(
            out.numpy(), np.full(3, 0.25, dtype="float32"), rtol=0, atol=0
        )

    def test_learnable_skips_init_when_perform_initialization_false(self):
        def boom(p):
            raise AssertionError(
                "init_method must not run when perform_initialization is False"
            )

        # Must not raise: the perform_initialization gate skips init_method.
        out = build_softmax_offset(
            _ProbeLayer(),
            _cfg("learnable", perform_init=False, init_method=boom),
            3,
            is_swa=False,
        )
        self.assertIsInstance(out, paddle.base.framework.EagerParamBase)

    def test_invalid_softmax_type_raises_value_error(self):
        with self.assertRaises(ValueError):
            build_softmax_offset(
                _ProbeLayer(), _cfg("bogus-type"), 4, is_swa=False
            )

    def test_full_sink_bias_promotes_only_when_not_swa(self):
        # add_full_attention_sink_bias promotes vanilla -> learnable, but ONLY
        # for a non-SWA layer. On an SWA layer the same flag must not promote.
        promoted = build_softmax_offset(
            _ProbeLayer(),
            _cfg("vanilla", add_full=True),
            4,
            is_swa=False,
        )
        self.assertIsInstance(promoted, paddle.base.framework.EagerParamBase)

        not_promoted = build_softmax_offset(
            _ProbeLayer(),
            _cfg("vanilla", add_full=True),
            4,
            is_swa=True,
        )
        self.assertIsNone(not_promoted)

    def test_swa_sink_bias_promotes_only_when_swa(self):
        # Symmetric: add_swa_attention_sink_bias promotes only for is_swa=True.
        promoted = build_softmax_offset(
            _ProbeLayer(),
            _cfg("vanilla", add_swa=True),
            4,
            is_swa=True,
        )
        self.assertIsInstance(promoted, paddle.base.framework.EagerParamBase)

        not_promoted = build_softmax_offset(
            _ProbeLayer(),
            _cfg("vanilla", add_swa=True),
            4,
            is_swa=False,
        )
        self.assertIsNone(not_promoted)


if __name__ == "__main__":
    unittest.main()
