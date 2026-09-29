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
"""Behavior tests for the ``use_accuracy_compatible`` (megatron target) code
path in ``DotProductAttention.forward`` and the ``_EagerQKScoresFn`` PyLayer it
uses to compute QK scores.

Every expected value is derived by hand from the attention definition and the
production forward's documented arithmetic:

  scores  = softmax_scale * (Q @ K^T)          (``_EagerQKScoresFn``, baddbmm)
  masked  = where(mask, -10000.0, scores)       (``attention_mask_func``)
  probs   = softmax(masked, axis=-1)            (fp32, forced by the compat path)
  context = probs @ V

The reference never calls the code under test; it is a plain numpy/paddle
re-expression of the formula above, so a swapped operand, wrong mask direction,
missing scale, or a broken custom backward is rejected rather than mirrored.

The whole suite needs Paddle (and, for the layer paths, a single accelerator to
back the default TP process group). Imports are guarded so that a machine
without Paddle honestly skips instead of erroring or faking a pass.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

try:
    import paddle

    from paddlefleet.transformer.dot_product_attention import (
        DotProductAttention,
        _EagerQKScoresFn,
    )
    from paddlefleet.transformer.enums import AttnMaskType
    from paddlefleet.transformer.transformer_config import TransformerConfig

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    # Only genuine missing-dependency errors are swallowed into a skip; any
    # other exception (compile break, API change) must surface as a failure.
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    None
    if _IMPORT_ERROR is None
    else f"Paddle unavailable, cannot exercise real attention path: {_IMPORT_ERROR!r}"
)


def _softmax_last(x):
    """Independent fp32 row softmax (numerically stabilized, like paddle)."""
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def _reference_attention(query, key, value, masked_out, scale, kv_repeat=1):
    """Hand-derived masked-softmax attention.

    ``query``/``key``/``value``: numpy ``[b, s, nh_or_ng, hd]``.
    ``masked_out``: boolean numpy broadcastable to ``[b, nh, sq, sk]`` where
    True marks a position that must not be attended (filled with -10000.0,
    matching ``attention_mask_func``).
    ``kv_repeat``: GQA expansion factor; each kv head feeds ``kv_repeat``
    consecutive query heads (``repeat_interleave`` semantics).
    """
    q = np.asarray(query, dtype=np.float64).transpose(
        0, 2, 1, 3
    )  # [b,nh,sq,hd]
    k = np.asarray(key, dtype=np.float64).transpose(0, 2, 1, 3)  # [b,ng,sk,hd]
    v = np.asarray(value, dtype=np.float64).transpose(0, 2, 1, 3)
    if kv_repeat > 1:
        k = np.repeat(k, kv_repeat, axis=1)  # consecutive -> repeat_interleave
        v = np.repeat(v, kv_repeat, axis=1)
    scores = scale * np.matmul(q, np.swapaxes(k, -1, -2))  # [b,nh,sq,sk]
    scores = np.where(np.asarray(masked_out, dtype=bool), -10000.0, scores)
    probs = _softmax_last(scores)
    ctx = np.matmul(probs, v)  # [b,nh,sq,hd]
    b, nh, sq, hd = ctx.shape
    return ctx.transpose(0, 2, 1, 3).reshape(b, sq, nh * hd)


@unittest.skipUnless(_SKIP_REASON is None, _SKIP_REASON or "")
class TestEagerQKScoresFn(unittest.TestCase):
    """The custom PyLayer must equal ``scale * (query @ key_t)`` on the forward
    and produce the exact autograd gradients of that expression on the
    backward."""

    def _fixed_operands(self, seed):
        rng = np.random.RandomState(seed)
        b, sq, sk, hn = 2, 3, 4, 5
        query = rng.standard_normal((b, sq, hn)).astype("float32")
        key_t = rng.standard_normal((b, hn, sk)).astype("float32")
        return query, key_t, b, sq, sk, hn

    def test_forward_matches_independent_scaled_matmul(self):
        q_np, kt_np, b, sq, sk, hn = self._fixed_operands(seed=101)
        scale = 0.7
        scores = _EagerQKScoresFn.apply(
            paddle.to_tensor(q_np), paddle.to_tensor(kt_np), scale
        )
        expected = scale * np.matmul(
            q_np.astype(np.float64), kt_np.astype(np.float64)
        )
        self.assertEqual(list(scores.shape), [b, sq, sk])
        np.testing.assert_allclose(
            scores.numpy(), expected, rtol=1e-6, atol=1e-6
        )

    def test_backward_matches_independent_autograd(self):
        q_np, kt_np, b, sq, sk, hn = self._fixed_operands(seed=202)
        scale = 0.7
        # Non-uniform upstream gradient so an inverted/mis-scaled backward is
        # visible (a ones() cotangent would hide many permutation errors).
        rng = np.random.RandomState(9)
        upstream = rng.standard_normal((b, sq, sk)).astype("float32")

        query = paddle.to_tensor(q_np)
        query.stop_gradient = False
        key_t = paddle.to_tensor(kt_np)
        key_t.stop_gradient = False
        scores = _EagerQKScoresFn.apply(query, key_t, scale)
        scores.backward(paddle.to_tensor(upstream))

        # Independent reference: plain autograd through the scaled matmul.
        q_ref = paddle.to_tensor(q_np)
        q_ref.stop_gradient = False
        k_ref = paddle.to_tensor(kt_np)
        k_ref.stop_gradient = False
        (paddle.matmul(q_ref, k_ref) * scale).backward(
            paddle.to_tensor(upstream)
        )

        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key_t.grad)
        # Reference gradients are non-trivial, so the comparison is sensitive.
        self.assertGreater(np.abs(q_ref.grad.numpy()).max(), 1e-2)
        self.assertGreater(np.abs(k_ref.grad.numpy()).max(), 1e-2)
        np.testing.assert_allclose(
            query.grad.numpy(), q_ref.grad.numpy(), rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            key_t.grad.numpy(), k_ref.grad.numpy(), rtol=1e-5, atol=1e-5
        )


@unittest.skipUnless(_SKIP_REASON is None, _SKIP_REASON or "")
class TestDotProductAttentionAccuracyCompatible(unittest.TestCase):
    """End-to-end forward/backward of the non-fused accuracy-compatible path
    (float32 inputs -> eager QK scores -> masked fp32 softmax -> V)."""

    def _make_config(self, num_key_value_heads=4, **overrides):
        defaults = {
            "hidden_size": 64,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": num_key_value_heads,
            "attention_dropout": 0.0,
            "masked_softmax_fusion": False,
            "attention_softmax_in_fp32": True,
            "apply_query_key_layer_scaling": False,
            "params_dtype": "float32",
            "use_accuracy_compatible": True,
        }
        defaults.update(overrides)
        return TransformerConfig(**defaults)

    def _build(self, **cfg_overrides):
        config = self._make_config(**cfg_overrides)
        return DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )

    def _inputs(self, seed, num_kv_heads=4):
        rng = np.random.RandomState(seed)
        b, s, nh, hd = 1, 4, 4, 16
        q = rng.standard_normal((b, s, nh, hd)).astype("float32")
        k = rng.standard_normal((b, s, num_kv_heads, hd)).astype("float32")
        v = rng.standard_normal((b, s, num_kv_heads, hd)).astype("float32")
        return q, k, v, (b, s, nh, hd)

    @staticmethod
    def _strict_upper_bool_mask(b, s):
        # True = masked-out == strict upper triangle (causal).
        return np.triu(np.ones((b, 1, s, s), dtype=bool), k=1)

    def test_forward_matches_independent_reference(self):
        attn = self._build()
        q, k, v, (b, s, nh, hd) = self._inputs(seed=7)
        scale = 1.0 / math.sqrt(hd)
        self.assertAlmostEqual(attn.softmax_scale, scale, places=12)

        masked = self._strict_upper_bool_mask(b, s)
        bool_mask = paddle.to_tensor(masked)
        out = attn(
            paddle.to_tensor(q),
            paddle.to_tensor(k),
            paddle.to_tensor(v),
            bool_mask,
            attn_mask_type=AttnMaskType.causal,
        )
        expected = _reference_attention(q, k, v, masked, scale)
        self.assertEqual(list(out.shape), [b, s, nh * hd])
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-5)

    def test_float32_mask_converted_with_correct_direction(self):
        # PaddleFleet float mask: 1.0 = attend, 0.0 = masked. The compat path
        # converts via (mask < 0.5) -> True(masked). Use an asymmetric causal
        # mask so the inverted interpretation gives a different result.
        q, k, v, (b, s, nh, hd) = self._inputs(seed=11)
        scale = 1.0 / math.sqrt(hd)
        float_mask = np.tril(np.ones((b, 1, s, s), dtype="float32"))  # attend
        masked_out = float_mask < 0.5  # strict upper triangle -> masked

        out = self._build()(
            paddle.to_tensor(q),
            paddle.to_tensor(k),
            paddle.to_tensor(v),
            paddle.to_tensor(float_mask),
            attn_mask_type=AttnMaskType.causal,
        )
        expected = _reference_attention(q, k, v, masked_out, scale)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-5)
        # The opposite mask direction must NOT match: guards the < 0.5 threshold.
        inverted = _reference_attention(q, k, v, ~masked_out, scale)
        self.assertGreater(np.abs(out.numpy() - inverted).max(), 1e-3)

    def test_float_mask_equals_equivalent_bool_mask(self):
        q, k, v, (b, s, nh, hd) = self._inputs(seed=13)
        float_mask = np.tril(np.ones((b, 1, s, s), dtype="float32"))
        bool_mask = self._strict_upper_bool_mask(b, s)

        out_float = self._build()(
            paddle.to_tensor(q),
            paddle.to_tensor(k),
            paddle.to_tensor(v),
            paddle.to_tensor(float_mask),
            attn_mask_type=AttnMaskType.causal,
        )
        out_bool = self._build()(
            paddle.to_tensor(q),
            paddle.to_tensor(k),
            paddle.to_tensor(v),
            paddle.to_tensor(bool_mask),
            attn_mask_type=AttnMaskType.causal,
        )
        np.testing.assert_allclose(
            out_float.numpy(), out_bool.numpy(), rtol=1e-6, atol=1e-6
        )

    def test_softmax_forced_fp32_and_output_matches_reference(self):
        attn = self._build()
        # Production forces softmax_in_fp32 True on the compat path; start from
        # False so the assertion observes the production flip, not our own set.
        attn.scale_mask_softmax.softmax_in_fp32 = False
        q, k, v, (b, s, nh, hd) = self._inputs(seed=17)
        scale = 1.0 / math.sqrt(hd)
        masked = self._strict_upper_bool_mask(b, s)

        out = attn(
            paddle.to_tensor(q),
            paddle.to_tensor(k),
            paddle.to_tensor(v),
            paddle.to_tensor(masked),
            attn_mask_type=AttnMaskType.causal,
        )
        self.assertTrue(attn.scale_mask_softmax.softmax_in_fp32)
        expected = _reference_attention(q, k, v, masked, scale)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-5)

    def test_gqa_key_value_repeat_interleave(self):
        # num_key_value_heads=2, num_attention_heads=4 -> each kv head feeds two
        # consecutive query heads. A wrong expansion (tile vs repeat_interleave)
        # would reorder heads and break the content comparison.
        attn = self._build(num_key_value_heads=2)
        q, k, v, (b, s, nh, hd) = self._inputs(seed=23, num_kv_heads=2)
        scale = 1.0 / math.sqrt(hd)
        masked = self._strict_upper_bool_mask(b, s)

        out = attn(
            paddle.to_tensor(q),
            paddle.to_tensor(k),
            paddle.to_tensor(v),
            paddle.to_tensor(masked),
            attn_mask_type=AttnMaskType.causal,
        )
        expected = _reference_attention(
            q, k, v, masked, scale, kv_repeat=nh // 2
        )
        self.assertEqual(list(out.shape), [b, s, nh * hd])
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-5)

    def test_backward_grads_match_independent_reference(self):
        attn = self._build()
        q, k, v, (b, s, nh, hd) = self._inputs(seed=29)
        scale = 1.0 / math.sqrt(hd)
        masked = self._strict_upper_bool_mask(b, s)
        rng = np.random.RandomState(31)
        upstream = rng.standard_normal((b, s, nh * hd)).astype("float32")

        query = paddle.to_tensor(q)
        key = paddle.to_tensor(k)
        value = paddle.to_tensor(v)
        query.stop_gradient = False
        key.stop_gradient = False
        value.stop_gradient = False
        out = attn(
            query,
            key,
            value,
            paddle.to_tensor(masked),
            attn_mask_type=AttnMaskType.causal,
        )
        out.backward(paddle.to_tensor(upstream))

        # Independent reference forward built from paddle primitives (no call to
        # DotProductAttention / _EagerQKScoresFn), then autograd backward.
        q_ref = paddle.to_tensor(q)
        k_ref = paddle.to_tensor(k)
        v_ref = paddle.to_tensor(v)
        for t in (q_ref, k_ref, v_ref):
            t.stop_gradient = False
        qh = q_ref.transpose([0, 2, 1, 3])  # [b,nh,s,hd]
        kh = k_ref.transpose([0, 2, 1, 3])
        vh = v_ref.transpose([0, 2, 1, 3])
        scores = paddle.matmul(qh, kh.transpose([0, 1, 3, 2])) * scale
        mask_t = paddle.to_tensor(masked)
        scores = paddle.where(
            mask_t, paddle.full_like(scores, -10000.0), scores
        )
        probs = paddle.nn.functional.softmax(scores, axis=-1)
        ctx = paddle.matmul(probs, vh).transpose([0, 2, 1, 3])
        ref_out = ctx.reshape([b, s, nh * hd])
        ref_out.backward(paddle.to_tensor(upstream))

        # Forward must already agree, else grad comparison is meaningless.
        np.testing.assert_allclose(
            out.numpy(), ref_out.numpy(), rtol=1e-5, atol=1e-5
        )
        for name, actual, expected in (
            ("query", query.grad, q_ref.grad),
            ("key", key.grad, k_ref.grad),
            ("value", value.grad, v_ref.grad),
        ):
            self.assertIsNotNone(actual, f"{name} grad missing")
            self.assertIsNotNone(expected, f"{name} ref grad missing")
            self.assertGreater(
                np.abs(expected.numpy()).max(), 1e-3, f"{name} ref grad trivial"
            )
            np.testing.assert_allclose(
                actual.numpy(),
                expected.numpy(),
                rtol=1e-4,
                atol=1e-5,
                err_msg=f"{name} gradient mismatch",
            )


if __name__ == "__main__":
    unittest.main()
