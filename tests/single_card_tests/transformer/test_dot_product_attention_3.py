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
"""Behavior tests for paddlefleet.transformer.dot_product_attention.

Scope is the CPU-executable numeric surface of the module:

* ``scaled_dot_product_attention_with_softmax_offset`` -- the manual
  sink-token ("off-by-one" / learnable-sink) softmax, including the
  GQA-preserving group reshape and the additive/boolean mask branches.
* ``_EagerQKScoresFn`` -- the baddbmm QK-score PyLayer with its explicit
  backward (forward AND backward are exercised and compared).
* ``DotProductAttention`` construction -- how ``softmax_scale`` is derived
  from ``head_dim``, how ``apply_query_key_layer_scaling`` divides by
  ``max(1, layer_number)``, and how ``softmax_type`` selects the sink offset.

Every numeric expectation is derived from an INDEPENDENT numpy reference
(the sink-softmax algorithm and the scaled-matmul gradient rule reimplemented
from first principles), never by calling the production function under test.
Fused/flashmask/CP paths need fp16/bf16 GPU kernels and are therefore not
asserted here; that is a runtime-environment limitation, not a claim that
those paths are verified.

Heavy imports (paddle + the module) are guarded so a missing runtime is
reported as an honest skip; only ImportError/ModuleNotFoundError is treated
as "dependency absent" so real API breaks still surface.
"""

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
        ),
        "src",
    ),
)

try:
    import numpy as np
    import paddle

    from paddlefleet.transformer.dot_product_attention import (
        DotProductAttention,
        _EagerQKScoresFn,
        scaled_dot_product_attention_with_softmax_offset,
    )
    from paddlefleet.transformer.enums import AttnMaskType
    from paddlefleet.transformer.transformer_config import TransformerConfig

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle/paddlefleet/numpy not importable in this environment: {_IMPORT_ERROR}"
    if _IMPORT_ERROR is not None
    else ""
)


def _make_config(**overrides):
    """Construct a real TransformerConfig for a small MHA/GQA attention.

    The field set mirrors what DotProductAttention.__init__ reads; values are
    deliberately small and distinguishable. This is a production object, not a
    stub -- construction failures surface as real errors.
    """
    defaults = {
        "hidden_size": 64,
        "num_attention_heads": 4,
        "head_dim": 16,
        "num_key_value_heads": 4,
        "num_hidden_layers": 2,
        "context_parallel_size": 1,
        "fp16": False,
        "bf16": False,
        "masked_softmax_fusion": False,
        "attention_softmax_in_fp32": True,
        "attention_dropout": 0.0,
        "apply_query_key_layer_scaling": False,
        "sliding_window": None,
        "softmax_type": "vanilla",
        "flashmask_use_varlen": False,
        "params_dtype": "float32",
        "perform_initialization": True,
        "init_method": paddle.nn.initializer.Normal(0.02),
        "sequence_parallel": False,
        "tensor_model_parallel_size": 1,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _build_attn(config, **kwargs):
    kwargs.setdefault("layer_number", 1)
    kwargs.setdefault("attn_mask_type", AttnMaskType.causal)
    kwargs.setdefault("attention_type", "self")
    return DotProductAttention(config=config, **kwargs)


def _sink_softmax_reference(
    query, key, value, scale, softmax_offset, attn_mask_kv=None, is_causal=False
):
    """Independent numpy reference for the manual sink-token softmax.

    Reimplements the documented algorithm with plain numpy in float64:

        scores  = (Q @ K^T) * scale                 [B, Hq, Q, K]
        row_max = max(max(scores, -1), sink)
        exp_s   = exp(scores - row_max)
        row_sum = sum(exp_s, -1) + exp(sink - row_max)   <- virtual token
        weights = exp_s / row_sum                        (rows sum < 1)
        out     = weights @ V

    GQA is handled by expanding K/V with a consecutive repeat_interleave of the
    KV heads (each KV head repeated ``groups`` times), which is the layout the
    group-preserving production reshape must agree with. Nothing here calls the
    production function.
    """
    q = np.asarray(query, dtype=np.float64)
    k = np.asarray(key, dtype=np.float64)
    v = np.asarray(value, dtype=np.float64)
    B, Q, Hq, dq = q.shape
    K = k.shape[1]
    Hkv = k.shape[2]
    groups = Hq // Hkv

    qf = q.transpose(0, 2, 1, 3)  # [B, Hq, Q, dq]
    kf = k.transpose(0, 2, 1, 3)  # [B, Hkv, K, dk]
    vf = v.transpose(0, 2, 1, 3)  # [B, Hkv, K, dv]
    kfe = np.repeat(kf, groups, axis=1)  # [B, Hq, K, dk]
    vfe = np.repeat(vf, groups, axis=1)  # [B, Hq, K, dv]

    scores = np.matmul(qf, kfe.transpose(0, 1, 3, 2)) * scale  # [B, Hq, Q, K]
    if is_causal and Q > 1:
        causal = np.tril(np.ones((Q, K)), k=K - Q)
        scores = scores + np.where(causal[None, None] == 0, -np.inf, 0.0)
    if attn_mask_kv is not None:
        scores = scores + np.asarray(attn_mask_kv, dtype=np.float64)

    sink = np.asarray(softmax_offset, dtype=np.float64).reshape(1, -1, 1, 1)
    row_max = np.maximum(scores.max(-1, keepdims=True), sink)
    exp_s = np.exp(scores - row_max)
    row_sum = exp_s.sum(-1, keepdims=True) + np.exp(sink - row_max)
    weights = exp_s / row_sum
    out = np.matmul(weights, vfe)  # [B, Hq, Q, dv]
    return out.transpose(0, 2, 1, 3), weights  # out [B, Q, Hq, dv]


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestSoftmaxScaleDerivation(unittest.TestCase):
    """softmax_scale is derived from head_dim and layer scaling."""

    def test_default_scale_is_inv_sqrt_head_dim(self):
        # hidden_size_per_attention_head == k_channels == head_dim (world_size 1),
        # so the default scale is 1/sqrt(head_dim), independent of hidden_size.
        attn = _build_attn(_make_config(head_dim=16))
        self.assertFalse(attn._has_custom_softmax_scale)
        self.assertAlmostEqual(attn.softmax_scale, 1.0 / (16**0.5), places=7)

        attn64 = _build_attn(_make_config(head_dim=64))
        self.assertAlmostEqual(attn64.softmax_scale, 1.0 / (64**0.5), places=7)
        # The two head_dims must give different scales; a scale that ignored
        # head_dim would collapse them.
        self.assertNotAlmostEqual(
            attn.softmax_scale, attn64.softmax_scale, places=4
        )

    def test_custom_scale_is_taken_verbatim(self):
        attn = _build_attn(_make_config(), softmax_scale=0.5)
        self.assertTrue(attn._has_custom_softmax_scale)
        self.assertEqual(attn.softmax_scale, 0.5)

    def test_layer_scaling_divides_by_max_1_layer_number(self):
        # apply_query_key_layer_scaling divides the base scale by
        # coeff = max(1, layer_number). layer_number below 1 must clamp to 1
        # (not divide by zero); larger layer numbers scale down.
        base = 1.0 / (16**0.5)
        cfg = _make_config(head_dim=16, apply_query_key_layer_scaling=True)

        attn0 = _build_attn(cfg, layer_number=0)
        self.assertAlmostEqual(attn0.softmax_scale, base, places=7)  # coeff=1

        attn3 = _build_attn(cfg, layer_number=3)
        self.assertAlmostEqual(attn3.softmax_scale, base / 3.0, places=7)

    def test_layer_scaling_also_scales_a_custom_scale(self):
        # A user-supplied scale still gets divided by the layer coefficient.
        cfg = _make_config(apply_query_key_layer_scaling=True)
        attn = _build_attn(cfg, layer_number=2, softmax_scale=0.5)
        self.assertAlmostEqual(attn.softmax_scale, 0.25, places=7)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestSoftmaxOffsetSelection(unittest.TestCase):
    """softmax_type selects the per-head sink offset (build_softmax_offset)."""

    def test_vanilla_has_no_offset(self):
        attn = _build_attn(_make_config(softmax_type="vanilla"))
        self.assertIsNone(attn.softmax_offset)

    def test_off_by_one_is_fixed_zero_vector_over_heads(self):
        # off-by-one is the softmax1 / "+1 in the denominator" variant: a fixed
        # (non-learnable) zero sink, one entry per local attention head.
        cfg = _make_config(num_attention_heads=4, softmax_type="off-by-one")
        attn = _build_attn(cfg)
        self.assertIsNotNone(attn.softmax_offset)
        self.assertEqual(
            list(attn.softmax_offset.shape),
            [attn.num_attention_heads_per_partition],
        )
        self.assertEqual(attn.num_attention_heads_per_partition, 4)
        np.testing.assert_array_equal(
            attn.softmax_offset.numpy(), np.zeros(4, dtype=np.float32)
        )

    def test_invalid_softmax_type_raises_value_error(self):
        cfg = _make_config(softmax_type="not-a-real-type")
        with self.assertRaises(ValueError):
            _build_attn(cfg)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestSinkSoftmaxAttention(unittest.TestCase):
    """scaled_dot_product_attention_with_softmax_offset numeric contract."""

    def test_tiny_mha_matches_hand_derived_anchor(self):
        # Fully hand-derived anchor (scale=1, zero sink, 2 keys):
        #   scores = [1, 0]; row_max = max(1, 0) = 1
        #   exp_s  = [1, e^-1];  row_sum = 1 + e^-1 + e^-1
        #   w      = [0.5761169, 0.2119416] (sum ~= 0.7880584 < 1)
        #   out    = w0*[1,0] + w1*[0,1] = [0.5761169, 0.2119416]
        query = paddle.to_tensor([[[[1.0, 0.0]]]], dtype="float32")
        key = paddle.to_tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]], dtype="float32")
        value = paddle.to_tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]]]], dtype="float32"
        )
        out = scaled_dot_product_attention_with_softmax_offset(
            query,
            key,
            value,
            attn_mask_kv=None,
            is_causal=False,
            softmax_offset=paddle.zeros([1]),
            q_head_dim=2,
            scale=1.0,
        )
        self.assertEqual(list(out.shape), [1, 1, 1, 2])
        np.testing.assert_allclose(
            out.numpy(),
            np.array([[[[0.57611688, 0.21194156]]]]),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_sink_makes_rows_sum_below_one_and_differs_from_vanilla(self):
        # The virtual sink token strictly reduces the attention mass: the
        # implied weights sum to < 1, and the output differs from a plain
        # (no-sink) softmax over the same scores. A no-op sink implementation
        # would collapse these two.
        rng = np.random.RandomState(7)
        q = rng.randn(1, 3, 1, 4).astype("float32")
        k = rng.randn(1, 5, 1, 4).astype("float32")
        v = rng.randn(1, 5, 1, 4).astype("float32")
        scale = 0.3

        out = scaled_dot_product_attention_with_softmax_offset(
            paddle.to_tensor(q),
            paddle.to_tensor(k),
            paddle.to_tensor(v),
            attn_mask_kv=None,
            is_causal=False,
            softmax_offset=paddle.zeros([1]),
            q_head_dim=4,
            scale=scale,
        )
        ref_out, ref_w = _sink_softmax_reference(q, k, v, scale, np.zeros(1))
        np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-5, atol=1e-6)

        # Every row must lose mass to the sink.
        row_sums = ref_w.sum(-1)
        self.assertTrue(np.all(row_sums < 1.0 - 1e-6))

        # Plain softmax (no sink) over the same scores gives a different output.
        qf = q.transpose(0, 2, 1, 3).astype(np.float64)
        kf = k.transpose(0, 2, 1, 3).astype(np.float64)
        vf = v.transpose(0, 2, 1, 3).astype(np.float64)
        scores = np.matmul(qf, kf.transpose(0, 1, 3, 2)) * scale
        ex = np.exp(scores - scores.max(-1, keepdims=True))
        w_plain = ex / ex.sum(-1, keepdims=True)
        out_plain = np.matmul(w_plain, vf).transpose(0, 2, 1, 3)
        self.assertGreater(float(np.abs(out_plain - ref_out).max()), 1e-3)

    def test_gqa_group_reshape_matches_expanded_kv(self):
        # groups = Hq / Hkv = 4 / 2 = 2. The group-preserving reshape must
        # produce exactly the same result as materially repeating each KV head
        # twice (consecutive). Distinct per-position content would expose a
        # head/group mix-up that a shape-only check cannot.
        rng = np.random.RandomState(11)
        q = rng.randn(2, 3, 4, 8).astype("float32")  # B2 Q3 Hq4 d8
        k = rng.randn(2, 5, 2, 8).astype("float32")  # Hkv2 K5
        v = rng.randn(2, 5, 2, 8).astype("float32")
        scale = 0.2
        out = scaled_dot_product_attention_with_softmax_offset(
            paddle.to_tensor(q),
            paddle.to_tensor(k),
            paddle.to_tensor(v),
            attn_mask_kv=None,
            is_causal=False,
            softmax_offset=paddle.zeros([4]),
            q_head_dim=8,
            scale=scale,
        )
        ref_out, _ = _sink_softmax_reference(q, k, v, scale, np.zeros(4))
        self.assertEqual(list(out.shape), [2, 3, 4, 8])
        np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-5, atol=1e-6)

    def test_additive_float_mask_is_applied_before_softmax(self):
        # A large negative additive mask on key position 0 must suppress it;
        # the result equals a reference that attends only to the unmasked keys.
        rng = np.random.RandomState(3)
        q = rng.randn(1, 2, 1, 4).astype("float32")
        k = rng.randn(1, 3, 1, 4).astype("float32")
        v = rng.randn(1, 3, 1, 4).astype("float32")
        scale = 0.5
        mask = np.zeros((1, 1, 2, 3), dtype="float32")
        mask[..., 0] = -1e9  # forbid attending to key 0
        out = scaled_dot_product_attention_with_softmax_offset(
            paddle.to_tensor(q),
            paddle.to_tensor(k),
            paddle.to_tensor(v),
            attn_mask_kv=paddle.to_tensor(mask),
            is_causal=False,
            softmax_offset=paddle.zeros([1]),
            q_head_dim=4,
            scale=scale,
        )
        ref_out, ref_w = _sink_softmax_reference(
            q, k, v, scale, np.zeros(1), attn_mask_kv=mask
        )
        np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-5, atol=1e-6)
        # Masked key contributes ~0 weight.
        self.assertLess(float(ref_w[..., 0].max()), 1e-6)

    def test_boolean_mask_masks_true_positions(self):
        # A bool attn_mask_kv marks True == masked-out; those keys are dropped.
        rng = np.random.RandomState(5)
        q = rng.randn(1, 2, 1, 4).astype("float32")
        k = rng.randn(1, 3, 1, 4).astype("float32")
        v = rng.randn(1, 3, 1, 4).astype("float32")
        scale = 0.5
        bool_mask = np.zeros((1, 1, 2, 3), dtype=bool)
        bool_mask[..., 2] = True  # forbid attending to key 2
        out = scaled_dot_product_attention_with_softmax_offset(
            paddle.to_tensor(q),
            paddle.to_tensor(k),
            paddle.to_tensor(v),
            attn_mask_kv=paddle.to_tensor(bool_mask),
            is_causal=False,
            softmax_offset=paddle.zeros([1]),
            q_head_dim=4,
            scale=scale,
        )
        # Equivalent additive -inf reference on the True positions.
        add_mask = np.where(bool_mask, -np.inf, 0.0).astype("float64")
        ref_out, ref_w = _sink_softmax_reference(
            q, k, v, scale, np.zeros(1), attn_mask_kv=add_mask
        )
        np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-5, atol=1e-6)
        self.assertEqual(float(ref_w[..., 2].max()), 0.0)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestSinkSoftmaxCausal(unittest.TestCase):
    """Causal masking inside the sink softmax (prefill, q_len > 1)."""

    def test_causal_mask_hides_future_keys(self):
        # is_causal with Q == K == 3 keeps the lower triangle (diagonal offset
        # K - Q = 0). Row i must place zero weight on keys j > i.
        rng = np.random.RandomState(9)
        q = rng.randn(1, 3, 1, 4).astype("float32")
        k = rng.randn(1, 3, 1, 4).astype("float32")
        v = rng.randn(1, 3, 1, 4).astype("float32")
        scale = 0.4
        out = scaled_dot_product_attention_with_softmax_offset(
            paddle.to_tensor(q),
            paddle.to_tensor(k),
            paddle.to_tensor(v),
            attn_mask_kv=None,
            is_causal=True,
            softmax_offset=paddle.zeros([1]),
            q_head_dim=4,
            scale=scale,
        )
        ref_out, ref_w = _sink_softmax_reference(
            q, k, v, scale, np.zeros(1), is_causal=True
        )
        np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-5, atol=1e-6)
        # Strict upper triangle must be zero (query 0 sees only key 0, etc.).
        w = ref_w[0, 0]  # [Q, K]
        self.assertAlmostEqual(float(w[0, 1]), 0.0, places=12)
        self.assertAlmostEqual(float(w[0, 2]), 0.0, places=12)
        self.assertAlmostEqual(float(w[1, 2]), 0.0, places=12)
        # Query 0 attends to exactly one key, but the sink still steals mass.
        self.assertLess(float(w[0].sum()), 1.0 - 1e-6)
        self.assertGreater(float(w[0, 0]), 0.0)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestEagerQKScoresFn(unittest.TestCase):
    """_EagerQKScoresFn: baddbmm forward and its explicit backward."""

    def _inputs(self):
        rng = np.random.RandomState(21)
        # G = b*np = 1, sq = 2, hn = 2, sk = 3 (sq != sk exposes transpose bugs).
        query = rng.randn(1, 2, 2).astype("float32")
        key_t = rng.randn(1, 2, 3).astype("float32")  # [G, hn, sk]
        upstream = rng.randn(1, 2, 3).astype("float32")  # [G, sq, sk]
        return query, key_t, upstream

    def test_forward_equals_scaled_qk(self):
        query, key_t, _ = self._inputs()
        scale = 0.7
        out = _EagerQKScoresFn.apply(
            paddle.to_tensor(query),
            paddle.to_tensor(key_t),
            scale,
        )
        expected = scale * np.matmul(query, key_t)  # [G, sq, sk]
        self.assertEqual(list(out.shape), [1, 2, 3])
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_backward_matches_scaled_matmul_gradients(self):
        # Independent gradient of scores = scale * (Q @ Kt), from calculus:
        #   dL/dQ  = scale * dScores @ Kt^T
        #   dL/dKt = scale * Q^T @ dScores
        # These are re-derived here, not copied from the production backward.
        query, key_t, upstream = self._inputs()
        scale = 0.7

        q_t = paddle.to_tensor(query)
        kt_t = paddle.to_tensor(key_t)
        q_t.stop_gradient = False
        kt_t.stop_gradient = False

        out = _EagerQKScoresFn.apply(q_t, kt_t, scale)
        out.backward(paddle.to_tensor(upstream))

        dq_expected = scale * np.matmul(upstream, key_t.transpose(0, 2, 1))
        dkt_expected = scale * np.matmul(query.transpose(0, 2, 1), upstream)

        self.assertIsNotNone(q_t.grad)
        self.assertIsNotNone(kt_t.grad)
        # Gradients are non-trivial, so the comparison is scale-sensitive.
        self.assertGreater(float(np.abs(dq_expected).max()), 1e-3)
        self.assertGreater(float(np.abs(dkt_expected).max()), 1e-3)
        np.testing.assert_allclose(
            q_t.grad.numpy(), dq_expected, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            kt_t.grad.numpy(), dkt_expected, rtol=1e-5, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
