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
"""Behavior tests for ``DotProductAttention`` scaling / softmax-sink config and
its ``scaled_dot_product_attention_with_softmax_offset`` sink math.

No-card (CPU) tests. Every path exercised here (constructor plumbing, the fp32
manual sink-softmax kernel, and the fp32 input-contract guards) runs without a
GPU; the fp16/bf16 flash-attention branches need a GPU and are covered by the
single-card suite, not here. Numeric expectations are derived independently in
float64 numpy, never read back from the layer.

Paddle is not importable in every environment; the whole file is skipped with an
honest reason when the real imports fail. Only ``ImportError`` /
``ModuleNotFoundError`` are treated as "dependency missing" so a compile break or
API change surfaces as an error instead of being swallowed into a skip.
"""

import math
import unittest

import numpy as np

try:
    import paddle

    from paddlefleet.transformer.dot_product_attention import (
        DotProductAttention,
        scaled_dot_product_attention_with_softmax_offset,
    )
    from paddlefleet.transformer.enums import AttnMaskType
    from paddlefleet.transformer.transformer_config import TransformerConfig
    from paddlefleet.utils import (
        init_method_normal,
        scaled_init_method_normal,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    _IMPORT_ERROR = exc

_SKIP_REASON = f"paddle/paddlefleet import unavailable: {_IMPORT_ERROR!r}"

# fixture dims: small, distinguishable, non-degenerate (seq > 1, head_dim > 1).
_HEAD_DIM = 4
_NUM_HEADS = 2
_HIDDEN = _HEAD_DIM * _NUM_HEADS


def _make_config(**overrides):
    """Build a minimal but real ``TransformerConfig`` for a dense attention.

    Only the plumbing needed to instantiate ``DotProductAttention`` is set. The
    numeric expectations in the tests are derived by hand, never from this
    object.
    """
    defaults = {
        "num_hidden_layers": 2,
        "hidden_size": _HIDDEN,
        "num_attention_heads": _NUM_HEADS,
        "num_key_value_heads": _NUM_HEADS,
        "head_dim": _HEAD_DIM,
        "softmax_scale": None,
        "use_bias": True,
        "recompute_granularity": None,
        "recompute_modules": None,
        "init_method": init_method_normal(0.02),
        "output_layer_init_method": scaled_init_method_normal(0.02, 1, 2.0),
        "rms_norm_eps": 1e-5,
        "context_parallel_size": 1,
        "sequence_parallel": False,
        "apply_query_key_layer_scaling": False,
        "sliding_window": None,
        "window_attn_skip_freq": None,
        "fp16": False,
        "bf16": False,
        "masked_softmax_fusion": False,
        "attention_softmax_in_fp32": True,
        "attention_dropout": 0.0,
        "softmax_type": "vanilla",
        "fa_version": None,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _build_attn(config, layer_number=1, **kwargs):
    return DotProductAttention(
        config=config,
        layer_number=layer_number,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Independent numpy reference for the sink-softmax kernel (no call into the code
# under test). ``weights_ij = exp(s_ij) / (sum_k exp(s_ik) + exp(sink_i))`` --
# the virtual-token off-by-one/learnable-sink definition; rows sum to < 1.
# --------------------------------------------------------------------------- #
def _np_sink_attention(q, k, v, scale, offset, groups):
    """q: [B, Q, Hq, dq]; k, v: [B, K, Hkv, dv]; offset: [Hq]. Non-causal.

    Query head ``hq`` reads kv head ``hq // groups`` (the production reshape
    splits Hq into (Hkv, groups) with Hkv as the outer dim).
    """
    B, Q, Hq, _ = q.shape
    Hkv = k.shape[2]
    Dv = v.shape[-1]
    out = np.zeros((B, Q, Hq, Dv), dtype=np.float64)
    for b in range(B):
        for hq in range(Hq):
            hk = hq // groups
            qh = q[b, :, hq, :].astype(np.float64)
            kh = k[b, :, hk, :].astype(np.float64)
            vh = v[b, :, hk, :].astype(np.float64)
            s = (qh @ kh.T) * scale  # [Q, K]
            e = np.exp(s)
            denom = e.sum(axis=-1, keepdims=True) + np.exp(
                np.float64(offset[hq])
            )
            w = e / denom  # rows sum < 1
            out[b, :, hq, :] = w @ vh
    return out


def _fixed_qkv(num_q_heads, num_kv_heads, seq_q, seq_kv, head_dim, seed=0):
    rng = np.random.RandomState(seed)
    q = (rng.standard_normal((1, seq_q, num_q_heads, head_dim)) * 0.5).astype(
        np.float32
    )
    k = (rng.standard_normal((1, seq_kv, num_kv_heads, head_dim)) * 0.5).astype(
        np.float32
    )
    v = (rng.standard_normal((1, seq_kv, num_kv_heads, head_dim)) * 0.5).astype(
        np.float32
    )
    return q, k, v


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestSoftmaxScaleScaling(unittest.TestCase):
    """QK-layer-scaling wiring: it must both pre-divide ``softmax_scale`` and arm
    the fused softmax stage with ``scale=coeff``, and it must key on the actual
    ``layer_number`` (coeff = max(1, layer_number))."""

    def test_layer_scaling_depends_on_layer_number(self):
        base = 1.0 / math.sqrt(_HEAD_DIM)

        # layer_number=1 -> coeff clamped to 1 -> a no-op (the degenerate point).
        attn1 = _build_attn(
            _make_config(apply_query_key_layer_scaling=True), layer_number=1
        )
        self.assertAlmostEqual(attn1.softmax_scale, base, places=6)
        self.assertEqual(attn1.scale_mask_softmax.scale, 1)

        # layer_number=4 -> coeff=4 -> scale divided, softmax armed with 4.
        attn4 = _build_attn(
            _make_config(apply_query_key_layer_scaling=True), layer_number=4
        )
        self.assertAlmostEqual(attn4.softmax_scale, base / 4, places=6)
        self.assertEqual(attn4.scale_mask_softmax.scale, 4)

        # The two layers must not collapse to the same scale.
        self.assertNotAlmostEqual(
            attn1.softmax_scale, attn4.softmax_scale, places=6
        )
        # Once scaled, the custom-scale flag is armed so the kernel receives it.
        self.assertTrue(attn4._has_custom_softmax_scale)

    def test_no_scaling_leaves_default_and_disarms_coeff(self):
        attn = _build_attn(
            _make_config(apply_query_key_layer_scaling=False), layer_number=4
        )
        self.assertAlmostEqual(
            attn.softmax_scale, 1.0 / math.sqrt(_HEAD_DIM), places=6
        )
        # coeff stays None when scaling is off, so the fused softmax is not armed.
        self.assertIsNone(attn.scale_mask_softmax.scale)
        self.assertFalse(attn._has_custom_softmax_scale)

    def test_custom_scale_is_still_divided_by_coeff(self):
        # An explicit softmax_scale must survive AND be divided by coeff when
        # layer scaling is on: 0.25 / max(1, 2) = 0.125.
        attn = _build_attn(
            _make_config(apply_query_key_layer_scaling=True),
            layer_number=2,
            softmax_scale=0.25,
        )
        self.assertAlmostEqual(attn.softmax_scale, 0.125, places=7)
        self.assertTrue(attn._has_custom_softmax_scale)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestSoftmaxOffsetConfig(unittest.TestCase):
    """``build_softmax_offset`` branch selection driven by config, observed
    through the constructed layer's ``softmax_offset`` parameter."""

    def test_off_by_one_is_fixed_zero_sink(self):
        attn = _build_attn(_make_config(softmax_type="off-by-one"))
        self.assertIsNotNone(attn.softmax_offset)
        self.assertEqual(list(attn.softmax_offset.shape), [_NUM_HEADS])
        # Fixed zero sink logit -> content is zeros, and it is NOT a trainable
        # parameter (distinguishes it from the learnable branch below).
        np.testing.assert_array_equal(
            attn.softmax_offset.numpy(),
            np.zeros(_NUM_HEADS, dtype=attn.softmax_offset.numpy().dtype),
        )
        self.assertTrue(attn.softmax_offset.stop_gradient)

    def test_full_attention_sink_bias_forces_learnable_offset(self):
        # softmax_type stays "vanilla", but add_full_attention_sink_bias on a
        # non-SWA layer must override it to a *learnable* per-head offset.
        config = _make_config(softmax_type="vanilla")
        config.add_full_attention_sink_bias = True
        config.add_swa_attention_sink_bias = False
        config.perform_initialization = False
        config.params_dtype = "float32"
        attn = _build_attn(config, is_swa=False)
        self.assertIsNotNone(attn.softmax_offset)
        self.assertEqual(list(attn.softmax_offset.shape), [_NUM_HEADS])
        # A learnable sink is a trainable parameter (stop_gradient False) and is
        # registered on the layer; the fixed zero sink is neither.
        self.assertFalse(attn.softmax_offset.stop_gradient)
        self.assertTrue(
            any(p is attn.softmax_offset for p in attn.parameters())
        )

    def test_full_sink_bias_does_not_leak_into_swa_layer(self):
        # add_full_* is gated by ``not is_swa``; with add_swa_* off, an SWA layer
        # must fall back to vanilla (no sink), not silently get a full-attn sink.
        config = _make_config(softmax_type="vanilla", sliding_window=8)
        config.add_full_attention_sink_bias = True
        config.add_swa_attention_sink_bias = False
        attn = _build_attn(config, is_swa=True)
        self.assertIsNone(attn.softmax_offset)

    def test_invalid_softmax_type_raises(self):
        with self.assertRaises(ValueError):
            _build_attn(_make_config(softmax_type="not-a-real-type"))


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestSinkSoftmaxKernelMath(unittest.TestCase):
    """``scaled_dot_product_attention_with_softmax_offset`` fp32 numerics against
    an independent numpy sink-softmax reference. A non-zero, per-head offset is
    used so the sink denominator is actually exercised (the off-by-one config
    only ever supplies a zero offset)."""

    def _run(self, num_q_heads, num_kv_heads, groups):
        seq_q, seq_kv, dim = 3, 4, _HEAD_DIM
        q_np, k_np, v_np = _fixed_qkv(
            num_q_heads, num_kv_heads, seq_q, seq_kv, dim
        )
        # Distinct per-head sink logits, including a non-trivial one (ln 2).
        offset = np.array(
            [math.log(2.0) * (h + 1) for h in range(num_q_heads)],
            dtype=np.float32,
        )
        scale = 1.0 / math.sqrt(dim)

        out = scaled_dot_product_attention_with_softmax_offset(
            paddle.to_tensor(q_np),
            paddle.to_tensor(k_np),
            paddle.to_tensor(v_np),
            attn_mask_kv=None,
            is_causal=False,
            softmax_offset=paddle.to_tensor(offset),
            q_head_dim=dim,
            scale=scale,
            dropout_p=0.0,
            training=False,
        )
        # Returned layout is [B, Q, Hq, dv].
        self.assertEqual(list(out.shape), [1, seq_q, num_q_heads, dim])
        expected = _np_sink_attention(q_np, k_np, v_np, scale, offset, groups)
        np.testing.assert_allclose(
            out.numpy().astype(np.float64), expected, rtol=1e-5, atol=1e-6
        )
        return out.numpy().astype(np.float64), expected

    def test_mha_sink_denominator(self):
        got, expected = self._run(_NUM_HEADS, _NUM_HEADS, groups=1)
        # Sanity on the reference itself: a positive sink logit makes every row
        # sum strictly below 1, so the kernel must not renormalize it away.
        # (checked implicitly by the tight allclose above; guard the anchor too)
        self.assertTrue(np.isfinite(expected).all())

    def test_gqa_maps_query_head_to_shared_kv_head(self):
        # Hq=4, Hkv=2, groups=2: query head hq reads kv head hq//2. A wrong
        # group axis or count would move values far past the tolerance.
        self._run(num_q_heads=4, num_kv_heads=2, groups=2)

    def test_wrong_offset_is_rejected_by_reference(self):
        # Negative control: comparing the kernel output against a reference that
        # used the WRONG (zero) sink must fail, proving the offset is load-bearing
        # and the comparison is not vacuous.
        seq_q, seq_kv, dim = 3, 4, _HEAD_DIM
        q_np, k_np, v_np = _fixed_qkv(
            _NUM_HEADS, _NUM_HEADS, seq_q, seq_kv, dim
        )
        offset = np.array([math.log(2.0), math.log(3.0)], dtype=np.float32)
        scale = 1.0 / math.sqrt(dim)
        out = (
            scaled_dot_product_attention_with_softmax_offset(
                paddle.to_tensor(q_np),
                paddle.to_tensor(k_np),
                paddle.to_tensor(v_np),
                softmax_offset=paddle.to_tensor(offset),
                q_head_dim=dim,
                scale=scale,
            )
            .numpy()
            .astype(np.float64)
        )
        wrong = _np_sink_attention(
            q_np, k_np, v_np, scale, np.zeros_like(offset), groups=1
        )
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(out, wrong, rtol=1e-5, atol=1e-6)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestForwardInputGuards(unittest.TestCase):
    """fp32-reachable input-contract guards in ``DotProductAttention.forward``."""

    def test_eager_rejects_packed_seq_params(self):
        config = _make_config()
        config._attn_implementation = "eager"
        attn = _build_attn(config)
        attn.eval()
        q = paddle.zeros([1, 4, _NUM_HEADS, _HEAD_DIM], dtype="float32")
        with self.assertRaisesRegex(ValueError, "eager"):
            # packed_seq_params is only inspected after the guard fires, so a
            # bare sentinel is enough to trip it.
            attn.forward(q, q, q, None, packed_seq_params=object())

    def test_packed_seq_requires_fp16_or_bf16(self):
        # Non-eager + packed_seq_params + fp32 must be rejected: the flashmask
        # packed path only supports fp16/bf16. This guard is reachable on CPU.
        config = _make_config()
        config._attn_implementation = "sdpa"
        attn = _build_attn(config)
        attn.eval()
        q = paddle.zeros([1, 4, _NUM_HEADS, _HEAD_DIM], dtype="float32")
        with self.assertRaisesRegex(AssertionError, "fp16/bf16"):
            attn.forward(q, q, q, None, packed_seq_params=object())


if __name__ == "__main__":
    unittest.main()
