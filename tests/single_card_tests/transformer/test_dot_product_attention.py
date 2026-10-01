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
"""Behavior tests for ``DotProductAttention`` (transformer module).

These are no-card (CPU) tests: the fp32 forward of ``DotProductAttention``
routes through the plain matmul/baddbmm + ``FusedScaleMaskSoftmax`` path, which
runs on CPU. The fp16/bf16 flash-attention paths need a GPU and are covered by
the single-card suite, not here.

Paddle is not guaranteed to be importable in every environment; the whole file
is skipped (with an honest reason) when the real imports fail, and only
``ImportError``/``ModuleNotFoundError`` are treated as "dependency missing" so a
compile break or API change surfaces instead of being swallowed.
"""

import math
import unittest

import numpy as np

try:
    import paddle

    from paddlefleet.transformer.dot_product_attention import (
        DotProductAttention,
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

# --- fixture dimensions (small, distinguishable, non-degenerate) ---
_HEAD_DIM = 4
_NUM_HEADS = 2
_HIDDEN = _HEAD_DIM * _NUM_HEADS
_SEQ = 3
_BATCH = 1


def _make_config(**overrides):
    """Build a minimal but real ``TransformerConfig`` for a dense attention.

    Only the plumbing needed to instantiate ``DotProductAttention`` is set; the
    numeric expectations in the tests are derived independently, never from this
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
# Independent numpy reference (no call into the code under test).
# --------------------------------------------------------------------------- #
def _np_softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _causal_masked_scores(q, k, scale):
    """Scores ``Q @ K^T * scale`` with the strict upper triangle replaced by
    ``-10000.0`` -- matching ``attention_mask_func``'s ``masked_fill_`` value and
    ``get_default_causal_mask`` (``triu(..., diagonal=1)``), not an additive mask.
    """
    s = (q @ k.T) * scale
    sq, sk = s.shape
    upper = np.triu(np.ones((sq, sk), dtype=bool), k=1)
    return np.where(upper, -10000.0, s)


def _reference_attention(q, k, v, scale, off_by_one=False):
    """Per-head causal attention reference in float64.

    q: ``[B, S, H, D]``; k, v: ``[B, S, Hkv, Dv]``. GQA is modelled the way the
    production ``repeat_interleave(H // Hkv, dim=2)`` does: query head ``h`` reads
    kv head ``h // (H // Hkv)``. Output is ``[B, S, H * Dv]`` with heads
    concatenated in order (matching the final reshape).
    """
    B, S, H, _ = q.shape
    Hkv = k.shape[2]
    Dv = v.shape[-1]
    rep = H // Hkv
    out = np.zeros((B, S, H, Dv), dtype=np.float64)
    for b in range(B):
        for h in range(H):
            hk = h // rep
            qh = q[b, :, h, :].astype(np.float64)
            kh = k[b, :, hk, :].astype(np.float64)
            vh = v[b, :, hk, :].astype(np.float64)
            s = _causal_masked_scores(qh, kh, scale)
            if off_by_one:
                # SoftmaxOne appends a zero "sink" logit to the denominator
                # (off-by-one softmax): weights sum to < 1.
                s_aug = np.concatenate([s, np.zeros((s.shape[0], 1))], axis=-1)
                p = _np_softmax(s_aug, axis=-1)[:, :-1]
            else:
                p = _np_softmax(s, axis=-1)
            out[b, :, h, :] = p @ vh
    return out.reshape(B, S, H * Dv)


def _fixed_inputs(num_q_heads, num_kv_heads, dtype="float32"):
    """Deterministic, distinguishable q/k/v; returns (paddle tensors, np copies)."""
    rng = np.random.RandomState(0)
    q = rng.standard_normal((_BATCH, _SEQ, num_q_heads, _HEAD_DIM)) * 0.5
    k = rng.standard_normal((_BATCH, _SEQ, num_kv_heads, _HEAD_DIM)) * 0.5
    v = rng.standard_normal((_BATCH, _SEQ, num_kv_heads, _HEAD_DIM)) * 0.5
    q_np, k_np, v_np = (
        q.astype(np.float32),
        k.astype(np.float32),
        v.astype(np.float32),
    )
    tq = paddle.to_tensor(q_np, dtype=dtype)
    tk = paddle.to_tensor(k_np, dtype=dtype)
    tv = paddle.to_tensor(v_np, dtype=dtype)
    return (tq, tk, tv), (q_np, k_np, v_np)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestDotProductAttentionConstructor(unittest.TestCase):
    """Constructor plumbing that later gates real forward behavior."""

    def test_default_softmax_scale_is_inv_sqrt_head_dim(self):
        # projection_size = head_dim * num_heads, so
        # hidden_size_per_attention_head == head_dim and the default scale is
        # 1/sqrt(head_dim). The flag must stay False so the SDPA path does NOT
        # forward an explicit ``scale`` kwarg.
        attn = _build_attn(_make_config())
        self.assertAlmostEqual(
            attn.softmax_scale, 1.0 / math.sqrt(_HEAD_DIM), places=6
        )
        self.assertFalse(attn._has_custom_softmax_scale)

    def test_custom_softmax_scale_sets_flag(self):
        attn = _build_attn(_make_config(), softmax_scale=0.25)
        self.assertEqual(attn.softmax_scale, 0.25)
        # The flag is load-bearing: it is what routes an explicit scale into the
        # fused/flash kernels instead of their internal default.
        self.assertTrue(attn._has_custom_softmax_scale)

    def test_qk_layer_scaling_divides_scale_and_arms_softmax_coeff(self):
        # layer_number > 1 so coeff != 1 (the source's layer_number=1 is a
        # degenerate no-op). Both halves of Megatron's QK-scaling trick must be
        # wired: softmax_scale is pre-divided by coeff, and the softmax stage is
        # armed with scale=coeff to multiply it back in fp32.
        attn = _build_attn(
            _make_config(apply_query_key_layer_scaling=True), layer_number=3
        )
        base = 1.0 / math.sqrt(_HEAD_DIM)
        self.assertAlmostEqual(attn.softmax_scale, base / 3, places=6)
        self.assertEqual(attn.scale_mask_softmax.scale, 3)
        self.assertTrue(attn._has_custom_softmax_scale)

    def test_softmax_type_vanilla_has_no_offset(self):
        attn = _build_attn(_make_config(softmax_type="vanilla"))
        self.assertIsNone(attn.softmax_offset)

    def test_softmax_type_off_by_one_offset_is_zeros(self):
        attn = _build_attn(_make_config(softmax_type="off-by-one"))
        self.assertIsNotNone(attn.softmax_offset)
        self.assertEqual(list(attn.softmax_offset.shape), [_NUM_HEADS])
        # Content, not just shape: off-by-one uses a fixed zero sink logit.
        np.testing.assert_array_equal(
            attn.softmax_offset.numpy(), np.zeros(_NUM_HEADS, dtype=np.float32)
        )

    def test_invalid_softmax_type_raises(self):
        with self.assertRaises(ValueError):
            _build_attn(_make_config(softmax_type="not-a-real-type"))


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestDotProductAttentionForwardFp32(unittest.TestCase):
    """fp32 forward numeric behavior (CPU-executable matmul/baddbmm path)."""

    def _check_matches_reference(self, config, off_by_one=False):
        attn = _build_attn(config)
        attn.eval()
        (tq, tk, tv), (q_np, k_np, v_np) = _fixed_inputs(_NUM_HEADS, _NUM_HEADS)
        # Independent scale, not read back from the layer.
        scale = 1.0 / math.sqrt(_HEAD_DIM)
        expected = _reference_attention(
            q_np, k_np, v_np, scale, off_by_one=off_by_one
        )
        out = attn(tq, tk, tv, None)
        self.assertEqual(list(out.shape), [_BATCH, _SEQ, _HIDDEN])
        # Tight tolerance: a wrong scale, missing causal mask, transposed head
        # layout, or swapped Q/K would move values far beyond this.
        np.testing.assert_allclose(
            out.numpy().astype(np.float64),
            expected,
            rtol=1e-4,
            atol=1e-5,
        )
        return out.numpy().astype(np.float64), expected

    def test_forward_matches_manual_causal_attention(self):
        self._check_matches_reference(_make_config())

    def test_eager_forward_matches_manual_causal_attention(self):
        # ``_attn_implementation='eager'`` routes fp32 through _EagerQKScoresFn
        # (baddbmm forward) instead of the default baddbmm branch; the causal
        # attention result must be identical.
        config = _make_config()
        config._attn_implementation = "eager"
        self._check_matches_reference(config)

    def test_off_by_one_softmax_path_raises_on_this_paddle(self):
        # Production bug (documented, not worked around): the off-by-one path
        # SoftmaxOne.forward in paddlefleet/fusions/fused_softmax.py:50 calls
        #   paddle.softmax(qk, axis=-1)
        # On this Paddle build ``paddle.softmax`` routes to
        # ``paddle.compat.nn.functional.softmax``, which rejects the ``axis``
        # keyword (it expects ``dim``), so any softmax_type='off-by-one' forward
        # raises TypeError before producing an output. Captured with
        # assertRaises so the broken path is pinned without modifying src/.
        attn = _build_attn(_make_config(softmax_type="off-by-one"))
        attn.eval()
        (tq, tk, tv), _ = _fixed_inputs(_NUM_HEADS, _NUM_HEADS)
        with self.assertRaises(TypeError):
            attn(tq, tk, tv, None)

    def test_gqa_shares_kv_heads_across_query_heads(self):
        # num_key_value_heads=1 < num_attention_heads=2: the single kv head is
        # repeat_interleaved so BOTH query heads read it. Distinguishable q/k/v
        # would expose a wrong repeat axis or count.
        config = _make_config(num_key_value_heads=1)
        attn = _build_attn(config)
        attn.eval()
        (tq, tk, tv), (q_np, k_np, v_np) = _fixed_inputs(
            num_q_heads=_NUM_HEADS, num_kv_heads=1
        )
        scale = 1.0 / math.sqrt(_HEAD_DIM)
        expected = _reference_attention(q_np, k_np, v_np, scale)
        out = attn(tq, tk, tv, None)
        self.assertEqual(list(out.shape), [_BATCH, _SEQ, _HIDDEN])
        np.testing.assert_allclose(
            out.numpy().astype(np.float64), expected, rtol=1e-4, atol=1e-5
        )


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestDotProductAttentionGuards(unittest.TestCase):
    """Explicit input-contract guards in forward."""

    def test_attention_bias_is_rejected(self):
        attn = _build_attn(_make_config())
        attn.eval()
        (tq, tk, tv), _ = _fixed_inputs(_NUM_HEADS, _NUM_HEADS)
        bias = paddle.zeros([_BATCH, _NUM_HEADS, _SEQ, _SEQ], dtype="float32")
        with self.assertRaises(AssertionError):
            attn(tq, tk, tv, None, attention_bias=bias)

    def test_eager_with_packed_seq_params_raises_valueerror(self):
        config = _make_config()
        config._attn_implementation = "eager"
        attn = _build_attn(config)
        attn.eval()
        (tq, tk, tv), _ = _fixed_inputs(_NUM_HEADS, _NUM_HEADS)
        with self.assertRaises(ValueError) as ctx:
            # packed_seq_params is only inspected after the guard; a plain
            # sentinel is enough to trip it.
            attn(tq, tk, tv, None, packed_seq_params=object())
        self.assertIn("packed_seq_params", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
