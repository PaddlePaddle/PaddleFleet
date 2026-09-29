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

"""Behaviour tests for the virtual-sink SDPA and the softmax-offset factory
of ``paddlefleet.transformer.dot_product_attention``.

Targets two CPU-executable pieces of production logic:

* ``scaled_dot_product_attention_with_softmax_offset`` -- the eager fp32
  attention with a virtual "sink" token in the softmax denominator (used by
  the ``off-by-one`` / ``learnable`` softmax types). Expected values are
  derived two ways that never call the code under test: (a) a from-scratch
  float64 numpy reference that expands K/V heads (production instead reshapes
  Q and broadcasts K/V, so a grouping mistake on either side surfaces as a
  numeric mismatch), and (b) closed-form hand anchors for the e^-1 sink case.
* ``build_softmax_offset`` -- the per-head sink parameter factory, including
  the ``add_*_attention_sink_bias`` promotion that depends on ``is_swa``.

No GPU/flash path is exercised here; imports are guarded so a build without
paddle skips honestly rather than reporting a false pass.
"""

import math
import types
import unittest

import numpy as np

try:
    import paddle

    from paddlefleet.transformer.dot_product_attention import (
        build_softmax_offset,
        scaled_dot_product_attention_with_softmax_offset,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as _exc:  # no paddle build available
    paddle = None
    build_softmax_offset = None
    scaled_dot_product_attention_with_softmax_offset = None
    _IMPORT_ERROR = _exc

_SKIP_REASON = (
    f"paddle/paddlefleet not importable in this environment: {_IMPORT_ERROR!r}"
)


def _ref_sink_attention(
    query,
    key,
    value,
    softmax_offset,
    scale,
    is_causal=False,
    attn_mask=None,
    mask_is_bool=False,
):
    """Independent float64 reference for the virtual-sink SDPA.

    Uses a different strategy than production: it materialises the grouped-query
    expansion by repeating K/V heads (production reshapes Q and broadcasts K/V).
    Query head ``h`` attends KV head ``h // groups`` in both, so a grouping bug
    on either side becomes a numeric mismatch instead of cancelling out.
    """
    q = np.asarray(query, np.float64).transpose(0, 2, 1, 3)  # [B, Hq, Q, dq]
    k = np.asarray(key, np.float64).transpose(0, 2, 1, 3)  # [B, Hkv, K, dk]
    v = np.asarray(value, np.float64).transpose(0, 2, 1, 3)  # [B, Hkv, K, dv]
    _, hq, qn, _ = q.shape
    hkv, kn = k.shape[1], k.shape[2]
    groups = hq // hkv
    k = np.repeat(k, groups, axis=1)  # head h uses kv head h // groups
    v = np.repeat(v, groups, axis=1)
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * float(
        scale
    )  # [B, Hq, Q, K]
    if is_causal and qn > 1:
        causal = np.tril(np.ones((qn, kn)), k=kn - qn)
        scores = np.where(causal[None, None] == 0, -np.inf, scores)
    if attn_mask is not None:
        m = np.asarray(attn_mask)
        if mask_is_bool:
            scores = np.where(m.astype(bool), -np.inf, scores)
        else:
            scores = scores + m.astype(np.float64)
    sink = np.asarray(softmax_offset, np.float64).reshape(1, hq, 1, 1)
    row_max = np.maximum(scores.max(-1, keepdims=True), sink)
    exp_s = np.exp(scores - row_max)
    row_sum = exp_s.sum(-1, keepdims=True) + np.exp(sink - row_max)
    weights = exp_s / row_sum
    out = np.matmul(weights, v)  # [B, Hq, Q, dv]
    return out.transpose(0, 2, 1, 3)  # [B, Q, Hq, dv]


def _run(query, key, value, offset, **kwargs):
    """Call the production SDPA on fp32 CPU tensors, return a numpy array."""
    out = scaled_dot_product_attention_with_softmax_offset(
        paddle.to_tensor(query, dtype="float32"),
        paddle.to_tensor(key, dtype="float32"),
        paddle.to_tensor(value, dtype="float32"),
        softmax_offset=paddle.to_tensor(offset, dtype="float32"),
        q_head_dim=int(np.asarray(query).shape[-1]),
        **kwargs,
    )
    return out.numpy()


@unittest.skipUnless(paddle is not None, _SKIP_REASON)
class TestSinkAttentionMHA(unittest.TestCase):
    """Multi-head (groups==1) virtual-sink attention numerics."""

    def setUp(self):
        # Positions/keys chosen so scores are the 2x2 identity: pos0 aligns with
        # key0, pos1 with key1. Values are distinct per key and per channel.
        self.q = np.array([[[[1.0, 0.0]], [[0.0, 1.0]]]])  # [1, 2, 1, 2]
        self.k = np.array([[[[1.0, 0.0]], [[0.0, 1.0]]]])  # [1, 2, 1, 2]
        self.v = np.array([[[[1.0, 2.0]], [[3.0, 4.0]]]])  # [1, 2, 1, 2]
        self.offset = np.array([0.0])  # Hq == 1

    def test_matches_independent_reference(self):
        out = _run(self.q, self.k, self.v, self.offset, scale=1.0)
        ref = _ref_sink_attention(self.q, self.k, self.v, self.offset, 1.0)
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)

    def test_closed_form_off_by_one_anchor(self):
        # sink offset 0 => "off-by-one" softmax: denom = sum(exp) + 1.
        # scores are identity, so each row has one 1.0 and one 0.0 logit.
        # rowsum = 1 + 2*e^-1; weights = [1, e^-1]/rowsum (order per row).
        out = _run(self.q, self.k, self.v, self.offset, scale=1.0)
        e1 = math.exp(-1.0)
        rs = 1.0 + 2.0 * e1
        expected = np.array(
            [
                [
                    [[(1.0 + 3.0 * e1) / rs, (2.0 + 4.0 * e1) / rs]],
                    [[(3.0 + e1) / rs, (4.0 + 2.0 * e1) / rs]],
                ]
            ]
        )
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)
        # pos0 channel-1 is 2.0 exactly because (2 + 4 e^-1) = 2 * rowsum; this
        # pins the sink into the denominator (a plain softmax would give a
        # different value once the +sink term is dropped).
        self.assertAlmostEqual(float(out[0, 0, 0, 1]), 2.0, places=5)

    def test_sink_offset_is_consumed(self):
        base = _run(self.q, self.k, self.v, self.offset, scale=1.0)
        # A huge sink dominates the denominator, driving every weight -> 0, so
        # the output collapses toward zero. If the offset were ignored the
        # output would be unchanged.
        big = _run(self.q, self.k, self.v, np.array([50.0]), scale=1.0)
        self.assertGreater(np.abs(base).max(), 0.5)
        self.assertLess(np.abs(big).max(), 1e-6)
        self.assertFalse(np.allclose(base, big))

    def test_default_scale_is_inverse_sqrt_head_dim(self):
        out = _run(self.q, self.k, self.v, self.offset)  # scale=None
        ref = _ref_sink_attention(
            self.q, self.k, self.v, self.offset, scale=2.0**-0.5
        )
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


@unittest.skipUnless(paddle is not None, _SKIP_REASON)
class TestSinkAttentionGQA(unittest.TestCase):
    """Group-query (groups>1) path: Q reshaped, K/V broadcast, per-head sink."""

    def setUp(self):
        # Hq=2, Hkv=1 (groups=2). Distinct query per (position, head) so the two
        # heads cannot coincide, and per-head sinks differ.
        self.q = np.array(
            [
                [
                    [[1.0, 0.0], [0.0, 1.0]],  # pos0: head0=[1,0], head1=[0,1]
                    [[0.0, 1.0], [1.0, 0.0]],  # pos1: head0=[0,1], head1=[1,0]
                ]
            ]
        )  # [1, 2, 2, 2]
        self.k = np.array([[[[1.0, 0.0]], [[0.0, 1.0]]]])  # [1, 2, 1, 2] Hkv=1
        self.v = np.array([[[[1.0, 2.0]], [[3.0, 4.0]]]])
        self.offset = np.array([0.0, 0.5])  # per-head sink, Hq=2

    def test_matches_independent_reference(self):
        out = _run(self.q, self.k, self.v, self.offset, scale=1.0)
        ref = _ref_sink_attention(self.q, self.k, self.v, self.offset, 1.0)
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)

    def test_heads_are_not_degenerate(self):
        out = _run(self.q, self.k, self.v, self.offset, scale=1.0)
        # head0 and head1 see different queries -> different outputs.
        self.assertFalse(np.allclose(out[0, :, 0, :], out[0, :, 1, :]))

    def test_per_head_sink_is_positional(self):
        out = _run(self.q, self.k, self.v, self.offset, scale=1.0)
        # Swapping the per-head sink values must change the result; a bug that
        # broadcast a single sink across heads would leave this unchanged.
        swapped = _run(self.q, self.k, self.v, np.array([0.5, 0.0]), scale=1.0)
        self.assertFalse(np.allclose(out, swapped))


@unittest.skipUnless(paddle is not None, _SKIP_REASON)
class TestSinkAttentionCausalAndMask(unittest.TestCase):
    """Causal masking and explicit key masks combined with the sink."""

    def setUp(self):
        # 3 positions, 3 keys, single head, dim 2. Distinct, non-orthogonal q/k
        # so scores are non-degenerate across positions.
        self.q = np.array([[[[1.0, 0.0]], [[0.5, 0.5]], [[0.0, 1.0]]]])
        self.k = np.array([[[[0.0, 1.0]], [[1.0, 1.0]], [[1.0, 0.0]]]])
        self.v = np.array([[[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]])
        self.offset = np.array([0.0])

    def test_causal_matches_reference(self):
        out = _run(
            self.q, self.k, self.v, self.offset, scale=1.0, is_causal=True
        )
        ref = _ref_sink_attention(
            self.q, self.k, self.v, self.offset, 1.0, is_causal=True
        )
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)

    def test_causal_first_position_sees_only_first_key(self):
        out = _run(
            self.q, self.k, self.v, self.offset, scale=1.0, is_causal=True
        )
        # pos0 attends only key0 (+ virtual sink). Hand-derive that single-key
        # softmax independently, not via the code under test.
        s0 = float(np.dot(self.q[0, 0, 0], self.k[0, 0, 0]))  # scale == 1.0
        rm = max(s0, 0.0)
        w0 = math.exp(s0 - rm) / (math.exp(s0 - rm) + math.exp(0.0 - rm))
        expected_pos0 = w0 * self.v[0, 0, 0]
        np.testing.assert_allclose(
            out[0, 0, 0], expected_pos0, rtol=1e-5, atol=1e-6
        )

    def test_causal_changes_early_positions(self):
        causal = _run(
            self.q, self.k, self.v, self.offset, scale=1.0, is_causal=True
        )
        full = _run(
            self.q, self.k, self.v, self.offset, scale=1.0, is_causal=False
        )
        # pos0 under full attention can see later keys; causal must differ.
        self.assertFalse(np.allclose(causal[0, 0], full[0, 0]))

    def test_bool_mask_blocks_key(self):
        # True == masked out. Block key1 for every query.
        mask = np.zeros((1, 1, 3, 3), dtype=bool)
        mask[..., 1] = True
        out = _run(
            self.q,
            self.k,
            self.v,
            self.offset,
            scale=1.0,
            attn_mask_kv=paddle.to_tensor(mask, dtype="bool"),
        )
        ref = _ref_sink_attention(
            self.q,
            self.k,
            self.v,
            self.offset,
            1.0,
            attn_mask=mask,
            mask_is_bool=True,
        )
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)
        unmasked = _run(self.q, self.k, self.v, self.offset, scale=1.0)
        self.assertFalse(np.allclose(out, unmasked))

    def test_float_mask_is_additive(self):
        mask = np.zeros((1, 1, 3, 3), dtype=np.float32)
        mask[..., 2] = -1.0e4  # strongly suppress key2
        out = _run(
            self.q,
            self.k,
            self.v,
            self.offset,
            scale=1.0,
            attn_mask_kv=paddle.to_tensor(mask, dtype="float32"),
        )
        ref = _ref_sink_attention(
            self.q,
            self.k,
            self.v,
            self.offset,
            1.0,
            attn_mask=mask,
            mask_is_bool=False,
        )
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


def _offset_config(
    softmax_type="vanilla",
    add_full=False,
    add_swa=False,
    perform_initialization=True,
    init_method=None,
):
    """Minimal stand-in for the fields build_softmax_offset reads."""
    return types.SimpleNamespace(
        softmax_type=softmax_type,
        add_full_attention_sink_bias=add_full,
        add_swa_attention_sink_bias=add_swa,
        params_dtype="float32",
        perform_initialization=perform_initialization,
        init_method=init_method,
    )


class _RecordingInit:
    """A genuine (non-mocked) initializer that also records invocation."""

    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __call__(self, tensor):
        self.calls += 1
        tensor.set_value(
            paddle.full(tensor.shape, self.value, dtype=tensor.dtype)
        )


@unittest.skipUnless(paddle is not None, _SKIP_REASON)
class TestBuildSoftmaxOffset(unittest.TestCase):
    """Per-head sink parameter factory: type switch and is_swa promotion."""

    def _layer(self):
        # A real Layer so create_parameter runs the genuine parameter path.
        return paddle.nn.Layer()

    def test_vanilla_returns_none(self):
        offset = build_softmax_offset(
            self._layer(), _offset_config("vanilla"), num_heads=4, is_swa=False
        )
        self.assertIsNone(offset)

    def test_off_by_one_is_zero_vector(self):
        offset = build_softmax_offset(
            self._layer(),
            _offset_config("off-by-one"),
            num_heads=4,
            is_swa=False,
        )
        self.assertIsNotNone(offset)
        self.assertEqual(list(offset.shape), [4])
        np.testing.assert_array_equal(offset.numpy(), np.zeros(4, np.float32))

    def test_learnable_creates_initialized_parameter(self):
        init = _RecordingInit(0.125)
        offset = build_softmax_offset(
            self._layer(),
            _offset_config(
                "learnable", perform_initialization=True, init_method=init
            ),
            num_heads=3,
            is_swa=False,
        )
        self.assertIsInstance(offset, paddle.Tensor)
        self.assertFalse(offset.stop_gradient)  # trainable parameter
        self.assertEqual(list(offset.shape), [3])
        self.assertEqual(init.calls, 1)
        np.testing.assert_allclose(
            offset.numpy(), np.full(3, 0.125, np.float32), rtol=1e-6
        )

    def test_learnable_skips_init_when_disabled(self):
        init = _RecordingInit(0.125)
        offset = build_softmax_offset(
            self._layer(),
            _offset_config(
                "learnable", perform_initialization=False, init_method=init
            ),
            num_heads=3,
            is_swa=False,
        )
        self.assertIsInstance(offset, paddle.Tensor)
        self.assertEqual(list(offset.shape), [3])
        self.assertEqual(init.calls, 0)  # init_method must not be called

    def test_full_sink_bias_promotes_only_non_swa(self):
        init = _RecordingInit(0.0)
        cfg = lambda: _offset_config("vanilla", add_full=True, init_method=init)
        # non-SWA layer: vanilla is promoted to learnable -> a parameter.
        promoted = build_softmax_offset(
            self._layer(), cfg(), num_heads=2, is_swa=False
        )
        self.assertIsInstance(promoted, paddle.Tensor)
        self.assertEqual(list(promoted.shape), [2])
        # SWA layer: full-attention bias does NOT apply -> stays vanilla -> None.
        self.assertIsNone(
            build_softmax_offset(self._layer(), cfg(), num_heads=2, is_swa=True)
        )

    def test_swa_sink_bias_promotes_only_swa(self):
        init = _RecordingInit(0.0)
        cfg = lambda: _offset_config("vanilla", add_swa=True, init_method=init)
        promoted = build_softmax_offset(
            self._layer(), cfg(), num_heads=2, is_swa=True
        )
        self.assertIsInstance(promoted, paddle.Tensor)
        self.assertEqual(list(promoted.shape), [2])
        self.assertIsNone(
            build_softmax_offset(
                self._layer(), cfg(), num_heads=2, is_swa=False
            )
        )

    def test_unknown_softmax_type_raises(self):
        with self.assertRaises(ValueError):
            build_softmax_offset(
                self._layer(),
                _offset_config("bogus-type"),
                num_heads=4,
                is_swa=False,
            )


if __name__ == "__main__":
    unittest.main()
