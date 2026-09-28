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

"""Behavior tests for paddlefleet.transformer.dot_product_attention.

Two surfaces are exercised on CPU (no distributed init, no flash-attention
kernel required):

1. ``scaled_dot_product_attention_with_softmax_offset`` -- the pure-paddle
   "attention sink" / off-by-one softmax path. Its output is compared against
   an INDEPENDENT numpy re-derivation of the documented formula
   ``weights = exp(scores) / (sum(exp(scores)) + exp(sink))`` so that a wrong
   scale, a dropped sink term, or a broken GQA reshape is rejected.

2. ``DotProductAttention.__init__`` / ``build_softmax_offset`` -- the config
   -> ``softmax_scale`` / head-partition / ``softmax_offset`` derivation.

The whole module is skipped only when paddle (or the ops package the module
imports at import time) is genuinely unavailable, and only on
ImportError/ModuleNotFoundError -- never on an arbitrary exception.
"""

import math
import unittest
from types import SimpleNamespace

_IMPORT_ERROR = None
try:
    import numpy as np
    import paddle

    from paddlefleet.transformer.dot_product_attention import (
        DotProductAttention,
        build_softmax_offset,
        scaled_dot_product_attention_with_softmax_offset,
    )
    from paddlefleet.transformer.transformer_config import TransformerConfig
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    None
    if _IMPORT_ERROR is None
    else f"paddle / paddlefleet import unavailable in this environment: {_IMPORT_ERROR!r}"
)


def _reference_sink_attention(
    query, key, value, scale, softmax_offset, is_causal
):
    """Independent numpy reference for the sink/off-by-one attention.

    Layout mirrors the production entry:
      query: [B, Q, Hq, dq]   key/value: [B, K, Hkv, d]
    Semantics (re-derived from the documented algorithm, NOT copied from the
    production reshape tricks):
      scores[b,h,i,j] = scale * <q[b,i,h], k[b,j,h//groups]>
      sink_h          = softmax_offset[h]  (virtual extra logit)
      w[b,h,i,j]      = exp(scores) / (sum_j exp(scores) + exp(sink_h))
      out[b,i,h]      = sum_j w[b,h,i,j] * v[b,j,h//groups]
    """
    q = np.asarray(query, dtype=np.float64)
    k = np.asarray(key, dtype=np.float64)
    v = np.asarray(value, dtype=np.float64)
    off = np.asarray(softmax_offset, dtype=np.float64)

    bsz, q_len, hq, dq = q.shape
    _, kv_len, hkv, _ = k.shape
    groups = hq // hkv
    dv = v.shape[-1]

    out = np.zeros((bsz, q_len, hq, dv), dtype=np.float64)
    for b in range(bsz):
        for h in range(hq):
            kvh = h // groups
            # scores[i, j]
            scores = scale * (q[b, :, h, :] @ k[b, :, kvh, :].T)
            if is_causal and q_len > 1:
                for i in range(q_len):
                    for j in range(kv_len):
                        if j > i + (kv_len - q_len):
                            scores[i, j] = -np.inf
            sink = off[h]
            # numerically identical to production's stabilized form
            row_max = np.maximum(scores.max(axis=-1, keepdims=True), sink)
            exp_s = np.exp(scores - row_max)
            denom = exp_s.sum(axis=-1, keepdims=True) + np.exp(sink - row_max)
            w = exp_s / denom
            out[b, :, h, :] = w @ v[b, :, kvh, :]
    return out


def _plain_softmax_attention(query, key, value, scale):
    """Reference for ordinary softmax (NO sink). Used as a negative control:
    the sink path must NOT coincide with this when the sink term is present."""
    q = np.asarray(query, dtype=np.float64)
    k = np.asarray(key, dtype=np.float64)
    v = np.asarray(value, dtype=np.float64)
    bsz, q_len, hq, dq = q.shape
    hkv = k.shape[2]
    groups = hq // hkv
    dv = v.shape[-1]
    out = np.zeros((bsz, q_len, hq, dv), dtype=np.float64)
    for b in range(bsz):
        for h in range(hq):
            kvh = h // groups
            scores = scale * (q[b, :, h, :] @ k[b, :, kvh, :].T)
            e = np.exp(scores - scores.max(axis=-1, keepdims=True))
            w = e / e.sum(axis=-1, keepdims=True)
            out[b, :, h, :] = w @ v[b, :, kvh, :]
    return out


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON)
class SinkAttentionNumericTest(unittest.TestCase):
    """Numeric behavior of scaled_dot_product_attention_with_softmax_offset."""

    def setUp(self):
        # Force CPU and restore the caller's device afterwards (rule 11:
        # any global mutation must be undone even if the test fails).
        self._orig_device = paddle.get_device()
        paddle.set_device("cpu")
        self.addCleanup(paddle.set_device, self._orig_device)

    def test_mha_offbyone_matches_hand_derived_values(self):
        # B=1, Q=1, Hq=Hkv=1, dq=dk=2, K=2. scale=0.5 chosen so that ignoring
        # the scale would change the scores (raw dots [0, 2] -> scaled [0, 1]).
        query = paddle.to_tensor([[[[1.0, 0.0]]]], dtype="float32")  # [1,1,1,2]
        key = paddle.to_tensor(
            [[[[0.0, 9.0]], [[2.0, 7.0]]]], dtype="float32"
        )  # [1,2,1,2] -> raw dots with q are [0, 2]
        value = paddle.to_tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]]]], dtype="float32"
        )  # [1,2,1,2]
        offset = paddle.zeros([1])  # off-by-one: virtual logit == 0

        out = scaled_dot_product_attention_with_softmax_offset(
            query,
            key,
            value,
            softmax_offset=offset,
            q_head_dim=2,
            scale=0.5,
        )

        # Hand derivation: scores=[0,1], denom = e^0 + e^1 + e^0 = 2 + e
        denom = 2.0 + math.e
        w0 = 1.0 / denom
        w1 = math.e / denom
        expected = np.array([[[[w0 * 1.0, w1 * 1.0]]]], dtype=np.float64)
        self.assertEqual(list(out.shape), [1, 1, 1, 2])
        np.testing.assert_allclose(
            out.numpy().astype(np.float64), expected, rtol=1e-6, atol=1e-7
        )
        # off-by-one attention weights must sum to < 1 (mass leaks to sink).
        self.assertLess(w0 + w1, 1.0)

    def test_sink_term_is_actually_consumed(self):
        # A large positive sink should pull attention mass away from the keys,
        # making the output differ from an ordinary (no-sink) softmax. If the
        # sink offset were ignored, these would coincide.
        rng = np.random.default_rng(0)
        q = rng.standard_normal((2, 3, 2, 4)).astype("float32")
        k = rng.standard_normal((2, 5, 2, 4)).astype("float32")
        v = rng.standard_normal((2, 5, 2, 4)).astype("float32")
        scale = 0.3
        offset = paddle.to_tensor([2.0, -1.0], dtype="float32")

        out = (
            scaled_dot_product_attention_with_softmax_offset(
                paddle.to_tensor(q),
                paddle.to_tensor(k),
                paddle.to_tensor(v),
                softmax_offset=offset,
                q_head_dim=4,
                scale=scale,
            )
            .numpy()
            .astype(np.float64)
        )

        ref_sink = _reference_sink_attention(q, k, v, scale, [2.0, -1.0], False)
        ref_plain = _plain_softmax_attention(q, k, v, scale)

        # Matches the independent sink reference exactly ...
        np.testing.assert_allclose(out, ref_sink, rtol=1e-5, atol=1e-6)
        # ... and is meaningfully different from plain softmax (sink consumed).
        self.assertGreater(np.abs(out - ref_plain).max(), 1e-3)

    def test_gqa_grouping_uses_correct_kv_head(self):
        # Hq=4, Hkv=2, groups=2. Distinct per-head q content and distinct kv
        # content so that a wrong q-head -> kv-head mapping would change output.
        rng = np.random.default_rng(7)
        q = rng.standard_normal((1, 3, 4, 4)).astype("float32")
        k = rng.standard_normal((1, 6, 2, 4)).astype("float32")
        v = rng.standard_normal((1, 6, 2, 4)).astype("float32")
        scale = 1.0 / math.sqrt(4)
        offset = paddle.zeros([4])

        out = (
            scaled_dot_product_attention_with_softmax_offset(
                paddle.to_tensor(q),
                paddle.to_tensor(k),
                paddle.to_tensor(v),
                softmax_offset=offset,
                q_head_dim=4,
                scale=scale,
            )
            .numpy()
            .astype(np.float64)
        )

        ref = _reference_sink_attention(q, k, v, scale, [0.0] * 4, False)
        self.assertEqual(list(out.shape), [1, 3, 4, 4])
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)

    def test_causal_masking_matches_reference(self):
        # Q==K==3 -> lower-triangular visibility (diagonal offset K-Q == 0).
        rng = np.random.default_rng(11)
        q = rng.standard_normal((1, 3, 1, 4)).astype("float32")
        k = rng.standard_normal((1, 3, 1, 4)).astype("float32")
        v = rng.standard_normal((1, 3, 1, 4)).astype("float32")
        scale = 0.5
        offset = paddle.zeros([1])

        out = (
            scaled_dot_product_attention_with_softmax_offset(
                paddle.to_tensor(q),
                paddle.to_tensor(k),
                paddle.to_tensor(v),
                softmax_offset=offset,
                q_head_dim=4,
                scale=scale,
                is_causal=True,
            )
            .numpy()
            .astype(np.float64)
        )

        ref = _reference_sink_attention(q, k, v, scale, [0.0], True)
        # Sanity: position 0 attends to key 0 only; independent check that the
        # causal mask really removed keys 1 and 2 for the first query row.
        row0_expected = _reference_sink_attention(
            q[:, :1], k[:, :1], v[:, :1], scale, [0.0], False
        )
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            out[:, :1], row0_expected, rtol=1e-5, atol=1e-6
        )


def _make_config(**overrides):
    defaults = {
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 32,
        "context_parallel_size": 1,
        "attention_dropout": 0.0,
        "attention_softmax_in_fp32": True,
        "masked_softmax_fusion": False,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _pg_stub(tp_world_size=1):
    # The process-group collection is a genuine not-under-test collaborator;
    # a plain namespace with the single attribute the constructor reads
    # (``.tp.world_size``) is enough and avoids MagicMock auto-attributes.
    return SimpleNamespace(tp=SimpleNamespace(world_size=tp_world_size))


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON)
class DotProductAttentionConstructionTest(unittest.TestCase):
    """Config -> softmax_scale / head partition / softmax_offset derivation."""

    def setUp(self):
        self._orig_device = paddle.get_device()
        paddle.set_device("cpu")
        self.addCleanup(paddle.set_device, self._orig_device)

    def _build(self, config, **kwargs):
        kwargs.setdefault("layer_number", 1)
        kwargs.setdefault("attn_mask_type", "padding")
        kwargs.setdefault("attention_type", "self")
        kwargs.setdefault("pg_collection", _pg_stub())
        return DotProductAttention(config, **kwargs)

    def test_default_softmax_scale_is_inverse_sqrt_head_dim(self):
        dpa = self._build(_make_config(head_dim=64, num_attention_heads=4))
        # projection_size / num_heads == head_dim == 64 -> 1/sqrt(64) == 0.125.
        self.assertAlmostEqual(dpa.softmax_scale, 1.0 / math.sqrt(64), places=7)
        self.assertEqual(dpa.softmax_scale, 0.125)
        # No explicit scale was given -> the "custom scale" flag stays False,
        # which routes the forward pass to the default-scale kernel branch.
        self.assertFalse(dpa._has_custom_softmax_scale)

    def test_explicit_softmax_scale_overrides_and_sets_custom_flag(self):
        dpa = self._build(_make_config(head_dim=32), softmax_scale=0.5)
        self.assertEqual(dpa.softmax_scale, 0.5)
        self.assertTrue(dpa._has_custom_softmax_scale)

    def test_query_key_layer_scaling_divides_by_layer_number(self):
        # layer_number=2 -> coeff = max(1, 2) = 2; base = 1/sqrt(32).
        dpa = self._build(
            _make_config(head_dim=32, apply_query_key_layer_scaling=True),
            layer_number=2,
        )
        expected = (1.0 / math.sqrt(32)) / 2.0
        self.assertAlmostEqual(dpa.softmax_scale, expected, places=7)
        # QK-layer-scaling forces the custom-scale flag even without an
        # explicit softmax_scale argument.
        self.assertTrue(dpa._has_custom_softmax_scale)

    def test_gqa_head_partition_counts(self):
        config = _make_config(
            num_attention_heads=8, num_key_value_heads=4, head_dim=32
        )
        dpa = self._build(config, pg_collection=_pg_stub(tp_world_size=2))
        # divide(8, 2) and divide(4, 2)
        self.assertEqual(dpa.num_attention_heads_per_partition, 4)
        self.assertEqual(dpa.num_query_groups_per_partition, 2)

    def test_vanilla_softmax_has_no_offset(self):
        dpa = self._build(_make_config(softmax_type="vanilla"))
        self.assertIsNone(dpa.softmax_offset)

    def test_off_by_one_offset_is_fixed_zero_tensor_not_parameter(self):
        dpa = self._build(_make_config(softmax_type="off-by-one"))
        self.assertIsNotNone(dpa.softmax_offset)
        # off-by-one uses a NON-trainable zero tensor sized per partition ...
        self.assertNotIsInstance(dpa.softmax_offset, paddle.nn.Parameter)
        self.assertEqual(
            list(dpa.softmax_offset.shape),
            [dpa.num_attention_heads_per_partition],
        )
        np.testing.assert_array_equal(
            dpa.softmax_offset.numpy(),
            np.zeros(dpa.num_attention_heads_per_partition, dtype=np.float32),
        )

    def test_learnable_offset_is_trainable_parameter(self):
        dpa = self._build(
            _make_config(softmax_type="learnable", perform_initialization=False)
        )
        # ... whereas learnable uses a trainable Parameter of the same length.
        self.assertIsInstance(dpa.softmax_offset, paddle.nn.Parameter)
        self.assertEqual(
            list(dpa.softmax_offset.shape),
            [dpa.num_attention_heads_per_partition],
        )

    def test_full_attention_sink_bias_promotes_vanilla_to_learnable(self):
        # softmax_type stays "vanilla", but add_full_attention_sink_bias on a
        # non-SWA layer must promote it to a learnable sink Parameter.
        config = _make_config(
            softmax_type="vanilla",
            add_full_attention_sink_bias=True,
            perform_initialization=False,
        )
        dpa = self._build(config, is_swa=False)
        self.assertIsInstance(dpa.softmax_offset, paddle.nn.Parameter)

    def test_invalid_softmax_type_raises_value_error(self):
        config = _make_config(softmax_type="not-a-real-type")
        with self.assertRaises(ValueError):
            self._build(config)


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON)
class BuildSoftmaxOffsetTest(unittest.TestCase):
    """Direct behavior of the shared build_softmax_offset helper."""

    def setUp(self):
        self._orig_device = paddle.get_device()
        paddle.set_device("cpu")
        self.addCleanup(paddle.set_device, self._orig_device)

    def test_swa_sink_bias_only_applies_to_swa_layers(self):
        # add_swa_attention_sink_bias defaults True. It must promote to
        # learnable ONLY when is_swa=True; a full-attention (is_swa=False)
        # layer with vanilla softmax and no full-attention sink stays None.
        config = _make_config(
            softmax_type="vanilla",
            add_full_attention_sink_bias=False,
            add_swa_attention_sink_bias=True,
            perform_initialization=False,
        )
        layer = paddle.nn.Layer()

        full_offset = build_softmax_offset(
            layer, config, num_heads=4, is_swa=False
        )
        self.assertIsNone(full_offset)

        swa_offset = build_softmax_offset(
            layer, config, num_heads=4, is_swa=True
        )
        self.assertIsInstance(swa_offset, paddle.nn.Parameter)
        self.assertEqual(list(swa_offset.shape), [4])


if __name__ == "__main__":
    unittest.main()
