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

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn

from paddlefleet.transformer.gated_delta_net import (
    GatedDeltaNet,
    GatedDeltaNetSublayersSpec,
)
from paddlefleet.transformer.transformer_config import TransformerConfig

_GDN_MODULE = "paddlefleet.transformer.gated_delta_net"


class NoBiasLinear(nn.Layer):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias_attr=False)

    def forward(self, x):
        return self.linear(x), None

    def backward_dw(self):
        pass


class SimpleRMSNorm(nn.Layer):
    def __init__(self, normalized_shape, eps=1e-5, **kwargs):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[normalized_shape],
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.eps = eps

    def forward(self, x):
        x_float = x.astype(paddle.float32)
        rms = paddle.rsqrt(
            x_float.pow(2).mean(axis=-1, keepdim=True) + self.eps
        )
        return (x_float * rms * self.weight.astype(paddle.float32)).astype(
            x.dtype
        )


class _FakeGroup:
    ranks = [0]
    nranks = 1
    rank = 0


class _FakePGCollection:
    def __init__(self):
        self.tp = _FakeGroup()


H, B, S = 64, 2, 32


def _make_gdn():
    config = TransformerConfig(
        hidden_size=H,
        num_attention_heads=4,
        num_hidden_layers=2,
        hidden_act=F.silu,
        rms_norm_eps=1e-5,
        normalization="RMSNorm",
        deterministic_mode=True,
    )
    spec = GatedDeltaNetSublayersSpec(
        in_proj=NoBiasLinear,
        out_norm=SimpleRMSNorm,
        out_proj=NoBiasLinear,
    )
    return GatedDeltaNet(
        config=config,
        sublayers_spec=spec,
        layer_number=1,
        bias=False,
        conv_bias=False,
        conv_init=1.0,
        use_qk_l2norm=True,
        A_init_range=(1, 16),
        pg_collection=_FakePGCollection(),
        conv_kernel_dim=4,
        key_head_dim=16,
        value_head_dim=16,
        num_key_heads=4,
        num_value_heads=4,
    )


def _make_gdn_sp():
    """GDN with SP simulated (config.sequence_parallel=True, sp_size=2)."""
    config = TransformerConfig(
        hidden_size=H,
        num_attention_heads=4,
        num_hidden_layers=2,
        hidden_act=F.silu,
        rms_norm_eps=1e-5,
        normalization="RMSNorm",
        deterministic_mode=True,
    )
    config.sequence_parallel = True
    spec = GatedDeltaNetSublayersSpec(
        in_proj=NoBiasLinear,
        out_norm=SimpleRMSNorm,
        out_proj=NoBiasLinear,
    )
    gdn = GatedDeltaNet(
        config=config,
        sublayers_spec=spec,
        layer_number=1,
        bias=False,
        conv_bias=False,
        conv_init=1.0,
        use_qk_l2norm=True,
        A_init_range=(1, 16),
        pg_collection=_FakePGCollection(),
        conv_kernel_dim=4,
        key_head_dim=16,
        value_head_dim=16,
        num_key_heads=4,
        num_value_heads=4,
    )
    gdn.sp_size = 2  # simulate TP=2 + SP
    return gdn


def _make_startend_indices(batch, seq_len, padding_start=None):
    """Create attn_mask_startend_row_indices [b, 1, s, 1]. padding_start=None means all valid."""
    indices = paddle.full(
        [batch, 1, seq_len, 1], fill_value=seq_len, dtype="int64"
    )
    if padding_start is not None:
        for pos in range(padding_start, seq_len):
            indices[:, :, pos, :] = pos
    return indices


def _keep_vector(batch, seq_len, padding_start=None, sample=None):
    """Build a [b, s] 0/1 keep mask. ``sample=None`` pads every sample."""
    keep = paddle.ones([batch, seq_len], dtype="int64")
    if padding_start is not None:
        if sample is None:
            keep[:, padding_start:] = 0
        else:
            keep[sample, padding_start:] = 0
    return keep


def _to_4d_causal(keep, dtype="float32"):
    """Expand a [b, s] keep vector into the collator's [b, 1, s, s] mask.

    Entry ``[i, j]`` is enabled iff ``j <= i`` and both tokens are real, so a
    padding row has a zero diagonal — which is exactly the signal
    ``_build_padding_mask`` reduces back to per-token validity.
    """
    batch, seq_len = keep.shape
    causal = paddle.tril(paddle.ones([seq_len, seq_len], dtype="float32"))
    keep_f = keep.astype("float32")
    mask = causal.unsqueeze(0) * keep_f.unsqueeze(1) * keep_f.unsqueeze(2)
    return mask.unsqueeze(1).astype(dtype)


def _to_4d_block_diagonal(seq_len, doc_lens, dtype="float32"):
    """Packed-sequence mask: causal within each document, blocked across."""
    mask = paddle.zeros([seq_len, seq_len], dtype="float32")
    start = 0
    for length in doc_lens:
        end = start + length
        block = paddle.tril(paddle.ones([length, length], dtype="float32"))
        mask[start:end, start:end] = block
        start = end
    return mask.unsqueeze(0).unsqueeze(0).astype(dtype)


class TestBuildPaddingMaskNoSP(unittest.TestCase):
    """_build_padding_mask without sequence parallel."""

    def setUp(self):
        self.gdn = _make_gdn()

    def test_both_none_returns_none(self):
        self.assertIsNone(self.gdn._build_padding_mask(None, None, B, S))

    def test_attention_mask_ndim3_raises(self):
        with self.assertRaises(ValueError):
            self.gdn._build_padding_mask(paddle.ones([B, 1, S]), None, B, S)

    def test_attention_mask_all_valid_returns_none(self):
        mask = paddle.ones([B, S], dtype="int64")
        self.assertIsNone(self.gdn._build_padding_mask(mask, None, B, S))

    def test_attention_mask_with_padding(self):
        mask = paddle.ones([B, S], dtype="int64")
        mask[0, -8:] = 0
        result = self.gdn._build_padding_mask(mask, None, B, S)
        self.assertEqual(list(result.shape), [B, S, 1])
        np.testing.assert_array_equal(result[0, :24, 0].numpy(), np.ones(24))
        np.testing.assert_array_equal(result[0, 24:, 0].numpy(), np.zeros(8))
        np.testing.assert_array_equal(result[1, :, 0].numpy(), np.ones(S))

    def test_startend_all_valid_returns_none(self):
        indices = _make_startend_indices(B, S, padding_start=None)
        self.assertIsNone(self.gdn._build_padding_mask(None, indices, B, S))

    def test_startend_with_padding(self):
        indices = _make_startend_indices(B, S, padding_start=24)
        result = self.gdn._build_padding_mask(None, indices, B, S)
        self.assertEqual(list(result.shape), [B, S, 1])
        np.testing.assert_array_equal(result[0, :24, 0].numpy(), np.ones(24))
        np.testing.assert_array_equal(result[0, 24:, 0].numpy(), np.zeros(8))

    def test_startend_per_sample_padding(self):
        indices = _make_startend_indices(B, S, padding_start=None)
        for pos in range(20, S):
            indices[0, :, pos, :] = pos  # only sample 0 has padding
        result = self.gdn._build_padding_mask(None, indices, B, S)
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result[0, 20:, 0].numpy(), np.zeros(12))
        np.testing.assert_array_equal(result[1, :, 0].numpy(), np.ones(S))

    def test_attention_mask_takes_priority(self):
        attn_mask = paddle.ones([B, S], dtype="int64")
        attn_mask[0, -4:] = 0
        indices = _make_startend_indices(B, S, padding_start=None)  # all valid
        result = self.gdn._build_padding_mask(attn_mask, indices, B, S)
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result[0, -4:, 0].numpy(), np.zeros(4))

    def test_boundary_indices_eq_pos_is_padding(self):
        indices = paddle.full([1, 1, S, 1], fill_value=S, dtype="int64")
        indices[0, 0, 0, 0] = 1  # 1 > 0 → valid
        indices[0, 0, 1, 0] = 1  # 1 > 1 → False → padding
        result = self.gdn._build_padding_mask(None, indices, 1, S)
        self.assertEqual(result[0, 0, 0].item(), 1.0)
        self.assertEqual(result[0, 1, 0].item(), 0.0)


class TestBuildPaddingMaskWithSP(unittest.TestCase):
    """_build_padding_mask with simulated SP=True, TP=2."""

    def setUp(self):
        self.gdn = _make_gdn_sp()

    def test_attention_mask_rank0_all_valid(self):
        mask = paddle.ones([B, S], dtype="int64")
        with patch(f"{_GDN_MODULE}.get_pg_rank", return_value=0):
            self.assertIsNone(self.gdn._build_padding_mask(mask, None, B, S))

    def test_attention_mask_rank1_with_padding(self):
        mask = paddle.ones([B, S], dtype="int64")
        mask[0, -4:] = 0  # positions 28-31 padding
        with patch(f"{_GDN_MODULE}.get_pg_rank", return_value=1):
            result = self.gdn._build_padding_mask(mask, None, B, S)
        # SP layout: [s_local=16, b=2, 1]
        self.assertEqual(list(result.shape), [16, B, 1])
        # Rank1 sees [16,32), positions 28-31 → local 12-15
        np.testing.assert_array_equal(result[:12, 0, 0].numpy(), np.ones(12))
        np.testing.assert_array_equal(result[12:, 0, 0].numpy(), np.zeros(4))
        np.testing.assert_array_equal(result[:, 1, 0].numpy(), np.ones(16))

    def test_startend_rank0_all_valid(self):
        indices = _make_startend_indices(B, S, padding_start=None)
        with patch(f"{_GDN_MODULE}.get_pg_rank", return_value=0):
            self.assertIsNone(self.gdn._build_padding_mask(None, indices, B, S))

    def test_startend_rank1_with_padding(self):
        indices = _make_startend_indices(B, S, padding_start=24)
        with patch(f"{_GDN_MODULE}.get_pg_rank", return_value=1):
            result = self.gdn._build_padding_mask(None, indices, B, S)
        # Rank1 [16,32): 16-23 valid, 24-31 padding → local 0-7 valid, 8-15 padding
        self.assertEqual(list(result.shape), [16, B, 1])
        np.testing.assert_array_equal(result[:8, 0, 0].numpy(), np.ones(8))
        np.testing.assert_array_equal(result[8:, 0, 0].numpy(), np.zeros(8))

    def test_startend_rank0_no_padding_rank1_has_padding(self):
        indices = _make_startend_indices(B, S, padding_start=20)
        with patch(f"{_GDN_MODULE}.get_pg_rank", return_value=0):
            self.assertIsNone(self.gdn._build_padding_mask(None, indices, B, S))
        with patch(f"{_GDN_MODULE}.get_pg_rank", return_value=1):
            result = self.gdn._build_padding_mask(None, indices, B, S)
        self.assertIsNotNone(result)
        # Rank1 [16,32): 16-19 valid, 20-31 padding → local 0-3 valid, 4-15 padding
        np.testing.assert_array_equal(result[:4, 0, 0].numpy(), np.ones(4))
        np.testing.assert_array_equal(result[4:, 0, 0].numpy(), np.zeros(12))

    def test_attention_mask_shape_mismatch_raises(self):
        """SP path: attention_mask full_seq != seq_len should raise ValueError."""
        wrong_len_mask = paddle.ones([B, S + 4], dtype="int64")
        with (
            patch(f"{_GDN_MODULE}.get_pg_rank", return_value=0),
            self.assertRaises(ValueError),
        ):
            self.gdn._build_padding_mask(wrong_len_mask, None, B, S)

    def test_startend_shape_mismatch_raises(self):
        """SP path: startend full_seq != seq_len should raise ValueError."""
        wrong_indices = _make_startend_indices(B, S + 4, padding_start=None)
        with (
            patch(f"{_GDN_MODULE}.get_pg_rank", return_value=0),
            self.assertRaises(ValueError),
        ):
            self.gdn._build_padding_mask(None, wrong_indices, B, S)


class TestForwardPaddingMask(unittest.TestCase):
    """Forward-level tests for mask application."""

    def setUp(self):
        self.gdn = _make_gdn()
        self.gdn.eval()

    def test_all_valid_mask_equals_no_mask(self):
        x = paddle.randn([B, S, H])
        out_none, _ = self.gdn(x, attention_mask=None)
        out_valid, _ = self.gdn(
            x, attention_mask=paddle.ones([B, S], dtype="int64")
        )
        np.testing.assert_allclose(
            out_none.numpy(), out_valid.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_padding_changes_output(self):
        x = paddle.randn([B, S, H])
        mask = paddle.ones([B, S], dtype="int64")
        mask[0, :] = 0
        out_masked, _ = self.gdn(x, attention_mask=mask)
        out_none, _ = self.gdn(x, attention_mask=None)
        # Sample 0 differs, sample 1 same
        self.assertFalse(
            np.allclose(out_masked[0].numpy(), out_none[0].numpy(), atol=1e-6)
        )
        np.testing.assert_allclose(
            out_masked[1].numpy(), out_none[1].numpy(), rtol=1e-5, atol=1e-5
        )

    def test_startend_indices_with_padding(self):
        x = paddle.randn([B, S, H])
        indices = _make_startend_indices(B, S, padding_start=24)
        out_pad, _ = self.gdn(
            x, attention_mask=None, attn_mask_startend_row_indices=indices
        )
        out_none, _ = self.gdn(x, attention_mask=None)
        self.assertFalse(
            np.allclose(out_pad.numpy(), out_none.numpy(), atol=1e-6)
        )

    def test_tokens_before_padding_unaffected(self):
        x = paddle.randn([1, S, H])
        mask = paddle.ones([1, S], dtype="int64")
        mask[0, 28:] = 0
        out_pad, _ = self.gdn(x, attention_mask=mask)
        out_none, _ = self.gdn(x, attention_mask=None)
        # Causal: tokens far before padding are unaffected
        np.testing.assert_allclose(
            out_none[0, :24].numpy(),
            out_pad[0, :24].numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_backward_gradients(self):
        x = paddle.randn([B, S, H])
        x.stop_gradient = False
        mask = paddle.ones([B, S], dtype="int64")
        mask[0, -8:] = 0
        out, _ = self.gdn(x, attention_mask=mask)
        out.sum().backward()
        self.assertTrue(paddle.isfinite(x.grad).all().item())
        # Padding positions get zero gradient
        np.testing.assert_array_equal(x.grad[0, -8:].numpy(), np.zeros([8, H]))
        # Valid positions get non-zero gradient
        self.assertFalse(np.allclose(x.grad[0, :24].numpy(), 0))


class TestBuildPaddingMask4D(unittest.TestCase):
    """4D ``[b, 1, s, s]`` masks, which the multimodal collator emits.

    GDN is a recurrence and only needs per-token validity, so the block-causal
    matrix is reduced to its diagonal. Before that reduction existed a 4D mask
    fell through to ``attn_mask_startend_row_indices`` and, when the VL collator
    supplied no indices, raised outright.
    """

    def setUp(self):
        self.gdn = _make_gdn()

    def test_4d_all_valid_returns_none(self):
        mask = _to_4d_causal(_keep_vector(B, S))
        self.assertIsNone(self.gdn._build_padding_mask(mask, None, B, S))

    def test_4d_with_padding_returns_mask(self):
        mask = _to_4d_causal(_keep_vector(B, S, padding_start=24, sample=0))
        result = self.gdn._build_padding_mask(mask, None, B, S)
        self.assertEqual(list(result.shape), [B, S, 1])
        np.testing.assert_array_equal(result[0, :24, 0].numpy(), np.ones(24))
        np.testing.assert_array_equal(result[0, 24:, 0].numpy(), np.zeros(8))
        np.testing.assert_array_equal(result[1, :, 0].numpy(), np.ones(S))

    def test_4d_matches_equivalent_2d(self):
        keep = _keep_vector(B, S, padding_start=20, sample=0)
        from_2d = self.gdn._build_padding_mask(keep, None, B, S)
        from_4d = self.gdn._build_padding_mask(_to_4d_causal(keep), None, B, S)
        np.testing.assert_array_equal(from_4d.numpy(), from_2d.numpy())

    def test_4d_bool_dtype(self):
        keep = _keep_vector(B, S, padding_start=28, sample=1)
        mask = _to_4d_causal(keep, dtype="bool")
        result = self.gdn._build_padding_mask(mask, None, B, S)
        np.testing.assert_array_equal(result[1, 28:, 0].numpy(), np.zeros(4))
        np.testing.assert_array_equal(result[0, :, 0].numpy(), np.ones(S))

    def test_4d_packed_block_diagonal_is_all_valid(self):
        """Document isolation is not padding: every token's diagonal is set.

        GDN takes document boundaries from ``attn_mask_startend_row_indices``,
        so a packed block-diagonal mask must not be mistaken for padding.
        """
        mask = _to_4d_block_diagonal(S, [12, 20])
        self.assertIsNone(self.gdn._build_padding_mask(mask, None, 1, S))

    def test_4d_shape_mismatch_falls_through_to_startend(self):
        wrong = _to_4d_causal(_keep_vector(B, S + 4))
        indices = _make_startend_indices(B, S, padding_start=24)
        result = self.gdn._build_padding_mask(wrong, indices, B, S)
        self.assertEqual(list(result.shape), [B, S, 1])
        np.testing.assert_array_equal(result[0, 24:, 0].numpy(), np.zeros(8))

    def test_4d_shape_mismatch_without_startend_raises(self):
        wrong = _to_4d_causal(_keep_vector(B, S + 4))
        with self.assertRaises(ValueError):
            self.gdn._build_padding_mask(wrong, None, B, S)

    def test_4d_non_square_falls_through(self):
        """A ``[b, 1, s, s_kv]`` cross-attention-shaped mask is not reducible."""
        mask = paddle.ones([B, 1, S, S + 8], dtype="float32")
        with self.assertRaises(ValueError):
            self.gdn._build_padding_mask(mask, None, B, S)

    def test_4d_takes_priority_over_startend(self):
        mask = _to_4d_causal(_keep_vector(B, S, padding_start=28, sample=0))
        indices = _make_startend_indices(B, S, padding_start=None)
        result = self.gdn._build_padding_mask(mask, indices, B, S)
        np.testing.assert_array_equal(result[0, 28:, 0].numpy(), np.zeros(4))

    def test_all_valid_check_survives_strided_input(self):
        """Pin the reduction: ``all()`` over a strided float buffer lies.

        ``paddle.diagonal`` returns a non-contiguous view, and
        ``paddle.Tensor.all()`` on such a float tensor reports ``False`` even
        when every element is 1.0 — which would return an all-ones mask instead
        of ``None``. Assert the raw primitive still misbehaves (so this test
        starts failing once Paddle fixes it and the guard can be dropped) and
        that ``_build_padding_mask`` is nonetheless correct.
        """
        ones_4d = _to_4d_causal(_keep_vector(1, S))
        strided = paddle.diagonal(
            ones_4d[:, 0, :, :], axis1=-2, axis2=-1
        ).unsqueeze(-1)
        self.assertFalse(strided.is_contiguous())
        np.testing.assert_array_equal(strided.numpy().ravel(), np.ones(S))
        self.assertFalse(bool(strided.all()))
        self.assertTrue(bool(strided.astype("bool").all()))
        self.assertIsNone(self.gdn._build_padding_mask(ones_4d, None, 1, S))

    def test_4d_returns_contiguous_mask(self):
        mask = _to_4d_causal(_keep_vector(B, S, padding_start=24, sample=0))
        result = self.gdn._build_padding_mask(mask, None, B, S)
        self.assertTrue(result.is_contiguous())


class TestBuildPaddingMask4DWithSP(unittest.TestCase):
    """4D masks reuse the SP slicing path after being reduced to 2D."""

    def setUp(self):
        self.gdn = _make_gdn_sp()

    def test_4d_rank0_all_valid(self):
        mask = _to_4d_causal(_keep_vector(B, S))
        with patch(f"{_GDN_MODULE}.get_pg_rank", return_value=0):
            self.assertIsNone(self.gdn._build_padding_mask(mask, None, B, S))

    def test_4d_rank1_with_padding(self):
        mask = _to_4d_causal(_keep_vector(B, S, padding_start=28, sample=0))
        with patch(f"{_GDN_MODULE}.get_pg_rank", return_value=1):
            result = self.gdn._build_padding_mask(mask, None, B, S)
        # SP layout is [s_local, b, 1]; rank 1 owns [16, 32) so 28-31 → 12-15.
        self.assertEqual(list(result.shape), [16, B, 1])
        np.testing.assert_array_equal(result[:12, 0, 0].numpy(), np.ones(12))
        np.testing.assert_array_equal(result[12:, 0, 0].numpy(), np.zeros(4))
        np.testing.assert_array_equal(result[:, 1, 0].numpy(), np.ones(16))

    def test_4d_rank0_clean_rank1_padded(self):
        mask = _to_4d_causal(_keep_vector(B, S, padding_start=20))
        with patch(f"{_GDN_MODULE}.get_pg_rank", return_value=0):
            self.assertIsNone(self.gdn._build_padding_mask(mask, None, B, S))
        with patch(f"{_GDN_MODULE}.get_pg_rank", return_value=1):
            result = self.gdn._build_padding_mask(mask, None, B, S)
        np.testing.assert_array_equal(result[:4, 0, 0].numpy(), np.ones(4))
        np.testing.assert_array_equal(result[4:, 0, 0].numpy(), np.zeros(12))

    def test_4d_shape_mismatch_raises_under_sp(self):
        wrong = _to_4d_causal(_keep_vector(B, S + 4))
        with (
            patch(f"{_GDN_MODULE}.get_pg_rank", return_value=0),
            self.assertRaises(ValueError),
        ):
            self.gdn._build_padding_mask(wrong, None, B, S)


class TestForward4DMask(unittest.TestCase):
    """Forward-level checks that a VL-style 4D mask is honored, not rejected."""

    def setUp(self):
        self.gdn = _make_gdn()
        self.gdn.eval()

    def test_forward_accepts_4d_mask(self):
        x = paddle.randn([B, S, H])
        keep = _keep_vector(B, S, padding_start=24, sample=0)
        out_4d, _ = self.gdn(x, attention_mask=_to_4d_causal(keep))
        out_2d, _ = self.gdn(x, attention_mask=keep)
        np.testing.assert_allclose(
            out_4d.numpy(), out_2d.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_forward_4d_all_valid_equals_no_mask(self):
        x = paddle.randn([B, S, H])
        out_4d, _ = self.gdn(
            x, attention_mask=_to_4d_causal(_keep_vector(B, S))
        )
        out_none, _ = self.gdn(x, attention_mask=None)
        np.testing.assert_allclose(
            out_4d.numpy(), out_none.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_forward_4d_padding_changes_output(self):
        x = paddle.randn([B, S, H])
        keep = _keep_vector(B, S, padding_start=0, sample=0)
        out_pad, _ = self.gdn(x, attention_mask=_to_4d_causal(keep))
        out_none, _ = self.gdn(x, attention_mask=None)
        self.assertFalse(
            np.allclose(out_pad[0].numpy(), out_none[0].numpy(), atol=1e-6)
        )
        np.testing.assert_allclose(
            out_pad[1].numpy(), out_none[1].numpy(), rtol=1e-5, atol=1e-5
        )

    def test_forward_4d_backward_zeroes_padding_grad(self):
        x = paddle.randn([B, S, H])
        x.stop_gradient = False
        keep = _keep_vector(B, S, padding_start=24, sample=0)
        out, _ = self.gdn(x, attention_mask=_to_4d_causal(keep))
        out.sum().backward()
        self.assertTrue(paddle.isfinite(x.grad).all().item())
        np.testing.assert_array_equal(x.grad[0, 24:].numpy(), np.zeros([8, H]))
        self.assertFalse(np.allclose(x.grad[0, :24].numpy(), 0))


if __name__ == "__main__":
    unittest.main()
