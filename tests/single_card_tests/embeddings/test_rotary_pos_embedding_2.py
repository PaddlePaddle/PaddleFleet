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
"""Behavior tests for the rotary position embeddings in
``paddlefleet.models.common.embeddings.rotary_pos_embedding``.

Every expected value here is derived independently (plain ``math`` / ``numpy``)
from the documented construction of RoPE -- the function under test is never
used to compute its own reference:

  * ``RotaryEmbedding.inv_freq`` is compared against ``base ** (-2i/dim)``.
  * ``rotary_percent < 1`` shrinks the rotary dimension.
  * ``get_freqs_non_repeated`` is the outer product ``(pos + offset) * inv_freq``.
  * ``get_cos_sin`` is elementwise cos/sin of those frequencies.
  * ``forward`` duplicates the half-frequency table -- concatenated for the
    default layout, pair-interleaved for ``rotary_interleaved`` -- and reshapes
    to ``[1, seq, 1, dim]``. Both layouts are checked position-by-position.
  * ``get_rotary_seq_len`` selects the sequence axis, applies the
    tensor-parallel multiplier under sequence-parallel, and honours packed
    sequence params.
  * ``MultimodalRotaryEmbedding.apply_interleaved_mrope`` re-interleaves the
    T/H/W frequency chunks; the source chunk landing in each channel is traced
    with distinguishable per-section markers.

The whole ``paddlefleet`` package imports ``paddle`` at import time. When paddle
is not installed the entire module is skipped with an honest reason rather than
faking a pass. Only a genuine missing-dependency ``ImportError`` triggers the
skip; API/compile errors are allowed to surface.
"""

import unittest

try:
    import numpy as np
    import paddle

    from paddlefleet.models.common.embeddings.rotary_pos_embedding import (
        MultimodalRotaryEmbedding,
        RotaryEmbedding,
    )

    PADDLE_AVAILABLE = True
    IMPORT_ERROR = ""
except ImportError as exc:  # real missing dep only, not API/compile errors
    PADDLE_AVAILABLE = False
    IMPORT_ERROR = repr(exc)

SKIP_REASON = (
    f"paddle / paddlefleet not importable in this environment: {IMPORT_ERROR}"
)


def ref_inv_freq(dim, rotary_base=10000.0):
    """Independent reference: inv_freq[i] = base ** (-(2i)/dim)."""
    return [1.0 / (rotary_base ** ((2 * i) / dim)) for i in range(dim // 2)]


class _Config:
    """Minimal stand-in for TransformerConfig fields read by
    ``get_rotary_seq_len`` -- real attributes, no truthy-magic mocks."""

    def __init__(self, sequence_parallel, tensor_model_parallel_size=1):
        self.sequence_parallel = sequence_parallel
        self.tensor_model_parallel_size = tensor_model_parallel_size


class _PackedSeqParams:
    def __init__(self, max_seqlen_q, max_seqlen_kv):
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_kv = max_seqlen_kv


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestRotaryEmbeddingInvFreq(unittest.TestCase):
    def test_inv_freq_values_match_base_power_law(self):
        head_dim = 64
        emb = RotaryEmbedding(head_dim=head_dim, rotary_percent=1.0)
        expected = ref_inv_freq(head_dim, rotary_base=10000.0)
        actual = emb.inv_freq.numpy()
        self.assertEqual(actual.shape, (head_dim // 2,))
        # inv_freq[0] must be exactly 1.0; it decays monotonically after that.
        self.assertAlmostEqual(float(actual[0]), 1.0, places=6)
        np.testing.assert_allclose(actual, np.array(expected), rtol=1e-6)
        self.assertTrue(np.all(np.diff(actual) < 0))

    def test_custom_base_changes_inv_freq(self):
        head_dim = 32
        emb = RotaryEmbedding(
            head_dim=head_dim, rotary_percent=1.0, rotary_base=5000
        )
        expected = ref_inv_freq(head_dim, rotary_base=5000.0)
        np.testing.assert_allclose(
            emb.inv_freq.numpy(), np.array(expected), rtol=1e-6
        )

    def test_rotary_percent_shrinks_dimension(self):
        # rotary_percent halves the rotary dim -> half as many frequencies.
        full = RotaryEmbedding(head_dim=64, rotary_percent=1.0)
        half = RotaryEmbedding(head_dim=64, rotary_percent=0.5)
        self.assertEqual(full.inv_freq.shape[0], 32)
        self.assertEqual(half.inv_freq.shape[0], 16)
        # The reduced table uses dim=32, not a truncation of the dim=64 table.
        np.testing.assert_allclose(
            half.inv_freq.numpy(), np.array(ref_inv_freq(32)), rtol=1e-6
        )


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestRotaryEmbeddingFreqs(unittest.TestCase):
    def test_freqs_are_outer_product_with_offset(self):
        head_dim = 64
        offset = 2
        max_seq_len = 4
        emb = RotaryEmbedding(head_dim=head_dim, rotary_percent=1.0)
        inv = ref_inv_freq(head_dim)
        # positions are [offset, offset+1, ..., offset+max_seq_len-1]
        positions = [offset + p for p in range(max_seq_len)]
        expected = np.array(
            [[pos * inv[i] for i in range(head_dim // 2)] for pos in positions]
        )
        got = emb.get_freqs_non_repeated(max_seq_len, offset=offset)
        self.assertEqual(list(got.shape), [max_seq_len, head_dim // 2])
        np.testing.assert_allclose(got.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_interpolation_factor_scales_positions(self):
        head_dim = 64
        factor = 2.0
        max_seq_len = 4
        emb = RotaryEmbedding(
            head_dim=head_dim,
            rotary_percent=1.0,
            seq_len_interpolation_factor=factor,
        )
        inv = ref_inv_freq(head_dim)
        # positions divided by the interpolation factor before the outer product.
        expected = np.array(
            [
                [(p / factor) * inv[i] for i in range(head_dim // 2)]
                for p in range(max_seq_len)
            ]
        )
        got = emb.get_freqs_non_repeated(max_seq_len)
        np.testing.assert_allclose(got.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_get_cos_sin_matches_cos_sin_of_freqs(self):
        head_dim = 64
        max_seq_len = 5
        emb = RotaryEmbedding(head_dim=head_dim, rotary_percent=1.0)
        inv = ref_inv_freq(head_dim)
        freqs = np.array(
            [
                [p * inv[i] for i in range(head_dim // 2)]
                for p in range(max_seq_len)
            ]
        )
        cos, sin = emb.get_cos_sin(max_seq_len)
        np.testing.assert_allclose(
            cos.numpy(), np.cos(freqs), rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            sin.numpy(), np.sin(freqs), rtol=1e-5, atol=1e-6
        )


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestRotaryEmbeddingForward(unittest.TestCase):
    def test_forward_default_layout_concatenates_halves(self):
        head_dim = 64
        max_seq_len = 3
        emb = RotaryEmbedding(head_dim=head_dim, rotary_percent=1.0)
        inv = ref_inv_freq(head_dim)
        half = head_dim // 2
        freqs = np.array(
            [[p * inv[i] for i in range(half)] for p in range(max_seq_len)]
        )
        out = emb(max_seq_len)
        # Shape is [1, seq, 1, dim] with dim == 2 * half.
        self.assertEqual(list(out.shape), [1, max_seq_len, 1, head_dim])
        block = out.numpy()[0, :, 0, :]
        # Default (non-interleaved): [f0..f_{h-1}, f0..f_{h-1}]
        np.testing.assert_allclose(block[:, :half], freqs, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(block[:, half:], freqs, rtol=1e-5, atol=1e-6)

    def test_forward_interleaved_layout_duplicates_pairs(self):
        head_dim = 64
        max_seq_len = 3
        emb = RotaryEmbedding(
            head_dim=head_dim, rotary_percent=1.0, rotary_interleaved=True
        )
        inv = ref_inv_freq(head_dim)
        half = head_dim // 2
        freqs = np.array(
            [[p * inv[i] for i in range(half)] for p in range(max_seq_len)]
        )
        out = emb(max_seq_len)
        self.assertEqual(list(out.shape), [1, max_seq_len, 1, head_dim])
        block = out.numpy()[0, :, 0, :]
        # Interleaved: [f0, f0, f1, f1, ...] -> even/odd channels equal freqs.
        np.testing.assert_allclose(block[:, 0::2], freqs, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(block[:, 1::2], freqs, rtol=1e-5, atol=1e-6)
        # The two layouts genuinely differ: concat != pair-interleave here.
        self.assertFalse(np.allclose(block[:, :half], block[:, 0::2]))


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestGetRotarySeqLen(unittest.TestCase):
    def test_seq_axis_without_sequence_parallel(self):
        emb = RotaryEmbedding(head_dim=64, rotary_percent=1.0)
        # Not sequence-parallel -> sequence axis is 1.
        x = paddle.zeros([2, 37, 64])
        self.assertEqual(
            emb.get_rotary_seq_len(x, _Config(sequence_parallel=False)), 37
        )

    def test_sequence_parallel_multiplies_by_tp_size(self):
        emb = RotaryEmbedding(head_dim=64, rotary_percent=1.0)
        # Sequence-parallel -> axis 0, then scaled by tensor_model_parallel_size.
        x = paddle.zeros([11, 2, 64])
        cfg = _Config(sequence_parallel=True, tensor_model_parallel_size=4)
        self.assertEqual(emb.get_rotary_seq_len(x, cfg), 44)

    def test_packed_seq_params_take_precedence(self):
        emb = RotaryEmbedding(head_dim=64, rotary_percent=1.0)
        x = paddle.zeros([2, 8, 64])
        packed = _PackedSeqParams(max_seqlen_q=100, max_seqlen_kv=250)
        # max of q/kv, independent of the tensor's own shape.
        self.assertEqual(
            emb.get_rotary_seq_len(
                x, _Config(sequence_parallel=False), packed_seq_params=packed
            ),
            250,
        )


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestApplyInterleavedMrope(unittest.TestCase):
    def test_interleaving_selects_correct_source_chunk(self):
        emb = MultimodalRotaryEmbedding(head_dim=8, rotary_percent=1.0)
        # freqs: [3(T/H/W), bs=1, seq=1, dim//2=4].
        # Encode source id as 1000 * section + channel so the origin of each
        # output channel is unambiguous.
        d = 4
        base = np.array(
            [[[[1000 * sec + c for c in range(d)]]] for sec in range(3)],
            dtype="float32",
        )
        freqs = paddle.to_tensor(base)
        # mrope_section = [1, 1, 1]:
        #   start from T (section 0) everywhere;
        #   H (section 1) overwrites channel 1 (offset=1, step=3, <3);
        #   W (section 2) overwrites channel 2 (offset=2, step=3, <3);
        #   channel 3 stays T.
        out = emb.apply_interleaved_mrope(freqs, [1, 1, 1])
        expected = np.array([[[0.0, 1001.0, 2002.0, 3.0]]], dtype="float32")
        self.assertEqual(list(out.shape), [1, 1, d])
        np.testing.assert_array_equal(out.numpy(), expected)


if __name__ == "__main__":
    unittest.main()
