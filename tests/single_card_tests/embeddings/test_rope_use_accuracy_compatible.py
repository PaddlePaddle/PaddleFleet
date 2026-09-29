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

"""CPU-observable behavior tests for RotaryEmbedding.

The whole paddlefleet package imports paddle at import time. If paddle (or the
package) is not importable in this environment the tests are honestly SKIPPED;
they are NOT reported as passing.

Design notes:
- Expected values are re-derived independently in NumPy from the standard RoPE
  definition (``inv_freq[i] = base ** (-(2 i) / dim)`` plus an outer product with
  the positions). The class under test is never called to compute its own
  expected output.
- Fixed, distinguishable positions/dims are used so that a transposed table, a
  wrong duplication order, or a dropped offset would be observable in content.
- The ONLY non-被测 collaborator replaced is ``parallel_state`` (which needs a
  real context-parallel group); ``inv_freq`` and ``forward`` run for real.
"""

import unittest
from unittest.mock import patch

import numpy as np

_PS_PATH = (
    "paddlefleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
)

try:
    import paddle  # noqa: F401

    from paddlefleet.models.common.embeddings.rotary_pos_embedding import (
        RotaryEmbedding,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle not installed in this environment
    RotaryEmbedding = None
    _IMPORT_ERROR = exc


def _reference_inv_freq(head_dim, rotary_percent, rotary_base=10000):
    """Independent NumPy re-derivation of the RoPE inverse-frequency vector.

    Standard RoPE definition: for the even channel indices 0, 2, ..., dim-2,
    ``inv_freq[i] = base ** (-(2 i) / dim)``. Computed with plain NumPy; the
    class under test is NOT invoked here.
    """
    dim = head_dim
    if rotary_percent < 1.0:
        dim = int(dim * rotary_percent)
    exponents = np.arange(0, dim, 2, dtype=np.float64) / dim
    return 1.0 / (rotary_base**exponents)


@unittest.skipUnless(
    RotaryEmbedding is not None,
    f"paddle/paddlefleet not importable in this environment: {_IMPORT_ERROR}",
)
class TestRotaryEmbeddingInvFreq(unittest.TestCase):
    """inv_freq must equal the hand-derived RoPE frequencies, per rotary dim."""

    @patch(_PS_PATH)
    def test_full_rotary_inv_freq_matches_reference(self, mock_ps):
        mock_ps.get_context_parallel_group.return_value = None
        rope = RotaryEmbedding(head_dim=64, rotary_percent=1.0)

        ref = _reference_inv_freq(64, 1.0)
        self.assertEqual(list(rope.inv_freq.shape), [32])
        np.testing.assert_allclose(
            rope.inv_freq.astype("float32").numpy(), ref, rtol=1e-5, atol=1e-7
        )
        # First channel is base**0 == 1.0 and the sequence strictly decreases:
        # a wrong sign in the exponent (base**(+2i/dim)) would flip this.
        self.assertAlmostEqual(float(rope.inv_freq.numpy()[0]), 1.0, places=6)
        diffs = np.diff(rope.inv_freq.astype("float32").numpy())
        self.assertTrue((diffs < 0).all())

    @patch(_PS_PATH)
    def test_partial_rotary_inv_freq_matches_reference(self, mock_ps):
        mock_ps.get_context_parallel_group.return_value = None
        rope = RotaryEmbedding(head_dim=64, rotary_percent=0.5)

        # rotary_percent<1.0 => dim = int(64*0.5)=32 => 16 even channels.
        ref = _reference_inv_freq(64, 0.5)
        self.assertEqual(list(rope.inv_freq.shape), [16])
        np.testing.assert_allclose(
            rope.inv_freq.astype("float32").numpy(), ref, rtol=1e-5, atol=1e-7
        )

    @patch(_PS_PATH)
    def test_custom_rotary_base_changes_frequencies(self, mock_ps):
        mock_ps.get_context_parallel_group.return_value = None
        rope = RotaryEmbedding(
            head_dim=64, rotary_percent=1.0, rotary_base=500000
        )

        ref = _reference_inv_freq(64, 1.0, rotary_base=500000)
        np.testing.assert_allclose(
            rope.inv_freq.astype("float32").numpy(), ref, rtol=1e-5, atol=1e-9
        )
        # A larger base must produce strictly smaller (slower) frequencies for
        # every channel except the first (which is always 1.0).
        default_ref = _reference_inv_freq(64, 1.0, rotary_base=10000)
        got = rope.inv_freq.astype("float32").numpy()
        self.assertTrue((got[1:] < default_ref[1:]).all())

    @patch(_PS_PATH)
    def test_use_accuracy_compatible_does_not_alter_inv_freq(self, mock_ps):
        """RotaryEmbedding accepts use_accuracy_compatible for API parity but
        (unlike MultimodalRotaryEmbedding) does not branch on it: inv_freq is
        computed unconditionally and it opts out of low-precision casting
        unconditionally. Both instances are anchored to an INDEPENDENT
        reference (not to each other), so this is not a self-comparison."""
        mock_ps.get_context_parallel_group.return_value = None
        ref = _reference_inv_freq(64, 1.0)

        rope_off = RotaryEmbedding(
            head_dim=64, rotary_percent=1.0, use_accuracy_compatible=False
        )
        rope_on = RotaryEmbedding(
            head_dim=64, rotary_percent=1.0, use_accuracy_compatible=True
        )

        np.testing.assert_allclose(
            rope_off.inv_freq.astype("float32").numpy(),
            ref,
            rtol=1e-5,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            rope_on.inv_freq.astype("float32").numpy(),
            ref,
            rtol=1e-5,
            atol=1e-7,
        )
        self.assertFalse(rope_off._cast_to_low_precision)
        self.assertFalse(rope_on._cast_to_low_precision)


@unittest.skipUnless(
    RotaryEmbedding is not None,
    f"paddle/paddlefleet not importable in this environment: {_IMPORT_ERROR}",
)
class TestRotaryEmbeddingForward(unittest.TestCase):
    """forward() must build the position-by-frequency table with the correct
    layout, duplication scheme, and offset handling."""

    @patch(_PS_PATH)
    def test_forward_builds_duplicated_outer_product(self, mock_ps):
        mock_ps.get_context_parallel_group.return_value = None
        rope = RotaryEmbedding(head_dim=64, rotary_percent=1.0)

        emb = rope.forward(max_seq_len=4, offset=0)
        self.assertEqual(list(emb.shape), [1, 4, 1, 64])

        inv = _reference_inv_freq(64, 1.0)
        positions = np.arange(4, dtype=np.float64)
        freqs = np.outer(positions, inv)  # [4, 32]
        expected = np.concatenate([freqs, freqs], axis=-1)[None, :, None, :]
        got = emb.astype("float32").numpy()
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

        # Non-interleaved layout duplicates the whole block: [:32] == [32:].
        np.testing.assert_array_equal(got[0, :, 0, :32], got[0, :, 0, 32:])

    @patch(_PS_PATH)
    def test_forward_offset_shifts_positions(self, mock_ps):
        mock_ps.get_context_parallel_group.return_value = None
        rope = RotaryEmbedding(head_dim=64, rotary_percent=1.0)

        emb = rope.forward(max_seq_len=3, offset=5)
        inv = _reference_inv_freq(64, 1.0)
        positions = np.arange(3, dtype=np.float64) + 5  # [5, 6, 7]
        freqs = np.outer(positions, inv)
        expected = np.concatenate([freqs, freqs], axis=-1)[None, :, None, :]
        np.testing.assert_allclose(
            emb.astype("float32").numpy(), expected, rtol=1e-5, atol=1e-6
        )
        # Row 0 corresponds to position 5, not 0: it must be non-zero.
        self.assertGreater(
            float(np.abs(emb.astype("float32").numpy()[0, 0]).max()), 0.0
        )

    @patch(_PS_PATH)
    def test_forward_interleaved_layout_differs(self, mock_ps):
        mock_ps.get_context_parallel_group.return_value = None
        rope_int = RotaryEmbedding(
            head_dim=64, rotary_percent=1.0, rotary_interleaved=True
        )
        rope_seq = RotaryEmbedding(
            head_dim=64, rotary_percent=1.0, rotary_interleaved=False
        )

        emb_int = rope_int.forward(max_seq_len=2, offset=0)
        emb_seq = rope_seq.forward(max_seq_len=2, offset=0)

        inv = _reference_inv_freq(64, 1.0)
        positions = np.arange(2, dtype=np.float64)
        freqs = np.outer(positions, inv)  # [2, 32]
        # Interleaved layout pairs each frequency with itself: f0,f0,f1,f1,...
        expected_int = np.repeat(freqs, 2, axis=-1)[None, :, None, :]
        np.testing.assert_allclose(
            emb_int.astype("float32").numpy(),
            expected_int,
            rtol=1e-5,
            atol=1e-6,
        )
        # The two layouts must genuinely differ (position 1 row distinguishes
        # [f,f,...] pairing from [block,block] duplication).
        self.assertFalse(
            np.allclose(
                emb_int.astype("float32").numpy(),
                emb_seq.astype("float32").numpy(),
            )
        )

    @patch(_PS_PATH)
    def test_get_cos_sin_matches_reference(self, mock_ps):
        mock_ps.get_context_parallel_group.return_value = None
        rope = RotaryEmbedding(head_dim=64, rotary_percent=1.0)

        cos, sin = rope.get_cos_sin(max_seq_len=4, offset=0)
        inv = _reference_inv_freq(64, 1.0)
        positions = np.arange(4, dtype=np.float64)
        freqs = np.outer(positions, inv)  # [4, 32], no duplication here
        np.testing.assert_allclose(
            cos.astype("float32").numpy(), np.cos(freqs), rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            sin.astype("float32").numpy(), np.sin(freqs), rtol=1e-5, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
