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
"""CPU-observable behavior tests for
``paddlefleet.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding``.

Design notes
------------
All expected values are hand-derived from the documented RoPE construction and
never produced by calling the function under test.  The anchor is a small head
dimension whose inverse frequencies are exact decimal powers of ten, so every
numeric expectation can be written as a literal:

    head_dim = 8, rotary_base = 10000 = 10**4
    dim indices used = arange(0, 8, 2) = [0, 2, 4, 6]
    inv_freq[i] = 10000 ** (-(2*i)/8) = 10 ** (-(2*i)/2)
                = [10**0, 10**-1, 10**-2, 10**-3]
                = [1.0, 0.1, 0.01, 0.001]

From that anchor the per-position frequency table is the outer product
``(position + offset) * inv_freq``; ``forward`` duplicates that half-table
(concatenated by default, pair-interleaved when ``rotary_interleaved=True``) and
reshapes to ``[1, seq, 1, dim]``; ``get_cos_sin`` is the elementwise cos/sin;
``get_rotary_seq_len`` selects the sequence axis and applies the TP / CP
multipliers.  Each is checked against an independent numpy/math reference.

``paddlefleet`` imports ``paddle`` at import time.  When paddle is absent the
whole module is skipped with an honest reason -- only a genuine ``ImportError``
triggers the skip so that API/compile regressions still surface.
"""

import unittest
from types import SimpleNamespace

try:
    import numpy as np
    import paddle

    from paddlefleet.models.common.embeddings.rotary_pos_embedding import (
        RotaryEmbedding,
    )

    PADDLE_AVAILABLE = True
    IMPORT_ERROR = ""
except ImportError as exc:  # genuine missing dependency only
    PADDLE_AVAILABLE = False
    IMPORT_ERROR = repr(exc)

SKIP_REASON = (
    "paddle / paddlefleet not importable in this environment: " + IMPORT_ERROR
)

# Hand-derived inverse frequencies for head_dim=8, rotary_base=10000.
# Independent of the module: 10000 ** (-(2*i)/8) for i in {0,1,2,3}.
INV_FREQ_HD8 = [1.0, 0.1, 0.01, 0.001]


def _ref_freqs(positions, inv_freq):
    """Independent outer product: freqs[s, k] = positions[s] * inv_freq[k]."""
    return np.array(
        [[p * f for f in inv_freq] for p in positions], dtype=np.float64
    )


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestInvFreq(unittest.TestCase):
    def test_full_rotary_inv_freq_is_powers_of_ten(self):
        emb = RotaryEmbedding(head_dim=8, rotary_percent=1.0)
        actual = emb.inv_freq.numpy().astype(np.float64)
        self.assertEqual(actual.shape, (4,))
        np.testing.assert_allclose(
            actual, np.array(INV_FREQ_HD8), rtol=0, atol=1e-9
        )

    def test_custom_rotary_base_changes_decay(self):
        # base=100=10**2 -> inv_freq[i] = 100 ** (-(2*i)/8) = 10 ** (-(i)/2)
        emb = RotaryEmbedding(head_dim=8, rotary_percent=1.0, rotary_base=100)
        expected = [100.0 ** (-(2 * i) / 8) for i in range(4)]
        np.testing.assert_allclose(
            emb.inv_freq.numpy().astype(np.float64), expected, rtol=1e-12
        )

    def test_partial_rotary_shrinks_dimension(self):
        # rotary_percent=0.5 -> dim = int(8 * 0.5) = 4, indices arange(0,4,2)=[0,2]
        # inv_freq = 10000 ** (-(2*i)/4) = [1.0, 10000**-1 = 1e-4]
        emb = RotaryEmbedding(head_dim=8, rotary_percent=0.5)
        actual = emb.inv_freq.numpy().astype(np.float64)
        self.assertEqual(actual.shape, (2,))
        np.testing.assert_allclose(
            actual, np.array([1.0, 1e-4]), rtol=0, atol=1e-12
        )


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestGetFreqsNonRepeated(unittest.TestCase):
    def test_freqs_are_position_times_inv_freq(self):
        emb = RotaryEmbedding(head_dim=8, rotary_percent=1.0)
        freqs = emb.get_freqs_non_repeated(max_seq_len=3).numpy()
        expected = _ref_freqs([0, 1, 2], INV_FREQ_HD8)
        # row 1 must be exactly the inv_freq vector; row 0 all zeros.
        self.assertEqual(freqs.shape, (3, 4))
        np.testing.assert_allclose(freqs, expected, rtol=1e-6, atol=1e-9)

    def test_offset_shifts_positions(self):
        emb = RotaryEmbedding(head_dim=8, rotary_percent=1.0)
        freqs = emb.get_freqs_non_repeated(max_seq_len=2, offset=5).numpy()
        expected = _ref_freqs([5, 6], INV_FREQ_HD8)
        np.testing.assert_allclose(freqs, expected, rtol=1e-6, atol=1e-9)

    def test_interpolation_factor_scales_positions(self):
        # seq_len_interpolation_factor=2.0 divides positions by 2.
        emb = RotaryEmbedding(
            head_dim=8, rotary_percent=1.0, seq_len_interpolation_factor=2.0
        )
        freqs = emb.get_freqs_non_repeated(max_seq_len=4).numpy()
        expected = _ref_freqs([0.0, 0.5, 1.0, 1.5], INV_FREQ_HD8)
        np.testing.assert_allclose(freqs, expected, rtol=1e-6, atol=1e-9)


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestGetCosSin(unittest.TestCase):
    def test_cos_sin_are_trig_of_freqs(self):
        emb = RotaryEmbedding(head_dim=8, rotary_percent=1.0)
        cos, sin = emb.get_cos_sin(max_seq_len=3)
        freqs = _ref_freqs([0, 1, 2], INV_FREQ_HD8)
        np.testing.assert_allclose(
            cos.numpy(), np.cos(freqs), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            sin.numpy(), np.sin(freqs), rtol=1e-6, atol=1e-6
        )
        # position 0 -> cos row is all ones, sin row is all zeros.
        np.testing.assert_allclose(cos.numpy()[0], np.ones(4), atol=1e-6)
        np.testing.assert_allclose(sin.numpy()[0], np.zeros(4), atol=1e-6)


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestForwardLayout(unittest.TestCase):
    def test_default_layout_concatenates_half_table(self):
        emb = RotaryEmbedding(head_dim=8, rotary_percent=1.0)
        out = emb(max_seq_len=2).numpy()
        self.assertEqual(out.shape, (1, 2, 1, 8))
        half = _ref_freqs([0, 1], INV_FREQ_HD8)
        # default (non-interleaved): emb = concat(freqs, freqs) along last axis.
        expected = np.concatenate([half, half], axis=-1)[None, :, None, :]
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-9)
        # spot check position 1: [1,.1,.01,.001, 1,.1,.01,.001]
        np.testing.assert_allclose(
            out[0, 1, 0, :],
            np.array([1, 0.1, 0.01, 0.001, 1, 0.1, 0.01, 0.001]),
            rtol=1e-6,
            atol=1e-9,
        )

    def test_interleaved_layout_duplicates_pairwise(self):
        emb = RotaryEmbedding(
            head_dim=8, rotary_percent=1.0, rotary_interleaved=True
        )
        out = emb(max_seq_len=2).numpy()
        self.assertEqual(out.shape, (1, 2, 1, 8))
        half = _ref_freqs([0, 1], INV_FREQ_HD8)
        # interleaved: each value duplicated adjacently -> [a,a,b,b,c,c,d,d].
        expected = np.repeat(half, 2, axis=-1)[None, :, None, :]
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(
            out[0, 1, 0, :],
            np.array([1, 1, 0.1, 0.1, 0.01, 0.01, 0.001, 0.001]),
            rtol=1e-6,
            atol=1e-9,
        )

    def test_interleaved_differs_from_default(self):
        default = RotaryEmbedding(head_dim=8, rotary_percent=1.0)(
            max_seq_len=3
        ).numpy()
        inter = RotaryEmbedding(
            head_dim=8, rotary_percent=1.0, rotary_interleaved=True
        )(max_seq_len=3).numpy()
        self.assertFalse(np.allclose(default, inter))


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestGetRotarySeqLen(unittest.TestCase):
    def test_uses_axis_1_without_sequence_parallel(self):
        emb = RotaryEmbedding(head_dim=8, rotary_percent=1.0)
        cfg = SimpleNamespace(
            sequence_parallel=False, tensor_model_parallel_size=4
        )
        x = paddle.zeros([2, 7, 8], dtype="float32")
        # cp_group is None (no CP init) and sequence_parallel False -> just shape[1].
        self.assertEqual(emb.get_rotary_seq_len(x, cfg), 7)

    def test_sequence_parallel_applies_tp_multiplier(self):
        emb = RotaryEmbedding(head_dim=8, rotary_percent=1.0)
        cfg = SimpleNamespace(
            sequence_parallel=True, tensor_model_parallel_size=4
        )
        # sequence_parallel True -> axis 0, then * tensor_model_parallel_size.
        x = paddle.zeros([3, 7, 8], dtype="float32")
        self.assertEqual(emb.get_rotary_seq_len(x, cfg), 12)

    def test_packed_seq_params_take_max(self):
        emb = RotaryEmbedding(head_dim=8, rotary_percent=1.0)
        cfg = SimpleNamespace(
            sequence_parallel=False, tensor_model_parallel_size=1
        )
        packed = SimpleNamespace(max_seqlen_q=32, max_seqlen_kv=48)
        x = paddle.zeros([2, 7, 8], dtype="float32")
        self.assertEqual(
            emb.get_rotary_seq_len(x, cfg, packed_seq_params=packed), 48
        )

    def test_context_parallel_group_multiplies_length(self):
        # cp_group is a plain data holder (not the unit under test); the length
        # multiplication by world_size is the real logic being observed.
        cp_group = SimpleNamespace(world_size=2)
        emb = RotaryEmbedding(head_dim=8, rotary_percent=1.0, cp_group=cp_group)
        cfg = SimpleNamespace(
            sequence_parallel=False, tensor_model_parallel_size=1
        )
        x = paddle.zeros([2, 7, 8], dtype="float32")
        self.assertEqual(emb.get_rotary_seq_len(x, cfg), 14)


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestForwardCache(unittest.TestCase):
    def test_cache_hit_returns_bit_identical_table(self):
        emb = RotaryEmbedding(
            head_dim=8, rotary_percent=1.0, rotary_embed_cache=True
        )
        first = emb(max_seq_len=5)
        second = emb(max_seq_len=5)
        # Same (max_seq_len, offset) key -> cached object is returned unchanged.
        self.assertIs(first, second)
        np.testing.assert_array_equal(first.numpy(), second.numpy())

    def test_cache_disabled_recomputes_new_object(self):
        emb = RotaryEmbedding(head_dim=8, rotary_percent=1.0)
        first = emb(max_seq_len=5)
        second = emb(max_seq_len=5)
        self.assertIsNot(first, second)
        # Value is still identical -- recompute is deterministic.
        np.testing.assert_allclose(first.numpy(), second.numpy())


if __name__ == "__main__":
    unittest.main()
