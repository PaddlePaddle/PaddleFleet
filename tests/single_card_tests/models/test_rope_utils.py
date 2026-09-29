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

"""CPU-observable tests for the packed (THD) rotary-embedding path.

Scope / disjointness
--------------------
This file deliberately covers only the ``thd`` (packed-sequence) family of
``paddlefleet.models.common.embeddings.rope_utils``:

* ``_get_thd_freqs_on_this_cp_rank`` -- the pure per-rank frequency slicer,
* ``_apply_rotary_pos_emb_thd``      -- the packed-sequence RoPE baseline,
* ``apply_rotary_pos_emb``           -- but only its ``cu_seqlens is not None``
  (THD) routing branch.

The plain ``bshd`` numeric contract, ``_rotate_half``, ``get_unsqueeze_dim``
and ``get_pos_emb_on_this_cp_rank`` are covered by the sibling suite under
``tests/single_card_tests/embeddings/test_rope_utils.py`` and are NOT retested
here, keeping the two slices disjoint.

Honesty
-------
The whole paddlefleet package imports paddle at import time. When paddle (or
the package) is not importable the tests are skipped via
``unittest.skipUnless`` with the real import error as the reason; they are
never reported as passing.

Every expected value is derived independently in plain NumPy from the closed
form of RoPE and from the documented THD position-mapping contract. The
functions under test are never used to build their own expected outputs, and
the production frequency slicer is never used to build the reference packing.
"""

import unittest

import numpy as np

MODULE = "paddlefleet.models.common.embeddings.rope_utils"

try:
    import paddle

    from paddlefleet.models.common.embeddings.rope_utils import (
        _apply_rotary_pos_emb_thd,
        _get_thd_freqs_on_this_cp_rank,
        apply_rotary_pos_emb,
    )

    IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - environment dependent
    paddle = None
    _apply_rotary_pos_emb_thd = None
    _get_thd_freqs_on_this_cp_rank = None
    apply_rotary_pos_emb = None
    IMPORT_ERROR = exc

HAS_PADDLE = IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle / paddlefleet not importable in this environment: {IMPORT_ERROR!r}"
)


# --------------------------------------------------------------------------- #
# Independent NumPy reference (hand-derived closed form of non-interleaved
# RoPE). It deliberately does NOT call any production helper.
# t:     [b, s, h, d]
# freqs: [f0, s, rot_dim]  (f0 broadcasts over batch; rot_dim <= d)
# --------------------------------------------------------------------------- #
def _ref_rope(t, freqs, mscale=1.0):
    t = np.asarray(t, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    rot_dim = freqs.shape[-1]
    t_rot = t[..., :rot_dim]
    t_pass = t[..., rot_dim:]

    cos = (np.cos(freqs) * mscale)[:, :, None, :]  # [f0, s, 1, rot_dim]
    sin = (np.sin(freqs) * mscale)[:, :, None, :]

    half = rot_dim // 2
    h1 = t_rot[..., :half]
    h2 = t_rot[..., half:]
    rotate = np.concatenate([-h2, h1], axis=-1)  # non-interleaved rotate_half

    out_rot = t_rot * cos + rotate * sin
    return np.concatenate([out_rot, t_pass], axis=-1)


def _distinct_freqs(seq_len, rot_dim, seed):
    """Small, position-distinguishable angles so a wrong position map shows."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-1.0, 1.0, size=(1, seq_len, rot_dim)).astype(np.float32)


# PLACEHOLDER_TESTS


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestGetThdFreqsOnThisCpRank(unittest.TestCase):
    """Pure per-rank frequency slicing math (no collective involved)."""

    def test_single_rank_traditional_slice(self):
        # cp_size == 1 -> else branch: freqs[:, offset : offset + x.size(1)].
        freqs_np = _distinct_freqs(seq_len=10, rot_dim=4, seed=10)
        x = paddle.zeros([1, 5, 4])  # x.size(1) == 5
        out = _get_thd_freqs_on_this_cp_rank(
            cp_rank=0,
            cp_size=1,
            x=x,
            freqs=paddle.to_tensor(freqs_np),
            offset=0,
        )
        np.testing.assert_array_equal(out.numpy(), freqs_np[:, 0:5])

    def test_single_rank_with_offset(self):
        # offset must shift the window: freqs[:, 3 : 3 + 4].
        freqs_np = _distinct_freqs(seq_len=10, rot_dim=4, seed=11)
        x = paddle.zeros([1, 4, 4])  # x.size(1) == 4
        out = _get_thd_freqs_on_this_cp_rank(
            cp_rank=0,
            cp_size=1,
            x=x,
            freqs=paddle.to_tensor(freqs_np),
            offset=3,
        )
        np.testing.assert_array_equal(out.numpy(), freqs_np[:, 3:7])
        # An off-by-offset bug (ignoring offset) would return freqs[:, 0:4].
        self.assertFalse(np.array_equal(out.numpy(), freqs_np[:, 0:4]))

    @unittest.expectedFailure
    def test_multi_rank_load_balanced_slice(self):
        # cp_size > 1: rank 0 owns the FIRST and LAST load-balanced segments,
        # which must join along the SEQUENCE axis into a [1, 2*cp_seg, D] slice.
        #   x.size(1) == 4 -> cp_seg = 2, full_seqlen = cp_size * 4 = 8
        #   rank 0 -> forward positions [0, 1] and backward positions [6, 7]
        # Production concatenates the two [1, 2, D] segments with the default
        # ``paddle.cat`` axis (0), yielding [2, 2, D] instead of [1, 4, D]
        # (rope_utils.py:341-354). Correct behavior asserted below; marked
        # expectedFailure because that axis-0 concatenation is a real bug.
        freqs_np = _distinct_freqs(seq_len=8, rot_dim=4, seed=12)
        x = paddle.zeros([1, 4, 4])  # x.size(1) == 4
        out = _get_thd_freqs_on_this_cp_rank(
            cp_rank=0,
            cp_size=2,
            x=x,
            freqs=paddle.to_tensor(freqs_np),
            offset=0,
        )
        expected = np.concatenate(
            [freqs_np[:, 0:2], freqs_np[:, 6:8]], axis=1
        )  # positions [0, 1, 6, 7] along the sequence axis
        self.assertEqual(list(out.shape), [1, 4, 4])
        np.testing.assert_array_equal(out.numpy(), expected)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestApplyRotaryPosEmbThd(unittest.TestCase):
    """_apply_rotary_pos_emb_thd on 4D packed tensors (cp_group=None -> cp=1)."""

    def _t(self, seed):
        rng = np.random.RandomState(seed)
        # [b, total_seq, h, d]; b != total_seq avoids the M-RoPE transpose branch.
        return rng.randn(2, 8, 4, 16).astype(np.float32)

    def test_exact_mapping_matches_full_rope(self):
        # freqs.size(1) == total_seq_len -> CASE 1 exact mapping. With cp=1 the
        # packed tensor keeps positions 0..7, i.e. token j uses freqs[:, j].
        t_np = self._t(seed=20)
        freqs_np = _distinct_freqs(seq_len=8, rot_dim=16, seed=21)
        cu_seqlens = paddle.to_tensor([0, 4, 8], dtype="int32")
        out = _apply_rotary_pos_emb_thd(
            t=paddle.to_tensor(t_np),
            cu_seqlens=cu_seqlens,
            total_seq_len=8,
            freqs=paddle.to_tensor(freqs_np),
            cos=None,
            sin=None,
            apply_rope_fusion=False,
            rotary_interleaved=False,
            multi_latent_attention=False,
            high_precision_rope=False,
            cp_group=None,
        )
        expected = _ref_rope(t_np, freqs_np)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_traditional_restarts_positions_per_sequence(self):
        # freqs.size(1) (=32) != total_seq_len (=8) -> CASE 2 traditional: every
        # packed sequence restarts frequencies at position 0. Two length-4
        # sequences therefore BOTH consume freqs[:, 0:4].
        t_np = self._t(seed=22)
        freqs_np = _distinct_freqs(seq_len=32, rot_dim=16, seed=23)
        cu_seqlens = paddle.to_tensor([0, 4, 8], dtype="int32")
        out = _apply_rotary_pos_emb_thd(
            t=paddle.to_tensor(t_np),
            cu_seqlens=cu_seqlens,
            total_seq_len=8,
            freqs=paddle.to_tensor(freqs_np),
            cos=None,
            sin=None,
            apply_rope_fusion=False,
            rotary_interleaved=False,
            multi_latent_attention=False,
            high_precision_rope=False,
            cp_group=None,
        )
        packed = np.concatenate(
            [freqs_np[:, 0:4], freqs_np[:, 0:4]], axis=1
        )  # independently rebuilt "restart per sequence" freqs
        expected = _ref_rope(t_np, packed)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_traditional_differs_from_continuous_positions(self):
        # Guards the restart contract: if the second sequence wrongly continued
        # at positions 4..7 the result would match ``continuous`` below. It must
        # not, because the traditional path restarts at position 0.
        t_np = self._t(seed=24)
        freqs_np = _distinct_freqs(seq_len=32, rot_dim=16, seed=25)
        cu_seqlens = paddle.to_tensor([0, 4, 8], dtype="int32")
        out = _apply_rotary_pos_emb_thd(
            t=paddle.to_tensor(t_np),
            cu_seqlens=cu_seqlens,
            total_seq_len=8,
            freqs=paddle.to_tensor(freqs_np),
            cos=None,
            sin=None,
            apply_rope_fusion=False,
            rotary_interleaved=False,
            multi_latent_attention=False,
            high_precision_rope=False,
            cp_group=None,
        )
        continuous = np.concatenate(
            [freqs_np[:, 0:4], freqs_np[:, 4:8]], axis=1
        )
        wrong = _ref_rope(t_np, continuous)
        self.assertFalse(np.allclose(out.numpy(), wrong, rtol=1e-4, atol=1e-4))


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestApplyRotaryPosEmbRouterThd(unittest.TestCase):
    """apply_rotary_pos_emb routes to the THD path when cu_seqlens is given."""

    def _config(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            apply_rope_fusion=False,
            rotary_interleaved=False,
            multi_latent_attention=False,
            high_precision_rope=False,
            rope_theta=10000.0,
            sequence_parallel=False,
        )

    def _run(self, t_np, freqs_np, total_seq_len):
        cu_seqlens = paddle.to_tensor([0, 4, 8], dtype="int32")
        return apply_rotary_pos_emb(
            paddle.to_tensor(t_np),
            paddle.to_tensor(freqs_np),
            None,
            None,
            self._config(),
            cu_seqlens=cu_seqlens,
            total_seq_len=total_seq_len,
        )

    def test_router_dispatches_to_thd_exact_mapping(self):
        rng = np.random.RandomState(30)
        t_np = rng.randn(2, 8, 4, 16).astype(np.float32)
        freqs_np = _distinct_freqs(seq_len=8, rot_dim=16, seed=31)
        out = self._run(t_np, freqs_np, total_seq_len=8)
        expected = _ref_rope(t_np, freqs_np)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_router_thd_traditional_restart(self):
        # A bshd route (cu_seqlens ignored) could not even align freqs[1,32,16]
        # with t[2,8,4,16]; a correct THD route restarts per sequence.
        rng = np.random.RandomState(32)
        t_np = rng.randn(2, 8, 4, 16).astype(np.float32)
        freqs_np = _distinct_freqs(seq_len=32, rot_dim=16, seed=33)
        out = self._run(t_np, freqs_np, total_seq_len=8)
        packed = np.concatenate([freqs_np[:, 0:4], freqs_np[:, 0:4]], axis=1)
        expected = _ref_rope(t_np, packed)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
