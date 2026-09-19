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

"""CPU-observable numeric tests for paddlefleet rope_utils (disjoint slice).

This file deliberately targets branches of ``apply_rotary_pos_emb`` and its
private thd helpers that are NOT exercised by
``tests/single_card_tests/embeddings/test_rope_utils.py`` (which only covers the
bshd route with ``cu_seqlens=None`` plus ``_rotate_half`` / ``get_unsqueeze_dim``
/ the ``seq_dim=0`` context-parallel slice). Here we cover instead:

- the packed ``thd`` route (``cu_seqlens`` provided): both the ``cp_size==1``
  short-circuit (CASE 1) and the traditional per-sequence-from-zero packing
  (CASE 2);
- ``_get_thd_freqs_on_this_cp_rank`` single-rank windowing with an offset;
- the M-RoPE ``freqs`` transpose branch that fires when the leading two dims of
  ``freqs`` are swapped relative to ``t`` (b != s), which test_rope_utils
  explicitly avoids;
- the ``inverse=True`` sin-negation path and its forward/inverse round-trip;
- ``get_pos_emb_on_this_cp_rank`` with ``seq_dim=1`` (leading batch dim kept).

Honesty notes:
- paddle imports at package import time. When paddle / paddlefleet is not
  importable the whole module is skipped via ``unittest.skipUnless`` with the
  real ImportError text; it is never reported as passing.
- Every expected value is hand-derived in plain float64 NumPy from the closed
  form of RoPE. The functions under test are never used to build their own
  expected outputs.
- The ONLY collaborators replaced are the topology providers ``get_pg_size`` /
  ``get_pg_rank`` (they need a real process group). We inject a single CP rank
  (size=1 / rank=0) or fixed size/rank; these functions do no collective, so
  this exercises the local single-rank path only. No real multi-rank
  communication is verified here.
"""

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

# Make the in-tree package importable when it is not installed.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

MODULE = "paddlefleet.models.common.embeddings.rope_utils"

try:
    import paddle

    from paddlefleet.models.common.embeddings.rope_utils import (
        _apply_rotary_pos_emb_thd,
        _get_thd_freqs_on_this_cp_rank,
        apply_rotary_pos_emb,
        get_pos_emb_on_this_cp_rank,
    )

    IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - environment dependent
    paddle = None
    _apply_rotary_pos_emb_thd = None
    _get_thd_freqs_on_this_cp_rank = None
    apply_rotary_pos_emb = None
    get_pos_emb_on_this_cp_rank = None
    IMPORT_ERROR = exc

HAS_PADDLE = IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle / paddlefleet not importable in this environment: {IMPORT_ERROR!r}"
)


# --------------------------------------------------------------------------- #
# Independent NumPy references (hand-derived closed form of RoPE).
# None of these call any production helper.
# --------------------------------------------------------------------------- #
def _ref_rotate_half_noninterleaved(x):
    """[.., h1, h2] -> concat(-h2, h1). Works for any trailing-even dim."""
    d = x.shape[-1]
    h = d // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def _ref_rotate_half_interleaved(x):
    """Pairs (x0,x1),(x2,x3),.. -> (-x1,x0,-x3,x2,..)."""
    even = x[..., 0::2]
    odd = x[..., 1::2]
    stacked = np.stack([-odd, even], axis=-1)
    return stacked.reshape(*x.shape[:-1], -1)


def _ref_rope_bshd(
    t,
    freqs,
    mscale=1.0,
    interleaved=False,
    mla=False,
    inverse=False,
    mla_out=False,
):
    """Reference RoPE for t [b, s, h, d] with freqs [b, s, rot_dim].

    Mirrors the eager (non-fused, non-high-precision) production path:
    optional MLA input de-interleave, cos/sin scaled by mscale, sin negated
    when ``inverse``, rotate-half add, optional MLA output re-interleave, then
    concatenate the un-rotated tail.
    """
    t = np.asarray(t, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    rot_dim = freqs.shape[-1]
    t_rot = t[..., :rot_dim]
    t_pass = t[..., rot_dim:]

    if mla:
        even = t_rot[..., 0::2]
        odd = t_rot[..., 1::2]
        t_rot = np.concatenate([even, odd], axis=-1)

    cos = (np.cos(freqs) * mscale)[:, :, None, :]
    sin = (np.sin(freqs) * mscale)[:, :, None, :]
    if inverse:
        sin = -sin
    rot = (
        _ref_rotate_half_interleaved(t_rot)
        if interleaved
        else _ref_rotate_half_noninterleaved(t_rot)
    )
    out_rot = t_rot * cos + rot * sin

    if mla and mla_out:
        half = out_rot.shape[-1] // 2
        x1 = out_rot[..., :half]
        x2 = out_rot[..., half:]
        out_rot = np.stack([x1, x2], axis=-1).reshape(*out_rot.shape[:-1], -1)

    return np.concatenate([out_rot, t_pass], axis=-1)


class _Config:
    """Minimal stand-in exposing exactly the attributes rope_utils reads."""

    def __init__(self, **overrides):
        self.apply_rope_fusion = False
        self.rotary_interleaved = False
        self.multi_latent_attention = False
        self.high_precision_rope = False
        self.rope_theta = 10000.0
        self.sequence_parallel = False
        for key, value in overrides.items():
            setattr(self, key, value)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestApplyRotaryPosEmbThdRoute(unittest.TestCase):
    """Packed ``thd`` route of apply_rotary_pos_emb (cu_seqlens provided).

    Single CP rank (size=1 / rank=0) is injected via the topology providers,
    so only the local single-rank packing path runs -- no cross-rank
    communication is exercised or claimed.
    """

    def test_case1_full_freqs_cp1_equals_bshd(self):
        # freqs.size(1) == total_seq_len -> CASE 1; with cp_size==1 the thd
        # helper short-circuits straight to the bshd math over the full freqs.
        rng = np.random.RandomState(10)
        t_np = rng.randn(1, 5, 2, 4).astype(np.float32)
        freqs_np = rng.randn(1, 5, 4).astype(np.float32)  # seq==cu_seqlens[-1]
        cu = np.array([0, 2, 5], dtype=np.int32)
        with (
            patch(f"{MODULE}.get_pg_size", return_value=1),
            patch(f"{MODULE}.get_pg_rank", return_value=0),
        ):
            out = apply_rotary_pos_emb(
                paddle.to_tensor(t_np),
                paddle.to_tensor(freqs_np),
                cos=None,
                sin=None,
                config=_Config(),
                cu_seqlens=paddle.to_tensor(cu),
            )
        expected = _ref_rope_bshd(t_np, freqs_np)
        self.assertEqual(list(out.shape), [1, 5, 2, 4])
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_case2_packs_each_sequence_from_position_zero(self):
        # freqs.size(1) != total_seq_len -> CASE 2. Two packed sequences of
        # length 2 and 3 each restart from frequency position 0. The reference
        # rebuilds that packed-freqs layout independently, then applies RoPE.
        rng = np.random.RandomState(11)
        t_np = rng.randn(1, 5, 1, 4).astype(np.float32)  # 2 + 3 tokens
        freqs_np = rng.randn(1, 3, 4).astype(np.float32)  # max_s = 3 (!= 5)
        cu = np.array([0, 2, 5], dtype=np.int32)
        with (
            patch(f"{MODULE}.get_pg_size", return_value=1),
            patch(f"{MODULE}.get_pg_rank", return_value=0),
        ):
            out = apply_rotary_pos_emb(
                paddle.to_tensor(t_np),
                paddle.to_tensor(freqs_np),
                cos=None,
                sin=None,
                config=_Config(),
                cu_seqlens=paddle.to_tensor(cu),
            )
        freqs_packed = np.concatenate(
            [freqs_np[:, 0:2, :], freqs_np[:, 0:3, :]], axis=1
        )
        expected = _ref_rope_bshd(t_np, freqs_packed)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)
        # Negative control: naively using the *global* positions 0..4 (i.e. not
        # restarting each sequence at 0) would give a different result, so this
        # assertion would fail if the per-sequence packing were dropped.
        wrong = _ref_rope_bshd(
            t_np, np.concatenate([freqs_np, freqs_np[:, :2, :]], axis=1)
        )
        self.assertFalse(np.allclose(out.numpy(), wrong, rtol=1e-3, atol=1e-3))


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestGetThdFreqsSingleRank(unittest.TestCase):
    """_get_thd_freqs_on_this_cp_rank: single-rank windowing by offset+length.

    With cp_size==1 the function returns ``freqs[:, offset:offset+x.size(1)]``.
    x.size(1) is the sequence length of the (4D) split; the window must honour
    both the length and the start offset.
    """

    def _slice(self, seq_len, offset, freq_positions, dim):
        x = paddle.zeros([1, seq_len, 1, 1])  # x.size(1) == seq_len
        freqs_np = np.arange(freq_positions * dim, dtype=np.float32).reshape(
            1, freq_positions, dim
        )
        out = _get_thd_freqs_on_this_cp_rank(
            0, 1, x, paddle.to_tensor(freqs_np), offset
        )
        return freqs_np, out.numpy()

    def test_offset_zero_takes_prefix(self):
        freqs_np, out = self._slice(
            seq_len=3, offset=0, freq_positions=6, dim=2
        )
        np.testing.assert_array_equal(out, freqs_np[:, 0:3, :])

    def test_offset_shifts_window(self):
        freqs_np, out = self._slice(
            seq_len=3, offset=2, freq_positions=6, dim=2
        )
        np.testing.assert_array_equal(out, freqs_np[:, 2:5, :])
        # The window genuinely moved: it must differ from the offset-0 prefix.
        self.assertFalse(np.array_equal(out, freqs_np[:, 0:3, :]))


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestMRopeFreqsTranspose(unittest.TestCase):
    """M-RoPE branch: freqs given as [S, B, D] must be transposed to [B, S, D].

    test_rope_utils deliberately uses b != s only to *avoid* this branch; here
    we drive it. A plain reshape would reinterpret memory without reordering
    and give wrong values for b != s, so we assert the transpose semantics and
    that a reshape interpretation is rejected.
    """

    def test_swapped_dims_are_transposed_not_reshaped(self):
        rng = np.random.RandomState(20)
        b, s, h, d = 2, 3, 1, 4  # b != s so transpose vs reshape differ
        t_np = rng.randn(b, s, h, d).astype(np.float32)
        freqs_sbd = rng.randn(s, b, d).astype(np.float32)  # [S, B, D]
        with (
            patch(f"{MODULE}.get_pg_size", return_value=1),
            patch(f"{MODULE}.get_pg_rank", return_value=0),
        ):
            out = apply_rotary_pos_emb(
                paddle.to_tensor(t_np),
                paddle.to_tensor(freqs_sbd),
                cos=None,
                sin=None,
                config=_Config(),
                cu_seqlens=None,
            )
        expected = _ref_rope_bshd(t_np, np.transpose(freqs_sbd, (1, 0, 2)))
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)
        # A reshape (not transpose) of [S,B,D]->[B,S,D] reorders data wrongly.
        reshaped = _ref_rope_bshd(t_np, freqs_sbd.reshape(b, s, d))
        self.assertFalse(
            np.allclose(out.numpy(), reshaped, rtol=1e-3, atol=1e-3)
        )


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestInverseRotation(unittest.TestCase):
    """inverse=True negates the sin component (reverse RoPE rotation)."""

    def test_inverse_matches_negated_sin_reference(self):
        rng = np.random.RandomState(30)
        t_np = rng.randn(1, 2, 1, 4).astype(np.float32)
        freqs_np = rng.randn(1, 2, 4).astype(np.float32)
        with (
            patch(f"{MODULE}.get_pg_size", return_value=1),
            patch(f"{MODULE}.get_pg_rank", return_value=0),
        ):
            out = apply_rotary_pos_emb(
                paddle.to_tensor(t_np),
                paddle.to_tensor(freqs_np),
                cos=None,
                sin=None,
                config=_Config(),
                cu_seqlens=None,
                inverse=True,
            )
        expected = _ref_rope_bshd(t_np, freqs_np, inverse=True)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)
        # inverse must actually differ from the forward rotation for this input.
        forward = _ref_rope_bshd(t_np, freqs_np, inverse=False)
        self.assertFalse(np.allclose(expected, forward, rtol=1e-3, atol=1e-3))

    def test_forward_then_inverse_recovers_input(self):
        # A proper RoPE frequency tensor duplicates each frequency across a
        # rotate-half partner pair (freqs = concat(base, base)). Under that
        # layout cos^2 + sin^2 == 1 per rotation plane, so forward followed by
        # inverse is an identity on the fully-rotated head (rot_dim == d),
        # independent of the numeric reference above.
        rng = np.random.RandomState(31)
        base = rng.randn(1, 3, 2).astype(np.float32)
        freqs_np = np.concatenate([base, base], axis=-1)  # [1, 3, 4]
        t_np = rng.randn(1, 3, 2, 4).astype(np.float32)
        cfg = _Config()
        with (
            patch(f"{MODULE}.get_pg_size", return_value=1),
            patch(f"{MODULE}.get_pg_rank", return_value=0),
        ):
            fwd = apply_rotary_pos_emb(
                paddle.to_tensor(t_np),
                paddle.to_tensor(freqs_np),
                cos=None,
                sin=None,
                config=cfg,
                cu_seqlens=None,
                inverse=False,
            )
            back = apply_rotary_pos_emb(
                fwd,
                paddle.to_tensor(freqs_np),
                cos=None,
                sin=None,
                config=cfg,
                cu_seqlens=None,
                inverse=True,
            )
        np.testing.assert_allclose(back.numpy(), t_np, rtol=1e-4, atol=1e-5)
        # Guard: the forward pass really rotated (so the round-trip is
        # non-trivial), not an accidental identity.
        self.assertFalse(np.allclose(fwd.numpy(), t_np, rtol=1e-3, atol=1e-3))


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestGetPosEmbSeqDimOne(unittest.TestCase):
    """get_pos_emb_on_this_cp_rank with seq_dim=1 keeps the leading batch dim.

    Distinct from test_rope_utils, which only covers seq_dim=0 on a 2D tensor.
    This is a pure local row-selection given a fixed rank/size (no collective),
    so injecting the topology providers exercises the real reshape +
    index_select + reshape logic.
    """

    def _select(self, cp_size, cp_rank, pos_np):
        with (
            patch(f"{MODULE}.get_pg_size", return_value=cp_size),
            patch(f"{MODULE}.get_pg_rank", return_value=cp_rank),
        ):
            out = get_pos_emb_on_this_cp_rank(
                paddle.to_tensor(pos_np), seq_dim=1, cp_group=object()
            )
        return out.numpy()

    def test_leading_dim_preserved_and_rows_load_balanced(self):
        # B=2 (distinguishable per batch), S=8, F=2; 2*cp_size=4 segments of
        # length 2. rank 0 -> cp_idx=[0, 3] -> seq rows [0,1] and [6,7].
        pos_np = np.arange(2 * 8 * 2, dtype=np.float32).reshape(2, 8, 2)
        out = self._select(cp_size=2, cp_rank=0, pos_np=pos_np)
        expected = pos_np[:, [0, 1, 6, 7], :]
        self.assertEqual(list(out.shape), [2, 4, 2])
        np.testing.assert_array_equal(out, expected)

    def test_rank1_selects_middle_rows_per_batch(self):
        pos_np = np.arange(2 * 8 * 2, dtype=np.float32).reshape(2, 8, 2)
        out = self._select(cp_size=2, cp_rank=1, pos_np=pos_np)
        # rank 1 -> cp_idx=[1, 2] -> seq rows [2,3] and [4,5].
        expected = pos_np[:, [2, 3, 4, 5], :]
        np.testing.assert_array_equal(out, expected)


if __name__ == "__main__":
    unittest.main()
