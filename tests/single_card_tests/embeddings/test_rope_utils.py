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

"""CPU-observable behavior tests for paddlefleet rope_utils.

The whole paddlefleet package imports paddle at import time. If paddle (or the
package) is not importable in this environment the tests are honestly skipped
via ``unittest.skipUnless``; they are NOT reported as passing.

Design notes:
- Every expected value is derived independently in plain NumPy from the closed
  form of rotary position embedding. The functions under test are never called
  to compute their own expected outputs.
- ``get_pos_emb_on_this_cp_rank`` reads the context-parallel topology through
  ``get_pg_size`` / ``get_pg_rank``. Those two topology providers are the ONLY
  collaborators replaced (they need a real process group). We feed fixed,
  distinguishable ranks/sizes and then assert the REAL reshape + index_select +
  reshape logic selects the correct load-balanced sequence rows by content.
"""

import unittest
from unittest.mock import patch

import numpy as np

MODULE = "paddlefleet.models.common.embeddings.rope_utils"

try:
    import paddle

    from paddlefleet.models.common.embeddings.rope_utils import (
        _apply_rotary_pos_emb_bshd,
        _rotate_half,
        apply_rotary_pos_emb,
        get_pos_emb_on_this_cp_rank,
        get_unsqueeze_dim,
    )

    IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - environment dependent
    paddle = None
    _apply_rotary_pos_emb_bshd = None
    _rotate_half = None
    apply_rotary_pos_emb = None
    get_pos_emb_on_this_cp_rank = None
    get_unsqueeze_dim = None
    IMPORT_ERROR = exc

HAS_PADDLE = IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle / paddlefleet not importable in this environment: {IMPORT_ERROR!r}"
)


# --------------------------------------------------------------------------- #
# Independent NumPy references (hand-derived closed form of RoPE).
# These deliberately do NOT call any production helper.
# --------------------------------------------------------------------------- #
def _ref_rotate_half_noninterleaved(x):
    """[.., a, b] with halves (h1, h2) -> concat(-h2, h1)."""
    d = x.shape[-1]
    h = d // 2
    h1 = x[..., :h]
    h2 = x[..., h:]
    return np.concatenate([-h2, h1], axis=-1)


def _ref_rotate_half_interleaved(x):
    """Pairs (x0,x1),(x2,x3),.. -> (-x1,x0,-x3,x2,..)."""
    even = x[..., 0::2]
    odd = x[..., 1::2]
    stacked = np.stack([-odd, even], axis=-1)
    return stacked.reshape(*x.shape[:-1], -1)


def _ref_rope_bshd(t, freqs, mscale=1.0, interleaved=False, mla=False):
    """Reference RoPE for t [b, s, h, d] with freqs [b, s, rot_dim].

    Only the leading ``rot_dim`` channels are rotated; the tail passes through.
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
    if interleaved:
        rot = _ref_rotate_half_interleaved(t_rot)
    else:
        rot = _ref_rotate_half_noninterleaved(t_rot)
    out_rot = t_rot * cos + rot * sin
    return np.concatenate([out_rot, t_pass], axis=-1)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestRotateHalf(unittest.TestCase):
    """_rotate_half must match the hand-derived sign/position permutation."""

    def test_non_interleaved_permutation(self):
        x_np = np.array([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=np.float32)
        out = _rotate_half(paddle.to_tensor(x_np), rotary_interleaved=False)
        # [1,2,3,4] -> halves [1,2] and [3,4] -> concat(-[3,4], [1,2])
        expected = np.array([[[[-3.0, -4.0, 1.0, 2.0]]]], dtype=np.float32)
        np.testing.assert_array_equal(out.numpy(), expected)
        np.testing.assert_array_equal(
            out.numpy(), _ref_rotate_half_noninterleaved(x_np)
        )

    def test_interleaved_permutation(self):
        x_np = np.array([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=np.float32)
        out = _rotate_half(paddle.to_tensor(x_np), rotary_interleaved=True)
        # pairs (1,2),(3,4) -> (-2, 1, -4, 3)
        expected = np.array([[[[-2.0, 1.0, -4.0, 3.0]]]], dtype=np.float32)
        np.testing.assert_array_equal(out.numpy(), expected)
        np.testing.assert_array_equal(
            out.numpy(), _ref_rotate_half_interleaved(x_np)
        )

    def test_interleaved_differs_from_non_interleaved(self):
        # A distinguishable input where the two modes cannot coincide.
        x_np = np.arange(8, dtype=np.float32).reshape(1, 1, 1, 8) + 1.0
        x = paddle.to_tensor(x_np)
        a = _rotate_half(x, rotary_interleaved=False).numpy()
        b = _rotate_half(x, rotary_interleaved=True).numpy()
        self.assertFalse(np.array_equal(a, b))


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestGetUnsqueezeDim(unittest.TestCase):
    """get_unsqueeze_dim returns 2 iff t's dim-1 equals freqs' seq length."""

    def test_batch_first_returns_two(self):
        t = paddle.zeros([1, 2, 3, 4])
        freqs = paddle.zeros([1, 2, 4])
        self.assertEqual(get_unsqueeze_dim(t, freqs), 2)

    def test_seq_first_returns_one(self):
        # t dim-1 (=1) != freqs dim-1 (=2)
        t = paddle.zeros([2, 1, 3, 4])
        freqs = paddle.zeros([1, 2, 4])
        self.assertEqual(get_unsqueeze_dim(t, freqs), 1)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestApplyRotaryPosEmbBshd(unittest.TestCase):
    """_apply_rotary_pos_emb_bshd numeric contract vs independent reference."""

    def _run(self, t_np, freqs_np, **kwargs):
        out = _apply_rotary_pos_emb_bshd(
            paddle.to_tensor(t_np),
            paddle.to_tensor(freqs_np),
            cos=None,
            sin=None,
            apply_rope_fusion=False,
            **kwargs,
        )
        return out.numpy()

    def test_full_rotation_matches_reference(self):
        rng = np.random.RandomState(0)
        # b=1, s=2, h=1, d=4 (b != s avoids the M-RoPE transpose branch).
        t_np = rng.randn(1, 2, 1, 4).astype(np.float32)
        freqs_np = rng.randn(1, 2, 4).astype(np.float32)
        got = self._run(t_np, freqs_np, rotary_interleaved=False)
        expected = _ref_rope_bshd(t_np, freqs_np, interleaved=False)
        self.assertEqual(list(got.shape), [1, 2, 1, 4])
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

    def test_mscale_scales_cos_and_sin(self):
        rng = np.random.RandomState(1)
        t_np = rng.randn(1, 2, 1, 4).astype(np.float32)
        freqs_np = rng.randn(1, 2, 4).astype(np.float32)
        got = self._run(t_np, freqs_np, mscale=2.0, rotary_interleaved=False)
        expected = _ref_rope_bshd(t_np, freqs_np, mscale=2.0)
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)
        # mscale=2 must genuinely differ from mscale=1 for this input.
        base = _ref_rope_bshd(t_np, freqs_np, mscale=1.0)
        self.assertFalse(np.allclose(expected, base, rtol=1e-3, atol=1e-3))

    def test_mscale_none_treated_as_one(self):
        rng = np.random.RandomState(2)
        t_np = rng.randn(1, 2, 1, 4).astype(np.float32)
        freqs_np = rng.randn(1, 2, 4).astype(np.float32)
        got = self._run(t_np, freqs_np, mscale=None, rotary_interleaved=False)
        expected = _ref_rope_bshd(t_np, freqs_np, mscale=1.0)
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

    def test_partial_rotary_passes_tail_through(self):
        rng = np.random.RandomState(3)
        # d=8 head dim, but freqs only cover the first 4 channels.
        t_np = rng.randn(1, 2, 1, 8).astype(np.float32)
        freqs_np = rng.randn(1, 2, 4).astype(np.float32)
        got = self._run(t_np, freqs_np, rotary_interleaved=False)
        expected = _ref_rope_bshd(t_np, freqs_np)
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)
        # The un-rotated tail (channels 4:) must be passed through verbatim.
        np.testing.assert_allclose(
            got[..., 4:], t_np[..., 4:], rtol=1e-6, atol=1e-6
        )

    def test_interleaved_matches_reference(self):
        rng = np.random.RandomState(4)
        t_np = rng.randn(1, 2, 1, 4).astype(np.float32)
        freqs_np = rng.randn(1, 2, 4).astype(np.float32)
        got = self._run(t_np, freqs_np, rotary_interleaved=True)
        expected = _ref_rope_bshd(t_np, freqs_np, interleaved=True)
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

    def test_multi_latent_attention_reorders_channels(self):
        rng = np.random.RandomState(5)
        t_np = rng.randn(1, 2, 1, 4).astype(np.float32)
        freqs_np = rng.randn(1, 2, 4).astype(np.float32)
        got = self._run(
            t_np,
            freqs_np,
            rotary_interleaved=False,
            multi_latent_attention=True,
        )
        expected = _ref_rope_bshd(t_np, freqs_np, mla=True)
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)
        # MLA de-interleaving must actually change the result here.
        plain = _ref_rope_bshd(t_np, freqs_np, mla=False)
        self.assertFalse(np.allclose(expected, plain, rtol=1e-3, atol=1e-3))


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestApplyRotaryPosEmbRouting(unittest.TestCase):
    """apply_rotary_pos_emb routes to the bshd path when cu_seqlens is None."""

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

    def test_bshd_route_produces_reference_values(self):
        rng = np.random.RandomState(6)
        t_np = rng.randn(1, 2, 1, 4).astype(np.float32)
        freqs_np = rng.randn(1, 2, 4).astype(np.float32)
        out = apply_rotary_pos_emb(
            paddle.to_tensor(t_np),
            paddle.to_tensor(freqs_np),
            cos=None,
            sin=None,
            config=self._config(),
            cu_seqlens=None,
        )
        expected = _ref_rope_bshd(t_np, freqs_np)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestGetPosEmbOnThisCpRank(unittest.TestCase):
    """Context-parallel load-balanced slicing of position embeddings."""

    def test_raises_without_group(self):
        pos_emb = paddle.zeros([8, 3])
        with self.assertRaises(ValueError):
            get_pos_emb_on_this_cp_rank(pos_emb, seq_dim=0, cp_group=None)

    def _select(self, cp_size, cp_rank):
        # 8 sequence positions, 3-wide feature; content identifies each row.
        pos_np = np.arange(8 * 3, dtype=np.float32).reshape(8, 3)
        with (
            patch(f"{MODULE}.get_pg_size", return_value=cp_size),
            patch(f"{MODULE}.get_pg_rank", return_value=cp_rank),
        ):
            out = get_pos_emb_on_this_cp_rank(
                paddle.to_tensor(pos_np), seq_dim=0, cp_group=object()
            )
        return pos_np, out.numpy()

    def test_rank0_takes_first_and_last_segment(self):
        # 2*cp_size=4 segments of length 2: seg0=[0,1] seg3=[6,7].
        # rank 0 -> cp_idx = [0, 2*2-0-1=3] -> rows 0,1,6,7.
        pos_np, out = self._select(cp_size=2, cp_rank=0)
        expected = pos_np[[0, 1, 6, 7], :]
        self.assertEqual(list(out.shape), [4, 3])
        np.testing.assert_array_equal(out, expected)

    def test_rank1_takes_middle_segments(self):
        # rank 1 -> cp_idx = [1, 2*2-1-1=2] -> rows 2,3,4,5.
        pos_np, out = self._select(cp_size=2, cp_rank=1)
        expected = pos_np[[2, 3, 4, 5], :]
        np.testing.assert_array_equal(out, expected)

    def test_ranks_are_disjoint_and_cover_all_positions(self):
        _, r0 = self._select(cp_size=2, cp_rank=0)
        _, r1 = self._select(cp_size=2, cp_rank=1)
        union = np.concatenate([r0, r1], axis=0)
        full = np.arange(8 * 3, dtype=np.float32).reshape(8, 3)
        # Every original row appears exactly once across the two ranks.
        np.testing.assert_array_equal(union[np.argsort(union[:, 0])], full)


if __name__ == "__main__":
    unittest.main()
