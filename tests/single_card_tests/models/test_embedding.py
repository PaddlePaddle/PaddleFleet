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

"""Behavior tests for paddlefleet.models.kimi_k25.embedding.

Model-layer tests designed from the production source. They exercise the
sincos positional-embedding math with independent NumPy references (never the
production function itself as its own oracle), the ``VisionEmbeddingSpec``
dataclass contract, and the ``Learnable2DInterpPosEmbDivided_fixed`` layer
(buffer content, the same-size add path, the multi-frame time-weight path, the
resize branch selection, and the ``t <= num_frames`` guard).

The module imports paddle at load time. The environment used to author this
file has no paddle installed, so every test is gated behind an honest
``skipUnless(IMPORT_OK, ...)`` guard and skips rather than fakes a pass. Only a
genuine ImportError is treated as a missing dependency; runtime API errors are
allowed to surface instead of being swallowed as a skip.
"""

import os
import sys
import unittest

import numpy as np

# Make ``src/`` importable when tests are run from a source checkout.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_SRC, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

IMPORT_OK = True
IMPORT_ERR = ""
try:
    import paddle

    from paddlefleet.models.kimi_k25.embedding import (
        Learnable2DInterpPosEmbDivided_fixed,
        VisionEmbeddingSpec,
        get_1d_sincos_pos_embed,
        get_1d_sincos_pos_embed_from_grid,
    )
except ImportError as exc:  # only genuine missing-dependency, not API errors
    IMPORT_OK = False
    IMPORT_ERR = f"paddle/paddlefleet not importable: {exc}"


def _ref_1d_sincos_from_grid(embed_dim, pos):
    """Independent NumPy reference for get_1d_sincos_pos_embed_from_grid.

    Re-derived from the documented formula, not from the production code:
    omega[i] = 1 / 10000 ** (i / (embed_dim / 2)) for i in [0, embed_dim/2),
    out = outer(pos, omega), emb = concat([sin(out), cos(out)], axis=1).
    Computed in float64 so it can anchor a float32 production output.
    """
    pos = np.asarray(pos, dtype=np.float64).reshape(-1)
    half = embed_dim // 2
    omega = np.arange(half, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / (10000.0**omega)
    out = np.outer(pos, omega)  # (M, half)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)  # (M, embed_dim)


@unittest.skipUnless(IMPORT_OK, IMPORT_ERR or "paddle not installed")
class TestGet1dSincosPosEmbedFromGrid(unittest.TestCase):
    """Numeric contract of get_1d_sincos_pos_embed_from_grid."""

    def test_matches_independent_reference(self):
        embed_dim = 8
        pos_values = [0.0, 1.0, 2.0, 3.0, 4.0]
        pos = paddle.to_tensor(pos_values, dtype=paddle.float32)
        result = get_1d_sincos_pos_embed_from_grid(embed_dim, pos).numpy()
        ref = _ref_1d_sincos_from_grid(embed_dim, pos_values)
        self.assertEqual(list(result.shape), [len(pos_values), embed_dim])
        np.testing.assert_allclose(result, ref, rtol=1e-5, atol=1e-6)

    def test_sin_half_then_cos_half_ordering(self):
        # Guard the concat order: first half is sin(out), second half cos(out).
        embed_dim = 8
        pos_values = [1.0, 2.5, 4.0]
        pos = paddle.to_tensor(pos_values, dtype=paddle.float32)
        result = get_1d_sincos_pos_embed_from_grid(embed_dim, pos).numpy()
        half = embed_dim // 2
        omega = np.arange(half, dtype=np.float64) / (embed_dim / 2.0)
        omega = 1.0 / (10000.0**omega)
        out = np.outer(np.asarray(pos_values, dtype=np.float64), omega)
        np.testing.assert_allclose(
            result[:, :half], np.sin(out), rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            result[:, half:], np.cos(out), rtol=1e-5, atol=1e-6
        )

    def test_position_zero_is_zeros_then_ones(self):
        # sin(0)=0, cos(0)=1 for every frequency -> exact, dtype-independent.
        result = get_1d_sincos_pos_embed_from_grid(
            6, paddle.to_tensor([0.0], dtype=paddle.float32)
        ).numpy()
        np.testing.assert_allclose(
            result[0], np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), atol=1e-6
        )

    def test_flattens_multidim_positions(self):
        # pos.reshape(-1): a (2, 3) grid must flatten row-major to M = 6.
        grid = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        pos = paddle.to_tensor(grid, dtype=paddle.float32)
        result = get_1d_sincos_pos_embed_from_grid(4, pos).numpy()
        ref = _ref_1d_sincos_from_grid(4, np.asarray(grid).reshape(-1))
        self.assertEqual(list(result.shape), [6, 4])
        np.testing.assert_allclose(result, ref, rtol=1e-5, atol=1e-6)

    def test_odd_embed_dim_raises(self):
        pos = paddle.arange(4, dtype=paddle.float32)
        with self.assertRaises(AssertionError):
            get_1d_sincos_pos_embed_from_grid(7, pos)


@unittest.skipUnless(IMPORT_OK, IMPORT_ERR or "paddle not installed")
class TestGet1dSincosPosEmbed(unittest.TestCase):
    """Contract of get_1d_sincos_pos_embed (temporal grid + optional cls row)."""

    def test_without_cls_matches_reference(self):
        embed_dim, t_size = 16, 5
        result = get_1d_sincos_pos_embed(embed_dim, t_size).numpy()
        # Independent reference over arange(t_size); does not call production.
        ref = _ref_1d_sincos_from_grid(embed_dim, np.arange(t_size))
        self.assertEqual(list(result.shape), [t_size, embed_dim])
        np.testing.assert_allclose(result, ref, rtol=1e-5, atol=1e-6)

    def test_with_cls_prepends_zero_row_and_keeps_rest(self):
        embed_dim, t_size = 8, 3
        result = get_1d_sincos_pos_embed(
            embed_dim, t_size, cls_token=True
        ).numpy()
        self.assertEqual(list(result.shape), [t_size + 1, embed_dim])
        # Row 0 is the injected cls zero row.
        np.testing.assert_allclose(result[0], np.zeros(embed_dim), atol=1e-6)
        # Remaining rows are the temporal embedding, unshifted and unmangled.
        ref = _ref_1d_sincos_from_grid(embed_dim, np.arange(t_size))
        np.testing.assert_allclose(result[1:], ref, rtol=1e-5, atol=1e-6)

    def test_cls_adds_exactly_one_row(self):
        embed_dim, t_size = 8, 4
        without = get_1d_sincos_pos_embed(embed_dim, t_size).numpy()
        with_cls = get_1d_sincos_pos_embed(
            embed_dim, t_size, cls_token=True
        ).numpy()
        self.assertEqual(with_cls.shape[0] - without.shape[0], 1)
        # The non-cls rows must equal the plain embedding (no reordering).
        np.testing.assert_allclose(with_cls[1:], without, rtol=1e-5, atol=1e-6)


@unittest.skipUnless(IMPORT_OK, IMPORT_ERR or "paddle not installed")
class TestVisionEmbeddingSpec(unittest.TestCase):
    """VisionEmbeddingSpec dataclass field contract."""

    def test_default_rope_embedding_is_none(self):
        self.assertIsNone(VisionEmbeddingSpec().rope_embedding)

    def test_stores_exact_object_identity(self):
        sentinel = object()
        spec = VisionEmbeddingSpec(rope_embedding=sentinel)
        # Identity, not just truthiness: the stored spec must be the same obj.
        self.assertIs(spec.rope_embedding, sentinel)

    def test_single_declared_field(self):
        import dataclasses

        names = [f.name for f in dataclasses.fields(VisionEmbeddingSpec)]
        self.assertEqual(names, ["rope_embedding"])


@unittest.skipUnless(IMPORT_OK, IMPORT_ERR or "paddle not installed")
class TestLearnable2DInterpPosEmbDivided(unittest.TestCase):
    """Layer construction and forward paths of the divided-fixed pos emb.

    Note on a production defect exercised here (embedding.py:137): the same-size
    fast path is guarded by ``(h, w) == self.weight.shape[:-1]``. ``.shape``
    returns a list while ``(h, w)`` is a tuple, so the equality is always False
    and the fast path is dead code -- the layer always routes through
    ``get_rope_shape``/interpolate. For a matching grid the bicubic resize to
    the identical size is an exact identity, so the numeric contract below still
    holds; the test asserts the intended (weight-flattened) result regardless of
    which branch runs. Production is not modified.
    """

    def _build(self, height=4, width=4, num_frames=2, dim=8):
        return Learnable2DInterpPosEmbDivided_fixed(
            height=height, width=width, num_frames=num_frames, dim=dim
        )

    def test_weight_shape(self):
        emb = self._build(height=4, width=4, num_frames=2, dim=8)
        self.assertEqual(list(emb.weight.shape), [4, 4, 8])

    def test_time_weight_matches_sincos_reference(self):
        # time_weight == get_1d_sincos_pos_embed(dim, num_frames).unsqueeze(1),
        # i.e. shape [num_frames, 1, dim] and content == independent sincos ref.
        num_frames, dim = 3, 8
        emb = self._build(height=4, width=4, num_frames=num_frames, dim=dim)
        tw = emb.time_weight.numpy()
        self.assertEqual(list(tw.shape), [num_frames, 1, dim])
        ref = _ref_1d_sincos_from_grid(dim, np.arange(num_frames))
        np.testing.assert_allclose(tw[:, 0, :], ref, rtol=1e-5, atol=1e-6)

    def test_forward_same_size_single_frame_adds_flattened_weight(self):
        h = w = 4
        dim = 8
        emb = self._build(height=h, width=w, num_frames=1, dim=dim)
        n = h * w
        x = paddle.arange(n * dim, dtype=paddle.float32).reshape([n, dim])
        grid_thws = paddle.to_tensor([[1, h, w]])
        out = emb(x, grid_thws).numpy()
        # Intended result: x + weight flattened over (h, w). weight is the
        # production parameter (state), used only as the additive anchor here.
        weight_flat = emb.weight.numpy().reshape([n, dim])
        expected = x.numpy() + weight_flat
        self.assertEqual(list(out.shape), [n, dim])
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4)

    def test_forward_multi_frame_adds_per_frame_time_weight(self):
        h = w = 4
        dim = 8
        t = 2
        emb = self._build(height=h, width=w, num_frames=t, dim=dim)
        n = h * w
        x = paddle.zeros([t * n, dim], dtype=paddle.float32)
        grid_thws = paddle.to_tensor([[t, h, w]])
        out = emb(x, grid_thws).numpy()
        # Per frame f: rows [f*n:(f+1)*n] == weight_flat + time_weight[f].
        weight_flat = emb.weight.numpy().reshape([n, dim])
        tw = emb.time_weight.numpy()  # [t, 1, dim]
        expected = np.concatenate(
            [weight_flat + tw[f, 0, :] for f in range(t)], axis=0
        )
        self.assertEqual(list(out.shape), [t * n, dim])
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4)
        # The two frames must differ by exactly the time-weight delta, proving
        # per-frame addition (not a duplicated single frame).
        frame0, frame1 = out[:n], out[n:]
        np.testing.assert_allclose(
            frame1 - frame0, tw[1, 0, :] - tw[0, 0, :], rtol=1e-4, atol=1e-4
        )

    def test_forward_resize_branch_uses_requested_grid(self):
        # Requesting a grid different from the stored (4, 4) must route through
        # the resize branch and produce h*w rows for the *requested* size.
        emb = self._build(height=4, width=4, num_frames=1, dim=8)
        h, w = 2, 2
        x = paddle.zeros([h * w, 8], dtype=paddle.float32)
        grid_thws = paddle.to_tensor([[1, h, w]])
        out = emb(x, grid_thws).numpy()
        self.assertEqual(list(out.shape), [h * w, 8])
        # Distinct from the stored-grid row count (16), confirming a real resize.
        self.assertNotEqual(out.shape[0], 4 * 4)

    def test_forward_t_greater_than_num_frames_raises(self):
        emb = self._build(height=4, width=4, num_frames=1, dim=8)
        x = paddle.zeros([2 * 16, 8], dtype=paddle.float32)
        grid_thws = paddle.to_tensor([[2, 4, 4]])  # t=2 > num_frames=1
        with self.assertRaises(AssertionError):
            emb(x, grid_thws)


if __name__ == "__main__":
    unittest.main()
