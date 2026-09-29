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

"""CPU-observable size arithmetic for the sub-512 latent / sub-tile-head path.

Scope. ``csa_sparse_attn`` drives the "cudnn" backend at fixed kernel widths --
64/128 query-head tiles and a 512 latent dim -- and reaches any narrower layer
by zero-padding on the way in and dropping the pad on the way out. That padding
plan is decided by four pure functions that run entirely on CPU:

* ``_dsa_latent_dim``   -- the guard that maps any ``hn <= 512`` to 512 and
                           rejects wider layers (the "switches behaviour when
                           latent < 512" decision).
* ``_pad_latent_dim``   -- zero-pads the last axis up to the kernel width, and
                           stays copy-free (returns the input object) at 512.
* ``_dsa_head_tile``    -- picks the smallest 64/128 head tile that fits, and
                           rejects > 128 heads.
* ``_pad_query_heads``  -- widens the head axis with zero query rows and a
                           ``-1e30`` sink.
* ``_real_rows`` / ``_drop_padded_rows`` -- the ``gcd``-based row-index math
                           that undoes both the head and latent padding in one
                           gather.

Every expected value below is hand-derived from those definitions (or, for the
realistic 512-wide widths, from an independent per-axis slice+gather that does
NOT call the production un-pad). This is a no-accelerator test: it verifies the
layout arithmetic and branch decisions only. The GPU attention numerics of the
FlashMLA forward / cuDNN backward are NOT exercised here -- that path needs a
real device and is covered separately; asserting it on CPU would be dishonest.
"""

import math
import unittest

import numpy as np

try:
    import paddle

    from paddlefleet.fusions.csa_sparse_attn import (
        _DSA_HEAD_TILES,
        _DSA_LATENT_DIM,
        _NEG_SINK,
        _drop_padded_rows,
        _dsa_head_tile,
        _dsa_latent_dim,
        _pad_latent_dim,
        _pad_query_heads,
        _real_rows,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - environment probe
    _IMPORT_ERROR = str(exc)

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    "paddle / paddlefleet.fusions.csa_sparse_attn could not be imported "
    f"({_IMPORT_ERROR})"
)


class _CpuTestCase(unittest.TestCase):
    """Force the default place to CPU (this suite verifies only host-side size
    math) and restore whatever it was, even on assertion failure."""

    def setUp(self):
        self._orig_device = paddle.get_device()
        self.addCleanup(paddle.set_device, self._orig_device)
        paddle.set_device("cpu")


def _per_axis_unpad(x, num_heads, head_tile, hn):
    """Independent reference for ``_drop_padded_rows``: a straight per-axis
    slice of the latent columns followed by a head-row gather.

    Deliberately a different algorithm from the production ``gcd`` row-gather,
    so it can serve as an independent oracle for the realistic 512-wide widths
    where a full literal would be unwieldy. It does NOT call the production
    un-pad or ``_real_rows``.
    """
    if hn != x.shape[-1]:
        x = x[..., :hn]
    if num_heads == head_tile:
        return x
    rows = x.reshape([-1, hn])
    n = rows.shape[0] // head_tile
    idx = (
        (paddle.arange(n) * head_tile).unsqueeze(1)
        + paddle.arange(num_heads).unsqueeze(0)
    ).flatten()
    return paddle.gather(rows, idx.astype("int64"), axis=0).reshape(
        [*x.shape[:-2], num_heads, hn]
    )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestDsaLatentDimGuard(_CpuTestCase):
    """``_dsa_latent_dim`` -- the width guard that switches behaviour at 512."""

    def test_constant_is_512(self):
        # The literal 512 used as the hand-derived expectation below is the
        # documented FlashMLA / cuDNN DSA kernel width.
        self.assertEqual(_DSA_LATENT_DIM, 512)

    def test_maps_any_sub_512_width_to_the_kernel_width(self):
        for hn in (1, 32, 64, 127, 128, 256, 384, 511, 512):
            with self.subTest(hn=hn):
                self.assertEqual(_dsa_latent_dim(hn), 512)

    def test_rejects_wider_than_512(self):
        for hn in (513, 576, 1024):
            with self.subTest(hn=hn):
                with self.assertRaisesRegex(
                    ValueError, r"at most 512 latent dims"
                ) as cm:
                    _dsa_latent_dim(hn)
                # The message must name the offending width, not a fixed one.
                self.assertIn(f"got {hn}", str(cm.exception))


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestPadLatentDim(_CpuTestCase):
    """``_pad_latent_dim`` -- zero-pad the latent axis; copy-free at width."""

    def test_pads_sub_width_with_zeros_and_keeps_prefix_exactly(self):
        # Distinct, exactly-representable content so a mis-copied or reordered
        # prefix, or a non-zero pad, is visible.
        x = paddle.arange(2 * 3 * 4, dtype="float32").reshape([2, 3, 4])
        out = _pad_latent_dim(x, 8)
        self.assertEqual(list(out.shape), [2, 3, 8])
        self.assertEqual(out.dtype, x.dtype)
        # First 4 columns are the input untouched; last 4 are exactly zero.
        np.testing.assert_array_equal(out[..., :4].numpy(), x.numpy())
        np.testing.assert_array_equal(
            out[..., 4:].numpy(), np.zeros([2, 3, 4], dtype=np.float32)
        )

    def test_pads_realistic_256_to_512(self):
        x = paddle.arange(1 * 5 * 256, dtype="float32").reshape([1, 5, 256])
        out = _pad_latent_dim(x, _DSA_LATENT_DIM)
        self.assertEqual(list(out.shape), [1, 5, 512])
        np.testing.assert_array_equal(out[..., :256].numpy(), x.numpy())
        self.assertEqual(float(out[..., 256:].abs().max()), 0.0)

    def test_native_width_returns_the_same_object(self):
        # pad == 0 branch: no allocation, so the wrapper's hn==512 path stays
        # bit-for-bit unchanged. Identity, not just equality.
        for width in (8, _DSA_LATENT_DIM):
            with self.subTest(width=width):
                x = paddle.arange(2 * width, dtype="float32").reshape(
                    [2, width]
                )
                self.assertIs(_pad_latent_dim(x, width), x)

    def test_does_not_hardcode_dtype(self):
        for dtype in ("float32", "float64"):
            with self.subTest(dtype=dtype):
                x = paddle.ones([1, 2, 4], dtype=dtype)
                out = _pad_latent_dim(x, 6)
                self.assertEqual(out.dtype, x.dtype)
                self.assertEqual(list(out.shape), [1, 2, 6])
                np.testing.assert_array_equal(
                    out[..., 4:].numpy(),
                    np.zeros([1, 2, 2], dtype=np.dtype(dtype)),
                )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestDsaHeadTile(_CpuTestCase):
    """``_dsa_head_tile`` -- smallest 64/128 head tile that fits."""

    def test_tiles_are_64_and_128(self):
        self.assertEqual(_DSA_HEAD_TILES, (64, 128))

    def test_selects_smallest_fitting_tile(self):
        # Hand-derived from ``num_heads <= tile`` over (64, 128).
        expected = {1: 64, 24: 64, 63: 64, 64: 64, 65: 128, 100: 128, 128: 128}
        for num_heads, tile in expected.items():
            with self.subTest(num_heads=num_heads):
                self.assertEqual(_dsa_head_tile(num_heads), tile)

    def test_rejects_more_than_max_tile(self):
        for num_heads in (129, 200):
            with self.subTest(num_heads=num_heads):
                with self.assertRaisesRegex(
                    ValueError, r"at most 128 query heads"
                ) as cm:
                    _dsa_head_tile(num_heads)
                self.assertIn(f"got {num_heads}", str(cm.exception))


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestPadQueryHeads(_CpuTestCase):
    """``_pad_query_heads`` -- zero query rows + ``-1e30`` sink for pad heads."""

    def test_neg_sink_constant(self):
        self.assertEqual(_NEG_SINK, -1e30)

    def test_widens_query_and_sink(self):
        # b=1, sq=2, h=2 real heads, hn=3; pad up to a tile of 4 heads.
        query = paddle.arange(1 * 2 * 2 * 3, dtype="float32").reshape(
            [1, 2, 2, 3]
        )
        attn_sink = paddle.to_tensor([10.0, -5.0], dtype="float32")
        q_out, sink_out = _pad_query_heads(query, attn_sink, head_tile=4)

        self.assertEqual(list(q_out.shape), [1, 2, 4, 3])
        # Real heads unchanged, the 2 pad heads are exactly zero.
        np.testing.assert_array_equal(q_out[:, :, :2, :].numpy(), query.numpy())
        np.testing.assert_array_equal(
            q_out[:, :, 2:, :].numpy(),
            np.zeros([1, 2, 2, 3], dtype=np.float32),
        )

        self.assertEqual(list(sink_out.shape), [4])
        self.assertEqual(sink_out.dtype, paddle.float32)
        np.testing.assert_array_equal(
            sink_out[:2].numpy(), np.array([10.0, -5.0], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            sink_out[2:].numpy(), np.full([2], np.float32(-1e30))
        )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestRealRows(_CpuTestCase):
    """``_real_rows`` -- row ids of the real data in the padded kernel view."""

    def test_hand_derived_row_ids(self):
        # (num_tokens, num_heads, head_tile, keep, total) -> expected flat ids.
        # Derived directly from
        #   token*(head_tile*total) + head*total + chunk, over the first
        #   num_heads heads and first ``keep`` chunks of each token.
        cases = {
            # 2 tokens, 2 of 3 heads, 1 of 2 chunks kept:
            #   t0: h0c0=0, h1c0=2 ; t1 base=6: h0c0=6, h1c0=8
            (2, 2, 3, 1, 2): [0, 2, 6, 8],
            # 1 token, 2 of 3 heads, 2 of 2 chunks:
            #   h0: 0,1 ; h1: 2,3
            (1, 2, 3, 2, 2): [0, 1, 2, 3],
            # 1 token, 1 head, 2 chunks: 0,1
            (1, 1, 2, 2, 2): [0, 1],
            # no padding (head_tile==heads, total==keep==1): identity 0..3
            (2, 2, 2, 1, 1): [0, 1, 2, 3],
        }
        for args, expected in cases.items():
            with self.subTest(args=args):
                self.assertEqual(_real_rows(*args).tolist(), expected)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestDropPaddedRows(_CpuTestCase):
    """``_drop_padded_rows`` -- fused head-row + latent-column un-pad."""

    def test_copy_free_when_nothing_is_padded(self):
        x = paddle.arange(3 * 2 * 4, dtype="float32").reshape([3, 2, 4])
        # num_heads == head_tile and hn == kernel_hn -> input returned as-is.
        self.assertIs(_drop_padded_rows(x, 2, 2, 4), x)

    def test_latent_only_drop_hand_derived(self):
        # head_tile == num_heads == 2, kernel_hn 4 -> keep first hn=2 columns.
        x = paddle.to_tensor(
            [[[0, 1, 2, 3], [4, 5, 6, 7]]], dtype="float32"
        )  # shape [1, 2, 4]
        out = _drop_padded_rows(x, num_heads=2, head_tile=2, hn=2)
        self.assertEqual(list(out.shape), [1, 2, 2])
        np.testing.assert_array_equal(
            out.numpy(), np.array([[[0, 1], [4, 5]]], dtype=np.float32)
        )

    def test_head_and_latent_drop_hand_derived_3d(self):
        # 3-D as the backward's dq is: [N=2, head_tile=3, kernel_hn=4],
        # drop to num_heads=2, hn=2. Head row 2 of each token is dropped, and
        # only the first 2 latent columns of the kept heads survive.
        x = paddle.arange(2 * 3 * 4, dtype="float32").reshape([2, 3, 4])
        out = _drop_padded_rows(x, num_heads=2, head_tile=3, hn=2)
        self.assertEqual(list(out.shape), [2, 2, 2])
        np.testing.assert_array_equal(
            out.numpy(),
            np.array(
                [[[0, 1], [4, 5]], [[12, 13], [16, 17]]], dtype=np.float32
            ),
        )

    def test_matches_independent_per_axis_reference(self):
        # Realistic kernel width (512) at both divisible and non-divisible
        # latents, with and without head padding, in the 4-D forward layout and
        # the 3-D backward layout. Compared against a per-axis slice+gather
        # oracle (a different algorithm from the gcd row-gather).
        cases = [
            (64, 64, 256, 512),  # latent only, 512/256 -> one 256-row of two
            (64, 64, 384, 512),  # latent only, 512/384 -> three 128-rows of 4
            (24, 64, 256, 512),  # head + latent (ernielite exp2 shape)
            (32, 64, 256, 512),  # head + latent
        ]
        for num_heads, head_tile, hn, kernel_hn in cases:
            g = math.gcd(hn, kernel_hn)
            for shape in (
                [3, head_tile, kernel_hn],
                [1, 2, head_tile, kernel_hn],
            ):
                with self.subTest(heads=num_heads, hn=hn, ndim=len(shape)):
                    x = paddle.arange(
                        int(np.prod(shape)), dtype="float32"
                    ).reshape(shape)
                    got = _drop_padded_rows(x, num_heads, head_tile, hn)
                    ref = _per_axis_unpad(x, num_heads, head_tile, hn)
                    self.assertEqual(tuple(got.shape), tuple(ref.shape))
                    self.assertEqual(list(got.shape[-2:]), [num_heads, hn])
                    np.testing.assert_array_equal(got.numpy(), ref.numpy())
                    # Sanity-check the documented chunking: keeping ``hn``
                    # columns is expressible as whole ``gcd``-wide rows.
                    self.assertEqual(hn % g, 0)
                    self.assertEqual(kernel_hn % g, 0)


if __name__ == "__main__":
    unittest.main()
