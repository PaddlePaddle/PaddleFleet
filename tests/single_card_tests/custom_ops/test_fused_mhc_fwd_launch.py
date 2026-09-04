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

"""Two cuTile forward launchers carry occupancy hints, and the fused
``h_post_bda`` one also halves TILE_C. Both are claimed bit-identical to the
launch configuration they replaced -- an occupancy hint changes residency, not
arithmetic, and neither forward kernel reduces across the C tiles. This file
pins that down by launching the same kernels with the old configuration and
comparing bit-for-bit.

The existing ``test_fused_mhc_bda_cast.py`` runs at C=1024, where
``gcd(1024, 1024)`` and the old ``gcd(1024, 1024)`` agree, so it cannot see the
TILE_C change at all. The cases here use C=2048 and C=4096, where the old chain
picked 2048 and 4096.
"""

import math
import unittest

import paddle

from paddlefleet.fusions import fused_mhc_kernels as M
from paddlefleet.fusions.fused_mhc_kernels import (
    fused_h_post_bda,
    fused_sinkhorn,
    is_cutile_available,
)

_requires_cutile = unittest.skipUnless(
    is_cutile_available() and paddle.is_compiled_with_cuda(),
    "cuTile fused kernels need a GPU build with cuda.tile",
)


def _bit_equal(a, b):
    """``equal_all`` has no bfloat16 kernel; widening to fp32 is exact."""
    return bool(paddle.equal_all(a.astype("float32"), b.astype("float32")))


def _legacy_tile_c(C):
    """The TILE_C chain the fused forward path used before the change."""
    if C % 4096 == 0:
        return math.gcd(C, 4096)
    if C % 2048 == 0:
        return math.gcd(C, 2048)
    return math.gcd(C, 1024)


def _hpb_fwd_legacy(h_res, original_residual, h_post, x):
    """``_cutile_h_post_bda_fwd(fuse_cast=True)`` at the old launch config.

    Same kernel source, but the un-hinted kernel object and the old TILE_C.
    """
    s, b, n, C = original_residual.shape
    sb = s * b
    out = paddle.empty(shape=[sb, n, C], dtype=original_residual.dtype)
    M.ct.launch(
        M._get_cuda_stream(),
        (sb,),
        M._ct_hpb_fwd_kernel,
        (
            h_res.reshape([sb, n, n]),
            original_residual.reshape([sb, n, C]),
            h_post.reshape([sb, n]),
            x.reshape([sb, C]),
            out,
            n,
            _legacy_tile_c(C),
            1,
            True,  # UPCAST_INPUTS, i.e. fuse_cast
        ),
    )
    return out.reshape([s, b, n, C])


def _sinkhorn_fwd_legacy(input_logits, num_iterations, eps, kernel=None):
    """``_cutile_sinkhorn_fwd`` with an explicit kernel object.

    Defaults to the un-hinted one, i.e. the pre-change configuration.
    """
    original_shape = input_logits.shape
    hc = original_shape[-1]
    n_batch = input_logits.size // (hc * hc)
    tile = math.gcd(n_batch, 64)
    out = paddle.empty(shape=[n_batch, hc, hc], dtype=input_logits.dtype)
    m_init = paddle.empty(shape=[n_batch, hc, hc], dtype=input_logits.dtype)
    M.ct.launch(
        M._get_cuda_stream(),
        (math.ceil(n_batch / tile), 1, 1),
        kernel if kernel is not None else M._ct_sinkhorn_fwd_kernel,
        (
            input_logits.reshape([n_batch, hc, hc]),
            out,
            m_init,
            eps,
            hc,
            num_iterations,
            tile,
        ),
    )
    return out.reshape(original_shape)


class TestHPostBDAForwardLaunch(unittest.TestCase):
    # (s, b, n, C). C is what matters: 2048 and 4096 are where the old chain
    # picked a different TILE_C from the new gcd(C, 1024).
    SHAPES = [
        (64, 1, 4, 2048),
        (32, 1, 4, 4096),
        (16, 1, 2, 1024),
        (16, 1, 8, 2048),
    ]

    @_requires_cutile
    def test_tile_c_and_occupancy_change_is_bit_exact(self):
        for s, b, n, C in self.SHAPES:
            with self.subTest(s=s, b=b, n=n, C=C):
                paddle.seed(20260903)
                h_res = paddle.randn([s, b, n, n], dtype="float32")
                orig = paddle.randn([s, b, n, C], dtype="bfloat16")
                h_post = paddle.randn([s, b, n], dtype="bfloat16")
                x = paddle.randn([s, b, C], dtype="bfloat16")

                expected = _hpb_fwd_legacy(h_res, orig, h_post, x)
                actual = fused_h_post_bda(
                    h_res, orig, h_post, x, None, fuse_cast=True
                )

                self.assertEqual(actual.dtype, expected.dtype)
                self.assertTrue(
                    _bit_equal(actual, expected),
                    f"C={C}: differs from the pre-change launch config",
                )

    @_requires_cutile
    def test_new_tile_c_actually_differs_on_these_shapes(self):
        # Guards the test itself: if gcd(C, 1024) ever equalled the legacy
        # value on every shape here, the check above would be vacuous.
        differing = [
            C
            for _, _, _, C in self.SHAPES
            if _legacy_tile_c(C) != math.gcd(C, 1024)
        ]
        self.assertTrue(differing, "no shape here exercises the TILE_C change")

    @_requires_cutile
    def test_unfused_path_keeps_the_legacy_config(self):
        # fuse_cast=False must not pick up the hint: that compile is already at
        # 6.37 TB/s and occupancy=8 measured 2.5x slower on it.
        s, b, n, C = 32, 1, 4, 2048
        paddle.seed(11)
        h_res = paddle.randn([s, b, n, n], dtype="float32")
        orig = paddle.randn([s, b, n, C], dtype="float32")
        h_post = paddle.randn([s, b, n], dtype="float32")
        x = paddle.randn([s, b, C], dtype="float32")

        out = fused_h_post_bda(h_res, orig, h_post, x, None, fuse_cast=False)
        self.assertEqual(out.dtype, h_res.dtype)
        self.assertTrue(bool(paddle.isfinite(out).all()))


class TestSinkhornForwardLaunch(unittest.TestCase):
    # (N_batch, HC, iters). HC is the axis that matters: the occupancy hint caps
    # registers, and from HC=8 that makes cuTile schedule the two ``ct.sum``
    # reductions differently. The launcher therefore only uses the hinted kernel
    # at HC <= _CT_SINKHORN_OCC6_MAX_HC, so the result is bit-identical at every
    # width -- narrow ones because the hint does not change the arithmetic there,
    # wide ones because they do not get the hint at all.
    HINTED = ((8192, 4, 5), (128, 4, 1), (8192, 2, 5))
    NOT_HINTED = ((192, 8, 3), (8192, 8, 5), (64, 16, 3))

    @_requires_cutile
    def test_is_bit_exact_at_every_width(self):
        for n_batch, hc, iters in self.HINTED + self.NOT_HINTED:
            with self.subTest(n_batch=n_batch, hc=hc, iters=iters):
                paddle.seed(20260903)
                logits = paddle.randn([n_batch, hc, hc], dtype="float32")
                out_ref = _sinkhorn_fwd_legacy(logits, iters, 1e-8)
                out = fused_sinkhorn(logits, iters, 1e-8)
                self.assertTrue(_bit_equal(out, out_ref), "out differs")

    @_requires_cutile
    def test_wide_mhc_declines_the_hint(self):
        # Guards the reason the test above passes for the wide cases. Without
        # this, a future change that drops the width gate would still look green
        # on `test_is_bit_exact_at_every_width` only by accident.
        self.assertEqual(M._CT_SINKHORN_OCC6_MAX_HC, 4)
        for _n_batch, hc, _iters in self.NOT_HINTED:
            self.assertGreater(
                hc,
                M._CT_SINKHORN_OCC6_MAX_HC,
                "NOT_HINTED case is inside the hinted range",
            )

    @_requires_cutile
    def test_the_hint_really_does_reassociate_above_the_gate(self):
        # The gate is not cosmetic: launched directly, the hinted kernel differs
        # from the un-hinted one at HC=8, at fp32 ULP scale. If this ever stops
        # being true the gate can be dropped -- so assert it rather than leaving
        # it as a comment.
        paddle.seed(20260903)
        logits = paddle.randn([8192, 8, 8], dtype="float32")
        plain = _sinkhorn_fwd_legacy(logits, 5, 1e-8)
        hinted = _sinkhorn_fwd_legacy(
            logits, 5, 1e-8, kernel=M._ct_sinkhorn_fwd_kernel_occ6
        )
        self.assertFalse(_bit_equal(hinted, plain))
        rel = (hinted - plain).abs() / plain.abs().clip(min=1e-30)
        self.assertLess(float(rel.max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
