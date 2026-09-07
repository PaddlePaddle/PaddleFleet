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

"""Numeric record for the sglang-aligned mHC forwards (``align_sglang=True``).

Two tilelang forwards reproduce sglang's reduction order and FMA contraction so
a training run can be bit-compared against inference; the backward stays cuTile
for both. What the tests pin down:

  - each aligned forward reproduces, bit for bit, what sglang's own kernels
    wrote for one fixed input (``TestSglangGoldenBits``);
  - each aligned forward computes the same *function* as cuTile: bitwise equal
    under exact arithmetic, fp32-ULP apart otherwise;
  - the shared cuTile backward still works and is unaffected by which forward
    ran (only the saved ``norm`` carries the align-order rounding);
  - ``align_sglang=False`` leaves the original path bit-for-bit untouched;
  - unsupported N/K/dtype fall back to cuTile instead of asserting, so turning
    the bit-compare on cannot crash a model that trains fine.

Shapes match production: n=4, C=4096, so proj_rms reduces over K=16384.
"""

import os
import unittest
from unittest import mock

import numpy as np
import paddle

from paddlefleet.fusions.fused_mhc_kernels import (
    align_h_post_bda_unsupported,
    align_proj_rms_unsupported,
    is_cutile_available,
)

_S, _B, _N, _C = 4, 2, 4, 4096
_K = _N * _C
_NMAP = _N * _N + 2 * _N


def _rand(*shape, dtype="float32"):
    return paddle.randn(list(shape), dtype="float32").astype(dtype)


def _small_int(*shape, dtype="float32"):
    """Integers in [-8, 8], exact in bf16 and in the mma's tfloat32.

    Every product and partial sum stays a whole number well under 2**24, so
    summation is exact whatever order it happens in -- which is what makes
    bitwise equality between the two accumulation orders a fair demand.
    """
    return paddle.randint(-8, 9, list(shape)).astype("float32").astype(dtype)


def _cmp(ref, got):
    """(bitwise, max_abs_diff, mean_abs_of_ref, relative)."""
    a, b = ref.astype("float32"), got.astype("float32")
    diff = float((a - b).abs().max())
    scale = float(a.abs().mean())
    return bool((a == b).all()), diff, scale, (diff / scale if scale else 0.0)


class _AlignCase(unittest.TestCase):
    """Shared check: run cuTile vs aligned, print deltas, assert tolerances."""

    # subclasses fill these in
    TOL: dict = {}

    def _run(self, align, integer=False):  # pragma: no cover
        raise NotImplementedError

    def _check(self, integer, tol_override=None):
        ref = self._run(align=False, integer=integer)
        got = self._run(align=True, integer=integer)
        tag = "exact" if integer else "rand "
        for key, tol in (tol_override or self.TOL).items():
            bitwise, diff, scale, rel = _cmp(ref[key], got[key])
            print(
                f"  [{tag}] {key:<6} bitwise={bitwise!s:<5} "
                f"max_abs={diff:.3e} scale={scale:.3e} rel={rel:.2e} "
                f"tol={tol:.0e}"
            )
            if tol == 0.0:
                self.assertTrue(
                    bitwise,
                    f"{key} must be bitwise identical, rel={rel:.3e}",
                )
            else:
                self.assertLess(
                    rel, tol, f"{key} drifted past tolerance: rel={rel:.3e}"
                )


@unittest.skipUnless(is_cutile_available(), "cuTile not available")
class TestProjRmsAlign(_AlignCase):
    """fused_proj_rms: sglang chain-accumulator order vs cuTile pairwise tree."""

    # Measured: the two orders agree to fp32 reassociation error. ``proj`` is
    # bitwise equal (the GEMM tiling is unchanged); ``r`` moves 2.1e-06
    # relative, i.e. sqrt(K)*fp32_eps over the K=16384 sum_sq chain and three
    # orders below bf16 resolution. The gradients came out bitwise equal even
    # so, because both leaves are bf16 and round that ULP away; the tolerance
    # stays non-zero because that is luck, not a guarantee.
    TOL = {"proj": 0.0, "r": 1e-5, "g_x": 1e-5, "g_w": 1e-5}

    def _run(self, align, integer=False):
        from paddlefleet.fusions.fused_mhc_kernels import fused_proj_rms

        paddle.seed(11)
        mk = _small_int if integer else _rand
        # The align path only takes bf16 x, which is what fuse_cast produces.
        x_leaf = mk(_S, _B, _K, dtype="bfloat16")
        w_leaf = mk(_K, _NMAP, dtype="bfloat16")
        for t in (x_leaf, w_leaf):
            t.stop_gradient = False

        proj, r = fused_proj_rms(
            x_leaf, w_leaf, 1e-6, fuse_cast=True, align_sglang=align
        )
        # weight both outputs so neither gradient path is left untested
        (proj.astype("float32").sum() + r.astype("float32").sum()).backward()
        return {
            "proj": proj,
            "r": r,
            "g_x": x_leaf.grad,
            "g_w": w_leaf.grad,
            "_proj_dtype": proj.dtype,
            "_r_dtype": r.dtype,
            "_g_x_dtype": x_leaf.grad.dtype,
        }

    def test_exact_arithmetic_is_bitwise(self):
        """Removes reduction order as an explanation for any difference."""
        self._check(integer=True, tol_override=dict.fromkeys(self.TOL, 0.0))

    def test_precision(self):
        self._check(integer=False)

    def test_backward_runs_and_dtypes_hold(self):
        got = self._run(align=True)
        # proj / r stay fp32: _compute_h builds h_res / h_post from them.
        self.assertEqual(got["_proj_dtype"], paddle.float32)
        self.assertEqual(got["_r_dtype"], paddle.float32)
        # the shared cuTile backward reached the bf16 leaves
        self.assertEqual(got["_g_x_dtype"], paddle.bfloat16)
        self.assertIsNotNone(got["g_w"])


@unittest.skipUnless(is_cutile_available(), "cuTile not available")
class TestHPostBDAAlign(_AlignCase):
    """fused_h_post_bda: sglang's FMA contraction vs cuTile's."""

    # Measured: the gradients are bitwise equal -- the backward is bilinear and
    # recomputes from the saved inputs, so it cannot see the forward's
    # contraction choice. The forward is not: cuTile rounds ``post*x`` while
    # ptxas rounds ``comb[0]*res[0]``, ~1 fp32 ULP apart, which the bf16 store
    # turns into a 1-LSB difference on a rounding tie -- 1 element in 131072
    # here (3.0e-04 relative), see test_forward_differs_by_at_most_one_bf16_lsb.
    TOL = {
        "out": 1e-3,
        "g_res": 0.0,
        "g_x": 0.0,
        "g_hres": 0.0,
        "g_hpost": 0.0,
    }

    def _run(self, align, integer=False):
        from paddlefleet.fusions.fused_mhc_kernels import fused_h_post_bda

        paddle.seed(13)
        mk = _small_int if integer else _rand
        # h_res / h_post are fp32 either way; residual / x are bf16 leaves.
        h_res = mk(_S, _B, _N, _N)
        h_post = mk(_S, _B, _N)
        res_leaf = mk(_S, _B, _N, _C, dtype="bfloat16")
        x_leaf = mk(_S, _B, _C, dtype="bfloat16")
        for t in (h_res, h_post, res_leaf, x_leaf):
            t.stop_gradient = False

        out = fused_h_post_bda(
            h_res,
            res_leaf,
            h_post,
            x_leaf,
            None,
            fuse_cast=True,
            align_sglang=align,
        )
        out.astype("float32").sum().backward()
        return {
            "out": out,
            "g_res": res_leaf.grad,
            "g_x": x_leaf.grad,
            "g_hres": h_res.grad,
            "g_hpost": h_post.grad,
            "_out_dtype": out.dtype,
        }

    def test_exact_arithmetic_is_bitwise(self):
        self._check(integer=True, tol_override=dict.fromkeys(self.TOL, 0.0))

    def test_precision(self):
        self._check(integer=False)

    def test_forward_differs_by_at_most_one_bf16_lsb(self):
        """The contraction choice may only cost a tie, never a real value.

        A relative bound cannot say that. bf16 patterns are monotonic in
        magnitude and the diffs are far too small to cross zero, so reading
        both outputs as int16 turns "1 ULP" into "the pattern moved by 1".
        """
        ref = self._run(align=False)["out"].view("int16").astype("int32")
        got = self._run(align=True)["out"].view("int16").astype("int32")
        worst = int((ref - got).abs().max())
        differing = int((ref != got).astype("int32").sum())
        print(
            f"  [lsb ] worst={worst} differing={differing}/{ref.size} "
            f"({100.0 * differing / ref.size:.2f}%)"
        )
        self.assertLessEqual(
            worst, 1, "the aligned forward is off by more than one bf16 LSB"
        )

    def test_backward_runs_and_dtypes_hold(self):
        got = self._run(align=True)
        # output follows the residual dtype, as in the cuTile path
        self.assertEqual(got["_out_dtype"], paddle.bfloat16)
        for key in ("g_res", "g_x", "g_hres", "g_hpost"):
            self.assertIsNotNone(got[key], f"{key} was not produced")


# One fixed input, and the numbers sglang's own kernels produced from it:
# ernie_lite/sglang0901 @ dd94e2e63de6, kernels/ops/layernorm/mhc.py,
# mhc_pre_gemm_sqrsum_tilelang(N=8, K=256) and mhc_post(n=2, C=8), run with its
# eight non-arithmetic deps stubbed out. num_tokens is 32 because the pre kernel
# stores a whole token_block=32 with no bounds guard; every token is fed the
# same row, so one golden row covers all 32.
_GT, _GK, _GN = 32, 256, 8
_GPN, _GPC = 2, 8

_GOLDEN_PROJ_ROW = [
    146.08734130859375,
    286.9250793457031,
    146.54324340820312,
    29.997783660888672,
    -54.35427474975586,
    -110.40202331542969,
    -139.2927703857422,
    -139.20892333984375,
]
_GOLDEN_SQRSUM = 287.0531005859375
_GOLDEN_POST_OUT = [
    0.000217437744140625, -2.25, 2.0, -0.66796875,
    -1.296875, 1.3359375, -1.3359375, 3.34375,
    0.88671875, -0.9296875, -1.109375, 1.5546875,
    -3.515625, -0.4453125, 2.21875, -2.4375,
]  # fmt: skip


def _fixed(shape, salt):
    """The fixed input, as a formula rather than a dump of numbers.

    fp64 integer arithmetic and division are correctly rounded everywhere, so
    torch (sglang's side) and paddle rebuild the same bits from this. The /3
    tail keeps the values off any exact grid, which is what makes the
    accumulation order visible in the first place.
    """
    k = np.arange(int(np.prod(shape)), dtype=np.float64)
    v = (((k * 7 + salt) % 11) - 5 + 1 / 3) / 3
    return paddle.to_tensor(v.reshape(shape).astype("float32")).cuda()


@unittest.skipUnless(is_cutile_available(), "cuTile not available")
class TestSglangGoldenBits(unittest.TestCase):
    """The aligned forwards against bits sglang itself produced.

    This is the contract the align path exists for; the tests above compare
    against cuTile, which cannot show sglang's order was reproduced.
    """

    def _same_bits(self, tag, ref, got):
        bitwise, diff, _, rel = _cmp(ref, got)
        print(f"  [gold] {tag:<8} bitwise={bitwise!s:<5} max_abs={diff:.3e}")
        self.assertTrue(bitwise, f"{tag} left the sglang golden: rel={rel:.3e}")

    def test_proj_rms_forward(self):
        from paddlefleet.fusions.fused_mhc_kernels import (
            _tilelang_proj_rms_fwd_align,
        )

        x = paddle.tile(_fixed([1, _GK], 1), [_GT, 1]).astype("bfloat16")
        proj, norm, _ = _tilelang_proj_rms_fwd_align(
            x, _fixed([_GN, _GK], 2), 1e-6
        )
        self._same_bits(
            "proj",
            paddle.to_tensor([_GOLDEN_PROJ_ROW] * _GT, dtype="float32"),
            proj,
        )
        # sglang hands back the raw sum of squares; this kernel keeps sqrt(sum)
        # for the backward. fp32 sqrt is correctly rounded on both sides, so
        # taking it here keeps this a bit comparison.
        self._same_bits(
            "norm",
            paddle.sqrt(paddle.full([_GT, 1], _GOLDEN_SQRSUM, dtype="float32")),
            norm,
        )

    def test_h_post_bda_forward(self):
        from paddlefleet.fusions.fused_mhc_kernels import (
            _tilelang_h_post_bda_fwd_align,
        )

        out = _tilelang_h_post_bda_fwd_align(
            _fixed([1, 1, _GPN, _GPN], 6),
            _fixed([1, 1, _GPN, _GPC], 4).astype("bfloat16"),
            _fixed([1, 1, _GPN], 5),
            _fixed([1, 1, _GPC], 3).astype("bfloat16"),
        )
        ref = paddle.to_tensor(_GOLDEN_POST_OUT, dtype="float32")
        self._same_bits("post_out", ref.reshape([1, 1, _GPN, _GPC]), out)


class TestAlignSupportPredicates(unittest.TestCase):
    """The support predicates, which decide fall back vs run (no GPU needed)."""

    def test_proj_rms_supported_production_shape(self):
        x = paddle.zeros([2, _K], dtype="bfloat16")
        w = paddle.zeros([_NMAP, _K], dtype="float32")
        self.assertEqual(align_proj_rms_unsupported(x, w), "")

    def test_proj_rms_rejects_wide_n(self):
        # num_residual_streams=8 is inside the public config range and gives
        # N = n*n + 2*n = 80, past the 32-wide GEMM tile.
        x = paddle.zeros([2, 8 * _C], dtype="bfloat16")
        w = paddle.zeros([80, 8 * _C], dtype="float32")
        self.assertIn("N=80", align_proj_rms_unsupported(x, w))

    def test_proj_rms_rejects_unaligned_k(self):
        x = paddle.zeros([2, _K + 128], dtype="bfloat16")
        w = paddle.zeros([_NMAP, _K + 128], dtype="float32")
        self.assertIn("not a multiple of 256", align_proj_rms_unsupported(x, w))

    def test_proj_rms_rejects_non_bf16_x(self):
        # fuse_cast only promises "x may be narrow", not "x is bf16".
        x = paddle.zeros([2, _K], dtype="float32")
        w = paddle.zeros([_NMAP, _K], dtype="float32")
        self.assertIn("only takes bf16", align_proj_rms_unsupported(x, w))

    def _bda_args(self, res_dtype="bfloat16", x_dtype="bfloat16"):
        return (
            paddle.zeros([1, 1, _N, _N], dtype="float32"),
            paddle.zeros([1, 1, _N, _C], dtype=res_dtype),
            paddle.zeros([1, 1, _N], dtype="float32"),
            paddle.zeros([1, 1, _C], dtype=x_dtype),
        )

    def test_h_post_bda_supported_production_shape(self):
        a = self._bda_args()
        self.assertEqual(align_h_post_bda_unsupported(*a, None, True), "")

    def test_h_post_bda_rejects_bias_and_no_fuse_cast(self):
        a = self._bda_args()
        bias = paddle.zeros([_C], dtype="float32")
        self.assertIn("bias", align_h_post_bda_unsupported(*a, bias, True))
        self.assertIn(
            "fuse_cast", align_h_post_bda_unsupported(*a, None, False)
        )

    def test_h_post_bda_rejects_non_bf16(self):
        a = self._bda_args(res_dtype="float32", x_dtype="float32")
        self.assertIn(
            "only takes bf16", align_h_post_bda_unsupported(*a, None, True)
        )


@unittest.skipUnless(is_cutile_available(), "cuTile not available")
class TestAlignFallbackIsCutile(unittest.TestCase):
    """An unsupported shape must produce the cuTile result, not an exception."""

    def test_proj_rms_n8_falls_back(self):
        from paddlefleet.fusions.fused_mhc_kernels import fused_proj_rms

        n = 8
        k, nmap = n * _C, n * n + 2 * n

        def run(align):
            paddle.seed(17)
            x = _small_int(_S, _B, k, dtype="bfloat16")
            w = _small_int(k, nmap, dtype="bfloat16")
            proj, r = fused_proj_rms(
                x, w, 1e-6, fuse_cast=True, align_sglang=align
            )
            return proj, r

        for ref, got in zip(run(False), run(True)):
            self.assertTrue(_cmp(ref, got)[0], "fallback diverged from cuTile")

    def test_h_post_bda_with_bias_falls_back(self):
        from paddlefleet.fusions.fused_mhc_kernels import fused_h_post_bda

        def run(align):
            paddle.seed(19)
            return fused_h_post_bda(
                _small_int(_S, _B, _N, _N),
                _small_int(_S, _B, _N, _C, dtype="bfloat16"),
                _small_int(_S, _B, _N),
                _small_int(_S, _B, _C, dtype="bfloat16"),
                _small_int(_C, dtype="bfloat16"),
                fuse_cast=False,
                align_sglang=align,
            )

        self.assertTrue(
            _cmp(run(False), run(True))[0], "fallback diverged from cuTile"
        )


@unittest.skipUnless(is_cutile_available(), "cuTile not available")
class TestAlignOffIsUntouched(unittest.TestCase):
    """``align_sglang=False`` must be the pre-existing call, bit for bit."""

    def setUp(self):
        # Pin the env gate off: the default is now "ask the environment", so
        # an exported ABLATION_INSPECT_TENSOR would otherwise skew ``run()``.
        patcher = mock.patch.dict(os.environ, {"ABLATION_INSPECT_TENSOR": "0"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_proj_rms(self):
        from paddlefleet.fusions.fused_mhc_kernels import fused_proj_rms

        def run(**kw):
            paddle.seed(23)
            x = _rand(_S, _B, _K, dtype="bfloat16")
            w = _rand(_K, _NMAP, dtype="bfloat16")
            return fused_proj_rms(x, w, 1e-6, fuse_cast=True, **kw)

        for ref, got in zip(run(), run(align_sglang=False)):
            self.assertTrue(_cmp(ref, got)[0])

    def test_h_post_bda(self):
        from paddlefleet.fusions.fused_mhc_kernels import fused_h_post_bda

        def run(**kw):
            paddle.seed(29)
            return fused_h_post_bda(
                _rand(_S, _B, _N, _N),
                _rand(_S, _B, _N, _C, dtype="bfloat16"),
                _rand(_S, _B, _N),
                _rand(_S, _B, _C, dtype="bfloat16"),
                None,
                fuse_cast=True,
                **kw,
            )

        self.assertTrue(_cmp(run(), run(align_sglang=False))[0])


if __name__ == "__main__":
    unittest.main()
