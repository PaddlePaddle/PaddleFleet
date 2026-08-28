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

"""Numeric record for the mHC forward cast fusion.

``HyperConnectionModule.forward`` used to materialize an fp32 copy of its whole
``[..., n*C]`` input so ``fused_proj_rms`` and ``fused_h_aggregate`` could
compute in fp32. With ``fuse_cast`` the input stays narrow and both kernels
widen it in-register instead.

These tests compare the two paths on identical values, forward and backward, and
assert the per-tensor tolerances measured below. They are the documented record
of what the fusion costs numerically -- if a tolerance here has to be loosened,
that is a finding, not a test to adjust.

Shapes match production: n=4, C=4096, so proj_rms reduces over K=16384.
"""

import unittest

import paddle

from paddlefleet.fusions.fused_mhc_kernels import is_cutile_available

_S, _B, _N, _C = 4, 2, 4, 4096
_K = _N * _C
_NMAP = _N * _N + 2 * _N


def _rand(*shape, dtype="float32"):
    return paddle.randn(list(shape), dtype="float32").astype(dtype)


def _small_int(*shape, dtype="float32"):
    """Integers in [-8, 8].

    Exact in bf16 (which represents integers up to 256) and in the tfloat32 the
    mma truncates to, so every product is exact and every partial sum stays a
    whole number well under 2**24. Summation is then exact regardless of the
    order it happens in -- which is what makes bitwise equality a fair demand.
    """
    return paddle.randint(-8, 9, list(shape)).astype("float32").astype(dtype)


def _cmp(ref, got):
    """(bitwise, max_abs_diff, mean_abs_of_ref, relative)."""
    a, b = ref.astype("float32"), got.astype("float32")
    diff = float((a - b).abs().max())
    scale = float(a.abs().mean())
    return bool((a == b).all()), diff, scale, (diff / scale if scale else 0.0)


class _FuseCastCase(unittest.TestCase):
    """Shared check: run both paths, print the deltas, assert tolerances."""

    # subclasses fill these in
    TOL: dict = {}

    def _run(self, fuse_cast, integer=False):  # pragma: no cover
        raise NotImplementedError

    def _check_exact(self):
        """With exact arithmetic every tensor must match bitwise.

        Reduction order is the one thing the fusion legitimately changes, and it
        only shows up when the summation rounds. Feeding integers removes that
        degree of freedom, so anything that still differs here is a real bug --
        wrong dtype handling, wrong indexing, a dropped or duplicated term --
        not the documented reassociation.
        """
        ref = self._run(fuse_cast=False, integer=True)
        got = self._run(fuse_cast=True, integer=True)
        for key in self.TOL:
            bitwise, diff, scale, _ = _cmp(ref[key], got[key])
            print(
                f"  [exact] {key:<10} bitwise={bitwise!s:<5} "
                f"max_abs={diff:.3e} max|val|={float(ref[key].astype('float32').abs().max()):.0f}"
            )
            self.assertTrue(
                bitwise,
                f"{key} differs under exact arithmetic, so the cause is not "
                f"reduction order: max_abs={diff:.3e}",
            )

    def _check(self):
        ref = self._run(fuse_cast=False)
        got = self._run(fuse_cast=True)
        for key, tol in self.TOL.items():
            bitwise, diff, scale, rel = _cmp(ref[key], got[key])
            print(
                f"  {key:<10} bitwise={bitwise!s:<5} max_abs={diff:.3e} "
                f"scale={scale:.3e} rel={rel:.2e} tol={tol:.0e}"
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
class TestProjRmsFuseCast(_FuseCastCase):
    """fused_proj_rms: narrow input widened in-kernel vs widened in Python."""

    # Measured: all four are bitwise identical. The mma truncates both
    # operands to tfloat32 either way, and a bf16-valued tile survives that
    # exactly, so widening in-register reproduces the Python-side widening.
    TOL = {"proj": 0.0, "r": 0.0, "g_x": 0.0, "g_w": 0.0}

    def _run(self, fuse_cast, integer=False):
        from paddlefleet.fusions.fused_mhc_kernels import fused_proj_rms

        paddle.seed(11)
        mk = _small_int if integer else _rand
        # Both leaves are bf16: that is how they exist in the real graph
        # (mapping_proj.weight follows the global default dtype).
        x_leaf = mk(_S, _B, _K, dtype="bfloat16")
        w_leaf = mk(_K, _NMAP, dtype="bfloat16")
        for t in (x_leaf, w_leaf):
            t.stop_gradient = False

        if fuse_cast:
            x, w = x_leaf, w_leaf
        else:
            x, w = x_leaf.astype("float32"), w_leaf.astype("float32")

        proj, r = fused_proj_rms(x, w, 1e-6, fuse_cast=fuse_cast)
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

    def test_precision(self):
        self._check()

    def test_exact_arithmetic_is_bitwise(self):
        self._check_exact()

    def test_dtypes(self):
        got = self._run(fuse_cast=True)
        # proj / r must stay fp32: _compute_h builds h_res / h_post from them,
        # and narrowing them would silently undo high_precision_mhc.
        self.assertEqual(got["_proj_dtype"], paddle.float32)
        self.assertEqual(got["_r_dtype"], paddle.float32)
        # the gradient goes back to a bf16 leaf
        self.assertEqual(got["_g_x_dtype"], paddle.bfloat16)


@unittest.skipUnless(is_cutile_available(), "cuTile not available")
class TestHAggregateFuseCast(_FuseCastCase):
    """fused_h_aggregate: narrow x widened in-kernel vs widened in Python."""

    # Measured: forward and g_x are bitwise identical; g_h_pre is not, since
    # it is the one value reduced over TILE_C and the narrower tiles let
    # cuTile reschedule that reduction. ~3e-07 relative, i.e. fp32 ULP and
    # four orders below bf16 resolution.
    TOL = {"aggregated": 0.0, "g_x": 0.0, "g_h_pre": 1e-6}

    def _run(self, fuse_cast, integer=False):
        from paddlefleet.fusions.fused_mhc_kernels import fused_h_aggregate

        paddle.seed(13)
        mk = _small_int if integer else _rand
        x_leaf = mk(_S, _B, _N, _C, dtype="bfloat16")
        # h_pre is fp32 under high_precision_mhc regardless of the fusion
        h_pre = mk(_S, _B, _N)
        for t in (x_leaf, h_pre):
            t.stop_gradient = False

        x = x_leaf if fuse_cast else x_leaf.astype("float32")
        out = fused_h_aggregate(x, h_pre, fuse_cast=fuse_cast)
        # the caller brings the result back to the residual dtype either way
        casted = out.to("bfloat16")
        (casted.astype("float32") * 1.0).sum().backward()
        return {
            "aggregated": casted,
            "g_x": x_leaf.grad,
            "g_h_pre": h_pre.grad,
            "_out_dtype": out.dtype,
            "_g_x_dtype": x_leaf.grad.dtype,
            "_g_h_pre_dtype": h_pre.grad.dtype,
        }

    def test_precision(self):
        self._check()

    def test_exact_arithmetic_is_bitwise(self):
        self._check_exact()

    def test_dtypes(self):
        got = self._run(fuse_cast=True)
        # aggregated follows x: its consumer is the bf16 layernorm, and the
        # caller was going to round it there anyway
        self.assertEqual(got["_out_dtype"], paddle.bfloat16)
        self.assertEqual(got["_g_x_dtype"], paddle.bfloat16)
        # h_pre stays fp32, so its gradient cannot follow x
        self.assertEqual(got["_g_h_pre_dtype"], paddle.float32)


if __name__ == "__main__":
    unittest.main()
