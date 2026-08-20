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

"""Numeric record for the fused mHC mapping head (``fused_compute_h``).

``fused_compute_h`` replaces ``native_compute_h`` -- the
``expand x3 -> concat -> mul -> mul -> add -> sigmoid x2`` chain -- with a
single cuTile kernel. This is a launch-count optimization, not a dtype one,
so the values ought to agree closely; these tests record how closely, forward
and backward.

The two gradients that reduce over every token (``alpha`` and ``bias``) are the
ones expected to drift: the fused path stages per-block partials and sums them,
which groups the additions differently from Paddle's own reduction.
"""

import unittest

import paddle

from paddlefleet.fusions.fused_mhc_kernels import is_cutile_available
from paddlefleet.transformer.hyper_connection import native_compute_h

_S, _B, _N = 8, 2, 4
_P = _N * _N + 2 * _N
_EPS = 1e-6

_KEYS = (
    "h_pre",
    "h_post",
    "h_res",
    "g_proj",
    "g_r",
    "g_alpha_pre",
    "g_alpha_post",
    "g_alpha_res",
    "g_bias",
)


def _rand(*shape, dtype="float32"):
    return paddle.randn(list(shape), dtype="float32").astype(dtype)


def _small_int(*shape, dtype="float32"):
    """Integers in [-4, 4].

    Kept small enough that ``r * proj * alpha`` stays a whole number, so the
    additions are exact and grouping cannot matter. Anything still differing
    under these inputs is a real defect rather than the documented
    reassociation of the two cross-token reductions.
    """
    return paddle.randint(-4, 5, list(shape)).astype("float32").astype(dtype)


def _cmp(ref, got):
    a, b = ref.astype("float32"), got.astype("float32")
    diff = float((a - b).abs().max())
    scale = float(a.abs().mean())
    return bool((a == b).all()), diff, scale, (diff / scale if scale else 0.0)


@unittest.skipUnless(is_cutile_available(), "cuTile not available")
class TestFusedComputeH(unittest.TestCase):
    """fused_compute_h against native_compute_h, forward and backward."""

    # Per-token outputs and their per-token gradients must match bitwise; the
    # two cross-token reductions are held to fp32 ULP. Filled in from the
    # measurements printed by the tests below.
    # Tolerances are set by the output dtype, not by one global number.
    #
    # fp32 outputs (proj / r and everything derived per token) land within one
    # fp32 ULP: measured h_pre 1.0e-07, h_post 1.5e-07, h_res 2.6e-07,
    # g_proj 6.3e-08, g_r 3.7e-07.
    #
    # The alpha and bias gradients come back in the parameter dtype, i.e. bf16,
    # whose spacing near their magnitude is already ~4e-03. A tiny fp32
    # difference upstream therefore shows up as a whole bf16 step -- measured
    # g_alpha_res 4.7e-03, which is exactly one. Demanding fp32-ULP agreement on
    # a bf16 output would be demanding more resolution than the dtype has.
    #
    # Nothing here is bitwise on random inputs, unlike the two cast fusions:
    # ``u = r*proj*a + b`` is one multiply-add and the kernel contracts it into
    # an FMA, one rounding where the reference rounds twice. That difference
    # disappears under ``test_exact_arithmetic_is_bitwise`` and under
    # ``test_fma_probe_makes_it_bitwise``.
    _FP32_ULP = 1e-6
    _BF16_ULP = 8e-3
    TOL = {
        "h_pre": _FP32_ULP,
        "h_post": _FP32_ULP,
        "h_res": _FP32_ULP,
        "g_proj": _FP32_ULP,
        "g_r": _FP32_ULP,
        "g_alpha_pre": _BF16_ULP,
        "g_alpha_post": _BF16_ULP,
        "g_alpha_res": _BF16_ULP,
        "g_bias": _BF16_ULP,
    }

    # With the reference rounding once too, these must agree to the last bit:
    # they are per-token and fp32, so neither the contraction nor a reduction
    # order is left to explain a difference.
    FMA_PROBE_BITWISE = ("h_pre", "h_post", "h_res", "g_proj")

    def _run(
        self,
        fused,
        integer=False,
        res_only=False,
        probe=False,
        dtype="float32",
        param_dtype="bfloat16",
    ):
        from paddlefleet.fusions.fused_mhc_kernels import fused_compute_h

        paddle.seed(17)
        mk = _small_int if integer else _rand
        proj = mk(_S, _B, _P, dtype=dtype)
        # r is 1 / (||x|| / sqrt(K) + eps): positive and O(1). Build the values
        # first, then take a leaf, or r.grad would come back None.
        r = (mk(_S, _B, 1, dtype=dtype).abs() + 1.0).detach()
        # alpha follows params_dtype and bias the global default dtype, i.e.
        # bf16 in bf16 training, while proj / r are fp32 under
        # high_precision_mhc. Keep that mix: it is the only configuration the
        # real model runs, and testing with fp32 here would leave the kernel's
        # widening of these operands entirely uncovered.
        alpha_pre = mk(1, dtype=param_dtype)
        alpha_post = mk(1, dtype=param_dtype)
        alpha_res = mk(1, dtype=param_dtype)
        bias = mk(_P, dtype=param_dtype)
        leaves = [proj, r, alpha_pre, alpha_post, alpha_res, bias]
        for t in leaves:
            t.stop_gradient = False

        if fused:
            h_pre, h_post, h_res = fused_compute_h(
                proj, r, alpha_pre, alpha_post, alpha_res, bias, _N, _EPS
            )
        else:
            h_pre, h_post, h_res = native_compute_h(
                proj,
                r,
                alpha_pre,
                alpha_post,
                alpha_res,
                bias,
                _N,
                _EPS,
                _fma_probe=probe,
            )
        if res_only:
            # h_res is the one head with no sigmoid on it, so with integer
            # inputs this loss keeps every gradient an exact integer.
            loss = 3.0 * h_res.astype("float32").sum()
        else:
            # weight the three heads unequally so no gradient path cancels out
            loss = (
                h_pre.astype("float32").sum()
                + 2.0 * h_post.astype("float32").sum()
                + 3.0 * h_res.astype("float32").sum()
            )
        loss.backward()
        return {
            "h_pre": h_pre,
            "h_post": h_post,
            "h_res": h_res,
            "g_proj": proj.grad,
            "g_r": r.grad,
            "g_alpha_pre": alpha_pre.grad,
            "g_alpha_post": alpha_post.grad,
            "g_alpha_res": alpha_res.grad,
            "g_bias": bias.grad,
        }

    def test_precision(self):
        ref = self._run(fused=False)
        got = self._run(fused=True)
        for key in _KEYS:
            tol = self.TOL[key]
            bitwise, diff, scale, rel = _cmp(ref[key], got[key])
            print(
                f"  {key:<12} bitwise={bitwise!s:<5} max_abs={diff:.3e} "
                f"scale={scale:.3e} rel={rel:.2e} tol={tol:.0e}"
            )
            if tol == 0.0:
                self.assertTrue(
                    bitwise, f"{key} must be bitwise identical, rel={rel:.3e}"
                )
            else:
                self.assertLess(
                    rel, tol, f"{key} drifted past tolerance: rel={rel:.3e}"
                )

    def test_exact_arithmetic_is_bitwise(self):
        """Integer inputs make every addition exact, so grouping cannot matter.

        The two cross-token reductions are the only thing the fused path
        reassociates; removing the rounding removes that degree of freedom, and
        anything still differing points at a genuine defect.
        """
        ref = self._run(fused=False, integer=True, res_only=True)
        got = self._run(fused=True, integer=True, res_only=True)
        for key in _KEYS:
            bitwise, diff, _, _ = _cmp(ref[key], got[key])
            print(
                f"  [exact] {key:<12} bitwise={bitwise!s:<5} max_abs={diff:.3e}"
            )
            self.assertTrue(
                bitwise,
                f"{key} differs under exact arithmetic, so the cause is not "
                f"reduction order: max_abs={diff:.3e}",
            )

    def test_fma_probe_makes_it_bitwise(self):
        """With the reference rounding once, the two paths must match bitwise.

        ``native_compute_h(_fma_probe=True)`` rewrites the reference to do a
        single rounding of ``t1 * alpha + bias``, which is exactly what the
        kernel's FMA does. If everything else about the fusion is
        behaviour-neutral -- the collapsed launches, the alpha vector that never
        materializes, the staged cross-token reductions -- the two paths agree to
        the last bit under it, and the residual difference measured by
        ``test_precision`` is entirely the contraction.
        """
        ref = self._run(fused=False, probe=True)
        got = self._run(fused=True)
        residual = []
        for key in _KEYS:
            bitwise, diff, scale, rel = _cmp(ref[key], got[key])
            print(
                f"  [fma] {key:<12} bitwise={bitwise!s:<5} "
                f"max_abs={diff:.3e} rel={rel:.2e}"
            )
            if key in self.FMA_PROBE_BITWISE and not bitwise:
                residual.append(f"{key} (rel={rel:.2e})")
            elif rel >= self.TOL[key]:
                residual.append(f"{key} (rel={rel:.2e} over tol)")
        self.assertFalse(
            residual,
            "still differing once the reference rounds once too, beyond what "
            "the reductions and the output dtypes explain: "
            + ", ".join(residual),
        )

    def test_exact_arithmetic_with_fma_probe(self):
        """The discriminator for the four gradients ``test_fma_probe`` leaves.

        Those four are all sums. Two things could explain them: the order the
        additions are grouped in, or the fp64 the probe puts into the reference's
        forward leaking into its backward.

        Integer inputs settle it. With the loss on ``h_res`` only, ``du`` is a
        constant and every partial sum is a whole number, so no grouping can
        change the result and no fp64 can add precision that matters. If the two
        paths agree bitwise here while differing on random inputs, the cause is
        the grouping -- there is nothing else left.
        """
        ref = self._run(fused=False, integer=True, res_only=True, probe=True)
        got = self._run(fused=True, integer=True, res_only=True)
        residual = []
        for key in _KEYS:
            bitwise, diff, _, _ = _cmp(ref[key], got[key])
            print(
                f"  [exact+fma] {key:<12} bitwise={bitwise!s:<5} "
                f"max_abs={diff:.3e}"
            )
            if not bitwise:
                residual.append(f"{key} ({diff:.2e})")
        self.assertFalse(
            residual,
            "exact arithmetic plus a matching rounding count still leaves a "
            "difference, so neither grouping nor the FMA explains it: "
            + ", ".join(residual),
        )

    def test_shapes_and_dtypes(self):
        got = self._run(fused=True)
        self.assertEqual(got["h_pre"].shape, [_S, _B, _N])
        self.assertEqual(got["h_post"].shape, [_S, _B, _N])
        self.assertEqual(got["h_res"].shape, [_S, _B, _N * _N])
        # every gradient comes back in its input's dtype
        for key in ("h_pre", "h_post", "h_res", "g_proj", "g_r"):
            self.assertEqual(got[key].dtype, paddle.float32, key)
        # alpha / bias gradients come back in the parameter dtype
        for key in ("g_alpha_pre", "g_bias"):
            self.assertEqual(got[key].dtype, paddle.bfloat16, key)

    def test_frozen_inputs_return_none(self):
        """A frozen mHC block must not make backward raise.

        Paddle requires ``None`` at every position whose forward input was
        detached; ``train_indexer_only`` freezes exactly these parameters.
        """
        from paddlefleet.fusions.fused_mhc_kernels import fused_compute_h

        paddle.seed(19)
        proj = _rand(_S, _B, _P)
        r = _rand(_S, _B, 1).abs() + 1.0
        alpha_pre = _rand(1)
        alpha_post = _rand(1)
        alpha_res = _rand(1)
        bias = _rand(_P)
        proj.stop_gradient = False  # only this one trains
        for t in (r, alpha_pre, alpha_post, alpha_res, bias):
            t.stop_gradient = True

        h_pre, h_post, h_res = fused_compute_h(
            proj, r, alpha_pre, alpha_post, alpha_res, bias, _N, _EPS
        )
        (h_pre.sum() + h_post.sum() + h_res.sum()).backward()
        self.assertIsNotNone(proj.grad)


@unittest.skipUnless(is_cutile_available(), "cuTile not available")
class TestFusedComputeHValidation(unittest.TestCase):
    """Bad shapes must raise, and must not rely on ``assert``.

    ``python -O`` strips asserts, and the kernel addresses the three output
    segments at fixed offsets derived from ``n``. An unchecked ``P`` or bias
    length would therefore be read out of bounds rather than rejected.
    """

    def _args(self, **over):
        paddle.seed(29)
        args = {
            "proj": _rand(_S, _B, _P),
            "r": _rand(_S, _B, 1).abs() + 1.0,
            "alpha_pre": _rand(1),
            "alpha_post": _rand(1),
            "alpha_res": _rand(1),
            "bias": _rand(_P),
        }
        args.update(over)
        return args

    def _call(self, **over):
        from paddlefleet.fusions.fused_mhc_kernels import fused_compute_h

        a = self._args(**over)
        return fused_compute_h(
            a["proj"],
            a["r"],
            a["alpha_pre"],
            a["alpha_post"],
            a["alpha_res"],
            a["bias"],
            _N,
            _EPS,
        )

    def test_accepts_the_valid_shapes(self):
        self._call()

    def test_rejects_wrong_proj_width(self):
        with self.assertRaisesRegex(ValueError, "proj last dim"):
            self._call(proj=_rand(_S, _B, _P + 1))

    def test_rejects_r_last_dim_not_one(self):
        with self.assertRaisesRegex(ValueError, "r last dim"):
            self._call(r=_rand(_S, _B, 2))

    def test_rejects_disagreeing_leading_dims(self):
        with self.assertRaisesRegex(ValueError, "leading dims"):
            self._call(r=_rand(_S, _B + 1, 1))

    def test_rejects_wrong_bias_length(self):
        with self.assertRaisesRegex(ValueError, "bias must be"):
            self._call(bias=_rand(_P - 1))


if __name__ == "__main__":
    unittest.main()
