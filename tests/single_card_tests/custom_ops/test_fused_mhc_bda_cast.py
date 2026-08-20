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

"""Tests for the h_post_bda cast fusion (``fuse_cast``).

The fusion lets the cuTile h_post_bda kernel take ``original_residual``/``x``
in the residual dtype and widen them to fp32 in-register, instead of the caller
materializing fp32 copies. Three things need to hold:

1. A ``ct.Constant[int]`` guard compiles away, so the un-fused specialization is
   bitwise identical to a kernel written without the widening at all. This is
   what lets one kernel source serve both paths.
2. Widening in-register is bitwise equivalent to widening in Python, for the
   forward output and for the two large gradients.
3. ``high_precision_mhc=False`` is untouched: the caller never widened there, so
   the flag must not reach the kernel.
"""

import unittest

import paddle

from paddlefleet.fusions.fused_mhc_kernels import is_cutile_available

_S, _B, _N, _C = 4, 2, 4, 1024


def _rand(*shape, dtype="float32"):
    return paddle.randn(list(shape), dtype="float32").astype(dtype)


def _small_int(*shape, dtype="float32"):
    """Integers in [-8, 8].

    Exact in bf16 (integers up to 256) so the inputs are unchanged by the narrow
    storage, while the partial sums grow past 256 -- far enough that any
    accumulation left in bf16 would round, yet well under 2**24 so fp32 keeps
    them exact. Summation is then order-independent, which turns bitwise
    equality into a fair demand and makes a missed widening visible.
    """
    return paddle.randint(-8, 9, list(shape)).astype("float32").astype(dtype)


@unittest.skipUnless(is_cutile_available(), "cuTile not available")
class TestConstGuard(unittest.TestCase):
    """A ``ct.Constant[bool]`` guard must fold away at compile time.

    This is the premise for keeping a single kernel source: with the guard off,
    the emitted code must contain no widening, i.e. be bitwise identical to a
    kernel that never had one. If this fails, the two paths need two kernels.
    """

    def test_guard_folds_and_matches_unguarded(self):
        import cuda.tile as ct

        ConstInt = ct.Constant[int]
        ConstBool = ct.Constant[bool]
        PAD = ct.PaddingMode.ZERO
        TILE = 256

        @ct.kernel
        def unguarded(src, other, out, TILE: ConstInt):
            pid = ct.bid(0)
            a = ct.load(src, index=(pid,), shape=(TILE,), padding_mode=PAD)
            b = ct.load(other, index=(pid,), shape=(TILE,), padding_mode=PAD)
            ct.store(out, index=(pid,), tile=(a * b).astype(out.dtype))

        @ct.kernel
        def guarded(src, other, out, TILE: ConstInt, UPCAST: ConstBool):
            pid = ct.bid(0)
            a = ct.load(src, index=(pid,), shape=(TILE,), padding_mode=PAD)
            if UPCAST:
                a = a.astype(ct.float32)
            b = ct.load(other, index=(pid,), shape=(TILE,), padding_mode=PAD)
            ct.store(out, index=(pid,), tile=(a * b).astype(out.dtype))

        n = 4096
        paddle.seed(0)
        f32 = _rand(n)
        other = _rand(n)
        bf16 = f32.astype("bfloat16")
        stream = paddle.device.current_stream().stream_base.cuda_stream

        def run(kern, src, *extra):
            out = paddle.empty([n], dtype="float32")
            ct.launch(
                stream, (n // TILE,), kern, (src, other, out, TILE, *extra)
            )
            paddle.device.synchronize()
            return out.astype("float32")

        plain = run(unguarded, f32)
        off = run(guarded, f32, False)
        on_f32 = run(guarded, f32, True)
        on_bf16 = run(guarded, bf16, True)

        # guard off == no guard at all
        self.assertTrue(bool((plain == off).all()), "UPCAST=False diverged")
        # widening an already-fp32 tile is a no-op
        self.assertTrue(
            bool((plain == on_f32).all()), "UPCAST=True on fp32 drifted"
        )
        # widening a bf16 tile reproduces the Python-side widening exactly
        ref = bf16.astype("float32") * other
        self.assertTrue(
            bool((on_bf16 == ref).all()), "UPCAST=True on bf16 mismatched"
        )


@unittest.skipUnless(is_cutile_available(), "cuTile not available")
class TestFuseCastEquivalence(unittest.TestCase):
    """fuse_cast=True must match the widen-in-Python path it replaces.

    Bitwise for the forward output and for ``g_residual``/``g_x``. The two
    reduced gradients ``g_h_res``/``g_h_post`` reassociate (see
    ``_ct_hpb_bwd_kernel``) and are only held to fp32-ULP agreement.
    """

    # fp32 ULP scale; the drift measured at C=4096 is ~2e-7 relative
    REDUCED_GRAD_RTOL = 1e-5

    def _run(self, fuse_cast, with_bias, integer=False):
        from paddlefleet.fusions.fused_mhc_kernels import fused_h_post_bda

        paddle.seed(7)
        mk = _small_int if integer else _rand
        # h_res / h_post are fp32 in the real graph regardless of
        # high_precision_mhc (_compute_h promotes them via its fp32 bias).
        h_res = mk(_S, _B, _N, _N)
        h_post = mk(_S, _B, _N)
        # residual / layer output enter the layer in the low-precision dtype.
        # They stay the leaves in both configurations, with the widening inside
        # the graph, so the gradients being compared are the ones that actually
        # reach the residual stream -- both bf16.
        orig_leaf = mk(_S, _B, _N, _C, dtype="bfloat16")
        x_leaf = mk(_S, _B, _C, dtype="bfloat16")
        bias_leaf = mk(_C, dtype="bfloat16") if with_bias else None
        leaves = [h_res, h_post, orig_leaf, x_leaf]
        if with_bias:
            leaves.append(bias_leaf)
        for t in leaves:
            t.stop_gradient = False

        if fuse_cast and not with_bias:
            orig, x, bias = orig_leaf, x_leaf, bias_leaf
        else:
            # What the caller does today. It mirrors the kernel's own bias veto,
            # so with a bias present it still pre-widens even with the flag
            # on -- which is why the bias case must come out bit-identical.
            orig = orig_leaf.astype("float32")
            x = x_leaf.astype("float32")
            bias = bias_leaf.astype("float32") if with_bias else None

        out = fused_h_post_bda(
            h_res, orig, h_post, x, bias, fuse_cast=fuse_cast
        )
        # the caller always brings the result back to the residual dtype
        casted = out.to("bfloat16")
        (casted.astype("float32") * 1.0).sum().backward()
        g = lambda t: t.grad.astype("float32")
        return {
            "out": casted.astype("float32"),
            "g_res": g(orig_leaf),
            "g_x": g(x_leaf),
            "g_h_res": g(h_res),
            "g_h_post": g(h_post),
            "g_bias": g(bias_leaf) if with_bias else None,
        }

    def _check(self, with_bias):
        ref = self._run(fuse_cast=False, with_bias=with_bias)
        got = self._run(fuse_cast=True, with_bias=with_bias)
        for key in ("out", "g_res", "g_x"):
            self.assertTrue(
                bool((ref[key] == got[key]).all()),
                f"{key} must be bitwise identical, got max diff "
                f"{float((ref[key] - got[key]).abs().max()):.3e}",
            )
        for key in ("g_h_res", "g_h_post"):
            scale = float(ref[key].abs().mean())
            diff = float((ref[key] - got[key]).abs().max())
            self.assertLess(
                diff,
                self.REDUCED_GRAD_RTOL * scale,
                f"{key} drifted beyond fp32 ULP: {diff:.3e} "
                f"vs scale {scale:.3e}",
            )

    def test_no_bias(self):
        self._check(with_bias=False)

    def test_exact_arithmetic_is_bitwise(self):
        """With exact arithmetic every tensor must match bitwise.

        Reduction order is the one thing the fusion legitimately changes, and it
        only surfaces when the summation rounds. Integers remove that degree of
        freedom, so anything still differing here is a real defect -- a tensor
        left narrow in the arithmetic, a wrong index, a dropped term.
        """
        ref = self._run(fuse_cast=False, with_bias=False, integer=True)
        got = self._run(fuse_cast=True, with_bias=False, integer=True)
        for key in ("out", "g_res", "g_x", "g_h_res", "g_h_post"):
            a, b = ref[key], got[key]
            diff = float((a - b).abs().max())
            print(
                f"  [exact] {key:<9} bitwise={bool((a == b).all())!s:<5} "
                f"max_abs={diff:.3e} max|val|={float(a.abs().max()):.0f}"
            )
            self.assertTrue(
                bool((a == b).all()),
                f"{key} differs under exact arithmetic, so the cause is not "
                f"reduction order: max_abs={diff:.3e}",
            )

    def test_bias_declines_the_fusion(self):
        """With a bias present the kernel must ignore ``fuse_cast`` entirely.

        ``g_bias`` is a ``[sb, C] -> [C]`` reduction over ``g_x``; handing it a
        narrow ``g_x`` costs bf16-scale precision, not ULP-scale, so the bias
        variant stays on the widen-in-Python path and must be bit-identical.
        """
        ref = self._run(fuse_cast=False, with_bias=True)
        got = self._run(fuse_cast=True, with_bias=True)
        for key in ("out", "g_res", "g_x", "g_h_res", "g_h_post", "g_bias"):
            self.assertTrue(
                bool((ref[key] == got[key]).all()),
                f"{key} changed even though bias should veto the fusion: "
                f"max diff {float((ref[key] - got[key]).abs().max()):.3e}",
            )

    def test_output_dtype(self):
        """fuse_cast decides the output dtype; without it nothing changes."""
        from paddlefleet.fusions.fused_mhc_kernels import fused_h_post_bda

        paddle.seed(7)
        h_res = _rand(_S, _B, _N, _N)
        h_post = _rand(_S, _B, _N)
        orig_bf16 = _rand(_S, _B, _N, _C, dtype="bfloat16")
        x_bf16 = _rand(_S, _B, _C, dtype="bfloat16")

        fused = fused_h_post_bda(
            h_res, orig_bf16, h_post, x_bf16, None, fuse_cast=True
        )
        self.assertEqual(fused.dtype, paddle.bfloat16)
        # fuse_cast=False keeps the pre-fusion rule (out follows h_res), which
        # is what protects the high_precision_mhc=False path from this change.
        plain = fused_h_post_bda(h_res, orig_bf16, h_post, x_bf16, None)
        self.assertEqual(plain.dtype, paddle.float32)


if __name__ == "__main__":
    unittest.main()
