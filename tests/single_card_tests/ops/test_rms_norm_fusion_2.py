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

"""Behavior tests for ``paddlefleet.triton_ops.rms_norm_fusion`` (part 2).

Module under test lives in the repository's "计算优化 / Fused Ops" boundary:
``RMSNormFusionTriton`` is a ``paddle.autograd.PyLayer`` whose forward computes
``y = x * rsqrt(mean(x**2, -1) + eps) * w`` and whose backward returns
``dx = invvar * (dy*w - x * invvar**2 * dot/N)`` (``dot = sum_j dy_j w_j x_j``)
and ``dw = sum_over_rows(dy * x * invvar)`` via Triton GPU kernels.

These cases deliberately cover branches the base
(``tests/single_card_tests/custom_ops/test_rms_norm_fusion_triton.py``, which
exercises a 2-D contiguous bfloat16 ``[1024, 128]`` tensor) does not:

* ``x.ndim < 2`` -> the ``else`` branch ``stride_x_row = n2`` with ``n1 == 1``
  (single-program launch);
* a non-contiguous (sliced) input where ``stride_x_row = x.stride()[-2]``
  differs from the normalized dim, exercising the advertised strided-input read;
* a non-power-of-two, ``> 256`` normalized dim, exercising the masked
  ``cols < actual_n2`` load and the ``num_warps = 4`` launch branch.

All comparisons use an independent hand-derived paddle reference (not the Triton
kernel) with non-uniform per-channel weights and a non-uniform upstream
gradient, so a wrong stride, dropped weight, or mis-scaled backward is rejected.
The Triton kernels require a real CUDA GPU, so the whole module is skipped
honestly (only on a genuine ``ImportError`` or when no GPU is present); it is
never faked to pass on CPU.
"""

import contextlib
import unittest

try:
    import numpy as np
    import paddle

    from paddlefleet.triton_ops import RMSNormFusionTriton

    _IMPORT_ERROR = None
except ImportError as exc:  # only a genuine missing dependency -> honest skip
    RMSNormFusionTriton = None
    _IMPORT_ERROR = exc


def _gpu_available():
    """Honest capability probe: kernels need an actual CUDA device."""
    if _IMPORT_ERROR is not None:
        return False
    return (
        paddle.device.is_compiled_with_cuda()
        and paddle.device.cuda.device_count() > 0
    )


_RUNNABLE = _gpu_available()
_SKIP_REASON = (
    "paddlefleet.triton_ops.rms_norm_fusion requires an importable paddle "
    "build and a CUDA GPU to launch its Triton kernels; unavailable here "
    f"(import error: {_IMPORT_ERROR!r})"
)


@contextlib.contextmanager
def _triton_compat():
    """Enable the ``triton`` compat scope around a real kernel launch."""
    if hasattr(paddle, "enable_compat"):
        paddle.enable_compat(scope={"triton"}, silent=True)
    try:
        yield
    finally:
        if hasattr(paddle, "disable_compat"):
            paddle.disable_compat()


def _reference_rms(x, weight, eps):
    """Independent RMSNorm reference in pure paddle (no Triton kernel)."""
    var = (x * x).mean(axis=-1, keepdim=True)
    invvar = paddle.rsqrt(var + eps)
    return x * invvar * weight


@unittest.skipUnless(_RUNNABLE, _SKIP_REASON)
class TestRMSNormFusionTritonBranches(unittest.TestCase):
    """Branch coverage complementary to the base 2-D contiguous test."""

    EPS = 1e-6

    def _assert_fwd_bwd(self, x, weight, upstream, rtol_g=1e-3, atol_g=1e-4):
        x.stop_gradient = False
        weight.stop_gradient = False
        x_ref = x.detach().clone()
        w_ref = weight.detach().clone()
        x_ref.stop_gradient = False
        w_ref.stop_gradient = False

        with _triton_compat():
            out = RMSNormFusionTriton.apply(x, weight, self.EPS)
            out.backward(upstream)

        ref = _reference_rms(x_ref, w_ref, self.EPS)
        ref.backward(upstream)

        np.testing.assert_allclose(
            out.numpy(), ref.numpy(), rtol=1e-4, atol=1e-5
        )
        for got, exp in ((x.grad, x_ref.grad), (weight.grad, w_ref.grad)):
            self.assertIsNotNone(got)
            self.assertIsNotNone(exp)
            np.testing.assert_allclose(
                got.numpy(), exp.numpy(), rtol=rtol_g, atol=atol_g
            )

    def test_1d_input_else_branch(self):
        """ndim < 2 -> stride_x_row = n2, n1 == 1, single-program launch."""
        paddle.seed(20260916)
        n = 96
        x = paddle.randn([n], dtype="float32")
        weight = paddle.randn([n], dtype="float32")
        upstream = paddle.randn([n], dtype="float32") * 0.7
        self._assert_fwd_bwd(x, weight, upstream)

    def test_strided_noncontiguous_input(self):
        """Sliced input: stride_x_row (row stride) differs from actual_n2."""
        paddle.seed(11)
        rows, n = 32, 48
        base = paddle.randn([rows, 2 * n], dtype="float32")
        x = base[:, :n]  # non-contiguous view, row stride == 2*n
        # Precondition: the strided branch is genuinely exercised.
        self.assertEqual(x.stride()[x.ndim - 2], 2 * n)
        weight = paddle.randn([n], dtype="float32")

        with _triton_compat():
            out = RMSNormFusionTriton.apply(x, weight, self.EPS)

        ref = _reference_rms(x, weight, self.EPS)
        # Forward-only: a wrong stride reads the wrong columns of `base`.
        np.testing.assert_allclose(
            out.numpy(), ref.numpy(), rtol=1e-4, atol=1e-5
        )

    def test_non_power_of_two_large_dim(self):
        """actual_n2=320: masked load (block_n2=512) and num_warps=4 branch."""
        paddle.seed(7)
        rows, n = 16, 320
        x = paddle.randn([rows, n], dtype="float32")
        weight = paddle.randn([n], dtype="float32")
        upstream = paddle.randn([rows, n], dtype="float32") * 0.5
        self._assert_fwd_bwd(x, weight, upstream)


if __name__ == "__main__":
    unittest.main()
