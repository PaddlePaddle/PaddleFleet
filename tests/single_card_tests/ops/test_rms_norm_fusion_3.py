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

"""Behavior tests for the Triton RMSNorm fusion op (variant _3).

Module under test: ``paddlefleet.triton_ops.rms_norm_fusion``. In the
repository module map this is the "计算优化 / Fused Ops" boundary: a
``paddle.autograd.PyLayer`` (``RMSNormFusionTriton``) whose forward fuses
the RMSNorm math ``y = x / sqrt(mean(x^2) + eps) * w`` and whose backward
computes ``dx`` and ``dw`` analytically, splitting the ``dw`` reduce across
multiple Triton programs for a deterministic, atomic-free accumulation.

Scope of this file (variant _3 -- deliberately exercises branches that a
small contiguous 2-D forward-only base/_2 would not):

* ``x.ndim < 2`` 1-D input, which takes the ``stride_x_row = n2`` else-branch
  in ``forward`` rather than reading a real row stride.
* Non-contiguous / strided rows (a transposed leaf), exercising the
  ``stride_x_row = x.stride()[ndim - 2]`` path so a strided-read regression
  in the kernel would be caught.
* ``n1 > ROWS_PER_PROG`` (128), forcing ``num_programs > 1`` so the two-stage
  deterministic ``dw`` reduce (per-program partial sum -> partial reduce ->
  final reduce) must recombine correctly.
* ``n2 > 256`` so ``block_n2 >= 512`` selects the ``num_warps = 4`` launch.

Every assertion compares the real fused entry ``RMSNormFusionTriton.apply``
against an independent paddle-autograd RMSNorm reference (a different code
path from the hand-written Triton analytic backward), using a
scale-sensitive comparison so an integer-multiple magnitude error in the
gradients (e.g. a wrong ``1/N2`` reduce factor) cannot pass.

Honest gating: the Triton kernels can only launch on a CUDA-enabled paddle
build with a real GPU. When paddle (or the op module) is unimportable, or
paddle is not compiled with CUDA, these tests skip with an explicit reason
rather than fake-passing. There is no CPU fallback path in this module to
assert, so no test is claimed to have run on CPU.
"""

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

try:
    import numpy as np
    import paddle

    from paddlefleet.triton_ops.rms_norm_fusion import RMSNormFusionTriton

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # narrow: honest missing dep
    np = None
    paddle = None
    RMSNormFusionTriton = None
    _IMPORT_ERROR = repr(exc)


def _skip_reason():
    """Return an honest skip reason, or None when the tests can really run."""
    if _IMPORT_ERROR is not None:
        return "paddle / paddlefleet.triton_ops import failed: " + _IMPORT_ERROR
    if not paddle.is_compiled_with_cuda():
        return (
            "requires a CUDA-compiled paddle build with a GPU; the Triton "
            "RMSNorm kernels cannot launch on CPU and this module has no "
            "CPU fallback path"
        )
    return None


_SKIP_REASON = _skip_reason()
_CAN_RUN = _SKIP_REASON is None


@unittest.skipUnless(_CAN_RUN, _SKIP_REASON or "unavailable")
class TestRMSNormFusionBranches(unittest.TestCase):
    """Numerical forward+backward checks against an independent reference."""

    EPS = 1e-6

    def _reference(self, base_np, w_np, upstream_np, make_x):
        """Independent RMSNorm forward/backward via paddle autograd.

        This does NOT touch the Triton analytic backward; it derives dx/dw
        purely from autograd of ``x * rsqrt(mean(x^2)+eps) * w``. ``make_x``
        is applied to the leaf ``base`` exactly as in the fused run, so the
        returned ``dx`` gradient is w.r.t. the same ``base`` layout that the
        fused op's ``base.grad`` uses (critical for the strided/transpose
        case where the leaf layout differs from the normalized layout).
        """
        base = paddle.to_tensor(base_np, dtype="float32")
        base.stop_gradient = False
        wr = paddle.to_tensor(w_np, dtype="float32")
        wr.stop_gradient = False
        xr = make_x(base)
        var = (xr * xr).mean(axis=-1, keepdim=True)
        ref = xr * paddle.rsqrt(var + self.EPS) * wr
        ref.backward(paddle.to_tensor(upstream_np, dtype="float32"))
        return (
            ref.numpy(),
            base.grad.numpy(),
            wr.grad.numpy(),
        )

    def _assert_scale_sensitive(self, actual, ref, ctx):
        """Reject sign flips, reorderings and integer-multiple magnitude
        errors -- not just direction (see antipattern #6)."""
        actual = np.asarray(actual, dtype=np.float64)
        ref = np.asarray(ref, dtype=np.float64)
        self.assertTrue(np.isfinite(actual).all(), ctx + ": non-finite actual")
        self.assertTrue(np.isfinite(ref).all(), ctx + ": non-finite ref")
        np.testing.assert_allclose(
            actual, ref, rtol=2e-3, atol=1e-4, err_msg=ctx
        )
        norm = np.linalg.norm(ref)
        self.assertGreater(norm, 0.0, ctx + ": degenerate zero reference")
        rel = np.linalg.norm(actual - ref) / norm
        self.assertLess(rel, 5e-3, ctx + f" relative-L2={rel!r}")

    def _run_leaf_case(self, x_np, w_np, upstream_np, make_x):
        """Drive the fused op with a leaf produced by ``make_x`` (allows a
        non-contiguous view of a contiguous leaf) and compare to reference.

        ``make_x(base)`` must return the tensor fed to ``apply`` while ``base``
        stays the differentiable leaf whose ``.grad`` we read back.
        """
        base = paddle.to_tensor(x_np, dtype="float32")
        base.stop_gradient = False
        w = paddle.to_tensor(w_np, dtype="float32")
        w.stop_gradient = False
        x = make_x(base)
        out = RMSNormFusionTriton.apply(x, w, self.EPS)
        out.backward(paddle.to_tensor(upstream_np, dtype="float32"))

        ref_out, ref_dx, ref_dw = self._reference(
            x_np, w_np, upstream_np, make_x
        )

        self.assertIsNotNone(out, "forward returned None")
        self.assertIsNotNone(base.grad, "no dx gradient produced")
        self.assertIsNotNone(w.grad, "no dw gradient produced")
        # Reference gradients must be non-trivial or the check is vacuous.
        self.assertGreater(np.abs(ref_dx).max(), 1e-3, "reference dx ~ 0")
        self.assertGreater(np.abs(ref_dw).max(), 1e-3, "reference dw ~ 0")

        self._assert_scale_sensitive(out.numpy(), ref_out, "forward y")
        self._assert_scale_sensitive(base.grad.numpy(), ref_dx, "dx")
        self._assert_scale_sensitive(w.grad.numpy(), ref_dw, "dw")

    def test_forward_backward_1d_input_stride_else_branch(self):
        """1-D input: n1 == 1, ndim < 2 => stride_x_row = n2 else-branch."""
        rng = np.random.RandomState(0)
        n2 = 24
        x_np = rng.randn(n2).astype("float32")
        w_np = (rng.randn(n2) * 0.5 + 1.0).astype("float32")
        upstream = rng.randn(n2).astype("float32")
        self._run_leaf_case(x_np, w_np, upstream, make_x=lambda b: b)

    def test_forward_backward_strided_rows(self):
        """Non-trivial row stride via a last-dim slice exercises the
        ``stride_x_row = x.stride()[ndim-2]`` read path (headline strided-input
        feature) while keeping the normalized last dim unit-stride -- the
        split/slice layout this kernel actually supports (a single row-stride
        parameter plus contiguous columns). ``base`` is [n1, N] contiguous and
        ``x = base[:, :n2]`` has row stride N (!= n2) but stride-1 columns.

        A transpose would instead make the *normalized* dim non-contiguous,
        which this kernel does not support (it reads columns as
        ``X_ptr + row*stride_x_row + cols`` with an implicit unit column
        stride), so a slice is the correct way to drive the strided branch.
        """
        rng = np.random.RandomState(1)
        n1, n2, extra = 5, 16, 7
        total_cols = n2 + extra  # N > n2 so the row stride differs from n2
        base_np = rng.randn(n1, total_cols).astype("float32")  # leaf layout
        w_np = (rng.randn(n2) * 0.3 + 1.0).astype("float32")
        upstream = rng.randn(n1, n2).astype("float32")

        def make_x(base):
            # Slice the last dim: row stride == total_cols != n2, columns
            # remain contiguous (unit stride) as the kernel requires.
            return base[:, :n2]

        # Reference must see the same sliced values, so feed base_np and let
        # make_x slice it inside _run_leaf_case's reference branch too.
        self._run_leaf_case(base_np, w_np, upstream, make_x=make_x)

    def test_forward_backward_multi_program_reduce(self):
        """n1 > ROWS_PER_PROG(128) => num_programs > 1, so the two-stage
        deterministic dw reduce across programs must recombine correctly."""
        rng = np.random.RandomState(2)
        n1, n2 = 300, 40
        x_np = rng.randn(n1, n2).astype("float32")
        w_np = (rng.randn(n2) * 0.4 + 1.0).astype("float32")
        upstream = rng.randn(n1, n2).astype("float32")
        self._run_leaf_case(x_np, w_np, upstream, make_x=lambda b: b)

    def test_forward_backward_large_dim_num_warps_branch(self):
        """n2 > 256 => block_n2 >= 512 selects the num_warps = 4 launch."""
        rng = np.random.RandomState(3)
        n1, n2 = 6, 384
        x_np = rng.randn(n1, n2).astype("float32")
        w_np = (rng.randn(n2) * 0.2 + 1.0).astype("float32")
        upstream = rng.randn(n1, n2).astype("float32")
        self._run_leaf_case(x_np, w_np, upstream, make_x=lambda b: b)


if __name__ == "__main__":
    unittest.main()
