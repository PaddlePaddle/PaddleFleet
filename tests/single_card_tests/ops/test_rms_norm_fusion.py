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

"""Behavior tests for ``paddlefleet.triton_ops.rms_norm_fusion``.

Repository module map placement: 计算优化 / Fused Ops (``triton_ops``). The
module under test exposes ``RMSNormFusionTriton``, a ``paddle.autograd.PyLayer``
whose forward computes ``y = x * rsqrt(mean(x**2, -1) + eps) * weight`` with an
fp32 accumulation and whose backward returns the matching ``dx`` / ``dweight``.
It also advertises support for strided (sliced / split) inputs via the saved
per-row stride.

These are single-card (GPU) numerical tests: the fused path can only run on a
CUDA paddle build with triton, so it is compared against an *independent*
pure-paddle RMSNorm autograd reference (elementary ops only, never the triton
path). When paddle/triton/GPU are unavailable the whole suite is skipped with
an honest reason rather than faking a pass -- in a CPU-only or paddle-less
environment nothing in this module can be exercised.
"""

import unittest

_IMPORT_ERROR = None
try:
    import numpy as np
    import paddle

    from paddlefleet.triton_ops.rms_norm_fusion import RMSNormFusionTriton

    _DEPS_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as exc:  # precise capability probe
    _DEPS_AVAILABLE = False
    _IMPORT_ERROR = repr(exc)


def _gpu_available():
    """True only when paddle can see at least one CUDA device."""
    if not _DEPS_AVAILABLE:
        return False
    return paddle.device.cuda.device_count() > 0


_RUN = _DEPS_AVAILABLE and _gpu_available()
_SKIP_REASON = (
    "requires a CUDA paddle build, a visible GPU and triton to launch the "
    "RMSNorm triton kernels"
    if _DEPS_AVAILABLE
    else f"paddle / rms_norm_fusion import unavailable: {_IMPORT_ERROR}"
)


def _rms_norm_reference(x, weight, eps):
    """Independent RMSNorm forward built from elementary paddle ops.

    Mirrors the fused kernel's fp32-accumulated definition without touching the
    triton path, so it is a valid ground truth for both the forward values and
    (through paddle autograd) the gradients.
    """
    x32 = x.astype("float32")
    var = paddle.mean(x32 * x32, axis=-1, keepdim=True)
    invvar = paddle.rsqrt(var + eps)
    return (x32 * invvar) * weight.astype("float32")


@unittest.skipUnless(_RUN, _SKIP_REASON)
class TestRMSNormFusionTriton(unittest.TestCase):
    """Numerical forward/backward contract of ``RMSNormFusionTriton``."""

    EPS = 1e-6

    def _fixed_inputs(self):
        """Fixed, non-degenerate, per-position-distinct inputs."""
        x = paddle.to_tensor(
            [
                [1.0, -2.0, 0.5, 3.0, -1.5, 2.5, -0.5, 4.0],
                [0.25, 1.75, -3.0, 2.0, -0.75, 0.5, 1.25, -2.5],
                [2.0, -1.0, 0.5, -0.25, 3.5, -2.0, 1.0, 0.75],
                [-1.25, 2.25, -0.5, 1.5, 0.5, -3.0, 2.75, -1.0],
            ],
            dtype="float32",
        )
        weight = paddle.to_tensor(
            [0.5, 1.5, -0.75, 2.0, 1.0, -1.25, 0.25, 1.75], dtype="float32"
        )
        return x, weight

    def test_forward_matches_independent_reference(self):
        x, weight = self._fixed_inputs()
        out = RMSNormFusionTriton.apply(x, weight, self.EPS)
        ref = _rms_norm_reference(x, weight, self.EPS)
        np.testing.assert_allclose(
            out.numpy(), ref.numpy(), rtol=1e-4, atol=1e-5
        )

    def test_forward_and_backward_match_reference(self):
        x, weight = self._fixed_inputs()
        x.stop_gradient = False
        weight.stop_gradient = False
        x_ref = x.detach().clone()
        x_ref.stop_gradient = False
        w_ref = weight.detach().clone()
        w_ref.stop_gradient = False

        # Non-uniform upstream grad exposes reduction / positional mistakes.
        upstream = paddle.to_tensor(
            [
                [0.3, -1.2, 0.7, 2.1, -0.4, 1.5, -2.0, 0.9],
                [1.1, 0.2, -0.6, 1.8, -1.3, 0.5, 2.2, -0.8],
                [-0.9, 1.4, 0.6, -2.3, 0.8, -1.1, 1.7, 0.4],
                [2.0, -0.5, 1.3, 0.1, -1.6, 0.7, -0.3, 1.9],
            ],
            dtype="float32",
        )

        out = RMSNormFusionTriton.apply(x, weight, self.EPS)
        ref = _rms_norm_reference(x_ref, w_ref, self.EPS)
        out.backward(upstream)
        ref.backward(upstream)

        for actual, expected in (
            (out, ref),
            (x.grad, x_ref.grad),
            (weight.grad, w_ref.grad),
        ):
            self.assertIsNotNone(actual)
            self.assertIsNotNone(expected)
            np.testing.assert_allclose(
                actual.numpy(), expected.numpy(), rtol=1e-4, atol=1e-5
            )

        # Reject silently-zeroed or wrong-scale grads that a loose tolerance
        # against a small reference could otherwise let slip through.
        self.assertGreater(np.abs(x_ref.grad.numpy()).max(), 1e-3)
        self.assertGreater(np.abs(w_ref.grad.numpy()).max(), 1e-3)

    def test_strided_input_forward_matches_reference(self):
        """A sliced (non-contiguous) input must read the correct per-row data."""
        _, weight = self._fixed_inputs()
        wide = paddle.arange(4 * 12, dtype="float32").reshape([4, 12])
        wide = wide * 0.1 - 2.0
        x = wide[:, :8]
        # Confirm the slice is genuinely strided (row stride != column count),
        # otherwise the case would silently degrade to a contiguous input and
        # the stride-handling path would not be exercised.
        self.assertNotEqual(int(x.stride()[0]), int(x.shape[1]))
        out = RMSNormFusionTriton.apply(x, weight, self.EPS)
        ref = _rms_norm_reference(x, weight, self.EPS)
        np.testing.assert_allclose(
            out.numpy(), ref.numpy(), rtol=1e-4, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
