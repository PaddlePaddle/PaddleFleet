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

"""Behavior tests for the fused sigmoid-gate Triton op (branch set 3).

Module under test: ``paddlefleet.triton_ops.sigmoid_gate_fusion`` (repository
module map: "计算优化" / Fused Ops). This file deliberately exercises branches
that existence/structure coverage of this module does NOT touch:

  * The forward input-validation contract of ``SigmoidGateFusionTriton.forward``
    -- three distinct ``assert`` branches (shape mismatch, dtype mismatch,
    unsupported dtype) that run in pure Python *before* any GPU kernel is
    launched. These are the only CPU-observable pure logic in the module and
    need paddle but not a GPU.
  * The forward fusion value ``out = attn_out * sigmoid(gate)`` and the
    backward gradient pair ``d_attn = dout * sigmoid(gate)`` and
    ``d_gate = dout * attn_out * sigmoid(gate) * (1 - sigmoid(gate))``
    (``fused_sigmoid_gate_fwd_kernel`` / ``fused_sigmoid_gate_bwd_kernel``).

Every expected value is hand-derived from the documented sigmoid-gate
semantics with an independent NumPy reference, never from the kernel output.

Environment note: the kernels are Triton/CUDA only and have no CPU
implementation, and the module imports ``paddle`` at top level. When paddle is
not installed the whole file is skipped honestly rather than fake-passing; the
numeric class additionally requires a CUDA GPU with an active Triton runtime.
"""

import unittest
from types import SimpleNamespace

try:
    import numpy as np
    import paddle
    import triton

    _IMPORT_OK = True
    _IMPORT_ERR = ""
except ImportError as exc:  # missing dependency -> honest skip, not swallowed
    np = None
    paddle = None
    triton = None
    _IMPORT_OK = False
    _IMPORT_ERR = str(exc)

if _IMPORT_OK:
    # A genuine API/compile error here must surface, so only ImportError is
    # treated as a "missing dependency" skip reason (not a broad except).
    try:
        from paddlefleet.triton_ops.sigmoid_gate_fusion import (
            SigmoidGateFusionTriton,
        )
    except ImportError as exc:
        _IMPORT_OK = False
        _IMPORT_ERR = str(exc)


def _gpu_ready():
    """True only when paddle+triton import and a CUDA GPU with an active
    Triton runtime is present. Hardware absence is a skip, not a pass."""
    if not _IMPORT_OK:
        return False
    if not paddle.is_compiled_with_cuda():
        return False
    try:
        triton.runtime.driver.active.get_current_device()
    except Exception:
        # No usable GPU/Triton runtime -> genuine hardware/runtime absence.
        return False
    return True


_IMPORT_SKIP = (
    "sigmoid_gate_fusion requires paddle (module imports paddle at top "
    "level); input validation runs on CPU but paddle must be installed "
    "(import error: %s)" % (_IMPORT_ERR or "none",)
)

_GPU_SKIP = (
    "fused sigmoid-gate kernels require paddle + triton and a CUDA GPU with "
    "an active Triton runtime; there is no CPU implementation to validate "
    "(import error: %s)" % (_IMPORT_ERR or "none",)
)


@unittest.skipUnless(_IMPORT_OK, _IMPORT_SKIP)
class TestSigmoidGateFusionValidation(unittest.TestCase):
    """The forward input guards fire in pure Python before the GPU kernel.

    ``forward`` is the real production method under test; a dummy ctx is
    unused because every assertion below short-circuits before ``ctx`` is
    read or any kernel is launched, so these run on CPU tensors.
    """

    def test_forward_rejects_shape_mismatch(self):
        # First guard: attn_out.shape == gate.shape. Distinct shapes must be
        # rejected before dtype/support checks or any kernel launch.
        attn_out = paddle.zeros([2, 3], dtype="float32")
        gate = paddle.zeros([2, 4], dtype="float32")
        with self.assertRaisesRegex(AssertionError, "same shape"):
            SigmoidGateFusionTriton.forward(SimpleNamespace(), attn_out, gate)

    def test_forward_rejects_dtype_mismatch(self):
        # Second guard: shapes equal but dtypes differ -> must raise on the
        # dtype check, not the shape check.
        attn_out = paddle.zeros([2, 3], dtype="float32")
        gate = paddle.zeros([2, 3], dtype="float16")
        with self.assertRaisesRegex(AssertionError, "same dtype"):
            SigmoidGateFusionTriton.forward(SimpleNamespace(), attn_out, gate)

    def test_forward_rejects_unsupported_dtype(self):
        # Third guard: shapes and dtypes match but the dtype is outside
        # {fp16, bf16, fp32}. float64 passes the first two asserts and must
        # fail only on the supported-dtype check.
        attn_out = paddle.zeros([2, 3], dtype="float64")
        gate = paddle.zeros([2, 3], dtype="float64")
        with self.assertRaisesRegex(AssertionError, "Unsupported dtype"):
            SigmoidGateFusionTriton.forward(SimpleNamespace(), attn_out, gate)


@unittest.skipUnless(_gpu_ready(), _GPU_SKIP)
class TestSigmoidGateFusionNumeric(unittest.TestCase):
    """Forward value and backward gradient pair against a NumPy reference."""

    def setUp(self):
        paddle.device.set_device("gpu:0")
        # Non-uniform, sign-varied inputs so a swapped operand, a dropped
        # (1 - sigmoid) factor, or a forward/backward mix-up changes results.
        self._attn = np.array(
            [[1.0, -2.0, 0.5], [3.0, 0.25, -1.5]], dtype=np.float32
        )
        self._gate = np.array(
            [[0.0, 1.0, -1.0], [2.0, -0.5, 0.5]], dtype=np.float32
        )
        self._sig = 1.0 / (1.0 + np.exp(-self._gate))

    def test_forward_matches_attn_times_sigmoid_gate(self):
        attn = paddle.to_tensor(self._attn)
        gate = paddle.to_tensor(self._gate)
        out = SigmoidGateFusionTriton.apply(attn, gate)
        expected = self._attn * self._sig
        # Guard against a degenerate reference (all-0.5 sigmoid would hide a
        # missing gate term): the fixture spans sigmoid values away from 0.5.
        self.assertGreater(float(np.abs(self._sig - 0.5).max()), 0.1)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-4, atol=1e-5)

    def test_backward_grads_match_hand_derived_reference(self):
        attn = paddle.to_tensor(self._attn)
        gate = paddle.to_tensor(self._gate)
        attn.stop_gradient = False
        gate.stop_gradient = False
        # Non-uniform upstream gradient exposes elementwise mis-routing.
        dout_np = np.array(
            [[0.5, -1.0, 2.0], [1.5, -0.25, 0.75]], dtype=np.float32
        )
        out = SigmoidGateFusionTriton.apply(attn, gate)
        out.backward(paddle.to_tensor(dout_np))

        d_attn_ref = dout_np * self._sig
        d_gate_ref = dout_np * self._attn * self._sig * (1.0 - self._sig)

        self.assertIsNotNone(attn.grad)
        self.assertIsNotNone(gate.grad)
        # The two grads are genuinely distinct, so a copy-paste bug is caught.
        self.assertGreater(float(np.abs(d_attn_ref - d_gate_ref).max()), 1e-3)
        np.testing.assert_allclose(
            attn.grad.numpy(), d_attn_ref, rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            gate.grad.numpy(), d_gate_ref, rtol=1e-4, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
