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

"""Behavior tests for the fused sigmoid-gate Triton op (branch set 2).

Module under test: ``paddlefleet.triton_ops.sigmoid_gate_fusion`` (repository
module map: "计算优化" / Fused Ops). ``SigmoidGateFusionTriton`` computes
``out = attn_out * sigmoid(gate)`` on the GPU via a Triton kernel, with a
matching deterministic backward.

This file deliberately exercises the *input-validation contract* of
``SigmoidGateFusionTriton.forward`` -- a set of branches distinct from the
numeric forward/backward math. Before any tensor is allocated or any Triton
kernel is launched, ``forward`` runs three guards in a fixed order:

  1. ``attn_out.shape == gate.shape``           (shape agreement)
  2. ``attn_out.dtype == gate.dtype``           (dtype agreement)
  3. ``attn_out.dtype in {fp16, bf16, fp32}``   (supported dtype)

These guards are pure CPU-observable logic: they raise ``AssertionError``
*before* ``paddle.empty_like`` / the kernel launch, so they can be validated
on a CPU build with no GPU present. The expected raising behaviour and the
guard *ordering* (shape checked before dtype, dtype-equality before
dtype-support) are hand-derived from the documented contract, and each test
uses inputs that discriminate which specific guard fired.

Environment note: the numeric forward/backward runs only through the Triton
CUDA kernel and has no CPU implementation; it is not asserted here. The module
imports ``paddle`` and ``triton`` at top level, so when either is missing the
whole suite is skipped honestly rather than fake-passing. Only ``ImportError``
is treated as a missing-dependency skip; a genuine API/compile error surfaces.
"""

import unittest

try:
    import paddle
    import triton  # noqa: F401

    _IMPORT_OK = True
    _IMPORT_ERR = ""
except ImportError as exc:  # missing dependency -> honest skip, not swallowed
    paddle = None
    _IMPORT_OK = False
    _IMPORT_ERR = str(exc)

if _IMPORT_OK:
    # A genuine API/compile error must surface; only ImportError -> skip.
    try:
        from paddlefleet.triton_ops.sigmoid_gate_fusion import (
            SigmoidGateFusionTriton,
        )
    except ImportError as exc:
        _IMPORT_OK = False
        _IMPORT_ERR = str(exc)


_SKIP_REASON = (
    "sigmoid_gate_fusion requires importable paddle + triton + the "
    "paddlefleet.triton_ops.sigmoid_gate_fusion module; the input-validation "
    "guards are CPU-observable and need no GPU, but the module cannot be "
    "imported here (import error: %s)" % (_IMPORT_ERR or "none",)
)


@unittest.skipUnless(_IMPORT_OK, _SKIP_REASON)
class TestSigmoidGateFusionValidation2(unittest.TestCase):
    """Guard-branch contract of ``SigmoidGateFusionTriton.forward``.

    These assertions run before the Triton kernel, so they exercise real
    production logic on CPU tensors without launching any GPU code.
    """

    def setUp(self):
        # Keep the validation path on CPU: the guards raise before any kernel
        # launch, so no GPU device is required or selected here.
        paddle.device.set_device("cpu")

    def test_shape_mismatch_rejected_before_kernel(self):
        # forward asserts attn_out.shape == gate.shape. Distinguishable sizes
        # ([2, 2] vs [2, 3]) must be rejected, and the guard fires before any
        # allocation/kernel launch, so this is observable on CPU.
        attn_out = paddle.zeros([2, 2], dtype="float32")
        gate = paddle.zeros([2, 3], dtype="float32")
        with self.assertRaises(AssertionError) as cm:
            SigmoidGateFusionTriton.apply(attn_out, gate)
        # Confirm it is the shape guard, not a later one.
        self.assertIn("shape", str(cm.exception))

    def test_dtype_mismatch_rejected(self):
        # Shapes agree, dtypes differ (fp32 vs fp16): the second guard fires.
        attn_out = paddle.zeros([2, 2], dtype="float32")
        gate = paddle.zeros([2, 2], dtype="float16")
        with self.assertRaises(AssertionError) as cm:
            SigmoidGateFusionTriton.apply(attn_out, gate)
        msg = str(cm.exception)
        # It is the dtype-equality guard, not the supported-dtype guard.
        self.assertIn("dtype", msg)
        self.assertNotIn("Unsupported", msg)

    def test_unsupported_dtype_rejected(self):
        # Shapes and dtypes agree but the dtype (float64) is outside the
        # supported {fp16, bf16, fp32} set: the third guard fires.
        attn_out = paddle.zeros([2, 2], dtype="float64")
        gate = paddle.zeros([2, 2], dtype="float64")
        with self.assertRaises(AssertionError) as cm:
            SigmoidGateFusionTriton.apply(attn_out, gate)
        self.assertIn("Unsupported dtype", str(cm.exception))

    def test_shape_guard_precedes_dtype_guards(self):
        # When BOTH shape and dtype disagree, the documented ordering requires
        # the shape guard to fire first. A discriminating input: [2, 2] fp32
        # vs [2, 3] fp16. If the order were reversed we would instead see a
        # dtype/Unsupported message.
        attn_out = paddle.zeros([2, 2], dtype="float32")
        gate = paddle.zeros([2, 3], dtype="float16")
        with self.assertRaises(AssertionError) as cm:
            SigmoidGateFusionTriton.apply(attn_out, gate)
        msg = str(cm.exception)
        self.assertIn("shape", msg)
        self.assertNotIn("Unsupported dtype", msg)


if __name__ == "__main__":
    unittest.main()
