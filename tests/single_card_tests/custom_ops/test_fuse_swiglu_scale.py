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

import subprocess
import sys
import unittest

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import base
from paddle.base import core

from paddlefleet.fusions.fused_swiglu_scale import fused_swiglu_scale_forward


class TestFusedSwiGLUScale(unittest.TestCase):
    def setUp(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA is not available")

        # Set random seed for reproducibility
        np.random.seed(42)
        paddle.seed(42)

    def _load_swiglu_ops(self):
        try:
            import paddlefleet_ops
        except ImportError:
            self.skipTest("SwiGLU custom ops are not available")
        return paddlefleet_ops

    def _require_bf16(self):
        if not core.is_bfloat16_supported(base.CUDAPlace(0)):
            self.skipTest("bf16 not supported")

    def get_reference_impl(self, x, scale):
        """
        Paddle native implementation of SwiGLU + Scale as ground truth.
        """
        # x shape: [B, 2*H] -> chunk -> gate, val
        gate, val = paddle.chunk(x, chunks=2, axis=-1)

        # SwiGLU: silu(gate) * val
        # F.silu(x) = x * sigmoid(x)
        swiglu_out = F.silu(gate) * val

        # Scale: swiglu_out * scale (Broadcast multiplication)
        out = swiglu_out * scale
        return out

    def run_fused_op_test(self, batch_size, hidden_size, dtype_x, dtype_scale):
        # 1. Construct input data
        shape_x = (batch_size, 2 * hidden_size)
        shape_scale = (batch_size, 1)

        # Use a reasonable range to avoid numerical instability
        x_np = np.random.normal(0, 1.0, shape_x).astype("float32")
        scale_np = np.random.uniform(0.5, 1.5, shape_scale).astype("float32")

        # 2. Create Reference Tensors
        if dtype_x == "bfloat16":
            x_ref = paddle.to_tensor(x_np).astype("bfloat16")
        else:
            x_ref = paddle.to_tensor(x_np, dtype=dtype_x)

        if dtype_scale == "bfloat16":
            scale_ref = paddle.to_tensor(scale_np).astype("bfloat16")
        else:
            scale_ref = paddle.to_tensor(scale_np, dtype=dtype_scale)

        x_ref.stop_gradient = False
        scale_ref.stop_gradient = False

        # 3. Create Custom Op Tensors (Independent copies)
        if dtype_x == "bfloat16":
            x_custom = paddle.to_tensor(x_np).astype("bfloat16")
        else:
            x_custom = paddle.to_tensor(x_np, dtype=dtype_x)

        if dtype_scale == "bfloat16":
            scale_custom = paddle.to_tensor(scale_np).astype("bfloat16")
        else:
            scale_custom = paddle.to_tensor(scale_np, dtype=dtype_scale)

        x_custom.stop_gradient = False
        scale_custom.stop_gradient = False

        # 4. Forward Pass
        # Reference implementation
        out_ref = self.get_reference_impl(x_ref, scale_ref)

        # Custom Op implementation
        # Note: The C++ op might return a list [Tensor], handle it if necessary
        ret = fused_swiglu_scale_forward(x_custom, scale_custom)
        out_custom = ret[0] if isinstance(ret, (list, tuple)) else ret

        # 5. Backward Pass
        # Create random output gradient
        grad_np = np.random.random(out_ref.shape).astype("float32")
        if out_ref.dtype == paddle.bfloat16:
            out_grad = paddle.to_tensor(grad_np).astype("bfloat16")
        else:
            out_grad = paddle.to_tensor(grad_np, dtype=out_ref.dtype)

        paddle.autograd.backward([out_ref], [out_grad])
        paddle.autograd.backward([out_custom], [out_grad])

        # 6. Verification
        # Set tolerance: BF16 has lower precision, requiring larger tolerance
        if "bfloat16" in [dtype_x, dtype_scale]:
            rtol, atol = 2e-2, 2e-2
            # Relax tolerance for BF16 gradient accumulation with large shape (Reduction dim > 1024)
            # This accounts for the accumulation error difference between FP32 (Fused) and BF16 (Ref)
            if hidden_size > 1024:
                rtol, atol = 0.1, 0.1
        else:
            rtol, atol = 1e-4, 1e-4

        # Verify Forward Output
        np.testing.assert_allclose(
            out_custom.astype("float32").numpy(),
            out_ref.astype("float32").numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=f"Forward output mismatch: dtype_x={dtype_x}, dtype_scale={dtype_scale}",
        )

        # Verify Input Gradient (dX)
        np.testing.assert_allclose(
            x_custom.grad.astype("float32").numpy(),
            x_ref.grad.astype("float32").numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=f"Gradient X mismatch: dtype_x={dtype_x}, dtype_scale={dtype_scale}",
        )

        # Verify Scale Gradient (dScale)
        np.testing.assert_allclose(
            scale_custom.grad.astype("float32").numpy(),
            scale_ref.grad.astype("float32").numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=f"Gradient Scale mismatch: dtype_x={dtype_x}, dtype_scale={dtype_scale}",
        )

    def test_fused_swiglu_fp32(self):
        self.run_fused_op_test(32, 64, "float32", "float32")

    def test_fused_swiglu_bf16(self):
        self._require_bf16()
        self.run_fused_op_test(32, 64, "bfloat16", "bfloat16")

    def test_fused_swiglu_mixed_precision(self):
        # Test mixed precision: Input=BF16, Scale=FP32
        self._require_bf16()
        self.run_fused_op_test(16, 128, "bfloat16", "float32")

    def test_fused_swiglu_large_shape(self):
        # Test large shape to ensure no index overflow or memory alignment issues
        self._require_bf16()
        self.run_fused_op_test(4, 4096, "bfloat16", "float32")

    def test_fused_swiglu_int32_overflow_numel(self):
        """Forward/backward when numel of x > 2**31.

        Exercises the int64-offset path in VectorizedFusedSwiGLUFwd/Bwd.
        Skipped on hosts without enough free GPU memory.
        """
        self._require_bf16()

        # [rows, 2*hidden] bf16 with rows * 2 * hidden > 2**31.
        # hidden must be divisible by VEC_SIZE=8 for the bf16 kernel.
        # 65536 * 2 * 16392 = 2,148,925,440 > 2,147,483,648.
        rows, hidden = 65536, 16392
        bytes_per_elem = 2
        x_bytes = rows * 2 * hidden * bytes_per_elem
        # x + out + d_x + d_out + scale + tmp ~ 4-5x x_bytes
        try:
            free_bytes, _ = paddle.device.cuda.mem_get_info()
        except Exception:
            try:
                import pynvml

                pynvml.nvmlInit()
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                free_bytes = pynvml.nvmlDeviceGetMemoryInfo(h).free
            except Exception:
                self.skipTest("cannot query GPU memory")
        if free_bytes < 40 * (1 << 30) or free_bytes < x_bytes * 5:
            self.skipTest(
                f"need >=40GB free GPU mem, have {free_bytes / 1e9:.1f}GB"
            )

        paddle.device.cuda.empty_cache()
        # Use small fp32 host arrays then cast on device to limit host RAM.
        x_ref = paddle.randn([rows, 2 * hidden], dtype="float32").astype(
            "bfloat16"
        )
        scale_ref = paddle.uniform([rows, 1], dtype="float32", min=0.5, max=1.5)
        x_ref.stop_gradient = False
        scale_ref.stop_gradient = False

        x_custom = x_ref.detach().clone()
        scale_custom = scale_ref.detach().clone()
        x_custom.stop_gradient = False
        scale_custom.stop_gradient = False

        out_ref = self.get_reference_impl(x_ref, scale_ref)
        ret = fused_swiglu_scale_forward(x_custom, scale_custom)
        out_custom = ret[0] if isinstance(ret, (list, tuple)) else ret

        sample_rows = [0, rows // 2, rows - 1]
        for r in sample_rows:
            np.testing.assert_allclose(
                out_custom[r].astype("float32").numpy(),
                out_ref[r].astype("float32").numpy(),
                rtol=0.1,
                atol=0.1,
                err_msg=f"fwd mismatch at row {r}",
            )

        out_grad = paddle.ones_like(out_ref)
        paddle.autograd.backward([out_ref], [out_grad])
        paddle.autograd.backward([out_custom], [out_grad])
        for r in sample_rows:
            np.testing.assert_allclose(
                x_custom.grad[r].astype("float32").numpy(),
                x_ref.grad[r].astype("float32").numpy(),
                rtol=0.1,
                atol=0.1,
                err_msg=f"dX mismatch at row {r}",
            )

    def _run_grid_stride_test(self, rows, hidden, dtype):
        """Drive rows > kMaxSwiGLUGridSize (65535) with a tiny hidden so the
        grid-stride row loop in VectorizedFusedSwiGLUFwd/Bwd fires multiple
        times per block. Memory footprint is ~ rows * 2*hidden * sizeof(dtype).
        """
        if dtype == "bfloat16" and not core.is_bfloat16_supported(
            base.CUDAPlace(0)
        ):
            self.skipTest("bf16 not supported")

        x_np = np.random.normal(0, 1.0, (rows, 2 * hidden)).astype("float32")
        scale_np = np.random.uniform(0.5, 1.5, (rows, 1)).astype("float32")

        def _to(t):
            if dtype == "bfloat16":
                return paddle.to_tensor(t).astype("bfloat16")
            return paddle.to_tensor(t, dtype=dtype)

        x_ref, scale_ref = _to(x_np), _to(scale_np)
        x_custom, scale_custom = _to(x_np), _to(scale_np)
        for t in (x_ref, scale_ref, x_custom, scale_custom):
            t.stop_gradient = False

        out_ref = self.get_reference_impl(x_ref, scale_ref)
        ret = fused_swiglu_scale_forward(x_custom, scale_custom)
        out_custom = ret[0] if isinstance(ret, (list, tuple)) else ret

        rtol, atol = (5e-2, 5e-2) if dtype == "bfloat16" else (1e-4, 1e-4)
        np.testing.assert_allclose(
            out_custom.astype("float32").numpy(),
            out_ref.astype("float32").numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=f"grid-stride fwd mismatch (rows={rows}, dtype={dtype})",
        )

        out_grad = paddle.ones_like(out_ref)
        paddle.autograd.backward([out_ref], [out_grad])
        paddle.autograd.backward([out_custom], [out_grad])
        np.testing.assert_allclose(
            x_custom.grad.astype("float32").numpy(),
            x_ref.grad.astype("float32").numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=f"grid-stride dX mismatch (rows={rows}, dtype={dtype})",
        )
        np.testing.assert_allclose(
            scale_custom.grad.astype("float32").numpy(),
            scale_ref.grad.astype("float32").numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=f"grid-stride dScale mismatch (rows={rows}, dtype={dtype})",
        )

    def test_fused_swiglu_grid_stride_loop_bf16(self):
        # rows > kMaxSwiGLUGridSize (65535) -> each block iterates the row
        # loop at least twice. hidden=8 is the bf16 VEC_SIZE so launch is
        # 1 thread per row's hidden lane; tiny memory footprint.
        self._run_grid_stride_test(rows=65536, hidden=8, dtype="bfloat16")

    def test_fused_swiglu_grid_stride_loop_fp32(self):
        # fp32 VEC_SIZE=4; rows just above the cap forces 2-iteration loop
        # only on the first 1 (=65536-65535) blocks while the rest run once.
        self._run_grid_stride_test(rows=65536, hidden=8, dtype="float32")

    def test_fused_swiglu_grid_stride_loop_far_above_cap(self):
        # rows ~ 3x cap so every block iterates the row loop multiple times,
        # and the trailing __syncthreads() between iterations is exercised
        # heavily on the bwd shared_sum reduction.
        self._run_grid_stride_test(rows=200000, hidden=8, dtype="bfloat16")

    def test_fused_swiglu_clamp_grid_stride_loop(self):
        """Same grid-stride coverage but for the clamp + weighted bwd path
        (VectorizedFusedSwiGLUWeightedBwd kernel)."""
        self._require_bf16()
        ops = self._load_swiglu_ops()

        rows, hidden = 65536, 8
        x_np = np.random.normal(0, 1.0, (rows, 2 * hidden)).astype("float32")
        probs_np = np.random.uniform(0.1, 0.9, (rows,)).astype("float32")
        dout_np = np.random.normal(0, 1.0, (rows, hidden)).astype("float32")

        x = paddle.to_tensor(x_np).astype("bfloat16")
        probs = paddle.to_tensor(probs_np).astype("bfloat16")
        dout = paddle.to_tensor(dout_np).astype("bfloat16")

        clamp_value = 7.0
        dx, dprobs, out = ops.fused_swiglu_weighted_clamp_bwd(
            x, probs, dout, clamp_value
        )

        # Reference: clamp(g, cv); clamp(v, -cv, cv); silu(g_eff)*v_eff
        cv = clamp_value
        gate, val = paddle.chunk(x.astype("float32"), 2, axis=-1)
        g_eff = paddle.minimum(gate, paddle.full_like(gate, cv))
        v_eff = paddle.clip(val, -cv, cv)
        silu_g = F.silu(g_eff)
        swiglu = silu_g * v_eff
        out_ref = (swiglu * probs.astype("float32").unsqueeze(-1)).astype(
            "bfloat16"
        )
        # dprobs[row] = sum_h(dout * silu(clamp(g)) * clamp(v))
        dprobs_ref = paddle.sum(
            dout.astype("float32") * swiglu, axis=-1, keepdim=True
        ).astype("bfloat16")

        # The point of this test is exercising the kernel's row grid-stride
        # loop and the shared_sum cross-iteration __syncthreads in the
        # dprobs reduction. Verifying out alone would not catch a broken
        # cross-iteration shared_sum sync, since out is computed without
        # any block-wide reduction.
        self.assertEqual(out.shape, [rows, hidden])
        self.assertEqual(dx.shape, [rows, 2 * hidden])
        self.assertEqual(dprobs.shape, [rows, 1])
        np.testing.assert_allclose(
            out.astype("float32").numpy(),
            out_ref.astype("float32").numpy(),
            rtol=5e-2,
            atol=5e-2,
            err_msg="weighted-clamp grid-stride fwd output mismatch",
        )
        np.testing.assert_allclose(
            dprobs.astype("float32").numpy(),
            dprobs_ref.astype("float32").numpy(),
            rtol=5e-2,
            atol=5e-2,
            err_msg="weighted-clamp grid-stride dprobs mismatch",
        )

    def test_fused_swiglu_clamp_rejects_shape_mismatch(self):
        """Reject mismatched auxiliary shapes before launching the kernel."""
        self._require_bf16()
        ops = self._load_swiglu_ops()

        x = paddle.zeros([2, 16], dtype="bfloat16")
        valid_dout = paddle.zeros([2, 8], dtype="bfloat16")
        cases = {
            "zero prob rows": (
                paddle.zeros([0, 1], dtype="bfloat16"),
                valid_dout,
                "Probs must contain exactly one value per X row",
            ),
            "zero prob hidden": (
                paddle.zeros([2, 0], dtype="bfloat16"),
                valid_dout,
                r"Probs must have shape \[rows\] or \[rows, 1\]",
            ),
            "zero dout rows": (
                paddle.ones([2, 1], dtype="bfloat16"),
                paddle.zeros([0, 8], dtype="bfloat16"),
                r"DOut must have shape \[X.shape\[0\], X.shape\[1\] / 2\]",
            ),
        }
        for name, (probs, d_out, message) in cases.items():
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(OSError, message),
            ):
                ops.fused_swiglu_weighted_clamp_bwd(x, probs, d_out, 7.0)

    def test_fused_swiglu_handles_empty_rows_and_hidden(self):
        """Return correctly shaped outputs for empty rows and hidden width."""
        self._require_bf16()
        ops = self._load_swiglu_ops()

        empty_hidden_forward = ops.fused_swiglu_scale(
            paddle.zeros([2, 0], dtype="bfloat16"),
            paddle.ones([2, 1], dtype="bfloat16"),
        )
        self.assertEqual(empty_hidden_forward.shape, [2, 0])

        empty_rows = ops.fused_swiglu_weighted_clamp_bwd(
            paddle.zeros([0, 16], dtype="bfloat16"),
            paddle.zeros([0, 1], dtype="bfloat16"),
            paddle.zeros([0, 8], dtype="bfloat16"),
            7.0,
        )
        self.assertEqual(
            [tensor.shape for tensor in empty_rows], [[0, 16], [0, 1], [0, 8]]
        )

        empty_hidden = ops.fused_swiglu_weighted_clamp_bwd(
            paddle.zeros([2, 0], dtype="bfloat16"),
            paddle.ones([2, 1], dtype="bfloat16"),
            paddle.zeros([2, 0], dtype="bfloat16"),
            7.0,
        )
        self.assertEqual(
            [tensor.shape for tensor in empty_hidden], [[2, 0], [2, 1], [2, 0]]
        )
        np.testing.assert_array_equal(
            empty_hidden[1].astype("float32").numpy(), np.zeros([2, 1])
        )

    def test_fused_swiglu_ops_reject_invalid_inputs(self):
        """All five entry points reject inputs their packed kernels cannot read."""
        self._require_bf16()
        ops = self._load_swiglu_ops()

        x = paddle.zeros([2, 16], dtype="bfloat16")
        scale = paddle.ones([2, 1], dtype="bfloat16")
        dout = paddle.ones([2, 8], dtype="bfloat16")
        cases = [
            (
                "forward X rank",
                lambda: ops.fused_swiglu_scale(
                    paddle.zeros([2, 16, 1], dtype="bfloat16"), scale
                ),
                r"X must have shape \[rows, 2 \* hidden_size\]",
            ),
            (
                "forward X odd width",
                lambda: ops.fused_swiglu_scale(
                    paddle.zeros([2, 15], dtype="bfloat16"), scale
                ),
                "last dimension of X must be divisible by 2",
            ),
            (
                "forward scale shape",
                lambda: ops.fused_swiglu_scale(
                    x, paddle.ones([2, 2], dtype="bfloat16")
                ),
                r"Scale must have shape \[rows\] or \[rows, 1\]",
            ),
            (
                "backward dout dtype",
                lambda: ops.fused_swiglu_scale_bwd(
                    x, scale, dout.astype("float32")
                ),
                "DOut dtype must match X dtype",
            ),
            (
                "backward dout shape",
                lambda: ops.fused_swiglu_scale_bwd(
                    x,
                    scale,
                    paddle.ones([2, 4], dtype="bfloat16"),
                ),
                r"DOut must have shape \[X.shape\[0\], X.shape\[1\] / 2\]",
            ),
            (
                "CPU inputs",
                lambda: ops.fused_swiglu_scale(
                    paddle.zeros([2, 16], dtype="bfloat16", device="cpu"),
                    paddle.ones([2, 1], dtype="bfloat16", device="cpu"),
                ),
                "expects GPU inputs",
            ),
            (
                "forward scale dtype",
                lambda: ops.fused_swiglu_scale(
                    x, paddle.ones([2, 1], dtype="float16")
                ),
                "Scale must be bfloat16 or float32",
            ),
            (
                "clamp forward value",
                lambda: ops.fused_swiglu_scale_clamp(x, scale, 0.0),
                "clamp_value must be finite and greater than zero",
            ),
            (
                "clamp backward vector width",
                lambda: ops.fused_swiglu_scale_clamp_bwd(
                    paddle.zeros([2, 20], dtype="bfloat16"),
                    scale,
                    paddle.ones([2, 10], dtype="bfloat16"),
                    7.0,
                ),
                "hidden_size must be divisible by the vector width",
            ),
            (
                "weighted clamp value",
                lambda: ops.fused_swiglu_weighted_clamp_bwd(
                    x, scale, dout, float("inf")
                ),
                "clamp_value must be finite and greater than zero",
            ),
            (
                "weighted fp32 with bf16 probs",
                lambda: ops.fused_swiglu_weighted_clamp_bwd(
                    x.astype("float32"),
                    scale,
                    dout.astype("float32"),
                    7.0,
                ),
                "float32 X requires float32 Probs",
            ),
            (
                "float16 forward",
                lambda: ops.fused_swiglu_scale(
                    x.astype("float16"), scale.astype("float32")
                ),
                "X must be bfloat16 or float32",
            ),
        ]
        for name, call, message in cases:
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(OSError, message),
            ):
                call()

    def test_fused_swiglu_accepts_non_contiguous_x(self):
        """Materialize non-contiguous X before packed kernel execution."""
        self._require_bf16()
        ops = self._load_swiglu_ops()
        x = paddle.randn([2, 32], dtype="float32").astype("bfloat16")[:, ::2]
        self.assertFalse(x.is_contiguous())
        probs = paddle.ones([2, 1], dtype="bfloat16")
        dout = paddle.ones([2, 8], dtype="bfloat16")

        result = ops.fused_swiglu_weighted_clamp_bwd(x, probs, dout, 7.0)
        expected = ops.fused_swiglu_weighted_clamp_bwd(
            x.contiguous(), probs, dout, 7.0
        )
        for actual, reference in zip(result, expected):
            np.testing.assert_array_equal(
                actual.astype("float32").numpy(),
                reference.astype("float32").numpy(),
            )

    def test_fused_swiglu_accepts_non_contiguous_scale(self):
        """Materialize non-contiguous Scale before packed kernel execution."""
        self._require_bf16()
        ops = self._load_swiglu_ops()
        x = paddle.ones([2, 16], dtype="bfloat16")
        scale = paddle.ones([2, 2], dtype="bfloat16")[:, :1]
        self.assertFalse(scale.is_contiguous())

        out = ops.fused_swiglu_scale(x, scale)
        expected = ops.fused_swiglu_scale(x, scale.contiguous())
        np.testing.assert_array_equal(
            out.astype("float32").numpy(),
            expected.astype("float32").numpy(),
        )

    def test_fused_swiglu_accepts_non_contiguous_probs(self):
        """Materialize non-contiguous Probs before packed kernel execution."""
        self._require_bf16()
        ops = self._load_swiglu_ops()
        x = paddle.ones([2, 16], dtype="bfloat16")
        dout = paddle.ones([2, 8], dtype="bfloat16")
        probs = paddle.ones([2, 2], dtype="bfloat16")[:, :1]
        self.assertFalse(probs.is_contiguous())

        result = ops.fused_swiglu_weighted_clamp_bwd(x, probs, dout, 7.0)
        expected = ops.fused_swiglu_weighted_clamp_bwd(
            x, probs.contiguous(), dout, 7.0
        )
        for actual, reference in zip(result, expected):
            np.testing.assert_array_equal(
                actual.astype("float32").numpy(),
                reference.astype("float32").numpy(),
            )

    def test_fused_swiglu_accepts_empty_rows_with_unaligned_hidden(self):
        """Skip packed vector-width checks when no rows reach the kernel."""
        ops = self._load_swiglu_ops()
        out = ops.fused_swiglu_scale(
            paddle.zeros([0, 4], dtype="float32"),
            paddle.zeros([0], dtype="float32"),
        )
        self.assertEqual(out.shape, [0, 2])

    def test_fused_swiglu_accepts_one_dimensional_scale(self):
        """Preserve rank-1 Scale shape on the non-clamp backward path."""
        self._require_bf16()
        ops = self._load_swiglu_ops()
        x = paddle.ones([2, 16], dtype="bfloat16")
        scale = paddle.ones([2], dtype="bfloat16")
        dout = paddle.ones([2, 8], dtype="bfloat16")
        out = ops.fused_swiglu_scale(x, scale)
        d_x, d_scale = ops.fused_swiglu_scale_bwd(x, scale, dout)
        self.assertEqual(out.shape, [2, 8])
        self.assertEqual(d_x.shape, [2, 16])
        self.assertEqual(d_scale.shape, [2])

    def test_fused_swiglu_preserves_unknown_output_dimension(self):
        """Keep an unknown hidden dimension unknown during shape inference."""
        ops = self._load_swiglu_ops()
        paddle.enable_static()
        try:
            main = paddle.static.Program()
            with paddle.static.program_guard(main):
                x = paddle.static.data("x", [-1, -1], dtype="bfloat16")
                scale = paddle.static.data("scale", [-1, 1], dtype="bfloat16")
                dout = paddle.static.data("dout", [-1, -1], dtype="bfloat16")

                out = ops.fused_swiglu_scale_clamp(x, scale, 7.0)
                weighted = ops.fused_swiglu_weighted_clamp_bwd(
                    x, scale, dout, 7.0
                )

                self.assertEqual(out.shape, [-1, -1])
                self.assertEqual(weighted[2].shape, [-1, -1])
        finally:
            paddle.disable_static()

    def test_fused_swiglu_rejects_invalid_static_x_shape(self):
        """Reject invalid X rank and odd width during shape inference."""
        self._load_swiglu_ops()
        cases = [
            (
                "clamp forward rank",
                [-1],
                "ops.fused_swiglu_scale_clamp(x, scale, 7.0)",
                r"X must have shape \[rows, 2 \* hidden_size\]",
            ),
            (
                "weighted backward odd width",
                [-1, 15],
                "ops.fused_swiglu_weighted_clamp_bwd(x, scale, dout, 7.0)",
                "last dimension of X must be divisible by 2",
            ),
        ]

        for name, x_shape, call, message in cases:
            with self.subTest(name=name):
                script = f"""
import paddle
import paddlefleet_ops as ops
paddle.enable_static()
main = paddle.static.Program()
with paddle.static.program_guard(main):
    x = paddle.static.data("x", {x_shape!r}, dtype="bfloat16")
    scale = paddle.static.data("scale", [-1, 1], dtype="bfloat16")
    dout = paddle.static.data("dout", [-1, -1], dtype="bfloat16")
    {call}
"""
                result = subprocess.run(
                    [sys.executable, "-c", script],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=120,
                )
                output = result.stdout + result.stderr
                self.assertNotEqual(result.returncode, 0, output)
                self.assertRegex(output, message)

    def test_fused_swiglu_backward_materializes_non_contiguous_dout(self):
        """Strided framework gradients must match explicitly contiguous DOut."""
        self._require_bf16()
        ops = self._load_swiglu_ops()
        x = paddle.randn([2, 16], dtype="float32").astype("bfloat16")
        scale = paddle.randn([2], dtype="float32").astype("bfloat16")
        non_contiguous_dout = (
            paddle.randn([8, 2], dtype="float32")
            .astype("bfloat16")
            .transpose([1, 0])
        )
        if non_contiguous_dout.is_contiguous():
            self.skipTest("transpose produced a contiguous tensor")
        dout_ref = non_contiguous_dout.contiguous()

        dx_a, ds_a = ops.fused_swiglu_scale_bwd(x, scale, non_contiguous_dout)
        dx_b, ds_b = ops.fused_swiglu_scale_bwd(x, scale, dout_ref)
        np.testing.assert_allclose(
            dx_a.astype("float32").numpy(),
            dx_b.astype("float32").numpy(),
            rtol=0,
            atol=0,
        )
        np.testing.assert_allclose(
            ds_a.astype("float32").numpy(),
            ds_b.astype("float32").numpy(),
            rtol=0,
            atol=0,
        )

        probs = paddle.randn([2, 1], dtype="float32").astype("bfloat16")
        weighted_a = ops.fused_swiglu_weighted_clamp_bwd(
            x, probs, non_contiguous_dout, 7.0
        )
        weighted_b = ops.fused_swiglu_weighted_clamp_bwd(
            x, probs, dout_ref, 7.0
        )
        for result_a, result_b in zip(weighted_a, weighted_b):
            np.testing.assert_allclose(
                result_a.astype("float32").numpy(),
                result_b.astype("float32").numpy(),
                rtol=0,
                atol=0,
            )

        dx_a, ds_a = ops.fused_swiglu_scale_clamp_bwd(
            x, scale, non_contiguous_dout, 7.0
        )
        dx_b, ds_b = ops.fused_swiglu_scale_clamp_bwd(x, scale, dout_ref, 7.0)
        np.testing.assert_allclose(
            dx_a.astype("float32").numpy(),
            dx_b.astype("float32").numpy(),
            rtol=0,
            atol=0,
        )
        np.testing.assert_allclose(
            ds_a.astype("float32").numpy(),
            ds_b.astype("float32").numpy(),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
