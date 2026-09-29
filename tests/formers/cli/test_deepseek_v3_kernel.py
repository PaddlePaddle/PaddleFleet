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

"""Behavior tests for the DeepSeek-V3 pretrain FP8 Triton kernels.

Module under test:
``paddlefleet.cli.train.deepseek_v3_pretrain.kernel`` -- the block-wise FP8
helpers ``act_quant`` (activation quantization), ``weight_dequant`` (weight
de-quantization) and ``fp8_gemm`` (scaled FP8 matmul). This is the "computation
optimization" layer (低精度/融合算子): the Python wrappers derive layout, scale
shapes and launch grids on the host, then dispatch a ``@triton.jit`` kernel that
only runs on a real GPU.

Two clearly separated evidence levels, matching the unit-test rules:

* CPU-observable host logic (no GPU needed). The support-condition guards
  (contiguity / divisibility / rank asserts) execute *before* any tensor
  allocation or kernel launch, so they are exercised directly on CPU with an
  independent, hand-derived expectation about *which* guard must fire (matched
  by message, not just ``AssertionError``). For ``fp8_gemm`` the layout
  derivation -- transpose of ``b``/``b_s``, the ``M``/``N``/``K`` extraction and
  the output shape / launch grid -- is verified by replacing the *non-tested*
  ``fp8_gemm_kernel`` collaborator with a spy that records the real arguments
  the wrapper computes. This proves the dispatch/layout contract only; it does
  NOT prove the GPU kernel numerics (explicitly declared).

* Real GPU kernel numerics. ``act_quant`` / ``weight_dequant`` / ``fp8_gemm``
  are compared against independent hand-derived references, but FP8 e4m3
  arithmetic needs an actual device (sm89+ for ``fp8e4nv``). Those tests are
  guarded with ``skipUnless``; when no such device is present they are recorded
  as "not run", never counted as passing, and never faked with mock call counts.

Importing the production module pulls the ``paddlefleet`` package (needs Paddle)
and Triton; when either dependency is absent the import fails and every test
skips (recorded, not silently passed). No production code is modified here.
"""

import unittest
from unittest import mock

import numpy as np

try:
    import paddle

    from paddlefleet.cli.train.deepseek_v3_pretrain import kernel as dsv3_kernel

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # Paddle / package missing.
    paddle = None
    dsv3_kernel = None
    _IMPORT_ERROR = exc
except (
    RuntimeError
) as exc:  # kernel.py re-raises Triton-missing as RuntimeError.
    if "Triton" not in str(exc):
        raise
    paddle = None
    dsv3_kernel = None
    _IMPORT_ERROR = exc

_KERNEL_AVAILABLE = dsv3_kernel is not None
_IMPORT_SKIP_REASON = (
    ""
    if _KERNEL_AVAILABLE
    else f"kernel import failed (dependency unavailable): {_IMPORT_ERROR!r}"
)


def _detect_gpu_fp8():
    """Return (ready, reason) for running real FP8 e4m3 Triton kernels.

    Only precise, well-understood conditions map to "not ready"; unexpected
    errors are surfaced rather than masqueraded as a missing dependency.
    """
    if not _KERNEL_AVAILABLE:
        return False, _IMPORT_SKIP_REASON
    if not paddle.device.is_compiled_with_cuda():
        return False, "Paddle is not compiled with CUDA"
    if paddle.device.cuda.device_count() == 0:
        return False, "no CUDA device visible"
    try:
        import triton

        dev = triton.runtime.driver.active.get_current_device()
        cc = triton.runtime.driver.active.get_device_properties(dev)["cc"]
    except (ImportError, ModuleNotFoundError) as exc:
        return False, f"Triton unavailable: {exc!r}"
    except (RuntimeError, AttributeError, KeyError) as exc:
        return False, f"Triton GPU driver unavailable: {exc!r}"
    # fp8e4nv (paddle.float8_e4m3fn) requires Ada/Hopper (sm89+).
    if cc < 89:
        return False, f"fp8e4nv needs sm89+, got sm{cc}"
    return True, ""


_GPU_FP8_READY, _GPU_SKIP_REASON = _detect_gpu_fp8()


def _cdiv(a, b):
    """Independent ceil-division, not sourced from triton/the module."""
    return (a + b - 1) // b


class _KernelLaunchSpy:
    """Stand-in for the (non-tested) ``@triton.jit`` launcher.

    Triton launches via ``kernel[grid](*args, **kwargs)``; this records the grid
    and the concrete arguments the *wrapper* computed, so the host-side layout
    derivation can be observed without a GPU. It deliberately does no math: the
    kernel numerics are covered separately on a real device.
    """

    def __init__(self):
        self.grid = None
        self.args = None
        self.kwargs = None
        self.call_count = 0

    def __getitem__(self, grid):
        self.grid = grid

        def _launch(*args, **kwargs):
            self.call_count += 1
            self.args = args
            self.kwargs = kwargs

        return _launch


@unittest.skipUnless(_KERNEL_AVAILABLE, _IMPORT_SKIP_REASON)
class TestActQuantHostGuards(unittest.TestCase):
    """CPU-observable support-condition guards of ``act_quant``.

    Both asserts run before any allocation/launch, so they are real host
    behavior. The regex pins *which* guard fired (divisibility vs contiguity),
    which a bare ``assertRaises(AssertionError)`` would not distinguish.
    """

    def test_last_dim_not_divisible_rejected(self):
        # 100 % 128 != 0; tensor is contiguous so the divisibility guard is the
        # one that must fire (independently derived from the assert message).
        x = paddle.zeros([2, 100], dtype="float32")
        self.assertTrue(x.is_contiguous())
        with self.assertRaisesRegex(AssertionError, "divisible by block_size"):
            dsv3_kernel.act_quant(x, block_size=128)

    def test_block_size_appears_in_divisibility_message(self):
        # A non-default block_size must be reflected in the guard message,
        # proving the argument is consumed by the check (not a hard-coded 128).
        x = paddle.zeros([1, 96], dtype="float32")  # 96 % 64 != 0
        with self.assertRaisesRegex(AssertionError, r"block_size=64"):
            dsv3_kernel.act_quant(x, block_size=64)

    def test_non_contiguous_input_rejected(self):
        # Transpose yields a non-contiguous view whose last dim (256) *is*
        # divisible by 128, so the contiguity guard is isolated as the cause.
        base = paddle.arange(256 * 128, dtype="float32").reshape([256, 128])
        x = paddle.transpose(base, [1, 0])  # shape [128, 256]
        if x.is_contiguous():
            self.skipTest(
                "transpose produced a contiguous tensor on this build"
            )
        self.assertEqual(x.shape, [128, 256])
        with self.assertRaisesRegex(AssertionError, "contiguous"):
            dsv3_kernel.act_quant(x, block_size=128)


@unittest.skipUnless(_KERNEL_AVAILABLE, _IMPORT_SKIP_REASON)
class TestWeightDequantHostGuards(unittest.TestCase):
    """CPU-observable support-condition guards of ``weight_dequant``."""

    def test_three_dim_scale_rejected(self):
        # x and s are both contiguous, so the contiguity guard passes and the
        # 2-D rank guard is the one exercised by a 3-D scale tensor.
        x = paddle.zeros([128, 256], dtype="float32")
        s = paddle.zeros([1, 2, 3], dtype="float32")
        self.assertTrue(x.is_contiguous() and s.is_contiguous())
        with self.assertRaisesRegex(AssertionError, "2 dimensions"):
            dsv3_kernel.weight_dequant(x, s, block_size=128)

    def test_non_contiguous_weight_rejected(self):
        base = paddle.arange(4 * 6, dtype="float32").reshape([4, 6])
        x = paddle.transpose(base, [1, 0])  # 2-D but non-contiguous
        s = paddle.zeros([1, 1], dtype="float32")
        if x.is_contiguous():
            self.skipTest(
                "transpose produced a contiguous tensor on this build"
            )
        with self.assertRaisesRegex(AssertionError, "contiguous"):
            dsv3_kernel.weight_dequant(x, s, block_size=128)


@unittest.skipUnless(_KERNEL_AVAILABLE, _IMPORT_SKIP_REASON)
class TestFp8GemmHostGuards(unittest.TestCase):
    """CPU-observable support-condition guards of ``fp8_gemm``.

    ``b`` and ``b_s`` are forced contiguous by the wrapper before the asserts,
    so only ``a`` / ``a_s`` contiguity is user-observable here; the regex pins
    which of the two guards fired.
    """

    def _valid_operands(self):
        a = paddle.zeros([4, 8], dtype="float32")
        a_s = paddle.zeros([4, 1], dtype="float32")
        b = paddle.zeros([8, 4], dtype="float32")
        b_s = paddle.zeros([1, 1], dtype="float32")
        return a, a_s, b, b_s

    def test_non_contiguous_a_rejected(self):
        _, a_s, b, b_s = self._valid_operands()
        a = paddle.transpose(paddle.zeros([8, 4], dtype="float32"), [1, 0])
        if a.is_contiguous():
            self.skipTest(
                "transpose produced a contiguous tensor on this build"
            )
        with self.assertRaisesRegex(
            AssertionError, "Input tensors must be contiguous"
        ):
            dsv3_kernel.fp8_gemm(a, a_s, b, b_s)

    def test_non_contiguous_scale_rejected(self):
        a, _, b, b_s = self._valid_operands()
        a_s = paddle.transpose(paddle.zeros([1, 4], dtype="float32"), [1, 0])
        if a_s.is_contiguous():
            self.skipTest(
                "transpose produced a contiguous tensor on this build"
            )
        # a and b are contiguous, so the first guard passes and the scaling-factor
        # guard is isolated as the cause.
        with self.assertRaisesRegex(
            AssertionError, "Scaling factor tensors must be contiguous"
        ):
            dsv3_kernel.fp8_gemm(a, a_s, b, b_s)


@unittest.skipUnless(_KERNEL_AVAILABLE, _IMPORT_SKIP_REASON)
class TestFp8GemmHostLayout(unittest.TestCase):
    """Verify the host-side layout/dispatch derivation of ``fp8_gemm``.

    The GPU kernel is a *non-tested* collaborator here: it is replaced by a spy
    that records the real arguments the wrapper computes. This proves the
    transpose of ``b``/``b_s``, the ``M``/``N``/``K`` extraction, the output
    shape and the launch grid -- NOT the kernel's numeric output (see the
    GPU-guarded numeric tests for that).
    """

    def test_transpose_shape_and_grid_derivation(self):
        M, K, N = 3, 128, 64
        a = paddle.arange(M * K, dtype="float32").reshape([M, K])
        a_s = paddle.ones([M, 1], dtype="float32")  # [M, ceil(K/128)]
        b = paddle.arange(K * N, dtype="float32").reshape([K, N])  # [K, N]
        b_s = paddle.ones([1, 1], dtype="float32")  # [ceil(K/128), ceil(N/128)]

        spy = _KernelLaunchSpy()
        with mock.patch.object(dsv3_kernel, "fp8_gemm_kernel", spy):
            c = dsv3_kernel.fp8_gemm(a, a_s, b, b_s)

        self.assertEqual(spy.call_count, 1)
        args, kwargs = spy.args, spy.kwargs

        # a and a_s are passed through untouched (identity); swapping them with
        # b/b_s would be caught here.
        self.assertIs(args[0], a)
        self.assertIs(args[3], a_s)

        # b and b_s are transposed to contiguous [N, K] / [Nblk, Kblk] layout.
        self.assertTrue(args[1].is_contiguous())
        np.testing.assert_array_equal(args[1].numpy(), b.numpy().T)
        np.testing.assert_array_equal(args[4].numpy(), b_s.numpy().T)

        # Output buffer: shape derived from a's leading dims and N (= original
        # b.shape[1]); it is the object returned to the caller.
        self.assertIs(args[2], c)
        self.assertEqual(c.shape, [M, N])
        # Output uses the framework default dtype (float32 fixtures here).
        self.assertEqual(c.dtype, a.dtype)

        # Scalar dims M, N, K.
        self.assertEqual(args[5], M)
        self.assertEqual(args[6], N)
        self.assertEqual(args[7], K)

        # Block sizes are the wrapper's fixed constants.
        self.assertEqual(kwargs["BLOCK_SIZE_M"], 32)
        self.assertEqual(kwargs["BLOCK_SIZE_N"], 64)
        self.assertEqual(kwargs["BLOCK_SIZE_K"], 128)

        # Launch grid tiles (M, N) by the M/N block sizes; independently derived.
        grid = spy.grid({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64})
        self.assertEqual(tuple(grid), (_cdiv(M, 32), _cdiv(N, 64)))


@unittest.skipUnless(
    _GPU_FP8_READY, _GPU_SKIP_REASON or "GPU FP8 not available"
)
class TestActQuantNumeric(unittest.TestCase):
    """Real GPU numerics for ``act_quant`` vs an independent reference.

    With ``x`` shaped ``[R, block_size]`` each row is exactly one quantization
    block, so per-row ``scale = max(|x|) / 448`` and ``y = cast_fp8(x / scale)``.
    The reference uses only elementwise ops, never ``act_quant`` itself.
    """

    def test_per_row_scale_and_quantized_values(self):
        paddle.seed(0)
        x = paddle.randn([4, 128], dtype="float32")
        y, s = dsv3_kernel.act_quant(x, block_size=128)

        self.assertEqual(y.dtype, paddle.float8_e4m3fn)
        self.assertEqual(s.dtype, paddle.float32)
        self.assertEqual(list(s.shape), [4, 1])

        scale_ref = x.abs().max(axis=-1, keepdim=True) / 448.0
        np.testing.assert_allclose(
            s.numpy(), scale_ref.numpy(), rtol=1e-6, atol=1e-8
        )
        # Same fp32 division then identical fp8 cast -> bit-identical codes.
        y_ref = (x / scale_ref).astype(paddle.float8_e4m3fn)
        np.testing.assert_array_equal(
            y.astype("float32").numpy(), y_ref.astype("float32").numpy()
        )


@unittest.skipUnless(
    _GPU_FP8_READY, _GPU_SKIP_REASON or "GPU FP8 not available"
)
class TestWeightDequantNumeric(unittest.TestCase):
    """Real GPU numerics for ``weight_dequant`` vs an independent reference.

    A ``[128, 256]`` weight with ``block_size=128`` has two column blocks, each
    scaled by its own factor. Distinct scales (2.0 vs 5.0) make a block swap
    observable; the reference multiplies each column half separately.
    """

    def test_blockwise_dequant(self):
        paddle.seed(0)
        raw = paddle.randn([128, 256], dtype="float32")
        x = raw.astype(paddle.float8_e4m3fn)
        s = paddle.to_tensor([[2.0, 5.0]], dtype="float32")  # [1, 2]

        y = dsv3_kernel.weight_dequant(x, s, block_size=128)
        self.assertEqual(list(y.shape), [128, 256])

        x_f32 = x.astype("float32").numpy()
        ref = np.empty_like(x_f32)
        ref[:, :128] = x_f32[:, :128] * 2.0
        ref[:, 128:] = x_f32[:, 128:] * 5.0
        np.testing.assert_allclose(
            y.astype("float32").numpy(), ref, rtol=1e-6, atol=1e-6
        )


@unittest.skipUnless(
    _GPU_FP8_READY, _GPU_SKIP_REASON or "GPU FP8 not available"
)
class TestFp8GemmNumeric(unittest.TestCase):
    """Real GPU numerics for ``fp8_gemm`` vs a dequantized-matmul reference.

    Single K/N blocks (K=N=128) keep the block-scale bookkeeping unambiguous:
    ``c[m, n] = (a_fp8 . b_fp8)[m, n] * a_s[m] * b_s``. Operands are quantized
    directly (not via the tested wrapper), and non-uniform row scales expose a
    dropped scale or a transposed operand.
    """

    def test_scaled_fp8_matmul(self):
        paddle.seed(0)
        M, K, N = 32, 128, 128
        a = paddle.randn([M, K], dtype="float32").astype(paddle.float8_e4m3fn)
        b = paddle.randn([K, N], dtype="float32").astype(paddle.float8_e4m3fn)
        a_s = paddle.to_tensor(
            [[0.5 + 0.1 * i] for i in range(M)], dtype="float32"
        )  # [M, 1], non-uniform
        b_s = paddle.to_tensor([[0.75]], dtype="float32")  # [1, 1]

        c = dsv3_kernel.fp8_gemm(a, a_s, b, b_s)
        self.assertEqual(list(c.shape), [M, N])

        a_f = a.astype("float32").numpy()
        b_f = b.astype("float32").numpy()
        ref = (a_f @ b_f) * a_s.numpy() * float(b_s.numpy())
        self.assertGreater(np.abs(ref).max(), 1e-3)  # non-degenerate reference
        got = c.astype("float32").numpy()
        self.assertTrue(np.isfinite(got).all())
        np.testing.assert_allclose(got, ref, rtol=2e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
