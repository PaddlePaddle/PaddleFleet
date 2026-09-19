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

"""Behaviour tests for ``paddlefleet.triton_ops.fused_sink_grad``.

The heavy numeric core is a Triton kernel and only runs on a real CUDA device;
on CPU-only / dependency-less collection every test class skips with an honest
reason (the real import-error repr). Two layers are exercised:

* the pure-python argument validation in ``fused_sink_grad`` -- it raises before
  any kernel launch, so it is CPU-testable;
* the kernel numeric result, compared against an *independent* numpy reference
  derived by hand from the documented sink-gradient math, guarded behind a real
  CUDA device (never a faked GPU / mocked kernel).

The numpy reference itself is anchored by tiny hand-computed cases so it cannot
silently drift into agreeing with a broken kernel.
"""

import unittest

import numpy as np

_IMPORT_ERROR = None
try:
    import paddle

    from paddlefleet.triton_ops.fused_sink_grad import fused_sink_grad
except (ImportError, ModuleNotFoundError) as exc:  # dependency may be absent
    _IMPORT_ERROR = exc
    paddle = None
    fused_sink_grad = None

_IMPORTS_OK = _IMPORT_ERROR is None
_SKIP_IMPORT = f"fused_sink_grad import unavailable: {_IMPORT_ERROR!r}"


def _cuda_usable():
    """True only when a real CUDA device can actually be selected.

    A CUDA-enabled build is not sufficient: the kernel is JIT-compiled and
    launched on the device, so an unusable driver/context must skip rather than
    error. Import availability is a precondition (``paddle`` is ``None`` when the
    guarded import failed).
    """
    if not _IMPORTS_OK:
        return False
    if not paddle.is_compiled_with_cuda():
        return False
    try:
        if paddle.device.cuda.device_count() == 0:
            return False
        paddle.set_device("gpu:0")
        place = str(paddle.empty([1]).place).lower()
    except Exception:  # driver/context init failure => no usable GPU
        return False
    return "gpu" in place or "cuda" in place


_CUDA_OK = _cuda_usable()
_SKIP_CUDA = "requires a usable CUDA device (Triton kernel is GPU-only)"


def _numpy_reference_sink_grad(out_np, do_np, lse_np, sink_np, num_heads):
    """Independent fp64 reference for the attention-sink gradient.

    Hand-derived from the documented math, NOT from production output:

        Delta[n, h]  = sum_dv(out * do)
        lse_full     = logaddexp(lse, sink)          # forward LSE is KV-only
        p_sink[n, h] = exp(sink - lse_full)
        d_sink[h]    = -sum_n( p_sink[n, h] * Delta[n, h] )

    Inputs arrive as fp32 numpy (already up-cast from the tensors the kernel
    consumed) and are promoted to fp64 here so the reference is strictly more
    accurate than the fp32 kernel. Only the first ``num_heads`` heads are
    returned; padded heads carry no signal.
    """
    out64 = out_np.astype(np.float64)
    do64 = do_np.astype(np.float64)
    delta = (out64 * do64).sum(axis=-1)  # [b, s, h_pad]
    sink64 = sink_np.astype(np.float64)  # [h_pad]
    lse64 = lse_np.astype(np.float64)  # [b, s, h_pad]
    lse_full = np.logaddexp(lse64, sink64[None, None, :])
    p_sink = np.exp(sink64[None, None, :] - lse_full)
    d_sink_full = -(p_sink * delta).sum(axis=(0, 1))  # [h_pad]
    return d_sink_full[:num_heads]


class TestNumpyReferenceMath(unittest.TestCase):
    """Anchor the reference itself against hand-computed values (no paddle)."""

    def test_single_head_single_row(self):
        # Delta = 1*3 + 2*4 = 11; logaddexp(0,0)=ln2 => p_sink=0.5;
        # d_sink = -(0.5 * 11) = -5.5.
        out = np.array([[[[1.0, 2.0]]]], dtype=np.float32)
        do = np.array([[[[3.0, 4.0]]]], dtype=np.float32)
        lse = np.array([[[0.0]]], dtype=np.float32)
        sink = np.array([0.0], dtype=np.float32)
        ref = _numpy_reference_sink_grad(out, do, lse, sink, 1)
        np.testing.assert_allclose(ref, [-5.5], rtol=0, atol=1e-12)

    def test_returns_only_real_heads(self):
        # Two heads present, only head 0 requested. Delta0 = 2*5 = 10;
        # p_sink0 = 0.5 => d_sink0 = -5.0. Head 1 must be dropped, not summed in.
        out = np.array([[[[2.0], [9.0]]]], dtype=np.float32)
        do = np.array([[[[5.0], [9.0]]]], dtype=np.float32)
        lse = np.array([[[0.0, 0.0]]], dtype=np.float32)
        sink = np.array([0.0, 0.0], dtype=np.float32)
        ref = _numpy_reference_sink_grad(out, do, lse, sink, 1)
        self.assertEqual(ref.shape, (1,))
        np.testing.assert_allclose(ref, [-5.0], rtol=0, atol=1e-12)

    def test_sink_probability_is_consumed(self):
        # Raising a head's sink logit relative to lse increases p_sink and thus
        # the gradient magnitude; a formula that ignored ``sink`` would not move.
        out = np.array([[[[1.0, 1.0]]]], dtype=np.float32)
        do = np.array([[[[1.0, 1.0]]]], dtype=np.float32)
        lse = np.array([[[0.0]]], dtype=np.float32)
        low = _numpy_reference_sink_grad(
            out, do, lse, np.array([-5.0], "float32"), 1
        )
        high = _numpy_reference_sink_grad(
            out, do, lse, np.array([5.0], "float32"), 1
        )
        # Delta = 2; p_sink = sigmoid(sink - lse), so high uses sigmoid(5).
        self.assertGreater(abs(high[0]), abs(low[0]))
        np.testing.assert_allclose(
            high[0], -2.0 * (1.0 / (1.0 + np.exp(-5.0))), rtol=1e-6
        )


@unittest.skipUnless(_IMPORTS_OK, _SKIP_IMPORT)
class TestFusedSinkGradValidation(unittest.TestCase):
    """Argument validation runs before any kernel launch, so it is CPU-safe.

    Each case is crafted to fall through every earlier guard and trip exactly
    the one under test, and asserts the specific ``ValueError`` message rather
    than merely that *some* error was raised.
    """

    def _base(self, b=1, s=8, h_pad=4, d_v=32):
        out = paddle.zeros([b, s, h_pad, d_v], dtype="bfloat16")
        do = paddle.zeros([b, s, h_pad, d_v], dtype="bfloat16")
        lse = paddle.zeros([b, s, h_pad], dtype="float32")
        sink = paddle.zeros([h_pad], dtype="float32")
        return out, do, lse, sink

    def test_out_do_shape_mismatch(self):
        out, _, lse, sink = self._base()
        do = paddle.zeros([1, 8, 4, 16], dtype="bfloat16")
        with self.assertRaisesRegex(ValueError, "out/do shape mismatch"):
            fused_sink_grad(out, do, lse, sink, 4)

    def test_out_do_dtype_mismatch(self):
        out, _, lse, sink = self._base()
        do = out.astype("float32")  # same shape, different dtype
        with self.assertRaisesRegex(ValueError, "out/do dtype mismatch"):
            fused_sink_grad(out, do, lse, sink, 4)

    def test_lse_must_be_fp32(self):
        out, do, lse, sink = self._base()
        with self.assertRaisesRegex(ValueError, "lse/sink must be fp32"):
            fused_sink_grad(out, do, lse.astype("bfloat16"), sink, 4)

    def test_sink_must_be_fp32(self):
        out, do, lse, sink = self._base()
        with self.assertRaisesRegex(ValueError, "lse/sink must be fp32"):
            fused_sink_grad(out, do, lse, sink.astype("bfloat16"), 4)

    def test_num_heads_exceeds_h_pad(self):
        out, do, lse, sink = self._base()  # h_pad == 4
        with self.assertRaisesRegex(ValueError, "exceeds h_pad"):
            fused_sink_grad(out, do, lse, sink, 8)

    def test_lse_wrong_shape(self):
        out, do, _, sink = self._base()
        bad_lse = paddle.zeros([1, 4, 4], dtype="float32")  # s=4 != 8
        with self.assertRaisesRegex(ValueError, "lse must be"):
            fused_sink_grad(out, do, bad_lse, sink, 4)

    def test_sink_wrong_shape(self):
        out, do, lse, _ = self._base()
        bad_sink = paddle.zeros([2], dtype="float32")  # 2 != h_pad(4)
        with self.assertRaisesRegex(ValueError, "sink must be"):
            fused_sink_grad(out, do, lse, bad_sink, 4)


@unittest.skipUnless(_CUDA_OK, _SKIP_CUDA)
class TestFusedSinkGradNumeric(unittest.TestCase):
    """Real kernel vs the independent numpy reference (GPU only)."""

    def _run_and_compare(
        self, b, s, h, h_pad, d_v, dtype="bfloat16", seed=0, tol=1e-3
    ):
        paddle.seed(seed)
        out = paddle.randn([b, s, h_pad, d_v]).cast(dtype)
        do = paddle.randn([b, s, h_pad, d_v]).cast(dtype)
        # Padded heads carry no signal and must not reach the [h] result.
        if h_pad > h:
            out[:, :, h:, :] = 0
            do[:, :, h:, :] = 0
        lse = paddle.randn([b, s, h_pad]).cast("float32") * 2.0
        sink = paddle.randn([h_pad]).cast("float32")
        if h_pad > h:
            sink[h:] = -1e30

        got = fused_sink_grad(out, do, lse, sink, h)
        ref = _numpy_reference_sink_grad(
            out.astype("float32").numpy(),
            do.astype("float32").numpy(),
            lse.numpy(),
            sink.numpy(),
            h,
        )

        self.assertEqual(list(got.shape), [h])
        self.assertEqual(got.dtype, paddle.float32)
        g = got.numpy()
        self.assertTrue(np.isfinite(g).all())
        self.assertTrue(np.isfinite(ref).all())
        scale = float(np.abs(ref).max())
        # This fixture is built to produce a non-trivial gradient; a scale of 0
        # would make the relative check vacuous.
        self.assertGreater(scale, 0.0)
        # Scale-sensitive: a 0.5x / 2x magnitude error yields rel >= 0.5 and is
        # rejected, unlike a cosine/direction-only comparison.
        rel = np.abs(g - ref).max() / scale
        self.assertLess(rel, tol, f"max|diff|/max|ref|={rel:.3e}")
        return got

    def test_matches_independent_reference(self):
        self._run_and_compare(1, 256, 8, 8, 512)

    def test_padded_heads_do_not_contribute(self):
        # h < h_pad (e.g. TP>1): result is [h] and padded heads are excluded.
        self._run_and_compare(1, 128, 16, 64, 512)

    def test_ragged_shapes(self):
        # n_rows and d_v not multiples of the kernel block sizes.
        self._run_and_compare(1, 1000, 8, 8, 100)

    def test_fp16_inputs(self):
        self._run_and_compare(1, 256, 8, 8, 512, dtype="float16")

    def test_deterministic_repeatable(self):
        # No atomics in the reduction: same inputs must be bit-identical.
        a = self._run_and_compare(1, 1024, 32, 64, 512, seed=7)
        b = self._run_and_compare(1, 1024, 32, 64, 512, seed=7)
        np.testing.assert_array_equal(a.numpy(), b.numpy())


if __name__ == "__main__":
    unittest.main()
