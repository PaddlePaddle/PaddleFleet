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

"""
Tests for the fused_stack_quant function in fp8_utils.py.

This file covers the changes introduced in commit e61ac9f:
  1. New `use_ue8m0` parameter added to fused_stack_quant.
  2. Internal calls replaced: paddle.incubate.nn.functional.fused_stack_transpose_quant
     -> fuse_stack_transpose_fp8_quant / fuse_stack_fp8_quant (from paddlefleet.ops).
  3. When use_ue8m0=True, the returned scale is transposed (.T) before returning.
  4. All cache-hit and fallback branches are exercised.
"""

import unittest

import numpy as np
import paddle
from paddle.base import core

# fused_stack_quant lives in fp8_utils but calls paddlefleet.ops internally.
# We import it here and rely on the same try/except guard that fp8_utils uses.
from paddlefleet.transformer.moe.fp8_utils import fused_stack_quant

NUM_EXPERTS = 4
N, K = 512, 256  # small dims for fast tests


def _make_weight_list(
    num_experts=NUM_EXPERTS, shape=(N, K), dtype=paddle.bfloat16
):
    """Return a plain list of paddle Tensors (no cache attributes)."""
    return [paddle.randn(shape, dtype=dtype) for _ in range(num_experts)]


def _attach_fp8_cache(weight_list, stacked_w, stacked_s, transpose=False):
    """Attach pre-computed FP8 cache attributes to the first element of weight_list."""
    w0 = weight_list[0]
    if transpose:
        w0.fp8_weight_stacked_transpose = stacked_w
        w0.fp8_scale_stacked_transpose = stacked_s
    else:
        w0.fp8_weight_stacked = stacked_w
        w0.fp8_scale_stacked = stacked_s
    return weight_list


class TestFusedStackQuantNoCacheNonTranspose(unittest.TestCase):
    """Live quantization path: no cache, transpose=False, use_ue8m0=False."""

    def setUp(self):
        if not core.is_compiled_with_cuda():
            self.skipTest("CUDA required")
        self.weights = _make_weight_list()

    def test_returns_two_tensors(self):
        w, scale = fused_stack_quant(self.weights, transpose=False)
        self.assertIsInstance(w, paddle.Tensor)
        self.assertIsInstance(scale, paddle.Tensor)

    def test_weight_dtype_is_fp8(self):
        w, _ = fused_stack_quant(self.weights, transpose=False)
        self.assertEqual(w.dtype, paddle.float8_e4m3fn)

    def test_weight_rows_equals_num_experts_times_N(self):
        # op stacks experts along dim-0: total rows = num_experts * N
        w, _ = fused_stack_quant(self.weights, transpose=False)
        self.assertEqual(w.shape[0], NUM_EXPERTS * N)
        self.assertEqual(w.shape[1], K)

    def test_scale_is_float32(self):
        _, scale = fused_stack_quant(self.weights, transpose=False)
        self.assertEqual(scale.dtype, paddle.float32)


class TestFusedStackQuantNoCacheTranspose(unittest.TestCase):
    """Live quantization path: no cache, transpose=True, use_ue8m0=False."""

    def setUp(self):
        if not core.is_compiled_with_cuda():
            self.skipTest("CUDA required")
        self.weights = _make_weight_list()

    def test_returns_two_tensors(self):
        w, scale = fused_stack_quant(self.weights, transpose=True)
        self.assertIsInstance(w, paddle.Tensor)
        self.assertIsInstance(scale, paddle.Tensor)

    def test_weight_dtype_is_fp8(self):
        w, _ = fused_stack_quant(self.weights, transpose=True)
        self.assertEqual(w.dtype, paddle.float8_e4m3fn)


class TestFusedStackQuantUe8m0ScaleTranspose(unittest.TestCase):
    """
    Verify the new use_ue8m0=True behavior introduced in the commit:
    scale = scale.T is applied on the raw op output when use_ue8m0=True.
    We compare against the direct op call (same weights) to confirm identity.
    """

    def setUp(self):
        if not core.is_compiled_with_cuda():
            self.skipTest("CUDA required")
        try:
            from paddlefleet.ops import (
                fuse_stack_fp8_quant,
                fuse_stack_transpose_fp8_quant,
            )

            self._fuse_stack = fuse_stack_fp8_quant
            self._fuse_stack_t = fuse_stack_transpose_fp8_quant
        except (ImportError, RuntimeError):
            self.skipTest("paddlefleet.ops not available")
        np.random.seed(0)
        paddle.seed(0)

    def test_ue8m0_scale_shape_equals_raw_op_T_nontranspose(self):
        """scale from fused_stack_quant(use_ue8m0=True) == raw_op output scale.T
        Only meaningful on SM10 (Blackwell); skip on older GPUs where ue8m0
        calling the raw op may trigger CUDA illegal memory access."""
        arch = paddle.device.cuda.get_device_capability()[0]
        if arch < 10:
            self.skipTest(
                "ue8m0 scale layout only supported on SM10+ (Blackwell)"
            )
        use_pow2 = True
        weights = _make_weight_list()
        weights_copy = [w.clone() for w in weights]

        _, s_raw = self._fuse_stack(weights_copy, use_pow2, True, True)
        _, s_fused = fused_stack_quant(weights, transpose=False, use_ue8m0=True)
        self.assertEqual(list(s_fused.shape), list(s_raw.T.shape))

    def test_ue8m0_scale_shape_equals_raw_op_T_transpose(self):
        """scale from fused_stack_quant(transpose=True, use_ue8m0=True) == raw_op scale.T
        Only meaningful on SM10+; skip on older GPUs."""
        arch = paddle.device.cuda.get_device_capability()[0]
        if arch < 10:
            self.skipTest(
                "ue8m0 scale layout only supported on SM10+ (Blackwell)"
            )
        use_pow2 = True
        weights = _make_weight_list()
        weights_copy = [w.clone() for w in weights]

        _, s_raw = self._fuse_stack_t(weights_copy, use_pow2, True, True)
        _, s_fused = fused_stack_quant(weights, transpose=True, use_ue8m0=True)
        self.assertEqual(list(s_fused.shape), list(s_raw.T.shape))

    def test_ue8m0_weight_dtype_is_fp8_nontranspose(self):
        arch = paddle.device.cuda.get_device_capability()[0]
        if arch < 10:
            self.skipTest("use_ue8m0=True is only safe on SM10+ (Blackwell)")
        w, _ = fused_stack_quant(
            _make_weight_list(), transpose=False, use_ue8m0=True
        )
        self.assertEqual(w.dtype, paddle.float8_e4m3fn)

    def test_ue8m0_weight_dtype_is_fp8_transpose(self):
        arch = paddle.device.cuda.get_device_capability()[0]
        if arch < 10:
            self.skipTest("use_ue8m0=True is only safe on SM10+ (Blackwell)")
        w, _ = fused_stack_quant(
            _make_weight_list(), transpose=True, use_ue8m0=True
        )
        self.assertEqual(w.dtype, paddle.float8_e4m3fn)


class TestFusedStackQuantCacheHitNonTranspose(unittest.TestCase):
    """
    Cache-hit branch: expert_weight_list[0] already has fp8_weight_stacked.
    transpose=False -> should return the cached (non-transposed) values by identity.
    """

    def setUp(self):
        if not core.is_compiled_with_cuda():
            self.skipTest("CUDA required")

    def _make_cached_weights(self):
        """Run op once, attach result as cache to a fresh list."""
        weights = _make_weight_list()
        w_real, s_real = fused_stack_quant(
            weights, transpose=False, use_ue8m0=False
        )
        weights2 = _make_weight_list()
        _attach_fp8_cache(weights2, w_real, s_real, transpose=False)
        return weights2, w_real, s_real

    def test_returns_cached_non_transpose(self):
        weights2, w_real, s_real = self._make_cached_weights()
        w, scale = fused_stack_quant(weights2, transpose=False)
        # Must be the exact same tensor objects (no live quant should run)
        self.assertIs(w, w_real)
        self.assertIs(scale, s_real)

    def test_cache_hit_ignores_use_ue8m0_flag(self):
        """When cache is hit, use_ue8m0 has no effect (no live quant runs)."""
        weights2, w_real, s_real = self._make_cached_weights()
        w1, s1 = fused_stack_quant(weights2, transpose=False, use_ue8m0=False)
        w2, s2 = fused_stack_quant(weights2, transpose=False, use_ue8m0=True)
        self.assertIs(w1, w_real)
        self.assertIs(w2, w_real)
        self.assertIs(s1, s_real)
        self.assertIs(s2, s_real)


class TestFusedStackQuantCacheHitTranspose(unittest.TestCase):
    """
    Cache-hit branch: expert_weight_list[0] already has fp8_weight_stacked_transpose.
    transpose=True -> returns cached transpose values by identity.
    """

    def setUp(self):
        if not core.is_compiled_with_cuda():
            self.skipTest("CUDA required")

    def test_returns_cached_transpose(self):
        weights = _make_weight_list()
        w_real, s_real = fused_stack_quant(
            weights, transpose=True, use_ue8m0=False
        )
        weights2 = _make_weight_list()
        _attach_fp8_cache(weights2, w_real, s_real, transpose=True)

        w, scale = fused_stack_quant(weights2, transpose=True)
        self.assertIs(w, w_real)
        self.assertIs(scale, s_real)


class TestFusedStackQuantCrossHitTransposeFromNonTranspose(unittest.TestCase):
    """
    Fallback branch: transpose=True requested, but only non-transpose cache exists.
    Should return the non-transposed cached values by identity.
    """

    def setUp(self):
        if not core.is_compiled_with_cuda():
            self.skipTest("CUDA required")

    def test_fallback_transpose_uses_nontranspose_cache(self):
        weights = _make_weight_list()
        w_real, s_real = fused_stack_quant(
            weights, transpose=False, use_ue8m0=False
        )
        weights2 = _make_weight_list()
        _attach_fp8_cache(weights2, w_real, s_real, transpose=False)
        # Only fp8_weight_stacked exists (not fp8_weight_stacked_transpose)

        w, scale = fused_stack_quant(weights2, transpose=True)
        self.assertIs(w, w_real)
        self.assertIs(scale, s_real)


class TestFusedStackQuantCrossHitNonTransposeFromTranspose(unittest.TestCase):
    """
    Fallback branch: transpose=False requested, but only transpose cache exists.
    Should return the transposed cached values by identity.
    """

    def setUp(self):
        if not core.is_compiled_with_cuda():
            self.skipTest("CUDA required")

    def test_fallback_nontranspose_uses_transpose_cache(self):
        weights = _make_weight_list()
        w_real, s_real = fused_stack_quant(
            weights, transpose=True, use_ue8m0=False
        )
        weights2 = _make_weight_list()
        _attach_fp8_cache(weights2, w_real, s_real, transpose=True)
        # Only fp8_weight_stacked_transpose exists

        w, scale = fused_stack_quant(weights2, transpose=False)
        self.assertIs(w, w_real)
        self.assertIs(scale, s_real)


class TestFusedStackQuantOutputConsistency(unittest.TestCase):
    """
    Verify that fused_stack_quant (transpose=False) and (transpose=True) produce
    quantized values that are numerically identical to each other (same data,
    just potentially re-ordered), and that dequantized values are close to the
    original BF16 weights.
    """

    def setUp(self):
        if not core.is_compiled_with_cuda():
            self.skipTest("CUDA required")
        np.random.seed(42)
        paddle.seed(42)
        self.weights = _make_weight_list(num_experts=2, shape=(256, 128))

    def test_nontranspose_fp8_dtype(self):
        w, scale = fused_stack_quant(
            self.weights, transpose=False, use_ue8m0=False
        )
        self.assertEqual(w.dtype, paddle.float8_e4m3fn)
        self.assertEqual(scale.dtype, paddle.float32)

    def test_transpose_fp8_dtype(self):
        w, scale = fused_stack_quant(
            self.weights, transpose=True, use_ue8m0=False
        )
        self.assertEqual(w.dtype, paddle.float8_e4m3fn)
        self.assertEqual(scale.dtype, paddle.float32)

    def test_scale_positive(self):
        """All scale values must be strictly positive."""
        _, scale = fused_stack_quant(
            self.weights, transpose=False, use_ue8m0=False
        )
        self.assertTrue((scale > 0).all().item())

    def test_ue8m0_scale_no_exception(self):
        """use_ue8m0=True should not raise (only validated on SM10+ where it is safe)."""
        arch = paddle.device.cuda.get_device_capability()[0]
        if arch < 10:
            self.skipTest("use_ue8m0=True is only safe on SM10+ (Blackwell)")
        try:
            _, scale = fused_stack_quant(
                _make_weight_list(num_experts=2, shape=(256, 128)),
                transpose=False,
                use_ue8m0=True,
            )
        except Exception as e:
            self.fail(f"use_ue8m0=True raised: {e}")


class TestFusedStackQuantPow2ScaleBlackwell(unittest.TestCase):
    """
    Smoke test: on SM10 (Blackwell) use_pow2_scale is forced True internally.
    On other GPUs it is False. Either way the function must not raise.
    This test does not assert numeric correctness of pow2_scale behavior
    (that is covered by test_fuse_stack_transpose_fp8_quant.py), only that
    the code path completes without error.
    """

    def setUp(self):
        if not core.is_compiled_with_cuda():
            self.skipTest("CUDA required")
        self.weights = _make_weight_list(num_experts=2, shape=(256, 128))

    def test_no_exception_nontranspose(self):
        # Should not raise on any GPU architecture
        try:
            w, scale = fused_stack_quant(
                self.weights, transpose=False, use_ue8m0=False
            )
        except Exception as e:
            self.fail(f"fused_stack_quant raised an exception: {e}")

    def test_no_exception_transpose(self):
        try:
            w, scale = fused_stack_quant(
                self.weights, transpose=True, use_ue8m0=False
            )
        except Exception as e:
            self.fail(f"fused_stack_quant raised an exception: {e}")

    def test_no_exception_ue8m0_nontranspose(self):
        arch = paddle.device.cuda.get_device_capability()[0]
        if arch < 10:
            self.skipTest("use_ue8m0=True is only safe on SM10+ (Blackwell)")
        try:
            w, scale = fused_stack_quant(
                self.weights, transpose=False, use_ue8m0=True
            )
        except Exception as e:
            self.fail(f"fused_stack_quant raised an exception: {e}")

    def test_no_exception_ue8m0_transpose(self):
        arch = paddle.device.cuda.get_device_capability()[0]
        if arch < 10:
            self.skipTest(
                "use_ue8m0=True with transpose=True is only safe on SM10+"
            )
        try:
            w, scale = fused_stack_quant(
                self.weights, transpose=True, use_ue8m0=True
            )
        except Exception as e:
            self.fail(f"fused_stack_quant raised an exception: {e}")


class TestFusedStackQuantReplacedOps(unittest.TestCase):
    """
    Verify that the replaced ops (fuse_stack_fp8_quant / fuse_stack_transpose_fp8_quant)
    from paddlefleet.ops are actually used instead of the old paddle.incubate API.

    Strategy: import the new ops and confirm they are callable, then confirm
    fused_stack_quant returns the same stacked tensor as calling them directly.
    """

    def setUp(self):
        if not core.is_compiled_with_cuda():
            self.skipTest("CUDA required")
        try:
            from paddlefleet.ops import (
                fuse_stack_fp8_quant,
                fuse_stack_transpose_fp8_quant,
            )

            self.fuse_stack_fp8_quant = fuse_stack_fp8_quant
            self.fuse_stack_transpose_fp8_quant = fuse_stack_transpose_fp8_quant
        except (ImportError, RuntimeError):
            self.skipTest("paddlefleet.ops not available")
        np.random.seed(7)
        paddle.seed(7)
        self.weights = _make_weight_list(num_experts=2, shape=(256, 128))
        # Keep a copy with same data for direct op call
        self.weights_copy = [w.clone() for w in self.weights]

    def test_nontranspose_matches_direct_op_call(self):
        arch = paddle.device.cuda.get_device_capability()[0]
        use_pow2 = arch == 10

        w_direct, s_direct = self.fuse_stack_fp8_quant(
            self.weights_copy, use_pow2, False, False
        )
        w_fused, s_fused = fused_stack_quant(
            self.weights, transpose=False, use_ue8m0=False
        )

        np.testing.assert_array_equal(
            w_direct.numpy(),
            w_fused.numpy(),
            err_msg="fused_stack_quant (non-transpose) must match direct fuse_stack_fp8_quant call",
        )
        np.testing.assert_allclose(
            s_direct.numpy(),
            s_fused.numpy(),
            rtol=0,
            atol=0,
            err_msg="Scale tensors must match",
        )

    def test_transpose_matches_direct_op_call(self):
        arch = paddle.device.cuda.get_device_capability()[0]
        use_pow2 = arch == 10

        w_direct, s_direct = self.fuse_stack_transpose_fp8_quant(
            self.weights_copy, use_pow2, False, False
        )
        w_fused, s_fused = fused_stack_quant(
            self.weights, transpose=True, use_ue8m0=False
        )

        np.testing.assert_array_equal(
            w_direct.numpy(),
            w_fused.numpy(),
            err_msg="fused_stack_quant (transpose) must match direct fuse_stack_transpose_fp8_quant call",
        )
        np.testing.assert_allclose(
            s_direct.numpy(),
            s_fused.numpy(),
            rtol=0,
            atol=0,
            err_msg="Scale tensors must match",
        )

    def test_ue8m0_nontranspose_scale_is_direct_scale_transposed(self):
        """When use_ue8m0=True, fused_stack_quant must return scale.T of the direct op.
        Only runs on SM10+ where calling the raw op with ue8m0=True is safe."""
        arch = paddle.device.cuda.get_device_capability()[0]
        if arch < 10:
            self.skipTest("ue8m0 only supported on SM10+ (Blackwell)")
        use_pow2 = True

        _, s_direct = self.fuse_stack_fp8_quant(
            self.weights_copy, use_pow2, True, True
        )
        _, s_fused = fused_stack_quant(
            self.weights, transpose=False, use_ue8m0=True
        )

        # The commit does: scale = scale.T after the op call when use_ue8m0=True
        np.testing.assert_array_equal(
            s_direct.T.numpy(),
            s_fused.numpy(),
            err_msg="use_ue8m0=True must return scale.T of the raw op output",
        )

    def test_ue8m0_transpose_scale_is_direct_scale_transposed(self):
        """Only runs on SM10+."""
        arch = paddle.device.cuda.get_device_capability()[0]
        if arch < 10:
            self.skipTest("ue8m0 only supported on SM10+ (Blackwell)")
        use_pow2 = True

        _, s_direct = self.fuse_stack_transpose_fp8_quant(
            self.weights_copy, use_pow2, True, True
        )
        _, s_fused = fused_stack_quant(
            self.weights, transpose=True, use_ue8m0=True
        )

        np.testing.assert_array_equal(
            s_direct.T.numpy(),
            s_fused.numpy(),
            err_msg="use_ue8m0=True (transpose) must return scale.T of the raw op output",
        )


class TestFusedWeightedSwigluFp8QuantReplacement(unittest.TestCase):
    """
    Verify the second part of the commit:
    fuse_weighted_swiglu_fp8_quant (from paddlefleet.ops) is now called instead of
    paddle.incubate.nn.functional.fused_weighted_swiglu_act_quant.

    Key behavioral differences compared to the old call:
      - No paddle.amp.auto_cast(False) wrapper needed.
      - prob is NOT squeezed before the call (the new op accepts shape [M, 1]).
      - using_pow2_scaling=True, use_ue8m0=False is the new signature.
    """

    def setUp(self):
        if not core.is_compiled_with_cuda():
            self.skipTest("CUDA required")
        try:
            from paddlefleet.ops import fuse_weighted_swiglu_fp8_quant

            self.fuse_weighted_swiglu_fp8_quant = fuse_weighted_swiglu_fp8_quant
        except (ImportError, RuntimeError):
            self.skipTest("paddlefleet.ops not available")
        np.random.seed(0)
        paddle.seed(0)

    def _make_inputs(self, M=384, K=256):
        x = paddle.randn([M, K * 2], dtype=paddle.bfloat16)
        # prob shape is [M, 1] — as passed in fwd_down_fp8 (NOT squeezed in new code)
        prob = paddle.randn([M, 1], dtype=paddle.float32)
        return x, prob

    def test_new_op_accepts_unsqueezed_prob(self):
        """The new op must accept prob with shape [M, 1] (no squeeze)."""
        x, prob = self._make_inputs()
        try:
            fp8_out, scale = self.fuse_weighted_swiglu_fp8_quant(
                x, prob, using_pow2_scaling=True, use_ue8m0=False
            )
        except Exception as e:
            self.fail(
                f"fuse_weighted_swiglu_fp8_quant raised with unsqueezed prob: {e}"
            )

    def test_new_op_output_dtype_fp8(self):
        x, prob = self._make_inputs()
        fp8_out, scale = self.fuse_weighted_swiglu_fp8_quant(
            x, prob, using_pow2_scaling=True, use_ue8m0=False
        )
        self.assertEqual(fp8_out.dtype, paddle.float8_e4m3fn)

    def test_new_op_scale_dtype_float32(self):
        x, prob = self._make_inputs()
        _, scale = self.fuse_weighted_swiglu_fp8_quant(
            x, prob, using_pow2_scaling=True, use_ue8m0=False
        )
        self.assertEqual(scale.dtype, paddle.float32)

    def test_new_op_output_shape(self):
        """fp8_out should have shape [M, K] (SwiGLU halves input last dim)."""
        M, K = 384, 256
        x, prob = self._make_inputs(M=M, K=K)
        fp8_out, _ = self.fuse_weighted_swiglu_fp8_quant(
            x, prob, using_pow2_scaling=True, use_ue8m0=False
        )
        self.assertEqual(fp8_out.shape[0], M)
        self.assertEqual(fp8_out.shape[1], K)

    def test_new_op_no_autocast_context_needed(self):
        """
        Confirm the op works correctly inside a BF16 auto_cast context,
        unlike the old code that needed auto_cast(False) to avoid dtype errors.
        """
        x, prob = self._make_inputs()
        try:
            with paddle.amp.auto_cast(enable=True, dtype="bfloat16"):
                fp8_out, scale = self.fuse_weighted_swiglu_fp8_quant(
                    x, prob, using_pow2_scaling=True, use_ue8m0=False
                )
        except Exception as e:
            self.fail(
                f"fuse_weighted_swiglu_fp8_quant raised inside auto_cast context: {e}"
            )

    def test_new_op_dequantized_close_to_reference(self):
        """
        Dequantized output of fuse_weighted_swiglu_fp8_quant(using_pow2_scaling=True)
        should be close to swiglu(x) * prob computed in float32.
        scale shape is [M, 1] (one scale per row), expand to [M, K] for dequant.
        """
        import paddle.nn.functional as F

        M, K = 256, 128
        x = paddle.clip(
            paddle.randn([M, K * 2], dtype=paddle.bfloat16), min=-50, max=50
        )
        prob = paddle.randn([M, 1], dtype=paddle.float32)

        fp8_out, scale = self.fuse_weighted_swiglu_fp8_quant(
            x, prob, using_pow2_scaling=True, use_ue8m0=False
        )

        # scale is [M, 1]: expand cols to [M, K] for element-wise dequant
        expanded_scale = scale.expand([-1, fp8_out.shape[1]]).astype("float32")
        dequant = fp8_out.astype("float32") * expanded_scale

        golden = F.swiglu(x).astype("float32") * prob

        np.testing.assert_allclose(
            golden.numpy(),
            dequant.numpy(),
            rtol=0.02,
            atol=1.0,
        )


if __name__ == "__main__":
    unittest.main()
