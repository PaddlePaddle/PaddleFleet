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

import unittest

import numpy as np
import paddle
from paddle.base import core

M, K, N = 4096, 7168, 4096
DTYPE_PD = paddle.bfloat16

import paddle

from paddlefleet.ops import fuse_stack_fp8_quant, fuse_stack_transpose_fp8_quant


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def get_tma_aligned_size(x: int, element_size: int) -> int:
    """
    Align x to TMA-required size.
    Args:
        x: size in elements
        element_size: size of each element in bytes
    Returns:
        Aligned size in elements
    """
    kNumTMAAlignmentBytes = 16
    assert kNumTMAAlignmentBytes % element_size == 0
    return align(x, kNumTMAAlignmentBytes // element_size)


def ceil_to_ue8m0_paddle(x: paddle.Tensor):
    """
    x > 0
    return 2 ^ ceil(log2(x))
    """
    # log2(x)
    log2_x = paddle.log(x) / paddle.log(paddle.to_tensor(2.0, dtype=x.dtype))
    # ceil
    ceil_log2_x = paddle.ceil(log2_x)
    # 2^k
    return paddle.pow(paddle.to_tensor(2.0, dtype=x.dtype), ceil_log2_x)


def _get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl(
    x: paddle.Tensor,
):
    assert x.dtype == paddle.float and x.dim() in (2, 3)

    ue8m0_tensor = (x.view(paddle.int) >> 23).to(paddle.uint8)

    mn, k = x.shape[-2], x.shape[-1]
    remove_dim = False

    if x.dim() == 2:
        x, remove_dim = x.unsqueeze(0), True
    b = x.shape[0]

    aligned_mn = get_tma_aligned_size(mn, 4)
    aligned_k = align(k, 4)

    padded = paddle.zeros(
        (b, aligned_mn, aligned_k), device=x.device, dtype=paddle.uint8
    )
    padded[:, :mn, :k] = ue8m0_tensor

    padded = (
        padded.view(-1)
        .view(dtype=paddle.int)
        .view(b, aligned_mn, aligned_k // 4)
    )

    transposed = paddle.zeros(
        (b, aligned_k // 4, aligned_mn), device=x.device, dtype=paddle.int
    ).mT
    transposed[:, :, :] = padded

    aligned_x = transposed[:, :mn, :]

    return aligned_x.squeeze(0) if remove_dim else aligned_x


def transform_scale_ue8m0(sf, mn, weight_block_size=None):
    get_mn_major_tma_aligned_packed_ue8m0_tensor = (
        _get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl
    )
    if weight_block_size:
        assert weight_block_size == [128, 128]
        sf = sf.index_select(-2, paddle.arange(mn, device=sf.device) // 128)
    sf = get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)
    return sf


def quant_ref(x_scale_fp32, mn, weight_block_size=None):
    # x_scale_fp32_ = ceil_to_ue8m0_paddle(x_scale_fp32)
    ref_e8m0_scale = transform_scale_ue8m0(
        x_scale_fp32, mn=mn, weight_block_size=weight_block_size
    )
    return ref_e8m0_scale


class TestFusedStackTransposeQuant(unittest.TestCase):
    def run_op(
        self,
        x_list,
        transpose,
        using_pow2_scaling,
        use_ue8m0_scale,
        output_scale_transpose,
    ):
        inputs = x_list

        if transpose:
            out, scale = fuse_stack_transpose_fp8_quant(
                inputs,
                using_pow2_scaling,
                use_ue8m0_scale,
                output_scale_transpose,
            )
        else:
            out, scale = fuse_stack_fp8_quant(
                inputs,
                using_pow2_scaling,
                use_ue8m0_scale,
                output_scale_transpose,
            )

        return out, scale

    def test_transpose_input_output_consistency(self):
        if not core.is_compiled_with_cuda():
            return

        np.random.seed(0)
        w_paddle_list = []

        for _ in range(3):
            w = paddle.randn([N, K], dtype=DTYPE_PD)
            # y = paddle.zeros([M, N], dtype=DTYPE_PD)
            w_paddle_list.append(w)

        # Case 1: output_scale_transpose = False, use_ue8m0_scale = True
        out_false, scale_false = self.run_op(
            w_paddle_list,
            transpose=True,
            using_pow2_scaling=False,
            use_ue8m0_scale=True,
            output_scale_transpose=False,
        )

        # Case 2: output_scale_transpose = True, use_ue8m0_scale = True
        out_true, scale_true = self.run_op(
            w_paddle_list,
            transpose=True,
            using_pow2_scaling=False,
            use_ue8m0_scale=True,
            output_scale_transpose=True,
        )

        # Case 3: output_scale_transpose = True, use_ue8m0_scale = False
        out_32_false, scale_32_false = self.run_op(
            w_paddle_list,
            transpose=True,
            using_pow2_scaling=True,
            use_ue8m0_scale=False,
            output_scale_transpose=False,
        )

        np.testing.assert_allclose(
            out_false.numpy(), out_true.numpy(), atol=0, rtol=0
        )
        np.testing.assert_allclose(
            out_false.numpy(), out_32_false.numpy(), atol=0, rtol=0
        )

        scale_false_np = scale_false.numpy()
        scale_true_np = scale_true.numpy()

        scale_false_T = scale_false_np.T

        scale_32_ref = quant_ref(
            scale_32_false, out_32_false.shape[-2], [128, 128]
        )

        np.testing.assert_allclose(
            scale_32_ref.numpy(), scale_true_np.T, atol=0, rtol=0
        )
        np.testing.assert_allclose(scale_false_T, scale_true_np, atol=0, rtol=0)

    def test_output_consistency(self):
        if not core.is_compiled_with_cuda():
            return

        np.random.seed(0)
        w_paddle_list = []

        for _ in range(3):
            w = paddle.randn([N, K], dtype=DTYPE_PD)
            # y = paddle.zeros([M, N], dtype=DTYPE_PD)
            w_paddle_list.append(w)

        # Case 1: output_scale_transpose = False, use_ue8m0_scale = True
        out_false, scale_false = self.run_op(
            w_paddle_list,
            transpose=False,
            using_pow2_scaling=False,
            use_ue8m0_scale=True,
            output_scale_transpose=False,
        )

        # Case 2: output_scale_transpose = True, use_ue8m0_scale = True
        out_true, scale_true = self.run_op(
            w_paddle_list,
            transpose=False,
            using_pow2_scaling=False,
            use_ue8m0_scale=True,
            output_scale_transpose=True,
        )

        # Case 3: output_scale_transpose = True, use_ue8m0_scale = False
        out_32_false, scale_32_false = self.run_op(
            w_paddle_list,
            transpose=False,
            using_pow2_scaling=True,
            use_ue8m0_scale=False,
            output_scale_transpose=False,
        )

        np.testing.assert_allclose(
            out_false.numpy(), out_true.numpy(), atol=0, rtol=0
        )
        np.testing.assert_allclose(
            out_false.numpy(), out_32_false.numpy(), atol=0, rtol=0
        )

        scale_false_np = scale_false.numpy()
        scale_true_np = scale_true.numpy()

        scale_false_T = scale_false_np.T

        scale_32_ref = quant_ref(
            scale_32_false, out_32_false.shape[-2], [128, 128]
        )

        np.testing.assert_allclose(
            scale_32_ref.numpy(), scale_true_np.T, atol=0, rtol=0
        )
        np.testing.assert_allclose(scale_false_T, scale_true_np, atol=0, rtol=0)

    def test_gemm_out(self):
        if not core.is_compiled_with_cuda():
            return

        np.random.seed(0)
        w_paddle_list = []

        for _ in range(3):
            w = paddle.randn([N, K], dtype=DTYPE_PD)
            # y = paddle.zeros([M, N], dtype=DTYPE_PD)
            w_paddle_list.append(w)

        # Case 1: output_scale_transpose = False, use_ue8m0_scale = True
        out_false, scale_false = self.run_op(
            w_paddle_list,
            transpose=False,
            using_pow2_scaling=False,
            use_ue8m0_scale=True,
            output_scale_transpose=False,
        )

        # Case 2: output_scale_transpose = True, use_ue8m0_scale = True
        out_true, scale_true = self.run_op(
            w_paddle_list,
            transpose=False,
            using_pow2_scaling=False,
            use_ue8m0_scale=True,
            output_scale_transpose=True,
        )

        # Case 3: output_scale_transpose = True, use_ue8m0_scale = False
        out_32_false, scale_32_false = self.run_op(
            w_paddle_list,
            transpose=False,
            using_pow2_scaling=True,
            use_ue8m0_scale=False,
            output_scale_transpose=False,
        )

        np.testing.assert_allclose(
            out_false.numpy(), out_true.numpy(), atol=0, rtol=0
        )
        np.testing.assert_allclose(
            out_false.numpy(), out_32_false.numpy(), atol=0, rtol=0
        )

        scale_false_np = scale_false.numpy()
        scale_true_np = scale_true.numpy()

        scale_false_T = scale_false_np.T

        scale_32_ref = quant_ref(
            scale_32_false, out_32_false.shape[-2], [128, 128]
        )

        np.testing.assert_allclose(
            scale_32_ref.numpy(), scale_true_np.T, atol=0, rtol=0
        )
        np.testing.assert_allclose(scale_false_T, scale_true_np, atol=0, rtol=0)


class TestFusedStackTransposeQuantVariableLength(unittest.TestCase):
    """
    Test the kVariableLength code path (N > 64) where device pointer
    allocation via cudaMemcpyAsync is required. This exercises the
    ptr_tensor + cudaMemcpyAsync logic added for large input lists.
    """

    # Use small but asymmetric dimensions to avoid OOM and to
    # ensure shape differences between transpose/non-transpose are visible.
    SMALL_N = 128
    SMALL_K = 256
    # 65 inputs triggers kVariableLength (threshold is > 64)
    NUM_INPUTS = 65

    def _make_inputs(self, num=None):
        num = num or self.NUM_INPUTS
        return [
            paddle.randn([self.SMALL_N, self.SMALL_K], dtype=DTYPE_PD)
            for _ in range(num)
        ]

    def run_op(
        self,
        x_list,
        transpose,
        using_pow2_scaling,
        use_ue8m0_scale,
        output_scale_transpose,
    ):
        if transpose:
            out, scale = fuse_stack_transpose_fp8_quant(
                x_list,
                using_pow2_scaling,
                use_ue8m0_scale,
                output_scale_transpose,
            )
        else:
            out, scale = fuse_stack_fp8_quant(
                x_list,
                using_pow2_scaling,
                use_ue8m0_scale,
                output_scale_transpose,
            )
        return out, scale

    def test_variable_length_nontranspose_basic(self):
        """65 inputs (> 64) should run without error in non-transpose mode."""
        if not core.is_compiled_with_cuda():
            return

        np.random.seed(0)
        inputs = self._make_inputs()
        out, scale = self.run_op(
            inputs,
            transpose=False,
            using_pow2_scaling=False,
            use_ue8m0_scale=False,
            output_scale_transpose=False,
        )

        self.assertEqual(out.dtype, paddle.float8_e4m3fn)
        self.assertEqual(out.shape[0], self.NUM_INPUTS * self.SMALL_N)
        self.assertEqual(out.shape[1], self.SMALL_K)
        self.assertEqual(scale.dtype, paddle.float32)

    def test_variable_length_transpose_basic(self):
        """65 inputs (> 64) should run without error in transpose mode."""
        if not core.is_compiled_with_cuda():
            return

        np.random.seed(0)
        inputs = self._make_inputs()
        out, scale = self.run_op(
            inputs,
            transpose=True,
            using_pow2_scaling=False,
            use_ue8m0_scale=False,
            output_scale_transpose=False,
        )

        self.assertEqual(out.dtype, paddle.float8_e4m3fn)
        self.assertEqual(out.shape[0], self.NUM_INPUTS * self.SMALL_K)
        self.assertEqual(out.shape[1], self.SMALL_N)
        self.assertEqual(scale.dtype, paddle.float32)

    def test_variable_length_output_consistency(self):
        """
        transpose=True transposes each input [N, K]->[K, N] then stacks,
        yielding [NUM*K, N]. Verify by reshaping the non-transpose output
        [NUM*N, K] -> [NUM, N, K], transposing last two dims -> [NUM, K, N],
        then flattening to [NUM*K, N] and comparing with out_t.
        """
        if not core.is_compiled_with_cuda():
            return

        np.random.seed(42)
        inputs = self._make_inputs()

        out_no_t, _ = self.run_op(
            inputs,
            transpose=False,
            using_pow2_scaling=False,
            use_ue8m0_scale=True,
            output_scale_transpose=False,
        )
        out_t, _ = self.run_op(
            inputs,
            transpose=True,
            using_pow2_scaling=False,
            use_ue8m0_scale=True,
            output_scale_transpose=False,
        )

        # out_no_t: [NUM*N, K] -> [NUM, N, K] -> transpose last two -> [NUM, K, N]
        # -> reshape [NUM*K, N], which should match out_t: [NUM*K, N]
        out_no_t_np = out_no_t.numpy().reshape(
            self.NUM_INPUTS, self.SMALL_N, self.SMALL_K
        )
        out_no_t_transposed = out_no_t_np.transpose(0, 2, 1).reshape(
            self.NUM_INPUTS * self.SMALL_K, self.SMALL_N
        )

        np.testing.assert_allclose(
            out_no_t_transposed, out_t.numpy(), atol=0, rtol=0
        )

    def test_variable_length_scale_transpose_flag(self):
        """
        output_scale_transpose=True vs False should yield transposed scales
        in the kVariableLength path.
        """
        if not core.is_compiled_with_cuda():
            return

        np.random.seed(7)
        inputs = self._make_inputs()

        _, scale_false = self.run_op(
            inputs,
            transpose=True,
            using_pow2_scaling=False,
            use_ue8m0_scale=True,
            output_scale_transpose=False,
        )
        _, scale_true = self.run_op(
            inputs,
            transpose=True,
            using_pow2_scaling=False,
            use_ue8m0_scale=True,
            output_scale_transpose=True,
        )

        scale_false_np = scale_false.numpy()
        scale_true_np = scale_true.numpy()
        np.testing.assert_allclose(
            scale_false_np.T, scale_true_np, atol=0, rtol=0
        )

    def test_variable_length_pow2_scaling(self):
        """pow2 scaling path should also work with > 64 inputs."""
        if not core.is_compiled_with_cuda():
            return

        np.random.seed(11)
        inputs = self._make_inputs()

        out, scale = self.run_op(
            inputs,
            transpose=False,
            using_pow2_scaling=True,
            use_ue8m0_scale=False,
            output_scale_transpose=False,
        )

        self.assertEqual(out.dtype, paddle.float8_e4m3fn)
        self.assertEqual(out.shape[0], self.NUM_INPUTS * self.SMALL_N)
        self.assertEqual(scale.dtype, paddle.float32)

    def test_variable_length_matches_fixed_length(self):
        """
        Results with 65 inputs should match a reference computed by
        individually quantizing each tensor (using the fixed-length path
        with small batches).
        """
        if not core.is_compiled_with_cuda():
            return

        np.random.seed(99)
        inputs = self._make_inputs()

        # Full variable-length call
        out_var, scale_var = self.run_op(
            inputs,
            transpose=False,
            using_pow2_scaling=False,
            use_ue8m0_scale=False,
            output_scale_transpose=False,
        )

        # Reference: use a small batch (fixed-length path) for the first 3
        out_ref, scale_ref = self.run_op(
            inputs[:3],
            transpose=False,
            using_pow2_scaling=False,
            use_ue8m0_scale=False,
            output_scale_transpose=False,
        )

        # The first 3*SMALL_N rows of variable-length output should match
        # the fixed-length output for the same 3 inputs
        np.testing.assert_allclose(
            out_var.numpy()[: 3 * self.SMALL_N, :],
            out_ref.numpy(),
            atol=0,
            rtol=0,
        )


if __name__ == "__main__":
    unittest.main()
