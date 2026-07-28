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
"""Unit tests for GroupedOutputFP8.

Tests cover:
* Forward numerical correctness vs bf16 einsum baseline.
* Backward (dgrad + wgrad) correctness vs autograd on the bf16 path.
* fp8_wgrad=True vs fp8_wgrad=False produce close results.
* save_original_input=True vs False produce identical forward output.
* Shape / assertion checks for invalid inputs.

Run with:
    PYTHONPATH=ernie/erniebot/third_party/PaddleFleet/src:$PYTHONPATH \
    python -m pytest tests/single_card_tests/fp8/test_grouped_output_fp8.py -v
"""

from __future__ import annotations

import unittest

import paddle

from paddlefleet.transformer.dsv4_hybrid_attention import GroupedOutputFP8

_HAS_GPU = (
    paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
)
_REQUIRE_GPU = unittest.skipUnless(
    _HAS_GPU, "GroupedOutputFP8 requires a CUDA device"
)

try:
    from paddlefleet_ops import deep_gemm

    _DEEP_GEMM_AVAILABLE = hasattr(deep_gemm, "fp8_einsum")
except (ImportError, RuntimeError):
    _DEEP_GEMM_AVAILABLE = False

# fp8_einsum requires SM100+ (Blackwell). On older GPUs (e.g. H20/SM90)
# the function exists but raises "Unsupported architecture" at runtime.
_SM100_PLUS = _HAS_GPU and paddle.device.cuda.get_device_capability()[0] >= 10

_REQUIRE_DEEP_GEMM = unittest.skipUnless(
    _DEEP_GEMM_AVAILABLE and _SM100_PLUS,
    "deep_gemm.fp8_einsum requires SM100+ (Blackwell) GPU",
)


def _bf16_grouped_output(x, weight, num_groups, o_lora_rank):
    """Reference bf16 grouped output: einsum '...gd,grd->...gr'."""
    b, sq, g, d = x.shape
    w = weight.reshape([num_groups, o_lora_rank, d])
    out = paddle.einsum("bsgd,grd->bsgr", x, w)
    return out.reshape([b, sq, num_groups * o_lora_rank])


class TestGroupedOutputFP8Forward(unittest.TestCase):
    """Forward pass correctness."""

    @_REQUIRE_GPU
    @_REQUIRE_DEEP_GEMM
    def test_forward_close_to_bf16(self):
        """FP8 forward output should be close to bf16 einsum baseline."""
        paddle.seed(42)
        b, sq, num_groups, d, o_lora_rank = 1, 128, 8, 256, 128
        x = paddle.randn([b, sq, num_groups, d], dtype="bfloat16")
        weight = paddle.randn([num_groups * o_lora_rank, d], dtype="bfloat16")

        fp8_out = GroupedOutputFP8.apply(
            x, weight, num_groups, o_lora_rank, True, False
        )
        bf16_out = _bf16_grouped_output(x, weight, num_groups, o_lora_rank)

        diff = (fp8_out.float() - bf16_out.float()).abs().max().item()
        rel_err = diff / (bf16_out.float().abs().max().item() + 1e-8)
        self.assertLess(
            rel_err, 0.05, f"FP8 forward too far from bf16: rel_err={rel_err}"
        )

    @_REQUIRE_GPU
    @_REQUIRE_DEEP_GEMM
    def test_forward_save_original_input_matches(self):
        """save_original_input=True and False should give identical forward."""
        paddle.seed(7)
        b, sq, num_groups, d, o_lora_rank = 1, 128, 4, 256, 128
        x = paddle.randn([b, sq, num_groups, d], dtype="bfloat16")
        weight = paddle.randn([num_groups * o_lora_rank, d], dtype="bfloat16")

        out_default = GroupedOutputFP8.apply(
            x, weight, num_groups, o_lora_rank, True, False
        )
        out_save = GroupedOutputFP8.apply(
            x, weight, num_groups, o_lora_rank, True, True
        )
        self.assertTrue(
            paddle.equal_all(
                out_default.astype("float32"), out_save.astype("float32")
            ).item(),
            "save_original_input should not affect forward output",
        )

    @_REQUIRE_GPU
    @_REQUIRE_DEEP_GEMM
    def test_forward_output_shape(self):
        """Output shape should be [b, sq, num_groups * o_lora_rank]."""
        b, sq, num_groups, d, o_lora_rank = 2, 128, 8, 256, 128
        x = paddle.randn([b, sq, num_groups, d], dtype="bfloat16")
        weight = paddle.randn([num_groups * o_lora_rank, d], dtype="bfloat16")

        out = GroupedOutputFP8.apply(
            x, weight, num_groups, o_lora_rank, True, False
        )
        self.assertEqual(list(out.shape), [b, sq, num_groups * o_lora_rank])


class TestGroupedOutputFP8Backward(unittest.TestCase):
    """Backward pass correctness (dgrad + wgrad)."""

    @_REQUIRE_GPU
    @_REQUIRE_DEEP_GEMM
    def test_backward_dgrad_close_to_bf16(self):
        """dgrad from FP8 backward should be close to autograd on bf16."""
        paddle.seed(123)
        b, sq, num_groups, d, o_lora_rank = 1, 128, 4, 256, 128
        x_data = paddle.randn([b, sq, num_groups, d], dtype="bfloat16")
        weight_data = paddle.randn(
            [num_groups * o_lora_rank, d], dtype="bfloat16"
        )

        # FP8 path
        x_fp8 = x_data.clone().detach()
        x_fp8.stop_gradient = False
        w_fp8 = weight_data.clone().detach()
        w_fp8.stop_gradient = False
        out_fp8 = GroupedOutputFP8.apply(
            x_fp8, w_fp8, num_groups, o_lora_rank, True, True
        )
        loss_fp8 = out_fp8.sum()
        loss_fp8.backward()

        # BF16 reference path
        x_bf16 = x_data.clone().detach()
        x_bf16.stop_gradient = False
        w_bf16 = weight_data.clone().detach()
        w_bf16.stop_gradient = False
        out_bf16 = _bf16_grouped_output(x_bf16, w_bf16, num_groups, o_lora_rank)
        loss_bf16 = out_bf16.sum()
        loss_bf16.backward()

        # Compare dgrad
        dgrad_diff = (
            (x_fp8.grad.float() - x_bf16.grad.float()).abs().max().item()
        )
        dgrad_ref = x_bf16.grad.float().abs().max().item() + 1e-8
        rel_err = dgrad_diff / dgrad_ref
        self.assertLess(rel_err, 0.1, f"dgrad rel_err={rel_err} too large")

    @_REQUIRE_GPU
    @_REQUIRE_DEEP_GEMM
    def test_backward_wgrad_close_to_bf16(self):
        """wgrad from FP8 backward should be close to autograd on bf16."""
        paddle.seed(456)
        b, sq, num_groups, d, o_lora_rank = 1, 128, 4, 256, 128
        x_data = paddle.randn([b, sq, num_groups, d], dtype="bfloat16")
        weight_data = paddle.randn(
            [num_groups * o_lora_rank, d], dtype="bfloat16"
        )

        # FP8 path with fp8_wgrad=True
        x_fp8 = x_data.clone().detach()
        x_fp8.stop_gradient = False
        w_fp8 = weight_data.clone().detach()
        w_fp8.stop_gradient = False
        out_fp8 = GroupedOutputFP8.apply(
            x_fp8, w_fp8, num_groups, o_lora_rank, True, True
        )
        loss_fp8 = out_fp8.sum()
        loss_fp8.backward()

        # BF16 reference path
        x_bf16 = x_data.clone().detach()
        x_bf16.stop_gradient = False
        w_bf16 = weight_data.clone().detach()
        w_bf16.stop_gradient = False
        out_bf16 = _bf16_grouped_output(x_bf16, w_bf16, num_groups, o_lora_rank)
        loss_bf16 = out_bf16.sum()
        loss_bf16.backward()

        # Compare wgrad
        wgrad_diff = (
            (w_fp8.grad.float() - w_bf16.grad.float()).abs().max().item()
        )
        wgrad_ref = w_bf16.grad.float().abs().max().item() + 1e-8
        rel_err = wgrad_diff / wgrad_ref
        self.assertLess(rel_err, 0.1, f"wgrad rel_err={rel_err} too large")

    @_REQUIRE_GPU
    @_REQUIRE_DEEP_GEMM
    def test_fp8_wgrad_false_produces_valid_wgrad(self):
        """fp8_wgrad=False falls back to bf16 einsum for wgrad."""
        paddle.seed(789)
        b, sq, num_groups, d, o_lora_rank = 1, 128, 4, 256, 128
        x_data = paddle.randn([b, sq, num_groups, d], dtype="bfloat16")
        weight_data = paddle.randn(
            [num_groups * o_lora_rank, d], dtype="bfloat16"
        )

        x = x_data.clone().detach()
        x.stop_gradient = False
        w = weight_data.clone().detach()
        w.stop_gradient = False
        out = GroupedOutputFP8.apply(x, w, num_groups, o_lora_rank, False, True)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(w.grad)
        self.assertEqual(list(w.grad.shape), list(weight_data.shape))
        # wgrad should not be all zeros
        self.assertGreater(w.grad.abs().max().item(), 0.0)

    @_REQUIRE_GPU
    @_REQUIRE_DEEP_GEMM
    def test_fp8_wgrad_false_save_original_input_false(self):
        """fp8_wgrad=False + save_original_input=False uses dequant fallback."""
        paddle.seed(101)
        b, sq, num_groups, d, o_lora_rank = 1, 128, 4, 256, 128
        x_data = paddle.randn([b, sq, num_groups, d], dtype="bfloat16")
        weight_data = paddle.randn(
            [num_groups * o_lora_rank, d], dtype="bfloat16"
        )

        x = x_data.clone().detach()
        x.stop_gradient = False
        w = weight_data.clone().detach()
        w.stop_gradient = False
        out = GroupedOutputFP8.apply(
            x, w, num_groups, o_lora_rank, False, False
        )
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(w.grad)
        self.assertEqual(list(w.grad.shape), list(weight_data.shape))
        self.assertGreater(w.grad.abs().max().item(), 0.0)

    """Input validation checks."""

    @_REQUIRE_GPU
    @_REQUIRE_DEEP_GEMM
    def test_d_not_divisible_by_128_raises(self):
        """Per-group hidden dim must be divisible by 128."""
        b, sq, num_groups, d, o_lora_rank = 1, 128, 4, 192, 128
        x = paddle.randn([b, sq, num_groups, d], dtype="bfloat16")
        weight = paddle.randn([num_groups * o_lora_rank, d], dtype="bfloat16")
        with self.assertRaises(AssertionError):
            GroupedOutputFP8.apply(
                x, weight, num_groups, o_lora_rank, True, False
            )

    @_REQUIRE_GPU
    @_REQUIRE_DEEP_GEMM
    def test_o_lora_rank_not_divisible_by_128_raises(self):
        """o_lora_rank must be divisible by 128."""
        b, sq, num_groups, d, o_lora_rank = 1, 128, 4, 256, 64
        x = paddle.randn([b, sq, num_groups, d], dtype="bfloat16")
        weight = paddle.randn([num_groups * o_lora_rank, d], dtype="bfloat16")
        with self.assertRaises(AssertionError):
            GroupedOutputFP8.apply(
                x, weight, num_groups, o_lora_rank, True, False
            )


if __name__ == "__main__":
    unittest.main()
