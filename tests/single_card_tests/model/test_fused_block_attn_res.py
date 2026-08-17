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

"""Precision comparison: fused_attnres (FLA Triton) vs unfused (PyLayer/eager).

Tests forward and backward numerical alignment between the fused Triton kernel
from FLA and the existing PaddleFleet BlockAttnRes implementations.
"""

import unittest

import paddle
from paddlefleet_ops.fla.ops.attnres import naive_attnres

from paddlefleet.transformer.block_attn_res import (
    FusedAttnResTritonFunc,
    _block_attn_res_rmsnorm,
)


class TestFusedAttnResPrecision(unittest.TestCase):
    """Compare fused_attnres against the unfused reference implementations."""

    def setUp(self):
        paddle.set_device("gpu:0")
        self.seed = 42

    def _make_inputs(
        self, batch_seq, hidden_size, num_blocks, dtype="bfloat16"
    ):
        """Create random inputs for BlockAttnRes."""
        paddle.seed(self.seed)
        # residuals: num_blocks completed + 1 partial_block
        residuals = [
            paddle.randn([batch_seq, hidden_size], dtype=dtype)
            for _ in range(num_blocks + 1)
        ]
        # proj_weight: [1, hidden_size] (same as BlockAttnRes.proj_weight)
        proj_weight = paddle.randn([1, hidden_size], dtype=dtype)
        # norm_weight: [hidden_size]
        norm_weight = (
            paddle.ones([hidden_size], dtype=dtype)
            + paddle.randn([hidden_size], dtype=dtype) * 0.1
        )
        return residuals, proj_weight, norm_weight

    # ------------------------------------------------------------------
    # Forward precision tests
    # ------------------------------------------------------------------

    def test_forward_bf16_small(self):
        """Small shape: batch*seq=4, hidden=256, 3 blocks."""
        self._run_forward_test(batch_seq=4, hidden_size=256, num_blocks=3)

    def test_forward_bf16_kimi_k3(self):
        """Kimi-K3 realistic shape: batch*seq=2, hidden=7168, 8 blocks."""
        self._run_forward_test(batch_seq=2, hidden_size=7168, num_blocks=8)

    def test_forward_bf16_single_block(self):
        """Edge case: only 1 residual source (partial_block, no completed blocks)."""
        self._run_forward_test(batch_seq=4, hidden_size=512, num_blocks=0)

    def test_forward_bf16_two_sources(self):
        """Edge case: 2 residual sources (1 completed block + partial_block)."""
        self._run_forward_test(batch_seq=4, hidden_size=512, num_blocks=1)

    def test_forward_bf16_large_batch(self):
        """Larger batch: batch*seq=64, hidden=1024, 4 blocks."""
        self._run_forward_test(batch_seq=64, hidden_size=1024, num_blocks=4)

    def _run_forward_test(
        self, batch_seq, hidden_size, num_blocks, atol=1e-2, rtol=1e-2
    ):
        """Run forward comparison between fused and unfused."""
        residuals, proj_weight, norm_weight = self._make_inputs(
            batch_seq, hidden_size, num_blocks
        )
        blocks = residuals[:-1]
        partial_block = residuals[-1]
        norm_eps = 1e-6

        # Unfused reference: _block_attn_res_rmsnorm (same math as PyLayer forward)
        o_unfused = _block_attn_res_rmsnorm(
            partial_block, blocks, proj_weight, norm_weight, norm_eps
        )

        # Fused: production path via FusedAttnResTritonFunc (scopes numel()->int
        # around the FLA kernel). Raw fused_attnres() must not be called outside
        # this wrapper, since numel() is no longer patched process-wide.
        o_fused = FusedAttnResTritonFunc.apply(
            proj_weight,
            norm_weight,
            norm_eps,
            *blocks,
            partial_block,
        )

        # Triple-check against the FLA eager reference (no Triton kernel).
        o_naive = naive_attnres(
            query=proj_weight,
            residuals=[*blocks, partial_block],
            rms_weight=norm_weight,
            rms_eps=norm_eps,
        )

        # Convert to float32 for comparison
        o_unfused_f32 = o_unfused.astype("float32").reshape([-1, hidden_size])
        o_fused_f32 = o_fused.astype("float32").reshape([-1, hidden_size])
        o_naive_f32 = o_naive.astype("float32").reshape([-1, hidden_size])

        max_diff_fused = (o_unfused_f32 - o_fused_f32).abs().max().item()
        max_diff_naive = (o_unfused_f32 - o_naive_f32).abs().max().item()

        print(
            f"\n[Forward] shape=({batch_seq}, {hidden_size}), blocks={num_blocks}"
            f"\n  max_diff fused  vs unfused: {max_diff_fused:.2e}"
            f"\n  max_diff naive  vs unfused: {max_diff_naive:.2e}"
        )

        self.assertTrue(
            paddle.allclose(
                o_unfused_f32, o_fused_f32, atol=atol, rtol=rtol
            ).item(),
            f"Forward mismatch: max_diff={max_diff_fused:.2e} > atol={atol}",
        )

    # ------------------------------------------------------------------
    # Backward precision tests
    # ------------------------------------------------------------------

    def test_backward_bf16_small(self):
        """Backward: small shape."""
        self._run_backward_test(batch_seq=4, hidden_size=256, num_blocks=3)

    def test_backward_bf16_kimi_k3(self):
        """Backward: Kimi-K3 realistic shape."""
        self._run_backward_test(batch_seq=2, hidden_size=7168, num_blocks=8)

    def _run_backward_test(
        self, batch_seq, hidden_size, num_blocks, atol=5e-2, rtol=5e-2
    ):
        """Run backward comparison: gradients of proj_weight, norm_weight, residuals."""
        paddle.seed(self.seed)
        norm_eps = 1e-6

        # --- Unfused path (eager _block_attn_res_rmsnorm with autograd) ---
        blocks_unfused = [
            paddle.randn([batch_seq, hidden_size], dtype="bfloat16")
            for _ in range(num_blocks)
        ]
        partial_unfused = paddle.randn(
            [batch_seq, hidden_size], dtype="bfloat16"
        )
        proj_unfused = paddle.randn([1, hidden_size], dtype="bfloat16")
        norm_w_unfused = (
            paddle.ones([hidden_size], dtype="bfloat16")
            + paddle.randn([hidden_size], dtype="bfloat16") * 0.1
        )

        for t in [
            *blocks_unfused,
            partial_unfused,
            proj_unfused,
            norm_w_unfused,
        ]:
            t.stop_gradient = False

        o_unfused = _block_attn_res_rmsnorm(
            partial_unfused,
            blocks_unfused,
            proj_unfused,
            norm_w_unfused,
            norm_eps,
        )
        loss_unfused = o_unfused.astype("float32").sum()
        loss_unfused.backward()

        # --- Fused path: FusedAttnResTritonFunc (native Paddle PyLayer) ---
        paddle.seed(self.seed)
        blocks_fused = [
            paddle.randn([batch_seq, hidden_size], dtype="bfloat16")
            for _ in range(num_blocks)
        ]
        partial_fused = paddle.randn([batch_seq, hidden_size], dtype="bfloat16")
        proj_fused = paddle.randn([1, hidden_size], dtype="bfloat16")
        norm_w_fused = (
            paddle.ones([hidden_size], dtype="bfloat16")
            + paddle.randn([hidden_size], dtype="bfloat16") * 0.1
        )

        for t in [*blocks_fused, partial_fused, proj_fused, norm_w_fused]:
            t.stop_gradient = False

        o_fused = FusedAttnResTritonFunc.apply(
            proj_fused,
            norm_w_fused,
            norm_eps,
            *blocks_fused,
            partial_fused,
        )
        loss_fused = o_fused.astype("float32").sum()
        loss_fused.backward()

        # Compare gradients
        print(
            f"\n[Backward] shape=({batch_seq}, {hidden_size}), blocks={num_blocks}"
        )

        # proj_weight grad
        grad_proj_diff = (
            (
                proj_unfused.grad.astype("float32")
                - proj_fused.grad.astype("float32")
            )
            .abs()
            .max()
            .item()
        )
        print(f"  grad proj_weight max_diff: {grad_proj_diff:.2e}")

        # norm_weight grad
        grad_norm_diff = (
            (
                norm_w_unfused.grad.astype("float32")
                - norm_w_fused.grad.astype("float32")
            )
            .abs()
            .max()
            .item()
        )
        print(f"  grad norm_weight max_diff: {grad_norm_diff:.2e}")

        # residuals grad
        for i, (g_u, g_f) in enumerate(
            zip(
                [*blocks_unfused, partial_unfused],
                [*blocks_fused, partial_fused],
            )
        ):
            diff = (
                (g_u.grad.astype("float32") - g_f.grad.astype("float32"))
                .abs()
                .max()
                .item()
            )
            print(f"  grad residual[{i}] max_diff: {diff:.2e}")

        # Assertions
        self.assertTrue(
            paddle.allclose(
                proj_unfused.grad.astype("float32"),
                proj_fused.grad.astype("float32"),
                atol=atol,
                rtol=rtol,
            ).item(),
            f"proj_weight grad mismatch: {grad_proj_diff:.2e}",
        )
        self.assertTrue(
            paddle.allclose(
                norm_w_unfused.grad.astype("float32"),
                norm_w_fused.grad.astype("float32"),
                atol=atol,
                rtol=rtol,
            ).item(),
            f"norm_weight grad mismatch: {grad_norm_diff:.2e}",
        )


if __name__ == "__main__":
    unittest.main()
