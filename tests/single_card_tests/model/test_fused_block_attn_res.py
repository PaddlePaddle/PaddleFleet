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

The fused cases need a GPU plus the optional FLA extension and are skipped when
either is missing; `TestBlockAttnResFallback` keeps the extension-free fallback
path covered in that case.
"""

import unittest

import paddle

from paddlefleet.transformer.block_attn_res import (
    HAVE_FUSED_ATTNRES,
    BlockAttnResFunc,
    FusedAttnResTritonFunc,
    _block_attn_res_rmsnorm,
)

# The FLA extension is optional in production (block_attn_res.py guards its
# import and falls back), so the tests must not fail at collection time when it
# is absent.  Mirror that guard here instead of importing unconditionally.
try:
    from paddlefleet_ops.fla.ops.attnres import naive_attnres

    _HAVE_NAIVE_ATTNRES = True
except (ImportError, AttributeError):
    naive_attnres = None
    _HAVE_NAIVE_ATTNRES = False

_HAVE_GPU = (
    paddle.device.is_compiled_with_cuda()
    and paddle.device.cuda.device_count() > 0
)

_FUSED_AVAILABLE = _HAVE_GPU and _HAVE_NAIVE_ATTNRES and HAVE_FUSED_ATTNRES
_SKIP_REASON = (
    "fused attnres requires a GPU and the FLA extension "
    f"(gpu={_HAVE_GPU}, naive_attnres={_HAVE_NAIVE_ATTNRES}, "
    f"fused_kernels={HAVE_FUSED_ATTNRES})"
)


@unittest.skipUnless(_FUSED_AVAILABLE, _SKIP_REASON)
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
        # The *residuals gradient tuple is the new interface of this PyLayer,
        # so every entry is checked on its own: a dropped, misshaped or
        # misordered gradient has to fail here rather than just be printed.
        residual_pairs = list(
            zip(
                [*blocks_unfused, partial_unfused],
                [*blocks_fused, partial_fused],
            )
        )
        self.assertEqual(
            len(residual_pairs),
            num_blocks + 1,
            "expected one gradient per completed block plus the partial block",
        )

        for i, (t_unfused, t_fused) in enumerate(residual_pairs):
            name = (
                f"residual[{i}]"
                if i < num_blocks
                else f"residual[{i}] (partial_block)"
            )

            self.assertIsNotNone(
                t_fused.grad,
                f"{name} grad is None: the fused PyLayer returned no gradient",
            )
            self.assertEqual(
                t_fused.grad.shape,
                t_unfused.grad.shape,
                f"{name} grad shape mismatch: "
                f"fused={t_fused.grad.shape} vs unfused={t_unfused.grad.shape}",
            )

            g_unfused = t_unfused.grad.astype("float32")
            g_fused = t_fused.grad.astype("float32")
            diff = (g_unfused - g_fused).abs().max().item()
            print(f"  grad {name} max_diff: {diff:.2e}")

            self.assertTrue(
                paddle.allclose(
                    g_unfused, g_fused, atol=atol, rtol=rtol
                ).item(),
                f"{name} grad mismatch: max_diff={diff:.2e} "
                f"(atol={atol}, rtol={rtol})",
            )

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


class TestBlockAttnResFallback(unittest.TestCase):
    """Cover the extension-free fallback path.

    `_block_attn_res_rmsnorm` and `BlockAttnResFunc` are pure Paddle, so these
    tests must keep running (on CPU if needed) when the FLA extension is
    unavailable and `BlockAttnRes` falls back to `_use_fused = False`.
    """

    def setUp(self):
        paddle.set_device("gpu:0" if _HAVE_GPU else "cpu")
        paddle.seed(42)
        self.norm_eps = 1e-6

    def _make_inputs(self, batch_seq, hidden_size, num_blocks):
        blocks = [
            paddle.randn([batch_seq, hidden_size], dtype="float32")
            for _ in range(num_blocks)
        ]
        partial_block = paddle.randn([batch_seq, hidden_size], dtype="float32")
        proj_weight = paddle.randn([1, hidden_size], dtype="float32")
        norm_weight = (
            paddle.ones([hidden_size], dtype="float32")
            + paddle.randn([hidden_size], dtype="float32") * 0.1
        )
        return blocks, partial_block, proj_weight, norm_weight

    def _reference(self, blocks, partial_block, proj_weight, norm_weight):
        """Independent implementation of the eager `BlockAttnRes.forward` else-branch."""
        all_repr = [*blocks, partial_block]
        logits = []
        for r in all_repr:
            variance = r.pow(2).mean(axis=-1, keepdim=True)
            normed = r * paddle.rsqrt(variance + self.norm_eps) * norm_weight
            logits.append((normed * proj_weight).sum(axis=-1))
        weights = paddle.nn.functional.softmax(
            paddle.stack(logits, axis=0), axis=0
        )
        h = weights[0].unsqueeze(-1) * all_repr[0]
        for i in range(1, len(all_repr)):
            h = h + weights[i].unsqueeze(-1) * all_repr[i]
        return h

    def test_eager_reduction_matches_reference(self):
        """`_block_attn_res_rmsnorm` agrees with the loop-based eager branch."""
        blocks, partial_block, proj_weight, norm_weight = self._make_inputs(
            batch_seq=8, hidden_size=64, num_blocks=3
        )

        got = _block_attn_res_rmsnorm(
            partial_block, blocks, proj_weight, norm_weight, self.norm_eps
        )
        expected = self._reference(
            blocks, partial_block, proj_weight, norm_weight
        )

        diff = (got - expected).abs().max().item()
        self.assertTrue(
            paddle.allclose(got, expected, atol=1e-5, rtol=1e-5).item(),
            f"eager reduction mismatch: max_diff={diff:.2e}",
        )

    def test_pylayer_gradients_match_autograd(self):
        """`BlockAttnResFunc` recomputed gradients match plain autograd.

        Also asserts the per-residual gradient ordering of the fallback
        PyLayer, mirroring the fused-path backward test.
        """
        num_blocks = 3
        blocks, partial_block, proj_weight, norm_weight = self._make_inputs(
            batch_seq=8, hidden_size=64, num_blocks=num_blocks
        )

        def run(use_pylayer):
            tensors = [
                t.detach().clone()
                for t in [*blocks, partial_block, proj_weight, norm_weight]
            ]
            for t in tensors:
                t.stop_gradient = False
            *blk, partial, proj, norm_w = tensors

            if use_pylayer:
                out = BlockAttnResFunc.apply(
                    partial, proj, norm_w, self.norm_eps, *blk
                )
            else:
                out = _block_attn_res_rmsnorm(
                    partial, blk, proj, norm_w, self.norm_eps
                )
            out.sum().backward()
            return out, tensors

        out_eager, tensors_eager = run(use_pylayer=False)
        out_pylayer, tensors_pylayer = run(use_pylayer=True)

        self.assertTrue(
            paddle.allclose(out_eager, out_pylayer, atol=1e-5, rtol=1e-5).item()
        )

        names = [
            *[f"residual[{i}]" for i in range(num_blocks)],
            "residual[-1] (partial_block)",
            "proj_weight",
            "norm_weight",
        ]
        for name, t_eager, t_pylayer in zip(
            names, tensors_eager, tensors_pylayer
        ):
            self.assertIsNotNone(t_pylayer.grad, f"{name} grad is None")
            self.assertEqual(
                t_pylayer.grad.shape,
                t_eager.grad.shape,
                f"{name} grad shape mismatch",
            )
            diff = (t_pylayer.grad - t_eager.grad).abs().max().item()
            self.assertTrue(
                paddle.allclose(
                    t_pylayer.grad, t_eager.grad, atol=1e-5, rtol=1e-5
                ).item(),
                f"{name} grad mismatch: max_diff={diff:.2e}",
            )

    def test_fused_disabled_without_extension(self):
        """`_use_fused` gating: kernel handles must be None when unavailable."""
        from paddlefleet.transformer import block_attn_res

        handles = [
            block_attn_res.fused_attnres_fwd,
            block_attn_res.fused_attnres_bwd,
            block_attn_res._build_ptr_table,
        ]
        if HAVE_FUSED_ATTNRES:
            self.assertTrue(all(h is not None for h in handles))
        else:
            self.assertTrue(all(h is None for h in handles))


if __name__ == "__main__":
    unittest.main()
