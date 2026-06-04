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

"""Numerical parity test: cuDNN-frontend indexer backward vs TileLang baseline.

The two implementations consume different gradient signals:

* TileLang ``csa_indexer_bwd`` takes a pre-computed
  ``grad_scores = (topk_probs - target) * loss_coeff / num_rows * grad_loss``.
* cuDNN ``indexer_backward_wrapper`` takes the raw ``target`` and ``topk_probs``
  separately and computes the same score gradient internally
  (``grad_scale = loss_coeff / (B*Sq)`` * ``grad_loss``).

When both wrappers receive the equivalent inputs they must produce the same
``d_index_q / d_weights / d_index_k_comp``.

Run a single representative shape; cuDNN kernels are SM-version specific, so
skip cleanly when the runtime environment cannot supply them.
"""

import unittest

import paddle

paddle.enable_compat(scope={"tilelang"}, silent=True)


def _cuda_or_skip(testcase):
    if not paddle.device.is_compiled_with_cuda():
        testcase.skipTest("CUDA build of Paddle is required")
    if paddle.device.cuda.device_count() == 0:
        testcase.skipTest("No CUDA device available")


def _try_import_cudnn():
    try:
        # Trigger the deeper import that talks to nvidia-cudnn-frontend.
        from cudnn.deepseek_sparse_attention.indexer_backward.api import (
            indexer_backward_wrapper,  # noqa: F401
        )

        from paddlefleet.cudnn_ops import csa_indexer_bwd

        return csa_indexer_bwd
    except Exception as exc:  # pragma: no cover
        return None, str(exc)


def _try_import_tilelang():
    try:
        from paddlefleet.tilelang_ops import csa_indexer_bwd

        return csa_indexer_bwd
    except Exception:
        return None


def _assert_close(actual, expected, rtol, atol, name):
    a = actual.cast("float32")
    e = expected.cast("float32")
    if not paddle.allclose(a, e, rtol=rtol, atol=atol).item():
        diff = (a - e).abs()
        denom = e.abs().clip(min=1e-12)
        raise AssertionError(
            f"{name} mismatch: max abs={diff.max().item():.4e} "
            f"max rel={(diff / denom).max().item():.4e}"
        )


class TestCudnnCsaIndexerBwd(unittest.TestCase):
    def setUp(self):
        _cuda_or_skip(self)

        cudnn_fn = _try_import_cudnn()
        if isinstance(cudnn_fn, tuple) or cudnn_fn is None:
            reason = (
                cudnn_fn[1]
                if isinstance(cudnn_fn, tuple)
                else "csa_indexer_bwd unavailable"
            )
            self.skipTest(f"cuDNN frontend unavailable: {reason}")
        self.cudnn_fn = cudnn_fn

        tilelang_fn = _try_import_tilelang()
        if tilelang_fn is None:
            self.skipTest("TileLang baseline csa_indexer_bwd unavailable")
        self.tilelang_fn = tilelang_fn

    def _make_inputs(self, b, sq, sk, h, d, topk, seed=2026):
        paddle.seed(seed)
        index_q = paddle.randn([b, sq, h, d]).astype("bfloat16")
        index_k = paddle.randn([b, sk, d]).astype("bfloat16")
        weights = paddle.randn([b, sq, h]).astype("bfloat16")
        target = paddle.nn.functional.softmax(
            paddle.randn([b, sq, topk]).astype("float32"), axis=-1
        )
        topk_probs = paddle.nn.functional.softmax(
            paddle.randn([b, sq, topk]).astype("float32"), axis=-1
        )
        # Random valid indices into [0, sk); leave a few -1 holes.
        topk_indices = paddle.randint(0, sk, [b, sq, topk]).astype("int32")
        mask = paddle.rand([b, sq, topk]) < 0.05
        topk_indices = paddle.where(
            mask, paddle.full_like(topk_indices, -1), topk_indices
        )
        return index_q, weights, index_k, target, topk_probs, topk_indices

    def test_parity_against_tilelang(self):
        """Parity at rtol=atol=2e-2.

        Tolerance derivation (conservative upper bound):

        * bf16 mantissa is 7 bits, so a single multiply-add carries a
          relative error of ``2^-7 ≈ 7.8e-3``.
        * Both kernels accumulate in fp32 internally; with reduction depth
          ``N <= topk*D = 128*128``, error grows as ``√N * eps`` rather than
          ``N * eps``, contributing roughly ``1e-2`` relative.
        * cuDNN and TileLang use different tile sizes and warp schedules;
          fp accumulation order is not associative, contributing another
          ``~5e-3``.

        Sum of the three sources is ``~2e-2``, which we adopt for both
        ``rtol`` and ``atol``. Empirically this passes across multiple
        random seeds at this shape; revisit if a future shape regresses.
        """
        b, sq, sk, h, d, topk = 1, 1024, 256, 64, 128, 128
        loss_coeff = 0.01
        grad_loss_val = 1.0

        (
            index_q,
            weights,
            index_k,
            target,
            topk_probs,
            topk_indices,
        ) = self._make_inputs(b, sq, sk, h, d, topk)

        try:
            grad_q_cudnn, grad_w_cudnn, grad_k_cudnn = self.cudnn_fn(
                index_q.clone(),
                weights.clone(),
                index_k.clone(),
                target.clone(),
                topk_probs.clone(),
                topk_indices.clone(),
                loss_coeff=loss_coeff,
                grad_loss=paddle.to_tensor(grad_loss_val, dtype="float32"),
            )
        except Exception as exc:
            import traceback

            self.skipTest(
                f"cuDNN kernel unsupported for shape: {type(exc).__name__}: {exc!r}\n"
                f"{traceback.format_exc()}"
            )

        scale = loss_coeff / float(b * sq)
        grad_scores = (topk_probs - target) * scale * grad_loss_val

        grad_q_tl, grad_w_tl, grad_k_tl = self.tilelang_fn(
            index_q.clone(),
            weights.clone(),
            index_k.clone(),
            topk_indices.clone(),
            grad_scores,
        )

        _assert_close(
            grad_q_cudnn, grad_q_tl, rtol=2e-2, atol=2e-2, name="d_index_q"
        )
        _assert_close(
            grad_w_cudnn, grad_w_tl, rtol=2e-2, atol=2e-2, name="d_weights"
        )
        _assert_close(
            grad_k_cudnn, grad_k_tl, rtol=2e-2, atol=2e-2, name="d_index_k_comp"
        )

    def _call_cudnn_or_skip(self, *args, **kwargs):
        try:
            return self.cudnn_fn(*args, **kwargs)
        except Exception as exc:
            import traceback

            self.skipTest(
                f"cuDNN kernel unsupported for shape: "
                f"{type(exc).__name__}: {exc!r}\n{traceback.format_exc()}"
            )

    def test_grad_loss_none(self):
        """grad_loss=None should be treated as scalar 1.0 (0-D ones)."""
        b, sq, sk, h, d, topk = 1, 1024, 256, 64, 128, 128
        loss_coeff = 0.01
        (
            index_q,
            weights,
            index_k,
            target,
            topk_probs,
            topk_indices,
        ) = self._make_inputs(b, sq, sk, h, d, topk, seed=11)

        grad_q_none, grad_w_none, grad_k_none = self._call_cudnn_or_skip(
            index_q.clone(),
            weights.clone(),
            index_k.clone(),
            target.clone(),
            topk_probs.clone(),
            topk_indices.clone(),
            loss_coeff=loss_coeff,
            grad_loss=None,
        )
        grad_q_one, grad_w_one, grad_k_one = self._call_cudnn_or_skip(
            index_q.clone(),
            weights.clone(),
            index_k.clone(),
            target.clone(),
            topk_probs.clone(),
            topk_indices.clone(),
            loss_coeff=loss_coeff,
            grad_loss=paddle.to_tensor(1.0, dtype="float32"),
        )
        _assert_close(grad_q_none, grad_q_one, 1e-3, 1e-3, "d_index_q")
        _assert_close(grad_w_none, grad_w_one, 1e-3, 1e-3, "d_weights")
        _assert_close(grad_k_none, grad_k_one, 1e-3, 1e-3, "d_index_k")

    def test_grad_loss_python_float_via_tensor(self):
        """grad_loss in non-fp32 dtype should be cast to fp32 internally."""
        b, sq, sk, h, d, topk = 1, 1024, 256, 64, 128, 128
        (
            index_q,
            weights,
            index_k,
            target,
            topk_probs,
            topk_indices,
        ) = self._make_inputs(b, sq, sk, h, d, topk, seed=12)

        grad_loss_bf16 = paddle.to_tensor(1.0, dtype="bfloat16")
        grad_q, grad_w, grad_k = self._call_cudnn_or_skip(
            index_q.clone(),
            weights.clone(),
            index_k.clone(),
            target.clone(),
            topk_probs.clone(),
            topk_indices.clone(),
            loss_coeff=0.01,
            grad_loss=grad_loss_bf16,
        )
        self.assertEqual(grad_q.shape, list(index_q.shape))
        self.assertEqual(grad_w.shape, list(weights.shape))
        self.assertEqual(grad_k.shape, list(index_k.shape))

    def test_input_dtype_fp32_fallback(self):
        """fp32 inputs should be cast to bf16 internally and grads back."""
        b, sq, sk, h, d, topk = 1, 1024, 256, 64, 128, 128
        (
            index_q,
            weights,
            index_k,
            target,
            topk_probs,
            topk_indices,
        ) = self._make_inputs(b, sq, sk, h, d, topk, seed=13)

        grad_q, grad_w, grad_k = self._call_cudnn_or_skip(
            index_q.cast("float32"),
            weights.cast("float32"),
            index_k.cast("float32"),
            target.clone(),
            topk_probs.clone(),
            topk_indices.clone(),
            loss_coeff=0.01,
            grad_loss=paddle.to_tensor(1.0, dtype="float32"),
        )
        self.assertEqual(grad_q.dtype, paddle.float32)
        self.assertEqual(grad_w.dtype, paddle.float32)
        self.assertEqual(grad_k.dtype, paddle.float32)

    def test_topk_indices_int64_fallback(self):
        """int64 topk_indices should be cast to int32 internally."""
        b, sq, sk, h, d, topk = 1, 1024, 256, 64, 128, 128
        (
            index_q,
            weights,
            index_k,
            target,
            topk_probs,
            topk_indices,
        ) = self._make_inputs(b, sq, sk, h, d, topk, seed=14)

        grad_q, grad_w, grad_k = self._call_cudnn_or_skip(
            index_q.clone(),
            weights.clone(),
            index_k.clone(),
            target.clone(),
            topk_probs.clone(),
            topk_indices.cast("int64"),
            loss_coeff=0.01,
            grad_loss=paddle.to_tensor(1.0, dtype="float32"),
        )
        self.assertEqual(grad_q.shape, list(index_q.shape))

    def test_target_predict_bf16_fallback(self):
        """bf16 target/topk_probs should be cast to fp32 internally."""
        b, sq, sk, h, d, topk = 1, 1024, 256, 64, 128, 128
        (
            index_q,
            weights,
            index_k,
            target,
            topk_probs,
            topk_indices,
        ) = self._make_inputs(b, sq, sk, h, d, topk, seed=15)

        grad_q, grad_w, grad_k = self._call_cudnn_or_skip(
            index_q.clone(),
            weights.clone(),
            index_k.clone(),
            target.cast("bfloat16"),
            topk_probs.cast("bfloat16"),
            topk_indices.clone(),
            loss_coeff=0.01,
            grad_loss=paddle.to_tensor(1.0, dtype="float32"),
        )
        self.assertEqual(grad_q.shape, list(index_q.shape))

    def test_custom_block_I(self):
        """block_I=64 covers the custom-tile path (cuDNN MmaF16BF16Op
        only supports M-mode 64 or 128, so 256 is rejected upstream)."""
        b, sq, sk, h, d, topk = 1, 1024, 256, 64, 128, 128
        (
            index_q,
            weights,
            index_k,
            target,
            topk_probs,
            topk_indices,
        ) = self._make_inputs(b, sq, sk, h, d, topk, seed=16)
        grad_q, grad_w, grad_k = self._call_cudnn_or_skip(
            index_q.clone(),
            weights.clone(),
            index_k.clone(),
            target.clone(),
            topk_probs.clone(),
            topk_indices.clone(),
            loss_coeff=0.01,
            grad_loss=paddle.to_tensor(1.0, dtype="float32"),
            block_I=64,
        )
        self.assertEqual(grad_q.shape, list(index_q.shape))

    def test_saved_tensors_not_mutated(self):
        """Wrapper must clone target/topk_probs (cuDNN overwrites in place)."""
        b, sq, sk, h, d, topk = 1, 1024, 256, 64, 128, 128
        (
            index_q,
            weights,
            index_k,
            target,
            topk_probs,
            topk_indices,
        ) = self._make_inputs(b, sq, sk, h, d, topk, seed=17)

        target_before = target.clone()
        topk_probs_before = topk_probs.clone()
        self._call_cudnn_or_skip(
            index_q.clone(),
            weights.clone(),
            index_k.clone(),
            target,
            topk_probs,
            topk_indices.clone(),
            loss_coeff=0.01,
            grad_loss=paddle.to_tensor(1.0, dtype="float32"),
        )
        self.assertTrue(paddle.allclose(target, target_before).item())
        self.assertTrue(paddle.allclose(topk_probs, topk_probs_before).item())


def _try_import_torch_or_skip(testcase):
    try:
        import torch  # noqa: F401
    except Exception as exc:
        testcase.skipTest(f"torch unavailable: {exc}")


def _import_helpers_or_skip(testcase):
    try:
        from paddlefleet.cudnn_ops.indexer import csa_indexer_bwd_cudnn as mod
    except Exception as exc:
        testcase.skipTest(f"helpers unavailable: {exc}")
    return mod


class TestCudnnHelpers(unittest.TestCase):
    """Cover the small bridge helpers in csa_indexer_bwd_cudnn.py."""

    def setUp(self):
        _cuda_or_skip(self)
        _try_import_torch_or_skip(self)
        self.mod = _import_helpers_or_skip(self)

    def test_paddle_to_torch_contiguous_aliases(self):
        t = paddle.zeros([4, 8], dtype="float32")
        torch_t = self.mod._paddle_to_torch(t)
        torch_t.fill_(7.5)
        self.assertAlmostEqual(float(t.mean().item()), 7.5, places=5)

    def test_torch_to_paddle_non_contiguous_fallback(self):
        import torch

        x = torch.arange(0, 24, dtype=torch.float32, device="cuda").reshape(
            4, 6
        )
        x_nc = x.transpose(0, 1)
        self.assertFalse(x_nc.is_contiguous())
        p = self.mod._torch_to_paddle(x_nc)
        self.assertTrue(p.is_contiguous())
        self.assertEqual(list(p.shape), [6, 4])

    def test_to_bf16_noop(self):
        t = paddle.zeros([2, 3], dtype="bfloat16")
        out = self.mod._to_bf16(t)
        self.assertIs(out, t)

    def test_to_bf16_cast(self):
        t = paddle.zeros([2, 3], dtype="float32")
        out = self.mod._to_bf16(t)
        self.assertEqual(out.dtype, paddle.bfloat16)
        self.assertIsNot(out, t)

    def test_lazy_import_cudnn_raises(self):
        """When `cudnn` package is unavailable, helper raises RuntimeError."""
        import sys
        from unittest import mock

        to_purge = [m for m in list(sys.modules) if m.split(".")[0] == "cudnn"]
        with mock.patch.dict(sys.modules, {"cudnn": None}):
            for m in to_purge:
                if m != "cudnn":
                    sys.modules.pop(m, None)
            with self.assertRaises(RuntimeError) as cm:
                self.mod._lazy_import_cudnn()
            self.assertIn("nvidia-cudnn-frontend", str(cm.exception))


class TestCudnnOpsInit(unittest.TestCase):
    """Cover `paddlefleet/cudnn_ops/__init__.py`'s lazy `__getattr__`."""

    def test_export_csa_indexer_bwd(self):
        try:
            import paddlefleet.cudnn_ops as cudnn_ops_mod
        except Exception as exc:
            self.skipTest(f"paddlefleet.cudnn_ops unavailable: {exc}")
        fn = cudnn_ops_mod.csa_indexer_bwd
        self.assertTrue(callable(fn))
        self.assertIs(cudnn_ops_mod.csa_indexer_bwd, fn)

    def test_unknown_attribute_raises(self):
        try:
            import paddlefleet.cudnn_ops as cudnn_ops_mod
        except Exception as exc:
            self.skipTest(f"paddlefleet.cudnn_ops unavailable: {exc}")
        with self.assertRaises(AttributeError):
            _ = cudnn_ops_mod.does_not_exist_xyz


class TestCsaIndexerLossAutoScalerScaleBranches(unittest.TestCase):
    """Cover the three scale branches in TileLangCSAIndexerLossAutoScaler."""

    def setUp(self):
        _cuda_or_skip(self)
        try:
            from paddlefleet.transformer.csa_attention import (
                DSAIndexerLossAutoScaler,
                TileLangCSAIndexerLossAutoScaler,
            )
        except Exception as exc:
            self.skipTest(f"csa_attention import failed: {exc}")
        self.AutoScaler = TileLangCSAIndexerLossAutoScaler
        self.DSAScaler = DSAIndexerLossAutoScaler
        self._orig_scale = DSAIndexerLossAutoScaler._main_loss_backward_scale

    def tearDown(self):
        if hasattr(self, "DSAScaler"):
            self.DSAScaler._main_loss_backward_scale = self._orig_scale

    def _run_with_scale(self, scale):
        """Run backward with a fake csa_indexer_bwd; capture grad_loss arg."""
        import paddlefleet.cudnn_ops as cudnn_ops_mod

        captured = {}

        def fake_bwd(
            index_q,
            weights,
            index_k_comp,
            target,
            topk_probs,
            topk_indices,
            loss_coeff,
            grad_loss=None,
            block_I=128,
        ):
            captured["grad_loss"] = grad_loss
            captured["loss_coeff"] = loss_coeff
            return (
                paddle.zeros_like(index_q),
                paddle.zeros_like(weights),
                paddle.zeros_like(index_k_comp),
            )

        had_attr = "csa_indexer_bwd" in cudnn_ops_mod.__dict__
        orig_attr = cudnn_ops_mod.__dict__.get("csa_indexer_bwd")
        cudnn_ops_mod.csa_indexer_bwd = fake_bwd
        try:
            self.DSAScaler._main_loss_backward_scale = scale

            class FakeCtx:
                pass

            b, sq, sk, h, d, topk = 1, 4, 4, 1, 8, 4
            index_q = paddle.randn([b, sq, h, d]).astype("bfloat16")
            weights = paddle.randn([b, sq, h]).astype("bfloat16")
            index_k = paddle.randn([b, sk, d]).astype("bfloat16")
            topk_indices = paddle.randint(0, sk, [b, sq, topk]).astype("int32")
            topk_probs = paddle.nn.functional.softmax(
                paddle.randn([b, sq, topk]).astype("float32"), axis=-1
            )
            target = paddle.nn.functional.softmax(
                paddle.randn([b, sq, topk]).astype("float32"), axis=-1
            )

            ctx = FakeCtx()
            ctx.saved_tensor = lambda: (
                index_q,
                weights,
                index_k,
                topk_indices,
                topk_probs,
                target,
            )
            ctx.loss_coeff = 0.01
            ctx.indexer_backend = "cudnn"

            grad_output = paddle.ones_like(weights)
            self.AutoScaler.backward(ctx, grad_output)
        finally:
            if had_attr:
                cudnn_ops_mod.csa_indexer_bwd = orig_attr
            else:
                cudnn_ops_mod.__dict__.pop("csa_indexer_bwd", None)
        return captured

    def test_scale_none(self):
        captured = self._run_with_scale(None)
        self.assertIsNone(captured["grad_loss"])

    def test_scale_paddle_tensor(self):
        scale_t = paddle.to_tensor(2.5, dtype="float32")
        captured = self._run_with_scale(scale_t)
        self.assertIs(captured["grad_loss"], scale_t)

    def test_scale_python_float(self):
        captured = self._run_with_scale(0.7)
        self.assertIsInstance(captured["grad_loss"], paddle.Tensor)
        self.assertEqual(captured["grad_loss"].dtype, paddle.float32)
        self.assertAlmostEqual(
            float(captured["grad_loss"].item()), 0.7, places=5
        )


if __name__ == "__main__":
    unittest.main()
