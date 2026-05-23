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
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import unittest
from unittest.mock import MagicMock, patch

import paddle

# ----------------------------------------------------------------------------
# Original tests (preserved verbatim from the pre-PR version of this file).
# Do not modify or remove.
# ----------------------------------------------------------------------------


class TestFusedSwigluScaleForward(unittest.TestCase):
    """Tests for fused_swiglu_scale_forward."""

    def test_forward_cpu_fallback(self):
        """Test fused_swiglu_scale_forward CPU fallback path."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        x = paddle.randn([4, 16])
        scale = paddle.ones([4, 1])
        result = fused_swiglu_scale_forward(x, scale)
        self.assertEqual(result.shape, [4, 8])

    def test_forward_with_1d_scale(self):
        """Test with 1D scale tensor."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        x = paddle.randn([4, 16])
        scale = paddle.ones([4])
        result = fused_swiglu_scale_forward(x, scale)
        self.assertEqual(result.shape, [4, 8])

    def test_forward_scale_broadcast(self):
        """Test scale broadcasting."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        x = paddle.randn([4, 16])
        scale = paddle.full([4, 1], 2.0)
        result = fused_swiglu_scale_forward(x, scale)
        self.assertEqual(result.shape, [4, 8])


class TestFusedSwigluScaleBackward(unittest.TestCase):
    """Tests for fused_swiglu_scale_backward."""

    def test_backward_cpu_fallback(self):
        """Test fused_swiglu_scale_backward CPU fallback path."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        x = paddle.randn([4, 16])
        scale = paddle.ones([4, 1])
        out_grad = paddle.randn([4, 8])
        d_x, d_scale = fused_swiglu_scale_backward(x, scale, out_grad)
        self.assertEqual(d_x.shape, [4, 16])
        self.assertEqual(d_scale.shape, [4, 1])

    def test_backward_shapes(self):
        """Test backward output shapes match inputs."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        x = paddle.randn([2, 32])
        scale = paddle.ones([2, 1])
        out_grad = paddle.randn([2, 16])
        d_x, d_scale = fused_swiglu_scale_backward(x, scale, out_grad)
        self.assertEqual(d_x.shape, [2, 32])
        self.assertEqual(d_scale.shape, [2, 1])


# ----------------------------------------------------------------------------
# New tests added by PR #999 — clamp / GPU-dispatch / static-graph InferShape.
# All additions live below; the original two classes above are untouched.
# ----------------------------------------------------------------------------


def _no_cuda():
    return patch(
        "paddlefleet.fusions.fused_swiglu_scale.paddle.is_compiled_with_cuda",
        return_value=False,
    )


# ============================================================================
# Forward: CPU fallback (covers lines 34-46)
# ============================================================================


class TestFusedSwigluScaleForwardCPUFallback(unittest.TestCase):
    """CPU-fallback path of fused_swiglu_scale_forward."""

    def test_forward_no_clamp(self):
        """Line 40: swiglu(x) path."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        with _no_cuda():
            x = paddle.randn([4, 16])
            scale = paddle.ones([4, 1])
            result = fused_swiglu_scale_forward(x, scale)
            self.assertEqual(result.shape, [4, 8])

    def test_forward_with_clamp(self):
        """Lines 34-38: clamp + silu path."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        with _no_cuda():
            x = paddle.full([2, 16], 100.0)
            scale = paddle.ones([2, 1])
            result = fused_swiglu_scale_forward(x, scale, clamp_value=1.0)
            self.assertEqual(result.shape, [2, 8])

    def test_forward_scale_expansion_1d(self):
        """Lines 42-44: scale ndim expansion from 1D."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        with _no_cuda():
            x = paddle.randn([4, 16])
            scale = paddle.ones([4])
            result = fused_swiglu_scale_forward(x, scale)
            self.assertEqual(result.shape, [4, 8])

    def test_forward_scale_expansion_2d(self):
        """Lines 42-44: scale ndim already matches."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        with _no_cuda():
            x = paddle.randn([4, 16])
            scale = paddle.ones([4, 1])
            result = fused_swiglu_scale_forward(x, scale)
            self.assertEqual(result.shape, [4, 8])

    def test_forward_return_scaled(self):
        """Line 46: return out * scale_exp."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        with _no_cuda():
            x = paddle.randn([2, 16])
            scale = paddle.full([2, 1], 2.0)
            result = fused_swiglu_scale_forward(x, scale)
            self.assertEqual(result.shape, [2, 8])


# ============================================================================
# Forward: GPU branch (covers lines 23, 25, 27, 29)
# ============================================================================


class TestFusedSwigluScaleForwardGPUBranch(unittest.TestCase):
    """GPU dispatch branch (lines 22-29)."""

    def test_gpu_clamp_path(self):
        """Lines 23, 25: clamp_value set -> calls fused_swiglu_scale_clamp."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        mock_op = MagicMock(return_value=paddle.randn([2, 8]))
        with (
            patch.object(paddle, "is_compiled_with_cuda", return_value=True),
            patch.dict(
                "sys.modules",
                {
                    "paddlefleet_ops": MagicMock(
                        fused_swiglu_scale_clamp=mock_op
                    )
                },
            ),
        ):
            x = paddle.randn([2, 16])
            scale = paddle.ones([2, 1])
            result = fused_swiglu_scale_forward(x, scale, clamp_value=5.0)
            mock_op.assert_called_once()
            self.assertEqual(result.shape, [2, 8])

    def test_gpu_no_clamp_path(self):
        """Lines 27, 29: no clamp_value -> calls fused_swiglu_scale."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        mock_op = MagicMock(return_value=paddle.randn([2, 8]))
        with (
            patch.object(paddle, "is_compiled_with_cuda", return_value=True),
            patch.dict(
                "sys.modules",
                {"paddlefleet_ops": MagicMock(fused_swiglu_scale=mock_op)},
            ),
        ):
            x = paddle.randn([2, 16])
            scale = paddle.ones([2, 1])
            result = fused_swiglu_scale_forward(x, scale)
            mock_op.assert_called_once()


# ============================================================================
# Backward: CPU fallback (covers lines 65-106)
# ============================================================================


class TestFusedSwigluScaleBackwardCPUFallback(unittest.TestCase):
    """CPU-fallback path of fused_swiglu_scale_backward."""

    def test_backward_no_clamp(self):
        """Lines 65-68, 78-81, 83-85, 87-94, 100, 102-106: no-clamp full path."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        with _no_cuda():
            x = paddle.randn([4, 16])
            scale = paddle.ones([4])
            out_grad = paddle.randn([4, 8])
            d_x, d_scale = fused_swiglu_scale_backward(x, scale, out_grad)
            self.assertEqual(d_x.shape, [4, 16])
            self.assertEqual(d_scale.shape, [4])

    def test_backward_with_clamp_saturated(self):
        """Lines 70-76, 96-98: clamp with saturated inputs (masks zero grads)."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        with _no_cuda():
            cv = 0.5
            x = paddle.concat(
                [
                    paddle.full([2, 4], 5.0),
                    paddle.full([2, 4], 5.0),
                ],
                axis=-1,
            )
            scale = paddle.ones([2])
            out_grad = paddle.randn([2, 4])
            d_x, d_scale = fused_swiglu_scale_backward(
                x, scale, out_grad, clamp_value=cv
            )
            self.assertEqual(d_x.shape, [2, 8])
            self.assertTrue(bool((d_x.abs().sum() == 0).item()))

    def test_backward_with_clamp_inside_window(self):
        """Lines 70-76, 96-98: clamp with values inside window (masks are 1)."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        with _no_cuda():
            cv = 10.0
            x = paddle.randn([2, 8])
            scale = paddle.ones([2])
            out_grad = paddle.randn([2, 4])
            d_x, d_scale = fused_swiglu_scale_backward(
                x, scale, out_grad, clamp_value=cv
            )
            self.assertEqual(d_x.shape, [2, 8])
            self.assertEqual(d_scale.shape, [2])
            self.assertTrue(bool((d_x.abs().sum() > 0).item()))

    def test_backward_scale_expansion(self):
        """Lines 87-89: scale ndim expansion."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        with _no_cuda():
            x = paddle.randn([2, 32])
            scale = paddle.ones([2])
            out_grad = paddle.randn([2, 16])
            d_x, d_scale = fused_swiglu_scale_backward(x, scale, out_grad)
            self.assertEqual(d_x.shape, [2, 32])
            self.assertEqual(d_scale.shape, [2])


# ============================================================================
# Backward: GPU branch (covers lines 52, 54, 58, 60)
# ============================================================================


class TestFusedSwigluScaleBackwardGPUBranch(unittest.TestCase):
    """GPU dispatch branch (lines 50-60)."""

    def test_gpu_clamp_path(self):
        """Lines 52, 54: clamp_value set -> calls fused_swiglu_scale_clamp_bwd."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        mock_op = MagicMock(
            return_value=(paddle.randn([2, 16]), paddle.randn([2]))
        )
        with (
            patch.object(paddle, "is_compiled_with_cuda", return_value=True),
            patch.dict(
                "sys.modules",
                {
                    "paddlefleet_ops": MagicMock(
                        fused_swiglu_scale_clamp_bwd=mock_op
                    )
                },
            ),
        ):
            x = paddle.randn([2, 16])
            scale = paddle.ones([2])
            out_grad = paddle.randn([2, 8])
            d_x, d_scale = fused_swiglu_scale_backward(
                x, scale, out_grad, clamp_value=5.0
            )
            mock_op.assert_called_once()

    def test_gpu_no_clamp_path(self):
        """Lines 58, 60: no clamp_value -> calls fused_swiglu_scale_bwd."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        mock_op = MagicMock(
            return_value=(paddle.randn([2, 16]), paddle.randn([2]))
        )
        with (
            patch.object(paddle, "is_compiled_with_cuda", return_value=True),
            patch.dict(
                "sys.modules",
                {"paddlefleet_ops": MagicMock(fused_swiglu_scale_bwd=mock_op)},
            ),
        ):
            x = paddle.randn([2, 16])
            scale = paddle.ones([2])
            out_grad = paddle.randn([2, 8])
            d_x, d_scale = fused_swiglu_scale_backward(x, scale, out_grad)
            mock_op.assert_called_once()


# ============================================================================
# Static-graph InferShape regression for the new clamp forward op.
#
# fused_swiglu_scale_clamp has 2 inputs (X, Scale) and 1 output of shape
# {rows, hidden2 / 2}, so it must use FusedFwdInferShape / FusedFwdInferDtype
# (not the 3-input FusedGradInferShape used by the *_bwd op). In eager mode a
# wrong InferShape registration would be hidden because the shape comes from
# the kernel return; in static-graph / compiled mode the framework relies on
# InferShape and a wrong registration aborts.
#
# The static-graph regression is run in a *subprocess* so that — on a buggy
# build — a C++ SIGABRT does not take down the entire pytest worker.
# ============================================================================


def _has_op(name):
    if not paddle.is_compiled_with_cuda():
        return False
    try:
        import paddlefleet_ops

        return hasattr(paddlefleet_ops, name)
    except ImportError:
        return False


def _run_static_infer_shape(op_name, hidden2, has_clamp):
    """Spawn a subprocess that builds a static program calling the op and
    prints the inferred output shape. Returns (returncode, stdout, stderr)."""
    import subprocess
    import textwrap

    code = textwrap.dedent(
        f"""
        import paddle
        from paddlefleet_ops import {op_name}

        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name='x', shape=[4, {hidden2}], dtype='float32')
            scale = paddle.static.data(name='scale', shape=[4, 1], dtype='float32')
            out = {op_name}(x, scale{", 5.0" if has_clamp else ""})
            print('SHAPE_OK', list(out.shape))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc.returncode, proc.stdout, proc.stderr


@unittest.skipUnless(
    _has_op("fused_swiglu_scale_clamp"),
    "fused_swiglu_scale_clamp custom op not built / CUDA unavailable",
)
class TestFusedSwigluScaleClampInferShape(unittest.TestCase):
    """Static-graph InferShape regression for the clamp forward op."""

    def test_clamp_forward_infer_shape_halves_hidden(self):
        rc, stdout, stderr = _run_static_infer_shape(
            "fused_swiglu_scale_clamp", hidden2=32, has_clamp=True
        )
        self.assertEqual(
            rc,
            0,
            f"static-graph build of fused_swiglu_scale_clamp crashed "
            f"(likely InferShape registration bug).\nSTDERR:\n{stderr}",
        )
        self.assertIn("SHAPE_OK [4, 16]", stdout, f"stdout was:\n{stdout}")

    def test_clamp_forward_eager_shape_matches(self):
        """Eager-mode sanity check that the kernel produces hidden2/2."""
        from paddlefleet_ops import fused_swiglu_scale_clamp

        x = paddle.randn([4, 32]).astype("bfloat16")
        scale = paddle.ones([4, 1], dtype="float32")
        out = fused_swiglu_scale_clamp(x, scale, 5.0)
        self.assertEqual(out.shape, [4, 16])
        self.assertEqual(out.dtype, x.dtype)


if __name__ == "__main__":
    unittest.main()
