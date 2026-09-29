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

"""Behavior tests for ``paddlefleet.triton_ops.sigmoid_gate_fusion``.

Module under test lives in the repository's "计算优化 / Fused Ops" boundary.
The only host-side, CPU-observable pure logic in ``SigmoidGateFusionTriton`` is:
argument validation (shape / dtype match and the supported-dtype membership
check that runs *before* any kernel launch), the ``autograd`` ctx control-flow
(what is saved for backward and the sizing stored on ctx), and the launch
geometry / argument routing computed by ``forward`` and ``backward`` (block
size 1024, ``grid = (cdiv(n_elements, 1024),)``, and the positional order in
which the tensors are handed to the kernel).

The two Triton kernels ``fused_sigmoid_gate_fwd_kernel`` /
``fused_sigmoid_gate_bwd_kernel`` are genuine not-under-test collaborators (they
compile to PTX and require CUDA); each is replaced by a launch spy that records
the grid and the forwarded arguments while performing no computation. These
tests therefore assert only the wrapper's pure-Python geometry, validation and
ctx handling against hand-derived expectations. They do NOT drive or verify the
kernels' numerical output (``out = attn_out * sigmoid(gate)`` and its gradient)
-- that requires a real GPU launch. The whole module is skipped honestly when
it (or its ``paddle`` dependency) cannot be imported.
"""

import unittest

try:
    import paddle

    from paddlefleet.triton_ops import sigmoid_gate_fusion as _sgf
    from paddlefleet.triton_ops.sigmoid_gate_fusion import (
        SigmoidGateFusionTriton,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # only a genuine missing dependency -> honest skip
    _sgf = None
    SigmoidGateFusionTriton = None
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    "paddlefleet.triton_ops.sigmoid_gate_fusion (or its paddle dependency) is "
    f"not importable in this environment: {_IMPORT_ERROR!r}"
)


class _KernelLaunchSpy:
    """Stand-in for a GPU kernel that records how it was launched.

    ``forward`` / ``backward`` invoke the kernel as ``kernel[grid](*args,
    **kwargs)``; ``__getitem__`` captures the grid and returns a callable that
    captures the forwarded arguments. It performs no computation, so the
    wrapper's own buffer allocation and ctx bookkeeping survive for inspection.
    """

    def __init__(self):
        self.grid = None
        self.args = None
        self.kwargs = None
        self.launch_count = 0

    def __getitem__(self, grid):
        self.grid = grid

        def _launch(*args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.launch_count += 1

        return _launch


class _RecordingContext:
    """Captures what ``forward`` saves via the autograd ctx protocol."""

    def __init__(self):
        self.saved = None
        self.n_elements = None
        self.block_size = None

    def save_for_backward(self, *tensors):
        self.saved = tensors


class _BackwardContext:
    """Replays a saved forward ctx for exercising ``backward`` in isolation."""

    def __init__(self, attn_out, gate, n_elements, block_size):
        self._saved = (attn_out, gate)
        self.n_elements = n_elements
        self.block_size = block_size

    def saved_tensor(self):
        return self._saved


@unittest.skipUnless(_sgf is not None, _SKIP_REASON)
class TestForwardLaunchGeometry(unittest.TestCase):
    """Sizing, argument routing and ctx control-flow of ``forward``."""

    def setUp(self):
        self.spy = _KernelLaunchSpy()
        original = _sgf.fused_sigmoid_gate_fwd_kernel
        _sgf.fused_sigmoid_gate_fwd_kernel = self.spy
        self.addCleanup(
            setattr, _sgf, "fused_sigmoid_gate_fwd_kernel", original
        )

    def test_geometry_arg_routing_and_ctx_save(self):
        # Non-square, non-power-of-two element count so a floor/ceil mistake in
        # the grid and an attn_out/gate ordering swap are both observable.
        attn_out = paddle.zeros([40, 64], dtype="float32")
        gate = paddle.ones([40, 64], dtype="float32")
        ctx = _RecordingContext()

        out = SigmoidGateFusionTriton.forward(ctx, attn_out, gate)

        self.assertEqual(self.spy.launch_count, 1)
        # n_elements = 40 * 64 = 2560, block_size = 1024.
        # grid = (cdiv(2560, 1024),) = (3,); floor division would give (2,).
        self.assertEqual(self.spy.grid, (3,))

        args = self.spy.args
        self.assertIs(args[0], attn_out)  # attn_out first, not gate
        self.assertIs(args[1], gate)
        self.assertIs(args[2], out)  # output buffer forwarded and returned
        self.assertEqual(args[3], 2560)  # n_elements taken from attn_out.size
        self.assertEqual(self.spy.kwargs["BLOCK_SIZE"], 1024)

        # Returned buffer is allocated like attn_out (kernel is a no-op spy).
        self.assertEqual(out.shape, [40, 64])
        self.assertEqual(out.dtype, paddle.float32)

        # ctx control-flow: exactly the two forward inputs, in order, plus the
        # sizing backward relies on.
        self.assertEqual(len(ctx.saved), 2)
        self.assertIs(ctx.saved[0], attn_out)
        self.assertIs(ctx.saved[1], gate)
        self.assertEqual(ctx.n_elements, 2560)
        self.assertEqual(ctx.block_size, 1024)

    def test_accepts_each_supported_dtype(self):
        for dtype in ("float32", "float16", "bfloat16"):
            self.spy.launch_count = 0
            attn_out = paddle.zeros([2, 3], dtype=dtype)
            gate = paddle.zeros([2, 3], dtype=dtype)
            SigmoidGateFusionTriton.forward(_RecordingContext(), attn_out, gate)
            # Validation passed and the kernel launch was reached.
            self.assertEqual(self.spy.launch_count, 1, dtype)


@unittest.skipUnless(_sgf is not None, _SKIP_REASON)
class TestForwardArgumentValidation(unittest.TestCase):
    """The three asserts must reject bad inputs before any kernel launch."""

    def setUp(self):
        self.spy = _KernelLaunchSpy()
        original = _sgf.fused_sigmoid_gate_fwd_kernel
        _sgf.fused_sigmoid_gate_fwd_kernel = self.spy
        self.addCleanup(
            setattr, _sgf, "fused_sigmoid_gate_fwd_kernel", original
        )

    def test_shape_mismatch_rejected_before_launch(self):
        attn_out = paddle.zeros([2, 4], dtype="float32")
        gate = paddle.zeros([2, 8], dtype="float32")
        with self.assertRaises(AssertionError):
            SigmoidGateFusionTriton.forward(_RecordingContext(), attn_out, gate)
        # Guard fired ahead of the kernel; removing the assert would launch it.
        self.assertEqual(self.spy.launch_count, 0)

    def test_dtype_mismatch_rejected_before_launch(self):
        attn_out = paddle.zeros([2, 4], dtype="float32")
        gate = paddle.zeros([2, 4], dtype="float16")
        with self.assertRaises(AssertionError):
            SigmoidGateFusionTriton.forward(_RecordingContext(), attn_out, gate)
        self.assertEqual(self.spy.launch_count, 0)

    def test_unsupported_dtype_rejected_before_launch(self):
        # Shapes and dtypes match, but int64 is outside the supported set.
        attn_out = paddle.zeros([2, 4], dtype="int64")
        gate = paddle.zeros([2, 4], dtype="int64")
        with self.assertRaises(AssertionError):
            SigmoidGateFusionTriton.forward(_RecordingContext(), attn_out, gate)
        self.assertEqual(self.spy.launch_count, 0)


@unittest.skipUnless(_sgf is not None, _SKIP_REASON)
class TestBackwardLaunchGeometry(unittest.TestCase):
    """Sizing (from ctx) and gradient buffer routing of ``backward``."""

    def setUp(self):
        self.spy = _KernelLaunchSpy()
        original = _sgf.fused_sigmoid_gate_bwd_kernel
        _sgf.fused_sigmoid_gate_bwd_kernel = self.spy
        self.addCleanup(
            setattr, _sgf, "fused_sigmoid_gate_bwd_kernel", original
        )

    def test_geometry_uses_saved_sizing_and_routes_grads(self):
        attn_out = paddle.zeros([10, 10], dtype="float32")  # 100 elements
        gate = paddle.ones([10, 10], dtype="float32")
        out_grad = paddle.ones([10, 10], dtype="float32")
        # Saved sizing deliberately differs from the tensors' element count so
        # that using ctx.n_elements (3000) vs. recomputing from the tensor (100)
        # is distinguishable.
        ctx = _BackwardContext(attn_out, gate, n_elements=3000, block_size=1024)

        attn_out_grad, gate_grad = SigmoidGateFusionTriton.backward(
            ctx, out_grad
        )

        self.assertEqual(self.spy.launch_count, 1)
        # grid = (cdiv(3000, 1024),) = (3,); the tensors' 100 elements would
        # give (1,), so this pins use of the saved n_elements.
        self.assertEqual(self.spy.grid, (3,))

        args = self.spy.args
        self.assertIs(args[0], out_grad)
        self.assertIs(args[1], attn_out)
        self.assertIs(args[2], gate)
        self.assertIs(args[3], attn_out_grad)  # d(attn_out) buffer, position 3
        self.assertIs(args[4], gate_grad)  # d(gate) buffer, position 4
        self.assertEqual(args[5], 3000)
        self.assertEqual(self.spy.kwargs["BLOCK_SIZE"], 1024)

        # Freshly allocated, distinct buffers returned in (attn_out, gate)
        # order matching the forward inputs; a swap here would misroute grads.
        self.assertIsNot(attn_out_grad, gate_grad)
        self.assertEqual(attn_out_grad.shape, [10, 10])
        self.assertEqual(gate_grad.shape, [10, 10])
        self.assertEqual(attn_out_grad.dtype, paddle.float32)
        self.assertEqual(gate_grad.dtype, paddle.float32)


if __name__ == "__main__":
    unittest.main()
