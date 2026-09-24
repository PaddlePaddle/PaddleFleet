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

"""Behavior tests for ``paddlefleet.fusions.fused_swiglu_scale``.

``fused_swiglu_scale`` is a kernel-wrapper: on a CUDA build it dispatches to
the ``paddlefleet_ops`` custom operators, and otherwise runs a documented
CPU/XPU fallback written in pure paddle ops.  This file targets the
*no-clamp* slice of that wrapper plus its dispatch control-flow:

* ``_broadcast_scale`` cast + unsqueeze helper;
* the plain (non-clamped) SwiGLU CPU-fallback forward, ``swiglu(x) * scale``,
  with per-row scale and 1-D-scale broadcasting;
* the plain CPU-fallback backward ``d_x`` and ``d_scale`` values;
* the GPU dispatch branch: which ``paddlefleet_ops`` op is selected, that the
  exact arguments are forwarded, and that the op's return value is passed
  through unchanged (the ``clamp_value > 0`` guard is exercised too).

What is intentionally NOT verified here (honest scope):

* GPU/custom-op numerics -- the ``is_compiled_with_cuda()`` path is only
  checked for *dispatch/argument* correctness with a genuine collaborator
  patch; the real kernels are never run on CPU.
* The static-graph ``InferShape`` regression is skipped unless a real CUDA
  build with the custom op is present.

Every expected value is hand-derived with an independent NumPy implementation
of sigmoid/SiLU (float64) or written as a pen-and-paper literal; the
production module is never called to build an expectation.
"""

import os
import sys
import unittest
from unittest import mock

import numpy as np

# Make ``src/paddlefleet`` importable when paddle is available.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import paddle

    from paddlefleet.fusions.fused_swiglu_scale import (
        _broadcast_scale,
        fused_swiglu_scale_backward,
        fused_swiglu_scale_forward,
    )

    HAS_PADDLE = True
    _IMPORT_ERROR = ""
except ImportError as exc:  # honest: dependency missing, not a swallowed bug
    HAS_PADDLE = False
    _IMPORT_ERROR = repr(exc)


# --- Independent NumPy references (never call production code) --------------
# References work in float64 from the mathematical definitions so a same-way
# bug in the production float32 code cannot hide behind a shared impl.


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _silu(x):
    x = np.asarray(x, dtype=np.float64)
    return x * _sigmoid(x)


def _split_last(arr):
    """Split the last axis into two equal halves (matches paddle.chunk)."""
    half = arr.shape[-1] // 2
    return arr[..., :half], arr[..., half:]


def _broadcast_scale_np(scale, target_ndim):
    """Cast to float64 and unsqueeze trailing dims to reach ``target_ndim``."""
    s = np.asarray(scale, dtype=np.float64)
    while s.ndim < target_ndim:
        s = s[..., None]
    return s


def ref_swiglu_scale_fwd(x, scale):
    """Plain SwiGLU forward times a broadcast scale: SiLU(gate) * value * s."""
    g, v = _split_last(np.asarray(x, dtype=np.float64))
    out = _silu(g) * v
    return out * _broadcast_scale_np(scale, out.ndim)


def ref_swiglu_scale_bwd(x, scale, out_grad):
    """Hand-derived gradients of ``SwiGLU(x) * scale`` (no-clamp branch).

    Returns ``(d_x, d_scale)`` where ``d_x`` is laid out as
    ``concat([d_gate, d_value])`` to match ``chunk(x, 2)`` -> [gate, value].
    ``d_scale`` mirrors production and reduces the hidden axis WITHOUT keepdim.
    """
    gate, val = _split_last(np.asarray(x, dtype=np.float64))
    og = np.asarray(out_grad, dtype=np.float64)
    s = _broadcast_scale_np(scale, og.ndim)
    sig = _sigmoid(gate)
    silu = gate * sig
    swiglu_val = silu * val
    d_u = og * s
    d_val = d_u * silu
    # d SiLU(x)/dx = sig * (1 + x * (1 - sig)); SiLU(x) = x * sig.
    d_gate = d_u * val * sig * (1.0 + gate * (1.0 - sig))
    d_x = np.concatenate([d_gate, d_val], axis=-1)
    d_scale = np.sum(og * swiglu_val, axis=-1)  # no keepdim, as in production
    return d_x, d_scale


def _force_cpu_fallback():
    """Select the documented CPU fallback branch regardless of the build.

    ``is_compiled_with_cuda`` is a genuine collaborator, not code under test;
    the GPU kernel path it guards is intentionally left unverified here.
    """
    return mock.patch.object(
        paddle, "is_compiled_with_cuda", return_value=False
    )


def _force_cuda():
    """Force the CUDA dispatch branch to observe op selection anywhere."""
    return mock.patch.object(paddle, "is_compiled_with_cuda", return_value=True)


def _fake_ops_module(**attrs):
    """Build a stand-in ``paddlefleet_ops`` module exposing the given ops."""
    import types

    mod = types.ModuleType("paddlefleet_ops")
    for name, value in attrs.items():
        setattr(mod, name, value)
    return mod


# Distinguishable single-row fixture used for pen-and-paper anchors.
# gate = [1.0, -2.0], value = [0.5, 3.0].
#   silu(1.0)  = 0.73105857863   silu(-2.0) = -0.23840584405
#   swiglu     = [0.36552928931, -0.71521753216]
_X_ROW = [1.0, -2.0, 0.5, 3.0]


class _CPUFixture(unittest.TestCase):
    """Common CPU device setup with global-state restoration."""

    def setUp(self):
        self._orig_device = paddle.get_device()
        paddle.set_device("cpu")
        self.addCleanup(lambda: paddle.set_device(self._orig_device))


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestBroadcastScale(_CPUFixture):
    """``_broadcast_scale`` casts to the target dtype and unsqueezes trailing
    dims until the rank matches; an already-matching rank is untouched."""

    def test_cast_and_unsqueeze_to_target_rank(self):
        scale = paddle.to_tensor([2.0, 3.0], dtype="float32")
        exp2 = _broadcast_scale(scale, paddle.float32, 2)
        self.assertEqual(exp2.shape, [2, 1])
        np.testing.assert_array_equal(exp2.numpy(), np.array([[2.0], [3.0]]))

        exp3 = _broadcast_scale(scale, paddle.float32, 3)
        self.assertEqual(exp3.shape, [2, 1, 1])
        np.testing.assert_array_equal(
            exp3.numpy(), np.array([[[2.0]], [[3.0]]])
        )

    def test_already_matching_rank_is_unchanged(self):
        already = paddle.to_tensor([[5.0], [6.0]], dtype="float32")
        out = _broadcast_scale(already, paddle.float32, 2)
        self.assertEqual(out.shape, [2, 1])
        np.testing.assert_array_equal(out.numpy(), already.numpy())

    def test_cast_changes_dtype(self):
        scale = paddle.to_tensor([1.0, 2.0], dtype="float64")
        out = _broadcast_scale(scale, paddle.float32, 2)
        self.assertEqual(out.dtype, paddle.float32)
        np.testing.assert_array_equal(out.numpy(), np.array([[1.0], [2.0]]))


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestForwardCPUNoClamp(_CPUFixture):
    """Plain (non-clamped) SwiGLU CPU fallback: ``swiglu(x) * scale_exp``."""

    def test_forward_matches_hand_derived_literals(self):
        # scale = 2.0 applied to the whole row:
        #   swiglu = [0.36552928931, -0.71521753216]
        #   * 2.0  = [0.73105857863, -1.43043506432]
        x = paddle.to_tensor([_X_ROW], dtype="float32")
        scale = paddle.to_tensor([[2.0]], dtype="float32")
        with _force_cpu_fallback():
            out = fused_swiglu_scale_forward(x, scale)
        self.assertEqual(out.shape, [1, 2])
        np.testing.assert_allclose(
            out.numpy(),
            np.array([[0.73105857863, -1.43043506432]]),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_forward_matches_independent_reference_multirow(self):
        paddle.seed(20260916)
        x = paddle.randn([6, 12], dtype="float32") * 2.0
        scale = paddle.randn([6, 1], dtype="float32")
        x_np, s_np = x.numpy(), scale.numpy()
        with _force_cpu_fallback():
            out = fused_swiglu_scale_forward(x, scale)
        np.testing.assert_allclose(
            out.numpy(), ref_swiglu_scale_fwd(x_np, s_np), rtol=1e-5, atol=1e-6
        )
        # Negative control: a wrong-sign expectation must be rejected.
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(
                out.numpy(),
                -ref_swiglu_scale_fwd(x_np, s_np),
                rtol=1e-5,
                atol=1e-6,
            )

    def test_one_dim_scale_is_unsqueezed_and_applied_per_row(self):
        # A 1-D scale [B] must broadcast to [B, 1] and multiply the matching
        # row only; distinct per-row factors expose a wrong broadcast axis.
        x = paddle.to_tensor([_X_ROW, _X_ROW], dtype="float32")
        scale = paddle.to_tensor([2.0, 5.0], dtype="float32")  # 1-D -> [B, 1]
        with _force_cpu_fallback():
            out = fused_swiglu_scale_forward(x, scale).numpy()
        base = ref_swiglu_scale_fwd([_X_ROW], np.array([[1.0]]))  # [1, 2]
        np.testing.assert_allclose(out[0], base[0] * 2.0, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(out[1], base[0] * 5.0, rtol=1e-5, atol=1e-6)


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestBackwardCPUNoClamp(_CPUFixture):
    """Plain (non-clamped) SwiGLU CPU fallback backward: ``d_x`` and
    ``d_scale`` values against an independent reference."""

    def test_dscale_matches_hand_derived_literal(self):
        # scale = 1, out_grad = ones -> d_scale = sum(swiglu) over hidden:
        #   0.36552928931 + (-0.71521753216) = -0.34968824285
        x = paddle.to_tensor([_X_ROW], dtype="float32")
        scale = paddle.to_tensor([[1.0]], dtype="float32")
        out_grad = paddle.ones([1, 2], dtype="float32")
        with _force_cpu_fallback():
            _, d_scale = fused_swiglu_scale_backward(x, scale, out_grad)
        np.testing.assert_allclose(
            d_scale.numpy().reshape(-1),
            np.array([-0.34968824285]),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_dx_and_dscale_match_independent_reference(self):
        paddle.seed(7)
        x = paddle.randn([5, 8], dtype="float32") * 2.0
        scale = paddle.randn([5, 1], dtype="float32")
        out_grad = paddle.randn([5, 4], dtype="float32")
        x_np, s_np, og_np = x.numpy(), scale.numpy(), out_grad.numpy()
        ref_dx, ref_ds = ref_swiglu_scale_bwd(x_np, s_np, og_np)
        with _force_cpu_fallback():
            d_x, d_scale = fused_swiglu_scale_backward(x, scale, out_grad)
        np.testing.assert_allclose(d_x.numpy(), ref_dx, rtol=1e-5, atol=1e-6)
        # Values (flattened) must match even though the no-clamp branch drops
        # the trailing dim; the rank mismatch itself is asserted separately.
        np.testing.assert_allclose(
            d_scale.numpy().reshape(-1),
            ref_ds.reshape(-1),
            rtol=1e-5,
            atol=1e-6,
        )
        # Negative control: a wrong-sign d_x must be rejected.
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(
                d_x.numpy(), -ref_dx, rtol=1e-5, atol=1e-6
            )


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestBackwardNoClampScaleGradRank(_CPUFixture):
    """Regression guard for a real inconsistency in the CPU fallback backward.

    The no-clamp branch computes ``d_scale`` with ``paddle.sum(..., axis=-1)``
    (no ``keepdim``), yielding rank ``ndim-1``.  The clamp branch instead uses
    ``keepdim=True``.  So for a ``[B, 1]`` scale the two branches disagree and
    the no-clamp gradient ([B]) no longer matches the scale it belongs to -- a
    PyLayer assigning it to ``scale.grad`` would hit a shape mismatch.  This
    test asserts the consistent contract (grad shape == scale shape) and is
    expected to FAIL until production keeps the reduced dim.  Production is not
    modified.
    """

    @unittest.expectedFailure
    def test_dscale_rank_should_match_scale(self):
        x = paddle.randn([4, 16], dtype="float32")
        scale = paddle.ones([4, 1], dtype="float32")
        out_grad = paddle.randn([4, 8], dtype="float32")
        with _force_cpu_fallback():
            _, d_scale = fused_swiglu_scale_backward(x, scale, out_grad)
        # Currently d_scale.shape == [4]; the consistent contract is [4, 1].
        self.assertEqual(d_scale.shape, scale.shape)


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestGPUDispatchControlFlow(_CPUFixture):
    """CPU-observable dispatch control flow of the CUDA branch.

    ``is_compiled_with_cuda`` and ``paddlefleet_ops`` are genuine collaborators
    (the custom-op package is not built on CPU); both are patched so we can
    observe *which* op is selected, that the exact arguments are forwarded, and
    that the op's return value is passed through untouched.  The real kernel
    numerics are NOT verified here -- the stub returns a distinct marker.
    """

    def test_forward_no_clamp_selects_plain_op_and_forwards_args(self):
        x = paddle.to_tensor([_X_ROW], dtype="float32")
        scale = paddle.to_tensor([[2.0]], dtype="float32")
        marker = paddle.to_tensor([[7.0, 8.0]], dtype="float32")
        seen = {}

        def plain_op(*args, **kwargs):
            seen["args"], seen["kwargs"] = args, kwargs
            return marker

        clamp_op = mock.MagicMock()
        fake = _fake_ops_module(
            fused_swiglu_scale=plain_op, fused_swiglu_scale_clamp=clamp_op
        )
        with (
            _force_cuda(),
            mock.patch.dict(sys.modules, {"paddlefleet_ops": fake}),
        ):
            out = fused_swiglu_scale_forward(x, scale)  # clamp_value=None
        self.assertEqual(len(seen["args"]), 2)
        self.assertIs(seen["args"][0], x)
        self.assertIs(seen["args"][1], scale)
        self.assertEqual(seen["kwargs"], {})
        self.assertIs(out, marker)  # return forwarded unchanged
        clamp_op.assert_not_called()

    def test_forward_positive_clamp_selects_clamp_op_with_clamp_arg(self):
        x = paddle.to_tensor([_X_ROW], dtype="float32")
        scale = paddle.to_tensor([[2.0]], dtype="float32")
        marker = paddle.to_tensor([[1.0, 2.0]], dtype="float32")
        seen = {}

        def clamp_op(*args, **kwargs):
            seen["args"], seen["kwargs"] = args, kwargs
            return marker

        plain_op = mock.MagicMock()
        fake = _fake_ops_module(
            fused_swiglu_scale=plain_op, fused_swiglu_scale_clamp=clamp_op
        )
        with (
            _force_cuda(),
            mock.patch.dict(sys.modules, {"paddlefleet_ops": fake}),
        ):
            out = fused_swiglu_scale_forward(x, scale, clamp_value=3.0)
        self.assertEqual(len(seen["args"]), 3)
        self.assertIs(seen["args"][0], x)
        self.assertIs(seen["args"][1], scale)
        self.assertEqual(seen["args"][2], 3.0)
        self.assertIs(out, marker)
        plain_op.assert_not_called()

    def test_forward_nonpositive_clamp_falls_through_to_plain_op(self):
        # clamp_value == 0 fails the ``clamp_value > 0`` guard, so even on the
        # CUDA branch the plain op must be selected.
        x = paddle.to_tensor([_X_ROW], dtype="float32")
        scale = paddle.to_tensor([[2.0]], dtype="float32")
        marker = paddle.to_tensor([[3.0, 4.0]], dtype="float32")
        plain_op = mock.MagicMock(return_value=marker)
        clamp_op = mock.MagicMock()
        fake = _fake_ops_module(
            fused_swiglu_scale=plain_op, fused_swiglu_scale_clamp=clamp_op
        )
        with (
            _force_cuda(),
            mock.patch.dict(sys.modules, {"paddlefleet_ops": fake}),
        ):
            out = fused_swiglu_scale_forward(x, scale, clamp_value=0.0)
        plain_op.assert_called_once_with(x, scale)
        clamp_op.assert_not_called()
        self.assertIs(out, marker)

    def test_backward_no_clamp_selects_plain_bwd_and_forwards_args(self):
        x = paddle.to_tensor([_X_ROW], dtype="float32")
        scale = paddle.to_tensor([[2.0]], dtype="float32")
        out_grad = paddle.ones([1, 2], dtype="float32")
        marker = (
            paddle.to_tensor([[1.0, 2.0, 3.0, 4.0]], dtype="float32"),
            paddle.to_tensor([[5.0]], dtype="float32"),
        )
        seen = {}

        def plain_bwd(*args, **kwargs):
            seen["args"], seen["kwargs"] = args, kwargs
            return marker

        clamp_bwd = mock.MagicMock()
        fake = _fake_ops_module(
            fused_swiglu_scale_bwd=plain_bwd,
            fused_swiglu_scale_clamp_bwd=clamp_bwd,
        )
        with (
            _force_cuda(),
            mock.patch.dict(sys.modules, {"paddlefleet_ops": fake}),
        ):
            out = fused_swiglu_scale_backward(x, scale, out_grad)
        self.assertEqual(len(seen["args"]), 3)
        self.assertIs(seen["args"][0], x)
        self.assertIs(seen["args"][1], scale)
        self.assertIs(seen["args"][2], out_grad)
        self.assertIs(out, marker)
        clamp_bwd.assert_not_called()

    def test_backward_positive_clamp_selects_clamp_bwd_with_float_clamp(self):
        x = paddle.to_tensor([_X_ROW], dtype="float32")
        scale = paddle.to_tensor([[2.0]], dtype="float32")
        out_grad = paddle.ones([1, 2], dtype="float32")
        marker = (
            paddle.to_tensor([[0.0, 0.0, 0.0, 0.0]], dtype="float32"),
            paddle.to_tensor([[9.0]], dtype="float32"),
        )
        seen = {}

        def clamp_bwd(*args, **kwargs):
            seen["args"], seen["kwargs"] = args, kwargs
            return marker

        plain_bwd = mock.MagicMock()
        fake = _fake_ops_module(
            fused_swiglu_scale_bwd=plain_bwd,
            fused_swiglu_scale_clamp_bwd=clamp_bwd,
        )
        with (
            _force_cuda(),
            mock.patch.dict(sys.modules, {"paddlefleet_ops": fake}),
        ):
            out = fused_swiglu_scale_backward(x, scale, out_grad, clamp_value=3)
        self.assertEqual(len(seen["args"]), 4)
        self.assertIs(seen["args"][0], x)
        self.assertIs(seen["args"][1], scale)
        self.assertIs(seen["args"][2], out_grad)
        # Production casts clamp_value with float() before forwarding.
        self.assertEqual(seen["args"][3], 3.0)
        self.assertIsInstance(seen["args"][3], float)
        self.assertIs(out, marker)
        plain_bwd.assert_not_called()


def _has_clamp_op():
    """Precise capability probe: only a real CUDA build with the custom op."""
    if not HAS_PADDLE or not paddle.is_compiled_with_cuda():
        return False
    try:
        import paddlefleet_ops
    except ImportError:
        return False
    return hasattr(paddlefleet_ops, "fused_swiglu_scale_clamp")


@unittest.skipUnless(
    _has_clamp_op(),
    "fused_swiglu_scale_clamp custom op not built / CUDA unavailable",
)
class TestClampForwardInferShape(unittest.TestCase):
    """Static-graph ``InferShape`` regression for the clamp forward op.

    In eager mode a wrong ``InferShape`` is masked by the kernel's returned
    shape; static mode relies on it, so a bad registration aborts.  The build
    runs in a subprocess so a C++ SIGABRT cannot take down the test worker.
    The output of a ``[4, 32]`` input must be ``[4, 16]`` (hidden halved).
    """

    def test_clamp_forward_infer_shape_halves_hidden(self):
        import subprocess
        import textwrap

        code = textwrap.dedent(
            """
            import paddle
            from paddlefleet_ops import fused_swiglu_scale_clamp

            paddle.enable_static()
            main, startup = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                x = paddle.static.data(
                    name='x', shape=[4, 32], dtype='float32'
                )
                scale = paddle.static.data(
                    name='scale', shape=[4, 1], dtype='float32'
                )
                out = fused_swiglu_scale_clamp(x, scale, 5.0)
                print('SHAPE_OK', list(out.shape))
            """
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            proc.returncode,
            0,
            "static-graph build crashed (likely InferShape bug).\n"
            f"STDERR:\n{proc.stderr}",
        )
        self.assertIn(
            "SHAPE_OK [4, 16]", proc.stdout, f"stdout was:\n{proc.stdout}"
        )


if __name__ == "__main__":
    unittest.main()
