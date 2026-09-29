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

"""Behavior unit tests for ``paddlefleet.fusions.fused_swiglu_scale`` (base).

Scope (base): the top-level kernel-wrapper entry points --
``_broadcast_scale`` (scale-factor plumbing: dtype cast + rank matching),
and the ``fused_swiglu_scale_forward`` / ``fused_swiglu_scale_backward``
control flow: which branch/kernel is dispatched under the
``is_compiled_with_cuda`` and ``clamp_value`` guards, the exact arguments
plumbed to each kernel, and the arity/shape/values the backward returns.
Deep clamp CPU numerics are left to the sibling tests per the task split.

Two honest device stances are used:
  * CPU fallback branch -- ``paddle.is_compiled_with_cuda`` is patched to
    ``False`` (a genuine environment probe, not the code under test) so the
    real pure-paddle fallback math executes on CPU. Its numerics are checked
    against independent references (a numpy re-derivation for forward, and
    paddle autograd of an independently-built forward graph as the oracle for
    the hand-written backward). No GPU kernel is involved.
  * CUDA dispatch branch -- ``is_compiled_with_cuda`` is patched to ``True``
    and a fake ``paddlefleet_ops`` module is injected so the dispatch/arg
    plumbing is observable on any build. The fake kernels return distinguishable
    markers; the tests assert which kernel is chosen, the exact args, and that
    the wrapper returns the kernel output verbatim. GPU kernel *numerics* are
    deliberately NOT claimed here -- only the wrapper's dispatch contract.

Note on the scope banner: the (untrusted) banner spoke of an autograd Function
with ctx save/restore. This production module contains no PyLayer -- it exposes
two module-level functions plus ``_broadcast_scale``. The "tuple arity" contract
maps to ``fused_swiglu_scale_backward`` returning the 2-tuple ``(d_x, d_scale)``,
which is asserted directly.

Every expected value is hand-derived from the math; collaborators that are
mocked (``is_compiled_with_cuda``, the external ``paddlefleet_ops`` kernels) are
genuine non-tested collaborators, never the functions under test.
"""

import unittest
from unittest.mock import patch

import numpy as np

try:
    import paddle

    # Import at module load, BEFORE any ``setUp`` pins the device to CPU. On a
    # CUDA-compiled build the first ``paddlefleet`` import pulls in
    # ``paddlefleet_ops``, whose package init calls ``get_device_capability()``
    # against the current device -- which must be a GPU. The process default
    # device is the GPU there, so importing first keeps that query valid; a
    # deferred import in ``setUp`` (after ``set_device("cpu")``) would probe a
    # CPU place and raise ``ValueError``.
    from paddlefleet.fusions.fused_swiglu_scale import (
        _broadcast_scale,
        fused_swiglu_scale_backward,
        fused_swiglu_scale_forward,
    )

    _PADDLE_IMPORT_ERROR = None
except ImportError as exc:  # honest: paddle genuinely absent, not swallowed
    paddle = None
    _broadcast_scale = None
    fused_swiglu_scale_forward = None
    fused_swiglu_scale_backward = None
    _PADDLE_IMPORT_ERROR = exc

_SKIP_REASON = (
    f"paddle is not installed in this environment: {_PADDLE_IMPORT_ERROR}"
)


def _np_silu(a):
    """Independent numpy SiLU: a * sigmoid(a)."""
    return a * (1.0 / (1.0 + np.exp(-a)))


def _np_swiglu_scale(x, scale, clamp_value=None):
    """Independent numpy reference for the CPU-fallback forward.

    Mirrors the documented math (swiglu = silu(gate) * value, optionally with
    the gate clamped above at clamp_value and value clamped to +/-clamp_value)
    then multiplies by ``scale`` broadcast over the trailing hidden axis. This
    is a hand re-derivation, not a copy of the production expression tree.
    """
    hidden = x.shape[-1] // 2
    gate = x[..., :hidden].astype(np.float64)
    val = x[..., hidden:].astype(np.float64)
    if clamp_value is not None and clamp_value > 0:
        gate = np.minimum(gate, clamp_value)
        val = np.clip(val, -clamp_value, clamp_value)
    out = _np_silu(gate) * val
    scale = np.asarray(scale, dtype=np.float64)
    while scale.ndim < out.ndim:
        scale = scale[..., None]
    return out * scale


# Fixed, distinguishable inputs shared across the numeric tests. Shape [2, 6]
# so hidden = 3; per-row/per-column values differ so a dropped scale, swapped
# gate/value halves, or wrong concat order would change the result.
_X = np.array(
    [[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], [0.5, -0.5, 1.5, -1.5, 2.5, 0.25]],
    dtype=np.float32,
)
_SCALE = np.array([2.0, 0.5], dtype=np.float32)  # per-row, != 1, distinct
_OUT_GRAD = np.array([[0.3, -0.7, 1.1], [-1.3, 0.9, -0.2]], dtype=np.float32)


@unittest.skipUnless(paddle is not None, _SKIP_REASON)
class TestBroadcastScale(unittest.TestCase):
    """Scale-factor plumbing: ``_broadcast_scale`` cast + rank matching."""

    def setUp(self):
        self._orig_device = paddle.get_device()
        self.addCleanup(paddle.set_device, self._orig_device)
        paddle.set_device("cpu")
        self._broadcast_scale = _broadcast_scale

    def test_1d_scale_unsqueezed_once_values_preserved(self):
        """[2] scale, target_ndim 2 -> [2, 1] with values unchanged.

        Hand-derived: ndim 1 < 2 so exactly one trailing unsqueeze; cast to
        float32 is the identity here, so the two values must survive verbatim.
        """
        scale = paddle.to_tensor([2.0, 0.5], dtype=paddle.float32)
        out = self._broadcast_scale(scale, paddle.float32, 2)
        self.assertEqual(out.shape, [2, 1])
        np.testing.assert_array_equal(out.numpy(), [[2.0], [0.5]])

    def test_0d_scale_unsqueezed_to_target_rank(self):
        """scalar scale, target_ndim 2 -> [1, 1]; ndim 0 needs two unsqueezes."""
        scale = paddle.to_tensor(3.0, dtype=paddle.float32)
        out = self._broadcast_scale(scale, paddle.float32, 2)
        self.assertEqual(out.shape, [1, 1])
        np.testing.assert_array_equal(out.numpy(), [[3.0]])

    def test_dtype_is_cast_to_target(self):
        """scale is cast to the requested target dtype, value preserved."""
        scale = paddle.to_tensor([2.0], dtype=paddle.float32)
        out = self._broadcast_scale(scale, paddle.float16, 2)
        self.assertEqual(out.dtype, paddle.float16)
        np.testing.assert_array_equal(
            out.astype(paddle.float32).numpy(), [[2.0]]
        )

    def test_matching_rank_not_unsqueezed(self):
        """scale already at target_ndim -> returned untouched (no extra dim)."""
        scale = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=paddle.float32
        )
        out = self._broadcast_scale(scale, paddle.float32, 2)
        self.assertEqual(out.shape, [2, 3])
        np.testing.assert_array_equal(
            out.numpy(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        )


@unittest.skipUnless(paddle is not None, _SKIP_REASON)
class TestForwardCpuFallback(unittest.TestCase):
    """CPU-fallback forward: scale application + clamp-guard control flow.

    The CUDA branch is disabled by patching the (genuine) probe
    ``paddle.is_compiled_with_cuda`` to False, so the real pure-paddle fallback
    runs on CPU and is compared to the independent numpy reference.
    """

    def setUp(self):
        self._orig_device = paddle.get_device()
        self.addCleanup(paddle.set_device, self._orig_device)
        paddle.set_device("cpu")
        p = patch("paddle.is_compiled_with_cuda", return_value=False)
        p.start()
        self.addCleanup(p.stop)
        self._forward = fused_swiglu_scale_forward

    def test_forward_applies_broadcast_scale(self):
        """out == swiglu(x) * scale, per-row scale actually consumed.

        Independent numpy reference; scale differs per row and != 1 so an
        ignored or mis-broadcast scale would fail.
        """
        x = paddle.to_tensor(_X)
        scale = paddle.to_tensor(_SCALE)
        out = self._forward(x, scale)
        self.assertEqual(out.shape, [2, 3])
        expected = _np_swiglu_scale(_X, _SCALE)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_clamp_positive_takes_clamp_path(self):
        """clamp_value > 0 clamps gate/value before swiglu (distinct result).

        Hand-derived clamped reference must match, and must differ from the
        unclamped reference on this fixture (row 0 gate=2,3 and value=3 are
        clamped at 1.0), proving the clamp branch is really taken.
        """
        x = paddle.to_tensor(_X)
        scale = paddle.to_tensor(_SCALE)
        out = self._forward(x, scale, clamp_value=1.0)
        clamped_ref = _np_swiglu_scale(_X, _SCALE, clamp_value=1.0)
        plain_ref = _np_swiglu_scale(_X, _SCALE)
        np.testing.assert_allclose(
            out.numpy(), clamped_ref, rtol=1e-5, atol=1e-6
        )
        self.assertFalse(np.allclose(clamped_ref, plain_ref))

    def test_clamp_zero_and_none_take_plain_path(self):
        """Guard is ``clamp_value is not None and clamp_value > 0``.

        clamp_value in {None, 0.0, -1.0} must all yield the unclamped swiglu
        result, i.e. 0.0 is excluded by the strict ``> 0`` (not truthiness).
        """
        x = paddle.to_tensor(_X)
        scale = paddle.to_tensor(_SCALE)
        plain_ref = _np_swiglu_scale(_X, _SCALE)
        for cv in (None, 0.0, -1.0):
            with self.subTest(clamp_value=cv):
                out = self._forward(x, scale, clamp_value=cv)
                np.testing.assert_allclose(
                    out.numpy(), plain_ref, rtol=1e-5, atol=1e-6
                )


@unittest.skipUnless(paddle is not None, _SKIP_REASON)
class TestBackwardCpuFallback(unittest.TestCase):
    """CPU-fallback backward: tuple arity + gradients vs an autograd oracle.

    The hand-written manual backward is checked against paddle autograd of an
    *independently* built forward graph (inline paddle ops, not the production
    forward), which is the natural independent oracle for a hand-coded gradient.
    """

    def setUp(self):
        self._orig_device = paddle.get_device()
        self.addCleanup(paddle.set_device, self._orig_device)
        paddle.set_device("cpu")
        p = patch("paddle.is_compiled_with_cuda", return_value=False)
        p.start()
        self.addCleanup(p.stop)
        self._backward = fused_swiglu_scale_backward

    def _autograd_reference(self):
        """d_x, d_scale from autograd of an inline swiglu*scale forward."""
        xr = paddle.to_tensor(_X)
        xr.stop_gradient = False
        sr = paddle.to_tensor(_SCALE)
        sr.stop_gradient = False
        gate = xr[:, :3]
        val = xr[:, 3:]
        u = paddle.nn.functional.silu(gate) * val
        out = u * sr.unsqueeze(-1)
        out.backward(paddle.to_tensor(_OUT_GRAD))
        return xr.grad.numpy(), sr.grad.numpy()

    def test_backward_returns_dx_dscale_matching_autograd(self):
        """Returns exactly (d_x, d_scale); both match the autograd oracle.

        Tuple arity is 2. d_x has x's shape [2, 6]; d_scale is reduced over the
        trailing hidden axis to [2] (per-row scalar scale), matching autograd's
        grad shape for the [2] scale.
        """
        x = paddle.to_tensor(_X)
        scale = paddle.to_tensor(_SCALE)
        out_grad = paddle.to_tensor(_OUT_GRAD)
        result = self._backward(x, scale, out_grad)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        d_x, d_scale = result
        self.assertEqual(d_x.shape, [2, 6])
        self.assertEqual(d_scale.shape, [2])
        d_x_ref, d_scale_ref = self._autograd_reference()
        np.testing.assert_allclose(d_x.numpy(), d_x_ref, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            d_scale.numpy(), d_scale_ref, rtol=1e-5, atol=1e-6
        )


def _install_cuda_and_fake_ops(tc, **funcs):
    """Force the CUDA branch and inject a fake ``paddlefleet_ops`` module.

    ``paddlefleet_ops`` is an external compiled-kernel collaborator, never the
    code under test. Replacing it lets us observe the wrapper's dispatch/arg
    plumbing deterministically on any build. Auto-restored via addCleanup.
    """
    import sys
    import types

    fake = types.ModuleType("paddlefleet_ops")
    for name, fn in funcs.items():
        setattr(fake, name, fn)
    p_cuda = patch("paddle.is_compiled_with_cuda", return_value=True)
    p_mod = patch.dict(sys.modules, {"paddlefleet_ops": fake})
    p_cuda.start()
    tc.addCleanup(p_cuda.stop)
    p_mod.start()
    tc.addCleanup(p_mod.stop)


@unittest.skipUnless(paddle is not None, _SKIP_REASON)
class TestForwardCudaDispatch(unittest.TestCase):
    """CUDA-branch forward dispatch: kernel selection + arg plumbing only.

    GPU kernel numerics are NOT verified here; the fake kernels return
    distinguishable markers so we can assert *which* kernel is called, with
    *which* args, and that the wrapper returns the kernel output verbatim.
    """

    def setUp(self):
        self._orig_device = paddle.get_device()
        self.addCleanup(paddle.set_device, self._orig_device)
        paddle.set_device("cpu")
        self._forward = fused_swiglu_scale_forward
        self.calls = {}

    def _make_ops(self):
        def plain(x, scale):
            self.calls["plain"] = (x, scale)
            return paddle.to_tensor([[42.0]])

        def clamp(x, scale, clamp_value):
            self.calls["clamp"] = (x, scale, clamp_value)
            return paddle.to_tensor([[99.0]])

        return {"fused_swiglu_scale": plain, "fused_swiglu_scale_clamp": clamp}

    def test_no_clamp_dispatches_plain_kernel(self):
        """clamp_value None -> fused_swiglu_scale(x, scale); output verbatim."""
        _install_cuda_and_fake_ops(self, **self._make_ops())
        x = paddle.to_tensor(_X)
        scale = paddle.to_tensor(_SCALE)
        out = self._forward(x, scale)
        self.assertNotIn("clamp", self.calls)
        self.assertIs(self.calls["plain"][0], x)
        self.assertIs(self.calls["plain"][1], scale)
        np.testing.assert_array_equal(out.numpy(), [[42.0]])

    def test_positive_clamp_dispatches_clamp_kernel(self):
        """clamp_value > 0 -> fused_swiglu_scale_clamp(x, scale, clamp_value)."""
        _install_cuda_and_fake_ops(self, **self._make_ops())
        x = paddle.to_tensor(_X)
        scale = paddle.to_tensor(_SCALE)
        out = self._forward(x, scale, clamp_value=1.5)
        self.assertNotIn("plain", self.calls)
        self.assertIs(self.calls["clamp"][0], x)
        self.assertIs(self.calls["clamp"][1], scale)
        self.assertEqual(self.calls["clamp"][2], 1.5)
        np.testing.assert_array_equal(out.numpy(), [[99.0]])

    def test_zero_clamp_dispatches_plain_kernel(self):
        """clamp_value 0.0 fails the strict ``> 0`` guard -> plain kernel."""
        _install_cuda_and_fake_ops(self, **self._make_ops())
        x = paddle.to_tensor(_X)
        scale = paddle.to_tensor(_SCALE)
        out = self._forward(x, scale, clamp_value=0.0)
        self.assertIn("plain", self.calls)
        self.assertNotIn("clamp", self.calls)
        np.testing.assert_array_equal(out.numpy(), [[42.0]])


@unittest.skipUnless(paddle is not None, _SKIP_REASON)
class TestBackwardCudaDispatch(unittest.TestCase):
    """CUDA-branch backward dispatch: kernel selection, args, tuple arity.

    GPU kernel numerics are NOT verified here. The fake kernels return a
    distinguishable 2-tuple so we can assert the wrapper selects the right
    kernel, forwards the exact args, and returns the kernel's tuple verbatim.
    """

    def setUp(self):
        self._orig_device = paddle.get_device()
        self.addCleanup(paddle.set_device, self._orig_device)
        paddle.set_device("cpu")
        self._backward = fused_swiglu_scale_backward
        self.calls = {}
        self._plain_ret = (paddle.to_tensor([[1.0]]), paddle.to_tensor([2.0]))
        self._clamp_ret = (paddle.to_tensor([[3.0]]), paddle.to_tensor([4.0]))

    def _make_ops(self):
        def plain_bwd(x, scale, out_grad):
            self.calls["plain"] = (x, scale, out_grad)
            return self._plain_ret

        def clamp_bwd(x, scale, out_grad, clamp_value):
            self.calls["clamp"] = (x, scale, out_grad, clamp_value)
            return self._clamp_ret

        return {
            "fused_swiglu_scale_bwd": plain_bwd,
            "fused_swiglu_scale_clamp_bwd": clamp_bwd,
        }

    def test_no_clamp_dispatches_plain_bwd_returns_tuple(self):
        """clamp None -> fused_swiglu_scale_bwd(x, scale, out_grad) verbatim."""
        _install_cuda_and_fake_ops(self, **self._make_ops())
        x = paddle.to_tensor(_X)
        scale = paddle.to_tensor(_SCALE)
        out_grad = paddle.to_tensor(_OUT_GRAD)
        result = self._backward(x, scale, out_grad)
        self.assertNotIn("clamp", self.calls)
        self.assertIs(self.calls["plain"][0], x)
        self.assertIs(self.calls["plain"][1], scale)
        self.assertIs(self.calls["plain"][2], out_grad)
        self.assertIs(result, self._plain_ret)
        self.assertEqual(len(result), 2)

    def test_positive_clamp_dispatches_clamp_bwd_with_float(self):
        """clamp > 0 -> fused_swiglu_scale_clamp_bwd(..., float(clamp_value)).

        clamp_value is passed as an int 2 but the wrapper coerces it to the
        python float 2.0 for the kernel (per ``float(clamp_value)``).
        """
        _install_cuda_and_fake_ops(self, **self._make_ops())
        x = paddle.to_tensor(_X)
        scale = paddle.to_tensor(_SCALE)
        out_grad = paddle.to_tensor(_OUT_GRAD)
        result = self._backward(x, scale, out_grad, clamp_value=2)
        self.assertNotIn("plain", self.calls)
        self.assertIs(self.calls["clamp"][0], x)
        self.assertIs(self.calls["clamp"][1], scale)
        self.assertIs(self.calls["clamp"][2], out_grad)
        self.assertIsInstance(self.calls["clamp"][3], float)
        self.assertEqual(self.calls["clamp"][3], 2.0)
        self.assertIs(result, self._clamp_ret)

    def test_zero_clamp_dispatches_plain_bwd(self):
        """clamp_value 0 fails the strict ``> 0`` guard -> plain bwd kernel."""
        _install_cuda_and_fake_ops(self, **self._make_ops())
        x = paddle.to_tensor(_X)
        scale = paddle.to_tensor(_SCALE)
        out_grad = paddle.to_tensor(_OUT_GRAD)
        result = self._backward(x, scale, out_grad, clamp_value=0)
        self.assertIn("plain", self.calls)
        self.assertNotIn("clamp", self.calls)
        self.assertIs(result, self._plain_ret)


if __name__ == "__main__":
    unittest.main()
