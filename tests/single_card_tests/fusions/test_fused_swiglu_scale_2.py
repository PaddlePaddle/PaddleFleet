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

Slice covered (matching the ``fused_swiglu_scale_2`` helper set):

* ``fused_swiglu_scale_forward``
* ``fused_swiglu_scale_backward``
* the shared ``_broadcast_scale`` helper they both use.

This module is a *kernel wrapper*.  On a CUDA build it dispatches to the
compiled ``paddlefleet_ops`` kernels; otherwise it runs a pure-Paddle CPU/XPU
fallback.  We therefore split the coverage into two honest halves:

* **CPU fallback numerics** -- the ``paddle.is_compiled_with_cuda`` environment
  probe (a genuine collaborator, not the code under test) is forced to
  ``False`` so the real fallback math executes.  Every expected value comes
  from an independent reference: NumPy for the forward, and Paddle autograd
  through an independently written forward for the backward.  The production
  functions are never used to build an expectation.
* **CUDA dispatch control flow** -- ``is_compiled_with_cuda`` is forced to
  ``True`` and a fake ``paddlefleet_ops`` module records the call.  We assert
  the exact kernel chosen and the exact arguments forwarded, plus return
  pass-through.  These tests do NOT assert real GPU kernel numerics (no device
  here); that is left to on-device runs.

Guards: ``paddle`` is imported under ``try/except ImportError`` and the whole
suite is ``skipUnless(HAS_PADDLE, ...)`` with the real import error as reason --
never a faked pass.
"""

import os
import sys
import types
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
    import paddle.nn.functional as F

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
# Work in float64 from the mathematical definition so a same-way bug in the
# production float32 path cannot hide behind a shared implementation.


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _silu(x):
    x = np.asarray(x, dtype=np.float64)
    return x * _sigmoid(x)


def _split_last(arr):
    half = arr.shape[-1] // 2
    return arr[..., :half], arr[..., half:]


def ref_swiglu(x_np):
    """Plain SwiGLU: SiLU(gate) * value over the last-axis halves."""
    g, v = _split_last(np.asarray(x_np, dtype=np.float64))
    return _silu(g) * v


def ref_clamped_swiglu(x_np, cv):
    """Clamped SwiGLU: gate clipped to (-inf, cv], value to [-cv, cv]."""
    g, v = _split_last(np.asarray(x_np, dtype=np.float64))
    return _silu(np.minimum(g, cv)) * np.clip(v, -cv, cv)


def ref_forward(x_np, scale_np, cv=None):
    """Full fallback forward reference: swiglu(x) * broadcast(scale)."""
    out = ref_swiglu(x_np) if cv is None else ref_clamped_swiglu(x_np, cv)
    scale = np.asarray(scale_np, dtype=np.float64)
    while scale.ndim < out.ndim:
        scale = scale[..., np.newaxis]
    return out * scale


def _fake_ops_module(**funcs):
    """Build a stand-in ``paddlefleet_ops`` module exposing given callables."""
    mod = types.ModuleType("paddlefleet_ops")
    for name, fn in funcs.items():
        setattr(mod, name, fn)
    return mod


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class _CPUFixture(unittest.TestCase):
    """Force CPU device and restore global device state on teardown."""

    def setUp(self):
        self._orig_device = paddle.get_device()
        paddle.set_device("cpu")
        self.addCleanup(lambda: paddle.set_device(self._orig_device))


class TestBroadcastScaleHelper(_CPUFixture):
    """Direct coverage of ``_broadcast_scale`` (cast + trailing unsqueeze)."""

    def test_unsqueeze_to_higher_rank_and_values(self):
        # scale rank 1 -> target rank 3 means two trailing unsqueezes:
        #   [2] -> [2, 1] -> [2, 1, 1]; values are preserved per row.
        scale = paddle.to_tensor([2.0, 3.0], dtype="float32")
        out = _broadcast_scale(scale, paddle.float32, 3)
        self.assertEqual(out.shape, [2, 1, 1])
        np.testing.assert_array_equal(
            out.numpy(), np.array([[[2.0]], [[3.0]]], dtype=np.float32)
        )

    def test_casts_to_target_dtype_without_changing_value(self):
        scale = paddle.to_tensor([1.5, -2.5], dtype="float32")
        out = _broadcast_scale(scale, paddle.float16, 2)
        self.assertEqual(out.dtype, paddle.float16)
        self.assertEqual(out.shape, [2, 1])
        np.testing.assert_allclose(
            out.astype("float32").numpy(),
            np.array([[1.5], [-2.5]], dtype=np.float32),
        )

    def test_no_unsqueeze_when_rank_already_matches(self):
        scale = paddle.to_tensor([[4.0], [5.0]], dtype="float32")
        out = _broadcast_scale(scale, paddle.float32, 2)
        self.assertEqual(out.shape, [2, 1])
        np.testing.assert_array_equal(
            out.numpy(), np.array([[4.0], [5.0]], dtype=np.float32)
        )


class TestForwardCPUFallback(_CPUFixture):
    """CPU fallback forward numerics (is_compiled_with_cuda forced False)."""

    def test_plain_swiglu_scalar_scale_literal(self):
        # gate=[1,-1], value=[2,0.5]:
        #   silu(1)  = 0.7310585786, silu(-1) = -0.2689414214
        #   swiglu   = [1.4621171572, -0.1344707107]
        #   * scale 3.0 = [4.3863514716, -0.4034121321]
        x = paddle.to_tensor([[1.0, -1.0, 2.0, 0.5]], dtype="float32")
        scale = paddle.to_tensor([3.0], dtype="float32")
        with mock.patch("paddle.is_compiled_with_cuda", return_value=False):
            out = fused_swiglu_scale_forward(x, scale)
        self.assertEqual(out.shape, [1, 2])
        expected = np.array([[4.3863514716, -0.4034121321]], dtype=np.float64)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_per_row_scale_broadcast_matches_reference(self):
        # scale rank 1 == batch: each row scaled independently ([B]->[B,1]).
        x = paddle.to_tensor(
            [[0.5, -1.0, 2.0, 3.0], [1.0, -2.0, -0.5, 4.0]], dtype="float32"
        )
        scale = paddle.to_tensor([2.0, 0.5], dtype="float32")
        with mock.patch("paddle.is_compiled_with_cuda", return_value=False):
            out = fused_swiglu_scale_forward(x, scale)
        self.assertEqual(out.shape, [2, 2])
        expected = ref_forward(x.numpy(), scale.numpy())
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_scale_broadcast_to_4d(self):
        paddle.seed(2026)
        x = paddle.randn([2, 3, 4, 8], dtype="float32")
        scale = paddle.to_tensor([1.5], dtype="float32")
        with mock.patch("paddle.is_compiled_with_cuda", return_value=False):
            out = fused_swiglu_scale_forward(x, scale)
        self.assertEqual(out.shape, [2, 3, 4, 4])
        expected = ref_forward(x.numpy(), scale.numpy())
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_clamp_branch_clips_gate_and_value(self):
        # clamp_value=1.0: gate clipped to max 1, value clipped to [-1,1].
        # gate row0=[2,-3] -> [1,-3]; value row0=[0.5,1.5] -> [0.5,1.0]; etc.
        x = paddle.to_tensor(
            [[2.0, -3.0, 0.5, 1.5], [-1.0, 4.0, 2.0, -2.0]], dtype="float32"
        )
        scale = paddle.to_tensor([2.0, 0.5], dtype="float32")
        cv = 1.0
        with mock.patch("paddle.is_compiled_with_cuda", return_value=False):
            out = fused_swiglu_scale_forward(x, scale, clamp_value=cv)
            # Negative control: same inputs, no clamp -> different result.
            out_noclamp = fused_swiglu_scale_forward(x, scale)
        self.assertEqual(out.shape, [2, 2])
        expected = ref_forward(x.numpy(), scale.numpy(), cv=cv)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)
        # The saturating fixture must make clamp visibly change the output.
        self.assertFalse(
            np.allclose(out.numpy(), out_noclamp.numpy(), atol=1e-4)
        )

    def test_nonpositive_clamp_falls_through_to_plain_swiglu(self):
        # clamp_value=0.0 fails the ``> 0`` guard -> plain swiglu path.
        x = paddle.to_tensor([[2.0, -3.0, 0.5, 1.5]], dtype="float32")
        scale = paddle.to_tensor([1.0], dtype="float32")
        with mock.patch("paddle.is_compiled_with_cuda", return_value=False):
            out_zero = fused_swiglu_scale_forward(x, scale, clamp_value=0.0)
        expected = ref_forward(x.numpy(), scale.numpy())  # unclamped reference
        np.testing.assert_allclose(
            out_zero.numpy(), expected, rtol=1e-5, atol=1e-6
        )

    def test_output_dtype_follows_input(self):
        x = paddle.randn([2, 8], dtype="float32").astype("float16")
        scale = paddle.to_tensor([2.0], dtype="float32")
        with mock.patch("paddle.is_compiled_with_cuda", return_value=False):
            out = fused_swiglu_scale_forward(x, scale)
        # scale is cast to x.dtype inside _broadcast_scale, so out is float16.
        self.assertEqual(out.dtype, paddle.float16)


def _autograd_reference_backward(x_np, scale_np, out_grad_np, cv=None):
    """Independent backward reference via Paddle autograd through a forward
    written from the SwiGLU definition (F.silu), not the production backward."""
    x = paddle.to_tensor(x_np, dtype="float32", stop_gradient=False)
    scale = paddle.to_tensor(scale_np, dtype="float32", stop_gradient=False)
    h = x.shape[-1] // 2
    gate = x[..., :h]
    val = x[..., h:]
    if cv is not None:
        gate = paddle.clip(gate, max=cv)
        val = paddle.clip(val, min=-cv, max=cv)
    swig = F.silu(gate) * val
    scale_exp = scale
    while scale_exp.ndim < swig.ndim:
        scale_exp = scale_exp.unsqueeze(-1)
    out = swig * scale_exp
    out.backward(paddle.to_tensor(out_grad_np, dtype="float32"))
    return x.grad.numpy(), scale.grad.numpy()


class TestBackwardCPUFallback(_CPUFixture):
    """CPU fallback backward gradients vs an independent autograd reference."""

    def test_plain_backward_matches_autograd(self):
        x_np = np.array(
            [[0.5, -1.0, 2.0, 3.0], [1.0, -2.0, -0.5, 4.0]], dtype=np.float32
        )
        scale_np = np.array([2.0, 0.5], dtype=np.float32)
        og_np = np.array([[0.5, -1.5], [2.0, 0.3]], dtype=np.float32)
        ref_dx, ref_ds = _autograd_reference_backward(x_np, scale_np, og_np)

        x = paddle.to_tensor(x_np, dtype="float32")
        scale = paddle.to_tensor(scale_np, dtype="float32")
        og = paddle.to_tensor(og_np, dtype="float32")
        with mock.patch("paddle.is_compiled_with_cuda", return_value=False):
            d_x, d_scale = fused_swiglu_scale_backward(x, scale, og)

        self.assertEqual(d_x.shape, [2, 4])
        self.assertEqual(d_scale.shape, [2])  # plain path: no keepdim
        np.testing.assert_allclose(d_x.numpy(), ref_dx, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(
            d_scale.numpy(), ref_ds, rtol=1e-4, atol=1e-6
        )
        # Reference gradients must be non-trivial so scaling errors are caught.
        self.assertGreater(np.abs(ref_dx).max(), 1e-2)
        self.assertGreater(np.abs(ref_ds).max(), 1e-2)

    def test_plain_backward_3d(self):
        paddle.seed(7)
        x_np = paddle.randn([2, 3, 8], dtype="float32").numpy()
        scale_np = np.array([1.5], dtype=np.float32)
        og_np = paddle.randn([2, 3, 4], dtype="float32").numpy()
        ref_dx, ref_ds = _autograd_reference_backward(x_np, scale_np, og_np)

        x = paddle.to_tensor(x_np, dtype="float32")
        scale = paddle.to_tensor(scale_np, dtype="float32")
        og = paddle.to_tensor(og_np, dtype="float32")
        with mock.patch("paddle.is_compiled_with_cuda", return_value=False):
            d_x, d_scale = fused_swiglu_scale_backward(x, scale, og)
        self.assertEqual(d_x.shape, [2, 3, 8])
        np.testing.assert_allclose(d_x.numpy(), ref_dx, rtol=1e-4, atol=1e-5)

    def test_clamp_backward_matches_autograd_and_zeroes_saturated(self):
        # gate row0=[2,-3] with cv=1 -> col0 gate saturated (grad must be 0).
        x_np = np.array(
            [[2.0, -3.0, 0.5, 1.5], [-1.0, 4.0, 2.0, -2.0]], dtype=np.float32
        )
        scale_np = np.array([2.0, 0.5], dtype=np.float32)
        og_np = np.array([[0.7, -1.2], [1.1, 0.4]], dtype=np.float32)
        cv = 1.0
        ref_dx, ref_ds = _autograd_reference_backward(
            x_np, scale_np, og_np, cv=cv
        )

        x = paddle.to_tensor(x_np, dtype="float32")
        scale = paddle.to_tensor(scale_np, dtype="float32")
        og = paddle.to_tensor(og_np, dtype="float32")
        with mock.patch("paddle.is_compiled_with_cuda", return_value=False):
            d_x, d_scale = fused_swiglu_scale_backward(
                x, scale, og, clamp_value=cv
            )

        self.assertEqual(d_x.shape, [2, 4])
        # NOTE: clamp path keeps the summed axis (keepdim=True) -> [B, 1],
        # unlike the plain path which yields [B].  Asserted here as observed
        # behavior; see report for the inconsistency note.
        self.assertEqual(d_scale.shape, [2, 1])
        np.testing.assert_allclose(d_x.numpy(), ref_dx, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(
            d_scale.numpy().reshape(-1), ref_ds, rtol=1e-4, atol=1e-6
        )
        # gate[0,0]=2 > cv -> saturated -> d_gate at [0,0] is exactly zero.
        self.assertEqual(float(d_x.numpy()[0, 0]), 0.0)
        # gate[1,1]=4 > cv -> saturated too.
        self.assertEqual(float(d_x.numpy()[1, 1]), 0.0)


class TestCUDADispatchControlFlow(_CPUFixture):
    """CPU-observable dispatch: correct kernel + exact forwarded args.

    ``is_compiled_with_cuda`` is forced True and ``paddlefleet_ops`` is faked.
    Real GPU kernel numerics are NOT asserted (no device); only which kernel
    is selected, the exact arguments forwarded, and return pass-through.
    """

    def test_forward_plain_dispatch(self):
        x = paddle.randn([4, 8], dtype="float32")
        scale = paddle.to_tensor([1.0], dtype="float32")
        marker = paddle.arange(16, dtype="float32").reshape([4, 4])
        rec = {}

        def fake_plain(a, b):
            rec.update(which="plain", args=(a, b))
            return marker

        def fake_clamp(a, b, c):
            rec.update(which="clamp", args=(a, b, c))
            return marker

        mod = _fake_ops_module(
            fused_swiglu_scale=fake_plain,
            fused_swiglu_scale_clamp=fake_clamp,
        )
        with (
            mock.patch("paddle.is_compiled_with_cuda", return_value=True),
            mock.patch.dict(sys.modules, {"paddlefleet_ops": mod}),
        ):
            out = fused_swiglu_scale_forward(x, scale)  # clamp_value None

        self.assertEqual(rec["which"], "plain")
        self.assertEqual(len(rec["args"]), 2)  # no clamp arg forwarded
        self.assertIs(rec["args"][0], x)
        self.assertIs(rec["args"][1], scale)
        np.testing.assert_array_equal(out.numpy(), marker.numpy())

    def test_forward_clamp_dispatch_forwards_clamp_value(self):
        x = paddle.randn([4, 8], dtype="float32")
        scale = paddle.to_tensor([1.0], dtype="float32")
        marker = paddle.full([4, 4], 7.0, dtype="float32")
        rec = {}

        def fake_plain(a, b):
            rec.update(which="plain", args=(a, b))
            return marker

        def fake_clamp(a, b, c):
            rec.update(which="clamp", args=(a, b, c))
            return marker

        mod = _fake_ops_module(
            fused_swiglu_scale=fake_plain,
            fused_swiglu_scale_clamp=fake_clamp,
        )
        with (
            mock.patch("paddle.is_compiled_with_cuda", return_value=True),
            mock.patch.dict(sys.modules, {"paddlefleet_ops": mod}),
        ):
            out = fused_swiglu_scale_forward(x, scale, clamp_value=2.5)

        self.assertEqual(rec["which"], "clamp")
        self.assertIs(rec["args"][0], x)
        self.assertIs(rec["args"][1], scale)
        self.assertEqual(rec["args"][2], 2.5)
        np.testing.assert_array_equal(out.numpy(), marker.numpy())

    def test_backward_plain_dispatch(self):
        x = paddle.randn([4, 8], dtype="float32")
        scale = paddle.to_tensor([1.0], dtype="float32")
        og = paddle.randn([4, 4], dtype="float32")
        m_dx = paddle.arange(32, dtype="float32").reshape([4, 8])
        m_ds = paddle.to_tensor([9.0], dtype="float32")
        rec = {}

        def fake_bwd(a, b, c):
            rec.update(which="plain", args=(a, b, c))
            return m_dx, m_ds

        def fake_clamp_bwd(a, b, c, d):
            rec.update(which="clamp", args=(a, b, c, d))
            return m_dx, m_ds

        mod = _fake_ops_module(
            fused_swiglu_scale_bwd=fake_bwd,
            fused_swiglu_scale_clamp_bwd=fake_clamp_bwd,
        )
        with (
            mock.patch("paddle.is_compiled_with_cuda", return_value=True),
            mock.patch.dict(sys.modules, {"paddlefleet_ops": mod}),
        ):
            d_x, d_scale = fused_swiglu_scale_backward(x, scale, og)

        self.assertEqual(rec["which"], "plain")
        self.assertEqual(len(rec["args"]), 3)
        self.assertIs(rec["args"][0], x)
        self.assertIs(rec["args"][1], scale)
        self.assertIs(rec["args"][2], og)
        np.testing.assert_array_equal(d_x.numpy(), m_dx.numpy())
        np.testing.assert_array_equal(d_scale.numpy(), m_ds.numpy())

    def test_backward_clamp_dispatch_casts_clamp_value_to_float(self):
        x = paddle.randn([4, 8], dtype="float32")
        scale = paddle.to_tensor([1.0], dtype="float32")
        og = paddle.randn([4, 4], dtype="float32")
        m_dx = paddle.full([4, 8], 3.0, dtype="float32")
        m_ds = paddle.to_tensor([1.0], dtype="float32")
        rec = {}

        def fake_bwd(a, b, c):
            rec.update(which="plain", args=(a, b, c))
            return m_dx, m_ds

        def fake_clamp_bwd(a, b, c, d):
            rec.update(which="clamp", args=(a, b, c, d))
            return m_dx, m_ds

        mod = _fake_ops_module(
            fused_swiglu_scale_bwd=fake_bwd,
            fused_swiglu_scale_clamp_bwd=fake_clamp_bwd,
        )
        with (
            mock.patch("paddle.is_compiled_with_cuda", return_value=True),
            mock.patch.dict(sys.modules, {"paddlefleet_ops": mod}),
        ):
            d_x, d_scale = fused_swiglu_scale_backward(
                x, scale, og, clamp_value=3
            )

        self.assertEqual(rec["which"], "clamp")
        self.assertIs(rec["args"][0], x)
        self.assertIs(rec["args"][1], scale)
        self.assertIs(rec["args"][2], og)
        # production forwards float(clamp_value): int 3 -> float 3.0.
        self.assertIsInstance(rec["args"][3], float)
        self.assertEqual(rec["args"][3], 3.0)
        np.testing.assert_array_equal(d_x.numpy(), m_dx.numpy())
        np.testing.assert_array_equal(d_scale.numpy(), m_ds.numpy())


if __name__ == "__main__":
    unittest.main()
