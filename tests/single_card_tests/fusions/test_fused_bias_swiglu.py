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

"""Behavior tests for the top-level SwiGLU entry in
``paddlefleet.fusions.fused_bias_swiglu``.

Scope (kept disjoint from the concurrent sibling suites -- ``_2``/``_3``/``_4``
and the ``use_accuracy_compatible`` sibling): the *control flow* of the base
SwiGLU entry only --

* ``bias_swiglu_impl`` dispatch -- ``bias is not None`` selects
  ``BiasSwiGLUFunction`` (bias is added before the activation); ``bias is None``
  selects ``SwiGLUFunction``; and the original rank (2D vs 3D) decides whether
  the flat output is viewed back to ``[B, S, H/2]``.
* ``SwiGLUFunction`` / ``BiasSwiGLUFunction`` ``forward``/``backward`` -- which
  tensors ``ctx`` saves and restores, the ``clamp_value`` branch selection
  (``clamp_value is not None and clamp_value > 0``), and the arity of the
  gradient tuple each ``backward`` returns (one grad for the no-bias variant,
  ``(input, bias)`` for the bias variant, where the bias grad equals the input
  grad because ``backward`` returns ``tmp, tmp``).
* ``clamp_value`` plumbing -- a positive value routes to the clamped kernel and
  actually clips the gate/value halves; ``0.0`` and ``None`` fall through the
  ``> 0`` guard to the unclamped path.

Everything runs eagerly on CPU (``jit_fuser`` is an identity decorator in this
tree, so no CINN/GPU compilation is involved), where the primitive ops
(``chunk``/``clip``/``silu``/``sigmoid``/``mul``) all have real CPU kernels.
Expected values are hand-derived independently: the forward is checked against a
closed-form NumPy SiLU reference, and every backward is checked against a
*separate* Paddle autograd graph rebuilt from primitive ops (``F.silu`` of a
clipped gate times the clipped value) -- never against the production
``*_back`` helpers themselves. The specialised fused GPU kernel numerics and the
native ``paddle._C_ops.swiglu_grad`` path used by the *unclamped* backward are
deliberately left to the GPU-path / sibling suites; this file pins only the
CPU-observable dispatch, ctx save/restore, and clamp plumbing.

``paddle`` (and ``numpy``) are imported at module load; when they -- or
``paddlefleet`` -- are missing the whole suite is skipped with an honest reason
instead of a hollow pass. The ``use_accuracy_compatible`` branch is intentionally
untouched here and belongs to its dedicated sibling.
"""

import os
import sys
import unittest

# Make the in-tree ``src/paddlefleet`` importable when the package has not been
# pip-installed (repo_root/src is 4 levels up from this file).
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import numpy as np
    import paddle
    import paddle.nn.functional as F

    import paddlefleet.fusions.fused_bias_swiglu as fbs

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet not installed on this host
    np = None
    paddle = None
    F = None
    fbs = None
    _IMPORT_ERROR = exc


_skip_reason = f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}"


# --------------------------------------------------------------------------
# Independent NumPy references (closed-form; never call the production module).
# --------------------------------------------------------------------------
def _np_sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def _np_silu(x):
    x = np.asarray(x, dtype=np.float64)
    return x * _np_sigmoid(x)


def _np_swiglu(y):
    """SwiGLU: split the last axis in half, return ``silu(gate) * value``."""
    y1, y2 = np.split(np.asarray(y, dtype=np.float64), 2, axis=-1)
    return _np_silu(y1) * y2


def _np_clamped_swiglu(y, clamp_value):
    """Clamp gate to ``(-inf, cv]`` and value to ``[-cv, cv]`` then SwiGLU."""
    y1, y2 = np.split(np.asarray(y, dtype=np.float64), 2, axis=-1)
    y1 = np.minimum(y1, clamp_value)
    y2 = np.clip(y2, -clamp_value, clamp_value)
    return _np_silu(y1) * y2


class _CpuTestCase(unittest.TestCase):
    """Base fixture pinning eager execution to CPU and restoring the device."""

    def setUp(self):
        # SwiGLU control flow is device independent; force CPU so the
        # assertions describe locally observable behaviour and never claim GPU
        # kernel numerics. The device is restored even on failure.
        self._orig_device = paddle.device.get_device()
        paddle.set_device("cpu")
        self.addCleanup(paddle.set_device, self._orig_device)

    def _clamped_reference_grads(self, tensors, clamp_value, upstream_np):
        """Grad(s) of the clamped SwiGLU built from an *independent* primitive
        Paddle autograd graph (``silu(clip(gate)) * clip(value)``).

        ``tensors`` are fresh leaves whose element-wise sum forms the activation
        input ``y`` (one leaf for the no-bias case, ``[input, bias]`` for the
        bias case). Returns ``(ref_out, [leaf.grad, ...])``. This never touches
        the production ``clamped_*_swiglu_back`` helpers, so a wrong saved
        tensor, wrong branch, or wrong gradient algebra in production is caught.
        """
        y = tensors[0]
        for extra in tensors[1:]:
            y = y + extra
        y1, y2 = paddle.chunk(y, 2, axis=-1)
        gate = paddle.clip(y1, max=clamp_value)
        value = paddle.clip(y2, min=-clamp_value, max=clamp_value)
        ref_out = F.silu(gate) * value
        ref_out.backward(paddle.to_tensor(upstream_np, dtype="float32"))
        return ref_out, [t.grad for t in tensors]


@unittest.skipUnless(_IMPORT_ERROR is None, _skip_reason)
class TestBiasSwiGLUImplDispatch(_CpuTestCase):
    """``bias_swiglu_impl`` routes on ``bias`` and restores the input rank."""

    def test_bias_branch_adds_bias_before_activation(self):
        """A non-None bias goes through ``BiasSwiGLUFunction`` -> swiglu(x+bias).

        The bias is non-zero, so the biased output must equal the independent
        reference on ``x + bias`` and must *differ* from the no-bias reading of
        the same ``x`` -- proving the bias branch actually consumed ``bias``.
        """
        x_np = np.array(
            [[0.5, -1.0, 2.0, 0.3], [1.5, 0.2, -0.7, 1.1]], dtype=np.float32
        )
        bias_np = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
        x = paddle.to_tensor(x_np, dtype="float32")
        bias = paddle.to_tensor(bias_np, dtype="float32")

        out_bias = fbs.bias_swiglu_impl(x, bias)
        np.testing.assert_allclose(
            out_bias.numpy(),
            _np_swiglu(x_np + bias_np),
            rtol=1e-5,
            atol=1e-6,
        )
        # The bias genuinely changed the result versus the no-bias reading.
        self.assertFalse(
            np.allclose(
                out_bias.numpy(), _np_swiglu(x_np), rtol=1e-5, atol=1e-6
            ),
            "non-zero bias must change the activation output",
        )

    def test_none_branch_runs_activation_directly(self):
        """A None bias goes through ``SwiGLUFunction`` -> swiglu(x)."""
        x_np = np.array(
            [[0.5, -1.0, 2.0, 0.3], [1.5, 0.2, -0.7, 1.1]], dtype=np.float32
        )
        out = fbs.bias_swiglu_impl(
            paddle.to_tensor(x_np, dtype="float32"), None
        )
        np.testing.assert_allclose(
            out.numpy(), _np_swiglu(x_np), rtol=1e-5, atol=1e-6
        )

    def test_3d_input_reshaped_back_matches_reference(self):
        """3D input is flattened, activated, then viewed back to ``[B, S, H/2]``.

        Uses ``B=2 != S=3`` so a stray transpose in the reshape round-trip
        cannot survive, and compares full content (not just shape) against the
        independent reference applied over the last axis.
        """
        x_np = (
            np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) / 7.0 - 1.0
        )
        out = fbs.bias_swiglu_impl(
            paddle.to_tensor(x_np, dtype="float32"), None
        )
        self.assertEqual(out.shape, [2, 3, 2])
        np.testing.assert_allclose(
            out.numpy(), _np_swiglu(x_np), rtol=1e-5, atol=1e-6
        )

    def test_2d_input_kept_2d(self):
        """The ``len(ori_shape) == 2`` branch returns the flat output as-is."""
        x_np = np.array(
            [[0.5, -1.0, 2.0, 0.3], [1.5, 0.2, -0.7, 1.1]], dtype=np.float32
        )
        out = fbs.bias_swiglu_impl(
            paddle.to_tensor(x_np, dtype="float32"), None
        )
        self.assertEqual(out.shape, [2, 2])
        np.testing.assert_allclose(
            out.numpy(), _np_swiglu(x_np), rtol=1e-5, atol=1e-6
        )

    def test_invalid_rank_raises_assertion(self):
        """Rank outside ``[2, 3]`` trips the guard assertion in ``impl``."""
        with self.assertRaises(AssertionError):
            fbs.bias_swiglu_impl(
                paddle.to_tensor([1.0, 2.0, 3.0, 4.0], dtype="float32"), None
            )
        with self.assertRaises(AssertionError):
            fbs.bias_swiglu_impl(paddle.zeros([1, 1, 2, 4]), None)


@unittest.skipUnless(_IMPORT_ERROR is None, _skip_reason)
class TestClampValuePlumbing(_CpuTestCase):
    """``clamp_value`` selects the clamped kernel only when ``> 0``."""

    # Large-magnitude entries so clamping is observable; none lands exactly on
    # a clamp boundary, so the clipped/unclipped masks are unambiguous.
    _X = np.array(
        [[3.0, -4.0, 5.0, -6.0], [2.0, 0.3, -0.5, 4.0]], dtype=np.float32
    )

    def test_positive_clamp_routes_to_clamped_kernel(self):
        """A positive ``clamp_value`` clips gate/value before the activation."""
        out = fbs.bias_swiglu_impl(
            paddle.to_tensor(self._X, dtype="float32"), None, clamp_value=1.0
        )
        np.testing.assert_allclose(
            out.numpy(),
            _np_clamped_swiglu(self._X, 1.0),
            rtol=1e-5,
            atol=1e-6,
        )
        # Clamping genuinely changed the result versus the unclamped path.
        self.assertFalse(
            np.allclose(out.numpy(), _np_swiglu(self._X), rtol=1e-4, atol=1e-4),
            "clamp_value=1.0 must bound the large-magnitude activation",
        )

    def test_zero_clamp_falls_through_to_unclamped(self):
        """``clamp_value == 0.0`` fails the ``> 0`` guard -> unclamped SwiGLU."""
        out = fbs.bias_swiglu_impl(
            paddle.to_tensor(self._X, dtype="float32"), None, clamp_value=0.0
        )
        np.testing.assert_allclose(
            out.numpy(), _np_swiglu(self._X), rtol=1e-5, atol=1e-6
        )

    def test_none_clamp_is_unclamped(self):
        """The default ``clamp_value is None`` takes the unclamped path."""
        out = fbs.bias_swiglu_impl(
            paddle.to_tensor(self._X, dtype="float32"), None
        )
        np.testing.assert_allclose(
            out.numpy(), _np_swiglu(self._X), rtol=1e-5, atol=1e-6
        )


@unittest.skipUnless(_IMPORT_ERROR is None, _skip_reason)
class TestSwiGLUFunctionCtx(_CpuTestCase):
    """``SwiGLUFunction`` saves only ``input`` and returns a single gradient."""

    # Mixed clipped (5.0, -4.0, 2.5) and unclipped entries, none on a boundary.
    _X = np.array(
        [[0.3, -0.7, 5.0, -4.0], [2.5, -0.2, 0.4, 0.9]], dtype=np.float32
    )
    _G = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float32)
    _CV = 1.0

    def test_clamped_forward_and_single_input_grad(self):
        """forward saves ``input``; backward restores it and returns one grad.

        Forward is pinned to the closed-form NumPy clamped reference; the input
        gradient is pinned to an independent primitive autograd graph. Because
        ``SwiGLUFunction.forward`` saves ``input`` (not the clamped tensor) and
        ``backward`` re-derives from it, a wrong saved tensor or wrong branch
        would surface as a gradient mismatch here.
        """
        x = paddle.to_tensor(self._X, dtype="float32")
        x.stop_gradient = False

        out = fbs.SwiGLUFunction.apply(x, False, False, self._CV)
        self.assertEqual(out.shape, [2, 2])
        np.testing.assert_allclose(
            out.numpy(),
            _np_clamped_swiglu(self._X, self._CV),
            rtol=1e-5,
            atol=1e-6,
        )

        out.backward(paddle.to_tensor(self._G, dtype="float32"))

        x_ref = paddle.to_tensor(self._X, dtype="float32")
        x_ref.stop_gradient = False
        _, (ref_grad,) = self._clamped_reference_grads(
            [x_ref], self._CV, self._G
        )
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, [2, 4])
        np.testing.assert_allclose(
            x.grad.numpy(), ref_grad.numpy(), rtol=1e-5, atol=1e-6
        )
        # Saturated gate/value legs contribute zero gradient (mask leg).
        self.assertEqual(float(x.grad.numpy()[0, 2]), 0.0)
        self.assertEqual(float(x.grad.numpy()[0, 3]), 0.0)


@unittest.skipUnless(_IMPORT_ERROR is None, _skip_reason)
class TestBiasSwiGLUFunctionCtx(_CpuTestCase):
    """``BiasSwiGLUFunction`` saves ``(input, bias)`` and returns two grads.

    ``backward`` returns ``tmp, tmp`` (the same gradient for both inputs), and
    the activation input is ``input + bias``. A same-shape bias is used so no
    broadcast reduction is involved and both returned gradients are compared
    element-wise against an independent primitive autograd graph.
    """

    _X = np.array(
        [[0.2, -0.5, 3.0, -2.0], [1.5, 0.4, -3.5, 0.6]], dtype=np.float32
    )
    _B = np.array(
        [[0.1, 0.3, -0.5, 0.2], [-0.4, 0.1, 0.8, -0.2]], dtype=np.float32
    )
    _G = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float32)
    _CV = 1.0

    def test_clamped_forward_saves_pair_and_returns_two_grads(self):
        """forward saves ``(input, bias)``; backward yields grads for both.

        The forward output must equal the clamped reference on ``input + bias``
        (proving the bias was added), and both ``input.grad`` and ``bias.grad``
        must equal the independent primitive-graph gradient w.r.t. ``input+bias``
        (equal to each other, since ``backward`` returns ``tmp, tmp``).
        """
        y_np = self._X + self._B
        x = paddle.to_tensor(self._X, dtype="float32")
        bias = paddle.to_tensor(self._B, dtype="float32")
        x.stop_gradient = False
        bias.stop_gradient = False

        out = fbs.BiasSwiGLUFunction.apply(x, bias, False, False, self._CV)
        self.assertEqual(out.shape, [2, 2])
        np.testing.assert_allclose(
            out.numpy(),
            _np_clamped_swiglu(y_np, self._CV),
            rtol=1e-5,
            atol=1e-6,
        )

        out.backward(paddle.to_tensor(self._G, dtype="float32"))

        x_ref = paddle.to_tensor(self._X, dtype="float32")
        b_ref = paddle.to_tensor(self._B, dtype="float32")
        x_ref.stop_gradient = False
        b_ref.stop_gradient = False
        _, (ref_x_grad, ref_b_grad) = self._clamped_reference_grads(
            [x_ref, b_ref], self._CV, self._G
        )

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(bias.grad)
        self.assertEqual(x.grad.shape, [2, 4])
        self.assertEqual(bias.grad.shape, [2, 4])
        np.testing.assert_allclose(
            x.grad.numpy(), ref_x_grad.numpy(), rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            bias.grad.numpy(), ref_b_grad.numpy(), rtol=1e-5, atol=1e-6
        )
        # ``backward`` returns ``tmp, tmp`` -> the two grads are identical.
        np.testing.assert_array_equal(x.grad.numpy(), bias.grad.numpy())


if __name__ == "__main__":
    unittest.main()
