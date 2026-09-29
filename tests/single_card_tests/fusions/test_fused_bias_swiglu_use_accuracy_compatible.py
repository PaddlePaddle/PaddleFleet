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

"""CPU dispatch tests for the ``use_accuracy_compatible`` branch of the two
bias/no-bias ``paddle.autograd.PyLayer`` wrappers in
``paddlefleet.fusions.fused_bias_swiglu``.

What this file pins
-------------------
``use_accuracy_compatible`` selects *which internal callable* the PyLayer
forward/backward dispatches to (``clamp_value=None`` throughout):

============  ==========================  ===============================
flag          SwiGLUFunction.forward      SwiGLUFunction.backward
============  ==========================  ===============================
False         ``swiglu``  (F.swiglu)      ``swiglu_back`` (native grad op)
truthy        ``swiglu_eager``            ``swiglu_back_eager(.., flag)``
============  ==========================  ===============================

and, for the bias variant, ``bias_swiglu`` vs ``bias_swiglu_eager`` in the
forward and ``bias_swiglu_back`` vs ``swiglu_back_eager(g, input+bias, flag)``
in the backward.

Why the dispatch decision is the observable
-------------------------------------------
The eager paths are *designed to be numerically equivalent* to the fused paths
(the eager backward's ``targets_hf(False/True/'megatron')`` branch is the plain
analytic SwiGLU gradient, identical to ``paddle._C_ops.swiglu_grad``).  So the
flag cannot be detected from output/grad numbers alone -- the CPU-observable
effect is *which collaborator runs*.  Each test therefore spies on the two
candidate module-level callables with ``wraps=`` (the real function still runs,
we only record the call), asserts the selected one ran and the other did not,
checks the arguments threaded into it (notably that the flag value reaches
``swiglu_back_eager`` as its ``accuracy_target``), and independently anchors the
numeric result of the taken path against a hand-derived NumPy reference.

``jit_fuser`` is an identity no-op and ``F.swiglu`` / ``swiglu_grad`` have CPU
kernels (device forced to CPU), so this is plain CPU tensor math.  The GPU/Triton
kernel numerics that ``F.swiglu`` may dispatch to on device are NOT asserted.
The production module is never used to build an expectation.
"""

import os
import sys
import unittest

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

    import paddlefleet.fusions.fused_bias_swiglu as swiglu_mod
    from paddlefleet.accuracy_target import targets_hf
    from paddlefleet.fusions.fused_bias_swiglu import (
        BiasSwiGLUFunction,
        SwiGLUFunction,
    )

    HAS_PADDLE = True
    _IMPORT_ERROR = ""
except ImportError as exc:  # honest: dependency missing, not a swallowed bug
    HAS_PADDLE = False
    _IMPORT_ERROR = repr(exc)

from unittest import mock

# --- Independent NumPy references (never call production code) --------------
# Work in float64 from the mathematical definition so a same-way bug in the
# production float32 path cannot hide behind a shared implementation.


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _silu(x):
    x = np.asarray(x, dtype=np.float64)
    return x * _sigmoid(x)


def _split_last(arr):
    """Split the last axis into two equal halves (matches paddle.chunk)."""
    half = arr.shape[-1] // 2
    return arr[..., :half], arr[..., half:]


def ref_swiglu_fwd(y):
    """Plain SwiGLU: SiLU(gate) * value, gate/value being last-axis halves."""
    g, v = _split_last(np.asarray(y, dtype=np.float64))
    return _silu(g) * v


def ref_swiglu_grad(g_up, y):
    """Analytic SwiGLU input gradient for upstream ``g_up`` and pre-split ``y``.

    d/dgate  = g_up * sig(gate) * (1 + gate*(1-sig(gate))) * value
    d/dvalue = g_up * silu(gate)
    Concatenated along the last axis -> gradient w.r.t. the full pre-split input.
    This is both the ``swiglu_back_eager`` megatron formula and what the native
    ``swiglu_grad`` op computes; deriving it here keeps the reference independent.
    """
    g_up = np.asarray(g_up, dtype=np.float64)
    gate, value = _split_last(np.asarray(y, dtype=np.float64))
    s = _sigmoid(gate)
    grad_gate = g_up * s * (1.0 + gate * (1.0 - s)) * value
    grad_value = g_up * (gate * s)
    return np.concatenate([grad_gate, grad_value], axis=-1)


# Fixed, non-degenerate fixtures (distinct signs/magnitudes, no zeros, gate !=
# value halves) so a swapped half, dropped term or sign flip is observable.
_X_ROWS = [[1.0, -1.0, 2.0, 0.5], [1.5, 0.25, -2.0, 3.0]]
_BIAS_ROW = [0.5, -0.75, 1.25, -0.5]
_GUP_ROWS = [[0.7, -0.3], [0.2, 1.1]]


class _CPUFixture(unittest.TestCase):
    """Force CPU so ``F.swiglu``/``swiglu_grad`` take their CPU kernels; restore
    the caller's device even if an assertion fails."""

    def setUp(self):
        self._orig_device = paddle.get_device()
        paddle.set_device("cpu")
        self.addCleanup(lambda: paddle.set_device(self._orig_device))

    def _spy(self, name):
        """Patch module attribute ``name`` with a call-recording wrapper that
        still delegates to the real collaborator (genuine-collaborator spy)."""
        return mock.patch.object(
            swiglu_mod, name, wraps=getattr(swiglu_mod, name)
        )


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestSwiGLUForwardDispatch(_CPUFixture):
    """SwiGLUFunction.forward selects swiglu_eager vs swiglu by the flag."""

    def test_true_routes_to_eager_callable(self):
        x = paddle.to_tensor(_X_ROWS, dtype="float32")
        with self._spy("swiglu") as fused, self._spy("swiglu_eager") as eager:
            out = SwiGLUFunction.apply(x, False, False, None, True)
        eager.assert_called_once()
        self.assertFalse(fused.called)
        # the eager path received the forward input unchanged
        self.assertIs(eager.call_args.args[0], x)
        self.assertEqual(out.shape, [2, 2])
        np.testing.assert_allclose(
            out.numpy(), ref_swiglu_fwd(_X_ROWS), rtol=1e-5, atol=1e-6
        )

    def test_false_routes_to_fused_callable(self):
        x = paddle.to_tensor(_X_ROWS, dtype="float32")
        with self._spy("swiglu") as fused, self._spy("swiglu_eager") as eager:
            out = SwiGLUFunction.apply(x, False, False, None, False)
        fused.assert_called_once()
        self.assertFalse(eager.called)
        self.assertIs(fused.call_args.args[0], x)
        np.testing.assert_allclose(
            out.numpy(), ref_swiglu_fwd(_X_ROWS), rtol=1e-5, atol=1e-6
        )
        # Negative control: the anchor rejects a wrong-sign result.
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(
                out.numpy(), -ref_swiglu_fwd(_X_ROWS), rtol=1e-5, atol=1e-6
            )


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestSwiGLUBackwardDispatch(_CPUFixture):
    """SwiGLUFunction.backward selects swiglu_back_eager vs swiglu_back, and
    threads the flag into the eager helper as its accuracy_target."""

    def test_true_routes_to_eager_back_with_flag(self):
        x = paddle.to_tensor(_X_ROWS, dtype="float32")
        x.stop_gradient = False
        g_up = paddle.to_tensor(_GUP_ROWS, dtype="float32")
        with (
            self._spy("swiglu_back") as native,
            self._spy("swiglu_back_eager") as eager,
        ):
            out = SwiGLUFunction.apply(x, False, False, None, True)
            (grad_x,) = paddle.grad([out], [x], grad_outputs=[g_up])
        eager.assert_called_once()
        self.assertFalse(native.called)
        args = eager.call_args.args
        # (grad_output, input, accuracy_target=flag)
        np.testing.assert_array_equal(args[0].numpy(), g_up.numpy())
        np.testing.assert_array_equal(args[1].numpy(), x.numpy())
        self.assertEqual(args[2], True)
        self.assertFalse(
            targets_hf(args[2])
        )  # bool True -> megatron arithmetic
        np.testing.assert_allclose(
            grad_x.numpy(),
            ref_swiglu_grad(_GUP_ROWS, _X_ROWS),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_false_routes_to_native_back(self):
        x = paddle.to_tensor(_X_ROWS, dtype="float32")
        x.stop_gradient = False
        g_up = paddle.to_tensor(_GUP_ROWS, dtype="float32")
        with (
            self._spy("swiglu_back") as native,
            self._spy("swiglu_back_eager") as eager,
        ):
            out = SwiGLUFunction.apply(x, False, False, None, False)
            (grad_x,) = paddle.grad([out], [x], grad_outputs=[g_up])
        native.assert_called_once()
        self.assertFalse(eager.called)
        np.testing.assert_array_equal(
            native.call_args.args[0].numpy(), g_up.numpy()
        )
        np.testing.assert_array_equal(
            native.call_args.args[1].numpy(), x.numpy()
        )
        np.testing.assert_allclose(
            grad_x.numpy(),
            ref_swiglu_grad(_GUP_ROWS, _X_ROWS),
            rtol=1e-5,
            atol=1e-6,
        )


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestBiasSwiGLUForwardDispatch(_CPUFixture):
    """BiasSwiGLUFunction.forward selects bias_swiglu_eager vs bias_swiglu."""

    def test_true_routes_to_bias_eager(self):
        x = paddle.to_tensor(_X_ROWS, dtype="float32")
        bias = paddle.to_tensor(_BIAS_ROW, dtype="float32")
        with (
            self._spy("bias_swiglu") as fused,
            self._spy("bias_swiglu_eager") as eager,
        ):
            out = BiasSwiGLUFunction.apply(x, bias, False, False, None, True)
        eager.assert_called_once()
        self.assertFalse(fused.called)
        self.assertIs(eager.call_args.args[0], x)
        self.assertIs(eager.call_args.args[1], bias)
        expected = ref_swiglu_fwd(np.asarray(_X_ROWS) + np.asarray(_BIAS_ROW))
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_false_routes_to_bias_fused(self):
        x = paddle.to_tensor(_X_ROWS, dtype="float32")
        bias = paddle.to_tensor(_BIAS_ROW, dtype="float32")
        with (
            self._spy("bias_swiglu") as fused,
            self._spy("bias_swiglu_eager") as eager,
        ):
            out = BiasSwiGLUFunction.apply(x, bias, False, False, None, False)
        fused.assert_called_once()
        self.assertFalse(eager.called)
        self.assertIs(fused.call_args.args[0], x)
        self.assertIs(fused.call_args.args[1], bias)
        expected = ref_swiglu_fwd(np.asarray(_X_ROWS) + np.asarray(_BIAS_ROW))
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)
        # Bias must be consumed: dropping it would change the result.
        self.assertGreater(
            np.abs(expected - ref_swiglu_fwd(_X_ROWS)).max(), 1e-3
        )


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestBiasSwiGLUBackwardDispatch(_CPUFixture):
    """BiasSwiGLUFunction.backward: truthy -> swiglu_back_eager on (input+bias);
    False -> bias_swiglu_back on (input, bias)."""

    def test_true_routes_to_eager_on_input_plus_bias(self):
        x = paddle.to_tensor(_X_ROWS, dtype="float32")
        x.stop_gradient = False
        bias = paddle.to_tensor(_BIAS_ROW, dtype="float32")
        # ``bias`` is a trainable parameter in real use; it must require grad so
        # the PyLayer backward (which returns a grad for BOTH forward inputs)
        # is allowed to hand back a non-None bias gradient.
        bias.stop_gradient = False
        g_up = paddle.to_tensor(_GUP_ROWS, dtype="float32")
        with (
            self._spy("bias_swiglu_back") as fused,
            self._spy("swiglu_back_eager") as eager,
        ):
            out = BiasSwiGLUFunction.apply(x, bias, False, False, None, True)
            # ``BiasSwiGLUFunction.backward`` returns a grad for BOTH forward
            # inputs (``return tmp, tmp``). Differentiating only ``x`` prunes the
            # bias position, so paddle expects None there and rejects the real
            # grad the PyLayer hands back. Request grads for both inputs so both
            # backward outputs are consumed; only ``grad_x`` is asserted below.
            grad_x, _ = paddle.grad([out], [x, bias], grad_outputs=[g_up])
        eager.assert_called_once()
        self.assertFalse(fused.called)
        args = eager.call_args.args
        np.testing.assert_array_equal(args[0].numpy(), g_up.numpy())
        # backward reconstructs y = input + bias before the eager grad
        y_np = np.asarray(_X_ROWS) + np.asarray(_BIAS_ROW)
        np.testing.assert_allclose(args[1].numpy(), y_np, rtol=1e-6, atol=1e-6)
        self.assertEqual(args[2], True)
        np.testing.assert_allclose(
            grad_x.numpy(),
            ref_swiglu_grad(_GUP_ROWS, y_np),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_false_routes_to_bias_swiglu_back(self):
        x = paddle.to_tensor(_X_ROWS, dtype="float32")
        x.stop_gradient = False
        bias = paddle.to_tensor(_BIAS_ROW, dtype="float32")
        # ``bias`` is a trainable parameter in real use; it must require grad so
        # the PyLayer backward (which returns a grad for BOTH forward inputs)
        # is allowed to hand back a non-None bias gradient.
        bias.stop_gradient = False
        g_up = paddle.to_tensor(_GUP_ROWS, dtype="float32")
        with (
            self._spy("bias_swiglu_back") as fused,
            self._spy("swiglu_back_eager") as eager,
        ):
            out = BiasSwiGLUFunction.apply(x, bias, False, False, None, False)
            # ``BiasSwiGLUFunction.backward`` returns a grad for BOTH forward
            # inputs (``return tmp, tmp``). Differentiating only ``x`` prunes the
            # bias position, so paddle expects None there and rejects the real
            # grad the PyLayer hands back. Request grads for both inputs so both
            # backward outputs are consumed; only ``grad_x`` is asserted below.
            grad_x, _ = paddle.grad([out], [x, bias], grad_outputs=[g_up])
        fused.assert_called_once()
        self.assertFalse(eager.called)
        args = fused.call_args.args
        np.testing.assert_array_equal(args[0].numpy(), g_up.numpy())
        np.testing.assert_array_equal(args[1].numpy(), x.numpy())
        np.testing.assert_array_equal(args[2].numpy(), bias.numpy())
        y_np = np.asarray(_X_ROWS) + np.asarray(_BIAS_ROW)
        np.testing.assert_allclose(
            grad_x.numpy(),
            ref_swiglu_grad(_GUP_ROWS, y_np),
            rtol=1e-5,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
