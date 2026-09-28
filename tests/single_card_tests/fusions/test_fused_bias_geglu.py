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

"""Behavior tests for the top-level GEGLU entry in
``paddlefleet.fusions.fused_bias_geglu``.

Scope (kept disjoint from concurrent sibling suites): the *control flow* of the
tanh-approx GEGLU entry only --

* ``bias_geglu_impl`` dispatch -- ``bias is not None`` selects
  ``BiasGeGLUFunction``; ``bias is None`` selects ``GeGLUFunction``; and the
  original rank (2D vs 3D) decides whether the flat output is reshaped back.
* ``GeGLUFunction`` / ``BiasGeGLUFunction`` ``forward``/``backward`` --
  which tensors ``ctx`` saves and restores, and the arity of the gradient tuple
  each ``backward`` returns (one grad for the no-bias variant, a ``(input,
  bias)`` pair for the bias variant, with the bias grad being the batch
  reduction produced by ``reduce_as``).

Everything runs eagerly on CPU (``jit_fuser`` is an identity decorator in this
tree, so no CINN/GPU compilation is involved). The tanh activation *numerics*
are deliberately left to the GPU-path / sibling suites: this file only pins the
dispatch and autograd plumbing, and it does so with expected values hand-derived
from exact anchors (``GELU(0) == 0`` so a zeroed activation half yields a zero
output and a closed-form gradient) rather than by re-deriving the tanh formula.

``paddle`` is imported at module load; when it (or ``paddlefleet``) is missing
the whole suite is skipped with an honest reason instead of a hollow pass.
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

    import paddlefleet.fusions.fused_bias_geglu as fbg

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet not installed on this host
    np = None
    paddle = None
    fbg = None
    _IMPORT_ERROR = exc


_skip_reason = f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}"


class _CpuTestCase(unittest.TestCase):
    """Base fixture pinning eager execution to CPU and restoring the device."""

    def setUp(self):
        # GEGLU control flow is device independent; force CPU so the assertions
        # describe locally observable behaviour and never claim GPU numerics.
        self._orig_device = paddle.device.get_device()
        paddle.set_device("cpu")
        self.addCleanup(paddle.set_device, self._orig_device)


@unittest.skipUnless(_IMPORT_ERROR is None, _skip_reason)
class TestBiasGeGLUImplDispatch(_CpuTestCase):
    """``bias_geglu_impl`` routes on ``bias`` and restores the input rank."""

    def test_bias_branch_adds_bias_before_activation(self):
        """A non-None bias goes through ``BiasGeGLUFunction`` -> ``geglu(x+bias)``.

        Anchor: ``GELU(0) == 0`` exactly (tanh(0)=0), so when ``x+bias`` has an
        all-zero *first* (gated) half the whole output is exactly zero. ``x``
        itself has a non-zero first half, so the no-bias reading of the same
        ``x`` is *not* zero -- proving the bias branch really consumed ``bias``
        rather than ignoring it.
        """
        # x[:, :2] = -bias[:2] so that (x + bias)[:, :2] == 0.
        x = paddle.to_tensor(
            [[-1.0, 2.0, 3.0, -5.0], [-1.0, 2.0, 2.0, 7.0]], dtype="float32"
        )
        bias = paddle.to_tensor([1.0, -2.0, 0.5, 3.0], dtype="float32")

        out_bias = fbg.bias_geglu_impl(x, bias)
        # (x+bias) gated half is zero -> GELU gate is zero -> product is zero.
        np.testing.assert_allclose(
            out_bias.numpy(), np.zeros((2, 2), dtype=np.float32), atol=0.0
        )

        # Without bias the gated half of x is [-1, 2] (non-zero) and the linear
        # half is non-zero too, so the activation cannot be all zero.
        out_no_bias_same_x = fbg.bias_geglu_impl(x, None)
        self.assertFalse(
            np.allclose(out_no_bias_same_x.numpy(), 0.0, atol=1e-6),
            "no-bias reading of x must differ from the biased (zeroed) output",
        )

    def test_none_branch_runs_activation_directly(self):
        """A None bias goes through ``GeGLUFunction`` -> ``geglu(x)``.

        Anchor: with the gated (first) half of ``x`` zeroed, ``GELU(0)==0`` makes
        the output exactly zero regardless of the linear half.
        """
        x = paddle.to_tensor(
            [[0.0, 0.0, 3.0, -5.0], [0.0, 0.0, 2.0, 7.0]], dtype="float32"
        )
        out = fbg.bias_geglu_impl(x, None)
        np.testing.assert_allclose(
            out.numpy(), np.zeros((2, 2), dtype=np.float32), atol=0.0
        )

    def test_bias_and_none_paths_are_consistent(self):
        """Dispatch consistency: ``impl(x, bias) == impl(x + bias, None)``.

        The bias branch adds ``bias`` then activates; feeding the pre-added
        tensor through the no-bias branch must reproduce it exactly. This pins
        the two dispatch branches to each other (it intentionally does not, and
        cannot, validate the shared tanh activation itself -- that is the GPU
        path's job).
        """
        x = paddle.to_tensor(
            [[0.5, -1.0, 2.0, 0.3], [1.5, 0.2, -0.7, 1.1]], dtype="float32"
        )
        bias = paddle.to_tensor([0.1, -0.2, 0.3, -0.4], dtype="float32")

        out_bias = fbg.bias_geglu_impl(x, bias)
        out_pre_added = fbg.bias_geglu_impl(x + bias, None)
        np.testing.assert_allclose(
            out_bias.numpy(), out_pre_added.numpy(), rtol=1e-6, atol=1e-6
        )

    def test_3d_input_reshaped_back_matches_flattened_2d(self):
        """3D input is flattened to 2D, activated, then viewed back to 3D.

        The ``len(ori_shape) == 3`` branch must restore ``[B, S, H]`` and the
        restore must be a pure view: the 3D result equals the 2D result on the
        flattened rows, reshaped. Content (not just shape) is compared.
        """
        x3d = paddle.to_tensor(
            [[[0.5, -1.0, 2.0, 0.3], [1.5, 0.2, -0.7, 1.1]]], dtype="float32"
        )  # shape [1, 2, 4]
        bias = paddle.to_tensor([0.1, -0.2, 0.3, -0.4], dtype="float32")

        out3d = fbg.bias_geglu_impl(x3d, bias)
        self.assertEqual(out3d.shape, [1, 2, 2])

        x2d = x3d.reshape([2, 4])
        out2d = fbg.bias_geglu_impl(x2d, bias)
        self.assertEqual(out2d.shape, [2, 2])
        np.testing.assert_allclose(
            out3d.numpy().reshape(2, 2), out2d.numpy(), rtol=1e-6, atol=1e-6
        )

    def test_2d_input_kept_2d(self):
        """The ``len(ori_shape) == 2`` branch returns the flat output as-is.

        Shape stays ``[N, H]`` and, via the zero-gate anchor, the content is
        pinned too (gated half zeroed -> exactly-zero output), so a stray
        reshape or a dropped-half bug cannot slip through on shape alone.
        """
        x = paddle.to_tensor(
            [[0.0, 0.0, 2.0, 0.3], [0.0, 0.0, -0.7, 1.1]], dtype="float32"
        )
        out = fbg.bias_geglu_impl(x, None)
        self.assertEqual(out.shape, [2, 2])
        np.testing.assert_allclose(
            out.numpy(), np.zeros((2, 2), dtype=np.float32), atol=0.0
        )

    def test_invalid_rank_raises_assertion(self):
        """Rank outside ``[2, 3]`` trips the guard assertion in ``impl``."""
        with self.assertRaises(AssertionError):
            fbg.bias_geglu_impl(paddle.to_tensor([1.0, 2.0, 3.0, 4.0]), None)
        with self.assertRaises(AssertionError):
            fbg.bias_geglu_impl(paddle.zeros([1, 1, 2, 4]), None)


@unittest.skipUnless(_IMPORT_ERROR is None, _skip_reason)
class TestGeGLUFunctionAutograd(_CpuTestCase):
    """``GeGLUFunction`` saves only ``input`` and returns a single gradient."""

    def test_forward_zero_gate_and_single_input_grad(self):
        """forward saves ``input``; backward returns one grad (for ``input``).

        With the gated (first) half of ``input`` zeroed, ``GELU(0)==0`` gives a
        zero forward output, and ``geglu_back`` collapses to a closed form
        independent of the tanh coefficients:
          - gated half of grad   = g * y2 * 0.5   (ff = 0.5 when y1 == 0)
          - linear half of grad  = 0              (GELU'(0) leg vanishes)
        so the whole input gradient is hand-derivable exactly.
        """
        x = paddle.to_tensor(
            [[0.0, 0.0, 3.0, -5.0], [0.0, 0.0, 2.0, 7.0]], dtype="float32"
        )
        x.stop_gradient = False
        g = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")

        out = fbg.GeGLUFunction.apply(x)
        # forward: GELU(gated=0) * linear == 0.
        np.testing.assert_allclose(
            out.numpy(), np.zeros((2, 2), dtype=np.float32), atol=0.0
        )

        out.backward(g)
        # y2 = x[:, 2:]; grad gated half = g * y2 * 0.5, linear half = 0.
        expected = np.array(
            [[1.5, -5.0, 0.0, 0.0], [3.0, 14.0, 0.0, 0.0]], dtype=np.float32
        )
        self.assertIsNotNone(x.grad)
        np.testing.assert_allclose(
            x.grad.numpy(), expected, rtol=1e-6, atol=1e-6
        )


@unittest.skipUnless(_IMPORT_ERROR is None, _skip_reason)
class TestBiasGeGLUFunctionAutograd(_CpuTestCase):
    """``BiasGeGLUFunction`` saves ``(input, bias)`` and returns a grad pair."""

    def test_forward_saves_pair_backward_returns_two_grads(self):
        """forward saves ``input`` and ``bias``; backward yields both grads.

        Anchor: ``bias`` is chosen so ``(input + bias)`` has an all-zero gated
        half. Then the forward output is exactly zero, and the restored
        ``(input, bias)`` combine (``geglu_back`` on ``input + bias``) to a
        closed-form gradient:
          - input grad gated half   = g * (x+b)[:, 2:] * 0.5
          - input grad linear half  = 0
          - bias grad               = reduce_as(input_grad, bias)
                                     = input_grad summed over the batch axis
        Both the pair arity and the ``reduce_as`` batch reduction are pinned.
        """
        x = paddle.to_tensor(
            [[-1.0, 2.0, 3.0, -5.0], [-1.0, 2.0, 2.0, 7.0]], dtype="float32"
        )
        bias = paddle.to_tensor([1.0, -2.0, 0.5, 3.0], dtype="float32")
        x.stop_gradient = False
        bias.stop_gradient = False
        g = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")

        out = fbg.BiasGeGLUFunction.apply(x, bias)
        # (x+bias) gated half is zero -> forward is exactly zero.
        np.testing.assert_allclose(
            out.numpy(), np.zeros((2, 2), dtype=np.float32), atol=0.0
        )

        out.backward(g)
        # (x+bias)[:, 2:] = [[3.5, -2.0], [2.5, 10.0]].
        expected_input_grad = np.array(
            [[1.75, -2.0, 0.0, 0.0], [3.75, 20.0, 0.0, 0.0]], dtype=np.float32
        )
        # reduce_as(bias): sum the [2, 4] input grad over the batch axis -> [4].
        expected_bias_grad = expected_input_grad.sum(axis=0)

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(bias.grad)
        np.testing.assert_allclose(
            x.grad.numpy(), expected_input_grad, rtol=1e-6, atol=1e-6
        )
        self.assertEqual(list(bias.grad.shape), [4])
        np.testing.assert_allclose(
            bias.grad.numpy(), expected_bias_grad, rtol=1e-6, atol=1e-6
        )

    def test_bias_grad_is_batch_reduction_of_input_grad(self):
        """``reduce_as`` reduces the input grad over the broadcast (batch) axis.

        Uses a generic distinguishable input (no zero anchor) and checks the
        production-defined relationship ``bias.grad == sum(input.grad, axis=0)``
        directly. This isolates the ``tmp.reduce_as(bias)`` line: a wrong axis,
        a dropped reduction, or ``reduce_as(input)`` would all fail here,
        independent of the tanh activation values themselves.
        """
        x = paddle.to_tensor(
            [[0.5, -1.0, 2.0, 0.3], [1.5, 0.2, -0.7, 1.1]], dtype="float32"
        )
        bias = paddle.to_tensor([0.1, -0.2, 0.3, -0.4], dtype="float32")
        x.stop_gradient = False
        bias.stop_gradient = False
        g = paddle.to_tensor([[1.0, -2.0], [0.5, 3.0]], dtype="float32")

        out = fbg.BiasGeGLUFunction.apply(x, bias)
        out.backward(g)

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(bias.grad)
        self.assertEqual(list(bias.grad.shape), [4])
        np.testing.assert_allclose(
            bias.grad.numpy(),
            x.grad.numpy().sum(axis=0),
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
