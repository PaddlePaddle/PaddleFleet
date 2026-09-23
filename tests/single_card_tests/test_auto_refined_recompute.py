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

"""Tests for AutoRefinedRecompute: output and gradients must not change."""

import unittest

import paddle
from paddle.distributed.fleet.utils import recompute

from paddlefleet.refined_recompute import AutoRefinedRecompute

_SEED = 42
_TOKENS, _HIDDEN, _OUT = 16, 16, 12


class _Boundary(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.proj = paddle.nn.Linear(_HIDDEN, _OUT, bias_attr=False)
        self.call_count = 0

    def forward(self, x):
        # Nonlinear on purpose: through a linear chain the input gradient does not
        # depend on activations, so a mispaired frame would still give the right
        # answer and the pairing test below would be vacuous.
        self.call_count += 1
        return paddle.nn.functional.silu(self.proj(x)), None


class _Block(paddle.nn.Layer):
    def __init__(self, use_rr):
        super().__init__()
        # The boundary input is produced inside the recompute region, as at every
        # real call site: a non-leaf under the first pass's no_grad reports
        # stop_gradient=True, which is the trap the helper works around.
        self.pre = paddle.nn.Linear(_HIDDEN, _HIDDEN)
        self.boundary = _Boundary()
        self.tail = paddle.nn.Linear(_OUT, _OUT)
        self.rr = AutoRefinedRecompute("boundary")
        self.use_rr = use_rr

    def _inner(self, x):
        x = paddle.nn.functional.silu(self.pre(x))
        if self.use_rr:
            out, _ = self.rr(self.boundary, x)
        else:
            out, _ = self.boundary(x)
        return self.tail(out)

    def forward(self, x):
        return recompute(self._inner, x)


def _build(use_rr):
    paddle.seed(_SEED)
    block = _Block(use_rr)
    block.train()
    return block


def _forward_backward(block, x):
    """Run forward + backward; return output, input grad and parameter grads."""
    x = x.clone()
    x.stop_gradient = False
    output = block(x)
    # Weight by position so a row mix-up cannot cancel out in the sum.
    weights = paddle.arange(1, _OUT + 1, dtype="float32")
    (output * weights).sum().backward()
    grads = {
        name: param.grad.detach()
        for name, param in block.named_parameters()
        if param.grad is not None
    }
    return output.detach(), x.grad.detach(), grads


class TestAutoRefinedRecompute(unittest.TestCase):
    def setUp(self):
        paddle.seed(123)
        self.x = paddle.randn([_TOKENS, _HIDDEN])

    def _assert_same(self, ref, got, name):
        self.assertTrue(
            paddle.equal_all(ref, got).item(),
            f"{name} mismatch: max diff = {(ref - got).abs().max().item()}",
        )

    def test_recompute_numerical_correctness(self):
        """RR path should produce identical output and grads as plain recompute."""
        base = _build(use_rr=False)
        refined = _build(use_rr=True)

        out_ref, grad_ref, params_ref = _forward_backward(base, self.x)
        out_rr, grad_rr, params_rr = _forward_backward(refined, self.x)

        self.assertEqual(base.boundary.call_count, 2)
        # Without this the comparison below is vacuous: an implementation that
        # quietly fell through to the baseline would pass every parity check.
        self.assertEqual(refined.boundary.call_count, 1)
        self.assertEqual(refined.rr.pending, 0, "a retained frame leaked")

        self._assert_same(out_ref, out_rr, "output")
        self._assert_same(grad_ref, grad_rr, "input grad")
        self.assertEqual(
            set(params_ref), set(params_rr), "a parameter lost its grad"
        )
        for name in params_ref:
            self._assert_same(
                params_ref[name], params_rr[name], f"grad of {name}"
            )

    def test_pairs_invocations_in_flight(self):
        """1F1B forwards several micro-batches before any backward; each must
        replay its own frame, in order."""
        batches = (self.x, 1.0 - 2.0 * self.x)
        expected = [_forward_backward(_build(False), b)[1] for b in batches]

        block = _build(use_rr=True)
        inputs, outputs = [], []
        for batch in batches:
            x = batch.clone()
            x.stop_gradient = False
            inputs.append(x)
            outputs.append(block(x))
        self.assertEqual(block.rr.pending, len(batches))

        weights = paddle.arange(1, _OUT + 1, dtype="float32")
        for index, (x, output) in enumerate(zip(inputs, outputs)):
            (output * weights).sum().backward()
            self.assertEqual(block.rr.pending, len(batches) - index - 1)
            self._assert_same(expected[index], x.grad.detach(), f"grad {index}")


class TestAutoRefinedRecomputeContract(unittest.TestCase):
    """Error paths and both output shapes must be exercised, not just parity.

    Driven directly: ``paddle.no_grad`` is how full recompute runs the first
    pass, so the first/replay branches are reachable without a real recompute
    region, and each error surfaces where it is raised.
    """

    def test_single_tensor_output_round_trips(self):
        helper = AutoRefinedRecompute("point")
        x = paddle.randn([_TOKENS, _HIDDEN])
        with paddle.no_grad():
            first = helper(paddle.nn.functional.relu, x)
        self.assertIsInstance(first, paddle.Tensor)
        out = helper(paddle.nn.functional.relu, x)
        self.assertIsInstance(out, paddle.Tensor)
        self.assertEqual(helper.pending, 0)

    def test_rejects_a_non_tensor_input(self):
        helper = AutoRefinedRecompute("point")
        with self.assertRaisesRegex(TypeError, "must be Tensors"):
            helper(lambda a, b: a * b, paddle.randn([_TOKENS, _HIDDEN]), 2.0)

    def test_rejects_a_non_tensor_output(self):
        helper = AutoRefinedRecompute("point")
        with (
            paddle.no_grad(),
            self.assertRaisesRegex(TypeError, "Tensor or flat tuple"),
        ):
            helper(lambda t: {"y": t}, paddle.randn([_TOKENS, _HIDDEN]))

    def test_rejects_a_replay_without_a_first_pass(self):
        helper = AutoRefinedRecompute("point")
        with self.assertRaisesRegex(RuntimeError, r"\[point\] no frame"):
            helper(paddle.nn.functional.relu, paddle.randn([_TOKENS, _HIDDEN]))

    def test_rejects_an_input_shape_change_between_passes(self):
        helper = AutoRefinedRecompute("point")
        with paddle.no_grad():
            helper(paddle.nn.functional.relu, paddle.randn([_TOKENS, _HIDDEN]))
        with self.assertRaisesRegex(RuntimeError, "input changed"):
            helper(
                paddle.nn.functional.relu,
                paddle.randn([_TOKENS + 1, _HIDDEN]),
            )

    def test_returns_no_gradient_for_a_stop_gradient_input(self):
        """A float input the outer graph does not want a gradient for still
        accumulates one internally; handing it back is a hard error, so the
        outer input's own flag decides what is returned."""
        helper = AutoRefinedRecompute("point")
        hidden = paddle.randn([_TOKENS, _HIDDEN])
        hidden.stop_gradient = False
        constant = paddle.full([_TOKENS, _HIDDEN], 2.0)
        self.assertTrue(constant.stop_gradient)

        with paddle.no_grad():
            helper(paddle.multiply, hidden, constant)
        helper(paddle.multiply, hidden, constant).sum().backward()

        self.assertIsNotNone(hidden.grad)
        self.assertIsNone(constant.grad)


if __name__ == "__main__":
    unittest.main()
