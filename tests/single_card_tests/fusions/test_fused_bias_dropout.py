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

"""CPU control-flow tests for ``paddlefleet.fusions.fused_bias_dropout``.

The module is a functional wrapper around ``paddle.nn.functional.dropout``.
These tests assert the *control flow* that is observable on CPU without any
accelerator: which branch runs for each ``(training, bias-is-None)`` combo,
that the dropout collaborator receives the correct pre-dropout tensor
(bias-add vs. no-bias-add dispatch), that ``prob`` and ``training`` are
forwarded, and that ``get_bias_dropout_add`` / ``bias_dropout_add_unfused``
wire the ``training`` flag into the returned callable.

Scope split (see sibling ``test_fused_bias_dropout_2.py``): this file owns the
training/eval selection, the bias dispatch, and the returned-fn wiring. The
prob==0 fast path and the residual-add arithmetic are left to the sibling. The
training-mode dropout mask is stochastic and its exact numerics require a GPU
reference, so those values are never hand-derived here; only the pre-dropout
input, the forwarded ``p``/``training`` flags, and the eval-mode identity
behaviour (which is deterministic on CPU) are asserted.
"""

import unittest

try:
    import numpy as np
    import paddle

    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

if HAS_PADDLE:
    from paddlefleet.fusions.fused_bias_dropout import (
        _bias_dropout_add_func,
        bias_dropout_add_unfused,
        get_bias_dropout_add,
    )


@unittest.skipUnless(
    HAS_PADDLE, "paddle is not installed; CPU control-flow test needs paddle"
)
class TestFusedBiasDropoutControlFlow(unittest.TestCase):
    """Branch selection, dispatch and wiring for fused_bias_dropout."""

    def setUp(self):
        # These are CPU-observable control-flow assertions; pin the device to
        # CPU and restore it afterwards so we neither depend on nor leak an
        # accelerator selection.
        self._orig_device = paddle.get_device()
        paddle.set_device("cpu")
        self.addCleanup(paddle.set_device, self._orig_device)

    def _install_dropout_spy(self):
        """Wrap the genuine ``paddle.nn.functional.dropout`` collaborator.

        The real implementation still runs (it is not the code under test);
        we only record the pre-dropout input tensor plus the ``p`` and
        ``training`` arguments the wrapper receives so we can assert on the
        dispatch and flag forwarding done by the module under test.
        """
        calls = []
        real = paddle.nn.functional.dropout

        def spy(
            x,
            p=0.5,
            axis=None,
            training=True,
            mode="upscale_in_train",
            name=None,
        ):
            calls.append(
                {"input": x.numpy().copy(), "p": p, "training": training}
            )
            return real(
                x, p=p, axis=axis, training=training, mode=mode, name=name
            )

        paddle.nn.functional.dropout = spy
        self.addCleanup(setattr, paddle.nn.functional, "dropout", real)
        return calls

    # -- training/eval selection: eval dropout is identity (deterministic) --

    def test_eval_mode_runs_dropout_in_inference_no_bias(self):
        # stop_gradient defaults to True for both -> ``not residual.stop_gradient``
        # is False -> inplace branch is NOT taken, so x is left untouched and
        # out = dropout(x, training=False) + residual = x + residual exactly.
        x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        residual = paddle.to_tensor(
            [[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]]
        )
        out = _bias_dropout_add_func((x, None), residual, 0.5, False)
        np.testing.assert_array_equal(
            out.numpy(),
            [[101.0, 202.0, 303.0], [404.0, 505.0, 606.0]],
        )
        # non-inplace: the caller's x must not be mutated
        np.testing.assert_array_equal(
            x.numpy(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        )

    def test_eval_mode_adds_bias_before_identity_dropout(self):
        # eval -> dropout identity; bias branch: out = residual + (x + bias).
        x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        bias = paddle.to_tensor([10.0, 20.0, 30.0])
        residual = paddle.to_tensor(
            [[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]]
        )
        out = _bias_dropout_add_func((x, bias), residual, 0.5, False)
        # x + bias = [[11,22,33],[14,25,36]]; + residual:
        np.testing.assert_array_equal(
            out.numpy(),
            [[111.0, 222.0, 333.0], [414.0, 525.0, 636.0]],
        )
        np.testing.assert_array_equal(
            x.numpy(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        )

    # -- bias-add / no-bias-add dispatch + prob & training forwarding --

    def test_bias_added_before_dropout_and_flags_forwarded(self):
        calls = self._install_dropout_spy()
        x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        bias = paddle.to_tensor([10.0, 20.0, 30.0])
        residual = paddle.to_tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        _bias_dropout_add_func((x, bias), residual, 0.25, True)

        self.assertEqual(len(calls), 1)
        # dispatch: the bias branch must add bias BEFORE calling dropout.
        np.testing.assert_array_equal(
            calls[0]["input"], [[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]]
        )
        self.assertEqual(calls[0]["p"], 0.25)
        self.assertTrue(calls[0]["training"])

    def test_no_bias_passes_x_directly_to_dropout(self):
        calls = self._install_dropout_spy()
        x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        residual = paddle.to_tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        _bias_dropout_add_func((x, None), residual, 0.75, True)

        self.assertEqual(len(calls), 1)
        # dispatch: the no-bias branch must feed x unchanged into dropout.
        np.testing.assert_array_equal(
            calls[0]["input"], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        )
        self.assertEqual(calls[0]["p"], 0.75)
        self.assertTrue(calls[0]["training"])

    # -- returned-fn wiring: get_bias_dropout_add / bias_dropout_add_unfused --

    def test_training_flag_threads_through_full_wiring(self):
        calls = self._install_dropout_spy()
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        residual = paddle.to_tensor([[0.0, 0.0], [0.0, 0.0]])

        fn_train = get_bias_dropout_add(True, fused=False)
        fn_train((x, None), residual, 0.5)
        fn_eval = get_bias_dropout_add(False, fused=False)
        fn_eval((x, None), residual, 0.5)

        # The bound ``training`` must reach dropout through
        # get_bias_dropout_add -> bias_dropout_add_unfused -> closure -> func.
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0]["training"])
        self.assertFalse(calls[1]["training"])

    def test_get_bias_dropout_add_ignores_fused_flag(self):
        calls = self._install_dropout_spy()
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        residual = paddle.to_tensor([[0.0, 0.0], [0.0, 0.0]])

        for fused in (True, False):
            fn = get_bias_dropout_add(True, fused=fused)
            fn((x, None), residual, 0.5)

        # Only the unfused path exists: regardless of ``fused`` the returned
        # callable runs the same dropout collaborator in training mode.
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0]["training"])
        self.assertTrue(calls[1]["training"])

    def test_unfused_returns_fn_binding_training_and_forwarding_args(self):
        calls = self._install_dropout_spy()
        x = paddle.to_tensor([[5.0, 6.0, 7.0]])
        bias = paddle.to_tensor([1.0, 1.0, 1.0])
        residual = paddle.to_tensor([[0.0, 0.0, 0.0]])

        fn = bias_dropout_add_unfused(False)  # binds training=False (eval)
        self.assertTrue(callable(fn))

        out = fn((x, bias), residual, 0.5)

        # training bound to False -> dropout identity -> out = residual + (x+bias)
        np.testing.assert_array_equal(out.numpy(), [[6.0, 7.0, 8.0]])
        # the closure forwarded prob, the bound training, and the bias branch
        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0]["training"])
        self.assertEqual(calls[0]["p"], 0.5)
        np.testing.assert_array_equal(calls[0]["input"], [[6.0, 7.0, 8.0]])


if __name__ == "__main__":
    unittest.main()
