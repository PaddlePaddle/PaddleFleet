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

"""Behavioral tests for the two-phase forward dispatch of
``RefinedRcomputeFlashAttention``.

The refined-recompute wrapper does not compute attention itself in a single
call; instead its public ``forward``/``__call__`` entry point *routes* to one
of two internal passes depending on the dygraph tracer's grad state:

* When gradients are disabled (the initial forward pass), it must call
  ``_first_fwd`` and forward every user argument (dropout/causal/scale/...).
* When gradients are enabled (the recompute replay during backward), it must
  call ``_second_fwd`` -- but only after asserting the hold-tensors queue that
  ``_first_fwd`` populated is non-empty.

These tests pin that routing contract and the empty-queue guard. The two inner
passes invoke GPU FlashAttention kernels, so they are the *non-under-test*
collaborators here: we replace them with markers that return distinguishable
values and record the exact arguments they received, then assert that the real
``forward`` dispatch logic (the production ``if not _has_grad`` branch plus the
``assert not queue.empty()`` guard) selected the correct pass and relayed the
result unchanged. Expected values are hand-authored sentinels, never produced
by calling the function under test.
"""

import os
import sys
import unittest

# Make the ``src`` layout importable when the package is not pip-installed.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import paddle
    from paddle import framework

    from paddlefleet.refined_recompute.flash_attn import (
        RefinedRcomputeFlashAttention,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet_ops absent in this env
    paddle = None
    framework = None
    RefinedRcomputeFlashAttention = None
    _IMPORT_ERROR = exc

_PADDLE_AVAILABLE = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle/paddlefleet not importable in this environment: {_IMPORT_ERROR}"
    if _IMPORT_ERROR is not None
    else ""
)


@unittest.skipUnless(_PADDLE_AVAILABLE, _SKIP_REASON)
class TestRefinedRcomputeFlashAttnForwardDispatch(unittest.TestCase):
    """forward()/__call__() must route by grad-state and guard the queue."""

    def setUp(self):
        # Keep everything on CPU: the dispatch logic under test is device
        # independent and the inner kernels are replaced with markers.
        paddle.set_device("cpu")
        self.rrfa = RefinedRcomputeFlashAttention()

        # Distinguishable, position-unique inputs so an argument swap would be
        # visible via identity checks.
        self.q = paddle.to_tensor([[1.0, 2.0]], dtype="float32")
        self.k = paddle.to_tensor([[3.0, 4.0]], dtype="float32")
        self.v = paddle.to_tensor([[5.0, 6.0]], dtype="float32")

        # Sentinels returned by the replaced passes. They are unrelated to any
        # attention computation, so only the routing decision can make them
        # appear in forward()'s return value.
        self._first_marker = ("FIRST", object())
        self._second_marker = ("SECOND", object())

        self.first_calls = []
        self.second_calls = []

    def _install_spies(self):
        rrfa = self.rrfa
        first_marker = self._first_marker
        second_marker = self._second_marker
        first_calls = self.first_calls
        second_calls = self.second_calls

        # Instance attributes shadow the class methods; production calls
        # ``self._first_fwd(...)`` so these are invoked without an implicit
        # ``self`` argument.
        def fake_first(
            q,
            k,
            v,
            dropout=0.0,
            causal=True,
            return_softmax=False,
            training=True,
            softmax_scale=None,
        ):
            first_calls.append(
                {
                    "q": q,
                    "k": k,
                    "v": v,
                    "dropout": dropout,
                    "causal": causal,
                    "return_softmax": return_softmax,
                    "training": training,
                    "softmax_scale": softmax_scale,
                }
            )
            return first_marker

        def fake_second(q, k, v):
            second_calls.append({"q": q, "k": k, "v": v})
            return second_marker

        rrfa._first_fwd = fake_first
        rrfa._second_fwd = fake_second

    def test_no_grad_routes_to_first_forward_and_relays_args(self):
        """Grad disabled => _first_fwd, with all kwargs relayed verbatim."""
        self._install_spies()

        with paddle.no_grad():
            self.assertFalse(framework._dygraph_tracer()._has_grad)
            out = self.rrfa.forward(
                self.q,
                self.k,
                self.v,
                dropout=0.25,
                causal=False,
                return_softmax=True,
                training=False,
                softmax_scale=0.125,
            )

        # Routed to the first pass exactly once, second pass untouched.
        self.assertEqual(len(self.first_calls), 1)
        self.assertEqual(len(self.second_calls), 0)

        # forward() returns the first pass result unchanged (same object).
        self.assertIs(out, self._first_marker)

        # Tensors relayed by identity; scalar options relayed by value.
        call = self.first_calls[0]
        self.assertIs(call["q"], self.q)
        self.assertIs(call["k"], self.k)
        self.assertIs(call["v"], self.v)
        self.assertEqual(call["dropout"], 0.25)
        self.assertEqual(call["causal"], False)
        self.assertEqual(call["return_softmax"], True)
        self.assertEqual(call["training"], False)
        self.assertEqual(call["softmax_scale"], 0.125)

        # The initial pass has not been faked to populate the queue here, and
        # forward() itself must not touch it on the first-pass branch.
        self.assertTrue(self.rrfa._hold_tensors_queue.empty())

    def test_call_delegates_to_forward(self):
        """__call__(*args) must delegate to forward() (same routing)."""
        self._install_spies()

        with paddle.no_grad():
            out = self.rrfa(self.q, self.k, self.v, causal=True)

        self.assertEqual(len(self.first_calls), 1)
        self.assertEqual(len(self.second_calls), 0)
        self.assertIs(out, self._first_marker)
        self.assertIs(self.first_calls[0]["q"], self.q)
        self.assertEqual(self.first_calls[0]["causal"], True)

    def test_grad_enabled_with_nonempty_queue_routes_to_second(self):
        """Grad enabled + non-empty queue => _second_fwd, result relayed."""
        self._install_spies()

        # Emulate a completed first pass having enqueued its hold-tensors; the
        # guard in forward() reads emptiness before dispatching to the second
        # pass, so a sentinel entry is sufficient to exercise the success path.
        sentinel = {"marker": "held"}
        self.rrfa._hold_tensors_queue.put(sentinel)

        with paddle.enable_grad():
            self.assertTrue(framework._dygraph_tracer()._has_grad)
            out = self.rrfa.forward(self.q, self.k, self.v)

        self.assertEqual(len(self.second_calls), 1)
        self.assertEqual(len(self.first_calls), 0)
        self.assertIs(out, self._second_marker)

        call = self.second_calls[0]
        self.assertIs(call["q"], self.q)
        self.assertIs(call["k"], self.k)
        self.assertIs(call["v"], self.v)

    def test_grad_enabled_with_empty_queue_raises_guard(self):
        """Grad enabled + empty queue must trip the non-empty assertion."""
        self._install_spies()

        self.assertTrue(self.rrfa._hold_tensors_queue.empty())

        with paddle.enable_grad():  # noqa: SIM117
            with self.assertRaises(AssertionError) as ctx:
                self.rrfa.forward(self.q, self.k, self.v)

        self.assertIn("queue should not be empty", str(ctx.exception))
        # The guard fires before the second pass runs.
        self.assertEqual(len(self.second_calls), 0)


if __name__ == "__main__":
    unittest.main()
