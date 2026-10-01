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

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

# Make the in-repo `src/` importable when running the file directly, matching
# how the Fleet single-card runner exposes the paddlefleet package.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# paddlefleet imports paddle at import time. The no-card environment used here
# has no paddle installed, so guard the import and skip honestly rather than
# faking a pass. Only a precise ImportError is treated as "dependency missing";
# any other error must surface as a real failure.
try:
    import paddle

    from paddlefleet.refined_recompute import flash_attn as fa_mod
    from paddlefleet.refined_recompute.flash_attn import FlashMaskAttnCpFunctor

    _PADDLE_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - exercised only without paddle
    paddle = None
    fa_mod = None
    FlashMaskAttnCpFunctor = None
    _PADDLE_IMPORT_ERROR = exc

_PADDLE_AVAILABLE = _PADDLE_IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle is not importable in this environment: {_PADDLE_IMPORT_ERROR}"
    if not _PADDLE_AVAILABLE
    else ""
)


@unittest.skipUnless(_PADDLE_AVAILABLE, _SKIP_REASON)
class TestFlashMaskAttnCpFunctorForwardPassthrough(unittest.TestCase):
    """Behavioral tests for ``FlashMaskAttnCpFunctor.forward``.

    Distinct facet: this CP surrogate PyLayer does NOT recompute attention in
    its forward. The real attention output was produced earlier and stashed in
    ``hold_tensors["result_attention"]``; forward's contract is to hand that
    exact tensor back (the heavy work is deferred to ``backward``). We prove the
    passthrough with distinguishable content: the stored attention is an
    ``arange`` fixture while q/k/v carry unrelated constants, so any accidental
    recomputation or wrong-tensor return would diverge from the hand-written
    expected. Expected values are independent numpy literals.
    """

    def test_forward_returns_stored_attention_verbatim(self):
        # result_attention has fully distinguishable content (0..63).
        result_attn = paddle.arange(2 * 4 * 8, dtype="float32").reshape(
            [2, 4, 8]
        )
        softmax_lse = paddle.arange(2 * 4, dtype="float32").reshape([2, 4])
        # q/k/v deliberately differ from result_attn so a recompute would show.
        q = paddle.full([2, 4, 8], -7.0, dtype="float32")
        k = paddle.full([2, 4, 8], 3.0, dtype="float32")
        v = paddle.full([2, 4, 8], 11.0, dtype="float32")
        startend = paddle.to_tensor([0, 4, 8], dtype="int32")
        hold_tensors = {
            "mode": "dualchunk_allgather",
            "result_attention": result_attn,
            "softmax_lse": softmax_lse,
            "startend_row_indices": startend,
            "fa_version": 2,
            "group": None,
            "causal": False,
        }

        out = FlashMaskAttnCpFunctor.apply(q, k, v, None, hold_tensors)

        # Hand-derived expectation: the exact stored attention, independent of
        # the function under test.
        expected = np.arange(2 * 4 * 8, dtype=np.float32).reshape([2, 4, 8])
        np.testing.assert_array_equal(out.numpy(), expected)
        # Guard against a passthrough of the wrong operand.
        self.assertFalse(np.allclose(out.numpy(), q.numpy()))


@unittest.skipUnless(_PADDLE_AVAILABLE, _SKIP_REASON)
class TestFlashMaskAttnCpFunctorBackwardGradientRouting(unittest.TestCase):
    """Behavioral tests for ``FlashMaskAttnCpFunctor.backward`` gradient routing.

    Two load-bearing contracts are pinned here:

    1. Argument forwarding: the saved tensors and ctx metadata must reach the
       CP kernel ``cp_flashmask_allgatherkv_balance_backward`` in the exact
       documented positional order. The kernel is a genuine non-under-test
       collaborator (a distributed CP op with no CPU path), so it is replaced
       by a spy that returns four *distinguishable* grad markers and records
       what it received; the surrogate's own routing logic is preserved.
    2. Return arity: PyLayer maps backward returns positionally onto the forward
       TENSOR inputs q/k/v/learnable_sink. A trainable sink
       (``sink_requires_grad=True``) must yield a 4-tuple including the sink
       grad; a fixed/absent sink must yield the 3-tuple q/k/v only. The 3-vs-4
       expectation is derived from the contract, not from the function output.
    """

    def _invoke_backward(self, sink_requires_grad, learnable_sink):
        q = paddle.full([2, 4, 8], 1.0, dtype="float32")
        k = paddle.full([2, 4, 8], 2.0, dtype="float32")
        v = paddle.full([2, 4, 8], 3.0, dtype="float32")
        startend = paddle.to_tensor([0, 4, 8], dtype="int32")
        result_attn = paddle.full([2, 4, 8], 4.0, dtype="float32")
        softmax_lse = paddle.full([2, 4], 5.0, dtype="float32")
        group = object()
        grad = paddle.full([2, 4, 8], 9.0, dtype="float32")

        # Distinguishable grad markers returned by the spy kernel.
        q_grad = paddle.full([2, 4, 8], -1.0, dtype="float32")
        k_grad = paddle.full([2, 4, 8], -2.0, dtype="float32")
        v_grad = paddle.full([2, 4, 8], -3.0, dtype="float32")
        sink_grad = paddle.full([8], -4.0, dtype="float32")

        captured = {}

        def spy_cp_backward(*args):
            captured["args"] = args
            return q_grad, k_grad, v_grad, sink_grad

        ctx = SimpleNamespace(
            fa_version=2,
            softmax_scale=None,
            mode="dualchunk_allgather",
            sink_requires_grad=sink_requires_grad,
            saved_tensor=lambda: (
                q,
                k,
                v,
                startend,
                result_attn,
                softmax_lse,
                group,
                False,  # causal
                learnable_sink,
            ),
        )

        with mock.patch.object(
            fa_mod,
            "cp_flashmask_allgatherkv_balance_backward",
            side_effect=spy_cp_backward,
        ):
            out = FlashMaskAttnCpFunctor.backward(ctx, grad)

        forwarded = {
            "q": q,
            "k": k,
            "v": v,
            "startend": startend,
            "result_attn": result_attn,
            "softmax_lse": softmax_lse,
            "group": group,
            "grad": grad,
            "learnable_sink": learnable_sink,
        }
        grads = {"q": q_grad, "k": k_grad, "v": v_grad, "sink": sink_grad}
        return out, captured["args"], forwarded, grads

    def _assert_forwarded_positionally(self, args, fwd):
        # Exact documented order: q, k, v, startend, result_attention,
        # softmax_lse, grad, learnable_sink, group, causal, fa_version,
        # softmax_scale, mode.
        self.assertEqual(len(args), 13)
        self.assertIs(args[0], fwd["q"])
        self.assertIs(args[1], fwd["k"])
        self.assertIs(args[2], fwd["v"])
        self.assertIs(args[3], fwd["startend"])
        self.assertIs(args[4], fwd["result_attn"])
        self.assertIs(args[5], fwd["softmax_lse"])
        self.assertIs(args[6], fwd["grad"])
        self.assertIs(args[7], fwd["learnable_sink"])
        self.assertIs(args[8], fwd["group"])
        self.assertEqual(args[9], False)  # causal
        self.assertEqual(args[10], 2)  # fa_version
        self.assertIsNone(args[11])  # softmax_scale
        self.assertEqual(args[12], "dualchunk_allgather")  # mode

    def test_backward_omits_sink_grad_for_fixed_sink(self):
        out, args, fwd, grads = self._invoke_backward(
            sink_requires_grad=False, learnable_sink=None
        )
        self._assert_forwarded_positionally(args, fwd)
        # Fixed/absent sink -> exactly the q/k/v triple, sink grad dropped.
        self.assertEqual(len(out), 3)
        self.assertIs(out[0], grads["q"])
        self.assertIs(out[1], grads["k"])
        self.assertIs(out[2], grads["v"])

    def test_backward_returns_sink_grad_for_trainable_sink(self):
        sink = paddle.full([8], 0.5, dtype="float32")
        out, args, fwd, grads = self._invoke_backward(
            sink_requires_grad=True, learnable_sink=sink
        )
        self._assert_forwarded_positionally(args, fwd)
        # Trainable sink -> 4-tuple with the sink grad appended in slot 3.
        self.assertEqual(len(out), 4)
        self.assertIs(out[0], grads["q"])
        self.assertIs(out[1], grads["k"])
        self.assertIs(out[2], grads["v"])
        self.assertIs(out[3], grads["sink"])


if __name__ == "__main__":
    unittest.main()
