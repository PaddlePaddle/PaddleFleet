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
"""CPU-observable behaviour tests for the base recompute flash-attention module.

Target production file:
  ``src/paddlefleet/refined_recompute/flash_attn.py``

Scope of this file (core / primary entry points of the base module):
  * ``flashattn_auto_cast`` -- the pure dtype-normalisation helper used by both
    the first and second forward passes.  Numeric content, per-operand routing
    and the no-copy identity contract are verified against hand-derived values.
  * ``RefinedRcomputeFlashAttention`` -- the standard (non-masked) refined
    recompute entry class: queue construction, the ``forward`` grad-state
    dispatch to ``_first_fwd`` / ``_second_fwd``, the empty-queue precondition
    of the recompute (second) pass, and ``__call__`` delegation.

The FlashMask / context-parallel variants in this module are covered by the
sibling ``test_flash_attn_*`` files and are intentionally not duplicated here.

The actual attention math runs in GPU C++ kernels (``_C_ops.flash_attn`` etc.)
which cannot execute on CPU; those numerical paths are single-card concerns and
are NOT claimed to be verified here.  We only exercise the CPU-observable
control logic and the pure cast helper.  ``paddlefleet`` imports ``paddle`` at
import time, so when paddle is unavailable the whole module is honestly skipped
(never faked as passing).
"""

import os
import queue
import sys
import unittest
from unittest.mock import MagicMock, patch

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 3)
)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

try:
    import numpy as np
    import paddle

    from paddlefleet.refined_recompute import flash_attn as fa_mod
    from paddlefleet.refined_recompute.flash_attn import (
        RefinedRcomputeFlashAttention,
        flashattn_auto_cast,
    )

    _HAS_PADDLE = True
    _SKIP_REASON = ""
except ImportError as exc:  # paddle / paddlefleet not installed in this env
    _HAS_PADDLE = False
    _SKIP_REASON = f"paddle/paddlefleet unavailable: {exc}"


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestFlashattnAutoCast(unittest.TestCase):
    """``flashattn_auto_cast`` normalises q/k/v dtype without touching content.

    All fixture values (1.0, 2.0, -3.0, 0.5, ...) are exactly representable in
    bfloat16 and float16, so a correct cast round-trips to float32 with no
    rounding error and equality comparisons are exact.
    """

    def test_casts_float32_to_default_bfloat16(self):
        # Default target dtype is bfloat16; float32 inputs must all be cast and
        # their numeric content preserved.
        q = paddle.to_tensor([[1.0, 2.0, -3.0, 0.5]], dtype="float32")
        k = paddle.to_tensor([[4.0, -0.25, 8.0, 16.0]], dtype="float32")
        v = paddle.to_tensor([[-2.0, 0.125, 32.0, -64.0]], dtype="float32")

        out_q, out_k, out_v = flashattn_auto_cast(q, k, v)

        for out in (out_q, out_k, out_v):
            self.assertEqual(out.dtype, paddle.bfloat16)
        np.testing.assert_array_equal(
            out_q.astype("float32").numpy(), q.numpy()
        )
        np.testing.assert_array_equal(
            out_k.astype("float32").numpy(), k.numpy()
        )
        np.testing.assert_array_equal(
            out_v.astype("float32").numpy(), v.numpy()
        )

    def test_returns_same_object_when_dtype_matches(self):
        # No-copy contract: an operand already at the target dtype is returned
        # unchanged (same object), i.e. the helper only casts on mismatch.
        q = paddle.to_tensor([[1.0, 2.0]], dtype="bfloat16")
        k = paddle.to_tensor([[3.0, 4.0]], dtype="bfloat16")
        v = paddle.to_tensor([[5.0, 6.0]], dtype="bfloat16")

        out_q, out_k, out_v = flashattn_auto_cast(q, k, v)

        self.assertIs(out_q, q)
        self.assertIs(out_k, k)
        self.assertIs(out_v, v)

    def test_only_mismatched_operands_cast_and_not_swapped(self):
        # Distinguishable content per operand: only q and v (float32) are cast
        # to the requested float16; k is already float16 and must pass through
        # untouched. Values must land on the matching output (no q/k/v swap).
        q = paddle.to_tensor([[1.0, 2.0]], dtype="float32")
        k = paddle.to_tensor([[10.0, 20.0]], dtype="float16")
        v = paddle.to_tensor([[100.0, 200.0]], dtype="float32")

        out_q, out_k, out_v = flashattn_auto_cast(q, k, v, dtype=paddle.float16)

        # k already matches -> returned unchanged (identity), not recreated.
        self.assertIs(out_k, k)
        # q and v are freshly cast to float16, values routed to correct slot.
        self.assertEqual(out_q.dtype, paddle.float16)
        self.assertEqual(out_v.dtype, paddle.float16)
        self.assertIsNot(out_q, q)
        self.assertIsNot(out_v, v)
        np.testing.assert_array_equal(
            out_q.astype("float32").numpy(), [[1.0, 2.0]]
        )
        np.testing.assert_array_equal(
            out_v.astype("float32").numpy(), [[100.0, 200.0]]
        )


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestRefinedRcomputeFlashAttention(unittest.TestCase):
    """Dispatch and precondition behaviour of the standard recompute entry.

    The tracer's ``_has_grad`` flag is a non-tested collaborator; patching it
    lets us drive the two control-flow branches of ``forward`` on CPU without
    reaching the GPU-only attention kernels.
    """

    def _tracer(self, has_grad):
        tracer = MagicMock()
        tracer._has_grad = has_grad
        return tracer

    def test_init_creates_empty_queue(self):
        attn = RefinedRcomputeFlashAttention()
        self.assertIsInstance(attn._hold_tensors_queue, queue.Queue)
        self.assertTrue(attn._hold_tensors_queue.empty())

    def test_forward_dispatches_to_first_fwd_with_all_params(self):
        # _has_grad False -> the normal (first) forward pass. forward must call
        # _first_fwd, forward every argument unchanged, and return its result.
        attn = RefinedRcomputeFlashAttention()
        captured = {}
        sentinel = ("FIRST_OUT", "FIRST_SOFTMAX")

        def fake_first_fwd(q, k, v, **kwargs):
            captured["pos"] = (q, k, v)
            captured["kwargs"] = kwargs
            return sentinel

        q = paddle.to_tensor([[1.0]], dtype="float32")
        k = paddle.to_tensor([[2.0]], dtype="float32")
        v = paddle.to_tensor([[3.0]], dtype="float32")

        with (
            patch.object(
                fa_mod.framework,
                "_dygraph_tracer",
                return_value=self._tracer(False),
            ),
            patch.object(attn, "_first_fwd", side_effect=fake_first_fwd),
            patch.object(
                attn, "_second_fwd", side_effect=AssertionError("must not run")
            ),
        ):
            result = attn.forward(
                q,
                k,
                v,
                dropout=0.1,
                causal=False,
                return_softmax=True,
                training=False,
                softmax_scale=0.25,
            )

        self.assertEqual(result, sentinel)
        self.assertIs(captured["pos"][0], q)
        self.assertIs(captured["pos"][1], k)
        self.assertIs(captured["pos"][2], v)
        self.assertEqual(captured["kwargs"]["dropout"], 0.1)
        self.assertIs(captured["kwargs"]["causal"], False)
        self.assertIs(captured["kwargs"]["return_softmax"], True)
        self.assertIs(captured["kwargs"]["training"], False)
        self.assertEqual(captured["kwargs"]["softmax_scale"], 0.25)

    def test_forward_dispatches_to_second_fwd_when_grad_active(self):
        # _has_grad True with a populated queue -> the recompute (second) pass.
        attn = RefinedRcomputeFlashAttention()
        attn._hold_tensors_queue.put({"marker": 1})
        captured = {}
        sentinel = ("SECOND_OUT", None)

        def fake_second_fwd(q, k, v):
            captured["pos"] = (q, k, v)
            return sentinel

        q = paddle.to_tensor([[1.0]], dtype="float32")
        k = paddle.to_tensor([[2.0]], dtype="float32")
        v = paddle.to_tensor([[3.0]], dtype="float32")

        with (
            patch.object(
                fa_mod.framework,
                "_dygraph_tracer",
                return_value=self._tracer(True),
            ),
            patch.object(attn, "_second_fwd", side_effect=fake_second_fwd),
            patch.object(
                attn, "_first_fwd", side_effect=AssertionError("must not run")
            ),
        ):
            result = attn.forward(q, k, v)

        self.assertEqual(result, sentinel)
        self.assertIs(captured["pos"][0], q)
        self.assertIs(captured["pos"][1], k)
        self.assertIs(captured["pos"][2], v)

    def test_second_pass_requires_non_empty_queue(self):
        # Recompute contract: with grad active but an empty queue (no prior
        # first pass recorded), forward must raise instead of silently
        # continuing into the surrogate layer.
        attn = RefinedRcomputeFlashAttention()
        self.assertTrue(attn._hold_tensors_queue.empty())

        q = paddle.to_tensor([[1.0]], dtype="float32")
        k = paddle.to_tensor([[2.0]], dtype="float32")
        v = paddle.to_tensor([[3.0]], dtype="float32")

        with (
            patch.object(
                fa_mod.framework,
                "_dygraph_tracer",
                return_value=self._tracer(True),
            ),
            self.assertRaises(AssertionError),
        ):
            attn.forward(q, k, v)

    def test_call_delegates_to_forward(self):
        # __call__ is a thin alias: it must forward *args/**kwargs verbatim and
        # return forward's result.
        attn = RefinedRcomputeFlashAttention()
        captured = {}
        sentinel = object()

        def fake_forward(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return sentinel

        with patch.object(attn, "forward", side_effect=fake_forward):
            result = attn("a", "b", "c", dropout=0.3, causal=True)

        self.assertIs(result, sentinel)
        self.assertEqual(captured["args"], ("a", "b", "c"))
        self.assertEqual(captured["kwargs"], {"dropout": 0.3, "causal": True})


if __name__ == "__main__":
    unittest.main()
