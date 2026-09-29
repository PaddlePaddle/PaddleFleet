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

"""Behavioral tests for the surrogate forward of ``FlashAttnFunctor``.

During refined recompute the *second* forward pass does not recompute
attention. ``FlashAttnFunctor.forward`` is a surrogate: it (a) returns the
attention output that the first pass already produced and stashed in
``hold_tensors``, and (b) decides *which* intermediates to hand to
``ctx.save_for_backward`` so the custom backward can call the right gradient
kernel. That save-set differs by FlashAttention version -- v2 must also carry
the RNG ``seed_offset`` and ``dropout`` needed by ``flash_attn_grad``, while
the v3/cutedsl paths must not.

The version decision (``get_fa_version``) and the cutedsl routing
(``uses_cutedsl_backend``) are collaborators owned by the facade and tested
there; here they are pinned to fixed returns so each surrogate branch is
exercised deterministically on CPU. ``ctx`` is Paddle's autograd context (also
not under test); it is replaced by a recording stub so the exact save-set can
be observed. The logic under test is the production ``forward`` body: what it
returns, what it reads from ``hold_tensors``, and what it saves. Expected
values are hand-authored, never produced by calling the function under test.
"""

import os
import sys
import unittest
from unittest import mock

# Make the ``src`` layout importable when the package is not pip-installed.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import numpy as np
    import paddle

    from paddlefleet.refined_recompute import flash_attn as fa_mod
    from paddlefleet.refined_recompute.flash_attn import FlashAttnFunctor

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet_ops absent in this env
    np = None
    paddle = None
    fa_mod = None
    FlashAttnFunctor = None
    _IMPORT_ERROR = exc

_PADDLE_AVAILABLE = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle/paddlefleet not importable in this environment: {_IMPORT_ERROR}"
    if _IMPORT_ERROR is not None
    else ""
)


class _RecordingCtx:
    """Stand-in for Paddle's PyLayer context.

    ``FlashAttnFunctor.forward`` only touches ``save_for_backward`` and sets two
    plain attributes on ``ctx``. Recording the save-set lets the test observe
    exactly which intermediates each version branch preserves.
    """

    def __init__(self):
        self.saved = None
        self.fa_version = None
        self.softmax_scale = None

    def save_for_backward(self, *tensors):
        self.saved = tensors


@unittest.skipUnless(_PADDLE_AVAILABLE, _SKIP_REASON)
class TestFlashAttnFunctorSurrogateForward(unittest.TestCase):
    """Return-value, head-dim relay, and per-version save-set of the surrogate."""

    def setUp(self):
        paddle.set_device("cpu")
        # q head dim 64, v head dim 128 -- deliberately different so a swapped
        # ``(q.shape[-1], v.shape[-1])`` relay would be visible.
        self.q = paddle.zeros([1, 2, 3, 64], dtype="float32")
        self.k = paddle.zeros([1, 2, 3, 64], dtype="float32")
        self.v = paddle.zeros([1, 2, 3, 128], dtype="float32")

    def _hold_v2(self):
        # A distinguishable precomputed output so an identity return is provable.
        result_attention = paddle.arange(24, dtype="float32").reshape(
            [1, 2, 3, 4]
        )
        return {
            "result_attention": result_attention,
            "result_softmax": paddle.full(
                [1], 7.0
            ),  # read but must NOT be saved
            "softmax_lse": paddle.full([2], 1.0),
            "seed_offset": paddle.to_tensor([11, 22], dtype="int64"),
            "dropout": paddle.full([1], 0.0),
            "causal": paddle.full([1], 1.0),
            "softmax_scale": 0.125,
        }

    def _hold_v3(self):
        result_attention = paddle.arange(24, dtype="float32").reshape(
            [1, 2, 3, 4]
        )
        return {
            "result_attention": result_attention,
            "softmax_lse": paddle.full([2], 2.0),
            "causal": paddle.full([1], 1.0),
            "softmax_scale": 0.25,
        }

    def test_forward_returns_precomputed_output_by_identity(self):
        hold = self._hold_v2()
        ctx = _RecordingCtx()
        with mock.patch.object(fa_mod, "get_fa_version", return_value=2):
            out = FlashAttnFunctor.forward(ctx, self.q, self.k, self.v, hold)
        # The surrogate must relay the first-pass output object, not recompute.
        self.assertIs(out, hold["result_attention"])
        np.testing.assert_array_equal(
            out.numpy(),
            np.arange(24, dtype="float32").reshape(1, 2, 3, 4),
        )

    def test_forward_relays_q_and_v_head_dims_in_order(self):
        hold = self._hold_v2()
        ctx = _RecordingCtx()
        seen = {}

        def fake_get_fa_version(*args):
            seen["args"] = args
            return 2

        with mock.patch.object(
            fa_mod, "get_fa_version", side_effect=fake_get_fa_version
        ):
            FlashAttnFunctor.forward(ctx, self.q, self.k, self.v, hold)
        # q's last dim first, then v's last dim -- a swap would give (128, 64).
        self.assertEqual(seen["args"], (64, 128))

    def test_v2_save_set_carries_rng_state_and_omits_softmax(self):
        hold = self._hold_v2()
        ctx = _RecordingCtx()
        with mock.patch.object(fa_mod, "get_fa_version", return_value=2):
            FlashAttnFunctor.forward(ctx, self.q, self.k, self.v, hold)

        self.assertEqual(ctx.fa_version, 2)
        # v2 backward feeds ``flash_attn_grad``: q,k,v + output + lse + the RNG
        # seed_offset + dropout + causal, in that exact order.
        expected = (
            self.q,
            self.k,
            self.v,
            hold["result_attention"],
            hold["softmax_lse"],
            hold["seed_offset"],
            hold["dropout"],
            hold["causal"],
        )
        self.assertEqual(len(ctx.saved), 8)
        for i, obj in enumerate(expected):
            self.assertIs(ctx.saved[i], obj)
        # ``result_softmax`` is read from hold_tensors but deliberately not saved.
        self.assertNotIn(id(hold["result_softmax"]), [id(t) for t in ctx.saved])

    def test_v3_save_set_drops_seed_offset_and_dropout(self):
        hold = self._hold_v3()
        ctx = _RecordingCtx()
        with (
            mock.patch.object(fa_mod, "get_fa_version", return_value=3),
            mock.patch.object(
                fa_mod, "uses_cutedsl_backend", return_value=False
            ),
        ):
            out = FlashAttnFunctor.forward(ctx, self.q, self.k, self.v, hold)

        self.assertIs(out, hold["result_attention"])
        self.assertEqual(ctx.fa_version, 3)
        # v3 backward (``flash_attn_v3_grad``) needs no RNG state: exactly
        # q,k,v + output + lse + causal, and no seed_offset/dropout keys are
        # even required in hold_tensors (a KeyError here would fail the test).
        expected = (
            self.q,
            self.k,
            self.v,
            hold["result_attention"],
            hold["softmax_lse"],
            hold["causal"],
        )
        self.assertEqual(len(ctx.saved), 6)
        for i, obj in enumerate(expected):
            self.assertIs(ctx.saved[i], obj)

    def test_softmax_scale_relayed_and_defaults_to_none_when_absent(self):
        # Present: the exact stashed scale is copied onto ctx.
        hold = self._hold_v3()
        ctx = _RecordingCtx()
        with (
            mock.patch.object(fa_mod, "get_fa_version", return_value=3),
            mock.patch.object(
                fa_mod, "uses_cutedsl_backend", return_value=False
            ),
        ):
            FlashAttnFunctor.forward(ctx, self.q, self.k, self.v, hold)
        self.assertEqual(ctx.softmax_scale, 0.25)

        # Absent: ``.get`` must yield None rather than raising KeyError.
        hold_no_scale = self._hold_v3()
        del hold_no_scale["softmax_scale"]
        ctx2 = _RecordingCtx()
        with (
            mock.patch.object(fa_mod, "get_fa_version", return_value=3),
            mock.patch.object(
                fa_mod, "uses_cutedsl_backend", return_value=False
            ),
        ):
            FlashAttnFunctor.forward(
                ctx2, self.q, self.k, self.v, hold_no_scale
            )
        self.assertIsNone(ctx2.softmax_scale)


if __name__ == "__main__":
    unittest.main()
