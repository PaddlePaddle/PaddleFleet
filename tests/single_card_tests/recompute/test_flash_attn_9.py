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

"""Behavior tests for the first-forward parameter-validation contract of
``RefinedRcomputeFlashMaskCpAttention``.

Facet under test: when the initial (no-grad) forward pass is entered, the
production code rejects three unsupported configurations -- ``dropout > 0``,
``causal=True`` for any mode other than ``contiguous_a2a``, and a non-None
``fixed_seed_offset`` -- each with a distinct ``NotImplementedError`` message.
These three guards execute BEFORE any hybrid-communicate-group / collective
call, so their identity, message text, and relative ordering are fully
CPU-observable with no distributed initialization and no accelerator.

Expected error identity, messages, and guard ordering below are HAND-DERIVED
from the source of ``_first_fwd`` and are NOT produced by calling the
function under test.
"""

import os
import sys
import unittest

# Make the in-tree ``src`` package importable when paddlefleet is not installed.
_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
for _p in (os.path.join(_ROOT, "src"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# paddlefleet imports paddle at import time; the local environment may have no
# paddle build. Guard the import honestly and skip (never fake a pass) when the
# dependency chain cannot be satisfied.
try:
    import paddle

    from paddlefleet.refined_recompute.flash_attn import (
        RefinedRcomputeFlashMaskCpAttention,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on environment
    paddle = None
    RefinedRcomputeFlashMaskCpAttention = None
    _IMPORT_ERROR = repr(exc)

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    "paddle / paddlefleet.refined_recompute.flash_attn not importable in this "
    f"environment: {_IMPORT_ERROR}"
)

# Hand-derived expected message fragments (copied conceptually from the three
# guards in _first_fwd, not obtained by executing the code under test).
_DROPOUT_MSG = "Dropout is not supported in FlashMask context parallel yet."
_CAUSAL_MSG = (
    "FlashMaskContextParallel does not support causal=True for mode "
    "other than 'contiguous_a2a'"
)
_SEED_MSG = "Fixed seed offset is not supported yet."


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestFirstForwardValidationContract(unittest.TestCase):
    """First-forward unsupported-parameter guards: identity, text, ordering."""

    def _cpu(self, data):
        return paddle.to_tensor(data, dtype="float32", place=paddle.CPUPlace())

    def setUp(self):
        self.attn = RefinedRcomputeFlashMaskCpAttention()
        # Guards run before any tensor math, so minimal CPU tensors suffice;
        # keep an even seq length so we never trip the later dualchunk assert
        # even if a guard were (incorrectly) skipped.
        base = [[[0.0] * 8 for _ in range(4)] for _ in range(2)]  # [2, 4, 8]
        self.q = self._cpu(base)
        self.k = self._cpu(base)
        self.v = self._cpu(base)
        self.startend = paddle.to_tensor(
            [0, 4, 8], dtype="int32", place=paddle.CPUPlace()
        )

    def _forward_first(self, **kwargs):
        # ``no_grad`` drives framework._dygraph_tracer()._has_grad to False,
        # which is the REAL production condition dispatching to _first_fwd.
        with paddle.no_grad():
            return self.attn.forward(
                self.q, self.k, self.v, self.startend, **kwargs
            )

    def test_dropout_guard_identity_and_message(self):
        with self.assertRaises(NotImplementedError) as cm:
            self._forward_first(dropout=0.1)
        self.assertEqual(str(cm.exception), _DROPOUT_MSG)

    def test_causal_guard_identity_and_message(self):
        # Default mode is "dualchunk_allgather" (!= "contiguous_a2a"), so
        # causal=True must be rejected.
        with self.assertRaises(NotImplementedError) as cm:
            self._forward_first(causal=True)
        self.assertEqual(str(cm.exception), _CAUSAL_MSG)

    def test_fixed_seed_offset_guard_identity_and_message(self):
        seed = paddle.to_tensor([0], dtype="int64", place=paddle.CPUPlace())
        with self.assertRaises(NotImplementedError) as cm:
            self._forward_first(fixed_seed_offset=seed)
        self.assertEqual(str(cm.exception), _SEED_MSG)

    def test_valid_params_pass_all_three_guards(self):
        # With dropout=0, causal=False, fixed_seed_offset=None the three guards
        # must NOT fire. Execution then reaches the hybrid-communicate-group
        # lookup, which fails WITHOUT a NotImplementedError in a non-initialized
        # single-process environment. So: any raised error here must be neither
        # of the three guard messages (proving the guards let valid config
        # through rather than mis-firing).
        try:
            self._forward_first()
        except NotImplementedError as exc:  # pragma: no cover - guard misfire
            self.fail(f"valid config wrongly rejected by a guard: {exc}")
        except Exception as exc:
            msg = str(exc)
            self.assertNotIn(_DROPOUT_MSG, msg)
            self.assertNotIn(_SEED_MSG, msg)
            self.assertNotIn("does not support causal=True", msg)

    def test_dropout_checked_before_causal(self):
        # dropout guard precedes the causal guard in _first_fwd. Supplying BOTH
        # invalid values must surface the dropout message, not the causal one.
        with self.assertRaises(NotImplementedError) as cm:
            self._forward_first(dropout=0.1, causal=True)
        self.assertEqual(str(cm.exception), _DROPOUT_MSG)
        self.assertNotIn("causal=True", str(cm.exception))

    def test_causal_checked_before_fixed_seed_offset(self):
        # causal guard precedes the fixed_seed_offset guard. Supplying both
        # (dropout left at 0) must surface the causal message.
        seed = paddle.to_tensor([0], dtype="int64", place=paddle.CPUPlace())
        with self.assertRaises(NotImplementedError) as cm:
            self._forward_first(causal=True, fixed_seed_offset=seed)
        self.assertEqual(str(cm.exception), _CAUSAL_MSG)
        self.assertNotIn("Fixed seed offset", str(cm.exception))

    def test_dropout_wins_over_all_three(self):
        # All three invalid at once still yields the first (dropout) guard,
        # pinning the full ordering dropout -> causal -> fixed_seed_offset.
        seed = paddle.to_tensor([0], dtype="int64", place=paddle.CPUPlace())
        with self.assertRaises(NotImplementedError) as cm:
            self._forward_first(
                dropout=0.5, causal=True, fixed_seed_offset=seed
            )
        self.assertEqual(str(cm.exception), _DROPOUT_MSG)


if __name__ == "__main__":
    unittest.main()
