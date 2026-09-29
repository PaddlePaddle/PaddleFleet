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

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
        ),
        "src",
    ),
)

# Behavioral target:
#   paddlefleet/refined_recompute/flash_attn.py
#   RefinedRcomputeFlashMaskCpAttention._first_fwd early parameter guards.
#
# These guards live at the very top of _first_fwd and reject unsupported
# argument combinations BEFORE any call to
# fleet.get_hybrid_communicate_group(). That ordering is what makes them
# CPU-observable without a distributed environment: each guard must raise
# NotImplementedError with its own distinctive message. The tests below
# pin the exception TYPE *and* the message content, so that removing or
# swapping any single guard causes a genuine failure (a removed guard would
# fall through to the un-initialised Fleet hybrid group and raise a
# different, non-matching error instead).
#
# paddlefleet imports paddle at import time; when paddle (or the extension
# module paddlefleet_ops) is unavailable the whole module import fails with
# ImportError/ModuleNotFoundError and every test here is honestly skipped.

try:
    from paddlefleet.refined_recompute.flash_attn import (
        RefinedRcomputeFlashMaskCpAttention,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:
    RefinedRcomputeFlashMaskCpAttention = None
    _IMPORT_ERROR = exc


@unittest.skipUnless(
    RefinedRcomputeFlashMaskCpAttention is not None,
    f"paddlefleet.refined_recompute.flash_attn import failed: {_IMPORT_ERROR}",
)
class TestFirstFwdParameterGuards(unittest.TestCase):
    """Early argument validation in _first_fwd, before any distributed setup.

    The dropout, causal/mode and fixed_seed_offset checks all precede
    ``fleet.get_hybrid_communicate_group()``. Each guard is asserted by its
    exception type and its own message, which distinguishes it from the
    other two guards and from the downstream Fleet-group error.
    """

    def _make_attn(self):
        # __init__ only builds a queue.Queue and registers it with the
        # global RR queue log; it needs neither a GPU nor a process group.
        return RefinedRcomputeFlashMaskCpAttention()

    def _dummy_tensor(self):
        # The guards under test do not read tensor contents, but _first_fwd
        # requires the positional q/k/v/mask arguments. Use a tiny CPU tensor
        # with an even seq_len so shape-based asserts further down would not
        # be the thing that fires if a guard were (incorrectly) skipped.
        import paddle

        return paddle.zeros([1, 2, 1, 8], dtype="float32")

    def test_positive_dropout_rejected(self):
        """dropout > 0.0 must raise NotImplementedError about dropout."""
        attn = self._make_attn()
        q = self._dummy_tensor()
        with self.assertRaisesRegex(
            NotImplementedError, r"Dropout is not supported"
        ):
            attn._first_fwd(
                q,
                q,
                q,
                None,
                dropout=0.5,
                causal=False,
                mode="dualchunk_allgather",
            )

    def test_causal_with_non_a2a_mode_rejected(self):
        """causal=True is only allowed for the contiguous_a2a mode."""
        attn = self._make_attn()
        q = self._dummy_tensor()
        # dropout stays at its default 0.0 so the dropout guard is a no-op and
        # the causal/mode guard is the one that must fire.
        with self.assertRaisesRegex(
            NotImplementedError, r"does not support causal=True"
        ):
            attn._first_fwd(
                q,
                q,
                q,
                None,
                dropout=0.0,
                causal=True,
                mode="dualchunk_allgather",
            )

    def test_fixed_seed_offset_rejected(self):
        """A non-None fixed_seed_offset must raise NotImplementedError."""
        attn = self._make_attn()
        q = self._dummy_tensor()
        # dropout=0.0 and causal=False keep the earlier guards inactive, so a
        # match here proves the fixed_seed_offset guard specifically fired.
        with self.assertRaisesRegex(
            NotImplementedError, r"Fixed seed offset is not supported"
        ):
            attn._first_fwd(
                q,
                q,
                q,
                None,
                fixed_seed_offset=123,
                dropout=0.0,
                causal=False,
                mode="dualchunk_allgather",
            )

    def test_causal_true_is_permitted_for_contiguous_a2a(self):
        """causal=True with mode='contiguous_a2a' must NOT hit the causal guard.

        With the allowed mode the causal/mode guard is skipped, so execution
        proceeds past all three early guards to the Fleet hybrid-group lookup.
        Without an initialised Fleet that lookup fails, but crucially it must
        NOT be the "does not support causal=True" NotImplementedError. This
        pins the guard's condition to the exact (causal, mode) pair rather
        than to causal alone.
        """
        attn = self._make_attn()
        q = self._dummy_tensor()
        try:
            attn._first_fwd(
                q,
                q,
                q,
                None,
                dropout=0.0,
                causal=True,
                mode="contiguous_a2a",
            )
        except NotImplementedError as exc:
            self.assertNotIn("does not support causal=True", str(exc))
        except Exception:
            # Any non-NotImplementedError (e.g. the un-initialised Fleet
            # hybrid communicate group) confirms we passed the early guards.
            pass


if __name__ == "__main__":
    unittest.main()
