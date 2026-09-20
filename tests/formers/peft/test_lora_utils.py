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

"""Behavior tests for ``paddlefleet.peft.lora.utils.rng_ctx``.

``rng_ctx`` is the RNG-context selector used by LoRA layers: it must hand back
the fleet model-parallel RNG state context *only* when running model-parallel in
dynamic mode, and a plain ``nullcontext`` otherwise. These tests exercise the
real function against the real fleet RNG tracker on CPU (no-card), checking the
actual random stream produced inside each branch instead of just call records.
"""

import copy
import unittest
from contextlib import nullcontext

import numpy as np
import paddle
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker

from paddlefleet.peft.lora.utils import rng_ctx

# Default state name consumed by ``rng_state()`` when called without arguments,
# matching production seed setup in trainer_utils.py / seed_utils.py.
_DEFAULT_RNG_NAME = "model_parallel_rng"
_TRACKED_SEED = 20240521
_GLOBAL_SEED = 777  # deliberately different so the two streams diverge


class TestRngCtx(unittest.TestCase):
    """Real-behavior tests for ``rng_ctx`` on CPU (no-card)."""

    def setUp(self):
        # Keep this a genuine no-card test and avoid GPU-generator nondeterminism;
        # restore the original device afterwards so sibling tests are unaffected.
        orig_device = paddle.get_device()
        self.addCleanup(paddle.set_device, orig_device)
        paddle.set_device("cpu")

        # The tracker is a process-global singleton shared with production code.
        # Snapshot and restore it so mutating it here does not pollute other tests
        # (restored even if an assertion fails, via addCleanup).
        self.tracker = get_rng_state_tracker()
        orig_states = copy.copy(self.tracker.states_)
        orig_seeds = copy.copy(self.tracker.seeds_)

        def _restore():
            self.tracker.states_.clear()
            self.tracker.states_.update(orig_states)
            self.tracker.seeds_.clear()
            self.tracker.seeds_.update(orig_seeds)

        self.addCleanup(_restore)

    def _clean_tracker_with_default_state(self, seed):
        """Give the shared tracker exactly one known state under the default name."""
        self.tracker.states_.clear()
        self.tracker.seeds_.clear()
        self.tracker.add(_DEFAULT_RNG_NAME, seed)

    def _draw_from_fresh_seed(self, seed):
        """Independent reference: first draw of the stream produced by ``seed``.

        ``tracker.add(name, seed)`` captures the generator state right after
        ``paddle.seed(seed)``, so seeding directly here reproduces the exact
        stream that ``rng_state()`` restores on enter, without going through
        ``rng_ctx`` itself.
        """
        paddle.seed(seed)
        return paddle.rand([4], dtype="float32").numpy()

    def test_returns_nullcontext_unless_mp_and_dynamic(self):
        # Only (is_mp=True, in_dynamic_mode=True) may leave the nullcontext path.
        for is_mp, in_dynamic_mode in [
            (False, False),
            (False, True),
            (True, False),
        ]:
            ctx = rng_ctx(is_mp=is_mp, in_dynamic_mode=in_dynamic_mode)
            self.assertIsInstance(
                ctx,
                nullcontext,
                msg=f"expected nullcontext for is_mp={is_mp}, "
                f"in_dynamic_mode={in_dynamic_mode}",
            )
            # A nullcontext is a usable, no-op context manager yielding None.
            with ctx as entered:
                self.assertIsNone(entered)

    def test_returns_tracker_state_when_mp_and_dynamic(self):
        self._clean_tracker_with_default_state(_TRACKED_SEED)
        ctx = rng_ctx(is_mp=True, in_dynamic_mode=True)
        # Must be the fleet RNG-state context manager, not a plain nullcontext.
        self.assertNotIsInstance(ctx, nullcontext)
        reference = get_rng_state_tracker().rng_state()
        self.assertIs(type(ctx), type(reference))

    def test_mp_dynamic_branch_uses_tracked_rng_stream(self):
        self._clean_tracker_with_default_state(_TRACKED_SEED)

        # Independent expected draw for the tracked seed.
        expected_tracked = self._draw_from_fresh_seed(_TRACKED_SEED)

        # Put the *global* generator on a different stream before entering, so a
        # buggy implementation that forgets to swap RNG would draw the global
        # stream and mismatch.
        paddle.seed(_GLOBAL_SEED)
        ctx = rng_ctx(is_mp=True, in_dynamic_mode=True)
        with ctx:
            actual = paddle.rand([4], dtype="float32").numpy()

        np.testing.assert_allclose(
            actual, expected_tracked, rtol=1e-6, atol=1e-6
        )
        # Sanity: the two seeds really do produce distinguishable streams, so the
        # assertion above is not trivially satisfiable.
        expected_global = self._draw_from_fresh_seed(_GLOBAL_SEED)
        self.assertFalse(np.allclose(expected_tracked, expected_global))

    def test_nullcontext_branch_uses_global_rng_stream(self):
        self._clean_tracker_with_default_state(_TRACKED_SEED)

        # nullcontext branch must NOT swap RNG: draws follow the global stream.
        paddle.seed(_GLOBAL_SEED)
        ctx = rng_ctx(is_mp=False, in_dynamic_mode=True)
        self.assertIsInstance(ctx, nullcontext)
        with ctx:
            actual = paddle.rand([4], dtype="float32").numpy()

        expected_global = self._draw_from_fresh_seed(_GLOBAL_SEED)
        np.testing.assert_allclose(
            actual, expected_global, rtol=1e-6, atol=1e-6
        )

    def test_mp_dynamic_branch_restores_global_rng_on_exit(self):
        self._clean_tracker_with_default_state(_TRACKED_SEED)

        # Enter the tracked context and consume some randomness inside it.
        paddle.seed(_GLOBAL_SEED)
        with rng_ctx(is_mp=True, in_dynamic_mode=True):
            _ = paddle.rand([4], dtype="float32")

        # After exit the global generator must be restored to its pre-enter state,
        # so the next global draw equals the *first* draw of the global stream.
        actual_after = paddle.rand([4], dtype="float32").numpy()
        expected_after = self._draw_from_fresh_seed(_GLOBAL_SEED)
        np.testing.assert_allclose(
            actual_after, expected_after, rtol=1e-6, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
