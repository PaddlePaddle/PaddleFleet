# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed on the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU behavior tests for ``paddlefleet.tensor_parallel.random`` (part 4).

Repository module: "分布式训练" (tensor parallel RNG plumbing). This file
covers the *module-level RNG-tracker orchestration* -- deliberately DIFFERENT
branches from the sibling files that test ``CudaRNGStatesTracker`` instance
methods (init/reset/get_states/set_states/add/fork) and the name getters:

  * ``initialize_rng_tracker``: idempotent "init once" early-return, the
    ``force_reset`` re-creation path, the training-vs-inference tracker class
    selection, and the unsupported-flag assertion guards;
  * ``get_cuda_rng_tracker``: unsupported-flag guards and lazy initialization
    returning the configured tracker object;
  * ``get_all_rng_states``: the uninitialized assertion guard and the
    "returns the *live* ``states_`` dict" identity contract; and
  * ``InferenceCudaRNGStatesTracker`` (built only via ``inference_rng_tracker``):
    its no-op ``add`` / ``set_states`` and null-context ``fork`` overrides,
    contrasted against the base-tracker contract.

The seeded ``add``/``fork`` value paths and ``model_parallel_cuda_manual_seed``
require a real CUDA device (they call ``paddle.cuda.*``) and are NOT exercised
here; that is single-card GPU work. Paddle is not installed in the no-card env,
hence the honest skip -- the assertions below would run on a paddle install.

All tests mutate the module globals ``_CUDA_RNG_STATE_TRACKER`` /
``_CUDA_RNG_STATE_TRACKER_INITIALIZED``; the mixin saves and restores them (and
the process RNG state) in setUp/tearDown so no state leaks to other tests.
"""

import unittest

try:
    import paddle

    from paddlefleet.tensor_parallel import random as tp_random
    from paddlefleet.tensor_parallel.random import (
        CudaRNGStatesTracker,
        get_all_rng_states,
        get_cuda_rng_tracker,
        initialize_rng_tracker,
    )

    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

_SKIP_REASON = "paddle is not installed in this environment"


class _RngTrackerStateGuard(unittest.TestCase):
    """Save/restore the module RNG-tracker globals and process RNG state.

    Every subclass drives ``initialize_rng_tracker`` and friends, which mutate
    module-global singletons. Without restoration those singletons would leak a
    fabricated tracker into any later test in the same process (antipattern
    #11). setUp also forces a known clean, *uninitialized* slate so each test
    starts independent.
    """

    def setUp(self):
        self._orig_tracker = tp_random._CUDA_RNG_STATE_TRACKER
        self._orig_initialized = tp_random._CUDA_RNG_STATE_TRACKER_INITIALIZED
        self._orig_rng_state = paddle.get_rng_state()
        self.addCleanup(self._restore)
        tp_random._CUDA_RNG_STATE_TRACKER = None
        tp_random._CUDA_RNG_STATE_TRACKER_INITIALIZED = False

    def _restore(self):
        tp_random._CUDA_RNG_STATE_TRACKER = self._orig_tracker
        tp_random._CUDA_RNG_STATE_TRACKER_INITIALIZED = self._orig_initialized
        paddle.set_rng_state(self._orig_rng_state)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestInitializeRngTracker(_RngTrackerStateGuard):
    """initialize_rng_tracker: class selection, idempotency, force_reset."""

    def test_training_tracker_is_plain_class(self):
        # Non-inference path must build exactly the base tracker class, not a
        # subclass, and flip the module INITIALIZED flag.
        initialize_rng_tracker(inference_rng_tracker=False, force_reset=True)
        tracker = tp_random._CUDA_RNG_STATE_TRACKER
        self.assertIs(type(tracker), CudaRNGStatesTracker)
        self.assertTrue(tp_random._CUDA_RNG_STATE_TRACKER_INITIALIZED)

    def test_init_is_idempotent_without_force_reset(self):
        # Second call with no force_reset must early-return and keep the very
        # same singleton object (init-once contract).
        initialize_rng_tracker(force_reset=True)
        first = tp_random._CUDA_RNG_STATE_TRACKER
        initialize_rng_tracker()
        self.assertIs(tp_random._CUDA_RNG_STATE_TRACKER, first)

    def test_force_reset_switches_to_inference_subclass(self):
        # A training tracker is live; force_reset must discard it and rebuild
        # the inference subclass instead of returning early.
        initialize_rng_tracker(inference_rng_tracker=False, force_reset=True)
        training = tp_random._CUDA_RNG_STATE_TRACKER
        initialize_rng_tracker(inference_rng_tracker=True, force_reset=True)
        rebuilt = tp_random._CUDA_RNG_STATE_TRACKER
        self.assertIsNot(rebuilt, training)
        self.assertIs(type(training), CudaRNGStatesTracker)
        self.assertEqual(
            type(rebuilt).__name__, "InferenceCudaRNGStatesTracker"
        )
        # The inference tracker is still a CudaRNGStatesTracker subclass.
        self.assertIsInstance(rebuilt, CudaRNGStatesTracker)

    def test_rejects_te_rng_tracker(self):
        with self.assertRaises(AssertionError):
            initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)

    def test_rejects_cudagraphable_rng(self):
        with self.assertRaises(AssertionError):
            initialize_rng_tracker(use_cudagraphable_rng=True, force_reset=True)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestGetCudaRngTracker(_RngTrackerStateGuard):
    """get_cuda_rng_tracker: guards and lazy init returning the singleton."""

    def test_lazy_init_returns_training_tracker(self):
        # From the clean uninitialized slate, the getter must initialize and
        # return the same object stored in the module global.
        self.assertFalse(tp_random._CUDA_RNG_STATE_TRACKER_INITIALIZED)
        tracker = get_cuda_rng_tracker()
        self.assertIs(tracker, tp_random._CUDA_RNG_STATE_TRACKER)
        self.assertIs(type(tracker), CudaRNGStatesTracker)
        self.assertTrue(tp_random._CUDA_RNG_STATE_TRACKER_INITIALIZED)

    def test_rejects_te_rng_tracker(self):
        with self.assertRaises(AssertionError):
            get_cuda_rng_tracker(use_te_rng_tracker=True)

    def test_rejects_cudagraphable_rng(self):
        with self.assertRaises(AssertionError):
            get_cuda_rng_tracker(use_cudagraphable_rng=True)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestGetAllRngStates(_RngTrackerStateGuard):
    """get_all_rng_states: uninitialized guard and live-dict identity."""

    def test_asserts_when_uninitialized(self):
        # setUp left the module uninitialized; the guard must fire.
        self.assertFalse(tp_random._CUDA_RNG_STATE_TRACKER_INITIALIZED)
        with self.assertRaises(AssertionError):
            get_all_rng_states()

    def test_returns_live_states_dict_not_a_copy(self):
        tracker = get_cuda_rng_tracker()
        result = get_all_rng_states()
        # Must expose the tracker's live states_ object itself...
        self.assertIs(result, tracker.states_)
        # ...unlike get_states(), which returns a fresh copy.
        self.assertIsNot(result, tracker.get_states())


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestInferenceTrackerOverrides(_RngTrackerStateGuard):
    """InferenceCudaRNGStatesTracker: no-op add/set_states, null-context fork.

    These override the base contract: the base tracker's add rejects duplicate
    seeds/names and fork raises for an unknown name; the inference variant must
    silently do nothing so inference code paths carry no RNG bookkeeping.
    """

    def _inference_tracker(self):
        initialize_rng_tracker(inference_rng_tracker=True, force_reset=True)
        return tp_random._CUDA_RNG_STATE_TRACKER

    def test_add_is_noop_and_never_raises(self):
        tracker = self._inference_tracker()
        tracker.add("model-parallel-rng", 7)
        # Base tracker would raise on the duplicate name/seed; inference no-ops.
        tracker.add("model-parallel-rng", 7)
        self.assertEqual(tracker.states_, {})
        self.assertEqual(tracker.seeds_, set())

    def test_set_states_is_noop(self):
        tracker = self._inference_tracker()
        tracker.set_states({"model-parallel-rng": object()})
        # Base set_states would store the dict and mark initialized; here both
        # stay untouched from construction.
        self.assertEqual(tracker.states_, {})
        self.assertFalse(tracker.is_initialized())

    def test_fork_unknown_name_is_null_context(self):
        tracker = self._inference_tracker()
        # Base fork raises for an untracked name; inference fork yields cleanly.
        with tracker.fork("never-added-name") as value:
            self.assertIsNone(value)


if __name__ == "__main__":
    unittest.main()
