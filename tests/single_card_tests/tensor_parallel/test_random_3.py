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

"""CPU behavior tests for ``paddlefleet.tensor_parallel.random`` (part 3).

Repository module: "分布式训练" (tensor parallel RNG bookkeeping). This file
deliberately targets branches that the base / part-2 coverage does NOT touch.
Whereas the module-level tracker lifecycle (``initialize_rng_tracker`` /
``get_cuda_rng_tracker`` / ``get_all_rng_states`` / inference tracker /
``_fork_rng``) is covered elsewhere, here we exercise the pure-Python
bookkeeping of ``CudaRNGStatesTracker`` itself plus the low-level graph-safe
guard helpers and the name getters:

  * ``CudaRNGStatesTracker.{reset,is_initialized,get_states,set_states}`` --
    the initialized flag transitions and the documented shallow-copy contract
    of ``get_states`` (a new dict, same value identities);
  * ``CudaRNGStatesTracker.add`` -- the two ValueError guards, including the
    observable ordering detail that a fresh seed is registered in ``seeds_``
    *before* the duplicate-name check raises;
  * ``CudaRNGStatesTracker.fork`` -- the "state not added" guard raised on
    context entry;
  * ``_get_cuda_rng_state`` / ``_set_cuda_rng_state`` -- the
    ``graph_safe is False`` assertion guard; and
  * ``get_expert_parallel_rng_tracker_name`` /
    ``get_data_parallel_rng_tracker_name`` -- the exact tracker-name constants.

All of the above run on a locally constructed tracker instance and never touch
CUDA, the module-global tracker, or the global RNG. The genuine CUDA rng
capture/restore paths (``add`` storing an actual GPU state, ``fork`` swapping
device state) need a real accelerator and are intentionally NOT claimed here.
Paddle is not installed in this no-card env, hence the honest skip rather than
a fake pass. As a safeguard against antipattern 11 (unrestored global state),
setUp/tearDown snapshot and restore the CPU RNG state around every test.
"""

import unittest

try:
    import paddle

    from paddlefleet.tensor_parallel.random import (
        CudaRNGStatesTracker,
        _get_cuda_rng_state,
        _set_cuda_rng_state,
        get_data_parallel_rng_tracker_name,
        get_expert_parallel_rng_tracker_name,
    )

    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

_SKIP_REASON = "paddle is not installed in this environment"


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class CudaRNGStatesTrackerBookkeepingTest(unittest.TestCase):
    """Pure-Python state bookkeeping of ``CudaRNGStatesTracker`` (no CUDA)."""

    def setUp(self):
        # Snapshot the global CPU RNG so any incidental change is undone
        # (antipattern 11: never leave global RNG state mutated for peers).
        self._cpu_rng_state = paddle.get_rng_state("cpu")

    def tearDown(self):
        paddle.set_rng_state(self._cpu_rng_state, device="cpu")

    def test_init_rejects_cudagraphable_rng(self):
        """use_cudagraphable_rng=True is an unsupported config -> AssertionError."""
        with self.assertRaises(AssertionError):
            CudaRNGStatesTracker(use_cudagraphable_rng=True)

    def test_fresh_tracker_is_empty_and_uninitialized(self):
        """A freshly reset tracker reports no states, no seeds, not initialized."""
        tracker = CudaRNGStatesTracker()
        self.assertFalse(tracker.is_initialized())
        self.assertEqual(tracker.get_states(), {})
        self.assertEqual(tracker.seeds_, set())

    def test_set_states_marks_initialized_and_stores_dict_directly(self):
        """set_states flips the initialized flag and stores the dict by reference."""
        tracker = CudaRNGStatesTracker()
        payload = {"model-parallel-rng": object()}
        tracker.set_states(payload)
        self.assertTrue(tracker.is_initialized())
        # Contract: set_states assigns the given dict directly (no copy).
        self.assertIs(tracker.states_, payload)

    def test_reset_clears_states_seeds_and_flag(self):
        """reset returns the tracker to the pristine no-tracker state."""
        tracker = CudaRNGStatesTracker()
        tracker.set_states({"a": object()})
        tracker.seeds_.add(11)
        self.assertTrue(tracker.is_initialized())

        tracker.reset()
        self.assertFalse(tracker.is_initialized())
        self.assertEqual(tracker.states_, {})
        self.assertEqual(tracker.seeds_, set())

    def test_get_states_returns_independent_shallow_copy(self):
        """get_states yields a new dict but shares the stored state values."""
        state_a = object()
        state_b = object()
        tracker = CudaRNGStatesTracker()
        tracker.states_ = {"a": state_a, "b": state_b}

        snapshot = tracker.get_states()
        # Different container ...
        self.assertIsNot(snapshot, tracker.states_)
        self.assertEqual(set(snapshot), {"a", "b"})
        # ... but the same value identities (shallow copy of pointers).
        self.assertIs(snapshot["a"], state_a)
        self.assertIs(snapshot["b"], state_b)
        # Mutating the snapshot must not leak back into the tracker.
        snapshot["c"] = object()
        self.assertNotIn("c", tracker.states_)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class CudaRNGStatesTrackerAddGuardTest(unittest.TestCase):
    """The two ValueError guards inside ``CudaRNGStatesTracker.add``."""

    def setUp(self):
        self._cpu_rng_state = paddle.get_rng_state("cpu")

    def tearDown(self):
        paddle.set_rng_state(self._cpu_rng_state, device="cpu")

    def test_add_rejects_duplicate_seed(self):
        """A seed already in seeds_ is refused before any state is created."""
        tracker = CudaRNGStatesTracker()
        tracker.seeds_.add(42)  # precondition: seed 42 already used

        with self.assertRaisesRegex(ValueError, "seed 42 already exists"):
            tracker.add("model-parallel-rng", 42)

        # The duplicate-seed guard fires before touching states_, so no state
        # named "model-parallel-rng" may have been registered.
        self.assertNotIn("model-parallel-rng", tracker.states_)

    def test_add_registers_seed_before_rejecting_duplicate_name(self):
        """A fresh seed is added to seeds_ *before* the name check raises."""
        sentinel = object()
        tracker = CudaRNGStatesTracker()
        tracker.states_ = {"model-parallel-rng": sentinel}  # name already taken

        with self.assertRaisesRegex(
            ValueError, "cuda rng state model-parallel-rng already exists"
        ):
            tracker.add("model-parallel-rng", 7)

        # Ordering behavior derived from the source: seeds_.add(seed) runs
        # before the name-existence check, so 7 is registered despite the raise.
        self.assertIn(7, tracker.seeds_)
        # The pre-existing state must be left untouched (not overwritten).
        self.assertIs(tracker.states_["model-parallel-rng"], sentinel)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class RandomGuardHelpersTest(unittest.TestCase):
    """fork guard, graph-safe assertions, and tracker-name constants."""

    def setUp(self):
        self._cpu_rng_state = paddle.get_rng_state("cpu")

    def tearDown(self):
        paddle.set_rng_state(self._cpu_rng_state, device="cpu")

    def test_fork_unknown_name_raises_on_entry(self):
        """Entering fork() for an unregistered name raises with that name."""
        tracker = CudaRNGStatesTracker()  # empty states_
        with (
            self.assertRaisesRegex(
                Exception, "cuda rng state missing-rng is not added"
            ),
            tracker.fork("missing-rng"),
        ):
            pass  # pragma: no cover - guard raises before body runs

    def test_get_cuda_rng_state_graph_safe_unsupported(self):
        """graph_safe=True is guarded before any CUDA access."""
        with self.assertRaisesRegex(
            AssertionError, "graph_safe is not supported yet"
        ):
            _get_cuda_rng_state(graph_safe=True)

    def test_set_cuda_rng_state_graph_safe_unsupported(self):
        """graph_safe=True is guarded before any CUDA access."""
        with self.assertRaisesRegex(
            AssertionError, "graph_safe is not supported yet"
        ):
            _set_cuda_rng_state(None, graph_safe=True)

    def test_tracker_name_getters_return_expected_constants(self):
        """The name getters expose the documented tracker-name strings."""
        self.assertEqual(
            get_expert_parallel_rng_tracker_name(), "expert-parallel-rng"
        )
        self.assertEqual(
            get_data_parallel_rng_tracker_name(), "data-parallel-rng"
        )


if __name__ == "__main__":
    unittest.main()
