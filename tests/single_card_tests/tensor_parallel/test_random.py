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

"""Behavior tests for ``paddlefleet.tensor_parallel.random``.

Repository module: "分布式训练" (tensor parallel) -- specifically the
CPU / control-flow half of the CUDA RNG tracker, NOT GPU kernel numerics:

  * ``CudaRNGStatesTracker`` seed/name bookkeeping and dedup ordering;
  * ``get_states`` shallow-copy semantics and the ``_is_initialized`` flag;
  * ``initialize_rng_tracker`` idempotency guard, ``force_reset`` re-creation
    and the inference-tracker branch (no-op ``add`` / null-context ``fork``);
  * the integer seed derivation of ``model_parallel_cuda_manual_seed``
    (data / tensor-model / expert seeds as a function of the ranks); and
  * ``fork`` switching to a seeded generator, advancing the stored state
    across successive forks, and restoring the outer generator.

Seed-derivation and fork-stream assertions drive the real CUDA generator and
are gated on a compiled-with-CUDA device (the single-card environment); the
pure bookkeeping / dispatch guards need only an importable paddle. Paddle is
not installed in the no-card environment, hence the honest skip. The global
RNG tracker is snapshotted and restored around every test that mutates it.
"""

import unittest

import numpy as np

try:
    import paddle

    import paddlefleet.tensor_parallel.random as tp_random
    from paddlefleet.tensor_parallel.random import (
        _DATA_PARALLEL_RNG_TRACKER_NAME,
        _EXPERT_PARALLEL_RNG_TRACKER_NAME,
        _MODEL_PARALLEL_RNG_TRACKER_NAME,
        CudaRNGStatesTracker,
        get_all_rng_states,
        get_cuda_rng_tracker,
        initialize_rng_tracker,
        model_parallel_cuda_manual_seed,
    )

    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

HAS_CUDA = (
    HAS_PADDLE
    and paddle.device.is_compiled_with_cuda()
    and paddle.device.cuda.device_count() > 0
)

_SKIP_NO_PADDLE = "paddle is not installed in this environment"
_SKIP_NO_CUDA = "requires a CUDA device (single-card environment)"


@unittest.skipUnless(HAS_PADDLE, _SKIP_NO_PADDLE)
class TestTrackerBookkeeping(unittest.TestCase):
    """Local CudaRNGStatesTracker state management (no CUDA generator)."""

    def test_new_tracker_is_empty_and_uninitialized(self):
        tracker = CudaRNGStatesTracker()
        self.assertFalse(tracker.is_initialized())
        self.assertEqual(tracker.states_, {})
        self.assertEqual(tracker.seeds_, set())

    def test_cudagraphable_construction_rejected(self):
        # Unsupported feature must fail loudly, not silently degrade.
        with self.assertRaises(AssertionError):
            CudaRNGStatesTracker(use_cudagraphable_rng=True)

    def test_fork_unknown_name_raises_before_touching_generator(self):
        tracker = CudaRNGStatesTracker()
        with self.assertRaises(Exception) as ctx:
            with tracker.fork("never-added"):
                pass
        message = str(ctx.exception)
        self.assertIn("never-added", message)
        self.assertIn("not added", message)

    def test_get_states_returns_independent_shallow_copy(self):
        tracker = CudaRNGStatesTracker()
        # Fixture precondition: two distinct sentinel state objects. The
        # assertions below target get_states' *copy* contract, not the values
        # themselves.
        state_a, state_b = object(), object()
        tracker.states_ = {"a": state_a, "b": state_b}
        snapshot = tracker.get_states()
        self.assertIsNot(snapshot, tracker.states_)  # a new container
        self.assertEqual(set(snapshot), {"a", "b"})
        self.assertIs(snapshot["a"], state_a)  # shallow: same value objects
        self.assertIs(snapshot["b"], state_b)
        snapshot["c"] = object()  # mutating the copy
        self.assertNotIn("c", tracker.states_)  # must not leak into original

    def test_set_states_replaces_and_marks_initialized(self):
        tracker = CudaRNGStatesTracker()
        self.assertFalse(tracker.is_initialized())
        sentinel = object()
        tracker.set_states({"m": sentinel})
        self.assertTrue(tracker.is_initialized())  # flag flipped by the logic
        self.assertIs(tracker.get_states()["m"], sentinel)


class _RngGlobalGuard(unittest.TestCase):
    """Snapshot and restore the module-global RNG tracker (antipattern #11).

    ``initialize_rng_tracker`` and ``model_parallel_cuda_manual_seed`` mutate
    process-wide module globals (and ``fork`` may set a class-level warning
    flag). Every subclass test saves the originals in ``setUp`` and restores
    them via ``addCleanup`` so a failure mid-test cannot leak state.
    """

    def setUp(self):
        super().setUp()
        self._orig_tracker = tp_random._CUDA_RNG_STATE_TRACKER
        self._orig_initialized = tp_random._CUDA_RNG_STATE_TRACKER_INITIALIZED
        self._had_warn = hasattr(
            CudaRNGStatesTracker, "_cpu_rng_warning_issued"
        )
        self.addCleanup(self._restore_globals)

    def _restore_globals(self):
        tp_random._CUDA_RNG_STATE_TRACKER = self._orig_tracker
        tp_random._CUDA_RNG_STATE_TRACKER_INITIALIZED = self._orig_initialized
        if not self._had_warn and hasattr(
            CudaRNGStatesTracker, "_cpu_rng_warning_issued"
        ):
            delattr(CudaRNGStatesTracker, "_cpu_rng_warning_issued")


@unittest.skipUnless(HAS_PADDLE, _SKIP_NO_PADDLE)
class TestInitializeRngTracker(_RngGlobalGuard):
    """initialize_rng_tracker dispatch: idempotency, reset, inference branch."""

    def test_creates_training_tracker_and_is_idempotent(self):
        initialize_rng_tracker(force_reset=True)
        first = get_cuda_rng_tracker()
        self.assertIsInstance(first, CudaRNGStatesTracker)
        self.assertFalse(first.is_inference_rng_tracker)
        # A second call without force_reset must be a no-op returning the same
        # object -- reconstructing it would silently drop already-added states.
        initialize_rng_tracker()
        self.assertIs(get_cuda_rng_tracker(), first)

    def test_force_reset_replaces_instance(self):
        initialize_rng_tracker(force_reset=True)
        first = get_cuda_rng_tracker()
        initialize_rng_tracker(force_reset=True)
        second = get_cuda_rng_tracker()
        self.assertIsInstance(second, CudaRNGStatesTracker)
        self.assertIsNot(second, first)  # force_reset built a fresh tracker

    def test_inference_tracker_add_noop_and_fork_nullcontext(self):
        initialize_rng_tracker(inference_rng_tracker=True, force_reset=True)
        tracker = get_cuda_rng_tracker()
        self.assertTrue(tracker.is_inference_rng_tracker)
        # Inference add/set_states are deliberately no-ops: registering a state
        # leaves the tracker empty (the training tracker would store it).
        tracker.add("ignored", 123)
        tracker.set_states({"ignored": object()})
        self.assertEqual(tracker.states_, {})
        self.assertEqual(tracker.seeds_, set())
        # ...and fork yields a null context that never raises, even for a name
        # that was never registered (the training tracker raises for that).
        with tracker.fork("anything-goes"):
            pass


@unittest.skipUnless(HAS_PADDLE, _SKIP_NO_PADDLE)
class TestUnsupportedFeatureGuards(unittest.TestCase):
    """Unsupported knobs must raise, never silently no-op."""

    def test_initialize_rejects_cudagraphable(self):
        with self.assertRaises(AssertionError):
            initialize_rng_tracker(use_cudagraphable_rng=True)

    def test_initialize_rejects_te_tracker(self):
        with self.assertRaises(AssertionError):
            initialize_rng_tracker(use_te_rng_tracker=True)

    def test_manual_seed_rejects_te_tracker(self):
        with self.assertRaises(AssertionError):
            model_parallel_cuda_manual_seed(42, te_rng_tracker=True)

    def test_manual_seed_rejects_cudagraphable(self):
        with self.assertRaises(AssertionError):
            model_parallel_cuda_manual_seed(42, use_cudagraphable_rng=True)


@unittest.skipUnless(HAS_CUDA, _SKIP_NO_CUDA)
class TestModelParallelSeedDerivation(_RngGlobalGuard):
    """model_parallel_cuda_manual_seed: hand-derived per-rank seed arithmetic.

    The registered seeds are pure integer functions of (seed, tp, ep, etp):
        data   = seed
        model  = seed + 2718 + tp_rank
        expert = seed + 1024 + 100 * ep_rank + etp_rank
    ``add`` touches the CUDA generator (hence the CUDA gate), but the values
    asserted here are the arithmetic, so this catches swapped rank offsets.
    """

    def _register(self, **kwargs):
        model_parallel_cuda_manual_seed(**kwargs)
        return get_cuda_rng_tracker()

    def test_registers_three_hand_derived_seeds(self):
        tracker = self._register(seed=42, tp_rank=1, ep_rank=2, etp_rank=3)
        data_seed = 42
        model_seed = 42 + 2718 + 1  # 2761
        expert_seed = 42 + 1024 + 100 * 2 + 3  # 1269
        self.assertEqual(tracker.seeds_, {data_seed, model_seed, expert_seed})
        self.assertEqual(
            set(tracker.states_),
            {
                _DATA_PARALLEL_RNG_TRACKER_NAME,
                _MODEL_PARALLEL_RNG_TRACKER_NAME,
                _EXPERT_PARALLEL_RNG_TRACKER_NAME,
            },
        )
        # get_all_rng_states exposes the very same live state mapping.
        self.assertIs(get_all_rng_states(), tracker.states_)

    def test_only_tensor_model_seed_tracks_tp_rank(self):
        # Each call resets the shared tracker, so snapshot seeds before re-call.
        tracker = self._register(seed=100, tp_rank=0, ep_rank=0, etp_rank=0)
        seeds_rank0 = set(tracker.seeds_)
        tracker = self._register(seed=100, tp_rank=5, ep_rank=0, etp_rank=0)
        seeds_rank5 = set(tracker.seeds_)

        self.assertIn(100, seeds_rank0)  # data seed is rank-independent
        self.assertIn(100, seeds_rank5)
        self.assertIn(100 + 2718 + 0, seeds_rank0)  # 2818
        self.assertIn(100 + 2718 + 5, seeds_rank5)  # 2823
        self.assertNotIn(100 + 2718 + 5, seeds_rank0)  # tp delta really applied
        expert_seed = 100 + 1024  # ep=etp=0 -> 1124
        self.assertIn(expert_seed, seeds_rank0)
        self.assertIn(expert_seed, seeds_rank5)  # expert seed ignores tp_rank

    def test_expert_seed_tracks_ep_and_etp_ranks(self):
        tracker = self._register(seed=7, tp_rank=0, ep_rank=3, etp_rank=4)
        expert_seed = 7 + 1024 + 100 * 3 + 4  # 1335
        self.assertIn(expert_seed, tracker.seeds_)
        # data & tensor-model seeds do not absorb the expert-rank offsets.
        self.assertIn(7, tracker.seeds_)
        self.assertIn(7 + 2718, tracker.seeds_)
        self.assertNotIn(expert_seed, {7, 7 + 2718})


@unittest.skipUnless(HAS_CUDA, _SKIP_NO_CUDA)
class TestForkStreamBehavior(_RngGlobalGuard):
    """fork: switch to the seeded generator, advance it, restore the outer."""

    def setUp(self):
        super().setUp()
        paddle.set_device("gpu")

    def test_fork_reproduces_seeded_stream_and_restores_outer(self):
        seed = 20260916
        # Independent reference: seeding the CUDA generator directly and
        # drawing twice yields the ordered stream a correct fork must
        # reproduce across two successive fork() calls.
        paddle.cuda.manual_seed(seed)
        ref_first = paddle.rand([8]).numpy().copy()
        ref_second = paddle.rand([8]).numpy().copy()

        tracker = CudaRNGStatesTracker()
        outer_seed = seed + 12345
        # Establish the OUTER generator's next draw, then rewind it so the real
        # run starts from the same outer position.
        paddle.cuda.manual_seed(outer_seed)
        outer_next = paddle.rand([8]).numpy().copy()
        paddle.cuda.manual_seed(outer_seed)

        # add() captures + restores the outer state and stores the seeded one.
        tracker.add("s", seed)

        with tracker.fork("s"):
            got_first = paddle.rand([8]).numpy().copy()
        with tracker.fork("s"):
            got_second = paddle.rand([8]).numpy().copy()

        # Outer generator resumes exactly where it was before add()/fork().
        outer_after = paddle.rand([8]).numpy().copy()

        np.testing.assert_allclose(got_first, ref_first, rtol=1e-6, atol=0)
        # The fork finally-block stores the advanced state, so the 2nd fork
        # continues the stream rather than restarting it.
        np.testing.assert_allclose(got_second, ref_second, rtol=1e-6, atol=0)
        self.assertFalse(np.allclose(got_second, got_first))
        np.testing.assert_allclose(outer_after, outer_next, rtol=1e-6, atol=0)


if __name__ == "__main__":
    unittest.main()
