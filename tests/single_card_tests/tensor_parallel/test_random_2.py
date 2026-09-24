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

"""CPU behavior tests for ``paddlefleet.tensor_parallel.random`` (part _2).

Repository module: "分布式训练" (tensor parallel) RNG infrastructure. This
file (_2) deliberately targets a different slice of the module than the
base file: the *module-global tracker lifecycle* and the *seed derivation*,
rather than the ``CudaRNGStatesTracker.add`` / ``fork`` mechanics or the
name getters.

Functions/branches exercised here:

  * ``initialize_rng_tracker`` -- fresh construction, idempotent early
    return, ``force_reset`` replacement, and the ``inference_rng_tracker``
    branch that installs an ``InferenceCudaRNGStatesTracker`` whose
    ``add`` / ``set_states`` are no-ops and whose ``fork`` yields a
    ``nullcontext``;
  * the ``use_te_rng_tracker`` / ``use_cudagraphable_rng`` assertion guards
    in both ``initialize_rng_tracker`` and ``get_cuda_rng_tracker``;
  * ``get_cuda_rng_tracker`` returning the installed module-global object;
  * ``get_all_rng_states`` -- the uninitialized assertion, the
    ``CudaRNGStatesTracker`` branch (returns the live ``states_`` dict), and
    the non-tracker fallback branch (returns ``{}``);
  * the seed arithmetic in ``model_parallel_cuda_manual_seed`` -- namely
    data-parallel = seed, tensor-model-parallel = seed + 2718 + tp_rank,
    expert = seed + 1024 + 100*ep_rank + etp_rank -- and the mapping of each
    derived seed to its named tracker slot.

The genuine ``paddle.cuda`` RNG collaborators (``manual_seed`` /
``get_rng_state`` / the ``set_rng_state`` wrapper) are the only mocked
pieces, and only in the seed-derivation test where they would otherwise
require a GPU; the seed arithmetic and the name->seed dispatch under test
run for real. All module-global mutations are snapshotted and restored in
cleanup (antipattern #11). paddle is imported at load time, so the whole
file skips honestly when paddle is unavailable rather than fake-passing.
"""

import contextlib
import random as _pyrandom
import unittest
from unittest import mock

try:
    import paddle  # noqa: F401

    from paddlefleet.tensor_parallel import random as random_mod
    from paddlefleet.tensor_parallel.random import (
        CudaRNGStatesTracker,
        get_all_rng_states,
        get_cuda_rng_tracker,
        initialize_rng_tracker,
        model_parallel_cuda_manual_seed,
    )

    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False
    random_mod = None

_SKIP_REASON = "paddle is not installed in this environment"

# The name constants the production module assigns each derived seed to.
_DATA_NAME = "data-parallel-rng"
_MODEL_NAME = "model-parallel-rng"
_EXPERT_NAME = "expert-parallel-rng"


class _RngGlobalsMixin(unittest.TestCase):
    """Snapshot/restore the module-global tracker so tests do not leak.

    ``initialize_rng_tracker`` and friends mutate module-level globals
    (``_CUDA_RNG_STATE_TRACKER`` / ``_CUDA_RNG_STATE_TRACKER_INITIALIZED``).
    Each test starts from a clean, uninitialized tracker and the original
    values are restored on cleanup, including on assertion failure.
    """

    def setUp(self):
        self._orig_tracker = random_mod._CUDA_RNG_STATE_TRACKER
        self._orig_initialized = random_mod._CUDA_RNG_STATE_TRACKER_INITIALIZED
        self.addCleanup(self._restore_globals)
        # Guard the CPU python RNG too; nothing here should advance it, but
        # restore it regardless (antipattern #11).
        py_state = _pyrandom.getstate()
        self.addCleanup(_pyrandom.setstate, py_state)
        random_mod._CUDA_RNG_STATE_TRACKER = None
        random_mod._CUDA_RNG_STATE_TRACKER_INITIALIZED = False

    def _restore_globals(self):
        random_mod._CUDA_RNG_STATE_TRACKER = self._orig_tracker
        random_mod._CUDA_RNG_STATE_TRACKER_INITIALIZED = self._orig_initialized


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestInitializeRngTracker(_RngGlobalsMixin):
    """Lifecycle of the module-global CUDA RNG tracker."""

    def test_fresh_init_installs_training_tracker(self):
        initialize_rng_tracker(force_reset=True)
        tracker = random_mod._CUDA_RNG_STATE_TRACKER
        self.assertTrue(random_mod._CUDA_RNG_STATE_TRACKER_INITIALIZED)
        # Exactly the training tracker, not the inference subclass.
        self.assertIs(type(tracker), CudaRNGStatesTracker)
        self.assertFalse(tracker.is_inference_rng_tracker)

    def test_second_call_is_idempotent(self):
        initialize_rng_tracker(force_reset=True)
        first = random_mod._CUDA_RNG_STATE_TRACKER
        # No force_reset: the already-initialized guard must return early and
        # leave the exact same object in place.
        initialize_rng_tracker()
        self.assertIs(random_mod._CUDA_RNG_STATE_TRACKER, first)

    def test_force_reset_replaces_object(self):
        initialize_rng_tracker(force_reset=True)
        first = random_mod._CUDA_RNG_STATE_TRACKER
        initialize_rng_tracker(force_reset=True)
        second = random_mod._CUDA_RNG_STATE_TRACKER
        self.assertIsNot(second, first)
        self.assertIs(type(second), CudaRNGStatesTracker)

    def test_te_rng_tracker_guard(self):
        with self.assertRaises(AssertionError):
            initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)

    def test_cudagraphable_guard(self):
        with self.assertRaises(AssertionError):
            initialize_rng_tracker(use_cudagraphable_rng=True, force_reset=True)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestInferenceRngTracker(_RngGlobalsMixin):
    """The inference-tracker branch overrides add/set_states/fork."""

    def _make_inference_tracker(self):
        initialize_rng_tracker(inference_rng_tracker=True, force_reset=True)
        return random_mod._CUDA_RNG_STATE_TRACKER

    def test_installed_type_and_flag(self):
        tracker = self._make_inference_tracker()
        self.assertEqual(
            type(tracker).__name__, "InferenceCudaRNGStatesTracker"
        )
        # Still a CudaRNGStatesTracker subclass, but flagged for inference.
        self.assertIsInstance(tracker, CudaRNGStatesTracker)
        self.assertTrue(tracker.is_inference_rng_tracker)

    def test_add_is_noop(self):
        tracker = self._make_inference_tracker()
        # Training tracker would record the seed/name and reject duplicates;
        # the inference override records nothing and never raises.
        tracker.add("dup", 7)
        tracker.add("dup", 7)
        self.assertEqual(tracker.states_, {})
        self.assertEqual(tracker.seeds_, set())

    def test_set_states_is_noop(self):
        tracker = self._make_inference_tracker()
        tracker.set_states({"a": object()})
        # Override does not store or mark initialized (base class would).
        self.assertEqual(tracker.states_, {})
        self.assertFalse(tracker.is_initialized())

    def test_fork_yields_nullcontext_without_added_state(self):
        tracker = self._make_inference_tracker()
        ctx = tracker.fork("never-added")
        # Base fork would raise for an unknown name; inference fork returns a
        # do-nothing context manager that enters and exits cleanly.
        self.assertIsInstance(ctx, contextlib.nullcontext)
        entered = False
        with ctx:
            entered = True
        self.assertTrue(entered)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestGetCudaRngTracker(_RngGlobalsMixin):
    """get_cuda_rng_tracker returns the installed global and guards flags."""

    def test_returns_installed_global(self):
        returned = get_cuda_rng_tracker()
        self.assertIs(returned, random_mod._CUDA_RNG_STATE_TRACKER)
        self.assertIsInstance(returned, CudaRNGStatesTracker)
        # Idempotent: same object on the second call.
        self.assertIs(get_cuda_rng_tracker(), returned)

    def test_te_rng_tracker_guard(self):
        with self.assertRaises(AssertionError):
            get_cuda_rng_tracker(use_te_rng_tracker=True)

    def test_cudagraphable_guard(self):
        with self.assertRaises(AssertionError):
            get_cuda_rng_tracker(use_cudagraphable_rng=True)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestGetAllRngStates(_RngGlobalsMixin):
    """get_all_rng_states: assertion + tracker branch + fallback branch."""

    def test_asserts_when_uninitialized(self):
        # setUp left the module uninitialized.
        self.assertFalse(random_mod._CUDA_RNG_STATE_TRACKER_INITIALIZED)
        with self.assertRaises(AssertionError):
            get_all_rng_states()

    def test_returns_live_states_dict_for_tracker(self):
        tracker = CudaRNGStatesTracker()
        sentinel = {"model-parallel-rng": object()}
        tracker.states_ = sentinel
        random_mod._CUDA_RNG_STATE_TRACKER = tracker
        random_mod._CUDA_RNG_STATE_TRACKER_INITIALIZED = True
        result = get_all_rng_states()
        # Must hand back the tracker's live states_ object, not a copy or {}.
        self.assertIs(result, sentinel)

    def test_returns_empty_for_non_tracker(self):
        # Initialized flag set but the installed object is not a
        # CudaRNGStatesTracker -> the else branch returns a fresh empty dict.
        random_mod._CUDA_RNG_STATE_TRACKER = object()
        random_mod._CUDA_RNG_STATE_TRACKER_INITIALIZED = True
        self.assertEqual(get_all_rng_states(), {})


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestModelParallelCudaManualSeed(_RngGlobalsMixin):
    """Seed derivation and name->seed dispatch of the manual-seed entry.

    The genuine per-device CUDA RNG state get/set are mocked (they need a
    GPU and are not the logic under test). The seed arithmetic and the
    assignment of each derived seed to its named slot run for real via the
    real ``CudaRNGStatesTracker.add`` bookkeeping.
    """

    def test_seed_arithmetic_and_named_dispatch(self):
        seed, tp_rank, ep_rank, etp_rank = 100, 3, 2, 1

        # Hand-derived expected seeds (from the module's documented formula).
        exp_data = seed  # 100
        exp_model = seed + 2718 + tp_rank  # 2821
        exp_expert = seed + 1024 + 100 * ep_rank + etp_rank  # 1325
        self.assertEqual((exp_data, exp_model, exp_expert), (100, 2821, 1325))

        # Make the mocked device RNG state echo the most recently set seed so
        # that each stored state is traceable back to the seed used for it.
        last = {"seed": None}

        def fake_manual_seed(value):
            last["seed"] = value

        def fake_get_rng_state(*args, **kwargs):
            return ("cuda_state", last["seed"])

        with (
            mock.patch("paddle.cuda.manual_seed", side_effect=fake_manual_seed),
            mock.patch(
                "paddle.cuda.get_rng_state", side_effect=fake_get_rng_state
            ),
            mock.patch.object(random_mod, "_set_cuda_rng_state"),
        ):
            model_parallel_cuda_manual_seed(
                seed,
                tp_rank=tp_rank,
                ep_rank=ep_rank,
                etp_rank=etp_rank,
            )

        tracker = random_mod._CUDA_RNG_STATE_TRACKER
        # All three derived seeds were booked exactly once.
        self.assertEqual(tracker.seeds_, {exp_data, exp_model, exp_expert})
        # Each named slot received the state produced right after seeding it
        # with its own derived seed -- proving name<->seed dispatch, not just
        # that three seeds exist.
        self.assertEqual(
            set(tracker.states_), {_DATA_NAME, _MODEL_NAME, _EXPERT_NAME}
        )
        self.assertEqual(tracker.states_[_DATA_NAME], ("cuda_state", exp_data))
        self.assertEqual(
            tracker.states_[_MODEL_NAME], ("cuda_state", exp_model)
        )
        self.assertEqual(
            tracker.states_[_EXPERT_NAME], ("cuda_state", exp_expert)
        )

    def test_distinct_ranks_shift_only_their_seed(self):
        # tp_rank affects only the model-parallel seed; ep/etp affect only the
        # expert seed; the data-parallel seed is always the raw seed.
        last = {"seed": None}

        def run(seed, tp_rank, ep_rank, etp_rank):
            def fake_manual_seed(value):
                last["seed"] = value

            def fake_get_rng_state(*args, **kwargs):
                return ("cuda_state", last["seed"])

            random_mod._CUDA_RNG_STATE_TRACKER = None
            random_mod._CUDA_RNG_STATE_TRACKER_INITIALIZED = False
            with (
                mock.patch(
                    "paddle.cuda.manual_seed", side_effect=fake_manual_seed
                ),
                mock.patch(
                    "paddle.cuda.get_rng_state", side_effect=fake_get_rng_state
                ),
                mock.patch.object(random_mod, "_set_cuda_rng_state"),
            ):
                model_parallel_cuda_manual_seed(
                    seed,
                    tp_rank=tp_rank,
                    ep_rank=ep_rank,
                    etp_rank=etp_rank,
                )
            tr = random_mod._CUDA_RNG_STATE_TRACKER
            return (
                tr.states_[_DATA_NAME][1],
                tr.states_[_MODEL_NAME][1],
                tr.states_[_EXPERT_NAME][1],
            )

        base = run(50, tp_rank=0, ep_rank=0, etp_rank=0)
        self.assertEqual(base, (50, 50 + 2718, 50 + 1024))

        # Bumping tp_rank by 1 shifts only the model-parallel seed by 1.
        tp_bumped = run(50, tp_rank=1, ep_rank=0, etp_rank=0)
        self.assertEqual(tp_bumped, (50, 50 + 2718 + 1, 50 + 1024))

        # Bumping ep_rank by 1 shifts only the expert seed by 100.
        ep_bumped = run(50, tp_rank=0, ep_rank=1, etp_rank=0)
        self.assertEqual(ep_bumped, (50, 50 + 2718, 50 + 1024 + 100))

        # Bumping etp_rank by 1 shifts only the expert seed by 1.
        etp_bumped = run(50, tp_rank=0, ep_rank=0, etp_rank=1)
        self.assertEqual(etp_bumped, (50, 50 + 2718, 50 + 1024 + 1))


if __name__ == "__main__":
    unittest.main()
