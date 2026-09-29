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

"""CPU behavior tests for ``paddlefleet.tensor_parallel.random`` (part 6).

Repository module: "分布式训练" (tensor-parallel RNG plumbing). This file
deliberately targets branches that the tracker-focused siblings do NOT cover:

  * ``model_parallel_cuda_manual_seed`` -- the actual *seed-derivation
    arithmetic* and dispatch order. The three tracked seeds are hand-derived
    from the documented formulas (data = seed; tensor-model = seed + 2718 +
    tp_rank; expert = seed + 1024 + 100*ep_rank + etp_rank) and checked against
    the exact ``(name, seed)`` pairs the function feeds to the tracker, in
    order, including the default-state ``paddle.cuda.manual_seed`` call. The
    genuine GPU seeding collaborators (``paddle.cuda.manual_seed`` and the
    CUDA-state-capturing tracker) are substituted by capturing fakes so the
    real arithmetic and orchestration under test run unchanged; the true GPU
    RNG numerics are NOT claimed here (single-card work). Explicit ranks are
    passed so no ``parallel_state`` initialization is required.

  * ``enable_share_grad_holder`` -- the ``if not old_value`` guard on BOTH the
    entry set and the ``finally`` reset. The already-enabled branch (flag True
    on entry must be left True, never clobbered to False) and the
    exception-path restore (``finally`` runs even when the body raises) are
    exercised; these are distinct from the disabled->enabled->disabled happy
    path.

Paddle is not installed in this no-card environment, so the suite skips
honestly rather than fake-passing. Every test snapshots and restores the
process RNG state, and the flag test restores the framework flag, so no global
state leaks to peers (antipattern #11).
"""

import unittest

try:
    import paddle

    from paddlefleet.tensor_parallel import random as tp_random
    from paddlefleet.tensor_parallel.random import (
        enable_share_grad_holder,
        model_parallel_cuda_manual_seed,
    )

    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

from unittest import mock

_SKIP_REASON = "paddle is not installed in this environment"
_SHARE_GRAD_FLAG = "FLAGS_share_tensor_for_grad_tensor_holder"


class _CapturingTracker:
    """Stand-in for the CUDA-state tracker that records reset/add events.

    The real tracker's ``add`` captures a live GPU RNG state, which needs an
    accelerator; here we only need to observe *what* the code under test asks
    it to register, so this fake logs into a shared event list instead.
    """

    def __init__(self, events):
        self._events = events

    def reset(self):
        self._events.append(("reset",))

    def add(self, name, seed):
        self._events.append(("add", name, seed))


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class ModelParallelCudaManualSeedTest(unittest.TestCase):
    """Seed derivation and dispatch order of model_parallel_cuda_manual_seed."""

    def setUp(self):
        # Snapshot process RNG so any incidental mutation is undone for peers
        # (antipattern #11). The tracker globals are handled by patch below.
        self._rng_state = paddle.get_rng_state()

    def tearDown(self):
        paddle.set_rng_state(self._rng_state)

    def test_derives_three_seeds_and_dispatches_in_order(self):
        """The default/tensor-model/expert seeds match the documented formulas.

        Ranks are chosen distinct and non-degenerate (tp=3, ep=2, etp=1) so a
        swap of any rank or a wrong constant produces a different number:
          data          = 100
          tensor-model  = 100 + 2718 + 3 = 2821
          expert        = 100 + 1024 + 100*2 + 1 = 1325
        Swapping ep/etp would yield 1226, and a wrong offset would move 2821,
        so the exact pairs pin the arithmetic rather than mere presence.
        """
        events = []
        fake_tracker = _CapturingTracker(events)

        def _record_manual_seed(value):
            events.append(("manual_seed", value))

        with (
            mock.patch.object(
                tp_random, "_CUDA_RNG_STATE_TRACKER", fake_tracker
            ),
            mock.patch.object(
                tp_random, "_CUDA_RNG_STATE_TRACKER_INITIALIZED", True
            ),
            mock.patch.object(
                paddle.cuda, "manual_seed", side_effect=_record_manual_seed
            ),
        ):
            model_parallel_cuda_manual_seed(
                100, tp_rank=3, ep_rank=2, etp_rank=1
            )

        # Full ordered contract: reset first, then default-state GPU seeding
        # with the data-parallel seed, then the three tracked (name, seed)
        # registrations in the source's order.
        self.assertEqual(
            events,
            [
                ("reset",),
                ("manual_seed", 100),
                ("add", "data-parallel-rng", 100),
                ("add", "model-parallel-rng", 2821),
                ("add", "expert-parallel-rng", 1325),
            ],
        )

    def test_expert_seed_isolates_ep_and_etp_contributions(self):
        """ep_rank scales by 100 while etp_rank adds 1: swapping them differs.

        A second point with ep/etp exchanged (ep=1, etp=2) must yield a
        different expert seed (100 + 1024 + 100 + 2 = 1226, not 1325), proving
        the 100*ep_rank + etp_rank weighting is really consumed.
        """
        events = []
        fake_tracker = _CapturingTracker(events)

        with (
            mock.patch.object(
                tp_random, "_CUDA_RNG_STATE_TRACKER", fake_tracker
            ),
            mock.patch.object(
                tp_random, "_CUDA_RNG_STATE_TRACKER_INITIALIZED", True
            ),
            mock.patch.object(paddle.cuda, "manual_seed"),
        ):
            model_parallel_cuda_manual_seed(
                100, tp_rank=3, ep_rank=1, etp_rank=2
            )

        expert_events = [
            e for e in events if e[:2] == ("add", "expert-parallel-rng")
        ]
        self.assertEqual(expert_events, [("add", "expert-parallel-rng", 1226)])


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class EnableShareGradHolderGuardTest(unittest.TestCase):
    """The ``if not old_value`` guard on both set and reset of the flag."""

    def setUp(self):
        self._rng_state = paddle.get_rng_state()
        # The context manager mutates a process-wide framework flag; snapshot
        # it so the original value is restored regardless of test outcome.
        self._orig_flag = paddle.get_flags([_SHARE_GRAD_FLAG])[_SHARE_GRAD_FLAG]

    def tearDown(self):
        paddle.set_flags({_SHARE_GRAD_FLAG: self._orig_flag})
        paddle.set_rng_state(self._rng_state)

    def _flag(self):
        return paddle.get_flags([_SHARE_GRAD_FLAG])[_SHARE_GRAD_FLAG]

    def test_enables_when_disabled_and_restores_off(self):
        """Disabled on entry: flag is True inside, back to False on exit."""
        paddle.set_flags({_SHARE_GRAD_FLAG: False})
        self.assertFalse(self._flag())
        with enable_share_grad_holder():
            self.assertTrue(self._flag())
        self.assertFalse(self._flag())

    def test_already_enabled_flag_is_left_enabled(self):
        """Enabled on entry: the CM must NOT reset an externally-set flag.

        Because ``old_value`` is truthy, neither the entry set nor the finally
        reset runs, so the flag stays True throughout and after. A production
        bug that unconditionally reset to False would flip it here.
        """
        paddle.set_flags({_SHARE_GRAD_FLAG: True})
        self.assertTrue(self._flag())
        with enable_share_grad_holder():
            self.assertTrue(self._flag())
        self.assertTrue(self._flag())

    def test_restores_disabled_state_when_body_raises(self):
        """Disabled on entry + exception inside: finally restores to False."""
        paddle.set_flags({_SHARE_GRAD_FLAG: False})

        class _Boom(RuntimeError):
            pass

        with self.assertRaises(_Boom), enable_share_grad_holder():
            self.assertTrue(self._flag())
            raise _Boom("body failure")

        # The finally clause must have run despite the exception.
        self.assertFalse(self._flag())


if __name__ == "__main__":
    unittest.main()
