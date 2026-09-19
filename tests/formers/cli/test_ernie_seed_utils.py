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

import random
import unittest

import numpy as np

# seed_utils.py imports paddle, paddle.distributed.fleet and get_rng_state_tracker
# at module import time (and pulls in paddlefleet package init -> paddle), so the
# whole module is unimportable without paddle. Guard the import honestly; the
# local CI box has no paddle installed and will skip these tests. Only genuine
# import failures are treated as "no dependency"; other errors propagate.
try:
    import paddle
    from paddle.distributed.fleet import fleet

    from paddlefleet.cli.train.ernie_pretrain.src.utils.seed_utils import (
        set_seed,
    )

    HAS_PADDLE = True
except (ImportError, ModuleNotFoundError):
    HAS_PADDLE = False

_SKIP_REASON = "importing paddlefleet seed_utils requires paddle; not installed"


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestSetSeed(unittest.TestCase):
    """Behavior of set_seed on a single, non-hybrid-parallel process.

    set_seed(seed) derives three quantities from the parallel ranks and seeds
    three RNGs:

      * Python ``random`` and NumPy are seeded with ``seed + dp_rank``.
      * The default Paddle generator is seeded with ``global_seed``.

    On a fresh process with no ``fleet.init`` and no distributed group the ranks
    collapse to mp=0/1, pp=0/1, dp_rank=0, dp_size=1, sharding_rank=0 and
    ``paddle.distributed.get_world_size()`` is 1. Under those conditions the
    expected values reduce to, for a given ``seed`` (all derived by hand from the
    formulas in the module, NOT read back from the function):

        effective random/numpy seed = seed + 0          = seed
        seed_offset                 = seed + 1024 + 1    = seed + 1025
        global_seed                 = seed_offset + 0    = seed + 1025
        local_seed                  = (seed_offset + 1) + 0 = seed + 1026

    Because ``get_rng_state_tracker`` is a process-wide singleton and
    ``tracker.add`` refuses to register the same state name twice, set_seed is
    designed to run once per process. This test therefore calls it exactly once
    and inspects every observable RNG afterwards, rather than calling it per
    assertion.
    """

    # A single, fixed seed so the hand-derived expectations below are concrete.
    SEED = 42

    @classmethod
    def setUpClass(cls):
        # The single-process derivation only holds when no hybrid communicate
        # group has been initialized. If a prior test in the same process already
        # ran fleet.init, the rank arithmetic differs; skip honestly instead of
        # asserting wrong expectations.
        if hasattr(fleet, "_hcg"):
            raise unittest.SkipTest(
                "fleet hybrid communicate group already initialized; the "
                "single-process seed derivation is not applicable"
            )

        cls.set_seed_error = None
        try:
            set_seed(cls.SEED)
        except Exception as exc:
            # Do not swallow: surface it as an explicit failing assertion in
            # test_set_seed_runs_without_error so the traceback is visible.
            cls.set_seed_error = exc
            return

        # Capture the "actual" draws immediately, before any reference reseeding
        # touches these generators. random / numpy / paddle default generators
        # are independent, so the capture order among them does not matter.
        cls.actual_paddle = paddle.rand([5]).numpy().tolist()
        cls.actual_python = [random.random() for _ in range(5)]
        cls.actual_numpy = np.random.rand(5).tolist()

    def test_set_seed_runs_without_error(self):
        # A genuine end-to-end invocation must not raise. If setUpClass caught an
        # exception from the real production call, fail loudly with it.
        if self.set_seed_error is not None:
            raise self.set_seed_error

    def test_python_random_seeded_with_dp_adjusted_seed(self):
        # dp_rank == 0, so the effective seed is exactly SEED. An independent
        # reseed with SEED must reproduce the same stream set_seed produced.
        self.assertIsNone(self.set_seed_error)
        random.seed(self.SEED)
        expected = [random.random() for _ in range(5)]
        self.assertEqual(self.actual_python, expected)

        # Guard: a different seed must NOT match, otherwise the check above would
        # pass for any seeding at all.
        random.seed(self.SEED + 1)
        wrong = [random.random() for _ in range(5)]
        self.assertNotEqual(self.actual_python, wrong)

    def test_numpy_seeded_with_dp_adjusted_seed(self):
        # NumPy must be seeded with the same effective value (SEED) as Python's
        # random. Independent reseed reproduces the identical draws.
        self.assertIsNone(self.set_seed_error)
        np.random.seed(self.SEED)
        expected = np.random.rand(5).tolist()
        self.assertEqual(self.actual_numpy, expected)

        np.random.seed(self.SEED + 7)
        wrong = np.random.rand(5).tolist()
        self.assertNotEqual(self.actual_numpy, wrong)

    def test_paddle_default_generator_seeded_with_global_seed(self):
        # paddle.seed(global_seed) is the last generator-affecting call in
        # set_seed, so the default generator is left at global_seed = SEED + 1025.
        # Reseeding to that hand-derived value must reproduce the captured draws.
        self.assertIsNone(self.set_seed_error)
        expected_global_seed = self.SEED + 1025  # 1024 constant + world_size(1)
        paddle.seed(expected_global_seed)
        expected = paddle.rand([5]).numpy().tolist()
        self.assertEqual(self.actual_paddle, expected)

        # A generator seeded with local_seed (SEED + 1026) or with a bare SEED
        # must differ, proving the specific +1025 offset is what got applied.
        paddle.seed(self.SEED + 1026)
        wrong_local = paddle.rand([5]).numpy().tolist()
        self.assertNotEqual(self.actual_paddle, wrong_local)

        paddle.seed(self.SEED)
        wrong_bare = paddle.rand([5]).numpy().tolist()
        self.assertNotEqual(self.actual_paddle, wrong_bare)

    def test_python_and_numpy_receive_same_effective_seed(self):
        # Cross-check the module's contract that random and numpy share the same
        # (seed + dp_rank) value. If set_seed accidentally fed numpy a different
        # seed, this dual reconstruction would fail on one of the two streams.
        self.assertIsNone(self.set_seed_error)
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        self.assertEqual(
            self.actual_python, [random.random() for _ in range(5)]
        )
        self.assertEqual(self.actual_numpy, np.random.rand(5).tolist())

    def test_tracker_registers_named_seed_states(self):
        # Supplementary (not sole) check: set_seed registers three named RNG
        # states on the shared tracker. This only asserts presence of the names;
        # the numeric correctness of the seeds is covered by the draw-based
        # tests above.
        self.assertIsNone(self.set_seed_error)
        from paddle.distributed.fleet.meta_parallel import (
            get_rng_state_tracker,
        )

        states = get_rng_state_tracker().states_
        self.assertIn("global_seed", states)
        self.assertIn("local_seed", states)
        self.assertIn("model_parallel_rng", states)

    # NOTE: The per-rank differentiation branch (seed += dp_rank, and the
    # dp/pp/sharding terms in global_seed/local_seed) cannot be exercised in a
    # single process. Per the distributed-training testing rules, verifying that
    # distinct ranks receive distinct seeds requires a real multi-process group;
    # that is out of scope for this no-card, single-process test.


if __name__ == "__main__":
    unittest.main()
