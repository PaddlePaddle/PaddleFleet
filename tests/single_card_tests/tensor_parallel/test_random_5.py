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

"""CPU behavior tests for ``paddlefleet.tensor_parallel.random`` (part 5).

Repository module: "分布式训练" (tensor-parallel RNG / recompute plumbing).
This file deliberately targets branches that the sibling coverage files do NOT
touch. Parts 3/4 exercise ``CudaRNGStatesTracker`` bookkeeping, the graph-safe
guards, the name getters, and the module-global tracker orchestration
(``initialize_rng_tracker`` / ``get_cuda_rng_tracker`` / ``get_all_rng_states``
/ inference tracker). Here we cover two distinct, CPU-runnable pieces of
behavior instead:

  * ``enable_share_grad_holder`` -- the context manager that temporarily flips
    the paddle flag ``FLAGS_share_tensor_for_grad_tensor_holder``. Its two
    branches are hand-derived from the "restore only what we changed" contract:
    when the flag starts disabled it is enabled for the body and reset to
    disabled on exit (including on exception); when the flag is *already*
    enabled it is left enabled on exit rather than being blindly cleared.

  * ``RecomputeWithoutOutput.run_recompute_now`` -- the documented no-op guard
    (``_recompute`` returns immediately when ``ctx is None``). On a freshly
    constructed instance this must be a side-effect-free no-op returning None.

The seeded ``add`` / ``fork`` CUDA value paths, ``_fork_rng``, the real
recompute replay (``recompute`` / ``_recompute`` with a live ctx), and
``model_parallel_cuda_manual_seed`` all require an actual accelerator and are
intentionally NOT claimed here. Paddle is not installed in this no-card env, so
the suite skips honestly rather than fake-passing. As required by antipattern
#11, setUp/tearDown snapshot and restore both the mutated global flag and the
process RNG state so nothing leaks to other tests in the same process.
"""

import unittest

try:
    import paddle

    from paddlefleet.tensor_parallel.random import (
        RecomputeWithoutOutput,
        enable_share_grad_holder,
    )

    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

_SKIP_REASON = "paddle is not installed in this environment"

_SHARE_FLAG = "FLAGS_share_tensor_for_grad_tensor_holder"


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class EnableShareGradHolderTest(unittest.TestCase):
    """The enable/restore branches of ``enable_share_grad_holder``."""

    def setUp(self):
        # Snapshot the global flag and process RNG state so both are restored
        # even if a test leaves them mutated (antipattern #11).
        self._orig_flag = paddle.get_flags([_SHARE_FLAG])[_SHARE_FLAG]
        self._orig_rng_state = paddle.get_rng_state("cpu")

    def tearDown(self):
        paddle.set_flags({_SHARE_FLAG: self._orig_flag})
        paddle.set_rng_state(self._orig_rng_state, device="cpu")

    def _current_flag(self):
        return paddle.get_flags([_SHARE_FLAG])[_SHARE_FLAG]

    def test_enables_flag_then_restores_disabled(self):
        """Starting disabled: flag is True inside, back to False on exit."""
        paddle.set_flags({_SHARE_FLAG: False})
        self.assertFalse(self._current_flag())

        with enable_share_grad_holder():
            # Body sees sharing enabled...
            self.assertTrue(self._current_flag())
        # ...and the manager undoes exactly what it turned on.
        self.assertFalse(self._current_flag())

    def test_leaves_flag_enabled_when_already_enabled(self):
        """Starting enabled: the manager must NOT clear a flag it did not set."""
        paddle.set_flags({_SHARE_FLAG: True})
        self.assertTrue(self._current_flag())

        with enable_share_grad_holder():
            self.assertTrue(self._current_flag())
        # old_value was truthy, so the finally branch skips the reset: still on.
        self.assertTrue(self._current_flag())

    def test_restores_disabled_flag_on_exception(self):
        """An exception in the body still triggers the reset-to-disabled path."""
        paddle.set_flags({_SHARE_FLAG: False})

        class _BodyError(RuntimeError):
            pass

        with self.assertRaises(_BodyError), enable_share_grad_holder():
            self.assertTrue(self._current_flag())
            raise _BodyError("boom")
        # finally must have run despite the raise.
        self.assertFalse(self._current_flag())


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class RecomputeWithoutOutputNoOpTest(unittest.TestCase):
    """The ``ctx is None`` no-op guard reached via ``run_recompute_now``."""

    def setUp(self):
        self._orig_rng_state = paddle.get_rng_state("cpu")

    def tearDown(self):
        paddle.set_rng_state(self._orig_rng_state, device="cpu")

    def test_fresh_instance_starts_empty(self):
        """A new wrapper holds no run_function, ctx, or saved outputs."""
        rec = RecomputeWithoutOutput()
        self.assertIsNone(rec.run_function)
        self.assertIsNone(rec.ctx)
        self.assertIsNone(rec.outputs)

    def test_run_recompute_now_is_noop_without_ctx(self):
        """With ctx still None, run_recompute_now returns and mutates nothing.

        This drives the ``if self.ctx is None: return`` early exit in
        ``_recompute`` (the documented "entered before a forward was captured"
        case), so no RecomputeStore/paddle work happens.
        """
        rec = RecomputeWithoutOutput()
        result = rec.run_recompute_now()
        self.assertIsNone(result)
        # State is untouched by the guarded early return.
        self.assertIsNone(rec.ctx)
        self.assertIsNone(rec.run_function)
        self.assertIsNone(rec.outputs)


if __name__ == "__main__":
    unittest.main()
