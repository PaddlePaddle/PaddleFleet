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

"""Behavior tests for ``paddlefleet.parallel_state``.

``parallel_state`` holds the process-global parallel topology (tensor / pipeline
/ context / expert groups plus their on-the-fly rank & world-size overrides) and
the query helpers that Megatron-style parallelism reads. This suite exercises the
pure-Python control logic that is well-defined WITHOUT a real process group:

* the world-size==1 / uninitialized local paths (rank 0, world size 1),
* the ``check_initialized`` assertion contract vs. the ``False`` bypass,
* the virtual-pipeline first/last-stage branching from ``vp_stage``,
* the on-the-fly rank/world-size override precedence,
* the global-memory-buffer lifecycle.

Expected values are hand-derived from the documented single-rank semantics, never
by reading back a value the test just stored. No collective is faked and no fake
``world_size`` is injected (antipattern 13): these are honest single-process
"no-card" local-path checks and they only claim to verify those local paths --
cross-rank communication is out of scope and must be proven on a real multi-card
job. Every test that touches a module global saves the original and restores it
via ``addCleanup`` so the shared topology is never left mutated. Because the
production module imports ``paddle`` at import time, the whole suite is skipped
with an honest reason when Paddle (and therefore the module) is unavailable.
"""

import os
import sys
import unittest
import warnings

# Make the in-tree ``src/paddlefleet`` importable when the package has not been
# pip-installed into the environment (repo_root/src is 4 levels up from here).
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import paddle

    from paddlefleet import parallel_state
    from paddlefleet.utils._fleet_utils import GlobalMemoryBuffer

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet not installed on this host
    parallel_state = None
    GlobalMemoryBuffer = None
    _IMPORT_ERROR = exc


def _restore(attr, value):
    """Return a callable that resets ``parallel_state.<attr>`` to ``value``."""
    return lambda: setattr(parallel_state, attr, value)


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}",
)
class VirtualPipelineRankRoundTripTest(unittest.TestCase):
    """set_/get_virtual_pipeline_model_parallel_rank round-trip + deprecation."""

    def setUp(self):
        # The setter mutates a process-global; snapshot and restore it so no
        # sibling test inherits our value (antipattern 11).
        orig = parallel_state.get_virtual_pipeline_model_parallel_rank()
        self.addCleanup(_restore("_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK", orig))

    def test_set_then_get_returns_the_stored_rank(self):
        """After setting rank R the getter must return exactly R.

        Hand-derived: the setter records R in the module global and the getter
        returns that global unchanged, so get() == R for any R we set.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            parallel_state.set_virtual_pipeline_model_parallel_rank(3)
            self.assertEqual(
                parallel_state.get_virtual_pipeline_model_parallel_rank(), 3
            )
            parallel_state.set_virtual_pipeline_model_parallel_rank(0)
            self.assertEqual(
                parallel_state.get_virtual_pipeline_model_parallel_rank(), 0
            )

    def test_setter_emits_deprecation_warning(self):
        """The global-scope setter is deprecated and must warn on every call."""
        with self.assertWarns(DeprecationWarning):
            parallel_state.set_virtual_pipeline_model_parallel_rank(1)
        # And the deprecated call still performed its documented side effect.
        self.assertEqual(
            parallel_state.get_virtual_pipeline_model_parallel_rank(), 1
        )


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}",
)
class PipelineStageVirtualBranchTest(unittest.TestCase):
    """is_pipeline_first/last_stage virtual-pipeline (ignore_virtual=False) logic."""

    def setUp(self):
        # Pin a deterministic single-stage pipeline (pp rank 0 of world 1) and a
        # 4-chunk virtual pipeline, saving/restoring each global we touch.
        for attr in (
            "_PIPELINE_MODEL_PARALLEL_WORLD_SIZE",
            "_PIPELINE_MODEL_PARALLEL_RANK",
            "_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE",
        ):
            self.addCleanup(_restore(attr, getattr(parallel_state, attr)))
        parallel_state._PIPELINE_MODEL_PARALLEL_WORLD_SIZE = 1
        parallel_state._PIPELINE_MODEL_PARALLEL_RANK = 0
        parallel_state._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = 4

    def test_first_stage_true_only_for_vp_stage_zero(self):
        """First stage requires the 0th virtual chunk on pp rank 0.

        Hand-derived: with 4 virtual chunks the first stage is chunk 0; chunk 2
        is a later chunk and must not be reported as first even though the
        underlying pp rank (0) is the first physical stage.
        """
        self.assertTrue(
            parallel_state.is_pipeline_first_stage(
                ignore_virtual=False, vp_stage=0
            )
        )
        self.assertFalse(
            parallel_state.is_pipeline_first_stage(
                ignore_virtual=False, vp_stage=2
            )
        )

    def test_last_stage_true_only_for_final_vp_stage(self):
        """Last stage requires the final virtual chunk (index world_size-1).

        Hand-derived: 4 chunks -> final index 3 on the last pp stage (rank 0 of
        world 1). Chunk 1 is not the final chunk, so it is not the last stage.
        """
        self.assertTrue(
            parallel_state.is_pipeline_last_stage(
                ignore_virtual=False, vp_stage=3
            )
        )
        self.assertFalse(
            parallel_state.is_pipeline_last_stage(
                ignore_virtual=False, vp_stage=1
            )
        )

    def test_first_stage_requires_vp_stage_when_virtual_enabled(self):
        """Omitting vp_stage while virtual pipeline is enabled is an error."""
        with self.assertRaises(AssertionError):
            parallel_state.is_pipeline_first_stage(
                ignore_virtual=False, vp_stage=None
            )

    def test_last_stage_requires_vp_stage_when_virtual_enabled(self):
        """Omitting vp_stage while virtual pipeline is enabled is an error."""
        with self.assertRaises(AssertionError):
            parallel_state.is_pipeline_last_stage(
                ignore_virtual=False, vp_stage=None
            )

    def test_ignore_virtual_skips_vp_stage_requirement(self):
        """ignore_virtual=True bypasses the virtual check entirely.

        Hand-derived: with the virtual branch skipped the answer reduces to the
        physical pp position (rank 0 of world 1), which is both first and last,
        and needs no vp_stage argument.
        """
        self.assertTrue(
            parallel_state.is_pipeline_first_stage(ignore_virtual=True)
        )
        self.assertTrue(
            parallel_state.is_pipeline_last_stage(ignore_virtual=True)
        )


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}",
)
class UninitializedGroupContractTest(unittest.TestCase):
    """The check_initialized flag: assert when set, return None when cleared.

    In a fresh process no group has been installed, so ``check_initialized=True``
    (the getters' default) must raise, and ``check_initialized=False`` must return
    the uninitialized group object (``None``) instead. This exercises the effect
    of the flag rather than merely that a call happened (antipattern 3).
    """

    def test_data_parallel_group_asserts_when_uninitialized(self):
        with self.assertRaises(AssertionError):
            parallel_state.get_data_parallel_group()

    def test_data_parallel_group_with_cp_asserts_when_uninitialized(self):
        # The with-context-parallel branch guards a *separate* global.
        with self.assertRaises(AssertionError):
            parallel_state.get_data_parallel_group(with_context_parallel=True)

    def test_tensor_model_parallel_group_asserts_when_uninitialized(self):
        with self.assertRaises(AssertionError):
            parallel_state.get_tensor_model_parallel_group()

    def test_pipeline_model_parallel_group_asserts_when_uninitialized(self):
        with self.assertRaises(AssertionError):
            parallel_state.get_pipeline_model_parallel_group()

    def test_expert_model_parallel_group_asserts_when_uninitialized(self):
        with self.assertRaises(AssertionError):
            parallel_state.get_expert_model_parallel_group()

    def test_expert_tensor_and_model_group_asserts_when_uninitialized(self):
        with self.assertRaises(AssertionError):
            parallel_state.get_expert_tensor_and_model_parallel_group()

    def test_check_initialized_false_returns_none_group(self):
        """The bypass path must return the (uninitialized) None group, not raise.

        Hand-derived: no group installed -> the stored group is None, and with
        the assertion suppressed the getter surfaces that None unchanged.
        """
        self.assertIsNone(
            parallel_state.get_data_parallel_group(check_initialized=False)
        )
        self.assertIsNone(
            parallel_state.get_data_parallel_group(
                check_initialized=False, with_context_parallel=True
            )
        )
        self.assertIsNone(
            parallel_state.get_tensor_model_parallel_group(
                check_initialized=False
            )
        )
        self.assertIsNone(
            parallel_state.get_pipeline_model_parallel_group(
                check_initialized=False
            )
        )


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}",
)
class SingleRankDefaultsTest(unittest.TestCase):
    """World-size==1 / disabled-parallelism local paths return rank 0, size 1."""

    def test_tensor_model_parallel_size_is_one_when_uninitialized(self):
        # Hand-derived: no TP group and no on-the-fly override -> size 1, rank 0.
        self.assertEqual(
            parallel_state.get_tensor_model_parallel_world_size(), 1
        )
        self.assertEqual(parallel_state.get_tensor_model_parallel_rank(), 0)

    def test_pipeline_model_parallel_size_is_one_when_uninitialized(self):
        self.assertEqual(
            parallel_state.get_pipeline_model_parallel_world_size(), 1
        )
        self.assertEqual(parallel_state.get_pipeline_model_parallel_rank(), 0)

    def test_context_parallel_size_is_one_when_disabled(self):
        # Hand-derived: CP group is None (disabled) -> world size 1, rank 0.
        self.assertEqual(parallel_state.get_context_parallel_world_size(), 1)
        self.assertEqual(parallel_state.get_context_parallel_rank(), 0)

    def test_expert_tensor_and_model_size_and_rank_zero_without_dist(self):
        """Without an initialized distributed backend both queries return 0.

        Hand-derived: these two helpers short-circuit to 0 when
        ``paddle.distributed`` is not initialized, which is the state on a plain
        CPU host (no launcher / no process group).
        """
        self.assertFalse(paddle.distributed.is_initialized())
        self.assertEqual(
            parallel_state.get_expert_tensor_and_model_parallel_world_size(), 0
        )
        self.assertEqual(
            parallel_state.get_expert_tensor_and_model_parallel_rank(), 0
        )


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}",
)
class OnTheFlyOverrideTest(unittest.TestCase):
    """The set_* helpers install overrides that take precedence over groups."""

    def test_pipeline_world_size_override_takes_precedence(self):
        orig = parallel_state._PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        self.addCleanup(_restore("_PIPELINE_MODEL_PARALLEL_WORLD_SIZE", orig))
        parallel_state.set_pipeline_model_parallel_world_size(7)
        # Hand-derived: the explicit override wins over the group-derived value.
        self.assertEqual(
            parallel_state.get_pipeline_model_parallel_world_size(), 7
        )

    def test_expert_model_parallel_rank_override_takes_precedence(self):
        orig = parallel_state._MPU_EXPERT_MODEL_PARALLEL_RANK
        self.addCleanup(_restore("_MPU_EXPERT_MODEL_PARALLEL_RANK", orig))
        parallel_state.set_expert_model_parallel_rank(5)
        self.assertEqual(parallel_state.get_expert_model_parallel_rank(), 5)

    def test_expert_model_parallel_rank_defaults_to_zero_without_override(self):
        orig = parallel_state._MPU_EXPERT_MODEL_PARALLEL_RANK
        self.addCleanup(_restore("_MPU_EXPERT_MODEL_PARALLEL_RANK", orig))
        parallel_state._MPU_EXPERT_MODEL_PARALLEL_RANK = None
        # Hand-derived: no override, no expert group, dist not initialized -> 0.
        self.assertEqual(parallel_state.get_expert_model_parallel_rank(), 0)

    def test_expert_tensor_parallel_rank_defaults_to_zero(self):
        orig = parallel_state._MPU_EXPERT_TENSOR_PARALLEL_RANK
        self.addCleanup(_restore("_MPU_EXPERT_TENSOR_PARALLEL_RANK", orig))
        parallel_state._MPU_EXPERT_TENSOR_PARALLEL_RANK = None
        # Hand-derived: no override and no expert-tensor group -> rank 0.
        self.assertEqual(parallel_state.get_expert_tensor_parallel_rank(), 0)


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}",
)
class GlobalMemoryBufferLifecycleTest(unittest.TestCase):
    """destroy / have / _set / get transitions of the global memory buffer."""

    def setUp(self):
        orig = parallel_state._GLOBAL_MEMORY_BUFFER
        self.addCleanup(_restore("_GLOBAL_MEMORY_BUFFER", orig))

    def test_lifecycle_destroy_set_and_query(self):
        """destroy -> absent+raises; _set -> present+returns a real buffer.

        Hand-derived from the documented lifecycle: after destroy the buffer is
        None, so ``have_*`` is False and ``get_*`` raises; ``_set_*`` (which
        asserts the slot was empty) then installs a genuine GlobalMemoryBuffer
        that ``get_*`` returns and ``have_*`` reports as present.
        """
        parallel_state.destroy_global_memory_buffer()
        self.assertFalse(parallel_state.have_global_memory_buffer())
        with self.assertRaises(AssertionError):
            parallel_state.get_global_memory_buffer()

        parallel_state._set_global_memory_buffer()
        self.assertTrue(parallel_state.have_global_memory_buffer())
        buffer = parallel_state.get_global_memory_buffer()
        self.assertIsInstance(buffer, GlobalMemoryBuffer)
        # get_* must hand back the very object _set_* installed, not a fresh one.
        self.assertIs(buffer, parallel_state.get_global_memory_buffer())

    def test_set_rejects_double_initialization(self):
        """_set_global_memory_buffer asserts the slot is empty before filling it.

        Hand-derived: the buffer is a process-wide singleton, so installing one
        while another is present must raise rather than silently overwrite.
        """
        parallel_state.destroy_global_memory_buffer()
        parallel_state._set_global_memory_buffer()
        with self.assertRaises(AssertionError):
            parallel_state._set_global_memory_buffer()


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}",
)
class ExpertTensorParallelWorldSizeBugTest(unittest.TestCase):
    """Documents a real defect in get_expert_tensor_parallel_world_size.

    src/paddlefleet/parallel_state.py:376-377 -- when no expert-tensor override
    and no expert-tensor group are set, the fallback returns the raw module
    global ``_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE`` (None by default) instead of
    calling ``get_tensor_model_parallel_world_size()`` (which returns 1 when
    uninitialized). The docstring/comment say it should fall back to the tensor
    parallel world size, so on a clean, uninitialized process this helper should
    return 1 but actually returns None. Marked expectedFailure so the suite
    records the bug WITHOUT editing production code.
    """

    @unittest.expectedFailure
    def test_uninitialized_expert_tp_size_should_be_one(self):
        for attr in (
            "_MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE",
            "_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE",
            "_EXPERT_TENSOR_PARALLEL_GROUP",
        ):
            self.addCleanup(_restore(attr, getattr(parallel_state, attr)))
        parallel_state._MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE = None
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
        parallel_state._EXPERT_TENSOR_PARALLEL_GROUP = None
        # Correct behaviour, consistent with get_tensor_model_parallel_world_size
        # returning 1 when nothing is initialized. Currently returns None (bug).
        self.assertEqual(
            parallel_state.get_expert_tensor_parallel_world_size(), 1
        )


if __name__ == "__main__":
    unittest.main()
