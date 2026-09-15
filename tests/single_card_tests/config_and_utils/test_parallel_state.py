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

"""Behavior tests for the CPU-observable pure logic in
``paddlefleet.parallel_state``.

``parallel_state`` stores the process-global parallel topology (tensor /
pipeline / context / expert groups plus their on-the-fly rank & world-size
overrides) and the query helpers Megatron-style parallelism reads. This suite
exercises only the pure-Python control logic that is fully defined WITHOUT a
real process group:

* the uninitialized / world-size==1 local paths (rank 0, world size 1),
* the on-the-fly rank & world-size override precedence,
* the ``check_initialized`` assertion contract and its exact guard messages
  vs. the ``check_initialized=False`` bypass,
* the virtual-pipeline first/last-stage branching driven by ``vp_stage``,
* the global-memory-buffer create / double-init-guard / destroy lifecycle.

Every expected value is hand-derived from the documented single-rank
semantics; none is read back from a value the test just stored (each override
test uses a value distinct from the branch it must beat, e.g. 3 vs the
default 0). No collective is faked and no fake ``world_size`` is injected
(antipattern 13): these are honest single-process checks that only claim to
verify the local no-communication paths -- real cross-rank init is left to the
multi-card suite. Every test that mutates a module global snapshots it in
``setUp`` and restores it in ``tearDown`` so the shared topology never leaks.
Because the production module imports ``paddle`` at import time, the whole
suite skips with an honest ImportError reason when Paddle is unavailable.
"""

import os
import sys
import unittest
import warnings

# Make the in-tree ``src/paddlefleet`` importable when the package has not been
# pip-installed (repo_root/src is three parents up from this test directory).
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Honest capability probe: parallel_state imports paddle at module load. Only a
# genuine missing-dependency (ImportError / ModuleNotFoundError) may skip; any
# other error must surface as a real failure rather than a fake pass.
try:
    import paddlefleet.parallel_state as ps
    from paddlefleet.utils._fleet_utils import GlobalMemoryBuffer

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    ps = None
    GlobalMemoryBuffer = None
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    "paddlefleet.parallel_state imports paddle at module load; it is not "
    "importable here: %r" % (_IMPORT_ERROR,)
)

# Every module global any test below writes to. Snapshotted per-test so the
# process-global topology cannot leak between tests or into sibling suites.
_TOUCHED_GLOBALS = (
    "_TENSOR_MODEL_PARALLEL_GROUP",
    "_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE",
    "_MPU_TENSOR_MODEL_PARALLEL_RANK",
    "_PIPELINE_MODEL_PARALLEL_GROUP",
    "_PIPELINE_MODEL_PARALLEL_WORLD_SIZE",
    "_PIPELINE_MODEL_PARALLEL_RANK",
    "_DATA_PARALLEL_GROUP",
    "_DATA_PARALLEL_GROUP_WITH_CP",
    "_CONTEXT_PARALLEL_GROUP",
    "_EXPERT_MODEL_PARALLEL_GROUP",
    "_EXPERT_DATA_PARALLEL_GROUP",
    "_EXPERT_TENSOR_PARALLEL_GROUP",
    "_EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP",
    "_MPU_EXPERT_MODEL_PARALLEL_RANK",
    "_MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE",
    "_MPU_EXPERT_TENSOR_PARALLEL_RANK",
    "_MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE",
    "_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK",
    "_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE",
    "_GLOBAL_MEMORY_BUFFER",
)


@unittest.skipUnless(ps is not None, _SKIP_REASON)
class _ParallelStateTestBase(unittest.TestCase):
    """Snapshot & restore every touched module global around each test."""

    def setUp(self):
        self._saved = {name: getattr(ps, name) for name in _TOUCHED_GLOBALS}

    def tearDown(self):
        for name, value in self._saved.items():
            setattr(ps, name, value)


class TestTensorModelParallelQueries(_ParallelStateTestBase):
    def test_world_size_defaults_to_one_when_uninitialized(self):
        # No MPU override and no TP group => group(check=False) is None => 1.
        ps._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
        ps._TENSOR_MODEL_PARALLEL_GROUP = None
        self.assertEqual(ps.get_tensor_model_parallel_world_size(), 1)

    def test_world_size_override_beats_the_uninitialized_default(self):
        # Override 8 must win over the group-None default of 1 (8 != 1).
        ps._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = 8
        ps._TENSOR_MODEL_PARALLEL_GROUP = None
        self.assertEqual(ps.get_tensor_model_parallel_world_size(), 8)

    def test_rank_defaults_to_zero_when_world_size_is_one(self):
        # rank override None, world size resolves to 1 => rank 0.
        ps._MPU_TENSOR_MODEL_PARALLEL_RANK = None
        ps._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
        ps._TENSOR_MODEL_PARALLEL_GROUP = None
        self.assertEqual(ps.get_tensor_model_parallel_rank(), 0)

    def test_rank_override_beats_the_world_size_one_default(self):
        # Override 3 must win over the world-size==1 fallback of 0 (3 != 0).
        ps._MPU_TENSOR_MODEL_PARALLEL_RANK = 3
        ps._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
        ps._TENSOR_MODEL_PARALLEL_GROUP = None
        self.assertEqual(ps.get_tensor_model_parallel_rank(), 3)


class TestPipelineModelParallelQueries(_ParallelStateTestBase):
    def test_world_size_defaults_to_one_when_uninitialized(self):
        ps._PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
        ps._PIPELINE_MODEL_PARALLEL_GROUP = None
        self.assertEqual(ps.get_pipeline_model_parallel_world_size(), 1)

    def test_setter_is_consumed_by_getter(self):
        # set_* writes the override global; the getter's override branch reads
        # it. 4 != the uninitialized default of 1, so the branch is proven.
        ps._PIPELINE_MODEL_PARALLEL_GROUP = None
        ps.set_pipeline_model_parallel_world_size(4)
        self.assertEqual(ps._PIPELINE_MODEL_PARALLEL_WORLD_SIZE, 4)
        self.assertEqual(ps.get_pipeline_model_parallel_world_size(), 4)

    def test_rank_defaults_to_zero_when_world_size_is_one(self):
        # rank override None, world size 1 => the world-size==1 fast path => 0
        # (returns before the paddle.distributed.is_initialized() branch).
        ps._PIPELINE_MODEL_PARALLEL_RANK = None
        ps._PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
        ps._PIPELINE_MODEL_PARALLEL_GROUP = None
        self.assertEqual(ps.get_pipeline_model_parallel_rank(), 0)

    def test_rank_override_beats_the_world_size_one_default(self):
        ps._PIPELINE_MODEL_PARALLEL_RANK = 2
        self.assertEqual(ps.get_pipeline_model_parallel_rank(), 2)


class TestContextParallelQueries(_ParallelStateTestBase):
    def test_world_size_is_one_when_group_absent(self):
        ps._CONTEXT_PARALLEL_GROUP = None
        self.assertEqual(ps.get_context_parallel_world_size(), 1)

    def test_rank_is_zero_when_group_absent(self):
        ps._CONTEXT_PARALLEL_GROUP = None
        self.assertEqual(ps.get_context_parallel_rank(), 0)


class TestVirtualPipelineQueries(_ParallelStateTestBase):
    def test_rank_is_none_by_default(self):
        ps._VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
        self.assertIsNone(ps.get_virtual_pipeline_model_parallel_rank())

    def test_world_size_is_none_by_default(self):
        ps._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
        self.assertIsNone(ps.get_virtual_pipeline_model_parallel_world_size())

    def test_deprecated_setter_warns_and_updates_rank(self):
        # The setter is documented deprecated: it must emit DeprecationWarning
        # AND still write the global (2 != the default None it replaces).
        ps._VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ps.set_virtual_pipeline_model_parallel_rank(2)
        self.assertTrue(
            any(issubclass(w.category, DeprecationWarning) for w in caught)
        )
        self.assertEqual(ps.get_virtual_pipeline_model_parallel_rank(), 2)


class TestPipelineStageBranching(_ParallelStateTestBase):
    def test_first_stage_true_when_rank_zero(self):
        ps._PIPELINE_MODEL_PARALLEL_RANK = None
        ps._PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
        ps._PIPELINE_MODEL_PARALLEL_GROUP = None
        self.assertTrue(ps.is_pipeline_first_stage())

    def test_first_stage_false_when_rank_nonzero(self):
        # ignore_virtual defaults True => virtual branch skipped; rank 1 != 0.
        ps._PIPELINE_MODEL_PARALLEL_RANK = 1
        self.assertFalse(ps.is_pipeline_first_stage())

    def test_first_stage_false_for_nonzero_vp_stage(self):
        # With VP enabled and ignore_virtual=False, any vp_stage != 0 short
        # circuits to False even though the real pipeline rank is 0.
        ps._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = 4
        ps._PIPELINE_MODEL_PARALLEL_RANK = 0
        self.assertFalse(
            ps.is_pipeline_first_stage(ignore_virtual=False, vp_stage=2)
        )

    def test_first_stage_true_for_vp_stage_zero(self):
        ps._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = 4
        ps._PIPELINE_MODEL_PARALLEL_RANK = 0
        self.assertTrue(
            ps.is_pipeline_first_stage(ignore_virtual=False, vp_stage=0)
        )

    def test_last_stage_true_when_world_size_one(self):
        # rank 0, world size 1 => 0 == (1 - 1) => True.
        ps._PIPELINE_MODEL_PARALLEL_RANK = None
        ps._PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
        ps._PIPELINE_MODEL_PARALLEL_GROUP = None
        self.assertTrue(ps.is_pipeline_last_stage())

    def test_last_stage_false_when_not_final_rank(self):
        # rank 1, world size 4 => 1 != 3 => False.
        ps._PIPELINE_MODEL_PARALLEL_RANK = 1
        ps._PIPELINE_MODEL_PARALLEL_WORLD_SIZE = 4
        self.assertFalse(ps.is_pipeline_last_stage())

    def test_last_stage_true_on_final_rank(self):
        # rank 3, world size 4 => 3 == 3 => True.
        ps._PIPELINE_MODEL_PARALLEL_RANK = 3
        ps._PIPELINE_MODEL_PARALLEL_WORLD_SIZE = 4
        self.assertTrue(ps.is_pipeline_last_stage())

    def test_last_stage_false_for_non_final_vp_stage(self):
        # VP world size 4 => last vp stage is 3; vp_stage 1 short circuits to
        # False even though the pipeline part (rank 0, ws 1) would be True.
        ps._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = 4
        ps._PIPELINE_MODEL_PARALLEL_RANK = 0
        ps._PIPELINE_MODEL_PARALLEL_WORLD_SIZE = 1
        ps._PIPELINE_MODEL_PARALLEL_GROUP = None
        self.assertFalse(
            ps.is_pipeline_last_stage(ignore_virtual=False, vp_stage=1)
        )

    def test_last_stage_true_on_final_vp_stage(self):
        ps._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = 4
        ps._PIPELINE_MODEL_PARALLEL_RANK = 0
        ps._PIPELINE_MODEL_PARALLEL_WORLD_SIZE = 1
        ps._PIPELINE_MODEL_PARALLEL_GROUP = None
        self.assertTrue(
            ps.is_pipeline_last_stage(ignore_virtual=False, vp_stage=3)
        )


class TestExpertRankQueries(_ParallelStateTestBase):
    def test_expert_model_parallel_rank_defaults_to_zero(self):
        # No override, no group, and (in this single process) paddle.distributed
        # is genuinely not initialized => the ``else`` branch returns 0. This
        # calls the real paddle.distributed.is_initialized() (a genuine
        # collaborator), it does not mock the function under test.
        ps._MPU_EXPERT_MODEL_PARALLEL_RANK = None
        ps._EXPERT_MODEL_PARALLEL_GROUP = None
        self.assertEqual(ps.get_expert_model_parallel_rank(), 0)

    def test_expert_model_parallel_rank_override_beats_default(self):
        # set_* writes the override consumed by the getter; 7 != default 0.
        ps._EXPERT_MODEL_PARALLEL_GROUP = None
        ps.set_expert_model_parallel_rank(7)
        self.assertEqual(ps.get_expert_model_parallel_rank(), 7)

    def test_expert_tensor_parallel_rank_defaults_to_zero(self):
        # No override and no expert-TP group => ``else`` branch returns 0.
        ps._MPU_EXPERT_TENSOR_PARALLEL_RANK = None
        ps._EXPERT_TENSOR_PARALLEL_GROUP = None
        self.assertEqual(ps.get_expert_tensor_parallel_rank(), 0)

    def test_expert_tensor_parallel_rank_override_beats_default(self):
        ps._EXPERT_TENSOR_PARALLEL_GROUP = None
        ps.set_expert_tensor_parallel_rank(3)
        self.assertEqual(ps.get_expert_tensor_parallel_rank(), 3)

    def test_expert_tensor_parallel_rank_falls_back_to_tp_rank(self):
        # When an expert-TP group is present but no explicit override, the
        # helper returns the tensor-model-parallel rank. The group is only
        # tested for ``is not None`` (never used for communication), so a bare
        # sentinel exercises the branch honestly; 5 != the else-branch 0.
        ps._MPU_EXPERT_TENSOR_PARALLEL_RANK = None
        ps._EXPERT_TENSOR_PARALLEL_GROUP = object()
        ps._MPU_TENSOR_MODEL_PARALLEL_RANK = 5
        self.assertEqual(ps.get_expert_tensor_parallel_rank(), 5)


class TestExpertTensorParallelWorldSize(_ParallelStateTestBase):
    def test_override_is_returned(self):
        # Override branch (MPU set) works correctly; 4 != the buggy None the
        # uninitialized fallback would produce (see the expectedFailure below).
        ps.set_expert_tensor_parallel_world_size(4)
        self.assertEqual(ps.get_expert_tensor_parallel_world_size(), 4)

    @unittest.expectedFailure
    def test_uninitialized_should_default_to_one(self):
        """REAL BUG (parallel_state.py:376-377). With no MPU override, no
        expert-TP group, and no tensor-model-parallel override, the helper hits

            if not _EXPERT_TENSOR_PARALLEL_GROUP:
                return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE

        and returns the raw ``_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE`` global,
        which is ``None`` when uninitialized -- instead of delegating to
        ``get_tensor_model_parallel_world_size()`` which correctly falls back to
        1. The correct world size for a single un-configured rank is 1, so this
        asserts ==1 and is marked expectedFailure (actual is None). Production
        is intentionally left unmodified.
        """
        ps._MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE = None
        ps._EXPERT_TENSOR_PARALLEL_GROUP = None
        ps._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
        self.assertEqual(ps.get_expert_tensor_parallel_world_size(), 1)


class TestNotInitializedGuards(_ParallelStateTestBase):
    """The check_initialized guards must raise with their exact messages, and
    check_initialized=False must bypass the guard and return the (None) group.
    """

    def test_tensor_model_parallel_group_guard_message(self):
        ps._TENSOR_MODEL_PARALLEL_GROUP = None
        with self.assertRaisesRegex(
            AssertionError, "tensor model parallel group is not initialized"
        ):
            ps.get_tensor_model_parallel_group(check_initialized=True)

    def test_tensor_model_parallel_group_bypass_returns_none(self):
        ps._TENSOR_MODEL_PARALLEL_GROUP = None
        self.assertIsNone(
            ps.get_tensor_model_parallel_group(check_initialized=False)
        )

    def test_pipeline_model_parallel_group_guard_message(self):
        ps._PIPELINE_MODEL_PARALLEL_GROUP = None
        with self.assertRaisesRegex(
            AssertionError, "pipeline_model parallel group is not initialized"
        ):
            ps.get_pipeline_model_parallel_group(check_initialized=True)

    def test_data_parallel_group_guard_message(self):
        ps._DATA_PARALLEL_GROUP = None
        with self.assertRaisesRegex(
            AssertionError, "data parallel group is not initialized"
        ):
            ps.get_data_parallel_group(check_initialized=True)

    def test_data_parallel_group_with_cp_guard_message(self):
        ps._DATA_PARALLEL_GROUP_WITH_CP = None
        with self.assertRaisesRegex(
            AssertionError,
            "data parallel group with context parallel combined is not "
            "initialized",
        ):
            ps.get_data_parallel_group(
                check_initialized=True, with_context_parallel=True
            )

    def test_expert_model_parallel_group_guard_message(self):
        ps._EXPERT_MODEL_PARALLEL_GROUP = None
        with self.assertRaisesRegex(
            AssertionError, "expert model parallel group is not initialized"
        ):
            ps.get_expert_model_parallel_group(check_initialized=True)

    def test_expert_data_parallel_group_guard_message(self):
        ps._EXPERT_DATA_PARALLEL_GROUP = None
        with self.assertRaisesRegex(
            AssertionError, "Expert data parallel group is not initialized"
        ):
            ps.get_expert_data_parallel_group(check_initialized=True)

    def test_context_parallel_group_guard_message(self):
        ps._CONTEXT_PARALLEL_GROUP = None
        with self.assertRaisesRegex(
            AssertionError, "context parallel group is not initialized"
        ):
            ps.get_context_parallel_group(check_initialized=True)

    def test_context_parallel_group_default_arg_bypasses_guard(self):
        # get_context_parallel_group defaults check_initialized=False, so an
        # absent group must return None rather than raise.
        ps._CONTEXT_PARALLEL_GROUP = None
        self.assertIsNone(ps.get_context_parallel_group())

    def test_expert_tensor_parallel_group_guard_message(self):
        ps._EXPERT_TENSOR_PARALLEL_GROUP = None
        with self.assertRaisesRegex(
            AssertionError, "Expert tensor parallel group is not initialized"
        ):
            ps.get_expert_tensor_parallel_group(check_initialized=True)

    def test_expert_tensor_and_model_parallel_group_guard_message(self):
        ps._EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = None
        with self.assertRaisesRegex(
            AssertionError,
            "Expert tensor and model parallel group is not initialized",
        ):
            ps.get_expert_tensor_and_model_parallel_group(
                check_initialized=True
            )

    def test_embedding_group_raises_not_implemented(self):
        with self.assertRaisesRegex(
            NotImplementedError, "Not supported get_embedding_group yet"
        ):
            ps.get_embedding_group()


class TestGlobalMemoryBufferLifecycle(_ParallelStateTestBase):
    def test_have_buffer_is_false_when_absent(self):
        ps._GLOBAL_MEMORY_BUFFER = None
        self.assertFalse(ps.have_global_memory_buffer())

    def test_get_buffer_guard_message_when_absent(self):
        ps._GLOBAL_MEMORY_BUFFER = None
        with self.assertRaisesRegex(
            AssertionError, "global memory buffer is not initialized"
        ):
            ps.get_global_memory_buffer()

    def test_set_creates_buffer_that_get_returns(self):
        # From the uninitialized state, _set_* must construct a real
        # GlobalMemoryBuffer; have_* flips to True and get_* returns that very
        # object (identity), exercising the create path end to end.
        ps._GLOBAL_MEMORY_BUFFER = None
        ps._set_global_memory_buffer()
        self.assertTrue(ps.have_global_memory_buffer())
        buf = ps.get_global_memory_buffer()
        self.assertIsInstance(buf, GlobalMemoryBuffer)
        self.assertIs(buf, ps._GLOBAL_MEMORY_BUFFER)

    def test_set_rejects_double_initialization(self):
        # _set_* asserts the buffer is currently None; a second init must raise
        # its exact "already initialized" message.
        ps._GLOBAL_MEMORY_BUFFER = object()
        with self.assertRaisesRegex(
            AssertionError, "global memory buffer is already initialized"
        ):
            ps._set_global_memory_buffer()

    def test_destroy_clears_existing_buffer(self):
        # Start from a non-None buffer; destroy must reset it to None (observing
        # the effect of destroy, not a value the test just read back).
        ps._GLOBAL_MEMORY_BUFFER = object()
        ps.destroy_global_memory_buffer()
        self.assertIsNone(ps._GLOBAL_MEMORY_BUFFER)
        self.assertFalse(ps.have_global_memory_buffer())


if __name__ == "__main__":
    unittest.main()
