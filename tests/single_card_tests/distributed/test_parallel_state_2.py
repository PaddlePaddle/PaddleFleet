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

"""Behavior tests for the ``paddlefleet.parallel_state`` module globals.

``parallel_state`` keeps the process-group handles and the mpu-override scalars
for every parallel dimension (tp / pp / cp / ep / expert-tp) as module-level
globals, and exposes getters/setters that layer a well-defined fallback contract
on top of them:

* group getters raise ``AssertionError`` when ``check_initialized=True`` and the
  group is unset, and return ``None`` when ``check_initialized=False``;
* world-size / rank getters honour an explicit mpu override first, then fall
  back to the group handle, and finally to the single-card default (world_size
  ``1``, rank ``0``);
* pipeline-stage predicates combine the pipeline rank/world-size with the
  optional virtual-pipeline stage.

These tests drive that real control flow on CPU with the groups left unset (the
single-card / uninitialized path -- interface-supported world_size==1, not a
stand-in for multi-card collective numerics). Expected values are derived by
hand from the contract above, never from the implementation. Every global that
a test mutates is snapshotted and restored via ``addCleanup`` so the shared
module state is never left polluted for sibling tests.

Because the production module imports ``paddle`` at import time, the whole suite
is skipped with an honest reason when Paddle / paddlefleet is unavailable on the
host rather than reporting a hollow pass.
"""

import os
import sys
import unittest

# Make the in-tree ``src/paddlefleet`` importable when the package has not been
# pip-installed into the environment (repo_root/src is 4 levels up from here).
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import paddle  # noqa: F401

    from paddlefleet import parallel_state
    from paddlefleet.utils._fleet_utils import GlobalMemoryBuffer

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet not installed on this host
    parallel_state = None
    GlobalMemoryBuffer = None
    _IMPORT_ERROR = exc


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}",
)
class _ParallelStateTestBase(unittest.TestCase):
    """Shared helper that mutates module globals but always restores them."""

    def _set_globals(self, **overrides):
        """Snapshot then overwrite named ``parallel_state`` globals.

        Every touched global is restored to its original value via
        ``addCleanup`` so a failing assertion cannot leak forced state into
        sibling tests running in the same process.
        """
        for name, value in overrides.items():
            self.assertTrue(
                hasattr(parallel_state, name),
                f"parallel_state has no global {name!r}",
            )
            original = getattr(parallel_state, name)
            self.addCleanup(setattr, parallel_state, name, original)
            setattr(parallel_state, name, value)


class TestUninitializedGroupGetters(_ParallelStateTestBase):
    """Group getters gate on ``check_initialized`` when the group is unset."""

    def test_tensor_group_raises_when_checked_returns_none_otherwise(self):
        """Unset tp group: assert on check, ``None`` when the check is waived."""
        self._set_globals(_TENSOR_MODEL_PARALLEL_GROUP=None)
        with self.assertRaises(AssertionError):
            parallel_state.get_tensor_model_parallel_group()
        self.assertIsNone(
            parallel_state.get_tensor_model_parallel_group(
                check_initialized=False
            )
        )

    def test_pipeline_group_raises_when_checked_returns_none_otherwise(self):
        """Unset pp group: assert on check, ``None`` when the check is waived."""
        self._set_globals(_PIPELINE_MODEL_PARALLEL_GROUP=None)
        with self.assertRaises(AssertionError):
            parallel_state.get_pipeline_model_parallel_group()
        self.assertIsNone(
            parallel_state.get_pipeline_model_parallel_group(
                check_initialized=False
            )
        )

    def test_data_group_default_and_with_context_parallel_are_distinct(self):
        """The plain and cp-combined dp handles are separate globals."""
        plain = object()
        with_cp = object()
        self._set_globals(
            _DATA_PARALLEL_GROUP=plain,
            _DATA_PARALLEL_GROUP_WITH_CP=with_cp,
        )
        # with_context_parallel selects the cp-combined handle, not the plain one.
        self.assertIs(parallel_state.get_data_parallel_group(), plain)
        self.assertIs(
            parallel_state.get_data_parallel_group(with_context_parallel=True),
            with_cp,
        )

    def test_data_group_raises_only_for_the_requested_variant(self):
        """A missing cp-combined dp group asserts only when that variant is asked."""
        plain = object()
        self._set_globals(
            _DATA_PARALLEL_GROUP=plain,
            _DATA_PARALLEL_GROUP_WITH_CP=None,
        )
        # Plain variant is present -> no error and returns the plain handle.
        self.assertIs(parallel_state.get_data_parallel_group(), plain)
        # cp-combined variant is unset -> assertion when checked.
        with self.assertRaises(AssertionError):
            parallel_state.get_data_parallel_group(with_context_parallel=True)

    def test_expert_model_and_data_groups_raise_when_unset(self):
        """Expert model/data group getters assert on the uninitialized default."""
        self._set_globals(
            _EXPERT_MODEL_PARALLEL_GROUP=None,
            _EXPERT_DATA_PARALLEL_GROUP=None,
        )
        with self.assertRaises(AssertionError):
            parallel_state.get_expert_model_parallel_group()
        with self.assertRaises(AssertionError):
            parallel_state.get_expert_data_parallel_group()
        # The check can be waived to build state before initialization.
        self.assertIsNone(
            parallel_state.get_expert_model_parallel_group(
                check_initialized=False
            )
        )
        self.assertIsNone(
            parallel_state.get_expert_data_parallel_group(
                check_initialized=False
            )
        )


# placeholder-more-classes


class TestWorldSizeAndRankFallbacks(_ParallelStateTestBase):
    """World-size / rank getters fall back to single-card defaults."""

    def test_tensor_world_size_and_rank_single_card_defaults(self):
        """No override and no group -> world_size 1, rank 0."""
        self._set_globals(
            _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE=None,
            _MPU_TENSOR_MODEL_PARALLEL_RANK=None,
            _TENSOR_MODEL_PARALLEL_GROUP=None,
        )
        self.assertEqual(
            parallel_state.get_tensor_model_parallel_world_size(), 1
        )
        self.assertEqual(parallel_state.get_tensor_model_parallel_rank(), 0)

    def test_tensor_world_size_and_rank_honour_explicit_override(self):
        """The mpu override wins over the group/default fallback."""
        self._set_globals(
            _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE=8,
            _MPU_TENSOR_MODEL_PARALLEL_RANK=5,
            _TENSOR_MODEL_PARALLEL_GROUP=None,
        )
        self.assertEqual(
            parallel_state.get_tensor_model_parallel_world_size(), 8
        )
        self.assertEqual(parallel_state.get_tensor_model_parallel_rank(), 5)

    def test_context_parallel_world_size_and_rank_default(self):
        """Unset cp group -> world_size 1, rank 0."""
        self._set_globals(_CONTEXT_PARALLEL_GROUP=None)
        self.assertEqual(parallel_state.get_context_parallel_world_size(), 1)
        self.assertEqual(parallel_state.get_context_parallel_rank(), 0)

    def test_pipeline_world_size_default_and_explicit_override(self):
        """Default falls to 1; an explicit world-size override is returned."""
        self._set_globals(
            _PIPELINE_MODEL_PARALLEL_WORLD_SIZE=None,
            _PIPELINE_MODEL_PARALLEL_GROUP=None,
        )
        self.assertEqual(
            parallel_state.get_pipeline_model_parallel_world_size(), 1
        )
        parallel_state.set_pipeline_model_parallel_world_size(4)
        self.assertEqual(
            parallel_state.get_pipeline_model_parallel_world_size(), 4
        )

    def test_pipeline_rank_uses_override_then_defaults_to_zero(self):
        """Explicit pp rank override is returned; otherwise 0 on single card."""
        self._set_globals(
            _PIPELINE_MODEL_PARALLEL_RANK=3,
            _PIPELINE_MODEL_PARALLEL_WORLD_SIZE=8,
        )
        self.assertEqual(parallel_state.get_pipeline_model_parallel_rank(), 3)
        # Clearing the override with world_size 1 collapses back to rank 0.
        self._set_globals(
            _PIPELINE_MODEL_PARALLEL_RANK=None,
            _PIPELINE_MODEL_PARALLEL_WORLD_SIZE=1,
            _PIPELINE_MODEL_PARALLEL_GROUP=None,
        )
        self.assertEqual(parallel_state.get_pipeline_model_parallel_rank(), 0)

    def test_expert_tensor_parallel_world_size_uses_explicit_override(self):
        """An explicit expert-tp world-size override is returned verbatim."""
        self._set_globals(_MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE=3)
        self.assertEqual(
            parallel_state.get_expert_tensor_parallel_world_size(), 3
        )


class TestExpertRankSetters(_ParallelStateTestBase):
    """Expert-parallel rank setters round-trip through their getters."""

    def test_expert_model_parallel_rank_set_get(self):
        """set_expert_model_parallel_rank feeds get_expert_model_parallel_rank."""
        self._set_globals(_MPU_EXPERT_MODEL_PARALLEL_RANK=None)
        parallel_state.set_expert_model_parallel_rank(2)
        self.assertEqual(parallel_state.get_expert_model_parallel_rank(), 2)

    def test_expert_tensor_parallel_rank_set_get(self):
        """set_expert_tensor_parallel_rank feeds get_expert_tensor_parallel_rank."""
        self._set_globals(_MPU_EXPERT_TENSOR_PARALLEL_RANK=None)
        parallel_state.set_expert_tensor_parallel_rank(1)
        self.assertEqual(parallel_state.get_expert_tensor_parallel_rank(), 1)

    def test_expert_tensor_parallel_rank_defaults_to_zero(self):
        """No override and no expert-tp group -> rank 0."""
        self._set_globals(
            _MPU_EXPERT_TENSOR_PARALLEL_RANK=None,
            _EXPERT_TENSOR_PARALLEL_GROUP=None,
        )
        self.assertEqual(parallel_state.get_expert_tensor_parallel_rank(), 0)


class TestPipelineStagePredicates(_ParallelStateTestBase):
    """first/last-stage predicates combine pp rank, world-size and vp stage."""

    def test_middle_rank_is_neither_first_nor_last(self):
        """rank 1 of a 4-stage pipeline is neither the first nor last stage."""
        self._set_globals(
            _PIPELINE_MODEL_PARALLEL_RANK=1,
            _PIPELINE_MODEL_PARALLEL_WORLD_SIZE=4,
        )
        self.assertFalse(parallel_state.is_pipeline_first_stage())
        self.assertFalse(parallel_state.is_pipeline_last_stage())

    def test_edge_ranks_map_to_first_and_last(self):
        """rank 0 is the first stage; rank world_size-1 is the last stage."""
        self._set_globals(
            _PIPELINE_MODEL_PARALLEL_RANK=0,
            _PIPELINE_MODEL_PARALLEL_WORLD_SIZE=4,
        )
        self.assertTrue(parallel_state.is_pipeline_first_stage())
        self.assertFalse(parallel_state.is_pipeline_last_stage())

        self._set_globals(_PIPELINE_MODEL_PARALLEL_RANK=3)
        self.assertFalse(parallel_state.is_pipeline_first_stage())
        self.assertTrue(parallel_state.is_pipeline_last_stage())

    def test_virtual_stage_gates_first_and_last(self):
        """With virtual pipeline on, a non-boundary vp_stage forces False."""
        self._set_globals(
            _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE=2,
            _PIPELINE_MODEL_PARALLEL_RANK=0,
            _PIPELINE_MODEL_PARALLEL_WORLD_SIZE=1,
        )
        # vp world size 2 -> first stage requires vp_stage 0, last requires 1.
        self.assertTrue(
            parallel_state.is_pipeline_first_stage(
                ignore_virtual=False, vp_stage=0
            )
        )
        self.assertFalse(
            parallel_state.is_pipeline_first_stage(
                ignore_virtual=False, vp_stage=1
            )
        )
        self.assertTrue(
            parallel_state.is_pipeline_last_stage(
                ignore_virtual=False, vp_stage=1
            )
        )
        self.assertFalse(
            parallel_state.is_pipeline_last_stage(
                ignore_virtual=False, vp_stage=0
            )
        )

    def test_virtual_stage_requires_explicit_vp_stage(self):
        """Virtual pipeline on with vp_stage omitted must assert."""
        self._set_globals(_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE=2)
        with self.assertRaises(AssertionError):
            parallel_state.is_pipeline_first_stage(ignore_virtual=False)
        with self.assertRaises(AssertionError):
            parallel_state.is_pipeline_last_stage(ignore_virtual=False)


class TestGlobalMemoryBufferLifecycle(_ParallelStateTestBase):
    """have / get / destroy / (re)initialize track the buffer singleton."""

    def test_destroy_then_get_raises_and_have_is_false(self):
        """After destroy the buffer is absent and get asserts."""
        self._set_globals(_GLOBAL_MEMORY_BUFFER=None)
        parallel_state.destroy_global_memory_buffer()
        self.assertFalse(parallel_state.have_global_memory_buffer())
        with self.assertRaises(AssertionError):
            parallel_state.get_global_memory_buffer()

    def test_reinitialize_exposes_a_real_buffer(self):
        """_set_global_memory_buffer installs a GlobalMemoryBuffer instance."""
        self._set_globals(_GLOBAL_MEMORY_BUFFER=None)
        parallel_state._set_global_memory_buffer()
        self.assertTrue(parallel_state.have_global_memory_buffer())
        buf = parallel_state.get_global_memory_buffer()
        self.assertIsInstance(buf, GlobalMemoryBuffer)


class TestEmbeddingGroupContract(_ParallelStateTestBase):
    """get_embedding_group is an explicit not-yet-supported stub."""

    def test_get_embedding_group_raises_not_implemented(self):
        """The stub raises NotImplementedError before touching any global."""
        with self.assertRaises(NotImplementedError):
            parallel_state.get_embedding_group()


class TestKnownBugs(_ParallelStateTestBase):
    """Behaviours that look wrong against the single-card world-size contract."""

    @unittest.expectedFailure
    def test_expert_tensor_world_size_should_default_to_one(self):
        """PROBABLE BUG: get_expert_tensor_parallel_world_size returns None.

        With no expert-tp override, no expert-tp group and no tp override
        (the fresh single-card default), the fallback at
        src/paddlefleet/parallel_state.py:376-377 returns
        ``_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE`` which is ``None``. Every
        other world-size getter (e.g. get_tensor_model_parallel_world_size,
        get_context_parallel_world_size) returns the integer ``1`` for the
        equivalent uninitialized state, and callers do integer arithmetic on
        the result, so ``None`` is a defect. This asserts the correct value
        (1) and is marked expectedFailure; do NOT edit production to satisfy
        it -- flip/remove the marker once the source is fixed.
        """
        self._set_globals(
            _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE=None,
            _EXPERT_TENSOR_PARALLEL_GROUP=None,
            _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE=None,
        )
        self.assertEqual(
            parallel_state.get_expert_tensor_parallel_world_size(), 1
        )


if __name__ == "__main__":
    unittest.main()
