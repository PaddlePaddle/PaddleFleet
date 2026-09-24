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

"""Behavior tests for ``paddlefleet.parallel_state`` wiring and fallbacks.

This module keeps the process-group handles and mpu-override scalars for every
parallel dimension as module-level globals. ``initialize_model_parallel`` copies
the handles off a hybrid-communicate-group (hcg) object into those globals, and
the getters layer a documented fallback contract on top:

* world-size / rank getters return the explicit mpu override first, then the
  group handle, and finally the single-card default (world_size ``1`` /
  rank ``0``) when neither is set and no distributed backend is initialized;
* the pipeline-stage predicates combine the pipeline rank / world-size with an
  optional virtual-pipeline stage index;
* ``_set_global_memory_buffer`` / ``get`` / ``destroy`` form a create-once,
  fetch-cached, tear-down lifecycle guarded by assertions.

These tests exercise that real control flow on a single CPU process (the
interface-supported world_size==1 / uninitialized path). They deliberately do
NOT fake ``world_size`` and mock collectives to stand in for multi-card numerics
-- the ``initialize_model_parallel`` wiring is pure attribute assignment with no
communication, and the hcg is a plain collaborator stub whose group handles are
distinct sentinel objects so an attribute swap is caught by identity. Every
expected value is derived by hand from the contract above, never from the
implementation, and every mutated global is snapshotted and restored so shared
module state is never left polluted.

The production module imports ``paddle`` at import time, so the whole suite is
skipped with an honest reason when paddle / paddlefleet is unavailable rather
than reporting a hollow pass.
"""

import os
import sys
import unittest
import warnings

# Make the in-tree ``src/paddlefleet`` importable when the package has not been
# pip-installed (repo_root/src is 4 directory levels above this file).
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


class _Group:
    """Minimal stand-in for a paddle communication group handle.

    Only the attributes that ``parallel_state`` actually reads are provided.
    Instances are distinct objects so tests can assert identity (``assertIs``)
    and catch handle-swap wiring bugs.
    """

    def __init__(self, nranks=1):
        self.nranks = nranks


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
        sibling tests running in the same interpreter.
        """
        for name, value in overrides.items():
            self.assertTrue(
                hasattr(parallel_state, name),
                f"parallel_state has no global {name!r}",
            )
            original = getattr(parallel_state, name)
            self.addCleanup(setattr, parallel_state, name, original)
            setattr(parallel_state, name, value)


class TestInitializeModelParallelWiring(_ParallelStateTestBase):
    """``initialize_model_parallel`` maps each hcg handle to the right global."""

    # Names of every global ``initialize_model_parallel`` may assign to; all are
    # snapshotted (and reset to a clean baseline) so the wiring under test starts
    # from a known state and nothing leaks afterwards.
    _TOUCHED = (
        "_TENSOR_MODEL_PARALLEL_GROUP",
        "_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS",
        "_PIPELINE_MODEL_PARALLEL_GROUP",
        "_DATA_PARALLEL_GROUP",
        "_EXPERT_MODEL_PARALLEL_GROUP",
        "_EXPERT_DATA_PARALLEL_GROUP",
        "_CONTEXT_PARALLEL_GROUP",
        "_DATA_PARALLEL_GROUP_WITH_CP",
        "_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK",
        "_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE",
        "_GLOBAL_MEMORY_BUFFER",
    )

    def _reset_touched(self):
        self._set_globals(**dict.fromkeys(self._TOUCHED, None))

    def _make_hcg(self, pp_nranks=1):
        """Build an hcg stub whose eight handles are all distinct objects."""
        hcg = _Group()
        hcg._mp_comm_group = _Group()
        hcg._mp_group = [3, 7, 11, 15]  # distinguishable global-rank list
        hcg._pp_comm_group = _Group(nranks=pp_nranks)
        hcg._sharding_comm_group = _Group()
        hcg._ep_comm_group = _Group()
        hcg._moe_sharding_comm_group = _Group()
        hcg._cp_comm_group = _Group()
        hcg._cp_sharding_comm_group = _Group()
        return hcg

    def test_each_handle_lands_in_its_own_global(self):
        """Every hcg group handle is routed to the matching getter, no swaps."""
        self._reset_touched()
        hcg = self._make_hcg()
        parallel_state.initialize_model_parallel(hcg)

        self.assertIs(
            parallel_state.get_tensor_model_parallel_group(),
            hcg._mp_comm_group,
        )
        self.assertEqual(
            parallel_state._TENSOR_MODEL_PARALLEL_GLOBAL_RANKS,
            [3, 7, 11, 15],
        )
        self.assertIs(
            parallel_state.get_pipeline_model_parallel_group(),
            hcg._pp_comm_group,
        )
        self.assertIs(
            parallel_state.get_data_parallel_group(),
            hcg._sharding_comm_group,
        )
        self.assertIs(
            parallel_state.get_expert_model_parallel_group(),
            hcg._ep_comm_group,
        )
        self.assertIs(
            parallel_state.get_expert_data_parallel_group(),
            hcg._moe_sharding_comm_group,
        )
        self.assertIs(
            parallel_state.get_context_parallel_group(),
            hcg._cp_comm_group,
        )
        self.assertIs(
            parallel_state.get_data_parallel_group(with_context_parallel=True),
            hcg._cp_sharding_comm_group,
        )
        # The memory buffer is created as a real GlobalMemoryBuffer as a side
        # effect of initialization.
        self.assertTrue(parallel_state.have_global_memory_buffer())
        self.assertIsInstance(
            parallel_state.get_global_memory_buffer(), GlobalMemoryBuffer
        )

    def test_virtual_pipeline_size_recorded_when_pp_is_multistage(self):
        """A >1 stage pp group lets vpp size be stored and rank start at 0."""
        self._reset_touched()
        hcg = self._make_hcg(pp_nranks=2)
        parallel_state.initialize_model_parallel(
            hcg, virtual_pipeline_model_parallel_size=4
        )
        self.assertEqual(
            parallel_state.get_virtual_pipeline_model_parallel_world_size(), 4
        )
        self.assertEqual(
            parallel_state.get_virtual_pipeline_model_parallel_rank(), 0
        )

    def test_virtual_pipeline_requires_multistage_pipeline(self):
        """vpp with a single-stage pp group is rejected with RuntimeError."""
        self._reset_touched()
        hcg = self._make_hcg(pp_nranks=1)
        with self.assertRaises(RuntimeError):
            parallel_state.initialize_model_parallel(
                hcg, virtual_pipeline_model_parallel_size=4
            )


class TestWorldSizeRankFallback(_ParallelStateTestBase):
    """world-size / rank getters follow override -> group -> single-card."""

    def test_single_card_defaults_when_nothing_is_set(self):
        """No group, no override, no dist init -> world_size 1 and rank 0."""
        self._set_globals(
            _TENSOR_MODEL_PARALLEL_GROUP=None,
            _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE=None,
            _MPU_TENSOR_MODEL_PARALLEL_RANK=None,
            _PIPELINE_MODEL_PARALLEL_GROUP=None,
            _PIPELINE_MODEL_PARALLEL_WORLD_SIZE=None,
            _PIPELINE_MODEL_PARALLEL_RANK=None,
            _CONTEXT_PARALLEL_GROUP=None,
            _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE=None,
        )
        self.assertEqual(
            parallel_state.get_tensor_model_parallel_world_size(), 1
        )
        self.assertEqual(parallel_state.get_tensor_model_parallel_rank(), 0)
        self.assertEqual(
            parallel_state.get_pipeline_model_parallel_world_size(), 1
        )
        self.assertEqual(parallel_state.get_pipeline_model_parallel_rank(), 0)
        self.assertEqual(parallel_state.get_context_parallel_world_size(), 1)
        self.assertEqual(parallel_state.get_context_parallel_rank(), 0)

    def test_tensor_world_size_prefers_override_then_group_nranks(self):
        """Override wins; without it, the group's ``nranks`` is used."""
        # Group present, no override -> read group.nranks (7, not the default 1).
        self._set_globals(
            _TENSOR_MODEL_PARALLEL_GROUP=_Group(nranks=7),
            _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE=None,
        )
        self.assertEqual(
            parallel_state.get_tensor_model_parallel_world_size(), 7
        )

    def test_tensor_world_size_override_shadows_group(self):
        """An explicit mpu override takes precedence over the group handle."""
        self._set_globals(
            _TENSOR_MODEL_PARALLEL_GROUP=_Group(nranks=7),
            _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE=16,
        )
        self.assertEqual(
            parallel_state.get_tensor_model_parallel_world_size(), 16
        )

    def test_pipeline_override_world_size_forces_rank_zero_off_device(self):
        """A >1 pp world_size without dist init still yields rank 0."""
        self._set_globals(
            _PIPELINE_MODEL_PARALLEL_GROUP=None,
            _PIPELINE_MODEL_PARALLEL_WORLD_SIZE=None,
            _PIPELINE_MODEL_PARALLEL_RANK=None,
        )
        parallel_state.set_pipeline_model_parallel_world_size(4)
        self.assertEqual(
            parallel_state.get_pipeline_model_parallel_world_size(), 4
        )
        # No distributed backend is initialized in this single process, so the
        # rank getter falls through to 0 even though world_size is 4.
        self.assertEqual(parallel_state.get_pipeline_model_parallel_rank(), 0)


class TestExpertOverrides(_ParallelStateTestBase):
    """Expert tp / ep override setters feed the matching getter branch."""

    def test_expert_tensor_world_size_override_and_tp_fallback(self):
        """Override wins; absent it (and no expert group) the tp size is reused."""
        # With no expert-tp group and no expert-tp override, the getter reuses
        # the tensor-model-parallel world size for backward compatibility.
        self._set_globals(
            _EXPERT_TENSOR_PARALLEL_GROUP=None,
            _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE=None,
            _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE=4,
        )
        self.assertEqual(
            parallel_state.get_expert_tensor_parallel_world_size(), 4
        )
        # Setting the explicit expert-tp override then shadows that fallback.
        parallel_state.set_expert_tensor_parallel_world_size(8)
        self.assertEqual(
            parallel_state.get_expert_tensor_parallel_world_size(), 8
        )

    def test_expert_tensor_rank_default_and_override(self):
        """Rank defaults to 0 with no group/override, then honours the override."""
        self._set_globals(
            _EXPERT_TENSOR_PARALLEL_GROUP=None,
            _MPU_EXPERT_TENSOR_PARALLEL_RANK=None,
        )
        self.assertEqual(parallel_state.get_expert_tensor_parallel_rank(), 0)
        parallel_state.set_expert_tensor_parallel_rank(3)
        self.assertEqual(parallel_state.get_expert_tensor_parallel_rank(), 3)

    def test_expert_model_rank_override_shadows_dist_lookup(self):
        """A set expert-model rank is returned before any distributed lookup."""
        self._set_globals(_MPU_EXPERT_MODEL_PARALLEL_RANK=None)
        # Default (no override, no dist init) is 0.
        self.assertEqual(parallel_state.get_expert_model_parallel_rank(), 0)
        parallel_state.set_expert_model_parallel_rank(5)
        self.assertEqual(parallel_state.get_expert_model_parallel_rank(), 5)


class TestPipelineStagePredicates(_ParallelStateTestBase):
    """First/last-stage predicates combine pp rank with the vpp stage index."""

    def _single_stage_with_vpp(self, vpp_world_size):
        """Single-card pp (rank 0, world 1) plus a chosen vpp world size."""
        self._set_globals(
            _PIPELINE_MODEL_PARALLEL_GROUP=None,
            _PIPELINE_MODEL_PARALLEL_WORLD_SIZE=None,
            _PIPELINE_MODEL_PARALLEL_RANK=None,
            _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE=vpp_world_size,
        )

    def test_first_stage_tracks_vp_stage_zero(self):
        """With vpp enabled, only vp_stage 0 can be the first stage."""
        self._single_stage_with_vpp(4)
        # vp_stage 0 does not short-circuit, and pp rank 0 -> first stage.
        self.assertTrue(
            parallel_state.is_pipeline_first_stage(
                ignore_virtual=False, vp_stage=0
            )
        )
        # A non-zero vp_stage is never the first stage regardless of pp rank.
        self.assertFalse(
            parallel_state.is_pipeline_first_stage(
                ignore_virtual=False, vp_stage=2
            )
        )
        # ignore_virtual bypasses the vpp check and looks only at pp rank 0.
        self.assertTrue(parallel_state.is_pipeline_first_stage())

    def test_first_stage_requires_vp_stage_when_virtual_enabled(self):
        """Omitting vp_stage while vpp is enabled asserts."""
        self._single_stage_with_vpp(4)
        with self.assertRaises(AssertionError):
            parallel_state.is_pipeline_first_stage(ignore_virtual=False)

    def test_last_stage_tracks_final_vp_stage(self):
        """With vpp world size 4, only vp_stage 3 can be the last stage."""
        self._single_stage_with_vpp(4)
        # vp_stage == vpp_world_size - 1 (3) does not short-circuit; pp rank 0
        # equals the single pp last stage (world 1 -> last index 0) -> True.
        self.assertTrue(
            parallel_state.is_pipeline_last_stage(
                ignore_virtual=False, vp_stage=3
            )
        )
        # An earlier vp_stage is not the last stage.
        self.assertFalse(
            parallel_state.is_pipeline_last_stage(
                ignore_virtual=False, vp_stage=0
            )
        )

    def test_last_stage_requires_vp_stage_when_virtual_enabled(self):
        """Omitting vp_stage while vpp is enabled asserts."""
        self._single_stage_with_vpp(4)
        with self.assertRaises(AssertionError):
            parallel_state.is_pipeline_last_stage(ignore_virtual=False)


class TestVirtualPipelineRankSetter(_ParallelStateTestBase):
    """The deprecated global vpp-rank setter both warns and mutates state."""

    def test_setter_emits_deprecation_and_updates_rank(self):
        """One call must raise DeprecationWarning and store the new rank."""
        self._set_globals(_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK=None)
        self.assertIsNone(
            parallel_state.get_virtual_pipeline_model_parallel_rank()
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            parallel_state.set_virtual_pipeline_model_parallel_rank(3)
        self.assertTrue(
            any(issubclass(w.category, DeprecationWarning) for w in caught),
            "expected a DeprecationWarning from the global vpp-rank setter",
        )
        # The warning must not suppress the actual state change.
        self.assertEqual(
            parallel_state.get_virtual_pipeline_model_parallel_rank(), 3
        )


class TestEmbeddingGroupUnsupported(_ParallelStateTestBase):
    """The embedding group is an explicit not-yet-supported contract."""

    def test_get_embedding_group_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            parallel_state.get_embedding_group()


class TestGlobalMemoryBufferLifecycle(_ParallelStateTestBase):
    """create-once / fetch-cached / destroy lifecycle with a real buffer."""

    def test_full_lifecycle_and_double_init_guard(self):
        self._set_globals(_GLOBAL_MEMORY_BUFFER=None)
        # Nothing created yet.
        self.assertFalse(parallel_state.have_global_memory_buffer())
        with self.assertRaises(AssertionError):
            parallel_state.get_global_memory_buffer()

        # Create once -> a real GlobalMemoryBuffer, fetched by identity.
        parallel_state._set_global_memory_buffer()
        self.assertTrue(parallel_state.have_global_memory_buffer())
        buf = parallel_state.get_global_memory_buffer()
        self.assertIsInstance(buf, GlobalMemoryBuffer)
        self.assertIs(parallel_state.get_global_memory_buffer(), buf)

        # A second creation without teardown is rejected.
        with self.assertRaises(AssertionError):
            parallel_state._set_global_memory_buffer()

        # Teardown clears the handle and re-arms the not-initialized guard.
        parallel_state.destroy_global_memory_buffer()
        self.assertFalse(parallel_state.have_global_memory_buffer())
        with self.assertRaises(AssertionError):
            parallel_state.get_global_memory_buffer()


if __name__ == "__main__":
    unittest.main()
