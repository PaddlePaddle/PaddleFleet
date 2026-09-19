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

"""Behavior tests for pipeline-stage selection in ``paddlefleet.parallel_state``.

The real logic under test is the "am I the first / last pipeline stage" decision
consumed all over the distributed training stack (embedding sharing, loss on the
last stage, pipeline input preparation). Two entry points implement it:

* ``is_pipeline_first_stage(ignore_virtual, vp_stage)`` -> ``True`` only when the
  caller's pipeline-parallel rank is ``0``. When interleaved / virtual pipeline is
  enabled and ``ignore_virtual=False``, it is additionally gated on ``vp_stage``
  being the *first* virtual chunk (``vp_stage == 0``); a non-zero chunk on pp
  rank 0 is therefore NOT the global first stage.
* ``is_pipeline_last_stage(ignore_virtual, vp_stage)`` -> ``True`` only when the
  rank equals ``pp_world_size - 1``. Under virtual pipeline it is additionally
  gated on ``vp_stage == vpp_world_size - 1``; the last pp rank running an earlier
  chunk is NOT the global last stage.

Both also assert that ``vp_stage`` is supplied whenever the virtual-pipeline gate
is active (``ignore_virtual=False`` and a virtual size is set).

Expected values below are truth tables derived by hand from that contract, not
from the implementation. Only the topology *globals* are set (with cleanup that
restores the originals even on failure) so the pure rank/stage arithmetic runs
single-process; no collective is invoked and no ``world_size`` is faked to stand
in for real multi-card numerics -- these assertions concern only the locally
observable stage-selection booleans, which is exactly the world-size-overridable
path the single-card environment is meant to cover.

The production module imports ``paddle`` transitively at import time, so the whole
suite is skipped with an honest reason when Paddle / paddlefleet is unavailable on
the host rather than reporting a hollow pass.
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

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet not installed on this host
    parallel_state = None
    _IMPORT_ERROR = exc


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}",
)
class TestPipelineStageSelection(unittest.TestCase):
    """First/last pipeline-stage decisions, plain and interleaved."""

    def _set_topology(self, pp_world, pp_rank, vpp_world=None):
        """Override the pipeline-topology globals, restoring them on teardown.

        The getters read these module globals first, so setting them lets the
        stage-selection arithmetic run in-process without any collective. Every
        override is paired with an ``addCleanup`` so a failing assertion cannot
        leak a fake topology into sibling tests.
        """
        for name, value in (
            ("_PIPELINE_MODEL_PARALLEL_WORLD_SIZE", pp_world),
            ("_PIPELINE_MODEL_PARALLEL_RANK", pp_rank),
            ("_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE", vpp_world),
        ):
            original = getattr(parallel_state, name)
            self.addCleanup(setattr, parallel_state, name, original)
            setattr(parallel_state, name, value)

    def test_single_stage_is_both_first_and_last(self):
        """With one pipeline stage, rank 0 is simultaneously first and last."""
        self._set_topology(pp_world=1, pp_rank=0)
        self.assertTrue(parallel_state.is_pipeline_first_stage())
        self.assertTrue(parallel_state.is_pipeline_last_stage())

    def test_plain_pipeline_boundaries_by_rank(self):
        """Across a 4-stage pipeline only the ends are first/last; the middle
        rank is neither. Catches an off-by-one such as using ``world_size``
        instead of ``world_size - 1`` for the last stage."""
        # First rank.
        self._set_topology(pp_world=4, pp_rank=0)
        self.assertTrue(parallel_state.is_pipeline_first_stage())
        self.assertFalse(parallel_state.is_pipeline_last_stage())

    def test_plain_pipeline_middle_rank_is_neither(self):
        self._set_topology(pp_world=4, pp_rank=1)
        self.assertFalse(parallel_state.is_pipeline_first_stage())
        self.assertFalse(parallel_state.is_pipeline_last_stage())

    def test_plain_pipeline_final_rank_is_last_only(self):
        self._set_topology(pp_world=4, pp_rank=3)
        self.assertFalse(parallel_state.is_pipeline_first_stage())
        self.assertTrue(parallel_state.is_pipeline_last_stage())

    def test_interleave_first_stage_needs_chunk_zero(self):
        """Interleaved (virtual) pipeline: pp rank 0 is the global first stage
        only while executing virtual chunk 0. A later chunk on the same rank
        must report ``False`` -- the interleave gate, not just the rank."""
        # pp_world=4, vpp_world=2, on the first pipeline rank.
        self._set_topology(pp_world=4, pp_rank=0, vpp_world=2)
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

    def test_interleave_first_stage_needs_rank_zero(self):
        """Chunk 0 alone is not enough: a non-zero pipeline rank running the
        first virtual chunk is still not the global first stage."""
        self._set_topology(pp_world=4, pp_rank=1, vpp_world=2)
        self.assertFalse(
            parallel_state.is_pipeline_first_stage(
                ignore_virtual=False, vp_stage=0
            )
        )

    def test_interleave_last_stage_needs_final_chunk(self):
        """Interleaved pipeline: the last pipeline rank is the global last stage
        only on the final virtual chunk (``vpp_world - 1``). An earlier chunk on
        that same rank must report ``False``."""
        self._set_topology(pp_world=4, pp_rank=3, vpp_world=2)
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

    def test_interleave_last_stage_needs_final_rank(self):
        """Final virtual chunk on a non-final pipeline rank is not the last
        stage."""
        self._set_topology(pp_world=4, pp_rank=2, vpp_world=2)
        self.assertFalse(
            parallel_state.is_pipeline_last_stage(
                ignore_virtual=False, vp_stage=1
            )
        )

    def test_ignore_virtual_bypasses_chunk_gate(self):
        """With ``ignore_virtual=True`` (the default) the virtual gate is
        skipped entirely: the decision is purely rank-based even when a virtual
        size is configured and no ``vp_stage`` is given."""
        self._set_topology(pp_world=4, pp_rank=0, vpp_world=2)
        self.assertTrue(parallel_state.is_pipeline_first_stage())
        self.assertFalse(parallel_state.is_pipeline_last_stage())

    def test_virtual_gate_requires_vp_stage(self):
        """When the virtual gate is active (``ignore_virtual=False`` and a
        virtual size set) omitting ``vp_stage`` is a contract violation and must
        raise rather than silently defaulting."""
        self._set_topology(pp_world=4, pp_rank=0, vpp_world=2)
        with self.assertRaises(AssertionError):
            parallel_state.is_pipeline_first_stage(ignore_virtual=False)
        with self.assertRaises(AssertionError):
            parallel_state.is_pipeline_last_stage(ignore_virtual=False)


if __name__ == "__main__":
    unittest.main()
