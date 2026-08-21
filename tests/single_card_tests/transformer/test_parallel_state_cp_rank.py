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

"""Single-card coverage for parallel_state.get_context_parallel_rank.

When context parallelism is NOT initialized (the single-card / CP=1 case)
``get_context_parallel_group()`` returns ``None`` and
``get_context_parallel_rank()`` must short-circuit to 0 without touching a
non-existent process group. This locks that behaviour and exercises the
``group is None -> return 0`` branch (parallel_state.py:231-232).
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

import paddlefleet.parallel_state as ps
from paddlefleet.parallel_state import (
    get_context_parallel_group,
    get_context_parallel_rank,
    get_context_parallel_world_size,
)


class TestContextParallelRankDisabled(unittest.TestCase):
    def test_group_is_none_when_cp_uninitialized(self) -> None:
        # check_initialized defaults to False here, so no assert is raised.
        self.assertIsNone(get_context_parallel_group())

    def test_rank_is_zero_when_cp_disabled(self) -> None:
        # Exercises the ``if get_context_parallel_group() is None: return 0``
        # branch (lines 231-232).
        self.assertEqual(get_context_parallel_rank(), 0)

    def test_world_size_is_one_when_cp_disabled(self) -> None:
        self.assertEqual(get_context_parallel_world_size(), 1)


class TestContextParallelRankEnabled(unittest.TestCase):
    """Covers the ``return get_context_parallel_group().rank`` branch
    (parallel_state.py:233) by monkeypatching the group getter to return a
    fake group with a ``.rank`` attribute (no real CP init needed).
    """

    def test_rank_reads_group_rank_when_cp_enabled(self) -> None:
        fake_group = types.SimpleNamespace(rank=0, world_size=2)
        with mock.patch.object(
            ps, "get_context_parallel_group", return_value=fake_group
        ):
            # group is not None -> line 233 returns group.rank.
            self.assertEqual(ps.get_context_parallel_rank(), 0)


if __name__ == "__main__":
    unittest.main()
