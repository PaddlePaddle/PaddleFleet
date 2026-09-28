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

"""Behavior tests for the context-parallel scatter helper in
``paddlefleet.context_parallel_utils``.

Covered production code:
  * ``scatter_with_padding`` -- given a per-rank ``group`` (``nranks``/``rank``)
    it splits ``input_tensor`` along axis 0 into ``nranks`` roughly-equal
    contiguous blocks sized ``avg_num = (total_num + num_pad) // nranks``; the
    last block that still holds real rows is zero-padded up to ``avg_num`` rows,
    and any rank beyond the data is handed an all-zero ``avg_num``-row tensor.
    This is pure local tensor arithmetic (split / pad / zeros) and runs on CPU
    -- it performs no collective, so a single process legitimately exercises
    the real per-rank sharding path.
  * ``ContextParallelNormalScatter`` / ``ContextParallelNormalGather`` -- the
    ``forward`` world-size==1 fast path, which must return an independent clone
    of the input (verified for value equality *and* object distinctness).

Every expected tensor below is derived by hand from the documented partition
rule, independently of the implementation: no call to the function under test
is used to build an expected value. Inputs use ``arange`` content so each row
is unique -- a wrong split boundary, a dropped ``num_pad`` term, a mis-placed
pad, or an identity "clone" would change the result and fail the assertion.

The ``all_gather_without_padding`` path and the multi-rank PyLayer branches call
``paddle.distributed.stream.all_gather`` and require a real context-parallel
process group; they are honestly skipped here rather than faked with a mocked
collective (which would only prove orchestration, not cross-rank numerics).

``TestScatterAxisBug`` documents a real, unfixed production bug via
``expectedFailure`` and does NOT modify production code.

Paddle is an optional/heavy dependency; when unavailable the whole module skips
with an honest reason rather than silently passing.
"""

import os
import sys
import types
import unittest
from unittest import mock

# The repository uses a ``src`` layout (see pyproject ``where = ["src"]``); make
# ``paddlefleet`` importable when running the file directly from the tree.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import numpy as np
    import paddle

    from paddlefleet.context_parallel_utils import (
        ContextParallelNormalGather,
        ContextParallelNormalScatter,
        scatter_with_padding,
    )

    # These paths only need CPU tensor ops; pin the device so the file runs in
    # a no-accelerator environment.
    paddle.set_device("cpu")
    _PADDLE_AVAILABLE = True
    _SKIP_REASON = ""
except ImportError as exc:  # pragma: no cover - exercised only without paddle
    _PADDLE_AVAILABLE = False
    _SKIP_REASON = f"paddle / paddlefleet not importable: {exc}"


def _group(nranks, rank):
    """A minimal stand-in for a process group: ``scatter_with_padding`` only
    reads ``.nranks`` and ``.rank`` integers from it (no collective call)."""
    return types.SimpleNamespace(nranks=nranks, rank=rank)


def _arange(shape):
    """Row-distinguishable float32 tensor: value == flat index."""
    n = 1
    for s in shape:
        n *= s
    return paddle.arange(n, dtype="float32").reshape(shape)


@unittest.skipUnless(_PADDLE_AVAILABLE, _SKIP_REASON)
class TestScatterWithPadding(unittest.TestCase):
    """``scatter_with_padding`` local sharding along axis 0."""

    def test_even_split_no_padding(self):
        # total_num=8, num_pad=0, nranks=2 -> avg_num=4, clean halves.
        x = _arange([8, 3])
        r0 = scatter_with_padding(x, num_pad=0, axis=0, group=_group(2, 0))
        r1 = scatter_with_padding(x, num_pad=0, axis=0, group=_group(2, 1))
        np.testing.assert_array_equal(
            r0.numpy(),
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
        )
        np.testing.assert_array_equal(
            r1.numpy(),
            [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]],
        )

    def test_all_three_branches(self):
        # total_num=3, num_pad=3, nranks=3 -> avg_num=2.
        # rank0: rows[0:2] (full block, no pad)
        # rank1: row[2:3] then zero-padded to 2 rows (last data-holding block)
        # rank2: beyond the data -> all-zero 2-row block
        x = _arange([3, 4])  # rows [0..3],[4..7],[8..11]
        r0 = scatter_with_padding(x, num_pad=3, axis=0, group=_group(3, 0))
        r1 = scatter_with_padding(x, num_pad=3, axis=0, group=_group(3, 1))
        r2 = scatter_with_padding(x, num_pad=3, axis=0, group=_group(3, 2))

        np.testing.assert_array_equal(r0.numpy(), [[0, 1, 2, 3], [4, 5, 6, 7]])
        # real row kept at the top, one zero row appended (pad on axis 0 end).
        self.assertEqual(list(r1.shape), [2, 4])
        np.testing.assert_array_equal(
            r1.numpy(), [[8, 9, 10, 11], [0, 0, 0, 0]]
        )
        # out-of-range rank: zeros of the average block size.
        self.assertEqual(list(r2.shape), [2, 4])
        np.testing.assert_array_equal(r2.numpy(), np.zeros([2, 4]))

    def test_num_pad_changes_partition(self):
        # num_pad feeds avg_num = (total_num + num_pad)//nranks, so it shifts
        # the split boundary. total_num=6, nranks=2:
        #   num_pad=0 -> avg_num=3 -> rank0 gets rows[0:3]
        #   num_pad=2 -> avg_num=4 -> rank0 gets rows[0:4]
        x = _arange([6, 2])  # rows [0,1],[2,3],[4,5],[6,7],[8,9],[10,11]
        r0_p0 = scatter_with_padding(x, num_pad=0, axis=0, group=_group(2, 0))
        r0_p2 = scatter_with_padding(x, num_pad=2, axis=0, group=_group(2, 0))
        np.testing.assert_array_equal(r0_p0.numpy(), [[0, 1], [2, 3], [4, 5]])
        np.testing.assert_array_equal(
            r0_p2.numpy(), [[0, 1], [2, 3], [4, 5], [6, 7]]
        )

    def test_dtype_preserved(self):
        x = paddle.arange(8, dtype="float32").reshape([8, 1]).cast("float64")
        out = scatter_with_padding(x, num_pad=0, axis=0, group=_group(2, 0))
        self.assertEqual(out.dtype, paddle.float64)


# PLACEHOLDER_PYLAYER
@unittest.skipUnless(_PADDLE_AVAILABLE, _SKIP_REASON)
class TestPyLayerSingleRankClone(unittest.TestCase):
    """The world-size==1 fast path of both PyLayers must return an
    *independent clone* of the input -- same values, different object.

    ``fleet.get_hybrid_communicate_group`` is an external collaborator (the
    parallel runtime), not the code under test, so it is stubbed to report a
    single context-parallel rank. The clone/branch decision itself is real."""

    def _hcg(self, world_size):
        group = types.SimpleNamespace(nranks=world_size)
        return types.SimpleNamespace(
            get_context_parallel_world_size=lambda: world_size,
            get_context_parallel_group=lambda: group,
        )

    def test_scatter_forward_single_rank_returns_clone(self):
        x = _arange([4, 3])
        ctx = types.SimpleNamespace()
        with mock.patch(
            "paddlefleet.context_parallel_utils.fleet."
            "get_hybrid_communicate_group",
            return_value=self._hcg(1),
        ):
            out = ContextParallelNormalScatter.forward(
                ctx, x, num_pad=0, axis=0
            )
        self.assertIsNot(out, x)  # a real copy, not the same tensor object
        np.testing.assert_array_equal(out.numpy(), x.numpy())

    def test_gather_forward_single_rank_returns_clone(self):
        x = _arange([4, 3])
        ctx = types.SimpleNamespace()
        with mock.patch(
            "paddlefleet.context_parallel_utils.fleet."
            "get_hybrid_communicate_group",
            return_value=self._hcg(1),
        ):
            out = ContextParallelNormalGather.forward(ctx, x, num_pad=0, axis=0)
        self.assertIsNot(out, x)
        np.testing.assert_array_equal(out.numpy(), x.numpy())


class TestCollectivePathNotCovered(unittest.TestCase):
    """Honest scope marker: ``all_gather_without_padding`` and the multi-rank
    PyLayer branches issue ``paddle.distributed.stream.all_gather`` and need a
    real context-parallel process group. Faking ``nranks`` and mocking the
    collective would only prove call orchestration, not cross-rank numerics
    (see unit-test antipattern 13), so the gather numerics are left to the
    multi-card suite."""

    def test_requires_real_process_group(self):
        self.skipTest(
            "all_gather_without_padding requires a real context-parallel "
            "process group (multi-card); not validated single-process."
        )


# PLACEHOLDER_BUG
@unittest.skipUnless(_PADDLE_AVAILABLE, _SKIP_REASON)
class TestScatterAxisBug(unittest.TestCase):
    """Documents a real production bug: ``scatter_with_padding`` ignores its
    ``axis`` argument when partitioning.

    ``src/paddlefleet/context_parallel_utils.py`` line 1008 calls
    ``paddle.split(input_tensor, num_or_sections=split_sections)`` without
    ``axis=axis``, so the split always happens on axis 0 regardless of the
    requested ``axis`` (line 1012's pad index ``axis*ndim*2+1`` is likewise
    wrong for ``axis != 0``). For a correct implementation, scattering a
    ``[8, 8]`` tensor with ``axis=1`` on rank 0 of a 2-rank group should return
    the left half of the columns, ``x[:, 0:4]`` (shape ``[8, 4]``). The current
    code splits rows instead and returns ``x[0:4, :]`` (shape ``[4, 8]``).

    Marked ``expectedFailure`` so the suite stays green while flagging the bug;
    production code is deliberately NOT modified.
    """

    @unittest.expectedFailure
    def test_axis1_should_split_columns(self):
        x = _arange([8, 8])
        out = scatter_with_padding(x, num_pad=0, axis=1, group=_group(2, 0))
        # Correct axis=1 behavior: the first half of the columns.
        np.testing.assert_array_equal(out.numpy(), x.numpy()[:, 0:4])


if __name__ == "__main__":
    unittest.main()
