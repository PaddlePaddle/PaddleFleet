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

"""Behavior tests for CPU-observable control logic in
``paddlefleet.context_parallel_utils``.

Covered production functions:
  * ``mark_context_parallel_parameter_disable_scale_grad`` /
    ``context_parallel_parameter_disable_scale_grad`` -- the type-dispatched
    setter (Layer marks weight + optional bias; Parameter/Tensor marks itself;
    anything else raises ``TypeError``) and its getter (attribute read,
    defaulting to ``False``).
  * ``scatter_balance`` -- the DualChunkSwap local slice selection: each rank
    keeps one chunk from the front and one from the back of the sequence.
  * ``scatter_contiguous`` -- the contiguous local slice selection plus its
    divisibility guard.
  * ``scatter_with_padding`` -- the per-rank ``split_sections`` computation,
    including the branch that returns an all-zero buffer for ranks that fall
    beyond the available data.

Scope / why no GPU or collective assertions here:
  ``scatter_balance``, ``scatter_contiguous`` and ``scatter_with_padding`` only
  perform *local* tensor slicing / splitting when ``group.nranks > 1``; they do
  NOT invoke any collective (all_gather / reduce_scatter live in separate
  functions). This file therefore verifies exactly the single-rank-local
  slice-selection contract -- which rows a given rank keeps for a given
  ``(nranks, rank)`` topology -- using a plain ``SimpleNamespace`` as the group
  collaborator (a genuine not-under-test object that only carries the topology
  integers). It makes NO claim about cross-rank communication, gather/reorder
  reassembly, or GPU numerics; those require a real process group and belong in
  the multi-card suite.

Every expected tensor below is hand-derived from the documented formula,
independently of the implementation; distinguishable ``arange`` content is used
so a wrong slice origin, reversed front/back selection, or swapped rank mapping
changes the result and fails the assertion.

Paddle is a heavy/optional dependency. When it is unavailable the whole module
skips with an honest reason rather than silently passing.
"""

import os
import sys
import unittest
from types import SimpleNamespace

# The repository uses a ``src`` layout (pyproject ``where = ["src"]``); make
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
        context_parallel_parameter_disable_scale_grad,
        mark_context_parallel_parameter_disable_scale_grad,
        scatter_balance,
        scatter_contiguous,
        scatter_with_padding,
    )

    # All functions exercised here are pure CPU tensor ops; pin the device so
    # the file runs in a no-accelerator environment.
    paddle.set_device("cpu")
    _PADDLE_AVAILABLE = True
    _SKIP_REASON = ""
except ImportError as exc:  # pragma: no cover - exercised only without paddle
    _PADDLE_AVAILABLE = False
    _SKIP_REASON = f"paddle / paddlefleet not importable: {exc}"


def _group(nranks, rank):
    """A minimal stand-in for a communication group.

    Only ``nranks`` / ``rank`` are read by the local slice logic under test, so
    a namespace with real integers is used (rather than a mock, which would
    return mock objects from arithmetic and hide bugs).
    """
    return SimpleNamespace(nranks=nranks, rank=rank)


@unittest.skipUnless(_PADDLE_AVAILABLE, _SKIP_REASON)
class TestMarkDisableScaleGrad(unittest.TestCase):
    """Type-dispatched marking + the getter's read-back / default."""

    def test_layer_marks_weight_and_bias(self):
        layer = paddle.nn.Linear(4, 3)
        # Before marking, the getter must report the documented default False.
        self.assertFalse(
            context_parallel_parameter_disable_scale_grad(layer.weight)
        )
        self.assertFalse(
            context_parallel_parameter_disable_scale_grad(layer.bias)
        )

        mark_context_parallel_parameter_disable_scale_grad(layer)

        # Both weight and bias must now be flagged, read through the real getter.
        self.assertTrue(
            context_parallel_parameter_disable_scale_grad(layer.weight)
        )
        self.assertTrue(
            context_parallel_parameter_disable_scale_grad(layer.bias)
        )

    def test_layer_without_bias_marks_only_weight(self):
        layer = paddle.nn.Linear(4, 3, bias_attr=False)
        self.assertIsNone(layer.bias)

        mark_context_parallel_parameter_disable_scale_grad(layer)

        self.assertTrue(
            context_parallel_parameter_disable_scale_grad(layer.weight)
        )

    def test_parameter_marks_itself(self):
        # A raw tensor takes the Parameter/Tensor branch and is marked directly.
        tensor = paddle.zeros([2, 2], dtype="float32")
        self.assertFalse(context_parallel_parameter_disable_scale_grad(tensor))

        mark_context_parallel_parameter_disable_scale_grad(tensor)

        self.assertTrue(context_parallel_parameter_disable_scale_grad(tensor))

    def test_getter_default_is_false(self):
        tensor = paddle.ones([3], dtype="float32")
        # Fresh tensor never marked -> getattr default path returns False.
        self.assertFalse(context_parallel_parameter_disable_scale_grad(tensor))

    def test_invalid_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            mark_context_parallel_parameter_disable_scale_grad("not_a_param")


@unittest.skipUnless(_PADDLE_AVAILABLE, _SKIP_REASON)
class TestScatterBalanceLocalSelection(unittest.TestCase):
    """``scatter_balance`` front/back chunk selection (no collective involved).

    For ``parallelism`` ranks and ``seq_len`` along ``axis``:
        interval = seq_len // parallelism // 2
        front = input[interval*rank : interval*(rank+1)]
        back  = input[L - interval*(rank+1) : L - interval*rank]
        out   = concat([front, back])
    """

    def test_single_rank_returns_independent_clone(self):
        x = paddle.arange(12, dtype="float32").reshape([4, 3])
        out = scatter_balance(x, group=_group(nranks=1, rank=0), axis=0)
        # Content preserved and a distinct object (clone), not the same buffer.
        np.testing.assert_array_equal(out.numpy(), x.numpy())
        self.assertNotEqual(id(out), id(x))

    def test_two_ranks_take_front_and_back_chunks(self):
        # seq_len=8, parallelism=2 -> interval = 8//2//2 = 2.
        # rank0: front rows [0,1], back rows [6,7] -> rows [0,1,6,7]
        # rank1: front rows [2,3], back rows [4,5] -> rows [2,3,4,5]
        x = paddle.arange(8 * 2, dtype="float32").reshape([8, 2])
        rows = x.numpy()

        out0 = scatter_balance(x, group=_group(nranks=2, rank=0), axis=0)
        out1 = scatter_balance(x, group=_group(nranks=2, rank=1), axis=0)

        np.testing.assert_array_equal(out0.numpy(), rows[[0, 1, 6, 7], :])
        np.testing.assert_array_equal(out1.numpy(), rows[[2, 3, 4, 5], :])
        # Together the two ranks partition the whole sequence exactly once.
        combined = np.concatenate([out0.numpy(), out1.numpy()], axis=0)
        np.testing.assert_array_equal(
            np.sort(combined[:, 0]), np.sort(rows[:, 0])
        )

    def test_axis1_selection(self):
        # Same rule along axis=1: seq_len=8, interval=2.
        x = paddle.arange(2 * 8, dtype="float32").reshape([2, 8])
        cols = x.numpy()
        out0 = scatter_balance(x, group=_group(nranks=2, rank=0), axis=1)
        # rank0 front cols [0,1], back cols [6,7].
        np.testing.assert_array_equal(out0.numpy(), cols[:, [0, 1, 6, 7]])

    def test_indivisible_seq_len_raises(self):
        # seq_len=6, parallelism*2=4, 6 % 4 != 0 -> assertion fires.
        x = paddle.arange(6 * 2, dtype="float32").reshape([6, 2])
        with self.assertRaises(AssertionError):
            scatter_balance(x, group=_group(nranks=2, rank=0), axis=0)


@unittest.skipUnless(_PADDLE_AVAILABLE, _SKIP_REASON)
class TestScatterContiguousLocalSelection(unittest.TestCase):
    """``scatter_contiguous``: rank r keeps ``input[r*chunk:(r+1)*chunk]``."""

    def test_single_rank_returns_independent_clone(self):
        x = paddle.arange(6, dtype="float32").reshape([3, 2])
        out = scatter_contiguous(x, group=_group(nranks=1, rank=0), axis=0)
        np.testing.assert_array_equal(out.numpy(), x.numpy())
        self.assertNotEqual(id(out), id(x))

    def test_two_ranks_take_contiguous_halves(self):
        # length=8, chunk=4: rank0 rows [0..3], rank1 rows [4..7].
        x = paddle.arange(8 * 2, dtype="float32").reshape([8, 2])
        rows = x.numpy()
        out0 = scatter_contiguous(x, group=_group(nranks=2, rank=0), axis=0)
        out1 = scatter_contiguous(x, group=_group(nranks=2, rank=1), axis=0)
        np.testing.assert_array_equal(out0.numpy(), rows[0:4, :])
        np.testing.assert_array_equal(out1.numpy(), rows[4:8, :])

    def test_non_divisible_length_raises_value_error(self):
        # length=7 not divisible by nranks=2 -> explicit ValueError guard.
        x = paddle.arange(7 * 2, dtype="float32").reshape([7, 2])
        with self.assertRaises(ValueError):
            scatter_contiguous(x, group=_group(nranks=2, rank=0), axis=0)


@unittest.skipUnless(_PADDLE_AVAILABLE, _SKIP_REASON)
class TestScatterWithPaddingSplit(unittest.TestCase):
    """``scatter_with_padding`` split-section + zero-buffer selection.

    avg_num = (total + num_pad) // cp_degree; each rank of ``rank_idx`` covered
    ranks gets its contiguous section (last covered rank right-padded by
    ``rank_pad``); ranks beyond the data get an all-zero buffer of length
    ``avg_num``.
    """

    def test_no_pad_even_split_content(self):
        # total=8, num_pad=0, cp_degree=2 -> avg_num=4, sections=[4,4].
        x = paddle.arange(8 * 2, dtype="float32").reshape([8, 2])
        rows = x.numpy()
        out0 = scatter_with_padding(
            x, num_pad=0, axis=0, group=_group(nranks=2, rank=0)
        )
        out1 = scatter_with_padding(
            x, num_pad=0, axis=0, group=_group(nranks=2, rank=1)
        )
        np.testing.assert_array_equal(out0.numpy(), rows[0:4, :])
        np.testing.assert_array_equal(out1.numpy(), rows[4:8, :])

    def test_rank_beyond_data_gets_zero_buffer(self):
        # total=2, num_pad=1, cp_degree=3 -> avg_num=(2+1)//3=1.
        #   rank0 section [0:1], rank1 section [1:2] cover all data (rank_idx=2)
        #   rank2 falls beyond -> all-zero buffer of shape [avg_num=1, 2].
        x = paddle.arange(2 * 2, dtype="float32").reshape([2, 2])
        rows = x.numpy()

        out0 = scatter_with_padding(
            x, num_pad=1, axis=0, group=_group(nranks=3, rank=0)
        )
        out1 = scatter_with_padding(
            x, num_pad=1, axis=0, group=_group(nranks=3, rank=1)
        )
        out2 = scatter_with_padding(
            x, num_pad=1, axis=0, group=_group(nranks=3, rank=2)
        )

        np.testing.assert_array_equal(out0.numpy(), rows[0:1, :])
        np.testing.assert_array_equal(out1.numpy(), rows[1:2, :])
        # The overflow rank must be zeros (content, not just shape).
        self.assertEqual(list(out2.shape), [1, 2])
        np.testing.assert_array_equal(
            out2.numpy(), np.zeros([1, 2], dtype=np.float32)
        )


if __name__ == "__main__":
    unittest.main()
