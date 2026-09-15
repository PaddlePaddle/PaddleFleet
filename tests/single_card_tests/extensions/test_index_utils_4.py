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

"""Behavior tests for the host-side *input contract* of
``paddlefleet_ops._extensions.flashmask.index_utils.prepare_maxmin``.

Distinct facet (siblings base/_3 already pin num_chunks ceiling division,
the (ceil(seq_len/512), bsz*num_heads) launch grid, output shape/dtype and
the forwarded kernel args). This file instead pins two host-observable
contracts that run *before / around* the GPU kernel and that the siblings do
not exercise:

  1. Rank contract -- ``bsz, num_heads, seq_len = input.shape`` requires a
     rank-3 input. A rank-2 or rank-4 tensor must raise ``ValueError`` at the
     unpack, before any kernel launch. This is pure host logic.
  2. Allocation identity / immutability -- the two returned tensors are
     freshly allocated buffers, distinct objects from each other AND from the
     input; the wrapper does not alias or mutate the caller's input tensor.

The GPU-only Triton kernel ``scan_maxmin_chunked`` is a non-tested
collaborator here; it is replaced by a spy that records launches and performs
no device work (see unit-test-antipatterns.md type 3). Everything asserted is
hand-derived. The kernel's own numeric max/min behavior is GPU-only and is
explicitly NOT claimed here.
"""

import os
import sys
import unittest
from unittest import mock

# Make the in-tree package importable when not pip-installed.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_PKG_SRC = os.path.join(_REPO_ROOT, "packages", "paddlefleet_ops", "src")
if _PKG_SRC not in sys.path:
    sys.path.insert(0, _PKG_SRC)

# paddlefleet_ops imports ``paddle`` and ``triton`` at import time; the local
# CPU env may have neither. Catch only ImportError/ModuleNotFoundError (an
# honest missing dependency) and skip with the real reason; never swallow
# other errors (that would hide a real regression -- antipattern type 10).
try:
    import paddle
    from paddlefleet_ops._extensions.flashmask import index_utils as _iu

    _IMPORT_ERROR = None
except ImportError as exc:  # ModuleNotFoundError is a subclass
    paddle = None
    _iu = None
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    "paddle / paddlefleet_ops flashmask index_utils not importable: "
    f"{_IMPORT_ERROR!r}"
    if _IMPORT_ERROR is not None
    else ""
)

# BN is a module-private launch constant baked into prepare_maxmin.
_EXPECTED_BN = 512


class _KernelSpy:
    """Stand-in for the GPU-only Triton kernel ``scan_maxmin_chunked``.

    Production launches it as ``scan_maxmin_chunked[grid](*args, **kwargs)``.
    This spy records the grid captured by ``__getitem__`` and the launch
    args/kwargs, and performs no device work (returns None like the real
    kernel, which writes its outputs in place). It never touches the input or
    output buffers, so any change to the input after a call would be the
    wrapper's own doing.
    """

    def __init__(self):
        self.grid = None
        self.args = None
        self.kwargs = None
        self.call_count = 0

    def __getitem__(self, grid):
        self.grid = grid

        def _launch(*args, **kwargs):
            self.call_count += 1
            self.args = args
            self.kwargs = kwargs

        return _launch


@unittest.skipUnless(_iu is not None, _SKIP_REASON)
class TestPrepareMaxminInputRankContract(unittest.TestCase):
    """The wrapper unpacks ``input.shape`` into exactly (bsz, num_heads,
    seq_len); wrong-rank inputs must fail at the host unpack, not reach the
    kernel. Runs on CPU with the kernel spied out.
    """

    def setUp(self):
        paddle.set_device("cpu")

    def test_rank2_input_raises_before_launch(self):
        """A rank-2 tensor cannot supply (bsz, num_heads, seq_len).

        ``a, b, c = shape_of_len_2`` raises ValueError ("not enough values to
        unpack") on the very first line, so the kernel must never launch.
        """
        spy = _KernelSpy()
        x = paddle.zeros([4, 8], dtype=paddle.int32)  # rank 2
        with mock.patch.object(_iu, "scan_maxmin_chunked", spy):
            with self.assertRaises(ValueError):
                _iu.prepare_maxmin(x, chunk_size=4)
        self.assertEqual(spy.call_count, 0)

    def test_rank4_input_raises_before_launch(self):
        """A rank-4 tensor over-fills the 3-tuple unpack.

        ``a, b, c = shape_of_len_4`` raises ValueError ("too many values to
        unpack"); again the kernel must never launch.
        """
        spy = _KernelSpy()
        x = paddle.zeros([2, 3, 4, 5], dtype=paddle.int32)  # rank 4
        with mock.patch.object(_iu, "scan_maxmin_chunked", spy):
            with self.assertRaises(ValueError):
                _iu.prepare_maxmin(x, chunk_size=4)
        self.assertEqual(spy.call_count, 0)

    def test_rank3_input_launches_exactly_once(self):
        """Control: a valid rank-3 input reaches the kernel exactly once.

        This distinguishes "unpack rejected the input" from "the wrapper never
        launches at all". Hand-derived for bsz=2, num_heads=7, seq_len=20,
        chunk_size=8:
          num_chunks = ceil(20/8) = 3
          grid       = (ceil(20/512), bsz*num_heads) = (1, 14)
        """
        spy = _KernelSpy()
        x = paddle.zeros([2, 7, 20], dtype=paddle.int32)  # rank 3
        with mock.patch.object(_iu, "scan_maxmin_chunked", spy):
            out_max, out_min = _iu.prepare_maxmin(x, chunk_size=8)
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(spy.grid, (1, 14))
        # bsz and num_heads pass through to the output leading dims WITHOUT
        # being swapped (2 != 7 makes a transpose observable).
        self.assertEqual(list(out_max.shape), [2, 7, 3])
        self.assertEqual(list(out_min.shape), [2, 7, 3])


@unittest.skipUnless(_iu is not None, _SKIP_REASON)
class TestPrepareMaxminAllocationIdentity(unittest.TestCase):
    """Outputs are fresh, distinct buffers and the input is left untouched."""

    def setUp(self):
        paddle.set_device("cpu")

    def _run(self, bsz, num_heads, seq_len, chunk_size):
        spy = _KernelSpy()
        # Distinguishable, non-degenerate input content so mutation would be
        # visible (all-zero would hide an accidental zero-fill).
        x = paddle.arange(bsz * num_heads * seq_len, dtype="int32").reshape(
            [bsz, num_heads, seq_len]
        )
        with mock.patch.object(_iu, "scan_maxmin_chunked", spy):
            out_max, out_min = _iu.prepare_maxmin(x, chunk_size=chunk_size)
        return x, out_max, out_min, spy

    def test_outputs_are_distinct_from_each_other_and_input(self):
        """max, min and input are three separate tensor objects.

        A wrapper that aliased max onto min (or returned the input) would
        corrupt results once the kernel writes both buffers.
        """
        x, out_max, out_min, _ = self._run(2, 3, 16, 4)
        self.assertIsNot(out_max, out_min)
        self.assertIsNot(out_max, x)
        self.assertIsNot(out_min, x)

    def test_input_content_not_mutated_by_wrapper(self):
        """The host wrapper only reads input.shape; it must not write input.

        With the kernel spied out (it never touches buffers), the input's
        contents must equal a hand-built independent reference after the call.
        """
        bsz, num_heads, seq_len = 2, 3, 16
        expected = list(range(bsz * num_heads * seq_len))  # hand-derived arange
        x, _out_max, _out_min, spy = self._run(bsz, num_heads, seq_len, 4)
        self.assertEqual(spy.call_count, 1)
        # The exact object the wrapper forwarded to the kernel is our input.
        self.assertIs(spy.args[0], x)
        self.assertEqual(x.flatten().tolist(), expected)

    def test_forwarded_output_buffers_are_the_returned_ones(self):
        """The kernel receives the very tensors handed back to the caller.

        Positional wiring is (input, output_max, output_min, ...); identity
        (not just equality) ensures no defensive copy is silently returned.
        """
        _x, out_max, out_min, spy = self._run(1, 2, 32, 8)
        self.assertIs(spy.args[1], out_max)
        self.assertIs(spy.args[2], out_min)
        # BN constant is pinned; chunk_size passes through verbatim.
        self.assertEqual(spy.kwargs["BN"], _EXPECTED_BN)
        self.assertEqual(spy.kwargs["chunk_size"], 8)


if __name__ == "__main__":
    unittest.main()
