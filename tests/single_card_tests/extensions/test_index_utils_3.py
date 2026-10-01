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

# Behavior tests for the CPU-observable dispatch/orchestration of
# paddlefleet_ops/_extensions/flashmask/index_utils.py::prepare_maxmin.
#
# Distinct facet: this file does NOT re-check the numeric max/min result the
# GPU kernel produces. It pins the wrapper's launch orchestration that runs on
# the host before the kernel fires:
#   * num_chunks = ceil(seq_len / chunk_size)               (ceiling division)
#   * launch grid = (ceil(seq_len / BN), bsz * num_heads)   with BN == 512
#   * output_max / output_min allocated as
#       [bsz, num_heads, num_chunks] int32
#   * the two allocated tensors are the exact objects passed to the kernel and
#     returned to the caller (identity), in (max, min) order.
#
# The GPU-only triton kernel `scan_maxmin_chunked` is a non-tested collaborator
# here, so it is replaced by a spy that records the launch grid and arguments
# (see unit-test-antipatterns.md type 3: mock the collaborator, observe the
# real orchestration). This makes the wrapper's host logic CPU-observable
# without a GPU; the kernel's own numeric behavior is explicitly NOT claimed.

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

# paddlefleet_ops imports paddle and triton at import time. The local CPU env
# may have neither. Catch only ImportError/ModuleNotFoundError (honest missing
# dependency) and skip with the real reason; do NOT swallow other errors.
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


class _KernelSpy:
    """Stand-in for the triton kernel object `scan_maxmin_chunked`.

    Production launches it as ``scan_maxmin_chunked[grid](*args, **kwargs)``.
    This spy records the grid passed via ``__getitem__`` and the positional /
    keyword arguments passed to the resulting launcher, so the wrapper's real
    dispatch decisions can be asserted on CPU.
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
            # Real kernel writes into output_max/output_min in place and returns
            # None; mimic the None return without touching the buffers.

        return _launch


@unittest.skipUnless(_iu is not None, _SKIP_REASON)
class TestPrepareMaxminDispatch(unittest.TestCase):
    """CPU-observable orchestration of prepare_maxmin."""

    def _run(self, bsz, num_heads, seq_len, chunk_size):
        """Call prepare_maxmin with the kernel replaced by a spy.

        Returns (out_max, out_min, spy).
        """
        spy = _KernelSpy()
        # Build a distinguishable input on CPU. Content is irrelevant here
        # because the numeric kernel is spied out; shape drives the dispatch.
        with paddle.base.dygraph.guard(paddle.CPUPlace()):
            inp = paddle.zeros([bsz, num_heads, seq_len], dtype=paddle.int32)
            with mock.patch.object(_iu, "scan_maxmin_chunked", spy):
                out_max, out_min = _iu.prepare_maxmin(inp, chunk_size)
        return inp, out_max, out_min, spy

    def test_num_chunks_is_ceiling_division(self):
        # seq_len=100, chunk_size=8 -> ceil(100/8)=13 (floor would give 12).
        _, out_max, out_min, spy = self._run(2, 3, 100, 8)
        self.assertEqual(spy.call_count, 1)
        # num_chunks is the 5th positional kernel arg.
        self.assertEqual(spy.args[4], 13)
        # And it drives the last output dim.
        self.assertEqual(list(out_max.shape), [2, 3, 13])
        self.assertEqual(list(out_min.shape), [2, 3, 13])

    def test_num_chunks_exact_multiple_no_extra_chunk(self):
        # seq_len=1024, chunk_size=16 -> exactly 64, no rounding-up artifact.
        _, out_max, out_min, spy = self._run(1, 2, 1024, 16)
        self.assertEqual(spy.args[4], 64)
        self.assertEqual(list(out_max.shape), [1, 2, 64])
        self.assertEqual(list(out_min.shape), [1, 2, 64])

    def test_num_chunks_partial_last_chunk(self):
        # seq_len=513, chunk_size=8 -> ceil(513/8)=65 ((513+7)//8).
        _, _, _, spy = self._run(1, 1, 513, 8)
        self.assertEqual(spy.args[4], 65)

    def test_grid_tiles_seqlen_by_512(self):
        # BN is hard-coded to 512. grid[0] = ceil(seq_len/512).
        # seq_len=100 -> 1 tile; grid[1] = bsz*num_heads = 2*3 = 6.
        _, _, _, spy = self._run(2, 3, 100, 8)
        self.assertEqual(spy.grid, (1, 6))

    def test_grid_boundary_at_512(self):
        # seq_len exactly 512 stays 1 tile; 513 crosses into 2 tiles.
        _, _, _, spy512 = self._run(1, 1, 512, 8)
        self.assertEqual(spy512.grid, (1, 1))
        _, _, _, spy513 = self._run(1, 1, 513, 8)
        self.assertEqual(spy513.grid, (2, 1))

    def test_grid_second_dim_is_bsz_times_heads(self):
        # 1024 -> ceil(1024/512)=2 tiles; second dim = 1*2 = 2.
        _, _, _, spy = self._run(1, 2, 1024, 16)
        self.assertEqual(spy.grid, (2, 2))

    def test_grid_larger_batch_head_product(self):
        # bsz=4, num_heads=5 -> second grid dim 20; seq_len=300 -> 1 tile.
        _, _, _, spy = self._run(4, 5, 300, 4)
        self.assertEqual(spy.grid, (1, 20))

    def test_outputs_are_int32(self):
        _, out_max, out_min, _ = self._run(2, 2, 64, 8)
        self.assertEqual(out_max.dtype, paddle.int32)
        self.assertEqual(out_min.dtype, paddle.int32)

    def test_returned_tensors_are_the_kernel_output_buffers(self):
        # The wrapper must hand the kernel the very tensors it returns, in
        # (max, min) order: args[1] == output_max, args[2] == output_min.
        _, out_max, out_min, spy = self._run(2, 3, 96, 8)
        self.assertIs(spy.args[1], out_max)
        self.assertIs(spy.args[2], out_min)
        # max and min buffers are distinct objects, not one aliased tensor.
        self.assertIsNot(out_max, out_min)

    def test_kernel_receives_input_and_seqlen_and_constexprs(self):
        inp, _, _, spy = self._run(2, 3, 100, 8)
        # Positional wiring: input, output_max, output_min, seqlen, num_chunks.
        self.assertIs(spy.args[0], inp)
        self.assertEqual(spy.args[3], 100)  # seq_len
        self.assertEqual(spy.args[4], 13)  # num_chunks (ceil(100/8))
        # constexpr keyword args: chunk_size forwarded verbatim, BN pinned 512.
        self.assertEqual(spy.kwargs["chunk_size"], 8)
        self.assertEqual(spy.kwargs["BN"], 512)


if __name__ == "__main__":
    unittest.main()
