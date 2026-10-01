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

"""Behavior tests for the CPU-observable orchestration of
``paddlefleet_ops._extensions.flashmask.index_utils.prepare_maxmin``.

``prepare_maxmin`` is a thin Python wrapper that (1) derives the number of
chunks from ``(seq_len, chunk_size)`` via ceiling division, (2) allocates the
two int32 output tensors with shape ``[bsz, num_heads, num_chunks]`` and
(3) launches the GPU Triton kernel ``scan_maxmin_chunked`` over a 2D grid
``(ceil(seq_len / 512), bsz * num_heads)``.

The kernel itself only runs on GPU, so here we replace it with a spy that
records exactly what the wrapper hands it. Everything asserted below is
hand-derived: chunk counts, output shapes/dtype, the launch grid and the
forwarded kernel arguments. We never call ``prepare_maxmin`` (or the kernel)
to compute its own expected values.
"""

import unittest
from unittest import mock

# paddlefleet_ops imports ``paddle`` (and Triton) at import time; the local
# CI image for this file may lack a working ``paddle`` install. Guard the
# import so the suite skips honestly instead of erroring at collection.
try:
    import paddle
    from paddlefleet_ops._extensions.flashmask import index_utils
    from paddlefleet_ops._extensions.flashmask.index_utils import (
        prepare_maxmin,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # only a genuinely missing dependency, not a bug
    paddle = None
    index_utils = None
    prepare_maxmin = None
    _IMPORT_ERROR = repr(exc)

_MODULE_AVAILABLE = _IMPORT_ERROR is None

# BN is a module-private launch constant baked into prepare_maxmin.
_EXPECTED_BN = 512


class _KernelLaunchSpy:
    """Stand-in for the GPU-only ``scan_maxmin_chunked`` Triton kernel.

    Triton kernels are launched as ``kernel[grid](*args, **kwargs)``; this
    records the grid captured by ``__getitem__`` and the positional/keyword
    arguments captured by the returned launcher, so the test can assert what
    ``prepare_maxmin`` actually forwarded. It performs no device work.
    """

    def __init__(self):
        self.grid = None
        self.args = None
        self.kwargs = None
        self.call_count = 0

    def __getitem__(self, grid):
        self.grid = grid

        def _launch(*args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.call_count += 1

        return _launch


@unittest.skipUnless(
    _MODULE_AVAILABLE,
    f"paddlefleet_ops.index_utils not importable ({_IMPORT_ERROR})",
)
class TestPrepareMaxmin(unittest.TestCase):
    """CPU-observable orchestration of ``prepare_maxmin``.

    Runs on CPU-only paddle: the GPU Triton kernel is replaced by a spy, so
    no CUDA device is required. Numeric max/min correctness is a GPU-kernel
    concern and is explicitly NOT claimed here.
    """

    def _run(self, bsz, num_heads, seq_len, chunk_size):
        """Invoke prepare_maxmin with the kernel spied, on CPU."""
        # Distinguishable, deterministic input; contents are irrelevant to
        # the wrapper's shape/grid logic but we avoid all-zero degeneracy.
        paddle.set_device("cpu")
        x = paddle.arange(bsz * num_heads * seq_len, dtype="int32").reshape(
            [bsz, num_heads, seq_len]
        )
        spy = _KernelLaunchSpy()
        with mock.patch.object(index_utils, "scan_maxmin_chunked", spy):
            out_max, out_min = prepare_maxmin(x, chunk_size=chunk_size)
        return x, out_max, out_min, spy

    def test_primary_shape_dtype_grid_and_forwarded_args(self):
        """Full contract for the core case seq_len=16, chunk_size=4.

        Hand derivation:
          num_chunks = ceil(16 / 4)      = 4
          grid       = (ceil(16 / 512), bsz*num_heads) = (1, 6)
        """
        bsz, num_heads, seq_len, chunk_size = 2, 3, 16, 4
        x, out_max, out_min, spy = self._run(
            bsz, num_heads, seq_len, chunk_size
        )

        expected_num_chunks = 4  # hand-derived ceil(16/4)
        expected_shape = [bsz, num_heads, expected_num_chunks]

        # Outputs: shape + dtype are a real allocation contract.
        self.assertEqual(out_max.shape, expected_shape)
        self.assertEqual(out_min.shape, expected_shape)
        self.assertEqual(out_max.dtype, paddle.int32)
        self.assertEqual(out_min.dtype, paddle.int32)

        # Kernel launched exactly once, over the hand-derived grid.
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(spy.grid, (1, bsz * num_heads))  # (1, 6)

        # Positional args forwarded to the kernel: the real input tensor,
        # the two freshly allocated outputs (identity), then seq_len and
        # num_chunks. Identity check ensures the wrapper hands the kernel the
        # tensors it returns, not copies.
        args = spy.args
        self.assertEqual(len(args), 5)
        self.assertIs(args[0], x)
        self.assertIs(args[1], out_max)
        self.assertIs(args[2], out_min)
        self.assertEqual(args[3], seq_len)  # 16
        self.assertEqual(args[4], expected_num_chunks)  # 4

        # Keyword args: chunk_size passthrough and the baked-in BN constant.
        self.assertEqual(spy.kwargs["chunk_size"], chunk_size)  # 4
        self.assertEqual(spy.kwargs["BN"], _EXPECTED_BN)  # 512

    def test_num_chunks_is_ceiling_division(self):
        """num_chunks must round up, not truncate.

        Each (seq_len, chunk_size) -> expected num_chunks is hand-computed;
        the uneven cases (15/8, 16/32, 4/3) distinguish ceil from floor.
        """
        # (seq_len, chunk_size, expected_num_chunks)
        cases = [
            (16, 16, 1),  # exact single chunk
            (16, 32, 1),  # chunk_size > seq_len -> still 1
            (15, 8, 2),  # ceil(15/8)=2 ; floor would give 1
            (4, 1, 4),  # chunk_size 1 -> one chunk per element
            (4, 3, 2),  # ceil(4/3)=2 ; floor would give 1
            (17, 4, 5),  # ceil(17/4)=5 ; floor would give 4
        ]
        for seq_len, chunk_size, expected_num_chunks in cases:
            with self.subTest(seq_len=seq_len, chunk_size=chunk_size):
                _, out_max, out_min, spy = self._run(1, 1, seq_len, chunk_size)
                self.assertEqual(out_max.shape, [1, 1, expected_num_chunks])
                self.assertEqual(out_min.shape, [1, 1, expected_num_chunks])
                # num_chunks is also forwarded to the kernel (positional 5th).
                self.assertEqual(spy.args[4], expected_num_chunks)

    def test_grid_tiles_along_sequence_when_longer_than_BN(self):
        """The launch grid's first axis is ceil(seq_len / 512).

        seq_len=600 > BN=512 must produce 2 tiles (floor would give 1),
        while the second grid axis stays bsz*num_heads.
        """
        bsz, num_heads, seq_len, chunk_size = 2, 5, 600, 8
        _, out_max, _out_min, spy = self._run(
            bsz, num_heads, seq_len, chunk_size
        )

        expected_num_chunks = 75  # ceil(600/8)
        self.assertEqual(out_max.shape, [bsz, num_heads, expected_num_chunks])
        # ceil(600/512) = 2 tiles ; second axis = bsz*num_heads = 10.
        self.assertEqual(spy.grid, (2, bsz * num_heads))
        self.assertEqual(spy.args[3], seq_len)  # 600
        self.assertEqual(spy.args[4], expected_num_chunks)  # 75

    def test_max_and_min_outputs_are_distinct_buffers(self):
        """The two returned tensors must be separate allocations.

        A wrapper that returned the same buffer twice (aliasing max and min)
        would corrupt results once the kernel writes both; assert identity is
        distinct and both are the tensors handed to the kernel.
        """
        x, out_max, out_min, spy = self._run(1, 1, 8, 2)
        self.assertIsNot(out_max, out_min)
        self.assertIs(spy.args[1], out_max)
        self.assertIs(spy.args[2], out_min)


if __name__ == "__main__":
    unittest.main()
