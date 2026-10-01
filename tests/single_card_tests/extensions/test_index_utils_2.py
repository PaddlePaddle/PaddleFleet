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

# Behavior tests for paddlefleet_ops._extensions.flashmask.index_utils
#
# Facet under test: the CPU-observable *kernel-launch dispatch contract* of
# ``prepare_maxmin``. ``prepare_maxmin`` unpacks the input shape, derives the
# per-head chunk count by ceiling division, allocates the two int32 output
# buffers, computes the triton launch grid, and forwards everything into the
# ``scan_maxmin_chunked[grid](...)`` kernel launch, returning the very buffers
# it handed to the kernel.
#
# The triton kernel itself needs a real GPU, so it is the one *not-under-test*
# collaborator we replace with a distinguishable stub (see unit-test rules,
# 计算优化 / 无卡: wrapper-layer dispatch may be validated with a kernel stub).
# We keep the real ``prepare_maxmin`` orchestration and assert, against
# independently hand-derived values, the launch grid, the positional/keyword
# arguments, and that the returned tensors ARE the buffers passed to the
# kernel (identity). Real kernel numerics are NOT verified here.

import unittest
from unittest import mock

# paddlefleet_ops.index_utils imports paddle (and triton) at import time. The
# local environment has no paddle, so guard the import and skip honestly rather
# than faking a pass. Only ImportError-family errors count as "missing dep".
try:
    import paddle
    from paddlefleet_ops._extensions.flashmask.index_utils import (
        prepare_maxmin,
        scan_maxmin_chunked,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # missing paddle / triton / the ops package
    paddle = None
    prepare_maxmin = None
    scan_maxmin_chunked = None
    _IMPORT_ERROR = exc

_DEPS_AVAILABLE = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"prepare_maxmin dependencies unavailable: {_IMPORT_ERROR!r}"
    if _IMPORT_ERROR is not None
    else ""
)

_INDEX_UTILS_MODULE = "paddlefleet_ops._extensions.flashmask.index_utils"
_BN = 512  # constant hardcoded inside prepare_maxmin


def _launch_capture(mock_kernel):
    """Extract the grid and the (args, kwargs) of a ``kernel[grid](...)`` launch.

    ``prepare_maxmin`` does ``scan_maxmin_chunked[grid](...)``. Against a
    MagicMock that is ``mock_kernel.__getitem__(grid)`` followed by calling the
    returned object. This DOES work with a plain MagicMock, which is why we can
    observe the launch without a GPU.
    """
    getitem = mock_kernel.__getitem__
    grid = getitem.call_args.args[0]
    launch = getitem.return_value.call_args
    return grid, launch.args, launch.kwargs


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class TestPrepareMaxminDispatch(unittest.TestCase):
    """prepare_maxmin wires shapes, grid and buffers into the kernel launch."""

    def test_dispatch_contract_forwards_buffers_and_args(self):
        # Hand-derived expectations (no call to the function under test):
        #   input shape [3, 2, 100], chunk_size 8
        #   num_chunks = ceil(100 / 8) = 13
        #   grid       = ((100 + 511) // 512, 3 * 2) = (1, 6)
        bsz, num_heads, seq_len, chunk_size = 3, 2, 100, 8
        expected_num_chunks = 13
        expected_grid = (1, 6)

        x = paddle.zeros([bsz, num_heads, seq_len], dtype="int32")

        with mock.patch(
            _INDEX_UTILS_MODULE + ".scan_maxmin_chunked"
        ) as mock_kernel:
            out_max, out_min = prepare_maxmin(x, chunk_size=chunk_size)

        # The kernel launch happened exactly once.
        self.assertEqual(mock_kernel.__getitem__.call_count, 1)
        self.assertEqual(mock_kernel.__getitem__.return_value.call_count, 1)

        grid, pos, kw = _launch_capture(mock_kernel)

        # Grid is the independently derived tuple, not merely a 2-tuple.
        self.assertEqual(grid, expected_grid)

        # Positional argument forwarding: input by identity, then the two
        # allocated buffers, then seq_len and the derived chunk count.
        self.assertIs(pos[0], x)
        self.assertEqual(pos[3], seq_len)
        self.assertEqual(pos[4], expected_num_chunks)

        # Keyword arguments carry the exact chunk_size and the hardcoded BN.
        self.assertEqual(kw["chunk_size"], chunk_size)
        self.assertEqual(kw["BN"], _BN)

        # Identity contract: the tensors returned to the caller are precisely
        # the buffers handed to the kernel (arg1 = max, arg2 = min), in order.
        # A swap of the two buffers, or returning freshly-made tensors, breaks
        # this even though shapes/dtypes would still match.
        self.assertIs(out_max, pos[1])
        self.assertIs(out_min, pos[2])
        self.assertIsNot(out_max, out_min)

        # Returned buffers have the hand-derived shape and int32 dtype.
        self.assertEqual(
            list(out_max.shape), [bsz, num_heads, expected_num_chunks]
        )
        self.assertEqual(
            list(out_min.shape), [bsz, num_heads, expected_num_chunks]
        )
        self.assertEqual(out_max.dtype, paddle.int32)
        self.assertEqual(out_min.dtype, paddle.int32)


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class TestPrepareMaxminChunkAndGridMath(unittest.TestCase):
    """num_chunks (ceil division) and launch grid across distinct shapes."""

    # (bsz, num_heads, seq_len, chunk_size,
    #  hand_num_chunks, hand_grid)
    # Every expected value is derived by hand, independent of production:
    #   num_chunks = ceil(seq_len / chunk_size)
    #   grid       = (ceil(seq_len / 512), bsz * num_heads)
    CASES = [
        # exact multiple, single grid tile
        (1, 5, 48, 12, 4, (1, 5)),
        # not divisible -> ceiling rounds up (100/8 -> 13)
        (3, 2, 100, 8, 13, (1, 6)),
        # seq_len spans two BN tiles (1024 > 512): grid[0] == 2
        (2, 3, 1024, 16, 64, (2, 6)),
        # degenerate seq_len == 1, chunk_size larger than seq_len -> 1 chunk
        (1, 5, 1, 4, 1, (1, 5)),
    ]

    def test_num_chunks_and_grid_match_hand_derivation(self):
        for (
            bsz,
            num_heads,
            seq_len,
            chunk_size,
            exp_num_chunks,
            exp_grid,
        ) in self.CASES:
            with self.subTest(seq_len=seq_len, chunk_size=chunk_size):
                x = paddle.zeros([bsz, num_heads, seq_len], dtype="int32")
                with mock.patch(
                    _INDEX_UTILS_MODULE + ".scan_maxmin_chunked"
                ) as mock_kernel:
                    out_max, out_min = prepare_maxmin(x, chunk_size=chunk_size)

                grid, pos, kw = _launch_capture(mock_kernel)

                # Grid tuple matches the independent ceil-division derivation.
                self.assertEqual(grid, exp_grid)
                # num_chunks passed to the kernel (positional slot 4).
                self.assertEqual(pos[4], exp_num_chunks)
                # seq_len forwarded verbatim (positional slot 3).
                self.assertEqual(pos[3], seq_len)
                # BN kwarg is constant 512 regardless of the shape.
                self.assertEqual(kw["BN"], _BN)

                # Output buffers carry the derived chunk count on the last axis.
                self.assertEqual(
                    list(out_max.shape),
                    [bsz, num_heads, exp_num_chunks],
                )
                self.assertEqual(
                    list(out_min.shape),
                    [bsz, num_heads, exp_num_chunks],
                )


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class TestScanMaxminChunkedKernel(unittest.TestCase):
    """The exported kernel is the triton entry prepare_maxmin dispatches to."""

    def test_kernel_wraps_expected_function_and_signature(self):
        # A triton JITFunction keeps the original python function on ``.fn``.
        # Assert identity/signature content, not mere existence: the kernel
        # prepare_maxmin launches must be the scan_maxmin_chunked scanner with
        # the parameter order prepare_maxmin relies on when forwarding args.
        fn = getattr(scan_maxmin_chunked, "fn", None)
        if fn is None:
            self.skipTest(
                "triton JITFunction does not expose .fn on this version"
            )

        self.assertEqual(fn.__name__, "scan_maxmin_chunked")

        import inspect

        params = list(inspect.signature(fn).parameters)
        # prepare_maxmin passes (input, output_max, output_min, seqlen,
        # num_chunks) positionally then chunk_size=, BN= by keyword; the
        # kernel signature order must line up with that call site.
        self.assertEqual(
            params,
            [
                "input_ptr",
                "output_max_ptr",
                "output_min_ptr",
                "seqlen",
                "num_chunks",
                "chunk_size",
                "BN",
            ],
        )


if __name__ == "__main__":
    unittest.main()
