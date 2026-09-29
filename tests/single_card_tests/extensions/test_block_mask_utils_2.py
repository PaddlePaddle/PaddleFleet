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

"""CPU-observable orchestration tests for
``paddlefleet_ops._extensions.flashmask.block_mask_utils.find_blocks_topp``.

``find_blocks_topp`` is the only host-side (non ``@triton.jit``) entry point in
the module. Its job, before launching the GPU Triton kernel ``top_p_kernel``,
is pure Python bookkeeping that IS observable on CPU:

  * flatten ``[b, h, m, n]`` (or ``[b, n]``) to ``[-1, n]`` rows,
  * ``BLOCK_SIZE = next_power_of_2(n)`` and ``NUM_DIMS = log2(BLOCK_SIZE)``,
  * launch grid ``(total_rows,)``,
  * forward the row stride, ``n``, and the two constexprs to the kernel,
  * reshape the kernel's output back to the caller's original shape.

The kernel itself only runs on a real GPU, so it is replaced here with a spy
that records the launch grid and the forwarded arguments. Everything asserted
is hand-derived; ``find_blocks_topp`` is never used to compute its own expected
values. Nucleus-selection numeric correctness lives in the GPU kernel and is
explicitly NOT claimed by these tests.

On this Paddle build the host-side orchestration completes successfully (both
``x.reshape(-1, n)`` and ``paddle.empty(..., device=x.device)`` are accepted),
so the tests below are plain positive assertions of the launch contract.

This facet (host launch-config derivation + output shape round-trip) is
distinct from the geometric block-mask kernels (``_is_block_fully_masked`` /
``check_partially_masked_state`` ...) exercised by the sibling files.
"""

import unittest
from unittest import mock

# paddlefleet_ops imports ``paddle`` (and Triton) at import time; the local CI
# image for this file may lack a working ``paddle`` install. Guard the import
# so the suite skips honestly instead of erroring at collection. Only a
# genuinely missing dependency is caught -- an API/compile error must surface.
try:
    import paddle
    from paddlefleet_ops._extensions.flashmask import block_mask_utils
    from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
        find_blocks_topp,
    )

    _IMPORT_ERROR = None
except ImportError as exc:
    paddle = None
    block_mask_utils = None
    find_blocks_topp = None
    _IMPORT_ERROR = repr(exc)

_MODULE_AVAILABLE = _IMPORT_ERROR is None


class _KernelLaunchSpy:
    """Stand-in for the GPU-only ``top_p_kernel`` Triton kernel.

    Triton kernels are launched as ``kernel[grid](*args, **kwargs)``. This
    records the grid captured by ``__getitem__`` and the positional/keyword
    arguments captured by the returned launcher so the test can assert exactly
    what ``find_blocks_topp`` forwarded. It performs no device work and leaves
    the (empty) output tensor untouched.
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
    f"paddlefleet_ops.block_mask_utils not importable ({_IMPORT_ERROR})",
)
class TestFindBlocksToppLaunchConfig(unittest.TestCase):
    """Host-side launch-config derivation and shape round-trip.

    Runs on CPU-only paddle: the GPU Triton kernel is replaced by a spy, so no
    CUDA device is required.

    On this Paddle build ``find_blocks_topp`` completes the host-side
    orchestration successfully: ``x.reshape(-1, n)`` accepts the varargs form
    and ``paddle.empty(x_reshaped.shape, dtype=paddle.bool, device=x.device)``
    is honored (the ``device`` keyword is accepted and a Tensor exposes
    ``.device``). The tests therefore assert the real orchestration contract
    directly (positive assertions), with the GPU kernel replaced by a spy so
    only the CPU-observable launch-config derivation and shape round-trip are
    claimed.
    """

    def setUp(self):
        # Keep everything on CPU; the spy removes the only GPU dependency in
        # the launch itself. Contents are irrelevant to the launch-config /
        # shape logic, so we use a distinguishable deterministic filling and
        # avoid all-zero degeneracy.
        paddle.set_device("cpu")

    def test_4d_input_launch_config_and_shape_roundtrip(self):
        """4D input [2, 3, 5, 8] -> flattened to 30 rows of width 8.

        Hand derivation (n = 8):
          total_rows = 2*3*5           = 30   -> grid == (30,)
          BLOCK_SIZE = next_pow2(8)    = 8
          NUM_DIMS   = log2(8)         = 3
          output shape restored to the caller's [2, 3, 5, 8], dtype bool.
        """
        b, h, m, n = 2, 3, 5, 8
        x = paddle.arange(b * h * m * n, dtype="float32").reshape([b, h, m, n])

        spy = _KernelLaunchSpy()
        with mock.patch.object(block_mask_utils, "top_p_kernel", spy):
            out = find_blocks_topp(x, p=0.7)

        # --- output contract: same shape as input, boolean dtype ---
        self.assertEqual(out.shape, [b, h, m, n])
        self.assertEqual(out.dtype, paddle.bool)

        # --- launch grid is one program per flattened row ---
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(spy.grid, (b * h * m,))  # (30,)

        # --- forwarded positional args: X, Out, row-stride, p, n ---
        args = spy.args
        self.assertEqual(len(args), 5)
        # arg[0] is the flattened [30, 8] view handed to the kernel.
        self.assertEqual(list(args[0].shape), [b * h * m, n])
        self.assertEqual(args[3], 0.7)  # threshold p passthrough
        self.assertEqual(args[4], n)  # N_COLS == 8

        # --- forwarded constexprs: hand-derived power-of-two block sizing ---
        self.assertEqual(spy.kwargs["BLOCK_SIZE"], 8)
        self.assertEqual(spy.kwargs["NUM_DIMS"], 3)

    def test_2d_input_rounds_block_size_up_to_power_of_two(self):
        """2D input [4, 10] with a non power-of-two width.

        Hand derivation (n = 10):
          total_rows = 4               -> grid == (4,)
          BLOCK_SIZE = next_pow2(10)   = 16   (floor/identity would be wrong)
          NUM_DIMS   = log2(16)        = 4
          output shape restored to [4, 10], dtype bool.
        """
        rows, n = 4, 10
        x = paddle.arange(rows * n, dtype="float32").reshape([rows, n])

        spy = _KernelLaunchSpy()
        with mock.patch.object(block_mask_utils, "top_p_kernel", spy):
            out = find_blocks_topp(x, p=0.5)

        self.assertEqual(out.shape, [rows, n])
        self.assertEqual(out.dtype, paddle.bool)

        self.assertEqual(spy.call_count, 1)
        self.assertEqual(spy.grid, (rows,))  # (4,)
        self.assertEqual(spy.args[4], n)  # N_COLS == 10
        self.assertEqual(spy.kwargs["BLOCK_SIZE"], 16)  # ceil to power of two
        self.assertEqual(spy.kwargs["NUM_DIMS"], 4)


if __name__ == "__main__":
    unittest.main()
