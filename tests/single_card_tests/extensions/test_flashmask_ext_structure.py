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

"""Behavior test for the CPU-observable orchestration of
``paddlefleet_ops._extensions.flashmask.block_mask_utils.find_blocks_topp``.

``find_blocks_topp`` is the only plain-Python entry point in the flashmask
extension modules; everything else (``bitonic_argsort_device``,
``_bitonic_merge``, ``_compare_and_swap``, ``top_p_kernel`` and the
``scan_maxmin_chunked``/``prepare_maxmin`` pair in ``index_utils``) is a
``@triton.jit`` GPU kernel that cannot execute on CPU. ``find_blocks_topp``
itself does real, host-side work before the launch: it flattens the input to
``[B, n]``, derives ``BLOCK_SIZE = next_power_of_2(n)`` and
``NUM_DIMS = log2(BLOCK_SIZE)``, allocates the bool output, and launches
``top_p_kernel`` over the 1-D grid ``(B,)``.

We replace the GPU kernel with a spy so the host orchestration runs on
CPU-only paddle, and assert hand-derived values (flattened shape, launch grid,
row stride, and the forwarded ``p``/``n``/``BLOCK_SIZE``/``NUM_DIMS``). We
never call ``find_blocks_topp`` to produce its own expected values, and we do
NOT claim the GPU top-p / bitonic-argsort numerics are verified here.

KNOWN PRODUCTION BUG (do not fix here). ``find_blocks_topp`` cannot complete on
any standard paddle build because it uses two non-existent paddle APIs before
the kernel launch:
  * block_mask_utils.py:340  ``x.reshape(-1, n)`` -- paddle ``Tensor.reshape``
    takes a shape *sequence*; the positional form binds ``shape=-1`` and
    ``name=n`` and raises. Every sibling call in the package uses the list form
    (e.g. index_utils ``prepare_maxmin`` uses ``paddle.empty([...])``).
  * block_mask_utils.py:349  ``paddle.empty(..., device=x.device)`` -- paddle
    ``empty`` has signature ``(shape, dtype=None, name=None)``; there is no
    ``device`` keyword, and paddle tensors expose ``.place`` rather than
    ``.device``.
The orchestration contract below is the CORRECT behavior once those two lines
are fixed; the test is marked ``expectedFailure`` so the suite stays honest
until the production code is corrected.
"""

import math
import unittest
from unittest import mock

# paddlefleet_ops imports ``paddle`` (and Triton) at import time; the local
# environment for this file may lack a working ``paddle`` install. Guard the
# import so the suite skips honestly instead of erroring at collection. Only a
# genuinely missing dependency is swallowed here -- not a bug in the code.
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


def _next_power_of_2(value):
    """Independent hand reference for triton.next_power_of_2 (no prod call)."""
    result = 1
    while result < value:
        result *= 2
    return result


class _KernelLaunchSpy:
    """Stand-in for the GPU-only ``top_p_kernel`` Triton kernel.

    Triton kernels launch as ``kernel[grid](*args, **kwargs)``; this records
    the grid captured by ``__getitem__`` and the args captured by the returned
    launcher, so the test can assert what ``find_blocks_topp`` forwards. It
    performs no device work and writes nothing back.
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
class TestFindBlocksToppOrchestration(unittest.TestCase):
    """CPU-observable orchestration of ``find_blocks_topp``.

    Runs on CPU-only paddle with the GPU kernel replaced by a spy, so no CUDA
    device is required. GPU top-p / bitonic-argsort numerics are NOT claimed.
    """

    @unittest.expectedFailure  # blocked by block_mask_utils.py:340 and :349
    def test_find_blocks_topp_orchestration_contract(self):
        """Full host-side launch contract for a [1, 1, 2, 4] input.

        Hand derivation (independent of the function under test):
          n          = 4  (last dim)
          B          = 1 * 1 * 2               = 2   (flattened rows)
          BLOCK_SIZE = next_power_of_2(4)      = 4
          NUM_DIMS   = log2(4)                 = 2
          grid       = (B,)                    = (2,)
          row stride = n                       = 4   (contiguous [B, n])
        """
        paddle.set_device("cpu")

        p = 0.9
        original_shape = [1, 1, 2, 4]
        n = original_shape[-1]
        # Distinguishable, non-degenerate probabilities; contents do not drive
        # the host orchestration but we avoid an all-zero input.
        x = paddle.to_tensor(
            [[[[0.1, 0.4, 0.2, 0.3], [0.5, 0.1, 0.3, 0.1]]]],
            dtype="float32",
        )
        self.assertEqual(x.shape, original_shape)

        expected_rows = 2  # hand-derived B = 1*1*2
        expected_block_size = _next_power_of_2(n)  # 4
        expected_num_dims = int(math.log2(expected_block_size))  # 2

        spy = _KernelLaunchSpy()
        with mock.patch.object(block_mask_utils, "top_p_kernel", spy):
            out = find_blocks_topp(x, p)

        # Kernel launched exactly once over the hand-derived 1-D grid.
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(spy.grid, (expected_rows,))  # (2,)

        # Positional args: (flattened_input, output_buffer, row_stride, p, n).
        args = spy.args
        self.assertEqual(len(args), 5)
        self.assertEqual(args[0].shape, [expected_rows, n])  # [2, 4]
        self.assertEqual(args[1].shape, [expected_rows, n])  # output buffer
        self.assertEqual(args[1].dtype, paddle.bool)
        self.assertEqual(args[2], n)  # contiguous row stride == n == 4
        self.assertEqual(args[3], p)  # threshold forwarded verbatim
        self.assertEqual(args[4], n)  # 4

        # Keyword args: the power-of-two block size and its log2.
        self.assertEqual(spy.kwargs["BLOCK_SIZE"], expected_block_size)  # 4
        self.assertEqual(spy.kwargs["NUM_DIMS"], expected_num_dims)  # 2

        # Result is restored to the original [b, h, m, n] shape as a bool mask.
        self.assertEqual(out.shape, original_shape)
        self.assertEqual(out.dtype, paddle.bool)


if __name__ == "__main__":
    unittest.main()
