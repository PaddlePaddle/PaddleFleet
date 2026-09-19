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

"""Behavior tests for the flashmask block_mask_utils production module.

Primary/core behavior under test: ``find_blocks_topp`` implements a top-p
(nucleus) block selection. For every row of the probability tensor it keeps
the smallest set of highest-probability entries whose *preceding* cumulative
sum is still below ``row_sum * p``; every remaining entry is dropped. The
expected boolean masks below are derived independently by hand from that
definition (never by calling ``find_blocks_topp`` to produce its own oracle).

The computation is a Triton kernel that only executes on a real GPU, and
paddlefleet imports paddle at module import time. When paddle / triton / a CUDA
device are unavailable these cases skip with an honest reason instead of
reporting a fake pass.
"""

import os
import sys
import unittest

# Make the paddlefleet_ops package importable when running from the repo tree.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_OPS_SRC = os.path.join(_REPO_ROOT, "packages", "paddlefleet_ops", "src")
if os.path.isdir(_OPS_SRC) and _OPS_SRC not in sys.path:
    sys.path.insert(0, _OPS_SRC)


# Only catch ImportError/ModuleNotFoundError here: a genuine import/compile/API
# failure must propagate rather than be silently downgraded to "missing dep".
_IMPORT_SKIP_REASON = None
find_blocks_topp = None
paddle = None
try:
    import paddle
    import triton  # noqa: F401
    from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
        find_blocks_topp,
    )
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    _IMPORT_SKIP_REASON = (
        "paddle / triton / paddlefleet_ops not importable in this "
        f"environment: {exc!r}"
    )


def _cuda_device_available():
    """Honest probe for a usable CUDA device (Triton kernels need a GPU)."""
    if paddle is None:
        return False
    if not paddle.is_compiled_with_cuda():
        return False
    return paddle.device.cuda.device_count() > 0


_RUNNABLE = _IMPORT_SKIP_REASON is None and _cuda_device_available()
_SKIP_REASON = _IMPORT_SKIP_REASON or (
    "no CUDA device available; find_blocks_topp is a Triton GPU kernel and "
    "its numeric behavior cannot be exercised on CPU"
)


@unittest.skipUnless(_RUNNABLE, _SKIP_REASON)
class TestFindBlocksTopp(unittest.TestCase):
    """Numeric behavior of the top-p block selection entry point."""

    def _run(self, values, p):
        """Call the production entry point on a GPU tensor, return bool mask."""
        x = paddle.to_tensor(values, dtype="float32").cuda()
        mask = find_blocks_topp(x, p=p)
        return mask.astype("bool").numpy().tolist()

    def test_topp_keeps_nucleus_and_scatters_to_original_positions(self):
        # Row [0.1, 0.4, 0.2, 0.3], sum = 1.0, p = 0.9 -> cutoff = 0.9.
        # Descending order: 0.4(i1) 0.3(i3) 0.2(i2) 0.1(i0)
        # preceding cumsum : 0.0    0.4    0.7    0.9
        # keep (prec < 0.9):  T      T      T      F
        # -> original positions [i0, i1, i2, i3] = [F, T, T, T]
        self.assertEqual(
            self._run([[0.1, 0.4, 0.2, 0.3]], p=0.9),
            [[False, True, True, True]],
        )

    def test_topp_threshold_is_consumed(self):
        # Same row, p = 0.3 -> cutoff = 0.3. Only the single largest entry has
        # a preceding cumsum (0.0) below 0.3; the next preceding sum is 0.4.
        # -> keep only 0.4 at index 1 -> [F, T, F, F]. A smaller p must select
        # strictly fewer tokens than p=0.9, proving p actually drives output.
        self.assertEqual(
            self._run([[0.1, 0.4, 0.2, 0.3]], p=0.3),
            [[False, True, False, False]],
        )

    def test_topp_rows_are_independent(self):
        # 4-D [b, h, m, n] input exercises the reshape([-1, n]) round trip and
        # per-row independence. Row-0 as above -> [F, T, T, T].
        # Row-1 [0.5, 0.05, 0.3, 0.15], sum = 1.0, p = 0.9, cutoff = 0.9.
        # Descending: 0.5(i0) 0.3(i2) 0.15(i3) 0.05(i1)
        # preceding : 0.0     0.5     0.8      0.95
        # keep      : T       T       T        F   -> [T, F, T, T]
        result = self._run(
            [[[[0.1, 0.4, 0.2, 0.3], [0.5, 0.05, 0.3, 0.15]]]],
            p=0.9,
        )
        self.assertEqual(
            result,
            [[[[False, True, True, True], [True, False, True, True]]]],
        )

    def test_zero_row_selects_nothing(self):
        # row_sum == 0.0 takes the early-return branch that writes an all-zero
        # (all-False) mask regardless of p.
        self.assertEqual(
            self._run([[0.0, 0.0, 0.0, 0.0]], p=0.9),
            [[False, False, False, False]],
        )

    def test_output_shape_matches_input(self):
        # Shape is a secondary contract; the content assertions above are the
        # primary check. Here we only confirm the [b, h, m, n] layout survives.
        x = paddle.to_tensor(
            [[[[0.2, 0.3, 0.5], [0.5, 0.4, 0.1]]]], dtype="float32"
        ).cuda()
        mask = find_blocks_topp(x, p=0.9)
        self.assertEqual(list(mask.shape), [1, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
