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

"""Extra single-card coverage for roll_tensor / _roll_tensor_packed_seq
guard clauses and the non-zero pad_value branch.

Complements test_roll_tensor_packed_seq.py by hitting the specific error
paths and the ``pad_value != 0`` fill that were not exercised there:

- ``_roll_tensor_packed_seq`` out-of-range ``dims`` guard
  (multi_token_prediction.py:124).
- ``_roll_tensor_packed_seq`` ``cu_seqlens_q is None`` guard (line 126).
- ``_roll_tensor_packed_seq`` accepting a plain Python list for
  ``cu_seqlens_q`` (the non-Tensor branch, line 131).
- ``roll_tensor`` non-packed path with ``pad_value != 0`` writing the fill
  value into the new-in tail position (line 257).
"""

from __future__ import annotations

import unittest

import numpy as np
import paddle

from paddlefleet.transformer.multi_token_prediction import (
    _roll_tensor_packed_seq,
    extract_local_zigzag_chunks,
    roll_tensor,
)


class TestRollTensorPackedSeqGuards(unittest.TestCase):
    def test_dims_out_of_range_raises(self) -> None:
        # dim=5 is out of range for a 2-D tensor -> line 124 raise.
        t = paddle.to_tensor(np.arange(8, dtype=np.int64).reshape(1, 8))
        cu = paddle.to_tensor([0, 4, 8], dtype="int32")
        with self.assertRaises(ValueError):
            _roll_tensor_packed_seq(t, shifts=-1, dims=5, cu_seqlens_q=cu)

    def test_cu_seqlens_none_raises(self) -> None:
        # cu_seqlens_q=None -> line 126 raise.
        t = paddle.to_tensor(np.arange(8, dtype=np.int64).reshape(1, 8))
        with self.assertRaises(ValueError):
            _roll_tensor_packed_seq(t, shifts=-1, dims=-1, cu_seqlens_q=None)

    def test_cu_seqlens_accepts_python_list(self) -> None:
        # A plain python list (not a paddle.Tensor) exercises the
        # ``list(cu_seqlens_q)`` branch on line 131 and must produce the
        # same result as the tensor form.
        arr = np.arange(1, 9, dtype=np.int64).reshape(1, 8)
        cu_list = [0, 4, 8]
        rolled_list, sum_list = _roll_tensor_packed_seq(
            paddle.to_tensor(arr), shifts=-1, dims=-1, cu_seqlens_q=cu_list
        )
        rolled_t, sum_t = _roll_tensor_packed_seq(
            paddle.to_tensor(arr),
            shifts=-1,
            dims=-1,
            cu_seqlens_q=paddle.to_tensor(cu_list, dtype="int32"),
        )
        np.testing.assert_array_equal(rolled_list.numpy(), rolled_t.numpy())
        self.assertEqual(int(sum_list.numpy()), int(sum_t.numpy()))


class TestRollTensorNonPackedPadValue(unittest.TestCase):
    def test_nonzero_pad_value_fills_tail(self) -> None:
        # Non-packed roll with pad_value != 0 -> line 257 (keep*mask + pad*(1-mask)).
        arr = np.arange(1, 9, dtype=np.int64).reshape(1, 8)
        rolled, _ = roll_tensor(
            paddle.to_tensor(arr), shifts=-1, dims=-1, pad_value=-100
        )
        out = rolled.numpy()
        # Standard left shift: [2,3,...,8, <fill>]; fill must be pad_value.
        self.assertEqual(int(out[0, -1]), -100)
        np.testing.assert_array_equal(out[0, :-1], arr[0, 1:])


class TestExtractLocalZigzagChunks(unittest.TestCase):
    """extract_local_zigzag_chunks is pure tensor slicing (no CP comm), so
    all of its branches are reachable single-card by passing cp_size
    directly.
    """

    def test_cp_size_one_is_identity(self) -> None:
        # cp_size == 1 -> early return of the full tensor (lines 292-293).
        t = paddle.to_tensor(np.arange(16, dtype=np.float32).reshape(1, 8, 2))
        out = extract_local_zigzag_chunks(t, cp_rank=0, cp_size=1, axis=1)
        np.testing.assert_array_equal(out.numpy(), t.numpy())

    def test_cp_size_two_zigzag_partition(self) -> None:
        # cp_size == 2 exercises the slice/concat body (lines 294-315).
        L = 8
        t = paddle.to_tensor(np.arange(L, dtype=np.float32).reshape(1, L, 1))
        interval = L // 2 // 2  # = 2
        for rank in range(2):
            out = extract_local_zigzag_chunks(
                t, cp_rank=rank, cp_size=2, axis=1
            )
            # start chunk: [interval*rank : interval*(rank+1)]
            # end chunk:   [L-interval*(rank+1) : L-interval*rank]
            start = list(range(interval * rank, interval * (rank + 1)))
            end = list(range(L - interval * (rank + 1), L - interval * rank))
            expected = start + end
            self.assertEqual(
                out.numpy().reshape(-1).astype(int).tolist(), expected
            )

    def test_indivisible_seq_len_raises(self) -> None:
        # seq_len not divisible by 2*cp_size -> ValueError (lines 297-298).
        t = paddle.to_tensor(np.arange(6, dtype=np.float32).reshape(1, 6, 1))
        with self.assertRaises(ValueError):
            extract_local_zigzag_chunks(t, cp_rank=0, cp_size=4, axis=1)


if __name__ == "__main__":
    unittest.main()
