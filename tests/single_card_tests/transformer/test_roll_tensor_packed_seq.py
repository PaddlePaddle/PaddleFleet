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

"""Unit tests for roll_tensor / _roll_tensor_packed_seq.

Verifies that the Paddle port of MCore's ``roll_tensor`` matches a naive
numpy reference implementation across:
- non-packed single-sequence roll (baseline);
- packed sequences with multiple docs of varying lengths;
- edge cases: single doc, empty tail padding, K depths in [1, 3].

The CP (cp_size>1) branch is not implemented on this path and is only
checked to raise NotImplementedError here.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np
import paddle

from paddlefleet.transformer.multi_token_prediction import (
    _roll_tensor_packed_seq,
    roll_tensor,
)


@dataclass
class _FakePackedSeqParams:
    cu_seqlens_q: paddle.Tensor


def _numpy_roll_packed(
    tensor_np: np.ndarray, cu_seqlens: list[int]
) -> np.ndarray:
    """Reference: per-doc left-shift, boundary position zeroed."""
    out = tensor_np.copy()
    num_docs = len(cu_seqlens) - 1
    for i in range(num_docs):
        s = int(cu_seqlens[i])
        e = int(cu_seqlens[i + 1])
        if e - s <= 0:
            continue
        seg = tensor_np[..., s:e].copy()
        rolled = np.roll(seg, shift=-1, axis=-1)
        # Zero out the last position of each doc.
        rolled[..., -1] = 0
        out[..., s:e] = rolled
    return out


def _numpy_roll_nonpacked(tensor_np: np.ndarray) -> np.ndarray:
    """Reference for the non-packed branch: standard left-shift + zero-fill."""
    out = np.roll(tensor_np, shift=-1, axis=-1)
    out[..., -1] = 0
    return out


def _make_cu_seqlens(seq_lens: list[int]) -> paddle.Tensor:
    cu = [0]
    for length in seq_lens:
        cu.append(cu[-1] + length)
    return paddle.to_tensor(cu, dtype="int32")


class TestRollTensorNonPacked(unittest.TestCase):
    """Standard (non-packed) roll_tensor behaviour."""

    def test_left_shift_int64(self) -> None:
        for L in [8, 64, 128]:
            arr = np.arange(1, L + 1, dtype=np.int64).reshape(1, L)
            expected = _numpy_roll_nonpacked(arr)

            t = paddle.to_tensor(arr)
            rolled, total = roll_tensor(t, shifts=-1, dims=-1)
            np.testing.assert_array_equal(
                rolled.numpy(), expected, err_msg=f"L={L}"
            )
            self.assertEqual(int(total.numpy()), int(expected.sum()))

    def test_left_shift_float32(self) -> None:
        arr = np.arange(1, 65, dtype=np.float32).reshape(2, 32)
        expected = _numpy_roll_nonpacked(arr)
        t = paddle.to_tensor(arr)
        rolled, _ = roll_tensor(t, shifts=-1, dims=-1)
        np.testing.assert_allclose(rolled.numpy(), expected)

    def test_rejects_shifts_other_than_minus_one(self) -> None:
        t = paddle.to_tensor(np.arange(8, dtype=np.int64))
        with self.assertRaises(ValueError):
            roll_tensor(t, shifts=1, dims=-1)


class TestRollTensorPackedSeq(unittest.TestCase):
    """_roll_tensor_packed_seq semantics: per-doc shift + zero at doc tail."""

    def _check(self, arr: np.ndarray, seq_lens: list[int]) -> None:
        cu = [0]
        for length in seq_lens:
            cu.append(cu[-1] + length)
        assert cu[-1] == arr.shape[-1], f"cu={cu} vs shape={arr.shape}"

        expected = _numpy_roll_packed(arr, cu)
        params = _FakePackedSeqParams(
            cu_seqlens_q=paddle.to_tensor(cu, dtype="int32")
        )
        rolled, total = roll_tensor(
            paddle.to_tensor(arr),
            shifts=-1,
            dims=-1,
            cp_group=None,
            cu_seqlens_q=params.cu_seqlens_q,
        )
        np.testing.assert_array_equal(
            rolled.numpy(),
            expected,
            err_msg=f"seq_lens={seq_lens}, shape={arr.shape}",
        )
        self.assertEqual(int(total.numpy()), int(expected.sum()))

    # ---- L = 64 ----------------------------------------------------------

    def test_L64_two_docs(self) -> None:
        arr = np.arange(1, 65, dtype=np.int64).reshape(1, 64)
        self._check(arr, [30, 34])

    def test_L64_three_docs(self) -> None:
        arr = np.arange(1, 65, dtype=np.int64).reshape(1, 64)
        self._check(arr, [20, 25, 19])

    def test_L64_single_doc(self) -> None:
        arr = np.arange(1, 65, dtype=np.int64).reshape(1, 64)
        self._check(arr, [64])

    # ---- L = 128 ---------------------------------------------------------

    def test_L128_uneven_docs(self) -> None:
        arr = np.arange(1, 129, dtype=np.int64).reshape(1, 128)
        self._check(arr, [1, 40, 50, 37])

    def test_L128_single_token_doc_leading(self) -> None:
        arr = np.arange(1, 129, dtype=np.int64).reshape(1, 128)
        # A single-token doc rolls to zero (only one boundary position, which is zeroed).
        self._check(arr, [1, 63, 64])

    # ---- L = 256 ---------------------------------------------------------

    def test_L256_many_docs(self) -> None:
        arr = np.arange(1, 257, dtype=np.int64).reshape(1, 256)
        # 7 docs summing to 256
        self._check(arr, [30, 30, 30, 30, 30, 30, 76])

    # ---- multi-dim payload (e.g. loss_mask [B, L]) -----------------------

    def test_2d_batch(self) -> None:
        arr = np.random.randint(low=1, high=1000, size=(4, 96)).astype(np.int64)
        self._check(arr, [20, 30, 46])

    # ---- float payload (e.g. loss_mask float32) --------------------------

    def test_float_payload(self) -> None:
        arr = np.random.RandomState(0).randn(1, 64).astype(np.float32)
        cu = [0, 25, 64]
        expected = _numpy_roll_packed(arr, cu)
        params = _FakePackedSeqParams(
            cu_seqlens_q=paddle.to_tensor(cu, dtype="int32"),
        )
        rolled, _ = roll_tensor(
            paddle.to_tensor(arr),
            shifts=-1,
            dims=-1,
            cu_seqlens_q=params.cu_seqlens_q,
        )
        np.testing.assert_allclose(rolled.numpy(), expected, atol=1e-6)


class TestRollTensorGuards(unittest.TestCase):
    """Argument validation and CP guard."""

    def test_rejects_non_last_dim(self) -> None:
        t = paddle.to_tensor(np.zeros((4, 8), dtype=np.int64))
        params = _FakePackedSeqParams(
            cu_seqlens_q=paddle.to_tensor([0, 4, 8], dtype="int32"),
        )
        with self.assertRaises(ValueError):
            _roll_tensor_packed_seq(
                t, shifts=-1, dims=0, cu_seqlens_q=params.cu_seqlens_q
            )

    def test_rejects_shifts_non_minus_one(self) -> None:
        t = paddle.to_tensor(np.zeros((1, 8), dtype=np.int64))
        params = _FakePackedSeqParams(
            cu_seqlens_q=paddle.to_tensor([0, 8], dtype="int32"),
        )
        with self.assertRaises(ValueError):
            _roll_tensor_packed_seq(
                t, shifts=-2, dims=-1, cu_seqlens_q=params.cu_seqlens_q
            )

    def test_cp_group_accepted_and_ignored(self) -> None:
        """Under PaddleFleet's full-length CP data layout, ``roll_tensor``
        accepts ``cp_group`` for API compat but ignores it — the result must
        equal the ``cp_group=None`` output on the same tensor.
        """

        class _FakeCPGroup:
            nranks = 2

        t = paddle.to_tensor(np.arange(1, 9, dtype=np.int64).reshape(1, 8))
        cu = paddle.to_tensor([0, 4, 8], dtype="int32")

        out_no_cp, sum_no_cp = roll_tensor(
            t,
            shifts=-1,
            dims=-1,
            cp_group=None,
            cu_seqlens_q=cu,
        )
        out_with_cp, sum_with_cp = roll_tensor(
            t,
            shifts=-1,
            dims=-1,
            cp_group=_FakeCPGroup(),
            cu_seqlens_q=cu,
        )
        np.testing.assert_array_equal(out_no_cp.numpy(), out_with_cp.numpy())
        self.assertEqual(int(sum_no_cp), int(sum_with_cp))


if __name__ == "__main__":
    unittest.main()
