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

"""Unit tests for extract_local_zigzag_chunks + full-length CP roll semantics.

Verifies that:

1. ``extract_local_zigzag_chunks`` reproduces PaddleFleet ``scatter_balance``'s
   dualchunk layout: each rank owns
   ``chunk_start[interval*r : interval*(r+1)] + chunk_end[L-interval*(r+1) : L-interval*r]``.
2. Concatenating every rank's local slice, then re-ordering back to global
   position order, yields the original full-length tensor bit-for-bit.
3. Under PaddleFleet's full-length CP data layout, ``roll_tensor(cp_group=g)``
   produces the same result as ``roll_tensor(cp_group=None)`` — the cp_group
   parameter is API-compat only, no cross-rank P2P is used.
"""

from __future__ import annotations

import unittest

import numpy as np
import paddle

from paddlefleet.transformer.multi_token_prediction import (
    extract_local_zigzag_chunks,
    roll_tensor,
)


def _numpy_zigzag_positions(
    seq_len: int, cp_rank: int, cp_size: int
) -> np.ndarray:
    """Reference: which global positions belong to a given cp_rank."""
    assert seq_len % (2 * cp_size) == 0
    interval = seq_len // cp_size // 2
    start = list(range(interval * cp_rank, interval * (cp_rank + 1)))
    end = list(
        range(seq_len - interval * (cp_rank + 1), seq_len - interval * cp_rank)
    )
    return np.asarray(start + end, dtype=np.int64)


class TestExtractLocalZigzagChunks(unittest.TestCase):
    def test_cp1_returns_input(self) -> None:
        t = paddle.arange(0, 8, dtype="int64").reshape([1, 8])
        out = extract_local_zigzag_chunks(t, cp_rank=0, cp_size=1, axis=1)
        # Identity semantics (may or may not be same object)
        np.testing.assert_array_equal(out.numpy(), t.numpy())

    def test_cp2_seq8_layout(self) -> None:
        """cp_size=2, L=8.

        rank 0 owns global positions [0,1,6,7] → [A,B,G,H]
        rank 1 owns global positions [2,3,4,5] → [C,D,E,F]
        """
        t = paddle.arange(0, 8, dtype="int64").reshape([1, 8])
        r0 = extract_local_zigzag_chunks(t, cp_rank=0, cp_size=2, axis=1)
        r1 = extract_local_zigzag_chunks(t, cp_rank=1, cp_size=2, axis=1)
        np.testing.assert_array_equal(r0.numpy(), np.array([[0, 1, 6, 7]]))
        np.testing.assert_array_equal(r1.numpy(), np.array([[2, 3, 4, 5]]))

    def test_cp4_seq16_layout(self) -> None:
        """From the design doc example: cp_size=4, L=16."""
        t = paddle.arange(0, 16, dtype="int64").reshape([1, 16])
        expected_positions = {
            0: [0, 1, 14, 15],
            1: [2, 3, 12, 13],
            2: [4, 5, 10, 11],
            3: [6, 7, 8, 9],
        }
        for rank, expected in expected_positions.items():
            out = extract_local_zigzag_chunks(
                t, cp_rank=rank, cp_size=4, axis=1
            )
            np.testing.assert_array_equal(out.numpy(), np.array([expected]))

    def test_ranks_partition_the_sequence(self) -> None:
        """The union of all ranks' local slices must cover every global
        position exactly once (partition property)."""
        for cp_size in (2, 4, 8):
            seq_len = 32 * cp_size  # ensures divisibility by 2*cp_size
            t = paddle.arange(0, seq_len, dtype="int64").reshape([1, seq_len])
            seen = np.zeros(seq_len, dtype=np.int64)
            for rank in range(cp_size):
                local = extract_local_zigzag_chunks(
                    t, cp_rank=rank, cp_size=cp_size, axis=1
                )
                for v in local.numpy().reshape(-1).tolist():
                    seen[v] += 1
            np.testing.assert_array_equal(
                seen, np.ones(seq_len, dtype=np.int64)
            )

    def test_rejects_non_divisible_seq_len(self) -> None:
        t = paddle.arange(0, 10, dtype="int64").reshape([1, 10])
        with self.assertRaisesRegex(ValueError, r"not divisible by"):
            extract_local_zigzag_chunks(t, cp_rank=0, cp_size=2, axis=1)

    def test_higher_dim_tensor(self) -> None:
        """axis must be interpreted correctly for [B, L, H] tensors."""
        B, L, H = 2, 8, 4
        t = paddle.arange(0, B * L * H, dtype="int64").reshape([B, L, H])
        # rank 0 should get L' = L/2 = 4 along axis=1
        r0 = extract_local_zigzag_chunks(t, cp_rank=0, cp_size=2, axis=1)
        self.assertEqual(list(r0.shape), [B, 4, H])
        # Verify values: rank 0 sees seq positions [0, 1, 6, 7] on axis=1
        expected_seq_indices = [0, 1, 6, 7]
        for b in range(B):
            for i, seq_pos in enumerate(expected_seq_indices):
                np.testing.assert_array_equal(
                    r0[b, i].numpy(),
                    t[b, seq_pos].numpy(),
                )

    def test_negative_axis(self) -> None:
        """axis=-1 on a [B, L] tensor should behave the same as axis=1."""
        t = paddle.arange(0, 8, dtype="int64").reshape([1, 8])
        r_pos = extract_local_zigzag_chunks(t, cp_rank=0, cp_size=2, axis=1)
        r_neg = extract_local_zigzag_chunks(t, cp_rank=0, cp_size=2, axis=-1)
        np.testing.assert_array_equal(r_pos.numpy(), r_neg.numpy())


class TestFullLengthRollCPCompat(unittest.TestCase):
    """roll_tensor's cp_group is API-compat only; result matches cp_group=None."""

    class _FakeCPGroup:
        def __init__(self, nranks):
            self.nranks = nranks

    def test_non_packed_cp_group_ignored(self) -> None:
        t = paddle.arange(1, 9, dtype="int64").reshape([1, 8])
        out_no_cp, _ = roll_tensor(t, shifts=-1, dims=-1, cp_group=None)
        out_cp, _ = roll_tensor(
            t,
            shifts=-1,
            dims=-1,
            cp_group=self._FakeCPGroup(nranks=2),
        )
        np.testing.assert_array_equal(out_no_cp.numpy(), out_cp.numpy())

    def test_packed_cp_group_ignored(self) -> None:
        t = paddle.arange(1, 9, dtype="int64").reshape([1, 8])
        cu = paddle.to_tensor([0, 4, 8], dtype="int32")
        out_no_cp, _ = roll_tensor(
            t, shifts=-1, dims=-1, cp_group=None, cu_seqlens_q=cu
        )
        out_cp, _ = roll_tensor(
            t,
            shifts=-1,
            dims=-1,
            cp_group=self._FakeCPGroup(nranks=4),
            cu_seqlens_q=cu,
        )
        np.testing.assert_array_equal(out_no_cp.numpy(), out_cp.numpy())

    def test_packed_semantics_end_to_end(self) -> None:
        """Tokens [1..8] with cu=[0,4,8] shift by -1:

        expected rolled = [2, 3, 4, 0, 6, 7, 8, 0]
        """
        t = paddle.arange(1, 9, dtype="int64").reshape([1, 8])
        cu = paddle.to_tensor([0, 4, 8], dtype="int32")
        out, _ = roll_tensor(t, shifts=-1, dims=-1, cu_seqlens_q=cu)
        np.testing.assert_array_equal(
            out.numpy(),
            np.array([[2, 3, 4, 0, 6, 7, 8, 0]], dtype=np.int64),
        )


class TestFullLengthRollExtractRoundTrip(unittest.TestCase):
    """The full pipeline: roll full-length → extract per-rank → verify each rank
    sees its correctly-rolled zigzag positions.
    """

    def test_pipeline_cp2(self) -> None:
        seq_len = 8
        cp_size = 2
        # Full-length input_ids on every CP rank (per dataloader broadcast).
        t = paddle.arange(1, seq_len + 1, dtype="int64").reshape([1, seq_len])
        cu = paddle.to_tensor([0, 4, 8], dtype="int32")

        # Full-length roll (identical on every rank; deterministic).
        rolled_full, _ = roll_tensor(t, shifts=-1, dims=-1, cu_seqlens_q=cu)
        rolled_np = rolled_full.numpy().reshape(-1)  # [2,3,4,0,6,7,8,0]

        for rank in range(cp_size):
            local = extract_local_zigzag_chunks(
                rolled_full, cp_rank=rank, cp_size=cp_size, axis=1
            )
            expected_positions = _numpy_zigzag_positions(seq_len, rank, cp_size)
            expected_local = rolled_np[expected_positions]
            np.testing.assert_array_equal(
                local.numpy().reshape(-1), expected_local
            )

    def test_pipeline_multi_depth_cp2(self) -> None:
        """K=2 accumulated roll — depth k applies (k+1) rolls.

        For cu=[0,8,16] (two length-8 docs), tokens [1..16], depth 0 & 1
        outputs should each preserve doc boundaries.
        """
        seq_len = 16
        cp_size = 2
        t = paddle.arange(1, seq_len + 1, dtype="int64").reshape([1, seq_len])
        cu = paddle.to_tensor([0, 8, 16], dtype="int32")

        rolled = t
        for depth in range(2):
            rolled, _ = roll_tensor(rolled, shifts=-1, dims=-1, cu_seqlens_q=cu)

        # After 2 rolls, doc 0 [1..8] → [3,4,5,6,7,8,0,0]; doc 1 [9..16] →
        # [11,12,13,14,15,16,0,0].
        expected = np.array(
            [[3, 4, 5, 6, 7, 8, 0, 0, 11, 12, 13, 14, 15, 16, 0, 0]],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(rolled.numpy(), expected)

        # Verify each rank's local slice matches its zigzag chunks of `expected`.
        for rank in range(cp_size):
            local = extract_local_zigzag_chunks(
                rolled, cp_rank=rank, cp_size=cp_size, axis=1
            )
            expected_positions = _numpy_zigzag_positions(seq_len, rank, cp_size)
            expected_local = expected.reshape(-1)[expected_positions]
            np.testing.assert_array_equal(
                local.numpy().reshape(-1), expected_local
            )


class TestExtractLocalZigzagChunksRandom(unittest.TestCase):
    """500 random cases: extraction preserves values (partition + parity)."""

    def test_random_partition_and_parity(self) -> None:
        rng = np.random.default_rng(seed=20260812)
        for _ in range(500):
            cp_size = int(rng.choice([2, 4, 8]))
            seq_len = int(rng.integers(1, 8)) * 2 * cp_size  # divisible
            B = int(rng.integers(1, 4))
            arr = rng.integers(-1000, 1000, size=(B, seq_len)).astype(np.int64)
            t = paddle.to_tensor(arr)

            # Reconstruct global tensor by scattering per-rank local slices
            # back to their global positions.
            reconstructed = np.zeros_like(arr)
            for rank in range(cp_size):
                local = extract_local_zigzag_chunks(
                    t, cp_rank=rank, cp_size=cp_size, axis=1
                ).numpy()
                positions = _numpy_zigzag_positions(seq_len, rank, cp_size)
                reconstructed[:, positions] = local
            np.testing.assert_array_equal(reconstructed, arr)


if __name__ == "__main__":
    unittest.main()
