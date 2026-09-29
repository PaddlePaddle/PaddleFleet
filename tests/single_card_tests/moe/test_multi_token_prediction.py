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

"""Behavior tests for the Multi-Token-Prediction sequence helpers.

Target production entry:
    paddlefleet.transformer.multi_token_prediction
        roll_tensor
        extract_local_zigzag_chunks
        extract_local_contiguous_chunk
        extract_local_cp_chunks

These are the pure tensor-bookkeeping primitives the erndata MTP path relies
on to build its per-depth prediction targets:

* ``roll_tensor`` left-shifts input_ids / labels / position_ids by one token so
  that the target of position ``i`` becomes the token at ``i+1``. The new-in
  slot created by the shift is filled with ``pad_value`` (0 for embeddings /
  ids, ``ignored_index`` for labels). With ``cu_seqlens_q`` the shift is done
  per packed document so a document's last token cannot leak the first token of
  the next document in as a target -- a mislabelled boundary silently trains on
  cross-document targets, so the boundary fill is the behaviour worth pinning.

* ``extract_local_{zigzag,contiguous}_chunk`` slice a full-length tensor (held
  on every CP rank) into the layout that ``config.cp_balance_mode`` scatters
  with; ``extract_local_cp_chunks`` dispatches between them. These are the
  layouts that decide which sequence positions each rank owns, so a wrong start
  offset or a wrong dispatch sends the rank the wrong tokens.

Every expected value below is derived by hand from the roll / slice definition,
never by calling the function under test. The helpers are single-process,
device-independent index math (paddle.roll / paddle.slice / paddle.concat), so
they are exercised for real on CPU. This does NOT verify any cross-rank CP
communication: it verifies the local-slice layout each rank computes from a
full-length tensor, which is exactly the documented contract of these
extraction helpers (they perform no collective).

CPU-only: this file needs a working Paddle install. When Paddle is not
importable the whole case is skipped with an honest reason rather than being
faked as passing.
"""

import unittest

try:
    import numpy as np
    import paddle

    from paddlefleet.transformer.multi_token_prediction import (
        extract_local_contiguous_chunk,
        extract_local_cp_chunks,
        extract_local_zigzag_chunks,
        roll_tensor,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet unavailable in this env
    paddle = None
    np = None
    roll_tensor = None
    extract_local_zigzag_chunks = None
    extract_local_contiguous_chunk = None
    extract_local_cp_chunks = None
    _IMPORT_ERROR = repr(exc)


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet not importable: {_IMPORT_ERROR}",
)
class TestRollTensor(unittest.TestCase):
    """roll_tensor shifts targets left by one and fills the new-in slot."""

    @classmethod
    def setUpClass(cls):
        # Pure index math; keep it off any visible accelerator so we only
        # claim to have verified the device-independent CPU path.
        paddle.set_device("cpu")

    def test_standard_left_shift_zero_fill(self):
        """[1,2,3,4] -> target of i is i+1, last slot zero-filled."""
        t = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0]])
        rolled, total = roll_tensor(t, shifts=-1, dims=-1)
        # position 0..2 read the next token; position 3 has no successor -> 0.
        self.assertEqual(rolled.numpy().tolist(), [[2.0, 3.0, 4.0, 0.0]])
        # returned sum mirrors MCore's num_tokens accumulation contract.
        self.assertAlmostEqual(float(total), 2.0 + 3.0 + 4.0 + 0.0, places=5)

    def test_standard_left_shift_pad_value_for_labels(self):
        """pad_value fills the boundary so labels mask it out of the loss."""
        t = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0]])
        rolled, total = roll_tensor(t, shifts=-1, dims=-1, pad_value=-100)
        self.assertEqual(rolled.numpy().tolist(), [[2.0, 3.0, 4.0, -100.0]])
        self.assertAlmostEqual(float(total), 2.0 + 3.0 + 4.0 - 100.0, places=5)

    def test_rejects_unsupported_shift(self):
        """Only single-token left shift is implemented; others must raise."""
        t = paddle.to_tensor([[1.0, 2.0, 3.0]])
        with self.assertRaises(ValueError):
            roll_tensor(t, shifts=-2, dims=-1)

    def test_packed_roll_does_not_cross_doc_boundary(self):
        """Two docs [1,2,3][4,5,6]: each doc's last slot zeroed, no leak."""
        t = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        cu = paddle.to_tensor([0, 3, 6], dtype="int32")
        rolled, total = roll_tensor(t, shifts=-1, dims=-1, cu_seqlens_q=cu)
        # doc0 -> [2,3,0]; doc1 -> [5,6,0]. Position 2 must be 0, NOT 4
        # (which would be doc1's first token leaking across the boundary).
        self.assertEqual(
            rolled.numpy().tolist(), [[2.0, 3.0, 0.0, 5.0, 6.0, 0.0]]
        )
        self.assertAlmostEqual(float(total), 2 + 3 + 0 + 5 + 6 + 0, places=5)

    def test_packed_roll_pad_value_at_each_boundary(self):
        """Label pad_value is written at every per-doc boundary."""
        t = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        cu = paddle.to_tensor([0, 3, 6], dtype="int32")
        rolled, _ = roll_tensor(
            t, shifts=-1, dims=-1, cu_seqlens_q=cu, pad_value=-100
        )
        self.assertEqual(
            rolled.numpy().tolist(),
            [[2.0, 3.0, -100.0, 5.0, 6.0, -100.0]],
        )

    def test_packed_roll_batch_flat_granularity(self):
        """cu spanning batch*seq flattens the batch, rolls, then restores.

        tensor [[1,2,3],[4,5,6]] with cu=[0,3,6] treats the batch as one flat
        [1,2,3,4,5,6] sequence of two length-3 docs, so each row is rolled
        within itself and its last slot zeroed.
        """
        t = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        cu = paddle.to_tensor([0, 3, 6], dtype="int32")
        rolled, total = roll_tensor(t, shifts=-1, dims=-1, cu_seqlens_q=cu)
        self.assertEqual(
            rolled.numpy().tolist(), [[2.0, 3.0, 0.0], [5.0, 6.0, 0.0]]
        )
        self.assertAlmostEqual(float(total), 2 + 3 + 0 + 5 + 6 + 0, places=5)


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet not importable: {_IMPORT_ERROR}",
)
class TestExtractLocalChunks(unittest.TestCase):
    """Local-slice layouts each CP rank computes from a full-length tensor.

    These verify only the local index math (no collective is invoked); the
    helpers are documented as extraction-only.
    """

    @classmethod
    def setUpClass(cls):
        paddle.set_device("cpu")

    def test_zigzag_ranks_own_mirrored_chunk_pair(self):
        """seq=8, cp=2, interval=2: rank r owns [start] + [mirrored end]."""
        full = paddle.arange(8, dtype="float32").reshape([1, 8])
        # rank 0: positions [0,1] and mirror [6,7]; rank 1: [2,3] and [4,5].
        r0 = extract_local_zigzag_chunks(full, cp_rank=0, cp_size=2, axis=1)
        r1 = extract_local_zigzag_chunks(full, cp_rank=1, cp_size=2, axis=1)
        self.assertEqual(r0.numpy().tolist(), [[0.0, 1.0, 6.0, 7.0]])
        self.assertEqual(r1.numpy().tolist(), [[2.0, 3.0, 4.0, 5.0]])
        # Together the two ranks partition every position exactly once.
        union = sorted(
            r0.numpy().reshape(-1).tolist() + r1.numpy().reshape(-1).tolist()
        )
        self.assertEqual(union, [float(i) for i in range(8)])

    def test_zigzag_requires_divisibility_by_two_cp(self):
        """seq not divisible by 2*cp_size is a hard error, not silent slice."""
        full = paddle.arange(6, dtype="float32").reshape([1, 6])
        with self.assertRaises(ValueError):
            extract_local_zigzag_chunks(full, cp_rank=0, cp_size=2, axis=1)

    def test_contiguous_ranks_own_disjoint_blocks(self):
        """seq=8, cp=2: rank 0 owns [0:4], rank 1 owns [4:8]."""
        full = paddle.arange(8, dtype="float32").reshape([1, 8])
        r0 = extract_local_contiguous_chunk(full, cp_rank=0, cp_size=2, axis=1)
        r1 = extract_local_contiguous_chunk(full, cp_rank=1, cp_size=2, axis=1)
        self.assertEqual(r0.numpy().tolist(), [[0.0, 1.0, 2.0, 3.0]])
        self.assertEqual(r1.numpy().tolist(), [[4.0, 5.0, 6.0, 7.0]])

    def test_contiguous_requires_divisibility_by_cp(self):
        """seq not divisible by cp_size must raise rather than mis-slice."""
        full = paddle.arange(6, dtype="float32").reshape([1, 6])
        with self.assertRaises(ValueError):
            extract_local_contiguous_chunk(full, cp_rank=0, cp_size=4, axis=1)

    def test_cp_dispatch_matches_underlying_layout(self):
        """extract_local_cp_chunks routes each mode to the right layout."""
        full = paddle.arange(8, dtype="float32").reshape([1, 8])
        dual = extract_local_cp_chunks(
            full, cp_rank=0, cp_size=2, axis=1, mode="dualchunk_allgather"
        )
        contig = extract_local_cp_chunks(
            full, cp_rank=0, cp_size=2, axis=1, mode="contiguous_allgather"
        )
        # dualchunk -> zigzag pair; contiguous -> leading block.
        self.assertEqual(dual.numpy().tolist(), [[0.0, 1.0, 6.0, 7.0]])
        self.assertEqual(contig.numpy().tolist(), [[0.0, 1.0, 2.0, 3.0]])

    def test_cp_dispatch_rejects_unsupported_mode(self):
        """An unvalidated cp_balance_mode is refused, not guessed."""
        full = paddle.arange(8, dtype="float32").reshape([1, 8])
        with self.assertRaises(ValueError):
            extract_local_cp_chunks(
                full, cp_rank=0, cp_size=2, axis=1, mode="contiguous_a2a"
            )

    def test_cp_size_one_returns_input_unchanged(self):
        """cp_size==1 is the documented identity path (same object back)."""
        full = paddle.arange(8, dtype="float32").reshape([1, 8])
        out = extract_local_cp_chunks(
            full, cp_rank=0, cp_size=1, axis=1, mode="dualchunk_allgather"
        )
        # Documented contract: returns tensor_full itself, not a copy.
        self.assertIs(out, full)


if __name__ == "__main__":
    unittest.main()
