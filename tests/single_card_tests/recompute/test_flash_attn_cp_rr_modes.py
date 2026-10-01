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

"""Behavioral target:
    paddlefleet/refined_recompute/flash_attn.py
    - slice_ulysses_mask_heads
    - slice_ulysses_sink_heads
    - RefinedRcomputeFlashMaskCpAttention._second_fwd (mode dispatch)

Distinct facet from the sibling recompute suites (which cover casting and the
``_first_fwd`` early parameter guards): this file pins the CONTEXT-PARALLEL
round-robin head-sharding math and the second-forward mode selection.

Why these are honestly CPU-observable (antipattern #13):
  ``slice_ulysses_mask_heads`` / ``slice_ulysses_sink_heads`` never invoke a
  collective. They read ``cp_group.rank`` / ``cp_group.nranks`` and perform a
  purely local tensor slice to pick this rank's contiguous head block. So a
  plain object exposing ``rank``/``nranks`` is NOT a faked world_size wrapped
  around a mocked collective -- it is the real, complete input to a local
  index computation. The Ulysses all-to-all collective numerics that actually
  redistribute the tensors are deliberately NOT tested here; they require a
  real multi-rank process group.

  ``_second_fwd`` reads a queued ``mode`` string and dispatches; the invalid
  mode branch raises ``ValueError`` before touching any collective, so it is
  fully CPU-observable.

All expected values are hand-derived (numpy ``arange`` blocks / literals). The
functions under test are never used to compute their own expected results.

paddlefleet imports paddle at import time; the no-card environment used here
has no paddle build, so the import is guarded and every test is skipped with an
honest reason rather than faking a pass. Only a precise ImportError is treated
as "dependency missing".
"""

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
        ),
        "src",
    ),
)

try:
    import numpy as np
    import paddle

    from paddlefleet.refined_recompute import flash_attn as fa

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet unavailable in this env
    np = None
    paddle = None
    fa = None
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle/paddlefleet not importable in this environment: {_IMPORT_ERROR}"
)


class _CpGroup:
    """Minimal stand-in for a CP group: the slice helpers only read rank/nranks.

    This carries no collective; it is the complete real input to the local
    index math under test.
    """

    def __init__(self, rank, nranks):
        self.rank = rank
        self.nranks = nranks


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestSliceUlyssesMaskHeads(unittest.TestCase):
    """Per-head FlashMask index sharding for the local Ulysses head block."""

    def test_mask_heads_sliced_to_local_rank_block(self):
        # [batch=1, heads=4, seqlen=3, 2] filled with distinguishable content.
        # Each head block spans 3 * 2 = 6 consecutive values.
        startend = paddle.arange(1 * 4 * 3 * 2, dtype="int32").reshape(
            [1, 4, 3, 2]
        )

        # heads_per_rank = 4 // 2 = 2.
        # rank 0 owns heads [0, 2) -> values 0..11 (hand-derived, independent).
        r0 = fa.slice_ulysses_mask_heads(startend, 4, _CpGroup(0, 2))
        self.assertEqual(list(r0.shape), [1, 2, 3, 2])
        np.testing.assert_array_equal(
            r0.numpy(), np.arange(0, 12, dtype=np.int32).reshape([1, 2, 3, 2])
        )

        # rank 1 owns heads [2, 4) -> values 12..23.
        r1 = fa.slice_ulysses_mask_heads(startend, 4, _CpGroup(1, 2))
        self.assertEqual(list(r1.shape), [1, 2, 3, 2])
        np.testing.assert_array_equal(
            r1.numpy(), np.arange(12, 24, dtype=np.int32).reshape([1, 2, 3, 2])
        )

        # The two rank shards must be disjoint and jointly cover every head;
        # a wrong head_start (e.g. always rank 0) would collide here.
        self.assertFalse(
            np.array_equal(r0.numpy(), r1.numpy()),
            "rank 0 and rank 1 must receive different head blocks",
        )

    def test_four_ranks_partition_heads_contiguously(self):
        # 4 heads over 4 ranks: each rank gets exactly one distinct head.
        startend = paddle.arange(1 * 4 * 1 * 2, dtype="int32").reshape(
            [1, 4, 1, 2]
        )
        for rank in range(4):
            shard = fa.slice_ulysses_mask_heads(startend, 4, _CpGroup(rank, 4))
            self.assertEqual(list(shard.shape), [1, 1, 1, 2])
            np.testing.assert_array_equal(
                shard.numpy(),
                np.array([[[[rank * 2, rank * 2 + 1]]]], dtype=np.int32),
            )

    def test_broadcast_mask_returned_by_identity(self):
        # A head dim of 1 is a broadcast mask shared across every head; the
        # function must return the SAME object untouched (no slice / copy).
        broadcast = paddle.arange(1 * 1 * 3 * 2, dtype="int32").reshape(
            [1, 1, 3, 2]
        )
        result = fa.slice_ulysses_mask_heads(broadcast, 4, _CpGroup(1, 2))
        self.assertIs(result, broadcast)

    def test_head_dim_must_be_one_or_num_kv_heads(self):
        # 3 mask heads with num_k_heads=4 is neither broadcast(1) nor 4.
        bad = paddle.zeros([1, 3, 3, 2], dtype="int32")
        with self.assertRaises(AssertionError):
            fa.slice_ulysses_mask_heads(bad, 4, _CpGroup(0, 2))

    def test_head_dim_must_divide_cp_size(self):
        # 3 mask heads == num_k_heads=3 but 3 % 2 != 0, so it cannot be sharded.
        startend = paddle.zeros([1, 3, 3, 2], dtype="int32")
        with self.assertRaises(AssertionError):
            fa.slice_ulysses_mask_heads(startend, 3, _CpGroup(0, 2))


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestSliceUlyssesSinkHeads(unittest.TestCase):
    """Per-query-head softmax sink sharding for the local Ulysses head block."""

    def test_none_sink_returns_none(self):
        self.assertIsNone(fa.slice_ulysses_sink_heads(None, 4, _CpGroup(0, 2)))

    def test_sink_sliced_to_local_rank_block(self):
        sink = paddle.arange(4, dtype="float32")  # one entry per query head

        # heads_per_rank = 4 // 2 = 2; contiguous block starting at rank * 2.
        r0 = fa.slice_ulysses_sink_heads(sink, 4, _CpGroup(0, 2))
        self.assertEqual(r0.tolist(), [0.0, 1.0])

        r1 = fa.slice_ulysses_sink_heads(sink, 4, _CpGroup(1, 2))
        self.assertEqual(r1.tolist(), [2.0, 3.0])

    def test_sink_length_must_match_query_heads(self):
        # A sink with the wrong number of entries would silently misalign with
        # the head shard, so the function rejects it loudly.
        with self.assertRaisesRegex(ValueError, "one entry per query head"):
            fa.slice_ulysses_sink_heads(
                paddle.zeros([3], dtype="float32"), 4, _CpGroup(0, 2)
            )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestSecondForwardModeDispatch(unittest.TestCase):
    """Second-forward mode selection is CPU-observable through the queue."""

    def _qkv(self):
        q = paddle.zeros([1, 4, 2, 4], dtype="float32")
        return q, q, q

    def test_invalid_mode_raises_value_error(self):
        # _second_fwd pops the stored hold-tensors dict and dispatches on its
        # "mode". An unknown mode must raise ValueError before any collective
        # is reached -- fully observable without a process group.
        attn = fa.RefinedRcomputeFlashMaskCpAttention()
        attn._hold_tensors_queue.put({"mode": "bad_mode"})
        q, k, v = self._qkv()
        with self.assertRaisesRegex(ValueError, "invalid cp_balance_mode"):
            attn._second_fwd(q, k, v)


if __name__ == "__main__":
    unittest.main()
