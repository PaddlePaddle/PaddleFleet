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

"""Behavior tests for long-context token rebalancing helpers.

Module under test:
``paddlefleet.transformers.ernie4_5_moe_vl.model.longcontext_ops``

Two pieces are pure-Python scheduling logic and fully CPU-verifiable here:

* ``MaxHeap`` — a max-heap keyed on the *last* element of each stored tuple
  (implemented over ``heapq`` by negating that key). Used to always pull the
  largest remaining surplus/deficit pile.
* ``redistribute_tokens`` — given per-rank token counts, plans the moves that
  balance every rank to ``total // n`` (the first ``total % n`` largest piles
  keep one extra), aiming for few transfers. It returns ``Movement(src, dst,
  tokens)`` records.

All expected values below are derived independently by hand (heap pop order and
the balancing arithmetic traced on paper), never by calling the function under
test. For ``redistribute_tokens`` we additionally re-derive the per-rank target
counts with a separate implementation and assert that *applying the production
moves* reaches exactly those targets while conserving the token total.

``TensorBalanceByTokenType`` is a ``PyLayer`` that performs the real cross-rank
``isend``/``irecv``/``all_gather`` transfers planned above. Its numerics require
a genuine multi-rank process group and are out of scope for this no-card test
(faking ``world_size`` + mocking the collectives would only exercise a single
process); see the skipped case at the bottom.
"""

import unittest

from paddlefleet.transformers.ernie4_5_moe_vl.model.longcontext_ops import (
    MaxHeap,
    redistribute_tokens,
)


def independent_targets(piles):
    """Re-derive the balanced per-rank target counts without touching the
    production planner.

    Mirrors the documented contract: every rank gets ``total // n`` tokens and
    the ``total % n`` piles with the most tokens each get one extra. Ties are
    broken by original index (stable ordering), matching the descending sort in
    the production code. Used only to validate that applying the production
    moves lands on these targets; it does not generate the move list.
    """
    total = sum(piles)
    n = len(piles)
    m = total // n
    r = total % n
    targets = [m] * n
    order = sorted(range(n), key=lambda i: piles[i], reverse=True)
    for i in range(r):
        targets[order[i]] += 1
    return targets


def apply_moves(piles, moves):
    """Apply Movement records to a copy of ``piles`` and return the result."""
    result = list(piles)
    for move in moves:
        result[move.src] -= move.tokens
        result[move.dst] += move.tokens
    return result


class TestMaxHeap(unittest.TestCase):
    """Max-heap keyed on the last tuple element; returns original items."""

    def test_empty_heap_is_empty_and_zero_length(self):
        heap = MaxHeap()
        self.assertTrue(heap.is_empty())
        self.assertEqual(len(heap), 0)

    def test_pop_returns_items_in_descending_last_element_order(self):
        # First tuple field is a distinguishable id; ordering is by the LAST
        # field. Pop must return the whole original tuple (id preserved), not
        # the internal negated key, in strictly descending last-field order.
        heap = MaxHeap()
        heap.push((10, 5))
        heap.push((20, 3))
        heap.push((30, 8))
        heap.push((40, 1))
        self.assertFalse(heap.is_empty())
        self.assertEqual(len(heap), 4)
        popped = [heap.pop() for _ in range(4)]
        self.assertEqual(popped, [(30, 8), (10, 5), (20, 3), (40, 1)])
        self.assertTrue(heap.is_empty())

    def test_top_returns_max_without_removing(self):
        heap = MaxHeap([(0, 5), (1, 3), (2, 8)])
        self.assertEqual(heap.top(), (2, 8))
        self.assertEqual(len(heap), 3)  # top must not remove
        # The same max is still there and comes out first on pop.
        self.assertEqual(heap.pop(), (2, 8))
        self.assertEqual(len(heap), 2)

    def test_init_from_data_orders_by_last_element(self):
        data = [(0, 3), (1, 7), (2, 1), (3, 5)]
        heap = MaxHeap(data)
        popped = [heap.pop() for _ in range(4)]
        self.assertEqual(popped, [(1, 7), (3, 5), (0, 3), (2, 1)])

    def test_pop_from_empty_raises_index_error(self):
        heap = MaxHeap()
        with self.assertRaises(IndexError):
            heap.pop()

    def test_top_from_empty_raises_index_error(self):
        heap = MaxHeap()
        with self.assertRaises(IndexError):
            heap.top()

    def test_ties_on_last_element_keep_every_item(self):
        # Equal last-field items must not be dropped/merged; the multiset of
        # returned items is preserved and last-fields come out non-increasing.
        data = [(0, 5), (1, 5), (2, 3), (3, 5)]
        heap = MaxHeap(data)
        popped = [heap.pop() for _ in range(len(data))]
        self.assertCountEqual(popped, data)
        last_fields = [item[-1] for item in popped]
        self.assertEqual(last_fields, sorted(last_fields, reverse=True))

    def test_push_grows_length_and_updates_max(self):
        heap = MaxHeap([(0, 2)])
        self.assertEqual(heap.top(), (0, 2))
        heap.push((1, 9))
        self.assertEqual(len(heap), 2)
        self.assertEqual(heap.top(), (1, 9))  # new larger item becomes max


class TestRedistributeTokens(unittest.TestCase):
    """Token-balancing move planner: exact plans + conservation invariants."""

    def _as_tuples(self, moves):
        return [(m.src, m.dst, m.tokens) for m in moves]

    def test_balanced_piles_need_no_moves(self):
        self.assertEqual(redistribute_tokens([5, 5, 5, 5]), [])

    def test_single_pile_needs_no_moves(self):
        self.assertEqual(redistribute_tokens([10]), [])

    def test_exact_plan_even_total(self):
        # total=12, n=4 -> target 3 each. Surplus {0:+3, 2:+1};
        # deficit {1:-1, 3:-3}. Largest surplus (pile 0) pairs with largest
        # deficit (pile 3) moving 3; then pile 2 -> pile 1 moving 1.
        moves = redistribute_tokens([6, 2, 4, 0])
        self.assertEqual(self._as_tuples(moves), [(0, 3, 3), (2, 1, 1)])
        self.assertEqual(apply_moves([6, 2, 4, 0], moves), [3, 3, 3, 3])

    def test_exact_plan_with_remainder(self):
        # total=17, n=4 -> base target 4; remainder 1 goes to the largest pile
        # (index 0, stable tie among the 5s) -> targets [5,4,4,4].
        # Surplus {1:+1, 2:+1}; deficit {3:-2}. Pile 1 sends 1 to pile 3
        # (remaining deficit 1), then pile 2 sends 1 to pile 3.
        moves = redistribute_tokens([5, 5, 5, 2])
        self.assertEqual(self._as_tuples(moves), [(1, 3, 1), (2, 3, 1)])
        self.assertEqual(apply_moves([5, 5, 5, 2], moves), [5, 4, 4, 4])

    def test_exact_plan_one_big_surplus_feeds_many(self):
        # total=12, n=4 -> target 3. Only pile 0 has surplus (+6); it feeds the
        # three deficit piles in ascending index order, 2 tokens each.
        moves = redistribute_tokens([9, 1, 1, 1])
        self.assertEqual(
            self._as_tuples(moves), [(0, 1, 2), (0, 2, 2), (0, 3, 2)]
        )
        self.assertEqual(apply_moves([9, 1, 1, 1], moves), [3, 3, 3, 3])

    def test_moves_reach_targets_and_conserve_tokens(self):
        # For a range of configs, the planned moves must balance every pile to
        # the independently derived targets, conserve the total, and each move
        # must be well-formed (positive amount, distinct source/destination).
        configs = [
            [6, 2, 4, 0],
            [5, 5, 5, 2],
            [9, 1, 1, 1],
            [3, 1],
            [7, 7, 0, 0, 1],
            [2, 0],
            [0, 0, 1, 0],
            [12, 8, 5, 3, 2],
        ]
        for piles in configs:
            with self.subTest(piles=piles):
                moves = redistribute_tokens(piles)
                targets = independent_targets(piles)
                self.assertEqual(apply_moves(piles, moves), targets)
                self.assertEqual(sum(apply_moves(piles, moves)), sum(piles))
                for move in moves:
                    self.assertGreater(move.tokens, 0)
                    self.assertNotEqual(move.src, move.dst)

    def test_targets_differ_by_at_most_one(self):
        # Sanity on the reference itself: balanced targets span at most 1.
        for piles in ([6, 2, 4, 0], [5, 5, 5, 2], [7, 7, 0, 0, 1]):
            targets = independent_targets(piles)
            self.assertLessEqual(max(targets) - min(targets), 1)

    def test_movement_records_expose_named_fields(self):
        move = redistribute_tokens([6, 2, 4, 0])[0]
        self.assertEqual((move.src, move.dst, move.tokens), (0, 3, 3))


class TestTensorBalanceByTokenType(unittest.TestCase):
    """Cross-rank rebalancing PyLayer (multi-card; not verified on CPU)."""

    def test_forward_backward_numerics_deferred_to_multi_card(self):
        # TensorBalanceByTokenType.forward/backward drive real
        # isend/irecv/all_gather over a hybrid model-parallel group: the
        # per-type token counts are turned into a move plan (redistribute_tokens
        # above) and tensor slices are physically exchanged between ranks, then
        # scattered back on the reverse pass. Correctness (which slice each peer
        # receives, ordering after concat, gradient routing) only manifests with
        # multiple ranks holding distinct data. Faking world_size + mocking the
        # collectives would exercise a single process and prove nothing about
        # the exchange (antipattern: single-process/mocked collective posing as
        # multi-card numerics). This belongs in a real multi-card process group.
        self.skipTest(
            "TensorBalanceByTokenType exchanges tensors across ranks via "
            "isend/irecv/all_gather; requires a real multi-card process group, "
            "verified under multi-card, not on CPU (no-card)."
        )


if __name__ == "__main__":
    unittest.main()
