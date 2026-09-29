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

"""Behavior tests for DataGenerator (SFT data-making util).

DataGenerator wraps a data source and turns it into an *infinite* stream:
- __init__ stores the source and materializes one iterator over it.
- __iter__ returns self, so the object is its own iterator.
- __next__ pulls from the current iterator; on StopIteration it rebuilds a
  fresh iterator *from the stored source* and pulls again. This is what makes
  a re-iterable source (e.g. a list / IterableDataset) cycle forever.

Consequences that these tests pin down with hand-derived expectations:
- Element order and element identity are preserved on every pass.
- Wrap-around rebuilds the iterator from the live source object, so it is not
  a cached snapshot of the first pass (mutating the source changes later
  cycles).
- The infinite behavior depends on the source being re-iterable. A one-shot
  iterator (a generator) is *not* re-iterable: iter(gen) returns the same
  exhausted generator, so the second next() re-raises StopIteration and it
  propagates instead of cycling. An empty source likewise cannot cycle.

Expected values below are derived by hand from the contract, never read back
from the implementation.
"""

import unittest

try:
    from paddlefleet.cli.train.sft.make_data_utils import DataGenerator

    _IMPORT_ERROR = None
except ImportError as exc:  # local env has no paddle; skip honestly
    DataGenerator = None
    _IMPORT_ERROR = exc


@unittest.skipUnless(
    DataGenerator is not None,
    "DataGenerator import failed (paddle not installed in this env): "
    f"{_IMPORT_ERROR}",
)
class DataGeneratorBehaviorTest(unittest.TestCase):
    def test_first_pass_preserves_order(self):
        gen = DataGenerator([10, 20, 30, 40])
        got = [next(gen) for _ in range(4)]
        # Hand-derived: first pass yields the source in its given order.
        self.assertEqual(got, [10, 20, 30, 40])

    def test_wraps_around_infinitely(self):
        gen = DataGenerator([10, 20, 30])
        got = [next(gen) for _ in range(10)]
        # Hand-derived: [10,20,30] repeated, truncated at 10 items.
        # 2 full cycles (6) + partial cycle of 4 -> the 10th item is 10 again.
        self.assertEqual(got, [10, 20, 30, 10, 20, 30, 10, 20, 30, 10])

    def test_for_loop_iterates_infinitely(self):
        # Exercises __iter__ + __next__ together via the for-protocol.
        gen = DataGenerator([1, 2, 3])
        collected = []
        for i, value in enumerate(gen):
            collected.append(value)
            if i == 6:  # stop after 7 items
                break
        # Hand-derived: [1,2,3] cycled, first 7 items.
        self.assertEqual(collected, [1, 2, 3, 1, 2, 3, 1])

    def test_iter_returns_self(self):
        # Contract: the generator is its own iterator, so iter(gen) is gen.
        gen = DataGenerator([1, 2, 3])
        self.assertIs(iter(gen), gen)

    def test_stores_source_reference(self):
        # The exact source object must be retained: wrap-around re-iterates it,
        # so identity (not just equality) is the contract.
        source = [5, 6, 7]
        gen = DataGenerator(source)
        self.assertIs(gen.data_source, source)

    def test_yields_same_element_objects_in_order(self):
        # Element identity must survive iteration and wrap-around: the stream
        # hands back the very objects from the source, not copies.
        a, b, c = object(), object(), object()
        gen = DataGenerator([a, b, c])
        self.assertIs(next(gen), a)
        self.assertIs(next(gen), b)
        self.assertIs(next(gen), c)
        self.assertIs(next(gen), a)  # wrap yields the same first object

    def test_wrap_reiterates_live_source_not_cached_snapshot(self):
        # Wrap-around rebuilds iter() from the stored source, so an element
        # replaced after the first pass shows up in later cycles. This rejects
        # any implementation that replays a cached copy of the first pass.
        # (Length is kept fixed at 3 so the first pass's still-live list
        # iterator raises StopIteration exactly at the wrap point.)
        source = [10, 20, 30]
        gen = DataGenerator(source)
        self.assertEqual([next(gen) for _ in range(3)], [10, 20, 30])

        source[0] = 99  # replace an existing element in place

        # Hand-derived: the wrap rebuilds iter(source) over [99, 20, 30], then
        # wraps once more back to 99. A cached snapshot would yield 10 here.
        got = [next(gen) for _ in range(4)]
        self.assertEqual(got, [99, 20, 30, 99])

    def test_list_source_wraps_where_one_shot_generator_raises(self):
        # Same logical content [7, 8], two source kinds with different
        # re-iterability, giving different post-exhaustion behavior.

        # A list is re-iterable: iter(list) is fresh each time -> it cycles.
        list_gen = DataGenerator([7, 8])
        self.assertEqual([next(list_gen) for _ in range(3)], [7, 8, 7])

        # A generator is one-shot: iter(gen) returns the same, now-exhausted
        # generator, so the rebuilt iterator is empty and StopIteration
        # propagates out of __next__ on the wrap attempt.
        def one_shot():
            yield 7
            yield 8

        gen_gen = DataGenerator(one_shot())
        self.assertEqual(next(gen_gen), 7)
        self.assertEqual(next(gen_gen), 8)
        with self.assertRaises(StopIteration):
            next(gen_gen)

    def test_empty_source_raises_stop_iteration(self):
        # An empty source cannot cycle: the first next() finds nothing, the
        # rebuilt iterator is also empty, and StopIteration propagates rather
        # than looping forever.
        gen = DataGenerator([])
        with self.assertRaises(StopIteration):
            next(gen)


if __name__ == "__main__":
    unittest.main()
