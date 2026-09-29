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

"""Behavior tests for paddlefleet.datasets.reader.mix_datasets.

These tests exercise the real mixing strategies (concat / random /
interleave) with content-distinguishable, fixed inputs and independently
derived expectations. Each source uses uniquely labeled items so that the
assertions can observe per-source sample identity, ordering, wrap-around and
mixing proportions rather than just counts or shapes. The multi-source
wrapper is replaced by a tiny real (non-mock) stub exposing ``_task_group``,
the only attribute BaseMixDataset consumes; the mixing logic under test is
kept real. All tests run on CPU and require no accelerator.
"""

import unittest

from paddlefleet.datasets.reader.mix_datasets import (
    ConcatDataset,
    InterLeaveDataset,
    RandomDataset,
    create_dataset_instance,
)


class _ListDataset:
    """A minimal iterable dataset backed by an in-memory list.

    Re-iterating restarts from the beginning, matching the contract the
    mixing datasets rely on when they call ``list(iter(dataset))`` or wrap a
    source in an infinite iterator.
    """

    def __init__(self, items):
        self._items = list(items)

    def __iter__(self):
        return iter(self._items)


class _MultiSourceStub:
    """Real stand-in for MultiSourceDataset exposing only ``_task_group``.

    BaseMixDataset reads ``task['dataset']`` and ``task['prob']`` from this
    attribute; nothing else is touched, so no file/tokenizer machinery is
    required.
    """

    def __init__(self, sources_and_probs):
        self._task_group = [
            {"dataset": ds, "prob": prob} for ds, prob in sources_and_probs
        ]


def _make_config(mix_strategy, **overrides):
    """Build a dataset_config with the four required keys plus overrides."""
    config = {
        "mix_strategy": mix_strategy,
        "random_seed": 42,
        "random_shuffle": False,
        "num_samples_each_epoch": 8,
    }
    config.update(overrides)
    return config


def _labeled(prefix, n):
    """Return ``n`` uniquely labeled items, e.g. ['a0', 'a1', ...]."""
    return [f"{prefix}{i}" for i in range(n)]


class TestProbabilityNormalization(unittest.TestCase):
    """BaseMixDataset.__init__ normalizes source probabilities."""

    def test_unnormalized_probs_scaled_to_exact_values(self):
        # probs sum to 4.0 -> each divided by 4.0 -> [0.75, 0.25].
        ms = _MultiSourceStub(
            [(_ListDataset(["a0"]), 3.0), (_ListDataset(["b0"]), 1.0)]
        )
        ds = ConcatDataset(ms, **_make_config("concat"))
        self.assertAlmostEqual(ds.datasets_prob[0], 0.75, places=7)
        self.assertAlmostEqual(ds.datasets_prob[1], 0.25, places=7)

    def test_already_normalized_probs_left_unchanged(self):
        ms = _MultiSourceStub(
            [(_ListDataset(["a0"]), 0.3), (_ListDataset(["b0"]), 0.7)]
        )
        ds = ConcatDataset(ms, **_make_config("concat"))
        # Sum is already 1.0, so values must be passed through verbatim.
        self.assertEqual(ds.datasets_prob, [0.3, 0.7])


class TestConcatDataset(unittest.TestCase):
    """ConcatDataset preserves every item and source order."""

    def test_iteration_yields_sources_back_to_back_in_order(self):
        ds_a = _ListDataset(_labeled("a", 3))
        ds_b = _ListDataset(_labeled("b", 2))
        ms = _MultiSourceStub([(ds_a, 0.5), (ds_b, 0.5)])
        concat = ConcatDataset(ms, **_make_config("concat"))
        # No shuffle: source A in order, then source B in order.
        self.assertEqual(list(concat), ["a0", "a1", "a2", "b0", "b1"])

    def test_len_equals_total_item_count(self):
        ds_a = _ListDataset(_labeled("a", 7))
        ds_b = _ListDataset(_labeled("b", 3))
        ms = _MultiSourceStub([(ds_a, 0.5), (ds_b, 0.5)])
        concat = ConcatDataset(ms, **_make_config("concat"))
        self.assertEqual(len(concat), 10)

    def test_epoch_index_advances_after_full_pass(self):
        ms = _MultiSourceStub([(_ListDataset(_labeled("a", 3)), 1.0)])
        concat = ConcatDataset(ms, **_make_config("concat"))
        self.assertEqual(concat.epoch_index, 0)
        list(concat)
        self.assertEqual(concat.epoch_index, 1)
        list(concat)
        self.assertEqual(concat.epoch_index, 2)


class TestRandomDataset(unittest.TestCase):
    """RandomDataset draws per-source counts proportional to probability."""

    def test_unshuffled_draw_respects_proportions_and_order(self):
        # target counts = int(prob * num_samples_each_epoch)
        #   source A: int(0.75 * 8) = 6  (wraps its 3 items twice)
        #   source B: int(0.25 * 8) = 2
        ds_a = _ListDataset(_labeled("a", 3))
        ds_b = _ListDataset(_labeled("b", 4))
        ms = _MultiSourceStub([(ds_a, 0.75), (ds_b, 0.25)])
        rand = RandomDataset(
            ms, **_make_config("random", num_samples_each_epoch=8)
        )
        # A contributes 6 items looping 0,1,2,0,1,2; B contributes b0,b1.
        self.assertEqual(
            list(rand),
            ["a0", "a1", "a2", "a0", "a1", "a2", "b0", "b1"],
        )

    def test_reverse_flips_the_emitted_sequence(self):
        ds_a = _ListDataset(_labeled("a", 3))
        ds_b = _ListDataset(_labeled("b", 4))
        ms = _MultiSourceStub([(ds_a, 0.75), (ds_b, 0.25)])
        rand = RandomDataset(
            ms,
            **_make_config("random", num_samples_each_epoch=8, reverse=True),
        )
        forward = ["a0", "a1", "a2", "a0", "a1", "a2", "b0", "b1"]
        self.assertEqual(list(rand), forward[::-1])

    def test_len_is_configured_num_samples(self):
        ms = _MultiSourceStub([(_ListDataset(_labeled("a", 3)), 1.0)])
        rand = RandomDataset(
            ms, **_make_config("random", num_samples_each_epoch=50)
        )
        self.assertEqual(len(rand), 50)


def _subsequence(items, prefix):
    """Items keeping only those starting with ``prefix``, order preserved."""
    return [it for it in items if it.startswith(prefix)]


class TestInterLeaveDataset(unittest.TestCase):
    """InterLeaveDataset samples sources sequentially with wrap-around."""

    def test_single_source_is_used_once_in_order(self):
        # One source, prob 1.0: every draw hits it; on the first full pass it
        # is exhausted and construction stops -> exactly the source, in order.
        ms = _MultiSourceStub([(_ListDataset(_labeled("a", 5)), 1.0)])
        ds = InterLeaveDataset(ms, **_make_config("interleave_over"))
        self.assertEqual(ds.mode, "oversampling")
        self.assertEqual(len(ds), 5)
        self.assertEqual(list(ds), ["a0", "a1", "a2", "a3", "a4"])

    def test_oversampling_covers_all_items_of_every_source(self):
        # oversampling (all_exhausted): keep drawing until BOTH sources have
        # been exhausted at least once, so every distinct item must appear.
        ds_a = _ListDataset(_labeled("a", 3))
        ds_b = _ListDataset(_labeled("b", 5))
        ms = _MultiSourceStub([(ds_a, 0.5), (ds_b, 0.5)])
        ds = InterLeaveDataset(ms, **_make_config("interleave_over"))
        self.assertEqual(ds.mode, "oversampling")

        emitted = list(ds)
        a_items = _subsequence(emitted, "a")
        b_items = _subsequence(emitted, "b")

        # Full coverage of each source (guaranteed by all_exhausted stop).
        self.assertEqual(set(a_items), set(_labeled("a", 3)))
        self.assertEqual(set(b_items), set(_labeled("b", 5)))

        # Within each source, items are consumed sequentially and wrap around
        # its own index space, independent of which source the RNG picked.
        for j, item in enumerate(a_items):
            self.assertEqual(item, f"a{j % 3}")
        for j, item in enumerate(b_items):
            self.assertEqual(item, f"b{j % 5}")

        # Emitted length is exactly the two subsequences combined.
        self.assertEqual(len(emitted), len(a_items) + len(b_items))
        self.assertEqual(len(ds), len(emitted))

    def test_upsampling_stops_at_first_exhaustion_without_repeats(self):
        # upsampling (first_exhausted): stops the moment any source finishes
        # its first pass, so no source is ever sampled beyond its size.
        ds_a = _ListDataset(_labeled("a", 3))
        ds_b = _ListDataset(_labeled("b", 5))
        ms = _MultiSourceStub([(ds_a, 0.5), (ds_b, 0.5)])
        ds = InterLeaveDataset(ms, **_make_config("interleave_under"))
        self.assertEqual(ds.mode, "upsampling")

        emitted = list(ds)
        a_items = _subsequence(emitted, "a")
        b_items = _subsequence(emitted, "b")

        # No wrap-around: each source drawn at most its own size, in order.
        self.assertLessEqual(len(a_items), 3)
        self.assertLessEqual(len(b_items), 5)
        for j, item in enumerate(a_items):
            self.assertEqual(item, f"a{j}")
        for j, item in enumerate(b_items):
            self.assertEqual(item, f"b{j}")

        # At least one source triggered the stop by being fully consumed.
        self.assertTrue(len(a_items) == 3 or len(b_items) == 5)
        self.assertEqual(len(ds), len(emitted))


class TestCreateDatasetInstance(unittest.TestCase):
    """create_dataset_instance dispatches names to working instances."""

    def test_concat_name_builds_functional_concat_dataset(self):
        ds_a = _ListDataset(_labeled("a", 2))
        ds_b = _ListDataset(_labeled("b", 2))
        ms = _MultiSourceStub([(ds_a, 0.5), (ds_b, 0.5)])
        result = create_dataset_instance("concat", ms, **_make_config("concat"))
        self.assertIsInstance(result, ConcatDataset)
        # Dispatch produced a real, iterable concat of both sources.
        self.assertEqual(list(result), ["a0", "a1", "b0", "b1"])

    def test_random_name_builds_random_dataset_with_configured_len(self):
        ms = _MultiSourceStub([(_ListDataset(_labeled("a", 3)), 1.0)])
        result = create_dataset_instance(
            "random", ms, **_make_config("random", num_samples_each_epoch=20)
        )
        self.assertIsInstance(result, RandomDataset)
        self.assertEqual(len(result), 20)

    def test_interleave_under_maps_to_upsampling_mode(self):
        ms = _MultiSourceStub([(_ListDataset(_labeled("a", 3)), 1.0)])
        result = create_dataset_instance(
            "interleave_under", ms, **_make_config("interleave_under")
        )
        self.assertIsInstance(result, InterLeaveDataset)
        self.assertEqual(result.mode, "upsampling")

    def test_interleave_over_maps_to_oversampling_mode(self):
        ms = _MultiSourceStub([(_ListDataset(_labeled("a", 3)), 1.0)])
        result = create_dataset_instance(
            "interleave_over", ms, **_make_config("interleave_over")
        )
        self.assertIsInstance(result, InterLeaveDataset)
        self.assertEqual(result.mode, "oversampling")

    def test_unknown_name_returns_none(self):
        self.assertIsNone(create_dataset_instance("does_not_exist"))


if __name__ == "__main__":
    unittest.main()
