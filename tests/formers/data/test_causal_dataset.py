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

"""Behavior tests for the GPT/causal dataset index-building machinery.

These are CPU / no-card tests. They exercise the real sample-construction
helpers (document index, sample index, shuffle index, epoch counting and the
train/valid/test document split) and the real indexed-dataset file I/O, and
compare against expectations derived independently by hand.  Paddle is a real
dependency of the modules under test and is imported normally (not mocked).
"""

import math
import os
import shutil
import tempfile
import unittest

import numpy as np

from paddlefleet.data.causal_dataset import (
    _build_doc_idx,
    _build_sample_idx,
    _build_shuffle_idx,
    _num_epochs,
    _num_tokens,
    check_data_split,
    get_datasets_weights_and_num_samples,
    get_train_valid_test_split_,
)
from paddlefleet.data.indexed_dataset import (
    MMapIndexedDataset,
    MMapIndexedDatasetBuilder,
)


class TestNumTokens(unittest.TestCase):
    """_num_tokens must sum sizes over the selected documents only."""

    def test_counts_only_selected_documents(self):
        # Per-sentence token counts; documents index into this array.
        sizes = np.array([4, 6, 5, 9], dtype=np.int32)

        # Full selection -> 4 + 6 + 5 + 9.
        self.assertEqual(int(_num_tokens(np.array([0, 1, 2, 3]), sizes)), 24)
        # A subset must exclude the unselected documents (catches a bug that
        # ignores `documents` and always sums everything).
        self.assertEqual(int(_num_tokens(np.array([1, 2]), sizes)), 11)
        self.assertEqual(int(_num_tokens(np.array([0, 3]), sizes)), 13)
        self.assertEqual(int(_num_tokens(np.array([2]), sizes)), 5)


class TestNumEpochs(unittest.TestCase):
    """_num_epochs counts how many passes over the data are needed.

    Contract (see _num_epochs): keep adding whole epochs until
    (total_tokens - 1) // seq_length >= num_samples.
    """

    def test_single_epoch_is_enough(self):
        # tokens_per_epoch=15, seq_length=4 -> one epoch yields
        # (15 - 1) // 4 = 3 samples, which already covers num_samples=3.
        self.assertEqual(_num_epochs(15, 4, 3), 1)

    def test_two_epochs_required(self):
        # One epoch only gives 3 samples (< 5); two epochs give
        # (30 - 1) // 4 = 7 >= 5, so exactly 2 epochs are needed.
        self.assertEqual(_num_epochs(15, 4, 5), 2)
        self.assertEqual(_num_epochs(15, 4, 7), 2)

    def test_boundary_needs_extra_epoch(self):
        # Asking for one more sample than a single epoch provides forces a
        # second epoch.
        self.assertEqual(_num_epochs(15, 4, 4), 2)


class TestBuildDocIdx(unittest.TestCase):
    """_build_doc_idx lays out documents across epochs and shuffles them."""

    def test_no_separate_last_epoch_repeats_each_document_per_epoch(self):
        documents = np.arange(0, 5, dtype=np.int32)
        num_epochs = 3
        rng = np.random.RandomState(2024)
        doc_idx = _build_doc_idx(documents, num_epochs, rng, False)

        # One entry per (epoch, document).
        self.assertEqual(doc_idx.shape[0], num_epochs * len(documents))
        self.assertEqual(doc_idx.dtype, np.int32)
        # Every document must appear exactly `num_epochs` times: shuffling may
        # not drop or duplicate documents (a multiset check).
        self.assertEqual(
            sorted(doc_idx.tolist()),
            sorted(list(documents) * num_epochs),
        )

    def test_same_seed_is_reproducible(self):
        documents = np.arange(0, 8, dtype=np.int32)
        first = _build_doc_idx(documents, 2, np.random.RandomState(7), False)
        second = _build_doc_idx(documents, 2, np.random.RandomState(7), False)
        np.testing.assert_array_equal(first, second)

    def test_shuffle_actually_reorders(self):
        # With 32 documents an unshuffled (sorted) result is astronomically
        # unlikely, so a sorted output would indicate the shuffle was dropped.
        documents = np.arange(0, 32, dtype=np.int32)
        doc_idx = _build_doc_idx(documents, 1, np.random.RandomState(99), False)
        self.assertNotEqual(doc_idx.tolist(), sorted(doc_idx.tolist()))

    def test_separate_last_epoch_splits_into_two_shuffled_blocks(self):
        documents = np.arange(0, 5, dtype=np.int32)
        num_epochs = 3
        rng = np.random.RandomState(11)
        doc_idx = _build_doc_idx(documents, num_epochs, rng, True)

        self.assertEqual(doc_idx.shape[0], num_epochs * len(documents))
        head = doc_idx[: (num_epochs - 1) * len(documents)]
        tail = doc_idx[(num_epochs - 1) * len(documents) :]
        # The last epoch is shuffled separately: its block is a permutation of
        # the documents, and the earlier epochs form their own multiset.
        self.assertEqual(sorted(tail.tolist()), sorted(documents.tolist()))
        self.assertEqual(
            sorted(head.tolist()),
            sorted(list(documents) * (num_epochs - 1)),
        )


class TestBuildSampleIdx(unittest.TestCase):
    """_build_sample_idx maps each sample to (doc_idx position, offset).

    Each sample spans seq_length + 1 tokens and the last token of a sample
    overlaps the first token of the next one (hence the -1 in the impl).
    Expected sample boundaries below are worked out by hand for the given
    document sizes and document order.
    """

    def test_single_epoch_sample_boundaries(self):
        # sizes[doc] token counts; doc_idx is the (identity) document order.
        sizes = np.array([4, 6, 5], dtype=np.int32)
        doc_idx = np.array([0, 1, 2], dtype=np.int32)
        seq_length = 4
        tokens_per_epoch = 15  # 4 + 6 + 5

        sample_idx = _build_sample_idx(
            sizes, doc_idx, seq_length, num_epochs=1, tokens_per_epoch=15
        )

        # (15 - 1) // 4 = 3 samples -> 4 boundary rows.
        self.assertEqual(sample_idx.dtype, np.int32)
        self.assertEqual(list(sample_idx.shape), [4, 2])
        # Hand-derived boundaries: sample 0 starts at doc-position 0 offset 0;
        # a 5-token window consumes doc0 (4 tokens) + 1 token of doc1, etc.
        np.testing.assert_array_equal(
            sample_idx,
            np.array([[0, 0], [1, 0], [1, 4], [2, 2]], dtype=np.int32),
        )
        _ = tokens_per_epoch  # documented for the reader

    def test_two_epoch_sample_boundaries_cross_epoch(self):
        # Two epochs are stitched together via a repeated doc_idx; sample
        # windows continue across the epoch boundary.
        sizes = np.array([4, 6, 5], dtype=np.int32)
        doc_idx = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
        seq_length = 4

        sample_idx = _build_sample_idx(
            sizes, doc_idx, seq_length, num_epochs=2, tokens_per_epoch=15
        )

        # (2 * 15 - 1) // 4 = 7 samples -> 8 boundary rows.
        self.assertEqual(list(sample_idx.shape), [8, 2])
        np.testing.assert_array_equal(
            sample_idx,
            np.array(
                [
                    [0, 0],
                    [1, 0],
                    [1, 4],
                    [2, 2],
                    [3, 1],
                    [4, 1],
                    [4, 5],
                    [5, 3],
                ],
                dtype=np.int32,
            ),
        )


class TestBuildShuffleIdx(unittest.TestCase):
    """_build_shuffle_idx permutes sample indices, optionally in two regions."""

    def test_single_region_is_a_full_permutation(self):
        n = 64
        idx = _build_shuffle_idx(n, n, np.random.RandomState(3))
        self.assertEqual(idx.dtype, np.uint32)
        self.assertEqual(idx.shape[0], n)
        # A permutation of [0, n): nothing lost, nothing duplicated.
        self.assertEqual(sorted(idx.tolist()), list(range(n)))
        # And it must actually be shuffled, not the identity ordering.
        self.assertNotEqual(idx.tolist(), list(range(n)))

    def test_two_regions_do_not_leak_across_the_boundary(self):
        num_samples = 40
        total_size = 64
        idx = _build_shuffle_idx(
            num_samples, total_size, np.random.RandomState(5)
        )

        self.assertEqual(idx.shape[0], total_size)
        first_region = idx[:num_samples]
        last_region = idx[num_samples:]
        # The first region is a permutation of [0, num_samples) and the second
        # of [num_samples, total_size); indices never cross the boundary.
        self.assertEqual(
            sorted(first_region.tolist()), list(range(num_samples))
        )
        self.assertEqual(
            sorted(last_region.tolist()), list(range(num_samples, total_size))
        )

    def test_same_seed_is_reproducible(self):
        first = _build_shuffle_idx(40, 64, np.random.RandomState(21))
        second = _build_shuffle_idx(40, 64, np.random.RandomState(21))
        np.testing.assert_array_equal(first, second)


class TestGetTrainValidTestSplit(unittest.TestCase):
    """get_train_valid_test_split_ turns a split string into doc boundaries."""

    def test_exact_boundaries_for_8_1_1(self):
        # "8,1,1" over 100 documents -> train [0,80), valid [80,90),
        # test [90,100).
        result = get_train_valid_test_split_("8,1,1", 100)
        self.assertEqual(result, [0, 80, 90, 100])

    def test_rounding_diff_correction_for_1_1_1(self):
        # Each third rounds to 33 (round(100/3) == 33), giving a running sum of
        # 99; the impl corrects the -1 shortfall by shifting the tail up, so
        # the boundaries become [0, 34, 67, 100] and still end exactly at 100.
        result = get_train_valid_test_split_("1,1,1", 100)
        self.assertEqual(result, [0, 34, 67, 100])
        self.assertEqual(result[-1], 100)

    def test_slash_separator_matches_comma(self):
        # "/"-separated strings must parse the same way as comma-separated.
        self.assertEqual(
            get_train_valid_test_split_("8/1/1", 100),
            get_train_valid_test_split_("8,1,1", 100),
        )


class TestCheckDataSplit(unittest.TestCase):
    """check_data_split enforces that enabled phases get a non-zero split."""

    def test_valid_split_passes(self):
        # No exception expected for a fully-populated split.
        self.assertIsNone(
            check_data_split(
                "0.8,0.1,0.1", do_train=True, do_eval=True, do_predict=True
            )
        )

    def test_zero_split_raises_only_for_the_enabled_phase(self):
        # Train split is 0 while training is requested -> rejected.
        with self.assertRaises(ValueError):
            check_data_split(
                "0,0.5,0.5", do_train=True, do_eval=True, do_predict=True
            )
        # Same split, but training is disabled -> the zero train share is fine.
        # This distinguishes real flag consumption from an unconditional guard.
        self.assertIsNone(
            check_data_split(
                "0,0.5,0.5", do_train=False, do_eval=True, do_predict=True
            )
        )

    def test_zero_eval_and_zero_predict_are_rejected_when_enabled(self):
        with self.assertRaises(ValueError):
            check_data_split(
                "0.5,0,0.5", do_train=True, do_eval=True, do_predict=True
            )
        with self.assertRaises(ValueError):
            check_data_split(
                "0.5,0.5,0", do_train=True, do_eval=True, do_predict=True
            )

    def test_zero_total_split_raises_assertion(self):
        with self.assertRaises(AssertionError):
            check_data_split(
                "0,0,0", do_train=False, do_eval=False, do_predict=False
            )


class TestGetDatasetsWeightsAndNumSamples(unittest.TestCase):
    """get_datasets_weights_and_num_samples parses weighted data prefixes."""

    def test_weights_are_normalized_and_prefixes_extracted(self):
        prefixes, weights, _ = get_datasets_weights_and_num_samples(
            ["3.0", "/a", "1.0", "/b"], [100, 10, 5]
        )
        self.assertEqual(prefixes, ["/a", "/b"])
        # 3 : 1 normalizes to 0.75 : 0.25.
        self.assertAlmostEqual(weights[0], 0.75)
        self.assertAlmostEqual(weights[1], 0.25)

    def test_prefix_is_stripped_and_num_samples_formula(self):
        prefixes, weights, num_samples = get_datasets_weights_and_num_samples(
            ["1.0", "  /path/data  "], [1000, 200, 40]
        )
        # Surrounding whitespace on the prefix must be stripped.
        self.assertEqual(prefixes, ["/path/data"])
        self.assertAlmostEqual(weights[0], 1.0)
        # For each requested count: ceil(count * weight * 1.005) + 20.
        expected = [
            math.ceil(1000 * 1.0 * 1.005) + 20,  # 1025
            math.ceil(200 * 1.0 * 1.005) + 20,  # 221
            math.ceil(40 * 1.0 * 1.005) + 20,  # 61
        ]
        self.assertEqual(num_samples, [expected])
        self.assertEqual(num_samples, [[1025, 221, 61]])

    def test_odd_length_prefix_list_raises(self):
        # Prefixes must come in (weight, path) pairs.
        with self.assertRaises(AssertionError):
            get_datasets_weights_and_num_samples(
                ["0.7", "/a", "0.3"], [1000, 100, 50]
            )

    def test_zero_total_weight_raises(self):
        with self.assertRaises(AssertionError):
            get_datasets_weights_and_num_samples(["0.0", "/a"], [1000, 100, 50])


class TestIndexedDatasetBackingStore(unittest.TestCase):
    """Real write -> close -> reopen -> read of the mmap store the causal
    dataset draws samples from.

    GPTDataset.__getitem__ extracts each sample by calling
    ``indexed_dataset.get(doc, offset=..., length=...)`` for the documents a
    sample spans.  This verifies that those primitive reads return the correct
    tokens at the correct token offsets and respect document boundaries, using
    content that is distinguishable per document so cross-document leakage
    would be caught.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="causal_ds_test_")
        # Each document uses a disjoint value range so a wrong pointer/offset
        # would surface as tokens from the wrong document.
        self.docs = [
            [10, 11, 12, 13],  # doc 0: 4 tokens
            [20, 21, 22, 23, 24, 25],  # doc 1: 6 tokens
            [30, 31, 32, 33, 34],  # doc 2: 5 tokens
        ]
        prefix = os.path.join(self.tmpdir, "toy")
        builder = MMapIndexedDatasetBuilder(prefix + ".bin", dtype=np.int32)
        for doc in self.docs:
            builder.add_item(np.array(doc, dtype=np.int32))
            builder.end_document()
        builder.finalize(prefix + ".idx")
        self.prefix = prefix

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_sizes_and_doc_boundaries_persist(self):
        ds = MMapIndexedDataset(self.prefix, skip_warmup=True)
        try:
            self.assertEqual(len(ds), 3)
            self.assertEqual(ds.sizes.tolist(), [4, 6, 5])
            # doc_idx marks sentence boundaries for the 3 single-sentence docs.
            self.assertEqual(ds.doc_idx.tolist(), [0, 1, 2, 3])
        finally:
            del ds

    def test_get_returns_exact_tokens_dtype_and_no_mask(self):
        ds = MMapIndexedDataset(self.prefix, skip_warmup=True)
        try:
            tokens, mask = ds.get(0)
            self.assertEqual(tokens.dtype, np.int32)
            self.assertEqual(tokens.tolist(), [10, 11, 12, 13])
            # No .lsm file was written, so no loss mask is returned.
            self.assertIsNone(mask)
        finally:
            del ds

    def test_get_honours_token_offset_and_length(self):
        ds = MMapIndexedDataset(self.prefix, skip_warmup=True)
        try:
            # Tail of doc 1 starting at token offset 4 -> [24, 25].
            tokens, _ = ds.get(1, offset=4)
            self.assertEqual(tokens.tolist(), [24, 25])
            # Prefix of doc 1: first 3 tokens.
            tokens, _ = ds.get(1, offset=0, length=3)
            self.assertEqual(tokens.tolist(), [20, 21, 22])
            # Explicit length within doc 2.
            tokens, _ = ds.get(2, length=3)
            self.assertEqual(tokens.tolist(), [30, 31, 32])
        finally:
            del ds

    def test_whole_document_reads_stay_within_boundaries(self):
        ds = MMapIndexedDataset(self.prefix, skip_warmup=True)
        try:
            # __getitem__ on an int returns the full document; each must
            # contain only its own value range (no leakage into neighbours).
            np.testing.assert_array_equal(ds[0], [10, 11, 12, 13])
            np.testing.assert_array_equal(ds[1], [20, 21, 22, 23, 24, 25])
            np.testing.assert_array_equal(ds[2], [30, 31, 32, 33, 34])
        finally:
            del ds


if __name__ == "__main__":
    unittest.main()
