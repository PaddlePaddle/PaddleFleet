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

import os
import shutil
import tempfile
import unittest

import numpy as np

from paddlefleet.data.indexed_dataset import (
    IndexedDataset,
    IndexedDatasetBuilder,
    MMapIndexedDataset,
    MMapIndexedDatasetBuilder,
    code,
    create_doc_idx,
    data_file_path,
    dtypes,
    index_file_path,
    loss_mask_file_path,
    make_builder,
    make_dataset,
    read_longs,
    read_shorts,
    write_longs,
    write_shorts,
)


class _TempDirTestCase(unittest.TestCase):
    """Base class providing an isolated temporary directory per test."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="idx_ds_test_")

    def tearDown(self):
        # ignore_errors so lingering mmap handles do not break cleanup.
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def prefix(self, name="data"):
        return os.path.join(self.tmpdir, name)


class TestReadWriteRoundtrip(_TempDirTestCase):
    """write_longs/read_longs and write_shorts/read_shorts do real file I/O."""

    def test_write_then_read_longs_preserves_values_and_dtype(self):
        path = self.prefix("longs.bin")
        data = np.array([7, 0, -3, 2**40, 15], dtype=np.int64)
        with open(path, "wb") as f:
            write_longs(f, data)
        with open(path, "rb") as f:
            out = read_longs(f, len(data))
        self.assertEqual(out.dtype, np.int64)
        np.testing.assert_array_equal(out, data)

    def test_write_then_read_shorts_preserves_values_and_dtype(self):
        path = self.prefix("shorts.bin")
        data = np.array([10, 20, 30, 40], dtype=np.int32)
        with open(path, "wb") as f:
            write_shorts(f, data)
        with open(path, "rb") as f:
            out = read_shorts(f, len(data))
        self.assertEqual(out.dtype, np.int32)
        np.testing.assert_array_equal(out, data)


class TestCodeAndDtypes(unittest.TestCase):
    """code() must be the exact inverse of the dtypes lookup table."""

    def test_code_matches_every_dtypes_entry(self):
        for expected_code, dtype in dtypes.items():
            self.assertEqual(code(dtype), expected_code)

    def test_code_roundtrips_through_dtypes(self):
        self.assertEqual(dtypes[code(np.int32)], np.int32)
        self.assertEqual(dtypes[code(np.float32)], np.float32)
        self.assertEqual(dtypes[code(np.uint16)], np.uint16)

    def test_code_rejects_unregistered_dtype(self):
        with self.assertRaises(ValueError):
            code(np.complex128)


class TestCreateDocIdx(unittest.TestCase):
    """A zero size marks a document boundary; output is the start sentence
    index of each document (with a leading 0)."""

    def test_zero_size_entries_mark_document_boundaries(self):
        # sizes index:      0  1  2  3  4  5
        sizes = [4, 3, 0, 5, 0, 2]
        # zeros at index 2 and 4 -> new documents start at sentence 3 and 5.
        self.assertEqual(create_doc_idx(sizes), [0, 3, 5])

    def test_no_boundaries_returns_single_document(self):
        self.assertEqual(create_doc_idx([5, 6, 7]), [0])


class TestIndexedDatasetRoundtrip(_TempDirTestCase):
    """Lazy IndexedDataset: build to disk, reopen, read back real content."""

    def _build(self, prefix, dtype=np.int32):
        builder = IndexedDatasetBuilder(data_file_path(prefix), dtype=dtype)
        builder.add_item(np.array([11, 12, 13], dtype=dtype))
        builder.add_item(np.array([21, 22], dtype=dtype))
        builder.end_document()
        builder.add_item(np.array([31, 32, 33, 34], dtype=dtype))
        builder.end_document()
        builder.finalize(index_file_path(prefix))

    def test_roundtrip_content_sizes_docidx_and_get(self):
        prefix = self.prefix("lazy")
        self._build(prefix)
        ds = IndexedDataset(prefix)
        try:
            self.assertEqual(len(ds), 3)
            self.assertEqual(ds.dtype, np.int32)
            np.testing.assert_array_equal(np.asarray(ds.sizes), [3, 2, 4])
            # doc boundaries: doc0 = sentences {0,1}, doc1 = sentence {2}.
            np.testing.assert_array_equal(np.asarray(ds.doc_idx), [0, 2, 3])
            np.testing.assert_array_equal(ds[0], [11, 12, 13])
            np.testing.assert_array_equal(ds[1], [21, 22])
            np.testing.assert_array_equal(ds[2], [31, 32, 33, 34])
            self.assertEqual(ds[0].dtype, np.int32)
            # get(): whole sentence and an interior offset/length window.
            np.testing.assert_array_equal(ds.get(1), [21, 22])
            np.testing.assert_array_equal(
                ds.get(2, offset=1, length=2), [32, 33]
            )
            self.assertEqual(ds.num_tokens(2), 4)
            self.assertEqual(ds.size(0), 3)
        finally:
            del ds


class TestMMapIndexedDatasetRoundtrip(_TempDirTestCase):
    """MMapIndexedDataset: build to disk, reopen via mmap, read back content."""

    def _build(self, prefix, dtype=np.int32):
        builder = MMapIndexedDatasetBuilder(data_file_path(prefix), dtype=dtype)
        builder.add_item(np.array([11, 12, 13], dtype=dtype))
        builder.add_item(np.array([21, 22], dtype=dtype))
        builder.end_document()
        builder.add_item(np.array([31, 32, 33, 34], dtype=dtype))
        builder.end_document()
        builder.finalize(index_file_path(prefix))

    def test_roundtrip_content_sizes_docidx_slice_and_get(self):
        prefix = self.prefix("mmap")
        self._build(prefix)
        ds = MMapIndexedDataset(prefix, skip_warmup=True)
        try:
            self.assertEqual(len(ds), 3)
            self.assertEqual(ds._index.dtype, np.int32)
            np.testing.assert_array_equal(np.asarray(ds.sizes), [3, 2, 4])
            np.testing.assert_array_equal(np.asarray(ds.doc_idx), [0, 2, 3])
            np.testing.assert_array_equal(ds[0], [11, 12, 13])
            np.testing.assert_array_equal(ds[2], [31, 32, 33, 34])
            self.assertEqual(ds[0].dtype, np.int32)
            # A contiguous slice returns per-sentence arrays, not a flat concat.
            sents = ds[0:2]
            self.assertEqual(len(sents), 2)
            np.testing.assert_array_equal(sents[0], [11, 12, 13])
            np.testing.assert_array_equal(sents[1], [21, 22])
            # get() returns (tokens, loss_mask); no .lsm file -> mask is None.
            tokens, mask = ds.get(1)
            np.testing.assert_array_equal(tokens, [21, 22])
            self.assertIsNone(mask)
            tokens2, mask2 = ds.get(2, offset=1, length=2)
            np.testing.assert_array_equal(tokens2, [32, 33])
            self.assertIsNone(mask2)
        finally:
            del ds


class TestMMapIndexedDatasetLossMask(_TempDirTestCase):
    """get() must derive the byte offset (into the token buffer) and the token
    offset (into the uint8 loss-mask buffer) separately."""

    def test_get_returns_token_aligned_loss_mask(self):
        prefix = self.prefix("mmap_lsm")
        dtype = np.int32
        builder = MMapIndexedDatasetBuilder(data_file_path(prefix), dtype=dtype)
        builder.add_item(np.array([11, 12, 13], dtype=dtype))
        builder.add_item(np.array([21, 22], dtype=dtype))
        builder.end_document()
        builder.add_item(np.array([31, 32, 33, 34], dtype=dtype))
        builder.end_document()
        builder.finalize(index_file_path(prefix))

        # One uint8 mask value per token, flat in token order.
        # flat tokens: [11,12,13, 21,22, 31,32,33,34]
        loss_mask = np.array([1, 0, 1, 0, 1, 1, 1, 0, 0], dtype=np.uint8)
        with open(loss_mask_file_path(prefix), "wb") as f:
            f.write(loss_mask.tobytes(order="C"))

        ds = MMapIndexedDataset(prefix, skip_warmup=True)
        try:
            # sentence 0: token bytes 0, token index 0 -> mask [1,0,1]
            tokens0, mask0 = ds.get(0)
            np.testing.assert_array_equal(tokens0, [11, 12, 13])
            self.assertEqual(mask0.dtype, np.uint8)
            np.testing.assert_array_equal(mask0, [1, 0, 1])
            # sentence 1: byte offset 12 but token offset 3 -> mask [0,1]
            tokens1, mask1 = ds.get(1)
            np.testing.assert_array_equal(tokens1, [21, 22])
            np.testing.assert_array_equal(mask1, [0, 1])
            # interior window of sentence 2: byte offset 24, token offset 6
            tokens2, mask2 = ds.get(2, offset=1, length=2)
            np.testing.assert_array_equal(tokens2, [32, 33])
            np.testing.assert_array_equal(mask2, [1, 0])
        finally:
            del ds


class TestMakeBuilderAndMakeDataset(_TempDirTestCase):
    """make_builder / make_dataset select the right implementation and the
    written content survives the full factory roundtrip."""

    def test_mmap_factory_roundtrip(self):
        prefix = self.prefix("factory")
        dtype = np.int32
        builder = make_builder(data_file_path(prefix), "mmap", save_dtype=dtype)
        self.assertIsInstance(builder, MMapIndexedDatasetBuilder)
        builder.add_item(np.array([101, 102], dtype=dtype))
        builder.end_document()
        builder.add_item(np.array([201, 202, 203], dtype=dtype))
        builder.end_document()
        builder.finalize(index_file_path(prefix))

        ds = make_dataset(prefix, "mmap", True)
        try:
            self.assertIsInstance(ds, MMapIndexedDataset)
            self.assertEqual(len(ds), 2)
            np.testing.assert_array_equal(ds[0], [101, 102])
            np.testing.assert_array_equal(ds[1], [201, 202, 203])
            np.testing.assert_array_equal(np.asarray(ds.doc_idx), [0, 1, 2])
        finally:
            del ds

    def test_lazy_factory_roundtrip(self):
        prefix = self.prefix("factory_lazy")
        dtype = np.int32
        builder = make_builder(data_file_path(prefix), "lazy", save_dtype=dtype)
        self.assertIsInstance(builder, IndexedDatasetBuilder)
        builder.add_item(np.array([5, 6, 7], dtype=dtype))
        builder.end_document()
        builder.finalize(index_file_path(prefix))

        ds = make_dataset(prefix, "lazy")
        try:
            self.assertIsInstance(ds, IndexedDataset)
            self.assertEqual(len(ds), 1)
            np.testing.assert_array_equal(ds[0], [5, 6, 7])
        finally:
            del ds

    def test_make_dataset_returns_none_for_missing_path(self):
        self.assertIsNone(make_dataset(self.prefix("missing"), "mmap", True))


if __name__ == "__main__":
    unittest.main()
