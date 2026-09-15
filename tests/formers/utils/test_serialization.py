# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import io
import os
import pickle
import tempfile
import unittest
from unittest import TestCase

import numpy as np
import paddle
from parameterized import parameterized

from paddlefleet.utils import load_torch
from paddlefleet.utils.serialization import (
    SafeUnpickler,
    SerializationError,
    StorageType,
    UnpicklerWrapperStage,
    _element_size,
    _maybe_decode_ascii,
    _rebuild_parameter,
    _rebuild_parameter_with_state,
    _rebuild_tensor_stage,
    _storage_type_to_dtype_to_map,
    _TYPES,
    dumpy,
    read_prefix_key,
    seek_by_string,
)
from tests.formers.testing_utils import require_package


class SerializationTest(TestCase):
    @parameterized.expand(
        [
            "float32",
            "float16",
            "bfloat16",
        ]
    )
    @require_package("torch")
    def test_simple_load(self, dtype: str):
        import torch

        # torch "normal_kernel_cpu" not implemented for 'Char', 'Int', 'Long', so only support float
        dtype_mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,  # test bfloat16
        }
        dtype = dtype_mapping[dtype]

        with tempfile.TemporaryDirectory() as tempdir:
            weight_file_path = os.path.join(tempdir, "pytorch_model.bin")
            torch.save(
                {
                    "a": torch.randn(2, 3, dtype=dtype),
                    "b": torch.randn(3, 4, dtype=dtype),
                    "a_parameter": torch.nn.Parameter(
                        torch.randn(2, 3, dtype=dtype)
                    ),  # test torch.nn.Parameter
                    "b_parameter": torch.nn.Parameter(
                        torch.randn(3, 4, dtype=dtype)
                    ),
                },
                weight_file_path,
            )
            numpy_data = load_torch(weight_file_path)
            torch_data = torch.load(weight_file_path)

            for key, arr in numpy_data.items():
                assert np.allclose(
                    paddle.to_tensor(arr).cast("float32").cpu().numpy(),
                    torch_data[key].detach().cpu().to(torch.float32).numpy(),
                )


class SerializationBehaviorTest(TestCase):
    """Behavior tests for paddlefleet/utils/serialization.py.

    Expected values are hand-derived independently of the functions under
    test; the serialization helpers are never used to build their own
    expectations. File-backed helpers exercise real temporary-file I/O.
    """

    # ------------------------------------------------------------------
    # dtype / element-size metadata
    # ------------------------------------------------------------------
    def test_element_size_independent_expected(self):
        # bytes per element derived from the IEEE / integer widths directly,
        # not from _element_size itself.
        expected = {
            np.float16: 2,
            np.float32: 4,
            np.float64: 8,
            np.int8: 1,
            np.int16: 2,
            np.int32: 4,
            np.int64: 8,
            np.uint8: 1,
        }
        for dtype, nbytes in expected.items():
            self.assertEqual(_element_size(dtype), nbytes)
        # bool is handled by an explicit branch, not iinfo/finfo.
        self.assertEqual(_element_size(np.bool_), 1)

    def test_storage_type_to_dtype_map_exact(self):
        mapping = _storage_type_to_dtype_to_map()
        # Full contract of the torch->numpy storage mapping, incl. the
        # bf16 hack that reuses uint16 as the transport dtype.
        self.assertEqual(mapping["DoubleStorage"], np.double)
        self.assertEqual(mapping["FloatStorage"], np.float32)
        self.assertEqual(mapping["HalfStorage"], np.half)
        self.assertEqual(mapping["LongStorage"], np.int64)
        self.assertEqual(mapping["IntStorage"], np.int32)
        self.assertEqual(mapping["ShortStorage"], np.int16)
        self.assertEqual(mapping["CharStorage"], np.int8)
        self.assertEqual(mapping["ByteStorage"], np.uint8)
        self.assertEqual(mapping["BoolStorage"], np.bool_)
        self.assertEqual(mapping["BFloat16Storage"], np.uint16)
        # lru_cache must hand back the same object, not rebuild it.
        self.assertIs(_storage_type_to_dtype_to_map(), mapping)

    def test_types_table_exact(self):
        # safetensors dtype-string -> numpy dtype used by load_torch.
        self.assertEqual(_TYPES["F64"], np.float64)
        self.assertEqual(_TYPES["F32"], np.float32)
        self.assertEqual(_TYPES["F16"], np.float16)
        self.assertEqual(_TYPES["I64"], np.int64)
        self.assertEqual(_TYPES["I32"], np.int32)
        self.assertEqual(_TYPES["I8"], np.int8)
        self.assertEqual(_TYPES["U8"], np.uint8)
        self.assertEqual(_TYPES["BOOL"], bool)
        # BF16 is transported as uint16 (raw 2-byte payload).
        self.assertEqual(_TYPES["BF16"], np.uint16)

    def test_storage_type_dtype_and_repr(self):
        st = StorageType("FloatStorage")
        self.assertEqual(st.dtype, np.float32)
        self.assertIn("float32", str(st))
        self.assertEqual(StorageType("BFloat16Storage").dtype, np.uint16)
        # Unknown storage names surface as KeyError from the mapping lookup.
        with self.assertRaises(KeyError):
            StorageType("NoSuchStorage")

    # ------------------------------------------------------------------
    # ascii decode helper
    # ------------------------------------------------------------------
    def test_maybe_decode_ascii(self):
        self.assertEqual(_maybe_decode_ascii(b"storage"), "storage")
        # str passes through unchanged (same object).
        s = "already-str"
        self.assertIs(_maybe_decode_ascii(s), s)
        # non-ascii bytes are a hard error, not silently mangled.
        with self.assertRaises(UnicodeDecodeError):
            _maybe_decode_ascii(b"\xff\xfe")

    # ------------------------------------------------------------------
    # tensor / parameter rebuild (pure numpy, no device needed)
    # ------------------------------------------------------------------
    def test_rebuild_tensor_stage_c_order_content(self):
        # stride[0] != 1 -> C / row-major reshape.
        storage = np.arange(6, dtype="float32")
        out = _rebuild_tensor_stage(storage, 0, [2, 3], [3, 1], False, [])
        self.assertEqual(out.shape, (2, 3))
        np.testing.assert_array_equal(
            out, np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype="float32")
        )

    def test_rebuild_tensor_stage_f_order_content(self):
        # stride[0] == 1 and stride[1] > 1 -> Fortran / column-major reshape.
        # Hand-derived column-major fill of [0..5] into (2, 3):
        #   col0=[0,1] col1=[2,3] col2=[4,5]
        storage = np.arange(6, dtype="float32")
        out = _rebuild_tensor_stage(storage, 0, [2, 3], [1, 2], False, [])
        self.assertEqual(out.shape, (2, 3))
        np.testing.assert_array_equal(
            out, np.array([[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]], dtype="float32")
        )

    def test_rebuild_tensor_stage_honours_offset(self):
        # storage_offset must slice into the flat storage before reshape.
        storage = np.arange(10, dtype="float32")
        out = _rebuild_tensor_stage(storage, 2, [2, 3], [3, 1], False, [])
        np.testing.assert_array_equal(
            out, np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype="float32")
        )

    def test_rebuild_parameter_is_identity(self):
        data = np.array([1.5, -2.5, 3.5], dtype="float32")
        out = _rebuild_parameter(data, True, [])
        self.assertIs(out, data)  # no copy / no mutation
        np.testing.assert_array_equal(out, [1.5, -2.5, 3.5])

    def test_rebuild_parameter_with_state_ignores_state(self):
        data = np.array([[7.0, 8.0]], dtype="float32")
        sentinel_state = {"unused": object()}
        out = _rebuild_parameter_with_state(data, False, [], sentinel_state)
        self.assertIs(out, data)
        np.testing.assert_array_equal(out, [[7.0, 8.0]])

    def test_dumpy_swallows_everything(self):
        self.assertIsNone(dumpy())
        self.assertIsNone(dumpy(1, 2, key="v"))

    # ------------------------------------------------------------------
    # UnpicklerWrapperStage dispatch table
    # ------------------------------------------------------------------
    def _make_wrapper(self):
        return UnpicklerWrapperStage(io.BytesIO(pickle.dumps(0)))

    def test_unpickler_dispatch_storage(self):
        w = self._make_wrapper()
        st = w.find_class("torch", "FloatStorage")
        self.assertIsInstance(st, StorageType)
        self.assertEqual(st.dtype, np.float32)

    def test_unpickler_dispatch_rebuild_functions(self):
        w = self._make_wrapper()
        # parameter vs parameter-with-state vs plain tensor must not be swapped.
        self.assertIs(
            w.find_class("torch._utils", "_rebuild_parameter"),
            _rebuild_parameter,
        )
        self.assertIs(
            w.find_class("torch._utils", "_rebuild_parameter_with_state"),
            _rebuild_parameter_with_state,
        )
        # any other torch._utils name falls back to the tensor rebuilder.
        self.assertIs(
            w.find_class("torch._utils", "_rebuild_tensor_v2"),
            _rebuild_tensor_stage,
        )

    def test_unpickler_dispatch_lightning(self):
        w = self._make_wrapper()
        self.assertIs(
            w.find_class("pytorch_lightning.utilities", "anything"), dumpy
        )

    # ------------------------------------------------------------------
    # SafeUnpickler: real pickle round-trip + real blocking
    # ------------------------------------------------------------------
    def test_safe_unpickler_roundtrips_builtins(self):
        data = {
            "a": [1, 2, 3],
            "b": (4, 5),
            "c": "text",
            "d": {"nested": [6, 7]},
            "e": {8, 9},
        }
        blob = pickle.dumps(data)
        restored = SafeUnpickler(io.BytesIO(blob)).load()
        self.assertEqual(restored, data)

    def test_safe_unpickler_blocks_non_builtin(self):
        import collections

        # OrderedDict resolves via find_class("collections", "OrderedDict"),
        # which is outside the builtin allow-list.
        blob = pickle.dumps(collections.OrderedDict([("x", 1)]))
        with self.assertRaises(pickle.UnpicklingError):
            SafeUnpickler(io.BytesIO(blob)).load()
        # object() resolves via builtins.object, also not in the allow-list.
        with self.assertRaises(pickle.UnpicklingError):
            SafeUnpickler(io.BytesIO(pickle.dumps(object()))).load()

    # ------------------------------------------------------------------
    # seek_by_string / read_prefix_key: real temp-file I/O
    # ------------------------------------------------------------------
    def test_seek_by_string_returns_end_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "blob.bin")
            with open(path, "wb") as f:
                f.write(b"xxabcyy")
            with open(path, "rb") as fh:
                # end index sits just past the matched "abc" -> position 5.
                self.assertEqual(seek_by_string(fh, "abc", 7), 5)

    def test_seek_by_string_missing_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "blob.bin")
            with open(path, "wb") as f:
                f.write(b"xxxxxx")
            with open(path, "rb") as fh:
                with self.assertRaises(SerializationError):
                    seek_by_string(fh, "abc", 6)

    def test_read_prefix_key_extracts_archive_prefix(self):
        # Mimic the torch-zip layout read_prefix_key parses: a 30-byte local
        # header, then "<prefix>/data.pkl". read_prefix_key seeks to byte 30
        # and reads (end_of_"data.pkl" - 30 - len("/data.pkl")) bytes.
        header = b"\x00" * 30  # MZ_ZIP_LOCAL_DIR_HEADER_SIZE
        body = b"myprefix/data.pkl"
        trailer = b"\x00" * 10
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "archive.pt")
            with open(path, "wb") as f:
                f.write(header + body + trailer)
            self.assertEqual(read_prefix_key(path), "myprefix")

    @unittest.expectedFailure
    def test_seek_by_string_overlapping_prefix_bug(self):
        # BUG (src/paddlefleet/utils/serialization.py:78-79): on a mismatch the
        # search resets word_index to 0 WITHOUT re-examining the current byte,
        # so overlapping prefixes are missed. "aab" is present in "xaaab"
        # (indices 2..4, correct end index = 5) yet seek_by_string raises
        # SerializationError instead of returning 5. Asserting the CORRECT
        # behavior; marked expectedFailure until the matcher is fixed.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "blob.bin")
            with open(path, "wb") as f:
                f.write(b"xaaab")
            with open(path, "rb") as fh:
                self.assertEqual(seek_by_string(fh, "aab", 5), 5)

    # ------------------------------------------------------------------
    # safetensors -> load_torch round-trip into a fresh state dict
    # ------------------------------------------------------------------
    @require_package("safetensors")
    def test_load_torch_safetensors_roundtrip(self):
        from safetensors.numpy import save_file

        # Distinct params sharing shape/dtype but with different content, so a
        # key swap cannot pass a content check. Plus mixed dtypes to exercise
        # the _TYPES mapping (F32 / F16 / I64).
        w1 = np.arange(6, dtype="float32").reshape(2, 3)
        w2 = np.arange(6, dtype="float32").reshape(2, 3) + 100.0
        half = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float16")
        ids = np.array([10, 20, 30], dtype="int64")
        source = {"w1": w1, "w2": w2, "half": half, "ids": ids}

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.safetensors")
            save_file(source, path)

            loaded = load_torch(path)  # returns paddle tensors

            self.assertEqual(set(loaded), {"w1", "w2", "half", "ids"})
            for key in source:
                got = np.asarray(loaded[key].cpu().numpy())
                self.assertEqual(
                    got.dtype, source[key].dtype, f"dtype mismatch for {key}"
                )
                np.testing.assert_array_equal(
                    got, source[key], err_msg=f"content mismatch for {key}"
                )

    @require_package("safetensors")
    def test_load_torch_rejects_corrupt_safetensors(self):
        # A file named like a safetensors checkpoint but holding garbage bytes
        # must raise during deserialization, not be swallowed into {} .
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.safetensors")
            with open(path, "wb") as f:
                f.write(b"this is not a valid safetensors payload")
            with self.assertRaises(Exception):
                load_torch(path)

    def test_load_torch_unrecognized_suffix_returns_empty(self):
        # load_torch only recognizes pytorch_model.bin / model.safetensors
        # style names; anything else yields an empty state dict (documented
        # fall-through, not an error).
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "weights.unknown")
            with open(path, "wb") as f:
                f.write(b"\x00\x01\x02")
            self.assertEqual(load_torch(path), {})


if __name__ == "__main__":
    unittest.main()
