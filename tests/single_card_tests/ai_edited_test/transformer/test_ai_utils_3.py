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
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import paddle

from paddlefleet.transformer.utils import (
    get_doc_lens,
    get_doc_starts,
    inspect_and_load_tensor,
)


class TestInspectAndLoadTensor(unittest.TestCase):
    """Tests for inspect_and_load_tensor."""

    def _make_tensor(self, shape=(2, 4), dtype="float32"):
        arr = np.random.randn(*shape).astype(np.float32)
        return paddle.to_tensor(arr)

    def test_returns_original_tensor_no_env(self):
        t = self._make_tensor()
        result = inspect_and_load_tensor("q", 0, t, load=True)
        self.assertIs(result, t)

    def test_returns_original_when_load_false(self):
        arr = np.random.randn(2, 4).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            np.save(os.path.join(tmpdir, "q.npy"), arr)
            with patch.dict(os.environ, {"ABLATION_LOAD_TENSOR_PATH": tmpdir}):
                t = paddle.zeros([2, 4])
                result = inspect_and_load_tensor("q", 0, t, load=False)
        self.assertIs(result, t)

    @patch.dict(os.environ, {"ABLATION_INSPECT_TENSOR": "1"})
    def test_dump_prints_info(self):
        t = self._make_tensor()
        with patch("builtins.print") as mock_print:
            result = inspect_and_load_tensor("attn_out", 2, t, load=False)
        self.assertIs(result, t)
        output = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("[ABLATION_train]", output)
        self.assertIn("tag=attn_out", output)
        self.assertIn("layer=2", output)

    @patch.dict(
        os.environ,
        {"ABLATION_INSPECT_TENSOR": "1", "ABLATION_INFO_SKIP_TAGS": "q,k"},
    )
    def test_info_skip_tags(self):
        t = self._make_tensor()
        with patch("builtins.print") as mock_print:
            inspect_and_load_tensor("q", 0, t, load=False)
        output = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertNotIn("[ABLATION_train]", output)

    @patch.dict(os.environ, {"ABLATION_INSPECT_TENSOR": "1"})
    def test_none_tensor_no_crash(self):
        with patch("builtins.print") as mock_print:
            result = inspect_and_load_tensor("q", 0, None, load=False)
        self.assertIsNone(result)
        mock_print.assert_not_called()

    def test_load_tensor_from_npy(self):
        arr = np.random.randn(2, 4).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            np.save(os.path.join(tmpdir, "q.npy"), arr)
            with patch.dict(os.environ, {"ABLATION_LOAD_TENSOR_PATH": tmpdir}):
                t = paddle.zeros([2, 4])
                result = inspect_and_load_tensor("q", 0, t, load=True)
        np.testing.assert_allclose(result.numpy(), arr, rtol=1e-5)

    def test_load_skipped_for_nonzero_layer(self):
        arr = np.random.randn(2, 4).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            np.save(os.path.join(tmpdir, "q.npy"), arr)
            with patch.dict(os.environ, {"ABLATION_LOAD_TENSOR_PATH": tmpdir}):
                t = paddle.zeros([2, 4])
                result = inspect_and_load_tensor("q", 1, t, load=True)
        self.assertIs(result, t)

    def test_dump_skip_tags_suppresses_load(self):
        arr = np.random.randn(2, 4).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            np.save(os.path.join(tmpdir, "q.npy"), arr)
            with patch.dict(
                os.environ,
                {
                    "ABLATION_LOAD_TENSOR_PATH": tmpdir,
                    "ABLATION_DUMP_SKIP_TAGS": "q",
                },
            ):
                t = paddle.zeros([2, 4])
                result = inspect_and_load_tensor("q", 0, t, load=True)
        self.assertIs(result, t)

    def test_load_auto_reshape(self):
        arr = np.arange(8, dtype=np.float32)  # shape (8,)
        with tempfile.TemporaryDirectory() as tmpdir:
            np.save(os.path.join(tmpdir, "q.npy"), arr)
            with patch.dict(os.environ, {"ABLATION_LOAD_TENSOR_PATH": tmpdir}):
                t = paddle.zeros([2, 4])
                result = inspect_and_load_tensor("q", 0, t, load=True)
        self.assertEqual(list(result.shape), [2, 4])
        np.testing.assert_allclose(result.numpy(), arr.reshape(2, 4), rtol=1e-5)

    def test_load_prints_diff_info(self):
        arr = np.ones((2, 4), dtype=np.float32) * 3.0
        with tempfile.TemporaryDirectory() as tmpdir:
            np.save(os.path.join(tmpdir, "v.npy"), arr)
            with patch.dict(os.environ, {"ABLATION_LOAD_TENSOR_PATH": tmpdir}):
                t = paddle.zeros([2, 4])
                with patch("builtins.print") as mock_print:
                    inspect_and_load_tensor("v", 0, t, load=True)
        output = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("[ABLATION_load_tensor]", output)
        self.assertIn("max_abs_diff", output)


class TestGetDocLens(unittest.TestCase):
    """Tests for get_doc_lens."""

    def test_single_doc(self):
        # single doc of length 4: all positions point to end=4
        indices = paddle.to_tensor([4, 4, 4, 4], dtype="int32").reshape(
            [1, 1, 4, 1]
        )
        doc_lens = get_doc_lens(indices)
        self.assertEqual(doc_lens.numpy().tolist(), [4])

    def test_two_docs(self):
        # doc1 length 2 (ends at 2), doc2 length 2 (ends at 4)
        indices = paddle.to_tensor([2, 2, 4, 4], dtype="int32").reshape(
            [1, 1, 4, 1]
        )
        doc_lens = get_doc_lens(indices)
        self.assertEqual(doc_lens.numpy().tolist(), [2, 2])

    def test_three_docs(self):
        # doc1=1, doc2=2, doc3=3
        indices = paddle.to_tensor([1, 3, 3, 6, 6, 6], dtype="int32").reshape(
            [1, 1, 6, 1]
        )
        doc_lens = get_doc_lens(indices)
        self.assertEqual(doc_lens.numpy().tolist(), [1, 2, 3])


class TestGetDocStarts(unittest.TestCase):
    """Tests for get_doc_starts."""

    def test_single_doc(self):
        doc_lens = paddle.to_tensor([5], dtype="int32")
        starts = get_doc_starts(doc_lens)
        self.assertEqual(starts.numpy().tolist(), [0])

    def test_multiple_docs(self):
        doc_lens = paddle.to_tensor([2, 3, 4], dtype="int32")
        starts = get_doc_starts(doc_lens)
        self.assertEqual(starts.numpy().tolist(), [0, 2, 5])

    def test_single_token_docs(self):
        doc_lens = paddle.to_tensor([1, 1, 1], dtype="int32")
        starts = get_doc_starts(doc_lens)
        self.assertEqual(starts.numpy().tolist(), [0, 1, 2])
