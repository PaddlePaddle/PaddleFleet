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

"""Behavior tests for the RLHF data-exchange protocol.

These tests exercise the real public entries in
``paddlefleet.datasets.rlhf_datasets.protocol`` and check the actual
content correspondence guaranteed by the protocol: which container a value
is routed to, whether merged dicts keep the right values, whether the
consistency guard rejects malformed non-tensor batches, and whether
``repeat`` keeps every label aligned with its own sample under both the
interleave and the tile ordering. Expected values are derived by hand,
never by calling the function under test.
"""

import unittest

import numpy as np
import paddle

from paddlefleet.datasets.rlhf_datasets.protocol import (
    DataProto,
    TensorDict,
    list_of_dict_to_dict_of_list,
    union_numpy_dict,
    union_tensor_dict,
    union_two_dict,
)


class TestListOfDictToDictOfList(unittest.TestCase):
    """list_of_dict_to_dict_of_list transposes records preserving order."""

    def test_transpose_keeps_per_key_order_and_content(self):
        records = [
            {"prompt": "p0", "reward": 10},
            {"prompt": "p1", "reward": 20},
            {"prompt": "p2", "reward": 30},
        ]
        out = list_of_dict_to_dict_of_list(records)
        # Every key collects its column in the original record order; a
        # transpose bug that reorders or drops a row would change these lists.
        self.assertEqual(
            out, {"prompt": ["p0", "p1", "p2"], "reward": [10, 20, 30]}
        )

    def test_empty_input_returns_empty_dict(self):
        self.assertEqual(list_of_dict_to_dict_of_list([]), {})


class TestUnionTwoDict(unittest.TestCase):
    """union_two_dict merges meta dicts and rejects conflicting values."""

    def test_merge_adds_new_keys_and_returns_first(self):
        d1 = {"lr": 0.1}
        d2 = {"steps": 5, "name": "run"}
        result = union_two_dict(d1, d2)
        self.assertIs(result, d1)  # merges into and returns the first dict
        self.assertEqual(result, {"lr": 0.1, "steps": 5, "name": "run"})

    def test_conflicting_equal_value_is_allowed(self):
        d1 = {"seed": 42, "lr": 0.1}
        d2 = {"seed": 42, "epochs": 3}
        result = union_two_dict(d1, d2)
        self.assertEqual(result, {"seed": 42, "lr": 0.1, "epochs": 3})

    def test_conflicting_different_value_raises(self):
        d1 = {"seed": 42}
        d2 = {"seed": 7}
        with self.assertRaises(AssertionError):
            union_two_dict(d1, d2)


class TestUnionNumpyDict(unittest.TestCase):
    """union_numpy_dict merges arrays and detects value conflicts."""

    def test_merge_adds_new_key_with_content(self):
        d1 = {"labels": np.array(["a", "b"], dtype=object)}
        d2 = {"scores": np.array([1.5, 2.5])}
        result = union_numpy_dict(d1, d2)
        self.assertIs(result, d1)
        self.assertEqual(set(result), {"labels", "scores"})
        self.assertEqual(result["labels"].tolist(), ["a", "b"])
        np.testing.assert_array_equal(result["scores"], [1.5, 2.5])

    def test_conflicting_equal_array_is_allowed(self):
        d1 = {"labels": np.array(["x", "y"], dtype=object)}
        d2 = {"labels": np.array(["x", "y"], dtype=object)}
        result = union_numpy_dict(d1, d2)
        self.assertEqual(result["labels"].tolist(), ["x", "y"])

    def test_conflicting_different_array_raises(self):
        d1 = {"labels": np.array([1, 2])}
        d2 = {"labels": np.array([1, 9])}
        with self.assertRaises(AssertionError):
            union_numpy_dict(d1, d2)


class TestTensorDict(unittest.TestCase):
    """TensorDict stores tensors and guards the batch dimension."""

    def test_set_get_preserves_content_and_keys(self):
        td = TensorDict(source={}, batch_size=None)
        tensor = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        td["obs"] = tensor
        np.testing.assert_array_equal(
            td["obs"].numpy(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        )
        self.assertIn("obs", td.keys())

    def test_source_tensors_are_registered_with_values(self):
        a = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        b = paddle.to_tensor([[9.0], [8.0]])
        td = TensorDict(source={"a": a, "b": b}, batch_size=a.shape[:1])
        self.assertEqual(set(td.keys()), {"a", "b"})
        np.testing.assert_array_equal(td["a"].numpy(), [[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(td["b"].numpy(), [[9.0], [8.0]])

    def test_batch_dim_mismatch_is_rejected(self):
        tensor = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        # tensor has leading dim 2; declaring batch size 5 must fail the guard.
        with self.assertRaises(AssertionError):
            TensorDict(source={"a": tensor}, batch_size=[5])


class TestUnionTensorDict(unittest.TestCase):
    """union_tensor_dict merges disjoint keys and checks batch size."""

    def test_disjoint_keys_merge_with_content(self):
        a = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        b = paddle.to_tensor([[10.0, 20.0], [30.0, 40.0]])
        td1 = TensorDict(source={"a": a}, batch_size=a.shape[:1])
        td2 = TensorDict(source={"b": b}, batch_size=b.shape[:1])
        result = union_tensor_dict(td1, td2)
        self.assertEqual(set(result.keys()), {"a", "b"})
        np.testing.assert_array_equal(
            result["a"].numpy(), [[1.0, 2.0], [3.0, 4.0]]
        )
        np.testing.assert_array_equal(
            result["b"].numpy(), [[10.0, 20.0], [30.0, 40.0]]
        )

    def test_batch_size_mismatch_raises(self):
        a = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])  # leading dim 2
        b = paddle.to_tensor([[1.0], [2.0], [3.0]])  # leading dim 3
        td1 = TensorDict(source={"a": a}, batch_size=a.shape[:1])
        td2 = TensorDict(source={"b": b}, batch_size=b.shape[:1])
        with self.assertRaises(AssertionError):
            union_tensor_dict(td1, td2)


class TestDataProtoFromDict(unittest.TestCase):
    """DataProto factory routing and batch-size validation."""

    def test_from_single_dict_routes_by_value_type(self):
        data = {
            "input_ids": paddle.to_tensor([[5, 6], [7, 8]]),
            "labels": np.array(["yes", "no"], dtype=object),
        }
        dp = DataProto.from_single_dict(data)
        # Tensors belong to the tensor batch, ndarrays to the non-tensor batch.
        self.assertIn("input_ids", dp.batch.keys())
        self.assertNotIn("input_ids", dp.non_tensor_batch)
        self.assertIn("labels", dp.non_tensor_batch)
        self.assertNotIn("labels", dp.batch.keys())
        np.testing.assert_array_equal(
            dp.batch["input_ids"].numpy(), [[5, 6], [7, 8]]
        )
        self.assertEqual(dp.non_tensor_batch["labels"].tolist(), ["yes", "no"])

    def test_from_single_dict_rejects_unsupported_type(self):
        with self.assertRaises(ValueError):
            DataProto.from_single_dict({"bad": "not-a-tensor-or-array"})

    def test_from_dict_rejects_inconsistent_batch_dims(self):
        with self.assertRaises(AssertionError):
            DataProto.from_dict(
                tensors={
                    "x": paddle.to_tensor([[1.0], [2.0]]),  # leading dim 2
                    "y": paddle.to_tensor([[1.0], [2.0], [3.0]]),  # dim 3
                },
                non_tensors={},
                meta_info={},
            )


class TestDataProtoConsistency(unittest.TestCase):
    """__post_init__ / check_consistency rejects malformed non-tensors."""

    def _batch(self, rows):
        tensor = paddle.to_tensor(rows)
        return TensorDict(
            source={"input_ids": tensor}, batch_size=tensor.shape[:1]
        )

    def test_non_tensor_length_must_match_batch_size(self):
        batch = self._batch([[1, 2], [3, 4]])  # batch size 2
        bad = {"labels": np.array(["a", "b", "c"], dtype=object)}  # length 3
        with self.assertRaises(AssertionError):
            DataProto(batch=batch, non_tensor_batch=bad)

    def test_non_tensor_must_be_object_dtype(self):
        batch = self._batch([[1, 2], [3, 4]])  # batch size 2
        # Correct length but a non-object dtype violates the protocol.
        bad = {"labels": np.array([1, 2], dtype=np.int64)}
        with self.assertRaises(AssertionError):
            DataProto(batch=batch, non_tensor_batch=bad)


class TestDataProtoUnion(unittest.TestCase):
    """DataProto.union merges tensor batch and meta_info by key."""

    def test_union_merges_batch_and_meta_content(self):
        dp1 = DataProto.from_dict(
            tensors={"x": paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])},
            non_tensors={},
            meta_info={"m1": 1},
        )
        dp2 = DataProto.from_dict(
            tensors={"y": paddle.to_tensor([[5.0, 6.0], [7.0, 8.0]])},
            non_tensors={},
            meta_info={"m2": 2},
        )
        dp1.union(dp2)
        self.assertEqual(set(dp1.batch.keys()), {"x", "y"})
        np.testing.assert_array_equal(
            dp1.batch["x"].numpy(), [[1.0, 2.0], [3.0, 4.0]]
        )
        np.testing.assert_array_equal(
            dp1.batch["y"].numpy(), [[5.0, 6.0], [7.0, 8.0]]
        )
        self.assertEqual(dp1.meta_info, {"m1": 1, "m2": 2})


class TestDataProtoRepeat(unittest.TestCase):
    """repeat must keep every label aligned with its own sample.

    interleave=True repeats each sample consecutively; interleave=False tiles
    the whole batch. The two modes yield different orderings, so checking the
    full sequence (not just the length) distinguishes them and catches a
    label that fails to track its sample.
    """

    def _make(self):
        return DataProto.from_dict(
            tensors={"input_ids": paddle.to_tensor([[10, 11], [20, 21]])},
            non_tensors={"labels": np.array(["s0", "s1"], dtype=object)},
            meta_info={},
        )

    def test_interleave_repeats_each_sample_consecutively(self):
        result = self._make().repeat(repeat_times=3, interleave=True)
        self.assertEqual(len(result), 6)
        np.testing.assert_array_equal(
            result.batch["input_ids"].numpy(),
            [[10, 11], [10, 11], [10, 11], [20, 21], [20, 21], [20, 21]],
        )
        self.assertEqual(
            result.non_tensor_batch["labels"].tolist(),
            ["s0", "s0", "s0", "s1", "s1", "s1"],
        )

    def test_non_interleave_tiles_the_whole_batch(self):
        result = self._make().repeat(repeat_times=2, interleave=False)
        self.assertEqual(len(result), 4)
        np.testing.assert_array_equal(
            result.batch["input_ids"].numpy(),
            [[10, 11], [20, 21], [10, 11], [20, 21]],
        )
        self.assertEqual(
            result.non_tensor_batch["labels"].tolist(),
            ["s0", "s1", "s0", "s1"],
        )


if __name__ == "__main__":
    unittest.main()
