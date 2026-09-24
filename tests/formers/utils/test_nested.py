# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

"""Behavior tests for paddlefleet.utils.nested structure helpers.

These exercise the pure-Python structure-mapping / flattening paths on real
API entries. Expected values are hand-derived literals; the functions under
test are never used to construct expectations. No tensors are required, so the
logic here is verifiable on CPU (无卡); the import of the module still requires
Paddle to be installed.
"""

import unittest

from paddlefleet.utils.nested import (
    flatten_list,
    nested_copy,
    nested_copy_place,
    nested_reduce_tensor,
)


class FlattenListTest(unittest.TestCase):
    def test_flattens_nested_lists_preserving_order(self):
        # Mixed depth; leaves are distinguishable so order errors show up.
        result = flatten_list([[3, 1], [2, [4, 5]], 6])
        self.assertEqual(result, [3, 1, 2, 4, 5, 6])

    def test_deeply_nested_lists(self):
        result = flatten_list([[[[7]]], [8, [9]], 10])
        self.assertEqual(result, [7, 8, 9, 10])

    def test_tuples_are_leaves_not_recursed(self):
        # Only ``list`` is recursed; tuples must pass through as single leaves.
        result = flatten_list([1, (2, 3), [4, (5, 6)]])
        self.assertEqual(result, [1, (2, 3), 4, (5, 6)])
        # The tuple leaves must remain tuples, not be exploded into ints.
        self.assertIsInstance(result[1], tuple)
        self.assertEqual(result[1], (2, 3))
        self.assertIsInstance(result[3], tuple)
        self.assertEqual(result[3], (5, 6))

    def test_strings_are_leaves(self):
        # Strings are iterable but are not lists, so they stay whole.
        result = flatten_list(["ab", ["cd", "ef"]])
        self.assertEqual(result, ["ab", "cd", "ef"])

    def test_empty_and_flat_inputs(self):
        self.assertEqual(flatten_list([]), [])
        self.assertEqual(flatten_list([1, 2, 3]), [1, 2, 3])
        # Empty nested lists contribute nothing.
        self.assertEqual(flatten_list([[], [1], [[]]]), [1])

    def test_returns_new_list(self):
        source = [1, 2, 3]
        result = flatten_list(source)
        self.assertEqual(result, [1, 2, 3])
        self.assertIsNot(result, source)


class NestedCopyTest(unittest.TestCase):
    def test_copies_dict_tree_but_shares_list_leaves(self):
        inner_dict = {"c": 3}
        shared_list = [1, 2, 3]
        source = {"a": 1, "b": inner_dict, "lst": shared_list}

        result = nested_copy(source)

        # Value equality across the whole structure.
        self.assertEqual(result, {"a": 1, "b": {"c": 3}, "lst": [1, 2, 3]})
        # Key order is preserved.
        self.assertEqual(list(result.keys()), ["a", "b", "lst"])
        # Top dict and nested dict are fresh objects.
        self.assertIsNot(result, source)
        self.assertIsNot(result["b"], inner_dict)
        # Lists are NOT recursed: the list leaf is returned by reference.
        self.assertIs(result["lst"], shared_list)
        # Scalars pass through by value.
        self.assertEqual(result["a"], 1)

    def test_dict_inside_list_is_not_copied(self):
        # nested_copy only recurses through dict boundaries, never lists.
        inner = {"x": 1}
        source = {"a": [inner]}

        result = nested_copy(source)

        self.assertIsNot(result, source)
        self.assertIs(result["a"], source["a"])
        self.assertIs(result["a"][0], inner)

    def test_non_dict_returned_by_reference(self):
        lst = [1, 2, 3]
        self.assertIs(nested_copy(lst), lst)
        self.assertEqual(nested_copy(42), 42)
        self.assertEqual(nested_copy("hello"), "hello")
        self.assertIsNone(nested_copy(None))

    def test_mutating_copy_does_not_affect_source(self):
        source = {"a": 1, "b": {"c": 2}}
        result = nested_copy(source)
        result["b"]["c"] = 999
        result["a"] = 0
        # Original nested dict is untouched because it was copied.
        self.assertEqual(source, {"a": 1, "b": {"c": 2}})


class NestedReduceTensorNonTensorTest(unittest.TestCase):
    """nested_reduce_tensor on structures with only non-tensor leaves.

    With no paddle.Tensor present, the function should rebuild container
    structure (preserving list/tuple type and dict key order) while passing
    scalar leaves through unchanged.
    """

    def test_scalar_leaves_pass_through(self):
        self.assertEqual(nested_reduce_tensor(42), 42)
        self.assertEqual(nested_reduce_tensor("x"), "x")
        self.assertIsNone(nested_reduce_tensor(None))

    def test_dict_is_copied_with_key_order_preserved(self):
        source = {"b": 1, "a": 2, "n": {"m": 5}}
        result = nested_reduce_tensor(source)

        self.assertEqual(result, {"b": 1, "a": 2, "n": {"m": 5}})
        # Insertion order is preserved through the shallow copy.
        self.assertEqual(list(result.keys()), ["b", "a", "n"])
        # A fresh top dict is produced (input must not be mutated/aliased).
        self.assertIsNot(result, source)
        # Nested dict is also rebuilt.
        self.assertIsNot(result["n"], source["n"])

    def test_list_and_tuple_types_preserved(self):
        source = {"a": [2, (3, 4)]}
        result = nested_reduce_tensor(source)

        self.assertEqual(result, {"a": [2, (3, 4)]})
        # Outer container stays a list, rebuilt as a new object.
        self.assertIsInstance(result["a"], list)
        self.assertIsNot(result["a"], source["a"])
        # Inner container stays a tuple (not converted to a list).
        self.assertIsInstance(result["a"][1], tuple)
        self.assertEqual(result["a"][1], (3, 4))

    def test_top_level_list_rebuilt(self):
        source = [1, 2, 3]
        result = nested_reduce_tensor(source)
        self.assertEqual(result, [1, 2, 3])
        self.assertIsInstance(result, list)
        self.assertIsNot(result, source)

    def test_top_level_tuple_rebuilt(self):
        result = nested_reduce_tensor((1, 2, 3))
        self.assertEqual(result, (1, 2, 3))
        self.assertIsInstance(result, tuple)

    def test_does_not_mutate_source_dict(self):
        source = {"a": [1, 2], "b": 3}
        nested_reduce_tensor(source)
        # Original object identities inside source are untouched.
        self.assertEqual(source, {"a": [1, 2], "b": 3})


class NestedCopyPlaceNonTensorTest(unittest.TestCase):
    """nested_copy_place on non-tensor structures (default place=None).

    Should rebuild dict structure recursively while returning non-dict values
    (lists, scalars) by reference.
    """

    def test_dict_tree_copied_lists_shared(self):
        inner = {"c": 3}
        lst = [1, 2]
        source = {"a": 1, "b": inner, "lst": lst}

        result = nested_copy_place(source)

        self.assertEqual(result, {"a": 1, "b": {"c": 3}, "lst": [1, 2]})
        self.assertEqual(list(result.keys()), ["a", "b", "lst"])
        self.assertIsNot(result, source)
        self.assertIsNot(result["b"], inner)
        # Lists are not recursed, so they are shared by reference.
        self.assertIs(result["lst"], lst)

    def test_non_dict_returned_by_reference(self):
        lst = [1, 2, 3]
        self.assertIs(nested_copy_place(lst), lst)
        self.assertEqual(nested_copy_place(42), 42)
        self.assertIsNone(nested_copy_place(None))

    def test_mutating_copy_does_not_affect_source(self):
        source = {"a": 1, "b": {"c": 2}}
        result = nested_copy_place(source)
        result["b"]["c"] = 999
        self.assertEqual(source, {"a": 1, "b": {"c": 2}})


if __name__ == "__main__":
    unittest.main()
