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

"""CPU-only behavior tests for ``tensors_clone`` in transformer_layer.

``tensors_clone`` is the recompute helper that deep-copies the tensors a
recompute span must keep alive. The contract we exercise here is: every
paddle.Tensor reachable through the supported container shapes is replaced
by an *independent* clone (distinct object, equal values, own storage),
while structure (tuple/list/dict identity, ordering, keys) is preserved and
non-tensor leaves are passed through untouched. Expected values are derived
by hand / with numpy, never by calling the function under test a second
time.
"""

import unittest

import numpy as np

try:
    import paddle

    from paddlefleet.transformer.transformer_layer import tensors_clone

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    paddle = None
    tensors_clone = None
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    ""
    if _HAS_DEPS
    else (
        "paddle / paddlefleet.transformer.transformer_layer unavailable: "
        f"{_IMPORT_ERROR!r}"
    )
)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestTensorsClone(unittest.TestCase):
    """Behavior of the ``tensors_clone`` recompute helper on CPU."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_single_tensor_returns_independent_clone(self):
        # A cloned tensor must be a different object that owns its storage:
        # mutating the original afterwards must not leak into the clone.
        x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = tensors_clone(x)

        self.assertIsNot(out, x)
        np.testing.assert_array_equal(out.numpy(), x.numpy())

        snapshot = x.numpy().copy()
        x[0, 0] = 999.0
        np.testing.assert_array_equal(out.numpy(), snapshot)
        self.assertEqual(float(x[0, 0]), 999.0)

    def test_tuple_clones_tensors_and_preserves_order(self):
        # Distinguishable per-element content so a reorder or a swapped
        # clone would be caught, not just a length check.
        a = paddle.to_tensor([1.0, 2.0])
        b = paddle.to_tensor([3.0, 4.0, 5.0])
        out = tensors_clone((a, b))

        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertIsNot(out[0], a)
        self.assertIsNot(out[1], b)
        np.testing.assert_array_equal(out[0].numpy(), [1.0, 2.0])
        np.testing.assert_array_equal(out[1].numpy(), [3.0, 4.0, 5.0])

        snapshot = a.numpy().copy()
        a[0] = -7.0
        np.testing.assert_array_equal(out[0].numpy(), snapshot)

    def test_list_passes_non_tensor_leaves_through(self):
        # The list/tuple branch clones tensors but forwards non-tensor,
        # non-dict leaves unchanged, keeping their value and position.
        a = paddle.to_tensor([8.0, 9.0])
        out = tensors_clone([a, 42, "kept"])

        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 3)
        self.assertIsNot(out[0], a)
        np.testing.assert_array_equal(out[0].numpy(), [8.0, 9.0])
        self.assertEqual(out[1], 42)
        self.assertEqual(out[2], "kept")

    def test_list_recurses_into_nested_dict(self):
        # A dict nested inside a list is deep-copied via the recursive
        # call: the returned dict is a fresh object whose tensor value is
        # an independent clone.
        inner = {"k": paddle.to_tensor([7.0, 8.0])}
        out = tensors_clone([inner])

        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 1)
        self.assertIsInstance(out[0], dict)
        self.assertIsNot(out[0], inner)
        self.assertEqual(set(out[0]), {"k"})
        self.assertIsNot(out[0]["k"], inner["k"])
        np.testing.assert_array_equal(out[0]["k"].numpy(), [7.0, 8.0])

    def test_dict_of_tensors_clones_values_and_preserves_keys(self):
        # Distinct values per key so a key/value mix-up would be visible.
        d = {"a": paddle.to_tensor([1.0, 2.0]), "b": paddle.to_tensor([3.0])}
        out = tensors_clone(d)

        self.assertIsInstance(out, dict)
        self.assertEqual(set(out), {"a", "b"})
        self.assertIsNot(out["a"], d["a"])
        self.assertIsNot(out["b"], d["b"])
        np.testing.assert_array_equal(out["a"].numpy(), [1.0, 2.0])
        np.testing.assert_array_equal(out["b"].numpy(), [3.0])

        snapshot = d["a"].numpy().copy()
        d["a"][0] = 123.0
        np.testing.assert_array_equal(out["a"].numpy(), snapshot)

    def test_unsupported_top_level_type_raises_value_error(self):
        # A scalar that is neither Tensor nor a supported container hits
        # the explicit guard and raises ValueError.
        with self.assertRaises(ValueError):
            tensors_clone(3.14)

    @unittest.expectedFailure
    def test_dict_with_non_tensor_value_should_pass_through(self):
        # Real bug (asymmetric handling): the tuple/list branch forwards
        # non-tensor leaves untouched, but the dict branch unconditionally
        # calls ``value.clone()`` on every value. A non-tensor dict value
        # therefore raises AttributeError instead of being preserved. The
        # consistent, correct behavior asserted below currently fails; we
        # do not edit production, we document the defect.
        x = paddle.to_tensor([1.0, 2.0])
        out = tensors_clone({"t": x, "n": 7})

        self.assertEqual(out["n"], 7)
        self.assertIsNot(out["t"], x)
        np.testing.assert_array_equal(out["t"].numpy(), [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
