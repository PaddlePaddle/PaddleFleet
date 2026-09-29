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

import unittest

try:
    import numpy as np
    import paddle

    from paddlefleet.transformer.identity_op import IdentityFuncOp, IdentityOp
    from paddlefleet.transformer.transformer_layer import (
        TransformerLayerSublayersSpec,
        tensors_clone,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    _IMPORT_ERROR = exc

_DEPS_AVAILABLE = _IMPORT_ERROR is None
_SKIP_REASON = (
    ""
    if _DEPS_AVAILABLE
    else f"paddle/paddlefleet not importable in this environment: {_IMPORT_ERROR!r}"
)


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class TestTensorsClone(unittest.TestCase):
    """Behavior of ``tensors_clone`` derived by hand from the production code.

    ``tensors_clone`` is a pure CPU helper that deep-copies the tensors it is
    handed so a recompute span keeps its own storage. Contract, per branch:

    * ``paddle.Tensor``            -> ``value.clone()`` (new storage, same value)
    * ``list`` / ``tuple``         -> per item: clone tensors; recurse into
                                      dict items; append everything else (ints,
                                      strings, ``None``, nested containers)
                                      unchanged, by reference.
    * ``dict``                     -> every value is cloned unconditionally.
    * anything else                -> ``ValueError``.
    """

    def test_tensor_input_returns_independent_clone(self):
        t = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32"
        )
        out = tensors_clone(t)
        # A real clone: distinct Python object AND distinct storage, same values.
        self.assertIsNot(out, t)
        self.assertNotEqual(out.data_ptr(), t.data_ptr())
        np.testing.assert_array_equal(out.numpy(), t.numpy())

    def test_list_preserves_nontensor_items_by_reference(self):
        t = paddle.to_tensor([10.0, 20.0], dtype="float32")
        sentinel_none = None
        sentinel_str = "tag"
        src = [t, 7, sentinel_str, sentinel_none]
        out = tensors_clone(src)

        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 4)
        # Tensor entry cloned (new storage), value preserved.
        self.assertIsNot(out[0], t)
        self.assertNotEqual(out[0].data_ptr(), t.data_ptr())
        np.testing.assert_array_equal(out[0].numpy(), [10.0, 20.0])
        # Non-tensor entries kept as-is, by identity.
        self.assertEqual(out[1], 7)
        self.assertIs(out[2], sentinel_str)
        self.assertIsNone(out[3])

    def test_tuple_type_and_values_preserved(self):
        t1 = paddle.to_tensor([1.0, 2.0], dtype="float32")
        t2 = paddle.to_tensor([[3.0, 4.0], [5.0, 6.0]], dtype="float32")
        out = tensors_clone((t1, t2))

        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertIsNot(out[0], t1)
        self.assertIsNot(out[1], t2)
        np.testing.assert_array_equal(out[0].numpy(), t1.numpy())
        np.testing.assert_array_equal(out[1].numpy(), t2.numpy())

    def test_empty_list_returns_empty_list(self):
        out = tensors_clone([])
        self.assertIsInstance(out, list)
        self.assertEqual(out, [])

    def test_dict_of_tensors_clones_each_value(self):
        ta = paddle.to_tensor([1.0, 2.0], dtype="float32")
        tb = paddle.to_tensor([9.0, 8.0, 7.0], dtype="float32")
        src = {"a": ta, "b": tb}
        out = tensors_clone(src)

        self.assertIsInstance(out, dict)
        self.assertEqual(set(out), {"a", "b"})
        self.assertIsNot(out["a"], ta)
        self.assertIsNot(out["b"], tb)
        self.assertNotEqual(out["a"].data_ptr(), ta.data_ptr())
        self.assertNotEqual(out["b"].data_ptr(), tb.data_ptr())
        np.testing.assert_array_equal(out["a"].numpy(), [1.0, 2.0])
        np.testing.assert_array_equal(out["b"].numpy(), [9.0, 8.0, 7.0])

    def test_list_containing_dict_is_recursed_into_new_dict(self):
        t = paddle.to_tensor([[1.0], [2.0]], dtype="float32")
        inner = {"x": t}
        out = tensors_clone([inner])

        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 1)
        # Recursion rebuilds the dict, so it is a fresh object holding a clone.
        self.assertIsInstance(out[0], dict)
        self.assertIsNot(out[0], inner)
        self.assertIsNot(out[0]["x"], t)
        np.testing.assert_array_equal(out[0]["x"].numpy(), t.numpy())

    def test_nested_list_is_shallow_not_deep_cloned(self):
        # Documented limitation: a nested list/tuple item hits the catch-all
        # ``append(item)`` branch, so neither the inner container nor the tensor
        # it holds is cloned. This pins the current (shallow) contract.
        t = paddle.to_tensor([1.0, 2.0], dtype="float32")
        inner = [t]
        out = tensors_clone([inner])
        self.assertIs(out[0], inner)
        self.assertIs(out[0][0], t)

    def test_unsupported_scalar_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            tensors_clone(42)
        message = str(ctx.exception)
        self.assertIn("Unsupported data type", message)
        self.assertIn("int", message)

    @unittest.expectedFailure
    def test_dict_with_none_value_should_be_preserved_like_list_branch(self):
        # Bug: the list branch preserves non-tensor values (see
        # test_list_preserves_nontensor_items_by_reference), but the dict branch
        # calls ``value.clone()`` unconditionally. A dict carrying a non-tensor
        # value (e.g. an optional ``attention_mask=None`` in the forward-args
        # dict passed to ``tensors_clone(inputs)``) therefore raises
        # AttributeError instead of preserving it. Asserting the consistent,
        # intended behavior; expected to fail until the dict branch guards
        # non-tensor values.
        out = tensors_clone({"a": None})
        self.assertIsNone(out["a"])


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class TestTransformerLayerSublayersSpecDefaults(unittest.TestCase):
    """Default sublayer wiring that drives ``build_spec_layer`` construction."""

    def test_identity_defaults_select_correct_op_classes(self):
        spec = TransformerLayerSublayersSpec()
        # Norm/attention/mlp slots default to the plain IdentityOp.
        for name in (
            "input_layernorm",
            "self_attention_hyper_connection",
            "self_attn",
            "pre_cross_attn_layernorm",
            "cross_attention",
            "post_attention_layernorm",
            "mlp_hyper_connection",
            "mlp",
            "block_attn_res",
        ):
            self.assertIs(getattr(spec, name), IdentityOp, name)
        # bias-dropout-add slots default to IdentityFuncOp, which is a distinct
        # subclass; assert the exact class so an IdentityOp default is rejected.
        for name in ("self_attn_bda", "cross_attn_bda", "mlp_bda"):
            self.assertIs(getattr(spec, name), IdentityFuncOp, name)
            self.assertIsNot(getattr(spec, name), IdentityOp, name)

    def test_sharded_map_is_fresh_dict_per_instance(self):
        a = TransformerLayerSublayersSpec()
        b = TransformerLayerSublayersSpec()
        self.assertEqual(a.sharded_state_dict_keys_map, {})
        self.assertEqual(b.sharded_state_dict_keys_map, {})
        # default_factory=dict must hand each instance an independent mapping,
        # not a shared mutable default.
        self.assertIsNot(
            a.sharded_state_dict_keys_map, b.sharded_state_dict_keys_map
        )
        a.sharded_state_dict_keys_map["k"] = "v"
        self.assertEqual(b.sharded_state_dict_keys_map, {})


if __name__ == "__main__":
    unittest.main()
