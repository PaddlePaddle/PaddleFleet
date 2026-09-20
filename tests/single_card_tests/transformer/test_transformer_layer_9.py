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
from types import SimpleNamespace

try:
    import numpy as np
    import paddle
    from paddle.distributed.fleet.meta_parallel import ScheduleNode

    from paddlefleet.transformer.identity_op import (
        IdentityFuncOp,
        IdentityOp,
    )
    from paddlefleet.transformer.moe.moe_layer import MoELayer
    from paddlefleet.transformer.transformer_layer import (
        TransformerLayerNode,
        TransformerLayerSublayersSpec,
        tensors_clone,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest capability probe
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    "dependencies available"
    if _HAS_DEPS
    else f"required deps unavailable: {_IMPORT_ERROR!r}"
)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestTensorsClone(unittest.TestCase):
    """Behavioural tests for ``tensors_clone``.

    ``clone`` must return a *new* tensor object holding identical values with
    independent storage, must preserve container type (tuple stays tuple, list
    stays list), must pass non-tensor scalars through unchanged, must recurse
    into nested dicts, and must reject unsupported top-level types.
    """

    def test_single_tensor_is_cloned_not_returned(self):
        t = paddle.arange(6, dtype="float32").reshape([2, 3])
        out = tensors_clone(t)
        self.assertIsInstance(out, paddle.Tensor)
        # A distinct object (not the identity) proves a real clone happened.
        self.assertIsNot(out, t)
        self.assertEqual(out.dtype, t.dtype)
        np.testing.assert_array_equal(out.numpy(), t.numpy())
        # Independent storage: mutating the clone must not touch the source.
        out[0, 0] = 999.0
        self.assertEqual(float(t[0, 0]), 0.0)

    def test_tuple_of_tensors_preserves_order_content_and_type(self):
        t1 = paddle.arange(6, dtype="float32").reshape([2, 3])
        t2 = paddle.arange(6, 26, dtype="float32").reshape([4, 5])
        out = tensors_clone((t1, t2))
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertIsNot(out[0], t1)
        self.assertIsNot(out[1], t2)
        np.testing.assert_array_equal(out[0].numpy(), t1.numpy())
        np.testing.assert_array_equal(out[1].numpy(), t2.numpy())

    def test_list_with_none_and_tensor_keeps_none_and_list_type(self):
        t = paddle.arange(6, dtype="float32").reshape([2, 3])
        out = tensors_clone([None, t])
        self.assertIsInstance(out, list)
        self.assertIsNone(out[0])
        self.assertIsNot(out[1], t)
        np.testing.assert_array_equal(out[1].numpy(), t.numpy())

    def test_non_tensor_scalars_pass_through_unchanged(self):
        t = paddle.arange(4, dtype="float32")
        out = tensors_clone([t, 5, "abc"])
        self.assertIsInstance(out, list)
        self.assertIsNot(out[0], t)
        np.testing.assert_array_equal(out[0].numpy(), t.numpy())
        self.assertEqual(out[1], 5)
        self.assertEqual(out[2], "abc")

    def test_dict_nested_in_tuple_is_recursed_and_cloned(self):
        inner = {"a": paddle.arange(3, dtype="float32")}
        out = tensors_clone((inner,))
        self.assertIsInstance(out, tuple)
        self.assertIsInstance(out[0], dict)
        self.assertIsNot(out[0]["a"], inner["a"])
        np.testing.assert_array_equal(out[0]["a"].numpy(), inner["a"].numpy())

    def test_top_level_dict_clones_every_value(self):
        d = {
            "x": paddle.arange(3, dtype="float32"),
            "y": paddle.arange(3, 6, dtype="float32"),
        }
        out = tensors_clone(d)
        self.assertIsInstance(out, dict)
        self.assertEqual(set(out), {"x", "y"})
        self.assertIsNot(out["x"], d["x"])
        self.assertIsNot(out["y"], d["y"])
        np.testing.assert_array_equal(out["x"].numpy(), d["x"].numpy())
        np.testing.assert_array_equal(out["y"].numpy(), d["y"].numpy())

    def test_unsupported_top_level_types_raise_value_error(self):
        with self.assertRaises(ValueError):
            tensors_clone(42)
        with self.assertRaises(ValueError):
            tensors_clone(None)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestTransformerLayerSublayersSpec(unittest.TestCase):
    """The dataclass must wire the documented identity defaults and must give
    each instance an independent ``sharded_state_dict_keys_map`` (mutable
    default correctness), while honouring explicitly supplied overrides."""

    def test_default_sublayers_are_identity_ops(self):
        spec = TransformerLayerSublayersSpec()
        self.assertIs(spec.input_layernorm, IdentityOp)
        self.assertIs(spec.self_attn, IdentityOp)
        self.assertIs(spec.mlp, IdentityOp)
        self.assertIs(spec.post_attention_layernorm, IdentityOp)
        # bias-dropout-add slots default to the functional identity op.
        self.assertIs(spec.self_attn_bda, IdentityFuncOp)
        self.assertIs(spec.mlp_bda, IdentityFuncOp)

    def test_sharded_keys_map_defaults_to_independent_empty_dicts(self):
        s1 = TransformerLayerSublayersSpec()
        s2 = TransformerLayerSublayersSpec()
        self.assertEqual(s1.sharded_state_dict_keys_map, {})
        self.assertEqual(s2.sharded_state_dict_keys_map, {})
        # Mutating one instance must not leak into another (no shared default).
        s1.sharded_state_dict_keys_map["old"] = "new"
        self.assertEqual(s1.sharded_state_dict_keys_map, {"old": "new"})
        self.assertEqual(s2.sharded_state_dict_keys_map, {})
        self.assertIsNot(
            s1.sharded_state_dict_keys_map, s2.sharded_state_dict_keys_map
        )

    def test_explicit_overrides_are_stored(self):
        attn = object()
        mlp = object()
        keys = {"a": "b"}
        spec = TransformerLayerSublayersSpec(
            self_attn=attn, mlp=mlp, sharded_state_dict_keys_map=keys
        )
        self.assertIs(spec.self_attn, attn)
        self.assertIs(spec.mlp, mlp)
        self.assertEqual(spec.sharded_state_dict_keys_map, {"a": "b"})
        # untouched slots keep their identity defaults.
        self.assertIs(spec.input_layernorm, IdentityOp)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestTransformerLayerNodeInit(unittest.TestCase):
    """``TransformerLayerNode.__init__`` selects its schedule sub-nodes from
    ``isinstance(node.mlp, MoELayer)``. Exercise the dense (non-sparse) branch
    with a real (non-MoE) collaborator and confirm the branch decision, the
    propagated config/layer_number/full_recompute, and that only the dense
    sub-nodes are built."""

    def _make_dense_node(self, full_recompute):
        # A plain object standing in for the layer; crucially NOT a MoELayer,
        # which is what drives the dense branch under test.
        mlp = SimpleNamespace()
        node = SimpleNamespace(
            compute_attention=lambda *a, **k: a,
            compute_mlp=lambda *a, **k: a,
            full_recompute=full_recompute,
            mlp=mlp,
        )
        return node, mlp

    def test_dense_branch_builds_only_attn_and_mlp_nodes(self):
        node, mlp = self._make_dense_node(full_recompute=False)
        self.assertNotIsInstance(mlp, MoELayer)  # justifies the dense branch
        config = SimpleNamespace(num_nextn_predict_layers=0)

        tln = TransformerLayerNode(node, config, name="layer", layer_number=7)

        self.assertIs(tln.config, config)
        self.assertEqual(tln.layer_number, 7)
        self.assertFalse(tln._is_sparse)
        self.assertIsInstance(tln.attn_node, ScheduleNode)
        self.assertIsInstance(tln.mlp_node, ScheduleNode)
        self.assertIsNot(tln.attn_node, tln.mlp_node)
        # sparse-only sub-nodes must NOT be constructed on the dense path.
        for sparse_attr in (
            "gate_node",
            "dispatch_node",
            "combine_node",
            "aux_loss_node",
            "pre_process_node",
            "post_process_node",
        ):
            self.assertFalse(
                hasattr(tln, sparse_attr),
                f"dense node unexpectedly built {sparse_attr}",
            )

    def test_full_recompute_flag_is_read_from_node(self):
        config = SimpleNamespace(num_nextn_predict_layers=0)
        node_true, _ = self._make_dense_node(full_recompute=True)
        node_false, _ = self._make_dense_node(full_recompute=False)

        tln_true = TransformerLayerNode(node_true, config)
        tln_false = TransformerLayerNode(node_false, config)

        self.assertTrue(tln_true.full_recompute)
        self.assertFalse(tln_false.full_recompute)
        # default layer_number when not supplied.
        self.assertEqual(tln_true.layer_number, 1)


if __name__ == "__main__":
    unittest.main()
