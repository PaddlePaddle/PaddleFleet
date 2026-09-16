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

"""Behaviour tests for paddlefleet.transformer.transformer_layer helpers.

Two production entry points are exercised, both CPU-only:

* ``tensors_clone`` -- a pure recursive copy helper. Its contract (read from
  the source): a ``paddle.Tensor`` is deep-copied; inside a ``list``/``tuple``
  tensor items are cloned, ``dict`` items are recursed, and every other object
  is preserved by identity; a bare ``dict`` clones each value; any other input
  type raises ``ValueError``. Expected results below are constructed by hand
  with distinguishable content, independent of the helper.
* ``TransformerLayerNode`` / ``TransformerLayerOverlappedScheduleNode`` -- only
  their CPU-safe construction assembly and input-guard assertions are checked
  (the MTP-overlap guard, the schedule-node type guard, and the ``split_bw``
  guard). The full forward/backward runs through Paddle's autograd
  ``ScheduleNode`` machinery and is NOT exercised here; those paths belong to a
  device/parallel test, so this suite does not claim to verify them.

paddle is imported behind a try/except; the suite is skipped honestly (with the
import error repr) when paddle / paddlefleet cannot be imported.
"""

import unittest

try:
    import numpy as np
    import paddle
    from paddle.distributed.fleet.meta_parallel import ScheduleNode

    from paddlefleet.transformer.transformer_layer import (
        TransformerLayerNode,
        TransformerLayerOverlappedScheduleNode,
        tensors_clone,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    "" if _HAS_DEPS else f"paddle/paddlefleet unavailable: {_IMPORT_ERROR!r}"
)


class _DenseLayerStub:
    """Minimal stand-in for a dense TransformerLayer consumed by the node.

    ``TransformerLayerNode.__init__`` reads ``mlp`` (a non-MoELayer keeps the
    node in its dense branch), ``full_recompute`` and the two ``compute_*``
    callables. Nothing is invoked at construction time.
    """

    def __init__(self, full_recompute=False):
        self.full_recompute = full_recompute
        self.mlp = object()

    def compute_attention(self, *args, **kwargs):
        raise AssertionError("compute_attention must not run in these tests")

    def compute_mlp(self, *args, **kwargs):
        raise AssertionError("compute_mlp must not run in these tests")


class _DenseConfig:
    num_nextn_predict_layers = None
    mtp_load_weight_only = False


class _MTPConfig:
    """A config that requests MTP-overlap layers, which the node rejects."""

    def __init__(self, num_nextn_predict_layers):
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.mtp_load_weight_only = False


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TensorsCloneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paddle.set_device("cpu")

    def test_bare_tensor_is_copied_not_aliased(self):
        src = paddle.arange(6, dtype="float32").reshape([2, 3])
        before = src.numpy().copy()
        cloned = tensors_clone(src)

        self.assertIsInstance(cloned, paddle.Tensor)
        self.assertIsNot(cloned, src)
        np.testing.assert_array_equal(cloned.numpy(), before)
        # Mutating the source in place must not disturb the clone.
        paddle.assign(paddle.full_like(src, 99.0), src)
        np.testing.assert_array_equal(cloned.numpy(), before)

    def test_list_clones_tensors_and_preserves_others_in_order(self):
        marker = object()
        t0 = paddle.to_tensor([1.0, 2.0], dtype="float32")
        t1 = paddle.to_tensor([3.0, 4.0, 5.0], dtype="float32")
        cloned = tensors_clone([t0, 7, "tag", marker, t1])

        self.assertIsInstance(cloned, list)
        self.assertEqual(len(cloned), 5)
        self.assertIsNot(cloned[0], t0)
        np.testing.assert_array_equal(cloned[0].numpy(), [1.0, 2.0])
        self.assertEqual(cloned[1], 7)
        self.assertEqual(cloned[2], "tag")
        self.assertIs(cloned[3], marker)
        self.assertIsNot(cloned[4], t1)
        np.testing.assert_array_equal(cloned[4].numpy(), [3.0, 4.0, 5.0])

    def test_tuple_returns_tuple_and_keeps_non_tensor_identity(self):
        marker = object()
        t = paddle.to_tensor([8.0], dtype="float32")
        cloned = tensors_clone((t, marker))

        self.assertIsInstance(cloned, tuple)
        self.assertIsNot(cloned[0], t)
        np.testing.assert_array_equal(cloned[0].numpy(), [8.0])
        self.assertIs(cloned[1], marker)

    def test_dict_item_inside_list_is_recursively_cloned(self):
        inner = paddle.to_tensor([10.0, 11.0], dtype="float32")
        cloned = tensors_clone([{"h": inner}])

        self.assertIsInstance(cloned, list)
        self.assertIsInstance(cloned[0], dict)
        self.assertEqual(list(cloned[0]), ["h"])
        self.assertIsNot(cloned[0]["h"], inner)
        np.testing.assert_array_equal(cloned[0]["h"].numpy(), [10.0, 11.0])

    def test_nested_list_item_is_preserved_by_identity_not_recursed(self):
        # Only dict items are recursed inside a sequence; a nested list falls
        # into the "else" branch and is returned unchanged (same object).
        inner_list = [paddle.to_tensor([1.0], dtype="float32")]
        cloned = tensors_clone([inner_list])

        self.assertIsInstance(cloned, list)
        self.assertIs(cloned[0], inner_list)

    def test_bare_dict_clones_each_tensor_value(self):
        a = paddle.to_tensor([1.0, 2.0], dtype="float32")
        b = paddle.to_tensor([3.0], dtype="float32")
        cloned = tensors_clone({"a": a, "b": b})

        self.assertIsInstance(cloned, dict)
        self.assertEqual(set(cloned), {"a", "b"})
        self.assertIsNot(cloned["a"], a)
        self.assertIsNot(cloned["b"], b)
        np.testing.assert_array_equal(cloned["a"].numpy(), [1.0, 2.0])
        np.testing.assert_array_equal(cloned["b"].numpy(), [3.0])

    def test_unsupported_scalar_type_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "Unsupported data type"):
            tensors_clone(123)

    @unittest.expectedFailure
    def test_dict_should_preserve_non_tensor_value_like_sequences(self):
        # BUG (transformer_layer.py:129-133): the bare-dict branch calls
        # ``value.clone()`` unconditionally, unlike the list/tuple branch which
        # preserves non-tensor items. A dict carrying a non-tensor value raises
        # AttributeError instead of preserving it. The consistent/correct
        # behaviour is asserted here; production currently violates it.
        marker = object()
        t = paddle.to_tensor([1.0], dtype="float32")
        cloned = tensors_clone({"tensor": t, "flag": marker})
        self.assertIsNot(cloned["tensor"], t)
        self.assertIs(cloned["flag"], marker)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TransformerLayerNodeConstructionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paddle.set_device("cpu")

    def test_dense_layer_assembles_dense_branch(self):
        layer = _DenseLayerStub(full_recompute=False)
        cfg = _DenseConfig()
        node = TransformerLayerNode(layer, cfg, name="dense", layer_number=4)

        self.assertFalse(node._is_sparse)
        self.assertFalse(node.full_recompute)
        self.assertIs(node.config, cfg)
        self.assertEqual(node.layer_number, 4)
        self.assertIsInstance(node.attn_node, ScheduleNode)
        self.assertIsInstance(node.mlp_node, ScheduleNode)
        # The sparse-only sublayer nodes must NOT exist on the dense path.
        self.assertFalse(hasattr(node, "gate_node"))
        self.assertFalse(hasattr(node, "dispatch_node"))

    def test_full_recompute_flag_is_propagated_from_layer(self):
        node = TransformerLayerNode(
            _DenseLayerStub(full_recompute=True), _DenseConfig(), name="d"
        )
        self.assertTrue(node.full_recompute)

    def test_forward_rejects_positive_mtp_and_reports_value(self):
        node = TransformerLayerNode(
            _DenseLayerStub(), _MTPConfig(1), name="mtp"
        )
        with self.assertRaisesRegex(AssertionError, r"but get 1"):
            node.forward(
                {"hidden_states": paddle.ones([1, 1], dtype="float32")}
            )

    def test_forward_guard_message_reflects_configured_value(self):
        # A different value flows into the guard message, proving the assertion
        # consumes the config rather than being unconditional.
        node = TransformerLayerNode(
            _DenseLayerStub(), _MTPConfig(3), name="mtp3"
        )
        with self.assertRaisesRegex(AssertionError, r"but get 3"):
            node.forward(
                {"hidden_states": paddle.ones([1, 1], dtype="float32")}
            )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TransformerLayerOverlappedScheduleNodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paddle.set_device("cpu")

    def _dense_node(self):
        return TransformerLayerNode(_DenseLayerStub(), _DenseConfig(), name="d")

    def test_accepts_two_transformer_layer_nodes(self):
        fwd = self._dense_node()
        bwd = self._dense_node()
        node = TransformerLayerOverlappedScheduleNode(fwd, bwd, name="ov")

        self.assertIs(node.forward_node, fwd)
        self.assertIs(node.backward_node, bwd)
        self.assertIs(node.config, fwd.config)

    def test_rejects_plain_schedule_node(self):
        plain = ScheduleNode(lambda x: x, name="plain")
        with self.assertRaises(AssertionError):
            TransformerLayerOverlappedScheduleNode(plain, plain)

    def test_forward_backward_rejects_split_bw(self):
        node = TransformerLayerOverlappedScheduleNode(
            self._dense_node(), self._dense_node(), name="ov"
        )
        with self.assertRaises(AssertionError):
            node.forward_backward({}, [], split_bw=True)


if __name__ == "__main__":
    unittest.main()
