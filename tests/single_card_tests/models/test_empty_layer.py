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
"""Behavior tests for ``paddlefleet.models.common.empty_layer.EmptyLayer``.

Derived from the production source, which defines three observable contracts:

* ``__init__(config)`` only forwards to ``FleetLayer.__init__`` which does
  ``super().__init__()`` then ``self.config = config``; the config object is
  *stored by identity* and never inspected, so the honest assertion is
  ``layer.config is config`` (a plain sentinel suffices -- no valid
  ``TransformerConfig`` is required, and a ``MagicMock`` would only hide that).
* ``forward(x)`` is ``return x`` -- it returns the *same* object for every
  input type, so ``forward(obj) is obj`` is the exact contract (stronger than
  the value-equality the old coverage test used for tensors).
* ``build_schedule_node()`` returns a fresh
  ``ScheduleNode(self.forward, name="EmptyLayer")`` on each call; the wrapped
  callable is the identity ``forward``, so running the node passes values
  through unchanged.

The whole module (via ``FleetLayer`` -> ``paddle.nn.Layer`` and
``ScheduleNode``) requires Paddle. Paddle is not installed in this
environment, so every test below is skipped with an honest reason rather than
faked. Only ``ImportError`` is treated as "dependency missing"; any other
import failure is allowed to surface as a real error.
"""

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

try:
    import numpy as np
    import paddle
    from paddle.distributed.fleet.meta_parallel import ScheduleNode

    from paddlefleet.models.common.empty_layer import EmptyLayer
    from paddlefleet.transformer.layer import FleetLayer
except ImportError as exc:  # honest: only a missing dependency skips
    np = None
    paddle = None
    ScheduleNode = None
    EmptyLayer = None
    FleetLayer = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

_HAS_PADDLE = _IMPORT_ERROR is None
_SKIP_REASON = f"paddle/paddlefleet import unavailable: {_IMPORT_ERROR}"


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestEmptyLayerInit(unittest.TestCase):
    """Construction stores the config object and wires up the base classes."""

    def test_config_is_stored_by_identity(self):
        # A plain sentinel: production never reads a config field, so using a
        # real object (not a MagicMock) proves the arg is stored, not faked.
        sentinel = object()
        layer = EmptyLayer(config=sentinel)
        self.assertIs(layer.config, sentinel)

    def test_is_fleet_and_paddle_layer(self):
        # Inheritance is a real contract: parameter/sublayer registration and
        # the __call__ dispatch below both depend on it.
        self.assertTrue(issubclass(EmptyLayer, FleetLayer))
        layer = EmptyLayer(config=object())
        self.assertIsInstance(layer, FleetLayer)
        self.assertIsInstance(layer, paddle.nn.Layer)


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestEmptyLayerForward(unittest.TestCase):
    """``forward`` returns the identical object it was given."""

    def setUp(self):
        self.layer = EmptyLayer(config=object())

    def test_forward_returns_same_object_for_each_type(self):
        # Distinguishable, non-degenerate values of several types; each must
        # come back as the *same* object (identity), which is exactly
        # ``return x``. Value-equality alone would pass a copy too.
        marker = object()
        cases = [
            marker,
            None,
            0,
            42,
            "empty-layer",
            [1, 2, 3],
            {"hidden_states": 7},
            (4, 5),
        ]
        for value in cases:
            with self.subTest(value=value):
                self.assertIs(self.layer.forward(value), value)

    def test_call_dispatch_passes_tensor_through(self):
        # The realistic entry is ``layer(x)`` via ``paddle.nn.Layer.__call__``;
        # it must reach ``forward`` and hand back the same tensor unchanged.
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32"
        )
        out = self.layer(x)
        self.assertIs(out, x)
        np.testing.assert_array_equal(out.numpy(), x.numpy())

    def test_forward_does_not_copy_mutable_input(self):
        # Because the same list object is returned, a later mutation is visible
        # through the returned reference -- confirms no defensive copy.
        data = [10, 20]
        out = self.layer.forward(data)
        self.assertIs(out, data)
        data.append(30)
        self.assertEqual(out, [10, 20, 30])


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestEmptyLayerBuildScheduleNode(unittest.TestCase):
    """``build_schedule_node`` wraps the identity ``forward`` in a ScheduleNode."""

    def setUp(self):
        self.layer = EmptyLayer(config=object())

    def test_returns_schedule_node_instance(self):
        node = self.layer.build_schedule_node()
        self.assertIsInstance(node, ScheduleNode)

    def test_each_call_builds_a_fresh_node(self):
        # Production constructs a new ScheduleNode on every call.
        first = self.layer.build_schedule_node()
        second = self.layer.build_schedule_node()
        self.assertIsInstance(second, ScheduleNode)
        self.assertIsNot(first, second)

    def test_node_runs_the_identity_forward(self):
        # Running the node must pass values through unchanged, which is the
        # signature of the wrapped identity ``forward`` (a node wrapping any
        # transforming function would fail this). ScheduleNode.forward(x) is a
        # supported call path (see modeling_pp.check_accept_none_grad).
        node = self.layer.build_schedule_node()
        x = paddle.arange(6, dtype="float32").reshape([2, 3])
        x.stop_gradient = False
        out = node.forward(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        np.testing.assert_array_equal(out.numpy(), x.numpy())


if __name__ == "__main__":
    unittest.main()
