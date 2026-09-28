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

"""CPU control-logic tests for ``paddlefleet.transformer.transformer_encoder``.

These tests exercise real production code paths that are pure Python control
logic (node partitioning / pairing in ``build_overlapped_nodes`` and the small
name/prefix accessors and ``use_fp8`` scan on ``TransformerEncoder``). They do
not require a GPU and perform no collective communication; they therefore only
claim to verify the partition, ordering and pairing logic, not any distributed
schedule execution.

Paddle is imported behind a narrow guard. When paddle / paddlefleet cannot be
imported the whole module is skipped with the concrete error, never faked pass.
"""

import unittest
from types import SimpleNamespace

try:
    from paddle.distributed.fleet.meta_parallel import (
        ScheduleChunk,
        ScheduleNode,
    )

    from paddlefleet.transformer.transformer_encoder import (
        TransformerEncoder,
        build_overlapped_nodes,
    )
    from paddlefleet.transformer.transformer_layer import (
        TransformerLayer,
        TransformerLayerNode,
        TransformerLayerOverlappedScheduleNode,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    _IMPORT_ERROR = exc

_SKIP_REASON = f"paddle/paddlefleet import unavailable: {_IMPORT_ERROR!r}"


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestBuildOverlappedNodes(unittest.TestCase):
    """``build_overlapped_nodes`` partitions forward/backward schedule chunks.

    The first ``min(fwd_layers, bwd_layers)`` ``TransformerLayerNode`` instances
    become overlap nodes; leading non-layer nodes are the "pre" chunk and the
    remainder (extra layer nodes + trailing non-layer nodes) are the "post"
    chunk. The backward side is walked in reverse and then reversed back, so
    original ordering must be restored.
    """

    def _layer_node(self, name):
        # Real subclass of the production node so isinstance() checks are
        # genuine; only the expensive real-layer wiring (attention / MoE that
        # needs a GPU) is bypassed. ``config`` is the sole attribute read while
        # constructing the overlap node.
        class _LayerNodeStub(TransformerLayerNode):
            def __init__(inner, node_name):
                ScheduleNode.__init__(inner, fwd_func=None, name=node_name)
                inner.config = SimpleNamespace()

        return _LayerNodeStub(name)

    def _plain_node(self, name):
        # A genuine ScheduleNode that is NOT a TransformerLayerNode.
        return ScheduleNode(fwd_func=None, name=name)

    def test_partition_and_overlap_pairing(self):
        fe = self._plain_node("fe")
        fl0 = self._layer_node("fl0")
        fl1 = self._layer_node("fl1")
        fl2 = self._layer_node("fl2")
        fn = self._plain_node("fn")
        forward_chunk = ScheduleChunk([fe, fl0, fl1, fl2, fn])

        be0 = self._plain_node("be0")
        be1 = self._plain_node("be1")
        bl0 = self._layer_node("bl0")
        bl1 = self._layer_node("bl1")
        bn0 = self._plain_node("bn0")
        bn1 = self._plain_node("bn1")
        backward_chunk = ScheduleChunk([be0, be1, bl0, bl1, bn0, bn1])

        fwd_pre, bwd_pre, overlap, fwd_post, bwd_post = build_overlapped_nodes(
            forward_chunk, backward_chunk
        )

        # forward: 3 layer nodes, backward: 2 -> overlap count is min(3, 2) = 2.
        # forward_pre = leading non-layer [fe];
        # overlap uses first two layer nodes [fl0, fl1];
        # forward_post = remaining layer node + trailing non-layer [fl2, fn].
        self.assertEqual(len(fwd_pre.nodes), 1)
        self.assertIs(fwd_pre.nodes[0], fe)
        self.assertEqual(len(fwd_post.nodes), 2)
        self.assertIs(fwd_post.nodes[0], fl2)
        self.assertIs(fwd_post.nodes[1], fn)

        # backward is scanned in reverse then reversed back: the two trailing
        # non-layer nodes become the pre chunk in ORIGINAL order [bn0, bn1];
        # the two leading non-layer nodes become the post chunk [be0, be1].
        self.assertEqual(len(bwd_pre.nodes), 2)
        self.assertIs(bwd_pre.nodes[0], bn0)
        self.assertIs(bwd_pre.nodes[1], bn1)
        self.assertEqual(len(bwd_post.nodes), 2)
        self.assertIs(bwd_post.nodes[0], be0)
        self.assertIs(bwd_post.nodes[1], be1)

        # overlap nodes zip forward-order layers with reverse-order backward
        # layers: (fl0, bl1) then (fl1, bl0).
        self.assertEqual(len(overlap.nodes), 2)
        self.assertIsInstance(
            overlap.nodes[0], TransformerLayerOverlappedScheduleNode
        )
        self.assertIs(overlap.nodes[0].forward_node, fl0)
        self.assertIs(overlap.nodes[0].backward_node, bl1)
        self.assertIs(overlap.nodes[1].forward_node, fl1)
        self.assertIs(overlap.nodes[1].backward_node, bl0)

    def test_no_layer_nodes_yields_no_overlap(self):
        fe = self._plain_node("fe")
        fn = self._plain_node("fn")
        be = self._plain_node("be")
        forward_chunk = ScheduleChunk([fe, fn])
        backward_chunk = ScheduleChunk([be])

        fwd_pre, bwd_pre, overlap, fwd_post, bwd_post = build_overlapped_nodes(
            forward_chunk, backward_chunk
        )

        self.assertEqual(len(overlap.nodes), 0)
        # With no layer node, is_pre never flips: everything is "pre".
        self.assertEqual(len(fwd_pre.nodes), 2)
        self.assertIs(fwd_pre.nodes[0], fe)
        self.assertIs(fwd_pre.nodes[1], fn)
        self.assertEqual(len(fwd_post.nodes), 0)
        self.assertEqual(len(bwd_pre.nodes), 1)
        self.assertIs(bwd_pre.nodes[0], be)
        self.assertEqual(len(bwd_post.nodes), 0)

    def test_rejects_non_schedulechunk_backward(self):
        with self.assertRaises(AssertionError):
            build_overlapped_nodes(ScheduleChunk([]), "not_a_chunk")


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestUseFp8Scan(unittest.TestCase):
    """``TransformerEncoder.use_fp8`` scans layers for any fp8-enabled one.

    The method ignores everything on ``self`` except the stage count and the
    layer container, so it can be driven directly on a lightweight holder to
    exercise the real loop / isinstance / short-circuit logic.
    """

    def test_returns_false_without_fp8_layers_non_vpp(self):
        holder = SimpleNamespace(
            _num_virtual_pipeline_stages=1,
            run_function=[object(), object()],
        )
        result = TransformerEncoder.use_fp8(holder)
        self.assertIs(result, False)

    def test_returns_true_when_an_fp8_layer_is_present_non_vpp(self):
        class _AlwaysFp8Layer(TransformerLayer):
            def __init__(inner):  # bypass heavy nn.Layer wiring
                pass

            def use_fp8(inner):
                return True

        # A non-layer object precedes the fp8 layer: it must be skipped by the
        # isinstance guard (otherwise object() has no use_fp8 and would raise),
        # and the real fp8 layer must flip the result to True.
        holder = SimpleNamespace(
            _num_virtual_pipeline_stages=1,
            run_function=[object(), _AlwaysFp8Layer()],
        )
        result = TransformerEncoder.use_fp8(holder)
        self.assertIs(result, True)

    @unittest.expectedFailure
    def test_vpp_branch_should_return_false_without_fp8_layers(self):
        # BUG: transformer_encoder.py:569-579 -- in the
        # ``_num_virtual_pipeline_stages > 1`` branch there is no ``return
        # False`` after the loop, so a stage with no fp8 layer returns None
        # instead of False (the else branch returns False correctly). Asserting
        # the correct contract; marked expectedFailure until production is fixed.
        holder = SimpleNamespace(
            _num_virtual_pipeline_stages=2,
            _model_chunks=[[object()], [object()]],
        )
        result = TransformerEncoder.use_fp8(holder)
        self.assertIs(result, False)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestSequentialLayerAccessors(unittest.TestCase):
    """add_sequential_layer / get_sequential_layers / name-prefix accessors.

    These record each layer descriptor together with its pipeline name prefix
    and later expose them by position. The accessors do not depend on the heavy
    ``PipelineLayer`` initialisation, so they run on a plain data holder.
    """

    def test_records_and_exposes_layers_in_order_with_prefixes(self):
        emb, blk0, blk1, norm = object(), object(), object(), object()

        # add_sequential_layer appends to the provided list (it does not use
        # self), pairing each descriptor with its prefix.
        layers = []
        TransformerEncoder.add_sequential_layer(None, layers, emb, "model")
        TransformerEncoder.add_sequential_layer(
            None, layers, blk0, "model.layers.0"
        )
        TransformerEncoder.add_sequential_layer(
            None, layers, blk1, "model.layers.1"
        )
        TransformerEncoder.add_sequential_layer(None, layers, norm, "model")

        self.assertEqual(
            [entry["name_prefix"] for entry in layers],
            ["model", "model.layers.0", "model.layers.1", "model"],
        )
        self.assertIs(layers[0]["layer"], emb)
        self.assertIs(layers[3]["layer"], norm)

        holder = SimpleNamespace(_sequential_layers=layers)

        # get_sequential_layers extracts the "layer" objects, preserving order
        # and identity (not just the count).
        extracted = TransformerEncoder.get_sequential_layers(holder)
        self.assertEqual(len(extracted), 4)
        self.assertIs(extracted[0], emb)
        self.assertIs(extracted[1], blk0)
        self.assertIs(extracted[2], blk1)
        self.assertIs(extracted[3], norm)

        # get_sequential_name_prefixes maps stringified position -> prefix; the
        # duplicated "model" prefix stays distinct by index (0 vs 3).
        prefixes = TransformerEncoder.get_sequential_name_prefixes(holder)
        self.assertEqual(
            prefixes,
            {
                "0": "model",
                "1": "model.layers.0",
                "2": "model.layers.1",
                "3": "model",
            },
        )


if __name__ == "__main__":
    unittest.main()
