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
    from paddle.distributed.fleet.meta_parallel import ScheduleChunk

    from paddlefleet.transformer.transformer_encoder import (
        TransformerEncoder,
        build_overlapped_nodes,
    )
    from paddlefleet.transformer.transformer_layer import (
        TransformerLayerNode,
        TransformerLayerOverlappedScheduleNode,
    )

    class _FakeTransformerLayerNode(TransformerLayerNode):
        """Identity-preserving fixture that is a genuine TransformerLayerNode.

        The real ``TransformerLayerNode.__init__`` wires up attention / MoE
        compute collaborators that require GPU + expert-parallel groups. The
        partition logic under test only inspects ``isinstance(n,
        TransformerLayerNode)`` and passes the node objects through unchanged,
        so a real subclass with a lightweight ``__init__`` is a faithful stand
        in without mocking the function under test.
        """

        def __init__(self, tag, config):
            # Deliberately skip the heavy base __init__; keep the node a real
            # TransformerLayerNode instance with a distinguishable identity.
            self.tag = tag
            self.config = config

    class _NonLayerNode:
        """Marker that is NOT a TransformerLayerNode.

        ``build_overlapped_nodes`` must route these to the pre/post chunks and
        never into the overlap chunk.
        """

        def __init__(self, tag):
            self.tag = tag

    class _EncoderProbe(TransformerEncoder):
        """TransformerEncoder subclass that skips PipelineLayer initialization.

        ``TransformerEncoder.__init__`` builds a full ``PipelineLayer`` which
        needs a distributed topology. The pure name-mapping getters only read
        ``self._sequential_layers``; a probe subclass exercises the real method
        bodies without standing up distributed state.
        """

        def __init__(self, sequential_layers):
            self._sequential_layers = sequential_layers

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    ScheduleChunk = None
    TransformerEncoder = None
    build_overlapped_nodes = None
    TransformerLayerNode = None
    TransformerLayerOverlappedScheduleNode = None
    _FakeTransformerLayerNode = None
    _NonLayerNode = None
    _EncoderProbe = None
    _IMPORT_ERROR = exc

_HAS_STACK = build_overlapped_nodes is not None
_SKIP_REASON = (
    "paddle/paddlefleet transformer stack is not importable on this host; "
    f"import raised {_IMPORT_ERROR!r}"
)


@unittest.skipUnless(_HAS_STACK, _SKIP_REASON)
class TestBuildOverlappedNodes(unittest.TestCase):
    """Independent-reference tests for build_overlapped_nodes partitioning."""

    def _tln(self, tag):
        # Each node carries its own config sentinel so that the overlap wrapper
        # can be checked to adopt the *forward* node's config specifically.
        return _FakeTransformerLayerNode(tag, config=object())

    def _plain(self, tag):
        return _NonLayerNode(tag)

    def test_mixed_partition_pairs_and_overflows(self):
        # Forward: [P0, T0, T1, X0, T2]  (3 transformer-layer nodes)
        # Backward: [T3, T4, Q0]         (2 transformer-layer nodes)
        # overlap_layers_num = min(3, 2) = 2 (hand derived).
        p0 = self._plain("P0")
        t0 = self._tln("T0")
        t1 = self._tln("T1")
        x0 = self._plain("X0")
        t2 = self._tln("T2")
        forward_chunk = ScheduleChunk([p0, t0, t1, x0, t2])

        t3 = self._tln("T3")
        t4 = self._tln("T4")
        q0 = self._plain("Q0")
        backward_chunk = ScheduleChunk([t3, t4, q0])

        (
            fwd_pre,
            bwd_pre,
            overlap,
            fwd_post,
            bwd_post,
        ) = build_overlapped_nodes(forward_chunk, backward_chunk)

        # Forward: P0 precedes the first layer node -> pre; the first two layer
        # nodes fill the 2 overlap slots; X0 and the overflow layer node T2
        # (beyond the overlap count) fall to post, preserving order.
        self.assertEqual([n.tag for n in fwd_pre.nodes], ["P0"])
        self.assertEqual([n.tag for n in fwd_post.nodes], ["X0", "T2"])
        self.assertIs(fwd_pre.nodes[0], p0)
        self.assertIs(fwd_post.nodes[0], x0)
        self.assertIs(fwd_post.nodes[1], t2)

        # Backward is scanned in reverse: reversed([T3,T4,Q0]) = [Q0,T4,T3].
        # Q0 (before any reversed layer node) -> pre; T4 then T3 fill the 2
        # overlap slots; nothing overflows to post. Pre/post lists are stored
        # re-reversed back to original order.
        self.assertEqual([n.tag for n in bwd_pre.nodes], ["Q0"])
        self.assertIs(bwd_pre.nodes[0], q0)
        self.assertEqual(list(bwd_post.nodes), [])

        # Overlap zips forward-overlap [T0,T1] with backward-overlap [T4,T3].
        self.assertEqual(len(overlap.nodes), 2)
        for node in overlap.nodes:
            self.assertIsInstance(node, TransformerLayerOverlappedScheduleNode)
        self.assertIs(overlap.nodes[0].forward_node, t0)
        self.assertIs(overlap.nodes[0].backward_node, t4)
        self.assertIs(overlap.nodes[1].forward_node, t1)
        self.assertIs(overlap.nodes[1].backward_node, t3)
        # The wrapper must adopt the forward node's config, not the backward's.
        self.assertIs(overlap.nodes[0].config, t0.config)
        self.assertIs(overlap.nodes[1].config, t1.config)

    def test_backward_overflow_reversed_reconstruction(self):
        # Forward: [T0]                 (1 transformer-layer node)
        # Backward: [Ta, R0, Tb, Tc]    (3 transformer-layer nodes)
        # overlap_layers_num = min(1, 3) = 1 (hand derived).
        t0 = self._tln("T0")
        forward_chunk = ScheduleChunk([t0])

        ta = self._tln("Ta")
        r0 = self._plain("R0")
        tb = self._tln("Tb")
        tc = self._tln("Tc")
        backward_chunk = ScheduleChunk([ta, r0, tb, tc])

        (
            fwd_pre,
            bwd_pre,
            overlap,
            fwd_post,
            bwd_post,
        ) = build_overlapped_nodes(forward_chunk, backward_chunk)

        # Forward has a single layer node consumed entirely by the one overlap
        # slot, so both forward pre and post are empty.
        self.assertEqual(list(fwd_pre.nodes), [])
        self.assertEqual(list(fwd_post.nodes), [])

        # Backward reversed scan: [Tc, Tb, R0, Ta].
        #   Tc -> only overlap slot (flips is_pre False immediately, so no pre)
        #   Tb -> overflow to post
        #   R0 -> post (already past pre)
        #   Ta -> overflow to post
        # backward_post_layers accumulate as [Tb, R0, Ta] then are re-reversed
        # to original order -> [Ta, R0, Tb].
        self.assertEqual(list(bwd_pre.nodes), [])
        self.assertEqual([n.tag for n in bwd_post.nodes], ["Ta", "R0", "Tb"])
        self.assertIs(bwd_post.nodes[0], ta)
        self.assertIs(bwd_post.nodes[1], r0)
        self.assertIs(bwd_post.nodes[2], tb)

        # Single overlap pair: forward T0 with the last backward layer node Tc.
        self.assertEqual(len(overlap.nodes), 1)
        self.assertIsInstance(
            overlap.nodes[0], TransformerLayerOverlappedScheduleNode
        )
        self.assertIs(overlap.nodes[0].forward_node, t0)
        self.assertIs(overlap.nodes[0].backward_node, tc)

    def test_no_layer_nodes_yields_empty_overlap(self):
        # No TransformerLayerNode anywhere -> overlap count 0; every plain node
        # stays in the pre chunk (is_pre never flips), post stays empty.
        f0 = self._plain("F0")
        f1 = self._plain("F1")
        b0 = self._plain("B0")
        forward_chunk = ScheduleChunk([f0, f1])
        backward_chunk = ScheduleChunk([b0])

        (
            fwd_pre,
            bwd_pre,
            overlap,
            fwd_post,
            bwd_post,
        ) = build_overlapped_nodes(forward_chunk, backward_chunk)

        self.assertEqual(len(overlap.nodes), 0)
        self.assertEqual([n.tag for n in fwd_pre.nodes], ["F0", "F1"])
        self.assertEqual([n.tag for n in bwd_pre.nodes], ["B0"])
        self.assertEqual(list(fwd_post.nodes), [])
        self.assertEqual(list(bwd_post.nodes), [])
        self.assertIs(fwd_pre.nodes[0], f0)
        self.assertIs(fwd_pre.nodes[1], f1)
        self.assertIs(bwd_pre.nodes[0], b0)

    def test_rejects_non_schedulechunk_arguments(self):
        chunk = ScheduleChunk([])
        # Both operands are asserted to be ScheduleChunk instances.
        with self.assertRaises(AssertionError):
            build_overlapped_nodes(chunk, "not_a_chunk")
        with self.assertRaises(AssertionError):
            build_overlapped_nodes("not_a_chunk", chunk)


@unittest.skipUnless(_HAS_STACK, _SKIP_REASON)
class TestTransformerEncoderSequentialMapping(unittest.TestCase):
    """Tests for the pipeline name/layer accessor contract."""

    def test_layers_and_prefixes_use_distinct_keys_in_order(self):
        # Distinguishable layer descriptors and prefixes so a swap of the
        # "layer"/"name_prefix" keys or an index off-by-one would be caught.
        probe = _EncoderProbe(
            [
                {"layer": "descA", "name_prefix": "model"},
                {"layer": "descB", "name_prefix": "model.layers.0"},
                {"layer": "descC", "name_prefix": "model.norm"},
            ]
        )

        # get_sequential_layers must extract the "layer" values, in order.
        self.assertEqual(
            probe.get_sequential_layers(), ["descA", "descB", "descC"]
        )

        # get_sequential_name_prefixes must generate string positional keys and
        # map them to the "name_prefix" values (a different key than above).
        self.assertEqual(
            probe.get_sequential_name_prefixes(),
            {"0": "model", "1": "model.layers.0", "2": "model.norm"},
        )


if __name__ == "__main__":
    unittest.main()
