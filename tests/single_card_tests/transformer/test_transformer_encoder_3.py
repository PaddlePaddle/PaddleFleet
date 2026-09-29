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

"""Behaviour tests for ``build_overlapped_nodes`` in ``transformer_encoder``.

``build_overlapped_nodes`` is a pure partitioning function: given a forward and
a backward ``ScheduleChunk`` it splits their nodes into pre / overlap / post
chunks based *solely* on whether each node ``isinstance`` a
``TransformerLayerNode`` (the decoder-layer marker type), pairs the leading
``min(#forward_TLN, #backward_TLN)`` decoder layers into
``TransformerLayerOverlappedScheduleNode`` wrappers, and pushes any surplus
decoder layers plus trailing non-decoder nodes into the post chunks.  It never
invokes any method on the nodes, so the expected partition and pairing can be
derived by hand from the input ordering.

The real production function is exercised end-to-end.  Only the node *inputs*
are lightweight: genuine ``ScheduleNode`` instances stand in for the
non-decoder nodes, and a tagged subclass of the real ``TransformerLayerNode``
stands in for decoder layers (its heavy ``__init__`` needs a fully built layer
and is irrelevant to this type-based partitioning).  This is a CPU-only,
single-process orchestration test; it makes no claim about pipeline-parallel
communication or numerics.
"""

import unittest

try:
    from paddle.distributed.fleet.meta_parallel import (
        ScheduleChunk,
        ScheduleNode,
    )

    from paddlefleet.transformer.transformer_encoder import (
        build_overlapped_nodes,
    )
    from paddlefleet.transformer.transformer_layer import (
        TransformerLayerNode,
        TransformerLayerOverlappedScheduleNode,
    )

    class _OverlapConfig:
        # Minimal stand-in for TransformerConfig. TransformerLayerOverlapped-
        # ScheduleNode.__init__ copies forward_node.config, so the fixture must
        # expose one; these are the only attributes the overlap path reads.
        num_nextn_predict_layers = None
        mtp_load_weight_only = False

    class _LayerNodeFixture(TransformerLayerNode):
        """A genuine ``TransformerLayerNode`` used purely as a typed input.

        ``build_overlapped_nodes`` partitions strictly by
        ``isinstance(node, TransformerLayerNode)`` and calls no method on the
        nodes, so the heavy real ``__init__`` (which requires a fully built
        decoder layer) is intentionally bypassed. ``tag`` makes every fixture
        individually identifiable so exact placement and the forward/backward
        pairing order can be asserted, not merely counted.
        """

        def __init__(self, tag):
            self.tag = tag
            self.config = _OverlapConfig()

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    _IMPORT_ERROR = exc


_SKIP_REASON = (
    ""
    if _IMPORT_ERROR is None
    else f"paddle/paddlefleet import unavailable: {_IMPORT_ERROR!r}"
)


def _plain(name):
    """A real non-decoder ScheduleNode (not a TransformerLayerNode)."""
    return ScheduleNode(lambda inputs: inputs, name=name)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestBuildOverlappedNodes(unittest.TestCase):
    def _assert_chunk_is(self, chunk, expected_nodes):
        self.assertIsInstance(chunk, ScheduleChunk)
        self.assertEqual(len(chunk.nodes), len(expected_nodes))
        for got, want in zip(chunk.nodes, expected_nodes):
            self.assertIs(got, want)

    def test_balanced_overlap_pairs_and_reverses_backward(self):
        # Forward order: pre P0, decoder L0, decoder L1, post P1.
        # Backward order: Q0, decoder M0, decoder M1, Q1.
        p0, p1 = _plain("fpre"), _plain("fpost")
        q0, q1 = _plain("bpost"), _plain("bpre")
        l0, l1 = _LayerNodeFixture("L0"), _LayerNodeFixture("L1")
        m0, m1 = _LayerNodeFixture("M0"), _LayerNodeFixture("M1")

        (
            fwd_pre,
            bwd_pre,
            overlap,
            fwd_post,
            bwd_post,
        ) = build_overlapped_nodes(
            ScheduleChunk([p0, l0, l1, p1]),
            ScheduleChunk([q0, m0, m1, q1]),
        )

        # Non-decoder nodes before the first decoder layer are "pre"; those
        # after it are "post". The backward chunk is walked in reverse, so its
        # trailing q1 becomes pre and its leading q0 becomes post.
        self._assert_chunk_is(fwd_pre, [p0])
        self._assert_chunk_is(fwd_post, [p1])
        self._assert_chunk_is(bwd_pre, [q1])
        self._assert_chunk_is(bwd_post, [q0])

        # overlap_layers_num == min(2, 2) == 2. Forward overlap keeps forward
        # order [L0, L1]; backward overlap is built from the reversed walk so it
        # is [M1, M0]; zip pairs (L0,M1) then (L1,M0).
        self.assertEqual(len(overlap.nodes), 2)
        for node in overlap.nodes:
            self.assertIsInstance(node, TransformerLayerOverlappedScheduleNode)
        self.assertIs(overlap.nodes[0].forward_node, l0)
        self.assertIs(overlap.nodes[0].backward_node, m1)
        self.assertIs(overlap.nodes[1].forward_node, l1)
        self.assertIs(overlap.nodes[1].backward_node, m0)

    def test_surplus_decoder_layers_spill_into_post(self):
        # Three forward decoders, two backward decoders, no plain nodes.
        l0, l1, l2 = (_LayerNodeFixture(f"L{i}") for i in range(3))
        m0, m1 = _LayerNodeFixture("M0"), _LayerNodeFixture("M1")

        (
            fwd_pre,
            bwd_pre,
            overlap,
            fwd_post,
            bwd_post,
        ) = build_overlapped_nodes(
            ScheduleChunk([l0, l1, l2]),
            ScheduleChunk([m0, m1]),
        )

        # overlap_layers_num == min(3, 2) == 2, so only L0,L1 overlap and the
        # surplus L2 is pushed to the forward post chunk.
        self._assert_chunk_is(fwd_pre, [])
        self._assert_chunk_is(fwd_post, [l2])
        self._assert_chunk_is(bwd_pre, [])
        self._assert_chunk_is(bwd_post, [])

        self.assertEqual(len(overlap.nodes), 2)
        self.assertIs(overlap.nodes[0].forward_node, l0)
        self.assertIs(overlap.nodes[0].backward_node, m1)
        self.assertIs(overlap.nodes[1].forward_node, l1)
        self.assertIs(overlap.nodes[1].backward_node, m0)

    def test_zero_backward_decoders_gives_no_overlap(self):
        # Backward chunk has no decoder layers, so overlap_layers_num == 0 and
        # the single forward decoder cannot overlap: it lands in forward post.
        p0, p1 = _plain("fpre"), _plain("fpost")
        q0 = _plain("bonly")
        l0 = _LayerNodeFixture("L0")

        (
            fwd_pre,
            bwd_pre,
            overlap,
            fwd_post,
            bwd_post,
        ) = build_overlapped_nodes(
            ScheduleChunk([p0, l0, p1]),
            ScheduleChunk([q0]),
        )

        self.assertEqual(len(overlap.nodes), 0)
        self._assert_chunk_is(fwd_pre, [p0])
        # L0 is hit as a decoder but overlap capacity is 0, so it is appended to
        # post; p1 (already past the first decoder) follows it, preserving order.
        self._assert_chunk_is(fwd_post, [l0, p1])
        # The lone backward non-decoder node is "pre" (no decoder precedes it).
        self._assert_chunk_is(bwd_pre, [q0])
        self._assert_chunk_is(bwd_post, [])

    def test_returns_five_schedule_chunks_in_order(self):
        result = build_overlapped_nodes(
            ScheduleChunk([_LayerNodeFixture("L0")]),
            ScheduleChunk([_LayerNodeFixture("M0")]),
        )
        self.assertEqual(len(result), 5)
        for chunk in result:
            self.assertIsInstance(chunk, ScheduleChunk)


if __name__ == "__main__":
    unittest.main()
