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
"""Behavior tests for ``build_overlapped_nodes``.

Target: ``paddlefleet.transformer.transformer_encoder.build_overlapped_nodes``,
the pure Python partitioner that splits a forward and a backward
``ScheduleChunk`` into (forward_pre, backward_pre, overlap, forward_post,
backward_post) chunks. Its whole contract is *structural*: which node ends up
in which chunk, in what order, and how the surviving overlap layers are paired
into ``TransformerLayerOverlappedScheduleNode`` wrappers.

Every expectation here is a partition derived BY HAND from the documented
loop semantics -- never by calling the function under test to generate the
expected grouping. Nodes carry distinct identities so a mis-routed node, a
dropped reversal, an off-by-one in the ``min`` count, or a wrong zip pairing
is caught by identity (not by count alone).

Overlap nodes must be genuine ``TransformerLayerNode`` instances because the
partitioner classifies purely via ``isinstance``. They are built through the
real (non-sparse) ``TransformerLayerNode.__init__`` using a lightweight
stand-in for the *non-tested* layer collaborator; the function under test is
executed for real end to end. Heavy imports (paddle + the module) are guarded
so a missing runtime is reported honestly as a skip; only
ImportError/ModuleNotFoundError counts as "dependency absent" so real API
breaks still surface.
"""

import os
import sys
import types
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

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle/paddlefleet not importable in this environment: {_IMPORT_ERROR!r}"
    if _IMPORT_ERROR is not None
    else ""
)


class _StandInLayer:
    """Minimal non-sparse layer collaborator for TransformerLayerNode.

    TransformerLayerNode.__init__ (non-sparse branch) only needs a callable
    ``compute_attention``/``compute_mlp``, a ``full_recompute`` flag, and an
    ``mlp`` that is NOT a MoELayer (so ``_is_sparse`` is False). This is the
    non-tested dependency; the tested partitioner never invokes these.
    """

    full_recompute = False
    mlp = None  # isinstance(None, MoELayer) -> False -> non-sparse path

    def compute_attention(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("collaborator compute must never run here")

    def compute_mlp(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("collaborator compute must never run here")


def _overlap(name):
    """Build a genuine (non-sparse) TransformerLayerNode instance."""
    return TransformerLayerNode(
        _StandInLayer(), config=types.SimpleNamespace(), name=name
    )


def _plain(name):
    """Build a plain ScheduleNode (never an overlap element)."""
    return ScheduleNode(lambda x: x, name=name)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestBuildOverlappedNodes(unittest.TestCase):
    def _assert_nodes_are(self, chunk, expected):
        self.assertIsInstance(chunk, ScheduleChunk)
        self.assertEqual(len(chunk.nodes), len(expected))
        for got, exp in zip(chunk.nodes, expected):
            self.assertIs(got, exp)

    def test_returns_five_schedule_chunks(self):
        result = build_overlapped_nodes(
            ScheduleChunk([_plain("f")]), ScheduleChunk([_plain("b")])
        )
        self.assertEqual(len(result), 5)
        for chunk in result:
            self.assertIsInstance(chunk, ScheduleChunk)

    def test_no_overlap_layers_partition(self):
        f0, f1 = _plain("f0"), _plain("f1")
        b0, b1 = _plain("b0"), _plain("b1")
        fwd_pre, bwd_pre, overlap, fwd_post, bwd_post = build_overlapped_nodes(
            ScheduleChunk([f0, f1]), ScheduleChunk([b0, b1])
        )
        # No TransformerLayerNode anywhere -> overlap empty, everything is
        # "pre" and backward order is restored after the internal reversal.
        self.assertEqual(len(overlap.nodes), 0)
        self._assert_nodes_are(fwd_pre, [f0, f1])
        self._assert_nodes_are(fwd_post, [])
        self._assert_nodes_are(bwd_pre, [b0, b1])
        self._assert_nodes_are(bwd_post, [])

    def test_min_count_reversal_extra_to_post_and_pairing(self):
        f_pre = _plain("f_pre")
        of0, of1, of2 = _overlap("of0"), _overlap("of1"), _overlap("of2")
        f_mid = _plain("f_mid")
        b_a = _plain("b_a")
        ob0, ob1 = _overlap("ob0"), _overlap("ob1")
        b_b = _plain("b_b")

        forward = ScheduleChunk([f_pre, of0, of1, f_mid, of2])  # 3 overlap
        backward = ScheduleChunk([b_a, ob0, ob1, b_b])  # 2 overlap
        # overlap_layers_num = min(3, 2) = 2.

        fwd_pre, bwd_pre, overlap, fwd_post, bwd_post = build_overlapped_nodes(
            forward, backward
        )

        # Forward: f_pre precedes first overlap -> pre; first two overlaps ->
        # overlap; f_mid comes after an overlap -> post; of2 is the surplus
        # overlap (count already == 2) -> post.
        self._assert_nodes_are(fwd_pre, [f_pre])
        self._assert_nodes_are(fwd_post, [f_mid, of2])

        # Backward traversed as reversed([b_a, ob0, ob1, b_b]) =
        # [b_b, ob1, ob0, b_a]: b_b -> pre; ob1, ob0 -> overlap; b_a -> post.
        # pre/post lists are reversed back to original order afterwards.
        self._assert_nodes_are(bwd_pre, [b_b])
        self._assert_nodes_are(bwd_post, [b_a])

        # Overlap pairing: zip(forward_overlap=[of0, of1],
        #                       backward_overlap=[ob1, ob0]).
        self.assertEqual(len(overlap.nodes), 2)
        for node in overlap.nodes:
            self.assertIsInstance(node, TransformerLayerOverlappedScheduleNode)
        self.assertIs(overlap.nodes[0].forward_node, of0)
        self.assertIs(overlap.nodes[0].backward_node, ob1)
        self.assertIs(overlap.nodes[1].forward_node, of1)
        self.assertIs(overlap.nodes[1].backward_node, ob0)

    def test_backward_leading_plains_land_in_post_after_reversal(self):
        of0, of1 = _overlap("of0"), _overlap("of1")
        f_mid = _plain("f_mid")
        b_pre1, b_pre2 = _plain("b_pre1"), _plain("b_pre2")
        ob0, ob1 = _overlap("ob0"), _overlap("ob1")

        forward = ScheduleChunk([of0, f_mid, of1])  # 2 overlap
        backward = ScheduleChunk([b_pre1, b_pre2, ob0, ob1])  # 2 overlap

        fwd_pre, bwd_pre, overlap, fwd_post, bwd_post = build_overlapped_nodes(
            forward, backward
        )

        # Forward: no plain before the first overlap -> empty pre; f_mid sits
        # between overlaps -> post.
        self._assert_nodes_are(fwd_pre, [])
        self._assert_nodes_are(fwd_post, [f_mid])

        # Backward reversed = [ob1, ob0, b_pre2, b_pre1]: overlaps first, so
        # is_pre is already False when the plains appear -> they go to post,
        # and post is reversed back to original order [b_pre1, b_pre2].
        self._assert_nodes_are(bwd_pre, [])
        self._assert_nodes_are(bwd_post, [b_pre1, b_pre2])

        self.assertEqual(len(overlap.nodes), 2)
        self.assertIs(overlap.nodes[0].forward_node, of0)
        self.assertIs(overlap.nodes[0].backward_node, ob1)
        self.assertIs(overlap.nodes[1].forward_node, of1)
        self.assertIs(overlap.nodes[1].backward_node, ob0)


if __name__ == "__main__":
    unittest.main()
