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

"""Behavior tests for ``build_overlapped_nodes`` in the pipeline TransformerEncoder.

``build_overlapped_nodes`` is pure Python scheduling orchestration: given a
forward and a backward ``ScheduleChunk``, it splits each into pre / overlap /
post sub-chunks based on which nodes are ``TransformerLayerNode`` instances,
clips the overlap length to ``min(#fwd_layers, #bwd_layers)``, and pairs the
forward-order overlap layers with the reverse-order backward overlap layers into
``TransformerLayerOverlappedScheduleNode`` objects.

The tests below drive the real function with genuine ``ScheduleChunk`` /
``ScheduleNode`` / ``TransformerLayerNode`` instances and check the exact
membership, ordering and forward/backward pairing of every produced sub-chunk.
No production behaviour is faked: the lightweight layer object only supplies the
attributes ``TransformerLayerNode.__init__`` reads at construction time; the
partitioning logic under test never invokes them.
"""

import unittest

_IMPORT_ERROR = None
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
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    _IMPORT_ERROR = exc
    ScheduleChunk = None
    ScheduleNode = None
    build_overlapped_nodes = None
    TransformerLayerNode = None
    TransformerLayerOverlappedScheduleNode = None

_SKIP_REASON = (
    "requires an importable paddle + paddlefleet.transformer stack "
    "(paddle.distributed.fleet.meta_parallel.ScheduleChunk/ScheduleNode, "
    "build_overlapped_nodes, TransformerLayerNode); "
    f"import failed with: {_IMPORT_ERROR!r}"
)


class _MinimalLayer:
    """Minimal construction input for a genuine ``TransformerLayerNode``.

    ``TransformerLayerNode.__init__`` reads ``full_recompute``, ``mlp`` (checked
    only via ``isinstance(..., MoELayer)``) and, on the non-sparse path,
    ``compute_attention`` / ``compute_mlp`` (wrapped into child ``ScheduleNode``
    ``fwd_func`` slots). ``build_overlapped_nodes`` classifies nodes purely by
    ``isinstance``; none of these callables are ever invoked by the code under
    test, so raising here documents that expectation.
    """

    full_recompute = False
    mlp = object()  # not a MoELayer -> non-sparse TransformerLayerNode

    def compute_attention(self, *args, **kwargs):
        raise AssertionError(
            "compute_attention must not be called by build_overlapped_nodes"
        )

    def compute_mlp(self, *args, **kwargs):
        raise AssertionError(
            "compute_mlp must not be called by build_overlapped_nodes"
        )


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestBuildOverlappedNodes(unittest.TestCase):
    """Partitioning and pairing contract of ``build_overlapped_nodes``."""

    def _plain(self, name):
        """A real ScheduleNode that is NOT a TransformerLayerNode."""
        return ScheduleNode(lambda *a, **k: None, name=name)

    def _layer_node(self):
        """A genuine TransformerLayerNode (non-sparse construction)."""
        return TransformerLayerNode(
            _MinimalLayer(), config=object(), name="tln"
        )

    def _assert_same_nodes(self, chunk, expected):
        """Assert chunk.nodes are exactly ``expected`` (by identity, in order)."""
        actual = list(chunk.nodes)
        self.assertEqual(
            len(actual),
            len(expected),
            f"expected {len(expected)} nodes, got {len(actual)}",
        )
        for got, exp in zip(actual, expected):
            self.assertIs(got, exp)

    def test_partitions_by_type_and_pairs_overlap_in_reverse(self):
        """Overlap clips to min layer count; overflow goes to post; overlap
        pairs forward-order layers with reverse-order backward layers."""
        fp_a = self._plain("fp_a")
        ft0 = self._layer_node()
        ft1 = self._layer_node()
        ft2 = self._layer_node()
        fp_b = self._plain("fp_b")
        forward_chunk = ScheduleChunk([fp_a, ft0, ft1, ft2, fp_b])

        bt0 = self._layer_node()
        bp_a = self._plain("bp_a")
        bt1 = self._layer_node()
        backward_chunk = ScheduleChunk([bt0, bp_a, bt1])

        (
            f_pre,
            b_pre,
            overlap,
            f_post,
            b_post,
        ) = build_overlapped_nodes(forward_chunk, backward_chunk)

        # overlap_layers_num = min(3 forward TLN, 2 backward TLN) = 2.
        # Forward: leading plain -> pre; first 2 TLN -> overlap; the 3rd TLN
        # overflows to post, trailing plain follows it in post.
        self._assert_same_nodes(f_pre, [fp_a])
        self._assert_same_nodes(f_post, [ft2, fp_b])

        # Backward is walked in reverse: [bt1, bp_a, bt0]. First TLN seen (bt1)
        # flips is_pre off, so no backward pre layers; bp_a lands in post; bt0
        # is the 2nd overlap layer.
        self._assert_same_nodes(b_pre, [])
        self._assert_same_nodes(b_post, [bp_a])

        self.assertEqual(len(overlap.nodes), 2)
        for node in overlap.nodes:
            self.assertIsInstance(node, TransformerLayerOverlappedScheduleNode)
        # Forward overlap preserves forward order [ft0, ft1]; backward overlap
        # consumes reverse backward order [bt1, bt0]; zip pairs them positionally.
        self.assertIs(overlap.nodes[0].forward_node, ft0)
        self.assertIs(overlap.nodes[0].backward_node, bt1)
        self.assertIs(overlap.nodes[1].forward_node, ft1)
        self.assertIs(overlap.nodes[1].backward_node, bt0)

    def test_no_overlap_when_one_side_has_no_transformer_layers(self):
        """min(#fwd, 0) == 0 => empty overlap; every forward TLN overflows to
        post; a leading backward plain node stays in pre."""
        ft0 = self._layer_node()
        ft1 = self._layer_node()
        forward_chunk = ScheduleChunk([ft0, ft1])

        bp = self._plain("bp")
        backward_chunk = ScheduleChunk([bp])

        (
            f_pre,
            b_pre,
            overlap,
            f_post,
            b_post,
        ) = build_overlapped_nodes(forward_chunk, backward_chunk)

        self._assert_same_nodes(overlap, [])
        self._assert_same_nodes(f_pre, [])
        self._assert_same_nodes(f_post, [ft0, ft1])
        self._assert_same_nodes(b_pre, [bp])
        self._assert_same_nodes(b_post, [])

    def test_backward_post_layers_restore_original_order(self):
        """Backward post layers are collected while iterating the reversed
        chunk, then re-reversed, so they come back in original chunk order."""
        ft = self._layer_node()
        forward_chunk = ScheduleChunk([ft])

        bp_1 = self._plain("bp_1")
        bp_2 = self._plain("bp_2")
        bt = self._layer_node()
        # Original order: [bp_1, bp_2, bt]. Reversed walk: [bt, bp_2, bp_1];
        # bt flips is_pre off, so bp_2 then bp_1 accumulate as post, and the
        # final reversal restores [bp_1, bp_2].
        backward_chunk = ScheduleChunk([bp_1, bp_2, bt])

        (
            f_pre,
            b_pre,
            overlap,
            f_post,
            b_post,
        ) = build_overlapped_nodes(forward_chunk, backward_chunk)

        self._assert_same_nodes(f_pre, [])
        self._assert_same_nodes(f_post, [])
        self._assert_same_nodes(b_pre, [])
        self._assert_same_nodes(b_post, [bp_1, bp_2])

        self.assertEqual(len(overlap.nodes), 1)
        self.assertIsInstance(
            overlap.nodes[0], TransformerLayerOverlappedScheduleNode
        )
        self.assertIs(overlap.nodes[0].forward_node, ft)
        self.assertIs(overlap.nodes[0].backward_node, bt)

    def test_rejects_non_schedule_chunk_inputs(self):
        """The function asserts both inputs are ScheduleChunk instances."""
        real = ScheduleChunk([self._layer_node()])
        with self.assertRaises(AssertionError):
            build_overlapped_nodes([self._layer_node()], real)
        with self.assertRaises(AssertionError):
            build_overlapped_nodes(real, [self._layer_node()])


if __name__ == "__main__":
    unittest.main()
