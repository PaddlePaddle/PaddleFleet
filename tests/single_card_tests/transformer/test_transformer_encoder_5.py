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

"""Behavior tests for ``paddlefleet.transformer.transformer_encoder``.

Two pieces of pure (CPU-only) control logic are exercised against
independently hand-derived expectations:

1. ``build_overlapped_nodes(forward_chunk, backward_chunk)`` -- partitions
   two ``ScheduleChunk`` node lists into pre / overlap / post chunks. The
   contract is more than the *counts* the source once relied on:
     * only ``TransformerLayerNode`` instances are eligible for overlap;
     * exactly ``min(#fwd_layers, #bwd_layers)`` layer nodes overlap, in
       forward order, extras spill to the *post* chunk;
     * a non-layer node seen *after* the first layer node moves to *post*
       (the ``is_pre`` flag flips), while leading non-layer nodes stay in
       *pre*;
     * the backward list is walked in reverse, so the k-th overlap pair is
       ``(forward_overlap[k], backward_overlap_in_reverse[k])`` -- i.e. the
       first forward layer pairs with the *last* backward layer;
     * ``pre``/``post`` backward chunks are reversed back to original order.
   The assertions below lock node *identity* and overlap *pairing*, so a
   swap, a lost reversal or a mis-count is rejected -- none of which a
   length-only check would catch.

2. ``TransformerEncoder.get_layer_desc_list`` /
   ``get_encoder_layer_desc_list`` / ``get_sequential_layers`` /
   ``get_sequential_name_prefixes`` -- assemble the ordered layer-desc
   list. Expected order, the shared ``model[.<modal>].layers.<i>`` index
   counter that runs head -> transformer -> tail, and the wrapping
   asymmetry (embedding / head / transformer / tail are wrapped in a fresh
   ``LayerDesc`` while ``layer_norm`` is appended raw) are all derived by
   hand and checked by content/identity, not by count.

The encoder methods are invoked on a bare instance built with
``object.__new__`` because the real ``__init__`` calls into Fleet /
distributed topology setup that is unavailable on CPU; the methods under
test themselves run for real and are the thing being observed.

paddle is a hard dependency of the code under test and cannot be mocked;
if it (or paddlefleet) is not importable the whole module is skipped with
the concrete import error, never silently "passed".
"""

from __future__ import annotations

import unittest

try:
    import paddle  # used for paddle.nn.Layer marker types
    from paddle.distributed.fleet.meta_parallel import LayerDesc, ScheduleChunk
    from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
        ScheduleNode,
    )

    from paddlefleet.transformer.transformer_encoder import (
        TransformerEncoder,
        build_overlapped_nodes,
    )
    from paddlefleet.transformer.transformer_layer import (
        TransformerLayerNode,
        TransformerLayerOverlappedScheduleNode,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    _IMPORT_ERROR = repr(exc)


_SKIP_REASON = f"paddle/paddlefleet import unavailable: {_IMPORT_ERROR}"


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestBuildOverlappedNodes(unittest.TestCase):
    """Partition + pairing contract of ``build_overlapped_nodes``."""

    def _layer_node(self, name):
        """A real ``TransformerLayerNode`` (overlap-eligible).

        ``__init__`` builds sub-nodes from a full TransformerLayer; the
        partitioner only relies on the *type* and on ``.config`` (read by
        ``TransformerLayerOverlappedScheduleNode``), so we materialize the
        exact type and give it a config sentinel.
        """
        node = object.__new__(TransformerLayerNode)
        ScheduleNode.__init__(node, fwd_func=None, name=name)
        node.config = object()
        return node

    def _plain_node(self, name):
        """A real ``ScheduleNode`` that is NOT overlap-eligible."""
        node = object.__new__(ScheduleNode)
        ScheduleNode.__init__(node, fwd_func=None, name=name)
        return node

    def test_mixed_leading_plain_nodes_and_forward_order_pairing(self):
        fo = self._plain_node("fo")
        ft1 = self._layer_node("ft1")
        ft2 = self._layer_node("ft2")
        bo = self._plain_node("bo")
        bt1 = self._layer_node("bt1")
        bt2 = self._layer_node("bt2")

        # backward chunk is stored in execution order [bt2, bt1, bo];
        # the function walks it in reverse for pairing.
        forward_chunk = ScheduleChunk([fo, ft1, ft2])
        backward_chunk = ScheduleChunk([bt2, bt1, bo])

        fwd_pre, bwd_pre, overlap, fwd_post, bwd_post = build_overlapped_nodes(
            forward_chunk, backward_chunk
        )

        # Leading non-layer nodes stay in pre; nothing spills to post.
        self.assertEqual(list(fwd_pre.nodes), [fo])
        self.assertEqual(list(bwd_pre.nodes), [bo])
        self.assertEqual(list(fwd_post.nodes), [])
        self.assertEqual(list(bwd_post.nodes), [])

        # Two overlap pairs; forward in order, backward paired in reverse.
        self.assertEqual(len(overlap.nodes), 2)
        for node in overlap.nodes:
            self.assertIsInstance(node, TransformerLayerOverlappedScheduleNode)
        self.assertIs(overlap.nodes[0].forward_node, ft1)
        self.assertIs(overlap.nodes[0].backward_node, bt1)
        self.assertIs(overlap.nodes[1].forward_node, ft2)
        self.assertIs(overlap.nodes[1].backward_node, bt2)

    def test_extra_forward_layers_spill_to_post_reverse_pairing(self):
        ft = [self._layer_node(f"ft{i}") for i in range(4)]
        bt = [self._layer_node(f"bt{i}") for i in range(2)]

        forward_chunk = ScheduleChunk(list(ft))
        backward_chunk = ScheduleChunk(list(bt))

        fwd_pre, bwd_pre, overlap, fwd_post, bwd_post = build_overlapped_nodes(
            forward_chunk, backward_chunk
        )

        # overlap = min(4, 2) = 2 forward layers, in forward order.
        self.assertEqual(list(fwd_pre.nodes), [])
        self.assertEqual(list(bwd_pre.nodes), [])
        # Surplus forward layer nodes (index 2, 3) spill to forward_post.
        self.assertEqual(list(fwd_post.nodes), [ft[2], ft[3]])
        self.assertEqual(list(bwd_post.nodes), [])

        # backward layers consumed in reverse -> ft0 pairs with bt1, etc.
        self.assertEqual(len(overlap.nodes), 2)
        self.assertIs(overlap.nodes[0].forward_node, ft[0])
        self.assertIs(overlap.nodes[0].backward_node, bt[1])
        self.assertIs(overlap.nodes[1].forward_node, ft[1])
        self.assertIs(overlap.nodes[1].backward_node, bt[0])

    def test_plain_node_after_first_layer_moves_to_post(self):
        fo1 = self._plain_node("fo1")
        ft1 = self._layer_node("ft1")
        fo2 = self._plain_node("fo2")
        ft2 = self._layer_node("ft2")
        bt = [self._layer_node(f"bt{i}") for i in range(2)]

        forward_chunk = ScheduleChunk([fo1, ft1, fo2, ft2])
        backward_chunk = ScheduleChunk(list(bt))

        fwd_pre, bwd_pre, overlap, fwd_post, bwd_post = build_overlapped_nodes(
            forward_chunk, backward_chunk
        )

        # fo1 (before any layer) -> pre; fo2 (after first layer) -> post.
        self.assertEqual(list(fwd_pre.nodes), [fo1])
        self.assertEqual(list(fwd_post.nodes), [fo2])
        self.assertEqual(list(bwd_pre.nodes), [])
        self.assertEqual(list(bwd_post.nodes), [])

        self.assertEqual(len(overlap.nodes), 2)
        self.assertIs(overlap.nodes[0].forward_node, ft1)
        self.assertIs(overlap.nodes[0].backward_node, bt[1])
        self.assertIs(overlap.nodes[1].forward_node, ft2)
        self.assertIs(overlap.nodes[1].backward_node, bt[0])

    def test_no_layer_nodes_all_pre_order_preserved(self):
        fo1 = self._plain_node("fo1")
        fo2 = self._plain_node("fo2")
        bo1 = self._plain_node("bo1")
        bo2 = self._plain_node("bo2")

        forward_chunk = ScheduleChunk([fo1, fo2])
        backward_chunk = ScheduleChunk([bo1, bo2])

        fwd_pre, bwd_pre, overlap, fwd_post, bwd_post = build_overlapped_nodes(
            forward_chunk, backward_chunk
        )

        self.assertEqual(len(overlap.nodes), 0)
        self.assertEqual(list(fwd_pre.nodes), [fo1, fo2])
        self.assertEqual(list(fwd_post.nodes), [])
        # backward walked in reverse then reversed back -> original order.
        self.assertEqual(list(bwd_pre.nodes), [bo1, bo2])
        self.assertEqual(list(bwd_post.nodes), [])


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestLayerDescList(unittest.TestCase):
    """Ordering / prefix / wrapping contract of the layer-desc builders."""

    def _bare_encoder(self, modal=None):
        # Real __init__ requires Fleet topology; the desc builders do not,
        # so we invoke them on a bare instance carrying only the inputs
        # they read (``modal`` and the collaborator methods on the class).
        encoder = object.__new__(TransformerEncoder)
        encoder.modal = modal
        return encoder

    def _make_spec(self):
        # Distinguishable marker layers (LayerDesc requires nn.Layer types).
        class _Emb(paddle.nn.Layer):
            pass

        class _Head0(paddle.nn.Layer):
            pass

        class _T0(paddle.nn.Layer):
            pass

        class _T1(paddle.nn.Layer):
            pass

        class _Tail0(paddle.nn.Layer):
            pass

        class _Norm(paddle.nn.Layer):
            pass

        class _Spec:
            embedding = _Emb
            head_empty_layers = [_Head0]
            transformer_layers = [_T0, _T1]
            tail_empty_layers = [_Tail0]
            layer_norm = _Norm

        return _Spec()

    def test_order_prefixes_and_wrapping_no_modal(self):
        encoder = self._bare_encoder(modal=None)
        spec = self._make_spec()

        layers = encoder.get_layer_desc_list(spec)
        encoder._sequential_layers = layers

        # Expected order: embedding, head, transformer x2, tail, layer_norm.
        self.assertEqual(len(layers), 6)

        # Shared "model.layers.<i>" counter runs head(0) -> t(1) -> t(2)
        # -> tail(3); embedding and layer_norm carry the bare "model".
        self.assertEqual(
            encoder.get_sequential_name_prefixes(),
            {
                "0": "model",
                "1": "model.layers.0",
                "2": "model.layers.1",
                "3": "model.layers.2",
                "4": "model.layers.3",
                "5": "model",
            },
        )

        seq = encoder.get_sequential_layers()
        # embedding / head / transformer / tail wrapped in a fresh LayerDesc
        # whose layer_func is exactly the marker type, in order.
        expected_wrapped = [
            spec.embedding,
            spec.head_empty_layers[0],
            spec.transformer_layers[0],
            spec.transformer_layers[1],
            spec.tail_empty_layers[0],
        ]
        for entry, marker in zip(seq[:5], expected_wrapped):
            self.assertIsInstance(entry, LayerDesc)
            self.assertIs(entry.layer_func, marker)

        # layer_norm is appended raw (NOT re-wrapped) -> identity holds and
        # it is not a LayerDesc instance.
        self.assertIs(seq[5], spec.layer_norm)
        self.assertNotIsInstance(seq[5], LayerDesc)

    def test_modal_prefix_applied_everywhere(self):
        encoder = self._bare_encoder(modal="vision")
        spec = self._make_spec()

        layers = encoder.get_layer_desc_list(spec)
        encoder._sequential_layers = layers

        self.assertEqual(
            encoder.get_sequential_name_prefixes(),
            {
                "0": "model.vision",
                "1": "model.vision.layers.0",
                "2": "model.vision.layers.1",
                "3": "model.vision.layers.2",
                "4": "model.vision.layers.3",
                "5": "model.vision",
            },
        )

    def test_encoder_layer_desc_list_index_counter(self):
        # get_encoder_layer_desc_list mutates ``layers`` in place, returns
        # None, and numbers head -> transformer -> tail with one counter.
        encoder = self._bare_encoder(modal=None)
        spec = self._make_spec()

        layers = []
        result = encoder.get_encoder_layer_desc_list(layers, spec, "model")
        self.assertIsNone(result)

        # 1 head + 2 transformer + 1 tail, no embedding/layer_norm here.
        self.assertEqual(
            [e["name_prefix"] for e in layers],
            [
                "model.layers.0",
                "model.layers.1",
                "model.layers.2",
                "model.layers.3",
            ],
        )
        self.assertEqual(
            [e["layer"].layer_func for e in layers],
            [
                spec.head_empty_layers[0],
                spec.transformer_layers[0],
                spec.transformer_layers[1],
                spec.tail_empty_layers[0],
            ],
        )


if __name__ == "__main__":
    unittest.main()
