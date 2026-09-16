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

"""CPU-only unit tests for paddlefleet.models.gpt.gpt_model.

These cover the pieces of the GPT pipeline model that carry logic on their own,
independent of any accelerator:

* ``is_vision_merge_key`` -- the loud guard that decides which state-dict keys
  belong to the vision encoder and rejects parameters placed directly on the
  ``vision_merge`` wrapper (which would otherwise be silently dropped from every
  checkpoint).
* ``GPTSublayersSpec`` -- the LayerSpec container whose field names/defaults are
  the contract the checkpoint name-mapping plumbing relies on.
* ``build_overlapped_nodes`` -- the schedule-chunk partitioner that splits
  forward/backward chunks into pre / overlap / post groups for 1F1B overlap.
* ``GPTModel`` sequential-layer bookkeeping (``add_sequential_layer``,
  ``get_sequential_layers``, ``get_sequential_name_prefixes``,
  ``get_hardware_flops``).

The module under test imports Paddle at load time. When Paddle (or the
paddlefleet package) is not importable the whole file is skipped with an honest
reason rather than reporting a pass. Every test that runs exercises the real
production entry points; no production formula is re-implemented in the test.
"""

import dataclasses
import importlib
import os
import sys
import types
import unittest

_REPO_SRC = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
    "src",
)
if os.path.isdir(_REPO_SRC) and _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)


_IMPORT_ERROR = None
try:
    from paddle.distributed.fleet.meta_parallel import (
        ScheduleChunk,
        ScheduleNode,
    )

    gpt_model = importlib.import_module("paddlefleet.models.gpt.gpt_model")
    from paddlefleet.transformer.transformer_layer import TransformerLayerNode
except (ImportError, ModuleNotFoundError) as exc:  # honest capability probe
    # Only genuine missing-dependency import failures are treated as "skip".
    # Compilation / API errors would raise a different exception type and
    # surface as real failures instead of being swallowed here.
    _IMPORT_ERROR = exc
    gpt_model = None
    ScheduleChunk = None
    ScheduleNode = None
    TransformerLayerNode = None


_AVAILABLE = _IMPORT_ERROR is None
_SKIP_REASON = (
    "paddle / paddlefleet not importable in this environment: "
    f"{_IMPORT_ERROR!r}"
    if _IMPORT_ERROR is not None
    else ""
)


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestIsVisionMergeKey(unittest.TestCase):
    """is_vision_merge_key classifies state-dict keys and rejects unsafe ones."""

    def test_non_vision_merge_key_returns_false(self):
        # Ordinary backbone keys are not owned by the vision wrapper.
        self.assertFalse(
            gpt_model.is_vision_merge_key("model.layers.0.attn.weight")
        )
        self.assertFalse(gpt_model.is_vision_merge_key(""))
        # A key that merely contains the substring but does not start with it.
        self.assertFalse(gpt_model.is_vision_merge_key("model.vision_merge.x"))

    def test_vision_model_key_returns_true(self):
        # Keys under vision_merge.vision_model.* are the re-exported ones.
        self.assertTrue(
            gpt_model.is_vision_merge_key(
                "vision_merge.vision_model.blocks.0.mlp.weight"
            )
        )

    def test_wrapper_direct_param_raises_valueerror(self):
        # A parameter placed directly on vision_merge (not under vision_model)
        # would be dropped from every checkpoint, so the guard must reject it
        # loudly. It must be a ValueError (not an assert, which -O strips) and
        # must name the offending key.
        bad_key = "vision_merge.projector.weight"
        with self.assertRaises(ValueError) as ctx:
            gpt_model.is_vision_merge_key(bad_key)
        self.assertIs(type(ctx.exception), ValueError)
        self.assertIn(bad_key, str(ctx.exception))

    def test_prefix_without_vision_model_segment_is_rejected(self):
        # "vision_merge." alone, and "vision_merge.vision_model" without the
        # trailing separator, both start with the wrapper prefix but not with
        # the "vision_merge.vision_model." export prefix -> loud rejection.
        for bad_key in (
            "vision_merge.",
            "vision_merge.vision_model",
        ):
            with self.assertRaises(ValueError):
                gpt_model.is_vision_merge_key(bad_key)


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestGPTSublayersSpec(unittest.TestCase):
    """GPTSublayersSpec is the LayerSpec container for a GPT pipeline model."""

    _EXPECTED_FIELDS = (
        "embedding",
        "head_empty_layers",
        "mhc_expand",
        "transformer_layers",
        "mhc_contract",
        "tail_empty_layers",
        "mtp",
        "output_block_attn_res",
        "layer_norm",
        "lm_head",
        "mtp_lm_head",
        "mtp_loss",
    )

    def test_is_dataclass_with_expected_field_set(self):
        self.assertTrue(dataclasses.is_dataclass(gpt_model.GPTSublayersSpec))
        names = tuple(
            f.name for f in dataclasses.fields(gpt_model.GPTSublayersSpec)
        )
        self.assertEqual(names, self._EXPECTED_FIELDS)

    def test_all_fields_default_to_none(self):
        spec = gpt_model.GPTSublayersSpec()
        for name in self._EXPECTED_FIELDS:
            self.assertIsNone(
                getattr(spec, name),
                msg=f"field {name!r} should default to None",
            )
        # Every declared default is the literal None (guards against a
        # mutable/shared default sneaking in).
        for f in dataclasses.fields(gpt_model.GPTSublayersSpec):
            self.assertIsNone(f.default, msg=f"field {f.name!r} default")

    def test_values_assigned_via_kwargs_are_stored(self):
        emb = object()
        lm = object()
        layers = [object(), object()]
        spec = gpt_model.GPTSublayersSpec(
            embedding=emb,
            lm_head=lm,
            transformer_layers=layers,
        )
        self.assertIs(spec.embedding, emb)
        self.assertIs(spec.lm_head, lm)
        self.assertIs(spec.transformer_layers, layers)
        # Untouched fields stay None.
        self.assertIsNone(spec.mtp)
        self.assertIsNone(spec.layer_norm)


def _input_chunk(nodes):
    """Build a ScheduleChunk input by bypassing node validation.

    build_overlapped_nodes only reads ``chunk.nodes`` on its inputs, so this
    mirrors how the pipeline engine hands it already-built chunks without
    forcing us to construct a full schedule.
    """
    chunk = ScheduleChunk.__new__(ScheduleChunk)
    chunk.nodes = list(nodes)
    return chunk


def _plain_node(name):
    """A non-TransformerLayer schedule node (goes to pre/post, never overlaps)."""
    return ScheduleNode(fwd_func=(lambda x: x), name=name)


def _layer_node(name):
    """A TransformerLayerNode fixture.

    TransformerLayerNode.__init__ needs a fully built TransformerLayer, which
    is a GPU-scale object. build_overlapped_nodes only uses ``isinstance`` on
    these nodes and, for the overlap pairs, reads ``.config``. So we bypass the
    heavy TransformerLayerNode.__init__ but still honestly initialise the
    ScheduleNode base and set ``.config`` -- the object really is a
    TransformerLayerNode for isinstance purposes.
    """
    node = TransformerLayerNode.__new__(TransformerLayerNode)
    ScheduleNode.__init__(node, fwd_func=None, name=name)
    node.config = types.SimpleNamespace()
    return node


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestBuildOverlappedNodes(unittest.TestCase):
    """build_overlapped_nodes partitions fwd/bwd chunks for 1F1B overlap.

    Returned tuple order (verified against the production signature):
    (forward_pre, backward_pre, overlap, forward_post, backward_post).
    """

    def test_empty_chunks(self):
        result = gpt_model.build_overlapped_nodes(
            _input_chunk([]), _input_chunk([])
        )
        self.assertEqual(len(result), 5)
        for chunk in result:
            self.assertEqual(chunk.nodes, [])

    def test_no_transformer_layers_all_go_to_pre(self):
        # With zero TransformerLayerNodes, is_pre never flips: every node lands
        # in the pre group and post/overlap stay empty.
        fa, fb = _plain_node("fa"), _plain_node("fb")
        ba = _plain_node("ba")
        (
            fwd_pre,
            bwd_pre,
            overlap,
            fwd_post,
            bwd_post,
        ) = gpt_model.build_overlapped_nodes(
            _input_chunk([fa, fb]), _input_chunk([ba])
        )
        self.assertEqual(fwd_pre.nodes, [fa, fb])
        self.assertEqual(fwd_post.nodes, [])
        # Backward is iterated in reverse then reversed back, so a single
        # pre node round-trips to itself.
        self.assertEqual(bwd_pre.nodes, [ba])
        self.assertEqual(bwd_post.nodes, [])
        self.assertEqual(overlap.nodes, [])

    def test_symmetric_overlap_partition_and_pairing(self):
        # forward = [Npre, T0, T1, Npost], backward = [B0, B1]
        # overlap count = min(2, 2) = 2.
        npre, npost = _plain_node("npre"), _plain_node("npost")
        t0, t1 = _layer_node("t0"), _layer_node("t1")
        b0, b1 = _layer_node("b0"), _layer_node("b1")
        (
            fwd_pre,
            bwd_pre,
            overlap,
            fwd_post,
            bwd_post,
        ) = gpt_model.build_overlapped_nodes(
            _input_chunk([npre, t0, t1, npost]),
            _input_chunk([b0, b1]),
        )
        # Forward: leading non-overlap -> pre, trailing non-overlap -> post.
        self.assertEqual(fwd_pre.nodes, [npre])
        self.assertEqual(fwd_post.nodes, [npost])
        # Backward here is all overlap layers, so no pre/post remains.
        self.assertEqual(bwd_pre.nodes, [])
        self.assertEqual(bwd_post.nodes, [])
        # Overlap pairs forward layers (in order) with backward layers taken
        # in reversed iteration order: [b1, b0].
        self.assertEqual(len(overlap.nodes), 2)
        self.assertIs(overlap.nodes[0].forward_node, t0)
        self.assertIs(overlap.nodes[0].backward_node, b1)
        self.assertIs(overlap.nodes[1].forward_node, t1)
        self.assertIs(overlap.nodes[1].backward_node, b0)

    def test_asymmetric_overlap_uses_min_and_posts_extras(self):
        # forward has 3 layers, backward has 1 -> overlap count = 1.
        # The first forward layer overlaps; the extra two spill into fwd_post.
        t0, t1, t2 = (
            _layer_node("t0"),
            _layer_node("t1"),
            _layer_node("t2"),
        )
        b0 = _layer_node("b0")
        (
            fwd_pre,
            bwd_pre,
            overlap,
            fwd_post,
            bwd_post,
        ) = gpt_model.build_overlapped_nodes(
            _input_chunk([t0, t1, t2]),
            _input_chunk([b0]),
        )
        self.assertEqual(fwd_pre.nodes, [])
        self.assertEqual(fwd_post.nodes, [t1, t2])
        self.assertEqual(bwd_pre.nodes, [])
        self.assertEqual(bwd_post.nodes, [])
        self.assertEqual(len(overlap.nodes), 1)
        self.assertIs(overlap.nodes[0].forward_node, t0)
        self.assertIs(overlap.nodes[0].backward_node, b0)


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestGPTModelSequentialLayers(unittest.TestCase):
    """GPTModel sequential-layer bookkeeping, exercised without full __init__.

    These methods only touch ``self._sequential_layers`` and constants, so a
    bare instance (no PipelineLayer setup, no device) is the correct scope.
    """

    def _bare_model(self):
        return gpt_model.GPTModel.__new__(gpt_model.GPTModel)

    def test_add_sequential_layer_appends_in_order(self):
        model = self._bare_model()
        layers = []
        d0, d1 = object(), object()
        model.add_sequential_layer(layers, d0, "model")
        model.add_sequential_layer(layers, d1, "model.layers.0")
        self.assertEqual(len(layers), 2)
        self.assertEqual(
            layers,
            [
                {"layer": d0, "name_prefix": "model"},
                {"layer": d1, "name_prefix": "model.layers.0"},
            ],
        )
        self.assertIs(layers[0]["layer"], d0)
        self.assertIs(layers[1]["layer"], d1)

    def test_add_sequential_layer_default_prefix_is_empty(self):
        model = self._bare_model()
        layers = []
        desc = object()
        model.add_sequential_layer(layers, desc)
        self.assertEqual(layers, [{"layer": desc, "name_prefix": ""}])

    def test_get_sequential_layers_returns_layer_objects_in_order(self):
        model = self._bare_model()
        a, b, c = object(), object(), object()
        model._sequential_layers = [
            {"layer": a, "name_prefix": "model"},
            {"layer": b, "name_prefix": "model.layers.0"},
            {"layer": c, "name_prefix": "model.lm_head"},
        ]
        result = model.get_sequential_layers()
        self.assertEqual(len(result), 3)
        self.assertIs(result[0], a)
        self.assertIs(result[1], b)
        self.assertIs(result[2], c)

    def test_get_sequential_name_prefixes_maps_index_to_prefix(self):
        model = self._bare_model()
        # Include an empty prefix: _set_pipeline_name_mapping relies on the
        # `prefixes[idx] == ""` branch, so an empty prefix must round-trip.
        model._sequential_layers = [
            {"layer": object(), "name_prefix": "model"},
            {"layer": object(), "name_prefix": ""},
            {"layer": object(), "name_prefix": "model.lm_head"},
        ]
        result = model.get_sequential_name_prefixes()
        self.assertEqual(
            result,
            {"0": "model", "1": "", "2": "model.lm_head"},
        )

    def test_get_hardware_flops_returns_fixed_value(self):
        model = self._bare_model()
        self.assertEqual(model.get_hardware_flops(), 989e3)


if __name__ == "__main__":
    unittest.main()
