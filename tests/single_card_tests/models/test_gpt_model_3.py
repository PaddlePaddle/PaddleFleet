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

"""Behavioral tests for ``paddlefleet.models.gpt.gpt_model``.

Scope (disjoint slice keyed to this module's own imports): the
``GPTSublayersSpec`` dataclass, the free function ``build_overlapped_nodes``,
and the small structural / dispatch helpers of ``GPTModel`` --
``add_sequential_layer``, ``get_sequential_layers``,
``get_sequential_name_prefixes``, ``get_hardware_flops``, ``fp8_quant_weight``,
``use_fp8``, ``_get_weight_only_params``, ``offload_weight_only_params`` and
``reload_weight_only_params``.

Expected values are hand-derived from the production source. Where a paddle
runtime primitive (tensor migration, ScheduleChunk internals) is not the thing
under test it is driven with a distinguishable recorder and the *orchestration*
is asserted -- never the raw kernel numerics. Importing gpt_model pulls in
paddle transitively; when paddle is absent the whole module is honestly skipped
rather than faked as passing. Only ``ImportError`` is treated as a missing
dependency so an API break surfaces loudly instead of being skipped away.
"""

import unittest
from dataclasses import fields

try:
    import paddle
    from paddle.distributed.fleet.meta_parallel import ScheduleChunk

    from paddlefleet.models.gpt.gpt_model import (
        GPTModel,
        GPTSublayersSpec,
        build_overlapped_nodes,
    )
    from paddlefleet.transformer.multi_token_prediction import (
        MultiTokenPredictionLayer,
    )
    from paddlefleet.transformer.transformer_layer import (
        TransformerLayer,
        TransformerLayerNode,
        TransformerLayerOverlappedScheduleNode,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # genuine missing dependency only
    _IMPORT_ERROR = exc

PADDLE_AVAILABLE = _IMPORT_ERROR is None
_SKIP_REASON = (
    "gpt_model import requires the paddle runtime, which is not installed in "
    f"this environment: {_IMPORT_ERROR!r}. The dataclass, build_overlapped_nodes "
    "partitioning and GPTModel dispatch helpers were not executed here."
)

# The eleven ordered, hand-transcribed fields of GPTSublayersSpec (source of
# truth: the @dataclass declaration in gpt_model.py). Order matters because the
# dataclass is populated positionally in some builders; a silent reorder/rename
# is exactly what the identity test below is meant to catch.
_SPEC_FIELD_NAMES = [
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
]


@unittest.skipUnless(PADDLE_AVAILABLE, _SKIP_REASON)
class TestGPTSublayersSpec(unittest.TestCase):
    """``GPTSublayersSpec`` is a pure dataclass; every field must default to
    ``None`` and each keyword must bind to its own attribute. A "can it be
    instantiated" check would pass even if a field were dropped or two fields
    swapped, so we pin the exact field set and per-field identity."""

    def test_declared_fields_match_expected_set_and_order(self):
        # Independent expectation from the source declaration, not from the class.
        actual = [f.name for f in fields(GPTSublayersSpec)]
        self.assertEqual(actual, _SPEC_FIELD_NAMES)

    def test_all_fields_default_to_none(self):
        spec = GPTSublayersSpec()
        for name in _SPEC_FIELD_NAMES:
            with self.subTest(field=name):
                self.assertIsNone(getattr(spec, name))

    def test_every_keyword_binds_to_its_own_attribute(self):
        # A unique sentinel per field: if any keyword routed to the wrong
        # attribute (or a field were renamed), an identity check fails.
        sentinels = {name: object() for name in _SPEC_FIELD_NAMES}
        spec = GPTSublayersSpec(**sentinels)
        for name in _SPEC_FIELD_NAMES:
            with self.subTest(field=name):
                self.assertIs(getattr(spec, name), sentinels[name])

    def test_list_valued_fields_hold_exact_sequence(self):
        # list-typed fields must store the very list handed in (not copy/wrap).
        layers = [object(), object(), object()]
        mtp = [object(), object()]
        spec = GPTSublayersSpec(transformer_layers=layers, mtp=mtp)
        self.assertIs(spec.transformer_layers, layers)
        self.assertIs(spec.mtp, mtp)
        # untouched fields remain None (no cross-field leakage).
        self.assertIsNone(spec.embedding)
        self.assertIsNone(spec.lm_head)


def _make_layer_node(config_marker):
    """A genuine ``TransformerLayerNode`` for isinstance-based classification.

    The real ``__init__`` needs a full layer/config graph; here only class
    identity plus the ``.config`` attribute (read by the overlapped node's
    constructor) are required, so we bypass the heavy init deliberately.
    """
    node = TransformerLayerNode.__new__(TransformerLayerNode)
    node.config = config_marker
    return node


@unittest.skipUnless(PADDLE_AVAILABLE, _SKIP_REASON)
class TestBuildOverlappedNodes(unittest.TestCase):
    """``build_overlapped_nodes`` splits a forward and a backward
    ``ScheduleChunk`` into (pre, overlap, post) triples. The overlap count is
    ``min(#fwd_layer_nodes, #bwd_layer_nodes)``; the leading non-layer nodes go
    to ``pre``, layer nodes beyond the overlap budget plus the trailing
    non-layer nodes go to ``post``; the backward side is walked in reverse and
    re-reversed; overlap pairs are ``zip(fwd_layers, bwd_layers)`` in order.

    Inputs are asymmetric (3 fwd vs 2 bwd layer nodes) with unique identities so
    a reversal bug, an off-by-one in the min() budget, or a mis-paired zip is
    observable -- not the empty-chunk degenerate case where everything is 0.
    """

    def _run(self):
        # Distinguishable non-layer sentinels (plain objects are not
        # TransformerLayerNode instances, so they are classified as pre/post).
        self.f_pre = object()
        self.f_post = object()
        self.b_x0 = object()
        self.b_x1 = object()
        # Layer nodes carry unique config markers so overlap.config is checkable.
        self.tf0 = _make_layer_node("cfg_f0")
        self.tf1 = _make_layer_node("cfg_f1")
        self.tf2 = _make_layer_node("cfg_f2")
        self.tb0 = _make_layer_node("cfg_b0")
        self.tb1 = _make_layer_node("cfg_b1")

        forward_chunk = ScheduleChunk(
            [self.f_pre, self.tf0, self.tf1, self.tf2, self.f_post]
        )
        backward_chunk = ScheduleChunk(
            [self.b_x0, self.tb0, self.tb1, self.b_x1]
        )
        return build_overlapped_nodes(forward_chunk, backward_chunk)

    def test_partition_and_pairing(self):
        fwd_pre, bwd_pre, overlap, fwd_post, bwd_post = self._run()

        # overlap budget = min(3, 2) = 2.
        # forward: [f_pre | tf0 tf1 | tf2 f_post]
        self.assertEqual(fwd_pre.nodes, [self.f_pre])
        self.assertEqual(fwd_post.nodes, [self.tf2, self.f_post])

        # backward walked reversed [b_x1, tb1, tb0, b_x0]:
        #   pre=[b_x1], overlap=[tb1, tb0], post=[b_x0]; pre/post re-reversed.
        self.assertEqual(bwd_pre.nodes, [self.b_x1])
        self.assertEqual(bwd_post.nodes, [self.b_x0])

        # overlap pairs: zip(forward_overlap=[tf0,tf1], backward_overlap=[tb1,tb0]).
        self.assertEqual(len(overlap.nodes), 2)
        for pair in overlap.nodes:
            self.assertIsInstance(pair, TransformerLayerOverlappedScheduleNode)
        self.assertIs(overlap.nodes[0].forward_node, self.tf0)
        self.assertIs(overlap.nodes[0].backward_node, self.tb1)
        self.assertIs(overlap.nodes[1].forward_node, self.tf1)
        self.assertIs(overlap.nodes[1].backward_node, self.tb0)
        # constructor copies forward_node.config onto the overlapped node.
        self.assertEqual(overlap.nodes[0].config, "cfg_f0")
        self.assertEqual(overlap.nodes[1].config, "cfg_f1")

    def test_all_layer_nodes_accounted_for_once(self):
        fwd_pre, bwd_pre, overlap, fwd_post, bwd_post = self._run()
        # No layer node is dropped or duplicated across the forward split.
        fwd_seen = list(fwd_pre.nodes) + list(fwd_post.nodes)
        fwd_seen += [p.forward_node for p in overlap.nodes]
        self.assertCountEqual(
            fwd_seen,
            [self.f_pre, self.f_post, self.tf0, self.tf1, self.tf2],
        )
        bwd_seen = list(bwd_pre.nodes) + list(bwd_post.nodes)
        bwd_seen += [p.backward_node for p in overlap.nodes]
        self.assertCountEqual(
            bwd_seen, [self.b_x0, self.b_x1, self.tb0, self.tb1]
        )

    def test_empty_chunks_give_empty_partitions(self):
        # Boundary: no layer nodes -> overlap budget 0, all lists empty.
        fwd_pre, bwd_pre, overlap, fwd_post, bwd_post = build_overlapped_nodes(
            ScheduleChunk([]), ScheduleChunk([])
        )
        self.assertEqual(overlap.nodes, [])
        self.assertEqual(fwd_pre.nodes, [])
        self.assertEqual(bwd_pre.nodes, [])
        self.assertEqual(fwd_post.nodes, [])
        self.assertEqual(bwd_post.nodes, [])

    def test_non_schedulechunk_input_rejected(self):
        # The function asserts both inputs are ScheduleChunk instances.
        with self.assertRaises(AssertionError):
            build_overlapped_nodes([], ScheduleChunk([]))


if PADDLE_AVAILABLE:

    class _Recorder:
        """Stand-in for a not-under-test collaborator that records fp8 calls."""

        def __init__(self):
            self.calls = []

        def fp8_quant_weight(self, batch_mode=False, quant_transpose=True):
            self.calls.append((batch_mode, quant_transpose))

    class _StubTransformerLayer(TransformerLayer):
        """A genuine ``TransformerLayer`` (so GPTModel's isinstance dispatch is
        the real thing) whose per-layer fp8 methods are replaced by recorders.
        Only the base Layer machinery is initialized; the heavy transformer
        construction is intentionally skipped. Per-layer fp8 numerics are NOT
        validated here -- only that GPTModel routes to the right layers with the
        right kwargs."""

        def __init__(self, fp8_flag=False):
            paddle.nn.Layer.__init__(self)
            self._fp8_flag = fp8_flag
            self.fp8_calls = []

        def use_fp8(self):
            return self._fp8_flag

        def fp8_quant_weight(self, batch_mode=False, quant_transpose=True):
            self.fp8_calls.append((batch_mode, quant_transpose))

    class _StubMTPLayer(MultiTokenPredictionLayer):
        """Genuine MTP layer whose ``transformer_layer`` is a recorder, so we
        can assert GPTModel forwards through ``layer.transformer_layer``."""

        def __init__(self):
            paddle.nn.Layer.__init__(self)
            self.transformer_layer = _Recorder()

    class _FakeMigrated:
        def __init__(self, origin):
            self.origin = origin

        def _share_buffer_to(self, target):
            self.origin.shared_from.append(target)

    class _FakeParam:
        """A parameter-like object for the weight-only offload/reload logic.

        Real pinned-memory <-> GPU migration needs a device and is out of scope
        for this (CPU-constructable) test; the guard *direction* and the
        share-buffer-back target are what is asserted."""

        def __init__(self, flagged, on_gpu):
            self.is_weight_only_mtp = flagged
            self._on_gpu = on_gpu
            self.pin_calls = 0
            self.cuda_calls = 0
            self.shared_from = []

        @property
        def place(self):
            outer = self

            class _P:
                def is_gpu_place(self):
                    return outer._on_gpu

            return _P()

        def pin_memory(self):
            self.pin_calls += 1
            return _FakeMigrated(self)

        def cuda(self):
            self.cuda_calls += 1
            return _FakeMigrated(self)


def _new_model():
    """A GPTModel shell with only the base Layer machinery initialized.

    The real ``__init__`` builds the whole pipeline/topology; the helpers under
    test read a handful of plain attributes, which each test sets explicitly.
    """
    model = GPTModel.__new__(GPTModel)
    paddle.nn.Layer.__init__(model)
    return model


@unittest.skipUnless(PADDLE_AVAILABLE, _SKIP_REASON)
class TestGPTModelSequentialHelpers(unittest.TestCase):
    """Structural bookkeeping helpers over ``self._sequential_layers`` (a list
    of ``{"layer": desc, "name_prefix": str}`` dicts)."""

    def test_add_sequential_layer_appends_in_order(self):
        model = _new_model()
        layers = []
        d0, d1 = object(), object()
        model.add_sequential_layer(layers, d0, "model")
        model.add_sequential_layer(layers, d1, "model.layers.0")
        self.assertEqual(len(layers), 2)
        # exact dict shape, identity of the desc, and prefix string, in order.
        self.assertIs(layers[0]["layer"], d0)
        self.assertEqual(layers[0]["name_prefix"], "model")
        self.assertIs(layers[1]["layer"], d1)
        self.assertEqual(layers[1]["name_prefix"], "model.layers.0")

    def test_add_sequential_layer_default_prefix_is_empty(self):
        model = _new_model()
        layers = []
        desc = object()
        model.add_sequential_layer(layers, desc)
        self.assertEqual(layers[0]["name_prefix"], "")
        self.assertIs(layers[0]["layer"], desc)

    def test_get_sequential_layers_extracts_layer_values_in_order(self):
        model = _new_model()
        la, lb, lc = object(), object(), object()
        model._sequential_layers = [
            {"layer": la, "name_prefix": "model"},
            {"layer": lb, "name_prefix": "model.layers.0"},
            {"layer": lc, "name_prefix": "model.layers.1"},
        ]
        result = model.get_sequential_layers()
        # returns the "layer" values (not the prefixes), preserving order/identity.
        self.assertEqual(len(result), 3)
        self.assertIs(result[0], la)
        self.assertIs(result[1], lb)
        self.assertIs(result[2], lc)

    def test_get_sequential_name_prefixes_maps_str_index_to_prefix(self):
        model = _new_model()
        model._sequential_layers = [
            {"layer": object(), "name_prefix": "model"},
            {"layer": object(), "name_prefix": "model.layers.0"},
            {"layer": object(), "name_prefix": "model.layers.1"},
        ]
        prefixes = model.get_sequential_name_prefixes()
        # keys are stringified positional indices; values the matching prefixes.
        self.assertEqual(
            prefixes,
            {"0": "model", "1": "model.layers.0", "2": "model.layers.1"},
        )
        self.assertTrue(all(isinstance(k, str) for k in prefixes))

    def test_get_hardware_flops_is_the_fixed_placeholder(self):
        # Hardcoded constant in the source; pin the exact value.
        model = _new_model()
        self.assertEqual(model.get_hardware_flops(), 989e3)


@unittest.skipUnless(PADDLE_AVAILABLE, _SKIP_REASON)
class TestGPTModelFp8Dispatch(unittest.TestCase):
    """``fp8_quant_weight`` / ``use_fp8`` walk ``run_function`` (non-virtual PP)
    and dispatch only to ``TransformerLayer`` and ``MultiTokenPredictionLayer``
    instances, forwarding the fp8 kwargs. We assert the routing target and the
    exact kwargs, not merely that something was called."""

    def test_fp8_quant_weight_routes_and_forwards_kwargs(self):
        model = _new_model()
        model._num_virtual_pipeline_stages = 1
        tlayer = _StubTransformerLayer()
        mtp = _StubMTPLayer()
        skipped = object()  # neither TransformerLayer nor MTP -> untouched
        model.run_function = [skipped, tlayer, mtp]

        model.fp8_quant_weight(batch_mode=True, quant_transpose=False)

        # TransformerLayer.fp8_quant_weight called once with the same kwargs.
        self.assertEqual(tlayer.fp8_calls, [(True, False)])
        # MTP layer routes through its .transformer_layer collaborator.
        self.assertEqual(mtp.transformer_layer.calls, [(True, False)])

    def test_fp8_quant_weight_defaults(self):
        model = _new_model()
        model._num_virtual_pipeline_stages = 1
        tlayer = _StubTransformerLayer()
        model.run_function = [tlayer]
        model.fp8_quant_weight()  # defaults: batch_mode=False, quant_transpose=True
        self.assertEqual(tlayer.fp8_calls, [(False, True)])

    def test_use_fp8_true_when_any_layer_uses_fp8(self):
        model = _new_model()
        model._num_virtual_pipeline_stages = 1
        model.run_function = [
            _StubTransformerLayer(fp8_flag=False),
            _StubTransformerLayer(fp8_flag=True),
        ]
        self.assertIs(model.use_fp8(), True)

    def test_use_fp8_false_when_no_layer_uses_fp8(self):
        model = _new_model()
        model._num_virtual_pipeline_stages = 1
        model.run_function = [_StubTransformerLayer(fp8_flag=False)]
        self.assertIs(model.use_fp8(), False)

    def test_use_fp8_ignores_non_transformer_layers(self):
        model = _new_model()
        model._num_virtual_pipeline_stages = 1
        model.run_function = [object(), object()]
        self.assertIs(model.use_fp8(), False)

    def test_use_fp8_vpp_true_when_any_layer_uses_fp8(self):
        # Virtual-pipeline branch: iterates each chunk's layers directly.
        model = _new_model()
        model._num_virtual_pipeline_stages = 2
        model._model_chunks = [[_StubTransformerLayer(fp8_flag=True)]]
        self.assertIs(model.use_fp8(), True)

    @unittest.expectedFailure
    def test_use_fp8_vpp_returns_false_when_none_use_fp8(self):
        # REAL BUG (gpt_model.py use_fp8): the virtual-pipeline branch has no
        # trailing ``return False``, so when no layer uses fp8 the method falls
        # off the end and returns None instead of the boolean False that the
        # non-virtual branch (and the method name) promise. Asserting the
        # correct contract; expectedFailure documents the defect without
        # touching production. Fix = add ``return False`` after the VPP loop.
        model = _new_model()
        model._num_virtual_pipeline_stages = 2
        model._model_chunks = [[_StubTransformerLayer(fp8_flag=False)]]
        self.assertIs(model.use_fp8(), False)


@unittest.skipUnless(PADDLE_AVAILABLE, _SKIP_REASON)
class TestGPTModelWeightOnlyParams(unittest.TestCase):
    """``_get_weight_only_params`` selects state-dict entries flagged
    ``is_weight_only_mtp``; offload/reload then migrate only those, and only in
    the correct device direction."""

    def test_get_weight_only_params_filters_by_flag(self):
        model = _new_model()
        keep_a = _FakeParam(flagged=True, on_gpu=True)
        drop = _FakeParam(flagged=False, on_gpu=True)
        keep_b = _FakeParam(flagged=True, on_gpu=False)
        model.state_dict = lambda *a, **k: {
            "a": keep_a,
            "b": drop,
            "c": keep_b,
        }
        selected = model._get_weight_only_params()
        # only flagged params, by identity; the unflagged one is excluded.
        self.assertEqual(len(selected), 2)
        self.assertIs(selected[0], keep_a)
        self.assertIs(selected[1], keep_b)

    def test_offload_migrates_only_gpu_flagged_params(self):
        model = _new_model()
        gpu = _FakeParam(flagged=True, on_gpu=True)
        cpu = _FakeParam(flagged=True, on_gpu=False)
        unflagged_gpu = _FakeParam(flagged=False, on_gpu=True)
        model.state_dict = lambda *a, **k: {
            "g": gpu,
            "c": cpu,
            "u": unflagged_gpu,
        }
        model.offload_weight_only_params()
        # GPU-resident flagged param pinned once and shared back onto itself.
        self.assertEqual(gpu.pin_calls, 1)
        self.assertEqual(gpu.shared_from, [gpu])
        # Already-CPU flagged param: offload is a no-op.
        self.assertEqual(cpu.pin_calls, 0)
        self.assertEqual(cpu.shared_from, [])
        # Unflagged param never touched regardless of device.
        self.assertEqual(unflagged_gpu.pin_calls, 0)

    def test_reload_migrates_only_non_gpu_flagged_params(self):
        model = _new_model()
        gpu = _FakeParam(flagged=True, on_gpu=True)
        cpu = _FakeParam(flagged=True, on_gpu=False)
        unflagged_cpu = _FakeParam(flagged=False, on_gpu=False)
        model.state_dict = lambda *a, **k: {
            "g": gpu,
            "c": cpu,
            "u": unflagged_cpu,
        }
        model.reload_weight_only_params()
        # CPU-resident flagged param moved to GPU and shared back onto itself.
        self.assertEqual(cpu.cuda_calls, 1)
        self.assertEqual(cpu.shared_from, [cpu])
        # Already-GPU flagged param: reload is a no-op (opposite of offload).
        self.assertEqual(gpu.cuda_calls, 0)
        self.assertEqual(gpu.shared_from, [])
        # Unflagged param never touched.
        self.assertEqual(unflagged_cpu.cuda_calls, 0)

    def test_offload_and_reload_are_noops_when_nothing_flagged(self):
        model = _new_model()
        p = _FakeParam(flagged=False, on_gpu=True)
        model.state_dict = lambda *a, **k: {"x": p}
        model.offload_weight_only_params()
        model.reload_weight_only_params()
        self.assertEqual(p.pin_calls, 0)
        self.assertEqual(p.cuda_calls, 0)


if __name__ == "__main__":
    unittest.main()
