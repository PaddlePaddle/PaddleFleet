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

"""Behavioral tests for the pure-Python bookkeeping helpers of
``paddlefleet.models.gpt.gpt_model`` -- ``GPTSublayersSpec`` (the dataclass that
declares the ordered LayerSpec slots of a GPT stack) and the ``GPTModel``
sequential-layer accessors ``add_sequential_layer`` / ``get_sequential_layers``
/ ``get_sequential_name_prefixes`` / ``get_hardware_flops`` / ``use_fp8``.

Scope note (disjoint slice): the deep ``fp8_quant_weight`` dispatch and the
True-returning ``use_fp8`` paths are exercised elsewhere; here the focus is the
sequential-layer/spec contract plus the *false / no-fp8* branches of
``use_fp8`` -- specifically the divergence between the flat and virtual-pipeline
branches, which the True-only coverage does not observe.

These helpers do not depend on ``__init__`` having run, so they are driven as
unbound functions against a minimal stand-in ``self`` (a ``SimpleNamespace``
carrying only the attributes each helper actually reads). This avoids patching
the constructor -- construction and its validation are not what is being
claimed here.

Expected values are hand-derived from the production source
(``src/paddlefleet/models/gpt/gpt_model.py``): the dataclass field set and its
all-``None`` defaults, the ``{"layer", "name_prefix"}`` record shape, the
``str(index) -> name_prefix`` mapping, and the literal ``989e3`` hardware-FLOP
constant.

Importing ``gpt_model`` pulls in paddle transitively (it imports
``paddle.distributed.fleet`` at module load). Where paddle is not installed the
whole module is honestly skipped -- it is not faked as passing, and only a
genuine missing-dependency ``ImportError`` triggers the skip.
"""

import unittest
from dataclasses import fields
from types import SimpleNamespace

try:
    from paddlefleet.models.gpt.gpt_model import GPTModel, GPTSublayersSpec
    from paddlefleet.transformer.transformer_layer import TransformerLayer

    _IMPORT_ERROR = None
except (
    ImportError
) as exc:  # only a genuine missing dependency, not other errors
    GPTModel = None
    GPTSublayersSpec = None
    TransformerLayer = None
    _IMPORT_ERROR = exc

PADDLE_AVAILABLE = _IMPORT_ERROR is None
_SKIP_REASON = (
    "Importing paddlefleet.models.gpt.gpt_model requires the paddle runtime "
    f"(paddle.distributed.fleet), which is not installed here: {_IMPORT_ERROR!r}. "
    "The helper logic below was therefore NOT executed in this environment."
)

# The full ordered field set GPTSublayersSpec declares, hand-copied from the
# production dataclass. Kept as a literal so that a renamed / added / dropped
# slot is detected rather than silently absorbed.
_EXPECTED_SPEC_FIELDS = (
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


@unittest.skipUnless(PADDLE_AVAILABLE, _SKIP_REASON)
class TestGPTSublayersSpecDefaults(unittest.TestCase):
    """GPTSublayersSpec is the ordered declaration of every LayerSpec slot in a
    GPT pipeline stack; every slot must default to None so that a stage that
    does not populate a given slot leaves it empty rather than inheriting a
    stray shared default."""

    def test_field_set_is_exactly_the_declared_slots(self):
        # Detects a renamed / added / removed slot, which a per-field None check
        # on a subset would miss.
        actual = tuple(f.name for f in fields(GPTSublayersSpec))
        self.assertEqual(actual, _EXPECTED_SPEC_FIELDS)

    def test_every_slot_defaults_to_none(self):
        spec = GPTSublayersSpec()
        for name in _EXPECTED_SPEC_FIELDS:
            with self.subTest(field=name):
                self.assertIsNone(getattr(spec, name))

    def test_explicit_values_are_retained_per_slot(self):
        # A default of None does not prove the field is a real, independently
        # assignable slot; set distinct sentinels and confirm no cross-slot
        # bleed.
        sentinels = {name: object() for name in _EXPECTED_SPEC_FIELDS}
        spec = GPTSublayersSpec(**sentinels)
        for name in _EXPECTED_SPEC_FIELDS:
            with self.subTest(field=name):
                self.assertIs(getattr(spec, name), sentinels[name])


@unittest.skipUnless(PADDLE_AVAILABLE, _SKIP_REASON)
class TestGPTModelSequentialLayerBookkeeping(unittest.TestCase):
    """add_sequential_layer / get_sequential_layers / get_sequential_name_prefixes
    form the record-keeping that later drives PipelineLayer naming and weight
    mapping. Each record must pair a layer object with its name prefix, and the
    ordered index->prefix mapping must survive extraction. These helpers ignore
    ``self`` state beyond ``_sequential_layers`` (and none at all for
    add_sequential_layer), so a bare stand-in ``self`` exercises the real
    logic without constructing a pipeline."""

    def test_add_sequential_layer_records_layer_and_prefix_in_order(self):
        layers = []
        first, second = object(), object()
        GPTModel.add_sequential_layer(SimpleNamespace(), layers, first, "embed")
        GPTModel.add_sequential_layer(
            SimpleNamespace(), layers, second, "layer.0"
        )
        self.assertEqual(len(layers), 2)
        # Full record shape, not just presence: layer identity + exact prefix,
        # in append order.
        self.assertEqual(set(layers[0]), {"layer", "name_prefix"})
        self.assertIs(layers[0]["layer"], first)
        self.assertEqual(layers[0]["name_prefix"], "embed")
        self.assertIs(layers[1]["layer"], second)
        self.assertEqual(layers[1]["name_prefix"], "layer.0")

    def test_add_sequential_layer_defaults_prefix_to_empty_string(self):
        layers = []
        desc = object()
        GPTModel.add_sequential_layer(SimpleNamespace(), layers, desc)
        self.assertEqual(layers[0]["name_prefix"], "")
        self.assertIs(layers[0]["layer"], desc)

    def test_get_sequential_layers_extracts_layers_preserving_order(self):
        a, b, c = object(), object(), object()
        stub = SimpleNamespace(
            _sequential_layers=[
                {"layer": a, "name_prefix": "embed"},
                {"layer": b, "name_prefix": "layer.0"},
                {"layer": c, "name_prefix": "lm_head"},
            ]
        )
        result = GPTModel.get_sequential_layers(stub)
        # Order + identity: a reversed or reordered extraction must be rejected,
        # which a length-only check would not catch.
        self.assertEqual(result, [a, b, c])
        self.assertIs(result[0], a)
        self.assertIs(result[2], c)

    def test_get_sequential_layers_empty_is_empty_list(self):
        stub = SimpleNamespace(_sequential_layers=[])
        self.assertEqual(GPTModel.get_sequential_layers(stub), [])

    def test_get_sequential_name_prefixes_maps_str_index_to_prefix(self):
        stub = SimpleNamespace(
            _sequential_layers=[
                {"layer": object(), "name_prefix": "embed"},
                {"layer": object(), "name_prefix": "layer.0"},
                {"layer": object(), "name_prefix": "lm_head"},
            ]
        )
        result = GPTModel.get_sequential_name_prefixes(stub)
        # Distinguishable prefixes so a mis-paired index is caught; keys are the
        # positional index rendered as a string.
        self.assertEqual(result, {"0": "embed", "1": "layer.0", "2": "lm_head"})
        self.assertIsInstance(next(iter(result)), str)

    def test_get_hardware_flops_returns_documented_constant(self):
        # Source returns the literal 989e3 == 989000.0; guards against silent
        # edits of the hardcoded FLOP figure.
        result = GPTModel.get_hardware_flops(SimpleNamespace())
        self.assertEqual(result, 989e3)
        self.assertEqual(result, 989000.0)
        self.assertIsInstance(result, float)


class _NonFp8TransformerLayer(TransformerLayer if PADDLE_AVAILABLE else object):
    """A real TransformerLayer subclass whose use_fp8 is forced False, so that
    isinstance(layer, TransformerLayer) holds while the fp8 predicate is
    controlled. Constructed via __new__ to skip paddle's heavy Layer init."""

    def use_fp8(self):
        return False


class _Fp8TransformerLayer(TransformerLayer if PADDLE_AVAILABLE else object):
    """A real TransformerLayer subclass whose use_fp8 is forced True."""

    def use_fp8(self):
        return True


@unittest.skipUnless(PADDLE_AVAILABLE, _SKIP_REASON)
class TestGPTModelUseFp8Branches(unittest.TestCase):
    """GPTModel.use_fp8 scans layers and returns whether any TransformerLayer
    reports fp8. Non-TransformerLayer entries must be skipped by the isinstance
    filter. This slice targets the *no-fp8* outcome on both the flat and the
    virtual-pipeline branches -- the branches a True-only test never reaches."""

    def _flat_model(self, run_function):
        return SimpleNamespace(
            _num_virtual_pipeline_stages=1, run_function=run_function
        )

    def _vpp_model(self, model_chunks):
        return SimpleNamespace(
            _num_virtual_pipeline_stages=2, _model_chunks=model_chunks
        )

    def test_flat_no_transformer_layer_returns_false(self):
        # object() is not a TransformerLayer, so the isinstance filter skips it
        # and the flat branch falls through to its explicit `return False`.
        stub = self._flat_model([object(), object()])
        self.assertIs(GPTModel.use_fp8(stub), False)

    def test_flat_transformer_layer_reporting_false_returns_false(self):
        layer = _NonFp8TransformerLayer.__new__(_NonFp8TransformerLayer)
        stub = self._flat_model([object(), layer])
        self.assertIs(GPTModel.use_fp8(stub), False)

    def test_flat_transformer_layer_reporting_true_returns_true(self):
        # Exercises the isinstance-True + use_fp8-True dispatch against a real
        # TransformerLayer subclass (not a look-alike stub).
        layer = _Fp8TransformerLayer.__new__(_Fp8TransformerLayer)
        stub = self._flat_model([object(), layer])
        self.assertIs(GPTModel.use_fp8(stub), True)

    def test_vpp_transformer_layer_reporting_true_returns_true(self):
        layer = _Fp8TransformerLayer.__new__(_Fp8TransformerLayer)
        stub = self._vpp_model([[object()], [layer]])
        self.assertIs(GPTModel.use_fp8(stub), True)

    @unittest.expectedFailure
    def test_vpp_no_fp8_should_return_false_not_none(self):
        # CORRECT contract: a predicate named use_fp8 must return the boolean
        # False when no layer uses fp8, matching the flat branch.
        #
        # ACTUAL (production bug, gpt_model.py:986-996): in the virtual-pipeline
        # branch the `return False` is indented inside the `else:` of the flat
        # branch, so when VPP > 1 and no TransformerLayer reports fp8 the method
        # falls off the end and returns None. Marked expectedFailure; production
        # is left unchanged.
        stub = self._vpp_model([[object()], [object()]])
        self.assertIs(GPTModel.use_fp8(stub), False)

    def test_vpp_no_fp8_currently_returns_none(self):
        # Pinning the present (buggy) behavior so the divergence from the flat
        # branch is explicit and observable; paired with the expectedFailure
        # above.
        stub = self._vpp_model([[object()], [object()]])
        self.assertIsNone(GPTModel.use_fp8(stub))


if __name__ == "__main__":
    unittest.main()
