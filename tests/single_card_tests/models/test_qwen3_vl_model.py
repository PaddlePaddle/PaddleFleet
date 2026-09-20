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

"""Unit tests for ``paddlefleet.models.qwen3_vl.qwen3_vl_model``.

Tests are designed from the production source, exercising real entry points:

* ``Qwen3VLVisionSublayersSpec`` -- LayerSpec container dataclass; declared
  field set/order, defaults and positional wiring.
* ``Qwen3VLVsisionTransformerSubLayerSpec`` -- subclass of
  ``TransformerLayerSublayersSpec`` that adds ``deepstack_merger``; verifies the
  inherited parent fields survive and the new field is appended.
* ``Qwen3VLVisionModel.get_layer_desc_list`` -- the real desc-list assembly is
  run (embedding first, encoder body in the middle, ``merger`` last) and the
  emitted ``name_prefix`` chain plus per-slot ``layer_func`` identity are pinned.
* ``Qwen3VLVisionTransformerLayer.forward`` -- the dict-assembly logic (key pop,
  context branch, deepstack list construction) is driven with a distinguishable
  ``_forward_impl`` stub so the packed output is observed by identity, not by
  key existence alone.

Two production defects are pinned with ``@unittest.expectedFailure`` and asserted
against the *correct* contract; production is never modified. See the module
report / inline comments for ``qwen3_vl_model.py:176`` and ``:184-188``.

The module under test imports Paddle at load time. When Paddle / paddlefleet is
not importable the whole file is skipped with an honest reason rather than
reporting a pass.
"""

import dataclasses
import os
import sys
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
    import paddle
    from paddle.distributed.fleet.meta_parallel import LayerDesc, LayerSpec

    from paddlefleet.models.qwen3_vl.qwen3_vl_model import (
        Qwen3VLVisionModel,
        Qwen3VLVisionSublayersSpec,
        Qwen3VLVisionTransformerLayer,
        Qwen3VLVsisionTransformerSubLayerSpec,
    )
    from paddlefleet.transformer.transformer_layer import (
        TransformerLayerSublayersSpec,
    )

    class _DummyLayer(paddle.nn.Layer):
        """Minimal concrete Layer so LayerSpec/LayerDesc accept it."""

        def __init__(self):
            super().__init__()

except (ImportError, ModuleNotFoundError) as exc:  # honest capability probe
    # Only genuine missing-dependency import failures are treated as skip.
    # Compilation / API-change errors raise other exception types and surface
    # as real failures instead of being swallowed here.
    _IMPORT_ERROR = exc

_AVAILABLE = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle / paddlefleet not importable: {_IMPORT_ERROR!r}"
    if _IMPORT_ERROR is not None
    else ""
)

# Field names in declared order, hand-copied from the production dataclass body
# (qwen3_vl_model.py). Used as the independent oracle.
_SUBLAYERS_FIELDS = (
    "embedding",
    "head_empty_layers",
    "transformer_layers",
    "tail_empty_layers",
    "merger",
)


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestQwen3VLVisionSublayersSpec(unittest.TestCase):
    """Contract of the Qwen3VLVisionSublayersSpec LayerSpec container."""

    def test_defaults_all_none(self):
        """A no-argument instance leaves every declared field at None."""
        spec = Qwen3VLVisionSublayersSpec()
        for name in _SUBLAYERS_FIELDS:
            self.assertIsNone(getattr(spec, name), msg=f"field {name} default")

    def test_declared_field_set_and_order(self):
        """Exactly the declared fields, in order, each defaulting to None.

        A bare existence check would still pass if a field were dropped,
        renamed or given a non-None default; the full ordered tuple plus the
        per-field default pins the container's public shape.
        """
        self.assertTrue(dataclasses.is_dataclass(Qwen3VLVisionSublayersSpec))
        fields = dataclasses.fields(Qwen3VLVisionSublayersSpec)
        self.assertEqual(tuple(f.name for f in fields), _SUBLAYERS_FIELDS)
        for f in fields:
            self.assertIsNone(f.default, msg=f"declared default for {f.name}")

    def test_positional_wiring_matches_declaration_order(self):
        """Positional construction routes each argument to the right field.

        Five distinguishable sentinels are supplied by position; each named
        attribute must hold the sentinel at its declared index. Swapping any
        two field declarations would misroute a sentinel and fail here.
        """
        sentinels = [object() for _ in _SUBLAYERS_FIELDS]
        spec = Qwen3VLVisionSublayersSpec(*sentinels)
        for name, sentinel in zip(_SUBLAYERS_FIELDS, sentinels):
            self.assertIs(getattr(spec, name), sentinel, msg=f"field {name}")


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestQwen3VLVisionTransformerSubLayerSpec(unittest.TestCase):
    """Contract of the transformer-layer sublayers spec subclass."""

    def test_is_subclass_of_transformer_sublayers_spec(self):
        """The spec extends TransformerLayerSublayersSpec (not a bare class)."""
        self.assertTrue(
            issubclass(
                Qwen3VLVsisionTransformerSubLayerSpec,
                TransformerLayerSublayersSpec,
            )
        )

    def test_adds_deepstack_merger_and_keeps_parent_fields(self):
        """deepstack_merger is appended after every inherited parent field.

        A field-existence check alone would not notice a dropped parent field.
        The parent's declared field names must all remain present and precede
        the single new ``deepstack_merger`` field, which defaults to None.
        """
        parent_names = tuple(
            f.name for f in dataclasses.fields(TransformerLayerSublayersSpec)
        )
        child_names = tuple(
            f.name
            for f in dataclasses.fields(Qwen3VLVsisionTransformerSubLayerSpec)
        )
        # Parent fields are preserved, in order, at the front.
        self.assertEqual(child_names[: len(parent_names)], parent_names)
        # Exactly one new field, appended last, named deepstack_merger.
        self.assertEqual(
            child_names[len(parent_names) :], ("deepstack_merger",)
        )
        self.assertIsNone(
            Qwen3VLVsisionTransformerSubLayerSpec().deepstack_merger
        )

    def test_deepstack_merger_stores_supplied_value(self):
        """An explicit deepstack_merger is stored verbatim."""
        sentinel = object()
        spec = Qwen3VLVsisionTransformerSubLayerSpec(deepstack_merger=sentinel)
        self.assertIs(spec.deepstack_merger, sentinel)


def _new_vision_model(modal):
    """Build a Qwen3VLVisionModel shell that only needs ``modal`` set.

    ``get_layer_desc_list`` (and the inherited ``get_encoder_layer_desc_list``
    / ``add_sequential_layer``) touch only ``self.modal`` and pure list
    plumbing, so ``__new__`` plus a modal assignment is sufficient to drive the
    real method without the heavy PipelineLayer construction.
    """
    model = Qwen3VLVisionModel.__new__(Qwen3VLVisionModel)
    model.modal = modal
    return model


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestQwen3VLVisionModelGetLayerDescList(unittest.TestCase):
    """Real desc-list assembly: order, name-prefix chain and slot identity."""

    def _spec(self, n_head, n_transformer, n_tail):
        embedding = LayerSpec(_DummyLayer)
        head = [LayerSpec(_DummyLayer) for _ in range(n_head)]
        body = [LayerSpec(_DummyLayer) for _ in range(n_transformer)]
        tail = [LayerSpec(_DummyLayer) for _ in range(n_tail)]
        merger = LayerSpec(_DummyLayer)
        spec = Qwen3VLVisionSublayersSpec(
            embedding=embedding,
            head_empty_layers=head,
            transformer_layers=body,
            tail_empty_layers=tail,
            merger=merger,
        )
        return spec, embedding, head, body, tail, merger

    def test_single_transformer_layer_with_modal(self):
        """embedding, one encoder layer, merger -- prefixes carry the modal."""
        spec, emb, _, body, _, merger = self._spec(0, 1, 0)
        model = _new_vision_model("vision")
        layers = model.get_layer_desc_list(spec)

        self.assertEqual(
            [entry["name_prefix"] for entry in layers],
            ["model.vision", "model.vision.layers.0", "model.vision.merger"],
        )
        self.assertEqual([type(e["layer"]) for e in layers], [LayerDesc] * 3)
        # First slot wraps embedding, last wraps merger (not swapped).
        self.assertIs(layers[0]["layer"].layer_func, emb)
        self.assertIs(layers[1]["layer"].layer_func, body[0])
        self.assertIs(layers[-1]["layer"].layer_func, merger)

    def test_shared_index_counter_across_sections(self):
        """head/transformer/tail share one 0-based ``.layers.{i}`` counter.

        A per-section index reset would repeat ``.layers.0`` and be caught by
        the exact prefix chain below; the layer_func identity list also pins
        that no section is dropped or reordered.
        """
        spec, emb, head, body, tail, merger = self._spec(1, 2, 1)
        model = _new_vision_model("vision")
        layers = model.get_layer_desc_list(spec)

        self.assertEqual(
            [entry["name_prefix"] for entry in layers],
            [
                "model.vision",
                "model.vision.layers.0",  # head_empty[0]
                "model.vision.layers.1",  # transformer[0]
                "model.vision.layers.2",  # transformer[1]
                "model.vision.layers.3",  # tail_empty[0]
                "model.vision.merger",
            ],
        )
        self.assertEqual(
            [entry["layer"].layer_func for entry in layers],
            [emb, head[0], body[0], body[1], tail[0], merger],
        )

    def test_none_modal_uses_bare_model_prefix(self):
        """modal=None collapses the prefix to ``model`` (no trailing dot)."""
        spec, _, _, _, _, _ = self._spec(0, 1, 0)
        model = _new_vision_model(None)
        layers = model.get_layer_desc_list(spec)
        self.assertEqual(
            [entry["name_prefix"] for entry in layers],
            ["model", "model.layers.0", "model.merger"],
        )

    def test_empty_string_modal_is_falsy_and_uses_bare_prefix(self):
        """An empty-string modal is falsy, so the bare ``model`` prefix wins.

        This distinguishes the production ``if self.modal:`` truthiness test
        from an ``is not None`` check, which would have produced ``model.``.
        """
        spec, _, _, _, _, _ = self._spec(0, 1, 0)
        model = _new_vision_model("")
        layers = model.get_layer_desc_list(spec)
        self.assertEqual(layers[0]["name_prefix"], "model")
        self.assertEqual(layers[-1]["name_prefix"], "model.merger")


def _new_vision_layer(modal="vision", deepstack_merger=None):
    """Build a Qwen3VLVisionTransformerLayer shell for ``forward`` testing.

    ``forward`` reads only ``self.full_recompute``, ``self.modal`` and delegates
    the numerics to ``self._forward_impl``; ``self.deepstack_merger`` is consumed
    inside ``_forward_impl`` (stubbed here). ``__new__`` plus these attribute
    assignments drive the real ``forward`` dict-assembly without constructing the
    parent TransformerLayer (which needs a full config and device weights).
    """
    layer = Qwen3VLVisionTransformerLayer.__new__(Qwen3VLVisionTransformerLayer)
    layer.full_recompute = False
    layer.modal = modal
    layer.deepstack_merger = deepstack_merger
    return layer


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestQwen3VLVisionTransformerLayerForward(unittest.TestCase):
    """forward dict-assembly, observed by identity of the packed values."""

    def test_pops_runtime_only_keys_before_impl_and_from_output(self):
        """dynamic_inference_decode_only and position_ids are stripped.

        They are popped from dict_args up front, so ``_forward_impl`` must not
        receive them and they must be absent from the returned dict. Any other
        caller-supplied key (attention_mask here) is threaded through unchanged.
        """
        layer = _new_vision_layer()
        seen = {}
        hidden = object()
        mask = object()
        out_hidden = object()

        def impl(**kwargs):
            seen.update(kwargs)
            return (out_hidden, None)

        layer._forward_impl = impl
        result = layer.forward(
            {
                "hidden_states": hidden,
                "attention_mask": mask,
                "dynamic_inference_decode_only": True,
                "position_ids": object(),
            }
        )

        self.assertNotIn("dynamic_inference_decode_only", seen)
        self.assertNotIn("position_ids", seen)
        self.assertIs(seen["hidden_states"], hidden)
        self.assertIs(seen["attention_mask"], mask)
        self.assertNotIn("dynamic_inference_decode_only", result)
        self.assertNotIn("position_ids", result)
        self.assertIs(result["attention_mask"], mask)

    def test_two_element_no_deepstack_produces_empty_list_and_no_context(self):
        """2-tuple return, no deepstack feature: empty list, no context key.

        The list-construction and context branch are correct here independently
        of the hidden_states unpacking defect (pinned separately below): the
        None deepstack feature is not appended and context stays absent.
        """
        layer = _new_vision_layer()
        layer._forward_impl = lambda **kw: (object(), None)
        result = layer.forward({"hidden_states": object()})
        self.assertEqual(result["deepstack_feature_lists"], [])
        self.assertNotIn("context", result)

    def test_two_element_with_deepstack_appends_single_feature(self):
        """2-tuple return carrying a feature: it lands in the fresh list."""
        layer = _new_vision_layer()
        feat = object()
        layer._forward_impl = lambda **kw: (object(), feat)
        result = layer.forward({"hidden_states": object()})
        self.assertEqual(len(result["deepstack_feature_lists"]), 1)
        self.assertIs(result["deepstack_feature_lists"][0], feat)
        self.assertNotIn("context", result)

    def test_three_element_return_unpacks_hidden_and_context(self):
        """3-tuple return path unpacks hidden_states and context correctly.

        This is the branch that is *not* affected by the unpacking defect, so
        hidden_states/context identity plus the appended feature all hold.
        """
        layer = _new_vision_layer()
        hidden_out, context_out, feat = object(), object(), object()
        layer._forward_impl = lambda **kw: (hidden_out, context_out, feat)
        result = layer.forward({"hidden_states": object()})
        self.assertIs(result["hidden_states"], hidden_out)
        self.assertIs(result["context"], context_out)
        self.assertEqual(result["deepstack_feature_lists"], [feat])

    def test_three_element_with_none_feature_keeps_list_empty(self):
        """3-tuple with a None trailing feature: context kept, list empty."""
        layer = _new_vision_layer()
        hidden_out, context_out = object(), object()
        layer._forward_impl = lambda **kw: (hidden_out, context_out, None)
        result = layer.forward({"hidden_states": object()})
        self.assertIs(result["hidden_states"], hidden_out)
        self.assertIs(result["context"], context_out)
        self.assertEqual(result["deepstack_feature_lists"], [])

    @unittest.expectedFailure
    def test_two_element_hidden_states_should_be_first_element(self):
        """BUG qwen3_vl_model.py:176 -- 2-tuple hidden_states unpacking.

        For a 2-element ``_forward_impl`` return the else-branch does
        ``output, context = outputs, None``, storing the *whole tuple* as
        hidden_states instead of ``outputs[0]``. The correct contract (asserted
        here) is that hidden_states is the first element; the current code
        returns ``(hidden_out, feat)`` so this fails. Production is left intact.
        """
        layer = _new_vision_layer()
        hidden_out, feat = object(), object()
        layer._forward_impl = lambda **kw: (hidden_out, feat)
        result = layer.forward({"hidden_states": object()})
        self.assertIs(result["hidden_states"], hidden_out)

    @unittest.expectedFailure
    def test_deepstack_features_should_accumulate_across_layers(self):
        """BUG qwen3_vl_model.py:184-188 -- prior deepstack features dropped.

        The plural ``deepstack_feature_lists`` plus the per-layer append and the
        reference encoder (transformers/qwen3_vl/modeling.py:789-822, which grows
        one list across layers) imply the list accumulates as the dict threads
        through successive vision layers. But ``rst`` is rebuilt fresh every call
        and the guard ``if "deepstack_feature_lists" not in rst`` inspects that
        fresh dict (always true), so ``rst = {**dict_args, **rst}`` overwrites the
        accumulated list from prior layers. Feeding a dict that already carries
        ``[feat_prev]`` and producing ``feat_cur`` should yield both; the current
        code yields only ``[feat_cur]``, so this fails. Production is left intact.
        """
        layer = _new_vision_layer()
        feat_prev, feat_cur = object(), object()
        layer._forward_impl = lambda **kw: (object(), feat_cur)
        result = layer.forward(
            {
                "hidden_states": object(),
                "deepstack_feature_lists": [feat_prev],
            }
        )
        self.assertEqual(
            result["deepstack_feature_lists"], [feat_prev, feat_cur]
        )


if __name__ == "__main__":
    unittest.main()
