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

"""Behaviour tests for the descriptor helper methods of
``paddlefleet.transformer.transformer_encoder.TransformerEncoder``.

The methods exercised here (``get_layer_desc_list``, ``add_sequential_layer``,
``get_sequential_layers``, ``get_sequential_name_prefixes`` and
``get_shardlayer_prefix``) are pure Python orchestration over descriptor
lists and are safe to run on CPU without an accelerator. They only read a
handful of instance attributes (``modal``, ``_sequential_layers``, ``layers``,
``_stage_id``) and the ``get_stage_from_index`` collaborator; they do not
depend on the expensive ``PipelineLayer.__init__`` (which needs a Fleet
hybrid-communicate group). The ``_HelperEncoder`` subclass therefore skips
that base initialiser but keeps every method under test as the *real*
production implementation, so a regression in prefix construction, layer
ordering, ``LayerDesc`` wrapping or the shared-layer stage guard is observed.

Expected values are derived by hand from the production source, not copied
from any coverage fixture.
"""

import unittest

try:
    from paddle.distributed.fleet.meta_parallel import (
        LayerDesc,
        SharedLayerDesc,
    )

    from paddlefleet.transformer import transformer_encoder

    class _HelperEncoder(transformer_encoder.TransformerEncoder):
        """Skip ``PipelineLayer.__init__`` while keeping the real helpers.

        Only the state the descriptor helpers actually read is populated;
        ``get_stage_from_index`` is a controllable (index-dependent)
        collaborator so the stage guard in ``get_shardlayer_prefix`` can be
        observed for both the matching and the mismatching stage.
        """

        def __init__(self, modal=None, stage_id=0, stage_by_index=None):
            self.modal = modal
            self._sequential_layers = []
            self._pipeline_name_mapping = None
            self.layers = []
            self._stage_id = stage_id
            self._stage_by_index = dict(stage_by_index or {})

        def get_stage_from_index(self, idx):
            return self._stage_by_index.get(idx, 0)

    IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    LayerDesc = None
    SharedLayerDesc = None
    transformer_encoder = None
    _HelperEncoder = None
    IMPORT_ERROR = exc


# Distinguishable marker classes. The helpers only store/wrap these classes;
# they are never instantiated, so plain classes (no paddle) are sufficient.
class _Emb:
    pass


class _Head:
    pass


class _T0:
    pass


class _T1:
    pass


class _Tail:
    pass


class _Norm:
    pass


class _Spec:
    """Minimal ``sublayers_spec`` shape consumed by the base helpers."""

    def __init__(
        self,
        embedding,
        head_empty_layers,
        transformer_layers,
        tail_empty_layers,
        layer_norm,
    ):
        self.embedding = embedding
        self.head_empty_layers = head_empty_layers
        self.transformer_layers = transformer_layers
        self.tail_empty_layers = tail_empty_layers
        self.layer_norm = layer_norm


@unittest.skipUnless(
    IMPORT_ERROR is None,
    f"paddle/paddlefleet import unavailable: {IMPORT_ERROR!r}",
)
class TestTransformerEncoderDescriptorHelpers(unittest.TestCase):
    def _spec(self):
        # Two distinct transformer layers so ordering / index assignment is
        # discriminated beyond a single-element degenerate case.
        return _Spec(
            embedding=_Emb,
            head_empty_layers=[_Head],
            transformer_layers=[_T0, _T1],
            tail_empty_layers=[_Tail],
            layer_norm=_Norm,
        )

    def test_get_layer_desc_list_model_prefix_order_and_wrapping(self):
        model = _HelperEncoder(modal=None)

        layers = model.get_layer_desc_list(self._spec())

        # Hand-derived: embedding, head(0), transformer(1), transformer(2),
        # tail(3) each carry ".layers.<i>"; embedding and the trailing
        # layer_norm both use the bare "model" prefix.
        self.assertEqual(
            [entry["name_prefix"] for entry in layers],
            [
                "model",
                "model.layers.0",
                "model.layers.1",
                "model.layers.2",
                "model.layers.3",
                "model",
            ],
        )

        # First five slots are wrapped in LayerDesc and preserve the exact
        # class in the exact slot (catches swaps between emb/head/tf/tail).
        expected_funcs = [_Emb, _Head, _T0, _T1, _Tail]
        for entry, expected_cls in zip(layers[:5], expected_funcs):
            self.assertIsInstance(entry["layer"], LayerDesc)
            self.assertIs(entry["layer"].layer_func, expected_cls)

        # The trailing layer_norm is passed through raw (NOT wrapped in a
        # LayerDesc) by the production code -- an asymmetry versus embedding.
        self.assertIs(layers[5]["layer"], _Norm)
        self.assertNotIsInstance(layers[5]["layer"], LayerDesc)

    def test_get_layer_desc_list_modal_prefix(self):
        model = _HelperEncoder(modal="vision")

        layers = model.get_layer_desc_list(self._spec())

        self.assertEqual(
            [entry["name_prefix"] for entry in layers],
            [
                "model.vision",
                "model.vision.layers.0",
                "model.vision.layers.1",
                "model.vision.layers.2",
                "model.vision.layers.3",
                "model.vision",
            ],
        )

    def test_sequential_name_prefixes_index_mapping(self):
        model = _HelperEncoder(modal=None)
        layers = model.get_layer_desc_list(self._spec())
        model._sequential_layers = layers

        self.assertEqual(
            model.get_sequential_name_prefixes(),
            {
                "0": "model",
                "1": "model.layers.0",
                "2": "model.layers.1",
                "3": "model.layers.2",
                "4": "model.layers.3",
                "5": "model",
            },
        )

    def test_add_sequential_layer_preserves_identity_and_order(self):
        model = _HelperEncoder(modal=None)
        buf = []
        first, second, third = object(), object(), object()

        model.add_sequential_layer(buf, first, "p.first")
        model.add_sequential_layer(buf, second, "p.second")
        model.add_sequential_layer(buf, third, "p.third")
        model._sequential_layers = buf

        got = model.get_sequential_layers()
        self.assertEqual(len(got), 3)
        self.assertIs(got[0], first)
        self.assertIs(got[1], second)
        self.assertIs(got[2], third)
        self.assertEqual(
            model.get_sequential_name_prefixes(),
            {"0": "p.first", "1": "p.second", "2": "p.third"},
        )

    def _build_shared_model(self):
        shared_embed = SharedLayerDesc(
            "embed", _Emb, shared_weight_attr="embedding_weight"
        )
        shared_head = SharedLayerDesc(
            "lm_head", _Head, shared_weight_attr="lm_head_weight"
        )
        # idx 0 -> stage 0 (== current stage), idx 2 -> stage 1 (mismatch).
        model = _HelperEncoder(stage_id=0, stage_by_index={0: 0, 2: 1})
        model.layers = [shared_embed, LayerDesc(_T0), shared_head]
        model._sequential_layers = [
            {"layer": shared_embed, "name_prefix": "model.embed"},
            {"layer": model.layers[1], "name_prefix": "model.layers.0"},
            {"layer": shared_head, "name_prefix": "model.lm_head"},
        ]
        return model

    def test_get_shardlayer_prefix_returns_matching_stage_prefix(self):
        model = self._build_shared_model()

        # "embed" lives at index 0 whose stage matches _stage_id -> its own
        # prefix must be returned (not "model.lm_head").
        self.assertEqual(
            model.get_shardlayer_prefix(["shared_layers", "embed", "weight"]),
            "model.embed",
        )

    def test_get_shardlayer_prefix_unknown_key_raises_assertion(self):
        model = self._build_shared_model()

        with self.assertRaises(AssertionError):
            model.get_shardlayer_prefix(["shared_layers", "missing", "weight"])

    def test_get_shardlayer_prefix_wrong_stage_raises_value_error(self):
        model = self._build_shared_model()

        # "lm_head" is a valid shared key (passes the membership assertion)
        # but resolves to index 2 -> stage 1 != current stage 0.
        with self.assertRaises(ValueError):
            model.get_shardlayer_prefix(["shared_layers", "lm_head", "weight"])


if __name__ == "__main__":
    unittest.main()
