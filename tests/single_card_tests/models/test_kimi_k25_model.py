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

"""CPU-only unit tests for paddlefleet.models.kimi_k25.kimi_k25_model.

Base file of the ``test_kimi_k25_model*`` family. Keyed to the single symbol
the coverage source imports (``KimiK25VisionSublayersSpec``), it owns a small,
self-contained slice of two definitions:

* ``KimiK25VisionSublayersSpec`` -- the LayerSpec container dataclass: default
  values, declared field set/order, positional/keyword wiring integrity and
  value equality;
* ``KimiK25VisionTransformerLayer.__init__`` -- the subclass constructor,
  verified for exact argument forwarding to the ``TransformerLayer`` parent and
  for storing ``modal`` without leaking it to the parent.

This stays disjoint from the sibling ``test_kimi_k25_model_2`` slice, which
owns the heavier runtime paths (``forward``, ``_forward_impl`` and
``KimiK25VisionModel.get_layer_desc_list``); those construct the layer via
``__new__`` and never exercise the real ``__init__`` covered here.

The module under test imports Paddle at load time. When Paddle / paddlefleet is
not importable the whole file is skipped with an honest reason rather than
reporting a pass.
"""

import dataclasses
import os
import sys
import unittest
from unittest.mock import patch

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
    import paddle  # noqa: F401

    from paddlefleet.models.kimi_k25.kimi_k25_model import (
        KimiK25VisionSublayersSpec,
        KimiK25VisionTransformerLayer,
    )
    from paddlefleet.transformer.transformer_layer import TransformerLayer
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

# Field names in their declared order (kimi_k25_model.py, dataclass body).
# Hand-copied from the production declaration; used as the independent oracle.
_DECLARED_FIELDS = (
    "embedding",
    "head_empty_layers",
    "transformer_layers",
    "tail_empty_layers",
    "final_layernorm",
    "sdtpool_merger",
    "merger",
)


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestKimiK25VisionSublayersSpec(unittest.TestCase):
    """Contract of the KimiK25VisionSublayersSpec LayerSpec container."""

    def test_defaults_all_none(self):
        """A no-argument instance leaves every declared field at None."""
        spec = KimiK25VisionSublayersSpec()
        for name in _DECLARED_FIELDS:
            self.assertIsNone(getattr(spec, name), msg=f"field {name} default")

    def test_declared_field_set_and_order(self):
        """dataclass exposes exactly the declared fields, in order, default None.

        A weaker existence check would pass even if a field were dropped,
        renamed or given a non-None default; the full ordered tuple plus the
        per-field default pins the container's public shape.
        """
        self.assertTrue(dataclasses.is_dataclass(KimiK25VisionSublayersSpec))
        fields = dataclasses.fields(KimiK25VisionSublayersSpec)
        self.assertEqual(tuple(f.name for f in fields), _DECLARED_FIELDS)
        for f in fields:
            self.assertIsNone(f.default, msg=f"declared default for {f.name}")

    def test_positional_wiring_matches_declaration_order(self):
        """Positional construction routes each argument to the right field.

        Seven distinguishable sentinels are supplied by position; each named
        attribute must hold the sentinel at its declared index. Swapping any
        two field declarations would misroute a sentinel and fail here.
        """
        sentinels = [object() for _ in _DECLARED_FIELDS]
        spec = KimiK25VisionSublayersSpec(*sentinels)
        for name, sentinel in zip(_DECLARED_FIELDS, sentinels):
            self.assertIs(
                getattr(spec, name), sentinel, msg=f"positional field {name}"
            )

    def test_keyword_wiring_has_no_cross_talk(self):
        """Keyword construction assigns each field independently.

        Only two fields are populated with distinct sentinels; the rest must
        remain at their None default. This detects a field being aliased onto
        another during assignment.
        """
        emb = object()
        mrg = object()
        spec = KimiK25VisionSublayersSpec(embedding=emb, merger=mrg)
        self.assertIs(spec.embedding, emb)
        self.assertIs(spec.merger, mrg)
        for name in _DECLARED_FIELDS:
            if name in ("embedding", "merger"):
                continue
            self.assertIsNone(
                getattr(spec, name), msg=f"untouched field {name}"
            )

    def test_value_equality_is_field_wise(self):
        """Auto-generated __eq__ compares all fields; a single diff breaks it."""
        base_kwargs = {name: name for name in _DECLARED_FIELDS}
        a = KimiK25VisionSublayersSpec(**base_kwargs)
        b = KimiK25VisionSublayersSpec(**base_kwargs)
        self.assertEqual(a, b)

        changed = dict(base_kwargs)
        changed["merger"] = "different"
        c = KimiK25VisionSublayersSpec(**changed)
        self.assertNotEqual(a, c)


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestKimiK25VisionTransformerLayerInit(unittest.TestCase):
    """Constructor contract of KimiK25VisionTransformerLayer.

    The parent ``TransformerLayer.__init__`` is a heavy collaborator (it wires
    real attention / MLP sublayers) and is not the unit under test here, so it
    is replaced by a recording stub. The subclass ``__init__`` itself runs in
    full: we observe the exact kwargs forwarded to the parent and the ``modal``
    attribute the subclass sets. The stub seeds the minimal ``paddle.nn.Layer``
    bookkeeping dicts so the subclass' own ``self.modal = ...`` assignment does
    not depend on the stubbed parent.
    """

    def _instantiate_with_spy(self, **init_kwargs):
        captured = {}

        def spy_init(inner_self, **kwargs):
            captured.update(kwargs)
            object.__setattr__(inner_self, "_parameters", {})
            object.__setattr__(inner_self, "_sub_layers", {})
            object.__setattr__(inner_self, "_buffers", {})

        with patch.object(TransformerLayer, "__init__", spy_init):
            layer = KimiK25VisionTransformerLayer(**init_kwargs)
        return layer, captured

    def test_init_forwards_kwargs_and_stores_modal(self):
        """Explicit args are forwarded verbatim; modal is stored, not leaked."""
        config = object()
        sublayers_spec = object()
        pg = object()
        layer, captured = self._instantiate_with_spy(
            config=config,
            sublayers_spec=sublayers_spec,
            layer_number=3,
            hidden_dropout_prob=0.1,
            pg_collection=pg,
            modal="vision",
        )

        # modal lands on the subclass and must not reach the parent ctor.
        self.assertEqual(layer.modal, "vision")
        self.assertNotIn("modal", captured)

        # The five constructor arguments are forwarded unchanged.
        self.assertIs(captured["config"], config)
        self.assertIs(captured["sublayers_spec"], sublayers_spec)
        self.assertEqual(captured["layer_number"], 3)
        self.assertEqual(captured["hidden_dropout_prob"], 0.1)
        self.assertIs(captured["pg_collection"], pg)

        # No stray kwargs beyond the forwarded five reach the parent.
        self.assertEqual(
            set(captured),
            {
                "config",
                "sublayers_spec",
                "layer_number",
                "hidden_dropout_prob",
                "pg_collection",
            },
        )

    def test_init_applies_signature_defaults(self):
        """Omitted optional args take their declared defaults on forwarding."""
        config = object()
        sublayers_spec = object()
        layer, captured = self._instantiate_with_spy(
            config=config,
            sublayers_spec=sublayers_spec,
        )

        self.assertIsNone(layer.modal)
        self.assertIs(captured["config"], config)
        self.assertIs(captured["sublayers_spec"], sublayers_spec)
        self.assertEqual(captured["layer_number"], 1)
        self.assertIsNone(captured["hidden_dropout_prob"])
        self.assertIsNone(captured["pg_collection"])


if __name__ == "__main__":
    unittest.main()
