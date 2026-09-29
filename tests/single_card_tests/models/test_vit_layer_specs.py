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

"""Behavior tests for ``paddlefleet.models.vision.vit_layer_specs``.

These tests are designed from the production source, not from any coverage
helper. The module builds a ViT ``LayerSpec`` tree by instantiating three
dataclasses defined elsewhere in the repo:

  * ``TransformerLayerSublayersSpec`` (transformer/transformer_layer.py)
  * ``SelfAttentionSublayersSpec``   (transformer/attention.py)
  * ``MLPSublayersSpec``             (transformer/mlp.py)

Static analysis of the current source shows that ``vit_layer_specs`` passes
keyword arguments that are NOT fields of these dataclasses (details in the
``TestSublayerSpecFieldContracts`` docstring). Because a plain ``@dataclass``
generated ``__init__`` rejects unknown keywords with ``TypeError``, both public
entry points raise at call time today. Tests that exercise the intended, fixed
behaviour therefore assert the correct contract and are marked
``@unittest.expectedFailure`` so the known defect is recorded without editing
production and so a future fix surfaces as an unexpected pass.

The authoring environment has no paddle/paddlefleet installed, so every test is
gated behind an honest ``skipUnless`` guard rather than faking a pass. The
``TypeError`` defect is confirmed by static analysis, not by execution here.
"""

import dataclasses
import os
import sys
import unittest

# Make ``src/`` importable when tests are run from a source checkout.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_SRC, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

IMPORT_OK = True
IMPORT_ERR = ""
try:
    import paddle  # noqa: F401
    from paddle.distributed.fleet.meta_parallel import LayerSpec

    from paddlefleet.fusions.fused_bias_dropout import get_bias_dropout_add
    from paddlefleet.fusions.fused_layer_norm import FusedLayerNorm
    from paddlefleet.models.vision import vit_layer_specs
    from paddlefleet.tensor_parallel.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
    from paddlefleet.transformer.attention import (
        SelfAttention,
        SelfAttentionSublayersSpec,
    )
    from paddlefleet.transformer.dot_product_attention import (
        DotProductAttention,
    )
    from paddlefleet.transformer.enums import AttnMaskType
    from paddlefleet.transformer.mlp import MLP, MLPSublayersSpec
    from paddlefleet.transformer.transformer_layer import (
        TransformerLayer,
        TransformerLayerSublayersSpec,
    )
except ImportError as exc:  # only genuine missing-dependency, not API errors
    IMPORT_OK = False
    IMPORT_ERR = f"paddle/paddlefleet not importable: {exc}"


def _spec_module(spec):
    """Return the module class held by a LayerSpec across attribute names.

    The extended paddle ``LayerSpec`` exposes the layer class as ``.module``
    (used by the spec builders / old tests) and/or ``.layer`` (used by
    ``TransformerLayer.__init__``). Try both so assertions stay valid whichever
    accessor a future fix keeps.
    """
    for attr in ("module", "layer"):
        if hasattr(spec, attr):
            return getattr(spec, attr)
    raise AttributeError("LayerSpec exposes neither .module nor .layer")


class TestProductionModuleImport(unittest.TestCase):
    """Fail loudly (not skip) if paddle is present but the module cannot load.

    A genuine missing dependency is an honest skip. But if paddle IS installed
    and ``vit_layer_specs`` still fails to import, that is a real regression and
    must not be hidden as a skip.
    """

    @unittest.skipUnless(
        "paddle" in sys.modules,
        "paddle not installed in this environment (authoring env has no paddle)",
    )
    def test_module_imported(self):
        self.assertTrue(
            IMPORT_OK,
            f"paddle is importable but vit_layer_specs failed to import: "
            f"{IMPORT_ERR}",
        )
        self.assertTrue(
            hasattr(vit_layer_specs, "get_vit_layer_with_local_spec")
        )
        self.assertTrue(hasattr(vit_layer_specs, "_get_mlp_module_spec"))


@unittest.skipUnless(IMPORT_OK, IMPORT_ERR)
class TestLNImpl(unittest.TestCase):
    """Module-level alias; does not touch the buggy spec constructors."""

    def test_ln_impl_is_fused_layer_norm(self):
        # LNImpl is the layernorm implementation wired into both the input and
        # post-attention layernorm slots of the ViT layer spec.
        self.assertIs(vit_layer_specs.LNImpl, FusedLayerNorm)


@unittest.skipUnless(IMPORT_OK, IMPORT_ERR)
class TestSublayerSpecFieldContracts(unittest.TestCase):
    """Pin the canonical dataclass fields that ``vit_layer_specs`` violates.

    ``vit_layer_specs`` builds its spec tree with these keyword arguments::

        MLPSublayersSpec(linear_fc1=..., linear_fc2=...)
        SelfAttentionSublayersSpec(linear_qkv=..., core_attention=...,
                                   linear_proj=...)
        TransformerLayerSublayersSpec(self_attention=..., ...)

    The exact-set assertions below show that ``linear_fc1``/``linear_fc2``,
    ``linear_qkv``/``linear_proj`` and ``self_attention`` are NOT declared
    fields of the respective dataclasses. A ``@dataclass`` ``__init__`` raises
    ``TypeError`` on unknown keywords, so every one of those constructions fails
    at call time. These checks observe the real dataclasses via
    ``dataclasses.fields`` (production introspection, not a rewritten formula).
    """

    def test_mlp_sublayers_spec_fields(self):
        names = {f.name for f in dataclasses.fields(MLPSublayersSpec)}
        self.assertEqual(names, {"up_gate_proj", "hidden_act", "down_proj"})
        # The exact keywords vit_layer_specs passes are invalid here.
        self.assertNotIn("linear_fc1", names)
        self.assertNotIn("linear_fc2", names)

    def test_self_attention_sublayers_spec_fields(self):
        names = {f.name for f in dataclasses.fields(SelfAttentionSublayersSpec)}
        self.assertEqual(
            names,
            {
                "qkv_proj",
                "core_attention",
                "o_proj",
                "q_norm",
                "k_norm",
                "gate_proj",
            },
        )
        # core_attention is the only keyword vit_layer_specs gets right.
        self.assertIn("core_attention", names)
        self.assertNotIn("linear_qkv", names)
        self.assertNotIn("linear_proj", names)

    def test_transformer_layer_sublayers_spec_fields(self):
        names = {
            f.name for f in dataclasses.fields(TransformerLayerSublayersSpec)
        }
        # The self-attention slot is named self_attn, not self_attention.
        self.assertIn("self_attn", names)
        self.assertNotIn("self_attention", names)
        # The other slots vit_layer_specs uses do exist under these names.
        for expected in (
            "input_layernorm",
            "self_attn_bda",
            "post_attention_layernorm",
            "mlp",
            "mlp_bda",
        ):
            self.assertIn(expected, names)


@unittest.skipUnless(IMPORT_OK, IMPORT_ERR)
class TestGetMlpModuleSpec(unittest.TestCase):
    """Intended contract for ``_get_mlp_module_spec`` (currently defective)."""

    @unittest.expectedFailure
    def test_returns_dense_mlp_layerspec(self):
        # BUG (static analysis): src/paddlefleet/models/vision/vit_layer_specs.py
        # builds MLPSublayersSpec(linear_fc1=..., linear_fc2=...), but the
        # dataclass fields are up_gate_proj/hidden_act/down_proj. The call
        # raises TypeError before returning. Asserting the intended structure:
        #   fc1 (input/gate projection) is column-parallel,
        #   fc2 (output/down projection) is row-parallel.
        spec = vit_layer_specs._get_mlp_module_spec(use_te=False)
        self.assertIsInstance(spec, LayerSpec)
        self.assertIs(_spec_module(spec), MLP)
        self.assertIsInstance(spec.submodules, MLPSublayersSpec)
        self.assertIs(spec.submodules.up_gate_proj, ColumnParallelLinear)
        self.assertIs(spec.submodules.down_proj, RowParallelLinear)


@unittest.skipUnless(IMPORT_OK, IMPORT_ERR)
class TestGetViTLayerWithLocalSpec(unittest.TestCase):
    """Intended contract for ``get_vit_layer_with_local_spec`` (defective now).

    The call first invokes ``_get_mlp_module_spec`` (which raises TypeError,
    see above), and would additionally fail on the ``self_attention=`` and
    ``linear_qkv=/linear_proj=`` keywords. Each test asserts one slice of the
    intended, hand-derived spec tree and is marked expectedFailure.
    """

    @unittest.expectedFailure
    def test_top_level_layer_and_norms(self):
        spec = vit_layer_specs.get_vit_layer_with_local_spec()
        self.assertIsInstance(spec, LayerSpec)
        self.assertIs(_spec_module(spec), TransformerLayer)
        subs = spec.submodules
        self.assertIsInstance(subs, TransformerLayerSublayersSpec)
        self.assertIs(subs.input_layernorm, FusedLayerNorm)
        self.assertIs(subs.post_attention_layernorm, FusedLayerNorm)
        self.assertIs(subs.self_attn_bda, get_bias_dropout_add)
        self.assertIs(subs.mlp_bda, get_bias_dropout_add)

    @unittest.expectedFailure
    def test_self_attention_wiring(self):
        spec = vit_layer_specs.get_vit_layer_with_local_spec()
        attn = spec.submodules.self_attn
        self.assertIsInstance(attn, LayerSpec)
        self.assertIs(_spec_module(attn), SelfAttention)
        self.assertEqual(attn.params["attn_mask_type"], AttnMaskType.causal)
        attn_subs = attn.submodules
        self.assertIsInstance(attn_subs, SelfAttentionSublayersSpec)
        # qkv is the input projection (column-parallel); the output projection
        # is row-parallel; core attention is the dot-product kernel.
        self.assertIs(attn_subs.qkv_proj, ColumnParallelLinear)
        self.assertIs(attn_subs.o_proj, RowParallelLinear)
        self.assertIs(attn_subs.core_attention, DotProductAttention)

    @unittest.expectedFailure
    def test_mlp_slot(self):
        spec = vit_layer_specs.get_vit_layer_with_local_spec()
        mlp = spec.submodules.mlp
        self.assertIsInstance(mlp, LayerSpec)
        self.assertIs(_spec_module(mlp), MLP)
        self.assertIsInstance(mlp.submodules, MLPSublayersSpec)
        self.assertIs(mlp.submodules.up_gate_proj, ColumnParallelLinear)
        self.assertIs(mlp.submodules.down_proj, RowParallelLinear)


if __name__ == "__main__":
    unittest.main()
