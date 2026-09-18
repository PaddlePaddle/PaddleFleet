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

"""Model-layer tests for ``paddlefleet.models.multimodal.llava_spec``.

Slice keyed to the builder ``decoder_model_with_local_default_spec`` and the
spec dataclasses it wires together (``TransformerLayerSublayersSpec``,
``SelfAttentionSublayersSpec``) plus ``AttnMaskType``.  All expected values are
hand-derived from the production source, not from the module under test.

Two real production bugs are documented here and asserted via
``expectedFailure`` (production is never edited):

* ``llava_spec.py:21`` imports ``get_mlp_layer_spec`` from
  ``paddlefleet.models.gpt.gpt_layer_specs``, but that module only defines
  ``get_mlp_layer_spec_for_backend``.  The bad name raises ``ImportError`` at
  module load, so ``llava_spec`` cannot be imported at all.
* Even past that, the builder wires the spec dataclasses with keyword names
  that do not exist on the installed dataclasses -- ``self_attention=`` on
  ``TransformerLayerSublayersSpec`` (field is ``self_attn``) and
  ``linear_qkv=`` / ``linear_proj=`` on ``SelfAttentionSublayersSpec`` (fields
  are ``qkv_proj`` / ``o_proj``).  Those calls would raise ``TypeError``.

Importing any ``paddlefleet`` module pulls in ``paddle`` transitively (the
package ``__init__`` imports ``parallel_state`` which imports ``paddle``), so
the whole suite skips with an honest reason when Paddle is unavailable.
Nothing here fakes a pass.
"""

import importlib
import os
import sys
import unittest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, _REPO_ROOT)
# Make the ``src/`` layout importable when the package is not installed.
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

try:
    import paddle  # noqa: F401

    _PADDLE_OK = True
    _PADDLE_ERR = ""
except ImportError as exc:  # pragma: no cover - depends on runtime deps
    _PADDLE_OK = False
    _PADDLE_ERR = repr(exc)

_SKIP_REASON = (
    "paddle is not installed in this environment (import paddle raised "
    f"{_PADDLE_ERR}); paddlefleet spec modules import paddle at load time, so "
    "the spec wiring cannot be exercised. Not verified: nothing is faked."
)


@unittest.skipUnless(_PADDLE_OK, _SKIP_REASON)
class TestGptLayerSpecsMlpHelper(unittest.TestCase):
    """The MLP helper name that ``llava_spec`` depends on."""

    def test_get_mlp_layer_spec_for_backend_exists(self):
        # The real, existing helper in gpt_layer_specs is the *_for_backend
        # variant. This positively confirms the API that llava_spec should be
        # calling (and pins the correct replacement for the buggy import).
        from paddlefleet.models.gpt import gpt_layer_specs

        self.assertTrue(
            hasattr(gpt_layer_specs, "get_mlp_layer_spec_for_backend")
        )
        self.assertTrue(
            callable(gpt_layer_specs.get_mlp_layer_spec_for_backend)
        )

    @unittest.expectedFailure
    def test_get_mlp_layer_spec_name_present(self):
        # CORRECT contract: llava_spec.py:21 imports the bare name
        # ``get_mlp_layer_spec`` from gpt_layer_specs, so a healthy codebase
        # must expose it. It does not (only ``*_for_backend`` exists), so this
        # fails today. Bug: llava_spec.py:21 / gpt_layer_specs missing symbol.
        from paddlefleet.models.gpt import gpt_layer_specs

        self.assertTrue(hasattr(gpt_layer_specs, "get_mlp_layer_spec"))


@unittest.skipUnless(_PADDLE_OK, _SKIP_REASON)
class TestLlavaSpecModule(unittest.TestCase):
    """Import- and builder-level contract of the llava_spec module."""

    @unittest.expectedFailure
    def test_module_imports_and_exposes_builder(self):
        # CORRECT contract: the module imports cleanly and exposes the public
        # builder. Fails today because llava_spec.py:21 imports a name that
        # gpt_layer_specs does not define -> ImportError at load.
        mod = importlib.import_module(
            "paddlefleet.models.multimodal.llava_spec"
        )
        self.assertTrue(hasattr(mod, "decoder_model_with_local_default_spec"))
        self.assertTrue(callable(mod.decoder_model_with_local_default_spec))

    @unittest.expectedFailure
    def test_builder_constructs_spec(self):
        # CORRECT contract: the builder returns a spec object. Fails today at
        # the import (ImportError); and even if that were fixed it would raise
        # TypeError from the self_attention=/linear_qkv=/linear_proj= keywords
        # (see TypeError contract tests below).
        mod = importlib.import_module(
            "paddlefleet.models.multimodal.llava_spec"
        )
        spec = mod.decoder_model_with_local_default_spec()
        self.assertIsNotNone(spec)


@unittest.skipUnless(_PADDLE_OK, _SKIP_REASON)
class TestSelfAttentionSublayersSpecContract(unittest.TestCase):
    """Field contract of the self-attention sublayers spec dataclass."""

    def test_accepts_installed_field_names(self):
        # Positive: the real dataclass fields are qkv_proj / core_attention /
        # o_proj; wiring them stores the exact classes (spec-wiring contract).
        from paddlefleet.tensor_parallel.layers import (
            ColumnParallelLinear,
            RowParallelLinear,
        )
        from paddlefleet.transformer.attention import (
            SelfAttentionSublayersSpec,
        )
        from paddlefleet.transformer.dot_product_attention import (
            DotProductAttention,
        )

        spec = SelfAttentionSublayersSpec(
            qkv_proj=ColumnParallelLinear,
            core_attention=DotProductAttention,
            o_proj=RowParallelLinear,
        )
        self.assertIs(spec.qkv_proj, ColumnParallelLinear)
        self.assertIs(spec.core_attention, DotProductAttention)
        self.assertIs(spec.o_proj, RowParallelLinear)
        # Unset optional norm fields default to None (source attention.py).
        self.assertIsNone(spec.q_norm)
        self.assertIsNone(spec.k_norm)

    def test_rejects_llava_spec_field_names(self):
        # llava_spec.py:59-61 constructs this dataclass with linear_qkv= and
        # linear_proj=, which are NOT fields (fields are qkv_proj / o_proj).
        # A dataclass rejects unknown kwargs with TypeError, proving that
        # construction call would raise if reached.
        from paddlefleet.transformer.attention import (
            SelfAttentionSublayersSpec,
        )

        with self.assertRaises(TypeError):
            SelfAttentionSublayersSpec(linear_qkv=object)
        with self.assertRaises(TypeError):
            SelfAttentionSublayersSpec(linear_proj=object)


@unittest.skipUnless(_PADDLE_OK, _SKIP_REASON)
class TestTransformerLayerSublayersSpecContract(unittest.TestCase):
    """Field contract of the transformer-layer sublayers spec dataclass."""

    def test_accepts_self_attn_and_layernorms(self):
        # Positive: the real field is self_attn (not self_attention); the
        # layernorm fields store the exact classes wired in.
        from paddlefleet.fusions.fused_layer_norm import FusedLayerNorm
        from paddlefleet.transformer.transformer_layer import (
            TransformerLayerSublayersSpec,
        )

        marker = object()
        spec = TransformerLayerSublayersSpec(
            input_layernorm=FusedLayerNorm,
            self_attn=marker,
            post_attention_layernorm=FusedLayerNorm,
        )
        self.assertIs(spec.input_layernorm, FusedLayerNorm)
        self.assertIs(spec.post_attention_layernorm, FusedLayerNorm)
        self.assertIs(spec.self_attn, marker)

    def test_rejects_self_attention_keyword(self):
        # llava_spec.py:55 passes self_attention=..., but the field is named
        # self_attn. The dataclass rejects the unknown kwarg with TypeError.
        from paddlefleet.transformer.transformer_layer import (
            TransformerLayerSublayersSpec,
        )

        with self.assertRaises(TypeError):
            TransformerLayerSublayersSpec(self_attention=object)

    def test_default_sublayers_are_identity(self):
        # Hand-derived from transformer_layer.py: unset sublayers default to
        # IdentityOp / IdentityFuncOp, not None. Distinguishes "wired" from
        # "left at default identity".
        from paddlefleet.transformer.identity_op import (
            IdentityFuncOp,
            IdentityOp,
        )
        from paddlefleet.transformer.transformer_layer import (
            TransformerLayerSublayersSpec,
        )

        spec = TransformerLayerSublayersSpec()
        self.assertIs(spec.input_layernorm, IdentityOp)
        self.assertIs(spec.self_attn, IdentityOp)
        self.assertIs(spec.post_attention_layernorm, IdentityOp)
        self.assertIs(spec.mlp, IdentityOp)
        self.assertIs(spec.self_attn_bda, IdentityFuncOp)
        self.assertIs(spec.mlp_bda, IdentityFuncOp)


@unittest.skipUnless(_PADDLE_OK, _SKIP_REASON)
class TestAttnMaskTypeContract(unittest.TestCase):
    """AttnMaskType values the llava decoder pins to (hand-derived)."""

    def test_causal_value_and_distinctness(self):
        # Values taken directly from transformer/enums.py. The llava decoder
        # pins attn_mask_type=AttnMaskType.causal; confirm causal is a distinct
        # member and not silently aliased to padding / no_mask.
        from paddlefleet.transformer.enums import AttnMaskType

        self.assertEqual(AttnMaskType.padding.value, 1)
        self.assertEqual(AttnMaskType.causal.value, 2)
        self.assertEqual(AttnMaskType.no_mask.value, 3)
        self.assertIsNot(AttnMaskType.causal, AttnMaskType.padding)
        self.assertIsNot(AttnMaskType.causal, AttnMaskType.no_mask)


if __name__ == "__main__":
    unittest.main()
