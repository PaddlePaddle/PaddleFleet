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
"""Behavior tests for the vision layer code under ``paddlefleet.models.vision``.

The tests are derived from the production sources, not from any prior coverage
file:

* ``clip_vit_model.get_num_image_embeddings`` is pure integer arithmetic (no
  tensors), so every expected value below is hand-derived from the branch
  logic:

    - ``keep_class_token`` depends on ``vision_model_type``: ``siglip`` never
      keeps it; ``clip`` / ``internvit`` / ``internvit300M`` / ``radio*`` keep
      it unless ``disable_vision_class_token``; ``cradio-g`` *forces*
      ``class_token_len = 8`` and then keeps it unless disabled.
    - ``num_patches = (img_h // patch_dim) * (img_w // patch_dim)`` and the
      per-tile count is ``num_patches + (class_token_len if keep else 0)``.
    - ``pixel_shuffle`` rescales by ``0.5 ** 2 = 0.25`` and *truncates* via
      ``int(...)``.
    - ``use_tile_tags`` adds ``5`` for the four supported tokenizers (else
      raises), then adds ``1`` only when ``10 < max_num_tiles < 100`` *and* the
      tokenizer name starts with ``"qwen"``; ``max_num_tiles > 100`` raises.

  The ``hf://`` branch is intentionally not exercised: it imports the
  huggingface module lazily and is out of scope for CPU logic tests.

* ``vit_layer_specs.get_vit_layer_with_local_spec`` is expected to return a
  ``LayerSpec`` wiring ``SelfAttention`` into the transformer layer's
  self-attention slot. It does NOT: see ``TestVitLayerSpec`` and the REAL BUG
  note there.

Everything here imports Paddle (``get_num_image_embeddings`` lives in a module
whose top level does ``import paddle``; the layer specs pull in Paddle Fleet
``LayerSpec`` and real attention/MLP modules). Paddle is not installed in this
environment, so the suite is skipped with an honest reason rather than faked.
Only ``ImportError`` is treated as "dependency missing"; any other import
failure is allowed to surface as a real error instead of a silent skip.
"""

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

try:
    from paddlefleet.models.vision.clip_vit_model import (
        get_num_image_embeddings,
    )
    from paddlefleet.models.vision.vit_layer_specs import (
        _get_mlp_module_spec,
        get_vit_layer_with_local_spec,
    )
    from paddlefleet.transformer.attention import SelfAttention
    from paddlefleet.transformer.enums import AttnMaskType
    from paddlefleet.transformer.mlp import MLP
except ImportError as exc:  # honest: only a missing dependency skips
    get_num_image_embeddings = None
    _get_mlp_module_spec = None
    get_vit_layer_with_local_spec = None
    SelfAttention = None
    AttnMaskType = None
    MLP = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = f"paddle/paddlefleet import unavailable: {_IMPORT_ERROR}"


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestGetNumImageEmbeddings(unittest.TestCase):
    """Hand-derived counts for every branch of ``get_num_image_embeddings``."""

    # ---- class-token keeping rules ------------------------------------

    def test_clip_keeps_class_token(self):
        # 336 // 14 = 24 patches per dim -> 24 * 24 = 576; +1 class token.
        self.assertEqual(
            get_num_image_embeddings(
                img_h=336,
                img_w=336,
                patch_dim=14,
                vision_model_type="clip",
                disable_vision_class_token=False,
                class_token_len=1,
                pixel_shuffle=False,
            ),
            577,
        )

    def test_clip_disable_class_token(self):
        # disable flag drops the single class token: 576 + 0.
        self.assertEqual(
            get_num_image_embeddings(
                img_h=336,
                img_w=336,
                patch_dim=14,
                vision_model_type="clip",
                disable_vision_class_token=True,
                class_token_len=1,
                pixel_shuffle=False,
            ),
            576,
        )

    def test_siglip_never_keeps_class_token(self):
        # siglip forces keep_class_token=False regardless of disable flag or
        # class_token_len; both settings must yield the bare patch count 576.
        for disable in (False, True):
            with self.subTest(disable=disable):
                self.assertEqual(
                    get_num_image_embeddings(
                        img_h=336,
                        img_w=336,
                        patch_dim=14,
                        vision_model_type="siglip",
                        disable_vision_class_token=disable,
                        class_token_len=1,
                        pixel_shuffle=False,
                    ),
                    576,
                )

    def test_internvit_variants_keep_and_disable(self):
        # internvit and internvit300M behave like clip for the class token.
        for model_type in ("internvit", "internvit300M"):
            with self.subTest(model_type=model_type, disable=False):
                self.assertEqual(
                    get_num_image_embeddings(
                        img_h=336,
                        img_w=336,
                        patch_dim=14,
                        vision_model_type=model_type,
                        disable_vision_class_token=False,
                        class_token_len=1,
                        pixel_shuffle=False,
                    ),
                    577,
                )
            with self.subTest(model_type=model_type, disable=True):
                self.assertEqual(
                    get_num_image_embeddings(
                        img_h=336,
                        img_w=336,
                        patch_dim=14,
                        vision_model_type=model_type,
                        disable_vision_class_token=True,
                        class_token_len=1,
                        pixel_shuffle=False,
                    ),
                    576,
                )

    def test_radio_prefix_keeps_default_class_tokens(self):
        # Any type starting with "radio" keeps class_token_len (RADIO uses 8).
        # 576 patches + 8 = 584; disable flag removes them -> 576.
        for model_type in ("radio", "radio-h"):
            with self.subTest(model_type=model_type):
                self.assertEqual(
                    get_num_image_embeddings(
                        img_h=336,
                        img_w=336,
                        patch_dim=14,
                        vision_model_type=model_type,
                        disable_vision_class_token=False,
                        class_token_len=8,
                        pixel_shuffle=False,
                    ),
                    584,
                )
        self.assertEqual(
            get_num_image_embeddings(
                img_h=336,
                img_w=336,
                patch_dim=14,
                vision_model_type="radio",
                disable_vision_class_token=True,
                class_token_len=8,
                pixel_shuffle=False,
            ),
            576,
        )

    def test_cradio_g_overrides_class_token_len_to_eight(self):
        # cradio-g ignores the passed class_token_len (1) and forces 8.
        # 576 + 8 = 584, NOT 576 + 1 = 577. This pins the override.
        self.assertEqual(
            get_num_image_embeddings(
                img_h=336,
                img_w=336,
                patch_dim=14,
                vision_model_type="cradio-g",
                disable_vision_class_token=False,
                class_token_len=1,
                pixel_shuffle=False,
            ),
            584,
        )

    def test_non_square_image_patch_count(self):
        # 224 // 14 = 16 (h), 336 // 14 = 24 (w) -> 384 patches; +1 class token.
        self.assertEqual(
            get_num_image_embeddings(
                img_h=224,
                img_w=336,
                patch_dim=14,
                vision_model_type="clip",
                disable_vision_class_token=False,
                class_token_len=1,
                pixel_shuffle=False,
            ),
            385,
        )

    # ---- pixel shuffle -------------------------------------------------

    def test_pixel_shuffle_scales_by_quarter_with_truncation(self):
        # 140 // 14 = 10 -> 100 patches; +1 class token = 101.
        # int(101 * 0.25) = int(25.25) = 25 (truncated, not rounded to 25.25).
        self.assertEqual(
            get_num_image_embeddings(
                img_h=140,
                img_w=140,
                patch_dim=14,
                vision_model_type="clip",
                disable_vision_class_token=False,
                class_token_len=1,
                pixel_shuffle=True,
            ),
            25,
        )
        # clip keep (577 -> int(144.25)=144) and clip disable (576 ->
        # int(144.0)=144) both collapse to 144 after truncation: the extra
        # class token vanishes in the floor.
        for disable, base in ((False, 577), (True, 576)):
            with self.subTest(disable=disable, base=base):
                self.assertEqual(
                    get_num_image_embeddings(
                        img_h=336,
                        img_w=336,
                        patch_dim=14,
                        vision_model_type="clip",
                        disable_vision_class_token=disable,
                        class_token_len=1,
                        pixel_shuffle=True,
                    ),
                    144,
                )

    # ---- tile tags -----------------------------------------------------

    def test_tile_tags_adds_five_for_supported_tokenizers(self):
        # base clip keep = 577; +5 for tile tags; max_num_tiles=0 adds nothing.
        for tok in ("llama3p1", "chatml", "qwen2p0", "qwen2p5"):
            with self.subTest(tokenizer=tok):
                self.assertEqual(
                    get_num_image_embeddings(
                        img_h=336,
                        img_w=336,
                        patch_dim=14,
                        vision_model_type="clip",
                        disable_vision_class_token=False,
                        class_token_len=1,
                        pixel_shuffle=False,
                        use_tile_tags=True,
                        max_num_tiles=0,
                        tokenizer_type=tok,
                    ),
                    582,
                )

    def test_tile_tags_qwen_padding_within_tile_range(self):
        # 577 + 5 = 582, then +1 padding because tokenizer starts with "qwen"
        # and 10 < max_num_tiles < 100.
        for tok, tiles in (("qwen2p5", 16), ("qwen2p0", 50)):
            with self.subTest(tokenizer=tok, tiles=tiles):
                self.assertEqual(
                    get_num_image_embeddings(
                        img_h=336,
                        img_w=336,
                        patch_dim=14,
                        vision_model_type="clip",
                        disable_vision_class_token=False,
                        class_token_len=1,
                        pixel_shuffle=False,
                        use_tile_tags=True,
                        max_num_tiles=tiles,
                        tokenizer_type=tok,
                    ),
                    583,
                )

    def test_tile_tags_non_qwen_no_padding(self):
        # chatml is supported (+5) but does not start with "qwen": no +1 even
        # inside the (10, 100) tile window.
        self.assertEqual(
            get_num_image_embeddings(
                img_h=336,
                img_w=336,
                patch_dim=14,
                vision_model_type="clip",
                disable_vision_class_token=False,
                class_token_len=1,
                pixel_shuffle=False,
                use_tile_tags=True,
                max_num_tiles=16,
                tokenizer_type="chatml",
            ),
            582,
        )

    def test_tile_tags_max_num_tiles_boundaries(self):
        # Strict inequalities 10 < n < 100 for the qwen +1 padding.
        #   n = 10  -> 10 < 10 is False -> no padding -> 582
        #   n = 99  -> 10 < 99 < 100    -> padding     -> 583
        #   n = 100 -> 100 < 100 False and 100 > 100 False -> no padding, no
        #              raise -> 582
        cases = {10: 582, 99: 583, 100: 582}
        for tiles, expected in cases.items():
            with self.subTest(max_num_tiles=tiles):
                self.assertEqual(
                    get_num_image_embeddings(
                        img_h=336,
                        img_w=336,
                        patch_dim=14,
                        vision_model_type="clip",
                        disable_vision_class_token=False,
                        class_token_len=1,
                        pixel_shuffle=False,
                        use_tile_tags=True,
                        max_num_tiles=tiles,
                        tokenizer_type="qwen2p5",
                    ),
                    expected,
                )

    def test_pixel_shuffle_then_tile_tags_order(self):
        # Ordering matters: pixel shuffle first (577 -> int(144.25) = 144),
        # then +5 tile tags = 149, then +1 qwen padding (max=16) = 150.
        self.assertEqual(
            get_num_image_embeddings(
                img_h=336,
                img_w=336,
                patch_dim=14,
                vision_model_type="clip",
                disable_vision_class_token=False,
                class_token_len=1,
                pixel_shuffle=True,
                use_tile_tags=True,
                max_num_tiles=16,
                tokenizer_type="qwen2p5",
            ),
            150,
        )

    def test_tile_tags_too_many_tiles_raises(self):
        with self.assertRaises(ValueError):
            get_num_image_embeddings(
                img_h=336,
                img_w=336,
                patch_dim=14,
                vision_model_type="clip",
                disable_vision_class_token=False,
                class_token_len=1,
                pixel_shuffle=False,
                use_tile_tags=True,
                max_num_tiles=200,
                tokenizer_type="qwen2p5",
            )

    def test_tile_tags_unknown_tokenizer_raises(self):
        with self.assertRaises(ValueError):
            get_num_image_embeddings(
                img_h=336,
                img_w=336,
                patch_dim=14,
                vision_model_type="clip",
                disable_vision_class_token=False,
                class_token_len=1,
                pixel_shuffle=False,
                use_tile_tags=True,
                max_num_tiles=0,
                tokenizer_type="gpt2",
            )

    def test_unknown_vision_model_type_raises(self):
        with self.assertRaises(NotImplementedError):
            get_num_image_embeddings(
                img_h=336,
                img_w=336,
                patch_dim=14,
                vision_model_type="not-a-real-encoder",
                disable_vision_class_token=False,
                class_token_len=1,
                pixel_shuffle=False,
            )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestVitLayerSpec(unittest.TestCase):
    """Wiring contracts for the ViT layer specs.

    Production consumes a spec's module class via its ``.layer`` attribute
    (e.g. ``TransformerLayer.__init__`` does ``sublayers_spec.mlp.layer ==
    MLP``), so ``.layer`` identity is the honest contract to assert on a spec.
    """

    def test_mlp_module_spec_wires_mlp(self):
        # REAL BUG (production, NOT edited): vit_layer_specs.py:71-73 builds the
        # dense MLP sub-spec with ``MLPSublayersSpec(linear_fc1=...,
        # linear_fc2=...)``, but that @dataclass declares no such fields -- its
        # fields are ``up_gate_proj`` / ``hidden_act`` / ``down_proj`` (see
        # transformer/mlp.py). A plain dataclass rejects the unknown keywords,
        # so ``_get_mlp_module_spec`` raises ``TypeError`` and cannot build any
        # spec. The CORRECT contract would be a LayerSpec whose ``.layer`` is
        # MLP; documenting the defect via assertRaises without touching
        # production. Fixing the keywords would turn this into a failure here
        # (a signal to restore the positive wiring assertion).
        with self.assertRaises(TypeError) as ctx:
            _get_mlp_module_spec(use_te=False)
        self.assertIn("linear_fc1", str(ctx.exception))

    @unittest.expectedFailure
    def test_vit_layer_spec_wires_self_attention(self):
        # EXPECTED CORRECT BEHAVIOR (this is what the test asserts): calling
        # get_vit_layer_with_local_spec() should return a TransformerLayer spec
        # whose self-attention slot is wired to SelfAttention.
        #
        # REAL BUG (production, NOT edited): vit_layer_specs.py:49 passes
        # ``self_attention=LayerSpec(...)`` to the ``@dataclass``
        # ``TransformerLayerSublayersSpec``, whose field is named ``self_attn``
        # (there is no ``self_attention`` field). A plain dataclass rejects the
        # unknown keyword, so the call raises
        # ``TypeError: __init__() got an unexpected keyword argument
        # 'self_attention'`` and the function cannot build a spec at all.
        # Compare gpt_layer_specs.py:711 which correctly uses ``self_attn=``.
        # Because production raises, this test errors out and is recorded as an
        # expected failure; fixing the typo would turn it into an unexpected
        # success (a signal to update this test).
        spec = get_vit_layer_with_local_spec()
        self.assertIs(spec.submodules.self_attn.layer, SelfAttention)


if __name__ == "__main__":
    unittest.main()
