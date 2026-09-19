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

"""Model-layer tests for ``get_num_image_embeddings`` in
``paddlefleet.models.vision.clip_vit_model``.

This file covers a slice disjoint from ``test_clip_vit_model.py`` (which
exercises ``CLIPViTModel`` construction / ``forward`` / ``set_input_tensor``).
Here the target is the pure integer helper ``get_num_image_embeddings``: how it
counts patches, when the class token is kept vs dropped per ``vision_model_type``,
the forced ``class_token_len == 8`` override for ``cradio-g``, pixel-shuffle
scaling with integer truncation, tile-tag additions and the qwen-only +1 padding,
and the exact error contracts. Every expected value is hand-derived from image /
patch geometry, not read back from the function under test.

The module imports ``paddle`` transitively, so the whole suite is skipped with an
honest reason when Paddle is unavailable. The tested logic itself is plain
integer arithmetic and needs neither a GPU nor Paddle math once imported.
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
# Also make the ``src/`` layout importable when the package is not installed.
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
        ),
        "src",
    ),
)

try:
    from paddlefleet.models.vision.clip_vit_model import (
        get_num_image_embeddings,
    )

    _IMPORT_OK = True
    _IMPORT_ERR = ""
except ImportError as exc:  # pragma: no cover - depends on runtime deps
    _IMPORT_OK = False
    _IMPORT_ERR = repr(exc)
    get_num_image_embeddings = None


# 336x336 image with 14px patches -> 24 patches per side -> 576 patches.
# 224x224 image with 16px patches -> 14 patches per side -> 196 patches.
@unittest.skipUnless(
    _IMPORT_OK,
    "paddlefleet.models.vision.clip_vit_model not importable "
    "(missing paddle or package): " + _IMPORT_ERR,
)
class TestGetNumImageEmbeddings(unittest.TestCase):
    """Hand-derived expectations for ``get_num_image_embeddings``."""

    def _call(self, **overrides):
        kwargs = {
            "img_h": 336,
            "img_w": 336,
            "patch_dim": 14,
            "vision_model_type": "clip",
            "disable_vision_class_token": False,
            "class_token_len": 1,
            "pixel_shuffle": False,
        }
        kwargs.update(overrides)
        return get_num_image_embeddings(**kwargs)

    # -- patch counting -------------------------------------------------

    def test_clip_class_token_len_is_added_not_hardcoded(self):
        # 576 patches + class_token_len; the token count must track the
        # argument, so len=1 -> 577 and len=8 -> 584 (not a fixed +1).
        self.assertEqual(self._call(class_token_len=1), 577)
        self.assertEqual(self._call(class_token_len=8), 584)

    def test_clip_disable_excludes_class_token(self):
        # disable_vision_class_token=True drops the token entirely: 576 exactly.
        self.assertEqual(
            self._call(disable_vision_class_token=True, class_token_len=1),
            576,
        )

    def test_non_square_image_uses_both_dims(self):
        # 336//14 = 24 rows, 224//14 = 16 cols -> 24*16 = 384 patches.
        # A bug that squared a single dimension would give 576 or 256.
        self.assertEqual(
            self._call(
                img_h=336,
                img_w=224,
                patch_dim=14,
                disable_vision_class_token=True,
            ),
            384,
        )

    def test_patch_dim_floor_division(self):
        # 225//16 = 14 (14*16=224 <= 225 < 240) -> 14*14 = 196 patches.
        self.assertEqual(
            self._call(
                img_h=225,
                img_w=225,
                patch_dim=16,
                disable_vision_class_token=True,
            ),
            196,
        )

    # -- per-model class-token policy -----------------------------------

    def test_siglip_forces_no_class_token(self):
        # SigLIP keeps no class token regardless of the passed flag/len.
        self.assertEqual(
            self._call(
                vision_model_type="siglip",
                disable_vision_class_token=False,
                class_token_len=5,
            ),
            576,
        )
        self.assertEqual(
            self._call(
                vision_model_type="siglip",
                disable_vision_class_token=True,
                class_token_len=5,
            ),
            576,
        )

    def test_internvit_variants_follow_disable_flag(self):
        for subtype in ("internvit", "internvit300M"):
            self.assertEqual(
                self._call(
                    vision_model_type=subtype,
                    disable_vision_class_token=False,
                    class_token_len=1,
                ),
                577,
                msg=subtype,
            )
            self.assertEqual(
                self._call(
                    vision_model_type=subtype,
                    disable_vision_class_token=True,
                    class_token_len=1,
                ),
                576,
                msg=subtype,
            )

    def test_radio_prefix_passes_class_token_len_through(self):
        # 224/16 -> 196 patches; radio* keeps the *given* class_token_len.
        for name in ("radio", "radioXL"):
            self.assertEqual(
                self._call(
                    img_h=224,
                    img_w=224,
                    patch_dim=16,
                    vision_model_type=name,
                    disable_vision_class_token=False,
                    class_token_len=3,
                ),
                199,
                msg=name,
            )
        self.assertEqual(
            self._call(
                img_h=224,
                img_w=224,
                patch_dim=16,
                vision_model_type="radio",
                disable_vision_class_token=True,
                class_token_len=3,
            ),
            196,
        )

    def test_cradio_g_overrides_class_token_len_to_eight(self):
        # cradio-g forces class_token_len=8 internally, so passing 1 must still
        # yield 196 + 8 = 204, never 196 + 1 = 197.
        self.assertEqual(
            self._call(
                img_h=224,
                img_w=224,
                patch_dim=16,
                vision_model_type="cradio-g",
                disable_vision_class_token=False,
                class_token_len=1,
            ),
            204,
        )
        # With the class token disabled the override is moot: 196.
        self.assertEqual(
            self._call(
                img_h=224,
                img_w=224,
                patch_dim=16,
                vision_model_type="cradio-g",
                disable_vision_class_token=True,
                class_token_len=1,
            ),
            196,
        )

    # -- pixel shuffle ---------------------------------------------------

    def test_pixel_shuffle_scales_after_class_token_and_truncates(self):
        # Scale factor is 0.5**2 = 0.25 applied to (patches + class token).
        # disable=True: int(576 * 0.25) = 144.
        self.assertEqual(
            self._call(disable_vision_class_token=True, pixel_shuffle=True),
            144,
        )
        # disable=False, len=4: int(580 * 0.25) = 145 (distinct from 144),
        # proving the class token is included before scaling.
        self.assertEqual(
            self._call(
                disable_vision_class_token=False,
                class_token_len=4,
                pixel_shuffle=True,
            ),
            145,
        )
        # Truncation is visible: 224/16 -> 196 + 1 = 197; int(197*0.25)=49.
        self.assertEqual(
            self._call(
                img_h=224,
                img_w=224,
                patch_dim=16,
                disable_vision_class_token=False,
                class_token_len=1,
                pixel_shuffle=True,
            ),
            49,
        )

    # -- tile tags -------------------------------------------------------

    def test_tile_tags_add_five_for_supported_tokenizers(self):
        # 577 base + 5 tile-tag tokens = 582; no qwen padding for these.
        for tok in ("llama3p1", "chatml"):
            self.assertEqual(
                self._call(
                    use_tile_tags=True,
                    max_num_tiles=0,
                    tokenizer_type=tok,
                ),
                582,
                msg=tok,
            )

    def test_tile_tags_qwen_medium_range_adds_padding(self):
        # qwen* in 10<tiles<100 gets one extra padding token: 577+5+1 = 583.
        self.assertEqual(
            self._call(
                use_tile_tags=True,
                max_num_tiles=12,
                tokenizer_type="qwen2p5",
            ),
            583,
        )
        # A non-qwen tokenizer at the same tile count gets no +1: 582.
        self.assertEqual(
            self._call(
                use_tile_tags=True,
                max_num_tiles=12,
                tokenizer_type="llama3p1",
            ),
            582,
        )

    def test_tile_tags_qwen_padding_boundaries(self):
        # Range is strict: 10 and 100 are excluded, 11 and 99 included.
        self.assertEqual(
            self._call(
                use_tile_tags=True,
                max_num_tiles=10,
                tokenizer_type="qwen2p0",
            ),
            582,
        )
        self.assertEqual(
            self._call(
                use_tile_tags=True,
                max_num_tiles=100,
                tokenizer_type="qwen2p0",
            ),
            582,
        )
        self.assertEqual(
            self._call(
                use_tile_tags=True,
                max_num_tiles=11,
                tokenizer_type="qwen2p0",
            ),
            583,
        )
        self.assertEqual(
            self._call(
                use_tile_tags=True,
                max_num_tiles=99,
                tokenizer_type="qwen2p0",
            ),
            583,
        )

    # -- error contracts -------------------------------------------------

    def test_tile_tags_unknown_tokenizer_raises(self):
        with self.assertRaises(ValueError):
            self._call(
                use_tile_tags=True,
                max_num_tiles=0,
                tokenizer_type="gpt2",
            )

    def test_tile_tags_too_many_tiles_raises(self):
        with self.assertRaises(ValueError):
            self._call(
                use_tile_tags=True,
                max_num_tiles=101,
                tokenizer_type="qwen2p5",
            )

    def test_unknown_vision_model_type_raises(self):
        with self.assertRaises(NotImplementedError):
            self._call(vision_model_type="foobar")
        with self.assertRaises(NotImplementedError):
            self._call(vision_model_type="")


if __name__ == "__main__":
    unittest.main()
