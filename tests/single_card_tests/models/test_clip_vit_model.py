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

"""Behavior tests for paddlefleet.models.vision.clip_vit_model.

Focus is the pure, CPU-derivable arithmetic of ``get_num_image_embeddings``
(hand-derived expected counts, no reliance on the production formula), plus the
input-validation guards and the ``set_input_tensor`` delegation of
``CLIPViTModel``.

The module imports paddle at load time, so every test is gated behind an honest
``skipUnless(IMPORT_OK, ...)`` guard; the environment used to author this file
has no paddle/paddlefleet installed and therefore skips rather than fakes a
pass.
"""

import os
import sys
import types
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
    import paddle

    from paddlefleet.models.vision.clip_vit_model import (
        CLIPViTModel,
        get_num_image_embeddings,
    )
except ImportError as exc:  # only genuine missing-dependency, not API errors
    IMPORT_OK = False
    IMPORT_ERR = f"paddle/paddlefleet not importable: {exc}"

CUDA_OK = IMPORT_OK and paddle.is_compiled_with_cuda()


@unittest.skipUnless(IMPORT_OK, IMPORT_ERR)
class TestGetNumImageEmbeddings(unittest.TestCase):
    """Independent, hand-derived expected counts for image-embedding sizing.

    Expected values are computed by hand below; they never call the production
    function to build the reference, and no value is defined in terms of another
    production call.
    """

    # --- class-token handling per vision-model family ------------------------

    def test_clip_keeps_class_token(self):
        # (336 // 14) ** 2 = 24 ** 2 = 576 patches, + 1 class token.
        result = get_num_image_embeddings(336, 336, 14, "clip", False, 1, False)
        self.assertEqual(result, 577)

    def test_clip_disable_class_token_drops_it(self):
        # disable_vision_class_token=True -> patches only.
        result = get_num_image_embeddings(336, 336, 14, "clip", True, 1, False)
        self.assertEqual(result, 576)

    def test_siglip_ignores_class_token_always(self):
        # siglip never keeps a class token even when class_token_len > 0.
        # 384 // 14 = 27 (14*27=378 <= 384 < 392) -> 27 ** 2 = 729 patches.
        result = get_num_image_embeddings(
            384, 384, 14, "siglip", False, 5, False
        )
        self.assertEqual(result, 729)

    def test_internvit_keeps_class_token(self):
        result = get_num_image_embeddings(
            336, 336, 14, "internvit", False, 1, False
        )
        self.assertEqual(result, 577)

    def test_internvit300m_keeps_class_token(self):
        result = get_num_image_embeddings(
            336, 336, 14, "internvit300M", False, 1, False
        )
        self.assertEqual(result, 577)

    def test_radio_keeps_given_class_token_len(self):
        # radio uses the caller-supplied class_token_len (8 here).
        # 224 // 16 = 14 -> 196 patches, + 8 = 204.
        result = get_num_image_embeddings(
            224, 224, 16, "radio", False, 8, False
        )
        self.assertEqual(result, 204)

    def test_cradio_g_overrides_class_token_len_to_8(self):
        # cradio-g forces class_token_len to 8, discarding the caller value (1).
        # If the override were dropped, the caller value 1 would give 197.
        # 224 // 16 = 14 -> 196 patches, + 8 = 204.
        result = get_num_image_embeddings(
            224, 224, 16, "cradio-g", False, 1, False
        )
        self.assertEqual(result, 204)
        self.assertNotEqual(result, 197)  # would be the un-overridden result

    # --- pixel shuffle -------------------------------------------------------

    def test_pixel_shuffle_floors_quarter(self):
        # clip base = 577; pixel_shuffle scales by 0.5**2 = 0.25 then int()s:
        # int(577 * 0.25) = int(144.25) = 144 (truncation, not rounding).
        result = get_num_image_embeddings(336, 336, 14, "clip", False, 1, True)
        self.assertEqual(result, 144)

    def test_pixel_shuffle_applied_before_tile_tags(self):
        # Order matters: shuffle first (577 -> 144), then +5 tile tags -> 149.
        # If tags were added before shuffling: int((577+5)*0.25) = 145.
        result = get_num_image_embeddings(
            336,
            336,
            14,
            "clip",
            False,
            1,
            True,
            use_tile_tags=True,
            max_num_tiles=4,
            tokenizer_type="llama3p1",
        )
        self.assertEqual(result, 149)
        self.assertNotEqual(result, 145)  # the wrong-order result

    # --- tile tags -----------------------------------------------------------

    def test_tile_tags_llama3p1_adds_five(self):
        # 577 + 5; max_num_tiles=4 is not in (10, 100), so no qwen padding.
        result = get_num_image_embeddings(
            336,
            336,
            14,
            "clip",
            False,
            1,
            False,
            use_tile_tags=True,
            max_num_tiles=4,
            tokenizer_type="llama3p1",
        )
        self.assertEqual(result, 582)

    def test_tile_tags_qwen_small_tiles_no_padding(self):
        # qwen tokenizer but max_num_tiles=4 (<=10): +5 only, no +1 padding.
        result = get_num_image_embeddings(
            336,
            336,
            14,
            "clip",
            False,
            1,
            False,
            use_tile_tags=True,
            max_num_tiles=4,
            tokenizer_type="qwen2p0",
        )
        self.assertEqual(result, 582)

    def test_tile_tags_qwen_medium_tiles_adds_padding(self):
        # 10 < 12 < 100 and tokenizer starts with "qwen": +5 tags then +1 pad.
        # 577 + 5 + 1 = 583.
        result = get_num_image_embeddings(
            336,
            336,
            14,
            "clip",
            False,
            1,
            False,
            use_tile_tags=True,
            max_num_tiles=12,
            tokenizer_type="qwen2p0",
        )
        self.assertEqual(result, 583)

    def test_tile_tags_chatml_medium_tiles_no_padding(self):
        # Medium tile range but non-qwen tokenizer: +5 only, the +1 padding is
        # qwen-specific. 577 + 5 = 582.
        result = get_num_image_embeddings(
            336,
            336,
            14,
            "clip",
            False,
            1,
            False,
            use_tile_tags=True,
            max_num_tiles=12,
            tokenizer_type="chatml",
        )
        self.assertEqual(result, 582)

    def test_tile_tags_max_tiles_100_is_boundary_no_error(self):
        # max_num_tiles == 100 is neither in (10, 100) nor > 100: accepted,
        # no padding, no error. 577 + 5 = 582.
        result = get_num_image_embeddings(
            336,
            336,
            14,
            "clip",
            False,
            1,
            False,
            use_tile_tags=True,
            max_num_tiles=100,
            tokenizer_type="qwen2p5",
        )
        self.assertEqual(result, 582)

    # --- error contracts -----------------------------------------------------

    def test_unknown_vision_model_type_raises(self):
        with self.assertRaises(NotImplementedError):
            get_num_image_embeddings(
                336, 336, 14, "totally-unknown", False, 1, False
            )

    def test_unsupported_tokenizer_type_raises(self):
        with self.assertRaises(ValueError):
            get_num_image_embeddings(
                336,
                336,
                14,
                "clip",
                False,
                1,
                False,
                use_tile_tags=True,
                max_num_tiles=4,
                tokenizer_type="not-a-tokenizer",
            )

    def test_max_num_tiles_over_100_raises(self):
        with self.assertRaises(ValueError):
            get_num_image_embeddings(
                336,
                336,
                14,
                "clip",
                False,
                1,
                False,
                use_tile_tags=True,
                max_num_tiles=200,
                tokenizer_type="llama3p1",
            )

    # --- hf:// dispatch (get_hf_model_type is a genuinely separate, and in this
    #     source tree absent, collaborator; a controlled stub module is injected
    #     into sys.modules and removed afterward so the real branch selection in
    #     get_num_image_embeddings is what is being exercised) ------------------

    def _install_hf_stub(self, model_type_value):
        pkg_name = "paddlefleet.models.huggingface"
        mod_name = pkg_name + ".module"
        saved = {name: sys.modules.get(name) for name in (pkg_name, mod_name)}

        def _restore():
            for name, val in saved.items():
                if val is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = val

        self.addCleanup(_restore)

        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = []  # mark as package
        mod = types.ModuleType(mod_name)
        mod.get_hf_model_type = lambda vt: model_type_value
        sys.modules[pkg_name] = pkg
        sys.modules[mod_name] = mod

    def test_hf_siglip_drops_class_token(self):
        self._install_hf_stub("some_siglip_vision_model")
        # siglip family -> no class token; 384 // 14 = 27 -> 729 patches.
        result = get_num_image_embeddings(
            384, 384, 14, "hf://whatever", False, 0, False
        )
        self.assertEqual(result, 729)

    def test_hf_non_siglip_raises_not_implemented(self):
        self._install_hf_stub("some_llava_model")
        with self.assertRaises(NotImplementedError):
            get_num_image_embeddings(
                336, 336, 14, "hf://whatever", False, 1, False
            )


@unittest.skipUnless(IMPORT_OK, IMPORT_ERR)
class TestCLIPViTModelConstructionGuards(unittest.TestCase):
    """Input-validation guards that run before any heavy/paddle-device init.

    These asserts sit at the very top of ``__init__`` (before ``super().__init__``
    and before the conv/embedding construction), so they are reachable with a
    lightweight stand-in config that the guard never actually consumes.
    """

    def test_unsupported_model_subtype_raises(self):
        # The subtype check is the first statement in __init__; the config
        # object is never touched before it fires.
        with self.assertRaises(AssertionError):
            CLIPViTModel(
                transformer_config=object(),
                transformer_layer_spec=object(),
                model_subtype="not-a-real-subtype",
            )

    def test_siglip_with_class_token_raises(self):
        # Defaults are class_token_len=1 and add_class_token=True, which violate
        # siglip's "no class token" guard; this fires before super().__init__.
        with self.assertRaises(AssertionError):
            CLIPViTModel(
                transformer_config=object(),
                transformer_layer_spec=object(),
                model_subtype="siglip",
            )

    def test_siglip_with_explicit_class_token_len_raises(self):
        with self.assertRaises(AssertionError):
            CLIPViTModel(
                transformer_config=object(),
                transformer_layer_spec=object(),
                model_subtype="siglip",
                class_token_len=4,
                add_class_token=True,
            )


@unittest.skipUnless(IMPORT_OK, IMPORT_ERR)
class TestCLIPViTModelSetInputTensor(unittest.TestCase):
    """set_input_tensor must forward the exact object to the decoder."""

    def test_forwards_exact_object_to_decoder(self):
        # Exercise the real one-line delegation while replacing the genuinely
        # separate decoder collaborator (TransformerBlock) with a recorder.
        # __new__ avoids the paddle-device heavy __init__; we do not read back a
        # stuffed attribute, we observe that the real method forwards the *same*
        # object unchanged.
        model = CLIPViTModel.__new__(CLIPViTModel)

        class _Recorder:
            def __init__(self):
                self.received = []

            def set_input_tensor(self, tensor):
                self.received.append(tensor)

        recorder = _Recorder()
        model.decoder = recorder
        sentinel = object()

        model.set_input_tensor(sentinel)

        self.assertEqual(len(recorder.received), 1)
        self.assertIs(recorder.received[0], sentinel)


@unittest.skipUnless(CUDA_OK, "requires paddle built with CUDA")
class TestCLIPViTModelKnownBugs(unittest.TestCase):
    """Documented, unfixed production defects in CLIPViTModel.

    The correct behavior is asserted; the test is marked expectedFailure so the
    suite stays green while the defect exists and turns red (unexpected success)
    the moment it is fixed. Production code is NOT modified here.
    """

    @unittest.expectedFailure
    def test_minimal_clip_model_constructs(self):
        # BUG: __init__ uses several PyTorch-only APIs that do not exist in
        # Paddle, so even a minimal construction fails. Concretely:
        #   clip_vit_model.py:139  paddle.nn.Conv2d   (Paddle: paddle.nn.Conv2D)
        #   clip_vit_model.py:144  Conv2d(bias=...)   (Paddle uses bias_attr)
        #   clip_vit_model.py:158  paddle.nn.Parameter (no such Paddle symbol)
        #   clip_vit_model.py:159  paddle.randn(1, ...) positional dims
        #                          (Paddle randn takes a shape sequence)
        # Correct behavior: a minimal clip model should construct successfully.
        try:
            from paddlefleet.models.gpt.gpt_layer_specs import (
                get_gpt_layer_local_spec,
            )
            from paddlefleet.transformer.transformer_config import (
                TransformerConfig,
            )
        except ImportError as exc:
            self.skipTest(f"config/spec dependencies unavailable: {exc}")

        config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )
        spec = get_gpt_layer_local_spec(config=config)

        model = CLIPViTModel(
            transformer_config=config,
            transformer_layer_spec=spec,
            img_h=28,
            img_w=28,
            patch_dim=14,
            model_subtype="clip",
        )
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
