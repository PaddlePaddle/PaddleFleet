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


import random
import unittest

import numpy as np
import paddle
from paddle.distributed import fleet

import paddlefleet.parallel_state as ps
from paddlefleet.models.gpt import GPTConfig
from paddlefleet.models.qwen3_5.qwen3_5_model import Qwen3_5Model
from paddlefleet.models.qwen3_5.qwen3_5_provider import Qwen3_5VisionProvider
from paddlefleet.pipeline_parallel import NoPipelineParallel

# ---- Test dimensions (small for fast unit testing) ----
HIDDEN_SIZE = 64
NUM_HEADS = 4
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS  # 16
NUM_LAYERS = 2
OUT_HIDDEN_SIZE = 96
INTERMEDIATE_SIZE = 128
PATCH_SIZE = 16
SPATIAL_MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2
IN_CHANNELS = 3
NUM_POSITION_EMBEDDINGS = 256  # 16 * 16

# Test image: 1 image, 2 temporal frames, 64x64 spatial
#   pixel_values shape: [1, 3, 2, 64, 64]
#   Conv3D kernel=[2,16,16] stride=[2,16,16]
#     -> output: [1, HIDDEN_SIZE, 1, 4, 4]
#   grid_thw = [[1, 4, 4]]
#   seq_len = 1 * 4 * 4 = 16
#   After spatial merge (2x2): 16 / 4 = 4 merged tokens
IMAGE_H = 64
IMAGE_W = 64
GRID_T = 1
GRID_H = IMAGE_H // PATCH_SIZE  # 4
GRID_W = IMAGE_W // PATCH_SIZE  # 4
SEQ_LEN = GRID_T * GRID_H * GRID_W  # 16
MERGED_TOKENS = SEQ_LEN // (SPATIAL_MERGE_SIZE**2)  # 4


# class TestQwen3_5VisionModel(unittest.TestCase):
#     def setUp(self):
#         seed = 42
#         random.seed(seed)
#         np.random.seed(seed)
#         paddle.seed(seed)

#         strategy = fleet.DistributedStrategy()
#         strategy.hybrid_configs = {
#             "dp_degree": 1,
#             "mp_degree": 1,
#             "pp_degree": 1,
#             "sharding_degree": 1,
#             "sep_degree": 1,
#             "cp_degree": 1,
#             "ep_degree": 1,
#             "moe_sharding_degree": 1,
#             "order": [
#                 "sharding",
#                 "moe_sharding",
#                 "pp",
#                 "sep",
#                 "cp",
#                 "dp",
#                 "ep",
#                 "mp",
#             ],
#         }
#         self.strategy = strategy

#         if not ps.have_global_memory_buffer():
#             fleet.init(is_collective=True, strategy=strategy)
#             hcg = fleet.get_hybrid_communicate_group()
#             ps.initialize_model_parallel(hcg)

#         # Step 1: Create Qwen3.5 vision encoder
#         config = Qwen3_5VisionProvider(
#             num_hidden_layers=NUM_LAYERS,
#             hidden_size=HIDDEN_SIZE,
#             num_attention_heads=NUM_HEADS,
#             head_dim=HEAD_DIM,
#             out_hidden_size=OUT_HIDDEN_SIZE,
#             intermediate_size=INTERMEDIATE_SIZE,
#             patch_size=PATCH_SIZE,
#             spatial_merge_size=SPATIAL_MERGE_SIZE,
#             temporal_patch_size=TEMPORAL_PATCH_SIZE,
#             in_channels=IN_CHANNELS,
#             num_position_embeddings=NUM_POSITION_EMBEDDINGS,
#             hidden_dropout_prob=0.0,
#             attention_dropout=0.0,
#             normalization="LayerNorm",
#             use_qk_norm=False,
#             gated_linear_unit=False,
#             apply_rope_fusion=False,
#         )
#         self.config = config
#         self.vision_model = config.provide()

#     def test_forward(self):
#         """Test full forward and backward of Qwen3.5 vision encoder.

#         Complete computation flow starting from raw pixel input:
#           pixel_values -> Conv3D patch embedding -> Transformer layers -> PatchMerger -> output
#         Then backward through the entire graph, checking gradient shapes.
#         """
#         # ---- Step 2: create NoPipelineParallel ----
#         vision = NoPipelineParallel(self.vision_model, self.strategy)

#         # ---- Verify model structure ----
#         layers = vision._layers.run_function
#         print(f"\nTotal layers in run_function: {len(layers)}")
#         for i, layer in enumerate(layers):
#             print(f"  [{i}] {type(layer).__name__}")

#         num_total = 1 + NUM_LAYERS + 1
#         assert len(layers) == num_total, (
#             f"Expected {num_total} layers, got {len(layers)}"
#         )
#         assert isinstance(layers[0], VisionEmbedding)
#         for i in range(1, 1 + NUM_LAYERS):
#             assert isinstance(layers[i], TransformerLayer)
#         assert isinstance(layers[-1], Qwen3VLVisionPathMerger)

#         # ---- Step 3: Forward computation (full flow from pixel input) ----
#         # vision._layers.forward() internally does:
#         #   x = input; for layer in run_function: x = layer(x)
#         # We chain through run_function equivalently, calling each component's
#         # actual sub-modules to test the complete computation graph.

#         embedding = layers[0]
#         merger = layers[-1]

#         # ---- 3a. Construct raw image input ----
#         # pixel_values: [N_chunks, C, temporal_patch_size, H, W]
#         # For 1 image with 2 temporal frames at 64x64:
#         pixel_values = paddle.randn(
#             [GRID_T, IN_CHANNELS, TEMPORAL_PATCH_SIZE, IMAGE_H, IMAGE_W]
#         )
#         grid_thw = paddle.to_tensor(
#             [[GRID_T, GRID_H, GRID_W]], dtype=paddle.int32
#         )

#         # ---- 3b. Patch embedding (Conv3D) ----
#         # Conv3D: [1, 3, 2, 64, 64] -> [1, HIDDEN_SIZE, 1, 4, 4]
#         patch_out = embedding.patch_embed(pixel_values)
#         assert list(patch_out.shape) == [
#             GRID_T,
#             HIDDEN_SIZE,
#             1,  # temporal: 2 / temporal_patch_size(2) = 1
#             GRID_H,  # spatial: 64 / patch_size(16) = 4
#             GRID_W,  # spatial: 64 / patch_size(16) = 4
#         ]

#         # Reshape to [1, seq_len, hidden_size]
#         # Conv3D out: [N, C_out, D, H, W] -> flatten D/H/W -> transpose
#         hidden_states = patch_out.flatten(2).transpose([0, 2, 1])
#         assert list(hidden_states.shape) == [1, SEQ_LEN, HIDDEN_SIZE]

#         # ---- 3c. Transformer layers ----
#         attention_mask = paddle.ones(
#             [1, 1, SEQ_LEN, SEQ_LEN], dtype=paddle.bool
#         )
#         dict_args = {
#             "hidden_states": hidden_states,
#             "attention_mask": attention_mask,
#             "rotary_pos_emb": None,
#             "rotary_pos_cos": None,
#             "rotary_pos_sin": None,
#             "packed_seq_params": None,
#         }

#         for i in range(1, 1 + NUM_LAYERS):
#             dict_args = layers[i](dict_args)

#         transformer_output = dict_args["hidden_states"]
#         assert list(transformer_output.shape) == [1, SEQ_LEN, HIDDEN_SIZE], (
#             f"Expected [1, {SEQ_LEN}, {HIDDEN_SIZE}], "
#             f"got {list(transformer_output.shape)}"
#         )

#         # ---- 3d. Patch merger ----
#         # Merger expects tensor [seq_len, hidden_size], not dict.
#         # norm([16, 64]) -> reshape([-1, 256]) = [4, 256] -> MLP -> [4, 96]
#         merger_input = transformer_output.squeeze(0)
#         output, _ = merger(merger_input)
#         assert list(output.shape) == [MERGED_TOKENS, OUT_HIDDEN_SIZE], (
#             f"Expected [{MERGED_TOKENS}, {OUT_HIDDEN_SIZE}], "
#             f"got {list(output.shape)}"
#         )

#         # ---- Step 4: Backward computation ----
#         loss = output.sum()
#         loss.backward()

#         # Check gradients for all parameters in the forward path:
#         #   - embedding.patch_embed (Conv3D weight/bias)
#         #   - transformer layers (attention + MLP)
#         #   - merger (norm + MLP)
#         # Excluded: embedding.pos_embed, embedding.rotary_pos_emb
#         #   (not used in this forward path)
#         SKIP_PREFIXES = ("0.pos_embed", "0.rotary_pos_emb")

#         params_with_grad = 0
#         for name, param in vision._layers.named_parameters():
#             if any(name.startswith(p) for p in SKIP_PREFIXES):
#                 continue
#             if param.grad is None:
#                 print(f"  [NO GRAD] {name}: shape={list(param.shape)}")
#                 continue

#             params_with_grad += 1

#             # Gradient shape must match parameter shape
#             assert list(param.shape) == list(param.grad.shape), (
#                 f"Gradient shape mismatch for {name}: "
#                 f"param={list(param.shape)}, grad={list(param.grad.shape)}"
#             )
#             # Gradients must be finite
#             assert paddle.isfinite(param.grad).all().item(), (
#                 f"Non-finite gradients for {name}"
#             )

#             grad_norm = param.grad.detach().norm().item()
#             print(
#                 f"  {name}: shape={list(param.shape)}, "
#                 f"grad_norm={grad_norm:.6f}"
#             )

#         assert params_with_grad > 0, "No parameters received gradients"


# ======================================================================
# Qwen3_5Model (VL composite) tests using real vision and language models
# ======================================================================

from paddlefleet.models.qwen3_5.layer_specs import get_qwen3_5_language_spec
from paddlefleet.spec_utils import build_layer

# ---- Qwen3_5Model test dimensions ----
VL_HIDDEN_SIZE = HIDDEN_SIZE  # 64, vision hidden size
VL_LM_HIDDEN_SIZE = OUT_HIDDEN_SIZE  # 96, must match vision output dim
VL_VOCAB_SIZE = 256
VL_IMAGE_TOKEN_ID = 200
VL_VIDEO_TOKEN_ID = 201
VL_NUM_LM_LAYERS = 2
VL_TEXT_BEFORE = 5
VL_TEXT_AFTER = 3
VL_NUM_IMAGE_TOKENS = MERGED_TOKENS  # 4
VL_SEQ_LEN = VL_TEXT_BEFORE + VL_NUM_IMAGE_TOKENS + VL_TEXT_AFTER  # 12


class TestQwen3_5Model(unittest.TestCase):
    """Test Qwen3_5Model (VL composite model) forward and backward.

    Uses real Qwen3_5VisionModel and GPTModel sub-models to verify:
    - Vision encoder (ViT + patch merger) produces correct features
    - Language decoder (GPT with transformer layers) processes embeddings
    - Vision-language feature merging via masked_scatter
    - 3D MRoPE position ID computation
    - Gradient flow through the entire computation graph
    """

    def setUp(self):
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
            "sharding_degree": 1,
            "sep_degree": 1,
            "cp_degree": 1,
            "ep_degree": 1,
            "moe_sharding_degree": 1,
            "order": [
                "sharding",
                "moe_sharding",
                "pp",
                "sep",
                "cp",
                "dp",
                "ep",
                "mp",
            ],
            "pp_configs": {"delay_scale_loss": False},
        }
        strategy.pipeline_configs = {
            "micro_batch_size": 1,
            "accumulate_steps": 1,
        }
        self.strategy = strategy

        if not ps.have_global_memory_buffer():
            fleet.init(is_collective=True, strategy=strategy)
            hcg = fleet.get_hybrid_communicate_group()
            ps.initialize_model_parallel(hcg)

        # Step 1: Create vision_config and Qwen3_5VisionModel
        vision_config = Qwen3_5VisionProvider(
            num_hidden_layers=NUM_LAYERS,
            hidden_size=HIDDEN_SIZE,
            num_attention_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            out_hidden_size=OUT_HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            patch_size=PATCH_SIZE,
            spatial_merge_size=SPATIAL_MERGE_SIZE,
            temporal_patch_size=TEMPORAL_PATCH_SIZE,
            in_channels=IN_CHANNELS,
            num_position_embeddings=NUM_POSITION_EMBEDDINGS,
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            normalization="LayerNorm",
            use_qk_norm=False,
            gated_linear_unit=False,
            apply_rope_fusion=False,
        )
        self.vision_config = vision_config
        vision_model = vision_config.provide()

        # Step 2: Create language_config and GPTModel
        language_config = GPTConfig(
            num_hidden_layers=VL_NUM_LM_LAYERS,
            hidden_size=VL_LM_HIDDEN_SIZE,
            num_attention_heads=NUM_HEADS,
            head_dim=VL_LM_HIDDEN_SIZE // NUM_HEADS,
            intermediate_size=INTERMEDIATE_SIZE,
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            normalization="LayerNorm",
            gated_linear_unit=False,
            apply_rope_fusion=False,
            vocab_size=VL_VOCAB_SIZE,
            max_sequence_length=1024,
            position_embedding_type="rope",
            rotary_percent=1.0,
            rotary_base=10000,
            rope_scaling=False,
            parallel_output=False,
            tie_word_embeddings=False,
            layer_types=["full_attention", "linear_attention"],
            gated_attention=True,
        )
        self.language_config = language_config

        language_spec = get_qwen3_5_language_spec(
            config=language_config,
        )
        language_model = build_layer(
            language_spec,
            seg_method="layer:TransformerLayer|EmptyLayer",
            num_stages=1,
        )

        # Step 3: Create Qwen3_5Model with vision and language models
        self.model = Qwen3_5Model(
            config=language_config,
            vision_model=NoPipelineParallel(vision_model, strategy),
            language_model=NoPipelineParallel(language_model, strategy),
            spatial_merge_size=SPATIAL_MERGE_SIZE,
            image_token_id=VL_IMAGE_TOKEN_ID,
            video_token_id=VL_VIDEO_TOKEN_ID,
        )

    def _clear_gradients(self):
        for param in self.model.parameters():
            if param.grad is not None:
                param.clear_gradient()
        self.model.rope_deltas = None

    def test_forward_backward_with_image(self):
        """Test full VL forward and backward with image input.

        Exercises the complete computation flow:
          1. Embed text tokens via language model embedding layer
          2. Encode image via real vision encoder (Conv3D + Transformer + PatchMerger)
          3. Merge image features into embedding sequence (masked_scatter)
          4. Compute 3D MRoPE position IDs
          5. Forward through language model transformer layers
          6. Backward through the entire graph
        """
        self._clear_gradients()
        batch_size = 1

        # ---- Construct multimodal input ----
        # input_ids: [text ... image_tokens ... text]
        input_ids = paddle.randint(0, 100, [batch_size, VL_SEQ_LEN])
        input_ids[0, VL_TEXT_BEFORE : VL_TEXT_BEFORE + VL_NUM_IMAGE_TOKENS] = (
            VL_IMAGE_TOKEN_ID
        )

        # mm_token_type_ids: 0=text, 1=image
        mm_token_type_ids = paddle.zeros(
            [batch_size, VL_SEQ_LEN], dtype="int64"
        )
        mm_token_type_ids[
            0, VL_TEXT_BEFORE : VL_TEXT_BEFORE + VL_NUM_IMAGE_TOKENS
        ] = 1

        image_grid_thw = paddle.to_tensor(
            [[GRID_T, GRID_H, GRID_W]], dtype="int32"
        )
        pixel_values = paddle.randn(
            [GRID_T, IN_CHANNELS, TEMPORAL_PATCH_SIZE, IMAGE_H, IMAGE_W]
        )

        dict_args = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
        }

        # ---- Forward ----
        output = self.model.forward(dict_args)

        # assert "hidden_states" in output, "Output must contain 'hidden_states'"
        # hidden_states = output["hidden_states"]
        # assert list(hidden_states.shape) == [batch_size, VL_SEQ_LEN, VL_LM_HIDDEN_SIZE], (
        #     f"Expected shape [{batch_size}, {VL_SEQ_LEN}, {VL_LM_HIDDEN_SIZE}], "
        #     f"got {list(hidden_states.shape)}"
        # )

        # ---- Backward ----
        loss = output.sum()
        loss.backward()

        # ---- Verify gradients ----
        params_with_grad = 0
        for name, param in self.model.named_parameters():
            if param.grad is None:
                print(f"  [NO GRAD] {name}: shape={list(param.shape)}")
                continue

            params_with_grad += 1
            assert list(param.shape) == list(param.grad.shape), (
                f"Gradient shape mismatch for {name}: "
                f"param={list(param.shape)}, grad={list(param.grad.shape)}"
            )
            assert paddle.isfinite(param.grad).all().item(), (
                f"Non-finite gradients for {name}"
            )
            grad_norm = param.grad.detach().norm().item()
            print(
                f"  {name}: shape={list(param.shape)}, grad_norm={grad_norm:.6f}"
            )

        assert params_with_grad > 0, "No parameters received gradients"

        # Both vision and language model parameters should receive gradients
        vision_has_grad = any(
            p.grad is not None for p in self.model.visual.parameters()
        )
        lm_has_grad = any(
            p.grad is not None for p in self.model.language_model.parameters()
        )
        assert vision_has_grad, (
            "Vision model parameters did not receive gradients"
        )
        assert lm_has_grad, (
            "Language model parameters did not receive gradients"
        )

    def test_forward_backward_text_only(self):
        """Test forward and backward with text-only input (no vision).

        When no pixel_values are provided, the model should:
        - Embed text via language model embedding layer
        - Skip vision encoding entirely
        - Forward through language model transformer layers
        - Gradient flows only through language model
        """
        self._clear_gradients()
        batch_size = 1
        text_seq_len = 10

        input_ids = paddle.randint(0, 100, [batch_size, text_seq_len])
        dict_args = {"input_ids": input_ids}

        # ---- Forward ----
        output = self.model.forward(dict_args)

        # assert "hidden_states" in output
        # hidden_states = output["hidden_states"]
        # assert list(output.shape) == [batch_size, text_seq_len, VL_LM_HIDDEN_SIZE], (
        #     f"Expected shape [{batch_size}, {text_seq_len}, {VL_LM_HIDDEN_SIZE}], "
        #     f"got {list(hidden_states.shape)}"
        # )

        # ---- Backward ----
        loss = output.sum()
        loss.backward()

        # Language model params should have gradients
        lm_params_with_grad = 0
        for name, param in self.model.language_model.named_parameters():
            if param.grad is not None:
                lm_params_with_grad += 1
                assert paddle.isfinite(param.grad).all().item(), (
                    f"Non-finite gradients for language_model.{name}"
                )
        assert lm_params_with_grad > 0, (
            "Language model parameters did not receive gradients"
        )

        # Vision model params should NOT have gradients (not used)
        for name, param in self.model.visual.named_parameters():
            assert param.grad is None, (
                f"Vision param {name} should not have gradient in text-only mode"
            )

    def test_get_rope_index(self):
        """Test 3D MRoPE position ID computation for mixed text+image tokens.

        Verifies that get_rope_index correctly computes 3D (temporal, height, width)
        position IDs for a sequence with interleaved text and image tokens.
        """
        batch_size = 1

        # Construct input_ids with image placeholders
        input_ids = paddle.randint(0, 100, [batch_size, VL_SEQ_LEN])
        input_ids[0, VL_TEXT_BEFORE : VL_TEXT_BEFORE + VL_NUM_IMAGE_TOKENS] = (
            VL_IMAGE_TOKEN_ID
        )

        mm_token_type_ids = paddle.zeros(
            [batch_size, VL_SEQ_LEN], dtype="int64"
        )
        mm_token_type_ids[
            0, VL_TEXT_BEFORE : VL_TEXT_BEFORE + VL_NUM_IMAGE_TOKENS
        ] = 1

        image_grid_thw = paddle.to_tensor(
            [[GRID_T, GRID_H, GRID_W]], dtype="int32"
        )

        # ---- Compute rope index ----
        position_ids, mrope_deltas = self.model.get_rope_index(
            input_ids,
            mm_token_type_ids,
            image_grid_thw=image_grid_thw,
        )

        # position_ids: [3, batch_size, seq_len] for (temporal, height, width)
        assert list(position_ids.shape) == [3, batch_size, VL_SEQ_LEN], (
            f"Expected position_ids shape [3, {batch_size}, {VL_SEQ_LEN}], "
            f"got {list(position_ids.shape)}"
        )
        # mrope_position_deltas: [batch_size, 1]
        assert list(mrope_deltas.shape) == [batch_size, 1], (
            f"Expected mrope_deltas shape [{batch_size}, 1], "
            f"got {list(mrope_deltas.shape)}"
        )
        # All position IDs should be non-negative
        assert (position_ids >= 0).all().item(), (
            "Position IDs contain negative values"
        )

        # Text tokens before image should have monotonically increasing position IDs
        # and the 3 axes should be identical for text tokens
        text_before_pos = position_ids[:, 0, :VL_TEXT_BEFORE]
        for axis in range(3):
            for j in range(1, VL_TEXT_BEFORE):
                assert (
                    text_before_pos[axis, j].item()
                    > text_before_pos[axis, j - 1].item()
                ), f"Text positions not monotonically increasing on axis {axis}"

    def test_get_placeholder_mask(self):
        """Test that placeholder masks correctly identify image/video tokens."""
        batch_size = 1

        input_ids = paddle.randint(0, 100, [batch_size, VL_SEQ_LEN])
        input_ids[0, VL_TEXT_BEFORE : VL_TEXT_BEFORE + VL_NUM_IMAGE_TOKENS] = (
            VL_IMAGE_TOKEN_ID
        )

        inputs_embeds = paddle.randn(
            [batch_size, VL_SEQ_LEN, VL_LM_HIDDEN_SIZE]
        )

        image_mask, video_mask = self.model.get_placeholder_mask(
            input_ids,
            inputs_embeds,
        )

        # Masks should be broadcastable to inputs_embeds shape
        assert list(image_mask.shape) == [
            batch_size,
            VL_SEQ_LEN,
            VL_LM_HIDDEN_SIZE,
        ]
        assert list(video_mask.shape) == [
            batch_size,
            VL_SEQ_LEN,
            VL_LM_HIDDEN_SIZE,
        ]

        # image_mask should be True at image token positions (expanded across hidden dim)
        image_mask_1d = image_mask[0, :, 0]  # [seq_len]
        for i in range(VL_SEQ_LEN):
            if VL_TEXT_BEFORE <= i < VL_TEXT_BEFORE + VL_NUM_IMAGE_TOKENS:
                assert image_mask_1d[i].item(), (
                    f"Position {i} should be masked as image"
                )
            else:
                assert not image_mask_1d[i].item(), (
                    f"Position {i} should NOT be masked as image"
                )

        # No video tokens in this input
        assert not video_mask.any().item(), "No video tokens should be detected"


if __name__ == "__main__":
    unittest.main()
