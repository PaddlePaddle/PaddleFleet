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

"""Qwen3.5 model definitions for PaddleFleet.

This module defines:

* ``Qwen3_5VisionModel`` – vision encoder (ViT + patch merger).
* ``Qwen3_5ForConditionalGeneration`` – the full VL model that composes
  vision encoder + language decoder (GPTModel).

The language model directly reuses ``GPTModel`` — no custom subclass is needed.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
from paddle import Tensor

from paddlefleet.models.gpt.gpt_embedding import GPTEmbedding
from paddlefleet.models.gpt.lm_head import GPTLMHead
from paddlefleet.pipeline_parallel import NoPipelineParallel, ScheduleNode
from paddlefleet.tensor_parallel.mappings import (
    scatter_to_sequence_parallel_region,
)
from paddlefleet.utils import get_tensor_model_parallel_group_if_none

from ...pipeline_parallel import LayerDesc
from ...transformer.layer import FleetLayer
from ...transformer.transformer_encoder import TransformerEncoder

try:
    from paddle.distributed.fleet.utils.sequence_parallel_utils import (
        mark_as_sequence_parallel_parameter,
    )
except ImportError:

    def mark_as_sequence_parallel_parameter(parameter):
        return parameter


if TYPE_CHECKING:
    from ...spec_utils import LayerSpec
    from ...transformer.transformer_config import TransformerConfig

logger = logging.getLogger(__name__)


# ======================================================================
# 1-centered RMSNorm (matching HuggingFace Qwen3_5RMSNorm)
# ======================================================================


class Qwen3_5RMSNorm(paddle.nn.Layer):
    """RMSNorm with 1-centered parameterization.

    Weight is initialized to 0 and the forward computes::

        output = rms_norm(x) * (1.0 + weight)

    This matches the HuggingFace ``Qwen3_5RMSNorm`` so that weight
    decay regularizes deviations from identity scale rather than
    pushing the scale toward zero.

    The constructor accepts both calling conventions used internally:
    - ``(config, hidden_size, eps, input_is_parallel)``
      used by ``TransformerLayer.build_layer``
    - ``(config, normalized_shape=..., norm_eps=...)``
      used by the ``SelfAttention._build_norm`` else-branch
    """

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int | None = None,
        eps: float | None = None,
        input_is_parallel: bool = False,
        normalized_shape: int | None = None,
        norm_eps: float | None = None,
        **kwargs,
    ):
        super().__init__()
        # Resolve hidden_size from either calling convention
        dim = hidden_size if hidden_size is not None else normalized_shape
        if dim is None:
            dim = config.hidden_size
        self.normalized_shape = dim

        # Resolve eps from either calling convention
        self.variance_epsilon = (
            eps
            if eps is not None
            else (norm_eps if norm_eps is not None else config.rms_norm_eps)
        )

        # Weight initialized to 0 (1-centered parameterization)
        self.weight = paddle.create_parameter(
            shape=[self.normalized_shape],
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Constant(0.0),
        )
        self.config = config

        if input_is_parallel:
            self.enable_sequence_parallel()

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype("float32")
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(
            variance + self.variance_epsilon
        )
        return (hidden_states * (1.0 + self.weight.astype("float32"))).astype(
            input_dtype
        )

    def enable_sequence_parallel(self):
        mark_as_sequence_parallel_parameter(self.weight)


class Qwen3_5RMSNormPipe(paddle.nn.Layer):
    """Pipeline-compatible wrapper for ``Qwen3_5RMSNorm``.

    Follows the same pattern as ``WrappedPaddleNormPipe``:
    handles dict I/O and MTP tensor splitting.
    """

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        input_is_parallel: bool | None = None,
    ):
        super().__init__()
        self.config = config
        self.norm = Qwen3_5RMSNorm(
            config,
            hidden_size,
            eps,
            input_is_parallel=input_is_parallel or False,
        )

    def forward(self, dict_args: dict):
        if (
            self.config.num_nextn_predict_layers is not None
            and self.config.num_nextn_predict_layers > 0
        ):
            hidden_states_concat = dict_args["hidden_states"]
            tensor_list = paddle.split(
                hidden_states_concat,
                self.config.num_nextn_predict_layers + 1,
            )
            dict_args["hidden_states"] = tensor_list[0]
        rst = {
            **dict_args,
            "hidden_states": self.norm(dict_args["hidden_states"]),
        }
        if (
            self.config.num_nextn_predict_layers is not None
            and self.config.num_nextn_predict_layers > 0
        ):
            hidden_states_concat = paddle.concat(
                [rst["hidden_states"], *tensor_list[1:]]
            )
            rst["hidden_states"] = hidden_states_concat
        return rst

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="Qwen3_5RMSNormPipe")


# ======================================================================
# Vision model
# ======================================================================


@dataclass
class Qwen3_5VisionSublayersSpec:
    """LayerSpecs for Qwen3.5 vision model: embedding + transformer layers + patch merger."""

    embedding: LayerSpec = None
    head_empty_layers: list[LayerSpec] = None
    transformer_layers: list[LayerSpec] = None
    tail_empty_layers: list[LayerSpec] = None
    merger: LayerSpec = None


class Qwen3_5VisionModel(TransformerEncoder):
    def get_layer_desc_list(self, spec: Qwen3_5VisionSublayersSpec):
        layers = []
        name_prefix = f"model.{self.modal}" if self.modal else "model"

        self.add_sequential_layer(
            layers, LayerDesc(spec.embedding), name_prefix
        )
        self.get_encoder_layer_desc_list(layers, spec, name_prefix)
        self.add_sequential_layer(
            layers, LayerDesc(spec.merger), f"{name_prefix}.merger"
        )

        return layers


# ======================================================================
# Qwen3.5 VL composite model
# ======================================================================


class Qwen3_5Model(FleetLayer):
    """Qwen3.5 Vision-Language model for conditional generation.

    Composes:
    * A ``Qwen3_5VisionModel`` (ViT encoder + patch merger) as ``self.visual``
    * A ``GPTModel`` (language decoder with hybrid full-attention +
      gated-delta-net layers) as ``self.language_model``

    The constructor receives pre-built sub-models and stores them directly,
    following the same layout as the HuggingFace ``Qwen3_5MoeModel``.

    Parameters
    ----------
    config : TransformerConfig
        Language-model config.
    vision_model : Qwen3_5VisionModel, optional
        Pre-built vision encoder.  Stored as ``self.visual``.
    language_model : GPTModel, optional
        Pre-built language decoder.
    spatial_merge_size : int
        Spatial merge factor from the vision config.
    image_token_id : int, optional
        Placeholder token id for images.
    video_token_id : int, optional
        Placeholder token id for videos.
    """

    def __init__(
        self,
        config: TransformerConfig,
        vision_model: Qwen3_5VisionModel | None = None,
        language_model=None,
        spatial_merge_size: int = 2,
        image_token_id: int | None = None,
        video_token_id: int | None = None,
    ):
        # TODO: support pipeline parallel
        assert isinstance(language_model, NoPipelineParallel)
        assert isinstance(vision_model, NoPipelineParallel)
        super().__init__(config=config)
        self.visual = vision_model
        self.language_model = language_model
        self.spatial_merge_size = spatial_merge_size
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.rope_deltas = None

        self.language_embedding = self.get_language_embedding_func(
            self.language_model
        )
        self.language_backbone = self.get_language_backbone(self.language_model)
        self.language_lm_head = self.get_lm_head(self.language_model)

        self.tp_group = get_tensor_model_parallel_group_if_none(None)

        # Disable reduce-scatter on embed_tokens so that its forward()
        # returns full [B, S, H] via all-reduce instead of sequence-parallel
        # scattered output.  We need the full tensor to merge vision features
        # before manually scattering to the SP region later in forward().
        if self.language_embedding is not None:
            embed_tokens = self.language_embedding.embedding.embed_tokens
            embed_tokens.reduce_scatter_embeddings = False

    # ------------------------------------------------------------------
    # Input embeddings helpers
    # ------------------------------------------------------------------
    def get_language_embedding_func(self, language_model):
        language_layers = self.language_model._layers.run_function
        for layer in language_layers:
            if isinstance(layer, GPTEmbedding):
                return layer
        return None

    def get_language_backbone(self, language_model):
        backbone_layers = []
        language_layers = self.language_model._layers.run_function
        for layer in language_layers:
            if not isinstance(layer, (GPTEmbedding, GPTLMHead)):
                backbone_layers.append(layer)
        return backbone_layers

    def get_lm_head(self, language_model):
        language_layers = self.language_model._layers.run_function
        for layer in language_layers:
            if isinstance(layer, GPTLMHead):
                return layer
        return None

    # ------------------------------------------------------------------
    # Vision feature extraction
    # ------------------------------------------------------------------
    def get_image_features(
        self,
        pixel_values: Tensor,
        image_grid_thw: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """Run the vision encoder and return merged image embeddings."""
        dict_input = {"pixel_values": pixel_values, "grid_thw": image_grid_thw}
        output = self.visual._layers.forward(dict_input)
        if isinstance(output, tuple):
            return output[0]
        return output

    def get_video_features(
        self,
        pixel_values_videos: Tensor,
        video_grid_thw: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """Run the vision encoder on video frames (same encoder as images)."""
        return self.get_image_features(
            pixel_values_videos, video_grid_thw, **kwargs
        )

    # ------------------------------------------------------------------
    # Placeholder mask (ported from HF Qwen3_5MoeModel.get_placeholder_mask)
    # ------------------------------------------------------------------
    def get_placeholder_mask(
        self,
        input_ids: Tensor,
        inputs_embeds: Tensor,
        image_features: Tensor | None = None,
        video_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Obtain multimodal placeholder masks from ``input_ids``.

        Returns a pair ``(image_mask, video_mask)`` broadcastable to
        ``inputs_embeds`` shape, suitable for ``paddle.where`` / ``masked_scatter``.
        """
        if input_ids is None:
            embed_fn = self.get_input_embeddings()
            special_image_mask = (
                inputs_embeds
                == embed_fn(
                    paddle.to_tensor(self.image_token_id, dtype="int64")
                )
            ).all(-1)
            special_video_mask = (
                inputs_embeds
                == embed_fn(
                    paddle.to_tensor(self.video_token_id, dtype="int64")
                )
            ).all(-1)
        else:
            special_image_mask = input_ids == self.image_token_id
            special_video_mask = input_ids == self.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(
            inputs_embeds
        )
        if image_features is not None:
            assert int(inputs_embeds[special_image_mask].numel()) == int(
                image_features.numel()
            ), (
                f"Image features and image tokens do not match: "
                f"tokens={int(n_image_tokens)}, features={image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(
            inputs_embeds
        )
        if video_features is not None:
            assert int(inputs_embeds[special_video_mask].numel()) == int(
                video_features.numel()
            ), (
                f"Video features and video tokens do not match: "
                f"tokens={int(n_video_tokens)}, features={video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask

    # ------------------------------------------------------------------
    # 3D MRoPE position ids  (ported from HF Qwen3_5MoeModel)
    # ------------------------------------------------------------------
    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw: list | Tensor,
        spatial_merge_size: int = 1,
        device: str | None = None,
    ) -> Tensor:
        """Compute 3D positional indices for a single image/video."""
        if isinstance(grid_thw, Tensor):
            t = int(grid_thw[0].item())
            h = int(grid_thw[1].item())
            w = int(grid_thw[2].item())
        else:
            t, h, w = int(grid_thw[0]), int(grid_thw[1]), int(grid_thw[2])

        llm_t = t
        llm_h = h // spatial_merge_size
        llm_w = w // spatial_merge_size
        seq_len = llm_t * llm_h * llm_w

        pos_w = paddle.arange(start_position, start_position + llm_w).tile(
            [llm_h * llm_t]
        )
        pos_h = paddle.arange(
            start_position, start_position + llm_h
        ).repeat_interleave(llm_w * llm_t)
        pos_t = paddle.full([seq_len], start_position, dtype="int64")

        return paddle.stack([pos_t, pos_h, pos_w], axis=0)

    def get_rope_index(
        self,
        input_ids: Tensor,
        mm_token_type_ids: Tensor,
        image_grid_thw: Tensor | None = None,
        video_grid_thw: Tensor | None = None,
        attention_mask: Tensor | None = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """Calculate 3D rope index based on image/video sizes.

        Returns ``(position_ids, mrope_position_deltas)`` following the
        HuggingFace reference implementation.
        """
        spatial_merge_size = self.spatial_merge_size

        mrope_position_deltas = []
        position_ids = paddle.zeros(
            [3, input_ids.shape[0], input_ids.shape[1]],
            dtype=input_ids.dtype,
        )

        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }

        for batch_idx in range(input_ids.shape[0]):
            current_input_ids = input_ids[batch_idx]
            input_token_type = mm_token_type_ids[batch_idx]

            if attention_mask is not None:
                mask = attention_mask[batch_idx].astype("bool")
                current_input_ids = current_input_ids[mask]
                input_token_type = input_token_type[mask]

            # Group consecutive tokens by modality type
            input_type_group = []
            for key, group in itertools.groupby(
                enumerate(input_token_type.tolist()), lambda x: x[1]
            ):
                group = list(group)
                input_type_group.append((key, group[0][0], group[-1][0] + 1))

            current_pos = 0
            llm_pos_ids_list = []
            for modality_type, start_idx, end_idx in input_type_group:
                if modality_type == 0:  # text
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        paddle.arange(text_len).reshape([1, -1]).expand([3, -1])
                        + current_pos
                    )
                    current_pos += text_len
                else:  # image (1) or video (2)
                    grid_thw = next(grid_iters[modality_type])
                    vision_position_ids = self.get_vision_position_ids(
                        current_pos,
                        grid_thw,
                        spatial_merge_size,
                    )
                    llm_pos_ids_list.append(vision_position_ids)
                    t_val = (
                        int(grid_thw[0].item())
                        if isinstance(grid_thw, Tensor)
                        else int(grid_thw[0])
                    )
                    h_val = (
                        int(grid_thw[1].item())
                        if isinstance(grid_thw, Tensor)
                        else int(grid_thw[1])
                    )
                    w_val = (
                        int(grid_thw[2].item())
                        if isinstance(grid_thw, Tensor)
                        else int(grid_thw[2])
                    )
                    current_pos += max(h_val, w_val) // spatial_merge_size

            llm_positions = paddle.concat(llm_pos_ids_list, axis=1).reshape(
                [3, -1]
            )

            if attention_mask is not None:
                mask = attention_mask[batch_idx].astype("bool")
                position_ids[:, batch_idx, mask] = llm_positions
            else:
                position_ids[:, batch_idx] = llm_positions

            mrope_position_deltas.append(
                int(llm_positions.max().item()) + 1 - len(current_input_ids)
            )

        mrope_position_deltas = paddle.to_tensor(
            mrope_position_deltas, dtype="int64"
        ).unsqueeze(1)

        return position_ids, mrope_position_deltas

    # ------------------------------------------------------------------
    # compute_3d_position_ids (ported from HF Qwen3_5MoeModel)
    # ------------------------------------------------------------------
    def compute_3d_position_ids(
        self,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        video_grid_thw: Tensor | None = None,
        attention_mask: Tensor | None = None,
        past_key_values=None,
        mm_token_type_ids: Tensor | None = None,
    ) -> Tensor | None:
        """Compute 3D MRoPE position ids for Qwen3.5.

        Handles both prefill (with ``input_ids``) and incremental decoding
        (with cached ``rope_deltas``), following the HF reference.
        """
        past_key_values_length = (
            0
            if past_key_values is None
            else past_key_values.get_seq_length()
            if hasattr(past_key_values, "get_seq_length")
            else 0
        )
        can_compute_mrope = (
            input_ids is not None
            and mm_token_type_ids is not None
            and (image_grid_thw is not None or video_grid_thw is not None)
        )

        if can_compute_mrope and (
            self.rope_deltas is None or past_key_values_length == 0
        ):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
            return position_ids

        if self.rope_deltas is not None and inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
            if attention_mask is not None:
                position_ids = attention_mask.astype("int64").cumsum(-1) - 1
                position_ids = paddle.where(
                    attention_mask == 0,
                    paddle.zeros_like(position_ids),
                    position_ids,
                )
                position_ids = position_ids.reshape([1, batch_size, -1]).tile(
                    [3, 1, 1]
                )
            else:
                position_ids = (
                    paddle.arange(
                        past_key_values_length,
                        past_key_values_length + seq_length,
                    )
                    .reshape([1, 1, -1])
                    .expand([3, batch_size, -1])
                )

            delta = self.rope_deltas
            if delta.shape[0] != batch_size:
                delta = delta.tile([batch_size // delta.shape[0], 1])
            position_ids = position_ids + delta.unsqueeze(0)
            return position_ids

        return None

    # ------------------------------------------------------------------
    # Forward (aligned with HF Qwen3_5Model.forward)
    # ------------------------------------------------------------------
    def forward(self, dict_args: dict) -> dict:
        """Qwen3.5 VL forward pass.

        The computation flow mirrors the HuggingFace ``Qwen3_5Model``:

        1. Embed text tokens via the language model's word embedding layer.
        2. If ``pixel_values`` is present, encode images through ``self.visual``
           and scatter the features into the embedding sequence using
           ``masked_scatter`` (via ``get_placeholder_mask``).
        3. Same for ``pixel_values_videos``.
        4. Compute 3D MRoPE position ids (or reuse cached ``rope_deltas``).
        5. Forward the language model backbone (transformer layers + final norm)
           with the merged embeddings.

        Returns a dict containing ``hidden_states`` (after final norm, before
        lm_head), matching HF's ``Qwen3_5Model`` which does not include the
        lm_head (that lives in ``Qwen3_5ForConditionalGeneration``).
        """
        input_ids = dict_args.get("input_ids", None)
        inputs_embeds = dict_args.get("inputs_embeds", None)
        pixel_values = dict_args.get("pixel_values", None)
        pixel_values_videos = dict_args.get("pixel_values_videos", None)
        image_grid_thw = dict_args.get("image_grid_thw", None)
        video_grid_thw = dict_args.get("video_grid_thw", None)
        attention_mask = dict_args.get("attention_mask", None)
        position_ids = dict_args.get("position_ids", None)
        mm_token_type_ids = dict_args.get("mm_token_type_ids", None)
        past_key_values = dict_args.get("past_key_values", None)

        # 1. Embed text tokens (word embeddings only, no position encoding)
        # Matches HF: inputs_embeds = self.get_input_embeddings()(input_ids)
        # VocabParallelEmbedding handles TP partitioning, masking, and
        # all-reduce internally.  reduce_scatter_embeddings is disabled in
        # __init__ so we get full [B, S, H] for vision feature merging.
        if (
            inputs_embeds is None
            and input_ids is not None
            and self.language_model is not None
        ):
            inputs_embeds = self.language_embedding.embedding.embed_tokens(
                input_ids
            )

        # 2. Encode and merge image features
        if pixel_values is not None and self.visual is not None:
            image_features = self.get_image_features(
                pixel_values, image_grid_thw
            )
            image_features = image_features.astype(inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids,
                inputs_embeds,
                image_features=image_features,
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask, image_features
            )

        # 3. Encode and merge video features
        if pixel_values_videos is not None and self.visual is not None:
            video_features = self.get_video_features(
                pixel_values_videos, video_grid_thw
            )
            video_features = video_features.astype(inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds,
                video_features=video_features,
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                video_mask, video_features
            )

        # 4. Compute 3D MRoPE position ids
        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        if self.config.sequence_parallel:
            inputs_embeds = inputs_embeds.transpose([1, 0, 2]).contiguous()
            inputs_embeds = scatter_to_sequence_parallel_region(
                inputs_embeds, group=self.tp_group
            )

        dict_args["position_ids"] = position_ids
        dict_args["input_ids"] = None

        # 5. Apply rotary position encoding (RoPE)
        lm_dict_args = self.language_embedding(
            dict_args, decoder_input=inputs_embeds
        )

        for layer in self.language_backbone:
            lm_dict_args = layer(lm_dict_args)

        # 6. Apply LMHead to get logits
        if self.language_lm_head is not None:
            logits = self.language_lm_head(lm_dict_args)
            return logits

        return lm_dict_args


class FleetQwen3_5ForConditionalGeneration(FleetLayer):
    def __init__(self, config, model, criterion):
        super().__init__(config)
        self.model = model
        self.criterion = criterion

    def forward(self, dict_args=None, **kwargs):
        if dict_args is None:
            dict_args = kwargs
        labels = dict_args.get("labels", None)
        logits = self.model(dict_args)
        loss = self.criterion(logits, labels)
        return loss

    def sharded_state_dict(self, structured_name_prefix: str = ""):
        """Build sharded state dict with proper name mapping for checkpoint loading.

        The Qwen3.5 model wraps language_model and visual in NoPipelineParallel,
        which adds `_layers.` prefix to parameter keys. This method bypasses
        NoPipelineParallel and directly calls sharded_state_dict on the underlying
        models (GPTModel for language, Qwen3_5VisionModel for vision).

        Both models handle pipeline layer name mapping internally via
        _pp_to_single_mapping, which converts numeric layer indices to semantic
        names with proper prefixes:
        - Language model: `0.embedding` -> `model.language_model.embedding`
        - Vision model: `0.patch_embed` -> `model.vision_model.patch_embed`

        The resulting keys will match the AOA config target format:
        - Language: `model.language_model.embedding.embed_tokens.weight`
        - Vision: `model.vision_model.patch_embed.proj.weight`
        """
        sharded_state_dict = {}

        # Get sharded state dict from language model (GPTModel wrapped in NoPipelineParallel)
        if self.model.language_model is not None:
            # Access the underlying PipelineLayer (GPTModel) directly
            # GPTModel.sharded_state_dict handles the model.language_model. prefix internally
            language_model = self.model.language_model._layers
            if hasattr(language_model, "sharded_state_dict"):
                lm_sharded = language_model.sharded_state_dict(
                    structured_name_prefix=""
                )
                sharded_state_dict.update(lm_sharded)

        # Get sharded state dict from vision model (Qwen3_5VisionModel wrapped in NoPipelineParallel)
        if self.model.visual is not None:
            # Access the underlying Qwen3_5VisionModel (TransformerEncoder) directly
            # TransformerEncoder.sharded_state_dict handles the model.vision_model. prefix
            # via _pp_to_single_mapping (since modal="vision_model")
            vision_model = self.model.visual._layers
            if hasattr(vision_model, "sharded_state_dict"):
                vm_sharded = vision_model.sharded_state_dict(
                    structured_name_prefix=""
                )
                sharded_state_dict.update(vm_sharded)

        # Get criterion parameters if any
        if self.criterion is not None:
            criterion_sharded = self.criterion.sharded_state_dict(
                structured_name_prefix=f"{structured_name_prefix}criterion."
            )
            sharded_state_dict.update(criterion_sharded)

        return sharded_state_dict
