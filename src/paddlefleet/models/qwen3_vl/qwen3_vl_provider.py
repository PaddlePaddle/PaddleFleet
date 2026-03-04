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
import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import paddle
from paddle.nn import functional as F
from paddleformers.transformers.gpt_provider import GPTModelProvider

from paddlefleet import parallel_state

from ...spec_utils import LayerSpec
from ...transformer import TransformerConfig
from .embedding import Qwen3VLTextEmbedding
from .qwen3_vl_builders import qwen3_vl_vision_builder
from .qwen3_vl_model import (
    Qwen3VLModelDist,
    Qwen3VLTextTransformerLayer,
    Qwen3VLVisionModel,
    Qwen3VLVisionTransformerLayer,
)


@dataclass
class Qwen3VLVisionProvider(TransformerConfig):
    patch_size: int = 16
    use_bias: bool = True
    add_qkv_bias: bool = True
    num_position_embeddings: int = 2304
    embed_dim: int = (1152,)
    hidden_size: int = 1152
    out_hidden_size: int = 4096
    in_channels: int = 3
    spatial_merge_size: int = 2
    spatial_patch_size: int = 16
    temporal_patch_size: int = 2
    hidden_dropout_prob: float = 0.0
    attention_dropout: float = 0.0
    intermediate_size: int = 4304
    initializer_range: float = 0.02
    gated_linear_unit: bool = False
    activation_func: Callable = F.gelu
    layernorm_zero_centered_gamma: bool = False
    apply_query_key_layer_scaling: bool = False
    persist_layer_norm: bool = True
    bias_activation_fusion: bool = False
    bias_dropout_fusion: bool = False
    attention_softmax_in_fp32: bool = True
    normalization: str = "LayerNorm"
    apply_rope_fusion: bool = True
    rms_norm_eps: float = 1e-6
    transformer_layer_spec: LayerSpec = Qwen3VLVisionTransformerLayer
    model_version: str = "qwen3_vl"
    img_h: int = 336
    img_w: int = 336
    add_class_token: bool = False
    class_token_len: int = 1
    high_precision_rope: bool = True
    rotary_percent: float = 1.0
    transform_rules = {
        "dtype": "params_dtype",
        "num_heads": "num_attention_heads",
        "depth": "num_hidden_layers",
        "initializer_range": "init_method_std",
    }

    def provide(self) -> "Qwen3VLVisionModel":
        pp_size = self.pipeline_model_parallel_size

        is_pipeline_asymmetric = getattr(
            self, "account_for_embedding_in_pipeline_split", False
        ) or getattr(self, "account_for_loss_in_pipeline_split", False)
        is_pipeline_asymmetric |= (
            getattr(self, "num_empty_layers_add_in_head", None)
            or getattr(self, "num_empty_layers_add_in_tail", None)
        ) is not None

        # Initialize model as meta data instead of allocating data on a device
        model_init_device_context = contextlib.nullcontext
        if self.init_model_with_meta_device:
            model_init_device_context = partial(paddle.device, device="meta")

        with model_init_device_context():
            fleet_model = qwen3_vl_vision_builder(
                self,
                seg_method="layer:TransformerLayer|EmptyLayer",
                num_stages=pp_size,
            )
            model = Qwen3VLVisionModel.__new__(Qwen3VLVisionModel)

            for attr_name in dir(fleet_model):
                if not attr_name.startswith("__"):
                    try:
                        attr_value = getattr(fleet_model, attr_name)
                        setattr(model, attr_name, attr_value)
                    except:
                        pass
        return model


@dataclass
class Qwen3VLTextProvider(GPTModelProvider):
    """
    Base config for Qwen3 Models.
    """

    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    use_bias: bool = False
    add_qkv_bias: bool = True
    seq_length: int = 4096
    init_method_std: int = 0.02
    hidden_dropout_prob: float = 0.0
    attention_dropout: float = 0.0
    vocab_size: int = 151936
    share_embeddings_and_output_weights: bool | None = False
    rms_norm_eps: float = 1e-6
    rotary_base: float = 1000000.0
    position_embedding_type: str = "rope"
    use_qk_norm: bool = True
    specific_embedding: type = Qwen3VLTextEmbedding
    specific_transformer_layer: type = Qwen3VLTextTransformerLayer
    max_sequence_length: int = 262144
    multimodal_embedding: bool = False
    _save_to_hf: bool = False
    use_fused_linear_cross_entropy: bool = True
    high_precision_rope: bool = True
    moe_grouped_gemm: bool = True

    n_shared_experts: int = 0
    transform_rules = {
        "dtype": "params_dtype",
        "num_heads": "num_attention_heads",
        "depth": "num_hidden_layers",
        "initializer_range": "init_method_std",
        "num_experts": "n_routed_experts",
    }

    def __post_init__(self):
        super().__post_init__()
        self.mrope_section = self.rope_scaling.get(
            "mrope_section", [24, 20, 20]
        )


@dataclass
class Qwen3VLProvider(TransformerConfig):
    text_config: Qwen3VLTextProvider | None = None
    vision_config: Qwen3VLVisionProvider | None = None

    drop_vision_class_token: bool = False
    vision_feature_layer: int = -2

    encoder_pipeline_model_parallel_size: int = 0
    encoder_tensor_model_parallel_size: int = 1

    seq_length: int = 1024

    language_model_from_pretrained: str | None = None
    vision_model_from_pretrained: str | None = None

    def provide(
        self, tokenizer=None, vp_stage: int | None = None
    ) -> "Qwen3VLModelDist":
        self.text_config.scatter_embedding_sequence_parallel = False
        self.text_config.tensor_model_parallel_size = (
            self.tensor_model_parallel_size
        )
        self.text_config.sequence_parallel = self.sequence_parallel
        self.text_config.context_parallel_size = self.context_parallel_size
        self.vision_config.tensor_model_parallel_size = (
            self.tensor_model_parallel_size
        )
        # self.vision_projection_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.text_config.pipeline_model_parallel_size = (
            self.pipeline_model_parallel_size
        )

        if self.encoder_pipeline_model_parallel_size > 0:
            assert self.encoder_pipeline_model_parallel_size == 1, (
                "ViT can only live on 1 pipeline stage."
            )
            self.vision_config.pipeline_model_parallel_size = (
                self.encoder_pipeline_model_parallel_size
            )
            # self.vision_projection_config.pipeline_model_parallel_size = self.encoder_pipeline_model_parallel_size
            self.text_config.encoder_pipeline_model_parallel_size = (
                self.encoder_pipeline_model_parallel_size
            )
            if self.encoder_tensor_model_parallel_size > 0:
                self.vision_config.tensor_model_parallel_size = (
                    self.encoder_tensor_model_parallel_size
                )
                # self.vision_projection_config.tensor_model_parallel_size = self.encoder_tensor_model_parallel_size

        config_attrs = [
            "cross_entropy_loss_fusion",
            "gradient_accumulation_fusion",
            "bias_activation_fusion",
            "bias_dropout_fusion",
            "masked_softmax_fusion",
            "attention_softmax_in_fp32",
            "apply_rope_fusion",
            "overlap_p2p_comm",
            "batch_p2p_comm",
        ]

        for config in [
            self.text_config,
            self.vision_config,
            # self.vision_projection_config,
        ]:
            for attr in config_attrs:
                setattr(config, attr, getattr(self, attr))

        self.text_config.tp_comm_overlap = self.tp_comm_overlap
        self.vision_config.tp_comm_overlap = False
        # self.vision_projection_config.tp_comm_overlap = False

        vp_stage = vp_stage or 0

        model = Qwen3VLModelDist(
            config=self,
            tokenizer=tokenizer,
            pre_process=parallel_state.is_pipeline_first_stage(
                ignore_virtual=False, vp_stage=vp_stage
            )
            or parallel_state.get_pipeline_model_parallel_rank()
            == self.encoder_pipeline_model_parallel_size,
            post_process=parallel_state.is_pipeline_last_stage(
                ignore_virtual=False, vp_stage=vp_stage
            ),
            add_encoder=parallel_state.is_pipeline_first_stage(
                ignore_virtual=False, vp_stage=vp_stage
            ),
            add_decoder=parallel_state.is_pipeline_last_stage(
                ignore_virtual=False, vp_stage=vp_stage
            )
            or parallel_state.get_pipeline_model_parallel_rank()
            >= self.encoder_pipeline_model_parallel_size,
            drop_vision_class_token=self.drop_vision_class_token,
            vp_stage=vp_stage,
        )

        return model

    @classmethod
    def from_config(cls, config):
        res = super().from_config(config)
        res.vision_config = Qwen3VLVisionProvider.from_config(
            config.vision_config
        )
        res.text_config = Qwen3VLTextProvider.from_config(config.text_config)
        res.vision_config.normalization = "LayerNorm"
        res.vision_config.gated_linear_unit = False
        res.text_config.multimodal_embedding = True
        res.text_config.position_embedding_type = "mrope"
        res.text_config.image_token_id = config.image_token_id
        res.text_config.video_token_id = config.video_token_id
        return res


__all__ = [
    "Qwen3VLVisionProvider",
    "Qwen3VLTextProvider",
    "Qwen3VLProvider",
]
