# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import logging
import sys
from contextlib import suppress
from typing import TYPE_CHECKING
from ..utils.lazy_import import _LazyModule


# from .auto.modeling import AutoModelForCausalLM
import_structure = {
    "model_outputs": ["CausalLMOutputWithPast"],
    "sequence_parallel_utils": [
        "AllGatherVarlenOp",
        "sequence_parallel_sparse_mask_labels",
    ],
    "model_utils": ["PretrainedModel", "register_base_model"],
    "tokenizer_utils": [
        "PretrainedTokenizer",
        "PreTrainedTokenizer",
        "PreTrainedTokenizerBase",
        "PreTrainedTokenizerFast",
        "BPETokenizer",
        "tokenize_chinese_chars",
        "is_chinese_char",
        "normalize_chars",
        "tokenize_special_chars",
        "convert_to_unicode",
        "AddedToken",
    ],
    "tensor_parallel_utils": [],
    "configuration_utils": ["PretrainedConfig"],
    "processing_utils": ["ProcessorMixin"],
    "feature_extraction_utils": ["BatchFeature", "FeatureExtractionMixin"],
    "image_processing_utils": ["PaddleImageProcessingMixin", "ImageProcessingMixin", "BaseImageProcessor"],
    "image_processing_utils_fast": ["BaseImageProcessorFast"],
    "video_processing_utils": ["BaseVideoProcessor"],
    "audio_processing_utils": ["SequenceFeatureExtractor"],
    "moe_gate": ["PretrainedMoEGate", "MoEGateMixin"],
    "auto.configuration": ["AutoConfig"],
    "auto.image_processing": ["AutoImageProcessor", "IMAGE_PROCESSOR_MAPPING"],
    "auto.modeling": [
        "AutoTokenizer",
        "AutoBackbone",
        "AutoModel",
        "AutoModelForPretraining",
        "AutoModelForSequenceClassification",
        "AutoModelForTokenClassification",
        "AutoModelForQuestionAnswering",
        "AutoModelForMultipleChoice",
        "AutoModelForMaskedLM",
        "AutoModelForCausalLMPipe",
        "AutoEncoder",
        "AutoDecoder",
        "AutoGenerator",
        "AutoDiscriminator",
        "AutoModelForConditionalGeneration",
        "AutoModelForConditionalGenerationPipe",
    ],
    "tokenizer_utils_base": [
        "PaddingStrategy",
        "PreTokenizedInput",
        "TextInput",
        "TensorType",
        "TruncationStrategy",
    ],
    "auto.processing": ["AutoProcessor"],
    "auto.tokenizer": ["AutoTokenizer", "TOKENIZER_MAPPING"],
    "auto.video_processing": ["AutoVideoProcessor", "VIDEO_PROCESSOR_MAPPING"],
    "auto.feature_extraction": ["AutoFeatureExtractor"],
    "deepseek_v32.configuration": ["DeepseekV32Config"],
    "deepseek_v32.modeling": [
        "DeepseekV32ForCausalLM",
        "DeepseekV32ForCausalLMPipe",
    ],
    "kimi_k25.vision_processor": ["KimiK25VisionProcessor"],
    "kimi_k25.processor": ["KimiK25Processor"],
    "kimi_k25.tokenizer": ["TikTokenTokenizer"],
    "kimi_k2.configuration": ["KimiK2Config"],
    "kimi_k2.modeling": ["KimiK2ForCausalLM", "KimiK2ForCausalLMPipe"],
    "kimi_k2.tokenizer": ["KimiK2TikTokenTokenizer"],
    "kimi_k3.configuration": [
        "KimiK3Config",
        "KimiK3TextConfig",
        "KimiK3VisionConfig",
    ],
    "kimi_k3.modeling": [
        "KimiK3Model",
        "KimiK3ForCausalLM",
        "KimiK3ForCausalLMPipe",
        "KimiK3ForConditionalGeneration",
        "KimiK3ModelProvider",
    ],
    "kimi_k3.tokenizer": ["KimiK3TikTokenTokenizer"],
    "kimi_k3.vision_processor": ["KimiK3VisionProcessor"],
    "kimi_k3.processor": ["KimiK3Processor"],
    "optimization": [
        "LinearDecayWithWarmup",
        "ConstScheduleWithWarmup",
        "CosineDecayWithWarmup",
        "PolyDecayWithWarmup",
        "CosineAnnealingWithWarmupDecay",
        "LinearAnnealingWithWarmupDecay",
    ],
    "qwen2.configuration": ["Qwen2Config"],
    "qwen2.modeling": [
        "Qwen2Model",
        "Qwen2PretrainedModel",
        "Qwen2ForCausalLM",
        "Qwen2ForCausalLMPipe",
        "Qwen2PretrainingCriterion",
        "Qwen2ForSequenceClassification",
        "Qwen2ForTokenClassification",
        "Qwen2SentenceEmbedding",
        "Qwen2ForCausalLMDeprecated",
        "Qwen2ForCausalLMPipeDeprecated",
    ],
    "qwen2.tokenizer": ["Qwen2Tokenizer"],
    "qwen2.tokenizer_fast": ["Qwen2TokenizerFast"],
    "qwen3_5.configuration": ["Qwen3_5VisionConfig"],
    "qwen3_5.modeling": ["Qwen3_5VisionModel"],
    "qwen3_vl.configuration": ["Qwen3VLConfig", "Qwen3VLTextConfig"],
    "qwen3_vl.modeling": [
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLForConditionalGenerationDeprecated",
        "Qwen3VLModel",
        "Qwen3VLModelDeprecated",
        "Qwen3VLPretrainedModel",
        "Qwen3VLTextModel",
        "Qwen3VLModelFleet",
    ],
    "qwen3_vl.processor": ["Qwen3VLProcessor"],
    "qwen3_vl.video_processor": ["Qwen3VLVideoProcessor"],
    "qwen3_vl_moe.configuration": ["Qwen3VLMoeConfig", "Qwen3VLMoeTextConfig"],
    "qwen3_vl_moe.modeling": [
        "Qwen3VLMoeForConditionalGeneration",
        "Qwen3VLMoeForConditionalGenerationDeprecated",
        "Qwen3VLMoeModelDeprecated",
        "Qwen3VLMoeModel",
        "Qwen3VLMoePretrainedModel",
        "Qwen3VLMoeTextModel",
    ],
    "qwen3_omni_moe.configuration": ["Qwen3OmniMoeConfig", "Qwen3OmniMoeThinkerConfig", "Qwen3OmniMoeTextConfig"],
    "qwen3_omni_moe.modeling": [
        "Qwen3OmniMoeForConditionalGeneration",
        "Qwen3OmniMoeThinkerForConditionalGeneration",
        "Qwen3OmniMoePreTrainedModel",
        "Qwen3OmniMoeThinkerTextPreTrainedModel",
        "Qwen3OmniMoeThinkerTextModel",
        "Qwen3OmniMoeTalkerModel",
    ],
    "qwen3_omni_moe.processor": ["Qwen3OmniMoeProcessor"],
    "qwen3_omni_moe.feature_extractor": ["WhisperFeatureExtractor"],
    "qwen2_moe.configuration": ["Qwen2MoeConfig"],
    "qwen2_moe.modeling": [
        "Qwen2MoeModel",
        "Qwen2MoePretrainedModel",
        "Qwen2MoeForCausalLM",
        "Qwen2MoeForCausalLMPipe",
        "Qwen2MoePretrainingCriterion",
        "Qwen2MoeForCausalLMDeprecated",
        "Qwen2MoeForCausalLMPipeDeprecated",
    ],
    "qwen2_vl.image_processor": ["Qwen2VLImageProcessor"],
    "qwen2_vl.image_processor_fast": ["Qwen2VLImageProcessorFast"],
    "qwen2_vl.processor": ["Qwen2VLProcessor"],
    "qwen2_vl.video_processor": ["Qwen2VLVideoProcessor"],
    "qwen2_vl.vision_process": ["process_vision_info"],
    "qwen3.configuration": ["Qwen3Config"],
    "qwen3.modeling": [
        "Qwen3Model",
        "Qwen3PretrainedModel",
        "Qwen3ForCausalLM",
        "Qwen3ForCausalLMPipe",
        "Qwen3PretrainingCriterion",
        "Qwen3ForSequenceClassification",
        "Qwen3ForTokenClassification",
        "Qwen3SentenceEmbedding",
        "Qwen3ForCausalLMDeprecated",
        "Qwen3ForCausalLMPipeDeprecated",
    ],
    "qwen3_moe.configuration": ["Qwen3MoeConfig"],
    "qwen3_moe.modeling": [
        "Qwen3MoeModel",
        "Qwen3MoePretrainedModel",
        "Qwen3MoeForCausalLM",
        "Qwen3MoeForCausalLMPipe",
        "Qwen3MoePretrainingCriterion",
        "Qwen3MoeForCausalLMDeprecated",
    ],
    "qwen2": [],
    "qwen3": [],
    "qwen3_vl": [],
    "qwen3_5": [],
    "qwen3_vl_moe": [],
    "qwen2_moe": [],
    "qwen2_vl": [],
    "qwen3_moe": [],
    "glm4_moe.configuration": ["Glm4MoeConfig"],
    "whisper.processor": ["WhisperFeatureExtractor"],
    "glm4_moe": ["Glm4MoeForCausalLMPipe", "Glm4MoeModel", "Glm4MoeForCausalLM", "Glm4MoeForCausalLMDeprecated"],
    "glm_moe_dsa.configuration": ["GlmMoeDsaConfig"],
    "glm_moe_dsa": ["GlmMoeDsaForCausalLMPipe", "GlmMoeDsaForCausalLM"],
    "minimax_m2.configuration": ["MiniMaxM2Config"],
    "minimax_m2": ["MiniMaxM2ForCausalLMPipe", "MiniMaxM2ForCausalLM"],
    "deepseek_v4.configuration": ["DeepseekV4Config"],
    "deepseek_v4": ["DeepseekV4ForCausalLMPipe", "DeepseekV4ForCausalLM"],
    "auto": ["AutoModelForCausalLM"],
    "legacy.tokenizer_utils_base": ["EncodingFast"],
    "legacy": [],
    "gemma4_moe.configuration": ["Gemma4MoeConfig"],
    "gemma4_moe.modeling": ["Gemma4MoeForCausalLM"],
    "gemma4_moe": [],
}

if TYPE_CHECKING:
    from .configuration_utils import PretrainedConfig
    from .model_utils import PretrainedModel, register_base_model
    from .tokenizer_utils import (
        PretrainedTokenizer,
        PreTrainedTokenizer,
        PreTrainedTokenizerBase,
        PreTrainedTokenizerFast,
        BPETokenizer,
        tokenize_chinese_chars,
        is_chinese_char,
        AddedToken,
        normalize_chars,
        tokenize_special_chars,
        convert_to_unicode,
    )
    from .processing_utils import ProcessorMixin
    from .feature_extraction_utils import BatchFeature, FeatureExtractionMixin
    from .audio_processing_utils import SequenceFeatureExtractor
    from .image_processing_utils import PaddleImageProcessingMixin, ImageProcessingMixin, BaseImageProcessor
    from .image_processing_utils_fast import BaseImageProcessorFast
    from .video_processing_utils import BaseVideoProcessor
    from .sequence_parallel_utils import AllGatherVarlenOp, sequence_parallel_sparse_mask_labels
    from .tensor_parallel_utils import parallel_matmul, fused_head_and_loss_fn
    from .moe_gate import *

    with suppress(Exception):
        from paddle.distributed.fleet.utils.sequence_parallel_utils import (
            GatherOp,
            ScatterOp,
            AllGatherOp,
            ReduceScatterOp,
            ColumnSequenceParallelLinear,
            RowSequenceParallelLinear,
            mark_as_sequence_parallel_parameter,
            register_sequence_parallel_allreduce_hooks,
        )

    # isort: split
    from .auto.configuration import *
    from .auto.image_processing import *
    from .auto.modeling import *
    from .auto.processing import *
    from .auto.tokenizer import *
    from .auto.video_processing import *
    from .kimi_k25 import *
    from .kimi_k2 import *
    from .kimi_k3 import *
    from .optimization import *
    from .qwen2 import *
    from .qwen2_moe import *
    from .qwen2_vl import *
    from .qwen3 import *
    from .qwen3_moe import *
    from .qwen3_vl import *
    from .qwen3_5 import *
    from .qwen3_vl_moe import *
    from .qwen3_omni_moe import *
    from .glm4_moe import *
    from .glm_moe_dsa import *
    from .minimax_m2 import *
    from .deepseek_v4 import *
    from .gemma4_moe import *
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )

logging.getLogger("transformers").addFilter(
    lambda record: "None of PyTorch, TensorFlow >= 2.0, or Flax have been found." not in str(record.getMessage())
)
