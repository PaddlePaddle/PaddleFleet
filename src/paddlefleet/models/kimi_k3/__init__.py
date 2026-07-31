# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from .embedding import (
    KimiK3VisionEmbedding,
    KimiK3VisionEmbeddingSpec,
    build_vision_block_diag_mask,
    build_vision_startend_row_indices,
    merge_vision_block_diag_mask,
)
from .kimi_k3_builders import (
    build_kimi_k3_vision_config,
    kimi_k3_vision_builder,
)
from .kimi_k3_model import KimiK3VisionModel, KimiK3VisionSublayersSpec
from .layer_specs import get_kimi_k3_vision_spec
from .merge_input_ids import merge_input_ids_with_image_features
from .sd2_tpool_merge import (
    KimiK3VisionPatchMerger,
    KimiK3VisionSd2TpoolMerger,
)

__all__ = [
    "KimiK3VisionEmbedding",
    "KimiK3VisionEmbeddingSpec",
    "KimiK3VisionModel",
    "KimiK3VisionPatchMerger",
    "KimiK3VisionSd2TpoolMerger",
    "KimiK3VisionSublayersSpec",
    "build_kimi_k3_vision_config",
    "build_vision_block_diag_mask",
    "build_vision_startend_row_indices",
    "get_kimi_k3_vision_spec",
    "kimi_k3_vision_builder",
    "merge_input_ids_with_image_features",
    "merge_vision_block_diag_mask",
]
