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

"""HyperEncoder components that live in PaddleFleet: spec builders and the
model-specific layers.

Config assembly and the top-level model live in PaddleFormers under
``transformers/hyperencoder/`` (following the same split as
``transformers/qwen3_vl``; see the module docstrings there for details).
"""

from .layer_specs import (
    get_hyperencoder_block_spec,
    get_hyperencoder_layer_specs,
)
from .modality_encoders import (
    AudioEncoderConv,
    ImageEncoderConv,
    MlpProjector,
    PatchEmbed,
    get_abs_pos_1d,
    get_abs_pos_2d,
)

__all__ = [
    "get_hyperencoder_block_spec",
    "get_hyperencoder_layer_specs",
    "AudioEncoderConv",
    "ImageEncoderConv",
    "MlpProjector",
    "PatchEmbed",
    "get_abs_pos_1d",
    "get_abs_pos_2d",
]
