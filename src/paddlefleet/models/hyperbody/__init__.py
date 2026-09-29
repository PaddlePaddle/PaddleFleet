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

"""HyperBody components that live in PaddleFleet: spec builders and the
model-specific layers for the unified HyperBody model.

The **encoder-trunk** pieces (fp32 RMSNorm, attention-backend selection,
image / audio modality towers and the encoder ``LayerSpec`` builders) are shared
with HyperEncoder and imported directly from ``models/hyperencoder`` instead of
being duplicated here. Only the **decoder-backbone** specs are HyperBody-specific
and defined locally in ``decoder_layer_specs.py``.

Config assembly and the top-level unified model live under
``transformers/hyperbody/`` (following the same split as ``transformers/qwen3_vl``).
"""

from paddlefleet.models.hyperencoder.attn_backend import (
    encoder_attn_backend,
    use_packed_decoder,
    use_triton_encoder_attn,
)
from paddlefleet.models.hyperencoder.layer_specs import (
    get_hyperencoder_block_spec as get_hyperbody_encoder_block_spec,
    get_hyperencoder_layer_specs as get_hyperbody_encoder_layer_specs,
)
from paddlefleet.models.hyperencoder.modality_encoders import (
    AudioEncoderConv,
    ImageEncoderConv,
    MlpProjector,
    PatchEmbed,
    get_abs_pos_1d,
    get_abs_pos_2d,
)
from paddlefleet.models.hyperencoder.norm import HyperEncoderRMSNorm

from .decoder_layer_specs import (
    get_hyperbody_decoder_block_spec,
    get_hyperbody_decoder_layer_specs,
)

__all__ = [
    "encoder_attn_backend",
    "use_packed_decoder",
    "use_triton_encoder_attn",
    "get_hyperbody_decoder_block_spec",
    "get_hyperbody_decoder_layer_specs",
    "get_hyperbody_encoder_block_spec",
    "get_hyperbody_encoder_layer_specs",
    "AudioEncoderConv",
    "ImageEncoderConv",
    "MlpProjector",
    "PatchEmbed",
    "get_abs_pos_1d",
    "get_abs_pos_2d",
    "HyperEncoderRMSNorm",
]
