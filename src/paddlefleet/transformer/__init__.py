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

<<<<<<< HEAD:src/paddlefleet/transformer/__init__.py
from .layer import FleetLayer as FleetLayer, GraphableFleetLayer
from .spec_utils import LayerSpec
from .transformer_config import TransformerConfig as TransformerConfig

__all__ = [
    "FleetLayer",
    "GraphableFleetLayer",
    "LayerSpec",
    "TransformerConfig",
]
=======
# Referred to NVIDIA Megatron-LM https://github.com/NVIDIA/Megatron-LM.git
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass

from ..fleet_config import FleetConfig
from ..utils import init_method_normal, scaled_init_method_normal


@dataclass
class TransformerConfig(FleetConfig):
    """Configuration object for  transformers."""

    ####################
    # model architecture
    ####################

    num_layers: int = 0
    """Number of transformer layers in a transformer block."""

    hidden_size: int = 0
    """Transformer hidden size."""

    num_attention_heads: int = 0
    """Number of transformer attention heads."""

    use_cpu_initialization: bool = False

    is_hybrid_model: bool = False
    """ Indicates whether this is a hybrid model. """

    ####################
    # initialization
    ####################
    init_method: callable = None
    """Method to initialize weights. Note that bias is always set to zero. Should be a function that
    takes a single Tensor and initializes it. If None, will be set to
    megatron.core.utils.init_method_normal(init_method_std) which is torch nn init normal with
    mean=0.0 and std=init_method_std."""

    output_layer_init_method: callable = None
    """Method to initialize weights of the output layer of both attention and MLP blocks. If None,
    will be set to megatron.core.utils.scaled_init_method_normal(init_method_std) which is torch nn
    init normal with mean=0.0 and std=init_method_std / math.sqrt(2.0 * num_layers)."""

    init_method_std: float = 0.02
    """Standard deviation of the zero mean normal for the default initialization method, not used if
    init_method and output_layer_init_method are provided."""

    init_model_with_meta_device: bool = False
    """
    If True, initializes the model with the meta device. This is helpful for
    training of very large models. This feature is only works when custom fsdp is turned on.
    """

    def __post_init__(self):
        super().__post_init__()

        if self.init_method is None:
            self.init_method = init_method_normal(self.init_method_std)

        if self.output_layer_init_method is None:
            self.output_layer_init_method = scaled_init_method_normal(
                self.init_method_std,
                self.num_layers,
                multiplier=2.0 if not self.is_hybrid_model else 1.0,
            )
>>>>>>> a2bc825 (add base functions and unit tests for tensor parallel):fleet/core/transformer/transformer_config.py
