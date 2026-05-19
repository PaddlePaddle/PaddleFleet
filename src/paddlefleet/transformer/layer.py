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

from __future__ import annotations

from typing import TYPE_CHECKING

import paddle

if TYPE_CHECKING:
    from paddlefleet.transformer.transformer_config import TransformerConfig


def build_spec_layer(spec, **kwargs):
    """Build a layer from a sublayer spec.

    If spec is None, returns None.
    If spec is a callable (class or factory), it is called with the provided kwargs.
    """
    if spec is None:
        return None
    if callable(spec):
        return spec(**kwargs)
    return None


class FleetLayer(paddle.nn.Layer):
    """Base Fleet Layer inherited by all Models.
    Args:
        config (TransformerConfig): Transformer config
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
