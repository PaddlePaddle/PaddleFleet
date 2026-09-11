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

"""The PaddleFleet side of the HyperBody decoder: **components only**.

| Side | Contents |
|---|---|
| **PaddleFleet (this package)** | ``layer_specs.py`` -- the ``LayerSpec`` list / block spec for the backbone |
| PaddleFormers ``transformers/hyperbody_decoder/`` | HF-style config -> ``GPTConfig`` conversion + model assembly + top-level Model |

Geometry constants and ``GPTConfig`` assembly do **not** live here -- they belong to
the "HF-style config to GPTConfig conversion" responsibility, which is entirely
collected in the PaddleFormers ``configuration.py`` / ``modeling.py``.
"""

from .layer_specs import (
    get_hyperbody_decoder_block_spec,
    get_hyperbody_decoder_layer_specs,
)

__all__ = [
    "get_hyperbody_decoder_block_spec",
    "get_hyperbody_decoder_layer_specs",
]
