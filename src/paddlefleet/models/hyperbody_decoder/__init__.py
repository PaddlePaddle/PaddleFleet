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

"""HyperBody decoder 在 PaddleFleet 侧的部分：**只提供组件**。

分工与参考实现 ``fleet_formers`` 的 hyperencoder 一致：

| 侧 | 内容 |
|---|---|
| **PaddleFleet（本包）** | ``layer_specs.py`` —— 12 层 backbone 的 ``LayerSpec`` 列表 / block spec |
| PaddleFormers ``transformers/hyperbody_decoder/`` | HF-style config → ``GPTConfig`` 的转换 + 组网装配 + 顶层 Model |

几何常量与 ``GPTConfig`` 的装配**不在这里** —— 它们属于「HF-style config 到
GPTConfig 的转换」这一职责，全部收在 formers 的 ``configuration.py`` /
``modeling.py``。这与 hyperencoder 那边 ``build_hyperencoder_config`` 位于
``PaddleFormers/paddleformers/transformers/hyperencoder/configuration.py``
是同一个约定。
"""

from .layer_specs import (
    get_hyperbody_decoder_block_spec,
    get_hyperbody_decoder_layer_specs,
)

__all__ = [
    "get_hyperbody_decoder_block_spec",
    "get_hyperbody_decoder_layer_specs",
]
