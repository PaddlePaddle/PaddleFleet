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

from .ernie_patch import patch_ernie_attention
from .llama_patch import patch_llama_attention
from .qwen_patch import patch_qwen_attention
from .rrattention import (
    RRAttnConfig,
    get_rrattn_config,
    rrattn_estimate,
    rrattn_prefill,
)

__all__ = [
    "rrattn_estimate",
    "rrattn_prefill",
    "patch_llama_attention",
    "patch_qwen_attention",
    "patch_ernie_attention",
    "RRAttnConfig",
    "get_rrattn_config",
]
