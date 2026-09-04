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

"""Leaf GLM-5.2 IEEE opt-in. Importing this module must not pull layers or MoE."""

from __future__ import annotations

import os


def ieee_kernel_enabled() -> bool:
    """Opt-in GLM-5.2 IEEE numeric paths on top of FLAG+UAC.

    Fleet CI Minimax / GLM-4.5 Air export FLAGS_use_accuracy_compatible_kernel=1
    and YAML use_accuracy_compatible=true. GLM-5.2 IEEE helpers that used those
    two switches leaked into that graph and moved step-1 vs Megatron. Formal
    and stack-top runners export MODEL_REPRO_IEEE_KERNEL=1; CI does not.

    This module is a leaf so tensor_parallel.layers can import it without
    cycling through transformer.moe.moe_utils.
    """
    return os.environ.get("MODEL_REPRO_IEEE_KERNEL", "0") == "1"
