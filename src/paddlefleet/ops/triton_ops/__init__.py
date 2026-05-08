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

"""
paddlefleet.ops.triton_ops — compatibility shim.

All Triton op implementations have been migrated to paddlefleet_ops.ops.triton_ops.
This module re-exports everything from there so that existing code using
`paddlefleet.ops.triton_ops` continues to work unchanged.

NOTE: paddlefleet.ops replaces itself in sys.modules with paddlefleet_ops.ops,
so Python's subpackage resolution for `paddlefleet.ops.triton_ops` automatically
finds paddlefleet_ops/ops/triton_ops/ on the filesystem. This shim file exists
only as documentation of the migration and is not actually loaded at runtime.
"""

from paddlefleet_ops.ops.triton_ops import (
    MoETopkFusion,
    RMSNormFusionTriton,
    SigmoidGateFusionTriton,
    routing_map_fusion_forward,
)

__all__ = [
    "RMSNormFusionTriton",
    "MoETopkFusion",
    "routing_map_fusion_forward",
    "SigmoidGateFusionTriton",
]
