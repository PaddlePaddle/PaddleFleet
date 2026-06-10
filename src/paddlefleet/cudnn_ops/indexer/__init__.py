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

__all__ = [
    "csa_indexer_bwd",
    "cudnn_indexer_forward",
    "cudnn_indexer_topk",
    "cudnn_indexer_topk_fwd",
]


def __getattr__(name):
    if name == "csa_indexer_bwd":
        from .csa_indexer_bwd_cudnn import csa_indexer_bwd

        globals()[name] = csa_indexer_bwd
        return csa_indexer_bwd
    if name in {
        "cudnn_indexer_forward",
        "cudnn_indexer_topk",
        "cudnn_indexer_topk_fwd",
    }:
        from . import cudnn_indexer

        obj = getattr(cudnn_indexer, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
