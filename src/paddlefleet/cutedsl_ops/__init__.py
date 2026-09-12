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

"""Optional CuTe DSL operators for PaddleFleet.

The CuTe DSL dependency is intentionally imported lazily.  Importing
``paddlefleet`` must continue to work on installations that do not include
CUDA Python and CuTe DSL.
"""


def indexer_topk_prefill(*args, **kwargs):
    """Lazily dispatch to the standalone Paddle CuTe DSL operator."""
    from .indexer_topk_prefill import indexer_topk_prefill as _impl

    return _impl(*args, **kwargs)


def precompile_indexer_topk_clc(*args, **kwargs):
    """Lazily precompile the process-local CLC top-k variants."""
    from .indexer_topk_prefill import (
        precompile_indexer_topk_clc as _impl,
    )

    return _impl(*args, **kwargs)


__all__ = [
    "indexer_topk_prefill",
    "precompile_indexer_topk_clc",
]
