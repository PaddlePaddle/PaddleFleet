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

"""HySparse block-attention TileLang operators (paper arXiv 2602.03560).

MQA/MLA variant: K/V are a single shared head, block selection is aggregated
across the query group by a group-wise maximum, and the sparse branch gathers
only the selected blocks.
"""

from .block_score_attn import (
    block_score_mqa_attn_fwd,
    block_scores_from_logit,
)
from .block_score_attn_bwd import block_score_mqa_bwd_interface
from .block_sparse_attn_mqa import block_sparse_mqa_attn_fwd
from .block_sparse_attn_mqa_bwd import block_sparse_mqa_bwd_interface
from .pipeline import (
    hysparse_forward_mqa,
    select_topk_blocks,
)

__all__ = [
    "block_score_mqa_attn_fwd",
    "block_scores_from_logit",
    "block_score_mqa_bwd_interface",
    "block_sparse_mqa_attn_fwd",
    "block_sparse_mqa_bwd_interface",
    "hysparse_forward_mqa",
    "select_topk_blocks",
]
