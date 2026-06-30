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

import paddle

paddle.enable_compat(scope={"tilelang"}, silent=True)

__all__ = [
    "csa_attn_target_reducesum",
    "csa_indexer_bwd",
    "csa_indexer_topk_fwd",
    "csa_sparse_attn",
    "dsa_indexer_bwd_interface",
    "dsa_indexer_topk_reducesum_interface",
    "dsa_prepare_varlen_metadata",
    "dsa_sparse_mla_bwd_interface",
    "dsa_sparse_mla_fwd_interface",
    "dsa_sparse_mla_topk_reducesum_interface",
]


def __getattr__(name):
    if name in {
        "csa_attn_target_reducesum",
        "csa_indexer_bwd",
        "csa_indexer_topk_fwd",
    }:
        from .indexer.csa_indexer import (
            csa_attn_target_reducesum,
            csa_indexer_bwd,
            csa_indexer_topk_fwd,
        )

        exports = {
            "csa_attn_target_reducesum": csa_attn_target_reducesum,
            "csa_indexer_bwd": csa_indexer_bwd,
            "csa_indexer_topk_fwd": csa_indexer_topk_fwd,
        }
        globals().update(exports)
        return exports[name]
    if name == "csa_sparse_attn":
        from paddlefleet.fusions.csa_sparse_attn import csa_sparse_attn

        globals()[name] = csa_sparse_attn
        return csa_sparse_attn
    if name in {
        "dsa_indexer_bwd_interface",
        "dsa_indexer_topk_reducesum_interface",
        "dsa_prepare_varlen_metadata",
    }:
        from .indexer.dsa_indexer import (
            dsa_indexer_bwd_interface,
            dsa_indexer_topk_reducesum_interface,
            dsa_prepare_varlen_metadata,
        )

        exports = {
            "dsa_indexer_bwd_interface": dsa_indexer_bwd_interface,
            "dsa_indexer_topk_reducesum_interface": dsa_indexer_topk_reducesum_interface,
            "dsa_prepare_varlen_metadata": dsa_prepare_varlen_metadata,
        }
        globals().update(exports)
        return exports[name]
    if name in {
        "dsa_sparse_mla_bwd_interface",
        "dsa_sparse_mla_fwd_interface",
        "dsa_sparse_mla_topk_reducesum_interface",
    }:
        from .attn.sparse_mla import (
            dsa_sparse_mla_bwd_interface,
            dsa_sparse_mla_fwd_interface,
            dsa_sparse_mla_topk_reducesum_interface,
        )

        exports = {
            "dsa_sparse_mla_bwd_interface": dsa_sparse_mla_bwd_interface,
            "dsa_sparse_mla_fwd_interface": dsa_sparse_mla_fwd_interface,
            "dsa_sparse_mla_topk_reducesum_interface": dsa_sparse_mla_topk_reducesum_interface,
        }
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
