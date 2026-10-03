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


"""Periodic DSA indexer layout shared by runtime and checkpoint mapping."""


def is_dsa_skip_topk_layer(
    layer_number: int, skip_topk_offset: int, topk_freq: int
) -> bool:
    """Return whether a 1-indexed layer reuses a previous DSA top-k result."""
    if layer_number < 1:
        raise ValueError(
            f"layer_number must be 1-indexed and positive, got {layer_number}."
        )
    if skip_topk_offset < 0:
        raise ValueError(
            f"skip_topk_offset must be non-negative, got {skip_topk_offset}."
        )
    if topk_freq < 1:
        raise ValueError(f"topk_freq must be positive, got {topk_freq}.")
    skip_topk_offset = max(skip_topk_offset, 1)
    return (max(layer_number - skip_topk_offset, 0) % topk_freq) != 0


def source_dsa_compute_layer(
    layer_number: int, skip_topk_offset: int, topk_freq: int
) -> int:
    """Return the computing layer whose DSA top-k a skip layer reuses."""
    is_dsa_skip_topk_layer(layer_number, skip_topk_offset, topk_freq)
    skip_topk_offset = max(skip_topk_offset, 1)
    if layer_number <= skip_topk_offset:
        return layer_number
    return layer_number - ((layer_number - skip_topk_offset) % topk_freq)
