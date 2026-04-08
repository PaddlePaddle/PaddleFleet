#!/usr/bin/env python3
"""
PP stage / VPP chunk 划分辅助函数。

统一约束：
1. 默认均分时，余数优先分配给前面的 stage/chunk
2. 自定义 stage_layer_counts 仅支持 VPP=1 的连续 stage 划分
3. layer 划分始终覆盖 [0, num_hidden_layers) 且不重叠
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple


def has_custom_stage_layer_counts(parallel) -> bool:
    return bool(getattr(parallel, "stage_layer_counts", []) or [])


def build_balanced_partition_counts(total_items: int, partition_count: int) -> List[int]:
    total = max(0, int(total_items))
    parts = max(1, int(partition_count))
    base, remainder = divmod(total, parts)
    return [base + (1 if idx < remainder else 0) for idx in range(parts)]


def _normalize_explicit_stage_layer_counts(
    total_layers: int,
    stage_count: int,
    stage_layer_counts: Sequence[int],
) -> List[int]:
    counts = [int(value) for value in stage_layer_counts]
    if len(counts) != int(stage_count):
        raise ValueError(
            f"stage_layer_counts 长度必须等于 pp={stage_count}，当前为 {len(counts)}"
        )
    if any(value <= 0 for value in counts):
        raise ValueError("stage_layer_counts 中每个 stage 的层数都必须 > 0")
    if sum(counts) != int(total_layers):
        raise ValueError(
            "stage_layer_counts 总和必须等于模型层数："
            f"sum={sum(counts)} num_hidden_layers={total_layers}"
        )
    return counts


def resolve_stage_layer_counts(total_layers: int, parallel) -> List[int]:
    stage_count = max(1, int(getattr(parallel, "pp", 1)))
    explicit_counts = getattr(parallel, "stage_layer_counts", []) or []
    if explicit_counts:
        if int(getattr(parallel, "vpp", 1)) > 1:
            raise ValueError("自定义 stage_layer_counts 暂不支持与 vpp>1 同时使用")
        return _normalize_explicit_stage_layer_counts(
            total_layers, stage_count, explicit_counts
        )
    return build_balanced_partition_counts(total_layers, stage_count)


def partition_counts_to_ranges(partition_counts: Sequence[int]) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    start = 0
    for count in partition_counts:
        width = max(0, int(count))
        end = start + width
        ranges.append((start, end))
        start = end
    return ranges


def resolve_chunk_ranges(total_layers: int, parallel) -> List[Tuple[int, int]]:
    pp = max(1, int(getattr(parallel, "pp", 1)))
    vpp = max(1, int(getattr(parallel, "vpp", 1)))
    if has_custom_stage_layer_counts(parallel):
        return partition_counts_to_ranges(resolve_stage_layer_counts(total_layers, parallel))
    chunk_count = max(1, pp * vpp)
    return partition_counts_to_ranges(
        build_balanced_partition_counts(total_layers, chunk_count)
    )


def resolve_stage_chunk_ranges(total_layers: int, parallel) -> List[List[Tuple[int, int]]]:
    pp = max(1, int(getattr(parallel, "pp", 1)))
    chunk_ranges = resolve_chunk_ranges(total_layers, parallel)
    if has_custom_stage_layer_counts(parallel):
        return [[chunk_ranges[stage_idx]] for stage_idx in range(pp)]
    return [chunk_ranges[stage_idx::pp] for stage_idx in range(pp)]


def resolve_stage_layer_indices(total_layers: int, parallel, stage_id: int) -> List[int]:
    layer_indices: List[int] = []
    for chunk_start, chunk_end in resolve_stage_chunk_ranges(total_layers, parallel)[stage_id]:
        layer_indices.extend(range(chunk_start, chunk_end))
    return layer_indices
