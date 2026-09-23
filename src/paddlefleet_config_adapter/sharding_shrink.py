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

"""Switches that compensate for a smaller sharding degree.

Fewer cards means a smaller ``sharding``, and two things get worse in
proportion to the shrink factor:

* the data stream is split into ``dataset_world_size = sharding / CP`` shards,
  so each rank has to walk a proportionally longer slice of a production file
  list before the first step;
* optimizer states are sharded over the same degree, so per-card optimizer
  memory grows by exactly that factor and a config that fits at full scale can
  OOM on one node.

The training entry point already understands a switch for each:
``debug_reeao_dataset_world_size`` overrides the shard count used for *data
loading only* (the rank stays the real one, so ranks still read distinct
slices), and ``tensorwise_offload_optimizer`` keeps the optimizer states in
host memory.

The offload switch drags in prerequisites, each enforced by a hard framework
error, hence :data:`OFFLOAD_PREREQUISITES`: offload conflicts with
``fuse_optimizer_states``, zero-cost checkpointing requires that same fused
storage, and flash-device saving requires zero-cost checkpointing.  Only the
keys the source actually declares with a conflicting value are rewritten -- the
framework defaults (``false`` / ``0``) are already compatible, and a ``--set``
that turns offload off skips the cascade entirely.

A production YAML often hides its own scale (``global_batch_size`` commented
out, no ``sharding_parallel_size``), and both switches still matter there, so an
underivable source width is assumed to be :data:`DEFAULT_SHRINK_FACTOR` times
the source's own ``dense_sharding`` -- the cards that actually load data --
rather than nothing at all.

All functions here are pure: nothing is written, the caller decides.
"""

from __future__ import annotations

OFFLOAD_KEY = "tensorwise_offload_optimizer"

#: Assumed source/target scale ratio when the source scale is unknown.
#:
#: A production YAML often comments ``global_batch_size`` out and declares no
#: ``sharding_parallel_size``, so the source card count cannot be derived.  The
#: source is then assumed to have run this many times its own ``dense_sharding``
#: (``EP / (TP * SEP)``, the width that actually loads data): that ratio comes
#: from the source's *own* degrees, so EP/PP shrinking cannot deflate it.
DEFAULT_SHRINK_FACTOR = 96

#: ``(key, value_required_by_offload, why)`` in cascade order.
OFFLOAD_PREREQUISITES = (
    (
        "fuse_optimizer_states",
        False,
        "offload 每步把优化器状态搬到 host，fuse 会发现状态不再与 GPU 上的 "
        "fused buffer 共享存储，于是每步重建整块 buffer，显存反而更高，"
        "框架直接报错",
    ),
    (
        "enable_zero_cost_checkpoint",
        False,
        "zero-cost checkpoint 断言 fuse_optimizer_states=true，"
        "而它已被 offload 关闭",
    ),
    (
        "flash_device_save_steps",
        0,
        "flash_device_save_steps>0 断言 enable_zero_cost_checkpoint=true，"
        "而它已被关闭",
    ),
)


def _format_factor(orig_ways, new_ways):
    """``orig_ways / new_ways`` as text: exact ratios stay integers.

    Integer division would round 96 / 64 down to ``1`` and claim the shrink
    changed nothing, so inexact ratios keep one decimal.
    """
    if orig_ways % new_ways == 0:
        return str(orig_ways // new_ways)
    return f"{orig_ways / new_ways:.1f}"


def plan_sharding_shrink_switches(
    config, orig_ways, new_ways, overrides=None, base_ways=None
):
    """Decide the switches a sharding shrink needs.

    ``orig_ways`` / ``new_ways`` are the source and target
    ``dataset_world_size``; a falsy ``orig_ways`` means the source scale could
    not be derived, so :data:`DEFAULT_SHRINK_FACTOR` times ``base_ways`` is
    assumed instead and the reason text says so.  ``base_ways`` is the source's
    own ``dense_sharding``, the width that actually loads data; it is measured
    from the source's degrees rather than the target's because EP/PP shrinking
    would otherwise deflate it, and it falls back to ``new_ways`` when the
    source declares no expert parallel.  ``overrides`` are the user's ``--set``
    values, which own :data:`OFFLOAD_KEY` when they mention it -- turning
    offload off that way must not drag its prerequisites down with it.  Returns
    ``[(key, value, reason), ...]`` for the YAML, empty when the target is no
    narrower than the source.
    """
    if not new_ways:
        return []
    assumed = not orig_ways
    if assumed:
        base_ways = base_ways or new_ways
        orig_ways = base_ways * DEFAULT_SHRINK_FACTOR
    # Nothing to compensate for unless the target is actually narrower, and an
    # assumed source width is no exception: a target wide enough to reach the
    # estimate needs neither a pinned data-split width nor offload.
    if orig_ways <= new_ways:
        return []

    factor = _format_factor(orig_ways, new_ways)
    if assumed:
        shrink = (
            f"源规模推不出来，按源配置实际加载数据的路数（dense_sharding）"
            f"{base_ways} 的 {DEFAULT_SHRINK_FACTOR} 倍"
            f"估计源规模为 {orig_ways} 路"
        )
    else:
        shrink = f"sharding 路数 {orig_ways} -> {new_ways}"
    switches = [
        (
            "debug_reeao_dataset_world_size",
            orig_ways,
            f"{shrink}，每路数据量放大 {factor} 倍、数据加载明显变慢；"
            f"把数据流路数固定回 {orig_ways}"
            f"（只影响数据切分，rank 仍是真实 rank）",
        ),
    ]
    overrides = overrides or {}
    if OFFLOAD_KEY in overrides:
        offload = bool(overrides[OFFLOAD_KEY])
    else:
        offload = True
        switches.append(
            (
                OFFLOAD_KEY,
                True,
                f"{shrink}，单卡优化器状态放大 {factor} 倍，"
                f"offload 到 host 内存防 OOM",
            )
        )
    if not offload:
        return switches
    for key, value, why in OFFLOAD_PREREQUISITES:
        if key in config and config[key] != value:
            switches.append((key, value, f"为开启 {OFFLOAD_KEY}：{why}"))
    return switches
