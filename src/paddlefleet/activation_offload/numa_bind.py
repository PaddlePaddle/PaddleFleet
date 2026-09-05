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

"""Bind the process to the CPUs of the NUMA node local to its GPU.

On a multi-socket host, pinned memory that ends up on the socket not directly
attached to the GPU loses a large fraction of the achievable D2H/H2D bandwidth.
Which socket a process lands on is up to the scheduler, so without an explicit
bind the bandwidth a run gets varies between ranks of the same job.

Setting CPU affinity is enough; ``--membind`` is not needed. The default Linux
policy is first-touch local, so the pages behind ``cudaHostAlloc`` follow the
thread that allocates them.

Timing: this must happen before the first pinned allocation. It does not need to
precede CUDA context creation, since page ownership is decided by the affinity
in effect at allocation time -- calling it from ``OffloadManager.__init__`` is
sufficient.

Cost: the process is left with the cores of one node only, which can slow down
dataloader threads, so the caller gates this on
``activation_offload_numa_bind``.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# GPUs attached to each NUMA node. Override on hosts with a different topology.
GPUS_PER_NUMA_ENV = "PADDLEFLEET_GPUS_PER_NUMA"


def _gpus_per_numa() -> int:
    try:
        return max(1, int(os.environ.get(GPUS_PER_NUMA_ENV, "2")))
    except ValueError:
        return 2


def cpus_of_numa_node(node: int) -> list[int]:
    """Read the cpulist of a NUMA node from sysfs (no libnuma/numactl needed)."""
    try:
        with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
            spec = f.read().strip()
    except OSError:
        return []
    cpus: list[int] = []
    for part in spec.split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            cpus.extend(range(int(lo), int(hi) + 1))
        else:
            cpus.append(int(part))
    return cpus


def current_gpu() -> int | None:
    """Physical GPU id this process uses.

    ``paddle.distributed.launch`` sets ``FLAGS_selected_gpus`` per worker; fall
    back to the first entry of ``CUDA_VISIBLE_DEVICES``.
    """
    for var in ("FLAGS_selected_gpus", "CUDA_VISIBLE_DEVICES"):
        value = os.environ.get(var)
        if value:
            head = value.split(",")[0].strip()
            if head.isdigit():
                return int(head)
    return None


def bind(gpu: int | None = None) -> bool:
    """Bind this process to the cores local to ``gpu``. Returns whether it bound.

    Does nothing, quietly, when the GPU id is unknown or sysfs has no cpulist for
    the node: this is a pure performance optimization and a failed topology probe
    must not stop training from starting.
    """
    if gpu is None:
        gpu = current_gpu()
    if gpu is None:
        logger.debug("[activation_offload] numa bind skipped: unknown GPU id")
        return False
    node = gpu // _gpus_per_numa()
    cpus = cpus_of_numa_node(node)
    if not cpus:
        logger.debug(
            "[activation_offload] numa bind skipped: node%d has no cpulist",
            node,
        )
        return False
    try:
        os.sched_setaffinity(0, set(cpus))
    except OSError as exc:  # containers may not permit this
        logger.warning("[activation_offload] numa bind failed: %s", exc)
        return False
    logger.info(
        "[activation_offload] bound to NUMA node%d (gpu=%d, cpus %d-%d, %d "
        "cores) for local host-copy bandwidth",
        node,
        gpu,
        cpus[0],
        cpus[-1],
        len(cpus),
    )
    return True
