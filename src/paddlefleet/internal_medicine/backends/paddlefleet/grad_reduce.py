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

"""Exact cross-rank norms for grad_health: plan the reduction, then check it.

Split out of ``grad_monitor`` because the hard part is not the arithmetic but
deciding *which* process groups may be summed. That depends on the topology
order, not on the parallel sizes, so it gets its own module and its own tests.

History, because it is the reason for every check below: the first version
chained the reductions -- sum over ``cp``, then keep going on the same buffer
over ``dp`` -- which silently assumes the two rank sets are orthogonal. On a
``sharding_first`` topology with ``data_parallel_size=1`` the samples live on the
sharding dimension, so ``_DATA_PARALLEL_GROUP`` spans ``cp`` and the
context-parallel contribution was counted twice: ``norm_global`` came out
``sqrt(cp)`` high (+41% at ``cp=2``) with nothing in the numbers to show it. The
reducer's own log line already contained the contradiction (``cp=2 x dp=32`` on a
32-rank job), it just was not checked. Chaining was only ever correct by luck.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Groups that split *one tensor* across ranks. ``tp`` counts only under sequence
# parallel: without SP the very same activations are replicated across the TP
# group, so summing would multiply every squared norm by ``tp_size``.
SHARD_CANDIDATES = ("cp", "tp")

# Groups that may carry *distinct data*. ``ep`` is a candidate rather than an
# assumed duplicate: whether it is nested inside the data dimension is a property
# of the topology order, and the orthogonality check answers that per run instead
# of per assumption.
DATA_CANDIDATES = ("dp", "cp_dp", "sharding", "expt_dp", "ep")


class GroupInfo:
    """One process group plus the metadata the plan needs."""

    __slots__ = ("name", "group", "size", "ranks")

    def __init__(self, name, group, size, ranks):
        self.name = name
        self.group = group
        self.size = int(size)
        self.ranks = frozenset(int(r) for r in ranks)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.name}={self.size}"


def is_orthogonal(candidate: GroupInfo, chosen: list[GroupInfo]) -> bool:
    """Whether ``candidate`` adds new ranks instead of re-counting chosen ones.

    Two mesh groups through the same rank are orthogonal exactly when they
    intersect in that one rank. Anything wider is an overlap, and summing over
    both would count the shared ranks twice -- the bug this module exists to
    prevent. An empty rank list (a backend that does not expose membership) is
    treated as *not* orthogonal, so an unverifiable layout degrades to a partial
    cover rather than to a silently wrong number.
    """
    if not candidate.ranks:
        return False
    for other in chosen:
        if not other.ranks or len(candidate.ranks & other.ranks) != 1:
            return False
    return True


def plan_total_groups(
    candidates: list[GroupInfo], target: int
) -> tuple[list[GroupInfo], int]:
    """Pick groups whose sizes multiply to ``target``, covering each rank once.

    Largest first: a single group that already spans the whole data dimension
    covers everything in one reduction, and taking it first stops a smaller
    nested group (``cp`` inside a sharding-first ``dp``) from being added on top.
    Returns the chosen groups and the product actually achieved, so the caller can
    compare it against ``target`` and decide what to suppress.
    """
    chosen: list[GroupInfo] = []
    product = 1
    for candidate in sorted(candidates, key=lambda info: -info.size):
        if candidate.size <= 1 or product * candidate.size > target:
            continue
        if not is_orthogonal(candidate, chosen):
            continue
        chosen.append(candidate)
        product *= candidate.size
        if product == target:
            break
    return chosen, product


def describe(groups: list[GroupInfo]) -> str:
    return (
        " x ".join(f"{info.name}={info.size}" for info in groups) or "nothing"
    )
