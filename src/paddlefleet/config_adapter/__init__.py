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

"""Adapt a large-cluster training YAML to a smaller machine scale.

Without any switch the adapter simply makes the config fit the target scale:
it recomputes ``sharding`` / batch and, when the target is incompatible with
the source parallelism, shrinks EP / PP (never below 2) and rewrites a copy of
``model_config.json`` accordingly.

Two orthogonal, optional test dimensions refine that:

* ``--test-performance`` -- freeze every parallel dimension and
  ``gradient_accumulation_steps`` so the step time stays comparable; only
  ``sharding`` and ``global_batch_size`` move.
* ``--test-accuracy`` -- pin the determinism switches in
  :mod:`paddlefleet.config_adapter.precision` so the run does not aadiff, and
  (unless the performance switch froze ``acc``) keep the effective batch.

Usage::

    python -m paddlefleet.config_adapter --input config.yaml \\
        --target-nodes 1 --test-accuracy
"""

from __future__ import annotations

from .cli import build_parser, main, parse_overrides
from .core import ConfigAdapter, inspect_config
from .options import AdaptOptions
from .plan import ParallelismPlan
from .planner import ShrinkPlanner, plan_frozen, plan_parallelism
from .precision import PRECISION_SWITCHES, plan_precision_switches
from .topology import TopologyValidator

__all__ = [
    "PRECISION_SWITCHES",
    "AdaptOptions",
    "ConfigAdapter",
    "ParallelismPlan",
    "ShrinkPlanner",
    "TopologyValidator",
    "build_parser",
    "inspect_config",
    "main",
    "parse_overrides",
    "plan_frozen",
    "plan_parallelism",
    "plan_precision_switches",
]
