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

"""The planning result shared by both planners."""

from __future__ import annotations


class ParallelismPlan:
    """Outcome of parallelism planning.

    Attributes:
        tp, pp, ep, cp, sep: the FINAL parallel dims used downstream by
            validation, sharding computation and reporting.
        yaml_changes: ``[(key, value, reason), ...]`` writes for the YAML.
        json_changes: ``[(key, value, reason), ...]`` writes for the
            companion ``model_config.json``.
        warnings: user-facing warnings raised while planning.
        note: one-line description of what the plan did, for the report.
    """

    def __init__(
        self,
        tp,
        pp,
        ep,
        cp,
        sep,
        yaml_changes=None,
        json_changes=None,
        warnings=None,
        note="",
    ):
        self.tp = tp
        self.pp = pp
        self.ep = ep
        self.cp = cp
        self.sep = sep
        self.yaml_changes = list(yaml_changes or [])
        self.json_changes = list(json_changes or [])
        self.warnings = list(warnings or [])
        self.note = note

    def dims(self):
        """Return ``(tp, pp, ep, cp, sep)``."""
        return self.tp, self.pp, self.ep, self.cp, self.sep
