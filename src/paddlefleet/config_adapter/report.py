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

"""Change bookkeeping and human-readable reporting.

Every write the adapter performs is recorded together with the reason it was
performed, so the console report can explain *what* changed and *why* instead
of only dumping the final values.  Change lines keep a stable, greppable
shape::

    CHANGE field=<key> old=<value> new=<value>
    ADD    field=<key> new=<value>
    DELETE field=<key> old=<value>

with the reason on the following indented line, and the machine-readable
summary (``ORIGINAL_CARDS=`` / ``TARGET_CARDS=`` / ``OUTPUT=``) at the very
end for upstream scripts.
"""

from __future__ import annotations

from collections import namedtuple
from datetime import date

Change = namedtuple(
    "Change", ["target", "kind", "field", "old", "new", "reason"]
)

_KIND_LABEL = {"modified": "CHANGE", "added": "ADD", "removed": "DELETE"}
_TARGET_LABEL = {"yaml": "YAML", "json": "model_config.json"}


class ChangeLog:
    """Ordered record of every *effective* write, tagged with its reason.

    Writes that leave the value untouched are dropped: a report should only
    list keys that actually changed.
    """

    def __init__(self):
        self.items = []

    def record(self, target, diffs, reason):
        """Record writer diffs (``[(kind, field, old, new), ...]``)."""
        for kind, field, old, new in diffs:
            if kind == "modified" and old == new:
                continue
            self.items.append(Change(target, kind, field, old, new, reason))

    def record_removed(self, target, field, old, reason):
        """Record a deletion the writers cannot express."""
        self.items.append(Change(target, "removed", field, old, None, reason))

    def by_target(self, target):
        """Changes for one document, in the order they were applied."""
        return [item for item in self.items if item.target == target]

    def __len__(self):
        return len(self.items)


def format_header(info):
    """Comment header prepended to the generated YAML."""
    lines = [
        f"# [config_adapter] source: {info['input']}",
        f"# [config_adapter] profile: {info['profile']} "
        f"(batch: {info['batch_strategy']})",
        f"# [config_adapter] scale: {info['orig_cards_label']} cards -> "
        f"{info['target_cards']} cards "
        f"({info['target_nodes']} node(s) x {info['cards_per_node']})",
        f"# [config_adapter] parallelism: {info['dims_line']}",
        f"# [config_adapter] sharding: {info['sharding_line']}",
        f"# [config_adapter] generated: {date.today().isoformat()}",
    ]
    return "\n".join(lines) + "\n\n"


def _format_changes(changes, target, destination):
    """Render one document's change block."""
    label = _TARGET_LABEL[target]
    if not changes:
        return [f"{label} 改动：无"]

    lines = [f"{label} 改动（{len(changes)} 项）-> {destination}"]
    for item in changes:
        kind = _KIND_LABEL[item.kind]
        if item.kind == "modified":
            head = (
                f"  {kind} field={item.field} old={item.old!r} new={item.new!r}"
            )
        elif item.kind == "added":
            head = f"  {kind} field={item.field} new={item.new!r}"
        else:
            head = f"  {kind} field={item.field} old={item.old!r}"
        lines.append(head)
        lines.append(f"      原因：{item.reason}")
    return lines


def format_report(info, changelog):
    """Render the full console report for a successful adaptation."""
    lines = [
        "===== config_adapter: 适配成功 =====",
        f"输入      ：{info['input']}",
        f"模式      ：{info['profile']}（{info['profile_flag']}），"
        f"batch 策略 {info['batch_strategy']}",
        f"机器规模  ：{info['orig_scale_label']} -> "
        f"{info['target_nodes']} 节点 / {info['target_cards']} 卡"
        f"（每节点 {info['cards_per_node']} 卡）",
        f"并行度    ：{info['dims_line']}",
        f"sharding  ：{info['sharding_line']}",
        f"缩容方案  ：{info['plan_note']}",
    ]

    lines += _format_changes(
        changelog.by_target("yaml"), "yaml", info["output"]
    )
    if info.get("model_config_output"):
        lines += _format_changes(
            changelog.by_target("json"),
            "json",
            info["model_config_output"],
        )

    for note in info.get("skipped_switches") or []:
        lines.append(f"跳过的精度开关：{note}")
    for warning in info.get("warnings") or []:
        lines.append(f"WARNING：{warning}")

    lines.append(f"ORIGINAL_CARDS={info['orig_cards_label']}")
    lines.append(f"ORIGINAL_NODES={info['orig_nodes_label']}")
    lines.append(f"TARGET_CARDS={info['target_cards']}")
    lines.append(f"OUTPUT={info['output']}")
    if info.get("model_config_output"):
        lines.append(f"MODEL_CONFIG_OUTPUT={info['model_config_output']}")
    return "\n".join(lines)
