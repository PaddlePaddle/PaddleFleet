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

The reasons are full sentences and the values can be whole per-layer lists, so
everything is wrapped to :data:`LINE_WIDTH` with a hanging indent and the
report is split into ruled sections -- an unwrapped report turns into a wall of
text the moment a terminal folds those lines.  Wrapping is done here rather
than with :mod:`textwrap` because Chinese prose has no spaces to break on.
"""

from __future__ import annotations

import unicodedata
from collections import namedtuple
from datetime import date

Change = namedtuple(
    "Change", ["target", "kind", "field", "old", "new", "reason"]
)

#: Console width the report wraps to.
LINE_WIDTH = 88

_KIND_LABEL = {"modified": "CHANGE", "added": "ADD", "removed": "DELETE"}
_TARGET_LABEL = {"yaml": "YAML", "json": "model_config.json"}
_WIDE = frozenset("WF")


def _width(text):
    """Terminal columns ``text`` occupies: CJK glyphs take two."""
    return sum(
        2 if unicodedata.east_asian_width(ch) in _WIDE else 1 for ch in text
    )


def _chunks(text):
    """Split ``text`` into unbreakable pieces.

    A CJK glyph and a space are each their own chunk, so a line may break
    between them; a run of narrow characters (an identifier, a number, a path)
    stays in one piece.
    """
    pieces, word = [], ""
    for ch in text:
        if ch == " " or unicodedata.east_asian_width(ch) in _WIDE:
            if word:
                pieces.append(word)
                word = ""
            pieces.append(ch)
        else:
            word += ch
    if word:
        pieces.append(word)
    return pieces


def _wrap(text, first_prefix="", cont_prefix=None):
    """``text`` as prefixed lines no wider than :data:`LINE_WIDTH`.

    Continuation lines get ``cont_prefix`` (spaces as wide as ``first_prefix``
    by default), which keeps a wrapped reason visually attached to its field.
    """
    if cont_prefix is None:
        cont_prefix = " " * _width(first_prefix)
    lines, prefix = [], first_prefix
    limit = LINE_WIDTH - _width(prefix)
    line, used = "", 0
    for chunk in _chunks(text):
        size = _width(chunk)
        if line and used + size > limit:
            lines.append((prefix + line).rstrip())
            prefix = cont_prefix
            limit = LINE_WIDTH - _width(prefix)
            line, used = "", 0
            if chunk == " ":
                continue
        line += chunk
        used += size
    lines.append((prefix + line).rstrip())
    return lines


def _rule(title=None):
    """A full-width ``-`` rule, optionally naming the section it opens."""
    if title is None:
        return "-" * LINE_WIDTH
    lead = f"--- {title} "
    return lead + "-" * max(LINE_WIDTH - _width(lead), 3)


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


def _format_change(item):
    """One change as its ``CHANGE``/``ADD``/``DELETE`` line plus reason."""
    kind = _KIND_LABEL[item.kind]
    values = []
    if item.kind != "added":
        values.append(f"old={item.old!r}")
    if item.kind != "removed":
        values.append(f"new={item.new!r}")
    head = f"  {kind} field={item.field} " + " ".join(values)
    if _width(head) <= LINE_WIDTH:
        lines = [head]
    else:
        # A per-layer list does not fit next to its field name.
        lines = [f"  {kind} field={item.field}"]
        for value in values:
            lines += _wrap(value, "      ", "          ")
    return lines + _wrap(item.reason, "      原因：", " " * 12)


def _format_changes(changes, target, destination):
    """Render one document's change block."""
    label = _TARGET_LABEL[target]
    if not changes:
        return [_rule(f"{label} 改动：无")]

    lines = [
        _rule(f"{label} 改动（{len(changes)} 项）"),
        f"写入      ：{destination}",
    ]
    for item in changes:
        lines.append("")
        lines += _format_change(item)
    return lines


def format_report(info, changelog):
    """Render the full console report for a successful adaptation."""
    lines = ["=" * LINE_WIDTH, "config_adapter: 适配成功", "=" * LINE_WIDTH]
    for label, value in (
        ("输入      ", info["input"]),
        (
            "模式      ",
            f"{info['profile']}（{info['profile_flag']}），"
            f"batch 策略 {info['batch_strategy']}",
        ),
        (
            "机器规模  ",
            f"{info['orig_scale_label']} -> {info['target_nodes']} 节点 / "
            f"{info['target_cards']} 卡（每节点 {info['cards_per_node']} 卡）",
        ),
        ("并行度    ", info["dims_line"]),
        ("sharding  ", info["sharding_line"]),
        ("缩容方案  ", info["plan_note"]),
    ):
        lines += _wrap(str(value), f"{label}：")

    lines.append("")
    lines += _format_changes(
        changelog.by_target("yaml"), "yaml", info["output"]
    )
    if info.get("model_config_output"):
        lines.append("")
        lines += _format_changes(
            changelog.by_target("json"),
            "json",
            info["model_config_output"],
        )

    notes = [
        f"跳过的精度开关：{note}" for note in info.get("skipped_switches") or []
    ] + [f"WARNING：{warning}" for warning in info.get("warnings") or []]
    if notes:
        lines += ["", _rule("提醒")]
        for note in notes:
            lines += _wrap(note)

    lines += ["", _rule("机器可读摘要")]
    lines.append(f"ORIGINAL_CARDS={info['orig_cards_label']}")
    lines.append(f"ORIGINAL_NODES={info['orig_nodes_label']}")
    lines.append(f"TARGET_CARDS={info['target_cards']}")
    lines.append(f"OUTPUT={info['output']}")
    if info.get("model_config_output"):
        lines.append(f"MODEL_CONFIG_OUTPUT={info['model_config_output']}")
    return "\n".join(lines)
