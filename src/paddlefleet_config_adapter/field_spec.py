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

"""Alias registry for the ``model_config.json`` fields shrinking depends on.

The same structural quantity is spelled differently per model family::

    routed experts : n_routed_experts (ERNIE5 / DeepSeek / GLM / Kimi),
                     num_local_experts (MiniMax / GPT-OSS),
                     num_experts (Qwen), moe_num_experts (ERNIE4.5)
    top-k          : num_experts_per_tok (most), moe_k (ERNIE4.5)
    dense prefix   : first_k_dense_replace, moe_layer_start_index
    MTP layers     : num_nextn_predict_layers

Centralising the aliases means supporting a new family is a one-line edit to
:data:`FIELD_SPECS`.  The resolver also remembers which physical key matched,
so a write-back never invents a second spelling of the same field.

Pure module: no I/O, no mutation of inputs.
"""

from __future__ import annotations

from collections import namedtuple

# needed_by only phrases diagnostics; ``default is None`` marks the field as
# required for that shrink axis.
FieldSpec = namedtuple("FieldSpec", ["aliases", "default", "needed_by"])

FIELD_SPECS = {
    "num_hidden_layers": FieldSpec(
        aliases=("num_hidden_layers",),
        default=None,
        needed_by="PP",
    ),
    "num_experts": FieldSpec(
        aliases=(
            "n_routed_experts",
            "num_local_experts",
            "num_experts",
            "moe_num_experts",
        ),
        default=None,
        needed_by="EP",
    ),
    "num_experts_per_tok": FieldSpec(
        aliases=("num_experts_per_tok", "moe_k"),
        default=None,
        needed_by="EP",
    ),
    "first_k_dense_replace": FieldSpec(
        aliases=("first_k_dense_replace", "moe_layer_start_index"),
        default=0,
        needed_by="PP",
    ),
    "num_nextn_predict_layers": FieldSpec(
        aliases=("num_nextn_predict_layers",),
        default=0,
        needed_by="PP",
    ),
}

ResolvedField = namedtuple(
    "ResolvedField", ["logical", "value", "origin", "writeback_key"]
)


def _writeback_key(spec, model_config):
    """First alias present in ``model_config``, else the canonical alias."""
    for alias in spec.aliases:
        if alias in model_config:
            return alias
    return spec.aliases[0]


def resolve_fields(model_config):
    """Resolve every registered field against ``model_config``.

    Returns ``(resolved, missing)``: ``resolved`` maps a logical name to a
    :class:`ResolvedField`, ``missing`` maps a logical name to its
    :class:`FieldSpec` for required fields that could not be found.
    """
    resolved = {}
    missing = {}

    for logical, spec in FIELD_SPECS.items():
        writeback_key = _writeback_key(spec, model_config)
        hit = next(
            (a for a in spec.aliases if model_config.get(a) is not None),
            None,
        )
        if hit is not None:
            resolved[logical] = ResolvedField(
                logical, model_config[hit], hit, writeback_key
            )
        elif spec.default is not None:
            resolved[logical] = ResolvedField(
                logical, spec.default, "<default>", writeback_key
            )
        else:
            missing[logical] = spec

    return resolved, missing


def describe_missing(logical, spec):
    """One-line actionable explanation for an unresolved required field."""
    tried = " / ".join(spec.aliases)
    return (
        f"{logical}（{spec.needed_by} 缩容所需）在 model_config.json 中读不到"
        f"（已尝试字段名 {tried}）；"
        f"请用 --set json:{spec.aliases[0]}=<值> 显式补上，"
        f"或在 field_spec.py 的 FIELD_SPECS[{logical!r}] 里登记新别名"
    )
