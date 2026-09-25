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

"""Normalizer (pipeline step 1): registry-driven lossless normalization
(legacy ``index_*`` aliases, numeric-string type coercion).

Extension point: new aliases come from the registry (``aliases`` extras);
new coercions come from ``_coerce_value``'s per-spec-type branches.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from paddlefleet.transformer.attention_plan.registry import (
    _ALIAS_TO_CANONICAL,
    ATTENTION_FIELD_REGISTRY,
)

# ---------------------------------------------------------------------------
# Normalizer (1) -- lossless conversion
# ---------------------------------------------------------------------------


@dataclass
class NormalizationReport:
    """Normalization result: records lossless conversions only, changing no
    semantics."""

    changes: list[dict] = field(
        default_factory=list
    )  # {field, old, new, reason}
    warnings: list[dict] = field(default_factory=list)  # {field, message}

    def to_json(self) -> dict:
        return {"changes": self.changes, "warnings": self.warnings}


_SPEC_PY_TYPES: dict[str, type | tuple[type, ...]] = {
    "int": int,
    "float": (int, float),  # int is an acceptable float source, coerced below
    "bool": bool,
    "str": str,
    "list": (list, tuple),
}

# Exact-type skip table for the coercion loop: values already of the
# canonical type are left untouched (int-for-float, tuple-for-list, and
# bool-for-int are still coerced; ``type`` keeps bool from matching int).
_SPEC_EXACT_TYPES: dict[str, type] = {
    "int": int,
    "float": float,
    "bool": bool,
    "str": str,
    "list": list,
}

# Sentinel: coercion attempted but failed (warning already recorded).
_COERCE_FAILED = object()


def _coerce_value(
    name: str, spec_type: str, value: Any, report: NormalizationReport
) -> Any:
    """Coerce ``value`` to the declared ``spec_type``; warn-only on failure.

    Covers every mismatch between declared and actual type: numeric strings
    ("160000.0", "3", arbitrary-precision integers), string bools
    ("true"/"false"), float->int (integral only, lossy warned), int->str,
    and JSON-parseable list strings.

    Accepted input shapes per spec type:
    - ``int``: ``bool``/``int`` (bool counts as 0/1), integral ``float``,
      or ``str`` holding a decimal integer (``"3"``) or integral float
      (``"16.0"``). Parsed with ``int()`` first so integers beyond 2^53
      keep exact precision; the float fallback only handles integral
      values.
    - ``bool``: ``bool``, ``int`` 0/1, or ``str`` in
      "true"/"1"/"yes"/"false"/"0"/"no" (case-insensitive). Anything else
      warns instead of being truthiness-coerced.
    - ``str``: any non-container scalar (``int``/``float``/``bool``).
      Containers (list/dict/...) warn rather than silently stringify.
    - ``list``: ``str`` holding a JSON array, ``tuple``/other non-dict
      iterables (element-wise ``list()``).
    """
    try:
        if spec_type == "int":
            if isinstance(value, str):
                text = value.strip()
                try:
                    coerced: Any = int(text, 10)
                except ValueError:
                    # integral numeric strings like "16.0"
                    coerced = float(text)
                    if not coerced.is_integer():
                        raise ValueError(f"non-integral value {value!r}")
                    coerced = int(coerced)
            elif isinstance(value, bool):
                coerced = int(value)
            elif isinstance(value, int):
                coerced = value
            elif isinstance(value, float):
                if not value.is_integer():
                    raise ValueError(f"non-integral value {value!r}")
                coerced = int(value)
            else:
                raise ValueError(f"cannot coerce {value!r} to int")
        elif spec_type == "float":
            coerced = float(value)
        elif spec_type == "bool":
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in ("true", "1", "yes"):
                    return True
                if lowered in ("false", "0", "no"):
                    return False
                raise ValueError(f"non-boolean string {value!r}")
            if isinstance(value, bool):
                return value
            if isinstance(value, int) and value in (0, 1):
                return bool(value)
            raise ValueError(f"cannot coerce {value!r} to bool")
        elif spec_type == "str":
            if isinstance(value, (bool, int, float)):
                return str(value)
            raise ValueError(f"cannot coerce {value!r} to str")
        elif spec_type == "list":
            if isinstance(value, str):
                parsed = json.loads(value)
                if not isinstance(parsed, list):
                    raise ValueError(f"non-list JSON {value!r}")
                return parsed
            if isinstance(value, (dict, bytes)) or not hasattr(
                value, "__iter__"
            ):
                raise ValueError(f"cannot coerce {value!r} to list")
            return list(value)
        else:  # "unknown" and anything undeclared: leave as-is
            return _COERCE_FAILED
    except (ValueError, TypeError):
        report.warnings.append(
            {
                "field": name,
                "message": (
                    f"'{name}' declared {spec_type} but got "
                    f"{type(value).__name__} {value!r} that cannot be coerced"
                ),
            }
        )
        return _COERCE_FAILED
    return coerced


def normalize_attention_config(
    config: Any, *, apply: bool = True
) -> NormalizationReport:
    """Lossless normalization driven by the registry declarations.

    Only two kinds of conversion are performed:
    - Alias normalization: when a legacy name (``index_*``) appears in the
      instance dict, map it to ``dsa_index_*`` (TransformerConfig has already
      been renamed via transform_rules; this covers direct-constructed /
      external configs);
    - Type coercion: fields declared float/int that receive numeric strings
      (e.g. ``"160000.0"``) are coerced; failure only warns (no raise in the
      warn-only phase).
    """
    report = NormalizationReport()

    # Alias normalization
    for old, canonical in _ALIAS_TO_CANONICAL.items():
        if old in getattr(config, "__dict__", {}):
            old_value = config.__dict__[old]
            if old_value is None:
                continue
            cur = getattr(config, canonical, None)
            if cur is None or cur == old_value:
                if apply:
                    setattr(config, canonical, old_value)
                report.changes.append(
                    {
                        "field": old,
                        "old": old_value,
                        "new": canonical,
                        "reason": f"alias -> {canonical} (registry)",
                    }
                )
            else:
                report.warnings.append(
                    {
                        "field": old,
                        "message": (
                            f"legacy alias '{old}'={old_value!r} conflicts "
                            f"with '{canonical}'={cur!r}; alias ignored"
                        ),
                    }
                )

    # Type coercion (declared type vs actual type)
    for name, spec in ATTENTION_FIELD_REGISTRY.items():
        if not hasattr(config, name):
            continue
        value = getattr(config, name)
        # Exact-type skip: only the canonical type of each spec matches;
        # int-for-float / tuple-for-list still coerce (declared acceptable
        # sources). ``type`` (not isinstance) so bool never masquerades
        # as int.
        if value is None or type(value) is _SPEC_EXACT_TYPES.get(
            spec.type, type(value)
        ):
            continue
        coerced = _coerce_value(name, spec.type, value, report)
        if coerced is _COERCE_FAILED:
            continue
        if apply:
            setattr(config, name, coerced)
        report.changes.append(
            {
                "field": name,
                "old": value,
                "new": coerced,
                "reason": f"{type(value).__name__} coerced to {spec.type} "
                "(registry type contract)",
            }
        )

    return report
