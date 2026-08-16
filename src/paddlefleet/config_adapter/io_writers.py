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

"""YAML / JSON read-modify-write layer.

All document mutation and serialization lives here so the rest of the package
only decides *what* to set, never *how* it is written.  ``apply_config_map``
semantics for both writers, given a flat ``{key: value}`` map:

* key already present -> overwritten in place (ruamel keeps the original
  position, comments and quote style);
* key missing -> appended as a new field;
* key listed in ``protected`` -> left untouched.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

from ruamel.yaml import YAML


def make_yaml():
    """Return a ruamel YAML that preserves quotes and never wraps lines."""
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096
    return yaml


def _apply(document, config_map, protected=None):
    """Merge ``config_map`` into ``document`` in place, returning the diff."""
    protected = set(protected or ())
    changes = []
    for key, value in config_map.items():
        if key in protected:
            continue
        if key in document:
            old = document[key]
            if old != value:
                changes.append(("modified", key, old, value))
        else:
            changes.append(("added", key, None, value))
        document[key] = value
    return changes


def _write_atomically(output_path, text):
    """Write ``text`` through a temp file + rename.

    Never truncates the destination in place, so an interrupted run cannot
    leave a half-written config behind.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".config_adapter.tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


class YamlWriter:
    """Loads a YAML document, applies a config map, serializes it back."""

    def __init__(self, yaml=None):
        self.yaml = yaml or make_yaml()

    def load(self, path):
        """Load the YAML document (``None`` for an empty file)."""
        with open(path, encoding="utf-8") as f:
            return self.yaml.load(f)

    def apply_config_map(self, config, config_map, protected=None):
        """Merge a flat map into ``config``; see the module docstring."""
        return _apply(config, config_map, protected)

    def write(self, config, output_path, header=""):
        """Write ``config``, prepending an optional comment ``header``."""
        buffer = io.StringIO()
        if header:
            buffer.write(header)
        self.yaml.dump(config, buffer)
        _write_atomically(output_path, buffer.getvalue())


class JsonWriter:
    """Loads a JSON document, applies a config map, serializes it back."""

    def __init__(self, indent=2, ensure_ascii=False):
        self.indent = indent
        self.ensure_ascii = ensure_ascii

    def load(self, path):
        """Load the JSON document.

        Raises ``ValueError`` when the file is empty or malformed.
        """
        text = Path(path).read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"JSON 文件为空：{path}")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON 解析失败 {path}: {exc}") from exc

    def apply_config_map(self, config, config_map, protected=None):
        """Merge a flat map into ``config``; see the module docstring."""
        return _apply(config, config_map, protected)

    def write(self, config, output_path):
        """Write ``config`` with a trailing newline (POSIX friendly)."""
        text = json.dumps(
            config, indent=self.indent, ensure_ascii=self.ensure_ascii
        )
        _write_atomically(output_path, text + "\n")
