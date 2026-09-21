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

"""Locate the ``model_config.json`` that belongs to a training YAML.

The path is always inferred from the YAML's own ``model_name_or_path`` -- the
same value the training entry point consumes -- so there is no separate CLI
flag to keep in sync:

1. an absolute value is used verbatim;
2. a relative value is tried against the YAML's directory first (co-located
   layout), then against the current working directory (Fleet resolves
   ``model_name_or_path`` against the launch CWD);
3. the first existing directory must contain ``model_config.json``, otherwise
   we fail fast listing every path tried.

Registry lookups (HuggingFace hub, model zoo) are deliberately not attempted:
adaptation only supports local file-based configs.
"""

from __future__ import annotations

import os
from pathlib import Path


class ModelConfigResolveError(RuntimeError):
    """Raised when the source ``model_config.json`` cannot be located."""


def resolve_model_config(model_name_or_path, yaml_dir):
    """Return ``(model_dir, model_config_json_path)``.

    Raises :class:`ModelConfigResolveError` when the value is empty or nothing
    on disk matches it.
    """
    if not model_name_or_path:
        raise ModelConfigResolveError(
            "YAML 里没有 model_name_or_path，无法定位 model_config.json；"
            "缩减模型结构（专家数 / 层数）需要它"
        )

    yaml_dir = Path(yaml_dir)
    raw = Path(str(model_name_or_path)).expanduser()

    if raw.is_absolute():
        candidates = [raw]
    else:
        candidates = list(dict.fromkeys([yaml_dir / raw, Path.cwd() / raw]))

    tried = []
    model_dir = None
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        tried.append(resolved)
        if resolved.is_dir():
            model_dir = resolved
            break

    if model_dir is None:
        tried_str = ", ".join(str(p) for p in tried)
        raise ModelConfigResolveError(
            f"模型目录不存在：已尝试 [{tried_str}]"
            f"（来自 model_name_or_path={model_name_or_path!r}，"
            f"yaml_dir={yaml_dir}，cwd={Path.cwd()}）"
        )

    json_path = model_dir / "model_config.json"
    if not json_path.is_file():
        raise ModelConfigResolveError(f"{model_dir} 下缺少 model_config.json")

    return model_dir, json_path


def build_adapted_dir(output_dir, source_stem, scale_tag):
    """Directory that receives the shrunk ``model_config.json``.

    Layout: ``<output_dir>/model_config_separated/<stem>_adapted_<tag>/``.
    The directory is not created here; the caller decides that.
    """
    return (
        Path(output_dir)
        / "model_config_separated"
        / f"{source_stem}_adapted_{scale_tag}"
    )


def rewrite_model_name_or_path(new_dir, yaml_dir, source_was_absolute):
    """Value to write back into ``model_name_or_path``.

    A relative source value stays relative only when the generated directory
    actually sits under the YAML's directory.  Otherwise an absolute path is
    written: a relative value would be resolved against the *launch CWD* by
    the training entry point, which need not be the YAML's directory, and a
    ``../../..`` chain out of the repo is both fragile and unreadable.
    """
    new_dir = Path(new_dir).resolve()
    if source_was_absolute:
        return str(new_dir)
    relative = os.path.relpath(str(new_dir), str(Path(yaml_dir).resolve()))
    if relative.startswith(".."):
        return str(new_dir)
    return relative
