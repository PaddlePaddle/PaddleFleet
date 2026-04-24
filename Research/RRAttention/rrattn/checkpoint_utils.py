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

from __future__ import annotations

import contextlib
import hashlib
import os
from pathlib import Path

PADDLE_CHECKPOINT_SENTINELS = (
    "model_state.pdparams",
    "model_state.pdparams.index.json",
)

HF_SAFETENSORS_SENTINELS = (
    "model.safetensors",
    "model.safetensors.index.json",
)


def is_paddle_checkpoint(model_name_or_path: str | os.PathLike) -> bool:
    path = Path(model_name_or_path)
    if not path.is_dir():
        return False
    return any((path / name).is_file() for name in PADDLE_CHECKPOINT_SENTINELS)


def is_hf_safetensors_checkpoint(model_name_or_path: str | os.PathLike) -> bool:
    path = Path(model_name_or_path)
    if not path.is_dir():
        return False
    if any((path / name).is_file() for name in HF_SAFETENSORS_SENTINELS):
        return True
    return any(path.glob("model-*.safetensors"))


def normalize_rope_scaling_dict(config_dict):
    rope_scaling = config_dict.get("rope_scaling")
    if not isinstance(rope_scaling, dict):
        return config_dict

    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
    if rope_type != "yarn":
        return config_dict

    rope_scaling["rope_type"] = rope_type
    original_max_position_embeddings = rope_scaling.get(
        "original_max_position_embeddings"
    )
    factor = rope_scaling.get("factor")
    if original_max_position_embeddings is not None and factor is not None:
        config_dict["max_position_embeddings"] = int(
            original_max_position_embeddings * factor
        )

    return config_dict


def load_config_for_model(model_cls, model_name_or_path: str | os.PathLike):
    config_cls = getattr(model_cls, "config_class", None)
    if config_cls is None:
        return None

    config_dict, _ = config_cls.get_config_dict(str(model_name_or_path))
    if config_dict is None:
        return None
    if config_cls.base_config_key and config_cls.base_config_key in config_dict:
        config_dict = config_dict[config_cls.base_config_key]
    normalize_rope_scaling_dict(config_dict)
    config = config_cls.from_dict(config_dict)
    config.name_or_path = str(model_name_or_path)
    return config


def _lock_path(model_name_or_path: str | os.PathLike) -> Path:
    lock_dir = Path(os.environ.get("RRATTN_CHECKPOINT_LOCK_DIR", "/tmp"))
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = Path(model_name_or_path)
    lock_key = str(path.resolve()) if path.exists() else str(model_name_or_path)
    key = hashlib.sha256(lock_key.encode("utf-8")).hexdigest()[:16]
    return lock_dir / f"rrattn-flex-checkpoint-{key}.lock"


@contextlib.contextmanager
def flex_checkpoint_load_lock(model_name_or_path: str | os.PathLike):
    lock_path = _lock_path(model_name_or_path)
    with lock_path.open("w") as handle:
        if os.name == "posix":
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "posix":
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_UN)


def load_pretrained_checkpoint(
    model_cls, model_name_or_path: str | os.PathLike, *, dtype
):
    model_name_or_path = str(model_name_or_path)
    if is_paddle_checkpoint(model_name_or_path):
        return model_cls.from_pretrained(
            model_name_or_path,
            dtype=dtype,
            load_checkpoint_format="sharding_io",
            convert_from_hf=False,
            use_safetensors=False,
        )

    if is_hf_safetensors_checkpoint(model_name_or_path):
        config = load_config_for_model(model_cls, model_name_or_path)
        kwargs = {"dtype": dtype, "load_checkpoint_format": "flex_checkpoint"}
        if config is not None:
            kwargs["config"] = config
        with flex_checkpoint_load_lock(model_name_or_path):
            return model_cls.from_pretrained(model_name_or_path, **kwargs)

    return model_cls.from_pretrained(
        model_name_or_path,
        dtype=dtype,
        load_checkpoint_format="sharding_io",
    )
