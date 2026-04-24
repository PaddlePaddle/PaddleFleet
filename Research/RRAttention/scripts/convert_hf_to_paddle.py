#!/usr/bin/env python

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

import argparse
import json
import shutil
import sys
from pathlib import Path

import paddle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rrattn.checkpoint_utils import (
    flex_checkpoint_load_lock,
    is_hf_safetensors_checkpoint,
    load_config_for_model,
)

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "generation_config.json",
)


def infer_model_type(input_dir: Path) -> str:
    with (input_dir / "config.json").open("r", encoding="utf-8") as handle:
        model_type = str(json.load(handle).get("model_type", "")).lower()

    if "qwen3" in model_type:
        raise ValueError("Qwen3 models are not supported in this release")
    if "qwen" in model_type:
        return "qwen"
    if "llama" in model_type:
        return "llama"
    if "ernie" in model_type and "moe" in model_type:
        return "ernie_moe"
    if "ernie" in model_type:
        return "ernie"
    raise ValueError(
        f"Cannot infer model type from config.json model_type={model_type!r}; pass --model-type"
    )


def get_model_class(model_type: str):
    if model_type == "llama":
        from paddleformers.transformers import LlamaForCausalLM

        return LlamaForCausalLM
    if model_type == "qwen":
        from paddleformers.transformers.qwen2.modeling import (
            Qwen2ForCausalLMDeprecated,
        )

        return Qwen2ForCausalLMDeprecated
    if model_type == "ernie":
        from paddleformers.transformers import Ernie4_5ForCausalLM

        return Ernie4_5ForCausalLM
    if model_type == "ernie_moe":
        from paddleformers.transformers.ernie4_5_moe.modeling import (
            Ernie4_5_MoeForCausalLM,
        )

        return Ernie4_5_MoeForCausalLM
    raise ValueError(f"Unsupported model_type={model_type!r}")


def copy_runtime_files(src: Path, dst: Path) -> None:
    for name in TOKENIZER_FILES:
        src_file = src / name
        if src_file.is_file():
            shutil.copy2(src_file, dst / name)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} is not empty; pass --overwrite to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def default_device() -> str:
    return "gpu:0" if paddle.device.cuda.device_count() > 0 else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face safetensors checkpoint to Paddle weights."
    )
    parser.add_argument(
        "--input", required=True, help="HF safetensors checkpoint directory"
    )
    parser.add_argument(
        "--output", required=True, help="Output Paddle checkpoint directory"
    )
    parser.add_argument(
        "--model-type",
        default="auto",
        choices=["auto", "llama", "qwen", "ernie", "ernie_moe"],
    )
    parser.add_argument(
        "--dtype", default="bfloat16", help="Model dtype used during conversion"
    )
    parser.add_argument(
        "--device", default=None, help="Paddle device, e.g. gpu:0 or cpu"
    )
    parser.add_argument(
        "--max-shard-size",
        default="10GB",
        help="Max shard size for save_pretrained",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a non-empty output directory",
    )
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    if input_dir == output_dir:
        raise ValueError("--input and --output must be different directories")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"{input_dir} is not a directory")
    if not is_hf_safetensors_checkpoint(input_dir):
        raise ValueError(
            f"{input_dir} does not look like a Hugging Face safetensors checkpoint directory"
        )

    model_type = (
        infer_model_type(input_dir)
        if args.model_type == "auto"
        else args.model_type
    )
    if model_type == "qwen":
        with (input_dir / "config.json").open("r", encoding="utf-8") as handle:
            config_model_type = str(
                json.load(handle).get("model_type", "")
            ).lower()
        if "qwen3" in config_model_type:
            raise ValueError("Qwen3 models are not supported in this release")
    model_cls = get_model_class(model_type)

    prepare_output_dir(output_dir, args.overwrite)
    paddle.set_device(args.device or default_device())

    config = load_config_for_model(model_cls, input_dir)
    kwargs = {"dtype": args.dtype, "load_checkpoint_format": "flex_checkpoint"}
    if config is not None:
        kwargs["config"] = config

    with flex_checkpoint_load_lock(input_dir):
        model = model_cls.from_pretrained(str(input_dir), **kwargs)
    model.eval()
    model.save_pretrained(
        str(output_dir),
        save_checkpoint_format="sharding_io",
        save_to_hf=False,
        safe_serialization=False,
        max_shard_size=args.max_shard_size,
    )
    copy_runtime_files(input_dir, output_dir)
    print(f"Saved Paddle checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
