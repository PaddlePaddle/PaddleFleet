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

"""Keep GLM52 input identities and native receipts visible after CI cleanup."""

import argparse
import hashlib
import json
from pathlib import Path

ENV_FIELDS = {
    "schema",
    "framework",
    "framework_version",
    "python_version",
    "device",
    "device_name",
    "dtype",
    "cuda",
    "cudnn",
    "nccl",
    "nccl_package",
    "topology",
    "world_size",
    "deterministic",
    "invocation_id",
    "model_id",
    "revision",
    "model_source",
    "model_config_sha256",
    "weights_loaded",
    "source_modules",
}


def fingerprint(path):
    """Return {path, bytes, sha256} for a single file."""
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def input_manifest(model, tokenizer, data):
    """Fingerprint the frozen model/tokenizer/data inputs into a manifest dict."""
    files = [model / "config.json", tokenizer / "tokenizer.json"]
    for name in ("tokenizer_config.json", "chat_template.jinja"):
        if (tokenizer / name).is_file():
            files.append(tokenizer / name)
    index = model / "model.safetensors.index.json"
    if index.is_file():
        files.append(index)
        index_data = json.loads(index.read_text())
        weight_map = index_data.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"safetensors index missing 'weight_map': {index}")
        names = sorted(set(weight_map.values()))
        for name in names:
            path = model / name
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError(
                    f"Checkpoint shard is outside model directory: {name}"
                )
            files.append(path)
    else:
        files.append(model / "model.safetensors")
    files.extend(
        data / f"alignment_{side}.jsonl" for side in ("paddle", "torch")
    )
    return {
        "schema": "glm52-ci-input-manifest/v1",
        "files": [fingerprint(p) for p in files],
    }


def report_receipts(run):
    """Echo each side's native env/input/loss receipts so CI cleanup cannot hide them."""
    for side in ("paddle", "torch"):
        for name in ("env.json", "input_receipt.json", "loss.json"):
            path = run / side / name
            if not path.is_file():
                continue
            try:
                value = json.loads(path.read_text())
                if name == "env.json":
                    value = {
                        key: item
                        for key, item in value.items()
                        if key in ENV_FIELDS
                    }
                print(
                    "GLM52_NATIVE_RECEIPT "
                    + json.dumps(
                        {"framework": side, "file": name, "receipt": value},
                        sort_keys=True,
                    ),
                    flush=True,
                )
            except (OSError, ValueError, TypeError, AttributeError) as error:
                print(
                    f"GLM52_NATIVE_RECEIPT_ERROR {side}/{name}: {error}",
                    flush=True,
                )


def main():
    """CLI entry: `inputs` writes the input manifest; `receipts` echoes native receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inputs", "receipts"))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    if args.mode == "receipts":
        report_receipts(args.run_dir)
        return
    if not all((args.model_dir, args.tokenizer_dir, args.data_dir)):
        parser.error("inputs requires model, tokenizer and data directories")
    result = input_manifest(args.model_dir, args.tokenizer_dir, args.data_dir)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / "input_manifest.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        "GLM52_INPUT_MANIFEST " + json.dumps(result, sort_keys=True), flush=True
    )


if __name__ == "__main__":
    main()
