#!/usr/bin/env python3

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

import argparse
import json
import math
import os
import struct
from typing import Any

_SCHEMA = "glm52-machine-loss/v1"
_FRAMEWORKS = {"paddle", "torch"}


def _ieee64_bits(x: float) -> int:
    return struct.unpack(">Q", struct.pack(">d", x))[0]


def _fail(msg: str) -> None:
    raise ValueError(msg)


def _load_loss_json(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        _fail(f"loss json missing: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        try:
            obj = json.load(fh)
        except json.JSONDecodeError as e:
            _fail(f"malformed json {path}: {e}")
    if not isinstance(obj, dict):
        _fail(f"root must be object: {path}")
    return obj


def _as_int_step(v: Any) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        _fail(f"step is not a JSON integer: {v!r}")
    return v


def _as_finite_float(v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        _fail(f"loss is not a JSON number: {v!r}")
    x = float(v)
    if not math.isfinite(x):
        _fail(f"non-finite loss: {v!r}")
    return x


def validate_loss_artifact(
    obj: dict[str, Any], *, expected_framework: str, required_steps: int
) -> list[float]:
    if required_steps < 1:
        _fail(f"required_steps must be >= 1, got {required_steps}")
    if obj.get("schema") != _SCHEMA:
        _fail(f"schema must be {_SCHEMA!r}, got {obj.get('schema')!r}")
    if (
        obj.get("raw") is not True
        or obj.get("stage") != "training_callback_complete"
    ):
        _fail("expected complete native raw-loss artifact")
    fw = obj.get("framework")
    if fw not in _FRAMEWORKS:
        _fail(f"framework must be paddle|torch, got {fw!r}")
    if fw != expected_framework:
        _fail(
            f"framework mismatch: expected {expected_framework!r}, got {fw!r}"
        )
    steps = obj.get("steps")
    losses = obj.get("losses")
    if not isinstance(steps, list) or not isinstance(losses, list):
        _fail("steps and losses must be arrays")
    if len(steps) == 0 or len(losses) == 0:
        _fail("steps and losses must be nonempty")
    if len(steps) != len(losses):
        _fail(f"steps/losses length mismatch: {len(steps)} vs {len(losses)}")
    if len(losses) != required_steps:
        _fail(f"required {required_steps} losses, got {len(losses)}")
    parsed_steps = [_as_int_step(s) for s in steps]
    parsed_losses = [_as_finite_float(x) for x in losses]
    expected = list(range(1, required_steps + 1))
    if parsed_steps != expected:
        _fail(
            f"steps must be contiguous 1..{required_steps}, got {parsed_steps!r}"
        )
    return parsed_losses


def resolve_loss_json(path: str) -> str:
    """Read an explicit file or run directory; never choose an older successful run."""
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        _fail(f"path does not exist: {path}")
    direct = os.path.join(path, "loss.json")
    if os.path.isfile(direct):
        return direct
    _fail(f"loss.json missing from run directory: {path}")


def compare_loss_json(pf_path: str, mg_path: str, required_steps: int) -> int:
    """Return 0 iff both artifacts are valid and IEEE64 bit-identical. Fail closed."""
    try:
        pf_file = resolve_loss_json(pf_path)
        mg_file = resolve_loss_json(mg_path)
        pf = validate_loss_artifact(
            _load_loss_json(pf_file),
            expected_framework="paddle",
            required_steps=required_steps,
        )
        mg = validate_loss_artifact(
            _load_loss_json(mg_file),
            expected_framework="torch",
            required_steps=required_steps,
        )
    except (ValueError, OSError, OverflowError, TypeError) as e:
        print(f"FAIL closed: {e}")
        return 1
    bad = []
    for i, (a, b) in enumerate(zip(pf, mg), start=1):
        if _ieee64_bits(a) != _ieee64_bits(b):
            bad.append(i)
    if bad:
        first = min(bad)
        print(
            f"  结论: IEEE64 不一致，首个分叉 step = {first}；"
            f"不一致 step 列表 = {bad}"
        )
        print(
            f"      PF step {first}: {pf[first - 1]!r} bits=0x{_ieee64_bits(pf[first - 1]):016x}"
        )
        print(
            f"      MG step {first}: {mg[first - 1]!r} bits=0x{_ieee64_bits(mg[first - 1]):016x}"
        )
        return 1
    print(f"  结论: {required_steps} steps IEEE64 完全一致 ✅")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Compare GLM52 native 100-step loss artifacts"
    )
    parser.add_argument("paddle")
    parser.add_argument("torch")
    parser.add_argument("--required-steps", type=int, default=100)
    args = parser.parse_args()
    return compare_loss_json(args.paddle, args.torch, args.required_steps)


if __name__ == "__main__":
    raise SystemExit(main())
