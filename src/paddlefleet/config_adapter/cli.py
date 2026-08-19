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

"""Command line entry point for the config adapter."""

from __future__ import annotations

import argparse
import difflib
import os
import sys
from pathlib import Path

from .core import ConfigAdapter, inspect_config
from .io_writers import YamlWriter
from .model_config_resolver import (
    ModelConfigResolveError,
    resolve_model_config,
)
from .options import AdaptOptions
from .utils import parse_value

EPILOG = """\
示例：
  # 1) 只看这份配置能跑在哪些机器规模上（不生成文件）
  python -m paddlefleet.config_adapter --input config.yaml

  # 2) 适配到 2 台机器（默认每台 8 卡 = 16 卡），必要时自动缩小 EP/PP
  python -m paddlefleet.config_adapter --input config.yaml --target-nodes 2

  # 3) 测速：冻结 TP/PP/EP/CP/SEP 与 acc，只改 sharding 和 GBS
  python -m paddlefleet.config_adapter --input config.yaml \\
      --target-nodes 2 --test-performance

  # 4) 精度测试：注入避免 aadiff 的开关，并保持等效 batch
  python -m paddlefleet.config_adapter --input config.yaml \\
      --target-nodes 1 --test-accuracy

  # 5) 两个维度可以同时给：既冻结并行策略，又注入精度开关
  python -m paddlefleet.config_adapter --input config.yaml \\
      --target-nodes 8 --test-performance --test-accuracy

  # 6) 单机 2 卡（用 --cards-per-node 表达非 8 卡机型）
  python -m paddlefleet.config_adapter --input config.yaml \\
      --target-nodes 1 --cards-per-node 2 --test-accuracy

  # 7) 就地改写源文件，并额外生成 <input>.patch
  python -m paddlefleet.config_adapter --input config.yaml \\
      --target-nodes 1 --test-accuracy --in-place

  # 8) 自定义字段：不带前缀时自动判断改 yaml 还是 model_config.json
  python -m paddlefleet.config_adapter --input config.yaml \\
      --target-nodes 1 --test-accuracy \\
      --set max_steps=10 --set n_routed_experts=32 \\
      --set json:some_new_field=1
"""


def build_parser():
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m paddlefleet.config_adapter",
        description=(
            "把面向大集群的训练 YAML 适配到更小的机器规模："
            "重算 sharding 与 batch，必要时缩小 EP/PP 并联动改写 "
            "model_config.json。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    parser.add_argument("--input", required=True, help="源 YAML 路径")
    parser.add_argument(
        "--target-nodes",
        type=int,
        default=None,
        help="目标机器台数；总卡数 = target-nodes × cards-per-node。"
        "不传则只列出合法的机器规模，不生成任何文件",
    )
    parser.add_argument(
        "--cards-per-node",
        type=int,
        default=8,
        help="每台机器的卡数（默认 8）",
    )
    parser.add_argument(
        "--test-performance",
        action="store_true",
        help="测速维度：冻结 TP/PP/EP/CP/SEP 与 acc，只改 sharding 和 GBS。"
        "与 --test-accuracy 正交，可单独给、可同时给、也可都不给"
        "（都不给时默认允许缩小 EP/PP）",
    )
    parser.add_argument(
        "--test-accuracy",
        action="store_true",
        help="精度维度：注入避免 aadiff 的开关；未同时指定 "
        "--test-performance 时还会保持等效 batch（GBS 不变、acc 放大）",
    )
    parser.add_argument(
        "--output-dir",
        default="./adapted_configs",
        help="输出目录（默认 ./adapted_configs）；--in-place 时忽略",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="[yaml:|json:]KEY=VALUE",
        help="自定义覆盖，可重复。不带前缀时自动判断改哪个文件"
        "（谁声明了这个 key 就改谁，两边都有就都改，两边都没有则新增到 "
        "yaml）；要把新字段加到 model_config.json 请显式写 json:KEY=VALUE。"
        "这些字段会被保护，不再参与自动适配",
    )
    parser.add_argument(
        "-i",
        "--in-place",
        action="store_true",
        help="就地改写源 YAML（同时就地改写 model_config.json），"
        "并生成 <input>.patch；请确保源文件已纳入版本管理",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="允许覆盖已存在的 model_config 生成目录；"
        "默认遇到同名目录会中止，避免覆盖上一次的产物",
    )
    return parser


def parse_overrides(items):
    """Split ``--set`` items into ``(yaml_map, json_map, auto_map)``.

    ``yaml:KEY=VALUE`` / ``json:KEY=VALUE`` state the target document
    explicitly; a prefix-less ``KEY=VALUE`` goes into ``auto_map`` and the
    adapter decides later by looking at which document declares the key.
    """
    yaml_map, json_map, auto_map = {}, {}, {}
    for item in items:
        target, _, rest = item.partition(":")
        if target in ("yaml", "json") and rest:
            bucket = yaml_map if target == "yaml" else json_map
            payload = rest
        else:
            bucket = auto_map
            payload = item

        key, sep, raw_value = payload.partition("=")
        if not sep or not key.strip():
            raise ValueError(
                f"--set 必须写成 KEY=VALUE（可带 yaml:/json: 前缀），"
                f"收到：{item!r}"
            )
        bucket[key.strip()] = parse_value(raw_value.strip())
    return yaml_map, json_map, auto_map


def _companion_json_path(yaml_path):
    """Best-effort ``model_config.json`` path, for in-place patch snapshots."""
    try:
        config = YamlWriter().load(yaml_path)
        if not config:
            return None
        _model_dir, json_path = resolve_model_config(
            config.get("model_name_or_path"), Path(yaml_path).parent
        )
        return json_path
    except (ModelConfigResolveError, ValueError, OSError):
        return None


def _read_lines(path):
    """Read a file as a list of lines, or ``None`` when it does not exist."""
    path = Path(path) if path else None
    if not path or not path.is_file():
        return None
    return path.read_text(encoding="utf-8").splitlines(keepends=True)


def _unified_diff(path, before):
    """Unified diff of ``path`` against its ``before`` snapshot."""
    if before is None:
        return []
    after = _read_lines(path)
    if after is None:
        return []
    rel = os.path.relpath(str(path), str(Path.cwd()))
    return list(
        difflib.unified_diff(
            before, after, fromfile=f"a/{rel}", tofile=f"b/{rel}"
        )
    )


def main(argv=None):
    """Parse arguments and run the adaptation."""
    args = build_parser().parse_args(argv)

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"错误：输入文件不存在：{input_path}", file=sys.stderr)
        return 1

    if args.cards_per_node < 1:
        print("错误：--cards-per-node 必须 >= 1", file=sys.stderr)
        return 1

    try:
        yaml_overrides, json_overrides, auto_overrides = parse_overrides(
            args.overrides
        )
    except ValueError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    # ---- inspection mode: no target scale, nothing is written ------------
    if args.target_nodes is None:
        orig_cards, orig_nodes, valid_nodes = inspect_config(
            input_path, cards_per_node=args.cards_per_node
        )
        print(f"ORIGINAL_CARDS={orig_cards if orig_cards else 'UNKNOWN'}")
        print(f"ORIGINAL_NODES={orig_nodes if orig_nodes else 'UNKNOWN'}")
        print("VALID_NODES=" + ",".join(str(n) for n in valid_nodes))
        print(
            "提示：加 --target-nodes <目标机器台数> 才会生成配置；"
            "按需叠加 --test-performance / --test-accuracy"
        )
        return 0

    if args.target_nodes < 1:
        print("错误：--target-nodes 必须 >= 1", file=sys.stderr)
        return 1

    options = AdaptOptions(
        test_performance=args.test_performance,
        test_accuracy=args.test_accuracy,
    )

    before_yaml = before_json = None
    json_path = None
    if args.in_place:
        before_yaml = _read_lines(input_path)
        json_path = _companion_json_path(input_path)
        before_json = _read_lines(json_path)

    adapter = ConfigAdapter(
        options=options,
        target_nodes=args.target_nodes,
        cards_per_node=args.cards_per_node,
        yaml_overrides=yaml_overrides,
        json_overrides=json_overrides,
        auto_overrides=auto_overrides,
        output_dir=args.output_dir,
        in_place=args.in_place,
        force=args.force,
    )
    ok, message = adapter.adapt(input_path)
    if not ok:
        print(f"适配失败：{message}", file=sys.stderr)
        return 1

    print(message)

    if args.in_place:
        diff = _unified_diff(input_path, before_yaml) + _unified_diff(
            json_path, before_json
        )
        patch_path = input_path.parent / (input_path.name + ".patch")
        patch_path.write_text("".join(diff), encoding="utf-8")
        print(f"PATCH={patch_path}")
    return 0
