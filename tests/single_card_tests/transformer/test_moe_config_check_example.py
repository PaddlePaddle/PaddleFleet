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

"""用一份训练 YAML 检查 MoE 配置的示例。

复制本文件, 只改下面 CONFIG 区的三个常量, 就能检查任意一份配置。

它读取 YAML, 从 YAML 的 ``model_name_or_path`` 找到同目录下的 ``model_config.json``,
把两者合并成 ERNIEBot 实际使用的扁平配置, 然后报告:

* 当前配置会命中哪条 dispatcher / router / expert / precision 路径;
* 哪些字段被路由过程改写了;
* 哪些字段配了但当前路径不会读取(作用域错误 / 死配置 / 派生值 / 属其他子系统)。

**YAML 键与 JSON 键的并集就是"用户显式配过的字段"**, 这正是诊断需要的输入。直接
去问 config 对象是不行的: 上游 PretrainedConfig 会把自己的默认值一并写进去, 实测
会让近半数 MoE 字段被误判成用户显式设置。

运行方式:

    cd third_party/PaddleFleet
    PYTHONPATH=src python -m unittest \\
        tests.single_card_tests.transformer.test_moe_config_check_example

因为 YAML 在 ERNIEBot 仓库里而不在 PaddleFleet 里, 找不到文件时本测试自动跳过,
不会影响 PaddleFleet 自身的 CI。
"""

import json
import os
import sys
import unittest
from types import SimpleNamespace

# Prefer the in-tree PaddleFleet over a site-packages install. Relative
# PYTHONPATH entries (e.g. ./third_party/PaddleFleet/src/) break after cd,
# and the installed package does not ship moe_config_check.
_PADDLEFLEET_SRC = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"
    )
)
if _PADDLEFLEET_SRC not in sys.path:
    sys.path.insert(0, _PADDLEFLEET_SRC)

import paddle
import yaml

from paddlefleet.transformer.moe.moe_config_check import (
    collect_findings,
    format_report,
    resolve_moe_plan,
)
from paddlefleet.transformer.transformer_config import TransformerConfig

# ==================== 改这里 ====================

# 要检查的 YAML。可以写绝对路径, 也可以写相对 ERNIEBot 仓库根目录的路径。
# YAML_PATH = "conf/online/ernielite_layer43_pretrain_mla_hca.yaml"
YAML_PATH = "conf/online/mimo_flash_pretrain32k_fleet_0702.yaml"
# YAML_PATH = "conf/experiment/eb5_v2/eb5_v2_A10B_230B_pretrain_0419_8k_bzz3_fleet_backend.yaml"

# EP 组大小。None 表示读 YAML 里的 expert_model_parallel_size。
# 这个值必须对: EP=1 时根本不走 dispatcher, 所有 dispatcher 专属字段都是无效的。
EXPERT_PARALLEL_SIZE = None

# 设备算力。None 表示用当前机器的算力; 想检查换到别的机型上会怎样, 就直接写死,
# 例如 (9, 0) 表示 Hopper, (8, 0) 表示 Ampere(会触发 deepep -> alltoall 回退)。
DEVICE_CAPABILITY = None

# ================================================


def _erniebot_root():
    """本文件位于 <erniebot>/third_party/PaddleFleet/tests/single_card_tests/
    transformer/, 上溯 6 层即 ERNIEBot 仓库根目录。"""
    path = os.path.abspath(__file__)
    for _ in range(6):
        path = os.path.dirname(path)
    return path


def _resolve(path):
    if os.path.isabs(path):
        return path
    return os.path.join(_erniebot_root(), path)


def _load_flat_config(yaml_path):
    """把 YAML 与 model_config.json 合并成 ERNIEBot 实际使用的扁平配置。

    Returns:
        (merged, yaml_keys, json_keys): merged 是合并后的 dict, 后两者用于区分字段
        来源——报告只诊断这两个集合里的字段。

    注: 用 yaml.safe_load 而不是 OmegaConf, 所以 YAML 里的 ``${...}`` 插值不会被
    求值。现有 conf/ 下的配置都没有用到插值。
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        flat = yaml.safe_load(f) or {}

    model_name_or_path = flat.get("model_name_or_path")
    if not model_name_or_path:
        raise ValueError(
            f"{yaml_path} 里没有 model_name_or_path, 无法定位 JSON"
        )

    json_path = os.path.join(_resolve(model_name_or_path), "model_config.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"model_config.json 不存在: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        structure = json.load(f)

    return {**structure, **flat}, set(flat), set(structure)


def _device_capability():
    if DEVICE_CAPABILITY is not None:
        return DEVICE_CAPABILITY
    if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count():
        return paddle.device.cuda.get_device_capability()
    return None


@unittest.skipUnless(
    os.path.exists(_resolve(YAML_PATH)),
    f"找不到 {YAML_PATH}(单独跑 PaddleFleet 仓库时正常)",
)
class TestMoEConfigCheckFromYaml(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        yaml_path = _resolve(YAML_PATH)
        cls.merged, cls.yaml_keys, cls.json_keys = _load_flat_config(yaml_path)
        cls.user_keys = cls.yaml_keys | cls.json_keys
        cls.config = TransformerConfig.from_config(
            SimpleNamespace(**cls.merged)
        )
        cls.ep_size = (
            EXPERT_PARALLEL_SIZE
            if EXPERT_PARALLEL_SIZE is not None
            else cls.merged.get("expert_model_parallel_size", 1)
        )
        cls.capability = _device_capability()
        cls.plan = resolve_moe_plan(
            cls.config,
            expert_parallel_size=cls.ep_size,
            device_capability=cls.capability,
        )
        cls.findings = collect_findings(
            cls.config,
            cls.plan,
            cls.user_keys,
            device_capability=cls.capability,
        )

    def test_no_field_is_configured_against_the_selected_path(self):
        """报告全文打到 stdout, 只有存在必须修复的问题时才判失败。

        死配置、派生值、跨子系统字段都只是提示: 现存实验配置里有约 20 份在设置死配
        置, 让它们直接失败没有意义。真正判失败的是"用户显式配了但当前路径不读"和
        "用户显式配的值被路由改写"这两类。
        """
        print(
            "\n"
            + format_report(
                self.plan,
                self.findings,
                config=self.config,
                user_specified_keys=self.user_keys,
            )
        )

        must_fix = [f for f in self.findings if f.policy.value == "error"]
        self.assertEqual(
            must_fix,
            [],
            "\n\n以下问题需要改配置, 不是工具本身报错: 这些字段都是显式配置过的, 但"
            "当前执行路径不会读取它们, 或者配的值在路由过程中被改写掉了。\n"
            + "\n".join(str(f) for f in must_fix),
        )

    def test_yaml_does_not_redeclare_model_structure_fields(self):
        """ERNIEBot 的约定是模型结构字段只写在 JSON、其余只写在 YAML, 两边不得重
        叠(ernie5/pretrain.py 的 _check_yaml_not_override_json 也在查这件事)。重叠
        意味着同一个开关有两个可信来源, 后合并的那个静默生效。"""
        self.assertEqual(
            sorted(self.yaml_keys & self.json_keys),
            [],
            "这些字段同时出现在 YAML 和 model_config.json 里",
        )


if __name__ == "__main__":
    unittest.main()
