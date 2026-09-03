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

"""MoE 配置检查: 路由推导、可判定规则、诊断报告与构造期入口。

MoE 会按路由决策(EP size、dispatcher 类型、expert 组合、精度、设备算力)只读取配置
字段的一个子集。归属于其他路径的字段配了也不会被读取, 而目前没有任何机制会告诉用
户这件事。本模块就是那个机制, 归属数据在 ``moe_config_registry`` 里。

按处理顺序分四段:

1. **路由推导** ``resolve_moe_plan()``: ``MoELayer.__init__`` 路由决策的只读镜像,
   输入 config 加上驱动决策的两个环境事实(EP size、设备算力), 输出命中了哪些子模块
   以及过程中改写了哪些字段。它不建任何东西、不改任何东西, 因此能在建层之前跑, 也
   能在 CPU 上覆盖任何单机装不出来的组合。
2. **可判定规则** ``MOE_CONFIG_RULES``: 登记表里的 ``requires`` / ``conflicts`` 是
   自由文本无法求值, 这里把其中**已对照源码核实**的部分写成可执行规则, 每条标注源
   码位置。相当一部分组合现有代码已经会报错, 只是报得晚(构造到一半、甚至跑到
   forward)且一次只报一个; 把它们前移到配置阶段并一次报全是主要价值。
3. **诊断报告** ``collect_findings()`` / ``format_report()``: 把计划与登记表对起
   来, 一次性给出全部结论。
4. **构造期入口** ``run_moe_config_check()``: 由
   ``TransformerConfig.__post_init__`` 调用, 三种模式见 ``MOE_CONFIG_CHECK_MODES``。

两个贯穿全模块的约定:

* **只有用户显式配置过的字段才参与诊断。** config 对象本身分不清"用户配的"和"框架
  默认值"(上游 PretrainedConfig 会把自身默认值一并灌进去, 实测约五成 MoE 字段会被
  误判), 所以显式字段集合必须由调用方传入; 拿不到时自动降级为只打印执行计划。
* **本模块不修改 config, 也不参与任何计算。** 除 strict 模式下抛
  ``MoEConfigError`` 之外没有任何副作用。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..activations import situ
from .moe_config_registry import MOE_FIELD_REGISTRY, Policy, Status

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

__all__ = [
    # 路由推导
    "Adjustment",
    "MoEPlan",
    "resolve_moe_plan",
    "VALID_DISPATCHER_TYPES",
    # 可判定规则
    "Rule",
    "MOE_CONFIG_RULES",
    "evaluate_rules",
    # 诊断报告
    "Finding",
    "MoEConfigError",
    "STATUS_LABELS",
    "collect_findings",
    "format_report",
    "format_errors",
    # 构造期入口
    "run_moe_config_check",
    "MOE_CONFIG_CHECK_MODES",
]


# ======================================================================
# 路由推导: MoELayer.__init__ 路由决策的只读镜像
# ======================================================================

VALID_DISPATCHER_TYPES = ("allgather", "alltoall", "deepep", "hybridep")


@dataclass(frozen=True)
class Adjustment:
    """一个生效值与配置请求值不一致的字段。"""

    name: str
    requested: object
    effective: object
    reason: str

    def __str__(self):
        return f"{self.name}: {self.requested!r} -> {self.effective!r} ({self.reason})"


@dataclass(frozen=True)
class MoEPlan:
    """一份配置最终落到的子模块组合, 以及为此付出的改写代价。"""

    expert_parallel_size: int

    dispatcher: str | None
    """选中的 dispatcher; EP <= 1 时为 None, 表示根本没有 dispatcher 参与。
    ``None`` 对诊断很重要: EP == 1 时无论 ``moe_token_dispatcher_type`` 配成什么,
    所有 dispatcher 专属字段都是无效的。"""

    router_branch: str
    hash_routing: bool
    expert_branch: str
    precision: str
    shared_expert: bool

    moe_use_fusion_node: bool
    moe_expert_fusion: bool
    """forward 实际读到的字段值。它不总是等于 ``expert_weights_fused``, 见该字段
    的说明。"""

    expert_weights_fused: bool
    """是否真的构造了 fused expert 权重。在 fp8 + ``moe_expert_fusion=True`` +
    无 DeepGEMM + 非 Sonic 的组合下, 构造期通过一个局部变量回退成了非 fused, 而
    ``moe_expert_fusion`` 仍保持原值, 于是两者不一致。"""

    moe_deep_gemm: bool
    moe_shared_expert_overlap: bool
    fp8_dispatch: bool
    fp8_dispatch_bwd: bool

    active_submodules: frozenset = field(default_factory=frozenset)
    adjustments: tuple = ()


_ROUTER_BRANCH_BY_TOPK_METHOD = {
    "greedy": "greedy",
    "group_limited_greedy": "group_limited",
    "noaux_tc": "noaux_tc",
    "quantile_balancing": "quantile_balancing",
}


def _device_capability_major(device_capability):
    """传 None 表示"非 CUDA 或算力未知", 此时跳过算力回退, 与
    ``paddle.is_compiled_with_cuda()`` 为 False 的行为一致。"""
    if device_capability is None:
        return None
    if isinstance(device_capability, int):
        return device_capability
    return device_capability[0]


def resolve_moe_plan(
    config,
    expert_parallel_size,
    device_capability=None,
    hybrid_ep_available=True,
):
    """推导一份配置会选中哪些 MoE 子模块。

    Args:
        config: 建层时会用的 ``TransformerConfig``。
        expert_parallel_size: EP 组大小; 为 1 表示没有 dispatcher 参与。
        device_capability: ``(major, minor)`` 或只给 major。传 None 则跳过算力回
            退, 对应非 CUDA 编译的情况。
        hybrid_ep_available: HybridEP runtime 能否 import。真实的层在无法 import
            时会抛 ImportError; 这里改为把回退记录下来, 好让诊断能一次给出完整
            图景, 而不是卡在第一个问题上就退出。

    Returns:
        MoEPlan。遇到不支持的组合也不会抛异常: 报错是调用方的职责, 而报错本身需要
        先有一份 plan 作为依据。
    """
    adjustments = []

    def adjust(name, requested, effective, reason):
        if requested != effective:
            adjustments.append(Adjustment(name, requested, effective, reason))
        return effective

    fp8 = getattr(config, "fp8", None)
    use_w4a8 = getattr(config, "use_w4a8", False)
    using_sonic_moe = getattr(config, "using_sonic_moe", False)
    fp8_wgrad = getattr(config, "fp8_wgrad", True)

    # moe_layer.py:202-206 -- 派生值, 从不从 config 读取。
    fp8_dispatch = bool(fp8) and not use_w4a8
    fp8_dispatch_bwd = fp8_dispatch and using_sonic_moe and bool(fp8_wgrad)

    moe_expert_fusion = getattr(config, "moe_expert_fusion", False)
    moe_use_fusion_node = getattr(config, "moe_use_fusion_node", True)
    moe_shared_expert_overlap = getattr(
        config, "moe_shared_expert_overlap", False
    )
    dispatcher_type = getattr(config, "moe_token_dispatcher_type", "deepep")

    # moe_layer.py:226-236 -- grouped GEMM 需要 fused expert 权重。
    moe_deep_gemm = getattr(config, "moe_deep_gemm", True)
    if moe_deep_gemm and not moe_expert_fusion:
        moe_deep_gemm = adjust(
            "moe_deep_gemm",
            True,
            False,
            "DeepGEMM 要求 moe_expert_fusion=True",
        )

    # moe_layer.py:309-324 -- Hopper 之前的回退, 与 EP size 无关。
    capability_major = _device_capability_major(device_capability)
    if capability_major is not None and capability_major < 9:
        if dispatcher_type in ("deepep", "hybridep"):
            dispatcher_type = adjust(
                "moe_token_dispatcher_type",
                dispatcher_type,
                "alltoall",
                f"设备算力 {capability_major}.x < 9.0",
            )
        if moe_deep_gemm:
            moe_deep_gemm = adjust(
                "moe_deep_gemm",
                True,
                False,
                f"设备算力 {capability_major}.x < 9.0",
            )

    if dispatcher_type == "hybridep" and not hybrid_ep_available:
        # 真实的层在这里会抛 ImportError; plan 继续往下走, 以便一次性报出配置的
        # 其余问题。
        dispatcher_type = adjust(
            "moe_token_dispatcher_type",
            "hybridep",
            "alltoall",
            "HybridEP runtime 不可用(真实的层会抛 ImportError)",
        )

    # moe_layer.py:327-352 -- 只有 EP > 1 时才存在 dispatcher。
    dispatcher = None
    if expert_parallel_size > 1:
        dispatcher = dispatcher_type
        if dispatcher in ("deepep", "hybridep"):
            if (
                dispatcher == "hybridep"
                and moe_use_fusion_node
                and moe_shared_expert_overlap
            ):
                moe_shared_expert_overlap = adjust(
                    "moe_shared_expert_overlap",
                    True,
                    False,
                    "HybridEP 后端没有 overlap 窗口",
                )
        elif dispatcher == "allgather":
            # moe_layer.py:855-887 -- allgather 会改写三个开关。
            moe_use_fusion_node = adjust(
                "moe_use_fusion_node",
                moe_use_fusion_node,
                True,
                "allgather 只能配合 fusion node 运行",
            )
            moe_expert_fusion = adjust(
                "moe_expert_fusion",
                moe_expert_fusion,
                True,
                "allgather 要求 fused expert 权重",
            )
            moe_deep_gemm = adjust(
                "moe_deep_gemm",
                moe_deep_gemm,
                False,
                "allgather 不支持 DeepGEMM",
            )
        else:
            moe_use_fusion_node = adjust(
                "moe_use_fusion_node",
                moe_use_fusion_node,
                False,
                "alltoall 没有融合引擎",
            )
            fp8_dispatch = adjust(
                "fp8_dispatch",
                fp8_dispatch,
                False,
                "alltoall 不承载 FP8 payload",
            )
            fp8_dispatch_bwd = fp8_dispatch_bwd and fp8_dispatch

    # moe_layer.py:374-393 -- 实际会构造出哪种 expert 权重。
    expert_weights_fused = moe_expert_fusion
    if fp8 and moe_expert_fusion and not moe_deep_gemm and not using_sonic_moe:
        expert_weights_fused = False
        adjustments.append(
            Adjustment(
                "moe_expert_fusion",
                moe_expert_fusion,
                False,
                "fp8 且无 DeepGEMM、非 Sonic 时构造的是非 fused expert 权重, 但该"
                "字段本身保持原值, forward 仍会读到 True(REPORT 11.10 issue#2)",
            )
        )

    if using_sonic_moe:
        expert_branch = "sonic"
    elif expert_weights_fused:
        expert_branch = "grouped"
    else:
        expert_branch = "standard"

    if use_w4a8:
        precision = "w4a8"
    elif fp8:
        precision = "fp8"
    else:
        precision = "bf16"

    topk_method = getattr(config, "topk_method", "greedy")
    router_branch = _ROUTER_BRANCH_BY_TOPK_METHOD.get(topk_method, topk_method)
    hash_routing = bool(getattr(config, "moe_n_hash_layers", 0))

    n_shared_experts = getattr(config, "n_shared_experts", None)
    shared_expert = bool(n_shared_experts)

    active = {
        "architecture",
        "router",
        f"router.{router_branch}",
        "expert",
        f"expert.{expert_branch}",
        "precision",
        f"precision.{precision}",
        "subbatch_recompute",
    }
    if hash_routing:
        active.add("router.hash")
    if dispatcher is not None:
        active.add(f"dispatcher.{dispatcher}")
    if shared_expert:
        active.add("shared_expert")

    return MoEPlan(
        expert_parallel_size=expert_parallel_size,
        dispatcher=dispatcher,
        router_branch=router_branch,
        hash_routing=hash_routing,
        expert_branch=expert_branch,
        precision=precision,
        shared_expert=shared_expert,
        moe_use_fusion_node=moe_use_fusion_node,
        moe_expert_fusion=moe_expert_fusion,
        expert_weights_fused=expert_weights_fused,
        moe_deep_gemm=moe_deep_gemm,
        moe_shared_expert_overlap=moe_shared_expert_overlap,
        fp8_dispatch=fp8_dispatch,
        fp8_dispatch_bwd=fp8_dispatch_bwd,
        active_submodules=frozenset(active),
        adjustments=tuple(adjustments),
    )


# ======================================================================
# 可判定规则: 已对照源码核实、能在配置期求值的条件
# ======================================================================


@dataclass(frozen=True)
class Rule:
    """一条可判定的规则。"""

    code: str
    """规则编号, 出现在报错信息里, 方便用户和维护者对同一条规则对话。"""

    category: str
    fields: tuple[str, ...]
    """规则涉及的字段。只要其中任意一个被用户显式配置过, 规则就参与判定。"""

    check: Callable
    """``(config, plan, capability_major) -> str | None``。返回问题描述表示命中,
    返回 None 表示通过。"""

    suggestion: str
    doc_ref: str
    policy: Policy = Policy.ERROR


def _dispatcher(plan):
    return plan.dispatcher


def _topk_method(config):
    return getattr(config, "topk_method", "greedy")


def _bias_enabled(config):
    """routed expert 的 bias 由 moe_routed_expert_use_bias 覆盖 use_bias;
    为 None 时沿用全局 use_bias(moe_layer.py:169-171)。"""
    override = getattr(config, "moe_routed_expert_use_bias", None)
    if override is not None:
        return bool(override)
    return bool(getattr(config, "use_bias", False))


def _cuda_version():
    try:
        import paddle

        return paddle.version.cuda()
    except Exception:  # pragma: no cover - 取不到就当作"未知,不判定"
        return None


def _sonic_moe_available():
    """返回 True/False, 环境里根本没有 paddlefleet_ops 时返回 None(不判定)。"""
    try:
        import paddlefleet_ops

        return bool(paddlefleet_ops.is_sonic_moe_available())
    except Exception:  # pragma: no cover
        return None


_DEPENDENCY_RULES = (
    Rule(
        code="B1",
        category="依赖缺失",
        fields=("use_w4a8_fused_quant",),
        check=lambda config, plan, cap: (
            "use_w4a8_fused_quant=True 但 use_w4a8=False, 融合量化 kernel 不会被调用"
            if getattr(config, "use_w4a8_fused_quant", False)
            and not getattr(config, "use_w4a8", False)
            else None
        ),
        suggestion="要么开启 use_w4a8, 要么去掉 use_w4a8_fused_quant。",
        doc_ref="fp8_utils.py:271-274, 仅 W4A8 量化路径会传入该参数",
    ),
    Rule(
        code="B2",
        category="依赖缺失",
        fields=("actual_vocab_size",),
        check=lambda config, plan, cap: (
            "actual_vocab_size 已设置但 moe_n_hash_layers=0, hash 路由没有启用"
            if getattr(config, "actual_vocab_size", None)
            and not getattr(config, "moe_n_hash_layers", 0)
            else None
        ),
        suggestion="hash 路由请设 moe_n_hash_layers>0; 否则删除 actual_vocab_size。",
        doc_ref="moe_router.py:1429",
    ),
    Rule(
        code="B3",
        category="依赖缺失",
        fields=("moe_n_hash_layers",),
        check=lambda config, plan, cap: (
            "moe_n_hash_layers>0 但未设置 actual_vocab_size, hash 层构造时会报错"
            if getattr(config, "moe_n_hash_layers", 0)
            and not getattr(config, "actual_vocab_size", None)
            else None
        ),
        suggestion="补上 actual_vocab_size。这条现有代码也会报, 但要等到 "
        "set_layer_number 调用时才报。",
        doc_ref="moe_router.py:1429-1432",
    ),
    Rule(
        code="B4",
        category="依赖缺失",
        fields=("moe_shared_expert_gate",),
        check=lambda config, plan, cap: (
            "moe_shared_expert_gate=True 但没有 shared expert(n_shared_experts 为空)"
            if getattr(config, "moe_shared_expert_gate", False)
            and not plan.shared_expert
            else None
        ),
        suggestion="门控在 shared expert 内部实现; 请设 n_shared_experts>0, 否则该"
        "开关无效。",
        doc_ref="moe_shared_expert.py:53",
    ),
    Rule(
        code="B5",
        category="依赖缺失",
        fields=("auto_subbatch_mode",),
        check=lambda config, plan, cap: (
            "auto_subbatch_mode 已设置但 use_auto_subbatch=False, 分块策略不会启用"
            if getattr(config, "auto_subbatch_mode", None)
            and not getattr(config, "use_auto_subbatch", False)
            else None
        ),
        suggestion="use_auto_subbatch 是总开关, 单独设 mode 不生效。",
        doc_ref="fusion_layer_utils.py:554-559",
    ),
    Rule(
        code="B6",
        category="依赖缺失",
        fields=("use_ue8m0",),
        check=lambda config, plan, cap: (
            "use_ue8m0=True 但 fp8 未开启, UE8M0 缩放格式不会被使用"
            if getattr(config, "use_ue8m0", False)
            and not getattr(config, "fp8", None)
            else None
        ),
        suggestion="UE8M0 是 FP8 的缩放格式; 不开 fp8 时它静默无效(代码里也没有联"
        "动校验)。",
        doc_ref="CONFIGS 2.10 / REPORT 11.9",
    ),
    Rule(
        code="B7",
        category="依赖缺失",
        fields=("moe_shared_expert_overlap",),
        check=lambda config, plan, cap: (
            "moe_shared_expert_overlap=True 但没有 shared expert"
            f"(n_shared_experts={getattr(config, 'n_shared_experts', None)!r})"
            if getattr(config, "moe_shared_expert_overlap", False)
            and not plan.shared_expert
            else None
        ),
        suggestion="overlap 把 shared expert 的计算塞进 combine 的通信窗口; 没有 "
        "shared expert 就没有东西可重叠, 该开关会被静默跳过。请设 "
        "n_shared_experts>0, 或去掉 moe_shared_expert_overlap。",
        doc_ref="moe_layer.py:1358-1362(四个条件的 and, 不满足则静默不重叠)",
    ),
    Rule(
        code="B8",
        category="依赖缺失",
        fields=("moe_shared_expert_overlap",),
        check=lambda config, plan, cap: (
            "moe_shared_expert_overlap=True 但最终 moe_use_fusion_node=False"
            if getattr(config, "moe_shared_expert_overlap", False)
            and plan.shared_expert
            and not plan.moe_use_fusion_node
            else None
        ),
        suggestion="overlap 窗口由 fusion node 提供; alltoall 会强制关闭 "
        "moe_use_fusion_node, 从而让 overlap 静默失效。",
        doc_ref="moe_layer.py:1358-1362",
    ),
    Rule(
        code="B9",
        category="依赖缺失",
        fields=("moe_shared_expert_overlap",),
        check=lambda config, plan, cap: (
            f"moe_shared_expert_overlap=True 但 EP={plan.expert_parallel_size}, "
            "没有专家并行通信可供重叠"
            if getattr(config, "moe_shared_expert_overlap", False)
            and plan.expert_parallel_size <= 1
            else None
        ),
        suggestion="overlap 要重叠的是 EP 的 dispatch/combine 通信; EP=1 时不存在这"
        "段通信, 该开关无效。",
        doc_ref="moe_layer.py:1358-1362",
    ),
)

_CONFLICT_RULES = (
    Rule(
        code="C1",
        category="互斥组合",
        fields=("topk_method", "moe_topk_fusion"),
        check=lambda config, plan, cap: (
            "topk_method=quantile_balancing 与 moe_topk_fusion=True 不兼容"
            if _topk_method(config) == "quantile_balancing"
            and getattr(config, "moe_topk_fusion", False)
            else None
        ),
        suggestion="quantile_balancing 明确禁止 topk fusion, 请设 "
        "moe_topk_fusion=False。",
        doc_ref="moe_router.py:455-460(构造期已 raise)",
    ),
    Rule(
        code="C2",
        category="互斥组合",
        fields=("topk_method", "n_group"),
        check=lambda config, plan, cap: (
            f"topk_method=quantile_balancing 要求 n_group=1, 实际 "
            f"{getattr(config, 'n_group', 1)}"
            if _topk_method(config) == "quantile_balancing"
            and getattr(config, "n_group", 1) != 1
            else None
        ),
        suggestion="QB 的直方图无法表达多组预选; 请设 n_group=1。",
        doc_ref="moe_router.py:1153(forward 期才 raise)",
    ),
    Rule(
        code="C3",
        category="互斥组合",
        fields=("topk_method", "moe_topk_fusion"),
        check=lambda config, plan, cap: (
            f"moe_topk_fusion=True 但 topk_method={_topk_method(config)}, 该分支不构造 "
            f"e_score_correction_bias"
            if getattr(config, "moe_topk_fusion", False)
            and _topk_method(config) in ("greedy", "group_limited_greedy")
            else None
        ),
        suggestion="只有 topk_method=noaux_tc 完整支持 moe_topk_fusion; 当前组合会在"
        "运行期抛 AttributeError, 而不是在构造期被拒绝。",
        doc_ref="BRANCHES 5.3 issue#3",
    ),
)

_PRECISION_CONFLICT_RULES = (
    Rule(
        code="C4",
        category="互斥组合",
        fields=("fp8", "moe_use_fusion_node", "moe_token_dispatcher_type"),
        check=lambda config, plan, cap: (
            "fp8 已开启但最终 moe_use_fusion_node=False"
            + (
                "(被 alltoall 强制关闭)"
                if _dispatcher(plan) == "alltoall"
                else ""
            )
            if getattr(config, "fp8", None) and not plan.moe_use_fusion_node
            else None
        ),
        suggestion="fp8 只能在 fusion node 上运行; 换 deepep/hybridep, 或关闭 fp8。",
        doc_ref="moe_layer.py:359(assert)",
    ),
    Rule(
        code="C5",
        category="互斥组合",
        fields=("fp8", "hidden_act"),
        check=lambda config, plan, cap: (
            "fp8 已开启但 hidden_act=situ, SiTU-GLU 融合只支持 BF16"
            if getattr(config, "fp8", None)
            and getattr(config, "hidden_act", None) is situ
            else None
        ),
        suggestion="SiTU 激活下请关闭 fp8。",
        doc_ref="moe_layer.py:209-213",
    ),
    Rule(
        code="C6",
        category="互斥组合",
        fields=("fp8", "moe_expert_fusion", "moe_deep_gemm"),
        check=lambda config, plan, cap: (
            "fp8 + moe_deep_gemm 要求 moe_expert_fusion=True"
            if getattr(config, "fp8", None)
            and getattr(config, "moe_deep_gemm", True)
            and not getattr(config, "moe_expert_fusion", False)
            else None
        ),
        suggestion="fp8 下的 k-grouped gemm 反向需要 fused 权重; 请开启 "
        "moe_expert_fusion, 或关闭 moe_deep_gemm。",
        doc_ref="moe_layer.py:375-382(ValueError)",
    ),
)

_PATH_CONFLICT_RULES = (
    Rule(
        code="C7",
        category="互斥组合",
        fields=("moe_token_dispatcher_type", "moe_expert_fusion"),
        check=lambda config, plan, cap: (
            "dispatcher 为 alltoall 但 moe_expert_fusion=True"
            if _dispatcher(plan) == "alltoall"
            and getattr(config, "moe_expert_fusion", False)
            else None
        ),
        suggestion="alltoall 没有融合引擎, 该组合会被直接拒绝; 请设 "
        "moe_expert_fusion=False, 或换 deepep/hybridep。注意算力低于 9.0 时 "
        "deepep/hybridep 会被回退成 alltoall, 从而间接触发这条。",
        doc_ref="moe_layer.py:348-351(ValueError)",
    ),
    Rule(
        code="C8",
        category="互斥组合",
        fields=("moe_token_dispatcher_type", "using_sonic_moe"),
        check=lambda config, plan, cap: (
            "dispatcher 为 allgather 但 using_sonic_moe=False"
            if _dispatcher(plan) == "allgather"
            and not getattr(config, "using_sonic_moe", False)
            else None
        ),
        suggestion="allgather 路径只实现了 SonicMoE 融合 kernel; 请开启 "
        "using_sonic_moe。",
        doc_ref="moe_layer.py:864-869(ValueError)",
    ),
    Rule(
        code="C9",
        category="互斥组合",
        fields=("moe_routed_expert_use_bias", "use_bias", "moe_expert_fusion"),
        check=lambda config, plan, cap: (
            f"expert 分支为 {plan.expert_branch} 但 routed expert 启用了 bias"
            if plan.expert_branch in ("grouped", "sonic")
            and _bias_enabled(config)
            else None
        ),
        suggestion="Grouped GEMM 尚不支持 bias; 请设 "
        "moe_routed_expert_use_bias=False, 或改用 standard expert。",
        doc_ref="moe_expert.py:192(assert)",
    ),
    Rule(
        code="C10",
        category="互斥组合",
        fields=("using_sonic_moe", "moe_expert_fusion"),
        check=lambda config, plan, cap: (
            "using_sonic_moe=True 但最终没有构造 fused expert 权重"
            if getattr(config, "using_sonic_moe", False)
            and not plan.expert_weights_fused
            else None
        ),
        suggestion="Sonic expert 必须使用 fused 权重; 请开启 moe_expert_fusion。",
        doc_ref="moe_layer.py:390-393(assert)",
    ),
)

_SCORING_FUNCS = (
    "softmax",
    "sigmoid",
    "tanh",
    "relu",
    "gelu",
    "leaky_relu",
    "sftplus",
    "sqrtsoftplus",
)
"""moe_router.py:536-553 逐个 elif 支持的取值, 其余走 NotImplementedError。"""

_NON_NEGATIVE_SCORING_FUNCS = (
    "softmax",
    "sigmoid",
    "relu",
    "sftplus",
    "sqrtsoftplus",
)
"""seq_aux_loss 要求非负打分函数(moe_router.py:351-361)。"""

_HASH_SCORING_FUNCS = ("softmax", "sigmoid", "sqrtsoftplus")
"""hash 路由支持的取值(moe_router.py:1423-1427)。"""

_ROUTER_VALUE_RULES = (
    Rule(
        code="C11",
        category="互斥组合",
        fields=("n_group", "n_routed_experts", "topk_method"),
        check=lambda config, plan, cap: (
            f"n_routed_experts={getattr(config, 'n_routed_experts', None)} 不能被 "
            f"n_group={getattr(config, 'n_group', 1)} 整除"
            if plan.router_branch in ("group_limited", "noaux_tc")
            and getattr(config, "n_routed_experts", None)
            and getattr(config, "n_group", 1)
            and config.n_routed_experts % config.n_group != 0
            else None
        ),
        suggestion="分组预选要求专家数能被组数整除; 请调整 n_group。",
        doc_ref="moe_router.py:1025 / 1067(assert)",
    ),
    Rule(
        code="C12",
        category="互斥组合",
        fields=("scoring_func",),
        check=lambda config, plan, cap: (
            f"scoring_func={getattr(config, 'scoring_func', None)!r} 不在支持列表内"
            if getattr(config, "scoring_func", "softmax") not in _SCORING_FUNCS
            else None
        ),
        suggestion=f"可选值: {', '.join(_SCORING_FUNCS)}。",
        doc_ref="moe_router.py:536-553(NotImplementedError)",
    ),
    Rule(
        code="C13",
        category="互斥组合",
        fields=("scoring_func", "moe_router_load_balancing_type"),
        check=lambda config, plan, cap: (
            "moe_router_load_balancing_type=seq_aux_loss 要求非负打分函数, 实际 "
            f"{getattr(config, 'scoring_func', None)!r}"
            if getattr(config, "moe_router_load_balancing_type", "aux_loss")
            == "seq_aux_loss"
            and getattr(config, "scoring_func", "softmax")
            not in _NON_NEGATIVE_SCORING_FUNCS
            else None
        ),
        suggestion=f"可选值: {', '.join(_NON_NEGATIVE_SCORING_FUNCS)}。",
        doc_ref="moe_router.py:351-361(ValueError)",
    ),
    Rule(
        code="C14",
        category="互斥组合",
        fields=("scoring_func", "moe_n_hash_layers"),
        check=lambda config, plan, cap: (
            f"hash 路由要求 scoring_func 属于 {_HASH_SCORING_FUNCS}, 实际 "
            f"{getattr(config, 'scoring_func', None)!r}"
            if plan.hash_routing
            and getattr(config, "scoring_func", "softmax")
            not in _HASH_SCORING_FUNCS
            else None
        ),
        suggestion="换成 softmax / sigmoid / sqrtsoftplus, 或关闭 hash 路由。",
        doc_ref="moe_router.py:1423-1427(ValueError)",
    ),
    Rule(
        code="C15",
        category="互斥组合",
        fields=("moe_split_feature_routing", "scoring_func"),
        check=lambda config, plan, cap: (
            "moe_split_feature_routing=True 要求 scoring_func='sigmoid', 实际 "
            f"{getattr(config, 'scoring_func', None)!r}"
            if getattr(config, "moe_split_feature_routing", False)
            and getattr(config, "scoring_func", "softmax") != "sigmoid"
            else None
        ),
        suggestion="双视图路由的契约是 sigmoid + sigmoid。",
        doc_ref="moe_router.py:1415-1418 / 1592-1595(ValueError)",
    ),
)

_CAPABILITY_RULES = (
    Rule(
        code="E1",
        category="能力不足",
        fields=("use_ue8m0",),
        check=lambda config, plan, cap: (
            f"use_ue8m0=True 需要 SM100(Blackwell), 当前设备算力为 {cap}.x"
            if getattr(config, "use_ue8m0", False)
            and cap is not None
            and cap != 10
            else None
        ),
        suggestion="换到 SM100 机型, 或关闭 use_ue8m0。",
        doc_ref="moe_layer.py:363-366(assert)",
    ),
    Rule(
        code="E2",
        category="能力不足",
        fields=("fp8",),
        check=lambda config, plan, cap: (
            "fp8 已开启但当前 CUDA 版本是 12.6, 该组合未实现"
            if getattr(config, "fp8", None) and _cuda_version() == "12.6"
            else None
        ),
        suggestion="换一个 CUDA 版本, 或关闭 fp8。",
        doc_ref="moe_layer.py:354-358(NotImplementedError)",
    ),
    Rule(
        code="E3",
        category="能力不足",
        fields=("using_sonic_moe",),
        check=lambda config, plan, cap: (
            "using_sonic_moe=True 但当前环境的 sonicmoe kernel 不可用"
            if getattr(config, "using_sonic_moe", False)
            and _sonic_moe_available() is False
            else None
        ),
        suggestion="装上 paddlefleet_ops.sonicmoe, 或关闭 using_sonic_moe。",
        doc_ref="moe_layer.py:217-222(assert)",
    ),
)

MOE_CONFIG_RULES = (
    _DEPENDENCY_RULES
    + _CONFLICT_RULES
    + _PRECISION_CONFLICT_RULES
    + _PATH_CONFLICT_RULES
    + _ROUTER_VALUE_RULES
    + _CAPABILITY_RULES
)


def evaluate_rules(config, plan, device_capability, user_specified_keys):
    """逐条判定规则, 返回 ``(rule, 问题描述)`` 列表。

    只判定那些至少涉及一个"用户显式配置过"字段的规则; 全是默认值的组合不算用户的
    问题, 报出来只会制造噪音。
    """
    capability_major = None
    if device_capability is not None:
        capability_major = (
            device_capability
            if isinstance(device_capability, int)
            else device_capability[0]
        )

    results = []
    user_keys = set(user_specified_keys)
    for rule in MOE_CONFIG_RULES:
        if not (set(rule.fields) & user_keys):
            continue
        problem = rule.check(config, plan, capability_major)
        if problem:
            results.append((rule, problem))
    return results


# ======================================================================
# 诊断报告: 把执行计划与归属登记表对起来
# ======================================================================

STATUS_LABELS = {
    Status.DEAD: "死配置",
    Status.FOREIGN: "属其他子系统",
    Status.DERIVED: "内部派生值",
    Status.UNDECLARED: "未声明字段",
}

SCOPE_LABEL = "作用域错误"
ADJUSTED_LABEL = "配置被改写"


@dataclass(frozen=True)
class Finding:
    """一条诊断结论。"""

    kind: str
    """给用户看的分类标签, 取自 ``STATUS_LABELS`` / ``SCOPE_LABEL`` /
    ``ADJUSTED_LABEL``。"""

    field: str
    value: object
    policy: Policy
    message: str

    def __str__(self):
        return (
            f"[{self.kind}] {self.field}={self.value!r}\n      {self.message}"
        )


def collect_findings(config, plan, user_specified_keys, device_capability=None):
    """比对一份配置与它命中的执行计划, 返回全部诊断结论。

    Args:
        config: 已构造好的 ``TransformerConfig``。
        plan: ``resolve_moe_plan`` 的返回值。
        user_specified_keys: 用户真正显式配过的字段名集合(YAML 键 ∪ JSON 键)。
            **必须由调用方提供**: config 对象本身分不清"用户配的"和"框架默认值",
            上游 PretrainedConfig 会把自己的默认值一并灌进去, 实测会让近半数 MoE
            字段被误判成用户显式设置。只诊断这个集合里的字段, 是避免误报的唯一
            办法。
        device_capability: ``(major, minor)`` 或 major; 判定 E 类(能力不足)规则时
            需要。传 None 则跳过这类规则。

    Returns:
        Finding 元组。**一次性返回全部问题**, 不在第一条就中断: 让用户改一轮就能
        全部改完, 而不是启动一次只暴露一个问题。
    """
    findings = []
    rule_covered_fields = set()

    for name in sorted(user_specified_keys):
        spec = MOE_FIELD_REGISTRY.get(name)
        if spec is None:
            # 与 MoE 无关的字段(数据、优化器、并行策略等), 不在本工具职责内。
            continue

        if spec.status is not Status.ACTIVE:
            findings.append(
                Finding(
                    kind=STATUS_LABELS[spec.status],
                    field=name,
                    value=getattr(config, name, None),
                    policy=spec.policy,
                    message=spec.suggestion or spec.activation,
                )
            )
            continue

        if not (set(spec.owner) & plan.active_submodules):
            if spec.selector:
                # 分支选择器被无条件读取, 是它决定自己那个分支存不存在。把
                # n_shared_experts=0 报成"归属 shared_expert, 不会被读取"是倒因为
                # 果; 这类字段只能靠本模块的规则段判定。
                continue
            value = getattr(config, name, None)
            # 关掉一个本来就不生效的开关不会造成任何后果: 用户没有要求任何行为, 只
            # 是把它显式写成了关。这种情况降级为提示, 免得把"必须修复"的列表撑满无
            # 需修复的条目。
            requested_something = bool(value)
            findings.append(
                Finding(
                    kind=SCOPE_LABEL,
                    field=name,
                    value=value,
                    policy=spec.policy if requested_something else Policy.INFO,
                    message=(
                        (
                            f"该字段归属 {'/'.join(spec.owner)}, 当前执行路径不包含"
                            f"这些子模块, 因此不会被读取。{spec.suggestion}"
                            if requested_something
                            else f"该字段归属 {'/'.join(spec.owner)}, 当前路径不读"
                            f"取; 但配的是 {value!r}(等于没有请求该行为), 无需修改。"
                        ).strip()
                    ),
                )
            )

    for adjustment in plan.adjustments:
        if adjustment.name not in user_specified_keys:
            # 用户没配过, 只是默认值与路由要求不同, 不构成"配置被改写"。
            continue
        findings.append(
            Finding(
                kind=ADJUSTED_LABEL,
                field=adjustment.name,
                value=adjustment.requested,
                policy=Policy.ERROR,
                message=(
                    f"你配置的 {adjustment.name}={adjustment.requested!r} 被改写为 "
                    f"{adjustment.effective!r}, 原因: {adjustment.reason}"
                ),
            )
        )

    for rule, problem in evaluate_rules(
        config, plan, device_capability, user_specified_keys
    ):
        involved = [name for name in rule.fields if name in user_specified_keys]
        rule_covered_fields.update(involved)
        findings.append(
            Finding(
                kind=rule.category,
                field="/".join(involved) or rule.fields[0],
                value=tuple(getattr(config, name, None) for name in involved),
                policy=rule.policy,
                message=f"[{rule.code}] {problem}。{rule.suggestion} "
                f"(依据 {rule.doc_ref})",
            )
        )

    # 同一个字段既命中通用作用域判定、又命中具体规则时, 只留规则那条: 规则给出的
    # 是"缺哪个前置条件、怎么修", 比"归属的子模块没命中"更具体。
    findings = [
        finding
        for finding in findings
        if not (
            finding.kind == SCOPE_LABEL and finding.field in rule_covered_fields
        )
    ]
    return tuple(findings)


class MoEConfigError(ValueError):
    """MoE 配置校验失败。消息里包含**本轮全部**必须修复的问题。"""

    def __init__(self, findings, plan=None):
        self.findings = tuple(findings)
        self.plan = plan
        super().__init__(format_errors(self.findings, plan))


def format_errors(findings, plan=None):
    """把必须修复的问题排成一条多行消息。

    一次性列出全部问题是刻意设计: 逐个报错会让用户反复"改一个、跑一次、再报一个",
    在需要排队等资源的训练任务上代价很高。
    """
    errors = [f for f in findings if f.policy is Policy.ERROR]
    lines = [f"MoE 配置检查未通过, 共 {len(errors)} 个问题需要修复:"]
    if plan is not None:
        lines.append(
            f"当前执行路径: dispatcher={plan.dispatcher} router={plan.router_branch} "
            f"expert={plan.expert_branch} precision={plan.precision} "
            f"(EP={plan.expert_parallel_size})"
        )
    for index, finding in enumerate(errors, start=1):
        lines.append(f"  {index}. {finding}")
    lines.append(
        "以上问题一次性全部列出, 请一并修改; 如需临时放行, 把 "
        "moe_config_check 设为 report。"
    )
    return "\n".join(lines)


def format_report(plan, findings, config=None, user_specified_keys=()):
    """把执行计划与诊断结论排成一份可读报告。

    ``config`` 与 ``user_specified_keys`` 都给出时, 报告末尾会附上"需人工确认的前置
    条件": 那些用户配过、当前生效、但登记表标注了 ``requires`` 的字段。这些条件是
    自然语言, 本工具不做求值, 只负责提醒。
    """
    lines = ["=" * 72, "MoE 配置检查报告", "=" * 72]

    lines.append("")
    lines.append(f"EP size            : {plan.expert_parallel_size}")
    lines.append(
        f"dispatcher         : {plan.dispatcher if plan.dispatcher else 'none(EP<=1, 不走 dispatcher)'}"
    )
    lines.append(
        f"router             : {plan.router_branch}"
        + ("  (叠加 hash 路由)" if plan.hash_routing else "")
    )
    lines.append(f"expert             : {plan.expert_branch}")
    lines.append(f"precision          : {plan.precision}")
    lines.append(f"shared expert      : {'有' if plan.shared_expert else '无'}")
    lines.append(
        f"派生值             : fp8_dispatch={plan.fp8_dispatch} "
        f"fp8_dispatch_bwd={plan.fp8_dispatch_bwd}"
    )

    lines.append("")
    lines.append(
        f"-- 路由过程中被改写的字段({len(plan.adjustments)} 项) " + "-" * 20
    )
    if plan.adjustments:
        for adjustment in plan.adjustments:
            lines.append(f"  {adjustment}")
    else:
        lines.append("  (无)")

    errors = [f for f in findings if f.policy is Policy.ERROR]
    others = [f for f in findings if f.policy is not Policy.ERROR]

    lines.append("")
    lines.append(f"-- 需要修复的问题({len(errors)} 项) " + "-" * 28)
    if errors:
        for finding in errors:
            lines.append(f"  {finding}")
    else:
        lines.append("  (无)")

    lines.append("")
    lines.append(f"-- 提示({len(others)} 项) " + "-" * 38)
    if others:
        for finding in others:
            lines.append(f"  {finding}")
    else:
        lines.append("  (无)")

    if config is not None:
        pending = []
        for name in sorted(user_specified_keys):
            spec = MOE_FIELD_REGISTRY.get(name)
            if spec is None or spec.status is not Status.ACTIVE:
                continue
            if not spec.unverifiable:
                continue
            if not (set(spec.owner) & plan.active_submodules):
                continue
            value = getattr(config, name, None)
            if not value:
                # 配成关的字段不会请求任何行为, 它的前置条件也就无需确认。
                continue
            pending.append(
                f"  {name}={value!r}: " + "; ".join(spec.unverifiable)
            )
        lines.append("")
        lines.append(
            f"-- 依赖运行时状态、无法在配置期判定的条件({len(pending)} 项) "
            + "-" * 6
        )
        lines.extend(pending or ["  (无)"])

    lines.append("=" * 72)
    return "\n".join(lines)


# ======================================================================
# 构造期入口: 由 TransformerConfig.__post_init__ 调用
# ======================================================================

MOE_CONFIG_CHECK_MODES = ("off", "report", "strict")

_REPORTED_FINGERPRINTS = set()


def _is_first_rank():
    """构造期分布式通信组可能还没初始化, 所以看启动器设置的环境变量而不是
    ``paddle.distributed.get_rank()``。"""
    return os.environ.get("PADDLE_TRAINER_ID", "0") == "0"


def _device_capability():
    try:
        import paddle

        if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count():
            return paddle.device.cuda.get_device_capability()
    except Exception:  # pragma: no cover - 取不到就退化为"算力未知"
        pass
    return None


def run_moe_config_check(
    config,
    mode=None,
    expert_parallel_size=None,
    device_capability=None,
    user_specified_keys=None,
):
    """执行一次配置检查。

    Args:
        config: 已完成 ``__post_init__`` 主体的 ``TransformerConfig``。
        mode: 覆盖 ``config.moe_config_check``; 取值见 ``MOE_CONFIG_CHECK_MODES``。
        expert_parallel_size: 覆盖 ``config.expert_model_parallel_size``。
        device_capability: 覆盖自动探测的设备算力。
        user_specified_keys: 覆盖 ``config.user_specified_keys``。没有这个集合时,
            只会打印执行计划而不做字段级诊断——否则框架默认值会被当成用户配置,
            实测误报率约五成。

    Returns:
        ``(plan, findings)``; 模式为 off 或不是 MoE 模型时返回 ``(None, ())``。

    Raises:
        MoEConfigError: 仅在 strict 模式且存在必须修复的问题时。
    """
    if mode is None:
        mode = getattr(config, "moe_config_check", "off") or "off"
    if mode not in MOE_CONFIG_CHECK_MODES:
        raise ValueError(
            f"moe_config_check 必须是 {MOE_CONFIG_CHECK_MODES} 之一, 收到 {mode!r}"
        )
    if mode == "off":
        return None, ()
    if not getattr(config, "n_routed_experts", None):
        # 稠密模型, 没有 MoE 层可检查。
        return None, ()

    if expert_parallel_size is None:
        expert_parallel_size = (
            getattr(config, "expert_model_parallel_size", 1) or 1
        )
    if device_capability is None:
        device_capability = _device_capability()
    if user_specified_keys is None:
        user_specified_keys = getattr(config, "user_specified_keys", None)

    plan = resolve_moe_plan(
        config,
        expert_parallel_size=expert_parallel_size,
        device_capability=device_capability,
    )

    findings = ()
    if user_specified_keys:
        findings = collect_findings(
            config,
            plan,
            set(user_specified_keys),
            device_capability=device_capability,
        )

    # 同一份配置可能被构造多次(provider、deepcopy、多次 from_config), 相同结论只打
    # 一遍, 否则日志会被刷屏。
    fingerprint = (
        plan.dispatcher,
        plan.router_branch,
        plan.expert_branch,
        plan.precision,
        plan.expert_parallel_size,
        tuple(str(adjustment) for adjustment in plan.adjustments),
        tuple(str(finding) for finding in findings),
    )
    if fingerprint not in _REPORTED_FINGERPRINTS:
        _REPORTED_FINGERPRINTS.add(fingerprint)
        if _is_first_rank():
            report = format_report(
                plan,
                findings,
                config=config,
                user_specified_keys=user_specified_keys or (),
            )
            logger.info("MoE 配置检查:\n%s", report)
            if not user_specified_keys:
                logger.info(
                    "未提供 user_specified_keys, 已跳过字段级诊断。上游可以把 "
                    "YAML 键与 model_config.json 键的并集赋给 "
                    "config.user_specified_keys 来启用。"
                )

    if mode == "strict":
        errors = [f for f in findings if f.policy.value == "error"]
        if errors:
            raise MoEConfigError(findings, plan)

    return plan, findings
