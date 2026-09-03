# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""MoE 配置开关的归属登记表。

MoE 子系统读取哪些 ``TransformerConfig`` 字段, 取决于路由决策变量选中了哪条执行
路径(EP size、``moe_token_dispatcher_type``、``moe_expert_fusion``、
``using_sonic_moe``、``fp8``、``use_w4a8``、设备算力)。归属于其他路径的字段不会
被任何代码读取, 因此配了等于没配, 而目前没有任何机制会告诉用户这件事。

本模块只有数据和查询辅助函数: 声明每个字段归哪个子模块所有、在什么条件下才真正
被消费。它在 import 期不依赖 paddle, 自身也不做任何校验, 因此可以脱离 GPU 做单
测。校验与报错建立在它之上。

归属按两级记录, 表示为点号路径: 由路由决策直接选中的顶层子模块(``router``、
``dispatcher.deepep`` ...), 以及子模块内部再次分支时命中的次级子模块
(``router.quantile_balancing``、``expert.grouped``、``precision.fp8``)。只写一级
表示该字段被这个子模块的全部次级分支共用。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "Stage",
    "Status",
    "Policy",
    "MoEFieldSpec",
    "MOE_FIELD_REGISTRY",
    "TOP_LEVEL_SUBMODULES",
    "SECONDARY_SUBMODULES",
    "DISPATCHER_SUBMODULES",
    "VALID_OWNERS",
    "is_secondary_owner",
    "spec_for",
    "fields_owned_by",
    "fields_with_status",
]


class Stage(str, Enum):
    """字段在哪个阶段被读取。

    构造期之后才读取的字段无法在启动时完整校验: 只有跑到那个阶段, 才知道这个值
    到底起不起作用。
    """

    CONSTRUCT = "construct"
    FORWARD = "forward"
    BACKWARD = "backward"
    OPTIMIZER_STEP = "optimizer_step"


class Status(str, Enum):
    """该字段究竟有没有消费者, 以及消费者在哪。"""

    ACTIVE = "active"
    """MoE 子系统至少有一条执行路径会读它。"""

    DEAD = "dead"
    """config 里声明了, 但全仓库没有任何消费者。"""

    FOREIGN = "foreign"
    """消费者在 MoE 之外(dense/TP 层、attention、CP)。配了确实有用, 只是对 MoE
    没用, 因此绝不能报成死配置。"""

    UNDECLARED = "undeclared"
    """通过 ``getattr(config, name, default)`` 读取, 没有声明成
    ``TransformerConfig`` 字段。这类字段名拼错会被 ``getattr`` 的默认值吸收,
    靠字段名比对是查不出来的。"""

    DERIVED = "derived"
    """根本不是用户开关: 由其他字段内部推导得出。在配置文件里写它, 会在任何代码
    读到之前就被覆盖。"""


class Policy(str, Enum):
    """该字段出现不匹配时, 报错应该有多严格。"""

    ERROR = "error"
    WARN = "warn"
    INFO = "info"


TOP_LEVEL_SUBMODULES = (
    "router",
    "dispatcher.alltoall",
    "dispatcher.deepep",
    "dispatcher.hybridep",
    "dispatcher.allgather",
    "expert",
    "shared_expert",
    "subbatch_recompute",
    "precision",
    # 不是被路由选中的子模块, 而是每条路径都会读的纯结构维度。放进登记表是为了
    # 不让任何 MoE 字段处于无人归属的状态。
    "architecture",
)

SECONDARY_SUBMODULES = {
    "router": (
        "greedy",
        "group_limited",
        "noaux_tc",
        "quantile_balancing",
        "hash",
    ),
    "expert": ("standard", "grouped", "sonic"),
    "precision": ("bf16", "fp8", "w4a8"),
}

DISPATCHER_SUBMODULES = tuple(
    name for name in TOP_LEVEL_SUBMODULES if name.startswith("dispatcher.")
)

VALID_OWNERS = frozenset(TOP_LEVEL_SUBMODULES) | frozenset(
    f"{top}.{secondary}"
    for top, secondaries in SECONDARY_SUBMODULES.items()
    for secondary in secondaries
)


def is_secondary_owner(owner):
    """``owner`` 指的是子模块内部的分支而非子模块本身时返回 True。
    ``dispatcher.deepep`` 属于顶层(四条 dispatcher 路径本身就是独立子模块),
    ``router.hash`` 属于次级。"""
    return owner not in TOP_LEVEL_SUBMODULES


@dataclass(frozen=True)
class MoEFieldSpec:
    """单个 MoE 配置字段的归属与生效条件元数据。"""

    name: str

    owner: tuple[str, ...]
    """会读取该字段的子模块路径。多于一项表示该字段被多条路径读取(例如
    ``moe_ep_barrier`` 被四条 dispatcher 路径共用)。"""

    activation: str
    """该字段何时生效的自然语言说明。"""

    requires: tuple[str, ...] = ()
    """该字段要起作用必须先满足的前置条件。自由文本, 供文档与报错解释使用; 其中可
    判定的条件已在 ``moe_config_check`` 的规则段里写成规则自动检查。"""

    unverifiable: tuple[str, ...] = ()
    """**在配置阶段无法判定**的前置条件, 因为它们取决于运行时状态(调用路径、实际
    token 数、能否拿到某个输入)。报告会把这些条件单独列出来提示人工确认——列在这里
    的条件必须确实判不了, 而不是"还没写规则", 否则就变成了拿噪音掩盖缺失的检查。"""

    conflicts: tuple[str, ...] = ()
    """已知不支持或互斥的组合。"""

    stage: Stage = Stage.CONSTRUCT
    status: Status = Status.ACTIVE
    policy: Policy = Policy.ERROR

    selector: bool = False
    """该字段是分支选择器: 它被无条件读取, 用来决定进入哪个子模块。

    选择器**不能**按"归属子模块是否命中"来判断作用域, 否则会得出自相矛盾的结论:
    ``n_shared_experts=0`` 会让 shared_expert 子模块不存在, 于是反过来报告
    "n_shared_experts 不会被读取"——而正是这个字段决定了它不存在。同理适用于
    ``moe_n_hash_layers=0``、``use_w4a8=False`` 等把自己的分支关掉的取值。

    这类字段配错的表现是"它开启的功能没生效", 只能靠 moe_config_check 里的规则
    逐条判定, 作用域检查对它们无能为力。
    """

    neutral_values: tuple = ()
    """作用域检查中视为"未请求任何行为"的真值。布尔开关写 False 天然是中性的,
    但有些数值字段的某个真值在语义上同样等价于关闭(``n_group=1`` 表示不分组):
    用户在不读该字段的路径下写这种值不构成需要修复的问题, 报 ERROR 会把无害的
    常见写法拦在 strict 模式外, 因此只降级为提示。"""

    suggestion: str = ""
    """报错时给用户的修复建议, 会被直接引用进诊断信息。"""

    doc_ref: str = ""
    """该归属结论的来源, 供审计核对。"""


def _f(name, owner, activation, **kwargs):
    """精简构造器: ``owner`` 既可以传单个点号路径, 也可以传元组。"""
    if isinstance(owner, str):
        owner = (owner,)
    return MoEFieldSpec(name=name, owner=owner, activation=activation, **kwargs)


_ROUTER_FIELDS = (
    _f(
        "num_experts_per_tok",
        "router",
        "总是生效; router 的每个次级分支都会读",
        doc_ref="CONFIGS 2.2",
    ),
    _f(
        "scoring_func",
        "router",
        "总是生效",
        requires=("router.hash 只支持 softmax / sigmoid / sqrtsoftplus",),
        doc_ref="CONFIGS 2.2",
    ),
    _f(
        "topk_method",
        "router",
        "总是生效; 它决定进入哪个 router 次级子模块",
        selector=True,
        doc_ref="CONFIGS 2.2",
    ),
    _f(
        "moe_router_load_balancing_type",
        "router",
        "总是生效, 但除 seq_aux_loss 之外的取值全部退化为同一套实现",
        policy=Policy.WARN,
        doc_ref="CONFIGS 2.2 / BRANCHES 5.3 issue#1",
    ),
    _f(
        "router_aux_loss_coef",
        "router",
        "总是生效; coef=0 等价于关闭该 loss",
        doc_ref="CONFIGS 2.2",
    ),
    _f("router_z_loss_coef", "router", "总是生效", doc_ref="CONFIGS 2.2"),
    _f(
        "norm_topk_prob",
        "router",
        "总是生效; 可与 routed_scaling_factor 叠加",
        doc_ref="CONFIGS 2.2",
    ),
)

_ROUTER_SECONDARY_FIELDS = (
    _f(
        "n_group",
        (
            "router.group_limited",
            "router.noaux_tc",
            "router.quantile_balancing",
        ),
        "group_limited_greedy 与 noaux_tc 用它做分组预选; quantile_balancing 只把"
        "它当作必须等于 1 的守卫来检查",
        requires=("n_routed_experts % n_group == 0",),
        conflicts=("quantile_balancing 下 n_group != 1 会直接报错",),
        neutral_values=(1,),
        suggestion="只有 topk_method=greedy 完全不读 n_group。与 noaux_tc 一起配"
        "是正确用法, 现有全部线上 model config 都是这么配的。",
        doc_ref="moe_router.py:1006, 1048, 1153; CONFIGS 2.2(已修正)",
    ),
    _f(
        "topk_group",
        ("router.group_limited", "router.noaux_tc"),
        "group_limited_greedy 与 noaux_tc 用它做分组预选",
        neutral_values=(1,),
        suggestion="topk_method=greedy 不读 topk_group; quantile_balancing 虽然会"
        "收到该参数但从不使用, 因为它要求 n_group == 1。",
        doc_ref="moe_router.py:1006, 1048",
    ),
    _f(
        "qb_n_bins",
        "router.quantile_balancing",
        "仅 topk_method=quantile_balancing 时生效",
        conflicts=(
            "quantile_balancing 下 moe_topk_fusion 被禁止",
            "n_group != 1",
            "router aux loss 系数非零",
        ),
        suggestion="qb_n_bins 不被任何其他 router 分支读取; 请改配 "
        "topk_method=quantile_balancing, 或删掉该字段。",
        doc_ref="CONFIGS 2.2",
    ),
    _f(
        "moe_n_hash_layers",
        "router.hash",
        ">0 时在命中的层上启用 hash 路由, 与 topk_method 正交叠加",
        selector=True,
        requires=("必须设置 actual_vocab_size",),
        unverifiable=(
            "input_ids 必须能传到该层(取决于运行时入口路径, 配置期判不了)",
        ),
        doc_ref="CONFIGS 2.2 / GPT 5 P0",
    ),
    _f(
        "actual_vocab_size",
        "router.hash",
        "仅 hash 路由下生效",
        requires=("moe_n_hash_layers > 0",),
        suggestion="actual_vocab_size 只被 hash 路由读取; moe_n_hash_layers 不 "
        "> 0 时它不起任何作用。",
        doc_ref="CONFIGS 2.2",
    ),
)

_ROUTER_FUSION_FIELDS = (
    _f(
        "routed_scaling_factor",
        "router",
        "总是生效",
        doc_ref="CONFIGS 2.2",
    ),
    _f(
        "routed_scaling_factor_learnable",
        "router",
        "总是生效",
        doc_ref="CONFIGS 2.2",
    ),
    _f(
        "moe_router_force_load_balancing",
        "router",
        "总是生效, 但它是诊断用的强制覆盖, 不是训练策略",
        policy=Policy.WARN,
        suggestion="moe_router_force_load_balancing 会用均衡分配替换真实路由结果, "
        "不应在正式训练中保持开启。",
        doc_ref="CONFIGS 2.2 / GPT 5 P1",
    ),
    _f(
        "moe_split_feature_routing",
        "router",
        "总是生效",
        requires=("scoring_func 必须是 sigmoid",),
        conflicts=("不作用于 hash 路由",),
        doc_ref="CONFIGS 2.2",
    ),
    _f(
        "moe_topk_fusion",
        "router",
        "条件生效: 只有 router.noaux_tc 提供完整支持",
        requires=("依赖 e_score_correction_bias, 只有 noaux_tc 会构造它",),
        conflicts=(
            "greedy / group_limited 缺少构造期校验, 会在运行期抛 AttributeError",
            "quantile_balancing 直接禁止该组合",
        ),
        stage=Stage.FORWARD,
        suggestion="在 greedy / group_limited 下要等到 forward 才会失败; 请改配 "
        "topk_method=noaux_tc, 或关闭 moe_topk_fusion。",
        doc_ref="CONFIGS 2.2 / BRANCHES 5.3 issue#3",
    ),
    _f(
        "routing_map_fusion",
        "router",
        "条件生效; 与 reference 路径语义等价",
        policy=Policy.INFO,
        doc_ref="CONFIGS 2.2",
    ),
    _f(
        "moe_router_fusion",
        "router",
        "从不生效: 字段有声明但没有任何消费者",
        status=Status.DEAD,
        policy=Policy.WARN,
        suggestion="moe_router_fusion 没有任何读取点, 请删除。router 侧的融合请用 "
        "moe_topk_fusion / routing_map_fusion。",
        doc_ref="CONFIGS 2.2 / REPORT 11.4",
    ),
)

_ALL_DISPATCHERS = (
    "dispatcher.alltoall",
    "dispatcher.deepep",
    "dispatcher.hybridep",
    "dispatcher.allgather",
)

_DISPATCHER_FIELDS = (
    _f(
        "moe_token_dispatcher_type",
        _ALL_DISPATCHERS,
        "总是生效; EP > 1 时由它选中 dispatcher 路径",
        conflicts=(
            "设备算力低于 9.0 时 deepep / hybridep 会静默回退为 alltoall",
            "hybridep runtime 缺失时直接抛 ImportError",
            "allgather 要求 using_sonic_moe=True",
        ),
        doc_ref="CONFIGS 2.3-2.6",
    ),
    _f(
        "moe_ep_barrier",
        _ALL_DISPATCHERS,
        "EP > 1 时总是生效; 位于通信热路径上",
        doc_ref="CONFIGS 2.3-2.6 / REPORT 11.5",
    ),
    _f(
        "moe_use_fusion_node",
        ("dispatcher.deepep", "dispatcher.hybridep", "dispatcher.allgather"),
        "条件生效: deepep / hybridep 支持, allgather 强制开启, alltoall 强制关闭",
        conflicts=(
            "alltoall 会强制关闭它, 并拒绝 moe_expert_fusion=True",
            "与 moe_expert_fusion 之间缺少成组校验",
        ),
        suggestion="在 alltoall 下这个开关会被覆盖; 需要融合引擎请改用 deepep 或 "
        "hybridep。",
        doc_ref="CONFIGS 2.3-2.6 / REPORT 11.10 issue#1",
    ),
    _f(
        "deepep_buffer_configs",
        "dispatcher.deepep",
        "仅 deepep 下生效; 它会改动进程级的 DeepEP buffer 大小",
        suggestion="deepep_buffer_configs 不被任何其他 dispatcher 读取。",
        doc_ref="CONFIGS 2.4",
    ),
    _f(
        "use_rr_deepep_combine",
        "dispatcher.deepep",
        "派生值: 由 recompute_modules 是否含 'moe_combine' 推导, 从不从 config 读取",
        requires=(
            "moe_token_dispatcher_type=deepep",
            "moe_shared_expert_overlap=True",
            "已设置 recompute_granularity, 且 full 或 MLP recompute 处于激活状态, "
            "否则抛 ValueError",
        ),
        status=Status.DERIVED,
        policy=Policy.WARN,
        suggestion="在 config 里设置 use_rr_deepep_combine 不起作用: 它会被 "
        "recompute_modules['moe_combine'] 覆盖。请改配 recompute_modules。",
        doc_ref="moe_layer.py:622-660; CONFIGS 2.4(已修正)",
    ),
    _f(
        "hybridep_buffer_configs",
        "dispatcher.hybridep",
        "仅 hybridep 下生效, 且通过 getattr 读取而非声明成 config 字段",
        status=Status.UNDECLARED,
        policy=Policy.WARN,
        suggestion="hybridep_buffer_configs 不在 config schema 里, 所以字段名拼错"
        "会被 getattr 的默认值静默吸收。",
        doc_ref="CONFIGS 2.5 / moe_layer.py:507",
    ),
    _f(
        "moe_allgather_gate_overlap",
        "dispatcher.allgather",
        "仅 allgather 下生效(默认 True)",
        suggestion="其他所有 dispatcher 都会静默忽略该字段。",
        doc_ref="CONFIGS 2.6",
    ),
)

_EXPERT_FIELDS = (
    _f(
        "moe_expert_fusion",
        "expert",
        "总是生效; 它决定使用 standard 还是 grouped expert 实现",
        selector=True,
        conflicts=(
            "alltoall 会拒绝该组合",
            "allgather 会强制开启它",
            "fp8 且无 deep_gemm、非 sonic 时, 对象是按局部回退值构造的, 而 "
            "forward 仍读该字段, 两者不一致",
        ),
        suggestion="expert 的次级分支由多个布尔字段组合隐式决定, 而不是单一枚举; "
        "请看 effective plan 的判定结果, 不要只看这个字段。",
        doc_ref="CONFIGS 2.7 / REPORT 11.10 issue#2",
    ),
    _f(
        "moe_deep_gemm",
        "expert.grouped",
        "仅在 grouped expert 内部生效",
        selector=True,
        requires=(
            "moe_expert_fusion=True",
            "设备算力 >= 9.0",
            "非 allgather 路径, 因为 allgather 会强制关闭它",
        ),
        suggestion="moe_deep_gemm 默认 True 而 moe_expert_fusion 默认 False, 这对"
        "默认值本身自相矛盾, 结果是静默地不走 DeepGEMM。",
        doc_ref="CONFIGS 2.7 / REPORT 11.9",
    ),
    _f(
        "using_sonic_moe",
        "expert",
        "总是生效; 它决定是否使用 Sonic expert 实现",
        selector=True,
        requires=(
            "moe_expert_fusion=True",
            "只支持 SwiGLU 激活",
            "sonicmoe 生态库必须可 import",
        ),
        doc_ref="CONFIGS 2.7",
    ),
    _f(
        "moe_routed_expert_use_bias",
        "expert.standard",
        "只有 standard expert 支持 bias",
        conflicts=("grouped 与 sonic expert 会对 bias 直接 assert",),
        suggestion="routed expert 上的 bias 要求 moe_expert_fusion=False 且 "
        "using_sonic_moe=False; grouped 与 Sonic expert 不是忽略它, 而是 assert。",
        doc_ref="CONFIGS 2.7",
    ),
    _f(
        "fp8_weight_quant_format",
        "expert.sonic",
        "只有 Sonic expert 会读它",
        requires=("必须开启 fp8",),
        suggestion="standard 与 grouped expert 只会对该字段打一条 warning, 值并不"
        "会被应用。",
        doc_ref="CONFIGS 2.7 / REPORT 11.6",
    ),
    _f(
        "moe_dequant_input",
        "expert",
        "从不生效: fusion node 内部把 dequant_input 硬编码为 True",
        status=Status.DEAD,
        policy=Policy.WARN,
        suggestion="moe_dequant_input 没有任何读取点; fusion node 总是会做反量化。"
        "请从 config 里删除。",
        doc_ref="CONFIGS 2.7 / REPORT 11.4",
    ),
)

_SHARED_EXPERT_FIELDS = (
    _f(
        "n_shared_experts",
        "shared_expert",
        "决定 shared expert 是否存在",
        selector=True,
        conflicts=("默认值 None 会在需要与之相乘的路径上于构造期抛 TypeError",),
        doc_ref="CONFIGS 2.8 / REPORT 11.9",
    ),
    _f(
        "moe_shared_expert_gate",
        "shared_expert",
        "shared expert 存在时总是生效",
        requires=("n_shared_experts > 0",),
        suggestion="没有 shared expert 却配 shared expert gate, 不起任何作用。",
        doc_ref="CONFIGS 2.8 / GPT 5 P1",
    ),
    _f(
        "moe_shared_expert_overlap",
        "shared_expert",
        "条件生效; 四个条件必须同时满足(shared expert 存在、fusion node 开启、"
        "EP > 1、dispatcher 后端有 overlap 窗口)",
        requires=(
            "n_shared_experts > 0",
            "moe_use_fusion_node=True",
            "EP > 1",
        ),
        unverifiable=("不在 scheduler 路径上(取决于运行时调度方式)",),
        conflicts=("hybridep + fusion node 组合会在构造期强制关闭 overlap",),
        suggestion="前三个条件任一不满足时, overlap 会被静默跳过(既不报错也不打"
        "warning); hybridep 下则会被构造期强制关闭。",
        doc_ref="moe_layer.py:1358-1362 / BRANCHES 3.3",
    ),
)

_SUBBATCH_FIELDS = (
    _f(
        "use_auto_subbatch",
        "subbatch_recompute",
        "条件生效; 它启用自动分块路径",
        doc_ref="CONFIGS 2.9",
    ),
    _f(
        "auto_subbatch_mode",
        "subbatch_recompute",
        "只有与 use_auto_subbatch 一起配才生效",
        requires=(
            "use_auto_subbatch=True; 只设 mode 不起作用",
            "pre_permute 还额外要求 expert fusion",
        ),
        suggestion="单独配 auto_subbatch_mode 是无效的; 请同时打开 "
        "use_auto_subbatch。",
        doc_ref="CONFIGS 2.9",
    ),
    _f(
        "moe_subbatch_token_num_after_dispatch",
        "subbatch_recompute",
        "条件生效; 它选中固定分块路径",
        unverifiable=(
            "tile 对齐与 fusion/DeepGEMM token 数一致性取决于运行时实际 token 数",
        ),
        doc_ref="CONFIGS 2.9",
    ),
    _f(
        "moe_subbatch_token_num_before_dispatch",
        "subbatch_recompute",
        "从不生效: 字段有声明但没有任何消费者",
        status=Status.DEAD,
        policy=Policy.WARN,
        suggestion="请改用 moe_subbatch_token_num_after_dispatch; before_dispatch "
        "这一版没有任何读取点。",
        doc_ref="CONFIGS 2.9",
    ),
    _f(
        "recompute_moe_gate_up",
        "subbatch_recompute",
        "条件生效; 固定 subbatch 也会隐式打开它, 且不回写配置快照",
        status=Status.UNDECLARED,
        policy=Policy.WARN,
        suggestion="通过 getattr 读取, 也可由 recompute_granularity=selective 加 "
        "recompute_modules 推导出来; 字段名拼错会被静默吸收。",
        doc_ref="CONFIGS 2.9 / moe_layer.py:550",
    ),
    _f(
        "recompute_moe_premute",
        "subbatch_recompute",
        "条件生效",
        requires=("必须同时开启 recompute_moe_gate_up",),
        status=Status.UNDECLARED,
        policy=Policy.WARN,
        suggestion="通过 getattr 读取, 也可由 recompute_granularity=selective 加 "
        "recompute_modules 推导出来; 字段名拼错会被静默吸收。",
        doc_ref="CONFIGS 2.9 / moe_layer.py:557",
    ),
    _f(
        "moe_subbatch_diag",
        "subbatch_recompute",
        "总是生效; 纯观测项, 既不改变数学也不改变显存策略",
        policy=Policy.INFO,
        doc_ref="CONFIGS 2.9",
    ),
)

_PRECISION_FIELDS = (
    _f(
        "fp8",
        ("precision", "expert", "dispatcher.deepep"),
        "总是生效; 它选中精度分支, 并同时改变 dispatch payload、expert kernel 和 "
        "wgrad dtype",
        selector=True,
        requires=(
            "moe_use_fusion_node=True, 否则会触发 assert",
            "CUDA 版本不能是 12.6",
            "不能是 SiTU 激活, 该激活只支持 BF16",
        ),
        doc_ref="CONFIGS 2.10",
    ),
    _f(
        "use_w4a8",
        "precision.w4a8",
        "只有在 deepep + grouped expert + fusion + deep_gemm 的组合下才真正生效; "
        "其余路径下它只是关掉 fp8 dispatch",
        selector=True,
        requires=(
            "dispatcher.deepep",
            "expert.grouped, 且 moe_expert_fusion 与 moe_deep_gemm 均开启",
            "已开启 fp8",
        ),
        conflicts=(
            "hybridep 不会把该参数传给 kernel",
            "allgather 强制走 Sonic, 而 Sonic 不接收 W4A8 参数",
        ),
        suggestion="在其他任何路径上, use_w4a8 会静默退化成只关闭 fp8 dispatch, "
        "没有任何 W4A8 计算。",
        doc_ref="CONFIGS 2.10 / REPORT 11.9",
    ),
    _f(
        "use_w4a8_fused_quant",
        "precision.w4a8",
        "只有 use_w4a8 生效时才生效",
        requires=("use_w4a8=True",),
        suggestion="没有 use_w4a8 却配 use_w4a8_fused_quant, 不起任何作用。",
        doc_ref="CONFIGS 2.10 / GPT 5 P1",
    ),
    _f(
        "use_ue8m0",
        "precision.fp8",
        "只有在 SM100 且已开启 fp8 时才生效",
        requires=("设备算力为 10(Blackwell)", "已开启 fp8"),
        conflicts=("与 fp8 之间没有联动校验, 所以 BF16 下它是静默无效的",),
        suggestion="use_ue8m0 在非 SM100 硬件上会 assert, 但 fp8 关闭时只是被静默"
        "忽略。",
        doc_ref="CONFIGS 2.10 / REPORT 11.9",
    ),
    _f(
        "fp8_wgrad",
        "precision.fp8",
        "仅 fp8 下生效; 它同时决定 wgrad dtype 和派生出的 combine 精度",
        requires=("已开启 fp8",),
        stage=Stage.BACKWARD,
        suggestion="一个字段耦合了两个本应独立的精度维度(wgrad dtype 与 dispatch "
        "combine 精度), 二者无法分别设置。",
        doc_ref="CONFIGS 2.10",
    ),
    _f(
        "fp8_recipe",
        "precision",
        "MoE 不读它; 消费者是 dense/TP linear 层与 context parallel 的 padding 计算",
        status=Status.FOREIGN,
        policy=Policy.INFO,
        suggestion="fp8_recipe 不是死配置, 只是 MoE 路径不读它; 其他子系统目前只接"
        "受 'blockwise'。",
        doc_ref="tensor_parallel/layers.py:1453, fp8/quantization.py:43",
    ),
    _f(
        "full_fp8_computation",
        "precision",
        "MoE 不读它; 消费者是 dense/TP linear 层与 hybrid attention",
        status=Status.FOREIGN,
        policy=Policy.INFO,
        suggestion="full_fp8_computation 控制的是 dense 侧的 FP8 路径, 不是 MoE; "
        "要切换 MoE 精度请用 fp8。",
        doc_ref="tensor_parallel/layers.py:1433, dsv4_hybrid_attention.py:1109",
    ),
    _f(
        "fp8_dispatch",
        ("precision", "dispatcher.deepep", "dispatcher.hybridep"),
        "由 ``fp8 and not use_w4a8`` 派生; alltoall 还会额外把它强制关闭, 因为该路"
        "径不承载 FP8 payload",
        status=Status.DERIVED,
        policy=Policy.WARN,
        suggestion="fp8_dispatch 无法设置: 它跟随 fp8 与 use_w4a8。",
        doc_ref="moe_layer.py:202, moe_layer.py:352",
    ),
    _f(
        "fp8_dispatch_bwd",
        ("precision", "dispatcher.deepep"),
        "由 ``fp8_dispatch and using_sonic_moe and fp8_wgrad`` 派生",
        stage=Stage.BACKWARD,
        status=Status.DERIVED,
        policy=Policy.WARN,
        suggestion="fp8_dispatch_bwd 无法设置: 它跟随 fp8、using_sonic_moe 与 "
        "fp8_wgrad。",
        doc_ref="moe_layer.py:204",
    ),
)

_ARCHITECTURE_FIELDS = (
    _f(
        "n_routed_experts",
        "architecture",
        "总是生效; routed expert 的数量",
        policy=Policy.INFO,
        doc_ref="transformer_config.py:555",
    ),
    _f(
        "moe_intermediate_size",
        "architecture",
        "总是生效; expert FFN 的宽度",
        policy=Policy.INFO,
        doc_ref="transformer_config.py:571",
    ),
    _f(
        "moe_layer_freq",
        "architecture",
        "总是生效; 决定哪些层是 MoE 层",
        policy=Policy.INFO,
        doc_ref="transformer_config.py:594",
    ),
    _f(
        "moe_latent_size",
        "architecture",
        "条件生效; >0 时在 MoE block 两侧启用 latent 投影",
        policy=Policy.INFO,
        doc_ref="moe_layer.py:240",
    ),
)

# 基于容量的 token 丢弃从未接入这套 MoE 实现。与上面那些死配置不同, 这几个字段
# 设计文档里没有提到, 因此单独成组并附上各自的说明。
_UNWIRED_CAPACITY_FIELDS = (
    _f(
        "moe_expert_capacity_factor",
        "expert",
        "从不生效: 这套实现里没有基于容量的 token 丢弃",
        status=Status.DEAD,
        policy=Policy.WARN,
        suggestion="不存在任何容量上限, 所有被路由的 token 都会送到 expert。请删除"
        "该字段。",
        doc_ref="全仓库无消费者",
    ),
    _f(
        "moe_pad_expert_input_to_capacity",
        "expert",
        "从不生效: 这套实现里没有基于容量的 padding",
        status=Status.DEAD,
        policy=Policy.WARN,
        suggestion="它依赖 moe_expert_capacity_factor, 而后者同样没有消费者。请删除"
        "该字段。",
        doc_ref="全仓库无消费者",
    ),
    _f(
        "moe_token_drop_policy",
        "expert",
        "从不生效: 不会丢弃任何 token, 所以这个策略无人查询",
        status=Status.DEAD,
        policy=Policy.WARN,
        suggestion="请删除该字段; token 丢弃功能没有实现。",
        doc_ref="全仓库无消费者",
    ),
    _f(
        "moe_logging",
        "subbatch_recompute",
        "从不生效: 字段有声明但没有任何消费者",
        status=Status.DEAD,
        policy=Policy.WARN,
        suggestion="MoE 侧的观测请用 moe_subbatch_diag; moe_logging 没有任何读取"
        "点, 尽管大量实验配置仍在设置它。",
        doc_ref="全仓库无消费者",
    ),
    _f(
        "moe_extended_tp",
        "architecture",
        "从不生效: 在 ModelParallelConfig 里有声明但没有任何消费者",
        status=Status.DEAD,
        policy=Policy.WARN,
        suggestion="expert 侧的张量并行不由这个字段决定; 请删除。",
        doc_ref="model_parallel_config.py:87, 全仓库无消费者",
    ),
)


def _build_registry():
    registry = {}
    for group in (
        _ROUTER_FIELDS,
        _ROUTER_SECONDARY_FIELDS,
        _ROUTER_FUSION_FIELDS,
        _DISPATCHER_FIELDS,
        _EXPERT_FIELDS,
        _SHARED_EXPERT_FIELDS,
        _SUBBATCH_FIELDS,
        _PRECISION_FIELDS,
        _ARCHITECTURE_FIELDS,
        _UNWIRED_CAPACITY_FIELDS,
    ):
        for spec in group:
            if spec.name in registry:
                raise RuntimeError(
                    f"MoE 字段 {spec.name!r} 被重复登记: 每个字段只能声明一次, "
                    "多个归属子模块应写在同一条 owner 元组里"
                )
            registry[spec.name] = spec
    return registry


MOE_FIELD_REGISTRY: dict[str, MoEFieldSpec] = _build_registry()


def spec_for(name):
    """返回 ``name`` 对应的 spec; 该字段与 MoE 无关时返回 None。"""
    return MOE_FIELD_REGISTRY.get(name)


def fields_owned_by(owner, include_secondary=True):
    """返回归 ``owner`` 所有的字段名。

    ``fields_owned_by("router")`` 返回全部 router 次级分支共用的字段, 并默认额外
    带上归属于某个具体次级分支(如 ``router.quantile_balancing``)的字段; 传
    ``include_secondary=False`` 则只返回共用字段。
    """
    prefix = owner + "."
    names = []
    for spec in MOE_FIELD_REGISTRY.values():
        for own in spec.owner:
            if own == owner or (include_secondary and own.startswith(prefix)):
                names.append(spec.name)
                break
    return tuple(names)


def fields_with_status(status):
    """返回 status 为 ``status`` 的全部字段名。"""
    return tuple(
        spec.name
        for spec in MOE_FIELD_REGISTRY.values()
        if spec.status is status
    )
