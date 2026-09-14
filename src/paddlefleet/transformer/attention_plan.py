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

"""Attention 治理管道（总体方案 §2.1 的 ①-④ 步）。

在加载之后、层构建之前，对扁平 ``TransformerConfig`` 执行一条显式管道：

    ① Normalizer  —— 字段归属注册表驱动的无损归一（别名、类型强转）
    ② Validator   —— A/B 系列组合校验（只告警模式：默认只告警不报错）
    ③ Resolver    —— 逐层产出 ``AttentionExecutionPlan``（契约见方案 §2.1.1）
    ④ Plan 打印   —— rank0 打印 ``[ATTN-PLAN]`` 摘要表 + JSON 落盘

本模块是 **sidecar**：``gpt_layer_specs`` / ``TransformerBlock`` 的构建链路
完全不消费这里的 Plan（P3 阶段才做等价重构）。Resolver 的决策树逐条复刻
``gpt_layer_specs.get_gpt_layer_local_spec`` 的现状优先级，仅用于观测与快照，
不改任何数值行为。
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from paddlefleet.transformer.utils import is_layer_window_attention

__all__ = [
    "AttentionExecutionPlan",
    "AttentionPlanBundle",
    "ATTENTION_FIELD_REGISTRY",
    "normalize_attention_config",
    "resolve_attention_plan",
    "run_attention_plan",
]

# ---------------------------------------------------------------------------
# ① AttentionFieldRegistry —— 字段归属注册表（单一事实源，雏形）
# ---------------------------------------------------------------------------

Owner = Literal[
    "common",  # 通用字段，所有 attention 家族可读
    "mla",  # 仅 MLA（含 dsv4 的 -2 层）
    "dsv4",  # 仅 dsv4_hybrid 变体（CSA/HCA/window 层）
    "dsa",  # 仅 DSA indexer
    "vha",  # 仅 VHA
    "swa",  # 仅 SWA 层 override
    "business",  # 跨仓业务字段，PaddleFleet 不读取（显式隔离，如 use_attn_merger）
    "alias",  # 已废弃别名（经 Normalizer 归一）
]


@dataclass(frozen=True)
class FieldSpec:
    """一条字段归属声明（方案 §2.2-(1)）。"""

    name: str
    owner: Owner
    type: str = "unknown"  # "int" / "float" / "bool" / "str" / "list"
    aliases: tuple[str, ...] = ()
    deprecated: bool = False
    applies_to: str = ""  # 生效条件说明（人读 + 无人消费检查依据）


# 注册表按 owner 分类登记：一个 owner 一个注册块，一行一个字段。
# 条目格式：(字段名, 类型, 生效条件说明) 或追加第 4 项 dict 传
# aliases/deprecated 等差异项。首版只登记治理对象字段（A/B/C/D 系列），
# 全量登记在 P2/P4 扩展。
_OWNER_FIELDS: dict[Owner, list[tuple]] = {
    "common": [
        # --- 变体选择（A 系列） ---
        (
            "experimental_attention_variant",
            "str",
            "非 None 时抢占 MLA flag 与 attention_layer_type",
        ),
        (
            "multi_latent_attention",
            "bool",
            "仅当 experimental_attention_variant != dsv4_hybrid 时生效",
        ),
        ("layer_types", "list", "仅非 dsv4 路径逐层类型来源"),
        # --- RoPE 基础（B 系列，详见 B_rope_config.md §3.1） ---
        ("rope_theta", "float", "唯一 RoPE base 主字段（B2 归一后）"),
        ("rope_type", "str", "MLA/DSA 路径的 rope|yarn 选择；dsv4 按层覆盖"),
        (
            "rotary_interleaved",
            "bool",
            "唯一 interleave 主开关；DSv4/CSA eager 路径现状忽略",
        ),
        ("rotary_percent", "float", "标准路径推导 rope 宽度（derived）"),
        # --- QK Norm ---
        (
            "use_qk_norm",
            "bool",
            "self/MLA 路径的 q/k LayerNorm 开关；变体：qk_l2_norm（L2 范数，"
            "MLA 不支持）、qk_norm_fusion（triton 融合）；dsv4 路径另读 "
            "qk_layernorm（默认 True）",
        ),
    ],
    "alias": [
        (
            "rotary_base",
            "float",
            "rope_theta 的废弃别名（仅 dsv4/kimi_k3 路径读取）。B2 "
            "(rope_config 分支) 合入后 __post_init__ 校验并剥离；当前 "
            "develop 上仍是普通字段，Normalizer 不做归一以免改变行为",
            {"deprecated": True},
        ),
    ],
    "mla": [
        ("q_lora_rank", "int", ""),
        ("kv_lora_rank", "int", ""),
        ("qk_nope_head_dim", "int", ""),
        (
            "qk_rope_head_dim",
            "int",
            "MLA 路径 rope 宽度；DSv4/CSA 层不读（V-ROPE-01）",
        ),
    ],
    "dsv4": [
        (
            "qk_pos_emb_head_dim",
            "int",
            "DSv4/CSA/HCA/window 层 rope 宽度；None → 0 即 RoPE 关闭（V-ROPE-02）",
        ),
        (
            "csa_compress_rotary_base",
            "float",
            "压缩层（ratio > 1）base；JSON 里常为字符串（B5 强转）",
        ),
        ("hca_rope_type", "str", "ratio == 128 层的 rope|yarn 覆盖"),
        ("csa_rope_type", "str", "2 <= ratio < 128 层的 rope|yarn 覆盖"),
        (
            "csa_compress_ratios",
            "list",
            "dsv4_hybrid 下的逐层类型来源（覆盖 layer_types，A2）",
        ),
        ("csa_window_size", "int", "CSA/window 层内部窗口"),
    ],
    "dsa": [
        (
            "dsa_index_n_heads",
            "int",
            "非 None 时 MLA core 切到 DSA",
            {"aliases": ("index_n_heads",)},
        ),
        ("dsa_index_head_dim", "int", "", {"aliases": ("index_head_dim",)}),
        ("dsa_index_topk", "int", "", {"aliases": ("index_topk",)}),
        (
            "dsa_indexer_rotary_interleaved",
            "bool",
            "indexer 独立布局开关（B4）",
        ),
    ],
    "vha": [
        (
            "use_vha_attention",
            "bool",
            "同一开关三种语义：self family 换整个类为 SelfAttentionVHA"
            "（gpt_layer_specs:230）；mla/MQA 与 dsv4 层只复用为 postmix"
            "（multi_latent_attention:667、dsv4_hybrid_attention:913）+ 可选 "
            "premix（use_vha_premix AND，dsv4:1659）；gdn/kda/gemma4 不读",
        ),
        (
            "use_vha_premix",
            "bool",
            "仅 dsv4 CSA/HCA/window 层生效，且需 use_vha_attention 同时为 True",
        ),
    ],
    "swa": [
        ("swa_qk_rope_head_dim", "int", ""),
        ("swa_rope_theta", "float", ""),
    ],
    "business": [
        ("use_attn_merger", "bool", "跨仓业务字段，PaddleFleet 不读取"),
    ],
}


def _build_field_registry() -> dict[str, FieldSpec]:
    registry: dict[str, FieldSpec] = {}
    for owner, entries in _OWNER_FIELDS.items():
        for entry in entries:
            name, type_, applies_to, *extra = entry
            spec = FieldSpec(
                name=name,
                owner=owner,
                type=type_,
                applies_to=applies_to,
                **(extra[0] if extra else {}),
            )
            registry[name] = spec
    return registry


ATTENTION_FIELD_REGISTRY: dict[str, FieldSpec] = _build_field_registry()

_ALIAS_TO_CANONICAL: dict[str, str] = {
    alias: spec.name
    for spec in ATTENTION_FIELD_REGISTRY.values()
    for alias in spec.aliases
}


def _explicitly_set(config: Any, name: str) -> bool:
    """字段是否被显式配置为非默认值（dataclass 默认值比对）。

    ``__post_init__`` 会就地推导部分字段（如 ``swa_head_dim``），所以只能
    近似为"当前值 != dataclass 默认"。只告警阶段该近似只影响告警噪声，不影响
    正确性。
    """
    if not hasattr(config, name):
        return False
    value = getattr(config, name)
    for f in dataclasses.fields(config):
        if f.name != name:
            continue
        if f.default is not dataclasses.MISSING:
            return value != f.default
        if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            return value != f.default_factory()  # type: ignore[misc]
    # TransformerConfig 是 dataclass，一般走不到这里；非 dataclass 视为显式。
    return True


# ---------------------------------------------------------------------------
# Normalizer（①）—— 无损转换
# ---------------------------------------------------------------------------


@dataclass
class NormalizationReport:
    """归一结果：只记录无损转换，不改变任何语义。"""

    changes: list[dict] = field(
        default_factory=list
    )  # {field, old, new, reason}
    warnings: list[dict] = field(default_factory=list)  # {field, message}

    def to_json(self) -> dict:
        return {"changes": self.changes, "warnings": self.warnings}


def normalize_attention_config(
    config: Any, *, apply: bool = True
) -> NormalizationReport:
    """按 registry 声明做无损归一（方案 §2.1 ①）。

    只做两类转换：
    - 别名归一：旧名（``index_*``）出现在实例字典时映射到 ``dsa_index_*``
      （TransformerConfig 经 transform_rules 已改名，这里兜直连/外部 config）；
    - 类型强转：声明为 float/int 的字段喂进字符串数字（如 ``"160000.0"``，
      B5）时强转，失败仅告警（只告警阶段不 raise）。
    """
    report = NormalizationReport()

    # 别名归一
    for old, canonical in _ALIAS_TO_CANONICAL.items():
        if old in getattr(config, "__dict__", {}):
            old_value = config.__dict__[old]
            if old_value is None:
                continue
            cur = getattr(config, canonical, None)
            if cur is None or cur == old_value:
                if apply and hasattr(config, canonical):
                    setattr(config, canonical, old_value)
                report.changes.append(
                    {
                        "field": old,
                        "old": old_value,
                        "new": canonical,
                        "reason": f"alias -> {canonical} (registry)",
                    }
                )
            else:
                report.warnings.append(
                    {
                        "field": old,
                        "message": (
                            f"legacy alias '{old}'={old_value!r} conflicts "
                            f"with '{canonical}'={cur!r}; alias ignored"
                        ),
                    }
                )

    # 类型强转（B5：声明类型 vs 实际类型）
    for name, spec in ATTENTION_FIELD_REGISTRY.items():
        if spec.type not in ("int", "float") or not hasattr(config, name):
            continue
        value = getattr(config, name)
        if not isinstance(value, str):
            continue
        try:
            coerced: Any = float(value) if spec.type == "float" else int(value)
        except ValueError:
            report.warnings.append(
                {
                    "field": name,
                    "message": (
                        f"'{name}' declared {spec.type} but got "
                        f"non-numeric string {value!r}"
                    ),
                }
            )
            continue
        if apply:
            setattr(config, name, coerced)
        report.changes.append(
            {
                "field": name,
                "old": value,
                "new": coerced,
                "reason": "numeric string coerced (registry type contract)",
            }
        )

    return report


# ---------------------------------------------------------------------------
# Validator（②）—— 组合校验，只告警模式
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """一条校验结果。severity 在只告警模式（log_only）下整体降级为 warning。"""

    rule_id: str  # 如 "V-VAR-01"，对应方案 §一 问题编号
    severity: Literal["error", "warning"]
    message: str
    remediation: str


def _validate(
    config: Any, layer_plans: list[AttentionExecutionPlan]
) -> list[Finding]:
    """A/B 系列组合校验（方案 §一 A1-A7、B1/B3）。返回 findings 列表。"""
    out: list[Finding] = []
    variant = getattr(config, "experimental_attention_variant", None)
    is_dsv4 = variant == "dsv4_hybrid"
    families = {p.family for p in layer_plans}
    decoder_plans = [p for p in layer_plans if not p.is_mtp]
    mtp_plans = [p for p in layer_plans if p.is_mtp]

    # VAR-01（§一 A1）：dsv4_hybrid 无条件抢占 multi_latent_attention 标志
    if is_dsv4 and getattr(config, "multi_latent_attention", False):
        out.append(
            Finding(
                "V-VAR-01",
                "error",
                "'multi_latent_attention=true' is set but "
                "experimental_attention_variant='dsv4_hybrid' takes precedence; "
                "per-layer types come from csa_compress_ratios and the flag "
                "never reaches the spec chain.",
                "Remove 'multi_latent_attention' (dsv4_hybrid already implies "
                "MLA layers via csa_compress_ratios == -2), or drop "
                "experimental_attention_variant.",
            )
        )

    # VAR-02（§一 A2）：dsv4_hybrid 下 layer_types 被 csa_compress_ratios 覆盖
    if is_dsv4 and getattr(config, "layer_types", None) is not None:
        out.append(
            Finding(
                "V-VAR-02",
                "error",
                "'layer_types' is set but is decorative under "
                "experimental_attention_variant='dsv4_hybrid': every layer type "
                "is rewritten from csa_compress_ratios.",
                "Remove 'layer_types' or switch off the dsv4_hybrid variant.",
            )
        )

    # VHA-01（§一 A3）：use_vha_attention 在 gdn/kda/gemma4 下无人读取（self 换类为
    # SelfAttentionVHA；mla/dsv4 复用为 postmix/premix，是合法生效）
    _vha_unread = {"gdn", "kda", "gemma4"}
    if (
        getattr(config, "use_vha_attention", False)
        and families
        and families <= _vha_unread
    ):
        out.append(
            Finding(
                "V-VHA-01",
                "error",
                "'use_vha_attention=true' is not read by any resolved layer: "
                f"all layers are {sorted(families)} (self swaps in "
                "SelfAttentionVHA; mla/dsv4 reuse it as postmix/premix; "
                "gdn/kda/gemma4 never read it).",
                "Remove 'use_vha_attention' or use an attention family that "
                "supports it (self / mla / dsv4).",
            )
        )
    # VHA-02（§一 A3 细分）：vha_premix 仅 dsv4 层读
    if getattr(config, "use_vha_premix", False) and not (families & {"dsv4"}):
        out.append(
            Finding(
                "V-VHA-02",
                "error",
                "'use_vha_premix=true' only applies to dsv4 "
                "(CSA/HCA/window) layers; no such layer is resolved.",
                "Remove 'use_vha_premix' or use "
                "experimental_attention_variant='dsv4_hybrid'.",
            )
        )

    # VAR-03（§一 A4）：MTP 层类型只看 multi_latent_attention 标志
    if mtp_plans and decoder_plans and not is_dsv4:
        mtp_families = {p.family for p in mtp_plans}
        decoder_families = {p.family for p in decoder_plans}
        if mtp_families - decoder_families:
            out.append(
                Finding(
                    "V-VAR-03",
                    "error",
                    f"MTP layers resolve to {sorted(mtp_families)} (driven only "
                    "by multi_latent_attention) while decoder layers resolve "
                    f"to {sorted(decoder_families)} (layer_types / "
                    "attention_layer_type); the two disagree.",
                    "Set multi_latent_attention to match the decoder layer "
                    "family, or accept homogeneous MTP layers.",
                )
            )

    # VAR-04（§一 A5）：gemma4 与 dsv4_hybrid 组合，gemma4 判定在 dsv4 重写之后
    layer_types = getattr(config, "layer_types", None)
    if (
        is_dsv4
        and isinstance(layer_types, (list, tuple))
        and "gemma4" in layer_types
    ):
        out.append(
            Finding(
                "V-VAR-04",
                "error",
                "'gemma4' appears in layer_types but the gemma4 branch is "
                "unreachable under experimental_attention_variant='dsv4_hybrid'.",
                "Remove 'gemma4' from layer_types or drop the dsv4_hybrid "
                "variant.",
            )
        )

    # MLA-01（§一 A6）：MLA 不支持 GQA/MQA（num_key_value_heads 被钉死），报错晚
    if "mla" in families and not is_dsv4:
        nkv = getattr(config, "num_key_value_heads", None)
        nq = getattr(config, "num_attention_heads", None)
        if nkv is not None and nq is not None and nkv != nq:
            out.append(
                Finding(
                    "V-MLA-01",
                    "error",
                    f"MLA pins num_key_value_heads={nq} (got {nkv}); the "
                    "mismatch is only surfaced late (shape error at build "
                    "time).",
                    "Set num_key_value_heads == num_attention_heads for MLA "
                    "models.",
                )
            )

    # VAR-05（§一 A7）：hy_sparse 的 MLA 资格检查用重写之前的 attention_layer_type 形参
    if getattr(config, "enable_hy_sparse_attention", False) and is_dsv4:
        if not getattr(config, "multi_latent_attention", False) and not any(
            p.family == "mla" for p in decoder_plans
        ):
            out.append(
                Finding(
                    "V-VAR-05",
                    "error",
                    "enable_hy_sparse_attention requires the pre-rewrite "
                    "attention_layer_type/MLA flag to say MLA, but under "
                    "dsv4_hybrid the eligibility check runs before the "
                    "per-layer rewrite, so the -2 (MLA) layers cannot satisfy "
                    "it via csa_compress_ratios alone.",
                    "Set multi_latent_attention=true alongside dsv4_hybrid "
                    "(the flag is otherwise ignored for spec selection), or "
                    "disable hy_sparse.",
                )
            )

    # ROPE-01（§一 B1）：MLA 维度字段误配到 DSv4/CSA 层
    if is_dsv4 and any(p.family == "dsv4" for p in decoder_plans):
        for mla_field in ("qk_rope_head_dim", "qk_nope_head_dim"):
            if _explicitly_set(config, mla_field):
                out.append(
                    Finding(
                        "V-ROPE-01",
                        "error",
                        f"'{mla_field}' is an MLA-only field; DSv4/CSA layers "
                        "read 'qk_pos_emb_head_dim' (currently "
                        f"{getattr(config, 'qk_pos_emb_head_dim', None)!r}).",
                        "Set qk_pos_emb_head_dim for the DSv4/CSA layers "
                        "instead.",
                    )
                )
                break

    # ROPE-02（§一 B2）：dsv4 下 qk_pos_emb_head_dim None → RoPE 宽度静默为 0
    if is_dsv4 and getattr(config, "qk_pos_emb_head_dim", None) is None:
        out.append(
            Finding(
                "V-ROPE-02",
                "error",  # 静默改变模型行为（RoPE 被关）：按 severity 标准是
                # error；只告警模式下仍只打 W 行，观察期结束后随拦截模式生效
                "'qk_pos_emb_head_dim' is None on the dsv4_hybrid path: "
                "DSv4/CSA layers resolve RoPE width to 0 (RoPE silently "
                "disabled).",
                "Set qk_pos_emb_head_dim explicitly (0 only if RoPE is "
                "intended to be off).",
            )
        )

    # ROPE-03（§一 B3）：rotary_interleaved 在 DSv4/CSA eager 路径被硬编码忽略
    if is_dsv4 and getattr(config, "rotary_interleaved", False):
        out.append(
            Finding(
                "V-ROPE-03",
                "error",
                "'rotary_interleaved=true' is not implemented on the DSv4/CSA "
                "eager RoPE path (hardcoded non-interleaved layout); the "
                "config is silently ignored.",
                "Set rotary_interleaved=false, or wait for the DSv4 "
                "interleaved implementation (B_rope_config.md §3.3).",
            )
        )

    # ROPE-04（§一 B4）：dsa_indexer_rotary_interleaved 仅在 DSA indexer 存在时生效
    if _explicitly_set(config, "dsa_indexer_rotary_interleaved") and not any(
        "indexer" in p.notes for p in layer_plans
    ):
        out.append(
            Finding(
                "V-ROPE-04",
                "warning",
                "'dsa_indexer_rotary_interleaved' is set but no DSA indexer "
                "is built under the resolved plan.",
                "Remove the field or enable the DSA indexer "
                "(dsa_index_n_heads).",
            )
        )

    return out


# ---------------------------------------------------------------------------
# ③ AttentionExecutionPlan（数据契约，方案 §2.1.1）
# ---------------------------------------------------------------------------

# family 只表示"注意力计算结构"（self/mla/dsv4/gdn/kda/gemma4）。VHA、
# gated_attention、hy_sparse、SWA 是横跨家族的纵向能力：VHA 用
# qkv_layout=shared_kv 表达，其余进 capabilities / swa 列。
Family = Literal["self", "mla", "dsv4", "gdn", "kda", "gemma4", "cross"]
# 纵向能力（可叠加，不互斥）
Capability = Literal[
    "vha",  # self family：换整个类为 SelfAttentionVHA
    "vha_postmix",  # mla/dsv4：复用开关，只在输出头空间加低秩 postmix
    "vha_premix",  # dsv4：Q up-projection 换结构化 premix（需 vha_postmix 同开）
    "gated_attention",
    "hy_sparse",
]
Core = Literal[
    "dot_product",  # 标准 backend.core_attention()（eager/sdpa 在运行期协商）
    "linear",  # GDN/KDA 线性注意力
    "dsa",  # DSAttention + DSA indexer
    "csa",  # CompressedSparseAttention（core_detail 区分 hca/csa/window/mqa）
    "mqa_latent",  # MQALatentAttention（latent MQA 吸收路径）
]
QkvLayout = Literal["mha", "gqa", "latent", "shared_kv", "linear"]
PositionType = Literal["none", "rope", "yarn", "mrope"]
PositionLayout = Literal[
    "STANDARD",  # 标准 partial rope（rope_first 段序）
    "MLA_EAGER",
    "MLA_FUSED_PAIR",
    "MLA_FUSED_INPLACE_INTERLEAVED",
    "DSV4_CSA_EAGER",
    "DSA_INDEXER",
]
SegmentOrder = Literal["nope_first", "rope_first"]


@dataclass(frozen=True)
class RopeSpec:
    """position 列的完整展开（方案 §2.1.2：RoPE 信息必须完整展示）。"""

    type: PositionType
    base: float
    dim: int
    layout: PositionLayout
    segment_order: SegmentOrder
    # True/False = 实际生效值；"ignored" = 该路径硬编码、config 被无视（挂 W 行）
    interleave: Any
    yarn_params: dict = field(default_factory=dict)  # 与默认不同的 yarn 参数

    def render(self) -> str:
        text = (
            f"{self.type}(base={_fmt_num(self.base)}, dim={self.dim}, "
            f"layout={self.layout}, seg={self.segment_order}"
        )
        # B3：被路径硬编码忽略的 interleave 也必须显式展示（方案 §2.1.2）
        text += f", interleave={self.interleave}"
        for key, value in sorted(self.yarn_params.items()):
            text += f", {key}={_fmt_num(value)}"
        return text + ")"


def _fmt_num(value: Any) -> str:
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value)


@dataclass(frozen=True)
class AttentionExecutionPlan:
    """每层一条的执行计划（契约见总体方案 §2.1.1）。

    与方案的差异（登记于方案文档实现记录）：
    - 新增 ``index`` / ``layer_number`` / ``is_mtp`` / ``core_detail`` /
      ``notes``：快照与打印需要层定位与 core 细分（hca/csa/window/mqa）；
    - ``precision`` / ``recompute`` 首版为摘要字符串（P3 重构时结构化）；
    - 枚举新增 ``core=dot_product|linear``、``qkv_layout=linear``。
    """

    index: int  # 逻辑层下标（decoder 从 0，MTP 紧随其后）
    layer_number: int  # 传给 get_gpt_layer_local_spec 的物理层号
    is_mtp: bool
    family: Family
    core: Core
    core_detail: str | None = None  # csa: hca|csa|window|mqa_full_causal
    swa: bool = False
    window_size: int | None = None
    qkv_layout: QkvLayout = "mha"
    position: RopeSpec | None = None
    precision: str = ""
    recompute: str = ""
    notes: tuple[str, ...] = ()
    # 本层实际生效的维度字段（镜像各家族 __init__ 的读取点，C1-C3 的观测面）：
    # MLA 读 q_lora_rank/qk_rope_head_dim 等，dsv4 -2 层读 hybrid_mla_*，
    # CSA/HCA/window 层读 v_head_dim/qk_pos_emb_head_dim，VHA 读
    # vha_q_lora_rank，GDN/KDA 读 linear_*。SWA 层记录 swa_* override 后的
    # 取值。进 JSON（打印/落盘/快照），摘要表不展示以免过宽。
    dims: dict = field(default_factory=dict)
    # 本层 q/k norm 实际类型（C4 的观测面）："none" | "LayerNorm" |
    # "RMSNorm" | "L2" | "triton-rms"（qk_norm_fusion 融合核）
    qk_norm: str = "none"
    # 纵向能力（横跨家族、可叠加）：vha / gated_attention / hy_sparse。
    # 与 family（计算结构）正交：VHA 是 self 的投影变体（qkv_layout=
    # shared_kv），门控是 self/mla/dsv4 都能挂的 gate_proj。
    capabilities: tuple[str, ...] = ()

    def to_json(self) -> dict:
        """打印、落盘、快照三者共用的序列化（方案 §2.1.2）。"""
        d = dataclasses.asdict(self)
        d["position"] = (
            dataclasses.asdict(self.position)
            if self.position is not None
            else None
        )
        return d


@dataclass
class AttentionPlanBundle:
    """整个 config 的 resolve 结果。"""

    variant: str | None
    layers: list[AttentionExecutionPlan] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    normalization: NormalizationReport = field(
        default_factory=NormalizationReport
    )
    log_only: bool = True

    # -- 聚合统计（§2.1.2 头部） --
    def family_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for p in self.layers:
            counts[p.family] = counts.get(p.family, 0) + 1
        return counts

    def rope_groups(self) -> int:
        """RoPE base/type 不一致的层组数。"""
        return len(
            {
                (p.position.type, p.position.base)
                for p in self.layers
                if p.position is not None
            }
        )

    def to_json(self) -> dict:
        return {
            "variant": self.variant,
            "log_only": self.log_only,
            "family_counts": self.family_counts(),
            "rope_groups": self.rope_groups(),
            "layers": [p.to_json() for p in self.layers],
            "findings": [dataclasses.asdict(f) for f in self.findings],
            "normalization": {
                "changes": self.normalization.changes,
                "warnings": self.normalization.warnings,
            },
        }

    def to_json_str(self) -> str:
        return json.dumps(self.to_json(), indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# ④ 打印（rank0 摘要表，§2.1.2）
# ---------------------------------------------------------------------------


def format_plan_lines(bundle: AttentionPlanBundle) -> list[str]:
    """渲染 ``[ATTN-PLAN]`` 摘要表（每行已带前缀）。"""
    lines: list[str] = []
    n_warn = len(bundle.findings)
    lines.append(
        f"[ATTN-PLAN] {len(bundle.layers)} layers resolved "
        f"(variant={bundle.variant or 'none'}, "
        f"rope_groups={bundle.rope_groups()}, "
        f"warnings={n_warn}{' [warn-only]' if bundle.log_only else ''})"
    )
    lines.append(
        "[ATTN-PLAN]   layer  family  core         swa        qkv_layout  norm  "
        "caps  position"
    )
    for p in bundle.layers:
        label = f"{p.index}" + ("m" if p.is_mtp else "")
        core = p.core + (f":{p.core_detail}" if p.core_detail else "")
        swa = f"swa({p.window_size})" if p.swa else "-"
        caps = "+".join(p.capabilities) if p.capabilities else "-"
        pos = p.position.render() if p.position is not None else "none"
        lines.append(
            f"[ATTN-PLAN]   {label:>5}  {p.family:<6}  {core:<12} {swa:<10} "
            f"{p.qkv_layout:<10} {p.qk_norm:<6} {caps:<6} {pos}"
        )
        for note in p.notes:
            lines.append(f"[ATTN-PLAN]          note: {note}")
    for f in bundle.findings:
        sev = "W" if bundle.log_only else f.severity
        lines.append(f"[ATTN-PLAN] {sev}  {f.rule_id}: {f.message}")
        lines.append(f"[ATTN-PLAN]        fix: {f.remediation}")
    return lines


def _is_rank0() -> bool:
    try:
        import paddle

        if not paddle.distributed.is_initialized():
            return True
        return paddle.distributed.get_rank() == 0
    except Exception:  # pragma: no cover - 无 paddle 分布式环境
        return True


def _dump_plan_json(config: Any, bundle: AttentionPlanBundle) -> None:
    """JSON 落盘：复用 config_logger（config_logger_dir 未设置则跳过）。"""
    try:
        from paddlefleet.config_logger import (
            has_config_logger_enabled,
            log_config_to_disk,
        )

        if not has_config_logger_enabled(config):
            return
        if not _is_rank0():
            return
        log_config_to_disk(
            config,
            {"attention_plan": bundle.to_json()},
            prefix="attention_plan",
        )
    except Exception:
        # 落盘失败不阻断构建（sidecar 约束）
        pass


# ---------------------------------------------------------------------------
# ③ Resolver —— 复刻 gpt_layer_specs 的现状优先级（逐字节对齐，仅观测）
# ---------------------------------------------------------------------------

_FAMILY_BY_LAYER_TYPE = {
    "full_attention": "self",
    "self_attention": "self",
    "linear_attention": "gdn",
    "gated_delta_net": "gdn",
    "kimi_delta_attention": "kda",
    "multi_latent_attention": "mla",
    "dsv4_hybrid_attention": "dsv4",
    "gemma4": "gemma4",
}


def _ratio_kind(ratio: int) -> tuple[str, str | None]:
    """csa_compress_ratios 取值 → (family, core_detail)。

    来源：gpt_layer_specs._get_dsv4_hybrid_attention_layer_type 与
    hf_export.py 的 ratio 约定注释（-2 → MLA；-1 → CSA full-causal MQA；
    0 → window；128 → HCA；[2,128) → CSA）。
    """
    if ratio == -2:
        return "mla", None
    if ratio == -1:
        return "dsv4", "mqa_full_causal"
    if ratio == 0:
        return "dsv4", "window"
    if ratio == 128:
        return "dsv4", "hca"
    if 2 <= ratio < 128:
        return "dsv4", "csa"
    raise ValueError(f"unknown csa_compress_ratio: {ratio!r}")


def _is_swa_layer(config: Any, logical_index: int) -> bool:
    """标准路径 SWA 判定（attention.py:250-273 的镜像，逻辑层号）。"""
    sliding_window = getattr(config, "sliding_window", None)
    if not sliding_window:
        return False
    return is_layer_window_attention(
        sliding_window,
        getattr(config, "window_attn_skip_freq", None),
        logical_index,
    )


_YARN_DEFAULTS = (
    ("rotary_scaling_factor", 40),
    ("original_max_position_embeddings", 4096),
    ("mscale", 1.0),
    ("mscale_all_dim", 0.0),
)


def _yarn_extras(config: Any) -> dict:
    return {
        key: getattr(config, key, default)
        for key, default in _YARN_DEFAULTS
        if getattr(config, key, default) != default
    }


def _standard_rope_spec(config: Any, is_swa: bool) -> RopeSpec:
    """标准 / VHA / gemma4 家族的 position。"""
    pet = getattr(config, "position_embedding_type", "learned_absolute")
    if pet in ("learned_absolute", "none"):
        return RopeSpec(
            type="none",
            base=0.0,
            dim=0,
            layout="STANDARD",
            segment_order="rope_first",
            interleave=False,
        )
    pos_type = {"rope": "rope", "yarn": "yarn", "mrope": "mrope"}.get(
        pet, "rope"
    )
    base = getattr(config, "swa_rope_theta", None) if is_swa else None
    base = base or getattr(config, "rope_theta", 10000.0)
    head_dim = getattr(config, "swa_head_dim", None) if is_swa else None
    head_dim = head_dim or getattr(config, "head_dim", 0)
    percent = getattr(config, "rotary_percent", 1.0)
    dim = (
        int(head_dim * percent)
        if isinstance(percent, (int, float))
        else head_dim
    )
    return RopeSpec(
        type=pos_type,
        base=float(base),
        dim=int(dim),
        layout="STANDARD",
        segment_order="rope_first",
        interleave=bool(getattr(config, "rotary_interleaved", False)),
        yarn_params=_yarn_extras(config) if pos_type == "yarn" else {},
    )


def _mla_rope_spec(config: Any, is_dsv4_hybrid: bool, is_swa: bool) -> RopeSpec:
    """MLA 家族（含 dsv4 -2 层）的 position（multi_latent_attention.py:476-505）。"""
    pos_type = (
        "yarn" if getattr(config, "rope_type", "yarn") == "yarn" else "rope"
    )
    base = getattr(config, "rope_theta", 10000.0)
    if is_dsv4_hybrid:
        dim = getattr(config, "hybrid_mla_qk_rope_head_dim", None) or getattr(
            config, "qk_rope_head_dim", 0
        )
    elif is_swa and getattr(config, "swa_qk_rope_head_dim", None):
        dim = config.swa_qk_rope_head_dim
    else:
        dim = getattr(config, "qk_rope_head_dim", 0)
    return RopeSpec(
        type=pos_type,
        base=float(base),
        dim=int(dim),
        layout="MLA_EAGER",
        segment_order="nope_first",
        interleave=bool(getattr(config, "rotary_interleaved", False)),
        yarn_params=_yarn_extras(config) if pos_type == "yarn" else {},
    )


def _dsv4_rope_spec(config: Any, ratio: int) -> RopeSpec:
    """DSv4/CSA/HCA/window 层的 position（dsv4_hybrid_attention.py:720-771 镜像）。"""
    if ratio == 128:
        per_type = getattr(config, "hca_rope_type", None)
    elif 2 <= ratio < 128:
        per_type = getattr(config, "csa_rope_type", None)
    else:
        per_type = None
    pos_type = per_type or ("yarn" if ratio > 1 else "rope")
    base = getattr(config, "rope_theta", 10000.0)
    if ratio > 1:
        base = getattr(config, "csa_compress_rotary_base", base)
        if isinstance(base, str):
            base = float(base)
    dim = getattr(config, "qk_pos_emb_head_dim", None) or 0
    return RopeSpec(
        type=pos_type,
        base=float(base),
        dim=int(dim),
        layout="DSV4_CSA_EAGER",
        segment_order="nope_first",
        interleave="ignored",  # B3：DSv4/CSA eager 路径硬编码非交错
        yarn_params=_yarn_extras(config) if pos_type == "yarn" else {},
    )


def _precision_summary(config: Any) -> str:
    parts = []
    for key in ("params_dtype", "bf16", "fp8"):
        value = getattr(config, key, None)
        if value not in (None, False):
            parts.append(f"{key}={value}")
    return ",".join(parts)


def _recompute_summary(config: Any) -> str:
    modules = getattr(config, "recompute_modules", None)
    return str(modules) if modules else ""


def _qk_norm_of(config: Any, family: str, is_dsv4_hybrid: bool) -> str:
    """本层 q/k norm 类型（gpt_layer_specs 的 q_norm/k_norm 选路镜像）。

    - self/vha：qk_l2_norm → L2；use_qk_norm → RMSNorm/LayerNorm，
      RMSNorm + qk_norm_fusion（head_dim=128 限制在 kernel 内）→ triton-rms
    - mla（含 dsv4 -2 层）：use_qk_norm → q_a/kv_a LayerNorm（RMSNorm 化）
    - dsv4 CSA/HCA/window：qk_layernorm（缺省 True）→ RMSNorm 化
    - gemma4：恒开标准 norm；gdn/kda：无 qk norm（仅 out_norm）
    """
    if family in ("gdn", "kda"):
        return "none"
    rms = getattr(config, "normalization", None) == "RMSNorm"
    if family == "gemma4":
        return "RMSNorm" if rms else "LayerNorm"
    if family == "dsv4":
        if not getattr(config, "qk_layernorm", True):
            return "none"
        return "RMSNorm" if rms else "LayerNorm"
    if family == "mla":
        if not getattr(config, "use_qk_norm", False):
            return "none"
        return "RMSNorm" if rms else "LayerNorm"
    # self / vha
    if getattr(config, "qk_l2_norm", False):
        return "L2"
    if not getattr(config, "use_qk_norm", False):
        return "none"
    if rms and getattr(config, "qk_norm_fusion", False):
        return "triton-rms"
    return "RMSNorm" if rms else "LayerNorm"


def _dims_mla(config: Any, is_dsv4_hybrid: bool, is_swa: bool) -> dict:
    """MLA 家族维度（multi_latent_attention.py:334-398 镜像）。"""
    if is_dsv4_hybrid:
        dims = {
            "q_lora_rank": config.hybrid_mla_q_lora_rank,
            "kv_lora_rank": config.hybrid_mla_kv_lora_rank,
            "qk_nope_head_dim": config.hybrid_mla_qk_nope_head_dim,
            "qk_rope_head_dim": config.hybrid_mla_qk_rope_head_dim,
            "v_head_dim": config.hybrid_mla_v_head_dim,
            "num_attention_heads": config.hybrid_mla_num_attention_heads,
            # num_key_value_heads 被钉死为 num_attention_heads（A6）
            "num_key_value_heads": config.hybrid_mla_num_attention_heads,
        }
        return dims
    dims = {
        "q_lora_rank": config.q_lora_rank,
        "kv_lora_rank": config.kv_lora_rank,
        "qk_nope_head_dim": config.qk_nope_head_dim,
        "qk_rope_head_dim": config.qk_rope_head_dim,
        "v_head_dim": config.v_head_dim,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_attention_heads,  # A6：钉死
    }
    if is_swa:  # D2：SWA 层整体换一套维度
        if getattr(config, "swa_qk_nope_head_dim", None) is not None:
            dims["qk_nope_head_dim"] = config.swa_qk_nope_head_dim
        if getattr(config, "swa_qk_rope_head_dim", None) is not None:
            dims["qk_rope_head_dim"] = config.swa_qk_rope_head_dim
    return dims


def _dims_dsv4(config: Any, ratio: int) -> dict:
    """DSv4/CSA/HCA/window 层维度（dsv4_hybrid_attention.py:687-708 镜像）。"""
    return {
        "num_attention_heads": config.num_attention_heads,  # n_local_heads
        "v_head_dim": config.v_head_dim,
        "q_head_dim": config.v_head_dim,  # q_head_dim = v_head_dim（C3）
        "qk_pos_emb_head_dim": getattr(config, "qk_pos_emb_head_dim", None)
        or 0,
        "compress_ratio": ratio,
        "window_size": config.csa_window_size,
    }


def _dims_standard(config: Any, is_swa: bool) -> dict:
    """标准 self/gemma4 家族维度（attention.py:275-303 镜像）。"""
    if is_swa:  # SWA 层换用 swa_* override（attention.py:275-280）
        return {
            "num_attention_heads": config.swa_num_attention_heads,
            "num_key_value_heads": config.swa_num_key_value_heads,
            "head_dim": config.swa_head_dim,
            "v_head_dim": config.swa_v_head_dim,
        }
    v_head_dim = getattr(config, "v_head_dim", None)
    return {
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "head_dim": getattr(config, "head_dim", None),
        "v_head_dim": v_head_dim
        if isinstance(v_head_dim, int)
        else getattr(config, "head_dim", None),
    }


def _dims_vha(config: Any, is_swa: bool) -> dict:
    """VHA 维度（attention.py:1197-1213 镜像；C2）。"""
    nq = getattr(config, "num_attention_heads", None)
    dims = {
        "vha_shared_kv": getattr(config, "vha_shared_kv", False),
        "vha_q_lora_rank": (
            config.swa_vha_q_lora_rank
            if is_swa and getattr(config, "swa_vha_q_lora_rank", None)
            else getattr(config, "vha_q_lora_rank", None)
        ),
        # 缺省静默推导 num_attention_heads // 4（C2）
        "vha_postmix_rank": (
            config.swa_vha_postmix_rank
            if is_swa and getattr(config, "swa_vha_postmix_rank", None)
            else getattr(config, "vha_postmix_rank", None)
        )
        or (nq // 4 if isinstance(nq, int) else None),
    }
    dims.update(_dims_standard(config, is_swa))
    return dims


def _dims_linear(config: Any) -> dict:
    """GDN/KDA 维度（gpt_layer_specs.py:284-291 的读取镜像）。"""
    return {
        "linear_conv_kernel_dim": getattr(config, "linear_conv_kernel_dim", 4),
        "linear_key_head_dim": getattr(config, "linear_key_head_dim", 128),
        "linear_value_head_dim": getattr(config, "linear_value_head_dim", 128),
        "linear_num_key_heads": getattr(config, "linear_num_key_heads", 16),
        "linear_num_value_heads": getattr(config, "linear_num_value_heads", 32),
    }


def _window_size_of(config: Any) -> Any:
    ws = getattr(config, "sliding_window", None)
    if isinstance(ws, (list, tuple)):
        ws = ws[0]
    return ws


def _resolve_dsv4_layer(
    config: Any, index: int, layer_number: int, is_mtp: bool, ratio: int
) -> AttentionExecutionPlan:
    family, core_detail = _ratio_kind(ratio)
    notes: list[str] = []
    capabilities: list[str] = []
    if getattr(config, "use_vha_attention", False):
        # MLA(-2) 层与 CSA/HCA/window 层都复用该开关做 postmix（:667/:913）
        capabilities.append("vha_postmix")
    if (
        family == "dsv4"
        and getattr(config, "use_vha_attention", False)
        and getattr(config, "use_vha_premix", False)
    ):
        capabilities.append("vha_premix")
    if family == "mla":
        mode = getattr(config, "hybrid_mla_attention", "mha")
        if mode == "mqa_dsa":
            core = "mqa_latent"
            notes.append("indexer: DSA (hybrid_mla_attention=mqa_dsa)")
        elif mode == "mqa_full_causal":
            core = "mqa_latent"
            notes.append("no indexer (hybrid_mla_attention=mqa_full_causal)")
        else:
            core = "dot_product"
            notes.append("dense MHA on -2 layer (hybrid_mla_attention=mha)")
        qkv_layout = "latent" if core == "mqa_latent" else "mha"
        position = _mla_rope_spec(config, is_dsv4_hybrid=True, is_swa=False)
        dims = _dims_mla(config, is_dsv4_hybrid=True, is_swa=False)
        swa = False
        window_size = None
    else:
        core = "csa"
        qkv_layout = "mha"
        position = _dsv4_rope_spec(config, ratio)
        dims = _dims_dsv4(config, ratio)
        if core_detail == "window":
            swa = True
            window_size = getattr(config, "csa_window_size", None)
        else:
            swa = False
            window_size = None
        if core_detail == "csa" and not getattr(
            config, "csa_dense_mode", False
        ):
            notes.append("indexer: CSA")
    return AttentionExecutionPlan(
        index=index,
        layer_number=layer_number,
        is_mtp=is_mtp,
        family=family,  # type: ignore[arg-type]
        core=core,  # type: ignore[arg-type]
        core_detail=core_detail,
        swa=swa,
        window_size=window_size,
        qkv_layout=qkv_layout,  # type: ignore[arg-type]
        position=position,
        precision=_precision_summary(config),
        recompute=_recompute_summary(config),
        notes=tuple(notes),
        dims=dims,
        qk_norm=_qk_norm_of(config, family, is_dsv4_hybrid=True),
        capabilities=(
            *capabilities,
            *(
                ("gated_attention",)
                if getattr(config, "gated_attention", False)
                else ()
            ),
        ),
    )


def _resolve_standard_layer(
    config: Any,
    index: int,
    layer_number: int,
    is_mtp: bool,
    layer_type: str,
) -> AttentionExecutionPlan:
    family = _FAMILY_BY_LAYER_TYPE[layer_type]
    notes: list[str] = []
    capabilities: list[str] = []
    if family == "self" and getattr(config, "use_vha_attention", False):
        # VHA 不是独立 family：self 的投影变体，qkv_layout 表达布局，
        # capabilities 记录能力开关（纵向，横跨家族）。
        capabilities.append("vha")
    if family == "mla" and getattr(config, "use_vha_attention", False):
        # MLA/MQA 不换类，复用开关只加输出头空间 postmix（:667）
        capabilities.append("vha_postmix")
    if getattr(config, "gated_attention", False) and family in (
        "self",
        "mla",
        "dsv4",
    ):
        capabilities.append("gated_attention")
    if family in ("gdn", "kda"):
        core = "linear"
        qkv_layout = "linear"
    else:
        core = "dot_product"
        if "vha" in capabilities:
            qkv_layout = "shared_kv"
        else:
            nkv = getattr(config, "num_key_value_heads", None)
            nq = getattr(config, "num_attention_heads", None)
            qkv_layout = (
                "gqa"
                if nkv is not None and nq is not None and nkv != nq
                else "mha"
            )
    if "vha" in capabilities:
        is_swa = _is_swa_layer(config, index)
        dims = _dims_vha(config, is_swa)
    elif family in ("gdn", "kda"):
        dims = _dims_linear(config)
    elif family == "mla":
        dims = _dims_mla(
            config,
            is_dsv4_hybrid=False,
            is_swa=_is_swa_layer(config, index),
        )
    else:
        dims = _dims_standard(config, _is_swa_layer(config, index))
    if family == "mla":
        if getattr(config, "enable_hy_sparse_attention", False):
            core = "mqa_latent"
            qkv_layout = "latent"
            capabilities.append("hy_sparse")
            notes.append("hy_sparse swaps the class to MQASelfAttention")
        elif getattr(config, "dsa_index_n_heads", None) is not None:
            core = "dsa"
            notes.append("indexer: DSA")
            qkv_layout = "latent"
        else:
            qkv_layout = "latent"
        is_swa = _is_swa_layer(config, index)
        position = _mla_rope_spec(config, is_dsv4_hybrid=False, is_swa=is_swa)
        swa = is_swa
        window_size = _window_size_of(config) if is_swa else None
    else:
        is_swa = _is_swa_layer(config, index)
        position = _standard_rope_spec(config, is_swa)
        swa = is_swa
        window_size = _window_size_of(config) if is_swa else None
    return AttentionExecutionPlan(
        index=index,
        layer_number=layer_number,
        is_mtp=is_mtp,
        family=family,  # type: ignore[arg-type]
        core=core,  # type: ignore[arg-type]
        swa=swa,
        window_size=window_size,
        qkv_layout=qkv_layout,  # type: ignore[arg-type]
        position=position,
        precision=_precision_summary(config),
        recompute=_recompute_summary(config),
        notes=tuple(notes),
        dims=dims,
        qk_norm=_qk_norm_of(config, family, is_dsv4_hybrid=False),
        capabilities=tuple(capabilities),
    )


def _effective_mtp_layers(config: Any) -> int:
    n = getattr(config, "num_nextn_predict_layers", 0) or 0
    if not isinstance(n, int) or isinstance(n, bool):
        return 0
    return n


def resolve_attention_plan(
    config: Any, *, log_only: bool = True
) -> AttentionPlanBundle:
    """Resolve：逐层产出 AttentionExecutionPlan（方案 §2.1 ③）。

    决策树复刻 ``gpt_layer_specs.get_gpt_layer_local_spec`` 的现状优先级：
    dsv4_hybrid（csa_compress_ratios 逐层）→ multi_latent_attention flag →
    attention_layer_type（layer_types 或缺省）。第一个版本必须与现状逐字节
    对齐（快照基线），之后才允许收敛。
    """
    variant = getattr(config, "experimental_attention_variant", None)
    is_dsv4 = variant == "dsv4_hybrid"
    num_layers = getattr(config, "num_hidden_layers", 1)
    head_offset = getattr(config, "num_empty_layers_add_in_head", 0) or 0

    layers: list[AttentionExecutionPlan] = []
    if is_dsv4:
        ratios = getattr(config, "csa_compress_ratios", None) or []
        for i in range(num_layers):
            ratio = ratios[i] if i < len(ratios) else None
            if ratio is None:
                # 现状：_get_dsv4_hybrid_attention_layer_type 在此 raise；
                # sidecar 只跳过（构建期会报，不重复）。
                continue
            layers.append(
                _resolve_dsv4_layer(
                    config, i, i + head_offset, False, int(ratio)
                )
            )
        for m in range(_effective_mtp_layers(config)):
            logical = num_layers + m
            ratio = ratios[logical] if logical < len(ratios) else None
            if ratio is None:
                continue
            layers.append(
                _resolve_dsv4_layer(config, logical, m, True, int(ratio))
            )
    else:
        layer_types = getattr(config, "layer_types", None)
        if layer_types is None:
            fallback = (
                "multi_latent_attention"
                if getattr(config, "multi_latent_attention", False)
                else "self_attention"
            )
            layer_types = [fallback] * num_layers
        for i, layer_type in enumerate(layer_types):
            # get_attention_spec 里的同义映射（gpt_layer_specs.py:200-203）
            layer_type = {
                "full_attention": "self_attention",
                "linear_attention": "gated_delta_net",
            }.get(layer_type, layer_type)
            if layer_type not in _FAMILY_BY_LAYER_TYPE:
                continue  # 构建期会 raise；sidecar 不重复
            layers.append(
                _resolve_standard_layer(
                    config, i, i + head_offset, False, layer_type
                )
            )
        for m in range(_effective_mtp_layers(config)):
            # A4 现状：MTP 只看 multi_latent_attention 标志（gpt_layer_specs:877）
            mtp_type = (
                "multi_latent_attention"
                if getattr(config, "multi_latent_attention", False)
                else "self_attention"
            )
            layers.append(
                _resolve_standard_layer(
                    config, num_layers + m, m, True, mtp_type
                )
            )

    bundle = AttentionPlanBundle(
        variant=variant, layers=layers, log_only=log_only
    )
    bundle.findings = _validate(config, layers)
    return bundle


# ---------------------------------------------------------------------------
# 管道入口（① → ② → ③ → ④）
# ---------------------------------------------------------------------------


def run_attention_plan(
    config: Any,
    *,
    log_only: bool = True,
    print_plan: bool = True,
) -> AttentionPlanBundle:
    """执行 ①-④ 全管道并返回 Plan（sidecar：不参与构建决策）。

    只告警模式（``log_only=True``，默认）：Validator 的 error 也只作为
    ``[ATTN-PLAN]`` 告警行输出，不 raise——观察一个版本周期后再转 hard
    error（方案 §三 原则 3）。``log_only=False`` 时 error 级 finding 直接
    抛出（含迁移指引文案）。
    """
    # ① Normalizer（无损转换，直接作用于 config）
    report = normalize_attention_config(config)

    # ③ Resolver（② Validator 在 resolve 内部跑，基于 resolve 后的层结构）
    bundle = resolve_attention_plan(config, log_only=log_only)
    bundle.normalization = report

    # ④ 打印 + 落盘
    if print_plan and _is_rank0():
        for line in format_plan_lines(bundle):
            print(line)
    _dump_plan_json(config, bundle)

    if not log_only:
        errors = [f for f in bundle.findings if f.severity == "error"]
        if errors:
            lines = [
                "Attention config validation failed "
                f"({len(errors)} error(s), warn-only mode off):"
            ]
            for f in errors:
                lines.append(f"  [{f.rule_id}] {f.message}")
                lines.append(f"    fix: {f.remediation}")
            raise ValueError("\n".join(lines))
    return bundle
