#!/usr/bin/env python3
"""
并行配置合法性校验。

目标：
1. 给出比 ParallelConfig.validate() 更细的错误原因
2. 把 TP/PP/DP/EP/SP 的联动规则沉淀成统一入口
3. 支持构造固定样例，作为 pdcostmodel 的回归检查集
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from ..config import ModelConfig, ParallelConfig


_VALID_SHARDING_STAGES = {"none", "", "stage1", "stage2", "stage3"}


@dataclass
class ParallelValidationIssue:
    """单条校验结果。"""

    code: str
    message: str


@dataclass
class ParallelValidationResult:
    """并行配置校验结果。"""

    errors: List[ParallelValidationIssue] = field(default_factory=list)
    warnings: List[ParallelValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def add_error(self, code: str, message: str) -> None:
        self.errors.append(ParallelValidationIssue(code=code, message=message))

    def add_warning(self, code: str, message: str) -> None:
        self.warnings.append(ParallelValidationIssue(code=code, message=message))

    def summary_lines(self) -> List[str]:
        status = "valid" if self.is_valid else "invalid"
        lines = [f"status: {status}"]
        if self.errors:
            lines.append("errors:")
            lines.extend(f"  - [{item.code}] {item.message}" for item in self.errors)
        if self.warnings:
            lines.append("warnings:")
            lines.extend(f"  - [{item.code}] {item.message}" for item in self.warnings)
        return lines


def _normalize_sharding(sharding: str) -> str:
    return str(sharding or "none").strip().lower()


def _resolved_sharding_degree(parallel: ParallelConfig) -> int:
    if _normalize_sharding(parallel.sharding) == "none":
        return 1
    if int(parallel.sharding_degree) > 0:
        return int(parallel.sharding_degree)
    return int(parallel.dp)


def _physical_world_size(parallel: ParallelConfig) -> int:
    return (
        max(1, int(parallel.tp))
        * max(1, int(parallel.pp))
                * max(1, int(parallel.dp))
    )


def _resolved_expert_tensor_parallel_size(
    parallel: ParallelConfig,
    expert_tensor_parallel_size: Optional[int],
) -> int:
    if expert_tensor_parallel_size is None:
        return max(1, int(parallel.tp))
    return max(1, int(expert_tensor_parallel_size))


def _resolved_stage_layer_counts(
    parallel: ParallelConfig,
    stage_layer_counts: Optional[Sequence[int]],
) -> List[int]:
    if stage_layer_counts is not None:
        return [int(item) for item in stage_layer_counts]
    return [int(item) for item in (getattr(parallel, "stage_layer_counts", []) or [])]


def validate_parallel_config(
    parallel: ParallelConfig,
    total_gpus: int,
    model: Optional[ModelConfig] = None,
    require_sp_with_tp_pp: bool = True,
    num_empty_layers_add_in_head: int = 0,
    num_empty_layers_add_in_tail: int = 0,
    expert_tensor_parallel_size: Optional[int] = None,
    moe_router_num_groups: Optional[int] = None,
    stage_layer_counts: Optional[Sequence[int]] = None,
    allow_uneven_pp: bool = True,
    hybrid_parallel_topo_order: str = "",
) -> ParallelValidationResult:
    """
    严格校验并行配置是否合法。

    当前规则分三层：
    1. 纯拓扑规则：正整数、dense/expert world-size、sharding/vpp/sp 联动
    2. 模型整除规则：layer/head/hidden/expert 与 TP/PP/EP 的整除关系
    3. 不均匀 PP 规则：允许 stage_layer_counts 或默认不均匀切分
    """

    result = ParallelValidationResult()

    positive_fields = {
        "tp": int(parallel.tp),
        "pp": int(parallel.pp),
        "dp": int(parallel.dp),
        "ep": int(parallel.ep),
        "vpp": int(parallel.vpp),
    }
    for field_name, value in positive_fields.items():
        if value < 1:
            result.add_error(
                f"{field_name}_lt_1",
                f"{field_name} 必须 >= 1，当前为 {value}",
            )

    if total_gpus < 1:
        result.add_error(
            "total_gpus_lt_1",
            f"total_gpus 必须 >= 1，当前为 {total_gpus}",
        )

    resolved_stage_layer_counts = _resolved_stage_layer_counts(
        parallel, stage_layer_counts
    )

    if num_empty_layers_add_in_head < 0 or num_empty_layers_add_in_tail < 0:
        result.add_error(
            "negative_empty_layers",
            "num_empty_layers_add_in_head / tail 不能为负数",
        )

    sharding = _normalize_sharding(parallel.sharding)
    if sharding not in _VALID_SHARDING_STAGES:
        result.add_error(
            "invalid_sharding_stage",
            f"sharding 只支持 none/stage1/stage2/stage3，当前为 {parallel.sharding}",
        )

    if sharding == "none" and int(parallel.sharding_degree) not in (-1, 1):
        result.add_error(
            "sharding_degree_with_none",
            "sharding='none' 时，sharding_degree 只能是 -1 或 1",
        )

    if sharding != "none" and int(parallel.sharding_degree) == 0:
        result.add_error(
            "sharding_degree_zero",
            "开启 sharding 时，sharding_degree 不能为 0",
        )

    if int(parallel.vpp) > 1 and int(parallel.pp) <= 1:
        result.add_error(
            "vpp_requires_pp",
            "vpp>1 只有在 pp>1 时才有意义",
        )

    if resolved_stage_layer_counts:
        if int(parallel.vpp) > 1:
            result.add_error(
                "custom_stage_layout_with_vpp",
                "自定义 stage_layer_counts 暂不支持与 vpp>1 同时使用",
            )
        if num_empty_layers_add_in_head > 0 or num_empty_layers_add_in_tail > 0:
            result.add_error(
                "custom_stage_layout_with_empty_layers",
                "自定义 stage_layer_counts 暂不支持与 num_empty_layers_add_in_head/tail 同时使用",
            )

    dense_group_size = (
        max(1, int(parallel.tp))
        * max(1, int(parallel.pp))
            )
    dense_dp = None
    if total_gpus % dense_group_size != 0:
        result.add_error(
            "dense_world_size_not_divisible",
            (
                f"total_gpus={total_gpus} 不能整除 dense group size "
                f"tp*pp={dense_group_size}"
            ),
        )
    else:
        dense_dp = total_gpus // dense_group_size
        if int(parallel.dp) != dense_dp:
            result.add_error(
                "dp_mismatch_dense_world_size",
                (
                    f"dp 与 dense 主干 world size 不匹配：期望 dp={dense_dp}，"
                    f"当前为 {parallel.dp}"
                ),
            )

    expert_tp = _resolved_expert_tensor_parallel_size(
        parallel, expert_tensor_parallel_size
    )
    if expert_tensor_parallel_size is not None and int(expert_tensor_parallel_size) < 1:
        result.add_error(
            "expert_tp_lt_1",
            f"expert_tensor_parallel_size 必须 >= 1，当前为 {expert_tensor_parallel_size}",
        )

    if int(parallel.ep) > 1:
        expert_group_size = (
            expert_tp
            * max(1, int(parallel.ep))
            * max(1, int(parallel.pp))
        )
        if total_gpus % expert_group_size != 0:
            result.add_error(
                "expert_world_size_not_divisible",
                (
                    f"total_gpus={total_gpus} 不能整除 expert group size "
                    f"etp*ep*pp={expert_group_size}"
                ),
            )
        elif dense_dp is not None:
            expert_dp = total_gpus // expert_group_size
            topo_order = str(hybrid_parallel_topo_order or "").strip().lower()
            if (
                int(parallel.pp) > 1
                and topo_order
                and not topo_order.endswith("pp")
                and expert_dp != dense_dp
            ):
                result.add_error(
                    "expert_dp_mismatch_dense_dp",
                    (
                        f"当前拓扑顺序下 expert_dp={expert_dp} 必须等于 dense dp={dense_dp}"
                    ),
                )

    if model is None:
        if require_sp_with_tp_pp and int(parallel.tp) > 1 and int(parallel.pp) > 1 and not bool(parallel.sp):
            result.add_error(
                "sp_required_for_tp_pp",
                "当 tp>1 且 pp>1 同时开启时，必须开启 sp",
            )
        elif int(parallel.tp) > 1 and int(parallel.pp) == 1 and not bool(parallel.sp):
            result.add_warning(
                "tp_without_sp",
                "当前配置开启了 tp 但未开启 sp；这在部分框架配置下可运行，但建议单独核对是否符合你的训练规则",
            )
        return result

    runtime_hidden_layers = (
        int(model.num_hidden_layers)
        + int(num_empty_layers_add_in_head)
        + int(num_empty_layers_add_in_tail)
    )

    if runtime_hidden_layers < int(parallel.pp):
        result.add_error(
            "pp_gt_runtime_hidden_layers",
            (
                f"pp={parallel.pp} 不能大于 runtime_hidden_layers={runtime_hidden_layers}，"
                "否则会出现空 stage"
            ),
        )

    if require_sp_with_tp_pp and int(model.num_experts) > 1 and int(parallel.tp) > 1 and not bool(parallel.sp):
        result.add_error(
            "sp_required_for_moe_tp",
            "MoE 训练在 tp>1 时必须开启 sp",
        )
    elif require_sp_with_tp_pp and int(parallel.tp) > 1 and int(parallel.pp) > 1 and not bool(parallel.sp):
        result.add_error(
            "sp_required_for_tp_pp",
            "当 tp>1 且 pp>1 同时开启时，必须开启 sp",
        )
    elif int(parallel.tp) > 1 and int(parallel.pp) == 1 and not bool(parallel.sp):
        result.add_warning(
            "tp_without_sp",
            "当前配置开启了 tp 但未开启 sp；这在部分框架配置下可运行，但建议单独核对是否符合你的训练规则",
        )

    if resolved_stage_layer_counts:
        if len(resolved_stage_layer_counts) != int(parallel.pp):
            result.add_error(
                "stage_layer_counts_length_mismatch",
                (
                    f"stage_layer_counts 长度必须等于 pp={parallel.pp}，"
                    f"当前为 {len(resolved_stage_layer_counts)}"
                ),
            )
        elif any(int(value) <= 0 for value in resolved_stage_layer_counts):
            result.add_error(
                "stage_layer_counts_non_positive",
                "stage_layer_counts 中每个 stage 的层数都必须 > 0",
            )
        elif sum(int(value) for value in resolved_stage_layer_counts) != int(model.num_hidden_layers):
            result.add_error(
                "stage_layer_counts_sum_mismatch",
                (
                    "stage_layer_counts 总和必须等于模型真实层数："
                    f"sum={sum(int(value) for value in resolved_stage_layer_counts)} "
                    f"num_hidden_layers={model.num_hidden_layers}"
                ),
            )
    elif not allow_uneven_pp and runtime_hidden_layers % int(parallel.pp) != 0:
        result.add_error(
            "layers_not_divisible_by_pp",
            (
                f"层数不满足 PP 切分：runtime_hidden_layers={runtime_hidden_layers} "
                f"不能整除 pp={parallel.pp}"
            ),
        )

    if int(parallel.vpp) > 1 and runtime_hidden_layers % (int(parallel.pp) * int(parallel.vpp)) != 0:
        result.add_error(
            "layers_not_divisible_by_pp_vpp",
            (
                f"层数不满足 VPP 切分：runtime_hidden_layers={runtime_hidden_layers} "
                f"不能整除 pp*vpp={int(parallel.pp) * int(parallel.vpp)}"
            ),
        )

    tp_checks = {
        "hidden_size": int(model.hidden_size),
        "intermediate_size": int(model.intermediate_size),
        "num_attention_heads": int(model.num_attention_heads),
        "num_key_value_heads": int(model.num_key_value_heads),
    }
    if int(model.num_experts) > 1:
        tp_checks["moe_intermediate_size"] = int(model.moe_intermediate_size)

    for field_name, value in tp_checks.items():
        if value % int(parallel.tp) != 0:
            result.add_error(
                f"tp_not_divide_{field_name}",
                f"{field_name}={value} 不能整除 tp={parallel.tp}",
            )

    if int(model.num_experts) <= 1:
        if int(parallel.ep) != 1:
            result.add_error(
                "ep_on_dense_model",
                "Dense 模型 num_experts<=1 时，ep 必须等于 1",
            )
        return result

    if int(parallel.ep) > int(model.num_experts):
        result.add_error(
            "ep_gt_num_experts",
            f"ep={parallel.ep} 不能大于 num_experts={model.num_experts}",
        )

    if int(model.num_experts) % int(parallel.ep) != 0:
        result.add_error(
            "ep_not_divide_num_experts",
            f"num_experts={model.num_experts} 不能整除 ep={parallel.ep}",
        )

    # EP 必须 <= sharding_degree 且 sharding_degree % ep == 0
    resolved_sd = _resolved_sharding_degree(parallel)
    if int(parallel.ep) > resolved_sd:
        result.add_error(
            "ep_gt_sharding_degree",
            f"ep={parallel.ep} 不能大于 sharding_degree={resolved_sd}",
        )
    elif resolved_sd % int(parallel.ep) != 0:
        result.add_error(
            "sd_not_divisible_by_ep",
            f"sharding_degree={resolved_sd} 必须能整除 ep={parallel.ep}",
        )

    if moe_router_num_groups is not None:
        groups = int(moe_router_num_groups)
        if groups < 1:
            result.add_error(
                "moe_router_num_groups_lt_1",
                f"moe_router_num_groups 必须 >= 1，当前为 {moe_router_num_groups}",
            )
        elif int(model.num_experts) % groups != 0:
            result.add_error(
                "num_experts_not_divide_router_groups",
                (
                    f"num_experts={model.num_experts} 不能整除 "
                    f"moe_router_num_groups={groups}"
                ),
            )

    return result
