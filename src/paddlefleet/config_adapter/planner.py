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

"""Parallelism planning: keep the dims frozen, or shrink EP / PP to fit.

:func:`plan_parallelism` picks one of two planners from the options:

* frozen (``--test-performance``) -- every parallel dimension stays as the
  source declares it; an incompatible target scale is rejected with the list
  of legal node counts rather than forced through.
* shrink (default) -- candidates are tried in order of increasing structural
  intrusion and the first feasible one wins::

      Case A (dims already legal) < EP-only < PP-only < EP+PP joint

  Every candidate must pass the communication-group constraints (C1..C4) and
  the model-structure constraints (M1/M2 for EP, M3/M4/M5 for PP).  Shrinking
  EP scales the routed-expert count, shrinking PP scales
  ``num_hidden_layers`` and realigns VPP / empty tail layers -- those writes
  land in a *separate* copy of ``model_config.json``, never in the source.

  Within a tier the rule is "shrink as little as possible", except for the
  joint tier, where the two axes follow the shrink priority ``EP > PP``: EP
  goes down to its floor first and PP absorbs only the remainder.

TP and SEP are never shrunk: a smaller TP raises per-card memory and risks
OOM.  No dimension that is > 1 in the source is ever reduced to 1, because
that would delete the communication group under test.  VPP is the exception:
whichever planner ran, :func:`enforce_vpp_limit` switches it off when the
final PP is too small for an interleaved schedule.
"""

from __future__ import annotations

import re

from .constraints import (
    MIN_PARALLEL_DEGREE,
    MIN_PP_FOR_VPP,
    check_ep_shrink,
    check_hardware,
    check_pp_shrink,
    ep_candidates,
    min_shrink_cards,
    pp_candidates,
)
from .field_spec import describe_missing, resolve_fields
from .layer_fields import effective_mtp_layers, plan_layer_field_shrink
from .plan import ParallelismPlan
from .topology import TopologyValidator

# Matches a communication-group rejection ("C2 不满足：..."), so the diagnostic
# can put the more actionable model-structure reasons first.
_CONSTRAINT_RE = re.compile(r"C[1-4]\s*不满足")

VPP_FIELD = "virtual_pipeline_model_parallel_size"


def _vpp_off_reason(vpp_old, pp):
    """Why VPP must be 1 at this PP."""
    return (
        f"PP={pp} 时不能开虚拟流水：框架断言 VPP>1 需要 PP>"
        f"{MIN_PP_FOR_VPP - 1}"
        f"（virtual pipeline must run under pp degree > 2），"
        f"VPP {vpp_old} -> 1"
    )


def plan_parallelism(
    config, dims, target_cards, cards_per_node, options, context=None
):
    """Plan the final parallel dims. Returns ``(plan, error_or_None)``."""
    if options.freeze_parallel:
        plan, err = plan_frozen(dims, target_cards, cards_per_node)
    else:
        plan, err = ShrinkPlanner().plan(
            config, dims, target_cards, cards_per_node, context
        )
    if err:
        return None, err
    enforce_vpp_limit(config, plan)
    return plan, None


def enforce_vpp_limit(config, plan):
    """Turn VPP off when the planned PP cannot run an interleaved schedule.

    :data:`~.constraints.MIN_PP_FOR_VPP` is a hard framework assert, and the
    PP-shrink planner already respects it, which leaves the plans that never
    touch PP -- a frozen plan, an EP-only shrink, a source that already fits
    the target -- carrying whatever VPP the source declared.  Dropping it to 1
    cannot break the layer alignment: the number of segments goes from
    ``PP * VPP`` down to ``PP``, which divides it.
    """
    if plan.pp >= MIN_PP_FOR_VPP:
        return
    planned = {key: value for key, value, _reason in plan.yaml_changes}
    vpp = planned.get(VPP_FIELD, config.get(VPP_FIELD, 1) or 1)
    if int(vpp) <= 1:
        return
    plan.yaml_changes.append((VPP_FIELD, 1, _vpp_off_reason(vpp, plan.pp)))


def plan_frozen(dims, target_cards, cards_per_node):
    """Keep every parallel dim; fail when the target scale needs changes."""
    tp, pp, ep, cp, sep = dims
    validator = TopologyValidator(target_cards, cards_per_node)
    ok, message, _details = validator.validate(tp, pp, ep, cp, sep)
    if not ok:
        return None, (
            f"{message}\n"
            f"  说明：--test-performance 只调整 sharding 与 "
            f"global_batch_size，不修改 TP/PP/EP/CP/SEP 和 acc，"
            f"因此目标机器规模必须与源并行度兼容。\n"
            f"  建议：换用上面列出的合法节点数，"
            f"或去掉 --test-performance（默认允许缩小 EP/PP）"
        )
    return (
        ParallelismPlan(
            tp,
            pp,
            ep,
            cp,
            sep,
            note="并行度全部保持不变（--test-performance 冻结 "
            "TP/PP/EP/CP/SEP 与 acc）",
        ),
        None,
    )


class ShrinkPlanner:
    """Shrinks EP / PP just enough to fit the target scale."""

    def plan(self, config, dims, target_cards, cards_per_node, context=None):
        """Plan the final dims, shrinking EP / PP only as much as needed."""
        tp, pp, ep, cp, sep = dims
        context = context or {}

        ok, why = check_hardware(
            target_cards, cards_per_node, tp, pp, ep, cp, sep
        )
        if not ok:
            return None, why

        validator = TopologyValidator(target_cards, cards_per_node)

        # ---- Case A: the source dims already fit the target scale --------
        ok, _msg, _details = validator.validate(tp, pp, ep, cp, sep)
        if ok:
            return (
                ParallelismPlan(
                    tp,
                    pp,
                    ep,
                    cp,
                    sep,
                    note="源并行度已满足目标机器规模的通信组约束，未做缩容",
                ),
                None,
            )

        # ---- Everything below needs the model structure ------------------
        model_config = context.get("model_config")
        if model_config is None:
            return None, (
                f"需要缩小 EP/PP 才能适配 {target_cards} 卡，"
                f"但读不到 model_config.json："
                f"{context.get('model_config_error') or '未提供'}"
            )

        resolved, missing = resolve_fields(model_config)

        def value_of(name):
            field = resolved.get(name)
            return field.value if field is not None else None

        num_experts = value_of("num_experts")
        topk = value_of("num_experts_per_tok")
        experts_key = (
            resolved["num_experts"].writeback_key
            if "num_experts" in resolved
            else "n_routed_experts"
        )
        # MoE detection is structural: EP > 1 means expert parallelism is on,
        # regardless of any use_moe flag (MiniMax only sets EP + the flag).
        can_shrink_ep = (
            ep > MIN_PARALLEL_DEGREE
            and num_experts is not None
            and topk is not None
        )

        # ---- Tier 1: EP only ---------------------------------------------
        rejections = []
        if can_shrink_ep:
            pool = []
            for ep_new in ep_candidates(ep, tp, sep):
                ok, why, experts_new = check_ep_shrink(
                    ep, ep_new, num_experts, topk
                )
                if not ok:
                    rejections.append(f"EP {ep} -> {ep_new}：{why}")
                    continue
                ok, why, _d = validator.validate(tp, pp, ep_new, cp, sep)
                if not ok:
                    rejections.append(
                        f"EP {ep} -> {ep_new}：{_first_constraint(why)}"
                    )
                    continue
                pool.append((ep_new, experts_new))

            if pool:
                # Largest surviving EP = least intrusive within the tier.
                ep_new, experts_new = max(pool, key=lambda item: item[0])
                return (
                    ParallelismPlan(
                        tp,
                        pp,
                        ep_new,
                        cp,
                        sep,
                        yaml_changes=[
                            (
                                "expert_model_parallel_size",
                                ep_new,
                                f"缩小 EP {ep} -> {ep_new} 以满足 "
                                f"{target_cards} 卡下的 C2/C3 约束"
                                f"（PP/TP/CP/SEP 不变）",
                            )
                        ],
                        json_changes=[
                            (
                                experts_key,
                                experts_new,
                                f"随 EP {ep} -> {ep_new} 等比缩减专家数 "
                                f"{num_experts} -> {experts_new}"
                                f"（保证每个 EP rank 专家数相等且 >= top-k"
                                f"={topk}）",
                            )
                        ],
                        note=f"仅缩 EP：{ep} -> {ep_new}",
                    ),
                    None,
                )

        # ---- Tier 2 / Tier 3 need the pipeline layout --------------------
        layout, err = self._pipeline_layout(
            config, resolved, missing, model_config
        )
        if pp > MIN_PARALLEL_DEGREE and err:
            return None, err

        if pp > MIN_PARALLEL_DEGREE and layout is not None:
            plan = self._plan_pp_only(
                validator, dims, layout, target_cards, rejections, model_config
            )
            if plan is not None:
                return plan, None

            if can_shrink_ep:
                plan = self._plan_joint(
                    validator,
                    dims,
                    layout,
                    (num_experts, topk, experts_key),
                    target_cards,
                    rejections,
                    model_config,
                )
                if plan is not None:
                    return plan, None

        return None, self._diagnose(
            dims,
            target_cards,
            cards_per_node,
            missing,
            can_shrink_ep,
            layout,
            rejections,
        )

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _pipeline_layout(config, resolved, missing, model_config):
        """Collect everything PP shrinking needs. ``(layout, error)``."""
        if "num_hidden_layers" in missing:
            return None, describe_missing(
                "num_hidden_layers", missing["num_hidden_layers"]
            )

        def value_of(name):
            field = resolved.get(name)
            return field.value if field is not None else None

        mtp = effective_mtp_layers(model_config)

        # What the framework actually segments (gpt_builders.build_gpt_model
        # + GPTModel.get_sequential_layers) is
        #     head_empty + num_hidden_layers + mtp + tail_empty
        # and the MTP layers only join that count when
        # separate_mtp_headloss puts MultiTokenPredictionLayer into
        # seg_method.  In that case the builder also drops exactly one tail
        # empty layer to make room for them
        # (``num_empty_layers_add_in_tail -= 1``), so the yaml tail must stay
        # >= 1 and contributes ``tail - 1`` segmented layers.  Net effect on
        # the total: ``mtp - 1``, which is zero for the usual single MTP
        # layer -- adding the full ``mtp`` here would overshoot by one and
        # produce a config that dies in do_segment().
        head = int(config.get("num_empty_layers_add_in_head", 0) or 0)
        min_tail = 0
        if config.get("separate_mtp_headloss"):
            head += mtp - 1
            min_tail = 1

        return {
            "layers": value_of("num_hidden_layers"),
            "layers_key": resolved["num_hidden_layers"].writeback_key,
            "head": head,
            "tail": int(config.get("num_empty_layers_add_in_tail", 0) or 0),
            "min_tail": min_tail,
            "vpp": int(
                config.get("virtual_pipeline_model_parallel_size", 1) or 1
            ),
            "first_k": int(value_of("first_k_dense_replace") or 0),
            "mtp": mtp,
        }, None

    @staticmethod
    def _pp_changes(
        pp, pp_new, layout, meta, target_cards, layer_changes, note_extra=""
    ):
        """YAML + JSON writes implied by a PP shrink."""
        yaml_changes = [
            (
                "pipeline_model_parallel_size",
                pp_new,
                f"缩小 PP {pp} -> {pp_new} 以满足 {target_cards} 卡下的 "
                f"C1/C2 约束{note_extra}",
            )
        ]
        if meta["vpp_new"] != layout["vpp"]:
            if meta["vpp_new"] == 1 and pp_new < MIN_PP_FOR_VPP:
                reason = _vpp_off_reason(layout["vpp"], pp_new)
            else:
                reason = (
                    f"PP 缩容后重新对齐 VPP "
                    f"{layout['vpp']} -> {meta['vpp_new']}"
                )
            yaml_changes.append((VPP_FIELD, meta["vpp_new"], reason))
        if meta["tail_new"] != layout["tail"]:
            yaml_changes.append(
                (
                    "num_empty_layers_add_in_tail",
                    meta["tail_new"],
                    f"调整尾部空层把总层数对齐到 PP×VPP："
                    f"{layout['tail']} -> {meta['tail_new']}",
                )
            )
        json_changes = [
            (
                layout["layers_key"],
                meta["layers_new"],
                f"随 PP {pp} -> {pp_new} 等比缩减层数 "
                f"{layout['layers']} -> {meta['layers_new']}",
            )
        ]
        # Every per-layer list must follow num_hidden_layers or the framework
        # refuses to start.
        json_changes.extend(layer_changes)
        return yaml_changes, json_changes

    @staticmethod
    def _layer_field_changes(model_config, layout, meta, rejections, label):
        """Per-layer list rewrites for a candidate. ``(changes, ok)``."""
        changes, err = plan_layer_field_shrink(
            model_config,
            layout["layers"],
            meta["layers_new"],
            layout["mtp"],
        )
        if err:
            rejections.append(f"{label}：{err}")
            return [], False
        return changes, True

    def _plan_pp_only(
        self, validator, dims, layout, target_cards, rejections, model_config
    ):
        """Tier 2: shrink PP only. Returns a plan or ``None``."""
        tp, pp, ep, cp, sep = dims
        pool = []
        for pp_new in pp_candidates(pp):
            ok, why, meta = check_pp_shrink(
                pp,
                pp_new,
                layout["layers"],
                layout["head"],
                layout["tail"],
                layout["vpp"],
                first_k_dense_replace=layout["first_k"],
                min_tail=layout["min_tail"],
            )
            if not ok:
                rejections.append(f"PP {pp} -> {pp_new}：{why}")
                continue
            ok, why, _d = validator.validate(tp, pp_new, ep, cp, sep)
            if not ok:
                rejections.append(
                    f"PP {pp} -> {pp_new}：{_first_constraint(why)}"
                )
                continue
            layer_changes, ok = self._layer_field_changes(
                model_config, layout, meta, rejections, f"PP {pp} -> {pp_new}"
            )
            if not ok:
                continue
            pool.append((pp_new, meta, layer_changes))

        if not pool:
            return None

        pp_new, meta, layer_changes = max(pool, key=lambda item: item[0])
        yaml_changes, json_changes = self._pp_changes(
            pp,
            pp_new,
            layout,
            meta,
            target_cards,
            layer_changes,
            "（EP/TP/CP/SEP 不变）",
        )
        return ParallelismPlan(
            tp,
            pp_new,
            ep,
            cp,
            sep,
            yaml_changes=yaml_changes,
            json_changes=json_changes,
            warnings=[meta["warning"]] if meta.get("warning") else [],
            note=f"仅缩 PP：{pp} -> {pp_new}",
        )

    def _plan_joint(
        self,
        validator,
        dims,
        layout,
        moe,
        target_cards,
        rejections,
        model_config,
    ):
        """Tier 3: shrink EP and PP together. Returns a plan or ``None``.

        Reached only when neither single-axis shrink survives -- e.g. a source
        ``(pp=8, ep=8)`` targeting 8 cards, where any one-dimension shrink
        still overshoots the card count.

        The two axes are not interchangeable, so they follow the shrink
        priority ``EP > PP``: EP absorbs the reduction down to its floor and PP
        is only shrunk for whatever is left over.  Shrinking EP costs routed
        experts and nothing else, while shrinking PP scales
        ``num_hidden_layers`` -- and with it every per-layer list, the VPP
        degree and the empty tail -- so a shallower pipeline cut is worth a
        deeper expert cut.
        """
        tp, pp, ep, cp, sep = dims
        num_experts, topk, experts_key = moe

        pool = []
        for ep_new in ep_candidates(ep, tp, sep):
            ok, why, experts_new = check_ep_shrink(
                ep, ep_new, num_experts, topk
            )
            if not ok:
                continue
            for pp_new in pp_candidates(pp):
                ok, why, meta = check_pp_shrink(
                    pp,
                    pp_new,
                    layout["layers"],
                    layout["head"],
                    layout["tail"],
                    layout["vpp"],
                    first_k_dense_replace=layout["first_k"],
                    min_tail=layout["min_tail"],
                )
                if not ok:
                    continue
                ok, why, _d = validator.validate(tp, pp_new, ep_new, cp, sep)
                if not ok:
                    rejections.append(
                        f"EP {ep} -> {ep_new} + PP {pp} -> {pp_new}："
                        f"{_first_constraint(why)}"
                    )
                    continue
                layer_changes, ok = self._layer_field_changes(
                    model_config,
                    layout,
                    meta,
                    rejections,
                    f"PP {pp} -> {pp_new}",
                )
                if not ok:
                    continue
                pool.append((ep_new, pp_new, experts_new, meta, layer_changes))

        if not pool:
            return None

        # Shrink priority EP > PP: smallest feasible EP first, then the
        # largest PP that still fits under it.
        ep_new, pp_new, experts_new, meta, layer_changes = min(
            pool, key=lambda item: (item[0], -item[1])
        )
        yaml_changes, json_changes = self._pp_changes(
            pp,
            pp_new,
            layout,
            meta,
            target_cards,
            layer_changes,
            "（与 EP 联合缩容）",
        )
        yaml_changes.insert(
            0,
            (
                "expert_model_parallel_size",
                ep_new,
                f"缩小 EP {ep} -> {ep_new}：单独缩 EP 或单独缩 PP 都无法"
                f"适配 {target_cards} 卡，改为联合缩容；按 EP 优先于 PP 的"
                f"缩容顺序，先把 EP 压到可行下限，再尽量少缩 PP"
                f"（缩 PP 会连带改层数/VPP/尾部空层/逐层配置）",
            ),
        )
        json_changes.insert(
            0,
            (
                experts_key,
                experts_new,
                f"随 EP {ep} -> {ep_new} 等比缩减专家数 "
                f"{num_experts} -> {experts_new}（>= top-k={topk}）",
            ),
        )
        return ParallelismPlan(
            tp,
            pp_new,
            ep_new,
            cp,
            sep,
            yaml_changes=yaml_changes,
            json_changes=json_changes,
            warnings=[meta["warning"]] if meta.get("warning") else [],
            note=f"EP+PP 联合缩容：EP {ep} -> {ep_new}，PP {pp} -> {pp_new}",
        )

    @staticmethod
    def _diagnose(
        dims,
        target_cards,
        cards_per_node,
        missing,
        can_shrink_ep,
        layout,
        rejections,
    ):
        """Explain why no candidate survived, with an actionable suggestion."""
        tp, pp, ep, cp, sep = dims
        reasons = []

        if ep > 1 and not can_shrink_ep:
            if ep == MIN_PARALLEL_DEGREE:
                reasons.append(
                    f"EP={ep} 已是允许的最小值"
                    f"（不允许把 EP 从 >1 缩到 1，否则等于去掉专家并行）"
                )
            for name in ("num_experts", "num_experts_per_tok"):
                if name in missing:
                    reasons.append(describe_missing(name, missing[name]))

        if pp > 1 and pp == MIN_PARALLEL_DEGREE:
            reasons.append(
                f"PP={pp} 已是允许的最小值"
                f"（不允许把 PP 从 >1 缩到 1，否则等于去掉流水并行）"
            )
        elif pp > MIN_PARALLEL_DEGREE and layout is not None:
            floor_layers = layout["layers"] * MIN_PARALLEL_DEGREE // pp
            reasons.append(
                f"PP 最多缩到 {MIN_PARALLEL_DEGREE}，此时层数为 "
                f"{floor_layers}（={layout['layers']}×"
                f"{MIN_PARALLEL_DEGREE}/{pp}），仍不满足约束"
            )

        floor = tp * sep * MIN_PARALLEL_DEGREE if pp > 1 else tp * sep
        if target_cards % (tp * sep) != 0:
            reasons.append(
                f"C1 约束：target_cards={target_cards} 不能被 TP×SEP="
                f"{tp}×{sep}={tp * sep} 整除"
            )
        elif target_cards < floor:
            reasons.append(
                f"目标卡数 {target_cards} 小于并行度下限 {floor}"
                f"（TP×SEP×PP_min）"
            )

        if not reasons:
            reasons.append("所有候选组合都不满足通信组约束或模型结构约束")

        # De-duplicated, truncated candidate rejections: the concrete reason a
        # given (EP, PP) pair was thrown away is usually the fastest way to
        # see what to change.  Model-structure reasons come first -- they are
        # actionable, while the C-constraint ones repeat the same arithmetic.
        unique = list(dict.fromkeys(rejections))
        topological = [r for r in unique if _CONSTRAINT_RE.search(r)]
        structural = [r for r in unique if r not in topological]
        detail = (structural + topological)[:6]
        detail_block = ""
        if detail:
            detail_block = "\n  候选淘汰明细：\n    " + "\n    ".join(detail)

        min_cards = min_shrink_cards(tp, pp, ep, cp, sep, cards_per_node)
        if min_cards is None:
            advice = (
                "当前维度组合本身不合法（C3 要求 EP % (TP×SEP) == 0，"
                "C5 禁止 SEP 与 CP 同时 >1），任何卡数都无法适配，"
                "必须先修改 TP/SEP/EP/CP"
            )
            return (
                f"{target_cards} 卡下找不到合法的 EP/PP 缩容方案 "
                f"(tp={tp}, pp={pp}, ep={ep}, cp={cp}, sep={sep})。\n"
                f"  阻断原因：\n    "
                + "\n    ".join(reasons)
                + detail_block
                + f"\n  建议：{advice}"
            )
        min_nodes = max(min_cards // cards_per_node, 1)
        if min_cards > target_cards:
            advice = (
                f"改用 --target-nodes {min_nodes}"
                f"（{min_cards} 卡，缩容下限），"
                f"或手动降低源 YAML 的 TP/PP 后重试"
            )
        else:
            advice = (
                f"{target_cards} 卡已达到该配置的缩容下限，"
                f"请调整模型结构（如 --set json:<专家数字段>=<能被目标 EP "
                f"整除的值>），或手动降低源 YAML 的 TP/PP/EP 后重试"
            )
        return (
            f"{target_cards} 卡下找不到合法的 EP/PP 缩容方案 "
            f"(tp={tp}, pp={pp}, ep={ep}, cp={cp}, sep={sep})。\n"
            f"  阻断原因：\n    " + "\n    ".join(reasons) + detail_block + "\n"
            "  建议：" + advice
        )


def _first_constraint(message):
    """Pull the first ``Cx`` line out of a validator message."""
    for line in message.splitlines():
        stripped = line.strip()
        if stripped.startswith("C") and "不满足" in stripped:
            return stripped
    return message.splitlines()[0].strip()
