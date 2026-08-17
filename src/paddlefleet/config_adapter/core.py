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

"""The adaptation pipeline.

``ConfigAdapter.adapt`` runs the whole rewrite in one pass::

    load YAML
      -> apply --set yaml: overrides (and pin them: nothing else may touch
         those keys afterwards)
      -> drop fa_version (an environment-specific flash-attention pin)
      -> load model_config.json + apply --set json: overrides, when the
         profile or the user needs it
      -> profile.plan(): decide the final TP/PP/EP/CP/SEP
      -> apply the plan's model_config.json writes
      -> inject the determinism switches (accuracy profile only)
      -> write model_config.json when it changed, and point
         model_name_or_path at it
      -> apply the plan's YAML writes and re-validate C1..C4
      -> scale the batch settings, sharding and data_parallel_size
      -> write the YAML and render the report

Every write goes through the change log, so the report can attribute each
field to the decision that produced it.
"""

from __future__ import annotations

from pathlib import Path

from .io_writers import JsonWriter, YamlWriter
from .model_config_resolver import (
    ModelConfigResolveError,
    build_adapted_dir,
    resolve_model_config,
    rewrite_model_name_or_path,
)
from .planner import plan_parallelism
from .precision import plan_precision_switches
from .report import ChangeLog, format_header, format_report
from .strategies import BATCH_STRATEGIES
from .topology import TopologyValidator
from .utils import PARALLEL_FIELDS, extract_parallel_params

#: Fields whose value the adapter derives itself. A ``--set`` pin on any of
#: them is refused rather than silently skipped, because the plan, the
#: validation and the report all assume the adapter's own value.
ADAPTER_CONTROLLED_FIELDS = frozenset(
    set(PARALLEL_FIELDS.values())
    | {
        "virtual_pipeline_model_parallel_size",
        "num_empty_layers_add_in_tail",
        "sharding_parallel_size",
        "data_parallel_size",
        "model_name_or_path",
    }
)


class ConfigAdapter:
    """Rewrites one training YAML (and its model_config.json) for a scale."""

    def __init__(
        self,
        options,
        target_nodes,
        cards_per_node=8,
        yaml_overrides=None,
        json_overrides=None,
        auto_overrides=None,
        output_dir="./adapted_configs",
        in_place=False,
        force=False,
    ):
        self.options = options
        self.target_nodes = target_nodes
        self.cards_per_node = cards_per_node
        self.target_cards = target_nodes * cards_per_node
        # --set values, split by how the target document was decided:
        # explicitly prefixed (yaml:/json:) versus auto-detected.
        self.yaml_overrides = dict(yaml_overrides or {})
        self.json_overrides = dict(json_overrides or {})
        self.auto_overrides = dict(auto_overrides or {})
        self.output_dir = Path(output_dir)
        self.in_place = in_place
        self.force = force

        self.scale_tag = f"{self.target_cards}cards"
        self.yaml_writer = YamlWriter()
        self.json_writer = JsonWriter()
        # Original bytes of an in-place rewritten model_config.json, used to
        # roll back if the YAML write then fails.
        self._json_backup = None

    # ------------------------------------------------------------------ main
    def adapt(self, input_path):
        """Adapt one YAML file. Returns ``(ok, message)``."""
        input_path = Path(input_path)
        log = ChangeLog()

        # Fields the adapter must own: pinning them with --set would make the
        # generated file disagree with the reported parallelism / sharding.
        # model_name_or_path is in the set even for --in-place runs, which do
        # not rewrite it: repointing it mid-run would decide *which*
        # model_config.json is loaded, shrunk and snapshotted for the patch,
        # and the answer differs between the prefixed and prefix-less forms.
        # Every --set form counts: prefix-less values are routed to the YAML
        # later, and a json: prefix would park a training-parallelism key in
        # model_config.json while the YAML keeps the adapter's own value.
        requested = (
            set(self.yaml_overrides)
            | set(self.json_overrides)
            | set(self.auto_overrides)
        )
        pinned = sorted(ADAPTER_CONTROLLED_FIELDS & requested)
        if pinned:
            return False, (
                f"以下字段由适配器统一计算，不能用 --set 锁定：{pinned}。"
                f"锁定后生成的配置会与报告里的并行度 / sharding 不一致；"
                f"请去掉对应的 --set，或直接改源 YAML 后再适配"
            )

        config = self.yaml_writer.load(input_path)
        if config is None:
            return False, f"配置文件为空：{input_path}"

        if self.yaml_overrides:
            log.record(
                "yaml",
                self.yaml_writer.apply_config_map(config, self.yaml_overrides),
                "用户通过 --set yaml: 指定，自动适配不会再覆盖该字段",
            )

        if "fa_version" in config and "fa_version" not in self.yaml_overrides:
            log.record_removed(
                "yaml",
                "fa_version",
                config.pop("fa_version"),
                "fa_version 是与环境强绑定的 flash-attention 版本 pin，"
                "适配后的配置不携带",
            )

        dims_before = extract_parallel_params(config)
        scale_source = {
            key: config.get(key)
            for key in (
                "sharding_parallel_size",
                "data_parallel_size",
                "global_batch_size",
                "per_device_train_batch_size",
                "gradient_accumulation_steps",
            )
        }

        model_config, model_dir, json_path, json_error = self._load_json(
            config, input_path
        )
        if self.json_overrides:
            if model_config is None:
                return False, f"--set json: 无法应用：{json_error}"
            log.record(
                "json",
                self.json_writer.apply_config_map(
                    model_config, self.json_overrides
                ),
                "用户通过 --set json: 指定，自动适配不会再覆盖该字段",
            )

        self._apply_auto_overrides(config, model_config, log)

        plan, err = plan_parallelism(
            config,
            dims_before,
            self.target_cards,
            self.cards_per_node,
            self.options,
            context={
                "model_config": model_config,
                "model_config_error": json_error,
                "input_path": input_path,
            },
        )
        if err:
            return False, f"{input_path.name}: {err}"

        for key, value, reason in plan.json_changes:
            log.record(
                "json",
                self.json_writer.apply_config_map(
                    model_config, {key: value}, protected=self.json_overrides
                ),
                reason,
            )

        skipped_switches = []
        if self.options.inject_precision:
            skipped_switches = self._inject_precision(config, model_config, log)

        for key, value, reason in plan.yaml_changes:
            log.record(
                "yaml",
                self.yaml_writer.apply_config_map(
                    config, {key: value}, protected=self.yaml_overrides
                ),
                reason,
            )

        dims_after = plan.dims()
        validator = TopologyValidator(self.target_cards, self.cards_per_node)
        ok, message, details = validator.validate(*dims_after)
        if not ok:
            return False, f"{input_path.name}: {message}"

        orig_cards, scale_warning, err = self._infer_orig_cards(
            scale_source, dims_before
        )
        if err:
            return False, f"{input_path.name}: {err}"
        if scale_warning:
            plan.warnings.append(scale_warning)

        err = self._scale_batch_and_sharding(
            config, log, scale_source, orig_cards, dims_after
        )
        if err:
            return False, f"{input_path.name}: {err}"

        # Nothing has touched the filesystem up to this point: every rewrite
        # above happened in memory, so a validation failure leaves both source
        # files exactly as they were.
        model_config_output, err = self._write_model_config(
            config, log, model_config, model_dir, json_path, input_path
        )
        if err:
            return False, f"{input_path.name}: {err}"

        info = self._build_info(
            input_path,
            orig_cards,
            dims_before,
            dims_after,
            details,
            plan,
            model_config_output,
            skipped_switches,
        )
        output_path = Path(info["output"])
        # In-place runs must not accumulate a new banner on every rewrite.
        header = "" if self.in_place else format_header(info)
        try:
            self.yaml_writer.write(config, output_path, header=header)
        except OSError as exc:
            # The companion JSON was already persisted; put it back so an
            # in-place run cannot leave the two source files disagreeing.
            if self._json_backup is not None:
                Path(json_path).write_bytes(self._json_backup)
            return False, f"{input_path.name}: 写出 {output_path} 失败：{exc}"
        return True, format_report(info, log)

    # --------------------------------------------------------------- stages
    def _apply_auto_overrides(self, config, model_config, log):
        """Route prefix-less ``--set`` values to whoever declares the key.

        * declared in the YAML only -> YAML;
        * declared in ``model_config.json`` only -> JSON;
        * declared in both -> both (the framework reads either);
        * declared nowhere -> added to the YAML (use ``--set json:KEY=VALUE``
          to create a brand-new model_config field instead).

        Keys routed here are remembered as protected, exactly like the
        explicitly prefixed ones.
        """
        for key, value in self.auto_overrides.items():
            in_yaml = key in config
            in_json = model_config is not None and key in model_config

            if in_yaml or not in_json:
                where = "yaml 与 json 都声明了该字段" if in_json else None
                if not in_yaml:
                    where = "yaml 与 json 都没有该字段，按默认新增到 yaml"
                elif where is None:
                    where = "自动匹配到 yaml"
                self.yaml_overrides[key] = value
                log.record(
                    "yaml",
                    self.yaml_writer.apply_config_map(config, {key: value}),
                    f"用户通过 --set 指定（{where}）",
                )
            if in_json:
                self.json_overrides[key] = value
                log.record(
                    "json",
                    self.json_writer.apply_config_map(
                        model_config, {key: value}
                    ),
                    "用户通过 --set 指定（"
                    + (
                        "yaml 与 json 都声明了该字段"
                        if in_yaml
                        else "自动匹配到 model_config.json"
                    )
                    + "）",
                )

    def _load_json(self, config, input_path):
        """Load ``model_config.json`` when it may be needed.

        Returns ``(model_config, model_dir, json_path, error)``.  A failure is
        not fatal here: shrinking only *needs* the JSON when EP / PP actually
        have to move, so the error is carried forward and reported by whoever
        needs it.
        """
        needed = (
            self.options.needs_model_config
            or bool(self.json_overrides)
            or bool(self.auto_overrides)
        )
        if not needed:
            return None, None, None, "本次运行不需要 model_config.json"

        try:
            model_dir, json_path = resolve_model_config(
                config.get("model_name_or_path"), input_path.parent
            )
            return self.json_writer.load(json_path), model_dir, json_path, None
        except (ModelConfigResolveError, ValueError) as exc:
            return None, None, None, str(exc)

    def _inject_precision(self, config, model_config, log):
        """Pin the determinism switches. Returns notes for skipped ones."""
        applied, skipped = plan_precision_switches(config, model_config)
        for target, key, value, reason in applied:
            if target == "yaml":
                diffs = self.yaml_writer.apply_config_map(
                    config, {key: value}, protected=self.yaml_overrides
                )
            else:
                diffs = self.json_writer.apply_config_map(
                    model_config, {key: value}, protected=self.json_overrides
                )
            log.record(target, diffs, reason)
        return skipped

    def _write_model_config(
        self, config, log, model_config, model_dir, json_path, input_path
    ):
        """Persist ``model_config.json`` when it changed.

        Returns ``(path_or_None, error_or_None)``.  Called only after every
        check has passed, so this is the first filesystem write of the run.
        A pre-existing adapted directory is refused unless ``--force`` was
        given: it usually means a stale artefact the user should look at.
        """
        self._json_backup = None
        if model_config is None or not log.by_target("json"):
            return None, None

        if self.in_place:
            # Keep the original bytes so a later YAML write failure can be
            # rolled back (see adapt()).
            self._json_backup = Path(json_path).read_bytes()
            self.json_writer.write(model_config, json_path)
            return str(json_path), None

        adapted_dir = build_adapted_dir(
            self.output_dir, model_dir.name, self.scale_tag
        )
        try:
            adapted_dir.mkdir(parents=True, exist_ok=self.force)
        except FileExistsError:
            return None, (
                f"已存在生成目录 {adapted_dir}，为避免覆盖上一次的产物而中止；"
                f"确认可以覆盖请加 -f/--force"
            )
        except OSError as exc:
            return None, f"无法创建 {adapted_dir}：{exc}"

        target_json = adapted_dir / "model_config.json"
        self.json_writer.write(model_config, target_json)

        raw = config.get("model_name_or_path")
        log.record(
            "yaml",
            self.yaml_writer.apply_config_map(
                config,
                {
                    "model_name_or_path": rewrite_model_name_or_path(
                        adapted_dir,
                        input_path.parent,
                        bool(raw) and str(raw).startswith("/"),
                    )
                },
            ),
            "指向本次生成的 model_config 目录（源 model_config.json 不修改）",
        )
        return str(target_json), None

    @staticmethod
    def _infer_orig_cards(scale_source, dims_before):
        """Infer the source job's GPU count.

        Returns ``(cards, warning, error)``.  Two independent estimates are
        computed when the config carries enough information:

        * comm groups: ``DP * sharding * TP * SEP * PP`` -- ``sharding`` alone
          is a group size, not the world size, so a dense job with ``DP > 1``
          would otherwise be under-counted;
        * batch settings: ``GBS / (micro_bs * acc) * TP * PP * CP``.

        When both exist and disagree, the estimate that is not missing a
        factor wins: the comm-group one only if ``data_parallel_size`` is
        declared, otherwise the batch one (an undeclared DP is exactly the
        factor the comm-group formula would be missing).  Either way the
        mismatch is reported as a warning, because it means the source YAML is
        not self-consistent.
        """
        tp, pp, ep, cp, sep = dims_before

        sharding = scale_source["sharding_parallel_size"]
        raw_dp = scale_source["data_parallel_size"]
        dp_declared = raw_dp is not None and int(raw_dp) > 0
        dp = int(raw_dp) if dp_declared else 1
        group_based = None
        if sharding is not None and int(sharding) > 0:
            group_based = dp * int(sharding) * tp * sep * pp

        gbs = scale_source["global_batch_size"]
        micro_bs = scale_source["per_device_train_batch_size"]
        grad_accum = scale_source["gradient_accumulation_steps"]
        batch_based = None
        if gbs and micro_bs and grad_accum:
            dataset_world_size = int(gbs) // (int(micro_bs) * int(grad_accum))
            if dataset_world_size > 0:
                batch_based = dataset_world_size * tp * pp * cp

        if group_based is not None and batch_based is not None:
            if group_based == batch_based:
                return group_based, None, None
            chosen = group_based if dp_declared else batch_based
            warning = (
                f"源卡数的两种推断不一致：按通信组"
                f"（DP={dp}×sharding={sharding}×TP={tp}×SEP={sep}×PP={pp}）"
                f"为 {group_based}，按 batch 字段"
                f"（GBS/(micro×acc)×TP×PP×CP）为 {batch_based}；"
                + (
                    f"源 YAML 未声明 data_parallel_size，"
                    f"通信组公式会漏掉这个因子，因此取 batch 字段的 {chosen}。"
                    if not dp_declared
                    else f"取通信组的 {chosen}。"
                )
                + "如果不对，请在源 YAML 里写明 data_parallel_size "
                "或修正 batch 字段后重新适配"
            )
            return chosen, warning, None
        if group_based is not None:
            return group_based, None, None
        if batch_based is not None:
            return batch_based, None, None

        if gbs is not None:
            return (
                None,
                None,
                (
                    "无法推断源作业的卡数：既没有可用的 sharding_parallel_size"
                    "（>0），也无法用 global_batch_size / "
                    "per_device_train_batch_size / gradient_accumulation_steps "
                    "反推。请在源 YAML 里补上 sharding_parallel_size / "
                    "data_parallel_size，或补全 batch 字段"
                ),
            )
        return None, None, None

    def _scale_batch_and_sharding(
        self, config, log, scale_source, orig_cards, dims_after
    ):
        """Rewrite batch / sharding / data_parallel_size. Returns an error."""
        gbs = scale_source["global_batch_size"]
        grad_accum = scale_source["gradient_accumulation_steps"]
        gbs = int(gbs) if gbs is not None else None
        grad_accum = int(grad_accum) if grad_accum else 1

        if orig_cards is None:
            batch_map = {"gradient_accumulation_steps": grad_accum}
            reason = (
                f"推断不出源卡数，acc 保持 {grad_accum} 不变，"
                f"GBS 由框架按实际卡数反推"
            )
        else:
            strategy = BATCH_STRATEGIES[self.options.batch_strategy]
            batch_map, reason, err = strategy(
                gbs, grad_accum, orig_cards, self.target_cards
            )
            if err:
                return err

        # Only rewrite batch fields the source actually declares.
        batch_map = {k: v for k, v in batch_map.items() if k in config}
        log.record(
            "yaml",
            self.yaml_writer.apply_config_map(
                config, batch_map, protected=self.yaml_overrides
            ),
            reason,
        )

        tp, pp, ep, cp, sep = dims_after
        new_sharding = self.target_cards // (tp * sep * pp)
        sharding_field = "sharding_parallel_size"
        current = config.get(sharding_field)
        if current is not None and int(current) != -1:
            log.record(
                "yaml",
                self.yaml_writer.apply_config_map(
                    config,
                    {sharding_field: new_sharding},
                    protected=self.yaml_overrides,
                ),
                f"sharding = 目标卡数 / (TP×SEP×PP) = {self.target_cards} / "
                f"({tp}×{sep}×{pp}) = {new_sharding}",
            )

        if "data_parallel_size" in config:
            log.record(
                "yaml",
                self.yaml_writer.apply_config_map(
                    config,
                    {"data_parallel_size": 1},
                    protected=self.yaml_overrides,
                ),
                "Fleet 要求纯 sharding 数据并行：data_parallel_size 固定为 1，"
                "数据并行度由 sharding 承担",
            )
        return None

    # ---------------------------------------------------------------- report
    def _build_info(
        self,
        input_path,
        orig_cards,
        dims_before,
        dims_after,
        details,
        plan,
        model_config_output,
        skipped_switches,
    ):
        """Assemble everything the header / report need."""
        tp0, pp0, ep0, cp0, sep0 = dims_before
        tp1, pp1, ep1, cp1, sep1 = dims_after

        dims_line = "  ".join(
            f"{name.upper()} {before}->{after}"
            for name, before, after in zip(
                PARALLEL_FIELDS,
                dims_before,
                dims_after,
                strict=True,
            )
        )

        orig_sharding = orig_cards // (tp0 * sep0 * pp0) if orig_cards else "?"
        sharding_line = f"{orig_sharding} -> {details['sharding']}"
        derived = []
        if ep1 > 1:
            derived.append(f"moe_sharding={details['moe_sharding']}")
            derived.append(f"dense_sharding={details['dense_sharding']}")
        if cp1 > 1:
            derived.append(f"cp_sharding={details['cp_sharding']}")
        if derived:
            sharding_line += "（" + ", ".join(derived) + "）"

        if orig_cards and orig_cards % self.cards_per_node == 0:
            orig_nodes_label = orig_cards // self.cards_per_node
            orig_scale_label = f"{orig_nodes_label} 节点 / {orig_cards} 卡"
        elif orig_cards:
            orig_nodes_label = "UNKNOWN"
            orig_scale_label = f"{orig_cards} 卡"
        else:
            orig_nodes_label = "UNKNOWN"
            orig_scale_label = "未知规模"

        if self.in_place:
            output = input_path
        else:
            output = self.output_dir / (
                f"{input_path.stem}_adapted_{self.scale_tag}{input_path.suffix}"
            )

        return {
            "input": str(input_path),
            "output": str(output),
            "profile": self.options.label,
            "profile_flag": self.options.flags,
            "batch_strategy": self.options.batch_strategy,
            "orig_cards_label": orig_cards if orig_cards else "UNKNOWN",
            "orig_nodes_label": orig_nodes_label,
            "orig_scale_label": orig_scale_label,
            "target_cards": self.target_cards,
            "target_nodes": self.target_nodes,
            "cards_per_node": self.cards_per_node,
            "dims_line": dims_line,
            "sharding_line": sharding_line,
            "plan_note": plan.note or "无",
            "model_config_output": model_config_output,
            "skipped_switches": skipped_switches,
            "warnings": plan.warnings,
        }


def inspect_config(input_path, cards_per_node=8, max_nodes=16):
    """Read-only inspection used when no target scale is given.

    Returns ``(orig_cards, orig_nodes, valid_nodes)``: the inferred source
    scale plus every node count within ``max_nodes`` whose GPU count satisfies
    C1..C4 for the source parallelism.  Nothing is written.
    """
    config = YamlWriter().load(input_path)
    if config is None:
        raise ValueError(f"配置文件为空：{input_path}")

    dims = extract_parallel_params(config)
    scale_source = {
        key: config.get(key)
        for key in (
            "sharding_parallel_size",
            "data_parallel_size",
            "global_batch_size",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
        )
    }
    orig_cards, _warning, _err = ConfigAdapter._infer_orig_cards(
        scale_source, dims
    )
    orig_nodes = (
        orig_cards // cards_per_node
        if orig_cards and orig_cards % cards_per_node == 0
        else None
    )

    validator = TopologyValidator(cards_per_node, cards_per_node)
    cards = validator.suggest_valid_cards(*dims, max_nodes=max_nodes)
    valid_nodes = sorted({max(c // cards_per_node, 1) for c in cards})
    return orig_cards, orig_nodes, valid_nodes
