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
      -> load model_config.json + apply --set json: overrides, when the
         profile or the user needs it
      -> profile.plan(): decide the final TP/PP/EP/CP/SEP
      -> apply the plan's model_config.json writes
      -> inject the determinism switches (accuracy profile only)
      -> write model_config.json when it changed, and point
         model_name_or_path at it
      -> apply the plan's YAML writes and re-validate C1..C4
      -> scale the batch settings, sharding and data_parallel_size, and add
         the switches that compensate for a smaller sharding degree
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
from .sharding_shrink import plan_sharding_shrink_switches
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


#: YAML key that carries the training sequence length.
SEQ_FIELD = "max_seq_length"


def _yaml_truthy(value):
    """Loose YAML bool: accepts real bools plus "true"/"yes"/"1" spellings."""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "on", "1"}
    return bool(value)


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
        scale_seq_length=None,
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
        # --scale-seq-length: rewrite max_seq_length and scale CP with it.
        self.scale_seq_length = (
            int(scale_seq_length) if scale_seq_length is not None else None
        )

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

        if self.scale_seq_length is not None and SEQ_FIELD in requested:
            return False, (
                f"--scale-seq-length 与 --set {SEQ_FIELD} 冲突："
                f"两者都要决定序列长度，请只保留其一"
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

        dims_planned, seq_warnings, err = self._apply_seq_scaling(
            config, dims_before, model_config, log
        )
        if err:
            return False, f"{input_path.name}: {err}"

        plan, err = plan_parallelism(
            config,
            dims_planned,
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
        plan.warnings.extend(seq_warnings)
        if "fa_version" in config:
            plan.warnings.append(
                f"fa_version={config['fa_version']} 按源配置原样保留：适配的"
                f"前提是目标机器与源作业同构（同 GPU 架构 / 同镜像）。若目标"
                f"环境缺少对应的 flash-attention kernel，请用 "
                f"--set fa_version=<版本> 改写或手动删除该字段"
            )

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
            config, log, scale_source, orig_cards, dims_before, dims_after
        )
        if err:
            return False, f"{input_path.name}: {err}"

        self._drop_stale_checkpoint_refs(config, log, dims_before, dims_after)

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

    def _apply_seq_scaling(self, config, dims_before, model_config, log):
        """Rewrite ``max_seq_length`` and scale CP by the same ratio.

        A longer sequence multiplies activation memory per card, so growing
        ``max_seq_length`` N-fold grows ``context_parallel_size`` N-fold to
        keep the per-card sequence slice (and thus activation memory) at the
        source level.  Shrinking the sequence shrinks CP the same way, floored
        at 1.  Returns ``(dims_for_planning, warnings, error_or_None)``; the
        two rewritten fields are pinned like ``--set`` values so no later
        stage may touch them.
        """
        if self.scale_seq_length is None:
            return dims_before, [], None

        new_seq = self.scale_seq_length
        if new_seq < 1:
            return dims_before, [], "--scale-seq-length 必须 >= 1"

        raw = config.get(SEQ_FIELD)
        try:
            old_seq = int(raw)
        except (TypeError, ValueError):
            old_seq = 0
        if old_seq < 1:
            return (
                dims_before,
                [],
                (
                    f"--scale-seq-length 需要源 YAML 声明 {SEQ_FIELD}"
                    f"（当前值：{raw!r}），否则无法计算 CP 的缩放比例"
                ),
            )

        tp, pp, ep, cp, sep = dims_before
        warnings = []
        if new_seq == old_seq:
            cp_new = cp
            warnings.append(
                f"--scale-seq-length {new_seq} 与源 {SEQ_FIELD} 相同，"
                f"序列长度与 CP 均保持不变"
            )
        elif new_seq > old_seq:
            if new_seq % old_seq != 0:
                return (
                    dims_before,
                    [],
                    (
                        f"--scale-seq-length {new_seq} 不是源 {SEQ_FIELD}="
                        f"{old_seq} 的整数倍，CP 无法同比例扩大；请改用 "
                        f"{old_seq} 的整数倍（如 {old_seq * 2} / {old_seq * 4}）"
                    ),
                )
            cp_new = cp * (new_seq // old_seq)
        else:
            if old_seq % new_seq != 0:
                return (
                    dims_before,
                    [],
                    (
                        f"--scale-seq-length {new_seq} 不能整除源 {SEQ_FIELD}="
                        f"{old_seq}，CP 无法同比例缩小；请改用能整除 "
                        f"{old_seq} 的值"
                    ),
                )
            factor = old_seq // new_seq
            # Ceil, not floor: the guarantee is "the per-card slice never
            # exceeds the source slice" (new_seq / cp_new <= old_seq / cp),
            # i.e. cp_new >= cp / factor.  Flooring CP=3 at factor=2 to 1
            # would grow the slice from old_seq/3 to old_seq/2 and can OOM.
            cp_new, remainder = divmod(cp, factor)
            if remainder:
                cp_new += 1
                warnings.append(
                    f"序列长度缩小 {factor} 倍但源 CP={cp} 不能被整除，"
                    f"CP 取上整为 {cp_new}，保证每卡序列片段不超过源配置"
                    f"（片段比按比例缩短的值更短一点，显存只会更省）"
                )
            cp_new = max(cp_new, 1)

        if sep > 1 and cp_new > 1:
            return (
                dims_before,
                [],
                (
                    f"序列长度缩放要求 CP {cp} -> {cp_new}，但源配置 SEP={sep}"
                    f" > 1，框架禁止 sep parallel 与 context parallel 同时使用"
                    f"（C5）；请先在源 YAML 里去掉 SEP 或 CP，再做序列长度缩放"
                ),
            )

        if model_config is not None:
            max_pos = model_config.get("max_position_embeddings")
            if max_pos is not None and int(max_pos) < new_seq:
                warnings.append(
                    f"新序列长度 {new_seq} 超过 model_config.json 的 "
                    f"max_position_embeddings={max_pos}，训练可能因位置编码"
                    f"越界报错；请确认模型支持长度外推，或改小序列长度"
                )

        log.record(
            "yaml",
            self.yaml_writer.apply_config_map(config, {SEQ_FIELD: new_seq}),
            f"--scale-seq-length：序列长度 {old_seq} -> {new_seq}",
        )
        if cp_new != cp:
            direction = "扩大" if new_seq > old_seq else "缩小"
            log.record(
                "yaml",
                self.yaml_writer.apply_config_map(
                    config, {PARALLEL_FIELDS["cp"]: cp_new}
                ),
                f"序列长度{direction} {max(new_seq, old_seq) // min(new_seq, old_seq)} "
                f"倍，context parallel 同比例{direction}：CP {cp} -> {cp_new}，"
                f"保持每卡序列片段不长于源配置，防止长序列 OOM",
            )
        # Pin both fields exactly like --set values: nothing later in the
        # pipeline may rewrite what the user asked for explicitly.
        self.yaml_overrides[SEQ_FIELD] = new_seq
        self.yaml_overrides[PARALLEL_FIELDS["cp"]] = cp_new
        return (tp, pp, ep, cp_new, sep), warnings, None

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
            # the seq scaling wants max_position_embeddings for its warning
            or self.scale_seq_length is not None
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
        * batch settings: ``GBS / (micro_bs * acc) * TP * SEP * PP * CP`` --
          the trainer defines ``dataset_world_size = DP * sharding`` (or
          ``sharding / CP`` when CP > 1), so every other degree multiplies
          back in.

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
                batch_based = dataset_world_size * tp * sep * pp * cp

        if group_based is not None and batch_based is not None:
            if group_based == batch_based:
                return group_based, None, None
            chosen = group_based if dp_declared else batch_based
            warning = (
                f"源卡数的两种推断不一致：按通信组"
                f"（DP={dp}×sharding={sharding}×TP={tp}×SEP={sep}×PP={pp}）"
                f"为 {group_based}，按 batch 字段"
                f"（GBS/(micro×acc)×TP×SEP×PP×CP）为 {batch_based}；"
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
        self, config, log, scale_source, orig_cards, dims_before, dims_after
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
            # The trainer asserts GBS == micro_bs * acc * dataset_world_size,
            # so batch fields must scale with the data-parallel width, not
            # the card count (the two differ whenever EP/PP/CP change).
            orig_units = self._dataset_ways(orig_cards, dims_before)
            new_units = self._dataset_ways(self.target_cards, dims_after)
            if orig_units and new_units:
                unit = "数据并行路数"
            else:
                orig_units, new_units = orig_cards, self.target_cards
                unit = "卡数"
            strategy = BATCH_STRATEGIES[self.options.batch_strategy]
            batch_map, reason, err = strategy(
                gbs, grad_accum, orig_units, new_units, unit=unit
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

        # muon_sharding_optimizer asserts that comm_group_call_opt is only
        # enabled when EP is a multiple of gpus_per_node, moe_sharding > 1
        # and TP == 1, and PaddleFormers' TrainingArguments additionally
        # asserts optim == muon whenever the switch is on; a shrunk EP (or a
        # non-Muon optimizer) easily violates that, so drop the switch
        # instead of shipping a config that dies on these asserts at startup.
        if _yaml_truthy(config.get("sharding_comm_group_call_opt")):
            moe_sharding = self.target_cards // (pp * ep)
            optim = str(config.get("optim") or "").strip().lower()
            group_call_ok = (
                optim == "muon"
                and self.cards_per_node > 1
                and ep % self.cards_per_node == 0
                and moe_sharding > 1
                and tp == 1
            )
            if not group_call_ok:
                log.record(
                    "yaml",
                    self.yaml_writer.apply_config_map(
                        config,
                        {"sharding_comm_group_call_opt": False},
                        protected=self.yaml_overrides,
                    ),
                    f"框架断言 comm_group_call_opt 只能在优化器为 muon、"
                    f"EP 是每节点卡数的整数倍、moe_sharding>1 且 TP=1 时"
                    f"开启；当前 optim={optim or '未设置'}、EP={ep}、每节点 "
                    f"{self.cards_per_node} 卡、moe_sharding={moe_sharding}、"
                    f"TP={tp} 不满足，关闭该优化以免启动即断言失败",
                )

        # C4 was validated above, so both divisions are exact.
        switches = plan_sharding_shrink_switches(
            config,
            self._dataset_ways(orig_cards, dims_before),
            new_sharding // cp,
            overrides=self.yaml_overrides,
            base_ways=self._dense_sharding_ways(dims_before),
        )
        for key, value, reason in switches:
            log.record(
                "yaml",
                self.yaml_writer.apply_config_map(
                    config, {key: value}, protected=self.yaml_overrides
                ),
                reason,
            )
        return None

    def _drop_stale_checkpoint_refs(self, config, log, dims_before, dims_after):
        """Drop checkpoint references once the model structure has shrunk.

        Shrinking EP rescales ``n_routed_experts`` and shrinking PP rescales
        ``num_hidden_layers``, so a checkpoint produced by the full-scale run
        no longer matches the adapted model.  Loading it anyway yields a
        silently corrupted model: the forward pass still runs (loss even looks
        plausible), but gradients blow up to NaN on the very first step.
        Dropping the reference falls back to random init, which is the only
        well-defined starting point for a shrunk smoke test.
        """
        tp_b, pp_b, ep_b, cp_b, sep_b = dims_before
        tp_a, pp_a, ep_a, cp_a, sep_a = dims_after
        if ep_a >= ep_b and pp_a >= pp_b:
            return
        shrunk = []
        if ep_a < ep_b:
            shrunk.append(f"EP {ep_b}->{ep_a}（专家数等比缩减）")
        if pp_a < pp_b:
            shrunk.append(f"PP {pp_b}->{pp_a}（层数等比缩减）")
        reason = (
            f"{'、'.join(shrunk)}后模型结构与原 checkpoint 不再匹配，"
            f"加载会得到权重错乱的模型（首步梯度即 NaN）；"
            f"摘除加载入口，改用随机初始化"
        )
        for key in ("resume_from_checkpoint",):
            if key in config and key not in self.yaml_overrides:
                log.record_removed("yaml", key, config.pop(key), reason)
        if _yaml_truthy(config.get("load_from_hf")) and (
            "load_from_hf" not in self.yaml_overrides
        ):
            log.record(
                "yaml",
                self.yaml_writer.apply_config_map(
                    config,
                    {"load_from_hf": False},
                    protected=self.yaml_overrides,
                ),
                reason,
            )

    @staticmethod
    def _dataset_ways(cards, dims):
        """``dataset_world_size`` for a scale: ``cards / (TP*SEP*PP*CP)``.

        ``None`` when the card count is unknown or does not divide evenly (a
        source YAML whose declared degrees and batch fields disagree).
        """
        if not cards:
            return None
        tp, pp, _ep, cp, sep = dims
        divisor = tp * sep * pp * cp
        if divisor <= 0 or cards % divisor != 0:
            return None
        return cards // divisor

    @staticmethod
    def _dense_sharding_ways(dims):
        """``dense_sharding`` for a set of degrees: ``EP / (TP * SEP)``.

        C3 makes that ratio exact, and it depends on the degrees alone, so it
        survives degrees the target card count could not actually run.  Without
        expert parallel there is no such ratio and ``None`` is returned.
        """
        tp, _pp, ep, _cp, sep = dims
        divisor = tp * sep
        if ep <= 1 or divisor <= 0 or ep % divisor != 0:
            return None
        return ep // divisor

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
