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

"""MoE 配置检查的全部单测: 登记表契约、路由推导、可判定规则、诊断入口。

全部在 CPU 上跑: 设备算力是 resolve_moe_plan 的入参而不是从设备读, 所以一台机器就能
覆盖 Ampere 回退、Blackwell 专属等任何组合。

按 YAML 检查真实配置的示例见 test_moe_config_check_example.py。
"""

import dataclasses
import re
import unittest
from types import SimpleNamespace

from paddlefleet.transformer.moe import moe_config_registry as registry
from paddlefleet.transformer.moe.moe_config_check import (
    MOE_CONFIG_RULES,
    MoEConfigError,
    collect_findings,
    evaluate_rules,
    resolve_moe_plan,
    run_moe_config_check,
)
from paddlefleet.transformer.transformer_config import TransformerConfig

Status = registry.Status
HOPPER = (9, 0)
AMPERE = (8, 0)


# ======================================================================
# 登记表契约: 防止登记表与 config schema 漂移
# ======================================================================

Status = registry.Status

SCHEMA_BACKED = (Status.ACTIVE, Status.DEAD, Status.FOREIGN)
NOT_SCHEMA_BACKED = (Status.UNDECLARED, Status.DERIVED)

# Fields whose name matches the MoE pattern but which the registry deliberately
# leaves out, with the reason. Anything else matching the pattern must be
# registered, so a newly added switch cannot slip in unowned.
UNREGISTERED_BY_DESIGN = {
    "moe_grad_group": "process group plumbing, not a user switch",
    "latent_moe_use_norm": "sub-option of moe_latent_size, shares its owner",
    "moe_config_check": "本套诊断机制自己的开关, 不参与 MoE 计算",
}

_MOE_FIELD_PATTERN = re.compile(
    r"^(moe_|n_routed_experts|n_shared_experts|num_experts_per_tok|scoring_func"
    r"|topk_|n_group|norm_topk_prob|routed_scaling_factor|router_"
    r"|qb_n_bins|actual_vocab_size|routing_map_fusion|using_sonic_moe"
    r"|use_w4a8|use_ue8m0|use_auto_subbatch|auto_subbatch_mode"
    r"|use_rr_deepep|deepep_|hybridep_)"
)


def _schema_fields():
    return {f.name for f in dataclasses.fields(TransformerConfig)}


class TestRegistryMatchesConfigSchema(unittest.TestCase):
    def test_schema_backed_entries_are_real_config_fields(self):
        schema = _schema_fields()
        missing = sorted(
            name
            for name, spec in registry.MOE_FIELD_REGISTRY.items()
            if spec.status in SCHEMA_BACKED and name not in schema
        )
        self.assertEqual(
            missing,
            [],
            "registry describes fields that TransformerConfig does not declare; "
            "either the field was renamed/removed, or its status should be "
            "UNDECLARED (read via getattr) or DERIVED (computed internally)",
        )

    def test_non_schema_entries_are_not_config_fields(self):
        schema = _schema_fields()
        unexpected = sorted(
            name
            for name, spec in registry.MOE_FIELD_REGISTRY.items()
            if spec.status in NOT_SCHEMA_BACKED and name in schema
        )
        self.assertEqual(
            unexpected,
            [],
            "these fields are now declared in TransformerConfig, so they are no "
            "longer getattr-only or purely derived; update their status",
        )

    def test_every_moe_config_field_has_an_owner(self):
        unowned = sorted(
            name
            for name in _schema_fields()
            if _MOE_FIELD_PATTERN.match(name)
            and name not in registry.MOE_FIELD_REGISTRY
            and name not in UNREGISTERED_BY_DESIGN
        )
        self.assertEqual(
            unowned,
            [],
            "new MoE config fields must be added to MOE_FIELD_REGISTRY (or to "
            "UNREGISTERED_BY_DESIGN with a reason), otherwise no diagnostic can "
            "tell a user whether the field applies to their execution path",
        )


class TestRegistryInternalConsistency(unittest.TestCase):
    def test_owner_paths_are_known_submodules(self):
        for name, spec in registry.MOE_FIELD_REGISTRY.items():
            self.assertTrue(spec.owner, f"{name} has no owner")
            for owner in spec.owner:
                self.assertIn(
                    owner,
                    registry.VALID_OWNERS,
                    f"{name} claims unknown submodule {owner!r}",
                )

    def test_fields_that_do_nothing_explain_what_to_do_instead(self):
        for name, spec in registry.MOE_FIELD_REGISTRY.items():
            if spec.status in (Status.DEAD, Status.FOREIGN, Status.DERIVED):
                self.assertTrue(
                    spec.suggestion,
                    f"{name} is {spec.status.value}: a diagnostic about it is "
                    "useless without a suggestion of what to set instead",
                )

    def test_secondary_owned_fields_explain_their_scope(self):
        """A field read by only some of the paths of its submodule is silently
        ignored on the others, so the registry must carry something the
        diagnostic can quote."""
        all_dispatchers = set(registry.DISPATCHER_SUBMODULES)
        for name, spec in registry.MOE_FIELD_REGISTRY.items():
            if spec.status is not Status.ACTIVE:
                continue
            owners = set(spec.owner)
            partial_dispatcher = bool(owners & all_dispatchers) and not (
                all_dispatchers <= owners
            )
            secondary = any(registry.is_secondary_owner(o) for o in spec.owner)
            if not (partial_dispatcher or secondary):
                continue
            self.assertTrue(
                spec.suggestion or spec.requires,
                f"{name} is read by only part of its submodule "
                f"({sorted(owners)}), so it is silently ignored elsewhere; it "
                "needs either a suggestion or explicit requires",
            )

    def test_lookup_helpers(self):
        self.assertIsNone(registry.spec_for("hidden_size"))
        self.assertEqual(
            registry.spec_for("qb_n_bins").owner,
            ("router.quantile_balancing",),
        )

        router_fields = registry.fields_owned_by("router")
        self.assertIn("topk_method", router_fields)
        self.assertIn("qb_n_bins", router_fields)
        self.assertNotIn(
            "qb_n_bins",
            registry.fields_owned_by("router", include_secondary=False),
        )

        self.assertIn(
            "moe_ep_barrier", registry.fields_owned_by("dispatcher.deepep")
        )
        self.assertIn(
            "moe_ep_barrier", registry.fields_owned_by("dispatcher.alltoall")
        )


# ======================================================================
# 路由推导: resolve_moe_plan 复刻的决策
# ======================================================================

HOPPER = (9, 0)
AMPERE = (8, 0)


def _plan_config(**overrides):
    """A config with the real TransformerConfig defaults for the fields the
    resolver reads, so a test only states what it changes."""
    base = {
        "moe_token_dispatcher_type": "deepep",
        "moe_use_fusion_node": True,
        "moe_expert_fusion": False,
        "moe_deep_gemm": True,
        "moe_shared_expert_overlap": False,
        "using_sonic_moe": False,
        "fp8": None,
        "fp8_wgrad": True,
        "use_w4a8": False,
        "topk_method": "greedy",
        "moe_n_hash_layers": 0,
        "n_shared_experts": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _plan_adjusted(plan):
    return {
        adjustment.name: adjustment.effective for adjustment in plan.adjustments
    }


class TestDispatcherSelection(unittest.TestCase):
    def test_no_dispatcher_without_expert_parallelism(self):
        """With EP == 1 the dispatcher branch in MoELayer is never entered, so
        moe_token_dispatcher_type does not select anything."""
        plan = resolve_moe_plan(
            _plan_config(), expert_parallel_size=1, device_capability=HOPPER
        )

        self.assertIsNone(plan.dispatcher)
        self.assertNotIn("dispatcher.deepep", plan.active_submodules)

    def test_deepep_survives_on_hopper(self):
        plan = resolve_moe_plan(
            _plan_config(moe_expert_fusion=True),
            expert_parallel_size=8,
            device_capability=HOPPER,
        )

        self.assertEqual(plan.dispatcher, "deepep")
        self.assertEqual(plan.adjustments, ())

    def test_default_config_contradicts_itself_on_deep_gemm(self):
        """moe_deep_gemm defaults to True and moe_expert_fusion to False, so the
        untouched defaults already resolve DeepGEMM away (REPORT 11.9)."""
        plan = resolve_moe_plan(
            _plan_config(), expert_parallel_size=8, device_capability=HOPPER
        )

        self.assertFalse(plan.moe_deep_gemm)
        self.assertEqual(_plan_adjusted(plan), {"moe_deep_gemm": False})

    def test_pre_hopper_falls_back_to_alltoall(self):
        plan = resolve_moe_plan(
            _plan_config(moe_expert_fusion=True),
            expert_parallel_size=8,
            device_capability=AMPERE,
        )

        self.assertEqual(plan.dispatcher, "alltoall")
        self.assertEqual(
            _plan_adjusted(plan),
            {
                "moe_token_dispatcher_type": "alltoall",
                "moe_deep_gemm": False,
                "moe_use_fusion_node": False,
            },
        )

    def test_unknown_capability_skips_the_fallback(self):
        plan = resolve_moe_plan(
            _plan_config(), expert_parallel_size=8, device_capability=None
        )

        self.assertEqual(plan.dispatcher, "deepep")

    def test_missing_hybridep_runtime_is_recorded_not_raised(self):
        plan = resolve_moe_plan(
            _plan_config(moe_token_dispatcher_type="hybridep"),
            expert_parallel_size=8,
            device_capability=HOPPER,
            hybrid_ep_available=False,
        )

        self.assertEqual(plan.dispatcher, "alltoall")
        self.assertIn("moe_token_dispatcher_type", _plan_adjusted(plan))


class TestAllGatherOverwrites(unittest.TestCase):
    """moe_layer.py:855-887 rewrites three switches; a diagnostic can only tell
    the user which of their values were discarded if the plan records them."""

    def test_allgather_forces_three_switches(self):
        plan = resolve_moe_plan(
            _plan_config(
                moe_token_dispatcher_type="allgather",
                using_sonic_moe=True,
                moe_use_fusion_node=False,
                moe_expert_fusion=False,
                moe_deep_gemm=True,
            ),
            expert_parallel_size=8,
            device_capability=HOPPER,
        )

        self.assertEqual(plan.dispatcher, "allgather")
        self.assertTrue(plan.moe_use_fusion_node)
        self.assertTrue(plan.moe_expert_fusion)
        self.assertFalse(plan.moe_deep_gemm)
        self.assertEqual(plan.expert_branch, "sonic")

        overwritten = _plan_adjusted(plan)
        self.assertEqual(overwritten["moe_use_fusion_node"], True)
        self.assertEqual(overwritten["moe_expert_fusion"], True)
        self.assertEqual(overwritten["moe_deep_gemm"], False)

    def test_allgather_records_nothing_when_config_already_agrees(self):
        plan = resolve_moe_plan(
            _plan_config(
                moe_token_dispatcher_type="allgather",
                using_sonic_moe=True,
                moe_use_fusion_node=True,
                moe_expert_fusion=True,
                moe_deep_gemm=False,
            ),
            expert_parallel_size=8,
            device_capability=HOPPER,
        )

        self.assertEqual(plan.adjustments, ())


class TestAllToAllDowngrades(unittest.TestCase):
    def test_alltoall_disables_fusion_node_and_fp8_payload(self):
        plan = resolve_moe_plan(
            _plan_config(
                moe_token_dispatcher_type="alltoall",
                moe_use_fusion_node=True,
                fp8="blockwise",
            ),
            expert_parallel_size=8,
            device_capability=HOPPER,
        )

        self.assertFalse(plan.moe_use_fusion_node)
        self.assertFalse(plan.fp8_dispatch)
        self.assertEqual(plan.precision, "fp8")


class TestExpertAndPrecisionBranches(unittest.TestCase):
    def test_fp8_without_deep_gemm_splits_field_from_reality(self):
        """REPORT 11.10 issue#2: construction unfuses the weights through a local
        variable while moe_expert_fusion keeps saying True."""
        plan = resolve_moe_plan(
            _plan_config(
                fp8="blockwise", moe_expert_fusion=True, moe_deep_gemm=False
            ),
            expert_parallel_size=8,
            device_capability=HOPPER,
        )

        self.assertTrue(plan.moe_expert_fusion)
        self.assertFalse(plan.expert_weights_fused)
        self.assertEqual(plan.expert_branch, "standard")
        self.assertIn(
            "forward 仍会读到 True",
            "".join(adjustment.reason for adjustment in plan.adjustments),
        )

    def test_grouped_expert_when_fusion_and_deep_gemm_agree(self):
        plan = resolve_moe_plan(
            _plan_config(
                fp8="blockwise", moe_expert_fusion=True, moe_deep_gemm=True
            ),
            expert_parallel_size=8,
            device_capability=HOPPER,
        )

        self.assertEqual(plan.expert_branch, "grouped")
        self.assertTrue(plan.expert_weights_fused)

    def test_w4a8_wins_over_fp8_and_kills_fp8_dispatch(self):
        plan = resolve_moe_plan(
            _plan_config(
                fp8="blockwise", use_w4a8=True, moe_expert_fusion=True
            ),
            expert_parallel_size=8,
            device_capability=HOPPER,
        )

        self.assertEqual(plan.precision, "w4a8")
        self.assertFalse(plan.fp8_dispatch)

    def test_fp8_dispatch_bwd_needs_sonic_and_wgrad(self):
        common = {
            "fp8": "blockwise",
            "moe_expert_fusion": True,
            "moe_deep_gemm": False,
        }

        sonic = resolve_moe_plan(
            _plan_config(using_sonic_moe=True, **common),
            expert_parallel_size=8,
            device_capability=HOPPER,
        )
        no_wgrad = resolve_moe_plan(
            _plan_config(using_sonic_moe=True, fp8_wgrad=False, **common),
            expert_parallel_size=8,
            device_capability=HOPPER,
        )

        self.assertTrue(sonic.fp8_dispatch_bwd)
        self.assertFalse(no_wgrad.fp8_dispatch_bwd)


class TestActiveSubmodules(unittest.TestCase):
    def test_active_submodules_are_registry_owners(self):
        """The validator matches these names against registry owners, so an
        unknown name would silently make every field look out of scope."""
        for dispatcher in ("alltoall", "deepep", "allgather"):
            plan = resolve_moe_plan(
                _plan_config(
                    moe_token_dispatcher_type=dispatcher,
                    using_sonic_moe=dispatcher == "allgather",
                    moe_expert_fusion=dispatcher == "allgather",
                    n_shared_experts=2,
                    moe_n_hash_layers=1,
                    topk_method="noaux_tc",
                ),
                expert_parallel_size=8,
                device_capability=HOPPER,
            )
            unknown = plan.active_submodules - registry.VALID_OWNERS
            self.assertEqual(unknown, set(), f"{dispatcher}: unknown owners")

    def test_shared_expert_absent_when_not_configured(self):
        plan = resolve_moe_plan(
            _plan_config(n_shared_experts=None),
            expert_parallel_size=8,
            device_capability=HOPPER,
        )

        self.assertFalse(plan.shared_expert)
        self.assertNotIn("shared_expert", plan.active_submodules)

    def test_router_branch_follows_topk_method(self):
        for topk_method, branch in (
            ("greedy", "greedy"),
            ("group_limited_greedy", "group_limited"),
            ("noaux_tc", "noaux_tc"),
            ("quantile_balancing", "quantile_balancing"),
        ):
            plan = resolve_moe_plan(
                _plan_config(topk_method=topk_method),
                expert_parallel_size=8,
                device_capability=HOPPER,
            )
            self.assertEqual(plan.router_branch, branch)
            self.assertIn(f"router.{branch}", plan.active_submodules)


# ======================================================================
# 可判定规则与诊断入口
# ======================================================================

HOPPER = (9, 0)


def _config(**overrides):
    base = {
        "n_routed_experts": 64,
        "moe_token_dispatcher_type": "deepep",
        "moe_use_fusion_node": True,
        "moe_expert_fusion": True,
        "moe_deep_gemm": True,
        "moe_shared_expert_overlap": False,
        "using_sonic_moe": False,
        "fp8": None,
        "fp8_wgrad": True,
        "use_w4a8": False,
        "use_w4a8_fused_quant": False,
        "use_ue8m0": False,
        "topk_method": "noaux_tc",
        "moe_topk_fusion": False,
        "n_group": 1,
        "moe_n_hash_layers": 0,
        "actual_vocab_size": None,
        "n_shared_experts": 1,
        "moe_shared_expert_gate": False,
        "use_auto_subbatch": False,
        "auto_subbatch_mode": None,
        "moe_routed_expert_use_bias": None,
        "use_bias": False,
        "hidden_act": "silu",
        "expert_model_parallel_size": 8,
        "moe_config_check": "report",
        "user_specified_keys": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _rule_codes(config, keys, capability=HOPPER, ep_size=8):
    plan = resolve_moe_plan(
        config, expert_parallel_size=ep_size, device_capability=capability
    )
    return {
        rule.code for rule, _ in evaluate_rules(config, plan, capability, keys)
    }


class TestRulesOnlyFireForUserConfiguredFields(unittest.TestCase):
    def test_default_only_config_reports_nothing(self):
        """全默认值的配置不该产出任何规则命中——即使默认值组合本身矛盾, 那也不是
        用户的错。"""
        self.assertEqual(_rule_codes(_config(), keys=set()), set())

    def test_same_problem_is_silent_until_the_user_sets_the_field(self):
        config = _config(use_ue8m0=True, fp8=None)

        self.assertEqual(_rule_codes(config, keys=set()), set())
        self.assertIn("B6", _rule_codes(config, keys={"use_ue8m0"}))


class TestDependencyRules(unittest.TestCase):
    def test_b1_w4a8_fused_quant_without_w4a8(self):
        codes = _rule_codes(
            _config(use_w4a8_fused_quant=True, use_w4a8=False),
            keys={"use_w4a8_fused_quant"},
        )
        self.assertIn("B1", codes)

    def test_b2_actual_vocab_size_without_hash_routing(self):
        codes = _rule_codes(
            _config(actual_vocab_size=100000, moe_n_hash_layers=0),
            keys={"actual_vocab_size"},
        )
        self.assertIn("B2", codes)

    def test_b3_hash_routing_without_actual_vocab_size(self):
        codes = _rule_codes(
            _config(moe_n_hash_layers=2, actual_vocab_size=None),
            keys={"moe_n_hash_layers"},
        )
        self.assertIn("B3", codes)

    def test_b4_shared_expert_gate_without_shared_expert(self):
        codes = _rule_codes(
            _config(moe_shared_expert_gate=True, n_shared_experts=None),
            keys={"moe_shared_expert_gate"},
        )
        self.assertIn("B4", codes)

    def test_b5_subbatch_mode_without_master_switch(self):
        codes = _rule_codes(
            _config(auto_subbatch_mode="post_permute", use_auto_subbatch=False),
            keys={"auto_subbatch_mode"},
        )
        self.assertIn("B5", codes)

    def test_dependencies_satisfied_produce_no_finding(self):
        codes = _rule_codes(
            _config(
                moe_n_hash_layers=2,
                actual_vocab_size=100000,
                auto_subbatch_mode="post_permute",
                use_auto_subbatch=True,
                moe_shared_expert_gate=True,
                n_shared_experts=1,
            ),
            keys={
                "moe_n_hash_layers",
                "actual_vocab_size",
                "auto_subbatch_mode",
                "use_auto_subbatch",
                "moe_shared_expert_gate",
            },
        )
        self.assertEqual(codes, set())


class TestConflictRules(unittest.TestCase):
    def test_c1_and_c2_quantile_balancing_conflicts(self):
        codes = _rule_codes(
            _config(
                topk_method="quantile_balancing",
                moe_topk_fusion=True,
                n_group=4,
            ),
            keys={"topk_method", "moe_topk_fusion", "n_group"},
        )
        self.assertIn("C1", codes)
        self.assertIn("C2", codes)

    def test_c3_topk_fusion_without_noaux_tc(self):
        codes = _rule_codes(
            _config(topk_method="greedy", moe_topk_fusion=True),
            keys={"topk_method", "moe_topk_fusion"},
        )
        self.assertIn("C3", codes)

    def test_c4_fp8_loses_the_fusion_node_on_alltoall(self):
        codes = _rule_codes(
            _config(
                fp8="e4m3",
                moe_token_dispatcher_type="alltoall",
                moe_expert_fusion=False,
                moe_deep_gemm=False,
            ),
            keys={"fp8", "moe_token_dispatcher_type"},
        )
        self.assertIn("C4", codes)

    def test_c5_fp8_with_situ_activation(self):
        from paddlefleet.transformer.activations import situ

        codes = _rule_codes(
            _config(fp8="e4m3", hidden_act=situ),
            keys={"fp8", "hidden_act"},
        )
        self.assertIn("C5", codes)

    def test_c6_fp8_deep_gemm_without_expert_fusion(self):
        codes = _rule_codes(
            _config(fp8="e4m3", moe_expert_fusion=False, moe_deep_gemm=True),
            keys={"fp8", "moe_expert_fusion", "moe_deep_gemm"},
        )
        self.assertIn("C6", codes)

    def test_c7_alltoall_rejects_expert_fusion(self):
        codes = _rule_codes(
            _config(
                moe_token_dispatcher_type="alltoall", moe_expert_fusion=True
            ),
            keys={"moe_token_dispatcher_type", "moe_expert_fusion"},
        )
        self.assertIn("C7", codes)

    def test_c7_also_fires_after_a_capability_fallback(self):
        """算力低于 9.0 时 deepep 会被回退成 alltoall, 于是间接触发同一条冲突。
        这正是"配置本身看不出问题、换机型才炸"的那类场景。"""
        codes = _rule_codes(
            _config(moe_token_dispatcher_type="deepep", moe_expert_fusion=True),
            keys={"moe_token_dispatcher_type", "moe_expert_fusion"},
            capability=(8, 0),
        )
        self.assertIn("C7", codes)

    def test_c8_allgather_requires_sonic(self):
        codes = _rule_codes(
            _config(
                moe_token_dispatcher_type="allgather", using_sonic_moe=False
            ),
            keys={"moe_token_dispatcher_type", "using_sonic_moe"},
        )
        self.assertIn("C8", codes)

    def test_c9_grouped_expert_rejects_bias(self):
        codes = _rule_codes(
            _config(moe_routed_expert_use_bias=True, moe_expert_fusion=True),
            keys={"moe_routed_expert_use_bias"},
        )
        self.assertIn("C9", codes)

    def test_c10_sonic_requires_fused_weights(self):
        codes = _rule_codes(
            _config(using_sonic_moe=True, moe_expert_fusion=False),
            keys={"using_sonic_moe", "moe_expert_fusion"},
        )
        self.assertIn("C10", codes)


class TestCapabilityRules(unittest.TestCase):
    def test_e1_ue8m0_needs_blackwell(self):
        config = _config(use_ue8m0=True, fp8="e4m3")

        self.assertIn(
            "E1", _rule_codes(config, {"use_ue8m0"}, capability=(9, 0))
        )
        self.assertEqual(
            _rule_codes(config, {"use_ue8m0"}, capability=(10, 0)) & {"E1"},
            set(),
        )

    def test_e1_is_skipped_when_capability_is_unknown(self):
        codes = _rule_codes(
            _config(use_ue8m0=True, fp8="e4m3"), {"use_ue8m0"}, capability=None
        )
        self.assertNotIn("E1", codes)


class TestRuleTableHygiene(unittest.TestCase):
    def test_codes_are_unique_and_documented(self):
        codes = [rule.code for rule in MOE_CONFIG_RULES]
        self.assertEqual(len(codes), len(set(codes)), "规则编号重复")
        for rule in MOE_CONFIG_RULES:
            self.assertTrue(rule.fields, f"{rule.code} 没有声明涉及的字段")
            self.assertTrue(rule.suggestion, f"{rule.code} 缺少修复建议")
            self.assertTrue(rule.doc_ref, f"{rule.code} 缺少源码依据")


class TestAllProblemsAreReportedAtOnce(unittest.TestCase):
    def test_one_run_lists_every_problem(self):
        """逐个报错会让用户"改一个、跑一次、再报一个", 在排队等卡的训练任务上代价
        很高, 所以一次必须报全。"""
        config = _config(
            topk_method="quantile_balancing",
            moe_topk_fusion=True,
            n_group=4,
            use_ue8m0=True,
            use_w4a8_fused_quant=True,
            moe_shared_expert_gate=True,
            n_shared_experts=None,
        )
        keys = {
            "topk_method",
            "moe_topk_fusion",
            "n_group",
            "use_ue8m0",
            "use_w4a8_fused_quant",
            "moe_shared_expert_gate",
        }
        plan = resolve_moe_plan(
            config, expert_parallel_size=8, device_capability=HOPPER
        )

        findings = collect_findings(
            config, plan, keys, device_capability=HOPPER
        )
        message = str(MoEConfigError(findings, plan))

        for code in ("B1", "B4", "B6", "C1", "C2"):
            self.assertIn(code, message, f"{code} 没有出现在汇总报错里")


class TestSharedExpertOverlapPreconditions(unittest.TestCase):
    """overlap 的四个前置条件是一串 and(moe_layer.py:1358-1362), 任一不满足就静默
    不重叠——既不报错也不打 warning, 属于最典型的"配了没用"。"""

    def _findings(self, keys, ep_size=8, **overrides):
        config = _config(**overrides)
        plan = resolve_moe_plan(
            config, expert_parallel_size=ep_size, device_capability=HOPPER
        )
        return (
            config,
            plan,
            collect_findings(config, plan, keys, device_capability=HOPPER),
        )

    def test_b7_overlap_without_shared_expert(self):
        _, _, findings = self._findings(
            {"moe_shared_expert_overlap", "n_shared_experts"},
            moe_shared_expert_overlap=True,
            n_shared_experts=0,
        )

        self.assertIn("B7", "".join(f.message for f in findings))

    def test_b7_is_not_duplicated_by_the_generic_scope_check(self):
        """规则和作用域判定会指向同一个字段; 只保留更具体的规则那条。"""
        _, _, findings = self._findings(
            {"moe_shared_expert_overlap", "n_shared_experts"},
            moe_shared_expert_overlap=True,
            n_shared_experts=0,
        )

        overlap_findings = [
            f for f in findings if "moe_shared_expert_overlap" in f.field
        ]
        self.assertEqual(len(overlap_findings), 1, overlap_findings)
        self.assertEqual(overlap_findings[0].kind, "依赖缺失")

    def test_b8_overlap_loses_the_fusion_node_on_alltoall(self):
        _, plan, findings = self._findings(
            {"moe_shared_expert_overlap", "moe_token_dispatcher_type"},
            moe_shared_expert_overlap=True,
            n_shared_experts=1,
            moe_token_dispatcher_type="alltoall",
            moe_expert_fusion=False,
            moe_deep_gemm=False,
        )

        self.assertFalse(plan.moe_use_fusion_node)
        self.assertIn("B8", "".join(f.message for f in findings))

    def test_b9_overlap_without_expert_parallelism(self):
        _, _, findings = self._findings(
            {"moe_shared_expert_overlap"},
            ep_size=1,
            moe_shared_expert_overlap=True,
            n_shared_experts=1,
        )

        self.assertIn("B9", "".join(f.message for f in findings))

    def test_all_preconditions_met_is_silent(self):
        _, _, findings = self._findings(
            {"moe_shared_expert_overlap", "n_shared_experts"},
            moe_shared_expert_overlap=True,
            n_shared_experts=1,
        )

        self.assertEqual(findings, ())


class TestBranchSelectorsAreNeverOutOfScope(unittest.TestCase):
    """选择器把自己的分支关掉后, 不能反过来说这个选择器本身无效——那是倒因为果。"""

    def _scope_fields(self, keys, **overrides):
        config = _config(**overrides)
        plan = resolve_moe_plan(
            config, expert_parallel_size=8, device_capability=HOPPER
        )
        return {
            f.field
            for f in collect_findings(
                config, plan, keys, device_capability=HOPPER
            )
            if f.kind == "作用域错误"
        }

    def test_n_shared_experts_zero(self):
        self.assertNotIn(
            "n_shared_experts",
            self._scope_fields({"n_shared_experts"}, n_shared_experts=0),
        )

    def test_use_w4a8_false(self):
        self.assertNotIn(
            "use_w4a8", self._scope_fields({"use_w4a8"}, use_w4a8=False)
        )

    def test_moe_n_hash_layers_zero(self):
        self.assertNotIn(
            "moe_n_hash_layers",
            self._scope_fields({"moe_n_hash_layers"}, moe_n_hash_layers=0),
        )

    def test_non_selector_fields_still_report_scope_errors(self):
        """对照组: qb_n_bins 不是选择器, 在非 QB 路径下必须照常报。"""
        self.assertIn(
            "qb_n_bins",
            self._scope_fields(
                {"qb_n_bins"}, topk_method="noaux_tc", qb_n_bins=512
            ),
        )


class TestRouterValueRules(unittest.TestCase):
    """这些条件原先只作为自由文本挂在 requires 上, 靠"人工确认"; 实际上都能算, 现在
    都写成规则了。"""

    def test_c11_expert_count_not_divisible_by_group_count(self):
        codes = _rule_codes(
            _config(topk_method="noaux_tc", n_routed_experts=64, n_group=6),
            keys={"n_group", "n_routed_experts"},
        )
        self.assertIn("C11", codes)

    def test_c11_is_silent_for_greedy_which_ignores_groups(self):
        codes = _rule_codes(
            _config(topk_method="greedy", n_routed_experts=64, n_group=6),
            keys={"n_group", "n_routed_experts"},
        )
        self.assertNotIn("C11", codes)

    def test_c12_unknown_scoring_func(self):
        codes = _rule_codes(
            _config(scoring_func="swish"), keys={"scoring_func"}
        )
        self.assertIn("C12", codes)

    def test_c13_seq_aux_loss_needs_a_non_negative_scoring_func(self):
        codes = _rule_codes(
            _config(
                moe_router_load_balancing_type="seq_aux_loss",
                scoring_func="tanh",
            ),
            keys={"scoring_func", "moe_router_load_balancing_type"},
        )
        self.assertIn("C13", codes)

    def test_c14_hash_routing_restricts_scoring_func(self):
        codes = _rule_codes(
            _config(
                moe_n_hash_layers=2, actual_vocab_size=1000, scoring_func="relu"
            ),
            keys={"scoring_func", "moe_n_hash_layers"},
        )
        self.assertIn("C14", codes)

    def test_c15_split_feature_routing_needs_sigmoid(self):
        codes = _rule_codes(
            _config(moe_split_feature_routing=True, scoring_func="softmax"),
            keys={"moe_split_feature_routing", "scoring_func"},
        )
        self.assertIn("C15", codes)

    def test_valid_router_values_are_silent(self):
        codes = _rule_codes(
            _config(
                topk_method="noaux_tc",
                n_routed_experts=64,
                n_group=8,
                scoring_func="sigmoid",
                moe_router_load_balancing_type="seq_aux_loss",
                moe_split_feature_routing=True,
            ),
            keys={
                "topk_method",
                "n_group",
                "n_routed_experts",
                "scoring_func",
                "moe_router_load_balancing_type",
                "moe_split_feature_routing",
            },
        )
        self.assertEqual(codes, set())


class TestRunMoEConfigCheck(unittest.TestCase):
    def test_off_mode_does_nothing(self):
        plan, findings = run_moe_config_check(
            _config(moe_config_check="off"), device_capability=HOPPER
        )

        self.assertIsNone(plan)
        self.assertEqual(findings, ())

    def test_dense_model_is_skipped(self):
        plan, findings = run_moe_config_check(
            _config(moe_config_check="strict", n_routed_experts=None),
            device_capability=HOPPER,
        )

        self.assertIsNone(plan)
        self.assertEqual(findings, ())

    def test_report_mode_never_raises(self):
        plan, findings = run_moe_config_check(
            _config(
                moe_config_check="report",
                topk_method="quantile_balancing",
                moe_topk_fusion=True,
                user_specified_keys=("topk_method", "moe_topk_fusion"),
            ),
            device_capability=HOPPER,
        )

        self.assertIsNotNone(plan)
        self.assertTrue(findings)

    def test_strict_mode_raises_with_every_problem(self):
        with self.assertRaises(MoEConfigError) as caught:
            run_moe_config_check(
                _config(
                    moe_config_check="strict",
                    topk_method="quantile_balancing",
                    moe_topk_fusion=True,
                    n_group=4,
                    user_specified_keys=(
                        "topk_method",
                        "moe_topk_fusion",
                        "n_group",
                    ),
                ),
                device_capability=HOPPER,
            )

        message = str(caught.exception)
        self.assertIn("C1", message)
        self.assertIn("C2", message)

    def test_without_user_keys_it_only_resolves_the_plan(self):
        """拿不到"用户显式配了哪些字段"时不做字段级诊断: 上游 PretrainedConfig 会把
        自身默认值灌进 config, 实测约五成 MoE 字段会被误判成用户设置。"""
        plan, findings = run_moe_config_check(
            _config(
                moe_config_check="strict",
                topk_method="quantile_balancing",
                moe_topk_fusion=True,
                user_specified_keys=None,
            ),
            device_capability=HOPPER,
        )

        self.assertIsNotNone(plan)
        self.assertEqual(findings, ())

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            run_moe_config_check(_config(moe_config_check="yes"))


if __name__ == "__main__":
    unittest.main()
