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

"""Attention 治理管道快照测试（总体方案 §三 P0/P1 交付物）。

config → AttentionExecutionPlan 的黄金快照：Resolver 的决策树复刻
``gpt_layer_specs.get_gpt_layer_local_spec`` 的现状优先级，任何构建链路上的
结构漂移都必须先在这里变红，之后才允许改语义（方案 §三 原则 1/2）。

覆盖：
- dsv4_hybrid 全 ratio 组合（-2/-1/0/4/128）+ MTP 层；
- 标准 MLA（含 DSA indexer）、GDN（layer_types）、VHA、SWA 标准路径；
- Validator 只告警模式：A1/A3/A4/A6/B1/B3 的 findings；
- Normalizer：B5 字符串数字强转、index_* 别名归一；
- ``[ATTN-PLAN]`` 打印格式与 ``to_json`` 单一序列化源。
"""

import unittest

import paddle

from paddlefleet.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
)
from paddlefleet.transformer.attention_plan import (
    format_plan_lines,
    normalize_attention_config,
    resolve_attention_plan,
    run_attention_plan,
)
from paddlefleet.transformer.transformer_config import TransformerConfig

# 真实构建链路的 attention 类名 → Plan family（对拍映射）
_LAYER_CLASS_TO_FAMILY = {
    "SelfAttention": "self",
    "SelfAttentionVHA": "self",  # + capabilities 含 vha
    "MLASelfAttention": "mla",
    "MQASelfAttention": "mla",  # hy_sparse（capabilities 含 hy_sparse）
    "GatedDeltaNet": "gdn",
    "KimiDeltaAttention": "kda",
    "DSv4HybridSelfAttention": "dsv4",
    "Gemma4SelfAttention": "gemma4",
}


def _dsv4_config(**overrides):
    """CPU-friendly dsv4_hybrid config（模板取自 test_hca_csa_independent_rope）。"""
    v_head_dim = 32
    qk_pos = 8
    kwargs = {
        "num_hidden_layers": 5,
        "num_nextn_predict_layers": 1,
        "hidden_size": 256,
        "num_attention_heads": 8,
        "params_dtype": paddle.bfloat16,
        "bf16": True,
        "use_bias": False,
        "multi_latent_attention": True,
        "experimental_attention_variant": "dsv4_hybrid",
        "q_lora_rank": 64,
        "kv_lora_rank": v_head_dim - qk_pos,
        "qk_nope_head_dim": v_head_dim - qk_pos,
        "qk_rope_head_dim": qk_pos,
        "qk_pos_emb_head_dim": qk_pos,
        "v_head_dim": v_head_dim,
        "hybrid_mla_q_lora_rank": 1536,
        "hybrid_mla_kv_lora_rank": 512,
        "hybrid_mla_qk_nope_head_dim": 192,
        "hybrid_mla_qk_rope_head_dim": 64,
        "hybrid_mla_v_head_dim": 256,
        "hybrid_mla_num_attention_heads": 64,
        "hybrid_mla_num_key_value_heads": 64,
        "o_groups": 4,
        "o_lora_rank": 32,
        "rope_type": "rope",
        "rotary_base": 10000.0,
        "rotary_percent": 1.0,
        "normalization": "RMSNorm",
        "use_qk_norm": True,
        # -2 / 0(window) / 4(CSA) / -1(full-causal MQA) / 128(HCA)，第 6 项为 MTP
        "csa_compress_ratios": [-2, 0, 4, -1, 128, -2],
        "csa_window_size": 16,
        "csa_compress_rotary_base": "160000.0",
        "dsa_index_n_heads": 4,
        "dsa_index_head_dim": 32,
        "dsa_index_topk": 8,
        "dsa_indexer_loss_coeff": 1.0,
        "dsa_indexer_use_sparse_loss": False,
        "dsa_indexer_rotary_interleaved": False,
        "apply_rope_fusion": False,
        "attention_dropout": 0.0,
        "attention_softmax_in_fp32": True,
        "masked_softmax_fusion": False,
        "softmax_type": "vanilla",
        "csa_indexer_backend": "unfused",
        "csa_sparse_attn_backend": "unfused",
        "tensor_model_parallel_size": 1,
        "context_parallel_size": 1,
        "csa_dense_mode": False,
    }
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def _mla_config(**overrides):
    kwargs = {
        "num_hidden_layers": 2,
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "multi_latent_attention": True,
        "q_lora_rank": 32,
        "kv_lora_rank": 16,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 8,
        "v_head_dim": 16,
        "rope_type": "rope",
    }
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


class TestDsv4HybridSnapshot(unittest.TestCase):
    """dsv4_hybrid：ratio → (family, core, core_detail, swa) 黄金快照。"""

    def test_ratio_to_layer_plan(self):
        bundle = resolve_attention_plan(_dsv4_config())
        expected = [
            # (index, is_mtp, family, core, core_detail, swa, window)
            (0, False, "mla", "dot_product", None, False, None),
            (1, False, "dsv4", "csa", "window", True, 16),
            (2, False, "dsv4", "csa", "csa", False, None),
            (3, False, "dsv4", "csa", "mqa_full_causal", False, None),
            (4, False, "dsv4", "csa", "hca", False, None),
            (5, True, "mla", "dot_product", None, False, None),
        ]
        self.assertEqual(len(bundle.layers), len(expected))
        for plan, (idx, is_mtp, family, core, detail, swa, window) in zip(
            bundle.layers, expected
        ):
            with self.subTest(index=idx):
                self.assertEqual(plan.index, idx)
                self.assertEqual(plan.is_mtp, is_mtp)
                self.assertEqual(plan.family, family)
                self.assertEqual(plan.core, core)
                self.assertEqual(plan.core_detail, detail)
                self.assertEqual(plan.swa, swa)
                self.assertEqual(plan.window_size, window)

    def test_rope_resolution_per_layer(self):
        bundle = resolve_attention_plan(_dsv4_config())
        # -2 (MLA) 层：rope_type="rope"，base 走 rope_theta，dim 走 hybrid_mla_*
        mla_pos = bundle.layers[0].position
        self.assertEqual(mla_pos.type, "rope")
        self.assertEqual(mla_pos.base, 10000.0)
        self.assertEqual(mla_pos.dim, 64)
        self.assertEqual(mla_pos.layout, "MLA_EAGER")
        self.assertEqual(mla_pos.segment_order, "nope_first")
        self.assertEqual(mla_pos.interleave, False)
        # window 层（ratio 0）：plain rope，base 走 rope_theta
        win_pos = bundle.layers[1].position
        self.assertEqual(win_pos.type, "rope")
        self.assertEqual(win_pos.base, 10000.0)
        self.assertEqual(win_pos.layout, "DSV4_CSA_EAGER")
        # B3：interleave 在 DSv4/CSA 路径被硬编码忽略
        self.assertEqual(win_pos.interleave, "ignored")
        # ratio -1（full-causal MQA）不是压缩层：plain rope + rope_theta
        mqa_pos = bundle.layers[3].position
        self.assertEqual(mqa_pos.type, "rope")
        self.assertEqual(mqa_pos.base, 10000.0)
        # 压缩层（ratio > 1）：yarn + csa_compress_rotary_base（含字符串强转）
        for i in (2, 4):
            pos = bundle.layers[i].position
            self.assertEqual(pos.type, "yarn", f"layer {i}")
            self.assertEqual(pos.base, 160000.0, f"layer {i}")
            self.assertEqual(pos.dim, 8, f"layer {i}")

    def test_rope_groups_counts_distinct_base_type_pairs(self):
        bundle = resolve_attention_plan(_dsv4_config())
        # rope(base=10000) 与 yarn(base=160000) 两组
        self.assertEqual(bundle.rope_groups(), 2)
        self.assertEqual(bundle.family_counts(), {"mla": 2, "dsv4": 4})

    def test_to_json_snapshot(self):
        """打印/落盘/快照共用的序列化（方案 §2.1.2 三者同源）。"""
        bundle = resolve_attention_plan(_dsv4_config())
        data = bundle.to_json()
        self.assertEqual(data["variant"], "dsv4_hybrid")
        self.assertEqual(len(data["layers"]), 6)
        layer0 = data["layers"][0]
        self.assertEqual(layer0["family"], "mla")
        self.assertEqual(layer0["position"]["layout"], "MLA_EAGER")
        self.assertEqual(layer0["position"]["dim"], 64)
        # 稳定快照：整份 JSON 可序列化且键集合固定
        keys = set(layer0.keys())
        self.assertEqual(
            keys,
            {
                "index",
                "layer_number",
                "is_mtp",
                "family",
                "core",
                "core_detail",
                "swa",
                "window_size",
                "qkv_layout",
                "position",
                "precision",
                "recompute",
                "notes",
                "dims",
                "capabilities",
                "qk_norm",
            },
        )

    def test_dims_snapshot(self):
        """每层实际生效的维度字段（镜像各家族 __init__ 的读取点）。"""
        bundle = resolve_attention_plan(_dsv4_config())
        # -2 (MLA) 层读 hybrid_mla_*；num_key_value_heads 被钉死为 heads（A6）
        self.assertEqual(
            bundle.layers[0].dims,
            {
                "q_lora_rank": 1536,
                "kv_lora_rank": 512,
                "qk_nope_head_dim": 192,
                "qk_rope_head_dim": 64,
                "v_head_dim": 256,
                "num_attention_heads": 64,
                "num_key_value_heads": 64,
            },
        )
        # CSA 层读 v_head_dim/qk_pos_emb_head_dim/compress_ratio（C3）
        self.assertEqual(
            bundle.layers[2].dims,
            {
                "num_attention_heads": 8,
                "v_head_dim": 32,
                "q_head_dim": 32,
                "qk_pos_emb_head_dim": 8,
                "compress_ratio": 4,
                "window_size": 16,
            },
        )

    def test_dims_plain_mla(self):
        bundle = resolve_attention_plan(_mla_config())
        self.assertEqual(
            bundle.layers[0].dims,
            {
                "q_lora_rank": 32,
                "kv_lora_rank": 16,
                "qk_nope_head_dim": 16,
                "qk_rope_head_dim": 8,
                "v_head_dim": 16,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,  # 钉死，不是 config 的 GQA 值
            },
        )

    def test_dims_vha_derived_postmix_rank(self):
        cfg = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=128,
            num_attention_heads=8,
            use_vha_attention=True,
            vha_q_lora_rank=16,
        )
        bundle = resolve_attention_plan(cfg)
        dims = bundle.layers[0].dims
        self.assertEqual(dims["vha_q_lora_rank"], 16)
        # C2：postmix 缺省静默推导 num_attention_heads // 4
        self.assertEqual(dims["vha_postmix_rank"], 2)

    def test_qk_norm_resolution(self):
        """C4 观测面：各家族 qk_norm 实际类型（gpt_layer_specs 选路镜像）。"""
        # self + use_qk_norm + RMSNorm + qk_norm_fusion → triton-rms
        cfg = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            use_qk_norm=True,
            qk_norm_fusion=True,
            normalization="RMSNorm",
        )
        self.assertEqual(
            resolve_attention_plan(cfg).layers[0].qk_norm, "triton-rms"
        )
        # self + qk_l2_norm → L2（优先于 use_qk_norm）。注意 qk_l2_norm
        # 未在 TransformerConfig 声明（C4：getattr 兜底读取），只能事后挂上，
        # 与外部 config 携带该字段的情形一致。
        cfg = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            use_qk_norm=True,
            normalization="RMSNorm",
        )
        cfg.qk_l2_norm = True
        self.assertEqual(resolve_attention_plan(cfg).layers[0].qk_norm, "L2")
        # self 默认 → none
        cfg = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=128,
            num_attention_heads=4,
        )
        self.assertEqual(resolve_attention_plan(cfg).layers[0].qk_norm, "none")
        # mla：use_qk_norm → RMSNorm；dsv4 CSA 层：qk_layernorm 缺省 True
        self.assertEqual(
            resolve_attention_plan(
                _mla_config(use_qk_norm=True, normalization="RMSNorm")
            )
            .layers[0]
            .qk_norm,
            "RMSNorm",
        )
        bundle = resolve_attention_plan(
            _dsv4_config()
        )  # RMSNorm + qk_layernorm 默认
        self.assertEqual(bundle.layers[2].qk_norm, "RMSNorm")  # CSA 层
        # gdn：无 qk norm
        cfg = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            normalization="RMSNorm",
        )
        cfg.layer_types = ["gated_delta_net"]
        self.assertEqual(resolve_attention_plan(cfg).layers[0].qk_norm, "none")

    def test_dims_swa_override(self):
        cfg = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            sliding_window=[128, 128],
            head_dim=32,
            swa_head_dim=64,
            swa_num_attention_heads=8,
        )
        cfg.position_embedding_type = "rope"
        bundle = resolve_attention_plan(cfg)
        # 层 0/1 都是 swa（window_attn_skip_freq 缺省 None → 全部 SWA）
        self.assertEqual(bundle.layers[0].dims["head_dim"], 64)
        self.assertEqual(bundle.layers[0].dims["num_attention_heads"], 8)
        # swa_v_head_dim 未显式设置 → 回落 v_head_dim（= head_dim）
        self.assertEqual(bundle.layers[0].dims["v_head_dim"], 32)


class TestStandardPathsSnapshot(unittest.TestCase):
    """非 dsv4 路径：MLA/GDN/VHA/SWA。"""

    def test_plain_mla_with_dsa_indexer(self):
        cfg = _mla_config(
            dsa_index_n_heads=2, dsa_index_head_dim=32, dsa_index_topk=8
        )
        bundle = resolve_attention_plan(cfg)
        for plan in bundle.layers:
            self.assertEqual(plan.family, "mla")
            self.assertEqual(plan.core, "dsa")
            self.assertEqual(plan.qkv_layout, "latent")
            self.assertEqual(plan.position.layout, "MLA_EAGER")
            self.assertEqual(plan.position.dim, 8)
        self.assertIn("indexer: DSA", bundle.layers[0].notes)

    def test_gdn_via_layer_types(self):
        cfg = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            normalization="RMSNorm",
        )
        cfg.layer_types = ["gated_delta_net", "self_attention"]
        bundle = resolve_attention_plan(cfg)
        self.assertEqual(bundle.layers[0].family, "gdn")
        self.assertEqual(bundle.layers[0].core, "linear")
        self.assertEqual(bundle.layers[0].qkv_layout, "linear")
        self.assertEqual(bundle.layers[1].family, "self")
        self.assertEqual(bundle.layers[1].core, "dot_product")

    def test_vha_is_a_capability_not_a_family(self):
        # VHA 是 self 的投影变体（纵向能力），不是独立 family：
        # family=self + qkv_layout=shared_kv + capabilities 含 vha
        cfg = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            use_vha_attention=True,
            vha_q_lora_rank=16,
            vha_postmix_rank=4,
        )
        bundle = resolve_attention_plan(cfg)
        for plan in bundle.layers:
            self.assertEqual(plan.family, "self")
            self.assertEqual(plan.qkv_layout, "shared_kv")
            self.assertIn("vha", plan.capabilities)
            # dims 仍按 VHA 读取点组装
            self.assertIn("vha_q_lora_rank", plan.dims)

    def test_gated_attention_is_a_capability(self):
        cfg = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            gated_attention=True,
        )
        bundle = resolve_attention_plan(cfg)
        self.assertIn("gated_attention", bundle.layers[0].capabilities)
        # GDN 不支持门控能力，不得误标
        cfg = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            gated_attention=True,
            normalization="RMSNorm",
        )
        cfg.layer_types = ["gated_delta_net"]
        bundle = resolve_attention_plan(cfg)
        self.assertNotIn("gated_attention", bundle.layers[0].capabilities)

    def test_swa_standard_path(self):
        cfg = TransformerConfig(
            num_hidden_layers=4,
            hidden_size=128,
            num_attention_heads=4,
            sliding_window=[128, 128],
            window_attn_skip_freq=2,
            head_dim=32,
        )
        cfg.position_embedding_type = "rope"
        bundle = resolve_attention_plan(cfg)
        swa_flags = [p.swa for p in bundle.layers]
        self.assertEqual(swa_flags, [False, True, False, True])
        self.assertEqual(bundle.layers[1].window_size, 128)
        self.assertEqual(bundle.layers[1].position.dim, 32)
        self.assertEqual(bundle.layers[1].position.layout, "STANDARD")
        self.assertEqual(bundle.layers[1].position.segment_order, "rope_first")

    def test_gqa_layout(self):
        cfg = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
        )
        bundle = resolve_attention_plan(cfg)
        self.assertEqual(bundle.layers[0].qkv_layout, "gqa")


class TestValidatorShadowMode(unittest.TestCase):
    """A/B 系列 findings（只告警模式：只收集，不 raise）。"""

    def test_var01_mla_flag_shadowed_by_dsv4(self):
        bundle = resolve_attention_plan(_dsv4_config())
        ids = [f.rule_id for f in bundle.findings]
        self.assertIn("V-VAR-01", ids)
        self.assertIn("V-ROPE-01", ids)  # 模板显式设置了 qk_rope_head_dim

    def test_b3_rotary_interleaved_ignored_on_dsv4(self):
        cfg = _dsv4_config(rotary_interleaved=True)
        bundle = resolve_attention_plan(cfg)
        ids = [f.rule_id for f in bundle.findings]
        self.assertIn("V-ROPE-03", ids)
        # position 列仍显示 interleave=ignored（路径行为）
        self.assertEqual(bundle.layers[1].position.interleave, "ignored")

    def test_vha01_vha_ignored_on_gdn(self):
        # gdn/kda/gemma4 是真正无人读 use_vha_attention 的 family
        cfg = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            use_vha_attention=True,
            normalization="RMSNorm",
        )
        cfg.layer_types = ["gated_delta_net"]
        bundle = resolve_attention_plan(cfg)
        ids = [f.rule_id for f in bundle.findings]
        self.assertIn("V-VHA-01", ids)

    def test_vha01_postmix_is_valid_on_mla(self):
        # 修正后的语义：MLA 复用该开关做 postmix（multi_latent_attention:667），
        # 是合法生效，不得报 V-VHA-01（旧规则在此场景误报）
        cfg = _mla_config(use_vha_attention=True, vha_postmix_rank=4)
        bundle = resolve_attention_plan(cfg)
        ids = [f.rule_id for f in bundle.findings]
        self.assertNotIn("V-VHA-01", ids)
        self.assertIn("vha_postmix", bundle.layers[0].capabilities)

    def test_vha02_premix_requires_dsv4(self):
        cfg = _mla_config(use_vha_attention=True, use_vha_premix=True)
        bundle = resolve_attention_plan(cfg)
        ids = [f.rule_id for f in bundle.findings]
        self.assertIn("V-VHA-02", ids)
        # dsv4 下合法：CSA 层带 vha_premix + vha_postmix，-2 层只有 postmix
        bundle = resolve_attention_plan(
            _dsv4_config(
                use_vha_attention=True,
                use_vha_premix=True,
            )
        )
        ids = [f.rule_id for f in bundle.findings]
        self.assertNotIn("V-VHA-02", ids)
        csa_layer = bundle.layers[2]  # ratio 4 = CSA
        mla_layer = bundle.layers[0]  # ratio -2 = MLA
        self.assertIn("vha_premix", csa_layer.capabilities)
        self.assertIn("vha_postmix", csa_layer.capabilities)
        self.assertIn("vha_postmix", mla_layer.capabilities)
        self.assertNotIn("vha_premix", mla_layer.capabilities)

    def test_mla01_gqa_late_error(self):
        cfg = _mla_config(num_key_value_heads=2)
        bundle = resolve_attention_plan(cfg)
        ids = [f.rule_id for f in bundle.findings]
        self.assertIn("V-MLA-01", ids)

    def test_var03_mtp_family_mismatch(self):
        cfg = _mla_config(num_nextn_predict_layers=1)
        cfg.layer_types = ["gated_delta_net", "gated_delta_net"]
        bundle = resolve_attention_plan(cfg)
        ids = [f.rule_id for f in bundle.findings]
        self.assertIn("V-VAR-03", ids)
        mtp = [p for p in bundle.layers if p.is_mtp]
        self.assertEqual(len(mtp), 1)
        self.assertEqual(mtp[0].family, "mla")  # 现状：MTP 只看 MLA flag

    def test_log_only_false_raises_with_remediation(self):
        cfg = _mla_config(num_key_value_heads=2)
        with self.assertRaisesRegex(ValueError, "V-MLA-01"):
            run_attention_plan(cfg, log_only=False, print_plan=False)
        # 只告警模式（默认）不 raise
        bundle = run_attention_plan(cfg, print_plan=False)
        self.assertTrue(bundle.log_only)

    def test_clean_config_has_no_findings(self):
        cfg = _mla_config()
        bundle = resolve_attention_plan(cfg)
        self.assertEqual(bundle.findings, [])


class TestNormalizer(unittest.TestCase):
    """① 无损归一：B5 字符串强转、index_* 别名。"""

    def test_b5_numeric_string_coerced(self):
        cfg = _dsv4_config()
        self.assertIsInstance(cfg.csa_compress_rotary_base, str)
        report = normalize_attention_config(cfg)
        self.assertEqual(cfg.csa_compress_rotary_base, 160000.0)
        self.assertEqual(len(report.changes), 1)
        self.assertEqual(report.changes[0]["field"], "csa_compress_rotary_base")
        self.assertEqual(report.changes[0]["old"], "160000.0")

    def test_index_alias_normalized(self):
        # 外部/直连 config 携带旧名 index_n_heads（transform_rules 只在
        # from_config 路径改名），Normalizer 兜底归一到 dsa_index_n_heads。
        cfg = _mla_config()
        cfg.__dict__["index_n_heads"] = 4
        report = normalize_attention_config(cfg)
        self.assertEqual(cfg.dsa_index_n_heads, 4)
        self.assertTrue(
            any(c["field"] == "index_n_heads" for c in report.changes)
        )

    def test_non_numeric_string_only_warns(self):
        cfg = _dsv4_config()
        cfg.csa_compress_rotary_base = "not-a-number"
        report = normalize_attention_config(cfg, apply=True)
        self.assertEqual(cfg.csa_compress_rotary_base, "not-a-number")
        self.assertTrue(
            any(
                w["field"] == "csa_compress_rotary_base"
                for w in report.warnings
            )
        )


class TestParityWithSpecChain(unittest.TestCase):
    """对拍护栏：Resolver（sidecar 镜像）必须与真实构建链路逐层同构。

    attention_plan 的决策树是 ``gpt_layer_specs`` 的人肉镜像——上游改了
    优先级/分支（如新增 family、调整 dsv4 ratio 语义）而没同步 Resolver
    时，本测试变红。这弥补了纯快照测试的盲区：快照只锁"Resolver 自己
    不变"，对拍锁"Resolver 与代码一致"。
    """

    @staticmethod
    def _layer_type_of(config, plan):
        """复刻 gpt_builders/get_gpt_decoder_layers_spec 传参给 spec 的方式：
        非逐层路径由 multi_latent_attention flag 决定，逐层路径取
        layer_types[index]；MTP 层只看 flag（A4 现状）。"""
        if plan.is_mtp:
            return None  # 走 multi_latent_attention 参数
        layer_types = getattr(config, "layer_types", None)
        if layer_types is None:
            return None
        return layer_types[plan.index]

    def _spec_family(self, config, plan):
        """用与 Resolver 相同的输入跑真实构建链路，取 attention 类。"""
        spec = get_gpt_layer_local_spec(
            config,
            layer_number=plan.layer_number,
            attention_layer_type=self._layer_type_of(config, plan)
            or "self_attention",
            multi_latent_attention=getattr(
                config, "multi_latent_attention", False
            ),
            normalization=getattr(config, "normalization", None),
            is_mtp_layer=plan.is_mtp,
        )
        layer_cls = spec.sublayers_spec.self_attn.layer.__name__
        return _LAYER_CLASS_TO_FAMILY[layer_cls], layer_cls

    def _assert_parity(self, config):
        bundle = resolve_attention_plan(config)
        self.assertTrue(bundle.layers, "resolver produced no layers")
        for plan in bundle.layers:
            with self.subTest(index=plan.index, is_mtp=plan.is_mtp):
                family, layer_cls = self._spec_family(config, plan)
                self.assertEqual(
                    plan.family,
                    family,
                    f"layer {plan.index}: resolver says {plan.family!r} but the "
                    f"spec chain builds {layer_cls} ({family!r}) — attention_plan "
                    "is out of sync with gpt_layer_specs",
                )

    def test_parity_dsv4_hybrid(self):
        # dsv4：spec 链路内部做 ratio 重写，Resolver 镜像同一优先级
        self._assert_parity(_dsv4_config())

    def test_parity_plain_mla(self):
        self._assert_parity(_mla_config())

    def test_parity_gdn_layer_types(self):
        cfg = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            normalization="RMSNorm",
        )
        cfg.layer_types = ["gated_delta_net", "self_attention"]
        self._assert_parity(cfg)

    def test_parity_vha_capability(self):
        # VHA 是 self 的投影变体：spec 链路建 SelfAttentionVHA（family=self）
        cfg = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            use_vha_attention=True,
            vha_q_lora_rank=16,
            vha_postmix_rank=4,
        )
        bundle = resolve_attention_plan(cfg)
        for plan in bundle.layers:
            family, layer_cls = self._spec_family(cfg, plan)
            self.assertEqual(family, "self")
            self.assertEqual(layer_cls, "SelfAttentionVHA")
            self.assertEqual(plan.family, "self")
            self.assertIn("vha", plan.capabilities)


class TestPlanPrinting(unittest.TestCase):
    """④ [ATTN-PLAN] 摘要表格式（§2.1.2）。"""

    def test_format_lines(self):
        bundle = run_attention_plan(_dsv4_config(), print_plan=False)
        lines = format_plan_lines(bundle)
        self.assertTrue(all(line.startswith("[ATTN-PLAN]") for line in lines))
        header = lines[0]
        self.assertIn("6 layers resolved", header)
        self.assertIn("variant=dsv4_hybrid", header)
        self.assertIn("[warn-only]", header)
        # 表头 + 每层一行 + note 行 + W 行（V-VAR-01/V-ROPE-01 各两行）
        layer_rows = [l for l in lines if "  dsv4  " in l or "  mla  " in l]
        self.assertEqual(len(layer_rows), 6)
        warn_lines = [l for l in lines if " W  V-" in l]
        self.assertEqual(len(warn_lines), len(bundle.findings))
        # RoPE 完整展示：position 列含 base/dim/layout/seg/interleave
        win_row = layer_rows[1]
        self.assertIn("layout=DSV4_CSA_EAGER", win_row)
        self.assertIn("interleave=ignored", win_row)
        self.assertIn("swa(16)", win_row)

    def test_pipeline_prints_and_returns_bundle(self):
        # print_plan=True 走 stdout（rank0 单进程）；不 raise 即通过
        bundle = run_attention_plan(_mla_config())
        self.assertEqual(len(bundle.layers), 2)


if __name__ == "__main__":
    unittest.main()
