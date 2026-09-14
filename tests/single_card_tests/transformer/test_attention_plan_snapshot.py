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

import dataclasses
import unittest
from types import SimpleNamespace

import paddle

from paddlefleet.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
)
from paddlefleet.transformer.attention_plan import (
    _COERCE_FAILED,
    NormalizationReport,
    _coerce_value,
    _effective_mtp_layers,
    _explicitly_set,
    _fmt_num,
    _qk_norm_of,
    _ratio_kind,
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

    def test_mla_layer_swa_follows_sliding_window(self):
        """-2 MLA 层按真实层号做 SWA 判定，不再钉死 swa=False。

        MultiLatentAttention 继承 Attention.__init__：sliding_window /
        window_attn_skip_freq 命中时 is_swa=True，并换用 swa_* 维度与
        RoPE（multi_latent_attention.py:388-398 / attention.py:281）。
        """
        cfg = _dsv4_config(
            sliding_window=64,
            window_attn_skip_freq=[1] * 6,  # 全部层命中 SWA
            swa_rope_theta=16000.0,
            swa_qk_nope_head_dim=128,
            swa_qk_rope_head_dim=32,
        )
        bundle = resolve_attention_plan(cfg)
        mla_plan = bundle.layers[0]
        self.assertEqual(mla_plan.family, "mla")
        self.assertTrue(mla_plan.swa)
        self.assertEqual(mla_plan.window_size, 64)
        # rope_theta 换成 swa_rope_theta，qk_rope_head_dim 换成 swa_*
        self.assertEqual(mla_plan.position.base, 16000.0)
        self.assertEqual(mla_plan.position.dim, 32)
        self.assertEqual(mla_plan.position.layout, "MLA_EAGER")
        # swa_* 覆盖 hybrid_mla_* 维度
        self.assertEqual(mla_plan.dims["qk_nope_head_dim"], 128)
        self.assertEqual(mla_plan.dims["qk_rope_head_dim"], 32)
        # 未被 swa_* 覆盖的维度仍读 hybrid_mla_*
        self.assertEqual(mla_plan.dims["v_head_dim"], 256)
        self.assertIn(
            "swa layer despite hybrid MLA (sliding_window hit)", mla_plan.notes
        )

        # 混合配置：层 0（-2 MLA）不命中，层 1（window）仍为 swa
        cfg = _dsv4_config(
            sliding_window=64,
            window_attn_skip_freq=[0, 1, 0, 1, 0, 1],
            swa_rope_theta=16000.0,
            swa_qk_nope_head_dim=128,
            swa_qk_rope_head_dim=32,
        )
        bundle = resolve_attention_plan(cfg)
        self.assertFalse(bundle.layers[0].swa)
        self.assertEqual(bundle.layers[0].position.base, 10000.0)
        self.assertEqual(bundle.layers[0].dims["qk_nope_head_dim"], 192)
        self.assertTrue(bundle.layers[1].swa)

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

    def test_gdn_with_sliding_window_not_flagged_swa(self):
        """线性注意力（GDN/KDA）不读 sliding_window，不得误标 swa。

        GatedDeltaNet / KimiDeltaAttention 是独立 FleetLayer 实现，没有
        Attention.__init__ 的 is_swa 逻辑；即便配置了 sliding_window，
        Plan 的 swa/window_size 也必须为 False/None。
        """
        cfg = TransformerConfig(
            num_hidden_layers=3,
            hidden_size=128,
            num_attention_heads=4,
            normalization="RMSNorm",
            sliding_window=[128, 128, 128],
            window_attn_skip_freq=None,  # 全部层都会命中 SWA 判定
        )
        cfg.layer_types = ["gated_delta_net"] * 3
        bundle = resolve_attention_plan(cfg)
        for plan in bundle.layers:
            with self.subTest(index=plan.index):
                self.assertEqual(plan.family, "gdn")
                self.assertFalse(plan.swa)
                self.assertIsNone(plan.window_size)

        # 同一配置下 self_attention 层照常被标记（对照组）
        cfg.layer_types = [
            "gated_delta_net",
            "self_attention",
            "gated_delta_net",
        ]
        cfg.position_embedding_type = "rope"
        bundle = resolve_attention_plan(cfg)
        self.assertFalse(bundle.layers[0].swa)
        self.assertTrue(bundle.layers[1].swa)
        self.assertEqual(bundle.layers[1].window_size, 128)
        self.assertFalse(bundle.layers[2].swa)

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

    def test_unknown_layer_type_records_error_finding(self):
        """未知 layer_types 不得静默丢层：真实构建链路会抛
        'Unknown attention_layer_type'，Plan 必须留下 error finding。"""
        cfg = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=128,
            num_attention_heads=4,
        )
        cfg.layer_types = ["bogus", "self_attention"]
        bundle = resolve_attention_plan(cfg)
        errors = [f for f in bundle.findings if f.rule_id == "V-RES-03"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].severity, "error")
        self.assertIn("bogus", errors[0].message)
        self.assertEqual([f.rule_id for f in bundle.findings[:1]], ["V-RES-03"])
        # JSON 序列化同样携带该 finding（打印/落盘不丢）
        data = bundle.to_json()
        self.assertEqual(data["findings"][0]["rule_id"], "V-RES-03")

    def test_layer_types_length_mismatch_records_error_finding(self):
        cfg = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=128,
            num_attention_heads=4,
        )
        cfg.layer_types = ["self_attention"]
        bundle = resolve_attention_plan(cfg)
        errors = [f for f in bundle.findings if f.rule_id == "V-RES-02"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].severity, "error")
        self.assertIn("1 entries", errors[0].message)

    def test_dsv4_missing_ratio_records_error_finding(self):
        """csa_compress_ratios 缺项同理：真实链路抛错，Plan 记 error。"""
        cfg = _dsv4_config()
        # 构造后截短，模拟外部/直连 config 的不完整 ratios
        cfg.csa_compress_ratios = [-2, 0]
        bundle = resolve_attention_plan(cfg)
        errors = [f for f in bundle.findings if f.rule_id == "V-RES-01"]
        # 层 2/3/4 主层 + 层 5 MTP 共 4 项缺失
        self.assertEqual(len(errors), 4)
        self.assertTrue(all(f.severity == "error" for f in errors))
        # 只解析出前两层，但 findings 明确解释了缺失
        self.assertEqual(len(bundle.layers), 2)

    def test_run_attention_plan_raises_on_resolver_error_when_not_log_only(
        self,
    ):
        cfg = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=128,
            num_attention_heads=4,
        )
        cfg.layer_types = ["bogus"]
        with self.assertRaises(ValueError):
            run_attention_plan(cfg, log_only=False, print_plan=False)


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

    def test_index_alias_on_external_config_creates_canonical(self):
        # 外部 config（SimpleNamespace 等）可能完全没有 canonical 字段：
        # 归一后必须新建该属性，报告与实例保持一致，resolver 才能读到。
        from types import SimpleNamespace

        cfg = SimpleNamespace(index_n_heads=4)
        report = normalize_attention_config(cfg)
        self.assertEqual(
            getattr(cfg, "dsa_index_n_heads", None),
            4,
            "canonical must be created when missing",
        )
        self.assertEqual(
            [c["field"] for c in report.changes], ["index_n_heads"]
        )
        # 多个别名 + 无任何 canonical 字段
        cfg = SimpleNamespace(index_n_heads=4, index_head_dim=32, index_topk=8)
        report = normalize_attention_config(cfg)
        self.assertEqual(cfg.dsa_index_n_heads, 4)
        self.assertEqual(cfg.dsa_index_head_dim, 32)
        self.assertEqual(cfg.dsa_index_topk, 8)
        # apply=False（dry-run）不改实例
        cfg = SimpleNamespace(index_n_heads=4)
        report = normalize_attention_config(cfg, apply=False)
        self.assertFalse(hasattr(cfg, "dsa_index_n_heads"))
        self.assertEqual(
            [c["field"] for c in report.changes], ["index_n_heads"]
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


class TestCoverageGaps(unittest.TestCase):
    """覆盖 Normalizer/Validator/Resolver/格式化中未触达的分支。"""

    # -- Normalizer: _coerce_value 各分支与报告序列化 --

    def test_coerce_int_float_bool_str_list(self):
        # int: 整数浮点可无损强转
        ns = SimpleNamespace(qk_nope_head_dim=16.0)
        report = normalize_attention_config(ns)
        self.assertEqual(ns.qk_nope_head_dim, 16)
        # int: 非整数值只告警不改
        ns = SimpleNamespace(qk_nope_head_dim=16.5)
        report = normalize_attention_config(ns)
        self.assertEqual(ns.qk_nope_head_dim, 16.5)
        self.assertTrue(
            any(w["field"] == "qk_nope_head_dim" for w in report.warnings)
        )
        # bool 字符串
        ns = SimpleNamespace(use_qk_norm="true")
        normalize_attention_config(ns)
        self.assertIs(ns.use_qk_norm, True)
        ns = SimpleNamespace(use_qk_norm="FALSE")
        normalize_attention_config(ns)
        self.assertIs(ns.use_qk_norm, False)
        # 非布尔字符串只告警
        ns = SimpleNamespace(use_qk_norm="bogus")
        report = normalize_attention_config(ns)
        self.assertEqual(ns.use_qk_norm, "bogus")
        self.assertTrue(
            any(w["field"] == "use_qk_norm" for w in report.warnings)
        )
        # bool 非字符串原值（如 int）走 bool()
        ns = SimpleNamespace(use_qk_norm=1)
        normalize_attention_config(ns)
        self.assertIs(ns.use_qk_norm, True)
        # str: 非字符串强转 str
        ns = SimpleNamespace(rope_type=123)
        normalize_attention_config(ns)
        self.assertEqual(ns.rope_type, "123")
        # list: JSON 字符串解析 / tuple 转 list
        ns = SimpleNamespace(layer_types='["a", "b"]')
        normalize_attention_config(ns)
        self.assertEqual(ns.layer_types, ["a", "b"])
        ns = SimpleNamespace(layer_types=(1, 2))
        normalize_attention_config(ns)
        self.assertEqual(ns.layer_types, [1, 2])
        # list: 非 list 的 JSON 只告警
        ns = SimpleNamespace(layer_types='{"a": 1}')
        report = normalize_attention_config(ns)
        self.assertEqual(ns.layer_types, '{"a": 1}')
        self.assertTrue(
            any(w["field"] == "layer_types" for w in report.warnings)
        )

    def test_alias_none_value_and_conflict(self):
        # 别名值为 None：跳过
        ns = SimpleNamespace(index_n_heads=None)
        report = normalize_attention_config(ns)
        self.assertEqual(report.changes, [])
        self.assertEqual(report.warnings, [])
        # 别名与 canonical 冲突：只告警不覆盖
        ns = SimpleNamespace(index_n_heads=4, dsa_index_n_heads=8)
        report = normalize_attention_config(ns)
        self.assertEqual(ns.dsa_index_n_heads, 8)
        self.assertEqual(report.changes, [])
        self.assertTrue(
            any("conflicts" in w["message"] for w in report.warnings)
        )

    def test_normalization_report_and_bundle_json_str(self):
        cfg = _dsv4_config()
        report = normalize_attention_config(cfg)
        data = report.to_json()
        self.assertIn("changes", data)
        self.assertIn("warnings", data)
        bundle = resolve_attention_plan(cfg)
        text = bundle.to_json_str()
        self.assertIsInstance(text, str)
        self.assertIn('"variant": "dsv4_hybrid"', text)

    # -- _explicitly_set: dataclass default / default_factory / 非字段属性 --

    def test_explicitly_set_dataclass_paths(self):
        @dataclasses.dataclass
        class _Cfg:
            x: int = 0
            ys: list = dataclasses.field(default_factory=list)

        cfg = _Cfg()
        self.assertFalse(_explicitly_set(cfg, "missing"))  # 不存在
        self.assertFalse(_explicitly_set(cfg, "x"))  # 等于 default
        cfg.x = 1
        self.assertTrue(_explicitly_set(cfg, "x"))
        self.assertFalse(_explicitly_set(cfg, "ys"))  # 等于 default_factory
        cfg.ys.append(1)
        self.assertTrue(_explicitly_set(cfg, "ys"))
        cfg.extra = 5  # 非字段属性：视为显式设置
        self.assertTrue(_explicitly_set(cfg, "extra"))

    # -- Validator 未触达规则 --

    def test_var02_layer_types_decorative_under_dsv4(self):
        cfg = _dsv4_config()
        cfg.layer_types = ["self_attention"] * 5
        bundle = resolve_attention_plan(cfg)
        self.assertIn("V-VAR-02", [f.rule_id for f in bundle.findings])

    def test_var04_gemma4_in_layer_types_under_dsv4(self):
        cfg = _dsv4_config()
        cfg.layer_types = ["gemma4"] * 5
        bundle = resolve_attention_plan(cfg)
        self.assertIn("V-VAR-04", [f.rule_id for f in bundle.findings])

    def test_var03_mtp_family_disagrees_with_decoder(self):
        cfg = _mla_config(num_nextn_predict_layers=1)
        cfg.layer_types = ["gated_delta_net"]
        bundle = resolve_attention_plan(cfg)
        self.assertIn("V-VAR-03", [f.rule_id for f in bundle.findings])

    def test_mla01_gqa_heads_mismatch(self):
        cfg = _mla_config(num_key_value_heads=2)
        bundle = resolve_attention_plan(cfg)
        self.assertIn("V-MLA-01", [f.rule_id for f in bundle.findings])

    def test_var05_hy_sparse_requires_mla_flag_under_dsv4(self):
        # 无 -2（MLA）层的 dsv4：hy_sparse 的资格检查在逐层改写前运行，
        # 仅靠 csa_compress_ratios 无法满足
        cfg = _dsv4_config(
            multi_latent_attention=False, enable_hy_sparse_attention=True
        )
        cfg.csa_compress_ratios = [0, 0, 4, -1, 128, 4]
        bundle = resolve_attention_plan(cfg)
        self.assertIn("V-VAR-05", [f.rule_id for f in bundle.findings])

    def test_rope02_qk_pos_emb_head_dim_none(self):
        cfg = _dsv4_config()
        cfg.qk_pos_emb_head_dim = None
        bundle = resolve_attention_plan(cfg)
        self.assertIn("V-ROPE-02", [f.rule_id for f in bundle.findings])

    def test_rope04_dsa_indexer_rotary_interleaved_without_indexer(self):
        cfg = _dsv4_config(csa_dense_mode=True)
        cfg.dsa_indexer_rotary_interleaved = True
        bundle = resolve_attention_plan(cfg)
        self.assertIn("V-ROPE-04", [f.rule_id for f in bundle.findings])

    def test_vha01_vha_unread_by_linear_layers(self):
        cfg = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            normalization="RMSNorm",
            use_vha_attention=True,
        )
        cfg.layer_types = ["gated_delta_net"]
        bundle = resolve_attention_plan(cfg)
        self.assertIn("V-VHA-01", [f.rule_id for f in bundle.findings])

    def test_vha02_premix_without_dsv4_layers(self):
        cfg = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            use_vha_premix=True,
        )
        bundle = resolve_attention_plan(cfg)
        self.assertIn("V-VHA-02", [f.rule_id for f in bundle.findings])

    # -- Resolver 未触达分支 --

    def test_unknown_csa_ratio_raises(self):
        with self.assertRaises(ValueError):
            _ratio_kind(1)

    def test_hybrid_mla_attention_modes(self):
        # mqa_dsa: -2 层换 latent MQA + DSA indexer note
        cfg = _dsv4_config(
            hybrid_mla_attention="mqa_dsa", dsa_index_head_dim=128
        )
        bundle = resolve_attention_plan(cfg)
        mla = bundle.layers[0]
        self.assertEqual(mla.core, "mqa_latent")
        self.assertEqual(mla.qkv_layout, "latent")
        self.assertIn("indexer: DSA (hybrid_mla_attention=mqa_dsa)", mla.notes)
        # mqa_full_causal: 无 indexer note
        cfg = _dsv4_config(hybrid_mla_attention="mqa_full_causal")
        bundle = resolve_attention_plan(cfg)
        mla = bundle.layers[0]
        self.assertEqual(mla.core, "mqa_latent")
        self.assertIn(
            "no indexer (hybrid_mla_attention=mqa_full_causal)", mla.notes
        )

    def test_standard_mla_swa_dim_overrides(self):
        # 非 dsv4 的 MLA SWA 路径：swa_* 覆盖 qk_nope/qk_rope 维度
        cfg = _mla_config(
            sliding_window=64,
            window_attn_skip_freq=None,
            swa_qk_nope_head_dim=24,
            swa_qk_rope_head_dim=12,
        )
        bundle = resolve_attention_plan(cfg)
        for plan in bundle.layers:
            self.assertTrue(plan.swa)
            self.assertEqual(plan.dims["qk_nope_head_dim"], 24)
            self.assertEqual(plan.dims["qk_rope_head_dim"], 12)
        # 只设置 nope 覆盖：rope 维持原值
        cfg = _mla_config(
            sliding_window=64,
            window_attn_skip_freq=None,
            swa_qk_nope_head_dim=24,
        )
        bundle = resolve_attention_plan(cfg)
        self.assertEqual(bundle.layers[0].dims["qk_nope_head_dim"], 24)
        self.assertEqual(bundle.layers[0].dims["qk_rope_head_dim"], 8)

    def test_hy_sparse_standard_mla(self):
        cfg = _mla_config(enable_hy_sparse_attention=True)
        bundle = resolve_attention_plan(cfg)
        for plan in bundle.layers:
            self.assertEqual(plan.core, "mqa_latent")
            self.assertEqual(plan.qkv_layout, "latent")
            self.assertIn("hy_sparse", plan.capabilities)
            self.assertIn(
                "hy_sparse swaps the class to MQASelfAttention", plan.notes
            )

    def test_effective_mtp_layers_non_int(self):
        self.assertEqual(
            _effective_mtp_layers(
                SimpleNamespace(num_nextn_predict_layers="2")
            ),
            0,
        )
        self.assertEqual(
            _effective_mtp_layers(
                SimpleNamespace(num_nextn_predict_layers=None)
            ),
            0,
        )

    # -- _qk_norm_of 各家族返回值 --

    def test_qk_norm_of_families(self):
        self.assertEqual(
            _qk_norm_of(
                SimpleNamespace(normalization="LayerNorm"), "gemma4", False
            ),
            "LayerNorm",
        )
        self.assertEqual(
            _qk_norm_of(SimpleNamespace(qk_layernorm=False), "dsv4", True),
            "none",
        )
        self.assertEqual(
            _qk_norm_of(
                SimpleNamespace(
                    normalization="LayerNorm",
                    qk_l2_norm=False,
                    use_qk_norm=True,
                    qk_norm_fusion=False,
                ),
                "self",
                False,
            ),
            "LayerNorm",
        )

    # -- RoPE 展示辅助（yarn 参数 / _fmt_num） --

    def test_rope_repr_with_yarn_params(self):
        cfg = _dsv4_config(
            rotary_scaling_factor=2.0,
            original_max_position_embeddings=8192,
            mscale=1.3,
            mscale_all_dim=1.0,
        )
        bundle = resolve_attention_plan(cfg)
        # 压缩层（ratio>1）为 yarn，携带非默认 yarn 参数
        yarn_pos = bundle.layers[2].position
        self.assertEqual(yarn_pos.type, "yarn")
        text = yarn_pos.render()
        self.assertIn("rotary_scaling_factor=2", text)
        self.assertIn("original_max_position_embeddings=8192", text)
        self.assertIn("mscale=1.3", text)
        self.assertIn("mscale_all_dim=1", text)
        # _fmt_num: 整数值浮点收敛为整数串
        self.assertEqual(_fmt_num(2.0), "2")
        self.assertEqual(_fmt_num(1.3), "1.3")
        self.assertEqual(_fmt_num("x"), "x")

    # -- dump 落盘失败不得阻断（sidecar 约束） --

    def test_dump_failure_does_not_raise(self):
        # 无 save 目录/分布式环境：log_config_to_disk 失败被吞掉，
        # run_attention_plan 正常返回
        cfg = TransformerConfig(
            num_hidden_layers=1, hidden_size=128, num_attention_heads=4
        )
        bundle = run_attention_plan(cfg, print_plan=False)
        self.assertEqual(len(bundle.layers), 1)

    def test_dump_with_config_logger_enabled_does_not_raise(self):
        # config_logger_dir 已设置：dump 走完整路径（单进程下
        # parallel_state.get_all_ranks 失败），任何异常都被吞掉，
        # run_attention_plan 正常返回（sidecar 约束）
        import tempfile

        cfg = TransformerConfig(
            num_hidden_layers=1, hidden_size=128, num_attention_heads=4
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.config_logger_dir = tmpdir
            bundle = run_attention_plan(cfg, print_plan=False)
            self.assertEqual(len(bundle.layers), 1)

    def test_coerce_unknown_spec_type_leaves_as_is(self):
        # 未声明类型的字段：不做强转（哨兵返回，不产生告警）
        report = NormalizationReport()
        result = _coerce_value("whatever", "unknown", 123, report)
        self.assertIs(result, _COERCE_FAILED)
        self.assertEqual(report.warnings, [])


if __name__ == "__main__":
    unittest.main()
