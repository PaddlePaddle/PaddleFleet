#!/usr/bin/env python3
"""
计算模型模块 - 预测 PaddleFormers 训练的计算时间

核心功能:
1. 分层计算时间建模 (Attention, MLP, MoE)
2. 基于 FLOPs 和硬件效率的预测
3. 流水线气泡时间预测
4. 支持 Recompute 开销估算
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from enum import Enum

from ..config import (
    ModelConfig,
    ParallelConfig,
    TrainingConfig,
    HardwareConfig,
    RecomputeGranularity,
    TRANSFORMER_DENSE_LAYER_KIND,
    TRANSFORMER_MOE_LAYER_KIND,
    INPUT_EMBEDDING_LAYER_KIND,
    OUTPUT_HEAD_LAYER_KIND,
)
from ..utils.stage_layout import (
    build_balanced_partition_counts,
    has_custom_stage_layer_counts,
    partition_counts_to_ranges,
    resolve_chunk_ranges,
    resolve_stage_chunk_ranges,
    resolve_stage_layer_indices,
)
from .pipeline_schedule import simulate_1f1b_makespan


class LayerType(Enum):
    """层类型"""
    ATTENTION = "attention"
    DENSE_MLP = "dense_mlp"
    MOE_ROUTER = "moe_router"
    MOE_EXPERT = "moe_expert"
    LAYERNORM = "layernorm"
    EMBEDDING = "embedding"


@dataclass
class LayerProfile:
    """层计算 Profile（保留 flops_per_token 供 MFU 计算使用）"""
    layer_type: LayerType
    flops_per_token: int = 0  # 每 token 的 FLOPs


class _BaseComputeModel:
    """
    计算模型

    算子级时间分解：每个 GEMM 查校准曲线获取动态效率，
    每个 memory-bound 操作用显存带宽求解，不使用任何固定效率因子。
    """
    LAYER_PROFILE_CACHE_VERSION = 6
    RECOMPUTE_UNIT_OVERHEAD_US = 70.0
    BACKWARD_ATTENTION_BASE = 1.95
    BACKWARD_ATTENTION_SHORT_SEQ_COEF = 0.10
    BACKWARD_DENSE_MLP_BASE = 1.80
    BACKWARD_DENSE_MLP_SHORT_SEQ_COEF = 0.25
    BACKWARD_DENSE_MLP_BATCH_LOG2_COEF = 0.05
    BACKWARD_MOE_BASE = 1.55
    BACKWARD_MOE_SHORT_SEQ_COEF = 0.45
    BACKWARD_MOE_BATCH_LOG2_COEF = 0.08
    BACKWARD_MOE_SHORT_SEQ_BATCH_COEF = 0.70
    BACKWARD_LAYERNORM_RATIO = 1.80
    BACKWARD_TP_LOG2_BONUS = 0.03
    BACKWARD_ATTENTION_PP_LOG2_DISCOUNT = 0.40
    BACKWARD_DENSE_MLP_PP_LOG2_DISCOUNT = 0.45
    BACKWARD_MOE_PP_LOG2_DISCOUNT = 0.60
    BACKWARD_ATTENTION_MIN_RATIO = 1.35
    BACKWARD_DENSE_MLP_MIN_RATIO = 1.35
    BACKWARD_MOE_MIN_RATIO = 1.25
    
    def __init__(self, model_config: ModelConfig,
                 hardware_config: HardwareConfig,
                 training_config: TrainingConfig):
        self.model = model_config
        self.hardware = hardware_config
        self.training = training_config
        self._layer_runtime_cache: Dict[str, Dict[str, Any]] = {}
        self._layer_profile_cache_dir = (
            Path(__file__).resolve().parents[1] / "layer_model_cache"
        )
        
        # 初始化层 Profile（保留 flops_per_token 供外部 MFU 计算使用）
        self.layer_profiles: Dict[LayerType, LayerProfile] = {}
        self._init_layer_profiles()

    # ──────────────────────────────────────────────
    # 基础原语：GEMM 时间 & memory-bound 时间
    # 所有效率均由硬件参数动态决定，无固定常数
    # ──────────────────────────────────────────────

    def _gemm_time_ms(self, m: int, n: int, k: int) -> float:
        """
        单个 GEMM (C = A×B, A:[M,K] B:[K,N]) 的计算时间 (ms)。
        效率来自校准曲线/Roofline，随 (M,N,K) 动态变化。
        """
        m, n, k = max(1, m), max(1, n), max(1, k)
        flops = 2.0 * m * n * k
        tflops = self.hardware.gpu.get_tflops_for_gemm(
            self.training.dtype, m, n, k,
            dtype_bytes=self.training.dtype_bytes,
        )
        return flops / (max(1e-6, tflops) * 1e12) * 1000.0

    def _membound_time_ms(self, total_bytes: float) -> float:
        """
        显存带宽受限操作的时间 (ms)。
        直接使用校准得到的显存带宽，不加额外折扣。
        """
        bw = self.hardware.gpu.memory_bandwidth_gbps * 1e9  # bytes/s
        if bw <= 0:
            return 0.0
        return total_bytes / bw * 1000.0

    # ──────────────────────────────────────────────
    # 层 Profile 初始化（保留 flops_per_token 供 MFU）
    # ──────────────────────────────────────────────

    def _init_layer_profiles(self):
        """初始化层 FLOPs Profile（仅用于 MFU 计算，不用于时间预测）。"""
        h = self.model.hidden_size
        ffn = self.model.intermediate_size
        moe_ffn = self.model.moe_intermediate_size
        num_heads = self.model.num_attention_heads
        kv_heads = self.model.num_key_value_heads
        head_dim = self.model.head_dim
        num_experts = self.model.num_experts
        
        kv_size = kv_heads * head_dim
        attention_flops = (
            h * h + h * kv_size + h * kv_size +
            num_heads * head_dim + num_heads * 5 + num_heads * head_dim +
            h * h
        )
        self.layer_profiles[LayerType.ATTENTION] = LayerProfile(
            layer_type=LayerType.ATTENTION, flops_per_token=attention_flops)
        
        mlp_flops = h * ffn + h * ffn + ffn * 10 + ffn + ffn * h
        self.layer_profiles[LayerType.DENSE_MLP] = LayerProfile(
            layer_type=LayerType.DENSE_MLP, flops_per_token=mlp_flops)
        
        router_flops = (h * num_experts +
                        num_experts * int(math.log2(num_experts + 1)) * 2)
        self.layer_profiles[LayerType.MOE_ROUTER] = LayerProfile(
            layer_type=LayerType.MOE_ROUTER, flops_per_token=router_flops)
        
        expert_flops = h * moe_ffn + h * moe_ffn + moe_ffn * 10 + moe_ffn + moe_ffn * h
        self.layer_profiles[LayerType.MOE_EXPERT] = LayerProfile(
            layer_type=LayerType.MOE_EXPERT, flops_per_token=expert_flops)
        
        self.layer_profiles[LayerType.LAYERNORM] = LayerProfile(
            layer_type=LayerType.LAYERNORM, flops_per_token=h * 7)

    def _is_moe_layer(self, layer_idx: int) -> bool:
        """根据模型结构判断某一层是否为 MoE 层。"""
        return self.model.is_moe_layer(layer_idx)

    def _transformer_layer_kind(self, layer_idx: int) -> str:
        """返回某个 Transformer block 的层类型。"""
        return self.model.transformer_layer_kind(layer_idx)

    def _stage_layer_ranges(self, pp_degree: int) -> list[tuple[int, int]]:
        """按 PP 拆分每个 stage 负责的层范围。"""
        pp = max(1, int(pp_degree))
        total_layers = max(0, int(self.model.num_hidden_layers))
        return partition_counts_to_ranges(
            build_balanced_partition_counts(total_layers, pp)
        )

    def _use_runtime_stage_slots(self, parallel: ParallelConfig) -> bool:
        """
        是否按 runtime slot 布局切 stage。

        当前仅在多机且无 VPP 时启用，避免影响单机路径。
        """
        if int(self.hardware.num_nodes) <= 1:
            return False
        if self._get_virtual_pipeline_size(parallel) > 1:
            return False
        return (
            int(getattr(self.training, "num_empty_layers_add_in_head", 0)) > 0 or
            int(getattr(self.training, "num_empty_layers_add_in_tail", 0)) > 0
        )

    def _stage_slot_ranges(self, parallel: ParallelConfig) -> list[tuple[int, int]]:
        """按 runtime slot 均匀切分 PP stage。"""
        pp = max(1, int(parallel.pp))
        total_layers = max(0, int(self.model.num_hidden_layers))
        head_empty = max(
            0, int(getattr(self.training, "num_empty_layers_add_in_head", 0))
        )
        tail_empty = max(
            0, int(getattr(self.training, "num_empty_layers_add_in_tail", 0))
        )
        total_slots = max(total_layers, total_layers + head_empty + tail_empty)
        return [
            (
                total_slots * stage_idx // pp,
                total_slots * (stage_idx + 1) // pp,
            )
            for stage_idx in range(pp)
        ]

    def _get_virtual_pipeline_size(self, parallel: ParallelConfig) -> int:
        """
        获取 VPP 大小。

        当前公开配置里没有显式 vpp 字段，这里仅在对象上存在 `vpp`
        属性时启用，默认按 1 处理。
        """
        raw_value = getattr(parallel, "vpp", 1)
        try:
            return max(1, int(raw_value))
        except Exception:
            return 1

    def _chunk_layer_ranges(self, parallel: ParallelConfig) -> list[tuple[int, int]]:
        """
        按 PP × VPP 切分 Transformer layers。

        chunk_size 语义与 PaddleFormers 一致：总层数被划分成
        `pp * vpp` 个本地 chunk。
        """
        return resolve_chunk_ranges(int(self.model.num_hidden_layers), parallel)

    def _stage_chunk_ranges(self, parallel: ParallelConfig) -> list[list[tuple[int, int]]]:
        """
        获取每个物理 PP stage 拥有的 chunk 列表。

        当 VPP>1 时，同一物理 stage 会持有多个 interleaved virtual chunks。
        """
        return resolve_stage_chunk_ranges(int(self.model.num_hidden_layers), parallel)

    def _stage_layer_indices(self,
                             parallel: ParallelConfig,
                             stage_id: int) -> list[int]:
        """获取某个物理 PP stage 实际拥有的全局层索引。"""
        if has_custom_stage_layer_counts(parallel):
            if self._use_runtime_stage_slots(parallel):
                raise ValueError(
                    "自定义 stage_layer_counts 暂不支持与 runtime empty layers 同时使用"
                )
            return resolve_stage_layer_indices(
                int(self.model.num_hidden_layers), parallel, stage_id
            )

        if self._use_runtime_stage_slots(parallel):
            head_empty = max(
                0, int(getattr(self.training, "num_empty_layers_add_in_head", 0))
            )
            total_layers = max(0, int(self.model.num_hidden_layers))
            start_slot, end_slot = self._stage_slot_ranges(parallel)[stage_id]
            layer_indices = []
            for slot_idx in range(start_slot, end_slot):
                layer_idx = slot_idx - head_empty
                if 0 <= layer_idx < total_layers:
                    layer_indices.append(layer_idx)
            return layer_indices

        return resolve_stage_layer_indices(
            int(self.model.num_hidden_layers), parallel, stage_id
        )

    def _make_layer_runtime_cache_key(self,
                                      batch_size: int,
                                      seq_len: int,
                                      tp_degree: int,
                                      ep_degree: int) -> tuple[str, Dict[str, Any]]:
        """构造按模型/硬件/运行时维度区分的层级建模缓存 key。"""
        payload = {
            "version": self.LAYER_PROFILE_CACHE_VERSION,
            "runtime": {
                "batch_size": int(batch_size),
                "seq_len": int(seq_len),
                "tp_degree": int(tp_degree),
                "ep_degree": int(ep_degree),
                "dtype": self.training.dtype,
                "dtype_bytes": int(self.training.dtype_bytes),
                "num_nodes": int(self.hardware.num_nodes),
                "num_empty_layers_add_in_head": int(
                    getattr(self.training, "num_empty_layers_add_in_head", 0)
                ),
                "num_empty_layers_add_in_tail": int(
                    getattr(self.training, "num_empty_layers_add_in_tail", 0)
                ),
                "attn_implementation": str(
                    getattr(self.training, "attn_implementation", "")
                ),
                "use_qk_norm": bool(
                    getattr(self.training, "use_qk_norm", False)
                ),
                "moe_token_dispatcher_type": str(
                    getattr(self.training, "moe_token_dispatcher_type", "")
                ),
                "moe_grouped_gemm": bool(
                    getattr(self.training, "moe_grouped_gemm", False)
                ),
                "moe_router_fusion": bool(
                    getattr(self.training, "moe_router_fusion", False)
                ),
                "moe_expert_fusion": bool(
                    getattr(self.training, "moe_expert_fusion", False)
                ),
                "moe_shared_expert_overlap": bool(
                    getattr(self.training, "moe_shared_expert_overlap", False)
                ),
                "moe_ep_barrier": bool(
                    getattr(self.training, "moe_ep_barrier", False)
                ),
                "variable_seq_lengths": bool(
                    getattr(self.training, "variable_seq_lengths", False)
                ),
                "enable_dynamic_shape": bool(
                    getattr(self.training, "enable_dynamic_shape", False)
                ),
            },
            "model": {
                "num_hidden_layers": int(self.model.num_hidden_layers),
                "hidden_size": int(self.model.hidden_size),
                "intermediate_size": int(self.model.intermediate_size),
                "num_attention_heads": int(self.model.num_attention_heads),
                "num_key_value_heads": int(self.model.num_key_value_heads),
                "head_dim": int(self.model.head_dim),
                "num_experts": int(self.model.num_experts),
                "num_shared_experts": int(self.model.num_shared_experts),
                "num_experts_per_tok": int(self.model.num_experts_per_tok),
                "moe_intermediate_size": int(self.model.moe_intermediate_size),
                "shared_expert_intermediate_size": int(
                    self.model.effective_shared_expert_intermediate_size
                ),
                "decoder_sparse_step": int(self.model.decoder_sparse_step),
                "mlp_only_layers": list(self.model.mlp_only_layers),
            },
            "hardware": {
                "gpu_name": self.hardware.gpu.name,
                "bf16_tflops": float(self.hardware.gpu.bf16_tflops),
                "memory_bandwidth_gbps": float(self.hardware.gpu.memory_bandwidth_gbps),
                "bf16_curve": self.hardware.gpu.bf16_curve,
                "bf16_gemm_samples": self.hardware.gpu.bf16_gemm_samples,
            },
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        cache_key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]
        return cache_key, payload

    def _build_layer_runtime_profile(self,
                                     batch_size: int,
                                     seq_len: int,
                                     tp_degree: int,
                                     ep_degree: int,
                                     cache_key: str,
                                     metadata: Dict[str, Any]) -> Dict[str, Any]:
        """构建按层展开的时间建模结果。"""
        dense = self.estimate_dense_layer_time_breakdown(batch_size, seq_len, tp_degree)
        moe = self.estimate_moe_layer_time_breakdown(
            batch_size, seq_len, tp_degree, ep_degree
        )
        dense_bwd = self.estimate_dense_layer_backward_time_breakdown(
            batch_size, seq_len, tp_degree
        )
        moe_bwd = self.estimate_moe_layer_backward_time_breakdown(
            batch_size, seq_len, tp_degree, ep_degree
        )

        templates = {
            TRANSFORMER_DENSE_LAYER_KIND: {
                "layer_kind": TRANSFORMER_DENSE_LAYER_KIND,
                "layer_type": "dense",
                "layer_role": "transformer",
                "forward_full": float(dense["full"]),
                "attention_core": float(dense["attention_core"]),
                "attention_projection": float(
                    dense.get(
                        "attention_projection",
                        dense["attention_total"] - dense["attention_core"],
                    )
                ),
                "layernorm": float(dense["layernorm"]),
                "mlp": float(dense.get("mlp", 0.0)),
                "moe_total": 0.0,
                "moe_router": 0.0,
                "moe_dispatch": 0.0,
                "moe_gate_up": 0.0,
                "moe_act": 0.0,
                "moe_down": 0.0,
                "backward_full": float(dense_bwd["full"]),
                "backward_attention_total": float(dense_bwd["attention_total"]),
                "backward_attention_projection": float(
                    dense_bwd["attention_projection"]
                ),
                "backward_attention_core": float(dense_bwd["attention_core"]),
                "backward_layernorm": float(dense_bwd["layernorm"]),
                "backward_mlp": float(dense_bwd.get("mlp", 0.0)),
                "backward_moe_total": 0.0,
                "backward_moe_router": 0.0,
                "backward_moe_dispatch": 0.0,
                "backward_moe_expert": 0.0,
            },
            TRANSFORMER_MOE_LAYER_KIND: {
                "layer_kind": TRANSFORMER_MOE_LAYER_KIND,
                "layer_type": "moe",
                "layer_role": "transformer",
                "forward_full": float(moe["full"]),
                "attention_core": float(moe["attention_core"]),
                "attention_projection": float(
                    moe.get(
                        "attention_projection",
                        moe["attention_total"] - moe["attention_core"],
                    )
                ),
                "layernorm": float(moe["layernorm"]),
                "mlp": 0.0,
                "moe_total": float(moe.get("moe_total", 0.0)),
                "moe_router": float(moe.get("moe_router", 0.0)),
                "moe_dispatch": float(moe.get("moe_dispatch", 0.0)),
                "moe_gate_up": float(moe.get("moe_gate_up", 0.0)),
                "moe_act": float(moe.get("moe_act", 0.0)),
                "moe_down": float(moe.get("moe_down", 0.0)),
                "backward_full": float(moe_bwd["full"]),
                "backward_attention_total": float(moe_bwd["attention_total"]),
                "backward_attention_projection": float(
                    moe_bwd["attention_projection"]
                ),
                "backward_attention_core": float(moe_bwd["attention_core"]),
                "backward_layernorm": float(moe_bwd["layernorm"]),
                "backward_mlp": 0.0,
                "backward_moe_total": float(moe_bwd.get("moe_total", 0.0)),
                "backward_moe_router": float(moe_bwd.get("moe_router", 0.0)),
                "backward_moe_dispatch": float(moe_bwd.get("moe_dispatch", 0.0)),
                "backward_moe_expert": float(moe_bwd.get("moe_expert", 0.0)),
            },
            INPUT_EMBEDDING_LAYER_KIND: {
                "layer_kind": INPUT_EMBEDDING_LAYER_KIND,
                "layer_type": "embedding",
                "layer_role": "input_embedding",
                "forward_full": float(
                    self._estimate_embedding_forward_time(batch_size, seq_len)
                ),
                "backward_full": float(
                    self._estimate_embedding_backward_time(batch_size, seq_len)
                ),
            },
            OUTPUT_HEAD_LAYER_KIND: {
                "layer_kind": OUTPUT_HEAD_LAYER_KIND,
                "layer_type": "output_head",
                "layer_role": "output_head",
                "forward_full": float(
                    self._estimate_output_head_forward_time(
                        batch_size, seq_len, tp_degree
                    )
                ),
                "backward_full": float(
                    self._estimate_output_head_backward_time(
                        batch_size, seq_len, tp_degree
                    )
                ),
            },
        }

        layers = []
        for layer_idx in range(self.model.num_hidden_layers):
            layer_kind = self._transformer_layer_kind(layer_idx)
            layers.append({
                "layer_idx": int(layer_idx),
                "layer_kind": layer_kind,
                "template_key": layer_kind,
                "layer_type": "moe" if layer_kind == TRANSFORMER_MOE_LAYER_KIND else "dense",
            })

        return {
            "cache_key": cache_key,
            "metadata": metadata,
            "templates": templates,
            "layers": layers,
        }

    def _get_runtime_layer_template(self,
                                    layer_payload: Dict[str, Any],
                                    layer_entry: Dict[str, Any]) -> Dict[str, Any]:
        """按 layer entry 获取对应的层类型模板。"""
        template_key = str(layer_entry.get("template_key", layer_entry.get("layer_kind", "")))
        return layer_payload["templates"][template_key]

    def _get_layer_runtime_profile(self,
                                   batch_size: int,
                                   seq_len: int,
                                   tp_degree: int,
                                   ep_degree: int) -> Dict[str, Any]:
        """
        获取按层展开的时间建模结果。

        首次命中某个 (model, hardware, runtime) 组合时会落盘到
        仓库根目录下的 `layer_model_cache/`，后续直接复用。
        """
        cache_key, metadata = self._make_layer_runtime_cache_key(
            batch_size, seq_len, tp_degree, ep_degree
        )
        if cache_key in self._layer_runtime_cache:
            return self._layer_runtime_cache[cache_key]

        cache_path = self._layer_profile_cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict) and isinstance(payload.get("layers"), list):
                self._layer_runtime_cache[cache_key] = payload
                return payload

        payload = self._build_layer_runtime_profile(
            batch_size, seq_len, tp_degree, ep_degree, cache_key, metadata
        )

        self._layer_profile_cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        tmp_path.replace(cache_path)

        self._layer_runtime_cache[cache_key] = payload
        return payload

    def _normalize_recompute_method(self,
                                    recompute_granularity: RecomputeGranularity,
                                    recompute_method: Optional[str]) -> str:
        """统一 recompute method 的默认语义。"""
        method = str(recompute_method or "").strip().lower()
        if recompute_granularity == RecomputeGranularity.FULL:
            return method
        if recompute_granularity == RecomputeGranularity.SELECTIVE:
            if method in ("block", "first_n"):
                return method
            return ""
        return ""

    def _validate_recompute_config(self,
                                   recompute_granularity: RecomputeGranularity,
                                   recompute_method: Optional[str],
                                   recompute_num_layers: Optional[int],
                                   recompute_modules: Optional[tuple[str, ...]] = None) -> None:
        """校验 recompute 配置语义。"""
        if recompute_granularity == RecomputeGranularity.NONE:
            return

        method = self._normalize_recompute_method(
            recompute_granularity, recompute_method
        )

        if recompute_granularity == RecomputeGranularity.FULL:
            if method not in ("uniform", "block", "first_n"):
                raise ValueError(
                    "when recompute_granularity=full, recompute_method must be one of "
                    "'uniform', 'block' and 'first_n'"
                )
            if recompute_num_layers is None:
                raise ValueError(
                    "when recompute_granularity=full, recompute_num_layers must not be None"
                )
            if int(recompute_num_layers) <= 0:
                raise ValueError(
                    "when recompute_granularity=full, recompute_num_layers must be > 0"
                )
            return

        if recompute_granularity == RecomputeGranularity.SELECTIVE:
            if method not in ("", "block", "first_n"):
                raise ValueError(
                    "when recompute_granularity=selective, recompute_method must be one of "
                    "'block', 'first_n' or None"
                )
            if method in ("block", "first_n"):
                if recompute_num_layers is None:
                    raise ValueError(
                        "when recompute_granularity=selective and recompute_method is "
                        "'block' or 'first_n', recompute_num_layers must not be None"
                    )
                if int(recompute_num_layers) <= 0:
                    raise ValueError(
                        "when recompute_granularity=selective and recompute_method is "
                        "'block' or 'first_n', recompute_num_layers must be > 0"
                    )
            return

        raise ValueError("recompute_granularity must be one of none, selective and full")

    def _select_recomputed_layer_indices(self,
                                         stage_start: int,
                                         stage_end: int,
                                         recompute_granularity: RecomputeGranularity,
                                         recompute_method: str,
                                         recompute_num_layers: Optional[int]) -> list[int]:
        """
        选择当前 stage 内会被重算的层索引。

        语义：
        - full + uniform: 当前 stage 的全部 Transformer 层都会重算；
          `recompute_num_layers` 表示每个均匀划分单元包含多少层
        - block: 每个流水线阶段内前 N 层参与重算
        - first_n: 全模型前 N 层参与重算
        """
        layer_count = max(0, stage_end - stage_start)
        if layer_count <= 0 or recompute_granularity == RecomputeGranularity.NONE:
            return []

        if recompute_granularity == RecomputeGranularity.FULL and recompute_method == "uniform":
            return list(range(layer_count))

        if recompute_method == "block":
            count = max(0, min(layer_count, int(recompute_num_layers or 0)))
            return list(range(count))

        if recompute_method == "first_n":
            global_count = max(0, int(recompute_num_layers or 0))
            if stage_start >= global_count:
                return []
            selected_end = min(stage_end, global_count)
            return list(range(max(0, selected_end - stage_start)))

        return list(range(layer_count))

    def _select_selective_recomputed_layers(self,
                                            stage_chunks: list[tuple[int, int]],
                                            recompute_method: str,
                                            recompute_num_layers: Optional[int]) -> list[int]:
        """
        选择 selective recompute 命中的全局 layer 索引。

        语义：
        - `block`: 每个 PP/VPP chunk 的前 N 层
        - `first_n`: 每个物理 PP stage 的前 N 层
        - `None`: 当前 stage 的全部层
        """
        if not stage_chunks:
            return []

        if recompute_method == "block":
            selected = []
            count = max(0, int(recompute_num_layers or 0))
            for chunk_start, chunk_end in stage_chunks:
                take = min(max(0, chunk_end - chunk_start), count)
                selected.extend(range(chunk_start, chunk_start + take))
            return selected

        if recompute_method == "first_n":
            remaining = max(0, int(recompute_num_layers or 0))
            selected = []
            for chunk_start, chunk_end in stage_chunks:
                if remaining <= 0:
                    break
                chunk_len = max(0, chunk_end - chunk_start)
                take = min(chunk_len, remaining)
                selected.extend(range(chunk_start, chunk_start + take))
                remaining -= take
            return selected

        selected = []
        for chunk_start, chunk_end in stage_chunks:
            selected.extend(range(chunk_start, chunk_end))
        return selected

    def _estimate_full_recompute_unit_count(self,
                                            stage_chunks: list[tuple[int, int]],
                                            recompute_method: str,
                                            recompute_num_layers: Optional[int]) -> int:
        """估算 full recompute 的 checkpoint 单元数。"""
        if not stage_chunks:
            return 0
        if recompute_method == "uniform":
            unit_layers = max(1, int(recompute_num_layers or 1))
            return sum(
                math.ceil(max(0, chunk_end - chunk_start) / unit_layers)
                for chunk_start, chunk_end in stage_chunks
                if chunk_end > chunk_start
            )
        return 1

    def _select_full_recomputed_layers(self,
                                       stage_chunks: list[tuple[int, int]],
                                       recompute_method: str,
                                       recompute_num_layers: Optional[int]) -> list[int]:
        """
        选择 full recompute 命中的全局 layer 索引。

        语义与 memory_model 保持一致：
        - `uniform`: 当前 stage 的全部层
        - `block`: 当前物理 stage 的前 N 层
        - `first_n`: 全模型前 N 层
        """
        if not stage_chunks:
            return []

        if recompute_method == "uniform":
            selected = []
            for chunk_start, chunk_end in stage_chunks:
                selected.extend(range(chunk_start, chunk_end))
            return selected

        if recompute_method == "block":
            remaining = max(0, int(recompute_num_layers or 0))
            selected = []
            for chunk_start, chunk_end in stage_chunks:
                if remaining <= 0:
                    break
                chunk_len = max(0, chunk_end - chunk_start)
                take = min(chunk_len, remaining)
                selected.extend(range(chunk_start, chunk_start + take))
                remaining -= take
            return selected

        if recompute_method == "first_n":
            global_count = max(0, int(recompute_num_layers or 0))
            selected = []
            for chunk_start, chunk_end in stage_chunks:
                if chunk_start >= global_count:
                    continue
                selected_end = min(chunk_end, global_count)
                selected.extend(range(chunk_start, selected_end))
            return selected

        return []

    # ──────────────────────────────────────────────
    # 算子级时间预测：逐个 GEMM + 逐个 memory-bound op
    # ──────────────────────────────────────────────

    def estimate_layer_time(self, layer_type: LayerType,
                           batch_size: int, seq_len: int,
                           tp_degree: int = 1) -> float:
        """
        估算单层计算时间 (ms)。

        不使用任何固定效率因子，而是逐算子分解：
        - 每个 GEMM：查校准曲线获取该 (M,N,K) 下的有效 TFLOPS
        - 每个 element-wise 操作（SiLU, multiply, softmax, LayerNorm 等）：
          用 total_bytes / memory_bandwidth 求解
        所有参数均可从模型结构、训练配置和硬件规格推导。
        """
        tokens = max(1, batch_size * seq_len)
        tp = max(1, tp_degree)
        h = self.model.hidden_size
        db = self.training.dtype_bytes  # bytes per element

        if layer_type == LayerType.ATTENTION:
            return self._estimate_attention_proj_time(tokens, tp, db)
        elif layer_type == LayerType.DENSE_MLP:
            return self._estimate_dense_mlp_time(tokens, tp, db)
        elif layer_type == LayerType.MOE_ROUTER:
            return self._estimate_router_time(tokens, db)
        elif layer_type == LayerType.MOE_EXPERT:
            return self._estimate_expert_mlp_time(tokens, tp, db)
        elif layer_type == LayerType.LAYERNORM:
            return self._estimate_layernorm_time(tokens, db)
        return 0.0

    def _estimate_attention_proj_time(self, tokens: int, tp: int, db: int) -> float:
        """
        Attention 线性投影部分时间 (QKV + O projection)。
        每个 projection 是独立 GEMM，效率由各自的 (M,N,K) 动态决定。
        """
        h = self.model.hidden_size
        kv_size = self.model.num_key_value_heads * self.model.head_dim

        # Q proj: (tokens, h/tp) = (tokens, h) × (h, h/tp)
        q_time = self._gemm_time_ms(tokens, max(1, h // tp), h)
        # K proj: (tokens, kv/tp) = (tokens, h) × (h, kv/tp)
        k_time = self._gemm_time_ms(tokens, max(1, kv_size // tp), h)
        # V proj: same as K
        v_time = self._gemm_time_ms(tokens, max(1, kv_size // tp), h)
        # O proj: (tokens, h) = (tokens, h/tp) × (h/tp, h)
        o_time = self._gemm_time_ms(tokens, h, max(1, h // tp))

        # memory-bound ops 穿插在 GEMM 之间:
        #   bias/residual/pack/rope 等按张量字节量估算。Qwen3 一类模型还包含 q_norm/k_norm，
        #   它们是对 Q/K projection 输出做 head-dim RMSNorm。
        q_out = max(1, h // tp)
        kv_out = max(1, kv_size // tp)
        non_gemm_bytes = (
            tokens * q_out * db * 4.0 +
            tokens * kv_out * db * 6.0 +
            tokens * h * db * 3.0
        )
        if bool(getattr(self.training, "use_qk_norm", False)):
            qk_norm_bytes = tokens * (q_out + kv_out) * db * 5.0
            non_gemm_bytes += qk_norm_bytes
        non_gemm_time = self._membound_time_ms(non_gemm_bytes)

        return q_time + k_time + v_time + o_time + non_gemm_time

    def _estimate_dense_mlp_time(self, tokens: int, tp: int, db: int) -> float:
        """
        Dense MLP (SwiGLU) 时间 = Gate GEMM + Up GEMM + SiLU + Multiply + Down GEMM。
        """
        h = self.model.hidden_size
        ffn_tp = max(1, self.model.intermediate_size // tp)

        # 3 个 GEMM，各自有独立的 (M,N,K)
        gate_time = self._gemm_time_ms(tokens, ffn_tp, h)
        up_time   = self._gemm_time_ms(tokens, ffn_tp, h)
        down_time = self._gemm_time_ms(tokens, h, ffn_tp)

        # SiLU activation: read gate_out, write silu_out → 2 passes
        silu_bytes = tokens * ffn_tp * db * 2.0
        silu_time = self._membound_time_ms(silu_bytes)

        # Element-wise multiply (silu_out × up_out): read 2, write 1 → 3 passes
        mul_bytes = tokens * ffn_tp * db * 3.0
        mul_time = self._membound_time_ms(mul_bytes)

        return gate_time + up_time + down_time + silu_time + mul_time

    def _estimate_expert_mlp_time(self, tokens: int, tp: int, db: int) -> float:
        """
        MoE Expert MLP 时间，结构与 Dense MLP 相同但使用 moe_intermediate_size。
        tokens 参数为该 expert 实际处理的 token 数（通常远小于 batch×seq）。
        """
        h = self.model.hidden_size
        moe_ffn_tp = max(1, self.model.moe_intermediate_size // tp)

        gate_time = self._gemm_time_ms(tokens, moe_ffn_tp, h)
        up_time   = self._gemm_time_ms(tokens, moe_ffn_tp, h)
        down_time = self._gemm_time_ms(tokens, h, moe_ffn_tp)

        silu_bytes = tokens * moe_ffn_tp * db * 2.0
        silu_time = self._membound_time_ms(silu_bytes)

        mul_bytes = tokens * moe_ffn_tp * db * 3.0
        mul_time = self._membound_time_ms(mul_bytes)

        return gate_time + up_time + down_time + silu_time + mul_time

    def _estimate_router_time(self, tokens: int, db: int) -> float:
        """
        MoE Router 时间 = 线性投影 GEMM + TopK 选择。
        """
        h = self.model.hidden_size
        n_exp = self.model.num_experts

        # 线性投影: (tokens, num_experts) = (tokens, h) × (h, num_experts)
        linear_time = self._gemm_time_ms(tokens, n_exp, h)

        # TopK selection: read scores(tokens × n_exp), write indices+values(tokens × topk × 2)
        topk = self.model.num_experts_per_tok
        topk_bytes = (tokens * n_exp + tokens * topk * 2) * db
        topk_time = self._membound_time_ms(topk_bytes)

        return linear_time + topk_time

    def _estimate_layernorm_time(self, tokens: int, db: int) -> float:
        """
        LayerNorm / RMSNorm 时间，纯 memory-bound。
        4 passes: read input → 计算 mean/var → read+normalize → write output
        + 1 pass: scale/bias (read params + read+write)
        """
        h = self.model.hidden_size
        norm_bytes = tokens * h * db * 5.0
        return self._membound_time_ms(norm_bytes)

    def _estimate_linear_backward_time_ms(self,
                                          tokens: int,
                                          input_dim: int,
                                          output_dim: int) -> Dict[str, float]:
        """
        线性层 backward 时间。

        Forward:
          Y = X @ W        , X:[tokens, input_dim], W:[input_dim, output_dim]

        Backward:
          dX = dY @ W^T    -> GEMM(tokens, input_dim, output_dim)
          dW = X^T @ dY    -> GEMM(input_dim, output_dim, tokens)
        """
        input_dim = max(1, int(input_dim))
        output_dim = max(1, int(output_dim))
        dgrad_time = self._gemm_time_ms(tokens, input_dim, output_dim)
        wgrad_time = self._gemm_time_ms(input_dim, output_dim, tokens)
        return {
            "dgrad": dgrad_time,
            "wgrad": wgrad_time,
            "total": dgrad_time + wgrad_time,
        }

    def _estimate_attention_proj_backward_time(self, tokens: int, tp: int, db: int) -> float:
        """
        Attention 线性投影 backward 时间。

        包含:
        - Q/K/V/O 四个投影的 dgrad + wgrad GEMM
        - residual split / RoPE backward / QKV pack-unpack 等轻量 memory-bound 操作
        """
        h = self.model.hidden_size
        q_out = max(1, h // tp)
        kv_out = max(1, (self.model.num_key_value_heads * self.model.head_dim) // tp)

        q_bwd = self._estimate_linear_backward_time_ms(tokens, h, q_out)
        k_bwd = self._estimate_linear_backward_time_ms(tokens, h, kv_out)
        v_bwd = self._estimate_linear_backward_time_ms(tokens, h, kv_out)
        o_bwd = self._estimate_linear_backward_time_ms(tokens, q_out, h)

        residual_bytes = tokens * h * db * 3.0
        rope_bytes = tokens * (q_out + kv_out) * db * 4.0
        qkv_pack_bytes = tokens * (q_out + 2 * kv_out) * db * 2.0
        qk_norm_backward_bytes = 0.0
        if bool(getattr(self.training, "use_qk_norm", False)):
            # backward: 读 x/y/norm 参数并写 dx，近似 8 passes
            qk_norm_backward_bytes = tokens * (q_out + kv_out) * db * 8.0
        non_gemm_time = self._membound_time_ms(
            residual_bytes + rope_bytes + qkv_pack_bytes + qk_norm_backward_bytes
        )

        return (
            q_bwd["total"] +
            k_bwd["total"] +
            v_bwd["total"] +
            o_bwd["total"] +
            non_gemm_time
        )

    def _estimate_attention_core_backward_time(self,
                                               batch_size: int,
                                               seq_len: int,
                                               tp_degree: int = 1) -> float:
        """
        Flash Attention core backward 时间。

        近似分解:
        - dV, dP, dQ, dK 四个主计算项，共约 8 * B * H * S^2 * D FLOPs
        - softmax backward / scale / masking 等轻量项，量级远小于主 GEMM，合并到 IO 中
        """
        num_heads = max(1, self.model.num_attention_heads // tp_degree)
        head_dim = self.model.head_dim
        db = self.training.dtype_bytes

        core_flops = float(batch_size) * num_heads * (
            8.0 * seq_len * seq_len * head_dim +
            6.0 * seq_len * seq_len
        )
        core_tflops = self.hardware.gpu.get_tflops_for_gemm(
            self.training.dtype, seq_len, seq_len, head_dim,
            dtype_bytes=db,
        )
        compute_time_ms = core_flops / (max(1e-6, core_tflops) * 1e12) * 1000.0

        tensor_bytes = float(batch_size) * num_heads * seq_len * head_dim * db
        stats_bytes = float(batch_size) * num_heads * seq_len * 8.0
        io_time_ms = self._membound_time_ms(tensor_bytes * 10.0 + stats_bytes)

        return max(compute_time_ms, io_time_ms)

    def _estimate_dense_mlp_backward_time(self, tokens: int, tp: int, db: int) -> float:
        """
        Dense MLP (SwiGLU) backward 时间。

        包含:
        - down 投影 dgrad + wgrad
        - multiply backward
        - SiLU backward
        - gate / up 两个投影各自的 dgrad + wgrad
        - 两路 hidden grad 累加
        """
        h = self.model.hidden_size
        ffn_tp = max(1, self.model.intermediate_size // tp)

        down_bwd = self._estimate_linear_backward_time_ms(tokens, ffn_tp, h)
        gate_bwd = self._estimate_linear_backward_time_ms(tokens, h, ffn_tp)
        up_bwd = self._estimate_linear_backward_time_ms(tokens, h, ffn_tp)

        mul_backward_bytes = tokens * ffn_tp * db * 5.0
        silu_backward_bytes = tokens * ffn_tp * db * 3.0
        hidden_accum_bytes = tokens * h * db * 3.0
        light_ops_time = self._membound_time_ms(
            mul_backward_bytes + silu_backward_bytes + hidden_accum_bytes
        )

        return (
            down_bwd["total"] +
            gate_bwd["total"] +
            up_bwd["total"] +
            light_ops_time
        )

    def _estimate_expert_mlp_backward_time(self, tokens: int, tp: int, db: int) -> float:
        """
        单 expert MLP backward 时间。
        """
        h = self.model.hidden_size
        moe_ffn_tp = max(1, self.model.moe_intermediate_size // tp)

        down_bwd = self._estimate_linear_backward_time_ms(tokens, moe_ffn_tp, h)
        gate_bwd = self._estimate_linear_backward_time_ms(tokens, h, moe_ffn_tp)
        up_bwd = self._estimate_linear_backward_time_ms(tokens, h, moe_ffn_tp)

        mul_backward_bytes = tokens * moe_ffn_tp * db * 5.0
        silu_backward_bytes = tokens * moe_ffn_tp * db * 3.0
        hidden_accum_bytes = tokens * h * db * 3.0
        light_ops_time = self._membound_time_ms(
            mul_backward_bytes + silu_backward_bytes + hidden_accum_bytes
        )

        return (
            down_bwd["total"] +
            gate_bwd["total"] +
            up_bwd["total"] +
            light_ops_time
        )

    def _estimate_shared_expert_time(self, tokens: int, tp: int, db: int) -> float:
        """估算 shared expert 前向时间。"""
        shared_ffn = max(0, int(self.model.effective_shared_expert_intermediate_size))
        if shared_ffn <= 0:
            return 0.0

        h = self.model.hidden_size
        shared_ffn_tp = max(1, shared_ffn // max(1, tp))

        gate_time = self._gemm_time_ms(tokens, shared_ffn_tp, h)
        up_time = self._gemm_time_ms(tokens, shared_ffn_tp, h)
        down_time = self._gemm_time_ms(tokens, h, shared_ffn_tp)
        silu_time = self._membound_time_ms(tokens * shared_ffn_tp * db * 2.0)
        mul_time = self._membound_time_ms(tokens * shared_ffn_tp * db * 3.0)
        return gate_time + up_time + down_time + silu_time + mul_time

    def _estimate_shared_expert_backward_time(self, tokens: int, tp: int, db: int) -> float:
        """估算 shared expert 反向时间。"""
        shared_ffn = max(0, int(self.model.effective_shared_expert_intermediate_size))
        if shared_ffn <= 0:
            return 0.0

        h = self.model.hidden_size
        shared_ffn_tp = max(1, shared_ffn // max(1, tp))

        down_bwd = self._estimate_linear_backward_time_ms(tokens, shared_ffn_tp, h)
        gate_bwd = self._estimate_linear_backward_time_ms(tokens, h, shared_ffn_tp)
        up_bwd = self._estimate_linear_backward_time_ms(tokens, h, shared_ffn_tp)
        mul_backward_bytes = tokens * shared_ffn_tp * db * 5.0
        silu_backward_bytes = tokens * shared_ffn_tp * db * 3.0
        hidden_accum_bytes = tokens * h * db * 3.0
        light_ops_time = self._membound_time_ms(
            mul_backward_bytes + silu_backward_bytes + hidden_accum_bytes
        )
        return (
            down_bwd["total"] +
            gate_bwd["total"] +
            up_bwd["total"] +
            light_ops_time
        )

    def _apply_grouped_gemm_runtime_penalty(self,
                                            expert_time: float,
                                            tokens_per_expert: float,
                                            experts_per_gpu: int,
                                            backward: bool = False) -> float:
        """
        估算 grouped GEMM 在多机下的小批量/多 expert runtime 损失。
        """
        if int(self.hardware.num_nodes) <= 1:
            return expert_time
        if not bool(getattr(self.training, "moe_grouped_gemm", False)):
            return expert_time
        if experts_per_gpu <= 1:
            return expert_time

        local_tokens = max(1.0, float(tokens_per_expert))
        occupancy_penalty = max(0.0, 1024.0 / local_tokens - 1.0)
        grouping_penalty = 1.0 + 0.15 * math.log2(max(1, experts_per_gpu))
        grouping_penalty += min(0.8, 0.25 * occupancy_penalty)
        launch_ms = experts_per_gpu * (0.12 if backward else 0.08)
        if bool(getattr(self.training, "moe_expert_fusion", False)):
            launch_ms *= 0.6
        return expert_time * grouping_penalty + launch_ms

    def _estimate_router_backward_time(self, tokens: int, db: int) -> float:
        """
        Router backward 时间。

        包含:
        - router 线性层 dgrad + wgrad
        - softmax / topk scatter 的梯度传播
        """
        h = self.model.hidden_size
        n_exp = self.model.num_experts
        topk = self.model.num_experts_per_tok

        linear_bwd = self._estimate_linear_backward_time_ms(tokens, h, n_exp)
        softmax_backward_bytes = tokens * n_exp * db * 4.0
        topk_scatter_bytes = (tokens * n_exp + tokens * topk * 2) * db * 2.0
        light_ops_time = self._membound_time_ms(
            softmax_backward_bytes + topk_scatter_bytes
        )

        return linear_bwd["total"] + light_ops_time

    def _estimate_layernorm_backward_time(self, tokens: int, db: int) -> float:
        """
        LayerNorm / RMSNorm backward 时间。

        近似为 9 次 hidden tensor 读写/规约。
        """
        h = self.model.hidden_size
        norm_bytes = tokens * h * db * 9.0
        return self._membound_time_ms(norm_bytes)

    def _estimate_embedding_forward_time(self, batch_size: int, seq_len: int) -> float:
        """估算输入 embedding lookup 的前向时间。"""
        tokens = max(1, batch_size * seq_len)
        h = self.model.hidden_size
        db = self.training.dtype_bytes
        lookup_bytes = tokens * h * db * 4.0
        return self._membound_time_ms(lookup_bytes)

    def _estimate_embedding_backward_time(self, batch_size: int, seq_len: int) -> float:
        """估算输入 embedding grad / scatter-add 的反向时间。"""
        tokens = max(1, batch_size * seq_len)
        h = self.model.hidden_size
        db = self.training.dtype_bytes
        scatter_bytes = tokens * h * db * 7.0
        return self._membound_time_ms(scatter_bytes)

    def _estimate_output_head_forward_time(self,
                                           batch_size: int,
                                           seq_len: int,
                                           tp_degree: int = 1) -> float:
        """估算 final norm + lm_head + loss 的前向时间。"""
        tokens = max(1, batch_size * seq_len)
        h = self.model.hidden_size
        vocab_tp = max(1, math.ceil(self.model.vocab_size / max(1, tp_degree)))
        db = self.training.dtype_bytes

        final_norm_time = self._estimate_layernorm_time(tokens, db)
        lm_head_time = self._gemm_time_ms(tokens, vocab_tp, h)

        # logits / softmax / cross-entropy 大多以 fp32 临时张量存在。
        logits_fp32_bytes = tokens * vocab_tp * 4.0
        loss_io_time = self._membound_time_ms(logits_fp32_bytes * 6.0)

        return final_norm_time + lm_head_time + loss_io_time

    def _estimate_output_head_backward_time(self,
                                            batch_size: int,
                                            seq_len: int,
                                            tp_degree: int = 1) -> float:
        """估算 final norm + lm_head + loss 的反向时间。"""
        tokens = max(1, batch_size * seq_len)
        h = self.model.hidden_size
        vocab_tp = max(1, math.ceil(self.model.vocab_size / max(1, tp_degree)))
        db = self.training.dtype_bytes

        final_norm_time = self._estimate_layernorm_backward_time(tokens, db)
        lm_head_bwd = self._estimate_linear_backward_time_ms(tokens, h, vocab_tp)

        logits_fp32_bytes = tokens * vocab_tp * 4.0
        loss_backward_time = self._membound_time_ms(logits_fp32_bytes * 8.0)

        return final_norm_time + lm_head_bwd["total"] + loss_backward_time

    def estimate_forward_stage_times(self,
                                     batch_size: int,
                                     seq_len: int,
                                     parallel: ParallelConfig) -> list[float]:
        """估算每个 PP stage 的前向时间。"""
        layer_payload = self._get_layer_runtime_profile(
            batch_size, seq_len, parallel.tp, parallel.ep
        )
        layers = layer_payload["layers"]

        stage_times = []
        stage_count = max(1, int(parallel.pp))
        for stage_id in range(stage_count):
            stage_layer_indices = self._stage_layer_indices(parallel, stage_id)
            stage_time = sum(
                self._get_runtime_layer_template(layer_payload, layers[layer_idx])["forward_full"]
                for layer_idx in stage_layer_indices
            )
            if stage_id == 0:
                stage_time += layer_payload["templates"][INPUT_EMBEDDING_LAYER_KIND][
                    "forward_full"
                ]
            if stage_id == stage_count - 1:
                stage_time += layer_payload["templates"][OUTPUT_HEAD_LAYER_KIND][
                    "forward_full"
                ]
            stage_times.append(stage_time)
        return stage_times

    def estimate_backward_stage_times(self,
                                      batch_size: int,
                                      seq_len: int,
                                      parallel: ParallelConfig) -> list[float]:
        """估算每个 PP stage 的反向时间。

        直接使用按子算子展开的 backward 模板，而不再对 forward 结果乘经验比值。
        这样能显式保留不同模型结构（如 q_norm/k_norm、不同 MLP 宽度、MoE dispatch）
        对 backward 时间的影响，泛化性也更好。
        """
        layer_payload = self._get_layer_runtime_profile(
            batch_size, seq_len, parallel.tp, parallel.ep
        )
        layers = layer_payload["layers"]

        stage_times = []
        stage_count = max(1, int(parallel.pp))
        for stage_id in range(stage_count):
            stage_layer_indices = self._stage_layer_indices(parallel, stage_id)
            stage_time = sum(
                self._get_runtime_layer_template(layer_payload, layers[layer_idx])["backward_full"]
                for layer_idx in stage_layer_indices
            )
            if stage_id == 0:
                stage_time += layer_payload["templates"][INPUT_EMBEDDING_LAYER_KIND][
                    "backward_full"
                ]
            if stage_id == stage_count - 1:
                stage_time += layer_payload["templates"][OUTPUT_HEAD_LAYER_KIND][
                    "backward_full"
                ]
            stage_times.append(stage_time)
        return stage_times
    
    def estimate_attention_time(self, batch_size: int, seq_len: int,
                                tp_degree: int = 1) -> float:
        """估算 Attention 计算时间 (ms)"""
        return self.estimate_attention_time_breakdown(
            batch_size, seq_len, tp_degree
        )["total"]

    def estimate_attention_time_breakdown(self, batch_size: int, seq_len: int,
                                          tp_degree: int = 1) -> Dict[str, float]:
        """
        估算 Attention 时间分解。

        `flash_attn` / `core_attn` selective recompute 只会重算注意力核心 kernel，
        不会重跑 QKV/O projection，因此这里把 attention 拆成：
        - `projection`: QKV/O projection 等线性部分（GEMM + bias/residual）
        - `core_attention`: Flash Attention 融合 kernel（QK^T + softmax + score*V）

        两部分均通过算子级分解动态计算效率，无固定常数。
        """
        # projection 部分：QKV/O GEMM + memory-bound bias/residual
        base_time = self.estimate_layer_time(
            LayerType.ATTENTION, batch_size, seq_len, tp_degree
        )

        # ── Flash Attention core ──
        # Flash Attention 将 QK^T + softmax + score*V 融合成单个 kernel，
        # 在 SRAM 中分块执行，避免将 O(N²) 的 attention matrix 写入 HBM。
        #
        # 时间取决于两部分，取 max（Roofline 思想）：
        # 1. 计算量: QK^T + score*V = 2 × batch × heads × seq² × head_dim FLOPs
        # 2. IO量: Flash Attention 的 HBM 读写 = O(N²d²/M_sram) + Q/K/V/O 各读写一次
        #    其中 M_sram ≈ GPU SRAM 大小 (H100: ~228KB per SM)
        #    对于大多数实际场景 (seq ≤ 32K)，IO 约为 O(batch × heads × seq × head_dim)
        #    即 Q/K/V/O 各做一遍线性扫描。

        num_heads = max(1, self.model.num_attention_heads // tp_degree)
        head_dim = self.model.head_dim
        db = self.training.dtype_bytes

        # 计算量: QK^T + score*V
        # = 2 * batch * num_heads * (2 * seq * seq * head_dim)
        core_flops = float(batch_size) * num_heads * 4.0 * seq_len * seq_len * head_dim
        # 用 per-head 的等效 GEMM (seq, seq, head_dim) 查校准曲线
        core_tflops = self.hardware.gpu.get_tflops_for_gemm(
            self.training.dtype, seq_len, seq_len, head_dim,
            dtype_bytes=db,
        )
        compute_time_ms = core_flops / (max(1e-6, core_tflops) * 1e12) * 1000.0

        # IO 量: Flash Attention 对 Q/K/V 各读一次，O 写一次 → 4 passes
        # 每个 tensor: batch × num_heads × seq × head_dim × dtype_bytes
        flash_io_bytes = (4.0 * batch_size * num_heads * seq_len * head_dim * db)
        io_time_ms = self._membound_time_ms(flash_io_bytes)

        # Flash Attention 时间 = max(计算, IO)，实际中通常 compute-bound
        core_attention_time_ms = max(compute_time_ms, io_time_ms)

        return {
            "projection": base_time,
            "core_attention": core_attention_time_ms,
            "flash_attn": core_attention_time_ms,
            "total": base_time + core_attention_time_ms,
        }

    def estimate_attention_backward_time_breakdown(self,
                                                   batch_size: int,
                                                   seq_len: int,
                                                   tp_degree: int = 1) -> Dict[str, float]:
        """
        估算 Attention backward 时间分解。
        """
        tokens = max(1, batch_size * seq_len)
        tp = max(1, tp_degree)
        db = self.training.dtype_bytes

        projection_time = self._estimate_attention_proj_backward_time(tokens, tp, db)
        core_attention_time = self._estimate_attention_core_backward_time(
            batch_size, seq_len, tp_degree
        )

        return {
            "projection": projection_time,
            "core_attention": core_attention_time,
            "total": projection_time + core_attention_time,
        }

    def estimate_moe_dispatch_overhead(self, batch_size: int, seq_len: int,
                                       ep_degree: int = 1) -> float:
        """
        估算 MoE token dispatch/combine 的设备侧开销 (ms)。
        """
        if self.model.num_experts <= 1:
            return 0.0

        tokens = max(1, int(batch_size) * int(seq_len))
        topk = max(1, int(self.model.num_experts_per_tok))
        h = max(1, int(self.model.hidden_size))
        dtype_bytes = max(1, int(self.training.dtype_bytes))

        # 两次主要数据搬运：dispatch + combine
        moved_bytes = tokens * topk * h * dtype_bytes * 2

        # 不规则 gather/scatter 有效带宽明显低于峰值显存带宽
        irregular_bw_gbps = max(
            40.0,
            min(240.0, float(self.hardware.gpu.memory_bandwidth_gbps) * 0.12),
        )
        transfer_ms = moved_bytes / (irregular_bw_gbps * 1e9) * 1000.0

        # 路由索引/置换相关的 kernel 启动和调度开销
        launch_ms = (6 + topk) * 0.03  # 约 30us / kernel
        ep_sync_ms = max(0, ep_degree - 1) * 0.02
        extra_runtime_ms = 0.0
        dispatcher = str(
            getattr(self.training, "moe_token_dispatcher_type", "")
        ).strip().lower()
        if dispatcher == "deepep" and int(self.hardware.num_nodes) > 1:
            extra_passes = 1.5
            if bool(getattr(self.training, "variable_seq_lengths", False)):
                extra_passes += 0.5
            if bool(getattr(self.training, "enable_dynamic_shape", False)):
                extra_passes += 0.5
            if bool(getattr(self.training, "moe_router_fusion", False)):
                extra_passes += 0.25
            if bool(getattr(self.training, "moe_ep_barrier", False)):
                extra_passes += 0.25

            pack_bw_gbps = max(
                30.0,
                min(160.0, float(self.hardware.gpu.memory_bandwidth_gbps) * 0.08),
            )
            extra_runtime_ms += (
                moved_bytes * extra_passes / (pack_bw_gbps * 1e9) * 1000.0
            )
            experts_per_gpu = max(1, self.model.num_experts // max(1, ep_degree))
            extra_runtime_ms += experts_per_gpu * 0.06

        return transfer_ms + launch_ms + ep_sync_ms + extra_runtime_ms
    
    def estimate_mlp_time(self, batch_size: int, seq_len: int,
                          tp_degree: int = 1) -> float:
        """估算 Dense MLP 计算时间 (ms)"""
        return self.estimate_layer_time(
            LayerType.DENSE_MLP, batch_size, seq_len, tp_degree
        )
    
    def estimate_moe_time(self, batch_size: int, seq_len: int,
                          tp_degree: int = 1, ep_degree: int = 1) -> float:
        """
        估算 MoE 层计算时间 (ms)
        
        包括 Router + TopK Expert 计算
        """
        return self.estimate_moe_time_breakdown(
            batch_size, seq_len, tp_degree, ep_degree
        )["total"]

    def estimate_moe_time_breakdown(self, batch_size: int, seq_len: int,
                                    tp_degree: int = 1,
                                    ep_degree: int = 1) -> Dict[str, float]:
        """
        估算 MoE 层内部各部分时间。
        """
        # Router 时间
        router_time = self.estimate_layer_time(
            LayerType.MOE_ROUTER, batch_size, seq_len, 1  # Router 不 TP 切分
        )
        
        # Expert 计算时间
        # 每个 token 只激活 TopK 个 Expert
        # EP 切分后，每个 GPU 只计算 num_experts/ep 个 Expert
        topk = self.model.num_experts_per_tok
        experts_per_gpu = max(1, self.model.num_experts // max(1, ep_degree))
        
        # 平均每个 GPU 处理的 token 数
        # 理想负载均衡下，每个 GPU 处理 tokens * topk / ep 个 token-expert pair
        tokens = batch_size * seq_len
        tokens_per_gpu = tokens * topk / ep_degree

        # 关键修正：
        # Expert kernel 的 GEMM 尺寸由“每个 expert 的 token 数”决定，
        # 而不是所有 token-expert pair 的总和。
        tokens_per_expert = max(1.0, tokens_per_gpu / experts_per_gpu)
        expert_time_single = self.estimate_layer_time(
            LayerType.MOE_EXPERT, 1, int(math.ceil(tokens_per_expert)), tp_degree
        )
        expert_time = expert_time_single * experts_per_gpu
        expert_time = self._apply_grouped_gemm_runtime_penalty(
            expert_time, tokens_per_expert, experts_per_gpu, backward=False
        )

        shared_expert_time = self._estimate_shared_expert_time(
            tokens, max(1, tp_degree), self.training.dtype_bytes
        )
        if shared_expert_time > 0:
            if bool(getattr(self.training, "moe_shared_expert_overlap", False)):
                expert_time = max(expert_time, shared_expert_time)
            else:
                expert_time += shared_expert_time

        dispatch_overhead = self.estimate_moe_dispatch_overhead(
            batch_size, seq_len, ep_degree=ep_degree
        )

        h = self.model.hidden_size
        moe_ffn = self.model.moe_intermediate_size
        expert_total_flops = (
            h * moe_ffn +
            h * moe_ffn +
            moe_ffn * 10 +
            moe_ffn +
            moe_ffn * h
        )
        expert_gate_up_flops = 2 * h * moe_ffn
        expert_activation_flops = moe_ffn * 11
        expert_down_flops = moe_ffn * h

        gate_up_fraction = (
            expert_gate_up_flops / expert_total_flops
            if expert_total_flops > 0
            else 0.0
        )
        activation_fraction = (
            expert_activation_flops / expert_total_flops
            if expert_total_flops > 0
            else 0.0
        )
        down_fraction = (
            expert_down_flops / expert_total_flops
            if expert_total_flops > 0
            else 0.0
        )

        return {
            "router": router_time,
            "dispatch": dispatch_overhead,
            "expert_total": expert_time,
            "expert_gate_up": expert_time * gate_up_fraction,
            "expert_activation": expert_time * activation_fraction,
            "expert_down": expert_time * down_fraction,
            "total": router_time + expert_time + dispatch_overhead,
        }

    def estimate_moe_backward_time_breakdown(self, batch_size: int, seq_len: int,
                                             tp_degree: int = 1,
                                             ep_degree: int = 1) -> Dict[str, float]:
        """
        估算 MoE backward 时间分解。
        """
        tokens = max(1, batch_size * seq_len)
        db = self.training.dtype_bytes

        router_time = self._estimate_router_backward_time(tokens, db)

        topk = self.model.num_experts_per_tok
        experts_per_gpu = max(1, self.model.num_experts // max(1, ep_degree))
        tokens_per_gpu = tokens * topk / ep_degree
        tokens_per_expert = max(1.0, tokens_per_gpu / experts_per_gpu)
        expert_time_single = self._estimate_expert_mlp_backward_time(
            int(math.ceil(tokens_per_expert)), tp_degree, db
        )
        expert_time = expert_time_single * experts_per_gpu
        expert_time = self._apply_grouped_gemm_runtime_penalty(
            expert_time, tokens_per_expert, experts_per_gpu, backward=True
        )

        shared_expert_time = self._estimate_shared_expert_backward_time(
            tokens, max(1, tp_degree), db
        )
        if shared_expert_time > 0:
            if bool(getattr(self.training, "moe_shared_expert_overlap", False)):
                expert_time = max(expert_time, shared_expert_time)
            else:
                expert_time += shared_expert_time

        # backward 中仍需执行一轮反向路由搬运，量级与 forward 的 dispatch/combine 相近
        dispatch_overhead = self.estimate_moe_dispatch_overhead(
            batch_size, seq_len, ep_degree=ep_degree
        )

        return {
            "router": router_time,
            "dispatch": dispatch_overhead,
            "expert_total": expert_time,
            "total": router_time + expert_time + dispatch_overhead,
        }
    
    def estimate_dense_layer_time(self, batch_size: int, seq_len: int,
                                  tp_degree: int = 1) -> float:
        """估算单个 Dense 层 (Attention + MLP) 时间 (ms)"""
        return self.estimate_dense_layer_time_breakdown(
            batch_size, seq_len, tp_degree
        )["full"]

    def estimate_dense_layer_time_breakdown(self, batch_size: int, seq_len: int,
                                            tp_degree: int = 1) -> Dict[str, float]:
        """估算 Dense 层时间分解。"""
        attn = self.estimate_attention_time_breakdown(batch_size, seq_len, tp_degree)
        mlp_time = self.estimate_mlp_time(batch_size, seq_len, tp_degree)
        ln_time = self.estimate_layer_time(
            LayerType.LAYERNORM, batch_size, seq_len, 1
        ) * 2
        return {
            "attention_total": attn["total"],
            "attention_core": attn["core_attention"],
            "layernorm": ln_time,
            "mlp": mlp_time,
            "full": attn["total"] + mlp_time + ln_time,
        }
    
    def estimate_moe_layer_time(self, batch_size: int, seq_len: int,
                                tp_degree: int = 1, ep_degree: int = 1) -> float:
        """估算单个 MoE 层 (Attention + MoE) 时间 (ms)"""
        return self.estimate_moe_layer_time_breakdown(
            batch_size, seq_len, tp_degree, ep_degree
        )["full"]

    def estimate_moe_layer_time_breakdown(self, batch_size: int, seq_len: int,
                                          tp_degree: int = 1,
                                          ep_degree: int = 1) -> Dict[str, float]:
        """估算 MoE 层时间分解。"""
        attn = self.estimate_attention_time_breakdown(batch_size, seq_len, tp_degree)
        moe = self.estimate_moe_time_breakdown(batch_size, seq_len, tp_degree, ep_degree)
        ln_time = self.estimate_layer_time(
            LayerType.LAYERNORM, batch_size, seq_len, 1
        ) * 2
        return {
            "attention_total": attn["total"],
            "attention_core": attn["core_attention"],
            "layernorm": ln_time,
            "moe_total": moe["total"],
            "moe_router": moe["router"],
            "moe_dispatch": moe["dispatch"],
            "moe_gate_up": moe["expert_gate_up"],
            "moe_act": moe["expert_activation"],
            "moe_down": moe["expert_down"],
            "full": attn["total"] + moe["total"] + ln_time,
        }

    def estimate_dense_layer_backward_time_breakdown(self, batch_size: int, seq_len: int,
                                                     tp_degree: int = 1) -> Dict[str, float]:
        """估算 Dense 层 backward 时间分解。"""
        attn = self.estimate_attention_backward_time_breakdown(
            batch_size, seq_len, tp_degree
        )
        tokens = max(1, batch_size * seq_len)
        db = self.training.dtype_bytes
        mlp_time = self._estimate_dense_mlp_backward_time(
            tokens, max(1, tp_degree), db
        )
        ln_time = self._estimate_layernorm_backward_time(tokens, db) * 2
        return {
            "attention_total": attn["total"],
            "attention_projection": attn["projection"],
            "attention_core": attn["core_attention"],
            "layernorm": ln_time,
            "mlp": mlp_time,
            "full": attn["total"] + mlp_time + ln_time,
        }

    def estimate_moe_layer_backward_time_breakdown(self, batch_size: int, seq_len: int,
                                                   tp_degree: int = 1,
                                                   ep_degree: int = 1) -> Dict[str, float]:
        """估算 MoE 层 backward 时间分解。"""
        attn = self.estimate_attention_backward_time_breakdown(
            batch_size, seq_len, tp_degree
        )
        moe = self.estimate_moe_backward_time_breakdown(
            batch_size, seq_len, tp_degree, ep_degree
        )
        tokens = max(1, batch_size * seq_len)
        db = self.training.dtype_bytes
        ln_time = self._estimate_layernorm_backward_time(tokens, db) * 2
        return {
            "attention_total": attn["total"],
            "attention_projection": attn["projection"],
            "attention_core": attn["core_attention"],
            "layernorm": ln_time,
            "moe_total": moe["total"],
            "moe_router": moe["router"],
            "moe_dispatch": moe["dispatch"],
            "moe_expert": moe["expert_total"],
            "full": attn["total"] + moe["total"] + ln_time,
        }
    
    def _estimate_framework_overhead_ms(self, batch_size: int, seq_len: int,
                                        parallel: ParallelConfig,
                                        forward_time_ms: float) -> float:
        """
        估算单个 microstep 的框架/运行时开销 (ms)。

        核心思路: 每个 CUDA kernel 需要 CPU 端 dispatch (Python 调度 + Paddle
        框架 op 准备 + CUDA launch)。对于执行时间很长的 kernel，CPU dispatch
        可以被 GPU 执行流水线隐藏；对于执行时间短的 kernel，GPU 会在 kernel
        之间产生 idle gap。

        所有参数均从模型结构动态推导，无固定效率因子。
        """
        pp = max(1, parallel.pp)
        layers_per_stage = self.model.num_hidden_layers // pp
        moe_layers_per_stage = self.model.num_moe_layers // pp
        dense_layers_per_stage = max(0, layers_per_stage - moe_layers_per_stage)

        # ── 每层每个 pass 的 kernel 数量（从模型结构推导）──
        # Dense layer kernels:
        #   Attention: LayerNorm(1) + QKV proj(1-3) + RoPE(1) + FlashAttn(1)
        #              + O proj(1) + residual(1) = ~8
        #   MLP: LayerNorm(1) + Gate GEMM(1) + Up GEMM(1) + SiLU(1)
        #        + Multiply(1) + Down GEMM(1) + residual(1) = ~7
        #   Total: ~15 kernels/layer/pass
        dense_kernels_per_layer = 15

        # MoE layer kernels:
        #   Attention: same as dense = ~8
        #   MoE: LayerNorm(1) + Router(1) + TopK(1) + dispatch/permute(3)
        #        + Expert GEMM x3(3) + SiLU(1) + Multiply(1)
        #        + combine/gather(3) + residual(1) = ~15
        #   Total: ~23 kernels/layer/pass
        moe_kernels_per_layer = 23

        # fwd + bwd = 3 等效 passes
        num_passes = 3.0
        total_kernels = (
            dense_layers_per_stage * dense_kernels_per_layer +
            moe_layers_per_stage * moe_kernels_per_layer
        ) * num_passes

        if total_kernels <= 0:
            return 0.0

        # CPU 侧 launch/dispatch 延迟。这里保留为硬件相关常数，
        # 其余部分尽量从张量字节量和 kernel 数量符号化推导。
        cpu_dispatch_us = 35.0

        tokens = max(1, batch_size * seq_len)
        tp = max(1, parallel.tp)
        db = self.training.dtype_bytes
        h = int(self.model.hidden_size)
        kv = int(self.model.num_key_value_heads * self.model.head_dim)
        ffn = int(self.model.intermediate_size)
        h_act = h // (tp if bool(getattr(parallel, "sp", False)) and tp > 1 else 1)
        q_out = max(1, h // tp)
        kv_out = max(1, kv // tp)
        ffn_tp = max(1, ffn // tp)

        dense_small_per_layer = 9 + (2 if bool(getattr(self.training, "use_qk_norm", False)) else 0)
        dense_big_per_layer = 6
        moe_small_per_layer = 13 + (2 if bool(getattr(self.training, "use_qk_norm", False)) else 0)
        moe_big_per_layer = 10

        total_small_kernels = (
            dense_layers_per_stage * dense_small_per_layer +
            moe_layers_per_stage * moe_small_per_layer
        ) * num_passes
        total_big_kernels = (
            dense_layers_per_stage * dense_big_per_layer +
            moe_layers_per_stage * moe_big_per_layer
        ) * num_passes

        dense_small_bytes_per_layer = (
            tokens * h_act * db * 10.0 +
            tokens * (q_out + kv_out) * db * 4.0 +
            tokens * ffn_tp * db * 5.0
        )
        if bool(getattr(self.training, "use_qk_norm", False)):
            dense_small_bytes_per_layer += tokens * (q_out + kv_out) * db * 5.0

        moe_small_bytes_per_layer = (
            tokens * h_act * db * 10.0 +
            tokens * (q_out + kv_out) * db * 4.0 +
            tokens * db * 16.0 * max(1, int(self.model.num_experts_per_tok))
        )
        if bool(getattr(self.training, "use_qk_norm", False)):
            moe_small_bytes_per_layer += tokens * (q_out + kv_out) * db * 5.0

        small_total_compute_ms = self._membound_time_ms(
            num_passes * (
                dense_layers_per_stage * dense_small_bytes_per_layer +
                moe_layers_per_stage * moe_small_bytes_per_layer
            )
        )
        pure_compute_ms = forward_time_ms * num_passes

        small_kernel_exec_us = (small_total_compute_ms / max(1.0, total_small_kernels)) * 1000.0
        small_overhead_us = max(0.0, cpu_dispatch_us - small_kernel_exec_us)
        small_total_ms = total_small_kernels * small_overhead_us / 1000.0

        if total_big_kernels > 0 and pure_compute_ms > 0:
            big_total_compute_ms = max(0.001, pure_compute_ms - small_total_compute_ms)
            avg_big_kernel_us = big_total_compute_ms / total_big_kernels * 1000.0
            big_overhead_us = max(0.0, cpu_dispatch_us - min(avg_big_kernel_us, cpu_dispatch_us))
            big_total_ms = total_big_kernels * big_overhead_us / 1000.0
        else:
            big_total_ms = 0.0

        return small_total_ms + big_total_ms

    def estimate_forward_time(self, batch_size: int, seq_len: int,
                              parallel: ParallelConfig) -> float:
        """
        估算完整前向传播时间 (ms)
        
        考虑 PP 切分，每个 stage 只计算部分层
        """
        stage_times = self.estimate_forward_stage_times(batch_size, seq_len, parallel)
        return max(stage_times) if stage_times else 0.0
    
    def estimate_backward_time(self, batch_size: int, seq_len: int,
                               parallel: ParallelConfig) -> float:
        """
        估算完整反向传播时间 (ms)
        
        使用按子模块加权的动态 backward ratio，而不是全局固定常数。

        经验上，backward 与 forward 的比值主要受以下因素影响:
        - `seq_len`: 序列越短，kernel launch / sparse routing / reduction 开销占比越高
        - `batch_size`: micro batch 越大，MoE expert 的 wgrad/dgrad 工作量上升更明显
        - `layer_type`: Attention / Dense MLP / MoE / LayerNorm 的 backward 比值不同
        """
        stage_times = self.estimate_backward_stage_times(batch_size, seq_len, parallel)
        return max(stage_times) if stage_times else 0.0

    def _estimate_backward_component_ratios(self, batch_size: int, seq_len: int,
                                            parallel: ParallelConfig) -> Dict[str, float]:
        """
        估算 backward / forward 的组件级比值。

        这个模型不再使用单一 `forward * 常数`，而是把 backward 拆成：
        - attention
        - dense mlp
        - moe
        - layernorm

        其中 MoE 对 `short_seq * log2(batch)` 更敏感：
        短序列下 expert wgrad、routing reduction 和小 kernel gap 更难被隐藏，
        micro-batch 变大后这部分会显著放大。
        """
        short_seq_factor = min(1.0, 4096.0 / max(1.0, float(seq_len)))
        batch_factor = math.log2(max(1.0, float(batch_size)))

        attention_ratio = (
            self.BACKWARD_ATTENTION_BASE +
            self.BACKWARD_ATTENTION_SHORT_SEQ_COEF * short_seq_factor
        )
        dense_mlp_ratio = (
            self.BACKWARD_DENSE_MLP_BASE +
            self.BACKWARD_DENSE_MLP_SHORT_SEQ_COEF * short_seq_factor +
            self.BACKWARD_DENSE_MLP_BATCH_LOG2_COEF * batch_factor
        )
        moe_ratio = (
            self.BACKWARD_MOE_BASE +
            self.BACKWARD_MOE_SHORT_SEQ_COEF * short_seq_factor +
            self.BACKWARD_MOE_BATCH_LOG2_COEF * batch_factor +
            self.BACKWARD_MOE_SHORT_SEQ_BATCH_COEF * short_seq_factor * batch_factor
        )
        layernorm_ratio = self.BACKWARD_LAYERNORM_RATIO

        if parallel.tp > 1:
            tp_bonus = self.BACKWARD_TP_LOG2_BONUS * math.log2(max(1, parallel.tp))
            attention_ratio += tp_bonus
            dense_mlp_ratio += tp_bonus
            moe_ratio += tp_bonus

        if parallel.pp > 1:
            # 当前 backward 标定样本主要来自 PP=1。
            # 对 PP>1，直接沿用 PP=1 的 ratio 会系统性高估 stage-local backward。
            # 这里按 log2(pp) 做折减，并保留合理下界。
            pp_discount = math.log2(max(1, parallel.pp))
            attention_ratio = max(
                self.BACKWARD_ATTENTION_MIN_RATIO,
                attention_ratio - self.BACKWARD_ATTENTION_PP_LOG2_DISCOUNT * pp_discount,
            )
            dense_mlp_ratio = max(
                self.BACKWARD_DENSE_MLP_MIN_RATIO,
                dense_mlp_ratio - self.BACKWARD_DENSE_MLP_PP_LOG2_DISCOUNT * pp_discount,
            )
            moe_ratio = max(
                self.BACKWARD_MOE_MIN_RATIO,
                moe_ratio - self.BACKWARD_MOE_PP_LOG2_DISCOUNT * pp_discount,
            )

        return {
            "attention": attention_ratio,
            "dense_mlp": dense_mlp_ratio,
            "moe": moe_ratio,
            "layernorm": layernorm_ratio,
        }
    
    def estimate_pipeline_bubble(self, forward_time: float, backward_time: float,
                                 pp_degree: int, num_micro_batches: int) -> float:
        """
        估算流水线气泡时间 (ms)
        
        1F1B 调度: bubble_ratio = (pp_degree - 1) / num_micro_batches
        """
        if pp_degree <= 1:
            return 0.0
        
        bubble_ratio = (pp_degree - 1) / num_micro_batches
        single_stage_time = forward_time + backward_time
        
        return single_stage_time * bubble_ratio

    def _estimate_recompute_stage_times(self, batch_size: int, seq_len: int,
                                        parallel: ParallelConfig,
                                        recompute_granularity: RecomputeGranularity,
                                        recompute_method: Optional[str] = None,
                                        recompute_num_layers: Optional[int] = None,
                                        recompute_modules: Optional[tuple[str, ...]] = None) -> list[float]:
        """估算单个 micro-batch 各 stage 的重计算开销。"""
        if recompute_granularity == RecomputeGranularity.NONE:
            return [0.0] * max(1, int(parallel.pp))

        modules = {
            str(module).strip().lower()
            for module in (recompute_modules or tuple())
            if module
        }

        self._validate_recompute_config(
            recompute_granularity,
            recompute_method,
            recompute_num_layers,
            recompute_modules,
        )

        if recompute_granularity == RecomputeGranularity.SELECTIVE and not modules:
            return [0.0] * max(1, int(parallel.pp))

        method = self._normalize_recompute_method(
            recompute_granularity, recompute_method
        )
        layer_payload = self._get_layer_runtime_profile(
            batch_size, seq_len, parallel.tp, parallel.ep
        )
        layers = layer_payload["layers"]

        stage_times = []
        if recompute_granularity == RecomputeGranularity.FULL:
            for stage_chunks in self._stage_chunk_ranges(parallel):
                selected_layers = self._select_full_recomputed_layers(
                    stage_chunks, method, recompute_num_layers
                )
                if not selected_layers:
                    stage_times.append(0.0)
                    continue

                stage_time = sum(
                    self._get_runtime_layer_template(layer_payload, layers[layer_idx])[
                        "forward_full"
                    ]
                    for layer_idx in selected_layers
                )
                unit_count = self._estimate_full_recompute_unit_count(
                    stage_chunks, method, recompute_num_layers
                )
                stage_time += unit_count * self.RECOMPUTE_UNIT_OVERHEAD_US / 1000.0
                stage_times.append(stage_time)
        else:
            for stage_chunks in self._stage_chunk_ranges(parallel):
                selected_layers = self._select_selective_recomputed_layers(
                    stage_chunks, method, recompute_num_layers
                )
                if not selected_layers:
                    stage_times.append(0.0)
                    continue

                stage_time = 0.0
                for layer_idx in selected_layers:
                    layer = self._get_runtime_layer_template(
                        layer_payload, layers[layer_idx]
                    )
                    if any(m in modules for m in {"mlp", "dense_mlp"}):
                        if layer["layer_kind"] == TRANSFORMER_MOE_LAYER_KIND:
                            stage_time += layer["moe_total"]
                        else:
                            stage_time += layer["mlp"]
                    if any(m in modules for m in {"attention", "core_attn", "flash_attn"}):
                        stage_time += layer["attention_core"]
                    if any(m in modules for m in {"proj", "qkv_proj", "o_proj"}):
                        stage_time += layer["attention_projection"]
                    if any(m in modules for m in {"norm", "layernorm", "qk_norm"}):
                        stage_time += layer["layernorm"]
                stage_times.append(stage_time)

        return stage_times if stage_times else [0.0] * max(1, int(parallel.pp))

    def estimate_recompute_time(self, batch_size: int, seq_len: int,
                                parallel: ParallelConfig,
                                recompute_granularity: RecomputeGranularity,
                                recompute_method: Optional[str] = None,
                                recompute_num_layers: Optional[int] = None,
                                recompute_modules: Optional[tuple[str, ...]] = None) -> float:
        """
        估算单个 micro-batch 的重计算工作量。
        """
        stage_times = self._estimate_recompute_stage_times(
            batch_size,
            seq_len,
            parallel,
            recompute_granularity,
            recompute_method,
            recompute_num_layers,
            recompute_modules,
        )
        return max(stage_times) if stage_times else 0.0

    def _estimate_multi_node_runtime_overhead_ms(self,
                                                 parallel: ParallelConfig,
                                                 num_micro_batches: int) -> float:
        """估算多机 pipeline/runtime 额外开销。"""
        if int(self.hardware.num_nodes) <= 1:
            return 0.0

        pp = max(1, int(parallel.pp))
        per_micro_ms = 0.0
        if pp > 1:
            per_micro_ms += 0.18 * pp
            if bool(getattr(self.training, "use_batch_p2p_comm", False)):
                per_micro_ms += 0.08 * max(0, pp - 1)
            if bool(getattr(self.training, "best_unbalanced_scheduler", False)):
                per_micro_ms += 0.05 * pp
        if bool(getattr(self.training, "variable_seq_lengths", False)):
            per_micro_ms += 0.12
        if bool(getattr(self.training, "enable_dynamic_shape", False)):
            per_micro_ms += 0.18

        return per_micro_ms * max(1, int(num_micro_batches))
    
    def estimate_step_compute_time(self, batch_size: int, seq_len: int,
                                   parallel: ParallelConfig,
                                   num_micro_batches: int,
                                   recompute_granularity: RecomputeGranularity = RecomputeGranularity.NONE,
                                   recompute_method: Optional[str] = None,
                                   recompute_num_layers: Optional[int] = None,
                                   recompute_modules: Optional[tuple[str, ...]] = None) -> Dict[str, float]:
        """
        估算一个 step 的计算时间
        
        Args:
            batch_size: micro batch size
            seq_len: 序列长度
            parallel: 并行配置
            num_micro_batches: gradient accumulation steps
            recompute_granularity: 重计算粒度
            recompute_method: 重计算方法
            recompute_num_layers: 重计算层数
            recompute_modules: selective recompute 模块列表
        
        Returns:
            时间详情字典
        """
        stage_forward_times = self.estimate_forward_stage_times(
            batch_size, seq_len, parallel
        )
        stage_backward_times = self.estimate_backward_stage_times(
            batch_size, seq_len, parallel
        )
        forward_time = max(stage_forward_times) if stage_forward_times else 0.0
        backward_time = max(stage_backward_times) if stage_backward_times else 0.0

        # ========== 重计算开销 ==========
        # 逻辑：重计算 = 在反向传播中重新执行指定模块的前向计算
        #   - 重计算的 FLOPs 工作量由 estimate_recompute_time() 给出
        #   - 此外还有每个模块的固定开销（kernel launch、激活 tensor 重建的内存 I/O）
        #
        # full recompute: 重算所有层的全部前向
        # selective recompute: 只重算指定子模块（mlp, flash_attn, norm 等）
        recompute_stage_times = self._estimate_recompute_stage_times(
            batch_size, seq_len, parallel,
            recompute_granularity=recompute_granularity,
            recompute_method=recompute_method,
            recompute_num_layers=recompute_num_layers,
            recompute_modules=recompute_modules,
        )
        recompute_time = max(recompute_stage_times) if recompute_stage_times else 0.0
        stage_backward_with_recompute = [
            base + extra
            for base, extra in zip(stage_backward_times, recompute_stage_times)
        ]
        backward_time_with_recompute = (
            max(stage_backward_with_recompute)
            if stage_backward_with_recompute
            else 0.0
        )

        # 流水线 makespan / bubble
        pipeline_compute_ms = simulate_1f1b_makespan(
            stage_forward_times,
            stage_backward_with_recompute,
            num_micro_batches,
        )
        steady_stage_cycle_ms = max(
            (
                fwd + bwd
                for fwd, bwd in zip(stage_forward_times, stage_backward_with_recompute)
            ),
            default=0.0,
        )
        bubble_time = max(
            0.0,
            pipeline_compute_ms - steady_stage_cycle_ms * max(1, num_micro_batches),
        )

        # 框架/运行时开销（动态图调度、kernel launch、GPU idle gaps）
        #
        # 模型: 每个 kernel 的执行由 CPU dispatch + GPU execution 两部分组成。
        # CPU dispatch (Python + Paddle framework) 是串行的，当 GPU kernel 执行
        # 时间大于 CPU dispatch 时间时，dispatch 可以被流水线隐藏；当 GPU kernel
        # 很小（memory-bound ops, 小 GEMM）时，GPU 需要等待 CPU，产生 idle gap。
        #
        # overhead = Σ_kernels max(0, cpu_dispatch_time - gpu_kernel_time)
        #
        # 简化为两类 kernel:
        #   "big" (GEMM, FlashAttn): exec_time >> dispatch_time → overhead ≈ 0
        #   "small" (elementwise, LayerNorm, bias, TopK, etc.): overhead ≈ dispatch_time
        #
        # 参数全部从模型结构动态推导。
        framework_overhead_per_micro_ms = self._estimate_framework_overhead_ms(
            batch_size, seq_len, parallel, forward_time
        )
        framework_overhead_ms = framework_overhead_per_micro_ms * num_micro_batches
        runtime_overhead_ms = self._estimate_multi_node_runtime_overhead_ms(
            parallel, num_micro_batches
        )
        
        # 总计算时间
        compute_time = pipeline_compute_ms + framework_overhead_ms + runtime_overhead_ms
        effective_recompute = (
            backward_time_with_recompute / backward_time
            if backward_time > 0
            else 1.0
        )
        
        return {
            "forward_time_ms": forward_time * num_micro_batches,
            "backward_time_ms": backward_time_with_recompute * num_micro_batches,
            "recompute_time_ms": recompute_time * num_micro_batches,
            "recompute_overhead": effective_recompute,
            "bubble_time_ms": bubble_time,
            "framework_overhead_ms": framework_overhead_ms,
            "runtime_overhead_ms": runtime_overhead_ms,
            "compute_time_ms": compute_time,
            "bubble_ratio": bubble_time / compute_time if compute_time > 0 else 0,
            "stage_forward_micro_ms": stage_forward_times,
            "stage_backward_micro_ms": stage_backward_with_recompute,
            "pipeline_compute_makespan_ms": pipeline_compute_ms,
        }


# --- merged stage-local wrapper logic ---
import copy
from typing import Any, Dict, Iterable, List, Optional, Sequence
from ..config import RecomputeGranularity
from .memory_model import RecomputeConfig
from .recompute_stage_sim import (
    build_stage_plans_from_recompute_configs,
    build_uniform_stage_plans,
    plans_to_dicts,
)

class ComputeModel(_BaseComputeModel):
    """
    Drop-in replacement for the original ComputeModel.

    Key changes:
    - block recompute becomes stage-local rather than one global proxy;
    - supports `per_stage_recompute` as an optional extension without changing
      existing callers;
    - exposes `stage_recompute_detail` in the returned dict.
    """

    def estimate_step_compute_time(
        self,
        batch_size: int = None,
        seq_len: int = None,
        parallel=None,
        num_micro_batches: int = None,
        recompute_granularity: Any = RecomputeGranularity.NONE,
        recompute_method: Optional[str] = None,
        recompute_num_layers: Optional[int] = None,
        recompute_modules: Optional[Sequence[str]] = None,
        per_stage_recompute: Optional[Sequence[RecomputeConfig]] = None,
        stage_layer_counts: Optional[Sequence[int]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Backward-compatible extension of the original interface.

        Existing callers can keep using:
            estimate_step_compute_time(..., recompute_granularity=..., ...)

        New callers may additionally pass:
            per_stage_recompute=[RecomputeConfig(...), ...]
        """
        legacy_batch_size = kwargs.pop("micro_batch_size", None)
        legacy_num_micro_batches = kwargs.pop(
            "gradient_accumulation_steps",
            None,
        )
        legacy_per_stage_recompute = kwargs.pop(
            "per_stage_recompute_configs",
            None,
        )
        if batch_size is None:
            batch_size = legacy_batch_size
        if num_micro_batches is None:
            num_micro_batches = legacy_num_micro_batches
        if per_stage_recompute is None:
            per_stage_recompute = legacy_per_stage_recompute
        if batch_size is None:
            raise TypeError(
                "estimate_step_compute_time() missing required argument: 'batch_size'"
            )
        if num_micro_batches is None:
            raise TypeError(
                "estimate_step_compute_time() missing required argument: "
                "'num_micro_batches'"
            )
        if seq_len is None:
            raise TypeError(
                "estimate_step_compute_time() missing required argument: 'seq_len'"
            )
        if parallel is None:
            raise TypeError(
                "estimate_step_compute_time() missing required argument: 'parallel'"
            )
        normalized_recompute_granularity = recompute_granularity
        if isinstance(recompute_granularity, str):
            granularity_map = {
                "none": RecomputeGranularity.NONE,
                "selective": RecomputeGranularity.SELECTIVE,
                "full": RecomputeGranularity.FULL,
            }
            normalized_recompute_granularity = granularity_map.get(
                recompute_granularity.strip().lower(),
                recompute_granularity,
            )

        current_raw = super().estimate_step_compute_time(
            batch_size,
            seq_len,
            parallel,
            num_micro_batches,
            recompute_granularity=normalized_recompute_granularity,
            recompute_method=recompute_method,
            recompute_num_layers=recompute_num_layers,
            recompute_modules=recompute_modules,
            **kwargs,
        )

        stage_layers = self._resolve_stage_layer_counts(parallel, stage_layer_counts)
        if not stage_layers:
            return current_raw

        if per_stage_recompute is not None:
            plans = build_stage_plans_from_recompute_configs(
                stage_layers,
                per_stage_recompute,
            )
        else:
            plans = build_uniform_stage_plans(
                stage_layers,
                normalized_recompute_granularity,
                recompute_method,
                recompute_num_layers,
            )

        if all(plan.granularity == "none" for plan in plans):
            current = copy.deepcopy(current_raw)
            current["stage_recompute_detail"] = plans_to_dicts(plans)
            current["stage_layer_counts"] = list(stage_layers)
            return current

        no_recompute = super().estimate_step_compute_time(
            batch_size,
            seq_len,
            parallel,
            num_micro_batches,
            recompute_granularity=RecomputeGranularity.NONE,
            recompute_method="uniform",
            recompute_num_layers=1,
            recompute_modules=None,
            **kwargs,
        )

        base_forward = self._pad_stage_values(no_recompute.get("stage_forward_micro_ms"), len(stage_layers))
        base_backward = self._pad_stage_values(no_recompute.get("stage_backward_micro_ms"), len(stage_layers))
        if not base_forward or not base_backward:
            # Fall back to the raw current result if the original model did not
            # expose stage-level timing.
            current = copy.deepcopy(current_raw)
            current["stage_recompute_detail"] = plans_to_dicts(plans)
            current["stage_layer_counts"] = list(stage_layers)
            return current

        corrected_forward = list(base_forward)
        corrected_backward: List[float] = []
        for sid, plan in enumerate(plans):
            extra_forward_ms = base_forward[sid] * float(plan.extra_forward_fraction)
            corrected_backward.append(base_backward[sid] + extra_forward_ms)

        base_stage_cycle = [f + b for f, b in zip(base_forward, base_backward)]
        corrected_stage_cycle = [f + b for f, b in zip(corrected_forward, corrected_backward)]
        base_slowest = max(max(base_stage_cycle), 1e-6)
        corrected_slowest = max(max(corrected_stage_cycle), 1e-6)
        compute_scale = corrected_slowest / base_slowest

        refined = copy.deepcopy(current_raw)
        refined["stage_forward_micro_ms"] = corrected_forward
        refined["stage_backward_micro_ms"] = corrected_backward
        refined["forward_time_ms"] = float(no_recompute.get("forward_time_ms", 0.0)) * (
            sum(corrected_forward) / max(sum(base_forward), 1e-6)
        )
        refined["backward_time_ms"] = float(no_recompute.get("backward_time_ms", 0.0)) * (
            sum(corrected_backward) / max(sum(base_backward), 1e-6)
        )
        refined["bubble_time_ms"] = float(no_recompute.get("bubble_time_ms", 0.0)) * compute_scale
        refined["compute_time_ms"] = float(no_recompute.get("compute_time_ms", 0.0)) * compute_scale
        refined["recompute_time_ms"] = max(
            0.0,
            float(refined["compute_time_ms"]) - float(no_recompute.get("compute_time_ms", 0.0)),
        )
        refined["recompute_overhead"] = 1.0 + refined["recompute_time_ms"] / max(
            float(no_recompute.get("compute_time_ms", 0.0)), 1e-6
        )
        # Preserve current runtime/framework fields from the original model.
        refined["framework_overhead_ms"] = float(current_raw.get("framework_overhead_ms", 0.0))
        refined["runtime_overhead_ms"] = float(current_raw.get("runtime_overhead_ms", 0.0))
        refined["stage_recompute_detail"] = plans_to_dicts(plans)
        refined["stage_layer_counts"] = list(stage_layers)
        return refined

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_stage_layer_counts(
        self,
        parallel,
        stage_layer_counts: Optional[Sequence[int]] = None,
    ) -> List[int]:
        if stage_layer_counts:
            return [max(0, int(v)) for v in stage_layer_counts]
        if getattr(parallel, "stage_layer_counts", None):
            return [max(0, int(v)) for v in getattr(parallel, "stage_layer_counts")]

        pp = max(1, int(getattr(parallel, "pp", 1) or 1))
        counts: List[int] = []
        try:
            for sid in range(pp):
                counts.append(len(self._stage_layer_indices(parallel, sid)))
            if counts:
                return counts
        except Exception:
            pass

        total_layers = int(getattr(self.model, "num_hidden_layers", 0) or getattr(self.model, "num_layers", 0) or 0)
        if total_layers <= 0:
            return [1] * pp
        base, rem = divmod(total_layers, pp)
        return [base + (1 if i < rem else 0) for i in range(pp)]

    def _pad_stage_values(self, values: Optional[Iterable[float]], target_len: int) -> List[float]:
        vals = [float(v) for v in (values or [])]
        if len(vals) >= target_len:
            return vals[:target_len]
        if not vals:
            vals = [0.0]
        vals.extend([vals[-1]] * (target_len - len(vals)))
        return vals


__all__ = ["ComputeModel"]

