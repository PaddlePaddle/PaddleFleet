#!/usr/bin/env python3
"""
配置模块 - 定义 PaddleFormers 训练的各类配置

支持从 PaddleFormers 的 config.json 或 TrainingArguments 解析配置
"""

import ast
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import json
import os
import math


class ShardingStage(Enum):
    """Sharding (ZeRO) 阶段"""
    NONE = "none"
    STAGE1 = "stage1"  # 优化器状态分片
    STAGE2 = "stage2"  # 优化器状态 + 梯度分片
    STAGE3 = "stage3"  # 优化器状态 + 梯度 + 参数分片


class RecomputeGranularity(Enum):
    """重计算粒度"""
    NONE = "none"
    SELECTIVE = "selective"  # 选择性重计算
    FULL = "full"  # 全量重计算


TRANSFORMER_DENSE_LAYER_KIND = "dense_transformer"
TRANSFORMER_MOE_LAYER_KIND = "moe_transformer"
INPUT_EMBEDDING_LAYER_KIND = "input_embedding"
OUTPUT_HEAD_LAYER_KIND = "output_head"


@dataclass
class GPUSpec:
    """GPU 硬件规格"""
    name: str = "H100-80GB-HBM3"
    memory_gb: float = 80.0
    
    # 算力 (TFLOPS)
    fp32_tflops: float = 67.0
    fp16_tflops: float = 989.0
    bf16_tflops: float = 989.0
    
    # 带宽 (GB/s)
    memory_bandwidth_gbps: float = 3350.0
    host_to_device_bandwidth_gbps: float = 16.0
    device_to_host_bandwidth_gbps: float = 16.0
    
    # 校准曲线（来自 hardware_spec.py 的 bf16_curve，可选）
    bf16_curve: Optional[Dict[str, Any]] = None
    # 代表性非方阵 BF16 GEMM 样本（来自 calibration.py，可选）
    bf16_gemm_samples: Optional[List[Dict[str, Any]]] = None
    host_to_device_bw_curve: Optional["NetworkBandwidthCurve"] = None
    device_to_host_bw_curve: Optional["NetworkBandwidthCurve"] = None
    
    @property
    def ridge_point(self) -> float:
        """
        Roofline 模型的拐点 (FLOP/Byte)。

        ridge_point = peak_flops / memory_bandwidth
        当算术强度 < ridge_point 时为 memory-bound，否则为 compute-bound。
        """
        peak_flops = self.bf16_tflops * 1e12  # FLOP/s
        mem_bw = self.memory_bandwidth_gbps * 1e9  # Byte/s
        if mem_bw <= 0:
            return float('inf')
        return peak_flops / mem_bw

    def get_roofline_efficiency(self, m: int, n: int, k: int,
                                dtype_bytes: int = 2) -> float:
        """
        基于 Roofline 模型，从 GEMM 维度 (M, N, K) 预测硬件利用效率。

        算术强度 AI = 2*M*N*K / ((M*K + K*N + M*N) * dtype_bytes)
        效率 = min(1.0, AI / ridge_point)

        当无校准曲线时作为兜底估算使用。
        """
        m, n, k = max(1, m), max(1, n), max(1, k)
        flops = 2.0 * m * n * k
        bytes_accessed = (float(m) * k + float(k) * n + float(m) * n) * dtype_bytes
        if bytes_accessed <= 0:
            return 0.01
        ai = flops / bytes_accessed
        rp = self.ridge_point
        if rp <= 0:
            return 1.0
        return max(0.001, min(1.0, ai / rp))

    def get_tflops(self, dtype: str = "bf16", gemm_size: Optional[int] = None) -> float:
        """
        获取指定数据类型算力（TFLOPS）

        - 默认返回峰值算力
        - 当存在 bf16 校准曲线且给定 gemm_size 时，BF16 优先使用实测点插值估算
        """
        dtype_norm = str(dtype).lower()
        if dtype_norm in ("bfloat16", "bf16"):
            dtype_key = "bf16"
        elif dtype_norm in ("float16", "fp16", "half"):
            dtype_key = "fp16"
        elif dtype_norm in ("float32", "fp32"):
            dtype_key = "fp32"
        else:
            dtype_key = "bf16"

        dtype_map = {
            "fp32": self.fp32_tflops,
            "fp16": self.fp16_tflops,
            "bf16": self.bf16_tflops,
        }
        peak = float(dtype_map.get(dtype_key, self.bf16_tflops))

        if dtype_key != "bf16" or gemm_size is None:
            return peak

        interpolated = self._lookup_bf16_curve_tflops(gemm_size)
        if interpolated is not None:
            return min(peak, interpolated)

        fitted = self._lookup_bf16_curve_fit_tflops(gemm_size, peak=peak)
        if fitted is not None:
            return min(peak, fitted)

        return peak

    def get_host_to_device_bandwidth(self, data_size_bytes: int = 0) -> float:
        curve = self.host_to_device_bw_curve
        if curve is not None and getattr(curve, "peak_bw_gbps", 0.0) > 0:
            return curve.effective_bandwidth(data_size_bytes)
        return float(self.host_to_device_bandwidth_gbps)

    def get_device_to_host_bandwidth(self, data_size_bytes: int = 0) -> float:
        curve = self.device_to_host_bw_curve
        if curve is not None and getattr(curve, "peak_bw_gbps", 0.0) > 0:
            return curve.effective_bandwidth(data_size_bytes)
        return float(self.device_to_host_bandwidth_gbps)

    def _lookup_bf16_curve_fit_tflops(self, gemm_size: float,
                                      peak: Optional[float] = None) -> Optional[float]:
        """旧版 log 曲线查询，保留给高度非方阵场景兜底。"""
        curve = self.bf16_curve
        if not isinstance(curve, dict):
            return None

        try:
            fit_a = float(curve.get("fit_a", 0.0))
            fit_b = float(curve.get("fit_b", 0.0))
            fit_max = float(curve.get("fit_max", 1.0))
            curve_peak = float(curve.get("peak_tflops", peak or self.bf16_tflops))
            size = max(1.0, float(gemm_size))
            efficiency = fit_a * math.log(size) + fit_b
            efficiency = max(0.01, min(fit_max, efficiency))
            return max(1e-6, curve_peak * efficiency)
        except Exception:
            return None

    def _lookup_bf16_curve_tflops(self, gemm_size: float) -> Optional[float]:
        """
        基于 BF16 方阵实测点做分段插值。

        相比单条 log 拟合曲线，分段插值更贴近真实测点，尤其是
        2048/4096 这一类处于上升拐点附近的 GEMM。
        """
        curve = self.bf16_curve
        if not isinstance(curve, dict):
            return None

        raw_points = curve.get("points")
        if not isinstance(raw_points, list) or not raw_points:
            return None

        points = []
        for point in raw_points:
            if not isinstance(point, dict):
                continue
            try:
                size = float(point.get("size", 0.0))
                tflops = float(point.get("tflops", 0.0))
            except Exception:
                continue
            if size > 0 and tflops > 0:
                points.append((size, tflops))

        if not points:
            return None

        points.sort(key=lambda item: item[0])
        size = max(1.0, float(gemm_size))

        if size <= points[0][0]:
            return points[0][1]
        if size >= points[-1][0]:
            return points[-1][1]

        for idx in range(1, len(points)):
            left_size, left_tflops = points[idx - 1]
            right_size, right_tflops = points[idx]
            if size > right_size:
                continue
            if abs(size - left_size) < 1e-9:
                return left_tflops
            if abs(size - right_size) < 1e-9:
                return right_tflops

            left_log = math.log(left_size)
            right_log = math.log(right_size)
            size_log = math.log(size)
            if abs(right_log - left_log) < 1e-12:
                return 0.5 * (left_tflops + right_tflops)

            ratio = (size_log - left_log) / (right_log - left_log)
            return left_tflops + ratio * (right_tflops - left_tflops)

    def _lookup_bf16_gemm_sample_tflops(self, m: int, n: int, k: int) -> Optional[float]:
        """
        基于少量代表性 BF16 GEMM 样本做最近邻查询。

        只在查询形状和样本足够接近时启用；否则回退到现有曲线/roofline。
        """
        samples = self.bf16_gemm_samples
        if not isinstance(samples, list) or not samples:
            return None

        target = (
            math.log(max(1.0, float(m))),
            math.log(max(1.0, float(n))),
            math.log(max(1.0, float(k))),
        )
        neighbors = []

        for sample in samples:
            if not isinstance(sample, dict):
                continue
            try:
                sm = float(sample.get("m", 0.0))
                sn = float(sample.get("n", 0.0))
                sk = float(sample.get("k", 0.0))
                st = float(sample.get("tflops", 0.0))
            except Exception:
                continue

            if sm <= 0 or sn <= 0 or sk <= 0 or st <= 0:
                continue

            dist = math.sqrt(
                (target[0] - math.log(sm)) ** 2 +
                (target[1] - math.log(sn)) ** 2 +
                (target[2] - math.log(sk)) ** 2
            )
            neighbors.append((dist, st))

        if not neighbors:
            return None

        neighbors.sort(key=lambda item: item[0])
        if neighbors[0][0] > 1.0:
            return None

        weighted_sum = 0.0
        total_weight = 0.0
        for dist, tflops in neighbors[:3]:
            weight = 1.0 / max(dist, 0.05)
            weighted_sum += weight * tflops
            total_weight += weight

        if total_weight <= 0:
            return None

        return min(float(self.bf16_tflops), max(1e-6, weighted_sum / total_weight))

    def _bf16_roofline_upper_bound_tflops(self, m: int, n: int, k: int,
                                          dtype_bytes: int = 2) -> float:
        """
        BF16 Roofline 上限。

        Roofline 在这里不直接作为预测值，只作为样本/曲线输出的物理上限约束。
        """
        peak = max(1e-6, float(self.bf16_tflops))
        efficiency = self.get_roofline_efficiency(m, n, k, dtype_bytes)
        return max(1e-6, peak * efficiency)

    def get_tflops_for_gemm(self, dtype: str, m: int, n: int, k: int,
                            dtype_bytes: int = 2) -> float:
        """
        根据 GEMM 的实际维度 (M, N, K) 获取有效算力 (TFLOPS)。

        优先使用 workload-aware 的 BF16 GEMM 样本；
        查不到时退回到等效方阵曲线，再退回到 Roofline。
        """
        # 用 min(M,N,K) 作为主导维度来映射等效 GEMM 尺寸
        # 对于 memory-bound 场景（M 很小的 MoE Expert），min 维度决定了算术强度
        m, n, k = max(1, m), max(1, n), max(1, k)
        dtype_norm = str(dtype).lower()
        peak = self.get_tflops(dtype)

        if dtype_norm in ("bfloat16", "bf16"):
            roofline_upper = self._bf16_roofline_upper_bound_tflops(
                m, n, k, dtype_bytes
            )
            sampled = self._lookup_bf16_gemm_sample_tflops(m, n, k)
            if sampled is not None:
                return min(sampled, roofline_upper)
        else:
            roofline_upper = peak

        min_dim = min(m, n, k)
        max_dim = max(m, n, k)

        # 混合策略：当最小维度远小于最大维度时（非方阵），
        # 用 min_dim 和立方根的加权平均，避免立方根高估小 GEMM 效率
        cube_root = int(round((float(m) * float(n) * float(k)) ** (1.0 / 3.0)))
        if max_dim > 0 and min_dim / max_dim < 0.1:
            # 高度非方阵（如 MoE Expert: M=8, N=768, K=2048）
            # min_dim 主导算术强度，给它更高权重
            eq_size = int(round(min_dim * 0.6 + cube_root * 0.4))
        else:
            eq_size = cube_root
        eq_size = max(64, min(65536, eq_size))

        if dtype_norm in ("bfloat16", "bf16"):
            aspect_ratio = (min_dim / max_dim) if max_dim > 0 else 1.0
            if aspect_ratio >= 0.25:
                calibrated = self.get_tflops(dtype, gemm_size=eq_size)
            else:
                fitted = self._lookup_bf16_curve_fit_tflops(eq_size, peak=peak)
                calibrated = fitted if fitted is not None else self.get_tflops(dtype, gemm_size=eq_size)

            calibrated = min(peak, calibrated, roofline_upper)
            if self.bf16_curve is not None:
                return max(1e-6, calibrated)
        else:
            calibrated = min(peak, self.get_tflops(dtype, gemm_size=eq_size))

        # 没有可用曲线时，才使用 Roofline 作为兜底值
        if abs(calibrated - peak) < 1e-6 and self.bf16_curve is None:
            roofline_eff = self.get_roofline_efficiency(m, n, k, dtype_bytes)
            return peak * roofline_eff

        return max(1e-6, calibrated)
    
    @classmethod
    def from_name(cls, name: str) -> "GPUSpec":
        """根据 GPU 型号创建规格"""
        presets = {
            "H100-80GB-HBM3": cls(
                name="H100-80GB-HBM3",
                memory_gb=80.0,
                fp32_tflops=67.0,
                fp16_tflops=989.0,
                bf16_tflops=989.0,
                memory_bandwidth_gbps=3350.0,
            ),
            "H100-80GB-PCIe": cls(
                name="H100-80GB-PCIe",
                memory_gb=80.0,
                fp32_tflops=51.0,
                fp16_tflops=756.0,
                bf16_tflops=756.0,
                memory_bandwidth_gbps=2000.0,
            ),
            "A100-80GB": cls(
                name="A100-80GB",
                memory_gb=80.0,
                fp32_tflops=19.5,
                fp16_tflops=312.0,
                bf16_tflops=312.0,
                memory_bandwidth_gbps=2039.0,
            ),
            "A100-40GB": cls(
                name="A100-40GB",
                memory_gb=40.0,
                fp32_tflops=19.5,
                fp16_tflops=312.0,
                bf16_tflops=312.0,
                memory_bandwidth_gbps=1555.0,
            ),
            "A800-80GB": cls(
                name="A800-80GB",
                memory_gb=80.0,
                fp32_tflops=19.5,
                fp16_tflops=312.0,
                bf16_tflops=312.0,
                memory_bandwidth_gbps=2039.0,
            ),
            "V100-32GB": cls(
                name="V100-32GB",
                memory_gb=32.0,
                fp32_tflops=15.7,
                fp16_tflops=125.0,
                bf16_tflops=0.0,  # V100 不支持 BF16
                memory_bandwidth_gbps=900.0,
            ),
        }
        return presets.get(name, presets["H100-80GB-HBM3"])


@dataclass
class NetworkBandwidthCurve:
    """
    带宽-消息大小关系曲线（logistic 拟合）

    物理含义：小消息受延迟主导，有效带宽远低于峰值；
    随消息增大，有效带宽逐渐趋近链路峰值。
    公式：effective_bw = peak_bw / (1 + exp(-k * (ln(size_bytes) - x0)))

    字段说明:
      peak_bw_gbps  — 链路峰值带宽 (GB/s)
      k             — logistic 增长速率
      x0            — 增长中心点 (ln(size_bytes))
      points        — 原始实测数据点
    """
    peak_bw_gbps: float = 0.0
    k: float = 1.0
    x0: float = 20.0
    points: List[Dict[str, float]] = field(default_factory=list)

    def _normalized_points(self) -> List[tuple[float, float]]:
        """返回按消息大小排序的有效实测点。"""
        normalized: List[tuple[float, float]] = []
        for point in self.points:
            if not isinstance(point, dict):
                continue
            try:
                size_bytes = float(point.get("size_bytes", 0.0))
                bandwidth_gbps = float(point.get("bandwidth_gbps", 0.0))
            except Exception:
                continue
            if size_bytes > 0 and bandwidth_gbps > 0:
                normalized.append((size_bytes, bandwidth_gbps))
        normalized.sort(key=lambda item: item[0])
        return normalized

    def _interpolate_points_bandwidth(self, data_size_bytes: int) -> Optional[float]:
        """
        基于实测点做分段插值。

        与计算曲线保持一致：按消息大小的 log 空间插值，避免在小消息区域
        过度放大噪声，同时保证对大消息平台区间足够平滑。
        """
        points = self._normalized_points()
        if not points:
            return None

        size = max(1.0, float(data_size_bytes))
        if size <= points[0][0]:
            return points[0][1]
        if size >= points[-1][0]:
            return points[-1][1]

        size_log = math.log(size)
        for idx in range(1, len(points)):
            left_size, left_bw = points[idx - 1]
            right_size, right_bw = points[idx]
            if size > right_size:
                continue

            left_log = math.log(left_size)
            right_log = math.log(right_size)
            if abs(right_log - left_log) < 1e-12:
                return 0.5 * (left_bw + right_bw)

            ratio = (size_log - left_log) / (right_log - left_log)
            return left_bw + ratio * (right_bw - left_bw)

    def effective_bandwidth(self, data_size_bytes: int) -> float:
        """返回给定消息大小的有效带宽 (GB/s)"""
        interpolated = self._interpolate_points_bandwidth(data_size_bytes)
        if interpolated is not None:
            return max(0.01, interpolated)
        if self.peak_bw_gbps <= 0:
            return 0.01
        log_size = math.log(max(1.0, float(data_size_bytes)))
        exponent = -self.k * (log_size - self.x0)
        exponent = max(-30.0, min(30.0, exponent))
        bw = self.peak_bw_gbps / (1.0 + math.exp(exponent))
        return max(0.01, bw)

    @classmethod
    def fit_from_points(cls, points: List[Dict[str, float]]) -> "NetworkBandwidthCurve":
        """
        从实测数据 [{size_bytes, bandwidth_gbps}, ...] 拟合曲线参数。

        使用 logistic 变换后的线性回归：
          令 y' = ln(peak/bw - 1)，则 y' = -k*ln(size) + k*x0
        """
        valid_points = []
        for point in points or []:
            if not isinstance(point, dict):
                continue
            try:
                size_bytes = float(point.get("size_bytes", 0.0))
                bandwidth_gbps = float(point.get("bandwidth_gbps", 0.0))
            except Exception:
                continue
            if size_bytes > 0 and bandwidth_gbps > 0:
                valid_points.append(
                    {
                        "size_bytes": int(size_bytes),
                        "bandwidth_gbps": float(bandwidth_gbps),
                    }
                )

        if len(valid_points) < 2:
            return cls()
        peak_bw = max(p["bandwidth_gbps"] for p in valid_points) * 1.05
        xs, ys = [], []
        for p in valid_points:
            bw = p["bandwidth_gbps"]
            sz = p["size_bytes"]
            if bw <= 0 or sz <= 0 or bw >= peak_bw:
                continue
            xs.append(math.log(sz))
            ys.append(math.log(peak_bw / bw - 1.0))
        if len(xs) < 2:
            return cls(peak_bw_gbps=peak_bw, points=list(valid_points))
        n = len(xs)
        sx = sum(xs)
        sy = sum(ys)
        sxy = sum(x * y for x, y in zip(xs, ys))
        sxx = sum(x * x for x in xs)
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-12:
            return cls(peak_bw_gbps=peak_bw, points=list(valid_points))
        a = (n * sxy - sx * sy) / denom
        b = (sy - a * sx) / n
        k = max(0.1, -a)
        x0 = b / k if k > 0 else 20.0
        return cls(peak_bw_gbps=peak_bw, k=k, x0=x0, points=list(valid_points))

    def to_dict(self) -> Dict:
        return {
            "peak_bw_gbps": round(self.peak_bw_gbps, 4),
            "k": round(self.k, 6),
            "x0": round(self.x0, 6),
            "points": [
                {"size_bytes": int(p["size_bytes"]),
                 "bandwidth_gbps": round(p["bandwidth_gbps"], 4)}
                for p in self.points
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "NetworkBandwidthCurve":
        return cls(
            peak_bw_gbps=float(data.get("peak_bw_gbps", 0.0)),
            k=float(data.get("k", 1.0)),
            x0=float(data.get("x0", 20.0)),
            points=list(data.get("points", [])),
        )


@dataclass
class NetworkSpec:
    """网络规格"""
    # 节点内通信 (NVLink/NVSwitch)
    intra_node_bandwidth_gbps: float = 900.0  # H100 NVLink: 900 GB/s
    intra_node_latency_us: float = 1.0
    
    # 节点间通信 (IB/RoCE)
    inter_node_bandwidth_gbps: float = 200.0  # 8x IB HDR: 200 GB/s
    inter_node_latency_us: float = 5.0
    
    # 通信效率因子
    allreduce_efficiency: float = 0.85
    allgather_efficiency: float = 0.80
    alltoall_efficiency: float = 0.70
    p2p_efficiency: float = 0.90

    # 多尺寸带宽曲线（可选，由校准生成）
    intra_node_bw_curve: Optional[NetworkBandwidthCurve] = None
    inter_node_bw_curve: Optional[NetworkBandwidthCurve] = None

    def get_effective_bandwidth(self, is_intra_node: bool,
                                data_size_bytes: int = 0) -> float:
        """
        获取给定消息大小的有效带宽 (GB/s)。

        优先使用实测曲线；回退到静态峰值带宽。
        """
        if is_intra_node:
            if self.intra_node_bw_curve is not None and self.intra_node_bw_curve.peak_bw_gbps > 0:
                return self.intra_node_bw_curve.effective_bandwidth(data_size_bytes)
            return self.intra_node_bandwidth_gbps
        else:
            if self.inter_node_bw_curve is not None and self.inter_node_bw_curve.peak_bw_gbps > 0:
                return self.inter_node_bw_curve.effective_bandwidth(data_size_bytes)
            return self.inter_node_bandwidth_gbps


@dataclass
class HardwareConfig:
    """硬件配置"""
    gpu: GPUSpec = field(default_factory=GPUSpec)
    network: NetworkSpec = field(default_factory=NetworkSpec)
    
    # 集群配置
    num_nodes: int = 1
    gpus_per_node: int = 8
    
    @property
    def total_gpus(self) -> int:
        return self.num_nodes * self.gpus_per_node
    
    def is_intra_node(self, degree: int) -> bool:
        """判断通信是否在节点内"""
        return degree <= self.gpus_per_node


@dataclass
class ModelConfig:
    """
    模型架构配置
    
    与 PaddleFormers 的 PretrainedConfig 对应
    """
    # 基本参数
    num_hidden_layers: int = 48
    hidden_size: int = 6144
    intermediate_size: int = 16384
    num_attention_heads: int = 32
    num_key_value_heads: int = 4  # GQA
    head_dim: int = 192
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    
    # MoE 参数
    num_experts: int = 128
    num_shared_experts: int = 0
    num_experts_per_tok: int = 8  # TopK
    moe_intermediate_size: int = 1408  # Expert FFN 大小
    shared_expert_intermediate_size: int = 0
    decoder_sparse_step: int = 1  # MoE 层间隔（1 表示每层都是 MoE）
    mlp_only_layers: List[int] = field(default_factory=list)  # 纯 MLP 层的索引
    
    # 其他参数
    vocab_size: int = 152064
    max_position_embeddings: int = 32768
    model_type: str = ""
    tie_word_embeddings: bool = True
    
    # 计算得到的属性
    @property
    def num_moe_layers(self) -> int:
        """MoE 层数"""
        return sum(
            1
            for layer_idx in range(self.num_hidden_layers)
            if self.is_moe_layer(layer_idx)
        )
    
    @property
    def num_dense_layers(self) -> int:
        """Dense 层数"""
        return self.num_hidden_layers - self.num_moe_layers

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 1

    @property
    def model_family(self) -> str:
        mt = self.model_type.lower()
        if "qwen" in mt and self.is_moe:
            return "qwen_moe"
        if "qwen" in mt:
            return "qwen"
        if "deepseek" in mt:
            return "deepseek"
        if "llama" in mt:
            return "llama"
        if "gpt" in mt:
            return "gpt"
        return "generic_moe" if self.is_moe else "generic_dense"

    @property
    def attention_style(self) -> str:
        return "mla" if self.uses_low_rank_attention else "mha"

    @property
    def effective_shared_expert_intermediate_size(self) -> int:
        """shared expert 的总 FFN 宽度。"""
        explicit = self.shared_expert_intermediate_size or 0
        if explicit > 0:
            return explicit
        if self.num_shared_experts <= 0:
            return 0
        return self.moe_intermediate_size * self.num_shared_experts

    @property
    def uses_low_rank_attention(self) -> bool:
        """是否启用 MLA / LoRA 风格的 attention 参数化。"""
        return any(
            (value or 0) > 0
            for value in (
                self.q_lora_rank,
                self.kv_lora_rank,
                self.qk_nope_head_dim,
                self.qk_rope_head_dim,
                self.v_head_dim,
            )
        )

    @property
    def effective_query_head_dim(self) -> int:
        if (self.qk_nope_head_dim or 0) > 0 or (self.qk_rope_head_dim or 0) > 0:
            return (self.qk_nope_head_dim or 0) + (self.qk_rope_head_dim or 0)
        return self.head_dim

    @property
    def effective_value_head_dim(self) -> int:
        if (self.v_head_dim or 0) > 0:
            return self.v_head_dim
        return self.head_dim

    def estimate_attention_params_per_layer(self) -> int:
        """估算单层 attention 参数量。"""
        h = self.hidden_size

        if not self.uses_low_rank_attention:
            q_size = h
            kv_size = self.num_key_value_heads * self.head_dim
            return q_size * h + 2 * kv_size * h + h * h

        query_out_dim = self.num_attention_heads * self.effective_query_head_dim
        value_out_dim = self.num_attention_heads * self.effective_value_head_dim

        if (self.q_lora_rank or 0) > 0:
            q_proj_params = h * self.q_lora_rank + self.q_lora_rank * query_out_dim
            q_norm_params = self.q_lora_rank
        else:
            q_proj_params = h * query_out_dim
            q_norm_params = 0

        if (self.kv_lora_rank or 0) > 0:
            kv_latent_dim = self.kv_lora_rank
            kv_a_out_dim = kv_latent_dim + (self.qk_rope_head_dim or 0)
            kv_b_out_dim = self.num_attention_heads * (
                (self.qk_nope_head_dim or self.head_dim) + self.effective_value_head_dim
            )
            kv_proj_params = h * kv_a_out_dim + kv_latent_dim * kv_b_out_dim
            kv_norm_params = kv_latent_dim
        else:
            kv_size = self.num_key_value_heads * self.head_dim
            kv_proj_params = 2 * h * kv_size
            kv_norm_params = 0

        o_proj_params = value_out_dim * h
        return q_proj_params + kv_proj_params + o_proj_params + q_norm_params + kv_norm_params

    def is_moe_layer(self, layer_idx: int) -> bool:
        """判断某个 Transformer block 是否为 MoE block。"""
        if self.num_experts <= 1:
            return False
        if layer_idx in {idx for idx in self.mlp_only_layers}:
            return False
        sparse_step = max(1, self.decoder_sparse_step)
        return ((layer_idx + 1) % sparse_step) == 0

    def transformer_layer_kind(self, layer_idx: int) -> str:
        """返回某个 Transformer block 的层类型。"""
        if self.is_moe_layer(layer_idx):
            return TRANSFORMER_MOE_LAYER_KIND
        return TRANSFORMER_DENSE_LAYER_KIND

    def transformer_layer_kinds(self) -> List[str]:
        """返回完整 Transformer 主干的层类型序列。"""
        return [
            self.transformer_layer_kind(layer_idx)
            for layer_idx in range(self.num_hidden_layers)
        ]
    
    def estimate_parameters(self) -> Dict[str, int]:
        """估算参数量"""
        h = self.hidden_size
        ffn = self.intermediate_size
        moe_ffn = self.moe_intermediate_size
        v = self.vocab_size
        
        # Embedding: v * h
        embedding_params = v * h
        
        attention_params_per_layer = self.estimate_attention_params_per_layer()
        
        # Dense MLP: 3 * h * ffn (gate, up, down with SwiGLU)
        dense_mlp_params = 3 * h * ffn
        
        # MoE 层: router + routed experts + shared experts
        router_params = h * self.num_experts
        expert_params = 3 * h * moe_ffn * self.num_experts  # 所有专家的参数
        shared_expert_params = 3 * h * self.effective_shared_expert_intermediate_size
        moe_layer_params = router_params + expert_params + shared_expert_params
        
        # LayerNorm: 2 * h per layer (input + post_attention)
        layernorm_params = 2 * h
        
        # 汇总
        total_attention = attention_params_per_layer * self.num_hidden_layers
        total_dense_mlp = dense_mlp_params * self.num_dense_layers
        total_moe = moe_layer_params * self.num_moe_layers
        total_layernorm = layernorm_params * self.num_hidden_layers
        total_embedding = embedding_params if self.tie_word_embeddings else embedding_params * 2
        
        total = total_attention + total_dense_mlp + total_moe + total_layernorm + total_embedding
        
        return {
            "embedding": total_embedding,
            "attention": total_attention,
            "dense_mlp": total_dense_mlp,
            "moe": total_moe,
            "layernorm": total_layernorm,
            "total": total,
            "total_billion": total / 1e9,
        }
    
    @classmethod
    def from_name(cls, name: str) -> "ModelConfig":
        """根据模型名称创建配置"""
        def _canonical_model_name(raw_name: str) -> str:
            return "".join(ch for ch in str(raw_name).lower() if ch.isalnum())

        presets = {
            # Qwen3 系列
            "qwen3-30b-a3b": cls(
                num_hidden_layers=48,
                hidden_size=2048,
                intermediate_size=6144,
                num_attention_heads=32,
                num_key_value_heads=4,
                head_dim=128,
                q_lora_rank=512,
                kv_lora_rank=512,
                qk_nope_head_dim=64,
                qk_rope_head_dim=64,
                v_head_dim=128,
                num_experts=128,
                num_experts_per_tok=8,
                moe_intermediate_size=768,
                decoder_sparse_step=1,
                vocab_size=151936,
                model_type="qwen3_moe",
                tie_word_embeddings=False,
            ),
            "qwen3-8b": cls(
                num_hidden_layers=36,
                hidden_size=4096,
                intermediate_size=12288,
                num_attention_heads=32,
                num_key_value_heads=8,
                head_dim=128,
                q_lora_rank=512,
                kv_lora_rank=512,
                qk_nope_head_dim=64,
                qk_rope_head_dim=64,
                v_head_dim=128,
                num_experts=1,
                vocab_size=151936,
                model_type="qwen3",
                tie_word_embeddings=False,
                max_position_embeddings=4096,
            ),
            "qwen3-1.7b": cls(
                num_hidden_layers=28,
                hidden_size=2048,
                intermediate_size=6144,
                num_attention_heads=16,
                num_key_value_heads=8,
                head_dim=128,
                q_lora_rank=512,
                kv_lora_rank=512,
                qk_nope_head_dim=64,
                qk_rope_head_dim=64,
                v_head_dim=128,
                num_experts=1,
                vocab_size=151936,
                model_type="qwen3",
                tie_word_embeddings=True,
                max_position_embeddings=4096,
            ),
            "qwen3-14b": cls(
                num_hidden_layers=48,
                hidden_size=5120,
                intermediate_size=13824,
                num_attention_heads=40,
                num_key_value_heads=8,
                head_dim=128,
                num_experts=1,
                vocab_size=152064,
                model_type="qwen3",
                tie_word_embeddings=False,
                max_position_embeddings=131072,
            ),
            "qwen3-235b-a22b": cls(
                num_hidden_layers=94,
                hidden_size=9216,
                intermediate_size=24576,
                num_attention_heads=64,
                num_key_value_heads=8,
                head_dim=144,
                num_experts=128,
                num_experts_per_tok=8,
                moe_intermediate_size=3072,
                decoder_sparse_step=1,
                vocab_size=152064,
            ),
            # DeepSeek MoE 系列
            "deepseek-v3": cls(
                num_hidden_layers=61,
                hidden_size=7168,
                intermediate_size=18432,
                num_attention_heads=56,
                num_key_value_heads=8,
                head_dim=128,
                num_experts=256,
                num_experts_per_tok=8,
                moe_intermediate_size=2048,
                decoder_sparse_step=1,
                mlp_only_layers=[0, 1, 2],  # 前3层是 Dense
                vocab_size=129024,
            ),
            # Dense 模型
            "llama3-70b": cls(
                num_hidden_layers=80,
                hidden_size=8192,
                intermediate_size=28672,
                num_attention_heads=64,
                num_key_value_heads=8,
                head_dim=128,
                num_experts=1,
                vocab_size=128256,
            ),
            "llama3-8b": cls(
                num_hidden_layers=32,
                hidden_size=4096,
                intermediate_size=14336,
                num_attention_heads=32,
                num_key_value_heads=8,
                head_dim=128,
                num_experts=1,
                vocab_size=128256,
            ),
            # GLM4 MoE 系列
            "glm-4.5-air": cls(
                num_hidden_layers=46,
                hidden_size=4096,
                intermediate_size=10944,
                num_attention_heads=96,
                num_key_value_heads=8,
                head_dim=128,
                num_experts=128,          # routed experts
                num_shared_experts=1,
                num_experts_per_tok=8,
                moe_intermediate_size=1408,
                decoder_sparse_step=1,
                mlp_only_layers=[0],      # first_k_dense_replace = 1
                vocab_size=151552,
                max_position_embeddings=131072,
            ),
        }
        
        name_lower = name.lower().replace("_", "-").replace(" ", "-")

        candidate_names = [name_lower]

        # 兼容如 "huggingface/GLM-4.5-Air" 这种带命名空间的写法
        if "/" in name_lower:
            candidate_names.append(name_lower.split("/")[-1])

        # 常见目录/模型命名里会带上 Base 后缀，例如 Qwen3-30B-A3B-Base。
        # 这里把 "-base" 视作别名后缀，优先在内置 presets 中回退匹配。
        normalized_candidates = []
        for candidate in candidate_names:
            normalized_candidates.append(candidate)
            if candidate.endswith("-base"):
                normalized_candidates.append(candidate[: -len("-base")])
            elif candidate.endswith("base"):
                normalized_candidates.append(candidate[: -len("base")])

        for candidate in normalized_candidates:
            if candidate in presets:
                return presets[candidate]

        alias_presets = {
            "qwen8b": "qwen3-8b",
            "qwen38b": "qwen3-8b",
            "qwen317b": "qwen3-1.7b",
            "qwen31p7b": "qwen3-1.7b",
            "qwen17b": "qwen3-1.7b",
            "qwen1_7b": "qwen3-1.7b",
        }
        canonical_presets = {
            _canonical_model_name(preset_name): preset_name
            for preset_name in presets
        }
        for alias_name, preset_name in alias_presets.items():
            canonical_presets[_canonical_model_name(alias_name)] = preset_name
        for candidate in normalized_candidates:
            canonical = _canonical_model_name(candidate)
            matched_name = canonical_presets.get(canonical)
            if matched_name is not None:
                return presets[matched_name]
        
        raise ValueError(f"Unknown model: {name}. Available: {list(presets.keys())}")
    
    @classmethod
    def from_any(cls, path_or_name: str) -> "ModelConfig":
        """
        智能加载模型配置。

        支持三种输入:
        1. 内置模型名称 (如 "qwen3-30b-a3b") — 调用 from_name()
        2. 模型目录 (包含 config.json) — 调用 from_config_json()
        3. yaml/json 文件 — 加载为 dict 后调用 from_dict()

        自动判断输入类型，无需手动指定。
        """
        expanded = os.path.expanduser(path_or_name)

        # 如果路径不存在，尝试作为模型名称
        if not os.path.exists(expanded):
            try:
                return cls.from_name(path_or_name)
            except (ValueError, Exception):
                pass
            raise FileNotFoundError(
                f"模型配置文件不存在，也不是已知模型名: {path_or_name}"
            )

        # 如果是目录，查找 config.json
        if os.path.isdir(expanded):
            config_json = os.path.join(expanded, "config.json")
            if os.path.exists(config_json):
                return cls.from_config_json(expanded)
            raise FileNotFoundError(f"目录中未找到 config.json: {expanded}")

        # 文件路径：根据格式解析
        from .utils.io import load_dict_from_file
        data = load_dict_from_file(path_or_name)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        """从 config.json 加载"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_config_json(cls, path_or_dir: str) -> "ModelConfig":
        """
        只从真实模型 config.json 加载。

        `path_or_dir` 可以是：
        - config.json 文件路径
        - 包含 config.json 的模型目录
        """
        raw_path = os.path.abspath(os.path.expanduser(str(path_or_dir)))
        config_path = (
            os.path.join(raw_path, "config.json")
            if os.path.isdir(raw_path)
            else raw_path
        )
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"找不到 config.json: {config_path}")
        return cls.from_json(config_path)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ModelConfig":
        """从字典创建"""
        num_hidden_layers = int(data.get("num_hidden_layers", data.get("n_layer", 48)))
        hidden_size = int(data.get("hidden_size", data.get("n_embd", 6144)))
        num_attention_heads = int(data.get("num_attention_heads", data.get("n_head", 32)))
        num_key_value_heads = int(data.get("num_key_value_heads", data.get("n_kv_head", 4)))
        head_dim_fallback = hidden_size // max(1, num_attention_heads)
        head_dim = int(data.get("head_dim", head_dim_fallback))

        intermediate_size = int(
            data.get("intermediate_size",
                     data.get("ffn_hidden_size",
                              data.get("ffn_size", 16384)))
        )

        # HF/MoE 兼容字段：
        # - Qwen/DeepSeek 常见: num_experts
        # - GLM4 MoE 常见: n_routed_experts (+ n_shared_experts)
        # 这里按 routed experts 作为 EP 约束基数，避免 shared experts 影响可整除搜索。
        num_experts = int(
            data.get("num_experts",
                     data.get("n_routed_experts",
                              data.get("num_local_experts", 1)))
        )
        num_shared_experts = int(
            data.get("num_shared_experts", data.get("n_shared_experts", 0))
        )
        num_experts_per_tok = int(
            data.get("num_experts_per_tok",
                     data.get("moe_top_k",
                              data.get("top_k", 8)))
        )
        moe_intermediate_size = int(
            data.get("moe_intermediate_size",
                     data.get("expert_intermediate_size", intermediate_size))
        )
        shared_expert_intermediate_size = int(
            data.get("shared_expert_intermediate_size",
                     data.get("shared_expert_ffn_hidden_size",
                              num_shared_experts * moe_intermediate_size))
        )

        model_type = str(data.get("model_type", "")).lower()
        # 仅在 config.json 明确给出 MLA / LoRA 风格字段时，才启用低秩注意力参数化。
        # 普通 dense Qwen3（如 Qwen3-1.7B / 8B）在公开 Hugging Face 实现中使用
        # 标准 Q/K/V/O 投影，并额外带 q_norm / k_norm，而不是自动推断的低秩 MLA。
        default_sparse_step = 1 if ("moe" in model_type or num_experts > 1) else num_hidden_layers + 1
        decoder_sparse_step = int(data.get("decoder_sparse_step", default_sparse_step))

        mlp_only_layers = data.get("mlp_only_layers")
        if mlp_only_layers is None:
            first_k_dense = int(data.get("first_k_dense_replace", 0) or 0)
            mlp_only_layers = list(range(min(first_k_dense, num_hidden_layers)))

        return cls(
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            q_lora_rank=int(data.get("q_lora_rank") or 0),
            kv_lora_rank=int(data.get("kv_lora_rank") or 0),
            qk_nope_head_dim=int(data.get("qk_nope_head_dim") or 0),
            qk_rope_head_dim=int(data.get("qk_rope_head_dim") or 0),
            v_head_dim=int(data.get("v_head_dim") or 0),
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
            decoder_sparse_step=decoder_sparse_step,
            mlp_only_layers=mlp_only_layers,
            vocab_size=int(data.get("vocab_size", 152064)),
            max_position_embeddings=int(data.get("max_position_embeddings", 32768)),
            model_type=model_type,
            tie_word_embeddings=bool(
                data.get("tie_word_embeddings",
                         data.get("share_embeddings_and_output_weights",
                                  data.get("share_embedding_weights", num_experts <= 1))))
        )


@dataclass
class ParallelConfig:
    """
    并行配置
    
    与 PaddleFormers TrainingArguments 对应
    """
    # 张量并行
    tp: int = 1  # tensor_model_parallel_size
    
    # 流水线并行
    pp: int = 1  # pipeline_model_parallel_size
    vpp: int = 1  # virtual_pipeline_model_parallel_size
    stage_layer_counts: List[int] = field(default_factory=list)  # 自定义 PP stage 层数划分
    
    # 数据并行
    dp: int = 1  # 自动计算或显式设置
    
    # Sharding (ZeRO)
    sharding: str = "stage1"  # "none", "stage1", "stage2", "stage3"
    sharding_degree: int = -1  # -1 表示等于 dp
    
    # Expert 并行 (MoE)
    ep: int = 1  # expert_model_parallel_size
    
    # Sequence 并行
    sp: bool = False  # sequence_parallel
    
    # Context 并行
    cp: int = 1  # context_parallel_size
    
    @property
    def sharding_stage(self) -> ShardingStage:
        """获取 Sharding 阶段"""
        mapping = {
            "none": ShardingStage.NONE,
            "": ShardingStage.NONE,
            "stage1": ShardingStage.STAGE1,
            "stage2": ShardingStage.STAGE2,
            "stage3": ShardingStage.STAGE3,
        }
        return mapping.get(self.sharding.lower(), ShardingStage.STAGE1)
    
    @property
    def effective_sharding_degree(self) -> int:
        """有效的 Sharding 度"""
        if self.sharding_stage == ShardingStage.NONE:
            return 1
        if self.sharding_degree > 0:
            return self.sharding_degree
        return self.dp
    
    @property
    def world_size(self) -> int:
        """总 GPU 数"""
        return self.dp * self.tp * self.pp
    
    def validate(self, total_gpus: int) -> bool:
        """验证配置是否合法"""
        # 基本约束
        if self.tp < 1 or self.pp < 1 or self.dp < 1:
            return False
        
        # dense 主干 world size 匹配
        if self.dp * self.tp * self.pp != total_gpus:
            return False
        
        # EP 约束: EP 度数不能超过 Expert 数（由用户保证）
        # MoE sharding 通常 = world_size / (pp * ep)
        
        return True
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "tp": self.tp,
            "pp": self.pp,
            "vpp": self.vpp,
            "stage_layer_counts": list(self.stage_layer_counts),
            "dp": self.dp,
            "sharding": self.sharding,
            "sharding_degree": self.sharding_degree,
            "ep": self.ep,
            "sp": self.sp,
            "cp": self.cp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ParallelConfig":
        """从字典创建"""
        raw_stage_layer_counts = data.get(
            "stage_layer_counts",
            data.get("pp_stage_layer_counts", []),
        )
        if isinstance(raw_stage_layer_counts, str):
            stripped = raw_stage_layer_counts.strip()
            if stripped:
                stage_layer_counts = [
                    int(item.strip())
                    for item in stripped.split(",")
                    if item.strip()
                ]
            else:
                stage_layer_counts = []
        else:
            stage_layer_counts = [int(item) for item in (raw_stage_layer_counts or [])]
        return cls(
            tp=data.get("tp", data.get("tensor_model_parallel_size", 1)),
            pp=data.get("pp", data.get("pipeline_model_parallel_size", 1)),
            vpp=data.get("vpp", data.get("virtual_pipeline_model_parallel_size", 1)),
            stage_layer_counts=stage_layer_counts,
            dp=data.get("dp", data.get("data_parallel_size", 1)),
            sharding=data.get("sharding", "stage1"),
            sharding_degree=data.get("sharding_degree", data.get("sharding_parallel_size", -1)),
            ep=data.get("ep", data.get("expert_model_parallel_size", 1)),
            sp=data.get("sp", data.get("sequence_parallel", False)),
            cp=data.get("cp", data.get("context_parallel_size", 1)),
        )
    
    def __str__(self) -> str:
        parts = [f"TP{self.tp}", f"PP{self.pp}", f"DP{self.dp}"]
        if self.vpp > 1:
            parts.append(f"VPP{self.vpp}")
        if self.stage_layer_counts:
            parts.append("StageLayers(" + ",".join(str(v) for v in self.stage_layer_counts) + ")")
        if self.ep > 1:
            parts.append(f"EP{self.ep}")
        if self.sharding != "none" and self.sharding:
            parts.append(f"Sharding({self.sharding})")
        if self.sp:
            parts.append("SP")
        if self.cp > 1:
            parts.append(f"CP{self.cp}")
        return "-".join(parts)


@dataclass
class TrainingConfig:
    """
    训练配置
    
    与 PaddleFormers TrainingArguments 对应
    """
    # Batch 配置
    micro_batch_size: int = 1  # per_device_train_batch_size
    global_batch_size: int = 512
    gradient_accumulation_steps: int = 64
    
    # 序列长度
    sequence_length: int = 8192
    
    # 数据类型
    dtype: str = "bfloat16"  # "float32", "float16", "bfloat16"
    
    # 混合精度
    fp16_opt_level: str = "O2"  # "O0", "O1", "O2"
    amp_master_grad: bool = True
    
    # 重计算
    recompute_granularity: str = "full"  # "none", "selective", "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 1
    recompute_modules: List[str] = field(default_factory=list)

    # 显存优化
    tensorwise_offload_optimizer: bool = False
    tensorwise_offload_ratio: float = 0.95
    sd_release_grads: bool = False

    # pipeline layout / runtime
    num_empty_layers_add_in_head: int = 0
    num_empty_layers_add_in_tail: int = 0

    # 运行时通信 / overlap 行为
    overlap_p2p_comm: bool = False
    use_batch_p2p_comm: bool = True
    p2p_cache_shape: bool = False
    stage1_overlap: bool = False
    enable_sharding_comm_overlap: bool = False
    variable_seq_lengths: bool = False
    enable_dynamic_shape: bool = False
    clear_every_step_cache: bool = True
    best_unbalanced_scheduler: bool = False
    hybrid_parallel_topo_order: str = ""

    # 算子 / runtime 细节
    attn_implementation: str = ""
    apply_rope_fusion: bool = False
    use_qk_norm: bool = False
    moe_token_dispatcher_type: str = "deepep"
    moe_grouped_gemm: bool = False
    moe_router_fusion: bool = False
    moe_expert_fusion: bool = True
    moe_shared_expert_overlap: bool = False
    moe_ep_barrier: bool = True
    
    @property
    def dtype_bytes(self) -> int:
        """数据类型字节数"""
        return {"float32": 4, "float16": 2, "bfloat16": 2}.get(self.dtype, 2)
    
    @property
    def recompute_config(self) -> "RecomputeGranularity":
        """获取重计算配置枚举"""
        mapping = {
            "none": RecomputeGranularity.NONE,
            "": RecomputeGranularity.NONE,
            "selective": RecomputeGranularity.SELECTIVE,
            "full": RecomputeGranularity.FULL,
        }
        return mapping.get(self.recompute_granularity.lower(), RecomputeGranularity.FULL)
    
    def to_dict(self) -> Dict:
        return {
            "micro_batch_size": self.micro_batch_size,
            "global_batch_size": self.global_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "sequence_length": self.sequence_length,
            "dtype": self.dtype,
            "fp16_opt_level": self.fp16_opt_level,
            "amp_master_grad": self.amp_master_grad,
            "recompute_granularity": self.recompute_granularity,
            "recompute_method": self.recompute_method,
            "recompute_num_layers": self.recompute_num_layers,
            "recompute_modules": list(self.recompute_modules),
            "tensorwise_offload_optimizer": self.tensorwise_offload_optimizer,
            "tensorwise_offload_ratio": self.tensorwise_offload_ratio,
            "sd_release_grads": self.sd_release_grads,
            "num_empty_layers_add_in_head": self.num_empty_layers_add_in_head,
            "num_empty_layers_add_in_tail": self.num_empty_layers_add_in_tail,
            "overlap_p2p_comm": self.overlap_p2p_comm,
            "use_batch_p2p_comm": self.use_batch_p2p_comm,
            "p2p_cache_shape": self.p2p_cache_shape,
            "stage1_overlap": self.stage1_overlap,
            "enable_sharding_comm_overlap": self.enable_sharding_comm_overlap,
            "variable_seq_lengths": self.variable_seq_lengths,
            "enable_dynamic_shape": self.enable_dynamic_shape,
            "clear_every_step_cache": self.clear_every_step_cache,
            "best_unbalanced_scheduler": self.best_unbalanced_scheduler,
            "hybrid_parallel_topo_order": self.hybrid_parallel_topo_order,
            "attn_implementation": self.attn_implementation,
            "apply_rope_fusion": self.apply_rope_fusion,
            "use_qk_norm": self.use_qk_norm,
            "moe_token_dispatcher_type": self.moe_token_dispatcher_type,
            "moe_grouped_gemm": self.moe_grouped_gemm,
            "moe_router_fusion": self.moe_router_fusion,
            "moe_expert_fusion": self.moe_expert_fusion,
            "moe_shared_expert_overlap": self.moe_shared_expert_overlap,
            "moe_ep_barrier": self.moe_ep_barrier,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "TrainingConfig":
        raw_dtype = data.get("dtype")
        if raw_dtype is None:
            if bool(data.get("bf16", False)):
                raw_dtype = "bfloat16"
            elif bool(data.get("fp16", False)):
                raw_dtype = "float16"
            else:
                raw_dtype = "bfloat16"

        raw_recompute_modules = data.get("recompute_modules", [])
        if isinstance(raw_recompute_modules, str):
            stripped = raw_recompute_modules.strip()
            if not stripped:
                recompute_modules = []
            elif stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = ast.literal_eval(stripped)
                except (SyntaxError, ValueError):
                    parsed = [item.strip() for item in stripped.strip("[]").split(",")]
                recompute_modules = [str(item).strip().strip("'\"") for item in parsed if str(item).strip()]
            else:
                recompute_modules = [item.strip() for item in stripped.split(",") if item.strip()]
        else:
            recompute_modules = list(raw_recompute_modules)

        raw_batch_p2p = data.get("use_batch_p2p_comm", data.get("batch_p2p_comm"))
        if raw_batch_p2p is None and "non_batch_p2p_comm" in data:
            raw_batch_p2p = not bool(data.get("non_batch_p2p_comm"))

        return cls(
            micro_batch_size=data.get("micro_batch_size", data.get("per_device_train_batch_size", 1)),
            global_batch_size=data.get("global_batch_size", 512),
            gradient_accumulation_steps=data.get("gradient_accumulation_steps", 64),
            sequence_length=data.get(
                "sequence_length",
                data.get("max_sequence_length", data.get("max_seq_len", 8192)),
            ),
            dtype=str(raw_dtype),
            fp16_opt_level=data.get("fp16_opt_level", "O2"),
            amp_master_grad=data.get("amp_master_grad", True),
            recompute_granularity=data.get("recompute_granularity", "full"),
            recompute_method=data.get("recompute_method", "uniform"),
            recompute_num_layers=data.get("recompute_num_layers", 1),
            recompute_modules=recompute_modules,
            tensorwise_offload_optimizer=bool(
                data.get("tensorwise_offload_optimizer", False)
            ),
            tensorwise_offload_ratio=float(
                data.get("tensorwise_offload_ratio", 0.95)
            ),
            sd_release_grads=bool(data.get("sd_release_grads", False)),
            num_empty_layers_add_in_head=int(
                data.get("num_empty_layers_add_in_head", 0)
            ),
            num_empty_layers_add_in_tail=int(
                data.get("num_empty_layers_add_in_tail", 0)
            ),
            overlap_p2p_comm=bool(data.get("overlap_p2p_comm", False)),
            use_batch_p2p_comm=bool(
                True if raw_batch_p2p is None else raw_batch_p2p
            ),
            p2p_cache_shape=bool(data.get("p2p_cache_shape", False)),
            stage1_overlap=bool(data.get("stage1_overlap", False)),
            enable_sharding_comm_overlap=bool(
                data.get(
                    "enable_sharding_comm_overlap",
                    data.get("sharding_comm_overlap", False),
                )
            ),
            variable_seq_lengths=bool(data.get("variable_seq_lengths", False)),
            enable_dynamic_shape=bool(data.get("enable_dynamic_shape", False)),
            clear_every_step_cache=bool(data.get("clear_every_step_cache", True)),
            best_unbalanced_scheduler=bool(
                data.get("best_unbalanced_scheduler", False)
            ),
            hybrid_parallel_topo_order=str(
                data.get("hybrid_parallel_topo_order", "")
            ),
            attn_implementation=str(
                data.get(
                    "attn_implementation",
                    data.get("_attn_implementation", ""),
                )
            ),
            apply_rope_fusion=bool(data.get("apply_rope_fusion", False)),
            moe_token_dispatcher_type=str(
                data.get("moe_token_dispatcher_type", "deepep")
            ),
            moe_grouped_gemm=bool(data.get("moe_grouped_gemm", False)),
            moe_router_fusion=bool(data.get("moe_router_fusion", False)),
            moe_expert_fusion=bool(data.get("moe_expert_fusion", True)),
            moe_shared_expert_overlap=bool(
                data.get("moe_shared_expert_overlap", False)
            ),
            moe_ep_barrier=bool(data.get("moe_ep_barrier", True)),
        )

