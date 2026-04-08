#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
硬件校准模块 - 自动检测、校准GPU性能并管理校准结果

注意：
- hardware_spec.py 与本文件在同一目录
- hardware_spec.py 是"纯配置文件"，只包含一行：HARDWARE_SPECS = {...}
- 通过 gpu_name + gpus_per_node 生成 key，O(1) 定位配置
- key 格式: {GPU}_{N}gpn (如 H800_8gpn)，与集群节点数无关
- 同型号机器只需校准一次，任意节点数均可复用
"""

import os
import time
import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)

_NETWORK_CURVE_SIZES_BYTES = [
    1 << 10,   # 1 KiB
    4 << 10,   # 4 KiB
    16 << 10,  # 16 KiB
    64 << 10,  # 64 KiB
    256 << 10, # 256 KiB
    1 << 20,   # 1 MiB
    4 << 20,   # 4 MiB
    16 << 20,  # 16 MiB
    64 << 20,  # 64 MiB
    256 << 20, # 256 MiB
]

# 尝试导入 paddle
_PADDLE_AVAILABLE = False
try:
    import paddle
    _PADDLE_AVAILABLE = True
except ImportError:
    pass

# 你原本的配置类型（保持不动）
from ..config import GPUSpec, NetworkSpec, NetworkBandwidthCurve, HardwareConfig


# ============================================================
# 路径：hardware_spec.py 与 calibration.py 同目录
# ============================================================
_THIS_DIR = Path(__file__).parent
HARDWARE_SPEC_PATH = _THIS_DIR / "hardware_spec.py"


# ============================================================
# 索引 key：gpu_name + gpus_per_node -> key
#
# 设计原则：
#   - 校准数据取决于"这类机器"，不取决于"集群规模"
#   - 计算性能（GEMM、显存带宽）：同型号 GPU 完全相同
#   - 网络性能（NVLink/IB）：同拓扑链路完全相同
#   - 因此：只要 GPU 型号 + 每节点 GPU 数相同，校准数据通用
#   - node_count 是运行时参数，不编入校准 key
#
# key 格式：{GPU}_{N}gpn   例: H800_8gpn
#   表示"8卡 H800 节点"的校准数据，1节点/4节点/100节点均可复用
# ============================================================
def _clean_gpu_name(gpu_name: str) -> str:
    """把 GPU 名字清洗成稳定 key（与之前文件名规则相近）"""
    clean = (gpu_name or "Unknown").replace("NVIDIA ", "").replace(" ", "_").replace("-", "_")
    while "__" in clean:
        clean = clean.replace("__", "_")
    clean = clean.strip("_")
    return clean if clean else "Unknown"


def _make_spec_key(gpu_name: str, gpus_per_node: int) -> str:
    """
    用于在 HARDWARE_SPECS 中快速定位的 key。

    key = {GPU}_{N}gpn  (gpn = GPUs Per Node)
    例: H800_8gpn
    """
    return f"{_clean_gpu_name(gpu_name)}_{int(gpus_per_node)}gpn"


def _make_legacy_key(gpu_name: str, gpu_count: int, node_count: int = 1) -> str:
    """旧版 key 格式，用于向后兼容加载"""
    return f"{_clean_gpu_name(gpu_name)}_{int(gpu_count)}gpu_{int(node_count)}node"


# ============================================================
# 读写 hardware_spec.py（纯字典文件）
# ============================================================
def _ensure_hardware_spec_file_exists():
    """确保 hardware_spec.py 存在且格式为纯字典"""
    if HARDWARE_SPEC_PATH.exists():
        return
    # 只写一行，保证“纯配置”
    HARDWARE_SPEC_PATH.write_text("HARDWARE_SPECS = {}\n", encoding="utf-8")


def _load_hardware_specs_table() -> Dict[str, Dict]:
    """
    从 hardware_spec.py 加载 HARDWARE_SPECS 字典。
    要求：hardware_spec.py 只包含 HARDWARE_SPECS = {...}
    """
    _ensure_hardware_spec_file_exists()

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("hardware_spec", HARDWARE_SPEC_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        table = getattr(module, "HARDWARE_SPECS", None)
        return table if isinstance(table, dict) else {}
    except Exception as e:
        logger.warning(f"Failed to load {HARDWARE_SPEC_PATH}: {e}")
        return {}


def _format_py_dict(data: Dict, indent: int = 0) -> str:
    """把 dict 格式化成可读的 Python 字典文本（不依赖 json）"""
    lines = ["{"]
    for k, v in data.items():
        prefix = "    " * (indent + 1)
        if isinstance(v, dict):
            lines.append(f'{prefix}"{k}": {_format_py_dict(v, indent + 1)},')
        elif isinstance(v, list):
            if len(v) == 0:
                lines.append(f'{prefix}"{k}": [],')
            elif isinstance(v[0], dict):
                lines.append(f'{prefix}"{k}": [')
                for item in v:
                    lines.append(f'{"    " * (indent + 2)}{_format_py_dict(item, indent + 2)},')
                lines.append(f"{prefix}],")
            else:
                lines.append(f'{prefix}"{k}": {repr(v)},')
        elif isinstance(v, str):
            # 避免引号破坏语法
            safe = v.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{prefix}"{k}": "{safe}",')
        elif isinstance(v, bool):
            lines.append(f'{prefix}"{k}": {v},')
        elif isinstance(v, (int, float)):
            lines.append(f'{prefix}"{k}": {v},')
        else:
            lines.append(f'{prefix}"{k}": {repr(v)},')
    lines.append("    " * indent + "}")
    return "\n".join(lines)


def _write_hardware_specs_table(table: Dict[str, Dict]):
    """
    写回 hardware_spec.py
    要求：hardware_spec.py 只包含 HARDWARE_SPECS = {...}（不写任何其它内容）
    """
    # 保持 diff 稳定：按 key 排序
    ordered = {k: table[k] for k in sorted(table.keys())}

    content = "HARDWARE_SPECS = " + _format_py_dict(ordered) + "\n"

    # 原子写：先写 tmp 再替换
    tmp_path = HARDWARE_SPEC_PATH.with_suffix(".py.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, HARDWARE_SPEC_PATH)


# 覆盖常见 LLM 训练 kernel 的代表性 BF16 GEMM 形状。
# 方阵曲线继续保留，下面这些点用于补足非方阵/小 batch 的查询。
_BF16_WORKLOAD_GEMM_SHAPES = [
    ("kv_proj", [(256, 512, 2048), (1024, 512, 2048), (4096, 512, 2048), (8192, 512, 2048)]),
    ("dense_ffn_up", [(256, 6144, 2048), (1024, 6144, 2048), (4096, 6144, 2048), (8192, 6144, 2048)]),
    ("dense_ffn_down", [(256, 2048, 6144), (1024, 2048, 6144), (4096, 2048, 6144), (8192, 2048, 6144)]),
    ("router", [(256, 128, 2048), (1024, 128, 2048), (4096, 128, 2048), (8192, 128, 2048)]),
    ("expert_ffn_up", [(16, 768, 2048), (64, 768, 2048), (256, 768, 2048), (1024, 768, 2048)]),
    ("expert_ffn_down", [(16, 2048, 768), (64, 2048, 768), (256, 2048, 768), (1024, 2048, 768)]),
]


# ============================================================
# 数据结构
# ============================================================
@dataclass
class PerformancePoint:
    """单个性能测试点"""
    size: int
    tflops: float
    efficiency: float
    time_ms: float


@dataclass
class PerformanceCurve:
    """性能曲线（多尺寸测试结果）"""
    dtype: str
    points: List[PerformancePoint]
    peak_tflops: float

    # 拟合参数 (efficiency = a * log(size) + b)
    fit_a: float = 0.0
    fit_b: float = 0.0
    fit_max: float = 1.0

    def predict_efficiency(self, size: int) -> float:
        import math
        if size <= 0:
            return 0.0
        efficiency = self.fit_a * math.log(max(size, 1)) + self.fit_b
        return max(0.01, min(self.fit_max, efficiency))

    def predict_tflops(self, size: int) -> float:
        return self.peak_tflops * self.predict_efficiency(size)

    def to_dict(self) -> Dict:
        return {
            "dtype": self.dtype,
            "peak_tflops": round(self.peak_tflops, 2),
            "fit_a": round(self.fit_a, 6),
            "fit_b": round(self.fit_b, 6),
            "fit_max": round(self.fit_max, 4),
            "points": [
                {
                    "size": int(p.size),
                    "tflops": round(float(p.tflops), 2),
                    "efficiency": round(float(p.efficiency), 4),
                    "time_ms": round(float(p.time_ms), 3),
                }
                for p in self.points
            ],
        }


@dataclass
class CalibrationResult:
    """校准结果"""
    gpu_name: str = "Unknown"
    gpu_memory_gb: float = 0.0
    gpu_count: int = 0

    fp32_tflops: float = 0.0
    fp16_tflops: float = 0.0
    bf16_tflops: float = 0.0

    memory_bandwidth_gbps: float = 0.0
    host_to_device_bandwidth_gbps: float = 0.0
    device_to_host_bandwidth_gbps: float = 0.0
    intra_node_bandwidth_gbps: float = 0.0

    # 多机通信校准字段
    inter_node_bandwidth_gbps: float = 0.0   # 节点间实测带宽 (GB/s)
    intra_node_latency_us: float = 0.0       # 节点内实测延迟 (μs)
    inter_node_latency_us: float = 0.0       # 节点间实测延迟 (μs)

    bf16_curve: Optional[PerformanceCurve] = None
    bf16_gemm_samples: Optional[List[Dict[str, Any]]] = None

    # 多尺寸带宽曲线
    host_to_device_bw_curve: Optional[NetworkBandwidthCurve] = None
    device_to_host_bw_curve: Optional[NetworkBandwidthCurve] = None
    intra_node_bw_curve: Optional[NetworkBandwidthCurve] = None
    inter_node_bw_curve: Optional[NetworkBandwidthCurve] = None

    calibrated: bool = False
    network_calibrated: bool = False          # 网络是否经过实测校准
    error_message: str = ""

    def to_dict(self) -> Dict:
        result = {
            "gpu_name": self.gpu_name,
            "gpu_memory_gb": round(self.gpu_memory_gb, 2),
            "gpu_count": int(self.gpu_count),
            "fp32_tflops": round(self.fp32_tflops, 2),
            "fp16_tflops": round(self.fp16_tflops, 2),
            "bf16_tflops": round(self.bf16_tflops, 2),
            "memory_bandwidth_gbps": round(self.memory_bandwidth_gbps, 2),
            "host_to_device_bandwidth_gbps": round(self.host_to_device_bandwidth_gbps, 2),
            "device_to_host_bandwidth_gbps": round(self.device_to_host_bandwidth_gbps, 2),
            "intra_node_bandwidth_gbps": round(self.intra_node_bandwidth_gbps, 2),
            "inter_node_bandwidth_gbps": round(self.inter_node_bandwidth_gbps, 2),
            "intra_node_latency_us": round(self.intra_node_latency_us, 3),
            "inter_node_latency_us": round(self.inter_node_latency_us, 3),
            "calibrated": bool(self.calibrated),
            "network_calibrated": bool(self.network_calibrated),
        }
        if self.bf16_curve:
            result["bf16_curve"] = self.bf16_curve.to_dict()
        if self.bf16_gemm_samples:
            result["bf16_gemm_samples"] = [
                {
                    "family": str(s.get("family", "unknown")),
                    "m": int(s.get("m", 0)),
                    "n": int(s.get("n", 0)),
                    "k": int(s.get("k", 0)),
                    "tflops": round(float(s.get("tflops", 0.0)), 2),
                    "efficiency": round(float(s.get("efficiency", 0.0)), 4),
                    "time_ms": round(float(s.get("time_ms", 0.0)), 3),
                }
                for s in self.bf16_gemm_samples
                if isinstance(s, dict)
            ]
        if self.intra_node_bw_curve is not None:
            result["intra_node_bw_curve"] = self.intra_node_bw_curve.to_dict()
        if self.inter_node_bw_curve is not None:
            result["inter_node_bw_curve"] = self.inter_node_bw_curve.to_dict()
        if self.host_to_device_bw_curve is not None:
            result["host_to_device_bw_curve"] = self.host_to_device_bw_curve.to_dict()
        if self.device_to_host_bw_curve is not None:
            result["device_to_host_bw_curve"] = self.device_to_host_bw_curve.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> "CalibrationResult":
        result = cls(
            gpu_name=data.get("gpu_name", "Unknown"),
            gpu_memory_gb=float(data.get("gpu_memory_gb", 0.0)),
            gpu_count=int(data.get("gpu_count", 0)),
            fp32_tflops=float(data.get("fp32_tflops", 0.0)),
            fp16_tflops=float(data.get("fp16_tflops", 0.0)),
            bf16_tflops=float(data.get("bf16_tflops", 0.0)),
            memory_bandwidth_gbps=float(data.get("memory_bandwidth_gbps", 0.0)),
            host_to_device_bandwidth_gbps=float(
                data.get("host_to_device_bandwidth_gbps", 0.0)
            ),
            device_to_host_bandwidth_gbps=float(
                data.get("device_to_host_bandwidth_gbps", 0.0)
            ),
            intra_node_bandwidth_gbps=float(data.get("intra_node_bandwidth_gbps", 0.0)),
            inter_node_bandwidth_gbps=float(data.get("inter_node_bandwidth_gbps", 0.0)),
            intra_node_latency_us=float(data.get("intra_node_latency_us", 0.0)),
            inter_node_latency_us=float(data.get("inter_node_latency_us", 0.0)),
            calibrated=bool(data.get("calibrated", False)),
            network_calibrated=bool(data.get("network_calibrated", False)),
        )

        if "bf16_curve" in data and isinstance(data["bf16_curve"], dict):
            curve_data = data["bf16_curve"]
            points = []
            for p in curve_data.get("points", []):
                if not isinstance(p, dict):
                    continue
                points.append(
                    PerformancePoint(
                        size=int(p.get("size", 0)),
                        tflops=float(p.get("tflops", 0.0)),
                        efficiency=float(p.get("efficiency", 0.0)),
                        time_ms=float(p.get("time_ms", 0.0)),
                    )
                )
            result.bf16_curve = PerformanceCurve(
                dtype=str(curve_data.get("dtype", "bfloat16")),
                points=points,
                peak_tflops=float(curve_data.get("peak_tflops", 0.0)),
                fit_a=float(curve_data.get("fit_a", 0.0)),
                fit_b=float(curve_data.get("fit_b", 0.0)),
                fit_max=float(curve_data.get("fit_max", 1.0)),
            )

        if isinstance(data.get("bf16_gemm_samples"), list):
            samples = []
            for sample in data["bf16_gemm_samples"]:
                if not isinstance(sample, dict):
                    continue
                samples.append(
                    {
                        "family": str(sample.get("family", "unknown")),
                        "m": int(sample.get("m", 0)),
                        "n": int(sample.get("n", 0)),
                        "k": int(sample.get("k", 0)),
                        "tflops": float(sample.get("tflops", 0.0)),
                        "efficiency": float(sample.get("efficiency", 0.0)),
                        "time_ms": float(sample.get("time_ms", 0.0)),
                    }
                )
            result.bf16_gemm_samples = samples or None

        if "intra_node_bw_curve" in data and isinstance(data["intra_node_bw_curve"], dict):
            result.intra_node_bw_curve = NetworkBandwidthCurve.from_dict(data["intra_node_bw_curve"])
        if "inter_node_bw_curve" in data and isinstance(data["inter_node_bw_curve"], dict):
            result.inter_node_bw_curve = NetworkBandwidthCurve.from_dict(data["inter_node_bw_curve"])
        if "host_to_device_bw_curve" in data and isinstance(data["host_to_device_bw_curve"], dict):
            result.host_to_device_bw_curve = NetworkBandwidthCurve.from_dict(
                data["host_to_device_bw_curve"]
            )
        if "device_to_host_bw_curve" in data and isinstance(data["device_to_host_bw_curve"], dict):
            result.device_to_host_bw_curve = NetworkBandwidthCurve.from_dict(
                data["device_to_host_bw_curve"]
            )

        return result


# ============================================================
# 校准器
# ============================================================
class HardwareCalibrator:
    """
    使用示例:
        calibrator = HardwareCalibrator()
        config = calibrator.auto_calibrate(node_count=1, force=True)
    """

    def __init__(self, device_id: int = 0, warmup_iters: int = 5, test_iters: int = 20):
        self.device_id = device_id
        self.warmup_iters = warmup_iters
        self.test_iters = test_iters
        self._result: Optional[CalibrationResult] = None

        _THIS_DIR.mkdir(parents=True, exist_ok=True)
        _ensure_hardware_spec_file_exists()

    @property
    def result(self) -> Optional[CalibrationResult]:
        return self._result

    def detect_gpu_info(self) -> Tuple[str, float, int]:
        """检测 GPU 信息: (gpu_name, memory_gb, gpu_count)"""
        gpu_name = "Unknown"
        memory_gb = 0.0
        gpu_count = 0

        # nvidia-smi
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if r.returncode == 0:
                lines = [x.strip() for x in r.stdout.strip().split("\n") if x.strip()]
                gpu_count = len(lines)
                if lines:
                    parts = [x.strip() for x in lines[0].split(",")]
                    gpu_name = parts[0]
                    # memory.total 单位 MiB
                    memory_gb = float(parts[1]) / 1024.0
        except Exception:
            pass

        # fallback: paddle
        if _PADDLE_AVAILABLE and gpu_count == 0:
            try:
                gpu_count = paddle.device.cuda.device_count()
                if gpu_count > 0:
                    props = paddle.device.cuda.get_device_properties(self.device_id)
                    gpu_name = props.name
                    memory_gb = props.total_memory / (1024 ** 3)
            except Exception:
                pass

        return gpu_name, float(memory_gb), int(gpu_count)

    def _benchmark_gemm(self, m: int, n: int, k: int, dtype: str = "bfloat16") -> Tuple[float, float]:
        """GEMM 测试，返回 (tflops, elapsed_ms)"""
        if not _PADDLE_AVAILABLE:
            return 0.0, 0.0

        try:
            paddle.set_device(f"gpu:{self.device_id}")

            dtype_map = {
                "float32": paddle.float32,
                "float16": paddle.float16,
                "bfloat16": paddle.bfloat16,
            }
            pd_dtype = dtype_map.get(dtype, paddle.bfloat16)

            a = paddle.randn([m, k], dtype=pd_dtype)
            b = paddle.randn([k, n], dtype=pd_dtype)

            for _ in range(self.warmup_iters):
                c = paddle.matmul(a, b)
                paddle.device.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(self.test_iters):
                c = paddle.matmul(a, b)
            paddle.device.cuda.synchronize()
            end = time.perf_counter()

            elapsed_ms = (end - start) * 1000.0 / self.test_iters
            flops = 2.0 * m * n * k
            tflops = flops / (elapsed_ms / 1000.0) / 1e12

            del a, b, c
            paddle.device.cuda.empty_cache()

            return float(tflops), float(elapsed_ms)
        except Exception as e:
            logger.warning(f"GEMM benchmark failed: {e}")
            return 0.0, 0.0

    def _benchmark_memory_bandwidth(self, size_mb: int = 256) -> float:
        """显存带宽测试，返回 GB/s（clone 近似 memcpy）"""
        if not _PADDLE_AVAILABLE:
            return 0.0

        try:
            paddle.set_device(f"gpu:{self.device_id}")

            num_elements = size_mb * 1024 * 1024 // 4
            src = paddle.randn([num_elements], dtype=paddle.float32)

            for _ in range(self.warmup_iters):
                dst = src.clone()
                paddle.device.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(self.test_iters):
                dst = src.clone()
            paddle.device.cuda.synchronize()
            end = time.perf_counter()

            elapsed_s = (end - start) / self.test_iters

            # 读+写 => 2x
            data_gb = (size_mb / 1024.0) * 2.0
            bw = data_gb / elapsed_s

            del src, dst
            paddle.device.cuda.empty_cache()

            return float(bw)
        except Exception as e:
            logger.warning(f"Memory bandwidth benchmark failed: {e}")
            return 0.0

    def _benchmark_host_device_profile(self,
                                       size_bytes_list: List[int]) -> Dict[str, Any]:
        """
        测量 CPU pinned memory <-> GPU 的传输带宽曲线。

        返回:
          - host_to_device_bandwidth_gbps
          - device_to_host_bandwidth_gbps
          - host_to_device_bw_curve
          - device_to_host_bw_curve
        """
        result: Dict[str, Any] = {
            "host_to_device_bandwidth_gbps": 0.0,
            "device_to_host_bandwidth_gbps": 0.0,
            "host_to_device_bw_curve": None,
            "device_to_host_bw_curve": None,
        }
        if not _PADDLE_AVAILABLE:
            return result

        try:
            import numpy as np

            paddle.set_device(f"gpu:{self.device_id}")
            gpu_place = paddle.CUDAPlace(self.device_id)
            pinned_place = paddle.CUDAPinnedPlace()
        except Exception as e:
            logger.warning(f"Host-device benchmark init failed: {e}")
            return result

        h2d_points: List[Dict[str, float]] = []
        d2h_points: List[Dict[str, float]] = []

        def _make_tensor(array, place):
            try:
                return paddle.to_tensor(array, dtype=paddle.float32, place=place)
            except TypeError:
                try:
                    return paddle.to_tensor(array, dtype=paddle.float32, device=place)
                except TypeError:
                    return paddle.to_tensor(array, dtype=paddle.float32)

        def _copy_tensor(src, dst_place, blocking: bool):
            try:
                return src.to(device=dst_place, blocking=blocking)
            except TypeError:
                try:
                    return src.to(dst_place, blocking=blocking)
                except TypeError:
                    return src.to(dst_place)

        def _measure_copy_bandwidth(src, dst_place, size_bytes: int,
                                    warmup_iters: int, test_iters: int) -> float:
            samples = []
            dst = None
            for _ in range(3):
                try:
                    for _ in range(warmup_iters):
                        dst = _copy_tensor(src, dst_place, blocking=False)
                    paddle.device.cuda.synchronize(self.device_id)

                    start = time.perf_counter()
                    for _ in range(test_iters):
                        dst = _copy_tensor(src, dst_place, blocking=False)
                    paddle.device.cuda.synchronize(self.device_id)
                    elapsed_s = (time.perf_counter() - start) / max(1, test_iters)
                    if elapsed_s > 0:
                        data_gb = float(size_bytes) / (1024 ** 3)
                        samples.append(data_gb / elapsed_s)
                except Exception:
                    return 0.0
            if dst is not None:
                del dst
            return self._median(samples)

        try:
            for size_bytes in size_bytes_list:
                num_elements = max(1, int((size_bytes + 3) // 4))
                host_np = np.ones([num_elements], dtype=np.float32)
                host_tensor = _make_tensor(host_np, pinned_place)
                gpu_tensor = _copy_tensor(host_tensor, gpu_place, blocking=True)
                paddle.device.cuda.synchronize(self.device_id)

                warmup_iters = self._network_curve_warmup_iters(size_bytes)
                test_iters = self._network_curve_test_iters(size_bytes)

                h2d_bw = _measure_copy_bandwidth(
                    host_tensor, gpu_place, size_bytes, warmup_iters, test_iters
                )
                d2h_bw = _measure_copy_bandwidth(
                    gpu_tensor, pinned_place, size_bytes, warmup_iters, test_iters
                )

                if h2d_bw > 0:
                    h2d_points.append(
                        {
                            "size_bytes": int(size_bytes),
                            "bandwidth_gbps": float(h2d_bw),
                        }
                    )
                if d2h_bw > 0:
                    d2h_points.append(
                        {
                            "size_bytes": int(size_bytes),
                            "bandwidth_gbps": float(d2h_bw),
                        }
                    )

                del host_tensor, gpu_tensor
                paddle.device.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Host-device bandwidth benchmark failed: {e}")
            return result

        if h2d_points:
            result["host_to_device_bandwidth_gbps"] = max(
                p["bandwidth_gbps"] for p in h2d_points
            )
            result["host_to_device_bw_curve"] = NetworkBandwidthCurve.fit_from_points(
                h2d_points
            )
        if d2h_points:
            result["device_to_host_bandwidth_gbps"] = max(
                p["bandwidth_gbps"] for p in d2h_points
            )
            result["device_to_host_bw_curve"] = NetworkBandwidthCurve.fit_from_points(
                d2h_points
            )
        return result

    def _fit_curve(self, curve: PerformanceCurve):
        """拟合性能曲线: efficiency = a * log(size) + b"""
        import math

        if not curve.points:
            return

        xs = []
        ys = []
        for p in curve.points:
            if p.size > 0 and p.efficiency > 0:
                xs.append(math.log(p.size))
                ys.append(p.efficiency)

        if len(xs) < 2:
            return

        n = len(xs)
        sum_x = sum(xs)
        sum_y = sum(ys)
        sum_xy = sum(x * y for x, y in zip(xs, ys))
        sum_xx = sum(x * x for x in xs)

        denom = n * sum_xx - sum_x * sum_x
        if abs(denom) < 1e-10:
            curve.fit_a = 0.0
            curve.fit_b = sum_y / n
        else:
            curve.fit_a = (n * sum_xy - sum_x * sum_y) / denom
            curve.fit_b = (sum_y - curve.fit_a * sum_x) / n

        max_measured = max(p.efficiency for p in curve.points)
        max_fitted = curve.fit_a * math.log(16384) + curve.fit_b
        curve.fit_max = float(min(max_measured * 1.05, max_fitted, 1.0))

    @staticmethod
    def _median(values: List[float]) -> float:
        """从多次测量中取中位数（偏保守的代表值）"""
        valid = sorted(v for v in values if v > 0)
        if not valid:
            return 0.0
        n = len(valid)
        if n % 2 == 0:
            return (valid[n // 2 - 1] + valid[n // 2]) / 2.0
        return valid[n // 2]

    @staticmethod
    def _network_curve_test_iters(size_bytes: int) -> int:
        """按消息大小自适应选择测试轮数。"""
        if size_bytes <= (4 << 10):
            return 200
        if size_bytes <= (64 << 10):
            return 120
        if size_bytes <= (1 << 20):
            return 80
        if size_bytes <= (16 << 20):
            return 40
        if size_bytes <= (64 << 20):
            return 20
        return 10

    @staticmethod
    def _network_curve_warmup_iters(size_bytes: int) -> int:
        """按消息大小自适应选择 warmup 轮数。"""
        if size_bytes <= (64 << 10):
            return 8
        if size_bytes <= (16 << 20):
            return 5
        return 3

    def _benchmark_allreduce_group_profile(self,
                                           dist,
                                           group,
                                           group_size: int,
                                           size_bytes_list: List[int]) -> Dict[str, Any]:
        """
        对一个通信 group 采集多消息尺寸 AllReduce 画像。

        返回:
          - collective_latency_us: 极小消息下单次 collective 的平均时延
          - alpha_us: 单个 ring step 的启动时延
          - bandwidth_gbps: 拟合得到的基础传输带宽峰值
          - bw_curve: 基于 transfer bandwidth 的多尺寸曲线
        """
        result: Dict[str, Any] = {
            "collective_latency_us": 0.0,
            "alpha_us": 0.0,
            "bandwidth_gbps": 0.0,
            "bw_curve": None,
        }

        if group is None or group_size <= 1:
            return result

        ring_factor = 2.0 * (group_size - 1) / group_size
        num_steps = max(1, 2 * (group_size - 1))
        allreduce_efficiency = max(1e-6, NetworkSpec().allreduce_efficiency)

        tiny = paddle.ones([1], dtype=paddle.float32)
        for _ in range(max(self.warmup_iters, 5)):
            dist.all_reduce(tiny, group=group)
        paddle.device.synchronize()

        lat_samples = []
        for _ in range(3):
            lat_iters = 80
            start = time.perf_counter()
            for _ in range(lat_iters):
                dist.all_reduce(tiny, group=group)
            paddle.device.synchronize()
            lat_samples.append((time.perf_counter() - start) / lat_iters)
        collective_latency_s = self._median(lat_samples)
        alpha_s = collective_latency_s / num_steps
        result["collective_latency_us"] = collective_latency_s * 1e6
        result["alpha_us"] = alpha_s * 1e6

        del tiny

        points: List[Dict[str, float]] = []
        for size_bytes in size_bytes_list:
            num_elements = max(1, int((size_bytes + 3) // 4))
            tensor = paddle.ones([num_elements], dtype=paddle.float32)

            warmup_iters = self._network_curve_warmup_iters(size_bytes)
            for _ in range(warmup_iters):
                dist.all_reduce(tensor, group=group)
            paddle.device.synchronize()

            round_samples = []
            iters = self._network_curve_test_iters(size_bytes)
            for _ in range(3):
                start = time.perf_counter()
                for _ in range(iters):
                    dist.all_reduce(tensor, group=group)
                paddle.device.synchronize()
                round_samples.append((time.perf_counter() - start) / iters)

            elapsed_s = self._median(round_samples)
            startup_s = num_steps * alpha_s
            transfer_s = max(elapsed_s - startup_s, elapsed_s * 0.02, 1e-9)
            comm_volume_gb = ring_factor * float(size_bytes) / (1024 ** 3)
            measured_transfer_bw = comm_volume_gb / transfer_s if transfer_s > 0 else 0.0
            base_transfer_bw = measured_transfer_bw / allreduce_efficiency if measured_transfer_bw > 0 else 0.0

            if base_transfer_bw > 0:
                points.append(
                    {
                        "size_bytes": int(size_bytes),
                        "bandwidth_gbps": float(base_transfer_bw),
                    }
                )

            del tensor

        paddle.device.cuda.empty_cache()

        if points:
            result["bandwidth_gbps"] = max(p["bandwidth_gbps"] for p in points)
            result["bw_curve"] = NetworkBandwidthCurve.fit_from_points(points)

        return result

    def _collect_representative_bf16_gemm_samples(self,
                                                   peak_tflops: float,
                                                   rounds: int = 2) -> List[Dict[str, Any]]:
        """
        采集少量 workload-aware BF16 GEMM 样本。

        这些点覆盖常见的非方阵 kernel，后续查询直接按 (M, N, K)
        做最近邻匹配，避免只靠方阵曲线外推。
        """
        samples: List[Dict[str, Any]] = []
        effective_rounds = max(1, int(rounds))

        for family, shapes in _BF16_WORKLOAD_GEMM_SHAPES:
            for m, n, k in shapes:
                round_tflops = []
                round_times = []
                for _ in range(effective_rounds):
                    tflops, time_ms = self._benchmark_gemm(m, n, k, "bfloat16")
                    if tflops > 0:
                        round_tflops.append(tflops)
                        round_times.append(time_ms)

                med_tflops = self._median(round_tflops)
                med_time = self._median(round_times)
                if med_tflops <= 0:
                    continue

                efficiency = med_tflops / peak_tflops if peak_tflops > 0 else 0.0
                samples.append(
                    {
                        "family": family,
                        "m": int(m),
                        "n": int(n),
                        "k": int(k),
                        "tflops": float(med_tflops),
                        "efficiency": float(max(0.0, min(1.0, efficiency))),
                        "time_ms": float(med_time),
                    }
                )

        return samples

    def _warmup_until_stable(self,
                              min_seconds: float = 30.0,
                              max_seconds: float = 120.0,
                              batch_iters: int = 50,
                              window: int = 5,
                              cv_threshold: float = 0.01,
                              verbose: bool = True,
                              all_gpus: bool = True) -> float:
        """
        两阶段 GPU 预热：固定预热 + CV 稳定性检测。

        阶段 1 - 固定预热（min_seconds 秒）：
            无论 CV 是否达标，持续跑 GEMM 至少 min_seconds 秒，
            确保 GPU 从 Boost 频率充分降到可持续频率。

        阶段 2 - 稳定性检测（min_seconds ~ max_seconds）：
            固定预热结束后，开始监测 CV（变异系数），
            CV < cv_threshold 即认为达到热稳态，结束预热。
            若超过 max_seconds 仍未达标则强制结束。

        当 all_gpus=True 时，会用后台线程同时在本节点所有 GPU 上跑 GEMM，
        以 self.device_id 的稳定性作为所有 GPU 的收敛指标。

        参数:
            min_seconds:  固定预热最短时间（秒），此期间不检查 CV
            max_seconds:  最大预热时间（秒），超时后强制结束
            batch_iters:  每批次的 GEMM 迭代数
            window:       滑动窗口大小（批次数）
            cv_threshold: CV 阈值，低于此值认为已稳定
            verbose:      是否打印日志
            all_gpus:     是否同时预热本节点所有 GPU（默认 True）

        返回:
            预热总耗时（秒）
        """
        import paddle
        import threading

        M = K = N = 8192
        dtype = paddle.bfloat16

        # ---------- 检测可用 GPU 数量 ----------
        gpu_count = 1
        if all_gpus:
            try:
                gpu_count = paddle.device.cuda.device_count()
            except Exception:
                gpu_count = 1

        # ---------- 后台线程：预热非主 GPU ----------
        stop_event = threading.Event()
        bg_threads = []

        def _bg_warmup(dev_id: int):
            """后台线程：在指定 GPU 上持续跑 GEMM 直到 stop_event 被设置"""
            try:
                paddle.set_device(f"gpu:{dev_id}")
                a = paddle.ones([M, K], dtype=dtype)
                b = paddle.ones([K, N], dtype=dtype)
                # CUDA context 初始化
                for _ in range(10):
                    paddle.matmul(a, b)
                paddle.device.cuda.synchronize(dev_id)
                # 持续运算
                while not stop_event.is_set():
                    for _ in range(batch_iters):
                        paddle.matmul(a, b)
                    paddle.device.cuda.synchronize(dev_id)
                del a, b
                paddle.device.cuda.empty_cache()
            except Exception as e:
                logger.debug(f"GPU {dev_id} 后台预热异常: {e}")

        if gpu_count > 1:
            for dev_id in range(gpu_count):
                if dev_id == self.device_id:
                    continue
                t = threading.Thread(target=_bg_warmup, args=(dev_id,), daemon=True)
                t.start()
                bg_threads.append(t)
            if verbose:
                logger.info(f"同时预热 {gpu_count} 张 GPU (主 GPU={self.device_id})")

        # ---------- 主 GPU 初始化 ----------
        paddle.set_device(f"gpu:{self.device_id}")
        a = paddle.ones([M, K], dtype=dtype)
        b = paddle.ones([K, N], dtype=dtype)

        for _ in range(10):
            paddle.matmul(a, b)
        paddle.device.cuda.synchronize()

        batch_times = []
        total_iters = 0
        t0 = time.perf_counter()

        if verbose:
            logger.info(f"阶段 1: 固定预热 {min_seconds:.0f}s ...")

        # ========== 阶段 1: 固定预热 ==========
        while True:
            elapsed = time.perf_counter() - t0
            if elapsed >= min_seconds or elapsed >= max_seconds:
                break

            paddle.device.cuda.synchronize()
            t_start = time.perf_counter()
            for _ in range(batch_iters):
                paddle.matmul(a, b)
            paddle.device.cuda.synchronize()
            t_batch = time.perf_counter() - t_start

            batch_times.append(t_batch)
            total_iters += batch_iters

        if verbose:
            elapsed_p1 = time.perf_counter() - t0
            if batch_times:
                recent = batch_times[-min(window, len(batch_times)):]
                mean_t = sum(recent) / len(recent)
                avg_s = mean_t / batch_iters
                cur_tflops = 2.0 * M * K * N / avg_s / 1e12 if avg_s > 0 else 0
                logger.info(f"阶段 1 完成: {elapsed_p1:.1f}s, {total_iters} 迭代, "
                            f"当前 BF16≈{cur_tflops:.0f} TFLOPS")
            logger.info("阶段 2: CV 稳定性检测 ...")

        # ========== 阶段 2: CV 稳定性检测 ==========
        batch_times.clear()

        while True:
            elapsed = time.perf_counter() - t0
            if elapsed >= max_seconds:
                if verbose:
                    logger.info(f"GPU 预热超时 ({max_seconds:.0f}s)，强制结束")
                break

            paddle.device.cuda.synchronize()
            t_start = time.perf_counter()
            for _ in range(batch_iters):
                paddle.matmul(a, b)
            paddle.device.cuda.synchronize()
            t_batch = time.perf_counter() - t_start

            batch_times.append(t_batch)
            total_iters += batch_iters

            if len(batch_times) >= window:
                recent = batch_times[-window:]
                mean_t = sum(recent) / len(recent)
                var_t = sum((x - mean_t) ** 2 for x in recent) / len(recent)
                cv = (var_t ** 0.5) / mean_t if mean_t > 0 else 1.0

                if cv < cv_threshold:
                    wall = time.perf_counter() - t0
                    flops_per_iter = 2.0 * M * K * N
                    avg_iter_s = mean_t / batch_iters
                    tflops = flops_per_iter / avg_iter_s / 1e12 if avg_iter_s > 0 else 0
                    if verbose:
                        gpu_tag = f" ({gpu_count} GPUs)" if gpu_count > 1 else ""
                        logger.info(
                            f"GPU 预热完成{gpu_tag}: 总耗时 {wall:.1f}s "
                            f"(固定 {min_seconds:.0f}s + CV检测 {wall - min_seconds:.1f}s), "
                            f"{total_iters} 迭代, "
                            f"稳态 BF16≈{tflops:.0f} TFLOPS (CV={cv:.4f})"
                        )
                    break

        wall = time.perf_counter() - t0

        # ---------- 清理 ----------
        del a, b
        paddle.device.cuda.empty_cache()

        if bg_threads:
            stop_event.set()
            for t in bg_threads:
                t.join(timeout=10.0)

        return wall

    # -------------------------
    # 网络通信基准测试
    # -------------------------
    def _benchmark_nccl_allreduce(self, size_mb: int = 256,
                                   warmup_iters: int = 5,
                                   test_iters: int = 20) -> Tuple[float, float]:
        """
        用 NCCL AllReduce 测量当前 collective group 的带宽和延迟。

        实测原理：
        - 大数据量（size_mb）测得带宽 (GB/s)
        - 小数据量（4KB）测得延迟 (μs)

        需要在 paddle.distributed 已初始化的环境中调用。
        返回 (bandwidth_gbps, latency_us)。若未初始化则返回 (0.0, 0.0)。
        """
        if not _PADDLE_AVAILABLE:
            return 0.0, 0.0

        try:
            import paddle.distributed as dist
            if not dist.is_initialized():
                logger.debug("paddle.distributed 未初始化，跳过 NCCL 带宽测量")
                return 0.0, 0.0
        except Exception:
            return 0.0, 0.0

        bw, lat = 0.0, 0.0
        try:
            paddle.set_device(f"gpu:{self.device_id}")

            # --- 1) 大消息测带宽 ---
            num_elements = size_mb * 1024 * 1024 // 4  # float32
            big_tensor = paddle.ones([num_elements], dtype=paddle.float32)

            for _ in range(warmup_iters):
                dist.all_reduce(big_tensor)
            paddle.device.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(test_iters):
                dist.all_reduce(big_tensor)
            paddle.device.cuda.synchronize()
            elapsed_s = (time.perf_counter() - start) / test_iters

            world = dist.get_world_size()
            # Ring AllReduce 算法带宽公式: bw = 2*(N-1)/N * data / time
            ring_factor = 2.0 * (world - 1) / world
            data_gb = size_mb / 1024.0
            if elapsed_s > 0:
                bw = ring_factor * data_gb / elapsed_s  # GB/s (busbw)

            del big_tensor
            paddle.device.cuda.empty_cache()

            # --- 2) 小消息测延迟 ---
            tiny_tensor = paddle.ones([1], dtype=paddle.float32)  # 4 bytes

            for _ in range(warmup_iters):
                dist.all_reduce(tiny_tensor)
            paddle.device.cuda.synchronize()

            start = time.perf_counter()
            lat_iters = max(test_iters, 50)  # 小消息多迭代取平均
            for _ in range(lat_iters):
                dist.all_reduce(tiny_tensor)
            paddle.device.cuda.synchronize()
            lat_elapsed = (time.perf_counter() - start) / lat_iters
            lat = lat_elapsed * 1e6  # 秒 -> 微秒

            del tiny_tensor
            paddle.device.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"NCCL AllReduce 带宽测量失败: {e}")

        return float(bw), float(lat)

    def _benchmark_nccl_p2p(self, size_mb: int = 64, peer_rank: int = -1,
                             warmup_iters: int = 5,
                             test_iters: int = 20) -> Tuple[float, float]:
        """
        用 NCCL Send/Recv 测量点对点带宽和延迟。

        适用于测量 PP 场景下跨节点 P2P 通信性能。
        只在 rank 0 和 peer_rank 之间进行通信。
        返回 (bandwidth_gbps, latency_us)。
        """
        if not _PADDLE_AVAILABLE:
            return 0.0, 0.0

        try:
            import paddle.distributed as dist
            if not dist.is_initialized():
                return 0.0, 0.0
        except Exception:
            return 0.0, 0.0

        bw, lat = 0.0, 0.0
        try:
            rank = dist.get_rank()
            world = dist.get_world_size()
            if peer_rank < 0:
                peer_rank = world - 1  # 默认测 rank0 <-> 最后一个rank（大概率跨节点）
            if world < 2 or peer_rank == rank:
                return 0.0, 0.0

            paddle.set_device(f"gpu:{self.device_id}")

            # --- 大消息测带宽 ---
            num_elements = size_mb * 1024 * 1024 // 4
            tensor = paddle.ones([num_elements], dtype=paddle.float32)

            for _ in range(warmup_iters):
                if rank == 0:
                    dist.send(tensor, dst=peer_rank)
                    dist.recv(tensor, src=peer_rank)
                elif rank == peer_rank:
                    dist.recv(tensor, src=0)
                    dist.send(tensor, dst=0)
                dist.barrier()
            paddle.device.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(test_iters):
                if rank == 0:
                    dist.send(tensor, dst=peer_rank)
                    dist.recv(tensor, src=peer_rank)
                elif rank == peer_rank:
                    dist.recv(tensor, src=0)
                    dist.send(tensor, dst=0)
                dist.barrier()
            paddle.device.cuda.synchronize()
            elapsed_s = (time.perf_counter() - start) / test_iters

            data_gb = size_mb / 1024.0
            if elapsed_s > 0:
                bw = data_gb / elapsed_s  # 单向带宽

            del tensor

            # --- 小消息测延迟 ---
            tiny = paddle.ones([1], dtype=paddle.float32)
            for _ in range(warmup_iters):
                if rank == 0:
                    dist.send(tiny, dst=peer_rank)
                    dist.recv(tiny, src=peer_rank)
                elif rank == peer_rank:
                    dist.recv(tiny, src=0)
                    dist.send(tiny, dst=0)
                dist.barrier()
            paddle.device.cuda.synchronize()

            lat_iters = max(test_iters, 50)
            start = time.perf_counter()
            for _ in range(lat_iters):
                if rank == 0:
                    dist.send(tiny, dst=peer_rank)
                    dist.recv(tiny, src=peer_rank)
                elif rank == peer_rank:
                    dist.recv(tiny, src=0)
                    dist.send(tiny, dst=0)
                dist.barrier()
            paddle.device.cuda.synchronize()
            lat_elapsed = (time.perf_counter() - start) / lat_iters
            lat = lat_elapsed * 1e6 / 2.0  # 来回 / 2 = 单程延迟

            del tiny
            paddle.device.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"NCCL P2P 带宽测量失败: {e}")

        return float(bw), float(lat)

    def benchmark_network(self, gpus_per_node: int = 8,
                           size_mb: int = 256) -> Dict[str, Any]:
        """
        综合网络基准测试：分别测量节点内和节点间通信性能。

        策略：
        1. 用当前 group 的 AllReduce 测多消息尺寸画像
        2. 从极小消息测单个 ring step 的启动时延 alpha
        3. 用 `elapsed - steps * alpha` 还原纯传输时间，并拟合 transfer bandwidth curve
        4. 如果是多机环境（world > gpus_per_node），额外构建跨节点子 group 测量

        需要在 paddle.distributed 已初始化的环境中调用。
        返回 dict:
            intra_node_bandwidth_gbps: 节点内基础传输带宽峰值 (GB/s)
            inter_node_bandwidth_gbps: 节点间基础传输带宽峰值 (GB/s)
            intra_node_latency_us: 节点内单 step 启动时延 alpha (μs)
            inter_node_latency_us: 节点间单 step 启动时延 alpha (μs)
        """
        result = {
            "intra_node_bandwidth_gbps": 0.0,
            "inter_node_bandwidth_gbps": 0.0,
            "intra_node_latency_us": 0.0,
            "inter_node_latency_us": 0.0,
            "intra_node_bw_curve": None,
            "inter_node_bw_curve": None,
        }

        if not _PADDLE_AVAILABLE:
            return result

        try:
            import paddle.distributed as dist
            if not dist.is_initialized():
                logger.info("paddle.distributed 未初始化，跳过网络校准")
                return result
        except Exception:
            return result

        rank = dist.get_rank()
        world = dist.get_world_size()

        # ============================================================
        # new_group() 是 collective 操作：所有 world 中的 rank 必须
        # 以相同顺序、相同参数调用。因此先统一创建所有 group，
        # 再各自使用属于自己的 group 做 AllReduce。
        # ============================================================
        num_nodes = max(1, world // gpus_per_node)
        node_id = rank // gpus_per_node

        # --- 创建所有节点内 group（所有 rank 必须参与每次 new_group 调用）---
        intra_groups = {}
        for nid in range(num_nodes):
            ranks_in_node = list(range(nid * gpus_per_node,
                                       min((nid + 1) * gpus_per_node, world)))
            if len(ranks_in_node) > 1:
                intra_groups[nid] = dist.new_group(ranks_in_node)

        # --- 创建跨节点 group（所有 rank 必须参与）---
        inter_group = None
        inter_ranks = []
        if num_nodes > 1:
            inter_ranks = [i * gpus_per_node for i in range(num_nodes)
                           if i * gpus_per_node < world]
            if len(inter_ranks) > 1:
                inter_group = dist.new_group(inter_ranks)

        # --- 1) 节点内通信测量：同节点内的 GPU 做 AllReduce ---
        my_intra_group = intra_groups.get(node_id)
        if my_intra_group is not None:
            try:
                intra_sizes = sorted(set(_NETWORK_CURVE_SIZES_BYTES + [size_mb * 1024 * 1024]))
                intra_profile = self._benchmark_allreduce_group_profile(
                    dist=dist,
                    group=my_intra_group,
                    group_size=len(list(range(node_id * gpus_per_node,
                                              min((node_id + 1) * gpus_per_node, world)))),
                    size_bytes_list=intra_sizes,
                )
                result["intra_node_bandwidth_gbps"] = float(intra_profile["bandwidth_gbps"])
                result["intra_node_latency_us"] = float(intra_profile["alpha_us"])
                result["intra_node_bw_curve"] = intra_profile["bw_curve"]
            except Exception as e:
                logger.warning(f"节点内带宽测量失败: {e}")

        # 同步所有 rank，确保节点内测量全部完成后再做节点间测量
        dist.barrier()

        # --- 2) 节点间通信测量：每个节点取一个 rank 做跨节点 AllReduce ---
        if inter_group is not None:
            try:
                if rank in inter_ranks:
                    inter_sizes = sorted(set(_NETWORK_CURVE_SIZES_BYTES + [size_mb * 1024 * 1024]))
                    inter_profile = self._benchmark_allreduce_group_profile(
                        dist=dist,
                        group=inter_group,
                        group_size=len(inter_ranks),
                        size_bytes_list=inter_sizes,
                    )
                    result["inter_node_bandwidth_gbps"] = float(inter_profile["bandwidth_gbps"])
                    result["inter_node_latency_us"] = float(inter_profile["alpha_us"])
                    result["inter_node_bw_curve"] = inter_profile["bw_curve"]

            except Exception as e:
                logger.warning(f"节点间带宽测量失败: {e}")

        if rank == 0:
            logger.info(
                f"网络校准结果: "
                f"节点内={result['intra_node_bandwidth_gbps']:.1f} GB/s "
                f"(延迟 {result['intra_node_latency_us']:.1f} μs), "
                f"节点间={result['inter_node_bandwidth_gbps']:.1f} GB/s "
                f"(延迟 {result['inter_node_latency_us']:.1f} μs)"
            )

        return result

    def _use_preset_values(self, result: CalibrationResult):
        """无法测试时使用预设值（近似值）"""
        name_lower = result.gpu_name.lower()

        presets = {
            "h100": (67.0, 989.0, 989.0, 3350.0),
            "h800": (67.0, 989.0, 989.0, 3350.0),
            "a100": (19.5, 312.0, 312.0, 2039.0),
            "a800": (19.5, 312.0, 312.0, 2039.0),
            "v100": (15.7, 125.0, 0.0, 900.0),
            "4090": (82.6, 330.0, 330.0, 1008.0),
        }

        for key, (fp32, fp16, bf16, mem_bw) in presets.items():
            if key in name_lower:
                result.fp32_tflops = fp32
                result.fp16_tflops = fp16
                result.bf16_tflops = bf16
                result.memory_bandwidth_gbps = mem_bw
                return

        result.fp32_tflops = 20.0
        result.fp16_tflops = 100.0
        result.bf16_tflops = 100.0
        result.memory_bandwidth_gbps = 1000.0

    # -------------------------
    # 统一保存/加载：hardware_spec.py
    # -------------------------
    def save_result(self, gpus_per_node: int = None) -> str:
        """保存校准结果到 hardware_spec.py（纯字典文件）"""
        if self._result is None:
            raise ValueError("No calibration result to save. Run auto_calibrate() first.")

        if gpus_per_node is None:
            gpus_per_node = self._result.gpu_count

        data = self._result.to_dict()
        data["gpus_per_node"] = int(gpus_per_node)
        data["calibrated_at"] = datetime.now().isoformat()
        data["pdcost_version"] = "1.0.0"

        key = _make_spec_key(self._result.gpu_name, gpus_per_node)

        table = _load_hardware_specs_table()
        # 清理同 GPU 型号的旧格式 key（如 H800_8gpu_1node）
        legacy_keys = [k for k in table if k.endswith("node")
                       and k.startswith(_clean_gpu_name(self._result.gpu_name))]
        for lk in legacy_keys:
            del table[lk]

        table[key] = data  # 覆盖/更新

        _write_hardware_specs_table(table)
        return str(HARDWARE_SPEC_PATH)

    def has_saved_result(self, gpu_name: str = None, gpus_per_node: int = None) -> bool:
        """检查 hardware_spec.py 是否存在对应条目"""
        if gpu_name is None or gpus_per_node is None:
            detected_name, _, detected_count = self.detect_gpu_info()
            gpu_name = gpu_name or detected_name
            gpus_per_node = detected_count if gpus_per_node is None else gpus_per_node

        table = _load_hardware_specs_table()
        # 先查新格式 key
        key = _make_spec_key(gpu_name, gpus_per_node)
        if key in table:
            return True
        # 再查旧格式 key（向后兼容）
        for node_count in [1, 2, 4, 8, 16]:
            legacy = _make_legacy_key(gpu_name, gpus_per_node * node_count, node_count)
            if legacy in table:
                return True
            # 单节点旧格式: gpu_count == gpus_per_node
            legacy_single = _make_legacy_key(gpu_name, gpus_per_node, node_count)
            if legacy_single in table:
                return True
        return False

    def load_result(self, gpu_name: str = None, gpus_per_node: int = None) -> Optional[CalibrationResult]:
        """从 hardware_spec.py 加载对应条目"""
        if gpu_name is None or gpus_per_node is None:
            detected_name, _, detected_count = self.detect_gpu_info()
            gpu_name = gpu_name or detected_name
            gpus_per_node = detected_count if gpus_per_node is None else gpus_per_node

        table = _load_hardware_specs_table()

        # 先查新格式 key
        key = _make_spec_key(gpu_name, gpus_per_node)
        data = table.get(key)

        # 再查旧格式 key（向后兼容）
        if not isinstance(data, dict):
            for node_count in [1, 2, 4, 8, 16]:
                for gc in [gpus_per_node, gpus_per_node * node_count]:
                    legacy = _make_legacy_key(gpu_name, gc, node_count)
                    data = table.get(legacy)
                    if isinstance(data, dict):
                        logger.info(f"加载旧格式校准数据: {legacy} (建议重新校准以使用新格式)")
                        break
                if isinstance(data, dict):
                    break

        if not isinstance(data, dict):
            return None

        try:
            self._result = CalibrationResult.from_dict(data)
            return self._result
        except Exception as e:
            logger.warning(f"Failed to parse entry in {HARDWARE_SPEC_PATH}: {e}")
            return None

    # -------------------------
    # 生成 HardwareConfig
    # -------------------------
    def create_hardware_config(self, num_nodes: int = 1, gpus_per_node: int = None) -> HardwareConfig:
        """根据校准结果创建 HardwareConfig"""
        if self._result is None:
            raise ValueError("No calibration result. Run auto_calibrate() or load_result() first.")

        r = self._result

        bf16_curve = None
        if r.bf16_curve is not None:
            bf16_curve = r.bf16_curve.to_dict()

        gpu = GPUSpec(
            name=r.gpu_name,
            memory_gb=r.gpu_memory_gb,
            fp32_tflops=r.fp32_tflops,
            fp16_tflops=r.fp16_tflops,
            bf16_tflops=r.bf16_tflops,
            memory_bandwidth_gbps=r.memory_bandwidth_gbps,
            host_to_device_bandwidth_gbps=(
                r.host_to_device_bandwidth_gbps
                if r.host_to_device_bandwidth_gbps > 0
                else 16.0
            ),
            device_to_host_bandwidth_gbps=(
                r.device_to_host_bandwidth_gbps
                if r.device_to_host_bandwidth_gbps > 0
                else 16.0
            ),
            bf16_curve=bf16_curve,
            bf16_gemm_samples=r.bf16_gemm_samples,
            host_to_device_bw_curve=r.host_to_device_bw_curve,
            device_to_host_bw_curve=r.device_to_host_bw_curve,
        )

        if gpus_per_node is None:
            gpus_per_node = r.gpu_count

        # 网络带宽: 优先使用实测值，否则按 GPU 型号给预设值
        name_lower = r.gpu_name.lower()
        if "h100" in name_lower or "h800" in name_lower:
            default_intra_bw, default_inter_bw = 900.0, 200.0
            default_intra_lat, default_inter_lat = 1.0, 5.0
        elif "a100" in name_lower or "a800" in name_lower:
            default_intra_bw, default_inter_bw = 600.0, 200.0
            default_intra_lat, default_inter_lat = 1.5, 8.0
        else:
            default_intra_bw, default_inter_bw = 300.0, 100.0
            default_intra_lat, default_inter_lat = 2.0, 10.0

        # 实测值 > 0 则覆盖预设值
        intra_bw = r.intra_node_bandwidth_gbps if r.intra_node_bandwidth_gbps > 0 else default_intra_bw
        inter_bw = r.inter_node_bandwidth_gbps if r.inter_node_bandwidth_gbps > 0 else default_inter_bw
        intra_lat = r.intra_node_latency_us if r.intra_node_latency_us > 0 else default_intra_lat
        inter_lat = r.inter_node_latency_us if r.inter_node_latency_us > 0 else default_inter_lat

        network = NetworkSpec(
            intra_node_bandwidth_gbps=intra_bw,
            inter_node_bandwidth_gbps=inter_bw,
            intra_node_latency_us=intra_lat,
            inter_node_latency_us=inter_lat,
            intra_node_bw_curve=r.intra_node_bw_curve,
            inter_node_bw_curve=r.inter_node_bw_curve,
        )

        return HardwareConfig(
            gpu=gpu,
            network=network,
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
        )

    def auto_calibrate(self, node_count: int = 1, force: bool = False,
                        verbose: bool = True, benchmark_network: bool = False) -> HardwareConfig:
        """
        自动校准或加载已有结果。

        校准数据按"机型"(GPU型号 + 每节点GPU数)存储，与节点数无关。
        同一机型无论跑 1/2/4/100 节点，校准数据复用同一份。

        Args:
            node_count: 节点数（仅影响 HardwareConfig.num_nodes，不影响校准 key）
            force: 强制重新校准
            verbose: 打印日志
            benchmark_network: 是否执行 NCCL 网络通信校准（需要 paddle.distributed 已初始化）
                               多机场景建议开启，只需在 2 节点上测一次即可
        """
        gpu_name, memory_gb, gpu_count = self.detect_gpu_info()
        # gpu_count 在单节点上 = gpus_per_node（nvidia-smi 只看到本机 GPU）
        gpus_per_node = gpu_count
        if verbose:
            logger.info(f"检测到硬件: {gpu_name} × {gpus_per_node}/node ({memory_gb:.1f} GB)")

        # 尝试加载已有校准（与 node_count 无关）
        if (not force) and self.has_saved_result(gpu_name, gpus_per_node):
            if verbose:
                logger.info("找到已保存的校准配置，正在加载...")
            r = self.load_result(gpu_name, gpus_per_node)
            if r:
                cfg = self.create_hardware_config(
                    num_nodes=node_count,
                    gpus_per_node=gpus_per_node,
                )
                if verbose:
                    net_info = ""
                    if r.network_calibrated:
                        net_info = (f", 节点内={r.intra_node_bandwidth_gbps:.1f} GB/s"
                                    f", 节点间={r.inter_node_bandwidth_gbps:.1f} GB/s")
                    logger.info(f"使用校准数据: BF16={r.bf16_tflops:.1f} TFLOPS, "
                                f"显存带宽={r.memory_bandwidth_gbps:.1f} GB/s{net_info}")
                return cfg

        # 执行完整校准
        if verbose:
            logger.info("开始硬件完整校准..." if not force else "强制重新校准...")

        r = CalibrationResult()

        # ---- 多尺寸多轮测试配置 ----
        CURVE_SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]   # BF16 曲线尺寸
        MEMBW_SIZES = [128, 256, 512]                                # 显存带宽测试尺寸 (MB)
        ROUNDS = 3                                                   # 每个测试点重复轮数
        SHAPE_ROUNDS = 2                                             # 代表性非方阵 GEMM 轮数

        r.gpu_name = gpu_name
        r.gpu_memory_gb = memory_gb
        r.gpu_count = gpus_per_node

        if gpus_per_node == 0:
            r.error_message = "No GPU detected"
            raise RuntimeError("No GPU detected")

        if not _PADDLE_AVAILABLE:
            logger.warning("PaddlePaddle 未安装，使用预设值")
            self._use_preset_values(r)
            r.calibrated = True
            self._result = r
        else:
            # ---- 预热 GPU 至热稳态（防止 Boost 频率导致测量偏高）----
            if verbose:
                logger.info("正在预热 GPU 至热稳态...")
            self._warmup_until_stable(verbose=verbose)

            # ---- FP32 峰值: 多轮取中位数 ----
            fp32_samples = []
            for _ in range(ROUNDS):
                t, _ = self._benchmark_gemm(8192, 8192, 8192, "float32")
                if t > 0:
                    fp32_samples.append(t)
            r.fp32_tflops = self._median(fp32_samples)

            # ---- FP16 峰值: 多轮取中位数 ----
            fp16_samples = []
            for _ in range(ROUNDS):
                t, _ = self._benchmark_gemm(8192, 8192, 8192, "float16")
                if t > 0:
                    fp16_samples.append(t)
            r.fp16_tflops = self._median(fp16_samples)

            # ---- BF16 多尺寸曲线: 每个尺寸多轮取中位数 ----
            points = []
            for size in CURVE_SIZES:
                round_tflops = []
                round_times = []
                for _ in range(ROUNDS):
                    tflops, time_ms = self._benchmark_gemm(size, size, size, "bfloat16")
                    if tflops > 0:
                        round_tflops.append(tflops)
                        round_times.append(time_ms)
                med_tflops = self._median(round_tflops)
                med_time = self._median(round_times)
                points.append(PerformancePoint(
                    size=size, tflops=med_tflops,
                    efficiency=0.0,  # 下面统一计算
                    time_ms=med_time,
                ))

            # BF16 峰值 = 大尺寸 (≥4096) 的中位数（保守）
            large_bf16 = [p.tflops for p in points if p.size >= 4096 and p.tflops > 0]
            r.bf16_tflops = self._median(large_bf16) if large_bf16 else r.fp16_tflops

            # 回填 efficiency
            if r.bf16_tflops > 0:
                for p in points:
                    p.efficiency = p.tflops / r.bf16_tflops

                r.bf16_curve = PerformanceCurve(
                    dtype="bfloat16", points=points, peak_tflops=r.bf16_tflops
                )
                self._fit_curve(r.bf16_curve)

                if verbose:
                    logger.info("采集代表性 BF16 非方阵 GEMM 样本...")
                r.bf16_gemm_samples = self._collect_representative_bf16_gemm_samples(
                    r.bf16_tflops, rounds=SHAPE_ROUNDS
                )

            # ---- 显存带宽: 多尺寸多轮取中位数 ----
            membw_samples = []
            for sz in MEMBW_SIZES:
                for _ in range(ROUNDS):
                    bw = self._benchmark_memory_bandwidth(sz)
                    if bw > 0:
                        membw_samples.append(bw)
            r.memory_bandwidth_gbps = self._median(membw_samples)

            host_device_result = self._benchmark_host_device_profile(
                sorted(set(_NETWORK_CURVE_SIZES_BYTES))
            )
            if host_device_result["host_to_device_bandwidth_gbps"] > 0:
                r.host_to_device_bandwidth_gbps = host_device_result[
                    "host_to_device_bandwidth_gbps"
                ]
            if host_device_result["device_to_host_bandwidth_gbps"] > 0:
                r.device_to_host_bandwidth_gbps = host_device_result[
                    "device_to_host_bandwidth_gbps"
                ]
            if host_device_result.get("host_to_device_bw_curve") is not None:
                r.host_to_device_bw_curve = host_device_result["host_to_device_bw_curve"]
            if host_device_result.get("device_to_host_bw_curve") is not None:
                r.device_to_host_bw_curve = host_device_result["device_to_host_bw_curve"]

            r.calibrated = True
            self._result = r

        if verbose:
            host_device_info = ""
            if r.host_to_device_bandwidth_gbps > 0 or r.device_to_host_bandwidth_gbps > 0:
                host_device_info = (
                    f", H2D={r.host_to_device_bandwidth_gbps:.1f} GB/s"
                    f", D2H={r.device_to_host_bandwidth_gbps:.1f} GB/s"
                )
            logger.info(
                f"计算校准完成: BF16={r.bf16_tflops:.1f} TFLOPS, "
                f"显存带宽={r.memory_bandwidth_gbps:.1f} GB/s{host_device_info}"
            )

        # ========== 网络通信校准 ==========
        if benchmark_network:
            if verbose:
                logger.info("开始网络通信校准 (NCCL)...")
            net_result = self.benchmark_network(gpus_per_node=gpus_per_node)
            if net_result["intra_node_bandwidth_gbps"] > 0:
                r.intra_node_bandwidth_gbps = net_result["intra_node_bandwidth_gbps"]
                r.intra_node_latency_us = net_result["intra_node_latency_us"]
            if net_result["inter_node_bandwidth_gbps"] > 0:
                r.inter_node_bandwidth_gbps = net_result["inter_node_bandwidth_gbps"]
                r.inter_node_latency_us = net_result["inter_node_latency_us"]
            # 保存带宽曲线
            if net_result.get("intra_node_bw_curve") is not None:
                r.intra_node_bw_curve = net_result["intra_node_bw_curve"]
            if net_result.get("inter_node_bw_curve") is not None:
                r.inter_node_bw_curve = net_result["inter_node_bw_curve"]
            r.network_calibrated = True
            if verbose:
                logger.info(f"网络校准完成: 节点内={r.intra_node_bandwidth_gbps:.1f} GB/s "
                            f"({r.intra_node_latency_us:.1f} μs), "
                            f"节点间={r.inter_node_bandwidth_gbps:.1f} GB/s "
                            f"({r.inter_node_latency_us:.1f} μs)")
        elif node_count > 1 and verbose:
            logger.info("提示: 多机场景建议使用 benchmark_network=True 进行网络实测校准 (只需 2 节点测一次)")

        # 保存到 hardware_spec.py（key 与 node_count 无关）
        saved_path = self.save_result(gpus_per_node=gpus_per_node)
        if verbose:
            logger.info(f"校准结果已写入: {saved_path} (key: {_make_spec_key(r.gpu_name, gpus_per_node)})")

        return self.create_hardware_config(
            num_nodes=node_count,
            gpus_per_node=gpus_per_node,
        )


# ============================================================
# 便捷函数
# ============================================================
def get_hardware_config(node_count: int = 1, force_calibrate: bool = False,
                        verbose: bool = True, benchmark_network: bool = False) -> HardwareConfig:
    """
    获取硬件配置（自动检测、校准或加载）

    Args:
        node_count: 节点数
        force_calibrate: 强制重新校准
        verbose: 打印日志
        benchmark_network: 是否执行 NCCL 网络通信实测（需要 paddle.distributed 已初始化）
                           多机场景建议开启，单机场景可忽略
    """
    calibrator = HardwareCalibrator()
    try:
        return calibrator.auto_calibrate(
            node_count=node_count, force=force_calibrate,
            verbose=verbose, benchmark_network=benchmark_network,
        )
    except RuntimeError as e:
        # 无 GPU 环境下，允许直接回退到 hardware_spec.py 中的已校准条目
        if force_calibrate or "No GPU detected" not in str(e):
            raise

        table = _load_hardware_specs_table()
        if not table:
            raise

        # 使用第一个条目（校准数据与 node_count 无关）
        chosen_key = sorted(table.keys())[0]
        entry = table[chosen_key]
        calibrator._result = CalibrationResult.from_dict(entry)
        gpus_per_node = int(entry.get("gpus_per_node", entry.get("gpu_count", 8)))
        return calibrator.create_hardware_config(
            num_nodes=node_count,
            gpus_per_node=gpus_per_node,
        )
