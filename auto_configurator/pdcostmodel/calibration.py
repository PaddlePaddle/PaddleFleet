#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
硬件校准模块 - 自动检测、校准GPU性能并管理校准结果

注意：
- hardware_spec.py 与本文件在同一目录
- hardware_spec.py 是“纯配置文件”，只包含一行：HARDWARE_SPECS = {...}
- 通过 gpu_name + gpu_count + node_count 生成 key，O(1) 定位配置
"""

import os
import time
import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)

# 尝试导入 paddle
_PADDLE_AVAILABLE = False
try:
    import paddle
    _PADDLE_AVAILABLE = True
except ImportError:
    pass

# 你原本的配置类型（保持不动）
from .config import GPUSpec, NetworkSpec, HardwareConfig


# ============================================================
# 路径：hardware_spec.py 与 calibration.py 同目录
# ============================================================
_THIS_DIR = Path(__file__).parent
HARDWARE_SPEC_PATH = _THIS_DIR / "hardware_spec.py"


# ============================================================
# 索引 key：gpu_name + gpu_count + node_count -> key
# ============================================================
def _clean_gpu_name(gpu_name: str) -> str:
    """把 GPU 名字清洗成稳定 key（与之前文件名规则相近）"""
    clean = (gpu_name or "Unknown").replace("NVIDIA ", "").replace(" ", "_").replace("-", "_")
    while "__" in clean:
        clean = clean.replace("__", "_")
    clean = clean.strip("_")
    return clean if clean else "Unknown"


def _make_spec_key(gpu_name: str, gpu_count: int, node_count: int = 1) -> str:
    """用于在 HARDWARE_SPECS 中快速定位的 key"""
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
    intra_node_bandwidth_gbps: float = 0.0

    bf16_curve: Optional[PerformanceCurve] = None

    calibrated: bool = False
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
            "intra_node_bandwidth_gbps": round(self.intra_node_bandwidth_gbps, 2),
            "calibrated": bool(self.calibrated),
        }
        if self.bf16_curve:
            result["bf16_curve"] = self.bf16_curve.to_dict()
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
            intra_node_bandwidth_gbps=float(data.get("intra_node_bandwidth_gbps", 0.0)),
            calibrated=bool(data.get("calibrated", False)),
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
    def save_result(self, node_count: int = 1) -> str:
        """保存校准结果到 hardware_spec.py（纯字典文件）"""
        if self._result is None:
            raise ValueError("No calibration result to save. Run auto_calibrate() first.")

        data = self._result.to_dict()
        data["node_count"] = int(node_count)
        data["calibrated_at"] = datetime.now().isoformat()
        data["pdcost_version"] = "1.0.0"

        key = _make_spec_key(self._result.gpu_name, self._result.gpu_count, node_count)

        table = _load_hardware_specs_table()
        table[key] = data  # 覆盖/更新

        _write_hardware_specs_table(table)
        return str(HARDWARE_SPEC_PATH)

    def has_saved_result(self, gpu_name: str = None, gpu_count: int = None, node_count: int = 1) -> bool:
        """检查 hardware_spec.py 是否存在对应条目"""
        if gpu_name is None or gpu_count is None:
            detected_name, _, detected_count = self.detect_gpu_info()
            gpu_name = gpu_name or detected_name
            gpu_count = detected_count if gpu_count is None else gpu_count

        key = _make_spec_key(gpu_name, gpu_count, node_count)
        table = _load_hardware_specs_table()
        return key in table

    def load_result(self, gpu_name: str = None, gpu_count: int = None, node_count: int = 1) -> Optional[CalibrationResult]:
        """从 hardware_spec.py 加载对应条目"""
        if gpu_name is None or gpu_count is None:
            detected_name, _, detected_count = self.detect_gpu_info()
            gpu_name = gpu_name or detected_name
            gpu_count = detected_count if gpu_count is None else detected_count

        key = _make_spec_key(gpu_name, gpu_count, node_count)
        table = _load_hardware_specs_table()
        data = table.get(key)
        if not isinstance(data, dict):
            return None

        try:
            self._result = CalibrationResult.from_dict(data)
            return self._result
        except Exception as e:
            logger.warning(f"Failed to parse entry {key} in {HARDWARE_SPEC_PATH}: {e}")
            return None

    # -------------------------
    # 生成 HardwareConfig
    # -------------------------
    def create_hardware_config(self, num_nodes: int = 1, gpus_per_node: int = None) -> HardwareConfig:
        """根据校准结果创建 HardwareConfig"""
        if self._result is None:
            raise ValueError("No calibration result. Run auto_calibrate() or load_result() first.")

        r = self._result

        gpu = GPUSpec(
            name=r.gpu_name,
            memory_gb=r.gpu_memory_gb,
            fp32_tflops=r.fp32_tflops,
            fp16_tflops=r.fp16_tflops,
            bf16_tflops=r.bf16_tflops,
            memory_bandwidth_gbps=r.memory_bandwidth_gbps,
        )

        if gpus_per_node is None:
            gpus_per_node = r.gpu_count

        # 简单估算网络带宽（你可后续用真实测量替换）
        name_lower = r.gpu_name.lower()
        if "h100" in name_lower or "h800" in name_lower:
            intra_bw, inter_bw = 900.0, 200.0
        elif "a100" in name_lower or "a800" in name_lower:
            intra_bw, inter_bw = 600.0, 200.0
        else:
            intra_bw, inter_bw = 300.0, 100.0

        if r.intra_node_bandwidth_gbps > 0:
            intra_bw = r.intra_node_bandwidth_gbps

        network = NetworkSpec(
            intra_node_bandwidth_gbps=intra_bw,
            inter_node_bandwidth_gbps=inter_bw,
        )

        return HardwareConfig(
            gpu=gpu,
            network=network,
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
        )

    def auto_calibrate(self, node_count: int = 1, force: bool = False, verbose: bool = True) -> HardwareConfig:
        """
        自动校准或加载已有结果
        - 若存在且 force=False：直接从 hardware_spec.py 读取
        - 否则执行完整校准并保存到 hardware_spec.py
        """
        gpu_name, memory_gb, gpu_count = self.detect_gpu_info()
        if verbose:
            logger.info(f"检测到硬件: {gpu_name} × {gpu_count} ({memory_gb:.1f} GB)")

        # 尝试加载已有校准
        if (not force) and self.has_saved_result(gpu_name, gpu_count, node_count):
            if verbose:
                logger.info("找到已保存的校准配置，正在加载...")
            r = self.load_result(gpu_name, gpu_count, node_count)
            if r:
                cfg = self.create_hardware_config(
                    num_nodes=node_count,
                    gpus_per_node=gpu_count // node_count if node_count > 0 else gpu_count,
                )
                if verbose:
                    logger.info(f"使用校准数据: BF16={r.bf16_tflops:.1f} TFLOPS, 带宽={r.memory_bandwidth_gbps:.1f} GB/s")
                return cfg

        # 执行完整校准
        if verbose:
            logger.info("开始硬件完整校准..." if not force else "强制重新校准...")

        r = CalibrationResult()
        gemm_size = 8192
        test_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

        r.gpu_name = gpu_name
        r.gpu_memory_gb = memory_gb
        r.gpu_count = gpu_count

        if gpu_count == 0:
            r.error_message = "No GPU detected"
            raise RuntimeError("No GPU detected")

        if not _PADDLE_AVAILABLE:
            logger.warning("PaddlePaddle 未安装，使用预设值")
            self._use_preset_values(r)
            r.calibrated = True
            self._result = r
        else:
            # FP32 峰值
            r.fp32_tflops, _ = self._benchmark_gemm(gemm_size, gemm_size, gemm_size, "float32")
            # FP16 峰值
            r.fp16_tflops, _ = self._benchmark_gemm(gemm_size, gemm_size, gemm_size, "float16")
            # BF16 峰值
            bf16, _ = self._benchmark_gemm(gemm_size, gemm_size, gemm_size, "bfloat16")
            r.bf16_tflops = bf16 if bf16 > 0 else r.fp16_tflops

            # 显存带宽
            r.memory_bandwidth_gbps = self._benchmark_memory_bandwidth(256)

            # 多尺寸 BF16 曲线
            points = []
            if r.bf16_tflops > 0:
                for size in test_sizes:
                    tflops, time_ms = self._benchmark_gemm(size, size, size, "bfloat16")
                    eff = (tflops / r.bf16_tflops) if r.bf16_tflops > 0 else 0.0
                    points.append(PerformancePoint(size=size, tflops=tflops, efficiency=eff, time_ms=time_ms))

                r.bf16_curve = PerformanceCurve(dtype="bfloat16", points=points, peak_tflops=r.bf16_tflops)
                self._fit_curve(r.bf16_curve)

            r.calibrated = True
            self._result = r

        if verbose:
            logger.info(f"校准完成: BF16={r.bf16_tflops:.1f} TFLOPS, 带宽={r.memory_bandwidth_gbps:.1f} GB/s")

        # 保存到 hardware_spec.py
        saved_path = self.save_result(node_count=node_count)
        if verbose:
            logger.info(f"校准结果已写入: {saved_path}")

        return self.create_hardware_config(
            num_nodes=node_count,
            gpus_per_node=gpu_count // node_count if node_count > 0 else gpu_count,
        )


# ============================================================
# 便捷函数
# ============================================================
def get_hardware_config(node_count: int = 1, force_calibrate: bool = False, verbose: bool = True) -> HardwareConfig:
    """
    获取硬件配置（自动检测、校准或加载）
    """
    calibrator = HardwareCalibrator()
    return calibrator.auto_calibrate(node_count=node_count, force=force_calibrate, verbose=verbose)