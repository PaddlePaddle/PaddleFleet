#!/usr/bin/env python3
"""
硬件校准模块 - 自动检测、校准GPU性能并管理校准结果

功能:
1. 自动检测 GPU 型号和显存
2. GEMM benchmark 测试实际算力
3. 多尺寸性能曲线测试
4. 显存带宽测试
5. 保存/加载校准结果到本地Python文件（字典格式）
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

from ..config import GPUSpec, NetworkSpec, HardwareConfig


# 校准文件存储目录
PROFILES_DIR = Path(__file__).parent.parent / "profiles"


@dataclass
class PerformancePoint:
    """单个性能测试点"""
    size: int  # 矩阵尺寸 (M=N=K)
    tflops: float  # 实测 TFLOPS
    efficiency: float  # 效率 (实测/理论峰值)
    time_ms: float  # 实测时间


@dataclass 
class PerformanceCurve:
    """性能曲线（多尺寸测试结果）"""
    dtype: str  # 数据类型
    points: List[PerformancePoint]  # 测试点列表
    peak_tflops: float  # 峰值 TFLOPS
    
    # 拟合参数 (efficiency = a * log(size) + b)
    fit_a: float = 0.0
    fit_b: float = 0.0
    fit_max: float = 1.0
    
    def predict_efficiency(self, size: int) -> float:
        """根据拟合曲线预测给定尺寸的效率"""
        import math
        if size <= 0:
            return 0.0
        efficiency = self.fit_a * math.log(max(size, 1)) + self.fit_b
        return max(0.01, min(self.fit_max, efficiency))
    
    def predict_tflops(self, size: int) -> float:
        """预测给定尺寸的 TFLOPS"""
        return self.peak_tflops * self.predict_efficiency(size)
    
    def to_dict(self) -> Dict:
        return {
            "dtype": self.dtype,
            "peak_tflops": round(self.peak_tflops, 2),
            "fit_a": round(self.fit_a, 6),
            "fit_b": round(self.fit_b, 6),
            "fit_max": round(self.fit_max, 4),
            "points": [
                {"size": p.size, "tflops": round(p.tflops, 2), 
                 "efficiency": round(p.efficiency, 4), "time_ms": round(p.time_ms, 3)}
                for p in self.points
            ]
        }


@dataclass
class CalibrationResult:
    """校准结果"""
    # GPU 信息
    gpu_name: str = "Unknown"
    gpu_memory_gb: float = 0.0
    gpu_count: int = 0
    
    # 实测算力 (TFLOPS)
    fp32_tflops: float = 0.0
    fp16_tflops: float = 0.0
    bf16_tflops: float = 0.0
    
    # 实测带宽 (GB/s)
    memory_bandwidth_gbps: float = 0.0
    intra_node_bandwidth_gbps: float = 0.0
    
    # 性能曲线
    bf16_curve: Optional[PerformanceCurve] = None
    
    # 校准状态
    calibrated: bool = False
    error_message: str = ""
    
    def to_dict(self) -> Dict:
        result = {
            "gpu_name": self.gpu_name,
            "gpu_memory_gb": round(self.gpu_memory_gb, 2),
            "gpu_count": self.gpu_count,
            "fp32_tflops": round(self.fp32_tflops, 2),
            "fp16_tflops": round(self.fp16_tflops, 2),
            "bf16_tflops": round(self.bf16_tflops, 2),
            "memory_bandwidth_gbps": round(self.memory_bandwidth_gbps, 2),
            "intra_node_bandwidth_gbps": round(self.intra_node_bandwidth_gbps, 2),
            "calibrated": self.calibrated,
        }
        if self.bf16_curve:
            result["bf16_curve"] = self.bf16_curve.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> "CalibrationResult":
        """从字典创建 CalibrationResult"""
        result = cls(
            gpu_name=data.get("gpu_name", "Unknown"),
            gpu_memory_gb=data.get("gpu_memory_gb", 0.0),
            gpu_count=data.get("gpu_count", 0),
            fp32_tflops=data.get("fp32_tflops", 0.0),
            fp16_tflops=data.get("fp16_tflops", 0.0),
            bf16_tflops=data.get("bf16_tflops", 0.0),
            memory_bandwidth_gbps=data.get("memory_bandwidth_gbps", 0.0),
            intra_node_bandwidth_gbps=data.get("intra_node_bandwidth_gbps", 0.0),
            calibrated=data.get("calibrated", False),
        )
        
        # 解析 bf16_curve
        if "bf16_curve" in data:
            curve_data = data["bf16_curve"]
            points = [
                PerformancePoint(
                    size=p["size"], tflops=p["tflops"],
                    efficiency=p["efficiency"], time_ms=p["time_ms"]
                )
                for p in curve_data.get("points", [])
            ]
            result.bf16_curve = PerformanceCurve(
                dtype=curve_data.get("dtype", "bfloat16"),
                points=points,
                peak_tflops=curve_data.get("peak_tflops", 0.0),
                fit_a=curve_data.get("fit_a", 0.0),
                fit_b=curve_data.get("fit_b", 0.0),
                fit_max=curve_data.get("fit_max", 1.0),
            )
        
        return result


def _get_profile_filename(gpu_name: str, gpu_count: int, node_count: int = 1) -> str:
    """生成校准文件名"""
    clean_name = gpu_name.replace("NVIDIA ", "").replace(" ", "_").replace("-", "_")
    while "__" in clean_name:
        clean_name = clean_name.replace("__", "_")
    clean_name = clean_name.strip("_")
    return f"{clean_name}_{gpu_count}gpu_{node_count}node.py"


def _get_profile_path(gpu_name: str, gpu_count: int, node_count: int = 1) -> Path:
    """获取校准文件完整路径"""
    filename = _get_profile_filename(gpu_name, gpu_count, node_count)
    return PROFILES_DIR / filename


class HardwareCalibrator:
    """
    硬件校准器
    
    使用示例:
        calibrator = HardwareCalibrator()
        
        # 方式1: 自动校准或加载已有结果
        config = calibrator.auto_calibrate()
        
        # 方式2: 强制重新校准
        config = calibrator.auto_calibrate(force=True)
    """
    
    def __init__(self, device_id: int = 0, warmup_iters: int = 5, test_iters: int = 20):
        self.device_id = device_id
        self.warmup_iters = warmup_iters
        self.test_iters = test_iters
        self._result: Optional[CalibrationResult] = None
        
        # 确保目录存在
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    
    @property
    def result(self) -> Optional[CalibrationResult]:
        return self._result
    
    def detect_gpu_info(self) -> Tuple[str, float, int]:
        """检测 GPU 信息: (gpu_name, memory_gb, gpu_count)"""
        gpu_name = "Unknown"
        memory_gb = 0.0
        gpu_count = 0
        
        # 使用 nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,count", 
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_count = len(lines)
                if lines:
                    parts = lines[0].split(', ')
                    gpu_name = parts[0].strip()
                    memory_gb = float(parts[1]) / 1024
        except Exception:
            pass
        
        # 备用: 使用 paddle
        if _PADDLE_AVAILABLE and gpu_count == 0:
            try:
                gpu_count = paddle.device.cuda.device_count()
                if gpu_count > 0:
                    props = paddle.device.cuda.get_device_properties(self.device_id)
                    gpu_name = props.name
                    memory_gb = props.total_memory / (1024 ** 3)
            except Exception:
                pass
        
        return gpu_name, memory_gb, gpu_count
    
    def _benchmark_gemm(self, m: int, n: int, k: int, dtype: str = "bfloat16") -> Tuple[float, float]:
        """GEMM 测试，返回 (tflops, elapsed_ms)"""
        if not _PADDLE_AVAILABLE:
            return 0.0, 0.0
        
        try:
            paddle.set_device(f'gpu:{self.device_id}')
            
            dtype_map = {
                "float32": paddle.float32,
                "float16": paddle.float16,
                "bfloat16": paddle.bfloat16,
            }
            pd_dtype = dtype_map.get(dtype, paddle.bfloat16)
            
            a = paddle.randn([m, k], dtype=pd_dtype)
            b = paddle.randn([k, n], dtype=pd_dtype)
            
            # 预热
            for _ in range(self.warmup_iters):
                c = paddle.matmul(a, b)
                paddle.device.cuda.synchronize()
            
            # 计时
            start_time = time.perf_counter()
            for _ in range(self.test_iters):
                c = paddle.matmul(a, b)
            paddle.device.cuda.synchronize()
            end_time = time.perf_counter()
            
            elapsed_ms = (end_time - start_time) * 1000 / self.test_iters
            flops = 2 * m * n * k
            tflops = flops / (elapsed_ms / 1000) / 1e12
            
            del a, b, c
            paddle.device.cuda.empty_cache()
            
            return tflops, elapsed_ms
            
        except Exception as e:
            logger.warning(f"GEMM benchmark failed: {e}")
            return 0.0, 0.0
    
    def _benchmark_memory_bandwidth(self, size_mb: int = 256) -> float:
        """显存带宽测试，返回 GB/s"""
        if not _PADDLE_AVAILABLE:
            return 0.0
        
        try:
            paddle.set_device(f'gpu:{self.device_id}')
            
            num_elements = size_mb * 1024 * 1024 // 4
            src = paddle.randn([num_elements], dtype=paddle.float32)
            
            for _ in range(self.warmup_iters):
                dst = src.clone()
                paddle.device.cuda.synchronize()
            
            start_time = time.perf_counter()
            for _ in range(self.test_iters):
                dst = src.clone()
            paddle.device.cuda.synchronize()
            end_time = time.perf_counter()
            
            elapsed_s = (end_time - start_time) / self.test_iters
            data_gb = size_mb / 1024 * 2
            bandwidth_gbps = data_gb / elapsed_s
            
            del src, dst
            paddle.device.cuda.empty_cache()
            
            return bandwidth_gbps
            
        except Exception as e:
            logger.warning(f"Memory bandwidth benchmark failed: {e}")
            return 0.0
    
    def _fit_curve(self, curve: PerformanceCurve):
        """拟合性能曲线: efficiency = a * log(size) + b"""
        import math
        
        if not curve.points:
            return
        
        x = [math.log(p.size) for p in curve.points if p.size > 0 and p.efficiency > 0]
        y = [p.efficiency for p in curve.points if p.size > 0 and p.efficiency > 0]
        
        if len(x) < 2:
            return
        
        n = len(x)
        sum_x, sum_y = sum(x), sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_xx = sum(xi * xi for xi in x)
        
        denominator = n * sum_xx - sum_x * sum_x
        if abs(denominator) < 1e-10:
            curve.fit_a = 0
            curve.fit_b = sum_y / n if n > 0 else 0
        else:
            curve.fit_a = (n * sum_xy - sum_x * sum_y) / denominator
            curve.fit_b = (sum_y - curve.fit_a * sum_x) / n
        
        max_measured = max(p.efficiency for p in curve.points)
        max_fitted = curve.fit_a * math.log(16384) + curve.fit_b
        curve.fit_max = min(max_measured * 1.05, max_fitted, 1.0)
    
    def _use_preset_values(self, result: CalibrationResult):
        """无法测试时使用预设值"""
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
        
        # 默认值
        result.fp32_tflops = 20.0
        result.fp16_tflops = 100.0
        result.bf16_tflops = 100.0
        result.memory_bandwidth_gbps = 1000.0
    
    def save_result(self, node_count: int = 1) -> str:
        """
        保存校准结果到本地 Python 文件（字典格式）
        
        Returns:
            保存的文件路径
        """
        if self._result is None:
            raise ValueError("No calibration result to save. Run calibrate() first.")
        
        data = self._result.to_dict()
        data["node_count"] = node_count
        data["calibrated_at"] = datetime.now().isoformat()
        data["pdcost_version"] = "1.0.0"
        
        profile_path = _get_profile_path(
            self._result.gpu_name,
            self._result.gpu_count,
            node_count
        )
        
        # 生成 Python 字典格式内容
        py_content = f'''#!/usr/bin/env python3
"""
硬件校准配置文件
GPU: {data.get("gpu_name", "Unknown")}
数量: {data.get("gpu_count", 0)}
节点: {data.get("node_count", 1)}
校准时间: {data.get("calibrated_at", "Unknown")}
"""

HARDWARE_PROFILE = {self._format_dict(data)}
'''
        
        with open(profile_path, 'w') as f:
            f.write(py_content)
        
        return str(profile_path)
    
    def _format_dict(self, data: Dict, indent: int = 0) -> str:
        """格式化字典为 Python 代码字符串"""
        lines = ["{"]
        for key, value in data.items():
            prefix = "    " * (indent + 1)
            if isinstance(value, dict):
                formatted_value = self._format_dict(value, indent + 1)
                lines.append(f'{prefix}"{key}": {formatted_value},')
            elif isinstance(value, list):
                if len(value) == 0:
                    lines.append(f'{prefix}"{key}": [],')
                elif isinstance(value[0], dict):
                    lines.append(f'{prefix}"{key}": [')
                    for item in value:
                        formatted_item = self._format_dict(item, indent + 2)
                        lines.append(f'{"    " * (indent + 2)}{formatted_item},')
                    lines.append(f'{prefix}],')
                else:
                    lines.append(f'{prefix}"{key}": {repr(value)},')
            elif isinstance(value, str):
                lines.append(f'{prefix}"{key}": "{value}",')
            elif isinstance(value, bool):
                lines.append(f'{prefix}"{key}": {value},')
            elif isinstance(value, (int, float)):
                lines.append(f'{prefix}"{key}": {value},')
            else:
                lines.append(f'{prefix}"{key}": {repr(value)},')
        lines.append("    " * indent + "}")
        return "\n".join(lines)
    
    def load_result(self, gpu_name: str = None, gpu_count: int = None, 
                    node_count: int = 1) -> Optional[CalibrationResult]:
        """
        从本地加载校准结果
        
        Args:
            gpu_name: GPU名称（不指定则自动检测）
            gpu_count: GPU数量（不指定则自动检测）
            node_count: 节点数量
        
        Returns:
            CalibrationResult 或 None（文件不存在时）
        """
        if gpu_name is None or gpu_count is None:
            detected_name, _, detected_count = self.detect_gpu_info()
            gpu_name = gpu_name or detected_name
            gpu_count = gpu_count if gpu_count is not None else detected_count
        
        profile_path = _get_profile_path(gpu_name, gpu_count, node_count)
        
        if not profile_path.exists():
            return None
        
        try:
            # 从 Python 文件加载字典
            import importlib.util
            spec = importlib.util.spec_from_file_location("profile", profile_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            data = module.HARDWARE_PROFILE
            
            self._result = CalibrationResult.from_dict(data)
            return self._result
        except Exception as e:
            logger.warning(f"Failed to load profile {profile_path}: {e}")
            return None
    
    def has_saved_result(self, gpu_name: str = None, gpu_count: int = None, 
                         node_count: int = 1) -> bool:
        """检查是否存在已保存的校准结果"""
        if gpu_name is None or gpu_count is None:
            detected_name, _, detected_count = self.detect_gpu_info()
            gpu_name = gpu_name or detected_name
            gpu_count = gpu_count if gpu_count is not None else detected_count
        
        profile_path = _get_profile_path(gpu_name, gpu_count, node_count)
        return profile_path.exists()
    
    def create_hardware_config(self, num_nodes: int = 1, 
                               gpus_per_node: int = None) -> HardwareConfig:
        """根据校准结果创建 HardwareConfig"""
        if self._result is None:
            raise ValueError("No calibration result. Run calibrate() or load_result() first.")
        
        result = self._result
        
        gpu = GPUSpec(
            name=result.gpu_name,
            memory_gb=result.gpu_memory_gb,
            fp32_tflops=result.fp32_tflops,
            fp16_tflops=result.fp16_tflops,
            bf16_tflops=result.bf16_tflops,
            memory_bandwidth_gbps=result.memory_bandwidth_gbps,
        )
        
        if gpus_per_node is None:
            gpus_per_node = result.gpu_count
        
        # 估算网络带宽
        name_lower = result.gpu_name.lower()
        if "h100" in name_lower or "h800" in name_lower:
            intra_bw, inter_bw = 900.0, 200.0
        elif "a100" in name_lower or "a800" in name_lower:
            intra_bw, inter_bw = 600.0, 200.0
        else:
            intra_bw, inter_bw = 300.0, 100.0
        
        if result.intra_node_bandwidth_gbps > 0:
            intra_bw = result.intra_node_bandwidth_gbps
        
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
    
    def auto_calibrate(self, node_count: int = 1, force: bool = False, 
                       verbose: bool = True) -> HardwareConfig:
        """
        自动校准或加载已有结果
        
        流程:
        1. 检测当前硬件
        2. 检查是否有匹配的已保存校准
        3. 如果有且 force=False，加载并返回
        4. 否则执行完整校准并保存
        
        完整校准包含:
        - GPU 信息检测
        - FP32/FP16/BF16 峰值算力测试
        - 显存带宽测试
        - 多尺寸 BF16 性能曲线测试
        
        Args:
            node_count: 节点数量
            force: 是否强制重新校准
            verbose: 是否打印信息
        
        Returns:
            HardwareConfig
        """
        gpu_name, memory_gb, gpu_count = self.detect_gpu_info()
        
        logger.info(f"检测到硬件: {gpu_name} × {gpu_count} ({memory_gb:.1f} GB)")
        
        # 尝试加载已有校准
        if not force and self.has_saved_result(gpu_name, gpu_count, node_count):
            logger.info("找到已保存的校准配置，正在加载...")
            
            result = self.load_result(gpu_name, gpu_count, node_count)
            if result:
                config = self.create_hardware_config(
                    num_nodes=node_count,
                    gpus_per_node=gpu_count // node_count if node_count > 0 else gpu_count
                )
                logger.info(f"使用校准数据: BF16={result.bf16_tflops:.1f} TFLOPS, 带宽={result.memory_bandwidth_gbps:.1f} GB/s")
                return config
        
        # 执行完整校准
        if force:
            logger.info("强制重新校准...")
        else:
            logger.info("未找到校准配置，开始完整校准...")
        
        result = CalibrationResult()
        gemm_size = 8192
        test_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
        
        logger.debug("开始硬件完整校准")
        
        # 1. GPU 信息已检测
        result.gpu_name = gpu_name
        result.gpu_memory_gb = memory_gb
        result.gpu_count = gpu_count
        
        logger.debug(f"[1/5] GPU 信息: {gpu_name}, 显存: {memory_gb:.1f} GB, 数量: {gpu_count}")
        
        if gpu_count == 0:
            result.error_message = "No GPU detected"
            raise RuntimeError("No GPU detected")
        
        if not _PADDLE_AVAILABLE:
            logger.warning("PaddlePaddle 未安装，使用预设值")
            self._use_preset_values(result)
            result.calibrated = True
            self._result = result
        else:
            # 2. 测试 FP32 峰值
            logger.debug("[2/5] 测试 FP32 峰值算力...")
            result.fp32_tflops, _ = self._benchmark_gemm(gemm_size, gemm_size, gemm_size, "float32")
            logger.debug(f"FP32 峰值: {result.fp32_tflops:.1f} TFLOPS")
            
            # 3. 测试 FP16/BF16 峰值
            logger.debug("[3/5] 测试 FP16/BF16 峰值算力...")
            result.fp16_tflops, _ = self._benchmark_gemm(gemm_size, gemm_size, gemm_size, "float16")
            logger.debug(f"FP16 峰值: {result.fp16_tflops:.1f} TFLOPS")
            
            try:
                result.bf16_tflops, _ = self._benchmark_gemm(gemm_size, gemm_size, gemm_size, "bfloat16")
            except Exception:
                result.bf16_tflops = result.fp16_tflops
            logger.debug(f"BF16 峰值: {result.bf16_tflops:.1f} TFLOPS")
            
            # 4. 测试显存带宽
            logger.debug("[4/5] 测试显存带宽...")
            result.memory_bandwidth_gbps = self._benchmark_memory_bandwidth(256)
            logger.debug(f"带宽: {result.memory_bandwidth_gbps:.1f} GB/s")
            
            # 5. 多尺寸 BF16 性能曲线
            logger.debug(f"[5/5] 多尺寸 BF16 性能曲线测试，尺寸: {test_sizes}")
            
            if result.bf16_tflops > 0:
                points = []
                for size in test_sizes:
                    tflops, time_ms = self._benchmark_gemm(size, size, size, "bfloat16")
                    efficiency = tflops / result.bf16_tflops if result.bf16_tflops > 0 else 0
                    points.append(PerformancePoint(size=size, tflops=tflops, efficiency=efficiency, time_ms=time_ms))
                    logger.debug(f"  {size}: {tflops:.1f} TFLOPS ({efficiency:.1%})")
                
                result.bf16_curve = PerformanceCurve(
                    dtype="bfloat16",
                    points=points,
                    peak_tflops=result.bf16_tflops
                )
                self._fit_curve(result.bf16_curve)
                
                logger.debug(f"拟合公式: efficiency = {result.bf16_curve.fit_a:.4f} * log(size) + {result.bf16_curve.fit_b:.4f}")
            
            result.calibrated = True
            self._result = result
        
        logger.info(f"校准完成: BF16={result.bf16_tflops:.1f} TFLOPS, 带宽={result.memory_bandwidth_gbps:.1f} GB/s")
        
        # 保存结果
        filepath = self.save_result(node_count)
        logger.info(f"校准结果已保存: {filepath}")
        
        return self.create_hardware_config(
            num_nodes=node_count,
            gpus_per_node=gpu_count // node_count if node_count > 0 else gpu_count
        )


# ============================================================
# 便捷函数
# ============================================================

def get_hardware_config(node_count: int = 1, force_calibrate: bool = False, 
                        verbose: bool = True) -> HardwareConfig:
    """
    获取硬件配置（自动检测、校准或加载）
    
    这是最常用的入口函数。
    
    Args:
        node_count: 节点数量
        force_calibrate: 是否强制重新校准
        verbose: 是否打印信息
    
    Returns:
        HardwareConfig
    
    使用示例:
        config = get_hardware_config()
    """
    calibrator = HardwareCalibrator()
    return calibrator.auto_calibrate(
        node_count=node_count,
        force=force_calibrate,
        verbose=verbose
    )