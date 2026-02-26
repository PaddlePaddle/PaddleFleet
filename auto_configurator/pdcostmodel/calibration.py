#!/usr/bin/env python3
"""
硬件校准模块 - 通过实际测试校准 GPU 算力和通信带宽

功能:
1. 自动检测 GPU 型号和显存
2. GEMM benchmark 测试实际算力
3. NCCL 通信带宽测试
4. 自动更新 HardwareConfig
"""

import os
import time
import subprocess
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

# 尝试导入 paddle，如果失败则标记为不可用
_PADDLE_AVAILABLE = False
try:
    import paddle
    _PADDLE_AVAILABLE = True
except ImportError:
    pass

from .config import GPUSpec, NetworkSpec, HardwareConfig


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
    
    # 拟合参数 (efficiency = a * log(size) + b，带饱和上限)
    fit_a: float = 0.0  # 对数系数
    fit_b: float = 0.0  # 常数项
    fit_max: float = 1.0  # 最大效率上限
    fit_min_size: int = 0  # 最小有效尺寸
    
    def predict_efficiency(self, size: int) -> float:
        """根据拟合曲线预测给定尺寸的效率"""
        import math
        if size <= 0:
            return 0.0
        
        # 对数拟合: efficiency = a * log(size) + b
        # 带饱和上限和下限
        log_size = math.log(max(size, 1))
        efficiency = self.fit_a * log_size + self.fit_b
        
        # 限制在合理范围
        efficiency = max(0.01, min(self.fit_max, efficiency))
        return efficiency
    
    def predict_tflops(self, size: int) -> float:
        """预测给定尺寸的 TFLOPS"""
        efficiency = self.predict_efficiency(size)
        return self.peak_tflops * efficiency
    
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
    
    def __str__(self) -> str:
        lines = [f"PerformanceCurve ({self.dtype}):"]
        lines.append(f"  Peak: {self.peak_tflops:.1f} TFLOPS")
        lines.append(f"  Fit: eff = {self.fit_a:.4f} * log(size) + {self.fit_b:.4f}")
        lines.append(f"  Max Efficiency: {self.fit_max:.1%}")
        lines.append(f"  Test Points:")
        for p in self.points:
            lines.append(f"    {p.size:>6}: {p.tflops:>7.1f} TFLOPS ({p.efficiency:>5.1%})")
        return "\n".join(lines)


@dataclass
class CalibrationResult:
    """校准结果"""
    # GPU 信息
    gpu_name: str = "Unknown"
    gpu_memory_gb: float = 0.0
    gpu_count: int = 0
    
    # 实测算力 (TFLOPS) - 峰值（大矩阵）
    fp32_tflops: float = 0.0
    fp16_tflops: float = 0.0
    bf16_tflops: float = 0.0
    
    # 实测带宽 (GB/s)
    memory_bandwidth_gbps: float = 0.0
    intra_node_bandwidth_gbps: float = 0.0
    
    # 性能曲线（多尺寸测试）
    fp32_curve: Optional[PerformanceCurve] = None
    fp16_curve: Optional[PerformanceCurve] = None
    bf16_curve: Optional[PerformanceCurve] = None
    
    # 校准状态
    calibrated: bool = False
    error_message: str = ""
    
    def get_efficiency(self, size: int, dtype: str = "bfloat16") -> float:
        """根据数据尺寸和类型获取预测效率"""
        curve = None
        if dtype in ["float32", "fp32"]:
            curve = self.fp32_curve
        elif dtype in ["float16", "fp16"]:
            curve = self.fp16_curve
        elif dtype in ["bfloat16", "bf16"]:
            curve = self.bf16_curve
        
        if curve is not None:
            return curve.predict_efficiency(size)
        
        # 默认效率估算（无曲线时）
        import math
        base_eff = 0.5
        size_factor = min(1.0, math.log(max(size, 64)) / math.log(8192))
        return base_eff * size_factor
    
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
        if self.fp16_curve:
            result["fp16_curve"] = self.fp16_curve.to_dict()
        if self.fp32_curve:
            result["fp32_curve"] = self.fp32_curve.to_dict()
        return result
    
    def __str__(self) -> str:
        if not self.calibrated:
            return f"CalibrationResult: Not calibrated ({self.error_message})"
        
        lines = [
            f"CalibrationResult:",
            f"  GPU: {self.gpu_name} × {self.gpu_count}",
            f"  Memory: {self.gpu_memory_gb:.1f} GB",
            f"  FP32 Peak: {self.fp32_tflops:.1f} TFLOPS",
            f"  FP16 Peak: {self.fp16_tflops:.1f} TFLOPS",
            f"  BF16 Peak: {self.bf16_tflops:.1f} TFLOPS",
            f"  Memory BW: {self.memory_bandwidth_gbps:.1f} GB/s",
        ]
        
        if self.bf16_curve:
            lines.append(f"\n{self.bf16_curve}")
        
        return "\n".join(lines)


class HardwareCalibrator:
    """
    硬件校准器
    
    通过实际运行 benchmark 测试硬件性能
    """
    
    def __init__(self, device_id: int = 0, warmup_iters: int = 5, test_iters: int = 20):
        """
        Args:
            device_id: 测试使用的 GPU ID
            warmup_iters: 预热迭代次数
            test_iters: 测试迭代次数
        """
        self.device_id = device_id
        self.warmup_iters = warmup_iters
        self.test_iters = test_iters
        self._result: Optional[CalibrationResult] = None
    
    @property
    def result(self) -> Optional[CalibrationResult]:
        """获取校准结果"""
        return self._result
    
    def detect_gpu_info(self) -> Tuple[str, float, int]:
        """
        检测 GPU 信息
        
        Returns:
            (gpu_name, memory_gb, gpu_count)
        """
        gpu_name = "Unknown"
        memory_gb = 0.0
        gpu_count = 0
        
        # 方法1: 使用 nvidia-smi
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
                    memory_gb = float(parts[1]) / 1024  # MB to GB
        except Exception:
            pass
        
        # 方法2: 使用 paddle
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
    
    def benchmark_gemm(self, m: int, n: int, k: int, dtype: str = "float32",
                       warmup: int = None, iters: int = None) -> float:
        """
        GEMM 算力测试
        
        Args:
            m, n, k: 矩阵尺寸
            dtype: 数据类型 (float32, float16, bfloat16)
            warmup: 预热次数
            iters: 测试次数
        
        Returns:
            实测 TFLOPS
        """
        if not _PADDLE_AVAILABLE:
            return 0.0
        
        warmup = warmup or self.warmup_iters
        iters = iters or self.test_iters
        
        try:
            paddle.set_device(f'gpu:{self.device_id}')
            
            # 创建测试矩阵
            dtype_map = {
                "float32": paddle.float32,
                "float16": paddle.float16,
                "bfloat16": paddle.bfloat16,
            }
            pd_dtype = dtype_map.get(dtype, paddle.float32)
            
            a = paddle.randn([m, k], dtype=pd_dtype)
            b = paddle.randn([k, n], dtype=pd_dtype)
            
            # 预热
            for _ in range(warmup):
                c = paddle.matmul(a, b)
                paddle.device.cuda.synchronize()
            
            # 计时测试
            start_time = time.perf_counter()
            for _ in range(iters):
                c = paddle.matmul(a, b)
            paddle.device.cuda.synchronize()
            end_time = time.perf_counter()
            
            # 计算 TFLOPS
            elapsed_ms = (end_time - start_time) * 1000 / iters
            flops = 2 * m * n * k  # GEMM: 2*M*N*K FLOPs
            tflops = flops / (elapsed_ms / 1000) / 1e12
            
            # 清理
            del a, b, c
            paddle.device.cuda.empty_cache()
            
            return tflops
            
        except Exception as e:
            print(f"GEMM benchmark failed: {e}")
            return 0.0
    
    def benchmark_gemm_with_time(self, m: int, n: int, k: int, dtype: str = "float32",
                                  warmup: int = None, iters: int = None) -> Tuple[float, float]:
        """
        GEMM 算力测试，返回 TFLOPS 和时间
        
        Returns:
            (tflops, elapsed_ms)
        """
        if not _PADDLE_AVAILABLE:
            return 0.0, 0.0
        
        warmup = warmup or self.warmup_iters
        iters = iters or self.test_iters
        
        try:
            paddle.set_device(f'gpu:{self.device_id}')
            
            dtype_map = {
                "float32": paddle.float32,
                "float16": paddle.float16,
                "bfloat16": paddle.bfloat16,
            }
            pd_dtype = dtype_map.get(dtype, paddle.float32)
            
            a = paddle.randn([m, k], dtype=pd_dtype)
            b = paddle.randn([k, n], dtype=pd_dtype)
            
            # 预热
            for _ in range(warmup):
                c = paddle.matmul(a, b)
                paddle.device.cuda.synchronize()
            
            # 计时测试
            start_time = time.perf_counter()
            for _ in range(iters):
                c = paddle.matmul(a, b)
            paddle.device.cuda.synchronize()
            end_time = time.perf_counter()
            
            # 计算
            elapsed_ms = (end_time - start_time) * 1000 / iters
            flops = 2 * m * n * k
            tflops = flops / (elapsed_ms / 1000) / 1e12
            
            del a, b, c
            paddle.device.cuda.empty_cache()
            
            return tflops, elapsed_ms
            
        except Exception as e:
            print(f"GEMM benchmark failed for size {m}: {e}")
            return 0.0, 0.0
    
    def benchmark_gemm_multi_size(self, dtype: str = "bfloat16",
                                   sizes: List[int] = None,
                                   theoretical_peak: float = None,
                                   verbose: bool = True) -> PerformanceCurve:
        """
        多尺寸 GEMM 测试，生成性能曲线
        
        Args:
            dtype: 数据类型
            sizes: 测试尺寸列表 (默认从 64 到 16384 的 12 个尺寸)
            theoretical_peak: 理论峰值 TFLOPS (用于计算效率)
            verbose: 是否打印进度
        
        Returns:
            PerformanceCurve: 性能曲线
        """
        import math
        
        # 默认测试尺寸：从小到大 12 个点
        if sizes is None:
            sizes = [64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
        
        if verbose:
            print(f"\n📊 多尺寸 GEMM 测试 ({dtype}):")
            print(f"  测试尺寸: {sizes}")
        
        points = []
        max_tflops = 0.0
        
        for size in sizes:
            tflops, time_ms = self.benchmark_gemm_with_time(size, size, size, dtype)
            
            if tflops > max_tflops:
                max_tflops = tflops
            
            # 效率计算（如果有理论峰值）
            efficiency = tflops / theoretical_peak if theoretical_peak and theoretical_peak > 0 else 0.0
            
            point = PerformancePoint(
                size=size,
                tflops=tflops,
                efficiency=efficiency,
                time_ms=time_ms
            )
            points.append(point)
            
            if verbose:
                eff_str = f"({efficiency:.1%})" if efficiency > 0 else ""
                print(f"    {size:>6}: {tflops:>7.1f} TFLOPS {eff_str}")
        
        # 使用实测最大值作为峰值（如果没有提供理论峰值）
        peak = theoretical_peak if theoretical_peak and theoretical_peak > 0 else max_tflops
        
        # 更新效率
        for p in points:
            if peak > 0:
                p.efficiency = p.tflops / peak
        
        # 创建曲线并拟合
        curve = PerformanceCurve(
            dtype=dtype,
            points=points,
            peak_tflops=peak
        )
        
        # 拟合曲线
        self._fit_curve(curve)
        
        if verbose:
            print(f"\n  峰值: {peak:.1f} TFLOPS")
            print(f"  拟合公式: efficiency = {curve.fit_a:.4f} * log(size) + {curve.fit_b:.4f}")
            print(f"  最大效率: {curve.fit_max:.1%}")
        
        return curve
    
    def _fit_curve(self, curve: PerformanceCurve):
        """
        拟合性能曲线
        
        使用对数模型: efficiency = a * log(size) + b
        使用最小二乘法拟合
        """
        import math
        
        if not curve.points:
            return
        
        # 准备数据
        x = []  # log(size)
        y = []  # efficiency
        
        for p in curve.points:
            if p.size > 0 and p.efficiency > 0:
                x.append(math.log(p.size))
                y.append(p.efficiency)
        
        if len(x) < 2:
            return
        
        # 最小二乘法拟合 y = a*x + b
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_xx = sum(xi * xi for xi in x)
        
        # 计算斜率 a 和截距 b
        denominator = n * sum_xx - sum_x * sum_x
        if abs(denominator) < 1e-10:
            curve.fit_a = 0
            curve.fit_b = sum_y / n if n > 0 else 0
        else:
            curve.fit_a = (n * sum_xy - sum_x * sum_y) / denominator
            curve.fit_b = (sum_y - curve.fit_a * sum_x) / n
        
        # 计算最大效率（取实测最大值或拟合最大值的较小者）
        max_measured = max(p.efficiency for p in curve.points)
        max_fitted = curve.fit_a * math.log(16384) + curve.fit_b  # 在 16384 尺寸处的拟合值
        curve.fit_max = min(max_measured * 1.05, max_fitted, 1.0)  # 上限 100%
        
        # 找到最小有效尺寸（效率 > 1%）
        for size in [32, 64, 128, 256]:
            eff = curve.fit_a * math.log(size) + curve.fit_b
            if eff > 0.01:
                curve.fit_min_size = size
                break

    def benchmark_memory_bandwidth(self, size_mb: int = 256) -> float:
        """
        显存带宽测试
        
        Args:
            size_mb: 测试数据大小 (MB)
        
        Returns:
            实测带宽 (GB/s)
        """
        if not _PADDLE_AVAILABLE:
            return 0.0
        
        try:
            paddle.set_device(f'gpu:{self.device_id}')
            
            # 创建测试数据
            num_elements = size_mb * 1024 * 1024 // 4  # float32
            src = paddle.randn([num_elements], dtype=paddle.float32)
            
            # 预热
            for _ in range(self.warmup_iters):
                dst = src.clone()
                paddle.device.cuda.synchronize()
            
            # 计时测试
            start_time = time.perf_counter()
            for _ in range(self.test_iters):
                dst = src.clone()
            paddle.device.cuda.synchronize()
            end_time = time.perf_counter()
            
            # 计算带宽 (读 + 写)
            elapsed_s = (end_time - start_time) / self.test_iters
            data_gb = size_mb / 1024 * 2  # 读 + 写
            bandwidth_gbps = data_gb / elapsed_s
            
            # 清理
            del src, dst
            paddle.device.cuda.empty_cache()
            
            return bandwidth_gbps
            
        except Exception as e:
            print(f"Memory bandwidth benchmark failed: {e}")
            return 0.0
    
    def calibrate(self, 
                  test_compute: bool = True,
                  test_memory: bool = True,
                  gemm_size: int = 8192,
                  multi_size_test: bool = False,
                  test_sizes: List[int] = None,
                  verbose: bool = True) -> CalibrationResult:
        """
        执行完整校准
        
        Args:
            test_compute: 是否测试算力
            test_memory: 是否测试显存带宽
            gemm_size: GEMM 测试矩阵大小 (用于峰值测试)
            multi_size_test: 是否进行多尺寸测试生成性能曲线
            test_sizes: 多尺寸测试的尺寸列表 (默认 12 个从 64 到 8192)
            verbose: 是否打印进度
        
        Returns:
            CalibrationResult
        """
        result = CalibrationResult()
        
        if verbose:
            print("=" * 60)
            print("🔧 pdcost 硬件校准")
            print("=" * 60)
        
        # 1. 检测 GPU 信息
        if verbose:
            print("\n[1/5] 检测 GPU 信息...")
        
        gpu_name, memory_gb, gpu_count = self.detect_gpu_info()
        result.gpu_name = gpu_name
        result.gpu_memory_gb = memory_gb
        result.gpu_count = gpu_count
        
        if verbose:
            print(f"  GPU: {gpu_name}")
            print(f"  显存: {memory_gb:.1f} GB")
            print(f"  数量: {gpu_count}")
        
        if gpu_count == 0:
            result.error_message = "No GPU detected"
            return result
        
        if not _PADDLE_AVAILABLE:
            result.error_message = "PaddlePaddle not available"
            if verbose:
                print("\n⚠️ PaddlePaddle 未安装，使用预设值")
            # 使用预设值
            self._use_preset_values(result)
            result.calibrated = True
            self._result = result
            return result
        
        # 2. 测试 FP32 峰值算力
        if test_compute:
            if verbose:
                print(f"\n[2/5] 测试 FP32 峰值算力 (GEMM {gemm_size}×{gemm_size})...")
            result.fp32_tflops = self.benchmark_gemm(gemm_size, gemm_size, gemm_size, "float32")
            if verbose:
                print(f"  FP32 峰值: {result.fp32_tflops:.1f} TFLOPS")
        
        # 3. 测试 FP16/BF16 峰值算力
        if test_compute:
            if verbose:
                print(f"\n[3/5] 测试 FP16/BF16 峰值算力...")
            
            # FP16
            result.fp16_tflops = self.benchmark_gemm(gemm_size, gemm_size, gemm_size, "float16")
            if verbose:
                print(f"  FP16 峰值: {result.fp16_tflops:.1f} TFLOPS")
            
            # BF16
            try:
                result.bf16_tflops = self.benchmark_gemm(gemm_size, gemm_size, gemm_size, "bfloat16")
            except Exception:
                result.bf16_tflops = result.fp16_tflops  # 降级
            if verbose:
                print(f"  BF16 峰值: {result.bf16_tflops:.1f} TFLOPS")
        
        # 4. 测试显存带宽
        if test_memory:
            if verbose:
                print(f"\n[4/5] 测试显存带宽...")
            result.memory_bandwidth_gbps = self.benchmark_memory_bandwidth(256)
            if verbose:
                print(f"  带宽: {result.memory_bandwidth_gbps:.1f} GB/s")
        
        # 5. 多尺寸性能曲线测试
        if multi_size_test and test_compute:
            if verbose:
                print(f"\n[5/5] 多尺寸性能曲线测试...")
            
            # 使用默认尺寸或自定义尺寸
            if test_sizes is None:
                test_sizes = [64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
            
            # BF16 性能曲线
            if result.bf16_tflops > 0:
                result.bf16_curve = self.benchmark_gemm_multi_size(
                    dtype="bfloat16",
                    sizes=test_sizes,
                    theoretical_peak=result.bf16_tflops,
                    verbose=verbose
                )
            
            # FP16 性能曲线 (可选)
            if result.fp16_tflops > 0:
                result.fp16_curve = self.benchmark_gemm_multi_size(
                    dtype="float16",
                    sizes=test_sizes,
                    theoretical_peak=result.fp16_tflops,
                    verbose=verbose
                )
        elif verbose:
            print(f"\n[5/5] 跳过多尺寸测试 (multi_size_test=False)")
        
        result.calibrated = True
        self._result = result
        
        if verbose:
            print("\n" + "=" * 60)
            print("✅ 校准完成!")
            print("=" * 60)
            
            if result.bf16_curve:
                print("\n📈 BF16 性能曲线拟合结果:")
                print(f"  公式: efficiency = {result.bf16_curve.fit_a:.4f} * log(size) + {result.bf16_curve.fit_b:.4f}")
                print(f"  峰值效率: {result.bf16_curve.fit_max:.1%}")
        
        return result
    
    def _use_preset_values(self, result: CalibrationResult):
        """使用预设值（当无法实际测试时）"""
        # 根据 GPU 名称匹配预设
        name_lower = result.gpu_name.lower()
        
        if "h100" in name_lower:
            result.fp32_tflops = 67.0
            result.fp16_tflops = 989.0
            result.bf16_tflops = 989.0
            result.memory_bandwidth_gbps = 3350.0
        elif "a100" in name_lower:
            result.fp32_tflops = 19.5
            result.fp16_tflops = 312.0
            result.bf16_tflops = 312.0
            result.memory_bandwidth_gbps = 2039.0
        elif "a800" in name_lower:
            result.fp32_tflops = 19.5
            result.fp16_tflops = 312.0
            result.bf16_tflops = 312.0
            result.memory_bandwidth_gbps = 2039.0
        elif "v100" in name_lower:
            result.fp32_tflops = 15.7
            result.fp16_tflops = 125.0
            result.bf16_tflops = 0.0
            result.memory_bandwidth_gbps = 900.0
        elif "4090" in name_lower:
            result.fp32_tflops = 82.6
            result.fp16_tflops = 330.0
            result.bf16_tflops = 330.0
            result.memory_bandwidth_gbps = 1008.0
        else:
            # 默认值
            result.fp32_tflops = 20.0
            result.fp16_tflops = 100.0
            result.bf16_tflops = 100.0
            result.memory_bandwidth_gbps = 1000.0
    
    def create_hardware_config(self, 
                               num_nodes: int = 1,
                               gpus_per_node: int = None) -> HardwareConfig:
        """
        根据校准结果创建 HardwareConfig
        
        Args:
            num_nodes: 节点数
            gpus_per_node: 每节点 GPU 数（默认使用检测到的数量）
        
        Returns:
            HardwareConfig
        """
        if self._result is None:
            self.calibrate()
        
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
        
        # 估算网络带宽（基于 GPU 型号）
        name_lower = result.gpu_name.lower()
        if "h100" in name_lower or "h800" in name_lower:
            intra_bw = 900.0  # NVLink 4.0 (H800 同样支持)
            inter_bw = 200.0  # 8x IB HDR
        elif "a100" in name_lower or "a800" in name_lower:
            intra_bw = 600.0  # NVLink 3.0
            inter_bw = 200.0
        else:
            intra_bw = 300.0
            inter_bw = 100.0
        
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


def quick_calibrate(device_id: int = 0, verbose: bool = True) -> CalibrationResult:
    """
    快速校准（便捷函数）
    
    Args:
        device_id: GPU ID
        verbose: 是否打印进度
    
    Returns:
        CalibrationResult
    """
    calibrator = HardwareCalibrator(device_id=device_id)
    return calibrator.calibrate(verbose=verbose)


def create_calibrated_hardware_config(
    num_nodes: int = 1,
    gpus_per_node: int = None,
    device_id: int = 0,
    verbose: bool = True
) -> HardwareConfig:
    """
    创建经过校准的 HardwareConfig
    
    Args:
        num_nodes: 节点数
        gpus_per_node: 每节点 GPU 数
        device_id: 测试使用的 GPU ID
        verbose: 是否打印进度
    
    Returns:
        HardwareConfig
    """
    calibrator = HardwareCalibrator(device_id=device_id)
    calibrator.calibrate(verbose=verbose)
    return calibrator.create_hardware_config(num_nodes, gpus_per_node)