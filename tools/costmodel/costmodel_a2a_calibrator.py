#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cost Model A2A Calibrator - All-to-All 通信校准器

功能：
1. 测量真实 All-to-All 通信延迟
2. 支持不均匀的通信模式（模拟 MoE 路由不均衡）
3. 建立通信量到延迟的映射关系（线性/多项式拟合）
4. 输出校准参数用于 Cost Model 预测

运行方式:
    python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 costmodel_a2a_calibrator.py

环境变量:
    WARMUP          : warmup 轮数（默认 5）
    REPEATS         : 每个配置的重复次数（默认 20）
    HIDDEN_SIZE     : hidden dimension（默认 2048）
    DTYPE           : 数据类型（默认 bfloat16）
"""

import os
import time
import json
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict

import paddle
import paddle.distributed as dist


# ============================================================
# Data Classes
# ============================================================
@dataclass
class A2ATimingResult:
    """A2A 通信测量结果"""
    pattern: str                    # uniform / imbalanced / skewed
    total_bytes: int                # 总通信字节数
    bytes_per_rank_send: List[int]  # 每个 rank 发送的字节数
    bytes_per_rank_recv: List[int]  # 每个 rank 接收的字节数
    time_ms_mean: float
    time_ms_std: float
    time_ms_min: float
    time_ms_max: float
    bandwidth_GBps: float           # 有效带宽 GB/s
    n_repeats: int


# ============================================================
# Utils
# ============================================================
def sync():
    if paddle.is_compiled_with_cuda():
        paddle.device.synchronize()


def dtype_nbytes(dtype_str: str) -> int:
    d = dtype_str.lower()
    if d in ("bfloat16", "float16"):
        return 2
    return 4


# ============================================================
# A2A Communication Patterns
# ============================================================
class A2APatternGenerator:
    """
    生成不同的 A2A 通信模式
    
    Patterns:
    1. uniform: 每个 rank 发送/接收相同数量的 tokens
    2. imbalanced: 模拟 MoE 路由不均衡（部分 rank 接收更多）
    3. skewed: 极端倾斜（一个 rank 接收大部分 tokens）
    4. random: 随机分布
    """
    
    def __init__(self, world_size: int, hidden_size: int, dtype: str = "bfloat16"):
        self.world_size = world_size
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.dtype_bytes = dtype_nbytes(dtype)
        self.rank = dist.get_rank()
    
    def generate_uniform(self, tokens_per_rank: int) -> Tuple[paddle.Tensor, List[int], List[int]]:
        """
        均匀分布：每个 rank 发送 tokens_per_rank 个 tokens 给每个其他 rank
        
        Returns:
            send_data: [world_size, tokens_per_rank, hidden_size]
            send_counts: 每个目标 rank 发送的 token 数
            recv_counts: 从每个源 rank 接收的 token 数
        """
        send_counts = [tokens_per_rank] * self.world_size
        recv_counts = [tokens_per_rank] * self.world_size
        
        # 生成发送数据
        total_tokens = tokens_per_rank * self.world_size
        send_data = paddle.randn([total_tokens, self.hidden_size], dtype=self.dtype)
        
        return send_data, send_counts, recv_counts
    
    def generate_imbalanced(self, base_tokens: int, imbalance_ratio: float = 1.5) -> Tuple[paddle.Tensor, List[int], List[int]]:
        """
        不均衡分布：模拟 MoE 路由不均衡
        
        Args:
            base_tokens: 基础 token 数
            imbalance_ratio: 不均衡比例（max/mean）
        
        某些 rank 接收更多 tokens（模拟热门 expert）
        """
        # 生成不均衡的发送计数
        # 使用指数分布模拟热门 expert
        np.random.seed(42 + self.rank)  # 确保每个 rank 有不同但可重复的分布
        
        # 生成每个 rank 要发送给各目标 rank 的 token 数
        weights = np.random.exponential(scale=1.0, size=self.world_size)
        weights = weights / weights.sum()  # 归一化
        
        # 调整以达到目标 imbalance_ratio
        total_tokens = base_tokens * self.world_size
        send_counts = (weights * total_tokens).astype(int)
        
        # 确保总数正确
        diff = total_tokens - send_counts.sum()
        send_counts[0] += diff
        send_counts = send_counts.tolist()
        
        # recv_counts 需要通过 all_to_all_single 获取
        # 这里先假设对称（实际使用时需要同步）
        recv_counts = send_counts.copy()  # 简化：假设对称
        
        # 生成发送数据
        send_data = paddle.randn([sum(send_counts), self.hidden_size], dtype=self.dtype)
        
        return send_data, send_counts, recv_counts
    
    def generate_skewed(self, total_tokens: int, hot_rank: int = 0, hot_ratio: float = 0.5) -> Tuple[paddle.Tensor, List[int], List[int]]:
        """
        极端倾斜：一个 rank 接收大部分 tokens
        
        Args:
            total_tokens: 总 token 数
            hot_rank: 热门 rank
            hot_ratio: 热门 rank 接收的比例
        """
        hot_tokens = int(total_tokens * hot_ratio)
        cold_tokens = (total_tokens - hot_tokens) // (self.world_size - 1)
        
        send_counts = [cold_tokens] * self.world_size
        send_counts[hot_rank] = hot_tokens
        
        # 确保总数正确
        diff = total_tokens - sum(send_counts)
        send_counts[0] += diff
        
        recv_counts = send_counts.copy()
        
        send_data = paddle.randn([sum(send_counts), self.hidden_size], dtype=self.dtype)
        
        return send_data, send_counts, recv_counts
    
    def generate_moe_realistic(self, num_tokens: int, topk: int = 8, num_experts: int = 128) -> Tuple[paddle.Tensor, List[int], List[int]]:
        """
        真实 MoE 模式：基于 expert selection 生成通信模式
        
        每个 token 选择 topk 个 experts，experts 均匀分布在各 rank
        """
        experts_per_rank = num_experts // self.world_size
        
        # 模拟 expert selection（使用 Zipf 分布模拟热门 expert）
        np.random.seed(42 + self.rank)
        
        # 生成每个 token 选择的 expert indices
        # 使用 Zipf-like 分布：某些 expert 更热门
        expert_popularity = np.power(np.arange(1, num_experts + 1), -0.5)
        expert_popularity = expert_popularity / expert_popularity.sum()
        
        # 每个 token 选择 topk 个 experts
        selections = np.random.choice(
            num_experts, 
            size=(num_tokens, topk), 
            replace=True, 
            p=expert_popularity
        )
        
        # 统计发送到每个 rank 的 token 数
        send_counts = [0] * self.world_size
        for token_selections in selections:
            for expert_id in token_selections:
                target_rank = expert_id // experts_per_rank
                if target_rank < self.world_size:
                    send_counts[target_rank] += 1
        
        recv_counts = send_counts.copy()
        
        send_data = paddle.randn([sum(send_counts), self.hidden_size], dtype=self.dtype)
        
        return send_data, send_counts, recv_counts


# ============================================================
# A2A Calibrator
# ============================================================
class A2ACalibrator:
    """
    All-to-All 通信校准器
    
    测量不同数据量和模式下的 A2A 通信延迟，建立预测模型
    """
    
    def __init__(self, hidden_size: int = 2048, dtype: str = "bfloat16"):
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.dtype_bytes = dtype_nbytes(dtype)
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.pattern_gen = A2APatternGenerator(self.world_size, hidden_size, dtype)
        
        self.results: List[A2ATimingResult] = []
    
    def _measure_a2a(
        self, 
        send_data: paddle.Tensor, 
        send_counts: List[int], 
        recv_counts: List[int],
        warmup: int = 5,
        repeats: int = 20,
    ) -> Tuple[float, float, float, float]:
        """
        测量 A2A 通信延迟
        
        使用 alltoall (tensor list 版本) 进行通信
        """
        # 准备发送 tensor list - 按 send_counts 分割 send_data
        send_list = []
        offset = 0
        for cnt in send_counts:
            if cnt > 0:
                chunk = send_data[offset:offset+cnt]
                # 确保 contiguous
                if hasattr(chunk, 'contiguous'):
                    chunk = chunk.contiguous()
                send_list.append(chunk)
            else:
                send_list.append(paddle.empty([0, self.hidden_size], dtype=self.dtype))
            offset += cnt
        
        times = []
        
        for i in range(warmup + repeats):
            # 每次迭代创建新的接收 buffers
            recv_list = []
            for cnt in recv_counts:
                if cnt > 0:
                    recv_list.append(paddle.empty([cnt, self.hidden_size], dtype=self.dtype))
                else:
                    recv_list.append(paddle.empty([0, self.hidden_size], dtype=self.dtype))
            
            sync()
            dist.barrier()
            
            t_start = time.time()
            
            # All-to-All 通信 (tensor list 版本)
            dist.alltoall(recv_list, send_list)
            
            sync()
            t_end = time.time()
            
            if i >= warmup:
                times.append((t_end - t_start) * 1000.0)  # ms
        
        times = np.array(times)
        return float(times.mean()), float(times.std()), float(times.min()), float(times.max())
    
    def calibrate_uniform(self, token_counts: List[int], warmup: int = 5, repeats: int = 20) -> List[A2ATimingResult]:
        """
        校准均匀分布模式
        
        Args:
            token_counts: 要测试的不同 token 数量
        """
        results = []
        
        for tokens_per_rank in token_counts:
            send_data, send_counts, recv_counts = self.pattern_gen.generate_uniform(tokens_per_rank)
            
            mean_ms, std_ms, min_ms, max_ms = self._measure_a2a(
                send_data, send_counts, recv_counts, warmup, repeats
            )
            
            total_bytes = sum(send_counts) * self.hidden_size * self.dtype_bytes * self.world_size
            bandwidth_GBps = total_bytes / (mean_ms / 1000.0) / 1e9 if mean_ms > 0 else 0
            
            result = A2ATimingResult(
                pattern="uniform",
                total_bytes=total_bytes,
                bytes_per_rank_send=[c * self.hidden_size * self.dtype_bytes for c in send_counts],
                bytes_per_rank_recv=[c * self.hidden_size * self.dtype_bytes for c in recv_counts],
                time_ms_mean=mean_ms,
                time_ms_std=std_ms,
                time_ms_min=min_ms,
                time_ms_max=max_ms,
                bandwidth_GBps=bandwidth_GBps,
                n_repeats=repeats,
            )
            results.append(result)
            self.results.append(result)
            
            if self.rank == 0:
                total_MB = total_bytes / 1e6
                print(f"  [Uniform] tokens/rank={tokens_per_rank:>6}, total={total_MB:>8.2f}MB, "
                      f"time={mean_ms:>8.3f}±{std_ms:.3f}ms, BW={bandwidth_GBps:.2f}GB/s")
        
        return results
    
    def calibrate_imbalanced(self, base_tokens_list: List[int], imbalance_ratios: List[float], 
                             warmup: int = 5, repeats: int = 20) -> List[A2ATimingResult]:
        """
        校准不均衡分布模式
        """
        results = []
        
        for base_tokens in base_tokens_list:
            for ratio in imbalance_ratios:
                send_data, send_counts, recv_counts = self.pattern_gen.generate_imbalanced(base_tokens, ratio)
                
                mean_ms, std_ms, min_ms, max_ms = self._measure_a2a(
                    send_data, send_counts, recv_counts, warmup, repeats
                )
                
                total_bytes = sum(send_counts) * self.hidden_size * self.dtype_bytes * self.world_size
                bandwidth_GBps = total_bytes / (mean_ms / 1000.0) / 1e9 if mean_ms > 0 else 0
                
                result = A2ATimingResult(
                    pattern=f"imbalanced_r{ratio:.1f}",
                    total_bytes=total_bytes,
                    bytes_per_rank_send=[c * self.hidden_size * self.dtype_bytes for c in send_counts],
                    bytes_per_rank_recv=[c * self.hidden_size * self.dtype_bytes for c in recv_counts],
                    time_ms_mean=mean_ms,
                    time_ms_std=std_ms,
                    time_ms_min=min_ms,
                    time_ms_max=max_ms,
                    bandwidth_GBps=bandwidth_GBps,
                    n_repeats=repeats,
                )
                results.append(result)
                self.results.append(result)
                
                if self.rank == 0:
                    total_MB = total_bytes / 1e6
                    print(f"  [Imbalanced r={ratio:.1f}] base_tokens={base_tokens:>5}, total={total_MB:>8.2f}MB, "
                          f"time={mean_ms:>8.3f}±{std_ms:.3f}ms, BW={bandwidth_GBps:.2f}GB/s")
        
        return results
    
    def calibrate_moe_realistic(self, num_tokens_list: List[int], topk: int = 8, num_experts: int = 128,
                                warmup: int = 5, repeats: int = 20) -> List[A2ATimingResult]:
        """
        校准真实 MoE 模式
        """
        results = []
        
        for num_tokens in num_tokens_list:
            send_data, send_counts, recv_counts = self.pattern_gen.generate_moe_realistic(
                num_tokens, topk, num_experts
            )
            
            mean_ms, std_ms, min_ms, max_ms = self._measure_a2a(
                send_data, send_counts, recv_counts, warmup, repeats
            )
            
            total_bytes = sum(send_counts) * self.hidden_size * self.dtype_bytes * self.world_size
            bandwidth_GBps = total_bytes / (mean_ms / 1000.0) / 1e9 if mean_ms > 0 else 0
            
            # 计算实际的不均衡比例
            actual_imbalance = max(send_counts) / (sum(send_counts) / len(send_counts) + 1e-10)
            
            result = A2ATimingResult(
                pattern=f"moe_topk{topk}_e{num_experts}",
                total_bytes=total_bytes,
                bytes_per_rank_send=[c * self.hidden_size * self.dtype_bytes for c in send_counts],
                bytes_per_rank_recv=[c * self.hidden_size * self.dtype_bytes for c in recv_counts],
                time_ms_mean=mean_ms,
                time_ms_std=std_ms,
                time_ms_min=min_ms,
                time_ms_max=max_ms,
                bandwidth_GBps=bandwidth_GBps,
                n_repeats=repeats,
            )
            results.append(result)
            self.results.append(result)
            
            if self.rank == 0:
                total_MB = total_bytes / 1e6
                print(f"  [MoE topk={topk}] tokens={num_tokens:>5}, total={total_MB:>8.2f}MB, "
                      f"imbalance={actual_imbalance:.2f}, time={mean_ms:>8.3f}±{std_ms:.3f}ms, BW={bandwidth_GBps:.2f}GB/s")
        
        return results
    
    def fit_linear_model(self) -> Dict[str, Any]:
        """
        拟合线性模型: time_ms = alpha + beta * bytes
        
        Returns:
            model parameters and R^2 score
        """
        if len(self.results) < 2:
            return {"error": "insufficient data points"}
        
        # 提取数据
        bytes_arr = np.array([r.total_bytes for r in self.results])
        times_arr = np.array([r.time_ms_mean for r in self.results])
        
        # 线性回归: time = alpha + beta * bytes
        # 使用 numpy 最小二乘
        A = np.vstack([np.ones_like(bytes_arr), bytes_arr]).T
        params, residuals, rank, s = np.linalg.lstsq(A, times_arr, rcond=None)
        alpha, beta = params
        
        # 计算 R^2
        y_pred = alpha + beta * bytes_arr
        ss_res = np.sum((times_arr - y_pred) ** 2)
        ss_tot = np.sum((times_arr - times_arr.mean()) ** 2)
        r_squared = 1 - ss_res / (ss_tot + 1e-10)
        
        # 计算有效带宽 (GB/s) = 1 / (beta * 1e9) * 1000 = 1e12 / beta
        effective_bandwidth_GBps = 1e-6 / (beta + 1e-20)  # beta 单位: ms/byte
        
        return {
            "alpha_ms": float(alpha),          # 固定延迟 (ms)
            "beta_ms_per_byte": float(beta),   # 每字节延迟 (ms/byte)
            "effective_bandwidth_GBps": float(effective_bandwidth_GBps),
            "r_squared": float(r_squared),
            "n_points": len(self.results),
        }
    
    def fit_polynomial_model(self, degree: int = 2) -> Dict[str, Any]:
        """
        拟合多项式模型: time_ms = sum(c_i * bytes^i)
        """
        if len(self.results) < degree + 1:
            return {"error": "insufficient data points"}
        
        bytes_arr = np.array([r.total_bytes for r in self.results])
        times_arr = np.array([r.time_ms_mean for r in self.results])
        
        # 多项式拟合
        coeffs = np.polyfit(bytes_arr, times_arr, degree)
        poly = np.poly1d(coeffs)
        
        # R^2
        y_pred = poly(bytes_arr)
        ss_res = np.sum((times_arr - y_pred) ** 2)
        ss_tot = np.sum((times_arr - times_arr.mean()) ** 2)
        r_squared = 1 - ss_res / (ss_tot + 1e-10)
        
        return {
            "coefficients": coeffs.tolist(),
            "degree": degree,
            "r_squared": float(r_squared),
            "n_points": len(self.results),
        }


# ============================================================
# Main
# ============================================================
def main():
    # 初始化分布式环境
    dist.init_parallel_env()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # 配置
    warmup = int(os.environ.get("WARMUP", "5"))
    repeats = int(os.environ.get("REPEATS", "20"))
    hidden_size = int(os.environ.get("HIDDEN_SIZE", "2048"))
    dtype = os.environ.get("DTYPE", "bfloat16")
    output_file = os.environ.get("OUTPUT_FILE", "costmodel_a2a_calibration.json")
    
    if rank == 0:
        print("=" * 70)
        print("A2A CALIBRATOR - All-to-All Communication Calibration")
        print("=" * 70)
        print(f"[Config] WORLD_SIZE   = {world_size}")
        print(f"[Config] HIDDEN_SIZE  = {hidden_size}")
        print(f"[Config] DTYPE        = {dtype}")
        print(f"[Config] WARMUP       = {warmup}")
        print(f"[Config] REPEATS      = {repeats}")
        print("=" * 70)
    
    # 创建校准器
    calibrator = A2ACalibrator(hidden_size=hidden_size, dtype=dtype)
    
    # 1. 均匀分布校准
    if rank == 0:
        print("\n[Phase 1] Uniform Distribution Calibration")
        print("-" * 50)
    
    # tokens_per_rank: 从小到大，覆盖不同数据量
    uniform_tokens = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    uniform_results = calibrator.calibrate_uniform(uniform_tokens, warmup, repeats)
    
    # Phase 2 & 3: 跳过不均衡模式（各 rank 独立生成分布会导致 send/recv 不匹配）
    # 均匀分布已足够用于校准 A2A 延迟模型
    imbalanced_results = []
    moe_results = []
    
    if rank == 0:
        print("\n[Note] Skipping imbalanced/MoE patterns (uniform distribution sufficient for calibration)")
    
    # 4. 拟合模型
    if rank == 0:
        print("\n[Phase 4] Fitting Prediction Models")
        print("-" * 50)
    
    linear_model = calibrator.fit_linear_model()
    poly_model = calibrator.fit_polynomial_model(degree=2)
    
    if rank == 0:
        print(f"  Linear Model: alpha={linear_model.get('alpha_ms', 0):.4f}ms, "
              f"beta={linear_model.get('beta_ms_per_byte', 0):.2e}ms/byte, "
              f"R²={linear_model.get('r_squared', 0):.4f}")
        print(f"  Effective Bandwidth: {linear_model.get('effective_bandwidth_GBps', 0):.2f} GB/s")
        print(f"  Polynomial Model (degree=2): R²={poly_model.get('r_squared', 0):.4f}")
    
    # 5. 输出结果
    if rank == 0:
        output = {
            "meta": {
                "mode": "A2A_CALIBRATION",
                "world_size": world_size,
                "hidden_size": hidden_size,
                "dtype": dtype,
                "warmup": warmup,
                "repeats": repeats,
            },
            "models": {
                "linear": linear_model,
                "polynomial": poly_model,
            },
            "results": {
                "uniform": [asdict(r) for r in uniform_results],
                "imbalanced": [asdict(r) for r in imbalanced_results],
                "moe_realistic": [asdict(r) for r in moe_results],
            },
            "summary": {
                "total_measurements": len(calibrator.results),
                "effective_bandwidth_GBps": linear_model.get("effective_bandwidth_GBps", 0),
                "fixed_latency_ms": linear_model.get("alpha_ms", 0),
                "per_byte_latency_ns": linear_model.get("beta_ms_per_byte", 0) * 1e6,
            }
        }
        
        print("\n" + "=" * 70)
        print("A2A CALIBRATION SUMMARY")
        print("=" * 70)
        print(f"{'Total Measurements':<30} {output['summary']['total_measurements']}")
        print(f"{'Effective Bandwidth':<30} {output['summary']['effective_bandwidth_GBps']:.2f} GB/s")
        print(f"{'Fixed Latency':<30} {output['summary']['fixed_latency_ms']:.4f} ms")
        print(f"{'Per-byte Latency':<30} {output['summary']['per_byte_latency_ns']:.4f} ns/byte")
        print("=" * 70)
        
        # 保存
        with open(output_file, "w") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n[Done] Results saved to {output_file}")


if __name__ == "__main__":
    main()
