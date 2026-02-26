#!/usr/bin/env python3
"""
PDCostModel - PaddleFormers 分布式训练代价模型主模块

整合:
- 硬件配置
- 计算模型
- 通信模型
- 显存模型

提供统一的预测接口
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import (
    ModelConfig, ParallelConfig, TrainingConfig, HardwareConfig,
    GPUSpec, NetworkSpec, ShardingStage, RecomputeGranularity
)
from .memory_model import MemoryModel, MemoryBreakdown, ShardingConfig, RecomputeConfig
from .compute_model import ComputeModel
from .comm_model import CommModel


@dataclass
class PredictionResult:
    """预测结果"""
    # ========== 时延 (ms) ==========
    step_time_ms: float = 0.0  # 总 step 时间
    
    # 计算时间
    compute_time_ms: float = 0.0
    forward_time_ms: float = 0.0
    backward_time_ms: float = 0.0
    
    # 通信时间
    tp_comm_time_ms: float = 0.0
    dp_comm_time_ms: float = 0.0
    ep_comm_time_ms: float = 0.0
    pp_comm_time_ms: float = 0.0
    sp_comm_time_ms: float = 0.0
    total_comm_time_ms: float = 0.0
    
    # 流水线气泡
    bubble_time_ms: float = 0.0
    bubble_ratio: float = 0.0
    
    # ========== 显存 (GB) ==========
    memory_gb: float = 0.0
    memory_breakdown: MemoryBreakdown = field(default_factory=MemoryBreakdown)
    fits_memory: bool = True
    
    # ========== 效率指标 ==========
    compute_efficiency: float = 0.0  # 计算效率
    mfu: float = 0.0  # Model FLOPs Utilization
    
    # ========== 吞吐量 ==========
    tokens_per_step: int = 0
    tokens_per_second: float = 0.0
    tokens_per_second_per_gpu: float = 0.0
    
    # ========== 配置信息 ==========
    parallel_config: Dict = field(default_factory=dict)
    recompute_overhead: float = 1.0
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "time": {
                "step_time_ms": round(self.step_time_ms, 2),
                "compute_time_ms": round(self.compute_time_ms, 2),
                "forward_time_ms": round(self.forward_time_ms, 2),
                "backward_time_ms": round(self.backward_time_ms, 2),
                "total_comm_time_ms": round(self.total_comm_time_ms, 2),
                "tp_comm_time_ms": round(self.tp_comm_time_ms, 2),
                "dp_comm_time_ms": round(self.dp_comm_time_ms, 2),
                "ep_comm_time_ms": round(self.ep_comm_time_ms, 2),
                "pp_comm_time_ms": round(self.pp_comm_time_ms, 2),
                "bubble_time_ms": round(self.bubble_time_ms, 2),
                "bubble_ratio": round(self.bubble_ratio, 4),
            },
            "memory": self.memory_breakdown.to_dict(),
            "fits_memory": self.fits_memory,
            "efficiency": {
                "compute_efficiency": round(self.compute_efficiency, 4),
                "mfu": round(self.mfu, 4),
            },
            "throughput": {
                "tokens_per_step": self.tokens_per_step,
                "tokens_per_second": round(self.tokens_per_second, 0),
                "tokens_per_second_per_gpu": round(self.tokens_per_second_per_gpu, 0),
            },
            "config": self.parallel_config,
        }
    
    def __str__(self) -> str:
        fits_str = "✅" if self.fits_memory else "❌"
        return (
            f"PredictionResult:\n"
            f"  Step Time: {self.step_time_ms:.2f} ms\n"
            f"    - Compute: {self.compute_time_ms:.2f} ms\n"
            f"    - Communication: {self.total_comm_time_ms:.2f} ms\n"
            f"    - Bubble: {self.bubble_time_ms:.2f} ms ({self.bubble_ratio:.1%})\n"
            f"  Memory: {self.memory_gb:.2f} GB {fits_str}\n"
            f"  MFU: {self.mfu:.1%}\n"
            f"  Throughput: {self.tokens_per_second:,.0f} tok/s "
            f"({self.tokens_per_second_per_gpu:,.0f} tok/s/GPU)"
        )


class PDCostModel:
    """
    PaddleFormers 分布式训练代价模型
    
    用于预测不同并行配置下的:
    - Step 时间
    - 显存占用
    - 硬件利用率
    - 训练吞吐量
    
    使用示例:
        model_config = ModelConfig.from_name("qwen3-30b-a3b")
        costmodel = PDCostModel(model_config)
        
        parallel = ParallelConfig(tp=8, pp=1, dp=1, ep=8)
        result = costmodel.predict(parallel, micro_batch_size=1, seq_len=8192)
        print(result)
        
    带硬件校准:
        costmodel = PDCostModel(model_config, auto_calibrate=True)
        # 或手动校准
        costmodel.calibrate()
    
    框架特性说明:
        PaddleFormers 框架有一个重要特性：存在"最小计算批次"
        - 当 seq_len <= min_compute_seq_len 时，step time 基本恒定
        - 当 seq_len > min_compute_seq_len 时，step time 线性增长
        这是因为框架会将小序列填充到最小计算单元大小
    """
    
    # ========== 框架校准参数（从实验数据拟合）==========
    # 这些参数是针对 PaddleFormers + Qwen3-30B-A3B 的经验值
    # 不同模型/框架可能需要重新校准
    
    # 最小计算序列长度阈值
    MIN_COMPUTE_SEQ_LEN = 2048
    
    # 基础 step time (seq_len <= MIN_COMPUTE_SEQ_LEN 时的固定时间)
    # 注意：这个值会根据 offload 配置动态调整
    BASE_STEP_TIME_S = 12.75
    
    # 序列长度增长斜率 (seq_len > MIN_COMPUTE_SEQ_LEN 时)
    SEQ_LEN_SLOPE_MS_PER_TOKEN = 2.6  # ms/token
    
    # Offload 时间估算参数
    PCIE_BANDWIDTH_GBPS = 16.0  # PCIe 4.0 x16
    OFFLOAD_OVERLAP_RATIO = 0.0  # 实测 offload 基本无法 overlap
    
    def __init__(self, 
                 model_config: ModelConfig,
                 hardware_config: HardwareConfig = None,
                 training_config: TrainingConfig = None,
                 auto_calibrate: bool = False,
                 calibrate_on_predict: bool = False,
                 use_cached_profile: bool = True,
                 node_count: int = 1):
        """
        初始化 CostModel
        
        Args:
            model_config: 模型架构配置
            hardware_config: 硬件配置 (默认 H100-80GB，如果 auto_calibrate=True 则自动检测)
            training_config: 训练配置 (默认 bf16, recompute=full)
            auto_calibrate: 是否在初始化时自动校准硬件
            calibrate_on_predict: 是否在每次预测前重新校准
            use_cached_profile: 是否使用本地缓存的校准配置 (默认 True)
            node_count: 节点数量 (默认 1)
        """
        self.model_config = model_config
        self.training_config = training_config or TrainingConfig()
        self._calibrated = False
        self._calibrate_on_predict = calibrate_on_predict
        self._calibration_result = None
        self._node_count = node_count
        
        # 硬件配置优先级:
        # 1. 用户显式传入 hardware_config
        # 2. use_cached_profile=True 时尝试加载本地缓存
        # 3. auto_calibrate=True 时执行校准
        # 4. 使用默认配置
        if hardware_config is not None:
            # 用户显式传入配置
            self.hardware_config = hardware_config
            self._calibrated = True
        elif use_cached_profile:
            # 尝试加载本地缓存的校准配置
            self.hardware_config = self._load_or_calibrate(verbose=False)
        elif auto_calibrate:
            self.hardware_config = self.calibrate(verbose=True)
        else:
            self.hardware_config = HardwareConfig()
        
        # 初始化子模型
        self._init_sub_models()
    
    def _load_or_calibrate(self, force_calibrate: bool = False, verbose: bool = True) -> HardwareConfig:
        """
        加载本地校准配置，如果不存在则执行校准
        
        Args:
            force_calibrate: 是否强制重新校准
            verbose: 是否打印信息
        
        Returns:
            HardwareConfig: 硬件配置
        """
        from .profile_manager import ProfileManager, auto_calibrate_or_load
        from .calibration import HardwareCalibrator
        
        manager = ProfileManager()
        calibrator = HardwareCalibrator()
        
        # 检测当前硬件
        gpu_name, memory_gb, gpu_count = calibrator.detect_gpu_info()
        
        if not force_calibrate and manager.has_profile(gpu_name, gpu_count, self._node_count):
            # 加载已保存的配置
            profile_data = manager.load_profile(gpu_name, gpu_count, self._node_count)
            if profile_data:
                self._calibrated = True
                if verbose:
                    print(f"✅ 已加载校准配置: {gpu_name} × {gpu_count}")
                return manager.create_hardware_config(profile_data)
        
        # 需要执行校准
        if verbose:
            print(f"🔧 未找到校准配置，开始校准 {gpu_name} × {gpu_count}...")
        
        result = calibrator.calibrate(verbose=verbose)
        self._calibration_result = result
        
        # 保存校准结果
        filepath = manager.save_calibration(result, self._node_count)
        if verbose:
            print(f"💾 校准结果已保存: {filepath}")
        
        self._calibrated = True
        
        return calibrator.create_hardware_config(
            num_nodes=self._node_count,
            gpus_per_node=gpu_count // self._node_count if self._node_count > 0 else gpu_count
        )
    
    def _init_sub_models(self):
        """初始化子模型"""
        self.memory_model = MemoryModel(self.model_config, self.training_config)
        self.compute_model = ComputeModel(self.model_config, self.hardware_config, self.training_config)
        self.comm_model = CommModel(self.hardware_config)
    
    def calibrate(self, 
                  num_nodes: int = 1,
                  gpus_per_node: int = None,
                  device_id: int = 0,
                  test_compute: bool = True,
                  test_memory: bool = True,
                  gemm_size: int = 8192,
                  verbose: bool = True) -> HardwareConfig:
        """
        执行硬件校准
        
        通过实际运行 benchmark 测试 GPU 算力和显存带宽，
        然后更新 HardwareConfig
        
        Args:
            num_nodes: 节点数
            gpus_per_node: 每节点 GPU 数 (默认自动检测)
            device_id: 测试使用的 GPU ID
            test_compute: 是否测试算力
            test_memory: 是否测试显存带宽
            gemm_size: GEMM 测试矩阵大小
            verbose: 是否打印进度
        
        Returns:
            HardwareConfig: 校准后的硬件配置
        """
        from .calibration import HardwareCalibrator
        
        calibrator = HardwareCalibrator(device_id=device_id)
        self._calibration_result = calibrator.calibrate(
            test_compute=test_compute,
            test_memory=test_memory,
            gemm_size=gemm_size,
            verbose=verbose
        )
        
        self.hardware_config = calibrator.create_hardware_config(
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node
        )
        
        # 重新初始化子模型
        self._init_sub_models()
        self._calibrated = True
        
        return self.hardware_config
    
    @property
    def calibration_result(self):
        """获取校准结果"""
        return self._calibration_result
    
    @property
    def is_calibrated(self) -> bool:
        """是否已校准"""
        return self._calibrated
    
    def predict(self,
                parallel: ParallelConfig,
                micro_batch_size: int = None,
                seq_len: int = None,
                max_seq_len: int = None,
                gradient_accumulation_steps: int = None,
                recompute_granularity: str = None,
                tensorwise_offload_optimizer: bool = None,
                tensorwise_offload_ratio: float = None,
                split_param: bool = True,
                sd_release_grads: bool = False) -> PredictionResult:
        """
        预测给定并行配置的性能
        
        Args:
            parallel: 并行配置
            micro_batch_size: micro batch size (默认使用 training_config)
            seq_len: 序列长度 (默认使用 training_config.sequence_length)
            max_seq_len: 最大序列长度，用于激活显存估算 (默认等于 seq_len)
            gradient_accumulation_steps: 梯度累积步数
            recompute_granularity: 重计算粒度 ("none", "selective", "full")
            tensorwise_offload_optimizer: 是否启用 tensorwise 优化器 offload
            tensorwise_offload_ratio: tensorwise offload 比例 (默认 0.95)
            split_param: PaddleFormers ShardingV2 参数分片 (默认 True)
            sd_release_grads: 迭代后释放梯度，降低峰值显存 (默认 False)
        
        Returns:
            PredictionResult: 预测结果
        """
        # 使用默认值
        if micro_batch_size is None:
            micro_batch_size = self.training_config.micro_batch_size
        if seq_len is None:
            seq_len = self.training_config.sequence_length
        if max_seq_len is None:
            max_seq_len = seq_len  # 默认使用当前 seq_len
        if gradient_accumulation_steps is None:
            gradient_accumulation_steps = self.training_config.gradient_accumulation_steps
        
        # 重计算配置
        if recompute_granularity is None:
            recompute_gran = self.training_config.recompute_config
        else:
            recompute_gran_map = {
                "none": RecomputeGranularity.NONE,
                "selective": RecomputeGranularity.SELECTIVE,
                "full": RecomputeGranularity.FULL,
            }
            recompute_gran = recompute_gran_map.get(recompute_granularity.lower(), RecomputeGranularity.FULL)
        
        # tensorwise offload 配置 (默认启用 offload 以节省显存)
        use_tensorwise = tensorwise_offload_optimizer if tensorwise_offload_optimizer is not None else True
        offload_ratio = tensorwise_offload_ratio if tensorwise_offload_ratio is not None else 0.95
        
        result = PredictionResult()
        result.parallel_config = parallel.to_dict()
        
        # ========== 显存预测 ==========
        sharding_config = ShardingConfig(
            stage=parallel.sharding_stage,
            degree=parallel.effective_sharding_degree,
            split_param=split_param,
            release_grads=sd_release_grads,
            tensorwise_offload=use_tensorwise,
            tensorwise_offload_ratio=offload_ratio,
        )
        
        recompute_config = RecomputeConfig(
            granularity=recompute_gran,
            method=self.training_config.recompute_method,
            num_layers=self.training_config.recompute_num_layers,
        )
        
        result.memory_breakdown = self.memory_model.estimate_memory(
            parallel, sharding_config, recompute_config, max_seq_len, micro_batch_size
        )
        result.memory_gb = result.memory_breakdown.total_memory_gb
        result.fits_memory = result.memory_gb <= self.hardware_config.gpu.memory_gb
        
        # ========== 计算时间预测 ==========
        result.recompute_overhead = recompute_config.get_recompute_overhead()
        
        compute_result = self.compute_model.estimate_step_compute_time(
            micro_batch_size, seq_len, parallel,
            gradient_accumulation_steps, result.recompute_overhead
        )
        
        result.forward_time_ms = compute_result["forward_time_ms"]
        result.backward_time_ms = compute_result["backward_time_ms"]
        result.bubble_time_ms = compute_result["bubble_time_ms"]
        result.compute_time_ms = compute_result["compute_time_ms"]
        result.bubble_ratio = compute_result["bubble_ratio"]
        
        # ========== 通信时间预测 ==========
        comm_result = self.comm_model.estimate_step_comm_time(
            self.model_config, self.training_config,
            parallel, gradient_accumulation_steps
        )
        
        result.tp_comm_time_ms = comm_result["tp_comm_time_ms"]
        result.dp_comm_time_ms = comm_result["dp_comm_time_ms"]
        result.ep_comm_time_ms = comm_result["ep_comm_time_ms"]
        result.pp_comm_time_ms = comm_result["pp_comm_time_ms"]
        result.sp_comm_time_ms = comm_result.get("sp_comm_time_ms", 0)
        result.total_comm_time_ms = comm_result["total_comm_time_ms"]
        
        # ========== 额外开销 ==========
        # 1. tensorwise_offload 的 CPU-GPU 数据传输开销
        offload_overhead_ms = 0.0
        if use_tensorwise:
            # 估算优化器更新时的 CPU-GPU 传输时间
            # tensorwise_offload 需要：
            # - 从 CPU 加载优化器状态到 GPU
            # - 在 GPU 上执行更新
            # - 将更新后的状态写回 CPU
            # 
            # 关键点：
            # 1. 每个 GPU 只 offload 自己负责的参数（Sharding 切分后）
            # 2. tensorwise 是流式传输，可以和计算部分 overlap
            # 3. 实际 overlap 效率约 50%
            
            param_count = self.model_config.estimate_parameters()["total"]
            # Sharding 切分后每个 GPU 的参数量
            sharding_degree = parallel.effective_sharding_degree
            params_per_gpu = param_count / sharding_degree
            
            # AdamW: 2 个 fp32 状态 + master weight = 12 bytes per param
            offload_bytes = params_per_gpu * 12 * offload_ratio
            
            # CPU-GPU 带宽约 16 GB/s (PCIe 4.0 x16)
            # 双向传输，但可以和计算 overlap 约 50%
            cpu_gpu_bandwidth_gbps = 16.0
            offload_time_raw = offload_bytes * 2 / (cpu_gpu_bandwidth_gbps * 1e9) * 1000
            offload_overhead_ms = offload_time_raw * 0.5  # 50% 可以 overlap
        
        # 2. 小 batch size 效率惩罚
        # batch_size=1 时 GPU 利用率极低，kernel launch 开销占比大
        batch_efficiency = min(1.0, micro_batch_size / 4.0) * 0.5 + 0.5
        
        # 3. 框架开销 (动态图、Python 调度等)
        # 约 10-20% 的额外开销
        framework_overhead_factor = 1.15
        
        # 4. MoE 负载不均衡开销
        moe_lb_overhead = 1.0
        if parallel.ep > 1 and self.model_config.num_moe_layers > 0:
            # 负载不均衡导致约 10-20% 的额外等待时间
            moe_lb_overhead = 1.15
        
        # ========== 总时延 ==========
        # 通信与计算的重叠
        # TP 通信在关键路径上
        # DP 通信可以与后向部分 overlap
        # EP 通信在 MoE 层关键路径上
        
        overlap_factor = 0.3  # 假设 30% 的通信可以 overlap
        effective_comm_time = (
            result.tp_comm_time_ms +
            result.ep_comm_time_ms +
            result.pp_comm_time_ms +
            result.dp_comm_time_ms * (1 - overlap_factor) +
            result.sp_comm_time_ms
        )
        
        # 计算时间加上各种开销
        adjusted_compute_time = result.compute_time_ms / batch_efficiency * framework_overhead_factor * moe_lb_overhead
        
        result.step_time_ms = adjusted_compute_time + effective_comm_time + offload_overhead_ms
        
        # ========== 效率指标 ==========
        result.compute_efficiency = self._calculate_compute_efficiency(result, parallel)
        result.mfu = self._calculate_mfu(
            result, parallel, micro_batch_size, seq_len, gradient_accumulation_steps
        )
        
        # ========== 吞吐量 ==========
        result.tokens_per_step, result.tokens_per_second, result.tokens_per_second_per_gpu = \
            self._calculate_throughput(result, parallel, micro_batch_size, seq_len, gradient_accumulation_steps)
        
        return result
    
    def predict_calibrated(self,
                           parallel: ParallelConfig,
                           micro_batch_size: int = None,
                           seq_len: int = None,
                           max_seq_len: int = None,
                           gradient_accumulation_steps: int = None,
                           recompute_granularity: str = None,
                           tensorwise_offload_optimizer: bool = None,
                           tensorwise_offload_ratio: float = None,
                           split_param: bool = True,
                           sd_release_grads: bool = False) -> PredictionResult:
        """
        使用校准后的分段线性模型进行预测
        
        这个方法使用从实验数据拟合的模型，更准确地预测 step time。
        
        模型公式:
            if seq_len <= MIN_COMPUTE_SEQ_LEN:
                step_time = BASE_STEP_TIME_S
            else:
                step_time = BASE_STEP_TIME_S + SEQ_LEN_SLOPE * (seq_len - MIN_COMPUTE_SEQ_LEN)
        
        关键发现：
            1. PaddleFormers 框架存在"最小计算批次"(~2048 tokens)
            2. seq_len <= 2048 时，step time 基本恒定（受 offload 主导）
            3. seq_len > 2048 时，step time 线性增长
        
        Args:
            parallel: 并行配置
            其他参数与 predict() 相同
        
        Returns:
            PredictionResult: 预测结果
        """
        # 使用默认值
        if micro_batch_size is None:
            micro_batch_size = self.training_config.micro_batch_size
        if seq_len is None:
            seq_len = self.training_config.sequence_length
        if max_seq_len is None:
            max_seq_len = seq_len
        if gradient_accumulation_steps is None:
            gradient_accumulation_steps = self.training_config.gradient_accumulation_steps
        
        # tensorwise offload 配置
        use_tensorwise = tensorwise_offload_optimizer if tensorwise_offload_optimizer is not None else False
        offload_ratio = tensorwise_offload_ratio if tensorwise_offload_ratio is not None else 0.95
        
        # 重计算配置
        if recompute_granularity is None:
            recompute_gran = self.training_config.recompute_config
        else:
            recompute_gran_map = {
                "none": RecomputeGranularity.NONE,
                "selective": RecomputeGranularity.SELECTIVE,
                "full": RecomputeGranularity.FULL,
            }
            recompute_gran = recompute_gran_map.get(recompute_granularity.lower(), RecomputeGranularity.FULL)
        
        result = PredictionResult()
        result.parallel_config = parallel.to_dict()
        
        # ========== 显存预测（使用原有逻辑）==========
        sharding_config = ShardingConfig(
            stage=parallel.sharding_stage,
            degree=parallel.effective_sharding_degree,
            split_param=split_param,
            release_grads=sd_release_grads,
            tensorwise_offload=use_tensorwise,
            tensorwise_offload_ratio=offload_ratio,
        )
        
        recompute_config = RecomputeConfig(
            granularity=recompute_gran,
            method=self.training_config.recompute_method,
            num_layers=self.training_config.recompute_num_layers,
        )
        
        result.memory_breakdown = self.memory_model.estimate_memory(
            parallel, sharding_config, recompute_config, max_seq_len, micro_batch_size
        )
        result.memory_gb = result.memory_breakdown.total_memory_gb
        result.fits_memory = result.memory_gb <= self.hardware_config.gpu.memory_gb
        result.recompute_overhead = recompute_config.get_recompute_overhead()
        
        # ========== Step Time 预测（使用分段线性模型）==========
        # 1. 计算 Offload 时间（固定开销）
        offload_time_ms = 0.0
        if use_tensorwise:
            param_count = self.model_config.estimate_parameters()["total"]
            sharding_degree = parallel.effective_sharding_degree
            params_per_gpu = param_count / sharding_degree
            offload_bytes = params_per_gpu * 12 * offload_ratio  # AdamW states
            # 双向传输，几乎无 overlap
            offload_time_ms = offload_bytes * 2 / (self.PCIE_BANDWIDTH_GBPS * 1e9) * 1000 * (1 - self.OFFLOAD_OVERLAP_RATIO)
        
        # 2. 使用分段线性模型计算 step time
        # 基础参数是在以下条件下测得的：
        # - seq_len=2048, mbs=1, gas=16, offload=0.95
        # - 对应 step_time = 12.75s
        
        # 分段线性模型（直接预测总 step time，已包含 offload）
        if seq_len <= self.MIN_COMPUTE_SEQ_LEN:
            # seq_len <= 阈值时，step time 固定
            base_step_time_ms = self.BASE_STEP_TIME_S * 1000
        else:
            # seq_len > 阈值时，step time 线性增长
            extra_time_ms = self.SEQ_LEN_SLOPE_MS_PER_TOKEN * (seq_len - self.MIN_COMPUTE_SEQ_LEN)
            base_step_time_ms = self.BASE_STEP_TIME_S * 1000 + extra_time_ms
        
        # 3. 根据其他配置调整
        # 基准配置：mbs=1, gas=16
        
        # 3.1 micro_batch_size 缩放
        # mbs=1 是基准，增加 mbs 会增加计算量但效率也会提升
        if micro_batch_size == 1:
            mbs_factor = 1.0
        else:
            # mbs > 1 时，计算时间增加但效率也提升
            # 假设效率提升 15%，即每增加 1 个 mbs，计算量增加 0.85 倍
            mbs_factor = 1.0 + (micro_batch_size - 1) * 0.85
        
        # 3.2 gradient_accumulation_steps 缩放（基准是 gas=16）
        gas_factor = gradient_accumulation_steps / 16.0
        
        # 调整 step time
        # 注意：offload 时间不随 mbs/gas 缩放
        # 分离 offload 和计算时间进行缩放
        offload_portion_ms = 6800  # 估计 offload 约 6.8s
        compute_portion_ms = base_step_time_ms - offload_portion_ms
        
        # 只有计算部分需要缩放
        scaled_compute_ms = compute_portion_ms * mbs_factor * gas_factor
        
        # 4. 总 step time
        step_time_ms = scaled_compute_ms + offload_portion_ms
        
        # 如果不使用 offload，需要调整
        if not use_tensorwise:
            # 无 offload 时，只有计算部分
            step_time_ms = scaled_compute_ms
        result.step_time_ms = step_time_ms
        
        # ========== 分解时间（用于报告）==========
        # 估算各部分时间占比
        actual_compute_ms = scaled_compute_ms if use_tensorwise else step_time_ms
        result.compute_time_ms = actual_compute_ms * 0.85  # 计算占 85%
        result.total_comm_time_ms = actual_compute_ms * 0.15  # 通信占 15%
        result.forward_time_ms = result.compute_time_ms * 0.33
        result.backward_time_ms = result.compute_time_ms * 0.67
        result.bubble_time_ms = 0.0
        result.bubble_ratio = 0.0
        
        if parallel.pp > 1:
            bubble_ratio = (parallel.pp - 1) / gradient_accumulation_steps
            result.bubble_time_ms = step_time_ms * bubble_ratio
            result.bubble_ratio = bubble_ratio
        
        # ========== 效率和吞吐量 ==========
        result.compute_efficiency = self._calculate_compute_efficiency(result, parallel)
        result.mfu = self._calculate_mfu(
            result, parallel, micro_batch_size, seq_len, gradient_accumulation_steps
        )
        result.tokens_per_step, result.tokens_per_second, result.tokens_per_second_per_gpu = \
            self._calculate_throughput(result, parallel, micro_batch_size, seq_len, gradient_accumulation_steps)
        
        return result
    
    def _calculate_compute_efficiency(self, result: PredictionResult,
                                      parallel: ParallelConfig) -> float:
        """计算效率"""
        if result.step_time_ms <= 0:
            return 0.0
        
        # 计算时间占比
        compute_ratio = result.compute_time_ms / result.step_time_ms
        
        # 并行效率损失
        tp_efficiency = 0.9 if parallel.tp > 1 else 1.0
        pp_efficiency = 1.0 - result.bubble_ratio
        ep_efficiency = 0.85 if parallel.ep > 1 else 1.0
        
        return compute_ratio * tp_efficiency * pp_efficiency * ep_efficiency
    
    def _calculate_mfu(self, result: PredictionResult,
                       parallel: ParallelConfig,
                       micro_batch_size: int,
                       seq_len: int,
                       gradient_accumulation_steps: int) -> float:
        """
        计算 Model FLOPs Utilization (MFU)
        
        MFU = 实际计算的 FLOPs / (峰值算力 × 时间)
        
        对于 MoE 模型，考虑稀疏激活 (只有 TopK 个专家参与计算)
        """
        if result.step_time_ms <= 0:
            return 0.0
        
        h = self.model_config.hidden_size
        num_layers = self.model_config.num_hidden_layers
        kv_heads = self.model_config.num_key_value_heads
        head_dim = self.model_config.head_dim
        
        # ========== Attention FLOPs (每 token) ==========
        # Q proj: h * h, K proj: h * kv_size, V proj: h * kv_size, O proj: h * h
        # QK^T + softmax + score*V: 约 4 * h * seq_len (简化)
        kv_size = kv_heads * head_dim
        attention_flops = 2 * (2 * h * h + 2 * h * kv_size) + 4 * h * seq_len
        
        # ========== FFN FLOPs (每 token) ==========
        if self.model_config.num_experts > 1:
            # MoE: 只有 TopK 个专家激活
            moe_ffn = self.model_config.moe_intermediate_size
            topk = self.model_config.num_experts_per_tok
            # Gate + Up + Down for TopK experts
            ffn_flops = 2 * 3 * h * moe_ffn * topk
            # Router
            router_flops = 2 * h * self.model_config.num_experts
            moe_layer_flops = attention_flops + ffn_flops + router_flops
            
            # Dense layers
            dense_ffn = self.model_config.intermediate_size
            dense_layer_flops = attention_flops + 2 * 3 * h * dense_ffn
            
            # 混合
            flops_per_token = (
                dense_layer_flops * self.model_config.num_dense_layers +
                moe_layer_flops * self.model_config.num_moe_layers
            )
        else:
            # Dense model
            ffn = self.model_config.intermediate_size
            layer_flops = attention_flops + 2 * 3 * h * ffn
            flops_per_token = layer_flops * num_layers
        
        # 总 tokens
        tokens = micro_batch_size * seq_len
        total_tokens = tokens * gradient_accumulation_steps * parallel.dp
        
        # 总 FLOPs (前向 + 后向 ≈ 3x 前向)
        total_flops = flops_per_token * total_tokens * 3
        
        # 峰值 FLOPs (所有 GPU)
        world_size = parallel.dp * parallel.tp * parallel.pp
        peak_tflops = self.hardware_config.gpu.get_tflops(self.training_config.dtype)
        peak_flops = peak_tflops * 1e12 * world_size * (result.step_time_ms / 1000)
        
        # MFU = 实际计算量 / 理论最大计算量
        mfu = total_flops / peak_flops if peak_flops > 0 else 0.0
        
        return min(mfu, 1.0)
    
    def _calculate_throughput(self, result: PredictionResult,
                              parallel: ParallelConfig,
                              micro_batch_size: int,
                              seq_len: int,
                              gradient_accumulation_steps: int) -> Tuple[int, float, float]:
        """计算吞吐量"""
        if result.step_time_ms <= 0:
            return 0, 0.0, 0.0
        
        # 每 step tokens 数
        tokens_per_step = micro_batch_size * seq_len * gradient_accumulation_steps * parallel.dp
        
        # 总吞吐量
        step_time_seconds = result.step_time_ms / 1000.0
        tokens_per_second = tokens_per_step / step_time_seconds
        
        # 每卡吞吐量
        world_size = parallel.dp * parallel.tp * parallel.pp
        tokens_per_second_per_gpu = tokens_per_second / world_size
        
        return tokens_per_step, tokens_per_second, tokens_per_second_per_gpu
    
    def rank_configurations(self, configs: List[Dict], 
                            top_k: int = 10,
                            micro_batch_size: int = None,
                            seq_len: int = None) -> List[Dict]:
        """
        对并行配置列表进行排序
        
        排序依据:
        1. 是否满足显存约束
        2. step 时延 (越小越好)
        3. 硬件利用率
        
        Args:
            configs: 并行配置列表
            top_k: 返回前 k 个最优配置
            micro_batch_size: micro batch size
            seq_len: 序列长度
        
        Returns:
            排序后的配置列表
        """
        results = []
        
        for cfg in configs:
            try:
                parallel = ParallelConfig.from_dict(cfg)
                prediction = self.predict(parallel, micro_batch_size, seq_len)
                
                results.append({
                    "rank": 0,
                    "config": cfg,
                    "config_str": str(parallel),
                    "step_time_ms": prediction.step_time_ms,
                    "memory_gb": prediction.memory_gb,
                    "fits_memory": prediction.fits_memory,
                    "mfu": prediction.mfu,
                    "tokens_per_second": prediction.tokens_per_second,
                    "tokens_per_second_per_gpu": prediction.tokens_per_second_per_gpu,
                    "prediction": prediction.to_dict(),
                })
            except Exception as e:
                print(f"Warning: Failed to predict config {cfg}: {e}")
                continue
        
        # 排序: 先按是否满足显存，再按时延
        results.sort(key=lambda x: (not x["fits_memory"], x["step_time_ms"]))
        
        # 更新排名
        for i, r in enumerate(results):
            r["rank"] = i + 1
        
        # 打印报告
        self._print_ranking_report(results[:top_k])
        
        return results[:top_k]
    
    def _print_ranking_report(self, results: List[Dict]):
        """打印排序报告"""
        if not results:
            return
        
        print("\n" + "=" * 120)
        print("🚀 PDCostModel - 并行配置排序报告")
        print("=" * 120)
        print(f"{'排名':<4} {'配置':<30} {'时延(ms)':<12} {'显存(GB)':<10} "
              f"{'约束':<6} {'MFU':<8} {'tok/s':<12} {'tok/s/GPU':<12}")
        print("-" * 120)
        
        for r in results:
            fits = "✅" if r["fits_memory"] else "❌"
            print(f"{r['rank']:<4} {r['config_str']:<30} "
                  f"{r['step_time_ms']:<12.2f} {r['memory_gb']:<10.2f} "
                  f"{fits:<6} {r['mfu']:<8.1%} "
                  f"{r['tokens_per_second']:<12,.0f} "
                  f"{r['tokens_per_second_per_gpu']:<12,.0f}")
        
        print("-" * 120)
        
        if results:
            best = results[0]
            print(f"\n📊 最优配置: {best['config_str']}")
            print(f"   • 预计时延: {best['step_time_ms']:.2f} ms")
            print(f"   • 显存占用: {best['memory_gb']:.2f} GB")
            print(f"   • MFU: {best['mfu']:.1%}")
            print(f"   • 吞吐量: {best['tokens_per_second']:,.0f} tok/s")
    
    def generate_search_space(self, total_gpus: int,
                              max_tp: int = 8,
                              max_pp: int = 8) -> List[Dict]:
        """
        生成并行配置搜索空间
        
        Args:
            total_gpus: 总 GPU 数
            max_tp: 最大 TP 度
            max_pp: 最大 PP 度
        
        Returns:
            配置列表
        """
        configs = []
        num_experts = self.model_config.num_experts
        
        for tp in [1, 2, 4, 8]:
            if tp > max_tp or tp > total_gpus:
                continue
            
            for pp in [1, 2, 4, 8]:
                if pp > max_pp or tp * pp > total_gpus:
                    continue
                
                dp = total_gpus // (tp * pp)
                if dp < 1 or tp * pp * dp != total_gpus:
                    continue
                
                # EP 搜索
                ep_candidates = [1]
                if num_experts > 1:
                    for ep in [2, 4, 8, 16, 32]:
                        if ep <= num_experts and ep <= total_gpus:
                            ep_candidates.append(ep)
                
                for ep in ep_candidates:
                    # Sharding 搜索
                    for sharding in ["stage1", "stage2"]:
                        configs.append({
                            "tp": tp,
                            "pp": pp,
                            "dp": dp,
                            "ep": ep,
                            "sharding": sharding,
                        })
        
        return configs
    
    def search_best_throughput(self,
                               total_gpus: int = 8,
                               seq_lens: List[int] = None,
                               micro_batch_sizes: List[int] = None,
                               gas_values: List[int] = None,
                               use_offload: bool = True,
                               offload_ratio: float = 0.95,
                               top_k: int = 10,
                               sort_by: str = "throughput") -> List[Dict]:
        """
        搜索最优吞吐量配置
        
        遍历不同的并行配置和训练超参数组合，找出满足显存约束且吞吐量最高的配置。
        
        Args:
            total_gpus: 总 GPU 数
            seq_lens: 序列长度列表 (默认 [2048, 4096, 8192])
            micro_batch_sizes: micro batch size 列表 (默认 [1, 2])
            gas_values: gradient accumulation steps 列表 (默认 [8, 16, 32])
            use_offload: 是否使用 tensorwise offload
            offload_ratio: offload 比例
            top_k: 返回前 k 个最优配置
            sort_by: 排序依据 ("throughput" 或 "step_time")
        
        Returns:
            排序后的配置列表，每个元素包含完整的配置和预测结果
        """
        if seq_lens is None:
            seq_lens = [2048, 4096, 8192]
        if micro_batch_sizes is None:
            micro_batch_sizes = [1, 2]
        if gas_values is None:
            gas_values = [8, 16, 32]
        
        results = []
        num_experts = self.model_config.num_experts
        
        # 生成并行配置搜索空间
        parallel_configs = []
        for tp in [1, 2, 4, 8]:
            if tp > total_gpus:
                continue
            for pp in [1, 2, 4]:
                if tp * pp > total_gpus:
                    continue
                dp = total_gpus // (tp * pp)
                if dp < 1 or tp * pp * dp != total_gpus:
                    continue
                
                # 关键约束：当使用 tensorwise_offload 时，DP 必须 > 1
                # 因为 offload 依赖 Sharding 机制，DP=1 时 Sharding 不工作
                if use_offload and dp <= 1:
                    continue
                
                # EP 候选
                ep_candidates = [1]
                if num_experts > 1:
                    for ep in [2, 4, 8]:
                        if ep <= num_experts and ep <= dp:
                            ep_candidates.append(ep)
                
                for ep in ep_candidates:
                    for sharding in ["stage1", "stage2"]:
                        parallel_configs.append({
                            "tp": tp, "pp": pp, "dp": dp, "ep": ep, "sharding": sharding
                        })
        
        print(f"\n🔍 搜索配置空间...")
        print(f"   并行配置数: {len(parallel_configs)}")
        print(f"   seq_lens: {seq_lens}")
        print(f"   micro_batch_sizes: {micro_batch_sizes}")
        print(f"   gas_values: {gas_values}")
        print(f"   总组合数: {len(parallel_configs) * len(seq_lens) * len(micro_batch_sizes) * len(gas_values)}")
        
        # 遍历所有组合
        for pcfg in parallel_configs:
            parallel = ParallelConfig.from_dict(pcfg)
            
            for seq_len in seq_lens:
                for mbs in micro_batch_sizes:
                    for gas in gas_values:
                        try:
                            result = self.predict_calibrated(
                                parallel,
                                micro_batch_size=mbs,
                                seq_len=seq_len,
                                gradient_accumulation_steps=gas,
                                tensorwise_offload_optimizer=use_offload,
                                tensorwise_offload_ratio=offload_ratio
                            )
                            
                            # 计算全局 batch size
                            global_bs = mbs * gas * parallel.dp
                            
                            results.append({
                                "rank": 0,
                                "parallel": pcfg,
                                "parallel_str": str(parallel),
                                "seq_len": seq_len,
                                "micro_batch_size": mbs,
                                "gradient_accumulation_steps": gas,
                                "global_batch_size": global_bs,
                                "step_time_s": result.step_time_ms / 1000,
                                "memory_gb": result.memory_gb,
                                "fits_memory": result.fits_memory,
                                "mfu": result.mfu,
                                "tokens_per_second": result.tokens_per_second,
                                "tokens_per_second_per_gpu": result.tokens_per_second_per_gpu,
                                "tokens_per_step": result.tokens_per_step,
                            })
                        except Exception as e:
                            continue
        
        # 过滤满足显存约束的配置
        valid_results = [r for r in results if r["fits_memory"]]
        
        # 排序
        if sort_by == "throughput":
            valid_results.sort(key=lambda x: -x["tokens_per_second_per_gpu"])
        else:
            valid_results.sort(key=lambda x: x["step_time_s"])
        
        # 更新排名
        for i, r in enumerate(valid_results):
            r["rank"] = i + 1
        
        # 打印报告
        self._print_throughput_report(valid_results[:top_k], total_gpus)
        
        return valid_results[:top_k]
    
    def _print_throughput_report(self, results: List[Dict], total_gpus: int):
        """打印吞吐量排序报告"""
        if not results:
            print("\n❌ 没有找到满足显存约束的配置！")
            return
        
        print("\n" + "=" * 140)
        print(f"🚀 PDCostModel - {total_gpus}卡最优吞吐量配置 Top {len(results)}")
        print("=" * 140)
        print(f"{'排名':<4} {'并行配置':<28} {'seq_len':<8} {'mbs':<4} {'gas':<4} "
              f"{'step(s)':<9} {'显存(GB)':<10} {'tok/s/GPU':<12} {'global_bs':<10}")
        print("-" * 140)
        
        for r in results:
            print(f"{r['rank']:<4} {r['parallel_str']:<28} {r['seq_len']:<8} "
                  f"{r['micro_batch_size']:<4} {r['gradient_accumulation_steps']:<4} "
                  f"{r['step_time_s']:<9.2f} {r['memory_gb']:<10.1f} "
                  f"{r['tokens_per_second_per_gpu']:<12,.0f} {r['global_batch_size']:<10}")
        
        print("-" * 140)
        
        if results:
            best = results[0]
            print(f"\n🏆 最优配置:")
            print(f"   并行: {best['parallel_str']}")
            print(f"   seq_len={best['seq_len']}, mbs={best['micro_batch_size']}, gas={best['gradient_accumulation_steps']}")
            print(f"   预计时延: {best['step_time_s']:.2f} s")
            print(f"   显存占用: {best['memory_gb']:.1f} GB")
            print(f"   吞吐量: {best['tokens_per_second_per_gpu']:,.0f} tok/s/GPU")
            print(f"   Global Batch Size: {best['global_batch_size']}")
    
    def generate_yaml_config(self, config: Dict, output_path: str = None) -> str:
        """
        根据搜索结果生成 YAML 配置文件
        
        Args:
            config: search_best_throughput 返回的配置字典
            output_path: 输出路径 (可选)
        
        Returns:
            YAML 配置内容字符串
        """
        yaml_content = f'''## Qwen3-30B-A3B-Base 自动生成配置 ##
## 配置: seq_len={config['seq_len']}, mbs={config['micro_batch_size']}, gas={config['gradient_accumulation_steps']} ##
## 预测吞吐量: {config['tokens_per_second_per_gpu']:.0f} tok/s/GPU ##

## data
train_dataset_type: erniekit
eval_dataset_type: erniekit
train_dataset_path: ./data/pt/train.jsonl
train_dataset_prob: "1.0"
eval_dataset_path: ./data/pt/eval.jsonl
eval_dataset_prob: "1.0"

eval_iters: 10
max_seq_len: {config['seq_len']}
num_samples_each_epoch: 6000000
packing: true
mix_strategy: concat
truncate_packing: true
dataloader_shuffle: false
dataloader_num_workers: 8
prefetch_factor: 4

### model
model_name_or_path: /root/paddlejob/workspace/env_run/zhangdongqi/Qwen3-30B-A3B-Base
_attn_implementation: flashmask
use_qk_norm: true

### benchmark 配置
stage: PT
fine_tuning: full
seed: 23
do_train: true
do_eval: false
per_device_eval_batch_size: 1
per_device_train_batch_size: {config['micro_batch_size']}
num_train_epochs: 1
max_steps: 30
eval_steps: 100
evaluation_strategy: steps
save_steps: 999999
save_total_limit: 0
save_strategy: "no"
logging_steps: 1
gradient_accumulation_steps: {config['gradient_accumulation_steps']}
logging_dir: ./log_benchmark_auto
output_dir: ./benchmark_output_auto
disable_tqdm: true
eval_accumulation_steps: 16

# train warmup
warmup_steps: 5
learning_rate: 1.0e-5

# performance - 并行配置
tensor_model_parallel_size: {config['parallel']['tp']}
sequence_parallel: {'true' if config['parallel']['tp'] > 1 else 'false'}
pipeline_model_parallel_size: {config['parallel']['pp']}
use_expert_parallel: {'true' if config['parallel']['ep'] > 1 else 'false'}
expert_model_parallel_size: {config['parallel']['ep']}

# recompute
recompute_granularity: full
recompute_method: uniform
recompute_num_layers: 1
using_sonic_moe: false

# sharding
sharding: {config['parallel']['sharding']}
split_param: true
stage1_overlap: true
sd_release_grads: true

apply_rope_fusion: true
moe_grouped_gemm: true
moe_ep_barrier: false
moe_router_fusion: true
moe_router_force_load_balancing: false

# pp配置
pp_delay_scale_loss: true
overlap_p2p_comm: true
variable_seq_lengths: true
best_unbalanced_scheduler: true
tp_delay_scale_loss: true

optim: adamw
bf16: true
fp16_opt_level: O2
amp_master_grad: true

# checkpoint
save_checkpoint_format: "flex_checkpoint"
load_checkpoint_format: "flex_checkpoint"
tensorwise_offload_optimizer: true
benchmark: true
continue_training: false
'''
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(yaml_content)
            print(f"\n✅ 配置已保存到: {output_path}")
        
        return yaml_content

    def save_config(self, path: str):
        """保存配置"""
        config_data = {
            "model": {
                "num_hidden_layers": self.model_config.num_hidden_layers,
                "hidden_size": self.model_config.hidden_size,
                "intermediate_size": self.model_config.intermediate_size,
                "num_attention_heads": self.model_config.num_attention_heads,
                "num_key_value_heads": self.model_config.num_key_value_heads,
                "num_experts": self.model_config.num_experts,
                "num_experts_per_tok": self.model_config.num_experts_per_tok,
                "vocab_size": self.model_config.vocab_size,
            },
            "hardware": {
                "gpu_name": self.hardware_config.gpu.name,
                "gpu_memory_gb": self.hardware_config.gpu.memory_gb,
                "bf16_tflops": self.hardware_config.gpu.bf16_tflops,
                "num_nodes": self.hardware_config.num_nodes,
                "gpus_per_node": self.hardware_config.gpus_per_node,
            },
            "training": self.training_config.to_dict(),
        }
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    @classmethod
    def from_config_file(cls, path: str) -> "PDCostModel":
        """从配置文件加载"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        model_config = ModelConfig.from_dict(data.get("model", {}))
        
        hw_data = data.get("hardware", {})
        hardware_config = HardwareConfig(
            gpu=GPUSpec.from_name(hw_data.get("gpu_name", "H100-80GB-HBM3")),
            num_nodes=hw_data.get("num_nodes", 1),
            gpus_per_node=hw_data.get("gpus_per_node", 8),
        )
        
        training_config = TrainingConfig.from_dict(data.get("training", {}))
        
        return cls(model_config, hardware_config, training_config)


# ==================== 便捷函数 ====================

def create_qwen3_30b_costmodel(gpu_memory_gb: float = 80.0,
                               num_nodes: int = 1,
                               gpus_per_node: int = 8) -> PDCostModel:
    """
    创建 Qwen3-30B-A3B 的 CostModel
    """
    model_config = ModelConfig.from_name("qwen3-30b-a3b")
    hardware_config = HardwareConfig(
        gpu=GPUSpec(memory_gb=gpu_memory_gb),
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
    )
    
    return PDCostModel(model_config, hardware_config)


def create_deepseek_v3_costmodel(gpu_memory_gb: float = 80.0,
                                 num_nodes: int = 1,
                                 gpus_per_node: int = 8) -> PDCostModel:
    """
    创建 DeepSeek-V3 的 CostModel
    """
    model_config = ModelConfig.from_name("deepseek-v3")
    hardware_config = HardwareConfig(
        gpu=GPUSpec(memory_gb=gpu_memory_gb),
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
    )
    
    return PDCostModel(model_config, hardware_config)