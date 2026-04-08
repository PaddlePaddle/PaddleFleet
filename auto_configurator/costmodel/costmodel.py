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
import logging
import math
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from .config import (
    ModelConfig, ParallelConfig, TrainingConfig, HardwareConfig,
    GPUSpec, ShardingStage, RecomputeGranularity
)
from .submodels.memory_model import MemoryModel, MemoryBreakdown, ShardingConfig, RecomputeConfig
from .submodels.compute_model import ComputeModel
from .submodels.comm_model import CommModel


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
    dp_exposed_comm_time_ms: float = 0.0
    ep_comm_time_ms: float = 0.0
    pp_comm_time_ms: float = 0.0
    sp_comm_time_ms: float = 0.0
    total_comm_time_ms: float = 0.0
    effective_comm_time_ms: float = 0.0
    
    # 流水线气泡
    bubble_time_ms: float = 0.0
    bubble_ratio: float = 0.0
    framework_overhead_ms: float = 0.0
    recompute_time_ms: float = 0.0
    offload_overhead_ms: float = 0.0
    optimizer_step_time_ms: float = 0.0
    runtime_overhead_ms: float = 0.0
    
    # ========== 显存 (GB) ==========
    memory_gb: float = 0.0
    allocated_memory_gb: float = 0.0
    reserved_memory_gb: float = 0.0
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
    stage_layer_counts: List[int] = field(default_factory=list)
    stage_forward_micro_ms: List[float] = field(default_factory=list)
    stage_backward_micro_ms: List[float] = field(default_factory=list)
    stage_tp_comm_micro_ms: List[float] = field(default_factory=list)
    stage_ep_comm_micro_ms: List[float] = field(default_factory=list)
    stage_sp_comm_micro_ms: List[float] = field(default_factory=list)
    stage_dp_exposed_step_ms: List[float] = field(default_factory=list)
    stage_cycle_micro_ms: List[float] = field(default_factory=list)
    slowest_stage_id: int = 0
    slowest_stage_time_ms: float = 0.0
    stage_recompute_detail: List[Dict] = field(default_factory=list)

    # ========== 置信度 ==========
    confidence: float = 0.0
    memory_confidence: float = 0.0
    time_confidence: float = 0.0
    confidence_breakdown: Dict = field(default_factory=dict)
    confidence_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "time": {
                "step_time_ms": round(self.step_time_ms, 2),
                "compute_time_ms": round(self.compute_time_ms, 2),
                "forward_time_ms": round(self.forward_time_ms, 2),
                "backward_time_ms": round(self.backward_time_ms, 2),
                "total_comm_time_ms": round(self.total_comm_time_ms, 2),
                "effective_comm_time_ms": round(self.effective_comm_time_ms, 2),
                "tp_comm_time_ms": round(self.tp_comm_time_ms, 2),
                "dp_comm_time_ms": round(self.dp_comm_time_ms, 2),
                "dp_exposed_comm_time_ms": round(self.dp_exposed_comm_time_ms, 2),
                "ep_comm_time_ms": round(self.ep_comm_time_ms, 2),
                "pp_comm_time_ms": round(self.pp_comm_time_ms, 2),
                "bubble_time_ms": round(self.bubble_time_ms, 2),
                "bubble_ratio": round(self.bubble_ratio, 4),
                "framework_overhead_ms": round(self.framework_overhead_ms, 2),
                "recompute_time_ms": round(self.recompute_time_ms, 2),
                "offload_overhead_ms": round(self.offload_overhead_ms, 2),
                "optimizer_step_time_ms": round(self.optimizer_step_time_ms, 2),
                "runtime_overhead_ms": round(self.runtime_overhead_ms, 2),
            },
            "memory": {
                "allocated_memory_gb": round(self.allocated_memory_gb, 3),
                "reserved_memory_gb": round(self.reserved_memory_gb, 3),
                **self.memory_breakdown.to_dict(),
            },
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
            "stage": {
                "layer_counts": list(self.stage_layer_counts),
                "forward_micro_ms": [round(v, 2) for v in self.stage_forward_micro_ms],
                "backward_micro_ms": [round(v, 2) for v in self.stage_backward_micro_ms],
                "tp_comm_micro_ms": [round(v, 2) for v in self.stage_tp_comm_micro_ms],
                "ep_comm_micro_ms": [round(v, 2) for v in self.stage_ep_comm_micro_ms],
                "sp_comm_micro_ms": [round(v, 2) for v in self.stage_sp_comm_micro_ms],
                "dp_exposed_step_ms": [round(v, 2) for v in self.stage_dp_exposed_step_ms],
                "cycle_micro_ms": [round(v, 2) for v in self.stage_cycle_micro_ms],
                "slowest_stage_id": self.slowest_stage_id,
                "slowest_stage_time_ms": round(self.slowest_stage_time_ms, 2),
                "recompute_detail": list(self.stage_recompute_detail),
            },
            "confidence": {
                "confidence": round(self.confidence, 4),
                "memory_confidence": round(self.memory_confidence, 4),
                "time_confidence": round(self.time_confidence, 4),
                "breakdown": dict(self.confidence_breakdown),
                "reasons": list(self.confidence_reasons),
            },
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
            f"    - Allocated: {self.allocated_memory_gb:.2f} GB\n"
            f"    - Reserved:  {self.reserved_memory_gb:.2f} GB\n"
            f"  MFU: {self.mfu:.1%}\n"
            f"  Throughput: {self.tokens_per_second:,.0f} tok/s "
            f"({self.tokens_per_second_per_gpu:,.0f} tok/s/GPU)"
        )


class _BasePDCostModel:
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
        result = costmodel.predict(parallel, micro_batch_size=1, max_seq_len=8192)
        logger.debug(result)
        
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
    
    HOST_TO_DEVICE_BANDWIDTH_GBPS = 16.0
    DEVICE_TO_HOST_BANDWIDTH_GBPS = 16.0
    OFFLOAD_BUCKET_MIN_BYTES = 64 * 1024 * 1024
    OFFLOAD_BUCKET_MAX_BYTES = 512 * 1024 * 1024
    
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

        # 尽量从 hardware_spec.py 注入已校准硬件曲线（尤其 BF16 curve）
        self._enrich_hardware_from_saved_spec()
        
        # 初始化子模型
        self._init_sub_models()

    def _enrich_hardware_from_saved_spec(self):
        """
        当存在 hardware_spec.py 对应条目时，用其覆盖/补全硬件参数。
        """
        try:
            from .utils.calibration import HardwareCalibrator

            calibrator = HardwareCalibrator()
            gpu_name = self.hardware_config.gpu.name
            gpus_per_node = int(self.hardware_config.gpus_per_node)
            node_count = self.hardware_config.num_nodes
            if not gpu_name or gpus_per_node <= 0:
                return
            loaded = calibrator.load_result(
                gpu_name=gpu_name,
                gpus_per_node=gpus_per_node,
            )
            if loaded is None:
                return

            calibrated_cfg = calibrator.create_hardware_config(
                num_nodes=node_count,
                gpus_per_node=self.hardware_config.gpus_per_node,
            )
            self.hardware_config = calibrated_cfg
            self._calibrated = True
        except Exception:
            # 没有可用校准信息时保持原配置
            return
    
    def _load_or_calibrate(self, force_calibrate: bool = False, verbose: bool = True) -> HardwareConfig:
        """
        加载本地校准配置，如果不存在则执行校准
        
        Args:
            force_calibrate: 是否强制重新校准
            verbose: 是否打印信息
        
        Returns:
            HardwareConfig: 硬件配置
        """
        from .utils.calibration import HardwareCalibrator, get_hardware_config

        # 与公共入口保持一致：默认优先复用 hardware_spec.py 中的已采集数据，
        # 在无 GPU 的离线预测环境也能正确构造多机硬件配置。
        config = get_hardware_config(
            node_count=self._node_count,
            force_calibrate=force_calibrate,
            verbose=verbose,
        )

        calibrator = HardwareCalibrator()
        self._calibration_result = calibrator.load_result(
            gpu_name=config.gpu.name,
            gpus_per_node=config.gpus_per_node,
        )
        self._calibrated = True
        return config
    
    def _init_sub_models(self):
        """初始化子模型"""
        self.memory_model = MemoryModel(
            self.model_config, self.training_config, self.hardware_config
        )
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
        from .utils.calibration import HardwareCalibrator
        
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
                max_seq_len: int = None,
                gradient_accumulation_steps: int = None,
                recompute_granularity: str = None,
                recompute_method: Optional[str] = None,
                recompute_num_layers: Optional[int] = None,
                recompute_modules: Optional[List[str]] = None,
                tensorwise_offload_optimizer: bool = None,
                tensorwise_offload_ratio: float = None,
                split_param: bool = True,
                sd_release_grads: Optional[bool] = None,
                overlap_p2p_comm: Optional[bool] = None,
                use_batch_p2p_comm: Optional[bool] = None,
                p2p_cache_shape: Optional[bool] = None,
                stage1_overlap: Optional[bool] = None,
                enable_sharding_comm_overlap: Optional[bool] = None,
                variable_seq_lengths: Optional[bool] = None,
                enable_dynamic_shape: Optional[bool] = None,
                clear_every_step_cache: Optional[bool] = None,
                best_unbalanced_scheduler: Optional[bool] = None,
                hybrid_parallel_topo_order: Optional[str] = None,
                num_empty_layers_add_in_head: Optional[int] = None,
                num_empty_layers_add_in_tail: Optional[int] = None,
                attn_implementation: Optional[str] = None,
                apply_rope_fusion: Optional[bool] = None,
                use_qk_norm: Optional[bool] = None,
                moe_token_dispatcher_type: Optional[str] = None,
                moe_grouped_gemm: Optional[bool] = None,
                moe_router_fusion: Optional[bool] = None,
                moe_expert_fusion: Optional[bool] = None,
                moe_shared_expert_overlap: Optional[bool] = None,
                moe_ep_barrier: Optional[bool] = None) -> PredictionResult:
        """
        预测给定并行配置的性能
        
        Args:
            parallel: 并行配置
            micro_batch_size: micro batch size (默认使用 training_config)
            max_seq_len: 序列长度 (默认使用 training_config.sequence_length)
            gradient_accumulation_steps: 梯度累积步数
            recompute_granularity: 重计算粒度 ("none", "selective", "full")
            recompute_method: 重计算方法 ("uniform", "block", "first_n")
            recompute_num_layers: 重计算层数/单元大小
            recompute_modules: selective recompute 的模块列表
            tensorwise_offload_optimizer: 是否启用 tensorwise 优化器 offload
            tensorwise_offload_ratio: tensorwise offload 比例 (默认 0.95)
            split_param: PaddleFormers ShardingV2 参数分片 (默认 True)
            sd_release_grads: 迭代后释放梯度，降低峰值显存。None 表示自动推断
            overlap_p2p_comm: PP 通信与计算重叠
            use_batch_p2p_comm: PP 采用 batch send/recv
            p2p_cache_shape: PP 缓存 shape 对应的运行时 buffer
            stage1_overlap: Stage1 sharding overlap
            enable_sharding_comm_overlap: 显式启用 sharding 通信 overlap
            variable_seq_lengths: 变长序列
            enable_dynamic_shape: 动态 shape 运行时
            clear_every_step_cache: 每步后是否清理运行时缓存
            best_unbalanced_scheduler: pipeline 不均衡调度
            hybrid_parallel_topo_order: 并行拓扑顺序
            num_empty_layers_add_in_head: pipeline head 额外空层
            num_empty_layers_add_in_tail: pipeline tail 额外空层
            attn_implementation: attention kernel 实现
            apply_rope_fusion: rope 融合
            use_qk_norm: Q/K 额外 RMSNorm（如 Qwen3）
            moe_token_dispatcher_type: MoE token dispatcher 类型
            moe_grouped_gemm: MoE expert grouped GEMM
            moe_router_fusion: router fusion
            moe_expert_fusion: expert fusion
            moe_shared_expert_overlap: shared expert overlap
            moe_ep_barrier: MoE EP barrier
        
        Returns:
            PredictionResult: 预测结果
        """
        # 使用默认值
        if micro_batch_size is None:
            micro_batch_size = self.training_config.micro_batch_size
        if max_seq_len is None:
            max_seq_len = self.training_config.sequence_length
        if gradient_accumulation_steps is None:
            gradient_accumulation_steps = self.training_config.gradient_accumulation_steps

        runtime_training_config = TrainingConfig.from_dict(
            self.training_config.to_dict()
        )
        runtime_training_config.micro_batch_size = micro_batch_size
        runtime_training_config.sequence_length = max_seq_len
        runtime_training_config.gradient_accumulation_steps = (
            gradient_accumulation_steps
        )
        if overlap_p2p_comm is not None:
            runtime_training_config.overlap_p2p_comm = bool(overlap_p2p_comm)
        if use_batch_p2p_comm is not None:
            runtime_training_config.use_batch_p2p_comm = bool(use_batch_p2p_comm)
        if p2p_cache_shape is not None:
            runtime_training_config.p2p_cache_shape = bool(p2p_cache_shape)
        if stage1_overlap is not None:
            runtime_training_config.stage1_overlap = bool(stage1_overlap)
        if enable_sharding_comm_overlap is not None:
            runtime_training_config.enable_sharding_comm_overlap = bool(
                enable_sharding_comm_overlap
            )
        if variable_seq_lengths is not None:
            runtime_training_config.variable_seq_lengths = bool(
                variable_seq_lengths
            )
        if enable_dynamic_shape is not None:
            runtime_training_config.enable_dynamic_shape = bool(enable_dynamic_shape)
        if clear_every_step_cache is not None:
            runtime_training_config.clear_every_step_cache = bool(
                clear_every_step_cache
            )
        if best_unbalanced_scheduler is not None:
            runtime_training_config.best_unbalanced_scheduler = bool(
                best_unbalanced_scheduler
            )
        if hybrid_parallel_topo_order is not None:
            runtime_training_config.hybrid_parallel_topo_order = str(
                hybrid_parallel_topo_order
            )
        if num_empty_layers_add_in_head is not None:
            runtime_training_config.num_empty_layers_add_in_head = int(
                num_empty_layers_add_in_head
            )
        if num_empty_layers_add_in_tail is not None:
            runtime_training_config.num_empty_layers_add_in_tail = int(
                num_empty_layers_add_in_tail
            )
        if attn_implementation is not None:
            runtime_training_config.attn_implementation = str(attn_implementation)
        if apply_rope_fusion is not None:
            runtime_training_config.apply_rope_fusion = bool(apply_rope_fusion)
        if use_qk_norm is not None:
            runtime_training_config.use_qk_norm = bool(use_qk_norm)
        if moe_token_dispatcher_type is not None:
            runtime_training_config.moe_token_dispatcher_type = str(
                moe_token_dispatcher_type
            )
        if moe_grouped_gemm is not None:
            runtime_training_config.moe_grouped_gemm = bool(moe_grouped_gemm)
        if moe_router_fusion is not None:
            runtime_training_config.moe_router_fusion = bool(moe_router_fusion)
        if moe_expert_fusion is not None:
            runtime_training_config.moe_expert_fusion = bool(moe_expert_fusion)
        if moe_shared_expert_overlap is not None:
            runtime_training_config.moe_shared_expert_overlap = bool(
                moe_shared_expert_overlap
            )
        if moe_ep_barrier is not None:
            runtime_training_config.moe_ep_barrier = bool(moe_ep_barrier)

        # 一些 pipeline/runtime 行为在框架中会由其它开关隐式触发。
        # 这里仅在调用方未显式指定时做语义推断，避免多机显存系统性低估。
        if (
            enable_dynamic_shape is None
            and runtime_training_config.variable_seq_lengths
        ):
            runtime_training_config.enable_dynamic_shape = True
        # PaddleFleet 的 pipeline runtime 中，overlap_p2p_comm 与
        # batch_p2p_comm 不能同时启用；开启 overlap 时会退回 non-batch p2p。
        if parallel.pp > 1 and runtime_training_config.overlap_p2p_comm:
            runtime_training_config.use_batch_p2p_comm = False
        if (
            p2p_cache_shape is None
            and parallel.pp > 1
            and (
                runtime_training_config.use_batch_p2p_comm
                or runtime_training_config.variable_seq_lengths
                or runtime_training_config.enable_dynamic_shape
            )
        ):
            runtime_training_config.p2p_cache_shape = True
        if (
            clear_every_step_cache is None
            and parallel.pp > 1
            and (
                runtime_training_config.overlap_p2p_comm
                or runtime_training_config.use_batch_p2p_comm
                or runtime_training_config.variable_seq_lengths
                or runtime_training_config.enable_dynamic_shape
            )
        ):
            runtime_training_config.clear_every_step_cache = False
        
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
        
        # tensorwise offload 仅在显式开启或训练配置中开启时生效。
        use_tensorwise = (
            bool(tensorwise_offload_optimizer)
            if tensorwise_offload_optimizer is not None
            else bool(runtime_training_config.tensorwise_offload_optimizer)
        )
        offload_ratio = (
            float(tensorwise_offload_ratio)
            if tensorwise_offload_ratio is not None
            else float(runtime_training_config.tensorwise_offload_ratio)
        )
        use_sd_release_grads = (
            bool(sd_release_grads)
            if sd_release_grads is not None
            else bool(runtime_training_config.sd_release_grads)
        )
        sharding_degree = self._resolve_sharding_degree(parallel)
        
        result = PredictionResult()
        result.parallel_config = parallel.to_dict()
        
        # ========== 显存预测 ==========
        sharding_config = ShardingConfig(
            stage=parallel.sharding_stage,
            degree=sharding_degree,
            split_param=split_param,
            release_grads=use_sd_release_grads,
            tensorwise_offload=use_tensorwise,
            tensorwise_offload_ratio=offload_ratio,
        )
        
        recompute_config = RecomputeConfig(
            granularity=recompute_gran,
            method=recompute_method or self.training_config.recompute_method,
            num_layers=(
                recompute_num_layers
                if recompute_num_layers is not None
                else self.training_config.recompute_num_layers
            ),
            modules=tuple(recompute_modules or self.training_config.recompute_modules),
        )
        
        memory_model = MemoryModel(
            self.model_config, runtime_training_config, self.hardware_config
        )
        peak_memory_breakdown = memory_model.estimate_memory(
            parallel, sharding_config, recompute_config, max_seq_len, micro_batch_size
        )
        result.memory_breakdown = peak_memory_breakdown
        result.allocated_memory_gb = peak_memory_breakdown.allocated_memory_gb
        result.reserved_memory_gb = peak_memory_breakdown.reserved_memory_gb
        result.memory_gb = peak_memory_breakdown.total_memory_gb
        result.fits_memory = result.memory_gb <= self.hardware_config.gpu.memory_gb
        result.parallel_config["memory_peak_stage_id"] = int(getattr(peak_memory_breakdown, "peak_stage_id", 0))
        result.parallel_config["display_memory_stage_id"] = 0 if int(getattr(parallel, "pp", 1) or 1) > 1 else int(getattr(peak_memory_breakdown, "peak_stage_id", 0))
        
        # ========== 计算时间预测 ==========
        # compute_model 需要使用当前调用的 runtime training config，
        # 否则像 empty layers / deepEP / dynamic shape 这类运行时开关不会生效。
        self.compute_model.training = runtime_training_config
        compute_result = self.compute_model.estimate_step_compute_time(
            micro_batch_size, max_seq_len, parallel,
            gradient_accumulation_steps,
            recompute_granularity=recompute_gran,
            recompute_method=recompute_config.method,
            recompute_num_layers=recompute_config.num_layers,
            recompute_modules=recompute_config.normalized_modules(),
        )
        result.recompute_overhead = compute_result.get(
            "recompute_overhead",
            recompute_config.get_recompute_overhead(),
        )
        
        result.forward_time_ms = compute_result["forward_time_ms"]
        result.backward_time_ms = compute_result["backward_time_ms"]
        result.bubble_time_ms = compute_result["bubble_time_ms"]
        result.compute_time_ms = compute_result["compute_time_ms"]
        result.bubble_ratio = compute_result["bubble_ratio"]
        result.framework_overhead_ms = compute_result.get("framework_overhead_ms", 0.0)
        result.recompute_time_ms = compute_result.get("recompute_time_ms", 0.0)
        result.runtime_overhead_ms = compute_result.get("runtime_overhead_ms", 0.0)
        result.stage_forward_micro_ms = list(
            compute_result.get("stage_forward_micro_ms", []) or []
        )
        result.stage_backward_micro_ms = list(
            compute_result.get("stage_backward_micro_ms", []) or []
        )
        
        # ========== 通信时间预测 ==========
        # 通信模型需要使用当前调用的 mbs/max_seq_len，而不是初始化时的 training_config 默认值
        comm_training_config = TrainingConfig.from_dict(
            runtime_training_config.to_dict()
        )
        comm_result = self.comm_model.estimate_step_comm_time(
            self.model_config, comm_training_config,
            parallel, gradient_accumulation_steps,
            stage_forward_time_ms=compute_result.get("stage_forward_micro_ms"),
            stage_backward_time_ms=compute_result.get("stage_backward_micro_ms"),
        )
        
        result.tp_comm_time_ms = comm_result["tp_comm_time_ms"]
        result.dp_comm_time_ms = comm_result["dp_comm_time_ms"]
        result.dp_exposed_comm_time_ms = comm_result.get(
            "dp_exposed_comm_time_ms", result.dp_comm_time_ms
        )
        result.ep_comm_time_ms = comm_result["ep_comm_time_ms"]
        result.pp_comm_time_ms = comm_result["pp_comm_time_ms"]
        result.sp_comm_time_ms = comm_result.get("sp_comm_time_ms", 0)
        result.total_comm_time_ms = comm_result["total_comm_time_ms"]
        result.stage_tp_comm_micro_ms = list(
            comm_result.get("stage_tp_comm_micro_ms", []) or []
        )
        result.stage_ep_comm_micro_ms = list(
            comm_result.get("stage_ep_comm_micro_ms", []) or []
        )
        result.stage_sp_comm_micro_ms = list(
            comm_result.get("stage_sp_comm_micro_ms", []) or []
        )
        result.stage_dp_exposed_step_ms = list(
            comm_result.get("dp_stage_exposed_comm_time_ms", []) or []
        )
        
        # ========== 额外开销 ==========
        # 1. tensorwise_offload 的 CPU-GPU 传输与 update 开销
        offload_overhead_ms = 0.0
        optimizer_step_time_ms = 0.0
        if use_tensorwise:
            offload_overhead_ms = self._estimate_offload_overhead_ms(
                parallel=parallel,
                sharding_degree=sharding_degree,
                offload_ratio=offload_ratio,
                training_config=runtime_training_config,
                compute_result=compute_result,
                comm_result=comm_result,
            )
        if sharding_degree > 0:
            optimizer_step_time_ms = self._estimate_optimizer_step_time_ms(
                parallel=parallel,
                sharding_degree=sharding_degree,
                offload_ratio=offload_ratio if use_tensorwise else 0.0,
                training_config=runtime_training_config,
                compute_result=compute_result,
                comm_result=comm_result,
                offload_tail_ms=offload_overhead_ms,
            )
        result.offload_overhead_ms = offload_overhead_ms
        result.optimizer_step_time_ms = optimizer_step_time_ms
        
        # ========== 总时延 ==========
        # 通信与计算的重叠
        # TP 通信在关键路径上
        # DP 通信可以与后向部分 overlap
        # EP 通信在 MoE 层关键路径上
        
        effective_comm_time = (
            result.tp_comm_time_ms +
            result.ep_comm_time_ms +
            result.pp_comm_time_ms +
            result.dp_exposed_comm_time_ms +
            result.sp_comm_time_ms
        )
        result.effective_comm_time_ms = effective_comm_time
        
        # 最终 step time 由子模型结果合成，避免重复叠加启发式惩罚
        result.step_time_ms = max(
            0.0,
            result.compute_time_ms +
            effective_comm_time +
            optimizer_step_time_ms
        )

        stage_count = max(1, int(parallel.pp))
        result.stage_layer_counts = [
            len(self.compute_model._stage_layer_indices(parallel, stage_id))
            for stage_id in range(stage_count)
        ]
        result.stage_cycle_micro_ms = []
        for stage_id in range(stage_count):
            forward_micro = (
                float(result.stage_forward_micro_ms[stage_id])
                if stage_id < len(result.stage_forward_micro_ms)
                else 0.0
            )
            backward_micro = (
                float(result.stage_backward_micro_ms[stage_id])
                if stage_id < len(result.stage_backward_micro_ms)
                else 0.0
            )
            tp_comm_micro = (
                float(result.stage_tp_comm_micro_ms[stage_id])
                if stage_id < len(result.stage_tp_comm_micro_ms)
                else 0.0
            )
            ep_comm_micro = (
                float(result.stage_ep_comm_micro_ms[stage_id])
                if stage_id < len(result.stage_ep_comm_micro_ms)
                else 0.0
            )
            sp_comm_micro = (
                float(result.stage_sp_comm_micro_ms[stage_id])
                if stage_id < len(result.stage_sp_comm_micro_ms)
                else 0.0
            )
            result.stage_cycle_micro_ms.append(
                forward_micro + backward_micro + tp_comm_micro + ep_comm_micro + sp_comm_micro
            )
        if result.stage_cycle_micro_ms:
            result.slowest_stage_id = max(
                range(len(result.stage_cycle_micro_ms)),
                key=result.stage_cycle_micro_ms.__getitem__,
            )
            result.slowest_stage_time_ms = float(
                result.stage_cycle_micro_ms[result.slowest_stage_id]
            )

        # ========== 效率指标 ==========
        result.compute_efficiency = self._calculate_compute_efficiency(result, parallel)
        result.mfu = self._calculate_mfu(
            result, parallel, micro_batch_size, max_seq_len, gradient_accumulation_steps
        )
        
        # ========== 吞吐量 ==========
        result.tokens_per_step, result.tokens_per_second, result.tokens_per_second_per_gpu = \
            self._calculate_throughput(result, parallel, micro_batch_size, max_seq_len, gradient_accumulation_steps)

        self._finalize_confidence(result, parallel, runtime_training_config)
        
        return result

    def predict_per_stage(self,
                          parallel: ParallelConfig,
                          per_stage_recompute_granularity: List[str] = None,
                          per_stage_recompute_method: List[str] = None,
                          per_stage_recompute_num_layers: List[int] = None,
                          per_stage_offload: List[bool] = None,
                          per_stage_offload_ratio: List[float] = None,
                          **kwargs) -> "PredictionResult":
        """
        支持每个 PP stage 使用不同 recompute/offload 策略的预测方法。

        对 **显存** 部分，使用 MemoryModel.estimate_memory_per_stage()
        为每个 stage 传入独立的 RecomputeConfig。

        对 **计算时间** 部分，使用各 stage 中最激进的 recompute 策略
        （因为 ComputeModel 内部已经按 stage 层数分别估算，但 recompute
        粒度只能全局指定）。

        对 **offload 开销** 部分，使用各 stage 中是否有任何一个开启 offload
        来决定全局 offload 估算（因为 offload 开销取决于瓶颈 stage）。

        Args:
            parallel: 并行配置
            per_stage_recompute_granularity: 每 stage 的 recompute 粒度列表
                (如 ["full", "selective", "none", ...])。长度应 == PP。
            per_stage_recompute_method: 每 stage 的 recompute 方法。
            per_stage_recompute_num_layers: 每 stage 的 recompute 层数。
            per_stage_offload: 每 stage 是否 offload。
            per_stage_offload_ratio: 每 stage 的 offload 比例。
            **kwargs: 其余参数透传给 predict()。

        Returns:
            PredictionResult: 预测结果（显存部分为 per-stage 精确估算）
        """
        num_stages = max(1, int(parallel.pp))

        # ---------- 确定全局最激进的 recompute 用于 compute model ----------
        recompute_gran_map = {
            "none": RecomputeGranularity.NONE,
            "selective": RecomputeGranularity.SELECTIVE,
            "full": RecomputeGranularity.FULL,
        }
        # 激进程度排序: full > selective > none
        aggressiveness = {"none": 0, "selective": 1, "full": 2}

        if per_stage_recompute_granularity and len(per_stage_recompute_granularity) == num_stages:
            most_aggressive_idx = max(
                range(num_stages),
                key=lambda i: aggressiveness.get(
                    per_stage_recompute_granularity[i].lower(), 0
                ),
            )
            global_recompute_gran = per_stage_recompute_granularity[most_aggressive_idx]
            global_recompute_method = (
                per_stage_recompute_method[most_aggressive_idx]
                if per_stage_recompute_method and len(per_stage_recompute_method) == num_stages
                else None
            )
            global_recompute_num_layers = (
                per_stage_recompute_num_layers[most_aggressive_idx]
                if per_stage_recompute_num_layers and len(per_stage_recompute_num_layers) == num_stages
                else None
            )
        else:
            global_recompute_gran = kwargs.pop("recompute_granularity", None)
            global_recompute_method = kwargs.pop("recompute_method", None)
            global_recompute_num_layers = kwargs.pop("recompute_num_layers", None)

        # ---------- 确定全局 offload 设置 ----------
        any_offload = False
        max_offload_ratio = 0.95
        if per_stage_offload and len(per_stage_offload) == num_stages:
            any_offload = any(per_stage_offload)
            if per_stage_offload_ratio and len(per_stage_offload_ratio) == num_stages:
                ratios_with_offload = [
                    r for o, r in zip(per_stage_offload, per_stage_offload_ratio) if o
                ]
                if ratios_with_offload:
                    max_offload_ratio = max(ratios_with_offload)
        else:
            any_offload = kwargs.pop("tensorwise_offload_optimizer", None)
            max_offload_ratio_kw = kwargs.pop("tensorwise_offload_ratio", None)
            if max_offload_ratio_kw is not None:
                max_offload_ratio = max_offload_ratio_kw

        # ---------- 先调用普通 predict 获取计算/通信/吞吐等结果 ----------
        result = self.predict(
            parallel,
            recompute_granularity=global_recompute_gran,
            recompute_method=global_recompute_method,
            recompute_num_layers=global_recompute_num_layers,
            tensorwise_offload_optimizer=any_offload,
            tensorwise_offload_ratio=max_offload_ratio if any_offload else None,
            **kwargs,
        )

        # ---------- 用 per-stage recompute 重新估算显存 ----------
        if per_stage_recompute_granularity and len(per_stage_recompute_granularity) == num_stages:
            # 构建 per-stage RecomputeConfig 列表
            per_stage_rc = []
            for sid in range(num_stages):
                gran_str = per_stage_recompute_granularity[sid].lower()
                gran_enum = recompute_gran_map.get(gran_str, RecomputeGranularity.FULL)
                method = (
                    per_stage_recompute_method[sid]
                    if per_stage_recompute_method and sid < len(per_stage_recompute_method)
                    else self.training_config.recompute_method
                )
                num_layers = (
                    per_stage_recompute_num_layers[sid]
                    if per_stage_recompute_num_layers and sid < len(per_stage_recompute_num_layers)
                    else self.training_config.recompute_num_layers
                )
                per_stage_rc.append(RecomputeConfig(
                    granularity=gran_enum,
                    method=method,
                    num_layers=num_layers,
                    modules=tuple(self.training_config.recompute_modules),
                ))

            # 构建 runtime training config (与 predict 内部一致)
            micro_batch_size = kwargs.get("micro_batch_size") or self.training_config.micro_batch_size
            max_seq_len = kwargs.get("max_seq_len") or self.training_config.sequence_length
            runtime_training_config = TrainingConfig.from_dict(
                self.training_config.to_dict()
            )
            if kwargs.get("micro_batch_size"):
                runtime_training_config.micro_batch_size = micro_batch_size
            if kwargs.get("max_seq_len"):
                runtime_training_config.sequence_length = max_seq_len

            use_tensorwise = bool(any_offload) if any_offload is not None else False
            offload_ratio = max_offload_ratio
            use_sd_release_grads = bool(
                kwargs.get("sd_release_grads")
                if kwargs.get("sd_release_grads") is not None
                else runtime_training_config.sd_release_grads
            )
            sharding_degree = self._resolve_sharding_degree(parallel)

            sharding_config = ShardingConfig(
                stage=parallel.sharding_stage,
                degree=sharding_degree,
                split_param=kwargs.get("split_param", True),
                release_grads=use_sd_release_grads,
                tensorwise_offload=use_tensorwise,
                tensorwise_offload_ratio=offload_ratio,
            )

            memory_model = MemoryModel(
                self.model_config, runtime_training_config, self.hardware_config
            )
            mem_breakdown = memory_model.estimate_memory_per_stage(
                parallel, sharding_config, per_stage_rc, max_seq_len, micro_batch_size
            )
            result.memory_breakdown = mem_breakdown
            result.allocated_memory_gb = mem_breakdown.allocated_memory_gb
            result.reserved_memory_gb = mem_breakdown.reserved_memory_gb
            result.memory_gb = mem_breakdown.total_memory_gb
            result.fits_memory = result.memory_gb <= self.hardware_config.gpu.memory_gb

        return result

    def _finalize_confidence(self, result: PredictionResult, parallel: ParallelConfig, runtime_training_config: TrainingConfig) -> None:
        structure = 1.0
        if int(getattr(self.model_config, "num_experts", 1) or 1) > 1:
            structure -= 0.03
        if bool(getattr(self.model_config, "uses_low_rank_attention", False)):
            structure -= 0.02
        if int(getattr(parallel, "cp", 1) or 1) > 1:
            structure -= 0.10
        structure = max(0.35, min(1.0, structure))

        hardware = 1.0
        gpu = self.hardware_config.gpu
        if not getattr(gpu, "bf16_gemm_samples", None) and not getattr(gpu, "fp16_gemm_samples", None):
            hardware -= 0.20
        network = self.hardware_config.network
        if int(getattr(parallel, "tp", 1) or 1) > 1 and not getattr(network, "intra_node_bw_curve", None):
            hardware -= 0.08
        if int(getattr(self.hardware_config, "num_nodes", 1) or 1) > 1 and not getattr(network, "inter_node_bw_curve", None):
            hardware -= 0.12
        hardware = max(0.35, min(1.0, hardware))

        extrapolation = 1.0
        max_pos = max(1, int(getattr(self.model_config, "max_position_embeddings", runtime_training_config.sequence_length) or runtime_training_config.sequence_length))
        seq_ratio = float(runtime_training_config.sequence_length) / float(max_pos)
        if seq_ratio > 1.0:
            extrapolation -= min(0.20, 0.10 * (seq_ratio - 1.0) + 0.03)
        elif seq_ratio > 0.85:
            extrapolation -= 0.03
        headroom = float(self.hardware_config.gpu.memory_gb) - float(result.reserved_memory_gb)
        if headroom < 4.0:
            extrapolation -= 0.14
        elif headroom < 8.0:
            extrapolation -= 0.08
        extrapolation = max(0.30, min(1.0, extrapolation))

        peaks = [
            float(getattr(result.memory_breakdown, "reserved_candidate_forward_backward_gb", 0.0)),
            float(getattr(result.memory_breakdown, "reserved_candidate_loss_gb", 0.0)),
            float(getattr(result.memory_breakdown, "reserved_candidate_optimizer_gb", 0.0)),
            float(getattr(result.memory_breakdown, "reserved_candidate_post_step_gb", 0.0)),
            float(getattr(result.memory_breakdown, "allocated_peak_memory_gb", 0.0)),
        ]
        peaks = sorted([max(0.0, x) for x in peaks], reverse=True)
        gap = ((peaks[0] - peaks[1]) / peaks[0]) if len(peaks) >= 2 and peaks[0] > 0 else 1.0
        peak_clarity = max(0.35, min(1.0, 0.55 + 0.90 * gap))

        stability = 1.0
        if bool(getattr(runtime_training_config, "variable_seq_lengths", False)):
            stability -= 0.08
        if bool(getattr(runtime_training_config, "enable_dynamic_shape", False)):
            stability -= 0.08
        if bool(getattr(runtime_training_config, "overlap_p2p_comm", False)):
            stability -= 0.04
        if bool(getattr(runtime_training_config, "tensorwise_offload_optimizer", False)):
            stability -= 0.06
        if bool(getattr(runtime_training_config, "best_unbalanced_scheduler", False)):
            stability -= 0.04
        stability = max(0.35, min(1.0, stability))

        memory_conf = 0.34 * structure + 0.18 * hardware + 0.20 * extrapolation + 0.20 * peak_clarity + 0.08 * stability
        time_conf = 0.26 * structure + 0.30 * hardware + 0.24 * extrapolation + 0.20 * stability
        confidence = 0.55 * memory_conf + 0.45 * time_conf

        reasons = []
        if int(getattr(parallel, "cp", 1) or 1) > 1:
            reasons.append("cp is ignored semantically; predictions assume no context-parallel effects")
        if bool(getattr(runtime_training_config, "variable_seq_lengths", False)):
            reasons.append("variable sequence lengths increase runtime variance")
        if bool(getattr(runtime_training_config, "enable_dynamic_shape", False)):
            reasons.append("dynamic shape execution increases runtime variance")
        if peak_clarity < 0.60:
            reasons.append("multiple memory peak candidates are close")
        if seq_ratio > 1.0:
            reasons.append("sequence length exceeds nominal max_position_embeddings")

        result.confidence = max(0.0, min(1.0, float(confidence)))
        result.memory_confidence = max(0.0, min(1.0, float(memory_conf)))
        result.time_confidence = max(0.0, min(1.0, float(time_conf)))
        result.confidence_breakdown = {
            "structure_coverage": round(structure, 4),
            "hardware_coverage": round(hardware, 4),
            "extrapolation_safety": round(extrapolation, 4),
            "peak_clarity": round(peak_clarity, 4),
            "runtime_stability": round(stability, 4),
        }
        result.confidence_reasons = reasons

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
        data_degree = self._effective_data_degree(parallel)
        total_tokens = tokens * gradient_accumulation_steps * data_degree
        
        # 总 FLOPs (前向 + 后向 ≈ 3x 前向)
        total_flops = flops_per_token * total_tokens * 3
        
        # 峰值 FLOPs (所有 GPU)
        world_size = parallel.tp * parallel.pp * data_degree
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
        
        data_degree = self._effective_data_degree(parallel)

        # 每 step tokens 数
        tokens_per_step = micro_batch_size * seq_len * gradient_accumulation_steps * data_degree
        
        # 总吞吐量
        step_time_seconds = result.step_time_ms / 1000.0
        tokens_per_second = tokens_per_step / step_time_seconds
        
        # 每卡吞吐量
        world_size = parallel.tp * parallel.pp * data_degree
        tokens_per_second_per_gpu = tokens_per_second / world_size
        
        return tokens_per_step, tokens_per_second, tokens_per_second_per_gpu

    def _resolve_sharding_degree(self, parallel: ParallelConfig) -> int:
        """
        解析 Sharding 度数。

        当用户未显式设置 `sharding_degree` 且 `dp==1` 时，
        PaddleFormers 常见拓扑会把 sharding 轴映射到 TP/PP 之外的全 GPU 轴。
        """
        if parallel.sharding_stage == ShardingStage.NONE:
            return 1
        if parallel.sharding_degree > 0:
            return max(1, parallel.sharding_degree)
        if parallel.dp > 1:
            return max(1, parallel.dp)

        model_partition_degree = max(1, parallel.tp * parallel.pp)
        inferred = self.hardware_config.total_gpus // model_partition_degree
        return max(1, inferred)

    def _effective_data_degree(self, parallel: ParallelConfig) -> int:
        """
        计算 tokens 统计和 MFU 所使用的数据维度。
        """
        if parallel.dp > 1:
            return parallel.dp
        if parallel.sharding_stage != ShardingStage.NONE:
            return self._resolve_sharding_degree(parallel)
        if parallel.ep > 1:
            model_partition_degree = max(1, parallel.tp * parallel.pp)
            max_axis = max(1, self.hardware_config.total_gpus // model_partition_degree)
            return max(1, min(max_axis, parallel.ep))
        return 1

    def _estimate_offload_bucket_bytes(self, state_bytes: float) -> int:
        return int(
            min(
                self.OFFLOAD_BUCKET_MAX_BYTES,
                max(self.OFFLOAD_BUCKET_MIN_BYTES, state_bytes / 32.0),
            )
        )

    def _estimate_optimizer_update_compute_ms(self,
                                              param_count: float,
                                              dtype_bytes: int) -> float:
        """
        估算 AdamW update kernel 的 GPU 端执行时间。

        近似按 memory-bound kernel 建模：
        - grad 读: dtype_bytes
        - param 读写: 2 * dtype_bytes
        - master / m / v 读写: 6 * 4 bytes
        """
        if param_count <= 0:
            return 0.0
        update_bytes = float(param_count) * (3.0 * dtype_bytes + 24.0)
        effective_bw = max(1.0, float(self.hardware_config.gpu.memory_bandwidth_gbps))
        return update_bytes / (effective_bw * 1e9) * 1000.0

    def _simulate_offload_pipeline_tail(self,
                                        state_bytes: float,
                                        update_compute_ms: float,
                                        bucket_count: int,
                                        host_to_device_bw_gbps: float,
                                        device_to_host_bw_gbps: float,
                                        prefetch_window_ms: float,
                                        update_ready_window_ms: float) -> float:
        if state_bytes <= 0 or bucket_count <= 0:
            return 0.0

        h2d_bw = max(1.0, float(host_to_device_bw_gbps))
        d2h_bw = max(1.0, float(device_to_host_bw_gbps))
        h2d_bucket_ms = (state_bytes / bucket_count) / (h2d_bw * 1e9) * 1000.0
        d2h_bucket_ms = (state_bytes / bucket_count) / (d2h_bw * 1e9) * 1000.0
        update_bucket_ms = max(0.0, float(update_compute_ms)) / bucket_count

        transfer_stream_free = 0.0
        compute_stream_free = 0.0
        finish_time_ms = 0.0
        pre_step_end_ms = max(0.0, max(prefetch_window_ms, update_ready_window_ms))

        for bucket_idx in range(bucket_count):
            prefetch_ready_ms = prefetch_window_ms * bucket_idx / bucket_count
            h2d_start_ms = max(transfer_stream_free, prefetch_ready_ms)
            h2d_end_ms = h2d_start_ms + h2d_bucket_ms
            transfer_stream_free = h2d_end_ms

            update_ready_ms = update_ready_window_ms * (bucket_idx + 1) / bucket_count
            update_start_ms = max(compute_stream_free, h2d_end_ms, update_ready_ms)
            update_end_ms = update_start_ms + update_bucket_ms
            compute_stream_free = update_end_ms

            d2h_start_ms = max(transfer_stream_free, update_end_ms)
            d2h_end_ms = d2h_start_ms + d2h_bucket_ms
            transfer_stream_free = d2h_end_ms
            finish_time_ms = d2h_end_ms

        return max(0.0, finish_time_ms - pre_step_end_ms)

    def _simulate_offload_pipeline_duration(self,
                                            state_bytes: float,
                                            update_compute_ms: float,
                                            bucket_count: int,
                                            host_to_device_bw_gbps: float,
                                            device_to_host_bw_gbps: float) -> float:
        """估算完整 offload/update/writeback pipeline 的持续时间。"""
        if state_bytes <= 0 or bucket_count <= 0:
            return 0.0

        h2d_bw = max(1.0, float(host_to_device_bw_gbps))
        d2h_bw = max(1.0, float(device_to_host_bw_gbps))
        h2d_bucket_ms = (state_bytes / bucket_count) / (h2d_bw * 1e9) * 1000.0
        d2h_bucket_ms = (state_bytes / bucket_count) / (d2h_bw * 1e9) * 1000.0
        update_bucket_ms = max(0.0, float(update_compute_ms)) / bucket_count

        transfer_stream_free = 0.0
        compute_stream_free = 0.0
        finish_time_ms = 0.0
        for _ in range(bucket_count):
            h2d_start_ms = transfer_stream_free
            h2d_end_ms = h2d_start_ms + h2d_bucket_ms
            transfer_stream_free = h2d_end_ms

            update_start_ms = max(compute_stream_free, h2d_end_ms)
            update_end_ms = update_start_ms + update_bucket_ms
            compute_stream_free = update_end_ms

            d2h_start_ms = max(transfer_stream_free, update_end_ms)
            d2h_end_ms = d2h_start_ms + d2h_bucket_ms
            transfer_stream_free = d2h_end_ms
            finish_time_ms = d2h_end_ms

        return finish_time_ms

    def _estimate_stage_optimizer_tensor_count(self,
                                               parallel: ParallelConfig,
                                               stage_id: int) -> int:
        """近似估算某个 PP stage 参与 optimizer step 的参数 tensor 数。"""
        layer_indices = self.compute_model._stage_layer_indices(parallel, stage_id)
        experts_per_gpu = max(1, self.model_config.num_experts // max(1, parallel.ep))
        shared_tensors = (
            3 if self.model_config.effective_shared_expert_intermediate_size > 0 else 0
        )

        tensor_count = 0
        for layer_idx in layer_indices:
            tensor_count += 4  # q, k, v, o
            tensor_count += 2  # norms
            if self.compute_model._is_moe_layer(layer_idx):
                tensor_count += 1  # router
                tensor_count += 3 * experts_per_gpu
                tensor_count += shared_tensors
            else:
                tensor_count += 3  # dense gate/up/down

        if stage_id == 0:
            tensor_count += 1  # embedding
        if stage_id == max(1, int(parallel.pp)) - 1:
            tensor_count += 2  # final norm + lm_head
        return tensor_count

    def _estimate_offload_overhead_ms(self,
                                      parallel: ParallelConfig,
                                      sharding_degree: int,
                                      offload_ratio: float,
                                      training_config: TrainingConfig,
                                      compute_result: Dict[str, float],
                                      comm_result: Dict[str, float]) -> float:
        if sharding_degree <= 0:
            return 0.0

        ratio = max(0.0, min(1.0, float(offload_ratio)))
        if ratio <= 0:
            return 0.0

        stage_param_counts = self.comm_model._estimate_stage_parameter_counts_per_gpu(
            self.model_config, parallel
        )
        if not stage_param_counts:
            return 0.0

        stage_backward_time_ms = compute_result.get("stage_backward_micro_ms", []) or []
        dp_stage_raw_time_ms = comm_result.get("dp_stage_raw_comm_time_ms", []) or []
        dp_stage_exposed_time_ms = comm_result.get("dp_stage_exposed_comm_time_ms", []) or []

        offload_stage_overheads = []
        pp = max(1, int(parallel.pp))
        bubble_time_ms = max(0.0, float(compute_result.get("bubble_time_ms", 0.0)))

        for stage_id, stage_params in enumerate(stage_param_counts):
            sharded_params = float(stage_params) / max(1, sharding_degree)
            offloaded_params = sharded_params * ratio
            if offloaded_params <= 0:
                offload_stage_overheads.append(0.0)
                continue

            state_bytes = offloaded_params * 12.0
            bucket_bytes = self._estimate_offload_bucket_bytes(state_bytes)
            bucket_count = max(1, math.ceil(state_bytes / max(1.0, bucket_bytes)))
            bucket_payload_bytes = state_bytes / bucket_count
            update_compute_ms = self._estimate_optimizer_update_compute_ms(
                offloaded_params,
                training_config.dtype_bytes,
            )
            host_to_device_bw_gbps = getattr(
                self.hardware_config.gpu,
                "get_host_to_device_bandwidth",
                None,
            )
            if callable(host_to_device_bw_gbps):
                h2d_bw = host_to_device_bw_gbps(int(bucket_payload_bytes))
            else:
                h2d_bw = float(
                    getattr(
                        self.hardware_config.gpu,
                        "host_to_device_bandwidth_gbps",
                        self.HOST_TO_DEVICE_BANDWIDTH_GBPS,
                    )
                )
            device_to_host_bw_gbps = getattr(
                self.hardware_config.gpu,
                "get_device_to_host_bandwidth",
                None,
            )
            if callable(device_to_host_bw_gbps):
                d2h_bw = device_to_host_bw_gbps(int(bucket_payload_bytes))
            else:
                d2h_bw = float(
                    getattr(
                        self.hardware_config.gpu,
                        "device_to_host_bandwidth_gbps",
                        self.DEVICE_TO_HOST_BANDWIDTH_GBPS,
                    )
                )

            local_backward_ms = (
                max(0.0, float(stage_backward_time_ms[stage_id]))
                if stage_id < len(stage_backward_time_ms)
                else 0.0
            )
            raw_dp_ms = (
                max(0.0, float(dp_stage_raw_time_ms[stage_id]))
                if stage_id < len(dp_stage_raw_time_ms)
                else 0.0
            )
            exposed_dp_ms = (
                max(0.0, float(dp_stage_exposed_time_ms[stage_id]))
                if stage_id < len(dp_stage_exposed_time_ms)
                else raw_dp_ms
            )
            hidden_dp_ms = max(0.0, raw_dp_ms - exposed_dp_ms)

            pp_drain_window_ms = 0.0
            if pp > 1:
                pp_drain_window_ms = (
                    bubble_time_ms * max(0, pp - 1 - stage_id) / max(1, pp - 1)
                )

            if int(self.hardware_config.num_nodes) > 1:
                prefetch_window_ms = hidden_dp_ms + pp_drain_window_ms
                update_ready_window_ms = hidden_dp_ms + pp_drain_window_ms
            else:
                prefetch_window_ms = local_backward_ms + hidden_dp_ms + pp_drain_window_ms
                update_ready_window_ms = local_backward_ms + hidden_dp_ms + pp_drain_window_ms

            exposed_ms = self._simulate_offload_pipeline_tail(
                state_bytes=state_bytes,
                update_compute_ms=update_compute_ms,
                bucket_count=bucket_count,
                host_to_device_bw_gbps=h2d_bw,
                device_to_host_bw_gbps=d2h_bw,
                prefetch_window_ms=prefetch_window_ms,
                update_ready_window_ms=update_ready_window_ms,
            )
            offload_stage_overheads.append(exposed_ms)

        return max(offload_stage_overheads) if offload_stage_overheads else 0.0

    def _estimate_optimizer_step_time_ms(self,
                                         parallel: ParallelConfig,
                                         sharding_degree: int,
                                         offload_ratio: float,
                                         training_config: TrainingConfig,
                                         compute_result: Dict[str, float],
                                         comm_result: Dict[str, float],
                                         offload_tail_ms: float) -> float:
        """估算 optimizer-step 阶段的暴露时间。"""
        if sharding_degree <= 0:
            return 0.0

        stage_param_counts = self.comm_model._estimate_stage_parameter_counts_per_gpu(
            self.model_config, parallel
        )
        if not stage_param_counts:
            return 0.0

        optimizer_stage_times = []
        ratio = max(0.0, min(1.0, float(offload_ratio)))
        for stage_id, stage_params in enumerate(stage_param_counts):
            sharded_params = float(stage_params) / max(1, sharding_degree)
            if sharded_params <= 0:
                optimizer_stage_times.append(0.0)
                continue

            tensor_count = self._estimate_stage_optimizer_tensor_count(parallel, stage_id)
            tensor_runtime_ms = tensor_count * (
                1.6 if int(self.hardware_config.num_nodes) > 1 else 0.35
            )
            if bool(training_config.best_unbalanced_scheduler):
                tensor_runtime_ms *= 1.08
            if bool(training_config.variable_seq_lengths) or bool(training_config.enable_dynamic_shape):
                tensor_runtime_ms *= 1.10

            resident_params = sharded_params * max(0.0, 1.0 - ratio)
            resident_update_ms = self._estimate_optimizer_update_compute_ms(
                resident_params, training_config.dtype_bytes
            )

            stage_time_ms = resident_update_ms + tensor_runtime_ms
            if ratio > 0.0:
                offloaded_params = sharded_params * ratio
                state_bytes = offloaded_params * 13.0
                bucket_bytes = self._estimate_offload_bucket_bytes(state_bytes)
                bucket_count = max(1, math.ceil(state_bytes / max(1.0, bucket_bytes)))
                bucket_payload_bytes = state_bytes / bucket_count

                h2d_getter = getattr(self.hardware_config.gpu, "get_host_to_device_bandwidth", None)
                if callable(h2d_getter):
                    h2d_bw = h2d_getter(int(bucket_payload_bytes))
                else:
                    h2d_bw = float(
                        getattr(
                            self.hardware_config.gpu,
                            "host_to_device_bandwidth_gbps",
                            self.HOST_TO_DEVICE_BANDWIDTH_GBPS,
                        )
                    )
                d2h_getter = getattr(self.hardware_config.gpu, "get_device_to_host_bandwidth", None)
                if callable(d2h_getter):
                    d2h_bw = d2h_getter(int(bucket_payload_bytes))
                else:
                    d2h_bw = float(
                        getattr(
                            self.hardware_config.gpu,
                            "device_to_host_bandwidth_gbps",
                            self.DEVICE_TO_HOST_BANDWIDTH_GBPS,
                        )
                    )

                update_compute_ms = self._estimate_optimizer_update_compute_ms(
                    offloaded_params, training_config.dtype_bytes
                )
                pipeline_ms = self._simulate_offload_pipeline_duration(
                    state_bytes=state_bytes,
                    update_compute_ms=update_compute_ms,
                    bucket_count=bucket_count,
                    host_to_device_bw_gbps=h2d_bw,
                    device_to_host_bw_gbps=d2h_bw,
                )
                host_bw = max(1.0, min(float(h2d_bw), float(d2h_bw)))
                cpu_bookkeeping_ms = (
                    offloaded_params * 24.0 / (host_bw * 0.55 * 1e9) * 1000.0
                )
                bucket_sync_ms = bucket_count * (6.0 if int(self.hardware_config.num_nodes) > 1 else 1.5)
                stage_time_ms += pipeline_ms + cpu_bookkeeping_ms + bucket_sync_ms

            optimizer_stage_times.append(stage_time_ms)

        peak_stage_ms = max(optimizer_stage_times) if optimizer_stage_times else 0.0

        # ------------------------------------------------------------------
        # MoE expert offload overhead correction
        # When offloading with many experts per GPU, the optimizer step incurs
        # significant per-expert-tensor overhead from framework state management
        # (H2D scheduling, routing bookkeeping, per-expert sync). An additional
        # memory-pressure factor accounts for CUDA allocator contention when
        # model states fill most of GPU memory.
        # ------------------------------------------------------------------
        num_experts = int(getattr(self.model_config, "num_experts", 0) or 0)
        experts_per_gpu = max(1, num_experts // max(1, int(parallel.ep))) if num_experts > 1 else 0
        if experts_per_gpu > 64 and ratio > 0.0:
            # Count expert tensors in the peak stage
            peak_stage_id = (
                optimizer_stage_times.index(peak_stage_ms) if optimizer_stage_times else 0
            )
            layer_indices = self.compute_model._stage_layer_indices(parallel, peak_stage_id)
            moe_layers_in_stage = sum(
                1 for idx in layer_indices if self.compute_model._is_moe_layer(idx)
            )
            expert_tensor_count = moe_layers_in_stage * experts_per_gpu * 3

            # Memory-pressure ratio from model states (weights + grads + master_grad)
            max_stage_params = float(max(stage_param_counts))
            dtype_bytes = float(training_config.dtype_bytes)
            has_master_grad = bool(getattr(training_config, "amp_master_grad", False))
            grad_bytes = 4.0 if (has_master_grad and dtype_bytes < 4) else dtype_bytes
            model_state_gb = max_stage_params * (dtype_bytes + dtype_bytes + grad_bytes) / (1024 ** 3)
            gpu_gb = float(getattr(self.hardware_config.gpu, "memory_gb", 80.0) or 80.0)
            mem_ratio = min(1.0, model_state_gb * 1.2 / gpu_gb)

            # Per-expert-tensor overhead (~30 ms base, scaled by memory pressure)
            mem_pressure_factor = 1.0
            if mem_ratio > 0.55:
                mem_pressure_factor = 1.0 + 0.3 * (mem_ratio - 0.55)
            per_tensor_overhead_ms = 30.0 * mem_pressure_factor
            peak_stage_ms += expert_tensor_count * per_tensor_overhead_ms

        return max(peak_stage_ms, offload_tail_ms)
    
    def rank_configurations(self, configs: List[Dict],
                            top_k: int = 10,
                            micro_batch_size: int = None,
                            max_seq_len: int = None) -> List[Dict]:
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
            max_seq_len: 序列长度
        
        Returns:
            排序后的配置列表
        """
        results = []
        
        for cfg in configs:
            try:
                parallel = ParallelConfig.from_dict(cfg)
                prediction = self.predict(parallel, micro_batch_size, max_seq_len)
                
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
                logger.warning(f"Failed to predict config {cfg}: {e}")
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
        """记录排序报告到日志"""
        if not results:
            return
        
        best = results[0]
        logger.info(f"最优配置: {best['config_str']}, 时延={best['step_time_ms']:.2f}ms, "
                   f"显存={best['memory_gb']:.2f}GB, MFU={best['mfu']:.1%}, "
                   f"吞吐量={best['tokens_per_second']:,.0f} tok/s")
    
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
                
                # EP 搜索：ep <= sd(dp) 且 num_experts % ep == 0
                ep_candidates = [1]
                if num_experts > 1:
                    for ep in [2, 4, 8, 16, 32]:
                        if ep <= dp and num_experts % ep == 0:
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
                
                # PaddleFormers 的 tensorwise optimizer offload 在 dp=1 时仍可能生效，
                # 这里不再硬性过滤这类配置。
                
                # EP 候选：ep <= sd(dp) 且 num_experts % ep == 0
                ep_candidates = [1]
                if num_experts > 1:
                    for ep in [2, 4, 8]:
                        if ep <= dp and num_experts % ep == 0:
                            ep_candidates.append(ep)
                
                for ep in ep_candidates:
                    for sharding in ["stage1", "stage2"]:
                        parallel_configs.append({
                            "tp": tp, "pp": pp, "dp": dp, "ep": ep, "sharding": sharding
                        })
        
        logger.info(f"搜索空间: 并行配置={len(parallel_configs)}, seq_lens={seq_lens}, "
                   f"mbs={micro_batch_sizes}, gas={gas_values}, "
                   f"总组合={len(parallel_configs) * len(seq_lens) * len(micro_batch_sizes) * len(gas_values)}")
        
        # 遍历所有组合
        for pcfg in parallel_configs:
            parallel = ParallelConfig.from_dict(pcfg)
            
            for seq_len in seq_lens:
                for mbs in micro_batch_sizes:
                    for gas in gas_values:
                        try:
                            result = self.predict(
                                parallel,
                                micro_batch_size=mbs,
                                max_seq_len=seq_len,
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
        """记录吞吐量排序报告到日志"""
        if not results:
            logger.warning("没有找到满足显存约束的配置")
            return
        
        best = results[0]
        logger.info(f"最优配置: {best['parallel_str']}, seq_len={best['seq_len']}, "
                   f"mbs={best['micro_batch_size']}, gas={best['gradient_accumulation_steps']}, "
                   f"时延={best['step_time_s']:.2f}s, 显存={best['memory_gb']:.1f}GB, "
                   f"吞吐量={best['tokens_per_second_per_gpu']:,.0f} tok/s/GPU")
    
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
model_name_or_path: {config.get('model_path', './Qwen3-30B-A3B-Base')}
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
            logger.info(f"配置已保存到: {output_path}")
        
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
                               gpus_per_node: int = 8) -> "PDCostModel":
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
                                 gpus_per_node: int = 8) -> "PDCostModel":
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


PDCostModel = _BasePDCostModel

from typing import Any, Iterable, Sequence
from .submodels.recompute_stage_sim import (
    build_stage_plans_from_recompute_configs,
    build_uniform_stage_plans,
    plans_to_dicts,
)
from .utils.runtime_model_calibration import (
    RuntimeCalibrationFitter,
    RuntimeCalibrationStore,
    apply_runtime_calibration_to_result,
    build_runtime_context_snapshot,
    canonicalize_model_name,
    default_model_key_from_config,
    get_runtime_calibration_supported_fields,
)
from .utils.similarity_calibration import (
    SimilarityCalibrationStore,
    apply_similarity_calibration_to_result,
    default_similarity_model_key_from_config,
)

class PDCostModel(_BasePDCostModel):
    TP_COLLECTIVES_PER_LAYER = 4.0
    TP_INTRA_NODE_SAFETY = 1.08
    TP_INTER_NODE_SAFETY = 1.18

    def __init__(self,
                 model_config: ModelConfig,
                 hardware_config: HardwareConfig = None,
                 training_config: TrainingConfig = None,
                 auto_calibrate: bool = False,
                 calibrate_on_predict: bool = False,
                 use_cached_profile: bool = True,
                 node_count: int = 1,
                 enable_runtime_calibration: bool = False,
                 runtime_calibration_store_path: Optional[str] = None,
                 runtime_calibration_model_name: Optional[str] = None,
                 enable_similarity_calibration: bool = True,
                 similarity_calibration_store_path: Optional[str] = None,
                 similarity_calibration_model_name: Optional[str] = None):
        super().__init__(
            model_config=model_config,
            hardware_config=hardware_config,
            training_config=training_config,
            auto_calibrate=auto_calibrate,
            calibrate_on_predict=calibrate_on_predict,
            use_cached_profile=use_cached_profile,
            node_count=node_count,
        )
        env_enable = str(os.getenv("PDCOST_ENABLE_RUNTIME_CALIBRATION", "")).strip().lower() in {"1", "true", "yes", "on"}
        self.enable_runtime_calibration = bool(enable_runtime_calibration or env_enable)
        self.runtime_calibration_store_path = (
            runtime_calibration_store_path
            or os.getenv("PDCOST_RUNTIME_CALIBRATION_STORE")
            or str(Path(__file__).parent / "utils" / "runtime_model_calibrations.json")
        )
        self.runtime_calibration_model_name = runtime_calibration_model_name
        self._runtime_calibration_store = RuntimeCalibrationStore(self.runtime_calibration_store_path)
        self._runtime_calibration_entry = None
        self.reload_runtime_calibration()

        env_similarity = str(os.getenv("PDCOST_ENABLE_SIMILARITY_CALIBRATION", "")).strip().lower()
        if env_similarity in {"0", "false", "no", "off"}:
            self.enable_similarity_calibration = False
        elif env_similarity in {"1", "true", "yes", "on"}:
            self.enable_similarity_calibration = True
        else:
            self.enable_similarity_calibration = bool(enable_similarity_calibration)
        self.similarity_calibration_store_path = (
            similarity_calibration_store_path
            or os.getenv("PDCOST_SIMILARITY_CALIBRATION_STORE")
            or str(Path(__file__).parent / "utils" / "similarity_calibrations.json")
        )
        self.similarity_calibration_model_name = similarity_calibration_model_name
        self._similarity_calibration_store = SimilarityCalibrationStore(self.similarity_calibration_store_path)
        self._similarity_calibration_entry = None
        self.reload_similarity_calibration()

    def _runtime_calibration_model_key(self) -> str:
        return canonicalize_model_name(
            self.runtime_calibration_model_name or default_model_key_from_config(self.model_config)
        )

    def reload_runtime_calibration(self) -> Optional[Dict[str, Any]]:
        self._runtime_calibration_store = RuntimeCalibrationStore(self.runtime_calibration_store_path)
        self._runtime_calibration_entry = self._runtime_calibration_store.get(self._runtime_calibration_model_key())
        return self._runtime_calibration_entry

    def _similarity_calibration_model_key(self) -> str:
        return canonicalize_model_name(
            self.similarity_calibration_model_name or default_similarity_model_key_from_config(self.model_config)
        )

    def reload_similarity_calibration(self) -> Optional[Dict[str, Any]]:
        self._similarity_calibration_store = SimilarityCalibrationStore(self.similarity_calibration_store_path)
        self._similarity_calibration_entry = self._similarity_calibration_store.get(self._similarity_calibration_model_key())
        return self._similarity_calibration_entry

    def fit_runtime_calibration_from_logs(self,
                                          log_paths: Sequence[str],
                                          model_name: Optional[str] = None,
                                          persist: bool = True,
                                          min_ok_observations: int = 4) -> Dict[str, Any]:
        fitter = RuntimeCalibrationFitter(self, self._runtime_calibration_store)
        entry = fitter.fit_from_log_files(
            model_name=model_name or self._runtime_calibration_model_key(),
            log_paths=list(log_paths),
            persist=persist,
            min_ok_observations=min_ok_observations,
        )
        self.reload_runtime_calibration()
        return entry

    def fit_runtime_calibration_from_log_texts(self,
                                               log_texts: Sequence[str],
                                               source_names: Optional[Sequence[str]] = None,
                                               model_name: Optional[str] = None,
                                               persist: bool = True,
                                               min_ok_observations: int = 4) -> Dict[str, Any]:
        fitter = RuntimeCalibrationFitter(self, self._runtime_calibration_store)
        entry = fitter.fit_from_log_texts(
            model_name=model_name or self._runtime_calibration_model_key(),
            log_texts=list(log_texts),
            source_names=list(source_names) if source_names is not None else None,
            persist=persist,
            min_ok_observations=min_ok_observations,
        )
        self.reload_runtime_calibration()
        return entry

    def get_runtime_calibration_supported_fields(self) -> List[Dict[str, Any]]:
        return get_runtime_calibration_supported_fields()

    def _maybe_apply_runtime_calibration(self,
                                         result: PredictionResult,
                                         *,
                                         parallel: ParallelConfig,
                                         runtime_training_config: TrainingConfig,
                                         recompute_granularity: Optional[str],
                                         recompute_method: Optional[str],
                                         recompute_num_layers: Optional[int],
                                         tensorwise_offload_optimizer: Optional[bool],
                                         tensorwise_offload_ratio: Optional[float],
                                         split_param: Optional[bool],
                                         sd_release_grads: Optional[bool],
                                         micro_batch_size: Optional[int],
                                         gradient_accumulation_steps: Optional[int],
                                         max_seq_len: Optional[int]) -> PredictionResult:
        if not self.enable_runtime_calibration:
            return result
        if not self._runtime_calibration_entry:
            return result
        runtime_context = build_runtime_context_snapshot(
            parallel=parallel,
            runtime_training_config=runtime_training_config,
            recompute_granularity=recompute_granularity,
            recompute_method=recompute_method,
            recompute_num_layers=recompute_num_layers,
            tensorwise_offload_optimizer=tensorwise_offload_optimizer,
            tensorwise_offload_ratio=tensorwise_offload_ratio,
            split_param=split_param,
            sd_release_grads=sd_release_grads,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_seq_len=max_seq_len,
        )
        apply_runtime_calibration_to_result(
            result=result,
            calibration_entry=self._runtime_calibration_entry,
            parallel=parallel,
            runtime_context=runtime_context,
            gpu_memory_gb=float(getattr(self.hardware_config.gpu, "memory_gb", 0.0) or 0.0),
            total_layers=int(getattr(self.model_config, "num_hidden_layers", 1) or 1),
        )
        return result

    def _maybe_apply_similarity_calibration(self,
                                            result: PredictionResult,
                                            *,
                                            runtime_context: Optional[Dict[str, Any]]) -> PredictionResult:
        if not self.enable_similarity_calibration:
            return result
        if not self._similarity_calibration_entry:
            return result
        apply_similarity_calibration_to_result(
            result=result,
            calibration_entry=self._similarity_calibration_entry,
            runtime_context=runtime_context,
            gpu_memory_gb=float(getattr(self.hardware_config.gpu, "memory_gb", 0.0) or 0.0),
        )
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        parallel: ParallelConfig,
        micro_batch_size: int = None,
        max_seq_len: int = None,
        gradient_accumulation_steps: int = None,
        recompute_granularity: str = None,
        recompute_method: Optional[str] = None,
        recompute_num_layers: Optional[int] = None,
        recompute_modules: Optional[List[str]] = None,
        tensorwise_offload_optimizer: bool = None,
        tensorwise_offload_ratio: float = None,
        split_param: bool = True,
        sd_release_grads: bool = None,
        **kwargs,
    ) -> PredictionResult:
        apply_runtime_calibration = bool(kwargs.pop("apply_runtime_calibration", True))
        apply_similarity_calibration = bool(kwargs.pop("apply_similarity_calibration", True))
        result = super().predict(
            parallel,
            micro_batch_size=micro_batch_size,
            max_seq_len=max_seq_len,
            gradient_accumulation_steps=gradient_accumulation_steps,
            recompute_granularity=recompute_granularity,
            recompute_method=recompute_method,
            recompute_num_layers=recompute_num_layers,
            recompute_modules=recompute_modules,
            tensorwise_offload_optimizer=tensorwise_offload_optimizer,
            tensorwise_offload_ratio=tensorwise_offload_ratio,
            split_param=split_param,
            sd_release_grads=sd_release_grads,
            **kwargs,
        )

        stage_layers = self._get_stage_layer_counts(parallel, result)
        plans = build_uniform_stage_plans(
            stage_layers,
            recompute_granularity if recompute_granularity is not None else self.training_config.recompute_config,
            recompute_method if recompute_method is not None else getattr(self.training_config, "recompute_method", "uniform"),
            recompute_num_layers if recompute_num_layers is not None else getattr(self.training_config, "recompute_num_layers", 1),
        )
        result.stage_recompute_detail = plans_to_dicts(plans)

        runtime = self._resolve_runtime_inputs(
            micro_batch_size=micro_batch_size,
            max_seq_len=max_seq_len,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        runtime_training_config = self._build_runtime_training_config(
            parallel=parallel,
            micro_batch_size=runtime["micro_batch_size"],
            max_seq_len=runtime["max_seq_len"],
            gradient_accumulation_steps=runtime["gradient_accumulation_steps"],
            kwargs=dict(kwargs),
        )
        runtime_context = build_runtime_context_snapshot(
            parallel=parallel,
            runtime_training_config=runtime_training_config,
            recompute_granularity=(recompute_granularity if recompute_granularity is not None else getattr(self.training_config, "recompute_config", "none")),
            recompute_method=(recompute_method if recompute_method is not None else getattr(self.training_config, "recompute_method", None)),
            recompute_num_layers=(recompute_num_layers if recompute_num_layers is not None else getattr(self.training_config, "recompute_num_layers", None)),
            tensorwise_offload_optimizer=(tensorwise_offload_optimizer if tensorwise_offload_optimizer is not None else getattr(self.training_config, "tensorwise_offload_optimizer", False)),
            tensorwise_offload_ratio=(tensorwise_offload_ratio if tensorwise_offload_ratio is not None else getattr(self.training_config, "tensorwise_offload_ratio", 0.95)),
            split_param=split_param,
            sd_release_grads=sd_release_grads,
            micro_batch_size=runtime["micro_batch_size"],
            gradient_accumulation_steps=runtime["gradient_accumulation_steps"],
            max_seq_len=runtime["max_seq_len"],
        )
        if apply_runtime_calibration:
            self._maybe_apply_runtime_calibration(
                result,
                parallel=parallel,
                runtime_training_config=runtime_training_config,
                recompute_granularity=(recompute_granularity if recompute_granularity is not None else getattr(self.training_config, "recompute_config", "none")),
                recompute_method=(recompute_method if recompute_method is not None else getattr(self.training_config, "recompute_method", None)),
                recompute_num_layers=(recompute_num_layers if recompute_num_layers is not None else getattr(self.training_config, "recompute_num_layers", None)),
                tensorwise_offload_optimizer=(tensorwise_offload_optimizer if tensorwise_offload_optimizer is not None else getattr(self.training_config, "tensorwise_offload_optimizer", False)),
                tensorwise_offload_ratio=(tensorwise_offload_ratio if tensorwise_offload_ratio is not None else getattr(self.training_config, "tensorwise_offload_ratio", 0.95)),
                split_param=split_param,
                sd_release_grads=sd_release_grads,
                micro_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_seq_len=max_seq_len,
            )
        if apply_similarity_calibration:
            self._maybe_apply_similarity_calibration(result, runtime_context=runtime_context)
        result.tokens_per_step, result.tokens_per_second, result.tokens_per_second_per_gpu = self._calculate_throughput(
            result,
            parallel,
            runtime["micro_batch_size"],
            runtime["max_seq_len"],
            runtime["gradient_accumulation_steps"],
        )
        return result

    def predict_per_stage(
        self,
        parallel: ParallelConfig,
        per_stage_recompute_granularity: List[str] = None,
        per_stage_recompute_method: List[str] = None,
        per_stage_recompute_num_layers: List[int] = None,
        per_stage_offload: List[bool] = None,
        per_stage_offload_ratio: List[float] = None,
        **kwargs,
    ) -> PredictionResult:
        apply_runtime_calibration = bool(kwargs.pop("apply_runtime_calibration", True))
        apply_similarity_calibration = bool(kwargs.pop("apply_similarity_calibration", True))
        stage_count = max(1, int(getattr(parallel, "pp", 1) or 1))
        runtime = self._resolve_runtime_inputs(
            micro_batch_size=kwargs.get("micro_batch_size"),
            max_seq_len=kwargs.get("max_seq_len"),
            gradient_accumulation_steps=kwargs.get("gradient_accumulation_steps"),
        )
        runtime_training_config = self._build_runtime_training_config(
            parallel=parallel,
            micro_batch_size=runtime["micro_batch_size"],
            max_seq_len=runtime["max_seq_len"],
            gradient_accumulation_steps=runtime["gradient_accumulation_steps"],
            kwargs=kwargs,
        )

        # Build per-stage recompute configs.
        per_stage_rc = self._build_per_stage_recompute_objects(
            stage_count=stage_count,
            per_stage_recompute_granularity=per_stage_recompute_granularity,
            per_stage_recompute_method=per_stage_recompute_method,
            per_stage_recompute_num_layers=per_stage_recompute_num_layers,
            fallback_granularity=kwargs.get("recompute_granularity"),
            fallback_method=kwargs.get("recompute_method"),
            fallback_num_layers=kwargs.get("recompute_num_layers"),
        )
        plans = build_stage_plans_from_recompute_configs(
            self._get_stage_layer_counts(parallel, None),
            per_stage_rc,
        )

        any_offload, max_offload_ratio = self._normalize_stage_offload(
            stage_count=stage_count,
            per_stage_offload=per_stage_offload,
            per_stage_offload_ratio=per_stage_offload_ratio,
            fallback_enable=kwargs.get("tensorwise_offload_optimizer"),
            fallback_ratio=kwargs.get("tensorwise_offload_ratio"),
        )

        # Use one representative global config to obtain a fully initialized
        # PredictionResult and preserve base runtime/helper behavior.
        global_gran, global_method, global_num_layers = self._representative_global_recompute(per_stage_rc)
        result = super().predict(
            parallel,
            micro_batch_size=runtime["micro_batch_size"],
            max_seq_len=runtime["max_seq_len"],
            gradient_accumulation_steps=runtime["gradient_accumulation_steps"],
            recompute_granularity=global_gran,
            recompute_method=global_method,
            recompute_num_layers=global_num_layers,
            recompute_modules=kwargs.get("recompute_modules"),
            tensorwise_offload_optimizer=any_offload,
            tensorwise_offload_ratio=max_offload_ratio if any_offload else None,
            split_param=kwargs.get("split_param", True),
            sd_release_grads=kwargs.get("sd_release_grads"),
            overlap_p2p_comm=kwargs.get("overlap_p2p_comm"),
            use_batch_p2p_comm=kwargs.get("use_batch_p2p_comm"),
            p2p_cache_shape=kwargs.get("p2p_cache_shape"),
            stage1_overlap=kwargs.get("stage1_overlap"),
            enable_sharding_comm_overlap=kwargs.get("enable_sharding_comm_overlap"),
            variable_seq_lengths=kwargs.get("variable_seq_lengths"),
            enable_dynamic_shape=kwargs.get("enable_dynamic_shape"),
            clear_every_step_cache=kwargs.get("clear_every_step_cache"),
            best_unbalanced_scheduler=kwargs.get("best_unbalanced_scheduler"),
            hybrid_parallel_topo_order=kwargs.get("hybrid_parallel_topo_order"),
            num_empty_layers_add_in_head=kwargs.get("num_empty_layers_add_in_head"),
            num_empty_layers_add_in_tail=kwargs.get("num_empty_layers_add_in_tail"),
            attn_implementation=kwargs.get("attn_implementation"),
            apply_rope_fusion=kwargs.get("apply_rope_fusion"),
            moe_token_dispatcher_type=kwargs.get("moe_token_dispatcher_type"),
            moe_grouped_gemm=kwargs.get("moe_grouped_gemm"),
            moe_router_fusion=kwargs.get("moe_router_fusion"),
            moe_expert_fusion=kwargs.get("moe_expert_fusion"),
            moe_shared_expert_overlap=kwargs.get("moe_shared_expert_overlap"),
            moe_ep_barrier=kwargs.get("moe_ep_barrier"),
        )

        # -------- exact memory with per-stage recompute --------
        use_sd_release_grads = bool(
            kwargs.get("sd_release_grads")
            if kwargs.get("sd_release_grads") is not None
            else runtime_training_config.sd_release_grads
        )
        sharding_degree = self._resolve_sharding_degree(parallel)
        sharding_config = ShardingConfig(
            stage=parallel.sharding_stage,
            degree=sharding_degree,
            split_param=kwargs.get("split_param", True),
            release_grads=use_sd_release_grads,
            tensorwise_offload=bool(any_offload),
            tensorwise_offload_ratio=float(max_offload_ratio),
        )
        memory_model = MemoryModel(self.model_config, runtime_training_config, self.hardware_config)
        mem_breakdown = memory_model.estimate_memory_per_stage(
            parallel,
            sharding_config,
            per_stage_rc,
            runtime["max_seq_len"],
            runtime["micro_batch_size"],
        )
        observed_mem_breakdown = mem_breakdown
        if int(getattr(parallel, "pp", 1) or 1) > 1:
            try:
                observed_mem_breakdown = memory_model._estimate_memory_for_stage(
                    parallel,
                    sharding_config,
                    per_stage_rc[0],
                    runtime["max_seq_len"],
                    runtime["micro_batch_size"],
                    0,
                )
            except Exception:
                observed_mem_breakdown = mem_breakdown
        result.memory_breakdown = observed_mem_breakdown
        result.allocated_memory_gb = getattr(observed_mem_breakdown, "allocated_memory_gb", result.allocated_memory_gb)
        result.reserved_memory_gb = getattr(observed_mem_breakdown, "reserved_memory_gb", result.reserved_memory_gb)
        result.memory_gb = getattr(mem_breakdown, "total_memory_gb", result.memory_gb)
        result.fits_memory = result.memory_gb <= self.hardware_config.gpu.memory_gb
        result.parallel_config["memory_peak_stage_id"] = int(getattr(mem_breakdown, "peak_stage_id", 0))
        result.parallel_config["display_memory_stage_id"] = 0 if int(getattr(parallel, "pp", 1) or 1) > 1 else int(getattr(mem_breakdown, "peak_stage_id", 0))

        # -------- exact compute with per-stage recompute --------
        self.compute_model.training = runtime_training_config
        compute_result = self.compute_model.estimate_step_compute_time(
            runtime["micro_batch_size"],
            runtime["max_seq_len"],
            parallel,
            runtime["gradient_accumulation_steps"],
            recompute_granularity=global_gran,
            recompute_method=global_method,
            recompute_num_layers=global_num_layers,
            recompute_modules=kwargs.get("recompute_modules"),
            per_stage_recompute=per_stage_rc,
            stage_layer_counts=self._get_stage_layer_counts(parallel, result),
        )
        result.recompute_overhead = compute_result.get("recompute_overhead", result.recompute_overhead)
        result.forward_time_ms = compute_result.get("forward_time_ms", result.forward_time_ms)
        result.backward_time_ms = compute_result.get("backward_time_ms", result.backward_time_ms)
        result.bubble_time_ms = compute_result.get("bubble_time_ms", result.bubble_time_ms)
        result.compute_time_ms = compute_result.get("compute_time_ms", result.compute_time_ms)
        result.bubble_ratio = compute_result.get("bubble_ratio", result.bubble_ratio)
        result.framework_overhead_ms = compute_result.get("framework_overhead_ms", result.framework_overhead_ms)
        result.recompute_time_ms = compute_result.get("recompute_time_ms", result.recompute_time_ms)
        result.runtime_overhead_ms = compute_result.get("runtime_overhead_ms", result.runtime_overhead_ms)
        result.stage_forward_micro_ms = list(compute_result.get("stage_forward_micro_ms", []) or [])
        result.stage_backward_micro_ms = list(compute_result.get("stage_backward_micro_ms", []) or [])
        result.stage_recompute_detail = list(compute_result.get("stage_recompute_detail", []) or plans_to_dicts(plans))
        result.stage_layer_counts = list(compute_result.get("stage_layer_counts", []) or self._get_stage_layer_counts(parallel, result))

        # -------- communication with corrected stage timing --------
        comm_training_config = TrainingConfig.from_dict(runtime_training_config.to_dict())
        comm_result = self.comm_model.estimate_step_comm_time(
            self.model_config,
            comm_training_config,
            parallel,
            runtime["gradient_accumulation_steps"],
            stage_forward_time_ms=compute_result.get("stage_forward_micro_ms"),
            stage_backward_time_ms=compute_result.get("stage_backward_micro_ms"),
        )
        result.tp_comm_time_ms = comm_result["tp_comm_time_ms"]
        result.dp_comm_time_ms = comm_result["dp_comm_time_ms"]
        result.dp_exposed_comm_time_ms = comm_result.get("dp_exposed_comm_time_ms", result.dp_comm_time_ms)
        result.ep_comm_time_ms = comm_result["ep_comm_time_ms"]
        result.pp_comm_time_ms = comm_result["pp_comm_time_ms"]
        result.sp_comm_time_ms = comm_result.get("sp_comm_time_ms", 0.0)
        result.total_comm_time_ms = comm_result["total_comm_time_ms"]
        result.stage_tp_comm_micro_ms = list(comm_result.get("stage_tp_comm_micro_ms", []) or [])
        result.stage_ep_comm_micro_ms = list(comm_result.get("stage_ep_comm_micro_ms", []) or [])
        result.stage_sp_comm_micro_ms = list(comm_result.get("stage_sp_comm_micro_ms", []) or [])
        result.stage_dp_exposed_step_ms = list(comm_result.get("dp_stage_exposed_comm_time_ms", []) or [])

        # -------- overheads --------
        offload_overhead_ms = 0.0
        optimizer_step_time_ms = 0.0
        if any_offload:
            offload_overhead_ms = self._estimate_offload_overhead_ms(
                parallel=parallel,
                sharding_degree=sharding_degree,
                offload_ratio=max_offload_ratio,
                training_config=runtime_training_config,
                compute_result=compute_result,
                comm_result=comm_result,
            )
        if sharding_degree > 0:
            optimizer_step_time_ms = self._estimate_optimizer_step_time_ms(
                parallel=parallel,
                sharding_degree=sharding_degree,
                offload_ratio=max_offload_ratio if any_offload else 0.0,
                training_config=runtime_training_config,
                compute_result=compute_result,
                comm_result=comm_result,
                offload_tail_ms=offload_overhead_ms,
            )
        result.offload_overhead_ms = offload_overhead_ms
        result.optimizer_step_time_ms = optimizer_step_time_ms

        # -------- TP correction + final composition --------
        self._apply_tp_correction(
            result=result,
            parallel=parallel,
            stage_layers=result.stage_layer_counts,
            micro_batch_size=runtime["micro_batch_size"],
            max_seq_len=runtime["max_seq_len"],
        )
        self._finalize_result(result=result, parallel=parallel)
        result.compute_efficiency = self._calculate_compute_efficiency(result, parallel)
        result.mfu = self._calculate_mfu(
            result,
            parallel,
            runtime["micro_batch_size"],
            runtime["max_seq_len"],
            runtime["gradient_accumulation_steps"],
        )
        self._calculate_throughput(
            result,
            parallel,
            runtime["micro_batch_size"],
            runtime["max_seq_len"],
            runtime["gradient_accumulation_steps"],
        )
        if apply_runtime_calibration:
            # per-stage path uses representative recompute/offload state for residual correction.
            self._maybe_apply_runtime_calibration(
                result,
                parallel=parallel,
                runtime_training_config=runtime_training_config,
                recompute_granularity=global_gran,
                recompute_method=global_method,
                recompute_num_layers=global_num_layers,
                tensorwise_offload_optimizer=any_offload,
                tensorwise_offload_ratio=max_offload_ratio,
                split_param=kwargs.get("split_param", True),
                sd_release_grads=kwargs.get("sd_release_grads"),
            )
        if apply_similarity_calibration:
            runtime_context = build_runtime_context_snapshot(
                parallel=parallel,
                runtime_training_config=runtime_training_config,
                recompute_granularity=global_gran,
                recompute_method=global_method,
                recompute_num_layers=global_num_layers,
                tensorwise_offload_optimizer=any_offload,
                tensorwise_offload_ratio=max_offload_ratio,
                split_param=kwargs.get("split_param", True),
                sd_release_grads=kwargs.get("sd_release_grads"),
                micro_batch_size=runtime["micro_batch_size"],
                gradient_accumulation_steps=runtime["gradient_accumulation_steps"],
                max_seq_len=runtime["max_seq_len"],
            )
            self._maybe_apply_similarity_calibration(result, runtime_context=runtime_context)
        self._calculate_throughput(
            result,
            parallel,
            runtime["micro_batch_size"],
            runtime["max_seq_len"],
            runtime["gradient_accumulation_steps"],
        )
        return result

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _resolve_runtime_inputs(
        self,
        micro_batch_size: Optional[int],
        max_seq_len: Optional[int],
        gradient_accumulation_steps: Optional[int],
    ) -> Dict[str, int]:
        return {
            "micro_batch_size": int(micro_batch_size or self.training_config.micro_batch_size),
            "max_seq_len": int(max_seq_len or self.training_config.sequence_length),
            "gradient_accumulation_steps": int(
                gradient_accumulation_steps or self.training_config.gradient_accumulation_steps
            ),
        }

    def _build_runtime_training_config(
        self,
        parallel: ParallelConfig,
        micro_batch_size: int,
        max_seq_len: int,
        gradient_accumulation_steps: int,
        kwargs: Dict[str, Any],
    ) -> TrainingConfig:
        runtime_training_config = TrainingConfig.from_dict(self.training_config.to_dict())
        runtime_training_config.micro_batch_size = micro_batch_size
        runtime_training_config.sequence_length = max_seq_len
        runtime_training_config.gradient_accumulation_steps = gradient_accumulation_steps

        scalar_fields = [
            "p2p_cache_shape",
            "stage1_overlap",
            "enable_sharding_comm_overlap",
            "variable_seq_lengths",
            "enable_dynamic_shape",
            "clear_every_step_cache",
            "best_unbalanced_scheduler",
            "hybrid_parallel_topo_order",
            "num_empty_layers_add_in_head",
            "num_empty_layers_add_in_tail",
            "attn_implementation",
            "apply_rope_fusion",
            "moe_token_dispatcher_type",
            "moe_grouped_gemm",
            "moe_router_fusion",
            "moe_expert_fusion",
            "moe_shared_expert_overlap",
            "moe_ep_barrier",
            "overlap_p2p_comm",
            "use_batch_p2p_comm",
        ]
        for name in scalar_fields:
            if name in kwargs and kwargs[name] is not None:
                setattr(runtime_training_config, name, kwargs[name])

        if (
            kwargs.get("enable_dynamic_shape") is None
            and getattr(runtime_training_config, "variable_seq_lengths", False)
        ):
            runtime_training_config.enable_dynamic_shape = True

        if parallel.pp > 1 and getattr(runtime_training_config, "overlap_p2p_comm", False):
            runtime_training_config.use_batch_p2p_comm = False

        if (
            kwargs.get("p2p_cache_shape") is None
            and parallel.pp > 1
            and (
                getattr(runtime_training_config, "use_batch_p2p_comm", False)
                or getattr(runtime_training_config, "variable_seq_lengths", False)
                or getattr(runtime_training_config, "enable_dynamic_shape", False)
            )
        ):
            runtime_training_config.p2p_cache_shape = True

        if (
            kwargs.get("clear_every_step_cache") is None
            and parallel.pp > 1
            and (
                getattr(runtime_training_config, "overlap_p2p_comm", False)
                or getattr(runtime_training_config, "use_batch_p2p_comm", False)
                or getattr(runtime_training_config, "variable_seq_lengths", False)
                or getattr(runtime_training_config, "enable_dynamic_shape", False)
            )
        ):
            runtime_training_config.clear_every_step_cache = False
        return runtime_training_config

    def _build_per_stage_recompute_objects(
        self,
        stage_count: int,
        per_stage_recompute_granularity: Optional[Sequence[str]],
        per_stage_recompute_method: Optional[Sequence[str]],
        per_stage_recompute_num_layers: Optional[Sequence[int]],
        fallback_granularity: Optional[str],
        fallback_method: Optional[str],
        fallback_num_layers: Optional[int],
    ) -> List[RecomputeConfig]:
        default_granularity = fallback_granularity or self.training_config.recompute_config
        default_method = fallback_method or getattr(self.training_config, "recompute_method", "uniform")
        default_num_layers = int(
            fallback_num_layers if fallback_num_layers is not None else getattr(self.training_config, "recompute_num_layers", 1)
        )
        gran_map = {
            "none": RecomputeGranularity.NONE,
            "selective": RecomputeGranularity.SELECTIVE,
            "full": RecomputeGranularity.FULL,
        }

        def _pick(seq, idx, fallback):
            if seq is None:
                return fallback
            if idx < len(seq) and seq[idx] is not None:
                return seq[idx]
            return fallback

        objs: List[RecomputeConfig] = []
        modules = tuple(getattr(self.training_config, "recompute_modules", tuple()) or tuple())
        for sid in range(stage_count):
            gran = _pick(per_stage_recompute_granularity, sid, default_granularity)
            method = _pick(per_stage_recompute_method, sid, default_method)
            num_layers = int(_pick(per_stage_recompute_num_layers, sid, default_num_layers) or 1)
            gran_str = str(getattr(gran, "value", gran)).lower()
            objs.append(
                RecomputeConfig(
                    granularity=gran_map.get(gran_str, RecomputeGranularity.FULL),
                    method=str(method or "uniform").lower(),
                    num_layers=max(1, num_layers),
                    modules=modules,
                )
            )
        return objs

    def _representative_global_recompute(self, per_stage_rc: Sequence[RecomputeConfig]) -> Tuple[str, str, int]:
        aggressiveness = {"none": 0, "selective": 1, "full": 2}
        if not per_stage_rc:
            return ("none", "uniform", 1)
        idx = max(
            range(len(per_stage_rc)),
            key=lambda i: (
                aggressiveness.get(str(getattr(per_stage_rc[i].granularity, "value", per_stage_rc[i].granularity)).lower(), 0),
                int(getattr(per_stage_rc[i], "num_layers", 1) or 1),
            ),
        )
        rc = per_stage_rc[idx]
        return (
            str(getattr(rc.granularity, "value", rc.granularity)).lower(),
            str(getattr(rc, "method", "uniform") or "uniform").lower(),
            int(getattr(rc, "num_layers", 1) or 1),
        )

    def _normalize_stage_offload(
        self,
        stage_count: int,
        per_stage_offload: Optional[Sequence[bool]],
        per_stage_offload_ratio: Optional[Sequence[float]],
        fallback_enable: Optional[bool],
        fallback_ratio: Optional[float],
    ) -> Tuple[bool, float]:
        if per_stage_offload and len(per_stage_offload) == stage_count:
            any_offload = any(bool(v) for v in per_stage_offload)
            ratios = []
            if per_stage_offload_ratio and len(per_stage_offload_ratio) == stage_count:
                for enabled, ratio in zip(per_stage_offload, per_stage_offload_ratio):
                    if enabled:
                        ratios.append(float(ratio))
            if ratios:
                return any_offload, max(ratios)
            return any_offload, float(fallback_ratio if fallback_ratio is not None else 0.95)
        any_offload = bool(
            fallback_enable
            if fallback_enable is not None
            else getattr(self.training_config, "tensorwise_offload_optimizer", False)
        )
        ratio = float(
            fallback_ratio
            if fallback_ratio is not None
            else getattr(self.training_config, "tensorwise_offload_ratio", 0.95)
        )
        return any_offload, ratio

    def _get_stage_layer_counts(
        self,
        parallel: ParallelConfig,
        result: Optional[PredictionResult],
    ) -> List[int]:
        stage_count = max(1, int(getattr(parallel, "pp", 1) or 1))
        for source in (
            getattr(parallel, "stage_layer_counts", None),
            getattr(result, "stage_layer_counts", None) if result is not None else None,
        ):
            if source:
                values = [int(v) for v in source]
                if len(values) == stage_count:
                    return values
        try:
            return [
                len(self.compute_model._stage_layer_indices(parallel, sid))
                for sid in range(stage_count)
            ]
        except Exception:
            total_layers = int(getattr(self.model_config, "num_hidden_layers", stage_count) or stage_count)
            base, rem = divmod(total_layers, stage_count)
            return [base + (1 if i < rem else 0) for i in range(stage_count)]

    def _apply_tp_correction(
        self,
        result: PredictionResult,
        parallel: ParallelConfig,
        stage_layers: Sequence[int],
        micro_batch_size: int,
        max_seq_len: int,
    ) -> None:
        corrected_tp_stage = self._estimate_tp_stage_comm_micro_ms(
            parallel=parallel,
            stage_layers=stage_layers,
            micro_batch_size=micro_batch_size,
            max_seq_len=max_seq_len,
        )
        base_stage_tp = self._pad_stage_values(getattr(result, "stage_tp_comm_micro_ms", []), len(stage_layers))
        base_anchor = max(max(base_stage_tp), 1e-6) if base_stage_tp else 0.0
        corrected_anchor = max(corrected_tp_stage) if corrected_tp_stage else 0.0
        result.stage_tp_comm_micro_ms = list(corrected_tp_stage)
        if base_anchor > 0 and corrected_anchor > 0:
            scale = corrected_anchor / base_anchor
            result.tp_comm_time_ms = float(result.tp_comm_time_ms) * scale
        elif corrected_anchor == 0.0:
            result.tp_comm_time_ms = 0.0
        result.tp_correction_detail = {
            "tp_degree": int(getattr(parallel, "tp", 1) or 1),
            "baseline_stage_tp_comm_micro_ms": [float(v) for v in base_stage_tp],
            "corrected_stage_tp_comm_micro_ms": [float(v) for v in corrected_tp_stage],
            "corrected_tp_comm_time_ms": float(result.tp_comm_time_ms),
        }

    def _estimate_tp_stage_comm_micro_ms(
        self,
        parallel: ParallelConfig,
        stage_layers: Sequence[int],
        micro_batch_size: int,
        max_seq_len: int,
    ) -> List[float]:
        tp = max(1, int(getattr(parallel, "tp", 1) or 1))
        if tp <= 1:
            return [0.0 for _ in stage_layers]
        bytes_per_elem = self._dtype_bytes()
        hidden = int(getattr(self.model_config, "hidden_size", 1) or 1)
        activation_bytes = int(max(1, micro_batch_size) * max(1, max_seq_len) * max(1, hidden) * bytes_per_elem)
        bw_gbps, latency_us, safety = self._tp_group_network_model(tp)
        ring_factor = 2.0 * (tp - 1) / tp
        per_collective_ms = (
            ring_factor * activation_bytes * 8.0 / max(bw_gbps, 1e-6) / 1e9 * 1000.0
            + max(tp - 1, 1) * latency_us / 1000.0
        )
        per_collective_ms *= safety
        per_layer_ms = self.TP_COLLECTIVES_PER_LAYER * per_collective_ms
        return [float(max(0, layers) * per_layer_ms) for layers in stage_layers]

    def _tp_group_network_model(self, tp: int) -> Tuple[float, float, float]:
        network = getattr(self.hardware_config, "network", None)
        intra_bw = float(getattr(network, "intra_node_bandwidth_gbps", 900.0) or 900.0)
        inter_bw = float(getattr(network, "inter_node_bandwidth_gbps", 200.0) or 200.0)
        intra_lat = float(getattr(network, "intra_node_latency_us", 1.0) or 1.0)
        inter_lat = float(getattr(network, "inter_node_latency_us", 5.0) or 5.0)
        gpn = int(getattr(self.hardware_config, "gpus_per_node", tp) or tp)
        if tp <= gpn:
            return intra_bw, intra_lat, self.TP_INTRA_NODE_SAFETY
        intra_edges = max(gpn - 1, 0)
        total_edges = max(tp - 1, 1)
        intra_frac = min(1.0, intra_edges / total_edges)
        inter_frac = 1.0 - intra_frac
        effective_bw = 1.0 / (
            intra_frac / max(intra_bw, 1e-6) + inter_frac / max(inter_bw, 1e-6)
        )
        effective_lat = intra_frac * intra_lat + inter_frac * inter_lat
        return effective_bw, effective_lat, self.TP_INTER_NODE_SAFETY

    def _dtype_bytes(self) -> int:
        dtype = str(getattr(self.training_config, "dtype", "bfloat16")).lower()
        if "fp8" in dtype:
            return 1
        if any(x in dtype for x in ("bf16", "bfloat16", "fp16", "float16", "half")):
            return 2
        return 4

    def _pad_stage_values(self, values: Iterable[float], target_len: int) -> List[float]:
        vals = [float(v) for v in (values or [])]
        if len(vals) >= target_len:
            return vals[:target_len]
        if not vals:
            vals = [0.0]
        vals.extend([vals[-1]] * (target_len - len(vals)))
        return vals

    def _finalize_result(self, result: PredictionResult, parallel: ParallelConfig) -> None:
        result.total_comm_time_ms = (
            float(result.tp_comm_time_ms)
            + float(result.dp_comm_time_ms)
            + float(result.ep_comm_time_ms)
            + float(result.pp_comm_time_ms)
            + float(getattr(result, "sp_comm_time_ms", 0.0))
        )
        result.effective_comm_time_ms = (
            float(result.tp_comm_time_ms)
            + float(result.ep_comm_time_ms)
            + float(result.pp_comm_time_ms)
            + float(result.dp_exposed_comm_time_ms)
            + float(getattr(result, "sp_comm_time_ms", 0.0))
        )
        result.step_time_ms = max(
            0.0,
            float(result.compute_time_ms)
            + float(result.effective_comm_time_ms)
            + float(result.optimizer_step_time_ms),
        )

        stage_count = max(1, int(getattr(parallel, "pp", 1) or 1))
        result.stage_cycle_micro_ms = []
        for stage_id in range(stage_count):
            forward_micro = float(result.stage_forward_micro_ms[stage_id]) if stage_id < len(result.stage_forward_micro_ms) else 0.0
            backward_micro = float(result.stage_backward_micro_ms[stage_id]) if stage_id < len(result.stage_backward_micro_ms) else 0.0
            tp_comm_micro = float(result.stage_tp_comm_micro_ms[stage_id]) if stage_id < len(result.stage_tp_comm_micro_ms) else 0.0
            ep_comm_micro = float(result.stage_ep_comm_micro_ms[stage_id]) if stage_id < len(result.stage_ep_comm_micro_ms) else 0.0
            sp_comm_micro = float(result.stage_sp_comm_micro_ms[stage_id]) if stage_id < len(result.stage_sp_comm_micro_ms) else 0.0
            result.stage_cycle_micro_ms.append(
                forward_micro + backward_micro + tp_comm_micro + ep_comm_micro + sp_comm_micro
            )
        if result.stage_cycle_micro_ms:
            result.slowest_stage_id = max(
                range(len(result.stage_cycle_micro_ms)),
                key=result.stage_cycle_micro_ms.__getitem__,
            )
            result.slowest_stage_time_ms = float(result.stage_cycle_micro_ms[result.slowest_stage_id])


__all__ = ["PDCostModel", "PredictionResult", "predict"]


# ============================================================
# 简化入口函数
# ============================================================

def predict(
    model_config_path: str,
    parallel_config_path: str,
    num_nodes: int = 1,
    gpus_per_node: int = 8,
) -> PredictionResult:
    """
    分布式训练代价模型预测（简化接口）

    只需提供三个输入即可获得预测结果，无需手动构造各类配置对象。

    Args:
        model_config_path: 模型配置文件路径，支持:
            - HuggingFace/PaddleFormers 的 config.json 文件路径
            - 包含 config.json 的模型目录路径
            - yaml/json 格式的模型架构配置文件
            - 内置模型名称 (如 "qwen3-30b-a3b", "deepseek-v3")

        parallel_config_path: 并行方案配置文件路径，支持:
            - PaddleFormers 训练 yaml 文件（从中提取并行和训练配置）
            - json/yaml 格式的并行配置文件

        num_nodes: 节点数量 (默认 1)

        gpus_per_node: 每节点 GPU 数量 (默认 8)

    Returns:
        PredictionResult: 预测结果，包含:
            - step_time_ms: 总 step 时间 (ms)
            - memory_gb: 显存占用 (GB)
            - fits_memory: 是否满足显存约束
            - mfu: Model FLOPs Utilization
            - tokens_per_second: 总吞吐量
            - tokens_per_second_per_gpu: 每卡吞吐量
            - 以及详细的时延/显存/通信分解

    示例:
        result = predict(
            model_config_path="qwen3-30b-a3b",
            parallel_config_path="./parallel.yaml",
            num_nodes=2,
            gpus_per_node=8,
        )
        print(result)
    """
    from .config import ModelConfig, ParallelConfig, TrainingConfig, HardwareConfig, GPUSpec
    from .utils.io import load_dict_from_file

    # 1. 加载模型配置（自动识别名称/目录/文件）
    model_config = ModelConfig.from_any(model_config_path)

    # 2. 加载并行配置（同时保存原始字典用于提取训练参数）
    parallel_data = load_dict_from_file(parallel_config_path)
    parallel_config = ParallelConfig.from_dict(parallel_data)

    # 3. 提取训练配置（从同一份 yaml 中获取 recompute/offload 等）
    training_config = TrainingConfig.from_dict(parallel_data)

    # 4. 构建硬件配置，交由 PDCostModel 内部自动加载校准数据
    hardware_config = HardwareConfig(
        gpu=GPUSpec(memory_gb=80.0),
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
    )

    # 5. 创建 CostModel 并预测
    costmodel = PDCostModel(
        model_config=model_config,
        hardware_config=hardware_config,
        training_config=training_config,
        use_cached_profile=True,
        auto_calibrate=False,
    )

    result = costmodel.predict(
        parallel_config,
        micro_batch_size=training_config.micro_batch_size,
        max_seq_len=training_config.sequence_length,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        recompute_granularity=training_config.recompute_granularity,
        recompute_method=training_config.recompute_method,
        recompute_num_layers=training_config.recompute_num_layers,
        tensorwise_offload_optimizer=training_config.tensorwise_offload_optimizer,
        tensorwise_offload_ratio=training_config.tensorwise_offload_ratio,
    )

    return result

