#!/usr/bin/env python3
"""
显存模型模块 - 精确预测 PaddleFormers 训练的显存占用

显存组成:
1. 参数 (Parameters) - 考虑 TP/PP/EP 切分
2. 梯度 (Gradients) - 考虑 ZeRO 分片
3. 优化器状态 (Optimizer States) - AdamW: 2 × fp32
4. 激活值 (Activations) - 考虑 Recompute/Checkpoint
5. 通信缓冲区 (Communication Buffers)
6. 临时缓冲区 (Temporary Buffers)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

from ..config import (
    HardwareConfig,
    ModelConfig,
    ParallelConfig,
    TrainingConfig,
    ShardingStage,
    RecomputeGranularity,
    TRANSFORMER_DENSE_LAYER_KIND,
    TRANSFORMER_MOE_LAYER_KIND,
    INPUT_EMBEDDING_LAYER_KIND,
    OUTPUT_HEAD_LAYER_KIND,
)
from ..stage_layout import (
    resolve_chunk_ranges,
    resolve_stage_chunk_ranges,
    resolve_stage_layer_indices,
)

GIB = 1024 ** 3

# PaddleFormers tensorwise offload 会在 GPU 上保留一个运行时 live window。
# 这部分与具体模型无关，而是优化器更新流水本身的框架行为。
TENSORWISE_OFFLOAD_LIVE_WINDOW_FRACTION = 0.625

# Transformer block 的 selective checkpoint 工作集：
# - 18 份 hidden-state/residual 边界张量
# - 2 份 KV 形状的 attention/rope 辅助状态
#
# 注意：MLP 输入张量单独计入 dense/moe_mlp_input。
# 这里不再把它重复算进 shared checkpoint 基线。
CHECKPOINT_HIDDEN_TENSOR_EQUIV = 18
CHECKPOINT_KV_TENSOR_EQUIV = 2

# full recompute 在框架里是整层 `recompute(self._forward_impl, ...)`，
# 只需要保留层输入和一小组 attention/rope 辅助状态。
# 与 selective 相比，它不再持有层内 MLP/MoE 边界张量。
FULL_RECOMPUTE_HIDDEN_TENSOR_EQUIV = 8
FULL_RECOMPUTE_KV_TENSOR_EQUIV = 3

# loss 阶段会额外 materialize FP32 logits，并伴随一部分 softmax/grad workspace。
CROSS_ENTROPY_WORKSPACE_FACTOR = 1.8

# allocator 会保留一部分已经释放的激活/临时缓冲。
ALLOCATOR_REUSE_FACTOR = 0.7

# full recompute 释放的激活更多且更碎片化，allocator 实际保留比例更低。
FULL_RECOMPUTE_REUSE_FACTOR = 0.36

# 通信运行时 buffer / retained pool 的经验系数。
DEFAULT_COMM_BUCKET_BYTES = 40 * 1024 * 1024
MAX_COMM_BUCKET_BYTES = 256 * 1024 * 1024
EP_PACK_UNPACK_FACTOR = 1.5
EP_SCRATCH_FACTOR_INTRA = 0.75
EP_SCRATCH_FACTOR_INTER = 1.25
COMM_RETAIN_PER_MOE_LAYER = 0.18
COMM_DYNAMIC_SHAPE_REUSE_BONUS = 0.9
COMM_NO_CLEAR_CACHE_REUSE_BONUS = 1.2
COMM_BATCH_P2P_REUSE_BONUS = 0.75
COMM_P2P_OVERLAP_REUSE_BONUS = 0.85
COMM_P2P_CACHE_SHAPE_BONUS = 0.85
COMM_SHARD_OVERLAP_REUSE_BONUS = 0.9
COMM_BEST_SCHEDULER_BONUS = 0.35
INTER_NODE_RUNTIME_BONUS = 0.5
LOSS_PHASE_EP_LIVE_FRACTION = 0.25
LOSS_PHASE_SHARDING_LIVE_FRACTION = 0.5
OPTIMIZER_PHASE_SHARDING_LIVE_FRACTION = 0.35
OVERLAP_RUNTIME_REUSE_FACTOR = 0.55
STEP_CACHE_BASE_BONUS = 0.35
VARIABLE_SEQ_CACHE_BONUS = 0.25
DYNAMIC_SHAPE_CACHE_BONUS = 0.20
P2P_SHAPE_CACHE_BONUS = 0.30
PP_RETAIN_FACTOR_CAP = 3.0
EP_RETAIN_FACTOR_CAP = 3.5
SHARDING_RETAIN_FACTOR_CAP = 2.75


@dataclass
class ShardingConfig:
    """
    Sharding (ZeRO) 配置
    
    PaddleFormers 特有优化:
    - split_param (ShardingV2): 即使 Stage1 也分片参数和梯度
    - sd_release_grads: 每次迭代后释放梯度，降低峰值显存
    - tensorwise_offload: 优化器状态按 tensor 粒度动态 offload
    """
    stage: ShardingStage = ShardingStage.STAGE1
    degree: int = 1  # Sharding 并行度
    
    # PaddleFormers ShardingV2 特性
    split_param: bool = True  # 启用参数分片（DygraphShardingOptimizerV2）
    
    # 梯度释放优化
    release_grads: bool = False  # sd_release_grads: 迭代后释放梯度
    
    # Offload 配置
    cpu_offload: bool = False
    tensorwise_offload: bool = False
    tensorwise_offload_ratio: float = 0.95
    
    def get_param_sharding_factor(self) -> float:
        """
        参数分片因子
        
        PaddleFormers ShardingV2 (split_param=True):
        - Stage1 + split_param: 参数也会被分片
        - Stage3: 参数分片
        """
        if self.stage == ShardingStage.STAGE3:
            return self.degree
        # ShardingV2: Stage1 也分片参数
        if self.split_param and self.stage == ShardingStage.STAGE1:
            return self.degree
        return 1
    
    def get_grad_sharding_factor(self) -> float:
        """
        梯度分片因子
        
        PaddleFormers ShardingV2:
        - Stage1 + split_param: 梯度也会被分片
        - Stage2/3: 梯度分片
        """
        if self.stage in [ShardingStage.STAGE2, ShardingStage.STAGE3]:
            return self.degree
        # ShardingV2: Stage1 也分片梯度
        if self.split_param and self.stage == ShardingStage.STAGE1:
            return self.degree
        return 1
    
    def get_optimizer_sharding_factor(self) -> float:
        """优化器状态分片因子"""
        if self.stage in [ShardingStage.STAGE1, ShardingStage.STAGE2, ShardingStage.STAGE3]:
            return self.degree
        return 1
    
    def get_optimizer_memory_factor(self) -> float:
        """
        优化器显存因子（考虑 offload）
        
        tensorwise_offload: 优化器状态按 tensor 粒度动态 offload
        
        重要约束：tensorwise_offload 依赖 Sharding 机制
        当 Sharding degree = 1 (DP=1) 时，offload 无法正常工作
        
        这里不再使用模型修正因子，而是显式建模 PaddleFormers
        tensorwise offload 的 live window：
        - 绝大部分状态可迁移到 CPU
        - 但 GPU 上仍需要保留当前更新 bucket 的预取/回写窗口
        - 这使得显存常驻比例有一个框架级下界
        """
        if self.tensorwise_offload:
            if self.degree <= 1:
                return 1.0
            configured_resident_fraction = max(
                0.0, 1.0 - self.tensorwise_offload_ratio
            )
            return min(
                1.0,
                max(
                    configured_resident_fraction,
                    TENSORWISE_OFFLOAD_LIVE_WINDOW_FRACTION,
                ),
            )
        elif self.cpu_offload:
            return 0.0
        return 1.0
    
    def uses_release_grads(self) -> bool:
        """是否启用梯度释放（显存优化）"""
        return self.release_grads


@dataclass
class RecomputeConfig:
    """重计算配置"""
    granularity: RecomputeGranularity = RecomputeGranularity.FULL
    method: str = "uniform"  # "uniform", "block", "first_n"
    num_layers: int = 1  # 重计算间隔
    modules: Tuple[str, ...] = field(default_factory=tuple)

    def normalized_modules(self) -> Tuple[str, ...]:
        return tuple(
            sorted({str(module).strip().lower() for module in self.modules if module})
        )

    def recomputes_module(self, module: str) -> bool:
        if self.granularity == RecomputeGranularity.FULL:
            return True
        if self.granularity != RecomputeGranularity.SELECTIVE:
            return False
        return module.strip().lower() in self.normalized_modules()

    def recomputes_mlp(self) -> bool:
        return self.recomputes_module("mlp")

    def recomputes_moe_gate_up(self) -> bool:
        if self.granularity == RecomputeGranularity.FULL:
            return True
        return self.recomputes_mlp() or self.recomputes_module("moe_gate_up")

    def get_recompute_overhead(self) -> float:
        """
        获取重计算的时间开销因子
        
        Full recompute: 反向传播时需要重新计算前向激活
        """
        if self.granularity == RecomputeGranularity.NONE:
            return 1.0
        
        if self.granularity == RecomputeGranularity.SELECTIVE:
            return 1.15  # 约 15% 额外计算
        
        if self.granularity == RecomputeGranularity.FULL:
            if self.method == "uniform":
                n = max(1, self.num_layers)
                if n == 1:
                    return 1.33  # 需要重算整个前向
                else:
                    recompute_ratio = (n - 1) / n
                    return 1.0 + recompute_ratio * 0.33
            return 1.33
        
        return 1.0


@dataclass
class MemoryBreakdown:
    """
    显存分解结果

    核心公式（适用于所有 recompute / offload 组合）:

    model_states = parameter + gradient + optimizer
    resident_aux = master_grad + tensor_fusion_buffer
    peak_ws      = max(activation, temporary_buffer, optimizer_update_workspace)

    allocated    = model_states + resident_aux + peak_ws + framework_overhead
    reserved     = allocated + temporary_buffer + activation_buffer_pool
    multi_node_reserved = reserved + multi_node_runtime_addon

    说明:
    - communication_buffer 拆成 TP/PP/EP/Sharding 四类 live pool
    - communication_runtime_pool / overlap_runtime_pool 仅在多机时作为附加层使用
    - activation_buffer_pool 代表 recompute 释放后 allocator 保留的已释放块

    PP > 1 时，estimate_memory 会对每个 PP stage 分别计算，
    返回显存最大的那个 stage 的 breakdown。peak_stage_id 指示是哪个 stage。
    """
    # 主要组成 (GB)
    parameter_memory_gb: float = 0.0
    gradient_memory_gb: float = 0.0
    optimizer_memory_gb: float = 0.0
    activation_memory_gb: float = 0.0

    # 其他 (GB)
    master_grad_memory_gb: float = 0.0
    tensor_fusion_buffer_gb: float = 0.0
    communication_buffer_gb: float = 0.0
    tp_comm_buffer_gb: float = 0.0
    pp_comm_buffer_gb: float = 0.0
    ep_comm_buffer_gb: float = 0.0
    sharding_comm_buffer_gb: float = 0.0
    forward_backward_comm_live_gb: float = 0.0
    loss_comm_live_gb: float = 0.0
    optimizer_comm_live_gb: float = 0.0
    forward_backward_overlap_live_gb: float = 0.0
    loss_overlap_live_gb: float = 0.0
    optimizer_overlap_live_gb: float = 0.0
    temporary_buffer_gb: float = 0.0
    peak_runtime_workspace_gb: float = 0.0
    peak_runtime_workspace_source: str = "activation"
    optimizer_update_workspace_gb: float = 0.0
    framework_overhead_gb: float = 2.67  # CUDA/Paddle runtime + cuBLASLt workspace
    activation_no_recompute_gb: float = 0.0
    activation_saved_by_recompute_gb: float = 0.0

    # PaddleFormers allocator 预留池:
    # recompute 释放的激活在 CUDA caching allocator 中被缓存（不归还 OS），
    # 后续阶段可复用这些块。buffer_pool 建模这种 reserved-allocated 差。
    activation_buffer_pool_gb: float = 0.0
    communication_runtime_pool_gb: float = 0.0
    overlap_runtime_pool_gb: float = 0.0
    communication_fragmentation_gb: float = 0.0
    reserved_candidate_forward_backward_gb: float = 0.0
    reserved_candidate_loss_gb: float = 0.0
    reserved_candidate_optimizer_gb: float = 0.0
    reserved_candidate_post_step_gb: float = 0.0
    allocated_peak_memory_gb: float = 0.0
    allocated_peak_stage_id: int = 0
    reserved_peak_memory_gb: float = 0.0
    reserved_peak_source: str = "allocated_peak"

    # PP stage 信息: 当 PP > 1 时，标识显存最大的 stage
    peak_stage_id: int = 0
    pp_inflight_micro_batches: float = 1.0

    @property
    def allocated_memory_gb(self) -> float:
        """
        Peak allocated 显存 (GB)。

        allocated = (
            model_states
            + master_grad
            + tensor_fusion_buffer
            + peak_runtime_workspace
            + framework_overhead
        )

        peak_runtime_workspace = max(
            activation, temporary_buffer, optimizer_update_workspace
        )
        """
        if self.allocated_peak_memory_gb > 0.0:
            return self.allocated_peak_memory_gb
        return (
            self.model_states_gb
            + self.master_grad_memory_gb
            + self.tensor_fusion_buffer_gb
            + self.peak_runtime_workspace_gb
            + self.framework_overhead_gb
        )

    @property
    def reserved_memory_gb(self) -> float:
        """
        Reserved 显存 (GB)。

        reserved 取阶段候选 high-water mark。
        当显式候选值已在 estimate_memory 中计算完成时，优先返回它；
        否则回退到共享池近似。
        """
        if self.reserved_peak_memory_gb > 0.0:
            return self.reserved_peak_memory_gb
        return (
            self.allocated_memory_gb
            + self.temporary_buffer_gb
            + self.activation_buffer_pool_gb
        )

    @property
    def total_memory_gb(self) -> float:
        """总显存占用 (等于 reserved)"""
        return self.reserved_memory_gb

    @property
    def model_states_gb(self) -> float:
        """模型状态显存 (参数 + 梯度 + 优化器)"""
        return (
            self.parameter_memory_gb
            + self.gradient_memory_gb
            + self.optimizer_memory_gb
        )

    @property
    def peak_phase(self) -> str:
        """峰值出现在哪个阶段: 'forward_backward' 或 'optimizer_update'"""
        if self.peak_runtime_workspace_source == "optimizer_update":
            return "optimizer_update"
        return "forward_backward"

    def to_dict(self) -> Dict:
        return {
            "parameter_memory_gb": round(self.parameter_memory_gb, 3),
            "gradient_memory_gb": round(self.gradient_memory_gb, 3),
            "optimizer_memory_gb": round(self.optimizer_memory_gb, 3),
            "activation_memory_gb": round(self.activation_memory_gb, 3),
            "master_grad_memory_gb": round(self.master_grad_memory_gb, 3),
            "tensor_fusion_buffer_gb": round(self.tensor_fusion_buffer_gb, 3),
            "communication_buffer_gb": round(self.communication_buffer_gb, 3),
            "tp_comm_buffer_gb": round(self.tp_comm_buffer_gb, 3),
            "pp_comm_buffer_gb": round(self.pp_comm_buffer_gb, 3),
            "ep_comm_buffer_gb": round(self.ep_comm_buffer_gb, 3),
            "sharding_comm_buffer_gb": round(self.sharding_comm_buffer_gb, 3),
            "forward_backward_comm_live_gb": round(
                self.forward_backward_comm_live_gb, 3
            ),
            "loss_comm_live_gb": round(self.loss_comm_live_gb, 3),
            "optimizer_comm_live_gb": round(self.optimizer_comm_live_gb, 3),
            "forward_backward_overlap_live_gb": round(
                self.forward_backward_overlap_live_gb, 3
            ),
            "loss_overlap_live_gb": round(self.loss_overlap_live_gb, 3),
            "optimizer_overlap_live_gb": round(
                self.optimizer_overlap_live_gb, 3
            ),
            "temporary_buffer_gb": round(self.temporary_buffer_gb, 3),
            "peak_runtime_workspace_gb": round(self.peak_runtime_workspace_gb, 3),
            "peak_runtime_workspace_source": self.peak_runtime_workspace_source,
            "optimizer_update_workspace_gb": round(self.optimizer_update_workspace_gb, 3),
            "activation_no_recompute_gb": round(self.activation_no_recompute_gb, 3),
            "activation_saved_by_recompute_gb": round(self.activation_saved_by_recompute_gb, 3),
            "activation_buffer_pool_gb": round(self.activation_buffer_pool_gb, 3),
            "communication_runtime_pool_gb": round(
                self.communication_runtime_pool_gb, 3
            ),
            "overlap_runtime_pool_gb": round(self.overlap_runtime_pool_gb, 3),
            "communication_fragmentation_gb": round(
                self.communication_fragmentation_gb, 3
            ),
            "reserved_candidate_forward_backward_gb": round(
                self.reserved_candidate_forward_backward_gb, 3
            ),
            "reserved_candidate_loss_gb": round(
                self.reserved_candidate_loss_gb, 3
            ),
            "reserved_candidate_optimizer_gb": round(
                self.reserved_candidate_optimizer_gb, 3
            ),
            "reserved_candidate_post_step_gb": round(
                self.reserved_candidate_post_step_gb, 3
            ),
            "allocated_peak_stage_id": self.allocated_peak_stage_id,
            "reserved_peak_source": self.reserved_peak_source,
            "peak_phase": self.peak_phase,
            "peak_stage_id": self.peak_stage_id,
            "pp_inflight_micro_batches": self.pp_inflight_micro_batches,
            "allocated_memory_gb": round(self.allocated_memory_gb, 3),
            "reserved_memory_gb": round(self.reserved_memory_gb, 3),
            "model_states_gb": round(self.model_states_gb, 3),
            "total_memory_gb": round(self.total_memory_gb, 3),
        }

    def __str__(self) -> str:
        stage_info = ""
        if self.pp_inflight_micro_batches > 1 or self.peak_stage_id > 0:
            stage_info = (
                f"  Peak Stage:    {self.peak_stage_id}"
                f"  (inflight: {self.pp_inflight_micro_batches} micro-batches)\n"
            )
        return (
            f"Memory Breakdown:\n"
            f"  Parameters:    {self.parameter_memory_gb:.2f} GB\n"
            f"  Gradients:     {self.gradient_memory_gb:.2f} GB\n"
            f"  Optimizer:     {self.optimizer_memory_gb:.2f} GB\n"
            f"  Master Grad:   {self.master_grad_memory_gb:.2f} GB\n"
            f"  Fusion Buf:    {self.tensor_fusion_buffer_gb:.2f} GB\n"
            f"  Activation:    {self.activation_memory_gb:.2f} GB\n"
            f"  Comm Buf:      {self.communication_buffer_gb:.2f} GB\n"
            f"    - TP/PP/EP/SD: {self.tp_comm_buffer_gb:.2f} / "
            f"{self.pp_comm_buffer_gb:.2f} / {self.ep_comm_buffer_gb:.2f} / "
            f"{self.sharding_comm_buffer_gb:.2f} GB\n"
            f"  Temp Buf:      {self.temporary_buffer_gb:.2f} GB\n"
            f"  Opt WS:        {self.optimizer_update_workspace_gb:.2f} GB\n"
            f"  ─────────────────────────\n"
            f"  Allocated:     {self.allocated_memory_gb:.2f} GB"
            f"  (peak: {self.peak_phase})\n"
            f"  Buffer Pool:   {self.activation_buffer_pool_gb:.2f} GB\n"
            f"  Comm Pool:     {self.communication_runtime_pool_gb:.2f} GB\n"
            f"  Overlap Pool:  {self.overlap_runtime_pool_gb:.2f} GB\n"
            + stage_info +
            f"  ─────────────────────────\n"
            f"  Reserved:      {self.reserved_memory_gb:.2f} GB"
        )


class _BaseMemoryModel:
    """
    显存模型
    
    精确预测 PaddleFormers 分布式训练的显存占用
    """
    
    def __init__(self,
                 model_config: ModelConfig,
                 training_config: TrainingConfig,
                 hardware_config: Optional[HardwareConfig] = None):
        self.model = model_config
        self.training = training_config
        self.hardware = hardware_config

    def _is_moe_layer(self, layer_idx: int) -> bool:
        return self.model.is_moe_layer(layer_idx)

    def _transformer_layer_kind(self, layer_idx: int) -> str:
        return self.model.transformer_layer_kind(layer_idx)

    def _is_multi_node_prediction(self) -> bool:
        if self.hardware is None:
            return False
        try:
            return int(getattr(self.hardware, "num_nodes", 1)) > 1
        except Exception:
            return False

    def _get_virtual_pipeline_size(self, parallel: ParallelConfig) -> int:
        raw_value = getattr(parallel, "vpp", 1)
        try:
            return max(1, int(raw_value))
        except Exception:
            return 1

    def _chunk_layer_ranges(self, parallel: ParallelConfig) -> list[tuple[int, int]]:
        return resolve_chunk_ranges(int(self.model.num_hidden_layers), parallel)

    def _stage_chunk_ranges(self, parallel: ParallelConfig) -> list[list[tuple[int, int]]]:
        return resolve_stage_chunk_ranges(int(self.model.num_hidden_layers), parallel)

    def _stage_layer_indices(self, parallel: ParallelConfig, stage_id: int) -> list[int]:
        return resolve_stage_layer_indices(
            int(self.model.num_hidden_layers), parallel, stage_id
        )

    def _stage_moe_layer_count(self, parallel: ParallelConfig, stage_id: int) -> int:
        return sum(
            1 for layer_idx in self._stage_layer_indices(parallel, stage_id)
            if self._is_moe_layer(layer_idx)
        )

    def _is_degree_intra_node(self, degree: int) -> bool:
        if self.hardware is None:
            return True
        return self.hardware.is_intra_node(max(1, degree))

    def _stage_group_width(self, parallel: ParallelConfig) -> int:
        pp = max(1, int(parallel.pp))
        total_gpus = int(getattr(self.hardware, "total_gpus", parallel.world_size))
        return max(1, total_gpus // pp)

    def _is_pipeline_boundary_intra_node(self,
                                         parallel: ParallelConfig,
                                         boundary_idx: int) -> bool:
        if self.hardware is None or parallel.pp <= 1:
            return True

        gpus_per_node = max(1, int(self.hardware.gpus_per_node))
        topo_order = str(
            getattr(self.training, "hybrid_parallel_topo_order", "") or ""
        ).strip().lower()

        if topo_order == "sharding_first":
            stage_width = max(1, parallel.effective_sharding_degree * parallel.tp)
            src_rank = boundary_idx * stage_width
            dst_rank = src_rank + stage_width
        else:
            stage_width = self._stage_group_width(parallel)
            src_rank = (boundary_idx + 1) * stage_width - 1
            dst_rank = src_rank + 1

        return (src_rank // gpus_per_node) == (dst_rank // gpus_per_node)

    def _normalize_recompute_method(self, recompute: RecomputeConfig) -> str:
        method = str(recompute.method or "").strip().lower()
        if recompute.granularity == RecomputeGranularity.FULL:
            return method
        if recompute.granularity == RecomputeGranularity.SELECTIVE:
            if method in ("block", "first_n"):
                return method
            return ""
        return ""

    def _validate_recompute_config(self, recompute: RecomputeConfig) -> None:
        if recompute.granularity == RecomputeGranularity.NONE:
            return

        method = self._normalize_recompute_method(recompute)

        if recompute.granularity == RecomputeGranularity.FULL:
            if method not in ("uniform", "block", "first_n"):
                raise ValueError(
                    "when recompute_granularity=full, recompute_method must be one of "
                    "'uniform', 'block' and 'first_n'"
                )
            if recompute.num_layers is None:
                raise ValueError(
                    "when recompute_granularity=full, recompute_num_layers must not be None"
                )
            if int(recompute.num_layers) <= 0:
                raise ValueError(
                    "when recompute_granularity=full, recompute_num_layers must be > 0"
                )
            return

        if recompute.granularity == RecomputeGranularity.SELECTIVE:
            if method not in ("", "block", "first_n"):
                raise ValueError(
                    "when recompute_granularity=selective, recompute_method must be one of "
                    "'block', 'first_n' or None"
                )
            if method in ("block", "first_n"):
                if recompute.num_layers is None:
                    raise ValueError(
                        "when recompute_granularity=selective and recompute_method is "
                        "'block' or 'first_n', recompute_num_layers must not be None"
                    )
                if int(recompute.num_layers) <= 0:
                    raise ValueError(
                        "when recompute_granularity=selective and recompute_method is "
                        "'block' or 'first_n', recompute_num_layers must be > 0"
                    )
            return

        raise ValueError("recompute_granularity must be one of none, selective and full")

    def _select_full_recomputed_layers(self,
                                       stage_chunks: list[tuple[int, int]],
                                       recompute_method: str,
                                       recompute_num_layers: Optional[int]) -> list[int]:
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

    def _select_selective_recomputed_layers(self,
                                            stage_chunks: list[tuple[int, int]],
                                            recompute_method: str,
                                            recompute_num_layers: Optional[int]) -> list[int]:
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

    def _estimate_full_uniform_unit_count(self,
                                          stage_chunks: list[tuple[int, int]],
                                          recompute_num_layers: Optional[int]) -> int:
        unit_layers = max(1, int(recompute_num_layers or 1))
        unit_count = 0
        for chunk_start, chunk_end in stage_chunks:
            chunk_len = max(0, chunk_end - chunk_start)
            if chunk_len > 0:
                unit_count += math.ceil(chunk_len / unit_layers)
        return unit_count

    def _build_activation_layer_components(self,
                                           parallel: ParallelConfig,
                                           seq_len: int,
                                           micro_batch_size: int) -> Dict[str, object]:
        tokens = self._effective_token_count(parallel, seq_len, micro_batch_size)
        hidden_partition_factor = self._activation_hidden_partition_factor(parallel)

        h = self.model.hidden_size
        kv = self.model.num_key_value_heads * self.model.head_dim
        ffn = self.model.intermediate_size
        moe_ffn = self.model.moe_intermediate_size
        topk = self.model.num_experts_per_tok
        dtype_bytes = self.training.dtype_bytes

        selective_base_checkpoint = tokens * (
            (CHECKPOINT_HIDDEN_TENSOR_EQUIV * h) / hidden_partition_factor
            + CHECKPOINT_KV_TENSOR_EQUIV * kv
        ) * dtype_bytes
        full_checkpoint = tokens * (
            (FULL_RECOMPUTE_HIDDEN_TENSOR_EQUIV * h) / hidden_partition_factor
            + FULL_RECOMPUTE_KV_TENSOR_EQUIV * kv
        ) * dtype_bytes

        dense_gate_up = tokens * (2 * ffn / max(1, parallel.tp)) * dtype_bytes
        dense_mlp_input = tokens * (h / hidden_partition_factor) * dtype_bytes
        moe_gate_up = tokens * topk * (2 * moe_ffn) * dtype_bytes
        moe_mlp_input = tokens * (topk * h + (h / hidden_partition_factor)) * dtype_bytes

        templates = {
            TRANSFORMER_DENSE_LAYER_KIND: {
                "layer_kind": TRANSFORMER_DENSE_LAYER_KIND,
                "layer_type": "dense",
                "no_recompute": selective_base_checkpoint + dense_gate_up + dense_mlp_input,
                "selective_base_checkpoint": selective_base_checkpoint,
                "full_checkpoint": full_checkpoint,
                "mlp_saved": dense_gate_up + dense_mlp_input,
                "moe_gate_up_saved": 0.0,
            },
            TRANSFORMER_MOE_LAYER_KIND: {
                "layer_kind": TRANSFORMER_MOE_LAYER_KIND,
                "layer_type": "moe",
                "no_recompute": selective_base_checkpoint + moe_gate_up + moe_mlp_input,
                "selective_base_checkpoint": selective_base_checkpoint,
                "full_checkpoint": full_checkpoint,
                "mlp_saved": moe_gate_up + moe_mlp_input,
                "moe_gate_up_saved": moe_gate_up,
            },
        }

        layers = []
        for layer_idx in range(self.model.num_hidden_layers):
            layer_kind = self._transformer_layer_kind(layer_idx)
            layers.append({
                "layer_idx": int(layer_idx),
                "layer_kind": layer_kind,
                "template_key": layer_kind,
            })
        return {
            "templates": templates,
            "layers": layers,
        }

    def _get_activation_layer_template(self,
                                       payload: Dict[str, object],
                                       layer_entry: Dict[str, Union[int, str]]) -> Dict[str, float]:
        template_key = str(layer_entry.get("template_key", layer_entry.get("layer_kind", "")))
        templates = payload["templates"]  # type: ignore[index]
        return templates[template_key]  # type: ignore[index]

    def _build_parameter_templates(self,
                                   parallel: ParallelConfig) -> Dict[str, Dict[str, int]]:
        h = self.model.hidden_size
        ffn = self.model.intermediate_size
        moe_ffn = self.model.moe_intermediate_size
        v = self.model.vocab_size

        tp = parallel.tp
        ep = parallel.ep

        attention_params = self.model.estimate_attention_params_per_layer() // tp
        transformer_layernorm_params = 4 * h
        final_norm_params = 2 * h
        dense_mlp_params = 3 * h * ffn // tp
        router_params = h * self.model.num_experts
        experts_per_gpu = self.model.num_experts // ep
        expert_params = 3 * h * moe_ffn * experts_per_gpu // tp
        single_embedding = v * h // tp

        return {
            TRANSFORMER_DENSE_LAYER_KIND: {
                "attention_params": attention_params,
                "layernorm_params": transformer_layernorm_params,
                "dense_mlp_params": dense_mlp_params,
                "router_params": 0,
                "expert_params": 0,
                "embedding_params": 0,
                "input_embedding_params": 0,
                "output_head_params": 0,
                "total_params": (
                    attention_params + transformer_layernorm_params + dense_mlp_params
                ),
            },
            TRANSFORMER_MOE_LAYER_KIND: {
                "attention_params": attention_params,
                "layernorm_params": transformer_layernorm_params,
                "dense_mlp_params": 0,
                "router_params": router_params,
                "expert_params": expert_params,
                "embedding_params": 0,
                "input_embedding_params": 0,
                "output_head_params": 0,
                "total_params": (
                    attention_params
                    + transformer_layernorm_params
                    + router_params
                    + expert_params
                ),
            },
            INPUT_EMBEDDING_LAYER_KIND: {
                "attention_params": 0,
                "layernorm_params": 0,
                "dense_mlp_params": 0,
                "router_params": 0,
                "expert_params": 0,
                "embedding_params": single_embedding,
                "input_embedding_params": single_embedding,
                "output_head_params": 0,
                "total_params": single_embedding,
            },
            OUTPUT_HEAD_LAYER_KIND: {
                "attention_params": 0,
                "layernorm_params": final_norm_params,
                "dense_mlp_params": 0,
                "router_params": 0,
                "expert_params": 0,
                "embedding_params": single_embedding,
                "input_embedding_params": 0,
                "output_head_params": single_embedding + final_norm_params,
                "total_params": single_embedding + final_norm_params,
            },
        }
    
    def estimate_parameter_count_per_gpu(self, parallel: ParallelConfig,
                                         stage_id: Optional[int] = None) -> Dict[str, int]:
        """
        估算每个 GPU 上的参数数量
        
        考虑 TP/PP/EP 切分

        Args:
            parallel: 并行配置
            stage_id: PP stage 编号 (0..PP-1)。None 时使用旧逻辑（兼容）。
                      指定后会精确计算该 stage 的 embedding 分配。
        """
        num_layers = self.model.num_hidden_layers
        pp = parallel.pp
        parameter_templates = self._build_parameter_templates(parallel)
        result = {
            "attention_params": 0,
            "layernorm_params": 0,
            "dense_mlp_params": 0,
            "router_params": 0,
            "expert_params": 0,
            "embedding_params": 0,
            "input_embedding_params": 0,
            "output_head_params": 0,
            "total_params": 0,
        }

        def _accumulate(template_key: str) -> None:
            template = parameter_templates[template_key]
            for key in result:
                result[key] += int(template.get(key, 0))

        if stage_id is not None:
            for layer_idx in self._stage_layer_indices(parallel, stage_id):
                _accumulate(self._transformer_layer_kind(layer_idx))
            if stage_id == 0:
                _accumulate(INPUT_EMBEDDING_LAYER_KIND)
            if stage_id == pp - 1:
                _accumulate(OUTPUT_HEAD_LAYER_KIND)
            return result

        dense_layers_per_stage = self.model.num_dense_layers // pp
        moe_layers_per_stage = self.model.num_moe_layers // pp
        for _ in range(dense_layers_per_stage):
            _accumulate(TRANSFORMER_DENSE_LAYER_KIND)
        for _ in range(moe_layers_per_stage):
            _accumulate(TRANSFORMER_MOE_LAYER_KIND)

        if pp == 1:
            _accumulate(INPUT_EMBEDDING_LAYER_KIND)
            _accumulate(OUTPUT_HEAD_LAYER_KIND)
        else:
            average_embedding_params = (
                parameter_templates[INPUT_EMBEDDING_LAYER_KIND]["embedding_params"]
            )
            result["embedding_params"] += int(average_embedding_params)
            result["total_params"] += int(average_embedding_params)

        return result
    
    def estimate_parameter_memory(self, parallel: ParallelConfig,
                                  sharding: ShardingConfig,
                                  stage_id: Optional[int] = None) -> float:
        """
        估算参数显存 (GB)
        
        关键点：专家参数和非专家参数的 Sharding 处理不同
        - 专家参数：已被 EP 切分，不受 Sharding 切分（专家是局部的）
        - 非专家参数：受 Sharding split_param 切分
        """
        param_count = self.estimate_parameter_count_per_gpu(parallel, stage_id)
        
        # 分离专家参数和非专家参数
        expert_params = param_count["expert_params"]  # 已被 EP 切分
        non_expert_params = param_count["total_params"] - expert_params
        
        # Sharding 分片因子（只对非专家参数生效）
        sharding_factor = sharding.get_param_sharding_factor()
        
        # 专家参数显存（不受 Sharding 切分）
        expert_bytes = expert_params * self.training.dtype_bytes
        
        # 非专家参数显存（受 Sharding 切分）
        non_expert_bytes = non_expert_params * self.training.dtype_bytes / sharding_factor
        
        total_bytes = expert_bytes + non_expert_bytes
        
        return total_bytes / (1024 ** 3)
    
    def estimate_gradient_memory(self, parallel: ParallelConfig,
                                 sharding: ShardingConfig,
                                 stage_id: Optional[int] = None) -> float:
        """
        估算梯度显存 (GB)

        PaddleFormers 梯度管理：
        - 梯度在 backward 阶段通过 fused comm buffer 逐 bucket 累积
        - gradient_accumulation_steps > 1 时，梯度持续累积直到 optimizer step
        - 在 ReduceScatter 之前，梯度 buffer 是完整大小（未分片）
        - 对 stage1+split_param：所有参数（含专家）统一使用 ReduceScatter
        """
        param_count = self.estimate_parameter_count_per_gpu(parallel, stage_id)
        total_params = param_count["total_params"]
        total_grad_bytes = total_params * self.training.dtype_bytes
        return total_grad_bytes / (1024 ** 3)
    
    def estimate_optimizer_memory(self, parallel: ParallelConfig,
                                  sharding: ShardingConfig,
                                  stage_id: Optional[int] = None) -> float:
        """
        估算优化器状态显存 (GB)
        
        AdamW: 2 × fp32 状态 (momentum + variance)
        
        关键点：
        1. 专家参数和非专家参数的 Sharding 处理不同
           - 专家参数：已被 EP 切分，不受 Sharding 切分
           - 非专家参数：受 Sharding 切分
        2. tensorwise_offload 在实际实现中：
           - 主要 offload 非专家参数的优化器状态
           - 专家参数优化器通常保留在 GPU 上以保证性能
        """
        param_count = self.estimate_parameter_count_per_gpu(parallel, stage_id)
        
        # 分离专家参数和非专家参数
        expert_params = param_count["expert_params"]
        non_expert_params = param_count["total_params"] - expert_params
        
        # ZeRO 优化器分片因子（只对非专家参数生效）
        sharding_factor = sharding.get_optimizer_sharding_factor()
        
        # Offload 因子
        offload_factor = sharding.get_optimizer_memory_factor()
        
        # AdamW: 2 个 fp32 状态
        # 修正：专家优化器也受 tensorwise_offload 影响
        # PaddleFormers 的 tensorwise_offload 对所有参数生效，包括专家
        # 但专家参数不受 Sharding 切分（因为已被 EP 切分）
        expert_opt_bytes = expert_params * 4 * 2 * offload_factor  # 专家也 offload
        
        # 非专家优化器显存（受 Sharding 切分，受 offload 影响）
        non_expert_opt_bytes = non_expert_params * 4 * 2 / sharding_factor * offload_factor
        
        optimizer_bytes = expert_opt_bytes + non_expert_opt_bytes
        
        # 注意:
        # amp_master_grad 表示梯度以 FP32 进行主副本更新，
        # 不是长期常驻的 full-size master weight。
        # 这里不将其按"参数等量常驻显存"计入 optimizer states，
        # 避免系统性高估。
        
        return optimizer_bytes / (1024 ** 3)

    def estimate_master_grad_memory(self,
                                    parallel: ParallelConfig,
                                    sharding: ShardingConfig,
                                    gradient_memory_gb: float) -> float:
        """
        估算常驻 FP32 master-grad 显存。

        Paddle O2 + amp_master_grad 在非 offload 路径下通常会保留一份
        FP32 master grad，用于 optimizer update。开启 tensorwise offload 时，
        这部分更接近 update 阶段工作集，而不是长期常驻状态。
        """
        if (
            not bool(getattr(self.training, "amp_master_grad", False))
            or self.training.dtype_bytes >= 4
            or gradient_memory_gb <= 0.0
        ):
            return 0.0
        if sharding.tensorwise_offload and sharding.degree > 1:
            return 0.0
        return gradient_memory_gb * (4.0 / float(self.training.dtype_bytes))

    def estimate_tensor_fusion_buffer(self,
                                      parallel: ParallelConfig,
                                      sharding: ShardingConfig,
                                      stage_id: Optional[int] = None) -> float:
        """
        估算 Paddle 静态/半静态 tensor fusion buffer。

        这些 buffer 在 allocator 里通常表现为 allocated，
        但不属于参数、梯度或 optimizer states。
        """
        param_count = self.estimate_parameter_count_per_gpu(parallel, stage_id)
        total_params = int(param_count["total_params"])
        expert_params = int(param_count["expert_params"])
        non_expert_params = max(0, total_params - expert_params)

        expert_grad_gb = expert_params * self.training.dtype_bytes / GIB
        non_expert_grad_gb = non_expert_params * self.training.dtype_bytes / GIB

        fusion_buffer_gb = 0.0
        if sharding.degree > 1 or parallel.dp > 1:
            dense_factor = 1.12
            if bool(getattr(self.training, "amp_master_grad", False)):
                dense_factor += 0.06
            if not self.training.clear_every_step_cache:
                dense_factor += 0.05
            if self.training.variable_seq_lengths or self.training.enable_dynamic_shape:
                dense_factor += 0.03
            if (
                self.training.stage1_overlap
                or self.training.enable_sharding_comm_overlap
            ):
                dense_factor += 0.04
            fusion_buffer_gb += non_expert_grad_gb * dense_factor

        if parallel.ep > 1 and expert_params > 0:
            expert_factor = 0.82
            if bool(getattr(self.training, "moe_expert_fusion", False)):
                expert_factor += 0.10
            if bool(getattr(self.training, "moe_grouped_gemm", False)):
                expert_factor += 0.03
            if not self.training.clear_every_step_cache:
                expert_factor += 0.03
            if self.training.variable_seq_lengths or self.training.enable_dynamic_shape:
                expert_factor += 0.02
            fusion_buffer_gb += expert_grad_gb * expert_factor

        return fusion_buffer_gb

    def _effective_token_count(self,
                               parallel: ParallelConfig,
                               seq_len: int,
                               micro_batch_size: int) -> int:
        effective_seq_len = max(1, seq_len // max(1, parallel.cp))
        return micro_batch_size * effective_seq_len

    def _activation_hidden_partition_factor(self, parallel: ParallelConfig) -> int:
        if parallel.sp and parallel.tp > 1:
            return parallel.tp
        return 1

    def _estimate_activation_workspace_bytes(
        self,
        parallel: ParallelConfig,
        recompute: RecomputeConfig,
        seq_len: int,
        micro_batch_size: int,
        stage_id: int,
    ) -> Tuple[float, float]:
        self._validate_recompute_config(recompute)

        activation_payload = self._build_activation_layer_components(
            parallel, seq_len, micro_batch_size
        )
        layers = activation_payload["layers"]  # type: ignore[index]
        stage_chunks = self._stage_chunk_ranges(parallel)[stage_id]
        stage_layer_indices = self._stage_layer_indices(parallel, stage_id)

        no_recompute_bytes = sum(
            self._get_activation_layer_template(activation_payload, layers[layer_idx])[
                "no_recompute"
            ]
            for layer_idx in stage_layer_indices
        )

        if recompute.granularity == RecomputeGranularity.NONE:
            return no_recompute_bytes, no_recompute_bytes

        method = self._normalize_recompute_method(recompute)

        if recompute.granularity == RecomputeGranularity.FULL:
            if method == "uniform":
                full_checkpoint = (
                    self._get_activation_layer_template(
                        activation_payload, layers[stage_layer_indices[0]]
                    )["full_checkpoint"]
                    if stage_layer_indices else 0.0
                )
                activation_bytes = (
                    self._estimate_full_uniform_unit_count(
                        stage_chunks, recompute.num_layers
                    ) * full_checkpoint
                )
                return activation_bytes, no_recompute_bytes

            selected_layers = set(
                self._select_full_recomputed_layers(
                    stage_chunks, method, recompute.num_layers
                )
            )
            activation_bytes = 0.0
            for layer_idx in stage_layer_indices:
                layer = self._get_activation_layer_template(
                    activation_payload, layers[layer_idx]
                )
                if layer_idx in selected_layers:
                    activation_bytes += layer["full_checkpoint"]
                else:
                    activation_bytes += layer["no_recompute"]
            return activation_bytes, no_recompute_bytes

        modules = set(recompute.normalized_modules())
        if not modules:
            return no_recompute_bytes, no_recompute_bytes

        selected_layers = set(
            self._select_selective_recomputed_layers(
                stage_chunks, method, recompute.num_layers
            )
        )
        activation_bytes = 0.0
        for layer_idx in stage_layer_indices:
            layer = self._get_activation_layer_template(
                activation_payload, layers[layer_idx]
            )
            if layer_idx not in selected_layers:
                activation_bytes += layer["no_recompute"]
                continue

            saved_bytes = 0.0
            if recompute.recomputes_mlp():
                saved_bytes += layer["mlp_saved"]
            elif recompute.recomputes_moe_gate_up():
                saved_bytes += layer["moe_gate_up_saved"]

            activation_bytes += max(
                layer["selective_base_checkpoint"],
                layer["no_recompute"] - saved_bytes,
            )

        return activation_bytes, no_recompute_bytes

    def estimate_activation_memory(self, parallel: ParallelConfig,
                                   recompute: RecomputeConfig,
                                   seq_len: int = None,
                                   micro_batch_size: int = None) -> float:
        """
        估算反向主峰阶段需要保留的激活工作集 (GB)。
        """
        micro_bsz = (
            micro_batch_size
            if micro_batch_size is not None
            else self.training.micro_batch_size
        )
        seq_len = seq_len if seq_len is not None else self.training.sequence_length
        activation_bytes, _ = self._estimate_activation_workspace_bytes(
            parallel, recompute, seq_len, micro_bsz, stage_id=0
        )
        return activation_bytes / GIB

    def estimate_loss_workspace(self, parallel: ParallelConfig,
                                seq_len: int = None,
                                micro_batch_size: int = None,
                                stage_id: Optional[int] = None) -> float:
        """
        估算 loss/logits 阶段的临时 FP32 工作集 (GB)。

        PP > 1 时，只有最后一个 stage 计算 loss。
        其他 stage 的 loss workspace = 0。
        """
        # PP > 1 且不是最后一个 stage → 该 stage 无 loss 计算
        if parallel.pp > 1 and stage_id is not None and stage_id != parallel.pp - 1:
            return 0.0

        micro_bsz = (
            micro_batch_size
            if micro_batch_size is not None
            else self.training.micro_batch_size
        )
        seq_len = seq_len if seq_len is not None else self.training.sequence_length
        tokens = self._effective_token_count(parallel, seq_len, micro_bsz)
        logits_workspace_bytes = (
            tokens
            * self.model.vocab_size
            * 4
            * CROSS_ENTROPY_WORKSPACE_FACTOR
        )
        return logits_workspace_bytes / GIB

    def _estimate_comm_pools(self,
                             parallel: ParallelConfig,
                             sharding: ShardingConfig,
                             recompute: RecomputeConfig,
                             seq_len: int,
                             micro_batch_size: int,
                             stage_id: int,
                             gradient_memory_gb: float) -> Dict[str, float]:
        tokens = self._effective_token_count(parallel, seq_len, micro_batch_size)
        dtype_bytes = self.training.dtype_bytes
        h = self.model.hidden_size
        topk = self.model.num_experts_per_tok

        base_activation_gb = (
            micro_batch_size * seq_len * h * dtype_bytes
        ) / GIB

        tp_live_gb = 0.0
        if parallel.tp > 1:
            tp_live_gb = base_activation_gb * 2.0

        pp_boundary_count = 0
        pp_inter_node_boundaries = 0
        if parallel.pp > 1:
            if stage_id > 0:
                pp_boundary_count += 1
                if not self._is_pipeline_boundary_intra_node(parallel, stage_id - 1):
                    pp_inter_node_boundaries += 1
            if stage_id < parallel.pp - 1:
                pp_boundary_count += 1
                if not self._is_pipeline_boundary_intra_node(parallel, stage_id):
                    pp_inter_node_boundaries += 1

        pp_live_factor = 2.0
        if self.training.use_batch_p2p_comm:
            pp_live_factor += 1.0
        if self.training.overlap_p2p_comm:
            pp_live_factor += 0.75
        pp_live_gb = (
            base_activation_gb
            * pp_boundary_count
            * pp_live_factor
            * (
                1.0
                + (
                    INTER_NODE_RUNTIME_BONUS
                    * (pp_inter_node_boundaries / max(1, pp_boundary_count))
                )
            )
        )

        stage_moe_layers = self._stage_moe_layer_count(parallel, stage_id)
        ep_live_gb = 0.0
        if parallel.ep > 1 and stage_moe_layers > 0:
            per_a2a_gb = (
                tokens * h * topk * dtype_bytes
            ) / GIB
            ep_is_intra = self._is_degree_intra_node(parallel.ep)
            ep_scratch_factor = (
                EP_SCRATCH_FACTOR_INTRA if ep_is_intra else EP_SCRATCH_FACTOR_INTER
            )
            ep_live_gb = (
                per_a2a_gb * 4.0
                + per_a2a_gb * EP_PACK_UNPACK_FACTOR
                + per_a2a_gb * ep_scratch_factor
            )

        sharding_degree = max(
            1, parallel.effective_sharding_degree, parallel.dp
        )
        sharding_live_gb = 0.0
        if sharding_degree > 1:
            grad_bytes = gradient_memory_gb * GIB
            bucket_bytes = min(
                MAX_COMM_BUCKET_BYTES,
                max(DEFAULT_COMM_BUCKET_BYTES, int(grad_bytes / 32.0)),
            )
            inflight_buckets = 3
            if (
                self.training.stage1_overlap
                or self.training.enable_sharding_comm_overlap
            ):
                inflight_buckets += 3
            sharding_live_gb = (bucket_bytes * inflight_buckets) / GIB

        total_live_gb = tp_live_gb + pp_live_gb + ep_live_gb + sharding_live_gb
        largest_live_class_gb = max(
            tp_live_gb, pp_live_gb, ep_live_gb, sharding_live_gb
        )

        pp_retained_factor = 0.75 if pp_boundary_count > 0 else 0.0
        if self.training.overlap_p2p_comm:
            pp_retained_factor += 0.45
        if self.training.use_batch_p2p_comm:
            pp_retained_factor += 0.35
        if self.training.p2p_cache_shape:
            pp_retained_factor += 0.35
        if self.training.variable_seq_lengths or self.training.enable_dynamic_shape:
            pp_retained_factor += 0.35
        if not self.training.clear_every_step_cache:
            pp_retained_factor += 0.45
        if self.training.best_unbalanced_scheduler:
            pp_retained_factor += 0.15
        if pp_inter_node_boundaries > 0:
            pp_retained_factor += INTER_NODE_RUNTIME_BONUS
        pp_retained_gb = pp_live_gb * max(
            0.0, min(PP_RETAIN_FACTOR_CAP, pp_retained_factor)
        )

        ep_retained_factor = 0.0
        if ep_live_gb > 0.0:
            ep_depth_bonus = 0.0
            if stage_moe_layers > 0:
                ep_depth_bonus = 0.25 * min(
                    2.0, math.log2(float(stage_moe_layers) + 1.0)
                )
            ep_retained_factor = 0.6 + ep_depth_bonus
            if self.training.variable_seq_lengths or self.training.enable_dynamic_shape:
                ep_retained_factor += 0.45
            if not self.training.clear_every_step_cache:
                ep_retained_factor += 0.55
            if self.training.best_unbalanced_scheduler:
                ep_retained_factor += 0.15
            if not self._is_degree_intra_node(parallel.ep):
                ep_retained_factor += INTER_NODE_RUNTIME_BONUS
        ep_retained_gb = ep_live_gb * max(
            0.0, min(EP_RETAIN_FACTOR_CAP, ep_retained_factor)
        )

        sharding_retained_factor = 0.0
        if sharding_live_gb > 0.0:
            sharding_retained_factor = 0.9
            if (
                self.training.stage1_overlap
                or self.training.enable_sharding_comm_overlap
            ):
                sharding_retained_factor += 0.45
            if self.training.variable_seq_lengths or self.training.enable_dynamic_shape:
                sharding_retained_factor += 0.20
            if not self.training.clear_every_step_cache:
                sharding_retained_factor += 0.30
            if not self._is_degree_intra_node(sharding_degree):
                sharding_retained_factor += 0.25
        sharding_retained_gb = sharding_live_gb * max(
            0.0, min(SHARDING_RETAIN_FACTOR_CAP, sharding_retained_factor)
        )

        tp_retained_gb = 0.0
        communication_runtime_pool_gb = (
            tp_retained_gb
            + pp_retained_gb
            + ep_retained_gb
            + sharding_retained_gb
        )

        persistent_shape_cache_factor = 0.0
        if not self.training.clear_every_step_cache:
            persistent_shape_cache_factor += STEP_CACHE_BASE_BONUS
        if self.training.variable_seq_lengths:
            persistent_shape_cache_factor += VARIABLE_SEQ_CACHE_BONUS
        if self.training.enable_dynamic_shape:
            persistent_shape_cache_factor += DYNAMIC_SHAPE_CACHE_BONUS
        if self.training.p2p_cache_shape:
            persistent_shape_cache_factor += P2P_SHAPE_CACHE_BONUS
        cache_basis_gb = largest_live_class_gb
        if (
            recompute.granularity == RecomputeGranularity.FULL
            and self._normalize_recompute_method(recompute) == "uniform"
        ):
            # uniform/full 会产生更稳定但更多类的 cached slab。
            # PP/EP 的 shape 直接跟随 token layout 变化，Sharding bucket 则保留一半作为
            # step-to-step allocator bin 扩张的近似。
            cache_basis_gb = (
                tp_live_gb
                + pp_live_gb
                + ep_live_gb
                + sharding_live_gb * 0.5
            )
        if persistent_shape_cache_factor > 0.0 and cache_basis_gb > 0.0:
            communication_runtime_pool_gb += (
                cache_basis_gb * persistent_shape_cache_factor
            )

        overlap_runtime_pool_gb = 0.0
        forward_backward_overlap_live_gb = 0.0
        loss_overlap_live_gb = 0.0
        optimizer_overlap_live_gb = 0.0
        if self.training.overlap_p2p_comm:
            forward_backward_overlap_live_gb += pp_live_gb
            overlap_runtime_pool_gb += pp_live_gb * OVERLAP_RUNTIME_REUSE_FACTOR
        if self.training.stage1_overlap or self.training.enable_sharding_comm_overlap:
            forward_backward_overlap_live_gb += sharding_live_gb
            optimizer_overlap_live_gb += (
                sharding_live_gb * OPTIMIZER_PHASE_SHARDING_LIVE_FRACTION
            )
            overlap_runtime_pool_gb += (
                sharding_live_gb * OVERLAP_RUNTIME_REUSE_FACTOR
            )
        if stage_moe_layers > 0 and (
            self.training.variable_seq_lengths or self.training.enable_dynamic_shape
        ):
            forward_backward_overlap_live_gb += ep_live_gb * 0.25
            overlap_runtime_pool_gb += ep_live_gb * 0.25

        if self.training.overlap_p2p_comm and self.training.use_batch_p2p_comm:
            loss_overlap_live_gb += pp_live_gb * 0.5

        forward_backward_comm_live_gb = (
            tp_live_gb
            + pp_live_gb
            + ep_live_gb
            + sharding_live_gb
        )
        loss_comm_live_gb = (
            pp_live_gb
            + ep_live_gb * LOSS_PHASE_EP_LIVE_FRACTION
            + sharding_live_gb * LOSS_PHASE_SHARDING_LIVE_FRACTION
        )
        optimizer_comm_live_gb = (
            sharding_live_gb * OPTIMIZER_PHASE_SHARDING_LIVE_FRACTION
        )

        return {
            "tp_comm_buffer_gb": tp_live_gb,
            "pp_comm_buffer_gb": pp_live_gb,
            "ep_comm_buffer_gb": ep_live_gb,
            "sharding_comm_buffer_gb": sharding_live_gb,
            "communication_buffer_gb": total_live_gb,
            "forward_backward_comm_live_gb": forward_backward_comm_live_gb,
            "loss_comm_live_gb": loss_comm_live_gb,
            "optimizer_comm_live_gb": optimizer_comm_live_gb,
            "forward_backward_overlap_live_gb": forward_backward_overlap_live_gb,
            "loss_overlap_live_gb": loss_overlap_live_gb,
            "optimizer_overlap_live_gb": optimizer_overlap_live_gb,
            "communication_runtime_pool_gb": communication_runtime_pool_gb,
            "overlap_runtime_pool_gb": overlap_runtime_pool_gb,
        }

    def estimate_communication_buffer(self, parallel: ParallelConfig,
                                      micro_batch_size: int = None,
                                      seq_len: int = None) -> float:
        """
        估算通信缓冲区 live 工作集显存 (GB)
        
        包括:
        - TP AllReduce buffer
        - PP Send/Recv buffer  
        - DP AllReduce/ReduceScatter buffer
        - EP AllToAll buffer
        """
        micro_bsz = (
            micro_batch_size
            if micro_batch_size is not None
            else self.training.micro_batch_size
        )
        seq_len = seq_len if seq_len is not None else self.training.sequence_length
        h = self.model.hidden_size
        
        buffer_bytes = 0
        
        # TP 通信缓冲区
        if parallel.tp > 1:
            tp_buffer = micro_bsz * seq_len * h * self.training.dtype_bytes
            buffer_bytes += tp_buffer * 2  # 前向 + 后向
        
        # PP 通信缓冲区
        if parallel.pp > 1:
            pp_buffer = micro_bsz * seq_len * h * self.training.dtype_bytes
            buffer_bytes += pp_buffer * 2  # send + recv
        
        # EP AllToAll 缓冲区
        if parallel.ep > 1:
            # AllToAll 的 dispatch 和 combine 各需要 input + output 缓冲区
            # dispatch: 每 token 路由到 topk 个 expert, 数据量 = tokens * topk * h
            # combine: 将 expert 输出收集回来, 数据量同上
            # NCCL/DeepEP 内部还需要与 ep_degree 相关的额外 scratch
            topk = self.model.num_experts_per_tok
            per_a2a_buffer = micro_bsz * seq_len * h * topk * self.training.dtype_bytes
            # dispatch + combine, 每次 AllToAll 需要 input + output (2x)
            buffer_bytes += per_a2a_buffer * 2 * 2
            # NCCL AllToAll scratch: 与 EP group size 相关的内部缓冲
            nccl_scratch = per_a2a_buffer * 0.5
            buffer_bytes += int(nccl_scratch)
        
        # SP AllGather 缓冲区
        if parallel.sp and parallel.tp > 1:
            sp_buffer = micro_bsz * seq_len * h * self.training.dtype_bytes
            buffer_bytes += sp_buffer
        
        # Sharding/DP gradient bucket buffer
        # PaddleFormers 的 Sharding 使用 gradient bucket 进行 AllReduce/ReduceScatter
        # bucket 大小通常为 25-40 MB，但多个 bucket 可以同时 in-flight
        if parallel.effective_sharding_degree > 1 or parallel.dp > 1:
            bucket_size_bytes = 40 * 1024 * 1024  # 40 MB per bucket
            num_inflight_buckets = 3  # pipeline: current + prefetch + write-back
            buffer_bytes += bucket_size_bytes * num_inflight_buckets

        return buffer_bytes / (1024 ** 3)
    
    def estimate_activation_buffer_pool(self,
                                        recompute: RecomputeConfig,
                                        activation_memory_gb: float,
                                        activation_no_recompute_gb: float,
                                        temporary_buffer_gb: float,
                                        optimizer_update_workspace_gb: float) -> float:
        """
        估算 allocator 在不同阶段之间保留下来的预留池。

        PP > 1 时，非末尾 stage 没有 loss workspace (temporary_buffer = 0)，
        但 full recompute 的 backward 仍会为每一层重算完整前向激活，
        用完后释放。CUDA caching allocator 会保留这些被释放的块，
        形成显著的 reserved - allocated 差距。
        """
        saved_activation_gb = max(
            0.0, activation_no_recompute_gb - activation_memory_gb
        )
        if saved_activation_gb <= 0.0:
            return 0.0

        # ── full recompute ──
        if recompute.granularity == RecomputeGranularity.FULL:
            if temporary_buffer_gb > 0.0:
                # 存在 loss workspace（PP=1 或 PP 末尾 stage）
                if activation_memory_gb > optimizer_update_workspace_gb:
                    # 激活仍是主峰，loss workspace 已涵盖 reserved-allocated 差
                    return 0.0
                # optimizer_update 为主峰，使用调和均值公式
                return (
                    FULL_RECOMPUTE_REUSE_FACTOR
                    * temporary_buffer_gb
                    * saved_activation_gb
                    / (saved_activation_gb + temporary_buffer_gb)
                )
            else:
                # 无 loss workspace（PP > 1 非末尾 stage）
                # backward recompute 每层重算全部激活后释放，allocator 缓存这些块
                return saved_activation_gb * FULL_RECOMPUTE_REUSE_FACTOR

        # ── selective recompute / no recompute ──
        if temporary_buffer_gb <= 0.0:
            return 0.0
        if (
            recompute.granularity == RecomputeGranularity.SELECTIVE
            and activation_memory_gb <= optimizer_update_workspace_gb
        ):
            # selective 且 optimizer update 为主峰时，allocator 更像是整块保留
            # 一份 temp/loss workspace 尺寸的 slab，而不是与 saved activation 做平均。
            return ALLOCATOR_REUSE_FACTOR * min(
                saved_activation_gb, temporary_buffer_gb
            )
        reuse = ALLOCATOR_REUSE_FACTOR
        return (
            reuse
            * temporary_buffer_gb
            * saved_activation_gb
            / (saved_activation_gb + temporary_buffer_gb)
        )

    def estimate_optimizer_update_workspace(self,
                                            parallel: ParallelConfig,
                                            sharding: ShardingConfig,
                                            parameter_memory_gb: float) -> float:
        """
        估算优化器 update 阶段的 FP32 master/update bucket 工作集。

        tensorwise offload 会在 update 时将当前参数 shard 拉回 GPU，
        以 FP32 bucket 进行更新；这一部分是框架运行时工作集，
        不属于长期常驻 optimizer state。
        """
        if (
            not sharding.tensorwise_offload
            or sharding.degree <= 1
            or sharding.get_optimizer_memory_factor() >= 1.0
        ):
            return 0.0
        return parameter_memory_gb * (4 / self.training.dtype_bytes)

    def _estimate_pp_effective_inflight_micro_batches(self,
                                                      parallel: ParallelConfig,
                                                      stage_id: int) -> float:
        """
        估算 1F1B 下每个 PP stage 的有效激活驻留数。

        旧模型直接使用 `pp - stage_id`，等价于把峰值当成纯 warmup 时刻，
        会系统性高估首段 stage 的 allocated，并低估末段 stage 的 reserved。

        实际峰值更接近前向堆积和后向回流交汇的 steady-state 过渡点。
        当 GAS 足够大时，可用 warmup/backward 两侧的平均占用近似：

            effective_inflight ~= (pp + 1) / 2
        """
        if parallel.pp <= 1:
            return 1.0
        gas = max(1, int(getattr(self.training, "gradient_accumulation_steps", 1)))
        steady_state_inflight = 0.5 * (float(parallel.pp) + 1.0)
        return min(float(gas), steady_state_inflight)

    def _scale_phase_pool(self,
                          total_pool_gb: float,
                          phase_live_gb: float,
                          max_phase_live_gb: float) -> float:
        if total_pool_gb <= 0.0 or phase_live_gb <= 0.0 or max_phase_live_gb <= 0.0:
            return 0.0
        return total_pool_gb * min(1.0, phase_live_gb / max_phase_live_gb)

    def _estimate_multi_node_fragmentation_gb(
        self,
        breakdown: MemoryBreakdown,
        recompute: RecomputeConfig,
    ) -> float:
        remaining_saved_activation_gb = max(
            0.0,
            breakdown.activation_saved_by_recompute_gb
            - breakdown.activation_buffer_pool_gb,
        )
        if remaining_saved_activation_gb <= 0.0:
            return 0.0

        method = self._normalize_recompute_method(recompute)
        if (
            recompute.granularity == RecomputeGranularity.FULL
            and method == "uniform"
        ):
            overlap_span_gb = max(
                breakdown.forward_backward_overlap_live_gb,
                breakdown.loss_overlap_live_gb,
                breakdown.optimizer_overlap_live_gb,
            )
            shape_sensitive_comm_gb = (
                breakdown.tp_comm_buffer_gb
                + breakdown.pp_comm_buffer_gb
                + breakdown.ep_comm_buffer_gb
                + breakdown.sharding_comm_buffer_gb * 0.5
            )
            uniform_recompute_carry_gb = (
                FULL_RECOMPUTE_REUSE_FACTOR
                * min(
                    remaining_saved_activation_gb,
                    max(
                        breakdown.peak_runtime_workspace_gb,
                        breakdown.activation_memory_gb,
                    ),
                )
            )
            uniform_fragmentation_span_gb = max(
                overlap_span_gb,
                breakdown.forward_backward_comm_live_gb,
                shape_sensitive_comm_gb + overlap_span_gb,
                uniform_recompute_carry_gb,
            )
            return min(
                remaining_saved_activation_gb,
                uniform_fragmentation_span_gb,
            )

        fragmentation_span_gb = max(
            breakdown.communication_buffer_gb,
            breakdown.forward_backward_comm_live_gb,
            breakdown.activation_memory_gb + breakdown.forward_backward_overlap_live_gb,
        )
        return min(remaining_saved_activation_gb, fragmentation_span_gb)

    def _estimate_multi_node_reserved_candidates(
        self,
        breakdown: MemoryBreakdown,
        recompute: RecomputeConfig,
    ) -> Dict[str, float]:
        live_static_gb = (
            breakdown.model_states_gb
            + breakdown.framework_overhead_gb
            + breakdown.temporary_buffer_gb
        )
        single_node_base_reserved_gb = (
            breakdown.allocated_memory_gb
            + breakdown.temporary_buffer_gb
            + breakdown.activation_buffer_pool_gb
        )
        post_step_base_gb = (
            breakdown.model_states_gb
            + breakdown.framework_overhead_gb
            + breakdown.temporary_buffer_gb
            + breakdown.activation_buffer_pool_gb
        )
        max_overlap_live_gb = max(
            breakdown.forward_backward_overlap_live_gb,
            breakdown.loss_overlap_live_gb,
            breakdown.optimizer_overlap_live_gb,
        )

        forward_backward_addon_gb = (
            breakdown.forward_backward_overlap_live_gb
            + self._scale_phase_pool(
                breakdown.overlap_runtime_pool_gb,
                breakdown.forward_backward_overlap_live_gb,
                max_overlap_live_gb,
            )
        )

        loss_addon_gb = (
            breakdown.loss_overlap_live_gb
            + self._scale_phase_pool(
                breakdown.overlap_runtime_pool_gb,
                breakdown.loss_overlap_live_gb,
                max_overlap_live_gb,
            )
        )

        optimizer_addon_gb = (
            breakdown.optimizer_overlap_live_gb
            + self._scale_phase_pool(
                breakdown.overlap_runtime_pool_gb,
                breakdown.optimizer_overlap_live_gb,
                max_overlap_live_gb,
            )
        )

        breakdown.communication_fragmentation_gb = (
            self._estimate_multi_node_fragmentation_gb(breakdown, recompute)
        )
        method = self._normalize_recompute_method(recompute)
        if (
            recompute.granularity == RecomputeGranularity.FULL
            and method == "uniform"
        ):
            post_step_addon_gb = (
                breakdown.communication_runtime_pool_gb
                + breakdown.overlap_runtime_pool_gb
                + breakdown.communication_fragmentation_gb
            )
        else:
            post_step_base_gb = max(
                post_step_base_gb,
                single_node_base_reserved_gb,
            )
            post_step_addon_gb = (
                breakdown.communication_runtime_pool_gb
                + breakdown.overlap_runtime_pool_gb
                + breakdown.communication_fragmentation_gb
            )

        return {
            "multi_node_forward_backward": (
                live_static_gb
                + breakdown.activation_memory_gb
                + forward_backward_addon_gb
            ),
            "multi_node_loss": (
                live_static_gb
                + breakdown.temporary_buffer_gb
                + loss_addon_gb
            ),
            "multi_node_optimizer": (
                live_static_gb
                + breakdown.optimizer_update_workspace_gb
                + optimizer_addon_gb
            ),
            "multi_node_post_step": post_step_base_gb + post_step_addon_gb,
        }

    def _estimate_memory_for_stage(
        self,
        parallel: ParallelConfig,
        sharding: ShardingConfig,
        recompute: RecomputeConfig,
        seq_len: int,
        mbs: int,
        stage_id: int,
    ) -> MemoryBreakdown:
        """
        估算单个 PP stage 的显存。

        Args:
            stage_id: PP stage 编号 (0 = first, PP-1 = last)
        """
        breakdown = MemoryBreakdown()
        breakdown.peak_stage_id = stage_id

        # ========== 模型状态 ==========
        breakdown.parameter_memory_gb = self.estimate_parameter_memory(
            parallel, sharding, stage_id
        )
        breakdown.optimizer_memory_gb = self.estimate_optimizer_memory(
            parallel, sharding, stage_id
        )
        breakdown.optimizer_update_workspace_gb = self.estimate_optimizer_update_workspace(
            parallel, sharding, breakdown.parameter_memory_gb
        )

        if sharding.uses_release_grads():
            param_count = self.estimate_parameter_count_per_gpu(parallel, stage_id)
            non_expert_params = (
                param_count["total_params"] - param_count["expert_params"]
            )
            breakdown.gradient_memory_gb = (
                non_expert_params * self.training.dtype_bytes
            ) / GIB
        else:
            breakdown.gradient_memory_gb = self.estimate_gradient_memory(
                parallel, sharding, stage_id
            )
        breakdown.master_grad_memory_gb = self.estimate_master_grad_memory(
            parallel, sharding, breakdown.gradient_memory_gb
        )
        breakdown.tensor_fusion_buffer_gb = self.estimate_tensor_fusion_buffer(
            parallel, sharding, stage_id
        )

        breakdown.communication_buffer_gb = self.estimate_communication_buffer(
            parallel, mbs, seq_len
        )
        if self._is_multi_node_prediction():
            comm_pools = self._estimate_comm_pools(
                parallel,
                sharding,
                recompute,
                seq_len,
                mbs,
                stage_id,
                breakdown.gradient_memory_gb,
            )
            breakdown.communication_buffer_gb = comm_pools["communication_buffer_gb"]
            breakdown.tp_comm_buffer_gb = comm_pools["tp_comm_buffer_gb"]
            breakdown.pp_comm_buffer_gb = comm_pools["pp_comm_buffer_gb"]
            breakdown.ep_comm_buffer_gb = comm_pools["ep_comm_buffer_gb"]
            breakdown.sharding_comm_buffer_gb = comm_pools["sharding_comm_buffer_gb"]
            breakdown.forward_backward_comm_live_gb = comm_pools[
                "forward_backward_comm_live_gb"
            ]
            breakdown.loss_comm_live_gb = comm_pools["loss_comm_live_gb"]
            breakdown.optimizer_comm_live_gb = comm_pools[
                "optimizer_comm_live_gb"
            ]
            breakdown.forward_backward_overlap_live_gb = comm_pools[
                "forward_backward_overlap_live_gb"
            ]
            breakdown.loss_overlap_live_gb = comm_pools[
                "loss_overlap_live_gb"
            ]
            breakdown.optimizer_overlap_live_gb = comm_pools[
                "optimizer_overlap_live_gb"
            ]
            breakdown.communication_runtime_pool_gb = comm_pools[
                "communication_runtime_pool_gb"
            ]
            breakdown.overlap_runtime_pool_gb = comm_pools["overlap_runtime_pool_gb"]

        # ========== 激活 ==========
        # _estimate_activation_workspace_bytes 返回 per-micro-batch 激活量
        # (此处已按 stage_id 逐 stage 精确计算)
        activation_bytes, activation_no_recompute_bytes = (
            self._estimate_activation_workspace_bytes(
                parallel, recompute, seq_len, mbs, stage_id
            )
        )

        # 1F1B 调度: 峰值更接近 steady-state 交汇点，而不是纯 warmup。
        pp_inflight = self._estimate_pp_effective_inflight_micro_batches(
            parallel, stage_id
        )
        breakdown.pp_inflight_micro_batches = pp_inflight

        activation_memory_bytes = activation_bytes * pp_inflight
        activation_no_recompute_total = activation_no_recompute_bytes * pp_inflight

        breakdown.activation_memory_gb = activation_memory_bytes / GIB
        activation_no_recompute_gb = activation_no_recompute_total / GIB
        breakdown.activation_no_recompute_gb = activation_no_recompute_gb
        breakdown.activation_saved_by_recompute_gb = max(
            0.0, activation_no_recompute_gb - breakdown.activation_memory_gb
        )

        # ========== Loss workspace ==========
        # PP > 1 时只有最后一个 stage 有 loss
        breakdown.temporary_buffer_gb = self.estimate_loss_workspace(
            parallel, seq_len, mbs, stage_id
        )

        # ========== Buffer pool ==========
        breakdown.activation_buffer_pool_gb = self.estimate_activation_buffer_pool(
            recompute,
            breakdown.activation_memory_gb,
            activation_no_recompute_gb,
            breakdown.temporary_buffer_gb,
            breakdown.optimizer_update_workspace_gb,
        )

        # ========== Peak runtime workspace ==========
        peak_candidates = {
            "activation": breakdown.activation_memory_gb,
            "loss": breakdown.temporary_buffer_gb,
            "optimizer_update": breakdown.optimizer_update_workspace_gb,
        }
        peak_source, peak_value = max(
            peak_candidates.items(),
            key=lambda item: (
                item[1],
                1 if item[0] == "optimizer_update" else 0,
                1 if item[0] == "activation" else 0,
            ),
        )
        breakdown.peak_runtime_workspace_source = peak_source
        breakdown.peak_runtime_workspace_gb = peak_value
        reserved_value = (
            breakdown.allocated_memory_gb
            + breakdown.temporary_buffer_gb
            + breakdown.activation_buffer_pool_gb
        )
        reserved_source = "single_node_base"
        if self._is_multi_node_prediction():
            reserved_candidates = self._estimate_multi_node_reserved_candidates(
                breakdown, recompute
            )
            breakdown.reserved_candidate_forward_backward_gb = (
                reserved_candidates["multi_node_forward_backward"]
            )
            breakdown.reserved_candidate_loss_gb = (
                reserved_candidates["multi_node_loss"]
            )
            breakdown.reserved_candidate_optimizer_gb = (
                reserved_candidates["multi_node_optimizer"]
            )
            breakdown.reserved_candidate_post_step_gb = (
                reserved_candidates["multi_node_post_step"]
            )
            reserved_source, reserved_value = max(
                reserved_candidates.items(),
                key=lambda item: item[1],
            )
        breakdown.reserved_peak_source = reserved_source
        breakdown.reserved_peak_memory_gb = reserved_value

        return breakdown

    def estimate_memory(self, parallel: ParallelConfig,
                        sharding: ShardingConfig = None,
                        recompute: RecomputeConfig = None,
                        max_seq_len: int = None,
                        micro_batch_size: int = None) -> MemoryBreakdown:
        """
        完整显存估算
        
        PP > 1 时，对每个 stage 分别计算显存，返回 reserved 最大的 stage 的 breakdown。
        PP = 1 时，等价于单 stage 计算。

        Args:
            parallel: 并行配置
            sharding: Sharding 配置
            recompute: 重计算配置
            max_seq_len: 最大序列长度 (用于激活显存估算)
            micro_batch_size: 每卡 batch size (用于激活和临时缓冲计算)
        
        Returns:
            MemoryBreakdown: 显存最大的 stage 的详细分解
        """
        if sharding is None:
            sharding = ShardingConfig(
                stage=parallel.sharding_stage,
                degree=parallel.effective_sharding_degree,
            )
        
        if recompute is None:
            recompute = RecomputeConfig(
                granularity=self.training.recompute_config,
                method=self.training.recompute_method,
                num_layers=self.training.recompute_num_layers,
                modules=tuple(self.training.recompute_modules),
            )
        
        seq_len = max_seq_len if max_seq_len is not None else self.training.sequence_length
        mbs = micro_batch_size if micro_batch_size is not None else self.training.micro_batch_size

        # 遍历每个 PP stage，返回 reserved 最大的 breakdown
        worst_breakdown = None
        max_allocated_breakdown = None
        for sid in range(max(1, parallel.pp)):
            breakdown = self._estimate_memory_for_stage(
                parallel, sharding, recompute, seq_len, mbs, sid
            )
            if (
                max_allocated_breakdown is None
                or breakdown.allocated_memory_gb > max_allocated_breakdown.allocated_memory_gb
            ):
                max_allocated_breakdown = breakdown
            if worst_breakdown is None or breakdown.reserved_memory_gb > worst_breakdown.reserved_memory_gb:
                worst_breakdown = breakdown

        if worst_breakdown is not None and max_allocated_breakdown is not None:
            worst_breakdown.allocated_peak_memory_gb = (
                max_allocated_breakdown.allocated_memory_gb
            )
            worst_breakdown.allocated_peak_stage_id = (
                max_allocated_breakdown.peak_stage_id
            )

        return worst_breakdown

    def estimate_memory_per_stage(self, parallel: ParallelConfig,
                                   sharding: ShardingConfig = None,
                                   per_stage_recompute: "list[RecomputeConfig]" = None,
                                   max_seq_len: int = None,
                                   micro_batch_size: int = None) -> MemoryBreakdown:
        """
        Per-stage 显存估算：每个 PP stage 使用独立的 RecomputeConfig。

        与 estimate_memory 的唯一区别：接受一个 RecomputeConfig 列表
        （长度 = PP stages），每个 stage 使用对应的 recompute 配置。

        Args:
            parallel: 并行配置
            sharding: Sharding 配置
            per_stage_recompute: 每个 stage 的重计算配置列表，长度应 == parallel.pp。
                                 如果为 None 或长度不匹配，退回到默认行为。
            max_seq_len: 最大序列长度
            micro_batch_size: micro batch size

        Returns:
            MemoryBreakdown: reserved 最大的 stage 的详细分解
        """
        num_stages = max(1, parallel.pp)

        # 如果没有提供 per-stage configs 或长度不匹配，退回普通方法
        if per_stage_recompute is None or len(per_stage_recompute) != num_stages:
            default_recompute = None
            if per_stage_recompute and len(per_stage_recompute) > 0:
                # 至少用第一个
                default_recompute = per_stage_recompute[0]
            return self.estimate_memory(
                parallel, sharding, default_recompute, max_seq_len, micro_batch_size
            )

        if sharding is None:
            sharding = ShardingConfig(
                stage=parallel.sharding_stage,
                degree=parallel.effective_sharding_degree,
            )

        seq_len = max_seq_len if max_seq_len is not None else self.training.sequence_length
        mbs = micro_batch_size if micro_batch_size is not None else self.training.micro_batch_size

        worst_breakdown = None
        max_allocated_breakdown = None
        for sid in range(num_stages):
            recompute = per_stage_recompute[sid]
            breakdown = self._estimate_memory_for_stage(
                parallel, sharding, recompute, seq_len, mbs, sid
            )
            if (
                max_allocated_breakdown is None
                or breakdown.allocated_memory_gb > max_allocated_breakdown.allocated_memory_gb
            ):
                max_allocated_breakdown = breakdown
            if worst_breakdown is None or breakdown.reserved_memory_gb > worst_breakdown.reserved_memory_gb:
                worst_breakdown = breakdown

        if worst_breakdown is not None and max_allocated_breakdown is not None:
            worst_breakdown.allocated_peak_memory_gb = (
                max_allocated_breakdown.allocated_memory_gb
            )
            worst_breakdown.allocated_peak_stage_id = (
                max_allocated_breakdown.peak_stage_id
            )

        return worst_breakdown

    def fits_memory(self, parallel: ParallelConfig,
                    gpu_memory_gb: float,
                    sharding: ShardingConfig = None,
                    recompute: RecomputeConfig = None) -> Tuple[bool, MemoryBreakdown]:
        """
        检查是否能放入 GPU 显存
        
        Returns:
            (fits, breakdown)
        """
        breakdown = self.estimate_memory(parallel, sharding, recompute)
        fits = breakdown.total_memory_gb <= gpu_memory_gb
        
        return fits, breakdown


# --- merged stage-local wrapper logic ---
from typing import Any, Dict, List, Optional, Sequence
from ..config import RecomputeGranularity
from .recompute_stage_sim import (
    build_stage_plans_from_recompute_configs,
    build_uniform_stage_plans,
    plans_to_dicts,
)

class MemoryModel(_BaseMemoryModel):
    """
    Drop-in replacement for the original MemoryModel.

    Strategy:
    - keep the original repository's detailed memory formulas;
    - for global `estimate_memory`, when PP>1 and recompute is active, route the
      computation through the repository's per-stage memory path so block/full
      recompute semantics are stage-local instead of implicitly global;
    - attach `stage_recompute_detail` for debugging and downstream analysis.
    """

    def estimate_memory(
        self,
        parallel,
        sharding: ShardingConfig = None,
        recompute: RecomputeConfig = None,
        max_seq_len: int = None,
        micro_batch_size: int = None,
        **kwargs,
    ) -> MemoryBreakdown:
        legacy_sharding = kwargs.pop("sharding_config", None)
        legacy_recompute = kwargs.pop("recompute_config", None)
        if sharding is None:
            sharding = legacy_sharding
        if recompute is None:
            recompute = legacy_recompute

        effective_recompute = recompute
        if effective_recompute is None:
            effective_recompute = RecomputeConfig(
                granularity=self.training.recompute_config,
                method=self.training.recompute_method,
                num_layers=self.training.recompute_num_layers,
                modules=tuple(self.training.recompute_modules),
            )
        stage_layers = self._resolve_stage_layer_counts(parallel)
        plans = build_uniform_stage_plans(
            stage_layers,
            getattr(effective_recompute, "granularity", RecomputeGranularity.NONE),
            getattr(effective_recompute, "method", "uniform"),
            getattr(effective_recompute, "num_layers", 1),
        )

        # When pipeline > 1 and recompute is active, force the calculation
        # through the per-stage interface so stage-local block semantics are not
        # lost.
        pp = max(1, int(getattr(parallel, "pp", 1) or 1))
        if pp > 1 and any(plan.granularity != "none" for plan in plans):
            per_stage_rc = [
                self._clone_recompute_config(effective_recompute)
                for _ in range(pp)
            ]
            breakdown = super().estimate_memory_per_stage(
                parallel,
                sharding,
                per_stage_rc,
                max_seq_len,
                micro_batch_size,
                **kwargs,
            )
        else:
            breakdown = super().estimate_memory(
                parallel,
                sharding,
                recompute,
                max_seq_len,
                micro_batch_size,
                **kwargs,
            )

        self._attach_stage_recompute_detail(breakdown, stage_layers, plans)
        return breakdown

    def estimate_memory_per_stage(
        self,
        parallel,
        sharding: ShardingConfig = None,
        per_stage_recompute: Sequence[RecomputeConfig] = None,
        max_seq_len: int = None,
        micro_batch_size: int = None,
        **kwargs,
    ) -> MemoryBreakdown:
        legacy_sharding = kwargs.pop("sharding_config", None)
        legacy_per_stage_recompute = kwargs.pop(
            "per_stage_recompute_configs",
            None,
        )
        if sharding is None:
            sharding = legacy_sharding
        if per_stage_recompute is None:
            per_stage_recompute = legacy_per_stage_recompute

        breakdown = super().estimate_memory_per_stage(
            parallel,
            sharding,
            per_stage_recompute,
            max_seq_len,
            micro_batch_size,
            **kwargs,
        )
        stage_layers = self._resolve_stage_layer_counts(parallel)
        if per_stage_recompute is not None:
            plans = build_stage_plans_from_recompute_configs(
                stage_layers,
                per_stage_recompute,
            )
        else:
            effective_recompute = RecomputeConfig(
                granularity=self.training.recompute_config,
                method=self.training.recompute_method,
                num_layers=self.training.recompute_num_layers,
                modules=tuple(self.training.recompute_modules),
            )
            plans = build_uniform_stage_plans(
                stage_layers,
                getattr(effective_recompute, "granularity", RecomputeGranularity.NONE),
                getattr(effective_recompute, "method", "uniform"),
                getattr(effective_recompute, "num_layers", 1),
            )
        self._attach_stage_recompute_detail(breakdown, stage_layers, plans)
        return breakdown

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_stage_layer_counts(self, parallel) -> List[int]:
        if getattr(parallel, "stage_layer_counts", None):
            return [max(0, int(v)) for v in getattr(parallel, "stage_layer_counts")]

        pp = max(1, int(getattr(parallel, "pp", 1) or 1))
        total_layers = int(
            getattr(self.model, "num_hidden_layers", 0)
            or getattr(self.model, "num_layers", 0)
            or 0
        )
        if total_layers <= 0:
            return [1] * pp
        base, rem = divmod(total_layers, pp)
        return [base + (1 if i < rem else 0) for i in range(pp)]

    def _clone_recompute_config(self, rc: RecomputeConfig) -> RecomputeConfig:
        return RecomputeConfig(
            granularity=getattr(rc, "granularity", RecomputeGranularity.NONE),
            method=getattr(rc, "method", "uniform"),
            num_layers=int(getattr(rc, "num_layers", 1) or 1),
            modules=tuple(getattr(rc, "modules", tuple()) or tuple()),
        )

    def _attach_stage_recompute_detail(
        self,
        breakdown: MemoryBreakdown,
        stage_layers: Sequence[int],
        plans,
    ) -> None:
        try:
            breakdown.stage_layer_counts = list(stage_layers)
            breakdown.stage_recompute_detail = plans_to_dicts(plans)
        except Exception:
            # MemoryBreakdown is often a dataclass but keep this defensive.
            pass
        # Some downstream code may serialize through to_dict(); store an extra
        # dict if the object exposes a generic metadata slot.
        if hasattr(breakdown, "extra") and isinstance(getattr(breakdown, "extra"), dict):
            breakdown.extra["stage_layer_counts"] = list(stage_layers)
            breakdown.extra["stage_recompute_detail"] = plans_to_dicts(plans)


__all__ = [
    "MemoryModel",
    "MemoryBreakdown",
    "ShardingConfig",
    "RecomputeConfig",
]
