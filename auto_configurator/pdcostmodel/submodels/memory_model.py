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
from typing import Dict, Optional, Tuple
from enum import Enum

from ..config import (
    ModelConfig, ParallelConfig, TrainingConfig,
    ShardingStage, RecomputeGranularity
)


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
        
        校准说明：
        - tensorwise_offload 配置的 ratio 表示 offload 到 CPU 的比例
        - 但实际测量 allocated 是峰值，此时优化器状态已被加载回 GPU
        - 根据真实数据校准，实际 GPU 上保留约 63% 的优化器状态
        """
        if self.tensorwise_offload:
            # 关键修复：offload 只在 Sharding degree > 1 时有效
            if self.degree <= 1:
                # DP=1 时，Sharding 不工作，offload 也不工作
                return 1.0
            # 校准后的 offload 效果
            # 真实数据显示：配置 95% offload，实际 allocated 约为全量的 63%
            # 这是因为 tensorwise 是逐个加载，测量时部分已加载
            CALIBRATED_OFFLOAD_RATIO = 0.37  # 实际 offload 到 CPU 的比例
            return 1.0 - CALIBRATED_OFFLOAD_RATIO
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
    method: str = "uniform"  # "uniform", "block"
    num_layers: int = 1  # 重计算间隔
    
    def get_memory_reduction_factor(self, total_layers: int = 48) -> float:
        """
        获取显存降低因子
        
        PaddleFormers recompute 策略:
        - recompute_granularity: full + recompute_method: uniform + recompute_num_layers: N
        - 每 N 层保存一次 checkpoint (边界激活)
        - 中间 N-1 层的激活在反向时重新计算
        """
        if self.granularity == RecomputeGranularity.NONE:
            return 1.0
        
        if self.granularity == RecomputeGranularity.SELECTIVE:
            # 选择性重计算: 只保存 attention 输出
            return 0.6
        
        if self.granularity == RecomputeGranularity.FULL:
            if self.method == "uniform":
                n = max(1, self.num_layers)
                if n == 1:
                    # 每层保存边界，层内中间激活重算
                    return 0.15
                else:
                    # 每 N 层保存一次边界
                    return min(1.0, (1.0 / n) * 0.15 + 0.05)
            else:
                return 0.15 / max(1, self.num_layers)
        
        return 1.0
    
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
    """显存分解结果"""
    # 主要组成 (GB)
    parameter_memory_gb: float = 0.0
    gradient_memory_gb: float = 0.0
    optimizer_memory_gb: float = 0.0
    activation_memory_gb: float = 0.0
    
    # 其他 (GB)
    communication_buffer_gb: float = 0.0
    temporary_buffer_gb: float = 0.0
    framework_overhead_gb: float = 2.0  # CUDA/Paddle 框架基础开销
    
    # PaddleFormers 框架预留的激活缓冲池 (reserved - allocated)
    activation_buffer_pool_gb: float = 0.0
    
    @property
    def allocated_memory_gb(self) -> float:
        """实际分配显存 (allocated)"""
        return (
            self.parameter_memory_gb +
            self.gradient_memory_gb +
            self.optimizer_memory_gb +
            self.activation_memory_gb +
            self.communication_buffer_gb +
            self.temporary_buffer_gb +
            self.framework_overhead_gb
        )
    
    @property
    def reserved_memory_gb(self) -> float:
        """预留显存 (reserved) = allocated + 框架激活缓冲池"""
        return self.allocated_memory_gb + self.activation_buffer_pool_gb
    
    @property
    def total_memory_gb(self) -> float:
        """总显存占用 (等于 reserved)"""
        return self.reserved_memory_gb
    
    @property
    def model_states_gb(self) -> float:
        """模型状态显存 (参数 + 梯度 + 优化器)"""
        return (
            self.parameter_memory_gb +
            self.gradient_memory_gb +
            self.optimizer_memory_gb
        )
    
    def to_dict(self) -> Dict:
        return {
            "parameter_memory_gb": round(self.parameter_memory_gb, 3),
            "gradient_memory_gb": round(self.gradient_memory_gb, 3),
            "optimizer_memory_gb": round(self.optimizer_memory_gb, 3),
            "activation_memory_gb": round(self.activation_memory_gb, 3),
            "communication_buffer_gb": round(self.communication_buffer_gb, 3),
            "temporary_buffer_gb": round(self.temporary_buffer_gb, 3),
            "reserved_memory_gb": round(self.reserved_memory_gb, 3),
            "model_states_gb": round(self.model_states_gb, 3),
            "total_memory_gb": round(self.total_memory_gb, 3),
        }
    
    def __str__(self) -> str:
        return (
            f"Memory Breakdown:\n"
            f"  Parameters: {self.parameter_memory_gb:.2f} GB\n"
            f"  Gradients:  {self.gradient_memory_gb:.2f} GB\n"
            f"  Optimizer:  {self.optimizer_memory_gb:.2f} GB\n"
            f"  Activation: {self.activation_memory_gb:.2f} GB\n"
            f"  Comm Buf:   {self.communication_buffer_gb:.2f} GB\n"
            f"  Temp Buf:   {self.temporary_buffer_gb:.2f} GB\n"
            f"  Reserved:   {self.reserved_memory_gb:.2f} GB\n"
            f"  ─────────────────────\n"
            f"  Total:      {self.total_memory_gb:.2f} GB"
        )


class MemoryModel:
    """
    显存模型
    
    精确预测 PaddleFormers 分布式训练的显存占用
    """
    
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        self.model = model_config
        self.training = training_config
    
    def estimate_parameter_count_per_gpu(self, parallel: ParallelConfig) -> Dict[str, int]:
        """
        估算每个 GPU 上的参数数量
        
        考虑 TP/PP/EP 切分
        """
        h = self.model.hidden_size
        ffn = self.model.intermediate_size
        moe_ffn = self.model.moe_intermediate_size
        v = self.model.vocab_size
        num_layers = self.model.num_hidden_layers
        
        tp = parallel.tp
        pp = parallel.pp
        ep = parallel.ep
        
        # 每个 PP stage 的层数
        layers_per_stage = num_layers // pp
        
        # ========== Attention 参数 ==========
        # Q: h * h, K: h * kv_size, V: h * kv_size, O: h * h
        # 被 TP 切分
        kv_size = self.model.num_key_value_heads * self.model.head_dim
        attention_params = (h * h + 2 * h * kv_size + h * h) // tp
        
        # ========== LayerNorm 参数 ==========
        # 每层 2 个 LayerNorm: 2 * 2 * h
        # 不切分 (通常复制)
        layernorm_params = 4 * h
        
        # ========== Dense MLP 参数 ==========
        # Gate + Up + Down: 3 * h * ffn (SwiGLU)
        # 被 TP 切分
        dense_mlp_params = 3 * h * ffn // tp
        
        # ========== MoE 参数 ==========
        # Router: h * num_experts (不切分)
        router_params = h * self.model.num_experts
        
        # Expert MLP: 3 * h * moe_ffn * (num_experts / ep) / tp
        experts_per_gpu = self.model.num_experts // ep
        expert_params = 3 * h * moe_ffn * experts_per_gpu // tp
        
        # ========== Embedding 参数 ==========
        # 被 TP 切分，只在第一个/最后一个 PP stage
        if pp == 1:
            embedding_params = v * h // tp * 2  # input + output
        else:
            # 分布在第一个和最后一个 stage
            embedding_params = v * h // tp
        
        # ========== 计算每层参数 ==========
        # Dense 层: attention + layernorm + dense_mlp
        dense_layer_params = attention_params + layernorm_params + dense_mlp_params
        
        # MoE 层: attention + layernorm + router + experts
        moe_layer_params = attention_params + layernorm_params + router_params + expert_params
        
        # ========== 汇总 (per PP stage) ==========
        dense_layers_per_stage = self.model.num_dense_layers // pp
        moe_layers_per_stage = self.model.num_moe_layers // pp
        
        total_dense_params = dense_layer_params * dense_layers_per_stage
        total_moe_params = moe_layer_params * moe_layers_per_stage
        total_params = total_dense_params + total_moe_params + embedding_params
        
        return {
            "attention_params": attention_params * layers_per_stage,
            "layernorm_params": layernorm_params * layers_per_stage,
            "dense_mlp_params": dense_mlp_params * dense_layers_per_stage,
            "router_params": router_params * moe_layers_per_stage,
            "expert_params": expert_params * moe_layers_per_stage,
            "embedding_params": embedding_params,
            "total_params": total_params,
        }
    
    def estimate_parameter_memory(self, parallel: ParallelConfig,
                                  sharding: ShardingConfig) -> float:
        """
        估算参数显存 (GB)
        
        关键点：专家参数和非专家参数的 Sharding 处理不同
        - 专家参数：已被 EP 切分，不受 Sharding 切分（专家是局部的）
        - 非专家参数：受 Sharding split_param 切分
        """
        param_count = self.estimate_parameter_count_per_gpu(parallel)
        
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
                                 sharding: ShardingConfig) -> float:
        """
        估算梯度显存 (GB)
        
        关键点：专家参数和非专家参数的 Sharding 处理不同
        - 专家参数：已被 EP 切分，不受 Sharding 切分
        - 非专家参数：受 Sharding split_param 切分
        """
        param_count = self.estimate_parameter_count_per_gpu(parallel)
        
        # 分离专家参数和非专家参数
        expert_params = param_count["expert_params"]
        non_expert_params = param_count["total_params"] - expert_params
        
        # ZeRO 梯度分片因子（只对非专家参数生效）
        sharding_factor = sharding.get_grad_sharding_factor()
        
        # 专家梯度显存（不受 Sharding 切分）
        expert_grad_bytes = expert_params * self.training.dtype_bytes
        
        # 非专家梯度显存（受 Sharding 切分）
        non_expert_grad_bytes = non_expert_params * self.training.dtype_bytes / sharding_factor
        
        total_grad_bytes = expert_grad_bytes + non_expert_grad_bytes
        
        return total_grad_bytes / (1024 ** 3)
    
    def estimate_optimizer_memory(self, parallel: ParallelConfig,
                                  sharding: ShardingConfig) -> float:
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
        param_count = self.estimate_parameter_count_per_gpu(parallel)
        
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
        
        # Master weight (如果 amp_master_grad)
        if self.training.amp_master_grad and self.training.dtype != "float32":
            # 专家 master weight 也 offload
            expert_master_bytes = expert_params * 4 * offload_factor
            non_expert_master_bytes = non_expert_params * 4 / sharding_factor * offload_factor
            optimizer_bytes += expert_master_bytes + non_expert_master_bytes
        
        return optimizer_bytes / (1024 ** 3)
    def estimate_activation_memory(self, parallel: ParallelConfig,
                                   recompute: RecomputeConfig,
                                   seq_len: int = None,
                                   micro_batch_size: int = None) -> float:
        """
        估算激活值显存 (GB)
        
        激活值是前向过程中需要保存用于后向的中间结果。
        受影响因素:
        - Sequence length (max_seq_len)
        - Batch size
        - 模型隐藏维度
        - Recompute 策略
        - 并行策略 (TP 切分激活)
        
        Args:
            parallel: 并行配置
            recompute: 重计算配置
            seq_len: 序列长度 (用于激活显存估算，通常使用 max_seq_len)
            micro_batch_size: 每卡 batch size
        """
        micro_bsz = micro_batch_size if micro_batch_size is not None else self.training.micro_batch_size
        seq_len = seq_len if seq_len is not None else self.training.sequence_length
        h = self.model.hidden_size
        num_layers = self.model.num_hidden_layers
        
        tp = parallel.tp
        pp = parallel.pp
        cp = parallel.cp
        
        # 每个 PP stage 的层数
        layers_per_stage = num_layers // pp
        
        # 序列长度考虑 CP
        effective_seq_len = seq_len // cp
        
        # ========== 每层激活估算 ==========
        # Attention 激活:
        # - Q, K, V: 3 * bsz * seq * h
        # - Attention scores: bsz * num_heads * seq * seq (FlashAttention 优化后大幅减少)
        # - Output: bsz * seq * h
        num_heads = self.model.num_attention_heads
        
        # FlashAttention 不保存完整 attention scores，只保存部分中间状态
        # 估算约 2 * bsz * num_heads * seq 的额外显存
        attention_activation = (
            3 * micro_bsz * effective_seq_len * h +  # Q, K, V
            2 * micro_bsz * num_heads * effective_seq_len +  # FA 中间状态
            micro_bsz * effective_seq_len * h  # output
        )
        
        # MLP 激活:
        # - Gate: bsz * seq * ffn
        # - Up: bsz * seq * ffn  
        # - Down input: bsz * seq * ffn
        ffn = self.model.intermediate_size
        mlp_activation = 3 * micro_bsz * effective_seq_len * ffn
        
        # MoE 层额外激活:
        # - Router logits: bsz * seq * num_experts
        # - Dispatch indices: bsz * seq * topk
        # - Expert outputs: bsz * seq * h * topk
        moe_extra_activation = (
            micro_bsz * effective_seq_len * self.model.num_experts +
            micro_bsz * effective_seq_len * self.model.num_experts_per_tok * 2 +
            micro_bsz * effective_seq_len * h * self.model.num_experts_per_tok
        )
        
        # TP 切分激活
        if tp > 1:
            attention_activation = attention_activation // tp
            mlp_activation = mlp_activation // tp
        
        # SP 进一步切分
        if parallel.sp and tp > 1:
            # LayerNorm 和某些激活在序列维度切分
            pass  # 简化处理
        
        # ========== 总激活显存 ==========
        dense_layers_per_stage = self.model.num_dense_layers // pp
        moe_layers_per_stage = self.model.num_moe_layers // pp
        
        dense_activation = (attention_activation + mlp_activation) * dense_layers_per_stage
        moe_activation = (attention_activation + mlp_activation + moe_extra_activation) * moe_layers_per_stage
        
        total_activation = dense_activation + moe_activation
        
        # 数据类型转换
        activation_bytes = total_activation * self.training.dtype_bytes
        
        # Recompute 降低因子
        recompute_factor = recompute.get_memory_reduction_factor(num_layers)
        
        activation_bytes = activation_bytes * recompute_factor
        
        return activation_bytes / (1024 ** 3)
    
    def estimate_communication_buffer(self, parallel: ParallelConfig,
                                      micro_batch_size: int = None) -> float:
        """
        估算通信缓冲区显存 (GB)
        
        包括:
        - TP AllReduce buffer
        - PP Send/Recv buffer  
        - DP AllReduce/ReduceScatter buffer
        - EP AllToAll buffer
        """
        micro_bsz = micro_batch_size if micro_batch_size is not None else self.training.micro_batch_size
        seq_len = self.training.sequence_length
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
            # Dispatch + Combine 两次 A2A
            topk = self.model.num_experts_per_tok
            ep_buffer = micro_bsz * seq_len * h * topk * self.training.dtype_bytes
            buffer_bytes += ep_buffer * 2
        
        # SP AllGather 缓冲区
        if parallel.sp and parallel.tp > 1:
            sp_buffer = micro_bsz * seq_len * h * self.training.dtype_bytes
            buffer_bytes += sp_buffer
        
        return buffer_bytes / (1024 ** 3)
    
    def estimate_activation_buffer_pool(self, max_seq_len: int) -> float:
        """
        估算 PaddleFormers 框架预留的激活缓冲池 (GB)
        
        PaddleFormers 框架会根据 max_seq_len 预分配一个激活缓冲池
        这部分显存在 reserved 中体现，但不在 allocated 中
        
        根据真实数据拟合:
        - seq <= 4096: 基础缓冲池 ~5.8 GB
        - seq > 4096: 每增加 4096 tokens，增加 ~9 GB
        - 8192 附近有额外阈值效应
        """
        # 基础缓冲池 (seq <= 4096)
        BASE_BUFFER_GB = 5.8
        
        if max_seq_len <= 4096:
            return BASE_BUFFER_GB
        
        # 超过 4096 的额外缓冲
        # 8192: +8.8 GB = 14.6 GB total
        # 8194: +13.4 GB = 19.2 GB total (有额外跳变)
        extra_tokens = max_seq_len - 4096
        
        # 基于 8192 数据点校准
        # (8192 - 4096) = 4096 extra tokens -> 8.8 GB extra
        BYTES_PER_EXTRA_TOKEN = 8.8 * (1024**3) / 4096  # ~2.25 MB/token
        
        extra_buffer = extra_tokens * BYTES_PER_EXTRA_TOKEN / (1024**3)
        
        # 8192 阈值效应: 如果超过 8192，有额外的跳变
        if max_seq_len > 8192:
            # 每超过 8192 一点，额外增加约 2.3 GB per token (非线性)
            # 这可能是框架在 8192 边界有特殊的内存池分配
            extra_beyond_8192 = max_seq_len - 8192
            extra_buffer += extra_beyond_8192 * 2.3 * (1024**3) / 1 / (1024**3)  # 简化
        
        return BASE_BUFFER_GB + extra_buffer
    
    def estimate_memory(self, parallel: ParallelConfig,
                        sharding: ShardingConfig = None,
                        recompute: RecomputeConfig = None,
                        max_seq_len: int = None,
                        micro_batch_size: int = None) -> MemoryBreakdown:
        """
        完整显存估算
        
        Args:
            parallel: 并行配置
            sharding: Sharding 配置
            recompute: 重计算配置
            max_seq_len: 最大序列长度 (用于激活显存估算)
            micro_batch_size: 每卡 batch size (用于激活和临时缓冲计算)
        
        Returns:
            MemoryBreakdown: 详细的显存分解，包含 allocated 和 reserved
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
            )
        
        seq_len = max_seq_len if max_seq_len is not None else self.training.sequence_length
        mbs = micro_batch_size if micro_batch_size is not None else self.training.micro_batch_size
        
        breakdown = MemoryBreakdown()
        
        breakdown.parameter_memory_gb = self.estimate_parameter_memory(parallel, sharding)
        breakdown.optimizer_memory_gb = self.estimate_optimizer_memory(parallel, sharding)
        breakdown.activation_memory_gb = self.estimate_activation_memory(parallel, recompute, seq_len, mbs)
        breakdown.communication_buffer_gb = self.estimate_communication_buffer(parallel, mbs)
        
        # 梯度显存处理
        # sd_release_grads: 每次迭代后释放梯度，梯度在下一次反向传播时重新创建
        # 此时峰值显存 = max(激活, 梯度) 而非 激活 + 梯度
        gradient_memory = self.estimate_gradient_memory(parallel, sharding)
        
        if sharding.uses_release_grads():
            # 梯度和激活取 max，梯度在反向时创建，激活可以部分释放
            # 但实际上反向传播时两者会同时存在一部分
            # 保守估计：梯度 × 0.3 (因为是逐层计算，不需要完整保存)
            breakdown.gradient_memory_gb = gradient_memory * 0.3
        else:
            breakdown.gradient_memory_gb = gradient_memory
        
        # 临时缓冲区
        # 主要包含 logits FP32 转换（计算 loss 时 logits 从 BF16 转为 FP32）
        # logits_fp32 = mbs × seq × vocab × 4 bytes
        # 
        # 重要修正 (2026-02-27):
        # 实测发现 cross_entropy_with_softmax 需要分配完整 logits 空间
        # OOM 错误: "Cannot allocate 18.546875GB" = 8 * 4096 * 151936 * 4 / 1024^3
        # 说明框架实际未启用 chunked cross-entropy，需要完整 logits
        # 
        # 此外 softmax 输出也需要同等大小的空间
        # 总计: logits (FP32) + softmax_output (FP32) ≈ 2x
        LOGITS_FACTOR = 2.0  # logits + softmax output
        logits_fp32_gb = (
            mbs * 
            seq_len * 
            self.model.vocab_size * 
            4  # float32 = 4 bytes
        ) / (1024 ** 3) * LOGITS_FACTOR
        
        # 其他临时缓冲（约为激活的 5%）
        other_temp_buffer = breakdown.activation_memory_gb * 0.05
        
        breakdown.temporary_buffer_gb = logits_fp32_gb + other_temp_buffer
        
        # PaddleFormers 框架预留的激活缓冲池
        breakdown.activation_buffer_pool_gb = self.estimate_activation_buffer_pool(seq_len)
        
        return breakdown
    
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