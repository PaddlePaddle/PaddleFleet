#!/usr/bin/env python3
"""
计算模型模块 - 预测 PaddleFormers 训练的计算时间

核心功能:
1. 分层计算时间建模 (Attention, MLP, MoE)
2. 基于 FLOPs 和硬件效率的预测
3. 流水线气泡时间预测
4. 支持 Recompute 开销估算
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional
from enum import Enum

from .config import ModelConfig, ParallelConfig, TrainingConfig, HardwareConfig


class LayerType(Enum):
    """层类型"""
    ATTENTION = "attention"
    DENSE_MLP = "dense_mlp"
    MOE_ROUTER = "moe_router"
    MOE_EXPERT = "moe_expert"
    LAYERNORM = "layernorm"
    EMBEDDING = "embedding"


@dataclass
class LayerProfile:
    """层计算 Profile"""
    layer_type: LayerType
    flops_per_token: int = 0  # 每 token 的 FLOPs
    efficiency: float = 0.5  # 硬件利用效率
    
    def estimate_time_ms(self, tokens: int, peak_tflops: float, 
                         parallel_factor: int = 1) -> float:
        """
        估算计算时间
        
        Args:
            tokens: token 数量
            peak_tflops: 峰值算力 (TFLOPS)
            parallel_factor: 并行切分因子 (TP)
        """
        # 总 FLOPs (考虑 FMA = 2 ops)
        total_flops = self.flops_per_token * tokens * 2
        
        # 并行切分
        parallel_flops = total_flops / parallel_factor
        
        # 计算时间 (ms)
        time_sec = parallel_flops / (peak_tflops * 1e12 * self.efficiency)
        
        return time_sec * 1000


class ComputeModel:
    """
    计算模型
    
    基于 FLOPs 和硬件效率预测计算时间
    """
    
    def __init__(self, model_config: ModelConfig, 
                 hardware_config: HardwareConfig,
                 training_config: TrainingConfig):
        self.model = model_config
        self.hardware = hardware_config
        self.training = training_config
        
        # 初始化层 Profile
        self.layer_profiles: Dict[LayerType, LayerProfile] = {}
        self._init_layer_profiles()
    
    def _init_layer_profiles(self):
        """
        初始化层计算 Profile
        
        效率因子说明:
        - 理论峰值 TFLOPS 很少能达到
        - 小 batch size 时内存带宽成为瓶颈
        - MoE 模型有额外的路由和通信开销
        - PaddlePaddle 动态图框架有开销
        """
        h = self.model.hidden_size
        ffn = self.model.intermediate_size
        moe_ffn = self.model.moe_intermediate_size
        num_heads = self.model.num_attention_heads
        kv_heads = self.model.num_key_value_heads
        head_dim = self.model.head_dim
        num_experts = self.model.num_experts
        
        # ========== Attention FLOPs ==========
        # Q projection: bsz * seq * h * h
        # K projection: bsz * seq * h * kv_size
        # V projection: bsz * seq * h * kv_size
        # QK^T: bsz * num_heads * seq * seq * head_dim
        # Softmax: bsz * num_heads * seq * seq (约 5 ops per element)
        # Score * V: bsz * num_heads * seq * seq * head_dim
        # O projection: bsz * seq * h * h
        
        kv_size = kv_heads * head_dim
        attention_flops = (
            h * h +           # Q proj
            h * kv_size +     # K proj
            h * kv_size +     # V proj
            num_heads * head_dim +  # QK^T per position (seq factor in tokens)
            num_heads * 5 +   # Softmax
            num_heads * head_dim +  # Score * V
            h * h             # O proj
        )
        
        self.layer_profiles[LayerType.ATTENTION] = LayerProfile(
            layer_type=LayerType.ATTENTION,
            flops_per_token=attention_flops,
            efficiency=0.45,  # Attention 效率较低
        )
        
        # ========== Dense MLP FLOPs ==========
        # Gate: h * ffn
        # Up: h * ffn
        # SiLU activation: ~10 ops
        # Element-wise multiply: ffn
        # Down: ffn * h
        mlp_flops = (
            h * ffn +     # Gate
            h * ffn +     # Up
            ffn * 10 +    # SiLU
            ffn +         # Multiply
            ffn * h       # Down
        )
        
        self.layer_profiles[LayerType.DENSE_MLP] = LayerProfile(
            layer_type=LayerType.DENSE_MLP,
            flops_per_token=mlp_flops,
            efficiency=0.55,
        )
        
        # ========== MoE Router FLOPs ==========
        # Linear: h * num_experts
        # TopK selection: ~num_experts * log(num_experts)
        router_flops = (
            h * num_experts +
            num_experts * int(math.log2(num_experts + 1)) * 2
        )
        
        self.layer_profiles[LayerType.MOE_ROUTER] = LayerProfile(
            layer_type=LayerType.MOE_ROUTER,
            flops_per_token=router_flops,
            efficiency=0.30,  # Router 效率较低
        )
        
        # ========== MoE Expert FLOPs ==========
        # 单个 Expert 与 MLP 相同，但使用 moe_ffn
        expert_flops = (
            h * moe_ffn +     # Gate
            h * moe_ffn +     # Up
            moe_ffn * 10 +    # SiLU
            moe_ffn +         # Multiply
            moe_ffn * h       # Down
        )
        
        # MoE Expert 效率受 batch size 和 EP 影响
        # 小 batch size 时每个专家处理的 token 很少，GEMM 效率极低
        # 实测 Qwen3-30B-A3B 在 bs=1, ep=8 时效率约 8%
        self.layer_profiles[LayerType.MOE_EXPERT] = LayerProfile(
            layer_type=LayerType.MOE_EXPERT,
            flops_per_token=expert_flops,
            efficiency=0.08,  # MoE 专家在小 batch 下效率极低
        )
        
        # ========== LayerNorm FLOPs ==========
        # Mean: h, Variance: h, Normalize: h * 3, Scale+Bias: h * 2
        layernorm_flops = h * 7
        
        self.layer_profiles[LayerType.LAYERNORM] = LayerProfile(
            layer_type=LayerType.LAYERNORM,
            flops_per_token=layernorm_flops,
            efficiency=0.20,  # LayerNorm 内存绑定
        )
    
    def estimate_layer_time(self, layer_type: LayerType,
                           batch_size: int, seq_len: int,
                           tp_degree: int = 1) -> float:
        """
        估算单层计算时间 (ms)
        
        Args:
            layer_type: 层类型
            batch_size: micro batch size
            seq_len: 序列长度
            tp_degree: TP 并行度
        """
        profile = self.layer_profiles.get(layer_type)
        if profile is None:
            return 0.0
        
        tokens = batch_size * seq_len
        peak_tflops = self.hardware.gpu.get_tflops(self.training.dtype)
        
        return profile.estimate_time_ms(tokens, peak_tflops, tp_degree)
    
    def estimate_attention_time(self, batch_size: int, seq_len: int,
                                tp_degree: int = 1) -> float:
        """估算 Attention 计算时间 (ms)"""
        # 基本计算
        base_time = self.estimate_layer_time(
            LayerType.ATTENTION, batch_size, seq_len, tp_degree
        )
        
        # Attention score 部分与 seq_len 平方相关
        # 额外考虑 O(seq^2) 的开销
        h = self.model.hidden_size
        num_heads = self.model.num_attention_heads // tp_degree
        head_dim = self.model.head_dim
        tokens = batch_size * seq_len
        
        # QK^T 和 Score*V 的额外时间
        qk_flops = batch_size * num_heads * seq_len * seq_len * head_dim * 2
        peak_tflops = self.hardware.gpu.get_tflops(self.training.dtype)
        efficiency = 0.45
        
        qk_time_ms = qk_flops / (peak_tflops * 1e12 * efficiency) * 1000
        
        return base_time + qk_time_ms
    
    def estimate_mlp_time(self, batch_size: int, seq_len: int,
                          tp_degree: int = 1) -> float:
        """估算 Dense MLP 计算时间 (ms)"""
        return self.estimate_layer_time(
            LayerType.DENSE_MLP, batch_size, seq_len, tp_degree
        )
    
    def estimate_moe_time(self, batch_size: int, seq_len: int,
                          tp_degree: int = 1, ep_degree: int = 1) -> float:
        """
        估算 MoE 层计算时间 (ms)
        
        包括 Router + TopK Expert 计算
        """
        # Router 时间
        router_time = self.estimate_layer_time(
            LayerType.MOE_ROUTER, batch_size, seq_len, 1  # Router 不 TP 切分
        )
        
        # Expert 计算时间
        # 每个 token 只激活 TopK 个 Expert
        # EP 切分后，每个 GPU 只计算 num_experts/ep 个 Expert
        topk = self.model.num_experts_per_tok
        experts_per_gpu = self.model.num_experts // ep_degree
        
        # 平均每个 GPU 处理的 token 数
        # 理想负载均衡下，每个 GPU 处理 tokens * topk / ep 个 token-expert pair
        tokens = batch_size * seq_len
        tokens_per_gpu = tokens * topk / ep_degree
        
        expert_time = self.estimate_layer_time(
            LayerType.MOE_EXPERT, 1, int(tokens_per_gpu), tp_degree
        )
        
        return router_time + expert_time
    
    def estimate_dense_layer_time(self, batch_size: int, seq_len: int,
                                  tp_degree: int = 1) -> float:
        """估算单个 Dense 层 (Attention + MLP) 时间 (ms)"""
        attn_time = self.estimate_attention_time(batch_size, seq_len, tp_degree)
        mlp_time = self.estimate_mlp_time(batch_size, seq_len, tp_degree)
        
        # LayerNorm 时间
        ln_time = self.estimate_layer_time(
            LayerType.LAYERNORM, batch_size, seq_len, 1
        ) * 2  # 2 个 LayerNorm
        
        return attn_time + mlp_time + ln_time
    
    def estimate_moe_layer_time(self, batch_size: int, seq_len: int,
                                tp_degree: int = 1, ep_degree: int = 1) -> float:
        """估算单个 MoE 层 (Attention + MoE) 时间 (ms)"""
        attn_time = self.estimate_attention_time(batch_size, seq_len, tp_degree)
        moe_time = self.estimate_moe_time(batch_size, seq_len, tp_degree, ep_degree)
        
        # LayerNorm 时间
        ln_time = self.estimate_layer_time(
            LayerType.LAYERNORM, batch_size, seq_len, 1
        ) * 2
        
        return attn_time + moe_time + ln_time
    
    def estimate_forward_time(self, batch_size: int, seq_len: int,
                              parallel: ParallelConfig) -> float:
        """
        估算完整前向传播时间 (ms)
        
        考虑 PP 切分，每个 stage 只计算部分层
        """
        tp = parallel.tp
        pp = parallel.pp
        ep = parallel.ep
        
        layers_per_stage = self.model.num_hidden_layers // pp
        dense_layers_per_stage = self.model.num_dense_layers // pp
        moe_layers_per_stage = self.model.num_moe_layers // pp
        
        # Dense 层时间
        dense_time = self.estimate_dense_layer_time(batch_size, seq_len, tp)
        total_dense_time = dense_time * dense_layers_per_stage
        
        # MoE 层时间
        moe_time = self.estimate_moe_layer_time(batch_size, seq_len, tp, ep)
        total_moe_time = moe_time * moe_layers_per_stage
        
        return total_dense_time + total_moe_time
    
    def estimate_backward_time(self, batch_size: int, seq_len: int,
                               parallel: ParallelConfig) -> float:
        """
        估算完整反向传播时间 (ms)
        
        通常约为前向的 2 倍
        """
        forward_time = self.estimate_forward_time(batch_size, seq_len, parallel)
        return forward_time * 2.0
    
    def estimate_pipeline_bubble(self, forward_time: float, backward_time: float,
                                 pp_degree: int, num_micro_batches: int) -> float:
        """
        估算流水线气泡时间 (ms)
        
        1F1B 调度: bubble_ratio = (pp_degree - 1) / num_micro_batches
        """
        if pp_degree <= 1:
            return 0.0
        
        bubble_ratio = (pp_degree - 1) / num_micro_batches
        single_stage_time = forward_time + backward_time
        
        return single_stage_time * bubble_ratio
    
    def estimate_step_compute_time(self, batch_size: int, seq_len: int,
                                   parallel: ParallelConfig,
                                   num_micro_batches: int,
                                   recompute_overhead: float = 1.0) -> Dict[str, float]:
        """
        估算一个 step 的计算时间
        
        Args:
            batch_size: micro batch size
            seq_len: 序列长度
            parallel: 并行配置
            num_micro_batches: gradient accumulation steps
            recompute_overhead: 重计算开销因子
        
        Returns:
            时间详情字典
        """
        # 前向时间
        forward_time = self.estimate_forward_time(batch_size, seq_len, parallel)
        
        # 后向时间 (考虑重计算开销)
        backward_time = self.estimate_backward_time(batch_size, seq_len, parallel)
        backward_time = backward_time * recompute_overhead
        
        # 流水线气泡
        bubble_time = self.estimate_pipeline_bubble(
            forward_time, backward_time, parallel.pp, num_micro_batches
        )
        
        # 总计算时间
        compute_time = (forward_time + backward_time) * num_micro_batches + bubble_time
        
        return {
            "forward_time_ms": forward_time * num_micro_batches,
            "backward_time_ms": backward_time * num_micro_batches,
            "bubble_time_ms": bubble_time,
            "compute_time_ms": compute_time,
            "bubble_ratio": bubble_time / compute_time if compute_time > 0 else 0,
        }