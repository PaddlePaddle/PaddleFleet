#!/usr/bin/env python3
"""
配置模块 - 定义 PaddleFormers 训练的各类配置

支持从 PaddleFormers 的 config.json 或 TrainingArguments 解析配置
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import json
import os


class ShardingStage(Enum):
    """Sharding (ZeRO) 阶段"""
    NONE = "none"
    STAGE1 = "stage1"  # 优化器状态分片
    STAGE2 = "stage2"  # 优化器状态 + 梯度分片
    STAGE3 = "stage3"  # 优化器状态 + 梯度 + 参数分片


class RecomputeGranularity(Enum):
    """重计算粒度"""
    NONE = "none"
    SELECTIVE = "selective"  # 选择性重计算
    FULL = "full"  # 全量重计算


@dataclass
class GPUSpec:
    """GPU 硬件规格"""
    name: str = "H100-80GB-HBM3"
    memory_gb: float = 80.0
    
    # 算力 (TFLOPS)
    fp32_tflops: float = 67.0
    fp16_tflops: float = 989.0
    bf16_tflops: float = 989.0
    
    # 带宽 (GB/s)
    memory_bandwidth_gbps: float = 3350.0
    
    def get_tflops(self, dtype: str = "bf16") -> float:
        """获取指定数据类型的理论算力"""
        dtype_map = {
            "fp32": self.fp32_tflops,
            "fp16": self.fp16_tflops,
            "bf16": self.bf16_tflops,
        }
        return dtype_map.get(dtype, self.bf16_tflops)
    
    @classmethod
    def from_name(cls, name: str) -> "GPUSpec":
        """根据 GPU 型号创建规格"""
        presets = {
            "H100-80GB-HBM3": cls(
                name="H100-80GB-HBM3",
                memory_gb=80.0,
                fp32_tflops=67.0,
                fp16_tflops=989.0,
                bf16_tflops=989.0,
                memory_bandwidth_gbps=3350.0,
            ),
            "H100-80GB-PCIe": cls(
                name="H100-80GB-PCIe",
                memory_gb=80.0,
                fp32_tflops=51.0,
                fp16_tflops=756.0,
                bf16_tflops=756.0,
                memory_bandwidth_gbps=2000.0,
            ),
            "A100-80GB": cls(
                name="A100-80GB",
                memory_gb=80.0,
                fp32_tflops=19.5,
                fp16_tflops=312.0,
                bf16_tflops=312.0,
                memory_bandwidth_gbps=2039.0,
            ),
            "A100-40GB": cls(
                name="A100-40GB", 
                memory_gb=40.0,
                fp32_tflops=19.5,
                fp16_tflops=312.0,
                bf16_tflops=312.0,
                memory_bandwidth_gbps=1555.0,
            ),
            "A800-80GB": cls(
                name="A800-80GB",
                memory_gb=80.0,
                fp32_tflops=19.5,
                fp16_tflops=312.0,
                bf16_tflops=312.0,
                memory_bandwidth_gbps=2039.0,
            ),
            "V100-32GB": cls(
                name="V100-32GB",
                memory_gb=32.0,
                fp32_tflops=15.7,
                fp16_tflops=125.0,
                bf16_tflops=0.0,  # V100 不支持 BF16
                memory_bandwidth_gbps=900.0,
            ),
        }
        return presets.get(name, presets["H100-80GB-HBM3"])


@dataclass
class NetworkSpec:
    """网络规格"""
    # 节点内通信 (NVLink/NVSwitch)
    intra_node_bandwidth_gbps: float = 900.0  # H100 NVLink: 900 GB/s
    intra_node_latency_us: float = 1.0
    
    # 节点间通信 (IB/RoCE)
    inter_node_bandwidth_gbps: float = 200.0  # 8x IB HDR: 200 GB/s
    inter_node_latency_us: float = 5.0
    
    # 通信效率因子
    allreduce_efficiency: float = 0.85
    allgather_efficiency: float = 0.80
    alltoall_efficiency: float = 0.70
    p2p_efficiency: float = 0.90


@dataclass
class HardwareConfig:
    """硬件配置"""
    gpu: GPUSpec = field(default_factory=GPUSpec)
    network: NetworkSpec = field(default_factory=NetworkSpec)
    
    # 集群配置
    num_nodes: int = 1
    gpus_per_node: int = 8
    
    @property
    def total_gpus(self) -> int:
        return self.num_nodes * self.gpus_per_node
    
    def is_intra_node(self, degree: int) -> bool:
        """判断通信是否在节点内"""
        return degree <= self.gpus_per_node


@dataclass
class ModelConfig:
    """
    模型架构配置
    
    与 PaddleFormers 的 PretrainedConfig 对应
    """
    # 基本参数
    num_hidden_layers: int = 48
    hidden_size: int = 6144
    intermediate_size: int = 16384
    num_attention_heads: int = 32
    num_key_value_heads: int = 4  # GQA
    head_dim: int = 192
    
    # MoE 参数
    num_experts: int = 128
    num_experts_per_tok: int = 8  # TopK
    moe_intermediate_size: int = 1408  # Expert FFN 大小
    decoder_sparse_step: int = 1  # MoE 层间隔（1 表示每层都是 MoE）
    mlp_only_layers: List[int] = field(default_factory=list)  # 纯 MLP 层的索引
    
    # 其他参数
    vocab_size: int = 152064
    max_position_embeddings: int = 32768
    
    # 计算得到的属性
    @property
    def num_moe_layers(self) -> int:
        """MoE 层数"""
        if self.num_experts <= 1:
            return 0
        # 根据 decoder_sparse_step 和 mlp_only_layers 计算
        moe_count = 0
        for i in range(self.num_hidden_layers):
            if i not in self.mlp_only_layers and (i + 1) % self.decoder_sparse_step == 0:
                moe_count += 1
        return moe_count
    
    @property
    def num_dense_layers(self) -> int:
        """Dense 层数"""
        return self.num_hidden_layers - self.num_moe_layers
    
    def estimate_parameters(self) -> Dict[str, int]:
        """估算参数量"""
        h = self.hidden_size
        ffn = self.intermediate_size
        moe_ffn = self.moe_intermediate_size
        v = self.vocab_size
        
        # Embedding: v * h
        embedding_params = v * h
        
        # 每层 Attention: 4 * h * h (Q,K,V,O) - 考虑 GQA
        q_size = h
        kv_size = self.num_key_value_heads * self.head_dim
        attention_params_per_layer = q_size * h + 2 * kv_size * h + h * h  # Q + K + V + O
        
        # Dense MLP: 3 * h * ffn (gate, up, down with SwiGLU)
        dense_mlp_params = 3 * h * ffn
        
        # MoE 层: router + experts
        router_params = h * self.num_experts
        expert_params = 3 * h * moe_ffn * self.num_experts  # 所有专家的参数
        moe_layer_params = router_params + expert_params
        
        # LayerNorm: 2 * h per layer (input + post_attention)
        layernorm_params = 2 * h
        
        # 汇总
        total_attention = attention_params_per_layer * self.num_hidden_layers
        total_dense_mlp = dense_mlp_params * self.num_dense_layers
        total_moe = moe_layer_params * self.num_moe_layers
        total_layernorm = layernorm_params * self.num_hidden_layers
        total_embedding = embedding_params * 2  # input + output (if not tied)
        
        total = total_attention + total_dense_mlp + total_moe + total_layernorm + total_embedding
        
        return {
            "embedding": total_embedding,
            "attention": total_attention,
            "dense_mlp": total_dense_mlp,
            "moe": total_moe,
            "layernorm": total_layernorm,
            "total": total,
            "total_billion": total / 1e9,
        }
    
    @classmethod
    def from_name(cls, name: str) -> "ModelConfig":
        """根据模型名称创建配置"""
        presets = {
            # Qwen3 MoE 系列 (实际配置从 config.json 读取)
            "qwen3-30b-a3b": cls(
                num_hidden_layers=48,
                hidden_size=2048,           # 实际值
                intermediate_size=6144,     # 实际值 (Dense MLP, 但 Qwen3-30B 全是 MoE 层)
                num_attention_heads=32,
                num_key_value_heads=4,
                head_dim=128,               # 实际值
                num_experts=128,
                num_experts_per_tok=8,
                moe_intermediate_size=768,  # 实际值
                decoder_sparse_step=1,
                vocab_size=151936,          # 实际值
            ),
            "qwen3-235b-a22b": cls(
                num_hidden_layers=94,
                hidden_size=9216,
                intermediate_size=24576,
                num_attention_heads=64,
                num_key_value_heads=8,
                head_dim=144,
                num_experts=128,
                num_experts_per_tok=8,
                moe_intermediate_size=3072,
                decoder_sparse_step=1,
                vocab_size=152064,
            ),
            # DeepSeek MoE 系列
            "deepseek-v3": cls(
                num_hidden_layers=61,
                hidden_size=7168,
                intermediate_size=18432,
                num_attention_heads=56,
                num_key_value_heads=8,
                head_dim=128,
                num_experts=256,
                num_experts_per_tok=8,
                moe_intermediate_size=2048,
                decoder_sparse_step=1,
                mlp_only_layers=[0, 1, 2],  # 前3层是 Dense
                vocab_size=129024,
            ),
            # Dense 模型
            "llama3-70b": cls(
                num_hidden_layers=80,
                hidden_size=8192,
                intermediate_size=28672,
                num_attention_heads=64,
                num_key_value_heads=8,
                head_dim=128,
                num_experts=1,
                vocab_size=128256,
            ),
            "llama3-8b": cls(
                num_hidden_layers=32,
                hidden_size=4096,
                intermediate_size=14336,
                num_attention_heads=32,
                num_key_value_heads=8,
                head_dim=128,
                num_experts=1,
                vocab_size=128256,
            ),
        }
        
        name_lower = name.lower().replace("_", "-").replace(" ", "-")
        if name_lower in presets:
            return presets[name_lower]
        
        raise ValueError(f"Unknown model: {name}. Available: {list(presets.keys())}")
    
    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        """从 config.json 加载"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ModelConfig":
        """从字典创建"""
        return cls(
            num_hidden_layers=data.get("num_hidden_layers", 48),
            hidden_size=data.get("hidden_size", 6144),
            intermediate_size=data.get("intermediate_size", 16384),
            num_attention_heads=data.get("num_attention_heads", 32),
            num_key_value_heads=data.get("num_key_value_heads", 4),
            head_dim=data.get("head_dim", data.get("hidden_size", 6144) // data.get("num_attention_heads", 32)),
            num_experts=data.get("num_experts", 1),
            num_experts_per_tok=data.get("num_experts_per_tok", 8),
            moe_intermediate_size=data.get("moe_intermediate_size", data.get("intermediate_size", 16384)),
            decoder_sparse_step=data.get("decoder_sparse_step", 1),
            mlp_only_layers=data.get("mlp_only_layers", []),
            vocab_size=data.get("vocab_size", 152064),
            max_position_embeddings=data.get("max_position_embeddings", 32768),
        )


@dataclass
class ParallelConfig:
    """
    并行配置
    
    与 PaddleFormers TrainingArguments 对应
    """
    # 张量并行
    tp: int = 1  # tensor_model_parallel_size
    
    # 流水线并行
    pp: int = 1  # pipeline_model_parallel_size
    
    # 数据并行
    dp: int = 1  # 自动计算或显式设置
    
    # Sharding (ZeRO)
    sharding: str = "stage1"  # "none", "stage1", "stage2", "stage3"
    sharding_degree: int = -1  # -1 表示等于 dp
    
    # Expert 并行 (MoE)
    ep: int = 1  # expert_model_parallel_size
    
    # Sequence 并行
    sp: bool = False  # sequence_parallel
    
    # Context 并行
    cp: int = 1  # context_parallel_size
    
    @property
    def sharding_stage(self) -> ShardingStage:
        """获取 Sharding 阶段"""
        mapping = {
            "none": ShardingStage.NONE,
            "": ShardingStage.NONE,
            "stage1": ShardingStage.STAGE1,
            "stage2": ShardingStage.STAGE2,
            "stage3": ShardingStage.STAGE3,
        }
        return mapping.get(self.sharding.lower(), ShardingStage.STAGE1)
    
    @property
    def effective_sharding_degree(self) -> int:
        """有效的 Sharding 度"""
        if self.sharding_stage == ShardingStage.NONE:
            return 1
        if self.sharding_degree > 0:
            return self.sharding_degree
        return self.dp
    
    @property
    def world_size(self) -> int:
        """总 GPU 数"""
        return self.dp * self.tp * self.pp
    
    def validate(self, total_gpus: int) -> bool:
        """验证配置是否合法"""
        # 基本约束
        if self.tp < 1 or self.pp < 1 or self.dp < 1:
            return False
        
        # 总数匹配
        if self.dp * self.tp * self.pp != total_gpus:
            return False
        
        # EP 约束: EP 度数不能超过 Expert 数（由用户保证）
        # MoE sharding 通常 = world_size / (pp * ep)
        
        return True
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "tp": self.tp,
            "pp": self.pp,
            "dp": self.dp,
            "sharding": self.sharding,
            "sharding_degree": self.sharding_degree,
            "ep": self.ep,
            "sp": self.sp,
            "cp": self.cp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ParallelConfig":
        """从字典创建"""
        return cls(
            tp=data.get("tp", data.get("tensor_model_parallel_size", 1)),
            pp=data.get("pp", data.get("pipeline_model_parallel_size", 1)),
            dp=data.get("dp", data.get("data_parallel_size", 1)),
            sharding=data.get("sharding", "stage1"),
            sharding_degree=data.get("sharding_degree", data.get("sharding_parallel_size", -1)),
            ep=data.get("ep", data.get("expert_model_parallel_size", 1)),
            sp=data.get("sp", data.get("sequence_parallel", False)),
            cp=data.get("cp", data.get("context_parallel_size", 1)),
        )
    
    def __str__(self) -> str:
        parts = [f"TP{self.tp}", f"PP{self.pp}", f"DP{self.dp}"]
        if self.ep > 1:
            parts.append(f"EP{self.ep}")
        if self.sharding != "none" and self.sharding:
            parts.append(f"Sharding({self.sharding})")
        if self.sp:
            parts.append("SP")
        if self.cp > 1:
            parts.append(f"CP{self.cp}")
        return "-".join(parts)


@dataclass
class TrainingConfig:
    """
    训练配置
    
    与 PaddleFormers TrainingArguments 对应
    """
    # Batch 配置
    micro_batch_size: int = 1  # per_device_train_batch_size
    global_batch_size: int = 512
    gradient_accumulation_steps: int = 64
    
    # 序列长度
    sequence_length: int = 8192
    
    # 数据类型
    dtype: str = "bfloat16"  # "float32", "float16", "bfloat16"
    
    # 混合精度
    fp16_opt_level: str = "O2"  # "O0", "O1", "O2"
    amp_master_grad: bool = True
    
    # 重计算
    recompute_granularity: str = "full"  # "none", "selective", "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 1
    
    @property
    def dtype_bytes(self) -> int:
        """数据类型字节数"""
        return {"float32": 4, "float16": 2, "bfloat16": 2}.get(self.dtype, 2)
    
    @property
    def recompute_config(self) -> "RecomputeGranularity":
        """获取重计算配置枚举"""
        mapping = {
            "none": RecomputeGranularity.NONE,
            "": RecomputeGranularity.NONE,
            "selective": RecomputeGranularity.SELECTIVE,
            "full": RecomputeGranularity.FULL,
        }
        return mapping.get(self.recompute_granularity.lower(), RecomputeGranularity.FULL)
    
    def to_dict(self) -> Dict:
        return {
            "micro_batch_size": self.micro_batch_size,
            "global_batch_size": self.global_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "sequence_length": self.sequence_length,
            "dtype": self.dtype,
            "fp16_opt_level": self.fp16_opt_level,
            "amp_master_grad": self.amp_master_grad,
            "recompute_granularity": self.recompute_granularity,
            "recompute_method": self.recompute_method,
            "recompute_num_layers": self.recompute_num_layers,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "TrainingConfig":
        return cls(
            micro_batch_size=data.get("micro_batch_size", data.get("per_device_train_batch_size", 1)),
            global_batch_size=data.get("global_batch_size", 512),
            gradient_accumulation_steps=data.get("gradient_accumulation_steps", 64),
            sequence_length=data.get("sequence_length", data.get("max_sequence_length", 8192)),
            dtype=data.get("dtype", "bfloat16"),
            fp16_opt_level=data.get("fp16_opt_level", "O2"),
            amp_master_grad=data.get("amp_master_grad", True),
            recompute_granularity=data.get("recompute_granularity", "full"),
            recompute_method=data.get("recompute_method", "uniform"),
            recompute_num_layers=data.get("recompute_num_layers", 1),
        )