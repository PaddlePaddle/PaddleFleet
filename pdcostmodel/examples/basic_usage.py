#!/usr/bin/env python3
"""
pdcost 基础使用示例

演示如何使用 pdcost 预测 PaddleFormers 分布式训练的性能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pdcost import (
    PDCostModel, 
    ModelConfig, 
    ParallelConfig, 
    TrainingConfig,
    HardwareConfig,
    GPUSpec,
)


def example_single_config():
    """示例 1: 预测单个并行配置"""
    print("\n" + "=" * 80)
    print("📌 示例 1: 预测单个并行配置")
    print("=" * 80)
    
    # 1. 创建模型配置 (Qwen3-30B-A3B MoE)
    model_config = ModelConfig.from_name("qwen3-30b-a3b")
    print(f"\n模型: Qwen3-30B-A3B")
    params = model_config.estimate_parameters()
    print(f"参数量: {params['total_billion']:.2f}B")
    print(f"  - Embedding: {params['embedding']/1e9:.2f}B")
    print(f"  - Attention: {params['attention']/1e9:.2f}B")
    print(f"  - Dense MLP: {params['dense_mlp']/1e9:.2f}B")
    print(f"  - MoE: {params['moe']/1e9:.2f}B")
    
    # 2. 创建硬件配置 (8x H100)
    hardware_config = HardwareConfig(
        gpu=GPUSpec.from_name("H100-80GB-HBM3"),
        num_nodes=1,
        gpus_per_node=8,
    )
    print(f"\n硬件: {hardware_config.gpu.name} × {hardware_config.total_gpus}")
    
    # 3. 创建训练配置
    training_config = TrainingConfig(
        micro_batch_size=1,
        sequence_length=8192,
        gradient_accumulation_steps=64,
        dtype="bfloat16",
        recompute_granularity="full",
    )
    print(f"训练配置: mbs={training_config.micro_batch_size}, "
          f"seq={training_config.sequence_length}, "
          f"grad_acc={training_config.gradient_accumulation_steps}")
    
    # 4. 创建 CostModel
    costmodel = PDCostModel(model_config, hardware_config, training_config)
    
    # 5. 预测并行配置
    parallel = ParallelConfig(
        tp=8,
        pp=1,
        dp=1,
        ep=8,
        sharding="stage1",
    )
    print(f"\n并行配置: {parallel}")
    
    result = costmodel.predict(parallel)
    print(f"\n预测结果:")
    print(result)
    
    # 详细显存分解
    print(f"\n显存详情:")
    print(result.memory_breakdown)


def example_compare_configs():
    """示例 2: 比较多个并行配置"""
    print("\n" + "=" * 80)
    print("📌 示例 2: 比较多个并行配置")
    print("=" * 80)
    
    model_config = ModelConfig.from_name("qwen3-30b-a3b")
    costmodel = PDCostModel(model_config)
    
    # 定义多个待比较的配置
    configs = [
        {"tp": 1, "pp": 1, "dp": 8, "ep": 1, "sharding": "stage2"},
        {"tp": 2, "pp": 1, "dp": 4, "ep": 2, "sharding": "stage1"},
        {"tp": 4, "pp": 1, "dp": 2, "ep": 4, "sharding": "stage1"},
        {"tp": 8, "pp": 1, "dp": 1, "ep": 8, "sharding": "stage1"},
        {"tp": 4, "pp": 2, "dp": 1, "ep": 4, "sharding": "stage1"},
    ]
    
    # 批量预测并排序
    results = costmodel.rank_configurations(configs, top_k=5)
    
    return results


def example_search_space():
    """示例 3: 搜索最优配置"""
    print("\n" + "=" * 80)
    print("📌 示例 3: 自动搜索最优配置")
    print("=" * 80)
    
    model_config = ModelConfig.from_name("qwen3-30b-a3b")
    
    # 设置更大的集群 (2 nodes × 8 GPUs)
    hardware_config = HardwareConfig(
        gpu=GPUSpec.from_name("H100-80GB-HBM3"),
        num_nodes=2,
        gpus_per_node=8,
    )
    
    training_config = TrainingConfig(
        micro_batch_size=1,
        sequence_length=8192,
        gradient_accumulation_steps=128,
    )
    
    costmodel = PDCostModel(model_config, hardware_config, training_config)
    
    # 生成搜索空间
    total_gpus = hardware_config.total_gpus
    print(f"\n总 GPU 数: {total_gpus}")
    
    configs = costmodel.generate_search_space(total_gpus, max_tp=8, max_pp=4)
    print(f"搜索空间大小: {len(configs)}")
    
    # 搜索最优配置
    best_configs = costmodel.rank_configurations(configs, top_k=10)
    
    return best_configs


def example_dense_model():
    """示例 4: Dense 模型 (非 MoE)"""
    print("\n" + "=" * 80)
    print("📌 示例 4: Dense 模型预测 (LLaMA-3 70B)")
    print("=" * 80)
    
    model_config = ModelConfig.from_name("llama3-70b")
    print(f"\n模型: LLaMA-3 70B (Dense)")
    params = model_config.estimate_parameters()
    print(f"参数量: {params['total_billion']:.2f}B")
    
    hardware_config = HardwareConfig(
        gpu=GPUSpec.from_name("H100-80GB-HBM3"),
        num_nodes=1,
        gpus_per_node=8,
    )
    
    training_config = TrainingConfig(
        micro_batch_size=1,
        sequence_length=4096,
        gradient_accumulation_steps=32,
    )
    
    costmodel = PDCostModel(model_config, hardware_config, training_config)
    
    # Dense 模型不需要 EP
    configs = [
        {"tp": 8, "pp": 1, "dp": 1, "ep": 1, "sharding": "stage1"},
        {"tp": 4, "pp": 2, "dp": 1, "ep": 1, "sharding": "stage1"},
        {"tp": 4, "pp": 1, "dp": 2, "ep": 1, "sharding": "stage2"},
        {"tp": 2, "pp": 4, "dp": 1, "ep": 1, "sharding": "stage1"},
    ]
    
    results = costmodel.rank_configurations(configs, top_k=4)
    
    return results


def example_custom_model():
    """示例 5: 自定义模型配置"""
    print("\n" + "=" * 80)
    print("📌 示例 5: 自定义模型配置")
    print("=" * 80)
    
    # 创建自定义模型配置
    custom_model = ModelConfig(
        num_hidden_layers=32,
        hidden_size=4096,
        intermediate_size=11008,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        num_experts=64,
        num_experts_per_tok=4,
        moe_intermediate_size=2048,
        vocab_size=32000,
    )
    
    print(f"\n自定义模型:")
    print(f"  - 层数: {custom_model.num_hidden_layers}")
    print(f"  - hidden_size: {custom_model.hidden_size}")
    print(f"  - num_experts: {custom_model.num_experts}")
    print(f"  - topk: {custom_model.num_experts_per_tok}")
    
    params = custom_model.estimate_parameters()
    print(f"  - 参数量: {params['total_billion']:.2f}B")
    
    costmodel = PDCostModel(custom_model)
    
    parallel = ParallelConfig(tp=4, pp=1, dp=2, ep=4, sharding="stage1")
    result = costmodel.predict(parallel, micro_batch_size=2, seq_len=4096)
    
    print(f"\n预测结果 ({parallel}):")
    print(result)


def main():
    """运行所有示例"""
    print("\n" + "🚀" * 30)
    print("  pdcost - PaddleFormers 分布式训练代价模型示例")
    print("🚀" * 30)
    
    # 示例 1: 单配置预测
    example_single_config()
    
    # 示例 2: 配置比较
    example_compare_configs()
    
    # 示例 3: 搜索空间
    example_search_space()
    
    # 示例 4: Dense 模型
    example_dense_model()
    
    # 示例 5: 自定义模型
    example_custom_model()
    
    print("\n" + "✅ 所有示例完成!")


if __name__ == "__main__":
    main()