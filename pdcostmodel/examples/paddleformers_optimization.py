#!/usr/bin/env python3
"""
PaddleFormers 框架优化机制演示

本示例展示 PaddleFormers 特有的优化机制对显存预测的影响:
1. ShardingV2 (split_param=True): Stage1 也分片参数和梯度
2. sd_release_grads: 每次迭代后释放梯度
3. tensorwise_offload_optimizer: 优化器状态动态 offload 到 CPU
"""

from pdcost import PDCostModel, ModelConfig, ParallelConfig, TrainingConfig, HardwareConfig, GPUSpec


def main():
    print("=" * 80)
    print("🚀 PaddleFormers 框架优化机制演示")
    print("=" * 80)
    
    # 创建 Qwen3-30B-A3B 配置
    model_config = ModelConfig.from_name("qwen3-30b-a3b")
    
    # H800 硬件配置
    hardware = HardwareConfig(
        gpu=GPUSpec(
            name="NVIDIA H800",
            memory_gb=79.6,
            bf16_tflops=788.0,
            fp16_tflops=763.0,
            fp32_tflops=51.6,
            memory_bandwidth_gbps=2780.0
        ),
        num_nodes=1,
        gpus_per_node=8
    )
    
    # 训练配置
    training = TrainingConfig(
        micro_batch_size=1,
        sequence_length=2048,
        gradient_accumulation_steps=16,
        dtype="bfloat16",
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
    )
    
    costmodel = PDCostModel(model_config, hardware, training)
    
    # 并行配置: TP1, PP1, DP8, EP8, Stage1
    parallel = ParallelConfig(tp=1, pp=1, dp=8, ep=8, sharding="stage1")
    
    print(f"\n📋 测试配置:")
    print(f"  模型: Qwen3-30B-A3B (128 experts, top-8)")
    print(f"  硬件: 单机 8 × H800 (79.6GB)")
    print(f"  并行: {parallel}")
    print(f"  micro_batch_size: 1, seq_len: 2048")
    
    print("\n" + "=" * 80)
    print("📊 PaddleFormers 优化机制对比")
    print("=" * 80)
    
    # ========== 方案对比 ==========
    
    # 方案 1: 传统 Stage1 (split_param=False)
    result1 = costmodel.predict(parallel, split_param=False, sd_release_grads=False)
    
    # 方案 2: ShardingV2 默认 (split_param=True)
    result2 = costmodel.predict(parallel, split_param=True, sd_release_grads=False)
    
    # 方案 3: ShardingV2 + sd_release_grads
    result3 = costmodel.predict(parallel, split_param=True, sd_release_grads=True)
    
    # 方案 4: 全优化 (ShardingV2 + release_grads + offload)
    result4 = costmodel.predict(
        parallel, 
        split_param=True, 
        sd_release_grads=True,
        tensorwise_offload_optimizer=True,
        tensorwise_offload_ratio=0.95
    )
    
    # 打印对比表格
    print(f"\n{'方案':<45} {'参数':<10} {'梯度':<10} {'优化器':<10} {'总显存':<10} {'状态':<6}")
    print("-" * 100)
    
    results = [
        ("1. 传统 Stage1 (split_param=False)", result1),
        ("2. ShardingV2 (split_param=True)", result2),
        ("3. ShardingV2 + sd_release_grads", result3),
        ("4. ShardingV2 + release_grads + offload", result4),
    ]
    
    for name, r in results:
        status = "✅" if r.memory_gb <= 79.6 else "❌"
        print(f"{name:<45} "
              f"{r.memory_breakdown.parameter_memory_gb:<10.2f} "
              f"{r.memory_breakdown.gradient_memory_gb:<10.2f} "
              f"{r.memory_breakdown.optimizer_memory_gb:<10.2f} "
              f"{r.memory_gb:<10.2f} "
              f"{status:<6}")
    
    print("-" * 100)
    print(f"H800 显存限制: 79.6 GB")
    
    # ========== 优化效果分析 ==========
    print("\n" + "=" * 80)
    print("📈 优化效果分析")
    print("=" * 80)
    
    print(f"\n1. ShardingV2 (split_param=True) 效果:")
    param_reduction = (result1.memory_breakdown.parameter_memory_gb - 
                       result2.memory_breakdown.parameter_memory_gb)
    grad_reduction = (result1.memory_breakdown.gradient_memory_gb - 
                      result2.memory_breakdown.gradient_memory_gb)
    print(f"   参数显存减少: {param_reduction:.2f} GB ({param_reduction/result1.memory_breakdown.parameter_memory_gb*100:.1f}%)")
    print(f"   梯度显存减少: {grad_reduction:.2f} GB ({grad_reduction/result1.memory_breakdown.gradient_memory_gb*100:.1f}%)")
    print(f"   原理: Stage1 + split_param 会分片参数和梯度 (类似 Stage3 效果)")
    
    print(f"\n2. sd_release_grads 效果:")
    grad_reduction2 = (result2.memory_breakdown.gradient_memory_gb - 
                       result3.memory_breakdown.gradient_memory_gb)
    print(f"   梯度显存减少: {grad_reduction2:.2f} GB ({grad_reduction2/result2.memory_breakdown.gradient_memory_gb*100:.1f}%)")
    print(f"   原理: 每次迭代后释放梯度，峰值显存 = max(激活, 梯度) 而非两者之和")
    
    print(f"\n3. tensorwise_offload_optimizer 效果:")
    opt_reduction = (result3.memory_breakdown.optimizer_memory_gb - 
                     result4.memory_breakdown.optimizer_memory_gb)
    print(f"   优化器显存减少: {opt_reduction:.2f} GB ({opt_reduction/result3.memory_breakdown.optimizer_memory_gb*100:.1f}%)")
    print(f"   原理: 优化器状态按 tensor 粒度动态 offload 到 CPU，只保留 5% 在 GPU")
    
    print(f"\n4. 总体优化效果:")
    total_reduction = result1.memory_gb - result4.memory_gb
    print(f"   总显存减少: {total_reduction:.2f} GB ({total_reduction/result1.memory_gb*100:.1f}%)")
    print(f"   从 {result1.memory_gb:.2f} GB 降低到 {result4.memory_gb:.2f} GB")
    
    # ========== 推荐配置 ==========
    print("\n" + "=" * 80)
    print("💡 PaddleFormers 训练推荐配置")
    print("=" * 80)
    
    print("""
对于 Qwen3-30B-A3B 在单机 8 卡 H800 上训练:

1. 基础配置 (推荐):
   - sharding: stage1
   - split_param: true (默认)
   - 预计显存: ~54 GB

2. 显存优化配置:
   - sharding: stage1
   - split_param: true
   - sd_release_grads: true
   - 预计显存: ~50 GB

3. 极致优化配置 (用于更大模型或更长序列):
   - sharding: stage1
   - split_param: true
   - sd_release_grads: true
   - tensorwise_offload_optimizer: true
   - 预计显存: ~16 GB

注意: tensorwise_offload_optimizer 会增加 CPU-GPU 数据传输开销，
      可能影响训练吞吐量，建议仅在显存紧张时使用。
""")


if __name__ == "__main__":
    main()