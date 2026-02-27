#!/usr/bin/env python3
"""PDCostModel 单配置预测示例"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pdcostmodel import ModelConfig, ParallelConfig, PDCostModel, get_hardware_config


if __name__ == "__main__":
    # 1. 加载模型和硬件 (从 JSON 加载，与实际训练一致)
    model = ModelConfig.from_json("../Qwen3-30B-A3B-Base/config.json")
    hardware = get_hardware_config(verbose=False)
    costmodel = PDCostModel(model, hardware)
    
    # 2. 定义并行配置 (从 benchmark_best.log 提取: TP=1, PP=1, DP=1, EP=8, Sharding=8)
    parallel = ParallelConfig(tp=1, pp=1, dp=1, ep=8, sharding='stage1', sharding_degree=8)
    
    # 3. 预测性能 (benchmark_best.log 配置: mbs=2, seq_len=8192, gas=64)
    result = costmodel.predict_calibrated(
        parallel,
        micro_batch_size=4,
        seq_len=4096,
        gradient_accumulation_steps=64,
        tensorwise_offload_optimizer=True,
        tensorwise_offload_ratio=0.95,
    )
    
    # 4. 输出结果
    print(f"并行配置: {parallel}")
    print(f"Step 时间: {result.step_time_ms / 1000:.2f} s")
    print(f"显存占用: {result.memory_gb:.1f} GB")
    print(f"吞吐量: {result.tokens_per_second_per_gpu:,.0f} tok/s/GPU")
    print(f"MFU: {result.mfu:.2%}")