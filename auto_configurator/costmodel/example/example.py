#!/usr/bin/env python3
"""最简预测示例 - Qwen3-30B-A3B 单机8卡"""

from costmodel import predict

result = predict(
    model_config_path="/root/paddlejob/workspace/env_run/zhangdongqi/Qwen3-30B-A3B/config.json",
    parallel_config_path="/root/paddlejob/workspace/env_run/zhangdongqi/test.yaml",
    num_nodes=1,
    gpus_per_node=8,
)

print(f"Step 时间: {result.step_time_ms:.1f} ms")
print(f"显存:     {result.memory_gb:.1f} GB ({'OK' if result.fits_memory else 'OOM'})")
print(f"MFU:      {result.mfu:.1%}")
print(f"吞吐:     {result.tokens_per_second_per_gpu:.0f} tok/s/GPU")
