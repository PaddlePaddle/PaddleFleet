# PDCostModel - 分布式训练代价模型

预测 PaddleFormers 分布式训练性能，在实际运行前估算：

- **Step 时间** - 训练迭代耗时
- **显存占用** - 支持 allocated / reserved 双指标
- **训练吞吐量** - tokens/s/GPU
- **OOM 检测** - 自动过滤超显存配置

## 快速开始

### 一行代码搜索最优配置

```python
from pdcostmodel import grid_search

# 搜索最优并行配置
results = grid_search("qwen3-30b-a3b", total_gpus=8)

# 获取最优配置
best = results.best
print(f"最优配置: {best.config_str}")
print(f"吞吐量: {best.tokens_per_second_per_gpu:.0f} tok/s/GPU")
print(f"显存: {best.memory_gb:.2f} GB")
```

### 自定义搜索参数

```python
results = grid_search(
    model="qwen3-30b-a3b",
    total_gpus=8,
    seq_len=4096,
    micro_batch_size=4,
    gradient_accumulation_steps=64,
    tp_candidates=[1, 2, 4],
    pp_candidates=[1, 2],
)
```

### 生成训练配置 YAML

```python
from pdcostmodel import GridSearcher

searcher = GridSearcher(model_name="qwen3-30b-a3b", total_gpus=8)
results = searcher.search()

# 生成最优配置 YAML
searcher.generate_yaml_config(output_path="best_config.yaml")
```

## 关键参数

| 参数 | 说明 |
|-----|------|
| `tp` | 张量并行度 |
| `pp` | 流水线并行度 |
| `dp` | 数据并行度 (自动计算) |
| `ep` | 专家并行度 (MoE 模型) |
| `sharding` | ZeRO 阶段: `stage1` / `stage2` |
| `micro_batch_size` | 单卡单次处理样本数 |
| `seq_len` | 序列长度 |
| `gradient_accumulation_steps` | 梯度累积步数 |



## 注意事项

1. 首次运行会自动执行硬件校准 (~1-2 分钟)
2. MoE 模型建议使用 `full` 重计算
3. 预测值为理论估算，建议用实际 benchmark 验证关键配置