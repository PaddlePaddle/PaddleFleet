# PDCostModel - PaddleFormers 分布式训练代价模型

`pdcostmodel` 是一个用于预测 PaddleFormers 分布式训练性能的工具，可以在实际运行前估算不同并行配置下的：

- **Step 时间** (训练迭代耗时)
- **显存占用** (支持双指标: allocated + reserved)
- **硬件利用率** (MFU)
- **训练吞吐量** (tokens/s/GPU)

## ✨ 特性亮点

- 🎯 **高精度预测**: Step time 误差 ~5%，显存误差 ~10%
- ?? **一键搜索**: `grid_search()` 一行代码找到最优并行配置
- 📊 **自动报告**: 格式化输出搜索结果，支持 JSON/YAML 导出
- ⚡ **MoE 专用**: 针对 Qwen3 MoE 等稀疏模型优化
- 🔬 **硬件校准**: 通过实测 GEMM benchmark 校准 GPU 算力

---

## 📁 模块结构

```
pdcostmodel/
├── __init__.py          # 主入口，导出所有公共 API
├── gridsearch.py        # 🔥 Grid Search 配置搜索模块
│   ├── grid_search      # 快速搜索函数
│   ├── GridSearcher     # 搜索器类
│   ├── GridSearchResult # 搜索结果
│   └── SearchResult     # 单个配置结果
├── config.py            # 配置类定义
├── costmodel.py         # 代价模型主类
├── memory_model.py      # 显存预测模型
├── compute_model.py     # 计算时间预测模型
├── comm_model.py        # 通信时间预测模型
├── calibration.py       # 硬件校准模块
├── profile_manager.py   # 校准配置管理
└── test/                # 单元测试
```

---

## 🚀 快速开始

### 🔥 一行代码搜索最优配置 (推荐)

```python
from pdcostmodel import grid_search

# 一行代码搜索最优并行配置
results = grid_search("qwen3-30b-a3b", total_gpus=8)

# 获取最优配置
best = results.best
print(f"最优配置: {best.config_str}")
print(f"吞吐量: {best.tokens_per_second_per_gpu:.0f} tok/s/GPU")
print(f"显存: {best.memory_gb:.2f} GB")
```

输出示例:
```
🔍 检测到硬件: NVIDIA H800 × 8 (79.6 GB)
✅ 找到已保存的校准配置，正在加载...
🔍 搜索配置空间...
   生成了 38 个候选配置
   评估完成: 38 个配置, 18 个满足显存约束

==============================================================================================================
🚀 PDCost 配置搜索报告 - qwen3-30b-a3b
==============================================================================================================
排名   配置                                  step时延(ms)     显存(GB)     约束     MFU      tok/s/GPU   
--------------------------------------------------------------------------------------------------------------
1    TP1-PP1-DP8-EP8-Sharding(stage1)    26447.15       64.02      ✅      17.8%    4,956       
2    TP1-PP1-DP8-EP8-Sharding(stage2)    26447.15       65.88      ✅      17.8%    4,956       
3    TP1-PP2-DP4-EP4-Sharding(stage1)    21664.45       61.59      ✅      10.9%    3,025       
...
```

---

## 🔍 Grid Search API 完整指南

### 1. `grid_search()` - 便捷搜索函数

最简单的使用方式，适合快速探索。

```python
from pdcostmodel import grid_search

results = grid_search(
    model="qwen3-30b-a3b",           # 模型名称或 ModelConfig 对象
    total_gpus=8,                     # 总 GPU 数
    micro_batch_size=1,               # 每设备批次大小
    seq_len=8192,                     # 序列长度
    gradient_accumulation_steps=16,   # 梯度累积步数
    tp_candidates=[1, 2, 4, 8],       # TP 候选值 (可选)
    pp_candidates=[1, 2, 4],          # PP 候选值 (可选)
    ep_candidates=[1, 2, 4, 8],       # EP 候选值 (可选)
    sharding_candidates=["stage1", "stage2"],  # Sharding 候选值 (可选)
    sort_by="throughput",             # 排序方式: "throughput" 或 "step_time"
    top_k=10,                         # 显示前 k 个配置
    print_report=True,                # 是否打印报告
    verbose=True,                     # 是否打印详细信息
)
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | str / ModelConfig | 必填 | 模型名称 (如 "qwen3-30b-a3b") 或自定义 ModelConfig |
| `total_gpus` | int | 8 | 总 GPU 数量 |
| `micro_batch_size` | int | 1 | 每设备批次大小 |
| `seq_len` | int | 8192 | 序列长度 |
| `gradient_accumulation_steps` | int | 16 | 梯度累积步数 |
| `tp_candidates` | List[int] | [1,2,4,8] | Tensor Parallel 候选值 |
| `pp_candidates` | List[int] | [1,2,4] | Pipeline Parallel 候选值 |
| `ep_candidates` | List[int] | 自动 | Expert Parallel 候选值 (MoE 模型自动设置) |
| `sharding_candidates` | List[str] | ["stage1","stage2"] | ZeRO Sharding 候选值 |
| `sort_by` | str | "throughput" | 排序方式: "throughput" 或 "step_time" |
| `top_k` | int | 10 | 显示前 k 个配置 |
| `print_report` | bool | True | 是否打印格式化报告 |
| `verbose` | bool | True | 是否打印详细信息 |

#### 返回值: `GridSearchResult`

```python
results.model_name          # 模型名称
results.total_gpus          # 总 GPU 数
results.total_configs       # 总配置数
results.valid_configs       # 有效配置数 (满足显存约束)
results.results             # List[SearchResult] - 排序后的配置列表
results.best                # SearchResult - 最优配置
results.search_space        # 搜索空间配置
results.training_params     # 训练参数
results.hardware_info       # 硬件信息
```

---

### 2. `GridSearcher` - 搜索器类

适合需要更细粒度控制的场景，如多次搜索、生成 YAML 配置。

```python
from pdcostmodel import GridSearcher

# 创建搜索器
searcher = GridSearcher(
    model_name="qwen3-30b-a3b",       # 模型名称
    model_config=None,                 # 或提供自定义 ModelConfig
    hardware_config=None,              # 硬件配置 (默认自动校准)
    total_gpus=8,                      # 总 GPU 数
    node_count=1,                      # 节点数
    tp_candidates=[1, 2, 4, 8],        # TP 候选值
    pp_candidates=[1, 2, 4],           # PP 候选值
    ep_candidates=[1, 2, 4, 8],        # EP 候选值
    sharding_candidates=["stage1", "stage2"],
    auto_calibrate=True,               # 是否自动校准硬件
    require_precise=True,              # 是否要求精确校准
    verbose=True,
)

# 执行搜索
results = searcher.search(
    micro_batch_size=1,
    seq_len=8192,
    gradient_accumulation_steps=16,
    recompute_granularity="full",
    sort_by="throughput",
    top_k=10,
    print_report=True,
)
```

#### GridSearcher 方法

| 方法 | 说明 |
|------|------|
| `search(...)` | 执行配置搜索，返回 `GridSearchResult` |
| `generate_search_space()` | 生成合法的配置搜索空间，返回 `List[Dict]` |
| `generate_yaml_config(result, output_path)` | 生成训练配置 YAML 文件 |

---

### 3. `SearchResult` - 单个配置结果

每个配置的预测结果。

```python
result = results.best  # 或 results.results[0]

# 配置信息
result.rank                          # 排名
result.config                        # 配置字典 {"tp":1, "pp":1, "dp":8, "ep":8, "sharding":"stage1"}
result.config_str                    # 配置字符串 "TP1-PP1-DP8-EP8-Sharding(stage1)"

# 性能预测
result.step_time_ms                  # Step 时延 (ms)
result.memory_gb                     # 显存占用 (GB)
result.fits_memory                   # 是否满足显存约束
result.mfu                           # Model FLOPs Utilization
result.tokens_per_second             # 总吞吐量 (tok/s)
result.tokens_per_second_per_gpu     # 每卡吞吐量 (tok/s/GPU)

# 训练参数
result.micro_batch_size              # 每设备批次大小
result.seq_len                       # 序列长度
result.gradient_accumulation_steps   # 梯度累积步数
result.global_batch_size             # 全局批次大小
```

---

### 4. `GridSearchResult` - 完整搜索结果

```python
# 保存到 JSON
results.save_json("search_results.json")

# 转为字典
result_dict = results.to_dict()

# 打印报告
results.print_report(top_k=10)
```

#### 导出 JSON 格式

```json
{
  "model_name": "qwen3-30b-a3b",
  "total_gpus": 8,
  "total_configs": 38,
  "valid_configs": 18,
  "best": {
    "rank": 1,
    "config": {"tp": 1, "pp": 1, "dp": 8, "ep": 8, "sharding": "stage1"},
    "config_str": "TP1-PP1-DP8-EP8-Sharding(stage1)",
    "step_time_ms": 26447.15,
    "memory_gb": 64.02,
    "fits_memory": true,
    "mfu": 0.178,
    "tokens_per_second_per_gpu": 4956
  },
  "results": [...],
  "search_space": {...},
  "training_params": {...},
  "hardware_info": {...}
}
```

---

### 5. 生成训练配置 YAML

```python
from pdcostmodel import GridSearcher

searcher = GridSearcher(model_name="qwen3-30b-a3b", total_gpus=8)

# 方式 1: 自动搜索并生成最优配置
yaml_content = searcher.generate_yaml_config(output_path="best_config.yaml")

# 方式 2: 为指定配置生成 YAML
results = searcher.search()
yaml_content = searcher.generate_yaml_config(
    result=results.results[1],  # 使用第 2 名配置
    output_path="second_best.yaml"
)
```

#### 生成的 YAML 示例

```yaml
## qwen3-30b-a3b 自动生成配置 ##
## PDCost 预测吞吐量: 4956 tok/s/GPU ##

# 数据配置
train_dataset_type: erniekit
eval_dataset_type: erniekit
max_seq_len: 8192

# 模型配置
model_name_or_path: qwen3-30b-a3b
_attn_implementation: flashmask

# 训练配置
do_train: true
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
# global_batch_size: 128

# 并行配置
tensor_model_parallel_size: 1
sequence_parallel: false
pipeline_model_parallel_size: 1
use_expert_parallel: true
expert_model_parallel_size: 8

# 重计算配置
recompute_granularity: full
recompute_method: uniform
recompute_num_layers: 1

# Sharding 配置
sharding: stage1
split_param: true

# 优化器配置
optim: adamw
bf16: true
tensorwise_offload_optimizer: true
```

---

## 📋 完整使用示例

### 示例 1: 快速搜索

```python
from pdcostmodel import grid_search

# 搜索 Qwen3-30B-A3B 的最优 8 卡配置
results = grid_search("qwen3-30b-a3b", total_gpus=8)

# 输出最优配置
print(f"最优配置: {results.best.config_str}")
print(f"吞吐量: {results.best.tokens_per_second_per_gpu:.0f} tok/s/GPU")
```

### 示例 2: 自定义搜索空间

```python
from pdcostmodel import grid_search

# 限制搜索空间
results = grid_search(
    "qwen3-30b-a3b",
    total_gpus=32,
    seq_len=4096,
    micro_batch_size=2,
    tp_candidates=[2, 4, 8],  # 只考虑 TP >= 2
    pp_candidates=[1, 2, 4, 8],
    top_k=5,
)
```

### 示例 3: 使用自定义模型配置

```python
from pdcostmodel import grid_search, ModelConfig

# 创建自定义 Dense 模型配置 (类似 LLaMA 8B)
custom_model = ModelConfig(
    num_hidden_layers=32,
    hidden_size=4096,
    intermediate_size=14336,
    num_attention_heads=32,
    num_key_value_heads=8,
    num_experts=1,  # Dense 模型
    vocab_size=128256,
)

# 搜索
results = grid_search(custom_model, total_gpus=8, seq_len=4096)
```

### 示例 4: 比较不同序列长度

```python
from pdcostmodel import GridSearcher

searcher = GridSearcher(model_name="qwen3-30b-a3b", total_gpus=8, verbose=False)

print(f"{'序列长度':<12} {'最优配置':<35} {'吞吐量':<15} {'显存':<10}")
print("-" * 75)

for seq_len in [2048, 4096, 8192]:
    results = searcher.search(seq_len=seq_len, print_report=False)
    if results.best:
        print(f"{seq_len:<12} {results.best.config_str:<35} "
              f"{results.best.tokens_per_second_per_gpu:<15,.0f} "
              f"{results.best.memory_gb:<10.2f}")
```

### 示例 5: 导出完整结果

```python
from pdcostmodel import GridSearcher

searcher = GridSearcher(model_name="qwen3-30b-a3b", total_gpus=8)
results = searcher.search()

# 保存搜索结果
results.save_json("search_results.json")

# 生成最优配置 YAML
searcher.generate_yaml_config(output_path="best_config.yaml")
```

---

## 📦 支持的并行策略

| 并行策略 | 参数 | 说明 |
|---------|------|------|
| Tensor Parallel (TP) | `tp` | 张量并行，切分 Attention 和 MLP 权重 |
| Pipeline Parallel (PP) | `pp` | 流水线并行，切分 Transformer 层 |
| Data Parallel (DP) | `dp` | 数据并行，复制模型 (自动计算) |
| Expert Parallel (EP) | `ep` | 专家并行，切分 MoE 专家 |
| Sharding (ZeRO) | `sharding` | 优化器状态/梯度分片 (stage1/stage2) |

### 搜索空间约束

Grid Search 自动应用以下约束过滤无效配置：

- `TP × PP × DP = total_gpus` — GPU 数量必须完全分配
- `PP 整除层数` — 流水线并行要求每个 stage 层数相同
- `EP ≤ DP` — 专家并行是数据并行的子分组
- `EP ≤ 专家数` — EP 不能超过模型的专家数量
- `显存 ≤ GPU 容量` — OOM 配置自动过滤

---

## 📊 支持的预设模型

```python
from pdcostmodel import ModelConfig

# 使用预设模型
model = ModelConfig.from_name("qwen3-30b-a3b")
```

| 模型名称 | 类型 | 参数量 | 说明 |
|---------|------|--------|------|
| `qwen3-30b-a3b` | MoE | ~30B | Qwen3 MoE, 128 experts, top-8 |
| `qwen3-235b-a22b` | MoE | ~235B | Qwen3 大模型 |
| `deepseek-v3` | MoE | ~685B | DeepSeek V3 |
| `llama3-70b` | Dense | ~70B | LLaMA 3 70B |
| `llama3-8b` | Dense | ~8B | LLaMA 3 8B |

---

## 🔧 硬件校准

Grid Search 默认自动执行硬件校准。也可以手动控制：

```python
from pdcostmodel import GridSearcher

# 禁用自动校准
searcher = GridSearcher(
    model_name="qwen3-30b-a3b",
    total_gpus=8,
    auto_calibrate=False,  # 使用默认硬件参数
)

# 或手动校准
from pdcostmodel import auto_calibrate_or_load

hardware_config, is_from_cache = auto_calibrate_or_load(
    node_count=1,
    force_calibrate=False,   # 强制重新校准
    precise=True,            # 精确校准模式
    require_precise=True,    # 要求精确校准
    verbose=True,
)

searcher = GridSearcher(
    model_name="qwen3-30b-a3b",
    total_gpus=8,
    hardware_config=hardware_config,
    auto_calibrate=False,
)
```

---

## 🎯 预测精度参考

在 Qwen3-30B-A3B + H800 8卡环境下：

| 指标 | 预测误差 |
|------|----------|
| Step Time | ~5% |
| 吞吐量 (tok/s/GPU) | ~5% |
| 显存 (allocated) | ~10% |

---

## 💡 使用建议

1. **MoE 模型**: 
   - 优先使用 EP 并行
   - **必须使用 `full` 重计算** (默认已启用)

2. **显存不足**: 
   - 增加 Sharding 阶段 (`stage1` → `stage2`)
   - 默认已启用 `tensorwise_offload_optimizer`

3. **大规模搜索**: 
   - 先用小范围候选值探索趋势
   - 再针对有希望的区域细化搜索

4. **多节点训练**: 
   - PP 适合跨节点
   - TP 建议节点内使用

---

## 📖 API 快速参考

```python
from pdcostmodel import (
    # Grid Search (推荐)
    grid_search,           # 快速搜索函数
    GridSearcher,          # 搜索器类
    GridSearchResult,      # 搜索结果
    SearchResult,          # 单个配置结果
    
    # 配置类
    ModelConfig,           # 模型架构配置
    ParallelConfig,        # 并行配置
    HardwareConfig,        # 硬件配置
    
    # 代价模型
    PDCostModel,           # 代价模型主类
    PredictionResult,      # 预测结果
    
    # 硬件校准
    auto_calibrate_or_load,# 自动校准或加载
)
```

---

## 📝 运行示例

```bash
cd /root/paddlejob/workspace/env_run/zhangdongqi/PaddleFleet/auto_configurator

# 运行 Grid Search 示例
python example_pdcost_gridsearch.py
```

---

## ⚠️ 注意事项

1. 预测结果为理论估算值，建议用实际 benchmark 验证关键配置
2. 首次运行会自动执行硬件校准 (约 1-2 分钟)
3. **MoE 模型必须使用 `full` 重计算**
4. `tensorwise_offload` 只在 `dp > 1` 时有效
5. 通信时间预测假设理想网络条件