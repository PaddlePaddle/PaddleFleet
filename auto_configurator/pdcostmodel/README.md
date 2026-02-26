# pdcost - PaddleFormers 分布式训练代价模型

`pdcost` 是一个用于预测 PaddleFormers 分布式训练性能的工具，可以在实际运行前估算不同并行配置下的：

- **Step 时间** (训练迭代耗时，已校准 seq_len 阈值效应)
- **显存占用** (支持双指标: allocated + reserved)
- **硬件利用率** (MFU)
- **训练吞吐量** (tokens/s/GPU)

## ✨ 特性亮点

- 🎯 **高精度预测**: Step time 误差 ~5%，显存误差 ~10%
- 📊 **双指标显存**: 同时预测 `allocated` (实际分配) 和 `reserved` (框架预留)
- 🔧 **seq_len 校准**: 内置阈值效应模型，准确处理不同序列长度
- 🔍 **配置搜索**: 自动搜索最优并行配置，支持 OOM 过滤
- ⚡ **MoE 专用**: 针对 Qwen3 MoE 等稀疏模型优化
- 🔬 **硬件校准**: 通过实测 GEMM benchmark 校准 GPU 算力

---

## 📁 模块结构

```
pdcost/
├── __init__.py          # 主入口，导出所有公共 API
├── config.py            # 配置类定义
│   ├── ModelConfig      # 模型架构配置
│   ├── ParallelConfig   # 并行策略配置
│   ├── TrainingConfig   # 训练配置
│   ├── HardwareConfig   # 硬件配置
│   ├── GPUSpec          # GPU 规格
│   └── NetworkSpec      # 网络规格
├── memory_model.py      # 显存预测模型
│   ├── MemoryModel      # 显存估算主类
│   ├── MemoryBreakdown  # 显存分解结果
│   ├── ShardingConfig   # Sharding 配置
│   └── RecomputeConfig  # 重计算配置
├── compute_model.py     # 计算时间预测模型
│   ├── ComputeModel     # 计算时间估算
│   └── LayerProfile     # 层计算 Profile
├── comm_model.py        # 通信时间预测模型
│   ├── CommModel        # 通信时间估算
│   └── CommResult       # 通信预测结果
├── calibration.py       # 硬件校准模块
│   ├── HardwareCalibrator   # 硬件校准器
│   ├── CalibrationResult    # 校准结果
│   ├── PerformanceCurve     # 性能曲线
│   ├── quick_calibrate      # 快速校准函数
│   └── create_calibrated_hardware_config  # 创建校准后配置
├── costmodel.py         # 主 CostModel 类
│   ├── PDCostModel      # 代价模型主类
│   └── PredictionResult # 预测结果
├── search_configs.py    # 配置搜索脚本
└── examples/            # 使用示例
    ├── basic_usage.py
    └── paddleformers_optimization.py
```

---

## 📦 支持的并行策略

| 并行策略 | 参数 | 说明 |
|---------|------|------|
| Tensor Parallel (TP) | `tp` | 张量并行，切分 Attention 和 MLP 权重 |
| Pipeline Parallel (PP) | `pp` | 流水线并行，切分 Transformer 层 |
| Data Parallel (DP) | `dp` | 数据并行，复制模型 |
| Expert Parallel (EP) | `ep` | 专家并行，切分 MoE 专家 |
| Sharding (ZeRO) | `sharding` | 优化器状态/梯度/参数分片 (stage1/2/3) |
| Sequence Parallel (SP) | `sp` | 序列并行，配合 TP 使用 |
| Context Parallel (CP) | `cp` | 上下文并行，切分序列长度 |

---

## 🚀 快速开始

### 最简用法

```python
from pdcost import ModelConfig, PDCostModel, ParallelConfig
from pdcost.config import HardwareConfig, GPUSpec

# 1. 加载模型配置
model = ModelConfig.from_json('Qwen3-30B-A3B-Base/config.json')

# 2. 定义硬件配置
hardware = HardwareConfig(
    gpu=GPUSpec(name='H800', memory_gb=79.6, bf16_tflops=788.0),
    num_nodes=1, gpus_per_node=8
)

# 3. 创建代价模型 (TrainingConfig 可选，有默认值)
costmodel = PDCostModel(model, hardware)

# 4. 预测 (训练参数直接传入 predict_calibrated)
parallel = ParallelConfig(tp=1, pp=1, dp=8, ep=8, sharding='stage1')
result = costmodel.predict_calibrated(
    parallel,
    seq_len=8192,                        # 序列长度
    micro_batch_size=1,                  # 每卡 batch size
    gradient_accumulation_steps=16,      # 梯度累积
    recompute_granularity='full',        # 重计算策略
    tensorwise_offload_optimizer=True    # 优化器 offload
)

# 5. 读取结果
print(f"吞吐量: {result.tokens_per_second_per_gpu:.0f} tok/s/GPU")
print(f"Step 时间: {result.step_time_ms/1000:.2f} s")
print(f"Allocated: {result.memory_breakdown.allocated_memory_gb:.2f} GB")
print(f"Reserved: {result.memory_breakdown.reserved_memory_gb:.2f} GB")
print(f"可运行: {'✅' if result.fits_memory else '❌ OOM'}")
```

### 使用预设模型

```python
from pdcost import PDCostModel, ModelConfig, ParallelConfig
from pdcost.config import HardwareConfig, GPUSpec

# 使用预设模型 (支持 qwen3-30b-a3b, llama3-70b, deepseek-v3 等)
model = ModelConfig.from_name("qwen3-30b-a3b")
hardware = HardwareConfig(
    gpu=GPUSpec(name='H800', memory_gb=79.6, bf16_tflops=788.0),
    num_nodes=1, gpus_per_node=8
)
costmodel = PDCostModel(model, hardware)

parallel = ParallelConfig(tp=1, pp=1, dp=8, ep=8, sharding="stage1")
result = costmodel.predict_calibrated(
    parallel, seq_len=8192, micro_batch_size=1,
    gradient_accumulation_steps=16,
    recompute_granularity='full',
    tensorwise_offload_optimizer=True
)
print(f"吞吐量: {result.tokens_per_second_per_gpu:.0f} tok/s/GPU")
```

> 💡 **提示**: 
> - `TrainingConfig` 是可选的，训练参数可直接传给 `predict_calibrated()`
> - 推荐使用 `predict_calibrated()` 而非 `predict()`，前者包含 seq_len 阈值效应校准
> - MoE 模型建议使用 `recompute_granularity='full'`

---

## 🔧 硬件校准

pdcost 支持通过 GEMM benchmark 测试实际 GPU 算力和显存带宽，自动校准硬件参数，提高预测精度。

### 快速校准

一步完成校准并创建可直接使用的 `HardwareConfig`：

```python
from pdcost.calibration import create_calibrated_hardware_config

# 运行 benchmark + 创建 HardwareConfig
hardware = create_calibrated_hardware_config(
    num_nodes=1,
    gpus_per_node=8,
    device_id=0,
    verbose=True
)

```

### 详细校准（含性能曲线）

如果需要更精细的控制或获取多尺寸性能曲线：

```python
from pdcost.calibration import HardwareCalibrator

calibrator = HardwareCalibrator(
    device_id=0,       # 测试 GPU ID
    warmup_iters=5,    # 预热次数
    test_iters=20      # 测试次数
)

# 完整校准（含多尺寸性能曲线）
result = calibrator.calibrate(
    test_compute=True,      # 测试算力
    test_memory=True,       # 测试显存带宽
    gemm_size=8192,         # GEMM 峰值测试矩阵大小
    multi_size_test=True,   # 多尺寸测试生成性能曲线
    test_sizes=[64, 128, 256, 512, 1024, 2048, 4096, 8192],
    verbose=True
)

# 创建校准后的 HardwareConfig
hardware = calibrator.create_hardware_config(num_nodes=1, gpus_per_node=8)
```

### 仅查看校准数据

如果只想查看校准结果而不创建配置：

```python
from pdcost.calibration import quick_calibrate

result = quick_calibrate(device_id=0, verbose=True)
print(result)
# CalibrationResult:
#   GPU: NVIDIA H800 × 8
#   Memory: 79.6 GB
#   FP32: 51.6 TFLOPS
#   FP16: 763.0 TFLOPS
#   BF16: 788.0 TFLOPS
#   Memory BW: 2781.8 GB/s
```

### 校准内容

| 测试项 | 说明 |
|--------|------|
| GPU 检测 | 自动检测 GPU 名称、显存、数量 |
| 算力测试 | FP32/FP16/BF16 大矩阵 GEMM |
| 带宽测试 | 显存读写带宽 |
| 性能曲线 | 多尺寸 GEMM 拟合效率曲线（详细校准） |
| 网络估算 | 根据 GPU 型号估算 NVLink/IB 带宽 |

---

## 📋 配置类详解

### ModelConfig - 模型架构配置

```python
ModelConfig(
    num_hidden_layers=48,       # Transformer 层数
    hidden_size=2048,           # 隐藏维度
    intermediate_size=6144,     # FFN 中间维度
    num_attention_heads=32,     # 注意力头数
    num_key_value_heads=4,      # KV 头数 (GQA)
    head_dim=64,                # 每个头的维度
    num_experts=128,            # MoE 专家数
    num_experts_per_tok=8,      # Top-K
    moe_intermediate_size=768,  # 专家 FFN 维度
    vocab_size=151936,          # 词表大小
)

# 从 config.json 加载
model = ModelConfig.from_json('Qwen3-30B-A3B-Base/config.json')

# 使用预设模型
model = ModelConfig.from_name("qwen3-30b-a3b")
```

### ParallelConfig - 并行配置

```python
ParallelConfig(
    tp=1,                # 张量并行度
    pp=1,                # 流水线并行度
    dp=8,                # 数据并行度
    ep=8,                # 专家并行度
    sharding="stage1",   # ZeRO 阶段: none/stage1/stage2/stage3
    sp=False,            # 序列并行
    cp=1,                # 上下文并行度
)
```

### TrainingConfig - 训练配置

```python
TrainingConfig(
    micro_batch_size=1,              # 每卡 batch size
    sequence_length=8192,            # 序列长度
    gradient_accumulation_steps=64,  # 梯度累积
    dtype="bfloat16",                # 数据类型: float32/float16/bfloat16
    recompute_granularity="full",    # 重计算: none/selective/full
    amp_master_grad=True,            # 混合精度 master grad
)
```

### HardwareConfig - 硬件配置

```python
HardwareConfig(
    gpu=GPUSpec(
        name="H800",
        memory_gb=79.6,
        bf16_tflops=788.0,
        fp16_tflops=788.0,
        fp32_tflops=51.0,
        memory_bandwidth_gbps=2800.0
    ),
    network=NetworkSpec(
        intra_node_bandwidth_gbps=900.0,  # NVLink
        inter_node_bandwidth_gbps=200.0,  # IB
    ),
    num_nodes=1,
    gpus_per_node=8,
)

# 使用预设 GPU
from pdcost.config import GPUSpec
gpu = GPUSpec.from_name("H100-80GB-HBM3")  # 支持 H100, A100, A800, V100 等
```

---

## 📈 预测函数参数

### predict_calibrated() - 校准预测（推荐）

```python
result = costmodel.predict_calibrated(
    parallel,                           # ParallelConfig: 并行配置
    seq_len=8192,                       # 序列长度
    micro_batch_size=1,                 # 每卡 batch size
    gradient_accumulation_steps=64,     # 梯度累积步数
    recompute_granularity="full",       # 重计算粒度: "none", "selective", "full"
    tensorwise_offload_optimizer=True,  # 是否启用 tensorwise 优化器 offload
    tensorwise_offload_ratio=0.95,      # offload 比例 (默认 95%)
)
```

### 关键参数说明

| 参数 | 说明 | 影响 |
|------|------|------|
| `seq_len` | 序列长度 | 影响激活显存和计算时间 |
| `micro_batch_size` | 每卡 batch size | 影响显存和吞吐量 |
| `gradient_accumulation_steps` | 梯度累积 | 影响全局 batch 和 step 时间 |
| `recompute_granularity` | 重计算策略 | `none` 不重计算；`full` 全部重计算，激活显存最低 |
| `tensorwise_offload_optimizer` | Tensorwise 优化器 offload | 优化器状态动态 offload 到 CPU（需要 dp > 1） |

### 重计算策略对 MoE 模型的影响

| granularity | 激活显存因子 | 说明 |
|-------------|-------------|------|
| `none` | 1.0 | 不重计算，显存最高 |
| `selective` | 1.0 (MoE) / 0.6 (Dense) | **对 MoE 几乎无效**，只重计算 attention |
| `full` | 0.15 | 全部重计算，显存最低 |


---

## 📊 预测结果 (PredictionResult)

```python
result = costmodel.predict_calibrated(parallel, ...)

# 时延指标
result.step_time_ms          # 总 step 时间 (ms)
result.compute_time_ms       # 计算时间 (ms)
result.total_comm_time_ms    # 通信时间 (ms)
result.bubble_time_ms        # 流水线气泡 (ms)

# 显存指标
result.memory_gb             # 总显存 (GB)
result.memory_breakdown      # 详细显存分解
result.fits_memory           # 是否满足显存约束

# 效率指标
result.mfu                   # Model FLOPs Utilization
result.compute_efficiency    # 计算效率

# 吞吐量
result.tokens_per_second     # 总吞吐量 (tok/s)
result.tokens_per_second_per_gpu  # 每卡吞吐量 (tok/s/GPU)
```

---

## 💾 显存分解 (MemoryBreakdown)

```python
breakdown = result.memory_breakdown

# 主要组成
breakdown.parameter_memory_gb       # 参数显存
breakdown.gradient_memory_gb        # 梯度显存
breakdown.optimizer_memory_gb       # 优化器状态显存
breakdown.activation_memory_gb      # 激活值显存
breakdown.communication_buffer_gb   # 通信缓冲区
breakdown.temporary_buffer_gb       # 临时缓冲区 (含 logits FP32 转换)
breakdown.framework_overhead_gb     # 框架基础开销

# 双指标显存 (PaddleFormers 特有)
breakdown.allocated_memory_gb       # 实际分配显存
breakdown.reserved_memory_gb        # 预留显存 (含激活缓冲池)
breakdown.activation_buffer_pool_gb # 框架激活缓冲池
```

### 双指标显存说明

PaddleFormers 框架有两个显存指标：
- **allocated**: 实际分配的显存，包括参数、梯度、优化器、激活等
- **reserved**: 框架预留的显存池，包括 allocated + 激活缓冲池

```python
# 示例
result = costmodel.predict_calibrated(parallel, seq_len=4096, ...)
mb = result.memory_breakdown

print(f"Allocated: {mb.allocated_memory_gb:.2f} GB")  # ~52.91 GB
print(f"Reserved: {mb.reserved_memory_gb:.2f} GB")   # ~58.71 GB
```

---

## 🔍 配置搜索

### 使用 search_configs.py 搜索最优配置

```python
from pdcost.search_configs import search_all_configs

# 搜索所有可运行配置
search_all_configs(
    model_config_path='Qwen3-30B-A3B-Base/config.json',
    total_gpus=8,
    gpu_memory_gb=79.6,
    output_file='all_runnable_configs.json'
)
```

### 手动搜索示例

```python
from pdcost import ModelConfig, PDCostModel, ParallelConfig
from pdcost.config import TrainingConfig, HardwareConfig, GPUSpec

model = ModelConfig.from_json('Qwen3-30B-A3B-Base/config.json')
hardware = HardwareConfig(
    gpu=GPUSpec(name='H800', memory_gb=79.6, bf16_tflops=788.0),
    num_nodes=1, gpus_per_node=8
)
training = TrainingConfig(micro_batch_size=1, sequence_length=8192, dtype='bfloat16')
costmodel = PDCostModel(model, hardware, training)

# 搜索空间
configs = []
for tp in [1, 2, 4, 8]:
    for pp in [1, 2, 4, 8]:
        if 8 % (tp * pp) != 0:
            continue
        dp = 8 // (tp * pp)
        for ep in [1, 2, 4, 8]:
            for seq_len in [4096, 8192]:
                parallel = ParallelConfig(tp=tp, pp=pp, dp=dp, ep=ep, sharding='stage1')
                result = costmodel.predict_calibrated(
                    parallel, seq_len=seq_len, micro_batch_size=1,
                    gradient_accumulation_steps=64,
                    recompute_granularity='full',
                    tensorwise_offload_optimizer=True
                )
                if result.fits_memory:
                    configs.append({
                        'tp': tp, 'pp': pp, 'dp': dp, 'ep': ep,
                        'seq': seq_len,
                        'tok_s': result.tokens_per_second_per_gpu,
                        'mem': result.memory_breakdown.reserved_memory_gb
                    })

# 按吞吐量排序
configs.sort(key=lambda x: x['tok_s'], reverse=True)
for c in configs[:5]:
    print(f"tp={c['tp']}, pp={c['pp']}, dp={c['dp']}, ep={c['ep']}, "
          f"seq={c['seq']}: {c['tok_s']:.0f} tok/s/GPU, {c['mem']:.1f} GB")
```

### 搜索空间约束

配置搜索会自动过滤无效配置：
- `tp * pp * dp == total_gpus` (GPU 数量约束)
- `ep <= num_experts` 且 `ep` 整除专家数
- 显存不超过 GPU 容量 (OOM 过滤)
- `tensorwise_offload` 需要 `dp > 1` (Sharding 约束)

---

## 📖 YAML 配置参数映射

从 PaddleFormers YAML 配置到 pdcost 参数的映射：

| YAML 参数 | pdcost 参数 | 说明 |
|-----------|-------------|------|
| `per_device_train_batch_size` | `micro_batch_size` | 每卡 batch size |
| `max_seq_len` | `seq_len` | 序列长度 |
| `gradient_accumulation_steps` | `gradient_accumulation_steps` | 梯度累积 |
| `tensor_model_parallel_size` | `tp` | 张量并行度 |
| `pipeline_model_parallel_size` | `pp` | 流水线并行度 |
| `expert_model_parallel_size` | `ep` | 专家并行度 |
| `sharding: stage1/stage2` | `sharding='stage1'/'stage2'` | Sharding 阶段 |
| `recompute_granularity` | `recompute_granularity` | 重计算粒度 |
| `tensorwise_offload_optimizer` | `tensorwise_offload_optimizer` | 优化器 offload |
| `bf16: true` | `dtype='bfloat16'` | 数据类型 |

---

## 📊 支持的预设模型

| 模型名称 | 类型 | 参数量 | 说明 |
|---------|------|--------|------|
| `qwen3-30b-a3b` | MoE | ~30B | Qwen3 MoE, 128 experts, top-8 |
| `qwen3-235b-a22b` | MoE | ~235B | Qwen3 大模型 |
| `deepseek-v3` | MoE | ~685B | DeepSeek V3 |
| `llama3-70b` | Dense | ~70B | LLaMA 3 70B |
| `llama3-8b` | Dense | ~8B | LLaMA 3 8B |

---

## 🎯 预测精度参考

在 Qwen3-30B-A3B + H800 8卡环境下的预测精度：

| 指标 | 预测误差 |
|------|----------|
| Step Time | ~5% |
| 吞吐量 (tok/s/GPU) | ~5% |
| Allocated 显存 | ~10% |
| Reserved 显存 | ~8% |

---

## 💡 使用建议

1. **MoE 模型**: 
   - 优先使用 EP 并行，通常 `ep = min(num_experts, total_gpus)`
   - **必须使用 `full` 重计算**，`selective` 对 MoE 几乎无效

2. **显存不足**: 
   - 增加 Sharding 阶段 (`stage1` → `stage2` → `stage3`)
   - 开启 `tensorwise_offload_optimizer=True`（需要 dp > 1）
   - 使用 `full` 重计算

3. **大序列长度**: 
   - 考虑使用 Context Parallel (CP) 或 Sequence Parallel (SP)
   - 注意 `seq_len > 4096` 会显著增加激活缓冲池

4. **多节点训练**: 
   - PP 适合跨节点
   - TP 建议节点内使用

5. **硬件校准**:
   - 建议首次使用时执行 `quick_calibrate()` 获取准确的硬件参数

---

## 📝 运行示例

```bash
cd /root/paddlejob/workspace/env_run/zhangdongqi/costmodel

# 运行基础示例
python pdcost/examples/basic_usage.py

# 运行 PaddleFormers 优化示例
python pdcost/examples/paddleformers_optimization.py

# 搜索最优配置
python pdcost/search_configs.py
```

---

## ⚠️ 注意事项

1. 预测结果为理论估算值，实际性能受多种因素影响
2. 建议在少量配置上进行实际 benchmark 验证
3. 通信时间预测假设理想的网络条件
4. **MoE 模型必须使用 `full` 重计算**，`selective` 对 MoE 基本无效
5. `tensorwise_offload` 只在 `dp > 1` 时有效

---

## 📖 API 快速参考

### 主要类

```python
from pdcost import (
    # 配置类
    ModelConfig,           # 模型架构配置
    TrainingConfig,        # 训练配置
    ParallelConfig,        # 并行配置
    HardwareConfig,        # 硬件配置
    GPUSpec,               # GPU 规格
    NetworkSpec,           # 网络规格
    
    # 子模型
    MemoryModel,           # 显存预测
    MemoryBreakdown,       # 显存分解
    ComputeModel,          # 计算预测
    CommModel,             # 通信预测
    
    # 主模型
    PDCostModel,           # 代价模型主类
    PredictionResult,      # 预测结果
    
    # 校准
    HardwareCalibrator,    # 硬件校准器
    CalibrationResult,     # 校准结果
    quick_calibrate,       # 快速校准
    create_calibrated_hardware_config,  # 创建校准配置
)
```