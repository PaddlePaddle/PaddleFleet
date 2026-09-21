# QK Stats Monitor

QK 注意力统计监控模块，覆盖 Q/K/V 激活向量尺度、attention logits、熵与 sink。

通过 Triton 优化的 Online Softmax 内核，在单次前向传播中高效计算注意力矩阵的统计特征，覆盖数值稳定性、注意力集中度和 attention sink 现象。

---

## 监控指标

### Q/K/V Vector Norms

对于 dense core-attention 实际接收的每个 token/head 向量，记录：

```text
{q,k,v}_norm_mean = mean(||q/k/v||₂)
{q,k,v}_norm_max  = max(||q/k/v||₂)
```

这组指标用于区分 Q/K 尺度增长与方向对齐造成的 logit 变化，并观察 V 分支的尺度漂移。CSA/HCA
稀疏调用只暴露组合后的 KV 表示，无法可靠拆出独立 V，因此不输出这组指标。

### 1. Max Logit (注意力 Logit 最大值)

**数学公式:**
```
max = max(Q × K^T / √d)
```

对注意力矩阵中所有有效位置（考虑因果 mask 和 padding）取全局最大值，然后在所有 head 之间取最大，再在 batch 之间取最大。

**跨 rank 聚合:** hook 内不做 TP collective；`gather_and_aggregate()` 在 flush/logging 阶段按 max 规则聚合。

**诊断意义:**
- 过大的值预示数值不稳定，可能导致 softmax 溢出
- 训练过程中应保持稳定，突增需要警惕

---

### 2. Mean Logit (注意力 Logit 均值)

**数学公式:**
```
mean = sum(valid_logits) / count(valid_positions)
```

对注意力矩阵中所有有效位置（排除因果 mask 的 `-inf` 位置）求均值，然后在所有 head 和 batch 之间取均值。

**跨 rank 聚合:** hook 内不做 TP collective；`gather_and_aggregate()` 在 flush/logging 阶段按 mean 规则聚合。

**诊断意义:**
- 反映注意力 logit 的基准量级
- 值过大或过小可能表示 Q/K 投影权重的量级异常

---

### 3. Entropy Avg (注意力熵均值)

**数学公式:**

标准 Shannon 熵：
```
H = -Σ(p_i × log(p_i))
```

其中 `p = softmax(Q × K^T / √d)` 为注意力概率分布。

**Triton 实现 (Online Softmax):**
```
entropy = log(L) - (1/L) × Σ((x_i - m) × exp(x_i - m))
```

其中 `m = max(logits)` 为行最大值，`L = Σ(exp(x_i - m))` 为 softmax 分母。该公式与标准 Shannon 熵数学等价，但通过 Online Softmax 可以单次遍历完成计算。

对每个 query 行计算熵后取均值，再在所有 head 和 batch 之间取均值。

**跨 rank 聚合:** hook 内不做 TP collective；`gather_and_aggregate()` 在 flush/logging 阶段按 mean 规则聚合。

**诊断意义:**
- **低熵**: 注意力集中在少数 token 上，可能存在注意力坍缩
- **高熵**: 注意力分布均匀，模型对当前位置的关注不够聚焦
- 正常训练过程中应随层深度变化呈现合理梯度

---

### 4. Sink (Attention Sink 权重)

**数学公式:**
```
sink = mean(softmax(logits)[..., 0])
```

即 attention 概率分布中分配给第一个 token (position 0) 的权重，对所有 query 行取均值。

**Triton 实现:**
```
sink_per_row = exp(x_0 - m) / L
```

其中 `x_0` 为 position 0 对应的 logit 值，通过 Online Softmax 状态变量 `s_i` 追踪。

**跨 rank 聚合:** hook 内不做 TP collective；`gather_and_aggregate()` 在 flush/logging 阶段按 mean 规则聚合。

**诊断意义:**
- **高 sink 值**: 出现 "attention sink" 现象 — 大量注意力被分配到第一个 token，而非语义相关的位置
- 是 LLM 训练中常见的现象，但过度 sink 可能影响模型质量

---

### 5. Entropy Min (最小熵 Head)

**数学公式:**
```
entropy_min = min(entropy_per_head)
```

从本 rank 的 local heads 中选取熵最小的 head，跨 rank 的全局最小值由 `gather_and_aggregate()` 在 flush/logging 阶段聚合。

**诊断意义:**
- 识别最"尖锐"的 head，即注意力最集中的 head
- 极低值可能表示某个 head 发生了注意力坍缩

---

### 6. Entropy Max (最大熵 Head)

**数学公式:**
```
entropy_max = max(entropy_per_head)
```

从本 rank 的 local heads 中选取熵最大的 head，跨 rank 的全局最大值由 `gather_and_aggregate()` 在 flush/logging 阶段聚合。

**诊断意义:**
- 识别最"分散"的 head，即注意力最均匀的 head
- 极高值可能表示某个 head 未学到有效的注意力模式

---

### 7. Sink Head Ratio (Sink Head 占比)

**数学公式:**
```
sink_head_ratio = count(sink_per_head > threshold) / num_heads
```

被分类为 "sink head" 的 head 占总 head 数的比例。默认阈值 0.3。

**理论基础:**
Sun et al. (2026, arXiv:2603.05498) 证明 attention sinks 是 per-head 现象：
某些 head 将 >30% 注意力分配给 token-0，作为 "learned gate" 来关闭不需要的信息通道。

**诊断意义:**
- **0.0 ~ 0.3**: 正常，少数 head 有 sink 行为
- **0.3 ~ 0.6**: 显著 sink 行为，模型在积极使用 implicit gating
- **> 0.6**: 大部分 head 都是 sink head，可能影响长上下文能力

---

### 8. Sink Head Max (最强 Sink Head 权重)

**数学公式:**
```
sink_head_max = max(sink_per_head)
```

所有 head 中对 token-0 分配的最大平均注意力权重。

**诊断意义:**
- 识别最极端的 sink head
- 值接近 1.0 说明该 head 几乎完全 "dormant"（所有注意力给 token-0）
- 可用于定位 pruning 候选 head

---

### 9. Sink Non-Sink Gap (Sink/非Sink Gap)

**数学公式:**
```
sink_nonsink_gap = mean(sink_heads_weight) - mean(nonsink_heads_weight)
```

Sink heads 和非 sink heads 在 token-0 注意力权重上的差异。

**理论基础:**
这是 "logit gap" 的代理指标。Sun et al. 发现 sink head 中 k^(s) 和 q^(n) 的子空间
比非 sink head 更接近（Figure 6），产生持续的 logit gap。

**诊断意义:**
- 高 gap = sink/非sink 分化明显，模型在清晰地区分两类 head
- 低 gap = 所有 head 行为相似，没有强 sink 分化
- Gap 上升趋势 = 模型在训练中逐步强化 sink 策略


## Triton 内核说明

### Online Softmax 算法

QK 统计的核心是 `qk_stats_kernel`，采用 Online Softmax 算法在单次遍历中同时计算 max logit、mean logit、entropy 和 sink weight，避免了完整注意力矩阵的存储。

每个 Triton program 处理一个 `(batch, head)` 对。通过以下状态变量实现增量更新：

| 状态变量 | 含义 |
|----------|------|
| `m_i` | 当前行的最大 logit |
| `d_i` | 当前行的 softmax 分母 `Σexp(x-m)` |
| `h_i` | 当前行的熵分子项 `Σ(x-m)·exp(x-m)` |
| `s_i` | 当前行的 sink 分子项 `exp(x₀-m)` |

当遇到新的 Key block 时，按以下公式更新 (设新最大值 `m_new = max(m_i, block_max)`)：
```
α_prev = exp(m_i - m_new)
α_curr = exp(logits - m_new)

d_new = d_i × α_prev + Σ(α_curr)
h_new = h_i × α_prev + (m_i - m_new) × d_i × α_prev + Σ((logits - m_new) × α_curr)
s_new = s_i × α_prev + α_curr[:, 0]    (仅当 block 包含 position 0)
```

最终：
```
entropy = log(d_i) - h_i / d_i
sink    = s_i / d_i
```

### GQA 处理

当 `num_q_heads ≠ num_k_heads` 时（Grouped Query Attention），K 通过 `repeat_interleave` 扩展以匹配 Q 的 head 数量。

### 因果 Mask

默认启用因果 mask (`causal=True`)，适用于 GPT 类自回归模型。注意力矩阵中 `query_pos < key_pos` 的位置被设为 `-1e10`。

### THD Packed 序列

当一个 batch 内多条变长序列被打包成单条 `[T, H, D]` 张量（THD 布局，`total_tokens = ΣLᵢ`）时，`core_attention` 通过 `packed_seq_params.cu_seqlens_q`（回退 `cu_seqlens_q_padded`）携带各序列边界。此时 QK monitor 走 **packed 专用路径** `compute_qk_stats_packed` → `qk_stats_packed_kernel`（split-M，grid `(num_heads, num_m_blocks)`）：

- **逐序列边界**：`_cu_seqlens_to_token_arrays` 用 `torch.searchsorted` 在 GPU 上把 `cu_seqlens` 展开成每 token 的 `seq_start`/`seq_end`（int32），全程无 `.item()`/`.cpu()`/`.tolist()`，可安全在 forward hook 内调用而不破坏 compute/comm overlap。kernel 内用 `same_seq = seq_start[m] ≤ n < seq_end[m]` 屏蔽跨序列注意力，**不会跨序列泄露**。
- **逐序列 sink**：sink token 是每条序列自己的首 token（`seq_start[m]` 列），而非全局 token-0。因此每条 packed 序列各自独立统计 attention sink。
- **padding 安全**：当张量的 `T` 维大于 `cu_seqlens[-1]`（padding slot）时，这些 token 的 `seq_start == seq_end == 0`，`same_seq` 使其有效范围为空，不贡献任何统计，也不会越界读取。
- **mean 语义**：与 dense split-M kernel 一致，采用 **row-first 均值**（先算每行均值，再对全体有效行取平均），而非按有效位置计数加权。

> 注：packed 路径当前不支持 SWA / 滑动窗口与 CP（`q_row_offset` / 非对称 Q/K 长度）的组合；如需叠加，须同时补充对应的正确性测试。

---

## 使用方式

### 基本用法

```python
from paddlefleet.internal_medicine import setup_internal_medicine

# 设置监控
monitor_dict = {}
model = setup_internal_medicine(
    model,
    monitors=["qk_stats"],
    monitor_dict=monitor_dict,
    monitor_interval=10,
    qk_stats={
        "causal": True,
        "use_triton": True,
        "log_per_layer": True,
        "log_global": True,
    },
)

monitor = monitor_dict["qk_stats"]

# 训练循环中
for step in range(num_steps):
    loss = model(inputs)
    loss.backward()
    optimizer.step()
    monitor.step()  # 必须在每步结束时调用
```

### 配置选项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `causal` | `True` | 是否应用因果 mask |
| `use_triton` | `True` | 使用 Triton 内核；需要 CUDA/Triton。设为 `False` 时使用 PyTorch reference path |
| `log_per_layer` | `True` | 记录每层指标 (`qk_stats/layer_{idx}/{metric}`) |
| `log_global` | `True` | 记录全局指标 (`qk_stats/global_{metric}`) |
| `monitor_interval` | `1` | 监控间隔 (步数) |
| `verbose` | `False` | 打印调试信息 |

### 日志格式

**Per-layer 指标:**
```
qk_stats/layer_0/max
qk_stats/layer_0/mean
qk_stats/layer_0/entropy_avg
qk_stats/layer_0/sink
qk_stats/layer_0/entropy_min
qk_stats/layer_0/entropy_max
qk_stats/layer_0/sink_head_ratio
qk_stats/layer_0/sink_head_max
qk_stats/layer_0/sink_nonsink_gap
...
```

**Global 指标:**
```
qk_stats/global_max
qk_stats/global_mean
qk_stats/global_entropy_avg
qk_stats/global_sink
qk_stats/global_entropy_min
qk_stats/global_entropy_max
qk_stats/global_sink_head_ratio
qk_stats/global_sink_head_max
qk_stats/global_sink_nonsink_gap
```

---

## Hook 挂载点

| Hook 位置 | 类型 | 捕获内容 | 产出指标 |
|-----------|------|----------|----------|
| `attention.core_attention` | forward **pre**-hook | `args[0]` = Query, `args[1]` = Key；`kwargs["packed_seq_params"]` = packed 边界（若有） | 全部 9 个指标 |

注意: 使用 `forward_pre_hook` 而非 `forward_hook`，直接拦截送入 `core_attention` 的 Q/K 张量。hook 以 `with_kwargs=True` 注册，以便读取 `packed_seq_params`：当 Q 为 3 维（`[T, H, D]`）且带 `cu_seqlens_q` 时走 THD packed 路径（见上文「THD Packed 序列」），否则走常规 dense 路径。

---

## 跨 rank 聚合机制

QK hook 内不做 `all_reduce` / `all_gather`。Hook 只把本 rank 的 GPU 0-dim 标量写入 `TorchProbe` 的 GPU buffer，`monitor.step()` flush 到 `training_logs` 后，再由 `training_logs.gather_and_aggregate()` 在 logging 阶段做跨 rank 聚合。

| 指标 | 跨 rank 聚合方式 |
|------|------------------|
| `max`, `entropy_max`, `sink_head_max` | `max(all_ranks)` |
| `entropy_min` | `min(all_ranks)` |
| `mean`, `entropy_avg`, `sink`, `sink_head_ratio`, `sink_nonsink_gap` | `mean(all_ranks)` |

这样避免在 attention hot path 中插入 NCCL collective。Mean 类指标假设各 rank 观察到的样本数一致；标准 TP head split 满足这个假设。

## 性能注意

QK stats 需要额外遍历 Q/K 并计算 attention-score 级别统计，复杂度接近 `O(seq_len²)`。长序列、大 GBS 或所有 layer 全开时开销会很明显。生产训练建议提高 `monitor_interval`，或只在需要诊断 QK/sink 问题时启用 `qk_stats`。
