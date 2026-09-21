# PLE Health Monitor

PLE (Per-Layer Embedding) 健康监控模块，监控 7 个核心指标。

PLE 采用双分支架构：
- **Token 分支**: `embed_tokens_per_layer` 输出的逐层 token embedding
- **Projection 分支**: hidden states 经 `per_layer_projection_norm` (RMSNorm) 归一化后的投影

两条分支合并后作为每层 `PLESubmodule` 的输入。

---

## 监控指标

### 全局指标 (Model-level)

以下指标在 `ErnieDevGPTModel` 的 post-hook 中计算，仅产生全局日志。

---

#### 1. Token PLE Norm (Token 分支范数)

**数学公式:**
```
token_ple_norm = mean(||token_ple||₂)
```

其中对 `[S, B, L, H_ple]` 张量在最后一维 (`H_ple`) 计算 L2 范数，然后对所有位置求均值。

**数据来源:** `embed_tokens_per_layer` (VocabParallelEmbedding) post-hook

**数据流:**
```
embed_tokens_per_layer output: [B, S, L*H_ple]
    → transpose + reshape → [S, B, L, H_ple]
    → norm(dim=-1).mean()
```

**诊断意义:**
- 监控 token embedding 分支的信号强度
- 值过小说明 token 分支对 PLE 输入贡献不足
- 应与 `proj_ple_norm` 量级匹配

---

#### 2. Proj PLE Norm (投影分支范数)

**数学公式:**
```
proj_ple_norm = mean(||proj_ple × H^{-0.5}||₂)
```

其中 `proj_ple` 为 `per_layer_projection_norm` (RMSNorm) 的输出，`H` 为 `hidden_size`。

**数据来源:** `per_layer_projection_norm` (WrappedTorchNorm) post-hook

**数据流:**
```
per_layer_projection_norm output: [S, B, L, H_ple] (RMSNorm 后)
    → × H^{-0.5} (缩放)
    → norm(dim=-1).mean()
```

**诊断意义:**
- 监控投影分支的信号强度
- 应与 `token_ple_norm` 量级大致匹配，两条支路量级失衡可能影响训练效果

---

#### 3. Per-Layer Inputs Norm (合并信号范数)

**数学公式:**
```
per_layer_inputs_norm = mean(||(token_ple + proj_ple) × 2^{-0.5}||₂)
```

`2^{-0.5}` 为合并后的缩放因子。

**数据来源:** ErnieDevGPTModel post-hook，组合 token 和 projection 两个缓冲区

**诊断意义:**
- 监控实际送入每层 PLESubmodule 的输入信号量级
- 值异常（过大或过小）可能导致训练不稳定

---

#### 4. Token-Proj Cosine (双分支余弦相似度)

**数学公式:**
```
token_proj_cosine = mean(cosine_similarity(token_flat, proj_flat, dim=-1))
```

将 `token_ple` 和 `proj_ple` 从 `[S, B, L, H_ple]` 展平为 `[S*B*L, H_ple]` 后，在 `H_ple` 维度计算余弦相似度，再对所有位置求均值。

**数据来源:** ErnieDevGPTModel post-hook

**诊断意义:**
- **接近 1.0**: 两条分支信息高度冗余，PLE 双分支设计未发挥作用
- **接近 0**: 两条分支互补，PLE 设计有效
- 理想值应显著低于 1.0

---

### 逐层指标 (Layer-level)

以下指标在每个 `PLESubmodule` 的 post-hook 中计算，同时产生逐层和全局日志。

---

#### 5. Residual Ratio (残差贡献比)

**数学公式:**
```
residual_ratio = ||output - hidden_states|| / ||hidden_states||
```

其中 `hidden_states` 为 PLESubmodule 的输入 `[S, B, H]`，`output` 为 PLESubmodule 的输出 (`hidden_states + post_norm(down_out)`)。分母 clamp 最小值 `1e-8` 防止除零。

**数据来源:** PLESubmodule post-hook (`inputs[0]` 和 `output`)

**诊断意义:**
- 表征 PLE 对每层输出的"修改幅度"
- **过小 (→ 0)**: PLE 几乎没有作用，退化为恒等映射
- **过大**: PLE 贡献过度，可能导致训练不稳定
- 应保持在适中范围

---

#### 6. Gate Activation Mean (门控激活均值)

**数学公式:**
```
gate_activation_mean = mean(|act_fn(gate_out)|)
```

其中 `gate_out` 为 `PLESubmodule.gate_proj` 的原始输出 `[S, B, H_ple]`，`act_fn` 为激活函数 (GELU 或 SiLU)。

**数据来源:**
- `PLESubmodule.gate_proj` post-hook 缓存 `gate_out`
- PLESubmodule post-hook 消费缓存并计算

**诊断意义:**
- 反映 PLE 门控路径的活跃程度
- **过小**: 门控信号弱，PLE 的 gated MLP 选择性失效
- 应保持非零且活跃

---

#### 7. Gate Sparsity (门控稀疏度)

**数学公式:**
```
gate_sparsity = (|act_fn(gate_out)| < threshold).float().mean()
```

默认 `threshold = 0.01`。

**数据来源:** 同 Gate Activation Mean

**诊断意义:**
- 表示门控激活值低于阈值的"死神经元"占比
- **高稀疏度**: 大量 gate 单元几乎为零，PLE 容量被浪费
- 适度稀疏可能正常，但持续上升需要警惕
- 可通过 `gate_sparsity_threshold` 参数调整阈值

---

## 使用方式

### 基本用法

```python
from paddlefleet.internal_medicine import setup_internal_medicine

# 设置监控
monitor_dict = {}
model = setup_internal_medicine(
    model,
    monitors=["ple_health"],
    monitor_dict=monitor_dict,
    monitor_interval=1,
    ple_health={
        "log_per_layer": True,
        "log_global": True,
        "gate_sparsity_threshold": 0.01,
    },
)

monitor = monitor_dict["ple_health"]

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
| `log_per_layer` | `True` | 记录每层指标 (`ple_health/layer_{idx}/{metric}`) |
| `log_global` | `True` | 记录全局指标 (`ple_health/global_{metric}`) |
| `monitor_interval` | `1` | 监控间隔 (步数) |
| `verbose` | `False` | 打印调试信息 |
| `gate_sparsity_threshold` | `0.01` | 门控稀疏度阈值 |

### 日志格式

**全局指标 (仅全局):**
```
ple_health/global_token_ple_norm
ple_health/global_proj_ple_norm
ple_health/global_per_layer_inputs_norm
ple_health/global_token_proj_cosine
```

**逐层指标:**
```
ple_health/layer_0/residual_ratio
ple_health/layer_0/gate_activation_mean
ple_health/layer_0/gate_sparsity
...
```

**逐层对应的全局聚合:**
```
ple_health/global_residual_ratio
ple_health/global_gate_activation_mean
ple_health/global_gate_sparsity
```

---

## Hook 挂载点

| Hook 位置 | 类型 | 捕获内容 | 产出指标 |
|-----------|------|----------|----------|
| `embed_tokens_per_layer` | post-hook | token PLE → 缓冲 `[S,B,L,H_ple]` | (缓存) |
| `per_layer_projection_norm` | post-hook | normed proj PLE × H^{-0.5} → 缓冲 | (缓存) |
| `ErnieDevGPTModel` | post-hook | 消费两个缓冲区 | `token_ple_norm`, `proj_ple_norm`, `per_layer_inputs_norm`, `token_proj_cosine` |
| `PLESubmodule.gate_proj` | post-hook | gate 原始输出 → 缓冲 | (缓存) |
| `PLESubmodule` | post-hook | 输入/输出 + 消费 gate 缓冲 | `residual_ratio`, `gate_activation_mean`, `gate_sparsity` |
