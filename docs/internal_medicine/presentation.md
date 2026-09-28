# Internal Medicine — 模型训练健康监控系统

## 技术方案汇报

---

## 一、问题与动机

### 1.1 大模型训练的"黑盒"困境

当前大模型训练的主要观测手段：

| 传统指标 | 特点 | 局限 |
|----------|------|------|
| Training Loss | 最终优化目标 | **严重滞后** — loss 异常时问题已发酵数百步 |
| Gradient Norm | 梯度健康指标 | **粒度不足** — 无法定位是哪个子模块出了问题 |
| Learning Rate | 调度状态 | 只反映预设策略，非模型内部状态 |
| GPU Utilization | 硬件效率 | 与模型健康无关 |

**核心矛盾**: 以上指标告诉你"模型训练得好不好"，但无法告诉你"模型内部正在发生什么"。

### 1.2 需要什么样的监控

类比医学体检：

```
传统指标  ≈  体温、血压        → 知道"是否生病"（滞后、粗粒度）
Internal Medicine  ≈  血液检查 + 影像学  → 知道"哪里在恶化"（前瞻、细粒度）
```

我们需要一套能够：
- **前瞻性**检测异常 — 在 loss 曲线异常之前发现苗头
- **定位性**诊断 — 精确到具体层、具体模块
- **零侵入**部署 — 不修改模型代码、不影响训练梯度
- **低开销**运行 — 不显著增加训练时间和通信量

### 1.3 Internal Medicine 的定位

针对模型内部结构的三个关键维度建立监控：

```
┌─────────────────────────────────────────────────────────┐
│                    Model Health                          │
│                                                          │
│  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐ │
│  │  注意力健康     │  │  专家系统健康   │  │  PLE 结构健康  │ │
│  │  QK Stats      │  │  MoE Health    │  │  PLE Health   │ │
│  │  (9 指标)      │  │  (13 指标)     │  │  (7 指标)     │ │
│  │               │  │               │  │              │ │
│  │ 数值稳定性     │  │ 路由均衡性     │  │ 分支冗余度    │ │
│  │ 注意力集中度   │  │ 专家发展均衡   │  │ 残差贡献比    │ │
│  │ Sink 现象     │  │ 共享/路由平衡  │  │ 门控活跃度    │ │
│  └───────────────┘  └───────────────┘  └──────────────┘ │
│                                                          │
│                   共 29 项诊断指标                         │
└─────────────────────────────────────────────────────────┘
```

---

## 二、系统架构

### 2.1 零侵入设计

核心原则：**只观察，不干预**。

```python
# 通过 PyTorch forward hook 采集数据，不修改模型代码
module.register_forward_hook(hook_fn)       # 捕获模块输入/输出
module.register_forward_pre_hook(hook_fn)   # 捕获模块输入（用于 QK）

# hook 内部所有计算均在 torch.no_grad() 下执行
# → 不产生梯度，不参与反向传播，不影响训练
```

**兼容 Activation Checkpointing**：所有 hook 通过 `torch.is_grad_enabled()` 判断是否处于 recompute 阶段，避免重复计算。

### 2.2 整体数据流

```
训练 Forward Pass
    │
    ├── QK Stats Hook (core_attention pre-hook)
    │   └── compute_qk_stats() → Triton 内核
    │
    ├── MoE Health Hook (router post-hook + moe_layer post-hook)
    │   └── compute_router_metrics() / compute_expert_metrics()
    │
    ├── PLE Health Hook (embed/proj/gate/ple post-hooks)
    │   └── compute_branch_norms() / compute_residual_ratio() / ...
    │
    ▼
training_logs.update(**metrics)          ← 各 hook 将指标写入全局单例
    │
    ▼
on_train_step_end (每 log_interval 步):
    ├── monitor.step()                   ← 更新各 monitor 计步器
    ├── training_logs.gather_and_aggregate()  ← 跨卡聚合
    ├── print_rank_0(metrics)            ← stdout 输出
    ├── writer.add_scalar(k, v, step)    ← TensorBoard 写入
    └── training_logs.reset()            ← 清空，准备下一轮
```

### 2.3 集成方式

**YAML 一行配置启用**：

```yaml
# conf/your_experiment.yaml
internal_medicine_monitors: ["qk_stats", "moe_health", "ple_health"]
# 或者全部启用:
# internal_medicine_monitors: ["all"]
```

**框架集成** (`src/trainers/pretraining_trainer.py`)：

```python
# 1. pretrain() 中注册 pre-wrap hook（DDP 包装前挂载，可以访问原始模型）
monitor_dict = {}
cfg.model.register_pre_wrap_hook(
    partial(setup_internal_medicine, monitors=cfg.internal_medicine_monitors,
            monitor_dict=monitor_dict, monitor_interval=log_interval)
)

# 2. 注册 on_train_step_end 回调
callback_manager.register("on_train_step_end",
    partial(_on_train_step_end_internal_medicine, monitor_dict=monitor_dict, log_interval=log_interval))
```

---

## 三、三大监控维度详解

### 3.1 注意力健康 — QK Stats Monitor

**监控目标**: 检测注意力矩阵的数值稳定性、注意力分布集中度、Attention Sink 现象。

**Hook 位置**: `attention.core_attention` 的 `forward_pre_hook`，直接截获送入注意力计算的 Q/K 张量。

#### 指标详解

**1) Max Logit — 数值稳定性哨兵**
```
max = max(Q · K^T / √d)
```
- 注意力 logit 的全局最大值
- **异常信号**: 值突然暴增 → softmax 溢出风险，可能触发 NaN
- **典型场景**: 某个 head 的 Q/K 投影权重爆炸

**2) Mean Logit — 量级基线**
```
mean = mean(valid_logits)
```
- 排除因果 mask 位置后的 logit 均值
- **异常信号**: 持续偏移 → Q/K 投影权重量级漂移

**3) Entropy Avg — 注意力集中度**
```
H = -Σ(p_i × log(p_i))，其中 p = softmax(Q · K^T / √d)
```
- 注意力概率分布的 Shannon 熵，对所有 head 取均值
- **高熵**: 注意力分散，模型"犹豫不决"
- **低熵**: 注意力高度集中在少数 token 上，可能发生 attention collapse
- 正常训练中应随层深度呈现合理梯度

**4) Sink — Attention Sink 检测**
```
sink = mean(softmax(logits)[..., position_0])
```
- 第一个 token 获得的平均注意力权重
- **高 sink**: 大量注意力被"倾倒"到第一个 token → Attention Sink 现象
- 是 LLM 训练的已知现象，但过度 sink 影响模型质量

**5-6) Entropy Min / Max — Head 行为极值**
```
entropy_min = min(H_per_head)    — 最"尖锐"的 head
entropy_max = max(H_per_head)    — 最"分散"的 head
```
- 不需要记录每个 head，用分布统计量即可捕捉异常
- **entropy_min 极低**: 某个 head 注意力坍缩，只盯着极少 token

**7-9) Sink Head 分类 — Head 级 sink 行为**
```
sink_head_ratio = count(sink_per_head > threshold) / num_heads
sink_head_max = max(sink_per_head)
sink_nonsink_gap = mean(sink_heads) - mean(nonsink_heads)
```
- 识别哪些 head 把大量 attention mass 分配给 token-0
- **sink_head_ratio 高**: 多数 head 进入 sink/gating 行为
- **sink_nonsink_gap 高**: sink/non-sink head 分化明显

#### Triton Online Softmax — 性能优化

传统实现需要先计算完整注意力矩阵 `[B, H, S, S]`，对于 S=8192 这意味着 ~256MB 的临时显存。

我们的 Triton 内核采用 **Online Softmax** 算法，在单次遍历中同时计算全部 4 个统计量：

```
状态变量（每行维护）:
  m_i  — 当前最大 logit
  d_i  — softmax 分母 Σexp(x-m)
  h_i  — 熵分子项 Σ(x-m)·exp(x-m)
  s_i  — sink 分子项 exp(x₀-m)

遇到新 Key block 时增量更新:
  m_new = max(m_i, block_max)
  α = exp(m_i - m_new)
  d_new = d_i·α + Σexp(logits - m_new)
  h_new = h_i·α + (m_i - m_new)·d_i·α + Σ(logits - m_new)·exp(logits - m_new)

最终:
  entropy = log(d_i) - h_i/d_i
  sink    = s_i/d_i
```

**优势**:
- 不实例化 `[S, S]` 注意力矩阵，显存开销 O(S) 而非 O(S²)
- 单次遍历完成全部统计，计算量与一次 matmul 相当
- Grid 为 `(batch × num_heads)`，充分利用 GPU 并行性

---

### 3.2 专家系统健康 — MoE Specialist Monitor

**监控目标**: 检测 MoE 路由决策的合理性、专家权重的均衡发展、负载均衡机制的有效性。

**Hook 位置**: `moe_layer.router` post-hook（路由指标）+ `moe_layer` post-hook（专家权重指标）。

#### 路由健康指标

**1) Router Entropy — 路由均匀度**
```
H = -Σ(p_i × log(p_i))
```
- `p` 为 softmax 后的路由概率分布（`_scores_for_aux_loss`）
- **高熵**: 路由均匀，专家利用率高
- **低熵**: 路由集中在少数专家 → 专家坍缩 (Expert Collapse) 风险

**2-4) Score Sum Mean/Min/Max — TopK 置信度**
```
topk_scores, _ = scores.topk(K, dim=-1)
score_sum = topk_scores.sum(dim=-1)      # 每个 token 被选中专家的总分数
```
- **score_sum 过低**: Router 对选择不够自信，专家差异化不足
- **score_sum 接近 1.0**: 路由过于集中，未被选中的专家得分极低

**5-6) Expert Bias Mean/Std — 负载均衡偏置**
```
expert_bias_mean = router.expert_bias.mean()
expert_bias_std  = router.expert_bias.std()
```
- 反映负载均衡机制的当前状态
- **均值偏离零**: 系统性偏向某些专家
- **标准差过大**: 不同专家间偏置差异显著

**7) Bias-Affinity Jaccard — 负载均衡 vs 路由偏好的冲突度**
```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|

A = bias 调整前的路由决策
B = bias 调整后的路由决策
```

这是最关键的 MoE 诊断指标之一：

| Jaccard 值 | 状态 | 含义 |
|------------|------|------|
| > 0.7 | OK | Bias 对路由影响小，Router 偏好与负载均衡一致 |
| 0.3 ~ 0.7 | WARNING | Bias 显著改变路由，Router 和负载均衡存在分歧 |
| < 0.3 | **SEVERE** | Bias 强行扭转了大部分路由 → Router 想让 token 去 Expert A，但被 Bias 强制改到 Expert B |

**低 Jaccard 意味着**: Router 学到的"最优路由"和负载均衡的"均匀分配"产生了严重冲突。模型质量和训练效率之间被迫做出妥协。

#### 专家权重指标

**8-11) Expert Norm Mean/Std/Min/Max — 专家发展均衡度**
```
norm_i = ||concat(expert_i.weight1, expert_i.weight2)||₂
```
- 对每个本地专家计算全参数的 L2 范数
- **范数差异过大 (std 高)**: 专家发展不均衡
- **极小范数**: 专家萎缩 (Atrophy) — 该专家几乎不被使用，权重趋近于零
- **极大范数**: 专家过载 (Inflammation) — 该专家承担了过多 token

**12) Shared Expert Norm**
```
shared_expert_norm = ||concat(all shared_expert params)||₂
```

**13) Shared/Routed Ratio — 共享专家 vs 路由专家的平衡**
```
ratio = shared_expert_norm / mean(routed_expert_norms)
```

| Ratio | 状态 | 含义 |
|-------|------|------|
| 0.3 ~ 3.0 | OK | 共享专家与路由专家贡献均衡 |
| < 0.3 | INEFFECTIVE | 共享专家权重极小，相当于不存在 |
| > 3.0 | **MONOPOLY** | 共享专家主导 → MoE 退化为 Dense 模型 |

---

### 3.3 PLE 结构健康 — PLE Health Monitor

**监控目标**: 检测 Per-Layer Embedding 双分支架构是否有效工作 — 分支是否冗余、PLE 贡献是否合理、门控机制是否活跃。

**PLE 双分支架构简介**:
```
                    ┌──── Token 分支: embed_tokens_per_layer ────┐
Input Token IDs ───►│                                            ├──► (token + proj) × 2^{-0.5}
                    │                                            │         │
Hidden States ─────►│──── Proj 分支: linear → RMSNorm → ×H^{-0.5} ──┘         │
                    └────────────────────────────────────────────┘         ▼
                                                                   Per-Layer Input
                                                                         │
                                                              ┌──────────▼──────────┐
                                                              │   PLESubmodule       │
                                                              │   gate_proj → act_fn │
                                                              │   up_proj → down_proj│
                                                              │   + residual         │
                                                              └──────────────────────┘
```

#### 全局指标 — 双分支信号质量

**1-2) Token PLE Norm / Proj PLE Norm — 分支信号强度**
```
token_ple_norm = mean(||token_ple||₂)           # [S, B, L, H_ple] → norm on H_ple → mean
proj_ple_norm  = mean(||proj_ple × H^{-0.5}||₂)
```
- 两条分支分别的 L2 范数均值
- **一条远大于另一条**: 信号量级失衡，弱分支的贡献被淹没

**3) Per-Layer Inputs Norm — 合并信号量级**
```
per_layer_inputs_norm = mean(||(token_ple + proj_ple) × 2^{-0.5}||₂)
```
- 实际送入每层 PLESubmodule 的输入信号量级
- 应保持稳定，异常变化预示训练不稳定

**4) Token-Proj Cosine — 双分支冗余度（核心指标）**
```
cosine = mean(cosine_similarity(token_flat, proj_flat))
```
- 将两条分支展平为 `[S×B×L, H_ple]` 后计算余弦相似度

| Cosine 值 | 含义 |
|-----------|------|
| → 1.0 | **双分支高度冗余** — 两条路径携带几乎相同的信息，PLE 的双分支设计失效 |
| → 0 | **双分支互补** — 各自携带不同信息，设计有效 |
| < 0 | 两条分支信息正交甚至相反 |

**为什么这个指标重要**: PLE 的核心假设是 token embedding 和 hidden state projection 提供互补信息。如果 cosine → 1.0，说明这个假设不成立，模型实际上只需要一条分支，当前的双分支结构浪费了参数和计算。

#### 逐层指标 — PLE 子模块健康

**5) Residual Ratio — PLE 贡献幅度**
```
residual_ratio = ||output - hidden_states|| / ||hidden_states||
```
- PLE 子模块对残差流的修改量相对于输入的比值
- **→ 0**: PLE 几乎没有作用，退化为恒等映射
- **过大**: PLE 修改幅度过大，可能破坏已有的表示

**6) Gate Activation Mean — 门控信号强度**
```
gate_activation_mean = mean(|act_fn(gate_out)|)
```
- gate_proj 经激活函数 (GELU/SiLU) 后的平均绝对值
- **过小**: 门控信号弱，gated MLP 的选择性机制失效

**7) Gate Sparsity — 门控死神经元占比**
```
gate_sparsity = (|act_fn(gate_out)| < 0.01).float().mean()
```
- 绝对值低于 0.01 的门控单元比例
- **高稀疏度**: 大量 gate 单元"死亡"，PLE 的有效容量缩水
- **持续上升**: 问题正在恶化，需要干预

---

## 四、工程设计亮点

### 4.1 Triton Online Softmax 内核

传统方式计算注意力统计：
```python
logits = Q @ K.T * scale          # O(S²) 显存
probs = softmax(logits)            # O(S²) 显存
entropy = -(probs * log(probs)).sum()  # 需要完整矩阵
```

Online Softmax 方式：
```python
# 逐 block 流式计算，只维护 4 个标量状态
for block in K_blocks:
    logits = Q_block @ block.T * scale
    m_new = max(m_i, block_max)
    # 增量更新 d_i, h_i, s_i ...
# 最终 entropy = log(d) - h/d
```

| 维度 | 传统实现 | Triton Online Softmax |
|------|----------|----------------------|
| 显存 | O(B×H×S²) | O(B×H×S) |
| 计算 pass | 多次（matmul + softmax + entropy） | **单次遍历** |
| 输出 | 完整注意力矩阵 | 仅 4 个统计量/head |

以 S=8192, B=1, H=64 为例：传统方式需要 ~2GB 临时显存，Online Softmax 仅需 ~2MB。

### 4.2 SmoothedValue 自动模式推断

```python
# training_logs.py — 根据键名自动选择聚合策略
if "/max" in key or key.endswith("_max"):
    mode = "max"     # 追踪历史最大值
elif "/min" in key or key.endswith("_min"):
    mode = "min"     # 追踪历史最小值
else:
    mode = "mean"    # 追踪累积均值
```

**约定优于配置**: monitor 开发者只需按命名约定写日志键，无需手动指定聚合方式。

### 4.3 Activation Checkpointing 兼容

```python
def hook_fn(module, args):
    if not torch.is_grad_enabled():
        return  # 跳过 recompute 阶段
    # ... 正常计算
```

Activation Checkpointing 会在反向传播时重新执行 forward，如果不做这个判断，每个指标会被计算两次。

### 4.4 Pipeline Parallelism 感知

```python
# 层索引考虑 PP rank，确保跨 PP stage 的层编号全局一致
global_idx = pp_rank * len(local_layers) + local_idx
```

日志键 `moe_health/layer_42/router_entropy` 中的 42 是全局层号，即使模型被切分到不同的 PP stage 上。

---

## 五、通信开销分析

### 5.1 两层通信模型

```
Layer 1: Hook 内部通信
  ├── QK / MoE / PLE: 无 hook-time collective
  └── MassiveAct: TP 切通道维时，对 per-channel max 做一次 MAX all_reduce
  原则: 除正确性必需外，hook hot path 避免 NCCL collective

Layer 2: 全局聚合 (gather_and_aggregate)
  └── dist.all_gather_object(info_list, all_metrics)
  范围: world_size（全部卡，跨节点）
  频率: 每 log_interval 步
  数据量: ~90 KB/rank (pickle 序列化的 Python dict)
```

### 5.2 开销估算（以 80 层 MoE+PLE 模型，512 卡为例）

| 通信 | 频率 | 单次数据量 | 每步总通信量 | 影响 |
|------|------|-----------|-------------|------|
| QK hook 内通信 | — | 0 | 0 | 无；QK 主要成本是额外 QK stats 计算 |
| PLE hook 内通信 | — | 0 | 0 | 无 |
| MoE hook 内通信 | — | 0 | 0 | 无 |
| `gather_and_aggregate` | 每 log_interval 步 | 90 KB × 512 rank | ~45 MB | **主要开销** |

### 5.3 `gather_and_aggregate` 的特点

```python
# 使用 dist.all_gather_object — CPU 侧序列化通信
dist.all_gather_object(info_list, all_metrics)  # Gloo backend, pickle
```

| 特性 | 说明 |
|------|------|
| 通信后端 | Gloo (CPU)，**不是** NCCL (GPU) |
| 序列化 | Python pickle，有编解码开销 |
| 同步模型 | 全局 barrier — 所有 rank 必须同时参与 |
| 数据量 | ~90 KB/rank 级别，带宽不是瓶颈 |
| 真正的开销 | **同步等待** — 最慢 rank 决定所有人的等待时间 |

### 5.4 优化空间

当前设计的通信开销在 `log_interval ≥ 10` 时完全可以接受。进一步优化方向：

1. **DP group 内聚合替代全局聚合**: PLE/MoE 指标在各 DP rank 上是对称的，只需 DP group 内取均值即可，无需跨 TP/PP group
2. **NCCL 替代 Gloo**: 将 dict 预编码为 tensor，走 NCCL all_reduce 而非 all_gather_object
3. **异步聚合**: 在下一步训练的 forward 开始时异步触发上一步的聚合

---

## 六、完整指标速查表

### 6.1 QK Stats Monitor — 注意力健康 (9 指标)

| # | 指标 | 公式 | 聚合模式 | 异常信号 |
|---|------|------|----------|----------|
| 1 | `max` | `max(Q·K^T/√d)` | max | 突然暴增 → 数值不稳定 |
| 2 | `mean` | `mean(valid_logits)` | mean | 持续偏移 → Q/K 权重漂移 |
| 3 | `entropy_avg` | `-Σ(p·log(p))` head 均值 | mean | 过低 → 注意力坍缩 |
| 4 | `sink` | `mean(p[..., 0])` | mean | 过高 → attention sink |
| 5 | `entropy_min` | `min(H_per_head)` | min | 极低 → 某 head 坍缩 |
| 6 | `entropy_max` | `max(H_per_head)` | max | 极高 → 某 head 无效 |
| 7 | `sink_head_ratio` | `count(sink>threshold)/N_heads` | mean | 高 → 多数 head 进入 sink |
| 8 | `sink_head_max` | `max(sink_per_head)` | max | 高 → 最强 sink head 极端 |
| 9 | `sink_nonsink_gap` | `mean(sink)-mean(nonsink)` | mean | 高 → sink/non-sink 分化明显 |

### 6.2 MoE Specialist Monitor — 专家系统健康 (13 指标)

| # | 指标 | 公式 | 聚合模式 | 异常信号 |
|---|------|------|----------|----------|
| 1 | `router_entropy` | `-Σ(p·log(p))` | mean | 低 → 路由坍缩 |
| 2 | `score_sum_mean` | `mean(topk_sum)` | mean | 低 → 置信度不足 |
| 3 | `score_sum_min` | `min(topk_sum)` | min | 极低 → 部分 token 路由失败 |
| 4 | `score_sum_max` | `max(topk_sum)` | max | →1.0 → 路由过于集中 |
| 5 | `expert_bias_mean` | `bias.mean()` | mean | 偏离零 → 系统性偏向 |
| 6 | `expert_bias_std` | `bias.std()` | mean | 过大 → 偏置差异显著 |
| 7 | `bias_affinity_jaccard` | `\|A∩B\|/\|A∪B\|` | mean | <0.3 → 路由冲突严重 |
| 8 | `expert_norm_mean` | `mean(L2_norms)` | mean | 持续变化 → 专家不稳定 |
| 9 | `expert_norm_std` | `std(L2_norms)` | mean | 过大 → 发展不均衡 |
| 10 | `expert_norm_min` | `min(L2_norms)` | min | 极小 → 专家萎缩 |
| 11 | `expert_norm_max` | `max(L2_norms)` | max | 极大 → 专家过载 |
| 12 | `shared_expert_norm` | `\|\|shared_params\|\|₂` | mean | — |
| 13 | `shared_routed_ratio` | `shared/routed_mean` | mean | <0.3 无效 / >3.0 垄断 |

### 6.3 PLE Health Monitor — PLE 结构健康 (7 指标)

| # | 指标 | 公式 | 聚合模式 | 异常信号 |
|---|------|------|----------|----------|
| 1 | `token_ple_norm` | `mean(\|\|token\|\|₂)` | mean | 量级异常 |
| 2 | `proj_ple_norm` | `mean(\|\|proj×H^{-0.5}\|\|₂)` | mean | 与 token 分支失衡 |
| 3 | `per_layer_inputs_norm` | `mean(\|\|(t+p)×2^{-0.5}\|\|₂)` | mean | 不稳定 |
| 4 | `token_proj_cosine` | `mean(cos_sim)` | mean | →1.0 → 双分支冗余 |
| 5 | `residual_ratio` | `\|\|Δ\|\|/\|\|h\|\|` | mean | →0 无效 / 过大不稳定 |
| 6 | `gate_activation_mean` | `mean(\|act(g)\|)` | mean | 过小 → 门控失效 |
| 7 | `gate_sparsity` | `(\|act(g)\|<0.01).mean()` | mean | 持续上升 → 死神经元增多 |

---

## 七、未来展望

### 7.1 自动报警机制

当前系统只做"采集 + 记录"，缺少主动报警。规划：

```
指标异常检测                      报警动作
──────────────────               ──────────────────
qk_stats/max > threshold     →   发送飞书告警
bias_affinity_jaccard < 0.3  →   标记 SEVERE + 告警
gate_sparsity 连续 N 步上升   →   趋势告警
expert_norm_min 持续下降      →   专家萎缩告警
```

不仅看绝对值，更看**趋势变化率** — 比如 gate_sparsity 从 0.1 涨到 0.3 可能比 gate_sparsity=0.5 但稳定更危险。

### 7.2 通信优化

- 用 NCCL tensor all_reduce 替代 Gloo pickle all_gather_object
- 改为 DP group 内聚合，避免全局 barrier
- 预计可将聚合开销降低一个数量级

### 7.3 跨实验对比分析

将不同实验的内科指标对齐到相同的 step 轴：
- 不同学习率对 `router_entropy` 的影响
- 不同 MoE topK 对 `bias_affinity_jaccard` 的影响
- 不同 PLE hidden_size 对 `token_proj_cosine` 的影响

### 7.4 指标与训练效果的关联分析

建立内科指标与下游指标（loss、eval metric）的关联模型：
- 哪些内科指标是 loss spike 的先行指标？
- 什么样的内科指标组合预示着最优的训练效率？
- 如何根据内科指标自动调整超参？

---

## 附录: 系统演进时间线

| 时间 | 里程碑 |
|------|--------|
| 2025.12 | QK Stats Monitor 上线 — 注意力数值稳定性监控 |
| 2025.12 | 新增 per-head entropy 分布统计 |
| 2025.12 | MoE Specialist Monitor 上线 — 专家路由健康监控 |
| 2025.12 | 修复 Activation Checkpointing 兼容性 |
| 2025.12 | 统一 YAML 配置接口 |
| 2026.04 | PLE Health Monitor 上线 — PLE 结构健康监控 |
| 2026.04 | 完善文档体系 |
