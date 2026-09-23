# KDA Health Monitor

监控 KDA (Kimi Delta Attention) 层的健康状况。KDA 是一个**固定容量的联想记忆**：状态
`S_t ∈ R^{d_k × d_v}` 不随序列增长，全部历史都压在里面。

```
S_t = (I - β_t k_t k_tᵀ) Diag(α_t) S_{t-1} + β_t k_t v_tᵀ,     õ_t = S_tᵀ q_t
```

它由三条通路支撑，缺一条这层就不再做序列建模：

| 通路 | 由谁控制 | 功能 |
|------|----------|------|
| **衰减** | `α_t ∈ (0,1)^{d_k}`，channel-wise | 每个 key 通道的记忆时间尺度。channel-wise 的功能目的是让同一头内不同通道承担不同时间尺度，短程与长程信息共存而不互相冲刷 |
| **写入** | `β_t ∈ (0,1)` | delta rule 的写入强度，即 `S` 朝 `k_t ↦ v_t` 自我纠正的步长 |
| **读出** | `Sigmoid(W_g x_t)` 输出门 + head-wise RMSNorm | `õ_t` 有多少进入残差流 |

衰减门的映射（fp32，`kda_gate`）：

```
g = lower_bound · Sigmoid(exp(A_log) · (z + dt_bias))      # gate_lower_bound 已设置
g = -exp(A_log) · Softplus(z + dt_bias)                    # gate_lower_bound = None
α = exp(g)
```

`z` 是 `f_b_proj` 输出的 raw decay logit。时间尺度 `τ ≈ -1/g`（token 数）。

---

## 开关与 no-op 保证

通过在 `internal_medicine_monitors` 中加入 `kda_health`（或 `all`）启用。启用它在**任何**模型上都是安全的——
以下任一情况下它彻底 no-op（不挂 hook、不声明/产生任何指标、不抛异常）：

1. **模型未使用 KDA 层**：`is_kda_layer` 检查 token mixer 上是否同时存在 `f_b_proj` 与
   `gate_lower_bound` 两个属性——这两个属性在整个 paddlefleet 里只出现在 `kimi_delta_attention.py`。
   与 `layer_discovery` 里其它 attention 的判定风格一致，不 import 生产类，因此模块被移动或重命名
   时也不会静默退化成"发现不到任何 KDA 层"。
2. **不会误判 GDN**：GDN 是可选的 `attention_layer_type`，与 KDA 共用 `A_log` / `dt_bias` /
   `conv1d` / `in_proj`，所以判定必须用这两个 KDA 独有属性，而不是任何单个通用属性。
   用 `hasattr` 而非真值判断：`gate_lower_bound=None` 是合法配置（选择无下界的 softplus gate）。

```yaml
internal_medicine_monitors:
    kda_health: true
```

---

## 采集方式：两个 forward post hook + 一次参数读取

`α` / `β` / `gate` 都是 `KimiDeltaAttention.forward` 的**局部变量**，不在任何子模块的返回里，
但都能从产生它们的投影上拿到：

| 采集点 | 拿到什么 | 用于 |
|--------|----------|------|
| `attn.f_b_proj` | raw decay logit `z`，`[b,s,v_dim]`（SP 下 `[s,b,v_dim]`） | 衰减 4 项 + `A_log_mean` |
| `attn.in_proj` | `qkvbz`，按 `[qkv \| beta \| gate]` 切分 | 写入 2 项 + 读出 1 项 |
| `attn.g_b_proj` | 低秩输出门的 logit（仅 `use_full_rank_gate=False` 时挂） | 读出 1 项 |

`beta` 与 `gate` 都是 **pre-sigmoid logit**（sigmoid 折进 kernel，`use_beta_sigmoid_in_kernel=True`），
监控侧补 `sigmoid`，fp32。`g` 用 `kda_metrics.kda_log_decay` 在 `no_grad` 下对 detach 后的 `z` 重算，
公式与 `kimi_delta_attention.kda_gate` 逐字对齐；门的形式（`safe_gate` / `lower_bound`）从层上读，
**不硬编码 `-5.0`**，所以 `gate_lower_bound=None` 的配置也按它自己的口径测量。

**不挂 `conv1d`**：`use_fused_kernels` 为真时走 `causal_conv1d` 函数而非 `self.conv1d.forward`，hook 挂不上；
L2Norm 也折进了 kernel。所以 conv / q / k 的**激活**幅度在 fused 路径上根本不可达。

**不 wrap 任何绑定方法**，因此没有 recompute 重放去重、也没有 `RecomputeWithoutOutput` 的显存持有约束
需要论证。热路径纪律见 `.claude/skills/monitor-hook-perf-rules`：hook 内无 D2H 同步、无集合通信，
schema 在 `allocate_buffers` 前一次性声明。

---

## 监控指标

每层 8 个指标，按上面三条通路 + 一个归因参数组织。日志键形如
`kda_health/layer_{i}/{attn_type}_{name}`，KDA 层的 `{attn_type}` 是 `kda`；对应的
`kda_health/global_{attn_type}_{name}` 由逐层累加器在 flush 时自动派生。

**三条硬性设计约束**（这套指标只有 8 项，是这三条筛出来的结果）：

1. **不引入拍出来的阈值。** 任何 `mean(1[x < thr])` 形式的比例指标都要求先知道真实分布才能定 `thr`，
   定偏了就恒为 0 或恒为 1。一条都没有。
2. **不依赖被监控方的内部实现常量。** kernel 的 `chunk_size=64`、sub-chunk `BC=16` 是实现细节而非配置项；
   指标定义里出现这类常量，kernel 一改指标就静默失效——读数照出、含义已错，比没有指标更糟。
3. **极值只在结构轴（head / channel）上取，绝不在 token 轴上取。**先沿 token 轴平均消掉样本量的影响，
   再沿结构轴取极值。理由：min/max 累加器跨整个 optimizer step 的全部 microbatch，token 轴样本量随梯度
   累积步数增长，而 `g ∈ (g_min, 0)`、`σ(β) ∈ (0,1)` 都是**有界量**，样本一多必然饱和到边界变成恒定读数。
   结构轴的宽度（head 数、channel 数）是固定的。

### 通路一：衰减 —— 记忆时间尺度（4 项）

回答：状态记得住东西吗？记多久？通道之间有分工吗？门还在看输入吗？最陡的通道有多陡？

| 指标 | 公式 | 诊断意义 |
|------|------|----------|
| `alpha_log_mean` | `mean(g)` | 时间尺度的**中心位置**，记忆寿命的总量指标。趋 `g_min` = 状态每步几乎清空，`S` 退化成只看最近一两个 token 的局部窗口，固定容量的联想记忆白搭；趋 `0` = 从不遗忘，历史 kv 关联无限累积进固定大小的 `S`，必然过载互相干扰。重点看漂移趋势而非绝对值 |
| `alpha_channel_spread` | `mean_{t,h}( std_d(g[t,h,:]) )` | 时间尺度的**跨通道多样性**。channel-wise 门的功能目的就是让通道承担不同时间尺度；→0 意味着整个头塌到单一时间尺度，短程与长程信息共用一个衰减率、互相冲刷。**不可替代**——`alpha_log_mean` 对"全通道同一个值"和"通道间分散但均值相同"完全不可分 |
| `alpha_token_spread` | `std_t( mean_{h,d} g )` | 衰减门的**输入敏感性**。`α` 是 data-dependent 的，这是 KDA 能"按内容决定记多久"的全部来源；→0 说明门与当前 token 解耦、退化成常数衰减的固定线性 RNN。与前两项正交：门冻结时均值和通道分散度都可以保持正常 |
| `alpha_log_channel_min` | `min_{h,d}( mean_t g )`，按 **min** 归约 | **最陡衰减通道**的平均 log 保留率。两个作用：(a) 时间尺度分布的左端位置，与 `alpha_log_mean` 一起给出分布跨度；(b) chunkwise 数值风险的输入量——intra-chunk 计算要把 key 按倒数累计衰减 `1/Γ` 重新缩放，这一项越负、倒数放大越剧烈。**故意不乘任何 tile 长度**（约束 2）：要估溢出余量时，用当时 kernel 的实际 tile 跨度乘一下，再和 BF16 上限 `ln ≈ 88.7` 比 |

### 通路二：写入（2 项）

回答：新信息还在往状态里写吗？有没有 head 整体停写？

| 指标 | 公式 | 诊断意义 |
|------|------|----------|
| `beta_mean` | `mean(σ(β))` | delta rule 的平均写入步长。与 `alpha_log_mean` 一起构成"写入 vs 衰减"的收支平衡——写入远小于衰减时状态趋于空，远大于时状态被单 token 主导 |
| `beta_head_min` | `min_h( mean_t σ(β) )`，按 **min** 归约 | 写入最弱 head 的平均步长。`β→0` 时 `S_t ≈ Diag(α) S_{t-1}`，该 head 只在衰减旧状态、不再写入新 kv 关联，它的序列混合彻底失效。**`beta_mean` 发现不了这件事**：96 个 value head 里 6 个停写、其余在 0.5，`beta_mean` = 0.469，相对健康态只降 6%，在图上与正常波动分不开。先对 token 平均再取 head 最小（约束 3），所以不会被单个 outlier token 拉到 0 |

### 通路三：读出（1 项）

| 指标 | 公式 | 诊断意义 |
|------|------|----------|
| `out_gate_mean` | `mean(σ(gate))` | 输出门开度。前两条通路全部正常、这一项趋 0，则状态算得再好也不进残差流，整层的计算被丢弃 |

### 归因参数（1 项）

| 指标 | 诊断意义 |
|------|----------|
| `A_log_mean` | per-head log-scale `A`。`exp(A)` 直接乘在门的 sigmoid 输入上，是通路一异常的**归因入口**：`alpha_log_mean` 或 `alpha_log_channel_min` 漂移时，先分清是 `z` 的分布变了还是 `A` 把整条 sigmoid 拉陡了。它是参数读取，无需自己的 hook，挂在衰减 hook 里以保持同一采集节奏 |

### 归约方式

| 归约 | 指标 |
|------|------|
| mean（6 项） | `alpha_log_mean`、`alpha_channel_spread`、`alpha_token_spread`、`beta_mean`、`out_gate_mean`、`A_log_mean` |
| **min**（2 项） | `alpha_log_channel_min`、`beta_head_min` |

两个 min 指标的名字都以 `_min` 结尾，因此 `PaddleProbe.MIN_AGGREGATED` 与
`training_logs._is_min_metric` 的判定天然一致（有回归测试守这条命名契约）——不一致会让跨 rank 聚合
把本该取 min 的量做成平均。

---

## 并行与重算

**TP**：KDA 沿 head 维切分（`num_key_heads % tp == 0`、`num_value_heads % tp == 0`），
`α` / `β` / `gate` / conv 全部按 head 切。所以：

- mean 类：每 rank 观测到相同数量的样本（等分 head × 相同 token 数），flush 时 `gather_and_aggregate`
  对 per-rank mean 求平均是数学正确的。
- min 类：极值取在**本 rank 的 head/channel 子集**上，flush 时的全局 min 正好等于全局结构轴上的 min。

八项指标全部落在这两类里，**零 hook 内通信、零 flush 通信**。（`dt_bias` 是 per-channel 且
`is_distributed`，本地 absmax ≠ 全局，所以 `dt_bias_absmax` 这类指标暂未纳入——它会引入 flush 时的
跨 TP 归约。）

**SP**：`config.sequence_parallel` 时 `in_proj` / `f_b_proj` 的输出是 `[s/tp, b, dim]`，
forward 里才转回 `[b, s, dim]`。本 monitor 的统计全是 elementwise + 全局 reduce，对 s/b 轴顺序不敏感；
`_as_tokens` 按**尾维**展平而不依赖前两维的含义，所以两种布局都正确。

**CP**：`cp_size > 1` 时 KDA 强制 `batch == 1` 且要求 contiguous 切分。每 rank 只看到自己那段 token，
等长，mean 仍正确；两个 min 指标先在本 rank token 段上求平均再取结构轴极值，跨 rank min 得到的是
"各 rank token 段平均后的最坏 head/channel"——与"全序列平均后取最坏"不完全等价（同一个 head 在各 rank
被分别取 min），但等长切分下偏差是二阶的，且方向保守，读数不会比真值更乐观。

**重算**：两个 hook 挂在 `in_proj` / `f_b_proj` 上，不在 `recompute_rms_norm_gated` 的选择性重算段内。
`_should_monitor()` 的 grad 门（`PaddleProbe`）在 `recompute_granularity="full"` 时会跳过 `no_grad`
下的真前向、在反向重放时采集；本 monitor 没有任何依赖执行顺序的指标，所以不需要额外排序或去重。

**MTP**：discovery 走 `iter_monitor_layers` + `mark_mtp_layers`，MTP KDA 层的 key 自动带 `_mtp` 后缀。

---

## 与 KDA–MLA 混合栈的层标签

Kimi-Linear / K3 是 3 KDA + 1 全局注意力的交错结构。`layer_discovery` 为这类栈补了第三条 `attn_type`
规则（原有两条是 `csa_compress_ratios` 与 `sliding_window`，KDA 混合栈两条都不满足）：

- KDA 层 → `"kda"`
- 同栈的全局层 → `"mla"`（`MQALatentAttention`）或 `"global"`（其他）

全局层的标签在 `iter_monitor_layers` 里赋予，因为那是唯一能看到整个栈的地方：单看一层无法区分
"KDA 混合栈里的全局层"（必须打标签，否则和 KDA 层共用一张图）和"同质栈里的普通层"（必须保持不打标签，
否则存量 run 丢 metric key）。检测扫描**完整层列表**而非 `matches` 筛过的子集——`qk_stats` 主动排除了
KDA 层，若只看子集它就看不到任何 `"kda"`，同一个物理层会因 monitor 不同而带不同的 key。

没有这条规则时，KDA 层和全局层的 `massive_act` / `moe_health` / `mlp_update` 统计会混进同一张图。

`qk_stats` 的 discovery 也相应排除了 KDA 层：KDA 是线性注意力，没有 QK logits、没有 softmax、
没有 `core_attention` 可挂。此前它会通过 `has_attention` 谓词、声明完整 schema，然后永远不记录任何值
（hook 只在 `core_attention` 存在时才挂），每个 KDA 层白占十几个永远为空的累加器。
