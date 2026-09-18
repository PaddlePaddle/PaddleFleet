# Internal Medicine — 模型健康监控系统

训练时模型健康的实时监控框架，通过 forward hook 零侵入式采集指标，不影响训练梯度。

包含十一大监控模块：
- **[MoE Health](./docs/moe_specialist.md)** — MoE 专家系统健康监控 (28 指标)
- **[QK Stats](./docs/qk_logits.md)** — 注意力 QK 统计监控 (9 指标 + CSA/HCA 层 2 项)
- **[Massive Activation Health](./docs/massive_activation.md)** — Residual Stream Massive Activation 健康监控 (20 指标)
- **[PLE Health](./docs/ple_health.md)** — Per-Layer Embedding 健康监控 (7 指标)
- **[mHC Health](./docs/mhc_health.md)** — Manifold-Constrained Hyper-Connections 映射监控 (每 hc 模块 29 标量指标 + `n²+2n` 条逐元素映射序列，megatron 后端 16 指标；仅在开启 mHC 层时生效)
- **[KDA Health](./docs/kda_health.md)** — Kimi Delta Attention 线性注意力层的衰减/写入/读出三通路监控 (8 指标；仅 paddlefleet；仅在模型含 KDA 层时生效)
- **DSA Health** — DeepSeek Sparse Attention 的 indexer 健康度、选择结构、与主干稠密 attention 的一致性 (`attn_recall` / `KL`)、以及跨层选择重合度 (19 指标；仅 paddlefleet；仅在模型含 DSA 层时生效；CP>1 下 `match` 一族关闭)
- **VHA Health** — Virtual Head Attention 的 Q Premix (近恒等虚拟头扩展) 与 Linear Postmix (`I + A Bᵗ` 低秩跨头融合) 结构监控 (仅 paddlefleet；仅在 `use_vha_attention` 时生效)
- **APE Health** — CSA/HCA compressor APE 参数健康监控 (P0 级 7 指标；仅 paddlefleet)
- **Attn Update** — QK 乘积增量 `Δ₂ = ΔW_q W_kᵗ + W_q ΔW_kᵗ` / `Δ₃ = ΔW_q ΔW_kᵗ` 的谱监控 (每项 3 指标；仅权重，不挂 forward hook)
- **MLP Update** — MoE 专家 MLP 的参数增量 `ΔW_m`，按 (层 × 专家 × gate/up/down) 分开测再按层汇总 (每层 36 指标；仅权重)

以及一个挂在**优化器**而不是模型上的模块，**始终装上**（无需点名，调用方零改动；上报频率同样受 `monitor_interval` 门控）：
- **[Optimizer Update](./docs/optim_update.md)** — `optim/update_rms`、`optim/param_rms`、`optim/update_param_ratio`，与 grad norm 并排看

---

## 快速开始

### 统一 API

```python
from paddlefleet.internal_medicine import setup_internal_medicine

# 创建 monitor_dict 用于存储 monitor 实例
monitor_dict = {}

# 启用全部监控 (默认)
model = setup_internal_medicine(
    model,
    monitors=['all'],              # 或指定 ['ape_health', 'moe_health', 'qk_stats', 'massive_act']
    monitor_dict=monitor_dict,
    monitor_interval=1,
    verbose=False,
)

# 训练循环
for step in range(num_steps):
    loss = model(inputs)
    loss.backward()
    optimizer.step()

    # 每步更新所有 monitor 的计步器
    for monitor in monitor_dict.values():
        monitor.step()
```

`monitors` 同时接受名称列表和逗号分隔字符串，例如
`"qk_stats,moe_health"`。未知名称会告警并跳过；已请求 monitor 的初始化失败会
直接抛出异常，避免训练在诊断功能未生效时静默继续。

### 配合 NeMo Trainer 使用

```python
from functools import partial

cfg.model.register_pre_wrap_hook(partial(
    setup_internal_medicine,
    monitors=['moe_health', 'qk_stats', 'massive_act', 'ple_health'],
    monitor_dict=monitor_dict,
))
```

NeMo YAML 中推荐使用 dict 形式，便于按 monitor 传独立参数：

```yaml
internal_medicine_monitor_interval: 50
internal_medicine_hook_timing: false
internal_medicine_monitors:
    qk_stats: true
    moe_health: true
    massive_act:
        log_activation_rms: false
        log_post_norm_metrics: false
```

`true` 表示启用该 monitor 并使用默认参数；嵌套 dict 会作为该 monitor 的 kwargs 透传，例如上面的 `massive_act` 等价于
`setup_internal_medicine(..., massive_act={"log_activation_rms": False, "log_post_norm_metrics": False})`。
`log_activation_rms` 统一控制残差尺度/增益一组指标（`activation_rms` + `spectral_norm_max/min`）——`activation_rms` 由 spectral hook 的 per-token pre-RMS 免费导出，故合用一个开关。
`log_lipschitz`（默认 `true`）控制每层 Lipschitz/梯度增益指标（`lipschitz_max/min`）——由 forward hook 在输入/输出 hidden 上注册 tensor grad hook，在反向传播采集 `‖∂L/∂x‖/‖∂L/∂y‖`。
`log_logit_lens_entropy`（默认 **`false`**）控制 logit-lens 熵指标（`logit_lens_entropy_mean` + `logit_lens_logsumexp_mean`，均只报告 token 均值）；`log_logit_lens_cross_entropy`（默认 **`false`**）额外开启逐层交叉熵（`logit_lens_cross_entropy_mean`，末层≈LM loss，与熵共用同一次投影，只多一次 gather）——将每层残差经 LM head 投影求 softmax 熵/对数配分/对真实 token 的交叉熵，**开销较大**故默认关闭。配套参数：`logit_lens_chunk_size`（默认 `1024`，按 token 分块的 tile 大小）、`logit_lens_apply_final_norm`（默认 `true`，投影前是否过 `decoder.final_layernorm`）、`logit_lens_layers`（默认 `None`=全部持有 head 的层；传 list 只监控指定 global 层索引）。交叉熵的 label 由 head-owning chunk 上的顶层 pre-hook 从 `forward(labels=...)` 捕获，loss_mask 在模型外施加故报告未加权 token 均值。PP 下仅持有 LM head 的 stage 生效，其余 stage 为 no-op。
`log_hidden_spectral_entropy`（默认 **`false`**）控制 post-RMSNorm 隐状态谱熵指标（`hidden_spectral_entropy`）——对归一化后 hidden 的 Gram 矩阵做 `eigvalsh` 求归一化特征值分布的熵（有效秩，衡量表征多样性/秩坍缩），无 full SVD、无 host sync。set-level 非线性量，跨 shard 均值为近似。
list 形式仍兼容，但需要按 monitor 传参时不要再额外加 `internal_medicine_monitor_kwargs` 字段。

### 读取指标

```python
from paddlefleet.internal_medicine import training_logs

# 获取全部最新指标
all_metrics = training_logs.get_latest()

# 按前缀过滤
moe_metrics = training_logs.get_latest(prefix='moe_health')
qk_metrics  = training_logs.get_latest(prefix='qk_stats')
massive_metrics = training_logs.get_latest(prefix='massive_act')
ple_metrics = training_logs.get_latest(prefix='ple_health')

# 跨卡聚合后获取
aggregated = training_logs.gather_and_aggregate()

# 格式化打印
training_logs.print_metrics(prefix='massive_act')

# 重置
training_logs.reset()
```

---

## 架构概览

```
setup_internal_medicine()
    ├── setup_moe_monitor()   → MoESpecialistMonitor → forward hooks on MoE layers
    ├── setup_qk_monitor()    → QKStatsMonitor     → forward pre-hooks on core_attention
    ├── setup_massive_activation_monitor() → MassiveActivationMonitor → forward pre-hooks on transformer layers
    ├── setup_ape_monitor() → APEHealthMonitor → forward pre-hooks on APE compressors
    └── setup_ple_monitor()   → PLEHealthMonitor   → forward hooks on PLE modules
                                        │
                                        ▼
                              GPU 0-dim accumulators
                                        │
                                        ▼
                              training_logs (singleton)
                              ├── SmoothedValue (mean/max/min)
                              └── gather_and_aggregate() → 跨卡聚合
```

### 日志键命名规则

所有指标遵循统一的命名格式：

```
{monitor_name}/layer_{global_idx}/{metric_name}        # 普通层逐层指标
{monitor_name}/layer_{global_idx}_mtp/{metric_name}    # MTP 层逐层指标
{monitor_name}/global_{metric_name}                     # 全局聚合指标
```

- `monitor_name`: `ape_health` | `moe_health` | `qk_stats` | `massive_act` | `ple_health` | `mhc_health` | `vha_health` | `attn_update` | `mlp_update` | `kda_health` | `dsa_health`
- `global_idx`: 全局层索引。优先取模块自带的 `layer.layer_number`（0-based 全局编号）；取不到时回退到
  `pp_rank × local_layers + local_idx`。`num_empty_layers_add_in_head > 0` 时所有层号整体偏移该值，看板对号要减掉
- `_mtp`: 仅 MTP layer 带有的层类型标记，随指标走现有聚合和日志链路

---

## 一、MoE Specialist Monitor (moe_health)

> 详细文档: [moe_specialist.md](./docs/moe_specialist.md)

监控 MoE (Mixture of Experts) 路由、专家权重和负载均衡健康状况。

| # | 指标 | 日志键 | 公式 | 级别 | 诊断意义 |
|---|------|--------|------|------|----------|
| 1 | `router_entropy` | `moe_health/.../router_entropy` | `-Σ(p × log(p))` | 每层+全局 | 路由分布均匀度 |
| 2 | `score_sum_mean` | `moe_health/.../score_sum_mean` | `mean(topk_scores.sum())` | 每层+全局 | TopK 分数和均值 |
| 3 | `score_sum_min` | `moe_health/.../score_sum_min` | `min(topk_scores.sum())` | 每层+全局 | TopK 分数和最小值 |
| 4 | `score_sum_max` | `moe_health/.../score_sum_max` | `max(topk_scores.sum())` | 每层+全局 | TopK 分数和最大值 |
| 5 | `expert_bias_mean` | `moe_health/.../expert_bias_mean` | `expert_bias.mean()` | 每层+全局 | 专家偏置均值 |
| 6 | `expert_bias_std` | `moe_health/.../expert_bias_std` | `expert_bias.std()` | 每层+全局 | 专家偏置标准差 |
| 7 | `bias_affinity_jaccard` | `moe_health/.../bias_affinity_jaccard` | `\|A∩B\| / \|A∪B\|` | 每层+全局 | Bias 前后路由一致性 |
| 8 | `expert_norm_mean` | `moe_health/.../expert_norm_mean` | `mean(expert_L2_norms)` | 每层+全局 | 专家权重范数均值 |
| 9 | `expert_norm_std` | `moe_health/.../expert_norm_std` | `std(expert_L2_norms)` | 每层+全局 | 专家权重范数标准差 |
| 10 | `expert_norm_min` | `moe_health/.../expert_norm_min` | `min(expert_L2_norms)` | 每层+全局 | 最小专家范数 |
| 11 | `expert_norm_max` | `moe_health/.../expert_norm_max` | `max(expert_L2_norms)` | 每层+全局 | 最大专家范数 |
| 12 | `shared_expert_norm` | `moe_health/.../shared_expert_norm` | `\|\|shared_params\|\|₂` | 每层+全局 | 共享专家权重范数 |
| 13 | `shared_routed_ratio` | `moe_health/.../shared_routed_ratio` | `shared_norm / routed_mean` | 每层+全局 | 共享/路由专家比例 |
| 14 | `load_max_min_ratio` | `moe_health/.../load_max_min_ratio` | `max(tokens) / min(tokens)` | 每层+全局 | 最忙/最闲专家 token 数比值 |
| 15 | `load_max_median_ratio` | `moe_health/.../load_max_median_ratio` | `max(tokens) / median(tokens)` | 每层+全局 | 最忙/中位专家 token 数比值 |
| 16 | `load_cv` | `moe_health/.../load_cv` | `std(tokens) / mean(tokens)` | 每层+全局 | 专家负载变异系数 (均衡=0) |
| 17 | `load_balance_entropy_norm` | `moe_health/.../load_balance_entropy_norm` | `H(p) / log(E)` | 每层+全局 | 归一化负载熵 (均衡=1, 坍缩=0) |
| 18 | `load_effective_experts` | `moe_health/.../load_effective_experts` | `exp(H)` | 每层+全局 | 有效专家数 (均衡=E, 坍缩=1) |
| 19 | `expert_gate_stable_rank_mean` | `moe_health/.../expert_gate_stable_rank_mean` | `mean_e(‖W_g‖_F² / ‖W_g‖_2²)` | 每层+全局 | 路由专家 SwiGLU 门控矩阵谱宽度均值 |
| 20 | `expert_gate_stable_rank_min` | `moe_health/.../expert_gate_stable_rank_min` | `min_e(...)` | 每层+全局 | 最先塌缩的专家 (秩退化预警) |
| 21 | `expert_gate_stable_rank_max` | `moe_health/.../expert_gate_stable_rank_max` | `max_e(...)` | 每层+全局 | 谱最宽的专家 (专家分化程度) |
| 22 | `expert_gate_singular_entropy_mean` | `moe_health/.../expert_gate_singular_entropy_mean` | `mean_e(-Σ pᵢ log pᵢ), pᵢ = σᵢ²/Σσⱼ²` | 每层+全局 | 门控矩阵整谱利用率均值 |
| 23 | `expert_gate_singular_entropy_min` | `moe_health/.../expert_gate_singular_entropy_min` | `min_e(...)` | 每层+全局 | 全谱枯竭最严重的专家 |
| 24 | `expert_gate_singular_entropy_max` | `moe_health/.../expert_gate_singular_entropy_max` | `max_e(...)` | 每层+全局 | 全谱最均匀的专家 |
| 25 | `shared_gate_stable_rank` | `moe_health/.../shared_gate_stable_rank` | `‖W_g‖_F² / ‖W_g‖_2²` | 每层+全局 | 共享专家门控矩阵谱宽度 |
| 26 | `shared_gate_singular_entropy` | `moe_health/.../shared_gate_singular_entropy` | `-Σ pᵢ log pᵢ, pᵢ = σᵢ²/Σσⱼ²` | 每层+全局 | 共享专家门控矩阵全谱利用率 |
| 27 | `expert_token_share` | `moe_health/layer_N/expert_token_share_eK` | `count_K / Σ count × 100%` | 每层每专家 | 专家 K 分到的 token 占本层路由量的百分比 |
| 28 | `expert_weight_share` | `moe_health/layer_N/expert_weight_share_eK` | `Σ probs_K / ΣΣ probs × 100%` | 每层每专家 | 专家 K 占本层总 combine 权重的百分比 (激活幅度贡献) |

> 19-26 针对的是**专家 MLP 的 SwiGLU 门控投影** (`fc1` 前一半输出，即经过 SiLU 的那一半)，不是 router 的 `gate.weight`。
> 奇异值经 Gram 矩阵特征分解得到 (`W Wᵀ` 或 `WᵀW` 取较小方阵)，比 `svdvals` 快约 40 倍；路由专家与共享专家的 Gram 拼成一个批次求解，避免 cuSOLVER 因批次形状变化重建 workspace。
> 随机初始化下 stable rank ≈ `mn/(√m+√n)²` 而非 `min(m,n)`（方阵约为 `n/4`），所以初值偏大是正常基线，有意义的是相对基线的下降幅度。

PaddleFleet 后端还会直接从每次 router forward 的实际选择结果输出以下指标：

| 指标组 | 公式 | 诊断意义 |
|--------|------|----------|
| `router_input_{rms,abs_max,abs_p99}` | Router 输入的 RMS、绝对最大值和绝对值 P99 | 区分输入整体放大与局部 outlier |
| `router_entropy_norm` | `router_entropy / log(E)` | 跨不同专家数比较路由亲和度的尖锐程度 |
| `router_margin_{mean,min,p10,p01}` | 最弱已选专家分数减最强未选专家分数 | 衡量 Top-K 决策边界的稳健程度 |
| `assignment_load_*` | 对各专家实际 hard assignment 数量的分布统计 | 衡量 token 数量负载是否均衡 |
| `gate_mass_*` | 对各专家已选正亲和度总量的分布统计 | 衡量专家获得的连续门控质量是否均衡 |

`assignment_load_*` 和 `gate_mass_*` 均包含 `cv`、`entropy_norm`、
`kl_uniform`、`max_frac`、`min_frac` 与 `max_min_ratio`。它们分别回答
“每个专家接收多少 token”和“每个专家接收多少正门控质量”，不可互相替代。

> **注**: 指标 14-18 (`load_*`) 在满足以下**任一**条件时输出, 优先使用前者:
> 1. `global_aux_loss` 已开启 (`get_aux_loss_coeff("global_aux_loss") > 0`): 复用
>    router 的 `global_tokens_per_expert` 缓冲区 (已跨 TPxDPxCP all-reduce 并在
>    global batch 内累加)。该缓冲区在 `finalize_model_grads` →
>    `reset_model_temporary_tensors` 中被清零, 因此在清零**之前**读取。
> 2. `moe_router_enable_expert_bias=true`: 复用 mcore 在 `get_updated_expert_bias`
>    中已做的 TPxCPxDP all-reduce (softmax 模型无法开启 expert-bias, 属此路径回退项)。
>
> 两种来源都在 global batch 级读取 (不在 forward hook 热路径上), 且计数已跨 rank
> reduce, 因此无需 monitor 侧 collective, 统计值已是全局正确值。load 比值本身
> scale-invariant, 直接用累加计数即可。

> **注**: 指标 27-28 (`expert_*_share`) 是**每层每专家一条曲线**的向量指标 (256 专家的
> 18 层模型 = 9216 个键), 只在 `log_per_layer=true` 时输出, 且不派生 `global_*`。
> 它们与 `assignment_load_*` / `gate_mass_*` 互补: 后者回答一层「有多不均衡」,
> 前者回答「是哪个专家造成的、是否长期是同一个」。
>
> `expert_token_share` 直接复用 `assignment_load_*` 已经算出的 hard-assignment
> 计数向量, 因此除归一化外不增加任何 kernel, 且与之严格一致
> (`max(expert_token_share) == assignment_load_max_frac × 100`)。
> `expert_weight_share` 用的是 **combine 权重** (归一化并乘 `routed_scaling_factor`
> 之后的 `probs`), 即该专家输出以多大权重进入残差 —— 这是「激活幅度」的可观测代理量。
> 这里刻意不复用 `gate_mass` (它不含 scaling factor, 在
> `routed_scaling_factor_param` 可学习时两者会分叉)。开 `moe_expert_fusion` 时融合
> MoE 算子只返回已合并的输出, 真正的专家级输出 L2 取不到。
>
> 两者都不出 GPU: 热路径上整条向量一次 `add_`, 而不是每个专家一个累加器, 无 D2H
> 同步、无 hook 内 collective。跨 rank 聚合就是各 DP rank 的普通均值, 与全局占比
> 一致 (每个 rank 路由的 token 数相同)。
>
> 读法: 完全均衡时两者都等于 `100 / num_experts` (256 专家 ≈ 0.39%)。按层取这组
> 曲线的 max / 中位 / min 最好用 —— max 抬升 = 热点专家, min 贴 0 = dead expert,
> 中位数应稳定在均衡点附近; max 与中位数拉开距离即长尾不均衡。

### 健康阈值

| 指标 | 值 | 状态 | 说明 |
|------|-----|------|------|
| `bias_affinity_jaccard` | > 0.7 | OK | Bias 对路由影响较小 |
| | 0.3 ~ 0.7 | WARNING | Bias 显著改变了路由 |
| | < 0.3 | SEVERE | Bias 强行扭转了大部分路由决策 |
| `shared_routed_ratio` | 0.3 ~ 3.0 | OK | 共享专家与路由专家贡献均衡 |
| | < 0.3 | INEFFECTIVE | 共享专家作用不大 |
| | > 3.0 | MONOPOLY | 共享专家主导，MoE 退化为 Dense |

### 按卡负载（`card` 族，仅 paddlefleet）

EP 下"这张卡本 step 收到了多少 token"及其峰均比。前面那些负载指标都是**发送侧**视角
（router 把本 rank 的 token 分给了哪些专家），这一族是 dispatch **之后**的**接收侧**视角
——决定 step 时间的是收得最多的那张卡，全组都要等它。

| 指标 | 含义 |
|------|------|
| `card_tokens_max` | 收到 token 最多的那张卡 |
| `card_tokens_mean` | 平均 |
| `card_tokens_peak_mean_ratio` | 峰均比。1.0 = 完全均衡 |

#### 两种口径，同一组 key

`_reduce_card_tokens_over_ep` 跑完后，**同一 EP 组内每张卡持有的这三个值是相同的**（max 与 mean
都是组内 `all_reduce` 出来的）。之后跨 rank 归约按 key 名再来一层，于是三条曲线各自代表不同的范围：

- `card_tokens_max` → `all_reduce(MAX)`：max of maxes = **全局最热的那张卡**
- `card_tokens_mean` → `all_reduce(SUM)/count`：G 个组各 `ep_size` 张卡都上报组均值 `m_g`，
  `(Σ_g ep_size·m_g) / (G·ep_size) = (Σ_g m_g)/G` = **全局所有卡的均值**（各组等大时）
- `card_tokens_peak_mean_ratio` → `all_reduce(SUM)/count` = **各 EP 组峰均比的平均**

所以：

| 想问的问题 | 看哪个 |
|---|---|
| 全世界最热的卡比全局平均重几倍（木桶效应，谁拖住了 step 时间） | 自己算 `card_tokens_max / card_tokens_mean` |
| 平均而言每个 EP 组内部有多少不均衡（路由层面，与 DP 数据分布差异无关） | `card_tokens_peak_mean_ratio` |

两者的差就是 DP 维贡献的那部分不均衡。举例：`PP=2, EP=4, DP=2`，某层两个 EP 组分别收到
`[1000,1000,1000,1000]` 和 `[4000,0,0,0]`，则 `max=4000`、`mean=1000`、`peak_mean_ratio=2.5`，
手算 `max/mean=4.0`。**`peak_mean_ratio` 才是与 `moe_logging` 逐点可比的那一个。**

口径与 PaddleFleet `moe_logging` 的 `local_tokens_per_card_layer_{N}_max_mean_ratio`
（`moe_utils.py:_log_local_tokens_per_card`）一致：统计范围是**一个 EP 组内**，不含 DP 维。

#### 微批的归约顺序是语义的一部分

峰值先在**微批内**跨卡取，之后才对微批求平均 —— 即
`mean_微批( max_卡 )`，与 `log_moe_balance` 相同。先把各微批的负载平均、再跨卡取 max
会得到 `max_卡( mean_微批 )`，由 `max` 的凸性它**恒不大于**前者，只有"每个微批里最热的都是同一张卡"
时才相等。12 层 EP=8 实测两者差到 8%。

同理，`card_tokens_peak_mean_ratio` 是**逐微批比值的平均**而不是两条上报曲线的比值：
只有当每微批的跨卡均值恒定（不丢 token 时成立）两者才重合。要木桶效应就自己拿
`max / mean` 除，那是另一个口径（见上表），不是这条曲线的近似。

取数点是 `moe_layer.dispatch` 上的一个 patch —— 计数是 dispatcher 的状态
（`token_dispatcher.get_dispatched_routing()[2]`），不在任何模块的 forward 返回值里，
而且只有 a2a 完成后才有效。热路径上只有一次 `sum` 和一次 append，没有 D2H、没有通信。

EP 归约批到 `step()` 里做：把各层各微批的本地值 stack 成一个 `[层数, 微批数]` 矩阵，
**整个栈一共两次 `all_reduce`**（MAX + SUM）—— 逐微批的峰值由一次逐元素 MAX 一并得到，
不需要每个微批各来一次通信。payload 从 `L` 个 float 变成 `L×M` 个，仍在 1 KB 以内。

对照之下 PaddleFleet 那份实现是每层**每微批**一次 `all_gather_object` 外加一次 D2H
（`log_moe_balance` 里 `tokens_per_expert.cpu()`，它自己的注释就写了那是同步点），
43 层 × 16 微批 = 688 次各一 —— 这也是它为什么只能突发采样
（`trainer_utils.py:129-134` 的 `(step+1) % (interval*interval) < interval`）。

> 该族依赖 `moe_layer.moe_group` 判定 EP 组。同一 EP 组的各 rank 持有相同的层、跑相同数量的
> 微批，所以矩阵形状在组内一致，集合通信不会错配。某层触发次数不同时按最短的那次截断，
> 而不是改矩阵形状。

---

## 二、QK Stats Monitor (qk_stats)

> 详细文档: [qk_logits.md](./docs/qk_logits.md)

监控注意力 QK logit 的数值稳定性、集中度和 sink 现象。基于 Triton Online Softmax 内核高效计算。
新增 Sink Head 分类指标，基于 Sun et al. (2026) arXiv:2603.05498 的发现。

| # | 指标 | 日志键 | 公式 | 级别 | 诊断意义 |
|---|------|--------|------|------|----------|
| 1 | `max` | `qk_stats/.../max` | `max(sQ·K^T)` | 每层+全局 | Logit 最大值，数值稳定性 |
| 2 | `mean` | `qk_stats/.../mean` | `mean(valid_logits)` | 每层+全局 | Logit 基准量级 |
| 3 | `entropy_avg` | `qk_stats/.../entropy_avg` | `-Σ(p·log(p))` 均值 | 每层+全局 | 注意力集中度 |
| 4 | `sink` | `qk_stats/.../sink` | `mean(softmax[..., 0])` | 每层+全局 | Token-0 注意力权重 |
| 5 | `entropy_min` | `qk_stats/.../entropy_min` | `min(per_head_entropy)` | 每层+全局 | 最尖锐 head 的熵 |
| 6 | `entropy_max` | `qk_stats/.../entropy_max` | `max(per_head_entropy)` | 每层+全局 | 最分散 head 的熵 |
| 7 | `sink_head_ratio` | `qk_stats/.../sink_head_ratio` | `count(sink>θ)/N_heads` | 每层+全局 | Sink head 占比 |
| 8 | `sink_head_max` | `qk_stats/.../sink_head_max` | `max(sink_per_head)` | 每层+全局 | 最强 sink head 权重 |
| 9 | `sink_nonsink_gap` | `qk_stats/.../sink_nonsink_gap` | `mean(sink) - mean(nonsink)` | 每层+全局 | Sink/非Sink 分化度 |
| 10 | `attn_sink_logit` | `qk_stats/.../attn_sink_logit` | `mean(sink_logit)` | 每层+全局 | learned sink logit 量级漂移；稀疏层取 `attn_sink`，full 层取 `softmax_offset` |
| 11 | `{q,k,v}_norm_mean` | `qk_stats/.../{q,k,v}_norm_mean` | `mean(||q/k/v||₂)` | 每层+全局 | Dense core-attention 的平均向量尺度 |
| 12 | `{q,k,v}_norm_max` | `qk_stats/.../{q,k,v}_norm_max` | `max(||q/k/v||₂)` | 每层+全局 | Dense core-attention 的局部尺度峰值 |

其中 `s` 取 attention 层运行时的 `softmax_scale`；未显式设置时回退为 `1 / sqrt(head_dim)`。

> **learnable / off-by-one softmax 的 sink 折叠**：当 `core_attention.softmax_offset`
> 存在时（`softmax_type` 为 `learnable` / `off-by-one`），`entropy_avg` / `sink` 会把这个
> per-head sink logit 作为一列额外的无 key 列折进 softmax 分母，使统计与模型真实分布一致
> （`sink` 分子与 `max` / `mean` 仍只统计真实 key）。vanilla softmax（无 offset）行为不变。
> megatron 与 paddlefleet 两个后端语义一致。

### 混合注意力栈的层类型标签 (attn_type)

指标键会带上层类型前缀，避免不同注意力类型的统计混在同一张图里。

- **`csa_compress_ratios` 描述的栈**（由 `experimental_attention_variant="dsv4_hybrid"` 标记）：
  按每层 ratio 分类为 `mla`(-2) / `mqa`(-1) / `window`(0) / `csa`(2..127) / `hca`(128)。
  例：`qk_stats/layer_8/hca_entropy_avg`、`qk_stats/global_mla_entropy_avg`。
  这类 config 不设 `sliding_window`，层类型完全由 ratio 决定。
- **`sliding_window`（+ `window_attn_skip_freq`）描述的栈**：用 `Attention.__init__` 逐层算出的
  `is_swa` → `swa` / `full`。
- **两个字段都没有的栈**：无标签，键布局保持不变（向后兼容，避免已有看板断档）。

### CSA / HCA 层的统计口径

这类层的 query 不做全因果注意力，key 集合是「sliding window ∪ 压缩 KV」，且 softmax 分母里
还有一个 learned per-head `attn_sink`。qk_stats 对它们走独立路径：包裹
`CompressedSparseAttention.compressed_sparse_attn`，从调用入参直接取真实的
`topk_idxs` / `kv_full` / `attn_sink` / `softmax_scale`（层自身的 `v_head_dim ** -0.5`），
再用两段式稀疏 kernel（`qk_stats_sparse_partial_kernel`）在真实 key 集合上做单次 softmax
统计。索引取自模型自身，因此 document mask 下的文档边界重置、以及 learned indexer 的 top-k
选择都自动覆盖。

指标语义：

- `sink`：该 row **可达的最早真实 key** 的权重。有压缩段时指向首个压缩块（概括序列/文档开头），
  否则是窗口内最旧位置；dense causal 层下退化为 token-0。
- `entropy` / `sink` 的分布口径：**learned sink 会被 fold 进 softmax**，即分布跨越「真实 key + 1 个
  无对应 key 的 sink 列」，与模型实际计算的分布一致。稀疏层由 `compressed_sparse_attn` 的 `attn_sink`
  入参提供，full 层由 `core_attention.softmax_offset` 提供（`off-by-one` softmax 的冻结零 buffer 同样
  fold——`exp(0)` 照样进分母）。`max` / `mean` 只描述真实 key，不受 sink 影响。
- `attn_sink_logit`：learned sink 参数的均值，用于观察它在训练中的漂移。两类层的来源不同但语义一致
  （都是 per-head sink logit）：CSA/HCA 等稀疏层读 `compressed_sparse_attn` 的 `attn_sink` 入参；
  MLA/MQA 等 full 层在 `add_full_attention_sink_bias=true`（或 SWA 层的 `add_swa_attention_sink_bias`）
  时读 `core_attention.softmax_offset`——此时 `softmax_type` 被提升为 `learnable`，kernel 以
  `learnable_sink` 接收它。`off-by-one` softmax 的 `softmax_offset` 是冻结的零 buffer，没有可追踪的
  学习量级，故不单独上报这条曲线（但如上所述仍会 fold 进分布）。

启用 learned indexer 的层（`compress_ratio` 在 `[2, 127]` 且未开 `csa_dense_mode`）压缩 key
是运行时 top-k 选出的，统计会覆盖真实 key 的超集，注册 hook 时会打一条 warning。

读数注意 —— **`sink` 与 `entropy` 的绝对值不可跨层类型比较**：

- 粒度不同：dense 层的 key 是单个 token，压缩段的 key 是一个 `compress_ratio` 大小的聚合块
- key 集合大小不同：dense 层 O(S)，稀疏层 O(window + S/ratio)，熵的上界随之不同
- logit 尺度不同：压缩 key 由 Compressor 的独立参数产出，与原始 K 不在同一尺度，
  而 softmax 对 logit 尺度敏感
- 分母组成不同：稀疏层含 learned `attn_sink`，dense 路径默认没有对应项

因此只做**同类层之间的横向对比**（如 `hca_sink` 随层号的变化）或**同一 key 的纵向趋势**。
global 指标已按层类型拆分，不要再把不同类型平均到一起。

---

## 三、Massive Activation Monitor (massive_act)

> 详细文档: [massive_activation.md](./docs/massive_activation.md)

监控 Residual Stream 中的 Massive Activations（极端异常激活值）。
基于 Sun et al. (2026) "The Spike, the Sparse and the Sink" (arXiv:2603.05498) 的发现。

| # | 指标 | 日志键 | 公式 | 级别 | 诊断意义 |
|---|------|--------|------|------|----------|
| 1 | `channel_max` | `massive_act/.../channel_max` | `max(abs(H_i))` | 每层+全局 | 通道峰值，追踪 spike 生命周期 |
| 2 | `channel_median` | `massive_act/.../channel_median` | `median(max(abs(H_i), tokens))` | 每层+全局 | 通道峰值分布的基准量级 |
| 3 | `channel_p95` | `massive_act/.../channel_p95` | `p95(per_channel_max)` | 每层+全局 | 高分位通道幅度 |
| 4 | `channel_p99` | `massive_act/.../channel_p99` | `p99(per_channel_max)` | 每层+全局 | 极高分位通道幅度 |
| 5 | `channel_max_ratio` | `massive_act/.../channel_max_ratio` | `max / median` | 每层+全局 | 少数通道 outlier 严重度 |
| 6 | `massive_act_channel_count` | `massive_act/.../massive_act_channel_count` | `count(ch > 100*median)` | 每层+全局 | median-relative 异常通道数量 |
| 7 | `channel_count_gt_10` | `massive_act/.../channel_count_gt_10` | `count(per_channel_max > 10)` | 每层+全局 | 广泛高于基准量级的通道数量 |
| 8 | `channel_count_gt_20` | `massive_act/.../channel_count_gt_20` | `count(per_channel_max > 20)` | 每层+全局 | 高幅度通道数量 |
| 9 | `channel_count_gt_30` | `massive_act/.../channel_count_gt_30` | `count(per_channel_max > 30)` | 每层+全局 | 接近当前 1.5B FP4 峰值区间的通道数量 |
| 10 | `topk_channel_norm` | `massive_act/.../topk_channel_norm` | `norm(topk(3))` | 每层+全局 | 对应论文 Figure 1 |
| 11 | `activation_rms` | `massive_act/.../activation_rms` | `sqrt(mean(H_i^2))` | 每层+全局 | residual stream 整体 scale |
| 12 | `post_norm_sparsity` | `massive_act/.../post_norm_sparsity` | `mean(abs(x) < eps)` | 每层+全局 | 归一化后稀疏度 |
| 13 | `post_norm_cosine` | `massive_act/.../post_norm_cosine` | `cos_sim(tokens)` | 每层+全局 | 近常量向量检测 |
| 14 | `spectral_norm_max` | `massive_act/.../spectral_norm_max` | `max(post_rms / pre_rms)` | 每层+全局 | 层增益比上界，谱范数(σ_max)下界 |
| 15 | `spectral_norm_min` | `massive_act/.../spectral_norm_min` | `min(post_rms / pre_rms)` | 每层+全局 | 层增益比下界，最小奇异值(σ_min)上界 |
| 16 | `lipschitz_max` | `massive_act/.../lipschitz_max` | `max(‖∂L/∂x‖ / ‖∂L/∂y‖)` | 每层+全局 | 反向梯度增益上界，σ_max(J)下界 = Lipschitz 常数 |
| 17 | `lipschitz_min` | `massive_act/.../lipschitz_min` | `min(‖∂L/∂x‖ / ‖∂L/∂y‖)` | 每层+全局 | 反向梯度增益下界，σ_min(J)上界 |
| 18 | `logit_lens_entropy_mean` | `massive_act/.../logit_lens_entropy_mean` | `mean_t(H(softmax(final_norm(h)·Wᵀ)))` | 每层+全局 | logit-lens 预测熵均值，随深度下降=表征逐层"定型" |
| 19 | `logit_lens_logsumexp_mean` | `massive_act/.../logit_lens_logsumexp_mean` | `mean_t(logsumexp(final_norm(h)·Wᵀ))` | 每层+全局 | logit-lens 对数配分均值，追踪 logit 原始尺度 |
| 20 | `logit_lens_cross_entropy_mean` | `massive_act/.../logit_lens_cross_entropy_mean` | `mean_t(log_z − l[y])` | 每层+全局 | logit-lens 逐层交叉熵，末层≈LM loss |
| 21 | `hidden_spectral_entropy` | `massive_act/.../hidden_spectral_entropy` | `-Σ p_i log p_i, p_i=σ_i²/‖h‖_F²` | 每层+全局 | post-norm 隐状态谱熵(有效秩)，表征多样性/秩坍缩 |
| 22 | `{position}_{rms,abs_max,abs_p99,outlier_ratio}` | `massive_act/.../{position}_*` | 五个真实执行位置的尺度与 tail | 每层+全局 | 定位异常由 Attention、FFN/MoE 还是 residual 累积产生 |
| 23 | `{attn,ffn}_update_rms_ratio` | `massive_act/.../{attn,ffn}_update_rms_ratio` | `RMS(residual_after-residual_before)/RMS(residual_before)` | 每层+全局 | 模块对 residual stream 的实际写入比例 |

PaddleFleet 的 `position` 为 `layer_input`、`attn_out`、`post_attn_residual`、
`ffn_or_moe_out`、`post_ffn_residual`。两个 residual 位置从真实 forward 边界采样，update ratio
由实际 residual 差分计算，因此包含 bias、dropout、BDA 与 mHC residual mixing 的影响。

> **注**: 指标 18-20 (`logit_lens_*`) 为**默认关闭**的可选项（熵/logsumexp 由 `log_logit_lens_entropy=True`
> 开启，交叉熵由 `log_logit_lens_cross_entropy=True` 开启）。
> 将每层残差经 LM head 投影到 vocab logits，同一次投影求三组量（**仅报告 token 均值**）：softmax 熵
> `H(p) = -Σ p·log p ∈ [0, log(vocab)]`（随深度下降=逐层定型）、对数配分 `log Z = logsumexp(l)`
> （softmax 归一化项，追踪 logit 原始尺度，直接复用熵里的 `log_z`，零额外开销），以及对真实下一个 token 的
> 交叉熵 `CE = log Z − l[y]`（同样复用 `log_z`，只多一次 gather；**末层 CE ≈ LM loss**）。相当于每监控层每监控步
> 做一次 LM-head 前向，开销较大，建议配合 `monitor_interval` 与 `logit_lens_layers`（只监控指定层）使用。
> 投影**按 token 分块**（`logit_lens_chunk_size`，默认 1024），任一时刻只物化一个 `[chunk, vocab]` tile，
> 绝不 materialize 完整 `[tokens, vocab]` logits。默认先过 `decoder.final_layernorm`
> （`logit_lens_apply_final_norm=True`，LM head 是在 final-norm 后的表征上训练的）。熵用 `H = log_z − E_p[l]`
> 写法（`log_z = torch.logsumexp(l)`、`E_p[l] = Σ softmax(l)·l`），直接用 `torch.logsumexp` / `torch.softmax`，
> 无手写 exp/log。**交叉熵 loss-mask 说明**：label 由 head-owning chunk 上的顶层 forward pre-hook 从
> `model.forward(labels=...)` 捕获并按 seq-major 对齐；loss_mask 在模型 forward 之外施加，monitor 拿不到，故报告
> 的是**未加权 token 均值 CE**（= LM loss up to loss-mask 加权）。label 缺失/对齐失败时当步不产出 CE（不报错）。
> **PP 覆盖**: 只有持有 LM head 权重的 PP stage（最后 stage / tied-embedding stage）会计算这组指标；
> 其余 stage `_resolve_lm_head` 返回 `(None, None)`，不 attach hook、不声明 key、彻底 no-op（不做权重广播）。
> **暂不支持 vocab-parallel TP**：`compute_logit_lens_entropy` 断言 `tp_size <= 1`（TP 切 vocab 时归一化需跨 rank
> reduce，无法与 `torch.logsumexp` 融合，后续再加）。
>
> **注**: 指标 21 (`hidden_spectral_entropy`) 为**默认关闭**的可选项（`log_hidden_spectral_entropy=True` 开启）。
> 对 post-RMSNorm 隐状态 `h∈R^{n,d}` 的谱（矩阵/von Neumann）熵：`p_i=σ_i²/‖h‖_F²` 是较小 Gram 矩阵
> （`hᵀh` 若 n≥d 否则 `h hᵀ`）的归一化特征值，`H=-Σ p_i log p_i∈[0, log(min(n,d))]`，`exp(H)` 即有效秩——
> 衡量 token 集张成多少个有效方向（低=秩坍缩，高=方向丰富）。用 `torch.linalg.eigvalsh` 一次 GPU 调用（无
> full SVD、无 host sync）。这是 **set-level 非线性量**，故跨 rank/microbatch 的均值只是近似（mean-of-shard-
> entropies ≠ 全 token 熵），按 per-shard 值在 flush 时平均，作为坍缩趋势信号可接受。

### 核心洞察

Massive activations 是 pre-norm Transformer 的**架构副产品**，独立于模型性能：
- PPL 不变但 spike 暴涨 → 部署时量化精度严重退化
- Spike token 经 RMSNorm 后变成稀疏近常量向量 → 为 attention sink 创造条件
- 监控 spike lifecycle (rise-plateau-fall) 可定位 step-up/step-down blocks
- `channel_max_ratio` 用于识别少数通道 outlier；`channel_p95/p99`、`activation_rms`、`channel_count_gt_10/20/30` 用于识别 residual stream 整体 scale growth。

---

## 四、PLE Health Monitor (ple_health)

> 详细文档: [ple_health.md](./docs/ple_health.md)

监控 Per-Layer Embedding 双分支架构的健康状况。

| # | 指标 | 日志键 | 公式 | 级别 | 诊断意义 |
|---|------|--------|------|------|----------|
| 1 | `token_ple_norm` | `ple_health/global_token_ple_norm` | `mean(\|\|token_ple\|\|₂, dim=-1)` | 全局 | Token 分支信号强度 |
| 2 | `proj_ple_norm` | `ple_health/global_proj_ple_norm` | `mean(\|\|proj_ple × H^{-0.5}\|\|₂, dim=-1)` | 全局 | 投影分支信号强度 |
| 3 | `per_layer_inputs_norm` | `ple_health/global_per_layer_inputs_norm` | `mean(\|\|(token+proj) × 2^{-0.5}\|\|₂, dim=-1)` | 全局 | 合并信号量级 |
| 4 | `token_proj_cosine` | `ple_health/global_token_proj_cosine` | `mean(cosine_sim(token, proj))` | 全局 | 双分支冗余度 (→1 冗余) |
| 5 | `residual_ratio` | `ple_health/layer_{i}/residual_ratio` | `\|\|output - input\|\| / \|\|input\|\|` | 每层+全局 | PLE 贡献幅度 |
| 6 | `gate_activation_mean` | `ple_health/layer_{i}/gate_activation_mean` | `mean(\|act_fn(gate_out)\|)` | 每层+全局 | 门控激活强度 |
| 7 | `gate_sparsity` | `ple_health/layer_{i}/gate_sparsity` | `(\|act\| < 0.01).mean()` | 每层+全局 | 死门控单元占比 |

---

## 五、mHC Health Monitor (mhc_health)

> 详细文档: [mhc_health.md](./docs/mhc_health.md)

监控 mHC (Manifold-Constrained Hyper-Connections) 层的三个 per-token 映射 `h_pre` / `h_post` / `h_res`，
以及 `x_{l+1} = H_resᵀ x_l + H_postᵀ F(H_pre x_l)` 两项的相对大小。
只在模型开启 mHC 层时生效，mHC 类无法 import 或模型不含 `HyperConnectionTransformerLayer` 时该 monitor 为
彻底 no-op。每层含两个 hyper-connection 模块（`attn` / `mlp`），各产出以下 29 个指标，
指标名以 `attn_` / `mlp_` 前缀区分；`branch_residual_share_max`、`h_res_logits_max`、
`h_res_logits_grad_max`、`composite_amax_gain_{fwd,bwd}_max`、`h_{pre,post}_logits_max`、
`bias_{pre,post,res}_abs_max` 取极大值，`h_res_logits_min`、`h_res_logits_grad_min` 与
`h_{pre,post}_logits_min` 取极小值，其余按 token/batch 求均值。

第 17-29 条对应论文 Eq. (7) 的静态与 pre-sigmoid 部分（`alpha` / `bias` / 门控 logits），
**目前仅 paddlefleet 后端实现**，megatron 后端仍为前 16 条。

| # | 指标 | 日志键 | 公式 | 级别 | 诊断意义 |
|---|------|--------|------|------|----------|
| 1 | `{c}_h_pre_mean` | `mhc_health/layer_{i}/{c}_h_pre_mean` | `mean(h_pre)` | 每层+全局 | 聚合门均值 |
| 2 | `{c}_h_pre_std` | `mhc_health/layer_{i}/{c}_h_pre_std` | `std(h_pre)` | 每层+全局 | 聚合门离散度 |
| 3 | `{c}_h_post_mean` | `mhc_health/layer_{i}/{c}_h_post_mean` | `mean(h_post)` | 每层+全局 | 扩展门均值 |
| 4 | `{c}_h_post_std` | `mhc_health/layer_{i}/{c}_h_post_std` | `std(h_post)` | 每层+全局 | 扩展门离散度（token 与 stream 混合） |
| 5 | `{c}_h_post_stream_concentration` | `mhc_health/layer_{i}/{c}_h_post_stream_concentration` | `mean_t(max_i h_post / mean_i h_post)` | 每层+全局 | stream 集中度，值域 `[1, n]`：1 = 均匀，n = 全压在一条流上 |
| 6 | `{c}_h_post_token_std` | `mhc_health/layer_{i}/{c}_h_post_token_std` | `std_t(mean_i h_post)` | 每层+全局 | token 间总门量离散度；→0 不等于每条 stream 的门都为常数 |
| 7 | `{c}_branch_residual_share` | `mhc_health/layer_{i}/{c}_branch_residual_share` | `mean_t(b / (b + r))`，`b = ‖H_postᵀ F(·)‖_F`、`r = ‖H_resᵀ x_l‖_F` | 每层+全局 | **本层是否还在写入残差流**，值域 `[0, 1]`：0.5 = 两项等量 |
| 8 | `{c}_branch_residual_share_max` | `mhc_health/layer_{i}/{c}_branch_residual_share_max` | 同上取最坏 token | 每层+全局 | 单 token 被分支主导的程度（贴 1 = 存在近零残差 token） |
| 9 | `{c}_amax_gain_fwd` | `mhc_health/layer_{i}/{c}_amax_gain_fwd` | `mean_t(max_i \|Σ_j h_res_ji\|)`（`h_res` 的列和 = `h_resᵀ` 的行和） | 每层+全局 | 单层前向最坏放大。列和被 Sinkhorn 最后一步钉死为 1，故这条是不变量守卫（见下） |
| 10 | `{c}_amax_gain_bwd` | `mhc_health/layer_{i}/{c}_amax_gain_bwd` | `mean_t(max_j \|Σ_i h_res_ji\|)`（`h_res` 的行和） | 每层+全局 | 单层反向最坏放大。承载 Sinkhorn 截断迭代的收敛残差 |
| 11 | `{c}_h_res_logits_min` | `mhc_health/layer_{i}/{c}_h_res_logits_min` | `min z`，`z` = Sinkhorn 输入 logits | 每层+全局 | 迭代前 residual-mixing logits 的最小值。按 step 内 layer/microbatch **min** 归约 |
| 12 | `{c}_h_res_logits_max` | `mhc_health/layer_{i}/{c}_h_res_logits_max` | `max z` | 每层+全局 | 同上取最大值。与 min 联合观察 raw logits 的 signed 范围。按 **max** 归约 |
| 13 | `{c}_h_res_logits_grad_min` | `mhc_health/layer_{i}/{c}_h_res_logits_grad_min` | `min dL/dz`，`z` = Sinkhorn 输入 logits | 每层+全局 | 迭代前 logits 的未缩放 activation gradient 最小值。按 step 内 layer/microbatch **min** 归约 |
| 14 | `{c}_h_res_logits_grad_max` | `mhc_health/layer_{i}/{c}_h_res_logits_grad_max` | `max dL/dz` | 每层+全局 | 同上取最大值。AMP loss scale 被反除，不除 gradient accumulation。按 **max** 归约 |
| 15 | `{c}_composite_amax_gain_fwd_max` | `mhc_health/layer_{i}/{c}_composite_amax_gain_fwd_max` | 入口到当前 branch 的 attention/MLP 交错 prefix 最大绝对行和 | 每层+全局 | 按 **max** 跨 microbatch/层归约 |
| 16 | `{c}_composite_amax_gain_bwd_max` | `mhc_health/layer_{i}/{c}_composite_amax_gain_bwd_max` | 当前 branch 到模型尾部的 attention/MLP 交错 suffix 最大绝对列和 | 每层+全局 | 按 **max** 跨 microbatch/层归约 |
| 17 | `{c}_h_pre_logits_min` | `mhc_health/layer_{i}/{c}_h_pre_logits_min` | `min(r·proj[:n]·α_pre + b_pre)` | 每层+全局 | `H_pre` 进 sigmoid 前的 logit 最小值，判断聚合门是否饱和到 0。按 **min** 归约 |
| 18 | `{c}_h_pre_logits_max` | `mhc_health/layer_{i}/{c}_h_pre_logits_max` | 同上取最大值 | 每层+全局 | 判断聚合门是否饱和到 1。按 **max** 归约 |
| 19 | `{c}_h_post_logits_min` | `mhc_health/layer_{i}/{c}_h_post_logits_min` | `min(r·proj[n:2n]·α_post + b_post)` | 每层+全局 | `H_post` 的 pre-sigmoid logit 最小值（因子 2 在 sigmoid 之外，不进 logit）。按 **min** 归约 |
| 20 | `{c}_h_post_logits_max` | `mhc_health/layer_{i}/{c}_h_post_logits_max` | 同上取最大值 | 每层+全局 | 扩展门饱和哨兵。按 **max** 归约 |
| 21 | `{c}_alpha_pre` | `mhc_health/layer_{i}/{c}_alpha_pre` | `α_pre`（标量参数） | 每层+全局 | 动态映射门控因子，从 `mhc_init_gating_factor` 起漂移 |
| 22 | `{c}_alpha_post` | `mhc_health/layer_{i}/{c}_alpha_post` | `α_post` | 每层+全局 | 同上 |
| 23 | `{c}_alpha_res` | `mhc_health/layer_{i}/{c}_alpha_res` | `α_res` | 每层+全局 | 同上；直接决定 Sinkhorn 输入 logits 的尺度 |
| 24 | `{c}_bias_pre_mean` | `mhc_health/layer_{i}/{c}_bias_pre_mean` | `mean(bias[:n])` | 每层+全局 | `b_pre` 静态偏置均值（初始为 0） |
| 25 | `{c}_bias_pre_abs_max` | `mhc_health/layer_{i}/{c}_bias_pre_abs_max` | `max\|bias[:n]\|` | 每层+全局 | 单元素跑飞哨兵。按 **max** 归约 |
| 26 | `{c}_bias_post_mean` | `mhc_health/layer_{i}/{c}_bias_post_mean` | `mean(bias[n:2n])` | 每层+全局 | `b_post` 均值 |
| 27 | `{c}_bias_post_abs_max` | `mhc_health/layer_{i}/{c}_bias_post_abs_max` | `max\|bias[n:2n]\|` | 每层+全局 | 同上取极值。按 **max** 归约 |
| 28 | `{c}_bias_res_mean` | `mhc_health/layer_{i}/{c}_bias_res_mean` | `mean(bias[2n:])` | 每层+全局 | `b_res` 均值 |
| 29 | `{c}_bias_res_abs_max` | `mhc_health/layer_{i}/{c}_bias_res_abs_max` | `max\|bias[2n:]\|` | 每层+全局 | 同上取极值。按 **max** 归约 |

`{c}` ∈ `{attn, mlp}`。第 21-29 条是 step 级参数量，在 `on_optimizer_begin`（optimizer step 之前）读一次参数，
不在热路径，且仅在本 step 确实跑过被监控的 forward 时记录。

除上述 29 个标量外，paddlefleet 后端还额外产出 **每元素展开的映射序列**（`n² + 2n` 条 / hc 模块，`n = 4` 时 24 条，
每层两个模块共 48 条）——即把 `h_res` 的 4×4 矩阵和 `h_pre` / `h_post` 两条 n 维向量逐元素记录，供还原论文
Figure 10 那类映射热图使用：

| 指标 | 日志键 | 公式 | 级别 | 诊断意义 |
|------|--------|------|------|----------|
| `{c}_h_res_cell{k}` | `mhc_health/layer_{i}/{c}_h_res_cell{k}` | `mean_t(h_res[t, r, c])`，`k = r·n + c` 行主序 | 仅每层 | 残差混合矩阵的单个 cell。`h_res` 按 `compute_mappings` 返回的朝向记录（与 `amax_gain_fwd` 读的列和同一朝向，**非** composite 链式用的转置） |
| `{c}_h_pre_idx{j}` | `mhc_health/layer_{i}/{c}_h_pre_idx{j}` | `mean_t(h_pre[t, j])` | 仅每层 | 第 `j` 条流的聚合门开度 |
| `{c}_h_post_idx{j}` | `mhc_health/layer_{i}/{c}_h_post_idx{j}` | `mean_t(h_post[t, j])` | 仅每层 | 第 `j` 条流的扩展门开度 |

这三组走 `declare_layer_vector` / `record_layer_vector`：
热路径上每个映射只有一次 `add_`，而不是每个元素一个 kernel；并且**不派生 `global_*` 键**——单个 cell 在
43 层上的均值没有判读意义。`log_per_layer=False` 时这三组整体不产出。

指标公式、采集路径、健康判读及论文 composite 定义统一维护在
[`docs/mhc_health.md`](docs/mhc_health.md)，README 不再重复展开。

---

## 六、VHA Health Monitor (vha_health)

> **仅 paddlefleet 后端**。只在 `use_vha_attention=true` 时生效——模型不含 VHA postmix 参数时该 monitor 为彻底
> no-op（不 wrap、不产生指标）。

VHA (Virtual Head Attention) 把**物理 query head 减半**、`H_kv` 设为 2，再用两个**严格线性**的算子把表达力
补回来：

- **Q Premix** — 在每个 KV group 内对 query 做**近恒等**线性变换，把减半的物理 query head 扩展成
  `H_v = H'_q × H_kv` 个虚拟头，恢复注意力多样性。PaddleFleet 初始化为 `I + 0.1/√d · N(0,1)`
  （`q_head_dim == head_dim` 时；否则退化为缩放正交初始化）。需开 `use_vha_premix`，未开启不声明 premix 指标。
- **Linear Postmix** — 在注意力输出的 head 轴上做低秩恒等残差 `I + A Bᵗ`（`r = H'_q`），融合跨组的 inter-head
  特征。位置在 gate 与 `o_proj` **之前**；PaddleFleet 的参数名是 `vha_postmix_U` / `vha_postmix_V`，即 `A` / `B`。
  `B` 零初始化，故初始为**精确恒等**、`delta ≡ 0`。

两者在推理时都会被折叠（Premix 进 `q_proj`、Postmix 进 `o_proj`），模型退化成标准 GQA-2。**也就是说这两个矩阵
只在训练期存在，训练期是唯一能观测其结构的时机。**

这两处对其他 monitor 都不可见：`qk_stats` 采样点在 `core_attention` 的**输入**（Postmix 发生在其后），
`massive_act` 只看残差流（Postmix 的效果被 `o_proj` 和残差加法稀释）。

| # | 指标 | 日志键 | 公式 | 级别 | 诊断意义 |
|---|------|--------|------|------|----------|
| 1 | `{b}_postmix_delta_rel_mean` | `vha_health/layer_{i}/{b}_postmix_delta_rel_mean` | `mean_t(‖delta_t‖₂ / ‖mixed_t‖₂)` | 每层+全局 | **首要指标**：从 0 起长，回答「Postmix 是否在学」 |
| 2 | `{b}_postmix_delta_rel_max` | `vha_health/layer_{i}/{b}_postmix_delta_rel_max` | `max_t(‖delta_t‖₂ / ‖mixed_t‖₂)` | 每层+全局 | 最坏 token 的修正幅度，看突刺 |
| 3 | `{b}_postmix_amax_gain_max` | `vha_health/layer_{i}/{b}_postmix_amax_gain_max` | `max_t(‖out_t‖_∞ / ‖mixed_t‖_∞)` | 每层+全局 | 低精度溢出余量（同 mHC `amax_gain` 口径） |
| 4 | `{b}_postmix_uv_sigma_max` | `vha_health/layer_{i}/{b}_postmix_uv_sigma_max` | `σ_max(A Bᵗ)` | 每层+全局 | 混合算子偏离恒等的强度上界 |
| 5 | `{b}_postmix_uv_eff_rank` | `vha_health/layer_{i}/{b}_postmix_uv_eff_rank` | `exp(-Σ p log p)`, `p = σ/Σσ` | 每层+全局 | 是否守在低秩预算内（**涨= 警告**，见下） |
| 6 | `{b}_postmix_offdiag_ratio` | `vha_health/layer_{i}/{b}_postmix_offdiag_ratio` | `‖M − diag(M)‖_F / ‖M‖_F` | 每层+全局 | 趋 0 = 退化成 per-head 缩放，无真实跨头混合 |
| 7 | `{b}_postmix_u_fro` / `{b}_postmix_v_fro` | `vha_health/layer_{i}/{b}_postmix_{u,v}_fro` | `‖A‖_F` / `‖B‖_F` | 每层+全局 | 哪个因子在长 |
| 8 | `{b}_head_out_norm_max/min/std` | `vha_health/layer_{i}/{b}_head_out_norm_*` | per-head 输出范数（token 均值）的 max/min/std | 每层+全局 | 某个 head 被压成 0 或被放大 |
| 9 | `{b}_postmix_head_cos_mean` | `vha_health/layer_{i}/{b}_postmix_head_cos_mean` | token 均值 head 向量间的平均 \|cos\| | 每层+全局 | 趋 1 = 各 head 退化成彼此的复制 |
| 10 | `premix_identity_dev` | `vha_health/layer_{i}/premix_identity_dev` | 组均值 `‖W_g − I‖_F` | 每层+全局 | Premix 离「无操作」多远 |
| 11 | `premix_group_div_ratio` | `vha_health/layer_{i}/premix_group_div_ratio` | `mean_{g<g'}‖W_g − W_g'‖²_F / mean_g‖W_g − I‖²_F` | 每层+全局 | group-conditioned query specialization：0 = 各组学成同一个变换，2 = 完全独立 |
| 12 | `premix_sigma_max` | `vha_health/layer_{i}/premix_sigma_max` | `σ_max(W)` | 每层+全局 | Premix 谱范数（近恒等时约 1） |
| 13 | `premix_orth_dev` | `vha_health/layer_{i}/premix_orth_dev` | `‖WᵀW − I‖_F` | 每层+全局 | 仅 `q_head_dim ≠ head_dim`（正交初始化）时代替 10~11 |

`{b}` ∈ `{main, sparse}`：MQA 栈的 block-sparse 分支持有独立的 `sparse_vha_postmix_U/V`，两套参数分开统计。
混合栈上注意力类型同样会前置到指标名，例如 `vha_health/layer_5/hca_main_postmix_delta_rel_max`。
premix 的指标集由权重形状决定（方阵 → 10~12；非方阵 → 12~13；`H_kv = 1` 时无 11），形状在注册时已知，
因此仍满足「declare 完再 allocate」。

### 健康判读

- `postmix_delta_rel_mean` 长期贴 0 → Postmix 没在学（检查 `B` 的梯度/学习率是否被误置零）。
- `postmix_delta_rel_max` 出现数量级突刺，或 `postmix_amax_gain_max` 明显 > 1 → 低精度下有溢出风险。
- `postmix_offdiag_ratio` 趋 0 而 `postmix_uv_sigma_max` 在长 → 学成了 per-head 缩放，跨头混合的容量没用上。
- `postmix_uv_eff_rank` 持续爬向配置的 `r` → **警告而非健康**。低秩瓶颈是设计上的结构正则，VHA 论文的消融显示
  `r = 32/128` 虽然训练 loss 更低但评测更差；有效秩顶满通常伴随过拟合倾向。
- `postmix_head_cos_mean` 趋 1 或 `head_out_norm_min` 趋 0 → head 退化。
- `premix_group_div_ratio` 趋 0 → 各 KV group 学成同一个 query 变换，虚拟头没有按组分化；趋 2 → 各组完全独立。
  论文观测到的是「显著不同但并非独立」，两端都是异常。

---

## 七、Attn Update Monitor (attn_update)

监控 QK 乘积增量（arXiv:2606.28116 §3.4）。记采样间隔 δ 内 `ΔW = W_t − W_{t−δ}`，
QK 乘积的变化可精确拆成 `Δ₁ = Δ₂ + Δ₃`，本 monitor 监控其中两项：

```
Δ₂ = ΔW_q W_kᵀ + W_q ΔW_kᵀ = [ΔW_q, W_q][W_k, ΔW_k]ᵀ     (一阶，秩 ≤ 2·head_dim)
Δ₃ = ΔW_q ΔW_kᵀ                                            (二阶，秩 ≤ head_dim)
```

**Δ₁ 故意不监控**：论文 Eq (5) 给出 `‖Δ₂‖_F = O(‖W‖_F‖ΔW‖_F)` 而 `‖Δ₃‖_F = O(‖ΔW‖_F²)`，
早中期 `‖ΔW‖_F ≪ ‖W‖_F` 时 `Δ₁ ≈ Δ₂`，再花一个完整的 `2·head_dim` 核去重复测 Δ₂ 不值。
**Δ₃ 保留**是因为它是唯一活在 Q/K 更新**耦合**空间里的项（论文：`can become visible before spectral
concentration is obvious in the separate factor updates ΔW_q, ΔW_k`）。但要清楚它比 Δ₂ **晚**报警
（论文实测排序 `Δ₁ ≈ Δ₂ < Δ₃ < ΔW`），买到的是一个新维度加一级便宜的确认，不是更早的预警。

当因子宽度小于 `hidden` 时把谱计算挪到 thin-QR 核 `R_A R_Bᵀ` 上，`hidden × hidden` 的稠密乘积从不物化。

三个指标都只是 `σ²` 的函数，所以用 Gram + `eigvalsh` 取谱，并且**所有层所有项拼成一次批量调用**
（先按 shape 分桶——Δ₂ 的核是 `2·head_dim` 宽、Δ₃ 是 `head_dim`，只有同形状能共享一次批量求解；
再按 `_MAX_BATCH_BYTES` 分块，`hidden=1024` 时每批 64 个）。

实测（H 系列单卡，`hidden=1024`、`head_dim=128`、18 层 × 1 head）：

- 逐层 `svd` 2054 ms → 逐层 Gram+`eigvalsh` 420 ms → 批量 Gram+`eigvalsh` 277 ms
- 只算 Δ₂ 138 ms；只算 Δ₃ 72 ms（Δ₂ 的 0.52×）；两项分桶合算 211 ms
- 即**加上 Δ₃ 是 ~1.5×，不是 ~1.1×** —— thin-QR 是 `O(d·r²)`，在这个尺寸上盖过了 `O(r³)` 的特征分解，
  所以核宽减半并不等于成本降到 1/8
- 端到端一个采样步 ~0.7 s，δ=10 摊到 4.0 s 的训练步上约 **1.8%**（只有 Δ₂ 时约 1.2%）

绝对值会随 GPU 争用浮动约 2×，看比例即可。

> Gram 平方会损失 `σ < sqrt(eps)·σ_max` 那部分奇异值的精度，但 α=2 下这些分量权重为 ~0；
> 单元测试把三个指标都对着 float64 稠密 SVD 钉住，相对误差 ≤ 3e-5。若把 `spectrum_alpha` 调小到 1，
> 小奇异值权重变大，这条论证不再成立。

本 monitor **不注册 forward hook**：它在 `step()` 边界直接读 QK 权重，把上一次读数作为基准点，
因此采样间隔 δ = 本 monitor 自己的间隔（`sample_interval`，未设置时回退到 `monitor_interval`），
且**第一个采样步只建立基准点、不产出指标**。

| # | 指标 | 日志键 | 公式 | 级别 | 诊断意义 |
|---|------|--------|------|------|----------|
| 1 | `delta2_norm` | `attn_update/layer_{i}/delta2_norm` | `‖Δ₂‖_F` | 每层+全局 | 一阶增量的整体幅度 |
| 2 | `delta2_stable_rank` | `attn_update/layer_{i}/delta2_stable_rank` | `‖Δ₂‖_F² / ‖Δ₂‖₂²` | 每层+全局 | top-1 奇异值主导程度的倒数 |
| 3 | `delta2_singular_spectrum` | `attn_update/layer_{i}/delta2_singular_spectrum` | `S₂ = exp(−Σ pᵢ log pᵢ), pᵢ = σᵢ²/Σσⱼ²` | 每层+全局 | 全谱有效秩，论文中最灵敏的坍缩前兆 |
| 4 | `delta3_norm` | `attn_update/layer_{i}/delta3_norm` | `‖Δ₃‖_F` | 每层+全局 | 二阶耦合项幅度，正常时应远小于 `delta2_norm` |
| 5 | `delta3_stable_rank` | `attn_update/layer_{i}/delta3_stable_rank` | `‖Δ₃‖_F² / ‖Δ₃‖₂²` | 每层+全局 | Q/K 更新耦合的 top-1 主导程度 |
| 6 | `delta3_singular_spectrum` | `attn_update/layer_{i}/delta3_singular_spectrum` | 同上，作用在 Δ₃ 的谱上 | 每层+全局 | Q/K 更新耦合变得 coherent 时下降 |

`resolve_qk_factors` 只从权重重建每个 head 的 QK 电路，支持四种布局，按从特殊到通用的顺序匹配：

- **DSv4-hybrid**：`linear_q_down_proj → q_layernorm → linear_q_up_proj`，`linear_kv_proj → kv_layernorm`，
  单个共享 KV head。
- **MLA**：`q_proj` 或 `q_a_proj → q_a_layernorm → q_b_proj`；K 只经由
  `kv_a_proj_with_mqa → kv_a_layernorm → kv_b_proj` 依赖 hidden。只有 NoPE 半边参与内容项，
  所以这里的电路宽度是 `qk_nope_head_dim` 而非 `q_head_dim`。切片先做、再与 latent 相乘，
  否则完整乘积是 `[hidden, num_heads × q_head_dim]`，在 128 head 上会是数百 MB。
- **融合 `qkv_proj` / `linear_qkv`**：输出按 KV head 分组为 `Q|(gate)|K|V`，分组算术会先与真实权重宽度
  对账，不符就整层跳过而不是错切。
- **独立 `q_proj` / `k_proj`**（含 GQA/MQA、VHA）：head 数从权重宽度推出，因此 TP 分片下每卡报自己实际持有的 head。

RMSNorm / QK-norm 的可学习 scale 在 QK 电路内部，故折进 `W_q` / `W_k`（per-head 与 per-layer 两种形状都认，
形状对不上时宁可不折也不猜）；输入相关的 `1/rms` 不是权重、不参与。不匹配任何布局的层直接跳过，不声明指标。
同一模型里混用布局时电路宽度可能不同，`_spectrum_metrics_over_pairs` 会先按 shape 分桶再各自批量求解。

配置参数：

- `sample_interval` (默认 `None`)：本 monitor 的采样间隔 δ，单位训练步；`None` 时用全局 `monitor_interval`。
  单独设置的意义在于这里的成本是每层每次采样一次特征分解，与 forward 无关，可以比 hook 类 monitor 采得更稀。
  另外 δ 不宜太小：论文 Observation 1 靠的是窗口内累积（coherent 分量按 `n` 线性增长、残差按 `√n`），
  δ=1 时 n=1，信号会被噪声压住。
- `num_heads_monitored` (默认 `1`)：每层取前若干 head，逐层指标为这些 head 的均值。每个 head 需要两次
  对称特征分解（Δ₂ 的 `2·head_dim` 核 + Δ₃ 的 `head_dim` 核），成本随该值线性增长。
- `spectrum_alpha` (默认 `2.0`)：`S_α` 的权重指数，论文推荐 2（即 "singular spectrum"）。

> 判读方式：**故障凭证是曲线相对健康基线的偏移，而不是低秩本身**——健康训练的更新本就带低秩结构
> （LoRA / GaLore / Muon 正是利用这一点），所以需要与 baseline run 对比，而非单看绝对值。

> **论文的绝对提前量不可直接搬用。** 论文报的「Δ₂ 比 loss 发散早 ~17,000 步、比 ΔW 谱早 ~8,000 步」
> 是在 plain MHA 上测的。§6 Limitations 明确说 MLA/GQA/MQA/DSA 的 QK 算子被压缩投影和共享 head 中介，
> `Δ₂ 代理需要为每个变体重新推导`，检测阈值是 variant-specific 的。本实现的代数对任意 `W_q`/`W_k`
> 都精确（`A_tB_tᵀ − ABᵀ = ΔA·Bᵀ + A·ΔBᵀ + ΔA·ΔBᵀ` 是恒等式，不要求它们是原始参数），
> 但 Observation 1 的机制是在**参数级** ΔW 上陈述的；MLA/DSv4-hybrid 下 `W_q^eff` 是多个参数的复合，
> 低精度 FA 打进各参数更新的 coherent 结构穿过 latent 之后还剩多少，论文没有回答。
> 结论：只与同架构的 baseline run 对比，不要拿论文数字当阈值。

> 尚未实现：§3.2 的 raw ΔW 谱指标（`ΔW_q` / `ΔW_k` 各自的 norm / stable rank / singular spectrum）。
> 它是比 Δ₂ 晚约 8,000 步的**第二级确认信号**，缺了它时 Δ₂ 单独波动无法交叉验证。

---

## 八、MLP Update Monitor (mlp_update)

监控 MoE 专家 MLP 的参数增量。记采样间隔 δ 内

```
ΔW_m^(l,e) = W_m,t^(l,e) − W_m,t−δ^(l,e),     m ∈ {gate, up, down}
```

`attn_update` 的 MoE 对应物：同样只读权重、不挂 forward hook，基准点是上一次被监控的读数。

**三块矩阵分开算，绝不拼在一起。** 因为它们在 `f_e(x) = W_down[SiLU(W_gate x) ⊙ (W_up x)]` 里
职能、维度、梯度尺度都不同——gate 控制哪些中间通道被激活，up 生成中间特征，down 把中间特征映射回模型维度。
拼起来算谱会让尺度大的那块主导结果，也无法定位异常来自哪一块。PaddleFleet 把 gate/up 融合存在
`weight1 [E, in, 2·inter]` 里，本实现沿**中间维**（最后一维）切成两半，前一半是过 SiLU 的 gate
（依据 `moe_expert.py` 的 `glu(x): hidden_act(x[0]) * x[1]`）；down 是 `weight2`，单独算。

**归约顺序**：每个 (层, 专家, 矩阵) 三元组各自算完，再在层内对专家汇总，永不跨专家拼矩阵。

| # | 指标 | 键 | 公式 | 粒度 | 含义 |
|---|------|-----|------|------|------|
| 1 | `{m}_rel_update_mean` | `mlp_update/layer_{i}/{m}_rel_update_mean` | `mean_e(r_m)`, `r_m = ‖ΔW_m‖_F/(‖W_m‖_F+ε)` | 每层+全局 | 该投影的整体更新水平 |
| 2 | `{m}_rel_update_median` | 同上 | `median_e(r_m)` | 每层+全局 | 抗异常专家干扰的中心值 |
| 3 | `{m}_rel_update_p10` / `_p90` | 同上 | `quantile_e(r_m, .1/.9)` | 每层+全局 | 专家间离散程度，`p90−p10` 拉大 = 学习速度不均衡 |
| 4 | `{m}_rel_update_max` | 同上 | `max_e(r_m)` | 每层+全局 | 捕捉更新突然异常增大的专家 |
| 5 | `{m}_rel_update_min` | 同上 | `min_e(r_m)` | 每层+全局 | 捕捉几乎不更新的「沉睡专家」 |
| 6 | `{m}_delta_norm_mean` | 同上 | `mean_e(‖ΔW_m‖_F)` | 每层+全局 | 绝对幅度，看全局尺度漂移（LR / grad scale） |
| 7 | `{m}_stable_rank_{mean,min,max}` | 同上 | `‖ΔW_m‖_F²/‖ΔW_m‖₂²` | 每层+全局 | 更新是否集中到少数奇异方向 |
| 8 | `{m}_singular_entropy_{mean,min,max}` | 同上 | `−Σ pᵢ log pᵢ, pᵢ = σᵢ²/Σσⱼ²` | 每层+全局 | 全谱利用率，需 `log_spectrum=True` |
| 9 | `update_zmax_{max,p90,min}` | `mlp_update/layer_{i}/update_zmax_max` | `S^(l,e) = max_m z(r_m^(l,e))` | 每层+全局 | 专家级汇总分数 |
| 10 | `shared_{m}_rel_update` | 同上 | `r_m` of shared expert | 每层+全局 | 共享专家，作路由专家的对照组 |

`{m}` 取 `gate` / `up` / `down`。每层 36 个键 = `update_zmax` 3 个 + 三块矩阵各 11 个（`rel_update` 6 + `delta_norm_mean` 1 + `stable_rank` 3 + `shared_{m}_rel_update` 1）；开 `log_spectrum` 后每块再加 3 个熵，共 45 个。

> 逐层键落在 `mlp_update/layer_{i}/…`，全局聚合落在 `mlp_update/global_…`。dense 层（无 MoE）不声明指标，所以 18 层 MoE + 1 组 global = 684 个键。

**专家级汇总分数**用 `max_m z(r_m)` 而不是平均：平均会让某一块矩阵的异常被另两块健康的摊薄，
而平均原始范数会让尺度最大的那块矩阵决定结论。先在**同层专家间**做标准化把三块拉到同一尺度，
再对 m 取 max，任意一个投影出问题都不会被吃掉。`_max` 是该层最异常的专家；`_min` 是在三块投影上
**同时**都低于同侪的专家，是这里能给出的最锐利的沉睡专家读数。

> `update_zmax` 有上界 `(E−1)/√E`（每 rank 32 个本地专家时是 5.48），会饱和——它是异常探测器，不是幅值计。
> 另外若某块矩阵在专家间完全没有差异（如整块零更新），其 z 恒为 0，而 S 取 max 会被 0 托住，
> 此时 `update_zmax_min` 失效；真实训练中三块都有离散度，不会触发，但看到它长期恰好为 0 要想到这个退化情况。

**`σ₁/‖ΔW‖_F` 没有单独记**：它恒等于 `1/√stable_rank`，是纯派生量。

配置参数：

- `sample_interval` (默认 `None`)：采样间隔 δ，`None` 时用全局 `monitor_interval`。
- `log_spectrum` (默认 `False`)：是否算奇异值熵。这是唯一需要完整特征分解的指标。
- `spectrum_interval` (默认 `1`)：谱走的更粗的时钟——相对更新量每个采样点都算，谱每
  `spectrum_interval` 个采样点算一次。

成本（实测，4B-A500M 形状，每 EP rank 32 专家，`weight1 [32,512,1024]` + `weight2 [32,512,512]`，单卡 H800）：

```
范数 + 幂迭代 σ₁          11.3 ms/层  →  0.20 s / 18 层     ← 默认档
完整 Gram + eigvalsh      500.9 ms/层 →  9.02 s / 18 层     ← log_spectrum=True
基准快照显存              48 MB/层    →  0.84 GB / 18 层    （fp32 会翻倍到 1.69 GB）
```

`σ₁` 走幂迭代而非特征分解，因为 stable rank 只需要 `σ₁`。收敛速度是 `(σ₂/σ₁)^(2k)`，30 次迭代下
`σ₂/σ₁ = 0.25` 的秩主导更新精确到 5e-8，`σ₂/σ₁ = 0.99` 的平谱更新 `σ₁` 差约 1.2%、stable rank 差约 2.4%。
不准的恰好是 stable rank 很大、精确值不改变判读的那一档；准的恰好是要检测的坍缩。误差单向偏低，
因此 stable rank 偏高。

两个必须知道的局限：

- 基准快照按参数原 dtype 克隆，bf16 权重下 ΔW 相对「可读到的东西」是精确的，但被量化在 bf16 网格上：
  **小于权重量级约 2⁻⁹ 的更新看不见，读出来就是 `r = 0`**。所以「沉睡专家」的准确含义是
  「在 δ 步内、在可表示权重上没动」，δ 越大越可靠。要突破需读优化器的 fp32 master weight。
- `_mean` / `_min` / `_max` 跨 EP rank 归约精确（每 rank 专家数整除），但 **`_median` / `_p10` / `_p90`
  以及 `update_zmax_*` 落到日志里的是各 rank 的值再平均，不是全局分位数**；标准化也是相对本 rank 的分片。

> **ΔW 小不等于功能没变。** 专家收到的 token 数不同，更新小可能只是路由给它的 token 少。
> 请对着 `moe_health` 已有的路由负载分布一起读：`assignment_load_cv`、`assignment_load_min_frac`、
> `assignment_load_max_min_ratio`、`gate_mass_*`，本 monitor 不重复实现这些。
> 真正的功能漂移度量（固定 probe batch 上的专家输出位移 `D_e`）需要把旧权重留在显存里再跑一次
> expert forward，超出纯权重探针的范围，未实现。

> 时序约束：快照必须在 **step begin** 读。开启 `offline_quant_expert_weight` 时
> `FP8QuantWeightCallback.on_step_begin` 会清掉 bf16 专家权重，所以本 monitor 的采集入口是
> `collect_expert_norms()`——那是 `InternalMedicineCallback.on_step_begin` 的契约方法名（不是描述），
> 且该 callback 注册在 FP8 callback 之前。放到 step end 会读到已清空的存储。

---

## 九、DSA Health Monitor (dsa_health)

DeepSeek Sparse Attention 层的监控。DSA 以 `core_attention` 的形式插进 MLA，选择规则是
`index_scores = Σ_h w_h · relu(q_h · k)` 取 top-k，因此它有两个独立的失效面：**indexer 本身是否健康**，
以及**它选出来的 key 是否就是主干 attention 真正要用的 key**。

发现方式是结构判别，不是类名也不是「有 indexer 属性」。`CSAIndexer` 带**完全相同**的
`forward_before_topk` / `index_topk` 一对，返回的还是同样 arity 的 `(q, k, weights)`——区别在
**key 轴的含义**：CSA 打分的是压缩位置（`n_compressed ≈ sq/ratio`），DSA 打分的是原始 token，而
shape 本身不说明是哪一种。所以 CSA 的标记要显式排掉：indexer 上的 `compressor` / `compress_ratio`，
以及 core 上的 `compressed_sparse_attn`（`qk_monitor` 的稀疏路径就是靠这个属性认的）。认错的后果不是
报错而是**静默错**：因果判定和距离全部会算在错误的序列上。

符号约定（都在**采样的 query 行 `r` × 采样 head `h`** 上算，见「采样与代价」）：
`s_t = Σ_h w_h · relu(q_h · k_t)` 是 indexer 打分；`causal_t = [t ≤ r]`、`n_causal = Σ_t causal_t` 是该行的因果长度；
`sel_t ∈ {0,1}` 是 top-k 选中标记（**已乘 causal**，所以有效 key 不足 k 的行不会虚计），`n_sel = Σ_t sel_t`；
`dist_t = (r − t) · causal_t`；`p_t = softmax_t(scale · q_h·k_t + causal mask)` 是主干稠密注意力概率。

| # | 键 | 族 | 公式 | 级别 | 诊断意义 |
|---|-----|-----|------|------|----------|
| 1 | `index_q_rms` | `indexer` | `sqrt(mean(q²))` | 每层+全局 | 打分 einsum 真正吃进去的 q（已过 RoPE + Hadamard）的量级 |
| 2 | `index_q_abs_max` | `indexer` | `max(\|q\|)` | 每层+全局 | q 的 outlier；配合 rms 区分整体放大与局部尖峰 |
| 3 | `index_k_rms` | `indexer` | `sqrt(mean(k²))` | 每层+全局 | indexer key 的量级 |
| 4 | `index_k_abs_max` | `indexer` | `max(\|k\|)` | 每层+全局 | indexer key 的 outlier |
| 5 | `index_weights_abs_max` | `indexer` | `max(\|w\|)` | 每层+全局 | 逐头重要性权重的量级（已吸收 `softmax_scale` 与 `n_heads^-0.5`） |
| 6 | `index_weights_neg_ratio` | `indexer` | `mean(w < 0)` | 每层+全局 | 负权重占比。`weights_proj` 无约束，负权重让那个头的证据**压制**某位置；趋近 0.5 = 逐头重要性不再是重要性 |
| 7 | `score_selected_mean` | `score` | `mean_r(Σ_t s·sel / n_sel)` | 每层+全局 | 留下的分数（不 clip——负向塌缩要看得见） |
| 8 | `score_unselected_mean` | `score` | `mean_r(Σ_t s·(causal−sel) / (n_causal−n_sel))` | 每层+全局 | 丢掉的分数。与上一行的差就是选择边界的清晰度 |
| 9 | `score_topk_mass` | `score` | `mean_r(Σ_t relu(s)·sel / Σ_t relu(s)·causal)` | 每层+全局 | 选中位置占总正分数的比例（先 clip 到 ≥0，否则符号和不构成「质量」） |
| 10 | `select_ratio` | `select` | `mean_r(n_sel / n_causal)` | 每层+全局 | 实际稀疏率 |
| 11 | `select_dist_mean` | `select` | `mean_r(Σ_t sel·dist / n_sel / n_causal)` | 每层+全局 | 平均回看距离，按该行因果长度归一化，跨序列长度可比 |
| 12 | `select_local_ratio` | `select` | `mean_r(Σ_t sel·[dist < W] / n_sel)`，`W = local_window` | 每层+全局 | 选中位置落在最近窗口内的比例。**退化告警**：indexer 塌成滑动窗口时趋 1 |
| 13 | `select_key_coverage` | `select` | `\|∪_r {t : sel_t = 1}\| / max_r n_causal` | 每层+全局 | 采样行的并集覆盖了多少 key 轴 |
| 14 | `attn_recall` | `match` | `mean_{h,r}(Σ_t p·sel)` | 每层+全局 | **首要指标**：稀疏 mask 之后存活的稠密注意力质量。回答「选择和主干模式匹配吗」 |
| 15 | `attn_recall_min` | `match` | `min_{h,r}(Σ_t p·sel)` | 每层+全局 | 最差的那个 (head, row)，均值掩盖的尾部风险 |
| 16 | `attn_kl` | `match` | `mean_{h,r}(−log recall)` | 每层+全局 | `KL(p_sparse ‖ p_dense)`。逐 (head, row) 恒等于 `−log(recall)`，但**报告值不冗余**：`mean(−log r) ≥ −log(mean r)`，两者的差正是「个别头被害得很惨」的代价 |
| 17 | `attn_top1_hit` | `match` | `mean_{h,r}(sel[argmax_t logit])` | 每层+全局 | 最大 logit 是否被选中。recall 能停在 0.95 而最重要的那个 key 一直被丢 |
| 18 | `attn_sparse_entropy` | `match` | `mean_{h,r}(−Σ_t p̃ log p̃)`，`p̃ = p·sel / recall` | 每层+全局 | mask + 重归一之后的注意力熵，看选择把分布削得多尖 |
| 19 | `select_iou_prev` | `crosslayer` | `mean_r(\|sel ∩ sel_prev\| / \|sel ∪ sel_prev\|)` | 每层+全局 | 与**同一次 forward 内前一个** DSA 层选择的 IoU。趋 1 = 各层在重复算同一个选择（跨层共享 index 的收益上界）；趋近稀疏率 = 各层独立选择 |

> 稠密侧的注意力熵**故意不在这里**：`qk_stats` 已经在同一批 DSA 层上报同一个量，重复一遍只会让两块看板对不上。
> `index_weights_mean` 和 `select_dist_max` 也去掉了——前者是近对称有符号向量的均值（实测 `1.95e-4`，信息已由
> `neg_ratio` + `abs_max` 覆盖），后者在实测里稳定饱和在 `~1.0`。

Key 布局同其他 monitor：`dsa_health/layer_{i}/{attn_type}_{name}`，global 由逐层累加器在 flush 时派生。

### 采样与代价

indexer 打分是 `O(sq·sk·h·d)`，主干稠密 attention 是 `O(sq·sk·H·d)`——全长复算等于把这层再跑一遍。
所以两者都只在**采样的 query 行 × 采样的 head** 上算：

- `row_samples` (默认 32)：行从 `index_topk` 之后取。因果有效 key 少于 `index_topk` 的行会把所有位置都选上，
  recall 恒为 1、稀疏率恒为 1，混进平均里会把信号冲掉。
- `head_samples` (默认 8)：只用于 `match` 一族。grouped KV 下 key 的头是**按下标挑**的，不做
  `repeat_interleave`——生产序列长度下那会先物化几百 MB 再全丢掉。
- `local_window` (默认 128)：`select_local_ratio` 的判定窗口。

做逐族消融时先关 `match`：它是唯一需要重算稠密 attention 的一族。

### 采集点

一个实例 patch + 一个 forward post hook，每层。patch 打在 `DSAIndexer.forward_before_topk` 上，因为
那是 indexer 的 `q` / `k` / `weights` 唯一存在的地方——训练路径上 `DSAttention.forward` 把它们直接喂给
fused loss kernel，`DSAIndexer.forward` 从不执行，挂在 indexer 上的 forward hook 永远不会触发。
top-k 选择也是照 `compute_index_scores` 的公式**重算**而不是从 fused kernel 里掏 `_last_topk_indices`，
这样指标在「有没有开 indexer loss」两种配置下语义一致。

同一个 patch 还顺手截下调用方传的 `position_offset`（`MQALatentAttention._indexer_projections` 是位置
传参）。这是把本 rank 的局部行号换算成全局位置的唯一依据，从调用参数里拿而不是自己按 CP rank 反推，
可以不关心 `cp_balance_mode` 是哪种切法。

### 上下文并行（CP）

CP 下 query 行是本 rank 的序列切片，而 indexer 的 K 已经 all-gather 到**全局**长度
（`dsa_attention.py:559-565`：在 `head_dim` 处 gather，比在 `hidden_size` 处少 32 倍流量）。
所以行采样用局部行号做 `index_select`，因果与距离判定用 `局部行号 + position_offset` 的全局位置 ——
拿局部行号去比全局 key 位置会静默地把大部分真实历史 mask 掉。

`sample_query_rows` 的下界也按全局位置切：切片起点已经超过 `index_topk` 的 rank 用整个切片，
只有 rank 0 需要跳过开头那些历史不足的行。

**`match` 一族在 CP>1 下关闭**并 warn。它要和主干注意力对照，而主干的 key 只有本 rank 的切片，
要复现全局 logits 就得把主干 key 也 all-gather 一次，代价不在这个 monitor 该付的范围内。
`indexer` / `score` / `select` / `crosslayer` 四族只依赖 indexer 的输出加全局行号，CP 下照常工作。

### 两个 phase 怎么读

`hybrid_mla_attention=mqa_dsa` 有两个训练阶段（`mqa_latent_attention.py:_phase()`），
由 `dsa_indexer_use_sparse_loss` 选择，指标的**读法**随之不同：

- **`warmup`**（`use_sparse_loss=false`，常配 `train_indexer_only`）：层两边都不用 top-k，
  attention 跑全因果，KL 覆盖每一个因果列。此时 `select_*` / `match_*` 描述的是一个**假设**的选择 ——
  「如果现在就切到稀疏，会损失多少」。这恰好是决定何时进入下一阶段的 go/no-go 信号，但**不要**
  读成「当前稀疏化的损失」。这一阶段真正反映训练健康的是 `indexer` 与 `score` 两族。
- **`sparse`**（`use_sparse_loss=true`）：层实际消费 `强制局部窗口 ∪ top-k`。

> 已知缺口（`sparse` 阶段）：本实现只复现 top-k 那一半，没有并入 `csa_window_size` 的强制窗口，
> 也没有把 `add_full_attention_sink_bias` 的 attention sink 计入 `match` 的 softmax。因此该阶段的
> `select_ratio` / `attn_recall` 会**系统性偏低**，偏差量就是窗口与 sink 贡献的那部分。

> 已知缺口（发现范围）：只在 `self_attn.core_attention` 上找 DSA。`DSv4HybridAttention` 的 `full_attn`
> 分段里也可以挂一个 indexer（见 `multi_latent_attention.py:985-991` 的注释），那个位置目前不会被
> 发现 —— 宁可漏报也不猜路径。

---

## 基础设施

### TrainingLogs

全局单例的指标存储，所有 Monitor 将指标写入此处。

```python
from paddlefleet.internal_medicine import training_logs
```

**SmoothedValue 聚合模式** — 由指标键名自动推断：

| 键名模式 | 推断模式 | 输出值 |
|----------|----------|--------|
| 包含 `/max` 或以 `_max` 结尾 | `max` | 历史最大值 |
| 以 `topk_channel_norm`、`channel_max_ratio`、`channel_median`、`channel_p95`、`channel_p99`、`activation_rms` 结尾 | `max` | 历史最大值 |
| `massive_act_channel_count` 或 `channel_count_gt_*` | `max` | 历史最大值 |
| 包含 `/min` 或以 `_min` 结尾 | `min` | 历史最小值 |
| 其他 | `mean` | 累积均值 |

### 跨卡聚合

`training_logs.gather_and_aggregate()` 收集所有 rank 的指标，然后按键名规则聚合：

| 键名模式 | 聚合方式 |
|----------|----------|
| 包含 `_max` 或以 `/max` 结尾 | `np.max(all_ranks)` |
| `channel_max_ratio`、`channel_median`、`channel_p95`、`channel_p99`、`topk_channel_norm`、`activation_rms` | `np.max(all_ranks)` |
| `massive_act_channel_count` 或 `channel_count_gt_*` | `np.max(all_ranks)` |
| 包含 `_min` 或以 `/min` 结尾 | `np.min(all_ranks)` |
| 其他 | `np.mean(all_ranks)` |

PaddleFleet 后端默认走**数值 all_reduce**：首次采样只 `all_gather_object` 各 rank 的 key 名
以对齐 layout，之后每次采样只交换一个 float64 向量（3 次 all_reduce：SUM 值与 count、MAX、MIN），
载荷不随 world size 增长。`all_gather_object` 仅保留为内部回退路径，不通过环境变量切换。

**`monitor_interval > 1` 时不要每步都调用**：`gather_and_aggregate()` 是全 world 的
集合通信，只有采样步才可能产出指标，其余步跑的是一次空的 `all_gather_object`
（6144 卡实测 872 ms/步）。调用方应当先问 `monitor.sampled_this_step`——它由
`step()` 在自增前锁存，是"刚结束这一步的 hook 有没有采样"的唯一权威答案：

```python
if any(m.sampled_this_step for m in monitor_dict.values()):
    aggregated = training_logs.gather_and_aggregate()
```

**不要**用调用方自己的步数取模去推这个相位（`global_step % interval == 1`）。
`step_count` 是进程内变量、续训从 0 重开，而 `global_step` 从 checkpoint 恢复；
两者相位一旦错开（resume 步不是 `monitor_interval` 整数倍时必然错开），门控会
静默挡掉全部采样步。同时注意：这个判断决定要不要进集合通信，**条件必须全 rank
一致**，`sampled_this_step` 只依赖各 rank 同步推进的 `step_count`，本地指标是否
为空则不满足这个要求（历史上据此早返回导致过 PP 死锁）。

从 checkpoint 续训时，如果要避免新进程 `step_count == 0` 导致恢复后的第一步
必然采样，应在首个 forward 前对所有 monitor 调用 `skip_next_steps(1)`。该门控
在 hook 做指标计算之前生效；被跳过的 step 仍推进本地 `step_count`，但不会 flush
指标 buffer，`sampled_this_step` 也保持为 `False`，调用方因而可同时跳过跨卡聚合
与日志输出。是否属于续训应由调用方使用已恢复的 Trainer state 判断，不应依赖
历史指标文件是否存在。

注意: QK Stats Monitor 不在 hook 内做 TP `all_reduce`/`all_gather`，避免在 attention 热路径插入通信；跨 rank 聚合统一交给 `gather_and_aggregate()`。MassiveAct 会先对 TP 内 per-channel maxima 做 `MAX all_reduce`（通道维被 TP 切分时这是正确性所需），再依赖 `gather_and_aggregate()` 做跨 rank 聚合；MoE/PLE 同样依赖 `gather_and_aggregate()` 统一处理。

### 通用配置参数

所有 Monitor 共享以下配置参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `log_per_layer` | `True` | 记录每层指标 |
| `log_global` | `True` | 记录全局聚合指标 |
| `monitor_interval` | `1` | 监控间隔 (每 N 步采集一次) |
| `verbose` | `False` | 打印调试信息 |
| `hook_timing_enabled` | `False` | 记录每个 monitor hook 的 CPU wall time；用于定位监控开销，默认关闭 |

NeMo Trainer 对应字段为 `internal_medicine_hook_timing`。开启后 trainer 会按日志间隔输出各 hook 的累计耗时和平均耗时；该开关只做 Python 侧计时，不会额外插入 CUDA 同步。

### 指标族粒度控制 (exclude_families)

每个 monitor 内部的指标按**族 (family)** 分组——和在线可视化 `tools/internal_medicine` 的分组同源（`core/metric_families.py` 的 `METRIC_TAXONOMY`）。可以按「模块:族」关掉不关心的族，被关掉的族**不进 schema、不参与跨 rank 聚合**，因此既省通信也省显存。

| Monitor | 族 |
|---------|-----|
| `mhc_health` | `gate` 门控强度（逐层聚合）/ `stream` 按流门控值 `h_{pre,post}_idx{i}` / `mix` 混合矩阵（逐层聚合）/ `cell` 混合矩阵元素 `h_res_cell{i}` / `gain` 增益放大 / `param` 可学习参数 / `share` 残差占比 |
| `qk_stats` | `sink` sink head / `logit` logits 量级 / `entropy` 注意力熵 / `qkv` q/k/v 范数 |
| `massive_act` | `channel` 通道量级分布 / `outlier` 超大通道计数 / `module` 各模块输出量级 / `norm` 归一化后表示 / `overall` 整体激活量级 |
| `ape_health` | `softmax` 位置 softmax 形状 / `coverage` 位置覆盖 / `scale` 量级与数值健康 |
| `vha_health` | `mix` postmix 混合矩阵 / `gain` postmix 增益与扰动 / `head` 头间一致性 |
| `moe_health` | `router` 路由打分质量 / `balance` 负载均衡 / `norm` 专家范数与 bias / `spectrum` gate 谱性质 / `act` 激活量级 / `shared` 共享 vs 路由 / `expert` 按专家分布 / `card` 按卡负载（EP 组内） |
| `dsa_health` | `indexer` indexer 投影量级 / `score` 打分分布与选择边界 / `select` 选择结构与稀疏率 / `match` 与稠密 attn 的一致性（最贵的一族，做消融先关它）/ `crosslayer` 跨层选择重合度 |

语义是**排除优先**：没声明的族默认全开，所以不配置这个参数时行为与以前逐字节一致；分类表若有遗漏，那条指标不属于任何排除集合，仍会被采集（fail-open）。反过来，族名写错会在启动时直接抛 `UnknownFamilyError`（fail-closed）——否则唯一的症状是「payload 没变小」，太难发现。

```python
setup_monitors(
    model,
    monitors=["moe_health", "mhc_health"],
    monitor_interval=50,
    exclude_families="mhc_health:mix, moe_health:expert+act",
)
```

单个 monitor 也可以直接传 `exclude_families=["mix"]`。只提供排除式、不提供白名单：白名单会把分类表的遗漏变成静默漏采（新族不在名单里 → 不采集，且无人察觉），而排除式下同样的遗漏只是「payload 没省下来」。想「只看某几族」就填它的补集。

### 线上默认与 debug 模式 (debug_mode)

训练侧不暴露族名字符串，只有一个布尔开关：**默认（`debug_mode=False`）排除按结构单元膨胀的族**，`debug_mode=True` 才全采。这个集合是代码里的常量 `DEBUG_ONLY_FAMILIES`：

| Monitor | 默认排除的族 | 内容 | 膨胀方式 |
|---------|------------|------|---------|
| `moe_health` | `expert` | `expert_{token,weight}_share_e{i}` | 每专家一条 |
| `mhc_health` | `cell` | `{attn,mlp}_h_res_cell{i}` | 每层每个混合矩阵元素一条 |
| `mhc_health` | `stream` | `{attn,mlp}_h_{pre,post}_idx{i}` | 每层每个输入流一条 |

在 43 层 mHC + MoE 的配置上，这三族占每步 3732 个 key 里的 1392 个（37.3%）——同比例地占跨 rank 聚合的 payload 和落盘的 jsonl。其余族的 key 数只随层数变化。

关键是这三族**恰好是诊断层不读的部分**：诊断看的是逐层聚合量（`h_res_logits_max`、`h_post_mean`、`h_post_stream_concentration` 等），它们留在 `mix` / `gate` 里照常采集。所以线上默认只省 payload，不损失任何诊断深度——这也正是把 `mix` / `gate` 各拆成「逐层聚合」和「按单元展开」两族的原因。

```yaml
internal_medicine_monitors: "qk_stats,moe_health,massive_act,mhc_health,vha_health"
internal_medicine_monitor_interval: 200
internal_medicine_debug_mode: false   # true = 连按专家/按 cell/按流的族一起采
```

```python
from paddlefleet.internal_medicine.core.metric_families import exclusions_for

setup_monitors(model, monitors=[...], exclude_families=exclusions_for(debug_mode=False))
```

---

## 附录: 完整指标速查表

指标按后端和模型能力动态输出；以下列出各监控器的核心指标。

| Monitor | 指标 | 公式 | SmoothedValue 模式 | 健康信号 |
|---------|------|------|--------------------|----------|
| **MoE** | `router_entropy` | `-Σ(p log p)` | mean | 高 = 均匀路由 |
| **MoE** | `score_sum_mean` | `mean(topk_sum)` | mean | 适中 |
| **MoE** | `score_sum_min` | `min(topk_sum)` | min | 不应过低 |
| **MoE** | `score_sum_max` | `max(topk_sum)` | max | 不应过高 |
| **MoE** | `expert_bias_mean` | `bias.mean()` | mean | 接近零 |
| **MoE** | `expert_bias_std` | `bias.std()` | mean | 适度 |
| **MoE** | `bias_affinity_jaccard` | `\|A∩B\|/\|A∪B\|` | mean | > 0.7 OK |
| **MoE** | `expert_norm_mean` | `mean(L2)` | mean | 稳定 |
| **MoE** | `expert_norm_std` | `std(L2)` | mean | 不应过大 |
| **MoE** | `expert_norm_min` | `min(L2)` | min | 不应萎缩 |
| **MoE** | `expert_norm_max` | `max(L2)` | max | 不应过载 |
| **MoE** | `shared_expert_norm` | `\|\|shared\|\|₂` | mean | 稳定 |
| **MoE** | `shared_routed_ratio` | `shared/routed` | mean | 0.3 ~ 3.0 OK |
| **MoE** | `load_max_min_ratio` | `max/min(tokens)` | mean | 越接近 1 越均衡 (global_aux_loss 或 expert_bias) |
| **MoE** | `load_max_median_ratio` | `max/median(tokens)` | mean | 越接近 1 越均衡 (global_aux_loss 或 expert_bias) |
| **MoE** | `load_cv` | `std/mean(tokens)` | mean | 越接近 0 越均衡 (global_aux_loss 或 expert_bias) |
| **MoE** | `load_balance_entropy_norm` | `H(p)/log(E)` | mean | 越接近 1 越均衡 (global_aux_loss 或 expert_bias) |
| **MoE** | `load_effective_experts` | `exp(H)` | mean | 越接近专家数 E 越均衡 (global_aux_loss 或 expert_bias) |
| **MoE** | `expert_gate_stable_rank_mean` | `mean_e(srank(W_g))` | mean | 稳定，不应持续下滑 |
| **MoE** | `expert_gate_stable_rank_min` | `min_e(srank(W_g))` | min | 不应萎缩 (秩塌缩预警) |
| **MoE** | `expert_gate_stable_rank_max` | `max_e(srank(W_g))` | max | 上升 = 专家分化 |
| **MoE** | `expert_gate_singular_entropy_mean` | `mean_e(H(σ²))` | mean | 初始约 `0.92·log k`，不应持续下滑 |
| **MoE** | `expert_gate_singular_entropy_min` | `min_e(H(σ²))` | min | 不应显著低于均值 |
| **MoE** | `expert_gate_singular_entropy_max` | `max_e(H(σ²))` | max | 上限 `log k`，平谱时取等 |
| **MoE** | `shared_gate_stable_rank` | `srank(W_g)` | mean | 稳定 |
| **MoE** | `shared_gate_singular_entropy` | `H(σ²)` | mean | 初始约 `0.96·log k`，不应持续下滑 |
| **MoE** | `expert_token_share_eK` | `count_K/Σcount × 100%` | mean | 每层每专家；均衡时 = 100/E，长期高企 = 热点，贴 0 = dead expert |
| **MoE** | `expert_weight_share_eK` | `Σprobs_K/ΣΣprobs × 100%` | mean | 每层每专家；与 token 占比偏离越大 = 该专家单 token 权重越重 |
| **QK** | `max` | `max(QK^T/√d)` | max | 不应暴增 |
| **QK** | `mean` | `mean(logits)` | mean | 稳定 |
| **QK** | `entropy_avg` | `-Σ(p log p)` avg | mean | 适中 |
| **QK** | `sink` | `p(token_0)` avg | mean | 不应过高 |
| **QK** | `entropy_min` | `min(head_H)` | min | 不应过低 |
| **QK** | `entropy_max` | `max(head_H)` | max | 合理范围 |
| **QK** | `sink_head_ratio` | `count(sink>threshold)/N_heads` | mean | Sink head 占比 |
| **QK** | `sink_head_max` | `max(sink_per_head)` | max | 最强 sink head |
| **QK** | `sink_nonsink_gap` | `mean(sink) - mean(nonsink)` | mean | Sink vs 非Sink gap |
| **QK** | `attn_sink_logit` | `mean(sink_logit)` | mean | learned sink 量级（稀疏层 `attn_sink` / full 层 `softmax_offset`） |
| **MassiveAct** | `channel_max` | `max(abs(H_i))` | max | 通道峰值激活 |
| **MassiveAct** | `channel_median` | `median(per_channel_max)` | max | 通道峰值基准量级 |
| **MassiveAct** | `channel_p95` | `p95(per_channel_max)` | max | 高分位通道幅度 |
| **MassiveAct** | `channel_p99` | `p99(per_channel_max)` | max | 极高分位通道幅度 |
| **MassiveAct** | `channel_max_ratio` | `max / median` | max | 异常值严重度 |
| **MassiveAct** | `massive_act_channel_count` | `count(ch > 100*median)` | max | median-relative 异常通道数 |
| **MassiveAct** | `channel_count_gt_10` | `count(ch > 10)` | max | 广泛高于基准量级 |
| **MassiveAct** | `channel_count_gt_20` | `count(ch > 20)` | max | 高幅度通道数 |
| **MassiveAct** | `channel_count_gt_30` | `count(ch > 30)` | max | 接近当前 FP4 峰值区间 |
| **MassiveAct** | `topk_channel_norm` | `norm(topk(3))` | max | Top-K 通道范数 |
| **MassiveAct** | `activation_rms` | `sqrt(mean(H_i^2))` | max | residual stream 整体 scale |
| **MassiveAct** | `post_norm_sparsity` | `mean(abs(x) < eps)` | mean | 归一化后稀疏度 |
| **MassiveAct** | `post_norm_cosine` | `cos_sim(tokens)` | mean | 近常量向量检测 |
| **MassiveAct** | `spectral_norm_max` | `max(post_rms / pre_rms)` | max | 谱范数(σ_max)下界 |
| **MassiveAct** | `spectral_norm_min` | `min(post_rms / pre_rms)` | min | 最小奇异值(σ_min)上界 |
| **MassiveAct** | `lipschitz_max` | `max(‖∂L/∂x‖ / ‖∂L/∂y‖)` | max | σ_max(J)下界 = Lipschitz 常数 |
| **MassiveAct** | `lipschitz_min` | `min(‖∂L/∂x‖ / ‖∂L/∂y‖)` | min | σ_min(J)上界 |
| **MassiveAct** | `logit_lens_entropy_mean` | `mean_t H(softmax(final_norm(h)·Wᵀ))` | mean | 预测熵均值 (可选, 默认关) |
| **MassiveAct** | `logit_lens_logsumexp_mean` | `mean_t logsumexp(final_norm(h)·Wᵀ)` | mean | 对数配分均值 (可选, 默认关) |
| **MassiveAct** | `logit_lens_cross_entropy_mean` | `mean_t(log_z − l[y])` | mean | 逐层交叉熵，末层≈LM loss (可选, 默认关) |
| **MassiveAct** | `hidden_spectral_entropy` | `-Σ p_i log p_i, p_i=σ_i²/‖h‖_F²` | mean | post-norm 谱熵/有效秩 (可选, 默认关) |
| **PLE** | `token_ple_norm` | `mean(\|\|token_ple\|\|₂)` | mean | 量级稳定 |
| **PLE** | `proj_ple_norm` | `mean(\|\|proj × H^{-0.5}\|\|₂)` | mean | 与 token 分支匹配 |
| **PLE** | `per_layer_inputs_norm` | `mean(\|\|(t+p)×2^{-0.5}\|\|₂)` | mean | 量级稳定 |
| **PLE** | `token_proj_cosine` | `mean(cos_sim)` | mean | 显著 < 1.0 |
| **PLE** | `residual_ratio` | `\|\|Δ\|\|/\|\|h\|\|` | mean | 适中 |
| **PLE** | `gate_activation_mean` | `mean(\|act(gate)\|)` | mean | 非零 |
| **PLE** | `gate_sparsity` | `dead_ratio` | mean | 不应持续上升 |
| **VHA** | `{b}_postmix_delta_rel_mean` | `mean_t(‖delta‖/‖mixed‖)` | mean | postmix 是否在学 (0 起步) |
| **VHA** | `{b}_postmix_delta_rel_max` | `max_t(‖delta‖/‖mixed‖)` | max | 最坏 token 修正幅度 |
| **VHA** | `{b}_postmix_amax_gain_max` | `max_t(‖out‖_∞/‖mixed‖_∞)` | max | 低精度溢出余量 |
| **VHA** | `{b}_postmix_uv_sigma_max` | `σ_max(A Bᵗ)` | max | 偏离恒等的强度上界 |
| **VHA** | `{b}_postmix_uv_eff_rank` | `exp(-Σ p log p)` | mean | 是否守在低秩预算内 (涨=警告) |
| **VHA** | `{b}_postmix_offdiag_ratio` | `‖A−diag(A)‖_F/‖I+A‖_F` | mean | 趋 0 = 退化成 per-head 缩放 |
| **VHA** | `{b}_postmix_u_fro` / `{b}_postmix_v_fro` | `‖A‖_F` / `‖B‖_F` | mean | 哪个因子在长 |
| **VHA** | `{b}_head_out_norm_max` | per-head 输出范数最大 | max | head 被放大 |
| **VHA** | `{b}_head_out_norm_min` | per-head 输出范数最小 | min | head 被压成 0 |
| **VHA** | `{b}_head_out_norm_std` | per-head 输出范数离散度 | mean | head 间幅度分化 |
| **VHA** | `{b}_postmix_head_cos_mean` | 均值 head 向量间平均 \|cos\| | mean | 趋 1 = head 退化成复制 |
| **VHA** | `premix_identity_dev` | 组均值 `‖W_g − I‖_F` | mean | Premix 离无操作多远 (需 use_vha_premix) |
| **VHA** | `premix_group_div_ratio` | `mean‖W_g−W_g'‖²/mean‖W_g−I‖²` | mean | 0=各组同一变换, 2=完全独立 |
| **VHA** | `premix_sigma_max` | `σ_max(W)` | max | premix 谱范数 (需 use_vha_premix) |
| **VHA** | `premix_orth_dev` | `‖WᵀW − I‖_F` | mean | 仅非方阵 premix (正交初始化) |
| **AttnUpdate** | `delta2_norm` | `‖Δ₂‖_F` | mean | 一阶 QK 增量幅度 |
| **AttnUpdate** | `delta2_stable_rank` | `‖Δ₂‖_F²/‖Δ₂‖₂²` | mean | 偏离健康基线 = 谱坍缩前兆 |
| **AttnUpdate** | `delta2_singular_spectrum` | `S₂ = exp(-Σ p log p), p=σ²/Σσ²` | mean | 同上，全谱、更灵敏 |
| **AttnUpdate** | `delta3_norm` | `‖Δ₃‖_F` | mean | 二阶耦合项幅度，应远小于 `delta2_norm` |
| **AttnUpdate** | `delta3_stable_rank` | `‖Δ₃‖_F²/‖Δ₃‖₂²` | mean | Q/K 更新耦合的 top-1 主导程度 |
| **AttnUpdate** | `delta3_singular_spectrum` | 同上，作用在 Δ₃ 的谱上 | mean | Q/K 更新耦合变 coherent 时下降 |
| **MLPUpdate** | `{m}_rel_update_mean` | `mean_e(‖ΔW_m‖_F/‖W_m‖_F)` | mean | 该投影整体更新水平，`m`=gate/up/down |
| **MLPUpdate** | `{m}_rel_update_max` | `max_e(...)` | max | 更新突然异常增大的专家 |
| **MLPUpdate** | `{m}_rel_update_min` | `min_e(...)` | min | 沉睡专家（注意 bf16 网格下限） |
| **MLPUpdate** | `{m}_rel_update_p10` / `_p90` | `quantile_e(...)` | mean | 专家间离散度；跨 rank 是近似 |
| **MLPUpdate** | `{m}_stable_rank_mean` | `‖ΔW_m‖_F²/‖ΔW_m‖₂²` | mean | 更新是否集中到少数方向 |
| **MLPUpdate** | `update_zmax_max` | `max_e max_m z(r_m)` | max | 层内最异常专家，上界 `(E−1)/√E` |
| **MLPUpdate** | `shared_{m}_rel_update` | `r_m` of shared expert | mean | 路由专家的对照组 |
| **Optim** | `update_rms` | `sqrt(mean((θ_new−θ_old)²))` | 已全局归约 | 本 step 参数更新幅度 (始终装上, 受 monitor_interval 门控) |
| **Optim** | `param_rms` | `sqrt(mean(θ_new²))` | 已全局归约 | 更新后参数尺度 (始终装上, 受 monitor_interval 门控) |
| **Optim** | `update_param_ratio` | `update_rms / param_rms` | 已全局归约 | trust ratio, ~1e-3 健康 (始终装上, 受 monitor_interval 门控) |
