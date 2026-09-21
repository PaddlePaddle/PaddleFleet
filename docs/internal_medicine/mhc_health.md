# mHC Health Monitor

监控 mHC (Manifold-Constrained Hyper-Connections) 层的健康状况。mHC 用如下传播替换普通残差：

```
x_{l+1} = H_res^T @ x_l + H_post^T · F(H_pre @ x_l)
```

每个 token、每个 hyper-connection 模块学习三个映射（`n = num_residual_streams`）：

- `h_pre`  `[s, b, n]`     — 多流聚合门（sigmoid）
- `h_post` `[s, b, n]`     — 多流扩展门（2·sigmoid）
- `h_res`  `[s, b, n, n]`  — Sinkhorn 双随机（doubly-stochastic）残差混合矩阵

每个 `HyperConnectionTransformerLayer` 含两个 hyper-connection 模块：`self_attention_hyper_connection`
（`attn`）与 `mlp_hyper_connection`（`mlp`），forward 中 attn 先于 mlp 执行。

---

## 开关与 no-op 保证

通过在 `internal_medicine_monitors` 中加入 `mhc_health`（或 `all`）启用。启用它在**任何**模型上都是安全的——
以下任一情况下它彻底 no-op（不 wrap 任何模块、不声明/产生任何指标、不抛异常）：

1. **mHC 类无法 import**：`setup_mhc_monitor` 在 import 失败时将类名绑定为 `None`，直接返回 `model`。
2. **模型未使用 mHC 层**：discovery 用 `isinstance(layer, HyperConnectionTransformerLayer)` 与
   `isinstance(mod, HyperConnectionModule)` 精确匹配（不做 duck-typing），普通 `TransformerLayer` 或
   `IdentityOp` 占位符都不会被匹配。

```yaml
internal_medicine_monitors:
    mhc_health: true
```

---

## 采集方式：wrap `compute_mappings`（非重算）

`h_pre` 不在 `HyperConnectionModule.forward` 的返回中，普通 forward hook 看不到它。因此本 monitor **包裹
（wrap）每个 mHC 模块的 `compute_mappings` 绑定方法**，直接捕获其真实返回的 `(h_pre, h_post, h_res)` —— 不重算。
另外包 `_compute_h`（拿 Sinkhorn 之前的 `h_res` logits）和 `fused_h_res_h_post_bda`（拿子层输出，算分支/残差占比）。

- `compute_mappings` 是普通 Python 方法（仅 `@nvtx_decorator`，非 `@torch.compile`），在 `_forward_normal` 中以
  `self.compute_mappings(...)` 调用，且**不被 checkpoint**，因此在 grad-enabled 的正向中恰好执行一次；实例属性
  wrapper 干净地遮蔽类方法。
- 整个捕获逻辑受 `_should_monitor()` 门控（含 grad 门），非监控步只是 `orig(x)` + 一次布尔判断。
- `remove_hooks()` 恢复原始方法并清空所有状态。

### VRAM 安全（无泄漏）

wrapper 不保留任何跨调用状态，只有固定的 0 维累加器。规则：

- 捕获后立即对 `h_pre/h_post/h_res` `.detach()`，并在 `no_grad()` 下做全部指标计算——否则一个仍带梯度的
  张量会通过反向把整层 autograd graph 钉住（大泄漏）。
- wrapper 原样返回 `out`，除 0 维标量外不保留任何对它/其视图的引用。
- `remove_hooks()` 恢复原始方法，不残留模块引用。


热路径纪律见 `.claude/skills/monitor-hook-perf-rules`：hook 内无 D2H 同步、无集合通信，schema 在 `allocate_buffers`
前声明。TP 不沿 `n` 切分映射，故无需 hook 内通信；跨 rank 归约在 flush 时由 `gather_and_aggregate` 完成（mean）。

---

## 监控指标

每个 hc 模块产出 29 个指标（本节均为 paddlefleet 后端；megatron 后端未随本次改动调整，仍是 16 个），指标名以
`attn_` / `mlp_` 前缀区分。`branch_residual_share_max`、`h_res_logits_max`、`h_res_logits_grad_max`、
`composite_amax_gain_{fwd,bwd}_max`、`h_{pre,post}_logits_max`、`bias_{pre,post,res}_abs_max` 取极大值，
`h_res_logits_min`、`h_res_logits_grad_min` 与 `h_{pre,post}_logits_min` 取极小值，其余按 token/batch 求均值
（并在 flush 时对 microbatch/rank 求均值）。日志键形如 `mhc_health/layer_{i}/{c}_{name}`，
`{c}` ∈ `{attn, mlp}`；对应的 `mhc_health/global_{c}_{name}` 由逐层累加器在 flush 时自动派生。

此外还有一组**逐元素展开的映射序列**（`n² + 2n` 条 / hc 模块），见下方「逐元素映射序列」。

### 逐元素映射序列（h_res cell / h_pre、h_post per-stream）

上面 29 个标量都是对 `h_res` / `h_pre` / `h_post` 做了某种聚合（均值、极值、行列和）之后的读数，
因此无法回答「矩阵长什么样」——例如 `h_res` 是否退化成近似单位矩阵（各流互不混合）、还是某一列吸走了
全部质量。论文 Figure 10 那类映射热图需要的是矩阵本身，所以这组指标把三个映射**逐元素**记录下来：

| 指标 | 公式 | 诊断意义 |
|------|------|----------|
| `{c}_h_res_cell{k}` | `mean_t( h_res[t, r, c] )`，`k = r·n + c`（行主序） | 残差混合矩阵的单个 cell。对角元 = 该流的自我保留，非对角元 = 跨流混合 |
| `{c}_h_pre_idx{j}` | `mean_t( h_pre[t, j] )` | 第 `j` 条流的聚合门开度 |
| `{c}_h_post_idx{j}` | `mean_t( h_post[t, j] )` | 第 `j` 条流的扩展门开度 |

**朝向**：`h_res_cell{k}` 记录的是 `compute_mappings` 返回的 `h_res` 本身，即与 `amax_gain_fwd`
读列和时同一个朝向，**不是** composite 乘积链用的 `h_resᵀ`（见「amax-gain」一节）。读热图时
行索引 `r = k // n`、列索引 `c = k % n`；按本文档的约定，**列**和是前向增益、**行**和是反向增益。

**行/列和不必单独出指标**：mHC 的 `h_res` 经 Sinkhorn 投影后是双随机矩阵，行和与列和恒为 1
（这也是 `amax_gain_{fwd,bwd}` 作为不变量守卫的依据）。论文 Figure 10 里标在 HC 那一行的
forward/backward gain 之所以有信息量，是因为原始 HC 的映射没有这个约束；mHC 这一行标的就是 1.00。

**实现方式**：这三组走 `declare_layer_vector` / `record_layer_vector`（与 `moe_health` 的 per-expert
占比同一机制），而不是 `n² + 2n` 次 `record_layer_metric`。两个后果：

- 热路径上每个映射只有一次向量 `add_`。若逐元素标量记录，`n = 4` 时每个模块每 microbatch 要多 24 次
  kernel launch，43 层 × 2 模块 × 16 microbatch ≈ 33k 次/step，直接违反
  `.claude/skills/monitor-hook-perf-rules` 的启动预算。
- 向量指标**不参与 `global_*` 派生**：单个 cell 在 43 层上求均值没有判读意义。跨层视图交给 viewer
  自己从各层曲线汇总。

`h_res` 的 token 均值本来就要为 composite 乘积算一次（`_h_res_snapshot`），这组指标复用同一个 reduce，
只多出一次 transpose 之外的 reshape，没有额外的归约开销。

`log_per_layer=False` 时这三组整体不产出（`declare_layer_vector` 直接 return）。
key 数量：`n = 4`、43 层、2 模块 = 2064 条逐层序列，是本 monitor 标量部分（29 × 2 × 43 ≈ 2494）的同量级。
`cell` / `idx` 这两个 tag 刻意选了任何标量指标名都不含的词——`_s{j}` 会和已有的 `h_pre_std` 共享前缀，
`_stream{j}` 会和 `h_post_stream_concentration` 冲突，都会破坏下游按前缀分组的逻辑。

### 门控统计（h_pre / h_post）

| 指标 | 公式 | 诊断意义 |
|------|------|----------|
| `{c}_h_pre_mean`  | `mean(h_pre)`  | 聚合门均值 |
| `{c}_h_pre_std`   | `std(h_pre)`   | 聚合门离散度 |
| `{c}_h_post_mean` | `mean(h_post)` | 扩展门均值 |
| `{c}_h_post_std`  | `std_{t,i}(h_post[t,i])` | token 与 stream 混合后的总离散度；单独看无法区分流间分工和 token 间变化 |

`h_post_std` 的信息量有限，但不是零：它能发现所有 gate entry 是否整体收缩为近似常数。它必须和
`h_post_token_std`、`h_post_stream_concentration` 联合解释。总离散度高但 token std 低，更像稳定的 stream 间分工；
两者都高，才说明每个 token 的总门量也在明显变化。反过来，`h_post_token_std -> 0` 仍允许不同 token 在 stream 间
重新分配相同总门量，因此不能据此断言门控已完全失去输入区分能力。

### 结构指标（paddlefleet 独有）

`h_post_mean` / `h_post_std` 把 token 轴和 stream 轴一起池化，因此「n 个流一起变弱」与「n-1 个流死掉、
剩一个扛全部」在均值上不可分；`branch_residual_share` 则回答 `h_post_mean` 回答不了的问题：门变小可能被
`F(·)` 变大抵消，只有两项的相对大小能说明该层是否还在写入。

| 指标 | 公式 | 诊断意义 |
|------|------|----------|
| `{c}_h_post_stream_concentration` | `mean_t( max_i h_post[t,i] / mean_i h_post[t,i] )` | 流间集中度，区间 `[1, n]`。1 = 各流等权；趋 `n` = 质量压到单流，该层退化成普通单流残差，`num_residual_streams` 预算未被使用 |
| `{c}_h_post_token_std` | `std_t( mean_i h_post[t,i] )` | token 间“总门量”的离散度。→0 只说明各 token 的 stream 均值接近，不代表每条 stream 的门都与 token 无关 |
| `{c}_branch_residual_share` | `mean_t( b / (b + r) )`，`b = ‖H_postᵀ F(·)‖_F`、`r = ‖H_resᵀ x_l‖_F` | 本层写入量在「写入 + 残差重混」中的占比，区间 `[0, 1]`。0.5 = 两项等量；趋 0 = 该层退化为纯残差搬运。|
| `{c}_branch_residual_share_max` | `max_t( b / (b + r) )` | 长尾告警：捕获均值掩盖的 branch-dominated token。对 token 数量和离群点敏感，且两项绝对值都很小时比值仍可贴 1，不能脱离均值和幅度单独诊断 |

两个范数都是精确值，但不物化 `[tokens, n, C]` 中间量：分支项是外积（`‖h_post_t ⊗ xb_t‖_F = ‖h_post_t‖₂·‖xb_t‖₂`），
残差项用 `[tokens, n, n]` 的 Gram/mix 收缩得到，`n = 4` 时开销约为本层投影的 `C/n²` 分之一。
分支项在 dropout **之前**测量，`hidden_dropout_prob > 0` 的配置会略微高估分支幅度（此处预训练配置为 0）。

### amax-gain（Sinkhorn 收敛哨兵）

paper 定义的最坏情况增益：算子的**最大绝对行和**界定前向传播的最坏放大。这里的算子不是 `h_res` 本身——
`apply_h_res` 与 fused cuTile kernel 都按 `out_i = Σ_j h_res_ji · x_j` 混流，即真正作用在流向量上的是 `h_resᵀ`
（见 `hyper_connection.py:466-471`、`fused_mhc_kernels.py:367-373`）。所以前向增益是 `h_res` 的**列**和、
反向增益是**行**和。对每个 token 的 `n×n` 矩阵计算，再对 token 求均值：

```
amax_gain_fwd = mean_t( max_i | Σ_j  h_res_ji | )      # h_res 的列和 = h_resᵀ 的行和（forward）
amax_gain_bwd = mean_t( max_j | Σ_i  h_res_ji | )      # h_res 的行和（backward）
```

| 指标 | 诊断意义 |
|------|----------|
| `{c}_amax_gain_fwd` | 单层前向最坏放大。列和被 Sinkhorn 最后一步归一钉死，是不变量守卫 |
| `{c}_amax_gain_bwd` | 单层反向最坏放大。承载 Sinkhorn 截断迭代的收敛残差 |

**哪个轴带信息**：我们的 `_sinkhorn_normalize` 最后一步是**列**归一，所以列和按构造恒为 1，行和才承载截断迭代的
残差（偏离量来自 `eps = 1e-6`，量级很小）。所以 `_bwd` 只能当哨兵、不要做诊断；`_fwd` 一旦偏离 1 反而更严重，
说明归一化本身出了问题（数值异常或实现改动）。

注意论文 Eq. (9) 最后一步是**行**归一，所以它 Fig 7 里会动的是 forward 那条，而我们会动的是 `_bwd`——
**两边曲线不可直接对比。**

### 迭代前 logits 范围

直接统计 `_compute_h` 产出、进入 Sinkhorn 前的 raw residual-mixing logits `z`。这两条保留符号，不执行
softmax，因此不会受概率饱和到 0/1 的截断影响：

| 指标 | 公式 | 诊断意义 |
|------|------|----------|
| `{c}_h_res_logits_min` | `min z` | step 内该层所有 microbatch/token/stream 的最小 raw logit。按 **min** 归约 |
| `{c}_h_res_logits_max` | `max z` | 同上取最大值。按 **max** 归约 |

`max-min` 可作为跨 token 的保守全局跨度，用于观察 logits 尺度持续扩张或异常尖峰；它不是逐行跨度的最大值，
因此不能直接套用单行 softmax 的精确饱和阈值。与下面的 logits gradient min/max 联合看，可以区分 logits 尺度漂移与
反向信号减弱。

### 迭代前 logits 梯度

对 `_compute_h` 产出、传入 Sinkhorn 的 raw `h_res` logits `z` 注册 tensor gradient hook，监控的是
`dL/dz`，不是 Sinkhorn 最终矩阵或循环中间 `M` 的梯度：

| 指标 | 公式 | 诊断意义 |
|------|------|----------|
| `{c}_h_res_logits_grad_min` | `min dL/dz` | step 内该层所有 microbatch/token/stream 的最小 activation gradient。按 **min** 归约 |
| `{c}_h_res_logits_grad_max` | `max dL/dz` | 同上取最大值。按 **max** 归约 |

hook 热路径只在 GPU 上计算 fp32 min/max 并写 0 维 accumulator。AMP 下 hook 先收到 scaled gradient，随后在
`on_optimizer_begin`（`scaler.step/update` 之前）用本 step loss scale 统一反除；不除 gradient accumulation，保留每个
microbatch 在真实反向路径上的 activation-gradient 量级。recompute 的首次 forward 运行于 `no_grad`，由现有
`_should_monitor()` 跳过；grad-enabled backward replay 再注册 hook，因此不会重复采集 checkpoint 的首次执行。

### 门控 logits 范围（h_pre / h_post，pre-sigmoid）

Eq. (8) 的 `H_pre = σ(·)`、`H_post = 2σ(·)` 把值域压进 `(0,1)` / `(0,2)`，饱和后从激活值已经看不出来
logit 有多极端。这两对指标读的是 sigmoid **之前**的 Eq. (7) 结果：

```
h_pre_logits  = r · proj[..., :n]    · α_pre  + b_pre
h_post_logits = r · proj[..., n:2n]  · α_post + b_post
```

| 指标 | 公式 | 诊断意义 |
|------|------|----------|
| `{c}_h_pre_logits_min` | `min` 上式 | 聚合门是否饱和到 0。按 **min** 归约 |
| `{c}_h_pre_logits_max` | `max` 上式 | 聚合门是否饱和到 1。按 **max** 归约 |
| `{c}_h_post_logits_min` | `min` 上式 | 扩展门下饱和哨兵。按 **min** 归约 |
| `{c}_h_post_logits_max` | `max` 上式 | 扩展门上饱和哨兵。按 **max** 归约 |

`_compute_h` 只返回激活后的门，所以这四条由 monitor 在 wrap 点用它自己的入参 `(proj, r)` 加模块的
`alpha` / `bias` **按同一行公式重算**（`mhc_metrics.gate_logits_extrema`），不是对 sigmoid 求逆——求逆会
把所有饱和元素丢成 `±inf`，恰好丢掉这两条指标存在的意义。代价是与模型实现耦合：`_compute_h` 里
`h` 的构成方式一旦改变，这四条会静默算错，改模型时必须同步。`H_post` 的因子 2 在 sigmoid 之外，不进 logit。

调用方没有传 `proj` / `r` 时（stub、或未来绕过该签名的 fused 路径）这四个累加器保持为空，不写值。

### Eq. (7) 静态参数（alpha / bias）

论文 Eq. (7) 的静态半边：三个标量门控因子 `α_pre` / `α_post` / `α_res`，以及一个 `[n² + 2n]` 的
`bias`——它按切片承担论文的 `b^pre`（`[:n]`）/ `b^post`（`[n:2n]`）/ `b^res`（`[2n:]`），因此分片统计而不是
合成一个数。

| 指标 | 公式 | 诊断意义 |
|------|------|----------|
| `{c}_alpha_pre` / `{c}_alpha_post` / `{c}_alpha_res` | 参数值 | 动态映射门控因子，从 `mhc_init_gating_factor` 起漂移；`α_res` 直接决定 Sinkhorn 输入 logits 的尺度 |
| `{c}_bias_{pre,post,res}_mean` | `mean(bias 切片)` | 静态偏置整体位置（初始为 0） |
| `{c}_bias_{pre,post,res}_abs_max` | `max abs(bias 切片)` | 单元素跑飞哨兵，均值会掩盖它。按 **max** 归约 |

采集在 `on_optimizer_begin`（经 `finalize_scaled_grad_metrics`）读一次参数，不在热路径——否则 9 条 × 层数 ×
2 component × microbatch 数会平白多出上万次 kernel launch。选这个钩子而不是 flush 是因为它在
`optimizer.step()` **之前**，读到的是本 step forward 实际用的参数值；`_flush_buffers` 保留为兜底路径（`on_step_end`
在 optimizer step 之后，走兜底时记到的是更新后的值，比同 step 其余曲线晚一次更新）。两处都调也只记一次。
参数在 TP 各 rank 上是同一份副本（`HyperConnectionModule` 用普通 `nn.Linear` 并把参数标记为
`is_distributed=False`），所以不需要任何 collective。

只有本 step 真正跑过被监控的 forward 才记录（`_captured_this_step`），这样 `monitor_interval > 1` 时参数曲线
与激活曲线落在同一批 step 上。注意命名是 `bias_*_abs_max` 而不是 `bias_*_absmax`：`training_logs` 的
max/min 分类器按 `_max` 后缀字面匹配。

### 复合映射（composite_amax_gain_{fwd,bwd}_max）

#### 论文定义

论文第 14 页 Figure 7(b) 按 residual branch 编号。记投影后的第 `l` 个 residual mixing 算子为
`R_l = P_{M_res}(H_l^res)`，总 branch 数为 `L`。论文的 forward composite 是从入口到当前位置的 prefix：

```
F_l = prod_{i=1}^{l} R_{l+1-i}
    = R_l @ R_{l-1} @ ... @ R_1
```

论文的 backward composite 则是从当前位置到模型尾部的 suffix：

```
B_l = prod_{i=1}^{L-l} R_{L-i}
    = R_{L-1} @ R_{L-2} @ ... @ R_l
```

前向信号经过 `F_l`；从模型输出反传到位置 `l` 的梯度经过 `B_l^T`。因此论文中的 composite forward amplification
应读 `||F_l||∞`，composite backward amplification 应读 `||B_l^T||∞ = ||B_l||1`。这里的 `l` 是每个顺序执行的
residual branch；映射到 Transformer 时，attention 和 MLP 是交错的相邻 branch，不能拆成两条互不相干的链。

#### 当前 PaddleFleet 实现

`apply_h_res` 的实际运算是 `mixed = h_resᵀ @ residual`，所以实现先对 token 求平均并转置，得到真正作用在
流向量上的算子 `T_q = mean_token(h_res_q)ᵀ`。所有非 MTP branch 按物理执行顺序排列：

```
T_{0,attn}, T_{0,mlp}, T_{1,attn}, T_{1,mlp}, ...
```

`{c}` 只标识当前观测点是 attention 还是 MLP branch，不再表示两条独立累积链。正序遍历构造入口到当前 branch 的
prefix `F_q = T_q @ ... @ T_0`；逆序遍历构造当前 branch 到尾部的 suffix
`B_q = T_tail @ ... @ T_q`。

| 指标 | 公式 | 精确语义 |
|------|------|----------|
| `{c}_composite_amax_gain_fwd_max` | `||F_q||∞ = max_i Σ_j |F_{q,i,j}|` | 入口到当前 `{c}` branch 的交错 prefix 前向最坏放大；global 按 **max** 归约 |
| `{c}_composite_amax_gain_bwd_max` | `||B_qᵀ||∞ = ||B_q||₁ = max_j Σ_i |B_{q,i,j}|` | 模型尾部反传到当前 `{c}` branch 的交错 suffix 反向最坏放大；global 按 **max** 归约 |

该实现与论文 Figure 7(b) 的 prefix-forward / suffix-backward 定义一致。

四个实现要点：

1. **累乘的是转置。** 直接对 `h_res` 累乘既不是前向链也不是反向链——`h_res_k ⋯ h_res_0` 与真实算子
   `A_k = (h_res_0 ⋯ h_res_k)ᵀ` 是不同的矩阵。快照里存的就是 `h_resᵀ`。单层的 `amax_gain_{fwd,bwd}` 同样按
   `h_resᵀ` 口径计算（只是没有显式转置，直接换了求和轴），所以两组指标的 `_fwd`/`_bwd` 语义一致、可以对着读。
2. **先对 token 求平均再累乘**（论文 Fig 7/8 同口径），所以每层每组件只存一个 `n×n` fp32 快照，而不是把 per-token
   `h_res` 在整个 step 内驻留。这是它当初被下线的第一条理由，现在不成立。
3. **累乘发生在 microbatch 结算时**，快照按 `(layer_idx, attn-before-mlp)` 排序，不在 hook 里增量累乘。因此结果与
   hook 触发顺序无关；recompute 即使按反向层序重放，也会恢复成物理 branch 顺序。回归测试会故意逆序触发 hook，
   并用非交换矩阵分别核对正序 prefix 和逆序 suffix。
4. **MTP 层被排除。** MTP 层不在主干传播路径上，乘进链条没有物理意义。

作用域限制：复合只覆盖**本 rank 持有的层**。在 `sharding: stage1` 且无 PP 时，每张卡持有全部 Transformer 层，
所以覆盖全网层范围。真开 PP 后它会退化成 stage 内语义，而我们既检测不到也修不了——`get_pipeline_model_parallel_rank`
和 `get_pipeline_model_parallel_world_size` 在 PaddleFleet 里目前是 stub（无条件返回 0 / 1，
`parallel_state.py:229`、`:246`）。要支持 PP 需要先等这两个函数实现，再补一次 flush 时的 all-gather；届时
**collective 必须放在所有 per-layer `try/except` 之外**，否则单个 rank 吞异常会让整个 job 挂死而不是报错。

### 曾经下线、现已恢复

- `{c}_amax_gain_bwd`：曾因读数与 `amax_gain_fwd` 一致而下线。这个结果本身是预期的（`_fwd` 读的列和被 Sinkhorn
  最后一步钉死为 1），两条现在都保留：`_fwd` 当不变量守卫，`_bwd` 才是承载收敛残差的那条。
- `{c}_composite_amax_gain_{fwd,bwd}_max`：恢复时先解决了 per-token `h_res` 全 step 驻留和 recompute 逆序触发；本次进一步
  按论文改为 attention/MLP 交错的 prefix-forward / suffix-backward，并在每个 microbatch 结束时结算到 GPU accumulator。
  因而一个 optimizer step 内的全部 gradient-accumulation microbatch 都参与既定的 max 归约。



---

## 与 microbatch / 激活重算的交互

- 每个梯度累积 microbatch 都会重新调用所有 hc 模块。中间 microbatch 在 `on_substep_end` 结算 composite，最后一个在
  `on_step_end` 的 flush 前结算；每次结算后立即清空快照。各 microbatch 的结果写入同一 GPU max accumulator，因此
  optimizer-step 日志表示全部 microbatch 中每个 branch 的最坏 composite gain。
- 整个捕获受 `_should_monitor()` 门控；非监控步只是 `orig(x)` + 一次布尔判断。
- `_should_monitor()` 要求 grad enabled。这个判据的名字暗示"跳过重算"，**实际语义相反**：recompute 的真前向跑在
  `no_grad` 下，反向重放才开 grad，所以采集实际发生在反向重放。数值不受影响（重算确定性），但依赖执行顺序的指标
  会出错——这曾经让复合映射下线；现在结算时按 `(layer_idx, attn-before-mlp)` 恢复物理顺序，不再受影响。
- 这个判据还会过滤同一 microbatch 中不承载梯度的重复路径：mHC 的 hyper-connection 在一次前向里会被进入
  两次（`high_precision_mhc` 打开时 AMP 关闭的 fp32 路径 + bf16 路径），两条都记会稀释所有均值。详见 README
  「已知限制：采集口径依赖 grad 判据」。
