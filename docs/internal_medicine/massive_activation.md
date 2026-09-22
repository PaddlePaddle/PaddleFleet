# Massive Activation Monitor

**Residual Stream Massive Activation 健康监控模块**，覆盖通道异常、模块输出与 residual 写入尺度。

基于论文发现实现：

> Sun, S., Canziani, A., LeCun, Y., & Zhu, J. (2026).
> *The Spike, the Sparse and the Sink: Anatomy of Massive Activations and Attention Sinks.*
> arXiv:2603.05498.

---

## 背景：什么是 Massive Activations？

在 pre-norm Transformer（如 Llama、Qwen、ERNIE）中，少数 token 在少数 hidden state channel 上出现**极端异常值**——比正常激活大 2-3 个数量级。这些异常值：

1. 由早期 FFN block（step-up block）通过 SwiGLU 的**方向性二次放大**注入
2. 通过残差连接在中间层**持续累积**
3. 在末尾 FFN block（step-down block）被**反向中和**

```
Layer:    1  2  3  4  ...  30  31  32
Spike:    -  -  -  ↑↑↑  ===  ===  ↓↓↓  -
                   step-up    step-down
```

### 为什么要监控？

- **Massive activations 独立于 PPL 变化**：loss 看不出问题，但 spike 暴涨可能导致量化精度严重退化（论文 Table 3）
- **是 attention sink 的上游信号**：spike token 经过 RMSNorm 后变成稀疏近常量向量，为 sink 创造条件
- **Weight decay 关联**：关闭 weight decay 后 spike 从 ~3800 涨到 ~12000，但 PPL 仅变 0.3
- **可诊断训练异常**：spike magnitude 突变通常意味着权重矩阵出现了异常高增益结构

---

## 监控指标

### 1. Channel Max (通道最大激活值)

**数学公式：**
```
channel_max = max(|H_i|)
```

对 post-residual hidden states 所有 position 和 channel 取绝对值最大值。

**诊断意义：**
- 正常 pre-norm Transformer 中间层：数百到数千量级
- 首尾层应显著低于中间层（生命周期特征）
- 训练中突增可能预示数值不稳定

---

### 2. Channel Median / P95 / P99 (通道峰值分布)

**数学公式：**
```
channel_median = median(per_channel_max)
channel_p95 = p95(per_channel_max)
channel_p99 = p99(per_channel_max)
```

对每个 hidden channel 的最大绝对值做分布统计。

**诊断意义：**
- `channel_median` 是 residual stream 的基准通道量级
- `channel_p95/p99` 用于判断是否是大范围 scale growth，而不是少数 outlier channel
- 如果 `channel_max` 上升但 `p95/p99` 基本不动，更像少数通道 spike；如果三者一起上升，更像整体激活量级膨胀

---

### 3. Channel Max Ratio (通道异常比)

**数学公式：**
```
channel_max_ratio = max(per_channel_max) / median(per_channel_max)
```

最大 channel 激活与中位 channel 激活的比值。

**诊断意义：**
- 典型 spike 层：ratio > 100x（少数 channel 远超其他）
- ratio 持续攀升说明模型在强化 spike channel 的放大路径
- 非 spike 层（首尾）应接近 1-10x

---

### 4. Massive Activation Channel Count (异常通道数)

**数学公式：**
```
massive_act_channel_count = |{c : max_pos(|H_i[:, c]|) > median × threshold_multiplier}|
```

超过阈值（默认 100× 中位数）的 channel 数量。

**诊断意义：**
- 论文发现 spike channel 通常只有 **2-5 个**（Property ii）
- 数量突增说明模型正在更多 channel 上制造极端值——可能是训练不稳定的信号
- 配合 topk_channel_norm 看趋势

---

### 5. Absolute Channel Counts (绝对阈值通道数)

**数学公式：**
```
channel_count_gt_10 = |{c : per_channel_max[c] > 10}|
channel_count_gt_20 = |{c : per_channel_max[c] > 20}|
channel_count_gt_30 = |{c : per_channel_max[c] > 30}|
```

按绝对激活幅度统计超过 10、20、30 的 channel 数量。

**诊断意义：**
- `massive_act_channel_count` 是 median-relative outlier 数量
- `channel_count_gt_10/20/30` 是 absolute scale growth 数量
- 默认 `10/20/30` 更适合当前 1.5B 训练的激活量级；如果观察大幅度 BF16 runaway，可以通过 `absolute_thresholds` 显式切到更高阈值，例如 `(50.0, 100.0, 200.0)`
- 这组指标适合对比不同训练设置下 residual stream 是否整体变大，例如 BF16 vs FP4

---

### 6. Top-3 Channel Norm (Top-K 通道范数)

**数学公式：**
```
topk_channel_norm = ||topk(per_channel_max, k=3)||₂
```

前 3 个最大 channel 的 L2 范数。直接对应论文 Figure 1 的 "top-3 channel magnitudes"。

**诊断意义：**
- 应呈现 "rise–plateau–fall" 模式（上升→平台→下降）
- 中间层平台值是模型的 "spike fingerprint"
- 跨步骤对比可检测 spike 强度的趋势变化

---

### 7. Activation RMS (整体激活 RMS)

**数学公式：**
```
activation_rms = sqrt(mean(H_i^2))
```

对 residual stream 的全部 token 和 channel 计算 RMS。

**诊断意义：**
- 用于观察整体激活量级，而不是只看最大 channel
- 与 `channel_p95/p99`、`channel_count_gt_*` 配合判断 broad scale growth
- 对量化训练很有参考价值：RMS 上升意味着更多通道进入较高动态范围

---

### Module Positions and Residual Writes

PaddleFleet 在五个真实执行位置记录 `rms`、`abs_max`、`abs_p99` 和 `outlier_ratio`：

```text
layer_input
attn_out
post_attn_residual
ffn_or_moe_out
post_ffn_residual
```

其中 `outlier_ratio = mean(|x| > 10 * RMS(x))`。`attn_out` 与 `ffn_or_moe_out` 是模块直接输出；两个
post-residual 值从 forward 的真实边界采样，不通过手工相加重建。实际写入比例定义为：

```text
update_rms_ratio = RMS(residual_after - residual_before) / RMS(residual_before)
```

因此该比例同时覆盖 bias、dropout、BDA，以及 mHC 中 residual mixing 对实际 residual stream 的影响。
若前后 residual 形状不一致（例如某些 MTP 打包路径），该比例不会输出，以避免比较不匹配的张量。

---

### 8. Post-Norm Sparsity (归一化后稀疏度)

**数学公式：**
```
post_norm_sparsity = mean(|RMSNorm(H_i)| < ε)
```

RMSNorm 后接近零的 entry 占比（默认 ε=0.01）。

**理论基础（论文 Eq. 24）：**
RMSNorm 将 spike token 的非 spike channel 压制为接近零，产生稀疏向量：
```
RMSNorm(h^(s)) ≈ Σ_{i∈C} h̃_i^(s) e_i
```
其中 C 是 spike channel 集合，结果是一个近似 multi-hot 的稀疏表示。

**诊断意义：**
- 高稀疏度 = 模型正在通过 spike+norm 创造 "implicit parameters"
- 结合 sink 指标看：高 sparsity + 高 sink ratio = 经典的 spike→sink 路径激活

---

### 9. Post-Norm Cosine Stability (归一化后余弦稳定性)

**数学公式：**
```
post_norm_cosine = mean(cosine_sim(RMSNorm(H[a]), RMSNorm(H[b])))
```

随机采样 token 对的归一化表示之间的余弦相似度。

**理论基础（论文 Eq. 25, Figure 5）：**
不同 spike token 归一化后坍缩为**近常量向量**：
```
RMSNorm(h^(a)) ≈ RMSNorm(h^(b))
```
这使得 sink token 的 key 向量几乎不变，创造稳定的 attention sink 位置。

**诊断意义：**
- cosine → 1.0 说明 step-up block 已完成 spike 注入（Figure 5）
- 在 step-up block 之前应该较低，之后应跳升到接近 1.0
- 用于定位 step-up block 的确切位置

---

### 10. Spectral Norm Bounds (谱范数上下界)

**数学公式：**
```
ratio_t = post_layer_rms_t / pre_layer_rms_t = ||y_t|| / ||x_t||   (每 token)
spectral_norm_max = max_t(ratio_t)     # 在一个 global batch 内对所有 token 取 max
spectral_norm_min = min_t(ratio_t)     # 在一个 global batch 内对所有 token 取 min
```

对每个 token，用该层输出残差 `y_t` 与输入残差 `x_t` 的 RMS 之比近似该层作为线性映射的
增益 `||y_t|| / ||x_t||`（`sqrt(H)` 约掉）。在一个 global batch 内对所有 token 归约：

- `spectral_norm_max`（ratio 的最大值）是该层**谱范数（最大奇异值 σ_max）的下界** ——
  任意观测到的增益都 ≤ `sup_x ||f(x)||/||x||`。
- `spectral_norm_min`（ratio 的最小值）是该层**最小奇异值 σ_min 的上界** ——
  任意观测到的增益都 ≥ `inf_x ||f(x)||/||x||`。

**实现要点：**
- 残差在 transformer layer 边界携带完整 hidden 维（LayerNorm 需要完整 H），因此每 token
  RMS 是真实的全向量 RMS，**无需 TP 通道归约**。
- 指标是对 token 取 max/min；token 可能被切分到不同 rank（sequence-parallel / DP / PP），
  但 `gather_and_aggregate()` 在 flush 时做全 world 的 `all_gather` + max/min 归约，因此整个
  global batch 的 max/min 在跨 rank 时正确合成，**hook 内无需任何 collective**。
- "一个 global batch" 的语义依赖既有约定：`monitor.step()`（flush 累加器）每个 optimizer
  step 调用一次；窗口内 `record_max`/`record_min` 跨所有 microbatch 累积极值。

**诊断意义：**
- `spectral_norm_max` 持续 > 1 且上升 → 该层放大残差流，可能驱动 spike 生命周期的 rise 段
- `spectral_norm_min` 远小于 1 → 该层对部分 token 强烈压缩，关注训练稳定性

---

### 11. Lipschitz / Gradient-Gain Bounds (每层 Lipschitz 常数)

**数学公式：**
```
dx_t = ∂L/∂x_t   (loss 对该层输入 hidden 的梯度)
dy_t = ∂L/∂y_t   (loss 对该层输出 hidden 的梯度)
ratio_t = ||dx_t|| / ||dy_t|| = ||Jᵀ dy_t|| / ||dy_t||   (每 token, sqrt(H) 约掉)
lipschitz_max = max_t(ratio_t)   # 一个 global batch 内对所有 token 取 max
lipschitz_min = min_t(ratio_t)   # 一个 global batch 内对所有 token 取 min
```

反向传播给出 `∂L/∂x = Jᵀ·∂L/∂y`（`J = ∂y/∂x` 为该层前向映射的 Jacobian）。因为 `J` 与
`Jᵀ` 奇异值相同，每 token 的梯度增益比 `||dx_t||/||dy_t||` 即 `Jᵀ` 作用在观测梯度上的增益。
在一个 global batch 内对所有 token 归约：

- `lipschitz_max`（ratio 的最大值）是该层 **σ_max(J) 的下界** —— 即该层前向映射
  **Lipschitz 常数的下界**：任意观测增益 ≤ `sup_v ||Jᵀv||/||v||`。
- `lipschitz_min`（ratio 的最小值）是该层 **σ_min(J) 的上界**：任意观测增益 ≥ `inf_v ||Jᵀv||/||v||`。

它同时直接刻画反向传播中的**梯度放大系数**：`lipschitz_max > 1` 表示梯度反向穿过该层时被放大
（爆炸风险），`<< 1` 表示被压缩（消失风险）。

**实现要点：**
- 该指标是**反向**量，但由 forward hook 在层的输入/输出 hidden 张量上注册 tensor grad hook
  （`.register_hook`）来采集 —— 因为 Megatron 以全 keyword 参数调用 layer，module 级
  `register_full_backward_hook` 的 `grad_input` 为空，拿不到输入梯度。
- forward hook 的 grad guard（`torch.is_grad_enabled()`）会跳过 activation recompute 的初始
  no-grad 前向，只在 grad-enabled 的重算前向上注册，故 grad hook **恰好触发一次**（已在
  reentrant / non-reentrant checkpoint 下验证）。
- 输入/输出梯度均为完整 H 向量，每 token RMS 是真实全向量 RMS，**无需 TP 通道归约**；
  max/min 跨 token-partition rank 由 `gather_and_aggregate()` 在 flush 时 `all_gather` +
  max/min 合成，**hook 内无 collective**。
- 每 token 分解忽略了 attention 的 token 耦合（真实 Jacobian 会混合 token），与前向
  `spectral_norm` 指标做同样的简化近似。
- 仅在训练反向传播的 monitored step 记录（eval/inference 下 `requires_grad=False` 会跳过）；
  被 CUDA graph 捕获的层不触发 eager hook，因而不产出该指标。

**诊断意义：**
- `lipschitz_max` 持续 > 1 且上升 → 该层反向放大梯度，谱范数/Lipschitz 增大，关注梯度爆炸与训练稳定性
- `lipschitz_min` 远小于 1 → 该层对部分方向强烈压缩梯度，关注梯度消失

---

### 12. Logit-Lens Predictive Entropy + Logsumexp + Cross-Entropy (逐层 logit-lens 预测熵 / 对数配分 / 交叉熵，可选)

> **默认关闭**（熵/logsumexp 由 `log_logit_lens_entropy=True` 开启；交叉熵由 `log_logit_lens_cross_entropy=True` 开启）。开销较大：每监控层每监控步做一次 LM-head 前向（三个指标共用同一次投影）。

**数学公式：**
```
l_t = final_norm(h_t) · Wᵀ          # 第 t 个 token 经 LM head 投影到 vocab logits
p_t = softmax(l_t)
H(p_t)  = -Σ_v p_t[v] · log p_t[v] ∈ [0, log(vocab)]
log Z_t = logsumexp(l_t) = log Σ_v exp(l_t[v])     # 对数配分 / softmax 归一化项
CE_t    = log Z_t − l_t[y_t]                       # 对真实下一个 token y_t 的交叉熵
logit_lens_entropy_mean        = mean_t H(p_t)     # 仅报告 token 均值
logit_lens_logsumexp_mean      = mean_t log Z_t    # 仅报告 token 均值
logit_lens_cross_entropy_mean  = mean_t CE_t       # 仅报告 token 均值
```

用 *logit lens* 把每层残差 `h`（先过 `decoder.final_layernorm`，因为 LM head 是在 final-norm 后的
表征上训练的）投影到 vocab logits，同一次投影里求三组量（**仅报告 token 均值**）：

- **softmax 熵 `H(p)`**：衡量"这一层的表征对下一个 token 的预测有多笃定"——熵通常随深度下降（模型逐层
  收敛到预测），突变是有用的健康/异常信号，与 spike 生命周期、谱范数/Lipschitz 增益互补。
- **对数配分 `log Z = logsumexp(l)`**：softmax 的归一化项，也是 logits 的 soft-max，追踪该层在 lens 下
  累积出的**原始 logit 尺度**（`log Z ≥ max_v l_v`，且当某个 logit 明显占优时紧贴 max logit）。它与熵互补：
  熵刻画分布形状，`log Z` 刻画分布的绝对尺度/置信幅度。`log Z` 直接复用熵计算里已经算好的 `log_z`，**零额外开销**。
- **交叉熵 `CE = log Z − l[y]`**：把 logit lens 当成 loss——衡量"这一层的表征直接拿去预测真实下一个 token
  会有多大损失"。**最后一层的 CE 就是 LM loss**（up to loss-mask 加权，见下），随深度下降说明模型逐层把表征
  推向正确答案；某层 CE 异常回升说明该层表征偏离目标。CE 复用同一次投影里的 `log_z`，只多一次
  `gather`（**近零额外开销**）。

**Loss-mask 说明（交叉熵）：** label 通过一个挂在 head-owning chunk 上的顶层 forward pre-hook 从
`model.forward(..., labels=...)` 的 kwargs 中捕获，并按 `labels[b,s].transpose→[s,b].reshape(-1)` 对齐到
seq-major 展平的 hidden。但 **loss_mask 是在模型 forward 之外**（`masked_next_token_loss`）施加的，monitor
拿不到它，因此报告的是**未加权的 token 均值 CE**——它等于 LM loss up to loss-mask 加权。若某步 label 缺失
（eval/inference）或对齐失败（形状不符），当步直接不产出 CE（不报错）。

**实现要点（VRAM + 数值 + 并行）：**
- **分块**：投影 `final_norm(h) @ Wᵀ` 按 token 切成 `[chunk_size, vocab]` 的 tile 逐块计算并丢弃
  （`logit_lens_chunk_size`，默认 1024），任一时刻只物化一个 tile，**绝不 materialize 完整
  `[tokens, vocab]` logits**；三个指标跨块只累积标量运行和。
- **数值（用 torch.logsumexp）**：写成 `H = log_z − E_p[l]`，其中 `log_z = torch.logsumexp(l)`、
  `E_p[l] = Σ softmax(l)·l`。直接用 `torch.logsumexp` / `torch.softmax` 这两个融合且数值稳定的原语，
  **不手写 `shifted.exp()` 也不单独 `.log()`**；`log Z` 与 `CE` 都复用这里的 `log_z`。
- **vocab-parallel TP（暂不支持）**：`compute_logit_lens_entropy` 断言 `tp_size <= 1`。vocab 维被 TP
  切分时 softmax 归一化跨 rank，需要 MAX + SUM all-reduce，而这两步无法与 `torch.logsumexp` 融合
  （log 域的 partial 不能直接 SUM），因此当前**先断言关闭**，后续再补。caller 只在持有 head 的 stage
  attach，且假定该 head 未按 vocab 切分。
- **PP 覆盖**：只有持有 LM head 权重的 PP stage（最后 stage / tied-embedding stage）计算这组指标；
  其余 stage `_resolve_lm_head` 返回 `(None, None)`——不 attach hook、不声明 key、彻底 no-op，
  **不做任何权重广播**。这是有意设计，不是待修复的限制。
- **权重引用**：hook attach 时捕获 LM head 的 live `Parameter` 引用，optimizer 就地更新它，因此每步
  读到的都是当前权重，无需每 hook 查找。
- `logit_lens_layers`（默认 `None`=持有 head 的全部层）可只监控指定 global 层索引，进一步降开销。

**诊断意义：**
- 熵随深度**单调下降**是健康的"逐层定型"；某层熵**异常回升**说明该层表征偏离预测方向
- 全程熵**过高**（接近 `log(vocab)`）说明表征几乎不携带预测信息；**过低**说明过早 over-commit
- `log Z` **随深度攀升**反映 logit 尺度逐层放大；`log Z` 暴涨（配合低熵）= 该层 logit 过度膨胀 / over-confident，
  是量化/数值健康的预警
- CE **随深度下降**是健康的"逐层逼近答案"；最后一层 CE ≈ LM loss，可作为 monitor 内部对训练 loss 的一致性核对；
  某层 CE **异常回升**说明该层表征偏离正确 token

---

### 13. Hidden Spectral Entropy (post-norm 隐状态谱熵 / 有效秩，可选)

> **默认关闭**（`log_hidden_spectral_entropy=True` 开启）。相对轻量：一次 `eigvalsh`，无 LM head、无 full SVD。

**数学公式：**
```
h ∈ R^{n,d}          # post-RMSNorm 隐状态，n=token 数, d=通道数
σ_i                  # h 的奇异值
p_i = σ_i² / Σ_k σ_k² = σ_i² / ‖h‖_F²
hidden_spectral_entropy = -Σ_i p_i log p_i ∈ [0, log(min(n, d))]
```

对该层 **post-RMSNorm** 隐状态（caller 传入 `input_layernorm` 的输出，per-token RMS 缩放已被移除）计算谱（矩阵 /
von Neumann）熵。`p_i` 是把奇异值平方归一化成的分布（合法，因 `‖h‖_F² = Σ σ_i²`），熵衡量 token 集张成多少个
**有效方向**：低=坍缩到少数方向（秩坍缩），高=方向丰富；`exp(H)` 即**有效秩**（Roy–Vetterli）。

**实现要点：**
- **无 full SVD**：`σ_i²` 恰为较小 Gram 矩阵（`n≥d` 时 `hᵀh`，否则 `h hᵀ`）的特征值，用 `torch.linalg.eigvalsh`
  一次 GPU 调用求出，成本 `O(min(n,d)³)` 而非完整 `[n,d]` SVD，**无 host sync**。数值负特征值 clamp 到 0，`log`
  内再 clamp `eps` 处理 `0·log0`。
- **post-norm 位置**：在 `input_layernorm` 输出上计算（"归一化后即中心化"），与 `post_norm_sparsity/cosine`
  同一 hook；因此仅在存在 `input_layernorm` 的层记录（fallback 的 residual pre-hook 无归一化张量，同 post-norm
  指标的限制，可接受）。
- **set-level 非线性量**：不同于逐 token 均值指标，谱熵是对**整批 token 集**定义的非线性量，故跨 rank / 跨
  microbatch 的均值只是**近似**（mean-of-per-shard-entropies ≠ 全体 token 的谱熵）。按 per-shard 值在 flush 时
  平均，作为坍缩趋势信号可接受（设计取舍）。

**诊断意义：**
- 谱熵 / 有效秩**持续下降** → 表征秩坍缩（token 表示趋同，多样性丧失），常与 `post_norm_cosine → 1` 呼应
- 某层谱熵**异常低** → 该层把表征压到极少数方向，可能是 attention sink / 信息瓶颈的上游信号
- 谱熵**过高**（接近 `log(min(n,d))`） → 表征接近各向同性/白噪声，未形成结构

---

## 健康阈值参考

| 指标 | 值 | 状态 | 说明 |
|------|-----|------|------|
| `channel_max` | < 100 | NORMAL | 非 spike 层 |
| | 100 ~ 5000 | SPIKE | 典型 massive activation |
| | > 10000 | SEVERE | 极端放大，检查 weight decay |
| `channel_p95/p99` | 趋势平稳 | NORMAL | 多数通道量级稳定 |
| | 持续上升 | WARNING | residual stream 整体 scale growth |
| `channel_max_ratio` | < 10 | NORMAL | 各 channel 量级接近 |
| | 10 ~ 1000 | SPIKE | 存在少数异常 channel |
| | > 1000 | SEVERE | 极端通道不平衡 |
| `massive_act_channel_count` | 0 ~ 5 | NORMAL | 典型 spike pattern |
| | > 10 | WARNING | 异常 channel 过多 |
| `channel_count_gt_10/20/30` | 趋势平稳 | NORMAL | 高幅度通道数量稳定 |
| | 持续上升 | WARNING | 大范围激活膨胀 |
| `activation_rms` | 趋势平稳 | NORMAL | 整体残差流量级稳定 |
| | 持续上升 | WARNING | 需要关注训练或量化动态范围 |
| `post_norm_sparsity` | < 0.5 | NORMAL | 归一化后信息丰富 |
| | > 0.8 | HIGH | 高度稀疏，implicit parameter 效应 |
| `post_norm_cosine` | < 0.5 | DIVERSE | token 表示多样 |
| | > 0.9 | COLLAPSED | 近常量向量，sink 前提条件满足 |
| `spectral_norm_max` | ~1.0 | NORMAL | 层增益接近恒等 |
| | 持续 > 1 且上升 | WARNING | 层放大残差流，关注 spike/训练稳定性 |
| `spectral_norm_min` | ~1.0 | NORMAL | 层增益接近恒等 |
| | << 1.0 | WARNING | 对部分 token 强烈压缩 |
| `lipschitz_max` | ~1.0 | NORMAL | 层 Jacobian 增益接近恒等 |
| | 持续 > 1 且上升 | WARNING | 反向梯度放大，Lipschitz 增大，关注梯度爆炸 |
| `lipschitz_min` | ~1.0 | NORMAL | 层 Jacobian 增益接近恒等 |
| | << 1.0 | WARNING | 对部分方向强烈压缩梯度，关注梯度消失 |
| `logit_lens_entropy_mean` | 随深度下降 | NORMAL | 表征逐层收敛到预测 |
| | 某层异常回升 | WARNING | 该层表征偏离预测方向 |
| | 全程接近 `log(vocab)` | WARNING | 表征几乎不携带预测信息 |
| `logit_lens_logsumexp_mean` | 随深度平稳攀升 | NORMAL | logit 尺度逐层放大 |
| | 某层暴涨(配合低熵) | WARNING | 该层 logit 过度膨胀 / over-confident，量化数值预警 |
| `logit_lens_cross_entropy_mean` | 随深度下降 | NORMAL | 表征逐层逼近真实 token；末层 ≈ LM loss |
| | 某层异常回升 | WARNING | 该层表征偏离正确 token |
| `hidden_spectral_entropy` | 趋势平稳 | NORMAL | 表征有效秩稳定，方向丰富 |
| | 持续下降 | WARNING | 秩坍缩，表征趋同（配合 post_norm_cosine↑） |
| | 异常偏低 | WARNING | 压到少数方向，attention sink/瓶颈上游信号 |

---

## 与其他 Monitor 的交叉诊断

| 组合信号 | 诊断 |
|----------|------|
| `channel_max` 暴涨 + `qk_stats/sink` 不变 | Spike 还没传导到 attention（可能在 step-up 前） |
| `channel_max` 暴涨 + `qk_stats/sink` 上升 | 经典 spike→sink 路径激活 |
| `post_norm_sparsity` 高 + `qk_stats/entropy_min` 低 | 存在 dormant sink heads |
| `channel_max` 高 + `moe_health/router_entropy` 低 | 模型在用 spike 走捷径 |
| `topk_channel_norm` 平稳 + `qk_stats/sink_head_ratio` 上升 | Sink 在没有更多 spike 的情况下增长（替代策略） |

---

## 性能说明

### 计算开销
- **Pre-norm 指标**（channel_max 等）：一次 `abs().max(dim=0)` 加 per-channel 统计，复杂度约 O(S×H)
- **TP per-channel 聚合**：Megatron/PaddleFleet TP 切通道维时会在 hook 内对 per-channel max 做一次 `MAX all_reduce`，这是正确性所需
- **Post-norm 指标**：需要额外一次 RMSNorm forward（无梯度），开销约等于一个 norm 层
- **Cosine stability**：采样 256 对，O(256×H)，通常较小
- **Module-position 指标**：每个位置执行 O(S×H) 的尺度/tail 统计，其中精确 `abs_p99` 需要 quantile；大模型建议配合 `monitor_interval` 或 `sample_layers`
- **Logit-lens entropy + logsumexp + cross-entropy**（可选，默认关）：每监控层每监控步一次 LM-head 前向，复杂度约 O(S×H×vocab)，是本 monitor 中**最重**的指标；熵、`log Z`、CE 共用同一次投影（logsumexp 与 CE 均近零额外开销，CE 只多一次 `gather`）。按 token 分块把峰值显存限制在一个 `[chunk, vocab]` tile，但计算量仍显著，建议配合大 `monitor_interval` 与 `logit_lens_layers` 使用
- **Hidden spectral entropy**（可选，默认关）：一次 Gram matmul（O(n·d²)）加 `eigvalsh`（O(min(n,d)³)），无 LM head、无 full SVD、无 host sync，比 logit-lens 轻很多；大 batch 下 `eigvalsh` 的立方项可观，建议配合 `monitor_interval`

### 内存开销
- 不保存激活值，不影响梯度计算
- Megatron backend 会把 0-dim metric tensors 记录到 GPU accumulator，`monitor.step()` 再批量 flush 到 `training_logs`

### 推荐配置

```python
# 全量监控（< 32 层的模型）
setup_massive_activation_monitor(model, monitor_interval=10)

# 采样监控（大模型，如 64+ 层）— 只看首、中、尾层
setup_massive_activation_monitor(
    model,
    sample_layers=[0, 1, 2, 3, 16, 30, 31],  # step-up 区 + 中间 + step-down 区
    monitor_interval=10,
)
```

---

## 使用方式

### 基本用法

```python
from paddlefleet.internal_medicine import setup_internal_medicine

monitor_dict = {}
model = setup_internal_medicine(
    model,
    monitors=['massive_act'],      # 或 'all' 启用全部
    monitor_dict=monitor_dict,
    monitor_interval=10,
)
```

### 读取指标

```python
from paddlefleet.internal_medicine import training_logs

# 获取所有 spike 指标
spike_metrics = training_logs.get_latest(prefix='massive_act')

# 查看特定层
layer_4_max = training_logs.get_latest(prefix='massive_act/layer_4')

# 格式化打印
training_logs.print_metrics(prefix='massive_act')
```

### 定位 Step-Up/Step-Down Blocks

```python
# 在全部层上运行一次，检查 channel_max 的 layer profile
spike_metrics = training_logs.get_latest(prefix='massive_act')

# 找到 channel_max 突增的层 = step-up block
# 找到 channel_max 突降的层 = step-down block
for key, val in sorted(spike_metrics.items()):
    if 'channel_max' in key and 'ratio' not in key and 'global' not in key:
        print(f"{key}: {val:.1f}")
```

---

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `log_per_layer` | `True` | 记录每层指标 |
| `log_global` | `True` | 记录全局聚合指标 |
| `monitor_interval` | `1` | 监控间隔 (每 N 步) |
| `verbose` | `False` | 打印调试信息 |
| `spike_threshold_multiplier` | `100.0` | spike channel 判定阈值 = median × 此值 |
| `topk_channels` | `3` | Top-K 通道数（对应论文 Figure 1） |
| `absolute_thresholds` | `(10.0, 20.0, 30.0)` | 绝对幅度通道计数阈值 |
| `sparsity_epsilon` | `0.01` | post-norm sparsity 判定阈值 |
| `cosine_sample_pairs` | `256` | cosine stability 的采样对数 |
| `sample_layers` | `None` | 要监控的层索引列表，None=全部 |
| `log_logit_lens_entropy` | `False` | 是否开启 logit-lens 预测熵 + logsumexp 指标（开销大，默认关） |
| `log_logit_lens_cross_entropy` | `False` | 是否开启 logit-lens 逐层交叉熵指标（末层≈LM loss；与熵共用投影，默认关） |
| `logit_lens_chunk_size` | `1024` | logit-lens 投影按 token 分块的 tile 大小 |
| `logit_lens_apply_final_norm` | `True` | 投影前是否过 `decoder.final_layernorm` |
| `logit_lens_layers` | `None` | logit-lens 只监控的 global 层索引列表，None=持有 head 的全部层 |
| `log_hidden_spectral_entropy` | `False` | 是否开启 post-norm 隐状态谱熵/有效秩指标（默认关） |
