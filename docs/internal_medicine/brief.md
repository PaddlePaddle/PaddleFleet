# Internal Medicine — 模型训练健康监控

## 组会技术方案汇报

---

## 一、背景

大模型训练中，我们能直接观测到的信号非常有限：

- **Loss** — 唯一的优化目标，但它是所有问题的"最终症状"。当 loss 异常时，问题已经发生了数百步
- **Grad Norm** — 梯度粒度的健康指标，但只能告诉你"梯度炸了"，无法告诉你是哪个子模块、什么原因

这两个指标相当于"体温计"：能判断发不发烧，但无法做出诊断。

**我们缺少的是模型内部的"体检报告"** — 注意力分布是否正常？MoE 路由是否均衡？专家是否在萎缩？这些问题在 loss 曲线上往往看不出来，或者看出来时已经太晚了。

---

## 二、目标

构建一套训练时的模型内部健康监控系统，覆盖两个核心维度：

| 维度 | 关注点 | 典型问题 |
|------|--------|----------|
| **注意力健康** | Q·K 注意力矩阵的统计特征 | Logit 爆炸、注意力坍缩、Attention Sink |
| **MoE 专家健康** | 路由决策、专家权重、负载均衡 | 路由坍缩、专家萎缩/过载、负载均衡冲突 |

设计原则：
- 零侵入 — 通过 PyTorch forward hook 采集，不修改模型代码、不影响梯度
- 低开销 — hook 内计算在 `torch.no_grad()` 下执行，按 `monitor_interval` 采样
- 一行配置 — YAML 中 `internal_medicine_monitors: ["qk_stats", "moe_health"]` 即可启用

---

## 三、QK Stats — 注意力健康监控

> 图: [`qk_stats_metrics.excalidraw`](./qk_stats_metrics.excalidraw)

图中展示了 7 个指标分三组的全景。以下是补充说明：

**Triton Online Softmax** — QK Stats 的核心工程亮点。传统计算需要先实例化完整的 `[B, H, S, S]` 注意力矩阵（S=8192 时约 2GB），我们的 Triton 内核采用 Online Softmax 算法，在单次遍历中同时计算 max/mean/entropy/sink 四个统计量，显存开销从 O(S²) 降到 O(S)。

**entropy 分布统计** — 我们不记录每个 head 的完整 entropy，而是用 min/max/std 三个分布统计量来捕捉异常。这样既控制了日志量，又能检测到"某个 head 坍缩"或"head 间行为分化"等问题。

**Attention Sink** — LLM 训练的已知现象：模型倾向于将大量注意力分配给第一个 token，作为一种"垃圾桶"机制。sink 指标追踪这个现象的程度，过度 sink 表明模型在注意力资源分配上存在浪费。

---

## 四、MoE Specialist — 专家系统健康监控

> 图: [`moe_health_metrics.excalidraw`](./moe_health_metrics.excalidraw)

图中展示了 13 个指标分三组（路由健康 / 负载均衡 / 专家权重），以及两个关键阈值体系。以下是补充说明：

**bias_affinity_jaccard ★** — 最重要的 MoE 诊断指标。它度量的是：负载均衡机制（Expert Bias）对路由决策的"篡改"程度。低 Jaccard 意味着 Router 学到的"最优路由"和负载均衡的"均匀分配"产生了严重冲突 — 模型质量和训练效率之间被迫做出妥协。

**shared_routed_ratio** — 监控 MoE 的架构有效性。共享专家是"所有 token 都经过"的通路，路由专家是"按需选择"的通路。如果 ratio > 3.0，说明共享专家主导了输出，路由专家形同虚设，MoE 退化为 Dense 模型。

**expert_norm** — 专家的"体重"。健康的 MoE 中各专家权重范数应大致均衡。如果某个专家 norm 极小（萎缩 Atrophy），说明它很少被路由到，权重趋近于零；如果极大（过载 Inflammation），说明它承担了不成比例的 token 量。

---

## 五、实现方案

> 架构图: [`internal_medicine_architecture.excalidraw`](./internal_medicine_architecture.excalidraw)

整体数据流：

```
Forward Pass
  │
  ├─ core_attention pre-hook ──► Triton QK Stats ──┐
  │                                                 │
  ├─ router post-hook ──► Router Metrics ───────────┤
  │                                                 ├──► training_logs (singleton)
  ├─ moe_layer post-hook ──► Expert Metrics ────────┤         │
  │                                                 │    on_step_end (每 log_interval 步)
  └─ (forward 正常继续，无任何影响)                    │         │
                                                    │    gather_and_aggregate()
                                                    │         │
                                                    │    ┌────┴────┐
                                                    │    │ stdout  │ TensorBoard │
                                                    │    └─────────┴─────────────┘
```

关键设计点：

1. **Hook 注册时机** — 在 DDP/FSDP 包装之前（`register_pre_wrap_hook`），可以访问原始模型结构
2. **Activation Checkpointing 兼容** — hook 内 `if not torch.is_grad_enabled(): return` 跳过 recompute
3. **PP 感知** — 层编号为全局索引 `global_idx = pp_rank × local_layers + local_idx`
4. **通信开销** — QK 在 TP group 内做小量 all_reduce (~100B/层)；全局聚合走 `all_gather_object` (Gloo/CPU) 按 `log_interval` 频率触发
