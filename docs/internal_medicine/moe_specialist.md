# MoE Health Monitor

`moe_health` 监控 Router 输入与决策、专家负载、门控质量、专家参数和
共享/路由专家激活。指标由模型实际暴露的能力决定，不支持的指标不会伪造。

## PaddleFleet 路由指标

设 Router 对一个 batch 中第 $t$ 个 token、专家 $e$ 的非负亲和度为
$p_{t,e}$，实际 Top-K 选择掩码为 $a_{t,e}\in\{0,1\}$，专家数为 $E$。

| 指标 | 定义 | 含义 |
|------|------|------|
| `router_input_rms` | $\sqrt{\operatorname{mean}(x^2)}$ | Router 输入整体尺度 |
| `router_input_abs_max` | $\max \lvert x\rvert$ | Router 输入峰值 |
| `router_input_abs_p99` | $P_{99}(\lvert x\rvert)$ | Router 输入尾部尺度 |
| `router_entropy` | $\operatorname{mean}_t[-\sum_e \bar p_{t,e}\log\bar p_{t,e}]$ | 每个 token 的路由亲和度熵 |
| `router_entropy_norm` | `router_entropy / log(E)` | 可跨专家数比较的归一化熵 |
| `score_sum_{mean,min,max}` | $\sum_{e\in\operatorname{TopK}(p_t)}p_{t,e}$ 的统计量 | Top-K 原始亲和度总量 |
| `router_margin_*` | $\min_{a=1}s_{t,e}-\max_{a=0}s_{t,e}$ | 已选/未选边界的稳健程度 |

`router_margin_*` 包含 `mean`、`min`、`p10` 和 `p01`。当 Router 使用
correction bias 时，$s$ 包含该 bias；group-limited 或 hash routing 的候选集合
不是简单全局 Top-K，因此不会输出可能误导的 margin。

## 负载与门控质量

Hard assignment 负载为

$$
n_e=\sum_t a_{t,e},\qquad q_e=\frac{n_e}{\sum_j n_j}.
$$

正门控质量为

$$
m_e=\sum_t p_{t,e}a_{t,e},\qquad r_e=\frac{m_e}{\sum_j m_j}.
$$

分别以 `assignment_load_` 和 `gate_mass_` 为前缀输出下列分布统计：

| 后缀 | 定义 | 健康方向 |
|------|------|----------|
| `cv` | $\operatorname{std}(q)/(1/E)$ | 越接近 0 越均衡 |
| `entropy_norm` | $-\sum_e q_e\log q_e/\log E$ | 越接近 1 越均衡 |
| `kl_uniform` | $\sum_e q_e\log(Eq_e)$ | 越接近 0 越均衡 |
| `max_frac` | $\max_e q_e$ | 识别最忙专家占比 |
| `min_frac` | $\min_e q_e$ | 识别冷专家或未使用专家 |
| `max_min_ratio` | $\max_e q_e/\max(\min_e q_e,10^{-12})$ | 识别负载极差 |

`gate_mass_*` 使用 $r$ 代替 $q$。它基于 Router 的正亲和度和实际选择掩码，
不读取可能包含后续可学习缩放的最终 combine weight，避免把负缩放误解释为
负的路由质量。

## 其他指标

- `bias_affinity_jaccard`：加入 correction bias 前后 Top-K 专家集合的 Jaccard。
- `expert_bias_{mean,std,min,max}`：correction bias 分布。
- `router_scalar_{mean,std,min,max,ratio}`：可学习 routed scaling 的分布。
- `expert_norm_{mean,std,min,max}`：路由专家参数 L2 范数分布。
- `shared_expert_norm`、`shared_routed_ratio`：共享专家参数范数及其相对尺度。
- `shared_act_*`、`routed_act_*`、`shared_routed_act_ratio`：共享/路由专家输出激活。

所有 forward-hook 指标先保留在设备端，沿统一的 `step()` 聚合与日志链路输出；
hook 中不引入额外的跨 rank collective。
