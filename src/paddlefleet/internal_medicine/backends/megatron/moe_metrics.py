# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MoE Health Metrics Computation Functions (Simplified).

Core metrics:
1. Router Entropy - 路由熵
2. Router Score Sum - TopK选择的score求和
3. Bias-Affinity Correlation - 使用Jaccard相似度
4. Expert Norms - 专家权重L2范数
5. Shared/Routed Ratio - Shared Expert与Routed Expert的比例

注意: 所有指标只计算本地值，EP 聚合在 training_logs 层面统一处理
(通过 gather_object_list 收集所有卡的结果，然后按 mean/max/min 聚合)
"""

import torch


def compute_router_entropy(router_probs: torch.Tensor) -> torch.Tensor:
    """
    计算 Router Entropy (路由熵).

    H = -sum(p * log(p))

    Args:
        router_probs: Softmax probabilities [tokens, num_experts] (scores_for_aux_loss)

    Returns:
        entropy_mean: 本地 batch 的平均熵 (Scalar Tensor)
    """
    probs = router_probs.float().clamp(min=1e-10)
    entropy = -(probs * probs.log()).sum(dim=-1)  # [tokens]
    return entropy.mean()


def compute_topk_score_sum(
    scores: torch.Tensor, topk: int
) -> dict[str, torch.Tensor]:
    """
    计算 Router TopK Score Sum.

    Args:
        scores: Router scores [tokens, num_experts]，softmax/sigmoid 后的分数
        topk: TopK 值

    Returns:
        Dict with score_sum statistics
    """
    topk_scores, _ = scores.float().topk(topk, dim=-1)  # [tokens, topk]
    score_sum = topk_scores.sum(dim=-1)  # [tokens]
    return {
        "score_sum_mean": score_sum.mean(),
        "score_sum_min": score_sum.min(),
        "score_sum_max": score_sum.max(),
    }


def compute_bias_affinity_jaccard(
    routing_map_before_bias: torch.Tensor,
    routing_map_after_bias: torch.Tensor,
    num_experts: int | None = None,
) -> torch.Tensor:
    """
    计算 Bias-Affinity Jaccard 相似度.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|

    Args:
        routing_map_before_bias: Bias前的routing map [tokens, topk] (index form)
            or [tokens, num_experts] (one-hot/bool form).
        routing_map_after_bias: Bias后的routing map.
        num_experts: required when routing_map is in index form (last dim == topk).
            Pass router.num_experts. Avoids a D2H sync on the hot path.

    Returns:
        jaccard: Jaccard相似度 (0-1)
    """
    if routing_map_before_bias.dim() == 2:
        is_onehot = routing_map_before_bias.shape[
            -1
        ] != 1 and routing_map_before_bias.dtype in (
            torch.bool,
            torch.uint8,
        )

        if (
            not is_onehot
            and num_experts is not None
            and routing_map_before_bias.dtype not in (torch.bool, torch.uint8)
        ):
            num_tokens, topk = routing_map_before_bias.shape
            before_onehot = torch.zeros(
                num_tokens,
                num_experts,
                device=routing_map_before_bias.device,
                dtype=torch.bool,
            )
            after_onehot = torch.zeros(
                num_tokens,
                num_experts,
                device=routing_map_after_bias.device,
                dtype=torch.bool,
            )

            for k in range(topk):
                before_onehot.scatter_(
                    1, routing_map_before_bias[:, k : k + 1].long(), True
                )
                after_onehot.scatter_(
                    1, routing_map_after_bias[:, k : k + 1].long(), True
                )

            routing_map_before_bias = before_onehot
            routing_map_after_bias = after_onehot

    before = routing_map_before_bias.bool()
    after = routing_map_after_bias.bool()

    intersection = (before & after).float().sum()
    union = (before | after).float().sum()
    return intersection / union.clamp(min=1e-8)


def compute_load_balance_ratios(
    tokens_per_expert: torch.Tensor,
) -> dict[str, torch.Tensor] | None:
    """专家负载极值比值, 基于已 all-reduce 的 tokens-per-expert.

    输入是 mcore 在 get_updated_expert_bias 里跨 TPxCPxDP all-reduce 之后的
    全局 per-expert token 计数 (每张卡上都相同), 因此这里算出的比值本身就是全局
    正确值, 无需 monitor 侧 collective, 也无需 training_logs 跨 rank 聚合.

    对每一层 (行) 计算:
      - load_max_min_ratio    = #tokens(most-routed) / #tokens(least-routed)
      - load_max_median_ratio = #tokens(most-routed) / #tokens(median expert)
      - load_cv               = std(counts) / mean(counts)  (变异系数, 完全均衡=0)
      - load_balance_entropy_norm = H(p) / log(E)  (归一化负载熵, 完全均衡=1, 坍缩到
        单专家=0). p_e = count_e / sum(count) 是本 batch 的专家负载分布, H = -sum(p log p).
      - load_effective_experts = exp(H)  (有效专家数 / perplexity, 完全均衡=E, 坍缩=1).
        比裸熵更好读: "256 个专家实际只有效用到了 40 个".

    熵与 CV 在接近均衡区间数学上近似 (H/log(E) ≈ 1 - CV²/(2 log E)), 但熵有界 [0,1],
    可跨专家数配置比较, 且 exp(H) 直接给出"有效专家数". 熵对分布做全体求和, 不像
    max/min 比值被单个死专家主导, 更平滑. 三者都 scale-invariant (与总 token 数无关),
    因此无需除 ga_steps.

    median 用 torch.median (偶数个专家时取下中位, 即某个真实专家的计数, 不做插值).
    极值比值分母 clamp(min=1.0) 防止死专家 (count=0) 产生 inf/NaN. CV 用 population
    std (unbiased=False) 除以 mean, mean clamp(min=1.0) 防止空批次除零. 熵用 p*log(p)
    并对 p clamp(min=1e-12) 防止 0*log0 的 NaN (死专家贡献 0, 是熵的正确极限). 全程 GPU
    tensor, 不在 hot path 上运行 (调用点在 finalize_model_grads, forward hook 之外).

    Args:
        tokens_per_expert: reduced counts. [num_layers, num_experts] (stacked)
            or [num_experts] (single layer). Last dim is the expert axis.

    Returns:
        Dict of per-layer 0-dim/1-dim tensors keyed by metric name (shape matches
        the leading layer dim), or None if the expert axis has < 2 experts.
    """
    counts = tokens_per_expert.to(torch.float32)
    if counts.dim() == 1:
        counts = counts.unsqueeze(0)
    if counts.dim() != 2 or counts.shape[-1] < 2:
        return None

    max_count = counts.max(dim=-1).values
    min_count = counts.min(dim=-1).values
    median_count = counts.median(dim=-1).values
    # Population std (unbiased=False): with the global per-expert totals this is
    # the exact CV, not a sample estimate. Balanced load -> std=0 -> CV=0.
    mean_count = counts.mean(dim=-1)
    std_count = counts.std(dim=-1, unbiased=False)

    # Load distribution p_e = count_e / sum(count), then Shannon entropy per layer.
    # clamp the total (not per-expert) to avoid divide-by-zero on an empty batch;
    # clamp p only inside log to make dead experts (p=0) contribute 0*log(~0)->0,
    # the correct 0*log0 limit, without NaN. num_experts drives the log(E) norm.
    num_experts = counts.shape[-1]
    total_count = counts.sum(dim=-1, keepdim=True).clamp(min=1.0)
    probs = counts / total_count
    entropy = -(probs * probs.clamp(min=1e-12).log()).sum(
        dim=-1
    )  # [num_layers]
    log_e = torch.log(torch.tensor(float(num_experts), device=counts.device))

    return {
        "load_max_min_ratio": max_count / min_count.clamp(min=1.0),
        "load_max_median_ratio": max_count / median_count.clamp(min=1.0),
        "load_cv": std_count / mean_count.clamp(min=1.0),
        "load_balance_entropy_norm": entropy / log_e,
        "load_effective_experts": torch.exp(entropy),
    }


def compute_expert_norms(
    expert_weights: list[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    计算 Expert Norms (专家权重L2范数).

    Args:
        expert_weights: 每个本地专家的权重张量列表

    Returns:
        Dict with norm statistics (本地专家)
    """
    if not expert_weights:
        return {
            "expert_norm_mean": torch.tensor(0.0),
            "expert_norm_std": torch.tensor(0.0),
            "expert_norm_min": torch.tensor(0.0),
            "expert_norm_max": torch.tensor(0.0),
        }

    norms = torch.stack([w.float().norm() for w in expert_weights])
    return {
        "expert_norm_mean": norms.mean(),
        "expert_norm_std": norms.std()
        if norms.numel() > 1
        else torch.tensor(0.0, device=norms.device),
        "expert_norm_min": norms.min(),
        "expert_norm_max": norms.max(),
    }


def compute_shared_expert_norm(
    shared_expert_weights: list[torch.Tensor],
) -> torch.Tensor:
    """
    计算 SharedExpert 的 L2 Norm.

    Args:
        shared_expert_weights: Shared Expert 的权重张量列表

    Returns:
        shared_norm: L2 Norm
    """
    if not shared_expert_weights:
        return torch.tensor(0.0)

    all_params = torch.cat([w.flatten() for w in shared_expert_weights])
    return all_params.float().norm()


def compute_shared_routed_ratio(
    shared_norm: torch.Tensor,
    routed_norm_mean: torch.Tensor,
) -> torch.Tensor:
    """
    计算 Shared/Routed Ratio.

    Args:
        shared_norm: SharedExpert 的 L2 Norm
        routed_norm_mean: Routed Experts 的平均 L2 Norm

    Returns:
        ratio: Shared/Routed Ratio
    """
    return shared_norm / routed_norm_mean.clamp(min=1e-8)
