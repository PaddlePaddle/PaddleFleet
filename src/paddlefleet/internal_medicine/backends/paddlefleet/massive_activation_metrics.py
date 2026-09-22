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
Massive Activation Metrics for PaddleFleet.

Monitors massive activations (extreme outlier values in hidden state channels)
as characterized by Sun et al. (2026) arXiv:2603.05498.

Core metrics:
1. Channel Max — absolute maximum activation across all channels
2. Channel Median/P95/P99 — distribution of per-channel peak magnitudes
3. Channel Max Ratio — ratio of max channel to median channel (outlier severity)
4. Channel Counts over absolute thresholds — broad residual-scale growth
5. Massive Activation Channel Count — channels over median-relative threshold
6. Top-K Channel Norm — L2 norm of the top-K largest channels
7. Activation RMS — residual-stream scale
8. Post-Norm Sparsity — fraction of near-zero entries after RMSNorm
9. Post-Norm Cosine Stability — cosine similarity of normalized representations
"""

import paddle

DEFAULT_ABSOLUTE_THRESHOLDS = (10.0, 20.0, 30.0)


def _threshold_key(threshold: float) -> str:
    text = f"{threshold:g}"
    return text.replace("-", "neg").replace(".", "p")


def compute_per_channel_max(hidden_states: paddle.Tensor) -> paddle.Tensor:
    """Per-channel maximum absolute activation for [..., H] hidden states."""
    h = hidden_states.reshape([-1, hidden_states.shape[-1]]).astype("float32")
    return h.abs().max(axis=0)


def compute_activation_scale_stats(
    hidden_states: paddle.Tensor,
) -> dict[str, paddle.Tensor]:
    """Absolute scale statistics over the full residual stream. Returns GPU tensors."""
    h = hidden_states.reshape([-1, hidden_states.shape[-1]]).astype("float32")
    return {
        "activation_rms": paddle.sqrt(paddle.square(h).mean()),
    }


def _nearest_quantile_from_sorted(
    sorted_values: paddle.Tensor, q: float
) -> paddle.Tensor:
    idx = min(
        max(round((sorted_values.shape[0] - 1) * q), 0),
        sorted_values.shape[0] - 1,
    )
    return sorted_values[idx]


def summarize_per_channel_max(
    per_channel_max: paddle.Tensor,
    threshold_multiplier: float = 100.0,
    k: int = 3,
    absolute_thresholds: tuple[float, ...] = DEFAULT_ABSOLUTE_THRESHOLDS,
) -> dict[str, paddle.Tensor]:
    """Derive scalar massive-activation metrics from per-channel maxima. Returns GPU tensors."""
    channel_max = per_channel_max.max()
    channel_median = paddle.median(per_channel_max)
    channel_max_ratio = channel_max / paddle.clip(channel_median, min=1e-8)

    threshold = channel_median * threshold_multiplier
    massive_act_channel_count = (
        (per_channel_max > threshold).astype("float32").sum()
    )

    topk_vals, _ = paddle.topk(
        per_channel_max, min(k, per_channel_max.shape[0])
    )
    topk_channel_norm = topk_vals.norm()
    sorted_channel_max = paddle.sort(per_channel_max)

    result = {
        "channel_max": channel_max,
        "channel_median": channel_median,
        "channel_p95": _nearest_quantile_from_sorted(sorted_channel_max, 0.95),
        "channel_p99": _nearest_quantile_from_sorted(sorted_channel_max, 0.99),
        "channel_max_ratio": channel_max_ratio,
        "massive_act_channel_count": massive_act_channel_count,
        "topk_channel_norm": topk_channel_norm,
    }
    for absolute_threshold in absolute_thresholds:
        key = f"channel_count_gt_{_threshold_key(absolute_threshold)}"
        result[key] = (
            (per_channel_max > absolute_threshold).astype("float32").sum()
        )
    return result


def compute_pre_norm_metrics(
    hidden_states: paddle.Tensor,
    threshold_multiplier: float = 100.0,
    k: int = 3,
    absolute_thresholds: tuple[float, ...] = DEFAULT_ABSOLUTE_THRESHOLDS,
) -> dict[str, paddle.Tensor]:
    """All pre-norm massive activation metrics in one pass.

    Computes per_channel_max once and derives all dependent metrics.

    Args:
        hidden_states: [..., H] post-residual hidden states.
        threshold_multiplier: multiplier on median to define "spike" threshold.
        k: number of top channels for top-K norm.

    Returns:
        Dict with per-channel peak, outlier, absolute-threshold, and top-K metrics (GPU tensors).
    """
    return summarize_per_channel_max(
        compute_per_channel_max(hidden_states),
        threshold_multiplier=threshold_multiplier,
        k=k,
        absolute_thresholds=absolute_thresholds,
    )


def compute_post_norm_sparsity(
    normalized_states: paddle.Tensor,
    epsilon: float = 0.01,
) -> paddle.Tensor:
    """Fraction of near-zero entries in post-RMSNorm hidden states. Returns GPU 0-dim tensor."""
    h = normalized_states.reshape([-1]).astype("float32")
    return (h.abs() < epsilon).astype("float32").mean()


def compute_post_norm_cosine_stability(
    normalized_states: paddle.Tensor,
    num_sample_pairs: int = 256,
) -> paddle.Tensor:
    """Cosine similarity among token representations after normalization. Returns GPU 0-dim tensor."""
    h = normalized_states.reshape([-1, normalized_states.shape[-1]]).astype(
        "float32"
    )
    num_tokens = h.shape[0]

    if num_tokens < 2:
        return paddle.ones(())

    n_pairs = min(num_sample_pairs, num_tokens * (num_tokens - 1) // 2)
    idx_a = paddle.randint(0, num_tokens, [n_pairs])
    idx_b = paddle.randint(0, num_tokens - 1, [n_pairs])
    idx_b = idx_b + (idx_b >= idx_a).astype("int64")

    vec_a = h[idx_a]
    vec_b = h[idx_b]

    cosine = paddle.nn.functional.cosine_similarity(vec_a, vec_b, axis=-1)
    return cosine.mean()
