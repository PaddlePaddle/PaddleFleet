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
Massive Activation Metrics Computation Functions.

Monitors massive activations (extreme outlier values in hidden state channels)
as characterized by Sun et al. (2026) "The Spike, the Sparse and the Sink:
Anatomy of Massive Activations and Attention Sinks" (arXiv:2603.05498).

Core metrics:
1. Channel Max — absolute maximum activation across all channels
2. Channel Median/P95/P99 — distribution of per-channel peak magnitudes
3. Channel Max Ratio — ratio of max channel to median channel (outlier severity)
4. Channel Counts over absolute thresholds — broad residual-scale growth
5. Massive Activation Channel Count — channels over median-relative threshold
6. Top-K Channel Norm — L2 norm of the top-K largest channels
7. Activation RMS — residual-stream scale
8. Post-Norm Sparsity — fraction of near-zero entries after RMSNorm (sparsification)
9. Post-Norm Cosine Stability — cosine similarity of normalized representations
   across tokens (near-constant vector detection)
10. Spectral Norm Bounds — per-token residual gain ratio (post_layer_rms /
    pre_layer_rms). The max ratio over the global batch lower-bounds the layer's
    spectral norm (largest singular value); the min ratio upper-bounds its
    smallest singular value.
11. Lipschitz / Gradient-Gain Bounds — per-token backward gradient-gain ratio
    (‖∂L/∂x‖ / ‖∂L/∂y‖). Since backprop gives ∂L/∂x = Jᵀ·∂L/∂y, this is the gain
    of the layer Jacobian's transpose on the observed gradient; the max over the
    global batch lower-bounds σ_max(J) (the layer's Lipschitz constant) and the
    min upper-bounds σ_min(J). Captured in the backward pass (see the monitor).
12. Logit-Lens Entropy + Logsumexp — per-layer predictive entropy H(p) and
    log-partition logsumexp of the residual projected through the LM head (opt-in).
13. Hidden Spectral Entropy — matrix (von Neumann) entropy of the post-RMSNorm
    hidden spectrum (effective rank); a representation-diversity / rank-collapse
    signal computed via eigvalsh of the Gram matrix, no LM head (opt-in).

All metrics compute local values only; cross-rank aggregation is handled
by training_logs.gather_and_aggregate().

Reference:
    Sun, S., Canziani, A., LeCun, Y., & Zhu, J. (2026).
    The Spike, the Sparse and the Sink: Anatomy of Massive Activations
    and Attention Sinks. arXiv:2603.05498.
"""

import torch

DEFAULT_ABSOLUTE_THRESHOLDS = (10.0, 20.0, 30.0)


def _threshold_key(threshold: float) -> str:
    text = f"{threshold:g}"
    return text.replace("-", "neg").replace(".", "p")


def compute_per_channel_max(hidden_states: torch.Tensor) -> torch.Tensor:
    """Per-channel maximum absolute activation for [..., H] hidden states."""
    h = hidden_states.reshape(-1, hidden_states.shape[-1]).float()
    return h.abs().max(dim=0).values


def compute_activation_scale_stats(
    hidden_states: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Absolute scale statistics over the full residual stream."""
    h = hidden_states.reshape(-1, hidden_states.shape[-1]).float()
    return {
        "activation_rms": h.square().mean().sqrt(),
    }


def compute_spectral_norm_bounds(
    pre_hidden: torch.Tensor,
    post_hidden: torch.Tensor,
    eps: float = 1e-8,
    include_activation_rms: bool = False,
) -> dict[str, torch.Tensor]:
    """Per-token residual gain ratio (post_layer_rms / pre_layer_rms) reduced to
    max/min bounds on the layer's spectral norm.

    For each token, ``rms(y_t) / rms(x_t) == ||y_t|| / ||x_t||`` (the ``sqrt(H)``
    cancels), the gain of the residual block on that token. Over all tokens:

    - ``spectral_norm_max`` (max ratio) lower-bounds the layer's largest singular
      value (spectral norm): every observed gain is <= sup_x ||f(x)||/||x||.
    - ``spectral_norm_min`` (min ratio) upper-bounds the smallest singular value:
      every observed gain is >= inf_x ||f(x)||/||x||.

    Both inputs are the full-H residual ([S, B, H] or [B, S, H]); the RMS is a true
    full-vector RMS, so no TP channel reduction is needed. The max/min compose
    across token-partitioned ranks via ``training_logs.gather_and_aggregate``.

    When ``include_activation_rms`` is set, the input residual's global RMS
    (``activation_rms``) is derived from the same per-token ``pre_rms`` for free
    (``activation_rms == sqrt(mean_t(pre_rms_t**2))``), so callers that already run
    this function need not recompute it in a separate pass. ``activation_rms_std``
    is the std of ``pre_rms`` across tokens (per-token RMS dispersion).

    Returns 0-dim GPU tensors (no host sync).
    """
    pre = pre_hidden.reshape(-1, pre_hidden.shape[-1]).float()
    post = post_hidden.reshape(-1, post_hidden.shape[-1]).float()
    pre_rms = pre.square().mean(dim=-1).sqrt()
    post_rms = post.square().mean(dim=-1).sqrt()
    ratio = post_rms / pre_rms.clamp(min=eps)
    metrics = {
        "spectral_norm_max": ratio.max(),
        "spectral_norm_min": ratio.min(),
    }
    if include_activation_rms:
        metrics["activation_rms"] = pre_rms.square().mean().sqrt()
        metrics["activation_rms_std"] = pre_rms.std()
    return metrics


def compute_grad_gain_bounds(
    grad_in: torch.Tensor,
    grad_out: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Per-token backward gradient-gain ratio (‖∂L/∂x‖ / ‖∂L/∂y‖) reduced to
    max/min bounds on the layer's Jacobian singular values (its Lipschitz constant).

    ``grad_in`` is the gradient of the loss w.r.t. the layer INPUT hidden states
    (∂L/∂x); ``grad_out`` is the gradient w.r.t. the layer OUTPUT hidden states
    (∂L/∂y). Backprop through the layer map ``f: x -> y`` gives ``∂L/∂x = Jᵀ·∂L/∂y``
    with ``J = ∂y/∂x``, so per token

        ``rms(grad_in_t) / rms(grad_out_t) == ‖∂L/∂x_t‖ / ‖∂L/∂y_t‖ == ‖Jᵀ dy_t‖ / ‖dy_t‖``

    (the ``sqrt(H)`` cancels), the gain of ``Jᵀ`` on the observed gradient. Over all
    tokens, and since ``J`` and ``Jᵀ`` share singular values:

    - ``lipschitz_max`` (max ratio) lower-bounds ``σ_max(J)`` — the layer's spectral
      norm, i.e. its Lipschitz constant: every observed gain is <= sup_v ‖Jᵀv‖/‖v‖.
    - ``lipschitz_min`` (min ratio) upper-bounds ``σ_min(J)``: every observed gain is
      >= inf_v ‖Jᵀv‖/‖v‖.

    It also reads directly as the layer's gradient amplification in backprop:
    ``lipschitz_max > 1`` => gradients grow passing backward through the layer
    (explosion risk); ``<< 1`` => they shrink (vanishing risk).

    Both inputs are the full-H hidden-state gradient ([S, B, H] or [B, S, H]); the RMS
    is a true full-vector RMS, so no TP channel reduction is needed. The max/min compose
    across token-partitioned ranks via ``training_logs.gather_and_aggregate``.

    The per-token decomposition ignores the sequence coupling introduced by attention
    (the true Jacobian mixes tokens); it is the same simplification the forward
    ``compute_spectral_norm_bounds`` metric makes.

    Returns 0-dim GPU tensors (no host sync).
    """
    gin = grad_in.reshape(-1, grad_in.shape[-1]).float()
    gout = grad_out.reshape(-1, grad_out.shape[-1]).float()
    in_rms = gin.square().mean(dim=-1).sqrt()
    out_rms = gout.square().mean(dim=-1).sqrt()
    ratio = in_rms / out_rms.clamp(min=eps)
    return {
        "lipschitz_max": ratio.max(),
        "lipschitz_min": ratio.min(),
    }


@torch.no_grad()
def compute_logit_lens_entropy(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    final_norm=None,
    chunk_size: int = 1024,
    tp_size: int = 1,
    labels: torch.Tensor | None = None,
    want_entropy: bool = True,
) -> dict[str, torch.Tensor]:
    """Per-token predictive entropy H(p) + logsumexp (+ optional cross-entropy) via the
    *logit lens*, chunked over tokens.

    Projects each token's residual through the LM head to vocab logits and returns,
    as the token-mean (only the mean is reported):

    - the softmax entropy ``H(p) = -Σ p·log p`` — "how committed is this layer to a
      next-token prediction" (entropy falls with depth); values in ``[0, log(vocab)]``.
      (Emitted only when ``want_entropy``.)
    - the log-partition ``log_z = logsumexp(l) = log Σ exp(l_v)`` — the softmax
      normalizer / a soft-max over the logits; it tracks the raw logit scale the layer
      has built up under the lens (it upper-bounds and closely tracks the max logit).
      (Emitted only when ``want_entropy``.)
    - the per-token cross-entropy ``CE = log_z − l[label]`` against the ground-truth
      next-token ``label`` — the logit lens applied as a loss. The final layer's CE
      equals the LM loss up to loss-mask weighting (the mask is applied outside the
      model, so this token-mean is unweighted). Emitted only when ``labels`` is given.

    Chunked so the full ``[tokens, vocab]`` logits are never materialized: one
    ``[chunk_size, vocab]`` tile at a time, accumulating only running scalar sums.
    Entropy is written as ``H = log_z − E_p[l]`` with ``log_z = logsumexp(l)`` and
    ``E_p[l] = Σ softmax(l)·l``, using ``torch.logsumexp`` / ``torch.softmax`` directly
    (no hand-rolled exp/log). ``log_z`` is reused for the logsumexp metric for free.

    Vocab-parallel TP is NOT supported yet (asserted off): a TP-sharded vocab would need
    the softmax normalizer reduced across ranks (a MAX + SUM all-reduce that can't fuse
    with ``torch.logsumexp``). The caller only attaches this metric on the head-owning
    stage and assumes the head is vocab-unsharded.

    Args:
        hidden: ``[s, b, h]`` layer output residual (caller passes it detached).
        lm_head_weight: LM-head weight ``[vocab, h]`` (``.detach()``ed here).
        final_norm: norm applied before projection (LM head is trained on final-normed
            states); ``None`` projects the raw residual.
        chunk_size: tokens per projection tile.
        tp_size: tensor-parallel world size; must be 1 (asserted — no TP support yet).
        labels: ``[num_tokens]`` ground-truth next-token ids, already aligned to
            ``hidden.reshape(-1, h)`` row order (seq-major). When given, the token-mean
            cross-entropy is added to the result. Must match the token count or it is
            ignored (defensive: no crash on a shape mismatch).
        want_entropy: when ``False``, skip the entropy/logsumexp metrics (used when only
            the cross-entropy is requested) — the projection is shared either way.

    Runs under ``@torch.no_grad()`` and detaches ``hidden`` / ``lm_head_weight``, so it
    never builds autograd graph or retains references into the training forward's graph.
    Returns 0-dim GPU tensors (no host sync).
    """
    assert tp_size <= 1, (
        "compute_logit_lens_entropy does not support vocab-parallel TP (tp_size > 1) yet"
    )

    hidden_dim = hidden.shape[-1]
    # Detach + @torch.no_grad() above: this is a monitoring probe, it must never
    # build autograd graph or hold references into the training forward's graph.
    h = hidden.detach().reshape(-1, hidden_dim)
    if final_norm is not None:
        h = final_norm(h)
    num_tokens = h.shape[0]

    # Cross-entropy needs labels aligned 1:1 with the flattened tokens. A mismatch means
    # the caller couldn't align them (unexpected orientation / CP mismatch) — drop CE
    # rather than gather garbage. .numel() is a Python-int attribute, no host sync.
    want_ce = labels is not None and labels.numel() == num_tokens
    labels_flat = (
        labels.detach().reshape(-1).long().clamp_min(0) if want_ce else None
    )

    if num_tokens == 0:
        z = hidden.new_zeros(())
        out: dict[str, torch.Tensor] = {}
        if want_entropy:
            out["logit_lens_entropy_mean"] = z
            out["logit_lens_logsumexp_mean"] = z
        if want_ce:
            out["logit_lens_cross_entropy_mean"] = z
        return out

    weight = lm_head_weight.detach()  # [vocab, h]

    # Only the token-mean is reported, so accumulate running sums (a scalar each)
    # instead of retaining per-token vectors.
    entropy_sum = h.new_zeros((), dtype=torch.float32)
    logsumexp_sum = h.new_zeros((), dtype=torch.float32)
    ce_sum = h.new_zeros((), dtype=torch.float32)
    for start in range(0, num_tokens, chunk_size):
        h_chunk = h[start : start + chunk_size]
        # Matmul in the weight dtype (cheap); reduce in fp32 (stable).
        logits = torch.matmul(h_chunk, weight.t()).float()  # [t, vocab]
        # log_z is shared by entropy, logsumexp, and cross-entropy — one logsumexp.
        log_z = torch.logsumexp(logits, dim=-1)  # [t]
        if want_entropy:
            # Fused, numerically-stable primitives (no hand-rolled exp on shifted
            # logits, no separate log). H = log_z − E_p[l], E_p[l] = Σ softmax(l)·l.
            probs = torch.softmax(logits, dim=-1)  # [t, vocab]
            # In-place weight into the `probs` buffer (safe: under @torch.no_grad) so the
            # `probs·logits` product reuses it instead of allocating a third [t, vocab] tile.
            probs.mul_(logits)  # probs now holds softmax(l)·l
            entropy = log_z - probs.sum(dim=-1)  # [t]
            entropy_sum += entropy.sum()
            logsumexp_sum += (
                log_z.sum()
            )  # reuse log_z — the logsumexp metric is free
        if want_ce:
            label_chunk = labels_flat[start : start + chunk_size]
            # CE_t = log_z_t − l_t[label_t] = -log softmax(l_t)[label_t]. Reuses log_z.
            tgt = logits.gather(1, label_chunk.unsqueeze(1)).squeeze(1)  # [t]
            ce_sum += (log_z - tgt).sum()

    out = {}
    if want_entropy:
        out["logit_lens_entropy_mean"] = entropy_sum / num_tokens
        out["logit_lens_logsumexp_mean"] = logsumexp_sum / num_tokens
    if want_ce:
        out["logit_lens_cross_entropy_mean"] = ce_sum / num_tokens
    return out


@torch.no_grad()
def compute_hidden_spectral_entropy(
    hidden: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Matrix (von Neumann) entropy of the hidden-state spectrum — a representation
    diversity / rank-collapse signal, computed WITHOUT the LM head.

    For ``hidden`` reshaped to ``[n, d]`` (n tokens, d channels) with singular values
    ``σ_i``, let ``p_i = σ_i² / Σ_k σ_k² = σ_i² / ‖h‖_F²`` (a valid distribution, since
    ``‖h‖_F² = Σ σ_i²``). The entropy ``H = -Σ p_i log p_i ∈ [0, log(min(n, d))]`` measures
    how many effective directions the token set spans: low H = collapsed onto a few
    directions, high H = spread across many (``exp(H)`` is the effective rank).

    Computed without a full SVD: the ``σ_i²`` are the eigenvalues of the smaller Gram
    matrix (``hᵀh`` when ``n >= d`` else ``h hᵀ``), via ``torch.linalg.eigvalsh`` — a single
    GPU call, no host sync. Tiny negative eigenvalues (numerical) are clamped to 0.

    Intended to run on the POST-RMSNorm hidden states (the caller passes the layer's
    ``input_layernorm`` output), so the per-token RMS scaling is already removed.

    This is a SET-LEVEL, nonlinear quantity (not per-token): unlike the per-token mean
    metrics, its cross-rank / cross-microbatch mean is only an APPROXIMATION of the global
    spectral entropy (mean-of-per-shard-entropies != entropy-over-all-tokens). It is
    reported as a per-shard value averaged at flush time, which is fine as a collapse trend
    signal (accepted by design).

    Returns a 0-dim GPU tensor (no host sync).
    """
    d = hidden.shape[-1]
    h = hidden.reshape(-1, d).float()
    n = h.shape[0]
    if n == 0:
        return hidden.new_zeros(())
    # Eigenvalues of the smaller Gram matrix are exactly σ_i²; pick min(n, d) side so the
    # eigvalsh cost is O(min(n, d)³) rather than a full [n, d] SVD.
    gram = h.t() @ h if n >= d else h @ h.t()
    evals = torch.linalg.eigvalsh(gram).clamp_min(
        0
    )  # ascending, real; no host sync
    p = evals / evals.sum().clamp_min(eps)  # p_i = σ_i² / ‖h‖_F²
    # 0·log0 -> 0: p already >= 0, clamp only inside the log (multiplied-out zeros vanish).
    return -(p * p.clamp_min(eps).log()).sum()


def _nearest_quantile_from_sorted(
    sorted_values: torch.Tensor, q: float
) -> torch.Tensor:
    idx = round((sorted_values.shape[0] - 1) * q)
    idx = min(max(idx, 0), sorted_values.shape[0] - 1)
    return sorted_values[idx]


def summarize_per_channel_max(
    per_channel_max: torch.Tensor,
    threshold_multiplier: float = 100.0,
    k: int = 3,
    absolute_thresholds: tuple[float, ...] = DEFAULT_ABSOLUTE_THRESHOLDS,
) -> dict[str, torch.Tensor]:
    """Derive scalar massive-activation metrics from per-channel maxima."""
    channel_max = per_channel_max.max()
    channel_median = per_channel_max.median()
    channel_max_ratio = channel_max / channel_median.clamp(min=1e-8)
    threshold = channel_median * threshold_multiplier
    topk_vals = per_channel_max.topk(min(k, per_channel_max.shape[0])).values
    sorted_channel_max = per_channel_max.sort().values

    metrics = {
        "channel_max": channel_max,
        "channel_median": channel_median,
        "channel_p95": _nearest_quantile_from_sorted(sorted_channel_max, 0.95),
        "channel_p99": _nearest_quantile_from_sorted(sorted_channel_max, 0.99),
        "channel_max_ratio": channel_max_ratio,
        "massive_act_channel_count": (per_channel_max > threshold)
        .sum()
        .float(),
        "topk_channel_norm": topk_vals.norm(),
    }
    for absolute_threshold in absolute_thresholds:
        metrics[f"channel_count_gt_{_threshold_key(absolute_threshold)}"] = (
            (per_channel_max > absolute_threshold).sum().float()
        )
    return metrics


def compute_channel_max(hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute per-channel maximum absolute activation statistics.

    Tracks the "rise–plateau–fall" lifecycle of massive activations across layers.
    A sudden spike in channel_max indicates a step-up block is injecting outliers.

    Args:
        hidden_states: [S, B, H] or [B, S, H] post-residual hidden states.
            (Flattened to [*, H] internally)

    Returns:
        Dict with:
            channel_max: max absolute value across all positions and channels (scalar)
            channel_median: median of per-channel max absolute values (scalar)
            channel_p95/channel_p99: high quantiles of per-channel max absolute values
            channel_max_ratio: channel_max / channel_median (outlier severity)
    """
    return summarize_per_channel_max(compute_per_channel_max(hidden_states))


def compute_massive_activation_channel_count(
    hidden_states: torch.Tensor,
    threshold_multiplier: float = 100.0,
) -> torch.Tensor:
    """Count channels with activations exceeding a dynamic threshold.

    The threshold is set relative to the median channel magnitude:
        threshold = median_channel_max × threshold_multiplier

    This captures Property (ii) from Sun et al. (2026): massive activations
    are confined to a small subset of channels.

    Args:
        hidden_states: [S, B, H] or [B, S, H] post-residual hidden states.
        threshold_multiplier: multiplier on median to define "spike" threshold.

    Returns:
        Scalar tensor: number of channels exceeding the threshold.
    """
    return summarize_per_channel_max(
        compute_per_channel_max(hidden_states),
        threshold_multiplier=threshold_multiplier,
    )["massive_act_channel_count"]


def compute_topk_channel_norm(
    hidden_states: torch.Tensor,
    k: int = 3,
) -> torch.Tensor:
    """L2 norm of the top-K largest channel activations.

    Tracks the magnitude of the most extreme channels. In models with massive
    activations, this should show the "rise–plateau–fall" pattern across layers
    (Figure 1 in Sun et al. 2026).

    Args:
        hidden_states: [S, B, H] or [B, S, H] post-residual hidden states.
        k: number of top channels to include.

    Returns:
        Scalar tensor: L2 norm of the top-K per-channel-max values.
    """
    return summarize_per_channel_max(
        compute_per_channel_max(hidden_states), k=k
    )["topk_channel_norm"]


def compute_post_norm_sparsity(
    normalized_states: torch.Tensor,
    epsilon: float = 0.01,
) -> torch.Tensor:
    """Fraction of near-zero entries in post-RMSNorm hidden states.

    After normalization, spike tokens become sparse vectors where non-spike
    channels are suppressed to near-zero (Equation 24, Sun et al. 2026).
    High sparsity indicates the model is creating "implicit parameters" via
    the normalization-spike interaction.

    Args:
        normalized_states: [S, B, H] or [B, S, H] hidden states AFTER RMSNorm.
        epsilon: threshold below which a value is considered "near-zero".

    Returns:
        Scalar tensor: fraction of entries with |x| < epsilon.
    """
    h = normalized_states.reshape(-1).float()
    return (h.abs() < epsilon).float().mean()


def compute_post_norm_cosine_stability(
    normalized_states: torch.Tensor,
    num_sample_pairs: int = 256,
) -> torch.Tensor:
    """Cosine similarity among token representations after normalization.

    Near-constant post-norm representations (cosine → 1.0) indicate that
    normalization has collapsed diverse spike tokens into identical vectors
    (Figure 5, Sun et al. 2026). This is a precondition for attention sinks.

    Args:
        normalized_states: [S, B, H] post-RMSNorm hidden states.
        num_sample_pairs: number of random pairs to sample for efficiency.

    Returns:
        Scalar tensor: mean pairwise cosine similarity (sampled).
    """
    # Flatten to [num_tokens, hidden_dim]
    h = normalized_states.reshape(-1, normalized_states.shape[-1]).float()
    num_tokens = h.shape[0]

    if num_tokens < 2:
        return torch.tensor(1.0, device=h.device)

    # Sample random pairs
    n_pairs = min(num_sample_pairs, num_tokens * (num_tokens - 1) // 2)
    idx_a = torch.randint(0, num_tokens, (n_pairs,), device=h.device)
    idx_b = torch.randint(0, num_tokens - 1, (n_pairs,), device=h.device)
    # Avoid same index
    idx_b = idx_b + (idx_b >= idx_a).long()

    vec_a = h[idx_a]
    vec_b = h[idx_b]

    cosine = torch.nn.functional.cosine_similarity(vec_a, vec_b, dim=-1)
    return cosine.mean()
