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

"""VHA (Virtual Head Attention) metric kernels for PaddleFleet.

VHA halves the physical query heads and sets ``H_kv = 2``, then restores
expressiveness with two strictly linear operations (Ding et al., *Virtual Head
Attention*):

- **Q Premix** — a near-identity transform applied per KV group, expanding the
  halved physical query heads into ``H_v = H'_q × H_kv`` virtual heads.
- **Linear Postmix** — a low-rank identity-residual on the head axis of the
  attention output, ``I + A Bᵗ`` with ``r = H'_q``, fusing cross-group inter-head
  features. PaddleFleet names the factors ``vha_postmix_U`` / ``vha_postmix_V``,
  i.e. ``A`` / ``B``::

    mixed = attn_out.reshape([..., nh, v_head_dim])
    delta = (mixed @ U) @ Vᵀ          # U, V: [nh, rank]
    out   = mixed + delta             # operator M = I + U Vᵀ on the head axis

Both fold away at inference (Premix into ``q_proj``, Postmix into ``o_proj``),
leaving a plain GQA-2 model — so these matrices only exist during training, and
this is the only place their trained structure can be observed.

``V`` is zero-initialised, so Postmix starts as the exact identity and ``delta``
starts at 0. Whether (and how fast) it moves away is the central question here.

Grouped postmix uses ``[groups, heads_per_group, rank]`` factors and mixes heads
only inside a group, i.e. ``M`` is block diagonal. Every function below treats
the ungrouped case as a single group so both topologies share one code path.

All functions return 0-dim GPU tensors and never sync to host: the monitor
records them through the GPU-buffer API and a single D2H happens at flush.
"""

from __future__ import annotations

import paddle

_EPS = 1e-8


def _as_grouped(factor: paddle.Tensor) -> paddle.Tensor:
    """Return a ``[groups, heads_per_group, rank]`` view of a postmix factor."""
    if factor.ndim == 2:
        return factor.unsqueeze(0)
    return factor


def postmix_operator_stats(
    U: paddle.Tensor, V: paddle.Tensor
) -> dict[str, paddle.Tensor]:
    """Spectral statistics of the postmix operator ``M = I + U Vᵀ``.

    Cheap by construction: the factors are ``[nh, rank]`` (e.g. ``[64, 16]``),
    so the Gram matrix is at most ``[nh, nh]``. Singular values come from
    ``eigvalsh`` of ``AᵀA`` rather than an SVD — symmetric, batched, and with no
    convergence caveats.

    Returns:
        ``postmix_uv_sigma_max`` — ``σ_max(U Vᵀ)``: how far the operator can
        stretch a head-space direction beyond the identity.
        ``postmix_uv_eff_rank`` — entropy-based effective rank of ``U Vᵀ``
        (``exp(-Σ p log p)`` over normalised singular values). Read this as
        "is the mixing staying inside its low-rank budget", **not** as capacity
        utilisation: the bottleneck ``r ≪ H`` is a deliberate structural
        regulariser, and the paper's ablation shows larger ranks (32, 128) hurt
        evaluation despite lower training loss. Growth towards the configured
        rank is therefore a warning, not health.
        ``postmix_offdiag_ratio`` — off-diagonal energy share of ``M``,
        ``‖A − diag(A)‖_F / ‖M‖_F``. Near 0 means the operator degenerated into
        a per-head rescaling and no real cross-head mixing is happening.
        ``postmix_u_fro`` / ``postmix_v_fro`` — factor magnitudes, to see which
        side is growing.
    """
    u = _as_grouped(U.detach().astype("float32"))
    v = _as_grouped(V.detach().astype("float32"))
    # Per group: A_g = U_g V_gᵀ, the off-identity part of the block operator.
    a = paddle.einsum("gjr,gkr->gjk", u, v)  # [G, n, n]

    gram = paddle.matmul(a, a, transpose_x=True)  # AᵀA, symmetric PSD
    eigenvalues = paddle.linalg.eigvalsh(gram)  # [G, n], ascending
    singular = paddle.sqrt(paddle.clip(eigenvalues, min=0.0))

    sigma_max = singular.max()
    probabilities = singular / paddle.clip(
        singular.sum(axis=-1, keepdim=True), min=_EPS
    )
    entropy = -(
        probabilities * paddle.log(paddle.clip(probabilities, min=_EPS))
    ).sum(axis=-1)
    eff_rank = paddle.exp(entropy).mean()

    diagonal = paddle.diagonal(a, axis1=-2, axis2=-1)  # [G, n]
    offdiag_sq = paddle.square(a).sum() - paddle.square(diagonal).sum()
    # ‖M‖_F² with M = A + I on every block; only the diagonal shifts by 1.
    full_sq = (
        paddle.square(a).sum()
        - paddle.square(diagonal).sum()
        + paddle.square(diagonal + 1.0).sum()
    )
    offdiag_ratio = paddle.sqrt(
        paddle.clip(offdiag_sq, min=0.0) / paddle.clip(full_sq, min=_EPS)
    )

    return {
        "postmix_uv_sigma_max": sigma_max,
        "postmix_uv_eff_rank": eff_rank,
        "postmix_offdiag_ratio": offdiag_ratio,
        "postmix_u_fro": u.norm(),
        "postmix_v_fro": v.norm(),
    }


def postmix_delta_stats(
    attn_out: paddle.Tensor, out: paddle.Tensor
) -> dict[str, paddle.Tensor]:
    """Per-token effect of the postmix correction on the attention output.

    ``attn_out`` is the input of ``_apply_vha_postmix`` and ``out`` its return
    value, so ``delta = out − attn_out`` needs no recomputation.

    The **return** value is always the flat head-space layout
    ``[..., nh * v_head_dim]`` (every backend ends with that reshape), but the
    **input** is not: the DSv4 hybrid attention leaves ``core_attn_out`` in the
    unflattened ``[b, sq, nh, v_head_dim]`` layout on the fused inverse-RoPE
    path (``apply_rope_fusion`` without ``high_precision_rope``). So the flat
    width is taken from ``out`` and both sides are folded to
    ``[tokens, nh * v_head_dim]`` — using each tensor's own last dim would put
    heads on the token axis for one side and break the subtraction.

    Returns:
        ``postmix_delta_rel_mean`` / ``postmix_delta_rel_max`` — per-token
        ``‖delta‖₂ / ‖mixed‖₂``. This is the primary read: it starts at 0 (zero
        init of ``V``) and its trend answers "is postmix learning anything".
        ``postmix_amax_gain_max`` — worst-token ``‖out‖_∞ / ‖mixed‖_∞``, the
        low-precision overflow headroom of the correction (same reading as the
        mHC ``amax_gain`` family).
    """
    result = out.detach().astype("float32")
    width = result.shape[-1]
    result = result.reshape([-1, width])
    mixed = attn_out.detach().astype("float32").reshape([-1, width])
    delta = result - mixed

    mixed_norm = paddle.clip(mixed.norm(axis=-1), min=_EPS)
    delta_rel = delta.norm(axis=-1) / mixed_norm

    mixed_amax = paddle.clip(mixed.abs().max(axis=-1), min=_EPS)
    amax_gain = result.abs().max(axis=-1) / mixed_amax

    return {
        "postmix_delta_rel_mean": delta_rel.mean(),
        "postmix_delta_rel_max": delta_rel.max(),
        "postmix_amax_gain_max": amax_gain.max(),
    }


def head_output_stats(
    out: paddle.Tensor, num_heads: int
) -> dict[str, paddle.Tensor]:
    """Head-space diversity of the postmix output.

    ``head_out_norm_*`` is the spread of the token-mean per-head output norm:
    a head collapsing to ~0 or blowing up shows here first.

    ``postmix_head_cos_mean`` is the mean absolute off-diagonal cosine of the
    token-mean head vectors — the direct read on "did the mixer collapse the
    heads into copies of each other". Using token-mean vectors (rather than a
    per-token pairwise mean) keeps this to one ``[nh, nh]`` Gram; it is a
    directional-agreement signal, not an exact per-token average.
    """
    flat = out.detach().astype("float32")
    flat = flat.reshape([-1, num_heads, flat.shape[-1] // num_heads])

    head_norm = flat.norm(axis=-1).mean(axis=0)  # [nh]
    stats = {
        "head_out_norm_max": head_norm.max(),
        "head_out_norm_min": head_norm.min(),
        "head_out_norm_std": head_norm.std()
        if num_heads > 1
        else paddle.zeros(()),
    }

    if num_heads < 2:
        stats["postmix_head_cos_mean"] = paddle.zeros(())
        return stats

    head_vec = flat.mean(axis=0)  # [nh, v_head_dim]
    head_vec = head_vec / paddle.clip(
        head_vec.norm(axis=-1, keepdim=True), min=_EPS
    )
    gram = paddle.matmul(head_vec, head_vec, transpose_y=True).abs()  # [nh, nh]
    offdiag_sum = gram.sum() - paddle.diagonal(gram).sum()
    stats["postmix_head_cos_mean"] = offdiag_sum / float(
        num_heads * (num_heads - 1)
    )
    return stats


def premix_metric_names(weight: paddle.Tensor) -> tuple[str, ...]:
    """Metric keys ``premix_stats`` will produce for this weight.

    The schema depends on the weight shape, which is known at registration time,
    so the monitor can still declare everything before allocating buffers.
    """
    groups = int(weight.shape[0]) if weight.ndim == 3 else 1
    rows, cols = int(weight.shape[-2]), int(weight.shape[-1])
    if rows != cols:
        return ("premix_sigma_max", "premix_orth_dev")
    names = ["premix_sigma_max", "premix_identity_dev"]
    if groups > 1:
        names.append("premix_group_div_ratio")
    return tuple(names)


def premix_stats(weight: paddle.Tensor) -> dict[str, paddle.Tensor]:
    """Structure of the trained Q Premix weight (``[H_kv, q_head_dim, head_dim]``).

    Premix is **near-identity** by construction: PaddleFleet initialises it as
    ``I + 0.1/√d · N(0, 1)`` per KV group when ``q_head_dim == head_dim``. So the
    diagnostic is deviation from the identity, not from orthogonality.

    ``premix_identity_dev`` — group-mean ``‖W_g − I‖_F``: how far each group's
    query transform has moved from a no-op.

    ``premix_group_div_ratio`` — ``mean_{g<g'} ‖W_g − W_g'‖²_F / mean_g ‖W_g − I‖²_F``.
    This is the paper's own measurement of *group-conditioned query
    specialization*: 0 means every group learned the same deviation (the virtual
    heads inside different KV groups are not specialising), 2 is what fully
    independent per-group deviations would give. The paper reports a value well
    below 2 and far above 0 — groups learn substantially but not independently
    different transforms. Both extremes are the warning signs.

    ``premix_orth_dev`` — only for the non-square case (``q_head_dim ≠ head_dim``),
    where PaddleFleet falls back to a scaled orthogonal init; then ``‖WᵀW − I‖_F``
    is the meaningful drift.
    """
    w = weight.detach().astype("float32")
    if w.ndim == 2:
        w = w.unsqueeze(0)
    groups, rows, cols = w.shape

    gram = paddle.matmul(w, w, transpose_x=True)  # [G, cols, cols]
    eigenvalues = paddle.linalg.eigvalsh(gram)
    sigma_max = paddle.sqrt(paddle.clip(eigenvalues, min=0.0)).max()

    if rows != cols:
        identity = paddle.eye(cols, dtype=w.dtype).unsqueeze(0)
        return {
            "premix_sigma_max": sigma_max,
            "premix_orth_dev": (gram - identity).norm(),
        }

    deviation = w - paddle.eye(rows, dtype=w.dtype).unsqueeze(0)  # [G, d, d]
    deviation_sq = paddle.square(deviation).sum(axis=[-2, -1])  # [G]
    stats = {
        "premix_sigma_max": sigma_max,
        "premix_identity_dev": paddle.sqrt(deviation_sq).mean(),
    }
    if groups > 1:
        # Pairwise ‖W_g − W_g'‖² over the (tiny) group axis; G is H_kv, e.g. 2.
        pair_sq = paddle.square(
            deviation.unsqueeze(0) - deviation.unsqueeze(1)
        ).sum(axis=[-2, -1])
        pair_mean = pair_sq.sum() / float(
            groups * (groups - 1)
        )  # off-diagonal mean
        stats["premix_group_div_ratio"] = pair_mean / paddle.clip(
            deviation_sq.mean(), min=_EPS
        )
    return stats
