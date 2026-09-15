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

"""mHC (Manifold-Constrained Hyper-Connections) metric compute functions.

Paddle port of ``backends/megatron/mhc_metrics.py``. Pure, stateless tensor
helpers for the ``mhc_health`` monitor, operating on the three mappings a
``HyperConnectionModule`` produces per token (``n = num_residual_streams``):

- ``h_pre``  [..., n]     — stream aggregation gate (sigmoid)
- ``h_post`` [..., n]     — stream expansion gate (2 * sigmoid)
- ``h_res``  [..., n, n]  — Sinkhorn doubly-stochastic residual-mixing matrix

``gate_logits_extrema`` and ``mapping_param_stats`` reach one step further back,
to the ``alpha`` / ``bias`` / pre-sigmoid-logit terms of paper Eq. (7).

All functions return 0-dim GPU tensors and never sync the host (no ``.item()`` /
``.cpu()``), so they are safe on a forward hot path. See
``.claude/skills/monitor-hook-perf-rules``.
"""

import paddle

_EPS = 1e-12


def amax_gain(mat: paddle.Tensor, axis: int) -> paddle.Tensor:
    """Per-token max-abs {row|col} sum of a batched ``[..., n, n]`` matrix, averaged over tokens.

    The paper's worst-case gain bound. Which axis is "forward" depends on the
    matrix passed in: streams mix as ``out = h_res^T @ x``, so on ``h_res`` as
    stored ``axis=-2`` (column sums) is the forward gain and ``axis=-1`` the
    backward one; on an already-transposed matrix the two swap. Returns a 0-dim
    tensor.
    """
    sums = mat.sum(axis=axis)  # [..., n]
    return sums.abs().max(axis=-1).mean()  # 0-dim


def h_res_logits_extrema(
    h_res_logits: paddle.Tensor,
) -> dict[str, paddle.Tensor]:
    """Min/max of the raw residual-mixing logits before Sinkhorn."""
    logits = h_res_logits.detach().astype("float32")
    return {
        "h_res_logits_min": logits.min(),
        "h_res_logits_max": logits.max(),
    }


def gate_logits_extrema(
    proj: paddle.Tensor,
    r: paddle.Tensor,
    alpha_pre: paddle.Tensor,
    alpha_post: paddle.Tensor,
    bias: paddle.Tensor,
    n: int,
) -> dict[str, paddle.Tensor]:
    """Min/max of the *pre-sigmoid* ``H_pre`` / ``H_post`` logits (paper Eq. 7).

    Keep in sync with ``HyperConnectionModule._compute_h``: if the model changes
    how ``h`` is formed, these four series go silently wrong.
    """
    proj32 = proj.detach().astype("float32")
    r32 = r.detach().astype("float32")
    bias32 = bias.detach().astype("float32")
    pre = (
        r32 * proj32[..., :n] * alpha_pre.detach().astype("float32")
        + bias32[:n]
    )
    post = (
        r32 * proj32[..., n : 2 * n] * alpha_post.detach().astype("float32")
        + bias32[n : 2 * n]
    )
    return {
        "h_pre_logits_min": pre.min(),
        "h_pre_logits_max": pre.max(),
        "h_post_logits_min": post.min(),
        "h_post_logits_max": post.max(),
    }


def mapping_param_stats(
    alpha_pre: paddle.Tensor,
    alpha_post: paddle.Tensor,
    alpha_res: paddle.Tensor,
    bias: paddle.Tensor,
    n: int,
) -> dict[str, paddle.Tensor]:
    """The static half of paper Eq. (7): the gating factors and the bias terms.

    ``alpha_*`` are per-module scalars. ``bias`` is a single ``[n^2 + 2n]``
    parameter whose slices are the paper's ``b^pre`` / ``b^post`` / ``b^res``, so
    they are reported per slice rather than as one number. ``mean`` shows where
    the slice sits (it starts at 0 and drifts), ``abs_max`` catches a single
    runaway entry the mean would hide.

    Step-level quantities: record once per optimizer step, not per microbatch.
    """
    alpha = {
        "alpha_pre": alpha_pre,
        "alpha_post": alpha_post,
        "alpha_res": alpha_res,
    }
    stats = {
        name: value.detach().astype("float32").mean()
        for name, value in alpha.items()
    }

    b = bias.detach().astype("float32")
    slices = {
        "pre": b[:n],
        "post": b[n : 2 * n],
        "res": b[2 * n :],
    }
    for name, part in slices.items():
        stats[f"bias_{name}_mean"] = part.mean()
        stats[f"bias_{name}_abs_max"] = part.abs().max()
    return stats


def gate_stats(h: paddle.Tensor) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Mean and (unbiased) std of a gate tensor ``h`` over all elements."""
    return h.mean(), h.std()


def h_post_structure_stats(h_post: paddle.Tensor) -> dict[str, paddle.Tensor]:
    """Structure of the expansion gate across streams and across tokens."""
    flat = h_post.detach().astype("float32")
    flat = flat.reshape([-1, flat.shape[-1]])  # [tokens, n]

    per_token_mean = flat.mean(axis=-1)  # [tokens]
    per_token_max = flat.max(axis=-1)  # [tokens]
    concentration = (
        per_token_max / paddle.clip(per_token_mean, min=_EPS)
    ).mean()
    token_std = per_token_mean.std() if flat.shape[0] > 1 else paddle.zeros(())

    return {
        "h_post_stream_concentration": concentration,
        "h_post_token_std": token_std,
    }


def branch_residual_share(
    h_res: paddle.Tensor,
    original_residual: paddle.Tensor,
    h_post: paddle.Tensor,
    layer_output: paddle.Tensor,
    bias: paddle.Tensor | None = None,
) -> dict[str, paddle.Tensor]:
    """How much of the mHC update this layer wrote itself, per token?

    Args:
        h_res: ``[..., n, n]`` Sinkhorn doubly-stochastic mixing matrix.
        original_residual: ``[..., n*C]`` incoming n-stream hidden states.
        h_post: ``[..., n]`` expansion gate.
        layer_output: ``[..., C]`` the sublayer output ``F(·)``.
        bias: optional ``[C]`` bias added to the sublayer output.

    Returns:
        ``branch_residual_share`` (token mean) and
        ``branch_residual_share_max`` (worst token), both in ``[0, 1]``.

    Note:
        The branch term is measured *before* dropout, so configs with
        ``hidden_dropout_prob > 0`` slightly over-estimate it; the pretraining
        configs here use 0.
    """
    n = int(h_post.shape[-1])
    xb = layer_output.detach().astype("float32")
    if bias is not None:
        xb = xb + bias.detach().astype("float32").reshape(
            [1] * (xb.ndim - 1) + [-1]
        )
    xb = xb.reshape([-1, xb.shape[-1]])  # [tokens, C]

    gate = h_post.detach().astype("float32").reshape([-1, n])  # [tokens, n]
    branch_norm = gate.norm(axis=-1) * xb.norm(axis=-1)  # [tokens]

    streams = original_residual.detach().astype("float32")
    streams = streams.reshape([-1, n, streams.shape[-1] // n])  # [tokens, n, C]
    res = h_res.detach().astype("float32").reshape([-1, n, n])  # [tokens, n, n]

    gram = paddle.einsum("tjc,tkc->tjk", streams, streams)  # [tokens, n, n]
    mix = paddle.einsum("tji,tki->tjk", res, res)  # [tokens, n, n]
    residual_sq = (gram * mix).sum(axis=[-2, -1])  # [tokens]
    residual_norm = paddle.sqrt(paddle.clip(residual_sq, min=0.0))

    # eps only guards the both-terms-zero token (share 0); it does not bound the
    # value, which is why this form is safe where the raw ratio was not.
    share = branch_norm / paddle.clip(branch_norm + residual_norm, min=_EPS)
    return {
        "branch_residual_share": share.mean(),
        "branch_residual_share_max": share.max(),
    }
