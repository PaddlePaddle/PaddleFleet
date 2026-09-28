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

"""KDA (Kimi Delta Attention) metric compute functions.

Pure, stateless tensor helpers for the ``kda_health`` monitor, grouped by the
pathway they measure: decay (``alpha``), write (``beta``), read (output gate).
What each metric means is in ``docs/kda_health.md``.

All functions return 0-dim GPU tensors and never sync the host (no ``.item()`` /
``.cpu()``), so they are safe on a forward hot path. See
``.claude/skills/monitor-hook-perf-rules``.
"""

import paddle

DECAY_METRICS = (
    "alpha_log_mean",
    "alpha_channel_spread",
    "alpha_token_spread",
    "alpha_log_channel_min",
)
WRITE_METRICS = ("beta_mean", "beta_head_min")
READ_METRICS = ("out_gate_mean",)
PARAM_METRICS = ("A_log_mean",)

#: Metric names reduced by min instead of mean, for ``Probe.MIN_AGGREGATED``.
MIN_METRICS = frozenset({"alpha_log_channel_min", "beta_head_min"})

ALL_METRICS = DECAY_METRICS + WRITE_METRICS + READ_METRICS + PARAM_METRICS


def _as_tokens(x: paddle.Tensor, last_dim: int) -> paddle.Tensor:
    """Flatten ``[..., last_dim]`` to ``[tokens, last_dim]``.

    Keyed off the trailing dimension, not the leading ones: with
    ``sequence_parallel`` the projections hand back ``[s, b, d]`` instead of
    ``[b, s, d]``, and every statistic here pools both of those axes anyway.
    """
    return x.reshape([-1, last_dim])


def kda_log_decay(
    z: paddle.Tensor,
    A_log: paddle.Tensor,
    dt_bias: paddle.Tensor | None,
    safe_gate: bool,
    lower_bound: float | None,
    num_heads: int,
    head_dim: int,
) -> paddle.Tensor:
    """Raw decay logits -> per-step log decay ``g``, shaped ``[tokens, hv, dk]``.

    Mirrors ``kimi_delta_attention.kda_gate`` in fp32. ``safe_gate`` /
    ``lower_bound`` are read off the layer rather than hardcoded, so a run with
    ``gate_lower_bound=None`` gets the unbounded softplus form it actually uses.
    """
    x = _as_tokens(z.astype("float32"), num_heads * head_dim).reshape(
        [-1, num_heads, head_dim]
    )
    if dt_bias is not None:
        x = x + dt_bias.astype("float32").reshape([num_heads, head_dim])
    a = A_log.astype("float32").exp().reshape([num_heads, 1])
    if safe_gate:
        if lower_bound is None:
            raise ValueError("safe_gate=True requires lower_bound to be set")
        return lower_bound * paddle.nn.functional.sigmoid(a * x)
    return -a * paddle.nn.functional.softplus(x)


def decay_gate_stats(g: paddle.Tensor) -> dict[str, paddle.Tensor]:
    """Memory-time-scale statistics of the log decay ``g`` ``[tokens, hv, dk]``."""
    g = g.astype("float32")
    per_token = g.mean(axis=[-2, -1])
    return {
        "alpha_log_mean": g.mean(),
        # std along the channel axis first, then pool tokens and heads: a std over
        # everything at once cannot tell channel diversity from token variation.
        "alpha_channel_spread": g.std(axis=-1).mean(),
        # std over a length-1 axis is undefined; report 0 rather than NaN.
        "alpha_token_spread": per_token.std()
        if per_token.shape[0] > 1
        else paddle.zeros(()),
        # Token-mean before the extremum, so one outlier token cannot pin the
        # reading to the gate's bound. Same rule as beta_head_min below.
        "alpha_log_channel_min": g.mean(axis=0).min(),
    }


def write_gate_stats(
    beta_logit: paddle.Tensor, num_heads: int
) -> dict[str, paddle.Tensor]:
    """Write-pathway statistics from the pre-sigmoid ``beta`` ``[..., hv]``.

    The sigmoid is folded into the kernel (``use_beta_sigmoid_in_kernel=True``),
    so it is applied here; fp32 matches the promotion the layer does first.
    """
    beta = paddle.nn.functional.sigmoid(
        _as_tokens(beta_logit.astype("float32"), num_heads)
    )
    return {
        "beta_mean": beta.mean(),
        "beta_head_min": beta.mean(axis=0).min(),
    }


def read_gate_stats(
    gate_logit: paddle.Tensor, gate_width: int
) -> dict[str, paddle.Tensor]:
    """Read-pathway statistic from the pre-sigmoid output gate."""
    gate = paddle.nn.functional.sigmoid(
        _as_tokens(gate_logit.astype("float32"), gate_width)
    )
    return {"out_gate_mean": gate.mean()}


def param_stats(A_log: paddle.Tensor) -> dict[str, paddle.Tensor]:
    """Attribution parameter for the decay pathway: ``exp(A_log)`` scales the gate."""
    return {"A_log_mean": A_log.detach().astype("float32").mean()}
