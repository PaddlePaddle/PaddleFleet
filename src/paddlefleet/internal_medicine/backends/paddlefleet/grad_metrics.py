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

"""Activation-gradient magnitude metrics for PaddleFleet.

Pure tensor functions, no monitor state. Everything returns a 0-dim GPU tensor so
the caller hands it straight to ``record_layer_metric`` without a D2H sync --
hot-path discipline: see ``.claude/skills/monitor-hook-perf-rules``.

The quantity being measured is dL/d(activation) at a module's *output*, i.e. the
gradient that flows back into that module. This is the backward-path counterpart
of ``massive_act``'s forward-path magnitudes, and the positions are named to
match so a gradient curve and its activation curve line up in the viewer.
"""

from __future__ import annotations

import math

import paddle

# Where on the backward path a gradient is read. ``layer_out`` is the residual
# stream leaving a decoder layer -- the series that answers "does the gradient
# survive depth"; the other two attribute a layer's share to its two branches.
POSITIONS = ("attn_out", "ffn_or_moe_out", "layer_out")

METRICS = ("norm", "rms", "abs_max")

# Exact cross-rank quantities. They need a collective, so they are produced at
# flush time from a running sum of squares rather than in the hook -- see
# ``grad_monitor.ExactNormReducer``. Names are separate keys, so the three
# approximate series above keep their exact previous values.
#
# - ``norm_mb``     one microbatch's *complete* tensor, i.e. summed back over
#                   whatever shards the same tensor (SP / CP / SP-in-TP), then
#                   averaged over microbatches and data ranks. Equals ``norm``
#                   exactly on a layout that does not shard activations.
# - ``rms_global``  sqrt(sum g^2 / sum N) over every rank and microbatch. The
#                   only one of the six that is invariant to cluster size,
#                   parallel layout and gradient-accumulation depth.
# - ``norm_global`` sqrt(sum g^2) over every rank and microbatch, i.e. the whole
#                   global batch. Closest in spirit to the optimizer's global
#                   grad norm, but it grows like sqrt(world_size * gas), so it is
#                   only comparable within one cluster shape.
GLOBAL_METRICS = ("norm_mb", "rms_global", "norm_global")

# Per-token decomposition. ``abs_max`` finds the largest single *element*, which
# cannot tell "one whole token carries a huge gradient" (a data problem: a rare
# token, an EOS, a dirty sample) from "one cell ran away" (a numerical / channel
# problem). Those two want opposite fixes, so the token axis gets its own series.
#
# The token axis, not the channel axis, is the right decomposition here:
# ``massive_act`` splits activations per channel because massive activations are
# a channel phenomenon, while gradient spikes are usually carried by individual
# tokens.
#
# All four are exact across ranks with no collective: every position is a
# post-projection full-hidden tensor, so each rank holds *whole* tokens and a max
# over ranks is a max over all tokens. ``token_outlier_ratio`` is a fraction
# rather than a count for the same reason -- a count would be per-shard.
TOKEN_METRICS = (
    "token_norm_max",
    "token_norm_p99",
    "token_norm_ratio",
    "token_outlier_ratio",
    "token_zero_ratio",
)

# ``abs_max`` is a max over microbatches / ranks / layers, the rest are means.
# The two token extremes join it: for a spike detector the worst microbatch is
# the interesting one, and averaging it away is exactly the wrong reduction.
MAX_METRICS = ("abs_max", "token_norm_max", "token_norm_ratio")

# How far above the per-token median a token counts as an outlier. Matches
# ``massive_act``'s 10x convention so the forward and backward "outlier_ratio"
# series are read the same way.
TOKEN_OUTLIER_MULTIPLIER = 10.0

ALL_METRICS = tuple(
    f"{position}_{metric}"
    for position in POSITIONS
    for metric in METRICS + GLOBAL_METRICS + TOKEN_METRICS
)

MAX_AGGREGATED = frozenset(
    f"{position}_{metric}" for position in POSITIONS for metric in MAX_METRICS
)


def grad_token_stats(value: paddle.Tensor) -> dict[str, paddle.Tensor]:
    """Per-token gradient-vector statistics of an already-fp32 tensor.

    ``value`` is flattened to ``[tokens, width]`` on the last axis, so this works
    for both ``[B, S, H]`` and ``[S, H]`` layouts without being told which it got.

    - ``token_norm_max``      the loudest token's ``||g_t||``
    - ``token_norm_p99``      the tail's shape; its distance from the max says
                              whether the spike is one isolated token or many
    - ``token_norm_ratio``    ``max / median``, the token-level peakiness --
                              the counterpart of ``massive_act``'s
                              ``channel_max_ratio``
    - ``token_outlier_ratio`` fraction of tokens above ``10x`` the median
    - ``token_zero_ratio``    fraction of tokens with no gradient at all

    **The median is taken over gradient-carrying tokens only.** A first version
    used the plain median and measured the wrong thing on a real 8k run:
    loss-masked positions gave whole rows of exact zeros, the median sat at ~0,
    and ``token_norm_ratio`` came out 2000-19000 -- a reading of *how much of the
    batch is masked* rather than of token peakiness. ``nanmedian`` over the
    non-zero rows restores the intended meaning, and ``token_zero_ratio`` exposes
    the masked fraction as its own series instead of letting it distort another.

    Costs one more full reduction than the scalar stats (``norm(axis=-1)`` over
    the whole tensor); the max / median / quantile that follow run on a vector of
    length ``tokens``, which is negligible. Materialising the squares to share a
    pass with ``mean_sq`` would cost a 4-byte-per-element temporary -- 268 MB on a
    43-layer 8k run -- so the second pass is the cheaper trade.
    """
    flat = value.reshape([-1, value.shape[-1]])
    token_norm = paddle.linalg.norm(flat, axis=-1)
    alive = token_norm > 0
    # NaN for the dead rows so ``nanmedian`` ignores them without a host-side
    # boolean index (which would need the count on the CPU, i.e. a D2H sync).
    nonzero = paddle.where(
        alive, token_norm, paddle.full_like(token_norm, float("nan"))
    )
    median = paddle.nanmedian(nonzero)
    # All-masked microbatch -> nanmedian is NaN; fall back to a positive epsilon so
    # the ratio stays finite instead of poisoning the step with NaN.
    median = paddle.where(
        paddle.isnan(median), paddle.zeros_like(median), median
    ).clip(min=1e-30)
    peak = token_norm.max()
    return {
        "token_norm_max": peak,
        "token_norm_p99": paddle.quantile(token_norm, 0.99),
        "token_norm_ratio": peak / median,
        "token_outlier_ratio": (token_norm > TOKEN_OUTLIER_MULTIPLIER * median)
        .astype("float32")
        .mean(),
        "token_zero_ratio": 1.0 - alive.astype("float32").mean(),
    }


def grad_square_and_stats(
    grad: paddle.Tensor,
) -> tuple[dict[str, paddle.Tensor], paddle.Tensor, int]:
    """``(display stats, sum of squares, element count)`` from one cast.

    The scalar metrics, the per-token series and the exact-reduction summand all
    read the same fp32 view of the gradient. ``numel`` is Python metadata
    (``shape`` in dygraph), so it never forces a D2H sync.
    """
    value = grad.detach().astype("float32")
    mean_sq = value.square().mean()
    numel = max(1, math.prod(value.shape))
    rms = paddle.sqrt(mean_sq)
    stats = {
        "norm": rms * math.sqrt(numel),
        "rms": rms,
        "abs_max": value.abs().max(),
    }
    stats.update(grad_token_stats(value))
    return stats, mean_sq * float(numel), numel


def grad_magnitude_stats(grad: paddle.Tensor) -> dict[str, paddle.Tensor]:
    """``norm`` / ``rms`` / ``abs_max`` of one activation gradient.

    All three are homogeneous of degree 1 in ``grad``. That is what lets a single
    division by the AMP loss scale de-scale the whole set at finalize time, for
    both the mean accumulators (a sum over microbatches) and the max ones.

    - ``norm`` -- the L2 norm the caller asked for, over *this rank's shard*.
      Shape-dependent, so read it across steps, not across layers.
    - ``rms`` -- ``norm / sqrt(N)``. Shard- and shape-invariant under equal
      sharding (each rank's mean square is an unbiased estimate of the global
      one), so this is the series that is comparable across layers and across
      parallel layouts.
    - ``abs_max`` -- ``max|g|``, the spike detector. Max-aggregated, so one
      outlier microbatch or rank still reaches the global key.

    Two reductions, not three: ``N`` comes from ``shape``, which is Python
    metadata in dygraph, so ``norm`` is a scale of ``rms`` rather than a second
    sum over the tensor.
    """
    value = grad.detach().astype("float32")
    rms = paddle.sqrt(value.square().mean())
    sqrt_numel = math.sqrt(max(1, math.prod(value.shape)))
    return {
        "norm": rms * sqrt_numel,
        "rms": rms,
        "abs_max": value.abs().max(),
    }
