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

"""GPU-side metric computation for APE health monitoring."""

import paddle


def compute_ape_p0_metrics(ape: paddle.Tensor) -> dict[str, paddle.Tensor]:
    """Compute P0 metrics for one ``[position, channel]`` APE tensor."""
    if len(ape.shape) != 2:
        raise ValueError(
            f"APE must be rank-2 [position, channel], got {ape.shape}"
        )
    if ape.shape[0] <= 0 or ape.shape[1] <= 0:
        raise ValueError(f"APE must have non-empty dimensions, got {ape.shape}")

    values = paddle.cast(ape, "float32")
    finite = paddle.isfinite(values)
    non_finite_ratio = 1.0 - paddle.cast(finite, "float32").mean()
    values = paddle.where(finite, values, paddle.zeros_like(values))

    rms = paddle.sqrt(paddle.mean(paddle.square(values)))
    centered = values - values.mean(axis=0, keepdim=True)
    centered_rms = paddle.sqrt(paddle.mean(paddle.square(centered)))

    position_range = values.max(axis=0) - values.min(axis=0)
    position_range_p95 = paddle.quantile(position_range, 0.95)

    probabilities = paddle.nn.functional.softmax(values, axis=0)
    entropy = -(
        probabilities * paddle.log(paddle.clip(probabilities, min=1e-12))
    ).sum(axis=0)
    position_count = paddle.to_tensor(float(values.shape[0]), dtype="float32")
    entropy_norm = entropy / paddle.log(position_count)
    max_probability = probabilities.max(axis=0)

    return {
        "non_finite_ratio": non_finite_ratio,
        "rms": rms,
        "centered_rms": centered_rms,
        "position_range_p95": position_range_p95,
        "softmax_entropy_norm_mean": entropy_norm.mean(),
        "softmax_max_prob_p95": paddle.quantile(max_probability, 0.95),
        "effective_positions_mean": paddle.exp(entropy).mean(),
    }
