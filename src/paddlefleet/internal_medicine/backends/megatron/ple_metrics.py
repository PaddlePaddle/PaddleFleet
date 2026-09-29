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
PLE Health Metrics Computation Functions.

Core metrics:
1. Residual Ratio - PLE contribution vs hidden states
2. Gate Statistics - gate activation health and sparsity
3. Branch Cosine - token/projection branch redundancy
4. Branch Norms - per-branch signal magnitude

All metrics compute local values only; cross-rank aggregation is handled
by training_logs.gather_and_aggregate().
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F


def compute_residual_ratio(
    hidden_states: torch.Tensor, output: torch.Tensor
) -> torch.Tensor:
    """PLE residual contribution relative to hidden states.

    residual_ratio = ||output - hidden_states|| / ||hidden_states||

    Args:
        hidden_states: [S, B, H] input to PLESubmodule
        output:        [S, B, H] output of PLESubmodule (= hidden_states + post_norm(down_out))

    Returns:
        Scalar tensor: ratio of PLE contribution norm to hidden states norm.
    """
    ple_contribution = output - hidden_states
    return ple_contribution.norm() / hidden_states.norm().clamp(min=1e-8)


def compute_gate_stats(
    gate_out: torch.Tensor,
    act_fn: Callable,
    threshold: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Gate activation statistics.

    Args:
        gate_out:  [S, B, H_ple] raw gate projection output (before activation)
        act_fn:    activation function (F.gelu or F.silu)
        threshold: absolute value below which a gate unit is considered dead

    Returns:
        Dict with gate_activation_mean and gate_sparsity.
    """
    activated = act_fn(gate_out)
    abs_activated = activated.abs()
    return {
        "gate_activation_mean": abs_activated.mean(),
        "gate_sparsity": (abs_activated < threshold).float().mean(),
    }


def compute_branch_cosine(
    token_ple: torch.Tensor, proj_ple: torch.Tensor
) -> torch.Tensor:
    """Cosine similarity between token and projection PLE branches.

    Near 1.0 means the two branches carry redundant information.

    Args:
        token_ple: [S, B, L, H_ple]
        proj_ple:  [S, B, L, H_ple] (after norm and H^-0.5 scale)

    Returns:
        Scalar tensor: mean cosine similarity across all positions.
    """
    t_flat = token_ple.reshape(-1, token_ple.shape[-1])
    p_flat = proj_ple.reshape(-1, proj_ple.shape[-1])
    return F.cosine_similarity(t_flat, p_flat, dim=-1).mean()


def compute_branch_norms(
    token_ple: torch.Tensor,
    proj_ple: torch.Tensor,
    per_layer_input_scale: float,
) -> dict[str, torch.Tensor]:
    """Norm statistics for PLE input branches.

    Args:
        token_ple:            [S, B, L, H_ple]
        proj_ple:             [S, B, L, H_ple] (after norm and H^-0.5 scale)
        per_layer_input_scale: 2^-0.5, the final scale applied to combined signal

    Returns:
        Dict with token_ple_norm, proj_ple_norm, per_layer_inputs_norm.
    """
    combined = (token_ple + proj_ple) * per_layer_input_scale
    return {
        "token_ple_norm": token_ple.norm(dim=-1).mean(),
        "proj_ple_norm": proj_ple.norm(dim=-1).mean(),
        "per_layer_inputs_norm": combined.norm(dim=-1).mean(),
    }
