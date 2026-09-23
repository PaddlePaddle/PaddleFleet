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

Pure, stateless tensor -> tensor helpers for the ``mhc_health`` monitor. They
operate on the three mappings a ``HyperConnectionModule`` produces per token:

- ``h_pre``  [s, b, n]      — stream aggregation gate (sigmoid)
- ``h_post`` [s, b, n]      — stream expansion gate (2*sigmoid)
- ``h_res``  [s, b, n, n]   — Sinkhorn doubly-stochastic residual-mixing matrix

``n = num_residual_streams``, ``s`` = sequence, ``b`` = micro-batch.

The ``amax_gain`` diagnostic follows the mHC paper: the max-abs **row** sum of a
mixing matrix bounds the worst-case forward-pass expansion, and the max-abs
**column** sum bounds the backward-pass expansion. For a single doubly-stochastic
``h_res`` both sit at ~1.0; on the *composite* mapping (cumulative product of
``h_res`` across layers) they drift away from 1.0 with depth, flagging
residual-stream amplification.

All functions return 0-dim GPU tensors and never sync the host (no ``.item()`` /
``.cpu()``), so they are safe to call from a forward hot path. See
``.claude/skills/monitor-hook-perf-rules``.
"""

import torch


def amax_gain(mat: torch.Tensor, dim: int) -> torch.Tensor:
    """Per-token max-abs {row|col} sum of a batched ``[..., n, n]`` matrix, averaged over tokens.

    ``dim=-1`` sums over columns -> per-row sums (forward gain); ``dim=-2`` sums
    over rows -> per-column sums (backward gain). ``sum(dim)`` collapses one
    ``n``-axis to ``[..., n]``; ``abs().amax(-1)`` takes the worst stream per token
    (a 1-per-token scalar); ``mean()`` averages over all tokens.

    Returns a 0-dim tensor on ``mat``'s device/dtype (fp32 from ``compute_mappings``).
    """
    sums = mat.sum(dim=dim)  # [..., n]
    return sums.abs().amax(dim=-1).mean()  # 0-dim


def gate_stats(h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and (unbiased) std of a gate tensor ``h`` [s, b, n] over all elements.

    Returns two 0-dim tensors ``(mean, std)``; no host sync.
    """
    return h.mean(), h.std()
