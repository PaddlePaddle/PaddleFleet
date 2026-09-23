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
Enhanced QK Stats: Sink Head Classification Metrics.

Extends the base QK Stats Monitor with per-head sink classification,
based on findings from Sun et al. (2026) arXiv:2603.05498:

- Sink heads: heads where token-0 attention > threshold (typically >0.3)
  These heads act as "learned gates" that modulate attention output.
- Non-sink heads: heads with normal attention distribution.

Additional metrics:
    qk_stats/.../sink_head_ratio   — fraction of heads classified as sink heads
    qk_stats/.../sink_head_max     — strongest sink head's token-0 weight
    qk_stats/.../sink_nonsink_gap  — mean sink weight gap between sink/non-sink heads

Reference:
    Sun, S., Canziani, A., LeCun, Y., & Zhu, J. (2026).
    The Spike, the Sparse and the Sink. arXiv:2603.05498.
"""

import torch


def compute_sink_head_classification(
    sink_per_head: torch.Tensor,
    threshold: float = 0.3,
) -> dict[str, torch.Tensor]:
    """Classify attention heads as sink vs non-sink.

    A "sink head" is one where the average attention weight on token-0
    exceeds the threshold. These heads act as implicit gates — they dump
    excess attention mass onto a fixed position rather than distributing
    it semantically (Section 4.3 of Sun et al. 2026).

    Args:
        sink_per_head: [num_heads] mean attention weight on token-0 per head.
            (After TP gather, this should be the full head set.)
        threshold: attention weight above which a head is classified as "sink".
            Default 0.3 based on empirical observation that sink heads
            typically allocate >30% attention to token-0.

    Returns:
        Dict with:
            sink_head_ratio: fraction of heads classified as sink (0.0 to 1.0)
            sink_head_max: maximum sink weight across all heads
            sink_nonsink_gap: mean(sink_heads) - mean(nonsink_heads)
                Measures the logit gap proxy. Higher gap = more extreme sinks.
    """
    if sink_per_head.numel() == 0:
        zeros = torch.tensor(0.0, device=sink_per_head.device)
        return {
            "sink_head_ratio": zeros,
            "sink_head_max": zeros,
            "sink_nonsink_gap": zeros,
        }

    is_sink = sink_per_head > threshold
    num_heads = sink_per_head.numel()
    sink_count = is_sink.sum().float()
    nonsink_count = num_heads - sink_count
    is_sink_f = is_sink.to(sink_per_head.dtype)

    sink_head_ratio = sink_count / num_heads
    sink_head_max = sink_per_head.max()

    # Keep this branchless: Python comparisons on GPU tensors would sync.
    sink_mean = (sink_per_head * is_sink_f).sum() / sink_count.clamp_min(1.0)
    nonsink_mean = (
        sink_per_head * (1.0 - is_sink_f)
    ).sum() / nonsink_count.clamp_min(1.0)
    mixed_gap = sink_mean - nonsink_mean
    all_sink_gap = sink_per_head.mean()
    zero = torch.zeros(
        (), device=sink_per_head.device, dtype=sink_per_head.dtype
    )
    gap = torch.where(
        sink_count == 0,
        zero,
        torch.where(sink_count == num_heads, all_sink_gap, mixed_gap),
    )

    return {
        "sink_head_ratio": sink_head_ratio,
        "sink_head_max": sink_head_max,
        "sink_nonsink_gap": gap,
    }
