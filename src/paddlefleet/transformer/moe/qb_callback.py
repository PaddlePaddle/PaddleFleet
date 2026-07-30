# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
MoE Quantile Balancing Callback.

Implements the QB bias update algorithm from Kimi K3 (Technical Report §2.3.3):
  1. Collects per-expert histograms of required_bias from all EP-parallel ranks.
  2. Recovers the (k/n)-quantile for each expert via cumulative-count + linear
     interpolation on the merged histogram.
  3. Assigns the quantile as the new expert bias (zero-mean normalized).

Usage (PaddleFormers TrainerCallback interface):
    from paddlefleet.transformer.moe.qb_callback import MoEQuantileBalancingCallback
    callback = MoEQuantileBalancingCallback()
    trainer.add_callback(callback)

Or standalone after each optimizer step:
    callback.on_optimizer_end(args=None, state=None, control=None, model=model)
"""

from __future__ import annotations

import logging

import paddle
import paddle.distributed as dist

logger = logging.getLogger(__name__)


def _try_get_comm_groups():
    """Get communication groups needed for QB histogram all-reduce.

    Returns (tp_group, cp_group, dp_group, sd_group) where each is None if not
    applicable (group missing, or a single-rank group that needs no reduction).

    Every group listed here corresponds to ranks whose routers observe a
    *different* set of tokens, so their histograms must be summed to recover the
    global-batch statistics:

    - TP: When SP is enabled with EP > 1, the router only sees a 1/TP fraction of
      the sequence on each TP rank (moe_layer skips the AllGather when EP > 1).
      Guarded at the call site, since with EP == 1 the AllGather already restores
      the full sequence.
    - CP: Context Parallel always splits the sequence, so each CP rank's router
      sees S/CP tokens. Unlike SP there is no gather back inside the router, so
      this reduction is unconditional.
    - DP / Sharding: different ranks process different micro-batches.

    EP is deliberately absent: all EP ranks see the same router scores.

    Communication order (TP -> CP -> DP -> Sharding) is irrelevant
    mathematically, since summation commutes.
    """
    try:
        from paddle.distributed import fleet

        from paddlefleet.parallel_state import get_context_parallel_group

        if not hasattr(fleet, "_hcg"):
            return None, None, None, None
        hcg = fleet.get_hybrid_communicate_group()

        def _needs_reduce(group):
            return group if group is not None and group.nranks > 1 else None

        return (
            _needs_reduce(hcg.get_model_parallel_group()),
            _needs_reduce(get_context_parallel_group()),
            _needs_reduce(hcg.get_data_parallel_group()),
            _needs_reduce(hcg.get_sharding_parallel_group()),
        )
    except Exception:
        return None, None, None, None


class MoEQuantileBalancingCallback:
    """Quantile Balancing callback for MoE load balancing.

    After each optimizer step, this callback:
      1. Collects qb_histogram from all QB-enabled router layers.
      2. All-reduces histograms across TP + CP + DP + Sharding ranks to aggregate
         statistics from all tokens in the global batch:
         - TP: When SP+EP>1, each TP rank only sees S/TP tokens (moe_layer
           skips AllGather when EP>1). Must reduce across TP to merge SP shards.
         - CP: Context Parallel always splits the sequence, so each CP rank's
           router sees S/CP tokens. Reduced unconditionally.
         - DP: Different DP ranks process different micro-batches.
         - Sharding: Same as DP (different data slices).
         - EP: NOT reduced (all EP ranks see the same router scores).
      3. Recovers the (k/n)-quantile from the merged histogram via cumsum + interpolation.
      4. Assigns the recovered quantile as the new expert bias (zero-mean normalized).
      5. Resets histograms and updates binning range for the next step.
    """

    def on_optimizer_end(self, args=None, state=None, control=None, **kwargs):
        """Called after optimizer.step() -- the main QB update entry point."""
        model = kwargs.get("model")
        if model is None:
            return

        # Collect all QB-enabled router layers
        layers = []

        def _collect(layer):
            from paddlefleet.transformer.moe.moe_router import StandardMoERouter

            if (
                isinstance(layer, StandardMoERouter)
                and layer.topk_method == "quantile_balancing"
            ):
                layers.append(layer)

        model.apply(_collect)
        if not layers:
            return

        # Determine communication groups
        tp_group, cp_group, dp_group, sd_group = _try_get_comm_groups()

        for layer in layers:
            self._update_single_layer(
                layer, tp_group, cp_group, dp_group, sd_group
            )

    def _update_single_layer(
        self, layer, tp_group, cp_group, dp_group, sd_group
    ):
        """Run QB recovery for a single router layer."""
        histogram = layer.qb_histogram  # [E, B], int32
        E, B = histogram.shape
        k = layer.num_experts_per_tok
        n = layer.num_experts

        # --- Step 1: All-reduce histogram across all ranks that see different tokens ---
        # When SP+EP>1: each TP rank only sees S/TP tokens (SP splits the sequence,
        # and moe_layer skips AllGather when EP>1). Must reduce across TP to get
        # the full sequence's histogram.
        # CP always splits the sequence, so each CP rank's router sees S/CP tokens
        # and the reduction is unconditional.
        # DP/Sharding ranks see different micro-batches, so also reduce there.
        hist_float = histogram.cast(paddle.float32)
        if (
            tp_group is not None
            and layer.config.sequence_parallel
            and layer.config.expert_model_parallel_size > 1
        ):
            dist.all_reduce(hist_float, group=tp_group)
        if cp_group is not None:
            dist.all_reduce(hist_float, group=cp_group)
        if dp_group is not None:
            dist.all_reduce(hist_float, group=dp_group)
        if sd_group is not None:
            dist.all_reduce(hist_float, group=sd_group)
        hist_global = hist_float.cast(paddle.int64)  # [E, B]

        # --- Step 2: Compute total token count and target quantile ---
        total_per_expert = hist_global.sum(axis=1)  # [E]

        # Guard: if no data was accumulated (e.g., callback called before first forward),
        # skip the update entirely.
        total_sum = int(total_per_expert.sum().item())
        if total_sum == 0:
            layer.qb_histogram.zero_()
            return

        # q = total_tokens * k / n for each expert (should be same across experts)
        # Use per-expert total for robustness (all should equal m, but guard against edge cases)
        q_target = (total_per_expert.cast(paddle.float64) * k / n).cast(
            paddle.int64
        )  # [E]
        # Ensure q_target >= 1 to avoid degenerate search
        q_target = paddle.clip(q_target, min=1)

        # --- Step 3: Cumulative sum to find quantile bin ---
        cumsum = hist_global.cumsum(axis=1)  # [E, B]

        # For each expert, find first bin where cumsum > q (strictly greater).
        # This finds the (q+1)-th smallest required_bias, which equals
        # -(the (q+1)-th largest margin), matching the QB formula:
        #   b_hat_j = -(margin's (q+1)-th largest) = req_bias's (q+1)-th smallest
        # Using strict > because cumsum >= q only gives the q-th smallest (off-by-one).
        mask = cumsum > q_target.unsqueeze(1)  # [E, B]
        # argmax on bool gives first True index; if all False (degenerate),
        # argmax returns 0 which maps to b_min -- acceptable fallback.
        # But to be safe, ensure at least the last bin satisfies (since cumsum[-1] = total >= q in normal cases).
        # If somehow q > total (shouldn't happen after clip), force last bin.
        has_any_true = mask.any(axis=1)  # [E]
        beta = mask.cast(paddle.int64).argmax(axis=1)  # [E]
        # For experts where no bin satisfies, use last bin (B-1)
        beta = paddle.where(has_any_true, beta, paddle.full_like(beta, B - 1))

        # --- Step 4: Linear interpolation within the bin ---
        b_min = layer.qb_bin_min
        b_max = layer.qb_bin_max
        total_range = b_max - b_min
        if total_range < 1e-8:
            total_range = 2.0  # degenerate: bias all-zero on first step
        bin_width = total_range / B

        # c_j = cumsum at (beta_j - 1), clamped
        beta_minus1 = paddle.clip(beta - 1, min=0)
        c = paddle.take_along_axis(
            cumsum, beta_minus1.unsqueeze(1), axis=1
        ).squeeze(1)  # [E]
        # When beta == 0, c should be 0
        c = c * (beta > 0).cast(c.dtype)

        # h_j = hist_global[e, beta_e]
        h = paddle.take_along_axis(
            hist_global, beta.unsqueeze(1), axis=1
        ).squeeze(1)  # [E]
        h = paddle.clip(h, min=1)  # avoid division by zero

        # fraction within the bin
        q_float = q_target.cast(paddle.float64)
        c_float = c.cast(paddle.float64)
        h_float = h.cast(paddle.float64)
        fraction = paddle.clip((q_float - c_float) / h_float, min=0.0, max=1.0)

        # Recovered bias value
        b_hat = b_min + (beta.cast(paddle.float64) + fraction) * bin_width
        b_hat = b_hat.cast(paddle.float32)  # [E]

        # --- Step 5: Zero-mean normalization ---
        b_new = b_hat - b_hat.mean()

        # --- Step 6: Assign to layer ---
        with paddle.no_grad():
            if layer.e_score_correction_bias.ndim == 2:
                # [1, E] shape variant
                layer.e_score_correction_bias.set_value(b_new.unsqueeze(0))
            else:
                layer.e_score_correction_bias.set_value(b_new)

        # --- Step 7: Update binning range for next step ---
        new_min = float(b_new.min().item()) - 1.0
        new_max = float(b_new.max().item()) + 1.0
        layer.qb_bin_min = new_min
        layer.qb_bin_max = new_max

        # --- Step 8: Reset histogram and expert_usage ---
        layer.qb_histogram.zero_()
        if hasattr(layer, "expert_usage"):
            layer.expert_usage.zero_()
