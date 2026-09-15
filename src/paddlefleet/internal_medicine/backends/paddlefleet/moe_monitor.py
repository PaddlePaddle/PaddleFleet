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
MoE Monitor for PaddleFleet.

Monitors MoE router health and expert weight norms using paddle hooks.

PaddleFleet MoE structure:
- MoELayer.gate — varies by model:
    StandardMoEGate: (capacity, top_gate, top_idx, gates_masked, mask, token_priority, l_aux, l_zloss)
    Other gates may return different formats.
- MoELayer.experts (nn.LayerList) or .grouped_gemm_experts (GroupedMLPExpert)
- MoELayer.shared_experts (StandardMLPSharedExpert or None)

TODO: Gate output format is model-specific. When switching to a different model
family, verify that outputs[2] (top_idx) is still at index 2 in the gate's
return tuple. Currently adapted for StandardMoEGate (PaddleFormers).
"""

import logging
import math

import paddle
import paddle.distributed as dist
from paddle import nn

from .base import PaddleProbe
from .layer_discovery import get_decoder_layers, iter_monitor_layers

logger = logging.getLogger(__name__)


def _compute_router_entropy(probs):
    """Router entropy from probability distribution. probs: [tokens, experts]."""
    probs = probs.astype("float32").clip(min=1e-10)
    probs = probs / probs.sum(axis=-1, keepdim=True)
    entropy = -(probs * probs.log()).sum(axis=-1)
    return entropy.mean()


def _distribution_metrics(mass: paddle.Tensor) -> dict[str, paddle.Tensor]:
    """Summarize a non-negative expert-mass vector without leaving the GPU."""
    mass = mass.astype("float32").reshape([-1]).clip(min=0.0)
    total = mass.sum()
    fraction = mass / total.clip(min=1e-30)
    num_experts = int(fraction.shape[0])
    uniform = 1.0 / max(num_experts, 1)
    safe_fraction = fraction.clip(min=1e-12)
    entropy = -(fraction * safe_fraction.log()).sum()
    zero = total * 0.0
    has_mass = total > 0
    entropy_norm = (
        entropy / math.log(num_experts) if num_experts > 1 else zero + 1.0
    )
    minimum = fraction.min()
    metrics = {
        "cv": paddle.sqrt(((fraction - uniform) ** 2).mean()) / uniform,
        "entropy_norm": entropy_norm,
        "kl_uniform": (fraction * (safe_fraction * num_experts).log()).sum(),
        "max_frac": fraction.max(),
        "min_frac": minimum,
        "max_min_ratio": fraction.max() / minimum.clip(min=1e-12),
    }
    return {
        name: paddle.where(has_mass, value, zero)
        for name, value in metrics.items()
    }


def _assignment_mask(
    probabilities: paddle.Tensor,
    outputs,
    k: int,
) -> paddle.Tensor | None:
    """Return the router's actual hard assignment mask as ``[tokens, experts]``."""
    num_experts = int(probabilities.shape[-1])
    if (
        isinstance(outputs, tuple | list)
        and len(outputs) > 4
        and isinstance(outputs[4], paddle.Tensor)
    ):
        mask = outputs[4]
        if int(mask.shape[-1]) == num_experts:
            return (
                mask.detach()
                .astype("float32")
                .reshape([-1, num_experts])
                .clip(min=0.0, max=1.0)
            )

    if not (
        isinstance(outputs, tuple | list)
        and len(outputs) > 2
        and isinstance(outputs[2], paddle.Tensor)
    ):
        return None
    indices = outputs[2].detach().astype("int64").reshape([-1, k])
    valid = (indices >= 0) & (indices < num_experts)
    safe_indices = indices.clip(min=0, max=num_experts - 1)
    one_hot = paddle.nn.functional.one_hot(
        safe_indices, num_classes=num_experts
    ).astype("float32")
    return (
        (one_hot * valid.unsqueeze(-1).astype("float32"))
        .sum(axis=1)
        .clip(max=1.0)
    )


def _routing_margin(
    selection_scores: paddle.Tensor,
    assignment_mask: paddle.Tensor,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Return per-token margins and a mask for rows where margin is defined."""
    selected_mask = assignment_mask > 0
    valid = selected_mask.any(axis=-1) & (~selected_mask).any(axis=-1)
    selected = paddle.where(
        selected_mask,
        selection_scores,
        paddle.full_like(selection_scores, float("inf")),
    ).min(axis=-1)
    unselected = paddle.where(
        selected_mask,
        paddle.full_like(selection_scores, float("-inf")),
        selection_scores,
    ).max(axis=-1)
    margin = selected - unselected
    return paddle.where(valid, margin, paddle.zeros_like(margin)), valid


def _masked_margin_metrics(
    margin: paddle.Tensor,
    valid: paddle.Tensor,
) -> dict[str, paddle.Tensor]:
    """Reduce routing margins over valid tokens only, without D2H sync."""
    valid_f = valid.astype("float32")
    valid_count = valid_f.sum()
    safe_count = valid_count.clip(min=1.0)
    zero = paddle.zeros([], dtype=margin.dtype)
    has_valid = valid_count > 0

    masked_margin = paddle.where(valid, margin, paddle.zeros_like(margin))
    mean = masked_margin.sum() / safe_count

    # Keep a fixed-size GPU tensor: invalid rows sort behind all valid rows and
    # tensor-valued ranks select quantiles from only the valid prefix.
    sorted_margin = paddle.sort(
        paddle.where(valid, margin, paddle.full_like(margin, float("inf")))
    )
    count_i64 = valid.astype("int64").sum()
    safe_count_i64 = count_i64.clip(min=1)

    def quantile(q: float) -> paddle.Tensor:
        rank = (safe_count_i64.astype("float32") - 1.0) * q
        lower = paddle.floor(rank).astype("int64")
        upper = paddle.ceil(rank).astype("int64")
        lower_value = paddle.gather(sorted_margin, lower.reshape([1])).squeeze(
            0
        )
        upper_value = paddle.gather(sorted_margin, upper.reshape([1])).squeeze(
            0
        )
        value = lower_value + (upper_value - lower_value) * (
            rank - lower.astype("float32")
        )
        return paddle.where(has_valid, value, zero)

    return {
        "router_margin_valid_ratio": valid_f.mean(),
        "router_margin_mean": paddle.where(has_valid, mean, zero),
        "router_margin_min": paddle.where(has_valid, sorted_margin[0], zero),
        "router_margin_p10": quantile(0.10),
        "router_margin_p01": quantile(0.01),
    }


def _compute_bias_affinity_jaccard(
    top_idx_with_bias, gates_no_bias, k, n_group=1, topk_group=1
):
    """Compute mean Jaccard similarity between routing with and without correction_bias.

    Paddle port of megatron/moe_metrics.compute_bias_affinity_jaccard, extended with
    group-limited topk support for PaddleFleet's TopKRouter.

    Args:
        top_idx_with_bias: [tokens, k] — actual routing indices (with bias)
        gates_no_bias: [tokens, experts] — original gate scores (no bias)
        k: num_experts_per_tok
        n_group / topk_group: group-limited topk params

    Group score is the sum of the group's top-2 experts, mirroring a literal in both
    router paths (``moe_router.TopKRouter`` and ``moe_topk_fusion``'s ``m1 + m2``);
    a group max would pick different groups and read < 1 even with an all-zero bias.
    Follow the router if it ever parameterises that 2. ``min(2, group_size)`` is only
    a shape guard.

    Returns:
        mean Jaccard similarity (1 = identical routing, 0 = completely different)
    """
    num_tokens, num_experts = gates_no_bias.shape

    if n_group > 1 and topk_group >= 1:
        group_size = num_experts // n_group
        gates_reshaped = gates_no_bias.reshape(
            [num_tokens, n_group, group_size]
        )
        group_scores = gates_reshaped.topk(min(2, group_size), axis=-1)[0].sum(
            axis=-1
        )
        _, top_groups = paddle.topk(group_scores, topk_group, axis=-1)
        group_mask = paddle.zeros([num_tokens, n_group], dtype="int32")
        group_mask = group_mask.put_along_axis(
            top_groups, paddle.to_tensor(1, dtype="int32"), axis=1
        )
        group_mask = (
            group_mask.unsqueeze(-1)
            .expand([-1, -1, group_size])
            .reshape([num_tokens, num_experts])
        )
        masked_gates = gates_no_bias.clone()
        masked_gates = paddle.where(
            group_mask > 0,
            masked_gates,
            paddle.full_like(masked_gates, float("-inf")),
        )
        _, top_idx_no_bias = paddle.topk(masked_gates, k, axis=-1)
    else:
        _, top_idx_no_bias = paddle.topk(gates_no_bias, k, axis=-1)

    set_with = paddle.zeros([num_tokens, num_experts], dtype="int32")
    set_without = paddle.zeros([num_tokens, num_experts], dtype="int32")
    set_with = set_with.put_along_axis(
        top_idx_with_bias, paddle.to_tensor(1, dtype="int32"), axis=1
    )
    set_without = set_without.put_along_axis(
        top_idx_no_bias, paddle.to_tensor(1, dtype="int32"), axis=1
    )

    intersection = (set_with & set_without).astype("float32").sum()
    union = (set_with | set_without).astype("float32").sum()
    return intersection / union.clip(min=1.0)


def _per_expert_stacked_sumsq(w1, w2=None):
    """Per-expert sum of squares over stacked expert weights, fully vectorized.

    Returns the sum of squares rather than the norm: under the 'allgather' MoE
    dispatcher each expert is sharded along its intermediate dim, so the
    per-shard sums must be reduced across the EP group *before* the sqrt.
    """
    num_experts = w1.shape[0]
    sq = (w1.detach().astype("float32").reshape([num_experts, -1]) ** 2).sum(
        axis=-1
    )
    if w2 is not None:
        sq = sq + (
            w2.detach().astype("float32").reshape([num_experts, -1]) ** 2
        ).sum(axis=-1)
    return sq


def _intermediate_shard_group(experts):
    """EP group along which every expert's intermediate dim is sharded, or None.

    ``moe_layer`` builds the fused expert module with
    ``intermediate_size_per_partition = moe_intermediate_size // EP`` when
    ``moe_token_dispatcher_type == 'allgather'`` and EP > 1 — every rank then
    holds *all* experts but only a 1/EP slice of each. Without reducing, every
    weight-norm metric reads low by sqrt(EP) (and shared_routed_ratio high by
    the same factor) the moment that dispatcher is switched on.
    """
    if experts is None:
        return None
    local = getattr(experts, "intermediate_size_per_partition", None)
    full = getattr(
        getattr(experts, "config", None), "moe_intermediate_size", None
    )
    if not local or not full or local >= full:
        return None
    return getattr(experts, "ep_group", None)


def _module_sumsq(module):
    """Sum of squares over all parameters of a module. Returns a 0-dim GPU tensor or None."""
    sq = None
    for p in module.parameters():
        part = (p.detach().astype("float32") ** 2).sum()
        sq = part if sq is None else sq + part
    return sq


def _gram(weight):
    """Gram matrix of ``weight``, shaped ``[..., k, k]`` with ``k = min(m, n)``.

    ``W W^T`` or ``W^T W``, whichever is the smaller square, so the eigensolve
    below always runs on ``k x k``. bf16 params are cast to float32 because the
    eigensolver needs at least fp32.

    Returns a GPU tensor, or None when ``weight`` is missing, has fewer than two
    dims, or is empty. Emptiness is read off the static shape rather than from
    ``numel()``, which returns a 0-dim GPU tensor here and would turn the guard
    into a host sync on every call.
    """
    if weight is None:
        return None
    w = weight.detach().astype("float32")
    if len(w.shape) < 2 or 0 in w.shape:
        return None
    m, n = w.shape[-2], w.shape[-1]
    return (
        paddle.matmul(w, w, transpose_y=True)
        if m <= n
        else paddle.matmul(w, w, transpose_x=True)
    )


def _gram_singular_values(gram):
    """Singular values from a Gram matrix: ``sigma_i = sqrt(lambda_i)``.

    ``[..., k, k]`` in, ``[..., k]`` out. The order is unspecified (``eigvalsh``
    returns ascending); both metrics below are order-invariant. Returns a GPU
    tensor (no host sync), or None for None input.
    """
    if gram is None:
        return None
    return paddle.sqrt(paddle.linalg.eigvalsh(gram).clip(min=0.0))


def _singular_values(weight):
    """Singular values of a matrix or a batch of matrices, via the Gram spectrum.

    Accepts ``[m, n]`` or ``[..., m, n]`` and returns ``[..., min(m, n)]``.

    ``sigma_i = sqrt(lambda_i(W^T W))`` is the SVD spectrum, but obtained far more
    cheaply: on one EP rank's per-expert SwiGLU gate stack ([32, 512, 512]) a
    batched ``paddle.linalg.svdvals`` measures ~8.5 s against ~196 ms for the
    batched Gram + ``eigvalsh`` here, agreeing to ~1e-4 relative. At 18 layers
    per step the SVD path would cost ~150 s/step, so it is not usable at this
    granularity.
    """
    return _gram_singular_values(_gram(weight))


def _stable_rank(sigma):
    """Stable (numerical) rank, reduced over the trailing singular-value axis.

    ``srank(W) = ||W||_F^2 / ||W||_2^2 = sum_i(sigma_i^2) / max_i(sigma_i)^2``,
    living in ``[1, rank(W)]``: 1 means a single direction carries all the energy,
    higher means the energy spreads over more directions. Continuous, unlike the
    hard rank, so collapse shows up as a smooth decline.

    Dominated by ``sigma_max`` — pair it with :func:`_singular_value_entropy`,
    which weighs the whole spectrum. ``[..., k]`` in, ``[...]`` out.
    """
    sq = sigma * sigma
    return sq.sum(axis=-1) / sq.max(axis=-1).clip(min=1e-12)


def _singular_value_entropy(sigma):
    """Shannon entropy of the squared-singular-value distribution (alpha = 2).

    ``H = -sum_i(p_i log p_i)`` with ``p_i = sigma_i^2 / sum_j(sigma_j^2)``, in
    nats, over ``[0, log(k)]``: 0 means the spectrum collapsed onto one
    direction, ``log(k)`` means a flat spectrum (all directions used equally).
    ``[..., k]`` in, ``[...]`` out.

    ``sigma^2`` is the energy each direction carries, so this shares its
    distribution with :func:`_stable_rank` and the pair is read on one scale
    (``exp(H) >= srank`` always). The megatron-side ``hidden_spectral_entropy``
    uses the same normalization, but it reads the post-norm hidden states rather
    than a weight matrix, so matching units do not make them one quantity.

    The sum guard is ``1e-24`` rather than ``1e-12`` because squaring halves the
    exponent range that reaches it.
    """
    sq = sigma * sigma
    p = (sq / sq.sum(axis=-1, keepdim=True).clip(min=1e-24)).clip(min=1e-12)
    return -(p * p.log()).sum(axis=-1)


def _swiglu_gate_half(fc1_weight):
    """The SiLU-gated half of a fused SwiGLU fc1 weight.

    ``GroupedMLPExpert`` doubles the fc1 output width for the gated unit and its
    ``glu()`` chunks the activation in two along the last axis, applying SiLU to
    the FIRST chunk. With paddle's ``[..., in, out]`` weight layout the gate
    projection is therefore ``w[..., : out // 2]`` and the linear "up" projection
    is the second half.
    """
    return fc1_weight[..., : fc1_weight.shape[-1] // 2]


def _expert_fc1_weight(moe_layer):
    """Routed-expert fc1 (fused gate+up) weight with a leading expert dim, or None.

    Mirrors the layouts handled by ``_collect_expert_sumsq``: grouped-gemm
    ``weight1``, the fused ``up_gate_proj.weight`` ([num_experts, in, 2*inter]),
    and the non-fused ``LayerList`` of per-expert MLPs (stacked here).
    """
    ggm = getattr(moe_layer, "grouped_gemm_experts", None)
    if ggm is not None and hasattr(ggm, "weight1"):
        return ggm.weight1
    experts = getattr(moe_layer, "experts", None)
    if experts is None:
        return None
    if hasattr(experts, "up_gate_proj"):
        return experts.up_gate_proj.weight
    if isinstance(experts, (list, nn.LayerList)) or hasattr(
        experts, "__iter__"
    ):
        per_expert = [
            e.up_gate_proj.weight
            for e in experts
            if e is not None and hasattr(e, "up_gate_proj")
        ]
        if per_expert:
            return paddle.stack(per_expert)
    return None


def _norm_stats(norms):
    """mean/std/min/max stats from a ``[num_experts]`` per-expert norm tensor (GPU tensors)."""
    if norms is None or norms.numel() == 0:
        return {}
    return {
        "expert_norm_mean": norms.mean(),
        "expert_norm_std": norms.std()
        if norms.numel() > 1
        else paddle.zeros(()),
        "expert_norm_min": norms.min(),
        "expert_norm_max": norms.max(),
    }


def _act_stats(act, name_prefix):
    """L2-norm mean, abs-max, and mean of an activation tensor. Returns dict of 0-dim GPU tensors."""
    act_f32 = act.detach().astype("float32")
    flat = act_f32.reshape([-1, act_f32.shape[-1]])  # [tokens, hidden]
    return {
        f"{name_prefix}_norm": flat.norm(p=2, axis=-1).mean(),
        f"{name_prefix}_abs_max": flat.abs().max(),
        f"{name_prefix}_mean": flat.mean(),
    }


class PaddleMoEMonitor(PaddleProbe):
    METRIC_PREFIX = "moe_health"
    MAX_AGGREGATED = {
        "score_sum_max",
        "expert_norm_max",
        "expert_bias_max",
        "shared_act_abs_max",
        "routed_act_abs_max",
        "router_scalar_max",
        "expert_gate_stable_rank_max",
        "expert_gate_singular_entropy_max",
        "router_input_abs_max",
        "router_input_abs_p99",
        "assignment_load_max_frac",
        "assignment_load_max_min_ratio",
        "gate_mass_max_frac",
        "gate_mass_max_min_ratio",
        "card_tokens_max",
    }
    MIN_AGGREGATED = {
        "score_sum_min",
        "expert_norm_min",
        "expert_bias_min",
        "router_scalar_min",
        "expert_gate_stable_rank_min",
        "expert_gate_singular_entropy_min",
        "assignment_load_min_frac",
        "gate_mass_min_frac",
        "router_margin_min",
    }

    def __init__(
        self,
        log_per_layer=True,
        log_global=True,
        monitor_interval=1,
        verbose=False,
        exclude_families=None,
    ):
        super().__init__(
            exclude_families=exclude_families,
            log_per_layer=log_per_layer,
            log_global=log_global,
            monitor_interval=monitor_interval,
            verbose=verbose,
        )
        self._patched_gates = []
        self._patched_moe_layers = []
        self._patched_dispatch_layers = []
        self._expert_norm_layers = []
        self._shared_act_norm_cache: dict = {}  # layer_idx -> 0-dim GPU tensor
        self._routed_act_norm_cache: dict = {}  # layer_idx -> 0-dim GPU tensor
        # Per-card dispatched token counts, one entry per microbatch, reduced
        # over the EP group once per step (see _reduce_card_tokens_over_ep).
        self._card_tokens_seq: dict = {}  # layer_idx -> [0-dim GPU tensor, per microbatch]
        self._moe_group = None
        self._ep_size = 1

    def register_hooks(self, model: nn.Layer):
        try:
            from paddlefleet.parallel_state import (
                get_pipeline_model_parallel_rank,
            )

            self.pp_rank = get_pipeline_model_parallel_rank()
        except Exception:
            pass

        moe_layers = self._find_moe_layers(model)
        if not moe_layers:
            logger.warning("[PaddleMoEMonitor] No MoE layers found!")
            return
        if self.verbose:
            logger.info(
                f"[PaddleMoEMonitor] Found {len(moe_layers)} MoE layers."
            )

        self._init_moe_group(moe_layers)

        # Declare metric schema
        for layer_idx, moe_layer in moe_layers:
            gate_metrics = [
                "router_input_rms",
                "router_input_abs_max",
                "router_input_abs_p99",
                "router_entropy",
                "router_entropy_norm",
                "score_sum_mean",
                "score_sum_min",
                "score_sum_max",
                "router_margin_mean",
                "router_margin_min",
                "router_margin_p10",
                "router_margin_p01",
                "router_margin_valid_ratio",
            ]
            for prefix in ("assignment_load", "gate_mass"):
                gate_metrics += [
                    f"{prefix}_{suffix}"
                    for suffix in (
                        "cv",
                        "entropy_norm",
                        "kl_uniform",
                        "max_frac",
                        "min_frac",
                        "max_min_ratio",
                    )
                ]
            if hasattr(moe_layer, "gate") and hasattr(
                moe_layer.gate, "e_score_correction_bias"
            ):
                gate_metrics += [
                    "bias_affinity_jaccard",
                    "expert_bias_mean",
                    "expert_bias_std",
                    "expert_bias_max",
                    "expert_bias_min",
                ]
            if hasattr(moe_layer, "gate") and hasattr(
                moe_layer.gate, "routed_scaling_factor_param"
            ):
                gate_metrics += [
                    "router_scalar_mean",
                    "router_scalar_std",
                    "router_scalar_max",
                    "router_scalar_min",
                    "router_scalar_ratio",
                ]
            expert_metrics = [
                "expert_norm_mean",
                "expert_norm_std",
                "expert_norm_min",
                "expert_norm_max",
                "shared_expert_norm",
                "shared_routed_ratio",
                "expert_gate_stable_rank_mean",
                "expert_gate_stable_rank_min",
                "expert_gate_stable_rank_max",
                "expert_gate_singular_entropy_mean",
                "expert_gate_singular_entropy_min",
                "expert_gate_singular_entropy_max",
                "shared_gate_stable_rank",
                "shared_gate_singular_entropy",
            ]
            act_metrics = []
            if (
                hasattr(moe_layer, "shared_experts")
                and moe_layer.shared_experts is not None
            ):
                act_metrics += [
                    "shared_act_norm",
                    "shared_act_abs_max",
                    "shared_act_mean",
                ]
            if hasattr(moe_layer, "_post_routed_output"):
                act_metrics += [
                    "routed_act_norm",
                    "routed_act_abs_max",
                    "routed_act_mean",
                    "shared_routed_act_ratio",
                ]
            for m in gate_metrics + expert_metrics + act_metrics:
                self.declare_layer_metric(layer_idx, m)
            # Dispatched tokens per card. Declared for every MoE layer whose
            # dispatcher can report them; the values are filled at step() time
            # after one EP-group reduction, not from the hook.
            if self._dispatch_reports_tokens(moe_layer):
                for m in (
                    "card_tokens_max",
                    "card_tokens_mean",
                    "card_tokens_peak_mean_ratio",
                ):
                    self.declare_layer_metric(layer_idx, m)
            # Per-expert share curves: one metric element per routed expert.
            # num_experts must come from static config here — a hook-time
            # reduction would need a D2H sync to size the schema.
            num_experts = int(
                getattr(getattr(moe_layer, "gate", None), "num_experts", 0) or 0
            )
            if num_experts > 0:
                for m in ("expert_token_share", "expert_weight_share"):
                    self.declare_layer_vector(layer_idx, m, num_experts)

        self.allocate_buffers()

        self._expert_norm_layers = []
        for layer_idx, moe_layer in moe_layers:
            if hasattr(moe_layer, "gate"):
                self._patch_gate_cache(moe_layer.gate)
                hook = moe_layer.gate.register_forward_post_hook(
                    self._make_gate_hook(layer_idx, moe_layer)
                )
                self.hooks.append(hook)
            # Shared expert activation hook
            if (
                hasattr(moe_layer, "shared_experts")
                and moe_layer.shared_experts is not None
            ):
                hook = moe_layer.shared_experts.register_forward_post_hook(
                    self._make_shared_expert_hook(layer_idx)
                )
                self.hooks.append(hook)
            # Routed expert activation: patch _post_routed_output (called right
            # after routed experts, before adding shared output — no D2H).
            if hasattr(moe_layer, "_post_routed_output"):
                self._patch_post_routed_output(moe_layer, layer_idx)
            if self._dispatch_reports_tokens(moe_layer):
                self._patch_dispatch(moe_layer, layer_idx)
            # Expert weight norms are NOT collected from a forward hook: under
            # offline FP8 quant the bf16 expert weights are cleared at step
            # begin. collect_expert_norms() reads them before quant instead.
            self._expert_norm_layers.append((layer_idx, moe_layer))

        logger.info(
            f"[PaddleMoEMonitor] Registered {len(self.hooks)} gate hooks and "
            f"{len(self._expert_norm_layers)} expert-norm layers on {len(moe_layers)} MoE layers."
        )

    @staticmethod
    def _uses_split_routing(gate) -> bool:
        """True when the router scores on the sum of two gate projections.

        ``moe_split_feature_routing`` gives the router a second projection
        (``weight_1``) and routes on ``f(logits_0) + f(logits_1)``. Hash layers
        opt out (``use_split = split and not is_hash_layer`` in moe_router) and
        capture their scores through the ``_hash_routing`` patch instead.
        """
        return bool(
            getattr(gate, "moe_split_feature_routing", False)
        ) and not getattr(gate, "is_hash_layer", False)

    def _patch_gate_cache(self, gate):
        """Monkey-patch gate.gate_score_func to cache pre-bias gates."""
        if not hasattr(gate, "gate_score_func"):
            if self.verbose:
                logger.warning(
                    "[PaddleMoEMonitor] Gate has no gate_score_func; router metrics may be unavailable"
                )
            return
        if hasattr(gate, "_im_patched"):
            if self.verbose:
                logger.warning(
                    "[PaddleMoEMonitor] Gate is already patched; skipping duplicate patch"
                )
            return
        original_fn = gate.gate_score_func
        monitor = self

        def cached_gate_score_func(logits):
            result = original_fn(logits)
            if not monitor._should_monitor():
                gate._cached_gates = None
                return result
            scores = result.detach()
            # Split-feature routing calls this twice per forward and routes on the
            # SUM of the two views (moe_router: gates = f(logits_0) + f(logits_1)),
            # so overwriting would leave us with view 1 alone — half of the routing
            # signal. Accumulate instead. The gate post-hook clears _cached_gates
            # after every forward, so nothing leaks across forwards. Only the split
            # path accumulates, so a stray double call elsewhere stays visible
            # rather than being silently summed.
            if monitor._uses_split_routing(gate):
                previous = getattr(gate, "_cached_gates", None)
                if previous is not None and previous.shape == scores.shape:
                    scores = previous + scores
            gate._cached_gates = scores
            return result

        gate._im_original_gate_score_func = original_fn
        gate.gate_score_func = cached_gate_score_func

        # Also patch _hash_routing for hash-routed layers (DeepSeek V4+).
        # Hash layers return early from forward() without calling gate_score_func,
        # so we intercept _hash_routing to capture the scores computed there.
        if hasattr(gate, "_hash_routing"):
            original_hash_routing = gate._hash_routing

            def cached_hash_routing(logits, flat_ids):
                result = original_hash_routing(logits, flat_ids)
                if not monitor._should_monitor():
                    gate._cached_gates = None
                    return result

                # _hash_routing computes scores internally. Recompute the full
                # [N, num_experts] distribution so router metrics can use it.
                import paddle.nn.functional as F

                logits_fp32 = logits.cast("float32")
                scoring_func = getattr(gate, "scoring_func", "softmax")
                if scoring_func == "softmax":
                    scores = F.softmax(logits_fp32, axis=-1)
                elif scoring_func == "sigmoid":
                    scores = F.sigmoid(logits_fp32)
                elif scoring_func == "sqrtsoftplus":
                    scores = paddle.sqrt(F.softplus(logits_fp32) + 1e-20)
                else:
                    gate._cached_gates = None
                    return result
                gate._cached_gates = scores.detach()
                return result

            gate._im_original_hash_routing = original_hash_routing
            gate._hash_routing = cached_hash_routing

        gate._im_patched = True
        self._patched_gates.append(gate)

    def _make_shared_expert_hook(self, layer_idx: int):
        monitor = self

        def hook_fn(layer, inputs, outputs):
            if not layer.training or not monitor._should_monitor():
                return
            try:
                with paddle.no_grad():
                    # shared_experts returns (output, output_bias) or just output
                    act = (
                        outputs[0]
                        if isinstance(outputs, (tuple, list))
                        else outputs
                    )
                    stats = _act_stats(act, "shared_act")
                    for name, val in stats.items():
                        monitor.record_layer_metric(layer_idx, name, val)
                    monitor._shared_act_norm_cache[layer_idx] = stats[
                        "shared_act_norm"
                    ]
            except Exception as e:
                if monitor.verbose:
                    logger.error(
                        f"[PaddleMoEMonitor] shared_expert hook error layer {layer_idx}: {e}"
                    )

        return hook_fn

    def _patch_post_routed_output(self, moe_layer, layer_idx: int):
        """Replace moe_layer._post_routed_output to capture routed expert output."""
        original_fn = moe_layer._post_routed_output
        monitor = self

        def patched_post_routed_output(output):
            result = original_fn(output)
            if moe_layer.training and monitor._should_monitor():
                try:
                    with paddle.no_grad():
                        stats = _act_stats(result, "routed_act")
                        for name, val in stats.items():
                            monitor.record_layer_metric(layer_idx, name, val)
                        # ratio: shared_act_norm / routed_act_norm (filled after shared hook runs)
                        monitor._routed_act_norm_cache[layer_idx] = stats[
                            "routed_act_norm"
                        ]
                except Exception as e:
                    if monitor.verbose:
                        logger.error(
                            f"[PaddleMoEMonitor] _post_routed_output patch error layer {layer_idx}: {e}"
                        )
            return result

        moe_layer._im_original_post_routed_output = original_fn
        moe_layer._post_routed_output = patched_post_routed_output
        self._patched_moe_layers.append(moe_layer)

    @staticmethod
    def _dispatch_reports_tokens(moe_layer) -> bool:
        """True when this layer's dispatcher can report dispatched token counts."""
        dispatcher = getattr(moe_layer, "token_dispatcher", None)
        return hasattr(moe_layer, "dispatch") and hasattr(
            dispatcher, "get_dispatched_routing"
        )

    def _init_moe_group(self, moe_layers) -> None:
        """Cache the EP group the per-card reduction runs on.

        ``moe_layer.moe_group`` is the same group PaddleFleet's own
        ``log_moe_balance`` all-gathers over, so the reported peak/mean ratio
        keeps that metric's semantics: unbalance *within one EP group*, not
        across data-parallel replicas.
        """
        for _layer_idx, moe_layer in moe_layers:
            group = getattr(moe_layer, "moe_group", None)
            if group is None:
                continue
            self._moe_group = group
            try:
                from paddlefleet.utils import get_pg_size

                self._ep_size = int(get_pg_size(group))
            except Exception:
                self._ep_size = 1
            return

    def _patch_dispatch(self, moe_layer, layer_idx: int):
        """Read ``tokens_per_expert`` right after dispatch returns.

        A patch on ``dispatch`` rather than a hook: the count lives on the
        dispatcher as state, not in any module's forward return, and it is only
        populated once the a2a has completed.

        Hot path cost is one ``sum`` plus one ``add_`` — no D2H, and no
        collective. The EP reduction that turns this into a peak/mean ratio is
        batched across all layers into two ``all_reduce`` calls at step() time.
        """
        original_fn = moe_layer.dispatch
        monitor = self

        def patched_dispatch(*args, **kwargs):
            result = original_fn(*args, **kwargs)
            if moe_layer.training and monitor._should_monitor():
                try:
                    tokens_per_expert = (
                        moe_layer.token_dispatcher.get_dispatched_routing()[2]
                    )
                    if tokens_per_expert is not None:
                        with paddle.no_grad():
                            monitor._accumulate_card_tokens(
                                layer_idx, tokens_per_expert
                            )
                except Exception as e:
                    if monitor.verbose:
                        logger.error(
                            f"[PaddleMoEMonitor] dispatch patch error layer {layer_idx}: {e}"
                        )
            return result

        moe_layer._im_original_dispatch = original_fn
        moe_layer.dispatch = patched_dispatch
        self._patched_dispatch_layers.append(moe_layer)

    def _accumulate_card_tokens(
        self, layer_idx: int, tokens_per_expert
    ) -> None:
        """Record this microbatch's local dispatched token total for one layer.

        Kept per microbatch rather than summed: the reported peak is
        ``mean_over_microbatches(max_over_cards(...))``, and averaging the loads
        first would compute ``max(mean)``, which by convexity of ``max`` is a
        different (always smaller) number — see _reduce_card_tokens_over_ep.
        """
        if not isinstance(tokens_per_expert, paddle.Tensor):
            tokens_per_expert = paddle.to_tensor(
                tokens_per_expert, dtype="float32"
            )
        local = tokens_per_expert.detach().astype("float32").sum()
        self._card_tokens_seq.setdefault(layer_idx, []).append(local)

    def _reduce_card_tokens_over_ep(self) -> None:
        """Turn per-card totals into EP-group max / mean / peak-mean ratio.

        The reduction order matters and matches PaddleFleet's ``log_moe_balance``:
        each microbatch's peak is taken across cards first, and only then are the
        microbatches averaged. Averaging the loads first would report
        ``max_cards(mean_microbatches(load))``, which is ``<=`` this by convexity
        of ``max`` and equal only when the same card is hottest in every
        microbatch — measured at up to 8% low on a 12-layer EP=8 run.

        Still **two** ``all_reduce`` calls per step for the whole stack, not one
        per microbatch: the locals are stacked into one ``[layers, microbatches]``
        matrix, so per-microbatch peaks come out of a single elementwise MAX. The
        payload grows from ``L`` to ``L*M`` floats and stays under a kilobyte.
        The alternative shape — one collective per microbatch — would be 16x the
        launches for the same numbers; ``log_moe_balance`` pays one
        ``all_gather_object`` *plus* one D2H per layer per microbatch.

        Every rank of an EP group holds the same layers and runs the same number
        of microbatches, so the matrix has one shape across the group and the
        collective cannot mismatch. A layer that fired a different number of
        times is truncated to the shortest run rather than reshaping the matrix.
        """
        if not self._card_tokens_seq:
            return
        layer_ids = sorted(self._card_tokens_seq)
        width = min(len(self._card_tokens_seq[i]) for i in layer_ids)
        if width == 0:
            self._card_tokens_seq = {}
            return
        local = paddle.stack(
            [paddle.stack(self._card_tokens_seq[i][:width]) for i in layer_ids]
        )  # [layers, microbatches]
        peak_per_mb, mean_per_mb = local, local
        if self._moe_group is not None and self._ep_size > 1:
            import paddle.distributed as dist

            peak_per_mb = local.clone()
            dist.all_reduce(
                peak_per_mb, op=dist.ReduceOp.MAX, group=self._moe_group
            )
            total = local.clone()
            dist.all_reduce(total, op=dist.ReduceOp.SUM, group=self._moe_group)
            mean_per_mb = total / float(self._ep_size)
        peak, mean, ratio = self._card_token_stats(peak_per_mb, mean_per_mb)
        for pos, layer_idx in enumerate(layer_ids):
            self.record_layer_metric(layer_idx, "card_tokens_max", peak[pos])
            self.record_layer_metric(layer_idx, "card_tokens_mean", mean[pos])
            self.record_layer_metric(
                layer_idx, "card_tokens_peak_mean_ratio", ratio[pos]
            )
        self._card_tokens_seq = {}

    @staticmethod
    def _card_token_stats(
        peak_per_mb: paddle.Tensor, mean_per_mb: paddle.Tensor
    ):
        """``[layers, microbatches]`` EP-reduced pair → per-layer peak / mean / ratio.

        Split out so the reduction order is testable without a process group: the
        ratio is the *mean of per-microbatch ratios*, not the ratio of the means.
        """
        return (
            peak_per_mb.mean(axis=-1),
            mean_per_mb.mean(axis=-1),
            (peak_per_mb / mean_per_mb.clip(min=1e-6)).mean(axis=-1),
        )

    def _flush_act_ratio(self):
        """After both shared and routed hooks have run, record the norm ratio."""
        for layer_idx in list(self._routed_act_norm_cache):
            routed_norm = self._routed_act_norm_cache.pop(layer_idx)
            shared_norm = self._shared_act_norm_cache.pop(layer_idx, None)
            if shared_norm is not None:
                ratio = shared_norm / routed_norm.clip(min=1e-8)
                self.record_layer_metric(
                    layer_idx, "shared_routed_act_ratio", ratio
                )
        # clear any leftover shared-only entries
        self._shared_act_norm_cache.clear()

    def _find_moe_layers(self, model: nn.Layer) -> list[tuple[int, nn.Layer]]:
        def has_moe(layer):
            return (
                (hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"))
                or hasattr(layer, "moe")
                or hasattr(layer, "gate")
            )

        layers = get_decoder_layers(model)
        if layers is None:
            for _name, sublayer in model.named_sublayers():
                if sublayer.__class__.__name__ == "MoELayer":
                    layers = [] if layers is None else layers
                    layers.append(sublayer)
            if layers is None:
                return []

        monitor_layers = iter_monitor_layers(
            layers, has_moe, pp_rank=self.pp_rank
        )
        self.mark_mtp_layers(item.idx for item in monitor_layers if item.is_mtp)
        moe_layers = []
        for item in monitor_layers:
            layer = item.layer
            moe_module = None
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
                moe_module = layer.mlp
            elif hasattr(layer, "moe"):
                moe_module = layer.moe
            elif hasattr(layer, "gate"):
                moe_module = layer
            if moe_module is not None:
                moe_layers.append((item.idx, moe_module))
        return moe_layers

    def _make_gate_hook(self, layer_idx: int, moe_layer: nn.Layer):
        def hook_fn(layer, inputs, outputs):
            if not layer.training:
                if hasattr(layer, "_cached_gates"):
                    layer._cached_gates = None
                return
            if not self._should_monitor():
                if hasattr(layer, "_cached_gates"):
                    layer._cached_gates = None
                return
            try:
                with paddle.no_grad():
                    self._compute_gate_metrics(
                        layer_idx, layer, inputs, outputs, moe_layer
                    )
            except Exception as e:
                if self.verbose:
                    logger.error(
                        f"[PaddleMoEMonitor] Gate hook error layer {layer_idx}: {e}"
                    )
            finally:
                if hasattr(layer, "_cached_gates"):
                    layer._cached_gates = None

        return hook_fn

    def collect_expert_norms(self):
        """Compute per-layer expert weight norms for all monitored MoE layers."""
        if not self._buffers_allocated or not self._should_monitor():
            return
        pending = []
        shard_group = None
        for layer_idx, moe_layer in self._expert_norm_layers:
            try:
                with paddle.no_grad():
                    self._compute_gate_spectrum_metrics(layer_idx, moe_layer)
            except Exception as e:
                if self.verbose:
                    logger.error(
                        f"[PaddleMoEMonitor] gate-spectrum collect error layer {layer_idx}: {e}"
                    )
            try:
                with paddle.no_grad():
                    routed_sq, shared_sq, group = self._collect_expert_sumsq(
                        moe_layer
                    )
            except Exception as e:
                if self.verbose:
                    logger.error(
                        f"[PaddleMoEMonitor] expert-norm collect error layer {layer_idx}: {e}"
                    )
                continue
            if group is not None:
                shard_group = group
            pending.append((layer_idx, routed_sq, shared_sq, group is not None))
        if not pending:
            return
        with paddle.no_grad():
            # One collective for the whole model rather than one per layer: the
            # intermediate-dim shards only carry 1/EP of each expert, so the
            # sums must be reduced before the sqrt.
            sharded = [
                sq
                for _, sq, _, needs_reduce in pending
                if needs_reduce and sq is not None
            ]
            if sharded and shard_group is not None:
                sizes = [int(sq.shape[0]) for sq in sharded]
                flat = paddle.concat([sq.reshape([-1]) for sq in sharded])
                dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=shard_group)
                reduced = iter(paddle.split(flat, sizes))
                pending = [
                    (
                        layer_idx,
                        next(reduced)
                        if (needs_reduce and sq is not None)
                        else sq,
                        shared_sq,
                        needs_reduce,
                    )
                    for layer_idx, sq, shared_sq, needs_reduce in pending
                ]
            for layer_idx, routed_sq, shared_sq, _ in pending:
                self._record_expert_metrics(layer_idx, routed_sq, shared_sq)

    def _compute_gate_metrics(
        self, layer_idx, gate, inputs, outputs, moe_layer
    ):
        """Compute router metrics from gate forward output."""
        cached_gates = getattr(gate, "_cached_gates", None)
        k = getattr(gate, "num_experts_per_tok", None)

        if cached_gates is None:
            if self.verbose:
                logger.warning(
                    f"[PaddleMoEMonitor] layer {layer_idx}: _cached_gates is None, gate patch may not work"
                )
            return

        cached_gates = cached_gates.reshape([-1, cached_gates.shape[-1]])
        router_input = (
            inputs[0]
            if inputs and isinstance(inputs[0], paddle.Tensor)
            else None
        )
        if router_input is not None:
            router_input = router_input.detach().astype("float32")
            absolute = router_input.abs()
            self.record_layer_metric(
                layer_idx,
                "router_input_rms",
                paddle.sqrt(router_input.square().mean()),
            )
            self.record_layer_metric(
                layer_idx, "router_input_abs_max", absolute.max()
            )
            self.record_layer_metric(
                layer_idx,
                "router_input_abs_p99",
                paddle.quantile(absolute.reshape([-1]), 0.99),
            )

        router_entropy = _compute_router_entropy(cached_gates)
        self.record_layer_metric(layer_idx, "router_entropy", router_entropy)
        num_experts = int(cached_gates.shape[-1])
        entropy_norm = (
            router_entropy / math.log(num_experts)
            if num_experts > 1
            else router_entropy * 0.0 + 1.0
        )
        self.record_layer_metric(layer_idx, "router_entropy_norm", entropy_norm)
        if k is not None:
            k = int(k)
            topk_vals, _ = paddle.topk(cached_gates, k, axis=-1)
            score_sum = topk_vals.sum(axis=-1)
            self.record_layer_metric(
                layer_idx, "score_sum_mean", score_sum.mean()
            )
            self.record_layer_metric(
                layer_idx, "score_sum_min", score_sum.min()
            )
            self.record_layer_metric(
                layer_idx, "score_sum_max", score_sum.max()
            )

            assignment_mask = _assignment_mask(cached_gates, outputs, k)
            if assignment_mask is not None and int(
                assignment_mask.shape[0]
            ) == int(cached_gates.shape[0]):
                assignment = assignment_mask.sum(axis=0)
                selected_affinity = (
                    cached_gates.astype("float32") * assignment_mask
                )
                if bool(getattr(gate, "norm_topk_prob", False)):
                    selected_affinity = (
                        selected_affinity
                        / selected_affinity.sum(axis=-1, keepdim=True).clip(
                            min=1e-12
                        )
                    )
                gate_mass = selected_affinity.sum(axis=0)
                for prefix, mass in (
                    ("assignment_load", assignment),
                    ("gate_mass", gate_mass),
                ):
                    for name, value in _distribution_metrics(mass).items():
                        self.record_layer_metric(
                            layer_idx, f"{prefix}_{name}", value
                        )
                self._record_expert_shares(layer_idx, assignment, outputs)

                if (
                    k < num_experts
                    and int(getattr(gate, "n_group", 1)) == 1
                    and not bool(getattr(gate, "is_hash_layer", False))
                ):
                    selection_scores = cached_gates.astype("float32")
                    if hasattr(gate, "e_score_correction_bias"):
                        selection_scores = (
                            selection_scores
                            + gate.e_score_correction_bias.detach().astype(
                                "float32"
                            )
                        )
                    margin, valid_margin = _routing_margin(
                        selection_scores, assignment_mask
                    )
                    for name, value in _masked_margin_metrics(
                        margin, valid_margin
                    ).items():
                        self.record_layer_metric(layer_idx, name, value)

        if hasattr(gate, "e_score_correction_bias"):
            top_idx_with_bias = None
            if isinstance(outputs, tuple) and len(outputs) >= 3:
                top_idx_with_bias = outputs[2]
            if top_idx_with_bias is not None and k is not None:
                n_group = getattr(gate, "n_group", 1)
                topk_group = getattr(gate, "topk_group", 1)
                self.record_layer_metric(
                    layer_idx,
                    "bias_affinity_jaccard",
                    _compute_bias_affinity_jaccard(
                        top_idx_with_bias, cached_gates, k, n_group, topk_group
                    ),
                )
            bias = gate.e_score_correction_bias
            self.record_layer_metric(layer_idx, "expert_bias_mean", bias.mean())
            self.record_layer_metric(layer_idx, "expert_bias_std", bias.std())
            self.record_layer_metric(layer_idx, "expert_bias_max", bias.max())
            self.record_layer_metric(layer_idx, "expert_bias_min", bias.min())

        if hasattr(gate, "routed_scaling_factor_param"):
            scalar = gate.routed_scaling_factor_param.detach().astype(
                "float32"
            )  # [num_experts]
            self.record_layer_metric(
                layer_idx, "router_scalar_mean", scalar.mean()
            )
            self.record_layer_metric(
                layer_idx, "router_scalar_std", scalar.std()
            )
            self.record_layer_metric(
                layer_idx, "router_scalar_max", scalar.max()
            )
            self.record_layer_metric(
                layer_idx, "router_scalar_min", scalar.min()
            )
            self.record_layer_metric(
                layer_idx,
                "router_scalar_ratio",
                scalar.max() / scalar.min().clip(min=1e-8),
            )

    def _compute_gate_spectrum_metrics(self, layer_idx, moe_layer):
        """Spectrum health of the experts' SwiGLU gate projection.

        One batched Gram eigensolve per layer covers every local expert plus the
        shared expert. The per-expert stable ranks / entropies are reduced to
        mean/min/max so the key count stays per-layer: 256 experts x 18 layers
        would otherwise be 4608 series. Under expert parallelism each rank holds
        its own shard of experts; the ``_max`` / ``_min`` keys reduce correctly
        across ranks in ``training_logs.gather_and_aggregate``, and the mean is
        exact because the expert count divides evenly across the EP group.

        The shared expert rides along in the routed batch as a Gram matrix rather
        than as a gate matrix. cuSOLVER re-initializes its ``syevj`` workspace
        whenever the batch shape changes between calls, so alternating a
        ``[32, k, k]`` solve with a ``[k, k]`` one costs 1287 ms/layer against
        197 ms for the equivalent single ``[33, k, k]`` batch (measured in
        isolated processes on the 4B-A500M shapes). Batching the gate matrices
        directly cannot work here: this model's routed gate is ``[512, 512]``
        while the shared one is ``[1024, 512]``. Their Grams are both ``k x k``
        with the same ``k``, so they batch even when the sources do not.
        """
        fc1 = _expert_fc1_weight(moe_layer)
        shared = getattr(moe_layer, "shared_experts", None)
        shared_fc1 = getattr(
            getattr(shared, "up_gate_proj", None), "weight", None
        )
        routed_gram = _gram(_swiglu_gate_half(fc1)) if fc1 is not None else None
        shared_gram = (
            _gram(_swiglu_gate_half(shared_fc1))
            if shared_fc1 is not None
            else None
        )

        if (
            routed_gram is not None
            and shared_gram is not None
            and shared_gram.shape == routed_gram.shape[1:]
        ):
            sigma = _gram_singular_values(
                paddle.concat([routed_gram, shared_gram.unsqueeze(0)], axis=0)
            )
            routed_sigma, shared_sigma = sigma[:-1], sigma[-1]
        else:
            routed_sigma = _gram_singular_values(routed_gram)
            shared_sigma = _gram_singular_values(shared_gram)

        if routed_sigma is not None:
            for name, vals in (
                ("stable_rank", _stable_rank(routed_sigma)),
                ("singular_entropy", _singular_value_entropy(routed_sigma)),
            ):
                self.record_layer_metric(
                    layer_idx, f"expert_gate_{name}_mean", vals.mean()
                )
                self.record_layer_metric(
                    layer_idx, f"expert_gate_{name}_min", vals.min()
                )
                self.record_layer_metric(
                    layer_idx, f"expert_gate_{name}_max", vals.max()
                )

        if shared_sigma is not None:
            self.record_layer_metric(
                layer_idx, "shared_gate_stable_rank", _stable_rank(shared_sigma)
            )
            self.record_layer_metric(
                layer_idx,
                "shared_gate_singular_entropy",
                _singular_value_entropy(shared_sigma),
            )

    def _record_expert_shares(self, layer_idx, assignment, outputs):
        """Per-expert share of this layer's routed load, in percent.

        The per-layer ``assignment_load_*`` / ``gate_mass_*`` metrics answer
        *how* skewed a layer is; these answer *which* expert is responsible and
        whether it stays the same one across steps. One curve per expert, so a
        hot expert and a dead expert are both identifiable by index.

        - ``expert_token_share`` reuses the caller's ``assignment`` vector (the
          hard-assignment count already summed for ``assignment_load_*``), so it
          adds no kernel beyond the normalisation and is exactly consistent with
          it: ``max(curves) == assignment_load_max_frac * 100``.
        - ``expert_weight_share`` uses ``probs`` — the post-normalisation,
          post-``routed_scaling_factor`` combine weight each expert's output
          carries into the residual. That is deliberately *not* ``gate_mass``,
          which omits the scaling factor: with a learnable
          ``routed_scaling_factor_param`` the two diverge, and only ``probs``
          tracks the magnitude actually reaching the residual. It is a proxy for
          activation magnitude, not the thing itself — the true per-expert output
          norm is unobservable under ``moe_expert_fusion``, where the fused MoE
          node returns already-combined output with no per-expert segmentation.

        A perfectly balanced layer sits at ``100 / num_experts`` on both. Neither
        leaves the GPU and neither needs a hook-time collective: the cross-rank
        mean over DP ranks equals the global share, because every rank routes the
        same number of tokens.
        """
        self.record_layer_vector(
            layer_idx,
            "expert_token_share",
            assignment / assignment.sum().clip(min=1e-12) * 100.0,
        )
        probs = (
            outputs[3]
            if isinstance(outputs, tuple | list) and len(outputs) > 3
            else None
        )
        if isinstance(probs, paddle.Tensor):
            weights = (
                probs.detach()
                .astype("float32")
                .reshape([-1, probs.shape[-1]])
                .sum(axis=0)
            )
            self.record_layer_vector(
                layer_idx,
                "expert_weight_share",
                weights / weights.sum().clip(min=1e-12) * 100.0,
            )

    def _collect_expert_sumsq(self, moe_layer):
        """Per-expert / shared-expert sums of squares for one MoE layer.

        Returns ``(routed_sumsq[num_experts] | None, shared_sumsq | None,
        shard_group | None)``. Sqrt is deferred to ``_record_expert_metrics`` so
        an intermediate-sharded layout can be reduced across EP first.
        """
        routed_sq = None
        shard_group = None

        # grouped-gemm experts: weight1/weight2 are [num_experts, ...] blocks.
        if (
            hasattr(moe_layer, "grouped_gemm_experts")
            and moe_layer.grouped_gemm_experts is not None
        ):
            ggm = moe_layer.grouped_gemm_experts
            if hasattr(ggm, "weight1") and hasattr(ggm, "weight2"):
                routed_sq = _per_expert_stacked_sumsq(ggm.weight1, ggm.weight2)
                shard_group = _intermediate_shard_group(ggm)

        elif hasattr(moe_layer, "experts") and moe_layer.experts is not None:
            experts = moe_layer.experts
            # Fused-expert layout (moe_expert_fusion=True): self.experts is a
            # single module whose up_gate_proj/down_proj weights carry a leading
            # expert dim [num_experts, ...]. Vectorize over that dim.
            if hasattr(experts, "up_gate_proj") and hasattr(
                experts, "down_proj"
            ):
                routed_sq = _per_expert_stacked_sumsq(
                    experts.up_gate_proj.weight, experts.down_proj.weight
                )
                shard_group = _intermediate_shard_group(experts)
            elif isinstance(experts, (list, nn.LayerList)) or hasattr(
                experts, "__iter__"
            ):
                # Non-fused layout: LayerList of per-expert modules. One sum-sq
                # per expert (each is a small handful of params), then stack.
                per_expert = []
                for expert in experts:
                    if expert is None:
                        continue
                    sq = _module_sumsq(expert)
                    if sq is not None:
                        per_expert.append(sq)
                if per_expert:
                    routed_sq = paddle.stack(per_expert)

        shared_sq = None
        if (
            hasattr(moe_layer, "shared_experts")
            and moe_layer.shared_experts is not None
        ):
            # Shared experts are replicated, never intermediate-sharded
            # (moe_layer builds them with the full moe_shared_expert_intermediate_size),
            # so they stay out of the EP reduction.
            shared_sq = _module_sumsq(moe_layer.shared_experts)

        return routed_sq, shared_sq, shard_group

    def _record_expert_metrics(self, layer_idx, routed_sq, shared_sq):
        routed_norm_mean = None
        if routed_sq is not None:
            norm_stats = _norm_stats(paddle.sqrt(routed_sq))
            for name, val in norm_stats.items():
                self.record_layer_metric(layer_idx, name, val)
            routed_norm_mean = norm_stats.get("expert_norm_mean")

        if shared_sq is not None:
            shared_norm = paddle.sqrt(shared_sq)
            self.record_layer_metric(
                layer_idx, "shared_expert_norm", shared_norm
            )
            if routed_norm_mean is not None:
                # clip 防止除零（对齐 megatron compute_shared_routed_ratio），保持 GPU 张量无 D2H
                self.record_layer_metric(
                    layer_idx,
                    "shared_routed_ratio",
                    shared_norm / routed_norm_mean.clip(min=1e-8),
                )

    def remove_hooks(self):
        super().remove_hooks()
        for gate in self._patched_gates:
            original_fn = getattr(gate, "_im_original_gate_score_func", None)
            if original_fn is not None:
                gate.gate_score_func = original_fn
            original_hash_routing = getattr(
                gate, "_im_original_hash_routing", None
            )
            if original_hash_routing is not None:
                gate._hash_routing = original_hash_routing
            for attr in (
                "_im_original_gate_score_func",
                "_im_original_hash_routing",
                "_im_patched",
                "_cached_gates",
            ):
                if hasattr(gate, attr):
                    delattr(gate, attr)
        self._patched_gates = []
        for moe_layer in self._patched_moe_layers:
            original_fn = getattr(
                moe_layer, "_im_original_post_routed_output", None
            )
            if original_fn is not None:
                moe_layer._post_routed_output = original_fn
            for attr in ("_im_original_post_routed_output",):
                if hasattr(moe_layer, attr):
                    delattr(moe_layer, attr)
        self._patched_moe_layers = []
        for moe_layer in self._patched_dispatch_layers:
            original_fn = getattr(moe_layer, "_im_original_dispatch", None)
            if original_fn is not None:
                moe_layer.dispatch = original_fn
            if hasattr(moe_layer, "_im_original_dispatch"):
                delattr(moe_layer, "_im_original_dispatch")
        self._patched_dispatch_layers = []
        self._card_tokens_seq = {}
        self._expert_norm_layers = []
        self._shared_act_norm_cache.clear()
        self._routed_act_norm_cache.clear()

    def step(self):
        self._flush_act_ratio()
        self._reduce_card_tokens_over_ep()
        super().step()


def setup_moe_monitor(
    model,
    log_per_layer=True,
    log_global=True,
    monitor_interval=1,
    verbose=False,
    monitor_dict=None,
    exclude_families=None,
):
    monitor = PaddleMoEMonitor(
        exclude_families=exclude_families,
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
    )
    monitor.register_hooks(model)
    logger.info(
        f"[PaddleMoEMonitor] Setup complete. Monitoring {len(monitor.hooks)} hooks."
    )
    if monitor_dict is not None:
        monitor_dict["moe_health"] = monitor
    return model
