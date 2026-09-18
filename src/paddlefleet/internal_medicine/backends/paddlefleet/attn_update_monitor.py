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

"""Attention update monitoring for PaddleFleet.

QK-product increment monitor from "Mechanism-Driven Monitors for Preemptive
Detection of LLM Training Instability" (arXiv:2606.28116) Section 3.4.

With the query/key factors sampled at two steps ``t`` and ``t - delta``, writing
``dWq = Wq_t - Wq_base``, ``dWk = Wk_t - Wk_base`` and evaluating the
un-subscripted factors at the base point ``t - delta``:

    delta1 = Wq_t Wk_t^T - Wq_base Wk_base^T = delta2 + delta3
    delta2 = dWq Wk^T + Wq dWk^T        (first order)
    delta3 = dWq dWk^T                  (second order)

Both monitored terms factorise exactly (eq. 4) — ``delta2 = [dWq, Wq][Wk, dWk]^T``
with factors ``[d, 2*head_dim]``, ``delta3 = dWq dWk^T`` with factors
``[d, head_dim]`` — so when the factor width is below ``d`` their nonzero
singular values are those of the small core ``R_A R_B^T`` built from thin QR
factors and the ``d x d`` product is never materialised.

``delta1`` is deliberately not monitored: eq. (5) gives
``||delta2||_F = O(||W||_F ||dW||_F)`` against ``||delta3||_F = O(||dW||_F^2)``,
so the early-to-mid regime ``||dW||_F << ||W||_F`` makes ``delta1 ~= delta2``,
and re-measuring it would cost another full ``2*head_dim`` core. ``delta3`` is
kept because it is the only term living in Q/K update-*coupling* space. Its core
is half as wide, but that buys less than the ``r^3`` eigensolve scaling suggests:
measured on one H-series GPU at ``hidden=1024, head_dim=128`` over 18 layers,
delta2 alone is 138 ms, delta3 alone 72 ms (0.52x), and both bucketed into one
driver call 211 ms — so adding delta3 costs ~1.5x, not ~1.1x. The thin QRs are
``O(d r^2)`` and dominate the ``O(r^3)`` eigensolve at these shapes.

Per layer and per term this reports the Frobenius norm, the stable rank
``||.||_F^2 / ||.||_2^2`` and the singular-spectrum effective rank
``exp(-sum_i p_i log p_i)`` with ``p_i = sigma_i^alpha / sum_j sigma_j^alpha``
(alpha = 2, the value the paper reports as the sensitivity/noise trade-off).

All three metrics are functions of ``sigma^2`` only, so the spectrum comes from
``eigvalsh`` on the Gram matrix and every layer goes through one batched call.
The decomposition is ~99% of the cost; snapshotting, stacking and the two
matmuls together measure ~10 ms. Same-run comparison at
``d = 2*head_dim = 1024`` over 18 layers, alongside a live training job:
looped ``svd`` 2054 ms, looped Gram+``eigvalsh`` 420 ms, batched
Gram+``eigvalsh`` 277 ms. Absolute numbers move by ~2x with GPU contention, so
treat them as ratios.

The fault certificate is the deviation of these curves from a healthy baseline
run, not low rank on its own: healthy updates already carry low-rank structure
that LoRA/GaLore/Muon exploit. The paper's absolute lead times are measured on
plain MHA and do not transfer: Section 6 notes that for MLA/GQA/MQA the QK
operator is mediated by compression projections and shared heads, so detection
thresholds stay variant-specific even though the algebra below is exact.

``resolve_qk_factors`` recovers the per-head circuit from weights alone for four
layouts — DSv4-hybrid, MLA (content/NoPE half only), a fused ``qkv_proj``, and
independent q/k projections — and skips any layer it cannot read. Adding a
layout means adding one resolver to ``_LAYOUT_RESOLVERS``.
"""

import logging
import math
from collections import defaultdict

import paddle

from .base import PaddleProbe
from .layer_discovery import get_decoder_layers, iter_monitor_layers

logger = logging.getLogger(__name__)

# Spectral quantities computed per monitored term (Section 3.2).
_SPECTRUM_KEYS = ("norm", "stable_rank", "singular_spectrum")
# Terms of the bilinear decomposition that are actually monitored. delta1 is
# omitted on purpose: eq. (5) gives ||delta2||_F = O(||W||_F ||dW||_F) against
# ||delta3||_F = O(||dW||_F^2), so in the early-to-mid regime delta1 ~= delta2
# and it would cost a full 2*head_dim core to re-measure delta2.
TERM_NAMES = ("delta2", "delta3")
METRIC_NAMES = tuple(
    f"{term}_{key}" for term in TERM_NAMES for key in _SPECTRUM_KEYS
)

# Cap on the transient stacked product / Gram tensors of one batched call.
_MAX_BATCH_BYTES = 256 << 20


def _squared_singular_values(a, b):
    """``sigma^2`` of ``a @ b.T`` for ``[..., d, r]`` factor pairs.

    ``sigma^2`` are the eigenvalues of the Gram matrix, so one symmetric
    ``eigvalsh`` replaces the SVD. Squaring costs precision on singular values
    below ``sqrt(eps) * sigma_max``, but every metric here weights by
    ``sigma^2`` (alpha = 2), so those components carry no weight — the unit
    tests pin all three against a dense float64 SVD to <= 3e-5 relative.

    When ``r < d`` the thin-QR identity ``a b^T = Qa (Ra Rb^T) Qb^T`` shrinks
    the work to the ``[r, r]`` core and the ``[d, d]`` product is never formed;
    at ``r >= d`` the core is no smaller, so the two QRs would be pure overhead.
    Leading axes are batch axes, which is what keeps this to one kernel launch
    per chunk instead of one per layer.
    """
    d, r = a.shape[-2], a.shape[-1]
    if r < d:
        a = paddle.linalg.qr(a, mode="r")
        b = paddle.linalg.qr(b, mode="r")
    product = paddle.matmul(a, b, transpose_y=True)
    gram = paddle.matmul(product, product, transpose_y=True)
    return paddle.linalg.eigvalsh(gram).clip(min=0.0)


def _spectrum_metrics(squared, alpha=2.0):
    """Frobenius norm, stable rank and singular-spectrum effective rank.

    ``squared`` holds ``sigma^2`` along the last axis; leading axes are batch
    axes and are preserved in every returned tensor. Keys are the bare
    :data:`_SPECTRUM_KEYS`; the caller prefixes them with the term it measured.
    """
    total = paddle.sum(squared, axis=-1)
    weights = squared if alpha == 2.0 else squared ** (alpha / 2.0)
    p = (
        weights / paddle.sum(weights, axis=-1, keepdim=True).clip(min=1e-24)
    ).clip(min=1e-12)
    return {
        "norm": paddle.sqrt(total),
        "stable_rank": total / paddle.max(squared, axis=-1).clip(min=1e-24),
        "singular_spectrum": paddle.exp(-paddle.sum(p * p.log(), axis=-1)),
    }


def _spectrum_metrics_for_shape(pairs, alpha=2.0):
    """Metrics for equally-shaped ``(a, b)`` factor pairs, batched into few calls.

    Chunked so the transient stacked product and Gram stay near
    ``_MAX_BATCH_BYTES`` each; at ``d = 1024`` that is 64 pairs per call. The
    chunks are made equal-sized rather than "fill then remainder" because
    cuSOLVER re-initializes its eigensolver workspace whenever the batch size
    changes between calls, which a short trailing chunk would trigger every step.
    """
    d = pairs[0][0].shape[-2]
    budget = max(1, _MAX_BATCH_BYTES // (d * d * 4))
    rows = math.ceil(len(pairs) / math.ceil(len(pairs) / budget))
    chunks = [
        _spectrum_metrics(
            _squared_singular_values(
                paddle.stack([item[0] for item in pairs[start : start + rows]]),
                paddle.stack([item[1] for item in pairs[start : start + rows]]),
            ),
            alpha,
        )
        for start in range(0, len(pairs), rows)
    ]
    if len(chunks) == 1:
        return chunks[0]
    return {
        key: paddle.concat([chunk[key] for chunk in chunks])
        for key in _SPECTRUM_KEYS
    }


def _spectrum_metrics_over_pairs(pairs, alpha=2.0):
    """Metrics for ``(a, b)`` factor pairs of possibly differing shapes.

    Only equally-shaped pairs can share a batched call, and the pair list mixes
    shapes on two axes: a model may mix attention layouts (or head_dims) across
    layers, and delta3's core is ``head_dim`` wide against delta2's
    ``2*head_dim``. Pairs are therefore bucketed by shape and the per-bucket
    results scattered back into the caller's order.
    """
    buckets: dict[tuple, list[tuple[int, tuple]]] = defaultdict(list)
    for position, pair in enumerate(pairs):
        buckets[(tuple(pair[0].shape), tuple(pair[1].shape))].append(
            (position, pair)
        )
    if len(buckets) == 1:
        return _spectrum_metrics_for_shape(pairs, alpha)

    slots: dict[str, list] = {
        key: [None] * len(pairs) for key in _SPECTRUM_KEYS
    }
    for bucket in buckets.values():
        metrics = _spectrum_metrics_for_shape(
            [pair for _position, pair in bucket], alpha
        )
        for offset, (position, _pair) in enumerate(bucket):
            for key in _SPECTRUM_KEYS:
                slots[key][position] = metrics[key][offset]
    return {key: paddle.stack(slots[key]) for key in _SPECTRUM_KEYS}


def _weight(module):
    """``module.weight``, or None when the module is absent or weightless."""
    return getattr(module, "weight", None) if module is not None else None


def _weight_shape(module):
    weight = _weight(module)
    return None if weight is None else list(weight.shape)


def _fp32(weight):
    """Detached float32 view of a parameter, for use inside an expression."""
    return weight.detach().astype("float32")


def _frozen(matrix):
    """Independent copy of a snapshot matrix.

    ``detach()`` shares storage and ``astype()`` is a no-op when the dtype
    already matches, so a returned slice can alias the live parameter. The base
    point has to survive the next optimizer step, hence the explicit copy.
    """
    return matrix.clone()


def _scale_vector(module):
    """Learnable norm scale as a flat float32 vector, or None."""
    weight = _weight(module)
    return None if weight is None else _fp32(weight).reshape([-1])


def _fold_scale(matrix, scale, index=0):
    """Fold a norm scale into the columns of ``matrix`` ``[in, d]``.

    A per-head norm carries ``d`` entries and applies to every head; a per-layer
    norm covers all heads, so head ``index`` takes its own ``d``-wide slice.
    Anything else cannot be read as a diagonal on these columns and is dropped
    rather than guessed at — the scale sits inside the QK circuit, so folding
    the wrong slice would be worse than folding nothing.
    """
    if scale is None:
        return matrix
    width = matrix.shape[-1]
    if scale.shape[0] == width:
        return matrix * scale.reshape([1, -1])
    start = index * width
    if scale.shape[0] >= start + width:
        return matrix * scale[start : start + width].reshape([1, -1])
    return matrix


def _first_module(attn, names):
    """First attribute in ``names`` that carries a 2-D ``weight``."""
    for name in names:
        module = getattr(attn, name, None)
        shape = _weight_shape(module)
        if shape is not None and len(shape) == 2:
            return module
    return None


def _head_counts(attn):
    """``(query_heads, kv_heads)`` as seen by this rank; either may be None.

    Prefers the ``*_per_partition`` names so a TP-sharded module reports local
    counts, which is what the sharded weight widths agree with.
    """

    def first_int(names):
        for name in names:
            value = getattr(attn, name, None)
            if isinstance(value, int) and value > 0:
                return value
        return None

    heads = first_int(
        (
            "num_attention_heads_per_partition",
            "num_attention_heads",
            "n_local_heads",
        )
    )
    kv_heads = first_int(
        (
            "num_query_groups_per_partition",
            "num_key_value_heads",
            "num_query_groups",
        )
    )
    return heads, kv_heads


def _resolve_dsv4_hybrid(attn):
    """``hidden -> q_down -> q_layernorm -> q_up`` with a single shared KV head.

    K is ``hidden -> kv_proj -> kv_layernorm``, so the per-head circuit is
    ``(Wqd diag(gq) Wqu_h) (Wkv diag(gk))^T``.
    """
    q_down = getattr(attn, "linear_q_down_proj", None)
    q_up = getattr(attn, "linear_q_up_proj", None)
    kv = getattr(attn, "linear_kv_proj", None)
    shapes = [_weight_shape(module) for module in (q_down, q_up, kv)]
    if any(shape is None or len(shape) != 2 for shape in shapes):
        return None
    if shapes[0][-1] != shapes[1][0]:
        return None
    head_dim = shapes[2][-1]
    if head_dim < 1 or shapes[1][-1] % head_dim != 0:
        return None
    return {
        "kind": "dsv4_hybrid",
        "q_down": q_down,
        "q_up": q_up,
        "kv": kv,
        "q_layernorm": getattr(attn, "q_layernorm", None),
        "kv_layernorm": getattr(attn, "kv_layernorm", None),
        "head_dim": head_dim,
        "num_heads": shapes[1][-1] // head_dim,
        "num_kv_heads": 1,
        "heads_per_group": shapes[1][-1] // head_dim,
    }


def _resolve_mla(attn):
    """MLA latent compression on both sides, with a RoPE/NoPE split of head_dim.

    Q is ``q_proj`` or ``q_a_proj -> q_a_layernorm -> q_b_proj``; K reaches the
    hidden state only through ``kv_a_proj_with_mqa -> kv_a_layernorm ->
    kv_b_proj``. Only the NoPE half of each head participates in the content
    term, so ``head_dim`` here is ``qk_nope_head_dim``.
    """
    kv_a = getattr(attn, "kv_a_proj_with_mqa", None)
    kv_b = getattr(attn, "kv_b_proj", None)
    q_b = getattr(attn, "q_b_proj", None)
    q_a = getattr(attn, "q_a_proj", None) if q_b is not None else None
    if q_b is None:
        q_b = getattr(attn, "q_proj", None)
    kv_a_shape, kv_b_shape, q_shape = (
        _weight_shape(m) for m in (kv_a, kv_b, q_b)
    )
    if any(
        shape is None or len(shape) != 2
        for shape in (kv_a_shape, kv_b_shape, q_shape)
    ):
        return None
    q_a_shape = _weight_shape(q_a)
    if q_a is not None and (q_a_shape is None or q_a_shape[-1] != q_shape[0]):
        return None
    kv_lora_rank = kv_b_shape[0]
    if not 0 < kv_lora_rank <= kv_a_shape[-1]:
        return None
    heads, _ = _head_counts(attn)
    if heads is None or q_shape[-1] % heads or kv_b_shape[-1] % heads:
        return None
    q_head_dim = q_shape[-1] // heads
    kv_head_width = kv_b_shape[-1] // heads
    nope = getattr(attn, "qk_nope_head_dim", None)
    if not isinstance(nope, int) or nope < 1:
        rope = getattr(attn, "qk_rope_head_dim", None)
        if not isinstance(rope, int) or rope < 0:
            rope = kv_a_shape[-1] - kv_lora_rank
        nope = q_head_dim - rope
    if not 0 < nope <= min(q_head_dim, kv_head_width):
        return None
    return {
        "kind": "mla",
        "q_a": q_a,
        "q_a_layernorm": getattr(attn, "q_a_layernorm", None),
        "q_b": q_b,
        "kv_a": kv_a,
        "kv_a_layernorm": getattr(attn, "kv_a_layernorm", None),
        "kv_b": kv_b,
        "kv_lora_rank": kv_lora_rank,
        "q_head_dim": q_head_dim,
        "kv_head_width": kv_head_width,
        "head_dim": nope,
        "num_heads": heads,
        "num_kv_heads": heads,
        "heads_per_group": 1,
    }


def _resolve_fused_qkv(attn):
    """One ``qkv_proj`` whose output is grouped per KV head as ``Q|(gate)|K|V``.

    The group arithmetic is verified against the real weight width before the
    layout is accepted, so a stack that packs qkv differently is rejected
    instead of being sliced wrongly.
    """
    qkv = _first_module(attn, ("qkv_proj", "linear_qkv"))
    shape = _weight_shape(qkv)
    if shape is None:
        return None
    heads, kv_heads = _head_counts(attn)
    if heads is None:
        return None
    kv_heads = heads if kv_heads is None else kv_heads
    if kv_heads < 1 or heads % kv_heads:
        return None
    head_dim = getattr(attn, "hidden_size_per_attention_head", None)
    if not isinstance(head_dim, int) or head_dim < 1:
        return None
    value_head_dim = getattr(attn, "value_hidden_size_per_attention_head", None)
    if not isinstance(value_head_dim, int) or value_head_dim < 1:
        value_head_dim = head_dim
    heads_per_group = heads // kv_heads
    q_dim = heads_per_group * head_dim
    gate_dim = (
        heads_per_group * value_head_dim
        if getattr(attn, "gated_attention", False)
        else 0
    )
    group_dim = q_dim + gate_dim + head_dim + value_head_dim
    if kv_heads * group_dim != shape[-1]:
        return None
    return {
        "kind": "fused_qkv",
        "qkv": qkv,
        "q_norm": getattr(attn, "q_norm", None),
        "k_norm": getattr(attn, "k_norm", None),
        "head_dim": head_dim,
        "num_heads": heads,
        "num_kv_heads": kv_heads,
        "heads_per_group": heads_per_group,
        "q_dim": q_dim,
        "gate_dim": gate_dim,
        "group_dim": group_dim,
    }


def _resolve_split_qk(attn):
    """Independent Q and K projections, GQA/MQA included.

    Head counts come from the weight widths rather than the module attributes so
    a TP-sharded projection reports the heads this rank actually holds.
    """
    q = _first_module(attn, ("q_proj", "linear_q_proj", "wq"))
    k = _first_module(attn, ("k_proj", "linear_k_proj", "shared_kv_proj", "wk"))
    q_shape, k_shape = _weight_shape(q), _weight_shape(k)
    if q_shape is None or k_shape is None or q_shape[0] != k_shape[0]:
        return None
    head_dim = getattr(attn, "hidden_size_per_attention_head", None)
    if not isinstance(head_dim, int) or head_dim < 1:
        heads, _ = _head_counts(attn)
        if heads is None or q_shape[-1] % heads:
            return None
        head_dim = q_shape[-1] // heads
    if head_dim < 1 or q_shape[-1] % head_dim or k_shape[-1] % head_dim:
        return None
    heads = q_shape[-1] // head_dim
    kv_heads = k_shape[-1] // head_dim
    if kv_heads < 1 or heads % kv_heads:
        return None
    return {
        "kind": "split_qk",
        "q": q,
        "k": k,
        "q_norm": getattr(attn, "q_norm", None),
        "k_norm": getattr(attn, "k_norm", None),
        "head_dim": head_dim,
        "num_heads": heads,
        "num_kv_heads": kv_heads,
        "heads_per_group": heads // kv_heads,
    }


# Most specific layout first: MLA and DSv4-hybrid both expose projections whose
# names overlap with the generic ones, so the generic resolvers must come last.
_LAYOUT_RESOLVERS = (
    _resolve_dsv4_hybrid,
    _resolve_mla,
    _resolve_fused_qkv,
    _resolve_split_qk,
)


def resolve_qk_factors(attn):
    """Query/key factors of one attention module, or None if unrecognised.

    Tries every known layout in :data:`_LAYOUT_RESOLVERS` and returns a dict
    describing the winner: ``kind``, the projection *modules*, ``head_dim`` (the
    width of the content circuit), ``num_heads`` / ``num_kv_heads`` /
    ``heads_per_group``, plus whatever slicing arithmetic that layout needs.

    Norm scales sitting between the projections are folded into the weights
    because they are inside the QK circuit; the input-dependent ``1/rms`` factor
    is not a weight and is left out.

    The *modules* are captured rather than their ``weight`` tensors so a weight
    object swapped out from under us (master-weight sync, quant callbacks) is
    still picked up on the next read.
    """
    for resolver in _LAYOUT_RESOLVERS:
        factors = resolver(attn)
        if factors is not None:
            return factors
    return None


def effective_wq(factors, head):
    """Effective ``[hidden, head_dim]`` query matrix of one head."""
    kind = factors["kind"]
    head_dim = factors["head_dim"]
    if kind == "dsv4_hybrid":
        latent = _fold_scale(
            _fp32(_weight(factors["q_down"])),
            _scale_vector(factors["q_layernorm"]),
        )
        start = head * head_dim
        return _frozen(
            latent
            @ _fp32(_weight(factors["q_up"]))[:, start : start + head_dim]
        )
    if kind == "mla":
        # Slice the head out of the up-projection before composing: the full
        # product would be [hidden, num_heads * q_head_dim], which is huge.
        start = head * factors["q_head_dim"]
        cols = _fp32(_weight(factors["q_b"]))[:, start : start + head_dim]
        q_a = factors["q_a"]
        if q_a is None:
            return _frozen(cols)
        latent = _fold_scale(
            _fp32(_weight(q_a)), _scale_vector(factors["q_a_layernorm"])
        )
        return _frozen(latent @ cols)
    if kind == "fused_qkv":
        group, within = divmod(head, factors["heads_per_group"])
        start = group * factors["group_dim"] + within * head_dim
        block = _fp32(_weight(factors["qkv"]))[:, start : start + head_dim]
    else:
        start = head * head_dim
        block = _fp32(_weight(factors["q"]))[:, start : start + head_dim]
    return _frozen(_fold_scale(block, _scale_vector(factors["q_norm"]), head))


def effective_wk(factors, head=0):
    """Effective ``[hidden, head_dim]`` key matrix for the group of ``head``."""
    kind = factors["kind"]
    head_dim = factors["head_dim"]
    if kind == "dsv4_hybrid":
        # One shared KV head, so every query head sees the same key matrix.
        return _frozen(
            _fold_scale(
                _fp32(_weight(factors["kv"])),
                _scale_vector(factors["kv_layernorm"]),
            )
        )
    if kind == "mla":
        start = head * factors["kv_head_width"]
        cols = _fp32(_weight(factors["kv_b"]))[:, start : start + head_dim]
        latent = _fold_scale(
            _fp32(_weight(factors["kv_a"]))[:, : factors["kv_lora_rank"]],
            _scale_vector(factors["kv_a_layernorm"]),
        )
        return _frozen(latent @ cols)
    group = head // factors["heads_per_group"]
    if kind == "fused_qkv":
        start = (
            group * factors["group_dim"]
            + factors["q_dim"]
            + factors["gate_dim"]
        )
        block = _fp32(_weight(factors["qkv"]))[:, start : start + head_dim]
    else:
        start = group * head_dim
        block = _fp32(_weight(factors["k"]))[:, start : start + head_dim]
    return _frozen(_fold_scale(block, _scale_vector(factors["k_norm"]), group))


class PaddleAttnUpdateMonitor(PaddleProbe):
    """Tracks the QK-product increment terms ``delta2`` / ``delta3`` per layer.

    No forward hooks are needed: the monitor reads query/key weights at step
    boundaries and keeps the previous reading as the base point, so the sampling
    interval ``delta`` is this monitor's own interval — ``sample_interval`` when
    given, otherwise the shared ``monitor_interval``. Sampling independently is
    useful because the cost here is one eigensolve per layer per sample, unlike
    the hook-driven monitors whose cost rides along with the forward pass.
    """

    METRIC_PREFIX = "attn_update"

    def __init__(
        self,
        log_per_layer=True,
        log_global=True,
        monitor_interval=1,
        verbose=False,
        num_heads_monitored=1,
        spectrum_alpha=2.0,
        sample_interval=None,
    ):
        interval = (
            monitor_interval
            if sample_interval is None
            else int(sample_interval)
        )
        if interval < 1:
            raise ValueError(
                f"sample_interval must be >= 1, got {sample_interval}"
            )
        super().__init__(
            log_per_layer=log_per_layer,
            log_global=log_global,
            monitor_interval=interval,
            verbose=verbose,
        )
        if int(num_heads_monitored) < 1:
            raise ValueError(
                f"num_heads_monitored must be >= 1, got {num_heads_monitored}"
            )
        self.num_heads_monitored = int(num_heads_monitored)
        self.spectrum_alpha = float(spectrum_alpha)
        # [(layer_idx, attn_type, factors)] and layer_idx -> (per-head Wq, Wk)
        self._layers = []
        self._snapshots = {}

    def register_hooks(self, model):
        try:
            from paddlefleet.parallel_state import (
                get_pipeline_model_parallel_rank,
            )

            self.pp_rank = get_pipeline_model_parallel_rank()
        except Exception:
            pass

        def has_attention(layer):
            return hasattr(layer, "self_attn") or hasattr(
                layer, "self_attention"
            )

        layers = get_decoder_layers(model)
        if layers is None:
            discovered = [
                m for _name, m in model.named_sublayers() if has_attention(m)
            ]
            layers = discovered or None
        if layers is None:
            logger.warning("[PaddleAttnUpdateMonitor] No decoder layers found!")
            return

        monitor_layers = iter_monitor_layers(
            layers, has_attention, pp_rank=self.pp_rank
        )
        self.mark_mtp_layers(item.idx for item in monitor_layers if item.is_mtp)

        self._layers = []
        for item in monitor_layers:
            attn = getattr(item.layer, "self_attn", None) or getattr(
                item.layer, "self_attention", None
            )
            factors = resolve_qk_factors(attn) if attn is not None else None
            if factors is None:
                if self.verbose:
                    logger.warning(
                        "[PaddleAttnUpdateMonitor] layer %s: unsupported QK layout, skipped",
                        item.idx,
                    )
                continue
            self._layers.append((item.idx, item.attn_type, factors))

        if not self._layers:
            logger.warning(
                "[PaddleAttnUpdateMonitor] No supported attention layouts found!"
            )
            return

        for layer_idx, attn_type, _factors in self._layers:
            for name in METRIC_NAMES:
                self.declare_layer_metric(layer_idx, name, attn_type=attn_type)
        self.allocate_buffers()

        logger.info(
            "[PaddleAttnUpdateMonitor] Tracking %s on %d layers (%s), %d head(s) per layer, alpha=%s, delta=%d steps.",
            "/".join(TERM_NAMES),
            len(self._layers),
            ", ".join(
                sorted(
                    {factors["kind"] for _idx, _type, factors in self._layers}
                )
            ),
            self.num_heads_monitored,
            self.spectrum_alpha,
            self.monitor_interval,
        )

    def _prepare_layer(self, layer_idx, factors):
        """Re-snapshot one layer and return ``{term: [(a, b), ...]}`` factor pairs.

        Empty on the first reading, which only establishes the base point
        ``t - delta``. Both terms come out of the same four snapshot matrices, so
        delta3 costs no extra weight reads — only an eigensolve on a ``head_dim``
        core against delta2's ``2*head_dim`` one.

        K is read per head rather than once per layer: under GQA/MQA several
        query heads share one key matrix, but under MLA each head has its own,
        and ``effective_wk`` resolves that per layout.
        """
        heads = min(self.num_heads_monitored, factors["num_heads"])
        wq_now = [effective_wq(factors, head) for head in range(heads)]
        wk_now = [effective_wk(factors, head) for head in range(heads)]

        previous = self._snapshots.get(layer_idx)
        self._snapshots[layer_idx] = (wq_now, wk_now)
        if previous is None:
            return {}

        wq_base, wk_base = previous
        shapes = [
            [m.shape for m in group]
            for group in (wq_base, wq_now, wk_base, wk_now)
        ]
        if shapes[0] != shapes[1] or shapes[2] != shapes[3]:
            if self.verbose:
                logger.warning(
                    "[PaddleAttnUpdateMonitor] layer %s: snapshot shape changed, rebasing",
                    layer_idx,
                )
            return {}

        delta2, delta3 = [], []
        for q_base, q_now, k_base, k_now in zip(
            wq_base, wq_now, wk_base, wk_now, strict=True
        ):
            dq, dk = q_now - q_base, k_now - k_base
            # eq. (4): delta2 = [dWq, Wq][Wk, dWk]^T and delta3 = dWq dWk^T.
            delta2.append(
                (
                    paddle.concat([dq, q_base], axis=-1),
                    paddle.concat([k_base, dk], axis=-1),
                )
            )
            delta3.append((dq, dk))
        return {"delta2": delta2, "delta3": delta3}

    def _record_all_layers(self):
        """Snapshot every layer, then take every term's spectrum in one batch."""
        pairs = []
        groups = []  # [(layer_idx, attn_type, term, head_count)] parallel to ``pairs``
        for layer_idx, attn_type, factors in self._layers:
            try:
                per_term = self._prepare_layer(layer_idx, factors)
            except Exception as exc:
                if self.verbose:
                    logger.error(
                        "[PaddleAttnUpdateMonitor] QK snapshot failed on layer %s: %s",
                        layer_idx,
                        exc,
                    )
                continue
            for term in TERM_NAMES:
                term_pairs = per_term.get(term)
                if term_pairs:
                    groups.append((layer_idx, attn_type, term, len(term_pairs)))
                    pairs.extend(term_pairs)

        if not pairs:
            return

        metrics = _spectrum_metrics_over_pairs(pairs, self.spectrum_alpha)
        offset = 0
        for layer_idx, attn_type, term, count in groups:
            for key in _SPECTRUM_KEYS:
                head_mean = metrics[key][offset : offset + count].mean()
                self.record_layer_metric(
                    layer_idx, f"{term}_{key}", head_mean, attn_type=attn_type
                )
            offset += count

    def step(self):
        if self._buffers_allocated and self._should_monitor():
            with paddle.no_grad():
                try:
                    self._record_all_layers()
                except Exception as exc:
                    if self.verbose:
                        logger.error(
                            "[PaddleAttnUpdateMonitor] QK increment failed: %s",
                            exc,
                        )
        super().step()

    def remove_hooks(self):
        super().remove_hooks()
        self._layers = []
        self._snapshots = {}


def setup_attn_update_monitor(
    model,
    log_per_layer=True,
    log_global=True,
    monitor_interval=1,
    verbose=False,
    monitor_dict=None,
    num_heads_monitored=1,
    spectrum_alpha=2.0,
    sample_interval=None,
):
    monitor = PaddleAttnUpdateMonitor(
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
        num_heads_monitored=num_heads_monitored,
        spectrum_alpha=spectrum_alpha,
        sample_interval=sample_interval,
    )
    monitor.register_hooks(model)
    logger.info(
        "[PaddleAttnUpdateMonitor] Setup complete on %d layers.",
        len(monitor._layers),
    )
    if monitor_dict is not None:
        monitor_dict["attn_update"] = monitor
    return model
