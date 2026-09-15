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

"""DSA (DeepSeek Sparse Attention) metric math for PaddleFleet.

Pure tensor functions so the semantics can be tested without a model. Every
function returns GPU 0-dim tensors; nothing here syncs to host.

All of it is computed on a **subsample of query rows** (see
:func:`sample_query_rows`). The indexer's own scoring is ``O(sq x sk x h x d)``
and the dense attention it is compared against is ``O(sq x sk x H x d)``, so
reproducing either at full sequence length would cost as much as the layer
itself. Rows are drawn from the region where the top-k rule actually selects
something -- below ``index_topk`` valid keys every causal position is selected
and every reading is trivially saturated.
"""

from __future__ import annotations

import paddle
import paddle.nn.functional as F

EPS = 1e-12

# Large finite additive mask instead of -inf: a row whose every entry is -inf
# comes back NaN from softmax, and `topk` returns garbage indices on -inf input
# (which is why `DSAIndexer.compute_index_scores` clips them).
NEG = 1e30

INDEXER_METRICS = (
    "index_q_rms",
    "index_q_abs_max",
    "index_k_rms",
    "index_k_abs_max",
    "index_weights_abs_max",
    "index_weights_neg_ratio",
)

SCORE_METRICS = (
    "score_selected_mean",
    "score_unselected_mean",
    "score_topk_mass",
)

SELECT_METRICS = (
    "select_ratio",
    "select_dist_mean",
    "select_local_ratio",
    "select_key_coverage",
)

MATCH_METRICS = (
    "attn_recall",
    "attn_recall_min",
    "attn_kl",
    "attn_top1_hit",
    "attn_sparse_entropy",
)

CROSSLAYER_METRICS = ("select_iou_prev",)

ALL_METRICS = (
    INDEXER_METRICS
    + SCORE_METRICS
    + SELECT_METRICS
    + MATCH_METRICS
    + CROSSLAYER_METRICS
)

# Keep in sync with the metric names: ``TrainingLogs`` picks the cross-rank
# reduction from the key suffix, so a max metric must end in ``_max``.
MAX_METRICS = (
    "index_q_abs_max",
    "index_k_abs_max",
    "index_weights_abs_max",
)
MIN_METRICS = ("attn_recall_min",)


def indexer_projection_stats(
    q: paddle.Tensor, k: paddle.Tensor, weights: paddle.Tensor
) -> dict:
    """Activation magnitudes of the three indexer projections.

    ``q`` ``[b, s, h, d]`` and ``k`` ``[b, s, d]`` are post-RoPE, post-Hadamard,
    so these are the tensors the scoring einsum actually consumes. ``weights``
    ``[b, s, h]`` has already absorbed ``softmax_scale`` and ``n_heads**-0.5``.

    ``index_weights_neg_ratio`` exists because ``weights_proj`` is unconstrained:
    the score is ``sum_h w_h * relu(q.k)``, so a negative ``w_h`` makes that
    head's evidence *suppress* a position. A ratio drifting toward 0.5 means the
    per-head importance has stopped being an importance.
    """
    q = q.astype("float32")
    k = k.astype("float32")
    w = weights.astype("float32")
    return {
        "index_q_rms": q.square().mean().sqrt(),
        "index_q_abs_max": q.abs().max(),
        "index_k_rms": k.square().mean().sqrt(),
        "index_k_abs_max": k.abs().max(),
        "index_weights_abs_max": w.abs().max(),
        "index_weights_neg_ratio": (w < 0).astype("float32").mean(),
    }


def sample_query_rows(
    seq_len_q: int, index_topk: int, row_samples: int, position_offset: int = 0
) -> paddle.Tensor:
    """Evenly spaced **local** query rows whose global position has real history.

    Rows with fewer than ``index_topk`` causally valid keys select all of them,
    so recall is 1 and the sparsity ratio is 1 by construction. The cut is on the
    *global* position, so under context parallel a rank whose slice starts past
    ``index_topk`` may use its whole slice while rank 0 skips its first rows.

    When even the last global row has less history than ``index_topk`` there is
    no such region and the second half of the slice is used instead, which at
    least biases toward the longest rows.
    """
    lo = index_topk - int(position_offset)
    if lo >= seq_len_q:
        lo = seq_len_q // 2
    lo = max(0, min(lo, seq_len_q - 1))
    count = min(int(row_samples), seq_len_q - lo)
    if count <= 1:
        return paddle.to_tensor([seq_len_q - 1], dtype="int64")
    return paddle.linspace(lo, seq_len_q - 1, count).astype("int64")


def index_scores_on_rows(
    q: paddle.Tensor,
    k: paddle.Tensor,
    weights: paddle.Tensor,
    rows: paddle.Tensor,
) -> paddle.Tensor:
    """``[b, R, sk]`` indexer row scores, transcribed from ``compute_index_scores``.

    Recomputed rather than read off the module: on the training path
    ``DSAttention`` hands the indexer outputs to the fused loss kernel and
    ``DSAIndexer.compute_index_scores`` never runs, so there is nothing to
    intercept. The formula is the same one, restricted to ``rows``.
    """
    q_rows = paddle.index_select(q, rows, axis=1).astype("float32")
    w_rows = paddle.index_select(weights, rows, axis=1).astype("float32")
    scores = paddle.einsum("brhd,btd->brht", q_rows, k.astype("float32"))
    return (w_rows.unsqueeze(-1) * F.relu(scores)).sum(axis=2)


def causal_geometry(
    rows: paddle.Tensor, seq_len_k: int
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """``(causal, dist)`` ``[R, sk]``: validity mask and look-back distance.

    ``rows`` must be **global** query positions and ``seq_len_k`` the global key
    length. Under context parallel the indexer key comes back all-gathered while
    the query rows are this rank's slice, so the caller adds ``position_offset``
    to the local row ids before getting here — comparing a local row id against a
    global key position would silently mask out most of the real history.
    """
    positions = paddle.arange(seq_len_k, dtype="int64")
    causal = (positions.unsqueeze(0) <= rows.unsqueeze(1)).astype("float32")
    dist = (rows.unsqueeze(1) - positions.unsqueeze(0)).astype(
        "float32"
    ) * causal
    return causal, dist


def selection_mask(
    index_scores: paddle.Tensor, causal: paddle.Tensor, index_topk: int
) -> paddle.Tensor:
    """``[b, R, sk]`` 1.0 where the indexer's top-k rule keeps a key."""
    masked = index_scores + (causal.unsqueeze(0) - 1.0) * NEG
    k = min(int(index_topk), int(index_scores.shape[-1]))
    idx = paddle.topk(masked, k=k, axis=-1)[1]
    sel = paddle.zeros_like(index_scores)
    sel = paddle.put_along_axis(
        sel, idx, paddle.ones_like(idx, dtype=sel.dtype), axis=-1
    )
    # Rows with fewer than k valid keys get padding picks; the causal mask is
    # what makes the count honest rather than always exactly k.
    return sel * causal.unsqueeze(0)


def selection_stats(
    sel: paddle.Tensor,
    causal: paddle.Tensor,
    dist: paddle.Tensor,
    local_window: int,
) -> dict:
    """Shape of the selected set: how sparse, how far back, how concentrated.

    Distances are reported as a fraction of the row's own causal length so the
    reading is comparable across rows and across sequence lengths.
    ``select_local_ratio`` is the degeneration guard: an indexer that has
    collapsed into a sliding window puts everything inside ``local_window``.
    """
    n_sel = sel.sum(axis=-1)
    valid = causal.sum(axis=-1).unsqueeze(0)
    dist_b = dist.unsqueeze(0)
    mean_dist = (sel * dist_b).sum(axis=-1) / (n_sel + EPS) / (valid + EPS)
    local = (sel * (dist_b < float(local_window)).astype(sel.dtype)).sum(
        axis=-1
    ) / (n_sel + EPS)
    # Union over the sampled rows: how much of the key axis any query looks at.
    covered = (sel.sum(axis=1) > 0).astype(sel.dtype).sum(axis=-1)
    return {
        "select_ratio": (n_sel / (valid + EPS)).mean(),
        "select_dist_mean": mean_dist.mean(),
        "select_local_ratio": local.mean(),
        "select_key_coverage": (
            covered / (causal.sum(axis=-1).max() + EPS)
        ).mean(),
    }


def score_stats(
    index_scores: paddle.Tensor, sel: paddle.Tensor, causal: paddle.Tensor
) -> dict:
    """Separation between kept and dropped scores, and the kept share of mass.

    ``index_scores`` can go negative (``weights_proj`` is unconstrained), and a
    ratio of signed sums is not a mass, so ``score_topk_mass`` clips at zero
    first to stay inside ``[0, 1]``. The two means are left unclipped: their gap
    is the signal, and clipping would hide a collapse into negative territory.
    """
    causal_b = causal.unsqueeze(0)
    unsel = causal_b - sel
    sel_mean = (index_scores * sel).sum(axis=-1) / (sel.sum(axis=-1) + EPS)
    unsel_mean = (index_scores * unsel).sum(axis=-1) / (
        unsel.sum(axis=-1) + EPS
    )
    positive = paddle.clip(index_scores, min=0.0)
    mass = (positive * sel).sum(axis=-1) / (
        (positive * causal_b).sum(axis=-1) + EPS
    )
    return {
        "score_selected_mean": sel_mean.mean(),
        "score_unselected_mean": unsel_mean.mean(),
        "score_topk_mass": mass.mean(),
    }


def dense_match_stats(
    q_rows: paddle.Tensor,
    key: paddle.Tensor,
    sel: paddle.Tensor,
    causal: paddle.Tensor,
    softmax_scale: float,
) -> dict:
    """How much of the dense attention the sparse selection keeps.

    ``q_rows`` ``[b, R, H, D]`` and ``key`` ``[b, sk, H, D]`` with heads already
    aligned and subsampled by the caller.

    ``attn_recall`` is the answer to "does the indexer pick the keys the trunk
    attention would have used": the dense probability mass that survives the
    mask. ``attn_kl`` is ``KL(p_sparse || p_dense)``; the two distributions
    differ only by renormalisation over the selected set, so per ``(head, row)``
    that KL is exactly ``-log(recall)``. The *reported* pair is not redundant
    though: the mean of ``-log`` is not the ``-log`` of the mean, and the gap
    between them is exactly what one badly served head costs -- which is the
    reason to watch the KL rather than only the recall.

    ``attn_top1_hit`` is the tail risk the averages hide: recall can sit at 0.95
    while the single largest logit is dropped.
    """
    logits = paddle.einsum(
        "brhd,bthd->bhrt", q_rows.astype("float32"), key.astype("float32")
    )
    logits = logits * float(softmax_scale) + (causal - 1.0) * NEG
    probs = F.softmax(logits, axis=-1)
    sel_b = sel.unsqueeze(1)
    recall = (probs * sel_b).sum(axis=-1)
    sparse = probs * sel_b / (recall.unsqueeze(-1) + EPS)
    top1 = logits.argmax(axis=-1, keepdim=True)
    hit = paddle.take_along_axis(sel_b.expand(probs.shape), top1, axis=-1)
    return {
        "attn_recall": recall.mean(),
        "attn_recall_min": recall.min(),
        "attn_kl": -paddle.log(paddle.clip(recall, min=1e-6)).mean(),
        "attn_top1_hit": hit.mean(),
        "attn_sparse_entropy": -(sparse * paddle.log(sparse + EPS))
        .sum(axis=-1)
        .mean(),
    }


def align_and_sample_heads(
    query: paddle.Tensor, key: paddle.Tensor, head_samples: int
) -> tuple[paddle.Tensor, paddle.Tensor] | None:
    """Pick at most ``head_samples`` query heads and their matching key heads.

    Key heads are selected by index rather than by materialising a repeated key
    tensor: under grouped KV a full ``repeat_interleave`` of
    ``[b, sk, n_kv, d]`` up to ``n_q`` heads is hundreds of MB at production
    sequence lengths, and every one of those heads would then be thrown away.

    Returns ``None`` when the head counts do not divide, which is the signal to
    skip the dense comparison rather than report a wrong one.
    """
    n_q, n_k = query.shape[2], key.shape[2]
    if n_k <= 0 or n_q % n_k != 0:
        return None
    count = min(int(head_samples), n_q)
    heads = (
        paddle.linspace(0, n_q - 1, count).astype("int64")
        if count < n_q
        else paddle.arange(n_q, dtype="int64")
    )
    return (
        paddle.index_select(query, heads, axis=2),
        paddle.index_select(key, heads // (n_q // n_k), axis=2),
    )


def selection_iou(sel: paddle.Tensor, prev_sel: paddle.Tensor) -> dict:
    """Per-row IoU against the previous layer's selection on the same rows.

    A stack whose layers all select the same keys is paying for one selection
    and using it many times, which is what makes a cross-layer shared index
    (hysparse-style) worth trying; an IoU near the sparsity ratio means the
    layers are choosing independently.
    """
    inter = (sel * prev_sel).sum(axis=-1)
    union = paddle.clip(sel + prev_sel, max=1.0).sum(axis=-1)
    return {"select_iou_prev": (inter / (union + EPS)).mean()}
