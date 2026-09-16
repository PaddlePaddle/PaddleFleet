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

"""Three equivalent representations of the prefix-LM three-region mask, plus
their dense expanders (used for cross-checking in unit tests).

## Geometry

A sequence is a concatenation of N mutually invisible segments, optionally
followed by a padding tail:

    │◄──── seg 0 ────►│◄──── seg 1 ────►│ … │◄── pad ──►│
    │ prefix │ suffix │ prefix │ suffix │   │           │

Segment ``i`` starts at ``s_i``, has a prefix (context) of length ``C_i``, a
suffix (query) of length ``Q_i``, and total length ``L_i = C_i + Q_i``.
``T = ΣL_i``; padding occupies ``[T, T+n_p)``; ``T' = T + n_p``.

Allowed relations (three regions plus the pad self-loop; ``True`` = **forbidden**):

| region of key column j | allowed query rows | meaning |
|---|---|---|
| prefix of seg i ``[s_i, s_i+C_i)`` | ``[s_i, s_i+L_i)`` | bidirectional within prefix; suffix sees the whole prefix |
| suffix of seg i ``[s_i+C_i, s_i+L_i)`` | ``[j, s_i+L_i)`` | causal within suffix; prefix cannot see suffix |
| pad ``[T, T')`` | ``{j}`` | self only, otherwise the whole pad row is masked and softmax(-inf) yields NaN |

Two implicit semantics have no explicit statement and are realized by keeping
the default "forbidden":
  * prefix rows cannot see suffix columns (a suffix column's lowest allowed row
    is ``j >= s_i+C_i``);
  * real tokens cannot see pad columns (a pad column only opens the diagonal).

## The three representations

| function | output | consumer |
|---|---|---|
| :func:`build_dense_mask`             | ``[1,1,T',T']`` bool | eager path (explicit score + mask_func) |
| :func:`build_flashmask_row_indices`  | ``[1,1,T',4]`` int32 | ``flashmask_attention`` column-sparse path |
| :func:`build_prefix_lm_layout`       | dict (4 fields)      | Triton kernel (geometry passed directly) |

:func:`expand_row_indices_to_dense` and :func:`expand_layout_to_dense` are for
unit tests only. They make it possible to *assert* that all three paths share
the same mask source of truth, rather than assume it.

## FlashMask 4-column encoding

Each key column ``j`` is given ``[LTS, LTE, UTS, UTE]``:

  * lower-triangle side (rows **>** j): row range ``[LTS, LTE)`` is **masked**
  * upper-triangle side (rows **<** j): row range ``[UTS, UTE)`` is **masked**

The diagonal (row == j) is **never masked by this encoding**, so the pad
self-loop is free. Substituting the table above (with ``UTS ≡ 0`` and
``LTE ≡ T'`` constant, only two values vary):

    UTE(j) = s_i        if j ∈ prefix_i        else j
    LTS(j) = s_i + L_i  if j ∈ seg i           else j + 1   (j ∈ pad)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import paddle

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "prefix_lm_segment_starts",
    "prefix_lm_pad_len",
    "build_dense_mask",
    "build_flashmask_row_indices",
    "build_prefix_lm_layout",
    "expand_row_indices_to_dense",
    "expand_layout_to_dense",
]

# Column order of the 4 fields, matching paddle's flashmask_attention.
_LTS, _LTE, _UTS, _UTE = 0, 1, 2, 3


def _check_lens(
    prefix_lens: Sequence[int], suffix_lens: Sequence[int], pad_len: int
) -> None:
    if len(prefix_lens) != len(suffix_lens):
        raise ValueError(
            f"prefix_lens and suffix_lens must have the same number of segments, "
            f"got {len(prefix_lens)} vs {len(suffix_lens)}"
        )
    if len(prefix_lens) == 0:
        raise ValueError("at least one segment is required")
    if any(int(c) < 0 for c in prefix_lens):
        raise ValueError(
            f"prefix_lens must be non-negative, got {list(prefix_lens)}"
        )
    if any(int(q) <= 0 for q in suffix_lens):
        # The suffix is the latent query, always a positive length; a length of
        # 0 would degenerate the row selection into an empty range.
        raise ValueError(
            f"suffix_lens must be positive, got {list(suffix_lens)}"
        )
    if int(pad_len) < 0:
        raise ValueError(f"pad_len must be non-negative, got {pad_len}")


def prefix_lm_segment_starts(
    prefix_lens: Sequence[int], suffix_lens: Sequence[int]
) -> list[int]:
    """Segment starts ``s_i`` (prefix sum), length N."""
    starts, acc = [], 0
    for c, q in zip(prefix_lens, suffix_lens):
        starts.append(acc)
        acc += int(c) + int(q)
    return starts


def prefix_lm_pad_len(
    seq_len: int, tp_size: int, sequence_parallel: bool, align: int = 128
) -> int:
    """Padding length needed to round ``seq_len`` up to a multiple of
    ``lcm(base, align)``.

    ``base`` is ``tp_size`` when sequence parallel is enabled (so the sequence
    scatters evenly across the SP partitions), otherwise 1; ``align`` adds a
    further fixed alignment. ``align`` is an explicit argument here rather than
    an environment variable.
    """
    base = int(tp_size) if sequence_parallel else 1
    mult = math.lcm(base, max(int(align), 1))
    return (mult - (int(seq_len) % mult)) % mult


def build_dense_mask(
    prefix_lens: Sequence[int],
    suffix_lens: Sequence[int],
    pad_len: int = 0,
    place=None,
) -> paddle.Tensor:
    """Representation 1: dense boolean mask ``[1, 1, T', T']``, ``True`` = forbidden.

    Start with everything forbidden, then open four regions per segment:
    prefix<->prefix (bidirectional), suffix->prefix, causal within suffix, and
    the pad self-loop. Different segments stay fully forbidden against each
    other, which enforces isolation.
    """
    _check_lens(prefix_lens, suffix_lens, pad_len)
    starts = prefix_lm_segment_starts(prefix_lens, suffix_lens)
    total = starts[-1] + int(prefix_lens[-1]) + int(suffix_lens[-1])
    tp = total + int(pad_len)

    mask = paddle.ones([1, 1, tp, tp], dtype="bool")
    if place is not None:
        mask = mask.to(place)

    for s, c, q in zip(starts, prefix_lens, suffix_lens):
        c, q = int(c), int(q)
        real_end = s + c + q
        if c > 0:
            # prefix <-> prefix, bidirectional
            mask[:, :, s : s + c, s : s + c] = False
            # suffix can see the whole prefix of its segment
            mask[:, :, s + c : real_end, s : s + c] = False
        # causal within suffix (triu(diagonal=1) = strict upper triangle forbidden)
        causal = paddle.triu(paddle.ones([q, q], dtype="bool"), diagonal=1)
        if place is not None:
            causal = causal.to(place)
        mask[:, :, s + c : real_end, s + c : real_end] = causal

    # pad rows see only themselves; without this the whole pad row is -inf and
    # softmax yields NaN
    if int(pad_len) > 0:
        idx = paddle.arange(total, tp, dtype="int64")
        if place is not None:
            idx = idx.to(place)
        mask[0, 0, idx, idx] = False
    return mask


def build_flashmask_row_indices(
    prefix_lens: Sequence[int],
    suffix_lens: Sequence[int],
    pad_len: int = 0,
    place=None,
) -> paddle.Tensor:
    """Representation 2: FlashMask column-sparse indices ``[1, 1, T', 4]`` int32.

    The 4 columns are ``[LTS, LTE, UTS, UTE]`` (see the encoding notes in the
    module docstring). There is no per-column Python loop: ``repeat_interleave``
    broadcasts the per-segment ``(s_i, s_i+L_i)`` onto the column axis, then two
    ``where`` calls against ``arange`` compute the fields.
    """
    _check_lens(prefix_lens, suffix_lens, pad_len)
    starts = prefix_lm_segment_starts(prefix_lens, suffix_lens)
    seg_lens = [int(c) + int(q) for c, q in zip(prefix_lens, suffix_lens)]
    total = starts[-1] + seg_lens[-1]
    tp = total + int(pad_len)

    seg_len_t = paddle.to_tensor(seg_lens, dtype="int32")
    start_t = paddle.to_tensor(starts, dtype="int32")
    prefix_t = paddle.to_tensor([int(c) for c in prefix_lens], dtype="int32")

    # Broadcast segment-level values onto the column axis:
    # col_start[j] = s_i, col_end[j] = s_i + L_i, col_prefix_end[j] = s_i + C_i
    col_start = paddle.repeat_interleave(start_t, seg_len_t)
    col_end = paddle.repeat_interleave(start_t + seg_len_t, seg_len_t)
    col_prefix_end = paddle.repeat_interleave(start_t + prefix_t, seg_len_t)

    j_real = paddle.arange(total, dtype="int32")
    is_prefix_col = j_real < col_prefix_end

    tp_t = paddle.full([total], tp, dtype="int32")
    lts_real = col_end  # rows >= s_i+L_i forbidden
    lte_real = tp_t
    uts_real = paddle.zeros([total], dtype="int32")
    ute_real = paddle.where(
        is_prefix_col, col_start, j_real
    )  # prefix col opens to s_i; suffix col opens to j

    if int(pad_len) > 0:
        j_pad = paddle.arange(total, tp, dtype="int32")
        lts = paddle.concat(
            [lts_real, j_pad + 1]
        )  # pad col: rows > j forbidden
        lte = paddle.concat(
            [lte_real, paddle.full([int(pad_len)], tp, dtype="int32")]
        )
        uts = paddle.concat(
            [uts_real, paddle.zeros([int(pad_len)], dtype="int32")]
        )
        ute = paddle.concat([ute_real, j_pad])  # pad col: rows < j forbidden
    else:
        lts, lte, uts, ute = lts_real, lte_real, uts_real, ute_real

    out = paddle.stack([lts, lte, uts, ute], axis=-1).reshape([1, 1, tp, 4])
    if place is not None:
        out = out.to(place)
    return out


def build_prefix_lm_layout(
    prefix_lens: Sequence[int],
    suffix_lens: Sequence[int],
    pad_len: int = 0,
) -> dict[str, object]:
    """Representation 3: geometry layout for the Triton kernel (plain Python
    scalars, no tensors)."""
    _check_lens(prefix_lens, suffix_lens, pad_len)
    return {
        "segment_starts": tuple(
            prefix_lm_segment_starts(prefix_lens, suffix_lens)
        ),
        "n_contexts": tuple(int(c) for c in prefix_lens),
        "n_queries": tuple(int(q) for q in suffix_lens),
        "pad_len": int(pad_len),
    }


def expand_row_indices_to_dense(row_indices: paddle.Tensor) -> paddle.Tensor:
    """Expand representation 2 into a dense ``[1,1,T',T']`` bool (``True`` =
    forbidden). **Unit tests only.**

    Derived strictly from the FlashMask semantics, without referencing the other
    two representations, so that it can serve as an independent cross-check.
    """
    if row_indices.ndim != 4 or row_indices.shape[-1] != 4:
        raise ValueError(
            f"row_indices shape should be [1,1,T',4], got {row_indices.shape}"
        )
    tp = int(row_indices.shape[2])
    ri = row_indices.reshape([tp, 4]).astype("int64")
    lts, lte, uts, ute = ri[:, _LTS], ri[:, _LTE], ri[:, _UTS], ri[:, _UTE]

    rows = paddle.arange(tp, dtype="int64").reshape([tp, 1])  # row r
    cols = paddle.arange(tp, dtype="int64").reshape([1, tp])  # column j

    lower = rows > cols  # lower-triangle side
    upper = rows < cols  # upper-triangle side
    in_lower_band = (rows >= lts.reshape([1, tp])) & (
        rows < lte.reshape([1, tp])
    )
    in_upper_band = (rows >= uts.reshape([1, tp])) & (
        rows < ute.reshape([1, tp])
    )

    forbidden = (lower & in_lower_band) | (upper & in_upper_band)
    return forbidden.reshape([1, 1, tp, tp])


def expand_layout_to_dense(layout: dict[str, object]) -> paddle.Tensor:
    """Expand representation 3 into a dense ``[1,1,T',T']`` bool (``True`` =
    forbidden). **Unit tests only.**

    Written directly from the "allowed relations" table in the module docstring,
    region by region, without reusing :func:`build_dense_mask`.
    """
    starts = list(layout["segment_starts"])  # type: ignore[arg-type]
    n_ctx = list(layout["n_contexts"])  # type: ignore[arg-type]
    n_q = list(layout["n_queries"])  # type: ignore[arg-type]
    pad_len = int(layout["pad_len"])  # type: ignore[arg-type]

    total = starts[-1] + n_ctx[-1] + n_q[-1]
    tp = total + pad_len

    rows = paddle.arange(tp, dtype="int64").reshape([tp, 1])
    cols = paddle.arange(tp, dtype="int64").reshape([1, tp])
    allowed = paddle.zeros([tp, tp], dtype="bool")

    for s, c, q in zip(starts, n_ctx, n_q):
        seg_end = s + c + q
        col_in_prefix = (cols >= s) & (cols < s + c)
        col_in_suffix = (cols >= s + c) & (cols < seg_end)
        row_in_seg = (rows >= s) & (rows < seg_end)
        # prefix column: allow all rows of the segment
        allowed = allowed | (col_in_prefix & row_in_seg)
        # suffix column j: allow rows [j, seg_end)
        allowed = allowed | (col_in_suffix & row_in_seg & (rows >= cols))

    if pad_len > 0:
        col_in_pad = cols >= total
        allowed = allowed | (col_in_pad & (rows == cols))

    return paddle.logical_not(allowed).reshape([1, 1, tp, tp])
