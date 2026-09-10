# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Core tensor probe for train_infer_consistent_inspect (training side).

`inspect_tensor` prints abssum/absmax/md5 per (tag, layer), optionally dumps the
tensor to `.npy` and optionally overrides it with the inference side's dump.

Call sites need no `if inspect_enabled():` wrapper: with `ABLATION_INSPECT_TENSOR`
unset the first statement returns the tensor untouched after reading one cached
module-level bool, and whatever massaging a probe needs (fp8 dequant, row
permutation) is handed over as `pre_save_func` / `post_load_func` so it is not
evaluated either.

The layout helpers live next door, one module per operator family: `permute.py`
for the expert-contiguous row order, `ffn_act.py` for the fused SwiGLU+fp8-quant
activation, `gate.py` for the fused two-view router logits.

Every probe entry point the network is expected to call is named
`inspect_tensor_*`, so grepping that prefix lists the whole surface.
"""

from __future__ import annotations

import fnmatch
import hashlib
import math
import os

import paddle

from paddlefleet.train_infer_consistent_ops.slice_util import (
    last_dim_segment,
    scatter_last_dim_segment,
)

_MODULE_PREFIX = "[ABLATION_train]"

# Snapshot of the ABLATION_* configuration, read once at import.
#
# `os.environ` is an `os._Environ`, not a dict: every `.get` runs a Python-level
# `__getitem__` plus fsencode/fsdecode and costs ~0.6us, which the disabled path
# would pay at each of the ~1k probes a forward passes through. The launcher
# exports these before the process starts, so reading them once is enough.
# Each tag filter is parsed into `(exact_frozenset, glob_tuple)`: entries with no
# glob metachar stay in the set for an O(1) probe, the `mhc_*` style ones go into
# the tuple and are matched with fnmatch. See `_parse_tag_filter` / `_tag_matches`.
_EMPTY_FILTER = (frozenset(), ())
_ENABLED = False
_WHITELIST = _EMPTY_FILTER
_BLACKLIST = _EMPTY_FILTER
_DUMP_SKIP_TAGS = _EMPTY_FILTER
_DUMP_TRAIN_TAGS = _EMPTY_FILTER
_SAVE_PATH = ""
_LOAD_PATH = ""
# Per-element diff breakdown on load, only when the two sides fail to md5-align.
# See `_print_detail_diff`; switched by ABLATION_INSPECT_DETAIL=1.
_DETAIL = False

# fnmatch metacharacters; their presence promotes an entry from exact to glob.
_GLOB_CHARS = ("*", "?", "[")

# How many diff positions `_print_detail_diff` lists, most-divergent first.
_DETAIL_MAX_POS = 20


def _parse_tag_filter(raw):
    """Parse a comma-separated ABLATION_TAG_* value into (exact_set, glob_tuple).

    Entries carrying a glob metacharacter (`*`, `?`, `[`) go into the glob tuple
    and are matched with `fnmatch`; the rest stay in a frozenset for an O(1)
    exact lookup. Splitting them up front keeps the common exact-tag case a set
    probe and only scans the (usually tiny) glob list when needed.
    """
    items = list(filter(None, raw.split(",")))
    is_glob = [any(c in i for c in _GLOB_CHARS) for i in items]
    exact = frozenset(i for i, g in zip(items, is_glob) if not g)
    globs = tuple(i for i, g in zip(items, is_glob) if g)
    return exact, globs


def refresh_env_cache():
    """Re-read the ABLATION_* environment variables into the module snapshot.

    Called once at import. Anything that flips these variables mid-process (the
    unit tests) has to call this afterwards.
    """
    global \
        _ENABLED, \
        _WHITELIST, \
        _BLACKLIST, \
        _DUMP_SKIP_TAGS, \
        _SAVE_PATH, \
        _LOAD_PATH, \
        _DETAIL
    env = os.environ
    _ENABLED = env.get("ABLATION_INSPECT_TENSOR", "0") == "1"
    _WHITELIST = _parse_tag_filter(env.get("ABLATION_TAG_WHITELIST", ""))
    _BLACKLIST = _parse_tag_filter(env.get("ABLATION_TAG_BLACKLIST", ""))
    _DUMP_SKIP_TAGS = _parse_tag_filter(env.get("ABLATION_DUMP_SKIP_TAGS", ""))
    _SAVE_PATH = env.get("ABLATION_SAVE_TENSOR_PATH", "")
    _LOAD_PATH = env.get("ABLATION_LOAD_TENSOR_PATH", "")
    _DETAIL = env.get("ABLATION_INSPECT_DETAIL", "1") == "1"


refresh_env_cache()


def inspect_enabled():
    """True when train_infer_consistent_inspect is on.

    Switched by `ABLATION_INSPECT_TENSOR=1`, snapshotted once at import.
    """
    return _ENABLED


def _filter_active(tag_filter):
    """Whether `tag_filter` holds any entry at all (exact or glob)."""
    return bool(tag_filter[0] or tag_filter[1])


def _tag_matches(tag, tag_filter):
    """True when `tag` is caught by `tag_filter`, an (exact_set, glob_tuple).

    Globs are tried first (`mhc_*` catches every `mhc_`-prefixed tag), then the
    exact set; either alone is enough. Matching is case-sensitive and an empty
    filter never matches. Only reached with the probes on, so the glob scan
    costs nothing on the disabled path.
    """
    exact, globs = tag_filter
    return any(fnmatch.fnmatchcase(tag, g) for g in globs) or tag in exact


def inspect_tag_enabled(tag):
    """True when the probes are on *and* `tag` survives the tag filters.

    The same gate `inspect_tensor` applies in its stage 1, exposed for the one
    helper that has to change what the model computes to make a probe comparable
    (`inspect_tensor_force_unit_probs`). That change has to follow the filters
    too: narrowing a run down to a few tags must not keep rewriting the math
    behind a probe nobody asked for.

    Whitelist / blacklist entries may be `fnmatch` globs (`mhc_*` catches every
    `mhc_`-prefixed tag) or plain exact tags; see `_tag_matches`.
    """
    return _ENABLED and not (
        (_filter_active(_WHITELIST) and not _tag_matches(tag, _WHITELIST))
        or _tag_matches(tag, _BLACKLIST)
    )


# ---------------------------------------------------------------------------
# Current-layer context
#
# Modules reused across layers that carry no layer id of their own (the
# shared-expert MLP, the dense `first_k_dense_replace` MLP, the routed-expert
# grouped GEMM node) read the layer id that the enclosing block publishes here,
# so their dumps land in the right layer_<id> directory.
# ---------------------------------------------------------------------------
_CURRENT_LAYER_IDX = -1


def inspect_tensor_set_current_layer(layer_idx):
    """Publish the 0-based decoder layer id currently executing its MLP/MoE block.

    Deliberately *not* gated on `inspect_enabled()`: it is a single int store into
    a module global that nothing outside the probes ever reads, so gating it only
    bought an extra function call. The flip side is that this is the one probe
    entry point that does touch module state while the probes are off -- harmless,
    but keep it in mind when reasoning about "probes off means nothing happens".
    """
    global _CURRENT_LAYER_IDX
    _CURRENT_LAYER_IDX = -1 if layer_idx is None else layer_idx


def get_current_layer():
    """Return the layer id published by `inspect_tensor_set_current_layer`."""
    return _CURRENT_LAYER_IDX


# Container types `inspect_tensor(..., index=...)` can reach into. Fused blocks
# pass their output around as a `(tensor, bias)` tuple; the list / dict forms show
# up where a stage hands back several buffers at once.
_INDEXABLE = (tuple, list, dict)


def _with_element(container, index, value):
    """Return `container` with element / key `index` replaced by `value`.

    Always a fresh container: the caller may still be holding the original, and
    the probe must never write into anything the network owns.
    """
    if isinstance(container, tuple):
        return (*container[:index], value, *container[index + 1 :])
    out = list(container) if isinstance(container, list) else dict(container)
    out[index] = value
    return out


def _stats(arr):
    """Return (abssum, absmax, md5) of a float32 numpy array.

    `math.fsum` over a flat host list keeps the sum order-independent, so the two
    frameworks stay comparable despite different reduce orders.

    The sign of zero is normalized before the md5. `-0.0` and `+0.0` compare
    equal but differ byte for byte (0x80000000 vs 0x00000000 in float32), and the
    128-aligned MoE padding rows routinely end up `-0.0` on one side and `+0.0` on
    the other, which used to raise a spurious "max_abs_diff=0 yet md5 differs".
    `arr + 0.0` collapses `-0.0` into `+0.0` and leaves every other value
    (inf / nan / subnormals included) bit-identical; abssum and absmax do not move
    either, since `np.abs` never looked at the sign of zero. So this only brings
    md5 in line with the other three metrics: equal values <=> all four equal.
    It touches the printed statistics only, never the bytes `np.save` writes, so
    dumps from earlier runs stay comparable. Both sides have to change together --
    normalize on one side only and md5 becomes incomparable again.
    """
    import numpy as np

    if arr.dtype.kind == "f":
        arr = arr + arr.dtype.type(0.0)
    abs_list = np.abs(arr).reshape(-1).tolist()
    return (
        float(math.fsum(abs_list)),
        float(max(abs_list)) if abs_list else 0.0,
        hashlib.md5(arr.tobytes()).hexdigest(),
    )


def _print_detail_diff(tag, layer_idx, rank, infer, train):
    """Per-element diff breakdown, printed only when the two sides fail to align.

    Ported from `src/analysis/dump_diff.py::compare`. `infer` is the loaded
    reference (`<tag>.npy`, the inference side) and `train` the live snapshot
    (this training side); both arrive already reshaped to the same shape. The
    md5 gate in stage 6 means this only fires when the plain abssum/absmax/md5
    line already showed the two sides do *not* match bit for bit, so it exists
    to say *where* and *by how much* they diverge:

      * diff element count / ratio;
      * the single largest absolute diff and its position, with both values;
      * the single largest relative diff (`abs / max(|infer|, |train|)`, so a
        zero on one side never divides) and its position;
      * a bf16 ULP-step ratio table -- for each of 1 / 2 / 3 / >3 ULP (the
        fp32 bit-pattern distance // 65536, i.e. how many bf16-representable
        steps apart the two values are), the share of diffing elements landing
        in that bucket. `1 ULP` is just last-bit rounding (noise floor); `>3
        ULP` means the operator semantics or the weight layout differ;
      * the `_DETAIL_MAX_POS` most-divergent positions, largest abs diff first.

    All on the diff subset only: the abs/rel/ULP metrics upcast just the unequal
    elements to float64, which is usually a tiny fraction of a large tensor.
    """
    import numpy as np

    def emit(msg):
        print(msg, flush=True)

    prefix = f"[ABLATION_detail] tag={tag} rank={rank} layer={layer_idx}"

    def emit_table(title, headers, rows):
        headers = tuple(str(value) for value in headers)
        rows = [tuple(str(value) for value in row) for row in rows]
        widths = [len(header) for header in headers]
        for row in rows:
            widths = [
                max(width, len(value)) for width, value in zip(widths, row)
            ]

        def format_row(row, right_align=False):
            cells = []
            for value, width in zip(row, widths):
                cells.append(
                    value.rjust(width) if right_align else value.ljust(width)
                )
            return "| " + " | ".join(cells) + " |"

        border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
        emit(title)
        emit(border)
        emit(format_row(headers))
        emit(border)
        for row in rows:
            emit(format_row(row, right_align=True))
        emit(border)

    a = np.squeeze(infer)
    b = np.squeeze(train)
    if a.shape != b.shape:
        if a.size != b.size:
            emit(
                f"{prefix} skip: element count differs "
                f"infer={infer.shape} train={train.shape}"
            )
            return
        a = a.reshape(b.shape)

    total = a.size
    # Both sides are bf16-quantized values; subtracting in float32 is exact, no
    # need to double the memory with float64 here.
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    flat_pos = np.flatnonzero(diff.reshape(-1))
    n_diff = flat_pos.size
    if n_diff == 0:
        emit_table(
            prefix,
            ("metric", "value"),
            (
                ("total", total),
                ("n_diff", n_diff),
                ("ratio", f"{n_diff / total:.6e} ({n_diff / total:.4%})"),
            ),
        )
        emit(f"{prefix} identical element-for-element")
        return

    # Upcast only the unequal elements for the abs/rel metrics.
    av = a.reshape(-1)[flat_pos].astype(np.float64)
    bv = b.reshape(-1)[flat_pos].astype(np.float64)
    abs_d = np.abs(av - bv)
    rel_d = abs_d / np.maximum(np.abs(av), np.abs(bv))

    i_abs = int(np.argmax(abs_d))
    i_rel = int(np.argmax(rel_d))
    nd_abs = tuple(int(x) for x in np.unravel_index(flat_pos[i_abs], a.shape))
    nd_rel = tuple(int(x) for x in np.unravel_index(flat_pos[i_rel], a.shape))
    emit_table(
        f"{prefix} summary",
        ("metric", "position", "infer", "train", "value"),
        (
            ("total", "", "", "", total),
            ("n_diff", "", "", "", n_diff),
            (
                "ratio",
                "",
                "",
                "",
                f"{n_diff / total:.6e} ({n_diff / total:.4%})",
            ),
            (
                "max_abs",
                nd_abs,
                f"{av[i_abs]:.9g}",
                f"{bv[i_abs]:.9g}",
                f"{abs_d[i_abs]:.9g}",
            ),
            (
                "max_rel",
                nd_rel,
                f"{av[i_rel]:.9g}",
                f"{bv[i_rel]:.9g}",
                f"{rel_d[i_rel]:.9g}",
            ),
        ),
    )

    # bf16 ULP steps: fp32 bit-pattern distance // 65536 (bf16 = fp32 high 16
    # bits). 1 = last-bit rounding; larger = operator or layout divergence.
    steps = (
        np.abs(
            a.reshape(-1)[flat_pos]
            .astype(np.float32)
            .view(np.uint32)
            .astype(np.int64)
            - b.reshape(-1)[flat_pos]
            .astype(np.float32)
            .view(np.uint32)
            .astype(np.int64)
        )
        // 65536
    )
    show = min(20, n_diff)
    ulp_groups = (
        ("1 ULP", steps == 1),
        ("2 ULP", steps == 2),
        ("3 ULP", steps == 3),
        (">3 ULP", steps > 3),
    )
    diff_rows = []
    for k in np.argsort(-abs_d)[:show]:
        nd = tuple(int(x) for x in np.unravel_index(flat_pos[k], a.shape))
        diff_rows.append(
            (
                nd,
                f"{av[k]:.9g}",
                f"{bv[k]:.9g}",
                f"{abs_d[k]:.6g}",
                f"{rel_d[k]:.6g}",
            )
        )
    ulp_row = (
        "ULP",
        *(f"{mask.sum() / n_diff:.4%}" for _, mask in ulp_groups),
    )
    sections = [
        [("metric", "1 ULP", "2 ULP", "3 ULP", ">3 ULP"), ulp_row],
        [
            ("metric", "position", "infer", "train", "abs_diff", "rel_diff"),
            *(("Top20", *row) for row in diff_rows),
        ],
    ]
    # Pad every section to the same number of columns and calculate one global
    # width per column; otherwise the two section-specific layouts cannot align.
    n_columns = max(len(row) for section in sections for row in section)
    sections = [
        [
            tuple(str(value) for value in row) + ("",) * (n_columns - len(row))
            for row in section
        ]
        for section in sections
    ]
    widths = [
        max(len(row[column]) for section in sections for row in section)
        for column in range(n_columns)
    ]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    lines = [
        f"{prefix} ULP ratios (denominator=n_diff), top {show}/{n_diff} by abs diff",
        border,
    ]
    for section_index, section in enumerate(sections):
        for row_index, row in enumerate(section):
            lines.append(
                "| "
                + " | ".join(
                    value.ljust(width) for value, width in zip(row, widths)
                )
                + " |"
            )
            if row_index == 0:
                lines.append(border)
        if section_index < len(sections) - 1:
            lines.append(border)
    lines.append(border)
    emit("\n".join(lines))


def _squeeze_shape(shape):
    """Drop size-1 dims.

    A leading batch dim of 1 is the one layout difference the two sides are allowed
    to have for free: the inference side reports `[11, 4096]` where the training
    side carries `[1, 11, 4096]`.
    """
    return tuple(int(d) for d in shape if d != 1)


def _load_shape_ok(dump_shape, live_shape):
    """Whether a dump of `dump_shape` may be reshaped into `live_shape`.

    Equal numel is necessary but nowhere near sufficient, and trusting it alone
    caused a real corruption: at `client_concurrency_per_endpoint: 8` the inference
    side dumps the dense MLP as `[88, 3584]` (88 dp-gathered tokens x a tp8 column
    shard) while the training side holds `[1, 11, 28672]` (11 tokens x the full
    width). Both are 315392 elements, so the dump was silently reshaped and fed in,
    poisoning layer 0 with `max_abs_diff=2.68`.

    What separates that collision from the legitimate layout differences is the
    **row count**: the two sides must agree on how many tokens/rows they describe,
    and only the trailing feature dims may be grouped differently. Hence, after
    squeezing size-1 dims:
      - identical shapes pass;
      - `[11, 4, 4096]` (4 mHC streams as their own dim) vs `[11, 16384]` (streams
        folded into hidden) passes -- same 11 rows, trailing dims regrouped;
      - `[88, 3584]` vs `[11, 28672]` fails -- 88 rows cannot describe 11 tokens.

    Plain broadcast compatibility is *not* the right test here: it would reject the
    `layer_input` / `layer_output` case above, which is a legitimate regroup that
    currently loads with `max_abs_diff=0`.

    Returns:
        (ok, reason) -- `reason` goes into the skip log line so the table shows why.
    """
    dump = tuple(int(d) for d in dump_shape)
    live = tuple(int(d) for d in live_shape)
    if math.prod(dump) != math.prod(live):
        return (
            False,
            f"numel mismatch dump={math.prod(dump)} live={math.prod(live)}",
        )
    if dump == live:
        return True, "exact"
    sq_dump, sq_live = _squeeze_shape(dump), _squeeze_shape(live)
    if sq_dump == sq_live:
        return True, "equal ignoring size-1 dims"
    if not sq_dump or not sq_live:
        return True, "degenerate shape, numel already matches"
    if sq_dump[0] != sq_live[0]:
        return False, (
            f"row count differs dump_rows={sq_dump[0]} live_rows={sq_live[0]} "
            f"(numel collides but the two sides describe a different number of "
            f"tokens/rows -- e.g. dp-gathered rows x a tp column shard)"
        )
    return True, "same row count, trailing dims regrouped"


# ---------------------------------------------------------------------------
# The probe itself
#
# The body below is split into 7 numbered stages, and the inference side's
# `inspect_tensor` runs the same 7 stages in the same order -- a change on one
# side belongs on the other. The only deliberate differences are:
#   * `save` / `load` defaults (this side loads the other side's dumps, the
#     inference side produces them);
#   * stage 3 additionally drops decode steps on the inference side, which a
#     training forward has no equivalent of;
#   * framework calls (paddle vs torch) and the log prefix.
# ---------------------------------------------------------------------------


def _as_f32_numpy(tensor):
    """Host float32 numpy copy -- the single conversion stages 4/5/6 share."""
    return tensor.astype("float32").numpy()


def interleave_rope_segment(t, nope_dim):
    t_rope = last_dim_segment(t, nope_dim)
    return to_interleave_rope_layout_(t_rope)


def interleave_rope_segment_inplace(t, nope_dim):
    t_rope = last_dim_segment(t, nope_dim)
    t_rope_interleave = to_interleave_rope_layout_(t_rope)
    return scatter_last_dim_segment(t, t_rope_interleave, nope_dim)


def to_interleave_rope_layout_(x, start=0, end=None):
    """Rewrite post-RoPE tensors from half-split to interleaved channel layout.

    The training-side fused kernel reads channel pairs ``(2j, 2j+1)`` but writes
    the two results to ``j`` and ``j + rope_dim // 2`` (a **half-split** layout),
    whereas sglang's RoPE writes each result back to the channel it came from (an
    **interleaved** layout). The two layouts differ by a channel permutation
    only::

        interleaved[2j]     == half[j]                    j = 0 .. rope_dim // 2 - 1
        interleaved[2j + 1] == half[j + rope_dim // 2]

    This helper takes the half-split tensor and rebuilds the interleaved one, so
    the training side can be compared against sglang in sglang's own layout.
    Applying it to q_pe and k_pe together leaves ``q . k`` -- and hence the
    attention output -- unchanged, so this is purely a comparison aid.

    ``start`` / ``end`` restrict the permutation to the ``[..., start:end]``
    segment of the last dim, leaving the head ``[..., :start]`` and tail
    ``[..., end:]`` verbatim. This is the ``[nope | rope]`` case: the rope block
    is half-split while the nope block, which RoPE never touches, is passed
    through. The default spans the whole last dim.

    Returns a fresh tensor via ``concat`` rather than writing into ``x``:
    the live buffer is usually still held by the autograd graph, and an
    in-place write bumps its dygraph inplace version and breaks
    backward / recompute.
    """
    width = x.shape[-1]
    stop = width if end is None else end
    seg = x[..., start:stop]
    seg_width = seg.shape[-1]
    if seg_width % 2 != 0:
        raise ValueError(
            "to_interleave_rope_layout_ needs an even-length segment to pair "
            f"channels (2j, 2j+1); got width {seg_width} for [{start}:{stop}]."
        )
    half = seg_width // 2
    seg = paddle.stack([seg[..., :half], seg[..., half:]], axis=-1).reshape(
        seg.shape
    )
    parts = []
    if start > 0:
        parts.append(x[..., :start])
    parts.append(seg)
    if stop < width:
        parts.append(x[..., stop:])
    return paddle.concat(parts, axis=-1) if len(parts) > 1 else parts[0]


def inspect_tensor(
    tag,
    layer_idx,
    tensor,
    index=None,
    save=False,
    load=True,
    pre_save_func=None,
    post_load_func=None,
):
    """Inspect tensor info, optionally save to .npy and/or load override from .npy.

    Stages: 1 gate -> 2 snapshot -> 3 context -> 4 info -> 5 save -> 6 load ->
    7 return. Stages 4/5/6 share one host float32 copy of the snapshot.

    Controlled by environment variables:
        ABLATION_INSPECT_TENSOR: "1" to enable the probes at all.
        ABLATION_TAG_WHITELIST: comma-separated tags; when set, tags outside the
            list return immediately (no hooks, no info, no save, no
            load). Entries may be `fnmatch` globs (`mhc_*` matches every
            `mhc_`-prefixed tag) or exact tags.
        ABLATION_TAG_BLACKLIST: comma-separated tags to return early on; glob or
            exact, same rule as the whitelist.
        ABLATION_SAVE_TENSOR_PATH: directory to dump `.npy` files into.
        ABLATION_LOAD_TENSOR_PATH: directory with the other side's `.npy` dumps.
        ABLATION_DUMP_SKIP_TAGS: comma-separated tags to skip for saving/loading
            only (they are still printed); glob or exact, same rule.
        ABLATION_INSPECT_DETAIL: "1" to add a per-element diff breakdown on load
            (`[ABLATION_detail]` lines: diff count/ratio, largest abs/rel diff
            with positions, a 1/2/3/>3 bf16 ULP-step ratio table, top divergent
            positions). Printed only when the loaded reference and the live
            snapshot fail to md5-align, so a bit-identical match stays quiet.

    Args:
        tag: identifier for the tensor checkpoint.
        layer_idx: transformer layer index (-1 for embedding / LM head).
        tensor: the live paddle tensor, or a tuple / list / dict holding it when
            `index` is given.
        index: element / key to probe inside `tensor`, handing back a copy of the
            container with only that element replaced -- the `(tensor, bias)`
            bundles fused blocks pass around, without open-coding the
            unwrap/rewrap at every call site. A plain tensor ignores it, so a call
            site whose value is sometimes a bundle and sometimes bare can pass it
            unconditionally. `tensor[index]` being None (an absent bias) aborts
            the probe exactly like `pre_save_func` returning None.
        save: if True, dump the snapshot to a .npy file.
        load: if True, look for the other side's dump and hand it back instead of
            `tensor` when it exists.
        pre_save_func: `tensor -> snapshot`, applied before printing/saving. Use
            it to build the comparable view (dequant, canonical row order) so the
            work is skipped when the probes are off or the tag is filtered out.
            It only feeds the info/save/load side and never the return value.
            Returning None aborts the probe and leaves `tensor` untouched.
        post_load_func: applied to the loaded dump, and only when one was really
            loaded; its result becomes the return value. Use it to fold the dump
            back into the live buffer (the snapshot is a derived view, so the
            caller is the only one who knows how to invert it).

    Returns:
        The loaded dump when one was applied (after `post_load_func` has had its
        chance to invert whatever view `pre_save_func` built), otherwise the input
        `tensor` -- the *same object*, so `result is tensor` is the test for
        "nothing was loaded" (valid only where no `post_load_func` rewraps the
        dump into a fresh container).

        The override is a *return value*, never an in-place write, so a call site
        that drops the result silently downgrades the probe to a print -- every
        overriding call site must rebind. Writing into the live tensor instead
        (`paddle.assign(src, tensor)`) was tried and reverted: it bumps the
        dygraph inplace version of a tensor the autograd graph still holds, and
        backward dies with `PermissionDeniedError: Tensor ... has been modified by
        an inplace operation. Its version is 3 but the expected version is 2`
        (raised from `MatmulGradNode` via `TensorWrapper::check_inplace_version`).
        The flip side is that `paddle.to_tensor` yields `stop_gradient=True`, so
        each applied override cuts the backward chain at that point -- acceptable
        because a train_infer_consistent_inspect run only needs the forward.
    """
    # --- 1. gate: probes off or tag filtered out -> hand the live tensor back --
    if tensor is None or not _ENABLED:
        return tensor
    # Whitelist first (when non-empty only its tags survive), then blacklist;
    # both gate the whole function, hooks included.
    if not inspect_tag_enabled(tag):
        return tensor

    # --- 2. snapshot: the comparable view; feeds info/save/load, never the
    #        return value. `index` reaches into a tuple/list/dict bundle first.
    #        None means "nothing comparable here", so give up. -----------------
    indexed = index is not None and isinstance(tensor, _INDEXABLE)
    target = tensor[index] if indexed else tensor
    snapshot = target if pre_save_func is None else pre_save_func(target)
    if snapshot is None:
        return tensor

    import numpy as np

    # --- 3. context: which rank's dump directory this belongs to. (The
    #        inference side also decides here whether this is a decode step.) ---
    rank = (
        paddle.distributed.get_rank()
        if paddle.distributed.is_initialized()
        else 0
    )

    # --- 4. info: print abssum/absmax/md5. The host copy made here is the one
    #        stages 5 and 6 reuse; a dtype the cast refuses (fp8) aborts the
    #        print and the save, but must never break the forward. -------------
    try:
        arr = _as_f32_numpy(snapshot)
    except Exception as e:
        arr = None
        print(
            f"{_MODULE_PREFIX} tag={tag} layer={layer_idx} info_failed={e}",
            flush=True,
        )
    if arr is not None:
        abssum, absmax, live_md5 = _stats(arr)
        print(
            f"{_MODULE_PREFIX} tag={tag} rank={rank} layer={layer_idx} "
            f"abssum={abssum} absmax={absmax} md5={live_md5} shape={list(snapshot.shape)} dtype={snapshot.dtype}",
            flush=True,
        )
    else:
        live_md5 = None

    # --- 5. save: dump the snapshot to rank_<r>/layer_<l>/<tag>.npy -----------
    if (
        arr is not None
        and save
        and _SAVE_PATH
        and not _tag_matches(tag, _DUMP_SKIP_TAGS)
    ):
        layer_dir = os.path.join(
            _SAVE_PATH, f"rank_{rank}", f"layer_{layer_idx}"
        )
        os.makedirs(layer_dir, exist_ok=True)
        fpath = os.path.join(layer_dir, f"{tag}.npy")
        np.save(fpath, arr)
        abssum, absmax, md5 = _stats(arr)
        print(
            f"[ABLATION_dump_tensor] saved {tag} rank={rank} layer={layer_idx} shape={list(snapshot.shape)} "
            f"dtype={snapshot.dtype} abssum={abssum} absmax={absmax} md5={md5} -> {fpath}",
            flush=True,
        )

    # --- 6. load: the other side's dump for this (rank, layer, tag), gated by
    #        `_load_shape_ok`, plus the diff report against the live snapshot ---
    loaded = None
    if load and _LOAD_PATH and not _tag_matches(tag, _DUMP_SKIP_TAGS):
        fpath = os.path.join(
            _LOAD_PATH, f"rank_{rank}", f"layer_{layer_idx}", f"{tag}.npy"
        )
        if os.path.exists(fpath):
            dump = np.load(fpath)
            shape_ok, reason = _load_shape_ok(dump.shape, snapshot.shape)
            if not shape_ok:
                print(
                    f"[ABLATION_load_tensor] skip {tag} rank={rank} layer={layer_idx} "
                    f"{reason} dump_shape={list(dump.shape)} "
                    f"live_shape={list(snapshot.shape)}",
                    flush=True,
                )
            else:
                if dump.shape != tuple(snapshot.shape):
                    dump = dump.reshape(tuple(snapshot.shape))
                # float32 first, then cast: `paddle.to_tensor` does not accept
                # float8 dtypes directly.
                loaded = paddle.to_tensor(
                    dump, dtype="float32", place=snapshot.place
                )
                if snapshot.dtype != paddle.float32:
                    loaded = loaded.astype(snapshot.dtype)
                load_f32 = loaded.astype("float32")
                load_np = load_f32.numpy()
                abssum, absmax, load_md5 = _stats(load_np)
                print(
                    f"[ABLATION_load_tensor] loaded {tag} rank={rank} shape={list(loaded.shape)} "
                    f"dtype={loaded.dtype} abssum={abssum} absmax={absmax} md5={load_md5}",
                    flush=True,
                )
                diff = (snapshot.astype("float32") - load_f32).abs()
                mean_abs_diff = diff.mean().item()
                print(
                    f"[ABLATION_load_tensor] diff {tag} max_abs_diff={diff.max().item()} "
                    f"mean_abs_diff={mean_abs_diff} "
                    f"relative_diff={mean_abs_diff / (load_f32.abs().mean().item() + 1e-12)}",
                    flush=True,
                )
                # Per-element breakdown, only when the two sides fail to
                # md5-align (a bit-identical match needs no diff table) and only
                # when ABLATION_INSPECT_DETAIL=1. `arr` is the live training
                # snapshot's host float32 copy from stage 4; `load_np` the loaded
                # inference reference, both at snapshot.shape.
                if _DETAIL and arr is not None and load_md5 != live_md5:
                    _print_detail_diff(tag, layer_idx, rank, load_np, arr)

    # --- 7. return: the override when one was loaded (after `post_load_func`
    #        inverts the view and `index` puts it back into the container),
    #        otherwise the live tensor itself. -------------------------------
    if loaded is None:
        return tensor
    if post_load_func is not None:
        loaded = post_load_func(loaded)
    return _with_element(tensor, index, loaded) if indexed else loaded
