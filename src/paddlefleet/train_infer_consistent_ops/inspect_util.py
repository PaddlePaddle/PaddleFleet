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

import hashlib
import math
import os

import paddle

_MODULE_PREFIX = "[ABLATION_train]"

# Snapshot of the ABLATION_* configuration, read once at import.
#
# `os.environ` is an `os._Environ`, not a dict: every `.get` runs a Python-level
# `__getitem__` plus fsencode/fsdecode and costs ~0.6us, which the disabled path
# would pay at each of the ~1k probes a forward passes through. The launcher
# exports these before the process starts, so reading them once is enough.
_ENABLED = False
_WHITELIST = frozenset()
_BLACKLIST = frozenset()
_DUMP_SKIP_TAGS = frozenset()
_SAVE_PATH = ""
_LOAD_PATH = ""


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
        _LOAD_PATH
    env = os.environ
    _ENABLED = env.get("ABLATION_INSPECT_TENSOR", "0") == "1"
    _WHITELIST = frozenset(
        filter(None, env.get("ABLATION_TAG_WHITELIST", "").split(","))
    )
    _BLACKLIST = frozenset(
        filter(None, env.get("ABLATION_TAG_BLACKLIST", "").split(","))
    )
    _DUMP_SKIP_TAGS = frozenset(
        filter(None, env.get("ABLATION_DUMP_SKIP_TAGS", "").split(","))
    )
    _SAVE_PATH = env.get("ABLATION_SAVE_TENSOR_PATH", "")
    _LOAD_PATH = env.get("ABLATION_LOAD_TENSOR_PATH", "")


refresh_env_cache()


def inspect_enabled():
    """True when train_infer_consistent_inspect is on.

    Switched by `ABLATION_INSPECT_TENSOR=1`, snapshotted once at import.
    """
    return _ENABLED


def inspect_tag_enabled(tag):
    """True when the probes are on *and* `tag` survives the tag filters.

    The same gate `inspect_tensor` applies in its stage 1, exposed for the one
    helper that has to change what the model computes to make a probe comparable
    (`inspect_tensor_force_unit_probs`). That change has to follow the filters
    too: narrowing a run down to a few tags must not keep rewriting the math
    behind a probe nobody asked for.
    """
    return _ENABLED and not (
        (_WHITELIST and tag not in _WHITELIST) or tag in _BLACKLIST
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
            load).
        ABLATION_TAG_BLACKLIST: comma-separated tags to return early on.
        ABLATION_SAVE_TENSOR_PATH: directory to dump `.npy` files into.
        ABLATION_LOAD_TENSOR_PATH: directory with the other side's `.npy` dumps.
        ABLATION_DUMP_SKIP_TAGS: comma-separated tags to skip for saving/loading
            only (they are still printed).

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
        abssum, absmax, md5 = _stats(arr)
        print(
            f"{_MODULE_PREFIX} tag={tag} rank={rank} layer={layer_idx} "
            f"abssum={abssum} absmax={absmax} md5={md5} shape={list(snapshot.shape)} dtype={snapshot.dtype}",
            flush=True,
        )

    # --- 5. save: dump the snapshot to rank_<r>/layer_<l>/<tag>.npy -----------
    if arr is not None and save and _SAVE_PATH and tag not in _DUMP_SKIP_TAGS:
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
    if load and _LOAD_PATH and tag not in _DUMP_SKIP_TAGS:
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
                abssum, absmax, md5 = _stats(load_f32.numpy())
                print(
                    f"[ABLATION_load_tensor] loaded {tag} rank={rank} shape={list(loaded.shape)} "
                    f"dtype={loaded.dtype} abssum={abssum} absmax={absmax} md5={md5}",
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

    # --- 7. return: the override when one was loaded (after `post_load_func`
    #        inverts the view and `index` puts it back into the container),
    #        otherwise the live tensor itself. -------------------------------
    if loaded is None:
        return tensor
    if post_load_func is not None:
        loaded = post_load_func(loaded)
    return _with_element(tensor, index, loaded) if indexed else loaded
