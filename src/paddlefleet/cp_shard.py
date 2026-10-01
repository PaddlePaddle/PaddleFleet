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

"""Sequence coordinates under context parallelism (CP).

Every tensor that runs along the sequence axis lives in exactly one of two
coordinate systems:

* **global** -- length ``L``, positions ``0 .. L-1`` of the whole packed
  sequence. Packed rolls, embedding lookups and full RoPE tables are built here.
* **local** -- length ``L / cp``, the share this rank owns, laid out by
  ``config.cp_balance_mode`` (two mirrored chunks for ``dualchunk_allgather``,
  one contiguous chunk for ``contiguous_allgather``). Transformer layers, the
  router and cross-entropy work here.

Batch keys have a fixed coordinate:

* global: ``cu_seqlens_q``, ``attn_mask_startend_row_indices``,
  ``mtp_full_input_ids``, ``position_ids``, ``labels`` as delivered.
* local: ``hidden_states`` (and every MTP slot), ``rotary_pos_emb`` /
  ``rotary_pos_cos`` / ``rotary_pos_sin`` and their SWA twins, the labels fed
  to cross-entropy.

The rule: **whoever finishes a tensor's global work calls** :func:`to_cp_local`
**on it, once.** Consumers never shard; they may assert the length.
:func:`to_cp_local` is deliberately not idempotent -- a tensor that is already
local is cut again to ``L / cp**2``. Do not add "skip if it looks sharded":
a shape test turns a missing or duplicated transform into a silent error.

:func:`to_cp_global` is the inverse and is only for the few places that must
see the whole sequence again after local compute (loss normalisation).

Two neighbouring operations are *not* this transform and keep their own code:

* attention's global mask handling -- Q is local, K/V are gathered back to
  ``L``, and ``attn_mask_startend_row_indices`` keeps global row numbers that
  ``preprocess_index*`` shifts per chunk (``context_parallel_utils``);
* the sequence-parallel ``ScatterOp`` -- another axis and another group,
  always applied after the CP transform.

How the transform is implemented depends on where the batch comes from, see
:func:`cp_shard_source`.
"""

from __future__ import annotations

import paddle

from paddlefleet import parallel_state
from paddlefleet.context_parallel_utils import (
    ContextParallelGatherOp,
    ContextParallelScatterOp,
)

__all__ = [
    "SUPPORTED_LOCAL_SLICE_MODES",
    "cp_shard_source",
    "embedding_grad_is_cp_gathered",
    "extract_local_contiguous_chunk",
    "extract_local_cp_chunks",
    "extract_local_zigzag_chunks",
    "to_cp_global",
    "to_cp_local",
]

# Layouts the local-slice implementation reproduces. contiguous_a2a shards the
# sequence contiguously too, but its mask contract differs
# (DotProductAttention.forward skips expand_attn_mask_startend_row_indices_for_cp
# under a2a) and has never run on the local-slice path.
SUPPORTED_LOCAL_SLICE_MODES = ("dualchunk_allgather", "contiguous_allgather")


def cp_shard_source(config, cp_size=None):
    """Where the global -> local transform happens for this configuration.

    * ``None`` -- ``cp_size == 1``; nothing to transform.
    * ``"scatter"`` -- ``experimental_dataflow``: the model receives global
      tensors and applies ``ContextParallelScatterOp``. Its backward all-gathers
      the gradient, so every rank ends up holding the full-length gradient of
      whatever produced the tensor (see :func:`embedding_grad_is_cp_gathered`).
    * ``"local"`` -- ``use_erndata``: the loader broadcasts global tensors to the
      whole CP group and the model takes its share with a plain slice. The
      backward returns only this rank's share; the default CP gradient scaling
      reassembles it.
    * ``"loader"`` -- neither flag: the trainer already sharded the batch before
      the model (``get_batch_on_this_cp_rank``). The transform point for this
      source lies outside the model, so :func:`to_cp_local` is the identity.

    Args:
        config: model config (a ``TransformerConfig`` or duck-typed test config).
        cp_size: CP world size. Defaults to the runtime CP group; layer
            constructors pass ``config.context_parallel_size`` instead because
            the group may not be initialised yet.
    """
    if cp_size is None:
        cp_size = parallel_state.get_context_parallel_world_size()
    if cp_size is None or cp_size <= 1:
        return None
    if getattr(config, "experimental_dataflow", False):
        return "scatter"
    if getattr(config, "use_erndata", False):
        return "local"
    return "loader"


def embedding_grad_is_cp_gathered(config):
    """Whether embedding parameters must skip the default CP gradient scaling.

    ``ContextParallelScatterOp`` hands every rank the full-length gradient, so
    scaling it by ``cp_size`` would over-count. This is the same decision as
    picking the ``"scatter"`` implementation in :func:`to_cp_local`; keep them
    tied to one predicate so the two cannot disagree.
    """
    return (
        cp_shard_source(config, getattr(config, "context_parallel_size", 1))
        == "scatter"
    )


def to_cp_local(x, config, axis=1):
    """Move ``x`` from global to local sequence coordinates.

    The caller asserts that ``x`` is global and that its global work is done.
    Returns ``x`` unchanged when it is ``None``, ``cp_size == 1``, or the batch
    was sharded by the loader. Not idempotent -- see the module docstring.
    """
    if x is None:
        return x
    source = cp_shard_source(config)
    if source == "scatter":
        return ContextParallelScatterOp.apply(
            x, axis=axis, mode=config.cp_balance_mode
        )
    if source == "local":
        return extract_local_cp_chunks(
            x,
            parallel_state.get_context_parallel_rank(),
            parallel_state.get_context_parallel_world_size(),
            axis=axis,
            mode=config.cp_balance_mode,
        )
    return x


def to_cp_global(x, config, axis=1):
    """Move ``x`` from local back to global sequence coordinates.

    Reserved for consumers that must see the whole sequence after local compute.
    Every source keeps its local tensors in the same layout, so the inverse is
    the same all-gather for all of them. ``None`` or ``cp_size == 1`` returns
    ``x`` unchanged.
    """
    if x is None or cp_shard_source(config) is None:
        return x
    return ContextParallelGatherOp.apply(
        x, axis=axis, mode=config.cp_balance_mode
    )


# --------------------------------------------------------------------------- #
# Local-slice implementation. Communication-free counterparts of
# scatter_balance / scatter_contiguous for tensors every rank already holds in
# full. Callers go through to_cp_local; these stay public for layout tests.
# --------------------------------------------------------------------------- #


def extract_local_zigzag_chunks(tensor_full, cp_rank, cp_size, axis=1):
    """Extract this CP rank's zigzag chunks from a full-length tensor.

    Mirrors ``context_parallel_utils.scatter_balance``: each rank owns two
    chunks —

    * ``chunk_start = tensor_full[..., interval*r : interval*(r+1), ...]``
    * ``chunk_end   = tensor_full[..., L-interval*(r+1) : L-interval*r, ...]``

    concatenated along the seq axis.

    Extraction only — no CP communication.

    Args:
        tensor_full: ``[..., L, ...]`` full-length tensor available on every rank.
        cp_rank: this rank's index within the CP group.
        cp_size: CP world size. ``cp_size == 1`` returns ``tensor_full`` unchanged.
        axis: sequence axis (default 1 for ``[B, L, ...]``).

    Returns:
        ``[..., L / cp_size, ...]`` tensor holding this rank's zigzag chunks.
    """
    if cp_size == 1:
        return tensor_full
    ndim = tensor_full.dim()
    dim = axis if axis >= 0 else ndim + axis
    seq_len = tensor_full.shape[dim]
    if seq_len % (cp_size * 2) != 0:
        raise ValueError(
            f"extract_local_zigzag_chunks: seq_len={seq_len} on axis={axis} "
            f"is not divisible by 2*cp_size={2 * cp_size}."
        )
    interval = seq_len // cp_size // 2
    chunk_start = paddle.slice(
        tensor_full,
        axes=[dim],
        starts=[interval * cp_rank],
        ends=[interval * (cp_rank + 1)],
    )
    chunk_end = paddle.slice(
        tensor_full,
        axes=[dim],
        starts=[seq_len - interval * (cp_rank + 1)],
        ends=[seq_len - interval * cp_rank],
    )
    return paddle.concat([chunk_start, chunk_end], axis=dim)


def extract_local_contiguous_chunk(tensor_full, cp_rank, cp_size, axis=1):
    """Extract this CP rank's contiguous chunk from a full-length tensor.

    Mirrors ``context_parallel_utils.scatter_contiguous``: rank ``r`` owns the
    single slice ``tensor_full[..., chunk*r : chunk*(r+1), ...]`` with
    ``chunk = L / cp_size``.

    Extraction only — no CP communication, same contract as
    ``extract_local_zigzag_chunks``.
    """
    if cp_size == 1:
        return tensor_full
    ndim = tensor_full.dim()
    dim = axis if axis >= 0 else ndim + axis
    seq_len = tensor_full.shape[dim]
    if seq_len % cp_size != 0:
        raise ValueError(
            f"extract_local_contiguous_chunk: seq_len={seq_len} on axis={axis} "
            f"is not divisible by cp_size={cp_size}."
        )
    chunk = seq_len // cp_size
    # Deliberately a bare slice, unlike scatter_contiguous's paddle.assign: the
    # per-depth caller keeps only this result, so a view holds F while a copy
    # holds F + F/cp until the source is freed. Measured peaks over the roll
    # loop are (2K+1)F for views vs 2F + K*F + (K+1)F/cp for assign -- worse at
    # K=1 (every erndata model config here), even at K=3. The dominant term in
    # both is roll_tensor's own grad-node retention, which neither changes.
    return paddle.slice(
        tensor_full,
        axes=[dim],
        starts=[chunk * cp_rank],
        ends=[chunk * (cp_rank + 1)],
    )


def extract_local_cp_chunks(tensor_full, cp_rank, cp_size, axis=1, *, mode):
    """Layout-aware local slice of a tensor every CP rank holds in full.

    The ``"local"`` implementation of :func:`to_cp_local`. The slice must match
    the layout the rest of the model scatters with, i.e.
    ``config.cp_balance_mode``:

    * ``dualchunk_allgather``  -> ``scatter_balance``    -> two zigzag chunks
    * ``contiguous_allgather`` -> ``scatter_contiguous`` -> one contiguous chunk

    ``contiguous_allgather`` is mandatory for the DSv4 hybrid stack, whose
    attention layers assert on it under CP, so hard-coding zigzag here is wrong.

    Args:
        tensor_full: ``[..., L, ...]`` full-length tensor present on every rank.
        cp_rank: this rank's index inside the CP group.
        cp_size: CP world size; ``1`` returns ``tensor_full`` unchanged.
        axis: sequence axis (default 1 for ``[B, L, ...]``).
        mode: ``config.cp_balance_mode``. Keyword-only and required: the bug this
            helper exists to fix was a call site that assumed a layout, and the
            wrong layout is a silently wrong loss rather than a crash.

    Returns:
        ``[..., L / cp_size, ...]`` tensor holding this rank's slice.

    Note:
        ``cp_size == 1`` returns ``tensor_full`` itself, not a copy — do not
        write into the result in place.
    """
    if cp_size == 1:
        return tensor_full
    if mode == "dualchunk_allgather":
        return extract_local_zigzag_chunks(
            tensor_full, cp_rank, cp_size, axis=axis
        )
    if mode == "contiguous_allgather":
        return extract_local_contiguous_chunk(
            tensor_full, cp_rank, cp_size, axis=axis
        )
    # See SUPPORTED_LOCAL_SLICE_MODES for why contiguous_a2a is refused.
    raise ValueError(
        f"extract_local_cp_chunks: unsupported cp_balance_mode={mode!r}; "
        f"expected one of {SUPPORTED_LOCAL_SLICE_MODES}."
    )
