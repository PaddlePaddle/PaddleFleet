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

"""Last-dim segment views for train_infer_consistent_inspect (training side).

The fused MLA RoPE path compares only a slice of its query / key buffer along the
last dim: the rotated `[..., qk_nope_head_dim:]` part, or the nope part the
rotation leaves alone. `last_dim_segment` is the comparable view of such a
segment and `scatter_last_dim_segment` its inverse, so the probe stays the plain
`inspect_tensor(..., pre_save_func=..., post_load_func=...)`.

Spelling the same thing at the call site as
`q[..., n:] = inspect_tensor(tag, layer, q[..., n:])` looks equivalent but is
not: the slice and the `__setitem__` both run even with the probes off. The slice
launches a kernel, and the setitem is a real in-place write that bumps `q`'s
dygraph inplace version (measured: 0 -> 1 per call with ABLATION_INSPECT_TENSOR
unset), so a probes-off forward stops matching an unprobed one -- and once such a
buffer is modified after a grad node saved it, backward / recompute dies with
`PermissionDeniedError: Tensor ... has been modified by an inplace operation`.
Handed to the probe as hooks instead, the slice happens only once the probe is
live, and the inverse rebuilds a fresh tensor with `concat` rather than writing
into the live buffer at all.

File layout: `pre_save_func` views first, then their `post_load_func` inverses.
No entry points live here -- the probe is always plain `inspect_tensor`.
"""

from __future__ import annotations

import paddle

# ---------------------------------------------------------------------------
# pre_save_func views: full-width buffer -> one segment of the last dim
# ---------------------------------------------------------------------------


def last_dim_segment(tensor, start=0, end=None):
    """`tensor[..., start:end]` -- the segment the two sides compare.

    Args:
        tensor: the live full-width tensor.
        start: first index of the segment along the last dim.
        end: one past its last index, or None for "through the end".

    Returns:
        The segment, or None when there is no tensor to view -- which aborts the
        probe the same way a missing `pre_save_func` value does.
    """
    if tensor is None:
        return None
    return tensor[..., start:] if end is None else tensor[..., start:end]


# ---------------------------------------------------------------------------
# post_load_func inverses: a loaded segment -> the full-width buffer
# ---------------------------------------------------------------------------


def scatter_last_dim_segment(tensor, segment, start=0, end=None):
    """Inverse of `last_dim_segment`: `tensor` with that segment replaced.

    Always a fresh `concat`, never an in-place write into `tensor`: the live
    buffer is usually still held by the autograd graph, and bumping its inplace
    version breaks backward / recompute (see the module docstring).

    Gradients flow through the head / tail slices taken from `tensor` but not
    through `segment`: a loaded dump arrives with `stop_gradient=True`, the same
    forward-only tradeoff `inspect_tensor` documents.
    """
    if segment is None:
        return tensor
    width = tensor.shape[-1]
    stop = width if end is None else end
    parts = []
    if start > 0:
        parts.append(tensor[..., :start])
    parts.append(segment.astype(tensor.dtype))
    if stop < width:
        parts.append(tensor[..., stop:])
    # A segment spanning the whole last dim already is the full-width tensor.
    return paddle.concat(parts, axis=-1) if len(parts) > 1 else parts[0]
