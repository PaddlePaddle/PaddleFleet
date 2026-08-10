# Copyright (c) 2024 Baidu, Inc. All Rights Reserved.
"""Muon orthogonal-slice helpers.

A fused weight often packs several logically independent matrices into one
tensor (Q/K/V, gate/up, stacked experts). Muon must orthogonalise each of them
on its own, so every helper here follows the same shape: split the weight along
one axis, hand each block to ``ortho_fn``, concatenate the results back into the
original layout.

The transformer submodules own the knowledge of *how* their weights are packed
and reference these helpers from ``muon_slice_specs``; the helpers themselves
know nothing about muon configs or module structure.
"""

import paddle


def _transposing(ortho_fn):
    """``ortho_fn`` applied to the transpose of each block, transposed back.

    Muon's scaling is not symmetric in the two matrix dims for
    ``muon_version`` 1 / 2 (``dout / din``), so a block stored transposed
    relative to the layout the update was tuned on would be scaled by the
    reciprocal ratio. Orthogonalising the transpose restores the original
    semantics for every version. Batched (3-D) blocks swap their last two dims.
    """

    def wrapped(block):
        perm = list(range(block.ndim))
        perm[-2], perm[-1] = perm[-1], perm[-2]
        return paddle.transpose(ortho_fn(paddle.transpose(block, perm)), perm)

    return wrapped


def ortho_blocks(weight, ortho_fn, sizes, axis=-1, transposed=False):
    """Split ``weight`` along ``axis``, orthogonalise each block, concat back.

    ``sizes`` is either an int (that many equal-width blocks) or a list of
    per-block widths, matching ``paddle.split``. ``transposed=True`` hands each
    block to ``ortho_fn`` transposed (see ``_transposing``).
    """
    if transposed:
        ortho_fn = _transposing(ortho_fn)
    blocks = paddle.split(weight, sizes, axis=axis)
    return paddle.concat([ortho_fn(b) for b in blocks], axis=axis)


def ortho_per_head(
    weight, ortho_fn, heads=1, head_sizes=None, axis=-1, transposed=False
):
    """Orthogonalise a projection head by head.

    ``head_sizes`` describes the widths a single head is made of (e.g.
    ``[nope_dim, rope_dim]`` or ``[k_dim, v_dim]``); each of those pieces is
    orthogonalised separately. When it is None a head is one contiguous block
    and the weight is simply cut into ``heads`` equal parts.

    ``transposed=True`` orthogonalises each block in its transpose, for weights
    stored transposed relative to the layout Muon's scaling was tuned on.
    """
    sizes = heads if head_sizes is None else head_sizes * heads
    return ortho_blocks(
        weight, ortho_fn, sizes, axis=axis, transposed=transposed
    )


def ortho_gate_up(weight, ortho_fn):
    """Orthogonalise the gate and up halves of a fused FFN weight separately.

    Works for 2D (single expert) and 3D (stacked experts) tensors. The split
    point comes from the tensor shape, so under tensor parallelism each rank
    splits its own shard rather than a global size.
    """
    assert weight.ndim == 2 or weight.ndim == 3, (
        "FFN gate_up split expects 2D or 3D tensor"
    )

    half = weight.shape[-1] // 2
    return ortho_blocks(weight, ortho_fn, [half, half])


def ortho_stacked(weight, ortho_fn):
    """Orthogonalise a stacked 3D weight, one matrix per leading-dim entry.

    Used for tensors whose first axis enumerates independent matrices (fused
    MoE experts, VHA premix); ``ortho_fn`` handles the leading axis as a batch.
    """
    if weight.ndim != 3:
        raise ValueError(
            f"Stacked split expects 3D tensor, got shape {weight.shape}"
        )

    return ortho_fn(weight)


def ortho_qkv_interleaved(
    weight,
    ortho_fn,
    groups=None,
    role_sizes=None,
    heads_per_group=None,
    per_head=True,
):
    """Orthogonalise a fused QKV weight laid out group by group.

    The weight is ``[group_0(Q, [Gate,] K, V), group_1(...), ...]`` where
    ``role_sizes`` gives the widths of one group's roles (3 entries without
    output gating, 4 with).

    ``per_head=True`` orthogonalises every head separately: Q (and Gate) are
    cut into ``heads_per_group`` heads, K and V are one head each.
    ``per_head=False`` instead gathers each role across all groups and
    orthogonalises it as a single matrix, then scatters it back.
    """
    group_slices = [
        paddle.split(group, role_sizes, axis=-1)
        for group in paddle.split(weight, groups, axis=-1)
    ]

    if not per_head:
        out_by_role = [
            paddle.split(
                ortho_fn(paddle.concat(list(role_parts), axis=-1)),
                groups,
                axis=-1,
            )
            for role_parts in zip(*group_slices)
        ]
        return paddle.concat(
            [
                paddle.concat([role[i] for role in out_by_role], axis=-1)
                for i in range(groups)
            ],
            axis=-1,
        )

    # Q (and Gate) span heads_per_group heads; K and V are a single head.
    heads_by_role = [heads_per_group] * (len(role_sizes) - 2) + [1, 1]
    return paddle.concat(
        [
            paddle.concat(
                [
                    ortho_blocks(part, ortho_fn, heads)
                    for part, heads in zip(parts, heads_by_role)
                ],
                axis=-1,
            )
            for parts in group_slices
        ],
        axis=-1,
    )


def ortho_qkv_contiguous(
    weight,
    ortho_fn,
    heads=None,
    groups=None,
    head_dim=None,
    v_head_dim=None,
):
    """Orthogonalise a fused QKV weight laid out as ``[all_Q | all_K | all_V]``.

    Every Q head, K head and V head is orthogonalised on its own. Output gating
    lives in a separate weight for this layout, so there is no Gate role here.
    """
    role_sizes = [heads * head_dim, groups * head_dim, groups * v_head_dim]
    heads_by_role = [heads, groups, groups]
    return paddle.concat(
        [
            ortho_blocks(part, ortho_fn, count)
            for part, count in zip(
                paddle.split(weight, role_sizes, axis=-1), heads_by_role
            )
        ],
        axis=-1,
    )
