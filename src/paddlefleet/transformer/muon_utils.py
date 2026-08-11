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
import paddle.distributed as dist


def ortho_blocks(weight, ortho_fn, sizes, axis=-1):
    """Split ``weight`` along ``axis``, orthogonalise each block, concat back.

    ``sizes`` is either an int (that many equal-width blocks) or a list of
    per-block widths, matching ``paddle.split``.
    """
    blocks = paddle.split(weight, sizes, axis=axis)
    return paddle.concat([ortho_fn(b) for b in blocks], axis=axis)


def ortho_per_head(weight, ortho_fn, heads=1, head_sizes=None, axis=-1):
    """Orthogonalise a projection head by head.

    ``head_sizes`` describes the widths a single head is made of (e.g.
    ``[nope_dim, rope_dim]`` or ``[k_dim, v_dim]``); each of those pieces is
    orthogonalised separately. When it is None a head is one contiguous block
    and the weight is simply cut into ``heads`` equal parts.
    """
    sizes = heads if head_sizes is None else head_sizes * heads
    return ortho_blocks(weight, ortho_fn, sizes, axis=axis)


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


def ortho_ep_full_intermediate(
    weight,
    ortho_fn,
    ep_group=None,
    shard_axis=-1,
    gate_up=False,
    split_gate_up=True,
):
    """Orthogonalise EP-sharded expert weights on their full intermediate dim.

    With the 'allgather' MoE dispatcher and EP > 1 a rank holds *every* expert
    but only ``moe_intermediate_size // EP`` of each one, so a local
    Newton-Schulz would orthogonalise a slab instead of the real matrix. This
    wrapper redistributes to the traditional EP layout (``E // EP`` experts,
    full intermediate) with an all-to-all, orthogonalises there, and sends the
    result back the same way, reproducing the update a non-EP run would make.

    ``shard_axis`` is the axis the intermediate dim was sharded along: -1 for
    the fc1 weight ``[E, H, 2I/EP]``, -2 for the fc2 weight ``[E, I/EP, H]``.

    Set ``gate_up`` when ``shard_axis`` packs a fused ``[gate | up]`` pair: each
    rank then owns ``[gate_shard | up_shard]``, so the halves must be
    de-interleaved before they can be concatenated into full-width matrices.
    ``split_gate_up`` (i.e. ``muon_ffn_split``) then decides whether gate and up
    are orthogonalised independently or as one fused matrix.
    """
    if weight.ndim != 3:
        raise ValueError(
            f"EP redistribution expects a 3D expert tensor, got {weight.shape}"
        )
    if ep_group is None:
        raise ValueError(
            "EP redistribution requires an expert-parallel process group"
        )

    ep_size = ep_group.nranks
    if weight.shape[0] % ep_size != 0:
        raise ValueError(
            f"Expert axis {weight.shape[0]} is not divisible by EP={ep_size}."
        )
    if gate_up and weight.shape[shard_axis] % 2 != 0:
        raise ValueError(
            f"Fused gate/up axis must be even, got shape {weight.shape}."
        )

    # Splitting a contiguous tensor along axis 0 yields contiguous blocks, so
    # the forward all-to-all needs no extra copy.
    send = list(paddle.split(weight, ep_size, axis=0))
    recv = [paddle.empty_like(send[0]) for _ in send]
    dist.alltoall(recv, send, group=ep_group)

    if not gate_up:
        owned = ortho_fn(paddle.concat(recv, axis=shard_axis))
        back = [
            chunk.contiguous()
            for chunk in paddle.split(owned, ep_size, axis=shard_axis)
        ]
    else:
        gates, ups = zip(
            *(paddle.split(chunk, 2, axis=shard_axis) for chunk in recv)
        )
        gate_in = paddle.concat(gates, axis=shard_axis)
        up_in = paddle.concat(ups, axis=shard_axis)

        if split_gate_up:
            # Stack on the expert axis so a single batched Newton-Schulz still
            # treats gate and up as independent matrices.
            n_experts = gate_in.shape[0]
            stacked = ortho_fn(paddle.concat([gate_in, up_in], axis=0))
            gate_out, up_out = stacked[:n_experts], stacked[n_experts:]
        else:
            gate_out, up_out = paddle.split(
                ortho_fn(paddle.concat([gate_in, up_in], axis=shard_axis)),
                2,
                axis=shard_axis,
            )

        back = [
            paddle.concat([gate, up], axis=shard_axis)
            for gate, up in zip(
                paddle.split(gate_out, ep_size, axis=shard_axis),
                paddle.split(up_out, ep_size, axis=shard_axis),
            )
        ]

    reverse = [paddle.empty_like(back[0]) for _ in back]
    dist.alltoall(reverse, back, group=ep_group)
    return paddle.concat(reverse, axis=0)


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
