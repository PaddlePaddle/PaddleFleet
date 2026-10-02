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
"""Gated-attention qkv AOA helper (checkpoint <-> model).

Some attention towers fuse an output gate into ``qkv_proj`` alongside q/k/v in a
per-KV-group ``[Q, Gate, K, V]`` layout that the generic ``fused_qkv`` macro
cannot express. The checkpoint keeps the gate as its own tensor ``gate_proj``
(the ERNIE-Lite layout): Q comes from ``q_proj``, the gate from ``gate_proj``,
and each is split then regrouped per KV group into the model ``qkv_proj``.
``head_dim`` may differ from the gate/value head dim, so each per-head slice is
sub-chunked at their GCD before regrouping. The inverse is implemented
independently, never derived from the forward text (the shared pure name-builder
is direction-agnostic).
"""

from __future__ import annotations

import math

from paddle.distributed.flex_checkpoint.aoa.generation import (
    resolve_checkpoint_name_from_anchor,
    resolve_single_name,
)


def _temp_names(base, num_heads, num_kv_groups, head_dim, v_head_dim, gated):
    """Pure name-builder: per-head Q/Gate and per-group K/V temp names.

    Returns ``(q_list, gate_list, k_list, v_list, fused)``. ``fused`` is the
    model per-group concatenation order ``for g: [Q_g, Gate_g, K_g, V_g]``.
    """
    gcd = math.gcd(head_dim, v_head_dim)
    hd_chunks = head_dim // gcd
    vhd_chunks = v_head_dim // gcd
    heads_per_group = num_heads // num_kv_groups
    q_list, gate_list, k_list, v_list, fused = [], [], [], [], []
    for g in range(num_kv_groups):
        q_g, gate_g = [], []
        for h in range(heads_per_group):
            q_h = [f"{base}._q_g{g}_h{h}_c{c}" for c in range(hd_chunks)]
            gate_h = (
                [f"{base}._gate_g{g}_h{h}_c{c}" for c in range(vhd_chunks)]
                if gated
                else []
            )
            q_g += q_h
            gate_g += gate_h
        k_g = [f"{base}._k_g{g}_c{c}" for c in range(hd_chunks)]
        v_g = [f"{base}._v_g{g}_c{c}" for c in range(vhd_chunks)]
        q_list += q_g
        gate_list += gate_g
        k_list += k_g
        v_list += v_g
        fused += q_g + gate_g + k_g + v_g
    return q_list, gate_list, k_list, v_list, fused


def _resolve_names(
    attn, ctx, structured_name_prefix, checkpoint_lookup_drop_segment, bias
):
    """Resolve the model ``qkv_proj`` single name and the checkpoint source
    names (``q_proj`` / ``gate_proj`` / ``k_proj`` / ``v_proj``) via anchor
    resolution."""
    suffix = "bias" if bias else "weight"
    qkv_local = f"qkv_proj.{suffix}"
    qkv_model = resolve_single_name(
        qkv_local,
        structured_name_prefix,
        ctx.pp_to_single_mapping,
        model_name_prefix=ctx.model_name_prefix,
    )

    def ckpt(local):
        return resolve_checkpoint_name_from_anchor(
            qkv_model,
            qkv_local,
            local,
            ctx.checkpoint_name_prefix,
            ctx.checkpoint_name_mapping,
            checkpoint_lookup_drop_segment=checkpoint_lookup_drop_segment,
            model_name_prefix=ctx.model_name_prefix,
        )

    return (
        qkv_model,
        ckpt(f"q_proj.{suffix}"),
        ckpt(f"gate_proj.{suffix}"),
        ckpt(f"k_proj.{suffix}"),
        ckpt(f"v_proj.{suffix}"),
    )


def _gen_qkv_statements(
    attn, ctx, structured_name_prefix, checkpoint_lookup_drop_segment, *, bias
):
    """Checkpoint -> model qkv statements for one of weight / bias."""
    qkv_model, q_ckpt, gate_ckpt, k_ckpt, v_ckpt = _resolve_names(
        attn, ctx, structured_name_prefix, checkpoint_lookup_drop_segment, bias
    )
    gated = attn.gated_attention
    q_names, gate_names, k_names, v_names, fused_names = _temp_names(
        qkv_model,
        num_heads=attn.num_attention_heads,
        num_kv_groups=attn.num_key_value_heads,
        head_dim=attn.head_dim,
        v_head_dim=attn.v_head_dim,
        gated=gated,
    )
    stmts = [f"{q_ckpt} -> {','.join(q_names)}, axis=0"]
    if gated:
        stmts.append(f"{gate_ckpt} -> {','.join(gate_names)}, axis=0")
    stmts.append(f"{k_ckpt} -> {','.join(k_names)}, axis=0")
    stmts.append(f"{v_ckpt} -> {','.join(v_names)}, axis=0")
    if bias:
        stmts.append(f"{','.join(fused_names)} -> {qkv_model}, axis=0")
    else:
        fused_tmp = f"{qkv_model}.qkv_fused_tmp"
        stmts.append(f"{','.join(fused_names)} -> {fused_tmp}, axis=0")
        stmts.append(f"{fused_tmp}^T -> {qkv_model}")
    return stmts


def _gen_inv_qkv_statements(
    attn, ctx, structured_name_prefix, checkpoint_lookup_drop_segment, *, bias
):
    """Model -> checkpoint qkv statements for one of weight / bias."""
    qkv_model, q_ckpt, gate_ckpt, k_ckpt, v_ckpt = _resolve_names(
        attn, ctx, structured_name_prefix, checkpoint_lookup_drop_segment, bias
    )
    gated = attn.gated_attention
    q_names, gate_names, k_names, v_names, fused_names = _temp_names(
        qkv_model,
        num_heads=attn.num_attention_heads,
        num_kv_groups=attn.num_key_value_heads,
        head_dim=attn.head_dim,
        v_head_dim=attn.v_head_dim,
        gated=gated,
    )
    if bias:
        stmts = [f"{qkv_model} -> {','.join(fused_names)}, axis=0"]
    else:
        fused_tmp = f"{qkv_model}.qkv_fused_tmp"
        stmts = [f"{qkv_model}^T -> {fused_tmp}"]
        stmts.append(f"{fused_tmp} -> {','.join(fused_names)}, axis=0")
    stmts.append(f"{','.join(q_names)} -> {q_ckpt}, axis=0")
    if gated:
        stmts.append(f"{','.join(gate_names)} -> {gate_ckpt}, axis=0")
    stmts.append(f"{','.join(k_names)} -> {k_ckpt}, axis=0")
    stmts.append(f"{','.join(v_names)} -> {v_ckpt}, axis=0")
    return stmts


def gen_gated_qkv_aoa(
    attn, ctx, structured_name_prefix, *, checkpoint_lookup_drop_segment=None
):
    """Forward qkv head statements (weight + optional bias) for gated attention.

    Replaces the standard ``fused_qkv`` head of the base ``SelfAttention``.
    """
    stmts = _gen_qkv_statements(
        attn,
        ctx,
        structured_name_prefix,
        checkpoint_lookup_drop_segment,
        bias=False,
    )
    if attn.qkv_proj.bias is not None:
        stmts += _gen_qkv_statements(
            attn,
            ctx,
            structured_name_prefix,
            checkpoint_lookup_drop_segment,
            bias=True,
        )
    return stmts


def gen_gated_qkv_inv_aoa(
    attn, ctx, structured_name_prefix, *, checkpoint_lookup_drop_segment=None
):
    """Inverse qkv head statements (weight + optional bias) for gated attention.

    Independently mirrors :func:`gen_gated_qkv_aoa`.
    """
    stmts = _gen_inv_qkv_statements(
        attn,
        ctx,
        structured_name_prefix,
        checkpoint_lookup_drop_segment,
        bias=False,
    )
    if attn.qkv_proj.bias is not None:
        stmts += _gen_inv_qkv_statements(
            attn,
            ctx,
            structured_name_prefix,
            checkpoint_lookup_drop_segment,
            bias=True,
        )
    return stmts
