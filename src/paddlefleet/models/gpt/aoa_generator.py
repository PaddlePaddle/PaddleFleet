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
"""Whole-model modular AOA statement generation for boundary models.

Pure functions backing the ``GPTModel`` boundary's modular AOA generation, plus
the container entries a multi-tower model uses to drive several roots. The flow
is:

1. :func:`build_aoa_context` builds the read-only ``AOAContext`` once, using the
   live model's ``_pp_to_single_mapping`` (same source as ``sharded_state_dict``,
   so names come from the live module tree rather than a re-derived guess).
2. :func:`collect_alias_plan` resolves tied / shared aliases and the pipeline
   names to skip in one pass, **before** recursion (no post-hoc text dedup).
   The skip set is pinned into ``ctx.excluded_names`` so every component
   override honors it.
3. :func:`gen_whole_model_aoa` / :func:`gen_whole_model_inv_aoa` hand each child
   to the standard ``Layer.gen_aoa_statements`` / ``gen_inv_aoa_statements``
   protocol -- that polymorphic call is the component dispatch point, so the
   module walk is never re-implemented here. The two directions are generated
   independently and never derived from one another.
4. :func:`gen_multi_tower_aoa` / :func:`gen_multi_tower_inv_aoa` are the entry
   points for a container that owns several roots (a vision tower beside a
   language model). Every tower is dispatched on what its root is: a boundary
   root goes through step 3, any other root through plain component recursion.
   The container keeps globalization to itself so it runs once over the union
   of all its towers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from paddle.distributed.flex_checkpoint.aoa.generation import (
    AOAContext,
    validate_checkpoint_name_mapping,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MTP checkpoint prefix protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MTPCheckpointPrefixSpec:
    """Checkpoint-side prefix protocol for MTP layers, resolved once per pass.

    Producer-side state built at the whole-model entry from the model-declared
    ``aoa_mtp_*`` attributes and threaded straight into
    :func:`_resolve_mtp_scopes`. Deliberately *not* a field of the
    component-facing :class:`AOAContext`: no component override reads it, so it
    does not belong on the frozen recursion context forwarded to every
    component.

    Defaults cover the common case: MTP layers share the normal-layer
    checkpoint numbering ``layers.$LAYER_ID`` and the inner transformer inherits
    that same prefix. A boundary overrides via the three model attributes.
    """

    checkpoint_prefix: str = "layers.$LAYER_ID"
    transformer_checkpoint_prefix: str = "layers.$LAYER_ID"
    is_absolute: bool = False


def resolve_mtp_checkpoint_prefix_spec(config) -> MTPCheckpointPrefixSpec:
    """Normalizes the model-declared MTP checkpoint-prefix attributes.

    Reads ``aoa_mtp_checkpoint_prefix`` /
    ``aoa_mtp_transformer_checkpoint_prefix`` /
    ``aoa_mtp_checkpoint_prefix_absolute`` off ``config`` with the defaulting an
    unmigrated model relies on: the MTP-own prefix defaults to
    ``layers.$LAYER_ID`` (same numbering as normal layers), the inner
    transformer prefix inherits the MTP-own prefix, and no absolute-prefix
    escape. Both prefixes route through :func:`_protocol_value`, so only an
    absent / ``None`` attribute falls back; an explicitly declared value
    (including an empty ``""`` for a top-level checkpoint) is taken verbatim.
    """
    checkpoint_prefix = _protocol_value(
        config, "aoa_mtp_checkpoint_prefix", "layers.$LAYER_ID"
    )
    return MTPCheckpointPrefixSpec(
        checkpoint_prefix=checkpoint_prefix,
        transformer_checkpoint_prefix=_protocol_value(
            config, "aoa_mtp_transformer_checkpoint_prefix", checkpoint_prefix
        ),
        is_absolute=bool(
            getattr(config, "aoa_mtp_checkpoint_prefix_absolute", False)
        ),
    )


# ---------------------------------------------------------------------------
# Checkpoint protocol defaults and context construction
# ---------------------------------------------------------------------------


# Defaults for the model-declarable ``aoa_*`` checkpoint protocol. Each default
# is named exactly once here and referenced by every consumer (the
# :class:`FleetAOAContext` fields, :func:`build_aoa_context` and
# :func:`_build_tower_ctx`), so no default literal is duplicated. The values
# describe the ERNIE-series self-developed checkpoint; an external model
# overrides only the attributes whose layout diverges.
#
# Only naming divergences belong in the mapping below: names the components
# already emit identically (layernorms, ``model.norm``, the MTP layer's
# ``enorm`` / ``hnorm`` / ``eh_proj``, and the CSA subtree) are deliberately
# absent. The logical layer root also intentionally omits ``transformer_layer``
# so the same entries cover ordinary layers and an MTP layer's inner
# transformer.
DEFAULT_CHECKPOINT_NAME_MAPPING = {
    "embedding.embed_tokens.weight": "embed_tokens.weight",
    "layers.$LAYER_ID.mlp.gate.weight": "layers.$LAYER_ID.block_sparse_moe.gate.weight",
    "layers.$LAYER_ID.mlp.gate.weight_1": "layers.$LAYER_ID.block_sparse_moe.gate.weight_1",
    "layers.$LAYER_ID.mlp.gate.routed_scaling_factor_param": "layers.$LAYER_ID.block_sparse_moe.gate.routed_scaling_factor_param",
    "layers.$LAYER_ID.mlp.gate.e_score_correction_bias": "layers.$LAYER_ID.block_sparse_moe.e_score_correction_bias",
    "layers.$LAYER_ID.mlp.fc1_latent_proj.weight": "layers.$LAYER_ID.block_sparse_moe.fc1_latent_proj.weight",
    "layers.$LAYER_ID.mlp.fc2_latent_proj.weight": "layers.$LAYER_ID.block_sparse_moe.fc2_latent_proj.weight",
    "layers.$LAYER_ID.mlp.experts.$EXPERT_ID.gate_proj.weight": "layers.$LAYER_ID.block_sparse_moe.experts.$EXPERT_ID.w1.weight",
    "layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_proj.weight": "layers.$LAYER_ID.block_sparse_moe.experts.$EXPERT_ID.w3.weight",
    "layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight": "layers.$LAYER_ID.block_sparse_moe.experts.$EXPERT_ID.w2.weight",
    "layers.$LAYER_ID.mlp.shared_experts.gate_proj.weight": "layers.$LAYER_ID.block_sparse_moe.shared_experts.w1.weight",
    "layers.$LAYER_ID.mlp.shared_experts.up_proj.weight": "layers.$LAYER_ID.block_sparse_moe.shared_experts.w3.weight",
    "layers.$LAYER_ID.mlp.shared_experts.down_proj.weight": "layers.$LAYER_ID.block_sparse_moe.shared_experts.w2.weight",
    "layers.$LAYER_ID.norm.weight": "layers.$LAYER_ID.shared_head.norm.weight",
}
DEFAULT_CHECKPOINT_NAME_PREFIX = "model"
DEFAULT_DTYPE_CAST_RULES = {}
DEFAULT_GATE_CHECKPOINT_LAYOUT = "separate"
DEFAULT_QKV_CHECKPOINT_PREFUSED = False
DEFAULT_MOE_EXPERT_CHECKPOINT_LAYOUT = "per_expert"
DEFAULT_MLP_GATE_UP_FUSED = True
DEFAULT_MAPPING_PROJ_CHECKPOINT_TRANSPOSED = False

# Model-side single-name root, taken from the live model rather than declared on
# the config; kept next to its checkpoint-side counterpart for readability.
DEFAULT_MODEL_NAME_PREFIX = "model"


@dataclass(frozen=True)
class FleetAOAContext(AOAContext):
    """Adds the checkpoint-layout declarations this library's components consume.

    Paddle owns the recursion contract (names, dtype rules, exclusions); the
    layouts below are paddlefleet's own protocol and grow with the models it
    supports, so they live here rather than in the framework. Each is a
    checkpoint-format fact that cannot be inferred from the live module, so the
    model declares it; it only selects an emit branch and never overrides the
    live structure.
    """

    gate_checkpoint_layout: str = DEFAULT_GATE_CHECKPOINT_LAYOUT
    """Attention output gate placement for gated attention. ``"separate"``
    (default) is the ERNIE-Lite layout with a standalone ``gate_proj`` tensor;
    ``"interleaved"`` is a gate packed head-interleaved inside ``q_proj`` that
    must be de-interleaved before fusing into ``qkv_proj``. Only consulted when
    the live attention is gated."""

    qkv_checkpoint_prefused: bool = DEFAULT_QKV_CHECKPOINT_PREFUSED
    """``False`` (default) is the usual checkpoint with distinct ``q_proj`` /
    ``k_proj`` / ``v_proj`` tensors; ``True`` is a single pre-fused ``qkv``
    tensor that must be split before the ``fused_qkv`` re-fusion."""

    moe_expert_checkpoint_layout: str = DEFAULT_MOE_EXPERT_CHECKPOINT_LAYOUT
    """``"per_expert"`` (default) for one projection tensor per expert,
    ``"packed"`` for two 3D tensors packing every expert. Only consulted on the
    grouped-GEMM branch, so one model may mix layouts across layers."""

    mlp_gate_up_fused: bool = DEFAULT_MLP_GATE_UP_FUSED
    """``True`` (default) is the gated MLP whose separate checkpoint
    ``gate_proj`` / ``up_proj`` fuse into the model ``up_gate_proj``; ``False``
    is a non-gated MLP handled by the generic Linear family."""

    mapping_proj_checkpoint_transposed: bool = (
        DEFAULT_MAPPING_PROJ_CHECKPOINT_TRANSPOSED
    )
    """Whether the checkpoint stores the mHC ``mapping_proj`` weight as
    ``(out, in)`` (the torch layout, needing a ``^T``) instead of the live
    ``(in, out)``. Orthogonal to the alpha layout; the accompanying leaf rename
    is expressed by ``checkpoint_name_mapping``, which cannot carry a
    transpose."""


def _protocol_value(config, attr, default):
    """Reads one model-declared ``aoa_*`` attribute with the single rule.

    An attribute that is absent or explicitly ``None`` falls back to the
    default; every other declared value is taken verbatim, including the falsy
    ones a real model relies on (an empty ``""`` checkpoint prefix, a ``False``
    ``mlp_gate_up_fused``). Keeping this the only reading path is what makes the
    eight attributes behave uniformly.
    """
    value = getattr(config, attr, None)
    return default if value is None else value


def build_aoa_context(model, config) -> FleetAOAContext:
    """Builds the read-only :class:`FleetAOAContext` for a whole-model pass.

    Ensures ``_set_pipeline_name_mapping`` has run (idempotent; the same
    side-effect ``state_dict`` / ``sharded_state_dict`` rely on), then resolves
    every model-declared ``aoa_*`` attribute through :func:`_protocol_value`
    against the module-level ``DEFAULT_*`` constants. The name mapping is the
    one field with extra handling: it is copied so the context never aliases the
    shared default, and validated as a name template.

    Args:
        model: The live ``GPTModel`` (or subclass) whose pipeline mapping backs
            the single-name resolution and whose ``_model_name_prefix``
            reports the authoritative model single-name root -- the same value
            ``get_layer_desc_list`` uses to name its pipeline layers, so pipeline
            naming and AOA name resolution never diverge.
        config: The sub-structure config threaded into the context (whole-model
            config for single-tower models, per-tower config for multi-tower).

    Returns:
        A frozen :class:`FleetAOAContext`.
    """
    if model._pipeline_name_mapping is None:
        model._set_pipeline_name_mapping()
    checkpoint_name_mapping = dict(
        _protocol_value(
            config,
            "aoa_checkpoint_name_mapping",
            DEFAULT_CHECKPOINT_NAME_MAPPING,
        )
    )
    validate_checkpoint_name_mapping(checkpoint_name_mapping)
    return FleetAOAContext(
        config=config,
        pp_to_single_mapping=model._pp_to_single_mapping or {},
        model_name_prefix=model._model_name_prefix(),
        checkpoint_name_mapping=checkpoint_name_mapping,
        checkpoint_name_prefix=_protocol_value(
            config,
            "aoa_checkpoint_name_prefix",
            DEFAULT_CHECKPOINT_NAME_PREFIX,
        ),
        dtype_cast_rules=_protocol_value(
            config, "aoa_dtype_cast_rules", DEFAULT_DTYPE_CAST_RULES
        ),
        gate_checkpoint_layout=_protocol_value(
            config,
            "aoa_gate_checkpoint_layout",
            DEFAULT_GATE_CHECKPOINT_LAYOUT,
        ),
        qkv_checkpoint_prefused=_protocol_value(
            config,
            "aoa_qkv_checkpoint_prefused",
            DEFAULT_QKV_CHECKPOINT_PREFUSED,
        ),
        moe_expert_checkpoint_layout=_protocol_value(
            config,
            "aoa_moe_expert_checkpoint_layout",
            DEFAULT_MOE_EXPERT_CHECKPOINT_LAYOUT,
        ),
        mlp_gate_up_fused=_protocol_value(
            config, "aoa_mlp_gate_up_fused", DEFAULT_MLP_GATE_UP_FUSED
        ),
        mapping_proj_checkpoint_transposed=_protocol_value(
            config,
            "aoa_mapping_proj_checkpoint_transposed",
            DEFAULT_MAPPING_PROJ_CHECKPOINT_TRANSPOSED,
        ),
    )
