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
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import paddle.distributed
from paddle.distributed.fleet.meta_parallel.parallel_layers.pp_layers import (
    PipelineLayerChunk,
)
from paddle.distributed.flex_checkpoint.aoa.generation import (
    AOAContext,
    AOANameScope,
    format_dtype_cast_attr,
    join_name,
    resolve_dtype_cast_rule,
    validate_checkpoint_name_mapping,
)
from paddle.distributed.flex_checkpoint.aoa.macros import (
    GLOBAL_ATTRIBUTE_KEYWORDS,
)

from paddlefleet.models.gpt.lm_head import (
    GPTLMHead,
    GPTMTPLMHead,
)
from paddlefleet.transformer.multi_token_prediction import (
    MultiTokenPredictionLayer,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Whole-world gather availability
# ---------------------------------------------------------------------------


def _world_can_gather() -> bool:
    """Whether a whole-world ``all_gather_object`` is actually available.

    ``get_world_size()`` alone is not a usable guard: with no process group
    initialized it falls back to ``PADDLE_TRAINERS_NUM``, so a
    launched-but-not-initialized process reports the launcher's rank count while
    the gather raises "The global group is not initialized." Both gather sites
    (:func:`_shared_layer_members` and :func:`_globalize_statements`) degrade to
    their rank-local answer when this returns ``False``.

    A world size >1 with no live group is logged rather than silently accepted:
    it is correct for a single-process run on a cluster node (the env vars are
    still set), but under real multi-rank it means AOA generation ran before
    distributed init, and staying rank-local then drops another stage's
    contribution.
    """
    if paddle.distributed.get_world_size() <= 1:
        return False
    if paddle.distributed.is_initialized():
        return True
    logger.warning(
        "AOA generation stayed rank-local: world_size=%d but no process group "
        "is initialized. Expected for a single-process run; under real "
        "multi-rank it means AOA ran before distributed init, and another PP "
        "stage's keys would be missing.",
        paddle.distributed.get_world_size(),
    )
    return False


def _assert_world_gather_available(config) -> None:
    """Fails loudly when a rank holds a model fragment but cannot gather.

    :func:`_world_can_gather` degrades to the rank-local answer, which is the
    correct degenerate result only when this rank's live module tree is the
    whole model. Under pipeline parallelism it never is: each stage owns
    different layers, so a rank-local config is missing another stage's keys,
    while the full-parameter save path requires every rank to pass the same
    config. Staying rank-local there is exactly the failure this globalization
    exists to prevent, so a declared pipeline with no group to gather over is
    an error rather than a degradation.

    Expert parallelism fragments the config as well, but only under the
    per-expert live layout -- a per-MoE-layer property rather than something
    this entry-level check can read off the config -- so it is not covered
    here.
    """
    if paddle.distributed.is_initialized():
        return
    if paddle.distributed.get_world_size() <= 1:
        return
    # Only a declared integer stage count establishes that the live module tree
    # is a fragment.
    stages = getattr(config, "pipeline_model_parallel_size", 1)
    if not isinstance(stages, int) or stages <= 1:
        return
    raise RuntimeError(
        f"AOA generation needs a whole-world gather: "
        f"pipeline_model_parallel_size={stages} means this rank's live module "
        f"tree is one stage of the model, but no process group is "
        f"initialized, so the generated config would stay rank-local and miss "
        f"the other stages' keys."
    )


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


# ---------------------------------------------------------------------------
# Tied / shared alias planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AliasCandidate:
    """One shared-layer alias candidate collected from the live model."""

    layer_name: str | None
    owned: list[tuple[str, str]]
    distinct: list[str]
    checkpoint_of: dict[str, str]
    collapses: bool


@dataclass(frozen=True)
class AliasGroup:
    """A set of model single-names backed by one logically shared tensor.

    Produced by tied weights -- a pp==1 embedding / lm_head sharing one tensor
    object, or a ``SharedLayerDesc`` group whose halves live on different PP
    stages. ``canonical_single`` is the producer kept in normal recursion;
    ``alias_singles`` are fanned out from ``canonical_checkpoint`` in the
    checkpoint->model direction and deleted (``-> _``) in the model->checkpoint
    direction.

    ``canonical_checkpoint`` is the already-resolved checkpoint name, not a
    structured name: for a group split across PP stages the canonical member is
    owned by another rank, and re-resolving its structured name here would run it
    through THIS rank's ``_pp_to_single_mapping`` and silently yield this rank's
    own single. Resolution therefore happens on the owning rank and only the
    finished name travels.

    ``emit_canonical`` is used when the canonical member is excluded from normal
    recursion because its checkpoint route differs from its model name. Ordinary
    alias groups leave it false because recursion already emits the canonical
    statement.
    """

    canonical_single: str
    alias_singles: tuple[str, ...]
    canonical_checkpoint: str
    emit_canonical: bool = False


_SHARED_LAYERS_ROOT = "shared_layers"

# Single-names whose role marks them as the ALIAS side of a tie, never the
# canonical producer. pp>1 registers the head under the shared prefixes
# ``shared_head`` / ``shared_mtp_lm_head`` (``get_layer_desc_list``) rather than
# ``lm_head``, and the magic-send re-embedding under ``mtp_embedding``, so all of
# them belong here -- choosing the right canonical must not depend on ``embed``
# happening to sort before ``shared`` / ``mtp``.
_ALIAS_SEGMENTS = frozenset(
    {
        "lm_head",
        "output_layer",
        "shared_head",
        "shared_mtp_lm_head",
        "mtp_embedding",
    }
)


def _grouped_shared_tensors(model):
    """Groups structured pipeline names by shared tensor identity.

    Uses the raw pipeline ``state_dict`` (with duplicate keys for shared
    tensors), so keys match ``_pp_to_single_mapping`` and object identity
    exposes VPP / shared duplicates and tied aliases. Only tensors registered
    under more than one structured name are returned.

    Returns:
        A list of ``(structured_names, single_names)`` pairs, positionally
        aligned, one per shared tensor.
    """
    if model._pipeline_name_mapping is None:
        model._set_pipeline_name_mapping()
    mapping = model._pp_to_single_mapping or {}
    raw = model._raw_structured_state_dict()
    id_to_structs: dict[int, list[str]] = {}
    for structured, tensor in raw.items():
        id_to_structs.setdefault(id(tensor), []).append(structured)
    groups = []
    for structs in id_to_structs.values():
        if len(structs) <= 1:
            continue
        singles = [mapping.get(s, s) for s in structs]
        groups.append((structs, singles))
    return groups


def _select_canonical(single_names: Iterable[str]) -> str:
    """Picks the canonical producer among tied single-names.

    Heuristic honoring the tie rule (the embedding is the canonical producer; the
    head / re-embedding is the alias): prefer a name carrying no
    ``_ALIAS_SEGMENTS`` segment, breaking ties lexicographically. Per-model
    migration may refine this.
    """
    ordered = sorted(dict.fromkeys(single_names))
    for name in ordered:
        segments = set(name.lower().split("."))
        if not segments & _ALIAS_SEGMENTS:
            return name
    return ordered[0]


def _shared_layer_group_key(structs) -> str | None:
    """Returns the ``SharedLayerDesc.layer_name`` a shared-tensor group belongs to.

    ``None`` for a group with no ``shared_layers.*`` member -- a purely
    intra-process share, i.e. the pp==1 tie (``skip_weight_param_allocation``
    makes the head reuse the embedding tensor outright) or VPP re-registration.

    A group may MIX a ``shared_layers.*`` registration with a plain path when the
    same tensor object is reachable both ways: with ``tie_word_embeddings=True``,
    ``pipeline_model_parallel_size == 1`` and ``spec.mtp_lm_head`` set, the
    embedding stays on a plain ``LayerDesc`` (``get_layer_desc_list`` gates its
    tie branch on pp>1) while the head becomes ``SharedLayerDesc("embed")``, and
    ``skip_weight_param_allocation`` ties them by object. Such a group must be
    planned as ONE unit under the shared layer_name -- otherwise the plain name
    escapes exclusion and two producers write the same checkpoint key.
    """
    layer_names = {
        s.split(".")[1]
        for s in structs
        if s.split(".")[0] == _SHARED_LAYERS_ROOT
    }
    if not layer_names:
        return None
    if len(layer_names) > 1:
        raise ValueError(
            f"one tensor is registered under several SharedLayerDesc names "
            f"{sorted(layer_names)}; their checkpoint-layout policies may "
            f"disagree, so the group cannot be planned as one unit"
        )
    return next(iter(layer_names))


def _shared_layer_members(model, ctx):
    """Resolves this rank's ``shared_layers.*`` members, then all-gathers them.

    Object identity cannot group these. Under PP every stage that declares a
    ``SharedLayerDesc`` builds its OWN instance (``pp_layers.py`` walks only
    ``_layers_desc[start:end]``, then ``self.shared_layers[layer_name] =
    layer.build_layer()``), kept in step by ``_synchronize_shared_weights`` and
    ``allreduce_shared_weight_gradients``. The two halves of a tie are different
    objects in different processes, so no ``id()`` relates them.
    ``SharedLayerDesc.layer_name`` is the key Paddle itself shares on and is
    globally identical, so it is the grouping key instead.

    Names must be resolved by their OWNING rank, which is why the gather carries
    resolved names rather than structured ones -- see :class:`AliasGroup`. An
    alias member's checkpoint name is resolved too but unused; only the
    canonical's is read.

    Returns:
        ``(owned, merged)``. ``owned`` maps layer_name -> this rank's
        ``(structured_name, single_name)`` pairs, which is all that steering this
        rank's own recursion needs. ``merged`` maps layer_name -> whole-world
        ``(single_name, checkpoint_name)`` pairs.
    """
    # Identity groups first, so a plain path aliased onto a shared_layers
    # registration is attributed to the same layer_name (the mixed group above).
    planned_under: dict[str, str] = {}
    for structs, _singles in _grouped_shared_tensors(model):
        layer_name = _shared_layer_group_key(structs)
        if layer_name is None:
            continue
        for structured in structs:
            planned_under[structured] = layer_name
    # Then the shared_layers registrations that are alone in their identity group
    # -- the ordinary PP case, where the peer half lives in another process.
    for structured in model._raw_structured_state_dict():
        segments = structured.split(".")
        if segments[0] == _SHARED_LAYERS_ROOT:
            planned_under.setdefault(structured, segments[1])

    owned: dict[str, list[tuple[str, str]]] = {}
    resolved: dict[str, list[tuple[str, str]]] = {}
    for structured, layer_name in planned_under.items():
        checkpoint_name, single_name = model._resolve_leaf_names(
            structured, ctx
        )
        owned.setdefault(layer_name, []).append((structured, single_name))
        resolved.setdefault(layer_name, []).append(
            (single_name, checkpoint_name)
        )

    # The gather needs a live process group, which `get_world_size()` does NOT
    # imply -- see `_world_can_gather`. Rank-local members are the correct
    # degenerate answer whenever there is no group to gather over.
    if not _world_can_gather():
        return owned, resolved
    gathered: list[dict[str, list[tuple[str, str]]]] = []
    paddle.distributed.all_gather_object(gathered, resolved)
    merged: dict[str, list[tuple[str, str]]] = {}
    for rank_members in gathered:
        for layer_name, members in rank_members.items():
            bucket = merged.setdefault(layer_name, [])
            for member in members:
                if member not in bucket:
                    bucket.append(member)
    return owned, merged


def _iter_alias_candidates(model, ctx):
    """Yields an :class:`AliasCandidate` per shared group.

    Two sources, normalized to one shape:

    - ``shared_layers.*`` groups, keyed on ``SharedLayerDesc.layer_name`` and
      merged whole-world, so a PP-split tie is visible on every rank. Policy
      comes from the model's declared checkpoint layout.
    - Intra-process object-identity groups with no ``shared_layers.*`` member:
      the pp==1 tie and pure VPP re-registration. Identity is the right
      judgement here because the duplication is intra-process by construction,
      and such a group always backs a single checkpoint tensor, hence
      ``collapses=True``.
    """
    owned_by_layer, merged_by_layer = _shared_layer_members(model, ctx)
    for layer_name, members in merged_by_layer.items():
        owned = owned_by_layer.get(layer_name, [])
        distinct = list(
            dict.fromkeys(single for single, _checkpoint in members)
        )
        collapses = model._aoa_shared_layer_collapses(layer_name)
        yield AliasCandidate(
            layer_name=layer_name,
            owned=owned,
            distinct=distinct,
            checkpoint_of=dict(members),
            collapses=collapses,
        )
    for structs, singles in _grouped_shared_tensors(model):
        if _shared_layer_group_key(structs) is not None:
            continue
        yield AliasCandidate(
            layer_name=None,
            owned=list(zip(structs, singles)),
            distinct=list(dict.fromkeys(singles)),
            checkpoint_of={
                single: model._resolve_leaf_names(structured, ctx)[0]
                for structured, single in zip(structs, singles)
            },
            collapses=True,
        )


def collect_alias_plan(model, ctx) -> tuple[frozenset[str], list[AliasGroup]]:
    """Resolves, in one pass, what recursion must skip and what to re-emit.

    Both halves derive from the same grouping and the same canonical selection,
    so they are produced together:

    - ``excluded``: the structured pipeline names recursion must skip. Stays
      RANK-LOCAL -- it only steers this rank's recursion, and each rank knows its
      own structured names. Two sources: pure VPP / shared dedup (a tensor under
      several structured names all mapping to the *same* single-name: keep the
      first, skip the rest), and tied aliases (keep one structured name for the
      canonical single, skip every alias member and any duplicate canonical
      registration).
    - ``alias_groups``: the tied groups whose members are re-emitted by
      :func:`emit_alias_aoa` / :func:`emit_alias_inv_aoa`. GLOBAL -- every rank
      emits a group's alias statements even when it owns no member, and
      :func:`_globalize_statements` dedups the identical strings. That is not
      merely harmless but required, since the load path validates against the
      whole-world union of model keys. Groups with one distinct single-name carry
      no fan-out and are handled purely by exclusion, so they produce no group.

    Returns:
        ``(excluded, alias_groups)``.
    """
    excluded: set[str] = set()
    alias_groups: list[AliasGroup] = []
    for candidate in _iter_alias_candidates(model, ctx):
        owned = candidate.owned
        distinct = candidate.distinct
        checkpoint_of = candidate.checkpoint_of
        if not candidate.collapses:
            # A non-collapsing group needs one producer per single name. For
            # MTP reuse, prefer the registration that carries transformer scope
            # when duplicate registrations resolve to the same single name.
            by_single: dict[str, list[str]] = {}
            for structured, single in owned:
                by_single.setdefault(single, []).append(structured)
            for structured_names in by_single.values():
                if candidate.layer_name == "mtp_reuse_transformer":
                    structured_names.sort(
                        key=lambda name: (
                            ".transformer_layer." not in name,
                            name.startswith("shared_layers."),
                            name,
                        )
                    )
                excluded.update(structured_names[1:])
            continue

        if candidate.layer_name == "embed" and getattr(
            ctx.config, "separate_mtp_headloss", False
        ):
            # The runtime embed group contains the embedding and two head
            # registrations, but the HF layout has one checkpoint route per
            # head attribute. Keep embedding recursion intact and materialize
            # the MTP head routes explicitly below.
            head_routes = (
                ("weight", "lm_head.weight"),
                ("multimax_ranges", "model.lm_head.multimax_ranges"),
                ("multimax_ts", "model.lm_head.multimax_ts"),
            )
            for suffix, checkpoint_name in head_routes:
                canonical = join_name(
                    ctx.model_name_prefix, f"shared_mtp_lm_head.{suffix}"
                )
                alias = join_name(
                    ctx.model_name_prefix, f"shared_head.{suffix}"
                )
                route_owned = [
                    (structured, single)
                    for structured, single in owned
                    if single in (canonical, alias)
                ]
                if not route_owned:
                    continue
                excluded.update(
                    structured for structured, _single in route_owned
                )
                if canonical not in distinct:
                    continue
                alias_groups.append(
                    AliasGroup(
                        canonical,
                        (alias,) if alias in distinct else (),
                        checkpoint_name,
                        emit_canonical=True,
                    )
                )
            continue

        if len(distinct) <= 1:
            excluded.update(structured for structured, _single in owned[1:])
            continue
        canonical = _select_canonical(distinct)
        canonical_kept = False
        for structured, single in owned:
            if single == canonical and not canonical_kept:
                canonical_kept = True
            else:
                excluded.add(structured)
        alias_groups.append(
            AliasGroup(
                canonical,
                tuple(single for single in distinct if single != canonical),
                checkpoint_of[canonical],
            )
        )
    return frozenset(excluded), alias_groups


def emit_alias_aoa(ctx, alias_groups) -> list[str]:
    """Checkpoint->model alias fan-out.

    The canonical single is already produced by the normal recursion; here the
    canonical checkpoint name additionally fans out to every alias model key.
    Each alias single resolves its own dtype-cast rule (keyed on the single
    name), so a fanned-out tensor lands in the alias's declared dtype even when
    it differs from the canonical single.

    ``canonical_checkpoint`` is taken as-is instead of being resolved here: for a
    group split across PP stages the canonical member is owned by another rank, so
    resolution must happen there -- see :class:`AliasGroup`.
    """
    statements = []
    for group in alias_groups:
        targets = group.alias_singles
        if group.emit_canonical:
            targets = (group.canonical_single, *targets)
        for target in targets:
            cast = format_dtype_cast_attr(
                resolve_dtype_cast_rule(
                    target, ctx.dtype_cast_rules, ctx.model_name_prefix
                )
            )
            statements.append(f"{group.canonical_checkpoint} -> {target}{cast}")
    return statements


def emit_alias_inv_aoa(alias_groups) -> list[str]:
    """Inverse (model -> checkpoint) alias handling.

    The canonical single -> checkpoint statement comes from the normal
    recursion; each alias model key is deleted (``-> _``) so only the canonical
    producer writes the checkpoint tensor.
    """
    statements = []
    for group in alias_groups:
        if group.emit_canonical:
            statements.append(
                f"{group.canonical_single} -> {group.canonical_checkpoint}"
            )
        for alias_single in group.alias_singles:
            statements.append(f"{alias_single} -> _")
    return statements


# ---------------------------------------------------------------------------
# Single-tower guard and model-side name resolution
# ---------------------------------------------------------------------------


def _assert_single_tower(model):
    """Fails loudly if a boundary declaring ``aoa_towers`` reaches this path.

    Multi-tower containers (e.g. Qwen3-VL) are not ``GPTModel`` subclasses; they
    own their own entry and loop over their towers. A ``GPTModel`` that declares
    ``aoa_towers`` is therefore a misconfiguration, not a supported case.
    """
    if getattr(model, "aoa_towers", None):
        raise NotImplementedError(
            f"{type(model).__name__} declares aoa_towers but the whole-model "
            f"generator only handles single-tower boundaries; a multi-tower "
            f"container must own its AOA entry"
        )


def resolve_actual_model_prefix(structured_prefix, pp_to_single_mapping) -> str:
    """Resolves a live structured subtree prefix to its single-name prefix.

    Live-tree source of truth: rather than assume the pipeline key, this finds
    any real leaf registered under ``structured_prefix`` in the authoritative
    ``pp_to_single_mapping`` and strips the shared suffix off its single name.
    ``_set_pipeline_name_mapping`` only remaps the root/index segment and
    preserves the suffix exactly, so the recovered root is identical regardless
    of which leaf under the prefix is picked.

    Args:
        structured_prefix: Live module path ending in ``.`` (e.g. ``"16."`` or
            the VPP two-segment ``"0.1."``).
        pp_to_single_mapping: Structured name -> single name mapping.

    Returns:
        The single-name subtree root (e.g. ``"model.layers.16"``).

    Raises:
        KeyError: If no mapping entry sits under ``structured_prefix``.
        ValueError: If the single name does not end with the shared suffix.
    """
    for structured_name, single_name in pp_to_single_mapping.items():
        if structured_name.startswith(structured_prefix):
            suffix = structured_name[len(structured_prefix) :]
            if not suffix:
                return single_name
            if single_name == suffix:
                return ""
            if single_name.endswith("." + suffix):
                return single_name[: -len(suffix) - 1]
            raise ValueError(
                f"single name {single_name!r} does not end with suffix "
                f"{suffix!r} for structured prefix {structured_prefix!r}"
            )
    raise KeyError(
        f"no pp_to_single_mapping entry under structured prefix "
        f"{structured_prefix!r}"
    )


def _parse_layers_index(root: str) -> int:
    """Extracts the integer following a ``layers`` segment in a single root.

    Args:
        root: Single-name subtree root (e.g. ``"model.layers.16"`` or
            ``"model.layers.16.transformer_layer"``).

    Returns:
        The int after the ``layers`` segment (the model layer id).

    Raises:
        ValueError: If no ``layers.<int>`` segment is present.
    """
    parts = root.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    raise ValueError(f"no 'layers.<int>' segment in root {root!r}")


def build_layer_name_scope(
    ctx,
    *,
    layer_id: int,
    checkpoint_prefix_template: str,
    structured_prefix: str,
    absolute: bool = False,
) -> AOANameScope:
    """Builds the per-layer :class:`AOANameScope`.

    Prefix injection, not template arithmetic: the ``layer_id`` (and the
    derived MTP-internal ``mtp_id``) are rendered into the checkpoint prefix
    template up front via ``str.replace``. ``$LAYER_ID`` is the transformer
    layer number; ``$MTP_LAYER_ID`` is the MTP-internal id from 0. Any residual
    ``$`` means the template referenced a placeholder we do not inject, which
    is a config error.

    Args:
        ctx: The read-only recursion context.
        layer_id: Transformer layer number, as it appears in the model's own
            ``layers.<i>`` naming.
        checkpoint_prefix_template: Prefix template, possibly with ``$LAYER_ID``
            / ``$MTP_LAYER_ID``.
        structured_prefix: Live module path for this subtree, ending in ``.``.
        absolute: When True, the resolved prefix is a full checkpoint prefix and
            the shared checkpoint prefix is not prepended.

    Returns:
        The frozen scope for this subtree.
    """
    mtp_id = layer_id - ctx.config.num_hidden_layers
    checkpoint_prefix = checkpoint_prefix_template.replace(
        "$LAYER_ID", str(layer_id)
    ).replace("$MTP_LAYER_ID", str(mtp_id))
    if "$" in checkpoint_prefix:
        raise ValueError(
            f"unresolved placeholder in checkpoint prefix "
            f"{checkpoint_prefix!r} (template={checkpoint_prefix_template!r}, "
            f"layer_id={layer_id}, mtp_id={mtp_id})"
        )
    return AOANameScope(
        checkpoint_prefix=checkpoint_prefix,
        logical_model_prefix=join_name(
            ctx.model_name_prefix, f"layers.{layer_id}"
        ),
        actual_model_prefix=resolve_actual_model_prefix(
            structured_prefix, ctx.pp_to_single_mapping
        ),
        is_checkpoint_prefix_absolute=absolute,
    )


def _resolve_lm_head_scope(ctx, structured_prefix: str) -> AOANameScope:
    """Builds the scope that emits the output head unprefixed at the ckpt root.

    The HF ``ForCausalLM`` layout keeps the output ``lm_head`` a top-level
    sibling of the backbone, so its checkpoint name carries no shared prefix
    (bare ``lm_head.weight``). The shared checkpoint prefix is always prepended
    by ``join_name`` and cannot be stripped by the mapping, so an
    absolute-prefix scope is the only way to emit it. The checkpoint root
    mirrors the head's model-relative root; no mapping leaf touches the head, so
    the logical and actual model roots coincide and mapping resolution is
    identity. Non-tied head only -- a tied head is an embedding alias handled by
    the alias plan and is never dispatched here.
    """
    actual_prefix = resolve_actual_model_prefix(
        structured_prefix, ctx.pp_to_single_mapping
    )
    prefix = ctx.model_name_prefix
    if actual_prefix == prefix:
        checkpoint_prefix = ""
    elif prefix and actual_prefix.startswith(prefix + "."):
        checkpoint_prefix = actual_prefix[len(prefix) + 1 :]
    else:
        checkpoint_prefix = actual_prefix
    return AOANameScope(
        checkpoint_prefix=checkpoint_prefix,
        logical_model_prefix=actual_prefix,
        actual_model_prefix=actual_prefix,
        is_checkpoint_prefix_absolute=True,
    )


def _iter_pipeline_units(model) -> Iterator[tuple[str, object]]:
    """Yields ``(structured_prefix, unit)`` for every live top-level layer.

    Live enumeration (not ``range()``): walks ``model._sub_layers`` in
    registration order and descends one level into VPP
    :class:`PipelineLayerChunk` containers so a chunk's inner layers surface
    with their real two-segment structured prefix (``f"{chunk}.{local}."``).
    Non-chunk top-level entries (embedding, transformer layers, MTP, lm_head,
    norm, shared_layers) surface directly.

    This mirrors the ``fp8_quant_weight`` live-layer idiom's coverage and, like
    it, does not special-case cudagraph / PipelineSublayers wrapping.
    """
    for name, unit in model._sub_layers.items():
        if unit is None:
            continue
        if isinstance(unit, PipelineLayerChunk):
            for local_name, sub in unit._sub_layers.items():
                if sub is not None:
                    yield f"{name}.{local_name}.", sub
        else:
            yield f"{name}.", unit


def _resolve_mtp_scopes(
    ctx, structured_prefix: str, unit, mtp_spec
) -> tuple[AOANameScope, AOANameScope, str]:
    """Resolves the MTP-own and inner-transformer scopes for one MTP layer.

    Returns ``(mtp_scope, transformer_scope, transformer_structured_prefix)``.
    The ``layer_id`` comes from the authoritative single root; the live
    ``layer_number`` (the MTP-internal id) is cross-checked against the derived
    ``mtp_id``. ``mtp_spec`` is the per-pass
    :class:`MTPCheckpointPrefixSpec` carrying the model-declared checkpoint
    prefixes; it is passed in directly rather than read off ``ctx`` because no
    component override consumes it.

    NOTE (unverified assumption): a boundary declaring
    ``aoa_mtp_checkpoint_prefix_absolute`` (currently only assumed for Qwen3.5's
    ``mtp`` prefix) routes BOTH the MTP-own and the inner-transformer checkpoint
    names at the checkpoint model root, skipping the shared prefix. Qwen3.5's
    ``mtp.*`` naming has not been verified against a real checkpoint; revisit
    when confirmed.
    """
    actual_prefix = resolve_actual_model_prefix(
        structured_prefix, ctx.pp_to_single_mapping
    )
    layer_id = _parse_layers_index(actual_prefix)
    mtp_id = layer_id - ctx.config.num_hidden_layers
    assert unit.layer_number == mtp_id, (
        f"MultiTokenPredictionLayer.layer_number={unit.layer_number} "
        f"disagrees with mtp_id={mtp_id} (layer_id={layer_id}, "
        f"structured_prefix={structured_prefix!r})"
    )
    absolute = mtp_spec.is_absolute
    mtp_scope = build_layer_name_scope(
        ctx,
        layer_id=layer_id,
        checkpoint_prefix_template=mtp_spec.checkpoint_prefix,
        structured_prefix=structured_prefix,
        absolute=absolute,
    )
    transformer_structured_prefix = f"{structured_prefix}transformer_layer."
    transformer_scope = build_layer_name_scope(
        ctx,
        layer_id=layer_id,
        checkpoint_prefix_template=mtp_spec.transformer_checkpoint_prefix,
        structured_prefix=transformer_structured_prefix,
        absolute=absolute,
    )
    return mtp_scope, transformer_scope, transformer_structured_prefix


# ---------------------------------------------------------------------------
# AOA statement text: parsing and invariants
# ---------------------------------------------------------------------------


def _split_top_level(text: str) -> list[str]:
    """Splits on top-level commas, respecting ``()``/``[]``/``{}`` nesting."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return parts


def _statement_targets(statement: str) -> list[str]:
    """Extracts the target names on the RHS of one AOA statement.

    The RHS is everything after ``->``; targets are the comma-separated tokens
    that are not attributes. Attributes are either ``key=value`` forms
    (``axis=``, ``permute=``, ``dtype=``, ...) or bareword fusion macros
    (``fused_ffn``, ``fused_qkv``, ...); both are identified by their keyword
    living in ``GLOBAL_ATTRIBUTE_KEYWORDS`` -- the lexer's single source of
    truth for what is an attribute rather than a tensor name. Bracketed
    attribute values (e.g. ``permute=[1,0]``) are kept intact by top-level
    splitting.
    """
    _, sep, rhs = statement.partition("->")
    if not sep:
        return []
    attr_keywords = set(GLOBAL_ATTRIBUTE_KEYWORDS)
    targets = []
    for seg in _split_top_level(rhs):
        seg = seg.strip()
        if not seg:
            continue
        keyword = seg.split("=", 1)[0].strip()
        if keyword in attr_keywords:
            continue
        targets.append(seg)
    return targets


def _statement_sources(statement: str) -> list[str]:
    """Extracts the source names on the left side of one AOA statement.

    Mirrors :func:`_statement_targets` for the left side: the source side is
    everything before ``->``; sources are the comma-separated tokens with any
    trailing ``^T`` transpose marker stripped, excluding attribute keywords.
    Used by :func:`_assert_unique_inverse_targets` to recognize a statement that
    rewrites a name it also reads, so an in-place rewrite is not mistaken for a
    second producer.
    """
    source_side, sep, _ = statement.partition("->")
    if not sep:
        return []
    attr_keywords = set(GLOBAL_ATTRIBUTE_KEYWORDS)
    sources = []
    for seg in _split_top_level(source_side):
        seg = seg.strip()
        if seg.endswith("^T"):
            seg = seg[:-2].strip()
        if not seg:
            continue
        keyword = seg.split("=", 1)[0].strip()
        if keyword in attr_keywords:
            continue
        sources.append(seg)
    return sources


def _assert_unique_inverse_targets(statements, origins=None) -> None:
    """Fails loudly on a duplicate inverse (model -> checkpoint) target.

    A many-to-one model -> checkpoint mapping (e.g. Qwen3.5's per-MTP-layer
    ``enorm`` copies all naming the single ``mtp.enorm.weight``) is not
    expressible: AOAEngine's inverse export has no target validation and would
    silently overwrite. This is an assert-only guard, never a text dedup: the
    ``-> _`` deletions the alias handler emits are skipped.

    A statement that reads back the name it writes is a rewrite in place, not a
    second producer, and takes over that name's ownership. The fused gate/up
    inverse split writes the model-side halves under their gate/up names
    (``up_gate_proj -> gate_proj, up_proj, fused_ffn``) and a following
    statement transposes each in place onto the checkpoint name
    (``gate_proj^T -> gate_proj``); in the Ernie layout the half-name and the
    checkpoint name render to the same string, so without this the second
    statement would look like a duplicate. A genuine many-to-one collision
    writes a name it does not itself read, so it stays caught even when some
    unrelated statement elsewhere happens to read that same name.

    ``origins``, when given, is a list parallel to ``statements`` naming the
    tower that emitted each one, so a cross-tower collision can say which towers
    disagree.
    """
    claimed_by: dict[str, int] = {}
    for index, statement in enumerate(statements):
        sources = frozenset(_statement_sources(statement))
        for target in _statement_targets(statement):
            if target == "_":
                continue
            previous = claimed_by.get(target)
            if previous is not None and target not in sources:
                origin_note = _duplicate_origin_note(origins, previous, index)
                raise ValueError(
                    f"inverse AOA emits duplicate checkpoint target "
                    f"{target!r}{origin_note}"
                    f"; a model->checkpoint many-to-one mapping "
                    f"cannot be expressed and would silently overwrite "
                    f"(statement: {statement!r})"
                )
            claimed_by[target] = index


def _duplicate_origin_note(origins, first: int, second: int) -> str:
    """Names the tower(s) behind a duplicate target, when origins are known.

    Empty for a single-tower caller, which passes no origins: the flat statement
    list is its own context there, so there is nothing to attribute.
    """
    if origins is None:
        return ""
    if origins[first] == origins[second]:
        return f" (both emitted by tower {origins[first]!r})"
    return f" (emitted by towers {origins[first]!r} and {origins[second]!r})"


# ---------------------------------------------------------------------------
# Whole-model entries and statement globalization
# ---------------------------------------------------------------------------


def gen_whole_model_aoa(model, ctx, *, globalize=True) -> dict[str, list[str]]:
    """Whole-model checkpoint->model generation.

    Resolves the alias plan, pins the excluded names into ``ctx`` so the whole
    recursion honors them, then hands each child to the standard
    ``Layer.gen_aoa_statements`` protocol -- that polymorphic call is what lets
    component overrides (Linear, SelfAttention, MoE, ...) emit their own rules,
    so this function must never re-implement the walk itself.

    ``globalize=False`` returns the rank-local set, for a caller that unions
    several of these and globalizes the result itself
    (:func:`gen_multi_tower_aoa`). A rank-local set is not a usable config on
    its own -- see :func:`_globalize_tagged_statements`.

    Returns a ``{"aoa_statements": list[str]}`` dict; the list is mutable to
    match the consumer contract.
    """
    _assert_world_gather_available(ctx.config)
    _assert_single_tower(model)
    excluded, alias_groups = collect_alias_plan(model, ctx)
    ctx = replace(ctx, excluded_names=excluded)
    mtp_spec = resolve_mtp_checkpoint_prefix_spec(ctx.config)
    statements = []
    # The boundary emits no own parameters, so recursion starts one level down.
    # Single-enumeration classify-and-dispatch: each live unit is
    # isinstance-classified once and dispatched exactly once. Two unit kinds
    # need a scope: MTP, whose checkpoint prefix is model-declared and whose
    # inner transformer sits one level deeper than the checkpoint layout; and
    # the output head, whose checkpoint name sits unprefixed at the model root
    # (HF ``ForCausalLM`` keeps ``lm_head`` a top-level sibling). The output
    # head is matched on the base ``GPTLMHead`` so the non-separate variant
    # (``separate_mtp_headloss=False``, a plain ``GPTLMHead`` instance) is
    # covered as well as the ``GPTMainLMHead`` subclass; the MTP head
    # ``GPTMTPLMHead`` is a ``GPTLMHead`` subclass too but is explicitly
    # excluded so it keeps flowing through the plain scope-less else path.
    # Every other unit (including normal transformer layers) resolves correctly
    # on the plain scope-less path.
    for structured_prefix, unit in _iter_pipeline_units(model):
        if isinstance(unit, MultiTokenPredictionLayer):
            mtp_scope, transformer_scope, transformer_prefix = (
                _resolve_mtp_scopes(ctx, structured_prefix, unit, mtp_spec)
            )
            # Checkpoint->model order: MTP-own params first, then the inner
            # transformer. GPTModel drives the inner transformer here because
            # the MTP layer's own recursion deliberately does not.
            statements += unit.gen_aoa_statements(
                ctx,
                structured_name_prefix=structured_prefix,
                aoa_name_scope=mtp_scope,
            )
            statements += unit.transformer_layer.gen_aoa_statements(
                ctx,
                structured_name_prefix=transformer_prefix,
                aoa_name_scope=transformer_scope,
            )
        elif isinstance(unit, GPTLMHead) and not isinstance(unit, GPTMTPLMHead):
            statements += unit.gen_aoa_statements(
                ctx,
                structured_name_prefix=structured_prefix,
                aoa_name_scope=_resolve_lm_head_scope(ctx, structured_prefix),
            )
        else:
            statements += unit.gen_aoa_statements(
                ctx, structured_name_prefix=structured_prefix
            )
    statements += emit_alias_aoa(ctx, alias_groups)
    if globalize:
        statements = _globalize_statements(statements)
    return {"aoa_statements": statements}


def _globalize_statements(local_statements: list[str]) -> list[str]:
    """Untagged form of :func:`_globalize_tagged_statements`.

    Used by every single-tower caller, which has no tower to attribute a
    statement to.
    """
    tagged = _globalize_tagged_statements(
        [(None, statement) for statement in local_statements]
    )
    return [statement for _origin, statement in tagged]


def _globalize_tagged_statements(local_tagged):
    """All-gathers per-PP/EP-rank-local AOA statements and dedups them.

    Applies to both passes: modular generation walks the LOCAL live module tree
    (:func:`_iter_pipeline_units`), so either statement set varies across PP
    (each stage owns different layers, so it always varies) and across EP under
    the per-expert live layout -- the MoE layer stores non-local experts as
    ``None`` (``moe_layer.py`` build) and generation skips them, so each EP rank
    emits only its local experts.

    Both consumers require a globally complete set, for mirrored reasons:

    - Inverse (model -> checkpoint): DCP ``save_full_param`` (``num_splits=1``)
      requires every rank's config to be globally identical so that each rank's
      ``destination_sharded_weight_desc`` contains all targets that appear in
      the globally all-gathered read plan.
    - Forward (checkpoint -> model): ``dist.load_state_dict`` builds its
      ``destination_state_shard_info`` with ``build_global_state_shard_info``,
      a whole-world ``all_gather_object``, so AOAEngine validates against the
      union of every rank's model keys. Its ``shape_propagation`` destination
      sweep demands a source for each one, and only same-name pass-through is
      implicit; a rank-local set therefore fails on the first key that needs
      renaming and is owned by another PP stage.

    TP / DP / sharding / grouped-EP replicas emit byte-identical statements, so
    an exact-string, order-preserving dedup collapses the whole-world gather
    back to the complete config without loss. Chained statements (an
    intermediate name emitted then consumed) stay contiguous because a chain
    never straddles ranks, and rank-ordered gathering plus order-preserving
    dedup leaves every rank with the same list in the same order -- AOA
    statements are order-sensitive, so that determinism matters. The gather
    spans the whole world (a superset of the assembler's h*v mesh) so the result
    is layout-agnostic and does not depend on which expert layout each MoE layer
    happens to use.

    Each element is an ``(origin, statement)`` pair. Dedup keys on the statement
    alone, so the first origin claiming a statement wins: replicas across ranks
    carry the same origin anyway, and collapsing an identical statement is the
    whole point of the dedup.
    """
    if not _world_can_gather():
        return local_tagged
    gathered: list[list[tuple[str | None, str]]] = []
    paddle.distributed.all_gather_object(gathered, local_tagged)
    seen: set[str] = set()
    merged: list[tuple[str | None, str]] = []
    for rank_tagged in gathered:
        for origin, statement in rank_tagged:
            if statement not in seen:
                seen.add(statement)
                merged.append((origin, statement))
    return merged


def gen_whole_model_inv_aoa(
    model, ctx, *, globalize=True
) -> dict[str, list[str]]:
    """Whole-model inverse (model -> checkpoint) generation.

    Independently recurses and emits through ``Layer.gen_inv_aoa_statements``;
    never derived from the checkpoint->model pass. Returns a
    ``{"aoa_statements": list[str]}`` dict.

    ``globalize=False`` returns the rank-local set for a caller that unions
    several of these (:func:`gen_multi_tower_inv_aoa`). The duplicate-target
    guard still runs on the rank-local set so a within-tower collision fails at
    its source; the union is re-checked by that caller.
    """
    _assert_world_gather_available(ctx.config)
    _assert_single_tower(model)
    excluded, alias_groups = collect_alias_plan(model, ctx)
    ctx = replace(ctx, excluded_names=excluded)
    mtp_spec = resolve_mtp_checkpoint_prefix_spec(ctx.config)
    statements = []
    # Independently recurses and emits, never derived from the checkpoint->model
    # pass. Units are walked in reverse to mirror the checkpoint->model
    # ordering; for statements this is cosmetic but faithful to the design.
    for structured_prefix, unit in reversed(list(_iter_pipeline_units(model))):
        if isinstance(unit, MultiTokenPredictionLayer):
            mtp_scope, transformer_scope, transformer_prefix = (
                _resolve_mtp_scopes(ctx, structured_prefix, unit, mtp_spec)
            )
            # Inverse order: inner transformer first, then MTP-own -- the
            # mirror of the checkpoint->model order.
            statements += unit.transformer_layer.gen_inv_aoa_statements(
                ctx,
                structured_name_prefix=transformer_prefix,
                aoa_name_scope=transformer_scope,
            )
            statements += unit.gen_inv_aoa_statements(
                ctx,
                structured_name_prefix=structured_prefix,
                aoa_name_scope=mtp_scope,
            )
        elif isinstance(unit, GPTLMHead) and not isinstance(unit, GPTMTPLMHead):
            statements += unit.gen_inv_aoa_statements(
                ctx,
                structured_name_prefix=structured_prefix,
                aoa_name_scope=_resolve_lm_head_scope(ctx, structured_prefix),
            )
        else:
            statements += unit.gen_inv_aoa_statements(
                ctx, structured_name_prefix=structured_prefix
            )
    statements += emit_alias_inv_aoa(alias_groups)
    if globalize:
        statements = _globalize_statements(statements)
    _assert_unique_inverse_targets(statements)
    return {"aoa_statements": statements}
