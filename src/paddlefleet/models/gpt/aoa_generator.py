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
"""Modular AOA generation for the ``GPTModel`` boundary.

:func:`build_aoa_context` builds the read-only ``AOAContext`` once per pass;
:func:`gen_whole_model_aoa` / :func:`gen_whole_model_inv_aoa` forward it to each
live child through the standard ``Layer.gen_aoa_statements`` /
``gen_inv_aoa_statements`` protocol, which is the component dispatch point;
:func:`_globalize_statements` all-gathers the pipeline-local results into the
globally complete config both consumers require. The two directions are
generated independently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import paddle.distributed
from paddle.distributed.fleet.meta_parallel.parallel_layers.pp_layers import (
    PipelineLayerChunk,
)
from paddle.distributed.flex_checkpoint.aoa.generation import (
    AOAContext,
    validate_checkpoint_name_mapping,
)

from paddlefleet.parallel_state import get_pipeline_model_parallel_group

if TYPE_CHECKING:
    from collections.abc import Iterator


# The checkpoint root prefix a model inherits when it declares no
# ``aoa_checkpoint_name_prefix``. Used only by the identity fallback for names a
# model does not remap; the model-specific name mapping itself is declared by
# the model through ``aoa_checkpoint_name_mapping`` and is not defaulted here.
DEFAULT_CHECKPOINT_NAME_PREFIX = "model"


def build_aoa_context(model, config) -> AOAContext:
    """Builds the read-only ``AOAContext`` for a whole-model pass.

    Takes the single-name mapping from the live model's
    ``_pp_to_single_mapping`` -- the same source ``sharded_state_dict`` uses --
    and the model root from ``_model_name_prefix()``, the value
    ``get_layer_desc_list`` names its pipeline layers with, so pipeline naming
    and AOA name resolution never diverge.
    """
    if model._pipeline_name_mapping is None:
        model._set_pipeline_name_mapping()
    # A model declares its own checkpoint-name mapping through
    # ``aoa_checkpoint_name_mapping``; an absent or ``None`` attribute leaves it
    # empty, so every name resolves through the identity fallback. The
    # checkpoint prefix, by contrast, has a shared default.
    name_mapping = getattr(config, "aoa_checkpoint_name_mapping", None)
    if name_mapping is None:
        name_mapping = {}
    name_prefix = getattr(config, "aoa_checkpoint_name_prefix", None)
    if name_prefix is None:
        name_prefix = DEFAULT_CHECKPOINT_NAME_PREFIX
    checkpoint_name_mapping = dict(name_mapping)
    model_name_prefix = model._model_name_prefix()
    validate_checkpoint_name_mapping(
        checkpoint_name_mapping,
        model_name_prefix=model_name_prefix,
    )
    return AOAContext(
        config=config,
        pp_to_single_mapping=model._pp_to_single_mapping or {},
        model_name_prefix=model_name_prefix,
        checkpoint_name_mapping=checkpoint_name_mapping,
        checkpoint_name_prefix=name_prefix,
    )


def _iter_pipeline_units(model) -> Iterator[tuple[str, object]]:
    """Yields ``(structured_prefix, unit)`` for every live top-level layer.

    Descends one level into VPP :class:`PipelineLayerChunk` containers so a
    chunk's inner layer surfaces with its real two-segment prefix
    (``f"{chunk}.{local}."``); every other entry surfaces directly. Like the
    ``fp8_quant_weight`` live-layer idiom, it does not special-case cudagraph /
    PipelineSublayers wrapping.
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


def _globalize_statements(config, local_statements: list[str]) -> list[str]:
    """All-gathers the pipeline-local statements into the whole-model set.

    The walk covers only the stage this rank owns, while both consumers need the
    globally complete set: DCP ``save_full_param`` (``num_splits=1``) requires
    every rank's config to be identical, and ``dist.load_state_dict`` validates
    against the whole-world union of model keys. Pipeline is the only dimension
    the live module tree is a fragment over -- a leaf names parameters without
    reading shapes, so tensor / data / sharding ranks emit identical statements
    -- hence a declared pipeline with no group to gather over is an error rather
    than a degradation.

    The group's ranks own disjoint layers and together own all of them (VPP
    included), so rank-order concatenation is complete and needs no dedup, and
    every rank ends up with the same order, which order-sensitive statements
    require.
    """
    group = get_pipeline_model_parallel_group(check_initialized=False)
    if group is not None and group.nranks > 1:
        gathered: list[list[str]] = []
        paddle.distributed.all_gather_object(gathered, local_statements, group)
        return [
            statement
            for rank_statements in gathered
            for statement in rank_statements
        ]
    stages = getattr(config, "pipeline_model_parallel_size", 1)
    if isinstance(stages, int) and stages > 1:
        raise RuntimeError(
            f"AOA generation needs a pipeline-group gather: "
            f"pipeline_model_parallel_size={stages} means this rank's live "
            f"module tree is one stage of the model, but no pipeline process "
            f"group is available, so the generated config would stay rank-local "
            f"and miss the other stages' keys."
        )
    return local_statements


def gen_whole_model_aoa(model, ctx) -> dict[str, list[str]]:
    """Whole-model checkpoint->model generation.

    Hands each live child to the standard ``Layer.gen_aoa_statements`` protocol,
    so component overrides (Linear, SelfAttention, MoE, ...) emit their own
    rules and the module walk is never re-implemented here.
    """
    statements = []
    # The boundary owns no parameters, so recursion starts one level down.
    for structured_prefix, unit in _iter_pipeline_units(model):
        statements += unit.gen_aoa_statements(
            ctx, structured_name_prefix=structured_prefix
        )
    return {"aoa_statements": _globalize_statements(ctx.config, statements)}


def gen_whole_model_inv_aoa(model, ctx) -> dict[str, list[str]]:
    """Whole-model model->checkpoint generation.

    Independently recurses through ``Layer.gen_inv_aoa_statements``; never
    derived from the checkpoint->model pass.
    """
    statements = []
    # Reverse order mirrors the checkpoint->model pass; cosmetic for statements.
    for structured_prefix, unit in reversed(list(_iter_pipeline_units(model))):
        statements += unit.gen_inv_aoa_statements(
            ctx, structured_name_prefix=structured_prefix
        )
    return {"aoa_statements": _globalize_statements(ctx.config, statements)}
