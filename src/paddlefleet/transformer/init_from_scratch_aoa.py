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
"""AOA "add" helper: a whole subtree that the checkpoint does not contain.

Multi-phase training grows the model between phases. A phase-1 checkpoint holds
no Indexer at all (``csa_dense_mode=true`` / ``hybrid_mla_attention="mha"``
builds none), yet phase 2 must load that checkpoint into a model that has one.
AOA expresses this with the source-less statement ``_ -> <model name>``: the key
is declared to have no checkpoint source, so it keeps what the freshly built
module initialised while every other tensor loads normally. Omitting the
statement instead would not work -- an uncovered key falls back to identity
naming and aborts with ``... should be assigned before!``.

:func:`gen_init_from_scratch_aoa` emits that statement for every tensor of a
subtree, and :func:`resolve_init_from_scratch` reads the switch that selects the
branch, refusing to guess when it is unset.

Checkpoint -> model only. The inverse direction is unconditional: whatever the
model owns is written out, so a component using this helper must not override
``gen_inv_aoa_statements``.
"""

from __future__ import annotations

from paddle.distributed.flex_checkpoint.aoa.generation import resolve_names


def resolve_init_from_scratch(config, what, gated_by):
    """Read the mandatory ``indexer_init_from_scratch`` switch off ``config``.

    Leaving it unset is an error rather than a default: it says which checkpoint
    the run starts from, which is a deliberate decision. Guessing it wrong is
    silent -- ``False`` against a checkpoint that lacks the tensors aborts with
    ``... should be assigned before!``, while ``True`` against one that has them
    throws away everything trained so far.

    Args:
        config: The ``TransformerConfig`` of the component.
        what: What the model builds, for the error message (e.g. ``"a DSA
            Indexer"``).
        gated_by: The config that makes the model build it, for the error
            message (e.g. ``'hybrid_mla_attention="mqa_dsa"'``).

    Returns:
        The switch value.

    Raises:
        ValueError: When ``indexer_init_from_scratch`` is ``None``.
    """
    field = "indexer_init_from_scratch"
    value = getattr(config, field, None)
    if value is None:
        raise ValueError(
            f"{field} must be set explicitly when loading a checkpoint into a "
            f"model that creates {what} ({gated_by}). It selects which "
            "checkpoint you start from:\n"
            f"  {field}: true   # phase-1 checkpoint: it does not have these "
            "tensors, initialize them randomly\n"
            f"  {field}: false  # checkpoint already contains them: load them "
            "(phase 2 restart, or phase 3 from phase 2, or phase 3 restart)"
        )
    return value


def gen_init_from_scratch_aoa(
    layer, ctx, *, structured_name_prefix="", aoa_name_scope=None
):
    """``_ -> <model name>`` for every tensor of ``layer`` and its sub-layers.

    Walks own parameters / persistable buffers and then recurses, the same
    traversal ``Layer.gen_aoa_statements`` and ``Layer.sharded_state_dict`` use,
    so the emitted keys are exactly the ones the engine has to assign. The
    recursion deliberately does not dispatch to sub-layer overrides: a
    source-less statement has no checkpoint side, and the model side of every
    override resolves the same way, so the plain walk is both sufficient and
    immune to their layout rewrites.

    No ``should_skip`` (an add is never redundant) and no dtype cast (there is
    no source to cast from). ``ctx.excluded_names`` is honoured, so a tensor
    another component claims stays claimed.

    Args:
        layer: Root of the subtree that the checkpoint does not contain.
        ctx: Read-only ``AOAContext`` for the generation pass.
        structured_name_prefix: Live module path prefix of ``layer``, ending in
            ``.`` when non-empty, as in ``sharded_state_dict``.
        aoa_name_scope: Optional checkpoint-side scope, passed down unchanged.

    Returns:
        One statement per tensor in the subtree.
    """
    statements = []
    own_state_dict = layer.state_dict(
        structured_name_prefix="", include_sublayers=False
    )
    for name in own_state_dict:
        if structured_name_prefix + name in ctx.excluded_names:
            continue
        _, single_name = resolve_names(
            name,
            ctx.checkpoint_name_prefix,
            structured_name_prefix,
            ctx.pp_to_single_mapping,
            ctx.checkpoint_name_mapping,
            aoa_name_scope=aoa_name_scope,
            model_name_prefix=ctx.model_name_prefix,
        )
        statements.append(f"_ -> {single_name}")
    for layer_name, sublayer in layer._sub_layers.items():
        if sublayer is not None:
            statements += gen_init_from_scratch_aoa(
                sublayer,
                ctx,
                structured_name_prefix=f"{structured_name_prefix}{layer_name}.",
                aoa_name_scope=aoa_name_scope,
            )
    return statements
