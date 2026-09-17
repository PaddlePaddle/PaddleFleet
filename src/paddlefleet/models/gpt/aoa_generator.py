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
"""Modular AOA checkpoint protocol for the ``GPTModel`` boundary.

:func:`build_aoa_context` builds the read-only ``AOAContext`` once per
whole-model pass. It is then forwarded unchanged down the module tree by the
standard ``Layer.gen_aoa_statements`` / ``gen_inv_aoa_statements`` protocol,
which is the component dispatch point.
"""

from __future__ import annotations

from paddle.distributed.flex_checkpoint.aoa.generation import (
    AOAContext,
    validate_checkpoint_name_mapping,
)

# The ERNIE-series checkpoint layout, used when a model declares no ``aoa_*``
# override. Both sides are absolute names carrying their root prefix. Only
# naming divergences belong here: names the components already emit identically
# (layernorms, ``model.norm``, an MTP layer's ``enorm`` / ``hnorm`` /
# ``eh_proj``, the CSA subtree) are absent. The layer root omits
# ``transformer_layer`` so one entry covers both an ordinary layer and an MTP
# layer's inner transformer. The output head is the one entry whose value sits
# outside the checkpoint root: the ``ForCausalLM`` layout keeps ``lm_head`` a
# top-level sibling of the backbone.
DEFAULT_CHECKPOINT_NAME_MAPPING = {
    "model.embedding.embed_tokens.weight": "model.embed_tokens.weight",
    "model.layers.$LAYER_ID.mlp.gate.weight": "model.layers.$LAYER_ID.block_sparse_moe.gate.weight",
    "model.layers.$LAYER_ID.mlp.gate.weight_1": "model.layers.$LAYER_ID.block_sparse_moe.gate.weight_1",
    "model.layers.$LAYER_ID.mlp.gate.routed_scaling_factor_param": "model.layers.$LAYER_ID.block_sparse_moe.gate.routed_scaling_factor_param",
    "model.layers.$LAYER_ID.mlp.gate.e_score_correction_bias": "model.layers.$LAYER_ID.block_sparse_moe.e_score_correction_bias",
    "model.layers.$LAYER_ID.mlp.fc1_latent_proj.weight": "model.layers.$LAYER_ID.block_sparse_moe.fc1_latent_proj.weight",
    "model.layers.$LAYER_ID.mlp.fc2_latent_proj.weight": "model.layers.$LAYER_ID.block_sparse_moe.fc2_latent_proj.weight",
    "model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.gate_proj.weight": "model.layers.$LAYER_ID.block_sparse_moe.experts.$EXPERT_ID.w1.weight",
    "model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_proj.weight": "model.layers.$LAYER_ID.block_sparse_moe.experts.$EXPERT_ID.w3.weight",
    "model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight": "model.layers.$LAYER_ID.block_sparse_moe.experts.$EXPERT_ID.w2.weight",
    "model.layers.$LAYER_ID.mlp.shared_experts.gate_proj.weight": "model.layers.$LAYER_ID.block_sparse_moe.shared_experts.w1.weight",
    "model.layers.$LAYER_ID.mlp.shared_experts.up_proj.weight": "model.layers.$LAYER_ID.block_sparse_moe.shared_experts.w3.weight",
    "model.layers.$LAYER_ID.mlp.shared_experts.down_proj.weight": "model.layers.$LAYER_ID.block_sparse_moe.shared_experts.w2.weight",
    "model.layers.$LAYER_ID.norm.weight": "model.layers.$LAYER_ID.shared_head.norm.weight",
    "model.lm_head.weight": "lm_head.weight",
    "model.lm_head.bias": "lm_head.bias",
    "model.shared_head.weight": "lm_head.weight",
    "model.shared_head.bias": "lm_head.bias",
}
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
    # An absent or ``None`` attribute falls back to the default; a declared one
    # is taken verbatim, including a falsy value (an empty checkpoint prefix).
    name_mapping = getattr(config, "aoa_checkpoint_name_mapping", None)
    if name_mapping is None:
        name_mapping = DEFAULT_CHECKPOINT_NAME_MAPPING
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
