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
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from functools import partial

from paddlefleet.models.common.language_loss.language_loss import LanguageLoss
from paddlefleet.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_layers_spec,
    get_gpt_layer_local_spec,
    get_gpt_mtp_layers_spec,
    get_gpt_spec,
)
from paddlefleet.spec_utils import build_layer


def gpt_builder(config, **kwargs):
    print("building GPT model ...")
    if config.n_routed_experts:
        # Define the decoder block spec
        transformer_layers_spec = get_gpt_decoder_layers_spec(
            config,
            normalization=config.normalization,
        )
    else:
        # Define the decoder layer spec
        transformer_layer_spec_func = _get_transformer_layer_spec_func(config)
        transformer_layers_spec = []
        for layer_number in range(config.num_hidden_layers):
            transformer_layers_spec.append(
                transformer_layer_spec_func(layer_number=layer_number)
            )
    mtp_layers_spec = None
    if config.num_nextn_predict_layers is not None:
        if (
            hasattr(transformer_layers_spec, "layer_specs")
            and len(transformer_layers_spec.layer_specs) == 0
        ):
            transformer_layers_spec_for_mtp_func = (
                _get_transformer_layer_spec_func(config)
            )
            transformer_layers_spec_for_mtp = []
            for layer_number in range(config.num_layers):
                transformer_layers_spec_for_mtp.append(
                    transformer_layers_spec_for_mtp_func(
                        layer_number=layer_number
                    )
                )
        else:
            transformer_layers_spec_for_mtp = transformer_layers_spec
        mtp_layers_spec = get_gpt_mtp_layers_spec(
            config,
            transformer_layers_spec_for_mtp,
        )

    gpt_spec = get_gpt_spec(
        config=config,
        transformer_layers_spec=transformer_layers_spec,
        mtp_layers_spec=mtp_layers_spec,
        vocab_size=config.vocab_size,
        share_embeddings_and_output_weights=config.share_embeddings_and_output_weights,
        max_sequence_length=config.max_sequence_length,
        position_embedding_type=config.position_embedding_type,
        rotary_percent=config.rotary_percent,
        rotary_base=config.rotary_base,
        rope_scaling=config.rope_scaling,
        parallel_output=config.parallel_output,
    )

    return build_layer(gpt_spec, loss_fn=LanguageLoss(config), **kwargs)


def _get_transformer_layer_spec_func(config):
    """Get transformer layer specification based on configuration.

    Args:
        config: Model configuration

    Returns:
        transformer_layer_spec: The transformer layer specification
    """
    return partial(
        get_gpt_layer_local_spec,
        config=config,
        use_qk_norm=config.use_qk_norm,
        num_experts=config.n_routed_experts,
        multi_latent_attention=config.multi_latent_attention,
        normalization=config.normalization,
    )
