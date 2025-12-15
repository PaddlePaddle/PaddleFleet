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
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from paddlefleet.pipeline_parallel import (
    LayerDesc,
    PipelineLayer,
    SharedLayerDesc,
)

if TYPE_CHECKING:
    from paddlefleet.spec_utils import LayerSpec

from paddle.distributed import fleet

from paddlefleet.models.gpt.gpt_embedding import GPTEmbedding
from paddlefleet.models.gpt.lm_head import GPTLMHead
from paddlefleet.transformer.transformer_layer import TransformerLayer

logger = logging.getLogger(__name__)


@dataclass
class GPTSublayersSpec:
    """p
    The dataclass for LayerSpecs of GPT sublayers_spec
    including embedding, n * transformer_layer, mtp, lm_head.
    """

    embedding: LayerSpec | None = None
    transformer_layers: list[LayerSpec] | None = None
    layer_norm: LayerSpec | None = None
    mtp: list[LayerSpec] | None = None
    lm_head: LayerSpec | None = None


class GPTModel(PipelineLayer):
    """GPT Transformer language model.

    Args:
        gpt_layer_desc:
    """

    def __init__(
        self,
        sublayers_spec: GPTSublayersSpec,
        **kwargs,
    ) -> None:
        self.config = kwargs["config"]
        share_embeddings_and_output_weights = (
            kwargs["share_embeddings_and_output_weights"]
            and self.config.pipeline_model_parallel_size > 1
        )
        skip_weight_param_allocation = (
            self.config.share_embeddings_and_output_weights
            and self.config.pipeline_model_parallel_size == 1
        )
        self._pipeline_name_mapping = None
        self._pp_to_single_mapping = None
        self._sequential_layers = self.get_layer_desc_list(
            sublayers_spec,
            share_embeddings_and_output_weights,
        )
        self.layers = self.get_sequential_layers()
        del kwargs["share_embeddings_and_output_weights"]
        del kwargs["config"]

        topology = (
            None
            if self.config.pipeline_model_parallel_size == 1
            else fleet.get_hybrid_communicate_group().topology()
        )

        super().__init__(
            layers=self.layers,
            topology=topology,
            **kwargs,
        )

        if skip_weight_param_allocation:
            shared_embed_weight = None
            for layer in self.run_function:
                if isinstance(layer, GPTEmbedding):
                    shared_embed_weight = layer.embedding_weight
                if isinstance(layer, GPTLMHead):
                    layer.weight = shared_embed_weight

    def get_layer_desc_list(self, spec, share_embeddings_and_output_weights):
        layers = []
        if share_embeddings_and_output_weights:
            self.add_sequential_layer(
                layers,
                SharedLayerDesc(
                    "embed",
                    spec.embedding,
                    shared_weight_attr="embedding_weight",
                ),
                "model",
            )
        else:
            self.add_sequential_layer(
                layers, LayerDesc(spec.embedding), "model"
            )
        i = 0
        for transformer_layer_spec in spec.transformer_layers:
            self.add_sequential_layer(
                layers, LayerDesc(transformer_layer_spec), f"model.layers.{i}"
            )
            i += 1
        self.add_sequential_layer(layers, LayerDesc(spec.layer_norm), "model")

        if spec.mtp is not None:
            for mtp_spec in spec.mtp:
                self.add_sequential_layer(
                    layers, LayerDesc(mtp_spec), f"model.layers.{i}"
                )
                i += 1

        if share_embeddings_and_output_weights:
            self.add_sequential_layer(
                layers,
                SharedLayerDesc(
                    "embed",
                    spec.lm_head,
                    shared_weight_attr="embedding_weight",
                ),
                "model.shared_head",
            )
        else:
            self.add_sequential_layer(
                layers, LayerDesc(spec.lm_head), "model.lm_head"
            )

        return layers

    def get_hardware_flops(self):
        return 989e3

    def add_sequential_layer(self, layers, layer_desc, name_prefix=""):
        """
        Add a sequential layer to the network with specified description and name prefix.

        Args:
            layers (list): List to store layer descriptions. Each element should be a dict
                with keys "layer" (LayerDesc) and "name_prefix" (str).
            layer_desc (LayerDesc|SharedLayerDesc): Layer description object containing
                layer self.configuration.
            name_prefix (str, optional): Prefix for layer names in the pipeline.
                Defaults to empty string.

        Returns:
            None: The layer description is appended to the input layers list.
        """
        layers.append({"layer": layer_desc, "name_prefix": name_prefix})

    def get_sequential_layers(self):
        """
        Get all layers in the sequential network.

        Returns:
            List[paddle.nn.Layer]: List containing all layers.
        """
        return [x["layer"] for x in self._sequential_layers]

    def get_sequential_name_prefixs(self):
        """
        Retrieve name prefixes for all parallel layers in the sequential network.

        Returns:
            Dict[str, str]: A dictionary mapping layer indices (as strings) to their
                corresponding name prefixes. The indices represent the position of
                each layer in the sequential order.
        """
        return {
            str(index): x["name_prefix"]
            for index, x in enumerate(self._sequential_layers)
        }

    def get_shardlayer_prefix(self, name_splited):
        """_summary_
            This function retrieves the prefix of a shared layer. The process involves:
            1. Identifying all key names of shared layers, like 'shared_weight01', 'shared_weight02', etc.
            2. For instance, given name_splited = ['shared_layers', 'shared_weight01', 'weight'],
                the 'shared_layer_key' would be name_splited[1], which is 'shared_weight01'.
            3. By traversing through all layers, the function checks if the specified
                shared_layer is present in the current stage. If found, it returns the corresponding prefix.

            Note: For retrieving all SharedLayer instances in Paddle, you can refer to the following Paddle code.
            https://github.com/PaddlePaddle/Paddle/blob/2cf724d055679a1a0e48766dfb1708b920273078/python/paddle/distributed/fleet/meta_parallel/parallel_layers/pp_layers.py#L460-L513
        Args:
            name_splited (_type_): _description_

        Returns:
            _type_: _description_
        """
        shared_layer_names = {
            s.layer_name for s in self.layers if isinstance(s, SharedLayerDesc)
        }
        assert name_splited[1] in shared_layer_names, (
            f"The shared layer name {name_splited[1]} must be in prefixes!"
        )
        shared_layer_key = name_splited[1]
        for idx, layer in enumerate(self.layers):
            if (
                isinstance(layer, SharedLayerDesc)
                and layer.layer_name == shared_layer_key
            ):
                if self.get_stage_from_index(idx) == self._stage_id:
                    return self.get_sequential_name_prefixs()[str(idx)]

        # the prefix must be in the current stage, else raise error
        raise ValueError(
            f"The shared layer {shared_layer_key} must be in the current stage!"
        )

    def _set_pipeline_name_mapping(self, mappings=None):
        """
        Set the name mapping for pipeline.

        Args:
            mappings (dict, optional): Dictionary storing name mapping relationships. Default is None, meaning no mapping operation.

        Returns:
            dict: Returns the updated or existing mapping relationship.

        """
        if mappings is not None:
            self._pipeline_name_mapping = mappings
        else:
            single_to_pp_mapping = {}
            pp_to_single_mapping = {}

            state_dict_keys = list(super().state_dict().keys())
            first_key = ""
            for k in state_dict_keys:
                if "shared_layers" not in k:
                    first_key = k
                    break
            first_key = first_key.split(".")
            # if use virtual pp_degree, the prefix is like 0.0.xxx
            # else it will be like 0.xxx
            use_virtual_pp_degree = (
                first_key[0].isdigit() and first_key[1].isdigit()
            )

            prefixes = self.get_sequential_name_prefixs()
            for k in state_dict_keys:
                name_splited = k.split(".")
                if use_virtual_pp_degree:
                    if name_splited[0].isdigit():
                        if name_splited[1].isdigit():
                            idx = str(
                                int(name_splited[0]) + int(name_splited[1])
                            )
                            single_name = [prefixes[idx]]
                            single_name.extend(name_splited[2:])
                        else:
                            single_name = [prefixes[str(len(prefixes) - 1)]]
                            single_name.extend(name_splited[2:])
                            logger.warning(
                                f"Please check! we treat this key as last layer, get {k}, \
                                        set origin name as {'.'.join(single_name)}"
                            )
                    elif name_splited[0] == "shared_layers":
                        single_name = [self.get_shardlayer_prefix(name_splited)]
                        single_name.extend(name_splited[2:])
                    else:
                        single_to_pp_mapping[k] = k
                        pp_to_single_mapping[k] = k
                        continue
                else:
                    idx = name_splited[0]
                    # for normal pp layer
                    if idx.isdigit():
                        # allow empty prefix
                        single_name = (
                            [] if prefixes[idx] == "" else [prefixes[idx]]
                        )
                        single_name.extend(name_splited[1:])
                    elif idx == "shared_layers":
                        single_name = [self.get_shardlayer_prefix(name_splited)]
                        single_name.extend(name_splited[2:])
                    else:
                        single_to_pp_mapping[k] = k
                        pp_to_single_mapping[k] = k
                        continue

                single_to_pp_mapping[".".join(single_name)] = k
                pp_to_single_mapping[k] = ".".join(single_name)

            self._pipeline_name_mapping = single_to_pp_mapping
            self._pp_to_single_mapping = pp_to_single_mapping

        return self._pipeline_name_mapping

    def _check_shared_model_state(self):
        if self._pipeline_name_mapping is None:
            self._set_pipeline_name_mapping()

        super_state_dict = super().state_dict()
        structure_name_to_tensor = {}
        for k, v in super_state_dict.items():
            k = self._pp_to_single_mapping[k]
            if k not in structure_name_to_tensor:
                structure_name_to_tensor[k] = v
            else:
                old_v = structure_name_to_tensor[k]
                assert old_v is v, (
                    f"Shared tensor with different structure name: {k}"
                )

        missing_shared_keys = {}
        for k, v in self._pp_to_single_mapping.items():
            mapped_k = self._pipeline_name_mapping[v]
            if k != mapped_k:
                missing_shared_keys[k] = mapped_k
        return missing_shared_keys

    def state_dict(self, *args, **kwargs):
        """
        Return a dictionary with Pipeline Stage mapping.

        Args:
            *args (tuple): Variable argument list passed to parent method.
            **kwargs (dict): Optional keyword arguments passed to parent method.

        Returns:
            dict: Dictionary containing Pipeline Stage mapping.

        """
        state_dict = super().state_dict(*args, **kwargs)

        if self._pipeline_name_mapping is None:
            self._set_pipeline_name_mapping()
        # assert len(self._pipeline_name_mapping) > 0, "The pipeline stage must have parameters!"

        for k in list(state_dict.keys()):
            v = state_dict.pop(k)
            state_dict[self._pp_to_single_mapping[k]] = v

        return state_dict

    def sharded_state_dict(self, *args, **kwargs):
        """
        sharded_state_dict method for PipelinePretrainedModel.

        Remaps parameter keys according to the pipeline stage mapping, and converts expert indices from local to global.
        """
        sharded_state_dict = super().sharded_state_dict(*args, **kwargs)
        if self._pipeline_name_mapping is None:
            self._set_pipeline_name_mapping()

        for k in list(sharded_state_dict.keys()):
            v = sharded_state_dict.pop(k)
            v.key = self._pp_to_single_mapping[k]
            sharded_state_dict[self._pp_to_single_mapping[k]] = v

        def increment_expert_number(s, increment):
            import re

            def replace(match):
                original_number = int(match.group(0))
                new_number = original_number + increment
                return str(new_number)

            return re.sub(r"(?<=experts\.)\d+", replace, s)

        renamed_sharded_state_dict = {}
        for k, v in sharded_state_dict.items():
            global_expert_id_offset = getattr(
                v, "global_expert_id_offset", None
            )
            layer_cnt = getattr(v, "layer_cnt", None)
            if global_expert_id_offset is not None:
                new_key = increment_expert_number(k, global_expert_id_offset)
                v.key = new_key
                delattr(v, "global_expert_id_offset")
                renamed_sharded_state_dict[new_key] = v
            elif layer_cnt is not None:
                new_key = k + "_layer_" + str(layer_cnt)
                v.key = new_key
                delattr(v, "layer_cnt")
                renamed_sharded_state_dict[new_key] = v
            else:
                renamed_sharded_state_dict[k] = v

        return renamed_sharded_state_dict

    def fp8_quant_weight(self, batch_mode=False, quant_transpose=True):
        for idx, layer in enumerate(self.run_function):
            if isinstance(layer, TransformerLayer):
                layer.fp8_quant_weight(
                    batch_mode=batch_mode, quant_transpose=quant_transpose
                )

    def use_fp8(self):
        for idx, layer in enumerate(self.run_function):
            if isinstance(layer, TransformerLayer) and layer.use_fp8():
                return True
        return False
