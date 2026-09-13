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
from typing import OrderedDict

from paddle.distributed.fleet.meta_parallel import PipelineLayer

from ..transformers.configuration_utils import PretrainedConfig
from ..transformers.model_utils import PipelinePretrainedModel
from .criterion import CriterionLayer


class CriterionLayerPipe(CriterionLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_tuple = False  # loss_func only return loss, no loss_sum

    def forward(self, logits, labels, mtp_logits=None):
        if isinstance(labels, tuple) and "sft" in self.loss_type:
            labels, loss_mask = labels
        loss = super().forward(logits, labels, mtp_logits=mtp_logits)
        return loss


class GeneralModelForCausalLMPipe(PipelinePretrainedModel, PipelineLayer):
    _get_tensor_parallel_mappings = None
    _init_weights = None
    _keep_in_fp32_modules = None
    _tied_weights_keys = ["lm_head.weight"]
    config_class = PretrainedConfig
    transpose_weight_keys = None

    @classmethod
    def register_cls_attr(cls, config_class=None, pretrained_model_class=None):
        if config_class is not None:
            cls.config_class = config_class
        if pretrained_model_class is not None:
            if hasattr(pretrained_model_class, "_get_tensor_parallel_mappings"):
                cls._get_tensor_parallel_mappings = pretrained_model_class._get_tensor_parallel_mappings
            if hasattr(pretrained_model_class, "_get_fuse_or_split_param_mappings"):
                cls._get_fuse_or_split_param_mappings = pretrained_model_class._get_fuse_or_split_param_mappings
            if hasattr(pretrained_model_class, "_init_weights"):
                cls._init_weights = pretrained_model_class._init_weights
            if hasattr(pretrained_model_class, "_keep_in_fp32_modules"):
                cls._keep_in_fp32_modules = pretrained_model_class._keep_in_fp32_modules
            if hasattr(pretrained_model_class, "transpose_weight_keys"):
                cls.transpose_weight_keys = pretrained_model_class.transpose_weight_keys
        return cls

    @classmethod
    def _prepare_pipeline_inputs_func(cls, inputs):
        first_stage_keys = [
            "input_ids",
            "attn_mask_startend_row_indices",
            "position_ids",
            "nbatch_pack_offset",
        ]
        if type(inputs) is dict or type(inputs) is OrderedDict:
            if "attention_mask" in inputs:
                first_stage_keys = [
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                    "nbatch_pack_offset",
                ]
            # (NOTE) attn_mask_start_row_indices is special for erniekit
            elif "attn_mask_start_row_indices" in inputs:
                first_stage_keys = [
                    "input_ids",
                    "attn_mask_start_row_indices",
                    "position_ids",
                    "nbatch_pack_offset",
                ]
        else:  # inputs is list
            if "attention_mask" in inputs[0]:
                first_stage_keys = [
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                    "nbatch_pack_offset",
                ]
            elif "attn_mask_start_row_indices" in inputs[0]:
                first_stage_keys = [
                    "input_ids",
                    "attn_mask_start_row_indices",
                    "position_ids",
                    "nbatch_pack_offset",
                ]
        last_stage_keys = ["labels", "loss_mask"]

        def get_expected_keys(inputs, keys):
            ret = tuple([inputs.pop(k) for k in keys if k in inputs])
            if len(ret) == 1:
                ret = ret[0]
            return ret

        if type(inputs) is dict or type(inputs) is OrderedDict:
            return [
                get_expected_keys(inputs, first_stage_keys),
                get_expected_keys(inputs, last_stage_keys),
            ]

        keys = list(inputs[0].keys())
        inputs_batch = {key: [data.pop(key) for data in inputs] for key in keys}
        return [
            get_expected_keys(inputs_batch, first_stage_keys),
            get_expected_keys(inputs_batch, last_stage_keys),
        ]
