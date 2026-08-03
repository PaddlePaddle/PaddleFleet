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

"""Kimi-K3 ``sd2_tpool`` merge and ``PatchMergerMLPV2`` projector.

Two pipeline stages, matching the HuggingFace reference:

* ``KimiK3VisionSd2TpoolMerger`` (no parameters): groups each 2x2 spatial
  neighbourhood and mean-pools over *all* temporal frames, producing one
  ``[new_h * new_w, kh * kw, hidden]`` tensor per media.
* ``KimiK3VisionPatchMerger``: flattens each group to ``kh * kw * hidden`` and
  projects it to the text hidden size with a bias-free 2-layer MLP, then
  applies the projector norm **after** the projection (K3 semantics; note that
  Kimi-K2.5 instead pre-norms before the projection).
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass

from paddle import nn
from paddle.nn import functional as F

from ...spec_utils import LayerSpec, build_layer
from ...tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from ...transformer.identity_op import IdentityOp
from ...transformer.mlp import MLP, MLPSublayersSpec


class KimiK3VisionSd2TpoolMerger(nn.Layer):
    """Spatial 2x2 grouping with temporal mean pooling (``sd2_tpool``)."""

    def __init__(self, config):
        super().__init__()
        self.merge_kernel_size = tuple(config.merge_kernel_size)

    def forward(self, dict_args: dict):
        hidden_states = dict_args["hidden_states"]
        grid_thws = dict_args["grid_thws"]
        if hidden_states.ndim == 3:
            hidden_states = hidden_states.squeeze(0)
        d_model = hidden_states.shape[-1]
        kernel_height, kernel_width = self.merge_kernel_size

        outputs = []
        pre_sum = 0
        for t, h, w in grid_thws.tolist():
            t, h, w = int(t), int(h), int(w)
            seq = hidden_states[pre_sum : pre_sum + t * h * w]
            new_height, new_width = h // kernel_height, w // kernel_width
            seq = seq.reshape(
                [t, new_height, kernel_height, new_width, kernel_width, d_model]
            )
            # temporal pooling over all frames
            seq = seq.transpose([0, 1, 3, 2, 4, 5]).mean(axis=0)
            outputs.append(
                seq.reshape(
                    [new_height * new_width, kernel_height * kernel_width, -1]
                )
            )
            pre_sum += t * h * w

        rst = OrderedDict(dict_args)
        rst["hidden_states"] = outputs
        return rst


@dataclass
class KimiK3VisionPatchMergerSpec:
    """LayerSpecs of the sublayers owned by ``KimiK3VisionPatchMerger``."""

    norm: LayerSpec | type = IdentityOp


class KimiK3VisionPatchMerger(nn.Layer):
    """``PatchMergerMLPV2``: bias-free ``4C -> 4C -> text_hidden`` + post norm."""

    def __init__(self, config, sublayers_spec: KimiK3VisionPatchMergerSpec):
        super().__init__()
        kernel_height, kernel_width = config.merge_kernel_size
        self.merged_size = config.mm_hidden_size * (
            kernel_height * kernel_width
        )

        # The encoder MLP uses tanh-approximated gelu, the projector uses the
        # exact gelu in the reference implementation.
        proj_config = copy.copy(config)
        proj_config.hidden_act = F.gelu
        self.proj = build_layer(
            LayerSpec(
                layer=MLP,
                sublayers_spec=MLPSublayersSpec(
                    up_gate_proj=ColumnParallelLinear,
                    down_proj=RowParallelLinear,
                ),
                extra_kwargs={
                    "config": proj_config,
                    "input_size": self.merged_size,
                    "intermediate_size": self.merged_size,
                    "hidden_size": config.text_hidden_size,
                },
            )
        )

        self.post_norm = build_layer(
            sublayers_spec.norm,
            config=config,
            hidden_size=config.text_hidden_size,
            eps=config.projector_ln_eps,
        )

    def _project(self, x):
        out = self.proj(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return self.post_norm(out)

    def forward(self, dict_args: dict):
        x = dict_args["hidden_states"]
        if isinstance(x, (list, tuple)):
            x = [
                self._project(item.reshape([item.shape[0], self.merged_size]))
                for item in x
            ]
        else:
            x = self._project(x.reshape([x.shape[0], -1, self.merged_size]))

        rst = OrderedDict(dict_args)
        rst["hidden_states"] = x
        return rst
