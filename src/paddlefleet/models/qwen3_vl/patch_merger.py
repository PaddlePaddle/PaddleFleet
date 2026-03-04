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
from dataclasses import dataclass

from paddle import nn

from ...spec_utils import LayerSpec, build_layer
from ...transformer.identity_op import IdentityOp


@dataclass
class Qwen3VLVisionPatchMergerSpec:
    norm: LayerSpec = IdentityOp


class Qwen3VLVisionPathMerger(nn.Module):
    def __init__(
        self,
        config,
        sublayers_spec: Qwen3VLVisionPatchMergerSpec,
        dim: int | None = None,
        context_dim: int | None = None,
        use_postshuffle_norm: bool = False,
    ):
        super().__init__()
        context_dim = (
            context_dim if context_dim is not None else config.hidden_size
        )
        dim = dim if dim is not None else config.out_hidden_size

        self.hidden_size = context_dim * (config.spatial_merge_size**2)
        norm_dim = self.hidden_size if use_postshuffle_norm else context_dim
        self.norm = build_layer(
            sublayers_spec.norm, config=config, hidden_size=norm_dim
        )
        self.use_postshuffle_norm = use_postshuffle_norm

        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, dim)

    def forward(self, dict_args):
        x = dict_args.pop("hidden_states")
        if self.use_postshuffle_norm:
            x = self.norm(x.reshape([-1, self.hidden_size]))
            x = x.reshape([-1, self.hidden_size])
        else:
            x = self.norm(x)
            x = x.reshape([-1, self.hidden_size])

        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        rst = {"hidden_states": x}
        rst = {**dict_args, **rst}
        return rst
