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


from copy import deepcopy

import paddle
import paddle.nn.functional as F

from paddlefleet.accuracy_target import targets_hf
from paddlefleet.transformer.mlp import MLP, MLPSublayersSpec
from paddlefleet.transformer.transformer_config import TransformerConfig


class StandardMLPSharedExpert(MLP):
    # 共享专家的两个投影对应的延后点。注意它的 backward 本来就跟 combine 集合
    # 通信重叠（token_dispatcher 把它作为 callback 交给 combine），所以打开这两项
    # 是把计算从 combine 窗口挪到 p2p 窗口，而不是凭空造出填充料，要单独量。
    _dw_up_gate_point = "moe_shared_expert_up_gate_proj"
    _dw_down_point = "moe_shared_expert_down_proj"

    def __init__(
        self,
        config: TransformerConfig,
        moe_intermediate_size: int,
        is_expert: bool,
        mlp_spec: MLPSublayersSpec,
    ):
        if moe_intermediate_size == config.intermediate_size:
            super().__init__(
                config,
                mlp_spec,
                is_expert=is_expert,
                intermediate_size=moe_intermediate_size,
                # tp_group=pg_collection.expt_tp,
            )
        else:
            # Local SequentialMLP can still be used here by overriding the intermediate_size
            # with a deepcopied config.
            sequential_mlp_config = deepcopy(config)
            sequential_mlp_config.intermediate_size = moe_intermediate_size
            super().__init__(
                sequential_mlp_config,
                mlp_spec,
                is_expert=is_expert,
                intermediate_size=moe_intermediate_size,
                # tp_group=pg_collection.expt_tp,
            )
        self.use_shared_expert_gate = config.moe_shared_expert_gate
        # Keep bf16 activation for backward; the gate already holds the
        # high-precision copy.
        self.up_gate_proj.save_original_input = True
        self._maybe_tag_up_gate_norm_groups()
        if self.use_shared_expert_gate:
            self.gate_weight = paddle.create_parameter(
                shape=[config.hidden_size, 1],
                dtype=config.params_dtype,
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
            # Initialize with Normal distribution aligned with Megatron.
            config.init_method(self.gate_weight)
        else:
            self.gate_weight = None

    def _maybe_tag_up_gate_norm_groups(self) -> None:
        """Tag ``up_gate_proj.weight`` with the reference's two projections.

        HF's ``Qwen3_5MoeMLP`` keeps ``gate_proj`` and ``up_proj`` as separate
        ``nn.Linear`` modules, so gradient clipping takes **two** per-tensor BF16
        norms over the fused weight's two column halves, not one over the whole
        thing. Squaring one BF16 norm of the concatenation is not the same as
        summing the squares of the two halves' BF16 norms, so the fused layout
        changes the global norm. Only the clip reads this; see
        ``paddleformers/utils/hf_bitexact_clip.py``.
        """
        # ``getattr``: the old short-circuited
        # ``HF_BITEXACT_ALIGN and self.use_accuracy_compatible`` never read the
        # attribute when the flag was off, so callers that stub out ``MLP.__init__``
        # (which is what sets it) used to work. Keep that tolerance.
        if not targets_hf(getattr(self, "use_accuracy_compatible", False)):
            return
        weight = getattr(self.up_gate_proj, "weight", None)
        if weight is None:
            return
        width = weight.shape[-1]
        if width % 2:
            return
        half = width // 2
        weight.hf_norm_groups = [
            paddle.to_tensor(list(range(0, half)), dtype="int64"),
            paddle.to_tensor(list(range(half, width)), dtype="int64"),
        ]

    def forward(
        self,
        hidden_states: paddle.Tensor,
        hidden_states_up: paddle.Tensor | None = None,
        hidden_states_gate: paddle.Tensor | None = None,
    ) -> paddle.Tensor:
        output, output_bias = super().forward(
            hidden_states, hidden_states_up=hidden_states_up
        )
        if self.use_shared_expert_gate:
            gate_source = (
                hidden_states
                if hidden_states_gate is None
                else hidden_states_gate
            )
            logits = F.linear(gate_source, self.gate_weight)
            gate_score = F.sigmoid(logits)
            output = output * gate_score
        return output, output_bias
