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


import paddle


@paddle.no_grad()
def new_attention_forward(
    self,
    hidden_states: paddle.Tensor,
    past_key_values=None,
    attention_mask: paddle.Tensor | None = None,
    attn_mask_startend_row_indices: paddle.Tensor | None = None,
    position_embeddings: tuple[paddle.Tensor, paddle.Tensor] | None = None,
    use_cache: bool = False,
):
    from .patch_utils import attention_branch

    if self.config.sequence_parallel:
        seq_len = self.config.max_sequence_length
        batch_size = (
            hidden_states.shape[0]
            * self.config.tensor_model_parallel_size
            // seq_len
        )
    else:
        batch_size, seq_len = hidden_states.shape[:2]

    q_shape = (batch_size, seq_len, -1, self.head_dim)
    kv_shape = (batch_size, seq_len, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).reshape(q_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).reshape(kv_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).reshape(kv_shape).transpose(1, 2)

    from paddleformers.transformers.llama.modeling import apply_rotary_pos_emb

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx
        )

    attn_output, attn_weights = attention_branch(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask=attention_mask,
        attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        dropout=0.0 if not self.training else self.attention_dropout,
        causal=True,
        scaling=self.scaling,
    )
    if self.config.sequence_parallel:
        attn_output = attn_output.reshape([-1, attn_output.shape[-1]])
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def patch_llama_attention(
    model,
    method: str = "rrattn",
    threshold: float = 0.9,
    stride: int = 8,
    rrattn_version: str = "v1",
    **kwargs,
):
    from paddleformers.transformers.llama.modeling import LLamaAttention

    from .patch_utils import patch_attention_layers

    return patch_attention_layers(
        model,
        LLamaAttention,
        new_attention_forward,
        method=method,
        threshold=threshold,
        stride=stride,
        rrattn_version=rrattn_version,
        **kwargs,
    )
