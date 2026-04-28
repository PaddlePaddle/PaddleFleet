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
    hidden_states,
    position_embeddings: tuple[paddle.Tensor, paddle.Tensor] | None = None,
    attention_mask: paddle.Tensor | None = None,
    past_key_values=None,
    use_cache: bool = False,
    attn_mask_startend_row_indices: paddle.Tensor | None = None,
    batch_size: int | None = None,
    **kwargs,
):
    from .patch_utils import attention_branch

    """Input shape: Batch x Time x Channel"""
    mix_layer = self.qkv_proj(hidden_states)
    if self.sequence_parallel:
        max_sequence_length = self.config.max_sequence_length
        bsz = (
            hidden_states.shape[0]
            * self.config.tensor_model_parallel_size
            // max_sequence_length
        )
        q_len = max_sequence_length
        target_shape = [
            bsz,
            q_len,
            self.num_key_value_heads,
            (self.num_key_value_groups + 2) * self.head_dim,
        ]
    else:
        target_shape = [
            0,
            0,
            self.num_key_value_heads,
            (self.num_key_value_groups + 2) * self.head_dim,
        ]
    mix_layer = paddle.reshape_(mix_layer, target_shape)
    query_states, key_states, value_states = paddle.split(
        mix_layer,
        num_or_sections=[
            self.num_key_value_groups * self.head_dim,
            self.head_dim,
            self.head_dim,
        ],
        axis=-1,
    )
    if self.gqa_or_mqa:
        query_states = paddle.reshape_(
            query_states, [0, 0, self.num_heads, self.head_dim]
        )

    # [bs, seq_len, num_head, head_dim] -> [bs, num_head, seq_len, head_dim]
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    from paddleformers.transformers.qwen2.modeling import apply_rotary_pos_emb

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )

    # [bs, seq_len, num_head]
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

    # if sequence_parallel is true, out shape are [q_len / n, bs, num_head * head_dim]
    # else their shape are [bs, q_len, num_head * head_dim], n is mp parallelism.
    if self.config.sequence_parallel:
        attn_output = attn_output.reshape([-1, attn_output.shape[-1]])
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights


def patch_qwen_attention(
    model,
    method: str = "rrattn",
    threshold: float = 0.9,
    stride: int = 8,
    rrattn_version: str = "v1",
    **kwargs,
):
    from paddleformers.transformers.qwen2.modeling import Qwen2Attention

    from .patch_utils import patch_attention_layers

    return patch_attention_layers(
        model,
        Qwen2Attention,
        new_attention_forward,
        method=method,
        threshold=threshold,
        stride=stride,
        rrattn_version=rrattn_version,
        **kwargs,
    )
