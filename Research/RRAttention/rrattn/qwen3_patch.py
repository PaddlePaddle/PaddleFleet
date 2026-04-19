from typing import Optional, Tuple

import paddle
from paddleformers.transformers.qwen3.modeling import Qwen3Attention, apply_rotary_pos_emb

try:
    from paddleformers.transformers.qwen3_moe.modeling import Qwen3MoeAttention
except ImportError:
    Qwen3MoeAttention = None

from .patch_utils import attention_branch, patch_attention_layers


@paddle.no_grad()
def new_attention_forward(
    self,
    hidden_states,
    position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
    attention_mask: Optional[paddle.Tensor] = None,
    past_key_values=None,
    use_cache: bool = False,
    attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
    batch_size: Optional[int] = None,
    **kwargs,
):
    """Input shape: Batch x Time x Channel"""
    mix_layer = self.qkv_proj(hidden_states)
    if self.sequence_parallel:
        max_sequence_length = self.config.max_sequence_length
        bsz = hidden_states.shape[0] * self.config.tensor_model_parallel_size // max_sequence_length
        q_len = max_sequence_length
        target_shape = [
            bsz,
            q_len,
            self.num_key_value_heads,
            (self.num_key_value_groups + 2) * self.head_dim,
        ]
    else:
        target_shape = [0, 0, self.num_key_value_heads, (self.num_key_value_groups + 2) * self.head_dim]
    mix_layer = paddle.reshape_(mix_layer, target_shape)
    query_states, key_states, value_states = paddle.split(
        mix_layer,
        num_or_sections=[self.num_key_value_groups * self.head_dim, self.head_dim, self.head_dim],
        axis=-1,
    )
    if self.gqa_or_mqa:
        query_states = paddle.reshape_(query_states, [0, 0, self.num_heads, self.head_dim])
    query_states = self.q_norm(query_states)
    key_states = self.k_norm(key_states)

    # [bs, seq_len, num_head, head_dim] -> [bs, num_head, seq_len, head_dim]
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    # key_states shape: [bs, seq_len, num_head, head_dim]
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

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


def patch_qwen3_attention(
    model,
    method: str = "rrattn",
    threshold: float = 0.9,
    stride: int = 8,
    rrattn_version: str = "v1",
    **kwargs,
):
    attention_cls = Qwen3Attention if Qwen3MoeAttention is None else (Qwen3Attention, Qwen3MoeAttention)
    return patch_attention_layers(
        model,
        attention_cls,
        new_attention_forward,
        method=method,
        threshold=threshold,
        stride=stride,
        rrattn_version=rrattn_version,
        **kwargs,
    )
