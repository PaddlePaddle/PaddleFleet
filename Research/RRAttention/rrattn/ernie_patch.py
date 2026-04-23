from typing import Optional, Tuple

import paddle
from paddleformers.transformers.ernie4_5.modeling import (
    Ernie4_5Attention,
    apply_fused_rope,
    apply_rotary_pos_emb,
)

from .patch_utils import attention_branch, patch_attention_layers


def get_ernie_attention_classes():
    attention_classes = [Ernie4_5Attention]
    try:
        from paddleformers.transformers.ernie4_5_moe.modeling import Ernie4_5_MoeAttention
    except ImportError:
        pass
    else:
        attention_classes.append(Ernie4_5_MoeAttention)
    return tuple(attention_classes)


@paddle.no_grad()
def new_attention_forward(
    self,
    hidden_states,
    past_key_values=None,
    attention_mask: Optional[paddle.Tensor] = None,
    attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
    position_embeddings: Optional[Tuple[paddle.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
):
    """Compute attention outputs."""
    mix_layer = self.qkv_proj(hidden_states)
    if self.config.sequence_parallel:
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

    # b l h d -> b h l d
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    if self.config.apply_rope_fusion:
        query_states, key_states = apply_fused_rope(query_states, key_states, self.config.rope_theta)
    else:
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    attn_output, attn_weights = attention_branch(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask=attention_mask,
        attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        dropout=self.config.get("attention_dropout_prob", 0.0) if self.training else 0.0,
        causal=True,
        scaling=self.scaling,
    )

    if self.config.sequence_parallel:
        attn_output = attn_output.reshape([-1, attn_output.shape[-1]])
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights


def patch_ernie_attention(
    model,
    method: str = "rrattn",
    threshold: float = 0.9,
    stride: int = 8,
    rrattn_version: str = "v1",
    **kwargs,
):
    return patch_attention_layers(
        model,
        get_ernie_attention_classes(),
        new_attention_forward,
        method=method,
        threshold=threshold,
        stride=stride,
        rrattn_version=rrattn_version,
        **kwargs,
    )
