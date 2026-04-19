import math
import types
from typing import Iterable, Optional, Tuple

import paddle
import paddle.nn.functional as F
from paddleformers.nn.attention.eager_attention import repeat_kv
from paddleformers.utils.masking_utils import _gen_from_sparse_attn_mask_indices

from .full_prefill import FA_Full_prefill
from .rrattention import rrattn_prefill

SUPPORTED_METHODS = ("rrattn", "full")
PATCHED_ATTN_IMPLEMENTATION = "sdpa"


def validate_method(method: str):
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method={method!r}; supported methods are: rrattn, full")


def select_layer_value(value, layer_idx: int):
    if isinstance(value, (list, tuple)):
        return value[layer_idx]
    return value


def get_decoder_layers(model) -> Iterable:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    return []


def configure_model_attn_implementation(model, attn_implementation: str = PATCHED_ATTN_IMPLEMENTATION):
    modules = [model]
    if hasattr(model, "model"):
        modules.append(model.model)

    for module in modules:
        if module is None or not hasattr(module, "config"):
            continue
        config = module.config
        if not hasattr(config, "_rrattn_original_attn_implementation"):
            config._rrattn_original_attn_implementation = getattr(config, "_attn_implementation", None)
        config._attn_implementation = attn_implementation

    for _, attn in model.named_sublayers() if hasattr(model, "named_sublayers") else []:
        if not hasattr(attn, "config"):
            continue
        config = attn.config
        if not hasattr(config, "_rrattn_original_attn_implementation"):
            config._rrattn_original_attn_implementation = getattr(config, "_attn_implementation", None)
        config._attn_implementation = attn_implementation


def patch_attention_layers(
    model,
    attention_cls,
    new_forward,
    method: str = "rrattn",
    threshold: float = 0.9,
    stride: int = 8,
    rrattn_version: str = "v1",
    **kwargs,
):
    validate_method(method)
    patched = 0
    seen = set()

    for layer_idx, layer in enumerate(get_decoder_layers(model)):
        if not hasattr(layer, "self_attn"):
            continue
        attn = layer.self_attn
        if attention_cls is not None and not isinstance(attn, attention_cls):
            continue
        configure_attention_layer(
            attn,
            layer_idx=layer_idx,
            method=method,
            threshold=threshold,
            stride=stride,
            rrattn_version=rrattn_version,
            **kwargs,
        )
        bind_attention_forward(attn, new_forward)
        seen.add(id(attn))
        patched += 1

    if hasattr(model, "named_sublayers"):
        for _, attn in model.named_sublayers():
            if id(attn) in seen:
                continue
            if attention_cls is not None and not isinstance(attn, attention_cls):
                continue
            layer_idx = getattr(attn, "layer_idx", patched)
            configure_attention_layer(
                attn,
                layer_idx=layer_idx,
                method=method,
                threshold=threshold,
                stride=stride,
                rrattn_version=rrattn_version,
                **kwargs,
            )
            bind_attention_forward(attn, new_forward)
            seen.add(id(attn))
            patched += 1

    if patched == 0:
        if attention_cls is None:
            attention_name = "attention"
        elif isinstance(attention_cls, tuple):
            attention_name = "/".join(cls.__name__ for cls in attention_cls)
        else:
            attention_name = attention_cls.__name__
        raise ValueError(f"No {attention_name} layers found")
    configure_model_attn_implementation(model)
    return model


def configure_attention_layer(
    attn,
    layer_idx: int,
    method: str,
    threshold: float,
    stride: int,
    rrattn_version: str,
    **kwargs,
):
    attn.method = method
    attn.threshold = select_layer_value(threshold, layer_idx)
    attn.stride = select_layer_value(stride, layer_idx)
    attn.rrattn_version = select_layer_value(rrattn_version, layer_idx)
    attn.sparse_ratio = 0.0
    for key, value in kwargs.items():
        setattr(attn, key, select_layer_value(value, layer_idx))


def bind_attention_forward(attn, new_forward):
    if not hasattr(attn, "_rrattn_original_forward"):
        attn._rrattn_original_forward = attn.forward
    attn.forward = types.MethodType(new_forward, attn)


def attention_branch(
    module,
    query_states: paddle.Tensor,
    key_states: paddle.Tensor,
    value_states: paddle.Tensor,
    attention_mask: Optional[paddle.Tensor] = None,
    attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
    dropout: float = 0.0,
    causal: bool = True,
    scaling: Optional[float] = None,
) -> Tuple[paddle.Tensor, Optional[paddle.Tensor]]:
    method = getattr(module, "method", "rrattn")
    validate_method(method)

    key_states = repeat_kv(key_states, module.num_key_value_groups)
    value_states = repeat_kv(value_states, module.num_key_value_groups)
    module.rrattn_last_q_len = query_states.shape[2]
    module.rrattn_last_k_len = key_states.shape[2]

    if key_states.shape[2] == query_states.shape[2]:
        module.rrattn_seen_prefill = True
        if method == "rrattn":
            module.rrattn_last_attention_path = "prefill_rrattn"
            threshold = getattr(module, "threshold", 0.9)
            attn_output, sparse_ratio = rrattn_prefill(
                query_states,
                key_states,
                value_states,
                norm=1,
                stride=getattr(module, "stride", 8),
                threshold=threshold,
                use_triton=getattr(module, "use_triton", True),
                keep_sink=getattr(module, "keep_sink", True),
                keep_recent=getattr(module, "keep_recent", True),
                chunk_size=getattr(module, "chunk_size", 16384),
                rrattn_version=getattr(module, "rrattn_version", "v1"),
                layer_idx=getattr(module, "layer_idx", None),
                startend_row_indices=attn_mask_startend_row_indices,
                config=getattr(module, "rrattn_config", None),
            )
            module.sparse_ratio = sparse_ratio
        else:
            module.rrattn_last_attention_path = "prefill_full"
            attn_output = FA_Full_prefill(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                causal=causal,
            )
            module.sparse_ratio = 0.0
    else:
        module.rrattn_seen_decode = True
        module.rrattn_last_attention_path = "decode_dense"
        attn_output = dense_decode_attention(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            dropout=dropout,
            causal=causal,
            scaling=scaling,
            training=module.training,
        )

    attn_output = attn_output.reshape([attn_output.shape[0], attn_output.shape[1], -1]).contiguous()
    return attn_output, None


def dense_decode_attention(
    query_states: paddle.Tensor,
    key_states: paddle.Tensor,
    value_states: paddle.Tensor,
    attention_mask: Optional[paddle.Tensor] = None,
    attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
    dropout: float = 0.0,
    causal: bool = True,
    scaling: Optional[float] = None,
    training: bool = False,
) -> paddle.Tensor:
    q_len = query_states.shape[-2]
    key_len = key_states.shape[-2]
    scale = scaling if scaling is not None else 1.0 / math.sqrt(query_states.shape[-1])

    if attention_mask is None and attn_mask_startend_row_indices is not None:
        if attn_mask_startend_row_indices.ndim == 3:
            attn_mask_startend_row_indices = attn_mask_startend_row_indices.unsqueeze(-1)
        if attn_mask_startend_row_indices.shape[-1] == 1:
            causal = True
        elif attn_mask_startend_row_indices.shape[-1] == 4:
            causal = False
        attention_mask = _gen_from_sparse_attn_mask_indices(
            attn_mask_startend_row_indices,
            query_states.dtype,
            causal,
        )

    attn_weights = paddle.matmul(query_states, key_states.transpose(2, 3)) * scale
    if causal and q_len > 1:
        q_pos = paddle.arange(q_len, device=query_states.device) + key_len - q_len
        k_pos = paddle.arange(key_len, device=query_states.device)
        causal_mask = q_pos[:, None] < k_pos[None, :]
        attn_weights = paddle.where(
            causal_mask.unsqueeze(0).unsqueeze(0),
            paddle.full(attn_weights.shape, float("-inf"), dtype=attn_weights.dtype, device=attn_weights.device),
            attn_weights,
        )

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = F.softmax(attn_weights, axis=-1, dtype=paddle.float32).astype(query_states.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=training)
    return paddle.matmul(attn_weights, value_states).transpose(1, 2)
