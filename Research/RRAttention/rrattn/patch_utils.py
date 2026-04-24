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

import types
from collections.abc import Iterable

import paddle
from paddle.nn.functional.flash_attention import flash_attention
from paddleformers.nn.attention.eager_attention import repeat_kv

from .flexprefill import flex_prefill
from .full_prefill import flash_full_prefill
from .rrattention import rrattn_prefill
from .xattention import xattn_prefill

SUPPORTED_METHODS = ("xattn", "rrattn", "flex", "full")
PATCHED_ATTN_IMPLEMENTATION = "sdpa"


def validate_method(method: str):
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported method={method!r}; supported methods are: xattn, rrattn, flex, full"
        )


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


def configure_model_attn_implementation(
    model, attn_implementation: str = PATCHED_ATTN_IMPLEMENTATION
):
    modules = [model]
    if hasattr(model, "model"):
        modules.append(model.model)

    for module in modules:
        if module is None or not hasattr(module, "config"):
            continue
        config = module.config
        if not hasattr(config, "_rrattn_original_attn_implementation"):
            config._rrattn_original_attn_implementation = getattr(
                config, "_attn_implementation", None
            )
        config._attn_implementation = attn_implementation

    for _, attn in (
        model.named_sublayers() if hasattr(model, "named_sublayers") else []
    ):
        if not hasattr(attn, "config"):
            continue
        config = attn.config
        if not hasattr(config, "_rrattn_original_attn_implementation"):
            config._rrattn_original_attn_implementation = getattr(
                config, "_attn_implementation", None
            )
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
            if attention_cls is not None and not isinstance(
                attn, attention_cls
            ):
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
    attention_mask: paddle.Tensor | None = None,
    attn_mask_startend_row_indices: paddle.Tensor | None = None,
    dropout: float = 0.0,
    causal: bool = True,
    scaling: float | None = None,
) -> tuple[paddle.Tensor, paddle.Tensor | None]:
    method = getattr(module, "method", "rrattn")
    validate_method(method)
    should_repeat_kv = method != "rrattn"
    if should_repeat_kv:
        key_states = repeat_kv(key_states, module.num_key_value_groups)
        value_states = repeat_kv(value_states, module.num_key_value_groups)

    if key_states.shape[2] == query_states.shape[2]:
        if method == "xattn":
            threshold = getattr(module, "threshold", 0.9)
            attn_output, sparse_ratio = xattn_prefill(
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
                layer_idx=getattr(module, "layer_idx", None),
            )
            module.sparse_ratio = sparse_ratio
        elif method == "rrattn":
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
                layer_idx=getattr(module, "layer_idx", None),
                startend_row_indices=attn_mask_startend_row_indices,
                config=getattr(module, "rrattn_config", None),
            )
            module.sparse_ratio = sparse_ratio
        elif method == "flex":
            result = flex_prefill(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                gamma=getattr(module, "threshold", 0.9),
                tau=getattr(module, "tau", 0.1),
                min_budget=getattr(module, "min_budget", None),
                max_budget=getattr(module, "max_budget", None),
                gqa_interleave=getattr(module, "gqa_interleave", False),
                softmax_scale=scaling,
                block_size=getattr(module, "block_size", 128),
            )
            if isinstance(result, tuple):
                attn_output, sparse_ratio = result
            else:
                attn_output = result
                sparse_ratio = 0.0
            module.sparse_ratio = sparse_ratio
        else:
            attn_output = flash_full_prefill(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                causal=causal,
            )
            module.sparse_ratio = 0.0
    else:
        q_len = query_states.shape[-2]
        if causal and q_len != 1:
            raise NotImplementedError(
                "flash decode attention only supports q_len=1 for cached causal decode"
            )
        # use full attention for decoding
        attn_output, _ = flash_attention(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            dropout=dropout,
            causal=False,
            softmax_scale=scaling,
        )

    attn_output = attn_output.reshape(
        [attn_output.shape[0], attn_output.shape[1], -1]
    ).contiguous()
    return attn_output, None
