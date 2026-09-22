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

from __future__ import annotations

import logging

import paddle
from paddle.distributed.fleet.meta_parallel import (
    PipelineLayer,
    PipelineParallel,
)

from paddlefleet.transformer.indexcache_state import apply_stop_gradient_mask

logger = logging.getLogger(__name__)
_INDEXCACHE_STATE_KEY = "indexcache_state"
_PIPELINE_KEY_ATTR = "_paddlefleet_pipeline_key"
_PIPELINE_SHAPE_ATTR = "_paddlefleet_pipeline_shape"
_PIPELINE_DTYPE_ATTR = "_paddlefleet_pipeline_dtype"


def _is_indexcache_key(key) -> bool:
    return isinstance(key, str) and key.startswith(_INDEXCACHE_STATE_KEY)


def _get_pipeline_key(tensor):
    key = getattr(tensor, "key", None)
    if key is None:
        key = getattr(tensor, _PIPELINE_KEY_ATTR, None)
    return key


def _save_pipeline_metadata(tensor, key):
    setattr(tensor, _PIPELINE_KEY_ATTR, key)
    setattr(tensor, _PIPELINE_SHAPE_ATTR, tuple(tensor.shape))
    setattr(tensor, _PIPELINE_DTYPE_ATTR, tensor.dtype)


def _has_indexcache_key(tensors) -> bool:
    if not isinstance(tensors, (tuple, list)):
        tensors = (tensors,)
    return any(
        _is_indexcache_key(_get_pipeline_key(tensor)) for tensor in tensors
    )


def _zeros_from_tensor_metadata(tensor):
    shape = getattr(tensor, _PIPELINE_SHAPE_ATTR, None)
    dtype = getattr(tensor, _PIPELINE_DTYPE_ATTR, None)
    if shape is None or dtype is None:
        raise RuntimeError(
            "IndexCache pipeline tensor lacks preserved shape/dtype metadata: "
            f"key={_get_pipeline_key(tensor)!r}."
        )
    grad = paddle.zeros(
        shape=list(shape),
        dtype=dtype,
    )
    grad.stop_gradient = False
    return grad


def _normalize_pipeline_input_gradients(input_tensor, input_tensor_grad):
    if input_tensor is None:
        return input_tensor_grad

    is_tuple_input = isinstance(input_tensor, tuple)
    inputs = input_tensor if is_tuple_input else (input_tensor,)
    if not _has_indexcache_key(inputs):
        return input_tensor_grad

    differentiable_inputs = [
        tensor
        for tensor in inputs
        if isinstance(tensor, paddle.Tensor) and not tensor.stop_gradient
    ]
    if is_tuple_input:
        if not isinstance(input_tensor_grad, (tuple, list)):
            raise RuntimeError(
                "IndexCache pipeline input gradients must preserve tuple "
                f"structure, but got {type(input_tensor_grad).__name__}."
            )
        gradients = list(input_tensor_grad)
    else:
        gradients = [input_tensor_grad]

    if len(gradients) != len(differentiable_inputs):
        raise RuntimeError(
            "IndexCache pipeline input gradient arity mismatch: "
            f"inputs={len(differentiable_inputs)}, gradients={len(gradients)}."
        )

    zero_filled_keys = []
    for idx, (tensor, grad) in enumerate(zip(differentiable_inputs, gradients)):
        key = _get_pipeline_key(tensor)
        if grad is None:
            if not _is_indexcache_key(key):
                raise RuntimeError(
                    "Pipeline input is missing a gradient outside IndexCache "
                    f"state: key={key!r}, shape={list(tensor.shape)}, "
                    f"dtype={tensor.dtype}."
                )
            grad = _zeros_from_tensor_metadata(tensor)
            gradients[idx] = grad
            zero_filled_keys.append(key)
        elif not isinstance(grad, paddle.Tensor):
            raise TypeError(
                "Pipeline input gradients must be paddle.Tensor or None, "
                f"but key={key!r} has {type(grad).__name__}."
            )

    if logger.isEnabledFor(logging.DEBUG) and zero_filled_keys:
        logger.debug(
            "[INDEXCACHE_PP_GRAD] boundary=pipeline zero_filled_keys="
            f"{zero_filled_keys}"
        )
    return tuple(gradients) if is_tuple_input else gradients[0]


def prepare_indexcache_pipeline_boundary(value):
    """Normalize a model-local boundary without changing Paddle's tuple codec.

    Paddle removes tensor.key while decoding a received tuple. Recover the
    key from the decoded dictionary and retain metadata on those same tensor
    objects, so the scheduler's input buffer can still materialize zero grads
    after Paddle releases tensor storage.
    """
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if item is None:
            continue
        if key == _INDEXCACHE_STATE_KEY:
            item = apply_stop_gradient_mask(item)
            if item is None:
                continue
        if isinstance(item, (tuple, list)):
            for index, tensor in enumerate(item):
                if not isinstance(tensor, paddle.Tensor):
                    raise TypeError(
                        f"Pipeline state {key}[{index}] must be a Tensor"
                    )
                _save_pipeline_metadata(tensor, f"{key} {index}")
        elif isinstance(item, paddle.Tensor):
            _save_pipeline_metadata(item, key)
        else:
            raise TypeError(
                f"Pipeline value {key} must be a Tensor or tensor sequence"
            )
        result[key] = item
    return result


class IndexCachePipelineLayer(PipelineLayer):
    """PipelineLayer with explicit, model-local IndexCache state boundaries."""

    def forward(self, input, chunk_id=None):
        if not getattr(
            getattr(self, "config", None), "indexcache_topk_pattern", None
        ):
            return super().forward(input, chunk_id=chunk_id)
        if self._num_stages > 1 and not getattr(
            self, "_indexcache_instance_wrapper", False
        ):
            raise RuntimeError(
                "IndexCache PP must be wrapped by paddlefleet.distributed.model.distributed_model"
            )
        input = prepare_indexcache_pipeline_boundary(input)
        output = super().forward(input, chunk_id=chunk_id)
        return prepare_indexcache_pipeline_boundary(output)


class IndexCachePipelineParallel(PipelineParallel):
    """Ordinary 1F1B scheduler with state-specific zero-gradient handling."""

    def _backward_step(self, input_tensor, *args, **kwargs):
        gradients = super()._backward_step(input_tensor, *args, **kwargs)
        return _normalize_pipeline_input_gradients(input_tensor, gradients)
