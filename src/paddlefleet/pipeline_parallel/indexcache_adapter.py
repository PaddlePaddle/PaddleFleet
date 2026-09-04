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

import functools

import paddle

_PATCH_FLAG = "_indexcache_pipeline_adapter_registered"
_INDEXCACHE_STATE_KEY = "indexcache_state"
_PIPELINE_KEY_ATTR = "_paddlefleet_pipeline_key"
_PIPELINE_SHAPE_ATTR = "_paddlefleet_pipeline_shape"
_PIPELINE_DTYPE_ATTR = "_paddlefleet_pipeline_dtype"
_INDEXCACHE_PRODUCER_LAYER_ATTR = "_paddlefleet_indexcache_producer_layer"
_INDEXCACHE_CONFIG = None


def _debug_enabled() -> bool:
    return bool(
        getattr(_INDEXCACHE_CONFIG, "indexcache_train_debug", False)
    )


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


def _copy_indexcache_producer_layer(source, target) -> None:
    producer_layer = getattr(source, _INDEXCACHE_PRODUCER_LAYER_ATTR, None)
    if producer_layer is not None:
        setattr(target, _INDEXCACHE_PRODUCER_LAYER_ATTR, int(producer_layer))


def _annotate_indexcache_producer_layer(value):
    if not _debug_enabled() or not isinstance(value, (tuple, list)):
        return value

    from paddlefleet.transformer.indexcache_state import (
        INDEXCACHE_DISTILL_STATE_LEN,
        INDEXCACHE_DISTILL_STATE_PRODUCER_LAYER,
        INDEXCACHE_DISTILL_STATE_TOPK_PROBS,
    )

    if len(value) != INDEXCACHE_DISTILL_STATE_LEN:
        return value
    gradient_tensor = value[INDEXCACHE_DISTILL_STATE_TOPK_PROBS]
    producer_tensor = value[INDEXCACHE_DISTILL_STATE_PRODUCER_LAYER]
    if not isinstance(gradient_tensor, paddle.Tensor) or not isinstance(
        producer_tensor, paddle.Tensor
    ):
        return value
    if (
        getattr(gradient_tensor, _INDEXCACHE_PRODUCER_LAYER_ATTR, None)
        is not None
    ):
        return value
    if not producer_tensor._is_initialized():
        return value
    setattr(
        gradient_tensor,
        _INDEXCACHE_PRODUCER_LAYER_ATTR,
        int(producer_tensor.item()),
    )
    return value


def _mark_indexcache_state_stop_gradient(value):
    from paddlefleet.transformer.indexcache_state import (
        apply_stop_gradient_mask,
    )

    if value is None:
        return value
    if not isinstance(value, (tuple, list)):
        value = (value,)
    return _annotate_indexcache_producer_layer(apply_stop_gradient_mask(value))


def _describe_tensor_keys(tensors):
    if not isinstance(tensors, (tuple, list)):
        tensors = (tensors,)
    desc = []
    for tensor in tensors:
        if not isinstance(tensor, paddle.Tensor):
            desc.append(type(tensor).__name__)
            continue
        desc.append(
            {
                "key": _get_pipeline_key(tensor),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "stop_gradient": bool(tensor.stop_gradient),
            }
        )
    return desc


def _indexcache_producer_layer(tensors):
    if not _debug_enabled():
        return None
    for tensor in tensors:
        if not isinstance(tensor, paddle.Tensor):
            continue
        producer_layer = getattr(
            tensor,
            _INDEXCACHE_PRODUCER_LAYER_ATTR,
            None,
        )
        if producer_layer is not None:
            return int(producer_layer)
    return None


def _debug_state5_gradient(key, grad, source, producer_layer):
    if not _debug_enabled() or key != f"{_INDEXCACHE_STATE_KEY} 5":
        return
    from paddlefleet.transformer.indexcache_state import (
        format_indexcache_gradient_summary,
        summarize_indexcache_gradients,
    )

    summary = summarize_indexcache_gradients([("grad", grad)])["grad"]
    print(
        "[INDEXCACHE_PP_GRAD] boundary=state5_grad_present "
        f"source={source} "
        f"producer_layer={producer_layer} "
        f"{format_indexcache_gradient_summary('grad', summary)}",
        flush=True,
    )


def _detach_and_requires_grad(value):
    if value is None:
        return None
    if isinstance(value, dict):
        if _INDEXCACHE_STATE_KEY in value:
            _annotate_indexcache_producer_layer(value[_INDEXCACHE_STATE_KEY])
        return {
            key: _detach_and_requires_grad(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, (tuple, list)):
        detached = [_detach_and_requires_grad(item) for item in value]
        return tuple(detached) if isinstance(value, tuple) else detached
    if isinstance(value, paddle.Tensor):
        detached = value.detach()
        detached.stop_gradient = value.stop_gradient
        _copy_indexcache_producer_layer(value, detached)
        return detached
    return value


class _IndexCacheAliasClone(paddle.autograd.PyLayer):
    """Preserve the gradient edge without allocating an output-sized buffer."""

    @staticmethod
    def forward(ctx, value):
        return value.view(value.shape)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def _clone_and_clear_dataptr(value, fake_clone, clear_dataptr=False):
    if value is None:
        return None
    if isinstance(value, dict):
        if _INDEXCACHE_STATE_KEY in value:
            _annotate_indexcache_producer_layer(value[_INDEXCACHE_STATE_KEY])
        cloned = {
            key: _clone_and_clear_dataptr(item, fake_clone, clear_dataptr)
            for key, item in value.items()
            if item is not None
        }
        return cloned
    if isinstance(value, (tuple, list)):
        cloned = [
            _clone_and_clear_dataptr(item, fake_clone, clear_dataptr)
            for item in value
            if item is not None
        ]
        return tuple(cloned) if isinstance(value, tuple) else cloned
    if isinstance(value, paddle.Tensor):
        # Paddle's FakeClone creates an empty_like allocation before the
        # caller clears its dataptr. At 32K, that transient hidden-state-sized
        # allocation can exceed the remaining memory once IndexCache state is
        # present. A view keeps the same one-to-one gradient edge and lets the
        # cloned tensor drop its holder without allocating a second buffer.
        clone = _IndexCacheAliasClone if clear_dataptr else fake_clone
        cloned = clone.apply(value)
        cloned.stop_gradient = value.stop_gradient
        _copy_indexcache_producer_layer(value, cloned)
        if clear_dataptr:
            cloned._clear_dataptr()
        return cloned
    return value


def _convert_tensor_dict_to_tuple(output_tensor_dict):
    output_tensor = []
    for key, tensor in output_tensor_dict.items():
        if tensor is None:
            continue
        if isinstance(tensor, (list, tuple)):
            if key == _INDEXCACHE_STATE_KEY:
                tensor = _mark_indexcache_state_stop_gradient(tensor)
            for idx, item in enumerate(tensor):
                if item is None:
                    continue
                if not isinstance(item, paddle.Tensor):
                    raise TypeError(
                        "Pipeline dict tuple values must be paddle.Tensor "
                        f"or None, but {key}[{idx}] is {type(item).__name__}."
                    )
                item.key = f"{key} {idx}"
                _save_pipeline_metadata(item, item.key)
                output_tensor.append(item)
        else:
            if not isinstance(tensor, paddle.Tensor):
                raise TypeError(
                    "Pipeline dict values must be paddle.Tensor or tensor "
                    f"tuple/list, but {key} is {type(tensor).__name__}."
                )
            tensor.key = key
            _save_pipeline_metadata(tensor, key)
            output_tensor.append(tensor)

    output_tensor = tuple(output_tensor)
    if _debug_enabled() and _has_indexcache_key(output_tensor):
        print(
            "[INDEXCACHE_PP_FLOW] dict_to_tuple "
            f"tensors={_describe_tensor_keys(output_tensor)}",
            flush=True,
        )
    return output_tensor


def _convert_tensor_tuple_to_dict(input_tensor_tuple):
    input_tensor_dict = {}
    if not isinstance(input_tensor_tuple, tuple):
        input_tensor_tuple = (input_tensor_tuple,)
    for tensor in input_tensor_tuple:
        key = tensor.key
        _save_pipeline_metadata(tensor, key)
        if " " in key:
            real_key, suffix = key.rsplit(" ", 1)
            if suffix.isdigit():
                input_tensor_dict.setdefault(real_key, []).append(tensor)
            else:
                input_tensor_dict[key] = tensor
        else:
            input_tensor_dict[key] = tensor
        delattr(tensor, "key")
    if isinstance(input_tensor_dict.get(_INDEXCACHE_STATE_KEY), list):
        input_tensor_dict[_INDEXCACHE_STATE_KEY] = tuple(
            input_tensor_dict[_INDEXCACHE_STATE_KEY]
        )
    if _INDEXCACHE_STATE_KEY in input_tensor_dict:
        input_tensor_dict[_INDEXCACHE_STATE_KEY] = (
            _mark_indexcache_state_stop_gradient(
                input_tensor_dict[_INDEXCACHE_STATE_KEY]
            )
        )
    if _debug_enabled() and _INDEXCACHE_STATE_KEY in input_tensor_dict:
        state = input_tensor_dict[_INDEXCACHE_STATE_KEY]
        print(
            "[INDEXCACHE_PP_FLOW] tuple_to_dict "
            f"keys={list(input_tensor_dict.keys())} "
            f"state_type={type(state).__name__} "
            f"state={_describe_tensor_keys(state)}",
            flush=True,
        )
    return input_tensor_dict


def _tuple_to_dict_helper(input_tensor):
    use_dict = False
    if isinstance(input_tensor, tuple):
        use_dict = len(input_tensor) > 0 and hasattr(input_tensor[0], "key")
    else:
        use_dict = hasattr(input_tensor, "key")
    if use_dict:
        input_tensor = _convert_tensor_tuple_to_dict(input_tensor)
    return input_tensor, use_dict


def _dict_to_tuple_helper(output_tensor):
    if isinstance(output_tensor, dict):
        return _convert_tensor_dict_to_tuple(output_tensor)
    return output_tensor


def _collect_input_gradients(inputs):
    gradients = []
    zero_filled_keys = []
    producer_layer = _indexcache_producer_layer(inputs)
    for tensor in inputs:
        if not isinstance(tensor, paddle.Tensor) or tensor.stop_gradient:
            continue
        grad = tensor.grad
        key = _get_pipeline_key(tensor)
        _debug_state5_gradient(key, grad, "schedule_node", producer_layer)
        if grad is None:
            if not _is_indexcache_key(key):
                raise RuntimeError(
                    "Pipeline input is missing a gradient outside IndexCache "
                    f"state: key={key!r}, shape={list(tensor.shape)}, "
                    f"dtype={tensor.dtype}."
                )
            grad = paddle.zeros_like(tensor)
            grad.stop_gradient = False
            zero_filled_keys.append(key)
        gradients.append(grad)
    if _debug_enabled() and zero_filled_keys:
        print(
            f"[INDEXCACHE_PP_GRAD] zero_filled_keys={zero_filled_keys}",
            flush=True,
        )
    return tuple(gradients)


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
    producer_layer = _indexcache_producer_layer(inputs)
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
        _debug_state5_gradient(key, grad, "pipeline", producer_layer)
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

    if _debug_enabled() and zero_filled_keys:
        print(
            "[INDEXCACHE_PP_GRAD] boundary=pipeline zero_filled_keys="
            f"{zero_filled_keys}",
            flush=True,
        )
    return tuple(gradients) if is_tuple_input else gradients[0]


def _wrap_pipeline_backward_step(original_backward_step):
    @functools.wraps(original_backward_step)
    def backward_step(self, input_tensor, *args, **kwargs):
        input_tensor_grad = original_backward_step(
            self,
            input_tensor,
            *args,
            **kwargs,
        )
        return _normalize_pipeline_input_gradients(
            input_tensor,
            input_tensor_grad,
        )

    return backward_step


def _schedule_node_backward(self, output_grad=None, scaler=None):
    if output_grad is None:
        if isinstance(self.outputs, (tuple, list)):
            assert len(self.outputs) == 1
            outputs = self.outputs[0]
        else:
            outputs = self.outputs
        assert isinstance(outputs, paddle.Tensor)
        if scaler is not None:
            paddle.autograd.backward(scaler.scale(outputs))
        else:
            paddle.autograd.backward(outputs)
    else:
        is_output_grad_tuple = isinstance(output_grad, tuple)
        if not isinstance(output_grad, (tuple, list)):
            is_output_grad_tuple = True
            output_grad = (output_grad,)

        outputs = _dict_to_tuple_helper(self.outputs)
        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs,)
        outputs = [
            tensor
            for tensor in outputs
            if isinstance(tensor, paddle.Tensor) and not tensor.stop_gradient
        ]

        output_grad = [grad for grad in output_grad if grad is not None]
        output_grad = (
            tuple(output_grad) if is_output_grad_tuple else list(output_grad)
        )

        assert len(outputs) == len(output_grad), (
            f"{len(outputs)} of {type(outputs[0])} vs "
            f"{len(output_grad)} of {type(output_grad[0])}"
        )
        paddle.autograd.backward(outputs, output_grad)

    inputs = _dict_to_tuple_helper(self.inputs)
    if not isinstance(inputs, (tuple, list)):
        inputs = (inputs,)
    grad = _collect_input_gradients(inputs)
    self._reset_states()
    return grad


def register_indexcache_pipeline_adapter(config) -> bool:
    """Register Paddle Pipeline compatibility only for IndexCache runs.

    The adapter is process-global because Paddle exposes the relevant Pipeline
    hooks as module functions and class methods. Registration is nevertheless
    explicit, conditional, and idempotent: a config with an empty
    ``index_topk_pattern`` is a no-op and a second registration does not wrap
    methods again. Pipeline diagnostics read ``indexcache_train_debug`` from
    this normalized TransformerConfig instead of process environment state.

    Returns:
        ``True`` only when this call installs the adapter.
    """
    index_topk_pattern = getattr(config, "index_topk_pattern", None)
    if not index_topk_pattern:
        return False

    global _INDEXCACHE_CONFIG
    _INDEXCACHE_CONFIG = config

    import paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils as fbo
    import paddle.distributed.fleet.meta_parallel.pp_utils.utils as pp_utils
    from paddle.distributed.fleet import meta_parallel
    from paddle.distributed.fleet.meta_parallel import pipeline_parallel

    if getattr(pp_utils, _PATCH_FLAG, False):
        return False

    def clone_and_clear_dataptr(outputs, clear_dataptr=False):
        return _clone_and_clear_dataptr(
            outputs,
            fbo.FakeClone,
            clear_dataptr=clear_dataptr,
        )

    pp_utils.convert_tensor_dict_to_tuple = _convert_tensor_dict_to_tuple
    pp_utils.convert_tensor_tuple_to_dict = _convert_tensor_tuple_to_dict
    pp_utils.tuple_to_dict_helper = _tuple_to_dict_helper
    pp_utils.dict_to_tuple_helper = _dict_to_tuple_helper
    setattr(pp_utils, _PATCH_FLAG, True)

    meta_parallel.dict_to_tuple_helper = _dict_to_tuple_helper
    meta_parallel.tuple_to_dict_helper = _tuple_to_dict_helper

    fbo.detach_and_requires_grad = _detach_and_requires_grad
    fbo.clone_and_clear_dataptr = clone_and_clear_dataptr
    fbo.ScheduleNode.backward = _schedule_node_backward
    setattr(fbo, _PATCH_FLAG, True)

    pipeline_parallel.PipelineParallel._backward_step = (
        _wrap_pipeline_backward_step(
            pipeline_parallel.PipelineParallel._backward_step
        )
    )
    setattr(pipeline_parallel, _PATCH_FLAG, True)
    return True
