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

import os

import paddle


_PATCH_FLAG = "_indexcache_pp_runtime_patch_applied"
_INDEXCACHE_STATE_KEY = "indexcache_state"


def _debug_enabled() -> bool:
    return os.environ.get("INDEXCACHE_TRAIN_DEBUG", "0") == "1"


def _is_indexcache_key(key) -> bool:
    return isinstance(key, str) and key.startswith(_INDEXCACHE_STATE_KEY)


def _has_indexcache_key(tensors) -> bool:
    if not isinstance(tensors, (tuple, list)):
        tensors = (tensors,)
    return any(_is_indexcache_key(getattr(tensor, "key", None)) for tensor in tensors)


def _mark_indexcache_state_stop_gradient(value):
    from paddlefleet.transformer.indexcache_state import (
        apply_stop_gradient_mask,
    )

    if value is None:
        return value
    if not isinstance(value, (tuple, list)):
        value = (value,)
    return apply_stop_gradient_mask(value)


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
                "key": getattr(tensor, "key", None),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "stop_gradient": bool(tensor.stop_gradient),
            }
        )
    return desc


def _detach_and_requires_grad(value):
    if value is None:
        return None
    if isinstance(value, dict):
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
        return detached
    return value


def _clone_and_clear_dataptr(value, fake_clone, clear_dataptr=False):
    if value is None:
        return None
    if isinstance(value, dict):
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
        cloned = fake_clone.apply(value)
        cloned.stop_gradient = value.stop_gradient
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
                output_tensor.append(item)
        else:
            if not isinstance(tensor, paddle.Tensor):
                raise TypeError(
                    "Pipeline dict values must be paddle.Tensor or tensor "
                    f"tuple/list, but {key} is {type(tensor).__name__}."
                )
            tensor.key = key
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
    grad = tuple(
        tensor.grad
        for tensor in inputs
        if isinstance(tensor, paddle.Tensor) and not tensor.stop_gradient
    )
    self._reset_states()
    return grad


def apply_indexcache_pp_runtime_patch() -> None:
    import paddle.distributed.fleet.meta_parallel as meta_parallel
    import paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils as fbo
    import paddle.distributed.fleet.meta_parallel.pp_utils.utils as pp_utils

    if getattr(pp_utils, _PATCH_FLAG, False):
        return

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
