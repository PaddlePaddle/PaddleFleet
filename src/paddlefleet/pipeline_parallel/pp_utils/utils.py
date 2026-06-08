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

import os

import paddle
from paddle.distributed.fleet.utils.log_util import get_sync_logger

PADDLE_TO_NUMBER = {
    paddle.float16: 0,
    paddle.float32: 1,
    paddle.float64: 2,
    paddle.int32: 3,
    paddle.int64: 4,
    paddle.bfloat16: 5,
    paddle.bool: 6,
}

NUMBER_TO_DTYPE = {
    0: "float16",
    1: "float32",
    2: "float64",
    3: "int32",
    4: "int64",
    5: "bfloat16",
    6: "bool",
}


def paddle_2_number(dtype):
    assert dtype in PADDLE_TO_NUMBER.keys()
    return PADDLE_TO_NUMBER[dtype]


def number_2_dtype(number):
    assert number in NUMBER_TO_DTYPE.keys()
    return NUMBER_TO_DTYPE[number]


def profile_pipeline_details(msg):
    GB = 1024.0 * 1024.0 * 1024.0
    if paddle.base.core.is_compiled_with_cuda():
        memory_allocated_size = paddle.device.cuda.memory_allocated() / GB
        memory_reserved_size = paddle.device.cuda.memory_reserved() / GB
    else:
        memory_allocated_size, memory_reserved_size = 0, 0
    get_sync_logger().info(
        f"{msg}: memory_allocated_size={memory_allocated_size:.2f}, memory_reserved_size={memory_reserved_size:.2f}"
    )


def _parse_int_set(env_name):
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return None
    values = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.add(int(item))
        except ValueError:
            return None
    return values


def _current_rank():
    try:
        if paddle.distributed.is_initialized():
            return paddle.distributed.get_rank()
    except Exception:
        pass
    try:
        return int(os.environ.get("PADDLE_TRAINER_ID", os.environ.get("RANK", "0")))
    except ValueError:
        return 0


def _current_local_rank():
    for name in ("PADDLE_RANK_IN_NODE", "FLAGS_selected_gpus", "CUDA_VISIBLE_DEVICES"):
        raw = os.environ.get(name, "").split(",")[0].strip()
        if not raw:
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return None


def _current_step():
    raw = os.environ.get("TRAINER_GLOBAL_STEP", "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _mem_probe_enabled(step=None, layer=None):
    if os.environ.get("SONIC_MOE_MEM_PROBE", "0").lower() not in {"1", "true", "yes", "on"}:
        return False
    ranks = _parse_int_set("SONIC_MOE_MEM_PROBE_RANKS")
    if ranks is not None and _current_rank() not in ranks:
        return False
    local_ranks = _parse_int_set("SONIC_MOE_MEM_PROBE_LOCAL_RANKS")
    if local_ranks is not None and _current_local_rank() not in local_ranks:
        return False
    steps = _parse_int_set("SONIC_MOE_MEM_PROBE_STEPS")
    if steps is not None and (step is None or int(step) not in steps):
        return False
    layers = _parse_int_set("SONIC_MOE_MEM_PROBE_LAYERS")
    if layers is not None and layer is not None and int(layer) not in layers:
        return False
    return True


def _try_record_memory(mem_dict, key, getter, unit):
    try:
        mem_dict[key] = getter() / unit
    except Exception:
        pass


def _tensor_nbytes(tensor):
    if tensor is None:
        return 0
    try:
        size = getattr(tensor, "size", None)
        if callable(size):
            size = size()
        if size is None:
            shape = getattr(tensor, "shape", None)
            if shape is None:
                return 0
            size = 1
            for dim in shape:
                size *= int(dim)
        itemsize = getattr(tensor, "itemsize", None)
        if itemsize is None:
            dtype = str(getattr(tensor, "dtype", ""))
            itemsize = 1 if any(x in dtype for x in ("float8", "int8", "uint8")) else 2 if any(x in dtype for x in ("float16", "bfloat16")) else 4
        return int(size) * int(itemsize)
    except Exception:
        return 0


def _collect_tensor_summaries(name, value, unit, summaries, max_items):
    if value is None or len(summaries) >= max_items:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _collect_tensor_summaries(f"{name}.{key}", item, unit, summaries, max_items)
        return
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _collect_tensor_summaries(f"{name}.{idx}", item, unit, summaries, max_items)
        return
    nbytes = _tensor_nbytes(value)
    if nbytes <= 0:
        return
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    summaries.append(f"{name}={nbytes / unit:.1f}MB(shape={list(shape) if shape is not None else '?'},dtype={dtype})")


def sonic_moe_memory_probe(msg, step=None, layer=None, logger=None, extra=None, tensors=None):
    if step is None:
        step = _current_step()
    if not _mem_probe_enabled(step=step, layer=layer):
        return
    unit = 1024.0 * 1024.0
    unit_name = "MB"
    mem_dict = {}
    if paddle.base.core.is_compiled_with_cuda():
        _try_record_memory(mem_dict, "memory_allocated", paddle.device.cuda.memory_allocated, unit)
        _try_record_memory(mem_dict, "memory_reserved", paddle.device.cuda.memory_reserved, unit)
        _try_record_memory(mem_dict, "max_memory_allocated", paddle.device.cuda.max_memory_allocated, unit)
        _try_record_memory(mem_dict, "max_memory_reserved", paddle.device.cuda.max_memory_reserved, unit)
        if hasattr(paddle.device.cuda, "max_pinned_memory_allocated"):
            _try_record_memory(mem_dict, "pinned_memory_allocated", paddle.device.cuda.pinned_memory_allocated, unit)
            _try_record_memory(mem_dict, "pinned_memory_reserved", paddle.device.cuda.pinned_memory_reserved, unit)
            _try_record_memory(mem_dict, "pinned_max_memory_allocated", paddle.device.cuda.max_pinned_memory_allocated, unit)
            _try_record_memory(mem_dict, "pinned_max_memory_reserved", paddle.device.cuda.max_pinned_memory_reserved, unit)
    if hasattr(paddle.device, "cpu"):
        if hasattr(paddle.device.cpu, "memory_allocated"):
            _try_record_memory(mem_dict, "cpu_memory_allocated", paddle.device.cpu.memory_allocated, unit)
        if hasattr(paddle.device.cpu, "memory_reserved"):
            _try_record_memory(mem_dict, "cpu_memory_reserved", paddle.device.cpu.memory_reserved, unit)
        if hasattr(paddle.device.cpu, "max_memory_allocated"):
            _try_record_memory(mem_dict, "cpu_max_memory_allocated", paddle.device.cpu.max_memory_allocated, unit)
        if hasattr(paddle.device.cpu, "max_memory_reserved"):
            _try_record_memory(mem_dict, "cpu_max_memory_reserved", paddle.device.cpu.max_memory_reserved, unit)
    fields = [f"rank={_current_rank()}"]
    local_rank = _current_local_rank()
    if local_rank is not None:
        fields.append(f"local_rank={local_rank}")
    if step is not None:
        fields.append(f"step={int(step)}")
    if layer is not None:
        fields.append(f"layer={int(layer)}")
    fields.extend(f"{key}: {value:.1f}{unit_name}" for key, value in mem_dict.items())
    if extra:
        fields.extend(f"{key}={value}" for key, value in extra.items())
    if tensors:
        tensor_summaries = []
        for key, value in tensors.items():
            _collect_tensor_summaries(key, value, unit, tensor_summaries, max_items=16)
        fields.extend(tensor_summaries)
    log_msg = f"[SonicMoE memory] {msg}: " + ", ".join(fields)
    (logger or get_sync_logger()).info(log_msg)


def sonic_moe_tensor_memory_probe(msg, tensors=None, step=None, layer=None, logger=None, extra=None):
    sonic_moe_memory_probe(msg, step=step, layer=layer, logger=logger, extra=extra, tensors=tensors or {})


def tuple_to_dict_helper(input_tensor):
    # recv tuple -> fwd input dict
    use_dict = False
    if isinstance(input_tensor, tuple):
        use_dict = hasattr(input_tensor[0], "key")
    else:  # single tensor
        use_dict = hasattr(input_tensor, "key")
    if use_dict:
        input_tensor = convert_tensor_tuple_to_dict(input_tensor)
    return input_tensor, use_dict


def dict_to_tuple_helper(output_tensor):
    if isinstance(output_tensor, dict):
        output_tensor_tuple = convert_tensor_dict_to_tuple(
            output_tensor_dict=output_tensor
        )
    else:  # single tensor or tensor tuple
        output_tensor_tuple = output_tensor
    return output_tensor_tuple


def convert_tensor_dict_to_tuple(output_tensor_dict):
    output_tensor = []
    for key, tensor in output_tensor_dict.items():
        if isinstance(tensor, (list, tuple)):
            for idx, t in enumerate(tensor):
                t.key = key + " " + str(idx)
                output_tensor.append(t)
        else:  # single tensor
            tensor.key = key
            output_tensor.append(tensor)

    return tuple(output_tensor)


def convert_tensor_tuple_to_dict(input_tensor_tuple):
    input_tensor_dict = {}
    for tensor in input_tensor_tuple:
        key = tensor.key
        if " " in key:
            real_key, _ = key.split(" ")
            if real_key in input_tensor_dict.keys():
                input_tensor_dict[real_key].append(tensor)
            else:
                input_tensor_dict[real_key] = [tensor]
        else:
            input_tensor_dict[key] = tensor
        delattr(tensor, "key")
    return input_tensor_dict
