# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import contextlib
import sys
from typing import TYPE_CHECKING

from ..utils.lazy_import import _LazyModule

_FLEET_UTILS_NAMES = (
    "WrappedTensor",
    "GlobalMemoryBuffer",
    "ensure_divisibility",
    "divide",
    "init_method_normal",
    "scaled_init_method_normal",
    "get_magic_init_method",
    "truncated_init_method_normal",
    "get_pg_size",
    "get_pg_rank",
    "log_single_rank",
    "get_tensor_model_parallel_group_if_none",
    "prepare_input_tensors_for_wgrad_compute",
    "get_paddle_version",
    "is_paddle_min_version",
    "get_batch_on_this_cp_rank",
    "nvtx_range_push",
    "nvtx_range_pop",
    "nvtx_decorator",
    "_nvtx_decorator_get_func_path",
    "_nvtx_range_messages",
    "_kernel_make_viewless_tensor",
    "get_attr_wrapped_model",
    "get_model_type",
    "get_model_xattn",
    "get_model_config",
    "MakeViewlessTensor",
    "make_viewless_tensor",
    "deprecate_inference_params",
)

import_structure = {
    "_fleet_utils": list(_FLEET_UTILS_NAMES),
    "nested": [
        "nested_reduce_tensor",
        "nested_empty_tensor",
        "nested_broadcast_tensor",
        "nested_broadcast_tensor_with_empty",
        "nested_copy",
        "nested_copy_place",
        "flatten_list",
        "TensorHolder",
    ],
    "import_utils": [
        "is_torch_available",
        "is_paddlenlp_ops_available",
        "auto_dynamic_graph_pybind",
        "is_paddle_cuda_available",
        "is_package_available",
        "is_tiktoken_available",
        "uninstall_package",
        "import_module",
        "_is_package_available",
        "is_sentencepiece_available",
        "is_paddle_available",
        "is_psutil_available",
        "is_protobuf_available",
        "is_tokenizers_available",
        "is_fast_tokenizer_available",
        "install_package",
        "is_g2p_en_available",
        "is_datasets_available",
        "is_transformers_available",
        "is_paddlefleet_available",
        "dynamic_graph_pybind_context",
        "custom_import",
        "direct_paddlefleet_import",
    ],
    "initializer": ["to"],
    "infohub": ["infohub", "InfoHub"],
    "memory_utils": ["empty_device_cache"],
    "moe_hybrid_parallel_optimizer": ["MoEHybridParallelOptimizer"],
    "paddle_patch": ["enhance_set_value", "new_repr", "_numel", "_numpy", "enhance_init", "enhance_to_tensor"],
    "serialization": [
        "seek_by_string",
        "load_torch_inner",
        "SerializationError",
        "_element_size",
        "_rebuild_tensor_stage",
        "_maybe_decode_ascii",
        "load_torch",
        "_rebuild_parameter",
        "_rebuild_parameter_with_state",
        "UnpicklerWrapperStage",
        "read_prefix_key",
        "_storage_type_to_dtype_to_map",
        "SafeUnpickler",
        "dumpy",
        "StorageType",
    ],
    "batch_sampler": ["DistributedBatchSampler"],
    "optimizer": ["AdamWMini", "AdamWCustom"],
    "env": ["CONFIG_NAME", "GENERATION_CONFIG_NAME", "LEGACY_CONFIG_NAME"],
    "log": ["logger"],
    "masking_utils": [
        "_gen_from_sparse_attn_mask_indices",
        "masked_fill",
        "is_casual_mask",
        "_make_causal_mask",
        "_expand_2d_mask",
        "build_alibi_tensor",
        "get_use_casual_mask",
        "get_triangle_upper_mask",
    ],
    "tools": [
        "device_guard",
    ],
    "downloader": ["get_weights_path_from_url"],
    "type_validators": [
        "positive_any_number",
        "positive_int",
        "padding_validator",
        "truncation_validator",
        "image_size_validator",
        "device_validator",
        "resampling_validator",
        "video_metadata_validator",
        "tensor_type_validator",
    ],
}


@contextlib.contextmanager
def device_guard(device="cpu", dev_id=0):
    import paddle

    origin_device = paddle.device.get_device()
    if device == "cpu":
        paddle.set_device(device)
    elif device in ["gpu", "xpu", "npu"]:
        paddle.set_device("{}:{}".format(device, dev_id))
    try:
        yield
    finally:
        paddle.set_device(origin_device)


if TYPE_CHECKING:
    import paddle

    from ._fleet_utils import *
    from .batch_sampler import *
    from .env import CONFIG_NAME, GENERATION_CONFIG_NAME, LEGACY_CONFIG_NAME
    from .import_utils import *
    from .infohub import infohub
    from .initializer import to
    from .log import logger
    from .memory_utils import empty_device_cache
    from .moe_hybrid_parallel_optimizer import MoEHybridParallelOptimizer

    try:
        from .optimizer import *
    except:
        logger.info("Not support custom optimizer")

    from type_validators import *

    from .serialization import load_torch

    # hack impl for EagerParamBase to function
    # https://github.com/PaddlePaddle/Paddle/blob/fa44ea5cf2988cd28605aedfb5f2002a63018df7/python/paddle/nn/layer/layers.py#L2077
    paddle.framework.io.EagerParamBase.to = to
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
        extra_objects={
            "__all__": [*_FLEET_UTILS_NAMES, "device_guard"],
            "device_guard": device_guard,
        },
    )
