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

import paddle


def is_fp8_tensor(x):
    """
    Check if the input is a tuple of FP8 tensor and its corresponding scale.

    The scale is float32 for the regular blockwise recipe and int32 when
    UE8M0-packed (``using_ue8m0_scale=True`` in
    ``paddle.incubate.nn.functional.fp8_quant_blockwise``).
    """
    if not isinstance(x, tuple):
        return False
    if len(x) != 2:
        return False
    tensor, scale = x
    assert tensor.dtype != paddle.float8_e5m2, (
        "FP8 tensor should not be float8_e5m2 dtype, not supported yet."
    )
    return tensor.dtype == paddle.float8_e4m3fn and scale.dtype in (
        paddle.float32,
        paddle.int32,
    )
