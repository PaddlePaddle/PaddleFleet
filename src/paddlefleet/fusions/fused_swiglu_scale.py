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

import paddle


def fused_swiglu_scale_forward(x, scale):
    if paddle.is_compiled_with_cuda():
        from paddlefleet.ops import fused_swiglu_scale

        return fused_swiglu_scale(x, scale)
    else:
        raise NotImplementedError(
            "fused_swiglu_scale not implemented on this backend!"
        )


def fused_swiglu_scale_backward(x, scale, out_grad):
    if paddle.is_compiled_with_cuda():
        from paddlefleet.ops import fused_swiglu_scale_bwd

        return fused_swiglu_scale_bwd(x, scale, out_grad)
    else:
        raise NotImplementedError(
            "fused_swiglu_scale_backward not implemented on this backend!"
        )
