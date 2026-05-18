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

paddle.enable_compat()
import pytest
import torch
from paddlefleet_ops import deep_gemm
from paddlefleet_ops.deep_gemm.testing import get_arch_major


def test_transform_sf_broadcasts_128_granularity_on_sm100() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for DeepGEMM layout transforms")
    if get_arch_major() != 10:
        pytest.skip("This path is only used on SM100")

    mn = 256
    k = 128
    sf = torch.tensor([[1.0], [2.0]], dtype=torch.float, device="cuda")

    print(
        "[TEST] calling transform_sf_into_required_layout with "
        "sf=float32, recipe=(1, 128, 128), is_sfa=False, arch=SM100",
        flush=True,
    )
    transformed = deep_gemm.transform_sf_into_required_layout(
        sf,
        mn,
        k,
        (1, 128, 128),
        None,
        False,
        False,
    )
    print("[TEST] returned from transform_sf_into_required_layout", flush=True)

    assert str(transformed.dtype) in ("paddle.int32", "torch.int32")
    assert transformed.shape == (mn, 1)
    transformed_cpu = transformed.cpu().numpy()
    assert (transformed_cpu[:128, 0] == 127).all()
    assert (transformed_cpu[128:, 0] == 128).all()
