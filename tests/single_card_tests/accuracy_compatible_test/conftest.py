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

"""Native tensor regressions run on the same GPU device under pytest and CLI."""

import pytest


@pytest.fixture(autouse=True)
def native_tensor_device(request):
    # The dispatch and NumPy adapter modules do not import Paddle or run kernels.
    paddle = getattr(request.module, "paddle", None)
    if paddle is None:
        yield
        return
    previous = paddle.get_device()
    paddle.set_device("gpu:0")
    try:
        yield
    finally:
        paddle.set_device(previous)
