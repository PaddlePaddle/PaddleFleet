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

import numpy as np
import paddle

from paddlefleet.tensor_parallel.random import (
    checkpoint,
)


def test_checkpoint():
    def test_forward(*input):
        return input[0] + input[1]

    res_ref = paddle.ones(16) * 3
    res = checkpoint(test_forward, None, paddle.ones(16), paddle.ones(16) * 2)
    np.testing.assert_allclose(res_ref.numpy(), res.numpy())

    input1 = paddle.ones((4, 4))
    checkpoint(test_forward, True, input1, paddle.ones((4, 4)) * 2)
    np.testing.assert_allclose(
        paddle.ones(input1.shape).numpy(), input1.numpy()
    )
