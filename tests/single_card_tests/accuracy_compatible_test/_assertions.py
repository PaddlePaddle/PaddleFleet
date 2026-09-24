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

"""Assertions for storage equality, including the sign of zero."""

import numpy as np


def assert_bitwise_equal(actual, expected):
    """Compare matching tensor/array metadata and unconverted storage bytes."""
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
    if hasattr(actual, "numpy"):
        actual = actual.numpy()
        expected = expected.numpy()
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
    np.testing.assert_array_equal(
        np.ascontiguousarray(actual).view(np.uint8),
        np.ascontiguousarray(expected).view(np.uint8),
    )
