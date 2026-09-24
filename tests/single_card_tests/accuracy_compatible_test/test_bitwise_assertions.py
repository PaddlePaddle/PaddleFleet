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

"""Protect the oracle against numerical-equality shortcuts."""

import numpy as np
import pytest

from tests.single_card_tests.accuracy_compatible_test._assertions import (
    assert_bitwise_equal,
)


def test_equal_noncontiguous_values_match():
    expected = np.arange(12, dtype=np.float32).reshape(3, 4).T
    assert_bitwise_equal(expected, expected.copy())


@pytest.mark.parametrize(
    "actual,expected",
    [
        (np.array([0.0], dtype=np.float32), np.array([-0.0], dtype=np.float32)),
        (
            np.array([1.0], dtype=np.float32),
            np.nextafter(np.array([1.0], dtype=np.float32), np.float32(2.0)),
        ),
        (np.array([1.0], dtype=np.float32), np.array([1.0], dtype=np.float64)),
        (
            np.array([[1.0]], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
        ),
    ],
)
def test_rejects_bit_dtype_and_shape_mismatches(actual, expected):
    with pytest.raises(AssertionError):
        assert_bitwise_equal(actual, expected)
