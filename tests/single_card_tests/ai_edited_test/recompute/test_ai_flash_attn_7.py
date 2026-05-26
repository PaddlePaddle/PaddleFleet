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
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import unittest

import paddle

from paddlefleet.refined_recompute.flash_attn import (
    flashattn_auto_cast,
)


class TestGetFAVersionXPU(unittest.TestCase):
    """Tests for get_fa_version - removed as function no longer exists in this module."""


class TestGetFAVersionGPU(unittest.TestCase):
    """Tests for get_fa_version - removed as function no longer exists in this module."""


class TestGetFAVersionDeterministic(unittest.TestCase):
    """Tests for get_fa_version - removed as function no longer exists in this module."""


class TestGetFAVersionNonDeterministic(unittest.TestCase):
    """Tests for get_fa_version - removed as function no longer exists in this module."""


class TestFlashattnAutoCastBasic(unittest.TestCase):
    """Tests for flashattn_auto_cast basic behavior."""

    def test_all_same_dtype_bfloat16(self):
        """Test no-op when all inputs are already bfloat16."""
        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        q_out, k_out, v_out = flashattn_auto_cast(q, k, v)
        self.assertIs(q_out, q)
        self.assertIs(k_out, k)
        self.assertIs(v_out, v)

    def test_all_same_dtype_float16(self):
        """Test no-op when all inputs are already float16."""
        q = paddle.randn([2, 4, 8], dtype=paddle.float16)
        k = paddle.randn([2, 4, 8], dtype=paddle.float16)
        v = paddle.randn([2, 4, 8], dtype=paddle.float16)
        q_out, k_out, v_out = flashattn_auto_cast(q, k, v, dtype=paddle.float16)
        self.assertIs(q_out, q)
        self.assertIs(k_out, k)
        self.assertIs(v_out, v)

    def test_cast_float32_to_bfloat16(self):
        """Test casting float32 tensors to bfloat16."""
        q = paddle.randn([2, 4, 8], dtype=paddle.float32)
        k = paddle.randn([2, 4, 8], dtype=paddle.float32)
        v = paddle.randn([2, 4, 8], dtype=paddle.float32)
        q_out, k_out, v_out = flashattn_auto_cast(q, k, v)
        self.assertEqual(q_out.dtype, paddle.bfloat16)
        self.assertEqual(k_out.dtype, paddle.bfloat16)
        self.assertEqual(v_out.dtype, paddle.bfloat16)

    def test_partial_cast(self):
        """Test only casting tensors that need it."""
        q = paddle.randn([2, 4, 8], dtype=paddle.float32)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        q_out, k_out, v_out = flashattn_auto_cast(q, k, v)
        self.assertEqual(q_out.dtype, paddle.bfloat16)
        # k and v already bfloat16, should not be cast
        self.assertIs(k_out, k)
        self.assertIs(v_out, v)

    def test_cast_to_float16(self):
        """Test casting to float16 target dtype."""
        q = paddle.randn([2, 4, 8], dtype=paddle.float32)
        k = paddle.randn([2, 4, 8], dtype=paddle.float32)
        v = paddle.randn([2, 4, 8], dtype=paddle.float32)
        q_out, k_out, v_out = flashattn_auto_cast(q, k, v, dtype=paddle.float16)
        self.assertEqual(q_out.dtype, paddle.float16)
        self.assertEqual(k_out.dtype, paddle.float16)
        self.assertEqual(v_out.dtype, paddle.float16)


if __name__ == "__main__":
    unittest.main()
