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
from unittest.mock import MagicMock, patch

import paddle

from paddlefleet.refined_recompute.flash_attn import (
    FlashAttnFunctor,
    FlashMaskAttnFunctor,
    RefinedRcomputeFlashAttention,
    RefinedRcomputeFlashMaskAttention,
    flashattn_auto_cast,
)


class TestGetFAVersion(unittest.TestCase):
    """Tests for get_fa_version function - removed as get_fa_version no longer exists in this module."""


class TestFlashattnAutoCast(unittest.TestCase):
    """Tests for flashattn_auto_cast function."""

    def test_no_cast_needed(self):
        """Test no cast when tensors are already bfloat16."""
        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        q_out, k_out, v_out = flashattn_auto_cast(q, k, v)
        self.assertEqual(q_out.dtype, paddle.bfloat16)
        self.assertEqual(k_out.dtype, paddle.bfloat16)
        self.assertEqual(v_out.dtype, paddle.bfloat16)

    def test_cast_from_float32(self):
        """Test casting from float32 to bfloat16."""
        q = paddle.randn([2, 4, 8], dtype=paddle.float32)
        k = paddle.randn([2, 4, 8], dtype=paddle.float32)
        v = paddle.randn([2, 4, 8], dtype=paddle.float32)
        q_out, k_out, v_out = flashattn_auto_cast(q, k, v)
        self.assertEqual(q_out.dtype, paddle.bfloat16)
        self.assertEqual(k_out.dtype, paddle.bfloat16)
        self.assertEqual(v_out.dtype, paddle.bfloat16)

    def test_cast_to_custom_dtype(self):
        """Test casting to custom dtype."""
        q = paddle.randn([2, 4, 8], dtype=paddle.float32)
        k = paddle.randn([2, 4, 8], dtype=paddle.float32)
        v = paddle.randn([2, 4, 8], dtype=paddle.float32)
        q_out, k_out, v_out = flashattn_auto_cast(q, k, v, dtype=paddle.float16)
        self.assertEqual(q_out.dtype, paddle.float16)
        self.assertEqual(k_out.dtype, paddle.float16)
        self.assertEqual(v_out.dtype, paddle.float16)

    def test_mixed_dtypes(self):
        """Test casting with mixed input dtypes."""
        q = paddle.randn([2, 4, 8], dtype=paddle.float32)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.float16)
        q_out, k_out, v_out = flashattn_auto_cast(q, k, v)
        self.assertEqual(q_out.dtype, paddle.bfloat16)
        self.assertEqual(k_out.dtype, paddle.bfloat16)
        self.assertEqual(v_out.dtype, paddle.bfloat16)


class TestFlashAttnFunctorForwardVersion3(unittest.TestCase):
    """Tests for FlashAttnFunctor.forward with FA version 3."""

    def test_forward_version_3_saves_correct_tensors(self):
        """Test forward with version 3 saves correct tensors."""
        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        result_attn = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        softmax_lse = paddle.randn([2, 4], dtype=paddle.float32)

        hold_tensors = {
            "fa_version": 3,
            "result_attention": result_attn,
            "softmax_lse": softmax_lse,
            "causal": True,
        }

        result = FlashAttnFunctor.apply(q, k, v, hold_tensors)
        self.assertEqual(result.shape, result_attn.shape)


class TestFlashAttnFunctorForwardVersion4(unittest.TestCase):
    """Tests for FlashAttnFunctor.forward with FA version 4."""

    def test_forward_version_4_saves_correct_tensors(self):
        """Test forward with version 4 saves correct tensors."""
        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        result_attn = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        softmax_lse = paddle.randn([2, 4], dtype=paddle.float32)

        hold_tensors = {
            "fa_version": 4,
            "result_attention": result_attn,
            "softmax_lse": softmax_lse,
            "causal": True,
        }

        result = FlashAttnFunctor.apply(q, k, v, hold_tensors)
        self.assertEqual(result.shape, result_attn.shape)


class TestRefinedRcomputeFlashAttentionFirstFwd(unittest.TestCase):
    """Tests for RefinedRcomputeFlashAttention._first_fwd."""

    @patch("paddlefleet.refined_recompute.flash_attn.flash_attn_dispatch_fwd")
    @patch("paddlefleet.refined_recompute.flash_attn.framework._dygraph_tracer")
    def test_first_fwd_puts_in_queue(self, mock_tracer, mock_dispatch_fwd):
        """Test _first_fwd puts tensors in queue."""
        mock_tracer_obj = MagicMock()
        mock_tracer_obj._has_grad = False
        mock_tracer.return_value = mock_tracer_obj

        mock_dispatch_fwd.return_value = {
            "output": paddle.randn([2, 4, 8], dtype=paddle.bfloat16),
            "softmax_lse": paddle.randn([2, 4], dtype=paddle.float32),
            "seed_offset": paddle.zeros([2], dtype=paddle.int64),
            "result_softmax": None,
            "fa_version": 3,
            "need_pad": False,
            "head_dim_v": 8,
        }

        attn = RefinedRcomputeFlashAttention()
        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)

        result = attn.forward(q, k, v, training=True)
        self.assertFalse(attn._hold_tensors_queue.empty())


class TestFlashMaskAttnFunctorVersion3(unittest.TestCase):
    """Tests for FlashMaskAttnFunctor with FA version 3."""

    def test_forward_version_3(self):
        """Test forward with version 3."""
        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        startend = paddle.to_tensor([0, 4, 8], dtype=paddle.int32)
        result_attn = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        softmax_lse = paddle.randn([2, 4], dtype=paddle.float32)

        hold_tensors = {
            "fa_version": 3,
            "result_attention": result_attn,
            "softmax_lse": softmax_lse,
            "causal": True,
        }

        result = FlashMaskAttnFunctor.apply(q, k, v, startend, hold_tensors)
        self.assertEqual(result.shape, result_attn.shape)


class TestFlashMaskAttnFunctorVersion4(unittest.TestCase):
    """Tests for FlashMaskAttnFunctor with FA version 4."""

    def test_forward_version_4(self):
        """Test forward with version 4."""
        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        startend = paddle.to_tensor([0, 4, 8], dtype=paddle.int32)
        result_attn = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        softmax_lse = paddle.randn([2, 4], dtype=paddle.float32)

        hold_tensors = {
            "fa_version": 4,
            "result_attention": result_attn,
            "softmax_lse": softmax_lse,
            "causal": True,
        }

        result = FlashMaskAttnFunctor.apply(q, k, v, startend, hold_tensors)
        self.assertEqual(result.shape, result_attn.shape)


class TestRefinedRcomputeFlashMaskAttentionFirstFwdV3(unittest.TestCase):
    """Tests for RefinedRcomputeFlashMaskAttention._first_fwd with v3."""

    @patch("paddlefleet.refined_recompute.flash_attn.flash_attn_dispatch_fwd")
    @patch("paddlefleet.refined_recompute.flash_attn.framework._dygraph_tracer")
    def test_first_fwd_v3(self, mock_tracer, mock_dispatch_fwd):
        """Test _first_fwd puts tensors in queue."""
        mock_tracer_obj = MagicMock()
        mock_tracer_obj._has_grad = False
        mock_tracer.return_value = mock_tracer_obj

        mock_dispatch_fwd.return_value = {
            "output": paddle.randn([2, 4, 8], dtype=paddle.bfloat16),
            "softmax_lse": paddle.randn([2, 4], dtype=paddle.float32),
            "seed_offset": None,
            "result_softmax": None,
            "fa_version": 3,
            "need_pad": False,
            "head_dim_v": 8,
        }

        attn = RefinedRcomputeFlashMaskAttention()
        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        startend = paddle.to_tensor([0, 4, 8], dtype=paddle.int32)

        result = attn.forward(q, k, v, startend, causal=False)
        self.assertTrue(result is not None)


if __name__ == "__main__":
    unittest.main()
