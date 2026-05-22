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

import unittest

import paddle

from paddlefleet.ops.tilelang_dsv4.attention_core import (
    DEFAULT_TOPK_PAD_TO,
    tilelang_compressed_sparse_attn_paddle_compat_autograd,
)
from paddlefleet.ops.tilelang_dsv4.kernel.tilelang_sparse_mla import sparse_attn_tilelang_paddle


class TestTileLangDSV4AttentionTileLang(unittest.TestCase):
    def setUp(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA is required for TileLang sparse attention tests")
        paddle.set_device("gpu")
        paddle.seed(2031)

    def _make_inputs(self):
        batch = 1
        seq_len = 4
        kv_len = 4
        heads = 64
        dim = 32
        q = paddle.randn([batch, seq_len, heads, dim], dtype="bfloat16").contiguous()
        kv = paddle.randn([batch, kv_len, dim], dtype="bfloat16").contiguous()
        attn_sink = paddle.randn([heads], dtype="float32").contiguous()
        topk_idxs = paddle.to_tensor(
            [[[0, -1], [0, 1], [1, 2], [2, 3]]],
            dtype="int32",
        ).contiguous()
        return q, kv, attn_sink, topk_idxs

    def test_tilelang_forward_returns_paddle_tensors(self):
        q, kv, attn_sink, topk_idxs = self._make_inputs()

        out, lse = sparse_attn_tilelang_paddle(
            q,
            kv,
            attn_sink,
            topk_idxs,
            sm_scale=0.5,
            topk_pad_to=DEFAULT_TOPK_PAD_TO,
        )

        self.assertIsInstance(out, paddle.Tensor)
        self.assertIsInstance(lse, paddle.Tensor)
        self.assertEqual(tuple(out.shape), tuple(q.shape))
        self.assertEqual(tuple(lse.shape), tuple(q.shape[:3]))

    def test_pylayer_backward_returns_expected_gradients(self):
        q, kv, attn_sink, topk_idxs = self._make_inputs()
        q.stop_gradient = False
        kv.stop_gradient = False
        attn_sink.stop_gradient = False

        out = tilelang_compressed_sparse_attn_paddle_compat_autograd(
            q,
            kv,
            attn_sink,
            topk_idxs,
            softmax_scale=0.5,
            topk_pad_to=DEFAULT_TOPK_PAD_TO,
        )
        out.cast("float32").sum().backward()

        self.assertIsInstance(q.grad, paddle.Tensor)
        self.assertIsInstance(kv.grad, paddle.Tensor)
        self.assertIsInstance(attn_sink.grad, paddle.Tensor)
        self.assertEqual(tuple(q.grad.shape), tuple(q.shape))
        self.assertEqual(tuple(kv.grad.shape), tuple(kv.shape))
        self.assertEqual(tuple(attn_sink.grad.shape), tuple(attn_sink.shape))


if __name__ == "__main__":
    unittest.main()
