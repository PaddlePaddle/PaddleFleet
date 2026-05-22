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

from paddlefleet.ops.tilelang_dsv4 import (
    tilelang_csa_compressed_indexer_bwd_paddle,
    tilelang_csa_compressed_indexer_topk_paddle,
)


class TestTileLangDSV4CSAIndexerCore(unittest.TestCase):
    def setUp(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA is required for TileLang CSA indexer wrapper test")
        paddle.set_device("gpu")
        paddle.seed(2028)

    def test_forward_backward_exported_entries(self):
        batch = 1
        seq_len = 16
        seq_len_comp = 4
        heads = 64
        dim = 128
        topk_effective = 6

        index_q = paddle.randn([batch, seq_len, heads, dim], dtype="bfloat16").contiguous()
        index_k_comp = paddle.randn([batch, seq_len_comp, dim], dtype="bfloat16").contiguous()
        weights = paddle.randn([batch, seq_len, heads], dtype="float32").contiguous()

        topk_indices, topk_scores = tilelang_csa_compressed_indexer_topk_paddle(
            index_q,
            index_k_comp,
            weights,
            ratio=4,
            topk_effective=topk_effective,
        )
        self.assertEqual(tuple(topk_indices.shape), (batch, seq_len, topk_effective))
        self.assertEqual(tuple(topk_scores.shape), (batch, seq_len, topk_effective))
        self.assertEqual(topk_indices.dtype, paddle.int32)
        self.assertEqual(topk_scores.dtype, paddle.float32)
        self.assertTrue(paddle.all(topk_scores[topk_indices < 0] == 0).item())

        grad_scores = paddle.randn([batch, seq_len, topk_effective], dtype="float32").contiguous()
        grad_q, grad_weights, grad_k_comp = tilelang_csa_compressed_indexer_bwd_paddle(
            index_q,
            weights,
            index_k_comp,
            topk_indices,
            grad_scores,
        )
        self.assertEqual(tuple(grad_q.shape), tuple(index_q.shape))
        self.assertEqual(tuple(grad_weights.shape), tuple(weights.shape))
        self.assertEqual(tuple(grad_k_comp.shape), tuple(index_k_comp.shape))
        self.assertEqual(grad_weights.dtype, paddle.float32)
        self.assertEqual(grad_k_comp.dtype, paddle.float32)


if __name__ == "__main__":
    unittest.main()
