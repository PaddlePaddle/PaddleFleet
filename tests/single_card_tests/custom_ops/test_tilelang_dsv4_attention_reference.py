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

from paddlefleet.tilelang_ops.attention_core import sparse_attn_paddle


def dense_sparse_attention_reference(q, kv, attn_sink, topk_idxs, sm_scale):
    q_dtype = q.dtype
    q = q.cast("float32")
    kv = kv.cast("float32")
    b, m, h, d = q.shape
    k_len = kv.shape[1]

    scores = paddle.einsum("bmhd,bkd->bmhk", q, kv) * sm_scale
    topk_mask = topk_idxs != -1
    safe_idxs = paddle.where(topk_mask, topk_idxs, paddle.zeros_like(topk_idxs)).cast("int64")
    selected = paddle.zeros([b, m, k_len], dtype="int32")
    selected = paddle.scatter_nd(
        paddle.stack(
            [
                paddle.arange(b, dtype="int64").reshape([b, 1, 1]).expand(safe_idxs.shape),
                paddle.arange(m, dtype="int64").reshape([1, m, 1]).expand(safe_idxs.shape),
                safe_idxs,
            ],
            axis=-1,
        ).reshape([-1, 3]),
        topk_mask.cast("int32").reshape([-1]),
        selected.shape,
    ).cast("bool")
    selected = selected.reshape([b, m, 1, k_len]).expand([b, m, h, k_len])
    masked_scores = paddle.where(selected, scores, paddle.full_like(scores, float("-inf")))

    max_scores = paddle.maximum(
        paddle.max(masked_scores, axis=-1),
        paddle.full([b, m, h], -1e30, dtype="float32"),
    )
    exp_scores = paddle.where(
        selected,
        paddle.exp(masked_scores - max_scores.unsqueeze(-1)),
        paddle.zeros_like(masked_scores),
    )
    numerator = paddle.einsum("bmhk,bkd->bmhd", exp_scores, kv)
    denominator = paddle.sum(exp_scores, axis=-1) + paddle.exp(attn_sink.reshape([1, 1, h]) - max_scores)
    return (numerator / denominator.unsqueeze(-1)).cast(q_dtype)


class TestTileLangDSV4AttentionReference(unittest.TestCase):
    def setUp(self):
        paddle.seed(2030)

    def _make_inputs(self):
        q = paddle.randn([2, 3, 4, 8], dtype="float32")
        kv = paddle.randn([2, 5, 8], dtype="float32")
        attn_sink = paddle.randn([4], dtype="float32")
        topk_idxs = paddle.to_tensor(
            [
                [[0, 2, -1], [1, 4, 0], [-1, -1, -1]],
                [[3, 1, -1], [4, 2, 0], [0, -1, -1]],
            ],
            dtype="int32",
        )
        return q, kv, attn_sink, topk_idxs

    def test_sparse_attn_matches_dense_mask_reference(self):
        q, kv, attn_sink, topk_idxs = self._make_inputs()

        for sm_scale in (0.25, 0.5):
            actual = sparse_attn_paddle(q, kv, attn_sink, topk_idxs, sm_scale=sm_scale)
            expected = dense_sparse_attention_reference(q, kv, attn_sink, topk_idxs, sm_scale=sm_scale)

            paddle.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_sparse_attn_casts_padded_indices_and_preserves_dtype(self):
        q, kv, attn_sink, topk_idxs = self._make_inputs()
        q = q.cast("bfloat16")
        kv = kv.cast("bfloat16")
        topk_idxs = topk_idxs.cast("int64")

        actual = sparse_attn_paddle(q, kv, attn_sink, topk_idxs, sm_scale=0.5)

        self.assertEqual(actual.dtype, paddle.bfloat16)
        self.assertEqual(tuple(actual.shape), tuple(q.shape))

    def test_sparse_attn_rejects_invalid_shapes_and_indices(self):
        q, kv, attn_sink, topk_idxs = self._make_inputs()

        with self.assertRaisesRegex(ValueError, "topk_idxs must have shape"):
            sparse_attn_paddle(q, kv, attn_sink, paddle.zeros([3], dtype="int32"))

        with self.assertRaisesRegex(ValueError, "topk_idxs shape"):
            sparse_attn_paddle(q, kv, attn_sink, paddle.zeros([2, 4, 3], dtype="int32"))

        with self.assertRaisesRegex(ValueError, "attn_sink shape"):
            sparse_attn_paddle(q, kv, paddle.zeros([3], dtype="float32"), topk_idxs)

        bad_topk = topk_idxs.clone()
        bad_topk[0, 0, 0] = kv.shape[1]
        with self.assertRaisesRegex(ValueError, "index >= kv length"):
            sparse_attn_paddle(q, kv, attn_sink, bad_topk)


if __name__ == "__main__":
    unittest.main()
