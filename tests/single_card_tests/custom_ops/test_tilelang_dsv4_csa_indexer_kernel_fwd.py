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
import paddle.nn.functional as F

from paddlefleet.tilelang_ops.kernel.tilelang_csa_indexer_fwd import (
    csa_indexer_topk_fwd_interface,
)


def ref_csa_indexer_topk(index_q, index_k_comp, weights, ratio, topk_effective):
    scores = paddle.einsum("bshd,btd->bsht", index_q.cast("float32"), index_k_comp.cast("float32"))
    scores = F.relu(scores)
    scores = (scores * weights.cast("float32").unsqueeze(-1)).sum(axis=2)
    scores = scores * (index_q.shape[-1] ** -0.5)

    batch, seq_len, seq_len_comp = scores.shape
    comp_ids = paddle.arange(seq_len_comp, dtype="int64").reshape([1, 1, seq_len_comp])
    positions = paddle.arange(1, seq_len + 1, dtype="int64").reshape([1, seq_len, 1])
    valid_end = positions // ratio
    valid_mask = comp_ids < valid_end
    scores = paddle.where(valid_mask, scores, paddle.full_like(scores, float("-inf")))

    actual_topk = min(topk_effective, seq_len_comp)
    topk_scores_raw, topk_indices = paddle.topk(scores, k=actual_topk, axis=-1)
    valid_topk = paddle.take_along_axis(
        paddle.expand(valid_mask, [batch, seq_len, seq_len_comp]).cast("int32"),
        topk_indices,
        axis=-1,
    ).cast("bool")
    topk_indices = paddle.where(valid_topk, topk_indices, paddle.full_like(topk_indices, -1))
    topk_scores_raw = paddle.where(valid_topk, topk_scores_raw, paddle.full_like(topk_scores_raw, float("-inf")))

    topk_probs = F.softmax(topk_scores_raw, axis=-1)
    topk_probs = paddle.where(valid_topk, topk_probs, paddle.zeros_like(topk_probs))

    if topk_effective > actual_topk:
        pad = topk_effective - actual_topk
        topk_indices = paddle.concat(
            [topk_indices, paddle.full([batch, seq_len, pad], -1, dtype=topk_indices.dtype)],
            axis=-1,
        )
        topk_probs = paddle.concat(
            [topk_probs, paddle.zeros([batch, seq_len, pad], dtype=topk_probs.dtype)],
            axis=-1,
        )

    return topk_indices.cast("int32"), topk_probs.cast("float32")


class TestTileLangDSV4CSAIndexerFwd(unittest.TestCase):
    def setUp(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA is required for TileLang CSA indexer test")
        paddle.set_device("gpu")
        paddle.seed(2026)

    def _run_case(self, topk_effective):
        batch = 1
        seq_len = 16
        seq_len_comp = 4
        heads = 64
        dim = 128
        ratio = 4

        index_q = paddle.randn([batch, seq_len, heads, dim], dtype="bfloat16").contiguous()
        index_k_comp = paddle.randn([batch, seq_len_comp, dim], dtype="bfloat16").contiguous()
        weights = paddle.randn([batch, seq_len, heads], dtype="float32").contiguous()

        out_indices, out_scores = csa_indexer_topk_fwd_interface(
            index_q,
            index_k_comp,
            weights,
            ratio=ratio,
            topk_effective=topk_effective,
            block_K=32,
            num_threads=128,
        )
        ref_indices, ref_scores = ref_csa_indexer_topk(
            index_q,
            index_k_comp,
            weights,
            ratio=ratio,
            topk_effective=topk_effective,
        )

        self.assertEqual(tuple(out_indices.shape), (batch, seq_len, topk_effective))
        self.assertEqual(tuple(out_scores.shape), (batch, seq_len, topk_effective))
        self.assertTrue(paddle.all(out_indices.cpu() == ref_indices.cpu()).item())
        valid = ref_indices >= 0
        paddle.testing.assert_close(out_scores.cpu()[valid.cpu()], ref_scores.cpu()[valid.cpu()], rtol=6e-2, atol=2e-2)
        self.assertTrue(paddle.all(out_scores.cpu()[~valid.cpu()] == ref_scores.cpu()[~valid.cpu()]).item())

        self.assertTrue(paddle.all(out_indices[:, :3, :] == -1).item())
        self.assertTrue(paddle.all(out_scores[:, :3, :] == 0).item())

    def test_selected_topk_forward(self):
        self._run_case(topk_effective=2)

    def test_full_candidate_forward(self):
        self._run_case(topk_effective=4)

    def test_output_padding_forward(self):
        self._run_case(topk_effective=6)


if __name__ == "__main__":
    unittest.main()
