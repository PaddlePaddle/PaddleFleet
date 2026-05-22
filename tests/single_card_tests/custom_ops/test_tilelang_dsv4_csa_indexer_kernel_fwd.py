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

import torch
import torch.nn.functional as F

from paddlefleet.ops.tilelang_dsv4.kernel.tilelang_csa_indexer_fwd import (
    csa_indexer_topk_fwd_interface,
)


def ref_csa_indexer_topk(index_q, index_k_comp, weights, ratio, topk_effective):
    scores = torch.einsum("bshd,btd->bsht", index_q.float(), index_k_comp.float())
    scores = F.relu(scores)
    scores = (scores * weights.float().unsqueeze(-1)).sum(dim=2)
    scores = scores * (index_q.shape[-1] ** -0.5)

    batch, seq_len, seq_len_comp = scores.shape
    comp_ids = torch.arange(seq_len_comp, device=scores.device).view(1, 1, seq_len_comp)
    positions = torch.arange(1, seq_len + 1, device=scores.device).view(1, seq_len, 1)
    valid_end = positions // ratio
    valid_mask = comp_ids < valid_end
    scores = scores.masked_fill(~valid_mask, float("-inf"))

    actual_topk = min(topk_effective, seq_len_comp)
    topk_scores_raw, topk_indices = torch.topk(scores, k=actual_topk, dim=-1)
    valid_topk = torch.gather(valid_mask.expand(batch, -1, -1), dim=-1, index=topk_indices)
    topk_indices = torch.where(valid_topk, topk_indices, torch.full_like(topk_indices, -1))
    topk_scores_raw = torch.where(valid_topk, topk_scores_raw, torch.full_like(topk_scores_raw, float("-inf")))

    topk_probs = torch.softmax(topk_scores_raw, dim=-1)
    topk_probs = torch.where(valid_topk, topk_probs, torch.zeros_like(topk_probs))

    if topk_effective > actual_topk:
        pad = topk_effective - actual_topk
        topk_indices = torch.cat(
            [topk_indices, torch.full((batch, seq_len, pad), -1, device=scores.device, dtype=topk_indices.dtype)],
            dim=-1,
        )
        topk_probs = torch.cat(
            [topk_probs, torch.zeros((batch, seq_len, pad), device=scores.device, dtype=topk_probs.dtype)],
            dim=-1,
        )

    return topk_indices.int(), topk_probs.float()


class TestTileLangDSV4CSAIndexerFwd(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for TileLang CSA indexer test")
        torch.manual_seed(2026)

    def _run_case(self, topk_effective):
        batch = 1
        seq_len = 16
        seq_len_comp = 4
        heads = 64
        dim = 128
        ratio = 4
        device = "cuda"

        index_q = torch.randn(batch, seq_len, heads, dim, device=device, dtype=torch.bfloat16).contiguous()
        index_k_comp = torch.randn(batch, seq_len_comp, dim, device=device, dtype=torch.bfloat16).contiguous()
        weights = torch.randn(batch, seq_len, heads, device=device, dtype=torch.float32).contiguous()

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
        torch.testing.assert_close(out_indices.cpu(), ref_indices.cpu(), rtol=0, atol=0)
        valid = ref_indices >= 0
        torch.testing.assert_close(out_scores.cpu()[valid.cpu()], ref_scores.cpu()[valid.cpu()], rtol=6e-2, atol=2e-2)
        torch.testing.assert_close(out_scores.cpu()[~valid.cpu()], ref_scores.cpu()[~valid.cpu()], rtol=0, atol=0)

        # Early positions have no valid compressed block for ratio=4.
        self.assertTrue(torch.all(out_indices[:, :3, :] == -1).item())
        self.assertTrue(torch.all(out_scores[:, :3, :] == 0).item())

    def test_selected_topk_forward(self):
        self._run_case(topk_effective=2)

    def test_full_candidate_forward(self):
        self._run_case(topk_effective=4)

    def test_output_padding_forward(self):
        self._run_case(topk_effective=6)


if __name__ == "__main__":
    unittest.main()
