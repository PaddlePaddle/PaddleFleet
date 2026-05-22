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

from paddlefleet.ops.tilelang_dsv4.kernel.tilelang_csa_indexer_bwd import (
    csa_indexer_bwd_interface,
)
from paddlefleet.ops.tilelang_dsv4.kernel.tilelang_csa_indexer_fwd import (
    csa_indexer_topk_fwd_interface,
)


def ref_csa_indexer_bwd(index_q, weights, index_k_comp, topk_indices, grad_scores):
    with torch.enable_grad():
        q = index_q.detach().clone().requires_grad_(True)
        w = weights.detach().clone().requires_grad_(True)
        k = index_k_comp.detach().clone().requires_grad_(True)
        scores = torch.einsum("bshd,btd->bsht", q.float(), k.float())
        scores = F.relu(scores * (q.shape[-1] ** -0.5))
        scores = (scores * w.float().unsqueeze(-1)).sum(dim=2)
        valid = topk_indices >= 0
        safe_indices = topk_indices.clamp(min=0)
        selected = torch.gather(scores, dim=-1, index=safe_indices.to(torch.int64))
        selected = torch.where(valid, selected, torch.zeros_like(selected))
        loss = (selected * grad_scores.float()).sum()
        loss.backward()
    return q.grad, w.grad, k.grad


class TestTileLangDSV4CSAIndexerBwd(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for TileLang CSA indexer test")
        torch.manual_seed(2027)

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

        topk_indices, _ = csa_indexer_topk_fwd_interface(
            index_q,
            index_k_comp,
            weights,
            ratio=ratio,
            topk_effective=topk_effective,
            block_K=32,
            num_threads=128,
        )
        grad_scores = torch.randn(batch, seq_len, topk_effective, device=device, dtype=torch.float32)
        grad_scores = torch.where(topk_indices >= 0, grad_scores, torch.zeros_like(grad_scores)).contiguous()

        out_dq, out_dw, out_dk = csa_indexer_bwd_interface(
            index_q,
            weights,
            index_k_comp,
            topk_indices.contiguous(),
            grad_scores,
            block_I=32,
            num_threads=128,
        )
        ref_dq, ref_dw, ref_dk = ref_csa_indexer_bwd(
            index_q,
            weights,
            index_k_comp,
            topk_indices,
            grad_scores,
        )

        self.assertEqual(tuple(out_dq.shape), tuple(index_q.shape))
        self.assertEqual(tuple(out_dw.shape), tuple(weights.shape))
        self.assertEqual(tuple(out_dk.shape), tuple(index_k_comp.shape))
        torch.testing.assert_close(out_dq.float().cpu(), ref_dq.float().cpu(), rtol=6e-2, atol=2e-2)
        torch.testing.assert_close(out_dw.cpu(), ref_dw.cpu(), rtol=6e-2, atol=3e-2)
        torch.testing.assert_close(out_dk.cpu(), ref_dk.float().cpu(), rtol=6e-2, atol=3e-2)

    def test_selected_topk_backward(self):
        self._run_case(topk_effective=2)

    def test_full_candidate_backward(self):
        self._run_case(topk_effective=4)

    def test_output_padding_backward(self):
        self._run_case(topk_effective=6)


if __name__ == "__main__":
    unittest.main()
