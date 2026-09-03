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

"""Coverage for ``dense_indexer_kl_bwd``'s ``grad_loss`` normalisation.

The cuDNN wrapper is mocked so the branches that coerce ``grad_loss`` into a
fp32 scalar tensor (``None`` -> ones, python number -> tensor, non-fp32 tensor
-> cast) run without a real kernel.
"""

import sys
import types
import unittest
from unittest.mock import patch

import paddle

from paddlefleet.cudnn_ops.indexer import dense_indexer_kl_cudnn as mod

_API = "paddlefleet_ops.cudnn.deepseek_sparse_attention.indexer_backward.api"


def _fake_wrapper(*args, **kwargs):
    return {
        "d_index_q": paddle.zeros([1], dtype="float32"),
        "d_weights": paddle.zeros([1], dtype="float32"),
        "d_index_k": paddle.zeros([1], dtype="float32"),
    }


class TestDenseIndexerKlBwdGradLoss(unittest.TestCase):
    def _call(self, grad_loss):
        index_q = paddle.randn([4, 2, 8])
        weights = paddle.randn([4, 2])
        index_k = paddle.randn([16, 8])
        attn_score = paddle.randn([4, 16])
        attn_l1norm = paddle.randn([4])
        index_score = paddle.randn([4, 16])
        index_lse = paddle.randn([4, 2])
        cu_q = paddle.to_tensor([0, 4], dtype="int32")
        cu_k = paddle.to_tensor([0, 16], dtype="int32")

        fake_api = types.ModuleType(_API)
        fake_api.dense_indexer_backward_wrapper = _fake_wrapper
        with (
            patch.dict(sys.modules, {_API: fake_api}),
            patch.object(mod, "_require_cudnn_frontend", lambda: None),
        ):
            return mod.dense_indexer_kl_bwd(
                index_q,
                weights,
                index_k,
                attn_score,
                attn_l1norm,
                index_score,
                index_lse,
                1.0,
                cu_q,
                cu_k,
                4,
                16,
                grad_loss=grad_loss,
                block_I=128,
            )

    def test_returns_triple_for_each_grad_loss_form(self):
        for grad_loss in (
            None,  # -> paddle.ones([])
            2.0,  # python number -> to_tensor
            paddle.ones([], dtype="float64"),  # non-fp32 tensor -> cast
            paddle.ones([], dtype="float32"),  # already fp32 (no-op branch)
        ):
            out = self._call(grad_loss)
            self.assertEqual(len(out), 3)


if __name__ == "__main__":
    unittest.main()
