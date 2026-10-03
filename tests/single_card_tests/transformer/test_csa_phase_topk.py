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

"""Keep CSA phase-two attention dense when auxiliary loss is inactive."""

import unittest
from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import paddle

from paddlefleet.transformer import csa_attention as csa


class FixedIndexer:
    index_topk = 1
    softmax_scale = 0.5

    def forward_before_topk(self, x, qr, *, docmask_meta=None):
        # Positive increasing keys give an unambiguous largest causal block.
        q = paddle.ones([1, 8, 2, 2])
        k = (
            paddle.arange(1, 5, dtype="float32")
            .reshape([1, 4, 1])
            .expand([1, 4, 2])
        )
        return q, k, paddle.ones([1, 8, 2])

    def __call__(self, *args, **kwargs):
        return csa.CSAIndexer.forward(self, *args, **kwargs)


class TestCSAPhaseTopK(unittest.TestCase):
    def check_candidates(self, sparse):
        for training, coefficient, no_grad in [
            (False, 1.0, False),
            (True, 0.0, False),
            (True, 1.0, True),
        ]:
            with self.subTest(
                training=training, coefficient=coefficient, no_grad=no_grad
            ):
                config = SimpleNamespace(
                    csa_indexer_backend="unfused",
                    dsa_indexer_loss_coeff=coefficient,
                    dsa_indexer_use_sparse_loss=sparse,
                )
                layer = SimpleNamespace(
                    training=training,
                    config=config,
                    indexer=FixedIndexer(),
                    compress_ratio=2,
                )
                layer._resolve_topk_effective = MethodType(
                    csa.CompressedSparseAttention._resolve_topk_effective, layer
                )
                with paddle.no_grad() if no_grad else nullcontext():
                    indices, loss, state = (
                        csa.CompressedSparseAttention._compute_indexer_compressed_topk_idxs(
                            layer,
                            paddle.ones([1, 8, 2, 2]),
                            paddle.ones([1, 8, 2]),
                            paddle.ones([1, 8, 2]),
                            paddle.ones([1, 4, 2]),
                            n_compressed=4,
                            offset=8,
                        )
                    )
                self.assertIsNone(loss)
                self.assertIsNone(state)
                self.assertEqual(
                    list(indices.shape), [1, 8, 1 if sparse else 4]
                )
                for position, row in enumerate(indices.numpy()[0]):
                    valid_blocks = (position + 1) // 2
                    expected = list(range(8, 8 + valid_blocks))
                    if sparse:
                        expected = expected[-1:]
                    self.assertEqual(
                        sorted(int(value) for value in row if value >= 0),
                        expected,
                    )

    def test_phase_two_keeps_every_causal_candidate_without_auxiliary_loss(
        self,
    ):
        self.check_candidates(sparse=False)

    def test_phase_three_keeps_the_configured_topk_without_auxiliary_loss(self):
        self.check_candidates(sparse=True)


if __name__ == "__main__":
    paddle.set_device("gpu:0")
    unittest.main()
