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

"""Real DSA forward/backward with distinct rank data and a simulated collective."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import paddle

from paddlefleet.transformer import dsa_attention as dsa


class TestDSASequenceParallel(unittest.TestCase):
    def test_rank_rows_global_keys_and_mask_shards_match_dense_values_and_gradients(
        self,
    ):
        rng = np.random.default_rng(20261004)
        arrays = [
            rng.normal(0, 0.4, (2, 4, 2, size)).astype("float32")
            for size in [2, 2, 3]
        ]
        indices = np.array([[0, -1, -1], [0, 1, -1], [0, 2, -1], [1, 2, 3]])
        topk = np.broadcast_to(indices, (2, 4, 3)).copy()
        causal = np.where(np.triu(np.ones((4, 4)), 1), -np.inf, 0).astype(
            "float32"
        )
        dense_mask = np.broadcast_to(causal, (2, 4, 4)).copy()
        dense_mask[0, 2, 0] = -np.inf
        dense_mask[1, 3, 2] = -np.inf
        sparse_mask = np.full((2, 4, 4), -np.inf)
        for row, keys in enumerate(indices):
            sparse_mask[:, row, keys[keys >= 0]] = 0
        scale = 2**-0.5
        upstream = rng.normal(size=(2, 2, 2, 3)).astype("float32")

        for rank in [0, 1]:
            for mask_kind in ["none", "causal", "global", "sharded"]:
                with self.subTest(rank=rank, mask=mask_kind):
                    rows = slice(rank * 2, rank * 2 + 2)
                    remote_rows = slice((1 - rank) * 2, (1 - rank) * 2 + 2)
                    tensors = [
                        paddle.to_tensor(value[:, rows], stop_gradient=False)
                        for value in arrays
                    ]
                    group = SimpleNamespace(nranks=2, rank=rank)
                    calls = []

                    def gather(tensor, *, group):
                        self.assertEqual(group.rank, rank)
                        if tensor.ndim == 4:
                            full = arrays[len(calls)].transpose(1, 0, 2, 3)
                        else:
                            full = dense_mask.transpose(2, 0, 1)
                        np.testing.assert_array_equal(
                            tensor.numpy(), full[rows]
                        )
                        calls.append(tuple(tensor.shape))
                        remote = paddle.to_tensor(full[remote_rows])
                        shards = (
                            [tensor, remote] if rank == 0 else [remote, tensor]
                        )
                        return paddle.concat(shards, axis=0)

                    def index(x, qr, mask):
                        self.assertEqual(list(x.shape), [2, 2, 4])
                        self.assertEqual(list(qr.shape), [2, 2, 4])
                        expected_mask = (
                            dense_mask
                            if mask_kind in ["global", "sharded"]
                            else causal[None, None]
                        )
                        np.testing.assert_array_equal(
                            mask.numpy(), expected_mask
                        )
                        return None, paddle.to_tensor(topk, dtype="int64")

                    layer = SimpleNamespace(
                        config=SimpleNamespace(
                            sequence_parallel=True,
                            use_accuracy_compatible=False,
                        ),
                        pg_collection=SimpleNamespace(tp=group),
                        indexer=SimpleNamespace(forward=index),
                        skip_topk=False,
                        index_share=False,
                        training=False,
                        softmax_scale=scale,
                    )
                    attention_mask = None
                    if mask_kind in ["global", "sharded"]:
                        values = (
                            dense_mask
                            if mask_kind == "global"
                            else dense_mask[..., rows]
                        )
                        attention_mask = paddle.to_tensor(values[:, None])
                    with (
                        patch.object(
                            dsa,
                            "gather_from_sequence_parallel_region",
                            side_effect=gather,
                        ),
                        patch.object(
                            dsa.parallel_state,
                            "get_tensor_model_parallel_rank",
                            return_value=rank,
                        ),
                    ):
                        output = dsa.DSAttention.forward(
                            layer,
                            *tensors,
                            attention_mask,
                            attn_mask_type=dsa.AttnMaskType.causal
                            if mask_kind == "causal"
                            else None,
                            x=paddle.ones([2, 2, 4], dtype="bfloat16"),
                            qr=paddle.ones([2, 2, 4], dtype="bfloat16"),
                        )
                        (
                            output * paddle.to_tensor(upstream.reshape(2, 2, 6))
                        ).sum().backward()
                    self.assertEqual(
                        len(calls), 5 if mask_kind == "sharded" else 3
                    )

                    query, key, value = [
                        array.astype("float64") for array in arrays
                    ]
                    query = query[:, rows]
                    mask = sparse_mask + causal
                    if attention_mask is not None:
                        mask += dense_mask
                    scores = (
                        np.einsum("bqhd,bkhd->bhqk", query, key) * scale
                        + mask[:, None, rows]
                    )
                    weights = np.exp(
                        scores - scores.max(axis=-1, keepdims=True)
                    )
                    weights /= weights.sum(axis=-1, keepdims=True)
                    expected = np.einsum("bhqk,bkhd->bqhd", weights, value)
                    weight_grad = np.einsum("bqhd,bkhd->bhqk", upstream, value)
                    score_grad = weights * (
                        weight_grad
                        - (weight_grad * weights).sum(axis=-1, keepdims=True)
                    )
                    gradients = [
                        np.einsum("bhqk,bkhd->bqhd", score_grad, key) * scale,
                        (
                            np.einsum("bhqk,bqhd->bkhd", score_grad, query)
                            * scale
                        )[:, rows],
                        np.einsum("bhqk,bqhd->bkhd", weights, upstream)[
                            :, rows
                        ],
                    ]
                    np.testing.assert_allclose(
                        output.numpy(),
                        expected.reshape(2, 2, 6),
                        rtol=5e-6,
                        atol=2e-6,
                    )
                    for tensor, gradient in zip(
                        tensors, gradients, strict=True
                    ):
                        np.testing.assert_allclose(
                            tensor.grad.numpy(), gradient, rtol=5e-6, atol=2e-6
                        )


if __name__ == "__main__":
    paddle.set_device("gpu:0")
    unittest.main()
