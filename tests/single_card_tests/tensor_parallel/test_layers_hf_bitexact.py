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
"""Tests for the ``"hf"`` pieces in ``tensor_parallel/layers``.

Two independent things:

* ``_HFEmbeddingGather``. torch's ``embedding_dense_backward`` groups tokens by id
  and reduces each group in blocks of 32 rows: the rows of a block are summed in
  FP32 and the block total is rounded once into the BF16 parameter gradient.
  Paddle's ``index_select`` backward rounds after *every* row, which moved up to
  3.1e-02 on the token ids that repeat inside a sequence.

* The grouped dgrad/wgrad in ``LinearWithGradAccumulationAndAsyncCommunication``.
  A fused projection standing in for several reference projections has a dgrad
  that is a K-direction split of the reference's per-projection dgrads, and BF16
  GEMM K-splitting is not associative. Each group additionally carries the
  reference gradient's *operand layout* (``column_major``), because cuBLAS picks a
  different reduction for a ``[K, M]`` view than for the equivalent row-major
  matrix. The wgrad half computes in the reference's ``[out, in]`` orientation and
  transposes afterwards, since evaluating ``x^T @ g`` swaps M and N and makes
  cuBLAS reach for a split-K variant the reference never selects.

``hf_dgrad_groups`` is read off the weight tensor as an attribute, which is the
seam the ``_maybe_tag_*`` methods use and the seam these tests drive.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np
import paddle

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

from paddlefleet.tensor_parallel.layers import (
    LinearWithGradAccumulationAndAsyncCommunication as _Linear,
    _HFEmbeddingGather,
    _index_put_columns,
)


def _mock_group(world_size=1, rank=0):
    m = MagicMock()
    m.world_size = world_size
    m.nranks = world_size
    m.rank = rank
    m.ranks = list(range(world_size))
    return m


class TestHFEmbeddingGatherForward(unittest.TestCase):
    def setUp(self):
        paddle.seed(20260908)
        self.weight = paddle.randn([10, 6], dtype=paddle.float32)
        self.weight.stop_gradient = False
        self.index = paddle.to_tensor([[1, 3], [3, 7]], dtype="int64")

    def test_forward_is_a_flat_gather(self):
        out = _HFEmbeddingGather.apply(self.weight, self.index)
        expected = paddle.gather(self.weight, self.index.reshape([-1]), axis=0)
        np.testing.assert_array_equal(out.numpy(), expected.numpy())

    def test_forward_is_flattened_over_the_index_shape(self):
        out = _HFEmbeddingGather.apply(self.weight, self.index)
        self.assertEqual(out.shape, [4, 6])

    def test_forward_preserves_weight_dtype(self):
        for dtype in (paddle.float32, paddle.bfloat16):
            with self.subTest(dtype=dtype):
                w = self.weight.astype(dtype)
                w.stop_gradient = False
                out = _HFEmbeddingGather.apply(w, self.index)
                self.assertEqual(out.dtype, dtype)


class TestHFEmbeddingGatherBackward(unittest.TestCase):
    """Block-wise FP32 reduction with one rounding per block."""

    def _grad(self, weight, index, grad_out):
        w = weight.detach()
        w.stop_gradient = False
        out = _HFEmbeddingGather.apply(w, index)
        return paddle.grad([out], [w], grad_outputs=[grad_out])[0]

    def test_scatters_into_the_repeated_rows(self):
        """Repeated ids accumulate; untouched rows stay exactly zero."""
        weight = paddle.zeros([5, 3], dtype=paddle.float32)
        index = paddle.to_tensor([0, 2, 2, 2], dtype="int64")
        g = paddle.ones([4, 3], dtype=paddle.float32)
        gw = self._grad(weight, index, g)
        np.testing.assert_array_equal(
            gw.numpy(),
            np.array(
                [[1, 1, 1], [0, 0, 0], [3, 3, 3], [0, 0, 0], [0, 0, 0]],
                dtype=np.float32,
            ),
        )

    def test_grad_shape_and_dtype_match_the_parameter(self):
        weight = paddle.zeros([7, 4], dtype=paddle.bfloat16)
        index = paddle.to_tensor([[1, 1], [2, 6]], dtype="int64")
        g = paddle.ones([4, 4], dtype=paddle.bfloat16)
        gw = self._grad(weight, index, g)
        self.assertEqual(gw.shape, [7, 4])
        self.assertEqual(gw.dtype, paddle.bfloat16)

    def test_short_sequence_rounds_once_per_id(self):
        """Under 32 tokens every id fits one block: sum in FP32, round once."""
        rows, hidden, tokens = 4, 2, 20
        self.assertLess(tokens, _HFEmbeddingGather.BLOCK_ROWS)
        weight = paddle.zeros([rows, hidden], dtype=paddle.bfloat16)
        index = paddle.zeros([tokens], dtype="int64")
        # Values chosen so per-row BF16 rounding would lose the tail but a
        # single FP32 sum keeps it: 1.0 plus 19 copies of 2**-9.
        g = paddle.concat(
            [
                paddle.ones([1, hidden], dtype=paddle.bfloat16),
                paddle.full(
                    [tokens - 1, hidden], 2.0**-9, dtype=paddle.bfloat16
                ),
            ]
        )
        gw = self._grad(weight, index, g)
        exact = 1.0 + (tokens - 1) * 2.0**-9
        np.testing.assert_allclose(
            gw[0].astype("float32").numpy(),
            np.full(hidden, exact, dtype=np.float32),
            rtol=0,
            atol=2.0**-8,
        )

    def test_multi_block_sequence_splits_at_32(self):
        """Beyond 32 tokens the reduction is split, one rounding per block."""
        rows, hidden = 2, 2
        tokens = _HFEmbeddingGather.BLOCK_ROWS * 2 + 5
        weight = paddle.zeros([rows, hidden], dtype=paddle.float32)
        index = paddle.zeros([tokens], dtype="int64")
        g = paddle.ones([tokens, hidden], dtype=paddle.float32)
        gw = self._grad(weight, index, g)
        np.testing.assert_allclose(
            gw[0].numpy(),
            np.full(hidden, float(tokens), dtype=np.float32),
            rtol=0,
            atol=0,
        )
        np.testing.assert_array_equal(
            gw[1].numpy(), np.zeros(hidden, dtype=np.float32)
        )

    def test_block_rows_is_torchs_blockdimy(self):
        self.assertEqual(_HFEmbeddingGather.BLOCK_ROWS, 32)


class TestIndexPutColumns(unittest.TestCase):
    """Column write-back used by the grouped wgrad."""

    def test_writes_the_named_columns(self):
        target = paddle.zeros([3, 4], dtype=paddle.float32)
        columns = paddle.to_tensor([1, 3], dtype="int64")
        values = paddle.to_tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype="float32"
        )
        out = _index_put_columns(target, columns, values)
        np.testing.assert_array_equal(
            out.numpy(),
            np.array(
                [[0, 1, 0, 2], [0, 3, 0, 4], [0, 5, 0, 6]], dtype=np.float32
            ),
        )

    def test_disjoint_groups_reconstruct_the_whole_matrix(self):
        paddle.seed(1)
        target = paddle.zeros([3, 6], dtype=paddle.float32)
        full = paddle.randn([3, 6], dtype=paddle.float32)
        for cols in ([0, 2, 4], [1, 3, 5]):
            idx = paddle.to_tensor(cols, dtype="int64")
            target = _index_put_columns(
                target, idx, full.index_select(axis=-1, index=idx)
            )
        np.testing.assert_array_equal(target.numpy(), full.numpy())


class TestLinearGroupedDgrad(unittest.TestCase):
    """``hf_dgrad_groups`` turns one wide-K dgrad into per-projection GEMMs."""

    def _apply(self, inp, weight, bias=None):
        return _Linear.apply(
            inp,
            weight,
            bias,
            False,  # gradient_accumulation_fusion
            False,  # allreduce_dgrad
            False,  # sequence_parallel
            None,  # grad_output_buffer
            0,  # wgrad_deferral_limit
            _mock_group(),  # tp_group
            "hf",  # use_accuracy_compatible
        )

    def _grads(self, groups, dtype=paddle.float32, seed=7):
        paddle.seed(seed)
        inp = paddle.randn([6, 8], dtype=dtype)
        inp.stop_gradient = False
        weight = paddle.randn([8, 6], dtype=dtype)
        weight.stop_gradient = False
        if groups is not None:
            weight.hf_dgrad_groups = groups
        out = self._apply(inp, weight)
        g = paddle.ones(out.shape, dtype=dtype)
        gi, gw = paddle.grad([out], [inp, weight], grad_outputs=[g])
        return gi, gw

    def _cols(self, *ranges):
        return [paddle.to_tensor(list(r), dtype="int64") for r in ranges]

    def test_grouped_dgrad_matches_the_ungrouped_value(self):
        """Same mathematical result; the split only changes the rounding."""
        gi_grouped, _ = self._grads(self._cols(range(0, 3), range(3, 6)))
        gi_plain, _ = self._grads(None)
        np.testing.assert_allclose(
            gi_grouped.numpy(), gi_plain.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_grouped_dgrad_shape_is_preserved(self):
        gi, _ = self._grads(self._cols(range(0, 2), range(2, 6)))
        self.assertEqual(gi.shape, [6, 8])

    def test_group_order_is_the_accumulation_order(self):
        """Reversing the groups reverses the chaining, so BF16 can differ."""
        a = self._cols(range(0, 3), range(3, 6))
        b = list(reversed(a))
        gi_a, _ = self._grads(a, dtype=paddle.bfloat16)
        gi_b, _ = self._grads(b, dtype=paddle.bfloat16)
        np.testing.assert_allclose(
            gi_a.astype("float32").numpy(),
            gi_b.astype("float32").numpy(),
            rtol=0,
            atol=0.5,
        )

    def test_column_major_entry_is_accepted_as_a_tuple(self):
        """``(columns, column_major)`` reproduces a transposed reference grad."""
        cols = self._cols(range(0, 3), range(3, 6))
        tupled = [(cols[0], True), (cols[1], False)]
        gi_tupled, _ = self._grads(tupled)
        gi_plain, _ = self._grads(cols)
        np.testing.assert_allclose(
            gi_tupled.numpy(), gi_plain.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_single_group_covering_everything(self):
        gi, _ = self._grads(self._cols(range(0, 6)))
        gi_plain, _ = self._grads(None)
        np.testing.assert_allclose(
            gi.numpy(), gi_plain.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_grouped_wgrad_matches_the_ungrouped_value(self):
        """The wgrad half computes in ``[out, in]`` orientation, then transposes."""
        _, gw_grouped = self._grads(self._cols(range(0, 3), range(3, 6)))
        _, gw_plain = self._grads(None)
        np.testing.assert_allclose(
            gw_grouped.numpy(), gw_plain.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_grouped_wgrad_writes_every_column(self):
        """A group missed by the write-back would leave zeros behind."""
        _, gw = self._grads(self._cols(range(0, 2), range(2, 4), range(4, 6)))
        self.assertFalse(bool(paddle.any(gw == 0.0)))

    def test_column_major_wgrad_matches_row_major(self):
        cols = self._cols(range(0, 3), range(3, 6))
        _, gw_cm = self._grads([(cols[0], True), (cols[1], True)])
        _, gw_rm = self._grads(cols)
        np.testing.assert_allclose(
            gw_cm.numpy(), gw_rm.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_empty_groups_fall_back_to_the_plain_path(self):
        """An empty list is falsy, so the wide-K GEMM must still run."""
        gi_empty, _ = self._grads([])
        gi_plain, _ = self._grads(None)
        np.testing.assert_array_equal(gi_empty.numpy(), gi_plain.numpy())

    def test_non_hf_target_ignores_the_tag(self):
        """A tagged weight must not change a Megatron-aligned run."""
        paddle.seed(7)
        inp = paddle.randn([6, 8], dtype=paddle.float32)
        inp.stop_gradient = False
        weight = paddle.randn([8, 6], dtype=paddle.float32)
        weight.stop_gradient = False
        weight.hf_dgrad_groups = self._cols(range(0, 3), range(3, 6))
        out = _Linear.apply(
            inp,
            weight,
            None,
            False,
            False,
            False,
            None,
            0,
            _mock_group(),
            "megatron",
        )
        g = paddle.ones(out.shape, dtype=paddle.float32)
        gi, _ = paddle.grad([out], [inp, weight], grad_outputs=[g])
        gi_plain, _ = self._grads(None)
        np.testing.assert_allclose(
            gi.numpy(), gi_plain.numpy(), rtol=1e-6, atol=1e-6
        )


class TestVocabParallelEmbeddingHFBranch(unittest.TestCase):
    """``forward`` routes through ``_HFEmbeddingGather`` only under ``"hf"``."""

    def _make(self, target):
        from paddlefleet.tensor_parallel.layers import VocabParallelEmbedding
        from paddlefleet.transformer.transformer_config import (
            TransformerConfig,
        )

        config = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=8,
            intermediate_size=32,
            num_attention_heads=2,
            use_accuracy_compatible=target,
        )
        return VocabParallelEmbedding(
            16,
            8,
            init_method=paddle.nn.initializer.Constant(0.0),
            config=config,
        )

    def test_hf_target_uses_the_block_reduced_gather(self):
        """Forward value is a gather either way; the backward is what differs."""
        paddle.seed(11)
        emb = self._make("hf")
        with paddle.no_grad():
            emb.weight.set_value(
                paddle.randn(emb.weight.shape, dtype=emb.weight.dtype)
            )
        ids = paddle.to_tensor([[1, 3], [3, 5]], dtype="int64")
        out = emb(ids)
        expected = paddle.gather(emb.weight, ids.reshape([-1]), axis=0).reshape(
            [2, 2, 8]
        )
        np.testing.assert_array_equal(out.numpy(), expected.numpy())

    def test_repeated_ids_accumulate_into_the_weight_grad(self):
        """id 3 appears twice, so its row gets both contributions."""
        emb = self._make("hf")
        ids = paddle.to_tensor([[1, 3], [3, 5]], dtype="int64")
        out = emb(ids)
        out.sum().backward()
        grad = emb.weight.grad.numpy()
        np.testing.assert_allclose(grad[3], np.full(8, 2.0), rtol=0, atol=0)
        np.testing.assert_allclose(grad[1], np.full(8, 1.0), rtol=0, atol=0)
        np.testing.assert_allclose(grad[0], np.zeros(8), rtol=0, atol=0)

    def test_non_hf_targets_match_the_hf_forward_value(self):
        """Only the gradient reduction differs, never the forward."""
        outs = []
        for target in ("hf", "megatron", False):
            paddle.seed(4)
            emb = self._make(target)
            with paddle.no_grad():
                emb.weight.set_value(
                    paddle.full(emb.weight.shape, 0.25, dtype=emb.weight.dtype)
                )
            ids = paddle.to_tensor([[2, 2], [7, 9]], dtype="int64")
            outs.append(emb(ids).numpy().copy())
        np.testing.assert_array_equal(outs[0], outs[1])
        np.testing.assert_array_equal(outs[0], outs[2])

    def test_output_shape_keeps_the_index_shape(self):
        emb = self._make("hf")
        ids = paddle.to_tensor([[1, 3, 4], [3, 5, 6]], dtype="int64")
        self.assertEqual(emb(ids).shape, [2, 3, 8])


if __name__ == "__main__":
    unittest.main()
