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

"""Unit tests for the module-level helpers of ``dsa_attention``.

Single card only: every collaborator that needs a process group is patched
(``get_pg_size``, ``gather_from_sequence_parallel_region``), and the
accuracy-compatible branches of ``DSAIndexer.forward_before_topk`` /
``DSAttention.forward`` are driven as unbound functions against a stub
``self`` so no distributed initialization is required.

Surface under test:
  * ``_accuracy_compat_linear`` bias routing
  * ``_SteQKMatmul`` straight-through QK matmul: fp32 accumulation,
    broadcast-key gradient reduction, key dtype restore
  * ``_AccuracyCompatibleQKMatmul`` head-axis expand forward/backward
  * ``_AccuracyCompatibleSoftmax`` masked forward/backward
  * ``_unfused_dsa_attention`` accuracy-compatible MQA path, including the
    value-from-absorbed-key slice
  * ``_unfused_absorbed_dsa_attention`` in both alignment modes
  * ``_normalize_dsa_mask`` / ``_align_dsa_indexer_mask`` /
    ``_align_sp_aux_to_query`` layout alignment
  * ``DSAIndexer.forward_before_topk`` functional-linear projections
  * ``DSAttention.forward`` absorbed core in both alignment modes
  * frozen parameters of a shared MTP indexer
"""

import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import paddle
import paddle.nn.functional as F
from paddle.distributed.fleet.meta_parallel import LayerSpec

from paddlefleet.transformer.dsa_attention import (
    DSAIndexer,
    DSAIndexerSublayersSpec,
    DSAttention,
    DSAttentionSublayersSpec,
    _accuracy_compat_linear,
    _AccuracyCompatibleQKMatmul,
    _AccuracyCompatibleSoftmax,
    _align_dsa_indexer_mask,
    _align_sp_aux_to_query,
    _normalize_dsa_mask,
    _SteQKMatmul,
    _unfused_absorbed_dsa_attention,
    _unfused_dsa_attention,
    rotate_activation,
)
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import init_method_normal

MODULE = "paddlefleet.transformer.dsa_attention"


def _fake_all_gather(tensor, *args, **kwargs):
    """Stand in for a two-rank all-gather along axis 0 on a single card."""
    return paddle.concat([tensor, tensor], axis=0)


def _true(tensor):
    """Unwrap a 0-d paddle bool tensor into a Python bool."""
    return bool(tensor)


def _equal_all(left, right):
    """Exact equality; bf16 is compared through its lossless fp32 cast.

    ``paddle.equal_all`` has no bfloat16 kernel, and widening bfloat16 to
    float32 is exact, so the comparison stays bit-for-bit.
    """
    if left.dtype in (paddle.bfloat16, paddle.float16):
        left = left.cast("float32")
        right = right.cast("float32")
    return paddle.equal_all(left, right)


class _ProjectionStub(paddle.nn.Layer):
    """Minimal parallel-linear surface used by ``_accuracy_compat_linear``.

    ``forward`` counts its calls so a test can prove the production code
    went through ``paddle.nn.functional.linear`` instead of the layer's own
    autograd function.
    """

    def __init__(
        self,
        in_features,
        out_features,
        skip_bias_add=False,
        dtype="float32",
        **kwargs,
    ):
        super().__init__()
        self.weight = self.create_parameter(
            [in_features, out_features],
            dtype=dtype,
            default_initializer=paddle.nn.initializer.Normal(std=0.05),
        )
        self.bias = self.create_parameter(
            [out_features],
            dtype=dtype,
            is_bias=True,
            default_initializer=paddle.nn.initializer.Constant(0.25),
        )
        self.skip_bias_add = skip_bias_add
        self.forward_calls = 0

    def forward(self, x):
        self.forward_calls += 1
        bias = None if self.skip_bias_add else self.bias
        return F.linear(x, self.weight, bias), (
            self.bias if self.skip_bias_add else None
        )


class _LayerNormStub(paddle.nn.Layer):
    """LayerNorm stub accepting the indexer's keyword spelling."""

    def __init__(
        self, normalized_shape=None, epsilon=1e-5, dtype="float32", **kwargs
    ):
        super().__init__()
        self.epsilon = epsilon
        self.weight = self.create_parameter(
            [normalized_shape],
            dtype=dtype,
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = x.var(axis=-1, keepdim=True, unbiased=False)
        return (x - mean) / paddle.sqrt(var + self.epsilon) * self.weight


class TestAccuracyCompatLinear(unittest.TestCase):
    """``_accuracy_compat_linear`` must route the bias like the layer does."""

    def test_bias_is_fused_when_not_skipped(self):
        projection = _ProjectionStub(6, 4)
        x = paddle.randn([2, 3, 6])

        output, output_bias = _accuracy_compat_linear(projection, x)

        self.assertIsNone(output_bias)
        self.assertEqual(list(output.shape), [2, 3, 4])
        expected = F.linear(x, projection.weight, projection.bias)
        self.assertTrue(_true(_equal_all(output, expected)))
        self.assertEqual(projection.forward_calls, 0)

    def test_bias_is_returned_when_skipped(self):
        projection = _ProjectionStub(6, 4, skip_bias_add=True)
        x = paddle.randn([2, 3, 6])

        output, output_bias = _accuracy_compat_linear(projection, x)

        self.assertIs(output_bias, projection.bias)
        unbiased = F.linear(x, projection.weight)
        self.assertTrue(_true(_equal_all(output, unbiased)))
        fused = F.linear(x, projection.weight, projection.bias)
        self.assertFalse(_true(_equal_all(output, fused)))


class TestSteQKMatmul(unittest.TestCase):
    """Opaque QK matmul: fp32 accumulation plus MQA gradient reduction."""

    def test_forward_accumulates_in_fp32(self):
        q4 = paddle.randn([1, 2, 3, 4]).cast("bfloat16")
        k4 = paddle.randn([1, 2, 5, 4]).cast("bfloat16")
        scale = paddle.full([], 0.5, dtype="float32")

        scores = _SteQKMatmul.apply(q4, k4, scale)

        self.assertEqual(scores.dtype, paddle.float32)
        self.assertEqual(list(scores.shape), [1, 2, 3, 5])
        expected = (
            paddle.matmul(
                q4.cast("float32"),
                k4.cast("float32").transpose([0, 1, 3, 2]),
            )
            * 0.5
        )
        self.assertTrue(_true(_equal_all(scores, expected)))

    def test_backward_reduces_broadcast_key_and_restores_dtype(self):
        q4 = paddle.randn([1, 3, 5, 4], dtype="float32")
        k4 = paddle.randn([1, 1, 7, 4]).cast("bfloat16")
        k4.stop_gradient = False
        scale = paddle.full([], 0.25, dtype="float32")

        scores = _SteQKMatmul.apply(q4, k4, scale)
        grad = paddle.randn([1, 3, 5, 7], dtype="float32")
        paddle.autograd.backward([scores], [grad])

        self.assertIsNotNone(k4.grad)
        self.assertEqual(k4.grad.dtype, paddle.bfloat16)
        self.assertEqual(list(k4.grad.shape), [1, 1, 7, 4])
        expected = (
            (paddle.matmul(grad.transpose([0, 1, 3, 2]), q4) * 0.25)
            .sum(axis=1, keepdim=True)
            .cast("bfloat16")
        )
        self.assertTrue(_true(_equal_all(k4.grad, expected)))


class TestAccuracyCompatibleQKMatmul(unittest.TestCase):
    """Batched QK matmul that expands only the head axis of the key."""

    def setUp(self):
        self.query = paddle.randn([2, 3, 5, 4], dtype="float32")
        self.key = paddle.randn([2, 1, 4, 7], dtype="float32")
        self.expanded = self.key.expand([2, 3, 4, 7])

    def test_forward_expands_head_axis_only(self):
        scores = _AccuracyCompatibleQKMatmul.apply(self.query, self.key)

        self.assertEqual(list(scores.shape), [2, 3, 5, 7])
        expected = paddle.matmul(self.query, self.expanded)
        self.assertTrue(paddle.allclose(scores, expected, atol=1e-5))

    def test_backward_sums_key_gradient_over_heads(self):
        query = self.query.detach()
        key = self.key.detach()
        query.stop_gradient = False
        key.stop_gradient = False

        scores = _AccuracyCompatibleQKMatmul.apply(query, key)
        grad = paddle.randn([2, 3, 5, 7], dtype="float32")
        paddle.autograd.backward([scores], [grad])

        self.assertEqual(list(key.grad.shape), [2, 1, 4, 7])
        self.assertEqual(list(query.grad.shape), [2, 3, 5, 4])
        expected_key = paddle.sum(
            paddle.matmul(query.transpose([0, 1, 3, 2]), grad),
            axis=1,
            keepdim=True,
        )
        expected_query = paddle.matmul(
            grad, self.expanded.transpose([0, 1, 3, 2])
        )
        self.assertTrue(paddle.allclose(key.grad, expected_key, atol=1e-5))
        self.assertTrue(paddle.allclose(query.grad, expected_query, atol=1e-5))


class TestAccuracyCompatibleSoftmax(unittest.TestCase):
    """Masked softmax with positive zeros and a masked gradient."""

    def setUp(self):
        self.sq, self.sk = 3, 4
        columns = paddle.arange(self.sk).reshape([1, 1, self.sk])
        rows = paddle.arange(self.sq).reshape([1, self.sq, 1])
        self.valid = (columns <= rows).expand([2, self.sq, self.sk])
        self.logits = paddle.randn([2, self.sq, self.sk], dtype="float32")

    def test_forward_zeroes_masked_positions(self):
        probabilities = _AccuracyCompatibleSoftmax.apply(
            self.logits, self.valid
        )

        self.assertEqual(list(probabilities.shape), [2, self.sq, self.sk])
        dense = F.softmax(self.logits, axis=-1)
        expected = paddle.where(
            self.valid, dense, paddle.zeros_like(probabilities)
        )
        self.assertTrue(_true(_equal_all(probabilities, expected)))
        masked = paddle.where(
            self.valid, paddle.zeros_like(probabilities), probabilities
        )
        self.assertEqual(float(masked.abs().sum()), 0.0)
        # The masked softmax does not renormalize: row 0 keeps a single
        # column, so its mass is that column's dense probability, not one.
        kept = float(probabilities[0, 0].sum())
        self.assertAlmostEqual(kept, float(dense[0, 0, 0]), places=6)
        self.assertLess(kept, 1.0)

    def test_backward_applies_the_masked_softmax_jacobian(self):
        logits = self.logits.detach()
        logits.stop_gradient = False

        probabilities = _AccuracyCompatibleSoftmax.apply(logits, self.valid)
        grad = paddle.randn([2, self.sq, self.sk], dtype="float32")
        paddle.autograd.backward([probabilities], [grad])

        zeros = paddle.zeros_like(probabilities)
        reference = paddle.where(self.valid, F.softmax(logits, axis=-1), zeros)
        expected = reference * (
            grad - paddle.sum(grad * reference, axis=-1, keepdim=True)
        )
        expected = paddle.where(self.valid, expected, zeros)
        self.assertTrue(paddle.allclose(logits.grad, expected, atol=1e-6))
        leaked = paddle.where(self.valid, zeros, logits.grad)
        self.assertEqual(float(leaked.abs().sum()), 0.0)


def _causal_float_mask(seq):
    """``[1, 1, seq, seq]`` additive causal mask."""
    return paddle.triu(
        paddle.full([seq, seq], float("-inf"), dtype="float32"),
        diagonal=1,
    ).reshape([1, 1, seq, seq])


def _mqa_absorbed_reference(query, key, mask, v_head_dim, softmax_scale):
    """Reference for the accuracy-compatible MQA path of the unfused core.

    Mirrors the documented contract: fp32 scores, masked softmax, and a value
    taken from the leading ``v_head_dim`` channels of the absorbed key.
    """
    b, seq, heads, qk_head_dim = (int(dim) for dim in query.shape)
    q = query.transpose([0, 2, 1, 3]).reshape([b * heads, seq, qk_head_dim])
    k = (
        key.expand([b, seq, heads, qk_head_dim])
        .transpose([0, 2, 1, 3])
        .reshape([b * heads, seq, qk_head_dim])
    )
    scores = paddle.bmm(q, k.transpose([0, 2, 1])) * softmax_scale
    scores = scores + mask.expand([b, heads, seq, seq]).reshape(
        [b * heads, seq, seq]
    )
    probabilities = paddle.where(
        paddle.isfinite(scores),
        F.softmax(scores, axis=-1),
        paddle.zeros_like(scores),
    )
    v = (
        key[..., :v_head_dim]
        .expand([b, seq, heads, v_head_dim])
        .transpose([0, 2, 1, 3])
        .reshape([b * heads, seq, v_head_dim])
    )
    return (
        paddle.bmm(probabilities, v)
        .reshape([b, heads, seq, v_head_dim])
        .transpose([0, 2, 1, 3])
        .reshape([b, seq, heads * v_head_dim])
    )


class TestUnfusedDSAAttentionAccuracyCompatible(unittest.TestCase):
    """Accuracy-compatible MQA path of ``_unfused_dsa_attention``."""

    def setUp(self):
        self.b, self.seq, self.heads = 1, 4, 2
        self.v_head_dim, self.rope = 6, 2
        self.qk_head_dim = self.v_head_dim + self.rope
        self.softmax_scale = 0.5
        self.query = paddle.randn(
            [self.b, self.seq, self.heads, self.qk_head_dim], dtype="float32"
        )
        self.key = paddle.randn(
            [self.b, self.seq, 1, self.qk_head_dim], dtype="float32"
        )
        self.mask = _causal_float_mask(self.seq)
        self.value_operand = paddle.zeros(
            [self.b, self.seq, 1, self.v_head_dim], dtype="float32"
        )

    def test_value_is_sliced_out_of_the_absorbed_key(self):
        noise = paddle.randn(
            [self.b, self.seq, 1, self.v_head_dim], dtype="float32"
        )

        from_zeros = _unfused_dsa_attention(
            self.query,
            self.key,
            self.value_operand,
            self.mask,
            self.softmax_scale,
            use_accuracy_compatible=True,
            tensor_parallel_size=2,
        )
        from_noise = _unfused_dsa_attention(
            self.query,
            self.key,
            noise,
            self.mask,
            self.softmax_scale,
            use_accuracy_compatible=True,
            tensor_parallel_size=2,
        )

        self.assertEqual(
            list(from_zeros.shape),
            [self.b, self.seq, self.heads * self.v_head_dim],
        )
        self.assertTrue(_true(_equal_all(from_zeros, from_noise)))
        expected = _mqa_absorbed_reference(
            self.query,
            self.key,
            self.mask,
            self.v_head_dim,
            self.softmax_scale,
        )
        self.assertTrue(paddle.allclose(from_zeros, expected, atol=1e-6))
        self.assertTrue(_true(paddle.any(from_zeros != 0)))

    def test_single_card_tp_keeps_the_same_numerics(self):
        tp2 = _unfused_dsa_attention(
            self.query,
            self.key,
            self.value_operand,
            self.mask,
            self.softmax_scale,
            use_accuracy_compatible=True,
            tensor_parallel_size=2,
        )
        tp1 = _unfused_dsa_attention(
            self.query,
            self.key,
            self.value_operand,
            self.mask,
            self.softmax_scale,
            use_accuracy_compatible=True,
            tensor_parallel_size=1,
        )

        self.assertTrue(_true(_equal_all(tp1, tp2)))

    def test_straight_through_gradient_reaches_the_key(self):
        key = self.key.detach()
        key.stop_gradient = False

        output = _unfused_dsa_attention(
            self.query,
            key,
            self.value_operand,
            self.mask,
            self.softmax_scale,
            use_accuracy_compatible=True,
            tensor_parallel_size=1,
        )
        output.sum().backward()

        self.assertIsNotNone(key.grad)
        self.assertEqual(list(key.grad.shape), list(key.shape))
        self.assertTrue(_true(paddle.any(key.grad != 0)))
        self.assertTrue(_true(paddle.all(paddle.isfinite(key.grad))))


class TestUnfusedAbsorbedDSAAttention(unittest.TestCase):
    """Absorbed-MLA sparse attention, both alignment modes."""

    def setUp(self):
        self.b, self.seq, self.heads = 2, 4, 2
        self.rank, self.v_out = 6, 3
        self.softmax_scale = 0.25
        self.query = paddle.randn(
            [self.b, self.seq, self.heads, self.rank], dtype="float32"
        )
        self.key = paddle.randn(
            [self.b, self.seq, 1, self.rank], dtype="float32"
        )
        self.value = paddle.randn(
            [self.b, self.seq, 1, self.rank], dtype="float32"
        )
        self.v_up = paddle.randn(
            [self.heads, self.rank, self.v_out], dtype="float32"
        )

    def _reference(self, mask, use_accuracy_compatible):
        expanded = self.key.transpose([0, 2, 3, 1]).expand(
            [self.b, self.heads, self.rank, self.seq]
        )
        scores = paddle.matmul(self.query.transpose([0, 2, 1, 3]), expanded)
        scores = scores * self.softmax_scale
        if mask is not None:
            scores = scores + mask
        dense = F.softmax(scores, axis=-1)
        if use_accuracy_compatible:
            dense = paddle.where(
                paddle.isfinite(scores), dense, paddle.zeros_like(dense)
            )
        latent = paddle.matmul(dense, self.value.transpose([0, 2, 1, 3]))
        projected = paddle.einsum("bhsr,hrd->bshd", latent, self.v_up)
        return projected.reshape([self.b, self.seq, self.heads * self.v_out])

    def test_matches_reference_in_both_modes(self):
        for use_accuracy_compatible in (False, True):
            for with_mask in (False, True):
                mask = _causal_float_mask(self.seq) if with_mask else None
                with self.subTest(
                    accuracy_compatible=use_accuracy_compatible,
                    masked=with_mask,
                ):
                    output = _unfused_absorbed_dsa_attention(
                        self.query,
                        self.key,
                        self.value,
                        self.v_up,
                        mask,
                        self.softmax_scale,
                        use_accuracy_compatible=use_accuracy_compatible,
                    )
                    self.assertEqual(
                        list(output.shape),
                        [self.b, self.seq, self.heads * self.v_out],
                    )
                    expected = self._reference(mask, use_accuracy_compatible)
                    self.assertTrue(
                        paddle.allclose(output, expected, atol=1e-5)
                    )


class TestNormalizeAndAlignIndexerMask(unittest.TestCase):
    """Mask normalization and sequence-parallel last-dim alignment."""

    def test_normalize_squeezes_singleton_head_axis(self):
        self.assertIsNone(_normalize_dsa_mask(None))
        mask = paddle.zeros([2, 1, 4, 4], dtype="float32")
        self.assertEqual(list(_normalize_dsa_mask(mask).shape), [2, 4, 4])
        with self.assertRaises(AssertionError):
            _normalize_dsa_mask(paddle.zeros([2, 3, 4, 4], dtype="float32"))

    def test_none_mask_returns_none(self):
        self.assertIsNone(_align_dsa_indexer_mask(None, 8))

    def test_matching_last_dim_is_passed_through(self):
        mask = paddle.zeros([2, 4, 4], dtype="float32")
        self.assertIs(_align_dsa_indexer_mask(mask, 4), mask)

    def test_mismatch_without_sequence_parallel_returns_none(self):
        mask = paddle.zeros([2, 4, 2], dtype="float32")
        self.assertIsNone(_align_dsa_indexer_mask(mask, 4))
        self.assertIsNone(
            _align_dsa_indexer_mask(
                mask,
                4,
                sequence_parallel=True,
                tp_group=SimpleNamespace(nranks=1),
            )
        )
        self.assertIsNone(
            _align_dsa_indexer_mask(
                mask,
                4,
                sequence_parallel=False,
                tp_group=SimpleNamespace(nranks=2),
            )
        )

    def _gathered(self, mask, score_sk):
        with patch(
            MODULE + ".gather_from_sequence_parallel_region", _fake_all_gather
        ):
            return _align_dsa_indexer_mask(
                mask,
                score_sk,
                sequence_parallel=True,
                tp_group=SimpleNamespace(nranks=2),
            )

    def test_two_dim_mask_is_gathered_on_the_last_axis(self):
        mask = paddle.arange(8, dtype="float32").reshape([4, 2])

        aligned = self._gathered(mask, 4)

        self.assertEqual(list(aligned.shape), [4, 4])
        expected = paddle.concat([mask, mask], axis=-1)
        self.assertTrue(_true(_equal_all(aligned, expected)))

    def test_three_dim_mask_is_gathered_on_the_last_axis(self):
        mask = paddle.arange(12, dtype="float32").reshape([1, 6, 2])

        aligned = self._gathered(mask, 4)

        self.assertEqual(list(aligned.shape), [1, 6, 4])
        expected = paddle.concat([mask, mask], axis=-1)
        self.assertTrue(_true(_equal_all(aligned, expected)))

    def test_unsupported_rank_returns_none(self):
        self.assertIsNone(self._gathered(paddle.zeros([2], "float32"), 4))


class TestAlignSpAuxToQuery(unittest.TestCase):
    """Lifting MLA auxiliaries onto the batch-first query layout."""

    BATCH, SEQ = 3, 4

    def setUp(self):
        self.query = paddle.zeros([self.BATCH, self.SEQ, 2, 2], dtype="float32")

    def _align(self, tensor, query=None):
        with patch(
            MODULE + ".gather_from_sequence_parallel_region", _fake_all_gather
        ):
            return _align_sp_aux_to_query(
                tensor, self.query if query is None else query
            )

    def test_none_tensor_and_non_4d_query_are_passed_through(self):
        tensor = paddle.zeros([1, 2, 3], dtype="float32")
        self.assertIsNone(self._align(None))
        self.assertIs(
            self._align(tensor, query=paddle.zeros([2, 3], dtype="float32")),
            tensor,
        )

    def test_3d_batch_first_is_passed_through(self):
        tensor = paddle.zeros([self.BATCH, self.SEQ, 5], dtype="float32")
        self.assertIs(self._align(tensor), tensor)

    def test_3d_seq_first_is_transposed(self):
        tensor = paddle.arange(
            self.SEQ * self.BATCH * 5, dtype="float32"
        ).reshape([self.SEQ, self.BATCH, 5])

        aligned = self._align(tensor)

        self.assertEqual(list(aligned.shape), [self.BATCH, self.SEQ, 5])
        expected = tensor.transpose([1, 0, 2])
        self.assertTrue(_true(_equal_all(aligned, expected)))

    def test_3d_batch_first_shard_is_gathered(self):
        tensor = paddle.arange(self.BATCH * 2 * 5, dtype="float32").reshape(
            [self.BATCH, 2, 5]
        )

        aligned = self._align(tensor)

        self.assertEqual(list(aligned.shape), [self.BATCH, self.SEQ, 5])
        expected = paddle.concat([tensor, tensor], axis=1)
        self.assertTrue(_true(_equal_all(aligned, expected)))

    def test_3d_seq_first_shard_is_gathered_then_transposed(self):
        tensor = paddle.arange(2 * self.BATCH * 5, dtype="float32").reshape(
            [2, self.BATCH, 5]
        )

        aligned = self._align(tensor)

        self.assertEqual(list(aligned.shape), [self.BATCH, self.SEQ, 5])
        expected = paddle.concat([tensor, tensor], axis=0).transpose([1, 0, 2])
        self.assertTrue(_true(_equal_all(aligned, expected)))

    def test_3d_shard_that_does_not_gather_to_seq_is_returned_raw(self):
        tensor = paddle.zeros([1, self.BATCH, 5], dtype="float32")

        aligned = self._align(tensor)

        self.assertEqual(list(aligned.shape), [2, self.BATCH, 5])

    def test_3d_unrecognised_layout_is_passed_through(self):
        tensor = paddle.zeros([5, 7, 5], dtype="float32")
        self.assertIs(self._align(tensor), tensor)

    def test_4d_batch_first_is_passed_through(self):
        tensor = paddle.zeros([self.BATCH, self.SEQ, 1, 2], dtype="float32")
        self.assertIs(self._align(tensor), tensor)

    def test_4d_seq_first_is_transposed(self):
        tensor = paddle.arange(
            self.SEQ * self.BATCH * 2, dtype="float32"
        ).reshape([self.SEQ, self.BATCH, 1, 2])

        aligned = self._align(tensor)

        self.assertEqual(list(aligned.shape), [self.BATCH, self.SEQ, 1, 2])
        expected = tensor.transpose([1, 0, 2, 3])
        self.assertTrue(_true(_equal_all(aligned, expected)))

    def test_4d_batch_first_shard_is_gathered(self):
        tensor = paddle.arange(self.BATCH * 2 * 2, dtype="float32").reshape(
            [self.BATCH, 2, 1, 2]
        )

        aligned = self._align(tensor)

        self.assertEqual(list(aligned.shape), [self.BATCH, self.SEQ, 1, 2])
        expected = paddle.concat([tensor, tensor], axis=1)
        self.assertTrue(_true(_equal_all(aligned, expected)))

    def test_4d_seq_first_shard_is_gathered_then_transposed(self):
        tensor = paddle.arange(2 * self.BATCH * 2, dtype="float32").reshape(
            [2, self.BATCH, 1, 2]
        )

        aligned = self._align(tensor)

        self.assertEqual(list(aligned.shape), [self.BATCH, self.SEQ, 1, 2])
        expected = paddle.concat([tensor, tensor], axis=0).transpose(
            [1, 0, 2, 3]
        )
        self.assertTrue(_true(_equal_all(aligned, expected)))

    def test_4d_shard_that_does_not_gather_to_seq_is_returned_raw(self):
        tensor = paddle.zeros([1, self.BATCH, 1, 2], dtype="float32")

        aligned = self._align(tensor)

        self.assertEqual(list(aligned.shape), [2, self.BATCH, 1, 2])

    def test_4d_full_seq_rope_is_sliced_to_the_local_window(self):
        tensor = paddle.arange(
            self.BATCH * 2 * self.SEQ * 2, dtype="float32"
        ).reshape([self.BATCH, 2 * self.SEQ, 1, 2])

        aligned = self._align(tensor)

        self.assertEqual(list(aligned.shape), [self.BATCH, self.SEQ, 1, 2])
        expected = tensor[:, : self.SEQ]
        self.assertTrue(_true(_equal_all(aligned, expected)))

    def test_4d_full_seq_not_divisible_is_passed_through(self):
        tensor = paddle.zeros([self.BATCH, 6, 1, 2], dtype="float32")
        self.assertIs(self._align(tensor), tensor)

    def test_2d_aux_is_passed_through(self):
        tensor = paddle.zeros([self.BATCH, self.SEQ], dtype="float32")
        self.assertIs(self._align(tensor), tensor)


class TestIndexerAccuracyCompatibleProjections(unittest.TestCase):
    """``forward_before_topk`` must use functional linear under TP alignment.

    Driven as an unbound method against a stub ``self``; RoPE frequencies are
    zero so the rotation is the identity and the projections stay visible in
    the output.
    """

    def setUp(self):
        self.b, self.seq = 1, 4
        self.hidden, self.q_rank = 8, 6
        self.n_heads, self.head_dim, self.rope_dim = 2, 8, 4
        self.freqs = paddle.zeros(
            [1, self.seq, 1, self.rope_dim], dtype="float32"
        )
        self.stub = SimpleNamespace(
            config=SimpleNamespace(
                sequence_parallel=False,
                rope_type="rope",
                use_accuracy_compatible=True,
                dsa_indexer_rope_fusion=False,
                dsa_indexer_rotary_interleaved=False,
                high_precision_rope=False,
            ),
            pg_collection=SimpleNamespace(tp=None, cp=None),
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_dim,
            softmax_scale=self.head_dim**-0.5,
            use_fast_hadamard=False,
            k_norm=_LayerNormStub(
                normalized_shape=self.head_dim, dtype="bfloat16"
            ),
            rotary_pos_emb=lambda seq_len, packed_seq=False: self.freqs,
            wq_b=_ProjectionStub(
                self.q_rank,
                self.n_heads * self.head_dim,
                dtype="bfloat16",
            ),
            wk=_ProjectionStub(self.hidden, self.head_dim, dtype="bfloat16"),
            weights_proj=_ProjectionStub(
                self.hidden, self.n_heads, dtype="bfloat16"
            ),
        )
        self.stub._apply_rope = MethodType(DSAIndexer._apply_rope, self.stub)
        self.x = paddle.randn([self.b, self.seq, self.hidden]).cast("bfloat16")
        self.qr = paddle.randn([self.b, self.seq, self.q_rank]).cast("bfloat16")

    def _forward(self):
        with patch(MODULE + ".get_pg_size", return_value=2):
            return DSAIndexer.forward_before_topk(self.stub, self.x, self.qr)

    def test_q_and_k_come_from_functional_linear(self):
        q, k, _ = self._forward()

        self.assertEqual(
            list(q.shape), [self.b, self.seq, self.n_heads, self.head_dim]
        )
        self.assertEqual(list(k.shape), [self.b, self.seq, self.head_dim])
        self.assertEqual(q.dtype, paddle.bfloat16)
        self.assertEqual(k.dtype, paddle.bfloat16)
        self.assertEqual(self.stub.wq_b.forward_calls, 0)
        self.assertEqual(self.stub.wk.forward_calls, 0)

        q_projected = F.linear(
            self.qr, self.stub.wq_b.weight, self.stub.wq_b.bias
        ).reshape([self.b, self.seq, self.n_heads, self.head_dim])
        expected_q = rotate_activation(
            self.stub._apply_rope(q_projected, self.freqs, 1.0)
        )
        self.assertTrue(_true(_equal_all(q, expected_q)))

        k_projected = self.stub.k_norm(
            F.linear(self.x, self.stub.wk.weight, self.stub.wk.bias)
        )
        expected_k = rotate_activation(
            self.stub._apply_rope(
                k_projected.unsqueeze(2), self.freqs, 1.0
            ).squeeze(2)
        )
        self.assertTrue(_true(_equal_all(k, expected_k)))

    def test_single_card_keeps_the_projection_layer_call(self):
        """Without TP the gate stays on the layer's own autograd path."""
        with patch(MODULE + ".get_pg_size", return_value=1):
            q, _, weights = DSAIndexer.forward_before_topk(
                self.stub, self.x, self.qr
            )

        self.assertEqual(self.stub.wq_b.forward_calls, 1)
        self.assertEqual(self.stub.wk.forward_calls, 1)
        self.assertEqual(self.stub.weights_proj.forward_calls, 1)
        self.assertEqual(
            list(q.shape), [self.b, self.seq, self.n_heads, self.head_dim]
        )
        self.assertEqual(list(weights.shape), [self.b, self.seq, self.n_heads])

    def test_weights_are_projected_and_scaled(self):
        _, _, weights = self._forward()

        self.assertEqual(list(weights.shape), [self.b, self.seq, self.n_heads])
        self.assertEqual(self.stub.weights_proj.forward_calls, 0)
        projected = F.linear(
            self.x,
            self.stub.weights_proj.weight,
            self.stub.weights_proj.bias,
        )
        expected = projected * (self.n_heads**-0.5) * self.stub.softmax_scale
        self.assertTrue(_true(_equal_all(weights, expected)))


def _forward_stub(
    *,
    sequence_parallel=False,
    tp_nranks=None,
    use_accuracy_compatible=True,
    index_share=False,
    skip_topk=False,
    indexer=None,
    softmax_scale=0.5,
):
    """Build a stub ``self`` for ``DSAttention.forward``."""
    tp = None if tp_nranks is None else SimpleNamespace(nranks=tp_nranks)
    stub = SimpleNamespace(
        config=SimpleNamespace(
            sequence_parallel=sequence_parallel,
            use_accuracy_compatible=use_accuracy_compatible,
            qk_rope_head_dim=2,
        ),
        pg_collection=SimpleNamespace(tp=tp),
        index_share=index_share,
        skip_topk=skip_topk,
        source_layer=0,
        layer_number=1,
        softmax_scale=softmax_scale,
        training=False,
        indexer=indexer,
        dsa_indexer_loss_coeff=0.0,
        retain_indexer_loss_graph=False,
    )
    stub._get_index_share_topk_holder = MethodType(
        DSAttention._get_index_share_topk_holder, stub
    )
    stub._lookup_index_share_topk = MethodType(
        DSAttention._lookup_index_share_topk, stub
    )
    stub._publish_index_share_topk = MethodType(
        DSAttention._publish_index_share_topk, stub
    )
    return stub


def _causal_topk_indices(batch, seq, topk):
    """Causal top-k indices, padded with the current position."""
    rows = []
    for position in range(seq):
        chosen = list(range(max(0, position - topk + 1), position + 1))
        chosen += [position] * (topk - len(chosen))
        rows.append(chosen)
    return paddle.tile(
        paddle.to_tensor(rows, dtype="int64").unsqueeze(0), [batch, 1, 1]
    )


def _expected_combined_mask(batch, seq, topk_indices):
    """Sparse index mask merged with the causal mask, as forward builds it."""
    index_mask = paddle.full([batch, seq, seq], float("-inf"), dtype="float32")
    index_mask = paddle.put_along_axis(
        index_mask,
        topk_indices,
        paddle.zeros(topk_indices.shape, dtype="float32"),
        axis=-1,
    )
    causal = paddle.triu(
        paddle.full([seq, seq], float("-inf"), dtype="float32"), diagonal=1
    )
    return (index_mask + causal.unsqueeze(0)).unsqueeze(1)


class TestDSAttentionForwardMask(unittest.TestCase):
    """Mask construction and shared top-k plumbing of ``forward``."""

    def setUp(self):
        self.heads, self.qk_head_dim, self.v_head_dim = 2, 4, 3
        self.topk = 2

    def _tensors(self, batch, seq):
        return (
            paddle.randn(
                [batch, seq, self.heads, self.qk_head_dim], dtype="float32"
            ),
            paddle.randn(
                [batch, seq, self.heads, self.qk_head_dim], dtype="float32"
            ),
            paddle.randn(
                [batch, seq, self.heads, self.v_head_dim], dtype="float32"
            ),
        )

    def test_sequence_parallel_gathers_the_indexer_sequence(self):
        batch, local = 1, 2
        seq = local * 2
        indices = _causal_topk_indices(batch, seq, self.topk)
        stub = _forward_stub(
            sequence_parallel=True,
            tp_nranks=2,
            index_share=True,
            skip_topk=True,
        )
        holder = {stub.source_layer: indices}
        x = paddle.randn([local, batch, 6]).cast("bfloat16")
        qr = paddle.randn([local, batch, 6]).cast("bfloat16")
        query, key, value = self._tensors(batch, seq)

        with patch(MODULE + ".get_pg_size", return_value=2):
            output = DSAttention.forward(
                stub,
                query,
                key,
                value,
                None,
                x=x,
                qr=qr,
                dsa_topk_holder=holder,
            )

        self.assertEqual(
            list(output.shape), [batch, seq, self.heads * self.v_head_dim]
        )
        expected = _unfused_dsa_attention(
            query,
            key,
            value,
            _expected_combined_mask(batch, seq, indices),
            stub.softmax_scale,
            use_accuracy_compatible=True,
            tensor_parallel_size=2,
        )
        self.assertTrue(paddle.allclose(output, expected, atol=1e-6))

    def test_unalignable_attention_mask_falls_back_to_causal(self):
        batch, seq = 1, 4
        indices = _causal_topk_indices(batch, seq, self.topk)
        stub = _forward_stub(index_share=True, skip_topk=True)
        holder = {stub.source_layer: indices}
        x = paddle.randn([batch, seq, 6]).cast("bfloat16")
        qr = paddle.randn([batch, seq, 6]).cast("bfloat16")
        query, key, value = self._tensors(batch, seq)
        attention_mask = paddle.zeros([batch, 1, seq, seq // 2], dtype="bool")

        with patch(MODULE + ".get_pg_size", return_value=1):
            output = DSAttention.forward(
                stub,
                query,
                key,
                value,
                attention_mask,
                x=x,
                qr=qr,
                dsa_topk_holder=holder,
            )

        expected = _unfused_dsa_attention(
            query,
            key,
            value,
            _expected_combined_mask(batch, seq, indices),
            stub.softmax_scale,
            use_accuracy_compatible=True,
            tensor_parallel_size=1,
        )
        self.assertTrue(paddle.allclose(output, expected, atol=1e-6))

    def test_index_share_publishes_topk_and_passes_the_causal_mask(self):
        batch, seq = 1, 4
        indices = _causal_topk_indices(batch, seq, self.topk)
        seen = {}

        class _IndexerStub:
            index_topk = 2

            def forward(self, x, qr, mask):
                seen["mask"] = mask
                return None, indices

        stub = _forward_stub(index_share=True, indexer=_IndexerStub())
        holder = {}
        x = paddle.randn([batch, seq, 6]).cast("bfloat16")
        qr = paddle.randn([batch, seq, 6]).cast("bfloat16")
        query, key, value = self._tensors(batch, seq)

        with patch(MODULE + ".get_pg_size", return_value=1):
            output = DSAttention.forward(
                stub,
                query,
                key,
                value,
                None,
                x=x,
                qr=qr,
                dsa_topk_holder=holder,
            )

        self.assertIs(holder[stub.source_layer], indices)
        self.assertEqual(list(seen["mask"].shape), [1, 1, seq, seq])
        causal = paddle.triu(
            paddle.full([seq, seq], float("-inf"), dtype="float32"),
            diagonal=1,
        )
        self.assertTrue(
            _true(_equal_all(seen["mask"], causal.unsqueeze(0).unsqueeze(0)))
        )
        self.assertEqual(
            list(output.shape), [batch, seq, self.heads * self.v_head_dim]
        )


class TestDSAttentionForwardAbsorbedCore(unittest.TestCase):
    """Absorbed-core branch of ``DSAttention.forward``."""

    def setUp(self):
        self.batch, self.seq, self.heads = 1, 4, 2
        self.kv_rank, self.rope, self.v_out = 6, 2, 3
        self.nope = self.kv_rank - self.rope
        self.indices = _causal_topk_indices(self.batch, self.seq, 2)
        self.mask = _expected_combined_mask(self.batch, self.seq, self.indices)
        self.x = paddle.randn([self.batch, self.seq, 6]).cast("bfloat16")
        self.qr = paddle.randn([self.batch, self.seq, 6]).cast("bfloat16")
        self.placeholder = paddle.zeros(
            [self.batch, self.seq, 1, 1], dtype="float32"
        )
        self.kv_compressed = paddle.randn(
            [self.batch, self.seq, self.kv_rank], dtype="float32"
        )
        self.v_b_proj_weight = paddle.randn(
            [self.heads, self.v_out, self.kv_rank], dtype="float32"
        )

    def _stub(self, use_accuracy_compatible):
        stub = _forward_stub(
            use_accuracy_compatible=use_accuracy_compatible,
            index_share=True,
            skip_topk=True,
        )
        stub.dsa_topk_holder = {stub.source_layer: self.indices}
        return stub

    def test_absorbed_query_is_built_and_projected_with_bmm(self):
        stub = self._stub(True)
        query = paddle.randn(
            [self.batch, self.seq, self.heads, self.kv_rank],
            dtype="float32",
        )
        k_abs_weight = paddle.randn(
            [self.heads, self.nope, self.kv_rank], dtype="float32"
        )
        k_pos_emb = paddle.randn(
            [self.batch, self.seq, self.rope], dtype="float32"
        )

        with patch(MODULE + ".get_pg_size", return_value=2):
            output = DSAttention.forward(
                stub,
                query,
                self.placeholder,
                self.placeholder,
                None,
                x=self.x,
                qr=self.qr,
                dsa_topk_holder=stub.dsa_topk_holder,
                kv_compressed=self.kv_compressed,
                k_pos_emb=k_pos_emb,
                v_b_proj_weight=self.v_b_proj_weight,
                k_abs_weight=k_abs_weight,
            )

        self.assertEqual(
            list(output.shape),
            [self.batch, self.seq, self.heads * self.v_out],
        )
        self.assertTrue(_true(paddle.all(paddle.isfinite(output))))
        self.assertTrue(_true(paddle.any(output != 0)))

        q_nope = query[..., : self.nope]
        q_pe = query[..., self.nope :]
        qn3 = q_nope.reshape(
            [self.batch * self.seq, self.heads, self.nope]
        ).transpose([1, 0, 2])
        q_absorbed = paddle.concat(
            [
                paddle.bmm(qn3, k_abs_weight)
                .transpose([1, 0, 2])
                .reshape([self.batch, self.seq, self.heads, self.kv_rank]),
                q_pe,
            ],
            axis=-1,
        )
        latent = self.kv_compressed + (self.kv_compressed * 0)
        k_latent = latent.unsqueeze(2)
        key_abs = paddle.concat([k_latent, k_pos_emb.unsqueeze(2)], axis=-1)
        latent_flat = _unfused_dsa_attention(
            q_absorbed,
            key_abs,
            paddle.zeros(k_latent.shape, dtype=k_latent.dtype),
            self.mask,
            stub.softmax_scale,
            use_accuracy_compatible=True,
            tensor_parallel_size=2,
        )
        latent_out = latent_flat.reshape(
            [self.batch, self.seq, self.heads, self.kv_rank]
        )
        expected = (
            paddle.bmm(
                latent_out.transpose([2, 0, 1, 3]).reshape(
                    [self.heads, self.batch * self.seq, self.kv_rank]
                ),
                self.v_b_proj_weight.transpose([0, 2, 1]),
            )
            .reshape([self.heads, self.batch, self.seq, -1])
            .transpose([1, 2, 0, 3])
            .reshape([self.batch, self.seq, self.heads * self.v_out])
        )
        self.assertTrue(paddle.allclose(output, expected, atol=1e-6))

    def test_default_mode_keeps_the_latent_value_and_einsum_projection(self):
        stub = self._stub(False)
        q_absorbed = paddle.randn(
            [
                self.batch,
                self.seq,
                self.heads,
                self.kv_rank + self.rope,
            ],
            dtype="float32",
        )
        k_pos_emb = paddle.randn(
            [self.batch, self.seq, 1, self.rope], dtype="float32"
        )

        with patch(MODULE + ".get_pg_size", return_value=1):
            output = DSAttention.forward(
                stub,
                q_absorbed,
                self.placeholder,
                self.placeholder,
                None,
                x=self.x,
                qr=self.qr,
                kv_compressed=self.kv_compressed,
                k_pos_emb=k_pos_emb,
                q_absorbed=q_absorbed,
                v_b_proj_weight=self.v_b_proj_weight,
            )

        k_latent = self.kv_compressed.unsqueeze(2)
        key_abs = paddle.concat([k_latent, k_pos_emb], axis=-1)
        latent_flat = _unfused_dsa_attention(
            q_absorbed,
            key_abs,
            k_latent,
            self.mask,
            stub.softmax_scale,
            use_accuracy_compatible=False,
            tensor_parallel_size=1,
        )
        latent_out = latent_flat.reshape(
            [self.batch, self.seq, self.heads, self.kv_rank]
        )
        expected = paddle.einsum(
            "bshc,hdc->bshd", latent_out, self.v_b_proj_weight
        ).reshape([self.batch, self.seq, self.heads * self.v_out])

        self.assertEqual(
            list(output.shape),
            [self.batch, self.seq, self.heads * self.v_out],
        )
        # A zero dummy value would collapse the output; the latent value is
        # what the default path must keep.
        self.assertTrue(_true(paddle.any(output != 0)))
        self.assertTrue(paddle.allclose(output, expected, atol=1e-6))


def _dsa_config():
    """Minimal DSA-capable config for building a real ``DSAttention``."""
    config = TransformerConfig(
        num_hidden_layers=2, hidden_size=16, num_attention_heads=2
    )
    config.num_key_value_heads = 2
    config.head_dim = 8
    config.q_lora_rank = 8
    config.kv_lora_rank = 8
    config.qk_nope_head_dim = 4
    config.qk_rope_head_dim = 4
    config.v_head_dim = 8
    config.multi_latent_attention = True
    config.rope_type = "rope"
    config.rope_theta = 10000.0
    config.rotary_interleaved = False
    config.rotary_percent = 1.0
    config.apply_rope_fusion = False
    config.dsa_index_n_heads = 2
    config.dsa_index_head_dim = 8
    config.dsa_index_topk = 2
    config.dsa_indexer_loss_coeff = None
    config.dsa_indexer_rotary_interleaved = False
    config.dsa_index_share_for_mtp_iteration = True
    config.num_nextn_predict_layers = 1
    config.init_method = init_method_normal(0.02)
    config.rms_norm_eps = 1e-5
    config.recompute_granularity = None
    config.sequence_parallel = False
    config.softmax_scale = None
    return config


class TestSharedMtpIndexerConstruction(unittest.TestCase):
    """A shared MTP layer still owns a dormant, frozen indexer."""

    def test_shared_mtp_indexer_is_built_and_frozen(self):
        spec = DSAttentionSublayersSpec(
            indexer=LayerSpec(
                layer=DSAIndexer,
                sublayers_spec=DSAIndexerSublayersSpec(
                    linear_wq_b=_ProjectionStub,
                    linear_wk=_ProjectionStub,
                    k_norm=_LayerNormStub,
                    linear_weights_proj=_ProjectionStub,
                ),
            )
        )

        model = DSAttention(
            config=_dsa_config(),
            sublayers_spec=spec,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            softmax_scale=0.5,
            is_mtp_layer=True,
            pg_collection=SimpleNamespace(tp=None, cp=None),
        )

        self.assertTrue(model.skip_topk)
        self.assertIsNotNone(model.indexer)
        parameters = list(model.indexer.parameters())
        self.assertGreater(len(parameters), 0)
        self.assertTrue(
            all(parameter.stop_gradient for parameter in parameters)
        )


if __name__ == "__main__":
    unittest.main()
