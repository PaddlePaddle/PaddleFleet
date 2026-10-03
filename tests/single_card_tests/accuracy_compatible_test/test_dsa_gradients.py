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

"""Numerical and gradient contracts for the unfused DSA primitives."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import paddle

from paddlefleet.transformer import dsa_attention as dsa
from tests.single_card_tests.accuracy_compatible_test._assertions import (
    assert_bitwise_equal,
)


def tensor(values, dtype="float32"):
    return paddle.to_tensor(values, dtype=dtype, stop_gradient=False)


def integers(shape):
    return (np.arange(np.prod(shape)).reshape(shape) % 7 - 3).astype("float32")


def attention_reference(query, key, value, scale, mask, projection=None):
    batch, seq, heads, _ = query.shape
    results = []
    for b in range(batch):
        per_head = []
        for h in range(heads):
            k = key[b, :, h if key.shape[2] > 1 else 0]
            v = value[b, :, h if value.shape[2] > 1 else 0]
            scores = query[b, :, h] @ k.T * scale
            if mask is not None:
                scores = scores + mask[b, 0]
            weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
            weights /= weights.sum(axis=-1, keepdims=True)
            context = weights @ v
            per_head.append(
                context if projection is None else context @ projection[h]
            )
        results.append(np.stack(per_head, axis=1).reshape(seq, -1))
    return np.stack(results)


def finite_gradient(function, values, argument):
    delta = 1e-4
    result = np.zeros_like(values[argument])
    for index in np.ndindex(result.shape):
        plus = [v.copy() for v in values]
        minus = [v.copy() for v in values]
        plus[argument][index] += delta
        minus[argument][index] -= delta
        result[index] = (function(*plus) - function(*minus)) / (2 * delta)
    return result


class AllKeysIndexer(paddle.nn.Layer):
    """Fix sparse selection to all keys so the full attention has a dense oracle."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, qr, mask):
        keys = paddle.arange(mask.shape[-1], dtype="int64")
        return None, keys.reshape([1, 1, -1]).expand(
            [x.shape[0], x.shape[1], -1]
        )


class TestDSAGradients(unittest.TestCase):
    def test_sequence_auxiliary_layouts_gather_the_sequence_axis(self):
        query = paddle.zeros([1, 4, 2, 3])
        for suffix in ((3,), (1, 3)):
            for sequence_first in (False, True):
                with self.subTest(suffix=suffix, sequence_first=sequence_first):
                    values = integers((4, 1, *suffix))
                    local = tensor(values[:2])
                    other = tensor(values[2:])
                    axes = [1, 0, *range(2, local.ndim)]
                    supplied = (
                        local if sequence_first else local.transpose(axes)
                    )

                    def gather_sequence(shard):
                        assert_bitwise_equal(shard, local)
                        return paddle.concat([shard, other], axis=0)

                    with patch.object(
                        dsa,
                        "gather_from_sequence_parallel_region",
                        side_effect=gather_sequence,
                    ) as gather:
                        actual = dsa._align_sp_aux_to_query(supplied, query)
                    gather.assert_called_once()
                    assert_bitwise_equal(actual.numpy(), values.transpose(axes))
                    actual.sum().backward()
                    assert_bitwise_equal(
                        local.grad.numpy(), np.ones_like(values[:2])
                    )
                    assert_bitwise_equal(
                        other.grad.numpy(), np.ones_like(values[2:])
                    )

    def test_aligned_auxiliaries_keep_views_and_rope_slicing_uses_the_local_rank(
        self,
    ):
        query = paddle.zeros([1, 2, 2, 3])
        self.assertIsNone(dsa._align_sp_aux_to_query(None, query))
        for suffix in ((3,), (1, 3)):
            values = integers((1, 2, *suffix))
            batch_first = tensor(values)
            axes = [1, 0, *range(2, batch_first.ndim)]
            self.assertIs(
                dsa._align_sp_aux_to_query(batch_first, query), batch_first
            )
            assert_bitwise_equal(
                dsa._align_sp_aux_to_query(batch_first.transpose(axes), query),
                batch_first,
            )
        full = tensor(integers((1, 4, 1, 3)))
        with patch.object(paddle.distributed, "get_rank", return_value=3):
            local = dsa._align_sp_aux_to_query(full, query)
        assert_bitwise_equal(local, full[:, 2:4])
        local.sum().backward()
        expected_gradient = np.zeros((1, 4, 1, 3), dtype="float32")
        expected_gradient[:, 2:4] = 1
        assert_bitwise_equal(full.grad.numpy(), expected_gradient)

    def test_attention_layer_absorbs_keys_and_projects_values_without_zeroing_output(
        self,
    ):
        logging = dsa.DSAIndexerLossLoggingHelper
        self.addCleanup(setattr, logging, "tracker", logging.tracker.copy())
        self.addCleanup(setattr, logging, "num_layers", logging.num_layers)
        qn = integers((1, 2, 2, 3)).astype("float64") / 8
        compressed = integers((1, 2, 2)).astype("float64") / 8
        rope = integers((1, 2, 1)).astype("float64") / 8
        k_weight = integers((2, 2, 2)).astype("float64") / 8
        v_weight = integers((2, 2, 2)).astype("float64") / 8
        upstream = integers((1, 2, 4)).astype("float64") / 8
        mask = np.array([[[[0, -np.inf], [0, 0]]]], dtype="float64")
        for compatible, preabsorbed, rope_4d in (
            (False, False, False),
            (True, False, True),
            (False, True, True),
            (True, True, False),
        ):
            with self.subTest(
                compatible=compatible, preabsorbed=preabsorbed, rope_4d=rope_4d
            ):
                config = SimpleNamespace(
                    num_hidden_layers=1,
                    num_empty_layers_add_in_head=0,
                    use_accuracy_compatible=compatible,
                    sequence_parallel=False,
                    dsa_index_share_for_mtp_iteration=False,
                    dsa_indexer_types=["full"],
                    dsa_indexer_topk_freq=1,
                    dsa_indexer_skip_topk_offset=0,
                    tensor_model_parallel_size=1,
                    dsa_indexer_loss_coeff=None,
                    dsa_indexer_use_sparse_loss=False,
                    qk_rope_head_dim=1,
                )
                layer = dsa.DSAttention(
                    config,
                    dsa.DSAttentionSublayersSpec(indexer=AllKeysIndexer),
                    layer_number=0,
                    attn_mask_type=dsa.AttnMaskType.causal,
                    attention_type="self",
                    softmax_scale=0.5,
                    pg_collection=SimpleNamespace(tp=None),
                )
                absorbed_query = np.concatenate(
                    [
                        np.einsum("bshd,hdr->bshr", qn[..., :2], k_weight),
                        qn[..., 2:],
                    ],
                    axis=-1,
                )
                query, kv, position, projection = map(
                    tensor, (qn, compressed, rope, v_weight)
                )
                absorbed = tensor(absorbed_query) if preabsorbed else None
                key = paddle.concat([kv, position], axis=-1).unsqueeze(2)
                output = layer(
                    query,
                    key,
                    paddle.zeros([1, 2, 1, 2]),
                    None,
                    x=paddle.zeros([1, 2, 4], dtype="bfloat16"),
                    qr=paddle.zeros([1, 2, 2], dtype="bfloat16"),
                    kv_compressed=kv,
                    k_pos_emb=position.unsqueeze(2) if rope_4d else position,
                    q_absorbed=absorbed,
                    v_b_proj_weight=projection,
                    k_abs_weight=None if preabsorbed else tensor(k_weight),
                )

                def reference(kv, position, projection):
                    key = np.concatenate([kv, position], axis=-1)[:, :, None, :]
                    return attention_reference(
                        absorbed_query,
                        key,
                        kv[:, :, None, :],
                        0.5,
                        mask,
                        projection.transpose(0, 2, 1),
                    )

                expected = reference(compressed, rope, v_weight)
                self.assertGreater(np.abs(expected).max(), 0)
                np.testing.assert_allclose(
                    output.numpy(), expected, rtol=3e-5, atol=3e-6
                )
                (
                    output * paddle.to_tensor(upstream, dtype="float32")
                ).sum().backward()
                values = [compressed, rope, v_weight]

                def loss(kv, position, projection):
                    return np.sum(
                        reference(kv, position, projection) * upstream
                    )

                for index, variable in enumerate((kv, position, projection)):
                    np.testing.assert_allclose(
                        variable.grad.numpy(),
                        finite_gradient(loss, values, index),
                        rtol=3e-4,
                        atol=3e-5,
                    )

    def test_qk_preserves_asymmetric_sequences_and_per_head_key_gradients(self):
        q = integers((2, 3, 2, 4))
        g = integers((2, 3, 2, 5))
        for key_heads in (1, 3):
            with self.subTest(key_heads=key_heads):
                k = integers((2, key_heads, 4, 5))
                query, key = tensor(q), tensor(k)
                scores = dsa._AccuracyCompatibleQKMatmul.apply(query, key)
                (scores * paddle.to_tensor(g)).sum().backward()
                expected_key = np.matmul(q.swapaxes(-1, -2), g)
                if key_heads == 1:
                    expected_key = expected_key.sum(axis=1, keepdims=True)
                assert_bitwise_equal(scores.numpy(), np.matmul(q, k))
                assert_bitwise_equal(
                    query.grad.numpy(), np.matmul(g, k.swapaxes(-1, -2))
                )
                assert_bitwise_equal(key.grad.numpy(), expected_key)
                key.set_value(key.detach() - key.grad * 0.25)
                assert_bitwise_equal(key.numpy(), k - expected_key * 0.25)

    def test_ste_only_adds_the_broadcast_key_gradient_and_restores_its_dtype(
        self,
    ):
        q = integers((2, 3, 2, 4))
        g = integers((2, 3, 2, 5))
        for key_heads, dtype in ((1, "float32"), (3, "float64")):
            with self.subTest(key_heads=key_heads, dtype=dtype):
                k = integers((2, key_heads, 5, 4))
                query, key = tensor(q, dtype), tensor(k, dtype)
                scale = paddle.full([], 0.5, dtype="float32")
                scores = dsa._SteQKMatmul.apply(query, key, scale)
                (scores * paddle.to_tensor(g)).sum().backward()
                expected_key = np.matmul(g.swapaxes(-1, -2), q) * 0.5
                if key_heads == 1:
                    expected_key = expected_key.sum(axis=1, keepdims=True)
                self.assertIsNone(query.grad)
                assert_bitwise_equal(
                    scores.numpy(), np.matmul(q, k.swapaxes(-1, -2)) * 0.5
                )
                assert_bitwise_equal(
                    key.grad.numpy(), expected_key.astype(dtype)
                )

    def test_masked_softmax_has_finite_gradients_and_positive_masked_zeros(
        self,
    ):
        values = np.array(
            [[0, -np.inf, 0, -np.inf], [-np.inf] * 4, [0] * 4],
            dtype="float32",
        )
        logits = tensor(values)
        valid = paddle.to_tensor(np.isfinite(values))
        result = dsa._AccuracyCompatibleSoftmax.apply(logits, valid)
        upstream = np.array(
            [[1, 9, 3, -9], [7] * 4, [0, 1, 2, 3]], dtype="float32"
        )
        (result * paddle.to_tensor(upstream)).sum().backward()
        probabilities = np.array(
            [[0.5, 0, 0.5, 0], [0] * 4, [0.25] * 4], dtype="float32"
        )
        gradients = np.array(
            [[-0.5, 0, 0.5, 0], [0] * 4, [-0.375, -0.125, 0.125, 0.375]],
            dtype="float32",
        )
        assert_bitwise_equal(result.numpy(), probabilities)
        assert_bitwise_equal(logits.grad.numpy(), gradients)

    def test_linear_bias_is_added_exactly_once_with_gradients(self):
        x, w, bias = integers((2, 4)), integers((4, 3)), integers((3,))
        for skip_bias_add in (False, True):
            with self.subTest(skip_bias_add=skip_bias_add):
                projection = paddle.nn.Linear(4, 3)
                projection.weight.set_value(w)
                projection.bias.set_value(bias)
                projection.skip_bias_add = skip_bias_add
                inputs = tensor(x)
                output, output_bias = dsa._accuracy_compat_linear(
                    projection, inputs
                )
                if skip_bias_add:
                    self.assertIs(output_bias, projection.bias)
                    assert_bitwise_equal(output.numpy(), x @ w)
                    output = output + output_bias
                else:
                    self.assertIsNone(output_bias)
                output.sum().backward()
                assert_bitwise_equal(output.numpy(), x @ w + bias)
                assert_bitwise_equal(
                    inputs.grad.numpy(), np.ones((2, 3), dtype="float32") @ w.T
                )
                assert_bitwise_equal(
                    projection.bias.grad.numpy(), np.full(3, 2, dtype="float32")
                )

    def test_k_absorption_retains_values_and_both_gradients(self):
        q, k = integers((3, 2, 4)), integers((3, 4, 5))
        for compatible in (False, True):
            with self.subTest(compatible=compatible):
                query, weight = tensor(q), tensor(k)
                output = dsa._absorb_q_nope_k_up(
                    query, weight, use_accuracy_compatible=compatible
                )
                output.sum().backward()
                ones = np.ones((3, 2, 5), dtype="float32")
                assert_bitwise_equal(output.numpy(), np.matmul(q, k))
                assert_bitwise_equal(
                    query.grad.numpy(), np.matmul(ones, k.swapaxes(-1, -2))
                )
                assert_bitwise_equal(
                    weight.grad.numpy(), np.matmul(q.swapaxes(-1, -2), ones)
                )

    def test_absorbed_attention_values_and_all_gradients_match_finite_differences(
        self,
    ):
        mask = np.array([[[[0, -np.inf, 0], [0, 0, -np.inf]]]], dtype="float64")
        for compatible, key_heads, selected_mask in (
            (False, 1, None),
            (True, 1, mask),
            (True, 2, None),
        ):
            with self.subTest(compatible=compatible, key_heads=key_heads):
                values = [
                    integers(shape).astype("float64") / 8
                    for shape in (
                        (1, 2, 2, 2),
                        (1, 3, key_heads, 2),
                        (1, 3, key_heads, 2),
                        (2, 2, 2),
                    )
                ]
                inputs = [tensor(v) for v in values]
                q, k, v, projection = inputs
                actual = dsa._unfused_absorbed_dsa_attention(
                    q,
                    k,
                    v,
                    projection,
                    None
                    if selected_mask is None
                    else paddle.to_tensor(selected_mask, dtype="float32"),
                    0.5,
                    use_accuracy_compatible=compatible,
                )
                upstream = integers((1, 2, 4)).astype("float64") / 8
                (
                    actual * paddle.to_tensor(upstream, dtype="float32")
                ).sum().backward()
                expected = attention_reference(
                    *values[:3], 0.5, selected_mask, values[3]
                )
                np.testing.assert_allclose(
                    actual.numpy(), expected, rtol=3e-5, atol=3e-6
                )

                def loss(q, k, v, w):
                    return np.sum(
                        attention_reference(q, k, v, 0.5, selected_mask, w)
                        * upstream
                    )

                for index, variable in enumerate(inputs):
                    np.testing.assert_allclose(
                        variable.grad.numpy(),
                        finite_gradient(loss, values, index),
                        rtol=3e-4,
                        atol=3e-5,
                    )

    def test_unfused_attention_keeps_the_absorbed_key_value_gradient(self):
        mask = np.array([[[[0, -np.inf, 0], [0, 0, -np.inf]]]], dtype="float64")
        for compatible, key_heads, tp in (
            (False, 1, 1),
            (True, 1, 1),
            (True, 1, 2),
            (True, 2, 1),
        ):
            with self.subTest(
                compatible=compatible, key_heads=key_heads, tp=tp
            ):
                values = [
                    integers(shape).astype("float64") / 8
                    for shape in (
                        (1, 2, 2, 3),
                        (1, 3, key_heads, 3),
                        (1, 3, key_heads, 2),
                    )
                ]
                q, k, v = [tensor(value) for value in values]
                actual = dsa._unfused_dsa_attention(
                    q,
                    k,
                    v,
                    paddle.to_tensor(mask, dtype="float32"),
                    0.5,
                    use_accuracy_compatible=compatible,
                    tensor_parallel_size=tp,
                )
                upstream = integers((1, 2, 4)).astype("float64") / 8
                (
                    actual * paddle.to_tensor(upstream, dtype="float32")
                ).sum().backward()
                absorbed = compatible and key_heads == 1
                reference_value = values[1][..., :2] if absorbed else values[2]
                expected = attention_reference(
                    values[0], values[1], reference_value, 0.5, mask
                )
                np.testing.assert_allclose(
                    actual.numpy(), expected, rtol=3e-5, atol=3e-6
                )

                def loss(q, k, v):
                    return np.sum(
                        attention_reference(
                            q, k, k[..., :2] if absorbed else v, 0.5, mask
                        )
                        * upstream
                    )

                for index, variable in enumerate((q, k, v)):
                    if absorbed and index == 2:
                        self.assertIsNone(variable.grad)
                    else:
                        np.testing.assert_allclose(
                            variable.grad.numpy(),
                            finite_gradient(loss, values, index),
                            rtol=3e-4,
                            atol=3e-5,
                        )


if __name__ == "__main__":
    paddle.set_device("gpu:0")
    unittest.main()
