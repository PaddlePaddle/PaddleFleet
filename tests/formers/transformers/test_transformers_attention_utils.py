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

"""Behavior tests for paddlefleet.transformers.attention_utils.

These tests exercise the real attention utilities and compare their output
against independent, hand-derived references (plain numpy re-implementations
of the documented math, never the functions under test). The attention core
is probed with distinguishable Q/K/V, an explicit key visibility mask and an
additive attn_mask so that visible range, output layout and mask-driven
gradients are all observable. Everything here runs on CPU.
"""

import unittest

import numpy as np
import paddle
from paddle import ParamAttr

from paddlefleet.transformers.attention_utils import (
    Attention,
    AttentionRegistry,
    DefaultAttention,
    Linear3D,
    MultiHeadAttention,
    Registry,
    _convert_param_attr_to_list,
)


# --------------------------------------------------------------------------
# Independent numpy references (NOT the code under test).
# --------------------------------------------------------------------------
def linear3d_ref(x, weight, bias, num_heads):
    """Independent replica of Linear3D.forward documented math.

    x: [B, T, D]; weight: [D, D]; bias: [D]. Returns [B, H, T, D/H].
    """
    x = np.asarray(x, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    b, t, d = x.shape
    out = x @ weight + bias.reshape(1, 1, d)
    out = out.reshape(b, t, num_heads, d // num_heads)
    out = np.transpose(out, (0, 2, 1, 3))
    return out


def default_attention_ref(q, k, v, d_head, qmask, kmask, attn_mask=None):
    """Independent replica of DefaultAttention.forward documented math.

    q/k/v: [B, H, T, D]; qmask: [B, 1, T, 1]; kmask: [B, 1, 1, T].
    """
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    qmask = np.asarray(qmask, dtype=np.float64)
    kmask = np.asarray(kmask, dtype=np.float64)

    product = np.matmul(q, np.swapaxes(k, -1, -2)) * (d_head**-0.5)
    product = product + (1.0 - np.matmul(qmask, kmask)) * -1e6
    if attn_mask is not None:
        product = product + np.asarray(attn_mask, dtype=np.float64)
    m = product.max(axis=-1, keepdims=True)
    e = np.exp(product - m)
    weights = e / e.sum(axis=-1, keepdims=True)
    return np.matmul(weights, v)


class TestRegistry(unittest.TestCase):
    def test_register_stores_and_returns_same_class(self):
        registry = Registry()

        @registry.register("probe")
        class Probe:
            pass

        # The decorator must return the class unchanged and store it by name.
        self.assertIs(Probe, registry.cls_dict["probe"])
        self.assertIsInstance(Probe(), Probe)

    def test_register_keeps_distinct_entries(self):
        registry = Registry()

        @registry.register("a")
        class A:
            pass

        @registry.register("b")
        class B:
            pass

        self.assertIs(registry.cls_dict["a"], A)
        self.assertIs(registry.cls_dict["b"], B)
        self.assertIsNot(registry.cls_dict["a"], registry.cls_dict["b"])

    def test_default_attention_registered_to_correct_class(self):
        self.assertIs(
            AttentionRegistry.cls_dict["default_attention"], DefaultAttention
        )


class TestConvertParamAttrToList(unittest.TestCase):
    def test_single_param_attr_gets_indexed_names(self):
        result = _convert_param_attr_to_list(ParamAttr(name="w"), 3)
        self.assertEqual(len(result), 3)
        self.assertEqual([a.name for a in result], ["w_0", "w_1", "w_2"])

    def test_bool_true_expands_to_real_attrs(self):
        result = _convert_param_attr_to_list(True, 2)
        self.assertEqual(len(result), 2)
        # True must become concrete (non-False) ParamAttr entries.
        for attr in result:
            self.assertIsNot(attr, False)
            self.assertNotEqual(attr, False)

    def test_bool_false_expands_to_false_list(self):
        result = _convert_param_attr_to_list(False, 4)
        self.assertEqual(result, [False, False, False, False])

    def test_list_preserves_per_entry_bool_semantics(self):
        result = _convert_param_attr_to_list([True, False, ParamAttr()], 3)
        self.assertEqual(len(result), 3)
        # True -> real attr, False -> False passthrough, attr -> real attr.
        self.assertIsNot(result[0], False)
        self.assertIs(result[1], False)
        self.assertIsNot(result[2], False)

    def test_list_wrong_length_raises(self):
        with self.assertRaises(AssertionError):
            _convert_param_attr_to_list([True, False], 3)


class TestLinear3D(unittest.TestCase):
    def _build_layer(self, weight_np, bias_np, num_heads):
        embed = weight_np.shape[0]
        layer = Linear3D(
            hidden_size=embed,
            num_attention_heads=num_heads,
            size_per_head=embed // num_heads,
        )
        layer.weight.set_value(paddle.to_tensor(weight_np, dtype="float32"))
        layer.bias.set_value(paddle.to_tensor(bias_np, dtype="float32"))
        return layer

    def test_forward_matches_independent_reference(self):
        # Distinguishable input (arange) + non-symmetric weight so a wrong
        # matmul, missing bias or transposed head layout would be caught.
        embed, num_heads = 4, 2
        weight_np = (
            np.arange(embed * embed, dtype=np.float32).reshape(embed, embed)
            / 10.0
        )
        bias_np = np.array([0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        layer = self._build_layer(weight_np, bias_np, num_heads)

        x_np = np.arange(1 * 3 * embed, dtype=np.float32).reshape(1, 3, embed)
        out = layer(paddle.to_tensor(x_np))

        # Output layout must be [B, H, T, D/H].
        self.assertEqual(out.shape, [1, num_heads, 3, embed // num_heads])
        ref = linear3d_ref(x_np, weight_np, bias_np, num_heads)
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5, atol=1e-5)

    def test_head_split_keeps_row_order(self):
        # Identity weight, zero bias: output must be the pure reshape/transpose
        # of the input, exposing any head-axis scrambling.
        embed, num_heads = 4, 2
        weight_np = np.eye(embed, dtype=np.float32)
        bias_np = np.zeros(embed, dtype=np.float32)
        layer = self._build_layer(weight_np, bias_np, num_heads)

        x_np = np.arange(1 * 2 * embed, dtype=np.float32).reshape(1, 2, embed)
        out = layer(paddle.to_tensor(x_np))
        expected = np.transpose(
            x_np.reshape(1, 2, num_heads, embed // num_heads), (0, 2, 1, 3)
        )
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)


class TestDefaultAttention(unittest.TestCase):
    def setUp(self):
        # Distinguishable Q/K/V so Q<->K swaps and V permutations are visible.
        self.d_head = 2
        self.q = np.array(
            [[[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]]], dtype=np.float32
        )
        self.k = np.array(
            [[[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]]], dtype=np.float32
        )
        self.v = np.array(
            [[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]], dtype=np.float32
        )
        # Key position 2 is masked out for every query row.
        self.qmask = np.ones((1, 1, 3, 1), dtype=np.float32)
        self.kmask = np.array([[[[1.0, 1.0, 0.0]]]], dtype=np.float32)

    def _call(self, q=None, k=None, v=None, attn_mask=None):
        attn = DefaultAttention()
        return attn(
            paddle.to_tensor(self.q if q is None else q),
            paddle.to_tensor(self.k if k is None else k),
            paddle.to_tensor(self.v if v is None else v),
            d_head=self.d_head,
            attn_mask=None
            if attn_mask is None
            else paddle.to_tensor(attn_mask),
            query_mask=paddle.to_tensor(self.qmask),
            key_mask=paddle.to_tensor(self.kmask),
        )

    def test_forward_matches_independent_reference(self):
        out = self._call()
        self.assertEqual(out.shape, [1, 1, 3, self.d_head])
        ref = default_attention_ref(
            self.q, self.k, self.v, self.d_head, self.qmask, self.kmask
        )
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5, atol=1e-5)

    def test_masked_key_is_invisible(self):
        base = self._call().numpy()
        # Changing the value at the masked key (pos 2) must not move the output.
        v_masked = self.v.copy()
        v_masked[0, 0, 2, :] = [999.0, -999.0]
        out_masked = self._call(v=v_masked).numpy()
        np.testing.assert_allclose(out_masked, base, rtol=1e-4, atol=1e-4)

        # Changing a visible key value (pos 1) must move the output.
        v_visible = self.v.copy()
        v_visible[0, 0, 1, :] = [999.0, -999.0]
        out_visible = self._call(v=v_visible).numpy()
        self.assertGreater(np.abs(out_visible - base).max(), 1.0)

    def test_attn_mask_is_additive(self):
        # Strongly bias attention toward visible key 0; output should collapse
        # onto V[0] = [1, 2] for every query row.
        attn_mask = np.zeros((1, 1, 3, 3), dtype=np.float32)
        attn_mask[0, 0, :, 0] = 1e4
        out = self._call(attn_mask=attn_mask).numpy()
        expected = np.tile(self.v[0, 0, 0, :], (3, 1)).reshape(1, 1, 3, 2)
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-3)

    def test_query_key_roles_are_not_symmetric(self):
        # Q@K^T is not symmetric; swapping Q and K must change the output.
        out_qk = self._call().numpy()
        out_kq = self._call(q=self.k, k=self.q).numpy()
        self.assertGreater(np.abs(out_qk - out_kq).max(), 1e-4)

    def test_gradient_respects_visible_range(self):
        attn = DefaultAttention()
        q = paddle.to_tensor(self.q, stop_gradient=False)
        k = paddle.to_tensor(self.k, stop_gradient=False)
        v = paddle.to_tensor(self.v, stop_gradient=False)
        out = attn(
            q,
            k,
            v,
            d_head=self.d_head,
            query_mask=paddle.to_tensor(self.qmask),
            key_mask=paddle.to_tensor(self.kmask),
        )
        out.backward(paddle.ones_like(out))

        self.assertIsNotNone(v.grad)
        vgrad = v.grad.numpy()
        # Masked key (pos 2) receives ~no gradient; visible keys do.
        self.assertLess(np.abs(vgrad[0, 0, 2, :]).max(), 1e-3)
        self.assertGreater(np.abs(vgrad[0, 0, 0, :]).max(), 1e-3)
        self.assertGreater(np.abs(vgrad[0, 0, 1, :]).max(), 1e-3)


class TestAttentionBase(unittest.TestCase):
    def test_base_forward_raises_not_implemented(self):
        attn = Attention()
        with self.assertRaises(NotImplementedError):
            attn(
                paddle.zeros([1, 1, 2, 2]),
                paddle.zeros([1, 1, 2, 2]),
                paddle.zeros([1, 1, 2, 2]),
                d_head=2,
            )


class TestMultiHeadAttention(unittest.TestCase):
    def _build_mha(self):
        return MultiHeadAttention(
            embed_dim=4,
            num_heads=2,
            attention_type="default_attention",
        )

    def test_init_derives_head_dim(self):
        mha = self._build_mha()
        self.assertEqual(mha.embed_dim, 4)
        self.assertEqual(mha.num_heads, 2)
        self.assertEqual(mha.head_dim, 2)

    def test_forward_matches_full_independent_pipeline(self):
        embed, num_heads, head_dim = 4, 2, 2
        mha = self._build_mha()

        # q_proj scales by 2, the rest are identity, so a mis-wired projection
        # (e.g. reusing k_proj for the query) would shift the output.
        wq = 2.0 * np.eye(embed, dtype=np.float32)
        eye = np.eye(embed, dtype=np.float32)
        zeros = np.zeros(embed, dtype=np.float32)
        for proj, w in ((mha.q_proj, wq), (mha.k_proj, eye), (mha.v_proj, eye)):
            proj.weight.set_value(paddle.to_tensor(w))
            proj.bias.set_value(paddle.to_tensor(zeros))
        # out_proj is nn.Linear (y = x @ W + b); identity keeps it traceable.
        mha.out_proj.weight.set_value(paddle.to_tensor(eye))
        mha.out_proj.bias.set_value(paddle.to_tensor(zeros))

        query = np.arange(1 * 3 * embed, dtype=np.float32).reshape(1, 3, embed)
        key = (query[:, ::-1, :] + 0.5).astype(np.float32)
        value = (query * 0.25 - 1.0).astype(np.float32)
        qmask = np.ones((1, 1, 3, 1), dtype=np.float32)
        kmask = np.array([[[[1.0, 1.0, 0.0]]]], dtype=np.float32)

        out = mha(
            paddle.to_tensor(query),
            paddle.to_tensor(key),
            paddle.to_tensor(value),
            query_mask=paddle.to_tensor(qmask),
            key_mask=paddle.to_tensor(kmask),
        )
        # Output must be projected back to [B, T, embed_dim].
        self.assertEqual(out.shape, [1, 3, embed])

        q_ref = linear3d_ref(query, wq, zeros, num_heads)
        k_ref = linear3d_ref(key, eye, zeros, num_heads)
        v_ref = linear3d_ref(value, eye, zeros, num_heads)
        attn = default_attention_ref(
            q_ref, k_ref, v_ref, head_dim, qmask, kmask
        )
        combined = np.transpose(attn, (0, 2, 1, 3)).reshape(1, 3, embed)
        np.testing.assert_allclose(out.numpy(), combined, rtol=1e-4, atol=1e-4)

    def test_key_value_default_to_query(self):
        mha = self._build_mha()
        query = paddle.to_tensor(
            np.arange(1 * 3 * 4, dtype=np.float32).reshape(1, 3, 4)
        )
        qmask = paddle.ones([1, 1, 3, 1])
        kmask = paddle.ones([1, 1, 1, 3])
        out_default = mha(query, None, None, query_mask=qmask, key_mask=kmask)
        out_explicit = mha(
            query, query, query, query_mask=qmask, key_mask=kmask
        )
        np.testing.assert_allclose(
            out_default.numpy(), out_explicit.numpy(), rtol=1e-6, atol=1e-6
        )

    def test_compute_kv_routes_key_and_value_without_swap(self):
        mha = self._build_mha()
        # Distinct key/value so a k/v swap in compute_kv would be caught.
        key = paddle.to_tensor(
            np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        )
        value = paddle.to_tensor(
            np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) + 100.0
        )
        k, v = mha.compute_kv(key, value)
        # Reference: the real (independent) projection layers.
        np.testing.assert_allclose(
            k.numpy(), mha.k_proj(key).numpy(), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            v.numpy(), mha.v_proj(value).numpy(), rtol=1e-6, atol=1e-6
        )
        self.assertEqual(k.shape, [2, 2, 3, 2])

    def test_gen_cache_static_holds_projected_kv(self):
        mha = self._build_mha()
        key = paddle.to_tensor(
            np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        )
        value = paddle.to_tensor(
            np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) + 100.0
        )
        cache = mha.gen_cache(key, value, type=MultiHeadAttention.StaticCache)
        self.assertIsInstance(cache, MultiHeadAttention.StaticCache)
        k_ref, v_ref = mha.compute_kv(key, value)
        np.testing.assert_allclose(cache.k.numpy(), k_ref.numpy())
        np.testing.assert_allclose(cache.v.numpy(), v_ref.numpy())

    def test_gen_cache_with_value_passes_through_unprojected(self):
        mha = self._build_mha()
        # type defaults to Cache; with a value supplied the raw tensors are
        # stored verbatim (no projection), so content must be identical.
        key = paddle.to_tensor([[[1.0, 2.0]]])
        value = paddle.to_tensor([[[3.0, 4.0]]])
        cache = mha.gen_cache(key, value)
        self.assertIsInstance(cache, MultiHeadAttention.Cache)
        np.testing.assert_array_equal(cache.k.numpy(), key.numpy())
        np.testing.assert_array_equal(cache.v.numpy(), value.numpy())

    def test_forward_returns_tuple_with_cache(self):
        mha = self._build_mha()
        query = paddle.to_tensor(
            np.arange(1 * 3 * 4, dtype=np.float32).reshape(1, 3, 4)
        )
        key = paddle.to_tensor(
            np.arange(1 * 3 * 4, dtype=np.float32).reshape(1, 3, 4) + 1.0
        )
        value = paddle.to_tensor(
            np.arange(1 * 3 * 4, dtype=np.float32).reshape(1, 3, 4) + 2.0
        )
        qmask = paddle.ones([1, 1, 3, 1])
        kmask = paddle.ones([1, 1, 1, 3])

        plain = mha(query, key, value, query_mask=qmask, key_mask=kmask)
        self.assertIsInstance(plain, paddle.Tensor)

        cache = mha.gen_cache(key, value, type=MultiHeadAttention.StaticCache)
        result = mha(
            query,
            key,
            value,
            query_mask=qmask,
            key_mask=kmask,
            cache=cache,
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        output, returned_cache = result
        self.assertEqual(output.shape, [1, 3, 4])
        self.assertIsInstance(returned_cache, MultiHeadAttention.StaticCache)


if __name__ == "__main__":
    unittest.main()
