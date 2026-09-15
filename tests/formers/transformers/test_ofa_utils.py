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

"""Behavior tests for paddlefleet.transformers.ofa_utils.

The production module supplies the OFA (Once-For-All) supernet slimming helpers
for ``paddle.nn.MultiHeadAttention`` / ``TransformerEncoder`` based models:

  * ``reorder_neuron`` / ``reorder_head`` / ``reorder_neuron_head`` permute the
    feed-forward neurons and attention heads of a layer in place, following an
    importance ordering (weight-layout surgery).
  * ``prepare_qkv_ofa`` / ``mha_ofa_forward`` / ``encoder_ofa_forward`` /
    ``encoder_layer_ofa_forward`` are monkey-patch ``forward`` implementations
    that thread a per-head ``head_mask`` through attention.
  * ``compute_neuron_head_importance`` accumulates head/neuron importance from
    real gradients.

Per the repo unit-test rules this is a *model-layer* module, so the suite uses
small real ``paddle`` layers with distinguishable (``arange``) weights/inputs
and independent, hand-derived expected values built from basic ops or numpy
indexing -- never by calling the function under test to build the expectation,
never shape-only, never ``assert_called``-only.

Execution note: 无卡 CPU. All layers run on CPU; no accelerator numerics are
claimed. ``compute_neuron_head_importance`` drives a real ``loss.backward()``.
Requires an installed CPU ``paddle`` to run (cannot be executed in this
sandbox); the module is validated with ``python3 -m py_compile``.
"""

import unittest

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddlefleet.transformers.ofa_utils import (
    compute_neuron_head_importance,
    encoder_layer_ofa_forward,
    encoder_ofa_forward,
    mha_ofa_forward,
    prepare_qkv_ofa,
    reorder_head,
    reorder_neuron,
    reorder_neuron_head,
)


def _fill_linear(linear, base=0.0):
    """Overwrite a Linear's weight/bias with distinguishable arange content.

    Distinct ``base`` per layer keeps different layers apart, so a wrong
    dispatch (e.g. reordering the wrong projection) is visible.
    """
    wshape = list(linear.weight.shape)
    n = int(np.prod(wshape))
    linear.weight.set_value(
        paddle.to_tensor(np.arange(n, dtype="float32").reshape(wshape) + base)
    )
    if linear.bias is not None:
        m = int(np.prod(list(linear.bias.shape)))
        linear.bias.set_value(
            paddle.to_tensor(np.arange(m, dtype="float32") + 1000.0 + base)
        )


class _FnWrapper:
    """Minimal stand-in for an OFA supernet layer that exposes ``.fn``.

    ``reorder_neuron`` unwraps ``layer.fn`` when present; this exercises that
    dispatch without depending on the real supernet implementation.
    """

    def __init__(self, fn):
        self.fn = fn


class TestReorderNeuron(unittest.TestCase):
    """reorder_neuron permutes Linear weights (and bias only for dim=1)."""

    @classmethod
    def setUpClass(cls):
        paddle.set_device("cpu")

    def test_dim0_permutes_input_rows_and_keeps_bias(self):
        # paddle Linear(in=4, out=8): weight [4, 8], bias [8]. dim=0 selects
        # along the in_features axis; bias must be left untouched.
        layer = nn.Linear(4, 8)
        _fill_linear(layer)
        w0 = layer.weight.numpy().copy()
        b0 = layer.bias.numpy().copy()

        index = paddle.to_tensor([2, 0, 3, 1], dtype="int64")
        reorder_neuron(layer, index, dim=0)

        np.testing.assert_array_equal(layer.weight.numpy(), w0[[2, 0, 3, 1], :])
        # dim=0 branch assigns the bias as-is (no reorder).
        np.testing.assert_array_equal(layer.bias.numpy(), b0)

    def test_dim1_permutes_output_cols_and_bias(self):
        # dim=1 selects along out_features; bias (len == out_features) is
        # reordered by the same index.
        layer = nn.Linear(4, 8)
        _fill_linear(layer)
        w0 = layer.weight.numpy().copy()
        b0 = layer.bias.numpy().copy()

        perm = [7, 6, 5, 4, 3, 2, 1, 0]
        index = paddle.to_tensor(perm, dtype="int64")
        reorder_neuron(layer, index, dim=1)

        np.testing.assert_array_equal(layer.weight.numpy(), w0[:, perm])
        np.testing.assert_array_equal(layer.bias.numpy(), b0[perm])

    def test_partial_permutation_is_a_gather_not_inplace_swap(self):
        # A repeated index proves the op is an index_select gather (columns can
        # be duplicated), which a naive in-place swap would get wrong.
        layer = nn.Linear(3, 4)
        _fill_linear(layer)
        w0 = layer.weight.numpy().copy()
        b0 = layer.bias.numpy().copy()

        perm = [0, 0, 2, 3]
        reorder_neuron(layer, paddle.to_tensor(perm, dtype="int64"), dim=1)

        np.testing.assert_array_equal(layer.weight.numpy(), w0[:, perm])
        np.testing.assert_array_equal(layer.bias.numpy(), b0[perm])

    def test_unwraps_fn_attribute(self):
        # When the passed object has a ``.fn`` (OFA supernet wrapper), the inner
        # Linear is the one that gets reordered.
        inner = nn.Linear(4, 6)
        _fill_linear(inner)
        w0 = inner.weight.numpy().copy()
        perm = [5, 4, 3, 2, 1, 0]

        reorder_neuron(
            _FnWrapper(inner), paddle.to_tensor(perm, "int64"), dim=1
        )

        np.testing.assert_array_equal(inner.weight.numpy(), w0[:, perm])


def _independent_head_column_index(num_heads, head_dim, head_perm):
    """Column index that reorder_head must apply to q/k/v projections.

    Independent derivation: view the ``num_heads * head_dim`` output columns as
    ``num_heads`` contiguous blocks of ``head_dim`` columns, then reorder the
    blocks by ``head_perm``. This mirrors the docstring contract, not the
    implementation's tensor ops.
    """
    idx = []
    for h in head_perm:
        idx.extend(range(h * head_dim, h * head_dim + head_dim))
    return idx


class TestReorderHead(unittest.TestCase):
    """reorder_head permutes attention heads across q/k/v/out projections."""

    @classmethod
    def setUpClass(cls):
        paddle.set_device("cpu")

    def _build_mha(self):
        mha = nn.MultiHeadAttention(embed_dim=4, num_heads=2)
        # Distinct base per projection so a q/k/v/out mix-up is visible.
        _fill_linear(mha.q_proj, base=0.0)
        _fill_linear(mha.k_proj, base=100.0)
        _fill_linear(mha.v_proj, base=200.0)
        _fill_linear(mha.out_proj, base=300.0)
        return mha

    def test_rejects_non_multihead_layer(self):
        with self.assertRaises(AssertionError):
            reorder_head(nn.Linear(4, 4), paddle.to_tensor([0], "int64"))

    def test_head_permutation_reorders_all_projections(self):
        mha = self._build_mha()
        q0 = mha.q_proj.weight.numpy().copy()
        qb0 = mha.q_proj.bias.numpy().copy()
        k0 = mha.k_proj.weight.numpy().copy()
        v0 = mha.v_proj.weight.numpy().copy()
        o0 = mha.out_proj.weight.numpy().copy()
        ob0 = mha.out_proj.bias.numpy().copy()

        head_perm = [1, 0]  # swap the two heads
        reorder_head(mha, paddle.to_tensor(head_perm, dtype="int64"))

        col_idx = _independent_head_column_index(
            num_heads=2, head_dim=2, head_perm=head_perm
        )
        self.assertEqual(col_idx, [2, 3, 0, 1])

        # q/k/v: output columns reordered by the per-head column index; their
        # bias (dim=1 path) is reordered by the same index.
        np.testing.assert_array_equal(mha.q_proj.weight.numpy(), q0[:, col_idx])
        np.testing.assert_array_equal(mha.q_proj.bias.numpy(), qb0[col_idx])
        np.testing.assert_array_equal(mha.k_proj.weight.numpy(), k0[:, col_idx])
        np.testing.assert_array_equal(mha.v_proj.weight.numpy(), v0[:, col_idx])

        # out_proj: input rows reordered (dim=0) and its bias left unchanged.
        np.testing.assert_array_equal(
            mha.out_proj.weight.numpy(), o0[col_idx, :]
        )
        np.testing.assert_array_equal(mha.out_proj.bias.numpy(), ob0)

    def test_identity_permutation_is_a_noop(self):
        mha = self._build_mha()
        q0 = mha.q_proj.weight.numpy().copy()
        o0 = mha.out_proj.weight.numpy().copy()

        reorder_head(mha, paddle.to_tensor([0, 1], dtype="int64"))

        np.testing.assert_array_equal(mha.q_proj.weight.numpy(), q0)
        np.testing.assert_array_equal(mha.out_proj.weight.numpy(), o0)


class TestReorderNeuronHead(unittest.TestCase):
    """reorder_neuron_head orchestrates head + neuron reordering per layer,
    driven by *descending* importance argsort."""

    @classmethod
    def setUpClass(cls):
        paddle.set_device("cpu")

    def test_descending_importance_drives_reorder(self):
        from types import SimpleNamespace

        mha = nn.MultiHeadAttention(embed_dim=4, num_heads=2)
        _fill_linear(mha.q_proj, base=0.0)
        _fill_linear(mha.k_proj, base=100.0)
        _fill_linear(mha.v_proj, base=200.0)
        _fill_linear(mha.out_proj, base=300.0)

        linear1 = nn.Linear(4, 3)  # intermediate: 3 ffn neurons
        linear2 = nn.Linear(3, 4)  # output projection
        _fill_linear(linear1, base=400.0)
        _fill_linear(linear2, base=500.0)

        layer = SimpleNamespace(
            self_attn=mha,
            linear1=_FnWrapper(linear1),
            linear2=_FnWrapper(linear2),
        )
        model = SimpleNamespace(
            base_model=SimpleNamespace(encoder=SimpleNamespace(layers=[layer]))
        )

        q0 = mha.q_proj.weight.numpy().copy()
        o0 = mha.out_proj.weight.numpy().copy()
        l1w0 = linear1.weight.numpy().copy()
        l1b0 = linear1.bias.numpy().copy()
        l2w0 = linear2.weight.numpy().copy()
        l2b0 = linear2.bias.numpy().copy()

        # head_importance -> desc argsort [1, 0]; neuron_importance -> [2, 0, 1].
        head_importance = [paddle.to_tensor([0.1, 0.9], dtype="float32")]
        neuron_importance = [np.array([0.5, 0.1, 0.9], dtype="float32")]

        reorder_neuron_head(model, head_importance, neuron_importance)

        head_cols = _independent_head_column_index(2, 2, [1, 0])  # [2,3,0,1]
        np.testing.assert_array_equal(
            mha.q_proj.weight.numpy(), q0[:, head_cols]
        )
        np.testing.assert_array_equal(
            mha.out_proj.weight.numpy(), o0[head_cols, :]
        )

        neuron_perm = [2, 0, 1]
        # linear1 reordered along out cols (dim=1), bias follows.
        np.testing.assert_array_equal(
            linear1.weight.numpy(), l1w0[:, neuron_perm]
        )
        np.testing.assert_array_equal(linear1.bias.numpy(), l1b0[neuron_perm])
        # linear2 reordered along in rows (dim=0), bias unchanged.
        np.testing.assert_array_equal(
            linear2.weight.numpy(), l2w0[neuron_perm, :]
        )
        np.testing.assert_array_equal(linear2.bias.numpy(), l2b0)


def _ref_attention(mha, query, key, value, attn_mask=None, head_mask=None):
    """Independent scaled-dot-product-attention reference from basic ops.

    Uses the layer's standard ``_prepare_qkv`` / ``out_proj`` collaborators
    (not the function under test) plus plain matmul/softmax so a wrong scale,
    a dropped additive mask, or a mis-applied head_mask would be caught.
    """
    q, k, v = mha._prepare_qkv(query, key, value)
    product = paddle.matmul(q * (mha.head_dim**-0.5), k, transpose_y=True)
    if attn_mask is not None:
        product = product + attn_mask
    weights = F.softmax(product)
    if head_mask is not None:
        weights = weights * head_mask
    out = paddle.matmul(weights, v)
    out = paddle.transpose(out, perm=[0, 2, 1, 3])
    out = paddle.reshape(out, shape=[0, 0, out.shape[2] * out.shape[3]])
    return mha.out_proj(out)


class TestMhaOfaForward(unittest.TestCase):
    """mha_ofa_forward: scaled dot-product attention with an optional per-head
    multiplicative head_mask (attn_mask[1])."""

    @classmethod
    def setUpClass(cls):
        paddle.set_device("cpu")

    def _setup(self):
        mha = nn.MultiHeadAttention(embed_dim=4, num_heads=2)
        _fill_linear(mha.q_proj, base=0.0)
        _fill_linear(mha.k_proj, base=10.0)
        _fill_linear(mha.v_proj, base=20.0)
        _fill_linear(mha.out_proj, base=30.0)
        mha.eval()  # disable dropout for deterministic output
        x = paddle.to_tensor(
            np.arange(2 * 3 * 4, dtype="float32").reshape([2, 3, 4]) / 10.0
        )
        return mha, x

    def test_no_mask_matches_independent_reference(self):
        mha, x = self._setup()
        out = mha_ofa_forward(mha, x, x, x, attn_mask=[None, None])
        ref = _ref_attention(mha, x, x, x)
        self.assertEqual(out.shape, [2, 3, 4])
        np.testing.assert_allclose(
            out.numpy(), ref.numpy(), rtol=1e-5, atol=1e-6
        )

    def test_head_mask_of_ones_is_identity(self):
        # Multiplying attention weights by an all-ones head_mask must not change
        # the output relative to no head_mask.
        mha, x = self._setup()
        out_none = mha_ofa_forward(mha, x, x, x, attn_mask=[None, None])
        head_mask = paddle.ones([1, 2, 1, 1], dtype="float32")
        out_ones = mha_ofa_forward(mha, x, x, x, attn_mask=[None, head_mask])
        np.testing.assert_allclose(
            out_ones.numpy(), out_none.numpy(), rtol=1e-6, atol=1e-6
        )

    def test_head_mask_of_zeros_yields_only_output_bias(self):
        # Zeroing every head zeroes the attention weights, so the combined
        # context is all-zero and the output equals out_proj(0) == bias,
        # broadcast over batch/seq. Derived independently of the function.
        mha, x = self._setup()
        head_mask = paddle.zeros([1, 2, 1, 1], dtype="float32")
        out = mha_ofa_forward(mha, x, x, x, attn_mask=[None, head_mask])
        ref = mha.out_proj(paddle.zeros([2, 3, 4], dtype="float32"))
        np.testing.assert_allclose(
            out.numpy(), ref.numpy(), rtol=1e-6, atol=1e-6
        )

    def test_head_mask_zeroing_one_head(self):
        # Zero head 0, keep head 1; compare to the independent reference and
        # confirm it differs from the all-heads-active result (so the per-head
        # gating is actually consumed, not ignored).
        mha, x = self._setup()
        head_mask = paddle.to_tensor(
            np.array([0.0, 1.0], dtype="float32").reshape([1, 2, 1, 1])
        )
        out = mha_ofa_forward(mha, x, x, x, attn_mask=[None, head_mask])
        ref = _ref_attention(mha, x, x, x, head_mask=head_mask)
        np.testing.assert_allclose(
            out.numpy(), ref.numpy(), rtol=1e-5, atol=1e-6
        )

        out_full = mha_ofa_forward(mha, x, x, x, attn_mask=[None, None])
        self.assertFalse(
            np.allclose(out.numpy(), out_full.numpy(), rtol=1e-4, atol=1e-4)
        )

    def test_additive_attn_mask_is_applied_before_softmax(self):
        # A large negative additive mask on selected key positions suppresses
        # them; compare to the independent reference and confirm it changes the
        # output vs. the unmasked case.
        mha, x = self._setup()
        attn_mask = paddle.zeros([2, 2, 3, 3], dtype="float32")
        attn_mask[:, :, :, 0] = -1e9  # block attending to key position 0
        out = mha_ofa_forward(mha, x, x, x, attn_mask=[attn_mask, None])
        ref = _ref_attention(mha, x, x, x, attn_mask=attn_mask)
        np.testing.assert_allclose(
            out.numpy(), ref.numpy(), rtol=1e-5, atol=1e-6
        )

        out_unmasked = mha_ofa_forward(mha, x, x, x, attn_mask=[None, None])
        self.assertFalse(
            np.allclose(out.numpy(), out_unmasked.numpy(), rtol=1e-4, atol=1e-4)
        )


class _RecordingLayer:
    """Encoder sub-layer stub that records the ``src_mask`` it receives and
    applies a distinguishable additive transform so chaining is observable."""

    def __init__(self, add):
        self.add = add
        self.received = []

    def __call__(self, x, src_mask=None):
        self.received.append(src_mask)
        return x + self.add


class _StubEncoder:
    def __init__(self, layers, norm=None):
        self.layers = layers
        self.num_layers = len(layers)
        self.norm = norm


class _AddNorm:
    def __init__(self, add):
        self.add = add
        self.called = False

    def __call__(self, x):
        self.called = True
        return x + self.add


class TestEncoderOfaForward(unittest.TestCase):
    """encoder_ofa_forward: reshapes/broadcasts head_mask and hands the correct
    per-layer slice to each sub-layer, then applies the final norm."""

    @classmethod
    def setUpClass(cls):
        paddle.set_device("cpu")

    def test_none_head_mask_passes_none_to_each_layer(self):
        attn = paddle.to_tensor(np.zeros([1, 1, 1, 1], dtype="float32"))
        layers = [_RecordingLayer(0.0), _RecordingLayer(0.0)]
        enc = _StubEncoder(layers)
        src = paddle.to_tensor(np.arange(8, dtype="float32").reshape([1, 2, 4]))

        encoder_ofa_forward(enc, src, src_mask=[attn, None])

        for layer in layers:
            self.assertEqual(len(layer.received), 1)
            passed = layer.received[0]
            self.assertIs(passed[0], attn)  # attn_mask threaded unchanged
            self.assertIsNone(passed[1])  # no head_mask

    def test_1d_head_mask_reshaped_and_shared_across_layers(self):
        head_mask = paddle.to_tensor(np.array([2.0, 3.0, 5.0], dtype="float32"))
        layers = [_RecordingLayer(0.0), _RecordingLayer(0.0)]
        enc = _StubEncoder(layers)
        src = paddle.to_tensor(np.zeros([1, 2, 6], dtype="float32"))

        encoder_ofa_forward(enc, src, src_mask=[None, head_mask])

        expected = np.array([2.0, 3.0, 5.0], dtype="float32").reshape(
            [1, 3, 1, 1]
        )
        for layer in layers:
            hm = layer.received[0][1]
            self.assertEqual(hm.shape, [1, 3, 1, 1])
            np.testing.assert_array_equal(hm.numpy(), expected)

    def test_2d_head_mask_distributes_distinct_rows_per_layer(self):
        # Per-layer rows must stay aligned: a layer/index swap would flip the
        # rows and be caught here.
        head_mask = paddle.to_tensor(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")
        )
        layers = [_RecordingLayer(0.0), _RecordingLayer(0.0)]
        enc = _StubEncoder(layers)
        src = paddle.to_tensor(np.zeros([1, 2, 6], dtype="float32"))

        encoder_ofa_forward(enc, src, src_mask=[None, head_mask])

        np.testing.assert_array_equal(
            layers[0].received[0][1].numpy(),
            np.array([1.0, 2.0, 3.0], dtype="float32").reshape([1, 3, 1, 1]),
        )
        np.testing.assert_array_equal(
            layers[1].received[0][1].numpy(),
            np.array([4.0, 5.0, 6.0], dtype="float32").reshape([1, 3, 1, 1]),
        )

    def test_layers_chained_then_norm_applied(self):
        layers = [_RecordingLayer(1.0), _RecordingLayer(10.0)]
        norm = _AddNorm(100.0)
        enc = _StubEncoder(layers, norm=norm)
        src = paddle.to_tensor(np.arange(4, dtype="float32").reshape([1, 1, 4]))

        out = encoder_ofa_forward(enc, src, src_mask=[None, None])

        # src -> +1 (layer0) -> +10 (layer1) -> +100 (norm)
        expected = np.arange(4, dtype="float32").reshape([1, 1, 4]) + 111.0
        np.testing.assert_array_equal(out.numpy(), expected)
        self.assertTrue(norm.called)

    def test_no_norm_returns_last_layer_output(self):
        layers = [_RecordingLayer(1.0), _RecordingLayer(10.0)]
        enc = _StubEncoder(layers, norm=None)
        src = paddle.to_tensor(np.arange(4, dtype="float32").reshape([1, 1, 4]))

        out = encoder_ofa_forward(enc, src, src_mask=[None, None])

        expected = np.arange(4, dtype="float32").reshape([1, 1, 4]) + 11.0
        np.testing.assert_array_equal(out.numpy(), expected)


class TestEncoderLayerOfaForward(unittest.TestCase):
    """encoder_layer_ofa_forward reproduces the standard TransformerEncoderLayer
    forward wiring (residual, norm placement, ffn) for the no-extra-mask path."""

    @classmethod
    def setUpClass(cls):
        paddle.set_device("cpu")

    def _run(self, normalize_before):
        layer = nn.TransformerEncoderLayer(
            d_model=8,
            nhead=2,
            dim_feedforward=16,
            dropout=0.0,
            normalize_before=normalize_before,
        )
        layer.eval()  # dropout == identity, deterministic
        src = paddle.to_tensor(
            np.arange(2 * 3 * 8, dtype="float32").reshape([2, 3, 8]) / 8.0
        )
        # Reference: the library's own forward is an independent implementation
        # relative to the OFA monkey-patch under test.
        ref = layer(src)
        out = encoder_layer_ofa_forward(layer, src, src_mask=None)
        self.assertEqual(out.shape, [2, 3, 8])
        np.testing.assert_allclose(
            out.numpy(), ref.numpy(), rtol=1e-5, atol=1e-6
        )

    def test_post_norm_matches_standard_forward(self):
        self._run(normalize_before=False)

    def test_pre_norm_matches_standard_forward(self):
        self._run(normalize_before=True)


class TestPrepareQkvOfa(unittest.TestCase):
    """prepare_qkv_ofa projects and reshapes Q into [b, heads, seq, head_dim]
    and computes K/V via the layer's compute_kv (no-cache path)."""

    @classmethod
    def setUpClass(cls):
        paddle.set_device("cpu")

    def test_qkv_layout_no_cache(self):
        mha = nn.MultiHeadAttention(embed_dim=4, num_heads=2)
        _fill_linear(mha.q_proj, base=0.0)
        _fill_linear(mha.k_proj, base=10.0)
        _fill_linear(mha.v_proj, base=20.0)
        query = paddle.to_tensor(
            np.arange(2 * 3 * 4, dtype="float32").reshape([2, 3, 4]) / 10.0
        )
        key = query + 1.0
        value = query + 2.0

        q, k, v = prepare_qkv_ofa(mha, query, key, value, cache=None)

        # Independent Q reference: project then split heads and move head axis
        # ahead of the sequence axis.
        q_ref = mha.q_proj(query)
        q_ref = paddle.reshape(q_ref, shape=[0, 0, mha.num_heads, mha.head_dim])
        q_ref = paddle.transpose(q_ref, perm=[0, 2, 1, 3])
        k_ref, v_ref = mha.compute_kv(key, value)

        self.assertEqual(q.shape, [2, 2, 3, 2])  # [b, heads, seq, head_dim]
        np.testing.assert_allclose(
            q.numpy(), q_ref.numpy(), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            k.numpy(), k_ref.numpy(), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            v.numpy(), v_ref.numpy(), rtol=1e-6, atol=1e-6
        )


class _TinyImportanceModel(nn.Layer):
    """Minimal model whose output genuinely depends on the head_mask, so a real
    backward populates head_mask.gradient(); its ffn Linears are named
    ``linear1``/``linear2`` to match compute_neuron_head_importance's scan."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 3)
        self.linear2 = nn.Linear(3, 4)

    def forward(self, input_ids, segment_ids=None, attention_mask=None):
        h = self.linear2(self.linear1(input_ids))
        if attention_mask is not None and attention_mask[1] is not None:
            # head_mask enters the graph so its gradient is well defined.
            h = h * attention_mask[1].sum()
        return h


class TestComputeNeuronHeadImportance(unittest.TestCase):
    """compute_neuron_head_importance: parameter scan, real backward-driven
    accumulation, and the loss_fct=None exception contract."""

    @classmethod
    def setUpClass(cls):
        paddle.set_device("cpu")

    def _batch(self):
        input_ids = paddle.to_tensor(
            np.arange(2 * 4, dtype="float32").reshape([2, 4]) / 4.0
        )
        segment_ids = paddle.zeros([2, 4], dtype="float32")
        labels = paddle.to_tensor([0, 1], dtype="int64")
        return [[input_ids, segment_ids, labels]]

    def test_missing_loss_fct_raises_not_implemented(self):
        # Explicit exception contract (not a swallowed error): with loss_fct
        # None the function must raise NotImplementedError once it reaches a
        # batch, after having run the model forward.
        model = _TinyImportanceModel()
        with self.assertRaises(NotImplementedError):
            compute_neuron_head_importance(
                model,
                self._batch(),
                num_layers=1,
                num_heads=2,
                loss_fct=None,
            )

    def test_importance_accumulated_from_real_gradients(self):
        model = _TinyImportanceModel()

        def loss_fct(logits, labels):
            return (logits**2).mean()

        head_importance, neuron_importance = compute_neuron_head_importance(
            model,
            self._batch(),
            num_layers=1,
            num_heads=2,
            loss_fct=loss_fct,
        )

        # head_importance has the requested [num_layers, num_heads] shape.
        self.assertEqual(list(head_importance.shape), [1, 2])
        hi = head_importance.numpy()
        # Each head_mask entry scales the output identically, so their gradients
        # (hence importances) are equal and strictly positive -- a wrong reshape
        # of head_mask.gradient() would break this equality.
        self.assertGreater(float(hi.min()), 0.0)
        np.testing.assert_allclose(hi[0, 0], hi[0, 1], rtol=1e-6, atol=1e-8)

        # One intermediate Linear (linear1, out_features=3) => one importance
        # vector of length 3, non-negative (accumulated absolute values).
        self.assertEqual(len(neuron_importance), 1)
        ni = neuron_importance[0]
        self.assertEqual(list(ni.shape), [3])
        self.assertEqual(ni.dtype, np.float32)
        self.assertTrue(np.all(ni >= 0.0))
        self.assertGreater(float(ni.max()), 0.0)


if __name__ == "__main__":
    unittest.main()
