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
"""Tests for ``SelfAttention._maybe_tag_qkv_dgrad_groups``.

HF's ``Qwen3_5MoeAttention`` has three ``nn.Linear`` modules -- ``q_proj``
(emitting query *and* gate per head), ``k_proj`` and ``v_proj`` -- so the gradient
w.r.t. their shared input is a chain of three narrow GEMMs. PaddleFleet fuses
them into one projection whose dgrad is the same sum computed as a single wide-K
GEMM, and BF16 GEMM K-splitting is not associative, so the two differ in the last
mantissa bit. This method records which fused columns belong to each reference
projection so the linear backward can reproduce the split.

Two orderings matter and are asserted separately:

* ``hf_dgrad_groups`` is ``v, k, q`` -- torch accumulates a multiply-used
  tensor's gradient in **reverse** module-creation order.
* ``hf_norm_groups`` is ``q, k, v`` -- gradient clipping takes one per-tensor
  norm per ``nn.Linear``, in forward order.

Inside a group, ``q_cols`` must interleave query and gate per head, because HF's
``q_proj`` emits ``[query, gate]`` head by head while PaddleFleet lays out all
queries of a group before all gates.

The method is exercised unbound on a namespace: it reads only a handful of
attributes, and constructing a full ``SelfAttention`` (which needs a spec, a
process group and ~25 config fields) would test the constructor rather than the
column algebra this file is about.
"""

import os
import sys
import unittest
from types import SimpleNamespace

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

from paddlefleet.transformer.attention import SelfAttention

_TAG = SelfAttention._maybe_tag_qkv_dgrad_groups


def _stub(
    *,
    target="hf",
    heads=4,
    groups=2,
    head_dim=8,
    v_head_dim=8,
    gated=True,
    with_qkv=True,
    with_weight=True,
    experimental=False,
    hidden=16,
):
    """Namespace carrying exactly the attributes the method reads."""
    heads_per_group = heads // groups
    q_dim = heads_per_group * head_dim
    gate_dim = heads_per_group * v_head_dim
    group_dim = q_dim + gate_dim + head_dim + v_head_dim
    width = groups * group_dim
    weight = paddle.zeros([hidden, width], dtype=paddle.float32)
    qkv = SimpleNamespace(weight=weight) if with_weight else SimpleNamespace()
    return (
        SimpleNamespace(
            config=SimpleNamespace(
                use_accuracy_compatible=target,
                gpt_model_use_experimental_version=experimental,
            ),
            gated_attention=gated,
            qkv_proj=qkv if with_qkv else None,
            num_attention_heads_per_partition=heads,
            num_query_groups_per_partition=groups,
            hidden_size_per_attention_head=head_dim,
            value_hidden_size_per_attention_head=v_head_dim,
        ),
        weight,
        width,
    )


class TestQKVDgradGroupsGating(unittest.TestCase):
    """Every early return leaves the weight untouched."""

    def test_non_hf_targets_do_not_tag(self):
        for target in (False, True, "megatron"):
            with self.subTest(target=target):
                stub, weight, _ = _stub(target=target)
                _TAG(stub)
                self.assertIsNone(
                    getattr(weight, "hf_dgrad_groups", None), target
                )
                self.assertIsNone(
                    getattr(weight, "hf_norm_groups", None), target
                )

    def test_experimental_version_does_not_tag(self):
        """The experimental model has a different projection layout."""
        stub, weight, _ = _stub(experimental=True)
        _TAG(stub)
        self.assertIsNone(getattr(weight, "hf_dgrad_groups", None))

    def test_ungated_attention_does_not_tag(self):
        """Without the gate the fused layout has no ``[query, gate]`` pairing."""
        stub, weight, _ = _stub(gated=False)
        _TAG(stub)
        self.assertIsNone(getattr(weight, "hf_dgrad_groups", None))

    def test_missing_qkv_proj_is_tolerated(self):
        stub, _, _ = _stub(with_qkv=False)
        _TAG(stub)

    def test_missing_weight_is_tolerated(self):
        stub, _, _ = _stub(with_weight=False)
        _TAG(stub)


class TestQKVDgradGroupsLayout(unittest.TestCase):
    """The column algebra: partition, ordering, and per-head interleaving."""

    def test_hf_target_tags_three_groups(self):
        stub, weight, _ = _stub()
        _TAG(stub)
        self.assertEqual(len(weight.hf_dgrad_groups), 3)
        self.assertEqual(len(weight.hf_norm_groups), 3)

    def test_groups_partition_every_column_exactly_once(self):
        """A missed or duplicated column would silently change the dgrad."""
        stub, weight, width = _stub()
        _TAG(stub)
        merged = np.concatenate([g.numpy() for g in weight.hf_dgrad_groups])
        np.testing.assert_array_equal(np.sort(merged), np.arange(width))

    def test_dgrad_order_is_v_k_q(self):
        """Reverse module-creation order, which is how torch accumulates."""
        stub, weight, _ = _stub()
        _TAG(stub)
        v, k, q = (g.numpy() for g in weight.hf_dgrad_groups)
        qn, kn, vn = (g.numpy() for g in weight.hf_norm_groups)
        np.testing.assert_array_equal(v, vn)
        np.testing.assert_array_equal(k, kn)
        np.testing.assert_array_equal(q, qn)

    def test_norm_order_is_q_k_v(self):
        """Forward order, which is how the clip walks the ``nn.Linear``s."""
        stub, weight, _ = _stub()
        _TAG(stub)
        dgrad = [g.numpy() for g in weight.hf_dgrad_groups]
        norm = [g.numpy() for g in weight.hf_norm_groups]
        self.assertFalse(np.array_equal(dgrad[0], norm[0]))
        np.testing.assert_array_equal(dgrad[0], norm[2])
        np.testing.assert_array_equal(dgrad[2], norm[0])

    def test_group_sizes_match_the_reference_projections(self):
        """q carries query+gate per head; k and v carry one head each."""
        heads, groups, head_dim, v_head_dim = 4, 2, 8, 8
        stub, weight, _ = _stub(
            heads=heads,
            groups=groups,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
        )
        _TAG(stub)
        v, k, q = (g.numpy() for g in weight.hf_dgrad_groups)
        heads_per_group = heads // groups
        self.assertEqual(
            q.size, groups * heads_per_group * (head_dim + v_head_dim)
        )
        self.assertEqual(k.size, groups * head_dim)
        self.assertEqual(v.size, groups * v_head_dim)

    def test_q_columns_interleave_query_and_gate_per_head(self):
        """HF's ``q_proj`` emits ``[query, gate]`` head by head."""
        heads, groups, head_dim, v_head_dim = 2, 1, 4, 4
        stub, weight, _ = _stub(
            heads=heads,
            groups=groups,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
        )
        _TAG(stub)
        q = weight.hf_dgrad_groups[2].numpy()
        # Fused layout for one group: q0 q1 | gate0 gate1 | k | v
        # so q_proj's own order is q0, gate0, q1, gate1.
        expected = np.array(
            [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(q, expected)

    def test_k_and_v_are_disjoint_and_after_the_gate(self):
        stub, weight, _ = _stub(heads=2, groups=1, head_dim=4, v_head_dim=4)
        _TAG(stub)
        v, k, q = (g.numpy() for g in weight.hf_dgrad_groups)
        self.assertEqual(set(k.tolist()) & set(v.tolist()), set())
        self.assertTrue(k.min() > q.max() - q.size)
        np.testing.assert_array_equal(k, np.array([16, 17, 18, 19]))
        np.testing.assert_array_equal(v, np.array([20, 21, 22, 23]))

    def test_indices_are_int64(self):
        """``index_select`` in the linear backward requires int64."""
        stub, weight, _ = _stub()
        _TAG(stub)
        for group in weight.hf_dgrad_groups + weight.hf_norm_groups:
            self.assertEqual(group.dtype, paddle.int64)

    def test_asymmetric_value_head_dim(self):
        """MLA-style configs use a different value head dim; layout must hold."""
        stub, weight, width = _stub(heads=4, groups=2, head_dim=8, v_head_dim=4)
        _TAG(stub)
        merged = np.concatenate([g.numpy() for g in weight.hf_dgrad_groups])
        np.testing.assert_array_equal(np.sort(merged), np.arange(width))
        v, k, _ = (g.numpy() for g in weight.hf_dgrad_groups)
        self.assertEqual(v.size, 2 * 4)
        self.assertEqual(k.size, 2 * 8)

    def test_single_group_is_multi_head_attention(self):
        """groups == heads means one head per group (no GQA sharing)."""
        stub, weight, width = _stub(heads=3, groups=3, head_dim=4, v_head_dim=4)
        _TAG(stub)
        merged = np.concatenate([g.numpy() for g in weight.hf_dgrad_groups])
        np.testing.assert_array_equal(np.sort(merged), np.arange(width))


class _BiasedLinear(paddle.nn.Layer):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.linear = paddle.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x), self.linear.bias


class _RMSNorm(paddle.nn.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        hidden = kwargs.get("normalized_shape", kwargs.get("hidden_size"))
        self.weight = paddle.nn.Parameter(paddle.zeros([hidden]))
        self.eps = kwargs.get("norm_eps", kwargs.get("eps"))
        #: Recorded so the test can see what the attention passed down.
        self.head_major_grad = kwargs.get("head_major_grad", "absent")

    def forward(self, x):
        d = paddle.rsqrt(x.pow(2).mean(axis=-1, keepdim=True) + self.eps)
        return x * d * self.weight


class TestQKNormHeadMajorKwarg(unittest.TestCase):
    """``head_major_grad`` is forwarded to the q/k norms only under ``"hf"``.

    It is passed as ``**kwargs`` so a default run's norm construction stays
    byte-identical to before the alignment work; these tests pin both halves of
    that contract on a real ``SelfAttention``.
    """

    def _build(self, target, qk_norm_type="per_head"):
        from paddlefleet.transformer.attention import (
            SelfAttention,
            SelfAttentionSublayersSpec,
        )
        from paddlefleet.transformer.dot_product_attention import (
            DotProductAttention,
        )
        from paddlefleet.transformer.enums import AttnMaskType
        from paddlefleet.transformer.transformer_config import (
            TransformerConfig,
        )
        from paddlefleet.utils import (
            init_method_normal,
            scaled_init_method_normal,
        )

        config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            use_accuracy_compatible=target,
        )
        config.num_key_value_heads = config.num_attention_heads
        config.head_dim = config.hidden_size // config.num_attention_heads
        config.softmax_scale = None
        config.use_bias = True
        config.no_rope_freq = None
        config.recompute_granularity = None
        config.fused_single_qkv_rope = False
        config.rotary_interleaved = False
        config.multi_latent_attention = False
        config.init_method = init_method_normal(0.02)
        config.output_layer_init_method = scaled_init_method_normal(
            0.02, 1, 2.0
        )
        config.rms_norm_eps = 1e-5
        config.context_parallel_size = 1
        config.apply_query_key_layer_scaling = False
        config.sliding_window = None
        config.window_attn_skip_freq = None
        config.fp16 = False
        config.bf16 = False
        config.masked_softmax_fusion = False
        config.attention_softmax_in_fp32 = True
        config.attention_dropout = 0.1
        config.softmax_type = "vanilla"
        config.qk_norm_type = qk_norm_type

        return SelfAttention(
            config,
            SelfAttentionSublayersSpec(
                qkv_proj=_BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=_BiasedLinear,
                q_norm=_RMSNorm,
                k_norm=_RMSNorm,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=1,
        )

    def test_hf_per_head_passes_head_major_true(self):
        attn = self._build("hf", "per_head")
        self.assertIs(attn.q_norm.head_major_grad, True)
        self.assertIs(attn.k_norm.head_major_grad, True)

    def test_hf_per_layer_passes_head_major_false(self):
        """A per-layer norm is not consumed head-major."""
        attn = self._build("hf", "per_layer")
        self.assertIs(attn.q_norm.head_major_grad, False)
        self.assertIs(attn.k_norm.head_major_grad, False)

    def test_non_hf_targets_omit_the_kwarg_entirely(self):
        """The default path's norm construction must be unchanged."""
        for target in (False, True, "megatron"):
            with self.subTest(target=target):
                attn = self._build(target)
                self.assertEqual(attn.q_norm.head_major_grad, "absent")
                self.assertEqual(attn.k_norm.head_major_grad, "absent")


if __name__ == "__main__":
    unittest.main()
