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

"""Tests for muon orthogonal-slice specs and the muon_utils slice functions.

The slice functions are pure tensor transforms, so they are tested directly
on synthetic weights. The per-module ``muon_slice_specs`` methods only read a
handful of attributes from ``self``, so they are invoked unbound with a
lightweight namespace object standing in for the layer instance; every
returned (slice_fn, kwargs) pair is then executed against a weight tensor of
the matching shape to validate that the split sizes are consistent.
"""

import unittest
from types import SimpleNamespace

import paddle

from paddlefleet.transformer.attention import (
    SelfAttention,
    SelfAttentionVHA,
)
from paddlefleet.transformer.csa_attention import Compressor, CSAIndexer
from paddlefleet.transformer.dsv4_hybrid_attention import (
    DSv4HybridSelfAttention,
)
from paddlefleet.transformer.mlp import MLP
from paddlefleet.transformer.moe.moe_expert import GroupedMLPExpert
from paddlefleet.transformer.multi_latent_attention import (
    MLASelfAttention,
)
from paddlefleet.transformer.muon_utils import (
    ortho_blocks,
    ortho_gate_up,
    ortho_per_head,
    ortho_qkv_contiguous,
    ortho_qkv_interleaved,
    ortho_stacked,
)

HIDDEN = 32


class _OrthoRecorder:
    """Identity ortho_fn that records the shape of every slice it receives."""

    def __init__(self):
        self.shapes = []

    def __call__(self, x):
        self.shapes.append(tuple(x.shape))
        return x


def _run_specs(specs, weight_shapes):
    """Execute every (slice_fn, kwargs) spec against its weight tensor."""
    outputs = {}
    for name, (fn, kwargs) in specs.items():
        ortho = _OrthoRecorder()
        weight = paddle.randn(weight_shapes[name])
        out = fn(weight, ortho, **kwargs)
        assert tuple(out.shape) == tuple(weight.shape), (
            f"{name}: output shape {out.shape} != input {weight.shape}"
        )
        outputs[name] = ortho
    return outputs


class TestMuonUtils(unittest.TestCase):
    def test_qkv_interleaved_per_head_non_gated(self):
        groups, heads_per_group, head_dim, v_head_dim = 2, 3, 4, 4
        split_dims = [heads_per_group * head_dim, head_dim, v_head_dim]
        width = groups * sum(split_dims)
        ortho = _OrthoRecorder()
        w = paddle.randn([HIDDEN, width])
        out = ortho_qkv_interleaved(
            w,
            ortho,
            groups=groups,
            role_sizes=split_dims,
            heads_per_group=heads_per_group,
        )
        self.assertEqual(tuple(out.shape), tuple(w.shape))
        # per group: heads_per_group Q slices + 1 K + 1 V
        self.assertEqual(len(ortho.shapes), groups * (heads_per_group + 2))

    def test_qkv_interleaved_per_head_gated(self):
        groups, heads_per_group, head_dim, v_head_dim = 2, 2, 4, 4
        split_dims = [
            heads_per_group * head_dim,
            heads_per_group * v_head_dim,
            head_dim,
            v_head_dim,
        ]
        width = groups * sum(split_dims)
        ortho = _OrthoRecorder()
        w = paddle.randn([HIDDEN, width])
        out = ortho_qkv_interleaved(
            w,
            ortho,
            groups=groups,
            role_sizes=split_dims,
            heads_per_group=heads_per_group,
        )
        self.assertEqual(tuple(out.shape), tuple(w.shape))
        # per group: Q heads + gate heads + K + V
        self.assertEqual(len(ortho.shapes), groups * (2 * heads_per_group + 2))

    def test_qkv_interleaved_per_role_non_gated(self):
        groups, heads_per_group, head_dim = 2, 2, 4
        split_dims = [heads_per_group * head_dim, head_dim, head_dim]
        w = paddle.randn([HIDDEN, groups * sum(split_dims)])
        ortho = _OrthoRecorder()
        out = ortho_qkv_interleaved(
            w, ortho, groups=groups, role_sizes=split_dims, per_head=False
        )
        self.assertEqual(tuple(out.shape), tuple(w.shape))
        # whole Q / K / V blocks
        self.assertEqual(len(ortho.shapes), 3)

    def test_qkv_interleaved_per_role_gated(self):
        groups, heads_per_group, head_dim = 2, 2, 4
        split_dims = [
            heads_per_group * head_dim,
            heads_per_group * head_dim,
            head_dim,
            head_dim,
        ]
        w = paddle.randn([HIDDEN, groups * sum(split_dims)])
        ortho = _OrthoRecorder()
        out = ortho_qkv_interleaved(
            w, ortho, groups=groups, role_sizes=split_dims, per_head=False
        )
        self.assertEqual(tuple(out.shape), tuple(w.shape))
        # whole Q / Gate / K / V blocks
        self.assertEqual(len(ortho.shapes), 4)

    def test_qkv_contiguous_per_head(self):
        num_heads, num_kv, head_dim, v_head_dim = 6, 2, 4, 4
        width = num_heads * head_dim + num_kv * head_dim + num_kv * v_head_dim
        w = paddle.randn([HIDDEN, width])
        ortho = _OrthoRecorder()
        out = ortho_qkv_contiguous(
            w,
            ortho,
            heads=num_heads,
            groups=num_kv,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
        )
        self.assertEqual(tuple(out.shape), tuple(w.shape))
        self.assertEqual(len(ortho.shapes), num_heads + 2 * num_kv)
        self.assertTrue(all(s == (HIDDEN, head_dim) for s in ortho.shapes))

    def test_ortho_blocks_int_and_list_sizes(self):
        w = paddle.randn([HIDDEN, 12])
        ortho = _OrthoRecorder()
        out = ortho_blocks(w, ortho, 3)
        self.assertEqual(tuple(out.shape), tuple(w.shape))
        self.assertEqual(ortho.shapes, [(HIDDEN, 4)] * 3)

        ortho = _OrthoRecorder()
        out = ortho_blocks(w, ortho, [2, 4, 6])
        self.assertEqual(tuple(out.shape), tuple(w.shape))
        self.assertEqual(ortho.shapes, [(HIDDEN, 2), (HIDDEN, 4), (HIDDEN, 6)])

    def test_gate_up_2d_and_3d(self):
        for shape in ([HIDDEN, 16], [2, HIDDEN, 16]):
            ortho = _OrthoRecorder()
            w = paddle.randn(shape)
            out = ortho_gate_up(w, ortho)
            self.assertEqual(tuple(out.shape), tuple(w.shape))
            self.assertEqual(len(ortho.shapes), 2)

    def test_gate_up_rejects_bad_ndim(self):
        with self.assertRaises(AssertionError):
            ortho_gate_up(paddle.randn([8]), lambda x: x)

    def test_per_head_even_and_sized(self):
        w = paddle.randn([HIDDEN, 12])
        ortho = _OrthoRecorder()
        out = ortho_per_head(w, ortho, heads=3)
        self.assertEqual(tuple(out.shape), tuple(w.shape))
        self.assertEqual(len(ortho.shapes), 3)

        ortho = _OrthoRecorder()
        out = ortho_per_head(w, ortho, heads=2, head_sizes=[2, 4])
        self.assertEqual(tuple(out.shape), tuple(w.shape))
        self.assertEqual(ortho.shapes, [(HIDDEN, 2), (HIDDEN, 4)] * 2)

    def test_stacked(self):
        w = paddle.randn([2, HIDDEN, 8])
        ortho = _OrthoRecorder()
        out = ortho_stacked(w, ortho)
        self.assertEqual(tuple(out.shape), tuple(w.shape))
        self.assertEqual(len(ortho.shapes), 1)

    def test_stacked_rejects_non_3d(self):
        with self.assertRaises(ValueError):
            ortho_stacked(paddle.randn([HIDDEN, 8]), lambda x: x)


class TestSelfAttentionSpecs(unittest.TestCase):
    def _fake(self, experimental=False, gated=False):
        return SimpleNamespace(
            config=SimpleNamespace(
                gpt_model_use_experimental_version=experimental
            ),
            gated_attention=gated,
            num_attention_heads_per_partition=4,
            num_query_groups_per_partition=2,
            hidden_size_per_attention_head=4,
            value_hidden_size_per_attention_head=4,
        )

    def _qkv_width(self, fake):
        heads = fake.num_attention_heads_per_partition
        groups = fake.num_query_groups_per_partition
        head_dim = fake.hidden_size_per_attention_head
        v_head_dim = fake.value_hidden_size_per_attention_head
        width = heads * head_dim + groups * (head_dim + v_head_dim)
        if fake.gated_attention:
            width += heads * v_head_dim
        return width

    def test_default_split_head(self):
        fake = self._fake()
        specs = SelfAttention.muon_slice_specs(fake, {})
        self.assertEqual(set(specs), {"qkv_proj.weight"})
        _run_specs(specs, {"qkv_proj.weight": [HIDDEN, self._qkv_width(fake)]})

    def test_gated_split_qkv(self):
        fake = self._fake(gated=True)
        specs = SelfAttention.muon_slice_specs(
            fake, {"muon_qkv_update_mode": "split_qkv"}
        )
        self.assertEqual(set(specs), {"qkv_proj.weight"})
        _run_specs(specs, {"qkv_proj.weight": [HIDDEN, self._qkv_width(fake)]})

    def test_experimental_non_gated(self):
        fake = self._fake(experimental=True)
        specs = SelfAttention.muon_slice_specs(fake, {})
        self.assertEqual(set(specs), {"qkv_proj.weight"})
        _run_specs(specs, {"qkv_proj.weight": [HIDDEN, self._qkv_width(fake)]})

    def test_experimental_gated_has_separate_gate_spec(self):
        fake = self._fake(experimental=True, gated=True)
        specs = SelfAttention.muon_slice_specs(fake, {})
        self.assertEqual(set(specs), {"qkv_proj.weight", "gate_proj.weight"})
        heads = fake.num_attention_heads_per_partition
        groups = fake.num_query_groups_per_partition
        head_dim = fake.hidden_size_per_attention_head
        v_head_dim = fake.value_hidden_size_per_attention_head
        # EC qkv_proj carries no gate columns; gate is a separate projection.
        qkv_width = heads * head_dim + groups * (head_dim + v_head_dim)
        _run_specs(
            specs,
            {
                "qkv_proj.weight": [HIDDEN, qkv_width],
                "gate_proj.weight": [HIDDEN, heads * v_head_dim],
            },
        )

    def test_experimental_split_qkv_unsupported(self):
        fake = self._fake(experimental=True, gated=True)
        specs = SelfAttention.muon_slice_specs(
            fake, {"muon_qkv_update_mode": "split_qkv"}
        )
        self.assertEqual(specs, {})

    def test_unknown_mode(self):
        specs = SelfAttention.muon_slice_specs(
            self._fake(), {"muon_qkv_update_mode": "none"}
        )
        self.assertEqual(specs, {})


class TestSelfAttentionVHASpecs(unittest.TestCase):
    def _fake(self):
        # VHA overrides num_attention_heads_per_partition to mean the number of
        # query heads *per group* (num_attention_heads // num_key_value_heads),
        # so keep it distinct from num_attention_heads here: a spec that grabs
        # the wrong one then fails on width instead of passing silently.
        return SimpleNamespace(
            num_attention_heads_per_partition=2,
            num_query_groups_per_partition=2,
            num_attention_heads=4,
            q_head_dim=4,
            head_dim=4,
            v_head_dim=4,
            shared_kv_proj=object(),
            k_proj=object(),
            v_proj=object(),
            gate_proj=object(),
            vha_premix_weight=object(),
        )

    def test_full_specs(self):
        fake = self._fake()
        specs = SelfAttentionVHA.muon_slice_specs(fake, {})
        self.assertEqual(
            set(specs),
            {
                "q_proj.weight",
                "shared_kv_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "gate_proj.weight",
                "vha_premix_weight",
            },
        )
        _run_specs(
            specs,
            {
                # q is low-rank: width == heads_per_group * q_head_dim
                "q_proj.weight": [HIDDEN, 2 * 4],
                "shared_kv_proj.weight": [HIDDEN, 2 * 4],
                "k_proj.weight": [HIDDEN, 2 * 4],
                "v_proj.weight": [HIDDEN, 2 * 4],
                # gate spans all heads: num_attention_heads * v_head_dim
                "gate_proj.weight": [HIDDEN, 4 * 4],
                "vha_premix_weight": [2, 4, 4],
            },
        )

    def test_minimal_specs(self):
        fake = self._fake()
        del fake.shared_kv_proj, fake.k_proj, fake.v_proj
        del fake.vha_premix_weight
        fake.gate_proj = None
        specs = SelfAttentionVHA.muon_slice_specs(fake, {})
        self.assertEqual(set(specs), {"q_proj.weight"})

    def test_non_split_head_mode(self):
        specs = SelfAttentionVHA.muon_slice_specs(
            self._fake(), {"muon_qkv_update_mode": "split_qkv"}
        )
        self.assertEqual(specs, {})


class TestMLASpecs(unittest.TestCase):
    def _fake(self, sparse_gated=False):
        fake = SimpleNamespace(
            config=SimpleNamespace(kv_lora_rank=8),
            kv_lora_rank=8,
            num_attention_heads_per_partition=2,
            qk_nope_head_dim=4,
            qk_rope_head_dim=2,
            v_head_dim=4,
            q_b_proj=object(),
            gate_proj=object(),
        )
        if sparse_gated:
            # Only MQASelfAttention builds this second gated branch.
            fake.sparse_gate_proj = object()
        return fake

    def test_full_specs(self):
        fake = self._fake()
        specs = MLASelfAttention.muon_slice_specs(fake, {})
        self.assertEqual(
            set(specs),
            {
                "q_b_proj.weight",
                "kv_a_proj_with_mqa.weight",
                "kv_b_proj.weight",
                "gate_proj.weight",
            },
        )
        _run_specs(
            specs,
            {
                "q_b_proj.weight": [HIDDEN, 2 * (4 + 2)],
                "kv_a_proj_with_mqa.weight": [HIDDEN, 8 + 2],
                "kv_b_proj.weight": [HIDDEN, 2 * (4 + 4)],
                "gate_proj.weight": [HIDDEN, 2 * 4],
            },
        )

    def test_uses_effective_kv_lora_rank(self):
        fake = SimpleNamespace(
            config=SimpleNamespace(kv_lora_rank=99),
            kv_lora_rank=8,
            num_attention_heads_per_partition=2,
            qk_nope_head_dim=4,
            qk_rope_head_dim=2,
            v_head_dim=4,
            q_b_proj=object(),
            gate_proj=None,
        )
        specs = MLASelfAttention.muon_slice_specs(fake, {})
        self.assertEqual(
            specs["kv_a_proj_with_mqa.weight"][1]["head_sizes"], [8, 2]
        )
        self.assertEqual(specs["kv_b_proj.weight"][1]["head_sizes"], [4, 4])

    def test_without_optional_projections(self):
        fake = self._fake()
        del fake.q_b_proj
        fake.gate_proj = None
        specs = MLASelfAttention.muon_slice_specs(fake, {})
        self.assertEqual(
            set(specs),
            {"kv_a_proj_with_mqa.weight", "kv_b_proj.weight"},
        )

    def test_sparse_gate_proj_absent_is_guarded(self):
        # Plain MLA has no block-sparse branch, so no spec should appear.
        specs = MLASelfAttention.muon_slice_specs(self._fake(), {})
        self.assertNotIn("sparse_gate_proj.weight", specs)

    def test_sparse_gate_proj_sliced_per_head(self):
        # MQA subclass adds a second gated branch whose weight mirrors
        # gate_proj: width == heads * v_head_dim, sliced into per-head blocks.
        fake = self._fake(sparse_gated=True)
        specs = MLASelfAttention.muon_slice_specs(fake, {})
        self.assertIn("sparse_gate_proj.weight", specs)
        self.assertEqual(
            specs["sparse_gate_proj.weight"], specs["gate_proj.weight"]
        )
        recorders = _run_specs(
            specs,
            {
                "q_b_proj.weight": [HIDDEN, 2 * (4 + 2)],
                "kv_a_proj_with_mqa.weight": [HIDDEN, 8 + 2],
                "kv_b_proj.weight": [HIDDEN, 2 * (4 + 4)],
                "gate_proj.weight": [HIDDEN, 2 * 4],
                "sparse_gate_proj.weight": [HIDDEN, 2 * 4],
            },
        )
        self.assertEqual(
            recorders["sparse_gate_proj.weight"].shapes,
            [(HIDDEN, 4), (HIDDEN, 4)],
        )

    def test_non_split_head_mode(self):
        specs = MLASelfAttention.muon_slice_specs(
            self._fake(), {"muon_qkv_update_mode": "split_qkv"}
        )
        self.assertEqual(specs, {})


class TestDSv4HybridSpecs(unittest.TestCase):
    def _fake(self, gated=False):
        return SimpleNamespace(
            num_attention_heads_per_partition=4,
            o_local_groups=2,
            use_vha_premix=False,
            gate_proj=object() if gated else None,
        )

    def test_specs(self):
        fake = self._fake()
        specs = DSv4HybridSelfAttention.muon_slice_specs(fake, {})
        self.assertEqual(
            set(specs), {"linear_q_up_proj.weight", "linear_o_group_proj"}
        )
        recorders = _run_specs(
            specs,
            {
                "linear_q_up_proj.weight": [HIDDEN, 4 * 4],
                # [o_local_groups * o_lora_rank, d], split along axis 0.
                "linear_o_group_proj": [2 * 3, HIDDEN],
            },
        )
        self.assertEqual(
            recorders["linear_o_group_proj"].shapes,
            [(3, HIDDEN), (3, HIDDEN)],
        )

    def test_gated_slices_gate_per_o_group(self):
        fake = self._fake(gated=True)
        specs = DSv4HybridSelfAttention.muon_slice_specs(fake, {})
        self.assertEqual(
            set(specs),
            {
                "linear_q_up_proj.weight",
                "linear_o_group_proj",
                "gate_proj.weight",
            },
        )
        # gate width == o_local_groups * o_lora_rank, sliced into o_lora_rank
        # wide blocks.
        recorders = _run_specs(
            specs,
            {
                "linear_q_up_proj.weight": [HIDDEN, 4 * 4],
                "linear_o_group_proj": [2 * 3, HIDDEN],
                "gate_proj.weight": [HIDDEN, 2 * 3],
            },
        )
        self.assertEqual(
            recorders["gate_proj.weight"].shapes,
            [(HIDDEN, 3), (HIDDEN, 3)],
        )

    def test_gate_proj_absent_is_guarded(self):
        specs = DSv4HybridSelfAttention.muon_slice_specs(self._fake(), {})
        self.assertNotIn("gate_proj.weight", specs)

    def test_non_split_head_mode(self):
        specs = DSv4HybridSelfAttention.muon_slice_specs(
            self._fake(gated=True), {"muon_qkv_update_mode": "split_qkv"}
        )
        self.assertEqual(specs, {})


class TestMLPSpecs(unittest.TestCase):
    def test_gated_with_ffn_split(self):
        fake = SimpleNamespace(config=SimpleNamespace(gated_linear_unit=True))
        specs = MLP.muon_slice_specs(fake, {"muon_ffn_split": True})
        self.assertEqual(set(specs), {"up_gate_proj.weight"})
        _run_specs(specs, {"up_gate_proj.weight": [HIDDEN, 16]})

    def test_non_gated_is_guarded(self):
        fake = SimpleNamespace(config=SimpleNamespace(gated_linear_unit=False))
        specs = MLP.muon_slice_specs(fake, {"muon_ffn_split": True})
        self.assertEqual(specs, {})

    def test_ffn_split_off(self):
        fake = SimpleNamespace(config=SimpleNamespace(gated_linear_unit=True))
        specs = MLP.muon_slice_specs(fake, {})
        self.assertEqual(specs, {})


class TestGroupedMLPExpertSpecs(unittest.TestCase):
    def test_gated_with_ffn_split(self):
        fake = SimpleNamespace(config=SimpleNamespace(gated_linear_unit=True))
        specs = GroupedMLPExpert.muon_slice_specs(
            fake, {"muon_ffn_split": True}
        )
        self.assertEqual(set(specs), {"weight1", "weight2"})
        _run_specs(
            specs,
            {"weight1": [2, HIDDEN, 16], "weight2": [2, 8, HIDDEN]},
        )

    def test_non_gated_keeps_weight2_only(self):
        fake = SimpleNamespace(config=SimpleNamespace(gated_linear_unit=False))
        specs = GroupedMLPExpert.muon_slice_specs(
            fake, {"muon_ffn_split": True}
        )
        self.assertEqual(set(specs), {"weight2"})


class TestCSASpecs(unittest.TestCase):
    def test_compressor_overlap(self):
        fake = SimpleNamespace(overlap=True, head_dim=4)
        specs = Compressor.muon_slice_specs(fake, {})
        self.assertEqual(
            set(specs), {"linear_wkv.weight", "linear_wgate.weight"}
        )
        _run_specs(
            specs,
            {
                "linear_wkv.weight": [HIDDEN, 8],
                "linear_wgate.weight": [HIDDEN, 8],
            },
        )

    def test_compressor_no_overlap(self):
        fake = SimpleNamespace(overlap=False, head_dim=4)
        specs = Compressor.muon_slice_specs(fake, {})
        self.assertEqual(specs, {})

    def test_compressor_non_split_head_mode(self):
        fake = SimpleNamespace(overlap=True, head_dim=4)
        specs = Compressor.muon_slice_specs(
            fake, {"muon_qkv_update_mode": "split_qkv"}
        )
        self.assertEqual(specs, {})

    def test_indexer(self):
        fake = SimpleNamespace(index_n_heads=2)
        specs = CSAIndexer.muon_slice_specs(fake, {})
        self.assertEqual(set(specs), {"linear_wq_b.weight"})
        _run_specs(specs, {"linear_wq_b.weight": [HIDDEN, 8]})

    def test_indexer_non_split_head_mode(self):
        fake = SimpleNamespace(index_n_heads=2)
        specs = CSAIndexer.muon_slice_specs(
            fake, {"muon_qkv_update_mode": "split_qkv"}
        )
        self.assertEqual(specs, {})


if __name__ == "__main__":
    unittest.main()
