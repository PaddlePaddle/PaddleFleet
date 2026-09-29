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
#
# Scope: ``MLASelfAttention``'s only component AOA override, the
# ``mqa_split_kv_b_proj`` mode. There ``kv_b_proj`` is not built at all and two
# standalone absorption parameters hold its elements instead: ``k_b_proj``
# folded from [heads, kv_lora, qk_nope] and ``v_b_proj`` folded from
# [heads, v_head, kv_lora]. The checkpoint still carries one
# ``kv_b_proj.weight`` key, transposed relative to the Fleet weight and
# head-major in its rows, so both directions are a single chain over that key:
# split into equal row blocks (granularity ``gcd(qk_nope, v_head)``), regroup
# per head, transpose the K half only.
#
# This test pins the block geometry, both directions' exact statements, the
# read-exactly-once property every temporary must have, the checkpoint-only
# naming of the absent ``kv_b_proj``, and that the two absorption params are
# skipped by the own-parameter walk while sibling sublayers are still recursed.
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import unittest

import paddle
from paddle.distributed.flex_checkpoint.aoa.generation import AOAContext

_PREFIX = "model.layers.0.self_attn."
_K = _PREFIX + "k_b_proj"
_V = _PREFIX + "v_b_proj"
_KV_CKPT = "hf.layers.0.self_attn.kv_b_proj.weight"


def _ctx():
    return AOAContext(
        config=None,
        checkpoint_name_prefix="hf",
        pp_to_single_mapping={},
        checkpoint_name_mapping={},
        model_name_prefix="model",
    )


def _make_split_mla(qk_nope=4, v_head=4, heads=2, kv_lora=3, extra_param=True):
    """Stand-in for a live ``MLASelfAttention`` in split mode.

    The generators read the geometry off exactly the attributes ``__init__``
    sizes the two parameters with, so the fake carries those same attributes
    and creates parameters of the same folded shapes. ``kv_b_proj`` is
    deliberately absent, which is what the real ``__init__`` does in this mode.
    """
    from paddlefleet.tensor_parallel.layers import Linear
    from paddlefleet.transformer.multi_latent_attention import MLASelfAttention

    class _LinearLeaf(paddle.nn.Layer):
        gen_aoa_statements = Linear.gen_aoa_statements
        gen_inv_aoa_statements = Linear.gen_inv_aoa_statements

        def __init__(self):
            super().__init__()
            self.weight = self.create_parameter(shape=[2, 2])
            self.bias = None

    class _SplitMLA(paddle.nn.Layer):
        gen_aoa_statements = MLASelfAttention.gen_aoa_statements
        gen_inv_aoa_statements = MLASelfAttention.gen_inv_aoa_statements
        _SPLIT_KV_B_LOCAL_NAMES = MLASelfAttention._SPLIT_KV_B_LOCAL_NAMES
        _split_kv_b_head_blocks = MLASelfAttention._split_kv_b_head_blocks
        _split_kv_b_names = MLASelfAttention._split_kv_b_names
        _gen_split_kv_b_aoa_statements = (
            MLASelfAttention._gen_split_kv_b_aoa_statements
        )
        _gen_inv_split_kv_b_aoa_statements = (
            MLASelfAttention._gen_inv_split_kv_b_aoa_statements
        )

        def __init__(self):
            super().__init__()
            self.mqa_latent_split_kv_b = True
            self.qk_nope_head_dim = qk_nope
            self.v_head_dim = v_head
            self.num_attention_heads_per_partition = heads
            self.k_b_proj = self.create_parameter(
                shape=[heads * kv_lora, qk_nope]
            )
            self.v_b_proj = self.create_parameter(
                shape=[heads * v_head, kv_lora]
            )
            if extra_param:
                # A plain own parameter, to prove the skip list is narrow.
                self.softmax_scale_param = self.create_parameter(shape=[1])
            self.o_proj = _LinearLeaf()

    return _SplitMLA()


def _parse(statement):
    """``(sources, targets)`` of one statement, transposes/attrs stripped."""
    lhs, rhs = statement.split(" -> ", 1)
    sources = [s.strip().removesuffix("^T") for s in lhs.split(",")]
    targets = [t.strip() for t in rhs.split(",") if t.strip() and "=" not in t]
    return sources, targets


def _temp_usage(statements, roots):
    """``(produced, consumed)`` counters for names under ``roots``."""
    produced = {}
    consumed = {}
    for statement in statements:
        sources, targets = _parse(statement)
        for name in sources:
            if any(name.startswith(root) for root in roots):
                consumed[name] = consumed.get(name, 0) + 1
        for name in targets:
            if any(name.startswith(root) for root in roots):
                produced[name] = produced.get(name, 0) + 1
    return produced, consumed


class TestSplitKVBGeometry(unittest.TestCase):
    """The row-block granularity is ``gcd(qk_nope, v_head)``, so a head spans
    ``n_k + n_v`` equal blocks."""

    def test_equal_head_dims_give_one_block_each(self):
        layer = _make_split_mla(qk_nope=4, v_head=4, heads=2)
        self.assertEqual(layer._split_kv_b_head_blocks(), (2, 1, 1))

    def test_unequal_head_dims_use_gcd(self):
        layer = _make_split_mla(qk_nope=4, v_head=2, heads=3)
        self.assertEqual(layer._split_kv_b_head_blocks(), (3, 2, 1))

    def test_coprime_head_dims_split_to_rows(self):
        layer = _make_split_mla(qk_nope=3, v_head=2, heads=1)
        self.assertEqual(layer._split_kv_b_head_blocks(), (1, 3, 2))


class TestSplitKVBForward(unittest.TestCase):
    """Checkpoint -> model chain."""

    def test_exact_statements(self):
        layer = _make_split_mla(qk_nope=4, v_head=4, heads=2)
        tmp = _K + "._kvb"
        self.assertEqual(
            layer._gen_split_kv_b_aoa_statements(_ctx(), _PREFIX, None),
            [
                f"{_KV_CKPT} -> "
                f"{tmp}_r0_0,{tmp}_r0_1,{tmp}_r1_0,{tmp}_r1_1, axis=0",
                f"{tmp}_r0_0 -> {tmp}_k0, axis=0",
                f"{tmp}_k0^T -> {tmp}_kt0",
                f"{tmp}_r0_1 -> {tmp}_v0, axis=0",
                f"{tmp}_r1_0 -> {tmp}_k1, axis=0",
                f"{tmp}_k1^T -> {tmp}_kt1",
                f"{tmp}_r1_1 -> {tmp}_v1, axis=0",
                f"{tmp}_kt0,{tmp}_kt1 -> {_K}, axis=0",
                f"{tmp}_v0,{tmp}_v1 -> {_V}, axis=0",
            ],
        )

    def test_unequal_dims_group_blocks_per_head(self):
        # qk_nope=4, v_head=2 -> chunk 2, so a head is 2 K blocks + 1 V block.
        layer = _make_split_mla(qk_nope=4, v_head=2, heads=1)
        tmp = _K + "._kvb"
        self.assertEqual(
            layer._gen_split_kv_b_aoa_statements(_ctx(), _PREFIX, None),
            [
                f"{_KV_CKPT} -> {tmp}_r0_0,{tmp}_r0_1,{tmp}_r0_2, axis=0",
                f"{tmp}_r0_0,{tmp}_r0_1 -> {tmp}_k0, axis=0",
                f"{tmp}_k0^T -> {tmp}_kt0",
                f"{tmp}_r0_2 -> {tmp}_v0, axis=0",
                f"{tmp}_kt0 -> {_K}, axis=0",
                f"{tmp}_v0 -> {_V}, axis=0",
            ],
        )

    def test_every_temporary_read_exactly_once(self):
        layer = _make_split_mla(qk_nope=6, v_head=4, heads=3)
        statements = layer._gen_split_kv_b_aoa_statements(_ctx(), _PREFIX, None)
        produced, consumed = _temp_usage(statements, (_K + "._kvb",))
        self.assertEqual(sorted(produced), sorted(consumed))
        self.assertEqual(set(produced.values()), {1})
        self.assertEqual(set(consumed.values()), {1})

    def test_absent_kv_b_proj_is_never_a_model_target(self):
        layer = _make_split_mla()
        statements = layer._gen_split_kv_b_aoa_statements(_ctx(), _PREFIX, None)
        self.assertEqual([s for s in statements if s.startswith("_ ->")], [])
        for statement in statements:
            _, targets = _parse(statement)
            for target in targets:
                self.assertNotIn("kv_b_proj", target)


class TestSplitKVBInverse(unittest.TestCase):
    """Model -> checkpoint chain."""

    def test_exact_statements(self):
        layer = _make_split_mla(qk_nope=4, v_head=4, heads=2)
        tmp = _K + "._inv_kvb"
        self.assertEqual(
            layer._gen_inv_split_kv_b_aoa_statements(_ctx(), _PREFIX, None),
            [
                f"{_K} -> {tmp}_k0,{tmp}_k1, axis=0",
                f"{_V}^T -> {tmp}_vt",
                f"{tmp}_vt -> {tmp}_v0,{tmp}_v1, axis=1",
                f"{tmp}_k0,{tmp}_v0 -> {tmp}_h0, axis=1",
                f"{tmp}_k1,{tmp}_v1 -> {tmp}_h1, axis=1",
                f"{tmp}_h0,{tmp}_h1 -> {tmp}_kv, axis=1",
                f"{tmp}_kv^T -> {_KV_CKPT}",
            ],
        )

    def test_head_major_column_order_is_k_then_v(self):
        layer = _make_split_mla(qk_nope=4, v_head=4, heads=2)
        tmp = _K + "._inv_kvb"
        statements = layer._gen_inv_split_kv_b_aoa_statements(
            _ctx(), _PREFIX, None
        )
        for h in range(2):
            self.assertIn(
                f"{tmp}_k{h},{tmp}_v{h} -> {tmp}_h{h}, axis=1", statements
            )

    def test_block_count_does_not_leak_into_the_inverse(self):
        # The inverse rebuilds whole heads, so unequal head dims change nothing
        # about its shape: one split per parameter, one concat per head.
        layer = _make_split_mla(qk_nope=6, v_head=2, heads=2)
        statements = layer._gen_inv_split_kv_b_aoa_statements(
            _ctx(), _PREFIX, None
        )
        self.assertEqual(len(statements), 7)

    def test_every_temporary_read_exactly_once(self):
        layer = _make_split_mla(qk_nope=6, v_head=4, heads=3)
        statements = layer._gen_inv_split_kv_b_aoa_statements(
            _ctx(), _PREFIX, None
        )
        produced, consumed = _temp_usage(statements, (_K + "._inv_kvb",))
        self.assertEqual(sorted(produced), sorted(consumed))
        self.assertEqual(set(produced.values()), {1})
        self.assertEqual(set(consumed.values()), {1})

    def test_absent_kv_b_proj_is_never_a_model_source(self):
        layer = _make_split_mla()
        statements = layer._gen_inv_split_kv_b_aoa_statements(
            _ctx(), _PREFIX, None
        )
        for statement in statements:
            sources, _ = _parse(statement)
            for source in sources:
                self.assertNotIn(".kv_b_proj", source)


class TestSplitKVBComposition(unittest.TestCase):
    """The two absorption params are handled by the chain and skipped by the
    generic own-parameter walk; sibling sublayers are still recursed."""

    def test_forward_skips_absorption_params_and_keeps_the_rest(self):
        layer = _make_split_mla()
        statements = layer.gen_aoa_statements(
            _ctx(), structured_name_prefix=_PREFIX
        )
        chain = layer._gen_split_kv_b_aoa_statements(_ctx(), _PREFIX, None)
        self.assertEqual(statements[: len(chain)], chain)
        # No identity statement competing with the chain for either parameter.
        self.assertEqual(
            [s for s in statements[len(chain) :] if "_b_proj" in s], []
        )
        # The sibling Linear child is still recursed (it needs its transpose).
        self.assertIn(
            f"hf.layers.0.self_attn.o_proj.weight^T -> {_PREFIX}o_proj.weight",
            statements,
        )

    def test_inverse_skips_absorption_params_and_keeps_the_rest(self):
        layer = _make_split_mla()
        statements = layer.gen_inv_aoa_statements(
            _ctx(), structured_name_prefix=_PREFIX
        )
        chain = layer._gen_inv_split_kv_b_aoa_statements(_ctx(), _PREFIX, None)
        self.assertEqual(statements[: len(chain)], chain)
        self.assertEqual(
            [s for s in statements[len(chain) :] if "_b_proj" in s], []
        )
        self.assertIn(
            f"{_PREFIX}o_proj.weight^T -> hf.layers.0.self_attn.o_proj.weight",
            statements,
        )


if __name__ == "__main__":
    unittest.main()
