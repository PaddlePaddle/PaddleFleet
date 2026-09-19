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
# per head, transpose the K half only. Left to the generic recursion this
# projection gets NO statements at all -- ``k_b_proj`` / ``v_b_proj`` resolve to
# identities that ``should_skip`` drops -- which loads uninitialised memory and
# saves two checkpoint keys no consumer understands.
#
# The non-split path is a bare ``super()`` delegation and is pinned structurally
# in ``test_ai_aoa_dsv4_mla_zero_override.py``; a fake that merely borrows the
# methods cannot exercise it, because zero-arg ``super()`` resolves against
# ``MLASelfAttention``.
#
# This test pins the block geometry, both directions' exact statements, the
# read-exactly-once property every temporary must have, the checkpoint-only
# naming of the absent ``kv_b_proj``, that the two absorption params are skipped
# by the own-parameter walk while other own params and sublayers are not, and
# the two loud failures (partial exclusion, inverse dtype cast).
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

from paddlefleet.models.gpt.aoa_generator import (
    FleetAOAContext,
)

_PREFIX = "model.layers.0.self_attn."
_K = _PREFIX + "k_b_proj"
_V = _PREFIX + "v_b_proj"
_KV_CKPT = "hf.layers.0.self_attn.kv_b_proj.weight"


def _ctx(
    checkpoint_name_mapping=None,
    dtype_cast_rules=None,
    excluded_names=(),
):
    return FleetAOAContext(
        config=None,
        checkpoint_name_prefix="hf",
        pp_to_single_mapping={},
        checkpoint_name_mapping=checkpoint_name_mapping or {},
        dtype_cast_rules=dtype_cast_rules or {},
        model_name_prefix="model",
        excluded_names=frozenset(excluded_names),
    )


def _make_split_mla(qk_nope=4, v_head=4, heads=2, kv_lora=3, extra_param=True):
    """Stand-in for a live ``MLASelfAttention`` in split mode.

    The generators read the geometry off exactly the attributes ``__init__``
    sizes the two parameters with, so the fake carries those same attributes
    and creates parameters of the same folded shapes; a marker that hard-coded
    the block counts would pin nothing. ``kv_b_proj`` is deliberately absent,
    which is what the real ``__init__`` does in this mode. Everything else a
    real MLA owns (rope, core attention, a full TransformerConfig) is
    irrelevant to plan generation.
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
        _split_kv_b_excluded = MLASelfAttention._split_kv_b_excluded
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
    # Attributes (``axis=``, ``src_dtype=`` ...) follow the target list.
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
        # ``get_var_mapping_chain_macro`` overwrites a variable's chain with a
        # plain assignment, so a temporary consumed twice silently drops the
        # first chain and leaves its destination uninitialised.
        layer = _make_split_mla(qk_nope=6, v_head=4, heads=3)
        statements = layer._gen_split_kv_b_aoa_statements(_ctx(), _PREFIX, None)
        produced, consumed = _temp_usage(statements, (_K + "._kvb",))
        self.assertEqual(sorted(produced), sorted(consumed))
        self.assertEqual(set(produced.values()), {1})
        self.assertEqual(set(consumed.values()), {1})

    def test_absent_kv_b_proj_is_never_a_model_target(self):
        # ``kv_b_proj`` is not built in this mode, so unlike the hand-written
        # generator (which predates that optimisation) nothing may be written
        # to it, and no dead-weight ``_ ->`` filler is emitted.
        layer = _make_split_mla()
        statements = layer._gen_split_kv_b_aoa_statements(_ctx(), _PREFIX, None)
        self.assertEqual([s for s in statements if s.startswith("_ ->")], [])
        for statement in statements:
            _, targets = _parse(statement)
            for target in targets:
                self.assertNotIn("kv_b_proj", target)

    def test_checkpoint_name_mapping_applies_to_the_fused_key(self):
        layer = _make_split_mla()
        ctx = _ctx(
            checkpoint_name_mapping={
                "layers.$LAYER_ID.self_attn.kv_b_proj.weight": (
                    "blocks.$LAYER_ID.attn.kv_b.w"
                )
            }
        )
        statements = layer._gen_split_kv_b_aoa_statements(ctx, _PREFIX, None)
        self.assertTrue(
            statements[0].startswith("hf.blocks.0.attn.kv_b.w -> "),
            statements[0],
        )


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
        # The checkpoint rows are, per head, the K rows followed by the V rows;
        # after the single closing transpose that is the per-head column order,
        # so the per-head concat must put K first.
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
        self.assertEqual([s for s in statements if s.endswith(" -> _")], [])
        for statement in statements:
            sources, _ = _parse(statement)
            for source in sources:
                self.assertNotIn(".kv_b_proj", source)


class TestSplitKVBComposition(unittest.TestCase):
    """The two absorption params are handled by the chain and skipped by the
    generic own-parameter walk; nothing else changes."""

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

    def test_renamed_absorption_params_still_get_no_identity(self):
        # With a mapping in play source != target, so ``should_skip`` would no
        # longer hide a stray identity: the explicit skip list is what keeps
        # the chain the single producer.
        layer = _make_split_mla()
        ctx = _ctx(
            checkpoint_name_mapping={
                "layers.$LAYER_ID.self_attn.k_b_proj": "renamed.k",
                "layers.$LAYER_ID.self_attn.v_b_proj": "renamed.v",
            }
        )
        statements = layer.gen_aoa_statements(
            ctx, structured_name_prefix=_PREFIX
        )
        self.assertEqual([s for s in statements if "renamed." in s], [])

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

    def test_excluding_both_params_drops_the_whole_chain(self):
        layer = _make_split_mla()
        ctx = _ctx(excluded_names=(_PREFIX + "k_b_proj", _PREFIX + "v_b_proj"))
        for statements in (
            layer.gen_aoa_statements(ctx, structured_name_prefix=_PREFIX),
            layer.gen_inv_aoa_statements(ctx, structured_name_prefix=_PREFIX),
        ):
            self.assertEqual([s for s in statements if "_b_proj" in s], [])
            self.assertEqual([s for s in statements if "kv_b_proj" in s], [])

    def test_partial_exclusion_fails_loudly(self):
        # Honoring half of one chain would leave the other half sourceless on
        # load / the checkpoint key half-built on save.
        layer = _make_split_mla()
        for excluded in ("k_b_proj", "v_b_proj"):
            ctx = _ctx(excluded_names=(_PREFIX + excluded,))
            with self.assertRaises(NotImplementedError):
                layer.gen_aoa_statements(ctx, structured_name_prefix=_PREFIX)
            with self.assertRaises(NotImplementedError):
                layer.gen_inv_aoa_statements(
                    ctx, structured_name_prefix=_PREFIX
                )


class TestSplitKVBDtypeCast(unittest.TestCase):
    """Neither direction carries a cast: both end in a concat, and a cast
    suffix only rides a one-to-one statement. Save rejects a matching rule;
    load ignores it."""

    _RULE = {"checkpoint_dtype": "float32", "model_dtype": "bfloat16"}
    _NOOP = {"checkpoint_dtype": "bfloat16", "model_dtype": "bfloat16"}

    def test_forward_ignores_a_cast_rule(self):
        layer = _make_split_mla(qk_nope=4, v_head=4, heads=2)
        ctx = _ctx(
            dtype_cast_rules={"layers.$LAYER_ID.self_attn.k_b_proj": self._RULE}
        )
        statements = layer._gen_split_kv_b_aoa_statements(ctx, _PREFIX, None)
        self.assertEqual(
            statements,
            layer._gen_split_kv_b_aoa_statements(_ctx(), _PREFIX, None),
        )
        self.assertEqual([s for s in statements if "dtype" in s], [])

    def test_inverse_rejects_a_cast(self):
        layer = _make_split_mla()
        for local in ("k_b_proj", "v_b_proj"):
            ctx = _ctx(
                dtype_cast_rules={
                    f"layers.$LAYER_ID.self_attn.{local}": self._RULE
                }
            )
            with self.assertRaises(NotImplementedError):
                layer._gen_inv_split_kv_b_aoa_statements(ctx, _PREFIX, None)

    def test_inverse_accepts_a_noop_cast_rule(self):
        # Equal dtypes format to an empty suffix, so there is nothing to drop.
        layer = _make_split_mla()
        ctx = _ctx(
            dtype_cast_rules={"layers.$LAYER_ID.self_attn.k_b_proj": self._NOOP}
        )
        self.assertEqual(
            len(layer._gen_inv_split_kv_b_aoa_statements(ctx, _PREFIX, None)),
            7,
        )


if __name__ == "__main__":
    unittest.main()
