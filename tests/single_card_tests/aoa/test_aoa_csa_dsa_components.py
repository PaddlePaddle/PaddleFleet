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
# Scope: CSA / DSA attention components. Their leaves are Linear (``^T`` handled
# by the Linear component), Norm (identity) and float32 direct params ``ape`` /
# ``attn_sink`` (identity), all covered by the base ``Layer`` recursion over the
# live module tree, so the layout needs no override. The two Indexer classes
# override forward only, and only to pick the ``indexer_init_from_scratch``
# branch (a checkpoint from an earlier training phase has no Indexer at all);
# the loading layout inside the branch is still the base recursion. The other
# model-specific twist is GLM5 DSA, whose checkpoint drops the
# ``core_attention`` path segment; that is expressed purely as a per-leaf
# ``aoa_checkpoint_name_mapping`` and resolved by the same base recursion.
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
from types import SimpleNamespace

import paddle
from paddle.distributed.flex_checkpoint.aoa.generation import (
    validate_checkpoint_name_mapping,
)

from paddlefleet.models.gpt.aoa_generator import FleetAOAContext
from paddlefleet.transformer.csa_attention import (
    CompressedSparseAttention,
    Compressor,
    CSAIndexer,
)
from paddlefleet.transformer.dsa_attention import DSAIndexer


class _Leaf(paddle.nn.Layer):
    """A single-``weight`` leaf (Linear/Norm stand-in for name resolution)."""

    def __init__(self, with_bias=False):
        super().__init__()
        self.weight = self.create_parameter(shape=[2])
        if with_bias:
            self.bias = self.create_parameter(shape=[2])


class _Node(paddle.nn.Layer):
    """Container registering direct params and child sub-layers by name.

    ``params`` model float32 direct params (CSA ``ape`` / ``attn_sink``); the
    keyword children model nested sub-layers. Both register through
    ``Layer.__setattr__`` exactly like a real module tree.
    """

    def __init__(self, params=(), **children):
        super().__init__()
        for pname in params:
            setattr(self, pname, self.create_parameter(shape=[2]))
        for name, child in children.items():
            setattr(self, name, child)


# GLM5 DSA canonical declaration: the checkpoint keeps the DSA
# indexer directly under ``self_attn`` (no ``core_attention`` segment), so five
# per-leaf model->checkpoint entries drop it. ``_match_template`` has no
# wildcard, hence one entry per leaf. Keys are model-root-relative, values are
# checkpoint-root-relative; ``$LAYER_ID`` matches a decimal segment.
GLM5_DSA_NAME_MAPPING = {
    "layers.$LAYER_ID.self_attn.core_attention.indexer.wq_b.weight": (
        "layers.$LAYER_ID.self_attn.indexer.wq_b.weight"
    ),
    "layers.$LAYER_ID.self_attn.core_attention.indexer.wk.weight": (
        "layers.$LAYER_ID.self_attn.indexer.wk.weight"
    ),
    "layers.$LAYER_ID.self_attn.core_attention.indexer.weights_proj.weight": (
        "layers.$LAYER_ID.self_attn.indexer.weights_proj.weight"
    ),
    "layers.$LAYER_ID.self_attn.core_attention.indexer.k_norm.weight": (
        "layers.$LAYER_ID.self_attn.indexer.k_norm.weight"
    ),
    "layers.$LAYER_ID.self_attn.core_attention.indexer.k_norm.bias": (
        "layers.$LAYER_ID.self_attn.indexer.k_norm.bias"
    ),
}


def _ctx(*, pp_to_single_mapping, checkpoint_name_mapping=None):
    """A directly-constructed frozen context (no live GPTModel needed)."""
    return FleetAOAContext(
        config=None,
        checkpoint_name_prefix="hf",
        pp_to_single_mapping=pp_to_single_mapping,
        checkpoint_name_mapping=checkpoint_name_mapping or {},
        dtype_cast_rules={},
        model_name_prefix="model",
        excluded_names=frozenset(),
    )


def _identity_pp_mapping(structured_names):
    """Structured live name -> single name, single == ``model.<structured>``."""
    return {name: f"model.{name}" for name in structured_names}


def _leaf_of(statement, side):
    """Returns the ``side`` (``0`` source / ``1`` target) endpoint of a stmt."""
    return statement.split(" -> ")[side]


class TestNoComponentOverride(unittest.TestCase):
    """No CSA/DSA class may re-implement the module walk.

    A stray override would silently diverge from the Linear ``^T`` / identity
    contract, bypassing the shared component layer. The two Indexers are the
    single exception, and forward only: they choose between loading and the
    ``_ ->`` add branch, and delegate the layout either way. Guarding on
    ``__dict__`` (not ``getattr``) pins the *class itself*, ignoring the
    inherited ``Layer`` method.
    """

    def test_only_the_indexers_override_and_forward_only(self):
        for cls in (Compressor, CompressedSparseAttention):
            self.assertNotIn(
                "gen_aoa_statements",
                cls.__dict__,
                f"{cls.__name__} unexpectedly overrides gen_aoa_statements",
            )
        for cls in (CSAIndexer, DSAIndexer):
            self.assertIn(
                "gen_aoa_statements",
                cls.__dict__,
                f"{cls.__name__} lost its init_from_scratch branch",
            )

    def test_no_class_overrides_the_inverse(self):
        # Saving is unconditional: whatever the model owns is written out.
        for cls in (
            Compressor,
            CSAIndexer,
            CompressedSparseAttention,
            DSAIndexer,
        ):
            self.assertNotIn(
                "gen_inv_aoa_statements",
                cls.__dict__,
                f"{cls.__name__} overrides gen_inv_aoa_statements",
            )


def _csa_core_attention_subtree():
    """A CompressedSparseAttention-shaped subtree (ERNIE-Lite / DSV4 style).

    ``attn_sink`` is a direct float32 param on the CSA module; ``compressor``
    holds a direct ``ape`` param plus a Linear (``linear_wkv``) and a Norm.
    """
    compressor = _Node(
        params=["ape"],
        linear_wkv=_Leaf(),
        norm=_Leaf(),
    )
    return _Node(params=["attn_sink"], compressor=compressor)


_CSA_PREFIX = "layers.0.self_attn.core_attention."
_CSA_STRUCTURED = [
    _CSA_PREFIX + "attn_sink",
    _CSA_PREFIX + "compressor.ape",
    _CSA_PREFIX + "compressor.linear_wkv.weight",
    _CSA_PREFIX + "compressor.norm.weight",
]


class TestCsaIdentityPath(unittest.TestCase):
    """ERNIE-Lite / DSV4 CSA: checkpoint relative path == model relative path.

    With an empty ``checkpoint_name_mapping`` the base recursion produces pure
    identity names (only the checkpoint prefix differs), so ``core_attention``
    survives on both sides. This is why these models declare nothing.
    """

    def setUp(self):
        self.model = _csa_core_attention_subtree()
        self.ctx = _ctx(
            pp_to_single_mapping=_identity_pp_mapping(_CSA_STRUCTURED)
        )

    def test_forward_is_identity_and_keeps_core_attention(self):
        stmts = self.model.gen_aoa_statements(
            self.ctx, structured_name_prefix=_CSA_PREFIX
        )
        self.assertEqual(len(stmts), len(_CSA_STRUCTURED))
        for s in stmts:
            src, dst = _leaf_of(s, 0), _leaf_of(s, 1)
            self.assertIn(".core_attention.", src)
            self.assertEqual(src, "hf." + dst[len("model.") :])

    def test_inverse_is_identity_and_keeps_core_attention(self):
        stmts = self.model.gen_inv_aoa_statements(
            self.ctx, structured_name_prefix=_CSA_PREFIX
        )
        self.assertEqual(len(stmts), len(_CSA_STRUCTURED))
        for s in stmts:
            src, dst = _leaf_of(s, 0), _leaf_of(s, 1)
            self.assertIn(".core_attention.", dst)
            self.assertEqual(dst, "hf." + src[len("model.") :])


def _dsa_indexer_subtree():
    """A DSAIndexer-shaped subtree: three Linears + a LayerNorm (weight+bias)."""
    return _Node(
        wq_b=_Leaf(),
        wk=_Leaf(),
        weights_proj=_Leaf(),
        k_norm=_Leaf(with_bias=True),
    )


_DSA_PREFIX = "layers.0.self_attn.core_attention.indexer."
_DSA_STRUCTURED = [
    _DSA_PREFIX + "wq_b.weight",
    _DSA_PREFIX + "wk.weight",
    _DSA_PREFIX + "weights_proj.weight",
    _DSA_PREFIX + "k_norm.weight",
    _DSA_PREFIX + "k_norm.bias",
]


class TestGlm5DsaCoreAttentionDropped(unittest.TestCase):
    """GLM5 DSA: the checkpoint drops ``core_attention`` on the indexer.

    The five-entry ``GLM5_DSA_NAME_MAPPING`` rewrites the checkpoint side only;
    the live single name (right of forward / left of inverse) keeps
    ``core_attention``. Both directions resolve the same pair through the base
    recursion -- no component override, no post-hoc text edit.
    """

    def setUp(self):
        self.model = _dsa_indexer_subtree()
        self.ctx = _ctx(
            pp_to_single_mapping=_identity_pp_mapping(_DSA_STRUCTURED),
            checkpoint_name_mapping=GLM5_DSA_NAME_MAPPING,
        )

    def test_frozen_mapping_is_wellformed(self):
        # The five-entry mapping is what a GLM5 config would carry verbatim;
        # it must pass the same validation the model __init__ runs.
        validate_checkpoint_name_mapping(GLM5_DSA_NAME_MAPPING)

    def test_forward_checkpoint_side_drops_core_attention(self):
        stmts = self.model.gen_aoa_statements(
            self.ctx, structured_name_prefix=_DSA_PREFIX
        )
        self.assertEqual(len(stmts), len(_DSA_STRUCTURED))
        for s in stmts:
            checkpoint, single = _leaf_of(s, 0), _leaf_of(s, 1)
            self.assertNotIn("core_attention", checkpoint)
            self.assertIn(".self_attn.indexer.", checkpoint)
            # The live single name is untouched: it still has core_attention.
            self.assertIn(".self_attn.core_attention.indexer.", single)
        self.assertIn(
            "hf.layers.0.self_attn.indexer.wq_b.weight -> "
            "model.layers.0.self_attn.core_attention.indexer.wq_b.weight",
            stmts,
        )
        self.assertIn(
            "hf.layers.0.self_attn.indexer.k_norm.bias -> "
            "model.layers.0.self_attn.core_attention.indexer.k_norm.bias",
            stmts,
        )

    def test_inverse_checkpoint_side_drops_core_attention(self):
        stmts = self.model.gen_inv_aoa_statements(
            self.ctx, structured_name_prefix=_DSA_PREFIX
        )
        self.assertEqual(len(stmts), len(_DSA_STRUCTURED))
        for s in stmts:
            single, checkpoint = _leaf_of(s, 0), _leaf_of(s, 1)
            self.assertNotIn("core_attention", checkpoint)
            self.assertIn(".self_attn.core_attention.indexer.", single)
        self.assertIn(
            "model.layers.0.self_attn.core_attention.indexer.wq_b.weight -> "
            "hf.layers.0.self_attn.indexer.wq_b.weight",
            stmts,
        )

    def test_forward_and_inverse_are_endpoint_swaps(self):
        fwd = self.model.gen_aoa_statements(
            self.ctx, structured_name_prefix=_DSA_PREFIX
        )
        inv = self.model.gen_inv_aoa_statements(
            self.ctx, structured_name_prefix=_DSA_PREFIX
        )
        reversed_fwd = [" -> ".join(reversed(s.split(" -> "))) for s in fwd]
        self.assertEqual(sorted(inv), sorted(reversed_fwd))


class _FakeDsaIndexer(DSAIndexer):
    """Real ``gen_aoa_statements``, real child names, no heavy ``__init__``.

    Subclassing keeps the zero-arg ``super()`` inside the override bound to the
    real MRO, so the loading branch really is the base recursion.
    """

    def __init__(self, init_from_scratch):
        paddle.nn.Layer.__init__(self)
        self.config = SimpleNamespace(
            indexer_init_from_scratch=init_from_scratch
        )
        self.wq_b = _Leaf()
        self.wk = _Leaf()
        self.weights_proj = _Leaf()
        self.k_norm = _Leaf(with_bias=True)


class _FakeCsaIndexer(CSAIndexer):
    """Same trick for CSA, whose subtree nests a Compressor."""

    def __init__(self, init_from_scratch):
        paddle.nn.Layer.__init__(self)
        self.config = SimpleNamespace(
            indexer_init_from_scratch=init_from_scratch
        )
        self.linear_wq_b = _Leaf()
        self.linear_weights_proj = _Leaf()
        self.compressor = _Node(
            params=["ape"],
            linear_wkv=_Leaf(),
            linear_wgate=_Leaf(),
            norm=_Leaf(),
        )


_CSA_INDEXER_STRUCTURED = [
    _DSA_PREFIX + "linear_wq_b.weight",
    _DSA_PREFIX + "linear_weights_proj.weight",
    _DSA_PREFIX + "compressor.ape",
    _DSA_PREFIX + "compressor.linear_wkv.weight",
    _DSA_PREFIX + "compressor.linear_wgate.weight",
    _DSA_PREFIX + "compressor.norm.weight",
]


class TestIndexerInitFromScratch(unittest.TestCase):
    """The phase-1 add branch: a checkpoint that has no Indexer at all.

    Growing the model between training phases means the Indexer subtree has no
    checkpoint source. AOA says that with ``_ -> <model name>``; omitting the
    statement instead falls back to identity naming and aborts. Both flavours
    read the same ``indexer_init_from_scratch`` switch.
    """

    def _cases(self):
        return (
            (_FakeDsaIndexer, _DSA_STRUCTURED),
            (_FakeCsaIndexer, _CSA_INDEXER_STRUCTURED),
        )

    def test_unset_switch_is_rejected(self):
        for cls, structured in self._cases():
            ctx = _ctx(pp_to_single_mapping=_identity_pp_mapping(structured))
            with self.assertRaises(ValueError) as cm:
                cls(None).gen_aoa_statements(
                    ctx, structured_name_prefix=_DSA_PREFIX
                )
            self.assertIn("indexer_init_from_scratch", str(cm.exception))

    def test_true_adds_the_whole_subtree_and_reads_nothing(self):
        for cls, structured in self._cases():
            ctx = _ctx(pp_to_single_mapping=_identity_pp_mapping(structured))
            stmts = cls(True).gen_aoa_statements(
                ctx, structured_name_prefix=_DSA_PREFIX
            )
            self.assertEqual(
                sorted(stmts),
                sorted(f"_ -> model.{name}" for name in structured),
            )

    def test_false_loads_through_the_base_recursion(self):
        for cls, structured in self._cases():
            ctx = _ctx(pp_to_single_mapping=_identity_pp_mapping(structured))
            stmts = cls(False).gen_aoa_statements(
                ctx, structured_name_prefix=_DSA_PREFIX
            )
            self.assertEqual(len(stmts), len(structured))
            for s in stmts:
                src, dst = _leaf_of(s, 0), _leaf_of(s, 1)
                self.assertNotEqual(src, "_")
                self.assertEqual(src, "hf." + dst[len("model.") :])

    def test_targets_are_the_same_keys_either_way(self):
        # The add branch must claim exactly the keys loading would have
        # assigned, or the engine is left with an unassigned tensor.
        for cls, structured in self._cases():
            ctx = _ctx(pp_to_single_mapping=_identity_pp_mapping(structured))
            adds = cls(True).gen_aoa_statements(
                ctx, structured_name_prefix=_DSA_PREFIX
            )
            loads = cls(False).gen_aoa_statements(
                ctx, structured_name_prefix=_DSA_PREFIX
            )
            self.assertEqual(
                sorted(_leaf_of(s, 1) for s in adds),
                sorted(_leaf_of(s, 1) for s in loads),
            )

    def test_saving_ignores_the_switch(self):
        for cls, structured in self._cases():
            ctx = _ctx(pp_to_single_mapping=_identity_pp_mapping(structured))
            saves = [
                cls(flag).gen_inv_aoa_statements(
                    ctx, structured_name_prefix=_DSA_PREFIX
                )
                for flag in (True, False)
            ]
            self.assertEqual(sorted(saves[0]), sorted(saves[1]))
            self.assertEqual(len(saves[0]), len(structured))
            for s in saves[0]:
                self.assertNotIn("_ ->", s)


if __name__ == "__main__":
    unittest.main()
