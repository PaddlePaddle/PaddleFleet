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

"""Muon optimizer routing for the ``hybrid_mla_attention`` parameter sets.

``optim: muon`` is the production optimizer for the layer43 runs, and Muon is a
*single* optimizer over a flat parameter list: every parameter is privately
tagged ``use_muon=True/False`` and the ones tagged False silently fall back to
the built-in AdamW branch. There are no ``param_groups`` to inspect, so a
mis-tagged parameter does not raise -- it just gets the wrong update rule for
the whole run. The three ``hybrid_mla_attention`` modes have different
parameter sets on the ``csa_compress_ratios == -2`` layers, so this suite pins
the tag of every one of them.

What is proven, all of it driven through the real production selector
``fleet_model/ernie5_v2/modeling.py::build_muon_param_info_map`` (the function
the ErnieBot trainer hands to ``paddle.optimizer.Muon``) over a module built
from the real production ``model_config.json`` + the real ``muon_configs``
block of the production YAML:

* the exact muon / AdamW split of all 16 parameters of a ``"mqa_dsa"`` ``-2``
  layer, i.e. the 5 new ``core_attention.indexer.*`` parameters and the
  ``add_full_attention_sink_bias`` sink land where they must: 2-D matrices in
  Muon, the 1-D ``k_norm`` weight/bias and the 1-D ``softmax_offset`` in AdamW;
* no parameter is dropped from *both* branches (the highest-severity failure
  mode: silently never optimised) and none is in both;
* ``"mha"`` and ``"mqa_full_causal"`` are byte-identical in Muon terms, and
  ``"mqa_dsa"`` is exactly that set plus the 5 indexer parameters -- nothing
  else about Muon varies with the enum;
* ``indexer.wq_b.weight`` is the only new parameter with a per-head slice
  function, with ``head_num == dsa_index_n_heads``, and that function preserves
  the shape both of the bare 2-D weight and of the 3-D batch Muon builds by
  stacking the 7 ``-2`` layers together;
* no registered slice key is dead (every key resolves to a real parameter);
* none of the new names collide with the production
  ``muon_exclude_patterns``, and the 1-D ones are excluded by the shape gate
  alone, so they stay out of Muon even if the pattern list changes;
* under ``train_indexer_only`` (phase 2) the trainable set is exactly the 5
  indexer parameters and both branches stay non-empty; an empty branch is a
  no-op rather than a crash, so a future all-1-D or all-2-D trainable set would
  not blow up either.
"""

import unittest

import paddle
from paddle.optimizer.muon import (
    Muon,
    MuonParamInfo,
    _default_should_use_muon,
)

from paddlefleet.transformer.csa_attention import CSAIndexer
from paddlefleet.transformer.dsa_attention import DSAIndexer
from paddlefleet.transformer.muon_utils import ortho_per_head

from .hybrid_mla_utils import (
    _CONFIG_DIR,
    _DSA_CFG,
    _DSA_SPARSE_LOSS_CFG,
    _FULL_CAUSAL_CFG,
    _MHA_CFG,
    _MINUS2_LAYERS,
    _PARENT_REPO_AVAILABLE,
    _add_repo_root_to_sys_path,
    _build_real_attn,
    _flash_attn_version,
    _load_provider,
    _load_yaml,
    _production_fa_version,
    _try_use_cuda_device,
)


def setUpModule():
    """Everything here is driven off the erniebot parent repo (its production
    ``model_config.json``/YAML *and* its Muon selector), and the modules only
    build on a real device, so skip from here rather than at import time -- a
    standalone PaddleFleet checkout still collects the tests and exits 0.
    """
    if not _PARENT_REPO_AVAILABLE:
        raise unittest.SkipTest(
            f"requires the erniebot parent repo configs at {_CONFIG_DIR}"
        )
    if not _try_use_cuda_device():
        raise unittest.SkipTest(
            "requires a usable CUDA device to build the attention modules"
        )


# The -2 layer under test and the ``_build_muon_slice_config`` key prefix for
# it (``modeling.py:689`` formats main layers as ``model.layers.{idx}``).
_LAYER = _MINUS2_LAYERS[0]
_PREFIX = f"model.layers.{_LAYER}.self_attn"


def _ernie5_v2_modeling():
    _add_repo_root_to_sys_path()
    import fleet_model.ernie5_v2.modeling as EM

    return EM


class _FakeModel:
    """The ``named_parameters()`` surface ``build_muon_param_info_map`` needs.

    Only one ``-2`` layer is built (a whole layer43 model does not fit on one
    card), so the layer-local names are re-prefixed to the full model path the
    slice-config keys use. ``_pp_to_single_mapping = {}`` is the identity
    mapping and keeps the selector off its ``_set_pipeline_name_mapping``
    fallback (``modeling.py:765-774``).
    """

    def __init__(self, attn, trainable_only=False):
        self._attn = attn
        self._trainable_only = trainable_only
        self._pp_to_single_mapping = {}

    def named_parameters(self):
        for rel, param in self._attn.named_parameters():
            if self._trainable_only and param.stop_gradient:
                continue
            yield f"{_PREFIX}.{rel}", param


def _freeze_to_indexer(attn):
    """``train_indexer_only``: trainable = the parameters owned by an indexer
    submodule, everything else frozen. Ownership is resolved by module tree,
    mirroring ``_collect_indexer_params``
    (``ernie5/src/trainers/pretraining_trainer.py:3633``); replicated rather
    than imported, as the sibling suites do, so PaddleFleet tests never need
    the parent trainer's dependency chain.
    """
    modules = []
    owned = set()
    for name, sub in attn.named_sublayers():
        if isinstance(sub, (CSAIndexer, DSAIndexer)):
            modules.append(name)
            owned.update(id(p) for p in sub.parameters())
    for param in attn.parameters():
        param.stop_gradient = id(param) not in owned
    return modules


class _Env:
    """Real provider + real ``-2`` attention layer + the real Muon metadata."""

    def __init__(self, cfg_name, freeze_to_indexer=False):
        self.cfg_name = cfg_name
        _, self.provider = _load_provider(cfg_name)
        # muon_configs reaches the provider from the trainer args, so the
        # provider built by _load_provider does not carry it; take the real
        # production block instead of inventing one.
        self.muon_configs = _load_yaml(cfg_name)["muon_configs"]
        self.provider.muon_configs = self.muon_configs
        with _flash_attn_version(_production_fa_version()):
            self.attn = _build_real_attn(self.provider, _LAYER)
        if freeze_to_indexer:
            self.indexer_modules = _freeze_to_indexer(self.attn)
        modeling = _ernie5_v2_modeling()
        self.modeling = modeling
        self.slice_config = modeling._build_muon_slice_config(
            None, self.provider
        )
        model = _FakeModel(self.attn, trainable_only=freeze_to_indexer)

        self.params = {
            n[len(_PREFIX) + 1 :]: p for n, p in model.named_parameters()
        }
        info_map = modeling.build_muon_param_info_map(model, self.provider)
        self.info = {rel: info_map[p.name] for rel, p in self.params.items()}

    @property
    def muon(self):
        return frozenset(k for k, v in self.info.items() if v.use_muon)

    @property
    def adamw(self):
        return frozenset(k for k, v in self.info.items() if not v.use_muon)

    def slice_fn(self, rel):
        """(fn name, kwargs) actually attached to a parameter, or None."""
        func = self.info[rel].split_concat_func
        return None if func is None else (func.func.__name__, func.keywords)


# The parameters a -2 layer owns in every mode. Latent MQA and dense MHA hold
# the same backbone tensors (the enum only swaps the core_attention class), and
# all four hybrid configs set add_full_attention_sink_bias, so the 1-D sink is
# in every mode too.
_BACKBONE_MUON = frozenset(
    {
        "q_a_proj.weight",
        "q_b_proj.weight",
        "kv_a_proj_with_mqa.weight",
        "kv_b_proj.weight",
        "o_proj.weight",
        "gate_proj.weight",
        "vha_postmix_U",
        "vha_postmix_V",
    }
)
_BACKBONE_ADAMW = frozenset(
    {
        "q_a_layernorm.weight",
        "kv_a_layernorm.weight",
        "core_attention.softmax_offset",
    }
)
# The five parameters "mqa_dsa" adds (DSAIndexer, dsa_attention.py:321-367).
_INDEXER_MUON = frozenset(
    {
        "core_attention.indexer.wq_b.weight",
        "core_attention.indexer.wk.weight",
        "core_attention.indexer.weights_proj.weight",
    }
)
_INDEXER_ADAMW = frozenset(
    {
        "core_attention.indexer.k_norm.weight",
        "core_attention.indexer.k_norm.bias",
    }
)
_INDEXER_ALL = _INDEXER_MUON | _INDEXER_ADAMW

_EXPECTED_EXCLUDE_PATTERNS = ["embed", "bias", "lm_head", "mlp.gate", "ape"]


class TestMqaDsaRouting(unittest.TestCase):
    """The phase-2 config: latent MQA + DSA indexer + sink."""

    @classmethod
    def setUpClass(cls):
        cls.env = _Env(_DSA_CFG)

    def test_exclude_patterns_are_the_production_ones(self):
        """Anti-vacuity: the whole suite is meaningless if the selector ran
        with an empty or drifted pattern list.
        """
        self.assertEqual(
            self.env.muon_configs["muon_exclude_patterns"],
            _EXPECTED_EXCLUDE_PATTERNS,
        )
        self.assertEqual(
            self.env.muon_configs["muon_qkv_update_mode"], "split_head"
        )

    def test_exact_split(self):
        self.assertEqual(self.env.muon, _BACKBONE_MUON | _INDEXER_MUON)
        self.assertEqual(self.env.adamw, _BACKBONE_ADAMW | _INDEXER_ADAMW)
        # 8 + 3 muon, 3 + 2 adamw: pin the counts too, so a parameter that
        # appears out of nowhere fails here and not somewhere downstream.
        self.assertEqual(len(self.env.muon), 11)
        self.assertEqual(len(self.env.adamw), 5)

    def test_new_parameter_shapes(self):
        """Every new parameter's shape, derived from the config rather than
        hardcoded, plus the ndim that decides the branch.
        """
        p = self.env.provider
        expected = {
            "core_attention.indexer.wq_b.weight": [
                p.hybrid_mla_q_lora_rank,
                p.dsa_index_n_heads * p.dsa_index_head_dim,
            ],
            "core_attention.indexer.wk.weight": [
                p.hidden_size,
                p.dsa_index_head_dim,
            ],
            "core_attention.indexer.k_norm.weight": [p.dsa_index_head_dim],
            "core_attention.indexer.k_norm.bias": [p.dsa_index_head_dim],
            "core_attention.indexer.weights_proj.weight": [
                p.hidden_size,
                p.dsa_index_n_heads,
            ],
            "core_attention.softmax_offset": [p.hybrid_mla_num_attention_heads],
        }
        for rel, shape in expected.items():
            with self.subTest(param=rel):
                self.assertEqual(list(self.env.params[rel].shape), shape)
                # ndim 2 -> Muon, ndim 1 -> AdamW, no exceptions.
                self.assertEqual(self.env.info[rel].use_muon, len(shape) == 2)

    def test_nothing_dropped_from_both_branches(self):
        """The severe failure mode: a parameter tagged into neither branch is
        never updated and nothing complains. ``build_muon_param_info_map``
        walks ``named_parameters()``, so the map must cover it exactly.
        """
        rels = set(self.env.params)
        self.assertEqual(self.env.muon | self.env.adamw, rels)
        self.assertEqual(self.env.muon & self.env.adamw, frozenset())
        self.assertEqual(len(rels), 16)
        # ...and specifically none of the new ones went missing.
        self.assertTrue(_INDEXER_ALL <= rels)
        self.assertIn("core_attention.softmax_offset", rels)

    def test_wq_b_is_the_only_sliced_new_parameter(self):
        """``modeling.py:680-684`` registers a per-head slice for
        ``indexer.wq_b.weight`` only; wk / weights_proj / k_norm get the
        whole-matrix treatment (or AdamW).
        """
        self.assertEqual(
            self.env.slice_fn("core_attention.indexer.wq_b.weight"),
            (
                "_muon_mla_per_head",
                {
                    "head_num": self.env.provider.dsa_index_n_heads,
                    "axis": -1,
                },
            ),
        )
        for rel in sorted(
            _INDEXER_ALL - {"core_attention.indexer.wq_b.weight"}
        ):
            with self.subTest(param=rel):
                self.assertIsNone(self.env.slice_fn(rel))
        self.assertIsNone(self.env.slice_fn("core_attention.softmax_offset"))

    def test_no_dead_slice_key(self):
        """A slice key that matches no parameter silently disables slicing.
        Every key registered for this layer must resolve.
        """
        keys = {
            k[len(_PREFIX) + 1 :]
            for k in self.env.slice_config
            if k.startswith(_PREFIX + ".")
        }
        self.assertEqual(keys - set(self.env.params), set())
        self.assertIn("core_attention.indexer.wq_b.weight", keys)

    def test_wq_b_slice_fn_survives_muon_batching(self):
        """Muon stacks same-shape 2-D parameters into one 3-D batch before
        orthogonalising (``paddle/optimizer/muon.py:621-627``), so every
        ``split_concat_func`` is called with both ranks. The 7 ``-2`` layers
        share this shape, hence a batch of 7.
        """
        rel = "core_attention.indexer.wq_b.weight"
        func = self.env.info[rel].split_concat_func
        heads = self.env.provider.dsa_index_n_heads
        shape = list(self.env.params[rel].shape)
        for batch in (None, len(_MINUS2_LAYERS)):
            with self.subTest(batch=batch):
                full = shape if batch is None else [batch, *shape]
                weight = paddle.randn(full)
                seen = []

                def ortho(x):
                    seen.append(tuple(x.shape))
                    return x

                out = func(weight, ortho)
                self.assertEqual(list(out.shape), full)
                self.assertEqual(len(seen), heads)
                expected_slice = (*full[:-1], full[-1] // heads)
                self.assertEqual(set(seen), {expected_slice})


class TestModeDifferences(unittest.TestCase):
    """Q: does anything about Muon change with the enum beyond "the indexer
    parameters exist or not"? Answer pinned here: no.
    """

    @classmethod
    def setUpClass(cls):
        cls.envs = {
            name: _Env(name)
            for name in (
                _MHA_CFG,
                _FULL_CAUSAL_CFG,
                _DSA_CFG,
                _DSA_SPARSE_LOSS_CFG,
            )
        }

    def test_mha_and_full_causal_are_identical(self):
        mha, full = self.envs[_MHA_CFG], self.envs[_FULL_CAUSAL_CFG]
        self.assertEqual(mha.muon, _BACKBONE_MUON)
        self.assertEqual(mha.adamw, _BACKBONE_ADAMW)
        self.assertEqual(full.muon, mha.muon)
        self.assertEqual(full.adamw, mha.adamw)
        self.assertEqual(
            {k: full.slice_fn(k) for k in full.info},
            {k: mha.slice_fn(k) for k in mha.info},
        )

    def test_mqa_dsa_adds_exactly_the_indexer(self):
        mha = self.envs[_MHA_CFG]
        for name in (_DSA_CFG, _DSA_SPARSE_LOSS_CFG):
            with self.subTest(config=name):
                env = self.envs[name]
                self.assertEqual(
                    set(env.params) - set(mha.params), _INDEXER_ALL
                )
                self.assertEqual(set(mha.params) - set(env.params), set())
                # The shared backbone keeps identical tags and slice fns.
                self.assertEqual(
                    {k: env.slice_fn(k) for k in mha.info},
                    {k: mha.slice_fn(k) for k in mha.info},
                )
                self.assertEqual(env.muon & set(mha.params), mha.muon)

    def test_sparse_loss_phase_matches_warmup_phase(self):
        """dsa_indexer_use_sparse_loss changes the attention/loss shape, never
        the optimizer routing.
        """
        warm, sparse = self.envs[_DSA_CFG], self.envs[_DSA_SPARSE_LOSS_CFG]
        self.assertEqual(sparse.muon, warm.muon)
        self.assertEqual(sparse.adamw, warm.adamw)
        self.assertEqual(
            {k: sparse.slice_fn(k) for k in sparse.info},
            {k: warm.slice_fn(k) for k in warm.info},
        )


class TestExcludePatterns(unittest.TestCase):
    """Name-pattern collisions, in both directions."""

    @classmethod
    def setUpClass(cls):
        cls.env = _Env(_DSA_CFG)

    def test_pattern_hits_are_exactly_as_expected(self):
        """``_default_should_use_muon`` matches patterns case-insensitively
        against a substring of the name (``muon.py:151-155``) and the selector
        applies it to the structured name *and* to paddle's internal
        ``param.name`` (``modeling.py:778-780``). Only ``k_norm.bias`` hits a
        pattern, via "bias"; nothing else collides and nothing that should be
        excluded is missed.
        """
        pats = _EXPECTED_EXCLUDE_PATTERNS
        for rel in sorted(_INDEXER_ALL | {"core_attention.softmax_offset"}):
            param = self.env.params[rel]
            structured = f"{_PREFIX}.{rel}"
            hit = [p for p in pats if p in structured.lower()] + [
                p for p in pats if p in param.name.lower()
            ]
            with self.subTest(param=rel):
                if rel.endswith("k_norm.bias"):
                    self.assertEqual(hit, ["bias"])
                else:
                    self.assertEqual(hit, [])

    def test_one_d_parameters_are_shape_gated_not_pattern_gated(self):
        """The shape gate runs first (``muon.py:148-149``), so the 1-D new
        parameters stay out of Muon even with an empty pattern list. This is
        what makes ``softmax_offset`` and ``k_norm.weight`` safe -- neither name
        matches any pattern.
        """
        for rel in (
            "core_attention.softmax_offset",
            "core_attention.indexer.k_norm.weight",
            "core_attention.indexer.k_norm.bias",
        ):
            param = self.env.params[rel]
            with self.subTest(param=rel):
                self.assertEqual(len(param.shape), 1)
                self.assertFalse(
                    _default_should_use_muon(
                        f"{_PREFIX}.{rel}", param.shape, []
                    )
                )
        # Anti-vacuity for the gate itself: a 2-D new parameter passes it.
        wk = self.env.params["core_attention.indexer.wk.weight"]
        self.assertTrue(
            _default_should_use_muon(
                f"{_PREFIX}.core_attention.indexer.wk.weight", wk.shape, []
            )
        )


class TestTrainIndexerOnlyRouting(unittest.TestCase):
    """Phase 2 freezes the backbone and trains only the indexer, and the
    optimizer then sees just the ``stop_gradient is False`` parameters
    (``pretraining_trainer.py:3697-3700``). The split must still cover exactly
    that set. Ownership is resolved by module tree, mirroring
    ``_collect_indexer_params`` (``pretraining_trainer.py:3633``) -- these tests
    must not import the parent trainer.
    """

    @classmethod
    def setUpClass(cls):
        cls.env = _Env(_DSA_CFG, freeze_to_indexer=True)

    def test_only_the_dsa_indexer_is_trainable(self):
        """The -2 layer owns exactly one indexer module, and freezing leaves
        exactly its 5 parameters trainable.
        """
        self.assertEqual(self.env.indexer_modules, ["core_attention.indexer"])
        self.assertEqual(set(self.env.params), _INDEXER_ALL)

    def test_split_covers_the_trainable_set_and_neither_side_is_empty(self):
        self.assertEqual(self.env.muon, _INDEXER_MUON)
        self.assertEqual(self.env.adamw, _INDEXER_ADAMW)
        self.assertEqual(self.env.muon | self.env.adamw, _INDEXER_ALL)
        self.assertEqual(self.env.muon & self.env.adamw, frozenset())
        self.assertEqual((len(self.env.muon), len(self.env.adamw)), (3, 2))

    def test_wq_b_keeps_its_per_head_slice_while_frozen(self):
        """The slice key is built from the config, not from the trainable set,
        so freezing the backbone must not lose the per-head split.
        """
        self.assertEqual(
            self.env.slice_fn("core_attention.indexer.wq_b.weight"),
            (
                "_muon_mla_per_head",
                {
                    "head_num": self.env.provider.dsa_index_n_heads,
                    "axis": -1,
                },
            ),
        )


class TestEmptyBranchIsANoOp(unittest.TestCase):
    """``train_indexer_only`` happens to keep both branches populated, but a
    future indexer with only 2-D (or only 1-D) trainable parameters would empty
    one of them. Muon's update loops (``muon.py:792-819``) are plain loops over
    possibly-empty collections; this pins that an empty branch neither raises
    nor stops the other branch from updating.
    """

    _EXCLUDE = _EXPECTED_EXCLUDE_PATTERNS

    def _step(self, shapes):
        params = list(
            paddle.nn.ParameterList(
                [
                    paddle.create_parameter(shape=s, dtype="float32")
                    for s in shapes
                ]
            ).parameters()
        )
        self.assertEqual(len(params), len(shapes))
        info = {
            p.name: MuonParamInfo(
                use_muon=_default_should_use_muon(
                    p.name, p.shape, self._EXCLUDE
                )
            )
            for p in params
        }
        opt = Muon(
            learning_rate=1e-3,
            parameters=params,
            muon_exclude_patterns=self._EXCLUDE,
            muon_param_info_map=info,
        )
        before = [p.numpy().copy() for p in params]
        sum((p * p).sum() for p in params).backward()
        opt.step()
        opt.clear_grad()
        n_muon = sum(1 for v in info.values() if v.use_muon)
        return n_muon, [
            bool((p.numpy() != b).any()) for p, b in zip(params, before)
        ]

    def test_empty_muon_branch(self):
        n_muon, changed = self._step([[8], [16]])
        self.assertEqual(n_muon, 0)
        self.assertEqual(changed, [True, True])

    def test_empty_adamw_branch(self):
        n_muon, changed = self._step([[8, 16], [16, 32]])
        self.assertEqual(n_muon, 2)
        self.assertEqual(changed, [True, True])

    def test_both_branches_populated(self):
        n_muon, changed = self._step([[8], [8, 16]])
        self.assertEqual(n_muon, 1)
        self.assertEqual(changed, [True, True])


class TestIndexerMuonSliceSpecSymmetry(unittest.TestCase):
    """``DSAIndexer`` and ``CSAIndexer`` must declare the same Muon treatment.

    The classes above cover the path ErnieBot production actually uses, where
    ``fleet_model/ernie5_v2/modeling.py`` supplies ``build_muon_param_info_map``
    and the per-module ``muon_slice_specs`` hook is never walked
    (``PaddleFormers/paddleformers/trainer/trainer.py:3540-3545`` short-circuits
    it). This class covers the *other* consumer, i.e. PaddleFleet used
    standalone: ``trainer.py:3427-3446`` walks the module tree and asks each
    module for its specs. ``CSAIndexer`` has always answered; ``DSAIndexer`` did
    not, so latent MQA + DSA silently lost per-head orthogonalisation on ``wq_b``
    there while the CSA indexer kept it. Kernel-free, hence not GPU gated.
    """

    _ON = {"muon_qkv_update_mode": "split_head"}
    _OFF = {"muon_qkv_update_mode": "whole_matrix"}

    class _StubDSA:
        n_heads = 64

    class _StubCSA:
        index_n_heads = 64

    def test_both_classes_expose_the_hook(self):
        for cls in (DSAIndexer, CSAIndexer):
            with self.subTest(cls=cls.__name__):
                self.assertTrue(
                    hasattr(cls, "muon_slice_specs"),
                    f"{cls.__name__} does not declare muon_slice_specs, so the "
                    "module-walk path would treat its q-up projection as one "
                    "matrix",
                )

    def test_dsa_indexer_slices_wq_b_per_head(self):
        spec = DSAIndexer.muon_slice_specs(self._StubDSA(), self._ON)
        self.assertEqual(list(spec), ["wq_b.weight"])
        fn, kwargs = spec["wq_b.weight"]
        self.assertIs(fn, ortho_per_head)
        self.assertEqual(kwargs, {"heads": self._StubDSA.n_heads})

    def test_same_shape_of_spec_as_csa_indexer(self):
        """Same helper, same kwarg name, same head count -- only the parameter
        name differs, because the sublayer attributes are named differently
        (``wq_b`` vs ``linear_wq_b``)."""
        dsa = DSAIndexer.muon_slice_specs(self._StubDSA(), self._ON)
        csa = CSAIndexer.muon_slice_specs(self._StubCSA(), self._ON)
        (dsa_key,), (csa_key,) = list(dsa), list(csa)
        self.assertTrue(dsa_key.endswith("wq_b.weight"), dsa_key)
        self.assertTrue(csa_key.endswith("wq_b.weight"), csa_key)
        self.assertEqual(dsa[dsa_key][0], csa[csa_key][0])
        self.assertEqual(dsa[dsa_key][1], csa[csa_key][1])

    def test_both_opt_out_when_mode_is_not_split_head(self):
        """Anti-vacuity for the guard: an unconditional ``return {...}`` would
        pass every assertion above."""
        self.assertEqual(
            DSAIndexer.muon_slice_specs(self._StubDSA(), self._OFF), {}
        )
        self.assertEqual(
            CSAIndexer.muon_slice_specs(self._StubCSA(), self._OFF), {}
        )

    def test_ortho_per_head_is_actually_per_head(self):
        """The spec is only useful if the helper orthogonalises each head's
        block separately. Feed a marker ``ortho_fn`` that returns the block
        index, and check the 64 output blocks are distinct and shape preserved.
        """
        heads, rows, cols = 4, 8, 16
        weight = paddle.zeros([rows, heads * cols], dtype="float32")
        seen = []

        def marker(block):
            seen.append(list(block.shape))
            return paddle.full_like(block, float(len(seen)))

        out = ortho_per_head(weight, marker, heads=heads)
        self.assertEqual(list(out.shape), [rows, heads * cols])
        self.assertEqual(len(seen), heads, f"blocks seen: {seen}")
        self.assertEqual(seen, [[rows, cols]] * heads)
        values = [float(out[0, i * cols].item()) for i in range(heads)]
        self.assertEqual(values, [1.0, 2.0, 3.0, 4.0])


if __name__ == "__main__":
    unittest.main()
