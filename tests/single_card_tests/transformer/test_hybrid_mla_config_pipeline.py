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

"""A1 CONFIG-PIPELINE validation for the hybrid-MLA feature.

Adversarial end-to-end check that every switch set in the *production*
``model_config.json`` / YAML actually reaches the built module, i.e. no switch
is silently dropped by a ``getattr(..., default)`` or an ``except TypeError``
fallback and then runs degraded without error.

The five production ``layer43`` configs under test (43 decoder layers + 1 MTP,
``-2`` at 7/14/21/28/35/42/43):

* ``ernielite_layer43_mla_hca``            -- phase 1 baseline,
                                              ``hybrid_mla_attention`` unset
                                              (defaults ``"mha"`` -> dense
                                              per-head attention on the ``-2``
                                              layers). Carries the sink.
* ``..._non_absorbed_mqa_dense``           -- ``"mqa_full_causal"``: latent MQA,
                                              no indexer (equivalence
                                              experiment).
* ``..._non_absorbed_mqa_hca_dsa``         -- phase 2, ``"mqa_dsa"``
                                              (+ YAML ``train_indexer_only``).
* ``..._non_absorbed_mqa_hca_dsa_sparse_loss`` -- phase 3/4, ``"mqa_dsa"`` with
                                              ``dsa_indexer_use_sparse_loss``.
* ``ernielite_layer43_mqa_hca``            -- CSA full-causal MQA
                                              (``csa_compress_ratios == -1``).
                                              NOT a hybrid-MLA config: a
                                              different attention class, no
                                              ``-2`` layer and no
                                              ``hybrid_mla_*`` key. Present here
                                              only as the negative control for
                                              "the switches must not leak".

The ``-2`` layers no longer carry hybrid-specific ``hybrid_index_*`` / mode /
sink fields. ``hybrid_mla_attention`` (enum ``"mha"`` / ``"mqa_dsa"`` /
``"mqa_full_causal"``) selects the ``-2`` layer path; the indexer reuses the
model-wide ``index_*`` (json) / ``dsa_index_*`` (provider) fields, and the sink
comes from the model-wide ``add_full_attention_sink_bias``.

Everything is driven off the real JSON/YAML on disk and rebuilt exactly the way
``ernie5/pretrain.py`` does (ErnieFleetModelConfig -> Ernie5V2Provider ->
apply_ernie_config_overrides -> LayerSpec dispatch -> build_spec_layer).
"""

import unittest
from functools import wraps

import paddle

from .hybrid_mla_utils import (
    _CONFIG_DIR,
    _CSA_MQA_CFG,
    _DSA_CFG as _DSA,
    _DSA_SPARSE_LOSS_CFG as _SPARSE_LOSS,
    _FULL_CAUSAL_CFG as _FULL_CAUSAL,
    _HYBRID_MLA_CFGS,
    _LAYER43_CFGS,
    _MHA_CFG as _MHA,
    _MINUS2_LAYERS,
    _MQA_DSA_CFGS,
    _PARENT_REPO_AVAILABLE,
    _build_real_attn,
    _flash_attn_version,
    _load_json,
    _load_provider,
    _load_yaml,
    _production_fa_version,
    _stub_device_capability,
    _try_use_cuda_device,
)

if not _try_use_cuda_device():
    _stub_device_capability()


def setUpModule():
    """Every test here is driven off the erniebot JSON/YAML on disk, so the
    whole module is meaningless in a standalone PaddleFleet checkout (which is
    what upstream CI builds). Skipping from here rather than at import time
    keeps the tests collected, so pytest still exits 0.
    """
    if not _PARENT_REPO_AVAILABLE:
        raise unittest.SkipTest(
            f"requires the erniebot parent repo configs at {_CONFIG_DIR}"
        )


def _requires_cuda(obj):
    reason = "requires a usable CUDA device to build the attention modules"

    def wrap(fn):
        @wraps(fn)
        def inner(*a, **k):
            if not _try_use_cuda_device():
                raise unittest.SkipTest(reason)
            return fn(*a, **k)

        return inner

    if isinstance(obj, type):
        for name, val in list(obj.__dict__.items()):
            if name.startswith("test") and callable(val):
                setattr(obj, name, wrap(val))
        return obj
    return wrap(obj)


# ---------------------------------------------------------------------------
# Production config-loading chain, mirroring ernie5/pretrain.py. The loader
# (``_load_provider``), the raw-JSON/YAML readers (``_load_json`` /
# ``_load_yaml``) and the real-module builder (``_build_real_attn``) are all
# shared via ``hybrid_mla_utils`` -- the sibling suite
# ``test_hybrid_mla_recompute_mtp_ckpt.py`` needs the same ones, and a private
# copy here is exactly how the config-name constants drifted before.
# ---------------------------------------------------------------------------
def _dispatch(provider, layer_idx):
    """Return (ratio, attn_cls, core_cls, indexer_cls) for one layer index.

    Uses the same LayerSpec dispatch the trainer builds from, without
    instantiating the (512-expert) model.
    """
    from paddlefleet.models.gpt.gpt_layer_specs import (
        _get_dsv4_hybrid_attention_layer_type,
        get_attention_spec,
    )
    from paddlefleet.transformer.enums import AttnMaskType

    n_hidden = provider.num_hidden_layers
    is_mtp = layer_idx >= n_hidden
    local_ln = (layer_idx - n_hidden) if is_mtp else layer_idx
    _, alt_type, ratio = _get_dsv4_hybrid_attention_layer_type(
        provider, local_ln, is_mtp
    )
    spec = get_attention_spec(
        config=provider,
        attention_layer_type=alt_type,
        attn_mask_type=AttnMaskType.causal,
    )
    attn_cls = spec.layer.__name__
    core = spec.sublayers_spec.core_attention
    core_cls = (
        getattr(core, "layer", core).__name__ if core is not None else None
    )
    indexer_cls = None
    if core is not None and hasattr(core, "sublayers_spec"):
        idx = getattr(core.sublayers_spec, "indexer", None)
        if idx is not None:
            indexer_cls = getattr(idx, "layer", idx).__name__
    return ratio, attn_cls, core_cls, indexer_cls


class TestLayerDispatchTable(unittest.TestCase):
    """Every one of the 44 layers dispatches to the intended classes.

    Consumers: ``gpt_layer_specs._get_dsv4_hybrid_attention_layer_type`` (ratio
    -> logical type) and ``get_attention_spec`` (spec build). Proves the mode
    switch in JSON reaches the spec.
    """

    def _assert_table(self, name, expected_core, expected_indexer):
        _, provider = _load_provider(name)
        n_total = provider.num_hidden_layers + (
            getattr(provider, "num_nextn_predict_layers", 0) or 0
        )
        self.assertEqual(n_total, 44)
        seen_minus2 = []
        for li in range(n_total):
            ratio, attn_cls, core_cls, indexer_cls = _dispatch(provider, li)
            if ratio == -2:
                seen_minus2.append(li)
                self.assertEqual(attn_cls, "MLASelfAttention", f"{name} L{li}")
                self.assertEqual(core_cls, expected_core, f"{name} L{li}")
                self.assertEqual(indexer_cls, expected_indexer, f"{name} L{li}")
            elif ratio == 128:
                # HCA layers are unaffected by the hybrid-MLA switches.
                self.assertEqual(
                    attn_cls, "DSv4HybridSelfAttention", f"{name} L{li}"
                )
                self.assertEqual(
                    core_cls, "CompressedSparseAttention", f"{name} L{li}"
                )
                self.assertEqual(indexer_cls, "CSAIndexer", f"{name} L{li}")
            else:
                self.fail(f"{name} L{li}: unexpected ratio {ratio}")
        self.assertEqual(seen_minus2, list(_MINUS2_LAYERS))

    def test_mha_baseline_uses_dot_product_no_indexer(self):
        self._assert_table(_MHA, "DotProductAttention", None)

    def test_mqa_uses_mqa_latent_with_dsa_indexer(self):
        """The *other* production ``"mqa_dsa"`` config dispatches identically.

        NAME/FIXTURE HISTORY: this used to run a config called
        ``ernielite_layer43_mla_mqa_hca`` -- an indexer-less latent-MQA variant
        that no longer exists on disk (it became ``..._non_absorbed_mqa_hca_dsa``,
        i.e. ``_DSA``). It was then repointed at ``ernielite_layer43_mqa_hca``,
        which is a **CSA full-causal MQA** config (``csa_compress_ratios == -1``,
        ``DSv4HybridSelfAttention`` + ``CompressedSparseAttention``, no
        ``hybrid_mla_*`` key at all) -- a different attention class, so
        ``_assert_table`` failed on "unexpected ratio -1". The negative control
        for that config now lives in
        ``test_csa_full_causal_mqa_config_never_builds_latent_mqa``.

        The property kept here is unchanged in strength: *every* production
        config that sets ``hybrid_mla_attention="mqa_dsa"`` must dispatch to
        ``MQALatentAttention`` + ``DSAIndexer`` on all seven ``-2`` layers. Since
        the enum rename the second such config is the phase-3/4 sparse-loss one,
        so this pins that ``dsa_indexer_use_sparse_loss`` does NOT leak into the
        class dispatch.
        """
        self._assert_table(_SPARSE_LOSS, "MQALatentAttention", "DSAIndexer")

    def test_csa_full_causal_mqa_config_never_builds_latent_mqa(self):
        """Negative control: the hybrid-MLA switches must not leak into a
        config that has no ``-2`` layer.

        ``ernielite_layer43_mqa_hca`` is CSA full-causal MQA: ``-1`` at exactly
        the indices where the hybrid-MLA configs put ``-2``, and no
        ``hybrid_mla_*`` / ``add_full_attention_sink_bias`` key. It must dispatch
        to the CSA family on every layer, own no ``MQALatentAttention`` and no
        ``DSAIndexer``, and report ``hybrid_mla_attention == "mha"``.
        """
        _, provider = _load_provider(_CSA_MQA_CFG)
        self.assertEqual(
            getattr(provider, "hybrid_mla_attention", "mha"), "mha"
        )
        n_total = provider.num_hidden_layers + (
            getattr(provider, "num_nextn_predict_layers", 0) or 0
        )
        self.assertEqual(n_total, 44)
        seen_minus1 = []
        for li in range(n_total):
            ratio, attn_cls, core_cls, indexer_cls = _dispatch(provider, li)
            self.assertEqual(
                attn_cls, "DSv4HybridSelfAttention", f"L{li} ratio={ratio}"
            )
            self.assertEqual(
                core_cls, "CompressedSparseAttention", f"L{li} ratio={ratio}"
            )
            self.assertEqual(indexer_cls, "CSAIndexer", f"L{li} ratio={ratio}")
            if ratio == -1:
                seen_minus1.append(li)
            else:
                self.assertEqual(ratio, 128, f"L{li}")
        self.assertEqual(seen_minus1, list(_MINUS2_LAYERS))

    def test_mqa_dsa_uses_mqa_latent_with_dsa_indexer(self):
        self._assert_table(_DSA, "MQALatentAttention", "DSAIndexer")

    def test_mqa_full_causal_drops_the_indexer(self):
        """``"mqa_full_causal"`` keeps the latent MQA core, removes the indexer.

        That is what makes the mode an MHA equivalent: the layer falls into
        ``MQALatentAttention``'s ``indexer is None`` branch
        (mqa_latent_attention.py:268) and attends to the full per-document causal
        set. Only the -2 layers may change; the HCA layers must keep their
        ``CSAIndexer``, which ``_assert_table`` checks on every layer.
        """
        _, provider = _load_provider(_DSA)
        self.assertEqual(provider.hybrid_mla_attention, "mqa_dsa")
        provider.hybrid_mla_attention = "mqa_full_causal"
        n_total = provider.num_hidden_layers + (
            getattr(provider, "num_nextn_predict_layers", 0) or 0
        )
        for li in _MINUS2_LAYERS:
            with self.subTest(layer=li):
                ratio, attn_cls, core_cls, indexer_cls = _dispatch(provider, li)
                self.assertEqual(ratio, -2)
                self.assertEqual(attn_cls, "MLASelfAttention")
                self.assertEqual(core_cls, "MQALatentAttention")
                self.assertIsNone(indexer_cls)
        for li in range(n_total):
            if li in _MINUS2_LAYERS:
                continue
            self.assertEqual(_dispatch(provider, li)[3], "CSAIndexer", f"L{li}")

    def test_full_causal_switch_reaches_provider_from_json(self):
        """The production ``"mqa_full_causal"`` config must dispatch indexer-less.

        The rejection of an out-of-enum value, and of a latent-MQA mode on a
        config with no ``-2`` layer, is asserted where a config is built from
        scratch (``test_mqa_latent_attention.py``
        ``test_illegal_hybrid_mla_attention_configs_are_rejected``); re-running
        ``__post_init__`` on an already-normalised provider trips unrelated
        validation.

        WAS: ``assertFalse(add_full_attention_sink_bias)`` -- "this config adds
        no parameter at all vs the baseline" used to hold because *neither* side
        had the sink. NO LONGER TRUE: the phase-1 baseline itself gained
        ``add_full_attention_sink_bias: true``, so this config had to gain it too
        or it would be *short* one parameter. NOW: assert the flag is on and
        equal to the baseline's -- the "zero new parameters" property is
        preserved, only its mechanism changed from "neither side has a sink" to
        "both sides do".
        """
        prov = _load_provider(_FULL_CAUSAL)[1]
        self.assertEqual(prov.hybrid_mla_attention, "mqa_full_causal")
        base = _load_provider(_MHA)[1]
        self.assertTrue(prov.add_full_attention_sink_bias)
        self.assertEqual(
            prov.add_full_attention_sink_bias,
            base.add_full_attention_sink_bias,
        )
        for li in _MINUS2_LAYERS:
            with self.subTest(layer=li):
                self.assertIsNone(_dispatch(prov, li)[3])

    def test_switches_reach_provider(self):
        """The JSON switches must survive onto the provider, not be defaulted.

        WAS: ``assertFalse(mha.add_full_attention_sink_bias)`` +
        ``assertTrue(...)`` on the MQA side, i.e. the sink was read as *part of*
        the mode switch. NO LONGER TRUE and deliberately so: the sink is a
        model-wide flag that every phase of the chain now sets, precisely so the
        ``core_attention.softmax_offset`` parameter exists on both sides and a
        phase-1 checkpoint stays loadable without a parameter-set change. NOW:
        the mode switch still has to differ per phase, and the sink flag has to
        be *identical* across the whole chain -- a strictly stronger statement
        than the old per-config constants, because it fails if any single phase
        drifts.
        """
        modes = {}
        sinks = {}
        for name in _HYBRID_MLA_CFGS:
            with self.subTest(config=name):
                prov = _load_provider(name)[1]
                modes[name] = getattr(prov, "hybrid_mla_attention", "mha")
                sinks[name] = getattr(
                    prov, "add_full_attention_sink_bias", False
                )
                self.assertTrue(
                    sinks[name],
                    f"{name} must keep the model-wide attention sink so the "
                    "phase chain stays checkpoint compatible",
                )
        self.assertEqual(
            set(sinks.values()), {True}, f"sink flag drifted: {sinks}"
        )
        self.assertEqual(modes[_MHA], "mha")
        self.assertEqual(modes[_FULL_CAUSAL], "mqa_full_causal")
        for name in _MQA_DSA_CFGS:
            self.assertEqual(modes[name], "mqa_dsa", name)


TestLayerDispatchTable = _requires_cuda(TestLayerDispatchTable)


class TestSinkParameterOnRealModules(unittest.TestCase):
    """Build the real ``-2`` layer and check the learnable sink.

    Consumer: ``build_softmax_offset`` (dot_product_attention.py:87), called by
    BOTH ``DotProductAttention.__init__`` (MHA phase) and
    ``MQALatentAttention.__init__`` (the latent MQA modes). Proves the
    model-wide ``add_full_attention_sink_bias`` JSON flag reaches a real bf16
    [num_heads] param at the SAME state_dict key
    (``core_attention.softmax_offset``) in both phases -- which is what keeps an
    MHA checkpoint loadable by an MQA run.
    """

    _SINK_KEY = "core_attention.softmax_offset"

    def _build(self, name):
        """Build the first ``-2`` layer under the production ``fa_version``.

        The dense-MHA sink path is guarded on ``FLAGS_flash_attn_version in
        (3, 4)`` (``multi_latent_attention.py:561-581``). Production gets 4 from
        ``TrainingArguments.__post_init__`` on these SM100 boxes; a bare pytest
        process never builds ``TrainingArguments`` and so keeps the image default
        2. Pin the production value rather than weaken the guard.
        """
        _, provider = _load_provider(name)
        with _flash_attn_version(_production_fa_version()):
            return _build_real_attn(provider, _MINUS2_LAYERS[0])

    def test_mha_baseline_has_no_sink(self):
        """NAME IS HISTORICAL -- the baseline now DOES carry the sink.

        WAS: ``assertIsNone(core_attention.softmax_offset)``, because phase 1 was
        the live reference run and deliberately did not perturb it with a new
        parameter. NO LONGER TRUE: the baseline ``model_config.json`` gained
        ``add_full_attention_sink_bias: true``, on purpose -- with the sink on
        *both* sides, ``core_attention.softmax_offset`` exists in phase 1's
        checkpoint and the later phases add **no** parameter, which is the whole
        point of the migration. Omitting it now would make the phase-1 checkpoint
        *short* one tensor.

        NOW: the baseline's dense ``DotProductAttention`` must own a real sink at
        exactly the same state_dict key the latent-MQA phases use, with the same
        shape/dtype -- i.e. this test moved from asserting an absence to
        asserting the checkpoint-compatibility property that absence used to
        stand for. (Rename suggested; kept as-is so no test method disappears.)
        """
        mod = self._build(_MHA)
        self.assertEqual(
            type(mod.core_attention).__name__, "DotProductAttention"
        )
        sink = mod.core_attention.softmax_offset
        self.assertIsNotNone(sink)
        self.assertEqual(list(sink.shape), [64])
        self.assertEqual(sink.dtype, paddle.bfloat16)
        self.assertFalse(sink.stop_gradient)
        self.assertEqual(
            [k for k in mod.state_dict() if k.endswith("softmax_offset")],
            [self._SINK_KEY],
        )

    def test_mqa_and_dsa_have_trainable_bf16_per_head_sink(self):
        for name in _MQA_DSA_CFGS:
            with self.subTest(config=name):
                mod = self._build(name)
                self.assertEqual(
                    type(mod.core_attention).__name__, "MQALatentAttention"
                )
                sink = mod.core_attention.softmax_offset
                self.assertIsNotNone(sink)
                self.assertEqual(list(sink.shape), [64])
                self.assertEqual(sink.dtype, paddle.bfloat16)
                self.assertFalse(sink.stop_gradient)
                # The shared ``build_softmax_offset`` seeds the sink with the
                # model ``init_method`` (NOT a hard zero) -- the same path the
                # dense DotProductAttention sink has always used -- so only
                # require a finite per-head logit here.
                self.assertTrue(
                    bool(paddle.isfinite(sink.astype("float32")).all())
                )
                self.assertEqual(
                    [
                        k
                        for k in mod.state_dict()
                        if k.endswith("softmax_offset")
                    ],
                    [self._SINK_KEY],
                )

    def test_sink_state_dict_key_identical_across_modes(self):
        """The sink key/shape/dtype must be identical in all four phases.

        WAS: compared only the two latent-MQA configs (and the baseline was
        expected to have no sink at all). NOW: the baseline is in the comparison
        too, which is the case that actually matters -- an MHA checkpoint is what
        the later phases resume from, so the dense ``DotProductAttention`` sink
        and the ``MQALatentAttention`` sink have to be interchangeable tensors at
        one key.
        """
        seen = {}
        for name in _HYBRID_MLA_CFGS:
            mod = self._build(name)
            keys = sorted(
                k for k in mod.state_dict() if k.endswith("softmax_offset")
            )
            sink = mod.core_attention.softmax_offset
            seen[name] = (tuple(keys), tuple(sink.shape), str(sink.dtype))
        self.assertEqual(
            len(set(seen.values())),
            1,
            f"sink parameter diverges across the phase chain: {seen}",
        )
        self.assertEqual(seen[_MHA][0], (self._SINK_KEY,))


TestSinkParameterOnRealModules = _requires_cuda(TestSinkParameterOnRealModules)


class TestHybridIndexerReadsModelWideIndexFields(unittest.TestCase):
    """The hybrid-MLA DSA indexer reads the model-wide ``dsa_index_*`` fields.

    The old hybrid-specific ``hybrid_index_*`` duplicates were removed; the
    ``-2`` layers' indexer now reuses the same ``index_*`` (json) /
    ``dsa_index_*`` (provider) fields the CSA layers already read. This proves
    those fields reach the built ``DSAIndexer`` (nothing is hard-coded) and its
    weights live under ``core_attention.indexer.*``.
    Consumer: ``dsa_attention.DSAIndexer`` with ``is_hybrid_mla_indexer=True``.
    """

    def test_dsa_indexer_reflects_model_wide_index_fields(self):
        _, provider = _load_provider(_DSA)
        # Production values (shared with the CSA layers). 2048 is what the
        # online configs train at, and it is exactly
        # ``mqa_latent_attention._LOSS_TOPK_CAP``, so at s=8192 the loss table
        # and the attention table end up the same width.
        self.assertEqual(provider.dsa_index_n_heads, 64)
        self.assertEqual(provider.dsa_index_head_dim, 128)
        self.assertEqual(provider.dsa_index_topk, 2048)
        # Discriminating probe: move the index dims off their production values
        # so a hard-coded/wrong-field read would be visible. Keep head_dim=128
        # (cuDNN kernel hard req) and topk a multiple of 128 (indexer backward
        # block size).
        provider.dsa_index_n_heads = 32
        provider.dsa_index_topk = 256
        mod = _build_real_attn(provider, _MINUS2_LAYERS[0])
        idx = mod.core_attention.indexer
        self.assertEqual(type(idx).__name__, "DSAIndexer")
        self.assertEqual(idx.n_heads, 32)
        self.assertEqual(idx.head_dim, 128)
        self.assertEqual(idx.index_topk, 256)
        # The indexer weights are exposed under core_attention.indexer.*.
        self.assertTrue(
            any(
                k.startswith("core_attention.indexer.")
                for k in mod.state_dict()
            ),
            "DSAIndexer weights must appear under core_attention.indexer.*",
        )


TestHybridIndexerReadsModelWideIndexFields = _requires_cuda(
    TestHybridIndexerReadsModelWideIndexFields
)


class TestAOAStatements(unittest.TestCase):
    """HF import/export (AOA) coverage. Runs without a GPU.

    NOTE: production trains/resumes with ``flex_checkpoint`` (save/load the
    module state_dict directly), so AOA only matters for HF weight conversion.
    """

    def _aoa(self, name, indexer_init_from_scratch=True):
        """``(cfg, fwd, inv)`` AOA statements for one production config.

        ``indexer_init_from_scratch`` is a YAML *provider* field
        (``config_check._KNOWN_YAML_PROVIDER_FIELDS``), not a JSON model-structure
        field, so it is absent from the ``ErnieFleetModelConfig`` built here and
        ``_resolve_init_from_scratch`` (modeling.py:857) hard-errors for any
        ``"mqa_dsa"`` config. That error is the intended product behaviour -- it
        refuses to guess which checkpoint the run starts from -- so the fixture
        supplies the value the way the trainer does instead of catching it. Both
        branches are exercised below.
        """
        from fleet_model.ernie5_v2.modeling import (
            _gen_aoa_config,
            _gen_inv_aoa_config,
        )
        from src.ernie_core_compat.configuration import ErnieFleetModelConfig

        cfg = ErnieFleetModelConfig.from_pretrained(
            str(_CONFIG_DIR / name), _configuration_file="model_config.json"
        )
        cfg.indexer_init_from_scratch = indexer_init_from_scratch
        fwd = _gen_aoa_config(cfg)["aoa_statements"]
        inv = _gen_inv_aoa_config(cfg)["aoa_statements"]
        return cfg, fwd, inv

    @staticmethod
    def _count(statements, needle):
        return sum(1 for s in statements if needle in s)

    def test_indexer_init_from_scratch_is_mandatory_and_switches_the_source(
        self,
    ):
        """The mandatory ``indexer_init_from_scratch`` gate, both branches.

        Unset must raise (it decides whether a phase-2 restart reloads the
        indexer it already trained or throws it away). ``True`` emits the AOA
        "add" primitive ``_ -> <key>`` for all five indexer tensors on each
        ``-2`` layer; ``False`` emits a real source mapping. The inverse (save)
        map is unconditional either way -- whatever the model owns is written out.
        """
        from fleet_model.ernie5_v2.modeling import _gen_aoa_config
        from src.ernie_core_compat.configuration import ErnieFleetModelConfig

        n_params, n_layers = 5, len(_MINUS2_LAYERS)
        for name in _MQA_DSA_CFGS:
            with self.subTest(config=name):
                cfg = ErnieFleetModelConfig.from_pretrained(
                    str(_CONFIG_DIR / name),
                    _configuration_file="model_config.json",
                )
                self.assertIsNone(
                    getattr(cfg, "indexer_init_from_scratch", None)
                )
                with self.assertRaises(ValueError):
                    _gen_aoa_config(cfg)

                _, fwd_scratch, inv_scratch = self._aoa(name, True)
                _, fwd_load, inv_load = self._aoa(name, False)
                needle = "core_attention.indexer."
                self.assertEqual(
                    self._count(
                        [s for s in fwd_scratch if s.startswith("_ -> ")],
                        needle,
                    ),
                    n_params * n_layers,
                )
                self.assertEqual(
                    self._count(
                        [s for s in fwd_load if s.startswith("_ -> ")], needle
                    ),
                    0,
                )
                self.assertEqual(
                    self._count(fwd_load, needle), n_params * n_layers
                )
                # Saving never depends on the switch.
                self.assertEqual(
                    self._count(inv_scratch, needle),
                    self._count(inv_load, needle),
                )
                self.assertEqual(
                    self._count(inv_load, needle), n_params * n_layers
                )

    def test_sink_aoa_reaches_every_hybrid_mla_layer(self):
        """Every hybrid-MLA phase must map the sink on every ``-2`` layer.

        WAS: the baseline was asserted to emit ZERO
        ``core_attention.softmax_offset`` statements, and only the latent-MQA
        configs to emit one per ``-2`` layer. NO LONGER TRUE: the baseline now
        sets ``add_full_attention_sink_bias`` too (see
        ``test_mha_baseline_has_no_sink``), so it legitimately emits seven.

        NOW: all four hybrid-MLA configs must emit at least one statement per
        ``-2`` layer in both directions, and the CSA full-causal MQA config --
        which owns ``core_attention.attn_sink``, a different tensor -- must emit
        none. That covers the old "the flag is not silently dropped" direction
        (the per-layer counts) and the old "no spurious statements" direction
        (the zero on the negative control) at once.
        """
        _, fwd_csa, inv_csa = self._aoa(_CSA_MQA_CFG)
        self.assertEqual(
            self._count(fwd_csa, "core_attention.softmax_offset"), 0
        )
        self.assertEqual(
            self._count(inv_csa, "core_attention.softmax_offset"), 0
        )
        for name in _HYBRID_MLA_CFGS:
            with self.subTest(config=name):
                _, fwd, inv = self._aoa(name)
                for li in _MINUS2_LAYERS:
                    needle = (
                        f"model.layers.{li}.self_attn.core_attention."
                        "softmax_offset"
                    )
                    self.assertGreaterEqual(
                        self._count(fwd, needle), 1, f"{name} fwd L{li}"
                    )
                    self.assertGreaterEqual(
                        self._count(inv, needle), 1, f"{name} inv L{li}"
                    )

    def test_sink_aoa_is_one_per_hybrid_mla_layer(self):
        # Regression guard for a bug introduced by moving the sink onto the
        # model-wide ``add_full_attention_sink_bias``: that flag also trips the
        # GENERAL softmax_offset block (modeling.py fwd / inv), which used not to
        # be gated on layer type. With no sliding_window, ``is_swa`` is False for
        # every layer, so it emitted a statement for ALL 44 layers -- including
        # the 37 CSA/HCA layers that own ``core_attention.attn_sink``, NOT
        # softmax_offset -- and the hybrid block then DUPLICATED it on the -2
        # layers. The general block is now gated on
        # ``not is_dsv4_hybrid or use_hybrid_mla`` and is the single emitter.
        # Now covers all four hybrid-MLA phases (the baseline included, since it
        # carries the sink as well).
        for name in _HYBRID_MLA_CFGS:
            with self.subTest(config=name):
                _, fwd, inv = self._aoa(name)
                self.assertEqual(
                    self._count(fwd, "core_attention.softmax_offset"),
                    len(_MINUS2_LAYERS),
                )
                self.assertEqual(
                    self._count(inv, "core_attention.softmax_offset"),
                    len(_MINUS2_LAYERS),
                )
                # And no statement lands on a CSA/HCA layer.
                for s in fwd + inv:
                    if "core_attention.softmax_offset" not in s:
                        continue
                    li = int(s.split("model.layers.")[1].split(".")[0])
                    self.assertIn(li, _MINUS2_LAYERS, s)

    def test_documented_bug_attention_gate_proj_dropped_from_aoa(self):
        # HIGH (pre-existing, symmetric, NOT specific to this feature): the
        # module builds ``self_attn.gate_proj`` because the provider sees
        # ``gated_attention=True`` (modeling.py ~:455 ORs use_gated_attn |
        # gated_attention), but the AOA statements at modeling.py:922 / :1383
        # gate ONLY on ``config.use_gated_attn`` -- which is False on the ERNIE
        # config (model_config.json sets ``gated_attention`` but not
        # ``use_gated_attn``). Result: on HF import/export the attention gate
        # weights are silently dropped, for ALL the layer43 configs including the
        # mha baseline. This asserts the CURRENT (buggy) behavior so a fix flips
        # it.
        for name in _LAYER43_CFGS:
            with self.subTest(config=name):
                cfg, fwd, inv = self._aoa(name)
                self.assertFalse(getattr(cfg, "use_gated_attn", False))
                self.assertTrue(getattr(cfg, "gated_attention", False))
                self.assertEqual(self._count(fwd, "self_attn.gate_proj"), 0)
                self.assertEqual(self._count(inv, "self_attn.gate_proj"), 0)

    @unittest.expectedFailure
    def test_attention_gate_proj_should_be_in_aoa(self):
        # Expected behavior: since the module owns ``self_attn.gate_proj`` on
        # every hybrid-MLA (-2) layer, HF conversion should carry it. Fails
        # today, documenting the gate_proj drop as a genuine bug.
        _, fwd, _ = self._aoa(_DSA)
        self.assertEqual(
            self._count(fwd, "self_attn.gate_proj"), len(_MINUS2_LAYERS)
        )


class TestConfigCheckCoverage(unittest.TestCase):
    """Every hybrid-MLA JSON field is registered in _MODEL_STRUCTURE_FIELDS.

    An unregistered *configured* field makes ``config_check`` hard-error at
    startup (``_check_no_unregistered_provider_fields``), so registration is
    what lets these configs run at all -- and it is also what a resume compares
    against.
    """

    _NEW_FIELDS = (
        "hybrid_mla_attention",
        "add_full_attention_sink_bias",
        "hybrid_mla_q_lora_rank",
        "hybrid_mla_kv_lora_rank",
        "hybrid_mla_qk_nope_head_dim",
        "hybrid_mla_qk_rope_head_dim",
        "hybrid_mla_v_head_dim",
        "hybrid_mla_num_attention_heads",
        "hybrid_mla_num_key_value_heads",
        "index_head_dim",
        "index_n_heads",
        "index_topk",
        "dsa_index_n_heads",
        "dsa_index_head_dim",
        "dsa_index_topk",
        "dsa_indexer_loss_coeff",
        "dsa_indexer_use_sparse_loss",
        "csa_compress_ratios",
        "csa_window_size",
        "csa_compress_rotary_base",
        "csa_dense_mode",
        "experimental_attention_variant",
        "use_fast_hadamard",
        "enable_hyper_connections",
        "qk_pos_emb_head_dim",
    )

    def test_new_fields_registered_as_structure_fields(self):
        from src.utils.config_check import _MODEL_STRUCTURE_FIELDS

        registered = set(_MODEL_STRUCTURE_FIELDS)
        missing = [f for f in self._NEW_FIELDS if f not in registered]
        self.assertEqual(
            missing, [], f"unregistered structure fields: {missing}"
        )

    def test_backend_switches_registered_as_yaml_provider_fields(self):
        from src.utils.config_check import _KNOWN_YAML_PROVIDER_FIELDS

        for f in (
            "csa_indexer_backend",
            "csa_sparse_attn_backend",
            "apply_rope_fusion",
            "context_parallel_size",
            "cp_balance_mode",
        ):
            with self.subTest(field=f):
                self.assertIn(f, _KNOWN_YAML_PROVIDER_FIELDS)


class TestIndexerLossNormalization(unittest.TestCase):
    """The indexer-loss denominator counts the right layers.

    Consumer: ``DSAIndexerLossLoggingHelper.track_indexer_metrics``
    (dsa_attention.py:1247-1257) fed from the pretraining trainer. When
    ``csa_compress_ratios`` is passed, the CSA layers (1 < ratio < 128) are
    counted, and if ``hybrid_mla_attention == "mqa_dsa"`` the ``-2`` (hybrid MLA)
    entries are added too, so the loss is normalised by indexer-layer count
    (``len(_MINUS2_LAYERS)`` == 7 for these configs), not the total layer count
    (44). This replicates that exact counting expression on the production
    ratios.
    """

    @staticmethod
    def _has_mqa_indexer(cfg):
        # Mirrors the trainer: only ``"mqa_dsa"`` builds a DSAIndexer on the -2
        # layers ("mqa_full_causal" has none, "mha" is not latent MQA at all).
        return cfg.get("hybrid_mla_attention", "mha") == "mqa_dsa"

    def _num_indexer_layers(self, ratios, hybrid_mla_has_indexer):
        n = sum(1 for r in ratios if 1 < r < 128)
        if hybrid_mla_has_indexer:
            n += sum(1 for r in ratios if r == -2)
        return n

    def test_counts_six_indexer_layers_when_mqa_dsa(self):
        """NAME IS HISTORICAL -- the count is now SEVEN, not six.

        WAS: hard-coded ``6``, which matched the old ``-2`` layout
        (8/17/26/34/42/43 -> 6 backbone layers). NO LONGER TRUE: the configs were
        realigned to the phase-1 online layout 7/14/21/28/35/42/43, so there are
        seven ``-2`` layers and the indexer-loss denominator is seven. Asserting
        ``len(_MINUS2_LAYERS)`` instead of a literal keeps the property (the
        denominator is the indexer-layer count, not the 44 total) and makes it
        track the layout constant rather than silently rot again.
        """
        for name in _MQA_DSA_CFGS:
            with self.subTest(config=name):
                cfg = _load_json(name)
                ratios = cfg["csa_compress_ratios"]
                self.assertEqual(len(ratios), 44)
                self.assertEqual(cfg.get("hybrid_mla_attention"), "mqa_dsa")
                self.assertEqual(
                    self._num_indexer_layers(
                        ratios, self._has_mqa_indexer(cfg)
                    ),
                    len(_MINUS2_LAYERS),
                )
                # Not the total, and not zero -- the two ways the counting
                # expression could collapse.
                self.assertNotEqual(len(_MINUS2_LAYERS), 44)

    def test_no_indexer_layers_for_mha_baseline(self):
        # mha leaves hybrid_mla_attention unset (-> "mha"): no CSA (1<ratio<128)
        # layers exist and 128 (HCA) is excluded, so the count is 0 -- the
        # counting expression must NOT collapse to the 44 total.
        cfg = _load_json(_MHA)
        ratios = cfg["csa_compress_ratios"]
        self.assertNotIn("hybrid_mla_attention", cfg)
        self.assertEqual(
            self._num_indexer_layers(ratios, self._has_mqa_indexer(cfg)),
            0,
        )
        # ``"mqa_full_causal"`` has no indexer either, so it must also count 0
        # even though it IS latent MQA.
        full_causal = _load_json(_FULL_CAUSAL)
        self.assertEqual(full_causal["hybrid_mla_attention"], "mqa_full_causal")
        self.assertEqual(
            self._num_indexer_layers(
                full_causal["csa_compress_ratios"],
                self._has_mqa_indexer(full_causal),
            ),
            0,
        )


class TestConfigDeltas(unittest.TestCase):
    """CONFIG-DRIFT SENTINELS: the five layer43 configs differ ONLY in the
    places listed below, key by key, against ``_MHA_CFG`` (phase 1).

    Both tests compare *parsed* keys (order- and comment-insensitive) against an
    explicit per-config allowlist. Anything outside the allowlist fails, and an
    allowlisted key whose value stops differing fails too -- so the allowlist
    cannot rot into a blanket exemption. That is the point of these two tests:
    they are the only thing standing between "phase 2 differs from phase 1 by one
    switch" and "someone quietly changed the optimizer in one of the four YAMLs".

    WAS (YAML side): a line-index text diff requiring equal line counts. That was
    unusable AND wrong: the four variants carry different multi-line Chinese
    comment headers, so the line counts differ and every subsequent line is
    misaligned -- while a real key difference hidden behind a reordering would be
    invisible. It also built the filename with
    ``replace("ernielite_layer43_mla", "ernielite_layer43_pretrain_mla")``, which
    only ever produced a path for the mla-named configs
    (``FileNotFoundError`` otherwise). NOW: ``yaml.safe_load`` + flattened keys +
    allowlist, with the filename built by ``_yaml_path``.
    """

    # The value a key takes when it is absent, so "present -> absent" is a
    # normal diff entry instead of a KeyError.
    _MISSING = "<MISSING>"

    # ------------------------------------------------------------------
    # Allowlist: what every non-baseline layer43 YAML is permitted to change
    # relative to ``ernielite_layer43_pretrain_mla_hca.yaml``. Values are
    # (baseline, variant) pairs and must match exactly.
    # ------------------------------------------------------------------
    # Shared by all four variants: they were branched from the baseline at a
    # point where the kernel/recompute policy differed. Not feature switches, but
    # deliberate and identical in all four, so they are pinned as a group -- if
    # one variant drifts out of the group this fails.
    _YAML_COMMON_DELTA = {
        # The MLA layers use plain RoPE; the fused path is the YaRN one.
        "apply_rope_fusion": (True, False),
        "dsv4_yarn_rope_fusion": (True, _MISSING),
        # full/uniform/1 recompute instead of the selective module list.
        "recompute_granularity": ("selective", "full"),
        "recompute_method": (_MISSING, "uniform"),
        "recompute_num_layers": (_MISSING, 1),
        "recompute_modules": (
            ["full_attn", "moe_gate_up", "moe_premute", "mhc_forward"],
            _MISSING,
        ),
    }

    def _yaml_allowlist(self, name):
        allowed = dict(self._YAML_COMMON_DELTA)
        allowed["model_name_or_path"] = (
            f"./model_config_separated/conf/fleet_align/{_MHA}",
            f"./model_config_separated/conf/fleet_align/{name}",
        )
        if name in _MQA_DSA_CFGS:
            # The DSA indexer on the -2 layers only has a cuDNN backward.
            allowed["csa_indexer_backend"] = ("tilelang", "cudnn")
            # ``indexer_init_from_scratch`` is mandatory once an indexer exists
            # (``modeling.py`` hard-errors on ``None``), so both mqa_dsa phases
            # set it -- with opposite values, which is the point: phase 2
            # resumes a phase-1 checkpoint that has no indexer tensors, phase
            # 3/4 resumes a phase-2 checkpoint that does.
            allowed["indexer_init_from_scratch"] = (
                self._MISSING,
                name == _DSA,
            )
            # Both mqa_dsa phases resume through the HF safetensors branch of
            # ``_load_flex_checkpoint`` rather than its DCP branch, because the
            # DCP branch drops every bf16 parameter from the model-state request
            # (``PaddleFormers trainer.py:1403``) and can only restore it from an
            # fp32 master weight -- which does not exist for a parameter that is
            # frozen (phase 2) or absent from the previous phase's optimizer
            # state (phase 3/4), and the assignment loop at ``:1440`` has no else
            # branch. ``load_from_hf`` requires ``ignore_load_lr_and_optim``
            # (hard assert at ``:1252``), and ``parallel_broadcast`` cannot serve
            # a resume whose parameter set changed (``resharder.py:459`` asserts
            # ``nranks > 1``, which fails at PP=1 / moe_sharding=1).
            # ``non_absorbed_mqa_dense`` deliberately keeps the DCP branch: it
            # adds no parameter and freezes nothing, so its master weights are
            # complete.
            allowed["load_from_hf"] = (self._MISSING, True)
            allowed["ignore_load_lr_and_optim"] = (False, True)
            allowed["flex_ckpt_comm_method"] = (
                "parallel_broadcast",
                "broadcast",
            )
        if name == _DSA:
            # Phase 2 = DSA warmup: only the indexer trains.
            allowed["train_indexer_only"] = (self._MISSING, True)
        return allowed

    @classmethod
    def _json_allowlist(cls, name):
        # ``index_topk`` is the value production actually trains at (2048, which
        # is also ``mqa_latent_attention._LOSS_TOPK_CAP``). The baseline
        # ``mla_hca`` config still carries the older 512, and it is deliberately
        # not edited, so every non-baseline config differs here. Note the field
        # is *not* MLA-only: the ratio-128 HCA indexer reads it too
        # (``csa_attention.py:1776``), so this difference is a real behavioural
        # gap between phase 1 and phases 2-4 on the HCA layers, not a cosmetic
        # one -- it is allowlisted so the drift check stays green, not because it
        # is harmless.
        topk = {"index_topk": (512, 2048)}
        if name == _FULL_CAUSAL:
            return {
                "hybrid_mla_attention": (cls._MISSING, "mqa_full_causal"),
                **topk,
            }
        if name == _DSA:
            return {
                "hybrid_mla_attention": (cls._MISSING, "mqa_dsa"),
                **topk,
            }
        if name == _SPARSE_LOSS:
            return {
                "hybrid_mla_attention": (cls._MISSING, "mqa_dsa"),
                # Phase 3/4: the indexer is trained enough for attention to
                # consume its ranking, so the narrow (sparse) KL is used.
                "dsa_indexer_use_sparse_loss": (False, True),
                **topk,
            }
        if name == _CSA_MQA_CFG:
            # A different attention class, not a phase of the hybrid-MLA chain:
            # -1 (CSA full-causal MQA) where the others put -2, and therefore
            # none of the latent-MLA geometry / sink / rope_type keys.
            return {
                "csa_compress_ratios": "SPECIAL",
                "add_full_attention_sink_bias": (True, cls._MISSING),
                "hybrid_mla_q_lora_rank": (1024, cls._MISSING),
                "hybrid_mla_kv_lora_rank": (512, cls._MISSING),
                "hybrid_mla_qk_nope_head_dim": (192, cls._MISSING),
                "hybrid_mla_qk_rope_head_dim": (64, cls._MISSING),
                "hybrid_mla_v_head_dim": (256, cls._MISSING),
                "hybrid_mla_num_attention_heads": (64, cls._MISSING),
                "hybrid_mla_num_key_value_heads": (64, cls._MISSING),
                "rope_type": ("rope", cls._MISSING),
                "use_vha_attention": (True, cls._MISSING),
                **topk,
            }
        raise AssertionError(f"no JSON allowlist for {name}")

    @classmethod
    def _flatten(cls, mapping, prefix=""):
        """Nested dict -> ``{"a.b": value}`` so nested YAML blocks are compared
        key-wise too (a whole sub-block being replaced would otherwise read as
        one opaque difference).
        """
        out = {}
        for key, value in mapping.items():
            path = f"{prefix}{key}"
            if isinstance(value, dict):
                out.update(cls._flatten(value, path + "."))
            else:
                out[path] = value
        return out

    def _delta(self, base, other):
        keys = set(base) | set(other)
        return {
            k: (base.get(k, self._MISSING), other.get(k, self._MISSING))
            for k in keys
            if base.get(k, self._MISSING) != other.get(k, self._MISSING)
        }

    def _assert_delta_matches(self, name, delta, allowed, what):
        unexpected = sorted(set(delta) - set(allowed))
        self.assertEqual(
            unexpected,
            [],
            f"{name} {what} drifted in keys outside the allowlist: "
            + ", ".join(
                f"{k}: {delta[k][0]!r} -> {delta[k][1]!r}" for k in unexpected
            ),
        )
        vanished = sorted(set(allowed) - set(delta))
        self.assertEqual(
            vanished,
            [],
            f"{name} {what}: allowlisted key(s) no longer differ from the "
            f"baseline: {vanished}. Tighten the allowlist instead of leaving a "
            "blanket exemption behind.",
        )
        for key in sorted(delta):
            if allowed[key] == "SPECIAL":
                continue
            with self.subTest(key=key):
                self.assertEqual(
                    delta[key], allowed[key], f"{name} {what} {key}"
                )

    def test_yamls_differ_only_in_model_name_or_path(self):
        """NAME IS HISTORICAL -- four more keys are allowlisted, see the class
        docstring and ``_YAML_COMMON_DELTA``.
        """
        base = self._flatten(_load_yaml(_MHA))
        self.assertGreater(len(base), 150, "baseline YAML did not parse")
        for name in _LAYER43_CFGS:
            if name == _MHA:
                continue
            with self.subTest(config=name):
                other = self._flatten(_load_yaml(name))
                self._assert_delta_matches(
                    name,
                    self._delta(base, other),
                    self._yaml_allowlist(name),
                    "YAML",
                )

    def test_json_deltas_are_only_the_feature_switches(self):
        """Each variant's ``model_config.json`` differs from phase 1 only in its
        own feature switch.

        WAS: a three-way ``mha``/``mqa``/``dsa`` comparison asserting
        ``delta(mha, mqa) == {"hybrid_mla_attention",
        "add_full_attention_sink_bias"}`` and ``delta(mqa, dsa) == {}``. NO
        LONGER TRUE on both counts: the sink moved into the baseline (so it is no
        longer a delta at all), and the ``mqa`` fixture pointed at a config of a
        different attention class. NOW: every one of the other four configs is
        diffed against the baseline against its own explicit allowlist, which is
        strictly more coverage (four configs instead of two comparisons) and
        pins values, not just key names.
        """
        base = _load_json(_MHA)
        base_ratios = base["csa_compress_ratios"]
        self.assertEqual(len(base_ratios), 44)
        self.assertEqual(
            [i for i, r in enumerate(base_ratios) if r < 0],
            list(_MINUS2_LAYERS),
        )
        self.assertEqual({r for r in base_ratios if r < 0}, {-2})
        for name in _LAYER43_CFGS:
            if name == _MHA:
                continue
            with self.subTest(config=name):
                other = _load_json(name)
                self._assert_delta_matches(
                    name,
                    self._delta(base, other),
                    self._json_allowlist(name),
                    "JSON",
                )

    def test_csa_full_causal_mqa_keeps_the_layer_layout(self):
        """The ``"SPECIAL"`` allowlist entry above, spelled out.

        ``ernielite_layer43_mqa_hca`` is allowed to differ from the baseline in
        ``csa_compress_ratios``, but ONLY by swapping the sentinel value: the
        negative indices must stay exactly where the hybrid-MLA configs put their
        ``-2``, and every other entry must be byte-identical. Otherwise the
        "differs only in the feature switch" claim would silently tolerate an
        arbitrary re-layout of the 44 layers.
        """
        base = _load_json(_MHA)["csa_compress_ratios"]
        other = _load_json(_CSA_MQA_CFG)["csa_compress_ratios"]
        self.assertEqual(len(other), len(base))
        self.assertEqual(
            [i for i, r in enumerate(other) if r < 0], list(_MINUS2_LAYERS)
        )
        self.assertEqual({r for r in other if r < 0}, {-1})
        self.assertEqual(
            [r for i, r in enumerate(base) if i not in _MINUS2_LAYERS],
            [r for i, r in enumerate(other) if i not in _MINUS2_LAYERS],
        )

    def test_baseline_intentionally_omits_the_switches(self):
        """NAME IS HISTORICAL -- the baseline omits only the *mode* switch.

        WAS: ``assertNotIn("add_full_attention_sink_bias", mha)`` as well, on the
        grounds that phase 1 was the untouched live reference. NO LONGER TRUE:
        the baseline sets the sink on purpose, so the parameter set is identical
        across the phase chain and a phase-1 checkpoint resumes into phase 2/3
        without adding a tensor.

        NOW: the baseline must (a) still leave ``hybrid_mla_attention`` unset, so
        it defaults to ``"mha"`` and the dense per-head path is what actually
        runs, and (b) carry the sink -- with the second half of the statement
        being that this is shared with, not distinguishing from, the other
        phases. The ``mqa_dsa``-only fields must stay out of it.
        """
        mha = _load_json(_MHA)
        self.assertNotIn("hybrid_mla_attention", mha)
        self.assertIs(mha["add_full_attention_sink_bias"], True)
        self.assertIs(mha["dsa_indexer_use_sparse_loss"], False)
        # Runtime-only switches never belong in the model structure JSON.
        for field in ("train_indexer_only", "indexer_init_from_scratch"):
            self.assertNotIn(field, mha)


if __name__ == "__main__":
    unittest.main()
