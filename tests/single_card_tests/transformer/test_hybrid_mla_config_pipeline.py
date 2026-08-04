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

The three production configs under test (43 decoder layers + 1 MTP):

* ``ernielite_layer43_mla_hca``      -- baseline, ``non_absorbed_mqa`` unset
                                        (defaults ``False`` -> dense MHA on the
                                        ``-2`` layers), no sink.
* ``ernielite_layer43_mla_mqa_hca``  -- ``non_absorbed_mqa=True`` + sink
                                        (``add_full_attention_sink_bias``).
* ``ernielite_layer43_mla_dsa_hca``  -- now *identical in meaning* to the mqa
                                        config: the DSA-less "mqa" mode is no
                                        longer reachable from config, so
                                        ``non_absorbed_mqa`` always builds the
                                        DSA indexer. It is kept as a separate
                                        JSON only for the live comparison runs.

The ``-2`` layers no longer carry hybrid-specific ``hybrid_index_*`` / mode /
sink fields. ``non_absorbed_mqa`` (bool) selects non-absorbed MQA + DSA indexer
vs dense MHA; the indexer reuses the model-wide ``index_*`` (json) /
``dsa_index_*`` (provider) fields, and the sink comes from the model-wide
``add_full_attention_sink_bias``.

Everything is driven off the real JSON/YAML on disk and rebuilt exactly the way
``ernie5/pretrain.py`` does (ErnieFleetModelConfig -> Ernie5V2Provider ->
apply_ernie_config_overrides -> LayerSpec dispatch -> build_spec_layer).
"""

import json
import unittest
from functools import wraps

import paddle

from .hybrid_mla_utils import (
    _CONFIG_DIR,
    _DENSE_CFG as _DENSE,
    _DSA_CFG as _DSA,
    _MHA_CFG as _MHA,
    _MINUS2_LAYERS,
    _MQA_CFG as _MQA,
    _PARENT_REPO_AVAILABLE,
    _REPO_ROOT,
    _build_real_attn,
    _load_provider,
    _stub_device_capability,
    _try_use_cuda_device,
)

_YAML_DIR = _REPO_ROOT / "conf" / "online"

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
# (``_load_provider``) and the real-module builder (``_build_real_attn``) are
# shared via ``hybrid_mla_utils``.
# ---------------------------------------------------------------------------
def _load_json(name):
    with open(_CONFIG_DIR / name / "model_config.json") as f:
        return json.load(f)


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
        # ``non_absorbed_mqa`` builds the DSA indexer unless
        # ``non_absorbed_mqa_dense`` drops it (see the test below), so mqa
        # dispatches exactly like dsa: MQALatentAttention core + DSAIndexer.
        self._assert_table(_MQA, "MQALatentAttention", "DSAIndexer")

    def test_mqa_dsa_uses_mqa_latent_with_dsa_indexer(self):
        self._assert_table(_DSA, "MQALatentAttention", "DSAIndexer")

    def test_non_absorbed_mqa_dense_drops_the_indexer(self):
        """``non_absorbed_mqa_dense`` keeps the MQA core but removes the indexer.

        That is what makes the mode a dense-MHA equivalent: the layer falls into
        ``MQALatentAttention``'s ``indexer is None`` branch
        (mqa_latent_attention.py:268) and attends to the full per-document causal
        set. Only the -2 layers may change; the HCA layers must keep their
        ``CSAIndexer``, which ``_assert_table`` checks on every layer.
        """
        _, provider = _load_provider(_DSA)
        self.assertTrue(provider.non_absorbed_mqa)
        provider.non_absorbed_mqa_dense = True
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

    def test_dense_switch_reaches_provider_from_json(self):
        """The production ``non_absorbed_mqa_dense`` config must dispatch dense.

        The rejection of the switch *alone* is asserted where a config is built
        from scratch (``test_mqa_latent_attention.py``
        ``test_dense_switch_alone_is_rejected``); re-running ``__post_init__`` on
        an already-normalised provider trips unrelated validation.
        """
        prov = _load_provider(_DENSE)[1]
        self.assertTrue(prov.non_absorbed_mqa)
        self.assertTrue(prov.non_absorbed_mqa_dense)
        # No sink either, so this config adds no parameter at all vs _MHA.
        self.assertFalse(getattr(prov, "add_full_attention_sink_bias", False))
        for li in _MINUS2_LAYERS:
            with self.subTest(layer=li):
                self.assertIsNone(_dispatch(prov, li)[3])

    def test_switches_reach_provider(self):
        # The JSON switches must survive onto the provider (not silently
        # defaulted). mha leaves both unset -> False; mqa/dsa set both True.
        mha = _load_provider(_MHA)[1]
        self.assertFalse(getattr(mha, "non_absorbed_mqa", False))
        self.assertFalse(getattr(mha, "add_full_attention_sink_bias", False))
        for name in (_MQA, _DSA):
            with self.subTest(config=name):
                prov = _load_provider(name)[1]
                self.assertTrue(prov.non_absorbed_mqa)
                self.assertTrue(prov.add_full_attention_sink_bias)


TestLayerDispatchTable = _requires_cuda(TestLayerDispatchTable)


class TestSinkParameterOnRealModules(unittest.TestCase):
    """Build the real ``-2`` layer and check the learnable sink.

    Consumer: ``build_softmax_offset`` (dot_product_attention.py:87), called by
    BOTH ``DotProductAttention.__init__`` (MHA phase) and
    ``MQALatentAttention.__init__`` (non_absorbed_mqa phase). Proves the
    model-wide ``add_full_attention_sink_bias`` JSON flag reaches a real bf16
    [num_heads] param at the SAME state_dict key
    (``core_attention.softmax_offset``) in both phases -- which is what keeps an
    MHA checkpoint loadable by an MQA run.
    """

    def test_mha_baseline_has_no_sink(self):
        # The baseline config intentionally omits the flag to avoid perturbing
        # the live reference run.
        _, provider = _load_provider(_MHA)
        mod = _build_real_attn(provider, _MINUS2_LAYERS[0])
        self.assertIsNone(mod.core_attention.softmax_offset)
        self.assertEqual(
            [k for k in mod.state_dict() if k.endswith("softmax_offset")], []
        )

    def test_mqa_and_dsa_have_trainable_bf16_per_head_sink(self):
        for name in (_MQA, _DSA):
            with self.subTest(config=name):
                _, provider = _load_provider(name)
                mod = _build_real_attn(provider, _MINUS2_LAYERS[0])
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
                    ["core_attention.softmax_offset"],
                )

    def test_sink_state_dict_key_identical_across_modes(self):
        keys = []
        for name in (_MQA, _DSA):
            _, provider = _load_provider(name)
            mod = _build_real_attn(provider, _MINUS2_LAYERS[0])
            keys.append(
                sorted(
                    k for k in mod.state_dict() if k.endswith("softmax_offset")
                )
            )
        self.assertEqual(keys[0], keys[1])


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
        # Production values (shared with the CSA layers).
        self.assertEqual(provider.dsa_index_n_heads, 64)
        self.assertEqual(provider.dsa_index_head_dim, 128)
        self.assertEqual(provider.dsa_index_topk, 512)
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

    def _aoa(self, name):
        from fleet_model.ernie5_v2.modeling import (
            _gen_aoa_config,
            _gen_inv_aoa_config,
        )
        from src.ernie_core_compat.configuration import ErnieFleetModelConfig

        cfg = ErnieFleetModelConfig.from_pretrained(
            str(_CONFIG_DIR / name), _configuration_file="model_config.json"
        )
        fwd = _gen_aoa_config(cfg)["aoa_statements"]
        inv = _gen_inv_aoa_config(cfg)["aoa_statements"]
        return cfg, fwd, inv

    @staticmethod
    def _count(statements, needle):
        return sum(1 for s in statements if needle in s)

    def test_sink_aoa_reaches_every_hybrid_mla_layer(self):
        # The sink flag must not be silently dropped: mha (no flag) emits zero
        # ``core_attention.softmax_offset`` statements, while mqa/dsa emit one
        # for each of the six -2 layers (both forward and inverse maps).
        _, fwd_mha, inv_mha = self._aoa(_MHA)
        self.assertEqual(
            self._count(fwd_mha, "core_attention.softmax_offset"), 0
        )
        self.assertEqual(
            self._count(inv_mha, "core_attention.softmax_offset"), 0
        )
        for name in (_MQA, _DSA):
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
        # the 38 CSA/HCA layers that own ``core_attention.attn_sink``, NOT
        # softmax_offset -- and the hybrid block then DUPLICATED it on the 6 -2
        # layers, giving 44 + 6 = 50 instead of 6. The general block is now gated
        # on ``not is_dsv4_hybrid or use_hybrid_mla`` and is the single emitter.
        for name in (_MQA, _DSA):
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
        # weights are silently dropped, for ALL three configs including the mha
        # baseline. This asserts the CURRENT (buggy) behavior so a fix flips it.
        for name in (_MHA, _MQA, _DSA):
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
        _, fwd, _ = self._aoa(_MQA)
        self.assertEqual(self._count(fwd, "self_attn.gate_proj"), 6)


class TestConfigCheckCoverage(unittest.TestCase):
    """Every hybrid-MLA JSON field is registered in _MODEL_STRUCTURE_FIELDS.

    An unregistered *configured* field makes ``config_check`` hard-error at
    startup (``_check_no_unregistered_provider_fields``), so registration is
    what lets these configs run at all -- and it is also what a resume compares
    against.
    """

    _NEW_FIELDS = (
        "non_absorbed_mqa",
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
    counted, and if ``non_absorbed_mqa`` is set the ``-2`` (hybrid MLA) entries
    are added too, so the loss is normalised by indexer-layer count (6), not the
    total layer count (44). This replicates that exact counting expression on
    the production ratios.
    """

    def _num_indexer_layers(self, ratios, non_absorbed_mqa):
        n = sum(1 for r in ratios if 1 < r < 128)
        if non_absorbed_mqa:
            n += sum(1 for r in ratios if r == -2)
        return n

    def test_counts_six_indexer_layers_when_non_absorbed_mqa(self):
        # Both mqa and dsa set non_absorbed_mqa=True now, so both count the six
        # -2 layers (no CSA 1<ratio<128 layers exist here).
        for name in (_MQA, _DSA):
            with self.subTest(config=name):
                cfg = _load_json(name)
                ratios = cfg["csa_compress_ratios"]
                self.assertEqual(len(ratios), 44)
                self.assertIs(cfg.get("non_absorbed_mqa"), True)
                self.assertEqual(
                    self._num_indexer_layers(
                        ratios, cfg.get("non_absorbed_mqa", False)
                    ),
                    6,
                )

    def test_no_indexer_layers_for_mha_baseline(self):
        # mha leaves non_absorbed_mqa unset (-> False): no CSA (1<ratio<128)
        # layers exist and 128 (HCA) is excluded, so the count is 0 -- the
        # counting expression must NOT collapse to the 44 total.
        cfg = _load_json(_MHA)
        ratios = cfg["csa_compress_ratios"]
        self.assertNotIn("non_absorbed_mqa", cfg)
        self.assertEqual(
            self._num_indexer_layers(
                ratios, cfg.get("non_absorbed_mqa", False)
            ),
            0,
        )


class TestConfigDeltas(unittest.TestCase):
    """The three variants differ only in the intended places."""

    def _yaml_path(self, name):
        return (
            _YAML_DIR
            / f"{name.replace('ernielite_layer43_mla', 'ernielite_layer43_pretrain_mla')}.yaml"
        )

    def test_yamls_differ_only_in_model_name_or_path(self):
        # All model-structure differences live in the JSON; the training YAMLs
        # must be otherwise identical so the three runs are truly comparable.
        texts = {}
        for name in (_MHA, _MQA, _DSA):
            with open(self._yaml_path(name)) as f:
                texts[name] = f.read().splitlines()
        base = texts[_MHA]
        for name in (_MQA, _DSA):
            other = texts[name]
            self.assertEqual(len(base), len(other), name)
            diff = [i for i in range(len(base)) if base[i] != other[i]]
            differing = [base[i].split(":", 1)[0].strip() for i in diff]
            self.assertEqual(
                differing,
                ["model_name_or_path"],
                f"{name} YAML diverges beyond model_name_or_path: lines {diff}",
            )

    def test_json_deltas_are_only_the_feature_switches(self):
        mha, mqa, dsa = (_load_json(n) for n in (_MHA, _MQA, _DSA))

        def delta(a, b):
            keys = set(a) | set(b)
            return {
                k: (a.get(k), b.get(k)) for k in keys if a.get(k) != b.get(k)
            }

        # mha -> mqa turns on the two model-wide switches (the old
        # hybrid_mla_attn_mode / hybrid_mla_attn_sink pair is gone).
        self.assertEqual(
            set(delta(mha, mqa)),
            {"non_absorbed_mqa", "add_full_attention_sink_bias"},
        )
        self.assertIs(mqa["non_absorbed_mqa"], True)
        self.assertIs(mqa["add_full_attention_sink_bias"], True)

        # mqa and dsa are now IDENTICAL in meaning: the DSA-less "mqa" mode is
        # unreachable from config, so non_absorbed_mqa always builds the indexer
        # and the old dsa-only hybrid_index_* keys were removed. The former
        # "mode + hybrid_index_*" delta no longer exists -- assert the two JSONs
        # are byte-for-value identical instead.
        self.assertEqual(delta(mqa, dsa), {})

    def test_baseline_intentionally_omits_the_switches(self):
        # The live-reference mha baseline must not carry either switch, so it
        # stays dense MHA with no sink.
        mha = _load_json(_MHA)
        self.assertNotIn("non_absorbed_mqa", mha)
        self.assertNotIn("add_full_attention_sink_bias", mha)


if __name__ == "__main__":
    unittest.main()
