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

"""Adversarial validation (agent A7): recompute / MTP / checkpoint compat for
the ``hybrid_mla_attention`` (dense MHA vs runtime-absorbed latent MQA, indexer
always built for ``"mqa_dsa"``) and ``add_full_attention_sink_bias`` (learnable
per-head sink) features of the ERNIE5 V2 ``dsv4_hybrid`` MoE. The mode labels
``mha``/``mqa``/``mqa_dsa`` below are retained only as test fixtures: ``mha``
maps to ``hybrid_mla_attention="mha"`` while ``mqa``/``mqa_dsa`` map to the
latent MQA modes (``"mqa_full_causal"`` / ``"mqa_dsa"``) and differ only by
whether the DSA indexer spec is wired into this direct-construction fixture.

Coverage map (see validation_reports/A7_recompute_mtp_ckpt.md for the analysis):
  1. Recompute equivalence: full-layer recompute ON == OFF, elementwise on the
     output and on every parameter / input gradient, for mqa and mqa_dsa, sink
     on/off. The load-bearing invariant is that the DSA ``token_indices`` are
     re-derived bit-identically on the recompute forward (a mismatch would
     silently differentiate a different sparsity pattern).
  2. Refined recompute guard: ``RefinedRcomputeFlashMaskAttention`` raises for a
     learnable sink unless the cute backend serves the picked fa version; on
     SM90 that means FA3 with ``FLASHMASK_FA3_USE_CUTEDSL`` on.
  3. MTP tracker slot arithmetic: MTP ``layer_number==0`` -> ``values[-1]`` is
     only accidentally correct for a 44-entry tracker; two MTP layers collide.
  4. Checkpoint key-set compat: MHA <-> MQA byte-identical; mqa_dsa adds the
     indexer keys; sink adds exactly one key per -2 layer.
  5. AOA name mapping: the sink is mapped both directions on the -2 layers only
     (incl. the MTP -2 layer), never on HCA (128) layers or non-hybrid configs.
  6. Optimizer state: the 1-D ``[num_heads]`` sink falls to Adam, not muon.
"""

import unittest

import numpy as np
import paddle

from paddlefleet.transformer.dsa_attention import DSAIndexerLossLoggingHelper

from .hybrid_mla_utils import (
    _CAPTURED,
    _CONFIG_DIR,
    _CSA_MQA_CFG,
    _DSA_CFG,
    _GPU,
    _HYBRID_MLA_CFGS,
    _MHA_CFG,
    _MINUS2_LAYERS,
    _MQA_DSA_CFGS,
    _NUM_HIDDEN,
    K_CHANNELS,
    H,
    _add_repo_root_to_sys_path,
    _build_module,
    _build_real_attn,
    _create_mqa_config,
    _flash_attn_version,
    _load_provider,
    _make_inputs,
    _production_fa_version,
    _rel,
    _row_end,
    _try_use_cuda_device,
    cpp_flashmask_backend,
)

_add_repo_root_to_sys_path()


# ===========================================================================
# Part 2 -- Refined recompute guard (no GPU kernels needed)
# ===========================================================================
class TestRefinedRecomputeSinkGuard(unittest.TestCase):
    """``RefinedRcomputeFlashMaskAttention.forward`` refuses a learnable sink
    unless the cute backend is active. The guard sits at the top of ``forward``
    (flash_attn.py:658-667), before any kernel call, so it can be exercised on
    tiny CPU tensors by forcing ``get_fa_version``'s return. FA3 only misses the
    sink while the hotfix switch routes it to the cpp kernel -- on cute it is
    served, so version 3 is checked inside ``cpp_flashmask_backend()``.
    """

    def _forward_module(self):
        from paddlefleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashMaskAttention,
        )

        return RefinedRcomputeFlashMaskAttention()

    def test_guard_raises_for_non_cute_versions(self):
        import paddlefleet.refined_recompute.flash_attn as fa_mod

        mod = self._forward_module()
        q = paddle.zeros([1, 4, 2, K_CHANNELS])  # last dim = q_head_dim 256
        k = paddle.zeros([1, 4, 2, K_CHANNELS])
        v = paddle.zeros([1, 4, 2, 128])  # last dim = v_head_dim 128
        sink = paddle.zeros([2])
        orig = fa_mod.get_fa_version
        try:
            with cpp_flashmask_backend():
                for version in (2, 3):
                    fa_mod.get_fa_version = lambda *a, **k_: version
                    with self.assertRaises(NotImplementedError) as ctx:
                        mod.forward(q, k, v, None, learnable_sink=sink)
                    self.assertIn(
                        "learnable_sink is only supported on the cute backend",
                        str(ctx.exception),
                    )
        finally:
            fa_mod.get_fa_version = orig

    def test_no_sink_skips_the_version_guard(self):
        """Sinkless refined recompute must not be gated by the guard: with
        ``learnable_sink=None`` the version block is skipped entirely."""
        import paddlefleet.refined_recompute.flash_attn as fa_mod

        mod = self._forward_module()
        q = paddle.zeros([1, 4, 2, K_CHANNELS])
        k = paddle.zeros([1, 4, 2, K_CHANNELS])
        v = paddle.zeros([1, 4, 2, 128])
        orig = fa_mod.get_fa_version

        def _boom(*a, **k_):
            raise AssertionError(
                "guard get_fa_version called for sinkless path"
            )

        try:
            fa_mod.get_fa_version = _boom
            # The guard is skipped; execution proceeds and fails later (in the
            # real kernel / _first_fwd), never with our sentinel AssertionError.
            with self.assertRaises(Exception) as ctx:
                mod.forward(q, k, v, None, learnable_sink=None)
            self.assertNotIn("guard get_fa_version called", str(ctx.exception))
        finally:
            fa_mod.get_fa_version = orig

    def test_production_dims_sink_support_follows_the_backend(self):
        """Under the production ``FLAGS_flash_attn_version`` the sink is served
        exactly when the picked version lands on the cute backend.

        On SM90 production picks FA3 and, with ``FLASHMASK_FA3_USE_CUTEDSL`` on
        (the default), cute serves it -- so the MLA pair (192, 128) does get the
        sink. (256, 128) is not in the cutedsl head-dim whitelist (256 requires
        ``head_dim_v == 256``), so it degrades to FA2 and the guard fires.
        Turning the hotfix switch off routes FA3 to Paddle's kernel, where the
        sink does not exist at any head dim."""
        try:
            from paddlefleet_ops.flash_mask_facade import (
                get_fa_version,
                uses_cutedsl_backend,
            )
        except Exception as exc:  # pragma: no cover - env dependent
            self.skipTest(f"flash_mask_facade unavailable: {exc}")
        production = _production_fa_version()
        if production == 2:
            self.skipTest("FA2-only box: no cute backend to dispatch to")

        def _cute(qd):
            try:
                return uses_cutedsl_backend(get_fa_version(qd, 128, None))
            except Exception as exc:  # pragma: no cover
                self.skipTest(f"get_fa_version({qd},128,None) failed: {exc}")

        with _flash_attn_version(production):
            self.assertTrue(_cute(192), "(192, 128) should stay on cute")
            self.assertFalse(_cute(256), "(256, 128) should degrade to FA2")
            with cpp_flashmask_backend():
                for qd in (192, 256):
                    self.assertFalse(
                        _cute(qd),
                        f"({qd}, 128) must leave cute once the switch is off",
                    )


# ===========================================================================
# Part 3 -- MTP indexer-loss tracker slot arithmetic (pure CPU logic)
# ===========================================================================
class TestMTPTrackerSlot(unittest.TestCase):
    """``DSAIndexerLossLoggingHelper.save_loss_to_tracker`` writes
    ``values[layer_number - 1]``. The MTP transformer layer is built with
    ``layer_number == 0`` (gpt_layer_specs.get_gpt_mtp_layers_spec_for_backend:
    ``layer_number=i`` with ``i`` starting at 0), so its loss lands in
    ``values[-1]``. For the production 43+1 model the tracker has 44 slots, so
    ``values[-1] == values[43]`` -- accidentally the trailing slot, and no main
    DSA (-2) layer maps there. This test pins that accident and shows it breaks
    for two MTP layers.
    """

    # -2 layers of the production model, expressed as the layer_number each
    # attention module receives (0-indexed decoder index == real_layer_number
    # when num_empty_layers_add_in_head == 0).
    MAIN_DSA_LAYERS = [8, 17, 26, 34, 42]

    def setUp(self):
        DSAIndexerLossLoggingHelper.tracker.clear()

    def tearDown(self):
        DSAIndexerLossLoggingHelper.tracker.clear()

    def _record(self, layer_number, num_layers, value):
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(
            loss=paddle.to_tensor(float(value)),
            layer_number=layer_number,
            num_layers=num_layers,
        )

    def test_single_mtp_lands_on_unique_trailing_slot(self):
        num_layers = 44  # 43 hidden + 1 MTP
        # main -2 layers write distinct interior slots ...
        for i, ln in enumerate(self.MAIN_DSA_LAYERS):
            self._record(ln, num_layers, i + 1)
        # ... MTP layer_number == 0 -> values[-1] == values[43]
        self._record(0, num_layers, 99.0)
        values = DSAIndexerLossLoggingHelper.tracker["values"].numpy()

        # MTP contribution recorded at the last slot.
        self.assertEqual(float(values[43]), 99.0)
        self.assertEqual(float(values[-1]), 99.0)
        # No collision: main layers occupy {7,16,25,33,41}, none is 43.
        main_slots = {ln - 1 for ln in self.MAIN_DSA_LAYERS}
        self.assertNotIn(43, main_slots)
        for slot in main_slots:
            self.assertNotEqual(float(values[slot]), 99.0)

    def test_two_mtp_layers_misplace_and_collide(self):
        """With ``num_nextn_predict_layers == 2`` the two MTP layers are built
        with ``layer_number`` 0 and 1. ``layer_number - 1`` maps them to
        ``values[-1]`` and ``values[0]``: the second MTP loss lands in the FIRST
        decoder slot instead of a trailing MTP slot -- a silent misplacement (and
        a collision if decoder layer 1 were also a -2/DSA layer)."""
        num_layers = 45  # 43 hidden + 2 MTP
        self._record(0, num_layers, 11.0)  # MTP #0 -> values[-1] == values[44]
        self._record(1, num_layers, 22.0)  # MTP #1 -> values[0]  (WRONG)
        values = DSAIndexerLossLoggingHelper.tracker["values"].numpy()

        self.assertEqual(float(values[44]), 11.0)
        # The bug: MTP #1 sits at slot 0, not at the intended trailing slot 43.
        self.assertEqual(float(values[0]), 22.0)
        self.assertEqual(float(values[43]), 0.0)
        # Demonstrate the collision hazard directly: a decoder layer with
        # layer_number == 1 would overwrite the same slot the second MTP used.
        self._record(1, num_layers, 5.0)
        values = DSAIndexerLossLoggingHelper.tracker["values"].numpy()
        self.assertEqual(float(values[0]), 27.0)  # 22 + 5 accumulated together


# ===========================================================================
# Part 5 -- AOA name mapping (runs on CPU; loads the real production configs)
# ===========================================================================
def _aoa_configs_available():
    try:
        import paddle as _pd

        # from_pretrained builds the provider, which may query device caps.
        if not _pd.is_compiled_with_cuda():
            _pd.cuda.get_device_capability = lambda device=None: (0, 0)
            _pd.device.cuda.get_device_capability = lambda device=None: (0, 0)
        from fleet_model.ernie5_v2.modeling import _gen_aoa_config  # noqa: F401
        from src.ernie_core_compat.configuration import (  # noqa: F401
            ErnieFleetModelConfig,
        )

        # Gate on the *baseline* config: it is the one every test here diffs
        # against. (It used to gate on ``_MQA_CFG``, a name that had drifted to
        # point at a directory which does not exist, so these tests silently
        # skipped instead of running.)
        return (_CONFIG_DIR / _MHA_CFG / "model_config.json").is_file()
    except Exception:
        return False


_AOA = unittest.skipUnless(
    _aoa_configs_available(),
    "requires the production model_config.json and the ernie5_v2 provider",
)


@_AOA
class TestSinkAOAMappingPerLayer(unittest.TestCase):
    """The learnable sink is AOA-mapped BOTH directions on exactly the -2
    layers (incl. the MTP -2 layer 43, whose Fleet name carries the
    ``.transformer_layer`` infix), and never on the HCA (128) layers or when the
    model is not a ``dsv4_hybrid`` hybrid-MLA. Complements agent A1's total
    count with the exact per-layer placement.
    """

    def _aoa(self, name):
        import paddle as _pd

        if not _pd.is_compiled_with_cuda():
            _pd.cuda.get_device_capability = lambda device=None: (0, 0)
            _pd.device.cuda.get_device_capability = lambda device=None: (0, 0)
        from fleet_model.ernie5_v2.modeling import (
            _gen_aoa_config,
            _gen_inv_aoa_config,
        )
        from src.ernie_core_compat.configuration import ErnieFleetModelConfig

        cfg = ErnieFleetModelConfig.from_pretrained(
            str(_CONFIG_DIR / name), _configuration_file="model_config.json"
        )
        # ``indexer_init_from_scratch`` is a YAML provider field, absent from the
        # JSON-only config built here, and ``_resolve_init_from_scratch``
        # (modeling.py:857) refuses to guess it for a ``"mqa_dsa"`` config. That
        # refusal is the intended behaviour (it decides which checkpoint the run
        # resumes from) and is asserted in
        # ``test_hybrid_mla_config_pipeline.py``; supply it the way the trainer
        # does. It never affects the sink statements this class inspects.
        cfg.indexer_init_from_scratch = True
        fwd = _gen_aoa_config(cfg)["aoa_statements"]
        inv = _gen_inv_aoa_config(cfg)["aoa_statements"]
        return cfg, fwd, inv

    @staticmethod
    def _sink_lines(statements):
        return [s for s in statements if "core_attention.softmax_offset" in s]

    def _layer_index_of(self, stmt):
        # statements look like "model.layers.<i>...core_attention.softmax_offset
        #   -> model.layers.<j>[.transformer_layer].self_attn...."
        import re

        idxs = re.findall(r"model\.layers\.(\d+)", stmt)
        return int(idxs[0]) if idxs else None

    def test_sink_mapped_on_exactly_the_minus2_layers_both_directions(self):
        # Now covers all four hybrid-MLA phases. It used to iterate
        # ``(_MQA_CFG, _DSA_CFG)``; ``_MQA_CFG`` had drifted onto
        # ``ernielite_layer43_mqa_hca``, a CSA full-causal MQA config with no
        # ``-2`` layer at all (it owns ``core_attention.attn_sink``, a different
        # tensor), so it could never satisfy this. The baseline is in the loop
        # now because it carries the sink too.
        for name in _HYBRID_MLA_CFGS:
            with self.subTest(config=name):
                _, fwd, inv = self._aoa(name)
                for direction, stmts in (("fwd", fwd), ("inv", inv)):
                    sink = self._sink_lines(stmts)
                    self.assertEqual(
                        len(sink), len(_MINUS2_LAYERS), f"{name}/{direction}"
                    )
                    layers = sorted({self._layer_index_of(s) for s in sink})
                    self.assertEqual(
                        layers, sorted(_MINUS2_LAYERS), f"{name}/{direction}"
                    )

    def test_mtp_sink_statement_uses_transformer_layer_infix(self):
        # The MTP layer (index 43) lives under ``.transformer_layer`` on the
        # Fleet side; its sink statement must carry that infix or the resume
        # loader would look for the parameter at the wrong key.
        _, fwd, inv = self._aoa(_DSA_CFG)
        for stmts in (fwd, inv):
            mtp = [
                s
                for s in self._sink_lines(stmts)
                if self._layer_index_of(s) == 43
            ]
            self.assertEqual(len(mtp), 1)
            self.assertIn(".transformer_layer", mtp[0])

    def test_no_sink_line_targets_a_128_hca_layer(self):
        _, fwd, inv = self._aoa(_DSA_CFG)
        hca_layers = set(range(_NUM_HIDDEN + 1)) - set(_MINUS2_LAYERS)
        for stmts in (fwd, inv):
            for s in self._sink_lines(stmts):
                self.assertNotIn(self._layer_index_of(s), hca_layers)

    def test_mha_config_maps_no_sink(self):
        """NAME IS HISTORICAL -- the negative control is now the CSA config.

        WAS: the mha baseline deliberately omitted
        ``add_full_attention_sink_bias`` (to leave the live reference run
        unperturbed) and therefore emitted zero sink statements. NO LONGER TRUE:
        the baseline sets the flag on purpose, so that
        ``core_attention.softmax_offset`` exists on both sides of the migration
        and the later phases add no parameter -- it emits one statement per
        ``-2`` layer, which the test above pins.

        NOW: the "no sink statement for a non-hybrid-MLA model" half of the
        property is asserted where it still holds --
        ``ernielite_layer43_mqa_hca``, a CSA full-causal MQA config with no ``-2``
        layer, which owns ``core_attention.attn_sink`` instead. That keeps a
        genuine negative control (the sink block must be gated on layer type, not
        emitted for every layer) instead of an assertion that only passed because
        the flag happened to be off.
        """
        _, fwd, inv = self._aoa(_CSA_MQA_CFG)
        self.assertEqual(self._sink_lines(fwd), [])
        self.assertEqual(self._sink_lines(inv), [])
        # ... and it does map its own, differently-named sink on every layer.
        self.assertTrue(
            any("core_attention.attn_sink" in s for s in fwd),
            "the CSA family sink (attn_sink) must still be mapped",
        )


# ===========================================================================
# Part 6 -- Optimizer state: the 1-D sink must fall to Adam, not muon
# ===========================================================================
class TestSinkOptimizerRouting(unittest.TestCase):
    """Muon only orthogonalizes 2-D/3-D matrices; a 1-D ``[num_heads]`` vector
    like the sink must be routed to the Adam fallback. Consumer:
    ``paddle.optimizer.muon._default_should_use_muon`` (the predicate
    ``paddleformers.trainer.trainer._build_muon_param_info_map`` uses).
    """

    def test_1d_sink_is_not_muon(self):
        from paddle.optimizer import muon

        self.assertFalse(
            muon._default_should_use_muon(
                "layers.8.self_attn.core_attention.softmax_offset", [H], []
            )
        )

    def test_2d_and_3d_weights_are_muon(self):
        from paddle.optimizer import muon

        self.assertTrue(muon._default_should_use_muon("w", [H, H], []))
        self.assertTrue(muon._default_should_use_muon("w", [4, H, H], []))

    def test_exclude_pattern_still_wins_for_matrices(self):
        # Sanity: the routing predicate is a shape gate first, then a name
        # gate; a 1-D sink can never reach the name gate.
        from paddle.optimizer import muon

        self.assertFalse(
            muon._default_should_use_muon("embed.weight", [H, H], ["embed"])
        )


# ===========================================================================
# Part 4 -- Checkpoint key-set compatibility (real modules; needs CUDA)
# ===========================================================================
_REAL = unittest.skipUnless(
    _try_use_cuda_device() and _aoa_configs_available(),
    "requires a usable CUDA device + the production ernie5_v2 configs",
)


def _key_sig(module):
    return {
        k: (tuple(v.shape), str(v.dtype))
        for k, v in module.state_dict().items()
    }


@_REAL
class TestCheckpointKeySets(unittest.TestCase):
    """A ``dsv4_hybrid`` dense-MHA checkpoint must load into an absorbed
    (``hybrid_mla_attention="mqa_dsa"``) run unchanged (every MLA parameter is
    byte identical), the absorbed run adding only the trained-from-scratch
    indexer keys. Since ``"mqa_dsa"`` always wires the indexer, the old DSA-less
    ``mqa`` surface no longer exists, so the tests that purely contrasted
    mqa-vs-mqa_dsa were dropped as redundant; the second ``"mqa_dsa"`` config is
    the phase-3 sparse-loss one, which must have the identical key set.

    SINK: the learnable sink used to be introduced *by* the absorbed phase, so it
    showed up as "one extra key per -2 layer". The baseline
    ``model_config.json`` now sets ``add_full_attention_sink_bias`` as well, on
    purpose -- with the sink on both sides the parameter set is unchanged across
    the migration, which is the entire point. The "pre-sink checkpoint resumed
    into a sink run" hazard is therefore no longer reachable from the on-disk
    configs and is reproduced by toggling the flag synthetically
    (``_keys(..., sink=False)``) in
    ``test_presink_ckpt_into_sink_run_is_short_the_sink_key``, so the loader
    behaviour documented below stays covered.

    Loaders (task-4 file:line, from the loader source, not run here):
      * from_pretrained (HF import): warns and newly-initializes missing keys
        -- model_utils.py:2671-2676 (``logger.warning("... newly initialized")``).
      * unified-checkpoint resume: RAISES -- unified_checkpoint/load_local.py:82-83
        ``raise ValueError(f"missing_keys: {missing_keys}")``.
      * strict state_dict load: RAISES -- model_utils.py:919-926.
    So a pre-sink ckpt + sink run is a warn-and-reinit under HF import but a hard
    ValueError under a unified-checkpoint resume.
    """

    _LAYER = _MINUS2_LAYERS[0]  # the first -2 hybrid-MLA layer
    _SINK = "core_attention.softmax_offset"

    def _keys(self, cfg_name, sink=None):
        provider = _load_provider(cfg_name)[1]
        if sink is not None:
            # sink is the one switch we toggle synthetically; mode/indexer come
            # faithfully from the real config (mode-toggling on one provider is
            # NOT reliable -- the indexer wiring does not reset).
            provider.add_full_attention_sink_bias = sink
        # The dense-MHA sink needs the flashmask v4 kernel
        # (multi_latent_attention.py:561-581); production derives that flag in
        # ``TrainingArguments.__post_init__``, which a bare pytest never runs.
        with _flash_attn_version(_production_fa_version()):
            return _key_sig(_build_real_attn(provider, self._LAYER))

    @staticmethod
    def _indexer_keys(keyset):
        return {k for k in keyset if "indexer" in k}

    def test_mha_core_loads_unchanged_into_mqa_and_dsa(self):
        # The activation-level absorption keeps every MLA parameter byte
        # identical, so all MHA keys exist -- same shape and dtype -- in the
        # MQA_DSA runs. An MHA checkpoint therefore loads unchanged.
        #
        # WAS: ``assertNotIn(self._SINK, mha)  # baseline is sinkless``. NO
        # LONGER TRUE (see the class docstring); NOW the sink is asserted to be
        # present in the baseline *and* to survive into both mqa_dsa configs with
        # the same shape/dtype, which is the stronger form of the same
        # checkpoint-compatibility claim.
        mha = self._keys(_MHA_CFG)
        self.assertIn(self._SINK, mha)
        self.assertEqual(self._indexer_keys(set(mha)), set())
        for target in _MQA_DSA_CFGS:
            with self.subTest(target=target):
                other = self._keys(target)
                for name, sig in mha.items():
                    self.assertIn(
                        name, other, f"MHA key {name} missing from {target}"
                    )
                    self.assertEqual(
                        other[name],
                        sig,
                        f"{target} shape/dtype drift on {name}",
                    )

    def test_mqa_dsa_adds_sink_plus_indexer_over_mha(self):
        """NAME IS HISTORICAL -- ``mqa_dsa`` now adds ONLY the indexer.

        WAS: ``assertIn(self._SINK, extra)`` -- the sink was one of the keys the
        absorbed phase introduced. NO LONGER TRUE: the baseline carries it, so
        the only new keys are the five DSA indexer parameters. That is a stronger
        statement about the migration (the phase-2 run adds nothing but the module
        it is there to train), so it is asserted as an exact equality, and the
        sink is separately asserted to be on *both* sides.
        """
        mha = set(self._keys(_MHA_CFG))
        for target in _MQA_DSA_CFGS:
            with self.subTest(target=target):
                dsa = set(self._keys(target))
                self.assertEqual(
                    mha - dsa, set(), f"{target} dropped an MHA key"
                )
                self.assertIn(self._SINK, mha)
                self.assertIn(self._SINK, dsa)
                extra = dsa - mha
                # Everything new is an indexer parameter (trained from scratch)
                # -- not a renamed/dropped MLA weight and not the sink.
                self.assertEqual(extra, self._indexer_keys(dsa))
                self.assertEqual(len(extra), 5)

    def test_presink_ckpt_into_sink_run_is_short_the_sink_key(self):
        # NEGATIVE: a pre-sink checkpoint resumed into a sink run. Both on-disk
        # phases now enable the sink, so the "saved" side is the baseline config
        # with ``add_full_attention_sink_bias`` forced off -- exactly the state of
        # a checkpoint taken before the flag was added to the baseline
        # (`model_config.json`), which is what makes this hazard historical rather
        # than hypothetical. The sink key is missing for every sink run; for
        # mqa_dsa the indexer keys are additionally missing but those are new
        # trainable modules, whereas the sink gap is the compatibility hazard.
        saved = set(self._keys(_MHA_CFG, sink=False))
        self.assertNotIn(self._SINK, saved)
        for target in _HYBRID_MLA_CFGS:
            with self.subTest(target=target):
                expected = set(self._keys(target))
                missing = expected - saved
                self.assertIn(self._SINK, missing)
                non_indexer_missing = missing - self._indexer_keys(expected)
                self.assertEqual(non_indexer_missing, {self._SINK})
                self.assertEqual(saved - expected, set())


# ===========================================================================
# Part 1 -- Recompute equivalence (GPU: FlashMLA sparse fwd + cuDNN DSA bwd)
# ===========================================================================
def _fresh_leaf(t):
    out = t.clone().detach()
    out.stop_gradient = False
    return out


def _grad_rel(g_on, g_off):
    if g_off is None and g_on is None:
        return 0.0
    assert (g_on is None) == (g_off is None), "one grad present, the other None"
    return _rel(g_on, g_off)


@_GPU
class TestRecomputeEquivalence(unittest.TestCase):
    """Full-layer recompute must be a pure compute/memory trade: ON == OFF on
    the output and on every gradient. The load-bearing invariant is that the
    DSA ``token_indices`` are re-derived bit-identically on the recompute
    forward -- a mismatch would silently differentiate a *different* sparsity
    pattern and corrupt gradients with no error.

    ``loss_coeff==0`` here isolates the sparse-attention recompute itself (no
    indexer KL graph): the only trainable leaves are ``query`` and the sink, and
    the whole attention is a deterministic function of them, so ON must equal
    OFF to kernel precision. The indexer-loss recompute path is covered by
    ``TestRecomputeIndicesReplay``.
    """

    def _run(self, module, query, key, w_v, x, qr, row_end, use_recompute):
        from paddle.distributed.fleet.utils import recompute

        module.train()
        module.clear_gradients()
        q = _fresh_leaf(query)
        kwargs = {"v_b_proj_weight": w_v}
        if module.indexer is not None:
            kwargs.update(x=x, qr=qr)

        def fn(qin):
            return module(qin, key, None, None, row_end, **kwargs)

        out = recompute(fn, q) if use_recompute else fn(q)
        out.cast("float32").sum().backward()
        grads = {
            name: (None if p.grad is None else p.grad.detach().cast("float32"))
            for name, p in module.named_parameters()
        }
        grads["__query__"] = (
            None if q.grad is None else q.grad.detach().cast("float32")
        )
        return out.detach().cast("float32"), grads

    def _check(self, mode, sink):
        seqlen = 256
        layout = [40, 88, 128]
        query, key, w_v, x, qr = _make_inputs(seqlen, seed=7, with_hidden=True)
        row_end = _row_end(layout, seqlen)
        module = _build_module(
            _create_mqa_config(mode, loss_coeff=0.0),
            # bf16 for both modes: ``"mqa"`` is the full-causal phase, which
            # hands the sink to dense FA4 as ``learnable_sink`` and asserts that
            # dtype on it (``flash_mask/cute/interface.py:598``), matching
            # production's ``params_dtype``.
            bf16=True,
            sink=sink,
        )

        # Pinned per call site rather than per module: the full-causal phase has
        # exactly one backend and ``_assert_dense_fa4`` raises otherwise, while
        # ``test_production_dims_sink_support_follows_the_backend`` pins the
        # production version itself, so a module-wide pin would fight it.
        with _flash_attn_version(4):
            _CAPTURED.clear()
            out_off, g_off = self._run(
                module, query, key, w_v, x, qr, row_end, use_recompute=False
            )
            idx_off = _CAPTURED[-1]

            _CAPTURED.clear()
            out_on, g_on = self._run(
                module, query, key, w_v, x, qr, row_end, use_recompute=True
            )
        # Recompute re-runs the forward during backward: the sparse kernel is
        # therefore called twice, and BOTH calls must select identical columns.
        self.assertGreaterEqual(
            len(_CAPTURED), 2, "recompute did not re-forward"
        )
        for cap in _CAPTURED:
            np.testing.assert_array_equal(
                cap, idx_off, "recompute selected a different sparsity pattern"
            )

        # Output: identical function of identical inputs -> equal to precision.
        self.assertLess(
            _rel(out_on, out_off), 1e-5, f"{mode} sink={sink} output"
        )
        # Every gradient matches (a silent indices mismatch would blow this up).
        self.assertEqual(set(g_on), set(g_off))
        for name in g_off:
            rel = _grad_rel(g_on[name], g_off[name])
            self.assertLess(
                rel, 5e-3, f"{mode} sink={sink} grad[{name}] rel={rel}"
            )
        if sink is not None:
            self.assertIsNotNone(g_off["softmax_offset"])
            self.assertGreater(
                float(g_off["softmax_offset"].abs().max()),
                0.0,
                "dead sink grad",
            )

    def test_mqa_recompute_equivalence_no_sink(self):
        self._check("mqa", None)

    def test_mqa_recompute_equivalence_with_sink(self):
        self._check("mqa", np.linspace(1.0, 3.0, H))

    def test_mqa_dsa_recompute_equivalence_no_sink(self):
        self._check("mqa_dsa", None)

    def test_mqa_dsa_recompute_equivalence_with_sink(self):
        self._check("mqa_dsa", np.linspace(1.0, 3.0, H))


@_GPU
class TestRecomputeIndicesReplay(unittest.TestCase):
    """With the indexer KL loss ON (``loss_coeff>0``), the reentrant recompute
    runs pass 1 under ``no_grad`` (indices only) and pass 2 with grad (loss
    attached). The top-k must be deterministic across the two passes, and the
    indexer parameters must receive finite gradients through the recompute."""

    def test_dsa_recompute_indices_and_indexer_grads(self):
        from paddle.distributed.fleet.utils import recompute

        seqlen = 512
        module = _build_module(
            _create_mqa_config("mqa_dsa", loss_coeff=0.01), bf16=True
        )
        module.train()
        query, key, w_v, x, qr = _make_inputs(seqlen, seed=11, with_hidden=True)
        q = _fresh_leaf(query)
        row_end = _row_end([200, 312], seqlen)

        _CAPTURED.clear()
        DSAIndexerLossLoggingHelper.tracker.clear()

        def fn(qin):
            return module(
                qin, key, None, None, row_end, v_b_proj_weight=w_v, x=x, qr=qr
            )

        out = recompute(fn, q)
        out.cast("float32").sum().backward()

        self.assertGreaterEqual(len(_CAPTURED), 2)
        for cap in _CAPTURED:
            np.testing.assert_array_equal(cap, _CAPTURED[0])
        self.assertIn("values", DSAIndexerLossLoggingHelper.tracker)
        for name in ("wq_b", "wk", "weights_proj"):
            p = getattr(module.indexer, name).linear.weight
            self.assertIsNotNone(p.grad, f"indexer.{name} has no grad")
            self.assertTrue(
                bool(paddle.isfinite(p.grad.cast("float32")).all()),
                f"indexer.{name} grad not finite",
            )


if __name__ == "__main__":
    unittest.main()
