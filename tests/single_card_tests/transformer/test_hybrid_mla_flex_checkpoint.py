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

"""Flex-checkpoint (DCP) save/resume audit for the ``hybrid_mla_attention``
phase chain, driven through the real production code paths: the provider ->
``build_spec_layer`` module of a ``csa_compress_ratios == -2`` layer,
``paddle.amp.decorate`` as the trainer applies it
(``ernie5/src/trainers/pretraining_trainer.py:2608-2610``),
``paddle.distributed.save_state_dict`` / ``load_state_dict`` as
``Trainer._save_flex_model_state`` / ``_load_flex_checkpoint`` call them
(PaddleFormers ``trainer/trainer.py:1161-1198``, ``1204-1536``) and the real
``paddleformers.trainer.trainer_utils.init_optimizer``.

Coverage map:
  1. ``amp.decorate(level="O2", dtype="bfloat16")`` casts EVERY parameter of a
     -2 layer to bf16, including the DSA indexer's ``paddle.nn.LayerNorm``
     ``k_norm``: ``need_keep_fp32`` (paddle ``amp/auto_cast.py:240-273``) keeps
     LayerNorm in fp32 only for ``dtype == 'float16'``, and nothing in the
     indexer sets ``_cast_to_low_precision = False``. Anti-vacuity: the same
     parameters are asserted fp32 *before* the decorate call.
  2. Key inventory: ``"mqa_dsa"`` adds exactly the five
     ``core_attention.indexer.*`` keys over ``"mha"`` and perturbs no shared
     key's shape or dtype; a real ``save_state_dict`` writes metadata for
     exactly the requested keys.
  3. phase1 -> phase2 resume: because all keys are bf16, the trainer's
     ``bf16_filtered_sharded_state_dict`` (``trainer.py:1403-1420``) empties the
     model-state request, ``check_resumable_locally`` is then trivially True
     and the load is a silent no-op -- no missing-key warning for the five
     brand-new indexer keys. Anti-vacuity: the UNFILTERED request over the same
     checkpoint raises, naming an indexer key.
  4. Master-weight coverage, which is what actually restores a bf16 parameter
     once (3) has emptied the model-state request. ``init_optimizer``
     (``trainer_utils.py:1507-1631``) creates an accumulator only for a
     parameter of *the optimizer* whose ``struct_name + {".w_0",
     ".moment1_0", ...}`` is in the merged checkpoint metadata, and
     ``_assign_master_weights_to_model`` (``trainer.py:1437-1450``) assigns only
     ``if param.name in master_weights`` with no fallback. Three regimes are
     pinned: full metadata + full optimizer recovers all 16; phase-2-style
     metadata (optimizer state for the indexer only, as written under
     ``train_indexer_only``) leaves the 11 backbone params unrecoverable; and a
     ``train_indexer_only`` optimizer resuming a phase-1 checkpoint recovers
     nothing at all.
"""

import os
import shutil
import tempfile
import unittest

import paddle
import paddle.distributed as dist

from .hybrid_mla_utils import (
    _CONFIG_DIR,
    _DSA_CFG,
    _GPU,
    _MHA_CFG,
    _MINUS2_LAYERS,
    _PARENT_REPO_AVAILABLE,
    _add_repo_root_to_sys_path,
    _build_real_attn,
    _flash_attn_version,
    _load_provider,
    _production_fa_version,
)

_add_repo_root_to_sys_path()


def _paddleformers_available():
    """``paddleformers`` is not part of the PaddleFleet CI image.

    It must not be imported at module scope: a top-level import turns "not
    installed" into a *collection* error that aborts the whole single-card run
    instead of skipping this one file. Every other test in this directory that
    reaches into PaddleFormers or the erniebot parent repo imports inside a
    function for the same reason.
    """
    try:
        from paddleformers.trainer.trainer_utils import (  # noqa: F401
            init_optimizer,
        )
    except Exception:  # pragma: no cover - depends on the image
        return False
    return True


def _init_optimizer():
    """Late import, so collection never depends on paddleformers."""
    from paddleformers.trainer.trainer_utils import init_optimizer

    return init_optimizer


def setUpModule():
    """Skip from here rather than at import time.

    Everything in this file is driven off the erniebot parent repo's production
    ``model_config.json`` (loaded through PaddleFormers), so a standalone
    PaddleFleet checkout still collects these tests and exits 0.
    """
    if not _PARENT_REPO_AVAILABLE:
        raise unittest.SkipTest(
            f"requires the erniebot parent repo configs at {_CONFIG_DIR}"
        )
    if not _paddleformers_available():
        raise unittest.SkipTest("requires paddleformers")


# The five parameters that only exist for ``hybrid_mla_attention="mqa_dsa"``.
_INDEXER_KEYS = (
    "core_attention.indexer.k_norm.bias",
    "core_attention.indexer.k_norm.weight",
    "core_attention.indexer.weights_proj.weight",
    "core_attention.indexer.wk.weight",
    "core_attention.indexer.wq_b.weight",
)

_OPT_STATE_SUFFIXES = (".moment1_0", ".moment2_0", ".beta1_pow_acc_0")

_MODULE_CACHE = {}


def _layer(cfg_name, decorate=True):
    """Real self-attention module of the first -2 layer of a production config.

    Cached: each build parses the 44-layer provider, which dominates runtime.
    ``decorate`` replays the trainer's ``paddle.amp.decorate(models=model,
    level="O2", dtype="bfloat16")``.
    """
    key = (cfg_name, decorate)
    if key not in _MODULE_CACHE:
        _MODULE_CACHE[key] = _fresh_layer(cfg_name, decorate)
    return _MODULE_CACHE[key]


def _fresh_layer(cfg_name, decorate=True):
    """Uncached ``_layer``, for the tests that mutate parameter values."""
    _, provider = _load_provider(cfg_name)
    with _flash_attn_version(_production_fa_version()):
        module = _build_real_attn(provider, _MINUS2_LAYERS[0])
    if decorate:
        module = paddle.amp.decorate(
            models=module, level="O2", dtype="bfloat16"
        )
    return module


def _dtypes(module):
    return {
        k: v.local_tensor.dtype for k, v in module.sharded_state_dict().items()
    }


def _bf16_filtered(sharded_state_dict):
    """Verbatim ``Trainer._load_flex_checkpoint``'s inner filter
    (PaddleFormers ``trainer/trainer.py:1403-1409``).
    """
    return {
        k: v
        for k, v in sharded_state_dict.items()
        if v.local_tensor.dtype != paddle.bfloat16
    }


def _synth_metadata(model_keys, optimizer_keys):
    """A merged ``state_dict_metadata`` shaped like ``_load_flex_checkpoint``
    builds it from the three checkpoint subdirectories
    (``trainer.py:1282-1293``): ``model_state`` holds every model key,
    ``master_weight`` / ``optimizer_state`` only the optimizer's params.
    """
    metadata = dict.fromkeys(model_keys, "model_state")
    for k in optimizer_keys:
        metadata[k + ".w_0"] = "master_weight"
        for suffix in _OPT_STATE_SUFFIXES:
            metadata[k + suffix] = "optimizer_state"
    return metadata


def _master_weight_keys(optimizer, sharded_state_dict):
    """Struct names of the params ``init_optimizer`` gave a master weight."""
    static_to_struct = {
        v.local_tensor.name: k for k, v in sharded_state_dict.items()
    }
    master_weights = getattr(optimizer, "_master_weights", None) or {}
    return sorted(static_to_struct.get(name, name) for name in master_weights)


# ===========================================================================
# Part 1 -- what amp.decorate does to the dtypes
# ===========================================================================
@_GPU
class TestAmpDecorateDtypes(unittest.TestCase):
    """The whole resume behaviour hinges on the post-decorate dtypes, because
    the trainer drops every bf16 key from the model-state request. Production
    decorates at ``pretraining_trainer.py:2610``; this pins the outcome for a
    real -2 layer of both the phase-1 and the phase-2 config.
    """

    def test_indexer_layernorm_is_fp32_before_decorate(self):
        # Anti-vacuity for the next test: ``DSAIndexer.k_norm`` is a plain
        # ``paddle.nn.LayerNorm`` built with no dtype
        # (``dsa_attention.py:348-353``), so it starts fp32 even though the
        # provider's ``params_dtype`` is bf16.
        dtypes = _dtypes(_layer(_DSA_CFG, decorate=False))
        fp32 = sorted(k for k, v in dtypes.items() if v == paddle.float32)
        self.assertIn("core_attention.indexer.k_norm.weight", fp32)
        self.assertIn("core_attention.indexer.k_norm.bias", fp32)

    def test_decorate_casts_every_parameter_to_bf16(self):
        for cfg in (_MHA_CFG, _DSA_CFG):
            with self.subTest(cfg=cfg):
                before = _dtypes(_layer(cfg, decorate=False))
                after = _dtypes(_layer(cfg, decorate=True))
                self.assertEqual(sorted(before), sorted(after))
                # at least one parameter actually changed dtype
                changed = [k for k in before if before[k] != after[k]]
                self.assertTrue(changed, "decorate changed nothing")
                self.assertEqual(
                    [k for k, v in after.items() if v != paddle.bfloat16],
                    [],
                    "a non-bf16 parameter would survive the trainer's "
                    "bf16 filter and be requested from the checkpoint",
                )


# ===========================================================================
# Part 2 -- key inventory and what a real save writes
# ===========================================================================
@_GPU
class TestCheckpointInventory(unittest.TestCase):
    """``Trainer._save_flex_model_state`` (``trainer.py:1161-1173``) saves
    ``self.model.sharded_state_dict()`` unfiltered, and
    ``GPTModel.sharded_state_dict`` (``gpt_model.py:712-768``) only renames
    keys. So the -2 layer's own sharded state dict IS the per-layer save
    inventory.
    """

    def test_mqa_dsa_adds_exactly_the_indexer_keys(self):
        mha = _layer(_MHA_CFG).sharded_state_dict()
        dsa = _layer(_DSA_CFG).sharded_state_dict()
        self.assertEqual(sorted(set(dsa) - set(mha)), sorted(_INDEXER_KEYS))
        self.assertEqual(sorted(set(mha) - set(dsa)), [])
        for k in sorted(set(mha) & set(dsa)):
            self.assertEqual(
                (
                    tuple(mha[k].global_shape),
                    tuple(mha[k].local_shape),
                    tuple(mha[k].global_offset),
                    mha[k].local_tensor.dtype,
                ),
                (
                    tuple(dsa[k].global_shape),
                    tuple(dsa[k].local_shape),
                    tuple(dsa[k].global_offset),
                    dsa[k].local_tensor.dtype,
                ),
                f"{k} drifted between the phase-1 and phase-2 config",
            )

    def test_save_state_dict_writes_metadata_for_every_key(self):
        sharded = _layer(_MHA_CFG).sharded_state_dict()
        self.assertTrue(sharded, "empty state dict would make this vacuous")
        path = tempfile.mkdtemp(prefix="hybrid_mla_ckpt_")
        try:
            dist.save_state_dict(sharded, path)
            metadata = paddle.load(os.path.join(path, "0.metadata"))
            self.assertEqual(
                sorted(metadata.state_dict_metadata), sorted(sharded)
            )
        finally:
            shutil.rmtree(path, ignore_errors=True)


# ===========================================================================
# Part 3 -- phase1 -> phase2 model-state resume
# ===========================================================================
@_GPU
class TestPhase1ToPhase2ModelStateResume(unittest.TestCase):
    """A phase-1 (``"mha"``) checkpoint resumed by a phase-2 (``"mqa_dsa"``)
    model. The five indexer keys are absent from the checkpoint, so the
    question is whether the DCP load errors, warns, or silently keeps the fresh
    initialisation. Both answers are pinned: unfiltered it raises, and with the
    trainer's bf16 filter it is a silent no-op.
    """

    @classmethod
    def setUpClass(cls):
        cls.path = tempfile.mkdtemp(prefix="hybrid_mla_phase1_")
        phase1 = _fresh_layer(_MHA_CFG)
        with paddle.no_grad():
            for p in phase1.parameters():
                p.set_value(paddle.full(p.shape, 0.25, dtype=p.dtype))
        dist.save_state_dict(phase1.sharded_state_dict(), cls.path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.path, ignore_errors=True)

    def test_unfiltered_request_raises_naming_an_indexer_key(self):
        # Anti-vacuity control for the filtered case below: DCP has no
        # tolerance for a requested key that the checkpoint lacks -- the
        # missing/unexpected-key computation only warns
        # (``load_state_dict.py:308-327``) and the reshard then fails on a
        # plain dict lookup (``resharder.py:249``).
        sharded = _fresh_layer(_DSA_CFG).sharded_state_dict()
        with self.assertRaises(Exception) as ctx:
            dist.load_state_dict(sharded, self.path)
        message = str(ctx.exception)
        self.assertTrue(
            any(k in message for k in _INDEXER_KEYS),
            f"expected an indexer key in the failure, got: {message}",
        )

    def test_bf16_filter_empties_the_request(self):
        sharded = _layer(_DSA_CFG).sharded_state_dict()
        self.assertEqual(_bf16_filtered(sharded), {})

    def test_filtered_resume_is_a_silent_no_op(self):
        from paddle.distributed.flex_checkpoint.dcp.load_state_dict import (
            _metadata_manager,
        )
        from paddle.distributed.flex_checkpoint.dcp.utils import (
            check_resumable_locally,
        )

        phase2 = _fresh_layer(_DSA_CFG)
        sharded = phase2.sharded_state_dict()
        request = _bf16_filtered(sharded)
        metadata = paddle.load(os.path.join(self.path, "0.metadata"))

        _metadata_manager.set_metadata_list([metadata])
        try:
            # An empty request makes the fast path trivially satisfiable
            # (``utils.py:700-745``), so ``load_state_dict`` skips the reshard
            # and never computes missing keys (``load_state_dict.py:853-874``).
            self.assertTrue(
                check_resumable_locally(
                    self.path, request, _metadata_manager, False, None
                )
            )
        finally:
            _metadata_manager.clear()

        before = {k: v.numpy().copy() for k, v in phase2.named_parameters()}
        dist.load_state_dict(request, self.path)
        after = dict(phase2.named_parameters())
        for k, value in before.items():
            self.assertTrue(
                (value == after[k].numpy()).all(),
                f"{k} changed although nothing was requested",
            )
        # ... and in particular the pretrained 0.25 never arrived: the whole
        # backbone is still at its fresh random initialisation.
        o_proj = after["o_proj.weight"].astype("float32").numpy()
        self.assertFalse(
            (o_proj == 0.25).all(), "checkpoint value unexpectedly present"
        )


# ===========================================================================
# Part 4 -- master-weight coverage (what actually restores a bf16 param)
# ===========================================================================
@_GPU
class TestMasterWeightCoverage(unittest.TestCase):
    """Once part 3 has emptied the model-state request, a bf16 parameter can
    only come back through ``recover_params_from_master_weight``
    (``trainer.py:1452-1511``), whose assignment is gated on ``param.name in
    master_weights`` with no fallback. A master weight exists only if
    ``init_optimizer`` created an accumulator for that param, which requires
    (a) the param to be in the optimizer and (b) its optimizer-state keys to be
    in the merged checkpoint metadata (``trainer_utils.py:1507-1631``).
    """

    def _optimizer(self, module, params, metadata):
        sharded = module.sharded_state_dict()
        optimizer = paddle.optimizer.AdamW(
            learning_rate=1e-4, parameters=params, multi_precision=True
        )
        _init_optimizer()(optimizer, sharded, metadata)
        return sharded, optimizer

    def test_full_metadata_and_full_optimizer_recovers_every_param(self):
        """Phase 2 -> phase 3 / phase 3 -> phase 4: identical parameter set,
        backbone unfrozen on both sides, so every key has a master weight.
        """
        module = _fresh_layer(_DSA_CFG)
        keys = list(module.sharded_state_dict())
        sharded, optimizer = self._optimizer(
            module,
            list(module.parameters()),
            _synth_metadata(keys, keys),
        )
        self.assertEqual(_master_weight_keys(optimizer, sharded), sorted(keys))
        master_weights = optimizer._master_weights
        for param in module.parameters():
            self.assertIn(param.name, master_weights)
            master = master_weights[param.name]
            self.assertEqual(master.dtype, paddle.float32)
            self.assertEqual(list(master.shape), list(param.shape))

    def test_phase2_metadata_leaves_the_backbone_unrecoverable(self):
        """The bug. Phase 2 runs with ``train_indexer_only: true``
        (``pretraining_trainer.py:3646-3689`` freezes, ``:3699-3703`` builds the
        optimizer from ``stop_gradient is False``), so
        ``_save_flex_optimizer_state`` (``trainer.py:1175-1198``) writes
        master weights for the indexer ONLY. Resuming that checkpoint with a
        full-backbone optimizer (phase 3), ``init_optimizer`` skips every
        backbone param, the bf16 filter has already dropped it from the
        model-state request, and it stays at its random initialisation with no
        warning anywhere.
        """
        module = _fresh_layer(_DSA_CFG)
        keys = list(module.sharded_state_dict())
        sharded, optimizer = self._optimizer(
            module,
            list(module.parameters()),
            _synth_metadata(keys, list(_INDEXER_KEYS)),
        )
        covered = _master_weight_keys(optimizer, sharded)
        self.assertEqual(covered, sorted(_INDEXER_KEYS))
        unrecoverable = sorted(set(keys) - set(covered))
        # every one of them is bf16, hence absent from the model-state request
        for k in unrecoverable:
            self.assertEqual(sharded[k].local_tensor.dtype, paddle.bfloat16)
        self.assertEqual(len(unrecoverable), len(keys) - len(_INDEXER_KEYS))
        self.assertIn("o_proj.weight", unrecoverable)
        self.assertIn("kv_b_proj.weight", unrecoverable)

    def test_train_indexer_only_resume_of_phase1_recovers_nothing(self):
        """The same failure one phase earlier, and total: a phase-1 checkpoint
        has master weights for the backbone and none for the indexer, while a
        ``train_indexer_only`` optimizer holds the indexer and none of the
        backbone. The intersection is empty, so not one of the 16 parameters is
        restored.
        """
        module = _fresh_layer(_DSA_CFG)
        keys = list(module.sharded_state_dict())
        backbone = [k for k in keys if "indexer" not in k]
        indexer_params = [
            p for n, p in module.named_parameters() if "indexer" in n
        ]
        self.assertEqual(len(indexer_params), len(_INDEXER_KEYS))
        sharded, optimizer = self._optimizer(
            module,
            indexer_params,
            _synth_metadata(backbone, backbone),
        )
        self.assertEqual(_master_weight_keys(optimizer, sharded), [])
        self.assertEqual(_bf16_filtered(sharded), {})


if __name__ == "__main__":
    unittest.main()
