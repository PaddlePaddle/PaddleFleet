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

"""HF-format save/load validation for the ``hybrid_mla_attention`` enum.

``test_hybrid_mla_config_pipeline.py::TestAOAStatements`` only inspects the
*strings* produced by ``_gen_aoa_config`` / ``_gen_inv_aoa_config``. This module
runs the statements through the real engine: it drives
``Layer.sharded_state_dict`` + ``full_param`` + ``save_full_param`` +
``replace_name_and_gen_index`` (what ``HFFormatFullParamSaver`` does, so what
``pretraining_trainer.py:3506-3560`` does on ``should_save_hf``) and
``dist.load_state_dict(..., safetensors=True)`` (what
``PaddleFormers trainer.py:1250-1280`` does under ``--load_from_hf``), on one
real ``csa_compress_ratios == -2`` layer built from the production
``model_config.json``.

What is proven here, and why each one matters for the phase-1 -> phase-2/3/4
continuation story:

1. **Save is complete.** Every tensor of the built module reaches disk exactly
   once, under the name the module uses, with the HF (transposed) orientation
   for 2-D weights. For ``mqa_dsa`` that includes the five
   ``core_attention.indexer.*`` tensors and ``core_attention.softmax_offset``.
   A tensor missing here would be silently lost at every ``save_hf_steps``.
2. **Round trips are bit-exact.** Save -> load into a differently-seeded module
   reproduces every value bit-for-bit, for both ``"mha"`` and ``"mqa_dsa"``. The
   anti-vacuity assertion requires the two modules to actually disagree before
   the load, otherwise "equal after load" would prove nothing.
3. **Cross-phase load is decided by ``indexer_init_from_scratch``, loudly.**
   Loading a phase-1 (``"mha"``, no indexer on disk) checkpoint into a
   ``"mqa_dsa"`` module succeeds with ``True`` (the ``_ -> key`` add primitive)
   and raises with ``False``. The failure mode of a mis-set switch must be an
   exception, not a zero-initialised indexer.
4. **The ``_ -> key`` primitive discards a checkpoint tensor of the same name.**
   ``aoa_engine.py:581-586`` routes ``_ -> key`` into ``need_add_output_vars``;
   ``:685-686`` then sets ``output_vars[key] = None`` and ``:715-720`` returns no
   source slices, so the destination keeps whatever the freshly built module
   held. Leaving ``indexer_init_from_scratch: true`` in the YAML when restarting
   phase 3 from a phase-2 checkpoint therefore throws away the trained indexer,
   reported only as a generic "Unexpected keys" warning
   (``load_state_dict.py:325-327``) whose text is wrong -- the keys *are* in the
   model state dict. The current behaviour is pinned as a documented hazard, and
   the behaviour we would want is a companion ``expectedFailure``.
"""

import unittest
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import paddle

from .hybrid_mla_utils import (
    _CONFIG_DIR,
    _DSA_CFG as _DSA,
    _MHA_CFG as _MHA,
    _PARENT_REPO_AVAILABLE,
    _add_repo_root_to_sys_path,
    _build_real_attn,
    _flash_attn_version,
    _load_provider,
    _production_fa_version,
    _stub_device_capability,
    _try_use_cuda_device,
)

if not _try_use_cuda_device():
    _stub_device_capability()

LAYER = 7
PREFIX = f"model.layers.{LAYER}.self_attn."

# ``DSAIndexer`` builds exactly these five (dsa_attention.py:290-367); they are
# the whole parameter delta of ``"mqa_dsa"`` over ``"mha"``.
_INDEXER_KEYS = (
    "core_attention.indexer.wq_b.weight",
    "core_attention.indexer.wk.weight",
    "core_attention.indexer.k_norm.weight",
    "core_attention.indexer.k_norm.bias",
    "core_attention.indexer.weights_proj.weight",
)
# ``k_norm`` is initialised deterministically (weight ones / bias zeros), so two
# differently seeded modules agree on it and it cannot witness a lost value.
_INDEXER_RANDOM_KEYS = (
    "core_attention.indexer.wq_b.weight",
    "core_attention.indexer.wk.weight",
    "core_attention.indexer.weights_proj.weight",
)


def setUpModule():
    """Driven off the erniebot ``model_config.json`` on disk, so meaningless in
    a standalone PaddleFleet checkout (what upstream CI builds). Skipping here
    rather than at import time keeps the tests collected.
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
# Harness. flex_checkpoint decides "distributed or not" from
# ``paddle.distributed.get_world_size()`` at call time (full_param.py:321,
# load_state_dict.py:89/604/1223/...). A training pod leaks
# ``PADDLE_TRAINERS_NUM=8`` into the environment, which makes those helpers try
# to all-gather over a process group this single-process test never initialises.
# Pinning the world size is the faithful single-card configuration.
# ---------------------------------------------------------------------------
def _single_process_world():
    return mock.patch.object(
        paddle.distributed, "get_world_size", return_value=1
    )


def _attn(name, seed):
    _, provider = _load_provider(name)
    with _flash_attn_version(_production_fa_version()):
        return _build_real_attn(provider, LAYER, seed=seed)


def _aoa(name, indexer_init_from_scratch=True):
    """``(forward, inverse)`` statement lists, exactly as the trainer builds
    them. ``indexer_init_from_scratch`` lives in the training YAML, which
    ``_load_provider`` deliberately does not read, so set it explicitly.
    """
    _add_repo_root_to_sys_path()
    from fleet_model.ernie5_v2.modeling import (
        _gen_aoa_config,
        _gen_inv_aoa_config,
    )
    from src.ernie_core_compat.configuration import ErnieFleetModelConfig

    cfg = ErnieFleetModelConfig.from_pretrained(
        str(_CONFIG_DIR / name), _configuration_file="model_config.json"
    )
    cfg.indexer_init_from_scratch = indexer_init_from_scratch
    return (
        _gen_aoa_config(cfg)["aoa_statements"],
        _gen_inv_aoa_config(cfg)["aoa_statements"],
    )


def _side_vars(side):
    """Variable names on one side of a statement, ``^T`` and attrs stripped."""
    out = []
    for tok in (t.strip() for t in side.split(",")):
        if not tok or "=" in tok or tok == "fused_ffn":
            continue
        out.append(tok.removesuffix("^T"))
    return out


def _filter(stmts, keys, fleet_on_left):
    """Statements whose Fleet-side variables all belong to ``keys``.

    The generators emit statements for all 44 layers plus the MoE/embedding
    weights; a single-layer module can only be driven by its own.
    """
    kept = []
    for s in stmts:
        lhs, rhs = s.split("->")
        names = [
            v for v in _side_vars(lhs if fleet_on_left else rhs) if v != "_"
        ]
        if names and all(v in keys for v in names):
            kept.append(s.strip())
    return kept


def _save_hf(attn, inv_stmts, path):
    """Write ``attn`` to HF safetensors through the production saver.

    Same three primitives ``HFFormatFullParamSaver`` chains
    (``PaddleFormers model_utils.py:4033-4113``): ``sharded_state_dict`` ->
    ``full_param(aoa_config=inverse)`` -> ``save_full_param`` ->
    ``replace_name_and_gen_index``. Returns the ``(name, shape, dtype)`` triples
    the AOA engine emitted, in emission order, so duplicates stay visible.
    """
    from paddle.distributed.flex_checkpoint.dcp.full_param import full_param
    from paddleformers.transformers.model_utils import (
        replace_name_and_gen_index,
        save_full_param,
    )

    path.mkdir(parents=True, exist_ok=True)
    emitted = []

    def tap():
        sd = attn.sharded_state_dict(PREFIX)
        for key, tensor in full_param(
            sd, aoa_config={"aoa_statements": inv_stmts}
        ):
            emitted.append((key, tuple(tensor.shape), str(tensor.dtype)))
            yield key, tensor

    with _single_process_world():
        total = save_full_param(
            itr=tap(),
            save_dir=str(path),
            rank=0,
            moe_sharding_world_size=1,
            num_saver_ranks=1,
        )
        replace_name_and_gen_index(str(path), total)
    return emitted


def _reset_metadata_cache():
    """``load_state_dict`` caches the checkpoint metadata in a module-global
    (``load_state_dict.py:74`` + ``:1245-1249``: it is only refilled when
    empty), so a second load of a *different* directory in the same process
    silently reuses the first one's file list. Production loads once per
    process; a test that loads several checkpoints has to clear it.
    """
    from paddle.distributed.flex_checkpoint.dcp import load_state_dict as _lsd

    _lsd._metadata_manager.clear()


def _load_hf(attn, fwd_stmts, path):
    _reset_metadata_cache()
    with _single_process_world():
        paddle.distributed.load_state_dict(
            attn.sharded_state_dict(PREFIX),
            str(path),
            aoa_config={"aoa_statements": fwd_stmts},
            safetensors=True,
        )


def _on_disk(path):
    """``{name: shape}`` actually present in the safetensors shards."""
    from safetensors.numpy import load_file

    out = {}
    for shard in sorted(Path(path).glob("model-*.safetensors")):
        out.update({k: v.shape for k, v in load_file(str(shard)).items()})
    return out


def _diff_keys(left, right):
    """Module keys whose values differ (fp32-compared, shape-aware)."""
    out = []
    for key, ref in right.items():
        got = left[key]
        if tuple(got.shape) != tuple(ref.shape):
            out.append(key)
            continue
        a = got.astype("float32")
        b = ref.astype("float32")
        if float((a - b).abs().max()) != 0.0:
            out.append(key)
    return sorted(out)


@_requires_cuda
class TestHFSaveCoverage(unittest.TestCase):
    """Q1: everything the module holds reaches disk, exactly once."""

    def _save_and_check(self, cfg_name):
        attn = _attn(cfg_name, seed=11)
        module_keys = {PREFIX + k for k in attn.state_dict()}
        _, inverse = _aoa(cfg_name, indexer_init_from_scratch=True)
        inv = _filter(inverse, module_keys, fleet_on_left=True)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "hf"
            emitted = _save_hf(attn, inv, path)
            disk = _on_disk(path)
        names = [e[0] for e in emitted]
        self.assertEqual(
            sorted(names),
            sorted(module_keys),
            f"{cfg_name}: HF save does not cover the module 1:1",
        )
        self.assertEqual(
            len(names),
            len(set(names)),
            f"{cfg_name}: a tensor was emitted twice",
        )
        self.assertEqual(set(disk), module_keys)
        return attn, dict(zip(names, (e[1] for e in emitted))), disk

    def test_mha_save_covers_every_module_tensor_once(self):
        attn, shapes, disk = self._save_and_check(_MHA)
        for key in attn.state_dict():
            self.assertIn(PREFIX + key, disk)
        self.assertIn(PREFIX + "core_attention.softmax_offset", disk)
        self.assertNotIn(PREFIX + _INDEXER_KEYS[0], disk)

    def test_mqa_dsa_save_covers_the_indexer_and_the_sink(self):
        attn, shapes, disk = self._save_and_check(_DSA)
        module = attn.state_dict()
        for key in _INDEXER_KEYS:
            with self.subTest(key=key):
                self.assertIn(PREFIX + key, disk, "indexer tensor never saved")
        # 2-D weights are stored transposed (HF/torch orientation); 1-D ones
        # keep their shape. A stale name or a missed transpose shows up here.
        for key in _INDEXER_KEYS:
            got = tuple(disk[PREFIX + key])
            want = tuple(module[key].shape)
            if len(want) == 2:
                want = want[::-1]
            self.assertEqual(got, want, f"{key} saved with the wrong shape")
        self.assertIn(PREFIX + "core_attention.softmax_offset", disk)
        self.assertEqual(
            tuple(disk[PREFIX + "core_attention.softmax_offset"]),
            tuple(module["core_attention.softmax_offset"].shape),
        )

    def test_documented_bug_gate_proj_is_saved_untransposed(self):
        """``self_attn.gate_proj`` has no AOA statement in either direction
        (the generators test ``use_gated_attn`` while the production JSON says
        ``gated_attention``), already pinned at the statement level by
        ``test_hybrid_mla_config_pipeline.py::
        test_documented_bug_attention_gate_proj_dropped_from_aoa``.

        The consequence on disk: the save path builds its engine with
        ``destination_state_shard_info=None`` (``full_param.py:103-126``), so
        ``aoa_engine.py:697-710`` passes every unconsumed input through under
        its own name. The tensor is therefore not lost -- it is written with
        Fleet orientation while every other 2-D weight is transposed, i.e. it
        would be wrong for a genuine torch-produced checkpoint but round trips
        inside this codebase.
        """
        attn = _attn(_DSA, seed=11)
        module_keys = {PREFIX + k for k in attn.state_dict()}
        _, inverse = _aoa(_DSA, indexer_init_from_scratch=True)
        inv = _filter(inverse, module_keys, fleet_on_left=True)
        gate = PREFIX + "gate_proj.weight"
        self.assertNotIn(
            gate, [s.split("->")[1].strip() for s in inv], "bug fixed?"
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "hf"
            _save_hf(attn, inv, path)
            disk = _on_disk(path)
        self.assertIn(gate, disk, "pass-through must still write the tensor")
        self.assertEqual(
            tuple(disk[gate]),
            tuple(attn.state_dict()["gate_proj.weight"].shape),
            "gate_proj is expected to be saved untransposed",
        )
        other = PREFIX + "q_a_proj.weight"
        self.assertEqual(
            tuple(disk[other]),
            tuple(attn.state_dict()["q_a_proj.weight"].shape)[::-1],
            "every mapped 2-D weight is transposed, so the contrast is real",
        )


@_requires_cuda
class TestHFRoundTrip(unittest.TestCase):
    """Q3: save -> load reproduces every value bit-for-bit."""

    def _round_trip(self, cfg_name):
        src = _attn(cfg_name, seed=11)
        dst = _attn(cfg_name, seed=99)
        module_keys = {PREFIX + k for k in src.state_dict()}
        forward, inverse = _aoa(cfg_name, indexer_init_from_scratch=False)
        inv = _filter(inverse, module_keys, fleet_on_left=True)
        fwd = _filter(forward, module_keys, fleet_on_left=False)
        before = {k: v.clone() for k, v in dst.state_dict().items()}
        differ = _diff_keys(before, src.state_dict())
        # ANTI-VACUITY: without a real disagreement, "equal after load" would
        # hold even if the loader did nothing at all. Only these five are
        # seeded deterministically (layernorm gains, k_norm, one VHA factor),
        # so everything else has to disagree.
        deterministic = {
            "q_a_layernorm.weight",
            "kv_a_layernorm.weight",
            "vha_postmix_V",
            "core_attention.indexer.k_norm.weight",
            "core_attention.indexer.k_norm.bias",
        }
        equal = {k for k in src.state_dict() if k not in differ}
        self.assertTrue(
            equal <= deterministic,
            f"{cfg_name}: unexpectedly equal before load: {equal}",
        )
        self.assertGreaterEqual(
            len(differ), 8, f"{cfg_name}: too few moving tensors"
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "hf"
            _save_hf(src, inv, path)
            _load_hf(dst, fwd, path)
        return src, dst, differ

    def test_mha_round_trip_is_bitwise_exact(self):
        src, dst, _ = self._round_trip(_MHA)
        self.assertEqual(_diff_keys(dst.state_dict(), src.state_dict()), [])
        for key, ref in src.state_dict().items():
            self.assertEqual(dst.state_dict()[key].dtype, ref.dtype)

    def test_mqa_dsa_round_trip_is_bitwise_exact(self):
        src, dst, differ = self._round_trip(_DSA)
        self.assertEqual(_diff_keys(dst.state_dict(), src.state_dict()), [])
        for key, ref in src.state_dict().items():
            self.assertEqual(dst.state_dict()[key].dtype, ref.dtype)
        # The indexer is the point of the exercise: it must have been one of
        # the tensors that actually moved.
        for key in _INDEXER_RANDOM_KEYS:
            self.assertIn(key, differ)


@_requires_cuda
class TestCrossPhaseLoad(unittest.TestCase):
    """Q2: a phase-1 ``"mha"`` checkpoint into a ``"mqa_dsa"`` module."""

    def _phase1_ckpt(self, tmp):
        src = _attn(_MHA, seed=11)
        keys = {PREFIX + k for k in src.state_dict()}
        _, inverse = _aoa(_MHA, indexer_init_from_scratch=True)
        path = Path(tmp) / "phase1"
        _save_hf(src, _filter(inverse, keys, fleet_on_left=True), path)
        for key in _INDEXER_KEYS:
            self.assertNotIn(
                PREFIX + key,
                _on_disk(path),
                "phase-1 checkpoint must not contain an indexer",
            )
        return src, path

    def test_init_from_scratch_true_keeps_the_freshly_built_indexer(self):
        dst = _attn(_DSA, seed=99)
        keys = {PREFIX + k for k in dst.state_dict()}
        forward, _ = _aoa(_DSA, indexer_init_from_scratch=True)
        fwd = _filter(forward, keys, fleet_on_left=False)
        adds = [s for s in fwd if s.split("->")[0].strip() == "_"]
        self.assertEqual(len(adds), len(_INDEXER_KEYS))
        untouched = {
            k: v.clone() for k, v in dst.state_dict().items() if "indexer" in k
        }
        with TemporaryDirectory() as tmp:
            src, path = self._phase1_ckpt(tmp)
            _load_hf(dst, fwd, path)
        # The backbone came from the phase-1 checkpoint ...
        after = dst.state_dict()
        self.assertEqual(_diff_keys(after, src.state_dict()), [])
        # ... and the indexer is exactly what the module built for itself.
        self.assertEqual(len(untouched), len(_INDEXER_KEYS))
        self.assertEqual(_diff_keys(after, untouched), [])

    def test_init_from_scratch_false_refuses_a_checkpoint_without_indexer(self):
        dst = _attn(_DSA, seed=99)
        keys = {PREFIX + k for k in dst.state_dict()}
        forward, _ = _aoa(_DSA, indexer_init_from_scratch=False)
        fwd = _filter(forward, keys, fleet_on_left=False)
        self.assertEqual(
            [s for s in fwd if s.split("->")[0].strip() == "_"],
            [],
            "with the switch off there must be no add primitive",
        )
        with TemporaryDirectory() as tmp:
            _, path = self._phase1_ckpt(tmp)
            with self.assertRaises(ValueError) as ctx:
                _load_hf(dst, fwd, path)
        self.assertIn("should be assigned before", str(ctx.exception))
        self.assertIn("indexer", str(ctx.exception))

    def test_non_hf_flex_path_has_no_indexer_escape_hatch(self):
        """``indexer_init_from_scratch`` is read only inside
        ``_gen_aoa_config`` (``fleet_model/ernie5_v2/modeling.py:1052``), and
        that is called only from ``trainer.py:1251`` under ``load_from_hf`` (and
        ``:1826`` under auto-parallel + ``convert_from_hf``). The plain
        ``flex_checkpoint`` resume path (``trainer.py:1333-1351``) passes
        ``args.aoa_config`` instead, so the switch cannot help there: a
        phase-1 checkpoint fails outright unless the YAML supplies its own
        ``_ -> key`` statements.
        """
        src = _attn(_MHA, seed=11)
        dst = _attn(_DSA, seed=99)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "flex"
            path.mkdir()
            with _single_process_world():
                paddle.distributed.save_state_dict(
                    src.sharded_state_dict(PREFIX), str(path)
                )
                self.assertTrue(list(path.glob("*.distcp")))
                with self.assertRaises(KeyError) as ctx:
                    paddle.distributed.load_state_dict(
                        dst.sharded_state_dict(PREFIX), str(path)
                    )
        self.assertIn("indexer", str(ctx.exception))


@_requires_cuda
class TestAddPrimitiveDiscardsTrainedIndexer(unittest.TestCase):
    """Q4: ``_ -> key`` drops a checkpoint tensor of the same name.

    This is the phase-2 -> phase-3 restart footgun: the YAML that produced the
    checkpoint has ``indexer_init_from_scratch: true``, and re-using it to
    resume silently re-initialises the indexer it just trained.
    """

    def _full_ckpt_then_load(self, tmp, init_from_scratch):
        src = _attn(_DSA, seed=11)
        dst = _attn(_DSA, seed=99)
        keys = {PREFIX + k for k in src.state_dict()}
        forward, inverse = _aoa(_DSA, init_from_scratch)
        path = Path(tmp) / "phase2"
        _save_hf(src, _filter(inverse, keys, fleet_on_left=True), path)
        disk = _on_disk(path)
        for key in _INDEXER_KEYS:
            self.assertIn(
                PREFIX + key, disk, "the phase-2 checkpoint must be complete"
            )
        own = {
            k: v.clone() for k, v in dst.state_dict().items() if "indexer" in k
        }
        # ANTI-VACUITY: the two indexers must really differ, or "the value was
        # dropped" and "the value was loaded" are the same observation.
        self.assertEqual(
            sorted(_diff_keys(own, {k: src.state_dict()[k] for k in own})),
            sorted(_INDEXER_RANDOM_KEYS),
        )
        _load_hf(dst, _filter(forward, keys, fleet_on_left=False), path)
        return src, dst, own

    def test_documented_hazard_scratch_true_discards_the_saved_indexer(self):
        with TemporaryDirectory() as tmp:
            src, dst, own = self._full_ckpt_then_load(tmp, True)
        after = dst.state_dict()
        # Not loaded: still bit-identical to what this module built itself ...
        self.assertEqual(_diff_keys(after, own), [])
        # ... and still different from the trained values on disk.
        self.assertEqual(
            sorted(
                _diff_keys(
                    after, {k: src.state_dict()[k] for k in _INDEXER_KEYS}
                )
            ),
            sorted(_INDEXER_RANDOM_KEYS),
        )
        # Everything outside the indexer did load, so the discard is specific
        # to the add primitive rather than a broken load.
        backbone = {
            k: v for k, v in src.state_dict().items() if "indexer" not in k
        }
        self.assertEqual(_diff_keys(after, backbone), [])

    def test_scratch_false_loads_the_saved_indexer(self):
        with TemporaryDirectory() as tmp:
            src, dst, _ = self._full_ckpt_then_load(tmp, False)
        self.assertEqual(_diff_keys(dst.state_dict(), src.state_dict()), [])

    @unittest.expectedFailure
    def test_add_primitive_should_not_discard_an_existing_checkpoint_tensor(
        self,
    ):
        """The behaviour we would want: ``_ -> key`` means "initialise this if
        the checkpoint has nothing", not "ignore what the checkpoint has".
        Either the engine should prefer the checkpoint tensor, or
        ``_gen_aoa_config`` should emit the add primitive only for keys absent
        from the checkpoint. Pinned as an expected failure so that fixing it
        turns this file red instead of leaving the hazard test as the only
        record.
        """
        with TemporaryDirectory() as tmp:
            src, dst, _ = self._full_ckpt_then_load(tmp, True)
        self.assertEqual(
            _diff_keys(
                dst.state_dict(),
                {k: src.state_dict()[k] for k in _INDEXER_KEYS},
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
