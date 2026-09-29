# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Behavior tests for trainer/unified_checkpoint/load_local.py.

Scope: the CPU-observable control logic of the two public entries,
``load_unified_checkpoint_locally`` and ``load_unified_optimizer_locally``.
Both live in the "Checkpoint / weight management" module, so the oracle
targets key mapping and ownership -- which struct-name maps to which static
name, and how the master-weight distinction rewrites keys -- rather than mere
file existence.

Real vs. stubbed:
  * The index selection + shard-metadata entries stay REAL: temp-dir JSON is
    written and read back by the production ``select_model_weight_index``,
    ``get_checkpoint_shard_files`` and ``get_optimizer_shard_files`` (and the
    real ``is_sharding_split_param_mode`` / ``nested_copy``).
  * Only NON-under-test collaborators that would otherwise demand a real
    ``PretrainedModel`` / ``fleet`` process group / on-disk safetensors are
    isolated: ``get_expected_state_dict`` (model weights), ``get_expected_keys``
    (Fleet), ``update_master_weight_status`` (optimizer probing) and
    ``load_state_dict`` (tensor I/O). The rename / missing-key logic under test
    is never patched; it runs for real against the fed inputs.

Independent oracle: every expected key string, ownership pairing and dtype is
hand-derived from the source semantics, NOT produced by calling the routine a
second time. The ``fp32_master_0`` literals are written out by hand instead of
importing the production constant.

Environment: no-card. paddle is imported only for the optimizer-rename tensor
work; when paddle/paddlefleet is absent the whole module skips (import guard),
which is "not run", not "not applicable". No real persistence or cross-rank
reshard is claimed here.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    import paddle

    from paddlefleet.trainer.unified_checkpoint.load_local import (
        load_unified_checkpoint_locally,
        load_unified_optimizer_locally,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet not installed on this runner
    paddle = None
    _IMPORT_ERROR = exc

PKG = "paddlefleet.trainer.unified_checkpoint.load_local"

# Env-constant filenames the production code selects (see utils/env.py). Written
# out literally so the fixture and the assertion do not borrow the constant the
# code under test uses to pick a filename.
MODEL_INDEX = "model_state.pdparams.index.json"
PADDLE_OPT_INDEX = "optimizer.pdopt.index.json"
SAFE_OPT_INDEX = "optimizer.safetensors.index.json"
SAFE_MW_INDEX = "master_weights.safetensors.index.json"


class _StubParam:
    """Stands in for a model_state_dict VALUE on the missing-key raise path.

    Only ``no_sync`` is read there (via getattr), so no paddle tensor is
    needed until the loop that this path never reaches.
    """

    def __init__(self, no_sync=False):
        self.no_sync = no_sync


class _StubConfig:
    def __init__(self, tp=1):
        self.tensor_model_parallel_size = tp


class _StubModel:
    def __init__(self, tp=1):
        self.config = _StubConfig(tp)


class _StubArgs:
    def __init__(
        self,
        use_expert_parallel=False,
        data_parallel_rank=0,
        sharding_parallel_size=1,
    ):
        self.use_expert_parallel = use_expert_parallel
        self.data_parallel_rank = data_parallel_rank
        self.sharding_parallel_size = sharding_parallel_size


class _StubOptimizer:
    def state_dict(self):
        return {}


def _write_index(dir_path, filename, weight_map, extra=None):
    """Write a real sharded-checkpoint index JSON to ``dir_path``."""
    index = {"metadata": {}, "weight_map": weight_map}
    if extra:
        index.update(extra)
    with open(os.path.join(dir_path, filename), "w") as handle:
        json.dump(index, handle)


@unittest.skipIf(
    _IMPORT_ERROR is not None, f"paddlefleet import failed: {_IMPORT_ERROR}"
)
class TestLoadUnifiedCheckpointExpectedKeys(unittest.TestCase):
    """load_unified_checkpoint_locally: expected-key derivation + missing-key guard.

    The checkpoint index is real (temp JSON read by the real
    ``select_model_weight_index`` + ``get_checkpoint_shard_files``); only
    ``get_expected_state_dict`` is isolated so we can pin exactly which model
    params are expected without building a real PretrainedModel. Because the
    engineered ``missing_keys`` is always non-empty, the routine raises before
    the shard-loading loop, so the VALUES only need a ``no_sync`` flag.
    """

    def setUp(self):
        self.ckpt = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.ckpt, ignore_errors=True)
        # Checkpoint owns only "z.weight"; the model wants a.weight/b.weight,
        # so both are missing. "a.weight" is a no_sync (expert) param.
        _write_index(
            self.ckpt,
            MODEL_INDEX,
            {"z.weight": "model-00001-of-00001.pdparams"},
        )
        self.model_sd = {
            "a.weight": _StubParam(no_sync=True),
            "b.weight": _StubParam(no_sync=False),
        }

    def _call(self, args):
        with patch(
            f"{PKG}.get_expected_state_dict", return_value=self.model_sd
        ):
            load_unified_checkpoint_locally(args, _StubModel(), self.ckpt)

    def test_non_expert_parallel_expects_all_model_keys(self):
        # Hand oracle: expected = {a.weight, b.weight}; loaded = {z.weight};
        # missing = {a.weight, b.weight} -> both named in the ValueError.
        with self.assertRaises(ValueError) as ctx:
            self._call(_StubArgs(use_expert_parallel=False))
        msg = str(ctx.exception)
        self.assertIn("a.weight", msg)
        self.assertIn("b.weight", msg)

    def test_expert_parallel_dp_rank0_still_expects_all_keys(self):
        # dp_rank == 0 does NOT filter, so it matches the non-EP set.
        with self.assertRaises(ValueError) as ctx:
            self._call(
                _StubArgs(use_expert_parallel=True, data_parallel_rank=0)
            )
        msg = str(ctx.exception)
        self.assertIn("a.weight", msg)
        self.assertIn("b.weight", msg)

    def test_expert_parallel_dp_rank_positive_keeps_only_no_sync(self):
        # dp_rank > 0 narrows expected_keys to no_sync params only, so the
        # regular "b.weight" is NOT expected and must not be reported missing;
        # only the no_sync "a.weight" remains. A branch that forgot to filter
        # would wrongly report "b.weight" here.
        with self.assertRaises(ValueError) as ctx:
            self._call(
                _StubArgs(use_expert_parallel=True, data_parallel_rank=1)
            )
        msg = str(ctx.exception)
        self.assertIn("a.weight", msg)
        self.assertNotIn("b.weight", msg)


@unittest.skipIf(
    _IMPORT_ERROR is not None, f"paddlefleet import failed: {_IMPORT_ERROR}"
)
class TestLoadUnifiedOptimizerIndexSelection(unittest.TestCase):
    """load_unified_optimizer_locally selects the optimizer index by safe flag.

    The first thing the routine does is ``open`` the index for the chosen
    serialization. With an empty checkpoint dir the open fails, and the
    filename inside the FileNotFoundError reveals which constant the branch
    picked -- a real contract (safe vs. paddle serialization -> different
    index file), verified without any tensor work.
    """

    def setUp(self):
        self.ckpt = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.ckpt, ignore_errors=True)

    def test_non_safe_serialization_opens_pdopt_index(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            load_unified_optimizer_locally(
                _StubArgs(),
                _StubModel(),
                _StubOptimizer(),
                self.ckpt,
                safe_serialization=False,
            )
        self.assertIn(PADDLE_OPT_INDEX, str(ctx.exception))
        self.assertNotIn(SAFE_OPT_INDEX, str(ctx.exception))

    def test_safe_serialization_opens_safetensors_index(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            load_unified_optimizer_locally(
                _StubArgs(),
                _StubModel(),
                _StubOptimizer(),
                self.ckpt,
                safe_serialization=True,
            )
        self.assertIn(SAFE_OPT_INDEX, str(ctx.exception))
        self.assertNotIn(PADDLE_OPT_INDEX, str(ctx.exception))


@unittest.skipIf(
    _IMPORT_ERROR is not None, f"paddlefleet import failed: {_IMPORT_ERROR}"
)
class TestLoadUnifiedOptimizerKeyRename(unittest.TestCase):
    """load_unified_optimizer_locally rewrites optimizer keys to static names.

    The rename loop maps ``<struct>/<typename>`` -> ``<static>_<typename>``
    (and inserts ``fp32_master_0`` for non-fp32 params when master weights are
    present). This is checkpoint key-mapping + ownership: struct "a" must map
    to its OWN static name, and each rewritten tensor's ``.name`` must equal
    its new key. The shard-metadata JSON is real; only the Fleet/model/tensor-
    I/O collaborators are isolated so the rename logic itself runs unmodified.
    """

    def _model_sd(self, dtypes):
        # struct name -> tensor whose .name is the static name and whose dtype
        # decides fp32_master insertion. Static names differ from struct names
        # so a swapped mapping is detectable.
        sd = {}
        static = {"a": "sa", "b": "sb"}
        for struct, dt in dtypes.items():
            t = paddle.zeros([2], dtype=dt)
            t.name = static[struct]
            sd[struct] = t
        return sd

    def test_rename_without_master_weights(self):
        ckpt = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, ckpt, ignore_errors=True)
        shard = "optimizer-00001-of-00001.safetensors"
        opt_keys = {
            "a/moment1_0": shard,
            "a/beta1_pow_acc_0": shard,
            "b/moment2_0": shard,
        }
        _write_index(ckpt, SAFE_OPT_INDEX, opt_keys)

        # Distinguishable tensor contents to confirm ownership survives rename.
        loaded = {
            "a/moment1_0": paddle.to_tensor([1.0, 2.0]),
            "a/beta1_pow_acc_0": paddle.to_tensor([3.0, 4.0]),
            "b/moment2_0": paddle.to_tensor([5.0, 6.0]),
        }
        model_sd = self._model_sd({"a": "float32", "b": "float32"})

        with (
            patch(f"{PKG}.get_expected_state_dict", return_value=model_sd),
            patch(f"{PKG}.get_expected_keys", return_value=set(opt_keys)),
            patch(
                f"{PKG}.update_master_weight_status", return_value=(False, None)
            ),
            patch(f"{PKG}.load_state_dict", return_value=loaded),
        ):
            out = load_unified_optimizer_locally(
                _StubArgs(),
                _StubModel(),
                _StubOptimizer(),
                ckpt,
                safe_serialization=True,
            )

        # Hand-derived expected keys: "<static>_<typename>", no fp32_master.
        self.assertEqual(
            set(out),
            {"sa_moment1_0", "sa_beta1_pow_acc_0", "sb_moment2_0"},
        )
        # Ownership + content: b's moment2 landed under sb, not sa.
        np.testing.assert_array_equal(
            out["sb_moment2_0"].numpy(), np.array([5.0, 6.0], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            out["sa_moment1_0"].numpy(), np.array([1.0, 2.0], dtype=np.float32)
        )
        # Each rewritten tensor carries its new key as .name.
        for key in ("sa_moment1_0", "sa_beta1_pow_acc_0", "sb_moment2_0"):
            self.assertEqual(out[key].name, key)

    def test_rename_with_master_weights_inserts_fp32_master(self):
        ckpt = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, ckpt, ignore_errors=True)
        opt_shard = "optimizer-00001-of-00001.safetensors"
        mw_shard = "master_weights-00001-of-00001.safetensors"
        opt_keys = {"a/moment1_0": opt_shard, "b/moment1_0": opt_shard}
        mw_keys = {"a": mw_shard, "b": mw_shard}
        _write_index(ckpt, SAFE_OPT_INDEX, opt_keys)
        _write_index(ckpt, SAFE_MW_INDEX, mw_keys)

        # "a" is bf16 (non-fp32) -> its optimizer key gets fp32_master_0
        # inserted; "b" is fp32 -> no insertion.
        model_sd = self._model_sd({"a": "bfloat16", "b": "float32"})
        loaded_optim = {
            "a/moment1_0": paddle.to_tensor([1.0, 2.0]),
            "b/moment1_0": paddle.to_tensor([3.0, 4.0]),
        }
        # Master weight tensors keyed by struct name; the bf16 one must be
        # cast to fp32 on restore.
        loaded_mw = {
            "a": paddle.zeros([2], dtype="bfloat16"),
            "b": paddle.zeros([2], dtype="float32"),
        }

        # get_expected_keys is called twice (optimizer, then master weights).
        with (
            patch(f"{PKG}.get_expected_state_dict", return_value=model_sd),
            patch(
                f"{PKG}.get_expected_keys",
                side_effect=[set(opt_keys), set(mw_keys)],
            ),
            patch(
                f"{PKG}.update_master_weight_status",
                return_value=(True, SAFE_MW_INDEX),
            ),
            patch(
                f"{PKG}.load_state_dict", side_effect=[loaded_optim, loaded_mw]
            ),
        ):
            out = load_unified_optimizer_locally(
                _StubArgs(),
                _StubModel(),
                _StubOptimizer(),
                ckpt,
                safe_serialization=True,
            )

        # Optimizer keys: bf16 "a" -> fp32_master inserted; fp32 "b" -> plain.
        self.assertIn("sa_fp32_master_0_moment1_0", out)
        self.assertIn("sb_moment1_0", out)
        self.assertNotIn("sa_moment1_0", out)
        self.assertEqual(
            out["sa_fp32_master_0_moment1_0"].name,
            "sa_fp32_master_0_moment1_0",
        )
        self.assertEqual(out["sb_moment1_0"].name, "sb_moment1_0")

        # Master weights land under static names and are cast to fp32.
        self.assertEqual(set(out["master_weights"]), {"sa", "sb"})
        self.assertEqual(out["master_weights"]["sa"].dtype, paddle.float32)
        self.assertEqual(out["master_weights"]["sb"].dtype, paddle.float32)
        self.assertEqual(out["master_weights"]["sa"].name, "sa_fp32_master_0")
        self.assertEqual(out["master_weights"]["sb"].name, "sb_fp32_master_0")


if __name__ == "__main__":
    unittest.main()
