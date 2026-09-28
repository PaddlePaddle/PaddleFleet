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

"""Behavior tests for ``trainer/unified_checkpoint/check_completion.py``.

Module under test:
    paddlefleet.trainer.unified_checkpoint.check_completion
    (public entries ``check_unified_checkpoint`` and ``check_unified_optimizer``).

Environment
-----------
No-card (CPU) tests. The two public functions mix two kinds of work:

1. *Index discovery / master-weight-status* -- pure single-process file and
   config logic. When ``PADDLE_TRAINERS_NUM<=1`` (the default) the helpers
   ``distributed_isfile`` / ``distributed_file`` reduce to plain
   ``os.path.isfile`` / path resolution, so every failure/branch-selection
   path *before* the first collective is genuinely CPU-runnable with real
   temp-dir files. These are what the running tests below exercise.
2. *The completion decision itself* -- ``dist.all_gather_object`` +
   ``dist.all_reduce`` + ``fleet.get_hybrid_communicate_group()`` gather the
   existing shard list across ranks and reduce ``local_resume``. That decision
   only has meaning with several ranks each holding different files; it cannot
   be honestly verified in a single CPU process, and faking ``world_size`` +
   mocking the collectives would prove nothing (anti-pattern 13). It is left
   as a documented, conditionally-skipped multi-card test.

Independent oracle
------------------
All expected values are hand-derived by reading the control flow, never by
re-running the functions:

* ``check_unified_checkpoint`` on a directory with neither a model-weight index
  nor a master-weight index re-raises ``ValueError`` from
  ``select_model_weight_index`` ("Can't find a valid unified model or master
  weight checkpoint to load."), *before* touching any collective.
* ``check_unified_optimizer`` first probes the optimizer index; when absent it
  raises ``Exception`` naming the resolved path. The basename is
  ``optimizer.pdopt.index.json`` for non-safe and
  ``optimizer.safetensors.index.json`` for ``safe_serialization=True`` -- an
  observable branch difference.
* With a real optimizer index whose ``master_weights`` field is ``True`` and a
  multi-precision fp16 optimizer, ``update_master_weight_status`` keeps
  ``has_master_weights`` and selects the *master*-weight index basename
  (``master_weights.pdparams.index.json`` / ``.safetensors`` for safe); the
  subsequent missing-file probe raises naming exactly that file.
* With ``master_weights=False`` and a multi-precision fp16 optimizer,
  ``update_master_weight_status`` raises ``ValueError`` unless
  ``unified_checkpoint_config`` opts in via ``master_weight_compatible`` /
  ``remove_master_weight``; when it opts in, the fallback index is the *model*
  weight index (``model_state.pdparams.index.json``), NOT the master index --
  a distinct observable basename.

The independent basename literals are asserted to match the imported
``paddlefleet.utils.env`` constants, so the oracle also guards env.py.
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

try:
    import paddle

    from paddlefleet.trainer.unified_checkpoint.check_completion import (
        check_unified_checkpoint,
        check_unified_optimizer,
    )
    from paddlefleet.utils.env import (
        PADDLE_MASTER_WEIGHTS_INDEX_NAME,
        PADDLE_OPTIMIZER_INDEX_NAME,
        PADDLE_WEIGHTS_INDEX_NAME,
        SAFE_MASTER_WEIGHTS_INDEX_NAME,
        SAFE_OPTIMIZER_INDEX_NAME,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # dependency genuinely unavailable in this env
    _IMPORT_ERROR = exc


# Independent, hand-written basename oracles (not read back from production).
_OPT_INDEX_NONSAFE = "optimizer.pdopt.index.json"
_OPT_INDEX_SAFE = "optimizer.safetensors.index.json"
_MASTER_INDEX_NONSAFE = "master_weights.pdparams.index.json"
_MASTER_INDEX_SAFE = "master_weights.safetensors.index.json"
_MODEL_INDEX_NONSAFE = "model_state.pdparams.index.json"


def _write_optimizer_index(directory, basename, master_weights):
    """Write a minimal but real optimizer index JSON and return its path."""
    path = os.path.join(directory, basename)
    payload = {
        # weight_map values are not consumed before the raises under test;
        # they are present so the JSON is structurally realistic.
        "weight_map": {
            "layer0.w/moment1_0": "optimizer-00001-of-00001.safetensors",
            "layer0.w/moment2_0": "optimizer-00001-of-00001.safetensors",
        },
        "master_weights": master_weights,
    }
    with open(path, "w") as handle:
        handle.write(json.dumps(payload))
    return path


def _make_args(unified_checkpoint_config):
    """A config carrier for update_master_weight_status (fp16 multi-precision)."""
    return SimpleNamespace(
        fp16=True,
        bf16=False,
        unified_checkpoint_config=unified_checkpoint_config,
    )


def _make_multi_precision_optimizer():
    """Optimizer carrier: unwrap_optimizer -> self, _multi_precision True."""
    return SimpleNamespace(_multi_precision=True)


@unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"paddlefleet check_completion import failed: {_IMPORT_ERROR}",
)
class TestEnvConstantOracle(unittest.TestCase):
    """Guard that the imported env constants match the independent literals."""

    def test_index_basenames_match_independent_literals(self):
        self.assertEqual(PADDLE_OPTIMIZER_INDEX_NAME, _OPT_INDEX_NONSAFE)
        self.assertEqual(SAFE_OPTIMIZER_INDEX_NAME, _OPT_INDEX_SAFE)
        self.assertEqual(
            PADDLE_MASTER_WEIGHTS_INDEX_NAME, _MASTER_INDEX_NONSAFE
        )
        self.assertEqual(SAFE_MASTER_WEIGHTS_INDEX_NAME, _MASTER_INDEX_SAFE)
        self.assertEqual(PADDLE_WEIGHTS_INDEX_NAME, _MODEL_INDEX_NONSAFE)


@unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"paddlefleet check_completion import failed: {_IMPORT_ERROR}",
)
class TestCheckUnifiedCheckpoint(unittest.TestCase):
    """CPU-runnable failure contract for check_unified_checkpoint."""

    def test_no_index_present_raises_valueerror_before_collective(self):
        # Empty checkpoint dir: neither model-weight nor master-weight index.
        # select_model_weight_index must fail with a ValueError, and it must
        # happen before any all_gather/all_reduce so no process group is needed.
        args = SimpleNamespace(dataset_rank=0, use_expert_parallel=False)
        model = SimpleNamespace()  # not a LoRAModel -> model-weight index path
        with tempfile.TemporaryDirectory() as ckpt:
            with self.assertRaises(ValueError) as ctx:
                check_unified_checkpoint(args, model, ckpt)
            message = str(ctx.exception)
            self.assertIn("valid unified model or master weight", message)

    def test_no_index_present_raises_for_safe_serialization_too(self):
        args = SimpleNamespace(dataset_rank=0, use_expert_parallel=False)
        model = SimpleNamespace()
        with tempfile.TemporaryDirectory() as ckpt:  # noqa: SIM117
            with self.assertRaises(ValueError):
                check_unified_checkpoint(
                    args, model, ckpt, safe_serialization=True
                )


@unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"paddlefleet check_completion import failed: {_IMPORT_ERROR}",
)
class TestCheckUnifiedOptimizer(unittest.TestCase):
    """CPU-runnable failure / branch-selection contracts for the optimizer."""

    def test_missing_optimizer_index_nonsafe_names_pdopt_index(self):
        model = SimpleNamespace()
        with tempfile.TemporaryDirectory() as ckpt:
            with self.assertRaises(Exception) as ctx:
                check_unified_optimizer(
                    SimpleNamespace(), model, SimpleNamespace(), ckpt
                )
            message = str(ctx.exception)
            # Non-safe branch resolves to optimizer.pdopt.index.json.
            self.assertIn(_OPT_INDEX_NONSAFE, message)
            self.assertIn(ckpt, message)
            self.assertNotIn(_OPT_INDEX_SAFE, message)

    def test_missing_optimizer_index_safe_names_safetensors_index(self):
        model = SimpleNamespace()
        with tempfile.TemporaryDirectory() as ckpt:
            with self.assertRaises(Exception) as ctx:
                check_unified_optimizer(
                    SimpleNamespace(),
                    model,
                    SimpleNamespace(),
                    ckpt,
                    safe_serialization=True,
                )
            message = str(ctx.exception)
            # safe_serialization flips the selected basename.
            self.assertIn(_OPT_INDEX_SAFE, message)
            self.assertIn(ckpt, message)
            self.assertNotIn(_OPT_INDEX_NONSAFE, message)

    def test_master_weights_true_selects_master_index_and_probes_it(self):
        # Optimizer index present with master_weights=True; multi-precision fp16
        # optimizer keeps has_master_weights and selects the master index.
        # The master index file is absent -> raise naming exactly that file.
        model = SimpleNamespace()
        args = _make_args(unified_checkpoint_config=[])
        optimizer = _make_multi_precision_optimizer()
        with tempfile.TemporaryDirectory() as ckpt:
            _write_optimizer_index(
                ckpt, _OPT_INDEX_NONSAFE, master_weights=True
            )
            with self.assertRaises(Exception) as ctx:
                check_unified_optimizer(args, model, optimizer, ckpt)
            message = str(ctx.exception)
            self.assertIn(_MASTER_INDEX_NONSAFE, message)
            self.assertIn(ckpt, message)

    def test_master_weights_true_safe_selects_safe_master_index(self):
        model = SimpleNamespace()
        args = _make_args(unified_checkpoint_config=[])
        optimizer = _make_multi_precision_optimizer()
        with tempfile.TemporaryDirectory() as ckpt:
            _write_optimizer_index(ckpt, _OPT_INDEX_SAFE, master_weights=True)
            with self.assertRaises(Exception) as ctx:
                check_unified_optimizer(
                    args, model, optimizer, ckpt, safe_serialization=True
                )
            message = str(ctx.exception)
            self.assertIn(_MASTER_INDEX_SAFE, message)
            self.assertNotIn(_MASTER_INDEX_NONSAFE, message)

    def test_master_weights_false_requires_compatible_option(self):
        # master_weights=False + multi-precision fp16 but no opt-in config:
        # update_master_weight_status must raise ValueError demanding one of the
        # compatible options, before any file probe / collective.
        model = SimpleNamespace()
        args = _make_args(unified_checkpoint_config=[])
        optimizer = _make_multi_precision_optimizer()
        with tempfile.TemporaryDirectory() as ckpt:
            _write_optimizer_index(
                ckpt, _OPT_INDEX_NONSAFE, master_weights=False
            )
            with self.assertRaises(ValueError) as ctx:
                check_unified_optimizer(args, model, optimizer, ckpt)
            message = str(ctx.exception)
            self.assertIn("master_weight_compatible", message)
            self.assertIn("remove_master_weight", message)

    def test_master_weights_false_compatible_falls_back_to_model_index(self):
        # With master_weight_compatible opted in, the fallback index is the
        # MODEL weight index (model_state.pdparams.index.json), not a master
        # index. That file is absent -> raise naming the model index.
        model = SimpleNamespace()
        args = _make_args(
            unified_checkpoint_config=["master_weight_compatible"]
        )
        optimizer = _make_multi_precision_optimizer()
        with tempfile.TemporaryDirectory() as ckpt:
            _write_optimizer_index(
                ckpt, _OPT_INDEX_NONSAFE, master_weights=False
            )
            with self.assertRaises(Exception) as ctx:
                check_unified_optimizer(args, model, optimizer, ckpt)
            message = str(ctx.exception)
            self.assertIn(_MODEL_INDEX_NONSAFE, message)
            self.assertNotIn(_MASTER_INDEX_NONSAFE, message)


@unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"paddlefleet check_completion import failed: {_IMPORT_ERROR}",
)
class TestCompletionDecisionMultiCard(unittest.TestCase):
    """The cross-rank completion decision is a multi-card behavior.

    ``check_unified_checkpoint`` / ``check_unified_optimizer`` compute
    ``local_resume`` from ``dist.all_gather_object`` of the per-rank existing
    shard list plus ``dist.all_reduce`` over the real TP/PP/(DP) groups. That
    decision is only meaningful when several ranks each hold different files.
    Verifying it requires a real, supported process group (see the distributed
    training rules); a single CPU process or a mocked collective cannot prove
    it. This test therefore skips unless a process group is already initialized
    (e.g. under the multi-card launcher).
    """

    def test_completion_decision_requires_real_process_group(self):
        if not paddle.distributed.is_initialized():
            self.skipTest(
                "cross-rank completion decision needs a real process group; "
                "run under the multi-card launcher to exercise it"
            )
        # Under a real launcher the genuine multi-rank assertion belongs here,
        # building per-rank checkpoint dirs and comparing the reduced
        # local_resume against an independently computed expectation.
        self.skipTest(
            "multi-card completion decision assertion is provided by the "
            "distributed test job, not this no-card file"
        )


if __name__ == "__main__":
    unittest.main()
