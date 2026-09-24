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

"""Behavior tests for the DPO training workflow.

Module under test: ``paddlefleet.cli.train.dpo.workflow``. A training
``workflow`` belongs to the "Trainer 训练引擎" layer of the repository module
map: it selects the task, normalizes/validates the training arguments,
assembles the model + data, and launches the Trainer. ``workflow.py`` exposes a
single public entry, ``run_dpo``; these tests drive that *real* entry and
observe the argument-normalization and validation decisions it makes before any
heavy model construction, instead of re-implementing the branch logic inside
the test (the anti-pattern the replaced coverage test exhibited).

Isolation follows the not-under-test boundary: the seed/device preamble
(``paddle.set_device``, ``set_random_seed``, ``set_seed``) is infrastructure,
not the behaviour being verified, so it is replaced with no-ops. The code under
test -- the ``AttentionInterface`` membership gate, the loss-type
normalization, and the pipeline-parallel guard -- is never mocked. Expected
values are hand-derived from the production source and from
``paddlefleet/nn/attention/interface.py`` (the real attention registry contains
exactly ``eager`` / ``sdpa`` / ``flashmask``).

Controlled stop: after loss-type normalization mutates ``training_args`` in
place, the next production statement is the pipeline-parallel guard
``assert clear_every_step_cache`` (only when ``pipeline_model_parallel_size >
1``). Setting ``pipeline_model_parallel_size = 2`` with no
``clear_every_step_cache`` makes ``run_dpo`` raise ``AssertionError`` at that
real guard, bounding execution right after the normalization so the mutation
can be inspected. The ``AssertionError`` is the intended stop; the real
verification is the observed mutation -- a broken normalization would leave the
args unchanged and fail the assertions even though the guard still fires.

These tests run on CPU. Importing the production module pulls ``paddlefleet``,
which imports Paddle at import time; when Paddle (or another hard dependency) is
absent the import raises ``ImportError`` and every test skips with a recorded
reason -- never a silent pass. No production code is modified by this file.
"""

import types
import unittest
from unittest import mock

try:
    from paddlefleet.cli.train.dpo import workflow
    from paddlefleet.cli.train.dpo.workflow import run_dpo

    _IMPORT_ERROR = None
except ImportError as exc:  # Paddle / paddlefleet not installed in this env.
    workflow = None
    run_dpo = None
    _IMPORT_ERROR = exc


# Hand-copied from ``paddlefleet/nn/attention/interface.py`` for an independent
# expectation of what the real attention registry accepts.
_REAL_ATTN_IMPLS = ("eager", "sdpa", "flashmask")


def _make_model_args(**overrides):
    defaults = {
        "model_name_or_path": "dummy/model",
        "download_hub": "huggingface",
        "copy_custom_file_list": None,
        "_attn_implementation": "eager",
        "lora": False,
    }
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _make_training_args(**overrides):
    # ``clear_every_step_cache`` is intentionally absent by default so the
    # pipeline-parallel guard fires when ``pipeline_model_parallel_size > 1``.
    defaults = {
        "device": "cpu",
        "seed": 42,
        "loss_type": "sigmoid",
        "reference_free": False,
        "sft_loss_ratio": 0.0,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": False,
    }
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


class _RunDPOTestBase(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(
                "paddlefleet.cli.train.dpo.workflow import failed "
                f"(dependency unavailable): {_IMPORT_ERROR!r}"
            )
        # Isolate the not-under-test seed/device preamble so the assertions
        # target only the workflow's own validation/normalization logic.
        for target, attr in (
            (workflow, "set_random_seed"),
            (workflow, "set_seed"),
            (workflow.paddle, "set_device"),
        ):
            patcher = mock.patch.object(target, attr, lambda *a, **k: None)
            patcher.start()
            self.addCleanup(patcher.stop)


class TestAttnImplementationValidation(_RunDPOTestBase):
    """The real AttentionInterface membership gate inside run_dpo."""

    def test_invalid_attn_implementation_raises_valueerror(self):
        model_args = _make_model_args(_attn_implementation="not_a_real_impl")
        training_args = _make_training_args()
        with self.assertRaises(ValueError) as ctx:
            run_dpo(model_args, None, None, training_args)
        msg = str(ctx.exception)
        # The rejected value and the *real* registered impls are surfaced,
        # proving run_dpo consulted the actual AttentionInterface mapping and
        # not a set reconstructed inside the test.
        self.assertIn("not_a_real_impl", msg)
        for impl in _REAL_ATTN_IMPLS:
            self.assertIn(impl, msg)

    def test_valid_attn_implementation_passes_gate(self):
        # "eager" is registered, so the attn gate must NOT raise ValueError;
        # execution instead reaches the pipeline-parallel guard and stops with
        # AssertionError. The benign loss_type is left unnormalized, confirming
        # the loss branch is conditional rather than unconditional.
        model_args = _make_model_args(_attn_implementation="eager")
        training_args = _make_training_args(
            loss_type="sigmoid",
            reference_free=False,
            pipeline_model_parallel_size=2,
        )
        with self.assertRaises(AssertionError):
            run_dpo(model_args, None, None, training_args)
        self.assertEqual(training_args.loss_type, "sigmoid")
        self.assertFalse(training_args.reference_free)


class TestLossTypeNormalization(_RunDPOTestBase):
    """run_dpo rewrites certain loss_types before building the Trainer.

    The pipeline-parallel guard (pp size 2, no clear_every_step_cache) is the
    controlled stop; the in-place mutation observed afterwards is the real
    verification.
    """

    def _run_until_pp_guard(self, training_args):
        model_args = _make_model_args(_attn_implementation="eager")
        with self.assertRaises(AssertionError):
            run_dpo(model_args, None, None, training_args)

    def test_orpo_is_rewritten_to_or_reference_free(self):
        training_args = _make_training_args(
            loss_type="orpo",
            reference_free=False,
            sft_loss_ratio=0.0,
            pipeline_model_parallel_size=2,
        )
        self._run_until_pp_guard(training_args)
        # Hand-derived from run_dpo: orpo == sft_loss + pref_ratio * or_loss, so
        # it expands into reference-free "or" with full (1.0) sft weight.
        self.assertTrue(training_args.reference_free)
        self.assertEqual(training_args.loss_type, "or")
        self.assertEqual(training_args.sft_loss_ratio, 1.0)

    def test_or_forces_reference_free_without_touching_loss_type(self):
        training_args = _make_training_args(
            loss_type="or",
            reference_free=False,
            pipeline_model_parallel_size=2,
        )
        self._run_until_pp_guard(training_args)
        self.assertTrue(training_args.reference_free)
        self.assertEqual(training_args.loss_type, "or")

    def test_simpo_forces_reference_free_without_touching_loss_type(self):
        training_args = _make_training_args(
            loss_type="simpo",
            reference_free=False,
            pipeline_model_parallel_size=2,
        )
        self._run_until_pp_guard(training_args)
        self.assertTrue(training_args.reference_free)
        self.assertEqual(training_args.loss_type, "simpo")

    def test_sigmoid_loss_type_is_not_normalized(self):
        # Contrast: a loss_type outside {orpo, or, simpo} is left untouched, so
        # the normalization is genuinely conditional on the loss_type value.
        training_args = _make_training_args(
            loss_type="sigmoid",
            reference_free=False,
            sft_loss_ratio=0.25,
            pipeline_model_parallel_size=2,
        )
        self._run_until_pp_guard(training_args)
        self.assertFalse(training_args.reference_free)
        self.assertEqual(training_args.loss_type, "sigmoid")
        self.assertEqual(training_args.sft_loss_ratio, 0.25)


class TestPipelineParallelGuard(_RunDPOTestBase):
    """pp size > 1 requires clear_every_step_cache; run_dpo asserts this."""

    def test_pp_gt_1_without_clear_cache_raises_with_message(self):
        model_args = _make_model_args(_attn_implementation="eager")
        # No clear_every_step_cache attribute -> hasattr(...) is False.
        training_args = _make_training_args(pipeline_model_parallel_size=2)
        with self.assertRaises(AssertionError) as ctx:
            run_dpo(model_args, None, None, training_args)
        self.assertIn("clear_every_step_cache", str(ctx.exception))

    def test_pp_gt_1_with_clear_cache_passes_guard(self):
        # Contrast: satisfying the guard lets execution proceed past it. A
        # sentinel raised from the (not-under-test) print_config collaborator
        # bounds execution and proves the pp guard itself did not fire.
        class _Stop(Exception):
            pass

        def _raise_stop(*a, **k):
            raise _Stop

        model_args = _make_model_args(_attn_implementation="eager")
        training_args = _make_training_args(
            pipeline_model_parallel_size=2,
            clear_every_step_cache=True,
            sequence_parallel=False,
        )
        training_args.print_config = _raise_stop
        with self.assertRaises(_Stop):
            run_dpo(model_args, None, None, training_args)


if __name__ == "__main__":
    unittest.main()
