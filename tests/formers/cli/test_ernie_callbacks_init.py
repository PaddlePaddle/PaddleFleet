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

"""Behavior tests for the ernie_pretrain callbacks package.

The production package re-exports nine callbacks from
``paddlefleet.cli.train.ernie_pretrain.src.callbacks``. Rather than asserting
that each symbol is merely importable/non-None (which passes even if the wrong
object is bound to a name), these tests drive real callback methods and check
the observable side effects against hand-derived expected values:

  * ``__all__`` names each resolve to the callback *class* of the same name
    (identity, not existence), so a mis-wired re-export is caught.
  * ``GCCallback`` toggles the garbage collector and calls ``gc.collect`` only
    on steps that are exact multiples of ``gc_interval`` (and never when the
    interval is disabled).
  * ``LoggingCallback.on_log`` strips ``total_flos``, forwards the remaining
    logs to a ``metrics_dumper``, and formats scalars with the documented
    per-key rules.

Paddle is an optional heavy dependency and the production package imports it at
top level (via ``paddlefleet.trainer.trainer_callback``); every test therefore
skips with an honest reason when that import fails. The local CI env has no
paddle installed, so these will skip there rather than falsely pass.
"""

import gc
import logging
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    from paddlefleet.cli.train.ernie_pretrain.src import callbacks as cb_pkg
    from paddlefleet.cli.train.ernie_pretrain.src.callbacks import (
        GCCallback,
        LoggingCallback,
    )
    from paddlefleet.cli.train.ernie_pretrain.src.callbacks import gc_callback

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle (or the package) unavailable in this env
    cb_pkg = None
    GCCallback = None
    LoggingCallback = None
    gc_callback = None
    _IMPORT_ERROR = exc


# Expected mapping of public export name -> the module the class lives in and
# the attribute path used to resolve the canonical class independently of the
# package __init__ re-export. Derived by reading the callback source modules,
# not from the package's own __all__/attributes.
_EXPECTED_EXPORTS = {
    "FP8QuantWeightCallback": "fp8_quant_weight_callback",
    "GCCallback": "gc_callback",
    "LoggingCallback": "logging_callback",
    "MoECorrectionBiasAdjustCallback": "moe_correction_bias_adjust_callback",
    "GlobalRNGCallback": "moe_logging_callback",
    "MoeLoggingCallback": "moe_logging_callback",
    "OrthogonalCallback": "ortho_loss_callback",
    "SPGradSyncCallback": "sp_grad_sync_callback",
    "TensorBoardCallback": "tensorboard_callback",
}


def _skip_if_no_paddle(test_case):
    if _IMPORT_ERROR is not None:
        test_case.skipTest(
            f"paddle/ernie callbacks import unavailable: {_IMPORT_ERROR!r}"
        )


class CallbacksPackageExportTest(unittest.TestCase):
    """The package must re-export exactly the documented callback classes,
    and each name must resolve to the class defined in its source module."""

    def setUp(self):
        _skip_if_no_paddle(self)

    def test_all_lists_exact_export_set(self):
        # __all__ contents must match the known export set (no missing / extra
        # names). assertCountEqual treats it as a multiset so a duplicated or
        # dropped entry fails.
        self.assertCountEqual(list(cb_pkg.__all__), _EXPECTED_EXPORTS.keys())

    def test_each_export_is_its_source_class(self):
        import importlib

        for name, submodule in _EXPECTED_EXPORTS.items():
            reexported = getattr(cb_pkg, name)
            source_mod = importlib.import_module(
                "paddlefleet.cli.train.ernie_pretrain.src.callbacks."
                + submodule
            )
            canonical = getattr(source_mod, name)
            # Identity, not mere existence: the re-export must be the same
            # class object defined in the source module.
            self.assertIs(reexported, canonical)
            self.assertIsInstance(reexported, type)
            self.assertEqual(reexported.__name__, name)


class GCCallbackBehaviorTest(unittest.TestCase):
    """GCCallback controls the cyclic garbage collector based on gc_interval
    and the current global step."""

    def setUp(self):
        _skip_if_no_paddle(self)
        # on_train_begin may flip the process-wide GC enabled flag; restore it
        # so this test cannot leak state into other tests in the same process.
        was_enabled = gc.isenabled()
        self.addCleanup(gc.enable if was_enabled else gc.disable)

    def test_on_train_begin_disables_gc_only_when_interval_positive(self):
        callback = GCCallback()
        control = SimpleNamespace()

        gc.enable()
        callback.on_train_begin(
            SimpleNamespace(gc_interval=4), SimpleNamespace(), control
        )
        self.assertFalse(gc.isenabled())  # positive interval -> auto GC off

        gc.enable()
        callback.on_train_begin(
            SimpleNamespace(gc_interval=0), SimpleNamespace(), control
        )
        self.assertTrue(gc.isenabled())  # disabled interval -> left enabled

    def test_on_step_end_collects_only_on_interval_multiples(self):
        callback = GCCallback()
        control = SimpleNamespace()
        args = SimpleNamespace(gc_interval=3)

        # Independently: gc.collect fires iff gc_interval > 0 and
        # global_step % gc_interval == 0. For interval 3 over steps 1..6 that
        # is steps 3 and 6 (2 calls); note step 0 would also qualify.
        with mock.patch.object(gc_callback.gc, "collect") as collect:
            for step in range(1, 7):
                callback.on_step_end(
                    args, SimpleNamespace(global_step=step), control
                )
            self.assertEqual(collect.call_count, 2)

            collect.reset_mock()
            # Step 0 is a multiple of any positive interval -> should fire.
            callback.on_step_end(args, SimpleNamespace(global_step=0), control)
            self.assertEqual(collect.call_count, 1)

            collect.reset_mock()
            # A non-multiple step must not trigger collection.
            callback.on_step_end(args, SimpleNamespace(global_step=5), control)
            self.assertEqual(collect.call_count, 0)

    def test_on_step_end_never_collects_when_interval_disabled(self):
        callback = GCCallback()
        control = SimpleNamespace()
        args = SimpleNamespace(gc_interval=0)

        with mock.patch.object(gc_callback.gc, "collect") as collect:
            # 0 % 0 would even raise; the guard on gc_interval > 0 must
            # short-circuit so no collection (and no error) occurs.
            for step in (0, 1, 2, 10):
                callback.on_step_end(
                    args, SimpleNamespace(global_step=step), control
                )
            self.assertEqual(collect.call_count, 0)


class _ListDumper:
    """Minimal metrics_dumper collaborator; records what it is handed."""

    def __init__(self):
        self.appended = []

    def append(self, logs):
        self.appended.append(logs)


class LoggingCallbackBehaviorTest(unittest.TestCase):
    """LoggingCallback.on_log strips total_flos, forwards the remaining logs
    to the metrics_dumper, and formats scalar values per the documented rules."""

    def setUp(self):
        _skip_if_no_paddle(self)

    def _capture_log_message(self, callback, args, state, control, **kwargs):
        """Run on_log while capturing the single record emitted by the
        callback's module logger, returning (message, kwargs-passthrough)."""
        from paddlefleet.cli.train.ernie_pretrain.src.callbacks import (
            logging_callback,
        )

        records = []

        class _Capture(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = _Capture()
        logging_callback.logger.addHandler(handler)
        prev_level = logging_callback.logger.level
        logging_callback.logger.setLevel(logging.INFO)
        try:
            callback.on_log(args, state, control, **kwargs)
        finally:
            logging_callback.logger.removeHandler(handler)
            logging_callback.logger.setLevel(prev_level)
        return records

    def test_total_flos_removed_and_forwarded_to_dumper(self):
        callback = LoggingCallback()
        dumper = _ListDumper()
        logs = {"loss": 0.5, "total_flos": 123456.0, "grad_norm": 2.0}

        callback.on_log(
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            logs=logs,
            metrics_dumper=dumper,
        )

        # The dumper must receive exactly the surviving keys, with total_flos
        # dropped and every other entry preserved by value.
        self.assertEqual(len(dumper.appended), 1)
        self.assertEqual(dumper.appended[0], {"loss": 0.5, "grad_norm": 2.0})
        self.assertNotIn("total_flos", dumper.appended[0])

    def test_scalar_formatting_rules(self):
        callback = LoggingCallback()
        # Hand-derived expectations from the branch rules:
        #   * key == "loss"        -> plain repr:            "loss: 0.5"
        #   * key contains cur_dp  -> plain repr:            "cur_dp_0: 7.0"
        #   * float and v < 1e-3   -> scientific %e:  "learning_rate: 1.000000e-05"
        #   * other float          -> fixed %f:       "grad_norm: 2.000000"
        #   * non-float            -> plain repr:            "step: 3"
        logs = {
            "loss": 0.5,
            "learning_rate": 1e-5,
            "grad_norm": 2.0,
            "cur_dp_0": 7.0,
            "step": 3,
        }
        messages = self._capture_log_message(
            callback,
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            logs=logs,
        )
        self.assertEqual(len(messages), 1)
        expected = (
            "loss: 0.5, "
            "learning_rate: 1.000000e-05, "
            "grad_norm: 2.000000, "
            "cur_dp_0: 7.0, "
            "step: 3"
        )
        self.assertEqual(messages[0], expected)

    def test_total_flos_absent_from_formatted_message(self):
        callback = LoggingCallback()
        logs = {"total_flos": 999.0, "loss": 1.25}
        messages = self._capture_log_message(
            callback,
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            logs=logs,
        )
        self.assertEqual(messages, ["loss: 1.25"])


if __name__ == "__main__":
    unittest.main()
