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

"""Behavior tests for the ERNIE pretrain LoggingCallback.

The production module imports ``paddlefleet.trainer.trainer_callback`` which
imports ``paddle`` at module load, so importing ``LoggingCallback`` requires a
working paddle install. The local dev environment has NO paddle; those runs
skip. Expected values below are derived by hand from the on_log source, not
copied from any existing test.
"""

import unittest

try:
    import paddle

    from paddlefleet.cli.train.ernie_pretrain.src.callbacks.logging_callback import (
        LoggingCallback,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # honest capability probe: only missing-module
    paddle = None
    LoggingCallback = None
    _IMPORT_ERROR = repr(exc)

_LOGGER_NAME = LoggingCallback.__module__ if LoggingCallback is not None else ""


@unittest.skipIf(
    LoggingCallback is None,
    f"paddle/production import unavailable (local env has no paddle): {_IMPORT_ERROR}",
)
class LoggingCallbackBehaviorTest(unittest.TestCase):
    def setUp(self):
        self.callback = LoggingCallback()

    def _log_message(self, logs, **kwargs):
        """Invoke on_log and return the single formatted log line it emits."""
        with self.assertLogs(_LOGGER_NAME, level="INFO") as cm:
            self.callback.on_log(
                object(), object(), object(), logs=logs, **kwargs
            )
        self.assertEqual(len(cm.records), 1)
        return cm.records[0].getMessage()

    def test_float_formatting_selects_branch_per_key(self):
        # Hand-derived per the on_log formatting rules:
        #   k == "loss" or "cur_dp" in k -> str(v) (no width spec)
        #   float and v < 1e-3            -> "%e"
        #   float and v >= 1e-3           -> "%f" (6 decimals)
        #   non-float                     -> str(v)
        logs = {
            "loss": 0.5,
            "cur_dp_loss": 0.25,
            "learning_rate": 0.01,
            "grad_norm": 1e-6,
            "global_step": 42,
        }
        expected = (
            "loss: 0.5, "
            "cur_dp_loss: 0.25, "
            "learning_rate: 0.010000, "
            "grad_norm: 1.000000e-06, "
            "global_step: 42"
        )
        self.assertEqual(self._log_message(logs), expected)

    def test_loss_key_is_not_fixed_width_formatted(self):
        # Guards the special-case: a plain non-loss float at 0.5 becomes
        # "0.500000", but "loss" must stay "0.5".
        self.assertEqual(self._log_message({"loss": 0.5}), "loss: 0.5")
        self.assertEqual(self._log_message({"reward": 0.5}), "reward: 0.500000")

    def test_total_flos_is_popped_from_caller_dict_and_output(self):
        logs = {"loss": 0.5, "total_flos": 12345.0, "learning_rate": 0.02}
        message = self._log_message(logs)
        # Removed in place from the caller's dict.
        self.assertNotIn("total_flos", logs)
        self.assertEqual(logs, {"loss": 0.5, "learning_rate": 0.02})
        # And therefore absent from the emitted line.
        self.assertNotIn("total_flos", message)
        self.assertEqual(message, "loss: 0.5, learning_rate: 0.020000")

    def test_metrics_dumper_receives_post_pop_logs(self):
        dumper = []
        logs = {"loss": 0.5, "total_flos": 999.0, "acc": 0.75}
        self._log_message(logs, metrics_dumper=dumper)
        self.assertEqual(len(dumper), 1)
        self.assertEqual(dumper[0], {"loss": 0.5, "acc": 0.75})

    def test_data_id_tensor_is_dash_joined_without_mutating_caller(self):
        # "-".join(map(str, tensor.numpy().tolist())) => "1-2-3"
        dumper = []
        logs = {"loss": 0.5}
        message = self._log_message(
            logs,
            inputs={"data_id": paddle.to_tensor([1, 2, 3])},
            metrics_dumper=dumper,
        )
        self.assertEqual(message, "loss: 0.5, data_id: 1-2-3")
        self.assertEqual(dumper[0], {"loss": 0.5, "data_id": "1-2-3"})
        # data_id path rebinds logs via dict(...), so caller dict is untouched.
        self.assertEqual(logs, {"loss": 0.5})

    def test_src_id_tensor_is_dash_joined_without_mutating_caller(self):
        dumper = []
        logs = {"loss": 0.5}
        message = self._log_message(
            logs,
            inputs={"src_id": paddle.to_tensor([4, 5])},
            metrics_dumper=dumper,
        )
        self.assertEqual(message, "loss: 0.5, src_id: 4-5")
        self.assertEqual(dumper[0], {"loss": 0.5, "src_id": "4-5"})
        self.assertEqual(logs, {"loss": 0.5})

    def test_data_type_tensor_updates_caller_dict_in_place(self):
        # data_type uses logs.update(...) (not dict(...)), so with no data_id/
        # src_id it mutates the caller's dict -- distinct from data_id/src_id.
        dumper = []
        logs = {"loss": 0.5}
        message = self._log_message(
            logs,
            inputs={"data_type": paddle.to_tensor([7, 8])},
            metrics_dumper=dumper,
        )
        self.assertEqual(message, "loss: 0.5, data_type: 7-8")
        self.assertEqual(dumper[0], {"loss": 0.5, "data_type": "7-8"})
        self.assertEqual(logs, {"loss": 0.5, "data_type": "7-8"})

    def test_all_id_fields_combine_in_insertion_order(self):
        dumper = []
        logs = {"loss": 0.5}
        message = self._log_message(
            logs,
            inputs={
                "data_id": paddle.to_tensor([1, 2]),
                "src_id": paddle.to_tensor([3, 4]),
                "data_type": paddle.to_tensor([5, 6]),
            },
            metrics_dumper=dumper,
        )
        self.assertEqual(
            message, "loss: 0.5, data_id: 1-2, src_id: 3-4, data_type: 5-6"
        )
        self.assertEqual(
            dumper[0],
            {
                "loss": 0.5,
                "data_id": "1-2",
                "src_id": "3-4",
                "data_type": "5-6",
            },
        )
        # data_id/src_id rebind to a fresh dict before data_type.update, so the
        # caller's original dict is left unchanged here.
        self.assertEqual(logs, {"loss": 0.5})


if __name__ == "__main__":
    unittest.main()
