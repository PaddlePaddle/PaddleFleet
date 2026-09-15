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

"""Behavior tests for the training tuner entry module.

Module under test: ``paddlefleet.cli.train.tuner``. In the repository module
map this is the "Trainer 训练引擎" / "配置与运行基础设施" boundary: the tuner
selects a training task from ``model_args.stage``, guards required dataset
paths, and dispatches to the concrete ``run_*`` entry of the chosen task.

Three real functions are exercised, all CPU-runnable pure control logic:

* ``check_path(path)`` -- raises ``ValueError`` only when ``path is None``.
  The guard is an *identity* check, not a truthiness check, so a non-empty
  path and even the falsy empty string are accepted.
* ``_training_function(config)`` -- extracts ``config["args"]``, resolves the
  five argument bundles through ``get_train_args``, applies the path-check
  branch, then dispatches on ``model_args.stage``. Expected dispatch targets,
  argument order/arity, and path-check applicability are derived by hand from
  the source, never by re-running the classifier.
* ``run_tuner(args)`` -- resolves ``read_args(args)`` and pipes the *result*
  into ``_training_function``'s ``config["args"]``.

Only genuine, not-under-test collaborators are replaced: ``get_train_args`` /
``read_args`` (config resolution) and the concrete ``run_sft`` / ``run_dpo`` /
``run_auto_parallel`` / pretrain runners. The dispatch, path-check and wiring
logic stays real. ``model_args``/``data_args`` use ``SimpleNamespace`` with real
string ``stage`` values so ``"VL" in stage`` and ``==`` behave like production;
a MagicMock ``stage`` would not.

``paddlefleet.cli.train.tuner`` imports ``paddle`` at module load, and its
parent package pulls the full framework, so the whole suite is guarded on a
real ``paddle`` being importable. This environment has no ``paddle`` installed,
so the suite skips honestly rather than faking the framework.
"""

import importlib
import sys
import types
import unittest
from unittest import mock

try:
    import paddle  # noqa: F401

    _PADDLE_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - env dependent
    paddle = None
    _PADDLE_IMPORT_ERROR = exc

HAS_PADDLE = paddle is not None

TUNER_MODULE = "paddlefleet.cli.train.tuner"

# Modules that tuner imports (eagerly or lazily) but which are NOT under test.
# Replacing them with MagicMocks isolates the heavy training stacks while
# leaving tuner's own dispatch/path-check logic real and observable.
_COLLABORATOR_MODULES = (
    "paddlefleet.cli.hparams",
    "paddlefleet.cli.train.auto_parallel",
    "paddlefleet.cli.train.dpo",
    "paddlefleet.cli.train.sft",
    "paddlefleet.cli.train.deepseek_v3_pretrain",
    "paddlefleet.cli.train.ernie_pretrain",
)


@unittest.skipUnless(
    HAS_PADDLE,
    "tuner imports paddle at module load and this environment has no paddle "
    f"installed (import error: {_PADDLE_IMPORT_ERROR!r}); tuner cannot be "
    "imported, so its dispatch/path-check logic is not verified here.",
)
class TunerBehaviorTest(unittest.TestCase):
    def setUp(self):
        # Inject MagicMock stand-ins for the not-under-test collaborator
        # modules, then import a fresh tuner so its module-level
        # ``from .sft import run_sft`` etc. bind to those stand-ins.
        self._saved_modules = {}
        for name in _COLLABORATOR_MODULES:
            self._saved_modules[name] = sys.modules.get(name)
            sys.modules[name] = mock.MagicMock(name=name)
        self._saved_modules[TUNER_MODULE] = sys.modules.get(TUNER_MODULE)
        sys.modules.pop(TUNER_MODULE, None)
        try:
            self.tuner = importlib.import_module(TUNER_MODULE)
        except ImportError as exc:
            # Deeper dependency missing: skip with an honest reason instead of
            # swallowing the failure into a pass.
            self.skipTest(f"cannot import {TUNER_MODULE}: {exc!r}")

    def tearDown(self):
        for name, mod in self._saved_modules.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod

    @staticmethod
    def _make_args(
        stage,
        dataset_type="pretrain",
        train_path="/data/train.jsonl",
        eval_path="/data/eval.jsonl",
    ):
        """Build the 5-tuple ``get_train_args`` returns.

        Each bundle is a distinct object so dispatch tests can assert argument
        identity and order, not just that a runner was called.
        """
        model_args = types.SimpleNamespace(stage=stage)
        data_args = types.SimpleNamespace(
            dataset_type=dataset_type,
            train_dataset_path=train_path,
            eval_dataset_path=eval_path,
        )
        generating_args = types.SimpleNamespace(tag="generating")
        finetuning_args = types.SimpleNamespace(tag="finetuning")
        preprocess_args = types.SimpleNamespace(tag="preprocess")
        return (
            model_args,
            data_args,
            generating_args,
            finetuning_args,
            preprocess_args,
        )

    # ------------------------------------------------------------------
    # check_path
    # ------------------------------------------------------------------
    def test_check_path_none_raises_with_message(self):
        with self.assertRaises(ValueError) as ctx:
            self.tuner.check_path(None)
        self.assertIn("Dataset Path is None", str(ctx.exception))

    def test_check_path_accepts_non_none_including_empty_string(self):
        # The guard is ``if path is None`` -- an identity check. A real path and
        # the falsy empty string are both accepted (return None, no raise).
        self.assertIsNone(self.tuner.check_path("/data/train.jsonl"))
        self.assertIsNone(self.tuner.check_path(""))

    # ------------------------------------------------------------------
    # _training_function: stage dispatch
    # ------------------------------------------------------------------
    def test_sft_stage_forwards_all_five_args_in_order(self):
        args_tuple = self._make_args("SFT", dataset_type="pretrain")
        model_args, data_args, gen, ft, pre = args_tuple
        config = {"args": object()}
        with (
            mock.patch.object(
                self.tuner, "get_train_args", return_value=args_tuple
            ) as gta,
            mock.patch.object(self.tuner, "run_sft") as run_sft,
            mock.patch.object(self.tuner, "run_dpo") as run_dpo,
            mock.patch.object(self.tuner, "run_auto_parallel") as run_ap,
        ):
            self.tuner._training_function(config)
        gta.assert_called_once_with(config["args"])
        run_dpo.assert_not_called()
        run_ap.assert_not_called()
        run_sft.assert_called_once()
        call = run_sft.call_args
        self.assertEqual(call.kwargs, {})
        # run_sft(model, data, generating, finetuning, preprocess) -- exact order.
        self.assertEqual(len(call.args), 5)
        self.assertIs(call.args[0], model_args)
        self.assertIs(call.args[1], data_args)
        self.assertIs(call.args[2], gen)
        self.assertIs(call.args[3], ft)
        self.assertIs(call.args[4], pre)

    def test_dpo_stage_forwards_four_args_without_preprocess(self):
        args_tuple = self._make_args("DPO", dataset_type="pretrain")
        model_args, data_args, gen, ft, pre = args_tuple
        with (
            mock.patch.object(
                self.tuner, "get_train_args", return_value=args_tuple
            ),
            mock.patch.object(self.tuner, "run_sft") as run_sft,
            mock.patch.object(self.tuner, "run_dpo") as run_dpo,
            mock.patch.object(self.tuner, "run_auto_parallel") as run_ap,
        ):
            self.tuner._training_function({"args": None})
        run_sft.assert_not_called()
        run_ap.assert_not_called()
        run_dpo.assert_called_once()
        call = run_dpo.call_args
        self.assertEqual(call.kwargs, {})
        # run_dpo(model, data, generating, finetuning) -- preprocess withheld.
        self.assertEqual(len(call.args), 4)
        self.assertIs(call.args[0], model_args)
        self.assertIs(call.args[1], data_args)
        self.assertIs(call.args[2], gen)
        self.assertIs(call.args[3], ft)
        self.assertNotIn(pre, call.args)

    def test_auto_parallel_stage_forwards_four_args(self):
        args_tuple = self._make_args("auto-parallel", dataset_type="pretrain")
        model_args, data_args, gen, ft, _pre = args_tuple
        with (
            mock.patch.object(
                self.tuner, "get_train_args", return_value=args_tuple
            ),
            mock.patch.object(self.tuner, "run_sft") as run_sft,
            mock.patch.object(self.tuner, "run_dpo") as run_dpo,
            mock.patch.object(self.tuner, "run_auto_parallel") as run_ap,
        ):
            self.tuner._training_function({"args": None})
        run_sft.assert_not_called()
        run_dpo.assert_not_called()
        run_ap.assert_called_once()
        call = run_ap.call_args
        self.assertEqual(len(call.args), 4)
        self.assertIs(call.args[0], model_args)
        self.assertIs(call.args[1], data_args)
        self.assertIs(call.args[2], gen)
        self.assertIs(call.args[3], ft)

    def test_all_sft_family_stages_dispatch_to_run_sft(self):
        # PT and VL-PT belong to the same dispatch set as SFT/VL-SFT.
        for stage in ("SFT", "PT", "VL-SFT", "VL-PT"):
            args_tuple = self._make_args(stage, dataset_type="pretrain")
            with (
                mock.patch.object(
                    self.tuner, "get_train_args", return_value=args_tuple
                ),
                mock.patch.object(self.tuner, "run_sft") as run_sft,
                mock.patch.object(self.tuner, "run_dpo") as run_dpo,
            ):
                self.tuner._training_function({"args": None})
            run_sft.assert_called_once()
            run_dpo.assert_not_called()

    def test_vl_dpo_stage_dispatches_to_run_dpo(self):
        args_tuple = self._make_args("VL-DPO", dataset_type="pretrain")
        with (
            mock.patch.object(
                self.tuner, "get_train_args", return_value=args_tuple
            ),
            mock.patch.object(self.tuner, "run_dpo") as run_dpo,
            mock.patch.object(self.tuner, "run_sft") as run_sft,
        ):
            self.tuner._training_function({"args": None})
        run_dpo.assert_called_once()
        run_sft.assert_not_called()

    def test_unknown_stage_raises_value_error_naming_stage(self):
        args_tuple = self._make_args("BOGUS", dataset_type="pretrain")
        with (
            mock.patch.object(
                self.tuner, "get_train_args", return_value=args_tuple
            ),
            mock.patch.object(self.tuner, "run_sft") as run_sft,
        ):
            with self.assertRaises(ValueError) as ctx:
                self.tuner._training_function({"args": None})
        # Message interpolates the offending stage exactly.
        self.assertEqual(str(ctx.exception), "Unknown task: BOGUS.")
        run_sft.assert_not_called()

    def test_dsv3_pretrain_stage_skips_path_check_and_dispatches(self):
        # dsv3_pretrain takes the first path-skip branch, so missing paths on a
        # non-pretrain dataset_type must NOT raise; dispatch reaches the runner.
        args_tuple = self._make_args(
            "dsv3_pretrain",
            dataset_type="sft",
            train_path=None,
            eval_path=None,
        )
        model_args, data_args, gen, ft, _pre = args_tuple
        dsv3_mod = sys.modules["paddlefleet.cli.train.deepseek_v3_pretrain"]
        with mock.patch.object(
            self.tuner, "get_train_args", return_value=args_tuple
        ):
            self.tuner._training_function({"args": None})
        dsv3_mod.run_dsv3_pretrain.assert_called_once()
        call = dsv3_mod.run_dsv3_pretrain.call_args
        self.assertEqual(len(call.args), 4)
        self.assertIs(call.args[0], model_args)
        self.assertIs(call.args[1], data_args)
        self.assertIs(call.args[2], gen)
        self.assertIs(call.args[3], ft)

    # ------------------------------------------------------------------
    # _training_function: path-check branch
    # ------------------------------------------------------------------
    def test_sft_non_pretrain_missing_train_path_raises_before_dispatch(self):
        args_tuple = self._make_args(
            "SFT", dataset_type="sft", train_path=None, eval_path="/data/eval"
        )
        with (
            mock.patch.object(
                self.tuner, "get_train_args", return_value=args_tuple
            ),
            mock.patch.object(self.tuner, "run_sft") as run_sft,
        ):
            with self.assertRaises(ValueError) as ctx:
                self.tuner._training_function({"args": None})
        self.assertIn("Dataset Path is None", str(ctx.exception))
        # Guard runs before dispatch, so the runner is never reached.
        run_sft.assert_not_called()

    def test_sft_non_pretrain_missing_eval_path_raises(self):
        args_tuple = self._make_args(
            "SFT", dataset_type="sft", train_path="/data/train", eval_path=None
        )
        with (
            mock.patch.object(
                self.tuner, "get_train_args", return_value=args_tuple
            ),
            mock.patch.object(self.tuner, "run_sft") as run_sft,
        ):
            with self.assertRaises(ValueError):
                self.tuner._training_function({"args": None})
        run_sft.assert_not_called()

    def test_pretrain_dataset_type_skips_path_check(self):
        args_tuple = self._make_args(
            "SFT", dataset_type="pretrain", train_path=None, eval_path=None
        )
        with (
            mock.patch.object(
                self.tuner, "get_train_args", return_value=args_tuple
            ),
            mock.patch.object(self.tuner, "run_sft") as run_sft,
        ):
            self.tuner._training_function({"args": None})
        run_sft.assert_called_once()

    def test_offline_dataset_type_skips_path_check(self):
        # "offline" is exempted from the path check alongside "pretrain".
        args_tuple = self._make_args(
            "SFT", dataset_type="offline", train_path=None, eval_path=None
        )
        with (
            mock.patch.object(
                self.tuner, "get_train_args", return_value=args_tuple
            ),
            mock.patch.object(self.tuner, "run_sft") as run_sft,
        ):
            self.tuner._training_function({"args": None})
        run_sft.assert_called_once()

    def test_vl_stage_skips_path_check_via_substring(self):
        # "VL" substring in the stage takes the path-skip branch even for a
        # non-pretrain dataset_type with missing paths.
        args_tuple = self._make_args(
            "VL-SFT", dataset_type="sft", train_path=None, eval_path=None
        )
        with (
            mock.patch.object(
                self.tuner, "get_train_args", return_value=args_tuple
            ),
            mock.patch.object(self.tuner, "run_sft") as run_sft,
        ):
            self.tuner._training_function({"args": None})
        run_sft.assert_called_once()

    def test_ernie_pretrain_stage_is_subject_to_path_check(self):
        # Unlike dsv3_pretrain, ernie_pretrain is NOT in the first path-skip
        # branch; with a non-pretrain dataset_type and a missing path it must
        # raise before reaching the ernie runner.
        args_tuple = self._make_args(
            "ernie_pretrain",
            dataset_type="sft",
            train_path=None,
            eval_path="/data/eval",
        )
        with mock.patch.object(
            self.tuner, "get_train_args", return_value=args_tuple
        ):
            with self.assertRaises(ValueError) as ctx:
                self.tuner._training_function({"args": None})
        self.assertIn("Dataset Path is None", str(ctx.exception))

    # ------------------------------------------------------------------
    # run_tuner: wiring read_args -> _training_function
    # ------------------------------------------------------------------
    def test_run_tuner_pipes_read_args_output_into_training_function(self):
        resolved_args = types.SimpleNamespace(marker="resolved")
        args_tuple = self._make_args("SFT", dataset_type="pretrain")
        captured = {}

        def fake_get_train_args(received):
            captured["args"] = received
            return args_tuple

        raw_input = {"output_dir": "/tmp/out"}
        with (
            mock.patch.object(
                self.tuner, "read_args", return_value=resolved_args
            ) as read_args,
            mock.patch.object(
                self.tuner, "get_train_args", side_effect=fake_get_train_args
            ),
            mock.patch.object(self.tuner, "run_sft") as run_sft,
        ):
            self.tuner.run_tuner(raw_input)
        read_args.assert_called_once_with(raw_input)
        # The resolved args object -- not the raw input -- must reach
        # get_train_args, proving run_tuner forwards read_args() through
        # config["args"].
        self.assertIs(captured["args"], resolved_args)
        run_sft.assert_called_once()

    def test_run_tuner_defaults_none_through_read_args(self):
        args_tuple = self._make_args("SFT", dataset_type="pretrain")
        with (
            mock.patch.object(
                self.tuner, "read_args", return_value=None
            ) as read_args,
            mock.patch.object(
                self.tuner, "get_train_args", return_value=args_tuple
            ) as gta,
            mock.patch.object(self.tuner, "run_sft") as run_sft,
        ):
            self.tuner.run_tuner()
        read_args.assert_called_once_with(None)
        gta.assert_called_once_with(None)
        run_sft.assert_called_once()


if __name__ == "__main__":
    unittest.main()
