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

"""Behavior tests for the real command dispatch in ``paddlefleet.cli.cli.main``.

These tests drive the *actual* ``main()`` entry point and observe the routing
decision it makes for a given ``sys.argv``:

* ``help`` / ``version`` / unknown commands fall to the ``COMMAND_MAP`` /
  fallback branch and print ``USAGE`` / ``WELCOME`` / an unknown-command notice.
* ``train`` / ``export`` are intercepted by the ``distributed_funcs`` branch
  *before* ``COMMAND_MAP``, so they are launched as a
  ``paddle.distributed.launch`` subprocess and the in-map ``run_tuner`` /
  ``run_export`` handlers are never called directly. We assert the full argv
  routed into the launch command (device flag, master/nnodes/rank, launcher
  file and the forwarded subcommand tokens) against a hand-derived expectation.

Only *peripheral* collaborators are replaced: the heavy handler modules
(``launcher``, ``train.tuner``, ``export.export``), ``detect_device``, the GPU
probe and ``subprocess.Popen``. The dispatch itself (the ``if/elif/else`` in
``main`` and the ``argv``-to-command derivation) is executed for real -- we do
NOT rebuild ``COMMAND_MAP`` or re-derive ``command`` inside the test, which is
the "测试内重写 CLI 分派" / "入口只检查被调用" antipattern.

The whole ``paddlefleet`` package imports ``paddle`` at import time, so tests
skip when Paddle is unavailable; they run for real on any CPU with Paddle.
Expected launch commands are hand-derived from the format strings in ``cli.py``.
"""

import contextlib
import io
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

try:
    from paddlefleet.cli import cli as cli_mod

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet not installed
    cli_mod = None
    _IMPORT_ERROR = exc


_MISSING = object()
_FAKE_LAUNCHER_FILE = "/fake/paddlefleet/cli/launcher.py"


class _FakeProcess:
    """Stand-in for the ``subprocess.Popen`` object main() drives.

    ``wait()`` optionally raises to exercise the interrupt/error branches;
    ``returncode`` is what main() ultimately passes to ``sys.exit`` in its
    ``finally`` clause.
    """

    def __init__(self, returncode=0, wait_exc=None):
        self.returncode = returncode
        self.pid = 4242
        self._wait_exc = wait_exc
        self.wait_called = False

    def wait(self):
        self.wait_called = True
        if self._wait_exc is not None:
            raise self._wait_exc
        return self.returncode


class CliDispatchTest(unittest.TestCase):
    """Exercise the real ``main()`` routing on CPU with heavy deps stubbed."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"paddlefleet/paddle unavailable: {_IMPORT_ERROR!r}")
        self.cli = cli_mod

        # Peripheral handler mocks. main() imports these unconditionally at the
        # top of the function; train/export dispatch never calls them (it goes
        # through the subprocess branch), and help/version/unknown never touch
        # them either -- so stubbing their import isolates unrelated heavy code
        # without hiding any dispatch decision.
        self.run_tuner = MagicMock(name="run_tuner")
        self.run_export = MagicMock(name="run_export")

        launcher = types.ModuleType("paddlefleet.cli.launcher")
        launcher.__file__ = _FAKE_LAUNCHER_FILE
        train_pkg = types.ModuleType("paddlefleet.cli.train")
        tuner = types.ModuleType("paddlefleet.cli.train.tuner")
        tuner.run_tuner = self.run_tuner
        train_pkg.tuner = tuner
        export_pkg = types.ModuleType("paddlefleet.cli.export")
        export_mod = types.ModuleType("paddlefleet.cli.export.export")
        export_mod.run_export = self.run_export
        export_pkg.export = export_mod

        gputil = types.ModuleType("GPUtil")
        # Two fake GPUs -> default visible cards "0,1" when CUDA_VISIBLE_DEVICES
        # is unset. Only consumed when building the launch command.
        gputil.getGPUs = lambda: [object(), object()]

        module_patch = patch.dict(
            sys.modules,
            {
                "paddlefleet.cli.launcher": launcher,
                "paddlefleet.cli.train": train_pkg,
                "paddlefleet.cli.train.tuner": tuner,
                "paddlefleet.cli.export": export_pkg,
                "paddlefleet.cli.export.export": export_mod,
                "GPUtil": gputil,
            },
        )
        module_patch.start()
        self.addCleanup(module_patch.stop)

        # ``from . import launcher`` / ``from .export.export import ...`` set
        # attributes on the *real* paddlefleet.cli package. Restore them so a
        # later real import in the same process is not shadowed by our stubs.
        import paddlefleet.cli as pkg

        self._pkg = pkg
        self._saved_attrs = {
            name: getattr(pkg, name, _MISSING)
            for name in ("launcher", "train", "export")
        }
        self.addCleanup(self._restore_pkg_attrs)

    def _restore_pkg_attrs(self):
        for name, val in self._saved_attrs.items():
            if val is _MISSING:
                if hasattr(self._pkg, name):
                    delattr(self._pkg, name)
            else:
                setattr(self._pkg, name, val)

    def _invoke_main(self, argv, device="gpu", popen=None, env=None):
        """Run the real ``cli.main()`` under a controlled environment.

        Patches only peripheral pieces: ``sys.argv`` (the dispatch input),
        ``detect_device`` (returns ``device``), a clean ``os.environ`` (so the
        launch command is deterministic) and ``subprocess.Popen``. Returns the
        captured stdout, the ``SystemExit`` code (or None) and the Popen mock.
        """
        popen_mock = popen if popen is not None else MagicMock(name="Popen")
        out = io.StringIO()
        code = None
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(self.cli.sys, "argv", list(argv)))
            stack.enter_context(
                patch.object(self.cli, "detect_device", new=lambda: device)
            )
            stack.enter_context(patch.dict(os.environ, env or {}, clear=True))
            stack.enter_context(
                patch.object(self.cli.subprocess, "Popen", popen_mock)
            )
            stack.enter_context(contextlib.redirect_stdout(out))
            try:
                self.cli.main()
            except SystemExit as exc:
                code = exc.code
        return out.getvalue(), code, popen_mock

    # ---- COMMAND_MAP / fallback branch (no subprocess) -----------------

    def test_no_args_defaults_to_help(self):
        """No argv[1] -> command defaults to 'help' -> prints USAGE."""
        out, code, popen = self._invoke_main(["paddlefleet-cli"])
        self.assertEqual(out, self.cli.USAGE + "\n")
        self.assertIsNone(code)
        popen.assert_not_called()
        self.run_tuner.assert_not_called()
        self.run_export.assert_not_called()

    def test_help_command_prints_usage(self):
        """Explicit 'help' routes to the USAGE printer, not a launch."""
        out, code, popen = self._invoke_main(["paddlefleet-cli", "help"])
        self.assertEqual(out, self.cli.USAGE + "\n")
        self.assertIsNone(code)
        popen.assert_not_called()

    def test_version_command_prints_welcome(self):
        """'version' routes to the WELCOME printer (distinct from USAGE)."""
        out, code, popen = self._invoke_main(["paddlefleet-cli", "version"])
        self.assertEqual(out, self.cli.WELCOME + "\n")
        self.assertNotIn(self.cli.USAGE, out)  # not the help handler
        self.assertIsNone(code)
        popen.assert_not_called()

    def test_unknown_command_reports_and_shows_usage(self):
        """An unrecognized command hits the else branch with a notice."""
        out, code, popen = self._invoke_main(["paddlefleet-cli", "frobnicate"])
        self.assertEqual(
            out, f"Unknown command: frobnicate.\n{self.cli.USAGE}\n"
        )
        self.assertIsNone(code)
        popen.assert_not_called()
        self.run_tuner.assert_not_called()
        self.run_export.assert_not_called()

    # ---- distributed_funcs branch (subprocess launch) ------------------

    def test_train_launches_distributed_subprocess(self):
        """'train' is routed to a paddle.distributed.launch subprocess.

        The full command is hand-derived from the format string in cli.py:
        default env -> log dir 'paddlefleet_dist_log', gpu flag with the two
        fake GPUs, master 127.0.0.1:8080, nnodes 1, rank 0, the launcher file,
        then the forwarded subcommand tokens. main() exits with the process
        return code via its ``finally`` clause.
        """
        proc = _FakeProcess(returncode=7)
        popen = MagicMock(name="Popen", return_value=proc)
        out, code, popen = self._invoke_main(
            ["paddlefleet-cli", "train", "--dataset", "d1"], popen=popen
        )
        expected_cmd = [
            "python",
            "-m",
            "paddle.distributed.launch",
            "--log_dir",
            "paddlefleet_dist_log",
            "--gpus",
            "0,1",
            "--master",
            "127.0.0.1:8080",
            "--nnodes",
            "1",
            "--rank",
            "0",
            "--run_mode=collective",
            _FAKE_LAUNCHER_FILE,
            "train",
            "--dataset",
            "d1",
        ]
        popen.assert_called_once()
        args, kwargs = popen.call_args
        self.assertEqual(args[0], expected_cmd)
        self.assertTrue(proc.wait_called)
        self.assertEqual(code, 7)
        # The COMMAND_MAP entry for 'train' is shadowed by this branch.
        self.run_tuner.assert_not_called()
        # Environment is propagated to the child (FLAGS defaults were set).
        child_env = dict(kwargs["env"])
        self.assertEqual(child_env.get("FLAGS_set_to_1d"), "False")

    def test_export_launches_distributed_subprocess(self):
        """'export' also takes the launch branch, not the run_export handler."""
        proc = _FakeProcess(returncode=0)
        popen = MagicMock(name="Popen", return_value=proc)
        out, code, popen = self._invoke_main(
            ["paddlefleet-cli", "export", "--out", "o"], popen=popen
        )
        expected_cmd = [
            "python",
            "-m",
            "paddle.distributed.launch",
            "--log_dir",
            "paddlefleet_dist_log",
            "--gpus",
            "0,1",
            "--master",
            "127.0.0.1:8080",
            "--nnodes",
            "1",
            "--rank",
            "0",
            "--run_mode=collective",
            _FAKE_LAUNCHER_FILE,
            "export",
            "--out",
            "o",
        ]
        popen.assert_called_once()
        self.assertEqual(popen.call_args.args[0], expected_cmd)
        self.assertEqual(code, 0)
        self.run_export.assert_not_called()

    def test_env_vars_route_into_launch_command(self):
        """NNODES/RANK/MASTER_*/CUDA_VISIBLE_DEVICES/log dir reach the command."""
        proc = _FakeProcess(returncode=0)
        popen = MagicMock(name="Popen", return_value=proc)
        env = {
            "NNODES": "4",
            "RANK": "2",
            "MASTER_ADDR": "10.0.0.5",
            "MASTER_PORT": "1234",
            "CUDA_VISIBLE_DEVICES": "2,3",
            "PADDLEFLEET_DIST_LOG": "mylog",
        }
        out, code, popen = self._invoke_main(
            ["paddlefleet-cli", "train"], popen=popen, env=env
        )
        expected_cmd = [
            "python",
            "-m",
            "paddle.distributed.launch",
            "--log_dir",
            "mylog",
            "--gpus",
            "2,3",
            "--master",
            "10.0.0.5:1234",
            "--nnodes",
            "4",
            "--rank",
            "2",
            "--run_mode=collective",
            _FAKE_LAUNCHER_FILE,
            "train",
        ]
        self.assertEqual(popen.call_args.args[0], expected_cmd)
        self.assertEqual(code, 0)

    def test_iluvatar_device_remaps_to_gpu_flag(self):
        """iluvatar_gpu is remapped to the '--gpus' flag in the launch cmd."""
        paddle_device = self.cli.paddle.device
        if not hasattr(paddle_device, "get_available_custom_device"):
            self.skipTest(
                "paddle.device.get_available_custom_device unavailable"
            )
        proc = _FakeProcess(returncode=0)
        popen = MagicMock(name="Popen", return_value=proc)
        with patch.object(
            paddle_device,
            "get_available_custom_device",
            return_value=["dev0", "dev1"],
        ):
            out, code, popen = self._invoke_main(
                ["paddlefleet-cli", "train"], device="iluvatar_gpu", popen=popen
            )
        cmd = popen.call_args.args[0]
        self.assertIn("--gpus", cmd)
        self.assertNotIn("--iluvatar_gpus", cmd)
        # Two custom devices -> default visible cards "0,1".
        self.assertEqual(cmd[cmd.index("--gpus") + 1], "0,1")
        self.assertEqual(code, 0)

    # ---- error handling on the launched process -----------------------

    def test_process_error_triggers_terminate(self):
        """When wait() errors, main() prints and terminates the process tree.

        This documents the *actual* behavior: the ``except Exception`` branch
        runs (terminate + message), but the ``finally: sys.exit(returncode)``
        overrides its ``sys.exit(1)`` -- so with an unfinished process
        (returncode None) the CLI exits 0. See the expected-failure test below.
        """
        proc = _FakeProcess(returncode=None, wait_exc=RuntimeError("boom"))
        popen = MagicMock(name="Popen", return_value=proc)
        with patch.object(self.cli, "terminate_process_tree") as term:
            out, code, popen = self._invoke_main(
                ["paddlefleet-cli", "train"], popen=popen
            )
        term.assert_called_once_with(proc.pid)
        self.assertIn("Server process failed", out)
        self.assertIsNone(code)  # finally's sys.exit(None) masks exit(1)

    def test_keyboard_interrupt_terminates_process(self):
        """Ctrl-C during wait() terminates the child and prints a notice."""
        proc = _FakeProcess(returncode=None, wait_exc=KeyboardInterrupt())
        popen = MagicMock(name="Popen", return_value=proc)
        with patch.object(self.cli, "terminate_process_tree") as term:
            out, code, popen = self._invoke_main(
                ["paddlefleet-cli", "train"], popen=popen
            )
        term.assert_called_once_with(proc.pid)
        self.assertIn("Received interrupt", out)
        self.assertIsNone(code)  # same finally-masking as above

    @unittest.expectedFailure
    def test_process_error_should_exit_nonzero(self):
        """INTENDED: a failing launch should propagate a non-zero exit code.

        Production bug: in ``main()`` the ``except Exception`` handler calls
        ``sys.exit(1)``, but the surrounding ``finally: sys.exit(process.
        returncode)`` fires afterwards and replaces the exit code. With a
        process that never finished (returncode is None) the CLI therefore
        exits 0, silently reporting success on failure. This test asserts the
        intended non-zero exit and is expected to fail. No production code is
        modified.
        """
        proc = _FakeProcess(returncode=None, wait_exc=RuntimeError("boom"))
        popen = MagicMock(name="Popen", return_value=proc)
        with patch.object(self.cli, "terminate_process_tree"):
            _out, code, _popen = self._invoke_main(
                ["paddlefleet-cli", "train"], popen=popen
            )
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
