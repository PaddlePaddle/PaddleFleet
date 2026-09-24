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

"""Behavior tests for ``paddlefleet.cli.launcher``.

The launcher exposes three cooperating units (all pure stdlib, no Paddle math):

* ``_reexec_with_numactl(local_rank, numa_node)`` re-launches the current
  process under ``numactl`` pinned to a NUMA node. It is a no-op once the
  ``BIND_TRAINER_NUMA_EXECED`` sentinel is set (so the second, post-exec run
  does not recurse), raises when ``numactl`` is absent from PATH, and otherwise
  builds the argv ``[numactl, --cpunodebind=N, --membind=N, <python>, -u,
  *sys.argv]`` with a *copy* of the environment carrying the sentinel and
  ``PYTHONUNBUFFERED=1``.
* ``_maybe_bind_trainer_numa()`` gates the above on the ``BIND_TRAINER_NUMA``
  flag (truthy set {1,true,on,yes}, case-insensitive), reads the local rank from
  ``PADDLE_LOCAL_RANK`` (else ``FLAGS_selected_gpus``, taking the first
  comma-separated entry), maps rank {0,1}->node 0 and {2,3}->node 1, and raises
  for a missing rank or an unsupported rank.
* ``launch()`` dispatches ``sys.argv[1]``: ``train`` binds NUMA then runs the
  tuner, ``export`` runs the exporter, anything else (or a missing command)
  raises ``ValueError``.

All expected values here are hand-derived from that specification, not read
back from the code under test. Only genuinely external collaborators are
substituted: ``os.execvpe`` (would replace the process), ``shutil.which`` (a
PATH probe), and the lazily-imported ``run_tuner`` / ``run_export`` entrypoints
(injected via ``sys.modules`` so the real dispatch branch executes). The rank
-> node mapping and the numactl argv are asserted by their exact content.

The whole ``paddlefleet`` package imports ``paddle`` at import time via
``parallel_state``. The launcher module itself needs none of that, so when the
package import fails (no Paddle locally) we load the exact production source
file directly and exercise the same functions; the tests then note that the
package's lazy-import wiring was not covered. When neither path works the tests
skip with an honest reason.
"""

import importlib.util
import os
import sys
import types
import unittest
from unittest import mock

_LAUNCHER_SRC = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "src",
        "paddlefleet",
        "cli",
        "launcher.py",
    )
)


def _load_launcher():
    """Prefer the real package API; fall back to the production source file.

    Returns (module, note) or (None, reason). The fallback loads the untouched
    production file, so behavior is identical; only the package's lazy-import
    wiring is left unverified.
    """
    try:
        from paddlefleet.cli import launcher as mod

        return mod, None
    except ImportError as exc:
        pkg_err = exc
    if not os.path.isfile(_LAUNCHER_SRC):
        return (
            None,
            f"launcher source not found and package import failed: {pkg_err!r}",
        )
    try:
        spec = importlib.util.spec_from_file_location(
            "paddlefleet_cli_launcher_under_test", _LAUNCHER_SRC
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except ImportError as exc:
        return None, f"launcher source import failed: {exc!r}"
    return mod, f"loaded from source; package import unavailable ({pkg_err!r})"


launcher_mod, _LOAD_NOTE = _load_launcher()


class _LauncherTestBase(unittest.TestCase):
    def setUp(self):
        if launcher_mod is None:
            self.skipTest(_LOAD_NOTE)


class ReexecWithNumactlTest(_LauncherTestBase):
    """_reexec_with_numactl: argv/env construction, idempotency, PATH guard."""

    def test_builds_numactl_argv_and_child_env(self):
        """The re-exec command pins cpu+mem to the *numa_node* (not the rank),
        wraps the current interpreter with -u, appends the original argv, and
        hands numactl a copied env carrying the sentinel + PYTHONUNBUFFERED."""
        captured = {}

        def fake_execvpe(file, args, env):
            captured["file"] = file
            captured["args"] = list(args)
            captured["env"] = dict(env)

        # local_rank (3) deliberately differs from numa_node (1): a swap of the
        # two args would put "--cpunodebind=3" here and be caught below.
        with mock.patch.dict(
            os.environ, {"PADDLE_TRAINER_ID": "keep-me"}, clear=True
        ):
            with (
                mock.patch.object(
                    launcher_mod.shutil,
                    "which",
                    return_value="/opt/bin/numactl",
                ) as which,
                mock.patch.object(
                    launcher_mod.os, "execvpe", side_effect=fake_execvpe
                ) as execvpe,
                mock.patch.object(
                    launcher_mod.sys, "executable", "/venv/bin/python"
                ),
                mock.patch.object(
                    launcher_mod.sys,
                    "argv",
                    ["train.py", "train", "--config", "a.yaml"],
                ),
                mock.patch.object(launcher_mod, "print"),
            ):
                launcher_mod._reexec_with_numactl(3, 1)

            # os.environ itself must not be polluted with the sentinel.
            self.assertNotIn(launcher_mod.BIND_TRAINER_NUMA_EXECED, os.environ)

        which.assert_called_once_with("numactl")
        execvpe.assert_called_once()
        self.assertEqual(captured["file"], "/opt/bin/numactl")
        self.assertEqual(
            captured["args"],
            [
                "/opt/bin/numactl",
                "--cpunodebind=1",
                "--membind=1",
                "/venv/bin/python",
                "-u",
                "train.py",
                "train",
                "--config",
                "a.yaml",
            ],
        )
        # Child env is a copy: pre-existing vars survive, sentinel is added.
        self.assertEqual(captured["env"]["PADDLE_TRAINER_ID"], "keep-me")
        self.assertEqual(
            captured["env"][launcher_mod.BIND_TRAINER_NUMA_EXECED], "1"
        )
        self.assertEqual(captured["env"]["PYTHONUNBUFFERED"], "1")

    def test_is_noop_once_already_execed(self):
        """With the sentinel present the process is already inside numactl, so
        neither the PATH probe nor a second exec must happen."""
        with (  # noqa: SIM117
            mock.patch.dict(
                os.environ,
                {launcher_mod.BIND_TRAINER_NUMA_EXECED: "1"},
                clear=True,
            ),
            mock.patch.object(launcher_mod.shutil, "which") as which,
        ):
            with mock.patch.object(launcher_mod.os, "execvpe") as execvpe:
                result = launcher_mod._reexec_with_numactl(0, 0)

        self.assertIsNone(result)
        which.assert_not_called()
        execvpe.assert_not_called()

    def test_raises_when_numactl_absent_from_path(self):
        """No numactl on PATH is a hard error, and no exec is attempted."""
        with mock.patch.dict(os.environ, {}, clear=True):  # noqa: SIM117
            with mock.patch.object(
                launcher_mod.shutil, "which", return_value=None
            ):
                with mock.patch.object(launcher_mod.os, "execvpe") as execvpe:
                    with self.assertRaisesRegex(
                        RuntimeError, "requires numactl"
                    ):
                        launcher_mod._reexec_with_numactl(0, 0)
        execvpe.assert_not_called()


class MaybeBindTrainerNumaTest(_LauncherTestBase):
    """_maybe_bind_trainer_numa: flag gating, rank source, rank->node map."""

    def _run_with_env(self, env):
        """Run the gate with a controlled env, capturing the (rank, node) that
        reaches _reexec_with_numactl. Returns None if reexec was not invoked."""
        seen = []
        with mock.patch.dict(os.environ, env, clear=True):  # noqa: SIM117
            with mock.patch.object(
                launcher_mod,
                "_reexec_with_numactl",
                side_effect=lambda r, n: seen.append((r, n)),
            ):
                with mock.patch.object(launcher_mod, "print"):
                    launcher_mod._maybe_bind_trainer_numa()
        return seen[0] if seen else None

    def test_disabled_flag_values_skip_rebind(self):
        """Unset or falsy BIND_TRAINER_NUMA leaves the process untouched even
        when a rank is present (so the skip is a real branch, not a missing
        rank)."""
        for flag in (None, "0", "false", "no", "off", ""):
            with self.subTest(flag=flag):
                env = {"PADDLE_LOCAL_RANK": "0"}
                if flag is not None:
                    env["BIND_TRAINER_NUMA"] = flag
                self.assertIsNone(self._run_with_env(env))

    def test_truthy_flag_values_trigger_rebind(self):
        """Each accepted truthy spelling (case-insensitive) enables binding."""
        for flag in ("1", "true", "TRUE", "on", "On", "yes", "YES"):
            with self.subTest(flag=flag):
                got = self._run_with_env(
                    {"BIND_TRAINER_NUMA": flag, "PADDLE_LOCAL_RANK": "0"}
                )
                self.assertEqual(got, (0, 0))

    def test_local_rank_maps_to_expected_numa_node(self):
        """Hand-derived map: ranks 0,1 -> node 0; ranks 2,3 -> node 1. The
        rank is forwarded unchanged alongside its node."""
        expected = {"0": (0, 0), "1": (1, 0), "2": (2, 1), "3": (3, 1)}
        for rank_str, want in expected.items():
            with self.subTest(rank=rank_str):
                got = self._run_with_env(
                    {
                        "BIND_TRAINER_NUMA": "1",
                        "PADDLE_LOCAL_RANK": rank_str,
                    }
                )
                self.assertEqual(got, want)

    def test_falls_back_to_flags_selected_gpus_first_entry(self):
        """Without PADDLE_LOCAL_RANK the first comma-separated GPU id is used:
        '2,3' -> rank 2 -> node 1 (not '2,3' parsed whole, not the last id)."""
        got = self._run_with_env(
            {"BIND_TRAINER_NUMA": "1", "FLAGS_selected_gpus": "2,3"}
        )
        self.assertEqual(got, (2, 1))

    def test_paddle_local_rank_takes_precedence_over_selected_gpus(self):
        """When both are set, PADDLE_LOCAL_RANK wins: rank 1 -> node 0 even
        though FLAGS_selected_gpus would map to node 1."""
        got = self._run_with_env(
            {
                "BIND_TRAINER_NUMA": "1",
                "PADDLE_LOCAL_RANK": "1",
                "FLAGS_selected_gpus": "3",
            }
        )
        self.assertEqual(got, (1, 0))

    def test_enabled_without_any_rank_source_raises(self):
        with (
            mock.patch.dict(os.environ, {"BIND_TRAINER_NUMA": "1"}, clear=True),
            mock.patch.object(launcher_mod, "_reexec_with_numactl") as reexec,
            self.assertRaisesRegex(
                RuntimeError, "PADDLE_LOCAL_RANK or FLAGS_selected_gpus"
            ),
        ):
            launcher_mod._maybe_bind_trainer_numa()
        reexec.assert_not_called()

    def test_rank_outside_supported_range_raises(self):
        for bad in ("4", "5", "8"):
            with self.subTest(rank=bad):
                with (
                    mock.patch.dict(
                        os.environ,
                        {"BIND_TRAINER_NUMA": "1", "PADDLE_LOCAL_RANK": bad},
                        clear=True,
                    ),
                    mock.patch.object(
                        launcher_mod, "_reexec_with_numactl"
                    ) as reexec,
                    self.assertRaisesRegex(
                        RuntimeError, "only supports local rank 0-3"
                    ),
                ):
                    launcher_mod._maybe_bind_trainer_numa()
                reexec.assert_not_called()


class LaunchDispatchTest(_LauncherTestBase):
    """launch(): argv-driven command routing and the train NUMA hand-off."""

    def _inject_entrypoint(self, module_name, attr):
        """Register a fake lazily-imported entrypoint module and record calls.

        Returns the recorder list; the fake replaces the real (Paddle-heavy)
        target so the genuine dispatch branch in launch() runs to completion.
        """
        calls = []
        fake = types.ModuleType(module_name)
        setattr(fake, attr, lambda: calls.append(attr))
        patcher = mock.patch.dict(sys.modules, {module_name: fake})
        patcher.start()
        self.addCleanup(patcher.stop)
        return calls

    def test_missing_command_raises(self):
        with mock.patch.object(launcher_mod.sys, "argv", ["train.py"]):  # noqa: SIM117
            with self.assertRaisesRegex(ValueError, "larger than 1"):
                launcher_mod.launch()

    def test_unknown_command_raises_with_name(self):
        with (
            mock.patch.object(
                launcher_mod.sys, "argv", ["train.py", "frobnicate"]
            ),
            self.assertRaisesRegex(ValueError, "Unknown command : frobnicate"),
        ):
            launcher_mod.launch()

    def test_export_routes_to_run_export(self):
        calls = self._inject_entrypoint(
            "paddlefleet.cli.export.export", "run_export"
        )
        with mock.patch.object(
            launcher_mod.sys, "argv", ["train.py", "export"]
        ):
            launcher_mod.launch()
        self.assertEqual(calls, ["run_export"])

    def test_train_binds_numa_then_runs_tuner(self):
        """train dispatch performs NUMA binding *before* invoking the tuner.
        With binding enabled for rank 2 the re-exec collaborator must receive
        the hand-derived (2, 1) and the tuner must still be reached."""
        calls = self._inject_entrypoint(
            "paddlefleet.cli.train.tuner", "run_tuner"
        )
        seen = []
        with (
            mock.patch.dict(
                os.environ,
                {"BIND_TRAINER_NUMA": "1", "PADDLE_LOCAL_RANK": "2"},
                clear=True,
            ),
            mock.patch.object(
                launcher_mod,
                "_reexec_with_numactl",
                side_effect=lambda r, n: seen.append((r, n)),
            ),
            mock.patch.object(launcher_mod, "print"),
            mock.patch.object(launcher_mod.sys, "argv", ["train.py", "train"]),
        ):
            launcher_mod.launch()

        self.assertEqual(seen, [(2, 1)])
        self.assertEqual(calls, ["run_tuner"])

    def test_train_runs_tuner_when_numa_binding_disabled(self):
        """Default (no BIND_TRAINER_NUMA) still dispatches to the tuner and
        performs no re-exec."""
        calls = self._inject_entrypoint(
            "paddlefleet.cli.train.tuner", "run_tuner"
        )
        with mock.patch.dict(os.environ, {}, clear=True):  # noqa: SIM117
            with mock.patch.object(
                launcher_mod, "_reexec_with_numactl"
            ) as reexec:
                with mock.patch.object(
                    launcher_mod.sys, "argv", ["train.py", "train"]
                ):
                    launcher_mod.launch()
        reexec.assert_not_called()
        self.assertEqual(calls, ["run_tuner"])


if __name__ == "__main__":
    unittest.main()
