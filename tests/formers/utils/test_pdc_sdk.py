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

"""Behavior tests for paddlefleet.utils.pdc_sdk.

The module is a thin wrapper around an external PDC agent binary plus
``tar`` / ``b3sum`` subprocesses.  The wrapper's real job is to build the
correct argv / JSON ``-config`` for those external tools, to guard the
pre-conditions, and to translate exit status into ``PDCErrorCode``.

Isolation strategy: the external process boundary
(``paddlefleet.utils.pdc_sdk.subprocess.run``) is mocked so that no real
binary is executed, while every argv / config the wrapper constructs is
kept real and inspected.  Existence pre-conditions use real temp files and
directories so ``os.path.exists`` runs for real.  Expected values are hand
derived from the documented command layout, never produced by calling the
function under test.
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from paddlefleet.utils.pdc_sdk import (
    FLASH_DEVICE,
    PDCErrorCode,
    PDCErrorMessageMap,
    PDCTools,
    pdc_flash_device_available,
)

PDC_MOD = "paddlefleet.utils.pdc_sdk"


class PDCToolsTestBase(unittest.TestCase):
    """Shared helpers for isolating the subprocess boundary."""

    @staticmethod
    def _completed(returncode=0, stdout="", stderr=""):
        """Stand-in for subprocess.CompletedProcess."""
        return SimpleNamespace(
            returncode=returncode, stdout=stdout, stderr=stderr
        )

    def _make_env(self):
        """Build a PDCTools whose three required binaries/config exist.

        Returns (tools, tmp_dir). ``tmp_dir`` is a real directory that can
        be used for extra fixture files. Cleanup is registered.
        """
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = tmp.name
        agent = os.path.join(d, "agent")
        hash_bin = os.path.join(d, "b3sum")
        train_conf = os.path.join(d, "train.conf")
        for p in (agent, hash_bin, train_conf):
            with open(p, "w") as f:
                f.write("x")
        tools = PDCTools()
        tools._pdc_agent_bin = agent
        tools._hash_sum_bin = hash_bin
        tools._train_config = train_conf
        return tools, d


class GetFileHashNameTest(PDCToolsTestBase):
    """_get_file_hash_name derives the sidecar hash filename."""

    def test_tar_suffix_is_stripped_before_appending(self):
        tools = PDCTools()
        # ".tar" (4 chars) removed, then ".b3sumhash" appended.
        self.assertEqual(
            tools._get_file_hash_name("model_step_100.tar"),
            "model_step_100.b3sumhash",
        )

    def test_non_tar_name_keeps_full_name(self):
        tools = PDCTools()
        # No ".tar" suffix -> nothing stripped.
        self.assertEqual(
            tools._get_file_hash_name("weights.bin"), "weights.bin.b3sumhash"
        )
        self.assertEqual(tools._get_file_hash_name("plain"), "plain.b3sumhash")

    def test_only_trailing_tar_is_stripped(self):
        tools = PDCTools()
        # "a.tar.gz" does not end with ".tar"; must not be truncated.
        self.assertEqual(
            tools._get_file_hash_name("a.tar.gz"), "a.tar.gz.b3sumhash"
        )


class FlashDeviceTest(PDCToolsTestBase):
    """pdc_flash_device_available reflects existence of FLASH_DEVICE."""

    def test_available_when_flash_path_exists(self):
        real_exists = os.path.exists

        def fake_exists(path):
            if path == FLASH_DEVICE:
                return True
            return real_exists(path)

        with mock.patch(f"{PDC_MOD}.os.path.exists", side_effect=fake_exists):
            self.assertTrue(pdc_flash_device_available())

    def test_unavailable_when_flash_path_missing(self):
        real_exists = os.path.exists

        def fake_exists(path):
            if path == FLASH_DEVICE:
                return False
            return real_exists(path)

        with mock.patch(f"{PDC_MOD}.os.path.exists", side_effect=fake_exists):
            self.assertFalse(pdc_flash_device_available())


class ErrorCodeMappingTest(PDCToolsTestBase):
    """PDCErrorMessageMap ties each surfaced code to its message."""

    def test_message_map_has_expected_human_text(self):
        # Independent expectations: the message string a caller would log.
        self.assertEqual(PDCErrorMessageMap[PDCErrorCode.Success], "success")
        self.assertEqual(
            PDCErrorMessageMap[PDCErrorCode.AFSToolsNotExist],
            "afs tools not exist",
        )
        self.assertEqual(
            PDCErrorMessageMap[PDCErrorCode.TrainConfigNotExist],
            "train config not exist",
        )
        self.assertEqual(
            PDCErrorMessageMap[PDCErrorCode.CommandTimeout],
            "pdc agent command timeout",
        )
        self.assertEqual(
            PDCErrorMessageMap[PDCErrorCode.CopyTreeFailed],
            "copy directory failed",
        )


class PreCheckTest(PDCToolsTestBase):
    """_pre_check gates on the three required filesystem entries."""

    def test_all_present_returns_success(self):
        tools, _ = self._make_env()
        self.assertEqual(tools._pre_check(), PDCErrorCode.Success)

    def test_missing_agent_binary_reports_afs_tools_missing(self):
        tools, d = self._make_env()
        tools._pdc_agent_bin = os.path.join(d, "no_such_agent")
        self.assertEqual(tools._pre_check(), PDCErrorCode.AFSToolsNotExist)

    def test_missing_hash_binary_reports_afs_tools_missing(self):
        tools, d = self._make_env()
        tools._hash_sum_bin = os.path.join(d, "no_such_b3sum")
        self.assertEqual(tools._pre_check(), PDCErrorCode.AFSToolsNotExist)

    def test_missing_train_config_reports_train_config_missing(self):
        tools, d = self._make_env()
        tools._train_config = os.path.join(d, "no_such_train.conf")
        self.assertEqual(tools._pre_check(), PDCErrorCode.TrainConfigNotExist)


class ExecCmdTest(PDCToolsTestBase):
    """_exec_cmd forwards argv to subprocess and maps exit status."""

    def test_zero_exit_returns_stdout_and_success(self):
        tools = PDCTools()
        argv = ["/bin/echo", "hello"]
        with mock.patch(f"{PDC_MOD}.subprocess.run") as run:
            run.return_value = self._completed(
                returncode=0, stdout="captured-out", stderr=""
            )
            stdout, code = tools._exec_cmd(argv)

        run.assert_called_once()
        called_args, called_kwargs = run.call_args
        # The exact argv is forwarded unchanged.
        self.assertEqual(called_args[0], argv)
        # Output is captured as text so stdout can be parsed downstream.
        self.assertTrue(called_kwargs["capture_output"])
        self.assertTrue(called_kwargs["text"])
        self.assertEqual(stdout, "captured-out")
        self.assertEqual(code, PDCErrorCode.Success)

    def test_nonzero_exit_maps_to_command_fail(self):
        tools = PDCTools()
        with mock.patch(f"{PDC_MOD}.subprocess.run") as run:
            run.return_value = self._completed(
                returncode=7, stdout="partial", stderr="boom"
            )
            stdout, code = tools._exec_cmd(["/bin/false"])

        # Non-zero exit code is translated to CommandFail, stdout still returned.
        self.assertEqual(code, PDCErrorCode.CommandFail)
        self.assertEqual(stdout, "partial")

    def test_subprocess_error_is_reraised(self):
        tools = PDCTools()
        with mock.patch(
            f"{PDC_MOD}.subprocess.run", side_effect=OSError("nope")
        ):
            with self.assertRaises(Exception) as ctx:
                tools._exec_cmd(["/bin/echo"])
        self.assertIn("nope", str(ctx.exception))


class CalculateHashTest(PDCToolsTestBase):
    """_calculate_hash builds the b3sum argv and parses the first token."""

    def test_builds_argv_and_extracts_hash_token(self):
        tools, _ = self._make_env()
        file_path = "/tmp/archive.tar"
        # b3sum prints "<hash>  <filename>"; wrapper takes the first token.
        with mock.patch(f"{PDC_MOD}.subprocess.run") as run:
            run.return_value = self._completed(
                returncode=0, stdout="abc123def456  archive.tar\n"
            )
            digest, code = tools._calculate_hash(file_path)

        argv = run.call_args.args[0]
        self.assertEqual(
            argv,
            [tools._hash_sum_bin, "--num-threads", "16", file_path],
        )
        self.assertEqual(code, PDCErrorCode.Success)
        self.assertEqual(digest, "abc123def456")

    def test_command_failure_yields_calculate_hash_fail(self):
        tools, _ = self._make_env()
        with mock.patch(f"{PDC_MOD}.subprocess.run") as run:
            run.return_value = self._completed(returncode=1, stdout="")
            digest, code = tools._calculate_hash("/tmp/archive.tar")
        # Failure path returns empty hash and the dedicated error code.
        self.assertEqual(digest, "")
        self.assertEqual(code, PDCErrorCode.CalculateHashFail)


class TarFileTest(PDCToolsTestBase):
    """_tar_file builds `tar -cf <target> -C <source> .`."""

    def test_builds_create_argv_with_change_dir(self):
        tools, d = self._make_env()
        source = os.path.join(d, "src_dir")
        os.makedirs(source)
        target = os.path.join(d, "out.tar")  # does not exist yet
        with mock.patch(f"{PDC_MOD}.subprocess.run") as run:
            run.return_value = self._completed(returncode=0)
            code = tools._tar_file(source, target)

        argv = run.call_args.args[0]
        self.assertEqual(
            argv, [tools._tar_bin, "-cf", target, "-C", source, "."]
        )
        self.assertEqual(code, PDCErrorCode.Success)

    def test_missing_source_returns_local_path_not_exist(self):
        tools, d = self._make_env()
        source = os.path.join(d, "missing_src")
        target = os.path.join(d, "out.tar")
        with mock.patch(f"{PDC_MOD}.subprocess.run") as run:
            code = tools._tar_file(source, target)
        # Guard rejects before any subprocess is spawned.
        run.assert_not_called()
        self.assertEqual(code, PDCErrorCode.LocalPathNotExist)


class UntarFileTest(PDCToolsTestBase):
    """_untar_file builds `tar -xf <source> -C <target>`."""

    def test_builds_extract_argv(self):
        tools, d = self._make_env()
        source = os.path.join(d, "in.tar")
        with open(source, "w") as f:
            f.write("data")
        target = os.path.join(d, "extract_here")  # created by makedirs
        with mock.patch(f"{PDC_MOD}.subprocess.run") as run:
            run.return_value = self._completed(returncode=0)
            code = tools._untar_file(source, target)

        argv = run.call_args.args[0]
        self.assertEqual(argv, [tools._tar_bin, "-xf", source, "-C", target])
        self.assertEqual(code, PDCErrorCode.Success)
        # Target directory is created as a side effect.
        self.assertTrue(os.path.isdir(target))

    def test_missing_source_returns_local_path_not_exist(self):
        tools, d = self._make_env()
        source = os.path.join(d, "absent.tar")
        target = os.path.join(d, "extract_here")
        with mock.patch(f"{PDC_MOD}.subprocess.run") as run:
            code = tools._untar_file(source, target)
        run.assert_not_called()
        self.assertEqual(code, PDCErrorCode.LocalPathNotExist)


class UploadFileTest(PDCToolsTestBase):
    """_upload_file builds the agent `upload` command with JSON config."""

    def test_builds_agent_argv_and_json_config(self):
        tools, d = self._make_env()
        local = os.path.join(d, "payload.tar")
        with open(local, "w") as f:
            f.write("payload")
        remote = "afs://bucket/dir/payload.tar"
        with mock.patch(f"{PDC_MOD}.subprocess.run") as run:
            run.return_value = self._completed(returncode=0)
            code = tools._upload_file(local, remote)

        argv = run.call_args.args[0]
        self.assertEqual(
            argv[:6],
            [
                tools._pdc_agent_bin,
                "-mode",
                "command",
                "-type",
                "upload",
                "-config",
            ],
        )
        # Config carries exactly the remote/local pair the caller passed.
        self.assertEqual(
            json.loads(argv[6]),
            {"remote_path": remote, "local_path": local},
        )
        self.assertEqual(code, PDCErrorCode.Success)

    def test_missing_local_file_returns_local_path_not_exist(self):
        tools, d = self._make_env()
        local = os.path.join(d, "nope.tar")
        with mock.patch(f"{PDC_MOD}.subprocess.run") as run:
            code = tools._upload_file(local, "afs://bucket/x.tar")
        run.assert_not_called()
        self.assertEqual(code, PDCErrorCode.LocalPathNotExist)


class DownloadFileTest(PDCToolsTestBase):
    """_download_file builds the agent `download` command with JSON config."""

    def test_builds_agent_argv_and_json_config(self):
        tools, d = self._make_env()
        remote = "afs://bucket/dir/data.tar"
        local = os.path.join(d, "local_dst.tar")  # does not exist -> no backup
        with mock.patch(f"{PDC_MOD}.subprocess.run") as run:
            run.return_value = self._completed(returncode=0)
            code = tools._download_file(remote, local)

        argv = run.call_args.args[0]
        self.assertEqual(
            argv[:6],
            [
                tools._pdc_agent_bin,
                "-mode",
                "command",
                "-type",
                "download",
                "-config",
            ],
        )
        self.assertEqual(
            json.loads(argv[6]),
            {"remote_path": remote, "local_path": local},
        )
        self.assertEqual(code, PDCErrorCode.Success)

    def test_existing_local_file_is_backed_up_before_download(self):
        tools, d = self._make_env()
        remote = "afs://bucket/dir/data.tar"
        local = os.path.join(d, "existing.tar")
        with open(local, "w") as f:
            f.write("stale")
        with mock.patch(f"{PDC_MOD}.subprocess.run") as run:
            run.return_value = self._completed(returncode=0)
            tools._download_file(remote, local)
        # Pre-existing destination is renamed to "<path>.old" to avoid clobber.
        self.assertTrue(os.path.exists(local + ".old"))
        with open(local + ".old") as f:
            self.assertEqual(f.read(), "stale")


class DownloadCheckpointImplTest(PDCToolsTestBase):
    """_pdc_download_checkpoint_impl builds a download_checkpoint command."""

    def test_builds_argv_with_step_config(self):
        tools, _ = self._make_env()
        with mock.patch(f"{PDC_MOD}.subprocess.run") as run:
            run.return_value = self._completed(returncode=0)
            code = tools._pdc_download_checkpoint_impl(step=42)

        argv = run.call_args.args[0]
        self.assertEqual(
            argv[:6],
            [
                tools._pdc_agent_bin,
                "-mode",
                "command",
                "-type",
                "download_checkpoint",
                "-config",
            ],
        )
        # The only config field is the requested resume step.
        self.assertEqual(json.loads(argv[6]), {"download_step": 42})
        self.assertEqual(code, PDCErrorCode.Success)

    def test_precheck_failure_short_circuits(self):
        tools, d = self._make_env()
        tools._pdc_agent_bin = os.path.join(d, "gone_agent")
        with mock.patch(f"{PDC_MOD}.subprocess.run") as run:
            code = tools._pdc_download_checkpoint_impl(step=1)
        # Missing agent binary is detected before any command runs.
        run.assert_not_called()
        self.assertEqual(code, PDCErrorCode.AFSToolsNotExist)


class ChecksumCommandTest(PDCToolsTestBase):
    """generateSum / checkSum commands carry a -path argument."""

    def test_generate_dir_checksum_builds_generate_sum_argv(self):
        tools, d = self._make_env()
        path = os.path.join(d, "ckpt")
        os.makedirs(path)
        with mock.patch(f"{PDC_MOD}.subprocess.run") as run:
            run.return_value = self._completed(returncode=0)
            code = tools.pdc_generate_dir_checksum(path)

        argv = run.call_args.args[0]
        self.assertEqual(
            argv,
            [
                tools._pdc_agent_bin,
                "-mode",
                "command",
                "-type",
                "generateSum",
                "-path",
                path,
            ],
        )
        self.assertEqual(code, PDCErrorCode.Success)

    def test_generate_dir_checksum_missing_path_returns_command_fail(self):
        tools, d = self._make_env()
        with mock.patch(f"{PDC_MOD}.subprocess.run") as run:
            code = tools.pdc_generate_dir_checksum(os.path.join(d, "absent"))
        run.assert_not_called()
        self.assertEqual(code, PDCErrorCode.CommandFail)

    def test_flash_do_check_builds_check_sum_argv(self):
        tools, d = self._make_env()
        path = os.path.join(d, "flash_ckpt")
        os.makedirs(path)
        with mock.patch(f"{PDC_MOD}.subprocess.run") as run:
            run.return_value = self._completed(returncode=0)
            code = tools.pdc_flash_do_check(path)

        argv = run.call_args.args[0]
        self.assertEqual(
            argv,
            [
                tools._pdc_agent_bin,
                "-mode",
                "command",
                "-type",
                "checkSum",
                "-path",
                path,
            ],
        )
        self.assertEqual(code, PDCErrorCode.Success)


class PdcUploadGuardTest(PDCToolsTestBase):
    """pdc_upload validates environment and arguments before working."""

    def test_precheck_failure_is_propagated(self):
        tools, d = self._make_env()
        tools._train_config = os.path.join(d, "missing.conf")
        local = os.path.join(d, "data.tar")
        with open(local, "w") as f:
            f.write("x")
        with mock.patch(f"{PDC_MOD}.subprocess.run") as run:
            code = tools.pdc_upload("afs://b/data.tar", local)
        run.assert_not_called()
        self.assertEqual(code, PDCErrorCode.TrainConfigNotExist)

    def test_missing_local_path_rejected(self):
        tools, d = self._make_env()
        local = os.path.join(d, "does_not_exist.tar")
        with mock.patch(f"{PDC_MOD}.subprocess.run") as run:
            code = tools.pdc_upload("afs://b/data.tar", local)
        run.assert_not_called()
        self.assertEqual(code, PDCErrorCode.LocalPathNotExist)

    def test_non_tar_remote_path_rejected(self):
        tools, d = self._make_env()
        local = os.path.join(d, "data.bin")
        with open(local, "w") as f:
            f.write("x")
        with mock.patch(f"{PDC_MOD}.subprocess.run") as run:
            # Remote does not end with ".tar": invalid, no work performed.
            code = tools.pdc_upload("afs://b/data.bin", local)
        run.assert_not_called()
        self.assertEqual(code, PDCErrorCode.InvalidArgument)


class PdcDownloadImplGuardTest(PDCToolsTestBase):
    """_pdc_download_impl guards local collision and remote extension."""

    def test_existing_local_path_returns_local_path_exist(self):
        tools, d = self._make_env()
        local = os.path.join(d, "already_here")
        os.makedirs(local)
        with mock.patch(f"{PDC_MOD}.subprocess.run") as run:
            code = tools._pdc_download_impl("afs://b/x.tar", local)
        run.assert_not_called()
        self.assertEqual(code, PDCErrorCode.LocalPathExist)

    def test_non_tar_remote_path_rejected(self):
        tools, d = self._make_env()
        local = os.path.join(d, "target_dir")  # absent
        with mock.patch(f"{PDC_MOD}.subprocess.run") as run:
            code = tools._pdc_download_impl("afs://b/x.bin", local)
        run.assert_not_called()
        self.assertEqual(code, PDCErrorCode.InvalidArgument)


class PdcDownloadThreadWrapperTest(PDCToolsTestBase):
    """pdc_download runs the impl on a worker thread and relays its result.

    Here the per-object ``_pdc_download_impl`` is a separately tested unit;
    isolating it lets these cases exercise only the thread/timeout wrapper
    and its result-vs-exception translation.
    """

    def test_worker_success_result_is_returned(self):
        tools, _ = self._make_env()
        tools._pdc_download_impl = mock.Mock(return_value=PDCErrorCode.Success)
        code = tools.pdc_download("afs://b/x.tar", "/tmp/dst", timeout=5)
        tools._pdc_download_impl.assert_called_once_with(
            "afs://b/x.tar", "/tmp/dst"
        )
        self.assertEqual(code, PDCErrorCode.Success)

    def test_worker_non_success_result_is_relayed_verbatim(self):
        tools, _ = self._make_env()
        tools._pdc_download_impl = mock.Mock(
            return_value=PDCErrorCode.CalculateHashFail
        )
        code = tools.pdc_download("afs://b/x.tar", "/tmp/dst", timeout=5)
        self.assertEqual(code, PDCErrorCode.CalculateHashFail)

    def test_worker_exception_becomes_unknown_error(self):
        tools, _ = self._make_env()
        tools._pdc_download_impl = mock.Mock(
            side_effect=RuntimeError("disk on fire")
        )
        # The worker stores str(exc); the wrapper maps that to UnknownError
        # rather than crashing or leaking the raw message as the status.
        code = tools.pdc_download("afs://b/x.tar", "/tmp/dst", timeout=5)
        self.assertEqual(code, PDCErrorCode.UnknownError)


class PdcDownloadCheckpointThreadWrapperTest(PDCToolsTestBase):
    """pdc_download_checkpoint relays the impl result via its worker thread."""

    def test_success_result_is_returned(self):
        tools, _ = self._make_env()
        tools._pdc_download_checkpoint_impl = mock.Mock(
            return_value=PDCErrorCode.Success
        )
        code = tools.pdc_download_checkpoint(resume_step=9, timeout=5)
        tools._pdc_download_checkpoint_impl.assert_called_once_with(9)
        self.assertEqual(code, PDCErrorCode.Success)

    def test_exception_becomes_unknown_error(self):
        tools, _ = self._make_env()
        tools._pdc_download_checkpoint_impl = mock.Mock(
            side_effect=ValueError("bad step")
        )
        code = tools.pdc_download_checkpoint(resume_step=9, timeout=5)
        self.assertEqual(code, PDCErrorCode.UnknownError)


class PdcBackupToFlashDeviceTest(PDCToolsTestBase):
    """pdc_backup_to_flash_device: checksum, copy, verify.

    NOTE: this test asserts the CORRECT contract (a successful copy plus a
    passing checksum should return PDCErrorCode.Success). It is marked
    expectedFailure because production calls ``shutil.copy_tree`` at
    pdc_sdk.py:604, which does not exist (the real API is
    ``shutil.copytree``; ``copy_tree`` lives in ``distutils.dir_util``).
    The AttributeError is swallowed by the broad ``except Exception`` and
    the method always returns CopyTreeFailed, so a genuine backup can never
    succeed. Production is intentionally left unmodified.
    """

    def test_missing_persistent_path_returns_local_path_not_exist(self):
        tools, d = self._make_env()
        code = tools.pdc_backup_to_flash_device(
            os.path.join(d, "no_such_src"), os.path.join(d, "flash")
        )
        self.assertEqual(code, PDCErrorCode.LocalPathNotExist)

    @unittest.expectedFailure
    def test_successful_backup_returns_success(self):
        tools, d = self._make_env()
        persistent = os.path.join(d, "persistent")
        os.makedirs(persistent)
        with open(os.path.join(persistent, "shard.bin"), "w") as f:
            f.write("weights")
        flash = os.path.join(d, "flash_dst")
        # Agent subprocesses (generateSum / checkSum) succeed; the only
        # failure would be the copy step. Correct behavior: Success.
        with mock.patch(f"{PDC_MOD}.subprocess.run") as run:
            run.return_value = self._completed(returncode=0)
            code = tools.pdc_backup_to_flash_device(persistent, flash)
        self.assertEqual(code, PDCErrorCode.Success)


if __name__ == "__main__":
    unittest.main()
