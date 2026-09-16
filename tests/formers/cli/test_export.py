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

"""Behavior tests for ``paddlefleet.cli.export.export``.

This module exposes three units of CLI export logic:

* ``check_download_repo`` decides whether a model reference is a local path
  (returned unchanged, with a torch-dtype notice printed only when the local
  ``config.json`` declares ``torch_dtype``) or a remote repo id (resolved via
  ``check_repo`` using an explicit ``download_hub`` argument, else the
  ``DOWNLOAD_SOURCE`` env var, else the ``"huggingface"`` default).
* ``logger_merge_config`` writes a filtered view of a merge config to the
  DEBUG logger: for LoRA merges only ``lora_model_path`` / ``base_model_path``
  are shown; for plain merges every attribute EXCEPT
  ``{model_path_str, device, tensor_type, merge_preifx}`` is shown.
* ``run_export`` orchestrates the export: it demands a valid checkpoint under
  ``output_dir`` (else ``FileNotFoundError``) and refuses non-LoRA exports
  (``ValueError``).

The tests drive the real functions. Only genuinely external collaborators are
substituted: the network downloader ``check_repo`` (given a distinguishable
marker whose identity and call args are both checked), the module ``logger``
(a real recorder whose exact emitted lines are compared), and the argument
builders ``read_args`` / ``get_export_args`` (so the real checkpoint-discovery
and guard branches execute against real temp dirs). Expected values are
hand-derived from the source, not read back from the code under test.

The whole ``paddlefleet`` package imports ``paddle`` at import time, so the
tests skip when Paddle (and therefore the package) is unavailable; they run
for real on any CPU where Paddle is installed.
"""

import contextlib
import io
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    from paddlefleet.cli.export import export as export_mod

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet not installed
    export_mod = None
    _IMPORT_ERROR = exc


class CheckDownloadRepoTest(unittest.TestCase):
    """check_download_repo: local passthrough vs. remote hub resolution."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"paddlefleet/paddle unavailable: {_IMPORT_ERROR!r}")

    def test_local_dir_with_torch_dtype_returns_path_and_notifies(self):
        """A local dir whose config.json has torch_dtype: path returned
        unchanged and the exact torch-dtype notice printed once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = os.path.join(tmpdir, "config.json")
            with open(cfg, "w", encoding="utf-8") as f:
                f.write('{"torch_dtype": "float32", "model_type": "llama"}')

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                result = export_mod.check_download_repo(tmpdir)

            # Identity: the local reference is returned unchanged, not rebuilt.
            self.assertIs(result, tmpdir)
            self.assertEqual(
                buf.getvalue(),
                "Loading local model which contains torch dtype.\n",
            )

    def test_local_dir_without_torch_dtype_is_silent(self):
        """A local dir whose config.json omits torch_dtype: path returned
        unchanged and nothing printed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = os.path.join(tmpdir, "config.json")
            with open(cfg, "w", encoding="utf-8") as f:
                f.write('{"model_type": "llama"}')

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                result = export_mod.check_download_repo(tmpdir)

            self.assertIs(result, tmpdir)
            self.assertEqual(buf.getvalue(), "")

    def test_remote_repo_uses_explicit_hub(self):
        """A non-local repo id resolves through check_repo with the explicit
        download_hub, and the resolved path is what gets returned."""
        marker = "/cached/models/explicit-hub-result"
        with mock.patch.object(
            export_mod, "check_repo", return_value=marker
        ) as check_repo:
            result = export_mod.check_download_repo(
                "org/some-repo", download_hub="huggingface"
            )

        # Real resolution result is propagated out (not the repo id).
        self.assertIs(result, marker)
        check_repo.assert_called_once_with("org/some-repo", "huggingface")

    def test_remote_repo_uses_env_download_source(self):
        """With no explicit hub, DOWNLOAD_SOURCE selects the hub passed to
        check_repo."""
        marker = "/cached/models/env-result"
        with (
            mock.patch.object(
                export_mod, "check_repo", return_value=marker
            ) as check_repo,
            mock.patch.dict(
                os.environ, {"DOWNLOAD_SOURCE": "modelscope"}, clear=False
            ),
        ):
            result = export_mod.check_download_repo("org/some-repo")

        self.assertIs(result, marker)
        check_repo.assert_called_once_with("org/some-repo", "modelscope")

    def test_remote_repo_defaults_to_huggingface(self):
        """With neither explicit hub nor DOWNLOAD_SOURCE set, the hub defaults
        to 'huggingface'."""
        marker = "/cached/models/default-result"
        env_without_source = {
            k: v for k, v in os.environ.items() if k != "DOWNLOAD_SOURCE"
        }
        with (
            mock.patch.object(
                export_mod, "check_repo", return_value=marker
            ) as check_repo,
            mock.patch.dict(os.environ, env_without_source, clear=True),
        ):
            result = export_mod.check_download_repo("org/some-repo")

        self.assertIs(result, marker)
        check_repo.assert_called_once_with("org/some-repo", "huggingface")


class LoggerMergeConfigTest(unittest.TestCase):
    """logger_merge_config: which config fields are emitted at DEBUG."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"paddlefleet/paddle unavailable: {_IMPORT_ERROR!r}")

    @staticmethod
    def _debug_lines(mock_logger):
        return [call[0][0] for call in mock_logger.debug.call_args_list]

    def test_lora_merge_emits_only_lora_paths(self):
        """lora_merge=True: a centered 'LoRA Merge Info' header followed by
        exactly the lora_model_path and base_model_path lines; output_path is
        not shown."""
        cfg = SimpleNamespace(
            lora_model_path="/path/to/lora",
            base_model_path="/path/to/base",
            output_path="/path/to/output",
        )
        with mock.patch.object(export_mod, "logger") as mock_logger:
            export_mod.logger_merge_config(cfg, lora_merge=True)

        expected = [
            "LoRA Merge Info".center(40),
            f"{'lora_model_path':30}: /path/to/lora",
            f"{'base_model_path':30}: /path/to/base",
        ]
        self.assertEqual(self._debug_lines(mock_logger), expected)

    def test_plain_merge_omits_excluded_fields(self):
        """lora_merge=False: a centered 'Mergekit Config Info' header, then
        every field except {model_path_str, device, tensor_type,
        merge_preifx}, preserving declaration order."""
        cfg = SimpleNamespace(
            model_path_str="/path/model",
            device="gpu",
            tensor_type="fp16",
            merge_preifx="merged",
            output_path="/path/to/output",
            save_safetensors=True,
        )
        with mock.patch.object(export_mod, "logger") as mock_logger:
            export_mod.logger_merge_config(cfg, lora_merge=False)

        lines = self._debug_lines(mock_logger)
        expected = [
            "Mergekit Config Info".center(40),
            f"{'output_path':30}: /path/to/output",
            f"{'save_safetensors':30}: True",
        ]
        self.assertEqual(lines, expected)
        # The four excluded keys must not appear on any emitted line.
        joined = "\n".join(lines)
        for excluded in (
            "model_path_str",
            "device",
            "tensor_type",
            "merge_preifx",
        ):
            self.assertNotIn(excluded, joined)


class RunExportTest(unittest.TestCase):
    """run_export: checkpoint-presence and LoRA-only guard contracts."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"paddlefleet/paddle unavailable: {_IMPORT_ERROR!r}")

    def _patch_arg_builders(self, model_args, finetuning_args):
        """Isolate only the argument builders and the device side effect, so
        the real checkpoint-discovery / guard logic runs. Returns a context
        manager."""
        export_tuple = (
            model_args,
            SimpleNamespace(),  # data_args (unused before the guards)
            SimpleNamespace(),  # generating_args (unused before the guards)
            finetuning_args,
            SimpleNamespace(),  # export_args (unused before the guards)
        )
        return (
            mock.patch.object(export_mod, "read_args", side_effect=lambda a: a),
            mock.patch.object(
                export_mod, "get_export_args", return_value=export_tuple
            ),
            mock.patch.object(export_mod.paddle, "set_device"),
        )

    def test_missing_checkpoint_raises_with_offending_path(self):
        """A non-existent output_dir yields no checkpoint, so run_export
        raises FileNotFoundError naming that exact directory."""
        missing = "/nonexistent/export/output/dir-xyz-12345"
        self.assertFalse(os.path.isdir(missing))  # real filesystem state

        model_args = SimpleNamespace(lora=True)
        finetuning_args = SimpleNamespace(output_dir=missing, device="cpu")
        p_read, p_get, p_dev = self._patch_arg_builders(
            model_args, finetuning_args
        )
        with p_read, p_get, p_dev:  # noqa: SIM117
            with self.assertRaises(FileNotFoundError) as ctx:
                export_mod.run_export({"output_dir": missing})

        message = str(ctx.exception)
        self.assertIn("No valid checkpoint", message)
        self.assertIn(missing, message)

    def test_valid_checkpoint_but_non_lora_raises_value_error(self):
        """A real model dir (contains a .safetensors file) is accepted as the
        checkpoint, but a non-LoRA export is rejected with ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Real weight file so is_valid_model_dir returns True for real.
            with open(os.path.join(tmpdir, "model.safetensors"), "wb") as f:
                f.write(b"\x00")

            model_args = SimpleNamespace(lora=False)
            finetuning_args = SimpleNamespace(output_dir=tmpdir, device="cpu")
            p_read, p_get, p_dev = self._patch_arg_builders(
                model_args, finetuning_args
            )
            with p_read, p_get, p_dev:  # noqa: SIM117
                with self.assertRaises(ValueError) as ctx:
                    export_mod.run_export({"output_dir": tmpdir})

        self.assertIn("Only support merge lora checkpoint", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
