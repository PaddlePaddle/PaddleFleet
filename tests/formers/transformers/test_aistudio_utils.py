# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

"""Genuine behavior tests for paddlefleet.transformers.aistudio_utils.

CPU-only / 无卡. This module (配置与运行基础设施: 本地权重/下载入口) has no CPU
math; the only external dependency is the network downloader
``aistudio_sdk.file_download.model_file_download`` (aliased as ``download`` in
the module). We keep the code-under-test (``aistudio_download`` and
``_add_subfolder``) REAL and only isolate that network collaborator with a
capturing fake, then assert on the arguments it actually received AND on the
value that flows back to the caller (antipattern type 3: not assert_called
only). Expected path/kwargs/messages are hand-derived independently.
"""

import unittest
from unittest import mock

from requests import HTTPError

from paddlefleet.transformers.aistudio_utils import (
    EntryNotFoundError,
    UnauthorizedError,
    _add_subfolder,
    aistudio_download,
)

_DOWNLOAD_ATTR = "paddlefleet.transformers.aistudio_utils.download"


def _capturing_download(return_value="/local/store/downloaded_file"):
    """A fake for the external ``download`` collaborator.

    Records the keyword arguments it was invoked with and returns a
    distinguishable sentinel so the test can prove the real
    ``aistudio_download`` propagates the downloader's return value.
    """
    captured = {}

    def fake(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return return_value

    return captured, fake


def _raising_download(exc):
    def fake(**kwargs):
        raise exc

    return fake


class AddSubfolderTest(unittest.TestCase):
    """_add_subfolder joins subfolder before the weights name, or is a no-op."""

    def test_prepends_subfolder_in_order(self):
        # Distinguishable names so a reversed join (name/subfolder) is caught.
        self.assertEqual(
            _add_subfolder("weights.bin", "ckpt"), "ckpt/weights.bin"
        )

    def test_nested_subfolder_prefix(self):
        self.assertEqual(
            _add_subfolder("model.safetensors", "a/b"),
            "a/b/model.safetensors",
        )

    def test_none_subfolder_returns_unchanged(self):
        self.assertEqual(
            _add_subfolder("model.safetensors", None), "model.safetensors"
        )

    def test_empty_subfolder_returns_unchanged(self):
        self.assertEqual(
            _add_subfolder("model.safetensors", ""), "model.safetensors"
        )


class CustomErrorsTest(unittest.TestCase):
    """The two public exception classes are distinct Exception subclasses."""

    def test_unauthorized_error_is_exception_and_keeps_message(self):
        self.assertTrue(issubclass(UnauthorizedError, Exception))
        err = UnauthorizedError("no token")
        self.assertIsInstance(err, Exception)
        self.assertEqual(str(err), "no token")

    def test_entry_not_found_error_is_exception_and_keeps_message(self):
        self.assertTrue(issubclass(EntryNotFoundError, Exception))
        self.assertEqual(str(EntryNotFoundError("missing")), "missing")

    def test_the_two_error_types_are_distinct(self):
        self.assertIsNot(UnauthorizedError, EntryNotFoundError)
        self.assertFalse(issubclass(UnauthorizedError, EntryNotFoundError))
        self.assertFalse(issubclass(EntryNotFoundError, UnauthorizedError))


class AistudioDownloadSuccessTest(unittest.TestCase):
    """Argument construction + return-value propagation on the success path."""

    def test_default_revision_and_plain_file_path(self):
        captured, fake = _capturing_download("/local/a")
        with mock.patch(_DOWNLOAD_ATTR, new=fake):
            result = aistudio_download("my/repo", filename="model.safetensors")
        # revision is never None inside download_kwargs: it defaults to "master".
        self.assertEqual(
            captured,
            {
                "repo_id": "my/repo",
                "file_path": "model.safetensors",
                "revision": "master",
            },
        )
        # aistudio_download returns exactly what download returned.
        self.assertEqual(result, "/local/a")

    def test_subfolder_is_applied_to_file_path(self):
        captured, fake = _capturing_download()
        with mock.patch(_DOWNLOAD_ATTR, new=fake):
            aistudio_download("my/repo", filename="model.bin", subfolder="ckpt")
        self.assertEqual(captured["file_path"], "ckpt/model.bin")
        self.assertEqual(captured["repo_id"], "my/repo")
        self.assertEqual(captured["revision"], "master")
        self.assertNotIn("local_dir", captured)

    def test_explicit_revision_passthrough(self):
        captured, fake = _capturing_download()
        with mock.patch(_DOWNLOAD_ATTR, new=fake):
            aistudio_download("my/repo", filename="model.bin", revision="v1.0")
        self.assertEqual(captured["revision"], "v1.0")
        self.assertNotIn("local_dir", captured)

    def test_cache_dir_maps_to_local_dir(self):
        captured, fake = _capturing_download()
        with mock.patch(_DOWNLOAD_ATTR, new=fake):
            aistudio_download(
                "my/repo", filename="model.bin", cache_dir="/tmp/cache"
            )
        self.assertEqual(
            captured,
            {
                "repo_id": "my/repo",
                "file_path": "model.bin",
                "revision": "master",
                "local_dir": "/tmp/cache",
            },
        )

    def test_cache_dir_and_revision_combined(self):
        captured, fake = _capturing_download()
        with mock.patch(_DOWNLOAD_ATTR, new=fake):
            aistudio_download(
                "my/repo",
                filename="model.bin",
                cache_dir="/tmp/cache",
                revision="v2.0",
            )
        self.assertEqual(
            captured,
            {
                "repo_id": "my/repo",
                "file_path": "model.bin",
                "revision": "v2.0",
                "local_dir": "/tmp/cache",
            },
        )

    def test_filename_none_passes_none_file_path(self):
        # subfolder default "" means _add_subfolder(None, "") returns None.
        captured, fake = _capturing_download("/local/z")
        with mock.patch(_DOWNLOAD_ATTR, new=fake):
            result = aistudio_download("my/repo", filename=None)
        self.assertEqual(
            captured,
            {"repo_id": "my/repo", "file_path": None, "revision": "master"},
        )
        self.assertEqual(result, "/local/z")

    def test_extra_kwargs_are_swallowed_not_forwarded(self):
        # **kwargs that are not revision/cache_dir must not reach download.
        captured, fake = _capturing_download()
        with mock.patch(_DOWNLOAD_ATTR, new=fake):
            aistudio_download(
                "my/repo", filename="model.bin", force_download=True
            )
        self.assertEqual(
            captured,
            {
                "repo_id": "my/repo",
                "file_path": "model.bin",
                "revision": "master",
            },
        )
        self.assertNotIn("force_download", captured)


class AistudioDownloadErrorMappingTest(unittest.TestCase):
    """Each downloader failure is re-raised as EnvironmentError with a message
    that is distinguishable per branch (order: ValueError, EntryNotFoundError,
    HTTPError, generic Exception)."""

    def test_value_error_maps_to_cached_files_message(self):
        with (
            mock.patch(
                _DOWNLOAD_ATTR, new=_raising_download(ValueError("boom"))
            ),
            self.assertRaises(EnvironmentError) as ctx,
        ):
            aistudio_download("my/repo", filename="model.safetensors")
        msg = str(ctx.exception)
        self.assertIn("Cannot find model.safetensors", msg)
        self.assertIn("cached files", msg)
        self.assertIn("my/repo", msg)
        # Distinct from the EntryNotFoundError branch's phrasing.
        self.assertNotIn("the requested file", msg)

    def test_entry_not_found_maps_to_requested_file_message(self):
        with (
            mock.patch(
                _DOWNLOAD_ATTR,
                new=_raising_download(EntryNotFoundError("nope")),
            ),
            self.assertRaises(EnvironmentError) as ctx,
        ):
            aistudio_download("my/repo", filename="model.safetensors")
        msg = str(ctx.exception)
        # Distinguishing phrase for this branch; catches removal of the
        # `except EntryNotFoundError` clause (which would fall through to the
        # generic "Please make sure" branch instead).
        self.assertIn("Cannot find the requested file model.safetensors", msg)
        self.assertIn("my/repo", msg)
        self.assertNotIn("Please make sure", msg)

    def test_http_error_embeds_original_error_and_repo(self):
        with (
            mock.patch(
                _DOWNLOAD_ATTR,
                new=_raising_download(HTTPError("503 service down")),
            ),
            self.assertRaises(EnvironmentError) as ctx,
        ):
            aistudio_download("my/repo", filename="model.safetensors")
        msg = str(ctx.exception)
        self.assertIn("specific connection error", msg)
        self.assertIn("my/repo", msg)
        # The original error text is interpolated into the message.
        self.assertIn("503 service down", msg)

    def test_generic_exception_maps_to_please_make_sure_message(self):
        with (
            mock.patch(
                _DOWNLOAD_ATTR,
                new=_raising_download(RuntimeError("weird")),
            ),
            self.assertRaises(EnvironmentError) as ctx,
        ):
            aistudio_download("my/repo", filename="model.safetensors")
        msg = str(ctx.exception)
        self.assertIn("Please make sure the model.safetensors", msg)
        self.assertIn("my/repo", msg)
        # Distinct from the EntryNotFoundError branch.
        self.assertNotIn("the requested file", msg)

    def test_entry_not_found_and_generic_produce_different_messages(self):
        with (
            mock.patch(
                _DOWNLOAD_ATTR,
                new=_raising_download(EntryNotFoundError("x")),
            ),
            self.assertRaises(EnvironmentError) as ctx_entry,
        ):
            aistudio_download("my/repo", filename="model.bin")
        with (
            mock.patch(
                _DOWNLOAD_ATTR, new=_raising_download(RuntimeError("x"))
            ),
            self.assertRaises(EnvironmentError) as ctx_generic,
        ):
            aistudio_download("my/repo", filename="model.bin")
        self.assertNotEqual(
            str(ctx_entry.exception), str(ctx_generic.exception)
        )

    def test_error_message_uses_subfolder_prefixed_filename(self):
        # Proves _add_subfolder runs before the download call and its result
        # is used in the error text (not just the raw filename).
        with (
            mock.patch(
                _DOWNLOAD_ATTR, new=_raising_download(ValueError("boom"))
            ),
            self.assertRaises(EnvironmentError) as ctx,
        ):
            aistudio_download("my/repo", filename="model.bin", subfolder="ckpt")
        self.assertIn("ckpt/model.bin", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
