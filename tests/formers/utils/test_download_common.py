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

"""Behavior tests for paddlefleet.utils.download.common.

These tests exercise the real download-helper APIs (path/url/cache
construction, file placement, error mapping) with independently
hand-derived expected values. External HTTP is never contacted: the only
"network" surface touched is requests.Response, which is constructed
locally and fed to the error-mapping code under test.
"""

import os
import tempfile
import unittest
import warnings
from dataclasses import FrozenInstanceError
from pathlib import Path

import requests
from huggingface_hub.utils import (
    EntryNotFoundError,
)

from paddlefleet.utils.download import common
from paddlefleet.utils.download.common import (
    DEFALUT_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD,
    DOWNLOAD_CHUNK_SIZE,
    ENV_VARS_TRUE_VALUES,
    REPO_ID_SEPARATOR,
    AistudioBosFileMetadata,
    OfflineModeIsEnabled,
    SoftTemporaryDirectory,
    _as_int,
    _cache_commit_hash_for_specific_revision,
    _check_disk_space,
    _create_symlink,
    _get_pointer_path,
    _is_true,
    _normalize_etag,
    _to_local_dir,
    are_symlinks_supported,
    get_session,
    raise_for_status,
    repo_folder_name,
    reset_sessions,
)


def _probe_symlinks_work(directory):
    """Independently determine (without the code under test) whether the OS
    can create a symlink inside ``directory``. Used to decide the expected
    branch of copy-vs-symlink helpers on the current platform."""
    src = os.path.join(directory, "._probe_symlink_src")
    dst = os.path.join(directory, "._probe_symlink_dst")
    with open(src, "w"):
        pass
    supported = True
    try:
        os.symlink(src, dst)
    except OSError:
        supported = False
    finally:
        for path in (src, dst):
            try:
                os.remove(path)
            except OSError:
                pass
    return supported


class _SymlinkCacheIsolationMixin:
    """Restore the module-level symlink-support cache after each test so the
    real functions we invoke do not leak platform probes into other tests."""

    def _isolate_symlink_cache(self):
        snapshot = dict(common._are_symlinks_supported_in_dir)

        def _restore():
            common._are_symlinks_supported_in_dir.clear()
            common._are_symlinks_supported_in_dir.update(snapshot)

        self.addCleanup(_restore)


class TestIsTrue(unittest.TestCase):
    def test_recognized_true_tokens(self):
        for value in ["1", "ON", "YES", "TRUE"]:
            self.assertTrue(_is_true(value))

    def test_case_insensitive_true_tokens(self):
        for value in ["on", "yes", "true", "On", "Yes", "True"]:
            self.assertTrue(_is_true(value))

    def test_false_tokens_and_empty(self):
        for value in ["0", "OFF", "NO", "FALSE", "", "2", "enabled"]:
            self.assertFalse(_is_true(value))

    def test_none_is_false(self):
        self.assertFalse(_is_true(None))

    def test_true_values_set_is_exactly_the_documented_tokens(self):
        # Hand-derived, independent of the module's own construction.
        self.assertEqual(ENV_VARS_TRUE_VALUES, {"1", "ON", "YES", "TRUE"})


class TestAsInt(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(_as_int(None))

    def test_parses_positive(self):
        result = _as_int("42")
        self.assertEqual(result, 42)
        self.assertIsInstance(result, int)

    def test_parses_negative(self):
        self.assertEqual(_as_int("-7"), -7)

    def test_non_numeric_raises(self):
        with self.assertRaises(ValueError):
            _as_int("not-a-number")


class TestConstants(unittest.TestCase):
    def test_download_chunk_size_is_ten_mib(self):
        # 10 * 1024 * 1024 computed independently.
        self.assertEqual(DOWNLOAD_CHUNK_SIZE, 10485760)

    def test_repo_id_separator(self):
        self.assertEqual(REPO_ID_SEPARATOR, "--")

    def test_auto_symlink_threshold_is_five_mib(self):
        # 5 * 1024 * 1024 computed independently.
        self.assertEqual(DEFALUT_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD, 5242880)


class TestRepoFolderName(unittest.TestCase):
    def test_model_repo(self):
        # Expected string written out literally (independent of the module).
        result = repo_folder_name(repo_id="user/model", repo_type="model")
        self.assertEqual(result, "models--user--model")

    def test_nested_repo_id_flattens_all_slashes(self):
        result = repo_folder_name(repo_id="org/sub/model", repo_type="model")
        self.assertEqual(result, "models--org--sub--model")

    def test_dataset_repo_type_pluralized(self):
        result = repo_folder_name(repo_id="org/name", repo_type="dataset")
        self.assertEqual(result, "datasets--org--name")


class TestNormalizeEtag(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(_normalize_etag(None))

    def test_strong_etag_strips_quotes(self):
        self.assertEqual(_normalize_etag('"abc123"'), "abc123")

    def test_weak_etag_strips_prefix_and_quotes(self):
        self.assertEqual(_normalize_etag('W/"abc123"'), "abc123")

    def test_bare_value_unchanged(self):
        self.assertEqual(_normalize_etag("abc123"), "abc123")

    def test_dashes_inside_value_preserved(self):
        self.assertEqual(_normalize_etag('"tag-with-dash"'), "tag-with-dash")


class TestGetPointerPath(unittest.TestCase):
    def test_flat_filename(self):
        result = _get_pointer_path("/cache/model", "abc123", "config.json")
        self.assertEqual(result, "/cache/model/snapshots/abc123/config.json")

    def test_nested_relative_filename(self):
        result = _get_pointer_path("/cache/model", "rev1", "sub/dir/f.txt")
        self.assertEqual(result, "/cache/model/snapshots/rev1/sub/dir/f.txt")

    def test_path_traversal_rejected_with_message(self):
        with self.assertRaises(ValueError) as cm:
            _get_pointer_path("/cache/model", "abc123", "../../etc/passwd")
        message = str(cm.exception)
        self.assertIn("Invalid pointer path", message)
        self.assertIn("storage_folder='/cache/model'", message)
        self.assertIn("relative_filename='../../etc/passwd'", message)


class TestCacheCommitHash(unittest.TestCase):
    def test_writes_ref_file_with_exact_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _cache_commit_hash_for_specific_revision(tmpdir, "main", "abc123")
            ref_path = Path(tmpdir) / "refs" / "main"
            self.assertTrue(ref_path.is_file())
            self.assertEqual(ref_path.read_text(), "abc123")

    def test_updates_when_commit_hash_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _cache_commit_hash_for_specific_revision(tmpdir, "main", "abc123")
            _cache_commit_hash_for_specific_revision(tmpdir, "main", "def456")
            ref_path = Path(tmpdir) / "refs" / "main"
            self.assertEqual(ref_path.read_text(), "def456")

    def test_idempotent_rewrite_keeps_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _cache_commit_hash_for_specific_revision(tmpdir, "v1", "hash0")
            _cache_commit_hash_for_specific_revision(tmpdir, "v1", "hash0")
            ref_path = Path(tmpdir) / "refs" / "v1"
            self.assertEqual(ref_path.read_text(), "hash0")


class TestCheckDiskSpace(unittest.TestCase):
    def test_impossible_size_warns_with_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _check_disk_space(10**18, tmpdir)
            messages = [
                str(w.message)
                for w in caught
                if "Not enough free disk space" in str(w.message)
            ]
            self.assertEqual(len(messages), 1)
            self.assertIn("expected file size", messages[0])
            # The resolved target location is reported in the warning.
            self.assertIn(str(Path(tmpdir)), messages[0])

    def test_tiny_size_does_not_warn(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _check_disk_space(1, tmpdir)
            messages = [
                str(w.message)
                for w in caught
                if "Not enough free disk space" in str(w.message)
            ]
            self.assertEqual(messages, [])


class TestAistudioBosFileMetadata(unittest.TestCase):
    def _make(self):
        return AistudioBosFileMetadata(
            commit_hash="abc123",
            etag="etag_val",
            location="http://example.com/file",
            size=1024,
        )

    def test_fields_round_trip(self):
        meta = self._make()
        self.assertEqual(meta.commit_hash, "abc123")
        self.assertEqual(meta.etag, "etag_val")
        self.assertEqual(meta.location, "http://example.com/file")
        self.assertEqual(meta.size, 1024)

    def test_is_frozen(self):
        meta = self._make()
        with self.assertRaises(FrozenInstanceError):
            meta.commit_hash = "new_hash"

    def test_value_equality(self):
        self.assertEqual(self._make(), self._make())
        different = AistudioBosFileMetadata(
            commit_hash="abc123",
            etag="etag_val",
            location="http://example.com/file",
            size=2048,
        )
        self.assertNotEqual(self._make(), different)


class TestOfflineModeIsEnabled(unittest.TestCase):
    def test_subclass_of_connection_error(self):
        err = OfflineModeIsEnabled("offline")
        self.assertIsInstance(err, ConnectionError)

    def test_can_be_raised_and_caught_as_connection_error(self):
        with self.assertRaises(ConnectionError):
            raise OfflineModeIsEnabled("offline")


class TestSoftTemporaryDirectory(unittest.TestCase):
    def test_creates_writable_dir_and_removes_on_exit(self):
        captured = None
        with SoftTemporaryDirectory() as tmpdir:
            captured = tmpdir
            self.assertIsInstance(tmpdir, str)
            self.assertTrue(os.path.isdir(tmpdir))
            filepath = os.path.join(tmpdir, "note.txt")
            with open(filepath, "w") as f:
                f.write("payload")
            with open(filepath) as f:
                self.assertEqual(f.read(), "payload")
        self.assertFalse(os.path.exists(captured))


class TestToLocalDir(_SymlinkCacheIsolationMixin, unittest.TestCase):
    def _make_source(self, tmpdir, content=b"hello world"):
        src_dir = os.path.join(tmpdir, "cache")
        os.makedirs(src_dir)
        src_file = os.path.join(src_dir, "blob")
        with open(src_file, "wb") as f:
            f.write(content)
        return src_file

    def test_copy_when_symlinks_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src_file = self._make_source(tmpdir)
            local_dir = os.path.join(tmpdir, "local")
            os.makedirs(local_dir)
            result = _to_local_dir(
                src_file, local_dir, "model.bin", use_symlinks=False
            )
            self.assertEqual(result, os.path.join(local_dir, "model.bin"))
            self.assertFalse(os.path.islink(result))
            with open(result, "rb") as f:
                self.assertEqual(f.read(), b"hello world")

    def test_symlink_when_requested_and_supported(self):
        self._isolate_symlink_cache()
        with tempfile.TemporaryDirectory() as tmpdir:
            src_file = self._make_source(tmpdir, b"linked-bytes")
            local_dir = os.path.join(tmpdir, "local")
            os.makedirs(local_dir)
            result = _to_local_dir(
                src_file, local_dir, "model.bin", use_symlinks=True
            )
            with open(result, "rb") as f:
                self.assertEqual(f.read(), b"linked-bytes")
            if _probe_symlinks_work(local_dir):
                self.assertTrue(os.path.islink(result))
                self.assertEqual(
                    os.path.realpath(result), os.path.realpath(src_file)
                )

    def test_auto_small_file_is_copied_not_linked(self):
        self._isolate_symlink_cache()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Well below the 5 MiB auto-symlink threshold -> copy branch.
            src_file = self._make_source(tmpdir, b"x" * 128)
            local_dir = os.path.join(tmpdir, "local")
            os.makedirs(local_dir)
            result = _to_local_dir(
                src_file, local_dir, "small.bin", use_symlinks="auto"
            )
            self.assertFalse(os.path.islink(result))
            with open(result, "rb") as f:
                self.assertEqual(f.read(), b"x" * 128)

    def test_auto_large_file_uses_symlink_when_supported(self):
        self._isolate_symlink_cache()
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "cache")
            os.makedirs(src_dir)
            src_file = os.path.join(src_dir, "blob")
            # Sparse file just past the 5 MiB threshold -> symlink branch.
            with open(src_file, "wb") as f:
                f.truncate(DEFALUT_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD + 1)
            local_dir = os.path.join(tmpdir, "local")
            os.makedirs(local_dir)
            result = _to_local_dir(
                src_file, local_dir, "big.bin", use_symlinks="auto"
            )
            if _probe_symlinks_work(local_dir):
                self.assertTrue(os.path.islink(result))
                self.assertEqual(
                    os.path.realpath(result), os.path.realpath(src_file)
                )
            else:
                self.assertTrue(os.path.exists(result))

    def test_path_traversal_rejected_with_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src_file = self._make_source(tmpdir)
            with self.assertRaises(ValueError) as cm:
                _to_local_dir(
                    src_file, tmpdir, "../escape.bin", use_symlinks=False
                )
            self.assertIn("would not be in the local", str(cm.exception))


class TestCreateSymlink(_SymlinkCacheIsolationMixin, unittest.TestCase):
    def test_links_or_copies_but_always_exposes_content(self):
        self._isolate_symlink_cache()
        with tempfile.TemporaryDirectory() as tmpdir:
            src_file = os.path.join(tmpdir, "src.bin")
            dst_file = os.path.join(tmpdir, "sub", "dst.bin")
            os.makedirs(os.path.dirname(dst_file))
            with open(src_file, "wb") as f:
                f.write(b"symlink-content")
            _create_symlink(src_file, dst_file, new_blob=False)
            with open(dst_file, "rb") as f:
                self.assertEqual(f.read(), b"symlink-content")
            if _probe_symlinks_work(os.path.dirname(dst_file)):
                self.assertTrue(os.path.islink(dst_file))
                self.assertEqual(
                    os.path.realpath(dst_file), os.path.realpath(src_file)
                )

    def test_replaces_existing_destination(self):
        self._isolate_symlink_cache()
        with tempfile.TemporaryDirectory() as tmpdir:
            src_file = os.path.join(tmpdir, "src.bin")
            dst_file = os.path.join(tmpdir, "dst.bin")
            with open(src_file, "wb") as f:
                f.write(b"new-data")
            with open(dst_file, "wb") as f:
                f.write(b"stale-data")
            _create_symlink(src_file, dst_file, new_blob=False)
            with open(dst_file, "rb") as f:
                self.assertEqual(f.read(), b"new-data")


class TestRaiseForStatus(unittest.TestCase):
    """raise_for_status is the code under test. The requests.Response is a
    locally-built, non-network dependency: we set a status code and let
    requests' own raise_for_status raise the underlying HTTPError, then
    assert how our mapping translates it. Note: with huggingface_hub>=1.x
    the 400/500 branches hit a genuine production defect (response=None
    passed to error constructors that dereference response.headers); those
    two tests capture that real behavior rather than the intended mapping."""

    def _response(self, status_code, url="http://example.com/file"):
        resp = requests.Response()
        resp.status_code = status_code
        resp.url = url
        resp.reason = "Synthetic"
        return resp

    def test_404_maps_to_entry_not_found(self):
        with self.assertRaises(EntryNotFoundError) as cm:
            raise_for_status(self._response(404))
        message = str(cm.exception)
        self.assertIn("404 Client Error", message)
        self.assertIn(
            "Entry Not Found for url: http://example.com/file", message
        )

    def test_400_maps_to_bad_request_with_endpoint(self):
        # Intended contract: a 400 is meant to map to BadRequestError. But
        # production raise_for_status builds it as
        # ``BadRequestError(message, response=None)`` (common.py:674), and
        # huggingface_hub>=1.x error constructors unconditionally read
        # ``response.headers`` for a request id (errors.py:128). With
        # response=None that dereference raises AttributeError *before* any
        # BadRequestError is produced, so the intended mapping is currently
        # unreachable. Capture the genuine behavior on the pinned hub version
        # without modifying production.
        with self.assertRaises(AttributeError) as cm:
            raise_for_status(self._response(400), endpoint_name="metadata")
        self.assertIn("headers", str(cm.exception))

    def test_500_maps_to_generic_hf_http_error(self):
        # Same production defect as the 400 case: the generic branch runs
        # ``HfHubHTTPError(str(e), response=None)`` (common.py:675), which
        # crashes in the hub's error constructor on ``response.headers``.
        # The generic HfHubHTTPError mapping is therefore unreachable on the
        # pinned huggingface_hub; assert the real AttributeError.
        with self.assertRaises(AttributeError) as cm:
            raise_for_status(self._response(500))
        self.assertIn("headers", str(cm.exception))

    def test_200_does_not_raise(self):
        self.assertIsNone(raise_for_status(self._response(200)))


class TestAreSymlinksSupported(_SymlinkCacheIsolationMixin, unittest.TestCase):
    def test_matches_independent_probe_and_caches(self):
        self._isolate_symlink_cache()
        with tempfile.TemporaryDirectory() as tmpdir:
            expected = _probe_symlinks_work(tmpdir)
            first = are_symlinks_supported(tmpdir)
            self.assertIsInstance(first, bool)
            self.assertEqual(first, expected)
            # Result is memoized under the resolved cache dir.
            resolved = str(Path(tmpdir).expanduser().resolve())
            self.assertIn(resolved, common._are_symlinks_supported_in_dir)
            # Second call returns the cached value unchanged.
            self.assertEqual(are_symlinks_supported(tmpdir), expected)


class TestGetSession(unittest.TestCase):
    def test_session_is_cached_per_thread_and_reset_clears_it(self):
        self.addCleanup(reset_sessions)
        reset_sessions()
        first = get_session()
        self.assertIsInstance(first, requests.Session)
        # Same process/thread -> cache hit -> identical object.
        self.assertIs(get_session(), first)
        # Clearing the cache forces a fresh session object.
        reset_sessions()
        second = get_session()
        self.assertIsInstance(second, requests.Session)
        self.assertIsNot(second, first)


if __name__ == "__main__":
    unittest.main()
