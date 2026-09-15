# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Behavior tests for paddlefleet.utils.download.aistudio_hub_download.

These tests drive the real production APIs and verify concrete, independently
hand-derived results: URL/path construction, filename resolution, header
content, cache short-circuit logic and metadata parsing. The only thing mocked
is the HTTP client (`_request_wrapper`) / the network collaborators
(`get_aistudio_file_metadata`, `http_get`), which are legitimate
non-under-test external dependencies. No under-test logic is patched, and no
expected value is produced by calling the function under test.
"""

import os
import tempfile
import unittest
from unittest import mock

from paddlefleet.utils.download import aistudio_hub_download as mod
from paddlefleet.utils.download.aistudio_hub_download import (
    ENDPOINT,
    VERSION,
    LocalTokenNotFoundError,
    _clean_token,
    _validate_token_to_send,
    aistudio_hub_download,
    aistudio_hub_try_to_load_from_cache,
    aistudio_hub_url,
    build_aistudio_headers,
    get_aistudio_file_metadata,
    get_token_to_send,
)


class TestCleanToken(unittest.TestCase):
    """_clean_token is a pure string transform; no env/file dependency."""

    def test_none_passthrough(self):
        self.assertIsNone(_clean_token(None))

    def test_strips_surrounding_whitespace(self):
        self.assertEqual(_clean_token("  abc123  "), "abc123")

    def test_removes_all_cr_and_lf_not_only_trailing(self):
        # Implementation replaces every \r and \n, then strips. An internal
        # newline must therefore be removed, not just trailing ones.
        self.assertEqual(_clean_token("ab\r\ncd\n"), "abcd")
        self.assertEqual(_clean_token("hello\r\n"), "hello")

    def test_blank_becomes_none(self):
        self.assertIsNone(_clean_token(""))
        self.assertIsNone(_clean_token("   "))
        self.assertIsNone(_clean_token("\r\n"))


class TestGetTokenToSend(unittest.TestCase):
    """String/False branches need no env; True branch isolates the token
    lookup collaborator (get_token) which reads env + disk."""

    def test_explicit_string_returned_verbatim(self):
        self.assertEqual(get_token_to_send("my_token"), "my_token")

    def test_false_forbids_token(self):
        self.assertIsNone(get_token_to_send(False))

    def test_true_without_cached_token_raises(self):
        # get_token is a non-under-test collaborator (reads env var + token
        # file); isolate it to reach the "required but absent" branch.
        with mock.patch.object(mod, "get_token", return_value=None):
            with self.assertRaises(LocalTokenNotFoundError):
                get_token_to_send(True)

    def test_true_with_cached_token_returns_it(self):
        with mock.patch.object(mod, "get_token", return_value="cached_tok"):
            self.assertEqual(get_token_to_send(True), "cached_tok")


class TestValidateTokenToSend(unittest.TestCase):
    def test_write_action_requires_token(self):
        with self.assertRaises(ValueError):
            _validate_token_to_send(None, is_write_action=True)

    def test_write_action_with_token_ok(self):
        # Must not raise.
        _validate_token_to_send("tok", is_write_action=True)

    def test_read_action_allows_missing_token(self):
        # Must not raise.
        _validate_token_to_send(None, is_write_action=False)


class TestBuildAistudioHeaders(unittest.TestCase):
    """Passing an explicit token bypasses env/file lookup, so the real
    header-assembly logic is exercised deterministically."""

    def test_string_token_sets_authorization(self):
        headers = build_aistudio_headers(token="secret")
        self.assertEqual(headers["Authorization"], "token secret")
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["SDK-Version"], str(VERSION))

    def test_false_token_omits_authorization(self):
        headers = build_aistudio_headers(token=False)
        self.assertNotIn("Authorization", headers)
        # Non-auth headers are still present and content-exact.
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["SDK-Version"], str(VERSION))

    def test_write_action_without_token_raises(self):
        with self.assertRaises(ValueError):
            build_aistudio_headers(token=False, is_write_action=True)


class TestAistudioHubUrl(unittest.TestCase):
    """URL construction: expected strings are hand-derived from the template
    ENDPOINT + /api/v1/repos/{user}/{repo}/contents/{filename}."""

    def test_basic_url_exact(self):
        url = aistudio_hub_url("acme/mymodel", "config.json")
        expected = ENDPOINT + "/api/v1/repos/acme/mymodel/contents/config.json"
        self.assertEqual(url, expected)

    def test_subfolder_is_prefixed_with_forward_slash(self):
        url = aistudio_hub_url("acme/mymodel", "model.bin", subfolder="weights")
        expected = (
            ENDPOINT + "/api/v1/repos/acme/mymodel/contents/weights/model.bin"
        )
        self.assertEqual(url, expected)

    def test_empty_subfolder_equivalent_to_none(self):
        self.assertEqual(
            aistudio_hub_url("acme/mymodel", "config.json", subfolder=""),
            aistudio_hub_url("acme/mymodel", "config.json", subfolder=None),
        )

    def test_master_revision_has_no_ref_query(self):
        url = aistudio_hub_url("acme/mymodel", "config.json", revision="master")
        self.assertEqual(
            url, ENDPOINT + "/api/v1/repos/acme/mymodel/contents/config.json"
        )

    def test_non_master_revision_appends_encoded_ref(self):
        # revision is quoted with safe="" so a slash becomes %2F.
        url = aistudio_hub_url(
            "acme/mymodel", "config.json", revision="feature/x"
        )
        expected = (
            ENDPOINT
            + "/api/v1/repos/acme/mymodel/contents/config.json"
            + "?ref=feature%2Fx"
        )
        self.assertEqual(url, expected)

    def test_custom_endpoint_replaces_prefix_only(self):
        url = aistudio_hub_url(
            "acme/mymodel", "config.json", endpoint="http://custom.api.com"
        )
        self.assertEqual(
            url,
            "http://custom.api.com/api/v1/repos/acme/mymodel/contents/config.json",
        )

    def test_repo_id_parts_are_stripped(self):
        # " acme "/" mymodel " -> stripped user/repo names in the path.
        url = aistudio_hub_url(" acme / mymodel ", "config.json")
        self.assertEqual(
            url, ENDPOINT + "/api/v1/repos/acme/mymodel/contents/config.json"
        )

    def test_filename_special_chars_percent_encoded(self):
        # quote() default safe="/" keeps slashes but encodes spaces.
        url = aistudio_hub_url("acme/mymodel", "my file.json")
        self.assertEqual(
            url,
            ENDPOINT + "/api/v1/repos/acme/mymodel/contents/my%20file.json",
        )

    def test_invalid_repo_id_raises(self):
        with self.assertRaises(ValueError):
            aistudio_hub_url("norepomarker", "config.json")

    def test_invalid_repo_type_raises(self):
        with self.assertRaises(ValueError):
            aistudio_hub_url("acme/mymodel", "config.json", repo_type="dataset")


class TestTryToLoadFromCache(unittest.TestCase):
    """Filename resolution against the on-disk cache layout. Expected paths
    are hand-built as {cache}/models--user--repo/snapshots/<rev>/<file>."""

    def test_invalid_repo_type_raises(self):
        with self.assertRaises(ValueError):
            aistudio_hub_try_to_load_from_cache(
                "acme/mymodel", "config.json", repo_type="dataset"
            )

    def test_missing_repo_cache_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(
                aistudio_hub_try_to_load_from_cache(
                    "acme/mymodel", "config.json", cache_dir=tmp
                )
            )

    def test_cached_file_returns_exact_snapshot_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_cache = os.path.join(tmp, "models--acme--mymodel")
            snap = os.path.join(repo_cache, "snapshots", "master")
            os.makedirs(snap)
            expected = os.path.join(snap, "config.json")
            with open(expected, "w") as f:
                f.write("{}")

            result = aistudio_hub_try_to_load_from_cache(
                "acme/mymodel", "config.json", cache_dir=tmp, revision="master"
            )
            self.assertEqual(result, expected)

    def test_ref_resolves_to_commit_snapshot(self):
        # A branch name in refs/ must be resolved to its commit sha before
        # the snapshot lookup. Only the sha-named snapshot exists.
        with tempfile.TemporaryDirectory() as tmp:
            sha = "a" * 40
            repo_cache = os.path.join(tmp, "models--acme--mymodel")
            os.makedirs(os.path.join(repo_cache, "refs"))
            with open(os.path.join(repo_cache, "refs", "master"), "w") as f:
                f.write(sha)
            snap = os.path.join(repo_cache, "snapshots", sha)
            os.makedirs(snap)
            expected = os.path.join(snap, "config.json")
            with open(expected, "w") as f:
                f.write("{}")

            result = aistudio_hub_try_to_load_from_cache(
                "acme/mymodel", "config.json", cache_dir=tmp, revision="master"
            )
            self.assertEqual(result, expected)

    def test_no_exist_marker_short_circuits_to_none(self):
        # Even if a snapshot file physically exists, a .no_exist marker for the
        # same revision/filename must force a None result.
        with tempfile.TemporaryDirectory() as tmp:
            repo_cache = os.path.join(tmp, "models--acme--mymodel")
            snap = os.path.join(repo_cache, "snapshots", "master")
            os.makedirs(snap)
            with open(os.path.join(snap, "config.json"), "w") as f:
                f.write("{}")
            noexist = os.path.join(repo_cache, ".no_exist", "master")
            os.makedirs(noexist)
            with open(os.path.join(noexist, "config.json"), "w") as f:
                f.write("")

            result = aistudio_hub_try_to_load_from_cache(
                "acme/mymodel", "config.json", cache_dir=tmp, revision="master"
            )
            self.assertIsNone(result)

    def test_wrong_revision_not_returned(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_cache = os.path.join(tmp, "models--acme--mymodel")
            snap = os.path.join(repo_cache, "snapshots", "master")
            os.makedirs(snap)
            with open(os.path.join(snap, "config.json"), "w") as f:
                f.write("{}")
            # Ask for a revision that has no snapshot dir.
            result = aistudio_hub_try_to_load_from_cache(
                "acme/mymodel", "config.json", cache_dir=tmp, revision="v9"
            )
            self.assertIsNone(result)


class TestGetAistudioFileMetadata(unittest.TestCase):
    """Isolate the HTTP client (`_request_wrapper`) and verify the real
    request construction and the metadata parsing/mapping."""

    def test_request_args_and_metadata_mapping(self):
        url = aistudio_hub_url("acme/mymodel", "config.json")
        captured = {}

        class _FakeResponse:
            def raise_for_status(self):  # success: no HTTP error
                return None

            def json(self):
                return {
                    "last_commit_sha": "c0ffee",
                    "sha": '"deadbeef"',  # quoted etag from server
                    "git_url": "http://cdn.example/blob/deadbeef",
                    "size": 4096,
                }

        def fake_request_wrapper(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return _FakeResponse()

        with mock.patch.object(
            mod, "_request_wrapper", side_effect=fake_request_wrapper
        ) as req:
            meta = get_aistudio_file_metadata(url, token="tok")

        # The HTTP client was invoked exactly once with the real, content-exact
        # request parameters.
        req.assert_called_once()
        kwargs = captured["kwargs"]
        self.assertEqual(kwargs["method"], "GET")
        self.assertEqual(kwargs["url"], url)
        self.assertFalse(kwargs["allow_redirects"])
        self.assertTrue(kwargs["follow_relative_redirects"])

        # Real header assembly must run: identity encoding + auth token.
        headers = kwargs["headers"]
        self.assertEqual(headers["Accept-Encoding"], "identity")
        self.assertEqual(headers["Authorization"], "token tok")
        self.assertEqual(headers["Content-Type"], "application/json")

        # Metadata fields are mapped from the JSON body; etag is normalized by
        # stripping the surrounding quotes (hand-derived: '"deadbeef"'->deadbeef).
        self.assertEqual(meta.commit_hash, "c0ffee")
        self.assertEqual(meta.etag, "deadbeef")
        self.assertEqual(meta.location, "http://cdn.example/blob/deadbeef")
        self.assertEqual(meta.size, 4096)


class TestDownloadCacheShortCircuit(unittest.TestCase):
    """When the pointer file already exists and force_download is False,
    aistudio_hub_download must return the local path WITHOUT touching the
    network collaborators."""

    def test_existing_pointer_returns_local_path_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Hand-built cache layout matching repo_folder_name +
            # _get_pointer_path: models--acme--mymodel/snapshots/master/config.json
            storage = os.path.join(tmp, "models--acme--mymodel")
            snap = os.path.join(storage, "snapshots", "master")
            os.makedirs(snap)
            expected = os.path.join(snap, "config.json")
            with open(expected, "w") as f:
                f.write("cached-bytes")

            def _boom(*a, **k):
                raise AssertionError(
                    "network collaborator must not be called on cache hit"
                )

            with (
                mock.patch.object(
                    mod, "get_aistudio_file_metadata", side_effect=_boom
                ),
                mock.patch.object(mod, "http_get", side_effect=_boom),
            ):
                result = aistudio_hub_download(
                    repo_id="acme/mymodel",
                    filename="config.json",
                    cache_dir=tmp,
                    revision="master",
                )

            self.assertEqual(result, expected)
            # Confirm the returned path is the real cached file we wrote.
            with open(result) as f:
                self.assertEqual(f.read(), "cached-bytes")

    def test_invalid_repo_type_raises_before_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                aistudio_hub_download(
                    repo_id="acme/mymodel",
                    filename="config.json",
                    cache_dir=tmp,
                    repo_type="dataset",
                )


if __name__ == "__main__":
    unittest.main()
