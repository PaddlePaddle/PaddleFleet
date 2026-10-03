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

import contextlib
import io
import os
import runpy
import unittest
from pathlib import Path
from unittest import mock


class PRTemplateTests(unittest.TestCase):
    branches = ("develop", "release/1.0", "glm52-stack/07-uac-moe")
    valid_body = (
        "### PR Category\nDistributed Strategy\n"
        "### PR Types\nBug fixes\n"
        "### Description\nPreserve checkpoint parameter aliases.\n"
    )

    def check_body(self, body, expected_exit):
        script = Path("ci/checkPRTemplate.py")
        event = {
            "number": 42,
            "comments_url": "https://example.invalid/comments",
            "body": body,
            "head": {"sha": "0" * 40},
            "title": "Fix checkpoint parameter aliases",
            "user": {"login": "test-author"},
        }
        for branch in self.branches:
            with (
                self.subTest(branch=branch),
                mock.patch.dict(os.environ, {"BRANCH": branch}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                namespace = runpy.run_path(str(script))
                with self.assertRaises(SystemExit) as result:
                    namespace["pull_request_event_template"](
                        event, "PaddleFleet"
                    )
                self.assertEqual(result.exception.code, expected_exit)

    def test_valid_description_passes_for_trunk_release_and_stack(self):
        self.check_body(self.valid_body, 0)

    def test_unknown_category_is_rejected(self):
        self.check_body(
            self.valid_body.replace("Distributed Strategy", "Unknown"), 7
        )

    def test_unknown_type_is_rejected(self):
        self.check_body(self.valid_body.replace("Bug fixes", "Unknown"), 7)

    def test_missing_description_is_rejected(self):
        self.check_body(None, 7)

    def test_commented_template_does_not_satisfy_policy(self):
        self.check_body("<!-- " + self.valid_body + " -->", 7)


if __name__ == "__main__":
    unittest.main()
