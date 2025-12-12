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

import sys
import unittest


class TestOpsImport(unittest.TestCase):
    TARGET_OPS = [
        "tokens_unzip_gather",
        "tokens_unzip_slice",
        "tokens_unzip_stable",
        "tokens_zip_prob",
        "tokens_zip_prob_seq_subbatch",
        "tokens_zip_unique_add",
        "tokens_zip_unique_add_subbatch",
        "fused_swiglu_scale",
        "fused_swiglu_scale_bwd",
    ]

    def setUp(self):
        if "paddlefleet.ops" in sys.modules:
            del sys.modules["paddlefleet.ops"]

        try:
            import paddlefleet.ops

            self.ops = paddlefleet.ops
        except Exception as e:
            self.fail(f"Failed to import paddlefleet.ops: {e}")

    def test_import_ops(self):
        self.assertIsNotNone(self.ops, "Failed to import paddlefleet.ops")

    def test_ops_submodule_availability(self):
        if self.ops is None:
            self.skipTest(
                "paddlefleet.ops not available. Skipping op availability tests."
            )
        else:
            self.assertIsNotNone(
                self.ops,
                "paddlefleet.ops is None, expected it to be loaded.",
            )

    def test_tokens_ops_availability(self):
        if self.ops is None:
            self.skipTest(
                "paddlefleet.ops not available. Skipping tokens_ ops availability tests."
            )
            return

        missing_ops = []
        for op_name in self.TARGET_OPS:
            if not hasattr(self.ops, op_name):
                missing_ops.append(op_name)

        if missing_ops:
            self.fail(
                f"The following operators are missing from paddlefleet.ops "
                f"(C++ extension likely not compiled correctly or is outdated): {', '.join(missing_ops)}"
            )


if __name__ == "__main__":
    unittest.main()
