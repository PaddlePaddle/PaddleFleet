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


class TestDeepGEMMImport(unittest.TestCase):
    def test_deep_gemm_import(self):
        import paddlefleet
        from paddlefleet.ops.deep_gemm import (  # noqa: F401
            cublaslt_gemm_tn,
            set_num_sms,
        )

        print(paddlefleet.ops.deep_gemm)

    def test_error_import(self):
        with self.assertRaises(ImportError):
            from paddlefleet.ops.deep_gemm import xxxx  # noqa: F401


class TestDeepEPImport(unittest.TestCase):
    def test_deep_gemm_import(self):
        import paddlefleet
        from paddlefleet.ops.deep_ep import (  # noqa: F401
            Buffer,
            Config,
            EventOverlap,
            topk_idx_t,
        )

        print(paddlefleet.ops.deep_ep)

    def test_error_import(self):
        with self.assertRaises(ImportError):
            from paddlefleet.ops.deep_ep import xxxx  # noqa: F401


class TestSM80ImportError(unittest.TestCase):
    # patch is_deep_gemm_or_deep_ep_available return False to simulate SM < 9.0 before import paddlefleet
    def setUp(self):
        import unittest.mock

        if "paddlefleet.ops" in sys.modules:
            del sys.modules["paddlefleet.ops"]

        self.patcher = unittest.mock.patch(
            "paddle.cuda.get_device_capability", return_value=(8, 0)
        )
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        # Clean up sys.meta_path
        sys.meta_path = [
            x
            for x in sys.meta_path
            if x.__class__.__name__ != "HardwareIncompatibleBlocker"
        ]

        if "paddlefleet.ops" in sys.modules:
            del sys.modules["paddlefleet.ops"]

    def test_deep_gemm_import_error(self):
        import paddlefleet.ops

        with self.assertRaises(RuntimeError) as cm:
            from paddlefleet.ops import deep_gemm  # noqa: F401
        self.assertIn(
            "Cannot access 'paddlefleet.ops.deep_gemm'", str(cm.exception)
        )
        with self.assertRaises(RuntimeError) as cm:
            from paddlefleet.ops.deep_gemm import cublaslt_gemm_tn  # noqa: F401
        self.assertIn(
            "Blocking import of 'paddlefleet.ops.deep_gemm'", str(cm.exception)
        )
        with self.assertRaises(RuntimeError) as cm:
            print(paddlefleet.ops.deep_gemm.cublaslt_gemm_tn)
        self.assertIn(
            "Cannot access 'paddlefleet.ops.deep_gemm'", str(cm.exception)
        )

    def test_deep_ep_import_error(self):
        import paddlefleet.ops

        with self.assertRaises(RuntimeError) as cm:
            from paddlefleet.ops import deep_ep  # noqa: F401
        self.assertIn(
            "Cannot access 'paddlefleet.ops.deep_ep'", str(cm.exception)
        )

        with self.assertRaises(RuntimeError) as cm:
            from paddlefleet.ops.deep_ep import Buffer  # noqa: F401
        self.assertIn(
            "Blocking import of 'paddlefleet.ops.deep_ep'", str(cm.exception)
        )

        with self.assertRaises(RuntimeError) as cm:
            print(paddlefleet.ops.deep_ep.Buffer)
        self.assertIn(
            "Cannot access 'paddlefleet.ops.deep_ep'", str(cm.exception)
        )


if __name__ == "__main__":
    unittest.main()
