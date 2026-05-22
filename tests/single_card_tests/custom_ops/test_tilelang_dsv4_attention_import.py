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

import ast
import importlib
import numpy as np
import os
import pathlib
import subprocess
import sys
import textwrap
import unittest


class TestTileLangDSV4AttentionImport(unittest.TestCase):
    def assert_no_top_level_torch_import(self, module):
        source_path = pathlib.Path(module.__file__)
        tree = ast.parse(source_path.read_text())
        top_level_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]

        imported_modules = []
        for node in top_level_imports:
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            else:
                imported_modules.append(node.module or "")

        self.assertNotIn("torch", imported_modules)
        self.assertFalse(any(module.startswith("torch.") for module in imported_modules))

    def test_attention_core_has_no_top_level_torch_import(self):
        import paddlefleet.ops.tilelang_dsv4.attention_core as attention_core

        self.assert_no_top_level_torch_import(attention_core)

    def test_csa_indexer_core_has_no_top_level_torch_import(self):
        import paddlefleet.ops.tilelang_dsv4.csa_indexer_core as csa_indexer_core

        self.assert_no_top_level_torch_import(csa_indexer_core)

    def test_attention_core_exposes_paddle_reference(self):
        module = importlib.import_module("paddlefleet.ops.tilelang_dsv4.attention_core")

        self.assertTrue(hasattr(module, "sparse_attn_paddle"))

    def test_tilelang_kernel_interfaces_have_no_top_level_torch_import(self):
        from paddlefleet.ops.tilelang_dsv4.kernel import tilelang_csa_indexer_bwd
        from paddlefleet.ops.tilelang_dsv4.kernel import tilelang_csa_indexer_fwd
        from paddlefleet.ops.tilelang_dsv4.kernel import tilelang_sparse_mla
        from paddlefleet.ops.tilelang_dsv4.kernel import tilelang_sparse_mla_bwd
        from paddlefleet.ops.tilelang_dsv4.kernel import tilelang_sparse_mla_fwd

        for module in (
            tilelang_csa_indexer_bwd,
            tilelang_csa_indexer_fwd,
            tilelang_sparse_mla,
            tilelang_sparse_mla_bwd,
            tilelang_sparse_mla_fwd,
        ):
            self.assert_no_top_level_torch_import(module)

    def test_csa_backward_rejects_non_paddle_topk_and_grad_scores(self):
        import paddle
        from paddlefleet.ops.tilelang_dsv4.csa_indexer_core import tilelang_csa_compressed_indexer_bwd_paddle

        index_q = paddle.zeros([1, 4, 8, 16], dtype="bfloat16")
        weights = paddle.zeros([1, 4, 8], dtype="float32")
        index_k_comp = paddle.zeros([1, 2, 16], dtype="bfloat16")
        topk_indices = paddle.zeros([1, 4, 2], dtype="int32")
        grad_scores = paddle.zeros([1, 4, 2], dtype="float32")

        with self.assertRaisesRegex(TypeError, "topk_indices must be a paddle.Tensor"):
            tilelang_csa_compressed_indexer_bwd_paddle(
                index_q,
                weights,
                index_k_comp,
                np.zeros([1, 4, 2], dtype="int32"),
                grad_scores,
            )

        with self.assertRaisesRegex(TypeError, "grad_scores must be a paddle.Tensor"):
            tilelang_csa_compressed_indexer_bwd_paddle(
                index_q,
                weights,
                index_k_comp,
                topk_indices,
                np.zeros([1, 4, 2], dtype="float32"),
            )

    def test_public_attention_import_does_not_require_real_torch(self):
        repo_src = pathlib.Path(__file__).resolve().parents[3] / "src"
        script = textwrap.dedent(
            """
            import importlib.abc
            import sys

            class TorchBlocker(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "torch" or fullname.startswith("torch."):
                        raise ImportError("blocked torch")
                    return None

            sys.modules.pop("torch", None)
            sys.meta_path.insert(0, TorchBlocker())

            import paddlefleet.ops.tilelang_dsv4.attention_core as attention_core

            assert hasattr(attention_core, "sparse_attn_paddle")
            exec("from paddlefleet.ops.tilelang_dsv4 import *")
            """
        )

        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_src)
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
