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

"""Check the production MLA table transformation and its default-off scope."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import paddle


class TestIEEEMTPRotaryTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (
            Path(__file__).resolve().parents[3]
            / "src/paddlefleet/transformer/multi_latent_attention.py"
        )
        tree = ast.parse(source.read_text())
        mla = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "MLASelfAttention"
        )
        method = next(
            n
            for n in mla.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "get_query_key_value_tensors"
        )
        blocks = [
            n
            for n in method.body
            if isinstance(n, ast.If)
            and any(
                isinstance(c, ast.Attribute)
                and c.attr == "use_accuracy_compatible"
                for c in ast.walk(n.test)
            )
            and any(
                isinstance(c, ast.Attribute) and c.attr == "is_mtp_layer"
                for c in ast.walk(n.test)
            )
        ]
        assert len(blocks) == 1
        cls.code = compile(
            ast.Module(body=blocks, type_ignores=[]), str(source), "exec"
        )

    def transform(
        self,
        *,
        depth=0,
        ieee=True,
        uac=True,
        mtp=True,
        training=True,
        packed=False,
        cp=1,
        rope="rope",
        fused=False,
    ):
        table = paddle.arange(8, dtype="float32").reshape([1, 4, 1, 2])
        namespace = {
            "paddle": paddle,
            "ieee_kernel_enabled": lambda: ieee,
            "get_context_parallel_world_size": lambda: cp,
            "self": SimpleNamespace(
                config=SimpleNamespace(
                    use_accuracy_compatible=uac,
                    rope_type=rope,
                    apply_rope_fusion=fused,
                ),
                is_mtp_layer=mtp,
                training=training,
                layer_number=depth,
            ),
            "packed_seq": packed,
            "rotary_pos_emb": table,
        }
        exec(self.code, namespace)
        return table, namespace["rotary_pos_emb"]

    def test_depth_one_and_two_predict_next_positions(self):
        for depth, expected in [
            (0, [2, 3, 4, 5, 6, 7, 0, 1]),
            (1, [4, 5, 6, 7, 0, 1, 2, 3]),
        ]:
            with self.subTest(depth=depth):
                _, actual = self.transform(depth=depth)
                self.assertEqual(actual.flatten().tolist(), expected)

    def test_disabled_cp_group_sentinel_predicts_next_positions(self):
        _, actual = self.transform(cp=-1)
        self.assertEqual(actual.flatten().tolist(), [2, 3, 4, 5, 6, 7, 0, 1])

    def test_other_paths_preserve_original_table(self):
        for change in [
            {"uac": False},
            {"mtp": False},
            {"training": False},
            {"packed": True},
            {"cp": 2},
            {"rope": "yarn"},
            {"fused": True},
        ]:
            with self.subTest(change=change):
                original, actual = self.transform(**change)
                self.assertIs(actual, original)


if __name__ == "__main__":
    paddle.set_device("gpu:0")
    unittest.main()
