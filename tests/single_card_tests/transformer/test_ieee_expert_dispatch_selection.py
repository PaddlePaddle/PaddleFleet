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

"""Exercise the actual constructor selection without creating a model or group."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


class TestIEEEExpertDispatchSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (
            Path(__file__).resolve().parents[3]
            / "src/paddlefleet/transformer/moe/moe_layer.py"
        )
        tree = ast.parse(source.read_text())
        moe = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "MoELayer"
        )
        init = next(
            n
            for n in moe.body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )
        choices = [
            n
            for n in init.body
            if isinstance(n, ast.If)
            and any(
                isinstance(x, ast.Attribute)
                and x.attr == "use_accuracy_compatible"
                for x in ast.walk(n.test)
            )
            and len(n.body) == 1
            and isinstance(n.body[0], ast.Assign)
            and isinstance(n.body[0].value, ast.Constant)
            and n.body[0].value.value == "alltoall"
        ]
        assert len(choices) == 1
        cls.code = compile(
            ast.Module(body=choices, type_ignores=[]), str(source), "exec"
        )

    def test_constructor_respects_actual_group_and_default_off_contract(self):
        cases = [
            # compatible, IEEE, fused, actual EP, requested, expected
            (True, True, True, 2, "deepep", "deepep"),
            (True, False, True, 2, "deepep", "alltoall"),
            (True, True, False, 2, "deepep", "alltoall"),
            (True, True, True, 1, "deepep", "alltoall"),
            (True, True, True, None, "deepep", "alltoall"),
            (True, True, True, 2, "alltoall", "alltoall"),
            (True, True, True, 2, "hybridep", "alltoall"),
            (False, False, True, 2, "hybridep", "hybridep"),
        ]
        for compatible, ieee, fused, ep, requested, expected in cases:
            with self.subTest(case=(compatible, ieee, fused, ep, requested)):
                instance = SimpleNamespace(
                    use_accuracy_compatible=compatible,
                    moe_token_dispatcher_type=requested,
                )
                namespace = {
                    "self": instance,
                    # A declared EP2 must not override an actual local/absent group.
                    "config": SimpleNamespace(
                        moe_expert_fusion=fused, expert_model_parallel_size=2
                    ),
                    "pg_collection": SimpleNamespace(
                        ep=None if ep is None else SimpleNamespace(nranks=ep)
                    ),
                    "utils": SimpleNamespace(
                        get_pg_size=lambda group: group.nranks
                    ),
                    "ieee_kernel_enabled": lambda: ieee,
                }
                exec(self.code, namespace)
                self.assertEqual(instance.moe_token_dispatcher_type, expected)


if __name__ == "__main__":
    unittest.main()
