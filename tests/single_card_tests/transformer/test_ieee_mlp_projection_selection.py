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

"""Check projection dispatch against actual groups without constructing a model."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class TestIEEEProjectionSelection(unittest.TestCase):
    def test_each_projection_keeps_its_actual_parallel_layer(self):
        source = (
            Path(__file__).resolve().parents[3]
            / "src/paddlefleet/transformer/mlp.py"
        )
        tree = ast.parse(source.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "MLP"
        )
        forward = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "forward"
        )
        choices = [
            n
            for n in forward.body
            if isinstance(n, ast.If)
            and any(
                isinstance(x, ast.Call)
                and isinstance(x.func, ast.Name)
                and x.func.id == "_accuracy_compatible_projection"
                for x in ast.walk(n)
            )
        ]
        self.assertEqual(len(choices), 2)
        for (
            ieee,
            configured,
            up_size,
            down_size,
            up_expected,
            down_expected,
        ) in [
            (True, 1, 2, 2, "native", "native"),
            (True, 1, 1, 1, "direct", "direct"),
            (True, 1, 1, 2, "direct", "native"),
            (True, 1, 2, 1, "native", "direct"),
            (True, 1, None, None, "native", "native"),
            (False, 1, 1, 1, "native", "native"),
            (True, 2, 2, 2, "native", "native"),
        ]:
            with self.subTest(case=(ieee, configured, up_size, down_size)):

                def projection(size):
                    return (
                        SimpleNamespace()
                        if size is None
                        else SimpleNamespace(world_size=size)
                    )

                up, down = projection(up_size), projection(down_size)
                instance = SimpleNamespace(
                    config=SimpleNamespace(
                        tensor_model_parallel_size=configured
                    ),
                    up_gate_proj=up,
                    down_proj=down,
                    _dw_up_gate_point="up",
                    _dw_down_point="down",
                )
                for choice, layer, expected in zip(
                    choices, [up, down], [up_expected, down_expected]
                ):
                    direct = Mock(return_value=("result", None))
                    native = Mock(return_value=("result", None))
                    namespace = {
                        "self": instance,
                        "_ACCURACY_COMPATIBLE_KERNEL": ieee,
                        "_accuracy_compatible_projection": direct,
                        "deferrable_linear": native,
                        "hidden_states": "input",
                        "intermediate_parallel": "activation",
                    }
                    exec(
                        compile(
                            ast.Module(body=[choice], type_ignores=[]),
                            str(source),
                            "exec",
                        ),
                        namespace,
                    )
                    if expected == "direct":
                        direct.assert_called_once()
                        self.assertIs(direct.call_args.args[0], layer)
                        native.assert_not_called()
                    else:
                        native.assert_called_once()
                        self.assertIs(native.call_args.args[2], layer)
                        direct.assert_not_called()


if __name__ == "__main__":
    unittest.main()
