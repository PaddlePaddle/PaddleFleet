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

"""Exercise the production registration block without initializing GPU groups."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


class TestSharedExpertSPRegistration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (
            Path(__file__).resolve().parents[3]
            / "src/paddlefleet/transformer/moe/moe_layer.py"
        )
        tree = ast.parse(source.read_text())
        layer = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MoELayer"
        )
        constructor = next(
            node
            for node in layer.body
            if getattr(node, "name", None) == "__init__"
        )
        blocks = [
            node
            for node in constructor.body
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "mark_as_sequence_parallel_parameter"
                for child in ast.walk(node)
            )
        ]
        assert len(blocks) == 1, "Identify the actual constructor marking block"
        cls.registration = compile(
            ast.Module(body=blocks, type_ignores=[]), str(source), "exec"
        )

    def registered(
        self, *, compatible=True, distributed=True, bias=True, **state
    ):
        def parameter(name, sharded):
            return SimpleNamespace(name=name, is_distributed=sharded)

        shared = SimpleNamespace(
            up_gate_proj=SimpleNamespace(
                weight=parameter("up_weight", distributed),
                bias=parameter("up_bias", distributed) if bias else None,
            ),
            down_proj=SimpleNamespace(
                weight=parameter("down_weight", distributed),
                bias=parameter("down_bias", False) if bias else None,
            ),
        )
        layer = SimpleNamespace(
            config=SimpleNamespace(gpt_model_use_experimental_version=False),
            sequence_parallel=True,
            expert_model_parallel_size=2,
            shared_experts=shared,
            use_accuracy_compatible=compatible,
        )
        for name, value in state.items():
            setattr(layer, name, value)
        marked = []
        exec(
            self.registration,
            {
                "self": layer,
                "shared_expert_config": SimpleNamespace(use_bias=bias),
                "mark_as_sequence_parallel_parameter": lambda p: marked.append(
                    p.name
                ),
            },
        )
        return marked

    def test_sharded_parameters_excluded_but_replicated_row_bias_retained(self):
        self.assertEqual(self.registered(), ["down_bias"])
        self.assertEqual(self.registered(bias=False), [])

    def test_default_mode_preserves_historical_registration(self):
        self.assertEqual(
            self.registered(compatible=False),
            ["up_weight", "up_bias", "down_weight", "down_bias"],
        )

    def test_replicated_parameters_keep_sp_sum(self):
        self.assertEqual(
            self.registered(distributed=False),
            ["up_weight", "up_bias", "down_weight", "down_bias"],
        )

    def test_outer_conditions_remain_unchanged(self):
        for state in (
            {"sequence_parallel": False},
            {"expert_model_parallel_size": 1},
            {"shared_experts": None},
            {
                "config": SimpleNamespace(
                    gpt_model_use_experimental_version=True
                )
            },
        ):
            with self.subTest(state=state):
                self.assertEqual(self.registered(**state), [])


if __name__ == "__main__":
    unittest.main()
