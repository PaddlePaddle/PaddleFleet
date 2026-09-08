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

"""Exercise the actual nested loss attachment with native device autograd."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import paddle


class TestIEEEMTPLossAttachment(unittest.TestCase):
    def attachment(self, *, ieee, compatible=True, add_mtp=True):
        source = (
            Path(__file__).resolve().parents[3]
            / "src/paddlefleet/models/common/language_loss/language_loss.py"
        )
        tree = ast.parse(source.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "LanguageLoss"
        )
        forward = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "forward"
        )
        helpers = [
            n
            for n in ast.walk(forward)
            if isinstance(n, ast.FunctionDef) and n.name == "add_loss"
        ]
        self.assertEqual(len(helpers), 1)
        namespace = {
            "self": SimpleNamespace(
                config=SimpleNamespace(add_mtp_loss=add_mtp)
            ),
            "_use_accuracy_compatible_kernel": lambda: compatible,
            "ieee_kernel_enabled": lambda: ieee,
        }
        exec(
            compile(
                ast.Module(body=helpers, type_ignores=[]), str(source), "exec"
            ),
            namespace,
        )
        return namespace["add_loss"]

    @staticmethod
    def inputs():
        # At this scale, adding MAIN to MTP first loses MAIN in FP32.
        main = paddle.full([], 1.0, dtype="float32")
        auxiliary = paddle.full([], 16777216.0, dtype="float32")
        main.stop_gradient = False
        auxiliary.stop_gradient = False
        return main, auxiliary

    def test_ieee_preserves_main_value_and_both_gradients(self):
        main, auxiliary = self.inputs()
        output = self.attachment(ieee=True)(main, auxiliary)
        self.assertEqual(output.item(), main.item())
        output.backward()
        self.assertEqual(main.grad.item(), 1.0)
        self.assertEqual(auxiliary.grad.item(), 1.0)

    def test_default_off_preserves_existing_rounding(self):
        main, auxiliary = self.inputs()
        output = self.attachment(ieee=False)(main, auxiliary)
        self.assertEqual(output.item(), 0.0)
        output.backward()
        self.assertEqual(main.grad.item(), 1.0)
        self.assertEqual(auxiliary.grad.item(), 1.0)

    def test_disabled_auxiliary_does_not_acquire_gradient(self):
        main, auxiliary = self.inputs()
        output = self.attachment(ieee=True, add_mtp=False)(main, auxiliary)
        self.assertEqual(output.item(), 1.0)
        output.backward()
        self.assertIsNone(auxiliary.grad)

    def test_noncompatible_path_still_adds_auxiliary_value(self):
        main, auxiliary = self.inputs()
        output = self.attachment(ieee=True, compatible=False)(main, auxiliary)
        self.assertEqual(output.item(), 16777216.0)
        output.backward()
        self.assertEqual(auxiliary.grad.item(), 1.0)


if __name__ == "__main__":
    paddle.set_device("gpu:0")
    unittest.main()
