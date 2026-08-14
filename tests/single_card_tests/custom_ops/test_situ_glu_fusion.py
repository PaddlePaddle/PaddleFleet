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

import os
import subprocess
import sys
import unittest

import paddle

from paddlefleet.transformer.activations import (
    situ_glu_scale_backward,
    situ_glu_scale_forward,
)
from paddlefleet.triton_ops.situ_glu import (
    situ_glu_scale_backward_triton,
    situ_glu_scale_forward_triton,
)


class TestSituGLUFusion(unittest.TestCase):
    def test_triton_entries_reject_cpu_inputs(self):
        cpu_place = paddle.CPUPlace()
        x = paddle.to_tensor([[1.0] * 8] * 2, dtype="float32", place=cpu_place)
        probs = paddle.to_tensor([1.0, 1.0], dtype="float32", place=cpu_place)
        out_grad = paddle.to_tensor(
            [[1.0] * 4] * 2, dtype="float32", place=cpu_place
        )

        with self.assertRaisesRegex(ValueError, "must be GPU tensors"):
            situ_glu_scale_forward_triton(x, probs)
        with self.assertRaisesRegex(ValueError, "must be GPU tensors"):
            situ_glu_scale_backward_triton(x, probs, out_grad)

    def test_triton_entries_reject_invalid_scales(self):
        invalid_scales = (
            -1.0,
            0.0,
            float("-inf"),
            float("inf"),
            float("nan"),
        )
        entries = (
            (situ_glu_scale_forward_triton, (None, None)),
            (situ_glu_scale_backward_triton, (None, None, None)),
        )

        for function, args in entries:
            for beta in invalid_scales:
                with (
                    self.subTest(function=function.__name__, beta=beta),
                    self.assertRaisesRegex(ValueError, "positive finite"),
                ):
                    function(*args, beta=beta)
            for linear_beta in invalid_scales:
                with (
                    self.subTest(
                        function=function.__name__, linear_beta=linear_beta
                    ),
                    self.assertRaisesRegex(ValueError, "positive finite"),
                ):
                    function(*args, linear_beta=linear_beta)

    def test_triton_guards_survive_optimized_mode(self):
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../..")
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [
                os.path.join(repo_root, "src"),
                env.get("PYTHONPATH", ""),
            ]
        )
        script = """
from paddlefleet.triton_ops.situ_glu import (
    situ_glu_scale_backward_triton,
    situ_glu_scale_forward_triton,
)

entries = (
    (situ_glu_scale_forward_triton, (None, None)),
    (situ_glu_scale_backward_triton, (None, None, None)),
)
for function, args in entries:
    for kwargs in ({"beta": 0.0}, {"linear_beta": float("nan")}):
        try:
            function(*args, **kwargs)
        except ValueError as error:
            if "positive finite" not in str(error):
                raise
        else:
            raise SystemExit(f"{function.__name__} accepted {kwargs}")
"""
        proc = subprocess.run(
            [sys.executable, "-O", "-c", script],
            capture_output=True,
            text=True,
            env=env,
            cwd=repo_root,
        )

        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_default_fused_matches_disabled_reference(self):
        paddle.seed(20260812)
        x = paddle.randn([17, 6144], dtype="bfloat16")
        probs = paddle.rand([17], dtype="float32")
        out_grad = paddle.randn([17, 3072], dtype="bfloat16")

        expected_forward = situ_glu_scale_forward(
            x, probs, 4.0, 25.0, situ_glu_fusion=False
        )
        expected_backward = situ_glu_scale_backward(
            x,
            probs,
            out_grad,
            4.0,
            25.0,
            situ_glu_fusion=False,
        )

        actual_forward = situ_glu_scale_forward(x, probs, 4.0, 25.0)
        actual_backward = situ_glu_scale_backward(x, probs, out_grad, 4.0, 25.0)
        self.assertTrue(
            paddle.equal_all(
                actual_forward.astype("float32"),
                expected_forward.astype("float32"),
            )
        )
        self.assertTrue(
            paddle.allclose(
                actual_backward[0].astype("float32"),
                expected_backward[0].astype("float32"),
                rtol=1e-5,
                atol=2e-8,
            )
        )
        self.assertTrue(
            paddle.equal_all(
                actual_backward[1].astype("float32"),
                expected_backward[1].astype("float32"),
            )
        )
        self.assertTrue(
            paddle.allclose(
                actual_backward[2],
                expected_backward[2],
                rtol=1e-5,
                atol=2e-5,
            )
        )

    def test_triton_backward_optimized_and_fallback_widths(self):
        paddle.seed(20260813)
        for width, linear_beta in ((3072, None), (2048, 25.0)):
            with self.subTest(width=width, linear_beta=linear_beta):
                x = paddle.randn([17, 2 * width], dtype="bfloat16")
                probs = paddle.rand([17], dtype="float32")
                out_grad = paddle.randn([17, width], dtype="bfloat16")
                expected = situ_glu_scale_backward(
                    x,
                    probs,
                    out_grad,
                    4.0,
                    linear_beta,
                    situ_glu_fusion=False,
                )
                actual = situ_glu_scale_backward(
                    x,
                    probs,
                    out_grad,
                    4.0,
                    linear_beta,
                )

                self.assertTrue(
                    paddle.allclose(
                        actual[0].astype("float32"),
                        expected[0].astype("float32"),
                        rtol=1e-5,
                        # The fused and unfused paths can round one BF16 ULP
                        # apart after storing the input gradient.
                        atol=1e-4,
                    )
                )
                self.assertTrue(
                    paddle.equal_all(
                        actual[1].astype("float32"),
                        expected[1].astype("float32"),
                    )
                )
                self.assertTrue(
                    paddle.allclose(
                        actual[2], expected[2], rtol=1e-5, atol=2e-5
                    )
                )

    def test_default_fused_forward_supports_more_than_65535_rows(self):
        rows = 65536
        x = paddle.ones([rows, 8], dtype="bfloat16")
        probs = paddle.ones([rows], dtype="float32")
        expected = situ_glu_scale_forward(x, probs, situ_glu_fusion=False)
        actual = situ_glu_scale_forward(x, probs)
        self.assertTrue(
            paddle.equal_all(
                actual.astype("float32"), expected.astype("float32")
            )
        )


if __name__ == "__main__":
    unittest.main()
