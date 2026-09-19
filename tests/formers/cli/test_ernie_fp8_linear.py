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

"""CPU-observable behavior tests for the ERNIE FP8 fused-MLP module.

Target production code:
``src/paddlefleet/cli/train/ernie_pretrain/models/fp8_linear.py``.

Scope (unit-test-rules.md "计算优化" + "配置与运行基础设施"):
  * ``padding`` -- the block-alignment helper that grows an axis up to the next
    512-multiple and zero-fills the tail. Verified against a hand-derived
    "next multiple of 512" oracle (independent of the module's own two-branch
    128/512 arithmetic), with original data preserved and the pad region proven
    to be exactly zero.
  * ``Fp8FusedMlp.__init__`` -- weight construction from ``config``. The fused
    ``w1`` must have ``intermediate_size * 2`` columns (SwiGLU gate+up), ``w2``
    must be ``[intermediate_size, hidden_size]``, both bfloat16.

The FP8 GEMM forward/backward (``Fp8FusedMlpFunc`` / ``MemEfficientFp8FusedMlpFunc``)
depend on ``deep_gemm`` and ``paddle.incubate.nn.functional.fp8_quant_blockwise``,
which require an FP8-capable GPU. Those numerics are NOT asserted here; they are
left unrun rather than faked. Every oracle below is hand-derived and does not
call the production implementation to build its own expected value.
"""

import importlib.util
import os
import sys
import types
import unittest

# --- Load the real production module directly from its source file, without
# importing the parent ``paddlefleet`` package __init__ chain (which pulls in
# unrelated workflow / tokenizer machinery). Only a lightweight parent-package
# stub is registered; the fp8_linear source that executes is genuine production
# code. This mirrors tests/formers/cli/test_deepseek_v3_fp8.py.
_PKG = "paddlefleet.cli.train.ernie_pretrain.models"
_MODULE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "src",
        "paddlefleet",
        "cli",
        "train",
        "ernie_pretrain",
        "models",
    )
)


def _load_real_module(sub_name):
    """Execute the real ``<sub_name>.py`` from the production directory."""
    if _PKG not in sys.modules:
        stub = types.ModuleType(_PKG)
        stub.__path__ = [_MODULE_DIR]
        stub.__package__ = _PKG
        sys.modules[_PKG] = stub
    full_name = f"{_PKG}.{sub_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(
        full_name, os.path.join(_MODULE_DIR, f"{sub_name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    import numpy as np
    import paddle

    _fp8 = _load_real_module("fp8_linear")
    padding = _fp8.padding
    Fp8FusedMlp = _fp8.Fp8FusedMlp
    _PADDLE_OK = True
    _PADDLE_REASON = ""
except ImportError as exc:  # paddle / paddlefleet not installed in this env
    paddle = None
    np = None
    _fp8 = None
    padding = None
    Fp8FusedMlp = None
    _PADDLE_OK = False
    _PADDLE_REASON = f"paddle/paddlefleet import unavailable: {exc}"


# Independent oracle: the documented contract of ``padding`` is to grow an axis
# up to the next multiple of 512 (a no-op when already a multiple). This is
# derived by analysis and is deliberately NOT the module's own 128-vs-512
# branch formula, so a broken branch selection is still caught.
def _next_multiple_of_512(n):
    if n % 512 == 0:
        return n
    return ((n // 512) + 1) * 512


@unittest.skipUnless(_PADDLE_OK, _PADDLE_REASON or "paddle unavailable")
class TestPadding(unittest.TestCase):
    """Behavior of the block-alignment ``padding`` helper (CPU-runnable)."""

    def _distinct(self, *shape):
        # Values start at 1.0 so genuine data never collides with the zero pad.
        n = 1
        for s in shape:
            n *= s
        return (paddle.arange(n, dtype="float32") + 1.0).reshape(list(shape))

    def test_axis0_grows_to_next_512_and_preserves_and_zero_fills(self):
        n, cols = 100, 64  # 100 -> 512 via the 512-branch (pad 412)
        x = self._distinct(n, cols)
        out = padding(x, 0)

        self.assertEqual(out.shape, [_next_multiple_of_512(n), cols])
        self.assertEqual(out.shape, [512, cols])
        # Original rows are preserved exactly, in order.
        np.testing.assert_array_equal(out[:n].numpy(), x.numpy())
        # The appended tail is exactly zero (not just "some" padding).
        np.testing.assert_array_equal(
            out[n:].numpy(), np.zeros([512 - n, cols], dtype=np.float32)
        )
        self.assertEqual(out.dtype, x.dtype)

    def test_axis0_128_branch_still_reaches_512_multiple(self):
        # n=896: (896 + 128 - 896%128) % 512 == 0 selects the 128 pad_size
        # branch; the result must still land on 1024 = next multiple of 512.
        n, cols = 896, 4
        x = self._distinct(n, cols)
        out = padding(x, 0)

        self.assertEqual(out.shape, [_next_multiple_of_512(n), cols])
        self.assertEqual(out.shape, [1024, cols])
        np.testing.assert_array_equal(out[:n].numpy(), x.numpy())
        np.testing.assert_array_equal(
            out[n:].numpy(), np.zeros([1024 - n, cols], dtype=np.float32)
        )

    def test_axis1_grows_to_next_512_and_preserves_and_zero_fills(self):
        rows, n = 8, 300  # 300 -> 512 on the column axis
        x = self._distinct(rows, n)
        out = padding(x, 1)

        self.assertEqual(out.shape, [rows, _next_multiple_of_512(n)])
        self.assertEqual(out.shape, [rows, 512])
        np.testing.assert_array_equal(out[:, :n].numpy(), x.numpy())
        np.testing.assert_array_equal(
            out[:, n:].numpy(), np.zeros([rows, 512 - n], dtype=np.float32)
        )

    def test_axis0_no_change_when_already_multiple_of_512(self):
        x = self._distinct(512, 3)
        out = padding(x, 0)
        # No concat on the aligned path: same object returned unchanged.
        self.assertIs(out, x)
        self.assertEqual(out.shape, [512, 3])
        np.testing.assert_array_equal(out.numpy(), x.numpy())

    def test_axis1_no_change_when_already_multiple_of_512(self):
        x = self._distinct(3, 1024)  # 1024 is a multiple of 512
        out = padding(x, 1)
        self.assertIs(out, x)
        self.assertEqual(out.shape, [3, 1024])


@unittest.skipUnless(_PADDLE_OK, _PADDLE_REASON or "paddle unavailable")
class TestFp8FusedMlpInit(unittest.TestCase):
    """``Fp8FusedMlp`` weight construction from config (CPU-runnable)."""

    def test_init_derives_weight_shapes_and_dtypes_from_config(self):
        # hidden_size != intermediate_size so a swapped-dimension bug is caught.
        config = types.SimpleNamespace(hidden_size=256, intermediate_size=512)
        layer = Fp8FusedMlp(config)

        self.assertEqual(layer.hidden_size, 256)
        self.assertEqual(layer.intermediate_size, 512)
        self.assertIs(layer.config, config)

        # w1 fuses SwiGLU gate+up -> 2 * intermediate_size columns.
        self.assertEqual(layer.w1.shape, [256, 512 * 2])
        # w2 projects the intermediate features back to hidden_size.
        self.assertEqual(layer.w2.shape, [512, 256])

        self.assertEqual(layer.w1.dtype, paddle.bfloat16)
        self.assertEqual(layer.w2.dtype, paddle.bfloat16)

    def test_init_tracks_config_dimensions_independently(self):
        # Different, mutually distinguishable dims to pin the * 2 factor and the
        # [intermediate, hidden] ordering of w2 rather than a coincidence.
        config = types.SimpleNamespace(hidden_size=64, intermediate_size=192)
        layer = Fp8FusedMlp(config)

        self.assertEqual(layer.w1.shape, [64, 384])
        self.assertEqual(layer.w2.shape, [192, 64])


if __name__ == "__main__":
    unittest.main()
