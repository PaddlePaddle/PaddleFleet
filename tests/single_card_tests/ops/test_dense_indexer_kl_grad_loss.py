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

"""Behavior test for ``dense_indexer_kl_bwd`` grad_loss normalisation.

Module under test: ``paddlefleet.cudnn_ops.indexer.dense_indexer_kl_cudnn``.
In the repository module map this is the "计算优化" (Fused Ops / cuDNN)
boundary. The dense indexer backward wrapper runs only on cuDNN-frontend GPU
kernels, which are not present in a no-card environment, so this file does NOT
claim to verify any kernel numerics.

What it DOES verify is the one piece of pure, CPU-observable control flow that
``dense_indexer_kl_bwd`` runs before it ever touches the kernel: the coercion
of the ``grad_loss`` argument into a float32 scalar ``paddle.Tensor``. The
function documents ``grad_scale = loss_coeff * grad_loss / total_q`` as fixed
inside the kernel, so the exact fp32 scalar it hands the kernel is a real
contract, not an implementation detail. The four coercion branches are:

* ``None``                    -> ``paddle.ones([], float32)`` (value 1.0)
* python number               -> ``paddle.to_tensor(float(x), float32)``
* non-fp32 ``paddle.Tensor``  -> ``.cast(float32)`` (value preserved)
* fp32 ``paddle.Tensor``      -> passed through unchanged (same object)

To reach that block on CPU we mock exactly two genuine not-under-test
collaborators: the ``_require_cudnn_frontend`` environment guard (which would
otherwise raise ``ImportError`` off-GPU) and the ``dense_indexer_backward_wrapper``
cuDNN kernel (replaced by a capture stub that records the ``grad_loss`` kwarg
the real coercion produced and returns valid tensors so the wrapper's return
unpacking still runs). The coercion itself is executed for real; we assert the
captured value's dtype, scalar magnitude, rank and object identity per branch.
"""

import sys
import types
import unittest
from unittest import mock

try:
    import paddle

    from paddlefleet.cudnn_ops.indexer import dense_indexer_kl_cudnn as mod

    _IMPORT_ERROR = None
except (
    ImportError
) as exc:  # no paddle / paddlefleet_ops available -> skip honestly
    paddle = None
    mod = None
    _IMPORT_ERROR = exc

_HAVE_MODULE = mod is not None
_SKIP_REASON = (
    "paddle / paddlefleet.cudnn_ops.indexer.dense_indexer_kl_cudnn not importable"
    f" ({_IMPORT_ERROR})"
)

_API = "paddlefleet_ops.cudnn.deepseek_sparse_attention.indexer_backward.api"


@unittest.skipUnless(_HAVE_MODULE, _SKIP_REASON)
class TestDenseIndexerKlBwdGradLoss(unittest.TestCase):
    """Exercises the CPU-side grad_loss coercion of ``dense_indexer_kl_bwd``."""

    def _run(self, grad_loss):
        """Call the real backward with ``grad_loss``; capture the coerced value.

        The kernel wrapper is replaced by a stub that records the ``grad_loss``
        kwarg it receives -- i.e. the exact tensor the production coercion
        produced -- and returns the three output tensors the caller unpacks.
        Fresh input tensors are built per call so no in-place aliasing carries
        between cases.
        """
        index_q = paddle.ones([4, 2, 8], dtype="bfloat16")
        weights = paddle.ones([4, 2], dtype="bfloat16")
        index_k = paddle.ones([16, 8], dtype="bfloat16")
        attn_score = paddle.ones([4, 16], dtype="float32")
        attn_l1norm = paddle.ones([4], dtype="float32")
        index_score = paddle.ones([4, 16], dtype="float32")
        index_lse = paddle.ones([4, 2], dtype="float32")
        cu_q = paddle.to_tensor([0, 4], dtype="int32")
        cu_k = paddle.to_tensor([0, 16], dtype="int32")

        captured = {}

        def _capture_wrapper(*args, **kwargs):
            captured["grad_loss"] = kwargs["grad_loss"]
            return {
                "d_index_q": paddle.zeros([4, 2, 8], dtype="float32"),
                "d_weights": paddle.zeros([4, 2], dtype="float32"),
                "d_index_k": paddle.zeros([16, 8], dtype="float32"),
            }

        fake_api = types.ModuleType(_API)
        fake_api.dense_indexer_backward_wrapper = _capture_wrapper

        with (
            mock.patch.dict(sys.modules, {_API: fake_api}),
            mock.patch.object(mod, "_require_cudnn_frontend", lambda: None),
        ):
            mod.dense_indexer_kl_bwd(
                index_q,
                weights,
                index_k,
                attn_score,
                attn_l1norm,
                index_score,
                index_lse,
                1.0,
                cu_q,
                cu_k,
                4,
                16,
                grad_loss=grad_loss,
            )

        self.assertIn("grad_loss", captured)  # the wrapper actually ran
        return captured["grad_loss"]

    def _assert_fp32_scalar(self, tensor, expected_value):
        """A float32 0-d tensor carrying ``expected_value``."""
        self.assertIsInstance(tensor, paddle.Tensor)
        self.assertEqual(tensor.dtype, paddle.float32)
        self.assertEqual(len(tensor.shape), 0)  # scalar, not e.g. [1]
        self.assertEqual(float(tensor), expected_value)

    def test_none_becomes_fp32_one(self):
        # None -> paddle.ones([], float32): the default grad_loss is exactly 1.0,
        # so grad_scale reduces to loss_coeff / total_q. A wrong default (0.0, or
        # a [1]-shaped tensor) would rescale every gradient the kernel emits.
        out = self._run(None)
        self._assert_fp32_scalar(out, 1.0)

    def test_python_float_is_coerced_to_fp32_tensor(self):
        # 2.5 (python float) -> to_tensor(float(2.5), float32), value preserved.
        out = self._run(2.5)
        self._assert_fp32_scalar(out, 2.5)

    def test_python_int_is_floated_then_coerced(self):
        # 3 (python int) -> to_tensor(float(3), float32) == 3.0. Guards the
        # float() conversion in the non-Tensor branch.
        out = self._run(3)
        self._assert_fp32_scalar(out, 3.0)

    def test_non_fp32_tensor_is_cast_preserving_value(self):
        # float64 tensor -> .cast(float32): dtype must change to float32 while
        # the magnitude survives. A skipped cast would leave float64; a broken
        # cast would corrupt the value.
        src = paddle.to_tensor(5.0, dtype="float64")
        out = self._run(src)
        self.assertEqual(out.dtype, paddle.float32)
        self.assertNotEqual(src.dtype, out.dtype)  # a new, recast tensor
        self.assertIsNot(out, src)
        self.assertEqual(float(out), 5.0)

    def test_fp32_tensor_is_passed_through_unchanged(self):
        # An already-fp32 tensor hits no coercion branch: the SAME object must
        # reach the kernel. assertIs distinguishes the no-op path from an
        # implementation that needlessly re-casts / copies every call.
        src = paddle.to_tensor(7.0, dtype="float32")
        out = self._run(src)
        self.assertIs(out, src)
        self.assertEqual(out.dtype, paddle.float32)
        self.assertEqual(float(out), 7.0)


if __name__ == "__main__":
    unittest.main()
