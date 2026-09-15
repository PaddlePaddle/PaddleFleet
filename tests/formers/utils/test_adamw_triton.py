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

"""Behaviour tests for ``paddlefleet.utils.adamw_triton``.

Two layers of behaviour live in this module:

* ``adamw_triton(...)`` is a *dispatcher*: it short-circuits on ``skip_update``,
  zeroes ``coeff`` when decay is off, drops the master weight when
  ``multi_precision`` is off, scales the learning rate by ``lr_ratio``, selects
  ``adamw_kernel`` vs ``adamw_kernel_skip``, marshals the paddle->triton dtype
  mapping, and finally advances ``beta1_pow``/``beta2_pow``. All of that is plain
  Python and is verified here on CPU (无卡). The GPU-only triton kernel is
  isolated behind a marker recorder so we can assert the *arguments* the
  dispatcher hands it -- these tests do NOT validate the kernel's numeric output.

* ``adamw_kernel`` performs the actual AdamW math and needs a real GPU + triton
  runtime. The numeric contract is expressed in a hand-derived test that is
  marked ``@unittest.skip`` (GPU-only); it is never faked on CPU.
"""

import unittest

try:
    import paddle
except ImportError:  # pragma: no cover - paddle missing
    paddle = None

# Importing the production module raises RuntimeError (its documented contract)
# when triton / use-triton-in-paddle are not installed. Treat only those precise
# signals as "triton unavailable"; anything else should surface as a real error.
_IMPORT_ERROR = None
_TRITON_AVAILABLE = False
if paddle is not None:
    try:
        import paddlefleet.utils.adamw_triton as adamw_triton_mod
        from paddlefleet.utils.adamw_triton import DTYPE_MAPPING, adamw_triton

        tl = adamw_triton_mod.tl
        _TRITON_AVAILABLE = True
    except (ImportError, RuntimeError) as exc:
        _IMPORT_ERROR = repr(exc)
else:
    _IMPORT_ERROR = "paddle is not importable"

_SKIP_REASON = "triton/use-triton-in-paddle not available: %s" % _IMPORT_ERROR


class _KernelRecorder:
    """Stand-in for a ``@triton.jit`` kernel.

    ``kernel[grid](*args)`` is how the dispatcher launches; we record the grid
    and the launch arguments so the CPU tests can assert exactly what the
    dispatcher passed. The recorder never touches a GPU.
    """

    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def _launch(*args, **kwargs):
            self.launches.append({"grid": grid, "args": args, "kwargs": kwargs})

        return _launch


@unittest.skipUnless(_TRITON_AVAILABLE, _SKIP_REASON)
class TestDtypeMapping(unittest.TestCase):
    """The dtype table decides the triton store dtype for param/moment."""

    def test_exact_paddle_to_triton_pairs(self):
        # A swapped pairing (e.g. bf16 -> float16) would silently corrupt the
        # stored moment/param precision, so pin every pair, not just presence.
        self.assertEqual(
            set(DTYPE_MAPPING.keys()),
            {paddle.bfloat16, paddle.float32, paddle.float16},
        )
        self.assertIs(DTYPE_MAPPING[paddle.bfloat16], tl.bfloat16)
        self.assertIs(DTYPE_MAPPING[paddle.float32], tl.float32)
        self.assertIs(DTYPE_MAPPING[paddle.float16], tl.float16)


@unittest.skipUnless(_TRITON_AVAILABLE, _SKIP_REASON)
class TestAdamWTritonDispatch(unittest.TestCase):
    """CPU control-flow / argument-marshalling of the dispatcher.

    The triton kernels are replaced with a marker recorder: these tests prove
    the dispatcher selects the right kernel and forwards the right arguments and
    that it advances ``beta_pow`` afterwards. They intentionally do NOT prove the
    GPU kernel computes AdamW correctly (see ``TestAdamWKernelNumeric``).
    """

    # positional layout of the non-skip kernel launch arguments
    _P, _G, _M1, _M2, _LR, _B1, _B2, _EPS, _COEFF = 0, 1, 2, 3, 4, 5, 6, 7, 8
    _B1POW, _B2POW, _MW, _N, _SKIPP, _PDTYPE, _MDTYPE, _BLOCK = (
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
    )

    def _patch_kernels(self):
        main = _KernelRecorder()
        skip = _KernelRecorder()
        self._orig_main = adamw_triton_mod.adamw_kernel
        self._orig_skip = adamw_triton_mod.adamw_kernel_skip
        adamw_triton_mod.adamw_kernel = main
        adamw_triton_mod.adamw_kernel_skip = skip
        self.addCleanup(
            setattr, adamw_triton_mod, "adamw_kernel", self._orig_main
        )
        self.addCleanup(
            setattr, adamw_triton_mod, "adamw_kernel_skip", self._orig_skip
        )
        return main, skip

    def _tensors(self, dtype="float32"):
        param = paddle.ones([4], dtype=dtype)
        grad = paddle.full([4], 0.1, dtype=dtype)
        moment1 = paddle.zeros([4], dtype=dtype)
        moment2 = paddle.zeros([4], dtype=dtype)
        master = paddle.ones([4], dtype="float32")
        return param, grad, moment1, moment2, master

    def test_skip_update_short_circuits_without_touching_state(self):
        # No mocking here: this exercises the real early-return branch.
        param, grad, moment1, moment2, master = self._tensors()
        beta1_pow = paddle.to_tensor([0.9], dtype="float32")
        beta2_pow = paddle.to_tensor([0.999], dtype="float32")
        b1_before = beta1_pow.numpy().copy()
        b2_before = beta2_pow.numpy().copy()
        p_before = param.numpy().copy()

        main, skip = self._patch_kernels()
        ret = adamw_triton(
            param,
            grad,
            0.1,
            moment1,
            moment2,
            beta1_pow,
            beta2_pow,
            master,
            True,  # skip_update truthy -> must return immediately
            0.9,
            0.999,
            1e-8,
            1.0,
            0.0,
            True,
            True,
            False,
        )

        self.assertIsNone(ret)
        self.assertEqual(main.launches, [])
        self.assertEqual(skip.launches, [])
        # beta_pow advance must NOT happen when the step is skipped
        self.assertTrue((beta1_pow.numpy() == b1_before).all())
        self.assertTrue((beta2_pow.numpy() == b2_before).all())
        self.assertTrue((param.numpy() == p_before).all())

    def test_passthrough_with_decay_and_multiprecision(self):
        param, grad, moment1, moment2, master = self._tensors()
        beta1_pow = paddle.to_tensor([0.9], dtype="float32")
        beta2_pow = paddle.to_tensor([0.999], dtype="float32")

        main, skip = self._patch_kernels()
        adamw_triton(
            param,
            grad,
            0.1,
            moment1,
            moment2,
            beta1_pow,
            beta2_pow,
            master,
            False,  # skip_update falsey
            0.9,
            0.999,
            1e-8,
            2.0,
            0.01,
            True,
            True,
            False,
        )

        self.assertEqual(len(main.launches), 1)
        self.assertEqual(skip.launches, [])
        args = main.launches[0]["args"]
        # lr = learning_rate * lr_ratio
        self.assertAlmostEqual(float(args[self._LR]), 0.1 * 2.0, places=6)
        # decay kept, master weight kept, N = numel, skip flag forwarded
        self.assertAlmostEqual(float(args[self._COEFF]), 0.01, places=6)
        self.assertIs(args[self._MW], master)
        self.assertEqual(args[self._N], 4)
        self.assertIs(args[self._P], param)
        self.assertIs(args[self._G], grad)
        self.assertFalse(args[self._SKIPP])
        # dtype marshalling: param dtype -> triton dtype, moment dtype -> triton
        self.assertIs(args[self._PDTYPE], DTYPE_MAPPING[param.dtype])
        self.assertIs(args[self._MDTYPE], DTYPE_MAPPING[moment1.dtype])
        # beta_pow advanced by one factor of beta after the launch
        self.assertAlmostEqual(float(beta1_pow[0]), 0.9 * 0.9, places=6)
        self.assertAlmostEqual(float(beta2_pow[0]), 0.999 * 0.999, places=6)

    def test_decay_and_masterweight_disabled(self):
        param, grad, moment1, moment2, master = self._tensors()
        beta1_pow = paddle.to_tensor([0.9], dtype="float32")
        beta2_pow = paddle.to_tensor([0.999], dtype="float32")

        main, skip = self._patch_kernels()
        adamw_triton(
            param,
            grad,
            0.1,
            moment1,
            moment2,
            beta1_pow,
            beta2_pow,
            master,
            False,
            0.9,
            0.999,
            1e-8,
            1.0,
            0.01,  # coeff supplied ...
            False,  # ... but with_decay=False must zero it
            False,  # multi_precision=False must drop the master weight
            False,
        )

        self.assertEqual(len(main.launches), 1)
        args = main.launches[0]["args"]
        self.assertEqual(float(args[self._COEFF]), 0.0)
        self.assertIsNone(args[self._MW])

    def test_skip_update_param_routes_to_skip_kernel(self):
        param, grad, moment1, moment2, master = self._tensors()
        beta1_pow = paddle.to_tensor([0.9], dtype="float32")
        beta2_pow = paddle.to_tensor([0.999], dtype="float32")

        main, skip = self._patch_kernels()
        adamw_triton(
            param,
            grad,
            0.1,
            moment1,
            moment2,
            beta1_pow,
            beta2_pow,
            master,
            False,
            0.9,
            0.999,
            1e-8,
            1.0,
            0.0,
            True,
            True,
            True,  # skip_update_param -> adamw_kernel_skip
        )

        self.assertEqual(main.launches, [])
        self.assertEqual(len(skip.launches), 1)
        skip_args = skip.launches[0]["args"]
        # skip kernel signature starts with grad (no param slot)
        self.assertIs(skip_args[0], grad)
        self.assertIs(skip_args[1], moment1)
        self.assertIs(skip_args[2], moment2)
        self.assertIs(skip_args[10], master)  # master_weight slot
        self.assertEqual(skip_args[11], 4)  # N
        self.assertTrue(skip_args[12])  # skip_update_param
        self.assertIs(skip_args[13], DTYPE_MAPPING[moment1.dtype])
        # beta_pow still advances on the skip-param path
        self.assertAlmostEqual(float(beta1_pow[0]), 0.9 * 0.9, places=6)
        self.assertAlmostEqual(float(beta2_pow[0]), 0.999 * 0.999, places=6)


class TestAdamWKernelNumeric(unittest.TestCase):
    """GPU-only numeric contract of the triton AdamW kernel.

    The kernel updates: p*= (1-lr*coeff); m1 = b1*m1+(1-b1)*g;
    m2 = b2*m2+(1-b2)*g^2; denom = sqrt(m2)/sqrt(1-b2_pow)+eps;
    p += (m1/denom)*(-lr/(1-b1_pow)). The expected values below are derived
    independently from that AdamW spec with numpy (never by calling the fn under
    test). Marked skip because the triton kernel requires a real GPU; we do not
    fake GPU numerics on CPU.
    """

    @unittest.skip(
        "Triton AdamW kernel is GPU-only; numeric update cannot be validated on CPU"
    )
    def test_single_step_matches_hand_derived(self):  # pragma: no cover
        import numpy as np

        paddle.set_device("gpu")
        p0 = np.array([1.0, 2.0, -1.0, 0.5], dtype=np.float32)
        g = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        lr, coeff, lr_ratio = 0.1, 0.01, 1.0
        b1_pow0, b2_pow0 = 0.9, 0.999

        # --- independent AdamW reference (no call to adamw_triton) ---
        lr_eff = lr * lr_ratio
        ref_p = p0 * (1.0 - lr_eff * coeff)
        ref_m1 = beta1 * 0.0 + (1.0 - beta1) * g
        ref_m2 = beta2 * 0.0 + (1.0 - beta2) * g * g
        denom = np.sqrt(ref_m2) / np.sqrt(1.0 - b2_pow0) + eps
        ref_p = ref_p + (ref_m1 / denom) * (-lr_eff / (1.0 - b1_pow0))
        ref_b1_pow = beta1 * b1_pow0
        ref_b2_pow = beta2 * b2_pow0

        param = paddle.to_tensor(p0, dtype="float32")
        grad = paddle.to_tensor(g, dtype="float32")
        moment1 = paddle.zeros([4], dtype="float32")
        moment2 = paddle.zeros([4], dtype="float32")
        beta1_pow = paddle.to_tensor([b1_pow0], dtype="float32")
        beta2_pow = paddle.to_tensor([b2_pow0], dtype="float32")

        adamw_triton(
            param,
            grad,
            lr,
            moment1,
            moment2,
            beta1_pow,
            beta2_pow,
            None,
            False,  # master_weight None, skip_update False
            beta1,
            beta2,
            eps,
            lr_ratio,
            coeff,
            True,
            False,
            False,
        )

        np.testing.assert_allclose(param.numpy(), ref_p, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            moment1.numpy(), ref_m1, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            moment2.numpy(), ref_m2, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(beta1_pow.numpy(), [ref_b1_pow], rtol=1e-6)
        np.testing.assert_allclose(beta2_pow.numpy(), [ref_b2_pow], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
