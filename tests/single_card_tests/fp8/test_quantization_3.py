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
"""Behavior tests for the non-cache quant closures of ``get_quant_func``.

These tests target the parts of ``paddlefleet.fp8.quantization.get_quant_func``
that only manifest when the returned closures are actually *invoked* on a raw
tensor and miss the pre-quantized-weight cache: the ``_mn_major`` (stride-only
``.T``) rewrite of scale tensors and the exact 4-/2-tuple slot mapping.

What is verified, and why it is CPU-observable:

* ``fp8_quant_blockwise`` is a GPU kernel collaborator, NOT the code under test.
  We replace it with a fake that returns four DISTINCT, non-square 2-D marker
  tensors and records the keyword arguments it received. This lets us observe
  how ``get_quant_func``'s closures thread flags into the kernel and how they
  reshape the kernel's outputs -- pure Python/transpose logic that runs on CPU.
* Default (non-UE8M0) ``weight_quant_func``: with ``out_scale_trans=False`` the
  four kernel outputs are returned UNCHANGED (identity, no transpose); with
  ``out_scale_trans=True`` only the two SCALE slots are ``.T``-rewritten while
  the two fp8 slots pass through by identity. Distinct markers + assertIs pin
  which slots are touched and independent numpy ``.T`` pins the values.
* UE8M0 ``inp_quant_func``: ``input_trans=True`` yields a 4-tuple
  ``(fp8, scale.T, fp8_t, scale_t.T)``; ``input_trans=False`` yields the 2-tuple
  ``(fp8, scale.T)`` (kernel output sliced ``[:2]``). fp8 slots are identity,
  scale slots are transposed.
* UE8M0 ``weight_quant_func`` (cache miss) transposes BOTH scale slots and
  drives the kernel with ``quant_method="128x128"`` and the ue8m0/pow2 flags.

Independence: expected scale values are computed with ``numpy.ndarray.T`` on the
marker data this test constructs -- never by calling ``_mn_major`` or any other
part of the function under test. Identity of untouched slots is checked with
``assertIs`` against the exact marker objects handed to the fake kernel.

Scope note: mocking ``fp8_quant_blockwise`` isolates the real GPU kernel; these
tests therefore prove the closure's flag-threading, slot mapping and transpose
rewrite, but do NOT verify the kernel's own quantization numerics.

Environment note: importing ``paddlefleet.fp8.quantization`` imports Paddle at
module load time. This no-card environment has no Paddle installed, so the
tests skip honestly with the captured ImportError rather than passing vacuously.
"""

import unittest
from unittest import mock

try:
    import numpy as np
    import paddle

    from paddlefleet.fp8.quantization import get_quant_func

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on local env
    np = None
    paddle = None
    get_quant_func = None
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    "paddlefleet.fp8.quantization requires paddle at import time; "
    f"paddle is not importable in this environment: {_IMPORT_ERROR}"
)

_KERNEL_ATTR = "fp8_quant_blockwise"


@unittest.skipUnless(get_quant_func is not None, _SKIP_REASON)
class _QuantKernelPatchMixin(unittest.TestCase):
    """Shared fake-kernel plumbing for the invoked-closure tests."""

    def _make_markers(self):
        # Four distinct, non-square tensors so any slot swap or wrong-axis
        # transpose is observable. Shapes chosen so ``.T`` changes the shape.
        return {
            "fp8_bwd": paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "scale_bwd": paddle.to_tensor(
                [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]
            ),
            "fp8_fwd": paddle.to_tensor(
                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
            ),
            "scale_fwd": paddle.to_tensor(
                [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]
            ),
        }

    def _patch_kernel(self, markers):
        """Patch the blockwise kernel with a recording fake.

        Returns a ``captured`` dict that will hold the positional ``x`` and the
        keyword arguments the closure passed to the kernel.
        """
        captured = {}

        def fake_quant(x, **kwargs):
            captured["x"] = x
            captured["kwargs"] = kwargs
            return (
                markers["fp8_bwd"],
                markers["scale_bwd"],
                markers["fp8_fwd"],
                markers["scale_fwd"],
            )

        patcher = mock.patch.object(
            paddle.incubate.nn.functional,
            _KERNEL_ATTR,
            create=True,
            new=fake_quant,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return captured


class TestDefaultWeightQuantNonCache(_QuantKernelPatchMixin):
    """Default-path ``weight_quant_func`` invoked on an un-cached tensor."""

    def test_out_scale_trans_false_returns_kernel_output_unchanged(self):
        markers = self._make_markers()
        captured = self._patch_kernel(markers)
        _, weight_func = get_quant_func("blockwise", out_scale_trans=False)

        raw = object()  # lacks fp8_weight_fwd -> cache miss -> kernel path
        result = weight_func(raw)

        # Kernel receives the raw weight and the hand-derived weight flags.
        self.assertIs(captured["x"], raw)
        self.assertEqual(captured["kwargs"]["quant_method"], "128x128")
        self.assertTrue(captured["kwargs"]["input_transpose"])
        self.assertFalse(captured["kwargs"]["output_scale_transpose"])
        self.assertNotIn("using_ue8m0_scale", captured["kwargs"])

        # No transpose: all four slots returned by identity, in order.
        self.assertEqual(len(result), 4)
        self.assertIs(result[0], markers["fp8_bwd"])
        self.assertIs(result[1], markers["scale_bwd"])
        self.assertIs(result[2], markers["fp8_fwd"])
        self.assertIs(result[3], markers["scale_fwd"])

    def test_out_scale_trans_true_transposes_only_scale_slots(self):
        markers = self._make_markers()
        self._patch_kernel(markers)
        _, weight_func = get_quant_func("blockwise", out_scale_trans=True)

        result = weight_func(object())

        self.assertEqual(len(result), 4)
        # fp8 slots pass through untouched.
        self.assertIs(result[0], markers["fp8_bwd"])
        self.assertIs(result[2], markers["fp8_fwd"])
        # scale slots are .T-rewritten (independent numpy transpose).
        np.testing.assert_array_equal(
            result[1].numpy(), markers["scale_bwd"].numpy().T
        )
        np.testing.assert_array_equal(
            result[3].numpy(), markers["scale_fwd"].numpy().T
        )
        # A transpose of a non-square tensor changes shape: guards against
        # an accidental identity that would still be numerically "equal".
        self.assertEqual(list(result[1].shape), [2, 3])
        self.assertEqual(list(result[3].shape), [3, 2])

    def test_pow2_scale_flag_threads_into_kernel(self):
        markers = self._make_markers()
        captured = self._patch_kernel(markers)
        _, weight_func = get_quant_func("blockwise", pow2_scale=True)
        weight_func(object())
        self.assertTrue(captured["kwargs"]["using_pow2_scale"])


class TestUe8m0InpQuantNonCache(_QuantKernelPatchMixin):
    """UE8M0 ``inp_quant_func`` transpose + slot mapping when invoked."""

    def test_input_trans_true_returns_four_slots_scales_transposed(self):
        markers = self._make_markers()
        captured = self._patch_kernel(markers)
        inp_func, _ = get_quant_func(
            "blockwise", input_trans=True, use_ue8m0=True
        )

        result = inp_func(object())

        self.assertEqual(captured["kwargs"]["quant_method"], "1x128")
        self.assertTrue(captured["kwargs"]["input_transpose"])
        self.assertTrue(captured["kwargs"]["using_ue8m0_scale"])
        self.assertTrue(captured["kwargs"]["using_pow2_scale"])
        self.assertTrue(captured["kwargs"]["output_scale_transpose"])

        # Layout: (fp8, scale.T, fp8_t, scale_t.T) where the fake maps
        # fp8->fp8_bwd, scale->scale_bwd, fp8_t->fp8_fwd, scale_t->scale_fwd.
        self.assertEqual(len(result), 4)
        self.assertIs(result[0], markers["fp8_bwd"])
        np.testing.assert_array_equal(
            result[1].numpy(), markers["scale_bwd"].numpy().T
        )
        self.assertIs(result[2], markers["fp8_fwd"])
        np.testing.assert_array_equal(
            result[3].numpy(), markers["scale_fwd"].numpy().T
        )

    def test_input_trans_false_returns_pair_scale_transposed(self):
        markers = self._make_markers()
        captured = self._patch_kernel(markers)
        inp_func, _ = get_quant_func(
            "blockwise", input_trans=False, use_ue8m0=True
        )

        result = inp_func(object())

        self.assertFalse(captured["kwargs"]["input_transpose"])
        # Only the first two kernel outputs are consumed ([:2]).
        self.assertEqual(len(result), 2)
        self.assertIs(result[0], markers["fp8_bwd"])
        np.testing.assert_array_equal(
            result[1].numpy(), markers["scale_bwd"].numpy().T
        )


class TestUe8m0WeightQuantNonCache(_QuantKernelPatchMixin):
    """UE8M0 ``weight_quant_func`` transposes both scale slots on cache miss."""

    def test_ue8m0_weight_transposes_both_scales(self):
        markers = self._make_markers()
        captured = self._patch_kernel(markers)
        _, weight_func = get_quant_func("blockwise", use_ue8m0=True)

        result = weight_func(object())

        self.assertEqual(captured["kwargs"]["quant_method"], "128x128")
        self.assertTrue(captured["kwargs"]["input_transpose"])
        self.assertTrue(captured["kwargs"]["using_ue8m0_scale"])

        self.assertEqual(len(result), 4)
        self.assertIs(result[0], markers["fp8_bwd"])
        np.testing.assert_array_equal(
            result[1].numpy(), markers["scale_bwd"].numpy().T
        )
        self.assertIs(result[2], markers["fp8_fwd"])
        np.testing.assert_array_equal(
            result[3].numpy(), markers["scale_fwd"].numpy().T
        )


if __name__ == "__main__":
    unittest.main()
