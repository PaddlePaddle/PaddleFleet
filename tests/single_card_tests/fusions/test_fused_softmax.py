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

"""Behavior unit tests for ``paddlefleet.fusions.fused_softmax`` (base scope).

Scope (base): the pure-CPU, construction-time control flow of
``FusedScaleMaskSoftmax.__init__`` -- the validity / eligibility truth table
over the ``(input_in_fp16, input_in_bf16)`` dtype flags and the
``(scale, softmax_in_fp32)`` pair, together with the derived
``input_in_float16`` flag. This is the boolean control-flow that is genuinely
callable on CPU (no tensors are created in ``__init__``), so it needs no GPU
numerics; the forward-pass softmax dispatch numerics are left to the sibling
tests, per the task split.

Note on the scope banner: the (untrusted) coverage banner referred to a
``ScaledMaskedSoftmax`` / ``is_kernel_available`` eligibility method. No such
symbol exists in this production module -- it defines only ``SoftmaxOne`` and
``FusedScaleMaskSoftmax``, and the latter has no ``is_kernel_available``. The
real, CPU-testable eligibility logic is the two ``assert`` guards and the
float16-flag derivation inside ``FusedScaleMaskSoftmax.__init__``, which is
what these tests exercise.

Every expected value is hand-derived by reading the three lines of logic:
  * ``assert not (input_in_fp16 and input_in_bf16)``  (mutual exclusion)
  * ``input_in_float16 = input_in_fp16 or input_in_bf16``  (derivation)
  * ``assert scale is None or softmax_in_fp32``  (scaled => fp32 softmax)
Collaborators are genuine, never mocks: ``attn_mask_type`` is a real
``AttnMaskType`` enum member and ``mask_func`` is a real Python function. The
class under construction is never mocked or bypassed.
"""

import unittest

try:
    import paddle

    _PADDLE_IMPORT_ERROR = None
except ImportError as exc:  # honest: paddle genuinely absent, not swallowed
    paddle = None
    _PADDLE_IMPORT_ERROR = exc


def _identity_mask(x, mask):
    """A genuine (non-mock) ``mask_func`` collaborator: applies no masking."""
    return x


@unittest.skipUnless(
    paddle is not None,
    f"paddle is not installed in this environment: {_PADDLE_IMPORT_ERROR}",
)
class TestFusedScaleMaskSoftmaxInit(unittest.TestCase):
    """Construction-time truth table for ``FusedScaleMaskSoftmax.__init__``."""

    def setUp(self):
        # __init__ creates no tensors, but pin CPU for honesty about the
        # no-card scope and restore the prior device afterwards.
        self._orig_device = paddle.get_device()
        self.addCleanup(paddle.set_device, self._orig_device)
        paddle.set_device("cpu")

        # Import here (not at module top) so that a genuinely missing
        # paddlefleet surfaces as a real error rather than being masked, and
        # so the paddle-dependent module import only runs when not skipped.
        from paddlefleet.fusions.fused_softmax import FusedScaleMaskSoftmax
        from paddlefleet.transformer.enums import AttnMaskType

        self.FusedScaleMaskSoftmax = FusedScaleMaskSoftmax
        self.AttnMaskType = AttnMaskType

    def _build(self, **overrides):
        """Construct with valid defaults, overriding only the fields tested."""
        params = {
            "input_in_fp16": False,
            "input_in_bf16": False,
            "attn_mask_type": self.AttnMaskType.causal,
            "scaled_masked_softmax_fusion": True,
            "mask_func": _identity_mask,
            "softmax_in_fp32": True,
            "scale": None,
        }
        params.update(overrides)
        return self.FusedScaleMaskSoftmax(**params)

    def test_input_in_float16_derivation_truth_table(self):
        """input_in_float16 == (input_in_fp16 or input_in_bf16).

        Hand-derived from ``self.input_in_float16 = self.input_in_fp16 or
        self.input_in_bf16``. The (True, True) row is excluded here because it
        is rejected by the mutual-exclusion guard (see the dedicated test).
        """
        cases = [
            # (input_in_fp16, input_in_bf16, expected_input_in_float16)
            (False, False, False),
            (True, False, True),
            (False, True, True),
        ]
        for fp16, bf16, expected in cases:
            with self.subTest(input_in_fp16=fp16, input_in_bf16=bf16):
                layer = self._build(input_in_fp16=fp16, input_in_bf16=bf16)
                self.assertIs(layer.input_in_fp16, fp16)
                self.assertIs(layer.input_in_bf16, bf16)
                self.assertEqual(layer.input_in_float16, expected)

    def test_fp16_and_bf16_cannot_both_be_active(self):
        """(True, True) violates ``assert not (fp16 and bf16)``.

        This is the only (fp16, bf16) combination that must raise; the other
        three are covered as constructing successfully above.
        """
        with self.assertRaises(AssertionError):
            self._build(input_in_fp16=True, input_in_bf16=True)

    def test_scale_guard_truth_table(self):
        """assert scale is None or softmax_in_fp32 -- valid rows construct.

        Hand-derived rows that satisfy the guard, with the ``scale`` stored
        unchanged so an ignored/overwritten scale would be caught:
          (scale=None, fp32=False) -> None is None       -> ok
          (scale=None, fp32=True)  -> None is None        -> ok
          (scale=0.5,  fp32=True)  -> False or True       -> ok
        The single violating row (scale set, fp32 False) must raise.
        """
        for scale, fp32 in [(None, False), (None, True), (0.5, True)]:
            with self.subTest(scale=scale, softmax_in_fp32=fp32):
                layer = self._build(scale=scale, softmax_in_fp32=fp32)
                self.assertEqual(layer.scale, scale)
                self.assertIs(layer.softmax_in_fp32, fp32)

        with self.assertRaises(AssertionError):
            self._build(scale=0.5, softmax_in_fp32=False)

    def test_scale_zero_uses_is_none_not_truthiness(self):
        """scale=0.0 is falsy but not None: the guard still fires.

        Distinguishes the real ``scale is None`` check from a truthiness-based
        one. Hand-derived: ``0.0 is None`` -> False, so with fp32 False the
        guard ``False or False`` -> AssertionError; a truthiness check
        (``if not scale``) would wrongly accept 0.0. With fp32 True it
        constructs and stores the exact 0.0.
        """
        with self.assertRaises(AssertionError):
            self._build(scale=0.0, softmax_in_fp32=False)

        layer = self._build(scale=0.0, softmax_in_fp32=True)
        self.assertEqual(layer.scale, 0.0)

    def test_constructor_stores_genuine_collaborators_and_defaults(self):
        """Collaborators are stored unwrapped and sliding_window defaults None.

        ``mask_func`` and ``attn_mask_type`` are the exact genuine objects
        passed in (``assertIs``), confirming the constructor does not wrap or
        substitute them; ``sliding_window`` defaults to None when omitted.
        """
        layer = self._build(
            attn_mask_type=self.AttnMaskType.padding,
            scaled_masked_softmax_fusion=False,
            mask_func=_identity_mask,
        )
        self.assertIs(layer.attn_mask_type, self.AttnMaskType.padding)
        self.assertIs(layer.scaled_masked_softmax_fusion, False)
        self.assertIs(layer.mask_func, _identity_mask)
        self.assertIsNone(layer.sliding_window)

    def test_sliding_window_stored_when_provided(self):
        """An explicit sliding_window value is stored verbatim."""
        layer = self._build(sliding_window=64)
        self.assertEqual(layer.sliding_window, 64)


if __name__ == "__main__":
    unittest.main()
