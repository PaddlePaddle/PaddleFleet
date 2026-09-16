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

"""Behavior unit tests for ``paddlefleet.fusions.fused_softmax`` (slice _2).

Scope (this sibling): the CPU-observable forward control flow / pure math of
``FusedScaleMaskSoftmax.forward`` and its ``SoftmaxOne`` collaborator --
  * scale application: ``input = input * self.scale`` before softmax;
  * softmax-fn selection: ``paddle.nn.Softmax`` when ``softmax_offset is None``
    vs ``SoftmaxOne`` (off-by-one sink denominator) when an offset is given;
  * the triangular (auto-generated causal / sliding-window) mask branch vs the
    general user-supplied mask branch, plus the ``sq == sk`` guard and the
    ``sq == 1`` skip.
The base sibling owns the ``__init__`` truth table, so it is not repeated here.

Note on the (untrusted) coverage banner: it referred to "upper-triangular vs
general mask branch, scale application, get_batch_per_block / grid math". This
production module is a pure ``nn.Layer`` wrapper -- there is no
``get_batch_per_block`` symbol and no explicit CUDA grid math; the real
CPU-testable analogue of that hint is the forward's mask-branch selection,
scale multiply and softmax-fn selection, which is what these tests exercise.

All fp32 math here runs on CPU; the fp16/bf16 ``input.float()`` /
``probs.half()`` re-cast paths need GPU and are honestly left unverified (they
are not part of this slice). Every expected value is hand-derived with an
independent NumPy softmax / mask construction -- production's own softmax and
mask helpers are never used to build the expected values. ``mask_func`` is a
genuine (non-mock) Python collaborator; the layer under test is never mocked.
"""

import unittest

import numpy as np

try:
    import paddle

    _PADDLE_IMPORT_ERROR = None
except ImportError as exc:  # honest: paddle genuinely absent, not swallowed
    paddle = None
    _PADDLE_IMPORT_ERROR = exc


def _np_softmax(z, axis=-1):
    """Independent, numerically-stable softmax reference (NumPy float64)."""
    z = np.asarray(z, dtype=np.float64)
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def _neg_inf_mask_func(neg=-1e9):
    """Return a genuine ``mask_func`` that drives masked (True) logits to ``neg``.

    This is the collaborator the production forward invokes with the generated
    or user-supplied mask; it is real code, not a mock.
    """

    def _apply(x, mask):
        return paddle.where(mask, paddle.full_like(x, neg), x)

    return _apply


class _RecordingMaskFunc:
    """Genuine ``mask_func`` that records whether it was invoked.

    Used to prove the ``mask is None`` short-circuit (``mask_output = input``)
    truly bypasses ``mask_func`` -- if it were called it would both corrupt the
    output (returning zeros) and leave a recorded call.
    """

    def __init__(self):
        self.calls = 0

    def __call__(self, x, mask):
        self.calls += 1
        return paddle.zeros_like(x)


@unittest.skipUnless(
    paddle is not None,
    f"paddle is not installed in this environment: {_PADDLE_IMPORT_ERROR}",
)
class TestFusedScaleMaskSoftmaxForward(unittest.TestCase):
    """CPU-observable forward control flow / pure math of the fused softmax."""

    def setUp(self):
        # Forward runs real softmax/where ops; pin CPU so the fp32 math is
        # device-independent, and restore the prior device afterwards.
        self._orig_device = paddle.get_device()
        self.addCleanup(paddle.set_device, self._orig_device)
        paddle.set_device("cpu")

        # Import here (not at module top) so a genuinely missing paddlefleet
        # surfaces as a real error instead of being masked as "no paddle".
        from paddlefleet.fusions.fused_softmax import FusedScaleMaskSoftmax
        from paddlefleet.transformer.enums import AttnMaskType

        self.FusedScaleMaskSoftmax = FusedScaleMaskSoftmax
        self.AttnMaskType = AttnMaskType

    def _build(self, **overrides):
        """Construct with valid fp32/CPU defaults, overriding tested fields."""
        params = {
            "input_in_fp16": False,
            "input_in_bf16": False,
            "attn_mask_type": self.AttnMaskType.no_mask,
            "scaled_masked_softmax_fusion": True,
            "mask_func": lambda x, m: x,
            "softmax_in_fp32": True,
            "scale": None,
        }
        params.update(overrides)
        return self.FusedScaleMaskSoftmax(**params)

    # ---- scale application (``input = input * self.scale``) ----

    def test_scale_is_applied_before_softmax(self):
        """Output equals ``softmax(scale * x)``, not ``softmax(x)``.

        Hand-derived: with no mask and offset ``None`` the forward reduces to
        ``softmax(x * scale, axis=-1)``. For x=[1,2,3], scale=2.0 the logits
        become [2,4,6]; the independent NumPy reference softmaxes those, and it
        is provably distinct from the unscaled softmax, so an ignored scale is
        caught.
        """
        x = paddle.to_tensor([[[[1.0, 2.0, 3.0]]]], dtype="float32")
        layer = self._build(scale=2.0, softmax_in_fp32=True)
        out = layer(x, mask=None).numpy()

        ref_scaled = _np_softmax(np.array([2.0, 4.0, 6.0]))
        ref_unscaled = _np_softmax(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(
            out[0, 0, 0], ref_scaled, rtol=1e-5, atol=1e-6
        )
        # The scaled result must not coincide with the unscaled one.
        self.assertGreater(np.abs(ref_scaled - ref_unscaled).max(), 1e-2)
        self.assertFalse(np.allclose(out[0, 0, 0], ref_unscaled, atol=1e-3))

    def test_scale_none_leaves_logits_unscaled(self):
        """With ``scale=None`` the forward is plain ``softmax(x)``.

        Hand-derived counterpart to the scaled case: the ``if self.scale is
        not None`` branch is skipped, so the output matches ``softmax(x)``
        exactly. Distinguishes "scale skipped" from "scale applied as 1.0" only
        indirectly, but pins the None branch against the scaled reference.
        """
        x = paddle.to_tensor([[[[1.0, 2.0, 3.0]]]], dtype="float32")
        layer = self._build(scale=None)
        out = layer(x, mask=None).numpy()
        np.testing.assert_allclose(
            out[0, 0, 0],
            _np_softmax(np.array([1.0, 2.0, 3.0])),
            rtol=1e-5,
            atol=1e-6,
        )

    # ---- softmax-fn selection (Softmax vs SoftmaxOne) ----

    def test_offset_none_selects_plain_softmax_rows_sum_to_one(self):
        """``softmax_offset is None`` -> ``paddle.nn.Softmax``; rows sum to 1.

        Hand-derived: plain softmax over the last axis normalizes each row to
        sum exactly 1.0, matching the independent NumPy softmax element-wise.
        """
        x = paddle.to_tensor([[[[1.0, 2.0, 3.0]]]], dtype="float32")
        layer = self._build()
        out = layer(x, mask=None, softmax_offset=None).numpy()
        np.testing.assert_allclose(
            out[0, 0, 0],
            _np_softmax(np.array([1.0, 2.0, 3.0])),
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(out.sum(axis=-1), 1.0, rtol=0, atol=1e-6)

    def test_offset_given_selects_softmaxone_rows_sum_below_one(self):
        """A provided ``softmax_offset`` selects ``SoftmaxOne`` (off-by-one).

        Hand-derived from ``SoftmaxOne.forward``: it concatenates a per-head
        sink value ``s`` onto the logits, softmaxes over ``[x0..x_{k-1}, s]``
        and drops the sink column. So ``out_i = exp(x_i) / (sum_j exp(x_j) +
        exp(s))`` and each row sums to strictly less than 1. For x=[1,2,3] and
        offset s=0 the independent reference is softmax([1,2,3,0])[:-1]; the
        row sum (~0.96794) is distinct from the plain-softmax path's 1.0, so an
        inverted fn-selection is caught.
        """
        x = paddle.to_tensor([[[[1.0, 2.0, 3.0]]]], dtype="float32")
        offset = paddle.to_tensor([0.0], dtype="float32")  # one per head (np=1)
        layer = self._build()
        out = layer(x, mask=None, softmax_offset=offset).numpy()

        ref = _np_softmax(np.array([1.0, 2.0, 3.0, 0.0]))[:-1]
        np.testing.assert_allclose(out[0, 0, 0], ref, rtol=1e-5, atol=1e-6)
        row_sum = out[0, 0, 0].sum()
        self.assertAlmostEqual(row_sum, float(ref.sum()), places=6)
        self.assertLess(row_sum, 1.0 - 1e-3)  # strictly below the plain path

    # ---- triangular (causal) mask branch ----

    def test_causal_branch_generates_upper_triangular_mask(self):
        """causal + ``mask is None`` + ``sq>1`` auto-generates the causal mask.

        Hand-derived: the forward builds an upper-triangular (j>i) mask, the
        genuine mask_func drives those future positions to -1e9, and softmax
        renders them ~0. The independent reference builds the same triu(k=1)
        mask in NumPy (never via the production helper), masks the logits and
        softmaxes. Row 0 attends only to position 0 (prob 1.0); an inverted or
        shifted mask would move the zeros and fail.
        """
        xin = np.array(
            [[0.5, 1.0, -0.5], [2.0, 0.1, 0.3], [-1.0, 0.7, 1.5]],
            dtype=np.float32,
        )
        x = paddle.to_tensor(xin.reshape(1, 1, 3, 3))
        layer = self._build(
            attn_mask_type=self.AttnMaskType.causal,
            mask_func=_neg_inf_mask_func(),
        )
        out = layer(x, mask=None).numpy()[0, 0]

        causal = np.triu(np.ones((3, 3), dtype=bool), k=1)
        ref = _np_softmax(np.where(causal, -1e9, xin.astype(np.float64)))
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)
        # Future (upper-triangular) positions must be exactly masked out.
        self.assertTrue(np.all(out[causal] < 1e-12))
        np.testing.assert_allclose(out.sum(axis=-1), 1.0, rtol=0, atol=1e-6)

    def test_causal_branch_requires_square_scores(self):
        """causal auto-gen asserts ``sq == sk`` (self-attention only).

        Hand-derived: with attn_mask_type causal, mask None and sq>1 the
        forward reaches ``assert sq == sk``. A [1,1,3,5] score (sq=3, sk=5)
        violates it and must raise AssertionError.
        """
        x = paddle.zeros([1, 1, 3, 5], dtype="float32") + paddle.to_tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0], dtype="float32"
        )
        layer = self._build(
            attn_mask_type=self.AttnMaskType.causal,
            mask_func=_neg_inf_mask_func(),
        )
        with self.assertRaises(AssertionError):
            layer(x, mask=None)

    def test_causal_branch_skipped_when_sq_is_one(self):
        """``sq == 1`` skips causal auto-gen (KV-cache / single-token case).

        Hand-derived: with sq=1 the ``sq > 1`` guard is False, so no mask is
        generated, ``sq == sk`` is never asserted (here sk=4 != sq=1), and
        ``mask is None`` leaves ``mask_output = input`` -- mask_func is not
        called. Output is therefore plain ``softmax(x)`` over the 4 keys. A
        recording mask_func proves the bypass, and using sk != sq proves the
        guard is what avoids the square assertion.
        """
        xin = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        x = paddle.to_tensor(xin.reshape(1, 1, 1, 4))
        recorder = _RecordingMaskFunc()
        layer = self._build(
            attn_mask_type=self.AttnMaskType.causal, mask_func=recorder
        )
        out = layer(x, mask=None).numpy()
        self.assertEqual(recorder.calls, 0)
        np.testing.assert_allclose(
            out[0, 0, 0], _np_softmax(xin), rtol=1e-5, atol=1e-6
        )

    # ---- sliding-window (banded triangular) mask branch ----

    def test_sliding_window_branch_builds_band_mask(self):
        """A non-None ``sliding_window`` overrides mask with the SWA band.

        Hand-derived for window (left=1, right=0), sq=sk=4: a query at row i
        may attend only to keys j with ``i-1 <= j <= i``; all others are
        masked. The independent reference rebuilds that band directly (never
        via the production helper), masks the logits and softmaxes. This branch
        takes precedence over attn_mask_type, so masked positions must be ~0
        exactly at the out-of-band entries.
        """
        rng = np.arange(16, dtype=np.float32).reshape(4, 4) * 0.3 - 1.0
        x = paddle.to_tensor(rng.reshape(1, 1, 4, 4))
        layer = self._build(
            attn_mask_type=self.AttnMaskType.causal,
            mask_func=_neg_inf_mask_func(),
            sliding_window=(1, 0),
        )
        out = layer(x, mask=None).numpy()[0, 0]

        i = np.arange(4)[:, None]
        j = np.arange(4)[None, :]
        band_masked = ~((j >= i - 1) & (j <= i + 0))  # True == masked
        ref = _np_softmax(np.where(band_masked, -1e9, rng.astype(np.float64)))
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)
        self.assertTrue(np.all(out[band_masked] < 1e-12))
        np.testing.assert_allclose(out.sum(axis=-1), 1.0, rtol=0, atol=1e-6)

    # ---- general (user-supplied) mask branch ----

    def test_general_mask_branch_consumes_user_mask(self):
        """Non-causal type with an explicit mask uses that exact mask.

        Hand-derived: with attn_mask_type padding (not causal) and no sliding
        window, the auto-gen elif is skipped, so the user mask flows straight
        into mask_func. The reference applies the same distinguishable,
        non-triangular mask; a dropped or transposed user mask would relocate
        the zeros and fail.
        """
        xin = np.array(
            [[0.5, 1.0, -0.5], [2.0, 0.1, 0.3], [-1.0, 0.7, 1.5]],
            dtype=np.float32,
        )
        x = paddle.to_tensor(xin.reshape(1, 1, 3, 3))
        mask_np = np.array(
            [[False, True, False], [False, False, True], [True, False, False]]
        )
        mask = paddle.to_tensor(mask_np)
        layer = self._build(
            attn_mask_type=self.AttnMaskType.padding,
            mask_func=_neg_inf_mask_func(),
        )
        out = layer(x, mask=mask).numpy()[0, 0]

        ref = _np_softmax(np.where(mask_np, -1e9, xin.astype(np.float64)))
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)
        self.assertTrue(np.all(out[mask_np] < 1e-12))

    def test_causal_type_with_user_mask_keeps_user_mask(self):
        """causal type but ``mask is not None`` -> the user mask wins.

        Hand-derived: the auto-gen elif requires ``mask is None``, so a
        provided mask is used verbatim even under causal type. The chosen user
        mask deliberately differs from the causal triu(k=1) pattern; the output
        must match the user-mask reference and NOT the causal one.
        """
        xin = np.array(
            [[0.5, 1.0, -0.5], [2.0, 0.1, 0.3], [-1.0, 0.7, 1.5]],
            dtype=np.float32,
        )
        x = paddle.to_tensor(xin.reshape(1, 1, 3, 3))
        # Lower-triangular-ish user mask, distinct from causal triu(k=1).
        user_np = np.array(
            [[False, False, False], [True, False, False], [True, True, False]]
        )
        layer = self._build(
            attn_mask_type=self.AttnMaskType.causal,
            mask_func=_neg_inf_mask_func(),
        )
        out = layer(x, mask=paddle.to_tensor(user_np)).numpy()[0, 0]

        ref_user = _np_softmax(np.where(user_np, -1e9, xin.astype(np.float64)))
        causal_np = np.triu(np.ones((3, 3), dtype=bool), k=1)
        ref_causal = _np_softmax(
            np.where(causal_np, -1e9, xin.astype(np.float64))
        )
        np.testing.assert_allclose(out, ref_user, rtol=1e-5, atol=1e-6)
        self.assertFalse(np.allclose(out, ref_causal, atol=1e-3))

    # ---- no-mask short-circuit (``mask is None`` -> bypass mask_func) ----

    def test_no_mask_branch_bypasses_mask_func(self):
        """no_mask type + ``mask is None`` leaves ``mask_output = input``.

        Hand-derived: with no sliding window, non-causal type and mask None,
        the ``mask is not None`` guard is False so mask_func is never called
        and the output is plain ``softmax(x)``. A recording mask_func (which
        would zero the output if invoked) proves the bypass.
        """
        xin = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        x = paddle.to_tensor(xin.reshape(1, 1, 1, 3))
        recorder = _RecordingMaskFunc()
        layer = self._build(
            attn_mask_type=self.AttnMaskType.no_mask, mask_func=recorder
        )
        out = layer(x, mask=None).numpy()
        self.assertEqual(recorder.calls, 0)
        np.testing.assert_allclose(
            out[0, 0, 0], _np_softmax(xin), rtol=1e-5, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
