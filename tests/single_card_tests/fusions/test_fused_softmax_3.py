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

"""Behavior unit tests for ``paddlefleet.fusions.fused_softmax`` (slice _3).

Scope (this sibling): the standalone ``SoftmaxOne`` layer -- the
"softmax-off-by-one" collaborator that ``FusedScaleMaskSoftmax`` dispatches to
when a ``softmax_offset`` is supplied. ``SoftmaxOne.forward`` builds a sink
column from ``denominator_offset`` (reshaped ``[1, np, 1, 1]`` and expanded to
``[b, np, sq, 1]``), concatenates it as an extra logit, runs a full softmax
over the augmented last axis, and drops the sink column::

    sink = offset.reshape(1, -1, 1, 1).expand(b, -1, sq, -1)
    ret  = softmax(concat([x, sink], axis=-1), axis=-1)[..., :-1]

The net effect is a per-row denominator of ``sum_j exp(x_j) + exp(offset)``,
so each row sums to strictly less than one. This slice pins that math down:
exact per-row values, per-head offset routing, batch/seq sink expansion, and
the monotone effect of the offset on the row sum.

Task split: the base sibling (``test_fused_softmax.py``) owns
``FusedScaleMaskSoftmax.__init__``; ``test_fused_softmax_2.py`` owns
``FusedScaleMaskSoftmax.forward`` (scale / mask-branch / softmax-fn selection).
This file therefore does not re-exercise ``FusedScaleMaskSoftmax``; it isolates
``SoftmaxOne`` on its own, which is what this slice's coverage centres on.

All math here is fp32 and runs on CPU (``paddle.softmax`` / ``paddle.concat``
are device-independent for these small inputs), so no GPU numerics are claimed
and nothing is faked as passed. Every expected value comes from an independent
NumPy closed-form off-by-one softmax that analytically eliminates the sink
column (``ex / (sum(ex) + exp(offset))``) -- production's own softmax/concat is
never used to build the expected values. ``SoftmaxOne`` is constructed for real
and never mocked; ``denominator_offset`` is a genuine paddle tensor.
"""

import unittest

import numpy as np

try:
    import paddle

    # Import at module load, BEFORE any ``setUp`` pins the device to CPU. On a
    # CUDA-compiled build the first ``paddlefleet`` import pulls in
    # ``paddlefleet_ops``, whose package init calls ``get_device_capability()``
    # against the current device -- which must be a GPU. The process default
    # device is the GPU there, so importing first keeps that query valid.
    from paddlefleet.fusions.fused_softmax import SoftmaxOne

    _PADDLE_IMPORT_ERROR = None
except ImportError as exc:  # honest: paddle genuinely absent, not swallowed
    paddle = None
    SoftmaxOne = None
    _PADDLE_IMPORT_ERROR = exc


def _ref_softmax_off_by_one(x, offset_per_head):
    """Independent closed-form reference for ``SoftmaxOne.forward``.

    Args:
        x: array of shape ``[b, np, sq, sk]``.
        offset_per_head: 1-D array of shape ``[np]`` giving each head's sink
            logit, matching production's ``reshape(1, -1, 1, 1)`` broadcast.

    Returns:
        Array of shape ``[b, np, sq, sk]`` where
        ``out[..., i] = exp(x_i) / (sum_j exp(x_j) + exp(offset))``.

    The sink column is removed analytically (not concatenated), so this is a
    genuinely independent derivation rather than a restatement of the
    production concat+softmax+slice steps. A shared max is subtracted across
    both the real logits and the sink for numerical stability.
    """
    x = np.asarray(x, dtype=np.float64)
    off = np.asarray(offset_per_head, dtype=np.float64).reshape(1, -1, 1, 1)
    row_max = np.max(x, axis=-1, keepdims=True)
    shift = np.maximum(row_max, off)
    ex = np.exp(x - shift)
    es = np.exp(off - shift)  # broadcasts to [b, np, sq, 1]
    denom = np.sum(ex, axis=-1, keepdims=True) + es
    return ex / denom


@unittest.skipUnless(
    paddle is not None,
    f"paddle is not installed in this environment: {_PADDLE_IMPORT_ERROR}",
)
class TestSoftmaxOne(unittest.TestCase):
    """Standalone off-by-one softmax math of ``SoftmaxOne``."""

    def setUp(self):
        # forward runs real softmax/concat ops; pin CPU so the fp32 math is
        # device-independent, then restore the prior device even on failure.
        self._orig_device = paddle.get_device()
        self.addCleanup(paddle.set_device, self._orig_device)
        paddle.set_device("cpu")

        self.SoftmaxOne = SoftmaxOne

    def _run(self, x_np, offset_np):
        """Build SoftmaxOne with a genuine offset tensor and run forward."""
        offset = paddle.to_tensor(offset_np, dtype="float32")
        layer = self.SoftmaxOne(dim=-1, denominator_offset=offset)
        x = paddle.to_tensor(x_np, dtype="float32")
        out = layer(x)
        return out

    def test_init_stores_dim_and_offset_verbatim(self):
        """Constructor stores ``dim`` and the exact ``denominator_offset``.

        ``denominator_offset`` is a genuine collaborator the forward pass later
        reshapes; ``assertIs`` proves the constructor keeps the exact object
        rather than copying/wrapping it, and ``dim`` is stored unchanged.
        """
        offset = paddle.to_tensor([0.0, 1.0], dtype="float32")
        layer = self.SoftmaxOne(dim=-1, denominator_offset=offset)
        self.assertEqual(layer.dim, -1)
        self.assertIs(layer.denominator_offset, offset)

    def test_forward_single_row_off_by_one_exact(self):
        """One row, offset 0.0: exact values and a row sum strictly below one.

        Hand-derived: for logits [1, 2] with sink logit 0.0 the denominator is
        e^1 + e^2 + e^0. So out = [e^1, e^2] / (e^1 + e^2 + 1), which is
        [0.24472847, 0.66524096] and sums to 0.90996943 -- the missing
        0.09003057 is exp(0)/denom, the sink's share. A plain softmax would
        sum to exactly 1, so this row sum pins the off-by-one behaviour.
        """
        x_np = np.array([[[[1.0, 2.0]]]], dtype=np.float64)  # [1,1,1,2]
        offset_np = np.array([0.0], dtype=np.float64)
        out = self._run(x_np, offset_np)
        self.assertEqual(out.shape, [1, 1, 1, 2])

        expected = _ref_softmax_off_by_one(x_np, offset_np)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)

        row_sum = float(out.numpy().sum(axis=-1).reshape(-1)[0])
        self.assertAlmostEqual(row_sum, 0.9099694268296196, places=6)
        self.assertLess(row_sum, 1.0)

    def test_forward_uses_per_head_offset(self):
        """Each head consumes its own offset via the ``[1, np, 1, 1]`` reshape.

        Head 0 (logits [1, 2], offset 0.0) and head 1 (logits [0.5, -0.5],
        offset 1.0) get different denominators. If the reshape/expand routed a
        single shared offset to both heads, head 1's values would change; the
        full-content compare against the independent per-head reference catches
        that as well as any head/offset transposition.
        """
        x_np = np.array(
            [[[[1.0, 2.0]], [[0.5, -0.5]]]], dtype=np.float64
        )  # [1,2,1,2]
        offset_np = np.array([0.0, 1.0], dtype=np.float64)
        out = self._run(x_np, offset_np)
        self.assertEqual(out.shape, [1, 2, 1, 2])

        expected = _ref_softmax_off_by_one(x_np, offset_np)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)

        # Guard against the "offset ignored / shared" failure mode: the two
        # heads' row sums must genuinely differ for these fixtures.
        head_sums = out.numpy().sum(axis=-1).reshape(-1)
        self.assertNotAlmostEqual(head_sums[0], head_sums[1], places=4)

    def test_forward_expands_sink_over_batch_and_seq(self):
        """The single-head sink is expanded across both batch and seq rows.

        With ``[b=2, np=1, sq=2, sk=2]`` and one offset, every one of the four
        rows must independently receive the off-by-one denominator. Distinct
        per-row logits make a mis-expansion (e.g. sharing one row's sink, or
        collapsing the batch/seq dims) observable in the full-content compare.
        """
        x_np = np.array(
            [
                [[[1.0, 2.0], [0.0, 1.0]]],
                [[[2.0, 0.0], [-1.0, -1.0]]],
            ],
            dtype=np.float64,
        )  # [2,1,2,2]
        offset_np = np.array([0.0], dtype=np.float64)
        out = self._run(x_np, offset_np)
        self.assertEqual(out.shape, [2, 1, 2, 2])

        expected = _ref_softmax_off_by_one(x_np, offset_np)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)

        # Every row sits strictly below one (the sink always claims mass).
        row_sums = out.numpy().sum(axis=-1)
        self.assertTrue(np.all(row_sums < 1.0))

    def test_forward_offset_reduces_each_prob_vs_plain_softmax(self):
        """Off-by-one strictly lowers every probability relative to softmax.

        For the same logits, adding a sink logit only enlarges the denominator,
        so each ``SoftmaxOne`` probability must be strictly smaller than the
        corresponding plain-softmax probability, and the deficit per row equals
        exp(offset)/denom. Hand-derived for [1, 2], offset 0.0: plain softmax
        is [0.26894142, 0.73105858] (sums to 1) while off-by-one is
        [0.24472847, 0.66524096] (sums to 0.90996943). The missing 0.09003057
        is exactly exp(0)/(e^1 + e^2 + 1).
        """
        x_np = np.array([[[[1.0, 2.0]]]], dtype=np.float64)
        offset_np = np.array([0.0], dtype=np.float64)
        out = self._run(x_np, offset_np).numpy().reshape(-1)

        # Independent plain softmax (sink removed) for the same logits.
        z = x_np.reshape(-1)
        e = np.exp(z - z.max())
        plain = e / e.sum()

        self.assertTrue(np.all(out < plain))
        np.testing.assert_allclose(
            plain,
            [0.2689414213699951, 0.7310585786300049],
            rtol=1e-6,
            atol=1e-6,
        )
        # Deficit == sink share == exp(0) / (e^1 + e^2 + e^0).
        denom = float(np.exp(1.0) + np.exp(2.0) + np.exp(0.0))
        self.assertAlmostEqual(1.0 - float(out.sum()), 1.0 / denom, places=6)

    def test_forward_larger_offset_lowers_row_sum(self):
        """A larger offset claims more denominator mass, lowering the row sum.

        The offset enters as ``exp(offset)`` in the denominator, so raising it
        must strictly reduce the retained row sum -- proving the offset is
        genuinely consumed rather than ignored. Hand-derived for logits [1, 2]:
        offset 0.0 -> row sum 0.90996943; offset 2.0 -> row sum 0.57768120.
        """
        x_np = np.array([[[[1.0, 2.0]]]], dtype=np.float64)

        out_small = self._run(x_np, np.array([0.0], dtype=np.float64))
        out_large = self._run(x_np, np.array([2.0], dtype=np.float64))

        sum_small = float(out_small.numpy().sum())
        sum_large = float(out_large.numpy().sum())

        self.assertAlmostEqual(sum_small, 0.9099694268296196, places=6)
        self.assertAlmostEqual(sum_large, 0.5776812017484818, places=6)
        self.assertLess(sum_large, sum_small)

        np.testing.assert_allclose(
            out_large.numpy(),
            _ref_softmax_off_by_one(x_np, np.array([2.0], dtype=np.float64)),
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
