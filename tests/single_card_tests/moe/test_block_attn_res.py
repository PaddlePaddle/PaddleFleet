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
"""Behavior tests for the Block Attention Residual reduction.

Target: ``paddlefleet.transformer.block_attn_res._block_attn_res_rmsnorm``.

This is the FP32 AttentionRes math shared by the memory-saving PyLayer forward
(``BlockAttnResFunc.forward``) and its backward recomputation. Given a list of
completed block representations plus the current partial block, it:

  1. stacks all representations as attention *values*,
  2. RMS-normalizes each (per token, per representation) with ``norm_eps``,
  3. scores each representation by dotting the normalized value with
     ``norm_weight * proj_weight`` (i.e. the RMSNorm gain folded into the
     projection),
  4. softmaxes those scores across representations, and
  5. returns the softmax-weighted sum of the *un-normalized* values, reshaped
     back to the partial block's shape.

The expected values here are derived from that definition in float64 NumPy,
independent of the Paddle implementation. Inputs are small, fixed and mutually
distinguishable (distinct per representation and per token, non-uniform
projection/norm weights) so that a dropped RMS normalization, a wrong reduction
axis, applying softmax to the wrong tensor, or scoring with the wrong
weight/value pairing would change the output and fail the comparison. (The
reduction is intentionally symmetric under permuting the representations, so
ordering is not asserted.)

No accelerator is required: the reduction runs on CPU. Paddle is a hard
dependency of the module under test, so if it (or the package) cannot be
imported the tests skip with an explicit reason rather than reporting a false
pass.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
        ),
        "src",
    ),
)

try:
    import paddle

    from paddlefleet.transformer.block_attn_res import (
        BlockAttnResSublayersSpec,
        _block_attn_res_rmsnorm,
    )
    from paddlefleet.transformer.identity_op import IdentityOp

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet not installed in this env
    paddle = None
    BlockAttnResSublayersSpec = None
    _block_attn_res_rmsnorm = None
    IdentityOp = None
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def _reference_attn_res_rmsnorm(
    partial_block, blocks, proj_weight, norm_weight, eps
):
    """Independent float64 reference for the AttentionRes reduction.

    Mirrors the documented contract, NOT the Paddle source: representations are
    ordered ``blocks`` first then ``partial_block`` last; each is RMS-normalized
    per token; scores use ``norm_weight * proj_weight``; softmax runs across the
    representation axis; the output is a weighted sum of the raw values.
    """
    partial_block = np.asarray(partial_block, dtype=np.float64)
    blocks = [np.asarray(b, dtype=np.float64) for b in blocks]
    proj_weight = np.asarray(proj_weight, dtype=np.float64).reshape(-1)
    norm_weight = np.asarray(norm_weight, dtype=np.float64).reshape(-1)

    hidden = partial_block.shape[-1]
    all_repr = [*blocks, partial_block]  # partial block is last
    # values: [num_tokens, num_repr, hidden]
    values = np.stack([r.reshape(-1, hidden) for r in all_repr], axis=1)

    variance = np.mean(values**2, axis=-1, keepdims=True)
    normalized = values / np.sqrt(variance + eps)

    score_weight = norm_weight * proj_weight  # [hidden]
    scores = np.sum(
        normalized * score_weight, axis=-1
    )  # [num_tokens, num_repr]

    shifted = scores - scores.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=-1, keepdims=True)  # [num_tokens, num_repr]

    # weighted sum of raw (un-normalized) values across representations
    output = np.einsum("tr,trh->th", probs, values)  # [num_tokens, hidden]
    return output.reshape(partial_block.shape), probs


@unittest.skipUnless(
    paddle is not None,
    f"paddle/paddlefleet not importable in this environment: {_IMPORT_ERROR}",
)
class TestBlockAttnResReduction(unittest.TestCase):
    """Numeric behavior of ``_block_attn_res_rmsnorm`` against a hand reference."""

    def setUp(self):
        # No-card test: keep the reduction on CPU and off any visible device.
        paddle.set_device("cpu")

        # Fixed, mutually distinguishable inputs. Two tokens (B=1, S=2), hidden
        # size 4, two completed blocks plus the partial block -> 3 reps.
        self.eps = 1e-5
        self.block0 = np.array(
            [[[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, 1.0]]], dtype=np.float32
        )
        self.block1 = np.array(
            [[[-2.0, 1.0, 0.5, 3.0], [1.0, 1.0, -1.0, 2.0]]], dtype=np.float32
        )
        self.partial = np.array(
            [[[2.0, -1.0, 1.0, 0.0], [3.0, 0.0, -2.0, 1.0]]], dtype=np.float32
        )
        # proj_weight keeps the Linear(hidden, 1) checkpoint layout [1, hidden].
        self.proj_weight = np.array([[0.5, -1.0, 2.0, 0.3]], dtype=np.float32)
        self.norm_weight = np.array([1.0, 0.5, -0.5, 2.0], dtype=np.float32)

    def _call_production(self):
        return _block_attn_res_rmsnorm(
            paddle.to_tensor(self.partial),
            [paddle.to_tensor(self.block0), paddle.to_tensor(self.block1)],
            paddle.to_tensor(self.proj_weight),
            paddle.to_tensor(self.norm_weight),
            self.eps,
        )

    def test_reduction_matches_independent_reference(self):
        expected, probs = _reference_attn_res_rmsnorm(
            self.partial,
            [self.block0, self.block1],
            self.proj_weight,
            self.norm_weight,
            self.eps,
        )
        out = self._call_production().numpy()

        # Shape contract: reshaped back to the partial block's shape.
        self.assertEqual(list(out.shape), list(self.partial.shape))

        # Exact numeric contract against the hand-derived reference.
        np.testing.assert_allclose(
            out.astype(np.float64), expected, rtol=1e-5, atol=1e-6
        )

        # Guard against a degenerate fixture: the softmax must actually
        # discriminate between representations (otherwise the weighted sum would
        # collapse to a plain mean and hide reduction-axis / ordering bugs).
        self.assertGreater(
            float(np.abs(probs - probs.mean(axis=-1, keepdims=True)).max()),
            1e-3,
            "fixture produced near-uniform attention weights; not discriminating",
        )

    def test_output_is_convex_combination_per_token(self):
        # A softmax-weighted sum of the representations must, per token and per
        # channel, lie within the element-wise min/max of those representations.
        # This is an implementation-independent property of any convex
        # combination and would reject a reduction that leaks in outside values
        # or mismatches the weight/value pairing.
        out = (
            self._call_production().numpy().reshape(-1, self.partial.shape[-1])
        )
        hidden = self.partial.shape[-1]
        stacked = np.stack(
            [
                self.block0.reshape(-1, hidden),
                self.block1.reshape(-1, hidden),
                self.partial.reshape(-1, hidden),
            ],
            axis=1,
        ).astype(np.float64)
        lo = stacked.min(axis=1)
        hi = stacked.max(axis=1)
        tol = 1e-5
        self.assertTrue(np.all(out >= lo - tol))
        self.assertTrue(np.all(out <= hi + tol))

    def test_zero_projection_weight_yields_uniform_mean(self):
        # With a zero projection weight every score is 0, so softmax is uniform
        # and the reduction must reduce to the plain mean of the representations.
        # (BlockAttnRes initializes proj_weight to Constant(0.0), so this is the
        # untrained starting behavior.) Verifying this separately anchors the
        # softmax-normalization independently of the weighted general case.
        hidden = self.partial.shape[-1]
        out = _block_attn_res_rmsnorm(
            paddle.to_tensor(self.partial),
            [paddle.to_tensor(self.block0), paddle.to_tensor(self.block1)],
            paddle.to_tensor(np.zeros((1, hidden), dtype=np.float32)),
            paddle.to_tensor(self.norm_weight),
            self.eps,
        ).numpy()
        expected_mean = np.mean(
            np.stack(
                [
                    self.block0.astype(np.float64),
                    self.block1.astype(np.float64),
                    self.partial.astype(np.float64),
                ],
                axis=0,
            ),
            axis=0,
        )
        np.testing.assert_allclose(
            out.astype(np.float64), expected_mean, rtol=1e-5, atol=1e-6
        )


@unittest.skipUnless(
    paddle is not None,
    f"paddle/paddlefleet not importable in this environment: {_IMPORT_ERROR}",
)
class TestBlockAttnResSublayersSpec(unittest.TestCase):
    """Config default contract for the sublayer spec."""

    def test_default_norm_is_identity_op(self):
        # The dataclass default selects a no-op norm; BlockAttnRes relies on this
        # to fall back off the RMSNorm PyLayer path when no norm is configured.
        spec = BlockAttnResSublayersSpec()
        self.assertIs(spec.norm, IdentityOp)


if __name__ == "__main__":
    unittest.main()
