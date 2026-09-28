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

"""Sub-tile head-count boundary of the CSA score-target head padding.

Module under test: ``paddlefleet.fusions.csa_sparse_attn`` -- the pure host-side
tile math ``score_target_qheads`` and the tensor padding helper
``pad_score_target_heads``. In the repository module map these sit at the
"计算优化 / Fused Ops" boundary.

Scope / disjointness. The general per-value width table and the h=24 / h=1
padding layout are already pinned by
``tests/single_card_tests/ops/test_score_target_head_pad.py``. To stay disjoint
this file targets specifically the SUB-TILE region -- head counts strictly below
a kernel tile -- and verifies a different class of contract there:

  * ``score_target_qheads``: over the WHOLE 1..128 sub-tile range, that the
    chosen width is valid and *stable* (a fixed point the kernel will not
    re-pad), never shrinks the layer (``width >= h``), and is monotone
    non-decreasing -- properties a sampled value table cannot establish.
  * ``pad_score_target_heads``: the pad *count* arithmetic (appended heads ==
    width - h) on sub-tile counts the ops test does not use (8/17/40), that
    padding is idempotent at the tensor level, that the padded query keeps its
    input dtype, and that the device branch alone decides whether a sub-tile
    layer is padded at all.

Environment. This is control-logic / layout only, so it runs on any card or no
card: the score-target GPU kernel that consumes the padded tensors is NOT
exercised here (its numerics need FlashMLA / cuDNN on a real device and are out
of scope). The device-capability probe is the sole genuine not-under-test
collaborator; it is patched so both the SM90 and SM100+ branches run
deterministically, while the tile arithmetic and the concat run for real. The
patch is a context manager, so the global probe is restored even on failure.

Every expected value is hand-derived from the two documented rules and never
read back from the function under test:
  * SM100+ (major >= 10): width = max(16, next power of two >= num_heads);
    valid tiles in range are exactly {16, 32, 64, 128}.
  * SM90  (major  < 10): width = num_heads, floored at 2, while num_heads <= 64.

Paddle is required (even CPU-only: the padding helper builds real tensors); if
it cannot be imported the module skips with an honest reason instead of passing.
"""

import unittest
from unittest.mock import patch

try:
    import paddle

    from paddlefleet.fusions.csa_sparse_attn import (
        pad_score_target_heads,
        score_target_qheads,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # honest probe: only a genuinely missing dependency
    paddle = None
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    f"paddle / paddlefleet.fusions.csa_sparse_attn unavailable: {_IMPORT_ERROR}"
)

# Hand-listed valid SM100+ score-target tiles across the 1..128 sub-tile range:
# the powers of two at or above the 16-head floor, up to the widest 128 tile.
_SM100_VALID_TILES = {16, 32, 64, 128}


def _as_arch(major):
    """Force the device-capability probe to report ``major``.0.

    ``score_target_qheads`` calls ``get_device_capability`` only to pick a
    branch; the tile math itself is device-independent, so patching the probe
    exercises both branches without the matching silicon. This mocks a genuine
    collaborator, not the logic under test -- the arithmetic still runs.
    """
    return patch.object(
        paddle.device.cuda, "get_device_capability", lambda: (major, 0)
    )


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestSubTileWidthInvariants(unittest.TestCase):
    """Range invariants of ``score_target_qheads`` over the sub-tile region."""

    def test_sm100_sub_tile_widths_valid_monotone_and_never_shrink(self):
        # Span the entire 1..128 sub-tile region rather than sampling points.
        heads = list(range(1, 129))
        with _as_arch(10):
            widths = [score_target_qheads(h) for h in heads]

        # (1) Every width is one of the hand-listed valid tiles -- the kernel
        # only instantiates these, so a width like 24 or 48 would be a bug.
        self.assertTrue(
            all(w in _SM100_VALID_TILES for w in widths),
            f"off-tile width produced: {sorted(set(widths))}",
        )
        # (2) Padding never drops a real head: width must cover the layer.
        self.assertTrue(
            all(w >= h for h, w in zip(heads, widths)),
            [(h, w) for h, w in zip(heads, widths) if w < h],
        )
        # (3) Monotone non-decreasing: a bigger layer never gets a smaller tile.
        self.assertTrue(
            all(a <= b for a, b in zip(widths, widths[1:])),
            "width is not monotone in head count",
        )
        # (4) The floor is concrete: 1..16 heads all land on the 16 tile.
        self.assertEqual(widths[:16], [16] * 16)

    def test_sm100_pad_targets_are_fixed_points(self):
        # A sub-tile layer padded to one of these widths must not be padded
        # again -- the target is stable. Expected values are literals, so this
        # is not self-referential.
        expected = {16: 16, 32: 32, 64: 64, 128: 128}
        with _as_arch(10):
            got = {w: score_target_qheads(w) for w in expected}
        self.assertEqual(got, expected)

    def test_sm90_sub_tile_served_natively_with_a_two_head_floor(self):
        # major < 10, single-tile region h <= 64: the width is the head count
        # itself, except a lone head is widened to two (the kernel asserts
        # qheads > 1). Hand-derived as "floor of two, otherwise pass-through".
        heads = list(range(1, 65))
        expected = [2 if h == 1 else h for h in heads]
        with _as_arch(9):
            widths = [score_target_qheads(h) for h in heads]
        self.assertEqual(widths, expected)
        # Same structural guarantees hold on this branch.
        self.assertTrue(all(w >= h for h, w in zip(heads, widths)))
        self.assertTrue(all(a <= b for a, b in zip(widths, widths[1:])))

    def test_sm90_sub_tile_widths_are_fixed_points(self):
        # Whatever SM90 pads a lone head up to must itself be stable.
        expected = {1: 2, 2: 2, 24: 24, 64: 64}
        with _as_arch(9):
            first = {h: score_target_qheads(h) for h in expected}
            second = {h: score_target_qheads(first[h]) for h in expected}
        self.assertEqual(first, expected)
        # Re-applying to the width leaves it unchanged (2->2, 24->24, 64->64).
        self.assertEqual(second, {h: expected[h] for h in expected})


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestPadScoreTargetSubTileLayout(unittest.TestCase):
    """Layout / arithmetic of ``pad_score_target_heads`` on sub-tile inputs."""

    def _query_lse(self, heads, b=1, s=2, head_dim=8, lse_dtype="float32"):
        # Per-head-distinct integer signatures, small enough to be exact in
        # bfloat16 (b*s*heads <= 256), so a reordered or mis-copied head changes
        # the compared values -- not just the shape. head_dim only carries the
        # signature, so its size is irrelevant to the padding under test.
        base = paddle.arange(b * s * heads, dtype="float32").reshape(
            [b, s, heads, 1]
        )
        query = paddle.tile(base, [1, 1, 1, head_dim]).astype("bfloat16")
        lse = (
            paddle.arange(b * s * heads, dtype="float32").reshape([b, s, heads])
            + 1.0
        ).astype(lse_dtype)
        return query, lse

    def test_sm100_appends_width_minus_h_zero_query_and_inf_lse(self):
        # Sub-tile counts the ops test does not use, with hand-derived widths
        # and pad counts:  8 -> 16 (pad 8), 17 -> 32 (pad 15), 40 -> 64 (pad 24).
        for heads, width, pad in ((8, 16, 8), (17, 32, 15), (40, 64, 24)):
            with self.subTest(heads=heads):
                query, lse = self._query_lse(heads)
                with _as_arch(10):
                    q_out, lse_out = pad_score_target_heads(query, lse)

                self.assertEqual(int(q_out.shape[2]), width)
                self.assertEqual(int(lse_out.shape[2]), width)
                # Appended head count is exactly width - h.
                self.assertEqual(int(q_out.shape[2]) - heads, pad)

                # Real prefix is bit-identical to the input.
                self.assertTrue(
                    paddle.equal_all(
                        q_out[:, :, :heads].astype("float32"),
                        query.astype("float32"),
                    ).item()
                )
                self.assertTrue(
                    paddle.equal_all(
                        lse_out[:, :, :heads], lse.astype("float32")
                    ).item()
                )
                # Pad heads: exactly-zero query, strictly-positive infinite LSE
                # (so exp(0 * scale - inf) == 0 keeps them out of the head sum).
                self.assertEqual(
                    float(
                        paddle.abs(q_out[:, :, heads:].astype("float32")).max()
                    ),
                    0.0,
                )
                self.assertTrue(
                    bool(paddle.isinf(lse_out[:, :, heads:]).all().item())
                )
                self.assertTrue(bool((lse_out[:, :, heads:] > 0).all().item()))

    def test_sm100_padding_is_idempotent(self):
        # 24 -> 32 once; padding the already-widened tensors is a no-op because
        # 32 is a stable tile. The second call must hand the query straight back.
        query, lse = self._query_lse(24)
        with _as_arch(10):
            q1, lse1 = pad_score_target_heads(query, lse)
            self.assertEqual(int(q1.shape[2]), 32)
            self.assertIsNot(q1, query)  # first call really padded

            q2, lse2 = pad_score_target_heads(q1, lse1)
        self.assertIs(q2, q1)  # no second pad: width already a fixed point
        self.assertEqual(int(q2.shape[2]), 32)
        self.assertTrue(paddle.equal_all(lse2, lse1).item())

    def test_sm100_padded_query_preserves_input_dtype(self):
        # The zero pad rows are built in the query's dtype, so a bf16 layer
        # stays bf16 after widening (a wrong pad dtype would upcast or fail).
        query, lse = self._query_lse(8)
        self.assertEqual(query.dtype, paddle.bfloat16)
        with _as_arch(10):
            q_out, _ = pad_score_target_heads(query, lse)
        self.assertEqual(int(q_out.shape[2]), 16)
        self.assertEqual(q_out.dtype, paddle.bfloat16)

    def test_arch_controls_whether_sub_tile_heads_get_padded(self):
        # Same 24-head sub-tile layer: SM100 widens it to 32 (a fresh tensor),
        # SM90 serves it natively (the query object is returned untouched). The
        # device branch alone flips the tensor-level layout.
        query, lse = self._query_lse(24)
        with _as_arch(10):
            q_sm100, _ = pad_score_target_heads(query, lse)
        with _as_arch(9):
            q_sm90, lse_sm90 = pad_score_target_heads(query, lse)

        self.assertEqual(int(q_sm100.shape[2]), 32)
        self.assertIsNot(q_sm100, query)

        self.assertEqual(int(q_sm90.shape[2]), 24)
        self.assertIs(q_sm90, query)  # passthrough: no padding on SM90
        # Passthrough still casts the LSE to fp32 for the kernel.
        self.assertEqual(lse_sm90.dtype, paddle.float32)


if __name__ == "__main__":
    unittest.main()
