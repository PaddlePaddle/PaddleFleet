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

"""Behavior tests for the CSA score-target query-head padding helpers.

Module under test: ``paddlefleet.fusions.csa_sparse_attn`` -- specifically the
pure head-tile math ``score_target_qheads`` and the tensor padding helper
``pad_score_target_heads``. In the repository module map these belong to the
"计算优化 / Fused Ops" boundary.

Scope and environment. The score-target kernel itself needs a GPU and is not
exercised here; what IS CPU-observable and asserted is (1) the integer tile
width the two device branches compute for a given head count, and (2) the exact
padded ``query``/``lse`` layout that ``pad_score_target_heads`` builds around
that width -- padded head count, the untouched real-head prefix, the zero query
rows, the +inf LSE rows, and the fp32 cast on both the padded and the
pass-through route. The device-capability probe is the only genuine
not-under-test collaborator and is patched so both the SM90 and SM100+ branches
are checked deterministically on any (or no) card; the tile arithmetic and the
concat logic run for real.

Expected values are hand-derived from the two documented rules and never read
back from the function under test:
  * SM100+ (major >= 10): width = max(16, next power of two >= num_heads).
  * SM90  (major  < 10): width = max(2, num_heads) up to 64, else num_heads
    rounded up to the next multiple of 64.

Paddle is required (even CPU-only, since the padding helper builds real
tensors). When it cannot be imported the whole module skips with an honest
reason rather than reporting a pass.
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
except ImportError as exc:  # honest capability probe: only missing dependency
    paddle = None
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    f"paddle / paddlefleet.fusions.csa_sparse_attn unavailable: {_IMPORT_ERROR}"
)


def _as_arch(major):
    """Force the device-capability probe to report ``major``.0.

    ``score_target_qheads`` dispatches on ``get_device_capability`` only to pick
    a branch; the tile math itself is device-independent, so patching the probe
    lets both branches be checked without the matching silicon. This mocks a
    genuine collaborator, not the logic under test.
    """
    return patch.object(
        paddle.device.cuda, "get_device_capability", lambda: (major, 0)
    )


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestScoreTargetQheads(unittest.TestCase):
    def test_sm100_width_is_next_pow2_floored_at_16(self):
        # major >= 10: max(16, next power of two >= h). Derived by hand below;
        # the boundary rows (16/17, 32/33, 64/65, 128/129) pin the ceiling step.
        cases = {
            1: 16,  # below floor -> 16
            8: 16,  # 8 rounds to 8, still below floor -> 16
            15: 16,
            16: 16,  # exactly 16
            17: 32,  # first value above 16 -> next pow2 32
            24: 32,
            31: 32,
            32: 32,  # exactly 32
            33: 64,
            40: 64,
            64: 64,
            65: 128,
            128: 128,
            129: 256,
            192: 256,
        }
        with _as_arch(10):
            got = {h: score_target_qheads(h) for h in cases}
        self.assertEqual(got, cases)

    def test_sm100_never_returns_below_the_16_head_floor(self):
        # Negative-direction check: no head count from 1..16 may drop below 16.
        with _as_arch(10):
            widths = [score_target_qheads(h) for h in range(1, 17)]
        self.assertTrue(all(w == 16 for w in widths), widths)

    def test_sm90_width_is_min2_up_to_64_then_multiples_of_64(self):
        # major < 10: max(2, h) while h <= 64, else h rounded up to a
        # multiple of 64. Hand-derived; 65/96/128 all share the 128 tile,
        # which distinguishes "round up to 64" from "next power of two".
        cases = {
            1: 2,  # floor of 2 (kernel asserts qheads > 1)
            2: 2,
            8: 8,  # served exactly, no padding on SM90
            24: 24,
            40: 40,
            63: 63,
            64: 64,  # last single-tile width
            65: 128,  # rounds up to 2*64
            96: 128,  # not 128-as-pow2: 96 -> 128 by the /64 ceil
            128: 128,
            129: 192,  # 3*64, a power-of-two rule would give 256 here
            192: 192,
        }
        with _as_arch(9):
            got = {h: score_target_qheads(h) for h in cases}
        self.assertEqual(got, cases)

    def test_branches_disagree_so_the_probe_actually_drives_dispatch(self):
        # 24 heads: SM90 keeps 24, SM100 pads to 32. If the capability probe
        # were ignored both would collapse to one answer.
        with _as_arch(9):
            sm90 = score_target_qheads(24)
        with _as_arch(10):
            sm100 = score_target_qheads(24)
        self.assertEqual(sm90, 24)
        self.assertEqual(sm100, 32)
        self.assertNotEqual(sm90, sm100)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestPadScoreTargetHeads(unittest.TestCase):
    def _query_lse(self, heads, b=2, s=3, head_dim=576, lse_dtype="float32"):
        # Distinguishable, deterministic content so a mis-copied or reordered
        # head would change the compared values, not just the shape.
        q_vals = paddle.arange(
            b * s * heads * head_dim, dtype="float32"
        ).reshape([b, s, heads, head_dim])
        query = q_vals.astype("bfloat16")
        lse = (
            paddle.arange(b * s * heads, dtype="float32").reshape([b, s, heads])
            + 1.0
        ).astype(lse_dtype)
        return query, lse

    def test_pads_query_zero_and_lse_posinf_up_to_sm100_tile(self):
        # 24 heads on SM100 -> tile 32, i.e. 8 pad heads appended after the
        # real 24. Real heads must survive bit-for-bit; pad query rows are
        # zero and pad LSE rows are +inf (so exp(0*scale - inf) == 0).
        query, lse = self._query_lse(24)
        with _as_arch(10):
            q_out, lse_out = pad_score_target_heads(query, lse)

        self.assertEqual(list(q_out.shape), [2, 3, 32, 576])
        self.assertEqual(list(lse_out.shape), [2, 3, 32])

        # Real prefix is the original data, untouched.
        self.assertTrue(
            paddle.equal_all(
                q_out[:, :, :24].astype("float32"), query.astype("float32")
            ).item()
        )
        self.assertTrue(
            paddle.equal_all(lse_out[:, :, :24], lse.astype("float32")).item()
        )

        # Pad heads: exactly zero query, strictly positive infinite LSE.
        pad_q = q_out[:, :, 24:].astype("float32")
        pad_lse = lse_out[:, :, 24:]
        self.assertEqual(float(paddle.abs(pad_q).max()), 0.0)
        self.assertTrue(bool(paddle.isinf(pad_lse).all().item()))
        self.assertTrue(bool((pad_lse > 0).all().item()))

    def test_passthrough_returns_query_unchanged_when_already_tiled(self):
        # 64 heads is already a valid SM100 tile -> no padding; query is handed
        # straight back (same object) and only the LSE is fp32-cast.
        query, lse = self._query_lse(64)
        with _as_arch(10):
            q_out, lse_out = pad_score_target_heads(query, lse)
        self.assertIs(q_out, query)
        self.assertEqual(list(lse_out.shape), [2, 3, 64])
        self.assertTrue(paddle.equal_all(lse_out, lse.astype("float32")).item())

    def test_lse_is_cast_to_fp32_on_both_the_padded_and_plain_routes(self):
        # The kernel reads LSE as fp32; a bf16 LSE from the attention forward
        # must be cast whether or not the heads get padded (24 pads, 64 does not).
        for heads in (24, 64):
            query, lse = self._query_lse(heads, lse_dtype="bfloat16")
            self.assertEqual(lse.dtype, paddle.bfloat16)
            with _as_arch(10):
                _, lse_out = pad_score_target_heads(query, lse)
            self.assertEqual(lse_out.dtype, paddle.float32)

    def test_sm90_keeps_the_unpadded_width_and_content(self):
        # 24 heads on SM90 is served in one tile -> no padding at all.
        query, lse = self._query_lse(24)
        with _as_arch(9):
            q_out, lse_out = pad_score_target_heads(query, lse)
        self.assertIs(q_out, query)
        self.assertEqual(list(lse_out.shape), [2, 3, 24])
        self.assertTrue(paddle.equal_all(lse_out, lse.astype("float32")).item())

    def test_sm90_widens_a_single_head_up_to_two(self):
        # SM90's one floor: the kernel asserts qheads > 1, so h == 1 -> 2,
        # a single zero-query / +inf-LSE pad head after the real one.
        query, lse = self._query_lse(1)
        with _as_arch(9):
            q_out, lse_out = pad_score_target_heads(query, lse)
        self.assertEqual(list(q_out.shape), [2, 3, 2, 576])
        self.assertEqual(list(lse_out.shape), [2, 3, 2])
        self.assertTrue(
            paddle.equal_all(
                q_out[:, :, :1].astype("float32"), query.astype("float32")
            ).item()
        )
        self.assertTrue(
            paddle.equal_all(lse_out[:, :, :1], lse.astype("float32")).item()
        )
        self.assertEqual(
            float(paddle.abs(q_out[:, :, 1:].astype("float32")).max()), 0.0
        )
        self.assertTrue(bool(paddle.isinf(lse_out[:, :, 1:]).all().item()))
        self.assertTrue(bool((lse_out[:, :, 1:] > 0).all().item()))


if __name__ == "__main__":
    unittest.main()
