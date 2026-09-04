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

"""Coverage for the sparse score-target path's query-head padding.

``pad_score_target_heads`` widens the query/LSE pair handed to
``sparse_attn_score_recompute`` to a head count that kernel's MMA ``M`` tile can
express, because on SM100+ ``m // 4`` has to be a valid
``tcgen05.copy.Repetition``. The helper is pure Paddle, so it is checked directly
rather than through a mock: what needs guarding is the padded width, the zeros in
the query and the ``+inf`` in the LSE. The numerics of the padded kernel itself
need a device and live with the op tests.

The supported width depends on the device -- the SM90 kernel tiles by 64 heads
and has no ``Repetition`` to satisfy -- so the capability is patched rather than
read, and both branches are covered on whichever card the suite runs on.
"""

import unittest
from unittest.mock import patch

import paddle

from paddlefleet.fusions.csa_sparse_attn import (
    pad_score_target_heads,
    score_target_qheads,
)


def _as_arch(major):
    """Pretend the running device is ``major``.0, so the tables are fixed."""
    return patch.object(
        paddle.device.cuda, "get_device_capability", lambda: (major, 0)
    )


class TestScoreTargetQheads(unittest.TestCase):
    def test_sm100_width_is_a_power_of_two_of_at_least_16(self):
        with _as_arch(10):
            for heads, want in (
                (1, 16),
                (8, 16),
                (16, 16),
                (24, 32),
                (32, 32),
                (40, 64),
                (64, 64),
                (128, 128),
                (192, 256),
            ):
                self.assertEqual(score_target_qheads(heads), want)

    def test_sm90_serves_any_width_up_to_64_and_multiples_above(self):
        # ``_interface_sm90._compute_tile_m`` caps ``tile_m`` at 64 and loops
        # ``qhpkv // tile_m`` head tiles, so padding a 24-head layer there would
        # only buy wasted MMA rows. Its one floor is the ``qhead_per_kvhead > 1``
        # assert at the top of that same helper.
        with _as_arch(9):
            for heads, want in (
                (1, 2),
                (2, 2),
                (8, 8),
                (24, 24),
                (40, 40),
                (64, 64),
                (96, 128),
                (192, 192),
            ):
                self.assertEqual(score_target_qheads(heads), want)


class TestPadScoreTargetHeads(unittest.TestCase):
    def _call(self, heads, b=2, s=4, head_dim=576, major=10):
        query = paddle.randn([b, s, heads, head_dim]).astype("bfloat16")
        lse = paddle.randn([b, s, heads]).astype("float32")
        with _as_arch(major):
            padded = pad_score_target_heads(query, lse)
        return query, lse, *padded

    def test_non_power_of_two_width_is_padded(self):
        query, lse, q_seen, lse_seen = self._call(24)
        self.assertEqual(list(q_seen.shape), [2, 4, 32, 576])
        self.assertEqual(list(lse_seen.shape), [2, 4, 32])
        # Real heads untouched.
        self.assertTrue(
            paddle.equal_all(
                q_seen[:, :, :24].astype("float32"), query.astype("float32")
            )
        )
        self.assertTrue(paddle.equal_all(lse_seen[:, :, :24], lse))
        # Pad heads: zero query, positive-infinite LSE, so they contribute
        # ``exp(0 * scale - inf)`` to the head sum instead of a finite score.
        self.assertEqual(float(paddle.abs(q_seen[:, :, 24:]).max()), 0.0)
        self.assertTrue(bool(paddle.isinf(lse_seen[:, :, 24:]).all()))
        self.assertTrue(bool((lse_seen[:, :, 24:] > 0).all()))

    def test_supported_width_is_passed_through_untouched(self):
        query, lse, q_seen, lse_seen = self._call(64)
        self.assertEqual(list(q_seen.shape), [2, 4, 64, 576])
        self.assertTrue(
            paddle.equal_all(q_seen.astype("float32"), query.astype("float32"))
        )
        self.assertTrue(paddle.equal_all(lse_seen, lse))

    def test_narrow_width_is_padded_up_to_the_floor(self):
        _, _, q_seen, lse_seen = self._call(8)
        self.assertEqual(int(q_seen.shape[2]), 16)
        self.assertEqual(int(lse_seen.shape[2]), 16)

    def test_sm90_keeps_the_unpadded_width(self):
        query, lse, q_seen, lse_seen = self._call(24, major=9)
        self.assertEqual(list(q_seen.shape), [2, 4, 24, 576])
        self.assertTrue(
            paddle.equal_all(q_seen.astype("float32"), query.astype("float32"))
        )
        self.assertTrue(paddle.equal_all(lse_seen, lse))

    def test_sm90_still_pads_a_single_head_up_to_two(self):
        # ``_compute_tile_m`` asserts ``qhead_per_kvhead > 1``, so ``h == 1`` is
        # the one width SM90 has to widen.
        query, lse, q_seen, lse_seen = self._call(1, major=9)
        self.assertEqual(list(q_seen.shape), [2, 4, 2, 576])
        self.assertEqual(list(lse_seen.shape), [2, 4, 2])
        self.assertTrue(
            paddle.equal_all(
                q_seen[:, :, :1].astype("float32"), query.astype("float32")
            )
        )
        self.assertTrue(paddle.equal_all(lse_seen[:, :, :1], lse))
        self.assertEqual(float(paddle.abs(q_seen[:, :, 1:]).max()), 0.0)
        self.assertTrue(bool(paddle.isinf(lse_seen[:, :, 1:]).all()))
        self.assertTrue(bool((lse_seen[:, :, 1:] > 0).all()))

    def test_lse_is_returned_in_fp32_whether_or_not_it_is_padded(self):
        # The kernel reads the LSE as fp32; a bf16 one from the attention
        # forward has to be cast on both the padded and the unpadded route.
        for heads in (24, 64):
            query = paddle.randn([1, 4, heads, 576]).astype("bfloat16")
            lse = paddle.randn([1, 4, heads]).astype("bfloat16")
            with _as_arch(10):
                _, lse_seen = pad_score_target_heads(query, lse)
            self.assertEqual(lse_seen.dtype, paddle.float32)


if __name__ == "__main__":
    unittest.main()
