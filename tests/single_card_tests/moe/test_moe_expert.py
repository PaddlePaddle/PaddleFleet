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

"""Behavior tests for paddlefleet.transformer.moe.moe_expert.

Target: SonicMoEExpert weight-layout conversions that move the fused
grouped-GEMM expert weights between the "grouped" layout consumed by the
DeepGEMM/batched-GEMM path and the "sonic" layout consumed by SonicMoE.

Getting these conversions wrong silently corrupts a trained MoE model: the
gate and up projections of the fused weight1 must stay paired and contiguous
across the transpose, and weight2 must be transposed without reordering
experts. These tests pin the exact element placement with hand-derived
expected tensors (independent of the production reshape/transpose code) using
per-expert distinguishable values so an expert-dim swap, a gate/up
interleaving error, or a wrong transpose axis is rejected.

No GPU numerics or collectives are exercised here; these are pure
tensor-layout contracts that run on CPU. Paddle is imported under a guard so
that environments without paddle/paddlefleet skip honestly instead of
reporting a false pass.
"""

import unittest

import numpy as np

try:
    import paddle

    from paddlefleet.transformer.moe.moe_expert import SonicMoEExpert

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet_ops not installed
    paddle = None
    SonicMoEExpert = None
    _IMPORT_ERROR = exc


@unittest.skipUnless(
    SonicMoEExpert is not None,
    f"paddle/paddlefleet not importable: {_IMPORT_ERROR}",
)
class TestSonicMoEWeightLayout(unittest.TestCase):
    """Exact-placement contracts for the weight1/weight2 layout swaps."""

    def test_grouped_w1_to_sonic_interleaves_gate_and_up(self):
        """weight1 grouped [E,H,2I] -> sonic [E,2I,H].

        grouped weight1 packs gate (first I cols) and up (last I cols) along
        the last dim. The sonic layout must transpose each expert to [.,H] and
        interleave gate/up row-pairs so gate[i] and up[i] are adjacent. Values
        below are laid out so a wrong axis, a dropped transpose, or a
        gate/up mixup changes the result.
        """
        # E=2, H=2, I=3  -> last dim 2I=6
        grouped = np.arange(24, dtype=np.float32).reshape(2, 2, 6)
        # Hand-derived expected [E, 2I, H] = [2, 6, 2]
        expected = np.array(
            [
                [[0, 6], [3, 9], [1, 7], [4, 10], [2, 8], [5, 11]],
                [[12, 18], [15, 21], [13, 19], [16, 22], [14, 20], [17, 23]],
            ],
            dtype=np.float32,
        )
        out = SonicMoEExpert._grouped_w1_to_sonic(paddle.to_tensor(grouped))
        self.assertEqual(list(out.shape), [2, 6, 2])
        np.testing.assert_array_equal(
            np.asarray(out.numpy(), dtype=np.float32), expected
        )

    def test_sonic_w1_to_grouped_restores_gate_up_blocks(self):
        """weight1 sonic [E,2I,H] -> grouped [E,H,2I].

        Independent hand-derived expected: de-interleave the row-pairs back
        into gate/up blocks, transpose each expert to [.,2I], and concatenate
        gate then up along the last dim.
        """
        # E=2, I=3, H=2 -> sonic first dim 2I=6
        sonic = np.arange(24, dtype=np.float32).reshape(2, 6, 2)
        # Hand-derived expected [E, H, 2I] = [2, 2, 6]
        expected = np.array(
            [
                [[0, 4, 8, 2, 6, 10], [1, 5, 9, 3, 7, 11]],
                [[12, 16, 20, 14, 18, 22], [13, 17, 21, 15, 19, 23]],
            ],
            dtype=np.float32,
        )
        out = SonicMoEExpert._sonic_w1_to_grouped(paddle.to_tensor(sonic))
        self.assertEqual(list(out.shape), [2, 2, 6])
        np.testing.assert_array_equal(
            np.asarray(out.numpy(), dtype=np.float32), expected
        )

    def test_w1_grouped_sonic_roundtrip_is_identity(self):
        """sonic_to_grouped must be the exact inverse of grouped_to_sonic.

        The two exact-value tests above anchor each direction independently;
        this guards that the pair composes to identity (no residual gate/up
        shuffle), which is what a training step relies on when it flips layouts
        between forward and checkpoint save.
        """
        grouped = np.arange(24, dtype=np.float32).reshape(2, 2, 6)
        as_sonic = SonicMoEExpert._grouped_w1_to_sonic(
            paddle.to_tensor(grouped)
        )
        back = SonicMoEExpert._sonic_w1_to_grouped(as_sonic)
        self.assertEqual(list(back.shape), [2, 2, 6])
        np.testing.assert_array_equal(
            np.asarray(back.numpy(), dtype=np.float32), grouped
        )

    def test_transpose_w2_layout_swaps_last_two_axes_per_expert(self):
        """weight2 [E,I,H] <-> [E,H,I] transposes only the last two axes.

        Per-expert distinguishable values catch an expert-dim reorder or a
        transpose that leaks across experts.
        """
        # E=2, I=2, H=3
        w2 = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        expected = np.array(
            [
                [[0, 3], [1, 4], [2, 5]],
                [[6, 9], [7, 10], [8, 11]],
            ],
            dtype=np.float32,
        )
        out = SonicMoEExpert._transpose_w2_layout(paddle.to_tensor(w2))
        self.assertEqual(list(out.shape), [2, 3, 2])
        np.testing.assert_array_equal(
            np.asarray(out.numpy(), dtype=np.float32), expected
        )

    def test_transpose_w2_layout_is_its_own_inverse(self):
        """Applying the [0,2,1] transpose twice recovers the original."""
        w2 = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        once = SonicMoEExpert._transpose_w2_layout(paddle.to_tensor(w2))
        twice = SonicMoEExpert._transpose_w2_layout(once)
        self.assertEqual(list(twice.shape), [2, 2, 3])
        np.testing.assert_array_equal(
            np.asarray(twice.numpy(), dtype=np.float32), w2
        )


if __name__ == "__main__":
    unittest.main()
