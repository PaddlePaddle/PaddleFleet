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

"""Behavior tests for paddlefleet.transformers.moe_layer_auto.

Scope / environment
-------------------
This file exercises the two CPU-runnable, non-distributed helpers of the MoE
dispatcher:

  * ``dispatching`` -- scatters tokens into ``[num_experts * capacity, dim]``
    expert-capacity slots, applying the per-slot dispatch mask (0 means the
    token was dropped / is padding for that slot) and summing collisions.
  * ``combining``   -- gathers each token's expert-slot outputs back and
    forms the probability-weighted sum, re-positioning results per token.

These are pure tensor ops (scatter / gather / matmul), so they run under a
no-card CPU environment. Expected values are hand-derived with fixed,
all-distinct inputs so that a wrong slot, a dropped-token leak, a swapped
weight, or a mis-positioned gather changes the numbers -- not just the shape.
The production functions are never called to build their own expected values.

The real cross-rank dispatch/combine (``MoELayer.forward`` and the
``LocalGate*`` / ``LocalCombine`` LocalLayers) requires an auto-parallel mesh
and a real AllToAll/reshard across multiple experts on multiple ranks. That
is a 多卡 (multi-card) concern and cannot be validated on a single CPU without
faking the collective; it is marked skipped below with the reason, per the
distributed-training rules (single-card must NOT fake multi-card).
"""

import unittest

import numpy as np
import paddle

from paddlefleet.transformers.moe_layer_auto import combining, dispatching


def _t(values, dtype="float32"):
    """Fixed, distinct-content tensor from a Python nested list."""
    return paddle.to_tensor(values, dtype=dtype)


class TestDispatching(unittest.TestCase):
    """dispatching must place each token at its scatter slot, apply the
    dispatch mask, keep unassigned slots as padding, and accumulate."""

    def test_places_tokens_and_leaves_padding_slots_zero(self):
        # 2 experts * capacity 2 = 4 slots; 3 distinct tokens, top_k = 1.
        # token0 -> slot0, token1 -> slot2, token2 -> slot3; slot1 unused.
        x = _t([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        num_experts, capacity = 2, 2
        dispatch_mask = [paddle.ones([3, 1], dtype="float32")]
        scatter_index = [paddle.to_tensor([0, 2, 3], dtype="int64")]

        out = dispatching(
            x, dispatch_mask, scatter_index, num_experts, capacity
        )

        expected = np.array(
            [
                [1.0, 2.0],  # slot0 <- token0
                [0.0, 0.0],  # slot1 padding (no token routed here)
                [3.0, 4.0],  # slot2 <- token1
                [5.0, 6.0],  # slot3 <- token2
            ],
            dtype=np.float32,
        )
        self.assertEqual(list(out.shape), [num_experts * capacity, 2])
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_accumulates_topk_and_applies_dispatch_mask(self):
        # top_k = 2. Slot collisions must sum; a zero mask entry means the
        # token is dropped for that slot and must contribute nothing.
        x = _t([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        num_experts, capacity = 2, 2
        # slot k=0: token0->0, token1->1, token2->2   (all kept)
        # slot k=1: token0->1, token1->1 (dropped), token2->3
        dispatch_mask = [
            _t([[1.0], [1.0], [1.0]]),
            _t([[1.0], [0.0], [1.0]]),
        ]
        scatter_index = [
            paddle.to_tensor([0, 1, 2], dtype="int64"),
            paddle.to_tensor([1, 1, 3], dtype="int64"),
        ]

        out = dispatching(
            x, dispatch_mask, scatter_index, num_experts, capacity
        )

        # k=0 buffer: [x0, x1, x2, 0]
        # k=1 buffer: slot1 += x0*1 + x1*0 = x0 ; slot3 += x2*1 = x2
        # total: slot0=x0, slot1=x1+x0, slot2=x2, slot3=x2
        expected = np.array(
            [
                [1.0, 2.0],  # x0
                [4.0, 6.0],  # x1 + x0 = [3,4]+[1,2]
                [5.0, 6.0],  # x2
                [5.0, 6.0],  # x2 (dropped token1 added nothing)
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_tensor_scatter_index_matches_independent_expected(self):
        # The Tensor branch (scatter_index.unbind(1)) must produce the same
        # placement as the equivalent list of column vectors.
        x = _t([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        num_experts, capacity = 2, 2
        dispatch_mask = [
            paddle.ones([3, 1], dtype="float32"),
            paddle.ones([3, 1], dtype="float32"),
        ]
        scatter_index = paddle.to_tensor(
            [[0, 1], [2, 1], [3, 3]], dtype="int64"
        )  # columns == the two top-k slots

        out = dispatching(
            x, dispatch_mask, scatter_index, num_experts, capacity
        )

        # col0: t0->0, t1->2, t2->3 ; col1: t0->1, t1->1, t2->3
        # slot0=t0, slot1=t0+t1, slot2=t1, slot3=t2+t2
        expected = np.array(
            [
                [1.0, 2.0],  # t0
                [4.0, 6.0],  # t0 + t1 = [1,2]+[3,4]
                [3.0, 4.0],  # t1
                [10.0, 12.0],  # t2 + t2 = [5,6]*2
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_preserves_input_dtype(self):
        x = _t([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        dispatch_mask = [paddle.ones([2, 1], dtype="float32")]
        scatter_index = [paddle.to_tensor([0, 1], dtype="int64")]
        out = dispatching(x, dispatch_mask, scatter_index, 1, 2)
        self.assertEqual(out.dtype, paddle.float32)


class TestCombining(unittest.TestCase):
    """combining must gather each token's expert-slot rows from the right
    positions and form the probability-weighted sum per token."""

    def test_weights_and_gather_positions(self):
        # 4 expert-capacity slots, dim=2, distinct per slot.
        x = _t(
            [
                [10.0, 20.0],  # slot0
                [30.0, 40.0],  # slot1
                [50.0, 60.0],  # slot2
                [70.0, 80.0],  # slot3
            ]
        )
        # seq=2, top_k=2.
        # token0 gathers slots 0 and 1 with weights 0.5 and 1.0
        # token1 gathers slots 3 and 2 with weights 2.0 and 3.0
        scatter_index = [
            paddle.to_tensor([0, 3], dtype="int64"),  # k=0
            paddle.to_tensor([1, 2], dtype="int64"),  # k=1
        ]
        combine_weights = [
            _t([[0.5], [2.0]]),  # k=0
            _t([[1.0], [3.0]]),  # k=1
        ]

        out = combining(x, combine_weights, scatter_index)

        # token0: 0.5*[10,20] + 1.0*[30,40] = [5,10]+[30,40] = [35,50]
        # token1: 2.0*[70,80] + 3.0*[50,60] = [140,160]+[150,180] = [290,340]
        expected = np.array([[35.0, 50.0], [290.0, 340.0]], dtype=np.float32)
        self.assertEqual(list(out.shape), [2, 2])
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_tensor_scatter_index_matches_list_form(self):
        # The Tensor scatter_index branch must gather identically to the
        # list-of-columns form; verified against the same hand-derived value.
        x = _t(
            [
                [10.0, 20.0],
                [30.0, 40.0],
                [50.0, 60.0],
                [70.0, 80.0],
            ]
        )
        scatter_index = paddle.to_tensor(
            [[0, 1], [3, 2]], dtype="int64"
        )  # row s = [slot for k0, slot for k1]
        combine_weights = [
            _t([[0.5], [2.0]]),
            _t([[1.0], [3.0]]),
        ]

        out = combining(x, combine_weights, scatter_index)

        expected = np.array([[35.0, 50.0], [290.0, 340.0]], dtype=np.float32)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_weight_token_pairing_is_not_symmetric(self):
        # Guard against a weight/token mis-pairing: swapping the two tokens'
        # weights must change the result (distinct weights + distinct slots).
        x = _t(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 0.0],
                [0.0, 2.0],
            ]
        )
        scatter_index = [
            paddle.to_tensor([0, 2], dtype="int64"),
            paddle.to_tensor([1, 3], dtype="int64"),
        ]
        weights = [_t([[1.0], [10.0]]), _t([[100.0], [1000.0]])]
        swapped = [_t([[10.0], [1.0]]), _t([[1000.0], [100.0]])]

        out = combining(x, weights, scatter_index)
        out_swapped = combining(x, swapped, scatter_index)

        # token0: 1*[1,0] + 100*[0,1] = [1,100]
        # token1: 10*[2,0] + 1000*[0,2] = [20,2000]
        expected = np.array([[1.0, 100.0], [20.0, 2000.0]], dtype=np.float32)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)
        # The swapped weighting must not coincide with the correct one.
        self.assertFalse(
            np.allclose(out.numpy(), out_swapped.numpy()),
            "weight/token pairing appears symmetric; mis-pairing undetected",
        )


class TestMoELayerForwardCrossRank(unittest.TestCase):
    """MoELayer.forward and the LocalGate*/LocalCombine LocalLayers perform
    real cross-rank AllToAll / reshard over an auto-parallel expert mesh.

    Validating token/expert identity, padding exclusion, output re-position
    and probability application *across ranks* requires a real multi-expert,
    multi-rank process group (多卡). It cannot be exercised on a single CPU
    without faking the collective + world_size, which the distributed-training
    rules explicitly forbid. Marked skipped rather than faked.
    """

    @unittest.skip(
        "MoELayer.forward requires a real multi-card auto-parallel expert "
        "mesh (AllToAll/reshard across ranks); cannot be validated on a "
        "single CPU without faking collectives. Run under a 多卡 job."
    )
    def test_moe_layer_forward_cross_rank(self):
        raise AssertionError("must run on a real multi-card process group")


if __name__ == "__main__":
    unittest.main()
