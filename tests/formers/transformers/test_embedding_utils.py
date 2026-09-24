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

"""Behavior tests for paddlefleet.transformers.embedding_utils.

The production module exposes a single function,
``dist_gather_tensor_with_gradient``, used by the Qwen embedding models to
gather query/passage representations across sharding- and data-parallel groups
for in-batch-negative contrastive loss. It is NOT a RoPE / rotary position
utility.

This suite verifies the CPU-observable control logic on a single process:
  * ``None`` input short-circuits to ``None``;
  * ``world_size <= 1`` returns the *same tensor object* untouched, preserving
    both content and gradient flow (the "with_gradient" contract).

The multi-rank gather branches (sharding_group.nranks > 1 /
data_group.nranks > 1) perform real ``paddle.distributed.all_gather`` calls and
their correctness (peer order, concat axis, local-slot substitution for the
gradient-carrying rank) depends on multiple ranks exchanging distinct data.
Per the repo unit-test rules, faking ``world_size`` + mocking the collective
would only prove local orchestration, not cross-rank numerics, so that path is
guarded to run solely under a real >1-rank launcher and is otherwise skipped
with an explicit reason -- never asserted as "passed" on a single card.
"""

import unittest

import numpy as np
import paddle

from paddlefleet.transformers.embedding_utils import (
    dist_gather_tensor_with_gradient,
)

# In a plain, non-distributed process paddle reports world_size == 1, so the
# single-card passthrough branch is exercised for real (no mocking required).
IS_SINGLE_CARD = paddle.distributed.get_world_size() <= 1


class TestDistGatherTensorWithGradient(unittest.TestCase):
    """CPU-only behavior tests for dist_gather_tensor_with_gradient."""

    @classmethod
    def setUpClass(cls):
        # Force CPU so the no-card control logic is exercised without any
        # accelerator; this test makes no claim about device numerics.
        paddle.set_device("cpu")

    def test_none_input_returns_none(self):
        # Contract: a None representation propagates as None (callers rely on
        # this to skip the gather for absent q_reps/p_reps).
        self.assertIsNone(dist_gather_tensor_with_gradient(None))

    def test_single_card_returns_same_object_unchanged(self):
        # Distinguishable content (arange), not randn, so a wrong reshape /
        # transpose / copy would be visible.
        tensor = paddle.arange(6, dtype="float32").reshape([2, 3])
        before = tensor.numpy().copy()

        result = dist_gather_tensor_with_gradient(tensor)

        # On a single card the function must return the *identical* object
        # (no gather, no clone). Identity is the load-bearing contract here:
        # an accidental .contiguous()/.clone() would break gradient aliasing
        # relied on by the caller.
        self.assertIs(result, tensor)
        np.testing.assert_array_equal(result.numpy(), before)
        self.assertEqual(result.shape, [2, 3])

    def test_single_card_preserves_arbitrary_shape(self):
        # A different rank/shape confirms the passthrough does not assume a
        # fixed [b, d] layout on the single-card path.
        tensor = paddle.arange(24, dtype="float32").reshape([2, 3, 4])
        result = dist_gather_tensor_with_gradient(tensor)
        self.assertIs(result, tensor)
        np.testing.assert_array_equal(
            result.numpy(),
            np.arange(24, dtype="float32").reshape([2, 3, 4]),
        )

    def test_single_card_preserves_gradient_flow(self):
        # The "with_gradient" contract: the single-card return must stay on the
        # autograd graph (no detach / stop_gradient flip). Hand-derived
        # expectation: for loss = sum(out**2), d(loss)/dx = 2*x.
        x = paddle.arange(6, dtype="float32").reshape([2, 3])
        x.stop_gradient = False

        out = dist_gather_tensor_with_gradient(x)
        self.assertIs(out, x)
        self.assertFalse(out.stop_gradient)

        loss = (out * out).sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        expected_grad = 2.0 * np.arange(6, dtype="float32").reshape([2, 3])
        np.testing.assert_allclose(
            x.grad.numpy(), expected_grad, rtol=1e-6, atol=0
        )

    @unittest.skipIf(
        IS_SINGLE_CARD,
        "Multi-rank all_gather semantics (peer order, concat axis, "
        "local-slot substitution) require a real >1-rank process group via "
        "paddle.distributed.launch; single-process mock collectives would only "
        "prove orchestration, not cross-rank numerics (see unit-test-rules #13).",
    )
    def test_multi_card_sharding_gather_concatenates_ranks(self):
        # Runs only under a real launcher with world_size > 1 and an
        # initialized hybrid communicate group whose sharding group has
        # nranks > 1. Each rank contributes distinguishable content; the
        # gathered result must be the rank-ordered concatenation along axis 0,
        # independently derived (not via the function under test).
        from paddle.distributed import fleet

        hcg = fleet.get_hybrid_communicate_group()
        sharding_group = hcg.get_sharding_parallel_group()
        if sharding_group.nranks <= 1:
            self.skipTest("sharding group nranks == 1; no cross-rank gather")

        rank = sharding_group.rank
        nranks = sharding_group.nranks
        # Rank-unique, position-unique content so a reversed/duplicated gather
        # would be rejected.
        local = paddle.full([1, 2], fill_value=float(rank), dtype="float32")
        local.stop_gradient = False

        result = dist_gather_tensor_with_gradient(local)

        expected = np.concatenate(
            [np.full((1, 2), float(r), dtype="float32") for r in range(nranks)],
            axis=0,
        )
        np.testing.assert_array_equal(result.numpy(), expected)
        # The local rank's slot must alias the original tensor to keep gradient.
        np.testing.assert_array_equal(result[rank].numpy(), local[0].numpy())


if __name__ == "__main__":
    unittest.main()
