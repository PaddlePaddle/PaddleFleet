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

import unittest

import numpy as np

try:
    import paddle

    from paddlefleet.transformer.moe.moe_utils import (
        AddAuxiliaryLoss,
        RandomSTE,
        _AllToAll,
        apply_random_logits,
        detach_and_requires_grad_,
        is_tensor,
        permute,
        unpermute,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # CPU env without paddle
    paddle = None
    _IMPORT_ERROR = exc

_HAS_PADDLE = paddle is not None
_SKIP_REASON = (
    "paddle / paddlefleet.transformer.moe.moe_utils is not importable in this "
    f"environment ({_IMPORT_ERROR!r}); moe_utils exercises real paddle tensor "
    "ops and PyLayers, so the numeric contracts below cannot run here."
)


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestPermute(unittest.TestCase):
    """permute() groups tokens by expert (expert-major, token-ascending)."""

    def test_single_expert_per_token_ordering_and_identity(self):
        # token0->e0, token1->e1, token2->e0, token3->e2
        tokens = paddle.to_tensor(
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]], dtype="float32"
        )
        routing_map = paddle.to_tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype="float32",
        )
        permuted, sorted_indices = permute(tokens, routing_map)
        # expert0 collects tokens 0,2 (ascending); expert1: token1; expert2: token3
        self.assertEqual(sorted_indices.tolist(), [0, 2, 1, 3])
        np.testing.assert_array_equal(
            permuted.numpy(),
            np.array([[0.0, 1.0], [4.0, 5.0], [2.0, 3.0], [6.0, 7.0]]),
        )

    def test_multi_expert_per_token_duplicates_rows(self):
        # token0->e0,e1 ; token1->e1,e2 ; token2->e0,e2
        tokens = paddle.to_tensor(
            [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]], dtype="float32"
        )
        routing_map = paddle.to_tensor(
            [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]], dtype="float32"
        )
        permuted, sorted_indices = permute(tokens, routing_map)
        # e0:tokens0,2 ; e1:tokens0,1 ; e2:tokens1,2
        self.assertEqual(sorted_indices.tolist(), [0, 2, 0, 1, 1, 2])
        expected = np.array(
            [[10, 11], [30, 31], [10, 11], [20, 21], [20, 21], [30, 31]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(permuted.numpy(), expected)

    def test_drop_and_pad_is_rejected(self):
        tokens = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        routing_map = paddle.to_tensor(
            [[1.0, 0.0], [0.0, 1.0]], dtype="float32"
        )
        with self.assertRaises(AssertionError):
            permute(tokens, routing_map, drop_and_pad=True)


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestUnpermute(unittest.TestCase):
    """unpermute() scatters permuted rows back by sorted_indices (scatter-add)."""

    def test_scatter_places_rows_by_index(self):
        permuted = paddle.to_tensor(
            [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]],
            dtype="float32",
        )
        sorted_indices = paddle.to_tensor([2, 0, 3, 1], dtype="int64")
        out = unpermute(permuted, sorted_indices, [4, 2])
        # out[2]=row0, out[0]=row1, out[3]=row2, out[1]=row3
        expected = np.array(
            [[20, 21], [40, 41], [10, 11], [30, 31]], dtype=np.float32
        )
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_duplicate_indices_accumulate(self):
        # scatter_(overwrite=False) into a zero buffer -> sum per destination row
        permuted = paddle.to_tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype="float32"
        )
        sorted_indices = paddle.to_tensor([0, 0, 1], dtype="int64")
        out = unpermute(permuted, sorted_indices, [2, 2])
        # out[0] = [1,2]+[3,4] = [4,6] ; out[1] = [5,6]
        np.testing.assert_array_equal(
            out.numpy(), np.array([[4, 6], [5, 6]], dtype=np.float32)
        )

    def test_probs_are_permuted_expert_major_and_scaled(self):
        permuted = paddle.to_tensor(
            [[2.0, 2.0], [4.0, 4.0], [6.0, 6.0], [8.0, 8.0]], dtype="float32"
        )
        sorted_indices = paddle.to_tensor([0, 1, 2, 3], dtype="int64")
        # 9.0 entries sit on non-routed positions and must be ignored.
        probs = paddle.to_tensor(
            [[0.5, 9.0], [0.25, 9.0], [9.0, 2.0], [9.0, 4.0]], dtype="float32"
        )
        routing_map = paddle.to_tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype="float32"
        )
        out = unpermute(
            permuted,
            sorted_indices,
            [4, 2],
            probs=probs,
            routing_map=routing_map,
        )
        # permuted_probs (expert-major) = [0.5, 0.25, 2.0, 4.0]
        # scaled rows = [1,1],[1,1],[12,12],[32,32], placed identically.
        expected = np.array(
            [[1, 1], [1, 1], [12, 12], [32, 32]], dtype=np.float32
        )
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_probs_without_routing_map_is_rejected(self):
        permuted = paddle.to_tensor([[1.0, 1.0], [2.0, 2.0]], dtype="float32")
        sorted_indices = paddle.to_tensor([0, 1], dtype="int64")
        probs = paddle.to_tensor([[0.5], [0.5]], dtype="float32")
        with self.assertRaises(AssertionError):
            unpermute(permuted, sorted_indices, [2, 1], probs=probs)

    def test_drop_and_pad_is_rejected(self):
        permuted = paddle.to_tensor([[1.0, 1.0], [2.0, 2.0]], dtype="float32")
        sorted_indices = paddle.to_tensor([0, 1], dtype="int64")
        with self.assertRaises(AssertionError):
            unpermute(permuted, sorted_indices, [2, 2], drop_and_pad=True)


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestAddAuxiliaryLoss(unittest.TestCase):
    """AddAuxiliaryLoss forwards a clone and injects a unit aux-loss gradient."""

    def test_forward_clone_and_backward_injects_unit_grad(self):
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype="float32"
        )
        x.stop_gradient = False
        loss = paddle.to_tensor([2.5], dtype="float32")
        loss.stop_gradient = False

        out = AddAuxiliaryLoss.apply(x, loss)
        # forward returns x.clone(): same values, different object.
        np.testing.assert_array_equal(out.numpy(), x.numpy())
        self.assertIsNot(out, x)

        upstream = paddle.to_tensor(
            [[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]],
            dtype="float32",
        )
        paddle.autograd.backward([out], [upstream])
        # dx is the identity passthrough of the upstream gradient ...
        np.testing.assert_array_equal(x.grad.numpy(), upstream.numpy())
        # ... and the aux loss receives a constant gradient of 1 (the "trick").
        np.testing.assert_array_equal(
            loss.grad.numpy(), np.array([1.0], dtype=np.float32)
        )

    def test_non_scalar_loss_is_rejected(self):
        x = paddle.to_tensor([[1.0, 2.0]], dtype="float32")
        loss = paddle.to_tensor([1.0, 2.0], dtype="float32")
        with self.assertRaises(AssertionError):
            AddAuxiliaryLoss.apply(x, loss)


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestRandomSTE(unittest.TestCase):
    """RandomSTE is a straight-through estimator: random forward, zero backward."""

    def test_backward_is_zero_regardless_of_upstream(self):
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype="float32"
        )
        x.stop_gradient = False
        out = RandomSTE.apply(x)
        # forward output is random by design; only its shape/dtype are contracts.
        self.assertEqual(out.shape, [2, 4])
        self.assertEqual(out.dtype, paddle.float32)

        upstream = paddle.full([2, 4], 9.0, dtype="float32")
        paddle.autograd.backward([out], [upstream])
        # STE: gradient is zeros, NOT the (nonzero) upstream gradient.
        np.testing.assert_array_equal(
            x.grad.numpy(), np.zeros([2, 4], dtype=np.float32)
        )


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestApplyRandomLogits(unittest.TestCase):
    """apply_random_logits routes through RandomSTE (zero-gradient backward)."""

    def test_delegates_to_random_ste(self):
        logits = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        logits.stop_gradient = False
        out = apply_random_logits(logits)
        self.assertEqual(out.shape, [2, 2])
        self.assertEqual(out.dtype, paddle.float32)

        upstream = paddle.full([2, 2], 5.0, dtype="float32")
        paddle.autograd.backward([out], [upstream])
        np.testing.assert_array_equal(
            logits.grad.numpy(), np.zeros([2, 2], dtype=np.float32)
        )


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestIsTensor(unittest.TestCase):
    """is_tensor recognises paddle tensors and rejects everything else."""

    def test_paddle_tensor_is_true(self):
        self.assertTrue(
            is_tensor(paddle.to_tensor([1.0, 2.0], dtype="float32"))
        )

    def test_non_tensors_are_false(self):
        self.assertFalse(is_tensor(42))
        self.assertFalse(is_tensor([1, 2, 3]))
        self.assertFalse(is_tensor("hello"))
        self.assertFalse(is_tensor(np.array([1, 2, 3])))
        self.assertFalse(is_tensor(None))


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestDetachAndRequiresGrad(unittest.TestCase):
    """detach_and_requires_grad_ detaches tensors, preserving values/flags."""

    def test_detach_preserves_value_and_stop_gradient_false(self):
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32"
        )
        x.stop_gradient = False
        (result,) = detach_and_requires_grad_(x)
        self.assertIsNot(result, x)  # a fresh detached tensor
        np.testing.assert_array_equal(result.numpy(), x.numpy())
        self.assertFalse(result.stop_gradient)

    def test_detach_preserves_stop_gradient_true(self):
        x = paddle.to_tensor([[7.0, 8.0]], dtype="float32")
        x.stop_gradient = True
        (result,) = detach_and_requires_grad_(x)
        np.testing.assert_array_equal(result.numpy(), x.numpy())
        self.assertTrue(result.stop_gradient)

    def test_non_tensors_pass_through_by_identity(self):
        x = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype="float32")
        x.stop_gradient = False
        sentinel = ["untouched"]
        result = detach_and_requires_grad_(x, "string", 42, sentinel)
        self.assertEqual(len(result), 4)
        np.testing.assert_array_equal(result[0].numpy(), x.numpy())
        self.assertFalse(result[0].stop_gradient)
        self.assertEqual(result[1], "string")
        self.assertEqual(result[2], 42)
        self.assertIs(result[3], sentinel)  # same object, not copied


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestAllToAllWorldSizeOne(unittest.TestCase):
    """_AllToAll world-size==1 local passthrough.

    Only the single-process fallback is exercised here; the real all-to-all
    branch requires a genuine EP process group (multi-card) and is out of scope
    for this CPU/single-process test.
    """

    def test_returns_input_unchanged_when_world_size_one(self):
        inp = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32"
        )
        out = _AllToAll.apply(inp.shape, inp)
        np.testing.assert_array_equal(out.numpy(), inp.numpy())


if __name__ == "__main__":
    unittest.main()
