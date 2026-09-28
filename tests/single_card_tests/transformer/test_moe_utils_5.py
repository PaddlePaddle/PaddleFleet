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

    from paddlefleet.training.global_vars import (
        get_global_training_logs,
        set_global_training_logs,
        unset_global_variables,
    )
    from paddlefleet.transformer.moe.moe_utils import (
        AddAuxiliaryLoss,
        FakeClone,
        detach_and_requires_grad_,
        is_tensor,
        log_moe_losses,
        permute,
        unpermute,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    _IMPORT_ERROR = exc


requires_paddle = unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet not importable on this host: {_IMPORT_ERROR!r}",
)


class _EnabledLogs(dict):
    """Training-logs stand-in whose MoE balance gate is on."""

    def is_moe_balance_logs_enabled(self):
        return True


class _DisabledLogs(dict):
    """Training-logs stand-in whose MoE balance gate is off."""

    def is_moe_balance_logs_enabled(self):
        return False


@requires_paddle
class TestIsTensor(unittest.TestCase):
    def test_true_only_for_paddle_tensors(self):
        self.assertIs(is_tensor(paddle.to_tensor([1.0, 2.0])), True)

    def test_false_for_python_and_numpy_values(self):
        self.assertIs(is_tensor(42), False)
        self.assertIs(is_tensor(3.14), False)
        self.assertIs(is_tensor([1, 2, 3]), False)
        self.assertIs(is_tensor("hello"), False)
        self.assertIs(is_tensor(None), False)
        self.assertIs(is_tensor(np.array([1, 2, 3])), False)


@requires_paddle
class TestDetachAndRequiresGrad(unittest.TestCase):
    def test_returns_list_of_detached_tensors_with_same_values(self):
        src = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        src.stop_gradient = False
        out = detach_and_requires_grad_(src)
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 1)
        self.assertTrue(paddle.is_tensor(out[0]))
        # detach must hand back a distinct tensor, not the graph-attached input
        self.assertIsNot(out[0], src)
        np.testing.assert_array_equal(out[0].numpy(), src.numpy())

    def test_preserves_stop_gradient_per_argument(self):
        trainable = paddle.to_tensor([1.0])
        trainable.stop_gradient = False
        frozen = paddle.to_tensor([2.0])
        frozen.stop_gradient = True
        out = detach_and_requires_grad_(trainable, frozen)
        self.assertEqual(out[0].stop_gradient, False)
        self.assertEqual(out[1].stop_gradient, True)
        np.testing.assert_array_equal(out[0].numpy(), [1.0])
        np.testing.assert_array_equal(out[1].numpy(), [2.0])

    def test_passes_non_tensor_values_through_unchanged(self):
        marker = object()
        out = detach_and_requires_grad_(7, "text", marker, [1, 2])
        self.assertEqual(out[0], 7)
        self.assertEqual(out[1], "text")
        self.assertIs(out[2], marker)
        self.assertEqual(out[3], [1, 2])

    def test_keeps_ordering_when_mixing_tensor_and_non_tensor(self):
        t = paddle.to_tensor([5.0, 6.0])
        t.stop_gradient = False
        out = detach_and_requires_grad_("a", t, 9)
        self.assertEqual(out[0], "a")
        self.assertTrue(paddle.is_tensor(out[1]))
        np.testing.assert_array_equal(out[1].numpy(), [5.0, 6.0])
        self.assertEqual(out[1].stop_gradient, False)
        self.assertEqual(out[2], 9)


@requires_paddle
class TestAddAuxiliaryLoss(unittest.TestCase):
    def test_forward_returns_value_equal_clone(self):
        x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        loss = paddle.to_tensor([0.5])
        out = AddAuxiliaryLoss.apply(x, loss)
        np.testing.assert_array_equal(out.numpy(), x.numpy())
        # clone: a distinct tensor, so the aux path cannot alias the input
        self.assertIsNot(out, x)

    def test_backward_forwards_x_grad_and_injects_unit_aux_grad(self):
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        x.stop_gradient = False
        loss = paddle.to_tensor([7.0])
        loss.stop_gradient = False
        out = AddAuxiliaryLoss.apply(x, loss)
        upstream = paddle.to_tensor([[0.1, 0.2], [0.3, 0.4]])
        out.backward(upstream)
        # x receives the upstream gradient unchanged
        np.testing.assert_allclose(
            x.grad.numpy(), upstream.numpy(), rtol=1e-6, atol=1e-7
        )
        # the aux-loss gradient is a constant one, independent of value/upstream
        np.testing.assert_array_equal(loss.grad.numpy(), np.array([1.0]))

    def test_aux_gradient_reaches_upstream_parameter(self):
        # ``loss`` is derived from ``w`` but does not affect the forward value
        # of ``out`` (which equals ``x``). With a zero upstream gradient the
        # only gradient reaching ``w`` is the injected unit aux-loss gradient,
        # i.e. d(loss)/d(w) * 1 = ones.
        w = paddle.to_tensor([2.0, 3.0, 4.0])
        w.stop_gradient = False
        loss = w.sum()
        x = paddle.to_tensor([[1.0, 1.0], [1.0, 1.0]])
        x.stop_gradient = False
        out = AddAuxiliaryLoss.apply(x, loss)
        out.backward(paddle.zeros_like(x))
        self.assertIsNotNone(w.grad)
        np.testing.assert_allclose(
            w.grad.numpy(),
            np.ones([3], dtype="float32"),
            rtol=1e-6,
            atol=1e-7,
        )


@requires_paddle
class TestPermute(unittest.TestCase):
    def test_groups_token_indices_by_ascending_expert(self):
        tokens = paddle.to_tensor(
            [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]]
        )
        routing_map = paddle.to_tensor(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype="int64"
        )
        permuted, sorted_indices = permute(tokens, routing_map)
        # expert0 <- tokens 0,3 ; expert1 <- token 1 ; expert2 <- token 2
        self.assertEqual(sorted_indices.numpy().tolist(), [0, 3, 1, 2])
        np.testing.assert_array_equal(
            permuted.numpy(),
            np.array(
                [[10.0, 11.0], [40.0, 41.0], [20.0, 21.0], [30.0, 31.0]],
                dtype="float32",
            ),
        )

    def test_token_routed_to_multiple_experts_is_replicated(self):
        tokens = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        routing_map = paddle.to_tensor([[1, 0, 1], [0, 1, 0]], dtype="int64")
        permuted, sorted_indices = permute(tokens, routing_map)
        # expert0 <- t0 ; expert1 <- t1 ; expert2 <- t0 (t0 appears twice)
        self.assertEqual(sorted_indices.numpy().tolist(), [0, 1, 0])
        np.testing.assert_array_equal(
            permuted.numpy(),
            np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]], dtype="float32"),
        )

    def test_rejects_drop_and_pad(self):
        with self.assertRaises(AssertionError):
            permute(
                paddle.to_tensor([[1.0, 2.0]]),
                paddle.to_tensor([[1, 0]], dtype="int64"),
                drop_and_pad=True,
            )


@requires_paddle
class TestUnpermute(unittest.TestCase):
    def test_scatters_rows_back_to_original_positions(self):
        permuted = paddle.to_tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        )
        sorted_indices = paddle.to_tensor([0, 3, 1, 2], dtype="int64")
        restored = unpermute(permuted, sorted_indices, [4, 2])
        # row i of ``permuted`` lands at position sorted_indices[i]
        np.testing.assert_array_equal(
            restored.numpy(),
            np.array(
                [[1.0, 2.0], [5.0, 6.0], [7.0, 8.0], [3.0, 4.0]],
                dtype="float32",
            ),
        )

    def test_duplicate_indices_accumulate(self):
        permuted = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]])
        sorted_indices = paddle.to_tensor([0, 1, 0], dtype="int64")
        restored = unpermute(permuted, sorted_indices, [2, 2])
        # position 0 receives rows 0 and 2 summed; position 1 receives row 1
        np.testing.assert_array_equal(
            restored.numpy(),
            np.array([[11.0, 22.0], [3.0, 4.0]], dtype="float32"),
        )

    def test_rejects_drop_and_pad(self):
        with self.assertRaises(AssertionError):
            unpermute(
                paddle.to_tensor([[1.0, 2.0]]),
                paddle.to_tensor([0], dtype="int64"),
                [1, 2],
                drop_and_pad=True,
            )


@requires_paddle
class TestPermuteUnpermuteRoundTrip(unittest.TestCase):
    def test_single_expert_per_token_round_trips_to_identity(self):
        tokens = paddle.to_tensor(
            [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]]
        )
        routing_map = paddle.to_tensor(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype="int64"
        )
        permuted, sorted_indices = permute(tokens, routing_map)
        restored = unpermute(permuted, sorted_indices, tokens.shape)
        np.testing.assert_array_equal(restored.numpy(), tokens.numpy())

    def test_multi_expert_token_round_trips_to_summed_copies(self):
        # token 0 selects two experts, so unpermute sums its two copies back
        # onto position 0 while token 1 is restored unchanged.
        tokens = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        routing_map = paddle.to_tensor([[1, 0, 1], [0, 1, 0]], dtype="int64")
        permuted, sorted_indices = permute(tokens, routing_map)
        restored = unpermute(permuted, sorted_indices, tokens.shape)
        np.testing.assert_array_equal(
            restored.numpy(),
            np.array([[2.0, 4.0], [3.0, 4.0]], dtype="float32"),
        )


@requires_paddle
class TestLogMoeLosses(unittest.TestCase):
    def setUp(self):
        unset_global_variables()
        set_global_training_logs(_EnabledLogs())

    def tearDown(self):
        unset_global_variables()

    def test_logs_aux_and_z_loss_with_global_and_layer_keys(self):
        log_moe_losses(
            layer_number=2,
            aux_loss=paddle.to_tensor([1.25]),
            z_loss=paddle.to_tensor([2.5]),
        )
        logs = get_global_training_logs()
        self.assertAlmostEqual(logs["aux_loss"].item(), 1.25, places=6)
        self.assertAlmostEqual(logs["aux_loss_layer_2"].item(), 1.25, places=6)
        self.assertAlmostEqual(logs["zloss"].item(), 2.5, places=6)
        self.assertAlmostEqual(logs["zloss_layer_2"].item(), 2.5, places=6)

    def test_skips_none_aux_loss_but_logs_z_loss(self):
        log_moe_losses(
            layer_number=4,
            aux_loss=None,
            z_loss=paddle.to_tensor([3.5]),
        )
        logs = get_global_training_logs()
        self.assertNotIn("aux_loss", logs)
        self.assertNotIn("aux_loss_layer_4", logs)
        self.assertAlmostEqual(logs["zloss"].item(), 3.5, places=6)
        self.assertAlmostEqual(logs["zloss_layer_4"].item(), 3.5, places=6)

    def test_without_layer_number_only_global_keys(self):
        log_moe_losses(
            layer_number=None,
            aux_loss=paddle.to_tensor([1.0]),
            z_loss=paddle.to_tensor([2.0]),
        )
        logs = get_global_training_logs()
        self.assertAlmostEqual(logs["aux_loss"].item(), 1.0, places=6)
        self.assertAlmostEqual(logs["zloss"].item(), 2.0, places=6)
        self.assertNotIn("aux_loss_layer_None", logs)
        self.assertNotIn("zloss_layer_None", logs)

    def test_returns_early_when_balance_logs_disabled(self):
        unset_global_variables()
        set_global_training_logs(_DisabledLogs())
        log_moe_losses(
            layer_number=1,
            aux_loss=paddle.to_tensor([1.0]),
            z_loss=paddle.to_tensor([2.0]),
        )
        # gate is off: nothing is written to the logs object
        self.assertEqual(get_global_training_logs(), {})


@requires_paddle
class TestFakeClone(unittest.TestCase):
    def test_forward_preserves_values(self):
        x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = FakeClone.apply(x)
        np.testing.assert_array_equal(out.numpy(), x.numpy())

    def test_backward_passes_gradient_through_unchanged(self):
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        x.stop_gradient = False
        out = FakeClone.apply(x)
        upstream = paddle.to_tensor([[0.5, 1.5], [2.5, 3.5]])
        out.backward(upstream)
        np.testing.assert_allclose(
            x.grad.numpy(), upstream.numpy(), rtol=1e-6, atol=1e-7
        )


if __name__ == "__main__":
    unittest.main()
