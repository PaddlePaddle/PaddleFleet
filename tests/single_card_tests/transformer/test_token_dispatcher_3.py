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

"""No-card behavior tests for paddlefleet.transformer.moe.token_dispatcher.

These exercise CPU-executable, non-collective logic only:
  * _DeepEPManager.setup_metadata -- top-k selection vs. router-provided reuse
  * MoETokenDispatcher.ep_group / ep_size -- EP-group property forwarding
  * AddAuxiliaryLoss (moe_utils)  -- clone forward + aux-loss gradient injection

setup_metadata builds routing metadata locally (paddle.topk / reshape); it does
NOT invoke any collective, so it is validated here directly. The module-level
`fused_dispatch` symbol is only an availability gate inside _DeepEPManager
construction (it raises ImportError when the DeepEP whl is missing); it is a
genuine not-under-test collaborator and is patched to a sentinel purely to let
the constructor run on CPU. The real fused dispatch/combine collectives and
cross-rank behavior are NOT verified by this file.

Expected values are derived independently with numpy / by hand from the
production source, not from any coverage_test file.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

# Sentinel used only to satisfy the `fused_dispatch is None` gate in
# _DeepEPManager.__init__; setup_metadata never touches this symbol.
_FUSED_DISPATCH_SENTINEL = object()

try:
    import paddle

    from paddlefleet.transformer.moe import token_dispatcher as td_module
    from paddlefleet.transformer.moe.moe_utils import AddAuxiliaryLoss
    from paddlefleet.transformer.moe.token_dispatcher import (
        MoETokenDispatcher,
        _DeepEPManager,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest, precise skip
    _IMPORT_ERROR = exc


_SKIP_REASON = (
    ""
    if _IMPORT_ERROR is None
    else f"token_dispatcher import failed: {_IMPORT_ERROR!r}"
)


def _numpy_topk(probs, k):
    """Independent top-k reference: descending values and their indices."""
    order = np.argsort(-probs, axis=-1, kind="stable")[:, :k]
    values = np.take_along_axis(probs, order, axis=-1)
    return values, order


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestDeepEPManagerSetupMetadata(unittest.TestCase):
    """_DeepEPManager.setup_metadata metadata construction (no collectives)."""

    def _make_manager(self, router_topk=2, num_experts=4):
        # Patch only spans construction; the availability gate is not under test.
        with patch.object(
            td_module, "fused_dispatch", _FUSED_DISPATCH_SENTINEL
        ):
            return _DeepEPManager(
                group=object(),
                router_topk=router_topk,
                num_experts=num_experts,
            )

    def test_topk_path_selects_correct_experts_and_probs(self):
        # Distinct per-row values so the top-k ordering is unambiguous.
        probs_np = np.array(
            [
                [0.1, 0.7, 0.3, 0.2],
                [0.9, 0.05, 0.5, 0.2],
                [0.2, 0.4, 0.1, 0.8],
            ],
            dtype="float32",
        )
        exp_values, exp_indices = _numpy_topk(probs_np, 2)
        # By hand: row0 -> (1,2), row1 -> (0,2), row2 -> (3,1).
        self.assertEqual(exp_indices.tolist(), [[1, 2], [0, 2], [3, 1]])

        manager = self._make_manager(router_topk=2, num_experts=4)
        routing_map = paddle.zeros([3, 4], dtype="float32")
        probs = paddle.to_tensor(probs_np)

        manager.setup_metadata(routing_map, probs)

        self.assertEqual(manager.token_indices.shape, [3, 2])
        self.assertEqual(manager.token_probs.shape, [3, 2])
        self.assertEqual(
            manager.token_indices.numpy().tolist(), exp_indices.tolist()
        )
        np.testing.assert_allclose(
            manager.token_probs.numpy(), exp_values, atol=1e-6
        )

    def test_router_provided_topk_is_used_verbatim(self):
        # probs would top-k to [[1,2],...]; provided indices differ on purpose,
        # so equality proves the internal paddle.topk was skipped, not recomputed.
        probs = paddle.to_tensor(
            [
                [0.1, 0.7, 0.3, 0.2],
                [0.9, 0.05, 0.5, 0.2],
                [0.2, 0.4, 0.1, 0.8],
            ],
            dtype="float32",
        )
        provided_indices = paddle.to_tensor(
            [[0, 3], [1, 3], [2, 0]], dtype="int64"
        )
        provided_weights = paddle.to_tensor(
            [[0.11, 0.22], [0.33, 0.44], [0.55, 0.66]], dtype="float32"
        )
        routing_map = paddle.zeros([3, 4], dtype="float32")

        manager = self._make_manager(router_topk=2, num_experts=4)
        manager.setup_metadata(
            routing_map,
            probs,
            topk_weights=provided_weights,
            topk_indices=provided_indices,
        )

        self.assertEqual(
            manager.token_indices.numpy().tolist(),
            provided_indices.numpy().tolist(),
        )
        np.testing.assert_allclose(
            manager.token_probs.numpy(), provided_weights.numpy(), atol=1e-6
        )
        # Production forces the reused indices to stop gradient.
        self.assertTrue(manager.token_indices.stop_gradient)

    def test_partial_router_topk_falls_back_to_internal_topk(self):
        # Only weights provided (indices None) -> must NOT take the reuse branch;
        # falls back to paddle.topk over probs.
        probs_np = np.array(
            [[0.2, 0.4, 0.1, 0.8], [0.9, 0.05, 0.5, 0.2]], dtype="float32"
        )
        exp_values, exp_indices = _numpy_topk(probs_np, 2)

        manager = self._make_manager(router_topk=2, num_experts=4)
        routing_map = paddle.zeros([2, 4], dtype="float32")
        manager.setup_metadata(
            routing_map,
            paddle.to_tensor(probs_np),
            topk_weights=paddle.to_tensor(
                [[9.0, 9.0], [9.0, 9.0]], dtype="float32"
            ),
            topk_indices=None,
        )

        self.assertEqual(
            manager.token_indices.numpy().tolist(), exp_indices.tolist()
        )
        np.testing.assert_allclose(
            manager.token_probs.numpy(), exp_values, atol=1e-6
        )


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestMoETokenDispatcherProperties(unittest.TestCase):
    """MoETokenDispatcher forwards the EP group and its world_size."""

    def test_ep_group_identity_and_ep_size_reads_world_size(self):
        # The EP group is a genuine not-under-test collaborator; a plain stub
        # with a specific world_size makes the forwarding observable.
        ep_group = SimpleNamespace(world_size=8)
        dispatcher = MoETokenDispatcher(ep_group)

        self.assertIs(dispatcher.ep_group, ep_group)
        self.assertEqual(dispatcher.ep_size, 8)

        # ep_size must re-read the attribute, not cache a constant.
        ep_group.world_size = 3
        self.assertEqual(dispatcher.ep_size, 3)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestAddAuxiliaryLoss(unittest.TestCase):
    """AddAuxiliaryLoss: identity clone forward + unit aux-loss gradient."""

    def test_forward_returns_value_clone_not_alias(self):
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32"
        )
        loss = paddle.to_tensor([0.5], dtype="float32")
        loss.stop_gradient = False

        out = AddAuxiliaryLoss.apply(x, loss)

        self.assertEqual(out.shape, x.shape)
        np.testing.assert_array_equal(out.numpy(), x.numpy())
        self.assertIsNot(out, x)  # forward returns x.clone(), a distinct tensor

    def test_non_scalar_loss_rejected(self):
        x = paddle.to_tensor([[1.0, 2.0]], dtype="float32")
        loss = paddle.to_tensor([0.5, 0.3], dtype="float32")  # numel != 1
        loss.stop_gradient = False
        with self.assertRaises(AssertionError):
            AddAuxiliaryLoss.apply(x, loss)

    def test_backward_passes_dx_through_and_injects_unit_aux_grad(self):
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32"
        )
        x.stop_gradient = False
        loss = paddle.to_tensor([0.5], dtype="float32")
        loss.stop_gradient = False

        out = AddAuxiliaryLoss.apply(x, loss)
        # Non-uniform upstream exposes scaling / permutation errors in dx.
        upstream = paddle.to_tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype="float32"
        )
        out.backward(upstream)

        self.assertIsNotNone(x.grad)
        np.testing.assert_allclose(x.grad.numpy(), upstream.numpy(), atol=1e-6)
        # Aux loss gradient is exactly ones(1), independent of upstream scale.
        self.assertIsNotNone(loss.grad)
        np.testing.assert_allclose(loss.grad.numpy(), np.ones(1), atol=1e-6)

    def test_backward_skips_aux_grad_when_loss_detached(self):
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32"
        )
        x.stop_gradient = False
        loss = paddle.to_tensor([0.5], dtype="float32")
        loss.stop_gradient = True  # required_aux_loss becomes False

        out = AddAuxiliaryLoss.apply(x, loss)
        upstream = paddle.to_tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype="float32"
        )
        out.backward(upstream)

        # dx still flows through unchanged...
        self.assertIsNotNone(x.grad)
        np.testing.assert_allclose(x.grad.numpy(), upstream.numpy(), atol=1e-6)
        # ...but no aux gradient is produced for a detached loss.
        self.assertIsNone(loss.grad)


if __name__ == "__main__":
    unittest.main()
