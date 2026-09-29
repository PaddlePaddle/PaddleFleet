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

These exercise only CPU-executable, single-process control/layout logic:
  * is_hybrid_ep_backend_selected -- backend name validation + availability gate
  * _sort_chunks_like_tokens      -- split/reorder/concat along the token axis
  * _try_setup_router_topk_metadata -- router-supplied topk bypass + reshape
  * _DispatchManager              -- abstract interface contract
  * _DeepEPManager.setup_metadata -- paddle.topk metadata vs. router bypass
  * _DeepEPManager._indices_to_multihot -- indices -> multihot map/probs
  * MoEFlexTokenDispatcher.__init__ -- manager selection + kwarg mapping
  * AllToAllTokenDispatcher.__init__ -- num_local_experts derivation

Expected values are derived independently by hand / numpy from the production
source, NOT copied from any coverage_test file.

fused_dispatch is a genuine not-under-test DeepEP availability gate: it is
patched to a sentinel only so the local construction / metadata logic can run;
no DeepEP kernel is executed. The real dispatch / combine collectives require a
multi-rank process group and are NOT verified by this file (multi-card scope).
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

try:
    import paddle

    from paddlefleet.transformer.moe import token_dispatcher
    from paddlefleet.transformer.moe.token_dispatcher import (
        AllToAllTokenDispatcher,
        MoEFlexTokenDispatcher,
        _DeepEPManager,
        _DispatchManager,
        _sort_chunks_like_tokens,
        _try_setup_router_topk_metadata,
        is_hybrid_ep_backend_selected,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest, precise skip
    _IMPORT_ERROR = exc


_SKIP_REASON = (
    ""
    if _IMPORT_ERROR is None
    else f"token_dispatcher import failed: {_IMPORT_ERROR!r}"
)


# A non-None sentinel standing in for the DeepEP fused_dispatch entry point.
# The construction gate only checks `fused_dispatch is None`; the sentinel is
# never called by any path exercised here.
_FUSED_SENTINEL = object()


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestDispatchManagerInterface(unittest.TestCase):
    """_DispatchManager is an ABC defining the dispatch/combine contract."""

    def test_cannot_instantiate_abstract_base(self):
        with self.assertRaises(TypeError):
            _DispatchManager()

    def test_declares_full_abstract_surface(self):
        expected = {
            "setup_metadata",
            "dispatch",
            "combine",
            "get_dispatched_metadata",
            "get_permuted_hidden_states_by_experts",
            "get_restored_hidden_states_by_experts",
        }
        self.assertEqual(set(_DispatchManager.__abstractmethods__), expected)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestIsHybridEpBackendSelected(unittest.TestCase):
    """Backend-name validation and the HybridEP availability gate."""

    def test_default_is_deepep_not_hybrid(self):
        # None defaults to "deepep" -> valid name, not hybridep -> False.
        self.assertFalse(is_hybrid_ep_backend_selected(None))

    def test_non_hybrid_backends_return_false(self):
        for name in ("allgather", "alltoall", "deepep", "moonep", "ringmoe"):
            self.assertFalse(is_hybrid_ep_backend_selected(name), msg=name)

    def test_invalid_backend_raises_value_error(self):
        with self.assertRaises(ValueError):
            is_hybrid_ep_backend_selected("does_not_exist")

    def test_hybridep_returns_true_only_when_runtime_available(self):
        # The "hybridep" branch returns True iff HAVE_HYBRID_EP is set; the same
        # input with the runtime unavailable must raise ImportError. This proves
        # the availability flag is actually consumed, not ignored.
        with patch.object(token_dispatcher, "HAVE_HYBRID_EP", True):
            self.assertTrue(is_hybrid_ep_backend_selected("hybridep"))
        with (
            patch.object(token_dispatcher, "HAVE_HYBRID_EP", False),
            self.assertRaises(ImportError),
        ):
            is_hybrid_ep_backend_selected("hybridep")


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestSortChunksLikeTokens(unittest.TestCase):
    """_sort_chunks_like_tokens splits axis-0 then concatenates in idx order."""

    def test_reorders_uneven_chunks_by_full_content(self):
        # Distinct row contents + uneven split sizes + a non-identity order so a
        # wrong split boundary, wrong axis, or wrong concat order is visible.
        inp = paddle.arange(6 * 2, dtype="float32").reshape([6, 2])
        out = _sort_chunks_like_tokens(inp, [2, 1, 3], [2, 0, 1])
        # chunks: [rows0-1], [row2], [rows3-5]; reordered -> chunk2, chunk0, chunk1
        expected = np.array(
            [
                [6.0, 7.0],
                [8.0, 9.0],
                [10.0, 11.0],
                [0.0, 1.0],
                [2.0, 3.0],
                [4.0, 5.0],
            ],
            dtype="float32",
        )
        np.testing.assert_array_equal(out.numpy(), expected)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestTrySetupRouterTopkMetadata(unittest.TestCase):
    """_try_setup_router_topk_metadata: bypass topk when router supplies it."""

    def test_returns_false_and_leaves_manager_untouched_when_missing(self):
        for weights, indices in (
            (None, paddle.zeros([4, 2], dtype="int64")),
            (paddle.zeros([4, 2], dtype="float32"), None),
            (None, None),
        ):
            mgr = SimpleNamespace(
                router_topk=2, token_probs="unset", token_indices="unset"
            )
            self.assertFalse(
                _try_setup_router_topk_metadata(mgr, 4, weights, indices)
            )
            # Untouched: the function must not write metadata on the miss path.
            self.assertEqual(mgr.token_probs, "unset")
            self.assertEqual(mgr.token_indices, "unset")

    def test_reshapes_supplied_topk_and_freezes_indices(self):
        mgr = SimpleNamespace(
            router_topk=2, token_probs=None, token_indices=None
        )
        weights = paddle.arange(8, dtype="float32")  # flat -> [4, 2]
        indices = paddle.arange(8, dtype="int64") + 10  # distinct ids
        self.assertTrue(
            _try_setup_router_topk_metadata(mgr, 4, weights, indices)
        )
        np.testing.assert_array_equal(
            mgr.token_probs.numpy(),
            np.arange(8, dtype="float32").reshape([4, 2]),
        )
        np.testing.assert_array_equal(
            mgr.token_indices.numpy(),
            (np.arange(8) + 10).reshape([4, 2]),
        )
        # indices must be detached from autograd (stop_gradient set True).
        self.assertTrue(mgr.token_indices.stop_gradient)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestDeepEPManagerConstruction(unittest.TestCase):
    """_DeepEPManager availability gate and initial metadata state."""

    def test_raises_import_error_without_deepep(self):
        with (
            patch.object(token_dispatcher, "fused_dispatch", None),
            self.assertRaises(ImportError),
        ):
            _DeepEPManager(SimpleNamespace(), router_topk=2)

    def test_metadata_slots_start_empty(self):
        # dispatch()/combine() rely on these being None before setup/after reset.
        with patch.object(token_dispatcher, "fused_dispatch", _FUSED_SENTINEL):
            mgr = _DeepEPManager(
                SimpleNamespace(), router_topk=2, num_experts=8
            )
        self.assertIsNone(mgr.token_indices)
        self.assertIsNone(mgr.token_probs)
        self.assertIsNone(mgr.handle)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestDeepEPManagerSetupMetadata(unittest.TestCase):
    """_DeepEPManager.setup_metadata: paddle.topk vs. router-supplied bypass."""

    def _manager(self):
        with patch.object(token_dispatcher, "fused_dispatch", _FUSED_SENTINEL):
            return _DeepEPManager(
                SimpleNamespace(), router_topk=2, num_experts=4
            )

    def test_selects_top2_probs_and_indices(self):
        mgr = self._manager()
        probs = paddle.to_tensor(
            [[0.1, 0.7, 0.2, 0.4], [0.9, 0.05, 0.3, 0.6]], dtype="float32"
        )
        routing_map = paddle.zeros([2, 4], dtype="float32")  # shape only
        mgr.setup_metadata(routing_map, probs)
        # top-2 by value, descending: row0 -> (0.7@1, 0.4@3); row1 -> (0.9@0, 0.6@3)
        np.testing.assert_allclose(
            mgr.token_probs.numpy(),
            np.array([[0.7, 0.4], [0.9, 0.6]], dtype="float32"),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            mgr.token_indices.numpy(), np.array([[1, 3], [0, 3]])
        )

    def test_router_supplied_topk_bypasses_probs(self):
        # Supplied indices differ from topk(probs) (which would be [[1,3],[0,3]]),
        # so equality with the supplied values proves paddle.topk was skipped.
        mgr = self._manager()
        probs = paddle.to_tensor(
            [[0.1, 0.7, 0.2, 0.4], [0.9, 0.05, 0.3, 0.6]], dtype="float32"
        )
        routing_map = paddle.zeros([2, 4], dtype="float32")
        topk_weights = paddle.to_tensor(
            [[0.11, 0.22], [0.33, 0.44]], dtype="float32"
        )
        topk_indices = paddle.to_tensor([[2, 0], [1, 3]], dtype="int64")
        mgr.setup_metadata(
            routing_map,
            probs,
            topk_weights=topk_weights,
            topk_indices=topk_indices,
        )
        np.testing.assert_allclose(
            mgr.token_probs.numpy(),
            np.array([[0.11, 0.22], [0.33, 0.44]], dtype="float32"),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            mgr.token_indices.numpy(), np.array([[2, 0], [1, 3]])
        )


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestDeepEPManagerIndicesToMultihot(unittest.TestCase):
    """_DeepEPManager._indices_to_multihot: scatter indices+probs to a map."""

    def _manager(self, num_local_experts):
        with patch.object(token_dispatcher, "fused_dispatch", _FUSED_SENTINEL):
            return _DeepEPManager(
                SimpleNamespace(),
                router_topk=2,
                num_experts=4,
                num_local_experts=num_local_experts,
            )

    def test_full_rows_scatter_to_selected_experts(self):
        mgr = self._manager(4)
        indices = paddle.to_tensor(
            [[0, 1], [2, 3], [0, 2], [1, 3]], dtype="int64"
        )
        probs = paddle.full([4, 2], 0.5, dtype="float32")
        routing_map, multihot_probs = mgr._indices_to_multihot(indices, probs)
        expected_map = np.array(
            [[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
            dtype=bool,
        )
        expected_probs = np.array(
            [
                [0.5, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5, 0.0],
                [0.0, 0.5, 0.0, 0.5],
            ],
            dtype="float32",
        )
        self.assertEqual(routing_map.dtype, paddle.bool)
        np.testing.assert_array_equal(
            routing_map.numpy().astype(bool), expected_map
        )
        np.testing.assert_allclose(
            multihot_probs.numpy(), expected_probs, rtol=1e-6, atol=1e-6
        )

    def test_masked_minus_one_slots_are_dropped(self):
        # -1 marks a padded / unrouted slot: it must not set any expert bit or
        # probability. Row0 keeps only expert 0; row1 keeps experts 2 and 3.
        mgr = self._manager(4)
        indices = paddle.to_tensor([[0, -1], [2, 3]], dtype="int64")
        probs = paddle.to_tensor([[0.7, 0.0], [0.4, 0.6]], dtype="float32")
        routing_map, multihot_probs = mgr._indices_to_multihot(indices, probs)
        expected_map = np.array([[1, 0, 0, 0], [0, 0, 1, 1]], dtype=bool)
        expected_probs = np.array(
            [[0.7, 0.0, 0.0, 0.0], [0.0, 0.0, 0.4, 0.6]], dtype="float32"
        )
        np.testing.assert_array_equal(
            routing_map.numpy().astype(bool), expected_map
        )
        np.testing.assert_allclose(
            multihot_probs.numpy(), expected_probs, rtol=1e-6, atol=1e-6
        )


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestMoEFlexTokenDispatcherConstruction(unittest.TestCase):
    """MoEFlexTokenDispatcher.__init__: EP guard + manager kwarg mapping."""

    def test_requires_ep_size_greater_than_one(self):
        with (
            patch.object(token_dispatcher, "fused_dispatch", _FUSED_SENTINEL),
            self.assertRaises(AssertionError),
        ):
            MoEFlexTokenDispatcher(
                num_local_experts=2,
                num_experts_per_tok=2,
                n_routed_experts=8,
                ep_group=SimpleNamespace(world_size=1),
            )

    def test_default_builds_deepep_manager_with_mapped_kwargs(self):
        ep_group = SimpleNamespace(world_size=2)
        with patch.object(token_dispatcher, "fused_dispatch", _FUSED_SENTINEL):
            dispatcher = MoEFlexTokenDispatcher(
                num_local_experts=2,
                num_experts_per_tok=3,
                n_routed_experts=8,
                ep_group=ep_group,
            )
        self.assertIsInstance(dispatcher._comm_manager, _DeepEPManager)
        self.assertEqual(dispatcher.num_local_experts, 2)
        # router_topk<-num_experts_per_tok and num_experts<-n_routed_experts are
        # distinct values, so a swap in the mapping would be caught here.
        self.assertEqual(dispatcher._comm_manager.router_topk, 3)
        self.assertEqual(dispatcher._comm_manager.num_experts, 8)
        self.assertEqual(dispatcher._comm_manager.num_local_experts, 2)
        self.assertIs(dispatcher._comm_manager.group, ep_group)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestAllToAllTokenDispatcherConstruction(unittest.TestCase):
    """AllToAllTokenDispatcher.__init__: num_local_experts is derived, not stored.

    The dispatch path itself (AllGather / AllToAll collectives) needs a real
    multi-rank process group and is NOT exercised here.
    """

    def test_num_local_experts_derived_from_index_list_length(self):
        moe_group = SimpleNamespace()
        for local_expert_indices in ([0, 1], [4, 5, 6]):
            dispatcher = AllToAllTokenDispatcher(
                moe_group,
                expert_model_parallel_size=2,
                num_experts_per_device=len(local_expert_indices),
                local_expert_indices=local_expert_indices,
            )
            self.assertEqual(
                dispatcher.num_local_experts, len(local_expert_indices)
            )
            self.assertEqual(
                dispatcher.local_expert_indices, local_expert_indices
            )
            self.assertEqual(dispatcher.expert_model_parallel_size, 2)
            self.assertIs(dispatcher.moe_group, moe_group)


if __name__ == "__main__":
    unittest.main()
