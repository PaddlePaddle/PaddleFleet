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
from types import SimpleNamespace
from unittest import mock

import numpy as np

try:
    import paddle

    from paddlefleet.transformer.moe.token_dispatcher import (
        HAVE_HYBRID_EP,
        AllToAllTokenDispatcher,
        MoETokenDispatcher,
        _DeepEPManager,
        _DispatchManager,
        _sort_chunks_like_tokens,
        _try_setup_router_topk_metadata,
        is_hybrid_ep_backend_selected,
    )

    _IMPORT_ERROR = None
except (
    ImportError,
    ModuleNotFoundError,
) as exc:  # CPU env without paddle/deepep
    paddle = None
    _IMPORT_ERROR = exc

_HAS_PADDLE = paddle is not None
_SKIP_REASON = (
    "paddle / paddlefleet.transformer.moe.token_dispatcher is not importable "
    f"in this environment: {_IMPORT_ERROR}"
)

# NOTE ON SCOPE
# These tests only exercise CPU-executable control logic of token_dispatcher:
# the dispatcher-type guard, router-topk metadata setup, chunk reordering,
# the DeepEP index->multihot scatter, the abstract-manager contract, and
# constructor-derived fields. The dispatch/combine/token_dispatch paths issue
# real collectives (AllGatherGroupOp / _AllToAll / fused_dispatch) whose
# correctness depends on multiple ranks exchanging distinct data; they are NOT
# covered here and must be verified with a real process group (multi-card).
# We deliberately do not fake world_size + mock collectives for such asserts.


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestIsHybridEPBackendSelected(unittest.TestCase):
    """is_hybrid_ep_backend_selected: type guard + hybridep availability."""

    def test_non_hybrid_backends_return_false(self):
        # Every valid non-hybridep dispatcher, incl. the None default which
        # resolves to "deepep", must report False (no hybridep runtime needed).
        self.assertFalse(is_hybrid_ep_backend_selected(None))
        for name in ("allgather", "alltoall", "deepep", "moonep", "ringmoe"):
            self.assertFalse(is_hybrid_ep_backend_selected(name))

    def test_hybridep_follows_runtime_availability(self):
        # "hybridep" returns True only when the runtime is present; otherwise
        # it must raise ImportError (not silently fall back to False).
        if HAVE_HYBRID_EP:
            self.assertTrue(is_hybrid_ep_backend_selected("hybridep"))
        else:
            with self.assertRaises(ImportError):
                is_hybrid_ep_backend_selected("hybridep")

    def test_unknown_backend_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            is_hybrid_ep_backend_selected("not_a_backend")
        # The message must enumerate the accepted values so misconfig is clear.
        self.assertIn("allgather", str(ctx.exception))
        self.assertIn("hybridep", str(ctx.exception))


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestTrySetupRouterTopkMetadata(unittest.TestCase):
    """_try_setup_router_topk_metadata: use router-provided topk verbatim."""

    def test_missing_topk_is_a_noop(self):
        # If either weights or indices are absent, the helper must decline
        # (return False) and leave the manager untouched so the caller falls
        # back to its own paddle.topk.
        weights = paddle.to_tensor([[0.1, 0.2]], dtype="float32")
        indices = paddle.to_tensor([[0, 1]], dtype="int64")
        for tw, ti in ((None, None), (weights, None), (None, indices)):
            manager = SimpleNamespace(router_topk=2)
            used = _try_setup_router_topk_metadata(
                manager, num_tokens=1, topk_weights=tw, topk_indices=ti
            )
            self.assertFalse(used)
            self.assertFalse(hasattr(manager, "token_probs"))
            self.assertFalse(hasattr(manager, "token_indices"))

    def test_reshapes_and_freezes_indices(self):
        manager = SimpleNamespace(router_topk=2)
        weights = paddle.to_tensor(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype="float32"
        )
        indices = paddle.to_tensor([0, 1, 2, 3, 0, 1], dtype="int64")

        used = _try_setup_router_topk_metadata(
            manager, num_tokens=3, topk_weights=weights, topk_indices=indices
        )

        self.assertTrue(used)
        np.testing.assert_array_equal(
            manager.token_probs.numpy(),
            np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            manager.token_indices.numpy(),
            np.array([[0, 1], [2, 3], [0, 1]], dtype=np.int64),
        )
        # Indices are detached from the graph for downstream gather.
        self.assertTrue(manager.token_indices.stop_gradient)


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestSortChunksLikeTokens(unittest.TestCase):
    """_sort_chunks_like_tokens: split along axis 0 then reorder chunks."""

    def test_reorders_variable_sized_chunks(self):
        # Rows carry distinct content so a wrong chunk order (or wrong split
        # boundary) changes the result; a shape/count check would not.
        inp = paddle.to_tensor(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
            dtype="float32",
        )
        out = _sort_chunks_like_tokens(
            inp, split_sizes=[2, 1, 1], sorted_idxs=[2, 0, 1]
        )
        # chunks: [[1,1],[2,2]] | [[3,3]] | [[4,4]]  -> order 2,0,1
        np.testing.assert_array_equal(
            out.numpy(),
            np.array(
                [[4.0, 4.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
                dtype=np.float32,
            ),
        )


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestDispatchManagerAbstractContract(unittest.TestCase):
    """_DispatchManager is an ABC declaring the full dispatch interface."""

    def test_declares_all_dispatch_hooks_abstract(self):
        self.assertEqual(
            _DispatchManager.__abstractmethods__,
            frozenset(
                {
                    "setup_metadata",
                    "dispatch",
                    "combine",
                    "get_dispatched_metadata",
                    "get_permuted_hidden_states_by_experts",
                    "get_restored_hidden_states_by_experts",
                }
            ),
        )

    def test_cannot_instantiate_base_or_partial_subclass(self):
        with self.assertRaises(TypeError):
            _DispatchManager()

        class Partial(_DispatchManager):
            def setup_metadata(self, *a, **k):
                return None

        with self.assertRaises(TypeError):
            Partial()

    def test_complete_subclass_is_instantiable(self):
        class Complete(_DispatchManager):
            def setup_metadata(self, *a, **k):
                return None

            def dispatch(self, *a, **k):
                return "dispatched"

            def combine(self, *a, **k):
                return None

            def get_dispatched_metadata(self):
                return None

            def get_permuted_hidden_states_by_experts(self, h):
                return None

            def get_restored_hidden_states_by_experts(self, h):
                return None

        obj = Complete()
        self.assertEqual(obj.dispatch(), "dispatched")


def _make_deepep_manager(**kwargs):
    """Construct a _DeepEPManager on CPU by treating the fused_dispatch kernel
    (an availability sentinel, not the logic under test) as present."""
    defaults = {
        "group": SimpleNamespace(nranks=1),
        "router_topk": 2,
        "num_experts": 4,
    }
    defaults.update(kwargs)
    with mock.patch(
        "paddlefleet.transformer.moe.token_dispatcher.fused_dispatch",
        object(),
    ):
        return _DeepEPManager(**defaults)


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestDeepEPManagerConstruction(unittest.TestCase):
    """_DeepEPManager.__init__: kernel guard, defaults, and stored fields."""

    def test_missing_kernel_raises_import_error(self):
        # When the DeepEP fused kernel is unavailable, construction must fail
        # loudly rather than produce a half-built manager.
        with (
            mock.patch(
                "paddlefleet.transformer.moe.token_dispatcher.fused_dispatch",
                None,
            ),
            self.assertRaises(ImportError) as ctx,
        ):
            _DeepEPManager(group=SimpleNamespace(), router_topk=2)
        self.assertIn("DeepEP", str(ctx.exception))

    def test_defaults_and_initial_metadata(self):
        group = SimpleNamespace(nranks=1)
        mgr = _make_deepep_manager(
            group=group, router_topk=2, num_experts=8, num_local_experts=4
        )
        self.assertIs(mgr.group, group)
        self.assertEqual(mgr.router_topk, 2)
        self.assertEqual(mgr.num_experts, 8)
        self.assertEqual(mgr.num_local_experts, 4)
        # Documented defaults.
        self.assertTrue(mgr.moe_ep_barrier)
        self.assertFalse(mgr.use_accuracy_compatible)
        # Metadata starts empty; it is only filled by setup_metadata/dispatch.
        self.assertIsNone(mgr.token_indices)
        self.assertIsNone(mgr.token_probs)
        self.assertIsNone(mgr.handle)

    def test_optional_flags_are_consumed(self):
        # Passing non-default flags must change the stored values, proving the
        # parameters are wired through rather than hard-coded.
        mgr = _make_deepep_manager(
            moe_ep_barrier=False, use_accuracy_compatible=True
        )
        self.assertFalse(mgr.moe_ep_barrier)
        self.assertTrue(mgr.use_accuracy_compatible)


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestDeepEPManagerSetupMetadata(unittest.TestCase):
    """_DeepEPManager.setup_metadata: derive topk experts (CPU, no comm)."""

    def test_topk_selection_from_probs(self):
        mgr = _make_deepep_manager(router_topk=2, num_experts=4)
        probs = paddle.to_tensor(
            [[0.1, 0.7, 0.2, 0.0], [0.5, 0.0, 0.3, 0.2]], dtype="float32"
        )
        routing_map = paddle.zeros([2, 4], dtype="float32")

        mgr.setup_metadata(routing_map, probs)

        # Independent top-2 (descending) per row.
        np.testing.assert_array_equal(
            mgr.token_indices.numpy(),
            np.array([[1, 2], [0, 2]], dtype=np.int64),
        )
        np.testing.assert_allclose(
            mgr.token_probs.numpy(),
            np.array([[0.7, 0.2], [0.5, 0.3]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_router_supplied_topk_bypasses_internal_topk(self):
        mgr = _make_deepep_manager(router_topk=2, num_experts=4)
        # probs whose argmax differs from the supplied indices; if the internal
        # paddle.topk ran, token_indices would be [[1,2],[0,2]] instead.
        probs = paddle.to_tensor(
            [[0.1, 0.7, 0.2, 0.0], [0.5, 0.0, 0.3, 0.2]], dtype="float32"
        )
        routing_map = paddle.zeros([2, 4], dtype="float32")
        topk_indices = paddle.to_tensor([[3, 0], [2, 1]], dtype="int64")
        topk_weights = paddle.to_tensor(
            [[0.11, 0.22], [0.33, 0.44]], dtype="float32"
        )

        mgr.setup_metadata(
            routing_map,
            probs,
            topk_weights=topk_weights,
            topk_indices=topk_indices,
        )

        np.testing.assert_array_equal(
            mgr.token_indices.numpy(),
            np.array([[3, 0], [2, 1]], dtype=np.int64),
        )
        np.testing.assert_allclose(
            mgr.token_probs.numpy(),
            np.array([[0.11, 0.22], [0.33, 0.44]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestDeepEPManagerIndicesToMultihot(unittest.TestCase):
    """_DeepEPManager._indices_to_multihot: scatter topk into dense map."""

    def test_scatters_with_masking(self):
        mgr = _make_deepep_manager(num_local_experts=3)
        # Row 1 has a masked slot (-1) that must be dropped.
        indices = paddle.to_tensor([[0, 2], [1, -1]], dtype="int64")
        probs = paddle.to_tensor([[0.6, 0.4], [0.9, 0.0]], dtype="float32")

        routing_map, multihot_probs = mgr._indices_to_multihot(indices, probs)

        np.testing.assert_array_equal(
            routing_map.numpy(),
            np.array([[True, False, True], [False, True, False]]),
        )
        self.assertEqual(routing_map.dtype, paddle.bool)
        np.testing.assert_allclose(
            multihot_probs.numpy(),
            np.array([[0.6, 0.0, 0.4], [0.0, 0.9, 0.0]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestMoETokenDispatcherContract(unittest.TestCase):
    """MoETokenDispatcher base: group wiring + unimplemented hooks."""

    def test_ep_group_identity_and_size(self):
        group = SimpleNamespace(world_size=7)
        dispatcher = MoETokenDispatcher(group)
        self.assertIs(dispatcher.ep_group, group)
        # ep_size is derived from the group's world_size (distinct value 7
        # guards against a hard-coded or swapped attribute).
        self.assertEqual(dispatcher.ep_size, 7)

    def test_permutation_hooks_are_abstract(self):
        dispatcher = MoETokenDispatcher(SimpleNamespace(world_size=1))
        with self.assertRaises(NotImplementedError):
            dispatcher.token_permutation(None, None, None)
        with self.assertRaises(NotImplementedError):
            dispatcher.token_unpermutation(None)


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestAllToAllTokenDispatcherConstruction(unittest.TestCase):
    """AllToAllTokenDispatcher.__init__: derived num_local_experts + fields."""

    def test_num_local_experts_from_indices(self):
        group = SimpleNamespace()
        dispatcher = AllToAllTokenDispatcher(
            moe_group=group,
            expert_model_parallel_size=4,
            num_experts_per_device=5,
            local_expert_indices=[2, 3],
        )
        self.assertIs(dispatcher.moe_group, group)
        self.assertEqual(dispatcher.expert_model_parallel_size, 4)
        self.assertEqual(dispatcher.num_experts_per_device, 5)
        self.assertEqual(dispatcher.local_expert_indices, [2, 3])
        # num_local_experts is the count of local indices (2), NOT the
        # per-device expert count (5); a swap would go unnoticed if they matched.
        self.assertEqual(dispatcher.num_local_experts, 2)
        self.assertFalse(dispatcher.use_accuracy_compatible)

    def test_accuracy_compatible_flag_consumed(self):
        dispatcher = AllToAllTokenDispatcher(
            moe_group=SimpleNamespace(),
            expert_model_parallel_size=2,
            num_experts_per_device=2,
            local_expert_indices=[0, 1],
            use_accuracy_compatible=True,
        )
        self.assertTrue(dispatcher.use_accuracy_compatible)


if __name__ == "__main__":
    unittest.main()
