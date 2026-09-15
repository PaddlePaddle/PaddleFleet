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
"""Behavior tests for ``paddlefleet.transformer.moe.token_dispatcher``.

Token dispatch as a whole is inherently multi-rank (it exchanges tokens across
expert-parallel ranks through collective communication); those paths are *not*
exercised here because a single process cannot observe cross-rank behavior and
faking ``world_size`` plus mocked collectives would prove nothing.  Instead each
test below targets a piece of *single-process-observable* routing/permute logic
whose contract is fully determined locally, and every expected value is
hand-derived from that contract independently of the implementation:

* ``_sort_chunks_like_tokens`` -- splits a tensor into variable-size chunks and
  reconcatenates them in a given order; verified by exact reordered row content.
* ``is_hybrid_ep_backend_selected`` -- backend-name validation / selection:
  rejects unknown names, returns ``False`` for every non-hybridep backend, and
  the ``hybridep`` branch depends only on the locally cached availability flag.
* ``_try_setup_router_topk_metadata`` -- reshapes router-provided top-k weights
  and indices onto a manager and reports whether it consumed them.
* ``_DeepEPManager._indices_to_multihot`` -- turns ``[num_tokens, topk]`` local
  expert indices (``-1`` == masked) into a multihot map over local experts and
  scatters the matching probabilities; the ``-1`` mask must not leak a weight.
* ``_HybridEPManager._indices_to_dense_metadata`` -- turns top-k indices and
  weights into a dense ``[num_tokens, num_experts]`` routing map and summed
  probability map, again dropping masked (``-1``) entries.

Paddle (and the paddlefleet package) are required to import the module under
test.  When unavailable the whole suite is skipped with an honest reason rather
than reported as passing.  ``fused_dispatch`` / ``HAVE_HYBRID_EP`` are patched
only to bypass optional-backend availability guards in the constructors; they
are collaborators of construction, not the logic under test.
"""

import os
import sys
import types
import unittest
from unittest import mock

import numpy as np

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

try:
    import paddle

    from paddlefleet.transformer.moe import token_dispatcher as td

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - environment-dependent
    paddle = None
    td = None
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    "paddle / paddlefleet.transformer.moe.token_dispatcher unavailable: "
    f"{_IMPORT_ERROR}"
)


@unittest.skipUnless(td is not None, _SKIP_REASON)
class TestSortChunksLikeTokens(unittest.TestCase):
    """``_sort_chunks_like_tokens`` reorders variable-size row chunks."""

    def test_reorders_variable_chunks_by_content(self):
        # 6 rows, each row tagged by its index so a wrong chunk order or a
        # wrong split boundary changes the observed content.
        rows = paddle.arange(6 * 3, dtype="float32").reshape([6, 3])
        # split into chunks of 2, 1, 3 rows -> chunk0=rows0-1, chunk1=row2,
        # chunk2=rows3-5. Reorder as [chunk1, chunk2, chunk0].
        out = td._sort_chunks_like_tokens(rows, [2, 1, 3], [1, 2, 0])

        expected_row_order = [2, 3, 4, 5, 0, 1]
        expected = rows.numpy()[expected_row_order]
        np.testing.assert_array_equal(out.numpy(), expected)
        self.assertEqual(out.shape, [6, 3])

    def test_identity_order_is_reconstruction(self):
        rows = paddle.arange(4 * 2, dtype="float32").reshape([4, 2])
        out = td._sort_chunks_like_tokens(rows, [1, 3], [0, 1])
        np.testing.assert_array_equal(out.numpy(), rows.numpy())


@unittest.skipUnless(td is not None, _SKIP_REASON)
class TestIsHybridEpBackendSelected(unittest.TestCase):
    """Backend-name validation and hybridep selection.

    This is pure local branching; no process group or collective is involved.
    """

    def test_unknown_backend_rejected(self):
        with self.assertRaises(ValueError):
            td.is_hybrid_ep_backend_selected("not_a_backend")

    def test_none_defaults_to_deepep_not_selected(self):
        # ``None`` means "use the default", which is deepep -> not hybridep.
        self.assertIs(td.is_hybrid_ep_backend_selected(None), False)

    def test_every_non_hybridep_backend_returns_false(self):
        for name in ("allgather", "alltoall", "deepep", "moonep", "ringmoe"):
            self.assertIs(
                td.is_hybrid_ep_backend_selected(name),
                False,
                msg=f"{name} must not be reported as hybridep",
            )

    def test_hybridep_requires_available_runtime(self):
        # When the runtime is unavailable the hybridep branch must fail loudly
        # instead of silently selecting an unusable backend.
        with mock.patch.object(td, "HAVE_HYBRID_EP", False):
            with self.assertRaises(ImportError):
                td.is_hybrid_ep_backend_selected("hybridep")

    def test_hybridep_selected_when_runtime_available(self):
        with mock.patch.object(td, "HAVE_HYBRID_EP", True):
            self.assertIs(td.is_hybrid_ep_backend_selected("hybridep"), True)


@unittest.skipUnless(td is not None, _SKIP_REASON)
class TestTrySetupRouterTopkMetadata(unittest.TestCase):
    """Router-provided top-k weights/indices are reshaped onto the manager."""

    def test_consumes_and_reshapes_router_output(self):
        # A bare data holder standing in for the manager: the function under
        # test writes onto it and we observe the result.
        manager = types.SimpleNamespace(router_topk=2)
        num_tokens = 3
        # Flat inputs whose reshape target [3, 2] is content-distinguishable.
        weights = paddle.to_tensor(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype="float32"
        )
        indices = paddle.to_tensor([10, 11, 12, 13, 14, 15], dtype="int64")

        consumed = td._try_setup_router_topk_metadata(
            manager, num_tokens, weights, indices
        )

        self.assertIs(consumed, True)
        self.assertEqual(manager.token_probs.shape, [3, 2])
        self.assertEqual(manager.token_indices.shape, [3, 2])
        np.testing.assert_array_equal(
            manager.token_probs.numpy(),
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
        )
        np.testing.assert_array_equal(
            manager.token_indices.numpy(),
            [[10, 11], [12, 13], [14, 15]],
        )
        # Indices are metadata, not a differentiable tensor.
        self.assertTrue(manager.token_indices.stop_gradient)

    def test_returns_false_without_mutating_when_indices_missing(self):
        manager = types.SimpleNamespace(router_topk=2)
        weights = paddle.to_tensor([0.0, 1.0], dtype="float32")

        consumed = td._try_setup_router_topk_metadata(manager, 1, weights, None)

        self.assertIs(consumed, False)
        self.assertFalse(hasattr(manager, "token_probs"))
        self.assertFalse(hasattr(manager, "token_indices"))

    def test_returns_false_when_weights_missing(self):
        manager = types.SimpleNamespace(router_topk=2)
        indices = paddle.to_tensor([0, 1], dtype="int64")

        consumed = td._try_setup_router_topk_metadata(manager, 1, None, indices)

        self.assertIs(consumed, False)
        self.assertFalse(hasattr(manager, "token_indices"))


@unittest.skipUnless(td is not None, _SKIP_REASON)
class TestDeepEPIndicesToMultihot(unittest.TestCase):
    """``_DeepEPManager._indices_to_multihot`` scatters topk -> multihot."""

    def _make_manager(self, num_local_experts):
        # The DeepEP manager constructor refuses to build when the fused
        # kernels are absent; patch the availability sentinel to a non-None
        # placeholder so construction proceeds. ``_indices_to_multihot`` never
        # touches ``fused_dispatch``; it only reads ``num_local_experts``.
        with mock.patch.object(td, "fused_dispatch", object()):
            return td._DeepEPManager(
                group=None,
                router_topk=2,
                num_experts=8,
                num_local_experts=num_local_experts,
            )

    def test_masked_entry_does_not_leak_and_probs_land_on_expert(self):
        manager = self._make_manager(num_local_experts=4)
        # Row1's second slot is masked (-1); its weight (0.1) must be dropped.
        indices = paddle.to_tensor([[0, 2], [1, -1], [2, 3]], dtype="int64")
        probs = paddle.to_tensor(
            [[0.5, 0.3], [0.9, 0.1], [0.4, 0.6]], dtype="float32"
        )

        routing_map, multihot_probs = manager._indices_to_multihot(
            indices, probs
        )

        expected_map = np.array(
            [
                [True, False, True, False],
                [False, True, False, False],
                [False, False, True, True],
            ]
        )
        expected_probs = np.array(
            [
                [0.5, 0.0, 0.3, 0.0],
                [0.0, 0.9, 0.0, 0.0],
                [0.0, 0.0, 0.4, 0.6],
            ],
            dtype=np.float32,
        )
        self.assertEqual(routing_map.dtype, paddle.bool)
        np.testing.assert_array_equal(
            routing_map.numpy().astype(bool), expected_map
        )
        np.testing.assert_allclose(
            multihot_probs.numpy(), expected_probs, atol=1e-6
        )

    def test_fully_masked_row_is_all_false(self):
        manager = self._make_manager(num_local_experts=3)
        indices = paddle.to_tensor([[-1, -1], [0, 2]], dtype="int64")
        probs = paddle.to_tensor([[0.7, 0.2], [0.6, 0.4]], dtype="float32")

        routing_map, multihot_probs = manager._indices_to_multihot(
            indices, probs
        )

        np.testing.assert_array_equal(
            routing_map.numpy().astype(bool),
            np.array([[False, False, False], [True, False, True]]),
        )
        np.testing.assert_allclose(
            multihot_probs.numpy(),
            np.array([[0.0, 0.0, 0.0], [0.6, 0.0, 0.4]], dtype=np.float32),
            atol=1e-6,
        )


@unittest.skipUnless(td is not None, _SKIP_REASON)
class TestHybridEPIndicesToDenseMetadata(unittest.TestCase):
    """``_HybridEPManager._indices_to_dense_metadata`` builds a dense map."""

    def _make_manager(self, num_experts):
        # Constructor requires the HybridEP runtime flag; flip it only to allow
        # construction. ``_indices_to_dense_metadata`` reads ``num_experts``
        # and does no communication.
        with mock.patch.object(td, "HAVE_HYBRID_EP", True):
            return td._HybridEPManager(
                group=None,
                router_topk=2,
                num_experts=num_experts,
                num_local_experts=2,
            )

    def test_masked_indices_dropped_and_weights_placed(self):
        manager = self._make_manager(num_experts=4)
        token_indices = paddle.to_tensor(
            [[0, 2], [1, -1], [2, 3]], dtype="int64"
        )
        token_weights = paddle.to_tensor(
            [[0.5, 0.3], [0.9, 0.1], [0.4, 0.6]], dtype="float32"
        )

        routing_map, probs = manager._indices_to_dense_metadata(
            token_indices, token_weights
        )

        expected_map = np.array(
            [
                [True, False, True, False],
                [False, True, False, False],
                [False, False, True, True],
            ]
        )
        expected_probs = np.array(
            [
                [0.5, 0.0, 0.3, 0.0],
                [0.0, 0.9, 0.0, 0.0],
                [0.0, 0.0, 0.4, 0.6],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(
            routing_map.numpy().astype(bool), expected_map
        )
        self.assertEqual(probs.dtype, paddle.float32)
        np.testing.assert_allclose(probs.numpy(), expected_probs, atol=1e-6)

    def test_duplicate_expert_sums_weights(self):
        # Both topk slots point at the same local expert: the dense map marks
        # it once (bool) while the probability map sums the two weights.
        manager = self._make_manager(num_experts=3)
        token_indices = paddle.to_tensor([[1, 1]], dtype="int64")
        token_weights = paddle.to_tensor([[0.3, 0.4]], dtype="float32")

        routing_map, probs = manager._indices_to_dense_metadata(
            token_indices, token_weights
        )

        np.testing.assert_array_equal(
            routing_map.numpy().astype(bool),
            np.array([[False, True, False]]),
        )
        np.testing.assert_allclose(
            probs.numpy(),
            np.array([[0.0, 0.7, 0.0]], dtype=np.float32),
            atol=1e-6,
        )

    def test_none_weights_yields_map_only(self):
        manager = self._make_manager(num_experts=3)
        token_indices = paddle.to_tensor([[0, 2]], dtype="int64")

        routing_map, probs = manager._indices_to_dense_metadata(
            token_indices, None
        )

        self.assertIsNone(probs)
        np.testing.assert_array_equal(
            routing_map.numpy().astype(bool),
            np.array([[True, False, True]]),
        )


if __name__ == "__main__":
    unittest.main()
