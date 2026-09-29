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
"""Single-process behavior tests for pure helpers in
``paddlefleet/transformer/moe/token_dispatcher.py``.

Scope. The AllGather EP dispatcher itself needs a real multi-rank process group
(AllGather + ReduceScatter across ranks), which cannot be validated in a single
process; per antipattern #13 we do NOT fake ``world_size`` + mock collectives to
claim distributed correctness. What we CAN validate here is the genuinely
single-process, deterministic logic that the dispatcher relies on:

  - ``_tokens_per_expert_histogram``: fixed-shape per-expert token counting that
    drops ``-1`` padding, checked against a hand-derived count computed with
    numpy (independent of the scatter-based implementation).
  - ``_sort_chunks_like_tokens``: split-along-axis-0 + reorder + concat, checked
    against a hand-built row permutation.
  - ``is_hybrid_ep_backend_selected``: dispatcher-type validation / backend
    gating control flow (raises on unknown type, returns False for non-hybridep
    backends, raises ImportError when hybridep is requested but its runtime is
    unavailable).

Expected values are derived by hand, never by re-invoking the code under test.
The local environment has no Paddle installed, so heavy imports are guarded and
the suite honestly skips (never fakes a pass) when the module cannot be
imported.
"""

import os
import sys
import unittest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import numpy as np
    import paddle

    from paddlefleet.transformer.moe import token_dispatcher as td

    _IMPORT_ERROR = None
except ImportError as exc:  # local env has no paddle / native EP deps
    np = None
    paddle = None
    td = None
    _IMPORT_ERROR = exc

_HAVE_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    "paddlefleet.transformer.moe.token_dispatcher not importable "
    f"(missing dependency: {_IMPORT_ERROR})"
    if not _HAVE_DEPS
    else ""
)


@unittest.skipUnless(_HAVE_DEPS, _SKIP_REASON)
class TokensPerExpertHistogramTest(unittest.TestCase):
    """``_tokens_per_expert_histogram`` counts real routed tokens per expert."""

    def _reference(self, indices_np, num_experts):
        # Independent count: keep valid routes (>= 0), tally each expert id.
        flat = indices_np.reshape(-1)
        counts = np.zeros(num_experts, dtype=np.int32)
        for v in flat:
            if v >= 0:
                counts[int(v)] += 1
        return counts

    def test_counts_drop_padding_and_land_on_correct_expert(self):
        # Distinguishable routing: experts 0 and 1 each receive two tokens,
        # expert 2 receives one, expert 3 receives none; -1 marks padding.
        indices_np = np.array(
            [[0, 1], [1, -1], [2, 0], [-1, -1]], dtype=np.int64
        )
        num_experts = 4
        indices = paddle.to_tensor(indices_np)

        out = td._tokens_per_expert_histogram(indices, num_experts)

        expected = self._reference(indices_np, num_experts)
        self.assertEqual(list(out.shape), [num_experts])
        self.assertEqual(out.dtype, paddle.int32)
        # Hand-derived: [2, 2, 1, 0]; a swap of expert columns or leaking
        # padding into a real expert would change these exact values.
        np.testing.assert_array_equal(out.numpy(), expected)
        np.testing.assert_array_equal(out.numpy(), np.array([2, 2, 1, 0]))

    def test_all_padding_yields_zero_histogram(self):
        indices_np = np.array([[-1, -1], [-1, -1]], dtype=np.int64)
        num_experts = 3
        indices = paddle.to_tensor(indices_np)

        out = td._tokens_per_expert_histogram(indices, num_experts)

        np.testing.assert_array_equal(out.numpy(), np.zeros(3, dtype=np.int32))

    def test_last_expert_column_is_counted(self):
        # Guards against an off-by-one that would drop the highest expert id
        # (e.g. sink column overlapping expert ``num_experts - 1``).
        indices_np = np.array([[3, 3], [3, 0]], dtype=np.int64)
        num_experts = 4
        indices = paddle.to_tensor(indices_np)

        out = td._tokens_per_expert_histogram(indices, num_experts)

        np.testing.assert_array_equal(
            out.numpy(), np.array([1, 0, 0, 3], dtype=np.int32)
        )


@unittest.skipUnless(_HAVE_DEPS, _SKIP_REASON)
class SortChunksLikeTokensTest(unittest.TestCase):
    """``_sort_chunks_like_tokens`` reorders variable-size row chunks."""

    def test_reorders_chunks_and_preserves_row_contents(self):
        # Unique per-row content so a wrong chunk boundary, wrong order, or a
        # dropped/duplicated row is observable.
        base = np.arange(12, dtype=np.float32).reshape([6, 2])
        inp = paddle.to_tensor(base)
        split_sizes = [2, 1, 3]  # chunks: rows[0:2], rows[2:3], rows[3:6]
        sorted_idxs = [2, 0, 1]  # -> chunk2, chunk0, chunk1

        out = td._sort_chunks_like_tokens(inp, split_sizes, sorted_idxs)

        # Hand-built permutation: rows 3,4,5 then 0,1 then 2.
        expected = np.concatenate([base[3:6], base[0:2], base[2:3]], axis=0)
        self.assertEqual(list(out.shape), [6, 2])
        np.testing.assert_array_equal(out.numpy(), expected)
        np.testing.assert_array_equal(
            out.numpy(),
            np.array(
                [[6, 7], [8, 9], [10, 11], [0, 1], [2, 3], [4, 5]],
                dtype=np.float32,
            ),
        )

    def test_identity_order_returns_original_rows(self):
        base = np.arange(8, dtype=np.float32).reshape([4, 2])
        inp = paddle.to_tensor(base)

        out = td._sort_chunks_like_tokens(inp, [1, 1, 2], [0, 1, 2])

        np.testing.assert_array_equal(out.numpy(), base)


@unittest.skipUnless(_HAVE_DEPS, _SKIP_REASON)
class HybridEPBackendSelectionTest(unittest.TestCase):
    """``is_hybrid_ep_backend_selected`` validates type and gates the backend."""

    def test_unknown_dispatcher_type_raises_value_error(self):
        with self.assertRaises(ValueError):
            td.is_hybrid_ep_backend_selected("not_a_real_backend")

    def test_non_hybrid_backends_return_false(self):
        for name in ("allgather", "alltoall", "deepep", "moonep", "ringmoe"):
            self.assertFalse(
                td.is_hybrid_ep_backend_selected(name),
                msg=f"{name} must not select the hybrid EP backend",
            )

    def test_default_none_resolves_to_deepep_and_returns_false(self):
        # None defaults to "deepep" internally, which is a valid non-hybrid type.
        self.assertFalse(td.is_hybrid_ep_backend_selected(None))

    def test_hybridep_requires_available_runtime(self):
        # HAVE_HYBRID_EP is a module-level availability flag derived from an
        # optional native package; toggling it is stubbing a genuine
        # not-under-test collaborator, not the function under test. Restore it
        # afterward to avoid leaking state into other tests.
        original = td.HAVE_HYBRID_EP
        self.addCleanup(setattr, td, "HAVE_HYBRID_EP", original)

        td.HAVE_HYBRID_EP = False
        with self.assertRaises(ImportError):
            td.is_hybrid_ep_backend_selected("hybridep")

        td.HAVE_HYBRID_EP = True
        self.assertTrue(td.is_hybrid_ep_backend_selected("hybridep"))


if __name__ == "__main__":
    unittest.main()
