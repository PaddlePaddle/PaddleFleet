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

"""Behavior tests for ``paddlefleet.overlap_context_parallel``.

The overlapped FlashMask context-parallel layer builds the gathered-KV mask
order and enforces the topology restrictions that the FA-4 in-kernel all-gather
assumes. Those pieces are ordinary CPU logic and are what these tests exercise
with independently hand-derived expectations:

  - ``traversal_rank``: the owner rank visited at each traversal position, for
    both the circular (single-node) and the hierarchical (multi-node) schedule;
  - ``block_order``: the natural-block permutation that folds the balance layout
    and the traversal into one index, checked against a from-scratch oracle
    built out of the documented block pairing;
  - ``gathered_kv_order``: the mask permutation itself, against a slice-and-
    concat oracle over per-block distinguishable content;
  - ``localize_mask``: mode routing, the exact chunk ids it derives, and its
    rejection of unsupported modes;
  - ``_require_contiguous_cp_ranks`` / ``_require_eight_gpu_node``: the topology
    guards;
  - ``overlap_flashmask_attention_cp``: the feature-rejection guards that run
    before the kernel is ever reached.

The FA-4 kernel and the CP process group are NOT exercised here; their real
numerics need an SM100 device plus a paddlefleet_ops build with the overlap
interface, and are out of scope for a CPU single-card test. Importing the module
still requires paddle, so every test honestly skips when paddle is absent rather
than reporting a false pass.
"""

import os
import sys
import unittest
from unittest import mock

# Test the in-tree dev version: src/ ahead of any installed package.
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

try:
    import paddle

    from paddlefleet import (
        context_parallel_utils as cpu,
        overlap_context_parallel as ocp,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # honest skip reason, not a swallowed failure
    paddle = None
    cpu = None
    ocp = None
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    "" if _HAS_DEPS else f"paddle/paddlefleet unavailable: {_IMPORT_ERROR!r}"
)

DUALCHUNK = "dualchunk_allgather_overlap"
CONTIGUOUS = "contiguous_allgather_overlap"


class _Group:
    """Minimal CP-group stand-in; the layer reads only these attributes."""

    def __init__(self, rank=0, world_size=1, ranks=None):
        self.rank = rank
        self.world_size = world_size
        if ranks is not None:
            self.ranks = ranks


def independent_traversal(rank, cp_size, gpus_per_node):
    """Owner rank visited at each traversal position, derived from the docstring.

    This is a from-scratch reconstruction, deliberately NOT the module's divmod
    arithmetic: circular is "start local, step forward one rank at a time"; the
    hierarchical schedule walks the local congruence group first (same intra-node
    slot, successive nodes), then each further intra-node slot the same way.
    """
    if gpus_per_node == 0:
        return [(rank + pos) % cp_size for pos in range(cp_size)]
    num_nodes = cp_size // gpus_per_node
    my_slot = rank % gpus_per_node
    my_node = rank // gpus_per_node
    order = []
    for slot_step in range(gpus_per_node):
        slot = (my_slot + slot_step) % gpus_per_node
        for node_step in range(num_nodes):
            node = (my_node + node_step) % num_nodes
            order.append(slot + node * gpus_per_node)
    return order


def independent_block_order(cp_size, rank, backward, mode):
    """Expected natural-block permutation, built independently of the module.

    Traversal direction: backward starts on the local chunk (position 0);
    forward leaves the local chunk last (circular rotated by one, hierarchical
    reversed). Block pairing per owner comes from the documented balance layout:
    DualChunkSwap owns the mirrored pair ``(r, 2*cp-1-r)``, contiguous owns the
    adjacent pair ``(2r, 2r+1)``.
    """
    gpus_per_node = 8 if cp_size > 8 else 0
    traversal = independent_traversal(rank, cp_size, gpus_per_node)
    if backward:
        positions = list(range(cp_size))
    elif gpus_per_node:
        positions = list(range(cp_size - 1, -1, -1))
    else:
        positions = [*list(range(1, cp_size)), 0]
    order = []
    for pos in positions:
        owner = traversal[pos]
        if mode == CONTIGUOUS:
            order += [2 * owner, 2 * owner + 1]
        else:
            order += [owner, 2 * cp_size - 1 - owner]
    return order


def make_block_labeled_mask(cp_size, block_len=3, num_vecs=2):
    """A mask whose ``b``-th natural key block carries a unique per-vec value."""
    n_blocks = 2 * cp_size
    seqlen = n_blocks * block_len
    values = paddle.zeros([1, 1, seqlen, num_vecs], dtype="int32")
    rows = []
    for block in range(n_blocks):
        for _ in range(block_len):
            rows.append([block + 1000 * vec for vec in range(num_vecs)])
    values = paddle.to_tensor([rows], dtype="int32").reshape(
        [1, 1, seqlen, num_vecs]
    )
    return values, block_len


def slice_concat_oracle(mask, order, block_len):
    """Reassemble ``mask`` by concatenating whole key blocks in ``order``.

    A slice-and-concat oracle independent of the reshape/index_select the layer
    uses, so a wrong axis or a scrambled block would diverge here.
    """
    blocks = [mask[:, :, b * block_len : (b + 1) * block_len, :] for b in order]
    return paddle.concat(blocks, axis=2)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestTraversalRankCircular(unittest.TestCase):
    """Single-node schedule: visit the local rank, then step forward circularly."""

    def test_hand_enumerated_orders(self):
        # Fully hand-listed expected sequences, no formula reuse.
        cases = {
            (4, 0): [0, 1, 2, 3],
            (4, 1): [1, 2, 3, 0],
            (4, 3): [3, 0, 1, 2],
            (8, 5): [5, 6, 7, 0, 1, 2, 3, 4],
        }
        for (cp_size, rank), expected in cases.items():
            got = [
                ocp.traversal_rank(pos, rank, cp_size, 0)
                for pos in range(cp_size)
            ]
            self.assertEqual(got, expected, f"cp_size={cp_size} rank={rank}")

    def test_is_a_permutation_starting_at_local(self):
        for cp_size in (1, 2, 4, 8):
            for rank in range(cp_size):
                visited = [
                    ocp.traversal_rank(pos, rank, cp_size, 0)
                    for pos in range(cp_size)
                ]
                self.assertEqual(visited[0], rank)
                self.assertEqual(sorted(visited), list(range(cp_size)))


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestTraversalRankHierarchical(unittest.TestCase):
    """Multi-node schedule fixed to 8-GPU nodes (engages when cp_size > 8)."""

    CP_SIZE = 16
    GPN = 8

    def test_hand_enumerated_orders(self):
        # Worked out by hand from the documented schedule, not the module math.
        expected_by_rank = {
            0: [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15],
            3: [3, 11, 4, 12, 5, 13, 6, 14, 7, 15, 0, 8, 1, 9, 2, 10],
        }
        for rank, expected in expected_by_rank.items():
            got = [
                ocp.traversal_rank(pos, rank, self.CP_SIZE, self.GPN)
                for pos in range(self.CP_SIZE)
            ]
            self.assertEqual(got, expected, f"rank={rank}")

    def test_matches_independent_schedule_for_every_rank(self):
        for rank in range(self.CP_SIZE):
            got = [
                ocp.traversal_rank(pos, rank, self.CP_SIZE, self.GPN)
                for pos in range(self.CP_SIZE)
            ]
            self.assertEqual(
                got, independent_traversal(rank, self.CP_SIZE, self.GPN)
            )

    def test_congruence_group_visited_first(self):
        num_nodes = self.CP_SIZE // self.GPN
        for rank in range(self.CP_SIZE):
            head = [
                ocp.traversal_rank(pos, rank, self.CP_SIZE, self.GPN)
                for pos in range(num_nodes)
            ]
            # Same intra-node slot, one distinct peer per node.
            self.assertEqual(
                {peer % self.GPN for peer in head}, {rank % self.GPN}
            )
            self.assertEqual(len(set(head)), num_nodes)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestBlockOrder(unittest.TestCase):
    """The natural-block permutation folding balance layout and traversal."""

    def setUp(self):
        # block_order caches per key in a module global; isolate every test.
        ocp.BLOCK_ORDER_CACHE.clear()
        self.addCleanup(ocp.BLOCK_ORDER_CACHE.clear)

    def test_hand_enumerated_small_cases(self):
        # Derived by hand from the pairing + traversal, listed explicitly.
        cases = {
            (2, 0, True, DUALCHUNK): [0, 3, 1, 2],
            (2, 0, False, DUALCHUNK): [1, 2, 0, 3],
            (2, 0, True, CONTIGUOUS): [0, 1, 2, 3],
            (2, 0, False, CONTIGUOUS): [2, 3, 0, 1],
        }
        for (cp_size, rank, backward, mode), expected in cases.items():
            got = ocp.block_order(cp_size, rank, backward, mode).tolist()
            self.assertEqual(
                got, expected, f"{cp_size} {rank} {backward} {mode}"
            )

    def test_matches_independent_oracle(self):
        for mode in (DUALCHUNK, CONTIGUOUS):
            for cp_size in (1, 2, 4, 8, 16):
                for rank in range(cp_size):
                    for backward in (False, True):
                        got = ocp.block_order(
                            cp_size, rank, backward, mode
                        ).tolist()
                        self.assertEqual(
                            got,
                            independent_block_order(
                                cp_size, rank, backward, mode
                            ),
                            f"{mode} cp={cp_size} rank={rank} bwd={backward}",
                        )

    def test_backward_starts_local_forward_ends_local(self):
        cp_size, rank = 4, 1
        local_pair = [rank, 2 * cp_size - 1 - rank]  # DualChunk local blocks
        backward = ocp.block_order(cp_size, rank, True, DUALCHUNK).tolist()
        forward = ocp.block_order(cp_size, rank, False, DUALCHUNK).tolist()
        self.assertEqual(backward[:2], local_pair)
        self.assertEqual(forward[-2:], local_pair)
        self.assertNotEqual(backward, forward)

    def test_cached_per_key(self):
        first = ocp.block_order(4, 1, False, DUALCHUNK)
        self.assertIs(first, ocp.block_order(4, 1, False, DUALCHUNK))
        for key in (
            (4, 2, False, DUALCHUNK),
            (4, 1, True, DUALCHUNK),
            (4, 1, False, CONTIGUOUS),
        ):
            self.assertIsNot(first, ocp.block_order(*key))


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestGatheredKvOrder(unittest.TestCase):
    """The mask key-axis permutation, against a slice-and-concat oracle."""

    def setUp(self):
        ocp.BLOCK_ORDER_CACHE.clear()
        self.addCleanup(ocp.BLOCK_ORDER_CACHE.clear)

    def _check(self, cp_size, rank, backward, mode):
        mask, block_len = make_block_labeled_mask(cp_size)
        group = _Group(rank=rank, world_size=cp_size)
        got = ocp.gathered_kv_order(mask, group, backward, mode)
        order = independent_block_order(cp_size, rank, backward, mode)
        expected = slice_concat_oracle(mask, order, block_len)
        self.assertEqual(list(got.shape), list(mask.shape))
        self.assertTrue(
            bool((got == expected).all()),
            f"{mode} cp={cp_size} rank={rank} bwd={backward}\n"
            f"got={got.numpy()[0, 0, :, 0].tolist()}\n"
            f"exp={expected.numpy()[0, 0, :, 0].tolist()}",
        )

    def test_dualchunk_matches_oracle(self):
        for cp_size in (1, 2, 4):
            for rank in range(cp_size):
                for backward in (False, True):
                    self._check(cp_size, rank, backward, DUALCHUNK)

    def test_contiguous_matches_oracle(self):
        for cp_size in (1, 2, 4):
            for rank in range(cp_size):
                for backward in (False, True):
                    self._check(cp_size, rank, backward, CONTIGUOUS)

    def test_second_vector_channel_travels_with_its_block(self):
        # vec 1 carries block+1000; a vec-axis scramble would break this.
        mask, block_len = make_block_labeled_mask(2)
        group = _Group(rank=0, world_size=2)
        got = ocp.gathered_kv_order(mask, group, True, DUALCHUNK)
        first_block = got[0, 0, :block_len, :].numpy()
        # Backward local block for rank 0 is natural block 0 -> values [0, 1000].
        self.assertEqual(first_block[0].tolist(), [0, 1000])


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestLocalizeMask(unittest.TestCase):
    """Mode routing and the exact chunk ids localize_mask derives."""

    def test_dualchunk_derives_mirrored_chunk_ids(self):
        captured = {}

        def spy(startend_row_indices, **kwargs):
            captured["mask"] = startend_row_indices
            captured.update(kwargs)
            return "dual-marker"

        mask = paddle.zeros([1, 1, 16, 2], dtype="int32")
        group = _Group(rank=1, world_size=4)
        with mock.patch.object(ocp, "preprocess_index_dual_chunks", spy):
            out = ocp.localize_mask(mask, 16, group, DUALCHUNK)
        self.assertEqual(out, "dual-marker")
        self.assertIs(captured["mask"], mask)
        self.assertEqual(captured["chunk_id_first"], 1)
        # second chunk id = 2 * world_size - rank - 1 = 2*4 - 1 - 1 = 6
        self.assertEqual(captured["chunk_id_second"], 6)
        self.assertEqual(captured["seq_blocksize"], 8)
        self.assertEqual(captured["max_seqlen_q"], 8)

    def test_contiguous_rebases_on_local_chunk(self):
        captured = {}

        def spy(startend_row_indices, **kwargs):
            captured["mask"] = startend_row_indices
            captured.update(kwargs)
            return "contig-marker"

        mask = paddle.zeros([1, 1, 16, 2], dtype="int32")
        group = _Group(rank=3, world_size=4)
        with mock.patch.object(ocp, "preprocess_index", spy):
            out = ocp.localize_mask(mask, 16, group, CONTIGUOUS)
        self.assertEqual(out, "contig-marker")
        self.assertEqual(captured["chunk_id"], 3)
        self.assertEqual(captured["seq_blocksize"], 16)
        self.assertEqual(captured["max_seqlen_q"], 16)

    def test_unsupported_mode_rejected(self):
        mask = paddle.zeros([1, 1, 8, 2], dtype="int32")
        with self.assertRaises(ValueError):
            ocp.localize_mask(mask, 4, _Group(0, 2), "contiguous_a2a_overlap")


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestTopologyGuards(unittest.TestCase):
    """The contiguous-rank and 8-GPU-node restrictions the backend assumes."""

    def test_single_node_skips_contiguity_check(self):
        # cp_size <= 8 is circular, so strided ranks are irrelevant and allowed.
        ocp._require_contiguous_cp_ranks(
            _Group(world_size=8, ranks=list(range(0, 16, 2)))
        )

    def test_contiguous_multinode_ranks_pass(self):
        ocp._require_contiguous_cp_ranks(
            _Group(world_size=16, ranks=list(range(16)))
        )
        ocp._require_contiguous_cp_ranks(
            _Group(world_size=16, ranks=list(range(8, 24)))
        )

    def test_strided_multinode_ranks_rejected(self):
        with self.assertRaises(ValueError):
            ocp._require_contiguous_cp_ranks(
                _Group(world_size=16, ranks=list(range(0, 32, 2)))
            )

    def _run_node_guard(self, cp_size, env):
        with mock.patch.dict(os.environ, env, clear=True):
            ocp._require_eight_gpu_node(cp_size)

    def test_eight_gpu_node_always_passes(self):
        self._run_node_guard(16, {"PADDLE_LOCAL_SIZE": "8"})

    def test_single_node_passes_on_any_size(self):
        self._run_node_guard(4, {"PADDLE_LOCAL_SIZE": "4"})

    def test_no_topology_signal_is_not_rejected(self):
        self._run_node_guard(16, {})

    def test_multinode_non_eight_is_rejected(self):
        with self.assertRaises(ValueError):
            self._run_node_guard(8, {"PADDLE_LOCAL_SIZE": "4"})
        with self.assertRaises(ValueError):
            self._run_node_guard(16, {"PADDLE_LOCAL_SIZE": "4"})


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestOverlapEntryGuards(unittest.TestCase):
    """Feature-rejection guards that run before the FA-4 kernel is reached.

    OVERLAP_SUPPORTED is a capability flag, not code under test; patching it True
    lets the guard branches run without an SM100 device or an ops build. The
    kernel apply is never reached by any of these cases.
    """

    @staticmethod
    def _query(seqlen=8):
        return paddle.zeros([1, seqlen, 1, 8], dtype="bfloat16")

    def _call(self, seqlen=8, **kwargs):
        q = self._query(seqlen)
        mask = paddle.zeros([1, 1, seqlen, 2], dtype="int32")
        return ocp.overlap_flashmask_attention_cp(q, q, q, mask, **kwargs)

    def test_requires_capable_build(self):
        with mock.patch.object(ocp, "OVERLAP_SUPPORTED", False):  # noqa: SIM117
            with self.assertRaises(AssertionError):
                self._call()

    def test_dropout_not_implemented(self):
        with mock.patch.object(ocp, "OVERLAP_SUPPORTED", True):  # noqa: SIM117
            with self.assertRaises(NotImplementedError):
                self._call(dropout=0.1)

    def test_causal_not_implemented(self):
        with mock.patch.object(ocp, "OVERLAP_SUPPORTED", True):  # noqa: SIM117
            with self.assertRaises(NotImplementedError):
                self._call(causal=True)

    def test_fixed_seed_offset_not_implemented(self):
        with mock.patch.object(ocp, "OVERLAP_SUPPORTED", True):  # noqa: SIM117
            with self.assertRaises(NotImplementedError):
                self._call(fixed_seed_offset=paddle.zeros([1], dtype="int64"))

    def test_odd_local_sequence_length_rejected(self):
        with mock.patch.object(ocp, "OVERLAP_SUPPORTED", True):  # noqa: SIM117
            with self.assertRaises(AssertionError):
                self._call(seqlen=7)


if __name__ == "__main__":
    unittest.main()
