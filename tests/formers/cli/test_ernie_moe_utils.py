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

"""CPU-observable behavior tests for the ERNIE MoE token-dispatcher utils.

Target production code:
``src/paddlefleet/cli/train/ernie_pretrain/models/moe/token_dispatcher/moe_utils.py``

Scope (unit-test-rules.md "模型层 / MoE" and "配置与运行基础设施"):
  * ``permute`` -- gather-based token reordering. The oracle compares the full
    reordered *content* against a hand-written expectation, so a wrong gather
    axis or a dropped row is caught (checking shape alone would not).
  * ``unpermute`` -- scatter-based restore. Verified for (a) exact restored
    content with a non-identity index map, (b) the ``overwrite=False`` ACCUMULATE
    semantics when two source rows land on the same destination row, and (c) the
    probability-weighting path, where ``prob_permuted_indices`` selects *distinct*
    probabilities so a wrong index map changes the numbers.
  * ``inplace_offload`` / ``inplace_offload_if_needed`` -- the no-grad early-exit
    guard and the ``memory_size >= threshold`` decision. These are observed via
    the warning contract (warn iff the offload branch is entered) and by proving
    a CPU tensor is left byte-for-byte unchanged.
  * ``topk_to_permuted_indices`` / ``_single`` -- per-expert index construction
    ``pos where routemap==expert`` grouped by expert, with ``token = pos // topk``.
    Expected index lists are derived by hand from a fixed routemap.
  * ``UnZipNode`` / ``ZipNode`` -- constructor state and ``reset_status``: the
    dispatcher reference identity, the default/explicit name, and that
    ``reset_status`` clears the cached fields back to ``None``.

The ``forward``/``backward`` methods call ``paddle.nn.functional.moe_permute`` /
``moe_unpermute`` (GPU custom ops) and are NOT exercised here -- the local
environment has no GPU and no paddle. The whole module imports ``paddle`` at load
time, so when paddle is absent every test skips with an honest reason instead of
reporting a fake pass. All expected values below are hand-derived and independent
of the production implementation.
"""

import unittest
import warnings

try:
    import numpy as np
    import paddle

    from paddlefleet.cli.train.ernie_pretrain.models.moe.token_dispatcher.moe_utils import (
        UnZipNode,
        ZipNode,
        inplace_offload,
        inplace_offload_if_needed,
        permute,
        topk_to_permuted_indices,
        topk_to_permuted_indices_single,
        unpermute,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle (or a paddle-dependent import) is missing
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    "moe_utils imports paddle at module load; paddle is not importable in this "
    f"environment ({_IMPORT_ERROR})"
)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestPermute(unittest.TestCase):
    """``permute`` reorders rows of ``tokens`` by ``token_permuted_indices``."""

    def test_permute_reorders_rows_by_index(self):
        # Rows carry distinct content so a wrong gather is visible, not just a
        # shape change. index [2, 0, 1] -> output rows [tokens[2], tokens[0],
        # tokens[1]], derived by hand from the gather contract.
        tokens = paddle.to_tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype="float32"
        )
        indices = paddle.to_tensor([2, 0, 1], dtype="int64")
        out = permute(tokens, indices)
        expected = np.array(
            [[5.0, 6.0], [1.0, 2.0], [3.0, 4.0]], dtype="float32"
        )
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_permute_can_repeat_and_select_rows(self):
        # A gather may repeat and skip rows; index [1, 1, 0] duplicates row 1
        # and drops row 2. This distinguishes a real gather from an identity /
        # permutation-only implementation.
        tokens = paddle.to_tensor(
            [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]], dtype="float32"
        )
        indices = paddle.to_tensor([1, 1, 0], dtype="int64")
        out = permute(tokens, indices)
        expected = np.array(
            [[20.0, 21.0], [20.0, 21.0], [10.0, 11.0]], dtype="float32"
        )
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_permute_drop_and_pad_unsupported(self):
        tokens = paddle.to_tensor([[1.0, 2.0]], dtype="float32")
        indices = paddle.to_tensor([0], dtype="int64")
        with self.assertRaises(AssertionError):
            permute(tokens, indices, drop_and_pad=True)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestUnpermute(unittest.TestCase):
    """``unpermute`` scatter-restores rows and optionally weights by probs."""

    def test_restores_rows_to_original_positions(self):
        # scatter_(overwrite=False) writes permuted_tokens[i] into
        # output[token_permuted_indices[i]]. With unique indices this is a plain
        # placement. Hand-derived: idx [1, 2, 0] sends the three rows to output
        # rows 1, 2, 0 respectively.
        permuted = paddle.to_tensor(
            [[5.0, 6.0], [1.0, 2.0], [3.0, 4.0]], dtype="float32"
        )
        token_idx = paddle.to_tensor([1, 2, 0], dtype="int64")
        prob_idx = paddle.to_tensor([0, 1, 2], dtype="int64")
        out = unpermute(permuted, token_idx, prob_idx, [3, 2])
        expected = np.array(
            [[3.0, 4.0], [5.0, 6.0], [1.0, 2.0]], dtype="float32"
        )
        # Content AND placement, not just shape [3, 2].
        self.assertEqual(out.shape, [3, 2])
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_duplicate_targets_accumulate(self):
        # The load-bearing contract is overwrite=False == reduce-add. Two source
        # rows target output row 0, so they must SUM; an overwrite=True impl
        # would instead keep only the last ([2, 2]) and fail here.
        permuted = paddle.to_tensor(
            [[1.0, 1.0], [2.0, 2.0], [10.0, 10.0]], dtype="float32"
        )
        token_idx = paddle.to_tensor([0, 0, 1], dtype="int64")
        prob_idx = paddle.to_tensor([0, 1, 2], dtype="int64")
        out = unpermute(permuted, token_idx, prob_idx, [2, 2])
        expected = np.array([[3.0, 3.0], [10.0, 10.0]], dtype="float32")
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_probs_weighting_uses_prob_permuted_indices(self):
        # probs is gathered by prob_permuted_indices (NOT token indices), then
        # each permuted row is scaled by its gathered scalar before scatter.
        # probs flat = [10, 20, 30, 40]; prob_idx [3, 1] -> gathered [40, 20].
        # row0 [1, 2]*40 = [40, 80] -> output row token_idx[0]=1
        # row1 [3, 4]*20 = [60, 80] -> output row token_idx[1]=0
        permuted = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        token_idx = paddle.to_tensor([1, 0], dtype="int64")
        prob_idx = paddle.to_tensor([3, 1], dtype="int64")
        probs = paddle.to_tensor([10.0, 20.0, 30.0, 40.0], dtype="float32")
        out = unpermute(permuted, token_idx, prob_idx, [2, 2], probs=probs)
        expected = np.array([[60.0, 80.0], [40.0, 80.0]], dtype="float32")
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_drop_and_pad_unsupported(self):
        tokens = paddle.to_tensor([[1.0, 2.0]], dtype="float32")
        idx = paddle.to_tensor([0], dtype="int64")
        with self.assertRaises(AssertionError):
            unpermute(tokens, idx, idx, [1, 2], drop_and_pad=True)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestInplaceOffload(unittest.TestCase):
    """``inplace_offload``: the CPU branch is a byte-preserving no-op."""

    def test_cpu_tensor_left_unchanged(self):
        # place already equals CPUPlace, so the `if not on-cpu` guard is False
        # and the tensor must be returned untouched -- same place, same bytes.
        x = paddle.to_tensor([1.5, -2.5, 3.25, 4.0], dtype="float32")
        before = x.numpy().copy()
        inplace_offload(x)
        self.assertTrue(x.place._equals(paddle.CPUPlace()))
        np.testing.assert_array_equal(x.numpy(), before)

    # NOTE: the GPU->CPU share_data_with branch requires a device tensor and is
    # not exercised here; the local environment has no GPU.


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestInplaceOffloadIfNeeded(unittest.TestCase):
    """``inplace_offload_if_needed``: no-grad guard + memory_size>=threshold."""

    def test_no_grad_returns_before_threshold_check(self):
        # Under no_grad the function returns immediately, BEFORE the size check.
        # A 3xfloat32 tensor is 12 bytes >= threshold=1, so if the early return
        # were removed the offload branch would fire and warn. Proving no
        # warning is emitted here pins the guard.
        x = paddle.to_tensor([1.0, 2.0, 3.0], dtype="float32")
        before = x.numpy().copy()
        with paddle.no_grad():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                inplace_offload_if_needed(x, threshold=1)
        self.assertEqual([w for w in caught], [])
        np.testing.assert_array_equal(x.numpy(), before)

    def test_grad_and_over_threshold_enters_offload_branch(self):
        # With grad tracking on and threshold below the tensor's byte size, the
        # offload branch runs and emits the documented warning. (On CPU the
        # offload itself is a no-op, so content is preserved.) 3xfloat32 = 12
        # bytes >= 1 by hand.
        x = paddle.to_tensor([1.0, 2.0, 3.0], dtype="float32")
        before = x.numpy().copy()
        with paddle.set_grad_enabled(True):
            with self.assertWarns(UserWarning):
                inplace_offload_if_needed(x, threshold=1)
        np.testing.assert_array_equal(x.numpy(), before)

    def test_grad_and_under_threshold_does_not_offload(self):
        # Same grad context but threshold above the byte size: 12 < 1<<40, so
        # memory_size >= threshold is False and no warning may be emitted.
        x = paddle.to_tensor([1.0, 2.0, 3.0], dtype="float32")
        with paddle.set_grad_enabled(True):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                inplace_offload_if_needed(x, threshold=1 << 40)
        self.assertEqual([w for w in caught], [])


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestTopkToPermutedIndices(unittest.TestCase):
    """Index construction from a routemap: group by expert, token = pos // topk.

    Fixed routemap (3 tokens, topk=2), row-major flattened as
    ``[0, 1, 1, 0, 2, 1]`` -- token0 chose experts {0,1}, token1 chose {1,0},
    token2 chose {2,1}. ``_restrict_nonzero`` returns matching flat positions in
    ascending order; expectations are derived by hand from that.
    """

    def test_single_expert_positions_and_token_map(self):
        routemap = paddle.to_tensor([[0, 1], [1, 0], [2, 1]], dtype="int64")
        # expert 1 occupies flat positions 1, 2, 5.
        token_idx, prob_idx = topk_to_permuted_indices_single(
            routemap, num_tokens=3, expert_id=1, topk=2
        )
        np.testing.assert_array_equal(prob_idx.numpy(), np.array([1, 2, 5]))
        # token = pos // topk : 1//2=0, 2//2=1, 5//2=2.
        np.testing.assert_array_equal(token_idx.numpy(), np.array([0, 1, 2]))

    def test_all_experts_concatenated_in_expert_order(self):
        routemap = paddle.to_tensor([[0, 1], [1, 0], [2, 1]], dtype="int64")
        # expert0 -> [0, 3], expert1 -> [1, 2, 5], expert2 -> [4],
        # concatenated in expert order.
        token_idx, prob_idx = topk_to_permuted_indices(
            routemap, num_tokens_per_expert_list=[2, 3, 1], topk=2
        )
        np.testing.assert_array_equal(
            prob_idx.numpy(), np.array([0, 3, 1, 2, 5, 4])
        )
        # token = pos // 2.
        np.testing.assert_array_equal(
            token_idx.numpy(), np.array([0, 1, 0, 1, 2, 2])
        )


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestUnZipNode(unittest.TestCase):
    """Constructor state and ``reset_status`` for the unzip node."""

    def test_init_stores_dispatcher_and_defaults(self):
        dispatcher = object()  # opaque sentinel: only identity is contracted
        node = UnZipNode(dispatcher, name="unzip_a")
        self.assertIs(node.token_dispatcher, dispatcher)
        self.assertEqual(node.name, "unzip_a")
        self.assertIsNone(node.unzipped_probs)
        self.assertIsNone(node.zipped_expertwise_rowmap)

    def test_default_name(self):
        node = UnZipNode(object())
        self.assertEqual(node.name, "unzip")

    def test_reset_status_clears_cached_fields(self):
        node = UnZipNode(object())
        # Populate with distinct sentinels so reset can only pass by clearing
        # BOTH fields back to None (not by leaving a stale value).
        node.unzipped_probs = "probs-sentinel"
        node.zipped_expertwise_rowmap = "rowmap-sentinel"
        node.reset_status()
        self.assertIsNone(node.unzipped_probs)
        self.assertIsNone(node.zipped_expertwise_rowmap)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestZipNode(unittest.TestCase):
    """Constructor state for the zip node."""

    def test_init_stores_dispatcher_and_name(self):
        dispatcher = object()
        node = ZipNode(dispatcher, name="zip_a")
        self.assertIs(node.token_dispatcher, dispatcher)
        self.assertEqual(node.name, "zip_a")

    def test_default_name(self):
        node = ZipNode(object())
        self.assertEqual(node.name, "zip")


if __name__ == "__main__":
    unittest.main()
