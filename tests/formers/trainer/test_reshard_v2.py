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

"""Behavior tests for ``trainer/utils/reshard/sharding_v2.py`` (v2 sharding).

Scope and oracle
----------------
This file exercises the CPU-observable tensor plumbing that the v2 sharding
reshard is built from, against the *real* production API. Expectations are
hand-derived from small, content-distinguishable inputs (row-major ``arange``
data, not all-zeros), so that a swap of ownership, a wrong offset, a reversed
slice, or a lost/mis-ordered payload would be rejected rather than passing on a
shape check alone:

  * ``is_bata``     -- an independent truth table, including the subtlety that
                       the ``_beta{1,2}_pow_acc_`` markers require the leading
                       underscore.
  * ``pad_tensor``  -- row-major flatten into a zero-filled buffer; the trailing
                       region must be zero and the input's shape must survive.
  * ``slice_tensor``-- the ``[begin:end]`` window contents and length.
  * ``merge_tensors``-- truncate-to-shape reconstruction; verified with an
                       independent numpy expectation and a pad -> merge
                       round-trip.

Distributed boundary (NOT covered here)
---------------------------------------
The chunk/ownership mapping in ``shard.split_func`` (``buffer_slice`` indexing,
``offset`` accumulation, ``has_slice_grad`` guards) and ``restore``,
``collect_split_info``, ``is_matched_optimizer_state_dict`` all require a real
``DygraphShardingOptimizerV2`` comm-buffer layout and a live sharding process
group. Cross-rank reshard tensor numerics and the per-rank offset/ownership
mapping cannot be proven in a single CPU process; faking ``group.nranks`` plus
metadata would only prove call plumbing, not the redistribution. Those paths are
explicitly skipped with that reason rather than simulated (see
``TestDistributedReshardBoundary``).
"""

import unittest

import numpy as np

# NOTE: do NOT plant empty-string ``FLAGS_selected_gpus``/``CUDA_VISIBLE_DEVICES``
# here. This module is imported at collection time into a persistent xdist
# worker that also runs GPU tests. ``paddle.distributed.ParallelEnv()`` reads
# ``int(os.getenv("FLAGS_selected_gpus", "0")[0])`` on every construction, so an
# empty string set process-wide makes ``int("")`` raise for *sibling* tests
# (e.g. any ``tensor.to("cuda")`` path) even though this file only needs CPU.
# ``paddle.set_device("cpu")`` below is sufficient to keep this module's own
# tensor-plumbing tests on CPU without mutating the shared environment.

try:
    import paddle

    paddle.set_device("cpu")

    from paddlefleet.trainer.utils.reshard.sharding_v2 import (
        is_bata,
        merge_tensors,
        pad_tensor,
        slice_tensor,
    )

    PADDLE_AVAILABLE = True
    IMPORT_ERROR = ""
except (
    ImportError
) as exc:  # only real missing-dependency, not API/compile errors
    PADDLE_AVAILABLE = False
    IMPORT_ERROR = repr(exc)

_SKIP_REASON = f"paddle / paddlefleet not importable: {IMPORT_ERROR}"


@unittest.skipUnless(PADDLE_AVAILABLE, _SKIP_REASON)
class TestIsBata(unittest.TestCase):
    """``is_bata`` recognises beta-power accumulators by an exact marker."""

    def test_truth_table(self):
        """Hand-derived: only ``_beta{1,2}_pow_acc_`` substrings are beta."""
        cases = {
            "fp32_master_0_beta1_pow_acc_0": True,
            "fp32_master_0_beta2_pow_acc_0": True,
            "p_beta1_pow_acc_0": True,
            "p_beta2_pow_acc_0": True,
            "p_moment1_0": False,
            "p_moment2_0": False,
            "linear_weight": False,
            "": False,
        }
        for name, expected in cases.items():
            self.assertEqual(is_bata(name), expected, name)

    def test_marker_requires_leading_underscore(self):
        """The literal is ``_beta1_pow_acc_``; a name lacking the leading
        underscore is deliberately not matched."""
        self.assertFalse(is_bata("beta1_pow_acc_0"))
        self.assertFalse(is_bata("beta2_pow_acc_0"))
        # But an embedded occurrence (with the underscore) does match.
        self.assertTrue(is_bata("x_beta1_pow_acc_3"))

    def test_beta1_and_beta2_independent(self):
        """Neither branch shadows the other."""
        self.assertTrue(is_bata("w_beta1_pow_acc_0"))
        self.assertTrue(is_bata("w_beta2_pow_acc_0"))


@unittest.skipUnless(PADDLE_AVAILABLE, _SKIP_REASON)
class TestPadTensor(unittest.TestCase):
    """``pad_tensor`` flattens row-major into a zero-filled length-N buffer."""

    def test_row_major_flatten_then_zeros(self):
        """Data region is the row-major flatten; the tail is zero."""
        src = paddle.to_tensor(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype="float32"
        )
        out = pad_tensor("k", src, 8)
        self.assertEqual(list(out.shape), [8])
        np.testing.assert_array_equal(
            out.numpy(), np.array([0, 1, 2, 3, 4, 5, 0, 0], dtype=np.float32)
        )

    def test_input_shape_is_restored(self):
        """The in-place ``flatten_`` is undone; caller's tensor keeps its
        original 2-D shape and values."""
        src = paddle.to_tensor([[7.0, 8.0], [9.0, 10.0]], dtype="float32")
        pad_tensor("k", src, 6)
        self.assertEqual(list(src.shape), [2, 2])
        np.testing.assert_array_equal(
            src.numpy(), np.array([[7, 8], [9, 10]], dtype=np.float32)
        )

    def test_dtype_preserved(self):
        """Padding buffer follows the input dtype."""
        src = paddle.to_tensor([[1, 2], [3, 4]], dtype="int64")
        out = pad_tensor("k", src, 5)
        self.assertEqual(out.dtype, src.dtype)
        np.testing.assert_array_equal(
            out.numpy(), np.array([1, 2, 3, 4, 0], dtype=np.int64)
        )

    def test_no_padding_needed_is_exact_flatten(self):
        """When padded_size == numel there is no trailing zero region."""
        src = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        out = pad_tensor("k", src, 4)
        np.testing.assert_array_equal(
            out.numpy(), np.array([1, 2, 3, 4], dtype=np.float32)
        )

    def test_same_shape_distinct_content_distinct_output(self):
        """Two params sharing shape but differing in content pad to their own
        distinct data regions -- content, not just shape, is carried."""
        a = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        b = paddle.to_tensor([[5.0, 6.0], [7.0, 8.0]], dtype="float32")
        oa = pad_tensor("a", a, 6).numpy()
        ob = pad_tensor("b", b, 6).numpy()
        np.testing.assert_array_equal(oa[:4], [1, 2, 3, 4])
        np.testing.assert_array_equal(ob[:4], [5, 6, 7, 8])
        self.assertFalse(np.array_equal(oa, ob))

    def test_oversized_input_rejected(self):
        """padded_size smaller than numel violates the documented contract."""
        src = paddle.to_tensor([1.0, 2.0, 3.0, 4.0], dtype="float32")
        with self.assertRaises(AssertionError):
            pad_tensor("k", src, 3)


@unittest.skipUnless(PADDLE_AVAILABLE, _SKIP_REASON)
class TestSliceTensor(unittest.TestCase):
    """``slice_tensor`` returns the half-open ``[begin, end)`` window."""

    def test_interior_window_content(self):
        src = paddle.arange(10, dtype="float32")
        out = slice_tensor(src, 2, 7)
        np.testing.assert_array_equal(out.numpy(), [2, 3, 4, 5, 6])

    def test_prefix_and_suffix_windows(self):
        src = paddle.arange(6, dtype="float32")
        np.testing.assert_array_equal(
            slice_tensor(src, 0, 3).numpy(), [0, 1, 2]
        )
        np.testing.assert_array_equal(
            slice_tensor(src, 3, 6).numpy(), [3, 4, 5]
        )

    def test_empty_window(self):
        src = paddle.arange(5, dtype="float32")
        out = slice_tensor(src, 3, 3)
        self.assertEqual(int(out.shape[0]), 0)

    def test_windows_are_disjoint_and_ordered(self):
        """Adjacent windows partition the buffer without overlap or reorder --
        the property the reshard chunker relies on for offset accounting."""
        src = paddle.arange(8, dtype="float32")
        left = slice_tensor(src, 0, 3).numpy()
        right = slice_tensor(src, 3, 8).numpy()
        np.testing.assert_array_equal(
            np.concatenate([left, right]), np.arange(8, dtype=np.float32)
        )


@unittest.skipUnless(PADDLE_AVAILABLE, _SKIP_REASON)
class TestMergeTensorsSingle(unittest.TestCase):
    """``merge_tensors`` with one shard: truncate to ``prod(shape)`` then reshape."""

    def test_reshape_and_truncate_content(self):
        """Independent numpy oracle: the first ``prod(shape)`` values, row-major
        into ``shape``; the padding tail is dropped."""
        padded = paddle.to_tensor(
            [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 0.0, 0.0], dtype="float32"
        )
        out = merge_tensors("k", [padded], [2, 3])
        self.assertEqual(list(out.shape), [2, 3])
        np.testing.assert_array_equal(
            out.numpy(),
            np.array([[10, 11, 12], [13, 14, 15]], dtype=np.float32),
        )

    def test_exact_fit_no_truncation(self):
        src = paddle.to_tensor([1.0, 2.0, 3.0, 4.0], dtype="float32")
        out = merge_tensors("k", [src], [2, 2])
        np.testing.assert_array_equal(
            out.numpy(), np.array([[1, 2], [3, 4]], dtype=np.float32)
        )

    def test_pad_then_merge_round_trips(self):
        """pad_tensor -> merge_tensors recovers the original param exactly
        (cross-check of the two halves of the split/restore pipeline)."""
        original = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32"
        )
        padded = pad_tensor("k", original, 8)
        restored = merge_tensors("k", [padded], [2, 3])
        np.testing.assert_array_equal(restored.numpy(), original.numpy())

    def test_undersized_shard_rejected(self):
        """A shard smaller than the target shape violates the contract."""
        src = paddle.to_tensor([1.0, 2.0, 3.0], dtype="float32")
        with self.assertRaises(AssertionError):
            merge_tensors("k", [src], [2, 2])


@unittest.skipUnless(PADDLE_AVAILABLE, _SKIP_REASON)
class TestMergeTensorsMultiple(unittest.TestCase):
    """``merge_tensors`` with several 1-D shards must concatenate them in order
    before truncating and reshaping."""

    def test_multi_shard_concatenates_in_order(self):
        """Multiple 1-D shards are joined head to tail, then truncated and
        reshaped to the target shape.

        s0 = [1, 2, 3] and s1 = [4, 5, 6] concatenate to [1, 2, 3, 4, 5, 6];
        reshaping to [2, 3] yields [[1, 2, 3], [4, 5, 6]]. The expected value is
        hand-derived from the concatenation contract, not from the function
        under test."""
        s0 = paddle.to_tensor([1.0, 2.0, 3.0], dtype="float32")
        s1 = paddle.to_tensor([4.0, 5.0, 6.0], dtype="float32")
        out = merge_tensors("k", [s0, s1], [2, 3])
        np.testing.assert_array_equal(
            out.numpy(),
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        )


class TestDistributedReshardBoundary(unittest.TestCase):
    """The reshard mapping/numerics are not CPU-observable in one process."""

    def test_reshard_requires_real_process_group(self):
        """``shard.split_func`` (buffer_slice ownership indices, offset
        accumulation, has_slice_grad guards), ``restore``,
        ``collect_split_info`` and ``is_matched_optimizer_state_dict`` depend on
        a real ``DygraphShardingOptimizerV2`` comm-buffer layout and a live
        sharding process group. Per-rank ownership/offset mapping and cross-rank
        reshard tensor numerics cannot be validated single-process; faking
        ``group.nranks`` and metadata would only prove call plumbing, not the
        redistribution. This requires the multi-card harness."""
        self.skipTest(
            "v2 sharding reshard mapping/offsets and cross-rank tensor "
            "numerics need a real sharding process group (multi-card); "
            "not verifiable on CPU single-process."
        )


if __name__ == "__main__":
    unittest.main()
